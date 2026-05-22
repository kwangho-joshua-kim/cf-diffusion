# =============================================================================
#  Geometry-Adaptive Counterfactual Distribution Learning
#  with Diffusion-Guided Smoothing
# -----------------------------------------------------------------------------
#  CelebA semi-synthetic experiment for DIS and DSS diagnostics
# =============================================================================
#
#  Overview
#  --------
#  Reproduces the CelebA semi-synthetic simulation study from the paper.
#  The script evaluates two estimators introduced there:
#
#    * DIS  -- Diffusion-Informed Smoothing for the counterfactual density
#              p_{a,h}^{geo}, an AIPW one-step on a locally-adaptive
#              anisotropic kernel whose geometry is learned in the ambient
#              representation space.
#
#    * DSS  -- Diffusion-informed Score Smoothing for the counterfactual
#              score s_{a,h}^{geo} = grad log p_{a,h}^{geo}, formed as the
#              ratio G_hat / P_hat of two AIPW one-step estimators that
#              share cross-fitted nuisances.
#
#  The experiment varies the ambient dimension r in {500, 2500, 10000} and
#  the working-sample size n on a logarithmic grid, and reports ISE on the
#  density side and Stein-functional MSE on the score side, averaged over
#  Monte Carlo replicates.
#
#  Inputs
#  ------
#    cfg$data_dir   path to a folder containing
#                     (i)  list_attr_celeba.{txt,csv}
#                     (ii) a pretrained embedding file, e.g.
#                          celeba_embeddings.{csv,rds}, with first column
#                          image_id.
#                   See extract_celeba_embeddings.py for one way to produce
#                   the embedding file.
#
#  Outputs (written to cfg$out_dir, default "FIG/")
#  ------------------------------------------------
#    fig_exp_density.png                density figure on n_values_main
#    fig_exp_stein.png                  Stein figure on n_values_main
#    fig_exp_density_extended_n8_v2.png density figure on n_values_ext
#    fig_exp_stein_extended_n8_v2.png   Stein figure on n_values_ext
#    table_slopes_density.csv           log-log slopes by (method, r)
#    table_slopes_stein.csv             log-log slopes by (method, r)
#    density_raw.csv, stein_raw.csv,
#    density_summary.csv, stein_summary.csv
#
#  Dependencies
#  ------------
#    R >= 4.1 with packages: data.table, ggplot2, RANN
#
# =============================================================================

# ----------------------------
# 0. User settings
# ----------------------------

cfg <- list(
  data_dir = "PATH_TO_CELEBA_FOLDER",
  out_dir = "FIG",

  attr_file = NULL,
  embedding_file = NULL,

  use_attribute_proxy_if_no_embedding = FALSE,

  treatment_name = "Smiling",
  target_a = 1L,

  ambient_dims = c(500L, 2500L, 10000L),
  eval_dim = 2L,
  # NULL = no cap, so the stress test in r actually varies.
  # Set to an integer if kNN over r-dim points becomes too slow.
  geom_dim_cap = NULL,

  n_values_main = c(500L, 1000L, 2000L, 5000L, 10000L, 25000L),
  n_values_ext  = c(500L, 750L, 1000L, 2000L, 5000L, 10000L, 25000L, 50000L),

  n_reps_ext  = 20L,

  ref_size = 50000L,
  grid_size_1d = 35L,

  n_folds = 3L,
  ridge_lambda = 1e-2,

  local_k = 40L,
  local_ridge = 1e-3,

  h_constant = 1.0,
  h_power = 1 / 5,

  chunk_grid = 300L,
  chunk_center = 1000L,

  stein_mc_size = 4000L,
  n_random_g = 4L,

  p_min_trunc = 1e-6,    # denominator floor for DSS ratio

  seed = 20251101L
)

set.seed(cfg$seed)
dir.create(cfg$out_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------
# 1. Packages
# ----------------------------

pkgs <- c("data.table", "ggplot2", "RANN")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("Missing package: ", p, ". Please install it first.")
  }
}
library(data.table); library(ggplot2); library(RANN)

# ----------------------------
# 2. Utilities
# ----------------------------

standardize_matrix <- function(x) {
  x <- as.matrix(x); x <- scale(x); x[is.na(x)] <- 0; x
}

find_first_existing <- function(paths) {
  paths <- paths[file.exists(paths)]
  if (length(paths) == 0L) return(NULL)
  paths[[1L]]
}

read_celeba_attributes <- function(data_dir, attr_file = NULL) {
  if (is.null(attr_file)) {
    attr_file <- find_first_existing(c(
      file.path(data_dir, "list_attr_celeba.csv"),
      file.path(data_dir, "list_attr_celeba.txt"),
      file.path(data_dir, "Anno", "list_attr_celeba.txt")
    ))
  }
  if (is.null(attr_file)) stop("Could not find list_attr_celeba file.")
  if (grepl("\\.csv$", attr_file, ignore.case = TRUE)) {
    dat <- data.table::fread(attr_file)
    if (!("image_id" %in% names(dat))) names(dat)[1L] <- "image_id"
    return(as.data.frame(dat))
  }
  lines2 <- readLines(attr_file, n = 2L)
  att_names <- strsplit(trimws(lines2[2L]), "\\s+")[[1L]]
  dat <- data.table::fread(attr_file, skip = 2L, header = FALSE)
  names(dat) <- c("image_id", att_names)
  as.data.frame(dat)
}

read_embedding_file <- function(data_dir, embedding_file = NULL) {
  if (is.null(embedding_file)) {
    embedding_file <- find_first_existing(c(
      file.path(data_dir, "celeba_embeddings.rds"),
      file.path(data_dir, "celeba_embeddings.csv"),
      file.path(data_dir, "embeddings.rds"),
      file.path(data_dir, "embeddings.csv")
    ))
  }
  if (is.null(embedding_file)) return(NULL)
  if (grepl("\\.rds$", embedding_file, ignore.case = TRUE)) {
    obj <- readRDS(embedding_file)
    if (is.matrix(obj)) obj <- as.data.frame(obj)
    obj <- as.data.frame(obj)
    if (!("image_id" %in% names(obj)))
      stop("Embedding RDS must contain image_id.")
    return(obj)
  }
  obj <- data.table::fread(embedding_file); obj <- as.data.frame(obj)
  if (!("image_id" %in% names(obj))) names(obj)[1L] <- "image_id"
  obj
}

make_folds <- function(n, k) sample(rep(seq_len(k), length.out = n))

safe_logit_fit <- function(x_train, a_train, x_new) {
  dat <- data.frame(a = as.integer(a_train), x_train)
  fit <- suppressWarnings(glm(a ~ ., data = dat, family = binomial()))
  pred <- suppressWarnings(predict(fit, newdata = data.frame(x_new),
                                   type = "response"))
  pmin(pmax(as.numeric(pred), 0.02), 0.98)
}

crossfit_propensity <- function(x, a, folds) {
  n <- length(a); out <- rep(NA_real_, n)
  for (fold in sort(unique(folds))) {
    idx_te <- which(folds == fold); idx_tr <- which(folds != fold)
    out[idx_te] <- safe_logit_fit(x[idx_tr, , drop = FALSE], a[idx_tr],
                                  x[idx_te, , drop = FALSE])
  }
  pmin(pmax(out, 0.02), 0.98)
}

ridge_multi_predict <- function(x_train, y_train, x_new, lambda) {
  x_train <- as.matrix(x_train); x_new <- as.matrix(x_new)
  y_train <- as.matrix(y_train)
  xb_train <- cbind(Intercept = 1, x_train)
  xb_new   <- cbind(Intercept = 1, x_new)
  xtx <- crossprod(xb_train)
  pen <- diag(ncol(xtx)); pen[1L, 1L] <- 0
  beta <- solve(xtx + lambda * pen, crossprod(xb_train, y_train))
  xb_new %*% beta
}

# geom_dim_cap may be NULL => no cap on the geometry sample dimension.
make_representations <- function(y_base, ambient_r, eval_dim, geom_dim_cap = NULL) {
  y_base <- standardize_matrix(y_base); q <- ncol(y_base)
  w_eval <- matrix(rnorm(q * eval_dim), q, eval_dim) / sqrt(q)
  y_eval <- standardize_matrix(y_base %*% w_eval)
  geom_dim <- as.integer(ambient_r)
  if (!is.null(geom_dim_cap)) geom_dim <- min(geom_dim, as.integer(geom_dim_cap))
  w_geom <- matrix(rnorm(q * geom_dim), q, geom_dim) / sqrt(q)
  y_geom <- standardize_matrix(y_base %*% w_geom)
  list(y_eval = y_eval, y_geom = y_geom)
}

make_grid <- function(y_eval, grid_size_1d = 35L, pad = 0.05) {
  d <- ncol(y_eval)
  if (!(d %in% c(2L, 3L)))
    stop("Grid construction currently supports eval_dim 2 or 3.")
  limits <- lapply(seq_len(d), function(j) {
    qs <- as.numeric(quantile(y_eval[, j], probs = c(0.03, 0.97), na.rm = TRUE))
    w <- qs[2L] - qs[1L]; c(qs[1L] - pad * w, qs[2L] + pad * w)
  })
  axes <- lapply(limits, function(z) seq(z[1L], z[2L], length.out = grid_size_1d))
  grid <- do.call(expand.grid, axes)
  names(grid) <- paste0("y", seq_len(d)); grid <- as.matrix(grid)
  cell_volume <- prod(vapply(axes, function(ax) mean(diff(ax)), numeric(1L)))
  list(grid = grid, cell_volume = cell_volume)
}

# When query and bank are the same set, exclude_self = TRUE drops each
# point's own index (column 1 of the kNN result).
local_cov_from_bank <- function(query_geom, bank_geom, bank_eval,
                                k = 40L, ridge = 1e-3,
                                exclude_self = FALSE) {
  query_geom <- as.matrix(query_geom)
  bank_geom  <- as.matrix(bank_geom)
  bank_eval  <- as.matrix(bank_eval)
  d <- ncol(bank_eval)
  k_use <- min(k + as.integer(exclude_self), nrow(bank_geom))
  nn <- RANN::nn2(data = bank_geom, query = query_geom, k = k_use)
  idx_mat <- nn$nn.idx
  if (exclude_self) idx_mat <- idx_mat[, -1L, drop = FALSE]
  covs <- vector("list", nrow(query_geom))
  for (i in seq_len(nrow(query_geom))) {
    yy <- bank_eval[idx_mat[i, ], , drop = FALSE]
    s <- stats::cov(yy)
    if (d == 1L) s <- matrix(as.numeric(s), 1, 1)
    if (any(!is.finite(s))) s <- diag(d)
    scale_ridge <- mean(diag(s))
    if (!is.finite(scale_ridge) || scale_ridge <= 0) scale_ridge <- 1
    covs[[i]] <- s + ridge * scale_ridge * diag(d)
  }
  covs
}

dmv_iso_matrix <- function(centers, grid, h) {
  centers <- as.matrix(centers); grid <- as.matrix(grid); d <- ncol(grid)
  center_norm <- rowSums(centers^2); grid_norm <- rowSums(grid^2)
  dist2 <- outer(center_norm, grid_norm, "+") - 2 * centers %*% t(grid)
  (2 * pi * h)^(-d / 2) * exp(-0.5 * dist2 / h)
}

# Cache (inv_s, log_det) once per center so the kernel pass is O(d^3) per
# center, not per (center, grid-chunk).
precompute_aniso <- function(covs, h) {
  lapply(covs, function(s0) {
    s <- h * s0
    list(inv_s = solve(s),
         log_det = determinant(s, logarithm = TRUE)$modulus[1L])
  })
}

dmv_aniso_matrix <- function(centers, grid, aniso_pre, h) {
  centers <- as.matrix(centers); grid <- as.matrix(grid)
  n <- nrow(centers); m <- nrow(grid); d <- ncol(grid)
  out <- matrix(0, n, m)
  for (i in seq_len(n)) {
    diff <- sweep(grid, 2L, centers[i, ], "-")
    qf <- rowSums((diff %*% aniso_pre[[i]]$inv_s) * diff)
    out[i, ] <- exp(-0.5 * (d * log(2 * pi) + aniso_pre[[i]]$log_det + qf))
  }
  out
}

kernel_matrix <- function(centers, grid, h, type = c("isotropic", "anisotropic"),
                          aniso_pre = NULL) {
  type <- match.arg(type)
  if (type == "isotropic") return(dmv_iso_matrix(centers, grid, h))
  if (is.null(aniso_pre)) stop("aniso_pre required for anisotropic.")
  dmv_aniso_matrix(centers, grid, aniso_pre, h)
}

# Gradient of kernel w.r.t. y, returned as array(n, m, d). Uses k_val.
kernel_gradient_array <- function(centers, points, h, type, aniso_pre, k_val) {
  centers <- as.matrix(centers); points <- as.matrix(points)
  n <- nrow(centers); m <- nrow(points); d <- ncol(points)
  out <- array(0, dim = c(n, m, d))
  if (type == "isotropic") {
    for (i in seq_len(n)) {
      diff <- sweep(points, 2L, centers[i, ], "-")     # m x d
      g_fac <- -diff / h                                # m x d
      out[i, , ] <- g_fac * matrix(k_val[i, ], nrow = m, ncol = d)
    }
  } else {
    for (i in seq_len(n)) {
      diff <- sweep(points, 2L, centers[i, ], "-")
      g_fac <- -diff %*% aniso_pre[[i]]$inv_s
      out[i, , ] <- g_fac * matrix(k_val[i, ], nrow = m, ncol = d)
    }
  }
  out
}

weighted_density_grid <- function(centers, grid, weights, h, type,
                                  aniso_pre = NULL, chunk_center = 1000L,
                                  normalize = TRUE) {
  centers <- as.matrix(centers); grid <- as.matrix(grid)
  weights <- as.numeric(weights)
  # normalize = TRUE (Hajek): divide by sum of weights.
  # normalize = FALSE (Horvitz-Thompson): pass weights pre-scaled by 1/n so
  # the result matches the AIPW one-step normalization.
  if (normalize) weights <- weights / sum(weights)
  m <- nrow(grid); out <- rep(0, m)
  starts <- seq(1L, nrow(centers), by = chunk_center)
  for (s in starts) {
    e <- min(s + chunk_center - 1L, nrow(centers)); idx <- s:e
    k_mat <- kernel_matrix(centers[idx, , drop = FALSE], grid, h, type,
                           if (is.null(aniso_pre)) NULL else aniso_pre[idx])
    out <- out + as.numeric(crossprod(weights[idx], k_mat))
  }
  out
}

# AIPW one-step density on a generic set of points (grid or MC).
# Returns both the AIPW one-step and the regression plug-in (P_n mu_hat).
one_step_density_at_points <- function(y_eval, y_geom, x, a, points, h, method,
                                       target_a, folds, pi_hat, lambda,
                                       local_k, local_ridge, chunk_grid = 300L) {
  n <- nrow(y_eval); m <- nrow(points)
  est <- rep(0, m); plug <- rep(0, m)
  point_starts <- seq(1L, m, by = chunk_grid)

  for (fold in sort(unique(folds))) {
    idx_te   <- which(folds == fold)
    idx_tr   <- which(folds != fold)
    idx_tr_a <- idx_tr[a[idx_tr] == target_a]
    if (length(idx_tr_a) < 20L)
      stop("Too few treated target arm observations in a training fold.")

    aniso_te_pre  <- NULL
    aniso_tra_pre <- NULL
    if (method == "anisotropic") {
      cov_te <- local_cov_from_bank(y_geom[idx_te, , drop = FALSE],
                                    y_geom[idx_tr_a, , drop = FALSE],
                                    y_eval[idx_tr_a, , drop = FALSE],
                                    local_k, local_ridge,
                                    exclude_self = FALSE)
      cov_tr_a <- local_cov_from_bank(y_geom[idx_tr_a, , drop = FALSE],
                                      y_geom[idx_tr_a, , drop = FALSE],
                                      y_eval[idx_tr_a, , drop = FALSE],
                                      local_k, local_ridge,
                                      exclude_self = TRUE)
      aniso_te_pre  <- precompute_aniso(cov_te, h)
      aniso_tra_pre <- precompute_aniso(cov_tr_a, h)
    }

    for (gs in point_starts) {
      ge <- min(gs + chunk_grid - 1L, m); gidx <- gs:ge
      pchunk <- points[gidx, , drop = FALSE]

      k_train <- kernel_matrix(y_eval[idx_tr_a, , drop = FALSE], pchunk, h,
                               method, aniso_tra_pre)

      mu_te <- ridge_multi_predict(x[idx_tr_a, , drop = FALSE], k_train,
                                   x[idx_te, , drop = FALSE], lambda)
      k_te  <- kernel_matrix(y_eval[idx_te, , drop = FALSE], pchunk, h,
                             method, aniso_te_pre)

      a_te <- as.numeric(a[idx_te] == target_a)
      ipw  <- a_te / pi_hat[idx_te]

      phi <- sweep(k_te, 1L, ipw, "*") - sweep(mu_te, 1L, ipw, "*") + mu_te
      est[gidx]  <- est[gidx]  + colSums(phi)
      plug[gidx] <- plug[gidx] + colSums(mu_te)
    }
  }
  list(one_step = est / n, plug_in = plug / n)
}

# AIPW one-step for P and G simultaneously at the same evaluation points.
# Returns P (length m) and G (m x d). The DSS score is G / P.
one_step_pq_at_points <- function(y_eval, y_geom, x, a, points, h, method,
                                  target_a, folds, pi_hat, lambda,
                                  local_k, local_ridge, chunk_grid = 300L) {
  n <- nrow(y_eval); m <- nrow(points); d <- ncol(points)
  P_sum <- rep(0, m); G_sum <- matrix(0, m, d)
  point_starts <- seq(1L, m, by = chunk_grid)

  for (fold in sort(unique(folds))) {
    idx_te   <- which(folds == fold)
    idx_tr   <- which(folds != fold)
    idx_tr_a <- idx_tr[a[idx_tr] == target_a]
    if (length(idx_tr_a) < 20L)
      stop("Too few treated target arm observations in a training fold.")

    aniso_te_pre  <- NULL
    aniso_tra_pre <- NULL
    if (method == "anisotropic") {
      cov_te <- local_cov_from_bank(y_geom[idx_te, , drop = FALSE],
                                    y_geom[idx_tr_a, , drop = FALSE],
                                    y_eval[idx_tr_a, , drop = FALSE],
                                    local_k, local_ridge,
                                    exclude_self = FALSE)
      cov_tr_a <- local_cov_from_bank(y_geom[idx_tr_a, , drop = FALSE],
                                      y_geom[idx_tr_a, , drop = FALSE],
                                      y_eval[idx_tr_a, , drop = FALSE],
                                      local_k, local_ridge,
                                      exclude_self = TRUE)
      aniso_te_pre  <- precompute_aniso(cov_te, h)
      aniso_tra_pre <- precompute_aniso(cov_tr_a, h)
    }

    for (ps in point_starts) {
      pe <- min(ps + chunk_grid - 1L, m); pidx <- ps:pe
      pchunk <- points[pidx, , drop = FALSE]
      mch <- nrow(pchunk)

      k_train  <- kernel_matrix(y_eval[idx_tr_a, , drop = FALSE], pchunk, h,
                                method, aniso_tra_pre)
      dk_train <- kernel_gradient_array(y_eval[idx_tr_a, , drop = FALSE],
                                        pchunk, h, method, aniso_tra_pre, k_train)

      # Stack regression targets: (k_train, dk_train[,,1], ..., dk_train[,,d])
      target_block <- k_train
      for (jj in seq_len(d)) target_block <- cbind(target_block, dk_train[, , jj])
      # target_block: n_tr_a x (mch * (1 + d))

      pred_te <- ridge_multi_predict(x[idx_tr_a, , drop = FALSE], target_block,
                                     x[idx_te, , drop = FALSE], lambda)
      mu_te <- pred_te[, seq_len(mch), drop = FALSE]
      nu_te <- array(0, dim = c(length(idx_te), mch, d))
      for (jj in seq_len(d)) {
        cs <- mch * jj + 1L; ce <- mch * (jj + 1L)
        nu_te[, , jj] <- pred_te[, cs:ce, drop = FALSE]
      }

      k_te  <- kernel_matrix(y_eval[idx_te, , drop = FALSE], pchunk, h,
                             method, aniso_te_pre)
      dk_te <- kernel_gradient_array(y_eval[idx_te, , drop = FALSE], pchunk, h,
                                     method, aniso_te_pre, k_te)

      a_te <- as.numeric(a[idx_te] == target_a)
      ipw  <- a_te / pi_hat[idx_te]

      phi_P <- sweep(k_te, 1L, ipw, "*") - sweep(mu_te, 1L, ipw, "*") + mu_te
      P_sum[pidx] <- P_sum[pidx] + colSums(phi_P)

      for (jj in seq_len(d)) {
        # Coerce to matrix to survive singleton dims when length(idx_te) == 1
        # or mch == 1.
        dk_te_j <- matrix(dk_te[, , jj], nrow = length(idx_te), ncol = mch)
        nu_te_j <- matrix(nu_te[, , jj], nrow = length(idx_te), ncol = mch)
        phi_G_j <- sweep(dk_te_j, 1L, ipw, "*") -
                   sweep(nu_te_j, 1L, ipw, "*") + nu_te_j
        G_sum[pidx, jj] <- G_sum[pidx, jj] + colSums(phi_G_j)
      }
    }
  }
  list(P = P_sum / n, G = G_sum / n)
}

# Reference / treated-only mixture score (kept for the plug-in baseline).
score_mixture <- function(points, centers, weights, h, type, covs = NULL,
                          chunk_center = 1000L) {
  points <- as.matrix(points); centers <- as.matrix(centers)
  weights <- as.numeric(weights); weights <- weights / sum(weights)
  m <- nrow(points); d <- ncol(points)
  dens <- rep(0, m); grad <- matrix(0, m, d)
  aniso_pre <- if (type == "anisotropic") precompute_aniso(covs, h) else NULL
  starts <- seq(1L, nrow(centers), by = chunk_center)
  for (s in starts) {
    e <- min(s + chunk_center - 1L, nrow(centers)); idx <- s:e
    k_mat <- kernel_matrix(centers[idx, , drop = FALSE], points, h, type,
                           if (is.null(aniso_pre)) NULL else aniso_pre[idx])
    dens <- dens + as.numeric(crossprod(weights[idx], k_mat))
    for (j in seq_along(idx)) {
      diff <- sweep(points, 2L, centers[idx[j], ], "-")
      gj <- if (type == "isotropic") -diff / h
            else                     -diff %*% aniso_pre[[idx[j]]]$inv_s
      grad <- grad + (weights[idx[j]] * k_mat[j, ]) * gj
    }
  }
  grad / pmax(dens, 1e-12)
}

# Closure-safe Stein test-function constructors. force()ing the parameter
# at construction time prevents lazy evaluation from capturing the loop
# variable by reference in make_g_functions() below.
make_basis_g <- function(jj) {
  force(jj)
  list(
    g = function(y) {
      y <- as.matrix(y); val <- exp(-rowSums(y^2) / 2)
      mat <- matrix(0, nrow(y), ncol(y)); mat[, jj] <- val; mat
    },
    div = function(y) {
      y <- as.matrix(y); -y[, jj] * exp(-rowSums(y^2) / 2)
    }
  )
}
make_random_g <- function(v) {
  force(v)
  list(
    g = function(y) {
      y <- as.matrix(y); val <- exp(-rowSums(y^2) / 2)
      sweep(matrix(rep(val, ncol(y)), ncol = ncol(y)), 2L, v, "*")
    },
    div = function(y) {
      y <- as.matrix(y); -as.numeric(y %*% v) * exp(-rowSums(y^2) / 2)
    }
  )
}
make_g_functions <- function(eval_dim, n_random = 4L) {
  out <- list()
  for (j in seq_len(eval_dim)) out[[paste0("basis_", j)]] <- make_basis_g(j)
  for (b in seq_len(n_random)) {
    v <- rnorm(eval_dim); v <- v / sqrt(sum(v^2))
    out[[paste0("random_", b)]] <- make_random_g(v)
  }
  out
}

# Stein-functional values for a list of test functions g, optionally
# weighted (weights = NULL gives an unweighted mean).
stein_values <- function(points, score, g_list, weights = NULL) {
  if (is.null(weights)) {
    return(vapply(g_list, function(obj) {
      mean(obj$div(points) + rowSums(obj$g(points) * score))
    }, numeric(1L)))
  }
  w <- weights / sum(weights)
  vapply(g_list, function(obj) {
    sum(w * (obj$div(points) + rowSums(obj$g(points) * score)))
  }, numeric(1L))
}

estimate_slopes <- function(df, error_col) {
  split_df <- split(df, list(df$method, df$r), drop = TRUE)
  do.call(rbind, lapply(split_df, function(z) {
    fit <- lm(log(z[[error_col]]) ~ log(z$n), data = z)
    data.frame(method = z$method[1L], r = z$r[1L],
               slope = unname(coef(fit)[2L]))
  }))
}

plot_density_results <- function(df, file) {
  ggplot(df, aes(x = n, y = ise, color = method, shape = factor(r),
                 group = interaction(method, r))) +
    geom_line(linewidth = 0.8) + geom_point(size = 2.3) +
    scale_x_log10() + scale_y_log10() +
    scale_color_manual(name = "Method",
      values = c("Isotropic one step" = "#1f77b4",
                 "Anisotropic plug in" = "#d62728",
                 "DIS local geometry"  = "#e377c2"),
      breaks = c("Isotropic one step", "Anisotropic plug in",
                 "DIS local geometry")) +
    scale_shape_manual(name = "Ambient dimension r",
      values = c("500" = 16, "2500" = 15, "10000" = 17),
      labels = c("r = 500", "r = 2500", "r = 10000")) +
    labs(x = "n", y = "ISE") +
    theme_bw(base_size = 13) + theme(legend.position = "right")
  ggsave(file, width = 8.5, height = 4.8, dpi = 320)
}

plot_stein_results <- function(df, file) {
  ggplot(df, aes(x = n, y = mse, color = method, shape = factor(r),
                 group = interaction(method, r))) +
    geom_line(linewidth = 0.8) + geom_point(size = 2.3) +
    scale_x_log10() + scale_y_log10() +
    scale_color_manual(name = "Method",
      values = c("Treated only plug in" = "#1f77b4",
                 "DSS isotropic"        = "#d62728",
                 "DSS anisotropic"      = "#e377c2"),
      breaks = c("Treated only plug in", "DSS isotropic",
                 "DSS anisotropic")) +
    scale_shape_manual(name = "Ambient dimension r",
      values = c("500" = 16, "2500" = 15, "10000" = 17),
      labels = c("r = 500", "r = 2500", "r = 10000")) +
    labs(x = "n", y = "MSE") +
    theme_bw(base_size = 13) + theme(legend.position = "right")
  ggsave(file, width = 8.5, height = 4.8, dpi = 320)
}

# ----------------------------
# 3. Load and prepare data
# ----------------------------

attrs <- read_celeba_attributes(cfg$data_dir, cfg$attr_file)
if (!(cfg$treatment_name %in% names(attrs)))
  stop("Treatment variable ", cfg$treatment_name, " not found.")

attrs[[cfg$treatment_name]] <- ifelse(attrs[[cfg$treatment_name]] > 0, 1L, 0L)
a_all   <- attrs[[cfg$treatment_name]]
x_names <- setdiff(names(attrs), c("image_id", cfg$treatment_name))

emb <- read_embedding_file(cfg$data_dir, cfg$embedding_file)
if (is.null(emb)) {
  if (!cfg$use_attribute_proxy_if_no_embedding)
    stop("No embedding file found; see cfg$use_attribute_proxy_if_no_embedding.")
  message("Using CelebA attributes as a proxy embedding (pipeline test only).")
  x_all <- as.matrix(attrs[, x_names, drop = FALSE])
  x_all <- ifelse(x_all > 0, 1, 0); x_all <- standardize_matrix(x_all)
  y_base <- x_all
} else {
  common <- intersect(attrs$image_id, emb$image_id)
  attrs <- attrs[match(common, attrs$image_id), , drop = FALSE]
  emb   <- emb[match(common, emb$image_id), , drop = FALSE]
  a_all <- attrs[[cfg$treatment_name]]
  x_all <- as.matrix(attrs[, x_names, drop = FALSE])
  x_all <- ifelse(x_all > 0, 1, 0); x_all <- standardize_matrix(x_all)
  y_base <- as.matrix(emb[, setdiff(names(emb), "image_id"), drop = FALSE])
  y_base <- standardize_matrix(y_base)
}

n_total <- nrow(x_all)
if (n_total < cfg$ref_size + max(cfg$n_values_ext))
  stop("Not enough units for the requested reference and max working sizes.")

ref_idx  <- sample(seq_len(n_total), cfg$ref_size)
pool_idx <- setdiff(seq_len(n_total), ref_idx)

# Reference propensity (in-sample is acceptable for the reference; large sample).
pi_ref <- safe_logit_fit(
  x_train = x_all[ref_idx, , drop = FALSE],
  a_train = a_all[ref_idx],
  x_new   = x_all[ref_idx, , drop = FALSE]
)

# Stein test class, fixed globally so the same functionals are evaluated
# across all (r, n) settings and replicates.
g_list <- make_g_functions(cfg$eval_dim, cfg$n_random_g)

# ----------------------------
# 4. Per-r precomputation
# ----------------------------

# Build representations and reference proxies once per ambient r. Note
# that the bandwidth h depends on n, so the reference density / score
# grids are recomputed per (r, n); the heavy quantities (projections,
# local covariances on the reference, MC sample points) are reused.
make_per_r_objects <- function(ambient_r) {
  reps <- make_representations(y_base, ambient_r, cfg$eval_dim, cfg$geom_dim_cap)
  y_eval_all <- reps$y_eval; y_geom_all <- reps$y_geom

  grid_obj <- make_grid(y_eval_all[ref_idx, , drop = FALSE], cfg$grid_size_1d)

  ref_a_local <- which(a_all[ref_idx] == cfg$target_a)   # positions in ref_idx
  ref_a_idx   <- ref_idx[ref_a_local]
  w_ref       <- 1 / pi_ref[ref_a_local]

  # Local covariances on the reference treated bank, exclude self.
  cov_ref_aniso <- local_cov_from_bank(
    query_geom = y_geom_all[ref_a_idx, , drop = FALSE],
    bank_geom  = y_geom_all[ref_a_idx, , drop = FALSE],
    bank_eval  = y_eval_all[ref_a_idx, , drop = FALSE],
    k = cfg$local_k, ridge = cfg$local_ridge, exclude_self = TRUE
  )

  mc_n     <- min(cfg$stein_mc_size, length(ref_a_idx))
  mc_local <- sample(seq_along(ref_a_idx), mc_n)         # indices into ref_a_idx
  mc_points <- y_eval_all[ref_a_idx[mc_local], , drop = FALSE]
  mc_weights <- w_ref[mc_local]                           # IPW weights at MC

  list(
    y_eval_all = y_eval_all,
    y_geom_all = y_geom_all,
    grid = grid_obj$grid, cell_volume = grid_obj$cell_volume,
    ref_a_idx = ref_a_idx, w_ref = w_ref,
    cov_ref_aniso = cov_ref_aniso,
    mc_points = mc_points, mc_weights = mc_weights
  )
}

# ----------------------------
# 5. One run for a given r and n
# ----------------------------

run_one_setting <- function(r_obj, ambient_r, n_work, n_reps) {
  y_eval_all <- r_obj$y_eval_all; y_geom_all <- r_obj$y_geom_all
  grid <- r_obj$grid; cell_volume <- r_obj$cell_volume
  ref_a_idx <- r_obj$ref_a_idx; w_ref <- r_obj$w_ref
  cov_ref_aniso <- r_obj$cov_ref_aniso
  mc_points <- r_obj$mc_points; mc_weights <- r_obj$mc_weights

  h <- cfg$h_constant * n_work^(-cfg$h_power)

  # Reference density uses Horvitz-Thompson scaling (weights / n_ref,
  # normalize = FALSE) so that p_ref and the AIPW one-step p_hat target the
  # same finite-sample object up to estimation noise.
  p_ref_iso <- weighted_density_grid(
    centers = y_eval_all[ref_a_idx, , drop = FALSE],
    grid = grid, weights = w_ref / cfg$ref_size, h = h, type = "isotropic",
    aniso_pre = NULL, chunk_center = cfg$chunk_center,
    normalize = FALSE
  )
  aniso_ref_pre <- precompute_aniso(cov_ref_aniso, h)
  p_ref_aniso <- weighted_density_grid(
    centers = y_eval_all[ref_a_idx, , drop = FALSE],
    grid = grid, weights = w_ref / cfg$ref_size, h = h, type = "anisotropic",
    aniso_pre = aniso_ref_pre, chunk_center = cfg$chunk_center,
    normalize = FALSE
  )

  # Reference score at MC points using IPW-weighted reference treated units.
  score_ref_iso <- score_mixture(
    points = mc_points, centers = y_eval_all[ref_a_idx, , drop = FALSE],
    weights = w_ref, h = h, type = "isotropic", covs = NULL,
    chunk_center = cfg$chunk_center
  )
  score_ref_aniso <- score_mixture(
    points = mc_points, centers = y_eval_all[ref_a_idx, , drop = FALSE],
    weights = w_ref, h = h, type = "anisotropic", covs = cov_ref_aniso,
    chunk_center = cfg$chunk_center
  )

  # Reference Stein values use IPW-weighted means at MC points.
  psi_ref_iso   <- stein_values(mc_points, score_ref_iso,   g_list, mc_weights)
  psi_ref_aniso <- stein_values(mc_points, score_ref_aniso, g_list, mc_weights)

  dens_rows  <- list(); stein_rows <- list()

  for (b in seq_len(n_reps)) {
    idx_work <- sample(pool_idx, n_work)
    x <- x_all[idx_work, , drop = FALSE]
    a <- a_all[idx_work]
    y_eval <- y_eval_all[idx_work, , drop = FALSE]
    y_geom <- y_geom_all[idx_work, , drop = FALSE]
    folds <- make_folds(n_work, cfg$n_folds)
    pi_hat <- crossfit_propensity(x, a, folds)

    iso_est <- one_step_density_at_points(
      y_eval, y_geom, x, a, grid, h, method = "isotropic",
      target_a = cfg$target_a, folds = folds, pi_hat = pi_hat,
      lambda = cfg$ridge_lambda, local_k = cfg$local_k,
      local_ridge = cfg$local_ridge, chunk_grid = cfg$chunk_grid
    )
    aniso_est <- one_step_density_at_points(
      y_eval, y_geom, x, a, grid, h, method = "anisotropic",
      target_a = cfg$target_a, folds = folds, pi_hat = pi_hat,
      lambda = cfg$ridge_lambda, local_k = cfg$local_k,
      local_ridge = cfg$local_ridge, chunk_grid = cfg$chunk_grid
    )

    ise_iso          <- sum((iso_est$one_step   - p_ref_iso  )^2) * cell_volume
    ise_dis          <- sum((aniso_est$one_step - p_ref_aniso)^2) * cell_volume
    ise_aniso_plugin <- sum((aniso_est$plug_in  - p_ref_aniso)^2) * cell_volume

    work_a_idx <- which(a == cfg$target_a)
    skip_stein <- (length(work_a_idx) < 30L)

    # Symmetric skip: density and Stein rows are recorded together.
    if (!skip_stein) {
      # AIPW DSS via P_hat and G_hat at MC points.
      pq_iso <- one_step_pq_at_points(
        y_eval, y_geom, x, a, mc_points, h, method = "isotropic",
        target_a = cfg$target_a, folds = folds, pi_hat = pi_hat,
        lambda = cfg$ridge_lambda, local_k = cfg$local_k,
        local_ridge = cfg$local_ridge, chunk_grid = cfg$chunk_grid
      )
      pq_aniso <- one_step_pq_at_points(
        y_eval, y_geom, x, a, mc_points, h, method = "anisotropic",
        target_a = cfg$target_a, folds = folds, pi_hat = pi_hat,
        lambda = cfg$ridge_lambda, local_k = cfg$local_k,
        local_ridge = cfg$local_ridge, chunk_grid = cfg$chunk_grid
      )
      score_dss_iso   <- pq_iso$G   / pmax(pq_iso$P,   cfg$p_min_trunc)
      score_dss_aniso <- pq_aniso$G / pmax(pq_aniso$P, cfg$p_min_trunc)

      # Treated-only plug-in baseline (unweighted KDE on treated, no IPW).
      score_treated <- score_mixture(
        points = mc_points,
        centers = y_eval[work_a_idx, , drop = FALSE],
        weights = rep(1, length(work_a_idx)),
        h = h, type = "isotropic", covs = NULL,
        chunk_center = cfg$chunk_center
      )

      psi_dss_iso   <- stein_values(mc_points, score_dss_iso,   g_list, mc_weights)
      psi_dss_aniso <- stein_values(mc_points, score_dss_aniso, g_list, mc_weights)
      psi_treated   <- stein_values(mc_points, score_treated,   g_list, mc_weights)

      mse_treated    <- mean((psi_treated   - psi_ref_iso  )^2)
      mse_dss_iso    <- mean((psi_dss_iso   - psi_ref_iso  )^2)
      mse_dss_aniso  <- mean((psi_dss_aniso - psi_ref_aniso)^2)

      dens_rows[[length(dens_rows)+1L]] <- data.frame(
        r = ambient_r, n = n_work, rep = b,
        method = "Isotropic one step",  ise = ise_iso)
      dens_rows[[length(dens_rows)+1L]] <- data.frame(
        r = ambient_r, n = n_work, rep = b,
        method = "Anisotropic plug in", ise = ise_aniso_plugin)
      dens_rows[[length(dens_rows)+1L]] <- data.frame(
        r = ambient_r, n = n_work, rep = b,
        method = "DIS local geometry",  ise = ise_dis)

      stein_rows[[length(stein_rows)+1L]] <- data.frame(
        r = ambient_r, n = n_work, rep = b,
        method = "Treated only plug in", mse = mse_treated)
      stein_rows[[length(stein_rows)+1L]] <- data.frame(
        r = ambient_r, n = n_work, rep = b,
        method = "DSS isotropic",        mse = mse_dss_iso)
      stein_rows[[length(stein_rows)+1L]] <- data.frame(
        r = ambient_r, n = n_work, rep = b,
        method = "DSS anisotropic",      mse = mse_dss_aniso)

      message("Finished r=", ambient_r, ", n=", n_work, ", rep=", b)
    } else {
      message("Skipping r=", ambient_r, ", n=", n_work, ", rep=", b,
              " (too few treated units in working sample).")
    }
  }
  list(
    density = if (length(dens_rows))  do.call(rbind, dens_rows)  else data.frame(),
    stein   = if (length(stein_rows)) do.call(rbind, stein_rows) else data.frame()
  )
}

# ----------------------------
# 6. Run experiment
# ----------------------------

all_density <- list(); all_stein <- list()

for (r_now in cfg$ambient_dims) {
  r_obj <- make_per_r_objects(r_now)        # built once per r
  for (n_now in cfg$n_values_ext) {
    res <- run_one_setting(r_obj, r_now, n_now, cfg$n_reps_ext)
    if (nrow(res$density)) all_density[[length(all_density)+1L]] <- res$density
    if (nrow(res$stein))   all_stein[[length(all_stein)+1L]]     <- res$stein
  }
}

density_raw <- do.call(rbind, all_density)
stein_raw   <- do.call(rbind, all_stein)

write.csv(density_raw, file.path(cfg$out_dir, "density_raw.csv"), row.names = FALSE)
write.csv(stein_raw,   file.path(cfg$out_dir, "stein_raw.csv"),   row.names = FALSE)

density_sum <- aggregate(ise ~ r + n + method, data = density_raw, FUN = mean)
stein_sum   <- aggregate(mse ~ r + n + method, data = stein_raw,   FUN = mean)

write.csv(density_sum, file.path(cfg$out_dir, "density_summary.csv"), row.names = FALSE)
write.csv(stein_sum,   file.path(cfg$out_dir, "stein_summary.csv"),   row.names = FALSE)

slope_density <- estimate_slopes(density_sum, "ise")
slope_stein   <- estimate_slopes(stein_sum,   "mse")
write.csv(slope_density, file.path(cfg$out_dir, "table_slopes_density.csv"),
          row.names = FALSE)
write.csv(slope_stein,   file.path(cfg$out_dir, "table_slopes_stein.csv"),
          row.names = FALSE)

# ----------------------------
# 7. Plots
# ----------------------------

density_sum$method <- factor(density_sum$method,
  levels = c("Isotropic one step", "Anisotropic plug in", "DIS local geometry"))
stein_sum$method <- factor(stein_sum$method,
  levels = c("Treated only plug in", "DSS isotropic", "DSS anisotropic"))

plot_density_results(density_sum,
  file.path(cfg$out_dir, "fig_exp_density_extended_n8_v2.png"))
plot_stein_results(stein_sum,
  file.path(cfg$out_dir, "fig_exp_stein_extended_n8_v2.png"))

density_main <- density_sum[density_sum$n %in% cfg$n_values_main, , drop = FALSE]
stein_main   <- stein_sum[  stein_sum$n   %in% cfg$n_values_main, , drop = FALSE]
plot_density_results(density_main, file.path(cfg$out_dir, "fig_exp_density.png"))
plot_stein_results(  stein_main,   file.path(cfg$out_dir, "fig_exp_stein.png"))

message("Done. Results and figures saved in: ", cfg$out_dir)
