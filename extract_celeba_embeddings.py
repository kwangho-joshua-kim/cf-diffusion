"""
extract_celeba_embeddings.py
============================

Compute pretrained ResNet-50 embeddings for the CelebA image set and write
them as a CSV file consumable by dis_dss_experiment.R.

The R script expects a file (default name: celeba_embeddings.csv or .rds)
in cfg$data_dir whose first column is image_id (matching the filenames
in list_attr_celeba.txt) and whose remaining columns are numeric features.

Layout assumed on disk
----------------------
CELEBA_DIR/
  list_attr_celeba.txt           (or list_attr_celeba.csv, or Anno/list_attr_celeba.txt)
  img_align_celeba/
    000001.jpg
    000002.jpg
    ...

Usage
-----
    python extract_celeba_embeddings.py --celeba-dir /path/to/celeba

Output
------
CSV with header: image_id, f0001, f0002, ..., f2048
Roughly 25 GB for the full 202,599-image CelebA at default precision; use
--max-images for quick tests, or post-compress with gzip (R's data.table::fread
reads .csv.gz transparently).

Dependencies
------------
    pip install torch torchvision pillow

GPU is used automatically if available; otherwise it falls back to CPU.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# Some CelebA JPEGs trip Pillow's strict EXIF parser; tolerate it.
ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class CelebADataset(Dataset):
    def __init__(self, img_dir, image_ids, transform):
        self.img_dir = Path(img_dir)
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = self.img_dir / image_id
        img = Image.open(path).convert("RGB")
        return image_id, self.transform(img)


def build_encoder(device):
    """ImageNet-pretrained ResNet-50, with the classifier head removed so
    the forward pass returns the 2048-d global-pooled feature vector."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model, weights.transforms()


def main():
    p = argparse.ArgumentParser(
        description="Extract ResNet-50 embeddings for CelebA images."
    )
    p.add_argument("--celeba-dir", required=True,
                   help="Path to the CelebA root folder.")
    p.add_argument("--img-subdir", default="img_align_celeba",
                   help="Subfolder containing the JPEGs.")
    p.add_argument("--out-file", default=None,
                   help="Output CSV path "
                        "(default: <celeba-dir>/celeba_embeddings.csv).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None,
                   help="If set, only embed the first N images "
                        "(sorted by filename).")
    p.add_argument("--precision", type=int, default=5,
                   help="Decimal places per feature (default: 5).")
    args = p.parse_args()

    celeba_dir = Path(args.celeba_dir)
    img_dir = celeba_dir / args.img_subdir
    if not img_dir.is_dir():
        sys.exit(f"Image directory not found: {img_dir}")

    out_file = (Path(args.out_file) if args.out_file
                else celeba_dir / "celeba_embeddings.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Use the filenames as image_id, matching the convention in
    # list_attr_celeba.txt.
    image_ids = sorted(
        f.name for f in img_dir.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    )
    if args.max_images:
        image_ids = image_ids[: args.max_images]
    print(f"Found {len(image_ids)} images in {img_dir}.")

    if not image_ids:
        sys.exit("No images to process.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, transform = build_encoder(device)

    dataset = CelebADataset(img_dir, image_ids, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    feature_dim = 2048
    fmt = f"{{:.{args.precision}f}}"
    header = ["image_id"] + [f"f{i + 1:04d}" for i in range(feature_dim)]

    n_done = 0
    log_every = max(1, 10 * args.batch_size)

    with open(out_file, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)

        with torch.inference_mode():
            for ids, batch in loader:
                batch = batch.to(device, non_blocking=True)
                feats = model(batch).cpu().numpy()
                for image_id, vec in zip(ids, feats):
                    writer.writerow([image_id] + [fmt.format(v) for v in vec])
                n_done += len(ids)
                if n_done % log_every < args.batch_size or n_done == len(image_ids):
                    print(f"  processed {n_done}/{len(image_ids)}")

    print(f"Wrote {out_file}")
    print("Next step: set cfg$data_dir in dis_dss_experiment.R to "
          f"{celeba_dir} and run the script.")


if __name__ == "__main__":
    main()
