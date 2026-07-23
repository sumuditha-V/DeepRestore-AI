"""
prepare_dataset.py
==================
Download and organise a clean-image dataset for training DeepRestore AI, split
into train / val / test folders.

The denoising autoencoder is trained *self-supervised*: we only need clean
images. Noise is added synthetically on the fly during training, and the model
learns to reconstruct the clean version. So this script only prepares clean
images.

Usage
-----
1) Download BSDS500 automatically and split it:

       python scripts/prepare_dataset.py --download

2) Use your own folder of clean images instead of downloading:

       python scripts/prepare_dataset.py --source path/to/my_images

Output layout::

    data/
      train/   (70%)
      val/     (15%)
      test/    (15%)
"""

import argparse
import os
import random
import shutil
import tarfile
import urllib.request

# Berkeley Segmentation Dataset 500 – 500 natural images, a standard denoising
# benchmark. ~70 MB download.
BSDS500_URL = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/"
    "BSR/BSR_bsds500.tgz"
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "_raw")


def download_bsds500(dest_dir):
    """Download and extract BSDS500, returning the folder holding the images."""
    os.makedirs(dest_dir, exist_ok=True)
    tgz_path = os.path.join(dest_dir, "BSR_bsds500.tgz")

    if not os.path.exists(tgz_path):
        print(f"Downloading BSDS500 (~70 MB) from:\n  {BSDS500_URL}")
        try:
            urllib.request.urlretrieve(BSDS500_URL, tgz_path)
        except Exception as e:
            raise SystemExit(
                f"\nDownload failed: {e}\n"
                "The Berkeley server can be slow/unreliable. Either retry, or "
                "download any clean-image folder yourself and run:\n"
                "  python scripts/prepare_dataset.py --source <your_folder>\n"
            )
    else:
        print("Archive already downloaded, reusing it.")

    print("Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(dest_dir)

    # Images live under BSR/BSDS500/data/images/{train,test,val}
    img_root = os.path.join(dest_dir, "BSR", "BSDS500", "data", "images")
    if not os.path.isdir(img_root):
        raise SystemExit(f"Could not find extracted images under {img_root}")
    return img_root


def collect_images(source_dir):
    """Recursively collect all image file paths under source_dir."""
    files = []
    for root, _, names in os.walk(source_dir):
        for name in names:
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(root, name))
    return files


def split_and_copy(files, seed=42, ratios=(0.70, 0.15, 0.15)):
    """Shuffle and copy files into data/train, data/val, data/test."""
    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    for split, split_files in splits.items():
        out_dir = os.path.join(DATA_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        for i, src in enumerate(split_files):
            ext = os.path.splitext(src)[1].lower()
            dst = os.path.join(out_dir, f"{split}_{i:04d}{ext}")
            shutil.copy(src, dst)
        print(f"  {split:5s}: {len(split_files)} images -> {out_dir}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Prepare the DeepRestore AI dataset.")
    parser.add_argument("--download", action="store_true",
                        help="Download BSDS500 automatically.")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to your own folder of clean images.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.download and not args.source:
        parser.error("Pass either --download or --source <folder>.")

    if args.source:
        source_dir = args.source
        if not os.path.isdir(source_dir):
            raise SystemExit(f"Source folder not found: {source_dir}")
    else:
        source_dir = download_bsds500(RAW_DIR)

    files = collect_images(source_dir)
    if not files:
        raise SystemExit(f"No images found under {source_dir}")

    print(f"\nFound {len(files)} images. Splitting 70/15/15...")
    split_and_copy(files, seed=args.seed)
    print("\nDone. Train with:  cd backend && python train.py")


if __name__ == "__main__":
    main()
