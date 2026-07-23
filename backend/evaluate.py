"""
evaluate.py
===========
Quantitatively evaluate a trained DeepRestore AI model on the held-out test set.

For every test image and each (noise_type, noise_level) combination it:
  1. adds synthetic noise,
  2. denoises with the model,
  3. measures PSNR and SSIM of the noisy vs. denoised image against the clean
     ground truth.

It prints an averaged results table (ready to paste into the README) and saves a
qualitative before/after grid to output/evaluation_grid.png.

    cd backend
    python evaluate.py                          # uses models/denoiser_final.h5
    python evaluate.py --model models/denoiser_best.h5 --limit 50
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

from denoise_core import add_noise, list_images, mae, make_flexible_input, psnr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, "data", "test")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

NOISE_TYPES = ["gaussian", "poisson", "salt_pepper"]
NOISE_LEVELS = [0.1]  # evaluate at a representative level; extend if desired


def pad_to_multiple(img, multiple=8):
    h, w = img.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return img, h, w


def denoise(model, noisy):
    padded, h, w = pad_to_multiple(noisy)
    out = model.predict(np.expand_dims(padded, 0), verbose=0)[0]
    return np.clip(out[:h, :w], 0.0, 1.0)


def load_clean(path, max_side=512):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Cap size so evaluation stays fast on CPU.
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def main():
    ap = argparse.ArgumentParser(description="Evaluate DeepRestore AI (PSNR/SSIM).")
    ap.add_argument("--model", default=os.path.join("models", "denoiser_final.h5"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Max number of test images to use.")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train it first (train.py).")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = tf.keras.models.load_model(args.model, custom_objects={"mae": mae})
    # Models are trained on fixed patches; allow any image size at eval time.
    model = make_flexible_input(model)

    files = list_images(TEST_DIR)
    if not files:
        raise SystemExit(
            f"No test images in {TEST_DIR}. "
            "Run: python scripts/prepare_dataset.py --download")
    if args.limit:
        files = files[:args.limit]
    print(f"Evaluating on {len(files)} test images...\n")

    # results[noise_type] -> list of (psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised)
    results = {nt: [] for nt in NOISE_TYPES}
    sample = None  # kept for the qualitative grid

    for path in files:
        clean = load_clean(path)
        if clean is None:
            continue
        for nt in NOISE_TYPES:
            for lvl in NOISE_LEVELS:
                noisy = add_noise(clean, nt, lvl).astype(np.float32)
                denoised = denoise(model, noisy)
                results[nt].append((
                    psnr(clean, noisy), psnr(clean, denoised),
                    ssim(clean, noisy, channel_axis=-1, data_range=1.0),
                    ssim(clean, denoised, channel_axis=-1, data_range=1.0),
                ))
                if sample is None and nt == "gaussian":
                    sample = (clean, noisy, denoised)

    # --- Print table ---------------------------------------------------------
    print(f"{'Noise':<14}{'PSNR noisy':>12}{'PSNR denoised':>15}"
          f"{'SSIM noisy':>13}{'SSIM denoised':>15}")
    print("-" * 69)
    md_rows = []
    for nt in NOISE_TYPES:
        arr = np.array(results[nt])
        if len(arr) == 0:
            continue
        pn, pd, sn, sd = arr.mean(axis=0)
        print(f"{nt:<14}{pn:>12.2f}{pd:>15.2f}{sn:>13.3f}{sd:>15.3f}")
        md_rows.append(f"| {nt:<13} | 0.10 | {pn:.2f} | {pd:.2f} | {sd:.3f} |")

    print("\nMarkdown (paste into README):\n")
    print("| Noise type | Level | PSNR (noisy) | PSNR (denoised) | SSIM (denoised) |")
    print("|------------|-------|--------------|-----------------|-----------------|")
    print("\n".join(md_rows))

    # --- Qualitative grid ----------------------------------------------------
    if sample is not None:
        clean, noisy, denoised = sample
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, title in zip(
            axes, [clean, noisy, denoised], ["Clean", "Noisy", "Denoised"]):
            ax.imshow(np.clip(im, 0, 1))
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        out = os.path.join(OUTPUT_DIR, "evaluation_grid.png")
        plt.savefig(out, dpi=120)
        print(f"\nSaved qualitative comparison to {out}")


if __name__ == "__main__":
    main()
