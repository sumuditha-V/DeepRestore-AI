"""
train.py
========
Train the DeepRestore AI denoising autoencoder on clean images, with a proper
train/validation split and validation-based early stopping / checkpointing.

Run `python scripts/prepare_dataset.py --download` first to create data/train,
data/val and data/test. Then:

    cd backend
    python train.py                       # DnCNN, default config
    python train.py --model unet --epochs 80

Designed to run on Google Colab (free GPU) or locally.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe (Colab / servers)
import matplotlib.pyplot as plt
import tensorflow as tf

from denoise_core import (NoisyPatchGenerator, build_model, list_images, mae)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "data", "train")
VAL_DIR = os.path.join(ROOT, "data", "val")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

NOISE_TYPES = ["gaussian", "poisson", "salt_pepper"]
NOISE_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25]


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepRestore AI denoiser.")
    p.add_argument("--model", choices=["dncnn", "unet"], default="dncnn")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["mae", "mse"], default="mae")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"TensorFlow {tf.__version__}")
    print("Devices:", [d.device_type for d in tf.config.list_physical_devices()])

    train_files = list_images(TRAIN_DIR)
    val_files = list_images(VAL_DIR)
    if not train_files:
        raise SystemExit(
            f"No training images in {TRAIN_DIR}. "
            "Run: python scripts/prepare_dataset.py --download")
    print(f"Train images: {len(train_files)} | Val images: {len(val_files)}")

    train_gen = NoisyPatchGenerator(
        train_files, args.batch_size, args.patch_size,
        NOISE_TYPES, NOISE_LEVELS, patches_per_image=8, augment=True)
    val_gen = NoisyPatchGenerator(
        val_files, args.batch_size, args.patch_size,
        NOISE_TYPES, NOISE_LEVELS, patches_per_image=4, augment=False) \
        if val_files else None

    model = build_model(args.model, args.patch_size)
    model.summary()

    loss = mae if args.loss == "mae" else "mse"
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=loss, metrics=["mae"])

    monitor = "val_loss" if val_gen else "loss"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "denoiser_best.h5"),
            monitor=monitor, save_best_only=True, mode="min", verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_DIR, "logs")),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = os.path.join(MODEL_DIR, "denoiser_final.h5")
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # Plot training curves.
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.title(f"DeepRestore AI ({args.model.upper()}) training history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    print("Saved training_history.png")


if __name__ == "__main__":
    main()
