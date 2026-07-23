"""
denoise_core.py
===============
Shared building blocks for DeepRestore AI: synthetic noise, the autoencoder
model architectures (DnCNN / U-Net), and the patch-based data generator.

Imported by train.py, evaluate.py and app.py so the noise model and the network
definitions stay identical across training, evaluation and inference.
"""

import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #
def add_noise(image, noise_type="gaussian", noise_level=0.1):
    """Add synthetic noise to a float image in [0, 1]. Returns a new array."""
    noise_level = float(noise_level)
    noisy = image.copy()

    if noise_type == "gaussian":
        noise = np.random.normal(loc=0.0, scale=noise_level, size=image.shape)
        noisy = np.clip(image + noise, 0.0, 1.0)

    elif noise_type == "poisson":
        # Scale up so Poisson sampling is meaningful, then scale back.
        scaled = image * 255.0
        noise = np.random.poisson(scaled * noise_level) / (255.0 * noise_level)
        noisy = np.clip(noise, 0.0, 1.0)

    elif noise_type == "salt_pepper":
        amount = noise_level
        s_vs_p = 0.5
        num_salt = int(np.ceil(amount * image.size * s_vs_p))
        num_pepper = int(np.ceil(amount * image.size * (1.0 - s_vs_p)))
        coords = tuple(np.random.randint(0, dim, num_salt) for dim in image.shape)
        noisy[coords] = 1.0
        coords = tuple(np.random.randint(0, dim, num_pepper) for dim in image.shape)
        noisy[coords] = 0.0

    return noisy


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_dncnn_model(input_shape=(None, None, 3), depth=17):
    """DnCNN residual denoiser: predicts the noise, then subtracts it."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding="same")(inputs)
    x = layers.Activation("relu")(x)

    for _ in range(depth - 2):
        x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    x = layers.Conv2D(3, 3, padding="same")(x)          # predicted noise residual
    outputs = layers.Subtract()([inputs, x])            # clean = noisy - noise
    return models.Model(inputs, outputs, name="DnCNN")


def build_unet_model(input_shape=(None, None, 3)):
    """Compact U-Net denoiser (good at preserving fine detail)."""
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(b)

    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Conv2D(64, 2, activation="relu", padding="same")(u1)
    u1 = layers.Concatenate()([c2, u1])
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(u1)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)

    u2 = layers.UpSampling2D(2)(c3)
    u2 = layers.Conv2D(32, 2, activation="relu", padding="same")(u2)
    u2 = layers.Concatenate()([c1, u2])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u2)
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(c4)

    outputs = layers.Conv2D(3, 1, activation="sigmoid")(c4)
    return models.Model(inputs, outputs, name="UNet")


def build_model(model_type="dncnn", patch_size=64):
    shape = (patch_size, patch_size, 3)
    if model_type == "unet":
        return build_unet_model(shape)
    return build_dncnn_model(shape)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def list_images(folder):
    """Return all image paths directly inside `folder` (and subfolders)."""
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(root, name))
    return files


class NoisyPatchGenerator(tf.keras.utils.Sequence):
    """Yields (noisy_patch, clean_patch) batches with on-the-fly augmentation."""

    def __init__(self, filepaths, batch_size, patch_size,
                 noise_types, noise_levels, patches_per_image=4, augment=True):
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.noise_types = noise_types
        self.noise_levels = noise_levels
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(
            len(self.filepaths) * self.patches_per_image / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.filepaths))
        np.random.shuffle(self.indices)

    def _random_patch(self, image):
        h, w = image.shape[:2]
        ps = self.patch_size
        if h > ps and w > ps:
            top = np.random.randint(0, h - ps)
            left = np.random.randint(0, w - ps)
            return image[top:top + ps, left:left + ps]
        return cv2.resize(image, (ps, ps))

    def _augment(self, patch):
        k = np.random.randint(0, 4)
        if k:
            patch = np.rot90(patch, k)
        if np.random.rand() > 0.5:
            patch = np.fliplr(patch)
        return patch

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            index = self.indices[(idx * self.batch_size + i) % len(self.indices)]
            img = cv2.imread(self.filepaths[index])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            patch = self._random_patch(img)
            if self.augment:
                patch = self._augment(patch)
            patch = np.ascontiguousarray(patch, dtype=np.float32)

            noise_type = np.random.choice(self.noise_types)
            noise_level = np.random.choice(self.noise_levels)
            noisy = add_noise(patch, noise_type, noise_level)

            batch_x.append(noisy)
            batch_y.append(patch)

        # Guard against an all-failed batch.
        while len(batch_x) < self.batch_size:
            blank = np.zeros((self.patch_size, self.patch_size, 3), np.float32)
            batch_x.append(batch_x[0] if batch_x else blank)
            batch_y.append(batch_y[0] if batch_y else blank)

        return np.array(batch_x), np.array(batch_y)


# --------------------------------------------------------------------------- #
# Metrics / losses
# --------------------------------------------------------------------------- #
def mae(y_true, y_pred):
    """Mean-absolute-error loss (registered as a custom object for loading)."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def psnr(clean, other):
    """Peak signal-to-noise ratio between two float images in [0, 1]."""
    mse = np.mean((clean.astype(np.float64) - other.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def make_flexible_input(loaded_model):
    """Rebuild a model so it accepts any image size.

    Models are trained on fixed-size patches (e.g. 64x64), but the denoisers are
    fully convolutional, so at inference they can run on full images. We clone
    the architecture with a variable spatial input (None, None, 3) and copy the
    trained weights across.
    """
    config = loaded_model.get_config()
    for layer in config.get("layers", []):
        if layer.get("class_name") == "InputLayer":
            c = layer["config"]
            for key in ("batch_shape", "batch_input_shape"):
                shape = c.get(key)
                if shape and len(shape) == 4:
                    shape = list(shape)
                    shape[1], shape[2] = None, None
                    c[key] = shape
    flexible = tf.keras.Model.from_config(config)
    flexible.set_weights(loaded_model.get_weights())
    return flexible
