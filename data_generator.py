"""
data_generator.py
-----------------
Synthetic CERN particle track data generator.

Generates 28x28 grayscale images modelled after the CERN TrackML dataset:
  Class 0 — Signal Event   : smooth helicoidal particle track from the centre
  Class 1 — Background Noise: scattered, random, uncorrelated energy deposits
"""

import numpy as np


# ── Constants ───────────────────────────────────────────────────────────────
IMAGE_SIZE = 28          # pixels per side
N_CLASSES  = 2           # binary: signal vs noise


# ── Internal helpers ─────────────────────────────────────────────────────────

def _draw_helix(img: np.ndarray, rng: np.random.Generator) -> None:
    """
    Paint a smooth helicoidal (spiral) track originating at the detector
    centre onto *img* in-place.

    The track is created by:
      1. Choosing random starting radius, angular velocity, and pitch.
      2. Sampling ~60 points along the parametric spiral.
      3. Adding small Gaussian smearing to each point to mimic finite
         detector resolution.
    """
    cx, cy   = IMAGE_SIZE / 2, IMAGE_SIZE / 2   # detector centre
    r0       = rng.uniform(0.5, 2.0)            # initial radius (px)
    omega    = rng.uniform(0.4, 1.0)            # angular velocity (rad/step)
    pitch    = rng.uniform(0.3, 0.8)            # radial growth per radian
    n_turns  = rng.uniform(1.5, 3.5)            # number of full turns
    t_max    = n_turns * 2 * np.pi
    sigma    = rng.uniform(0.4, 0.9)            # detector smearing (px)

    ts = np.linspace(0, t_max, 60)
    rs = r0 + pitch * ts                        # radius grows along the track

    for t, r in zip(ts, rs):
        x_raw = cx + r * np.cos(omega * t)
        y_raw = cy + r * np.sin(omega * t)

        # smear each hit over a few pixels
        for _ in range(4):
            xi = int(round(x_raw + rng.normal(0, sigma)))
            yi = int(round(y_raw + rng.normal(0, sigma)))
            if 0 <= xi < IMAGE_SIZE and 0 <= yi < IMAGE_SIZE:
                img[yi, xi] = min(1.0, img[yi, xi] + rng.uniform(0.5, 1.0))


def _draw_background(img: np.ndarray, rng: np.random.Generator) -> None:
    """
    Scatter random, uncorrelated energy deposits across the detector plane
    to simulate thermal noise or detector background.

    ~30–60 randomly placed hits of random intensity are drawn, with no
    spatial correlation between them.
    """
    n_hits = rng.integers(30, 61)
    xs = rng.integers(0, IMAGE_SIZE, size=n_hits)
    ys = rng.integers(0, IMAGE_SIZE, size=n_hits)
    intensities = rng.uniform(0.3, 1.0, size=n_hits)
    for x, y, val in zip(xs, ys, intensities):
        img[y, x] = val


def _normalise(img: np.ndarray) -> np.ndarray:
    """Normalise pixel values to [0, 1]."""
    mx = img.max()
    return img / mx if mx > 0 else img


# ── Public API ───────────────────────────────────────────────────────────────

def generate_sample(label: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a single 28×28 grayscale image.

    Parameters
    ----------
    label : int
        0 → Signal (helix), 1 → Background (noise)
    seed  : int | None
        Optional RNG seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (28, 28), dtype float32, values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    if label == 0:
        _draw_helix(img, rng)
    else:
        _draw_background(img, rng)

    return _normalise(img)


def generate_dataset(
    n_samples: int,
    seed: int = 42,
    balance: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced (or imbalanced) dataset of particle-track images.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    seed      : int
        Master RNG seed.
    balance   : bool
        If True, half of *n_samples* will be signal, half background.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 28, 28, 1), float32
        Images with a channel dimension (required by Conv2D).
    y : np.ndarray, shape (n_samples,), float32
        Labels: 0.0 = signal, 1.0 = background.
    """
    rng    = np.random.default_rng(seed)
    images = []
    labels = []

    for i in range(n_samples):
        if balance:
            label = i % 2          # alternate signal / background
        else:
            label = int(rng.integers(0, 2))

        img = generate_sample(label, seed=int(rng.integers(0, 2**31)))
        images.append(img)
        labels.append(float(label))

    X = np.stack(images, axis=0)[..., np.newaxis]   # (N, 28, 28, 1)
    y = np.array(labels, dtype=np.float32)

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
