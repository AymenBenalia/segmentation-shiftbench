"""Synthetic ring phantoms and specimen-aware data splits.

These are generated geometric patterns, not microscopy acquisitions. Multiple
views of a specimen share geometry and must remain in the same data partition.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class Dataset:
    images: np.ndarray
    masks: np.ndarray
    groups: np.ndarray

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (self.images, self.masks, self.groups):
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


def make_phantoms(*, seed: int = 42, specimens: int = 40, views: int = 3, size: int = 64) -> Dataset:
    if specimens < 6 or views < 1 or size < 32 or size % 4:
        raise ValueError("Use at least 6 specimens, 1 view and image size >=32 divisible by 4.")
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[:size, :size]
    images, masks, groups = [], [], []
    for specimen in range(specimens):
        mask = np.zeros((size, size), dtype=np.int64)
        # Separated cells ensure that both foreground classes exist.
        centers = [(size * x / 4, size * y / 4) for x in (1, 2, 3) for y in (1, 2, 3)]
        for cx, cy in centers:
            cx, cy = cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2)
            outer = rng.uniform(size * .065, size * .10)
            inner = outer * rng.uniform(.45, .72)
            dist = (xx - cx) ** 2 + (yy - cy) ** 2
            mask[dist <= outer ** 2] = 1
            mask[dist <= inner ** 2] = 2
        # The same underlying geometry is observed with different illumination/noise.
        for view in range(views):
            levels = np.array([.13, .45, .82], dtype=np.float32)
            image = levels[mask] + rng.normal(0, .025, mask.shape)
            image = image * rng.uniform(.92, 1.08) + rng.uniform(-.02, .02)
            image += np.linspace(-.025, .025, size)[None, :]
            images.append(np.clip(image, 0, 1).astype(np.float32)[None])
            masks.append(mask.copy())
            groups.append(f"specimen-{specimen:03d}")
    return Dataset(np.stack(images), np.stack(masks), np.asarray(groups))


def grouped_split(groups: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(np.unique(groups)) < 6:
        raise ValueError("At least six distinct specimen IDs are required.")
    dummy = np.zeros(len(groups))
    train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(dummy, groups=groups))
    train_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1).split(
        dummy[train_val], groups=groups[train_val]))
    partitions = {"train": train_val[train_local], "validation": train_val[val_local], "test": test}
    sets = [set(groups[index]) for index in partitions.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Specimen leakage across partitions.")
    return partitions
