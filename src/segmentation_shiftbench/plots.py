"""Scientific figures with explicit synthetic-data and uncertainty labels."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch

from .model import threshold_predict
from .shifts import CONDITIONS, perturb


COLORS = {'standard': '#1d6f9b', 'augmented': '#ba5834', 'threshold': '#638178'}


def plot_results(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, family, label in zip(axes, ('noise', 'gamma', 'blur'), ('Gaussian noise', 'Gamma darkening', 'Gaussian blur')):
        for model in ('threshold', 'standard', 'augmented'):
            subset = summary[(summary.model == model) & summary.family.isin(('clean', family))].sort_values('severity')
            x = subset.severity.to_numpy()
            ax.plot(x, subset.mean_dice, label=model, color=COLORS[model], marker='o', markersize=4)
            ax.fill_between(x, subset.ci_low, subset.ci_high, color=COLORS[model], alpha=.12)
        ax.set(title=label, xlabel='Predefined severity', xticks=[0, 1, 2, 3], ylim=(0, 1.03))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Foreground Dice (equal specimen weight)')
    axes[-1].legend(loc='lower left', frameon=False)
    fig.suptitle('Segmentation ShiftBench | Synthetic ring phantoms', fontsize=15, x=.08, ha='left')
    fig.text(.08, .015, 'Shading: 95% specimen-bootstrap intervals, conditional on one training run. Not a medical benchmark.', fontsize=9)
    fig.tight_layout(rect=(.02, .055, 1, .92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_examples(models: dict, images: torch.Tensor, masks: np.ndarray, path: Path, *, seed: int) -> None:
    selected = [(index, condition) for index, condition in enumerate(CONDITIONS)
                if condition.name in ('clean', 'noise-3', 'gamma-3', 'blur-3')]
    fig, axes = plt.subplots(4, 5, figsize=(10, 8.5))
    cmap = ListedColormap(['#16252e', '#47aa9e', '#efaa65'])
    for row, (index, condition) in enumerate(selected):
        shifted = perturb(images, condition, seed=seed + index)
        predictions = []
        for name in ('standard', 'augmented'):
            models[name].eval()
            with torch.inference_mode():
                predictions.append(models[name](shifted).argmax(1)[0].numpy())
        panels = [shifted[0, 0].numpy(), masks[0], *predictions, threshold_predict(shifted)[0].numpy()]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel, cmap='gray' if col == 0 else cmap,
                                  vmin=0, vmax=1 if col == 0 else 2, interpolation='nearest')
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(['Input', 'Known labels', 'Standard', 'Augmented', 'Threshold'][col], fontsize=10)
        axes[row, 0].set_ylabel(condition.name, fontsize=10)
    fig.suptitle('First held-out synthetic view | Fixed example, not selected by performance', fontsize=12)
    fig.text(.11, .015, 'Labels: dark = background; green = ring; orange = core. Shapes are not biological measurements.', fontsize=9)
    fig.tight_layout(rect=(.02, .035, 1, .95))
    fig.savefig(path, dpi=160)
    plt.close(fig)
