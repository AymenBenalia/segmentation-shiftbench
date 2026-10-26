"""Train on specimen-disjoint data, select on validation, then evaluate fixed shifts."""
from __future__ import annotations

import copy
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import __version__
from .data import Dataset, grouped_split, make_phantoms
from .metrics import foreground_dice, summarize
from .model import TinyUNet, threshold_predict
from .shifts import CONDITIONS, augment_training, perturb


@dataclass(frozen=True)
class Config:
    seed: int = 42
    epochs: int = 12
    specimens: int = 40
    views: int = 3
    size: int = 64
    batch_size: int = 8
    learning_rate: float = .003
    bootstrap: int = 1000
    threads: int = 2

    def validate(self) -> None:
        if self.seed < 0 or min(self.epochs, self.batch_size, self.bootstrap, self.threads) < 1:
            raise ValueError("Use a nonnegative seed and positive epochs, batch size, bootstrap and threads.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")


def train_model(dataset: Dataset, splits: dict, config: Config, *, augmented: bool) -> tuple[TinyUNet, list[dict]]:
    config.validate()
    torch.manual_seed(config.seed)
    model = TinyUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train = splits['train']
    validation = splits['validation']
    loader = DataLoader(TensorDataset(torch.from_numpy(dataset.images[train]), torch.from_numpy(dataset.masks[train])),
                        batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    val_images = torch.from_numpy(dataset.images[validation])
    val_masks = torch.from_numpy(dataset.masks[validation])
    # Class weights use the training partition only.
    counts = np.bincount(dataset.masks[train].ravel(), minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Every target class must occur in the training data.")
    weights = 1 / counts
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    augment_generator = torch.Generator().manual_seed(config.seed + 100)
    best_loss = float('inf')
    best_state = None
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.
        for images, masks in loader:
            if augmented:
                images = augment_training(images, augment_generator)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a nonfinite loss.")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
        model.eval()
        with torch.inference_mode():
            validation_loss = criterion(model(val_images), val_masks).item()
        history.append({'epoch': epoch, 'train_loss': train_loss / len(train), 'validation_loss': validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def evaluate(models: dict[str, nn.Module], images: torch.Tensor, masks: np.ndarray,
             groups: np.ndarray, indices: np.ndarray, *, seed: int) -> pd.DataFrame:
    """Compare every predictor on exactly the same perturbed test inputs.

    Neural predictors accept [N,1,H,W] float images and return [N,3,H,W]
    logits. The built-in threshold baseline is included automatically.
    """
    if not len(images) or len(images) != len(masks) or len(images) != len(groups) or len(images) != len(indices):
        raise ValueError("Nonempty test inputs, masks, specimen IDs and indices must have equal lengths.")
    if 'threshold' in models:
        raise ValueError("The predictor name 'threshold' is reserved for the baseline.")
    rows = []
    for condition_index, condition in enumerate(CONDITIONS):
        shifted = perturb(images, condition, seed=seed + condition_index)
        predictions = {'threshold': threshold_predict(shifted).numpy()}
        for name, model in models.items():
            model.eval()
            with torch.inference_mode():
                logits = model(shifted)
            if tuple(logits.shape) != (len(images), 3, *images.shape[-2:]) or not torch.isfinite(logits).all():
                raise ValueError(f"Predictor {name} must return finite logits with shape [N,3,H,W].")
            predictions[name] = logits.argmax(1).numpy()
        for name, prediction in predictions.items():
            for position, (actual, target) in enumerate(zip(prediction, masks)):
                rows.append({'sample_id': int(indices[position]), 'specimen': str(groups[position]), 'model': name,
                             'condition': condition.name, 'family': condition.family, 'severity': condition.severity,
                             **foreground_dice(actual, target)})
    return pd.DataFrame(rows)


def run_experiment(output: Path, config: Config) -> dict:
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty to preserve previous experiments.")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    dataset = make_phantoms(seed=config.seed, specimens=config.specimens, views=config.views, size=config.size)
    splits = grouped_split(dataset.groups, config.seed)
    split_rows = []
    for partition, indices in splits.items():
        for index in indices:
            split_rows.append({'sample_id': int(index), 'specimen': dataset.groups[index], 'partition': partition})
    pd.DataFrame(split_rows).to_csv(output / 'splits.csv', index=False)
    models, history = {}, []
    for name, augmented in [('standard', False), ('augmented', True)]:
        print(f'Training {name}: {config.epochs} epochs on CPU', flush=True)
        model, records = train_model(dataset, splits, config, augmented=augmented)
        models[name] = model
        torch.save(model.state_dict(), output / f'{name}.pt')
        history.extend({**record, 'model': name} for record in records)
    pd.DataFrame(history).to_csv(output / 'training.csv', index=False)
    test = splits['test']
    test_images = torch.from_numpy(dataset.images[test])
    scores = evaluate(models, test_images, dataset.masks[test], dataset.groups[test], test, seed=config.seed + 1000)
    scores.to_csv(output / 'per_image.csv', index=False)
    summary = summarize(scores, seed=config.seed + 2000, bootstrap=config.bootstrap)
    summary.to_csv(output / 'summary.csv', index=False)
    metadata = {
        'tool_version': __version__, 'config': asdict(config), 'dataset_sha256': dataset.fingerprint(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scikit-learn': sklearn.__version__, 'device': 'cpu'},
        'split_counts': {name: {'images': len(indices), 'specimens': len(set(dataset.groups[indices]))}
                         for name, indices in splits.items()},
        'data': 'Generated geometric ring phantoms. No clinical or microscopy acquisitions.',
        'checkpoint_selection': 'Minimum clean validation cross-entropy, independently for each model.',
        'initialization': 'Same model seed and data-order seed for both training conditions.',
        'conditions': [asdict(condition) for condition in CONDITIONS],
        'training_augmentation': {'gamma': [.7, 1.5], 'brightness_offset': [-.08, .08], 'noise_sigma': [0, .08]},
        'metric': 'Per-image mean Dice of ring/core; absent-in-both class is undefined and omitted.',
        'aggregation': 'Average views within each specimen, then give each specimen equal weight.',
        'interval': '95% percentile bootstrap over test specimens, conditional on this synthetic cohort and fitted model.',
        'limitations': ['One training seed, not a multi-seed benchmark.', 'No validation on real scientific acquisitions.',
                        'Perturbations alter intensities only; their severity is not a calibrated microscope simulation.',
                        'Reproducibility across hardware and dependency versions is not guaranteed.'],
    }
    (output / 'run.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    from .plots import plot_results, plot_examples
    plot_results(summary, output / 'robustness.png')
    plot_examples(models, test_images, dataset.masks[test], output / 'examples.png', seed=config.seed + 1000)
    print(f'Completed. Results: {output}', flush=True)
    return metadata
