from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from segmentation_shiftbench.data import grouped_split, make_phantoms
from segmentation_shiftbench.experiment import Config, evaluate, run_experiment, train_model
from segmentation_shiftbench.metrics import foreground_dice, summarize
from segmentation_shiftbench.model import TinyUNet, threshold_predict
from segmentation_shiftbench.shifts import CONDITIONS, Condition, augment_training, gaussian_blur, perturb


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_dataset_reproducible_and_specimens_have_related_views():
    first = make_phantoms(specimens=6, views=2, seed=4, size=32)
    second = make_phantoms(specimens=6, views=2, seed=4, size=32)
    assert first.fingerprint() == second.fingerprint()
    assert first.images.shape == (12, 1, 32, 32)
    assert np.array_equal(first.masks[0], first.masks[1])
    assert not np.array_equal(first.images[0], first.images[1])
    assert first.groups[0] == first.groups[1]
    assert np.isfinite(first.images).all()
    assert 0 <= first.images.min() <= first.images.max() <= 1
    assert set(np.unique(first.masks)) == {0, 1, 2}


def test_seed_changes_dataset():
    assert make_phantoms(specimens=6, seed=4).fingerprint() != make_phantoms(specimens=6, seed=5).fingerprint()


@pytest.mark.parametrize('kwargs', [{'specimens': 5}, {'views': 0}, {'size': 30}, {'size': 33}])
def test_invalid_dataset_configuration(kwargs):
    with pytest.raises(ValueError):
        make_phantoms(**kwargs)


def test_partitions_are_disjoint_by_specimen_and_cover_all_views():
    groups = np.repeat([f'g{i}' for i in range(20)], 3)
    split = grouped_split(groups)
    sets = {name: set(groups[index]) for name, index in split.items()}
    assert not sets['train'] & sets['test']
    assert not sets['train'] & sets['validation']
    assert not sets['validation'] & sets['test']
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(len(groups)))
    assert all(np.array_equal(index, grouped_split(groups)[name]) for name, index in split.items())


def test_group_split_requires_multiple_specimens():
    with pytest.raises(ValueError):
        grouped_split(np.repeat(['one', 'two'], 20))


def test_model_output_shape_and_gradient():
    model = TinyUNet()
    images = torch.rand(2, 1, 32, 36)
    logits = model(images)
    assert logits.shape == (2, 3, 32, 36)
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, 32, 36, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()


@pytest.mark.parametrize('condition', CONDITIONS)
def test_perturbations_preserve_shape_range_and_source(condition):
    images = torch.linspace(0, 1, 1024).reshape(1, 1, 32, 32)
    original = images.clone()
    first = perturb(images, condition, seed=12)
    second = perturb(images, condition, seed=12)
    assert torch.equal(first, second)
    assert torch.equal(images, original)
    assert first.shape == images.shape
    assert torch.isfinite(first).all()
    assert 0 <= first.min() <= first.max() <= 1


def test_noise_changes_with_seed_and_clean_does_not():
    images = torch.full((1, 1, 32, 32), .5)
    condition = Condition('noise', 'noise', 1, .1)
    assert not torch.equal(perturb(images, condition, seed=1), perturb(images, condition, seed=2))
    assert torch.equal(perturb(images, CONDITIONS[0], seed=1), images)


@pytest.mark.parametrize('condition', [Condition('bad', 'unknown', 1, 0), Condition('bad', 'noise', 1, -1),
                                      Condition('bad', 'gamma', 1, 0), Condition('bad', 'blur', 1, -1),
                                      Condition('bad', 'gamma', 1, float('nan'))])
def test_bad_perturbations_rejected(condition):
    with pytest.raises(ValueError):
        perturb(torch.zeros(1, 1, 32, 32), condition, seed=1)


def test_gaussian_kernel_preserves_constant_field():
    images = torch.full((1, 1, 32, 32), .4)
    assert torch.allclose(gaussian_blur(images, 1.5), images, atol=1e-6)


def test_augmentation_does_not_mutate_inputs():
    images = torch.full((2, 1, 32, 32), .5)
    original = images.clone()
    result = augment_training(images, torch.Generator().manual_seed(1))
    assert torch.equal(images, original)
    assert not torch.equal(result, images)


def test_threshold_baseline_uses_known_cutoffs():
    images = torch.tensor([[[[.13, .45, .82]]]])
    assert threshold_predict(images).tolist() == [[[0, 1, 2]]]


def test_dice_perfect_disjoint_and_absent_classes():
    labels = np.array([[0, 1], [2, 0]])
    assert foreground_dice(labels, labels)['foreground_dice'] == 1
    assert foreground_dice(np.zeros_like(labels), labels)['foreground_dice'] == 0
    result = foreground_dice(np.zeros((2, 2)), np.zeros((2, 2)))
    assert np.isnan(result['foreground_dice'])
    ring = np.ones((2, 2), dtype=int)
    assert foreground_dice(ring, ring)['foreground_dice'] == 1
    assert np.isnan(foreground_dice(ring, ring)['core_dice'])


def test_dice_invalid_shapes_and_labels():
    with pytest.raises(ValueError):
        foreground_dice(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        foreground_dice(np.full((2, 2), 127), np.zeros((2, 2)))


def test_bootstrap_weights_specimens_not_views():
    rows = pd.DataFrame([{'model': 'm', 'condition': 'clean', 'family': 'clean', 'severity': 0,
                          'specimen': specimen, 'foreground_dice': score}
                         for specimen, score in [('a', 0), ('b', 1), ('b', 1), ('b', 1)]])
    first = summarize(rows, seed=42)
    second = summarize(rows, seed=42)
    assert first.iloc[0].mean_dice == .5  # Not .75, which would overweight b.
    assert first.iloc[0].n_specimens == 2
    pd.testing.assert_frame_equal(first, second)


def test_single_specimen_has_no_confidence_interval():
    rows = pd.DataFrame([{'model': 'm', 'condition': 'clean', 'family': 'clean', 'severity': 0,
                          'specimen': 'a', 'foreground_dice': 1}])
    result = summarize(rows, seed=1).iloc[0]
    assert np.isnan(result.ci_low) and np.isnan(result.ci_high)


def test_training_is_reproducible_and_never_uses_test_data():
    dataset = make_phantoms(specimens=6, views=1, size=32)
    splits = grouped_split(dataset.groups)
    config = Config(epochs=1, specimens=6, views=1, size=32, bootstrap=10, threads=1)
    model, history = train_model(dataset, splits, config, augmented=False)
    # Corrupt only the test data. Training and model selection must be unchanged.
    dataset.images[splits['test']] = 0
    dataset.masks[splits['test']] = 0
    second, second_history = train_model(dataset, splits, config, augmented=False)
    assert history == second_history
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in model.state_dict().items())


def test_evaluation_uses_same_inputs_for_predictors():
    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []
        def forward(self, images):
            self.inputs.append(images.clone())
            return torch.zeros(len(images), 3, *images.shape[-2:])
    first, second = Recorder(), Recorder()
    images = torch.full((2, 1, 32, 32), .5)
    masks = np.ones((2, 32, 32), dtype=int)
    scores = evaluate({'a': first, 'b': second}, images, masks, np.array(['a', 'b']), np.arange(2), seed=1)
    assert all(torch.equal(a, b) for a, b in zip(first.inputs, second.inputs))
    assert len(scores) == 3 * len(CONDITIONS) * 2


def test_complete_experiment_smoke(tmp_path):
    config = Config(epochs=1, specimens=6, views=1, size=32, bootstrap=10, threads=1)
    output = tmp_path / 'run'
    metadata = run_experiment(output, config)
    assert metadata['split_counts']['test']['specimens'] >= 1
    for name in ['standard.pt', 'augmented.pt', 'splits.csv', 'training.csv', 'per_image.csv',
                 'summary.csv', 'run.json', 'robustness.png', 'examples.png']:
        assert (output / name).is_file()
    summary = pd.read_csv(output / 'summary.csv')
    assert len(summary) == 3 * len(CONDITIONS)
    assert summary.mean_dice.between(0, 1).all()
    with pytest.raises(ValueError, match='empty'):
        run_experiment(output, config)
