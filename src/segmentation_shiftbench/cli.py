import argparse
from pathlib import Path

from .experiment import Config, run_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Train and evaluate a CPU segmentation experiment on synthetic ring phantoms.')
    parser.add_argument('--output', type=Path, required=True, help='An empty output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--specimens', type=int, default=40)
    parser.add_argument('--views', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        run_experiment(args.output, Config(seed=args.seed, epochs=args.epochs, specimens=args.specimens, views=args.views))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Cannot run experiment: {exc}\n')
    return 0
