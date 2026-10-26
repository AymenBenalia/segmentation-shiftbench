# Verification

Verified locally on 26 October 2026:

- 35 tests passed, including a complete training, evaluation and artifact-export smoke test.
- Ran the default 12-epoch experiment twice with seed 42 on CPU. The two runs produced identical `splits.csv`, `training.csv`, `per_image.csv`, `summary.csv` and `run.json` bytes. Every tensor in both checkpoint state dictionaries was exactly equal between runs.
- Inspected the generated robustness and prediction figures. Figures label the data as synthetic and the intervals as conditional on one training run.
- Built and installed the editable package into an isolated local virtual environment and verified the CLI entry point.

Recorded environment: Python 3.12.14, PyTorch 2.14.1, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.9.1 and matplotlib 3.11.2, CPU on macOS ARM64. The runtime supports deterministic repetition in this tested environment; no cross-platform bitwise equality is claimed.

The GitHub Actions workflow runs a small test experiment on Python 3.11 and 3.12. It does not assert the recorded default model's numerical scores across environments.
