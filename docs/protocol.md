# Experimental protocol

## Question

How do clean-trained and augmentation-trained versions of the same compact segmentation network behave under predefined intensity shifts? A fixed threshold predictor is included to make the comparison interpretable on simple generated patterns.

## Data

Generate 40 geometric specimens with randomized ring radii, core-to-ring ratios and center positions on a grid. Each specimen has three observations sharing the same geometric mask but with independently generated illumination and noise. This repeated-view structure is intentional: splitting by image would allow related geometry into different partitions.

Images are float32 tensors in [0, 1], size 1 x 64 x 64. Labels are background=0, ring=1 and core=2. Intensities are generated around 0.13, 0.45 and 0.82. The threshold baseline uses the corresponding midpoints 0.29 and 0.635, so its behavior is informed by the generator, not learned from the test set.

These are schematic mathematical phantoms. They are not an acquisition simulation or an anatomical model. No employer data, patient data or external datasets are used.

## Splits and learning

Use two seeded GroupShuffleSplit steps: hold out 20% of specimens for testing, then 25% of the remaining specimens for validation. This yields a 60/20/20 specimen allocation at the default size. Assert that groups do not overlap and export every assignment.

Train two TinyUNet models for 12 epochs on CPU. Each uses eight base channels, Adam with learning rate 0.003, batch size 8, and cross-entropy weights computed from training masks only. The two models have identical initialization and data-order seeds. Standard training sees the generated images. Augmented training additionally applies independently sampled gamma, offset and Gaussian noise during each batch. No test image is transformed as part of training.

Select each checkpoint independently using the minimum clean validation loss. Selection does not inspect test labels, shifted test performance, plots, or aggregate test scores. The epoch budget, shift families and severity values were fixed before the recorded run, and were not tuned after seeing its outputs.

## Evaluation

Run all predictors on identical held-out inputs under ten conditions:

- Clean.
- Additive zero-mean Gaussian noise with standard deviation 0.06, 0.12 and 0.20, followed by clipping to [0, 1].
- Gamma darkening with exponent 1.3, 1.7 and 2.2.
- Gaussian blur with sigma 0.5, 1.0 and 1.5 pixels, a kernel truncated at three sigma and reflect padding.

Generate a condition once, then reuse its tensor for every predictor. These photometric changes do not change the known geometric labels. No claim is made that these conditions represent the range or frequency of real acquisition artifacts.

Dice for a class is twice the predicted/target intersection divided by the sum of their pixel counts. A class missing from both maps is undefined and omitted from the foreground mean. A class missing from only one map has Dice zero. Scores exclude background.

First average views within each test specimen, then compute the equally weighted mean across test specimens. Draw 1,000 bootstrap resamples of specimen means and report the 2.5th and 97.5th percentiles. This choice avoids treating related views as independent observations. The intervals are conditional on this synthetic test cohort and one trained model, not an estimate of clinical performance or multi-seed uncertainty.

## Figures and artifacts

Plot every predefined severity, including results unfavorable to augmentation. The prediction figure uses the first held-out view in split order; it is not selected based on a metric. Save per-image scores, summaries, splits, loss history, checkpoints and environment metadata. Checkpoints stay local and are ignored by Git.

## Interpretation boundaries

The comparison is intentionally small and educational. A perfect clean threshold score follows from the well-separated phantom intensities. A network's resilience to noise may reflect spatial smoothing, which can also create wrong boundaries. Better scores for one perturbation are not evidence of a universally superior training method.

Real scientific use would require representative acquisitions, documented labels, domain review, multiple independent training runs and a predeclared evaluation procedure. The code is intended to make those engineering questions visible, not to claim they are solved here.
