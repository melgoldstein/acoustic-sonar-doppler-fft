# Acoustic sonar Doppler features

Extract spectral centroid, 85-percent energy roll-off and Hilbert-envelope blade
modulation frequency. Compare scaled logistic regression and RBF SVC on independently
seeded amplitude-modulated propeller signals with noise and frequency shifts.

```sh
pip install -e '.[test]'
sonar-classify
pytest -q
```

| CLI field | Meaning |
|---|---|
| balanced_accuracy | Mean recall on independently generated holdout trials |
| confusion_matrix | True rows, predicted columns: light, medium, heavy |
| train_seed / holdout_seed | Reproduction seeds, 31 / 87 |

| Model | Holdout balanced accuracy | Correct / trials |
|---|---:|---:|
| Logistic regression | 1.000 | 90 / 90 |
| RBF SVC | 1.000 | 90 / 90 |

The full confusion matrices and seeds are recorded in `evaluation.json`.

Each split contains 90 four-second mono waveforms at 2048 Hz. Scaling is fitted
only on training features. The signal model ties class to carrier and modulation
bands; its score does not estimate performance on field recordings or establish
tonnage from acoustics. Use labeled recordings and vessel-disjoint splits before
interpreting classes physically. Frequency shifts model Doppler effects, without
estimating radial velocity. Waveforms must contain a nonzero finite signal.
MIT.
