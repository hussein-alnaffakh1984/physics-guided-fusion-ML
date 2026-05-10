# HIF-ML-Reliability

> **Physics-Guided Machine Learning for Heavy-Ion Fusion Cross Sections: Achieving State-of-the-Art Accuracy and Identifying Reliability Limits Under Extrapolation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)
[![Data: EXFOR](https://img.shields.io/badge/Data-EXFOR%20(IAEA)-green)](https://nds.iaea.org/exfor/)

A reproducible benchmark for machine learning models predicting heavy-ion fusion cross sections, with a focus on **reliability under physically meaningful extrapolation** rather than only random-split accuracy.

---

## Highlights

- **State-of-the-art accuracy.** A physics-guided XGBoost model (`Wong + XGBoost`) reaches **MAE = 0.111** in log-space on random splits, improving 14% over a recent SOTA (Li et al. 2023, MAE = 0.129).
- **A systematic extrapolation benchmark.** Six physically meaningful train/test splits (energy, mass, shell-closure, asymmetry, leave-one-system-out) reveal **2.2× to 16× degradation** of accuracy compared to random splits.
- **Uncertainty quantification breaks under extrapolation.** Coverage of nominally-90% prediction intervals collapses to **1.4% – 78%** for all four UQ methods studied (Quantile Regression, Deep Ensembles, Bootstrap, Conformal Prediction), including conformal variants that hold under exchangeability.
- **Open and reproducible.** All data (1,933 fusion cross-section points from 127 reaction systems compiled from EXFOR) and code are released here.

---

## Key results at a glance

### Random split benchmark (interpolation regime)

| Model | MAE (log σ) | RMSE | R² |
|---|---|---|---|
| Wong analytical (baseline) | 1.003 | 1.672 | -0.621 |
| Linear regression | 0.619 | 0.850 | 0.579 |
| Random Forest | 0.160 | 0.297 | 0.948 |
| LightGBM (direct) | 0.138 | 0.259 | 0.960 |
| XGBoost (direct) | 0.115 | 0.237 | 0.967 |
| **Wong + XGBoost (residual)** | **0.111** | **0.243** | **0.965** |

### Extrapolation benchmark (the main result)

| Scenario | MAE (Direct ML) | Degradation | Coverage of 90% Conformal CI |
|---|---|---|---|
| Random split | 0.115 | 1.0× | 88% (calibrated) |
| Energy: below → above | 0.704 | 6.1× | — |
| **Energy: above → below** | **1.844** | **16.0×** | **1.4%** (catastrophic) |
| Mass: light → heavy | 0.385 | 3.3× | 54% |
| Mass: heavy → light | 0.323 | 2.8× | 77% |
| Shell: far → near magic | 0.256 | 2.2× | 78% |
| Asymmetry: sym → asym | 0.453 | 3.9× | 52% |
| LOSO (mean over 95 systems) | 0.279 | 2.4× | — |

> **Take-home message.** Random-split MAE is *not* a reliable indicator of how well a model will perform on a fusion system or energy regime it has never seen. Out-of-distribution coverage of standard UQ methods can drop to single-digit percentages.

---

## Quick start

```bash
git clone https://github.com/<USER>/HIF-ML-Reliability.git
cd HIF-ML-Reliability
pip install -r requirements.txt
```

Run the full pipeline (uses the cleaned dataset bundled with the repo, no need to re-download EXFOR):

```python
from src.data import load_dataset
from src.models import HybridXGBoostModel
from src.evaluation import random_split_evaluation

df = load_dataset()                       # 1,933 points, 35 features
model = HybridXGBoostModel(use_wong=True)
metrics = random_split_evaluation(model, df, n_folds=5)
print(metrics)
# {'mae': 0.111, 'rmse': 0.243, 'r2': 0.965}
```

For the extrapolation experiments and uncertainty quantification, see the notebooks in `notebooks/`.

---

## Repository structure

```
HIF-ML-Reliability/
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt                # Python dependencies
├── CITATION.cff                    # citation metadata
│
├── notebooks/                      # numbered analysis notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_extrapolation.ipynb
│   ├── 05_uncertainty.ipynb
│   └── 06_results_analysis.ipynb
│
├── src/                            # reusable modules
│   ├── data/                       # data loading and preprocessing
│   ├── models/                     # ML models and physics baselines
│   ├── evaluation/                 # cross-validation and extrapolation
│   └── utils/                      # physics helpers (Bass, Wong, ...)
│
├── data/
│   ├── raw/                        # initial EXFOR pulls
│   ├── processed/                  # cleaned, deduplicated dataset
│   └── features/                   # engineered features (final)
│
├── results/
│   ├── figures/                    # 22 publication-quality figures
│   └── tables/                     # 7 result tables (CSV)
│
├── models/                         # trained model artifacts
├── docs/                           # methodology and reproducibility docs
└── tests/                          # unit tests
```

---

## Methodology

### Data
Heavy-ion fusion cross-section data (`,SIG,,FUS`) were extracted from the IAEA EXFOR library (snapshot `x4i3_X4-2023-04-29`, 25,796 entries) using the [`x4i3`](https://github.com/afedynitch/x4i3) Python interface. After deduplication and physical filtering (0.5 ≤ E_cm/V_B ≤ 3.5, σ > 10⁻⁶ mb), **1,933 measurements from 127 reaction systems** remained.

### Features (35)
Grouped by physical category:
- **Atomic numbers (9):** Z, N, A for projectile, target, and compound nucleus.
- **Coulomb / barrier (3):** Bass barrier height V_B, interaction radius R_int, reduced mass.
- **Asymmetry (4):** mass asymmetry, N/Z asymmetry of projectile/target/CN.
- **Energy-derived (5):** E_cm, log(E_cm), 1/E_cm, E - V_B, E/V_B.
- **Shell / magic (9):** distance to nearest magic number for Z, N of each fragment and CN; doubly-magic indicators.
- **Pairing (3):** even-even / even-odd / odd-odd type for proj, targ, CN.
- **Physics-derived (2):** fusion Q-value (Bethe-Weizsäcker), Wong analytical cross section.

A correlation/VIF analysis was used to remove six redundant features (e.g., `coulomb_param ≡ ZpZt/A_sum_third`). The full list is in [`data/features/feature_info_v2.json`](data/features/feature_info_v2.json).

### Models
- **Wong's formula** (analytical baseline).
- **Linear / Ridge** (sanity baselines).
- **Random Forest, XGBoost, LightGBM** (tree ensembles).
- **Hybrid (`Wong + ML`):** ML learns the residual `log σ_exp - log σ_Wong`. This approach reduces MAE by 3.5% on random splits but does not consistently help under extrapolation.

### Uncertainty quantification
Four methods evaluated against a 90% nominal coverage target:
- Quantile Regression (LightGBM, q ∈ {0.05, 0.50, 0.95}).
- Deep Ensemble (10 bagged XGBoost models, ±1.645σ).
- Split Conformal Prediction (with held-out calibration set).
- Bootstrap (50 resamples).

Two advanced conformal variants (locally-weighted, group conformal binned by E/V_B) are also tested.

### Extrapolation scenarios
1. Energy: train below-barrier → test above-barrier (and reverse).
2. Mass: train A_CN < 100 → test A_CN > 150 (and reverse).
3. Mass: train middle (100 ≤ A_CN ≤ 150) → test extremes.
4. Shell-closure: train far from magic numbers → test near magic.
5. Asymmetry: train symmetric (η < 0.3) → test asymmetric (η ≥ 0.5).
6. Leave-one-system-out (95 systems with ≥ 8 points each).

---

## Reproducibility

The full pipeline is deterministic given fixed random seeds (default: 42). To reproduce all numbers in the paper from the bundled data:

```bash
# end-to-end
make all

# or step-by-step
python -m src.evaluation.run_random_splits
python -m src.evaluation.run_extrapolation
python -m src.evaluation.run_uncertainty
```

Re-pulling EXFOR from scratch (optional, ~10 min):

```bash
python -m src.data.exfor_collector --output data/raw/exfor_fusion.csv
```

System tested: Ubuntu 22.04, Python 3.10, scikit-learn 1.6, XGBoost 2.x, LightGBM 4.x. Should also work on Google Colab.

---

## Citation

If you use this work, please cite:

```bibtex
@article{HIF_ML_Reliability_2026,
  title   = {Physics-Guided Machine Learning for Heavy-Ion Fusion Cross Sections:
             State-of-the-Art Accuracy and Reliability Limits Under Extrapolation},
  author  = {<author list to be added>},
  journal = {Nuclear Science and Techniques},
  year    = {2026},
  note    = {Manuscript in preparation.}
}
```

The dataset itself rests on the IAEA/NDS EXFOR library; please also cite:

> N. Otuka et al., "Towards a More Complete and Accurate Experimental Nuclear Reaction Data Library (EXFOR)," *Nucl. Data Sheets* 120, 272 (2014).

---

## Contributing

Issues and pull requests are welcome. We are particularly interested in:
- Additional reaction systems compiled from sources beyond EXFOR.
- Alternative physics baselines (CCFULL, ECC, density-constrained TDHF) for the residual approach.
- More sophisticated UQ methods (e.g., adaptive conformal under distribution shift).

Please open an issue first to discuss substantial changes.

---

## License

This project is released under the MIT License — see [`LICENSE`](LICENSE).

The EXFOR data are public-domain experimental nuclear-physics measurements distributed by the IAEA/NDS; please respect the original publication citations listed in each EXFOR entry.

---

## Acknowledgments

We thank the IAEA Nuclear Data Section and the NRDC network for maintaining EXFOR, and the developers of `x4i3` for making programmatic access practical.
