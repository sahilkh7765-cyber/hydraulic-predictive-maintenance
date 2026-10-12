# Predictive Maintenance of Industrial Hydraulic Systems
## Model comparison and results report

All numbers in this report were produced by executing `src/train.py` and
`src/evaluate.py` on the supplied UCI dataset. No value is estimated or copied
from the literature.

## 1. Dataset

- Source: UCI *Condition monitoring of hydraulic systems* (ZeMA gGmbH, Saarbrücken)
- Operating cycles (samples): **2,205**, each a constant-load cycle of 60 s
- Sensors: **17**
- Raw dimensions per cycle: **43,680**
- Total raw values: **96,314,400**
- Missing values: none (confirmed by the feature-quality audit)

### Sensors and sampling rates

| Sensor | Quantity | Unit | Sampling rate | Samples per cycle |
|---|---|---|---|---|
| PS1 | Pressure | bar | 100 Hz | 6000 |
| PS2 | Pressure | bar | 100 Hz | 6000 |
| PS3 | Pressure | bar | 100 Hz | 6000 |
| PS4 | Pressure | bar | 100 Hz | 6000 |
| PS5 | Pressure | bar | 100 Hz | 6000 |
| PS6 | Pressure | bar | 100 Hz | 6000 |
| EPS1 | Motor power | W | 100 Hz | 6000 |
| FS1 | Volume flow | l/min | 10 Hz | 600 |
| FS2 | Volume flow | l/min | 10 Hz | 600 |
| TS1 | Temperature | degC | 1 Hz | 60 |
| TS2 | Temperature | degC | 1 Hz | 60 |
| TS3 | Temperature | degC | 1 Hz | 60 |
| TS4 | Temperature | degC | 1 Hz | 60 |
| VS1 | Vibration | mm/s | 1 Hz | 60 |
| CE | Cooling efficiency (virtual) | % | 1 Hz | 60 |
| CP | Cooling power (virtual) | kW | 1 Hz | 60 |
| SE | Efficiency factor (virtual) | % | 1 Hz | 60 |

### Target variables

| Component | Values | Classes | Distribution |
|---|---|---|---|
| Cooler condition [%] | 3 = close to total failure, 20 = reduced efficiency, 100 = full efficiency | 3 | 3: 732, 20: 732, 100: 741 |
| Valve condition [%] | 73 = close to total failure, 80 = severe lag, 90 = small lag, 100 = optimal switching behavior | 4 | 73: 360, 80: 360, 90: 360, 100: 1125 |
| Internal pump leakage [level] | 0 = no leakage, 1 = weak leakage, 2 = severe leakage | 3 | 0: 1221, 1: 492, 2: 492 |
| Hydraulic accumulator [bar] | 90 = close to total failure, 100 = severely reduced pressure, 115 = slightly reduced pressure, 130 = optimal pressure | 4 | 90: 808, 100: 399, 115: 399, 130: 599 |

## 2. Feature engineering

One row of the engineered dataset is one complete 60-second operating cycle.

| Block | Features generated |
|---|---|
| Statistical | 251 |
| Time domain | 162 |
| Rolling | 54 |
| Frequency | 63 |
| Domain | 96 |
| **Total generated** | **626** |

### Feature counts through the pipeline

| Stage | Count |
|---|---|
| Raw dimensions per cycle | 43,680 |
| Engineered features generated | 626 |
| Dropped: constant | 4 |
| Dropped: near-zero variance | 4 |
| Dropped: exact duplicates | 65 |
| After cleaning | 553 |
| Cooler condition (selected model) | after cleaning: 17 -> reaching the classifier: 17 |
| Valve condition (selected model) | after cleaning: 553 -> after correlation filter: 238 -> after mutual information: 119 -> after tree importance: 40 -> reaching the classifier: 40 |
| Internal pump leakage (selected model) | after cleaning: 553 -> after correlation filter: 236 -> after mutual information: 118 -> after tree importance: 40 -> reaching the classifier: 40 |
| Hydraulic accumulator (selected model) | after cleaning: 553 -> reaching the classifier: 553 |

## 3. Experimental design

- Split: 80 % training / 20 % test, stratified per component target.
- Model selection: 5-fold stratified cross-validation **on the training set only**.
- Selection rule: the configuration with the highest mean cross-validated macro F1 is chosen, then evaluated once on the untouched test set.
- All learned transformations (imputation, scaling, correlation filtering, mutual information, tree importance) live inside a single scikit-learn `Pipeline`, so they are re-fitted on each training fold and never see validation or test rows.
- Leakage sanity check: `StratifiedGroupKFold` over contiguous blocks of 25 cycles, so a cycle is never in the same fold as its immediate neighbours.

### Three feature experiments

| Experiment | Representation |
|---|---|
| baseline | per-sensor cycle mean (17 features) |
| engineered | all 553 cleaned engineered features |
| selected | engineered features + correlation filter + mutual information + tree importance |

## 4. Cross-validated results (training set)

| Component | Feature set | Model | Accuracy | Macro F1 | Weighted F1 | Fit time (s) |
|---|---|---|---|---|---|---|
| cooler | baseline | logistic_regression | 0.9977 | 0.9977 | 0.9977 | 0.01 |
| cooler | baseline | random_forest | 0.9977 | 0.9977 | 0.9977 | 0.44 |
| cooler | baseline | gradient_boosting | 0.9983 | 0.9983 | 0.9983 | 0.12 |
| cooler | baseline | svm | 0.9983 | 0.9983 | 0.9983 | 0.00 |
| cooler | engineered | logistic_regression | 0.9977 | 0.9977 | 0.9977 | 0.06 |
| cooler | engineered | random_forest | 0.9977 | 0.9977 | 0.9977 | 1.18 |
| cooler | engineered | gradient_boosting | 0.9977 | 0.9977 | 0.9977 | 2.77 |
| cooler | engineered | svm | 0.9966 | 0.9966 | 0.9966 | 0.10 |
| cooler | selected | logistic_regression | 0.9983 | 0.9983 | 0.9983 | 1.58 |
| cooler | selected | random_forest | 0.9977 | 0.9977 | 0.9977 | 2.10 |
| cooler | selected | gradient_boosting | 0.9983 | 0.9983 | 0.9983 | 1.69 |
| cooler | selected | svm | 0.9972 | 0.9972 | 0.9972 | 1.41 |
| valve | baseline | logistic_regression | 0.7131 | 0.6722 | 0.7113 | 0.04 |
| valve | baseline | random_forest | 0.9683 | 0.9621 | 0.9681 | 0.87 |
| valve | baseline | gradient_boosting | 0.9739 | 0.9682 | 0.9740 | 0.59 |
| valve | baseline | svm | 0.5567 | 0.5287 | 0.5950 | 0.04 |
| valve | engineered | logistic_regression | 0.9739 | 0.9654 | 0.9739 | 0.48 |
| valve | engineered | random_forest | 0.9943 | 0.9925 | 0.9943 | 2.26 |
| valve | engineered | gradient_boosting | 0.9932 | 0.9917 | 0.9932 | 6.55 |
| valve | engineered | svm | 0.9620 | 0.9499 | 0.9619 | 0.34 |
| valve | selected | logistic_regression | 0.9932 | 0.9913 | 0.9932 | 1.97 |
| valve | selected | random_forest | 0.9955 | 0.9946 | 0.9955 | 2.69 |
| valve | selected | gradient_boosting | 0.9955 | 0.9942 | 0.9955 | 2.41 |
| valve | selected | svm | 0.9909 | 0.9891 | 0.9910 | 1.95 |
| pump | baseline | logistic_regression | 0.9813 | 0.9755 | 0.9814 | 0.03 |
| pump | baseline | random_forest | 0.9932 | 0.9906 | 0.9932 | 0.57 |
| pump | baseline | gradient_boosting | 0.9915 | 0.9881 | 0.9915 | 0.25 |
| pump | baseline | svm | 0.9864 | 0.9823 | 0.9865 | 0.01 |
| pump | engineered | logistic_regression | 0.9932 | 0.9911 | 0.9932 | 0.37 |
| pump | engineered | random_forest | 0.9960 | 0.9943 | 0.9960 | 1.89 |
| pump | engineered | gradient_boosting | 0.9960 | 0.9946 | 0.9960 | 3.92 |
| pump | engineered | svm | 0.9887 | 0.9855 | 0.9886 | 0.23 |
| pump | selected | logistic_regression | 0.9955 | 0.9947 | 0.9955 | 1.64 |
| pump | selected | random_forest | 0.9966 | 0.9954 | 0.9966 | 2.62 |
| pump | selected | gradient_boosting | 0.9966 | 0.9954 | 0.9966 | 1.98 |
| pump | selected | svm | 0.9921 | 0.9886 | 0.9921 | 1.60 |
| accumulator | baseline | logistic_regression | 0.6582 | 0.6424 | 0.6680 | 0.03 |
| accumulator | baseline | random_forest | 0.9717 | 0.9684 | 0.9717 | 0.67 |
| accumulator | baseline | gradient_boosting | 0.9745 | 0.9706 | 0.9746 | 0.44 |
| accumulator | baseline | svm | 0.7755 | 0.7735 | 0.7861 | 0.03 |
| accumulator | engineered | logistic_regression | 0.9796 | 0.9768 | 0.9796 | 0.41 |
| accumulator | engineered | random_forest | 0.9932 | 0.9922 | 0.9932 | 2.82 |
| accumulator | engineered | gradient_boosting | 0.9915 | 0.9907 | 0.9915 | 9.05 |
| accumulator | engineered | svm | 0.9819 | 0.9813 | 0.9819 | 0.27 |
| accumulator | selected | logistic_regression | 0.9756 | 0.9716 | 0.9757 | 2.13 |
| accumulator | selected | random_forest | 0.9915 | 0.9904 | 0.9915 | 3.08 |
| accumulator | selected | gradient_boosting | 0.9921 | 0.9911 | 0.9920 | 2.61 |
| accumulator | selected | svm | 0.9921 | 0.9905 | 0.9921 | 2.00 |

### Does feature engineering help? (best macro F1 per feature set)

| Component | baseline | engineered | selected | engineered - baseline | selected - baseline |
|---|---|---|---|---|---|
| accumulator | 0.9706 | 0.9922 | 0.9911 | +0.0216 | +0.0206 |
| cooler | 0.9983 | 0.9977 | 0.9983 | -0.0006 | -0.0000 |
| pump | 0.9906 | 0.9946 | 0.9954 | +0.0039 | +0.0048 |
| valve | 0.9682 | 0.9925 | 0.9946 | +0.0243 | +0.0263 |

## 5. Final test-set results (each model evaluated exactly once)

| Component | Model | Feature set | Test cycles | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|---|
| Cooler condition | gradient_boosting | baseline | 441 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Valve condition | random_forest | selected | 441 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Internal pump leakage | gradient_boosting | selected | 441 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hydraulic accumulator | random_forest | engineered | 441 | 0.9887 | 0.9881 | 0.9886 | 0.9999 |

### Cross-validated vs test vs temporal-block scores (macro F1)

| Component | CV (random folds) | Test (random split) | Temporal-block CV | Optimism of random split |
|---|---|---|---|---|
| Cooler condition | 0.9983 | 1.0000 | 0.9954 | +0.0029 |
| Valve condition | 0.9946 | 1.0000 | 0.9839 | +0.0107 |
| Internal pump leakage | 0.9954 | 1.0000 | 0.9683 | +0.0271 |
| Hydraulic accumulator | 0.9922 | 0.9881 | 0.9161 | +0.0760 |

## 6. Confusion matrices (test set)

**Cooler condition** - rows are true classes, columns predicted

| true \ predicted | 3 | 20 | 100 |
|---|---|---|---|
| 3 (close to total failure) | 147 | 0 | 0 |
| 20 (reduced efficiency) | 0 | 146 | 0 |
| 100 (full efficiency) | 0 | 0 | 148 |

**Valve condition** - rows are true classes, columns predicted

| true \ predicted | 73 | 80 | 90 | 100 |
|---|---|---|---|---|
| 73 (close to total failure) | 72 | 0 | 0 | 0 |
| 80 (severe lag) | 0 | 72 | 0 | 0 |
| 90 (small lag) | 0 | 0 | 72 | 0 |
| 100 (optimal switching behavior) | 0 | 0 | 0 | 225 |

**Internal pump leakage** - rows are true classes, columns predicted

| true \ predicted | 0 | 1 | 2 |
|---|---|---|---|
| 0 (no leakage) | 244 | 0 | 0 |
| 1 (weak leakage) | 0 | 99 | 0 |
| 2 (severe leakage) | 0 | 0 | 98 |

**Hydraulic accumulator** - rows are true classes, columns predicted

| true \ predicted | 90 | 100 | 115 | 130 |
|---|---|---|---|---|
| 90 (close to total failure) | 161 | 0 | 0 | 0 |
| 100 (severely reduced pressure) | 5 | 75 | 0 | 0 |
| 115 (slightly reduced pressure) | 0 | 0 | 80 | 0 |
| 130 (optimal pressure) | 0 | 0 | 0 | 120 |

## 7. Most important features per component

**Cooler condition** (top 10 by impurity importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `PS6_mean` | 0.5329 |
| 2 | `CE_mean` | 0.1289 |
| 3 | `CP_mean` | 0.0846 |
| 4 | `TS4_mean` | 0.0845 |
| 5 | `PS5_mean` | 0.0542 |
| 6 | `PS1_mean` | 0.0481 |
| 7 | `PS2_mean` | 0.0438 |
| 8 | `EPS1_mean` | 0.0146 |
| 9 | `VS1_mean` | 0.0020 |
| 10 | `PS4_mean` | 0.0016 |

**Valve condition** (top 10 by impurity importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `PS2_cv` | 0.2215 |
| 2 | `PS2_band3_energy_ratio` | 0.0896 |
| 3 | `PS2_max_abs_change` | 0.0877 |
| 4 | `dp_PS1_PS2_mean` | 0.0836 |
| 5 | `PS2_std_diff` | 0.0754 |
| 6 | `SE_std_diff` | 0.0728 |
| 7 | `FS1_rollstd_max` | 0.0348 |
| 8 | `FS1_max` | 0.0280 |
| 9 | `PS1_kurtosis` | 0.0274 |
| 10 | `PS1_slope` | 0.0235 |

**Internal pump leakage** (top 10 by impurity importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `FS1_q75` | 0.3419 |
| 2 | `FS1_rms` | 0.1565 |
| 3 | `EPS1_max` | 0.0882 |
| 4 | `EPS1_q25` | 0.0756 |
| 5 | `SE_std` | 0.0450 |
| 6 | `EPS1_mean` | 0.0393 |
| 7 | `PS2_cv` | 0.0193 |
| 8 | `VS1_skew` | 0.0187 |
| 9 | `cooling_conductance_TS2_TS4` | 0.0170 |
| 10 | `SE_max_abs_change` | 0.0139 |

**Hydraulic accumulator** (top 10 by impurity importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `PS3_rollmean_max` | 0.0419 |
| 2 | `PS3_rollmean_range` | 0.0413 |
| 3 | `PS1_band3_energy_ratio` | 0.0348 |
| 4 | `PS1_band2_energy_ratio` | 0.0253 |
| 5 | `PS1_rollstd_max` | 0.0225 |
| 6 | `EPS1_rollstd_max` | 0.0202 |
| 7 | `PS1_band1_energy_ratio` | 0.0196 |
| 8 | `PS1_std_diff` | 0.0186 |
| 9 | `PS1_spectral_centroid` | 0.0185 |
| 10 | `EPS1_band2_energy_ratio` | 0.0157 |

## 8. Limitations

- **No true Remaining Useful Life.** The dataset annotates a *condition level* per cycle; it contains no maintenance timestamps and no failure times, so RUL cannot be estimated and is not claimed anywhere in this project.
- **Single test rig, laboratory conditions.** All 2,205 cycles come from one rig running one repeated load programme, with faults induced deliberately. Transfer to a different hydraulic installation is untested.
- **Temporal correlation between cycles.** Neighbouring cycles were recorded moments apart under the same component setting. The random split therefore gives a mildly optimistic picture; the temporal-block cross-validation in section 5 quantifies the gap.
- **A chronological split is not possible on this dataset.** The conditions were varied as a designed sequence, so the last 20 % of cycles contain only one cooler class. Stratified splitting is the only protocol that leaves every class testable.
- **Fault levels are discrete and superimposed.** The rig sets each component to a fixed degradation level; real degradation is continuous, and several faults are active simultaneously in the same cycle.
- **The virtual channels CE, CP and SE are already derived quantities** computed by the rig, not independent measurements. Features built on them inherit whatever model the rig used.
- **Impurity-based importance is biased** towards continuous, high-cardinality features; the permutation importances in `outputs/figures/feature_importance/` are the more trustworthy ranking.

## 9. Reproducing these numbers

```bash
pip install -r requirements.txt
python src/train.py      # feature engineering, experiments, model selection
python src/evaluate.py   # single test-set evaluation + this report
streamlit run app.py     # dashboard
```

Random state: 42. Gradient boosting backend: XGBoost.
