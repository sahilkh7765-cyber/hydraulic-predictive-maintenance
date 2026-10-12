# Presentation, viva preparation and plain-language explanation

Companion to `model_comparison.md`. Every number quoted here comes from the
actual execution of `src/train.py` and `src/evaluate.py`.

---

# Part A — The project in plain engineering language

### The problem
A hydraulic power unit drives machinery. If the pump, a valve, the cooler or the
accumulator degrades and nobody notices, the unit eventually fails and stops
whatever it feeds. Fixing things on a fixed schedule wastes healthy parts and
still misses faults that appear between services. **Predictive maintenance**
means listening to the machine's own sensors and judging component health while
it runs.

### The dataset
A laboratory test rig repeats the same 60-second load programme 2,205 times. It
has a main working circuit and a cooling circuit joined through the oil tank. 17
sensors record pressure (PS1–PS6), motor power (EPS1), flow (FS1, FS2),
temperature (TS1–TS4), vibration (VS1) and three quantities the rig computes
itself (CE, CP, SE). While this runs, four components are deliberately set to
known degradation levels, and every cycle is labelled with those levels.

### What an "operating cycle" is
One cycle = one complete run of that 60-second programme = **one row** of the
final dataset. It is the natural unit here because the rig always does the same
thing, so two cycles are directly comparable.

### Why raw measurements are not ideal
One cycle holds **43,680 numbers** (7 sensors × 6000 at 100 Hz, 2 × 600 at
10 Hz, 8 × 60 at 1 Hz), and there are only 2,205 cycles. Three problems:

1. **More columns than rows.** Any model has vastly more parameters to fit than
   examples to learn from, and will memorise noise.
2. **The channels are not time-aligned.** Sample 500 of PS1 is 5 seconds in;
   sample 500 of TS1 does not exist. The raw data isn't one table.
3. **The fault signature lives between channels.** A leaking pump does not
   change any single reading dramatically — it changes the *relationship*
   between the power drawn and the flow delivered.

### Feature engineering — the fix
Describe each cycle by a few hundred meaningful numbers instead of 43,680 raw
ones. Five blocks, 626 features total, a **70× reduction**:

- **Statistical (251)** — the *shape* of each signal within the cycle: mean,
  median, standard deviation, min, max, range, RMS, skewness, kurtosis,
  quartiles. Degradation widens or shifts a distribution long before it moves
  the average.
- **Time-domain (162)** — how the signal *moves*: peak-to-peak, energy,
  first-difference statistics, the linear slope across the cycle, zero
  crossings, peak count. Two cycles can share a histogram and behave completely
  differently.
- **Rolling / local (54)** — a 1-second moving mean and moving standard
  deviation, then summarised. Catches brief events that a 60-second average
  erases. Window length comes from each sensor's own sampling rate, so 1 second
  is really 1 second on every channel.
- **Frequency (63)** — dominant frequency, spectral energy, spectral centroid,
  band energy ratios. Mechanical degradation moves energy around in frequency.
- **Hydraulic domain (96)** — the physics: pressure differentials, flow
  imbalance FS1−FS2, hydraulic power p·Q/600, electro-hydraulic efficiency,
  specific power W per l/min, temperature spreads, cooling conductance CP/ΔT.

### Feature selection
626 features are still a lot, and many are near-duplicates (a sensor's mean,
median and 50th percentile are almost the same number). Three stages, cheapest
first:

1. **Correlation filter** — drop one of any pair with \|r\| > 0.95. Unsupervised.
2. **Mutual information** — keep the top 50 % by statistical dependence on the
   target. Catches non-linear relationships a correlation would miss.
3. **Random Forest importance** — keep the top 40, scored in the presence of
   all the others.

Result: 626 → 553 (cleaning) → ~40 features reaching the classifier.

### Why these models
Logistic Regression as a linear yardstick; Random Forest for non-linearity plus
an interpretable importance ranking; XGBoost as the usual best tabular
performer; SVM for a genuinely different inductive bias. **No deep learning** —
2,205 samples with a few hundred features is precisely where boosted trees win,
and a neural network would cost the interpretability a maintenance system needs.

### How data leakage is prevented
Anything that *learns* from data — the imputation median, the scaler's mean and
spread, the correlation threshold, the mutual-information ranking, the tree
importances — is a **stage of one scikit-learn `Pipeline`**. Only that object is
cross-validated, so scikit-learn re-fits every stage on each training fold
automatically. Leaking by accident becomes impossible rather than merely
discouraged. The test set is touched exactly once, in a separate script.

### Evaluation
Accuracy, precision, recall, F1 (macro and weighted), confusion matrices,
classification reports and one-vs-rest ROC-AUC where technically valid.
**Macro F1 is the headline** because it weights every class equally — a model
cannot score well by ignoring a rare but critical failure state.

### The predictive-maintenance workflow, end to end
Sensors record a cycle → features extracted → saved pipeline imputes, scales and
selects → four models predict cooler, valve, pump and accumulator condition →
each maps to Normal / Degraded / Critical using the dataset's own documented
levels → the dashboard shows the status plus the features driving it, so an
engineer can check the alarm against the physics.

---

# Part B — Ten-slide presentation outline

**1. Title**
- Predictive Maintenance of Industrial Hydraulic Systems Using Feature Engineering
- Your name, roll number, AI & Data Science, DY Patil University
- UCI / ZeMA hydraulic condition-monitoring dataset, 2,205 cycles
- One-line claim: engineered features lift the hard targets by ~0.02 macro F1

**2. Problem statement**
- Unplanned hydraulic failure stops the line it feeds
- Scheduled maintenance wastes healthy parts and still misses faults
- Research question: do engineered statistical, temporal, frequency and hydraulic
  features improve ML-based condition monitoring?
- Explicitly *not* claimed: Remaining Useful Life — the dataset has no failure
  timestamps

**3. Dataset**
- 2,205 cycles × 60 s constant load; 17 sensors; 43,680 raw values per cycle
- Four targets: cooler (3 classes), valve (4), pump leakage (3), accumulator (4)
- Zero missing values, confirmed by the feature-quality audit
- Stable flag kept as metadata, not modelled

**4. Sensor architecture**
- Three sampling rates: 100 Hz (PS1–6, EPS1), 10 Hz (FS1–2), 1 Hz (TS1–4, VS1,
  CE, CP, SE)
- Primary working circuit + cooling-filtration circuit sharing the oil tank
- CE, CP, SE are *virtual* channels computed by the rig
- Consequence: the raw data is not one rectangular table — aggregation is forced

**5. Feature engineering** *(the core slide — spend the most time here)*
- 626 features in five blocks: statistical 251, time-domain 162, rolling 54,
  frequency 63, hydraulic domain 96
- 70× compression from raw dimensionality
- Rate-aware design: rolling windows and FFTs only for ≥10 Hz channels
- Log-spaced FFT bands, because the load programme runs at ~0.033 Hz
- Domain features justified from units alone: flow imbalance, hydraulic power,
  electro-hydraulic efficiency, cooling conductance

**6. Feature selection**
- Why: engineered features are collinear by construction
- Correlation filter (\|r\| > 0.95) → mutual information (top 50 %) → RF
  importance (top 40)
- Funnel: 626 → 553 after cleaning → ~40 at the classifier
- All three stages live inside the pipeline, so they are re-fitted per fold

**7. ML models and experimental design**
- Logistic Regression, Random Forest, XGBoost, SVM — and why each is present
- Three experiments: baseline (17 sensor means) / engineered / selected
- 80-20 stratified split; 5-fold CV on training only; 48 experiments
- Selection rule fixed in advance: highest CV macro F1, then one test evaluation

**8. Results**
- Feature engineering gain: valve **+0.024**, accumulator **+0.022**, pump
  +0.004, cooler −0.001
- SVM on valve: 0.53 → 0.95 → 0.99 across the three experiments
- Test macro F1: cooler 1.000, valve 1.000, pump 1.000, accumulator 0.988
- Top features are physically sensible: CE/CP for cooler, FS1/EPS1 for pump

**9. Streamlit dashboard**
- Six sections: overview, dataset explorer, feature engineering, prediction,
  maintenance status, explainability
- Accepts a dataset cycle or an uploaded raw-sensor CSV
- Loads saved pipelines — never retrains
- Normal / Degraded / Critical mapped from the dataset's own documented levels

**10. Conclusion and future work**
- Engineered features help where it matters: the documented hard targets
- Honest caveat: temporal-block CV costs the accumulator ~0.08 macro F1
- Future: ordinal loss, joint component modelling, second-rig validation, SHAP,
  drift monitoring on live data

---

# Part C — Viva questions and answers

### Predictive maintenance and the dataset

**1. What is predictive maintenance, and how does it differ from preventive
maintenance?**
Preventive maintenance services components on a fixed schedule regardless of
condition. Predictive maintenance infers actual health from sensor data and
intervenes only when degradation is detected — it avoids both unnecessary part
replacement and failures occurring between scheduled services.

**2. Why this dataset?**
It is one of the few public datasets where component conditions are *labelled
per cycle* under controlled, known degradation levels, across four components
simultaneously, with raw multi-rate sensor data rather than pre-extracted
features. That makes it ideal for studying feature engineering specifically,
since the raw signals are available to engineer from.

**3. What is an operating cycle here?**
One repetition of the rig's 60-second constant-load programme. It becomes one
row of the engineered dataset. It is the right unit because the programme is
identical each time, making cycles directly comparable.

**4. Why can't you predict Remaining Useful Life?**
RUL requires knowing time-to-failure. This dataset annotates a *condition level*
per cycle with no maintenance timestamps, no failure events and no run-to-failure
trajectories. Any RUL number would be fabricated. The project deliberately claims
condition classification only.

**5. What is the stable flag and why isn't it a target?**
Column 5 of `profile.txt` marks cycles where static conditions may not have been
reached. It describes the *rig's* state, not a component's health, so it is kept
as metadata.

### Feature engineering

**6. Why not feed the raw 43,680 samples to the model?**
Three reasons: 43,680 columns against 2,205 rows is an extreme p ≫ n regime; the
three sampling rates mean the channels are not time-aligned column-by-column; and
fault signatures are cross-channel relationships that a per-sample view never
exposes.

**7. Explain mean, standard deviation and RMS, and why you compute all three.**
Mean is the average level. Standard deviation is the spread about that level.
RMS is the quadratic mean, √(mean of x²) — for pressure and vibration it is
proportional to the signal's energy, not just its level. A signal can hold its
mean constant while its spread and energy change, which is exactly what early
degradation looks like.

**8. What do skewness and kurtosis add?**
Skewness measures asymmetry, kurtosis measures peakedness/tail weight. A valve
with switching lag spends more time in transition, which changes the *shape* of
the pressure distribution even when mean and standard deviation are nearly
unchanged.

**9. What is an FFT and why did you use it here?**
The Fast Fourier Transform decomposes a time signal into its frequency
components. Pressure and motor power contain periodic content from the pump and
the load programme; mechanical degradation redistributes energy across frequency
in ways time-domain statistics cannot see.

**10. Why only a few FFT features per channel?**
A 6000-point signal has 3001 spectrum bins. Using them all would add ~21,000
columns for the seven fast channels and recreate the exact p ≫ n problem feature
engineering exists to solve. Four to six interpretable descriptors — dominant
frequency, spectral energy, spectral centroid, band ratios — carry the useful
information at a fraction of the width.

**11. Why are the FFT bands log-spaced rather than equal-width?**
The load programme repeats at about 0.033 Hz, so nearly all energy sits below
1 Hz. With linear bands over 0–50 Hz, band 1 absorbed ~100 % of the energy and
bands 2–3 were constant zeros — useless features that cleaning would have
dropped. Log spacing gives resolution where the hydraulics actually live. I
verified this empirically before changing it.

**12. Why no FFT or rolling features on the temperature channels?**
They are sampled at 1 Hz, giving 60 points per cycle: a Nyquist limit of 0.5 Hz
and a frequency resolution of ~0.017 Hz. An FFT of that describes essentially
nothing, and a rolling window over 60 points just reproduces the global
statistics.

**13. What are rolling features and how did you choose the window?**
A moving mean and moving standard deviation over a 1-second window, then
summarised (max rolling std, range of rolling mean, etc.). The window is derived
from each sensor's own sampling rate — 100 samples at 100 Hz, 10 at 10 Hz — so
the *physical* window stays constant. A fixed sample count would mean 0.5 s on
one channel and 50 s on another.

**14. Give an example of a domain-specific feature and justify it physically.**
`flow_diff_FS1_FS2`, the imbalance between the two flow measurements. Internal
pump leakage means fluid returns to tank without doing work, so delivered flow no
longer matches flow measured elsewhere in the circuit. It is the most directly
physical leakage indicator available. Similarly, hydraulic power is
p[bar]·Q[l/min]/600 kW, and dividing it by EPS1 gives electro-hydraulic
efficiency, which falls as the pump wears.

**15. How did you avoid inventing physics you don't know?**
The documentation gives each channel's quantity and unit but *not* the rig's
internal piping topology. So no feature assumes which pipe connects to what —
every domain feature is justified from units alone. Where topology would have
been needed, I used all pressure pairs rather than guessing which pair matters.

**16. How do you handle division by zero?**
All ratios go through a `safe_divide` helper that returns a finite fill value
where the denominator is below a tolerance, rather than propagating an infinity.
The resulting table contains zero infinite and zero missing cells, verified by
the feature-quality report.

### Feature selection and leakage

**17. Describe your three-stage selection and why that order.**
Correlation filter (unsupervised, removes redundancy), then mutual information
(supervised, univariate), then Random Forest importance (supervised,
multivariate). Cheapest and least informed first. Filtering redundancy *before*
the supervised stages means they rank genuinely distinct candidates instead of
twenty near-copies of one signal.

**18. Why mutual information rather than correlation with the target?**
Correlation only detects monotonic linear relationships. Mutual information
detects *any* statistical dependence. Some hydraulic relationships here are
non-monotonic — a pressure ratio can move one way for mild valve lag and back for
severe lag — and a linear filter would score those near zero and discard them.

**19. What is data leakage, and how did you prevent it?**
Leakage is any test information reaching the model during training. The classic
mistake is computing scaling parameters or a feature ranking on the whole dataset
before splitting. Here every learned step is a stage of one `Pipeline`, and only
that object is cross-validated, so scikit-learn re-fits each stage on the
training fold alone. The prevention is structural, not procedural.

**20. Why didn't you use a chronological split for time-series data?**
I checked, and it is impossible on this dataset: the conditions were varied as a
designed sequence, so the final 20 % of cycles contain **only one cooler class**.
A chronological holdout would leave entire classes untestable. Stratified
splitting is the only workable protocol — and the residual temporal optimism is
quantified separately.

**21. So how do you know the random split isn't optimistic?**
I measured it. `StratifiedGroupKFold` over contiguous 25-cycle blocks holds out
whole runs, so a cycle is never scored while its immediate neighbours are in
training. The accumulator drops from 0.992 to 0.916 macro F1 under that protocol
— roughly 7–8 points of its apparent performance comes from temporal proximity.
That gap is reported in the results rather than hidden.

### Models, metrics and imbalance

**22. Why Random Forest, XGBoost and SVM rather than just the best one?**
Each has a different inductive bias. Random Forest gives interpretable
importances; XGBoost usually performs best on tabular data; SVM is margin-based
and provides a check that the result isn't an artefact of tree-based models.
Logistic Regression is the linear yardstick: if a line already separates the
classes, complexity is unjustified.

**23. What's the difference between Random Forest and XGBoost?**
Random Forest builds many deep trees **in parallel** on bootstrapped samples and
averages them — it reduces variance. XGBoost builds shallow trees
**sequentially**, each correcting the previous ensemble's residual errors, with
regularisation — it reduces bias. Bagging versus boosting.

**24. Why not deep learning?**
2,205 samples with a few hundred engineered features is exactly the regime where
gradient-boosted trees dominate. A neural network would need far more data,
would be harder to justify to a maintenance engineer, and would sacrifice the
feature-importance interpretability that makes an alarm actionable. Adding it
would be complexity for its own sake.

**25. Why is macro F1 your headline metric and not accuracy?**
The classes are imbalanced — the valve target is 51 % one class. A model
predicting only the majority class would score 51 % accuracy while being useless.
Macro F1 averages per-class F1 with equal weight, so ignoring a rare but
operationally critical failure state is heavily penalised.

**26. What's the difference between macro and weighted F1?**
Macro averages per-class F1 equally regardless of class size. Weighted averages
them in proportion to class support, so it tracks accuracy more closely. Macro is
the stricter and more appropriate metric for imbalanced fault detection.

**27. How did you handle class imbalance, and why not SMOTE?**
Class weighting where the imbalance ratio exceeds 1.5. SMOTE interpolates
between neighbouring feature vectors — but these classes are *ordered degradation
levels*, so interpolating between "weak leakage" and "severe leakage" would
manufacture cycles the rig never produced, precisely where the decision boundary
is hardest. Also the imbalance here is mild (worst ≈ 3:1), so resampling isn't
warranted.

**28. Why four separate models instead of one multi-output model?**
The four components fail through different physical mechanisms and their labels
have different class counts and different difficulty. Separate models let each
select its own features — the cooler model keys on CE/CP, the pump model on
FS1/EPS1 — and let each be diagnosed independently. A joint model is listed as
future work since the faults do interact through the shared oil circuit.

**29. When is ROC-AUC appropriate for a multiclass problem?**
It must be extended, typically one-vs-rest with macro averaging, and it needs
probability estimates plus every class present in the test set. My code returns
`n/a` when those conditions fail rather than computing it over a subset of
classes and mislabelling the result.

### PCA, results and deployment

**30. Why did you use PCA only for visualisation?**
Principal components are linear mixtures of hundreds of features, so a model
built on them cannot tell an engineer *why* an alarm fired — which defeats the
purpose of condition monitoring. As a 2-D picture it is useful: it showed the
cooler classes forming separate clusters and the accumulator classes overlapping,
correctly predicting which targets would be easy and which hard.

**31. Did feature engineering improve every component?**
No, and reporting that honestly matters. Valve improved by +0.024 and
accumulator by +0.022 macro F1 — exactly the targets the dataset's own
documentation calls hard. The cooler got −0.001, because cooling efficiency
already separates its classes almost linearly, so a per-sensor mean solves it.
Claiming a uniform win would have been overstating the result.

**32. Three of your test scores are 100 %. Isn't that suspicious?**
It warrants scrutiny, which is why I did three checks. First, the original paper
on this dataset reports cooler and valve as "easy" targets with perfect
classification achieved. Second, the top features are physically correct —
CE/CP for the cooler, flow and power for the pump — so the model isn't keying on
an artefact. Third, the temporal-block CV shows the scores are optimistic by a
measurable amount, which I report. The near-ceiling results do limit how well the
models can be discriminated from one another, and that is stated as a limitation.

**33. What are the main limitations?**
No RUL; a single rig with one load programme and deliberately induced faults;
temporal correlation inflating random-split scores by up to ~0.08 macro F1;
degradation discretised into levels when reality is continuous; CE/CP/SE being
virtual channels that inherit the rig's own model; and impurity importance being
biased towards high-cardinality features.

**34. What does the dashboard do, and what does it deliberately not do?**
It loads the saved pipelines and scores a cycle — chosen from the dataset or
uploaded as a raw-sensor CSV — showing per-component status, probabilities and
the features driving the prediction. It **never retrains**. If it re-fitted a
scaler on whatever data it happened to hold, the classifier's inputs would be on
a different scale from training and every prediction would be silently wrong.

**35. What would deploying this in a real plant require?**
Validation on the target installation, since the models are trained on one rig.
A streaming feature-extraction service computing the same features per cycle.
Drift monitoring on the feature distributions, since sensor ageing shifts them.
Recalibration of the Normal/Degraded/Critical thresholds against that plant's
maintenance policy. And an alarm-handling process — a prediction nobody acts on
has no value.
