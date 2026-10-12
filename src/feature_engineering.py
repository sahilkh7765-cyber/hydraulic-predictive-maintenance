"""
feature_engineering.py
======================
The academic core of this project: turning 43,680 raw samples per operating
cycle into a compact, physically interpretable feature vector.

WHY FEATURE ENGINEERING IS THE POINT
    A cycle is 60 s of a repeated constant-load programme. Feeding the 43,680
    raw samples straight into a classifier is possible but bad practice here:
        * the raw dimensionality (43,680) dwarfs the sample count (2,205), so
          any model is in an extreme p >> n regime;
        * the sensors are sampled at three different rates, so column k of PS1
          and column k of TS1 refer to completely different instants;
        * the fault signatures are physical - a leaking pump changes the
          relationship between motor power and delivered flow, a degraded
          cooler changes a temperature difference - and those relationships are
          *between* channels, which no per-sample representation exposes.
    Summarising each channel with shape-describing statistics and then building
    cross-channel physical quantities compresses ~43.7k numbers into a few
    hundred, each of which an engineer can name and defend.

FEATURE BLOCKS PRODUCED (one row per cycle)
    A. statistical  - distribution shape of each channel within the cycle
    B. time-domain  - how the signal evolves during the cycle
    C. rolling      - local behaviour, window chosen per sampling rate
    D. frequency    - a small FFT summary, only for >= 10 Hz channels
    E. domain       - hydraulic relationships between channels

INPUT  : raw sensor matrices of shape (n_cycles, n_samples)
OUTPUT : a pandas DataFrame with one row per cycle and named float columns
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.ndimage import uniform_filter1d

from data_loader import (
    HIGH_RATE_SENSORS,
    RAW_DATA_DIR,
    SENSOR_INDEX,
    SENSOR_NAMES,
    SensorSpec,
    load_sensor_matrix,
)
from utils import print_header, print_kv, safe_divide, sanitize_matrix

# Feature-block names, used for reporting how many features each block created.
FEATURE_BLOCKS: tuple[str, ...] = ("statistical", "time_domain", "rolling", "frequency", "domain")


# ==========================================================================
# A. STATISTICAL FEATURES
# ==========================================================================
def statistical_features(matrix: np.ndarray, sensor: str) -> Dict[str, np.ndarray]:
    """Describe the within-cycle distribution of one sensor.

    WHAT: classical descriptive statistics computed across the time axis of
    every cycle, so each statistic becomes one column of the feature table.

    WHY: degradation changes the *distribution* of a signal long before it
    changes its average. A valve with switching lag spends more time in
    transition, which widens the pressure distribution (higher std, lower
    kurtosis) even when the mean pressure is unchanged. Skewness and kurtosis
    capture asymmetry and peakedness, which mean/std alone cannot express.

    Input : matrix of shape (n_cycles, n_samples); the sensor name for labels.
    Output: dict mapping feature name -> 1-D array of length n_cycles.
    """
    features: Dict[str, np.ndarray] = {}
    x = matrix.astype(np.float64, copy=False)

    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=1)
    minimum = x.min(axis=1)
    maximum = x.max(axis=1)
    q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=1)

    features[f"{sensor}_mean"] = mean
    features[f"{sensor}_median"] = q50
    features[f"{sensor}_std"] = std
    features[f"{sensor}_var"] = x.var(axis=1, ddof=1)
    features[f"{sensor}_min"] = minimum
    features[f"{sensor}_max"] = maximum
    features[f"{sensor}_range"] = maximum - minimum
    # RMS is the quadratic mean: for pressure and vibration it is proportional
    # to the energy content of the signal, not just its average level.
    features[f"{sensor}_rms"] = np.sqrt((x**2).mean(axis=1))
    features[f"{sensor}_mav"] = np.abs(x).mean(axis=1)
    # Skewness and kurtosis are undefined for a perfectly constant signal
    # (the variance in the denominator is zero). A few cycles do hold a
    # channel constant for the whole 60 s, so the resulting NaN is caught and
    # replaced by 0 - the value a symmetric, non-peaked distribution would
    # have - rather than allowed to propagate into the feature table.
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = scipy_stats.skew(x, axis=1, bias=False)
        kurtosis = scipy_stats.kurtosis(x, axis=1, bias=False)
    features[f"{sensor}_skew"] = np.nan_to_num(skew, nan=0.0)
    features[f"{sensor}_kurtosis"] = np.nan_to_num(kurtosis, nan=0.0)
    features[f"{sensor}_q25"] = q25
    features[f"{sensor}_q75"] = q75
    features[f"{sensor}_iqr"] = q75 - q25
    # Coefficient of variation is only meaningful for strictly positive
    # quantities on a ratio scale. Temperature in degC has an arbitrary zero,
    # so CV is not generated for temperature channels - a CV of a Celsius
    # reading has no physical interpretation.
    spec = SENSOR_INDEX.get(sensor)
    if spec is not None and spec.group not in {"temperature"}:
        features[f"{sensor}_cv"] = safe_divide(std, np.abs(mean))
    return features


# ==========================================================================
# B. TIME-DOMAIN FEATURES
# ==========================================================================
def time_domain_features(matrix: np.ndarray, sensor: str) -> Dict[str, np.ndarray]:
    """Describe how the signal moves through the cycle.

    WHAT: first-difference statistics, signal energy, linear trend and a peak
    count.

    WHY: two cycles can share an identical histogram yet differ completely in
    dynamics. An accumulator that has lost pre-charge pressure cannot damp
    pressure transients, so the rig shows faster and larger pressure changes -
    visible in the first difference, invisible in the mean. The linear slope
    captures drift within the cycle, which is how a cooler losing efficiency
    shows up in the temperature channels.

    Input : matrix of shape (n_cycles, n_samples); sensor name for labels.
    Output: dict of feature name -> array of length n_cycles.
    """
    features: Dict[str, np.ndarray] = {}
    x = matrix.astype(np.float64, copy=False)
    n_cycles, n_samples = x.shape

    maximum = x.max(axis=1)
    minimum = x.min(axis=1)
    features[f"{sensor}_peak_to_peak"] = maximum - minimum
    # Peak amplitude = largest excursion from zero, the usual "peak" in
    # vibration and pressure analysis.
    features[f"{sensor}_peak_amplitude"] = np.abs(x).max(axis=1)
    # Signal energy, normalised by length so that channels with different
    # sampling rates remain comparable in magnitude.
    features[f"{sensor}_energy"] = (x**2).sum(axis=1) / n_samples

    diff = np.diff(x, axis=1)
    features[f"{sensor}_mean_abs_change"] = np.abs(diff).mean(axis=1)
    features[f"{sensor}_max_abs_change"] = np.abs(diff).max(axis=1)
    features[f"{sensor}_std_diff"] = diff.std(axis=1, ddof=1)
    features[f"{sensor}_mean_diff"] = diff.mean(axis=1)

    # Linear trend: slope of a least-squares straight line fitted across the
    # cycle. Computed in closed form (cov(t, x) / var(t)) because a per-row
    # polyfit over 2205 x 6000 points would be needlessly slow.
    t = np.arange(n_samples, dtype=np.float64)
    t_centered = t - t.mean()
    denominator = (t_centered**2).sum()
    slope = (x - x.mean(axis=1, keepdims=True)) @ t_centered / denominator
    features[f"{sensor}_slope"] = slope

    # Number of zero crossings of the mean-centred signal: a cheap, robust
    # proxy for how often the signal oscillates around its own level. For a
    # switching valve this tracks the number of load changes actually executed.
    centered = x - x.mean(axis=1, keepdims=True)
    sign_changes = np.diff(np.signbit(centered).astype(np.int8), axis=1)
    features[f"{sensor}_zero_crossings"] = np.count_nonzero(sign_changes, axis=1).astype(
        np.float64
    )

    # Peak count is only meaningful when the signal is sampled fast enough to
    # resolve individual peaks; at 1 Hz (60 points) it would be noise.
    spec = SENSOR_INDEX.get(sensor)
    if spec is not None and spec.sampling_rate_hz >= 10:
        # A local maximum that also exceeds mean + 1 std, so that ripple on a
        # flat plateau is not counted as a peak.
        threshold = (x.mean(axis=1) + x.std(axis=1, ddof=1))[:, None]
        is_local_max = (x[:, 1:-1] > x[:, :-2]) & (x[:, 1:-1] > x[:, 2:])
        features[f"{sensor}_n_peaks"] = np.count_nonzero(
            is_local_max & (x[:, 1:-1] > threshold), axis=1
        ).astype(np.float64)
    return features


# ==========================================================================
# C. ROLLING / LOCAL FEATURES
# ==========================================================================
def rolling_window_for(spec: SensorSpec, seconds: float = 1.0) -> int:
    """Choose a rolling window length from the sensor's own sampling rate.

    WHY: applying one fixed window (say 50 samples) to every channel would mean
    0.5 s for a 100 Hz pressure signal but 50 s - almost the whole cycle - for a
    1 Hz temperature signal. Deriving the window from the documented sampling
    rate keeps the *physical* window constant at `seconds` for every sensor.

    Input : the sensor specification and a physical window length in seconds.
    Output: window length in samples (at least 3).
    """
    return max(3, int(round(spec.sampling_rate_hz * seconds)))


def rolling_features(
    matrix: np.ndarray, sensor: str, seconds: float = 1.0
) -> Dict[str, np.ndarray]:
    """Summarise local (within-window) behaviour of a fast channel.

    WHAT: a 1-second moving mean and moving standard deviation are computed,
    then those two derived series are themselves summarised.

    WHY: global statistics average away short events. A valve with severe
    switching lag produces a brief pressure disturbance lasting a fraction of a
    second; across a 60 s cycle it barely moves the global std, but it clearly
    raises the *maximum local* std. `max_rolling_std` is therefore a local
    instability detector, while the spread of the rolling mean describes how
    much the operating point moves during the cycle.

    IMPLEMENTATION NOTE: `uniform_filter1d` computes the moving average in
    O(n) per row and operates directly on the 2-D array, which keeps the whole
    2205 x 6000 pressure matrix tractable. The moving variance uses the
    identity Var = E[x^2] - E[x]^2 over the same filter.

    Input : matrix (n_cycles, n_samples); sensor name; physical window length.
    Output: dict of feature name -> array of length n_cycles. Empty dict for
            sensors too slow for a meaningful window.
    """
    spec = SENSOR_INDEX.get(sensor)
    if spec is None or spec.sampling_rate_hz < 10:
        # 1 Hz channels give 60 points per cycle; a rolling window over them
        # would just reproduce the global statistics, so none are created.
        return {}

    window = rolling_window_for(spec, seconds)
    x = matrix.astype(np.float64, copy=False)

    roll_mean = uniform_filter1d(x, size=window, axis=1, mode="nearest")
    roll_mean_sq = uniform_filter1d(x**2, size=window, axis=1, mode="nearest")
    # Clipped at zero because floating-point cancellation can make the
    # difference marginally negative on a perfectly constant segment.
    roll_var = np.clip(roll_mean_sq - roll_mean**2, 0.0, None)
    roll_std = np.sqrt(roll_var)

    features: Dict[str, np.ndarray] = {
        f"{sensor}_rollmean_std": roll_mean.std(axis=1, ddof=1),
        f"{sensor}_rollmean_range": roll_mean.max(axis=1) - roll_mean.min(axis=1),
        f"{sensor}_rollmean_min": roll_mean.min(axis=1),
        f"{sensor}_rollmean_max": roll_mean.max(axis=1),
        f"{sensor}_rollstd_mean": roll_std.mean(axis=1),
        f"{sensor}_rollstd_max": roll_std.max(axis=1),
    }
    return features


# ==========================================================================
# D. FREQUENCY-DOMAIN FEATURES
# ==========================================================================
def frequency_features(matrix: np.ndarray, sensor: str, n_bands: int = 3) -> Dict[str, np.ndarray]:
    """Extract a deliberately small FFT summary of a fast channel.

    WHAT: dominant frequency, total spectral energy, spectral centroid and the
    relative energy in `n_bands` frequency bands.

    WHY: pressure and motor-power signals contain periodic content from the
    pump and from the load programme. Mechanical degradation redistributes
    energy across frequency - for example a leaking pump alters the pulsation
    pattern - which time-domain statistics cannot see. The spectral centroid is
    the energy-weighted mean frequency, i.e. whether the signal's "centre of
    mass" has moved up or down the spectrum.

    WHY ONLY A FEW FEATURES: a full spectrum of a 6000-point signal has 3001
    bins. Turning each bin into a feature would add 21,000 columns for the
    seven fast channels alone and reintroduce exactly the p >> n problem that
    feature engineering is meant to solve. Four to six interpretable spectral
    descriptors per channel carry the useful information at 0.03 % of the width.

    NOTE: the mean is removed before the transform, otherwise the DC bin
    dominates the spectrum and the "dominant frequency" would always be 0 Hz.

    Input : matrix (n_cycles, n_samples), sensor name, number of energy bands.
    Output: dict of feature name -> array of length n_cycles. Empty for
            sensors sampled below 10 Hz.
    """
    spec = SENSOR_INDEX.get(sensor)
    if spec is None or spec.sampling_rate_hz < 10:
        # A 1 Hz / 60-sample signal has a Nyquist limit of 0.5 Hz and a
        # resolution of ~0.017 Hz; an FFT of it describes almost nothing.
        return {}

    x = matrix.astype(np.float64, copy=False)
    n_samples = x.shape[1]
    x = x - x.mean(axis=1, keepdims=True)  # remove DC component

    spectrum = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / spec.sampling_rate_hz)

    power = spectrum**2
    total_power = power.sum(axis=1)

    features: Dict[str, np.ndarray] = {}
    # Dominant frequency: bin with the largest magnitude, excluding the DC bin.
    dominant_idx = np.argmax(spectrum[:, 1:], axis=1) + 1
    features[f"{sensor}_dom_freq"] = freqs[dominant_idx]
    features[f"{sensor}_dom_freq_mag"] = spectrum[np.arange(len(x)), dominant_idx]
    # log1p keeps spectral energy on a comparable scale to the other features;
    # raw values span several orders of magnitude between channels.
    features[f"{sensor}_spectral_energy"] = np.log1p(total_power)
    features[f"{sensor}_spectral_centroid"] = safe_divide(
        (freqs[None, :] * power).sum(axis=1), total_power
    )

    # Relative band energies. The band edges are LOG-spaced between the
    # fundamental (1 / 60 s) and the Nyquist limit rather than equally spaced.
    # Reason: the load programme repeats on a tens-of-seconds scale, so almost
    # all signal energy sits below 1 Hz. With linear bands over 0-50 Hz the
    # first band would absorb ~100 % of the energy and the remaining bands
    # would be constant zeros - useless features. Log spacing gives the
    # low-frequency region, where the hydraulic dynamics actually live, the
    # resolution it deserves, and adapts automatically to each sensor's own
    # Nyquist limit.
    f_min, f_max = freqs[1], freqs[-1]
    edges_hz = np.logspace(np.log10(f_min), np.log10(f_max), n_bands + 1)
    edge_idx = np.searchsorted(freqs, edges_hz)
    edge_idx[0] = max(edge_idx[0], 1)  # never include the DC bin
    edge_idx[-1] = len(freqs)
    for band in range(n_bands):
        lo, hi = edge_idx[band], max(edge_idx[band + 1], edge_idx[band] + 1)
        band_power = power[:, lo:hi].sum(axis=1)
        features[f"{sensor}_band{band + 1}_energy_ratio"] = safe_divide(
            band_power, total_power
        )
    return features


# ==========================================================================
# E. HYDRAULIC DOMAIN-SPECIFIC FEATURES
# ==========================================================================
def domain_features(cycle_stats: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Build cross-channel features with a hydraulic interpretation.

    WHAT: pressure differentials, flow balance, hydraulic vs electrical power,
    temperature spreads and cooling relationships.

    WHY: this is the block that encodes engineering knowledge rather than
    generic signal processing. The dataset documentation gives each channel's
    physical quantity and unit, and these features are built strictly from that
    - no assumption is made about the rig's internal piping topology, which the
    documentation does not specify. Every feature below is defensible from the
    units alone:

      * Pressure differential (bar - bar): a pressure drop between two points
        of the circuit. Restrictions, valve lag and leakage all change how
        pressure is distributed across the circuit, so differentials react to
        faults that leave individual absolute pressures nearly unchanged.
      * Flow balance FS1 - FS2 and FS1 / FS2 (l/min): internal pump leakage
        means fluid returns to tank without doing work, so delivered flow no
        longer matches the flow measured elsewhere in the circuit. This is the
        most directly physical leakage indicator available.
      * Hydraulic power p*Q/600 (bar*l/min -> kW): the useful power actually
        delivered to the fluid.
      * Overall efficiency = hydraulic power / electrical power (EPS1 in W):
        a leaking or worn pump consumes electrical power without converting it
        into useful hydraulic power, so this ratio falls.
      * Specific energy = EPS1 / FS1 (W per l/min): electrical cost of moving
        one unit of flow; rises with internal leakage.
      * Temperature differences (K): the oil temperature spread across the
        circuit. A cooler losing efficiency cannot remove the same heat, which
        narrows or shifts these differences.
      * Cooling power per Kelvin, CP / dT: an approximate thermal conductance
        of the cooling circuit, which drops as the cooler fouls.
      * Pressure and flow variability ratios: dimensionless instability
        indicators. A degraded accumulator damps less, so relative pressure
        variability rises.

    Input : `cycle_stats[sensor]` = dict of already-computed per-cycle
            aggregates for that sensor ("mean", "std", "max", "min", "range").
            Reusing them avoids a second pass over the raw matrices.
    Output: dict of feature name -> array of length n_cycles.
    """
    features: Dict[str, np.ndarray] = {}

    def stat(sensor: str, name: str) -> Optional[np.ndarray]:
        """Fetch a previously computed aggregate, or None if unavailable."""
        return cycle_stats.get(sensor, {}).get(name)

    # -- Pressure differentials -------------------------------------------
    # All ordered pairs of pressure sensors. With six channels this is 15
    # differentials: a small, complete and fully interpretable set.
    pressure_sensors = [s for s in SENSOR_NAMES if SENSOR_INDEX[s].group == "pressure"]
    for i, first in enumerate(pressure_sensors):
        for second in pressure_sensors[i + 1 :]:
            mean_a, mean_b = stat(first, "mean"), stat(second, "mean")
            if mean_a is None or mean_b is None:
                continue
            features[f"dp_{first}_{second}_mean"] = mean_a - mean_b
            # A safe ratio: how one pressure level scales with another. Guarded
            # division returns 0 where the denominator is effectively zero.
            features[f"pratio_{first}_{second}"] = safe_divide(mean_a, mean_b)

    # Standard deviation of a differential cannot be derived from per-channel
    # statistics (it needs the joint signal), so only the differential of the
    # standard deviations is formed here: a measure of which of the two points
    # in the circuit fluctuates more.
    for i, first in enumerate(pressure_sensors):
        for second in pressure_sensors[i + 1 :]:
            std_a, std_b = stat(first, "std"), stat(second, "std")
            if std_a is not None and std_b is not None:
                features[f"dstd_{first}_{second}"] = std_a - std_b

    # -- Pressure variability ---------------------------------------------
    # Relative variability: std normalised by mean level. Dimensionless, so it
    # is comparable across pressure levels and across cycles.
    for sensor in pressure_sensors:
        std, mean = stat(sensor, "std"), stat(sensor, "mean")
        if std is not None and mean is not None:
            features[f"pvar_rel_{sensor}"] = safe_divide(std, np.abs(mean))

    # -- Flow balance ------------------------------------------------------
    fs1_mean, fs2_mean = stat("FS1", "mean"), stat("FS2", "mean")
    if fs1_mean is not None and fs2_mean is not None:
        # Absolute and relative mismatch between the two flow measurements.
        features["flow_diff_FS1_FS2"] = fs1_mean - fs2_mean
        features["flow_ratio_FS1_FS2"] = safe_divide(fs1_mean, fs2_mean)
        features["flow_imbalance_rel"] = safe_divide(
            fs1_mean - fs2_mean, np.abs(fs1_mean) + np.abs(fs2_mean)
        )
    for sensor in ("FS1", "FS2"):
        std, mean = stat(sensor, "std"), stat(sensor, "mean")
        if std is not None and mean is not None:
            features[f"flow_var_rel_{sensor}"] = safe_divide(std, np.abs(mean))

    # -- Power / flow / pressure relationships -----------------------------
    eps_mean = stat("EPS1", "mean")
    if eps_mean is not None and fs1_mean is not None:
        # Specific electrical energy: watts spent per litre/min delivered.
        features["specific_power_EPS1_per_FS1"] = safe_divide(eps_mean, fs1_mean)
    if eps_mean is not None and fs2_mean is not None:
        features["specific_power_EPS1_per_FS2"] = safe_divide(eps_mean, fs2_mean)

    for pressure in pressure_sensors:
        p_mean = stat(pressure, "mean")
        if p_mean is None or fs1_mean is None:
            continue
        # Hydraulic power in kW: P = p[bar] * Q[l/min] / 600.
        hydraulic_kw = p_mean * fs1_mean / 600.0
        features[f"hyd_power_kW_{pressure}_FS1"] = hydraulic_kw
        if eps_mean is not None:
            # Overall electro-hydraulic efficiency. EPS1 is in W, hydraulic
            # power in kW, hence the factor 1000.
            features[f"efficiency_{pressure}_FS1"] = safe_divide(
                hydraulic_kw * 1000.0, eps_mean
            )
        # Hydraulic resistance proxy: pressure needed per unit of flow.
        features[f"resistance_{pressure}_FS1"] = safe_divide(p_mean, fs1_mean)

    # -- Temperature relationships ----------------------------------------
    temperature_sensors = [
        s for s in SENSOR_NAMES if SENSOR_INDEX[s].group == "temperature"
    ]
    temperature_diffs: Dict[str, np.ndarray] = {}
    for i, first in enumerate(temperature_sensors):
        for second in temperature_sensors[i + 1 :]:
            t_a, t_b = stat(first, "mean"), stat(second, "mean")
            if t_a is None or t_b is None:
                continue
            delta = t_a - t_b
            features[f"dT_{first}_{second}"] = delta
            temperature_diffs[f"{first}_{second}"] = delta

    if temperature_sensors:
        means = [stat(s, "mean") for s in temperature_sensors]
        means = [m for m in means if m is not None]
        if means:
            stacked = np.vstack(means)
            # Oil temperature spread across the circuit and its overall level.
            features["T_spread"] = stacked.max(axis=0) - stacked.min(axis=0)
            features["T_mean_all"] = stacked.mean(axis=0)
        rise = [stat(s, "range") for s in temperature_sensors]
        rise = [r for r in rise if r is not None]
        if rise:
            # How much the oil heats up within a single 60 s cycle.
            features["T_rise_max"] = np.vstack(rise).max(axis=0)

    # -- Cooling relationships --------------------------------------------
    cp_mean, ce_mean, se_mean = stat("CP", "mean"), stat("CE", "mean"), stat("SE", "mean")
    if cp_mean is not None and temperature_diffs:
        # Thermal conductance proxy: cooling power delivered per Kelvin of
        # available temperature difference. Falls as the cooler degrades.
        for name, delta in temperature_diffs.items():
            features[f"cooling_conductance_{name}"] = safe_divide(cp_mean, np.abs(delta))
    if cp_mean is not None and ce_mean is not None:
        features["cooling_power_per_efficiency"] = safe_divide(cp_mean, ce_mean)
    if ce_mean is not None and se_mean is not None:
        # Relationship between the cooling circuit and the overall efficiency
        # factor: both are supplied as virtual channels by the rig.
        features["CE_SE_ratio"] = safe_divide(ce_mean, se_mean)
    if cp_mean is not None and eps_mean is not None:
        # Fraction of consumed electrical power that ends up being removed as
        # heat by the cooling circuit (CP in kW, EPS1 in W).
        features["cooling_share_of_power"] = safe_divide(cp_mean * 1000.0, eps_mean)

    # -- Vibration relationships ------------------------------------------
    vs_mean, vs_std = stat("VS1", "mean"), stat("VS1", "std")
    if vs_mean is not None and eps_mean is not None:
        # Vibration per unit of electrical power: separates "vibrating because
        # it is working hard" from "vibrating because something is wrong".
        features["vibration_per_power"] = safe_divide(vs_mean, eps_mean) * 1000.0
    if vs_std is not None and vs_mean is not None:
        features["vibration_rel_var"] = safe_divide(vs_std, np.abs(vs_mean))

    return features


# ==========================================================================
# ORCHESTRATION
# ==========================================================================
def build_feature_table(
    raw_dir=RAW_DATA_DIR,
    sensors: Sequence[str] = SENSOR_NAMES,
    rolling_seconds: float = 1.0,
    n_bands: int = 3,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Run the full feature-engineering pipeline over the raw dataset.

    PROCESSING STRATEGY: sensors are loaded, summarised and released one at a
    time. Holding all seven 100 Hz matrices simultaneously would need roughly
    370 MB in float32; streaming keeps the peak near a single sensor's 53 MB.

    Input : raw data directory, list of sensors, rolling window in seconds,
            number of FFT bands.
    Output: (feature_table, block_map) where
            * feature_table is a DataFrame with a `cycle_id` column followed by
              all engineered features, one row per cycle;
            * block_map maps each block name ("statistical", "time_domain",
              "rolling", "frequency", "domain") to its feature names, which is
              what the FEATURE ENGINEERING SUMMARY reports.
    """
    columns: Dict[str, np.ndarray] = {}
    block_map: Dict[str, List[str]] = {block: [] for block in FEATURE_BLOCKS}
    # Per-sensor aggregates kept for the cross-channel domain block.
    cycle_stats: Dict[str, Dict[str, np.ndarray]] = {}
    n_cycles: Optional[int] = None

    if verbose:
        print_header("FEATURE EXTRACTION")

    for sensor in sensors:
        matrix = load_sensor_matrix(sensor, raw_dir)
        if n_cycles is None:
            n_cycles = matrix.shape[0]
        elif matrix.shape[0] != n_cycles:
            raise ValueError(
                f"{sensor} has {matrix.shape[0]} cycles but previous sensors had "
                f"{n_cycles}; the dataset files are inconsistent."
            )

        stats_block = statistical_features(matrix, sensor)
        time_block = time_domain_features(matrix, sensor)
        roll_block = rolling_features(matrix, sensor, seconds=rolling_seconds)
        freq_block = frequency_features(matrix, sensor, n_bands=n_bands)

        for block_name, block in (
            ("statistical", stats_block),
            ("time_domain", time_block),
            ("rolling", roll_block),
            ("frequency", freq_block),
        ):
            for name, values in block.items():
                columns[name] = values
                block_map[block_name].append(name)

        # Cache the aggregates the domain block needs.
        cycle_stats[sensor] = {
            "mean": stats_block[f"{sensor}_mean"],
            "std": stats_block[f"{sensor}_std"],
            "min": stats_block[f"{sensor}_min"],
            "max": stats_block[f"{sensor}_max"],
            "range": stats_block[f"{sensor}_range"],
        }

        if verbose:
            spec = SENSOR_INDEX[sensor]
            print(
                f"  {sensor:<5} {spec.sampling_rate_hz:>3} Hz  "
                f"{matrix.shape[1]:>5} samples/cycle  ->  "
                f"stat {len(stats_block):>2} | time {len(time_block):>2} | "
                f"roll {len(roll_block):>2} | freq {len(freq_block):>2}"
            )
        del matrix  # release ~53 MB before loading the next sensor

    # Cross-channel hydraulic features, computed once from the cached
    # per-sensor aggregates.
    dom_block = domain_features(cycle_stats)
    for name, values in dom_block.items():
        columns[name] = values
        block_map["domain"].append(name)
    if verbose:
        print(f"  domain features (cross-channel hydraulic): {len(dom_block)}")

    feature_table = pd.DataFrame(
        {name: sanitize_matrix(values) for name, values in columns.items()}
    )
    feature_table.insert(0, "cycle_id", np.arange(len(feature_table), dtype=int))
    return feature_table, block_map


def print_feature_summary(
    block_map: Dict[str, List[str]],
    total_raw_dimensions: int,
    counts_after: Optional[Dict[str, int]] = None,
) -> None:
    """Print the FEATURE ENGINEERING SUMMARY block required by the brief.

    Input : block_map from `build_feature_table`, the raw dimensionality, and
            optionally the feature counts surviving each later stage.
    Output: formatted console output only.
    """
    print_header("FEATURE ENGINEERING SUMMARY")
    total = sum(len(names) for names in block_map.values())
    print_kv("Raw dimensions per cycle", f"{total_raw_dimensions:,}")
    print_kv("Statistical features", len(block_map.get("statistical", [])))
    print_kv("Time-domain features", len(block_map.get("time_domain", [])))
    print_kv("Rolling/local features", len(block_map.get("rolling", [])))
    print_kv("Frequency-domain features", len(block_map.get("frequency", [])))
    print_kv("Hydraulic domain features", len(block_map.get("domain", [])))
    print_kv("Total engineered features", total)
    if total_raw_dimensions:
        print_kv(
            "Compression vs raw representation",
            f"{total_raw_dimensions / max(total, 1):.1f}x fewer columns",
        )
    if counts_after:
        for label, count in counts_after.items():
            print_kv(label, count)


def extract_features_from_cycle(
    cycle: Dict[str, np.ndarray],
    rolling_seconds: float = 1.0,
    n_bands: int = 3,
) -> pd.DataFrame:
    """Extract the same feature vector for a SINGLE unseen cycle.

    WHY THIS EXISTS: the Streamlit dashboard must be able to accept one raw
    cycle (uploaded as a CSV) and score it. It has to produce *exactly* the
    same columns, in the same order, as the training pipeline - otherwise the
    saved scaler and feature selector would be applied to the wrong columns.
    Reusing the same four block functions guarantees that.

    Input : dict mapping sensor name -> 1-D array of that sensor's samples for
            one cycle (length must match the documented samples per cycle).
    Output: single-row DataFrame of engineered features.
    Raises: ValueError if a sensor's sample count contradicts its specification.
    """
    columns: Dict[str, np.ndarray] = {}
    cycle_stats: Dict[str, Dict[str, np.ndarray]] = {}

    for sensor, values in cycle.items():
        if sensor not in SENSOR_INDEX:
            continue
        spec = SENSOR_INDEX[sensor]
        array = np.asarray(values, dtype=np.float64).ravel()
        if array.size != spec.n_samples:
            raise ValueError(
                f"{sensor}: got {array.size} samples but {spec.n_samples} are "
                f"required for a 60 s cycle at {spec.sampling_rate_hz} Hz."
            )
        matrix = array[None, :]  # promote to (1, n_samples)

        stats_block = statistical_features(matrix, sensor)
        columns.update(stats_block)
        columns.update(time_domain_features(matrix, sensor))
        columns.update(rolling_features(matrix, sensor, seconds=rolling_seconds))
        columns.update(frequency_features(matrix, sensor, n_bands=n_bands))

        cycle_stats[sensor] = {
            "mean": stats_block[f"{sensor}_mean"],
            "std": stats_block[f"{sensor}_std"],
            "min": stats_block[f"{sensor}_min"],
            "max": stats_block[f"{sensor}_max"],
            "range": stats_block[f"{sensor}_range"],
        }

    columns.update(domain_features(cycle_stats))
    return pd.DataFrame({name: sanitize_matrix(value) for name, value in columns.items()})


def build_baseline_table(
    raw_dir=RAW_DATA_DIR, sensors: Sequence[str] = SENSOR_NAMES, verbose: bool = True
) -> pd.DataFrame:
    """Build the EXPERIMENT 1 baseline representation.

    WHAT: the arithmetic mean of each sensor over the cycle - 17 columns, one
    per channel.

    WHY THIS IS THE RIGHT BASELINE: it is the minimum transformation that makes
    the data rectangular at all. The three sampling rates make the raw signals
    unusable as a single table, so *some* aggregation is unavoidable; taking
    the mean is the most obvious choice an engineer would make without doing
    any feature engineering. Comparing against it isolates the contribution of
    the engineered blocks rather than the contribution of "having a table".

    Input : raw data directory and list of sensors.
    Output: DataFrame with cycle_id and one mean column per sensor.
    """
    columns: Dict[str, np.ndarray] = {}
    for sensor in sensors:
        matrix = load_sensor_matrix(sensor, raw_dir)
        columns[f"{sensor}_mean"] = matrix.astype(np.float64).mean(axis=1)
        del matrix
    table = pd.DataFrame(columns)
    table.insert(0, "cycle_id", np.arange(len(table), dtype=int))
    if verbose:
        print_kv("Baseline representation", f"{table.shape[1] - 1} features (per-sensor mean)")
    return table
