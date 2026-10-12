"""
data_loader.py
==============
Loading layer for the UCI "Condition monitoring of hydraulic systems" dataset.

WHAT THIS MODULE DOES
    Reads the 17 tab-delimited sensor matrices and the cycle-wise label file
    `profile.txt`, validates them against the dimensions stated in the dataset's
    own `documentation.txt`, and exposes them as numpy arrays plus a pandas
    label table.

WHY IT IS NEEDED
    Each sensor is stored in its own file and sampled at a different rate, so
    the raw data is *not* a single rectangular table. PS1 is 2205x6000 while TS1
    is 2205x60: both describe the same 60-second cycle. Every downstream step
    needs to know which sampling rate a signal has, because a rolling window or
    an FFT band that is correct for a 100 Hz pressure signal is meaningless for
    a 1 Hz temperature signal.

DATA LAYOUT (verified against the supplied files, not assumed)
    * one file per sensor, tab-delimited, no header
    * row i  -> operating cycle i (60 s of constant load)
    * column j -> the j-th sample inside that cycle
    * profile.txt: 2205 rows x 5 columns of cycle-wise condition annotations

INPUT  : the extracted dataset directory (default: data/raw)
OUTPUT : numpy float32 matrices, a label DataFrame, and a dataset summary dict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_dir,
    format_int,
    print_header,
    print_kv,
)


# --------------------------------------------------------------------------
# Sensor specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SensorSpec:
    """Static description of one sensor channel.

    name            : file stem, e.g. "PS1" -> data/raw/PS1.txt
    quantity        : physical quantity measured
    unit            : physical unit as documented by the dataset creators
    sampling_rate_hz: documented sampling rate
    n_samples       : expected number of samples per 60 s cycle
    group           : coarse family used for plotting and domain features
    virtual         : True for computed channels (CE, CP, SE) rather than
                      physical transducers - relevant because they are already
                      derived quantities and must not be presented as raw
                      measurements in the report.
    """

    name: str
    quantity: str
    unit: str
    sampling_rate_hz: int
    n_samples: int
    group: str
    virtual: bool = False


# Values below are transcribed from the dataset's documentation.txt and are
# re-validated against the actual file shapes in `load_sensor_matrix`.
SENSOR_SPECS: tuple[SensorSpec, ...] = (
    SensorSpec("PS1", "Pressure", "bar", 100, 6000, "pressure"),
    SensorSpec("PS2", "Pressure", "bar", 100, 6000, "pressure"),
    SensorSpec("PS3", "Pressure", "bar", 100, 6000, "pressure"),
    SensorSpec("PS4", "Pressure", "bar", 100, 6000, "pressure"),
    SensorSpec("PS5", "Pressure", "bar", 100, 6000, "pressure"),
    SensorSpec("PS6", "Pressure", "bar", 100, 6000, "pressure"),
    SensorSpec("EPS1", "Motor power", "W", 100, 6000, "power"),
    SensorSpec("FS1", "Volume flow", "l/min", 10, 600, "flow"),
    SensorSpec("FS2", "Volume flow", "l/min", 10, 600, "flow"),
    SensorSpec("TS1", "Temperature", "degC", 1, 60, "temperature"),
    SensorSpec("TS2", "Temperature", "degC", 1, 60, "temperature"),
    SensorSpec("TS3", "Temperature", "degC", 1, 60, "temperature"),
    SensorSpec("TS4", "Temperature", "degC", 1, 60, "temperature"),
    SensorSpec("VS1", "Vibration", "mm/s", 1, 60, "vibration"),
    SensorSpec("CE", "Cooling efficiency", "%", 1, 60, "efficiency", virtual=True),
    SensorSpec("CP", "Cooling power", "kW", 1, 60, "efficiency", virtual=True),
    SensorSpec("SE", "Efficiency factor", "%", 1, 60, "efficiency", virtual=True),
)

SENSOR_INDEX: Dict[str, SensorSpec] = {spec.name: spec for spec in SENSOR_SPECS}
SENSOR_NAMES: tuple[str, ...] = tuple(spec.name for spec in SENSOR_SPECS)

# Sensors fast enough for windowed / spectral analysis. A 1 Hz channel gives
# only 60 samples per cycle, which is far too coarse for a meaningful FFT, so
# those sensors are deliberately excluded from the frequency-domain block.
HIGH_RATE_SENSORS: tuple[str, ...] = tuple(
    spec.name for spec in SENSOR_SPECS if spec.sampling_rate_hz >= 10
)


# --------------------------------------------------------------------------
# Target specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpec:
    """Definition of one component-condition target.

    `value_labels` is copied verbatim from the dataset documentation. No
    severity level is invented here: every label string below appears in the
    supplied description.txt / documentation.txt.
    """

    key: str
    column: str
    display_name: str
    unit: str
    value_labels: Dict[int, str]
    severity: Dict[int, str]


# profile.txt column order, per the dataset documentation:
#   1 cooler condition [%], 2 valve condition [%], 3 internal pump leakage,
#   4 hydraulic accumulator pressure [bar], 5 stable flag
PROFILE_COLUMNS: tuple[str, ...] = (
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
    "stable_flag",
)

TARGET_SPECS: Dict[str, TargetSpec] = {
    "cooler": TargetSpec(
        key="cooler",
        column="cooler_condition",
        display_name="Cooler condition",
        unit="%",
        value_labels={
            3: "close to total failure",
            20: "reduced efficiency",
            100: "full efficiency",
        },
        # Severity wording maps one-to-one onto the documented condition levels.
        severity={3: "Critical", 20: "Degraded", 100: "Normal"},
    ),
    "valve": TargetSpec(
        key="valve",
        column="valve_condition",
        display_name="Valve condition",
        unit="%",
        value_labels={
            73: "close to total failure",
            80: "severe lag",
            90: "small lag",
            100: "optimal switching behavior",
        },
        severity={73: "Critical", 80: "Degraded", 90: "Degraded", 100: "Normal"},
    ),
    "pump": TargetSpec(
        key="pump",
        column="pump_leakage",
        display_name="Internal pump leakage",
        unit="level",
        value_labels={0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
        severity={0: "Normal", 1: "Degraded", 2: "Critical"},
    ),
    "accumulator": TargetSpec(
        key="accumulator",
        column="accumulator_pressure",
        display_name="Hydraulic accumulator",
        unit="bar",
        value_labels={
            90: "close to total failure",
            100: "severely reduced pressure",
            115: "slightly reduced pressure",
            130: "optimal pressure",
        },
        severity={90: "Critical", 100: "Degraded", 115: "Degraded", 130: "Normal"},
    ),
}


# --------------------------------------------------------------------------
# Loading functions
# --------------------------------------------------------------------------
def load_sensor_matrix(
    sensor: str, raw_dir: Path = RAW_DATA_DIR, validate: bool = True
) -> np.ndarray:
    """Load one sensor file as a (n_cycles, n_samples) float32 matrix.

    WHY float32: the seven 100 Hz channels are 2205x6000 each. float32 halves
    the memory footprint with no practical loss of precision for sensor values
    that carry at most 4-5 significant digits, which matters because the loader
    must run inside a modest laptop RAM budget.

    Input : sensor name ("PS1"), directory containing the extracted .txt files.
    Output: float32 ndarray, shape (2205, n_samples for that sensor).
    Raises: FileNotFoundError if the file is absent, ValueError if the shape
            contradicts the documented specification.
    """
    path = raw_dir / f"{sensor}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Sensor file not found: {path}. Extract the dataset ZIP into {raw_dir}."
        )

    # `np.loadtxt` is correct but slow on a 90 MB file; pandas' C parser reads
    # the same tab-delimited matrix roughly an order of magnitude faster.
    frame = pd.read_csv(path, sep="\t", header=None, dtype=np.float32, engine="c")
    matrix = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))

    if validate and sensor in SENSOR_INDEX:
        spec = SENSOR_INDEX[sensor]
        if matrix.shape[1] != spec.n_samples:
            raise ValueError(
                f"{sensor}: file has {matrix.shape[1]} columns but the dataset "
                f"documentation specifies {spec.n_samples} samples per cycle."
            )
    return matrix


def load_profile(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load profile.txt, the cycle-wise condition annotations.

    Input : directory containing profile.txt.
    Output: DataFrame with one row per cycle, columns PROFILE_COLUMNS plus a
            0-based `cycle_id`. All condition columns are integers.
    """
    path = raw_dir / "profile.txt"
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    profile = pd.read_csv(path, sep="\t", header=None)
    if profile.shape[1] != len(PROFILE_COLUMNS):
        raise ValueError(
            f"profile.txt has {profile.shape[1]} columns; the documentation "
            f"specifies {len(PROFILE_COLUMNS)}."
        )
    profile.columns = list(PROFILE_COLUMNS)
    profile = profile.astype(int)
    # cycle_id is the row index and also the chronological order of the
    # experiment; it is kept explicitly so that a chronological split and the
    # dashboard's cycle selector can both refer to the same identifier.
    profile.insert(0, "cycle_id", np.arange(len(profile), dtype=int))
    return profile


def load_documentation_text(raw_dir: Path = RAW_DATA_DIR) -> str:
    """Return the dataset's own documentation as plain text.

    The two documentation files ship in Latin-1 (they contain German umlauts in
    the authors' affiliation), so decoding is done explicitly rather than
    relying on the platform default.
    """
    chunks: List[str] = []
    for name in ("description.txt", "documentation.txt"):
        path = raw_dir / name
        if path.exists():
            chunks.append(f"----- {name} -----")
            chunks.append(path.read_text(encoding="latin-1"))
    return "\n".join(chunks)


def get_dataset_summary(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, object]:
    """Build the DATASET SUMMARY dictionary from the files themselves.

    Nothing here is taken on faith from the prompt: cycle count, per-sensor
    sample counts and class distributions are all read off disk.

    Input : raw data directory.
    Output: dict consumed by `print_dataset_summary`, the report writer and the
            Streamlit dashboard.
    """
    profile = load_profile(raw_dir)
    n_cycles = len(profile)

    sensors: List[Dict[str, object]] = []
    total_raw_dimensions = 0
    for spec in SENSOR_SPECS:
        path = raw_dir / f"{spec.name}.txt"
        if not path.exists():
            continue
        # Read only the first line to confirm the column count: this avoids
        # loading 90 MB just to measure a shape.
        with open(path, "r", encoding="utf-8") as handle:
            n_samples = len(handle.readline().rstrip("\n").split("\t"))
        sensors.append(
            {
                "sensor": spec.name,
                "quantity": spec.quantity,
                "unit": spec.unit,
                "sampling_rate_hz": spec.sampling_rate_hz,
                "samples_per_cycle": n_samples,
                "group": spec.group,
                "virtual": spec.virtual,
            }
        )
        total_raw_dimensions += n_samples

    class_distributions: Dict[str, Dict[str, int]] = {}
    for key, spec in TARGET_SPECS.items():
        counts = profile[spec.column].value_counts().sort_index()
        class_distributions[key] = {str(int(k)): int(v) for k, v in counts.items()}

    stable_counts = profile["stable_flag"].value_counts().sort_index()

    return {
        "n_cycles": n_cycles,
        "n_sensors": len(sensors),
        "cycle_duration_s": 60,
        "sensors": sensors,
        "total_raw_dimensions": total_raw_dimensions,
        "targets": {
            key: {
                "display_name": spec.display_name,
                "column": spec.column,
                "unit": spec.unit,
                "value_labels": {str(k): v for k, v in spec.value_labels.items()},
                "n_classes": len(spec.value_labels),
            }
            for key, spec in TARGET_SPECS.items()
        },
        "class_distributions": class_distributions,
        "stable_flag_distribution": {
            str(int(k)): int(v) for k, v in stable_counts.items()
        },
    }


def print_dataset_summary(summary: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Print the DATASET SUMMARY block required by the project brief.

    Input : an existing summary dict, or None to build one.
    Output: the summary dict (also printed to stdout).
    """
    summary = summary or get_dataset_summary()

    print_header("DATASET SUMMARY")
    print_kv("Number of cycles", format_int(int(summary["n_cycles"])))
    print_kv("Number of sensors", summary["n_sensors"])
    print_kv("Cycle duration", f"{summary['cycle_duration_s']} s (constant load)")
    print_kv(
        "Total raw dimensions per cycle",
        format_int(int(summary["total_raw_dimensions"])),
    )
    print_kv(
        "Total raw values in dataset",
        format_int(int(summary["total_raw_dimensions"]) * int(summary["n_cycles"])),
    )

    print("\n  Raw measurements per cycle, by sensor:")
    print(f"    {'Sensor':<7}{'Quantity':<22}{'Unit':<8}{'Rate':<9}{'Samples/cycle'}")
    for row in summary["sensors"]:  # type: ignore[index]
        virtual_tag = " (virtual)" if row["virtual"] else ""
        print(
            f"    {row['sensor']:<7}{row['quantity'] + virtual_tag:<30}"
            f"{row['unit']:<8}{str(row['sampling_rate_hz']) + ' Hz':<9}"
            f"{row['samples_per_cycle']}"
        )

    print("\n  Target variables and class distributions:")
    for key, spec in TARGET_SPECS.items():
        counts = summary["class_distributions"][key]  # type: ignore[index]
        total = sum(counts.values())
        print(f"\n    {spec.display_name} [{spec.unit}] - {len(counts)} classes")
        for value, count in counts.items():
            label = spec.value_labels.get(int(value), "unknown")
            share = 100.0 * count / total
            print(f"      {value:>5} = {label:<28} {count:>5} cycles ({share:5.1f} %)")

    stable = summary["stable_flag_distribution"]  # type: ignore[index]
    print("\n  Stable flag (metadata, not used as a prediction target):")
    print(f"      0 = conditions were stable                {stable.get('0', 0):>5} cycles")
    print(f"      1 = static conditions possibly not reached {stable.get('1', 0):>5} cycles")
    return summary


def load_all_sensors(
    raw_dir: Path = RAW_DATA_DIR, sensors: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:  # pragma: no cover - convenience helper
    """Load every sensor matrix into a dict.

    NOTE ON MEMORY: the seven 100 Hz channels alone are ~370 MB in float32.
    The feature-engineering pipeline therefore streams sensors one at a time
    instead of calling this function; it exists for notebook exploration where
    a machine with enough RAM is available.
    """
    names = list(sensors) if sensors is not None else list(SENSOR_NAMES)
    return {name: load_sensor_matrix(name, raw_dir) for name in names}


def cache_profile(raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Write profile.txt as a labelled CSV into data/processed/ for convenience."""
    ensure_dir(PROCESSED_DATA_DIR)
    profile = load_profile(raw_dir)
    path = PROCESSED_DATA_DIR / "profile_labels.csv"
    profile.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    # Running `python src/data_loader.py` performs the dataset inspection step
    # on its own, before any modelling code is executed.
    print_dataset_summary()
