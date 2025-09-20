'''
2025-08-28
Author: Dan Schumacher
How to run:
   python ./src/data_managment/TimerBed/cleaning_step_1.py
'''
import os
import json
import sys
from typing import List, Union, Dict, Any
sys.path.append("./src")

# -------------------------------
# Subset-specific plotting/meta
# -------------------------------
SUBSET_META = {
    "CTU": {
        "dims": ["power_consumption"],
        "x_label": "Time",
        "y_label": "Power consumption",
        "title_template": "CTU Power Consumption: {idx}",
    },
    "ECG": {
        "dims": ["ecg_amplitude"],
        "x_label": "Time",
        "y_label": "ECG amplitude",
        "title_template": "ECG (single-lead): {idx}",
    },
    "EMG": {
        "dims": ["emg_amplitude"],
        "x_label": "Time",
        "y_label": "EMG amplitude",
        "title_template": "EMG: {idx}",
    },
    "HAR": {
        "dims": ["X", "Y", "Z"],
        "x_label": "Time",
        "y_label": "Acceleration",
        "title_template": "HAR Accelerometer (X,Y,Z): {idx}",
    },
    "TEE": {
        "dims": ["power_density"],
        "x_label": "Time",
        "y_label": "Power density",
        "title_template": "FORTE Power Density: {idx}",
    },
    # RCW included for completeness (skipped if .ts files don’t exist)
    "RCW": {
        "dims": ["audio_amplitude"],
        "x_label": "Time",
        "y_label": "Amplitude",
        "title_template": "RCW Audio: {idx}",
    },
}

def _to_bool(s: str) -> bool:
    return s.strip().lower() in {"true", "t", "1", "yes"}

def _maybe_int(s: str):
    s = s.strip()
    return int(s) if s.lstrip("-").isdigit() else s

def _parse_header(path: str):
    header = {
        "problem_name": None,
        "timestamps": None,
        "univariate": None,
        "equal_length": None,
        "series_length": None,
        "class_label": None,
        "label_values": None,
        "data_start_index": None,
    }
    with open(path, "r") as f:
        lines = f.readlines()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            tmp = line.lstrip("#").strip()
            if tmp.startswith("@"):
                line = tmp
        if not line.startswith("@"):
            continue
        low = line.lower()
        if low.startswith("@problemname"):
            header["problem_name"] = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else None
        elif low.startswith("@timestamps"):
            header["timestamps"] = _to_bool(line.split()[-1])
        elif low.startswith("@univariate"):
            header["univariate"] = _to_bool(line.split()[-1])
        elif low.startswith("@equallength"):
            header["equal_length"] = _to_bool(line.split()[-1])
        elif low.startswith("@serieslength"):
            header["series_length"] = int(line.split()[-1])
        elif low.startswith("@classlabel"):
            toks = line.split()
            header["class_label"] = _to_bool(toks[1]) if len(toks) > 1 else None
            header["label_values"] = toks[2:] if header["class_label"] else None
        elif low.startswith("@data"):
            header["data_start_index"] = i
            break
    return header, lines

def _trim_trailing_common_zeros(ts: Dict[str, List[float]], eps: float = 1e-12) -> Dict[str, List[float]]:
    last_sig = -1
    for dim_vals in ts.values():
        j = len(dim_vals) - 1
        while j >= 0 and abs(dim_vals[j]) <= eps:
            j -= 1
        if j > last_sig:
            last_sig = j
    if last_sig == -1:
        return ts
    new_len = last_sig + 1
    return {k: v[:new_len] for k, v in ts.items()}

def _build_dimension_map(subset_upper: str, n_dims: int) -> Dict[str, str]:
    dims_cfg = SUBSET_META.get(subset_upper, {}).get("dims", [])
    return {f"dim_{i}": (dims_cfg[i] if i < len(dims_cfg) else f"dim_{i}") for i in range(n_dims)}

def _plot_meta_for_row(subset_upper: str, idx: str) -> Dict[str, str]:
    meta = SUBSET_META.get(subset_upper, {})
    return {
        "x_label": meta.get("x_label", "Time"),
        "y_label": meta.get("y_label", "Value"),
        "title": (meta.get("title_template", "{idx}").format(idx=idx)),
    }

def read_ts_file(subset: str, split: str) -> Union[List[dict], None]:
    subset_upper = subset.upper()
    subset_lower = subset.lower()
    path = f"data/datasets/_raw_data/TimerBed/{subset_upper}/{subset_upper}_{split}.ts"
    if not os.path.exists(path):
        print(f"PATH DOES NOT EXIST: {path}\n")
        return None

    header, lines = _parse_header(path)
    if header["data_start_index"] is None:
        print(f"No @data section found in {path}")
        return None
    if header["univariate"] is None:
        print(f"@univariate not specified in header for {path}; cannot proceed safely.")
        return None

    univariate = header["univariate"]
    rows = []
    for raw in lines[header["data_start_index"] + 1 :]:
        line = raw.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(":")]
        if univariate:
            if len(parts) != 2:
                continue
            series_part, label_part = parts[0], parts[1]
            ts_vals = [float(x) for x in series_part.split(",") if x != ""]
            ts = {"dim_0": ts_vals}
            label = _maybe_int(label_part)
        else:
            if len(parts) < 2:
                continue
            label_part = parts[-1]
            dim_parts = parts[:-1]
            dim_dict = {}
            for dim_idx, dp in enumerate(dim_parts):
                vals = [float(x) for x in dp.split(",") if x != ""]
                dim_dict[f"dim_{dim_idx}"] = vals
            ts = dim_dict
            label = _maybe_int(label_part)

        ts = _trim_trailing_common_zeros(ts, eps=1e-12)

        idx = f"IDX_{len(rows)}"
        n_dims = len(ts)
        dim_map = _build_dimension_map(subset_upper, n_dims)
        plot_meta = _plot_meta_for_row(subset_upper, idx)

        rows.append({
            "idx": idx,
            "question": "PLACEHOLDER",
            "answer": "PLACEHOLDER",
            "label": label,
            "split": split.lower(),
            "dataset": "timerbed",
            "subset": subset_lower,
            "ts": ts,
            "task_type": "classification",
            "multivariate": not univariate,
            "dimension_map": dim_map,
            "x_label": plot_meta["x_label"],
            "y_label": plot_meta["y_label"],
            "title": plot_meta["title"],
        })

    if header["series_length"] is not None and len(rows) > 0:
        if univariate:
            bad = [r["idx"] for r in rows if len(r["ts"]["dim_0"]) != header["series_length"]]
        else:
            if header["equal_length"]:
                bad = [
                    r["idx"] for r in rows
                    if any(len(arr) != header["series_length"] for arr in r["ts"].values())
                ]
            else:
                bad = []
        if bad:
            print(f"WARNING: {subset_upper} {split}: {len(bad)} series differ from @seriesLength={header['series_length']} (e.g., {bad[:3]})")
    return rows

def main():
    splits = ["TRAIN", "TEST"]
    subsets = ["CTU", "ECG", "EMG", "HAR", "TEE", "RCW"]  # RCW included (skipped if absent)

    master_data: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}
    master_data.setdefault("timerbed", {})

    for subset in subsets:
        for split in splits:
            data = read_ts_file(subset, split)
            if not data:
                continue
            master_data["timerbed"].setdefault(subset.lower(), {})
            master_data["timerbed"][subset.lower()][split.lower()] = data

            print(f"{subset}: {split}")
            print(f"{len(data)} lines loaded")
            first = data[0]
            print(f"IDX: {first['idx']}  LABEL(raw): {first['label']}  "
                  f"TS len: {len(first['ts']['dim_0'])}  DIM MAP: {first['dimension_map']}")
            print()

    out_path = "./data/datasets/_raw_data/TimerBed/cleaning_step_1.json"
    with open(out_path, "w") as fo:
        json.dump(master_data, fo, indent=4)
    print(f"Wrote master to {out_path}")

if __name__ == "__main__":
    main()
