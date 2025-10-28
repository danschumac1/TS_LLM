"""
2025-10-22
Author: Dan Schumacher

Convert TimerBed-HAR JSONL to numeric .npy:
  X: (N, 3, L) from ts.dim_0/1/2 (X,Y,Z)
  y: (N,) mapped from label A..F -> 0..5 (options order)

Usage:
  python ./src/data_managment/jsonl_to_npy.py \
    --jsonl_path ./data/datasets/TimerBed/HAR/train.jsonl \
    --npy_path   ./data/datasets/classification/TimeMixer/har/train.npy \
    --save_labels \
    --channel_first \
    --pad_to max

  python ./src/data_managment/jsonl_to_npy.py \
    --jsonl_path ./data/datasets/TimerBed/HAR/test.jsonl \
    --npy_path   ./data/datasets/classification/TimeMixer/har/test.npy \
    --save_labels \
    --channel_first \
    --pad_to max
"""

import sys; sys.path.append("./src")
import os, json, argparse
import numpy as np

# Optional: use your util if available
def _load_jsonl_fallback(p):
    recs = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

try:
    from utils.file_io import load_jsonl as _load_jsonl
except Exception:
    _load_jsonl = _load_jsonl_fallback

# Label map according to options order [A..F]
LABEL_MAP = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5}

def _pad_or_trim_1d(x, L, pad_value=0.0):
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == L:
        return x
    if x.size > L:
        return x[:L]
    out = np.full((L,), pad_value, dtype=np.float32)
    out[:x.size] = x
    return out

def _extract_xyz_ts(rec):
    """
    rec['ts'] is a dict like {'dim_0': [...], 'dim_1': [...], 'dim_2': [...]}
    Use dimension_map if present to order channels as X,Y,Z.
    """
    ts = rec.get("ts")
    if not isinstance(ts, dict):
        raise ValueError("Expected rec['ts'] to be a dict with dim_0/dim_1/dim_2.")

    # Prefer dimension_map order (X,Y,Z) if present
    dim_map = rec.get("dimension_map", {})
    # Build a list of (dim_key, axis_name)
    items = list(ts.items())  # [('dim_0',[...]), ('dim_1',[...]), ...]
    if dim_map:
        # sort by desired axis order X,Y,Z if available; else fall back to dim_0.. order
        order = {"X":0,"Y":1,"Z":2}
        items.sort(key=lambda kv: order.get(dim_map.get(kv[0], ""), 99))
    else:
        # default dim_0, dim_1, dim_2
        items.sort(key=lambda kv: kv[0])

    chans = [np.asarray(v, dtype=np.float32).ravel() for _, v in items]
    return chans  # [x, y, z] (variable length)

def save_as_numeric_npy(jsonl_path, npy_path, channel_first=True, pad_to="max", pad_value=0.0, save_labels=False):
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    recs = _load_jsonl(jsonl_path)
    if not recs:
        raise ValueError(f"No records in {jsonl_path}")

    # Extract channels and labels
    all_chans = []  # list of [x,y,z]
    labels = []
    lengths = []
    for r in recs:
        xyz = _extract_xyz_ts(r)         # list of 3 arrays
        if len(xyz) != 3:
            raise ValueError("Expected 3 axes in 'ts' (X,Y,Z).")
        Ls = [len(c) for c in xyz]
        if not (Ls[0] == Ls[1] == Ls[2]):
            raise ValueError(f"Channel lengths differ: {Ls}")
        L = Ls[0]
        lengths.append(L)
        all_chans.append(xyz)

        # label: prefer 'label' (A..F); if missing, try from 'answer'
        lab = r.get("label")
        if lab is None and "answer" in r:
            # map textual answer back to A..F if options present
            ans = r["answer"]
            opts = r.get("options", {})
            inv = {v: k for k, v in opts.items()} if isinstance(opts, dict) else {}
            lab = inv.get(ans)
        labels.append(-1 if lab is None else LABEL_MAP.get(lab, -1))

    # Decide target length
    if pad_to == "max" or pad_to is None:
        L = max(lengths)
    elif pad_to == "min":
        L = min(lengths)
    else:
        # numeric
        try:
            L = int(pad_to)
            assert L > 0
        except Exception:
            raise ValueError("--pad_to must be 'max', 'min', or a positive integer")

    # Build X array
    N = len(all_chans)
    X = np.zeros((N, 3, L), dtype=np.float32) if channel_first else np.zeros((N, L, 3), dtype=np.float32)
    for i, (x,y,z) in enumerate(all_chans):
        x = _pad_or_trim_1d(x, L, pad_value)
        y = _pad_or_trim_1d(y, L, pad_value)
        z = _pad_or_trim_1d(z, L, pad_value)
        if channel_first:
            X[i, 0, :] = x
            X[i, 1, :] = y
            X[i, 2, :] = z
        else:
            X[i, :, 0] = x
            X[i, :, 1] = y
            X[i, :, 2] = z

    np.save(npy_path, X)
    msg = f"Saved X {X.shape} to {npy_path}"

    if save_labels:
        y = np.asarray(labels, dtype=np.int64)
        y_path = os.path.splitext(npy_path)[0] + "_labels.npy"
        np.save(y_path, y)
        msg += f"; y {y.shape} to {y_path}"
    print(msg)

def parse_args():
    p = argparse.ArgumentParser(description="Convert TimerBed HAR JSONL to numeric NPY.")
    p.add_argument("--jsonl_path", required=True, type=str)
    p.add_argument("--npy_path",   required=True, type=str)
    p.add_argument("--channel_first", action="store_true", help="Outputs (N,3,L) if set, else (N,L,3)")
    p.add_argument("--pad_to", type=str, default="max", help="'max' (default), 'min', or integer length")
    p.add_argument("--pad_value", type=float, default=0.0)
    p.add_argument("--save_labels", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    save_as_numeric_npy(
        args.jsonl_path,
        args.npy_path,
        channel_first=args.channel_first,
        pad_to=args.pad_to,
        pad_value=args.pad_value,
        save_labels=args.save_labels,
    )
