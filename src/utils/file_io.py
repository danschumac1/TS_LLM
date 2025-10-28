import json
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

def load_jsonl(file_path):
    """Load a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def yield_jsonl(file_path, max_rows=None):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to a JSON Lines file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_csv(file_path):
    """Load a CSV file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def save_csv(data, file_path):
    """Save data to a CSV file."""
    if not data:
        return
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def read_txt_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the plain .txt split where each line is:
      <label> v1 v2 ... v720
    (Scientific notation possible; whitespace separated.)
    Returns:
      X: float32 array, shape (N, L)
      y: int64 array, shape (N,)
    """
    X_list, y_list = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            lab = int(float(toks[0]))
            vals = [float(v) for v in toks[1:]]
            X_list.append(np.array(vals, dtype=np.float32))
            y_list.append(lab)
    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y

def shuffle_in_unison(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic row-wise shuffle of X and y together."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]

def save_to_npy_pair(X: np.ndarray, y: np.ndarray, prefix: str):
    """Saves X and y as NPY files with consistent names."""
    np.save(prefix + ".npy", X.astype(np.float32))
    np.save(prefix + "_labels.npy", y.astype(np.int64))
    print(f"Saved X {X.shape} → {prefix}.npy | y {y.shape} → {prefix}_labels.npy")




# --------------------------- FILE HELPERS -----------------------------
def list_aiff_files(root: str) -> List[str]:
    paths = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".aiff", ".aif")):
                paths.append(os.path.join(dp, fn))
    paths.sort()
    return paths

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def load_labels_csv(csv_path: str) -> Dict[str, int]:
    """
    CSV rows: filename,label (header optional).
    Keys are file stems. Labels may be {0,1} or {1,2}; we map {1,2}→{0,1}.
    """
    mp: Dict[str, int] = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing labels CSV: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        sniff = f.read(2048)
        f.seek(0)
        has_header = any(h in sniff.lower() for h in ["filename", "file", "label"])
        reader = csv.reader(f)
        if has_header:
            _ = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            fn, lab = row[0].strip(), row[1].strip()
            st = stem(fn)
            y_raw = int(float(lab))
            if y_raw in (0, 1):
                y = y_raw
            elif y_raw in (1, 2):
                y = y_raw - 1  # map 1→0, 2→1
            else:
                raise ValueError(f"Unexpected label {lab} for {fn}; expected 0/1 or 1/2.")
            mp[st] = y
    return mp

# ----------------------- AUDIO / ARRAY HELPERS -----------------------
def _to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)

def _normalize(x: np.ndarray, dtype:type) -> np.ndarray:
    x = x.astype(dtype, copy=False)
    return x

def _pad_trim_1d(x: np.ndarray, L: int, pad_value: float = 0.0) -> np.ndarray:
    n = x.shape[0]
    if n == L:
        return x
    if n > L:
        start = (n - L) // 2
        return x[start:start+L]
    pad_total = L - n
    left = pad_total // 2
    right = pad_total - left
    return np.pad(x, (left, right), mode="constant", constant_values=pad_value)

def read_aiff(path: str, dtype:type) -> np.ndarray:
    x, _sr = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=dtype)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return _normalize(x, dtype)

def load_aiff_as_row(path: str, target_len:int, dtype:type) -> np.ndarray:
    x = read_aiff(path, dtype)
    x = _to_mono(x)
    x = _pad_trim_1d(x, target_len, 0.0).astype(dtype, copy=False)
    return x[None, :]

# ----------------------- DATASET BUILDING ----------------------------
def stratified_train_test_split(y: np.ndarray, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        k = int(round(train_frac * len(cls_idx)))
        train_idx.append(cls_idx[:k])
        test_idx.append(cls_idx[k:])
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    test_idx  = np.concatenate(test_idx)  if test_idx  else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx

def save_split(prefix: str, X: np.ndarray, y: np.ndarray, dtype:type):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    np.save(prefix + ".npy", X.astype(dtype))
    np.save(prefix + "_labels.npy", y.astype(np.int64))
    print(f"Saved X {X.shape} → {prefix}.npy")
    print(f"Saved y {y.shape} → {prefix}_labels.npy")
