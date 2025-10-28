'''
2025-10-10
Author: Dan Schumacher
How to run:
   python ./src/data_managment/MONSTER/load_data.py
'''
import os
import sys
import json
import shutil
import numpy as np
from collections import defaultdict
from huggingface_hub import hf_hub_download

sys.path.append("./src")

SUBSETS = [
    # Audio
    # "AudioMNIST","AudioMNIST-DS","CornellWhaleChallenge","FruitFlies",
    # "InsectSound","MosquitoSound","WhaleSounds",
    # # Satellite Image Time Series
    # "LakeIce","S2Agri","S2Agri-10pc","TimeSen2Crop","Tiselac",
    # # EEG
    # "CrowdSourced","DreamerA","DreamerV","STEW",
    # # Human Activity Recognition
    # "Opportunity","PAMAP2","Skoda","UCIActivity","USCActivity","WISDM","WISDM2",
    # Counts
    "Pedestrian","Traffic",
    # Other
    "FordChallenge","LenDB",
]

RAW_ROOT = "./data/datasets/_raw_data/MONSTER"
SEED = 42


def try_download(repo_id: str, candidates: list[str]) -> str:
    """
    Try a list of filenames in order; return local HF cache path of the first that exists.
    Raises FileNotFoundError if none are found.
    """
    last_err = None
    for fn in candidates:
        try:
            return hf_hub_download(repo_id=repo_id, filename=fn, repo_type="dataset")
        except Exception as e:
            last_err = e
    raise FileNotFoundError(f"None of {candidates} found in {repo_id}. Last error: {last_err}")


def copy_if_missing(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"  Copied -> {dst}")
    return dst


def load_labels(path: str):
    """Supports .npy and .npz labels (uses common keys or first array)."""
    if path.lower().endswith(".npy"):
        return np.load(path)
    elif path.lower().endswith(".npz"):
        z = np.load(path)
        for k in ("labels", "y", "arr_0"):
            if k in z:
                return z[k]
        # fallback: first key
        return z[list(z.keys())[0]]
    else:
        raise ValueError(f"Unsupported label file: {path}")


def stratified_split(y, train_frac=0.8, seed=SEED):
    """Return (train_idx, test_idx). Tries sklearn; falls back to simple per-class split."""
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_frac, random_state=seed)
        (tr, te), = sss.split(np.zeros_like(y), y)
        return tr, te
    except Exception:
        pass

    # Manual approx
    rng = np.random.default_rng(seed)
    labels = np.asarray(y)
    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[lab].append(i)
    tr, te = [], []
    for _, idxs in groups.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        ntr = int(round(train_frac * len(idxs)))
        tr.extend(idxs[:ntr]); te.extend(idxs[ntr:])
    tr = np.array(tr); te = np.array(te)
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te


def write_jsonl(path: str, X, Y, indices):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in indices:
            x = X[i].reshape(-1).tolist()  # flatten all remaining dims
            y = Y[i]
            # clean numpy scalars
            if hasattr(y, "item"):
                try: y = y.item()
                except Exception: pass
            if isinstance(y, (np.integer,)):
                y = int(y)
            f.write(json.dumps({"series": x, "label": y}) + "\n")
    print(f"  Wrote {len(indices)} -> {path}")


def process_subset(subset: str):
    print(f"\n=== {subset} ===")
    repo_id = f"monster-monash/{subset}"

    # Raw local dir per subset
    local_dir = os.path.join(RAW_ROOT, subset)
    os.makedirs(local_dir, exist_ok=True)

    # Filenames we’ll try (case variants + npz for labels)
    x_candidates = [f"{subset}_X.npy", f"{subset}_x.npy"]
    y_candidates = [f"{subset}_Y.npy", f"{subset}_y.npy", f"{subset}_labels.npy", f"{subset}_labels.npz"]

    # Download (from HF cache)
    try:
        hf_x = try_download(repo_id, x_candidates)
        hf_y = try_download(repo_id, y_candidates)
    except FileNotFoundError as e:
        print(f"  ! Skipping: {e}")
        return

    # Normalize names locally
    local_x = os.path.join(local_dir, f"{subset}_X.npy")
    local_y = os.path.join(local_dir, os.path.basename(hf_y))  # keep original ext for labels

    copy_if_missing(hf_x, local_x)
    copy_if_missing(hf_y, local_y)

    # Load arrays
    try:
        X = np.load(local_x, mmap_mode="r")
        Y = load_labels(local_y)
    except Exception as e:
        print(f"  ! Failed to load arrays: {e}")
        return

    if len(X) != len(Y):
        print(f"  ! Length mismatch X:{len(X)} vs Y:{len(Y)}. Skipping.")
        return

    print(f"  Shapes -> X:{X.shape} | Y:{Y.shape} | N={len(Y)}")

    # Split (80/20)
    train_idx, test_idx = stratified_split(Y, train_frac=0.8, seed=SEED)

    # Write JSONL (full sizes)
    train_path = os.path.join(local_dir, "train.jsonl")
    test_path  = os.path.join(local_dir, "test.jsonl")
    write_jsonl(train_path, X, Y, train_idx)
    write_jsonl(test_path,  X, Y, test_idx)


def main():
    for i, subset in enumerate(SUBSETS):
        print(f"{i}: {subset}")
        process_subset(subset)


if __name__ == "__main__":
    main()
