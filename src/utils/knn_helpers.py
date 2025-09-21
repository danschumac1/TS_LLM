from typing import Any, Dict, List, Tuple
import numpy as np


def _sort_key_dim(k: str) -> int:
    try:
        return int(str(k).split("_")[1])
    except Exception:
        return 10**9

def ts_dict_to_np(ts: Any) -> np.ndarray:
    """row['ts'] -> (T, D). Univariate -> D=1."""
    if isinstance(ts, dict):
        dims = [np.asarray(ts[d], dtype="float32") for d in sorted(ts.keys(), key=_sort_key_dim)]
        return np.stack(dims, axis=1)  # (T, D)
    else:
        return np.asarray(ts, dtype="float32")[:, None]

def downsample_td(arr: np.ndarray, step: int) -> np.ndarray:
    if step is None or step <= 1:
        return arr
    return arr[::step]

def build_panel(rows: List[Dict[str, Any]], *, downsample_step: int) -> List[np.ndarray]:
    return [downsample_td(ts_dict_to_np(r["ts"]), downsample_step) for r in rows]

def pad_stack(X_list: List[np.ndarray]) -> np.ndarray:
    """Pad along time to common length; stack -> (N, T, D)."""
    T = max(x.shape[0] for x in X_list)
    D = X_list[0].shape[1]
    out = np.empty((len(X_list), T, D), dtype="float32")
    for i, x in enumerate(X_list):
        t = x.shape[0]
        if t < T:
            out[i] = np.pad(x, ((0, T - t), (0, 0)), mode="edge")
        else:
            out[i] = x
    return out

def make_class_lists(options: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Returns (letters_in_order, classnames_in_same_order)."""
    letters = sorted(options.keys())  # ['A','B',...]
    classes = [options[a] for a in letters]
    return letters, classes

def human_to_letter(pred_str: str, letters: List[str], classes: List[str]) -> str:
    try:
        idx = classes.index(pred_str)
        return letters[idx]
    except ValueError:
        return "?"
