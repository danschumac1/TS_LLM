'''
2025-10-09
Author: Dan Schumacher
How to run:
   python ./src/method/imputation/miss_forest.py ./data/datasets/TSQA/dev.jsonl
'''
import os, re, sys, json, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def _first_bracketed_list(text: str):
    m = re.search(r"\[.*?\]", text)
    if not m:
        raise ValueError("No bracketed list found.")
    return m.group(0)

def parse_series_from_question(question: str) -> np.ndarray:
    """Parse the list from 'question', mapping 'X' -> np.nan."""
    lst = ast.literal_eval(_first_bracketed_list(question))
    out = []
    for v in lst:
        if isinstance(v, str) and v.strip().upper() == 'X':
            out.append(np.nan)
        else:
            out.append(float(v))
    return np.array(out, dtype=float)

def parse_truth_from_answer(answer: str | dict | None) -> np.ndarray | None:
    """Parse the full ground-truth list from 'answer' if present (string or dict)."""
    if answer is None:
        return None
    if isinstance(answer, dict) and "filled_series" in answer:
        return np.array(answer["filled_series"], dtype=float)
    if isinstance(answer, str):
        try:
            lst = ast.literal_eval(_first_bracketed_list(answer))
            return np.array([float(x) for x in lst], dtype=float)
        except Exception:
            return None
    return None


def main():
    if len(sys.argv) < 1+1:
        print("Usage: python ./src/method/imputation/miss_forest_jsonl_truth.py <input.jsonl>")
        sys.exit(1)

    in_path = sys.argv[1]
    with open(in_path, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    y_missing = parse_series_from_question(rec["question"])
    y_true = parse_truth_from_answer(rec.get("answer"))   # full series; may be None

    n = len(y_missing)
    nan_mask = np.isnan(y_missing)

    # Build a minimal 2D frame for MissForest (add time index as predictor)
    t = np.arange(n, dtype=float)
    df_missing = pd.DataFrame({"signal": y_missing, "t": t})

    imputer = MissForest(
        rgr=RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
        clf=RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        max_iter=5, early_stopping=True, verbose=0
    )
    df_imp = imputer.fit_transform(df_missing)
    y_imp = df_imp["signal"].to_numpy()

    # ---------- Plot ----------
    x = np.arange(n)
    plt.figure(figsize=(10, 4))

    # continuous dashed ground-truth line (if available); else dashed imputed line
    if y_true is not None:
        plt.plot(x, y_true, linestyle="--", label="true")
    else:
        plt.plot(x, y_imp, linestyle="--", label="imputed (full)")

    # red dots ONLY at missing positions (the points MissForest filled)
    plt.scatter(x[nan_mask], y_imp[nan_mask], s=70, label="imputed", color="red")

    plt.title("MissForest imputation (continuous dashed ground truth + red imputed points)")
    plt.xlabel("time index")
    plt.ylabel("value")
    plt.legend()

    os.makedirs("./images", exist_ok=True)
    out_img = "./images/missforest_1d_jsonl_truth.png"
    plt.savefig(out_img, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {out_img}")

    # also print series with imputed values filled
    print("Filled series:", y_imp.tolist())


if __name__ == "__main__":
    main()
