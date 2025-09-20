'''
2025-08-28
Author: Dan Schumacher
How to run:
   python ./src/data_managment/TimerBed/cleaning_step_2.py
'''
import sys
sys.path.append("./src")
import os
import json
import random
from typing import Dict, Any, List

# -------------------------------
# Label ↔ name mappings per subset
# -------------------------------
LABEL_MAPPING = {
    "RCW": {0: "No right whale call", 1: "Right whale call present"},
    "TEE": {
        0: "CG Positive Initial Return Stroke",
        1: "IR Negative Initial Return Stroke",
        2: "SR Subsequent Negative Return Stroke",
        3: "I Impulsive Event",
        4: "I2 Impulsive Event Pair (TIPP)",
        5: "KM Gradual Intra-Cloud Stroke",
        6: "O Off-record",
    },
    "ECG": {
        0: "Normal sinus rhythm",
        1: "Atrial fibrillation",
        2: "Other rhythm",
        3: "Too noisy to classify",
    },
    "EMG": {0: "Healthy (normal)", 1: "Myopathy", 2: "Neuropathy"},
    "CTU": {1: "Desktop", 2: "Laptop"},  # note ids start at 1
    "HAR": {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING",
    },
}

# -------------------------------
# Task descriptions per subset
# -------------------------------
TASK_DESCRIPTION = {
    "CTU": (
        "Play as a computer energy consumption analysis expert: determine whether this computer is a desktop or a laptop "
        "based on the 24-hour power consumption time series."
    ),
    "ECG": "As a cardiologist, classify the patient's heart rhythm from a single-lead ECG segment.",
    "EMG": "As an EMG analysis expert, determine the subject type based on the EMG time series.",
    "HAR": (
        "As a human activity recognition expert, determine the activity based on the tri-axial accelerometer series (x, y, z) over time."
    ),
    "TEE": (
        "Based on the FORTE satellite power-density time series, select the transient electromagnetic event that best matches. "
        "There are seven event types with distinct temporal patterns (e.g., sharp turn-on + noise, slow ramp-up to attachment + spike + exponential decay, paired impulsive peaks, gradual intra-cloud increase, off-record)."
    ),
    "RCW": (
        "Play the role of a marine biology expert: decide whether the recording contains a North Atlantic right whale call "
        "(e.g., an up-call with a rising contour ~0.5–1.5 s, typically ~50–300 Hz)."
    ),
}

# -------------------------------
# Build question strings
# -------------------------------
import string
def _letters(n: int):
    letters = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(string.ascii_uppercase[rem])
    return "".join(reversed(letters))

def _sort_key_for_label_id(k):
    try:
        return (0, int(k))
    except (TypeError, ValueError):
        return (1, str(k))

def build_question_text(subset_key_upper: str) -> str:
    task = TASK_DESCRIPTION.get(subset_key_upper, "").strip()
    labels = LABEL_MAPPING.get(subset_key_upper, {})
    sorted_items = sorted(labels.items(), key=lambda kv: _sort_key_for_label_id(kv[0]))
    label_texts = [v for _, v in sorted_items]
    options_lines = [f"[{_letters(i+1)}] {opt}" for i, opt in enumerate(label_texts)]
    options_block = "Your options are:\n\t" + "\n\t".join(options_lines)
    return f"{task} {options_block}" if task else options_block

QUESTION_BY_SUBSET: Dict[str, str] = {
    sub: build_question_text(sub) for sub in (set(TASK_DESCRIPTION) & set(LABEL_MAPPING))
}

# -------------------------------
# I/O helpers
# -------------------------------
IN_PATH = "./data/datasets/_raw_data/TimerBed/cleaning_step_1.json"
OUT_ROOT = "./data/datasets/TimerBed"
RNG_SEED = 1337

def read_master(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------------------------
# Enrichment: fill question + answer + options
# -------------------------------
def enrich_row(row: Dict[str, Any], subset_key_upper: str) -> Dict[str, Any]:
    labels = LABEL_MAPPING.get(subset_key_upper, {})
    sorted_items = sorted(labels.items(), key=lambda kv: _sort_key_for_label_id(kv[0]))
    options = { _letters(i+1): v for i, (_, v) in enumerate(sorted_items) }
    id_to_pos = { k: i for i, (k, _) in enumerate(sorted_items) }

    # raw label (could be int or string)
    label_val = row.get("label")
    try:
        raw_id = int(label_val)
    except Exception:
        raw_id = label_val

    answer_text = labels.get(raw_id, str(label_val))
    gold_letter = None
    if raw_id in id_to_pos:
        gold_letter = _letters(id_to_pos[raw_id] + 1)
    elif isinstance(label_val, str) and len(label_val) == 1 and label_val.isalpha():
        gold_letter = label_val.upper()

    row["label"] = gold_letter if gold_letter else str(label_val)
    row["answer"] = answer_text
    row["options"] = options

    question_text = QUESTION_BY_SUBSET.get(subset_key_upper) or build_question_text(subset_key_upper)
    row["question"] = question_text

    # Ensure dataset/subset exist
    row.setdefault("dataset", "timerbed")
    row.setdefault("subset", row.get("subset", "").lower())
    return row

# -------------------------------
# Main
# -------------------------------
def main():
    master = read_master(IN_PATH)
    root = master["timerbed"] if "timerbed" in master else master

    rng = random.Random(RNG_SEED)
    write_count = 0

    for subset_lower, splits in root.items():
        subset_upper = subset_lower.upper()
        for split_name, rows in splits.items():
            if not rows:
                continue
            enriched = [enrich_row(dict(r), subset_upper) for r in rows]
            rng.shuffle(enriched)

            out_dir = os.path.join(OUT_ROOT, subset_upper)
            out_path = os.path.join(out_dir, f"{split_name.lower()}.jsonl")
            write_jsonl(out_path, enriched)
            write_count += len(enriched)

            # brief preview
            r0 = enriched[0]
            preview = {
                "idx": r0.get("idx"),
                "dataset": r0.get("dataset"),
                "subset": r0.get("subset"),
                "split": r0.get("split"),
                "label": r0.get("label"),
                "answer": r0.get("answer"),
                "options": r0.get("options"),
                "dimension_map": r0.get("dimension_map"),
                "x_label": r0.get("x_label"),
                "y_label": r0.get("y_label"),
                "title": r0.get("title"),
            }
            print(f"Wrote {len(enriched)} rows → {out_path}")
            print("Preview:", json.dumps(preview, ensure_ascii=False))
            print()

    print(f"Total written rows across all subsets/splits: {write_count}")

if __name__ == "__main__":
    main()
