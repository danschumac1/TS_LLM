import re
from enum import Enum

class EvalType(Enum):
    TRUE_FALSE = "true_false"
    MULTIPLE_CHOICE = "multiple_choice"
    FORECASTING_MSE = "forecasting_mse"
    IMPUTATION_MSE = "imputation_mse"
    ACC = "acc"                    # reserved (LLM judge), not used below
    ANOMALY = "anomaly"
    CLASSIFICATION = "classification"
    OTHER_OPEN_ENDED_QA = "other_open_ended_qa"  # for open-ended QA tasks

_MC_CHOICES = re.compile(r'(?<!\w)([a-e])[\)\.]')  # matches a), b), c. etc (lowercase before/after normalize)

def detect_eval_type(line:dict) -> EvalType:
    """
    Decide EvalType from task metadata + question text.
    This mirrors your current heuristics but runs once at build time.
    """
    pt = (line['parent_task'] or "").lower()
    tt = (line['task_type'] or "").lower()
    q  = (line['question'] or "").lower()

    if pt in {"anomaly_detection"}:
        return EvalType.ANOMALY
    if pt in {"classification"}:
        return EvalType.CLASSIFICATION
    if pt in {"forecasting_imputation1", "forecasting_imputation2"}:
        if tt == "forecasting":
            return EvalType.FORECASTING_MSE
        elif tt == "imputation":
            return EvalType.IMPUTATION_MSE
        else:
            raise ValueError(f"Unknown task_type under forecasting_imputation*: {tt}")

    open_ended_qa_count = 0
    if pt in {"open_ended_qa"}:
        # T/F?
        if ("true" in q) and ("false" in q):
            return EvalType.TRUE_FALSE
        # MC?
        # count unique a)-e) (or a. b. etc)
        letters = set(_MC_CHOICES.findall(q))
        if len(letters) >= 3:
            return EvalType.MULTIPLE_CHOICE
        
        # otherwise, assume open-ended QA
        open_ended_qa_count += 1
        return EvalType.OTHER_OPEN_ENDED_QA
    print("[INFO] number of open-ended QA tasks:", open_ended_qa_count)

    print(f"[WARNING] Unknown parent_task: {pt} with task_type: {tt} and question: {q}")
