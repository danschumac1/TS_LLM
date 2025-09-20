import re
from typing import List, Tuple
from tqdm import tqdm
from utils.logging_utils import MasterLogger
import string
import re
from enum import Enum
from utils.prompter import OpenAIPrompter

def is_true_false(line: dict) -> bool:
    logger = MasterLogger.get_instance()
    """Check if the question contains both 'true' and 'false' (case-insensitive)."""
    q = line['question'].lower()
    is_tf = "true" in q and "false" in q
    if is_tf:
        logger.info(f"Detected True/False question: {line['question']}")
    return is_tf

def is_multiple_choice(line: dict, min_choices: int = 3) -> bool:
    logger = MasterLogger.get_instance()
    q = line['question'].lower()
    letters = set(re.findall(r'(?<!\w)([a-e])[\)\.]', q))
    mc = len(letters) >= min_choices
    if mc:
        logger.info(f"Detected Multiple Choice with choices: {sorted(letters)} in {line['question']}")
    return mc

def extract_tf(text: str) -> str:
    """Extract 'true' or 'false' from a string. Raises if neither is present."""
    logger = MasterLogger.get_instance()
    text = text.lower()
    for val in ["true", "false"]:
        if val in text:
            logger.info(f"Extracted '{val}' from text: {text}")
            return val
    raise ValueError("Text must contain either 'true' or 'false'.")

def eval_tf(line: dict) -> int:
    """Evaluate True/False prediction."""
    logger = MasterLogger.get_instance()
    evaluation = int(extract_tf(line["GT"]) == extract_tf(line["PRED"]))
    logger.info(f"Evaluated True/False: {line['GT']} vs {line['PRED']} = {evaluation}")
    return evaluation

def extract_choice(text: str, valid_choices: str = "abcde") -> str:
    """Extract multiple choice answer from text (e.g., 'a)', 'b)')."""
    logger = MasterLogger.get_instance()
    text = text.lower()
    for choice in valid_choices:
        if f"{choice})" in text:
            logger.info(f"Extracted choice '{choice}' from text: {text}")
            return choice
    logger.warning(f"Text does not contain a valid multiple choice option ({', '.join(valid_choices)}): {text}")
    return "NO_CHOICE"

def eval_mc(line: dict) -> int:
    """Evaluate multiple choice prediction."""
    logger = MasterLogger.get_instance()
    pred = extract_choice(line["PRED"])
    gt = extract_choice(line["GT"])
    result = int(pred == gt)
    logger.info(f"Evaluated MC: GT='{gt}' vs PRED='{pred}' → {result}")
    return result

def normalize_answer(answer: str) -> str:
    """Normalize the answer by stripping whitespace and converting to lowercase."""
    logger = MasterLogger.get_instance()
    norm = answer.strip().lower()
    if "the answer is" in norm:
        norm = norm.split("the answer is")[-1].strip()
        logger.info(f"Stripped 'the answer is' → '{norm}'")
    else:
        logger.info(f"Normalized answer: '{norm}'")
    return norm

def grab_list_from_string(string_list: str) -> List[str]:
    """Extracts a list [x, y, z] from a string that may contain extra text."""
    logger = MasterLogger.get_instance()
    start = string_list.find('[')
    end = string_list.find(']')
    if start == -1 or end == -1:
        logger.warning(f"Could not find brackets in: {string_list}")
        return []
    list_str = string_list[start + 1:end].strip()
    items = [item.strip() for item in list_str.split(',') if item.strip()]
    return items

def calc_acc_llm(line: dict, prompter: OpenAIPrompter) -> int:
    logger = MasterLogger.get_instance()
    pred = normalize_answer(line["PRED"])
    gt = normalize_answer(line["GT"])
    question = line["question"]

    logger.info(f"Sending to LLM for eval: GT='{gt}', PRED='{pred}'")
    output = prompter.get_completion(
        input_texts={"question": question, "GT": gt, "PRED": pred},
        max_workers=1
    )

    if output == "true":
        acc = 1
    elif output == "false":
        acc = 0
    else:
        logger.error(f"LLM output must be 'true' or 'false', got: {output}")
        raise ValueError(f"LLM output must be 'true' or 'false', got: {output}")
    logger.info(f"LLM Eval: {gt} vs {pred} → {acc}")
    return acc

def calc_anomaly_acc(line: dict) -> int:
    """Calculate accuracy for anomaly detection tasks."""
    logger = MasterLogger.get_instance()
    pred = normalize_answer(line["PRED"])
    gt = normalize_answer(line["GT"])
    gt_bool = 1 if "anomaly point" in gt else 0
    pred_bool = 1 if "anomaly point" in pred else 0
    acc = int(pred_bool == gt_bool)
    logger.info(f"Anomaly Eval: GT={gt_bool} vs PRED={pred_bool} → {acc}")
    return acc

def normalize_choice(choice: str) -> str:
    """Lowercase, remove punctuation, and strip whitespace."""
    return re.sub(r'[^\w\s]', '', choice).strip().lower()

def is_choice_in_text(choice: str, text: str) -> bool:
    """Check if the choice appears as a whole word/phrase in the text."""
    pattern = r'\b' + re.escape(choice) + r'\b'
    return re.search(pattern, text) is not None

def calc_classification_acc(line: dict) -> int:
    """Calculate accuracy for classification tasks."""
    logger = MasterLogger.get_instance()
    pred_text = normalize_choice(line["PRED"])
    gt_text = normalize_choice(line["GT"])
    question = line["question"]

    try:
        substring = question.split("corresponds to", 1)[1].split("given inf", 1)[0]
    except IndexError:
        logger.error(f"Could not extract answer choices from question: {question}")
        raise ValueError(f"Could not extract answer choices from question: {question}")

    raw_choices = substring.strip().split("or")
    viable_choices = [normalize_choice(choice) for choice in raw_choices if choice.strip()]
    logger.info(f"Viable choices: {viable_choices}")

    gt_choice = next((choice for choice in viable_choices if is_choice_in_text(choice, gt_text)), None)
    pred_choice = next((choice for choice in viable_choices if is_choice_in_text(choice, pred_text)), None)

    if gt_choice is None:
        logger.error(f"GT choice '{gt_text}' not found in: {viable_choices}")
        raise ValueError(f"GT choice '{gt_text}' not found in viable choices.")
    if pred_choice is None:
        logger.error(f"PRED choice '{pred_text}' not found in: {viable_choices}")
        raise ValueError(f"Predicted choice '{pred_text}' not found in viable choices.")

    acc = int(pred_choice == gt_choice)
    logger.info(f"Classification Eval: GT='{gt_choice}' vs PRED='{pred_choice}' → {acc}")
    return acc

def calc_MSE(line: dict) -> float:
    logger = MasterLogger.get_instance()
    try:
        preds = grab_list_from_string(line["PRED"])
        gts = grab_list_from_string(line["GT"])
        preds = [float(p.strip().lstrip("x")) for p in preds]
        gts = [float(g.strip().lstrip("x")) for g in gts]
    except ValueError as e:
        logger.error(f"Error parsing predictions or ground truths: {e}")
        logger.error(f"GT: {line['GT']}, PRED: {line['PRED']}")
        return None

    if len(preds) != len(gts):
        logger.error("Predictions and ground truths must have the same length.")
        return None

    if len(preds) == 0 or len(gts) == 0:
        logger.warning("Empty prediction or ground truth list")
        return None

    mse = sum((p - g) ** 2 for p, g in zip(preds, gts)) / len(preds)
    logger.info(f"MSE: {mse:.4f} for GT and PRED of length {len(preds)}")
    return mse
