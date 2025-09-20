'''
2025-09-19
Authors:
  - AdityaLab at Georgia Tech (original)
  - Dan Schumacher (refactor for row-wise processing, debug, image-on-the-fly, no pandas)

What changed (functionality preserved):
1) The pipeline runs per ROW (single example) instead of dataframe-level batches.
2) Images are generated on-the-fly from the time series (no pre-generated images).
3) Extensive debug prints are added (toggle with debug=True).
4) Pandas removed in favor of plain Python dicts/lists.
5) The monolithic work() is split into focused helpers.
6) All non-English comments/docstrings translated to English.
7) Loaded OpenAI key with .env file

IMPORTANT: Scientifically identical behavior where applicable:
- Same prompt templates for V / LV / L.
- Same few-shot composition logic (random per-class shots by default).
- Same class mapping and answer formatting rules.
- Same seed/temperature/control options.
'''
#region IMPORTS
import os
import re
import time
import base64
import random
from typing import List, Dict, Tuple, Optional, Any

from PIL import Image
from dotenv import load_dotenv
import numpy as np

# Use non-interactive backend for headless servers
import matplotlib

from utils.logging_utils import MasterLogger
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import soundfile as sf  # retained for RCW audio use-case
except Exception:
    sf = None

from openai import OpenAI


#endregion
#region ==============================
# Debug helpers
# ====================================

def debug_print(msg: str, debug: bool = True) -> None:
    if debug:
        print(msg)

def _hr(label: str = "") -> None:
    return "\n" + "=" * 30 + f" {label} " + "=" * 30 + "\n"

def _preview(s: Any, n: int = 800) -> str:
    s = str(s)
    return (s[:n] + " ...[TRUNCATED]...") if len(s) > n else s

#endregion
#region ==============================
# API wrapper (translated & tidy)
# ====================================

class GPT4VAPI:
    """
    Wrapper for GPT-series multimodal/vision models.

    Parameters
    ----------
    model : str
        Model name, e.g., "gpt-4o-mini-2024-07-18".
    img_token : str
        Placeholder token in the prompt that will be replaced by images.
    seed : int
        Random seed for generation.
    temperature : float
        Sampling temperature.
    detail : str
        Image resolution detail in {'low','high','auto'}.
    modal : str
        One of {"V","LV","L"} (Vision, Vision+Values, Values-only).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-2024-07-18",
        img_token: str = "<<IMG>>",
        seed: int = 66,
        temperature: float = 0.0,
        detail: str = "auto",
        modal: str = "V",
        debug: bool = False
    ):
        self.model = model
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.detail = detail
        self.modal = modal
        self.client = OpenAI(api_key=self._load_env())
        self.token_usage = (0, 0, 0)
        self.response_times: List[float] = []
        self.debug = debug

    def _load_env(self) -> str:
        """Loads API key from .env"""
        load_dotenv("./resources/.env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"API Key not found. Set OPENAI_API_KEY=xxxx in ./resources/.env")
        return api_key

    @staticmethod
    def _encode_image_to_b64(image_path: str) -> str:
        # TIF -> JPEG conversion for compatibility
        if str(image_path).lower().endswith(".tif") or str(image_path).lower().endswith(".tiff"):
            with Image.open(image_path) as img:
                img.convert("RGB").save("temp_vl_time.jpeg", "JPEG")
            image_path = "temp_vl_time.jpeg"
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _image_part(self, image_path: str, detail: str = "low") -> dict:
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64, {self._encode_image_to_b64(image_path)}",
                "detail": detail,
            },
        }

    @staticmethod
    def _text_part(text: str) -> dict:
        return {"type": "text", "text": text}

    def __call__(
        self,
        prompt: str,
        image_paths: List[str],
        real_call: bool = True,
        max_tokens: int = 4096,
        content_only: bool = True,
        debug: bool = False,
    ) -> Any:
        """
        Send a message with interleaved text and images. `prompt` is split by img_token
        and images are inserted in those positions.
        """
        # Build the content stream (text-image-text-...)
        parts = prompt.split(self.img_token)
        assert len(parts) == len(image_paths) + 1, (
            "The prompt should contain exactly one image token per image provided."
        )

        messages: List[dict] = []
        if parts[0]:
            messages.append(self._text_part(parts[0]))
        for i in range(1, len(parts)):
            messages.append(self._image_part(image_paths[i - 1], detail=self.detail if self.detail else "low"))
            if parts[i].strip():
                messages.append(self._text_part(parts[i]))

        if debug:
            debug_print(f"[API] content blocks={len(messages)} (text/image interleaved).", debug)

        if not real_call:
            return messages

        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": messages}],
            max_tokens=max_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )
        t1 = time.time()
        self.response_times.append(t1 - t0)

        # track usage
        self.token_usage = (
            self.token_usage[0] + (response.usage.completion_tokens or 0),
            self.token_usage[1] + (response.usage.prompt_tokens or 0),
            self.token_usage[2] + (response.usage.total_tokens or 0),
        )

        return response.choices[0].message.content if content_only else response


#region ==============================
# Data formatting utilities
#endregion ===========================

def format_time_series_values(series_obj: Any, decimal_places: Optional[int] = 1) -> str:
    """
    Convert a time series object to a compact string. Supports:
      - dict with 'dim_0', 'dim_1', ... -> "dim_0: v1,v2; dim_1: v1,v2"
      - list/np.ndarray -> "v1,v2,..."
    """
    def fmt(x: float) -> str:
        return f"{x:.{decimal_places}f}" if decimal_places is not None else str(x)

    if isinstance(series_obj, dict):
        out_per_dim = []
        for k in sorted(series_obj.keys()):  # ensure deterministic order
            arr = series_obj[k]
            if isinstance(arr, np.ndarray):
                arr = arr.tolist()
            out_per_dim.append(f"{k}: {','.join(fmt(float(v)) for v in arr)}")
        return "; ".join(out_per_dim)

    if isinstance(series_obj, (list, tuple, np.ndarray)):
        arr = series_obj.tolist() if isinstance(series_obj, np.ndarray) else list(series_obj)
        return ",".join(fmt(float(v)) for v in arr)

    # fallback to str
    return str(series_obj)


def read_aiff_series_as_string(index: str, prefix: str, folder_path: str = "./Dataset/RCW", decimal_places: int = 3) -> str:
    """
    Read a specific AIFF file and convert it to a string with fixed decimals.
    (Used for RCW dataset compatibility.)
    """
    if sf is None:
        raise RuntimeError("soundfile unavailable; install pysoundfile to read AIFF.")
    file_name = f"{prefix}/{index}.aiff"
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    data, _ = sf.read(file_path)
    if data.ndim > 1:
        data = data.flatten()
    return ",".join(f"{float(x):.{decimal_places}f}" for x in data)

#endregion
#region ==============================
# Plotting: generate image on-the-fly
# ====================================

def _dim_sort_key(k: str) -> int:
    """Sort dims like dim_0, dim_1, ... numerically; unknowns go last."""
    m = re.match(r"^dim_(\d+)$", str(k))
    return int(m.group(1)) if m else 10**9

def generate_ts_image_png(
    item: Dict[str, Any],
    save_folder: str,
    file_suffix: str = ".png",
    dpi: int = 140
) -> str:
    """
    Create and save a multivariate (or univariate) time series plot as PNG from a single `item` dict.
    Uses:
      - item["ts"]            : dict of dim_* -> list[float] (or a list for univariate)
      - item["dimension_map"] : dict mapping 'dim_i' -> human-readable name (optional)
      - item["x_label"]       : x-axis label (fallback 'Time')
      - item["y_label"]       : y-axis label (fallback 'Value')
      - item["title"]         : plot title    (fallback f"Time Series: {idx}")
      - item["idx"]           : identifier used for filename/title fallback
    """
    os.makedirs(save_folder, exist_ok=True)
    idx = item.get("idx", "unknown")
    ts  = item.get("ts", {})

    # Normalize ts to a dict
    if not isinstance(ts, dict):
        ts = {"dim_0": list(ts)}

    img_path = os.path.join(save_folder, f"{idx}{file_suffix or '.png'}")

    # check if it already exists
    if os.path.exists(img_path):
        return img_path

    # Meta (with safe fallbacks)
    dim_map = item["dimension_map"] 
    x_label = item["x_label"] 
    y_label = item["y_label"]
    title   = item["title"]

    # Plot
    plt.figure()
    plotted = 0
    # Plot in numeric dim order: dim_0, dim_1, ...
    for dim_key in sorted(ts.keys(), key=_dim_sort_key):
        series = ts.get(dim_key, None)
        if series is None:
            continue
        label = dim_map[dim_key]  # use human-readable if available
        plt.plot(series, label=label)
        plotted += 1

    if plotted > 1:
        plt.legend()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(img_path, dpi=dpi)
    plt.close()

    return img_path

#endregion
#region ==============================
# Few-shot selection
# ====================================

def select_random_few_shots_per_class(
    demo_pool: List[Dict],
    classes: List[str],
    class_desp: List[str],
    num_shot_per_class: int,
    save_folder: str,
    file_suffix: str,
    debug: bool = False
) -> List[Tuple[str, str, str, str]]:
    """
    Pick `num_shot_per_class` items from demo_pool for each class in `classes`.
    For each chosen demo:
       - Generate its plot image on-the-fly
       - Prepare its numeric series string (LV/L)
    Returns a list of tuples: (idx, class_description, values_string, image_path)

    NOTE: We assume each demo item structure matches your input JSONL items:
      {
        "idx": "...",
        "answer": "...",    # human-readable GT (e.g., "WALKING")
        "label": "A",       # letter
        "ts": { "dim_0": [...], "dim_1": [...], ... },
        ...
      }
    """
    # class name -> indices in classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # map item to class index via its "label"
    label_to_classname = {chr(ord('A') + i): classes[i] for i in range(len(classes))}

    buckets: Dict[str, List[Dict]] = {c: [] for c in classes}
    for ex in demo_pool:
        lbl = ex.get("label")
        if lbl and lbl in label_to_classname:
            cname = label_to_classname[lbl]
            buckets[cname].append(ex)

    fewshots: List[Tuple[str, str, str, str]] = []
    for cname in classes:
        pool = buckets.get(cname, [])
        random.shuffle(pool)
        pick = pool[:num_shot_per_class]
        debug_print(f"[DEMO] Class '{cname}': pool={len(pool)} -> picked={len(pick)}", debug)
        for item in pick:
            idx = item["idx"]
            ts = item["ts"]
            img_path = generate_ts_image_png(item, save_folder, file_suffix=file_suffix)
            values_str = format_time_series_values(ts, decimal_places=1)
            fewshots.append((idx, class_desp[class_to_idx[cname]], values_str, img_path))
    debug_print(f"[DEMO] Total few-shots collected: {len(fewshots)}", debug)
    return fewshots

#endregion
#region ==============================
# Prompt builders (preserving formats)
#=====================================

def build_prompt_V(
    question: str,
    class_desp: List[str],
    image_token: str,
    prior_knowledge: str,
    hint: str,
    demo_tuples: List[Tuple[str, str, str, str]],  # (idx, class_desc, values_str, img_path)
    test_idx: str, # have to leave this so consistant (even though it is not accessed)
    print_prompt:bool = False
) -> Tuple[str, List[str]]:
    """
    Vision-only prompt: uses image blocks + label answers for demos; then asks the test question.
    """
    prompt = ""
    if prior_knowledge:
        prompt += f"Task Description: {prior_knowledge}\n"
    if hint:
        prompt += f"Hint: {hint}\n"

    # demos (ignore values_str)
    image_paths: List[str] = []
    for _, class_desc, _, img_path in demo_tuples:
        image_paths.append(img_path)
        prompt += (
            f"{image_token}Given the image above, answer the following question using the specified format.\n"
            f"Question: {question}\n"
            f"Choices: {str(class_desp)}\n"
            f"Answer Choice: {class_desc}\n"
        )

    # test
    # (image created outside)
    prompt += (
        f"{image_token}Given the image above, answer the following question using the specified format.\n"
        f"Question 1: {question}\n"
        f"Choices 1: {str(class_desp)}\n\n"
        "Please respond with the following format for each question:\n"
        "---BEGIN FORMAT TEMPLATE FOR QUESTION 1---\n"
        "Answer Choice 1: [Your Answer Choice Here for Question 1]\n"
        "Confidence Score 1: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question 1]\n"
        "---END FORMAT TEMPLATE FOR QUESTION 1---\n\n"
        "Do not deviate from the above format. Repeat the format template for the answer.\n"
    )
    if print_prompt:
        logger = MasterLogger.get_instance()
        logger.info(prompt)
        print(prompt)

    return prompt, image_paths


def build_prompt_LV(
    question: str,
    class_desp: List[str],
    image_token: str,
    prior_knowledge: str,
    hint: str,
    demo_tuples: List[Tuple[str, str, str, str]],  # (idx, class_desc, values_str, img_path)
    test_values_str: str,
    print_prompt:bool = False
) -> Tuple[str, List[str]]:
    """
    Vision + Values prompt: same as original LV template.
    """
    prompt = ""
    if prior_knowledge:
        prompt += f"Task Description: {prior_knowledge}\n"
    if hint:
        prompt += f"Hint: {hint}\n"

    image_paths: List[str] = []
    for _, class_desc, values_str, img_path in demo_tuples:
        image_paths.append(img_path)
        prompt += (
            f"{image_token}Given the image above, and the corresponding specific values are as follows: {values_str}.\n"
            f"Answer the following question using the specified format.\n"
            f"Question: {question}\n"
            f"Choices: {str(class_desp)}\n"
            f"Answer Choice: {class_desc}\n"
        )

    prompt += (
        f"{image_token}Given the image above, and the corresponding specific values are as follows: {test_values_str}.\n"
        f"Answer the following question using the specified format.\n"
        f"Question 1: {question}\n"
        f"Choices 1: {str(class_desp)}\n\n"
        "Please respond with the following format for each question:\n"
        "---BEGIN FORMAT TEMPLATE FOR QUESTION 1---\n"
        "Answer Choice 1: [Your Answer Choice Here for Question 1]\n"
        "Confidence Score 1: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question 1]\n"
        "---END FORMAT TEMPLATE FOR QUESTION 1---\n\n"
        "Do not deviate from the above format. Repeat the format template for the answer.\n"
    )

    if print_prompt:
        logger = MasterLogger.get_instance()
        logger.info(prompt)
        print(prompt)

    return prompt, image_paths


def build_prompt_L(
    question: str,
    class_desp: List[str],
    prior_knowledge: str,
    hint: str,
    demo_tuples: List[Tuple[str, str, str, str]],  # (idx, class_desc, values_str, img_path) (img ignored)
    test_values_str: str,
    print_prompt:bool = False
) -> Tuple[str, List[str]]:
    """
    Values-only prompt (no images inserted).
    """
    prompt = ""
    if prior_knowledge:
        prompt += f"Task Description: {prior_knowledge}\n"
    if hint:
        prompt += f"Hint: {hint}\n"

    for _, class_desc, values_str, _ in demo_tuples:
        prompt += (
            f"Given the corresponding specific numerical series are as follows: {values_str}.\n"
            f"Answer the following question using the specified format.\n"
            f"Question: {question}\n"
            f"Choices: {str(class_desp)}\n"
            f"Answer Choice: {class_desc}\n"
        )

    prompt += (
        f"Given the corresponding specific numerical series are as follows: {test_values_str}.\n"
        f"Answer the following question using the specified format.\n"
        f"Question 1: {question}\n"
        f"Choices 1: {str(class_desp)}\n\n"
        "Please respond with the following format for each question:\n"
        "---BEGIN FORMAT TEMPLATE FOR QUESTION 1---\n"
        "Answer Choice 1: [Your Answer Choice Here for Question 1]\n"
        "Confidence Score 1: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question 1]\n"
        "---END FORMAT TEMPLATE FOR QUESTION 1---\n\n"
        "Do not deviate from the above format. Repeat the format template for the answer.\n"
    )

    if print_prompt:
        logger = MasterLogger.get_instance()
        logger.info(prompt)
        print(prompt)

    return prompt, []  # no images

#endregion
#region
#region ==============================
# Parsing model output
# ====================================

def parse_final_answer(
    text: str,
    classes: List[str],
    class_desp: List[str],
    debug: bool = False
) -> Tuple[str, str]:
    """
    Extract the model's final answer as:
       - label_letter in {"A","B",...}
       - human string of the class (from class_desp)

    Robust to:
      - "Answer Choice 1: [A]" or "...: WALKING" or a full sentence.

    Returns ("?", "?") if nothing can be resolved.
    """
    debug_print(f"[PARSE] Raw model text:\n{_preview(text, 1200)}", debug)

    # 1) Look for explicit letter after 'Answer Choice'
    m = re.search(r"Answer Choice\s*\d*\s*:\s*[\[\(]?\s*([A-F])\s*[\]\)]?", text, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(class_desp):
            return letter, class_desp[idx]

    # 2) Try to match any class description token textually
    for i, desc in enumerate(class_desp):
        # try exact or bracketed label "[A] WALKING" or just "WALKING"
        token = desc
        if token in text:
            letter = chr(ord('A') + i)
            return letter, desc

    # 3) Fallback: search for [X] patterns
    m2 = re.search(r"\[([A-F])\]", text)
    if m2:
        letter = m2.group(1).upper()
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(class_desp):
            return letter, class_desp[idx]

    return "?", "?"

#endregion
#region ==============================
# Main row-wise worker
# ====================================

def work(
    item: Dict,
    *,
    # model + API
    model: str = "gpt-4o-mini-2024-07-18",
    modal: str = "LV",              # "V" | "LV" | "L"
    seed: int = 66,
    temperature: float = 0.0,
    detail: str = "low",
    image_token: str = "<<IMG>>",

    # few-shot & classes
    classes: List[str],
    class_desp: List[str],
    demo_pool: List[Dict],
    num_shot_per_class: int = 1,

    # storage / plotting
    save_folder: str = "./VLTimeImgs",
    file_suffix: str = ".png",

    # prompt & metadata
    question: str,
    prior_knowledge: str = "",
    hint: str = "",

    # controls
    real_call: bool = True,
    debug: bool = False,
    print_prompt = False
) -> Dict:
    """
    Row-wise processing for a single JSONL item of the form in your example.

    Returns the requested output dict:
    {
      "idx": item["idx"],
      "question": item["question"],
      "PRED": {"final_answer": "<LETTER>", ...},
      "task_type": item["task_type"],
      "GT": item["answer"],
      "label": item["label"],
      "parent_task": "NA",
      "application_domain": "NA",
      "eval_type": "classification"
    }
    """
    debug_print(_hr("work(row-wise)"), debug)
    debug_print(f"[ROW] idx={item.get('idx')}", debug)
    debug_print(f"[ROW] dataset={item.get('dataset')} | split={item.get('split')} | modal={modal}", debug)

    # Map classes (deterministic A,B,C...)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    debug_print(f"[MAP] classes -> indices: {class_to_idx}", debug)
    debug_print(f"[MAP] class_desp (A..): {class_desp}", debug)

    # Generate image for TEST item now (on-the-fly)
    test_idx = item["idx"]
    ts = item["ts"]
    test_img_path = generate_ts_image_png(item, save_folder, file_suffix=file_suffix)

    # Build few-shot demos (random per class)
    demo_tuples = select_random_few_shots_per_class(
        demo_pool=demo_pool,
        classes=classes,
        class_desp=class_desp,
        num_shot_per_class=num_shot_per_class,
        save_folder=save_folder,
        file_suffix=file_suffix,
        debug=debug
    )

    # Prepare LV/L numeric strings
    test_values_str = format_time_series_values(ts, decimal_places=1)

    # Build prompt by modal
    if modal == "V":
        prompt, image_paths = build_prompt_V(
            question=question,
            class_desp=class_desp,
            image_token=image_token,
            prior_knowledge=prior_knowledge,
            hint=hint,
            demo_tuples=demo_tuples,
            test_idx=test_idx,
            print_prompt=print_prompt
        )
        # Append test image path at end (after demos)
        image_paths = list(image_paths) + [test_img_path]

    elif modal == "LV":
        prompt, image_paths = build_prompt_LV(
            question=question,
            class_desp=class_desp,
            image_token=image_token,
            prior_knowledge=prior_knowledge,
            hint=hint,
            demo_tuples=demo_tuples,
            test_values_str=test_values_str,
            print_prompt=print_prompt
        )
        image_paths = list(image_paths) + [test_img_path]

    else:  # "L"
        prompt, image_paths = build_prompt_L(
            question=question,
            class_desp=class_desp,
            prior_knowledge=prior_knowledge,
            hint=hint,
            demo_tuples=demo_tuples,
            test_values_str=test_values_str,
            print_prompt=print_prompt
        )
        image_paths = []  # values-only

    debug_print("\n[PROMPT PREVIEW]\n" + _preview(prompt, 2000), debug)
    debug_print(f"[IMAGES] {image_paths}", debug)

    # API call
    api = GPT4VAPI(model=model, img_token=image_token, seed=seed, temperature=temperature, detail=detail, modal=modal)
    try:
        res_text = api(prompt, image_paths=image_paths, real_call=real_call, max_tokens=4096, content_only=True, debug=debug)
    except Exception as e:
        res_text = f"ERROR: {e}"
    debug_print("\n[RAW RESPONSE]\n" + _preview(res_text, 2000), debug)

    # Parse model answer -> letter + description
    pred_letter, pred_desc = parse_final_answer(res_text, classes=classes, class_desp=class_desp, debug=debug)

    # Compose result row
    out = {
        "idx": item.get("idx"),
        "question": item.get("question"),
        "PRED": {
            "final_answer": pred_letter,
            "parsed_desc": pred_desc,
            "raw_text": res_text
        },
        "task_type": item.get("task_type", "classification"),
        "GT": item.get("answer"),
        "label": item.get("label"),
        "parent_task": "NA",
        "application_domain": "NA",
        "eval_type": "classification"
    }

    if debug:
        _hr("DONE (row-wise)")
        debug_print(f"[OUT] {_preview(out, 1200)}", True)

    return out
