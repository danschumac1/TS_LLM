'''
2025-09-19
Author: AdityaLab at Georgia Tech.
This code is written by AdityaLab at Georgia Tech.
It has been modified (by Dan Schumacher) from the original version to fit our needs.
The following adjustments have been made:
1. We generate the images as we go, instead of pre-generating them.
2. We added lots of debugging prints
3. We changed the work() function to work on a per row basis, instead of at the df level.
4. We removed pandas in favor of standard python data structures.
5. Split the large function into smaller more manageable functions.
'''

import base64
import time
import pandas as pd
from openai import OpenAI
from PIL import Image
import os
import pickle
import pandas as pd
import os
import soundfile as sf

#region LLM WRAPPERS
class GPT4VAPI:
    def __init__(
        self,
        model="gpt-4o-mini-2024-07-18",
        img_token="<<IMG>>",
        seed=66,
        temperature=0,
        detail="auto",modal="V",
    ):
        """
        Class for API calls to GPT-series models

        model[str]: the specific model checkpoint to use e.g. "gpt-4-turbo-2024-04-09"
        img_token[str]: string to be replaced with images
        seed[int]: seed for generation
        temperature[int]: temperature for generation
        detail[str]: resolution for images. Should be in ['low', 'high', 'auto'].
        """

        self.model = model
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.detail = detail
        self.client = OpenAI(api_key=self._load_api_key())
        self.token_usage = (0, 0, 0)
        self.response_times = []
        self.modal=modal

    def _load_api_key(self):
        # get from environment variable
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("Set OPENAI_API_KEY environment variable using export OPENAI_API_KEY='sk-...'")
        return openai_api_key

    def generate_image_url(self, image_path, detail="low"):
        # Given an image_path, return a dict
        # Function to encode the image
        def encode_image(image_path):
            if str(image_path).lower().endswith("tif"):
                with Image.open(image_path) as img:
                    img.convert("RGB").save("temp.jpeg", "JPEG")
                image_path = "temp.jpeg"
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64, {encode_image(image_path)}",
                "detail": detail,
            },
        }

    def generate_text_url(self, text):
        return {"type": "text", "text": text}

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        # count_time=False,
        # max_tokens=50,
        content_only=True,
    ):
        """
        Call the API to get the response for given prompt and images
        """
        if self.modal=="V" or self.modal=="LV":
            if not isinstance(image_paths, list):  # For single file
                image_paths = [image_paths]
            prompt = prompt.split(self.img_token)
            assert len(prompt) == len(image_paths) + 1
            if prompt[0] != "":
                messages = [self.generate_text_url(prompt[0])]
            else:
                messages = []
            for idx in range(1, len(prompt)):
                messages.append(
                    self.generate_image_url(image_paths[idx - 1], detail='low')
                )
                if prompt[idx].strip() != "":
                    messages.append(self.generate_text_url(prompt[idx]))
            if not real_call:
                return messages
        else:
            #只有语言模态，不需要图片
            # insert code here 
            messages = [self.generate_text_url(prompt)]
            if not real_call:
                return messages
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": messages}],
            #max_tokens=min(4096, max_tokens),
            max_tokens=4096,
            temperature=self.temperature,
            seed=self.seed,
        )

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        # results = [prompt, image_paths, response, end_time - start_time]

        self.token_usage = (
            self.token_usage[0] + response.usage.completion_tokens,
            self.token_usage[1] + response.usage.prompt_tokens,
            self.token_usage[2] + response.usage.total_tokens,
        )

        if content_only:
            return response.choices[0].message.content
        else:
            return response
#endregion

#region HELPERS
def process_aiff_file(index, prefix,folder_path='./Dataset/RCW',decimal_places=3):
    """
    读取指定的AIFF文件,将其转换为数组,然后返回为指定精度的字符串
    
    参数:
    folder_path (str): 包含AIFF文件的文件夹路径
    index (int): 文件名中的索引号
    decimal_places (int): 保留的小数位数
    
    返回:
    str: 包含数组数据的字符串,每个数字保留指定的小数位数
    """
    # 构造文件名
    file_name = f"{prefix}/{index}.aiff"
    file_path = os.path.join(folder_path, file_name)
    #print(file_path)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    # 读取AIFF文件
    data, _ = sf.read(file_path)
    
    # 确保数据是一维数组
    if data.ndim > 1:
        data = data.flatten()
    
    # 将数组转换为指定精度的字符串
    #print(data)
    formatted_data = [f"{x:.{decimal_places}f}" for x in data]
    
    # 将格式化后的数据连接成一个字符串
    result_string = ",".join(formatted_data)
    
    return result_string


def df2str(df_value, df_index, decimal_places=1):
    row_value = df_value.iloc[df_index]
    formatted_data = []

    def format_number(num):
        if decimal_places is not None:
            return f"{num:.{decimal_places}f}"
        return str(num)

    if isinstance(row_value, pd.Series):
        for item in row_value:
            if isinstance(item, pd.Series):
                # 如果是多维时间序列，将每个维度转换为逗号分隔的字符串
                formatted_data.append(','.join(map(format_number, item.values)))
            else:
                # 如果是单个值，直接转换为字符串
                formatted_data.append(format_number(item))
    else:
        # 如果row_value只是一个单一的值，直接转换为字符串
        formatted_data = [format_number(row_value)]

    # 返回结果
    if len(formatted_data) == 1:
        return f"{formatted_data[0]}"
    else:
        return ";".join(f"dim_{i}: {data}" for i, data in enumerate(formatted_data))


def debug_print(msg, debug=True):
    if debug:
        print(msg)


def _hr(label=""):
    print("\n" + "="*30 + f" {label} " + "="*30 + "\n")


def _preview(s, n=800):
    s = str(s)
    return (s[:n] + " ...[TRUNCATED]...") if len(s) > n else s


def _head(df, n=3):
    try:
        return df.head(n).to_string()
    except Exception:
        return f"<could not render head for {type(df)}>"

#endregion

#region MAIN WORK FUNCTION
def work(
    model,
    num_shot_per_class,
    # location,
    num_qns_per_round,
    test_df,
    demo_df,
    classes,
    class_desp,
    SAVE_FOLDER,
    dataset_name,
    detail="auto",
    file_suffix="",
    question="What is in the image above",
    prior_knowledge="",
    similar_use=0,
    similar_num=-1,
    image_token="<<IMG>>",
    name="",
    modal="LV",
    demo_value=0,
    test_value=0,
    hint="",
    debug=False
):
    """
    DEBUG-INSTRUMENTED VERSION
    - prints A LOT of state
    - runs ONLY ONE ROUND
    - exits BEFORE pickle.dump
    """
    import sys, textwrap, traceback, os, random, pickle, ast  # pprint
    from collections import Counter
    import numpy as np
    from tqdm import tqdm

    debug_print("\n\n######################## DEBUG: work() START ########################", debug)
    debug_print(f"model={model}", debug)
    debug_print(f"modal={modal}  (V=vision only, LV=vision+values, L=values only)", debug)
    debug_print(f"num_shot_per_class={num_shot_per_class} | num_qns_per_round={num_qns_per_round}", debug)
    debug_print(f"dataset_name={dataset_name} | SAVE_FOLDER={SAVE_FOLDER} | file_suffix={file_suffix}", debug)
    debug_print(f"similar_use={similar_use} | similar_num={similar_num}", debug)
    debug_print(f"question={question}", debug)
    debug_print(f"prior_knowledge={_preview(prior_knowledge, 200)}", debug)
    debug_print(f"hint={_preview(hint, 200)}", debug)
    debug_print(f"image_token={image_token}  | name={name}", debug)
    debug_print(f"classes={classes}", debug)
    debug_print(f"class_desp={class_desp}", debug)

    # DataFrame sanity
    debug_print("\n[DF] demo_df.shape =", getattr(demo_df, "shape", None), debug)
    debug_print("[DF] test_df.shape =", getattr(test_df, "shape", None), debug)
    debug_print("\n[DF] demo_df.head():\n" + _head(demo_df), debug)
    debug_print("\n[DF] test_df.head():\n" + _head(test_df), debug)

    if isinstance(demo_value, np.ndarray):
        debug_print(f"\n[DF] demo_value: numpy array with shape {demo_value.shape}", debug)
    else:
        try:
            debug_print(f"\n[DF] demo_value.shape: {getattr(demo_value,'shape',None)}", debug)
        except:
            debug_print("[DF] demo_value: <no shape>", debug)

    if isinstance(test_value, np.ndarray):
        debug_print(f"[DF] test_value: numpy array with shape {test_value.shape}", debug)
    else:
        try:
            debug_print(f"[DF] test_value.shape: {getattr(test_value,'shape',None)}", debug)
        except:
            debug_print("[DF] test_value: <no shape>", debug)

    # Map classes
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    debug_print("\n[class_to_idx]", class_to_idx, debug)

    # EXP_NAME construction
    if similar_use == 0:
        num_shots = num_shot_per_class * len(classes)
        shot_tag = "randomshot"
    else:
        num_shots = similar_num
        shot_tag = "similarshot"

    EXP_NAME = f"{dataset_name}_{model}model_{modal}modal_{num_shots}{shot_tag}_{num_qns_per_round}preround"
    folder_path = f"./{name}/"
    os.makedirs(folder_path, exist_ok=True)
    EXP_NAME = os.path.join(folder_path, EXP_NAME)

    debug_print(f"\n[EXP] EXP_NAME={EXP_NAME}", debug)
    debug_print(f"[EXP] test size = {len(test_df)}", debug)
    debug_print(f"[EXP] Output folder ensured: {folder_path}", debug)

    # API selection
    debug_print("\n[API] Selecting API wrapper ...", debug)
    if model.startswith("gpt"):
        debug_print("[API] -> GPT4VAPI", debug)
        api = GPT4VAPI(model=model, detail=detail, modal=modal)
    # elif model.startswith("internvl"):
    #     print("[API] -> InternVLAPI")
    #     api = InternVLAPI()
    # elif model.startswith("claude"):
    #     print("[API] -> ClaudeAPI")
    #     api = ClaudeAPI()
    # elif model.startswith("qwen"):
    #     print("[API] -> QWENAPI")
    #     api = QWENAPI(model=model, detail=detail, modal=modal)
    # else:
    #     print("[API] -> GeminiAPI (assert model == 'Gemini1.5')")
    #     assert model == "Gemini1.5"
    #     api = GeminiAPI(location=location)

    # Helpers they call
    def _df2str(df_value, df_index, decimal_places=1):
        s = df2str(df_value, df_index, decimal_places=decimal_places)
        debug_print(f"[NUMERIC] df2str(idx={df_index}) -> {_preview(s, 160)}", debug)
        return s

    def _process_aiff(ts_index, split):
        s = process_aiff_file(ts_index, split)
        debug_print(f"[AUDIO] process_aiff_file(index={ts_index}, split={split}) -> len={len(s)} chars", debug)
        return s

    # Prepare demo examples
    _hr("BUILD DEMO EXAMPLES")
    demo_examples = []
    if similar_use == 0:
        debug_print("[DEMO] Using random few-shot per class", debug)
        for class_name in classes:
            num_cases_class = 0
            debug_print(f"  -> class: {class_name}", debug)
            for j in demo_df[demo_df[class_name] == 1].itertuples():
                if num_cases_class == num_shot_per_class:
                    debug_print(f"    reached num_shot_per_class={num_shot_per_class}", debug)
                    break
                if modal in ("L", "LV"):
                    if dataset_name != "RCW":
                        ts_index = int(j.Index.split("_")[-1].split(".")[0])
                        debug_print(f"    DEMO idx={j.Index} -> ts_index={ts_index} (non-RCW)", debug)
                        data_L = _df2str(demo_value, ts_index)
                    else:
                        ts_index = j.Index.split(".")[0]
                        debug_print(f"    DEMO idx={j.Index} -> ts_index={ts_index} (RCW)", debug)
                        data_L = _process_aiff(ts_index, "train")
                    demo_examples.append((j.Index, class_desp[class_to_idx[class_name]], data_L))
                else:
                    demo_examples.append((j.Index, class_desp[class_to_idx[class_name]]))
                num_cases_class += 1
        debug_print(f"[DEMO] Total demo examples collected: {len(demo_examples)}", debug)
        debug_print("[DEMO] Preview first 3:", _preview(demo_examples[:3], 500), debug)
        assert len(demo_examples) == num_shot_per_class * len(classes), \
            f"Expected {num_shot_per_class*len(classes)} demos, got {len(demo_examples)}"
    else:
        debug_print("[DEMO] Using similarity-based demos (num_qns_per_round forced to 1)", debug)
        num_qns_per_round = 1
        if similar_num == -1:
            similar_num = len(classes)
        for _, row in test_df.iterrows():
            demo_examples_temp = []
            similar_samples = ast.literal_eval(row['similar_samples_low_to_high'])
            selected_samples = similar_samples[-similar_num:]
            debug_print(f"  TestRow idx={row.name} similar -> {selected_samples}", debug)
            for sample_idx in selected_samples:
                sample_class = demo_df.loc[sample_idx, classes].idxmax()
                if modal in ("L", "LV"):
                    if dataset_name != "RCW":
                        ts_index = int(sample_idx.split("_")[-1].split(".")[0])
                        debug_print(f"    DEMO(sim) idx={sample_idx} -> ts_index={ts_index} (non-RCW)", debug)
                        data_L = _df2str(demo_value, ts_index)
                    else:
                        ts_index = sample_idx.split(".")[0]
                        debug_print(f"    DEMO(sim) idx={sample_idx} -> ts_index={ts_index} (RCW)", debug)
                        data_L = _process_aiff(ts_index, "train")
                    # NOTE: fixed bug: use sample_idx/sample_class (not j/class_name)
                    demo_examples_temp.append((sample_idx, class_desp[class_to_idx[sample_class]], data_L))
                else:
                    demo_examples_temp.append((sample_idx, class_desp[class_to_idx[sample_class]]))
            demo_examples.append(demo_examples_temp)
        debug_print(f"[DEMO] Collected per-row similar demo sets: {len(demo_examples)}", debug)
        debug_print("[DEMO] Preview first set:", _preview(demo_examples[0], 500) if demo_examples else "<empty>", debug)

    # Load existing results (but we will exit before saving anyway)
    results_path = f"{EXP_NAME}.pkl"
    if os.path.isfile(results_path):
        debug_print(f"\n[CACHE] Found existing results at {results_path}, loading for reference ...", debug)
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        debug_print("[CACHE] Keys:", list(results.keys())[:5], debug)
    else:
        debug_print(f"\n[CACHE] No existing results at {results_path}", debug)
        results = {}

    # Shuffle test set and run ONLY ONE ROUND
    _hr("RUN ONE ROUND")
    test_df_shuf = test_df.sample(frac=1, random_state=66)
    debug_print("[TEST] Shuffled test_df (seed=66). Size:", len(test_df_shuf), debug)
    debug_print("[TEST] Head after shuffle:\n" + _head(test_df_shuf), debug)

    for start_idx in tqdm(range(0, len(test_df_shuf), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df_shuf), start_idx + num_qns_per_round)
        debug_print(f"\n[ROUND] batch start_idx={start_idx} end_idx={end_idx}", debug)

        if similar_use == 0:
            random.shuffle(demo_examples)
            now_demo_examples = demo_examples
            debug_print("[ROUND] Using randomly shuffled demos", debug)
        else:
            now_demo_examples = demo_examples[start_idx]
            label_counter = Counter(demo[1] for demo in now_demo_examples)
            most_common_label = label_counter.most_common(1)[0][0]
            debug_print(f"[ROUND] Using similarity demos | most_common_label={most_common_label}", debug)

        debug_print("[ROUND] now_demo_examples (preview):", _preview(now_demo_examples[:min(3, len(now_demo_examples))], 500), debug)

        prompt = ""
        if prior_knowledge:
            prompt += f"Task Description: {prior_knowledge}\n"
        if hint:
            prompt += f"Hint: {hint}\n"

        image_paths = []

        # Build image paths for demos
        for demo in now_demo_examples:
            if isinstance(demo, tuple):
                img_id = demo[0]
            elif isinstance(demo, str):
                img_id = demo
            else:
                debug_print(f"[WARN] Unexpected demo type: {type(demo)}. Skipping.", debug)
                continue
            image_path = os.path.join(SAVE_FOLDER, img_id + file_suffix)
            image_paths.append(image_path)

        debug_print("[ROUND] image_paths after demos (preview):", image_paths[:min(5, len(image_paths))], debug)

        # ======== VISION ONLY (we keep for completeness) ========
        if modal == "V":
            debug_print("[MODAL] VISION ONLY", debug)
            for demo in now_demo_examples:
                prompt += (
                    f"{image_token}Given the image above, answer the following question using the specified format.\n"
                    f"Question: {question}\n"
                    f"Choices: {str(class_desp)}\n"
                    f"Answer Choice: {demo[1]}\n"
                )

            qns_idx = []
            for idx, i in enumerate(test_df_shuf.iloc[start_idx:end_idx].itertuples()):
                qns_idx.append(i.Index)
                image_paths.append(os.path.join(SAVE_FOLDER, i.Index + file_suffix))
                qn_idx = idx + 1
                prompt += (
                    f"{image_token}Given the image above, answer the following question using the specified format.\n"
                    f"Question {qn_idx}: {question}\n"
                    f"Choices {qn_idx}: {str(class_desp)}\n\n"
                )

            # format template
            for i in range(start_idx, end_idx):
                qn_idx = i - start_idx + 1
                prompt += textwrap.dedent(f"""
                    Please respond with the following format for each question:
                    ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
                    Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
                    Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
                    ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

                    Do not deviate from the above format. Repeat the format template for the answer. """).strip() + "\n"

            debug_print("\n[PROMPT/V] >>>\n" + _preview(prompt, 1600), debug)
            debug_print("[IMAGES] >>>", image_paths[:min(10, len(image_paths))], debug)

            qns_id = str(qns_idx)
            debug_print("[CALL] qns_id=", qns_id, debug)
            for retry in range(3):
                if (qns_id in results) and (not results[qns_id].startswith("ERROR")):
                    debug_print("[CALL] Cached successful result found; skipping call.", debug)
                    break
                try:
                    debug_print(f"[CALL] API call attempt {retry+1}/3 ...", debug)
                    res = api(prompt, image_paths=image_paths, real_call=True, max_tokens=60*num_qns_per_round)
                except Exception:
                    res = f"ERROR!!!! {traceback.format_exc()}"
                debug_print("[RAW RES] >>>\n" + _preview(res, 1200), debug)
                results[qns_id] = res
                if not res.startswith("ERROR"):
                    break

        # ======== VISION + VALUES (LV) ========
        if modal == "LV":
            debug_print("[MODAL] VISION + VALUES", debug)
            for demo in now_demo_examples:
                prompt += (
                    f"{image_token}Given the image above, and the corresponding specific values are as follows: {demo[-1]}.\n"
                    f"Answer the following question using the specified format.\n"
                    f"Question: {question}\n"
                    f"Choices: {str(class_desp)}\n"
                    f"Answer Choice: {demo[1]}\n"
                )
            qns_idx = []
            for idx, i in enumerate(test_df_shuf.iloc[start_idx:end_idx].itertuples()):
                qns_idx.append(i.Index)
                image_paths.append(os.path.join(SAVE_FOLDER, i.Index + file_suffix))
                qn_idx = idx + 1
                if dataset_name != "RCW":
                    ts_index = int(i.Index.split("_")[-1].split(".")[0])
                    debug_print(f"[TEST] idx={i.Index} -> ts_index={ts_index} (non-RCW)", debug)
                    data_L = _df2str(test_value, ts_index)
                else:
                    ts_index = i.Index.split(".")[0]
                    debug_print(f"[TEST] idx={i.Index} -> ts_index={ts_index} (RCW)", debug)
                    data_L = _process_aiff(ts_index, "train")

                prompt += (
                    f"{image_token}Given the image above, and the corresponding specific values are as follows: {data_L}.\n"
                    f"Answer the following question using the specified format.\n"
                    f"Question {qn_idx}: {question}\n"
                    f"Choices {qn_idx}: {str(class_desp)}\n\n"
                )

            for i in range(start_idx, end_idx):
                qn_idx = i - start_idx + 1
                prompt += textwrap.dedent(f"""
                    Please respond with the following format for each question:
                    ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
                    Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
                    Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
                    ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

                    Do not deviate from the above format. Repeat the format template for the answer. """).strip() + "\n"

            debug_print("\n[PROMPT/LV] >>>\n" + _preview(prompt, 2000), debug)
            debug_print("[IMAGES] >>>", image_paths[:min(10, len(image_paths))], debug)

            qns_id = str(qns_idx)
            debug_print("[CALL] qns_id=", qns_id, debug)
            for retry in range(3):
                if (qns_id in results) and (not results[qns_id].startswith("ERROR")):
                    debug_print("[CALL] Cached successful result found; skipping call.", debug)
                    break
                try:
                    debug_print(f"[CALL] API call attempt {retry+1}/3 ...", debug)
                    res = api(prompt, image_paths=image_paths, real_call=True, max_tokens=60*num_qns_per_round)
                except Exception:
                    res = f"ERROR!!!! {traceback.format_exc()}"
                debug_print("[RAW RES] >>>\n" + _preview(res, 1600), debug)
                results[qns_id] = res
                if not res.startswith("ERROR"):
                    break

        # ======== VALUES ONLY (L) ========
        if modal == "L":
            debug_print("[MODAL] VALUES ONLY", debug)
            for demo in now_demo_examples:
                prompt += (
                    f"Given the corresponding specific numerical series are as follows: {demo[-1]}.\n"
                    f"Answer the following question using the specified format.\n"
                    f"Question: {question}\n"
                    f"Choices: {str(class_desp)}\n"
                    f"Answer Choice: {demo[1]}\n"
                )
            qns_idx = []
            for idx, i in enumerate(test_df_shuf.iloc[start_idx:end_idx].itertuples()):
                qns_idx.append(i.Index)
                qn_idx = idx + 1
                if dataset_name != "RCW":
                    ts_index = int(i.Index.split("_")[-1].split(".")[0])
                    debug_print(f"[TEST] idx={i.Index} -> ts_index={ts_index} (non-RCW)", debug)
                    data_L = _df2str(test_value, ts_index)
                else:
                    ts_index = i.Index.split(".")[0]
                    debug_print(f"[TEST] idx={i.Index} -> ts_index={ts_index} (RCW)", debug)
                    data_L = _process_aiff(ts_index, "train")

                prompt += (
                    f"Given the corresponding specific numerical series are as follows: {data_L}.\n"
                    f"Answer the following question using the specified format.\n"
                    f"Question {qn_idx}: {question}\n"
                    f"Choices {qn_idx}: {str(class_desp)}\n\n"
                )

            for i in range(start_idx, end_idx):
                qn_idx = i - start_idx + 1
                prompt += textwrap.dedent(f"""
                    Please respond with the following format for each question:
                    ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
                    Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
                    Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
                    ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

                    Do not deviate from the above format. Repeat the format template for the answer. """).strip() + "\n"

            debug_print("\n[PROMPT/L] >>>\n" + _preview(prompt, 1600), debug)
            debug_print("[IMAGES](ignored in L) >>>", image_paths[:min(10, len(image_paths))], debug)

            qns_id = str(qns_idx)
            debug_print("[CALL] qns_id=", qns_id, debug)
            for retry in range(3):
                if (qns_id in results) and (not results[qns_id].startswith("ERROR")):
                    debug_print("[CALL] Cached successful result found; skipping call.", debug)
                    break
                try:
                    debug_print(f"[CALL] API call attempt {retry+1}/3 ...", debug)
                    res = api(prompt, image_paths=image_paths, real_call=True, max_tokens=60*num_qns_per_round)
                except Exception:
                    res = f"ERROR!!!! {traceback.format_exc()}"
                debug_print("[RAW RES] >>>\n" + _preview(res, 1600), debug)
                results[qns_id] = res
                if not res.startswith("ERROR"):
                    break

        # we only do ONE batch for debugging:

    # Token usage (their code disables it; we keep their behavior but log)
    results["token_usage"] = -1
    debug_print("\n[TOKENS] token_usage set to -1 (disabled)", debug)

    if debug:
        _hr("EXIT BEFORE PICKLE.DUMP (AS REQUESTED)")
        print(f"Would write results to: {EXP_NAME}.pkl")
        print("results keys preview:", list(results.keys())[:5])
        print("Exiting now (sys.exit(0)) ...\n")
        sys.exit(0)

    with open(f"{EXP_NAME}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Data successfully saved to {EXP_NAME}.pkl")

#endregion