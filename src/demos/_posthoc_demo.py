#!/usr/bin/env python3
"""
Run with e.g.:
  echo "RUNNING GPT RUNNING GPT RUNNING GPT RUNNING GPT RUNNING GPT RUNNING GPT 
  python ./src/_scratchpad.py --model_type gpt --device_map 0 --show_prompt 1
"""
import sys

from utils.prompter import QAs; sys.path.append("./src")
import argparse
import json
import os

from utils.prompter_factory import make_prompter

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run structured output demo with a single shared model instance."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt",
        choices=["gpt", "llama", "mistral", "gemma"],
    )
    parser.add_argument(
        "--device_map",
        type=int,               # force integers
        required=True,
        help="0, 1, 2, or 3 (on a four gpu server)." # on a four gpu server.
    )
    parser.add_argument("--show_prompt", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    show_prompt = bool(args.show_prompt)


    examples = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote 'Romeo and Juliet'?", "answer": "William Shakespeare"},
        {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
        {"question": "What does DNA stand for?", "answer": "Deoxyribonucleic Acid"},
        {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
        {"question": "What is the square root of 144?", "answer": "12"},
        {"question": "In which continent is the Sahara Desert located?", "answer": "Africa"},
        {"question": "What is the chemical symbol for water?", "answer": "H2O"},
        {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
        {"question": "What is the fastest land animal?", "answer": "Cheetah"},
        {"question": "Which language has the most native speakers?", "answer": "Mandarin Chinese"},
        {"question": "What year did the Titanic sink?", "answer": "1912"},
        {"question": "How many continents are there?", "answer": "Seven"},
        {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
        {"question": "What gas do plants use during photosynthesis?", "answer": "Carbon dioxide"},
        {"question": "Who was the first person to walk on the moon?", "answer": "Neil Armstrong"},
        {"question": "What is the hardest natural substance?", "answer": "Diamond"},
        {"question": "Which element has the chemical symbol 'O'?", "answer": "Oxygen"},
        {"question": "What is the currency of Japan?", "answer": "Yen"},
        {"question": "How many sides does a hexagon have?", "answer": "Six"}
    ]



    # ---------- Build both prompters; weights are shared via factory cache ----------
    prompter = make_prompter(
        model_name=args.model_type,
        device_map=args.device_map,  # <-- pass raw; factory normalizes
        prompt_path="./src/utils/prompts/demo/simpleQA.yaml",
        prompt_headers={"question": "HERE IS THE QUESTION:"},
        temperature=0.01,
        show_prompt=show_prompt,
    )

    QA_list = [QAs(question={"question": row["question"]}, answer=row["answer"]) for row in examples]
    prompter.add_examples_posthoc(QA_list[:3])

    # ---------- Run ----------
    question = "Who was the first major league baseball player to come out as gay?"
    out1 = prompter.get_completion({"question": question})

    print(out1)


if __name__ == "__main__":
    main()
