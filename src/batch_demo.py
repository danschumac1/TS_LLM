'''
2025-08-06
Author: Dan Schumacher
How to run:
   python ./src/batch_demo.py
'''

import os
from utils.get_prompter import get_prompter_class_and_kwargs
from utils.logging_utils import MasterLogger

# Dummy test data (question-context pairs)
test_data = [
    {
        "question": "What year did the event take place?",
        "context": "The conference was held in San Francisco in 2019."
    },
    {
        "question": "Who won the award in 2020?",
        "context": "In 2020, Jane Doe received the prestigious award for her work."
    },
    {
        "question": "Where was the final match played?",
        "context": "The championship game took place at Wembley Stadium."
    },
    {
        "question": "Who was the president in 2011?",
        "context": "In 2011, Barack Obama was the President of the United States."
    }
]

# Shared prompt info
prompt_path = "./resources/prompts/context_question/context_question.yaml"
prompt_headers = {
    "question": "HERE IS THE QUESTION:",
    "context": "HERE IS THE CONTEXT:"
}
device_map = [2]
temperature = 0.7
show_prompt = False

def run_test(model_type: str, batched: bool):
    print(f"\n=== TEST: {model_type.upper()} {'BATCHED' if batched else 'NON-BATCHED'} ===")

    # Setup prompter
    PrompterClass, shared_kwargs = get_prompter_class_and_kwargs(
        model_type=model_type,
        device_map=device_map,
        show_prompt=show_prompt
    )

    prompter = PrompterClass(
        prompt_path=prompt_path,
        prompt_headers=prompt_headers,
        temperature=temperature,
        **shared_kwargs
    )

    if batched:
        outputs = prompter.get_completion(test_data)
        for i, output in enumerate(outputs):
            print(f"[{model_type} - Batched] Example {i} → {output}\n")
    else:
        for i, item in enumerate(test_data):
            output = prompter.get_completion(item)
            print(f"[{model_type} - Non-Batched] Example {i} → {output}\n")

def main():
    # Optional logger if you want to keep logs
    logger = MasterLogger(
        log_path="./logs/TKGForge_test.log",
        init=True,
        clear=True
    )

    for model_type in [
        "llama", 
        "gpt", 
        "gemma"
        ]:
        run_test(model_type=model_type, batched=False)
        run_test(model_type=model_type, batched=True)

if __name__ == "__main__":
    main()
