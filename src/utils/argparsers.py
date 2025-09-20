import argparse

def baseline_parse_args():
    parser = argparse.ArgumentParser(description="Batch Demo for Different Models")
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument(
        '--model_type',
        type=str,
        default="gpt",
        choices=[
            "gpt",
            "llama",
            "mistral",
            "gemma",
            # "TimeMQA_llama",
            # "TimeMQA_qwen",
            # "TimeMQA_mistral",
        ],
    )
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--device_map', type=int, nargs='+', default=[0])  # list ok; normalize happens downstream
    parser.add_argument('--show_prompt', type=int, default=0, choices=[0, 1], help="1 to show prompt, 0 to hide")
    return parser.parse_args()


def vl_time_parse_args():
    parser = argparse.ArgumentParser(description="Batch Demo for Different Models")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input data.")
    parser.add_argument('--output_path', type=str, required=True, help="Where to save final data.")
    parser.add_argument('--shots', type=str, required=True, help='Zero shot (zs) or Few shot (fs)', choices=["fs","zs"])
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=[
            "gpt",
        ]
    ),
    parser.add_argument(
        '--debug', 
        type=int, 
        required=False, 
        default=0, 
        choices=[0,1],
        help='does debugging prints and limits to three examples'
        )
    parser.add_argument(
        '--debug_prints',
        type=int,
        choices=[0,1],
        required=False,
        default=0
    )
    return parser.parse_args()

