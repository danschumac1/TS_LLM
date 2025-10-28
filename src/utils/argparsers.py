import argparse
from pathlib import Path

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
    parser.add_argument('--n_shots', type=int, default=3)
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


def knn_parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # paths
    p.add_argument("--input_path", required=True)
    p.add_argument("--train_path", required=False, default=None,
                   help="If omitted, will try to derive by replacing '/test.jsonl' with '/train.jsonl'")
    p.add_argument("--output_path", required=True)

    # light controls (no hyperparameters)
    p.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds for tuning")
    p.add_argument("--scoring", type=str, default="accuracy", choices=["accuracy", "macro_f1"])
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--n_jobs", type=int, default=-1)

    # compatibility/debug flags
    p.add_argument("--debug", type=int, default=0)
    p.add_argument("--debug_prints", type=int, default=0)
    return p.parse_args()

def dtw1nn_parser(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1-NN with DTW baseline (classification)")
    p.add_argument("--input_path", type=str, required=True,
                   help="Directory containing train.jsonl and test.jsonl")
    p.add_argument("--output_path", type=str, required=True,
                   help="Path to write output JSONL (pred/gt per line)")
    p.add_argument("--n_yield_rows", type=int, default=1000,
                   help="Max rows to read from each split (useful for speed)")
    return p.parse_args(argv)

def random_baseline_parser() -> argparse.Namespace:
    """
    Args:
        --input_path   (required): Path to input JSONL.
        --output_path  (required): Where to write predictions JSONL.
        --batch_size   (optional): Number of rows per progress chunk (no effect on randomness). Default: 64
        --seed         (optional): Seed for reproducibility. If omitted, stays truly random.

    Example:
        python ./src/method/random_benchmark.py \
            --input_path ./data/datasets/TimerBed/CTU/test.jsonl \
            --output_path ./data/generations/random/TimerBed/CTU/test.jsonl \
            --batch_size 128 \
            --seed 1337
    """
    parser = argparse.ArgumentParser(description="Random-choice baseline (classification)")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSONL (expects fields: idx, question, options, label, ...)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write output JSONL (one dict per line)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for progress reporting (does not change randomness)")

    # Optional niceties: light validation without touching the filesystem
    args = parser.parse_args(argv)
    # basic sanity checks
    if not args.input_path.lower().endswith(".jsonl"):
        raise ValueError("input_path must be a .jsonl file")
    if not args.output_path.lower().endswith(".jsonl"):
        raise ValueError("output_path must be a .jsonl file")

    # normalize to strings to avoid pathlib types leaking elsewhere
    args.input_path = str(Path(args.input_path))
    args.output_path = str(Path(args.output_path))
    return args

def random_class_parser(argv=None):
    p = argparse.ArgumentParser(description="Random baseline for time series classification")
    p.add_argument("--input_path", type=str, required=True,
                   help="Directory containing train.jsonl and test.jsonl")
    p.add_argument("--output_path", type=str, required=True,
                   help="Where to write JSONL of {pred, gt}")
    p.add_argument("--n_yield_rows", type=int, default=1000,
                   help="Max rows to read from each split")
    p.add_argument("--mode", type=str, default="prior", choices=["prior","uniform","majority"],
                   help="Sampling strategy")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args(argv)