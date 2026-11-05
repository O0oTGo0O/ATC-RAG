"""Shared runnable entry point; use --help for data/model and stage options."""
from args_config import parse_args
from model import BACKBONES, ROOT, build_index, prepare_data, train_generator, predict
import subprocess
import sys

def run_native_rq3(args):
    if args.stage in ["index", "prepare"]:
        raise ValueError("RQ3 builds and updates its retrieval index during joint training; use --stage train/all")
    if args.precision != "fp32" or args.gradient_checkpointing:
        raise ValueError("Native AG-RAG RQ3 uses FP32 without gradient checkpointing")
    if args.stage != "evaluate":
        stem = {"CodeBERT": "codebert", "GraphCodeBERT": "graphcodebert", "UniXcoder": "unixcoder"}[args.experiment]
        command = [sys.executable, str(ROOT / "RQ3" / args.experiment / f"{stem}_main.py"),
                   "--model_name_or_path", args.backbone_model or BACKBONES[args.experiment],
                   "--dataset", args.dataset, "--data", str(args.data), "--output_dir", str(args.output_dir),
                   "--num_train_epochs", str(args.epochs), "--train_batch_size", str(args.batch_size),
                   "--eval_batch_size", str(args.batch_size), "--test_batch_size", str(args.test_batch_size),
                   "--gradient_accumulation_steps", str(args.grad_accum), "--learning_rate", str(args.learning_rate),
                   "--max_source_length", str(args.max_source_length), "--max_target_length", str(args.max_target_length),
                   "--query_length", str(args.query_length), "--passage_number", str(args.k),
                   "--beam_size", str(args.beams), "--seed", str(args.seed),
                   "--atc_prefix", str(args.atc_prefix), "--atc_focal", str(args.atc_focal)]
        if args.stage in ["train", "all"]:
            command += ["--do_train", "--do_eval"]
        if args.stage in ["predict", "all"]:
            command += ["--do_test"]
        if args.device == "cpu":
            command += ["--no_cuda"]
        for split in ["train", "val", "test"]:
            limit = getattr(args, f"{split}_limit")
            if limit is not None:
                command += [f"--{split}_limit", str(limit)]
        if args.data.is_dir():
            from args_config import NAMES
            for split, flag in [("train", "train_filename"), ("val", "dev_filename"), ("test", "test_filename")]:
                command += [f"--{flag}", str(args.data / NAMES[args.dataset] / f"assert_{split}_{args.dataset}.csv")]
        subprocess.run(command, check=True)
    if args.stage in ["all", "evaluate"]:
        from evaluator.acc import evaluate_predictions
        evaluate_predictions(args.output_dir)

def main(argv=None):
    args = parse_args(argv)
    if args.experiment in BACKBONES:
        run_native_rq3(args)
        return
    stages = ["index", "prepare", "train", "predict", "evaluate"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "index":
            if args.experiment != "no_retriever":
                build_index(args)
        elif stage == "prepare":
            prepare_data(args)
        elif stage == "train":
            train_generator(args)
        elif stage == "predict":
            predict(args)
        elif stage == "evaluate":
            from evaluator.acc import evaluate_predictions
            evaluate_predictions(args.output_dir)

if __name__ == "__main__":
    main()

