"""Shared arguments for the AG-RAG-shaped ATC-RAG experiments."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {"old": "OldDataSet", "new": "NewDataSet"}

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["index", "prepare", "train", "predict", "evaluate", "all"],
                        default="all")
    parser.add_argument("--dataset", choices=NAMES, default="old")
    parser.add_argument("--experiment", choices=["full", "no_f2", "no_atc", "diff_only", "IR_retriever", "finetuned_retriever",
                        "no_retriever", "pretrained_retriever", "CodeBERT", "GraphCodeBERT", "UniXcoder"],
                        default="full")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.tar.gz")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="Salesforce/codet5-base")
    parser.add_argument("--retriever-model", default="microsoft/codebert-base")
    parser.add_argument("--retriever-checkpoint", help="HF save_pretrained directory; required for finetuned_retriever")
    parser.add_argument("--backbone-model", help="Local path or HF ID for RQ3 encoder")
    parser.add_argument("--retrieval-mode", choices=["dense", "lexical"], default="dense")
    parser.add_argument("--k", "--passage_number", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--query-length", type=int, default=256)
    parser.add_argument("--atc-prefix", type=int, default=120)
    parser.add_argument("--atc-focal", type=int, default=56)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--index-limit", type=int)
    args = parser.parse_args(argv)
    if args.experiment in ["CodeBERT", "GraphCodeBERT", "UniXcoder"]:
        supplied = {value.split("=", 1)[0] for value in (sys.argv[1:] if argv is None else argv)}
        for flag, field, default in [("--epochs", "epochs", 20), ("--batch-size", "batch_size", 8),
                                     ("--grad-accum", "grad_accum", 4),
                                     ("--max-target-length", "max_target_length", 64),
                                     ("--beams", "beams", 1)]:
            if flag not in supplied:
                setattr(args, field, default)
        args.test_batch_size = args.batch_size if "--batch-size" in supplied else 4
    if args.output_dir is None:
        group = ROOT / "RQ1" if args.experiment == "full" else ROOT / (
            "RQ3" if args.experiment in ["CodeBERT", "GraphCodeBERT", "UniXcoder"] else "RQ2"
        ) / args.experiment
        if args.experiment in ["CodeBERT", "GraphCodeBERT", "UniXcoder"]:
            args.output_dir = group / "saved_models" / f"{NAMES[args.dataset]}_{args.k}"
        elif args.experiment in ["no_f2", "no_atc", "diff_only"]:
            args.output_dir = group / NAMES[args.dataset]
        else:
            args.output_dir = group / "result" / NAMES[args.dataset]
    return args

