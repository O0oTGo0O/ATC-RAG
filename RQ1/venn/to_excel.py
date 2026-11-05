"""Join correctness by ID into an Excel-readable CSV; keep the original entry filename."""
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", help="Name=path/to/per_sample.jsonl; exact matching sample IDs")
    parser.add_argument("--dataset", choices=["old", "new"], default="new")
    parser.add_argument("--baselines", nargs="+", default=["EditAS", "AG-RAG"])
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out is None:
        args.out = str(Path(__file__).parent / f"baselines_{args.dataset}.csv")
    if not args.methods:
        result_dir = Path(__file__).resolve().parents[1] / "result" / (
            "OldDataSet" if args.dataset == "old" else "NewDataSet")
        columns = {}
        for name in args.baselines:
            prefix = "" if name == "AG-RAG" and args.dataset == "old" else name + "_"
            labels = (result_dir / f"{prefix}Label.txt").read_text(encoding="utf-8-sig").splitlines()
            predictions = (result_dir / f"{prefix}Result.txt").read_text(encoding="utf-8-sig").splitlines()
            if len(labels) != len(predictions):
                raise ValueError(f"{name}: label/prediction row counts differ")
            columns[name] = [reference.strip() == prediction.strip()
                             for reference, prediction in zip(labels, predictions)]
        if len({len(values) for values in columns.values()}) != 1:
            raise ValueError("Reference methods have different sample counts; sample-ID alignment is required. "
                             "Old RetriGen has 15668 rows, other old methods have 15676; do not truncate.")
        table = pd.DataFrame(columns)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index_label="id", encoding="utf-8-sig")
        print("Baseline alignment uses the original archive's row order; correctness uses each method's own labels.")
        print(args.out)
        return
    table = None
    for specification in args.methods:
        name, path = specification.split("=", 1)
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        frame = pd.DataFrame(rows).set_index("id")
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate IDs: {name}")
        if table is None:
            table = pd.DataFrame(index=frame.index)
        if set(table.index) != set(frame.index):
            raise ValueError(f"Sample IDs differ: {name}")
        table[name] = frame["correct"].astype(bool)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index_label="id", encoding="utf-8-sig")

if __name__ == "__main__":
    main()

