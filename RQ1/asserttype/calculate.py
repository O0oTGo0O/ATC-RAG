"""Assertion-family counts and strict accuracy from aligned prediction files."""
import argparse
import csv
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluator.acc import read_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--dataset", choices=["old", "new"])
    args = parser.parse_args()
    groups = {name: [0, 0] for name in ["Equals", "True", "That", "NotNull", "False", "Null", "ArrayEquals", "Same", "Other"]}
    for _, reference, prediction in read_pairs(args.gold_file, args.result_file):
        match = re.search(r"\bassert(Equals|True|That|NotNull|False|Null|ArrayEquals|Same)\s*\(", reference)
        name = match.group(1) if match else "Other"
        groups[name][0] += 1
        groups[name][1] += int(reference == prediction)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "total", "correct", "ACC_strict_percent"])
        for name, (total, correct) in groups.items():
            writer.writerow([name, total, correct, 100 * correct / total if total else ""])
    print(args.output_file)

if __name__ == "__main__":
    main()

