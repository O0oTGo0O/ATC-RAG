"""Strict and lexical-token exact match; aligned ID<TAB>assertion files."""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from itertools import zip_longest

TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][A-Za-z_0-9$]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||\S')

def read_pairs(gold_file, output_file):
    with open(gold_file, encoding="utf-8-sig") as gold, open(output_file, encoding="utf-8-sig") as output:
        for expected, actual in zip_longest(gold, output):
            if expected is None or actual is None:
                raise ValueError("Gold/output row counts differ")
            ref_id, reference = expected.rstrip("\r\n").split("\t", 1)
            pred_id, prediction = actual.rstrip("\r\n").split("\t", 1)
            if ref_id != pred_id:
                raise ValueError(f"Sample IDs differ: {ref_id} / {pred_id}")
            yield ref_id, reference, prediction

def calc_accuracy(gold_file, output_file):
    pairs = list(read_pairs(gold_file, output_file))
    correct = sum(reference == prediction for _, reference, prediction in pairs)
    score = 100 * correct / len(pairs)
    print(f"Accuracy (strict): {score:.4f}")
    return score

def evaluate_predictions(directory):
    from codebleu import calc_codebleu
    from .bleu import compute_bleu
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    strict = sum(reference == prediction for reference, prediction in zip(references, predictions))
    by_type = defaultdict(list)
    for row in rows:
        row["correct"] = row["reference"] == row["prediction"]
        row["token_correct"] = TOKEN.findall(row["reference"]) == TOKEN.findall(row["prediction"])
        by_type[row["assertion_type"]].append(row["correct"])
    bleu = compute_bleu([[reference.split()] for reference in references], [pred.split() for pred in predictions])[0]
    codebleu = calc_codebleu(references, predictions, lang="java", weights=(.25, .25, .25, .25))
    metrics = {"n": len(rows), "strict_correct": strict, "ACC_strict_percent": 100 * strict / len(rows),
               "ACC_token_percent": 100 * sum(row["token_correct"] for row in rows) / len(rows),
               "BLEU_percent": 100 * bleu, "CodeBLEU_percent": 100 * codebleu["codebleu"],
               "codebleu_components": codebleu, "evidence_level": "offline-proxy",
               "normalization": "strict exact match; lexical-token exact match reported separately"}
    (directory / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (directory / "per_type.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "total", "correct", "ACC_strict_percent"])
        for name, hits in by_type.items():
            writer.writerow([name, len(hits), sum(hits), 100 * sum(hits) / len(hits)])
    with (directory / "per_sample.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False))
    return metrics

if __name__ == "__main__":
    calc_accuracy(sys.argv[1], sys.argv[2])

