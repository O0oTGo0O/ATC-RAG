"""Java CodeBLEU using Windows-compatible tree-sitter wheels."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluator.acc import read_pairs

def main():
    from codebleu import calc_codebleu
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", required=True)
    parser.add_argument("--hyp", required=True)
    parser.add_argument("--lang", default="java", choices=["java"])
    args = parser.parse_args()
    rows = list(read_pairs(args.refs, args.hyp))
    result = calc_codebleu([row[1] for row in rows], [row[2] for row in rows], lang=args.lang,
                          weights=(.25, .25, .25, .25))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

