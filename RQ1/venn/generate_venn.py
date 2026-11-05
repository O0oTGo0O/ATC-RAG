"""Draw 2-6 method correctness intersections from the joined CSV."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from venn import get_labels, venn2, venn3, venn4, venn5, venn6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).parent / "baselines_new.csv"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "new.pdf"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, index_col="id")
    methods = list(frame.columns)
    sets = [set(frame.index[frame[name].astype(bool)]) for name in methods]
    draw = {2: venn2, 3: venn3, 4: venn4, 5: venn5, 6: venn6}[len(methods)]
    draw(get_labels(sets, fill=["number"]), names=methods)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close("all")

if __name__ == "__main__":
    main()

