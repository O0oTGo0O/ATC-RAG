"""Run the ATC-RAG no_f2 component ablation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run import main

if __name__ == "__main__":
    main(["--experiment", "no_f2"] + sys.argv[1:])
