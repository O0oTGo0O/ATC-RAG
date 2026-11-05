"""Run the ATC-RAG diff_only component ablation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run import main

if __name__ == "__main__":
    main(["--experiment", "diff_only"] + sys.argv[1:])
