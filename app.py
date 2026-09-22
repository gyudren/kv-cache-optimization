"""Source-checkout entry point; shares the package CLI and report validation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from kv_eval.main import DEFAULT_QUERY, main, run

if __name__ == "__main__":
    main()
