"""Evaluate a SuperMimic checkpoint on reference motions."""

from pathlib import Path
import sys

import tyro


EXTENSION_ROOT = Path(__file__).resolve().parents[1] / "extensions/supermimic"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXTENSION_ROOT / "scripts"))

from eval_benchmark import main


if __name__ == "__main__":
    tyro.cli(main)
