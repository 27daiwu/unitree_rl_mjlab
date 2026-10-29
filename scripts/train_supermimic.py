"""Train SuperMimic in its isolated extension environment."""

from pathlib import Path
import sys


EXTENSION_SRC = Path(__file__).resolve().parents[1] / "extensions/supermimic/src"
sys.path.insert(0, str(EXTENSION_SRC))

from supermimic.scripts.train import main


if __name__ == "__main__":
    main()
