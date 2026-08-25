"""Allow the scripts to run directly without installing the local package."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("SKYRMIONIUM_RESULTS", ROOT / "results")).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
