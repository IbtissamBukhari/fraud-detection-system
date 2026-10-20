import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keep test predictions out of the real history file
os.environ.setdefault("HISTORY_PATH", str(Path(tempfile.mkdtemp()) / "history.json"))
