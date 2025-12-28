# tests/conftest.py
import sys
from pathlib import Path

# add repo root to python path so "import api" works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
