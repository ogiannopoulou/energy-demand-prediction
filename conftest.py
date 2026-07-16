import sys
from pathlib import Path

# Add project root to path so `api` and `src` are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
