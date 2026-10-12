"""FastAPI service that exposes the trained hydraulic condition models to the web frontend."""

import sys
from pathlib import Path

# The project's modules live in src/ and import each other by bare name
# (`from data_loader import ...`), exactly as app.py and MLapp.py use them.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
