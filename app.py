"""
Streamlit entrypoint at repo root (for Streamlit Community Cloud).

Set **Main file path** to `app.py` (not the nested folder path). This avoids
spaces in paths and keeps dependency resolution on the root `requirements.txt`.

The real UI lives in `AI photo automation/app.py`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_APP_DIR = _ROOT / "AI photo automation"

if not _APP_DIR.is_dir():
    raise RuntimeError(
        f"Expected app directory at {_APP_DIR}. "
        "Clone the full repo (including the 'AI photo automation' folder)."
    )

sys.path.insert(0, str(_APP_DIR))
os.chdir(_APP_DIR)

_spec = importlib.util.spec_from_file_location("bird_ai_photo_app", _APP_DIR / "app.py")
if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load AI photo automation/app.py")

_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.main()
