"""Shared pytest configuration for the lab helper tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("USPAS_LABS_SUPPRESS_PLOTS", "1")
os.environ.setdefault("QF_LAB_SUPPRESS_PLOTS", "1")
