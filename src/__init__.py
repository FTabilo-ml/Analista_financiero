"""
Nodum AI | Trading Toolkit
Expose DATA_DIR y NOTEBOOKS_DIR para uso rápido en notebooks.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

__all__ = ["DATA_DIR", "NOTEBOOKS_DIR"]
