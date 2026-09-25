"""
DTE Framework CLI - Thin wrapper that re-exports the CLI from main.py.

This module exists so that the ``console_scripts`` entry point
``dte=dte.cli:cli`` works correctly when the package is installed.
"""

import sys
from pathlib import Path

# Add project root to path so main.py can be imported
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from main import cli  # noqa: E402, F401

__all__ = ["cli"]
