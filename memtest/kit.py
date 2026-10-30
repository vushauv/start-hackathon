"""Locations of the vendored RxAI participant kit."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "vendor" / "rxai"
CLIENT_PY = KIT / "client.py"
PREDICTIONS_PY = KIT / "predictions.py"
SCORE_PY = KIT / "score.py"


@cache
def client():
    """Load ``vendor/rxai/client.py`` without installing it as a package.

    Importing the module does not open a socket. ``build_message`` and
    ``parse_question_date`` are used to size evidence packs; ``main`` is what
    the online runner executes in a subprocess.
    """

    if not CLIENT_PY.is_file():
        raise FileNotFoundError(f"vendored Reactor client is missing: {CLIENT_PY}")
    spec = importlib.util.spec_from_file_location("rxai_client", CLIENT_PY)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {CLIENT_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rxai_client"] = module
    spec.loader.exec_module(module)
    return module
