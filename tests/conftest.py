import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def build_personas():
    """以模块方式加载 scripts/build_personas.py（scripts/ 不是 package）。"""
    spec = importlib.util.spec_from_file_location(
        "build_personas", REPO_ROOT / "scripts" / "build_personas.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
