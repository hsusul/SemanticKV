from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_readme_mentioned_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "run_all_experiments.py").exists()
    assert (root / "scripts" / "check_project.py").exists()
    assert (root / "configs" / "experiments").exists()


def test_check_project_script_runs():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/check_project.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "SemanticKV project check passed" in result.stdout
    assert "FastAPI app import: ok" in result.stdout

