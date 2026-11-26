"""Verify zero PyTorch dependency in core inference, routing, and API packages."""

from __future__ import annotations

import subprocess
import sys


def test_zero_torch_import_in_runtime_packages() -> None:
    code = (
        "import sys; "
        "import medjudge, medjudge.schemas, medjudge.rubrics, medjudge.slicers, "
        "medjudge.tier0, medjudge.tier1, medjudge.tier2, medjudge.tier3, "
        "medjudge.router, medjudge.audit, medjudge.drift, medjudge.api; "
        "assert 'torch' not in sys.modules, 'torch must not be imported in runtime packages!'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
