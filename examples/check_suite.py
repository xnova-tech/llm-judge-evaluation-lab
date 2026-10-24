"""Run offline regression checks through pytest with stable summary output."""

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


def main() -> None:
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        result = pytest.main(["-q", str(Path(__file__).with_name("test_analysis.py"))])
    if result != pytest.ExitCode.OK or "9 passed" not in output.getvalue():
        raise SystemExit(output.getvalue() or "offline regression suite failed")
    sys.stdout.write("offline_regression_tests=9 passed\n")


if __name__ == "__main__":
    main()
