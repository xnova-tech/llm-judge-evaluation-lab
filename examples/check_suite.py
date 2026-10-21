"""Stable stdout wrapper for offline regression checks in article verify."""

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_analysis  # noqa: E402


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_analysis)
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("offline regression suite failed")
    print(f"offline_regression_tests={result.testsRun} passed")


if __name__ == "__main__":
    main()
