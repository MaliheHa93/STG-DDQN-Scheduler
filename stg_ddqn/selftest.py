from __future__ import annotations

import argparse
from pathlib import Path
import unittest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-alignment regression tests")
    parser.add_argument("--verbosity", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

