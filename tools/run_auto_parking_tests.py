#!/usr/bin/env python3
"""AUTO_HOST / Recovery / 자동주차 관련 독립 테스트 러너.

Django/gymnasium/YOLO 없이 controller + host_control + integration 계층만 검증한다.
프로젝트 전체 환경이 아직 준비되지 않은 PC에서도 이번 자동주차 변경사항을 빠르게
검증하기 위한 스모크/회귀 테스트 진입점이다.

사용:
    python tools/run_auto_parking_tests.py
    python tools/run_auto_parking_tests.py -v
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIRS = (
    ROOT / "controller" / "tests",
    ROOT / "host_control" / "tests",
    ROOT / "integration" / "tests",
)


def build_suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_dir in TEST_DIRS:
        suite.addTests(loader.discover(str(test_dir), pattern="test*.py"))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(build_suite())
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
