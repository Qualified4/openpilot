"""Run CCNC checks without modifying the original test suites."""
import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("suite", choices=("docs", "display", "vehicle", "all"))
  args = parser.parse_args()
  os.chdir(ROOT)
  sys.path[:0] = [str(ROOT), str(ROOT / "opendbc_repo")]
  os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
  import pytest

  paths = []
  if args.suite in ("docs", "all"):
    paths.append("tools/docs/wiki_settings/tests")
  if args.suite in ("display", "all"):
    paths.append("tools/ccnc/test_radar_display.py")
  if args.suite in ("vehicle", "all"):
    paths.extend(f"opendbc_repo/opendbc/car/hyundai/tests/test_ccnc_{name}.py"
                 for name in ("cluster", "lead", "fault_filter", "extension"))
  return pytest.main(["-c", str(Path(__file__).with_name("pytest.ini")), "--noconftest",
                      "-p", "tools.ccnc.pytest_plugin", "-q", *paths])


if __name__ == "__main__":
  raise SystemExit(main())
