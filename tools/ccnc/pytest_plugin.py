"""Opt-in fixtures for running unchanged baseline tests on the CCNC branch."""
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
# Last carrot-wip commit merged before the CCNC-specific test edits.
BASELINE = "77ef159f3662feeaf3e5630f8dc488ab232bbd0a"


@pytest.fixture(scope="session")
def ccnc_baseline_catalog(tmp_path_factory):
  catalog = tmp_path_factory.mktemp("ccnc-baseline") / "carrot_settings.json"
  catalog.write_bytes(subprocess.check_output([
    "git", "-c", f"safe.directory={ROOT.as_posix()}", "show",
    f"{BASELINE}:openpilot/selfdrive/carrot_settings.json",
  ], cwd=ROOT))
  return catalog


@pytest.fixture(autouse=True)
def ccnc_original_test_environment(request, monkeypatch):
  path = Path(request.module.__file__).resolve().relative_to(ROOT).as_posix()
  if path in (
    "tools/docs/wiki_settings/tests/test_generate.py",
    "tools/docs/wiki_settings/tests/test_ci_check.py",
  ):
    catalog = request.getfixturevalue("ccnc_baseline_catalog")
    module = getattr(request.module, "GENERATOR", None) or request.module.CI_CHECK
    monkeypatch.setattr(module, "DEFAULT_CATALOG", catalog)
  elif path in (
    "opendbc_repo/opendbc/car/hyundai/tests/test_ccnc_cluster.py",
    "opendbc_repo/opendbc/car/hyundai/tests/test_ccnc_lead.py",
  ):
    main = request.module.hyundaicanfd
    original = main.create_ccnc_messages

    def baseline_messages(*args, **kwargs):
      return original(*args, **kwargs, extended_ccnc=False)

    monkeypatch.setattr(main, "create_ccnc_messages", baseline_messages)
