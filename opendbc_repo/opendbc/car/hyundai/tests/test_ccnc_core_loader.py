import importlib.util
from pathlib import Path
import platform
import shutil

import pytest

from opendbc.car.hyundai import ccnc_core_loader

CORE_DIR = Path(ccnc_core_loader.__file__).with_name("ccnc_core")
BINARY_PATH = Path("prebuilt") / f"linux-{platform.machine()}" / "ccnc_core.abi3.so"


@pytest.fixture
def copied_loader(tmp_path):
  core_copy = tmp_path / "ccnc_core"
  shutil.copytree(CORE_DIR / "src", core_copy / "src")
  for name in ("Cargo.toml", "Cargo.lock", "build.rs"):
    shutil.copyfile(CORE_DIR / name, core_copy / name)
  binary = core_copy / BINARY_PATH
  binary.parent.mkdir(parents=True)

  loader_path = tmp_path / "ccnc_core_loader.py"
  shutil.copyfile(ccnc_core_loader.__file__, loader_path)
  spec = importlib.util.spec_from_file_location("copied_ccnc_core_loader", loader_path)
  assert spec is not None and spec.loader is not None
  loader = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(loader)
  return loader, binary


def test_loads_built_extension():
  module = ccnc_core_loader.load()
  assert module is not None
  assert Path(module.__file__).resolve() == (CORE_DIR / BINARY_PATH).resolve()
  assert module.source_hash() == ccnc_core_loader._source_hash(CORE_DIR)
  assert module.abi_version() == ccnc_core_loader.ABI_VERSION == 2


def test_rust_panic_is_an_ordinary_exception():
  # The display catches Exception; PyO3's PanicException is a BaseException and would end card.
  with pytest.raises(RuntimeError, match="panicked"):
    ccnc_core_loader.load()._panic_for_test()


def test_rejects_changed_source(copied_loader, tmp_path, caplog):
  loader, binary = copied_loader
  shutil.copyfile(CORE_DIR / BINARY_PATH, binary)
  with (tmp_path / "ccnc_core" / "src" / "lib.rs").open("ab") as source:
    source.write(b"\n// Changed since the extension was built.\n")

  assert loader.load() is None
  assert caplog.messages == ["CCNC Rust core unavailable: source hash mismatch"]


def test_caches_missing_extension(copied_loader, caplog):
  loader, binary = copied_loader
  assert loader.load() is None
  shutil.copyfile(CORE_DIR / BINARY_PATH, binary)
  assert loader.load() is None
  assert len(caplog.messages) == 1
  assert "missing extension" in caplog.messages[0]


def test_rejects_empty_extension(copied_loader, caplog):
  loader, binary = copied_loader
  binary.touch()
  assert loader.load() is None
  assert len(caplog.messages) == 1
  assert str(binary) in caplog.messages[0]


def test_caches_loaded_module(copied_loader, tmp_path):
  loader, binary = copied_loader
  shutil.copyfile(CORE_DIR / BINARY_PATH, binary)
  module = loader.load()
  assert module is not None
  binary.unlink()
  (tmp_path / "ccnc_core" / "Cargo.lock").unlink()
  assert loader.load() is module
