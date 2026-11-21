"""Load a CCNC Rust extension built from the current source tree."""

from functools import cache
from hashlib import sha256
import importlib.util
import logging
from pathlib import Path
import platform
import sys
from threading import Lock
from types import ModuleType

_CORE_DIR = Path(__file__).with_name("ccnc_core")
ABI_VERSION = 12  # Keep in sync with abi_version in src/lib.rs.
_LOAD_LOCK = Lock()


def _source_hash(core_dir: Path) -> str:
  paths = ["Cargo.toml", "Cargo.lock", "build.rs"]
  paths.extend(path.relative_to(core_dir).as_posix() for path in (core_dir / "src").rglob("*") if path.is_file())
  # Keep in sync with build.rs. Each field is prefixed with its 8-byte length.
  digest = sha256()
  for path in sorted(paths):
    for field in (path.encode("utf-8"), (core_dir / path).read_bytes()):
      digest.update(len(field).to_bytes(8, "big"))
      digest.update(field)
  return digest.hexdigest()


@cache
def _load_once() -> ModuleType | None:
  try:
    arch = platform.machine()
    if sys.platform != "linux" or arch not in ("x86_64", "aarch64"):
      raise ImportError(f"unsupported platform {sys.platform}/{arch}")

    binary = _CORE_DIR / "prebuilt" / f"linux-{arch}" / "ccnc_core.abi3.so"
    if not binary.is_file():
      raise FileNotFoundError(f"missing extension {binary}")

    spec = importlib.util.spec_from_file_location("ccnc_core", binary)
    if spec is None or spec.loader is None:
      raise ImportError(f"no extension loader for {binary}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.source_hash() != _source_hash(_CORE_DIR):
      raise ImportError("source hash mismatch")
    if (version := module.abi_version()) != ABI_VERSION:
      raise ImportError(f"ABI version {version}, expected {ABI_VERSION}")
    return module
  except Exception as exc:
    logging.getLogger(__name__).warning("CCNC Rust core unavailable: %s", exc)
    return None


def load() -> ModuleType | None:
  """Cache the first result, including a failed load, across callers."""
  with _LOAD_LOCK:
    return _load_once()
