"""Run protocol/catalog/job tests without a compiled openpilot driving runtime.

Only unrelated feature registration, hardware serial lookup and replay-report
parsing are replaced. Road Viewer server, dashcam discovery, upload jobs and
network transfers remain real. See ../TESTING.md.
"""
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(root))
for name in ('openpilot.selfdrive.carrot.server.features', 'openpilot.selfdrive.carrot.server.features.dashcam'):
  package = types.ModuleType(name)
  package.__path__ = [str(root.joinpath(*name.split('.')))]
  sys.modules[name] = package
hardware = types.ModuleType('openpilot.system.hardware')
hardware.HARDWARE = types.SimpleNamespace(get_serial=lambda: 'desktop-test')
hardware.TICI = False
hardware.PC = True
sys.modules[hardware.__name__] = hardware
report = types.ModuleType('openpilot.selfdrive.carrot.server.features.dashcam.report')


def unused_report(*args, **kwargs):
  raise AssertionError('Replay report parsing is outside these upload tests')


report.build_route_report = unused_report
sys.modules[report.__name__] = report

if __name__ == '__main__':
  import pytest
  targets = sys.argv[1:] or [str(Path(__file__).with_name('test_road_viewer.py'))]
  raise SystemExit(pytest.main(['-c', '/dev/null', '--rootdir=' + str(root),  # noqa: TID251 - one isolated invocation
                               '--confcutdir=' + str(root / 'openpilot/selfdrive/carrot/server'), *targets]))
