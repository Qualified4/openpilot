# CanfdBvmView: our SVM_STATUS_400 (0x400) copy goes out only right behind a stock 0x400 frame on bus 0 (the car sends it
# at 5 Hz, Sonata 2024 11-12 logs), at most one per stock frame, and only in the shape the car itself uses.
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py

BVM = 0x400
SONATA = bytes.fromhex('0000007000000000')  # byte 3 is always 0x70 on the Sonata 2024
IONIQ5 = bytes(8)
HDA1_PARAMS = (2 | 4 | 8, 4 | 8, 8, 2 | 4 | 8 | 32)  # Sonata 2024 (hybrid gas, long, camera SCC) and variants
HDA2_LONG = 1 | 4 | 8 | 16  # Ioniq 5 PE


class Device:
  def __init__(self, param):
    self.s = libsafety_py.libsafety
    assert self.s.set_safety_hooks(CarParams.SafetyModel.hyundaiCanfd, param) == 0
    self.s.init_tests()
    self.t = 1_000_000
    self.s.set_timer(self.t)

  def advance(self, us):
    self.t += us
    self.s.set_timer(self.t)

  def tx(self, dat, bus=0):
    """(accepted, held)"""
    pkt = libsafety_py.make_CANPacket(BVM, bus, bytes(dat))
    self.s.safety_tx_buffered_for_fwd = False
    return bool(self.s.safety_tx_hook(pkt)), bool(self.s.safety_tx_buffered_for_fwd)

  def stock(self, dat=SONATA, bus=0):
    """a stock 0x400 arrives: what panda sends right behind it, or None"""
    pkt = libsafety_py.make_CANPacket(BVM, bus, bytes(dat))
    self.s.safety_rx_hook(pkt)
    out = libsafety_py.ffi.new("CANPacket_t *")
    if not self.s.safety_rx_tx_hook(pkt, out):
      return None
    assert out[0].addr == BVM and out[0].bus == bus
    return bytes(out[0].data)[:8]


def view(base, byte4):
  dat = bytearray(base)
  dat[4] = byte4
  return bytes(dat)


def test_shape():
  for param in HDA1_PARAMS + (HDA2_LONG,):
    d = Device(param)
    for base in (SONATA, IONIQ5):
      for byte4 in range(7):
        assert d.tx(view(base, byte4)) == (True, True), (param, base.hex(), byte4)
    for bad in ('0000007007000000', '00000070ff000000', '0000007100000000', '0000000100000000', '0100007003000000',
                '0000007003000001', '0000007003010000'):
      assert d.tx(bytes.fromhex(bad)) == (False, False), (param, bad)


def test_never_sent_directly_and_only_behind_a_stock_frame():
  d = Device(HDA1_PARAMS[0])
  # no stock frame seen yet: still held, never queued on its own
  assert d.tx(view(SONATA, 3)) == (True, True)
  d.advance(150_000)
  assert d.stock() == view(SONATA, 3)
  assert d.stock() is None  # one of ours per stock frame
  # the newest host frame wins while held; still one per stock frame (never faster than the car's 5 Hz)
  d.tx(view(SONATA, 3))
  d.tx(view(SONATA, 6))
  d.advance(100_000)
  assert d.stock() == view(SONATA, 6)
  d.advance(200_000)
  assert d.stock() is None


def test_a_stale_held_frame_is_dropped():
  d = Device(HDA1_PARAMS[0])
  d.stock()
  assert d.tx(view(SONATA, 6)) == (True, True)
  d.advance(260_000)  # longer than HYUNDAI_BVM_PB_HOLD_US: the host stopped asking
  assert d.stock() is None


def test_rejected_frames_are_not_held():
  d = Device(HDA1_PARAMS[0])
  d.stock()
  assert d.tx(view(SONATA, 3), bus=1) == (False, False)  # not on the tx list
  assert d.tx(view(SONATA, 3), bus=2) == (False, False)
  d.s.set_relay_malfunction(True)
  assert d.tx(view(SONATA, 3)) == (False, False)
  d.s.set_relay_malfunction(False)
  assert d.stock() is None
  assert d.stock(bus=2) is None  # a 0x400 seen on another bus never releases anything


def test_spas_piggyback_is_untouched():
  # the 0x400 slot is separate: a held 0x400 is not sent behind SPAS2 and vice versa
  from opendbc.car.hyundai.spas import frame
  d = Device(HDA1_PARAMS[0])
  d.tx(view(SONATA, 3))
  spas2 = libsafety_py.make_CANPacket(0x16A, 0, frame(0x16A, 1, None, 0).dat)
  d.s.safety_rx_hook(spas2)
  out = libsafety_py.ffi.new("CANPacket_t *")
  assert not d.s.safety_rx_tx_hook(spas2, out)
  assert d.stock() == view(SONATA, 3)
