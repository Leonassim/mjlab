"""Asynchronous gamepad input client.

Adapted from Thomas Flayols (LAAS CNRS),
https://github.com/thomasfla/solopython

Run `python -m mjlab.utils.gamepad_client` to display raw values.
"""

import array
import fcntl
import struct
import time
from ctypes import c_bool, c_double
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from typing import Any

import inputs

# Axis ranges are asked of the kernel, not assumed. Pads disagree: Xbox
# 360/One (xpad) reports ABS_X/Y/RX/RY as signed 16-bit, DualShock 3/4
# (hid-sony, hid-playstation) reports 0..255. Hardcoding either one pins the
# other's sticks to centre -- (raw + 32768) / 65535 maps a DualShock's whole
# 0..255 travel into [0.500, 0.504], which reads as a dead stick.
_FALLBACK_MIN = -32768
_FALLBACK_MAX = 32767

# linux/input-event-codes.h
_ABS_CODES = {
  "ABS_X": 0x00,
  "ABS_Y": 0x01,
  "ABS_Z": 0x02,
  "ABS_RX": 0x03,
  "ABS_RY": 0x04,
  "ABS_RZ": 0x05,
}
# EVIOCGABS(abs) = _IOR('E', 0x40 + abs, struct input_absinfo), and
# input_absinfo is 6 signed 32-bit fields: value, min, max, fuzz, flat, res.
_ABSINFO_SIZE = 24


def _probe_axis_ranges(path: str) -> dict[str, tuple[int, int]]:
  """Real (min, max) per absolute axis, straight from the driver.

  Axes the pad does not carry answer min == max, which is how the right-stick
  choice below tells ABS_RX/RY from a DualShock's ABS_Z/RZ. Returns {} if the
  device cannot be opened, leaving the caller on the fallback range.
  """
  ranges: dict[str, tuple[int, int]] = {}
  try:
    with open(path, "rb") as fd:
      for name, code in _ABS_CODES.items():
        req = (2 << 30) | (_ABSINFO_SIZE << 16) | (ord("E") << 8) | (0x40 + code)
        buf = array.array("b", bytes(_ABSINFO_SIZE))
        try:
          fcntl.ioctl(fd, req, buf, True)
        except OSError:
          continue
        _, lo, hi, _, _, _ = struct.unpack("6i", buf.tobytes())
        if hi > lo:
          ranges[name] = (lo, hi)
  except OSError:
    return {}
  return ranges


def _pick_right_stick(ranges: dict[str, tuple[int, int]]) -> tuple[str, str]:
  """ABS_RX/RY when the pad has them, else ABS_Z/RZ.

  Order matters and cannot be flipped: on an Xbox pad ABS_Z/RZ are the
  triggers, so probing Z/RZ first would drive yaw from the trigger pull.
  """
  if "ABS_RX" in ranges and "ABS_RY" in ranges:
    return "ABS_RX", "ABS_RY"
  if "ABS_Z" in ranges and "ABS_RZ" in ranges:
    return "ABS_Z", "ABS_RZ"
  return "ABS_RX", "ABS_RY"


def _normalize_axis(
  raw: int, lo: int = _FALLBACK_MIN, hi: int = _FALLBACK_MAX
) -> float:
  # Clamped hard: whatever the true native range of a given pad turns out to
  # be, this value feeds directly into a velocity command with no downstream
  # bound. An unclamped misread sends the policy an out-of-distribution
  # command on the very first frame.
  span = float(hi - lo) if hi > lo else 1.0
  return max(0.0, min(1.0, (raw - lo) / span))


class GamepadClient:
  # multiprocessing shared values; typed Any because SynchronizedBase
  # does not expose `.value` in type stubs.
  running: Any
  startButton: Any
  backButton: Any
  northButton: Any
  eastButton: Any
  southButton: Any
  westButton: Any
  leftJoystickX: Any
  leftJoystickY: Any
  rightJoystickX: Any
  rightJoystickY: Any
  R1Button: Any
  L1Button: Any

  def __init__(self):
    self.running = Value(c_bool, lock=True)
    self.startButton = Value(c_bool, lock=True)
    self.backButton = Value(c_bool, lock=True)
    self.northButton = Value(c_bool, lock=True)
    self.eastButton = Value(c_bool, lock=True)
    self.southButton = Value(c_bool, lock=True)
    self.westButton = Value(c_bool, lock=True)
    self.leftJoystickX = Value(c_double, lock=True)
    self.leftJoystickY = Value(c_double, lock=True)
    self.rightJoystickX = Value(c_double, lock=True)
    self.rightJoystickY = Value(c_double, lock=True)
    self.R1Button = Value(c_bool, lock=True)
    self.L1Button = Value(c_bool, lock=True)

    self.startButton.value = False
    self.backButton.value = False
    self.northButton.value = False
    self.eastButton.value = False
    self.southButton.value = False
    self.westButton.value = False
    self.leftJoystickX.value = 0.5
    self.leftJoystickY.value = 0.5
    self.rightJoystickX.value = 0.5
    self.rightJoystickY.value = 0.5
    self.R1Button.value = False
    self.L1Button.value = False

    # Probe here rather than in the child: it happens once, and the mapping it
    # picks is worth printing where the operator will see it.
    ranges: dict[str, tuple[int, int]] = {}
    try:
      pad = inputs.devices.gamepads[0]
      path = getattr(pad, "_character_device_path", None)
      if path:
        ranges = _probe_axis_ranges(path)
    except IndexError:
      print("[gamepad] no gamepad found")
    rx, ry = _pick_right_stick(ranges)
    if ranges:
      print(f"[gamepad] axis ranges from driver: {ranges}")
      print(f"[gamepad] right stick -> {rx}/{ry}")
    else:
      print(
        f"[gamepad] could not read axis ranges, assuming "
        f"{_FALLBACK_MIN}..{_FALLBACK_MAX} (Xbox-style)"
      )

    args = (
      self.running,
      self.startButton,
      self.backButton,
      self.northButton,
      self.eastButton,
      self.southButton,
      self.westButton,
      self.leftJoystickX,
      self.leftJoystickY,
      self.rightJoystickX,
      self.rightJoystickY,
      self.R1Button,
      self.L1Button,
      ranges,
      rx,
      ry,
    )
    self.process = Process(target=self.run, args=args)
    self.process.start()
    time.sleep(0.2)

  def run(
    self,
    running,
    startButton,
    backButton,
    northButton,
    eastButton,
    southButton,
    westButton,
    leftJoystickX,
    leftJoystickY,
    rightJoystickX,
    rightJoystickY,
    R1Button,
    L1Button,
    ranges,
    rxCode,
    ryCode,
  ):
    running.value = True

    def norm(code, raw):
      lo, hi = ranges.get(code, (_FALLBACK_MIN, _FALLBACK_MAX))
      return _normalize_axis(raw, lo, hi)

    while running.value:
      events = inputs.get_gamepad()
      for event in events:
        # print(event.ev_type, event.code, event.state)
        if event.ev_type == "Absolute":
          if event.code == "ABS_X":
            leftJoystickX.value = norm("ABS_X", event.state)
          if event.code == "ABS_Y":
            leftJoystickY.value = norm("ABS_Y", event.state)
          if event.code == rxCode:
            rightJoystickX.value = norm(rxCode, event.state)
          if event.code == ryCode:
            rightJoystickY.value = norm(ryCode, event.state)
        if event.ev_type == "Key":
          if event.code == "BTN_START":
            startButton.value = event.state
          elif event.code == "BTN_TR":
            R1Button.value = event.state
          elif event.code == "BTN_TL":
            L1Button.value = event.state
          elif event.code == "BTN_SELECT":
            backButton.value = event.state
          elif event.code == "BTN_NORTH":
            northButton.value = event.state
          elif event.code == "BTN_EAST":
            eastButton.value = event.state
          elif event.code == "BTN_SOUTH":
            southButton.value = event.state
          elif event.code == "BTN_WEST":
            westButton.value = event.state

  def stop(self):
    self.running.value = False
    self.process.terminate()
    self.process.join()


if __name__ == "__main__":
  gp = GamepadClient()
  for _ in range(1000):
    print("LX = ", gp.leftJoystickX.value, end=" ; ")
    print("LY = ", gp.leftJoystickY.value, end=" ; ")
    print("RX = ", gp.rightJoystickX.value, end=" ; ")
    print("RY = ", gp.rightJoystickY.value, end=" ; ")
    print("start = ", gp.startButton.value)
    print("back = ", gp.backButton.value)
    print("R1 = ", gp.R1Button.value)
    print("L1 = ", gp.L1Button.value)
    time.sleep(0.1)

  gp.stop()
