"""
Global push-to-talk hotkey, implemented by reading raw kernel input events
via evdev rather than a display-server hook (pynput/xdotool-style global
hotkeys don't reliably work under Wayland -- evdev bypasses that entirely
and works the same under X11 or Wayland).

Requires your user to be in the 'input' group:
    sudo usermod -aG input $USER
    (log out and back in for it to take effect)
"""
import selectors
import threading

import evdev
from evdev import ecodes, list_devices

import config


class HotkeyListener(threading.Thread):
    def __init__(self, key_name, on_press, on_release):
        super().__init__(daemon=True)
        if not hasattr(ecodes, key_name):
            raise ValueError(f"'{key_name}' is not a valid evdev key name")
        self.key_code = getattr(ecodes, key_name)
        self.on_press = on_press
        self.on_release = on_release
        self._devices = self._find_keyboards()
        self._running = True

    def _find_keyboards(self):
        if config.KEYBOARD_DEVICE_PATH:
            return [evdev.InputDevice(config.KEYBOARD_DEVICE_PATH)]

        devices = []
        for path in list_devices():
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            caps = dev.capabilities().get(ecodes.EV_KEY, [])
            if self.key_code in caps:
                devices.append(dev)

        if not devices:
            raise RuntimeError(
                "No accessible keyboard device exposes the configured hotkey.\n"
                "Fixes to try:\n"
                "  1) sudo usermod -aG input $USER   (then log out & back in)\n"
                "  2) run `evtest` to find the right /dev/input/eventX and set "
                "AWF_KEYBOARD_DEVICE to it\n"
                "  3) pick a different AWF_HOTKEY that your keyboard actually has"
            )
        return devices

    def run(self):
        sel = selectors.DefaultSelector()
        for dev in self._devices:
            sel.register(dev, selectors.EVENT_READ)

        while self._running:
            for key, _ in sel.select(timeout=1):
                dev = key.fileobj
                try:
                    for event in dev.read():
                        if event.type == ecodes.EV_KEY and event.code == self.key_code:
                            if event.value == 1:  # key down
                                self.on_press()
                            elif event.value == 0:  # key up
                                self.on_release()
                except (OSError, BlockingIOError):
                    continue

    def stop(self):
        self._running = False
