from __future__ import annotations

import sys
import time
from pathlib import Path
from queue import Empty

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = ROOT.parent.parent / "remote-direct-bridge"
if str(BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGE_DIR))

from bridge_gui import AsyncBridgeBackend  # type: ignore


def drain_status(backend: AsyncBridgeBackend) -> None:
    while True:
        try:
            connected = backend.connection_queue.get_nowait()
            print(f"[LINK] {'CONNECTED' if connected else 'DISCONNECTED'}")
        except Empty:
            break

    latest = None
    while True:
        try:
            latest = backend.status_queue.get_nowait()
        except Empty:
            break
    if latest is not None:
        keys = (
            "state",
            "mode",
            "encoder_count",
            "wait_reason",
            "last_processed_cmd_seq",
        )
        print("[STATUS]", " ".join(f"{k}={latest.get(k)}" for k in keys))


def main() -> None:
    print("=== Hanium precision DIRECT_CONTROL test ===")
    print("SAFETY: wheels must be lifted for the first test.")
    print("This tester clamps |throttle| <= 0.70 and pulse duration <= 1.5 s.")
    print()
    print("Commands:")
    print("  status")
    print("  m                         # request REMOTE_DIRECT")
    print("  d <throttle> <steer> <s>  # pulse, then automatic zero")
    print("     examples:")
    print("       d  0.12  0.0  0.5")
    print("       d  0.15  1.0  0.5")
    print("       d -0.10  0.0  0.5")
    print("  z                         # zero immediately")
    print("  q                         # zero and quit")
    print()

    backend = AsyncBridgeBackend()
    backend.start()
    time.sleep(1.0)
    drain_status(backend)

    try:
        while True:
            cmd = input("precision> ").strip()
            if not cmd:
                drain_status(backend)
                continue

            parts = cmd.split()
            op = parts[0].lower()

            if op == "status":
                drain_status(backend)
                continue

            if op == "m":
                backend.set_drive(0.0, 0.0)
                backend.mode_remote()
                time.sleep(0.9)
                drain_status(backend)
                continue

            if op == "z":
                backend.set_drive(0.0, 0.0)
                print("[ZERO] throttle=0 steering=0")
                continue

            if op == "d":
                if len(parts) != 4:
                    print("usage: d <throttle> <steer> <seconds>")
                    continue
                try:
                    throttle = float(parts[1])
                    steering = float(parts[2])
                    duration = float(parts[3])
                except ValueError:
                    print("numbers required")
                    continue

                if abs(throttle) > 0.70:
                    print("blocked: precision tester allows |throttle| <= 0.70")
                    continue
                if abs(steering) > 1.0:
                    print("blocked: steering must be within [-1.0, 1.0]")
                    continue
                if not (0.05 <= duration <= 1.5):
                    print("blocked: duration must be 0.05..1.5 seconds")
                    continue

                print(
                    f"[PULSE] throttle={throttle:+.2f} "
                    f"steering={steering:+.2f} duration={duration:.2f}s"
                )
                backend.set_drive(throttle, steering)
                time.sleep(duration)
                backend.set_drive(0.0, 0.0)
                print("[ZERO] automatic")
                time.sleep(0.2)
                drain_status(backend)
                continue

            if op in {"q", "quit", "exit"}:
                break

            print("unknown command")
    finally:
        backend.set_drive(0.0, 0.0)
        time.sleep(0.2)
        backend.shutdown()
        print("stopped safely")


if __name__ == "__main__":
    main()
