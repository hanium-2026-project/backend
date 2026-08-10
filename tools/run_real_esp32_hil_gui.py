from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import PipelineConfig
from pipeline.runner import ParkingPipeline
from control.hybrid_gui import run_gui


def guess_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "UNKNOWN"
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--car-id", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig()
    cfg.control_mode = "auto-host"
    if args.host is not None:
        cfg.server_host = args.host
    if args.port is not None:
        cfg.server_port = args.port

    pipeline = ParkingPipeline(cfg)
    try:
        pipeline.start()
        print(f"ESP32 target: {guess_lan_ip()}:{pipeline.server.bound_port}")
        run_gui(pipeline, car_id=args.car_id)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()