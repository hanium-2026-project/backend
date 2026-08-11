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
from tools.drive_logger import DriveLogger


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
    parser.add_argument("--log", default=None,
                        help="주행 로그 CSV 경로 (예: logs/manual.csv). "
                             "카메라를 안 쓰므로 pose/목표 칸은 빈다")
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
    logger = None
    try:
        pipeline.start()
        print(f"ESP32 target: {guess_lan_ip()}:{pipeline.server.bound_port}")
        if args.log:
            logger = DriveLogger(args.log, pipeline.server, car_id=args.car_id)
            logger.start()
            print(f"주행 로그: {args.log}")
        run_gui(pipeline, car_id=args.car_id)
    finally:
        if logger is not None:
            logger.stop()
            print(f"로그 {logger.rows}행 기록: {args.log}")
        pipeline.stop()


if __name__ == "__main__":
    main()