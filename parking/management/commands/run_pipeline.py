"""CV → RL → 통신 파이프라인 실행 진입점.

Django 설정을 로드한 상태로 돌기 때문에 대시보드 WebSocket 브로드캐스트가
함께 동작한다.

    python manage.py run_pipeline --camera 0 --weights best05.pt

주의: 대시보드로 실시간 정보를 흘리려면 `REDIS_URL` 이 설정돼 있어야 한다.
설정하지 않으면 채널 레이어가 InMemory 로 동작해 프로세스 간 전달이 되지 않고,
이 명령의 브로드캐스트가 웹 서버 쪽 WebSocket 에 도달하지 않는다.
"""

from __future__ import annotations

import logging
import os

from django.core.management.base import BaseCommand

from pipeline import ParkingPipeline, PipelineConfig


class Command(BaseCommand):
    help = "Run the camera → detection → RL → vehicle-control pipeline."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--camera", default="0",
                            help="카메라 인덱스 또는 영상 파일 경로 (기본 0)")
        parser.add_argument("--weights", default="yolo26n.pt", help="YOLO 가중치 경로")
        parser.add_argument("--port", type=int, default=5000, help="차량 TCP 서버 포트")
        parser.add_argument("--conf", type=float, default=0.4, help="탐지 신뢰도 임계값")
        parser.add_argument("--imgsz", type=int, default=1280, help="추론 해상도")
        parser.add_argument("--max-frames", type=int, default=None, help="처리할 최대 프레임")
        parser.add_argument("--show", action="store_true", help="탐지 화면 표시")

    def handle(self, *args, **options) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        camera: int | str = options["camera"]
        if isinstance(camera, str) and camera.isdigit():
            camera = int(camera)

        if not os.getenv("REDIS_URL"):
            self.stdout.write(self.style.WARNING(
                "REDIS_URL 미설정 — 대시보드 브로드캐스트가 웹 서버에 전달되지 않습니다 "
                "(채널 레이어가 프로세스 내부 전용)."
            ))

        pipeline = ParkingPipeline(PipelineConfig(
            camera_source=camera,
            weights_path=options["weights"],
            confidence_threshold=options["conf"],
            imgsz=options["imgsz"],
            server_port=options["port"],
        ))
        pipeline.start()
        self.stdout.write(self.style.SUCCESS(
            f"vehicle server on :{pipeline.server.bound_port} — 카메라 루프 시작 (Ctrl+C 종료)"
        ))
        try:
            pipeline.run_camera(max_frames=options["max_frames"], show=options["show"])
        except KeyboardInterrupt:
            self.stdout.write("중단 요청 — 정리 중")
        finally:
            pipeline.stop()
