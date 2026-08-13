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

from pathlib import Path

from django.core.management.base import BaseCommand

from control import VehicleLimits
from controller.config import ControllerConfig
from pipeline import ParkingPipeline, PipelineConfig
from tools.run_recorder import RunRecorder


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
        parser.add_argument("--calibration", default=None,
                            help="tools/calibrate_camera.py 로 저장한 JSON 경로")
        parser.add_argument("--control-mode", choices=["waypoint-auto", "auto-host"],
                            default="waypoint-auto",
                            help="auto-host: WAYPOINT/GO 없이 host 내부 waypoint + "
                                 "DIRECT_CONTROL 만 사용 (현재 1대 전용). "
                                 "지정 시 --direct-control 은 자동으로 켜진다")
        parser.add_argument("--direct-control", action="store_true",
                            help="B안 주행 제어: 노트북이 throttle/steering 을 계산해 "
                                 "DIRECT_CONTROL 로 내려보낸다 (기본 꺼짐)")
        parser.add_argument("--max-throttle", type=float, default=None,
                            help="제어값 상한 (실차 튜닝용). auto-host 기본 %.2f, "
                                 "waypoint-auto 기본 %.2f"
                                 % (ControllerConfig.max_throttle,
                                    VehicleLimits.max_throttle))
        parser.add_argument("--steering-sign", type=float, default=None,
                            choices=[1.0, -1.0],
                            help="wire 조향 부호. 실차 확인값은 -1 (음수 = 좌회전)")
        parser.add_argument("--manual", action="store_true",
                            help="수동 계측 모드: 슬롯 배정·자동 주행을 하지 않고 "
                                 "WASD 창으로 직접 몬다. 카메라 pose 는 계속 "
                                 "기록되므로 선회 반경·속도 실측에 쓴다")
        parser.add_argument("--turn-radius", type=float, default=None, metavar="CM",
                            help="경로 계획에 쓸 최소 선회 반경(cm). 기본 61 (실측). "
                                 "왼쪽 아래에서 우회전 진입을 시험하려면 낮춰 잡는다 "
                                 "— 차가 실제로 못 도는 반경을 주면 원호 바깥으로 밀린다")
        parser.add_argument("--strong-turn-throttle", type=float, default=None,
                            metavar="V",
                            help="최대 조향(|steering|>0.5)에서 쓸 throttle 하한. "
                                 "기본 0.70. 0 을 주면 끈다 — 펌웨어 duty 가 "
                                 "38~40 에 묶여 차가 안 움직일 수 있다")
        parser.add_argument("--bbox-offset", default=None, metavar="X,Y",
                            help="탐지 bbox 중심 → 차량 기준점 보정(cm). "
                                 "2클래스 모델은 rc_car 박스만 쓰므로 전방 쿠션 "
                                 "길이만큼 뒤로 치우친다 (예: 4,0)")
        parser.add_argument("--no-reverse", action="store_true",
                            help="후진 복구를 끈다 (auto-host 기본은 켜짐). "
                                 "실차에서 후진이 위험할 때만 사용")
        parser.add_argument("--record", default=None, metavar="DIR",
                            help="Run 단위 실차 기록을 남길 상위 디렉터리 "
                                 "(예: runs). run_YYYYMMDD_HHMMSS/ 가 생성된다")

    def handle(self, *args, **options) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        camera: int | str = options["camera"]
        if isinstance(camera, str) and camera.isdigit():
            camera = int(camera)

        homography_src = None
        lot_w = lot_h = 1200.0
        if options["calibration"]:
            import json
            data = json.loads(Path(options["calibration"]).read_text(encoding="utf-8"))
            homography_src = [tuple(p) for p in data["homography_src"]]
            lot_w, lot_h = data["lot_width_mm"], data["lot_height_mm"]
            self.stdout.write(f"캘리브레이션 로드: {options['calibration']} "
                              f"({lot_w:.0f}x{lot_h:.0f}mm)")

        if not os.getenv("REDIS_URL"):
            self.stdout.write(self.style.WARNING(
                "REDIS_URL 미설정 — 대시보드 브로드캐스트가 웹 서버에 전달되지 않습니다 "
                "(채널 레이어가 프로세스 내부 전용)."
            ))

        limits = VehicleLimits(
            **{k: v for k, v in (
                ("max_throttle", options["max_throttle"]),
                ("steering_sign", options["steering_sign"]),
            ) if v is not None}
        )
        auto_host = options["control_mode"] == "auto-host"
        direct_control = options["direct_control"] or auto_host

        # auto-host 는 VehicleLimits 를 쓰지 않는다 (그쪽은 waypoint-auto 전용
        # WaypointController 의 설정이다). 같은 CLI 값을 실제 제어기가 읽는
        # ControllerConfig 로도 넘겨야 --max-throttle 이 효력을 갖는다.
        controller_config = ControllerConfig(
            # 후진 복구(parking/recovery.py)가 실제로 나가려면 여기서 열어야 한다.
            # phase 게이트(reverse_allowed_phases)가 CRUISE/TURN 을 계속 막는다.
            allow_reverse=not options["no_reverse"],
            **({} if options["strong_turn_throttle"] is None else {
                "strong_turn_min_throttle": (
                    None if options["strong_turn_throttle"] <= 0
                    else options["strong_turn_throttle"])}),
            **{k: v for k, v in (
                ("max_throttle", options["max_throttle"]),
                ("wire_steering_sign", options["steering_sign"]),
            ) if v is not None}
        ) if auto_host else ControllerConfig(allow_reverse=not options["no_reverse"])

        if auto_host:
            self.stdout.write(self.style.WARNING(
                "AUTO_HOST 모드 — WAYPOINT/GO 를 보내지 않습니다. "
                "충돌 회피는 비활성이므로 차량 1대로만 운용하세요."
            ))
            self.stdout.write(self.style.WARNING(
                f"AUTO_HOST 제어값 — throttle 상한 {controller_config.max_throttle:.2f} "
                f"(정밀주차 구간 {controller_config.parking_max_throttle}), "
                f"wire_steering_sign {controller_config.wire_steering_sign:+.0f}. "
                "ESP32 의 ENABLE_ACTUATOR_OUTPUT 을 먼저 확인하세요."
            ))
        elif direct_control:
            self.stdout.write(self.style.WARNING(
                f"B안 주행 제어 켜짐 — throttle 상한 {limits.max_throttle:.2f}, "
                f"steering_sign {limits.steering_sign:+.0f}. "
                "ESP32 의 ENABLE_ACTUATOR_OUTPUT 이 0 인지 먼저 확인하세요."
            ))

        bbox_offset = (0.0, 0.0)
        if options["bbox_offset"]:
            bx, by = (float(v) * 10.0 for v in options["bbox_offset"].split(","))
            bbox_offset = (bx, by)
            self.stdout.write(f"bbox 보정: {bx:.0f},{by:.0f}mm")

        pipeline = ParkingPipeline(PipelineConfig(
            camera_source=camera,
            bbox_offset_mm=bbox_offset,
            weights_path=options["weights"],
            confidence_threshold=options["conf"],
            imgsz=options["imgsz"],
            server_port=options["port"],
            homography_src=homography_src,
            lot_width_mm=lot_w,
            lot_height_mm=lot_h,
            control_mode=options["control_mode"],
            direct_control=direct_control,
            plan_turn_radius_mm=(None if options["turn_radius"] is None
                                 else options["turn_radius"] * 10.0),
            vehicle_limits=limits,
            controller_config=controller_config,
            manual_only=options["manual"],
        ))
        recorder = self._start_recorder(pipeline, options, controller_config,
                                        lot_w, lot_h)
        pipeline.start()
        self.stdout.write(self.style.SUCCESS(
            f"vehicle server on :{pipeline.server.bound_port} — 카메라 루프 시작 (Ctrl+C 종료)"
        ))
        outcome = "OK"
        try:
            if options["manual"]:
                self._run_manual(pipeline, options)
            else:
                pipeline.run_camera(max_frames=options["max_frames"],
                                    show=options["show"])
        except KeyboardInterrupt:
            outcome = "ABORTED"
            self.stdout.write("중단 요청 — 정리 중")
        except Exception:
            outcome = "ERROR"
            raise
        finally:
            if recorder is not None:
                summary = recorder.stop(outcome=outcome)
                self.stdout.write(self.style.SUCCESS(
                    f"Run 기록 저장: {recorder.dir} "
                    f"(pose {summary.get('pose_rows', 0)}행, "
                    f"control {summary.get('control_rows', 0)}행)"
                ))
            pipeline.stop()

    def _run_manual(self, pipeline, options) -> None:
        """수동 계측: 카메라 루프는 스레드, WASD 창은 메인 스레드.

        tkinter 는 macOS 에서 메인 스레드만 쓸 수 있고 cv2.imshow 도 마찬가지라
        둘을 같이 띄울 수 없다. 계측에는 화면 미리보기가 필요 없으므로 카메라
        쪽을 show=False 로 돌린다 — pose 기록은 그대로 된다.
        """
        import threading
        from control.hybrid_gui import run_gui

        self.stdout.write(self.style.WARNING(
            "수동 계측 모드 — 슬롯 배정·자동 주행을 하지 않습니다. "
            "WASD 로 몰고, 창을 닫으면 종료됩니다. (미리보기는 꺼집니다)"
        ))
        cam = threading.Thread(
            target=pipeline.run_camera,
            kwargs={"max_frames": options["max_frames"], "show": False},
            name="camera-loop", daemon=True)
        cam.start()
        try:
            run_gui(pipeline, car_id=1)
        finally:
            pipeline.stop()          # 카메라 루프도 같이 내린다
            cam.join(timeout=2.0)

    def _start_recorder(self, pipeline, options, controller_config,
                        lot_w: float, lot_h: float):
        """--record 가 있으면 Run 기록기를 붙인다 (요청문 6~11절).

        기록기는 차량 1대 기준이라 auto_hosts 에서 첫 러너를 집는다.
        auto-host 가 아니면 러너가 없어 미션 관련 칸이 비므로 경고만 남긴다.
        """
        if not options["record"]:
            return None
        if options["control_mode"] != "auto-host":
            self.stdout.write(self.style.WARNING(
                "--record 는 auto-host 기준으로 만들어졌습니다. "
                "waypoint-auto 에서는 미션/제어 칸이 비어 있습니다."
            ))

        cfg = controller_config or ControllerConfig()
        recorder = RunRecorder(
            options["record"], pipeline.server, car_id=1,
            pose_provider=lambda: pipeline.last_pose_rec,
            runner_provider=lambda: next(iter(pipeline.auto_hosts.values()), None),
            control_period_s=pipeline.config.auto_host_period_s,
            params={
                "entrypoint": "manage.py run_pipeline",
                "control_mode": options["control_mode"],
                "max_throttle": cfg.max_throttle,
                "parking_max_throttle": cfg.parking_max_throttle,
                "wire_steering_sign": cfg.wire_steering_sign,
                "max_wire_steering": cfg.max_wire_steering,
                "steer_kp": cfg.steer_kp,
                "steer_normalize_deg": cfg.steer_normalize_deg,
                "approach_capture_tolerance_cm": cfg.approach_capture_tolerance_cm,
                "final_confirm_observations": cfg.final_confirm_observations,
                "allow_reverse": cfg.allow_reverse,
                "plan_turn_radius_mm": pipeline.config.plan_turn_radius_mm,
                "control_period_s": pipeline.config.auto_host_period_s,
                "imgsz": options["imgsz"], "conf": options["conf"],
                "weights": options["weights"],
            },
            calibration={
                "source": options["calibration"] or "full-frame",
                "lot_width_mm": lot_w, "lot_height_mm": lot_h,
            },
        )
        pipeline.on_pose_record = recorder.log_pose
        pipeline.on_route_load = lambda wps, rec: recorder.write_route(wps, recovery=rec)
        recorder.start()
        self.stdout.write(f"Run 기록: {recorder.dir}")
        return recorder
