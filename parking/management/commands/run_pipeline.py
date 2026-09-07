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

from dataclasses import asdict
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from control import VehicleLimits
from controller.config import ControllerConfig
from parking.waypoints import CAR_LENGTH_MM, CAR_WIDTH_MM
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
        parser.add_argument("--calibrate-speed", action="store_true",
                            help="throttle→속도/정지거리 실측 시퀀스를 자동으로 "
                                 "수행한다 (--manual 을 함께 켠 것과 같다). "
                                 "WASD 는 W/S 가 항상 1.0 이라 0.10/0.15/0.25 "
                                 "같은 정확한 크기를 낼 수 없어서 필요하다.")
        parser.add_argument("--calibrate-throttles", default="0.10,0.15,0.25",
                            help="측정할 throttle 크기 목록 (쉼표 구분). "
                                 "--parking-throttle 을 넘는 값은 잘라낸다.")
        parser.add_argument("--calibrate-vehicle", action="store_true",
                            help="차량 시스템 식별 전체 시퀀스를 한 번에 수행한다: "
                                 "deadband/속도 -> 정지거리 -> 조향 곡률 -> "
                                 "좌우/전후진 비대칭. primitive 마다 STOP 으로 "
                                 "끊고, 끝나면 같은 run 을 자동 분석한다.")
        parser.add_argument("--calibrate-steerings", default="0.4,-0.4,0.7,-0.7,1.0,-1.0",
                            help="--calibrate-vehicle 의 조향 명령 목록. "
                                 "부호가 좌우를 가른다.")
        parser.add_argument("--calibrate-repeats", type=int, default=1,
                            help="각 조건 반복 횟수. 첫 실차 검증은 1 을 쓴다.")
        parser.add_argument("--calibrate-steering-throttle", type=float, default=0.15,
                            help="조향 primitive 를 구동할 throttle 크기.")
        parser.add_argument("--calibrate-once", action="store_true",
                            help="첫 throttle 로 전진 primitive 한 번만 수행한다. "
                                 "6개를 연속으로 돌리기 전 배선 확인용.")
        parser.add_argument("--calibrate-seconds", type=float, default=2.5,
                            help="한 primitive 를 구동하는 시간(초)")
        parser.add_argument("--manual", action="store_true",
                            help="수동 계측 모드: 슬롯 배정·자동 주행을 하지 않고 "
                                 "WASD 창으로 직접 몬다. 카메라 pose 는 계속 "
                                 "기록되므로 선회 반경·속도 실측에 쓴다")
        parser.add_argument("--parking-mode", choices=["handoff", "rear"],
                            default="handoff",
                            help="handoff=슬롯 앞 인계(08-12 실차 성공), "
                                 "rear=후진 슬롯 진입 (현재 B1 만 검증)")
        parser.add_argument("--turn-radius", type=float, default=None, metavar="CM",
                            help="경로 계획에 쓸 최소 선회 반경(cm). 기본 61 (실측). "
                                 "왼쪽 아래에서 우회전 진입을 시험하려면 낮춰 잡는다 "
                                 "— 차가 실제로 못 도는 반경을 주면 원호 바깥으로 밀린다")
        parser.add_argument("--steer-normalize", type=float, default=None,
                            metavar="DEG",
                            help="이 heading 오차에서 조향이 포화된다 (기본 30). "
                                 "값을 낮추면 같은 오차에 더 크게 꺾는다 — 최소 "
                                 "선회반경 원호를 추종하려면 필요하다")
        parser.add_argument("--parking-throttle", type=float, default=None,
                            metavar="V",
                            help="APPROACH/ALIGN/ENTRY/FINAL 구간 throttle 상한 "
                                 "(기본 0.25). --max-throttle 은 이 구간에 "
                                 "영향을 주지 못한다")
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
        parser.add_argument("--record-video", action="store_true",
                            help="--record run 디렉터리에 annotated e2e.mp4와 "
                                 "video_frames.jsonl을 기록한다")

    def handle(self, *args, **options) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        if options["record_video"] and not options["record"]:
            raise CommandError("--record-video requires --record DIR")
        if (options["parking_mode"] == "rear"
                and options["control_mode"] != "auto-host"):
            raise CommandError(
                "rear production parking requires --control-mode auto-host "
                "(REMOTE_DIRECT); WAYPOINT_AUTO is not permitted")
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
            **({} if options["steer_normalize"] is None else {
                "steer_normalize_deg": options["steer_normalize"]}),
            # 주차 구간 상한은 전진/후진 두 값이 같이 움직여야 의미가 있다 —
            # 둘 중 작은 쪽이 이기므로 하나만 올리면 효과가 없다.
            **({} if options["parking_throttle"] is None else {
                "parking_max_throttle": options["parking_throttle"],
                "reverse_max_throttle": options["parking_throttle"]}),
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
            parking_mode=options["parking_mode"],
            plan_turn_radius_mm=(None if options["turn_radius"] is None
                                 else options["turn_radius"] * 10.0),
            vehicle_limits=limits,
            controller_config=controller_config,
            manual_only=(options["manual"] or options["calibrate_speed"]
                         or options["calibrate_vehicle"]),
        ))
        recorder = self._start_recorder(pipeline, options, controller_config,
                                        lot_w, lot_h)
        pipeline.start()
        self.stdout.write(self.style.SUCCESS(
            f"vehicle server on :{pipeline.server.bound_port} — 카메라 루프 시작 (Ctrl+C 종료)"
        ))
        outcome = "OK"
        try:
            if options["calibrate_vehicle"]:
                self._run_vehicle_calibration(pipeline, options, recorder)
            elif options["calibrate_speed"]:
                self._run_speed_calibration(pipeline, options)
            elif options["manual"]:
                self._run_manual(pipeline, options)
            else:
                pipeline.run_camera(max_frames=options["max_frames"],
                                    show=options["show"],
                                    frame_sink=(recorder.log_video_frame
                                                if options["record_video"] else None))
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
                video = summary.get("video") or {}
                if options["record_video"]:
                    if video.get("path"):
                        self.stdout.write(self.style.SUCCESS(
                            f"Video saved: {video['path']}"))
                    else:
                        self.stdout.write(self.style.ERROR(
                            f"Video recording failed: "
                            f"{video.get('error') or 'no frames recorded'}"))
            pipeline.stop()

    def _run_speed_calibration(self, pipeline, options) -> None:
        """throttle 크기별 직진/후진 + 정지거리 실측 시퀀스.

        새 actuator 경로를 만들지 않는다. WASD 창이 쓰는 것과 **똑같은**
        production 경로(set_manual_drive -> HybridControlMux.set_manual_wire ->
        HostController.tick -> VehicleServerDirectSender -> REMOTE_DIRECT)로
        내려간다. 다른 점은 입력이 키보드가 아니라 정해진 숫자라는 것뿐이다.

        WASD 로는 이 측정을 할 수 없다: control.wasd_logic.compute_throttle 이
        0.0 / +1.0 / -1.0 만 돌려주기 때문에 0.10 같은 크기를 낼 방법이 없다.

        안전: primitive 마다 zero -> 물리적 정지 확인을 끼우고, 경계에 다가가면
        그 primitive 를 즉시 끝낸다. throttle 은 --parking-throttle 로 자른다.
        기존 boundary/COMM/stale 계약은 그대로 살아 있다 (manual_only 는 슬롯
        배정만 건너뛴다).
        """
        import threading
        import time as _time

        from parking.waypoints import _path_clearance

        car_id = 1
        cap = abs(float(options["parking_throttle"]))
        try:
            levels = [abs(float(v)) for v in
                      str(options["calibrate_throttles"]).split(",") if v.strip()]
        except ValueError:
            raise CommandError("--calibrate-throttles 는 숫자 목록이어야 합니다")
        levels = [min(v, cap) for v in levels if v > 0.0]
        if not levels:
            raise CommandError("측정할 throttle 이 없습니다")
        drive_s = max(0.3, float(options["calibrate_seconds"]))
        # 차체가 경계에 이만큼까지 다가오면 그 primitive 를 끝낸다. 계획용
        # 여유(35mm)보다 넉넉히 잡는다 — 사람이 지켜보는 계측이므로 보수적으로.
        stop_clearance_mm = 120.0

        self.stdout.write(self.style.WARNING(
            "속도 계측 모드 — 슬롯 배정·자동 주행 없음. "
            f"throttle {levels} × 전진/후진, 각 {drive_s:.1f}s. Ctrl+C 로 중단."))

        cam = threading.Thread(
            target=pipeline.run_camera,
            kwargs={"max_frames": options["max_frames"], "show": options["show"]},
            name="camera-loop", daemon=True)
        cam.start()

        def pose():
            """차량 pose. **track_of_car 에 의존하지 않는다.**

            _bind_car 는 차가 entry_nodes(junction/entrance) 에 있을 때만
            불린다. 계측은 통로 한가운데에서 하므로 binding 이 영영 안 생긴다 —
            실측 run_20260903_123402: node=lane_pt_3, car_id=None 인 채로 171
            프레임이 흘렀고, 계측 루프는 pose 를 못 찾아 대기만 하다 끝났다.

            계측 모드는 1대 전용이므로, 묶인 track 이 있으면 그것을 쓰고
            없으면 **관측된 track 이 정확히 하나일 때만** 그것을 쓴다.
            둘 이상이면 어느 것이 차인지 알 수 없으므로 움직이지 않는다.
            """
            bound = pipeline.track_of_car.get(car_id)
            if bound is not None and bound in pipeline.views:
                view = pipeline.views[bound]
            else:
                seen = [v for v in pipeline.views.values()
                        if v.heading_deg is not None]
                if len(seen) != 1:
                    return None
                view = seen[0]
            if view.heading_deg is None:
                return None
            return (view.position_mm[0], view.position_mm[1], view.heading_deg)

        def clearance():
            p = pose()
            return None if p is None else _path_clearance([p])[1]

        def hold_zero(seconds: float) -> None:
            end = _time.monotonic() + seconds
            while _time.monotonic() < end:
                pipeline.set_manual_drive(car_id, 0.0, 0.0)
                _time.sleep(0.1)

        try:
            # 차량 접속과 첫 pose 를 기다린다. 무엇을 기다리는지 알려준다 —
            # 조용히 60초 서 있으면 사용자는 차가 고장난 줄 안다.
            waited = 0.0
            while waited < 60.0:
                has_session = car_id in pipeline.hybrid_controls
                p = pose()
                if has_session and p is not None:
                    break
                if waited and abs(waited % 5.0) < 0.05:
                    self.stdout.write(
                        f"  대기 {waited:.0f}s — session={has_session} "
                        f"pose={'OK' if p else 'None'} "
                        f"tracks={len(pipeline.views)}")
                _time.sleep(0.1)
                waited += 0.1
            else:
                raise CommandError(
                    "차량 세션 또는 카메라 pose 를 못 받았습니다 "
                    f"(session={car_id in pipeline.hybrid_controls}, "
                    f"tracks={len(pipeline.views)})")
            pipeline.switch_to_manual(car_id)
            hold_zero(1.0)

            if options["calibrate_once"]:
                # 첫 실차 검증용: +throttle 전진 한 번만. 6개 primitive 를
                # 연속으로 돌리기 전에 배선이 살아 있는지부터 확인한다.
                levels = levels[:1]
                directions = ((1.0, "FORWARD"),)
                self.stdout.write("  단일 primitive 모드 (전진 1회만)")
            else:
                directions = ((1.0, "FORWARD"), (-1.0, "REVERSE"))

            for level in levels:
                for sign, label in directions:
                    start = pose()
                    self.stdout.write(
                        f"  throttle {sign * level:+.2f} {label} … "
                        f"start=({start[0]:.0f},{start[1]:.0f})")
                    end = _time.monotonic() + drive_s
                    reason = "duration"
                    while _time.monotonic() < end:
                        c = clearance()
                        if c is not None and c < stop_clearance_mm:
                            reason = f"boundary({c:.0f}mm)"
                            break
                        pipeline.set_manual_drive(car_id, sign * level, 0.0)
                        _time.sleep(0.05)
                    pipeline.set_manual_drive(car_id, 0.0, 0.0)
                    hold_zero(3.0)          # 타행이 끝날 때까지 zero 유지
                    fin = pose()
                    self.stdout.write(
                        f"      stop=({fin[0]:.0f},{fin[1]:.0f}) [{reason}]")
        except KeyboardInterrupt:
            self.stdout.write("계측 중단 — 정지")
            raise
        finally:
            try:
                pipeline.manual_stop(car_id)
            except Exception:               # noqa: BLE001
                pass
            pipeline.stop()
            cam.join(timeout=2.0)

    def _run_vehicle_calibration(self, pipeline, options, recorder) -> None:
        """차량 시스템 식별 전체 시퀀스 (단일 명령).

        구동은 WASD 창과 **같은** production 경로다
        (set_manual_drive -> HybridControlMux -> HostController ->
        REMOTE_DIRECT). 여기서 더하는 것은 orchestration 과 기록뿐이고,
        안전 판정은 tools.calibration_sequence 가 갖고 있다 — 그쪽은 실차
        없이 경계/무동작/정지/통신/중단 경로가 전부 단위 테스트되어 있다.
        """
        import csv as _csv
        import json as _json
        import math as _math
        import threading
        import time as _time

        from parking.waypoints import _path_clearance
        from tools.calibration_sequence import (CALIBRATION_SCHEMA_VERSION,
                                                CalibrationAborted,
                                                CalibrationLimits,
                                                CalibrationSequence,
                                                build_plan)

        car_id = 1
        cap = abs(float(options["parking_throttle"]))

        def _floats(raw, what):
            try:
                return [float(v) for v in str(raw).split(",") if v.strip()]
            except ValueError:
                raise CommandError(f"{what} 는 숫자 목록이어야 합니다")

        throttles = [min(abs(v), cap) for v in
                     _floats(options["calibrate_throttles"],
                             "--calibrate-throttles") if v]
        steerings = [max(-1.0, min(1.0, v)) for v in
                     _floats(options["calibrate_steerings"],
                             "--calibrate-steerings")]
        repeats = max(1, int(options["calibrate_repeats"]))
        if not throttles:
            raise CommandError("측정할 throttle 이 없습니다")
        limits = CalibrationLimits(max_throttle=cap)
        plan = build_plan(throttles, steerings, repeats)

        self.stdout.write(self.style.WARNING(
            "차량 계측 모드 - 슬롯 배정/자동 주행 없음. "
            f"primitive {len(plan)}개 (throttle {throttles}, "
            f"steering {steerings}, repeat {repeats}). Ctrl+C 로 즉시 중단."))

        cam = threading.Thread(
            target=pipeline.run_camera,
            kwargs={"max_frames": options["max_frames"], "show": options["show"]},
            name="camera-loop", daemon=True)
        cam.start()

        run_dir = getattr(recorder, "dir", None)
        events_file = None
        if run_dir is not None:
            events_file = (run_dir / "calibration_events.jsonl").open(
                "w", buffering=1, encoding="utf-8")

        def emit(event: dict) -> None:
            # recorder 가 source of truth. 기록 실패가 구동/정지를 막지 않는다.
            if events_file is None:
                return
            try:
                events_file.write(_json.dumps(event, ensure_ascii=False) + "\n")
            except Exception:                       # noqa: BLE001
                pass

        def pose():
            """계측 pose. binding 에 의존하지 않는다 (run_20260903_123402).

            그리고 신뢰 못 하는 heading 으로는 계측하지 않는다 — 측정
            무결성과 안전이 같은 방향이다.
            """
            bound = pipeline.track_of_car.get(car_id)
            if bound is not None and bound in pipeline.views:
                view = pipeline.views[bound]
            else:
                seen = [v for v in pipeline.views.values()
                        if v.heading_deg is not None]
                if len(seen) != 1:
                    return None
                view = seen[0]
            if (view.heading_deg is None
                    or view.heading_source not in ("FRONT_CUSHION",
                                                   "TRAJECTORY")):
                return None
            return (view.position_mm[0], view.position_mm[1], view.heading_deg)

        def clearance():
            p = pose()
            return None if p is None else _path_clearance([p])[1]

        def comm_ok() -> bool:
            return car_id not in getattr(pipeline, "_comm_lost", set())

        sequence = CalibrationSequence(
            drive=lambda t, s: pipeline.set_manual_drive(car_id, t, s),
            pose=pose, clearance=clearance, comm_ok=comm_ok,
            now=_time.monotonic, sleep=_time.sleep,
            emit=emit, log=lambda m: self.stdout.write(m),
            limits=limits,
            steering_throttle=float(options["calibrate_steering_throttle"]))

        aborted = ""
        try:
            waited = 0.0
            while waited < 60.0:
                if car_id in pipeline.hybrid_controls and pose() is not None:
                    break
                if waited and abs(waited % 5.0) < 0.05:
                    self.stdout.write(
                        f"  대기 {waited:.0f}s - "
                        f"session={car_id in pipeline.hybrid_controls} "
                        f"pose={'OK' if pose() else 'None'} "
                        f"tracks={len(pipeline.views)}")
                _time.sleep(0.1)
                waited += 0.1
            else:
                raise CommandError(
                    "차량 세션 또는 신뢰 가능한 pose 를 못 받았습니다 "
                    f"(session={car_id in pipeline.hybrid_controls}, "
                    f"tracks={len(pipeline.views)})")
            pipeline.switch_to_manual(car_id)

            if run_dir is not None:
                manifest = {
                    "schema_version": CALIBRATION_SCHEMA_VERSION,
                    "throttles": throttles, "steerings": steerings,
                    "repeats": repeats, "max_throttle": cap,
                    "steering_throttle": float(
                        options["calibrate_steering_throttle"]),
                    "limits": dict(limits.__dict__),
                    "start_pose": pose(),
                    "weights": options.get("weights"),
                    "calibration": options.get("calibration"),
                    "primitive_count": len(plan),
                }
                (run_dir / "calibration_manifest.json").write_text(
                    _json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8")

            sequence.run(plan)
        except CalibrationAborted as exc:
            aborted = exc.reason
            self.stdout.write(self.style.WARNING(f"계측 중단: {exc}"))
        except KeyboardInterrupt:
            aborted = "USER_ABORT"
            self.stdout.write("계측 중단 - 정지")
            raise
        finally:
            # 순서가 중요하다: 먼저 차를 세우고, 그 다음에 분석한다.
            try:
                pipeline.manual_stop(car_id)
            except Exception:                       # noqa: BLE001
                pass
            if events_file is not None:
                events_file.close()
            self.stdout.write(
                f"primitive {len(sequence.results)}/{len(plan)} 수행"
                + (f" (중단: {aborted})" if aborted else ""))
            self._calibration_summary(run_dir, sequence.results,
                                      _csv, _json, _math)
            pipeline.stop()
            cam.join(timeout=2.0)

    def _calibration_summary(self, run_dir, results, _csv, _json, _math) -> None:
        """차가 멈춘 뒤에만 부른다. 분석 실패가 계측 결과를 망치지 않는다."""
        if not results:
            return
        try:
            rows = [r.as_event() for r in results]
            moved = [r for r in results if r.motion_detected]
            still = [r for r in results if not r.motion_detected]
            self.stdout.write("")
            self.stdout.write("=== DEADBAND / NO-MOTION ===")
            for r in still:
                self.stdout.write(
                    f"  {r.direction:<7} thr={r.requested_throttle:.2f} "
                    f"steer={r.requested_steering:+.1f} "
                    f"move={r.displacement_mm:5.1f}mm  {r.termination_reason}")
            if not still:
                self.stdout.write("  (없음 - 모든 조건에서 움직였습니다)")
            self.stdout.write("=== MOTION ===")
            for r in moved:
                secs = max(1e-6, (r.zero_t or 0.0) - (r.command_start_t or 0.0))
                extra = ""
                if r.kind == "ARC" and abs(r.heading_change_deg) > 1e-6:
                    radius = r.displacement_mm / abs(
                        _math.radians(r.heading_change_deg))
                    extra = f" R~{radius:6.0f}mm"
                self.stdout.write(
                    f"  {r.phase:<8} {r.direction:<7} "
                    f"thr={r.requested_throttle:.2f} "
                    f"steer={r.requested_steering:+.1f} "
                    f"v~{r.displacement_mm / secs:6.1f}mm/s "
                    f"coast={r.coast_mm:5.1f}mm{extra}")
            if run_dir is not None:
                (run_dir / "vehicle_dynamics_summary.json").write_text(
                    _json.dumps(rows, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                with (run_dir / "calibration_samples.csv").open(
                        "w", newline="", encoding="utf-8") as fh:
                    writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                self.stdout.write(
                    f"calibration 결과: {run_dir} "
                    "(calibration_events.jsonl / calibration_manifest.json / "
                    "vehicle_dynamics_summary.json / calibration_samples.csv)")
        except Exception as exc:                    # noqa: BLE001
            self.stdout.write(self.style.WARNING(
                f"요약 생성 실패(원시 기록은 무사): {exc}"))

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
            lifecycle_provider=lambda: pipeline.lifecycle_snapshot(1),
            control_period_s=pipeline.config.auto_host_period_s,
            params={
                "entrypoint": "manage.py run_pipeline",
                "control_mode": options["control_mode"],
                "max_throttle": cfg.max_throttle,
                "parking_max_throttle": cfg.parking_max_throttle,
                "wire_steering_sign": cfg.wire_steering_sign,
                "max_wire_steering": cfg.max_wire_steering,
                "steer_kp": cfg.steer_kp,
                "steer_kd": cfg.steer_kd,
                "steer_normalize_deg": cfg.steer_normalize_deg,
                "approach_capture_tolerance_cm": cfg.approach_capture_tolerance_cm,
                "final_confirm_observations": cfg.final_confirm_observations,
                "allow_reverse": cfg.allow_reverse,
                "plan_turn_radius_mm": pipeline.config.plan_turn_radius_mm,
                "control_period_s": pipeline.config.auto_host_period_s,
                "imgsz": options["imgsz"], "conf": options["conf"],
                "weights": options["weights"],
                "record_video": bool(options["record_video"]),
                "video_fps": 4.0 if options["record_video"] else None,
                "vehicle_length_mm": CAR_LENGTH_MM,
                "vehicle_width_mm": CAR_WIDTH_MM,
                "runtime_controller_config": asdict(cfg),
            },
            calibration={
                "source": options["calibration"] or "full-frame",
                "lot_width_mm": lot_w, "lot_height_mm": lot_h,
            },
        )
        pipeline.on_pose_record = recorder.log_pose
        pipeline.on_event_record = recorder.event
        pipeline.on_route_load = lambda wps, rec: recorder.write_route(wps, recovery=rec)
        recorder.start()
        if options["record_video"]:
            recorder.start_video(fps=4.0)
        self.stdout.write(f"Run 기록: {recorder.dir}")
        return recorder
