"""제어 가능한 차량 시스템 식별(system identification) 시퀀스.

이건 "자동주행" 이 아니다. **작은 bounded primitive 를 STOP 으로 분리해
순차 실행하는 측정 도구**다. 각 primitive 는 항상

    PRECHECK -> COMMAND -> OBSERVE -> ZERO -> PHYSICAL STOP -> RECORD

순서를 지키고, 어떤 경로로 끝나든 마지막 명령은 zero 다.

하드웨어를 모른다 — pose/구동/여유를 전부 콜백으로 받는다. 그래서 실차 없이
경계·정지·무동작·통신·중단 경로를 전부 단위 테스트할 수 있다. 실차 앞에서
처음 돌려보는 코드가 되면 안 되기 때문이다.

측정 대상
--------
PHASE A (STRAIGHT)  throttle -> 속도, deadband, 정지거리(타행)
PHASE B (ARC)       steering -> 선회반경, 좌우 비대칭, 전후진 비대칭

각 조건은 FORWARD 뒤에 같은 명령으로 REVERSE 를 붙여 **왔던 길을 되짚게**
한다. 이상적인 자전거 모델이면 출발점 근처로 돌아오므로 맵 공간을 아끼고,
같은 조건에서 전/후진 데이터를 함께 얻는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Sequence

CALIBRATION_SCHEMA_VERSION = 1

# 종료 사유
TERM_DISTANCE = "TARGET_DISTANCE"
TERM_HEADING = "TARGET_HEADING_CHANGE"
TERM_DURATION = "DURATION"
TERM_BOUNDARY = "BOUNDARY"
TERM_NO_MOTION = "NO_MOTION"
TERM_POSE_STALE = "POSE_STALE"
TERM_COMM_FAIL = "COMM_FAIL"
TERM_USER_ABORT = "USER_ABORT"
TERM_PRECHECK = "PRECHECK_FAILED"
TERM_DRIFT = "DRIFT_LIMIT"


@dataclass(frozen=True)
class CalibrationLimits:
    """전부 보수적으로 잡는다. 계측이라고 안전 기준을 낮추지 않는다."""

    # 직진 primitive
    straight_target_mm: float = 150.0     # 추정기 최소 창(40mm)보다 넉넉히
    straight_max_s: float = 4.0
    # 원호 primitive
    arc_target_deg: float = 25.0          # 반경 추정 최소 8도보다 넉넉히
    arc_max_s: float = 4.0
    # 무동작 판정: 이만큼 명령했는데 이만큼도 못 움직이면 deadband 후보
    no_motion_grace_s: float = 1.2
    no_motion_mm: float = 15.0            # 정지 pose 잡음(~5mm)의 3배
    # 물리적 정지 판정
    stop_noise_mm: float = 8.0
    stop_settle_s: float = 0.8
    stop_timeout_s: float = 4.0
    # 경계: 계측 중에는 더 보수적으로 (production 문턱을 낮추지 않는다)
    min_clearance_mm: float = 120.0
    # 한 F/R 쌍이 끝난 뒤 세션 시작점에서 이만큼 벗어나면 중단
    drift_limit_mm: float = 400.0
    drift_limit_deg: float = 45.0
    # 명령 갱신 주기. MANUAL_INPUT_LEASE_S(0.35s)보다 훨씬 짧아야 한다 —
    # lease 만료로 멈추게 두지 않는다.
    command_period_s: float = 0.05
    max_throttle: float = 0.25


@dataclass
class Primitive:
    primitive_id: int
    phase: str                 # SPEED | STEERING
    kind: str                  # STRAIGHT | ARC
    direction: str             # FORWARD | REVERSE
    throttle: float            # 항상 양수 크기
    steering: float            # wire 부호 그대로
    repeat: int = 1


@dataclass
class PrimitiveResult:
    primitive_id: int
    phase: str
    kind: str
    direction: str
    requested_throttle: float
    requested_steering: float
    repeat: int
    planned_duration_s: float
    command_start_t: float | None = None
    zero_t: float | None = None
    physical_stop_t: float | None = None
    end_t: float | None = None
    start_pose: tuple[float, float, float] | None = None
    command_end_pose: tuple[float, float, float] | None = None
    stop_pose: tuple[float, float, float] | None = None
    termination_reason: str = ""
    motion_detected: bool = False
    displacement_mm: float = 0.0
    heading_change_deg: float = 0.0
    coast_mm: float = 0.0
    coast_s: float = 0.0
    note: str = ""

    def as_event(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = CALIBRATION_SCHEMA_VERSION
        return d


def _heading_delta(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


def build_plan(throttles: Sequence[float], steerings: Sequence[float],
               repeats: int = 1) -> list[Primitive]:
    """PHASE A(직진) 뒤 PHASE B(원호). 각 조건은 FORWARD 다음 REVERSE 쌍.

    쌍으로 묶는 이유는 두 가지다: 되짚어 오면 맵을 적게 쓰고, 같은 명령의
    전/후진 비대칭을 바로 옆에서 얻는다.
    """
    plan: list[Primitive] = []
    pid = 0
    for rep in range(1, repeats + 1):
        for thr in throttles:
            for direction in ("FORWARD", "REVERSE"):
                pid += 1
                plan.append(Primitive(pid, "SPEED", "STRAIGHT", direction,
                                      abs(thr), 0.0, rep))
    for rep in range(1, repeats + 1):
        for steer in steerings:
            for direction in ("FORWARD", "REVERSE"):
                pid += 1
                plan.append(Primitive(pid, "STEERING", "ARC", direction,
                                      0.0, steer, rep))
    return plan


class CalibrationAborted(RuntimeError):
    """더 진행하면 안 되는 상태. 호출부는 zero 를 보장한 뒤 종료한다."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


class CalibrationSequence:
    """primitive 를 순서대로 실행한다. 하드웨어는 콜백으로만 만진다.

    drive(throttle, steering) : 한 tick 명령. 호출부가 production 경로로 보낸다.
    pose()                    : (x, y, heading) 또는 None (신뢰 불가/미관측)
    clearance()               : 차체~맵 여유 mm 또는 None
    comm_ok()                 : 통신 정상 여부
    now() / sleep(s)          : 시간
    emit(dict)                : primitive 결과 1건 기록
    log(str)                  : 사람이 보는 진행 출력
    """

    def __init__(self, *, drive, pose, clearance, now, sleep,
                 emit=None, log=None, comm_ok=None,
                 limits: CalibrationLimits | None = None,
                 steering_throttle: float = 0.15) -> None:
        self.drive = drive
        self.pose = pose
        self.clearance = clearance
        self.now = now
        self.sleep = sleep
        self.emit = emit or (lambda _e: None)
        self.log = log or (lambda _m: None)
        self.comm_ok = comm_ok or (lambda: True)
        self.limits = limits or CalibrationLimits()
        self.steering_throttle = min(abs(steering_throttle),
                                     self.limits.max_throttle)
        self.results: list[PrimitiveResult] = []
        self.session_start_pose: tuple[float, float, float] | None = None

    # 저수준 헬퍼
    def _zero_for(self, seconds: float) -> None:
        """zero 를 **계속** 보낸다. lease 만료로 멈추길 기대하지 않는다."""
        end = self.now() + seconds
        while self.now() < end:
            self.drive(0.0, 0.0)
            self.sleep(self.limits.command_period_s)
        self.drive(0.0, 0.0)

    def _require_pose(self, reason_if_missing: str):
        p = self.pose()
        if p is None:
            raise CalibrationAborted(reason_if_missing, "신뢰 가능한 pose 없음")
        return p

    def _wait_physical_stop(self, start_pose):
        """카메라 pose 가 잡음 band 안에서 일정 시간 머물면 정지로 본다.

        고정 sleep 을 믿지 않는다 — 타행 거리 자체가 측정 대상이다.
        """
        lim = self.limits
        deadline = self.now() + lim.stop_timeout_s
        anchor = start_pose
        anchor_t = self.now()
        last = start_pose
        while self.now() < deadline:
            self.drive(0.0, 0.0)
            self.sleep(lim.command_period_s)
            p = self.pose()
            if p is None:
                continue
            last = p
            if math.hypot(p[0] - anchor[0], p[1] - anchor[1]) > lim.stop_noise_mm:
                anchor, anchor_t = p, self.now()
                continue
            if self.now() - anchor_t >= lim.stop_settle_s:
                return p, True
        return last, False

    # primitive 실행
    def run_primitive(self, prim: Primitive) -> PrimitiveResult:
        lim = self.limits
        planned = lim.straight_max_s if prim.kind == "STRAIGHT" else lim.arc_max_s
        result = PrimitiveResult(
            primitive_id=prim.primitive_id, phase=prim.phase, kind=prim.kind,
            direction=prim.direction, requested_throttle=prim.throttle,
            requested_steering=prim.steering, repeat=prim.repeat,
            planned_duration_s=planned)

        # PRECHECK — 통과 못 하면 **구동 명령은 보내지 않는다**. 다만 나가기
        # 전에 명시적 zero 를 한 번 보낸다: actuator 를 "아무 명령도 안 준
        # 상태" 로 두지 않는 것이 이 도구의 불변식이다.
        def _reject(reason: str, note: str, pose=None) -> PrimitiveResult:
            result.termination_reason = reason
            result.note = note
            result.start_pose = pose
            self.drive(0.0, 0.0)
            return result

        p0 = self.pose()
        if p0 is None:
            return _reject(TERM_POSE_STALE, "precheck: pose")
        if not self.comm_ok():
            return _reject(TERM_COMM_FAIL, "precheck: comm", p0)
        c = self.clearance()
        if c is not None and c < lim.min_clearance_mm:
            return _reject(TERM_BOUNDARY,
                           f"precheck: clearance {c:.0f}mm", p0)

        start = p0
        result.start_pose = start
        result.command_start_t = self.now()
        throttle = (self.steering_throttle if prim.kind == "ARC"
                    else prim.throttle)
        throttle = min(abs(throttle), lim.max_throttle)
        signed = throttle * (1.0 if prim.direction == "FORWARD" else -1.0)

        reason = TERM_DURATION
        end_by = result.command_start_t + planned
        current = start
        try:
            while True:
                t = self.now()
                if t >= end_by:
                    reason = TERM_DURATION
                    break
                if not self.comm_ok():
                    reason = TERM_COMM_FAIL
                    break
                p = self.pose()
                if p is None:
                    reason = TERM_POSE_STALE
                    break
                current = p
                c = self.clearance()
                if c is not None and c < lim.min_clearance_mm:
                    reason = TERM_BOUNDARY
                    break
                moved = math.hypot(p[0] - start[0], p[1] - start[1])
                turned = _heading_delta(start[2], p[2])
                if prim.kind == "STRAIGHT" and moved >= lim.straight_target_mm:
                    reason = TERM_DISTANCE
                    break
                if prim.kind == "ARC" and abs(turned) >= lim.arc_target_deg:
                    reason = TERM_HEADING
                    break
                # 무동작: 유예 시간이 지났는데도 잡음 수준이면 deadband 후보다.
                if (t - result.command_start_t >= lim.no_motion_grace_s
                        and moved < lim.no_motion_mm):
                    reason = TERM_NO_MOTION
                    break
                self.drive(signed, prim.steering)
                self.sleep(lim.command_period_s)
        except KeyboardInterrupt:
            reason = TERM_USER_ABORT
            raise
        finally:
            # 어떤 경로로 나가든 마지막 명령은 zero 다.
            self.drive(0.0, 0.0)
            result.zero_t = self.now()
            result.command_end_pose = current
            result.termination_reason = reason
            result.displacement_mm = math.hypot(current[0] - start[0],
                                                current[1] - start[1])
            result.heading_change_deg = _heading_delta(start[2], current[2])
            result.motion_detected = (reason != TERM_NO_MOTION
                                      and result.displacement_mm
                                      >= lim.no_motion_mm)

        stop_pose, settled = self._wait_physical_stop(current)
        result.physical_stop_t = self.now()
        result.stop_pose = stop_pose
        result.coast_mm = math.hypot(stop_pose[0] - current[0],
                                     stop_pose[1] - current[1])
        result.coast_s = (result.physical_stop_t or 0.0) - (result.zero_t or 0.0)
        result.end_t = self.now()
        if not settled:
            result.note = (result.note + " stop_timeout").strip()
        return result

    # 전체 시퀀스
    def _check_drift(self) -> None:
        if self.session_start_pose is None:
            return
        p = self.pose()
        if p is None:
            return
        s = self.session_start_pose
        dist = math.hypot(p[0] - s[0], p[1] - s[1])
        dh = abs(_heading_delta(s[2], p[2]))
        if dist > self.limits.drift_limit_mm or dh > self.limits.drift_limit_deg:
            raise CalibrationAborted(
                TERM_DRIFT,
                f"시작점에서 {dist:.0f}mm / {dh:.0f}도 벗어났습니다 — "
                "차를 다시 놓고 실행하세요")

    def run(self, plan: Sequence[Primitive]) -> list[PrimitiveResult]:
        self.session_start_pose = self._require_pose(TERM_PRECHECK)
        self.log(f"calibration 시작 pose="
                 f"({self.session_start_pose[0]:.0f},"
                 f"{self.session_start_pose[1]:.0f},"
                 f"{self.session_start_pose[2]:.0f}deg) "
                 f"primitives={len(plan)}")
        self._zero_for(0.5)
        try:
            for prim in plan:
                res = self.run_primitive(prim)
                self.results.append(res)
                self.emit(res.as_event())
                self.log(
                    f"  #{res.primitive_id:02d} {res.phase:<8} {res.direction:<7} "
                    f"thr={res.requested_throttle:.2f} "
                    f"steer={res.requested_steering:+.1f} "
                    f"-> {res.termination_reason:<22} "
                    f"move={res.displacement_mm:6.1f}mm "
                    f"turn={res.heading_change_deg:+6.1f}deg "
                    f"coast={res.coast_mm:5.1f}mm")
                if res.termination_reason in (TERM_COMM_FAIL, TERM_POSE_STALE):
                    raise CalibrationAborted(
                        res.termination_reason,
                        "관측/통신이 끊겨 계측을 계속할 수 없습니다")
                # 경계 때문에 **시작조차 못 했다면** 차가 이미 가장자리에 있는
                # 것이다. 남은 primitive 를 하나씩 거절해 세션을 낭비하지 않고
                # 여기서 끝낸다 — 사람이 차를 다시 놓아야 한다. (자동으로
                # 위험하게 빠져나오려 시도하지 않는다.)
                if (res.termination_reason == TERM_BOUNDARY
                        and res.command_start_t is None):
                    raise CalibrationAborted(
                        TERM_BOUNDARY,
                        "차가 맵 가장자리에 너무 가까워 계측을 시작할 수 "
                        "없습니다 — 통로 중앙으로 다시 놓고 실행하세요")
                # F/R 쌍이 끝나는 시점에만 누적 drift 를 본다.
                if res.direction == "REVERSE":
                    self._check_drift()
        finally:
            self._zero_for(0.3)
        return self.results
