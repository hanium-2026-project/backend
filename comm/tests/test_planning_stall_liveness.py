"""ENTRY_STAGING planning 정지 구간의 통신 fail-safe 계약.

ENTRY_STAGING planner 는 실측 3.4~4.2s 동안 **동기로** 계산한다 (후보 슬롯을
여러 개 시도하면 그 배수). 그동안 pipeline/camera 스레드는 완전히 멈춘다.
펌웨어에는 500ms DIRECT_CONTROL deadman 과 1000ms HEARTBEAT deadman 이 있다.

여기서 고정하는 것은 하나다: **계획 시간이 통신 fail-safe 를 유발하지 않는다.**

성립 근거는 스레드 분리다. 송신은 pipeline 스레드가 아니라
VehicleServer._tick_loop 데몬 스레드가 한다:
  - HEARTBEAT  : 250ms 주기 (펌웨어 1000ms deadman 대비 4x)
  - DIRECT_CONTROL: 100ms 주기 (펌웨어 500ms deadman 대비 5x)
  - control_stale_ms(300ms) 를 넘겨 갱신이 끊기면 **서버가 스스로 zero 로
    교체해서** 보낸다 — 이전 non-zero 명령을 계속 재전송하지 않는다.

마지막 항목이 핵심이다. 그게 없으면 "planner 4초 계산 -> sender 가 직전
주행 명령을 계속 resend" 가 되어 차가 계획 중에 굴러간다.
"""

from __future__ import annotations

import threading
import time
import unittest

from comm import protocol
from comm.server import VehicleServer
from comm.tests.mock_firmware import MockFirmware

# 펌웨어 실제 값 (integrated/esp32_main/main/app_config.h)
FW_DIRECT_DEADMAN_MS = 500.0     # DIRECT_CONTROL_TIMEOUT_MS
FW_HEARTBEAT_DEADMAN_MS = 1000.0  # HEARTBEAT_TIMEOUT_MS

# ENTRY_STAGING worst-case 계획 시간 (실측 3.4~4.2s) 보다 넉넉히 잡는다.
PLANNING_BLOCK_S = 4.5


def wait_until(cond, timeout: float = 3.0, interval: float = 0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False


class _RxTimeline:
    """mock 펌웨어가 실제로 수신한 프레임의 시각을 기록한다."""

    def __init__(self, esp: MockFirmware) -> None:
        self.direct: list[tuple[float, float, float]] = []   # (t, throttle, steering)
        self.heartbeat: list[float] = []
        self._lock = threading.Lock()
        base_direct = esp._on_direct
        base_hb = esp._on_heartbeat

        def on_direct(msg):
            base_direct(msg)
            with self._lock:
                self.direct.append((time.monotonic(),
                                    float(msg["throttle"]),
                                    float(msg["steering"])))

        def on_hb(msg):
            base_hb(msg)
            with self._lock:
                self.heartbeat.append(time.monotonic())

        esp._on_direct = on_direct
        esp._on_heartbeat = on_hb

    def snapshot(self):
        with self._lock:
            return list(self.direct), list(self.heartbeat)

    @staticmethod
    def max_gap_ms(times: list[float], start: float, end: float) -> float:
        """[start, end] 구간의 최대 수신 간격 (ms). 구간 경계도 간격으로 센다."""
        marks = [start] + [t for t in times if start <= t <= end] + [end]
        return max((b - a) * 1000.0 for a, b in zip(marks, marks[1:]))


class PlanningStallLiveness(unittest.TestCase):

    def setUp(self) -> None:
        self.server = VehicleServer(port=0, known_car_ids={1})
        self.failures: list[dict] = []
        self.server.on_comm_fail = lambda _cid, info: self.failures.append(info)
        self.server.direct_control_enabled = True
        self.server.start()
        self.esp = MockFirmware(self.server.bound_port, status_interval=0.2)
        self.assertTrue(wait_until(lambda: self.esp.state == "READY"))
        self.timeline = _RxTimeline(self.esp)

    def tearDown(self) -> None:
        self.esp.close()
        self.server.stop()

    def _block_pipeline_thread(self, seconds: float) -> tuple[float, float]:
        """planner 를 흉내내어 **호출 스레드를 통째로** 점유한다.

        sleep 이 아니라 실제 CPU 를 도는 루프여야 한다. planner 는 순수 파이썬
        기하 탐색이라 GIL 을 잡고 돈다 — 그 조건까지 재현해야 의미가 있다.
        """
        start = time.monotonic()
        end = start + seconds
        while time.monotonic() < end:
            sum(i * i for i in range(2000))
        return start, time.monotonic()

    # ─── 핵심 계약 ──────────────────────────────────────────────────────────

    def test_direct_control_never_gaps_past_the_firmware_deadman(self):
        """계획 4.5초 동안 DIRECT_CONTROL 수신 간격이 500ms 를 넘지 않는다."""
        self.server.push_control(1, 0.0, 0.0)
        self.assertTrue(wait_until(lambda: self.timeline.direct != []))

        start, end = self._block_pipeline_thread(PLANNING_BLOCK_S)
        time.sleep(0.15)

        direct, _ = self.timeline.snapshot()
        gap = _RxTimeline.max_gap_ms([t for t, _thr, _st in direct], start, end)
        self.assertLess(gap, FW_DIRECT_DEADMAN_MS,
                        f"DIRECT_CONTROL 최대 공백 {gap:.0f}ms")

    def test_heartbeat_never_gaps_past_the_firmware_deadman(self):
        """같은 구간에서 HEARTBEAT 도 1000ms 를 넘지 않는다.

        펌웨어 HEARTBEAT deadman 은 **HEARTBEAT 로만** 갱신된다
        (network_client.c: reset_heartbeat_watchdog 는 HEARTBEAT/HELLO_ACK
        경로에만 있다). DIRECT_CONTROL 이나 STATUS 로는 갱신되지 않는다.
        """
        start, end = self._block_pipeline_thread(PLANNING_BLOCK_S)
        time.sleep(0.15)

        _, hb = self.timeline.snapshot()
        gap = _RxTimeline.max_gap_ms(hb, start, end)
        self.assertLess(gap, FW_HEARTBEAT_DEADMAN_MS,
                        f"HEARTBEAT 최대 공백 {gap:.0f}ms")

    def test_no_comm_failure_is_raised_during_planning(self):
        self._block_pipeline_thread(PLANNING_BLOCK_S)
        time.sleep(0.15)
        self.assertEqual(self.failures, [])
        self.assertTrue(self.server._session(1).alive)

    # ─── stale non-zero 금지 (요청문 13절) ──────────────────────────────────

    def test_previous_nonzero_command_is_not_resent_through_planning(self):
        """직전 주행 명령이 계획 구간 내내 재전송되면 차가 굴러간다.

        서버는 control_stale_ms(300ms) 를 넘으면 latest_control 을 zero 로
        **교체**한다. 그래서 계획이 4.5초 걸려도 차가 받는 값은 zero 다.
        """
        self.server.push_control(1, 0.25, -0.30)
        self.assertTrue(wait_until(
            lambda: any(t == 0.25 for _, t, _ in self.timeline.snapshot()[0])))

        start, end = self._block_pipeline_thread(PLANNING_BLOCK_S)
        time.sleep(0.15)

        direct, _ = self.timeline.snapshot()
        during = [(t, thr, st) for t, thr, st in direct if start <= t <= end]
        self.assertTrue(during, "계획 구간에 DIRECT_CONTROL 이 하나도 없다")

        # 마지막 non-zero 수신은 stale 판정(300ms) 안에서 끝나야 한다.
        last_nonzero = max((t for t, thr, st in during if thr or st),
                           default=start)
        self.assertLess((last_nonzero - start) * 1000.0, 400.0,
                        "non-zero 명령이 stale 교체 시점을 넘겨 재전송됐다")

        # 그 이후로는 전부 zero 여야 한다.
        tail = [(thr, st) for t, thr, st in during if t > last_nonzero + 0.05]
        self.assertTrue(tail)
        self.assertTrue(all(thr == 0.0 and st == 0.0 for thr, st in tail),
                        f"계획 중 non-zero 재전송: {set(tail)}")

    def test_server_substitutes_zero_rather_than_holding_the_last_value(self):
        """교체 지점을 직접 확인한다 (타이밍이 아니라 상태로)."""
        self.server.push_control(1, 0.25, -0.30)
        sess = self.server._session(1)
        self.assertEqual(sess.latest_control["throttle"], 0.25)

        self.assertTrue(wait_until(
            lambda: sess.latest_control["throttle"] == 0.0, timeout=2.0),
            "control_stale_ms 를 넘겨도 마지막 값이 유지된다")
        self.assertEqual(sess.latest_control["steering"], 0.0)

    # ─── 다중 차량 격리 (요청문 16절) ───────────────────────────────────────

    def test_one_car_planning_does_not_starve_another_session(self):
        """car1 계획 중에도 car2 의 DIRECT_CONTROL/HEARTBEAT 이 끊기지 않는다."""
        server = VehicleServer(port=0, known_car_ids={1, 2})
        server.direct_control_enabled = True
        server.start()
        esp1 = MockFirmware(server.bound_port, car_id="CAR_01",
                            status_interval=0.2)
        esp2 = MockFirmware(server.bound_port, car_id="CAR_02",
                            status_interval=0.2, boot_id="B0000002")
        try:
            self.assertTrue(wait_until(lambda: esp1.state == "READY"))
            self.assertTrue(wait_until(lambda: esp2.state == "READY"))
            tl2 = _RxTimeline(esp2)
            server.push_control(1, 0.0, 0.0)
            server.push_control(2, 0.0, 0.0)
            self.assertTrue(wait_until(lambda: tl2.direct != []))

            start, end = self._block_pipeline_thread(PLANNING_BLOCK_S)
            time.sleep(0.15)

            direct2, hb2 = tl2.snapshot()
            self.assertLess(
                _RxTimeline.max_gap_ms([t for t, _thr, _st in direct2],
                                       start, end),
                FW_DIRECT_DEADMAN_MS)
            self.assertLess(_RxTimeline.max_gap_ms(hb2, start, end),
                            FW_HEARTBEAT_DEADMAN_MS)
        finally:
            esp1.close()
            esp2.close()
            server.stop()


class TimingContract(unittest.TestCase):
    """상수 사이의 관계를 고정한다. 하나만 바뀌어도 계약이 깨진다."""

    def test_backend_stream_intervals_beat_the_firmware_deadmen(self):
        self.assertLess(protocol.TIMING["CONTROL_INTERVAL"] * 2,
                        FW_DIRECT_DEADMAN_MS)
        self.assertLess(protocol.TIMING["HEARTBEAT_INTERVAL"] * 2,
                        FW_HEARTBEAT_DEADMAN_MS)

    def test_stale_substitution_happens_before_the_direct_deadman(self):
        """zero 교체가 펌웨어 deadman 보다 **먼저** 일어나야 의미가 있다.

        늦으면 차는 그 사이 이전 명령대로 계속 움직인다.
        """
        self.assertLess(VehicleServer(port=0).control_stale_ms,
                        FW_DIRECT_DEADMAN_MS)


if __name__ == "__main__":
    unittest.main()
