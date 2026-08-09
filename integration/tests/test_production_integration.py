"""production integration tests — 실제 backend 최신 API 계약(v4).

reference 계약 재현(spec)으로 실행한다. 실제 backend full source 는 이 환경에 없다
(SOURCE_COMPATIBILITY.md 참고). 실제 wire E2E 는 tests/real_backend/ 에서 별도 처리.

검증(프롬프트 15장 A~J):
- push_control(car_id:int, throttle, steering) -> None
- server 소유 control_seq (adapter 미생성)
- 비동기 SET_MODE handshake (send_set_mode->seq, ACCEPTED 후 arm)
- direct_control_enabled True (AUTO_HOST)
- int car_id
- camera stale → latest_control 0/0
- no auto resume
- comm fail/recovered
- mission progression, WAYPOINT/GO 0
- alignment → zero + replan
"""

from __future__ import annotations

import math
import unittest

from controller.config import SimulationConfig
from host_control import HostController, HostWaypointMission, Authority
from host_control.mission import MissionStatus
from integration.backend_adapter import VehicleServerDirectSender, waypoints_from_backend
from integration.control_scheduler import ControlScheduler
from integration.remote_direct_session import RemoteDirectSession, ModeHandshakeError
from integration.backend_contract import (
    SpecVehicleServer as Server, MockFirmware as Firmware, SpecWaypoint as Waypoint)

CAR = 1  # ★ 내부 car_id 는 int


def make_stack(route):
    fw = Firmware()
    server = Server(firmware=fw)
    server.register_car(CAR)
    host = HostController(mission=HostWaypointMission(waypoints_from_backend(route)),
                          sender=VehicleServerDirectSender(server, CAR))
    sched = ControlScheduler(host)
    sess = RemoteDirectSession(host, server, CAR)
    return fw, server, host, sched, sess


def arm_auto(host, server, sess, *, result="ACCEPTED"):
    """비동기 handshake 를 동기 테스트에서 완료.

    result="ACCEPTED": firmware 를 실제 REMOTE_DIRECT 로 전환(auto_accept_set_mode)하고 arm.
    그 외: 해당 결과를 배달만 하여 거절 경로 검증.
    """
    sess.attach()
    seq = sess.begin_handshake()
    if result == "ACCEPTED":
        server.auto_accept_set_mode(CAR)   # firmware.on_set_mode → REMOTE_DIRECT + 결과 배달
        sess._enable_direct_stream()
        host.arm_auto()
    else:
        server.deliver_command_result(CAR, seq, result)
    return seq


def bike(x, y, h, thr, wire, sim):
    d = math.radians(-wire * sim.max_wheel_angle_deg)
    v = thr * sim.sim_speed_cm_s_at_full_throttle * 10.0
    dt = sim.dt_s
    hr = math.radians(h)
    return (x + v * math.cos(hr) * dt, y + v * math.sin(hr) * dt,
            (math.degrees(hr + (v / sim.wheelbase_mm) * math.tan(d) * dt)) % 360)


class TestProductionIntegration(unittest.TestCase):
    # A. push_control 시그니처 / 반환 None
    def test_A_push_control_signature_returns_none(self):
        fw, server, host, sched, sess = make_stack([Waypoint(600, 0, is_final=True)])
        ret = server.push_control(CAR, 0.3, -0.2)
        self.assertIsNone(ret)  # ★ None
        self.assertEqual(server.latest_control(CAR), (0.3, -0.2))

    def test_A_payload_style_rejected(self):
        fw, server, host, sched, sess = make_stack([Waypoint(600, 0, is_final=True)])
        with self.assertRaises((TypeError, AssertionError)):
            server.push_control({"throttle": 0.3})  # 옛 payload 방식

    # B. server-owned control_seq (adapter 미생성)
    def test_B_server_owns_control_seq(self):
        fw, server, host, sched, sess = make_stack([Waypoint(600, 0, is_final=True)])
        sender = host.sender
        self.assertFalse(hasattr(sender, "_seq"))  # adapter 자체 seq 없음
        server.push_control(CAR, 0.2, 0.0)
        server.push_control(CAR, 0.2, 0.0)
        self.assertEqual(server.control_seq(CAR), 2)  # 서버가 증가

    # C. 비동기 SET_MODE handshake
    def test_C_async_set_mode_seq_and_gate(self):
        fw, server, host, sched, sess = make_stack([Waypoint(600, 0, is_final=True)])
        sess.attach()
        seq = sess.begin_handshake()
        self.assertIsInstance(seq, int)
        # ACCEPTED 전에는 arm 되지 않음(non-zero 금지)
        self.assertNotEqual(host.authority.state, Authority.AUTO_HOST)
        server.deliver_command_result(CAR, seq, "ACCEPTED")
        self.assertTrue(sess.accepted)
        sess._enable_direct_stream(); host.arm_auto()
        self.assertEqual(host.authority.state, Authority.AUTO_HOST)

    def test_C_set_mode_reject_faults(self):
        fw, server, host, sched, sess = make_stack([Waypoint(600, 0, is_final=True)])
        sess.attach()
        seq = sess.begin_handshake()
        server.deliver_command_result(CAR, seq, "INVALID_STATE")
        self.assertEqual(host.authority.state, Authority.FAULTED)

    # D. direct_control_enabled True (AUTO_HOST)
    def test_D_direct_control_enabled_on_arm(self):
        fw, server, host, sched, sess = make_stack([Waypoint(600, 0, is_final=True)])
        self.assertFalse(server.direct_control_enabled)
        arm_auto(host, server, sess)
        self.assertTrue(server.direct_control_enabled)  # ★ arm 시 True 보장

    # E. int car_id
    def test_E_int_car_id(self):
        fw, server, host, sched, sess = make_stack([Waypoint(600, 0, is_final=True)])
        self.assertIsInstance(host.sender.car_id, int)
        with self.assertRaises(AssertionError):
            VehicleServerDirectSender(server, "CAR_01")  # str 금지

    # F. camera stale → latest_control 0/0
    def test_F_stale_zeros_latest_control(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        arm_auto(host, server, sess)
        host.pose_source.observe(0, 0, 0.0, obs_time=100.0)
        r0 = sched.step(now=100.0)
        self.assertGreater(r0.command.throttle, 0.0)
        self.assertNotEqual(server.latest_control(CAR), (0.0, 0.0))
        for k in range(1, 8):
            sched.step(now=100.0 + k * 0.1)  # 새 관측 없이 계속
        self.assertEqual(server.latest_control(CAR), (0.0, 0.0))  # ★ zero 갱신
        self.assertEqual(host.authority.state, Authority.FAULTED)

    # G. no auto resume
    def test_G_no_auto_resume(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        arm_auto(host, server, sess)
        host.pose_source.observe(0, 0, 0.0, obs_time=100.0)
        sched.step(now=100.0)
        for k in range(1, 8):
            sched.step(now=100.0 + k * 0.1)
        self.assertEqual(host.authority.state, Authority.FAULTED)
        host.pose_source.observe(10, 0, 0.0, obs_time=101.0)
        r = sched.step(now=101.0)
        self.assertEqual(r.authority, Authority.FAULTED)
        self.assertEqual((r.command.throttle, r.command.steering), (0.0, 0.0))

    # H. comm fail → FAULTED, recovered 유지
    def test_H_comm_fail_recovered(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        arm_auto(host, server, sess)
        host.pose_source.observe(0, 0, 0.0, obs_time=100.0)
        sched.step(now=100.0)
        server.trigger_comm_fail(CAR)
        self.assertEqual(host.authority.state, Authority.FAULTED)
        server.trigger_comm_recovered(CAR)
        self.assertEqual(host.authority.state, Authority.FAULTED)  # 여전히 FAULTED

    def test_H_callback_fanout_preserves_existing(self):
        # 기존 pipeline callback 을 덮어쓰지 않는지
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        seen = []
        server.on_comm_fail = lambda cid, st: seen.append(("existing", cid))
        sess.attach()  # fan-out 으로 감쌈
        server.trigger_comm_fail(CAR)
        self.assertIn(("existing", CAR), seen)         # 기존 호출 보존
        self.assertEqual(host.authority.state, Authority.FAULTED)  # host 도 호출됨

    # I. mission progression, WAYPOINT/GO 0
    def test_I_progression_no_waypoint_go(self):
        sim = SimulationConfig()
        route = [Waypoint(400, 50, position_tolerance_cm=8),
                 Waypoint(800, 150, position_tolerance_cm=8),
                 Waypoint(1100, 150, position_tolerance_cm=8, is_final=True)]
        fw, server, host, sched, sess = make_stack(route)
        arm_auto(host, server, sess)
        x, y, h = 0.0, 0.0, 0.0
        t = 100.0
        prog = []
        for _ in range(4000):
            host.pose_source.observe(x, y, h, obs_time=t)
            r = sched.step(now=t)
            server.tick(CAR, now=t)
            if not prog or prog[-1] != host.mission.index:
                prog.append(host.mission.index)
            if r.mission_status is MissionStatus.DONE:
                break
            x, y, h = bike(x, y, h, r.command.throttle, r.command.steering, sim)
            t += sim.dt_s
        self.assertEqual(prog, [0, 1, 2])
        self.assertIs(host.mission.status, MissionStatus.DONE)
        self.assertEqual(fw.count("WAYPOINT"), 0)
        self.assertEqual(fw.count("GO"), 0)
        self.assertGreater(fw.count("DIRECT_CONTROL"), 0)

    # J. alignment → zero + replan
    def test_J_alignment_replan(self):
        fw, server, host, sched, sess = make_stack(
            [Waypoint(100, 0, position_tolerance_cm=50, heading_required=True,
                      target_heading_deg=90.0, heading_tolerance_deg=10.0, is_final=True)])
        arm_auto(host, server, sess)
        host.pose_source.observe(90, 0, 0.0, obs_time=100.0)
        r = sched.step(now=100.0)
        self.assertEqual((r.command.throttle, r.command.steering), (0.0, 0.0))
        self.assertIs(host.mission.status, MissionStatus.REPLAN_REQUIRED)

    # 부호(wire) 최종 검증
    def test_wire_steering_sign(self):
        fw, server, host, sched, sess = make_stack([Waypoint(0, 1000, is_final=True)])
        arm_auto(host, server, sess)
        host.pose_source.observe(0, 0, 0.0, obs_time=100.0)
        r = sched.step(now=100.0)
        server.tick(CAR, now=100.0)
        self.assertLess(r.command.steering, 0.0)         # +Y = LEFT → wire 음수
        self.assertLess(fw.last_steering, 0.0)           # 서버 경유 wire 도 음수

    # E 타이밍
    def test_scheduler_faster_than_firmware_timeout(self):
        from integration.backend_contract import DIRECT_CONTROL_TIMEOUT_S
        self.assertLess(ControlScheduler(HostController()).period_s, DIRECT_CONTROL_TIMEOUT_S)


class TestV5CallbackContract(unittest.TestCase):
    """v5 회귀: 실제 backend callback arity 및 ACCEPTED seq 규칙."""

    # on_comm_recovered 는 실제로 인자 1개(car_id). fan-out 도 1개로 호출.
    def test_comm_recovered_one_arg_fanout(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        seen = []
        server.on_comm_recovered = lambda cid: seen.append(("existing", cid))  # ★ 1-arg
        sess.attach()
        arm_auto(host, server, sess)
        server.trigger_comm_fail(CAR)                 # 먼저 FAULTED
        self.assertEqual(host.authority.state, Authority.FAULTED)
        # 1-arg recovered fan-out: TypeError 없이 existing + host 둘 다 호출
        server.trigger_comm_recovered(CAR)
        self.assertIn(("existing", CAR), seen)
        self.assertEqual(host.authority.state, Authority.FAULTED)  # 자동 복귀 금지

    # ACCEPTED seq 불일치는 무시
    def test_accepted_seq_mismatch_ignored(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        sess.attach()
        seq = sess.begin_handshake()
        server.deliver_command_result(CAR, seq + 99, "ACCEPTED")  # 다른 seq
        self.assertFalse(sess.accepted)                 # 무시
        self.assertNotEqual(host.authority.state, Authority.AUTO_HOST)
        server.deliver_command_result(CAR, seq, "ACCEPTED")       # 올바른 seq
        self.assertTrue(sess.accepted)

    # negative terminal result → FAULTED + zero, on_command_rejected 도 호출
    def test_negative_terminal_result_faults(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        rejected = []
        server.on_command_rejected = lambda cid, res, msg: rejected.append((cid, res))
        sess.attach()
        seq = sess.begin_handshake()
        server.deliver_command_result(CAR, seq, "INVALID_STATE")
        self.assertEqual(host.authority.state, Authority.FAULTED)
        self.assertIn((CAR, "INVALID_STATE"), rejected)  # 기존 rejected 콜백 보존

    # re-arm 시 SET_MODE 재협상(seq 재발급)
    def test_re_arm_reissues_set_mode(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        arm_auto(host, server, sess)
        host.pose_source.observe(0, 0, 0.0, obs_time=100.0)
        sched.step(now=100.0)
        server.trigger_comm_fail(CAR)
        self.assertEqual(host.authority.state, Authority.FAULTED)
        n_before = len(server.mode_log)
        # re-arm: 새 SET_MODE 발행 + ACCEPTED 후 복귀
        sess.begin_handshake()
        server.auto_accept_set_mode(CAR)
        sess._enable_direct_stream(); host.re_arm_auto()
        self.assertGreater(len(server.mode_log), n_before)  # SET_MODE 재발행
        self.assertEqual(host.authority.state, Authority.AUTO_HOST)

    # comm_fail 은 global direct_control_enabled 를 끄지 않는다(다중 차량 안전)
    def test_comm_fail_keeps_global_direct_enabled(self):
        fw, server, host, sched, sess = make_stack([Waypoint(500, 0, is_final=True)])
        arm_auto(host, server, sess)
        self.assertTrue(server.direct_control_enabled)
        server.trigger_comm_fail(CAR)
        self.assertTrue(server.direct_control_enabled)   # ★ global 유지
        self.assertEqual(server.latest_control(CAR), (0.0, 0.0))  # 이 차량만 zero


if __name__ == "__main__":
    unittest.main()
