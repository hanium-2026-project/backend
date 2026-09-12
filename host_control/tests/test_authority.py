"""authority 상태기계 테스트."""

from __future__ import annotations

import unittest

from host_control.authority import Authority, AuthorityError, ControlAuthority


class TestAuthority(unittest.TestCase):
    def setUp(self) -> None:
        self.a = ControlAuthority()

    def test_initial_disarmed(self) -> None:
        self.assertIs(self.a.state, Authority.DISARMED)
        self.assertFalse(self.a.is_driving_allowed)

    def test_arm_manual_from_disarmed(self) -> None:
        self.a.arm_manual()
        self.assertIs(self.a.state, Authority.MANUAL)
        self.assertTrue(self.a.is_manual)
        self.assertTrue(self.a.is_driving_allowed)

    def test_arm_auto_from_disarmed(self) -> None:
        self.a.arm_auto()
        self.assertIs(self.a.state, Authority.AUTO_HOST)
        self.assertTrue(self.a.is_auto)

    def test_cannot_arm_auto_while_manual(self) -> None:
        self.a.arm_manual()
        with self.assertRaises(AuthorityError):
            self.a.arm_auto()  # 동시 활성 방지: MANUAL 에서 AUTO 무장 불가

    def test_cannot_arm_manual_while_auto(self) -> None:
        self.a.arm_auto()
        with self.assertRaises(AuthorityError):
            self.a.arm_manual()

    def test_disarm_returns_to_disarmed(self) -> None:
        self.a.arm_auto()
        self.a.disarm()
        self.assertIs(self.a.state, Authority.DISARMED)

    def test_fault_latches_from_any_state(self) -> None:
        self.a.arm_auto()
        self.a.fault("POSE_STALE")
        self.assertIs(self.a.state, Authority.FAULTED)
        self.assertEqual(self.a.fault_reason, "POSE_STALE")
        self.assertFalse(self.a.is_driving_allowed)

    def test_stop_latches_faulted(self) -> None:
        self.a.arm_manual()
        self.a.stop()
        self.assertIs(self.a.state, Authority.FAULTED)

    def test_clear_fault_only_goes_to_disarmed_not_driving(self) -> None:
        self.a.arm_auto()
        self.a.fault("COMM_LOSS")
        self.a.clear_fault()
        # ★ fault clear 만으로 주행 복귀 불가 — DISARMED(zero) 로만.
        self.assertIs(self.a.state, Authority.DISARMED)
        self.assertFalse(self.a.is_driving_allowed)

    def test_re_arm_required_after_fault(self) -> None:
        self.a.arm_auto()
        self.a.fault("X")
        # 명시적 재무장 없이는 AUTO 로 복귀 불가
        self.assertFalse(self.a.is_auto)
        self.a.re_arm_auto()  # clear_fault + arm_auto
        self.assertIs(self.a.state, Authority.AUTO_HOST)

    def test_disarm_from_faulted_raises(self) -> None:
        self.a.arm_auto()
        self.a.fault("X")
        with self.assertRaises(AuthorityError):
            self.a.disarm()  # FAULTED 에서는 clear_fault 를 써야 함

    def test_clear_fault_when_not_faulted_raises(self) -> None:
        with self.assertRaises(AuthorityError):
            self.a.clear_fault()


if __name__ == "__main__":
    unittest.main()
