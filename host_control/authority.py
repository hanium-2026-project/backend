"""Host-side control authority 상태기계.

한 시점에 **하나의 control producer 만** non-zero 명령을 낼 수 있도록 강제한다.
프롬프트 2장/6장/15장의 authority 요구를 구현한다.

상태
----
- DISARMED : 초기/무장 해제. 항상 zero. 여기서만 arm 가능.
- MANUAL   : 사람 입력(ManualControlProducer)만 반영. auto 무시.
- AUTO_HOST: 자율(AutoControlProducer)만 반영. manual 무시.
- FAULTED  : latched. 항상 zero. **clear 만으로 주행 복귀 불가** → DISARMED 로만 빠지고,
             다시 arm 해야 한다(explicit re-arm).

전이 규칙
--------
- arm_manual(): DISARMED → MANUAL
- arm_auto()  : DISARMED → AUTO_HOST
- disarm()    : MANUAL/AUTO_HOST → DISARMED
- fault(reason): 임의 상태 → FAULTED (latched)
- stop()      : 임의 상태 → FAULTED (비상정지도 latched. 재출발엔 re-arm 필요)
- clear_fault(): FAULTED → DISARMED (주행 복귀 아님. zero 유지)

manual 과 auto 는 단일 enum 상태로 표현되므로 **동시 활성이 구조적으로 불가능**하다.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class Authority(str, Enum):
    DISARMED = "DISARMED"
    MANUAL = "MANUAL"
    AUTO_HOST = "AUTO_HOST"
    FAULTED = "FAULTED"


class AuthorityError(RuntimeError):
    """허용되지 않은 authority 전이 시도."""


class ControlAuthority:
    """authority 상태와 전이를 관리."""

    def __init__(self) -> None:
        self._state: Authority = Authority.DISARMED
        self._fault_reason: str = ""

    # --------------------------------------------------------------- 조회
    @property
    def state(self) -> Authority:
        return self._state

    @property
    def fault_reason(self) -> str:
        return self._fault_reason

    @property
    def is_driving_allowed(self) -> bool:
        """non-zero 명령이 나갈 수 있는 상태인가(MANUAL/AUTO_HOST)."""
        return self._state in (Authority.MANUAL, Authority.AUTO_HOST)

    @property
    def is_manual(self) -> bool:
        return self._state is Authority.MANUAL

    @property
    def is_auto(self) -> bool:
        return self._state is Authority.AUTO_HOST

    @property
    def is_faulted(self) -> bool:
        return self._state is Authority.FAULTED

    # --------------------------------------------------------------- 전이
    def arm_manual(self) -> None:
        self._require(Authority.DISARMED, "arm_manual")
        self._state = Authority.MANUAL

    def arm_auto(self) -> None:
        self._require(Authority.DISARMED, "arm_auto")
        self._state = Authority.AUTO_HOST

    def disarm(self) -> None:
        # 무장 상태에서만 정상 disarm. FAULTED 는 clear_fault 로 처리.
        if self._state is Authority.FAULTED:
            raise AuthorityError("FAULTED 상태에서는 clear_fault() 를 사용하십시오.")
        self._state = Authority.DISARMED
        self._fault_reason = ""

    def fault(self, reason: str) -> None:
        """어느 상태에서든 FAULTED 로 latch (이미 FAULTED 면 사유 유지)."""
        if self._state is not Authority.FAULTED:
            self._fault_reason = reason or "FAULT"
            self._state = Authority.FAULTED

    def stop(self) -> None:
        """비상 정지. 안전을 위해 FAULTED 로 latch (재출발 시 re-arm 필요)."""
        self.fault("STOP")

    def clear_fault(self) -> None:
        """FAULTED 해제 → DISARMED. **주행 복귀가 아니다.** 재출발엔 arm_* 재호출 필요."""
        if self._state is not Authority.FAULTED:
            raise AuthorityError("FAULTED 상태가 아닙니다.")
        self._state = Authority.DISARMED
        self._fault_reason = ""

    def re_arm_auto(self) -> None:
        """편의: FAULTED → (clear) → DISARMED → AUTO_HOST 명시적 재무장."""
        if self._state is Authority.FAULTED:
            self.clear_fault()
        self.arm_auto()

    def re_arm_manual(self) -> None:
        if self._state is Authority.FAULTED:
            self.clear_fault()
        self.arm_manual()

    # --------------------------------------------------------------- 내부
    def _require(self, expected: Authority, action: str) -> None:
        if self._state is not expected:
            raise AuthorityError(
                f"{action} 는 {expected.value} 에서만 가능(현재 {self._state.value})."
            )
