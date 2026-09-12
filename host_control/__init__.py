"""host_control — B안 host-side autonomous control 패키지.

계층:
- authority.py       : ControlAuthority / Authority (DISARMED/MANUAL/AUTO_HOST/FAULTED)
- producers.py       : ManualControlProducer / AutoControlProducer / ManualInput
- mission.py         : HostWaypointMission / MissionStatus (host-owned waypoint 진행)
- pose_source.py     : CameraPoseSource (관측 timestamp 보존 → stale fail-safe)
- direct_control.py  : DirectControlSender / TransportTiming (공용 DIRECT_CONTROL transport)
- host_controller.py : HostController (위를 묶는 tick 루프)

core 제어 계산은 controller/ 패키지를 재사용한다. 이 패키지는 backend/network 를 import 하지 않는다.
"""

from .authority import Authority, AuthorityError, ControlAuthority
from .approach_guard import ApproachEvent, ApproachProgress, ApproachProgressGuard, ApproachStage
from .final_pose_guard import FinalPoseGuard, FinalPoseProgress
from .direct_control import DirectControlSender, TransportTiming
from .host_controller import HostController, TickResult
from .mission import HostWaypointMission, MissionStatus
from .producers import (
    AutoControlProducer,
    ManualControlProducer,
    ManualInput,
)
from .pose_source import CameraPoseSource

__all__ = [
    "Authority",
    "AuthorityError",
    "ControlAuthority",
    "ApproachEvent",
    "ApproachProgress",
    "ApproachProgressGuard",
    "ApproachStage",
    "FinalPoseGuard",
    "FinalPoseProgress",
    "DirectControlSender",
    "TransportTiming",
    "HostController",
    "TickResult",
    "HostWaypointMission",
    "MissionStatus",
    "AutoControlProducer",
    "ManualControlProducer",
    "ManualInput",
    "CameraPoseSource",
]
