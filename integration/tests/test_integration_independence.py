"""host_control / integration 독립성 테스트.

core 계산 계층(controller, host_control)은 backend/network/Django/YOLO 를 import 하지 않는다.
integration adapter 는 backend 를 duck-typing 으로만 다루고 실제 import 하지 않는다.
"""

from __future__ import annotations

import ast
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# core 계산 계층(controller, host_control)은 엄격히 순수해야 한다:
# backend/framework/network 뿐 아니라 동시성/소켓도 금지(pure calculation).
FORBIDDEN_CORE = {
    "comm", "django", "redis", "socket", "threading", "asyncio",
    "cv2", "numpy", "torch", "ultralytics", "channels", "rest_framework",
    "server", "pipeline", "requests",
}

# integration 은 런타임 glue 계층이다: stdlib 동시성(threading/time)은 허용하되
# backend/framework/network/CV 는 여전히 금지(duck-typing 으로만 연결).
FORBIDDEN_INTEGRATION = {
    "comm", "django", "redis", "socket",
    "cv2", "numpy", "torch", "ultralytics", "channels", "rest_framework",
    "server", "pipeline", "requests",
}

# 검사 대상: controller/*.py, host_control/*.py, integration/*.py (tests 제외)
def _py_files(pkg: str):
    d = os.path.join(REPO_ROOT, pkg)
    for name in sorted(os.listdir(d)):
        if name.endswith(".py"):
            yield os.path.join(d, name)


def _top_imports(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                tops.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                tops.add(node.module.split(".")[0])
    return tops


class TestPackageIndependence(unittest.TestCase):
    def _check_pkg(self, pkg: str, forbidden: set) -> None:
        for path in _py_files(pkg):
            tops = _top_imports(path)
            bad = tops & forbidden
            self.assertFalse(
                bad, f"{pkg}/{os.path.basename(path)} 가 금지 모듈 import: {bad}"
            )

    def test_controller_clean(self) -> None:
        self._check_pkg("controller", FORBIDDEN_CORE)

    def test_host_control_clean(self) -> None:
        self._check_pkg("host_control", FORBIDDEN_CORE)

    def test_integration_clean(self) -> None:
        # integration 은 backend/network/CV 를 실제 import 하지 않아야 함(duck-typing).
        # stdlib 동시성(threading/time)은 허용.
        self._check_pkg("integration", FORBIDDEN_INTEGRATION)

    def test_importable_without_backend(self) -> None:
        import importlib
        for mod in [
            "controller.pose_controller",
            "host_control.host_controller",
            "host_control.authority",
            "host_control.mission",
            "integration.backend_adapter",
            "integration.camera_adapter",
        ]:
            importlib.import_module(mod)


if __name__ == "__main__":
    unittest.main()
