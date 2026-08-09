"""독립성 테스트: core 모듈이 backend/comm/Django/network 를 import 하지 않는지 검증.

AST 로 실제 import 문을 파싱해 금지 목록과 대조한다.
core 대상: models, geometry, config, pose_controller.
adapter_example 은 duck-typing 예시이므로 backend/network 를 import 하지 않아야 하지만,
core 계산 경로에는 포함되지 않는다(별도로도 검증).
"""

from __future__ import annotations

import ast
import os
import unittest

CORE_MODULES = ["models.py", "geometry.py", "config.py", "pose_controller.py"]
ADAPTER_MODULE = "adapter_example.py"

FORBIDDEN_PREFIXES = (
    "comm",          # backend comm.server / orchestrator
    "django",
    "redis",
    "socket",
    "threading",
    "asyncio",
    "cv2",
    "numpy",         # stdlib-only 원칙
    "torch",
    "ultralytics",   # YOLO
    "channels",
    "rest_framework",
    "server",        # comm.server 를 직접 import 하는 경우
    "pipeline",
    "requests",
)

CONTROLLER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def imported_names(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # relative import(level>0)는 같은 패키지 내부이므로 허용
            if node.level and node.level > 0:
                continue
            if node.module:
                names.append(node.module)
    return names


class TestCoreIndependence(unittest.TestCase):
    def _check(self, filename: str) -> None:
        path = os.path.join(CONTROLLER_DIR, filename)
        for name in imported_names(path):
            top = name.split(".")[0]
            self.assertNotIn(
                top, FORBIDDEN_PREFIXES,
                f"{filename} 가 금지된 모듈을 import 함: {name}",
            )

    def test_core_modules_clean(self) -> None:
        for m in CORE_MODULES:
            self._check(m)

    def test_adapter_clean(self) -> None:
        # adapter 도 backend/network 를 실제로 import 하지 않아야 함(duck-typing 만)
        self._check(ADAPTER_MODULE)

    def test_core_importable_without_backend(self) -> None:
        # backend 없이도 import 되는지 확인
        import importlib
        for mod in ["controller.models", "controller.geometry",
                    "controller.config", "controller.pose_controller"]:
            importlib.import_module(mod)


if __name__ == "__main__":
    unittest.main()
