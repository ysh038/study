import importlib.util
import sys
from pathlib import Path

import pytest


def load_exercise(test_file: str, name: str):
    """테스트 파일과 같은 폴더의 실습 모듈을 불러온다.

    phase마다 ex1_*.py 같은 이름이 겹쳐도 충돌하지 않도록 phase 이름을 붙인 모듈명으로 등록한다.
    """
    folder = Path(test_file).resolve().parent
    module_name = f"_{folder.name}_{name}".replace("-", "_")
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))  # 실습 파일끼리 서로 import 할 수 있게
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(folder))
    sys.modules[module_name] = module
    return module


def call_or_skip(fn, *args, **kwargs):
    """아직 구현하지 않은 실습(NotImplementedError)은 실패 대신 skip으로 보여준다."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"TODO: {fn.__name__} 아직 구현 전")
