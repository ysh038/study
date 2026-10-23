import json
from pathlib import Path


def results_dir(script_file: str) -> Path:
    """실습 파일 옆의 results/ 폴더 경로 (없으면 만든다)."""
    path = Path(script_file).resolve().parent / "results"
    path.mkdir(exist_ok=True)
    return path


def save_json(obj, script_file: str, name: str) -> Path:
    path = results_dir(script_file) / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    return path
