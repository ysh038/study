import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """cuda > mps > cpu 순서로 쓸 수 있는 장치를 고른다. prefer로 강제할 수 있다."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """장치별 기본 숫자 정밀도 (문서 6장). CPU는 정확도 비교용으로 FP32를 쓴다."""
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16  # T4 같은 구형 GPU
    return torch.bfloat16


def sync(device: torch.device) -> None:
    """GPU 작업은 비동기로 돌아서, 시간을 재기 전에 끝날 때까지 기다려야 정확하다."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
