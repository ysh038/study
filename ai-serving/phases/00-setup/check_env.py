"""환경 점검.

    uv run python phases/00-setup/check_env.py             # 장치와 라이브러리 확인
    uv run python phases/00-setup/check_env.py --download  # 실습 모델까지 내려받아 한 번 돌려 보기
"""

import argparse
import platform
import time

import torch
import transformers

from servlab.device import default_dtype, pick_device, sync
from servlab.models import DEFAULT_MODEL

TOKENIZER_ONLY = ["openai-community/gpt2"]  # Phase 1 실습 1에서 토크나이저 비교용


def check_device() -> torch.device:
    print(f"python        {platform.python_version()}")
    print(f"torch         {torch.__version__}")
    print(f"transformers  {transformers.__version__}")
    print(f"cuda          {torch.cuda.is_available()}")
    print(f"mps           {torch.backends.mps.is_available()}")

    device = pick_device()
    dtype = default_dtype(device)
    print(f"→ 사용할 장치 {device}, 기본 정밀도 {dtype}")

    # 큰 행렬 곱 한 번으로 장치가 실제로 계산하는지, 대략 얼마나 빠른지 본다 (6장)
    n = 2048
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    a @ b  # 워밍업: 첫 실행은 커널 준비 때문에 느리다
    sync(device)
    reps = 20
    start = time.perf_counter()
    for _ in range(reps):
        a @ b
    sync(device)
    elapsed = (time.perf_counter() - start) / reps
    tflops = 2 * n**3 / elapsed / 1e12
    print(f"matmul {n}x{n} {dtype}: {elapsed * 1e3:.2f} ms → 약 {tflops:.1f} TFLOPS")
    return device


def check_models() -> None:
    from transformers import AutoTokenizer

    from servlab.models import StreamPrinter, encode_chat, load, stop_token_ids

    for name in TOKENIZER_ONLY:
        AutoTokenizer.from_pretrained(name)
        print(f"토크나이저 OK: {name}")

    tok, model, device = load()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 OK: {DEFAULT_MODEL} ({n_params / 1e6:.0f}M 파라미터, {device}, {model.dtype})")

    input_ids = encode_chat(tok, "안녕! 한 문장으로 자기소개 해줘.", device)
    out = model.generate(input_ids, max_new_tokens=40, do_sample=False, eos_token_id=list(stop_token_ids(model, tok)))
    printer = StreamPrinter(tok)
    for token_id in out[0, input_ids.shape[1]:].tolist():
        printer.put(token_id)
    printer.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="실습 모델을 내려받아 한 번 돌려 본다")
    args = parser.parse_args()

    check_device()
    if args.download:
        check_models()
    print("OK")
