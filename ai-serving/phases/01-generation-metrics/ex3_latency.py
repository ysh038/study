"""실습 3. TTFT, TPOT, E2E 직접 재기 (3장)

실습 2의 generate_stream()을 돌리면서 토큰이 나오는 시각을 기록한다.
프롬프트 길이와 출력 길이를 각각 바꿔 가며 무엇이 무엇을 늘리는지 본다.

    uv run python phases/01-generation-metrics/ex3_latency.py
    → results/latency.json, results/ttft_vs_prompt.png, results/e2e_vs_output.png
"""

# %%
import time
from dataclasses import asdict, dataclass

import torch

from ex2_greedy_loop import generate_stream


@dataclass
class Timing:
    ttft: float  # 초. 요청 시작 → 첫 토큰
    tpot: float  # 초. 첫 토큰 이후 토큰 사이 간격의 평균
    e2e: float  # 초. 요청 시작 → 마지막 토큰
    n_tokens: int
    gaps: list[float]  # 초. 토큰 사이 간격 하나하나 (길이 n_tokens - 1)


def measure(model, input_ids: torch.Tensor, max_new_tokens: int) -> Timing:
    """generate_stream()을 끝까지 돌리며 시간을 잰다. 멈춤 토큰 없이 max_new_tokens개를 모두 만든다."""
    # TODO
    # 1) t0 = time.perf_counter()  ← 요청을 보낸 시각
    # 2) generate_stream(model, input_ids, max_new_tokens)을 for문으로 돌리며,
    #    토큰이 yield 될 때마다 time.perf_counter()를 리스트에 저장
    # 3) ttft, gaps, tpot, e2e를 계산해 Timing으로 돌려준다
    #
    # 생각해 볼 것: GPU는 비동기로 도는데, yield 시각을 '토큰이 실제로 만들어진 시각'으로 봐도 될까?
    #   (힌트: generate_stream 안의 .item())
    raise NotImplementedError


def make_prompt_ids(tokenizer, n_tokens: int, device) -> torch.Tensor:
    """정확히 n_tokens 길이의 프롬프트를 만든다 (내용보다 길이가 중요한 실험이라 문단을 반복해서 자른다)."""
    paragraph = "모델 서빙은 다 만든 AI 모델을 서버에 올려두고 수많은 요청에 빠르고 싸게 답하게 만드는 일이다. "
    ids = tokenizer(paragraph * (n_tokens // 10 + 10), return_tensors="pt").input_ids[:, :n_tokens]
    assert ids.shape[1] == n_tokens
    return ids.to(device)


PROMPT_LENGTHS = [32, 128, 512, 1024, 2048]  # TTFT 실험: 출력은 8토큰 고정
OUTPUT_LENGTHS = [8, 16, 32, 64, 128]  # E2E 실험: 프롬프트는 128토큰 고정


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from servlab.models import load
    from servlab.results import results_dir, save_json

    tok, model, device = load()
    print(f"장치 {device}, 정밀도 {model.dtype}")

    # 워밍업: 첫 실행은 커널 준비 등으로 느리다. 얼마나 느린지 기록만 하고 버린다.
    first = measure(model, make_prompt_ids(tok, 32, device), 4)
    second = measure(model, make_prompt_ids(tok, 32, device), 4)
    print(f"워밍업: 첫 실행 TTFT {first.ttft * 1e3:.0f} ms vs 두 번째 {second.ttft * 1e3:.0f} ms")

    # 입력 길이가 처음 보는 크기면 장치가 그 크기용 커널을 새로 준비해서 느려진다.
    # 그래서 설정마다 한 번 버리고 두 번째 측정을 쓴다. (궁금하면 첫 측정값도 출력해 비교해 보세요)
    by_prompt = []
    for n in PROMPT_LENGTHS:
        measure(model, make_prompt_ids(tok, n, device), 8)
        t = measure(model, make_prompt_ids(tok, n, device), 8)
        by_prompt.append({"prompt_tokens": n, **asdict(t)})
        print(f"프롬프트 {n:>5}토큰 → TTFT {t.ttft * 1e3:7.1f} ms, TPOT {t.tpot * 1e3:6.1f} ms")

    by_output = []
    for n in OUTPUT_LENGTHS:
        measure(model, make_prompt_ids(tok, 128, device), n)
        t = measure(model, make_prompt_ids(tok, 128, device), n)
        predicted = t.ttft + (t.n_tokens - 1) * by_output[0]["tpot"] if by_output else t.e2e
        by_output.append({"output_tokens": n, "formula_e2e": predicted, **asdict(t)})
        print(f"출력 {n:>4}토큰 → E2E {t.e2e:6.2f} s (공식 예측 {predicted:6.2f} s), TPOT {t.tpot * 1e3:6.1f} ms")

    path = save_json(
        {"device": str(device), "dtype": str(model.dtype), "by_prompt": by_prompt, "by_output": by_output},
        __file__,
        "latency.json",
    )
    print(f"\n저장: {path}")

    out = results_dir(__file__)
    plt.figure()
    plt.plot(PROMPT_LENGTHS, [r["ttft"] * 1e3 for r in by_prompt], marker="o")
    plt.xlabel("prompt tokens")
    plt.ylabel("TTFT (ms)")
    plt.title(f"TTFT vs prompt length ({device})")
    plt.grid(alpha=0.3)
    plt.savefig(out / "ttft_vs_prompt.png", dpi=120)

    # E2E 공식의 TPOT는 가장 짧은 출력(8토큰)에서 잰 값을 그대로 쓴다.
    # 출력이 길어질수록 측정값과 공식이 벌어진다면, TPOT가 일정하지 않다는 뜻이다. 왜일까?
    plt.figure()
    plt.plot(OUTPUT_LENGTHS, [r["e2e"] for r in by_output], marker="o", label="measured")
    plt.plot(OUTPUT_LENGTHS, [r["formula_e2e"] for r in by_output], marker="x", linestyle="--", label="TTFT + (N-1)×TPOT₈")
    plt.xlabel("output tokens")
    plt.ylabel("E2E (s)")
    plt.title(f"E2E vs output length ({device})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out / "e2e_vs_output.png", dpi=120)
    print(f"그래프: {out / 'ttft_vs_prompt.png'}, {out / 'e2e_vs_output.png'}")

    # 더 해보기
    # - 마지막 실행의 gaps를 그래프로 그려 보기: 토큰 순번이 늘수록 간격이 어떻게 변하나? (→ Phase 02)
    # - load(device="cpu")로 바꿔 같은 실험 → MPS와 몇 배 차이?
