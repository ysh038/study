"""실습 4. 평균 대신 p50 / p95 / p99 (3장)

    uv run pytest phases/01-generation-metrics -k percentile   # 먼저 percentile() 자기채점
    uv run python phases/01-generation-metrics/ex4_percentiles.py
"""

# %%
import math


def percentile(values: list[float], p: float) -> float:
    """p 백분위수 (0 <= p <= 100). numpy.percentile의 기본 방식(선형 보간)과 같은 값을 돌려준다.

    방식: 값을 정렬하고, 위치 k = (n - 1) * p / 100 를 구한다.
          k가 정수면 그 자리 값, 아니면 앞뒤 두 값을 소수 부분만큼 섞는다.
    예: [10, 20, 30, 40], p=50 → k=1.5 → 20과 30의 중간 = 25
    """
    # TODO: numpy 없이 구현하세요 (sorted, math.floor 정도면 충분해요)
    raise NotImplementedError


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def show(title: str, values: list[float], unit: str = "s") -> None:
    s = summarize(values)
    print(f"{title}: " + ", ".join(f"{k} {v:.3f}{unit}" for k, v in s.items()))


# %%
if __name__ == "__main__":
    # 1) 머릿속 실험: notes.md에 예측을 먼저 적고 실행하세요.
    show("0.2초 99개 + 5초 1개", [0.2] * 99 + [5.0])
    show("0.2초 98개 + 5초 2개", [0.2] * 98 + [5.0] * 2)
    show("0.2초 95개 + 5초 5개", [0.2] * 95 + [5.0] * 5)

    # 2) 실제 측정: 같은 요청을 여러 번 보내 TTFT 분포를 본다.
    from ex3_latency import make_prompt_ids, measure

    from servlab.models import load
    from servlab.results import save_json

    tok, model, device = load()
    input_ids = make_prompt_ids(tok, 256, device)
    measure(model, input_ids, 2)  # 워밍업

    runs = 50
    ttfts = []
    for i in range(runs):
        ttfts.append(measure(model, input_ids, 2).ttft)
        # TODO(선택): 중간에 다른 무거운 앱을 켜거나, i % 10 == 0 일 때 time.sleep을 넣는 등
        #             '가끔 느린 요청'을 일부러 만들어 p99가 어떻게 움직이는지 보세요.
    show(f"실측 TTFT ({runs}회, {device})", [t * 1e3 for t in ttfts], unit="ms")
    print("→ 정렬해서 가장 느린 3개:", [f"{t * 1e3:.1f}ms" for t in sorted(ttfts)[-3:]])

    save_json({"device": str(device), "ttft_s": ttfts, "summary": summarize(ttfts)}, __file__, "ttft_percentiles.json")
