"""Phase 01 자기채점. 구현하지 않은 과제는 skip으로 표시된다.

    uv run pytest phases/01-generation-metrics
"""

import random

import numpy as np
import pytest
import torch
from transformers import GenerationConfig

from servlab.testing import call_or_skip, load_exercise

ex2 = load_exercise(__file__, "ex2_greedy_loop")
ex3 = load_exercise(__file__, "ex3_latency")
ex4 = load_exercise(__file__, "ex4_percentiles")


@pytest.fixture(scope="module")
def cpu_model():
    """정확한 비교를 위해 CPU + FP32로 띄운다."""
    from servlab.models import load

    return load(device="cpu")


# --- 실습 4 ---------------------------------------------------------------


@pytest.mark.parametrize("p", [0, 1, 25, 50, 90, 95, 99, 99.9, 100])
def test_percentile_matches_numpy(p):
    rng = random.Random(p)
    for n in [1, 2, 7, 100]:
        values = [rng.expovariate(5) for _ in range(n)]
        got = call_or_skip(ex4.percentile, values, p)
        assert got == pytest.approx(float(np.percentile(values, p)))


def test_percentile_does_not_mutate_input():
    values = [3.0, 1.0, 2.0]
    call_or_skip(ex4.percentile, values, 50)
    assert values == [3.0, 1.0, 2.0]


# --- 실습 2 ---------------------------------------------------------------


@pytest.mark.slow
def test_greedy_matches_generate(cpu_model):
    tok, model, device = cpu_model
    from servlab.models import encode_chat, stop_token_ids

    input_ids = encode_chat(tok, "서빙 엔지니어가 하는 일을 한 문장으로.", device)
    stop = stop_token_ids(model, tok)
    n = 16

    mine = call_or_skip(ex2.greedy_generate, model, input_ids, n, stop)

    # 모델 기본 설정(Qwen은 repetition_penalty=1.1, 샘플링)이 섞이지 않게 순수 greedy로 덮어쓴다
    config = GenerationConfig(
        max_new_tokens=n, do_sample=False, repetition_penalty=1.0, eos_token_id=list(stop), pad_token_id=tok.eos_token_id
    )
    reference = model.generate(input_ids, generation_config=config)[0, input_ids.shape[1]:].tolist()

    assert mine == reference, f"\n내 루프: {tok.decode(mine)!r}\ngenerate: {tok.decode(reference)!r}"


@pytest.mark.slow
def test_greedy_stops_at_stop_token(cpu_model):
    tok, model, device = cpu_model
    from servlab.models import encode_chat

    input_ids = encode_chat(tok, "숫자 1만 출력해.", device)
    first = call_or_skip(ex2.greedy_generate, model, input_ids, 1)
    out = ex2.greedy_generate(model, input_ids, 10, stop_ids={first[0]})
    assert out == first, "stop_ids에 든 토큰이 나오면 그 토큰까지만 내고 멈춰야 해요"


@pytest.mark.slow
def test_greedy_does_not_modify_input(cpu_model):
    tok, model, device = cpu_model
    input_ids = torch.tensor([[tok.encode("hello")[0]]])
    before = input_ids.clone()
    call_or_skip(ex2.greedy_generate, model, input_ids, 3)
    assert torch.equal(input_ids, before)


# --- 실습 3 ---------------------------------------------------------------


@pytest.mark.slow
def test_measure_is_consistent(cpu_model):
    tok, model, device = cpu_model
    input_ids = ex3.make_prompt_ids(tok, 16, device)
    t = call_or_skip(ex3.measure, model, input_ids, 5)

    assert t.n_tokens == 5
    assert len(t.gaps) == 4
    assert 0 < t.ttft <= t.e2e
    assert t.tpot == pytest.approx(sum(t.gaps) / len(t.gaps))
    assert t.e2e == pytest.approx(t.ttft + sum(t.gaps)), "E2E = TTFT + 모든 간격의 합 이어야 해요"
