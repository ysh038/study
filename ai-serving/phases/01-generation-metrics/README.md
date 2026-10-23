# Phase 01 · LLM 생성 과정과 지표

읽을 장: 2장(LLM이 답을 만드는 과정), 3장(속도를 재는 지표)

## 목표

- 모델은 글자가 아니라 **토큰**을 다룬다는 걸 직접 본다.
- `model.generate()` 없이 **자기회귀 생성 루프**를 직접 만든다.
- 그 루프로 **TTFT, TPOT, E2E**를 직접 재고, 프롬프트 길이와 출력 길이가 각각 무엇을 늘리는지 확인한다.
- **p50/p95/p99**를 직접 계산하고, 평균만 보면 안 되는 이유를 숫자로 확인한다.

## 실습 순서

각 실습 전에 [notes.md](notes.md)에 예측부터 적어요. 코드에서 `TODO`라고 적힌 부분을 채우면 돼요.

| # | 파일 | 할 일 | 실행 / 채점 |
|---|---|---|---|
| 1 | [ex1_tokens.py](ex1_tokens.py) | 두 토크나이저로 한국어·영어 문장을 쪼개 보고 글자당 토큰 수 비교 | `uv run python phases/01-generation-metrics/ex1_tokens.py` |
| 2 | [ex2_greedy_loop.py](ex2_greedy_loop.py) | `generate_stream()` 구현: 다음 토큰 예측 → 입력 끝에 붙이기 반복 | `uv run pytest phases/01-generation-metrics -k greedy` |
| 3 | [ex3_latency.py](ex3_latency.py) | `measure()` 구현 후 프롬프트 길이·출력 길이를 바꿔 가며 측정, 그래프 저장 | `uv run python phases/01-generation-metrics/ex3_latency.py` |
| 4 | [ex4_percentiles.py](ex4_percentiles.py) | `percentile()` 구현, 같은 요청 반복 측정으로 p50/p95/p99 계산 | `uv run pytest phases/01-generation-metrics -k percentile` 후 파일 실행 |

전체 자기채점은 `uv run pytest phases/01-generation-metrics`로 해요. 아직 구현하지 않은 과제는 skip으로 표시돼요.
`slow` 테스트는 실제 모델을 CPU에 올려서 돌리기 때문에 수십 초 걸릴 수 있어요.

## 통과 기준

- [ ] 2번 테스트 통과: 내가 만든 루프의 출력이 `model.generate()`의 greedy 출력과 토큰 단위로 같다.
- [ ] `results/ttft_vs_prompt.png`, `results/e2e_vs_output.png` 두 그래프로 "프롬프트를 늘리면 TTFT가, 출력을 늘리면 E2E가 늘어난다"를 설명할 수 있다.
- [ ] `E2E ≈ TTFT + (N−1) × TPOT` 공식이 내 측정에서 얼마나 맞는지, 어긋난다면 왜인지 설명할 수 있다.
- [ ] 평균, p50, p99가 서로 다르게 나오는 예를 내 데이터로 보여줄 수 있다.

## 다음 phase와의 연결

이번 루프는 일부러 **KV 캐시 없이**(`use_cache=False`) 만들어요. 그래서 토큰이 쌓일수록 매 스텝이 조금씩 느려지고, TPOT가 일정하다는 E2E 공식이 정확히 맞지 않을 거예요.
그 이유를 Phase 02에서 KV 캐시를 붙여 직접 확인해요.
