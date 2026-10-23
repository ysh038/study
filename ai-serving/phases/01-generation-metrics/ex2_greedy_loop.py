"""실습 2. 자기회귀 생성 루프 직접 만들기 (2장)

model.generate() 없이 '다음 토큰 하나 예측 → 입력 끝에 붙이기'를 반복한다.
이번에는 일부러 KV 캐시를 쓰지 않는다(use_cache=False). Phase 02에서 캐시를 붙여 비교한다.

    uv run python phases/01-generation-metrics/ex2_greedy_loop.py     # 직접 돌려 보기
    uv run pytest phases/01-generation-metrics -k greedy              # 자기채점
"""

# %%
from collections.abc import Iterator

import torch


@torch.inference_mode()
def generate_stream(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_ids: set[int] | None = None,
) -> Iterator[int]:
    """새로 만든 토큰 id(int)를 하나씩 yield 한다.

    input_ids: shape (1, seq_len)
    stop_ids: 이 중 하나가 나오면 그 토큰까지 yield 하고 멈춘다.
    """
    # TODO: max_new_tokens 번까지 아래를 반복하세요.
    # 1) logits = model(input_ids=input_ids, use_cache=False).logits   # shape (1, seq_len, vocab)
    # 2) 마지막 위치의 로짓에서 argmax로 다음 토큰 id를 고른다 (greedy)
    # 3) .item()으로 파이썬 int로 바꿔 yield 한다
    # 4) stop_ids에 있으면 멈춘다
    # 5) input_ids 끝에 새 토큰을 붙인다 (torch.cat, shape (1, seq_len + 1))
    #
    # 생각해 볼 것: 매 반복마다 모델에 들어가는 입력 길이는 어떻게 변하나?
    raise NotImplementedError


def greedy_generate(model, input_ids, max_new_tokens, stop_ids=None) -> list[int]:
    return list(generate_stream(model, input_ids, max_new_tokens, stop_ids))


# %%
if __name__ == "__main__":
    from servlab.models import StreamPrinter, encode_chat, load, stop_token_ids

    tok, model, device = load()
    input_ids = encode_chat(tok, "모델 서빙이 뭔지 두 문장으로 설명해줘.", device)
    print(f"프롬프트 {input_ids.shape[1]}토큰, 장치 {device}\n")

    printer = StreamPrinter(tok)
    for token_id in generate_stream(model, input_ids, 80, stop_token_ids(model, tok)):
        printer.put(token_id)
    printer.end()
