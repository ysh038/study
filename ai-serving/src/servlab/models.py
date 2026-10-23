import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from servlab.device import default_dtype, pick_device

# 실습 기본 모델. 0.5B라 Mac에서도 가볍게 돌고, GQA 등 최신 구조를 갖고 있다.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load(model_name: str = DEFAULT_MODEL, device: str | None = None, dtype: torch.dtype | None = None):
    """(tokenizer, model, device)를 돌려준다. 처음 실행하면 Hugging Face에서 내려받는다."""
    dev = pick_device(device)
    dtype = dtype or default_dtype(dev)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(dev).eval()
    return tokenizer, model, dev


def encode_chat(tokenizer, user_text: str, device: torch.device) -> torch.Tensor:
    """채팅 형식 프롬프트를 input_ids (1, seq_len) 텐서로 만든다."""
    messages = [{"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def stop_token_ids(model, tokenizer) -> set[int]:
    """생성을 멈출 토큰 id들 (모델의 generation_config와 tokenizer의 eos를 합친 것)."""
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids] if ids is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


class StreamPrinter:
    """토큰을 하나씩 받아 화면에 이어서 출력한다.

    한글은 토큰 하나가 글자의 일부 바이트일 때가 있어서, 토큰을 하나씩 decode하면 깨진다.
    그래서 지금까지 받은 토큰 전체를 decode한 뒤 새로 늘어난 부분만 출력한다.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.printed = ""

    def put(self, token_id: int) -> None:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):  # 아직 글자가 완성되지 않음
            return
        print(text[len(self.printed):], end="", flush=True)
        self.printed = text

    def end(self) -> None:
        print()
