"""실습 1. 토큰 (2장)

모델은 글자가 아니라 토큰 단위로 읽고 쓴다. 두 토크나이저로 같은 문장을 쪼개 본다.

    uv run python phases/01-generation-metrics/ex1_tokens.py
"""

# %%
from transformers import AutoTokenizer

from servlab.models import DEFAULT_MODEL

TOKENIZERS = {
    "gpt2": "openai-community/gpt2",  # 2019년, 영어 위주로 학습
    "qwen2.5": DEFAULT_MODEL,  # 다국어 학습
}

SENTENCES = {
    "ko": ["안녕하세요", "토스 상담원입니다. 무엇을 도와드릴까요?"],
    "en": ["Hello, world!", "The quick brown fox jumps over the lazy dog."],
}


def tokenize(tokenizer, text: str) -> tuple[list[int], list[str]]:
    """토큰 id 목록과, 각 토큰이 어떤 조각인지 보여주는 문자열 목록을 돌려준다."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    # 토큰을 하나씩 글자로 되돌려 본다. 한글 한 글자(3바이트)가 여러 토큰으로 쪼개지면
    # 조각 하나만으로는 글자가 안 돼서 '�'로 보여요. GPT-2에서 특히 많이 보일 거예요.
    pieces = [tokenizer.decode([i]) for i in ids]
    return ids, pieces


def tokens_per_char(tokenizer, texts: list[str]) -> float:
    """여러 문장을 합쳐 '글자 하나당 평균 토큰 수'를 계산한다."""
    # TODO: tokenize()를 써서 전체 토큰 수 ÷ 전체 글자 수를 돌려주세요.
    raise NotImplementedError


# %%
if __name__ == "__main__":
    for label, name in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(name)
        print(f"\n== {label}  (어휘 크기 {tok.vocab_size:,})")
        for lang, texts in SENTENCES.items():
            for text in texts:
                ids, pieces = tokenize(tok, text)
                print(f"[{lang}] {len(text):>3}자 → {len(ids):>3}토큰 | {pieces}")
        try:
            for lang, texts in SENTENCES.items():
                print(f"  {lang} 글자당 토큰: {tokens_per_char(tok, texts):.2f}")
        except NotImplementedError:
            print("  (TODO: tokens_per_char 구현 후 다시 실행)")

    # 더 해보기
    # - 내가 자주 쓰는 문장, 코드 한 줄, 이모지를 SENTENCES에 넣어 보기
    # - 같은 내용을 한국어/영어로 썼을 때 토큰 수 차이 → 같은 답을 해도 비용이 다를까?
