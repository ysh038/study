# Phase 00 · 환경 준비

## 목표

- 이 트랙 전용 파이썬 환경을 만들고, Mac GPU(MPS)를 쓸 수 있는지 확인해요.
- 실습 모델을 미리 내려받아 둬요.

## 체크리스트

모든 명령은 `ai-serving/` 폴더에서 실행해요.

- [ ] 환경 설치
  ```bash
  uv sync
  ```
- [ ] 환경 점검. 장치가 `mps`로 나오고 matmul 결과가 출력되면 정상이에요.
  ```bash
  uv run python phases/00-setup/check_env.py
  ```
- [ ] 실습 모델 내려받기 (약 1GB, `~/.cache/huggingface`에 저장돼요)
  ```bash
  uv run python phases/00-setup/check_env.py --download
  ```
- [ ] 빠른 테스트가 통과하는지 확인 (아직 구현 안 한 과제는 skip으로 나와요)
  ```bash
  uv run pytest -m "not slow"
  ```
- [ ] (선택) VS Code에서 `ai-serving/.venv`를 인터프리터로 선택하면 `# %%` 셀 단위 실행을 쓸 수 있어요.
- [ ] (Phase 4부터) 클라우드 GPU 계정 준비: Colab Pro, RunPod, Lambda 중 하나. 무료 T4는 BF16과 FlashAttention 2/3을 지원하지 않아요.

## 통과 기준

- [ ] `check_env.py`가 장치, 정밀도, 모델 로딩까지 모두 OK로 끝난다.
- [ ] `lab-log.md`에 첫 기록을 남겼다.
