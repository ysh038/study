# AI 서빙 엔지니어링

원본 자료: [AI 서빙 엔지니어링 학습 가이드](https://claude.ai/artifact/K2V98qkbGZEx8MhQrUmbJr) (토스 서빙 발표 해설, 1~19장)

문서의 장 순서대로 **직접 재 보고 직접 만들어 보며** 기초를 다지는 트랙이에요.

## 진행 현황

| Phase | 주제 | 문서 장 | 환경 | 상태 |
|---|---|---|---|---|
| [00](phases/00-setup/) | 환경 준비 | - | Mac | 🟨 |
| [01](phases/01-generation-metrics/) | LLM 생성 과정과 지표 | 2, 3 | Mac | ⬜ |
| 02 | KV 캐시 | 4 | Mac | ⬜ |
| 03 | 배칭과 스케줄링 | 5 | Mac + GPU | ⬜ |
| 04 | GPU 기초와 커널 | 6, 7 | 클라우드 GPU | ⬜ |
| 05 | 어텐션 변형 | 8 | Mac | ⬜ |
| 06 | 서빙 엔진 직접 운용 | 9, 10, 11 | 클라우드 GPU | ⬜ |
| 07 | 라우팅과 부하 제어 | 12, 13 | Mac | ⬜ |
| 08 | 쿠버네티스와 오토스케일링 | 14 | Mac | ⬜ |
| 09 | 음성 파이프라인 | 15 | Mac | ⬜ |
| 10 | 운영 + 캡스톤 | 16, 17 | Mac + GPU | ⬜ |

상태 표시: ⬜ 시작 전 · 🟨 진행 중 · ✅ 통과 기준 충족

## 학습 규칙

모든 실습은 같은 순서로 해요.

1. 해당 장을 읽어요.
2. 실험 전에 결과를 **예측해서** 그 phase의 `notes.md`에 적어요.
3. 실험을 돌리고 숫자를 기록해요. 그래프와 json은 `results/`에 저장돼요.
4. 예측과 결과가 왜 다른지 `notes.md`에 정리해요.
5. 그날 한 일을 [lab-log.md](lab-log.md)에 한두 줄 남겨요.

phase의 `README.md`에 있는 **통과 기준**을 내가 측정한 숫자로 설명할 수 있으면 다음 phase로 넘어가요.

## 폴더 구조

```
ai-serving/
├── README.md            # 이 파일: 로드맵과 진행 현황
├── lab-log.md           # 날짜별 학습 일지
├── pyproject.toml       # 이 트랙 전용 파이썬 환경 (uv)
├── src/servlab/         # 여러 phase가 같이 쓰는 코드 (모델 로딩, 장치 선택 등)
├── phases/
│   ├── 00-setup/
│   └── 01-generation-metrics/
│       ├── README.md    # 목표, 실습 순서, 통과 기준
│       ├── notes.md     # 예측 → 결과 → 해석
│       ├── ex*.py       # 실습 코드 (TODO를 채워요)
│       ├── test_*.py    # 자기채점 테스트
│       └── results/     # 실험 결과 (그래프, json)
├── templates/phase/     # 새 phase를 만들 때 복사하는 뼈대
└── scripts/new_phase.sh
```

공통 코드는 `src/servlab/`에만 두고, phase 폴더에는 그 단계의 실습만 둬요. 한 phase에서 만든 코드가 다음 phase에서도 계속 필요해지면 그때 `servlab`으로 옮겨요.

## 자주 쓰는 명령

모든 명령은 `ai-serving/` 폴더에서 실행해요.

```bash
uv sync                                             # 환경 설치/갱신
uv run python phases/00-setup/check_env.py          # 환경 점검
uv run python phases/01-generation-metrics/ex1_tokens.py   # 실습 실행
uv run pytest phases/01-generation-metrics          # 자기채점 (구현 안 한 과제는 skip으로 보여요)
uv run pytest -m "not slow"                         # 모델을 띄우지 않는 빠른 테스트만
./scripts/new_phase.sh 02 kv-cache "KV 캐시"        # 새 phase 폴더 만들기
```

`.py` 실습 파일은 `# %%`로 셀이 나뉘어 있어서, VS Code나 PyCharm에서 셀 단위로 실행해 볼 수도 있어요.
