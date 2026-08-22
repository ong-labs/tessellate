# 2026-08-22 캡처 실행 — resnet18 / calibrated_neurosim_v1_5

발표자료(10·11·13페이지)에 쓰는 실측 실행의 전체 기록.

| 파일 | 내용 |
|---|---|
| `terminal.log` | 터미널 출력 **원본 바이트** (cp949). 수정 금지 |
| `terminal.utf8.log` | 같은 내용의 UTF-8 사본 |
| `events.jsonl` | 관측 이벤트 (LLM rationale 전문·이상 징후·보정계수 포함) |
| `report.html` | `--dashboard-out`이 생성한 대시보드 리포트 |
| `checkpoints.sqlite` | LangGraph 체크포인트 (thread `demo-neurosim-0p88`) |
| `run.sh` | 실행에 사용한 명령 그대로 |

리포트는 체크포인트에서 언제든 다시 만들 수 있다 (`_snap/report_from_ckpt.py` 참고).

## checkpoints.sqlite 는 저장소에 없습니다

217MB(GitHub 파일당 100MB 제한 초과)라 `.gitignore` 로 제외했습니다.
로컬에는 이 디렉터리에 그대로 있고, 리포트를 다시 뽑을 때 그 파일을 씁니다.
저장소에 남는 것은 터미널 로그(`terminal.log`), 이벤트(`events.jsonl`),
이미 생성된 리포트(`report.html`)이며, 발표에서 인용하는 수치는 전부
이 셋으로 검증 가능합니다.
