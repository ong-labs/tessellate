# 아키텍처 상세

그래프 구성과 노드별 구현 범위입니다. 개요와 다이어그램은 [README.md](../README.md#아키텍처)에 있습니다.

## 완료 상태의 의미

이 파이프라인의 `__end__` 상태는 최종 확정이 아니라 사람이 검토할 후보가 준비되었다는 뜻입니다. `hitl_human_approval`도 예외 처리 경로가 아니라, 이 시스템의 신뢰성 한계 때문에 실제로 자주 타는 설계상 핵심적인 경로입니다.

## HITL 이후의 두 경로

`hitl_human_approval`에서 나가는 두 경로는 `graph.py`의 `hitl_router`가 고릅니다: 연구자가 새 탐색 범위를 승인하면 `@planner`로 돌아가고, 더 진행하지 않고 **지금까지의 최선 후보로 종결**하기로 하면(수동 [3]번 선택, 또는 `--auto-hitl`이 라운드 상한에 도달) `low_confidence_conclusion` 표시와 함께 바로 `__end__`로 갑니다. 이 경우 `@precision_verifier`를 건너뜁니다(애초에 목표를 만족한 적 없는 후보라 정밀 재검증이 확인할 게 없기 때문). 이 표시가 붙은 결과는 `is_converged`를 `True`로 보고하지 않으며, CLI 최종 출력과 대시보드 모두 "검토 필요"로 구분해 보여줍니다.

## 노드별 구현 범위

| 노드 | 역할 | 실제 구현 여부 |
|---|---|---|
| `@planner` | LHS 워밍업 + IDW 대리모델 기반 다중목적(accuracy/energy_pj/noc_latency_ms) NSGA-II 탐색(`pymoo`, stage별 독립, 최대 40차원)이 수치를 확정하고, LLM은 도구 호출로 근거(rationale) 설명과 이상 징후(anomaly) 플래깅만 담당 (수치 판단 없음) | 실제 |
| `@tuner` | 크로스바 컬럼(출력 채널)별 독립 스케일을 쓰는 column-wise 가중치 fake-quantization + partial-sum(ADC 직전 출력) 양자화 + 구조적 컬럼 프루닝, 실제 QAT fine-tuning (`tools/qat.py`) | 실제 (CIFAR10 서브셋, CUDA/MPS/CPU 자동 감지) |
| `@mapper` / `@profiler` | NoC 홉카운트/대역폭 매핑, ADC/DAC/크로스바 에너지·지연 물리 시뮬레이션 | 실제 (fast-approximation) |
| `@verifier` | IR-drop/노이즈 마진 기반 수렴 판정 | 실제 |
| `@evaluator` | Pydantic 스키마 검증, NSGA-II 스타일 Pareto rank, 재탐색/HITL 라우팅 | 실제 |
| `@precision_verifier` | `@evaluator`가 수렴으로 판단한 후보만 골라 2단계 정밀 재검증하고, 그 결과로 `calibration_factors`를 갱신 (`tools/precision_check.py`) | 부분 실제 (ADC 에너지: `hwcomponents_adc` 문헌 기반 모델(항상) / DAC·크로스바-MAC 에너지: `hwcomponents_neurosim`(Docker 한정) / SRAM 버퍼 에너지: `hwcomponents_cacti`(Docker 한정), 로컬(비-Docker) 환경에서는 네 항 모두 fast-approximation 자리 표시자로 폴백) |

## 정밀 재검증의 환경 의존성

`@precision_verifier`의 네 항목은 실행 환경에 따라 실제 모델과 근사 자리 표시자로 갈립니다. 조건은 [`limitations.md`](limitations.md)와 `tools/precision_check.py`의 모듈 docstring에 있습니다.
