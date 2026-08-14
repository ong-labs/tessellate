# 개념 가이드 — CIM과 AutoCIM-Agent를 처음 접하는 사람을 위해

> 이 문서는 [`README.md`](../README.md)를 대체하지 않습니다. README는 "이 프로젝트가 무엇을 실제로 구현했고 무엇을 못 하는가"를 다루고, 이 문서는 그걸 읽기 전에 필요한 **배경 개념**을 처음부터 설명합니다. CIM 반도체나 HW-SW 공동설계를 한 번도 접해본 적이 없어도 따라올 수 있도록 순서대로 구성했습니다.
>
> 읽는 순서 제안: 이 문서 → [`README.md`](../README.md) → (더 깊게 알고 싶다면) [`docs/research_plan.md`](research_plan.md).

---

## 1. 왜 "CIM"인가 — 폰 노이만 병목

일반적인 컴퓨터(CPU/GPU)는 **메모리**(가중치·데이터를 저장)와 **연산기**(ALU, 곱셈-누산기)가 물리적으로 분리되어 있습니다. 딥러닝 추론은 대부분 "가중치를 메모리에서 읽어와 곱하고 더하는" 연산(MAC, Multiply-Accumulate)의 반복인데, 이 구조에서는 매 연산마다 데이터를 메모리↔연산기 사이로 실어 날라야 합니다. 이 데이터 이동이 실제 곱셈·덧셈보다 훨씬 많은 시간과 에너지를 잡아먹는 현상을 **폰 노이만 병목(von Neumann bottleneck)** 또는 **메모리 벽(memory wall)** 이라 부릅니다.

**CIM(Compute-In-Memory, 메모리 내 연산)** 은 이 병목을 근본적으로 없애자는 아이디어입니다. 데이터를 메모리에서 연산기로 옮기는 대신, **메모리 셀 자체가 연산을 수행**하도록 만듭니다. 이 프로젝트가 다루는 것은 그중에서도 **아날로그 크로스바(crossbar) 기반 CIM**입니다 — 아래에서 하나씩 풀어 설명합니다.

## 2. 크로스바(Crossbar) 어레이 — CIM의 핵심 부품

크로스바는 가로줄(row)과 세로줄(column)이 격자로 교차하는 배선망이고, 교차점마다 **소자(device/cell)** 하나가 붙어 있습니다. 이 소자는 특정 **전도도(conductance)** 값을 갖도록 프로그래밍할 수 있는데, 신경망의 **가중치(weight)** 를 이 전도도 값으로 저장합니다.

- 각 row에 입력값에 비례하는 전압을 흘리면
- 옴의 법칙(전류 = 전압 × 전도도)에 따라 각 교차점에서 "입력 × 가중치"에 해당하는 전류가 흐르고
- 키르히호프의 전류 법칙(한 지점에 모이는 전류는 합쳐진다)에 따라 같은 column의 모든 전류가 **한 번에 합산**됩니다.

즉 크로스바 한 번의 "읽기" 동작이 곧 **행렬-벡터 곱셈(matrix-vector multiply) 전체**를 물리 법칙만으로 수행합니다 — 이게 CIM이 "빠르고 에너지 효율적"이라고 불리는 이유입니다. 이 프로젝트 코드에서 `crossbar_rows` / `crossbar_cols`(`schemas/config.py`의 `HWConfig`)가 바로 이 격자의 크기입니다. 크로스바 하나로 표현할 수 있는 행렬 크기는 유한하므로, 신경망 레이어가 크면 여러 개의 크로스바(=**타일, tile**)에 나눠 담아야 합니다(`num_tiles`) — 이건 §5(NoC)에서 다시 다룹니다.

**소자(device) 종류**: 전도도를 어떻게 저장하느냐에 따라 여러 소자 기술이 있습니다. 이 프로젝트는 `CIMDeviceType`으로 네 가지를 구분합니다:

| 소자 | 풀네임 | 한 줄 특징 |
|---|---|---|
| RRAM | Resistive RAM (멤리스터) | 저항 변화로 전도도 저장, 비휘발성 |
| SRAM | Static RAM | 디지털 셀을 아날로그 전류원처럼 활용, 휘발성이지만 내구성/속도 우수 |
| PCM | Phase-Change Memory | 소재의 결정/비결정 상태 전이로 저항 조절 |
| STT-MRAM | Spin-Transfer Torque MRAM | 자기 스핀 방향으로 저항 조절, MRAM의 대표 구현체 |

같은 신경망이라도 어떤 소자 위에 올리느냐에 따라 에너지·정확도·노이즈 특성이 달라지므로, 실제 CIM 칩 논문은 항상 소자 종류를 명시합니다.

## 3. ADC/DAC — 디지털 세계와 아날로그 세계를 잇는 다리

신경망의 입력·가중치는 디지털 숫자(비트)이지만, 크로스바 내부의 전압·전류·전도도는 **아날로그** 물리량입니다. 그래서 크로스바 앞뒤로 변환기가 필요합니다.

- **DAC (Digital-to-Analog Converter)**: 디지털 입력값을 크로스바에 흘릴 아날로그 전압으로 바꿔줍니다.
- **ADC (Analog-to-Digital Converter)**: 크로스바에서 나온 아날로그 합산 전류(=**partial sum**, 아직 다음 레이어로 넘기기 전의 누적값)를 다시 디지털 숫자로 바꿔줍니다.

ADC/DAC에는 **해상도(resolution, bits)**라는 개념이 있습니다 — 몇 비트로 아날로그 값을 표현하느냐입니다(`HWConfig.adc_bits`/`dac_bits`). 해상도가 높을수록 더 정밀하게 값을 표현하지만:

- **에너지**: ADC는 일반적으로 해상도가 1비트 늘 때마다 에너지가 지수적으로 증가합니다(플래시 ADC 기준). DAC는 상대적으로 선형적으로 증가합니다.
- **지연시간**: 변환에 걸리는 시간(사이클 수)도 해상도에 비례해 늘어납니다.
- **정확도**: 반대로 해상도가 낮으면 값을 뭉뚱그려 표현하게 되어(양자화 오차) 신경망 정확도가 떨어집니다.

즉 ADC/DAC 해상도는 "에너지·지연시간 vs 정확도"를 맞바꾸는 핵심 손잡이입니다. 이 프로젝트에서는 `adc_bits`/`dac_bits`가 칩 하나가 공유하는 **물리 상수**로 취급됩니다 — 레이어마다 다르게 설정할 수 있는 게 아니라, `HWConfig`(칩 스펙)에서 한 번 정해지면 모든 레이어가 그 한계 안에서 동작해야 합니다(§7의 `weight_bits ≤ min(adc_bits, dac_bits)` 제약 참고).

## 4. 소자 비이상성(Non-idealities) — 이상적이지 않은 현실

앞의 §2~3은 "이상적인" 크로스바를 가정했습니다. 실제 칩에는 두 가지 물리적 한계가 항상 존재합니다.

- **IR-drop (배선 저항에 의한 전압 강하)**: 배선도 저항을 가지므로(`wire_resistance_ohm_per_um`), 전압이 배선을 타고 흐르는 동안 일부가 손실됩니다(전압 = 전류 × 저항, "IR"). 크로스바가 크고 배선이 길수록 row/column 위치에 따라 실제로 도달하는 전압이 달라져 계산이 왜곡됩니다.
- **소자 노이즈(device noise)**: 소자의 전도도 값 자체가 프로그래밍 오차나 시간에 따른 표류(drift) 때문에 완벽히 일정하지 않고 확률적으로 흔들립니다(`device_noise_sigma`로 표준편차를 모델링).

두 현상 모두 "소프트웨어 지표(정확도)"만으로는 안 보이는 **물리적 실패**를 만들 수 있습니다 — 예를 들어 정확도는 높게 나왔더라도 실제 칩에서는 IR-drop 때문에 그 정확도가 재현이 안 될 수 있습니다. 그래서 이 프로젝트의 `@verifier`는 정확도와 별개로 `ir_drop_error_pct`/`noise_margin_db`가 허용 범위 안에 있는지를 항상 확인합니다.

## 5. NoC(Network-on-Chip)와 멀티타일 매핑

레이어 하나가 크로스바 하나(`crossbar_rows × crossbar_cols`)보다 크면 여러 타일에 나눠 담아야 합니다. 여러 타일이 늘어서면 타일 사이에 데이터(중간 결과)를 주고받을 배선망이 필요한데, 이를 **NoC(Network-on-Chip)**라 부릅니다 — CPU 여러 개를 잇는 네트워크의 온칩(on-chip) 버전이라고 생각하면 됩니다.

- **토폴로지(topology)**: 타일들을 어떤 모양으로 연결하느냐(`NoCTopology`): `mesh`(격자), `torus`(격자+양끝 연결), `ring`(원형), `crossbar_bus`(모두가 공유 버스 하나에 연결).
- **홉 카운트(hop count)**: 타일 A에서 B로 데이터가 가려면 몇 번의 "중간 정거장"을 거쳐야 하는지 — 토폴로지에 따라 계산 방식이 다릅니다(예: mesh는 맨해튼 거리, ring은 원 위의 최단 호).
- **대역폭(bandwidth, `noc_link_bandwidth_gbps`)**: 링크 하나가 단위 시간에 옮길 수 있는 데이터량. 홉 수가 많고 대역폭이 낮을수록 레이어 간 데이터 전달 지연시간(latency)이 커집니다.

이 프로젝트는 레이어를 **파이프라인(layer-pipelined)** 방식으로 타일에 순서대로 배치합니다 — 신경망의 레이어 순서 자체가 타일들이 이어지는 순서가 됩니다.

## 6. 모델 경량화 — 가중치를 CIM에 맞게 압축하기

CIM 크로스바는 물리적으로 표현할 수 있는 정밀도에 한계가 있으므로(ADC/DAC 비트 수), 신경망 가중치를 원래보다 훨씬 적은 비트로 표현하도록 압축해야 합니다. 여기 쓰이는 핵심 개념들입니다.

### 6.1 양자화 (Quantization)
원래 32비트 부동소수점으로 표현되던 가중치를, 예컨대 4비트 정수처럼 훨씬 적은 비트로 반올림해서 표현하는 것. 비트 수가 적을수록 크로스바에 더 잘 맞고 에너지·메모리가 줄지만, 반올림 오차 때문에 정확도가 떨어집니다.

- **PTQ (Post-Training Quantization)**: 이미 학습이 끝난 모델을 사후에 양자화만 적용 — 빠르지만 정확도 손실이 큰 편.
- **QAT (Quantization-Aware Training)**: 학습(또는 미세조정) 과정 중에 "양자화되면 이런 오차가 생긴다"는 걸 시뮬레이션하면서 가중치를 다시 학습 — 양자화 오차를 감안한 가중치로 수렴하기 때문에 PTQ보다 정확도 손실이 적습니다. 이 프로젝트의 `@tuner`(`tools/qat.py`)는 실제 PyTorch로 QAT를 수행합니다(직선근사 추정, straight-through estimator 기법 사용).

### 6.2 컬럼별(column-wise) 양자화
레이어 전체가 스케일(비트당 표현 범위) 하나를 공유하는 게 아니라, 크로스바의 **컬럼(=출력 채널) 하나마다 독립적인 스케일**을 씁니다. 실제 하드웨어에서 컬럼마다 물리적으로 분리된 읽기(read-out) 경로를 갖고, 컬럼마다 가중치 분포가 다를 수 있기 때문입니다. 스케일을 잘게 나눌수록 같은 비트 수로도 양자화 오차가 줄어듭니다.

### 6.3 Partial-sum 양자화
ADC가 실제로 읽는 값은 "가중치"가 아니라 크로스바 컬럼의 **아날로그 누적값(partial sum)** 그 자체입니다. 그래서 가중치 정밀도와 별개로, 레이어의 출력값 자체도 `adc_bits` 해상도로 컬럼별 독립 양자화합니다 — ADC 해상도가 만드는 오차를 가중치 양자화 오차와 분리해서 시뮬레이션에 반영하는 것입니다.

### 6.4 프루닝 (Pruning)
가중치 중 일부를 아예 0으로 만들어 "없앤" 것처럼 취급하는 기법. 이 프로젝트는 크로스바의 **컬럼(출력 채널) 단위로 통째로 잘라내는 구조적 프루닝(structured column pruning)**을 씁니다 — 임의의 개별 가중치를 듬성듬성 없애는 비구조적(unstructured) 프루닝과 달리, 컬럼 전체를 없애면 실제로 크로스바 배선 자체를 줄일 수 있어 하드웨어 이득으로 바로 이어집니다.

## 7. 다중목적 설계 공간 탐색 (Multi-objective Design Space Exploration)

"레이어마다 몇 비트로 양자화할지, 얼마나 프루닝할지"를 정하는 조합은 레이어 수가 늘어날수록 기하급수적으로 많아집니다(예: 20개 레이어 × 2개 파라미터 = 40차원 탐색 공간). 이걸 사람이 손으로 하나씩 시도하는 대신 알고리즘으로 탐색하는 것이 이 프로젝트의 핵심 축입니다.

- **목적함수(objective)**: 이 프로젝트는 **정확도(accuracy)**, **에너지(energy_pj)**, **지연시간(noc_latency_ms)** 세 가지를 동시에 최적화합니다. 하나만 보면(예: 정확도만) 쉬운데, 셋을 동시에 만족시키려 하면 서로 트레이드오프 관계라 "하나의 정답"이 없습니다.
- **파레토 최적(Pareto optimal) / 파레토 프론트(Pareto front)**: 어떤 후보 A가 다른 후보 B보다 모든 목적에서 더 나쁘지 않고 적어도 하나는 더 낫다면, B는 "지배(dominate)"당한 것이고 탐색 후보에서 제외됩니다. 어느 후보에게도 지배당하지 않는 후보들의 집합이 파레토 프론트입니다 — "정확도를 더 올리려면 에너지를 더 써야 한다"는 트레이드오프 곡선 그 자체라고 보면 됩니다. `@evaluator`가 후보들을 이 기준으로 순위를 매깁니다.
- **NSGA-II**: 다중목적 최적화에 널리 쓰이는 유전 알고리즘(genetic algorithm)입니다. "세대(generation)"마다 여러 후보(population)를 만들고, 파레토 지배 관계와 다양성(같은 지점에 후보가 몰리지 않게)을 기준으로 우수한 후보를 골라 다음 세대를 만드는 과정을 반복합니다. 이 프로젝트는 `pymoo` 라이브러리의 실제 NSGA-II 구현을 씁니다(`tools/search.py`).
- **대리모델(Surrogate model)**: 후보 하나를 실제로 평가하려면 QAT 학습 + 물리 시뮬레이션을 다 돌려야 해서 비용이 큽니다. NSGA-II가 "세대를 거듭하며 탐색"하려면 후보를 수백~수천 개 평가해야 하는데, 그만큼 전부 실제로 학습시킬 수는 없습니다. 그래서 지금까지 실제로 평가해본 소수의 후보들을 바탕으로 "이 근방이면 대략 이런 결과가 나올 것"이라고 **예측만** 하는 값싼 모델(대리모델)을 세우고, NSGA-II는 이 예측값 위에서 돌아갑니다. 이 프로젝트는 **IDW(Inverse Distance Weighting, 역거리 가중)** 방식을 씁니다 — 가까이 있는 실측 후보일수록 더 큰 가중치를 줘서 예측하는 단순한 보간법입니다.
- **LHS 워밍업(Latin Hypercube Sampling)**: 대리모델을 세우려면 최소한의 "실측 데이터"가 있어야 합니다. 탐색을 시작하기 전, 탐색 공간 전체에 고르게 퍼진 소수의 초기 후보를 뽑아 실제로 평가해두는 단계가 LHS 워밍업입니다 — 무작위로 뽑는 것보다 공간을 더 고르게 커버하도록 설계된 샘플링 기법입니다. 이 프로젝트는 세션당 최대 20개, 서로 독립적이므로(§9의 HITL 없이) 병렬로 한 번에 평가합니다(`tools/batch_warmup.py`).

> **왜 이 대리모델의 신뢰도에 한계가 있다고 README가 강조하는가**: 대리모델이 학습하는 실측 후보 수(최대 20개)에 비해 탐색 공간 자체의 차원(레이어 수 × 2)이 크면, 세대를 거듭해도 "정교해지는 탐색"이라기보다는 "몇 안 되는 실측 지점 주변을 폭넓게 보간하는" 수준에 머물 수 있습니다. NSGA-II 알고리즘 자체는 진짜(mock이 아님)이지만, 그 알고리즘이 최적화하는 대상(대리모델의 예측값)이 소량의 데이터로 학습된 근사치라는 뜻입니다.

## 8. 시뮬레이션 신뢰도 — 근사 모델과 정밀 시뮬레이터

CIM 칩의 실제 동작을 검증하는 방법은 두 극단이 있습니다.

- **정밀 시뮬레이터(precise simulator)**: NeuroSim, CIM-Loop 같은 도구는 실제 회로 동작(전압·전류·타이밍)을 세밀하게 시뮬레이션합니다. 정확하지만 느립니다 — 탐색 중 만들어지는 수백 개 후보 전부를 이걸로 검증하기엔 너무 비쌉니다.
- **고속 근사(fast-approximation)**: 물리 법칙에 기반한 스케일링 공식(예: "ADC 에너지는 해상도에 지수적으로 비례한다")으로 값을 빠르게 추정합니다. 빠르지만, 공식에 들어가는 상수(`CIM_TECH_PARAMS`)가 특정 실리콘 공정에서 실측된 값이 아니라 "이 정도 자릿수(order of magnitude)"의 대표값이라 절대 수치의 정확도는 낮습니다.

이 둘을 잇는 두 가지 장치가 있습니다.

- **보정 계수(Calibration factor)**: 근사 모델이 내놓은 에너지 값에 곱해서 실제 문헌(논문에 보고된 실측 칩 수치)에 가깝게 맞추는 배율입니다. 이 프로젝트는 크로스바 크기·ADC 해상도·소자 종류가 일치하는 논문 값을 찾아(`tools/calibration.py`), 정확히 일치하면 "정확한 보정", 비슷한 것만 있으면 "근사 보정"이라고 항상 정직하게 표시합니다(대시보드 배지). 일치하는 참조가 아예 없으면 배율 1.0(미보정) 상태로 남습니다 — 없는 근거를 지어내지 않는다는 원칙입니다.
- **다중 충실도(Multi-fidelity) 검증**: 탐색 중 만들어지는 모든 후보가 아니라, 고속 근사가 "일단 수렴했다"고 판단한 극히 일부 후보만 골라 더 정밀한 모델(`@precision_verifier`)로 다시 검증합니다. 이 재검증 결과로 위의 보정 계수를 자동으로 갱신해, 이후 후보들의 고속 근사 정확도까지 함께 개선되는 폐루프(closed-loop)를 만듭니다.

## 9. 이 프로젝트의 실행 구조 — 에이전트 파이프라인과 HITL

지금까지의 개념들이 실제로 어떤 순서로 실행되는지는 [`README.md`](../README.md)의 다이어그램과 표에 정리되어 있습니다. 여기서는 새로 등장하는 소프트웨어 개념만 짚습니다.

- **LangGraph**: 이 프로젝트가 쓰는 오케스트레이션 프레임워크로, "노드(에이전트)들을 그래프로 잇고, 조건에 따라 다음 노드로 라우팅한다"는 실행 모델을 제공합니다. `planner → tuner → (mapper, profiler 병렬) → verifier → evaluator`로 이어지고, `evaluator`가 상황에 따라 다시 `planner`로 돌아가거나(재탐색), 사람에게 묻거나(HITL), 다음 단계로 넘어갑니다.
- **State(상태)**: 그래프를 도는 동안 계속 누적/갱신되는 하나의 데이터 구조(`AutoCIMState`)로, 지금까지 시도한 후보 이력·계산된 물리 수치·보정 계수 등이 전부 여기 담깁니다. 노드는 이 state를 읽고 자기 몫만 갱신합니다.
- **HITL (Human-in-the-Loop)**: 탐색이 정체되거나(재시도 횟수 초과) LLM이 이상 징후를 감지하면, 파이프라인이 멈추고 **사람에게 판단을 요청**합니다. 이 프로젝트에서 HITL은 "예외 상황이 생겼을 때만 타는 안전장치"가 아니라, "이 탐색·근사 시스템은 완전히 신뢰할 수 없다"는 전제 위에서 설계된 **핵심 경로**입니다 — 그래서 파이프라인의 최종 상태(`__end__`)도 "정답이 확정됐다"가 아니라 "사람이 검토할 후보가 준비됐다"는 뜻입니다.
- **LLM의 역할 범위**: 이 프로젝트에서 LLM(`@planner`가 호출)은 수치(비트 수, 프루닝 비율)를 직접 정하지 않습니다. 수치는 전부 §7의 탐색 알고리즘(NSGA-II/대리모델)이 결정하고, LLM은 그 결정에 대한 **근거 설명(rationale)**과 **이상 징후 플래깅(anomaly)**만 담당합니다 — LLM이 회로/물리 수치를 직접 추론하면 환각(hallucination)이 생기기 쉽다는 문제를 피하기 위한 역할 분리입니다.

## 10. 파이프라인 단계별 상세 — 각 에이전트/도구는 무엇을 하고, 왜 그렇게 만들었는가

§9는 전체 흐름을 요약했습니다. 여기서는 그래프에 등록된 8개 노드(`graph.py`)와 그 뒤에서 실제 연산을 수행하는 도구(`tools/*.py`)를 하나씩 짚어, **무엇을 하는지**와 **왜 그 기능이 필요했는지**(코드 주석·커밋 이력에 남아 있는 실제 이유)를 설명합니다. "왜"를 아는 것이 중요한 이유는, 이 프로젝트의 많은 설계가 "처음부터 이렇게 정했다"가 아니라 **실제로 돌려보다가 발견한 문제를 고치며 지금 모습이 됐기** 때문입니다.

### 10.1 `@planner` — 무엇을 얼마나 압축할지 제안 (`nodes/planner.py`, `tools/search.py`)

**역할**: 다음에 시도해볼 후보, 즉 모델의 각 스테이지(레이어 묶음)마다 `(weight_bits, column_pruning_ratio)` 값을 정합니다. 세션 초반에는 LHS 워밍업 후보를(§7), 어느 정도 실측 이력이 쌓이면 IDW 대리모델 + NSGA-II 탐색이 다음 후보를 제안합니다. 이 수치가 정해지면 LLM을 호출해 "왜 이 후보가 타당한지"를 자연어로 설명하게 하고, 필요하면 이상 징후를 플래그합니다.

**왜 LLM이 수치를 직접 정하지 않는가**: 애초 기획(`docs/research_plan.md`)은 LLM이 `tool_calls`로 파라미터까지 판단하는 그림이었습니다. 하지만 실제 세션에서 로컬 tool-calling 모델이 "정확도가 낮으니 `weight_bits`를 줄이자"처럼 **실제로는 정확도를 더 떨어뜨리는 방향**으로 거꾸로 추론하는 사례가 실측으로 발견됐습니다(`nodes/planner.py` 모듈 docstring). 그래서 수치는 전부 통계적 탐색 알고리즘이 정하도록 역할을 좁히고, LLM은 이미 정해진 후보에 대한 **근거 설명(rationale)**과 **이상 징후 플래깅(anomaly_note)**만 강제 tool-calling으로 받습니다 — 자유 텍스트에서 숫자를 파싱하지 않는 것도 같은 이유(환각 방지)입니다.

**왜 스테이지별로 독립적으로 탐색하는가**: 예전에는 모델 전체에 하나의 `(weight_bits, pruning_ratio)` 점을 정하고 "가장자리 레이어는 조금 다르게" 같은 고정 규칙으로 흩뿌렸습니다. 지금은 실제 스테이지 수만큼(`tools.qat.get_layer_groups`) 독립적인 차원을 탐색합니다(스테이지 수 × 2 = resnet18 10스테이지 → 20차원, mobilenet_v2 20스테이지 → 40차원) — 레이어마다 민감도가 다르다는 걸 고정 규칙이 아니라 탐색 자체가 찾아내게 하기 위해서입니다.

**왜 HITL 오버라이드를 여기서 소비/정리하는가**: 사람이 HITL에서 승인한 새 탐색 범위(`human_overrides`)를 읽어서 실제 탐색 경계(`search_bounds()`)에 반영하고, 같은 스텝에서 `needs_hitl`/`retry_count`를 리셋합니다. 이걸 안 하면 같은 HITL 인터럽트가 영원히 반복되거나(리셋 안 함), 반대로 `evaluator`가 카운트한 재시도 횟수가 매번 planner 호출마다 지워져 `needs_hitl`이 절대 True가 될 수 없는(무한 루프, 실제로 `GraphRecursionError`로 재현됨) 버그가 생깁니다. 승인된 범위 자체(`search_bound_overrides`)는 이 한 번의 후보에만 적용되는 게 아니라 세션이 끝날 때까지 계속 유지됩니다.

### 10.2 `@tuner` — 실제 QAT로 압축을 적용 (`nodes/tuner.py`, `tools/qat.py`)

**역할**: `@planner`가 정한 스테이지별 설정을 실제 PyTorch 모델에 적용합니다 — 컬럼별(column-wise) fake-quantization + partial-sum 양자화(§6.2~6.3) + 구조적 컬럼 프루닝(§6.4)을 걸고, 실제 그래디언트 하강으로 미세조정(fine-tune)한 뒤 held-out 세트로 정확도를 측정합니다.

**왜 컬럼별/partial-sum 양자화까지 따로 구현했는가**: 레이어 전체가 스케일 하나를 공유하면, 크로스바 컬럼마다 실제로는 물리적으로 분리된 읽기 경로를 갖는다는 사실을 반영하지 못합니다. 실측 결과(README) 같은 resnet18 가중치에서 4비트 기준 레이어별 59~94%의 양자화 오차(MSE) 감소를 확인했습니다. 여기서 더 나아가, ADC가 실제로 읽는 값이 가중치가 아니라 컬럼의 아날로그 누적값(partial sum)이라는 점에 착안해 레이어 출력 자체도 `adc_bits` 해상도로 별도 양자화합니다 — 가중치 정밀도와 무관하게 ADC 해상도 자체가 만드는 오차를 분리해서 시뮬레이션에 반영하기 위해서입니다.

**왜 디바이스(CUDA/MPS/CPU)를 자동 감지하는가**: 대부분의 CI/개발 환경에 GPU가 없다는 전제(docs/coding_guidelines.md 5.C) 아래, `_resolve_device`가 CUDA → Apple MPS → CPU 순으로 자동 판단하고 `AUTOCIM_QAT_DEVICE`로 강제 오버라이드할 수 있게 해뒀습니다. 캐시된 베이스 모델/로더는 항상 CPU에 두고, 후보별 딥카피만 디바이스로 옮겨 공유 캐시가 특정 디바이스에 묶이지 않게 합니다.

**왜 CIFAR10 서브셋(기본 512/128/1000장)으로 동작하는가**: 후보 하나를 QAT로 평가하는 데 걸리는 시간이 탐색 전체의 병목이기 때문에, 빠른 반복 실험을 우선하는 **의도적인 스코프 결정**입니다(숨겨진 지름길이 아니라 README/roadmap에 명시). `AUTOCIM_QAT_TRAIN_SIZE`/`_VAL_SIZE`/`_TEST_SIZE`/`AUTOCIM_QAT_MAX_EPOCHS`로 필요하면 늘릴 수 있습니다. train/val/test는 진짜 3분할이라, 조기 종료(early stopping)가 test set을 절대 들여다보지 않습니다 — 그래야 보고되는 정확도가 그 정확도를 만든 결정에 새어 들어가지 않습니다(정보 누수 방지).

**왜 아키텍처/데이터셋이 하드코딩된 3~1개가 아니라 레지스트리인가**: `_MODEL_REGISTRY`/`_DATASET_REGISTRY`는 내장 항목(resnet18/mobilenet_v2/vit_tiny, cifar10) 외에도 `register_model`/`register_dataset` 런타임 등록이나 `"module.path:function_name"` 문자열 스펙으로 임의의 `nn.Module`/데이터셋을 연결할 수 있게 합니다 — docs/coding_guidelines.md 5.C의 "Model Agnostic" 요구사항(`@tuner`/`@mapper` 로직이 특정 아키텍처에 묶이면 안 됨)을 코드 변경 없이 만족시키기 위해서입니다.

### 10.3 `@mapper` — 타일 배치와 NoC 지연시간 (`nodes/mapper.py`, `tools/cim_physics.py`)

**역할**: `@tuner`가 실제로 사용한 레이어 설정을 받아, 레이어마다 크로스바 타일 하나씩 순서대로 배치하고(레이어 파이프라인 방식), 타일 간 데이터 전송에 걸리는 시간을 계산합니다.

**왜 홉 카운트를 실제 그래프 거리 공식으로 계산하는가**: "NoC 지연시간은 대략 이 정도"라는 임의 상수를 쓰는 대신, mesh는 맨해튼 거리, ring은 원 위의 최단 호, torus는 wraparound를 감안한 거리처럼 **토폴로지별 실제 그래프 이론 공식**을 씁니다(`hop_distance`) — `HWConfig.noc_topology`가 바뀌면 지연시간 계산도 그 토폴로지의 실제 성질을 따라 바뀌게 하기 위해서입니다. 대역폭(`noc_link_bandwidth_gbps`)도 1 Gb/s = 1 bit/ns라는 단위 그대로 변환해서 쓰기 때문에, NoC 플릿(flit) 헤더 크기 하나를 제외하면 임의의 보정 상수가 필요 없습니다.

**왜 `@profiler`와 병렬로 실행되는가**: `@tuner`가 끝나면 `@mapper`(지연시간 위주)와 `@profiler`(에너지 위주)는 서로의 결과가 필요 없는 독립적인 계산이라, LangGraph의 fan-out으로 동시에 실행해 탐색 전체 시간을 줄입니다. 두 노드가 동시에 `state["metrics_store"]`에 쓰기 때문에, 각자 자기 이름(`"mapper"`/`"profiler"`)의 네임스페이스에만 쓰도록 강제하는 `namespaced_metrics_reducer`(state.py)가 없으면 경쟁 조건(race condition)으로 서로의 결과를 덮어쓸 수 있습니다(docs/coding_guidelines.md 5.A).

**왜 `state["planned_layer_configs"]`가 아니라 `@tuner`의 출력에서 설정을 읽는가**: `@planner`가 아직 아무 계획도 못 냈거나 LLM 호출이 실패해 `@tuner`가 자체 기본값(`default_layer_configs`)으로 폴백한 경우에도, `@mapper`/`@profiler`가 `@tuner`가 **실제로 사용한** 설정을 그대로 시뮬레이션하도록 보장하기 위해서입니다.

### 10.4 `@profiler` — 에너지·지연시간 물리 시뮬레이션 (`nodes/profiler.py`, `tools/cim_physics.py`)

**역할**: ADC/DAC/크로스바 MAC 에너지와 컴퓨트 지연시간을 물리 법칙 기반 스케일링 공식으로 추정합니다 — ADC 에너지는 해상도에 지수적으로(플래시 ADC 특성), DAC 에너지는 선형적으로, 크로스바 에너지는 배열 크기와 가중치 비트폭에 비례하게(§3). 계산 결과에 이 하드웨어(`hw_spec_id`)의 보정 계수(§8)를 곱해 돌려줍니다.

**왜 SRAM 버퍼 에너지(`buffer_energy_pj`)를 컴퓨트 에너지(`compute_energy_pj`)와 분리해서 더하는가**: `tools/calibration.py`가 참조하는 문헌 수치들은 전부 "크로스바 매크로 자체"의 pJ/MAC만 측정한 것이라 SRAM 버퍼를 채우는 비용을 포함하지 않습니다. 만약 버퍼 에너지를 그냥 총 에너지에 합쳐서 보정 계수를 계산하면, 레이어 수와 무관한 고정 비용(버퍼 크기 1회성 충전)이 MAC당 비율 계산을 왜곡합니다 — 특히 크로스바 배열이 작을수록 왜곡이 커집니다. 그래서 보정 계수는 `compute_energy_pj`에만 적용하고, `buffer_energy_pj`는 보정 없이 최종 `energy_pj`에 그대로 더합니다.

**왜 이 노드가 `calibration_factors`를 state로 다시 돌려주는가**: 새로운 `hw_spec_id`를 처음 보면 배율 1.0(미보정)으로 등록해야 하는데, 이 갱신된 dict를 예전에는 그냥 버렸습니다(코드 주석에 남은 실제 버그) — 그래서 이전 반복이 만든 보정 계수가 `state["calibration_factors"]`에 전혀 누적되지 않는 문제가 있었고, 지금은 이 노드가 명시적으로 반환해 `merge_dicts` 리듀서로 누적되게 고쳤습니다.

### 10.5 `@verifier` — 물리적 실현 가능성 판정 (`nodes/verifier.py`, `tools/simulators.py`의 `verifier_tool`)

**역할**: `@mapper`+`@profiler` 결과를 하나로 합쳐(fan-in), IR-drop 오차(§4)와 노이즈 마진이 허용 범위 안에 있는지 확인합니다. `ir_drop_error_pct < 5.0`이고 `noise_margin_db > 0.0`이어야 `is_converged=True`입니다.

**왜 정확도가 아니라 물리적 실현 가능성만 보는가**: 정확도는 `@tuner`가, 목표 달성 여부(§7의 target)는 `@evaluator`가 별도로 확인합니다. `@verifier`가 보는 건 그와 독립적인 질문 — "소프트웨어 지표가 좋아도, 배선 저항이나 소자 노이즈 때문에 이 칩이 물리적으로 실패하지 않는가"입니다. 이 판정을 정확도 판정과 분리해둔 것 자체가 "소프트웨어 지표만으로 완성을 선언하지 않는다"는 이 프로젝트의 원칙입니다.

**왜 임계값이 하드코딩된 상수가 아니라 `HWConfig`에서 오는가**: `wire_resistance_ohm_per_um`/`device_noise_sigma`/`adc_bits`가 전부 `HWConfig`(주입된 하드웨어 스펙)에서 오기 때문에, 특정 칩 하나에 맞춘 허용치가 아니라 검증 대상 하드웨어마다 다른 물리적 한계를 반영합니다(docs/coding_guidelines.md 5.C).

### 10.6 `@evaluator` — 스키마 검증, 이력 누적, 수렴/라우팅 판정 (`nodes/evaluator.py`, `tools/search.py`의 `compute_pareto_rank`)

**역할**: 이번 반복에서 4개 도구(tuner/mapper/profiler/verifier)의 출력이 전부 스키마에 맞고 성공했는지 검증하고, `candidate_history`에 이번 후보를 기록하고(파레토 순위 포함), 세 가지 조건 중 어디로 라우팅할지(`evaluator_router`) 정합니다: **수렴(Converged)**, **사람 개입 필요(HITL)**, **재탐색(Re-plan)**.

**왜 "수렴"이 `@verifier`의 물리 검증과 목표(target) 검증 둘 다를 만족해야 하는가**: 예전에는 IR-drop/노이즈만 괜찮으면 "수렴"으로 끝났는데, 이러면 정확도가 목표(예: 0.7)에 한참 못 미쳐도(예: 0.20) 물리적으로만 괜찮으면 세션이 끝나버리는 문제가 있었습니다. 그래서 `--target-accuracy`/`--target-energy-pj`/`--target-latency-ms`가 **설정된 경우에만** 추가로 확인하는 `check_targets`가 더해졌습니다 — 목표를 안 준 세션은 예전과 동일하게 동작합니다(하위 호환).

**왜 검증 오류가 있으면 이번 후보를 `candidate_history`에 아예 기록하지 않는가**: 신뢰할 수 없는(스키마 오류/실패) 정확도·에너지·지연시간 값을 대리모델 학습 데이터로 섞어 넣으면 이후 탐색 전체가 오염됩니다. `has_validation_errors`가 기록 여부를 가르는 유일한 기준인 이유는, `@verifier`가 항상 `SUCCESS` 상태를 반환하면서 상류 도구의 실패를 그대로 에코하기만 할 수 있어(즉 `@verifier` 자신의 필드가 None이 아닐 수도 있음), "이번 수치가 신뢰할 만한가"를 판단할 유일한 권위 있는 신호가 스키마 검증 결과이기 때문입니다.

**왜 이상 징후(anomaly)가 재시도 횟수(`MAX_RETRY_LIMIT`=3)를 기다리지 않고 즉시 HITL로 가는가**: `@planner`의 LLM이 "이 실패가 반복되는 패턴이다" 같은 걸 감지하면, 일반적인 3회 재시도를 다 채울 때까지 기다리는 게 아니라 그 즉시 연구자에게 알리기 위해서입니다 — 매 반복 새로 설정되는 값이라 지난 반복의 낡은 플래그가 새지 않습니다.

### 10.7 `hitl_human_approval` — 사람 개입 지점 (`nodes/hitl.py`)

**역할**: LangGraph의 동적 `interrupt()`를 호출해 그래프 실행을 멈추고, 실패 이력·이상 징후·현재 탐색 범위·목표값을 사람에게 보여준 뒤, 사람이 승인한 새 탐색 범위(`new_bounds`)를 받아 `@planner`로 돌아갑니다.

**왜 `compile()`의 `interrupt_before`가 아니라 노드 안의 `interrupt()`를 쓰는가**: LangGraph v0.2+ 표준(docs/coding_guidelines.md 5.B)이 권장하는 방식이라는 것 외에, `interrupt_before`는 그래프 컴파일 시점에 고정된 정지 지점이라 "왜 멈췄는지"에 대한 맥락(실패 이력, 이상 징후 등)을 실행 중 상태에서 동적으로 담아 보여줄 수 없습니다. 노드 안에서 `interrupt()`를 호출하면 그 순간의 state 전체를 담아 사람에게 넘길 수 있습니다.

**왜 `search_bound_overrides`(누적) 전체를 payload에 담는가**: 사람이 두 번째 HITL을 만났을 때, `main.py`의 제안 로직(`suggest_override_bounds`)이 하드웨어의 원본 한계치가 아니라 **이미 승인된 이전 범위**를 기준으로 다음 제안을 좁혀야 하기 때문입니다 — 그렇지 않으면 이전 승인보다 느슨한 제안이 나와 조용히 이전 결정을 무효화할 수 있습니다.

### 10.8 `@precision_verifier` — 2단계 정밀 재검증 (`nodes/precision_verifier.py`, `tools/precision_check.py`)

**역할**: `@evaluator`가 "수렴했다"고 판단한 후보만 골라(전체 탐색 후보가 아님) 더 정밀한 모델로 에너지를 다시 계산합니다. 이 결과가 목표를 여전히 만족하면 최종 `__end__`로, 아니면 `@evaluator`와 똑같은 재시도/HITL 흐름으로 되돌립니다. 재검증 결과로 `calibration_factors`/`calibration_provenance`도 갱신합니다.

**왜 모든 후보가 아니라 "이미 수렴한" 후보만 재검증하는가**: 정밀 시뮬레이터(NeuroSim 실제 C++ 코어 등) 호출은 `tools/cim_physics.py`의 해석적 공식보다 훨씬 비쌉니다. 탐색이 시도하는 수십~수백 개 후보 전부를 이걸로 검증하면 비용이 감당 안 되므로, 세션이 "이제 끝내려는" 극히 일부 후보에만 이 비용을 씁니다.

**왜 ADC 에너지 항은 항상 실제 모델(`hwcomponents_adc`)인데 DAC/크로스바-MAC/SRAM 버퍼는 Docker에서만 실제 모델(`hwcomponents_neurosim`/`hwcomponents_cacti`)인가**: `hwcomponents_adc`는 순수 Python이라 어디서나 동작하지만, `hwcomponents-neurosim`은 설치 시점에 NeuroSim의 실제 C++ 소스를 `g++`로 컴파일해야 해서 C++ 툴체인이 없는 로컬(이 프로젝트의 Windows 개발 환경 포함)에서는 설치 자체가 안 됩니다. `hwcomponents-cacti`는 컴파일은 필요 없지만 POSIX 전용 `fcntl`을 import해서 Windows에서 로드가 안 됩니다. 그래서 이 둘은 `requirements-docker.txt` + `Dockerfile`의 `build-essential`을 통해 Docker 이미지에서만 설치·활성화되고, 없는 환경에서는 조용히 실패하는 대신 `tools/cim_physics.py`의 근사 자리 표시자로 폴백합니다.

**왜 `HWConfig.device_type`을 명시해야 "진짜 그 소자"의 수치가 되는가**: Docker에서 NeuroSim이 활성화돼도 `device_type`을 안 정하면 NeuroSim의 **제네릭 placeholder 셀** 위에서 계산됩니다 — 이건 "실제 NeuroSim 모델"이긴 하지만 "실제 RRAM/SRAM/PCM/STT-MRAM 칩"을 인용하는 수치는 아닙니다. `device_type`을 정해야 NeuroSim이 번들한 실제 NVMExplorer-fit `.cell` 설정을 골라 씁니다 — 개발 중 이 캐시가 `cell_config`를 캐시 키에서 빠뜨려 소자 타입을 바꿔도 이전 결과를 재사용하는 버그가 실제로 발견되어 수정된 이력이 있습니다.

**왜 이 노드가 자기 자신의 재검증 결과로 보정 계수를 다시 학습하는가**: 정밀 재검증 한 번이 "이 후보가 진짜 목표를 만족하는가"와 "그 과정에서 드러난 근사-정밀 오차 비율"을 동시에 알려주기 때문에, 그 오차 비율을 그대로 다음 후보들의 고속 근사를 고치는 데 재사용합니다(§8의 폐루프).

### 10.9 공통 인프라 — `wrap_tool_call`과 컨텍스트 주입 (`middleware.py`)

모든 백엔드 도구(`tuner_tool`/`mapper_tool`/`profiler_tool`/`verifier_tool`/`precision_check_tool`)는 직접 호출되지 않고 `wrap_tool_call` 데코레이터를 통해서만 실행됩니다. 이 함수가 하는 일과 이유:

- **`ExecutionContext`/`HWConfig`를 강제로 주입하고 LLM이 준 값은 버림**: `model_id`/`hw_spec_id` 같은 식별자를 LLM이 자유 텍스트로 잘못 말할 위험(연구계획서가 "프롬프트 오염 및 환각 방지"라 부르는 것)을 원천 차단하기 위해, 세션이 시작될 때 설정해둔 실제 값으로 항상 덮어씁니다.
- **Pydantic 스키마로 인자 검증**: 잘못된 인자가 도구 내부까지 들어가 알아보기 힘든 예외를 던지는 대신, 진입 지점에서 바로 구조화된 실패(`ToolResult(status=FAILED, ...)`)로 변환합니다.
- **모든 예외(Python이든 C-확장이든)를 가둠**: 도구 하나가 예외를 던져도 LangGraph 전체 실행이 죽지 않고, `{node_name: ToolResult}` 형태로 실패 사실만 state에 남게 합니다 — 에이전트 루프가 한 도구의 실패로 전체가 죽는 걸 막는 장치입니다.

### 10.10 탐색을 뒷받침하는 두 도구 — `tools/search.py`와 `tools/batch_warmup.py`

**`tools/search.py`가 왜 무거운 의존성(`pymoo`)까지 들여왔는가**: 이 프로젝트는 보통 최소 의존성을 지향하지만(docs/coding_guidelines.md 2), 다중목적 탐색만큼은 직접 만든 단일목적 UCB 스캔을 세 목적으로 억지로 늘리는 대신, 검증된 실제 다중목적 알고리즘(NSGA-II)을 쓰는 쪽을 의도적으로 택했습니다 — 실제로 이전 버전(정확도만 대리모델링하고 에너지/지연은 사후 랭킹에만 쓰던 버전)이 사실상 단일목적으로 동작한다는 걸 실측 로그로 확인한 뒤의 교체였습니다.

**왜 랜덤/단일목적 전략(`search_strategy`)이 코드에 남아 있는가**: `tools/run_ablation.py`가 "탐색이 낫다"를 그냥 주장하는 대신 실제로 비교할 수 있도록 하기 위한 대조군입니다 — `random`(대리모델 없음), `single_objective`(이전에 실제로 배포됐던 정확도 전용 UCB), `multi_objective`(현재 기본값) 세 전략이 똑같은 워밍업에서 시작해 acquisition 단계만 다르게 갈라지도록 만들어, 비교가 공정하게(같은 조건에서만 다르게) 이뤄지게 했습니다.

**`tools/batch_warmup.py`가 왜 필요한가**: LHS 워밍업 후보들은 서로의 결과에 의존하지 않는 독립적인 점들이라, 순차 그래프 루프로 하나씩 평가하는 대신 스레드 풀로 한꺼번에 평가할 수 있습니다(PyTorch/NumPy 연산이 실제 계산 중 GIL을 놓아주므로 CPU-bound 작업도 스레드로 진짜 병렬화됩니다). 다만 두 가지는 의도적으로 순차 경로와 다릅니다: (1) `contextvars`가 스레드 풀 워커로 전파되지 않아 `wrap_tool_call`을 우회하고 `*ToolInput`을 직접 만들어 호출하며, (2) 워밍업 후보 수만큼 LLM을 동시에 호출하면 비용/레이트리밋이 감당 안 되므로 이 단계는 LLM 확인 없이 원본 LHS 점을 그대로 씁니다 — 순차 경로의 "LLM 호출 실패 시 탐색 점 그대로 사용" 폴백과 동일한 값을 그냥 처음부터 쓰는 셈입니다. 단일 CUDA GPU에서는 여러 스레드가 동시에 cudnn을 호출하면 크래시가 날 수 있어(`--parallel-warmup-workers` 미지정 시 CUDA에서 자동으로 1개 워커), 이 병렬화는 CPU/MPS에서 더 안전합니다.

## 11. 개념 → 이 프로젝트의 코드/CLI 매핑

| 개념 | 관련 CLI 옵션 / 코드 | 상세 |
|---|---|---|
| 크로스바 크기 | `HWConfig.crossbar_rows`/`crossbar_cols`/`num_tiles` (`--hw-config`) | §2 |
| ADC/DAC 해상도 | `HWConfig.adc_bits`/`dac_bits` | §3 |
| IR-drop / 소자 노이즈 | `HWConfig.wire_resistance_ohm_per_um`/`device_noise_sigma`, `@verifier` | §4, §10.5 |
| NoC 토폴로지/대역폭 | `HWConfig.noc_topology`/`noc_link_bandwidth_gbps`, `@mapper` | §5, §10.3 |
| 소자 종류 | `HWConfig.device_type` (RRAM/SRAM/PCM/STTRAM) | §2, §10.8 |
| 양자화/프루닝 실행 | `@tuner` (`tools/qat.py`) | §6, §10.2 |
| 후보 제안(탐색) | `@planner` (`tools/search.py`), `--target-accuracy`/`--target-energy-pj`/`--target-latency-ms` | §7, §10.1, §10.10 |
| 고속 근사 물리 시뮬레이션 | `@mapper`/`@profiler` (`tools/cim_physics.py`) | §8, §10.3, §10.4 |
| 보정 계수 | `tools/calibration.py`, `--exact-calibration-only` | §8, §10.4, §10.8 |
| 다중 충실도 재검증 | `@precision_verifier` (`tools/precision_check.py`) | §8, §10.8 |
| 수렴 판정 | `@verifier`/`@evaluator` | §10.5, §10.6 |
| 사람 개입 | `hitl_human_approval` 노드 | §9, §10.7 |
| 도구 실행 안전장치 | `wrap_tool_call` (`middleware.py`) | §10.9 |
| 워밍업 병렬화 | `tools/batch_warmup.py` | §10.10 |

## 12. 용어집 (가나다순)

- **CIM (Compute-In-Memory)**: 메모리 셀 자체가 연산을 수행하도록 만든 구조. §1
- **크로스바(Crossbar)**: row/column 격자에 소자를 배치해 행렬-벡터 곱을 물리적으로 수행하는 배열. §2
- **소자(Device/Cell)**: 크로스바 교차점에서 전도도로 가중치를 저장하는 소자. RRAM/SRAM/PCM/STT-MRAM. §2
- **ADC/DAC**: 아날로그↔디지털 변환기. 해상도(bits)가 에너지·지연·정확도를 좌우. §3
- **Partial sum**: ADC가 실제로 읽는, 크로스바 컬럼의 아날로그 누적값. §6.3
- **IR-drop**: 배선 저항으로 인한 전압 손실. §4
- **소자 노이즈(Device noise)**: 전도도 값의 확률적 흔들림. §4
- **NoC (Network-on-Chip)**: 여러 크로스바 타일을 잇는 온칩 통신망. §5
- **양자화(Quantization)**: 가중치를 적은 비트로 반올림 표현. §6.1
- **QAT vs PTQ**: 학습 중 양자화 오차를 반영(QAT)하느냐, 학습 후 사후 적용(PTQ)하느냐. §6.1
- **컬럼별 양자화(Column-wise quantization)**: 크로스바 컬럼마다 독립 스케일 사용. §6.2
- **프루닝(Pruning)**: 가중치(이 프로젝트는 컬럼 단위)를 통째로 제거. §6.4
- **다중목적 최적화(Multi-objective optimization)**: 정확도/에너지/지연시간을 동시에 최적화. §7
- **파레토 최적/프론트(Pareto optimal/front)**: 어느 목적도 더 나빠지지 않고는 개선할 수 없는 후보(들). §7
- **NSGA-II**: 다중목적 최적화용 유전 알고리즘. §7
- **대리모델(Surrogate model)**: 실측 대신 값싸게 결과를 예측하는 근사 모델(이 프로젝트는 IDW). §7
- **LHS (Latin Hypercube Sampling)**: 탐색 공간을 고르게 커버하는 초기 샘플링 기법. §7
- **고속 근사(Fast-approximation)**: 물리 공식 기반의 빠르지만 부정확할 수 있는 시뮬레이션. §8
- **보정 계수(Calibration factor)**: 근사 모델 값을 실제 문헌값에 맞추는 배율. §8
- **다중 충실도 검증(Multi-fidelity verification)**: 수렴 후보만 골라 더 정밀한 모델로 재검증. §8
- **HITL (Human-in-the-Loop)**: 탐색 정체/이상 징후 시 사람의 판단을 요청하는 지점. §9
- **LangGraph / State**: 에이전트 간 실행 흐름을 관리하는 그래프 프레임워크와 그 공유 데이터 구조. §9

## 다음으로 읽을 문서

- [`README.md`](../README.md) — 실제 구현 여부, 스코프 한계, 실측으로 확인한 결과
- [`docs/research_plan.md`](research_plan.md) — 최초 기획 의도(일부 구현과 다를 수 있음, README 우선)
- [`docs/coding_guidelines.md`](../docs/coding_guidelines.md) — 이 코드베이스에서 코드를 작성/수정할 때의 규칙
