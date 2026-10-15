# 코드 작성 규칙

이 저장소에서 코드를 고칠 때 지켜야 하는 규칙입니다. 파이프라인 구조에서 직접 나온 제약이라, 어기면 조용히 깨지는 지점들을 모아 둔 것입니다. 코드 주석과 docstring이 이 문서의 절 기호(A부터 D까지)로 참조합니다.

## A. 상태 관리와 리듀서

- **스키마 준수.** `TessellateState` 정의를 따릅니다. Pydantic이나 TypedDict 스키마에 없는 키를 상태에 임의로 추가하지 않습니다.
- **네임스페이스 메트릭.** `metrics_store`를 갱신할 때는 항상 노드 이름 아래에 중첩해 넣습니다(예: `metrics_store: {"mapper": {"status": "SUCCESS", "data": ...}}`). `@mapper`와 `@profiler`가 병렬로 실행되므로, 같은 평면에 쓰면 서로의 결과를 덮어쓰는 경쟁 조건이 생깁니다.
- **상태 정리.** `@planner`는 `human_overrides`를 소비한 직후 비우고, 임시 플래그(`needs_hitl`, `retry_count`)를 초기화합니다.

## B. LangGraph 실행과 HITL

- **v0.2 이후 인터럽트 표준.** `compile()`에 `interrupt_before`를 쓰지 않습니다. 노드 안에서 동적 `interrupt()`를 호출하거나 `Command(resume=...)`를 넘깁니다. `interrupt_before`는 컴파일 시점에 정지 지점이 고정되어, 왜 멈췄는지에 대한 실행 중 맥락을 사람에게 전달할 수 없습니다.
- **재개 중복 금지.** `graph.update_state()`와 동적 `interrupt()`를 섞어서 같은 상태 갱신이 두 번 적용되게 만들지 않습니다.

## C. 하드웨어 설정 주입과 모델 추상화

- **스펙 하드코딩 금지.** 하드웨어 파라미터(`128x128` 크로스바, ADC/DAC 비트 등)나 대상 신경망 구조를 노드 로직에 직접 적지 않습니다. `HWConfig`나 `ExecutionContext` 메타데이터에서 읽어 옵니다.
- **모델 비의존.** `@tuner`와 `@mapper`의 레이어 순회 로직은 임의의 PyTorch `nn.Module`이나 ONNX 그래프를 받을 수 있을 만큼 일반적이어야 합니다.

## D. 툴 인터셉터와 외부 솔버

- **인터셉터 프로토콜.** 백엔드 시뮬레이터와 PyTorch 툴은 `wrap_tool_call`로 감싸서 `ExecutionContext`를 주입하고, 예외가 그래프 루프를 깨지 않게 처리합니다.
- **목업 우선.** 외부 시뮬레이터(NeuroSim, CIM-Loop)는 목업 데이터 인터페이스를 먼저 정의한 뒤에 바이너리나 C 확장 연동을 시도합니다. 연동이 되면 같은 자리에서 실제 모델로 교체합니다.
