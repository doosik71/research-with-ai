---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Continual Learning in Practice

- Tom Diethe, Tom Borchert, Eno Thereska, Borja Balle, Neil Lawrence
- arXiv 2019 / NeurIPS 2018 Continual Learning Workshop
- Continual Learning as a production ML systems problem

---

## 이 논문의 성격

- 이 논문은 새로운 forgetting 방지 알고리즘을 제안하는 논문이 아니다.
- 핵심은 운영 중인 ML 시스템을 **지속적으로 감시하고 재학습하는 참조 아키텍처** 제안이다.
- 저자들은 이를 `Auto-Adaptive ML` 또는 `zero-touch ML`의 중간 단계로 본다.
- 핵심 관점:
  - Continual Learning은 모델 하나의 학습 문제가 아니라 배포 이후 전체 수명주기 관리 문제다.

---

## 문제 정의

- 실제 운영 환경에서는 다음 문제가 반복된다.
  - 데이터 분포가 시간에 따라 변한다.
  - 라벨이 늦게 도착하거나 외부 시스템에서 합류한다.
  - 재학습 자체에도 시간과 인프라 비용이 든다.
  - 모델 교체는 하류 시스템에 리스크를 준다.
  - 정확도만이 아니라 실패 비용, 재학습 비용,
    서비스 중요도를 함께 고려해야 한다.
- 따라서 배포 시점 테스트만으로는 충분하지 않다.

---

## 핵심 주장

저자들의 핵심 메시지는 다음과 같다.

- ML 시스템은 정적 소프트웨어가 아니라
  **계속 진단되고 갱신되는 적응형 시스템**이어야 한다.
- Continual Learning은 단지 망각 방지 문제가 아니라
  **monitoring + retraining + rollback + provenance** 문제다.
- 회귀 테스트에 대응되는 개념으로
  시간 변화에 대한 **progression testing**이 필요하다.

---

## 참조 아키텍처 개요 (1/2)

- 논문이 제안하는 주요 컴포넌트:
  - Streams
  - Sketcher / Compressor
  - Joiner
  - Shared Infrastructure
  - Data Monitoring
  - Prediction Monitoring
  - Trainer / HPO / Predictor
  - Model Policy Engine

---

## 참조 아키텍처 개요 (2/2)

- 포인트:
  - batch-first가 아니라 **stream-first** 관점이다.

---

## Streams / Sketcher / Joiner (1/2)

### Streams

- 데이터는 빠르게, 늦게, 순서가 바뀐 채,
  또는 일부 누락된 상태로 들어올 수 있다.

### Sketcher / Compressor

- 모든 원본 데이터를 그대로 유지하지 않고
  sketch로 요약해 운영 비용을 줄인다.
- 예: Bloom filter, Count-min sketch, HyperLogLog, t-digest, random projection

---

## Streams / Sketcher / Joiner (2/2)

### Joiner

- 입력과 지연 라벨을 결합한다.
- 실서비스의 delayed supervision을 흡수하는 핵심 계층이다.

---

## Shared Infrastructure (1/2)

- 주요 저장 대상:
  - Model DB
  - Training DB
  - Validation Data Reservoir
  - System State DB
  - Diagnostic Logs

---

## Shared Infrastructure (2/2)

- 핵심은 provenance다.
  - 어떤 모델이 왜 선택되었는가
  - 왜 재학습이 수행되었는가
  - 왜 롤백이 발생했는가
- 즉, 저장은 단순 캐시가 아니라 운영 의사결정의 근거를 남기는 역할을 한다.

---

## Monitoring 계층 (1/2)

### Data Monitoring

- drift
- anomaly
- change-point
- shift type / magnitude / uncertainty 추정

---

## Monitoring 계층 (2/2)

### Prediction Monitoring

- 최근 재학습 이후 시간
- 현재 상태 중요도
- 실패 비용
- 재학습 비용
- 상태별 정책 가치

핵심: 입력 이상과 출력 이상을 분리해 관찰해야 한다.

---

## Trainer / HPO / Predictor

- Trainer는 기존 학습 파이프라인을 감싸는 wrapper 역할을 한다.
- 중요한 것은 새 알고리즘보다
  **재학습 절차와 메타데이터 기록의 체계화**다.
- HPO는 warm-start 기반 자동화를 지원한다.
- Predictor는 새 모델의 실제 배포와 연결된다.

기록해야 할 메타데이터 예: validation metric, training time, hyperparameter, trained model provenance

---

## Model Policy Engine (1/2)

이 논문에서 가장 중요한 모듈이다.

정책 엔진이 답해야 할 질문:

- 지금 재학습할 것인가
- 기존 모델을 유지할 것인가
- 이전 모델로 롤백할 것인가
- transfer learning을 적용할 것인가

---

## Model Policy Engine (2/2)

판단 축:

- Horizon
- Cadence
- Provenance
- Costs

저자들은 장기적으로 이를 **강화학습 기반 policy problem**으로 본다.

---

## 이 논문의 강점

- Continual Learning을 벤치마크가 아니라
  운영 문제로 끌어내렸다.
- CL과 MLOps를 하나의 설계 문제로 연결했다.
- monitoring, rollback, provenance, retraining policy를
  한 아키텍처로 묶었다.
- 오늘 기준으로도 model registry, observability,
  retraining workflow 설계에 직접 연결된다.

---

## 한계

- 일반적인 CL 논문처럼 정량 성능 비교 실험이 거의 없다.
- 참조 아키텍처의 효과를 대규모 실험으로 검증하지 않는다.
- 컴포넌트 간 인터페이스나 장애 처리 같은
  구현 세부는 추상 수준에 머문다.
- RL 기반 policy engine은 개념적으로 매력적이지만
  실제 운영에서는 안정성 문제가 크다.

---

## 발표용 핵심 메시지

- 이 논문의 질문은
  "어떻게 안 잊게 학습할까?"보다
  **"운영 중인 ML 시스템을 어떻게 스스로 유지보수하게 만들까?"** 에 가깝다.
- Continual Learning은 알고리즘만으로 완성되지 않는다.
- 실무에서는 monitoring, delayed labels,
  retraining policy, rollback, provenance가 핵심이다.
- 따라서 이 논문은 CL 알고리즘 논문이라기보다 **production continual learning architecture 논문**으로 읽는 것이 정확하다.

---

## 결론

- `Continual Learning in Practice`는
  continual learning을 production ML lifecycle 문제로 확장했다.
- 핵심 기여는 알고리즘이 아니라
  **Auto-Adaptive ML 참조 아키텍처**다.
- 오늘의 관점에서도
  MLOps, observability, adaptive retraining system을 설계할 때
  여전히 유효한 문제 설정을 제공한다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/continual_learning_in_practice_slide.md>
