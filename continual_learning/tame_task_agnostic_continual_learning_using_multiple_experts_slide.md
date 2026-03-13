---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# TAME: Task Agnostic Continual Learning using Multiple Experts

- Haoran Zhu, Maryam Majzoubi, Arihant Jain, Anna Choromanska
- arXiv 2022 / CVPRW 2024
- Task-agnostic continual learning with experts, selector, pruning

---

## 문제 설정

- 이 논문은 **task-agnostic continual learning**을 다룬다.
- 학습 시점과 추론 시점 모두에서 task ID와 task boundary가 주어지지 않는다.
- 모델은 입력 스트림만 보고 다음을 스스로 판단해야 한다.
  - task가 바뀌었는가
  - 기존 expert를 재사용할 것인가
  - 새로운 expert를 만들 것인가
  - 추론 시 어느 expert를 쓸 것인가
- 핵심 질문: task label 없이 continual learning을 수행할 수 있는가?

---

## 핵심 아이디어

- TAME은 문제를 세 개의 모듈로 나눈다.
  - **task experts**
  - **task switch detector**
  - **selector network**
- 구조적 아이디어:
  - 학습 중에는 loss deviation으로 task switch를 감지한다.
  - 기존 expert가 설명 가능하면 그 expert로 전환한다.
  - 아니면 새 expert를 추가한다.
  - 추론 시에는 selector가 입력을 적절한 expert로 라우팅한다.

---

## 왜 이 설정이 어려운가

- 기존 CL 방법 다수는 최소한 학습 시에는 task boundary를 안다.
- 예:
  - EWC, SI, RWALK
  - A-GEM
  - DEN
- 하지만 실제 온라인 환경에서는 분포가 갑자기 또는 점진적으로 변한다.
핵심 문제는 forgetting 보다 **task segmentation**과 **routing**이다.
- 이 논문은 그 문제를 정면으로 다룬다.

---

## 전체 파이프라인

1. 현재 active expert로 학습을 진행한다.
2. smoothed loss를 모니터링한다.
3. loss가 threshold를 넘으면 task switch를 의심한다.
4. 기존 expert들을 다시 평가한다.
5. 적절한 expert가 있으면 그 expert로 전환한다.
6. 모두 실패하면 새 expert를 생성한다.
7. 테스트 시에는 selector가 expert를 고른다.

- 핵심: recurring task도 재사용 가능하다.

---

## Loss-based Task Switch Detection

- 현재 expert의 배치 손실 $L_c$를 smooth하게 추적한다.
  $$
  L_s \leftarrow \alpha L_c + (1-\alpha)L_s
  $$
  - $\alpha = 0.2$
  - threshold window $W_{th} = 100$
- threshold는 recent loss 평균과 표준편차로 정한다.
  $$
  \tau = \mu + 3\sigma
  $$
- 직관: 현재 expert가 더 이상 현재 입력을 잘 설명하지 못하면 loss가 통계적으로 유의미하게 증가한다.

---

## 기존 expert 재사용 vs 새 expert 생성

- task switch가 감지된 뒤의 규칙은 단순하다.
  - threshold 이하 loss를 보이는 기존 expert가 있으면 재사용
  - 모든 expert가 threshold를 넘으면 새 expert 생성
- 장점:
  - recurring task를 새 task로 오인하지 않을 수 있다.
  - expert 수를 무조건 늘리지 않고 reuse가 가능하다.
- 즉,
  - TAME은 single-model CL이 아니라
    **online expert management system**에 가깝다.

---

## Selector Network

- 학습 중에는 loss를 볼 수 있지만,
테스트 시에는 정답 라벨이 없어 loss 기반 판단이 어렵다.
- 그래서 selector network를 둔다.
  - 학습 중 저장한 소규모 샘플 버퍼를 사용해 selector를 학습한다.
  - selector는 입력 $x$를 보고 어떤 expert를 써야 할지 예측한다.
- 의미:
  - 학습 시 task detection 문제와
    추론 시 routing 문제를 분리해 해결한다.

---

## Pruning의 역할

- multi-expert 구조의 약점은 expert 수 증가에 따른 모델 팽창이다.
- TAME은 이를 줄이기 위해 pruning을 적용한다.
  - selector pruning + retraining
  - 각 expert pruning + 소규모 버퍼로 retraining
- 논문 Table 1 설정:
  - expert pruning rate: 98%
  - selector pruning rate: 50%
- 핵심: 높은 정확도를 유지하면서도 모델 크기를 줄인다.

---

## 실험 설정

- 벤치마크:
  - Permuted MNIST, Split MNIST, Split CIFAR-100, Split CIFAR-10, SVHN-MNIST, MNIST-SVHN
- 구조 예:
  - MNIST류: small conv expert + MLP selector
  - CIFAR류: VGG11 expert + pretrained ResNet18 selector
- 비교 대상:
  - EWC, SI, A-GEM, RWALK, DEN, iTAML, BGD, CN-DPM, HCL

---

## 핵심 결과

- Table 2 기준:
  - Permuted MNIST: **87.32 / 55.53K params**
  - Split MNIST: **98.63 / 37.02K params**
  - Split CIFAR-100 (20): **62.39 / 9.02M params**
- 관찰:
  - 세 benchmark 모두에서 최고 정확도
  - MNIST류에서는 정확도와 파라미터 효율 모두 강함
  - Split CIFAR-100에서는 iTAML, A-GEM보다도 높음
- 메시지: task-agnostic setting에서도 strong baselines를 이길 수 있다.

---

## 추가 결과와 병목 해석

- Table 3 기준:
  - SVHN-MNIST, MNIST-SVHN, Split CIFAR-10, Split CIFAR-100 (10)에서도
    HCL-FR / HCL-GR보다 우수하다.
- 중요한 관찰:
  - selector 정확도는 task similarity에 크게 좌우된다.
  - Split CIFAR-100에서 원래 task 구성보다
    super-class 구성에서 selector 성능이 더 높다.
- 의미: 병목은 expert보다 **routing difficulty**일 수 있다.

---

## 강점

- genuinely task-agnostic한 설정을 정면으로 다룬다.
- loss 기반 detection이 단순하고 해석 가능하다.
- recurring task reuse가 가능하다.
- pruning으로 multi-expert의 크기 문제를 완화한다.

---

## 한계

- loss threshold 가정이 noisy training dynamics에 취약할 수 있다.
- selector 품질이 전체 시스템 병목이 될 수 있다.
- expert 수 증가는 장기적으로 여전히 문제다.
- pruning/retraining은 완전한 streaming 관점에서는 후처리에 가깝다.

---

## 발표용 핵심 메시지

- task-agnostic CL에서는 forgetting 이전에
  **task segmentation 문제**가 먼저다.
- TAME은 single-model regularization이 아니라
  **multi-expert routing 문제**로 설정을 바꾼다.
- 학습 중에는 loss-based switch detection,
  추론 시에는 selector routing으로 문제를 분리한다.
- pruning은 multi-expert 구조를 실용적으로 만드는 핵심 장치다.

---

## 결론

- TAME은 task label 없이도 expert를 전환하거나 추가하는
  task-agnostic continual learning 프레임워크다.
- loss deviation 기반 switch detection,
  selector-based inference routing,
  pruning-based model compression을 결합한다.
- 결과적으로 정확도와 모델 크기 모두에서 강한 성능을 보인다.
- 이 논문은 task-agnostic CL을
  **routing and expert management 문제**로 바꿔 본 중요한 설계 사례다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/tame_task_agnostic_continual_learning_using_multiple_experts_slide.md>
