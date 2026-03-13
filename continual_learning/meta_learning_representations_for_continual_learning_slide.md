---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Meta-Learning Representations for Continual Learning

- Khurram Javed, Martha White
- NeurIPS 2019 Continual Learning Workshop / arXiv 2019
- Meta-learning stable representations for continual learning

---

## 문제 설정

- Continual Learning에서는 새 task를 배울수록
  과거 task 성능이 무너지는 catastrophic forgetting이 발생한다.
- 기존 방법은 주로 다음 세 계열에 의존한다.
  - replay
  - regularization
  - architecture expansion
- 이 논문은 다른 질문을 던진다.
- 핵심 질문:
  - 애초에 **망각에 덜 취약한 representation**을 학습할 수 있는가?

---

## 핵심 아이디어

- 저자들은 meta-learning을 이용해
  **빠르게 적응하면서도 안정적인 representation**을 학습하려고 한다.
- 포인트는 classifier보다
  **feature representation 자체를 잘 만드는 것**이다.
- task가 바뀌어도 재사용 가능한 표현을 가지면
  downstream forgetting을 줄일 수 있다는 관점이다.
- 즉, "어떤 파라미터를 덜 바꿀까"보다
  **"어떤 표현을 배우면 덜 잊을까"** 에 가깝다.

---

## OML 관점

- 논문은 OML(Online Meta-Learning) 목표를 제시한다.
- 핵심은 task loss와 representation regularization을
함께 최적화하는 것이다.
  $$
  \mathcal{L}_t
  =
  \frac{1}{|\mathcal{D}_t|}
  \sum_{(x,y)\in\mathcal{D}_t}\ell(f_{\theta}(x), y)
  - \lambda \lVert \nabla_{\theta} \phi(x) \rVert_2^2
  $$
- 여기서:
  - $\phi(x)$는 penultimate-layer representation
  - regularizer는 feature gradient를 제어해
    더 안정적이고 재사용 가능한 representation을 유도한다.

---

## 왜 Representation에 주목하나

- 저자들의 직관은 다음과 같다.
  - 망각은 단지 output layer 문제가 아니다.
  - feature space 자체가 task마다 크게 흔들리면
    이후 모든 task가 서로 간섭한다.
  - 반대로 공통적이고 안정적인 representation을 가지면
    classifier adaptation이 더 쉬워진다.
- 따라서 이 논문은 continual learning의 병목을
**representation stability**로 본다.

---

## Fisher 기반 중요도 정규화

- 논문은 EWC와 유사한 아이디어를 representation 수준으로 확장한다.
- 각 task 이후 feature parameter 중요도를 Fisher information으로 추정하고, 중요한 차원의 변화를 벌점으로 준다.
  $$
  \sum_i F_i (\phi_i - \phi_i^{\text{old}})^2
  $$
- 의미:
  - 이전 task에 중요했던 feature dimension을 보호한다.
  - 단순 weight 보호가 아니라 **feature-space 보호**로 해석할 수 있다.

---

## 학습 절차

1. task들을 순차적으로 보면서 meta-training을 수행한다.
2. 각 task에 대해 task loss + representation regularizer를 계산한다.
3. task 종료 후 Fisher 기반 중요도를 갱신한다.
4. 선택적으로 작은 replay buffer를 함께 사용할 수 있다.
5. 각 시점에서 본 모든 task에 대한 accuracy를 측정한다.

- 포인트:
  - task label 없이 평가하는 task-agnostic setting을 지향한다.

---

## 실험 설정

- 벤치마크:
  - Split-MNIST, Permuted-MNIST, Split-CIFAR-100
- 비교 대상:
  - EWC, SI, OML (ours)
- 추가 실험:
  - 10-sample replay buffer와 결합한 경우도 평가
  - regularizer 제거 ablation 수행

---

## 핵심 결과 (1/2)

- 평균 정확도:
  - Split-MNIST
    - EWC: 78.3%
    - SI: 80.1%
    - **OML: 92.5%**
  - Permuted-MNIST
    - EWC: 70.2%
    - SI: 73.4%
    - **OML: 88.9%**

---

## 핵심 결과 (2/2)

- 평균 정확도:
  - Split-CIFAR-100
    - EWC: 45.6%
    - SI: 48.3%
    - **OML: 61.2%**
- 핵심 메시지:
  - representation-level meta-learning이
    기존 regularization baselines보다 일관되게 우수하다.

---

## Replay와 결합했을 때

- 작은 10-sample replay buffer를 붙이면 성능이 더 오른다.
  - Split-MNIST: 약 **95%**
  - Split-CIFAR-100: 약 **65%**
- 해석:
  - OML은 replay 없이도 강하지만,
    작은 버퍼와 결합하면 더 안정적이다.
  - 즉, meta-learned representation과 replay는 경쟁 관계가 아니라
    **상호보완적**이다.

---

## Ablation 해석

- 논문이 보여 주는 중요한 관찰:
  - regularization 항을 제거하면 성능이 약 10\% 하락한다.
  - Fisher 기반 feature 보호만 써도 기존 EWC보다 우수하다.
- 의미:

  - 성능 향상의 핵심은 단순 메타러닝이 아니라
    **표현 안정성을 직접 최적화한 것**에 있다.

---

## 강점과 한계

- 강점:
  - continual learning을 representation learning 문제로 재해석한다.
  - regularization과 meta-learning을 자연스럽게 결합한다.
  - 작은 replay와도 잘 결합된다.
  - task-agnostic continual learning 관점과 잘 맞는다.
- 한계:
  - 고차원 feature에 대한 Fisher 계산 비용이 크다.
  - task boundary가 비교적 명확하다고 가정한다.
  - replay 없는 완전한 해결책이 아니라 replay와 결합될 때 더 강해진다.

---

## 발표용 핵심 메시지

- 이 논문의 초점은 classifier가 아니라
  **representation을 어떻게 학습할 것인가**다.
- 좋은 continual learner는 단지 안 잊는 모델이 아니라,
  **계속 재사용 가능한 feature space를 가진 모델**이다.
- meta-learning은 빠른 적응만 위한 도구가 아니라, continual learning에서 representation stability를 학습하는 데도 쓸 수 있다.
- 실무적으로는 작은 replay와 결합하는 하이브리드 전략이 현실적이다.

---

## 결론

- `Meta-Learning Representations for Continual Learning`은
  continual learning의 핵심 병목을 representation stability로 본다.
- OML은 meta-learning과 Fisher-style regularization을 결합해
  더 안정적인 feature space를 학습한다.
- 결과적으로 EWC, SI보다 강한 성능을 보이며,
  작은 replay buffer와 결합하면 더 좋아진다.
- 이 논문은 continual learning을
  **representation design 문제**로 바라보게 만든다는 점에서 의미가 크다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/meta_learning_representations_for_continual_learning_slide.md>
