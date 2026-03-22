---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Optimization and Generalization of Regularization-Based Continual Learning: a Loss Approximation Viewpoint

- Dong Yin, Mehrdad Farajtabar, Ang Li, Nir Levine, Alex Mott
- arXiv 2020
- Regularization-based CL through a loss approximation lens

---

## 문제 설정

- EWC, SI, MAS 같은 정규화 기반 continual learning은 널리 쓰인다.
- 하지만 기존 설명은 주로 직관 수준에 머문다.
- 이 논문은 더 근본적인 질문을 던진다.

핵심 질문:

- 정규화 기반 continual learning은 **실제로 무엇을 근사하고 있는가?**
- 그 근사가 정확하지 않으면 최적화와 일반화는 어떻게 망가지는가?

---

## 핵심 아이디어

- 논문의 핵심 관점은 간단하다.
  - regularization-based CL은 이전 task loss를 직접 다시 계산하지 못한다.
  - 대신 이전 loss를 현재 파라미터 근처에서 **2차 Taylor approximation**으로 대체한다.
  - 결국 각 방법의 차이는 **Hessian을 얼마나 정확히 근사하느냐**의 차이다.
- 즉, EWC, SI, Kronecker 방법을 하나의 loss approximation 프레임에서 해석한다.

---

## 손실 근사 프레임워크

- task $k$ 종료 시점의 파라미터 $\hat{w}_k$ 근처에서 empirical loss를 2차 근사한다.
  $$
  L^{\mathrm{prox}}_k(w)
  = \sum_{k'=1}^{k}
  \Big[
  \hat{L}_{k'}(\hat{w}_{k'})
  + (w-\hat{w}_{k'})^\top \nabla \hat{L}_{k'}(\hat{w}_{k'})
  + \frac{1}{2}(w-\hat{w}_{k'})^\top H_{k'}(w-\hat{w}_{k'})
  \Big]
  $$
- 실제로는 보통 gradient term을 생략해 이전 task별 quadratic penalty 합으로 구현된다.

---

## 기존 방법을 어떻게 해석하나

- 이 관점에서 보면 차이는 Hessian approximation 방식이다.
  - **EWC**: Fisher diagonal approximation
  - **Kronecker-factored Laplace**: Kronecker 구조를 이용한 더 정교한 근사
  - **SI / MAS**: 유도 방식은 다르지만 결과적으로 quadratic regularizer
- 핵심 메시지: 성능 차이는 **이전 loss landscape를 얼마나 정확히 복원하느냐**로 발생한다.

---

## 최적화 관점 핵심 통찰

- 근사 loss에 대해 gradient descent를 해도 실제 loss를 줄이려면 다음이 중요하다.
  - Hessian approximation error, task 간 파라미터 이동 거리, Hessian Lipschitz 성질
- 직관:
  - Hessian 근사가 정확할수록,
  현재 파라미터가 이전 해 근처에 있을수록,
  loss가 더 quadratic-like할수록
  근사 loss 최적화가 실제 loss 감소와 더 잘 정렬된다.

---

## Theorem 레벨 메시지

- 논문이 보여 주는 핵심은 두 가지다.
  1. 한 번의 gradient step이 실제 loss 감소로 이어질 충분조건이 있다.
  2. 그 조건은 최악의 경우 사실상 필요하다.
- 의미:
  - 잘못된 curvature 정보를 쓰면 근사 loss gradient 방향과 실제 loss gradient 방향이 어긋날 수 있다.
  - 즉, forgetting은 단지 정규화 강도의 문제가 아니라 **잘못된 Hessian 근사 문제**이기도 하다.

---

## 일반화 관점 핵심 통찰

- Theorem 4는 일반화 오차를 두 부분으로 분리한다.
  - 표본 수가 늘면 줄어드는 **통계적 오차**
  - 표본 수가 늘어도 사라지지 않는 **근사 오차**
- 의미:
  - 데이터를 더 많이 모아도 성능 저하가 남는다면,
    원인은 단순 overfitting이 아닐 수 있다.
  - 그 원인은 **quadratic approximation 자체의 bias**일 수 있다.
- 이 점이 실무적으로 매우 중요하다.

---

## 실험 설정

- 벤치마크:
  - Permuted MNIST, Rotated MNIST, Split CIFAR
- 비교 대상:
  - Vanilla SGD, EWC, SI, Kronecker-factored Laplace approximation
- 핵심 비교 포인트:
  - 더 정확한 Hessian approximation이 실제 성능 차이로 이어지는가?

---

## 핵심 결과 (1/2)

- Permuted MNIST
  - Multi-task upper bound: $96.8$
  - Kronecker, 32 epoch: **$96.0$**
  - EWC, 32 epoch: **$56.9$**
  - Vanilla SGD, 32 epoch: **$52.6$**

---

## 핵심 결과 (2/2)

- Rotated MNIST
  - Multi-task upper bound: $97.2$
  - Kronecker, 32 epoch: **$81.9$**
  - EWC: **$62.4$**
  - SI: **$58.3$**
- 결론: 더 정교한 Hessian approximation이 훨씬 강하다.

---

## Split CIFAR 결과와 해석 (1/2)

- Multi-task upper bound: $62.6$
- Kronecker, 32 epoch: **$59.9$**
- EWC, 32 epoch: **$44.0$**
- Vanilla SGD, 128 epoch: **$30.9$**

---

## Split CIFAR 결과와 해석 (2/2)

- 중요한 관찰:
  - 개별 task를 더 오래 학습한다고 항상 좋은 것이 아니다.
  - 근사 오차가 큰 방법은 epoch를 늘리면 오히려 성능이 다시 떨어진다.
- 즉,
  - long training은 improvement가 아니라
    **forgetting accumulation**으로 이어질 수 있다.

---

## 계산 비용 Trade-off

- Kronecker 방법은 가장 잘 되지만 비용이 크다.
  - Permuted / Rotated MNIST: EWC 대비 약 5배 시간
  - Split CIFAR: EWC 대비 약 10배 시간
- 메시지:
  - 더 정확한 curvature approximation은 더 좋은 성능을 주지만,
    계산량과 메모리 비용을 함께 올린다.

---

## 강점

- 정규화 기반 continual learning을 하나의 이론 틀로 묶었다.
- EWC, SI, Kronecker를 공통 언어로 비교할 수 있다.
- forgetting과 optimization failure를 연결해 설명한다.
- generalization error와 approximation error를 분리한다.

---

## 한계

- 이론 가정이 비교적 강하다.
- 실험 벤치마크가 고전적이다.
- replay / expansion 계열과의 비교는 중심이 아니다.
- 실제 구현에서는 gradient term을 대개 생략한다.

---

## 발표용 핵심 메시지

- 이 논문의 요점은 "EWC가 왜 약한가"를 정규화 직관이 아니라
  **loss landscape approximation 정확도**로 설명했다는 데 있다.
- regularization-based CL의 본질은
  이전 task loss를 quadratic surrogate로 보존하는 것이다.
- 따라서 좋은 방법은
  **더 좋은 Hessian approximation을 가진 방법**이다.
- 실무적으로는 early stopping, learning-rate decay,
  approximation quality가 함께 중요하다.

---

## 결론

- 이 논문은 regularization-based continual learning을
  이전 task loss의 2차 근사 문제로 재해석한다.
- 핵심 병목은 정규화 강도보다
  **Hessian / curvature approximation accuracy**다.
- 더 정확한 근사는 더 좋은 성능을 주지만,
  그만큼 계산 비용도 크다.
- 이 논문은 정규화 기반 continual learning을
  **loss approximation 문제**로 바라보는 기준점을 제공한다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/optimization_and_generalization_of_regularization_based_continual_learning_a_loss_approximation_viewpoint_slide.md>
