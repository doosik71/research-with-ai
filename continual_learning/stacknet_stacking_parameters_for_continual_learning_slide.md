---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# StackNet: Stacking Parameters for Continual Learning

- Jangho Kim, Jeesoo Kim, Nojun Kwak
- arXiv 2018 / CVPR Workshops 2020
- Task-wise parameter stacking with an index module

---

## 문제 설정

- Continual Learning에서는 새 task를 배울 때
  이전 task를 잊는 catastrophic forgetting이 핵심 문제다.
- 기존 접근은 대체로 세 가지다.
  - regularization
  - replay
  - dynamic architecture
- 이 논문은 특히 PackNet류 pruning 기반 방법의 대안을 찾는다.
- 핵심 질문: pruning 없이도 task별 파라미터를 구조적으로 분리할 수 있는가?

---

## 핵심 아이디어

- 논문은 두 요소를 결합한다.
  - **StackNet**: 새 task가 들어오면 새 파라미터 구간을 쌓아 올린다.
  - **Index module**: 입력이 어느 task에서 왔는지 추정한다.
- 핵심 목적:
  - 이전 task 성능 저하 없이
  - 새 task용 capacity를 점진적으로 추가한다.
- 즉, forgetting을 regularizer가 아니라
    **parameter allocation 구조**로 막는 접근이다.

---

## StackNet 직관

- StackNet의 동작은 단순하다.
  1. 첫 task는 전체 모델 용량의 일부만 사용한다.
  2. 다음 task는 이전 task 파라미터를 고정한다.
  3. 남은 용량 일부를 새 task 전용으로 학습한다.
  4. 이후 task도 같은 방식으로 반복한다.
- 포인트:
  - 이전 task 파라미터는 frozen
  - 현재 task 파라미터만 trainable
- 이 구조가 forgetting을 직접 차단한다.

---

## 수식 관점

- 핵심은 loss보다 parameter partition에 있다.
  $$
  W_k = W_k^P \cup W_k^T,
  \quad
  W_k^P \cap W_k^T = \varnothing
  $$
- 여기서:
  - $W_k^P$: 이전 task에서 사용되어 고정된 파라미터
  - $W_k^T$: 현재 task를 위해 새로 학습되는 파라미터
- 현재 task는 일반적인 classification loss로 학습한다.
  $$
  \mathcal{L}_{\text{cls}} = - \sum_{c=1}^{C_J} y_c \log p_c
  $$
- 핵심은 "무엇을 학습하느냐"보다 **어느 파라미터가 gradient를 받느냐**다.

---

## 왜 StackNet이 유리한가

- 이 접근의 장점은 명확하다.
  - 이전 task에 쓰인 파라미터를 건드리지 않는다.
  - 따라서 catastrophic forgetting이 구조적으로 줄어든다.
  - pruning, mask 관리, iterative fine-tuning이 필요 없다.
  - 새 task는 이전 task 파라미터를 초기 자원처럼 활용하면서도
    자신의 전용 capacity를 확보할 수 있다.
- 즉, simple but strong dynamic architecture baseline이다.

---

## Index Module의 역할

- 문제는 추론 시점이다.
  - 어떤 입력이 어느 task에서 왔는지 알아야
    어떤 parameter block과 classifier를 쓸지 결정할 수 있다.
- 이를 위해 index module을 둔다.
- 논문에서 제안한 대표 구현:
  - task별 generator / binary classifier 기반 module
- 추론 규칙:
  $$
  J = \arg\max_i B_i(x)
  $$
- 즉, index module은 task routing을 담당한다.

---

## 전체 파이프라인

```text
입력 x
  -> Index module
  -> task index J 추정
  -> StackNet의 task J 구간 활성화
  -> task J classifier로 예측
```

- 역할 분리:
  - Index module: "이 입력은 어느 task인가?"
  - StackNet: "그 task 안에서 무엇인가?"
- 이 task routing 분리가 이 논문의 중요한 설계 포인트다.

---

## 실험 설정

- 주요 데이터셋:
  - MNIST, SVHN, CIFAR-10, ImageNet-A, ImageNet-B
- 추가 5-task 실험:
  - CIFAR-10, SVHN, KMNIST, FashionMNIST, MNIST
- 비교 대상:
  - LwF, PackNet, Single network baseline
- 백본:
  - VGG-16, ResNet-32, ResNet-50

---

## 핵심 결과 (1/2)

- MNIST -> SVHN
  - StackNet 평균 정확도: **95.90%**
  - PackNet 평균 정확도: **95.47%**
  - LwF 평균 정확도: 약 **95.5%**
- SVHN -> CIFAR-10
  - StackNet 평균 정확도: **84.46%**
  - PackNet 평균 정확도: **84.81%**

---

## 핵심 결과 (2/2)

- MNIST -> SVHN -> CIFAR-10
  - StackNet 평균 정확도: **88.76%**
  - PackNet 평균 정확도: **85.88%**
- 메시지:
  - StackNet은 pruning 없이도 PackNet과 경쟁력 있거나 더 좋다.

---

## ImageNet Subset 결과

- ImageNet-A -> ImageNet-B 실험:
  - StackNet 평균 정확도: **85.98%**
  - PackNet 평균 정확도: **85.44%**
  - LwF 평균 정확도: **84%대**
- 의미:
  - 더 현실적인 이미지 분류 설정에서도
    StackNet은 단순 toy result에 머물지 않는다.
  - shortcut connection이 있는 ResNet-50에서도 동작한다.

---

## Index Module 결과

- 논문은 task identity 추정도 높은 정확도를 보인다고 보고한다.
- 예:
  - MNIST -> SVHN: 약 **99.97%**
  - SVHN -> CIFAR-10: 약 **98.45%**
- 즉,
  - StackNet의 핵심은 단지 parameter stacking이 아니라
    **task routing까지 함께 해결하려는 시도**다.

---

## 강점

- forgetting을 구조적으로 차단한다.
- PackNet 대비 절차가 단순하다.
- label-expandable continual learning과 잘 맞는다.
- task routing 문제를 명시적으로 다룬다.

---

## 한계

- task 수가 많아지면 결국 용량 한계에 부딪힌다.
- index module 품질에 성능이 의존한다.
- 유사한 task 분포에서는 task 식별이 어려워질 수 있다.
- vision classification 중심 검증이다.

---

## 발표용 핵심 메시지

- 이 논문의 본질은 **"안 잊게 정규화하자"가 아니라 "애초에 task별 파라미터를 분리하자"** 이다.
- StackNet은 pruning 없이도 강한 dynamic architecture baseline이 된다.
- continual learning에서는 classifier뿐 아니라
  **입력이 어느 task에 속하는가를 추정하는 routing 문제**도 중요하다.
- 따라서 StackNet은 parameter allocation + task routing을 함께 제안한 실용적 구조다.

---

## 결론

- `StackNet`은 task별 파라미터 구간을 순차적으로 쌓아 올리는 방법이다.
- 이전 task 파라미터를 고정하고 새 task용 파라미터만 학습해
  catastrophic forgetting을 구조적으로 억제한다.
- index module을 통해 task identity가 명시되지 않은 상황도 다루려 한다.
- 결과적으로 PackNet과 경쟁력 있는 성능을 더 단순한 절차로 달성한 점이
  이 논문의 핵심 가치다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/stacknet_stacking_parameters_for_continual_learning_slide.md>
