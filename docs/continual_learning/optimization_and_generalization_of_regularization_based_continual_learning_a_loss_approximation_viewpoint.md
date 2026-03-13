<!-- markdownlint-disable MD013 MD041 -->
# Optimization and Generalization of Regularization-Based Continual Learning: a Loss Approximation Viewpoint
<!-- markdownlint-enable MD013 MD041 -->

**저자**: Dong Yin, Mehrdad Farajtabar, Ang Li, Nir Levine,
Alex Mott

**연도**: 2020

**arXiv ID**: <https://arxiv.org/abs/2006.10974v3>

**DOI**: <https://doi.org/10.48550/arXiv.2006.10974>

---

## 1. 연구 배경 및 문제 정의

Regularization-based continual learning은
과거 데이터를 저장하지 않고도 catastrophic forgetting을 줄이기 위한
대표적 계열이다.
대표적으로 EWC, SI, MAS, Kronecker-factored Laplace 계열이 여기에 속한다.
하지만 기존 연구들은 보통
"이 정규화 항이 왜 작동하는가"를 직관적으로 설명하는 수준에 머물렀고,
최적화와 일반화 관점의 이론적 해석은 부족했다.

이 논문은 이 공백을 메우려 한다.
저자들의 문제의식은 단순하다.
정규화 기반 continual learning은 결국
이전 작업의 손실을 현재 시점에서 직접 다시 계산하지 못하므로,
그 손실을 어떤 근사 형태로 대체해 쓰는 것이다.
그렇다면 핵심 질문은 다음이 된다.

- 이전 작업 손실의 근사가 얼마나 정확해야 하는가
- 근사 손실에 대해 gradient descent를 수행해도
  실제 원래 손실을 줄일 수 있는가
- 이러한 근사 기반 학습은
  finite-sample 관점에서 어떻게 일반화되는가

논문은 이 질문들을
**이전 작업 손실의 2차 Taylor 근사(second-order Taylor approximation)**
라는 관점으로 통일해 다룬다.

---

## 2. 핵심 기여

이 논문의 기여는 네 가지로 정리할 수 있다.

1. **통합 관점 제시**
   정규화 기반 continual learning을
   이전 작업 손실의 2차 손실 근사 문제로 정식화한다.
   이 틀 안에서 EWC와 Kronecker-factored Laplace를
   같은 프레임워크의 특수한 경우로 해석한다.

2. **최적화 이론 제공**
   근사 손실에 대한 gradient descent가
   실제 손실도 줄이게 되는 충분조건을 제시하고,
   그 조건이 최악의 경우 사실상 필요하다는 점도 보인다.

3. **수렴 분석 제공**
   비볼록과 볼록 손실 각각에 대해,
   Hessian 근사 오차와 task 간 이동 거리,
   고차 미분 항이 수렴에 어떤 영향을 주는지 분석한다.

4. **일반화 분석 제공**
   regularized empirical objective와 실제 population objective 사이의 차이를
   finite-sample bound로 제시한다.
   여기서 Rademacher complexity에 의한 표본 오차와
   손실 근사 오차가 분리되어 등장한다.

---

## 3. 방법론 요약

### 3.1 문제 설정

논문은 순차적으로 도착하는 $K$개의 supervised task
$T_1, \dots, T_K$를 가정한다.
각 task $T_k$는 분포 $D_k$를 가지며,
population loss는 다음과 같이 정의된다.

$$
L_k(w) := \mathbb{E}_{(x,y) \sim D_k}[\ell_k(w; x, y)]
$$

continual learning에서는 task $k$를 학습할 때
그 task의 샘플만 사용할 수 있고,
학습이 끝난 뒤에는 원본 데이터에 다시 접근할 수 없다.
대신 task별 부가 정보(side information)는 저장할 수 있다.

### 3.2 손실 근사 프레임워크

핵심 아이디어는 task $k$가 끝났을 때 얻은 파라미터
$\hat{w}_k$ 근처에서,
그 task의 empirical loss를 2차 Taylor 전개로 근사하는 것이다.

두 task만 있는 단순한 경우,
첫 번째 task 손실 $\hat{L}_1(w)$를 $\hat{w}_1$ 주변에서
다음처럼 근사한다.

$$
L^{\mathrm{prox}}_1(w) = \hat{L}_1(\hat{w}_1) +
(w-\hat{w}_1)^\top \nabla \hat{L}_1(\hat{w}_1) +
\frac{1}{2}(w-\hat{w}_1)^\top \nabla^2 \hat{L}_1(\hat{w}_1)(w-\hat{w}_1)
$$

일반적인 $k$ task 누적 버전은,
이전 task들의 손실을 모두 이런 식으로 저장해
현재 task 학습 시 surrogate로 사용하는 구조다.
논문은 이를 다음 형태로 정리한다.

$$
L^{\mathrm{prox}}_k(w)
= \sum\_{k'=1}^{k}
\Big[
\hat{L}\_{k'}(\hat{w}\_{k'}) +
(w-\hat{w}\_{k'})^\top \nabla \hat{L}\_{k'}(\hat{w}\_{k'}) +
\frac{1}{2}(w-\hat{w}\_{k'})^\top H\_{k'}(w-\hat{w}\_{k'})
\Big]
$$

여기서 $H_{k'}$는 정확한 Hessian일 수도 있고,
그 근사치일 수도 있다.
실제 학습에서는 상수항은 무시되고,
각 task 종료 시 gradient가 작다고 보고
선형항도 보통 생략된다.
그러면 실질적으로는
**이전 task별 quadratic penalty의 합**이 된다.

### 3.3 기존 방법과의 연결

이 관점에서 보면,
기존 regularization-based 방법들은
결국 Hessian을 어떻게 근사하느냐만 다르다.

- **EWC**:
  negative log-likelihood 조건에서
  Fisher information의 대각 원소를 이용해
  Hessian의 diagonal approximation을 쓴다.
- **Kronecker-factored Laplace**:
  Fisher/Hessian을 Kronecker factorization으로 근사해
  off-diagonal 구조 일부까지 반영한다.
- **SI, MAS**:
  유도 방식은 다르지만,
  결과적으로 quadratic regularizer를 만든다는 점에서
  같은 틀 안에서 분석 가능하다.

저자들의 주장은 명확하다.
regularization-based continual learning의 성능 차이는
결국 **이전 손실 곡면을 얼마나 정확히 복원하느냐**의 차이로
이해할 수 있다는 것이다.

---

## 4. 이론 분석

### 4.1 최적화 관점의 핵심 통찰

논문은 근사 손실에 대해 gradient descent를 할 때,
실제 우리가 줄이고 싶은 손실도 함께 감소하려면
세 가지가 중요하다고 말한다.

- **Hessian 근사 오차**
- **task 간 파라미터 이동 거리**
- **3차 미분에 해당하는 Hessian Lipschitz 성질**

직관적으로,
근사 Hessian이 정확할수록,
현재 파라미터가 이전 task optimum에서 멀리 벗어나지 않을수록,
그리고 손실 곡면이 지역적으로 더 "quadratic-like"할수록,
근사 손실을 최적화해도 실제 손실을 함께 줄이기 쉽다.

### 4.2 One-step 충분조건과 필요성

논문의 Theorem 1은
한 번의 gradient step이 실제 손실 감소로 이어질 충분조건을 제시한다.
이 조건은 본질적으로 다음 경향을 담고 있다.

- Hessian 근사 오차가 작을수록 유리하다.
- 현재 파라미터가 이전 task 해 근처에 있을수록 유리하다.
- 고차 곡률이 작을수록 유리하다.
- 학습 후반으로 갈수록 learning rate decay가 더 중요해진다.

또한 Proposition 1은
이 조건이 단순한 보수적 충분조건이 아니라,
최악의 경우 이를 어기면
근사 손실의 gradient 방향과
실제 손실의 gradient 방향이 반대로 갈 수도 있음을 보인다.
즉, 이 논문은
정규화 기반 continual learning에서 "잘못된 curvature 정보"가
직접적인 망각으로 이어질 수 있음을 이론적으로 분명히 한다.

### 4.3 수렴 분석

논문은 비볼록과 볼록의 두 경우를 나눠 본다.

- **Theorem 2 (non-convex)**:
  평균 gradient norm에 대한 bound를 제시한다.
  여기서는 Hessian 근사 오차와 task 간 이동 거리 항이
  수렴을 방해하는 항으로 직접 등장한다.
- **Theorem 3 (convex)**:
  convex loss에서도
  근사 오차가 남아 있으면 실제 최적점까지의 수렴을
  보장할 수 없음을 보인다.
  반대로 손실이 정확히 quadratic이고
  full Hessian을 저장할 수 있다면,
  표준 gradient descent와 같은 수렴률을 회복할 수 있다.

핵심 메시지는 간단하다.
regularization-based CL의 실패는
"정규화 세기를 잘못 골랐기 때문"만이 아니라,
**이전 손실을 부정확하게 근사했기 때문**이라고 볼 수 있다.

### 4.4 일반화 분석

일반화 분석에서 논문은
마지막 task에서 실제로 사용하는 regularized empirical loss와
우리가 궁극적으로 관심 있는 평균 population loss 사이의 차이를 bound한다.

Theorem 4의 구조는 두 부분으로 나뉜다.

- **표본 수에 따라 줄어드는 통계적 오차**:
  loss, gradient, Hessian에 대한 Rademacher complexity로 표현된다.
- **표본 수가 늘어도 사라지지 않는 근사 오차**:
  이전 손실을 quadratic으로 대체한 데서 오는 bias다.

이 분해는 실무적으로 중요하다.
데이터를 더 많이 모아도 계속 남는 성능 저하가 있다면,
그 원인은 일반화 부족이 아니라
**Hessian 근사 자체의 부정확성**일 수 있다는 뜻이기 때문이다.

---

## 5. 실험 설정

### 5.1 벤치마크

논문은 세 가지 표준 continual learning 벤치마크를 사용한다.

- **Permuted MNIST**:
  20-task sequence.
  각 task는 픽셀 permutation이 다른 MNIST 분류 문제다.
- **Rotated MNIST**:
  20-task sequence.
  각 task는 서로 다른 회전 각도를 갖는 MNIST 분류 문제다.
- **Split CIFAR**:
  CIFAR-100의 coarse label 20개를 task로 분할한 설정이다.

### 5.2 모델 구조와 학습 설정

- Permuted MNIST, Rotated MNIST:
  hidden layer 2개를 가진 MLP 사용
- Split CIFAR:
  2개 convolution layer와 1개 fully connected hidden layer를 가진 CNN 사용
- Split CIFAR에서는 task별 output head를 두는 multi-head 구조 사용
- 학습률은 $10^{-3}$,
  batch size는 10 사용
- 결과는 여러 independent run의 평균과 표준편차로 보고
- regularization coefficient $\lambda$는
  전체 task 평균 정확도가 가장 좋은 값으로 선택

### 5.3 비교 대상

실험에는 다음 네 방법이 포함된다.

- **Vanilla SGD**
- **EWC**
- **SI**
- **Kronecker-factored Laplace approximation**

논문의 목적상,
Kronecker 방법은 더 정확한 Hessian 근사의 대표,
EWC와 SI는 상대적으로 거친 근사의 대표로 사용된다.

---

## 6. 실험 결과

### 6.1 Permuted MNIST

Permuted MNIST에서 multi-task upper bound는
$96.8 \pm 0.0$이다.
Kronecker는 epoch 수가 늘수록 일관되게 가장 좋은 결과를 보였고,
32 epoch에서는 **$96.0 \pm 0.0$**에 도달한다.
반면 EWC는 1 epoch에서 **$62.8 \pm 0.7$**,
32 epoch에서도 **$56.9 \pm 1.9$**에 머문다.
Vanilla SGD는 1 epoch **$59.1 \pm 0.9$**에서
32 epoch **$52.6 \pm 1.6$**으로 오히려 감소한다.

이 결과는 diagonal Fisher 수준의 근사만으로는
Permutation이 큰 다중 task를 충분히 보존하기 어렵고,
off-diagonal 구조까지 반영하는 Hessian 근사가
훨씬 효과적이라는 논문의 주장을 지지한다.

### 6.2 Rotated MNIST

Rotated MNIST에서도 같은 패턴이 나타난다.
multi-task 기준은 $97.2 \pm 0.1$이고,
Kronecker는 32 epoch에서 **$81.9 \pm 0.9$**,
EWC는 **$62.4 \pm 2.5$**,
SI는 **$58.3 \pm 0.8$**를 기록한다.

즉, task 간 변화가 permutation이 아니라 rotation일 때도,
보다 정교한 curvature approximation이
더 나은 최종 평균 성능으로 이어진다.

### 6.3 Split CIFAR

Split CIFAR는 더 어려운 벤치마크이며,
multi-task upper bound는 $62.6 \pm 0.2$이다.
여기서도 Kronecker가 가장 강하다.

- 16 epoch: Kronecker **$54.2 \pm 0.7$**
- 32 epoch: Kronecker **$59.9 \pm 0.7$**
- 64 epoch: Kronecker **$59.7 \pm 0.6$**
- 128 epoch: Kronecker **$53.1 \pm 1.1$**

EWC와 SI, Vanilla SGD는 일정 epoch까지는 좋아지다가,
더 오래 학습하면 성능이 다시 떨어지는 경향이 뚜렷하다.
예를 들어 Vanilla SGD는
16 epoch에서 **$43.2 \pm 1.4$** 까지 올랐다가
128 epoch에서는 **$30.9 \pm 1.6$** 로 크게 하락한다.
EWC 역시 32 epoch **$44.0 \pm 0.9$** 이후
128 epoch **$37.0 \pm 0.9$** 로 감소한다.

이 현상은 논문의 Theorem 2와 직접 연결된다.
개별 task를 더 오래 학습하는 것이
항상 더 좋은 최종 평균 정확도로 이어지지 않으며,
근사 오차가 큰 방법일수록
과적합이 아니라 **망각 누적**으로 인해 성능이 악화될 수 있다.

### 6.4 계산 비용 trade-off

Kronecker의 우수한 성능에는 비용이 따른다.
논문에 따르면 계산 시간은
Permuted/Rotated MNIST에서 EWC 대비 약 5배,
Split CIFAR에서는 약 10배다.
즉, 이 논문은 "더 정확한 Hessian 근사가 좋다"는 결론과 함께,
그 정확도와 계산 비용 사이의 실질적 trade-off도 분명히 보여 준다.

---

## 7. 한계 및 비판적 검토

### 7.1 이론의 가정이 강하다

논문의 이론은 smoothness,
Hessian Lipschitz,
근사 Hessian 오차의 boundedness 같은 가정에 크게 의존한다.
이 가정들은 현대 딥러닝에서 완전히 비현실적이라고 말할 수는 없지만,
실제 대규모 비선형 네트워크에서 엄밀히 확인하기는 어렵다.

### 7.2 task 설정이 비교적 고전적이다

실험은 Permuted MNIST, Rotated MNIST, Split CIFAR에 집중되어 있다.
이들은 continual learning 논문에서 자주 쓰이는 표준 벤치마크지만,
오늘 기준으로는 class-incremental,
online continual learning,
대규모 vision-language setting을 대표하지는 못한다.

### 7.3 regularization 계열 내부 비교 중심이다

이 논문은 regularization-based CL의 이론을 세우는 것이 목적이므로,
replay-based method나 expansion-based method와의 강한 비교는 하지 않는다.
따라서 practical SOTA를 비교하는 논문이라기보다는,
**한 계열의 방법을 깊게 해석하는 이론 논문**으로 읽는 편이 맞다.

### 7.4 gradient 항 생략의 현실성 문제

통합 프레임워크는 원래 gradient term까지 포함한 2차 근사지만,
실제 알고리즘 설명에서는 gradient가 작다는 이유로
대개 이를 생략한다.
이는 실용적으로는 자연스럽지만,
엄밀한 관점에서는 근사 손실과 실제 구현 사이의 차이를 남긴다.

---

## 8. 연구적 의미

이 논문의 가장 중요한 가치는
regularization-based continual learning을
"중요한 파라미터를 덜 바꾸자"는 직관에서 끌어올려,
**이전 task 손실을 2차 근사로 보존하는 문제**로 재해석했다는 점이다.

이 관점의 장점은 크다.

- EWC, Kronecker, SI, MAS를 같은 좌표계에서 비교할 수 있다.
- 왜 어떤 방법이 더 잘 작동하는지
  Hessian 근사 정확도라는 공통 언어로 설명할 수 있다.
- 학습을 오래 할수록 오히려 망각이 심해질 수 있다는 현상을
  이론적으로 설명할 수 있다.
- 일반화 오차와 근사 오차를 분리해 해석할 수 있다.

continual learning 분야에서 이 논문은
새 알고리즘을 제안했다기보다,
**기존 알고리즘들을 해석하는 상위 수준의 분석 틀**을 제공한 작업으로 보는 것이 정확하다.

---

## 9. 실무적 인사이트

- EWC 같은 diagonal approximation이 기대보다 약하게 동작한다면,
  정규화 세기보다 먼저 Hessian 근사 정확도를 의심해야 한다.
- 더 오래 학습하는 것이 항상 좋은 것은 아니다.
  regularization-based CL에서는 early stopping과
  learning rate decay가 구조적으로 중요하다.
- 더 정확한 curvature approximation은 성능을 높일 수 있지만,
  계산량과 메모리 비용도 급격히 늘어난다.
- 실제 시스템에서는 Kronecker 계열처럼 무거운 방법 대신,
  blockwise approximation이나 low-rank approximation 같은
  중간 타협 지점을 찾는 것이 현실적일 수 있다.

---

## 10. 결론

이 논문은 regularization-based continual learning을
이전 task 손실의 2차 Taylor 근사 문제로 재해석하고,
그 위에서 최적화와 일반화의 이론을 정리한 논문이다.

핵심 메시지는 분명하다.
regularization-based continual learning의 성패는
결국 **이전 손실 곡면을 얼마나 정확히 근사하느냐**에 달려 있다.
더 정확한 Hessian 근사는 더 좋은 최종 성능으로 이어지지만,
그만큼 큰 계산 비용이 필요하다.
또한 개별 task를 오래 학습하는 것이 반드시 유리하지 않으며,
근사 오차가 크면 오히려 망각이 심해질 수 있다.

따라서 이 논문은
정규화 기반 continual learning을 설계하거나 해석할 때,
정규화 항을 경험적으로 조정하는 수준을 넘어
**loss landscape approximation 문제**로 바라봐야 한다는 기준점을 제공한다.
