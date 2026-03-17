# Evolutionary Optimization of Deep Learning Activation Functions

## 1. Paper Overview

이 논문은 딥러닝에서 activation function 선택이 성능에 큰 영향을 주는데도, 실제로는 여전히 ReLU가 가장 널리 쓰인다는 문제의식에서 출발한다. 저자들은 사람이 직접 activation을 설계하는 대신, **evolutionary algorithm으로 activation function 자체를 자동 탐색**할 수 있는지를 묻는다. 이를 위해 activation function을 tree 구조로 표현하고, mutation, crossover, exhaustive search를 통해 방대한 후보 공간을 탐색한다. 실험은 CIFAR-10과 CIFAR-100에서 Wide Residual Network를 학습시키는 방식으로 수행되며, 진화로 발견된 activation이 ReLU를 능가하고, task-specific하게 진화시킬 때 가장 좋은 성능을 낼 수 있음을 보인다.  

이 문제가 중요한 이유는 activation design이 아직도 비교적 덜 탐색된 설계 축이기 때문이다. topology, optimizer, loss, learning rate에 비해 activation은 소수의 표준 함수에 의존하는 경우가 많다. 논문은 이를 **metalearning의 새로운 차원**으로 해석한다. 즉, 단일 모델의 weight를 학습하는 것을 넘어서, 모델이 사용할 activation law 자체를 자동으로 찾아내겠다는 관점이다.

## 2. Core Idea

핵심 아이디어는 activation function을 고정된 수학식으로 보지 않고, **연산자 트리(tree of operators)** 로 표현해 진화 가능한 개체로 다루는 것이다. 각 activation은 unary operator와 binary operator의 조합으로 구성되며, mutation은 트리의 한 노드를 다른 연산자로 바꾸고, crossover는 두 activation tree의 같은 깊이 subtree를 교환한다. 이렇게 하면 사람이 직관적으로 떠올리기 어려운 복잡한 activation도 탐색할 수 있다.

또 하나의 핵심은 “보편적으로 좋은 activation 하나를 찾는 것”보다, **특정 architecture와 dataset에 특화된 activation을 진화시키는 것**이 더 강력할 수 있다는 주장이다. 논문은 Ramachandran et al.의 Swish처럼 하나의 범용 activation을 찾는 접근보다 한 걸음 더 나아가, activation function을 task-specific하게 metalearning하는 방향을 제안한다.

## 3. Detailed Method Explanation

### 3.1 Activation function의 표현 방식

논문에서 activation function은 수식 문자열이 아니라 **트리 구조**로 표현된다. 구조적으로는 두 unary operator가 하나의 binary operator로 합쳐지는 계층적 형태를 반복한다. 이 표현 방식의 장점은 연산 조합이 매우 유연해지며, subtree 단위 mutation/crossover가 가능해진다는 점이다. 논문은 이 검색 공간이 수십억 개의 후보를 포함할 수 있다고 설명한다.

검색 공간에 포함되는 unary operator는 매우 다양하다. 예를 들어 상수 0, 1, $x$, $-x$, $|x|$, $x^2$, $x^3$, $\sqrt{x}$, $e^x$, $e^{-x^2}$, $\log(1+e^x)$, $\log(|x+\epsilon|)$, $\sin(x)$, $\cos(x)$, $\tanh(x)$, $\arctan(x)$, $\sigma(x)$, $\mathrm{erf}(x)$, ReLU 등이 포함된다. binary operator도 덧셈, 곱셈, min, max 등으로 구성된다. 이 덕분에 매우 단순한 함수도, 매우 비직관적인 함수도 같은 표현 틀 안에서 탐색 가능하다.

### 3.2 Mutation

mutation은 activation tree의 한 노드를 무작위로 골라 검색 공간 내 다른 operator로 교체하는 방식이다. 논문 Figure 1의 예시에서는

$$
(\min{1,\cosh(x)})^3 \cdot \sigma(e^x + \arctan(x))
$$

같은 함수의 일부가 바뀌어

$$
(\min{1,\cosh(x)})^3 \cdot |e^x + \arctan(x)|
$$

처럼 변한다. 저자들의 해석은 mutation이 탐색 다양성을 보장해, 초기에 우연히 높은 성능을 얻은 activation이 population 전체를 너무 빨리 지배하지 않도록 막는다는 것이다.

### 3.3 Crossover

crossover에서는 두 parent activation function이 같은 깊이의 subtree를 교환해 새로운 child를 만든다. 깊이를 맞추는 제약을 두는 이유는 child가 여전히 같은 search space 안에 머물도록 보장하기 위해서다. 저자들은 crossover가 “좋은 activation의 특징”을 population에 퍼뜨려 random search보다 더 빠르게 좋은 함수를 찾게 해 준다고 설명한다.  

### 3.4 Search 전략

논문은 여러 탐색 전략을 비교한다.

* 작은 search space $S_1$ 에서의 exhaustive search
* 큰 search space $S_2$ 에서의 random search
* $S_2$ 에서 accuracy-based fitness를 쓰는 evolution
* $S_2$ 에서 loss-based fitness를 쓰는 evolution

특히 중요한 결과는 **loss-based evolution이 accuracy-based evolution이나 random search보다 더 빨리 더 좋은 activation을 찾는다**는 것이다. 논문 Figure 3은 각 generation에서 최고 validation accuracy가 어떻게 향상되는지를 보여 주며, loss-based fitness가 가장 효율적이었다고 정리한다.  

### 3.5 실험 설계

activation search는 WRN-28-10을 CIFAR-10에서 50 epoch 학습시키는 validation accuracy 또는 loss를 fitness로 삼아 진행된다. 이후 실제 성능을 더 정확히 보기 위해, 최상위 activation들을 WRN-28-10으로 **200 epoch씩 5회 반복 학습**하여 CIFAR-10과 CIFAR-100 test set에서 median accuracy를 보고한다. 논문은 CIFAR-100에 대해 별도 search는 하지 않았지만, CIFAR-10에서 진화한 함수들이 CIFAR-100으로 어느 정도 일반화된다는 점도 함께 분석한다.  

또한 큰 search 실험에서는 generation당 50개 activation을 평가하고, 그중 일부는 새로 무작위 생성하고 일부는 기존 population에 mutation/crossover를 적용해 만든다. 저자들은 10 generations의 evolution에 약 **2,000 GPU hours** 가 들었다고 설명한다.  

## 4. Experiments and Findings

### 4.1 Evolution이 실제로 ReLU를 넘는가

논문의 가장 중요한 결론은 **예, 그렇다**는 것이다. abstract에서 저자들은 evolved activation으로 ReLU를 대체하면 정확도가 통계적으로 유의하게 증가한다고 밝힌다. 그리고 실험 본문에서는 CIFAR-10 및 CIFAR-100에서 실제로 여러 evolved function이 ReLU를 능가했다고 보고한다.  

### 4.2 Loss-based evolution의 우위

논문은 loss-based fitness를 사용하는 evolution이 가장 효과적이었다고 반복해서 강조한다. Figure 3 설명에 따르면, loss-based evolution은 accuracy-based evolution이나 random search보다 더 빠르게 더 좋은 activation을 찾는다. 또한 Section 5.1에서는 $S_2$ 공간에서 loss-based evolution으로 찾은 상위 3개 함수 중 **하나는 CIFAR-10에서 ReLU와 Swish를 모두 이겼고**, **둘은 CIFAR-100에서 ReLU와 Swish를 모두 이겼다**고 정리한다.  

이는 중요한 해석을 낳는다. accuracy는 poor activation을 충분히 강하게 벌주지 못하지만, loss는 학습 과정의 질을 더 민감하게 반영하기 때문에 search signal로 더 적합하다는 것이다. 저자들도 accuracy-based fitness는 나쁜 activation에 대한 패널티가 약해서 덜 효과적이라고 설명한다.

### 4.3 Random search도 어느 정도 의미는 있다

흥미롭게도 random search가 완전히 무의미한 것은 아니다. 저자들은 random search로 나온 activation들이 매우 비직관적이지만, baseline을 넘지는 못해도 꽤 그럴듯한 정확도에 도달한다고 말한다. 다만 일부 함수는 $x=-\epsilon$ 에서 asymptote를 가져 학습이 끝까지 되지 않는 문제도 있었다. 즉, 거대한 함수 공간에서는 “그럭저럭 작동하는 함수”는 자주 발견되지만, **지속적으로 좋은 함수를 찾는 데에는 evolution이 훨씬 유리하다**는 메시지다.  

### 4.4 발견된 activation의 형태

논문은 손으로 쉽게 떠올리기 어려운 복잡한 activation도 높은 성능을 낼 수 있음을 보여 준다. 하지만 동시에, loss-based evolution이 찾은 최고의 함수 중에는 **simple, smooth, monotonic** 한 형태도 존재한다고 말한다. Figure 4는 generation 1의 최고 validation accuracy 92.2가 generation 10에서 93.9까지 올라감을 보여 주며, 최상위 activation이 ReLU와 달리 **전 구간에서 smooth** 하다는 점을 강조한다. 저자들은 이 smoothness가 성능 향상의 이유일 가능성을 시사한다.  

또한 paper summary figure에서는 진화가 찾은 대표 함수 중 하나로

$$
\sigma(x)\cdot \mathrm{erf}(x)
$$

를 언급하며, 이것이 단순하고도 고성능인 evolved activation의 예라고 설명한다.

### 4.5 Generalization과 specialization

논문은 activation function이 얼마나 task-specific한지도 따로 실험한다. CIFAR-10의 WRN-28-10에서 진화한 activation들을 CIFAR-100의 WRN-40-4로 옮겨 보았을 때, 일부 함수는 잘 전이되지만 일부는 특정 task에 강하게 특화되어 전이가 잘 되지 않았다. 저자들의 결론은 다음과 같다.

* **일반화도 가능하다**
* 하지만 **최고 성능은 architecture와 dataset별로 따로 진화시킬 때 나온다**

즉, activation metalearning의 진짜 힘은 범용 함수 하나를 찾는 것보다, **특정 문제에 맞는 함수를 specialization하는 것**에 있다는 것이다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 activation design을 본격적인 **search problem** 으로 재정의했다는 점이다. 기존 연구들이 predefined list에서 activation을 고르거나, RL로 후보를 생성하더라도 한두 개의 대표 함수를 중심으로 해석한 것과 달리, 이 논문은 **수십억 개 규모의 함수 공간**을 evolutionary operator로 탐색한다. 이는 activation을 수동 설계의 대상이 아니라 metalearning의 대상이라고 본 점에서 의미가 크다.

또한 실험 메시지가 분명하다. loss-based evolution이 random search보다 낫고, ReLU/Swish보다 더 좋은 activation을 찾을 수 있으며, architecture-dataset pair에 맞춘 specialization이 특히 강력하다는 점이 일관되게 나온다. 단순 아이디어 논문이 아니라, search strategy 비교까지 포함한 점도 강점이다.  

### 한계

한계도 분명하다. 첫째, 탐색 비용이 매우 크다. 논문이 직접 밝히듯 10 generations evolution에 약 2,000 GPU hours가 든다. 이는 activation 하나를 찾기 위한 비용으로는 상당히 크다.

둘째, 발견된 함수들 중 일부는 지나치게 복잡하거나 비직관적이며, 때로는 asymptote 때문에 학습 안정성 문제가 생긴다. 즉, search가 “좋은 수식”을 찾는다기보다 “작동하는 수식”을 찾는 경향도 있다. 이 때문에 해석 가능성과 안정성은 후속 필터링이 필요하다.  

셋째, 실험 범위는 주로 WRN + CIFAR에 집중되어 있다. 따라서 결과를 모든 architecture나 modality로 일반화하려면 추가 검증이 필요하다.

### 해석

비판적으로 보면, 이 논문의 진짜 가치는 “새 activation 하나를 발견했다”보다 **activation function 자체도 architecture search의 일부로 최적화될 수 있다**는 점을 보여 준 데 있다. 또한 범용 activation 하나만을 찾는 접근보다, 특정 task에 특화된 activation을 진화시키는 방향이 더 큰 성능 향상을 낼 수 있다는 점은 이후 NAS와 metalearning 전반에도 자연스럽게 연결된다.

## 6. Conclusion

이 논문은 activation function을 트리 기반 연산자 조합으로 표현하고, mutation, crossover, exhaustive search를 이용해 자동으로 최적화하는 **evolutionary activation search** 방법을 제안했다. 실험 결과, 진화로 찾은 activation은 CIFAR-10과 CIFAR-100의 Wide Residual Network에서 ReLU를 능가할 수 있었고, 특히 **loss-based evolution** 이 가장 효과적인 탐색 전략이었다. 또한 일부 activation은 다른 task로 전이되지만, 최고 성능은 architecture와 dataset에 특화해 진화시켰을 때 나온다는 점도 확인했다.

실무적으로 보면, 이 논문은 “ReLU냐 Swish냐” 같은 수동 비교를 넘어서, activation을 자동 설계 대상으로 보는 관점을 제시한다. 연구적으로는 activation function metalearning이 neural architecture search만큼이나 유망한 방향일 수 있음을 보여 준 작업이다.
