# Learning Activation Functions to Improve Deep Neural Networks

## 1. Paper Overview

이 논문은 딥러닝에서 보통 고정되어 있는 activation function을, 각 뉴런마다 **학습 가능한 형태로 바꾸면 성능이 실제로 얼마나 좋아질 수 있는가**를 다룬다. 저자들은 당시 널리 쓰이던 sigmoid, tanh, ReLU, maxout 같은 activation들이 충분히 큰 네트워크에서는 복잡한 함수를 근사할 수 있지만, **유한 크기 네트워크에서는 activation의 형태가 학습 동역학과 표현력 모두에 큰 영향을 준다**고 본다. 그래서 미리 정해 둔 activation을 쓰는 대신, 각 뉴런이 자신만의 piecewise linear activation을 gradient descent로 직접 학습하도록 하는 **APL(Adaptive Piecewise Linear) unit** 을 제안한다. 논문 초록에 따르면 이 방법은 CIFAR-10에서 7.51%, CIFAR-100에서 30.83% 오류율을 기록했고, Higgs boson decay benchmark에서도 state-of-the-art를 달성했다.  

이 문제가 중요한 이유는 분명하다. activation function은 단순히 “비선형성을 넣는 함수”가 아니라, 실제로는 네트워크가 어떤 형태의 feature를 더 쉽게 표현하는지, 깊은 네트워크가 얼마나 안정적으로 학습되는지, 그리고 같은 파라미터 수 안에서 어느 정도의 효율을 낼 수 있는지를 좌우한다. 저자들은 maxout처럼 표현력이 큰 activation도 존재하지만, 그 공간은 여전히 충분히 탐색되지 않았다고 지적하며, activation 자체를 **뉴런별로 적응적으로 학습하는 방향**을 제안한다.

## 2. Core Idea

논문의 핵심 아이디어는 간단하면서도 강력하다. 각 뉴런의 activation을 하나의 고정 함수로 두지 않고, **ReLU에 여러 개의 hinge-shaped 보정항을 더하는 형태**로 만들자는 것이다. 그 결과 activation은 piecewise linear가 되며, 각 뉴런은 자신의 입력 분포와 역할에 맞춰 서로 다른 함수 모양을 학습할 수 있다. 저자들은 이 방식을 통해 단일 뉴런이 **convex뿐 아니라 non-convex 함수까지 표현**할 수 있다고 강조한다. 이는 maxout이 단일 유닛 수준에서는 주로 convex 함수를 근사하는 것과 대비된다.  

더 중요한 점은 이 activation이 별도의 진화 알고리즘이나 discrete search 없이, **일반적인 backpropagation과 gradient descent로 바로 학습된다**는 것이다. 즉, activation function search를 별도의 바깥 최적화 문제로 두지 않고, 신경망 파라미터 학습 안으로 자연스럽게 포함시켰다. 이것이 이 논문의 가장 큰 실용적 장점이다.

## 3. Detailed Method Explanation

### 3.1 APL unit의 정의

논문이 제안한 APL unit는 다음과 같이 정의된다.

$$
h_i(x)=\max(0,x)+\sum_{s=1}^{S} a_i^s \max(0,-x+b_i^s)
$$

여기서 $S$ 는 미리 정하는 하이퍼파라미터인 hinge 개수이고, $a_i^s$ 와 $b_i^s$ 는 뉴런 $i$ 에 대해 학습되는 파라미터다. 직관적으로 보면 첫 항 $\max(0,x)$ 는 기본 ReLU이고, 뒤의 항들은 음의 방향에서 여러 개의 hinge를 추가해 activation의 기울기와 꺾임 위치를 조절한다. $a_i^s$ 는 각 구간의 slope를, $b_i^s$ 는 hinge의 위치를 정한다.

이 구조는 매우 유연하다. 기본 ReLU를 중심으로 시작하지만, 학습이 진행되면 각 뉴런은 자신의 역할에 맞게 다른 activation shape를 갖게 된다. 논문 그림 설명에서도 실제로 학습된 함수들이 leaky ReLU처럼 보이기도 하고, 명확히 non-convex한 모양을 띠기도 한다고 설명한다.  

### 3.2 파라미터 수와 계산 부담

APL unit가 추가하는 파라미터 수는 전체 hidden unit 수를 $M$ 이라고 할 때 **$2SM$** 이다. 저자들은 이 수가 전형적인 딥네트워크의 전체 weight 수와 비교하면 작다고 말한다. 즉, activation을 뉴런별로 학습 가능하게 만들면서도 파라미터 폭증은 피했다는 주장이다.

### 3.3 표현력: 왜 maxout과 다른가

논문은 APL이 maxout과 비슷한 piecewise linear family에 속하지만, 중요한 차이가 있다고 설명한다. maxout은 단일 유닛으로는 주로 convex function을 근사하지만, APL은 단일 유닛만으로도 **non-convex 함수**를 표현할 수 있다. 또한 정리(Theorem 1)를 통해, 두 가지 비대칭 조건만 만족하면 임의의 연속 piecewise-linear 함수 $g(x)$ 를 APL 형식으로 표현할 수 있음을 보인다. 구체적으로는 큰 양수 영역에서 $g(x)=x$ 이고, 충분히 작은 영역에서 선형 꼬리를 갖는 continuous piecewise-linear 함수라면 APL로 쓸 수 있다고 주장한다.  

이 정리는 논문 해석에서 중요하다. APL은 “약간 더 유연한 ReLU” 수준이 아니라, 실제로 상당히 넓은 함수 공간을 뉴런 단위에서 포괄한다. 동시에 maxout network가 특정 weight-tying을 쓰면 APL을 재현할 수 있지만, APL은 더 직접적이고 파라미터 효율적인 형태라고 논문은 해석한다.

### 3.4 학습 방식과 정규화

APL 파라미터는 일반적인 신경망 파라미터와 함께 gradient descent로 최적화된다. 논문은 과적합 방지를 위해 activation 파라미터에도 regularization을 넣었고, 실험에서는 대체로 **$S=2$** 를 기본값으로 사용했다. 이는 표현력을 충분히 늘리면서도 추가 파라미터를 과도하게 늘리지 않기 위한 실용적 선택으로 보인다. Higgs 실험에서도 APL units에는 $S=2$ 를 사용했다고 직접 명시한다.  

## 4. Experiments and Findings

### 4.1 CIFAR-10 / CIFAR-100

논문은 CIFAR-10과 CIFAR-100에서 표준 CNN과 NIN(Network in Network) 아키텍처를 사용해 APL을 평가했다. 저자들의 요약에 따르면, **APL은 baseline 대비 CIFAR-10에서 1% 이상, CIFAR-100에서 거의 3% 가까이 error를 줄였다**. 논문은 이를 상대적으로 각각 **9.4%와 7.5% error reduction** 으로 해석한다. 또한 NIN + APL 설정에서 data augmentation을 사용할 경우, 자신들이 보고한 결과가 당시 CIFAR-10과 CIFAR-100에서 최고 성능이라고 주장한다.  

초록에 명시된 최종 수치도 강하다. CIFAR-10 오류율은 **7.51%**, CIFAR-100 오류율은 **30.83%** 다. 이 수치들은 이 논문이 단순히 activation visualization paper가 아니라, 실제 benchmark 성능 향상까지 이끌어 냈음을 보여 준다.

또한 저자들은 leaky ReLU와도 직접 비교해, **APL units가 일관되게 leaky ReLU보다 낫다**고 말한다. 즉, 단순히 음수 기울기를 조금 열어 주는 정도가 아니라, 뉴런별로 activation shape 자체를 튜닝하는 것이 더 큰 이득을 준다는 것이 이 논문의 실험적 메시지다.

### 4.2 Higgs boson decay benchmark

논문은 고에너지 물리의 Higgs boson decay benchmark에도 APL을 적용했다. 이 실험에서는 기존 Baldi et al.의 architecture를 그대로 사용하되, 상위 두 hidden layer에 dropout을 넣고 APL에는 역시 **$S=2$** 를 사용했다. 결과적으로 저자들은 **APL을 사용한 단일 네트워크가 dropout baseline뿐 아니라 기존 5개 네트워크 앙상블보다도 더 좋은 결과를 달성했다**고 설명한다. 이는 APL이 이미지 분류뿐 아니라 tabular/scientific classification에서도 효과적일 수 있음을 보여 준다.  

### 4.3 하이퍼파라미터 S의 영향

논문은 hinge 개수 $S$ 의 효과도 살펴본다. visible result에 따르면 $S$ 를 크게 늘린다고 항상 좋아지는 것은 아니며, 실험적으로는 작은 $S$ 에서도 충분한 성능 향상이 나타난다. 즉, APL의 장점은 “매우 복잡한 activation”이 아니라, **적당한 수의 학습 가능한 꺾임만으로도 충분히 표현력 향상을 얻는다**는 데 있다. CIFAR/Higgs 실험 전반에서 저자들이 $S=2$ 를 주로 사용한 것도 이 점과 맞닿아 있다.  

### 4.4 시각화 분석

논문 후반의 시각화 섹션은 꽤 흥미롭다. CIFAR-100과 Higgs 실험에서 학습된 activation들을 그려 보면, 뉴런마다 activation 모양이 크게 달라지고, 일부는 leaky ReLU와 비슷하지만 일부는 명백히 non-convex하다. 또 higher layer로 갈수록 activation variance가 감소하는 경향도 관찰된다고 설명한다. 즉, 네트워크는 실제로 각 층과 각 뉴런의 역할에 맞춰 activation shape를 분화시키고 있다. 이는 APL이 단순히 “추가 파라미터만 넣은 트릭”이 아니라, 실제로 서로 다른 비선형성을 학습하고 있음을 보여 주는 정성적 증거다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **activation function을 뉴런별로 직접 학습한다**는 발상을, 복잡한 외부 탐색 없이 표준 gradient descent 안에 자연스럽게 넣었다는 점이다. maxout처럼 큰 표현력을 갖는 activation과 비교해도, APL은 훨씬 적은 추가 파라미터로 non-convex 함수까지 다룰 수 있다. 또한 실제로 CIFAR-10, CIFAR-100, Higgs benchmark에서 성능 향상을 보였기 때문에, 단순 아이디어 차원을 넘어 실험적 설득력도 충분하다.  

또 하나의 강점은 이 논문이 activation research의 방향을 바꾸었다는 점이다. activation을 고정 함수로 둘 것이 아니라, **데이터와 층의 역할에 맞게 학습되는 함수 family** 로 볼 수 있다는 시각을 제시했다. 이후 SReLU, activation ensemble, Dynamic ReLU 등 많은 후속 연구와도 자연스럽게 이어진다. 이 해석은 논문 내용과 이후 activation literature 흐름을 연결한 것이다.

### 한계

한계도 분명하다. 논문이 제시한 Theorem 1은 강력하지만, APL이 표현 가능한 함수 family에는 여전히 양끝 꼬리 조건이 붙는다. 즉, 임의의 모든 함수가 아니라, **연속 piecewise-linear 함수 중에서도 큰 양수 쪽에서 identity, 큰 음수 쪽에서 선형 꼬리**를 갖는 형태를 전제로 한다. 물론 실제 neural network 안에서는 다음 선형층이 이 제약을 어느 정도 상쇄한다고 논문은 설명하지만, 이건 완전히 자유로운 activation family는 아니라는 뜻이다.

또한 APL은 rightmost branch가 사실상 $g(x)=x$ 로 고정되는 구조적 제약이 있다. 이후 SReLU 논문이 APL의 한계로 바로 이 점을 지적하며, 큰 양수 영역에서 adaptive scaling이 부족하다고 비판한 것도 의미심장하다. 즉, APL은 당시로서는 강력했지만, activation family 전체 관점에서 보면 **중요한 중간 단계**에 가깝다.

### 해석

비판적으로 보면, 이 논문의 진짜 가치는 “7.51%, 30.83%” 같은 benchmark 수치만이 아니다. 더 중요한 메시지는 **activation function도 학습 대상이 될 수 있다**는 점을 실제로 보여 준 것이다. 당시에는 weight만 학습하고 activation은 사람이 고르는 것이 자연스러웠는데, 이 논문은 그 고정관념을 깼다. 또한 개별 뉴런이 서로 다른 activation을 학습한다는 아이디어는 네트워크 내부 representation이 훨씬 이질적이고 역할 분화되어 있을 수 있음을 시사한다. 이는 이후 adaptive/non-static activation 연구의 출발점으로 읽을 수 있다.

## 6. Conclusion

이 논문은 **APL(Adaptive Piecewise Linear) unit** 을 제안해, activation function을 뉴런별로 gradient descent로 직접 학습할 수 있음을 보였다. 핵심 수식은

$$
h_i(x)=\max(0,x)+\sum_{s=1}^{S} a_i^s \max(0,-x+b_i^s)
$$

이며, 이 구조는 적은 수의 추가 파라미터만으로 convex와 non-convex piecewise linear function을 모두 표현할 수 있다. 이론적으로는 특정 조건을 만족하는 임의의 연속 piecewise-linear 함수를 표현 가능하다는 정리를 제시했고, 실험적으로는 CIFAR-10, CIFAR-100, Higgs boson decay benchmark에서 당시 최고 수준 결과를 보고했다.  

정리하면, 이 논문은 “고정 activation을 더 좋은 고정 activation으로 바꾸자”가 아니라, **activation 자체를 학습 가능하게 만들자**는 방향을 처음으로 강하게 밀어붙인 대표 논문이다. 이후 더 유연한 activation들이 등장했지만, APL은 adaptive activation 연구의 중요한 출발점으로 평가할 가치가 충분하다.
