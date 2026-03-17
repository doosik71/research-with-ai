# Improving Deep Neural Network with Multiple Parametric Exponential Linear Units

## 1. Paper Overview

이 논문은 ELU와 PReLU의 장점을 하나의 activation으로 통합하려는 시도다. 저자들은 ReLU/PReLU 계열은 음수 구간에서 **linear family** 는 잘 표현하지만 non-linear family는 못 표현하고, ELU는 반대로 **non-linear family** 는 표현하지만 linear family는 못 표현한다는 “표현 공간의 간극”을 문제로 본다. 이를 해결하기 위해 **MPELU (Multiple Parametric Exponential Linear Unit)** 를 제안하고, 동시에 ELU/MPELU처럼 exponential linear unit을 쓰는 매우 깊은 네트워크를 안정적으로 학습시키기 위한 **새 weight initialization** 도 함께 제시한다. 마지막으로 이 둘을 이용한 deep MPELU ResNet을 구성해 CIFAR-10/100에서 당시 state-of-the-art 수준 성능을 보고한다.

이 문제가 중요한 이유는 activation function과 initialization이 단순 부품이 아니라, **표현력과 학습 가능 깊이** 를 동시에 제한하기 때문이다. 저자들은 특히 ELU가 좋은 학습 특성을 가지면서도 Batch Normalization과 잘 맞지 않을 수 있고, 또 ReLU용 He/MSRA initialization은 exponential unit에는 직접 맞지 않는다고 본다. 따라서 이 논문은 “새 activation”과 “그 activation을 깊은 네트워크에서 실제로 돌릴 수 있게 하는 initialization”을 한 세트로 제안하는 논문이다.

## 2. Core Idea

핵심 아이디어는 두 가지다.

첫째, **MPELU** 다. MPELU는 ELU의 음수 구간에 learnable parameter 두 개, $\alpha$ 와 $\beta$ 를 도입해 음수 영역의 **포화 크기** 와 **곡률/형태** 를 함께 조절한다. 이를 통해 같은 activation 안에서 ReLU, PReLU, ELU를 special case처럼 포함하도록 만든다. 저자들은 이것을 통해 음수 구간에서 linear와 non-linear 표현 공간을 모두 덮을 수 있다고 주장한다.  

둘째, **초기화 이론의 확장** 이다. 기존 Xavier는 선형 가정에, He/MSRA는 rectifier 계열에 맞춰져 있었다. 저자들은 MPELU를 0 근방에서 1차 Taylor expansion으로 근사하는 방식으로, ELU/MPELU 같은 exponential linear unit까지 포함하는 새 initialization을 유도한다. 이 initialization 덕분에 ELU/MPELU를 사용하는 매우 깊은 네트워크도 안정적으로 학습할 수 있다고 보인다.  

## 3. Detailed Method Explanation

### 3.1 MPELU의 정의

논문에서 MPELU는 다음과 같이 정의된다.

$$
f(y_i)=
\begin{cases}
y_i, & y_i > 0 \
\alpha_c (e^{\beta_c y_i}-1), & y_i \le 0
\end{cases}
$$

여기서 $\beta_c > 0$ 이고, $\alpha_c$, $\beta_c$ 는 채널별(channel-wise) 또는 채널 공유(channel-shared) learnable parameter다. 저자들은 전 실험에서 channel-wise 버전을 사용했다고 밝힌다. 추가 파라미터 수는 전체 채널 수의 두 배 이하라서, 전체 weight 수에 비하면 매우 작다고 주장한다.  

### 3.2 ReLU, PReLU, ELU를 어떻게 포괄하는가

MPELU의 중요한 성질은 기존 activation들이 special case로 들어온다는 점이다.

* $\alpha = 0$ 이면 **ReLU**
* 큰 $\alpha$ 와 작은 $\beta$ 조합이면 **PReLU 근사**
* $\alpha, \beta = 1$ 이면 **ELU**

즉, MPELU는 단순히 ELU를 매개변수화한 정도가 아니라, **rectified family와 exponential family를 잇는 일반형** 으로 설계되었다. 이게 논문 제목의 “multiple parametric”이 의미하는 바다.

### 3.3 Batch Normalization과의 관계

저자들은 MPELU가

$$
\text{MPELU} = \widetilde{\text{ELU}}[\text{PReLU}(x)]
$$

또는 BN을 포함하면

$$
\text{MPELU} = \widetilde{\text{ELU}}{\text{PReLU}[\text{BN}(x)]}
$$

처럼 해석될 수 있다고 설명한다. 이 해석의 포인트는 BN의 출력이 먼저 PReLU를 지나고 그 다음 learnable ELU를 지난다는 점이다. 저자들은 이것이 **ELU 단독 사용 시 BN과 충돌하는 문제를 완화** 하면서, PReLU와 ELU의 장점을 함께 취하게 해 준다고 본다.

### 3.4 초기화 방법

초기화 쪽에서 논문은 기존 Glorot/Xavier와 He/MSRA를 출발점으로 삼되, ELU/MPELU에는 맞지 않는다고 본다. 그래서 MPELU를 0 근방에서 1차 Taylor approximation으로 다루고, 거기서 분산 보존 조건을 유도해 **ELU/MPELU용 analytic initialization** 을 제안한다. 저자들은 이것이 ELU와 MPELU뿐 아니라 non-convex activation에도 적용 가능하다고 주장한다.  

이 초기화가 중요한 이유는, 논문 주장대로라면 Gaussian 초기화나 LSUV를 쓸 때는 깊은 ELU 네트워크가 폭주하거나 멈출 수 있지만, 제안 초기화는 실제로 매우 깊은 ELU/MPELU 네트워크를 수렴시킨다는 데 있다.

## 4. Experiments and Findings

### 4.1 CIFAR-10에서 NIN 실험

논문은 먼저 CIFAR-10 위의 NIN(Network in Network)에서 MPELU를 비교한다. 이 실험의 목적은 **ELU에 learnable parameter를 도입하는 것이 실제로 이득이 있는가** 를 보는 것이다. 저자들의 요약에 따르면, MPELU는 ReLU/PReLU/ELU 대비 더 좋은 classification performance와 convergence를 보였다.  

### 4.2 ImageNet 실험

이후 더 깊은 네트워크와 더 큰 데이터셋인 ImageNet 2012에서도 비슷한 경향을 보였다고 한다. 논문은 MPELU가 CIFAR-10과 ImageNet 2012에서 모두 **더 나은 분류 성능과 더 좋은 수렴 특성** 을 보였다고 직접 말한다. 또 carefully optimized Caffe implementation에서는 실제 running time이 PReLU보다 약간 느린 수준으로 억제된다고 주장한다.  

### 4.3 Initialization 검증

초기화 실험은 이 논문의 또 다른 핵심이다. 논문은 30-layer, 52-layer ELU 네트워크에서 **LSUV는 BN 없이 몇 iteration 만에 폭주** 하지만, 제안 초기화는 수렴시킨다고 보고한다. 또한 15-layer ImageNet 실험에서도 top-1 error를 작지만 일관되게 낮춘다고 정리한다. 즉, 이 initialization은 단순 theoretical extension이 아니라 실제로 **매우 깊은 exponential-unit network를 가능하게 하는 도구** 로 제시된다.  

### 4.4 Deep MPELU ResNet

가장 강한 결과는 CIFAR의 deep residual network 실험이다. 논문은 MPELU ResNet이 ReLU/ELU 기반보다 더 넓은 solution space를 가지며 더 좋은 결과를 낸다고 해석한다. 최종적으로 **MPELU nopre ResNet-1001** 이 CIFAR-10에서 **3.57%**, CIFAR-100에서 **18.81%** test error를 기록했다고 명시하며, 이는 당시 original Pre-ResNet보다 더 낮은 수치라고 주장한다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **activation과 initialization을 함께 제안했다**는 점이다. 많은 activation 논문이 “좋은 함수 모양”만 말하고 끝나는데, 이 논문은 그 activation이 **깊은 네트워크에서 실제로 학습되도록 만드는 initialization** 까지 포함한다. 그래서 제안이 훨씬 완결된 형태다.

또한 MPELU는 ReLU/PReLU/ELU를 하나의 family로 묶으면서, 음수 구간의 linear/non-linear space를 모두 다룰 수 있다는 점에서 설계 철학이 분명하다. BN과 ELU의 궁합 문제를 완화하려 한 점도 실용적이다.

### 한계

한계도 있다. MPELU는 ELU보다 파라미터와 계산이 늘고, 지수 연산이 필요하다. 논문은 코드 최적화로 PReLU에 가까운 속도를 낼 수 있다고 말하지만, ReLU급 단순성은 아니다.

또 하나의 한계는, 이 논문의 strongest claim이 주로 **CNN/CIFAR/ImageNet** 문맥에 묶여 있다는 점이다. 즉, 이후 다른 modality나 Transformer 계열까지 일반화되는지는 이 논문 자체만으로는 말하기 어렵다.

### 해석

비판적으로 보면, 이 논문의 더 큰 의미는 “MPELU가 최고 activation”이라기보다, **activation function family를 연속적 공간으로 일반화하고, 그에 맞는 initialization까지 함께 설계해야 한다**는 점을 보여 준 데 있다. 또한 ELU류 activation의 약점이 단순 함수 형태보다 **학습 가능성(initialization, BN compatibility)** 에도 있음을 잘 드러낸다.

## 6. Conclusion

이 논문은 **MPELU (Multiple Parametric Exponential Linear Unit)** 를 제안해 ReLU/PReLU의 rectified family와 ELU의 exponential family를 하나의 일반형 안에서 통합했다. MPELU는 learnable $\alpha, \beta$ 를 통해 음수 구간의 포화 크기와 곡률을 조절하며, ReLU, PReLU, ELU를 special case처럼 포함한다. 동시에 저자들은 ELU/MPELU를 위한 **analytic initialization** 을 제안해, 기존 rectifier용 이론을 exponential linear unit까지 확장했다.  

실험적으로 논문은 NIN부터 1001-layer ResNet까지 폭넓게 검증했고, MPELU가 classification accuracy와 convergence를 개선하며, 제안 초기화가 매우 깊은 ELU/MPELU 네트워크를 실제로 수렴시키고 generalization도 향상시킨다고 주장한다. 최종적으로 deep MPELU ResNet은 CIFAR-10/100에서 state-of-the-art 수준 성능을 보고했다.  
