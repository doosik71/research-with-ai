# Maxout Networks

## 1. Paper Overview

이 논문은 dropout이 잘 작동하도록 **모델 자체를 dropout 친화적으로 설계할 수 있는가**라는 질문에서 출발한다. 저자들은 dropout을 단순한 regularization trick으로 보는 대신, **매우 큰 ensemble을 parameter sharing 하에 근사적으로 학습하고 평균내는 방법**으로 해석한다. 그런데 깊은 신경망에서는 이 근사적 model averaging이 항상 잘 맞는 것이 아니므로, dropout의 장점을 더 잘 살릴 수 있는 activation/unit이 필요하다고 본다. 이를 위해 제안된 것이 **maxout unit** 이고, 저자들은 maxout + dropout 조합이 optimization과 model averaging 양쪽 모두를 개선한다고 주장한다. 실제로 MNIST, CIFAR-10, CIFAR-100, SVHN에서 당시 state-of-the-art classification 성능을 보고했다.

이 문제가 중요한 이유는 당시 딥러닝의 강력한 성능 향상 요인 중 하나가 dropout이었기 때문이다. 하지만 기존 activation인 sigmoid, tanh, ReLU는 dropout이 이상적으로 작동하는 조건과 꼭 잘 맞는다고 보기 어려웠다. 논문은 “dropout을 아무 모델에나 붙이는 것”보다, **dropout이 ensemble averaging을 더 정확히 수행하도록 돕는 구조를 직접 설계하는 편이 더 낫다**는 관점을 제시한다. 이 점에서 maxout은 단순 activation 제안이 아니라, dropout 시대의 모델 설계 철학을 바꾼 논문이다.

## 2. Core Idea

핵심 아이디어는 매우 간단하다. hidden unit 하나가 하나의 affine response에 고정된 activation을 적용하는 대신, **여러 개의 affine response 중 최대값을 출력**하도록 만들자는 것이다. 논문에서 maxout hidden unit은 다음과 같이 정의된다.

$$
h_i(x)=\max_{j\in [1,k]} z_{ij}
$$

여기서

$$
z_{ij}=x^T W_{\cdots ij}+b_{ij}
$$

이다. 즉, 뉴런 하나가 단일 선형 변환 뒤에 ReLU나 tanh를 거는 것이 아니라, **$k$개의 선형 응답 중 가장 큰 것을 선택**한다. 이 구조 덕분에 maxout unit은 입력에 따라 서로 다른 선형 조각(piecewise linear branch)을 활성화할 수 있다.

이 설계가 중요한 이유는 두 가지다.

첫째, maxout은 **임의의 convex function에 대한 piecewise linear approximation** 으로 해석할 수 있다. 논문 Figure 1도 maxout이 rectified linear, absolute value rectifier, quadratic 비슷한 곡선까지 구현할 수 있음을 보여 준다. 즉, activation을 고정 함수로 쓰지 않고 **뉴런이 activation 자체를 학습**하게 만든다.

둘째, dropout과 잘 맞는다. 저자들은 dropout이 잘 작동하려면 parameter update가 비교적 크고, 각 update가 사실상 서로 다른 sub-model을 학습하는 것처럼 동작해야 한다고 본다. maxout은 gradient 흐름과 표현 유연성 측면에서 이런 regime에 잘 맞는 구조라고 주장한다. 즉, maxout은 “dropout의 자연스러운 동반자”로 설계되었다.

## 3. Detailed Method Explanation

### 3.1 Dropout에 대한 논문의 해석

논문은 dropout을 다음처럼 본다. 입력 $v$ 와 hidden layers $\mathbf{h}$ 를 가진 feed-forward network에서, dropout은 binary mask $\mu$ 를 통해 입력과 hidden variable의 일부를 제거한 **sub-model family** 를 학습한다. 각 training step에서는 다른 $\mu$ 가 샘플링되고, 그때마다 조금씩 다른 모델이 업데이트된다. 이 점에서 dropout은 bagging과 비슷하지만, 각 모델이 parameter를 공유한다는 점이 다르다.

또한 softmax classifier에서는 geometric mean averaging이 간단히 구현될 수 있고, single-layer softmax의 경우 weight를 2로 나눈 full model이 dropout ensemble의 geometric mean과 정확히 대응된다고 설명한다. 그러나 깊은 모델에서는 이 근사가 정확하지 않으므로, **깊은 구조에서 dropout averaging approximation error를 줄이는 모델 설계**가 중요하다고 본다. maxout은 바로 이 목적을 가진다.

### 3.2 Maxout unit의 구조

maxout hidden layer는 입력 $x\in \mathbb{R}^d$ 에 대해, 각 unit $i$ 마다 $k$개의 affine response를 계산하고 그 최대를 취한다.

$$
h_i(x)=\max_{j\in [1,k]} z_{ij}, \qquad
z_{ij}=x^T W_{\cdots ij}+b_{ij}
$$

여기서 $W\in \mathbb{R}^{d\times m\times k}$, $b\in \mathbb{R}^{m\times k}$ 이다. $m$은 maxout unit 수, $k$는 한 unit 안의 linear piece 수라고 볼 수 있다. convolutional network에서는 공간 위치별이 아니라 **채널 방향으로 여러 affine feature map을 만든 뒤 그 max를 취하는 방식**으로 구현할 수 있다.

중요한 구현 디테일 하나는 dropout mask를 max operator 입력 각각에 직접 거는 것이 아니라, **weights와 곱하기 직전에 element-wise multiplication** 을 수행한다는 점이다. 논문은 max operator의 입력 자체를 drop하지 않는다고 명시한다. 이는 dropout과 max selection이 불안정하게 상호작용하는 것을 피하기 위한 선택으로 볼 수 있다.

### 3.3 Maxout의 표현력

논문은 maxout unit 하나가 **임의의 convex function의 piecewise linear approximation** 이라고 설명한다. 더 나아가, maxout network 전체는 universal approximator가 될 수 있음을 보인다. snippet에 나온 Figure 3 설명에 따르면, **두 개의 maxout unit만 포함한 MLP도 임의의 연속함수를 임의 정밀도로 근사**할 수 있다. 직관은 다음과 같다.

* 하나의 maxout unit은 convex piecewise linear function을 표현한다.
* 다른 하나도 또 다른 convex function을 표현한다.
* 마지막 layer에서 이 둘의 차이를 취하면 non-convex continuous function까지 근사할 수 있다.

이 부분이 ReLU와의 큰 차이다. ReLU도 깊은 네트워크 안에서는 매우 강력하지만, 개별 unit 차원에서 보면 maxout이 훨씬 더 유연한 activation family를 가진다. 논문은 이 점을 “네트워크는 hidden units 간 관계뿐 아니라 각 hidden unit의 activation function 자체도 학습한다”라고 요약한다.

### 3.4 Sparse representation과의 관계

흥미롭게도 논문은 maxout이 전통적인 activation 설계 원칙들을 일부러 버렸다고 말한다. 대표적으로 **activation 자체는 sparse하지 않다**. Figure 2 캡션도 “The activations of maxout units are not sparse”라고 직접 말한다. 대신 저자들은 gradient는 매우 sparse하고, dropout이 training 중 effective representation을 인위적으로 희소화한다고 본다. 또한 maxout은 bounded하지 않을 수도 있고, local linearity가 거의 모든 구간에서 유지된다. 그럼에도 불구하고 실제로는 robust하고, dropout과 함께 매우 잘 학습된다고 주장한다.

이 부분은 당시 activation 설계 상식과는 꽤 다르다. 즉, “좋은 activation은 sparse해야 한다”, “bounded해야 한다”, “특정 curvature를 가져야 한다” 같은 직관이 항상 필요한 것은 아니며, **dropout과 optimization 관점에서 더 중요한 기준이 따로 있을 수 있다**는 메시지를 준다.

## 4. Experiments and Findings

논문 초록은 maxout과 dropout을 함께 사용해 **MNIST, CIFAR-10, CIFAR-100, SVHN** 네 benchmark에서 state-of-the-art classification 성능을 달성했다고 직접 말한다. 이 자체가 논문의 가장 강한 실험적 결론이다. 즉, maxout은 단순 이론적 activation이 아니라 실제 대형 benchmark에서도 통했다.

실험 메시지는 크게 두 가지로 정리된다.

첫째, **optimization이 더 잘 된다**는 점이다. 저자들은 maxout이 dropout 하에서 gradient 흐름과 parameter update가 더 잘 맞도록 설계되었고, 실제로 학습이 쉽고 robust하다고 강조한다. 단순히 function class가 커진 것뿐 아니라, dropout이 요구하는 “큰 step, 여러 sub-model” 학습 regime에 더 적합한 activation이라는 것이다.

둘째, **dropout의 approximate model averaging이 더 잘 작동한다**는 점이다. 기존 activation에서는 dropout ensemble average를 full model로 근사할 때 오차가 커질 수 있는데, maxout은 이 approximation error를 줄이는 방향으로 설계되었다고 주장한다. 따라서 maxout의 성능 향상은 단순 capacity 증가가 아니라, **dropout과의 구조적 궁합** 에서도 나온다.

현재 대화에 제공된 본문은 중간에서 잘려 있어, 각 benchmark별 정확한 test error 숫자와 개별 architecture 설정, 데이터 augmentation 세부값까지는 여기서 완전히 재구성할 수 없다. 하지만 논문이 적어도 초록과 앞부분에서 분명히 밝히는 핵심은 다음이다.

* maxout + dropout은 네 개 benchmark에서 당시 최고 성능을 냈다.
* 이 성능은 optimization 개선과 model averaging 개선 두 효과의 결합으로 해석된다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **dropout을 중심으로 모델을 다시 설계했다**는 점이다. 많은 regularization 기법은 기존 모델 위에 얹는 부가기법처럼 사용되지만, 이 논문은 아예 “dropout이 가장 잘 작동하게 하려면 어떤 unit를 써야 하는가?”를 묻는다. 즉, training procedure와 model family를 함께 설계한 점이 매우 강하다.

또 다른 강점은 **표현력과 실용성의 균형**이다. maxout은 수학적으로는 convex piecewise linear approximation, 더 나아가 universal approximation까지 가능하고, 실제로는 MLP와 CNN에 바로 끼워 넣어 쓸 수 있다. 복잡한 discrete search나 별도의 outer-loop optimization 없이, 표준 backpropagation으로 학습 가능하다는 점도 중요하다.

### 한계

한계도 분명하다. 첫째, maxout은 unit당 여러 affine response를 유지해야 하므로 **파라미터와 계산량이 증가**한다. ReLU 하나로 끝나는 구조보다 무겁다. 둘째, representation 자체는 sparse하지 않아서, 희소성 자체를 중요한 inductive bias로 보는 경우에는 덜 매력적일 수 있다. 논문도 이것을 인정한다.

또한 이후 역사적으로 보면, maxout은 개념적으로 매우 중요했지만 ReLU만큼 범용 표준이 되지는 못했다. 이는 성능 자체보다도 **계산/메모리 비용** 과 단순성의 문제와 관련 있다고 해석할 수 있다. 이 부분은 논문 이후의 맥락을 반영한 해석이다.

### 해석

비판적으로 보면, 이 논문의 가장 큰 유산은 “maxout이 SOTA였다”보다도 **activation function을 고정 함수가 아니라 학습 가능한 function family로 봤다**는 점이다. maxout unit은 뉴런이 activation shape 자체를 data-driven하게 선택하게 만든다. 이 발상은 이후 APL, SReLU, activation search, Dynamic ReLU 같은 많은 연구의 선구적 형태로 볼 수 있다.

또한 dropout과 model family의 궁합을 구조적으로 고민했다는 점도 중요하다. 즉, regularization은 모델에 독립적인 외부 도구가 아니라, **어떤 activation/architecture와 결합되느냐에 따라 훨씬 더 강해질 수 있다**는 교훈을 준다.

## 6. Conclusion

이 논문은 **maxout unit** 을 제안하며, dropout이 단순 regularizer가 아니라 ensemble-like learning과 model averaging 기법이라는 관점에서, 그것과 가장 잘 맞는 activation family를 설계하려 했다. maxout unit은

$$
h_i(x)=\max_{j\in [1,k]} z_{ij}, \qquad
z_{ij}=x^T W_{\cdots ij}+b_{ij}
$$

형태로 정의되며, 개별 뉴런 수준에서 convex piecewise linear function을 표현하고, 네트워크 전체로는 연속함수의 보편 근사까지 가능하다. 동시에 dropout 하에서 optimization과 approximate model averaging을 개선하도록 설계되었다.

실험적으로는 MNIST, CIFAR-10, CIFAR-100, SVHN에서 당시 최고 수준 분류 성능을 보고했다. 따라서 이 논문은 단지 “새 activation 하나”를 제안한 것이 아니라, **dropout 시대의 딥러닝 모델을 어떻게 설계해야 하는가**에 대한 강한 답을 준 논문으로 볼 수 있다. 그리고 더 넓게 보면, activation function을 사람이 고정해서 넣는 것이 아니라 **학습 가능한 function family** 로 다루는 방향을 연 중요한 전환점이기도 하다.
