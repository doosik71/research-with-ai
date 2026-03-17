# Activation Ensembles for Deep Neural Networks

## 1. Paper Overview

이 논문은 딥러닝에서 activation function을 하나로 고정해 두는 관행에 문제를 제기한다. 저자들은 ReLU, sigmoid, tanh, inverse absolute value, softplus, exponential linear unit 같은 서로 다른 activation이 각기 다른 특징을 포착하는데도, 실제 모델 설계에서는 보통 한두 개만 선택해 전 네트워크에 적용한다는 점에 주목한다. 그래서 “어떤 activation이 가장 좋은가?”를 사람이 미리 정하는 대신, **각 neuron이 여러 activation을 동시에 후보로 두고 학습 과정에서 그 조합을 스스로 선택하게 하자**는 activation ensemble 개념을 제안한다. 이 방법은 neuron 단위에서 activation의 convex combination을 학습하며, MNIST, ISOLET, CIFAR-100, STL-10에서 기존 고정 activation보다 더 나은 결과를 보고한다.  

이 문제가 중요한 이유는 activation function 선택이 종종 경험적 trial-and-error에 의존하기 때문이다. 논문은 이 선택을 hyperparameter search의 대상이 아니라 **학습 가능한 구성요소**로 바꾸려 한다. 또한 같은 데이터셋이라도 FFN과 CNN에서 선호 activation이 다르고, 같은 네트워크라도 층마다 선호 activation이 달라질 수 있다고 주장한다. 즉, 이 논문의 문제의식은 “하나의 activation이 항상 최선인가?”이며, 답은 명확히 “아니다”이다.  

## 2. Core Idea

핵심 아이디어는 간단하지만 꽤 강력하다. 각 neuron에 대해 하나의 activation $f(x)$만 두지 않고, 여러 activation 후보 $f^1, f^2, \dots, f^m$를 둔 뒤, 이들을 정규화하고 가중합하여 최종 activation으로 사용한다. 이때 가중치 역할을 하는 것이 $\alpha$이며, 각 neuron마다 각 activation 함수별로 별도의 $\alpha$를 가진다. 따라서 어떤 neuron은 ReLU를 주로 쓰고, 어떤 neuron은 tanh나 inverse absolute value를 더 크게 쓰는 식으로 학습된다. 저자들은 이를 “activation ensemble”이라 부른다.

하지만 단순 가중합만으로는 문제가 생긴다. activation 함수들은 출력 범위와 스케일이 서로 다르기 때문에, magnitude가 큰 함수가 다른 함수들을 압도해 버린다. 이를 해결하기 위해 저자들은 각 activation 출력을 **상대적으로 comparable한 범위로 정규화**하고, 추가적인 offset 변수 $\eta$, $\delta$를 도입해 정규화 구간을 조정한다. 즉, 이 논문에서의 novelty는 “여러 activation을 섞는다” 그 자체보다, **서로 다른 activation들의 크기 차이를 제어한 뒤 convex combination 형태로 안정적으로 학습시키는 방법**에 있다.  

## 3. Detailed Method Explanation

### 3.1 기본 구성

논문은 각 neuron이 여러 activation 후보를 갖도록 설계한다. 가장 순진한 방법은 단순히 여러 activation을 더하는 것이지만, 저자들은 이것이 부적절하다고 본다. 예를 들어 ReLU 계열은 unbounded이고 sigmoid/tanh는 bounded이므로, 그냥 더하면 큰 값을 내는 activation이 학습을 지배하게 된다.

그래서 각 activation $f^j$에 대해 먼저 정규화된 출력 $h_i^j(z)$를 만든다. 논문에서 제시한 형태는 다음과 같다.

$$
h_i^j(z)=\frac{f^j(z)-\min_k(f^j(z_{ki}))}{\max_k(f^j(z_{ki}))-\min_k(f^j(z_{ki}))+\epsilon}
$$

여기서 핵심은 각 activation의 출력을 $[0,1]$ 범위에 가깝게 맞춰, 함수 간 상대적 기여가 공정해지도록 만드는 것이다. 저자들은 $[-1,1]$ 정규화도 실험했지만, $[0,1]$ 쪽이 비슷하거나 약간 더 낫다고 보고한다. 또한 음수 허용은 이후 $\alpha$ 선택을 불안정하게 만들 수 있다고 설명한다.

### 3.2 최종 activation ensemble

정규화된 각 activation 출력에 대해 neuron별 가중치 $\alpha$를 부여한다. 이 $\alpha$는 단순 파라미터가 아니라, **각 activation이 해당 neuron에서 얼마나 중요한지 나타내는 선택 변수**다. abstract에서도 저자들은 $\alpha$가 큰 activation이 결국 더 큰 magnitude를 갖게 되어 사실상 그 activation이 “선택”된다고 설명한다.

논문 텍스트상 정확한 최종 식이 현재 첨부본에서 일부 잘려 있지만, 설명 구조는 분명하다.

1. 여러 activation을 적용한다.
2. activation마다 출력 스케일을 정규화한다.
3. $\alpha$를 통해 convex combination을 만든다.
4. $\eta$, $\delta$로 normalization offset을 조정한다.
5. 이 모든 파라미터를 backpropagation으로 함께 학습한다.

즉, activation ensemble layer는 새로운 activation function을 handcraft하는 것이 아니라, **기존 함수들의 학습 가능한 조합을 neuron 수준에서 구성**하는 모듈이다.  

### 3.3 왜 $\eta$, $\delta$가 필요한가

이 논문에서 놓치기 쉬운 부분은 $\alpha$만이 아니라 $\eta$, $\delta$도 중요하다는 점이다. 저자들은 activation ensemble이 두 파트로 구성된다고 명시한다. $\alpha$는 activation function의 mixing weight이고, $\eta$, $\delta$는 normalization range를 동적으로 조정하는 offset 파라미터다. 이 변수들도 일반 학습 과정에서 함께 학습된다.

이 설계는 의미가 있다. activation ensemble이 단순히 “가장 좋은 함수 하나를 고르는 sparse gate”가 아니라, activation 함수들이 데이터 분포와 layer의 역할에 맞게 **적절한 스케일과 위치로 재정렬된 뒤 조합**될 수 있게 하기 때문이다.

### 3.4 기존 방법과의 차이

논문은 여러 선행연구와의 차이를 분명히 한다.

* **Maxout**은 여러 선형 출력을 두고 max를 취하지만, activation마다 전체 weight matrix가 추가되어 비용이 크고, max를 취하면서 다른 후보의 정보를 버린다. 반면 본 논문은 적은 추가 파라미터로 여러 activation을 결합한다.
* **Agostinelli et al.** 식의 learned activation은 ReLU 기반에 추가 변수를 붙이는 방식인데, 본 논문은 특정 family 내부를 일반화하는 것이 아니라 서로 다른 activation family 자체를 조합하려고 한다.
* 단순히 여러 activation을 더하는 기존 아이디어들과 달리, 이 논문은 **magnitude normalization + convex combination**을 통해 activation 간 불균형을 제어하려 한다.

## 4. Experiments and Findings

논문은 MNIST, ISOLET, CIFAR-100, STL-10을 사용한다. 구현은 Theano와 Titan X GPU에서 수행했고, in-house 네트워크에는 batch normalization을 각 ensemble layer 이전에 넣었다. optimizer는 AdaDelta를 사용했으며 learning rate 1.0을 썼다고 적고 있다.

### 4.1 MNIST와 activation 선택 패턴

저자들은 먼저 activation ensemble이 실제로 어떤 activation을 선택하는지 본다. 첫 번째 activation set(sigmoid, tanh, ReLU, softplus, exponential linear, inverse absolute value)에서는 전반적으로 **ReLU가 가장 자주 우세**했지만, deeper layer로 갈수록 한 activation이 압도하지 않고 가중치가 더 섞이는 경향이 나타났다. 또한 일부 lower layer neuron에서는 inverse absolute value가 중요하게 선택되었다고 해석한다.  

이 결과는 중요한 함의를 가진다. activation ensemble이 결국 ReLU 하나만 고르는 trivial mechanism이 아니라, **layer depth와 dataset에 따라 다른 activation을 실제로 활용**한다는 것이다.

### 4.2 같은 데이터셋에서도 FFN과 CNN은 다른 activation을 선호

논문은 MNIST에서 FFN과 CNN을 모두 실험하며, 같은 데이터셋이라도 모델 구조에 따라 선호 activation이 달라진다고 주장한다. 결론 부분에서도 MNIST를 예로 들어 FFN과 CNN의 optimal activation이 다르다고 명시한다. top layer 근처에서는 둘 다 ReLU를 선호하지만, bottom layer에서는 hyperbolic tangent와 inverse absolute value가 더 중요해졌다고 요약한다.

즉, “MNIST엔 ReLU가 좋다” 같은 단순 결론보다, **같은 데이터라도 representation stage에 따라 필요한 비선형성이 다르다**는 것이 저자들의 핵심 해석이다.

### 4.3 ISOLET

ISOLET 실험에서는 같은 FFN 구조를 써도 MNIST와 다른 activation 선호 패턴이 나타난다. 저자들은 ISOLET에서는 특정 activation이 전층에서 일관되게 우세하지 않으며, 특히 bottom layer 쪽에서 hyperbolic tangent가 더 선호된다고 정리한다. 이는 activation ensemble이 dataset-specific preference를 반영한다는 주장을 강화한다.  

### 4.4 CIFAR-100

CIFAR-100에서는 residual network에 activation ensemble을 삽입한다. 저자들은 residual network가 본질적으로 ReLU 중심으로 설계되어 있기 때문에 ensemble 삽입이 쉽지 않다고 설명한다. residual block 뒤나 residual addition 직전 등에 넣는 여러 시도를 했지만 실패했고, **residual block 내부 중간 지점에 ensemble을 넣는 방식이 가장 잘 작동했다**고 보고한다. 또 이 네트워크에서는 세 가지 activation set 중 **intercept가 다른 5개의 ReLU류로 구성된 두 번째 set**이 가장 잘 작동했다고 한다. 이는 ResNet이 본래 ReLU dynamics에 의존하기 때문이라고 해석한다.

이 실험은 activation ensemble이 단순 MLP/CNN뿐 아니라, 구조적으로 민감한 residual architecture에도 들어갈 수 있음을 보여 준다. 다만 동시에 placement가 매우 중요하다는 점도 드러낸다.

### 4.5 STL-10

결론부에 따르면 STL-10에서는 reconstruction loss 개선을 보고한다. 첨부된 조각들만으로 autoencoder의 세부 수치와 정확한 loss table 전부를 완전히 재구성하기는 어렵지만, 저자들의 종합 결론은 STL-10에서도 activation ensemble이 유효했다는 것이다.

### 4.6 전반적 결론

저자들은 activation ensemble이 MNIST, ISOLET, CIFAR-100, STL-10에서 전반적으로 더 나은 성능을 보였다고 주장한다. 또한 더 중요한 관찰로, 어떤 activation이 좋은지는 데이터셋과 모델, 그리고 층에 따라 달라진다고 결론짓는다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 activation selection 문제를 architecture search의 일부가 아니라 **end-to-end 학습 가능한 모듈**로 바꿨다는 점이다. 또한 “ReLU가 좋다” 같은 단일 서사를 넘어서, activation function이 데이터와 layer 역할에 따라 달리 쓰일 수 있음을 실험적으로 보여 준다. 특히 MNIST/ISOLET/ResNet 실험을 통해 dataset-dependent, architecture-dependent activation preference를 논의한 점이 흥미롭다.  

### 한계

한계도 분명하다. 첫째, 방법이 직관적이지만 **추가 파라미터와 normalization 관리**가 필요하다. activation 자체는 가볍지만, neuron마다 여러 activation을 모두 계산해야 하므로 계산량은 증가한다. 둘째, residual network 실험에서 보이듯이 네트워크 구조에 따라 ensemble placement가 까다롭다. 셋째, 첨부된 본문 조각 기준으로는 모든 실험 수치표가 완전히 보이지 않아, 이 논문을 엄밀한 SOTA 개선 논문으로 읽기보다 **설계 철학과 activation behavior 분석 논문**으로 읽는 것이 더 적절하다.

### 해석

비판적으로 해석하면, 이 논문의 중요한 통찰은 “새 activation을 또 하나 제안하는 것”이 아니라, **activation function space 자체를 학습의 대상**으로 바꾼 데 있다. 이후 mixture-of-experts적 비선형성, learned activation, adaptive gating 계열 아이디어와도 연결되는 문제의식이다. 다만 오늘날 기준으로 보면 parameter-efficient learned activation이나 smoother parametric activation들이 더 실용적일 수 있어서, 이 방법이 직접적으로 널리 채택되지는 않았다고 볼 수 있다. 그럼에도 아이디어 차원에서는 충분히 선구적이다.

## 6. Conclusion

이 논문은 activation ensemble이라는 개념을 통해, 각 neuron이 여러 activation 함수를 동시에 후보로 두고 학습 과정에서 그 조합을 선택하도록 만드는 방법을 제안했다. 이를 위해 activation 간 magnitude 차이를 해결하는 정규화, neuron별 mixing coefficient $\alpha$, 그리고 normalization offset인 $\eta$, $\delta$를 도입했다. 결과적으로 저자들은 MNIST, ISOLET, CIFAR-100, STL-10에서 성능 향상을 보고했고, 더 나아가 **최적 activation은 데이터셋, 모델 구조, 층 깊이에 따라 달라진다**는 점을 강조했다.  

실무적 관점에서 보면, 이 논문은 “ReLU냐 tanh냐” 같은 수동 선택 대신, 모델이 필요한 비선형성을 내부적으로 선택하게 만드는 방향을 제안한 셈이다. activation design을 고정된 함수 선택 문제에서 **학습 가능한 조합 문제**로 바꿨다는 점이 이 논문의 가장 큰 공헌이다.
