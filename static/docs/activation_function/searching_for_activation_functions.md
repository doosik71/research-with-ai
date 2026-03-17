# Searching for Activation Functions

## 1. Paper Overview

이 논문은 activation function 설계를 사람이 손으로 만들기보다, **자동 탐색(search)** 으로 찾을 수 있는지에 대한 대표적인 연구다. 저자들은 ReLU가 여전히 가장 널리 쓰이지만, 대체 activation들의 성능 향상은 모델과 데이터셋에 따라 들쭉날쭉해서 널리 채택되지 못했다고 본다. 그래서 activation function 자체를 탐색 대상에 넣고, **작은 search space에서는 exhaustive search**, 큰 search space에서는 **RNN controller + reinforcement learning** 을 사용해 새로운 scalar activation들을 찾는다. 그 결과 가장 강력한 함수로 **Swish**, 즉 $f(x)=x\cdot \sigma(\beta x)$ 를 발견했고, 이 함수가 이미지 분류와 기계번역 등 여러 어려운 과제에서 ReLU를 일관되게 능가하거나 최소한 동등 이상이라고 주장한다. 특히 ImageNet에서는 Mobile NASNet-A에서 top-1 정확도 **0.9%**, Inception-ResNet-v2에서 **0.6%** 향상을 보고한다.

이 문제가 중요한 이유는 activation function이 단순한 부품이 아니라, **학습 동역학과 최종 성능에 직접적인 영향**을 주기 때문이다. 그런데 기존 activation들은 대부분 사람이 “좋을 것 같은 성질”을 기준으로 손으로 설계했다. 이 논문은 그 과정을 자동화해서, neural architecture search가 convolution cell을 찾듯이 **activation도 search로 발견할 수 있다**는 것을 보여 준다. 이 점 때문에 이 논문은 단순히 Swish를 제안한 논문이 아니라, activation 설계 자체를 **search problem** 으로 바꿔 놓은 논문으로 볼 수 있다.  

## 2. Core Idea

핵심 아이디어는 activation function을 미리 정해진 수학식으로 보지 않고, **unary function과 binary function의 조합으로 생성되는 compositional object** 로 보자는 것이다. 논문은 Figure 1에서 “core unit”을 정의하는데, 이는 두 입력에 각각 unary function을 적용한 뒤 binary function으로 결합하는 구조다. 이 core unit을 한 번 또는 여러 번 반복해 하나의 scalar-to-scalar activation function을 만든다.

예를 들어 unary function 후보에는 $x$, $-x$, $|x|$, $x^2$, $x^3$, $\sqrt{x}$, $\beta x$, $x+\beta$, $\log(|x|+\epsilon)$, $\exp(x)$, $\sin(x)$, $\cos(x)$, $\tanh(x)$, $\sigma(x)$ 등 다양한 기본 연산이 들어가고, binary function은 곱셈이나 덧셈, 더 복잡한 결합을 포함한다. 이 조합을 search algorithm이 선택해 activation을 구성한다. 이렇게 하면 사람이 직접 떠올리기 어려운 함수도 탐색할 수 있다.

이 search의 결과로 발견된 가장 중요한 함수가 바로 **Swish** 다.

$$
f(x)=x\cdot \sigma(\beta x)
$$

여기서 $\beta$ 는 상수일 수도 있고 학습 가능한 파라미터일 수도 있다. 특히 $\beta=1$ 인 경우를 저자들은 **Swish-1** 이라 부른다. 이 함수의 직관은 ReLU처럼 hard threshold로 자르는 대신, 입력을 sigmoid gate로 **부드럽게 조절(soft gating)** 하는 것이다. 즉, 큰 양수는 거의 그대로 통과하고, 큰 음수는 강하게 억제되지만, 0 근처에서는 매끄럽게 변화한다. 이 점이 논문이 발견한 가장 중요한 설계 원리다.  

## 3. Detailed Method Explanation

### 3.1 Search space 설계

논문은 search space 설계에서 **표현력과 탐색 가능성의 균형**을 중요하게 본다. search space가 너무 좁으면 새로운 activation을 찾을 수 없고, 너무 넓으면 탐색 자체가 불가능하다. 그래서 unary/binary function 조합으로 이루어진 compositional search space를 만든다. core unit 하나만 쓰는 경우는 exhaustive enumeration이 가능하지만, core unit을 여러 번 반복하면 search space가 **$10^{12}$ 규모**까지 커져 exhaustive search가 비현실적이 된다.

### 3.2 두 가지 search 방식

작은 공간에서는 **exhaustive search** 를 사용한다. 이 경우 가능한 activation들을 전부 생성해, child network를 학습시킨 뒤 validation accuracy로 순위를 매긴다. 큰 공간에서는 **RNN controller** 를 쓴다. 이 controller는 각 timestep마다 activation function의 한 구성 요소를 예측하고, 그 출력을 다음 timestep의 입력으로 다시 넣는 autoregressive 방식으로 전체 수식을 생성한다. controller는 validation accuracy를 reward로 받으며 **reinforcement learning** 으로 학습된다.

### 3.3 평가 프로토콜

search 단계에서는 child network로 **ResNet-20** 을 사용하고, **CIFAR-10에서 10K steps** 동안 학습해 validation accuracy를 측정한다. 저자들도 인정하듯이, 이런 작은 child network에서 찾은 activation이 더 큰 모델에도 일반화되는지는 별도의 검증이 필요하다. 그래서 search 후반부에서 ResNet-164, Wide ResNet 28-10, DenseNet 100-12 등 더 큰 CIFAR 모델에도 적용해 robustness를 확인한다. 그 결과 상위 activation 8개 중 6개가 generalize했고, 그중 2개는 바로 Swish 계열이었다고 보고한다.  

### 3.4 Swish의 구현과 실전 팁

논문은 Swish를 구현하기 매우 쉽다고 강조한다. TensorFlow 예시로는 `x * tf.sigmoid(beta * x)` 또는 이후 버전의 `tf.nn.swish(x)` 를 언급한다. 다만 BatchNorm을 사용할 때는 주의가 필요하다고 말한다. 일부 high-level library는 ReLU를 가정해 BatchNorm의 scale parameter를 꺼 두는데, **Swish에서는 이 설정이 적절하지 않다**고 지적한다. 또한 학습 시에는 ReLU 네트워크에서 쓰던 learning rate를 약간 낮추는 것이 좋았다고 보고한다.

## 4. Experiments and Findings

### 4.1 Search 단계에서의 발견

작은 child network 기준 CIFAR 실험에서, 검색으로 발견된 novel activation들 다수가 ReLU를 능가하거나 근접했다. 특히 상위 activation 가운데 $x\cdot \sigma(\beta x)$ 형태가 반복적으로 등장했고, 이것이 후속 대규모 실험의 주인공이 되는 **Swish** 다. 즉, Swish는 논문의 본문 실험에서 우연히 하나 골라낸 함수가 아니라, search 과정에서 반복적으로 좋은 보상을 준 구조로 선택된 것이다.  

### 4.2 CIFAR: 큰 모델로의 일반화

논문은 ResNet-164, Wide ResNet 28-10, DenseNet 100-12 위에서 Swish를 평가한다. 요약 문장에 따르면, **Swish와 Swish-1은 CIFAR-10과 CIFAR-100에서 모든 모델에 대해 ReLU를 일관되게 능가하거나 최소한 동등** 했고, 거의 모든 경우에서 최고 baseline과 맞먹거나 더 좋았다. 저자들은 특히 activation마다 “best baseline”이 모델별로 달라지는 반면, Swish는 여러 모델에서 안정적으로 강한 결과를 보인다는 점을 강조한다.

### 4.3 ImageNet

ImageNet에서는 Mobile NASNet-A, Inception-ResNet-v2, Inception-v4, MobileNet 등의 ReLU 기반 아키텍처에 Swish를 넣어 비교한다. 논문 초록과 실험 요약에 따르면, **Mobile NASNet-A에서는 top-1이 0.9%**, **Inception-ResNet-v2에서는 0.6%** 향상된다. 또 본문은 Inception-ResNet-v2에서 ReLU 대비 **0.5%** improvement를 “nontrivial”하다고 표현하며, mobile-sized model에서 효과가 특히 좋다고 해석한다. Mobile NASNet-A 표 일부에서도 Swish가 ReLU보다 높은 top-1 / top-5 정확도를 보인다.

### 4.4 Machine Translation

12-layer Transformer의 WMT English→German 실험에서도 Swish 계열은 강한 결과를 보인다. 논문은 **Swish가 다른 baseline을 능가하거나 최소한 동등** 하다고 정리하고, 특히 **Swish-1은 newstest2016에서 차상위 baseline보다 0.6 BLEU 높다**고 말한다. 반면 Softplus는 도메인에 따라 불안정한 성능을 보여, activation의 안정성과 범용성 측면에서 Swish가 더 설득력 있다고 해석한다.  

### 4.5 전체 요약 비교

논문은 다양한 모델·데이터셋 결과를 한데 모아, Swish가 ReLU 및 다른 baseline activation보다 **통계적으로 유의하게 더 낫다**고 주장한다. aggregate comparison은 one-sided paired sign test 기준으로 significance를 보였다고 적는다. 즉, 이 논문은 “어떤 한 모델에서만 우연히 좋았다”가 아니라, **여러 도메인에 걸쳐 일관된 우위**를 핵심 근거로 든다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **activation function discovery를 자동화했다**는 점이다. 기존 activation 연구는 사람이 함수 모양을 설계하고 실험으로 검증하는 방식이었는데, 이 논문은 search를 통해 activation을 발견하는 프로세스를 보여 줬다. 이는 이후 activation research뿐 아니라 neural architecture search 전반의 흐름과도 잘 맞는다.  

또 하나의 강점은 발견된 함수가 복잡하지 않다는 점이다. 많은 search 논문은 최종 결과가 지나치게 복잡해 실무에 쓰기 어렵지만, Swish는

$$
f(x)=x\cdot \sigma(\beta x)
$$

처럼 매우 단순하고, ReLU 대체도 쉽다. 저자들도 바로 이 실용성을 강조한다.  

### 한계

한계도 있다. 첫째, search 과정 자체는 매우 계산비용이 크다. activation 하나를 평가하려면 child network를 실제로 학습해야 하므로, distributed training이 필요할 정도다. 즉, 논문의 핵심 산출물은 실용적이지만, 그것을 **찾아내는 과정은 비싸다**.

둘째, search는 ResNet-20과 CIFAR-10이라는 비교적 작은 환경에서 진행되므로, 발견된 함수가 다른 아키텍처에 일반화된다는 보장은 별도 실험이 필요하다. 논문이 실제로 후속 검증을 하긴 했지만, 여전히 search objective와 최종 deployment objective 사이에는 간극이 있다.

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 “Swish가 ReLU보다 조금 좋다”에만 있지 않다. 더 중요한 것은 **activation function 설계가 더 이상 손작업(human hand-design)에만 의존하지 않아도 된다**는 점을 보여 준 것이다. 또한 Swish의 형태를 보면, search가 결국 hard-threshold보다 **smooth self-gating** 이 더 좋을 수 있다는 설계 원리까지 드러냈다고 해석할 수 있다. 이후 SiLU/Swish류 activation이 널리 채택된 것도 이 통찰과 연결된다.

## 6. Conclusion

이 논문은 activation function을 unary/binary 연산 조합의 search space 안에서 자동 탐색하는 방법을 제안하고, 그 결과 가장 강력한 함수로 **Swish** 를 발견했다.

$$
f(x)=x\cdot \sigma(\beta x)
$$

Swish는 단순하면서도 ReLU를 여러 도메인에서 일관되게 능가하거나 최소한 동등 이상이었다. 논문은 CIFAR의 여러 CNN, ImageNet의 Mobile NASNet-A 및 Inception-ResNet-v2, 그리고 WMT Transformer까지 폭넓게 검증해, Swish가 **이미지 분류와 기계번역 모두에서 안정적인 개선**을 준다고 주장한다. 특히 ImageNet에서는 Mobile NASNet-A에서 **0.9%**, Inception-ResNet-v2에서 **0.6%** top-1 향상을 보고했다.

정리하면, 이 논문은 activation research에서 두 가지를 남겼다. 하나는 **Swish라는 실용적 activation**, 다른 하나는 **activation을 자동 검색 대상으로 본다는 새로운 연구 패러다임** 이다. 후자의 의미가 특히 크며, 이 논문은 activation design을 경험적 craft에서 algorithmic discovery로 옮긴 대표 작업으로 볼 수 있다.
