# How important are activation functions in regression and classification? A survey, performance comparison, and future directions

## 1. Paper Overview

이 논문은 activation function을 개별 기법 하나로 소개하는 논문이 아니라, **분류와 회귀를 함께 포괄하는 대규모 survey + 비교 연구**다. 저자들은 activation function이 인공신경망의 학습에서 핵심적이지만, 문제 유형에 따라 적합한 함수가 달라지고, 특히 최근에는 classification뿐 아니라 **physics-informed machine learning(PIML)** 같은 scientific machine learning 환경에서 요구 조건이 크게 달라진다는 점을 문제의식으로 둔다. 그래서 이 논문은 고전적 fixed activation, adaptive activation, complex-valued/quantized activation까지 폭넓게 정리하고, 실제로는 **MNIST, CIFAR-10, CIFAR-100**에서 분류 성능을 비교하며, 추가로 **PDE 기반 PIML 회귀 문제**에서도 activation의 차이를 실험한다.  

이 논문이 중요한 이유는, activation 연구를 단순히 “ReLU vs GELU” 같은 분류 성능 비교로 보지 않고, **문제 유형별 요구 조건의 차이**를 정리했다는 데 있다. 저자들의 핵심 결론은 분명하다. 일반적인 이미지 분류에서는 ReLU와 그 변형들이 여전히 매우 강력하지만, **PIML처럼 고차 미분이 필요한 회귀 문제에서는 ReLU 계열이 오히려 부적절**하며, tanh, swish, sine, 그리고 adaptive activation이 더 적합하다는 것이다. 즉, activation의 중요성은 “있다/없다” 수준이 아니라, **문제 정의에 따라 최적 선택이 크게 달라진다**는 메시지를 전달한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 activation function을 하나의 보편적 정답으로 보지 않고, **taxonomy + benchmark + application constraints**의 세 축으로 이해하자는 것이다.

첫째, activation function을 **고정형(fixed), 적응형(adaptive), 비표준(non-standard)**으로 나누고, 다시 응용 관점에서는 **real-valued, complex-valued, quantized** activation으로 구분한다. 저자들은 이 분류가 기존의 단순한 함수 나열보다 실제 문제 해결에 더 도움이 된다고 본다.

둘째, classification에서는 다양한 activation을 실제 CNN 백본인 **MobileNet과 VGG16** 위에서 비교해, 단순 문헌 요약이 아니라 일정 수준의 실험적 비교를 제공한다.

셋째, 회귀, 특히 **physics-informed machine learning**에서는 activation의 선택 기준이 완전히 달라진다고 주장한다. 여기서는 함수값 자체보다 **도함수의 존재성과 고차 미분의 안정성**이 더 중요해진다. 이 때문에 분류에서 잘 되는 ReLU 계열이 PIML에서는 오히려 잘 작동하지 않을 수 있다고 본다.  

즉, 이 논문의 central claim은 “activation function은 매우 중요하지만, 중요성의 의미는 task마다 다르다”는 것이다.

## 3. Detailed Method Explanation

### 3.1 논문의 전체 구성

논문은 다음 흐름으로 전개된다.

* activation function의 역사와 생물학적/인공신경 관점 비교
* activation이 갖추면 좋은 성질 정리
* characterization/application 기반 taxonomy 제시
* 고전적 fixed activation, complex-valued activation, quantized activation, adaptive activation 정리
* 분류 benchmark 비교
* physics-informed ML에서의 activation 요구 사항과 성능 비교
* 최종 요약 및 향후 방향 제시

이 구조 덕분에 논문은 단순 survey를 넘어서, **왜 activation을 문제별로 다시 생각해야 하는지**를 설계 원리 수준에서 설명한다.

### 3.2 저자들이 제시하는 activation의 바람직한 성질

논문은 어떤 activation이든 보편적으로 고려해야 할 특성들을 정리한다. 그중 핵심은 다음 다섯 가지다.

1. **Nonlinearity**
2. **계산 비용이 낮을 것**
3. **vanishing/exploding gradient 문제를 줄일 것**
4. **유한 범위 또는 boundedness가 있을 것**
5. **미분 가능성(differentiability)**

이 정리는 이후 논문의 두 축, 즉 classification과 PIML 비교를 이해하는 기준이 된다. 예를 들어 image classification에서는 비포화성, 계산 효율, 학습 안정성이 주로 중요하지만, PIML에서는 **고차 미분이 실제로 계산 가능해야 한다**는 조건이 훨씬 더 중요하게 작동한다. 이 점이 논문의 후반부 핵심으로 이어진다.

### 3.3 Taxonomy

논문은 activation taxonomy를 두 방식으로 제안한다.

첫째, **characterization-based taxonomy**다. 여기서는 activation을 크게

* fixed activation
* adaptive activation
* non-standard activation

으로 나눈다. adaptive activation 안에서도 parametric, ensemble, stochastic, fractional 등 여러 방식의 적응성이 가능하다고 본다.

둘째, **application-based taxonomy**다. 여기서는 activation을

* real-valued
* complex-valued
* quantized

관점으로 본다. 저자들은 scientific computation, acoustics, robotics, bioinformatics처럼 complex-valued representation이 필요한 분야를 별도로 강조하며, quantized activation은 edge computing과 효율성 측면에서 중요하다고 본다.

이 taxonomy의 의미는, activation을 단지 함수의 모양이나 미분 가능성만으로 비교할 것이 아니라 **문제 도메인과 구현 제약**까지 포함해 봐야 한다는 것이다.

### 3.4 Classification 비교 실험

논문은 classification 비교에서 **MobileNet**과 **VGG16**을 사용한다. 학습률은 $10^{-4}$, optimizer는 Adam, loss는 cross-entropy이며, 데이터셋은 MNIST, CIFAR-10, CIFAR-100이다. 즉, 비교 자체는 매우 최신 SOTA 설계는 아니지만, activation의 상대적 차이를 보기에는 충분히 일관된 setup을 제공한다.

### 3.5 Physics-informed ML 비교

논문 후반부는 이 survey의 가장 차별화된 부분이다. 저자들은 PIML에서 activation이 만족해야 할 핵심 조건으로 **고차 미분 가능성**을 강조한다. PDE residual을 계산해야 하므로 activation의 2차, 3차 이상 도함수가 실제로 안정적으로 존재하고 계산 가능해야 한다는 것이다. 이 요구 때문에 ReLU 계열이나 ELU 일부 변형은 classification에서는 잘 작동해도 PIML에서는 곧바로 쓰기 어렵다.  

또한 저자들은 TensorFlow, PyTorch, JAX를 모두 사용해 예측 정확도와 runtime도 비교한다. 이 부분은 activation 연구를 단순 함수 비교가 아니라 **framework-level practical study**로 확장했다는 점에서 의미가 있다.

## 4. Experiments and Findings

### 4.1 Classification에서의 관찰

논문 요약과 결론부에 따르면, **MNIST에서는 거의 모든 activation이 잘 작동**한다. 반면 더 어려운 CIFAR-10, CIFAR-100에서는 **swish, ReLU, Leaky ReLU, ELU, sine** 등이 비교적 강한 성능을 보이며, 특히 **adaptive activation과 Rowdy activation**은 전 데이터셋에서 전반적으로 좋은 성능을 냈다고 정리한다.  

이 결과의 의미는, classification에서는 여전히 ReLU 계열이 매우 강력하지만, **adaptive activation이 고정형을 넘어서는 경향**도 분명히 존재한다는 것이다. 저자들은 “adaptive activation outperforms its classical counterparts for almost any problem”이라고까지 정리한다.

### 4.2 PIML에서의 관찰

이 논문의 가장 중요한 실험적 결론은 여기 있다. PIML에서는 **sine, tanh, swish**가 전반적으로 잘 작동하며, **ELU는 일부 문제에서 성능이 낮고, 고차 도함수가 존재하지 않아 일부 PDE 문제에는 사용할 수 없고, sigmoid는 고차 도함수의 진폭이 작아 성능이 좋지 않다**고 명시한다.  

더 강한 결론은 abstract와 summary에 이미 들어 있다. **ReLU와 그 변형은 classification에서는 SOTA급이지만 PIML에서는 잘 작동하지 않는다**는 것이다. 이유는 “derivatives must exist”라는 엄격한 요구 때문이다. 반대로 adaptive activation은, 특히 **multiscale problem**에서 더 우수한 성능을 보인다고 정리한다.  

### 4.3 ML framework 비교

논문은 PIML 문제에서 **JAX가 TensorFlow와 PyTorch보다 예측 정확도에서 한 자릿수(about order-of-magnitude) 수준의 개선을 줄 수 있고, JIT/XLA 덕분에 계산 효율도 좋다**고 정리한다. 또한 Hessian 계산 같은 higher-order optimization 전략도 JAX가 더 쉽게 구현된다고 본다.

이 부분은 activation 자체의 본질적 비교는 아니지만, 실제 activation 선택과 도함수 계산이 framework의 autodiff 성능과 밀접하게 연결된다는 점을 보여 준다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **activation을 classification과 regression/PIML이라는 서로 다른 맥락에서 함께 본 것**이다. 대부분의 activation survey는 image classification 중심으로 끝나는데, 이 논문은 scientific ML로 논의를 확장하면서 activation selection의 기준이 완전히 달라질 수 있음을 잘 보여 준다.

또 다른 강점은 taxonomy가 넓다는 점이다. fixed vs adaptive뿐 아니라, complex-valued, quantized activation까지 포함해 activation landscape 전체를 조망하려고 한다.

### 한계

한편 한계도 있다. 첫째, classification 비교는 MobileNet과 VGG16 위주라서, 최신 Transformer/LLM 계열 activation 선택 문제까지 직접 다루지는 않는다. 둘째, adaptive activation이 거의 항상 더 좋다고 요약하지만, 이런 비교는 추가 파라미터와 tuning cost까지 공정하게 따져 봐야 하므로, “언제나 실용적으로 우월하다”로 곧바로 일반화하기는 어렵다. 이 부분은 논문의 결론을 바탕으로 한 비판적 해석이다.

셋째, 논문은 매우 넓은 범위를 다루기 때문에, 개별 activation 하나의 수학적 깊이나 최신 초거대 모델에서의 behavior를 깊게 파고들기보다는 **폭넓은 정리와 비교**에 더 가깝다.

### 해석

비판적으로 읽으면, 이 논문의 가장 중요한 메시지는 “activation이 중요하다”보다 더 구체적이다. 바로 **‘좋은 activation’은 task-dependent하다**는 것이다. 분류에서는 ReLU 계열과 modern smooth activations가 강하고, PIML에서는 differentiability와 고차 도함수 구조가 더 중요하며, multiscale scientific problem에서는 adaptive activation이 강점이 있다. 즉, activation selection은 더 이상 부차적 하이퍼파라미터가 아니라 **문제 정의의 일부**라는 해석이 가능하다.  

## 6. Conclusion

이 논문은 activation function을 분류와 회귀, 특히 physics-informed machine learning까지 포함해 폭넓게 정리한 **종합 survey + 비교 연구**다. 고정형, 적응형, 복소수형, 양자화 activation을 taxonomy로 정리하고, 실제 분류 benchmark와 PDE 기반 scientific ML 문제에서 성능 차이를 비교했다. 핵심 결론은 다음과 같다.

* **classification에서는 ReLU와 그 변형, swish, sine 등이 강력하며, adaptive activation은 전반적으로 더 좋은 경향이 있다.**
* **PIML에서는 ReLU 계열이 적절하지 않을 수 있으며, tanh, swish, sine, adaptive activation이 더 적합하다.**
* **multiscale scientific problem에서는 adaptive activation의 이점이 특히 크다.**
* **JAX는 PIML activation 실험에서 정확도와 효율 측면에서 강한 장점을 보인다.**  

종합하면 이 논문은 “어떤 activation이 최고인가?”라는 질문에 단일 답을 주지 않는다. 대신 **‘어떤 문제를 푸는가에 따라 activation의 중요성이 달라지고, 최적 선택도 달라진다’**는 더 중요한 답을 준다. 이 점이 이 논문의 가장 큰 가치다.
