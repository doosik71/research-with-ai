# Activation Functions: Comparison of Trends in Practice and Research for Deep Learning

## 1. Paper Overview

이 논문은 새로운 딥러닝 모델이나 특정 activation function 하나를 제안하는 연구가 아니라, **deep learning에서 사용되어 온 activation functions(AF)**을 폭넓게 정리하고, 실제 실무 아키텍처에서 어떤 함수가 주로 쓰이는지와 연구 논문에서 어떤 함수들이 제안되고 성능 향상을 주장하는지를 **비교·정리한 survey paper**다. 저자들의 문제의식은 분명하다. 딥러닝이 다양한 응용 분야로 확장되면서 activation function 선택이 모델 성능과 학습 안정성에 큰 영향을 주지만, 기존 문헌은 개별 함수 소개나 성능 비교에 치우쳐 있었고, **“연구에서 뜨는 함수”와 “실제 배포/주요 아키텍처에서 쓰이는 함수” 사이의 간극**을 체계적으로 정리한 문헌이 부족하다는 것이다.  

왜 이 문제가 중요한가도 논문에서 분명히 설명된다. 딥러닝이 깊어질수록 vanishing gradient, exploding gradient 같은 문제가 커지고, activation function은 이런 gradient dynamics와 표현력에 직접 영향을 준다. 따라서 activation function은 단순한 구현 디테일이 아니라, 초기화·정규화·regularization과 함께 학습 성공 여부를 좌우하는 핵심 설계 요소다. 저자들은 이 논문을 통해 어떤 activation이 어떤 위치와 응용에서 쓰였는지 정리하고, 실제 선택에 참고할 수 있는 “의사결정 지도”를 제공하려 한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 실험적 novelty가 아니라 **정리 방식의 novelty**에 있다. 저자들은 activation function을 단순히 수학식이나 장단점으로만 나열하지 않고, 다음 두 관점을 결합한다.

첫째, **activation function 자체의 계보와 특성**을 정리한다. Sigmoid, tanh, ReLU 같은 대표 함수에서 시작해 hard sigmoid, SiLU, dSiLU, ELU, PReLU, SReLU, HardELiSH 등 다양한 변형을 소개하고, 각 함수가 어떤 약점을 보완하려고 나왔는지를 설명한다.

둘째, **연구 결과와 실제 사용 추세를 비교**한다. 많은 최신 논문들이 새로운 activation을 제안하고 기존 함수보다 우수하다고 보고하지만, 실제 state-of-the-art 아키텍처에서는 여전히 **hidden layer에 ReLU**, **output layer에 Softmax 또는 Sigmoid** 같은 보수적 선택이 주류라는 점을 강조한다. 즉, 이 논문의 핵심 메시지는 다음과 같이 요약할 수 있다.

* 연구는 지속적으로 “ReLU보다 나은 함수”를 제안한다.
* 그러나 실무와 대표 아키텍처는 여전히 검증된 함수에 의존한다.
* 따라서 activation 선택은 단순히 최신 논문을 따라가는 문제가 아니라, **task, architecture, output semantics, 계산비용, 검증 수준**을 함께 고려해야 한다.  

## 3. Detailed Method Explanation

### 3.1 논문의 구조와 분석 프레임

이 논문은 총 6개 섹션으로 구성되며, 앞부분에서 deep learning과 activation function의 역할을 소개하고, 중간에서 activation function들의 종류를 폭넓게 정리한 다음, 후반부에서 **실제 딥러닝 아키텍처에 쓰인 activation의 추세**를 비교하고 마지막에 논의와 결론을 제시한다. 저자들은 스스로도 이 논문이 “기존 AF들을 폭넓게 모아 놓고, practical deployment에서의 사용 경향과 연구 결과를 함께 정리했다”는 점을 novelty로 내세운다.  

### 3.2 activation function의 역할 정의

논문은 activation function을 neural network에서 **weighted sum과 bias의 결과를 비선형으로 변환해 다음 층으로 전달하는 transfer function**으로 정의한다. 선형 변환만 반복하면 네트워크 깊이가 늘어도 본질적으로 하나의 선형 변환과 크게 다르지 않기 때문에, 고차 비선형 패턴을 학습하려면 activation이 필요하다고 설명한다. 또한 activation은 output layer에서는 예측의 의미를 정하고, hidden layer에서는 표현력과 gradient flow를 조절한다.

논문은 기본 선형 모델을 다음처럼 설명한다.

$$
f(x)=w^T x+b
$$

그리고 activation이 적용된 비선형 출력은 개념적으로

$$
y=\alpha(w_1x_1+w_2x_2+\cdots+w_nx_n+b)
$$

형태라고 쓴다. 여기서 $\alpha$가 activation function이다. 이 설명은 수학적으로 아주 새롭지는 않지만, survey 논문으로서 activation의 위치와 역할을 명확히 구분해 준다.

### 3.3 activation 선택의 핵심 기준

저자들이 반복해서 강조하는 기준은 세 가지다.

첫째, **비선형성 제공**이다. activation은 선형 모델을 비선형으로 바꿔 깊은 네트워크가 복잡한 함수를 학습할 수 있게 한다.

둘째, **gradient dynamics**다. 논문은 vanishing/exploding gradient를 activation 설계의 핵심 문제로 제시한다. derivative가 너무 작으면 gradient가 사라지고, 너무 크면 폭주한다. activation function은 이런 현상을 완화하도록 설계되며, 각 함수의 등장 배경도 상당 부분 여기에 있다.

셋째, **응용/위치 적합성**이다. 어떤 activation은 hidden layer에서 유리하고, 어떤 것은 output layer에서 확률 해석에 적합하다. 예를 들어 sigmoid는 output layer의 probability modeling에 자주 쓰이지만, hidden layer에서는 saturation과 non-zero-centered output 때문에 불리할 수 있다고 정리한다.

### 3.4 개별 activation들의 정리 방식

논문은 activation들을 chronology가 아니라 “대표 함수와 그 변형”의 구조로 정리한다. 예를 들어 sigmoid 계열에서는 다음 흐름이 보인다.

* **Sigmoid**: bounded, differentiable, binary classification의 output에 적합하지만 saturation과 slow convergence 문제가 있다.
* **Hard Sigmoid**: sigmoid 근사형으로 계산비용이 낮고 binary classification에서 유망하다고 소개된다. 식은 대략 다음 형태다.
  $$
  f(x)=\mathrm{clip}\left(\frac{x+1}{2},0,1\right)
  $$

* **SiLU**: sigmoid와 input의 곱으로 구성되는 함수로, reinforcement learning 기반 시스템에서 소개되며 ReLU보다 나은 결과가 보고되었다고 정리한다.
* **dSiLU**: SiLU의 derivative 형태를 activation처럼 사용하며 standard sigmoid보다 더 낫다고 인용한다.

이런 식으로 논문은 여러 activation family를 폭넓게 요약한다. 다만 이 논문은 개별 함수의 이론을 깊게 증명하는 paper가 아니라, **등장 배경·수식·장단점·대표 적용 사례**를 정리하는 catalog형 survey에 가깝다.

### 3.5 trend comparison 방법

논문의 방법론 중 가장 중요한 부분은 Section IV의 trend comparison이다. 저자들은 activation usage trend를 보기 위해 **ILSVRC/ImageNet 계열의 대표 딥러닝 아키텍처**와 관련 문헌을 기준으로 삼는다. 이유는 ImageNet competition이 딥러닝의 첫 대중적 성공을 낳았고, 이후 대표 아키텍처 변화가 실무 adoption을 잘 반영한다고 보기 때문이다. 또한 vision 외에도 NLP 등 activation 함수가 등장한 다른 응용도 함께 참고했다고 말한다.

즉, 이 논문은 controlled benchmark experiment를 새로 돌리기보다는,

* 기존 문헌에서 activation 수식과 장단점을 정리하고
* 대표 아키텍처에서 실제로 어떤 activation이 어디에 쓰였는지 표로 비교하고
* 연구에서 제안된 최신 함수들과 실무 추세의 차이를 해석하는 방식으로 진행된다.  

## 4. Experiments and Findings

이 논문은 전형적인 “하나의 모델을 만들어 benchmark를 돌리는 실험 논문”은 아니다. 대신 survey 기반의 정리와 comparative discussion이 핵심이며, 그 결과물은 주로 **Table I, Table II, 그리고 discussion/conclusion의 해석**으로 나타난다. 따라서 이 섹션의 핵심은 저자들이 문헌 검토를 통해 끌어낸 주요 관찰들이다.

### 4.1 activation function은 많아졌지만 실무는 보수적이다

논문이 가장 강하게 말하는 관찰은 이것이다. **새로운 activation functions는 지속적으로 제안되고, 일부는 기존 ReLU보다 더 좋은 성능을 보고한다.** 그런데도 실제 최신 딥러닝 아키텍처는 대부분 **hidden layer에서 ReLU**, **output layer에서 Softmax**, 경우에 따라 **Sigmoid**를 계속 사용한다. 저자들은 이를 “newer activation functions are rarely used in practice”라고 해석한다.

### 4.2 practical trend: hidden은 ReLU, output은 Softmax/Sigmoid

Table I에 대한 저자들의 직접적인 해석에 따르면, practical DL applications에서 지배적인 activation은 **ReLU와 Softmax**다. 특히 Softmax는 대부분의 실무 응용에서 output layer에 사용되고, hidden layer에서는 ReLU 계열이 주류다. 다만 최근 일부 architecture에서는 output prediction에 sigmoid가 쓰이는 사례도 언급한다.  

이는 중요한 결론이다. 연구 논문들은 ELU, PReLU, SReLU, SiLU 등 더 정교한 함수들을 제안하지만, 실제 널리 쓰이는 네트워크 설계는 여전히 단순하고 검증된 조합에 머무른다는 뜻이기 때문이다.

### 4.3 최신 아키텍처들도 여전히 ReLU 중심이다

저자들은 DenseNet, MobileNets, ResNeXt 같은 보다 최근의 architectures에도 ReLU가 내장되어 있다고 설명한다. 또한 SeNet 같은 당시 최신 recognition architecture에서도 hidden layer는 ReLU, output은 sigmoid를 사용했다고 정리한다. 이 점을 근거로, 최신 SOTA가 곧 최신 activation adoption을 의미하지는 않는다고 주장한다.  

### 4.4 연구는 다양한 activation을 탐색 중이다

논문은 Section III에서 sigmoid 계열, hard sigmoid, SiLU, dSiLU, 그리고 이후 여러 ReLU 변형·ELU 변형 등을 소개하며, activation research가 매우 활발하게 진행되고 있음을 보여 준다. 저자들이 보기에 activation 연구의 방향은 점점 **compound activation** 또는 기존 함수의 약점을 보완한 변형들로 가고 있다. 결론에서도 “compounded activation functions looking towards the future”라고 정리한다.

### 4.5 도메인별 output semantics가 activation 선택을 바꾼다

논문은 activation이 object recognition, speech, segmentation, scene description, machine translation, text-to-speech, cancer detection, weather forecast, self-driving cars 등 다양한 영역에서 쓰인다고 정리한다. 하지만 이 vast application map에도 불구하고, practical choice는 surprisingly narrow하다. 이 대비가 논문의 핵심 관찰 중 하나다. 즉, **응용은 다양하지만 선택은 제한적**이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **survey의 초점이 명확하다**는 점이다. activation function을 전부 나열하는 데 그치지 않고, “연구에서 제안된 함수”와 “실제 널리 쓰이는 함수”를 구분해 보여 준다. 이는 논문을 읽는 사람에게 매우 실용적이다. 단순히 최신 activation 이름을 아는 것보다, **왜 아직도 ReLU/Softmax가 지배적인지**를 이해하게 해 주기 때문이다.  

또한 activation의 역할을 gradient flow, nonlinearity, output semantics라는 관점에서 정리한 점도 유용하다. survey paper지만 단순 백과사전식 나열보다는 선택 기준을 제공하려는 의도가 분명하다.

### 한계

한계도 명확하다. 첫째, 이 논문은 **새로운 통제 실험을 수행하는 empirical paper가 아니다**. 따라서 각 activation의 우열을 강하게 일반화하기 어렵다. 저자들은 기존 문헌을 요약하며 “어떤 함수가 더 좋았다”는 보고를 소개하지만, 동일한 조건에서 대규모 재검증한 것은 아니다.

둘째, trend comparison이 주로 ImageNet/ILSVRC 계열 대표 아키텍처에 기반하고 있어, 당시 기준의 vision-centric bias가 있다. NLP, speech 등도 언급되지만 analysis의 중심은 대표 vision architectures다.

셋째, 첨부 HTML 기준으로 이 논문의 강점은 정리력이지, activation 이론에 대한 깊은 수학적 분석은 아니다. 예를 들어 왜 어떤 activation이 optimization landscape를 더 좋게 만드는지에 대한 엄밀한 분석보다는, 기존 문헌의 설명과 장단점 정리에 더 가깝다.

### 해석

비판적으로 해석하면, 이 논문은 “최고의 activation function은 무엇인가?”에 대한 답을 주기보다, 오히려 그 질문 자체가 너무 단순하다고 말하는 논문이다. 실제 메시지는 다음과 가깝다.

* hidden layer와 output layer의 목적은 다르다.
* 연구에서 좋은 함수와 실무에서 채택되는 함수는 다르다.
* activation 선택은 수학적 elegance보다 **검증 수준, 계산비용, 구현 편의성, 아키텍처 친화성**에 크게 좌우된다.

이 점에서 이 논문은 activation function research를 따라가는 사람에게 좋은 입문 정리이면서, 동시에 “왜 산업 현장은 보수적인가”를 보여 주는 실용적 survey라고 볼 수 있다.  

## 6. Conclusion

이 논문은 deep learning에서 사용되는 activation functions를 폭넓게 정리하고, 특히 **실제 아키텍처에서의 사용 추세와 연구 문헌에서의 최신 제안을 비교**했다는 점에서 의미가 있다. 저자들의 핵심 결론은 다음과 같다.

* activation function은 딥러닝 학습 성능, gradient flow, 일반화에 중요한 설계 요소다.
* 연구에서는 ReLU를 넘어서는 다양한 activation이 제안되고 있다.
* 그러나 실제 최신 딥러닝 아키텍처들은 여전히 **hidden layer에서 ReLU**, **output layer에서 Softmax 또는 Sigmoid**를 주로 사용한다.
* 따라서 activation 선택은 “가장 최신 함수”보다, **문제 유형과 배포 가능성에 맞는 검증된 선택**이 중요하다.  

실무적 관점에서 보면, 이 논문은 activation function을 새로 설계하고 싶은 연구자에게는 아이디어 맵을, 모델을 실제로 배포해야 하는 엔지니어에게는 “왜 아직도 ReLU를 쓰는가”에 대한 정리된 답을 제공한다.
