# Dynamic ReLU

## 1. Paper Overview

이 논문은 ReLU와 그 일반화들이 지금까지는 모두 **static activation** 이라는 점, 즉 어떤 입력이 들어와도 같은 방식으로 작동한다는 점을 문제로 삼는다. 저자들은 이 한계를 넘기 위해 **Dynamic ReLU (DY-ReLU)** 를 제안한다. DY-ReLU의 핵심은 activation 함수의 파라미터를 고정하지 않고, **입력 전체의 global context를 읽는 hyper function** 으로부터 동적으로 생성한다는 것이다. 다시 말해, 같은 네트워크라도 입력 이미지가 달라지면 activation 곡선 자체가 달라진다. 논문은 이 방식이 특히 경량 모델에서 표현력을 크게 높이면서도 계산량 증가는 매우 작다고 주장한다. 대표적으로 MobileNetV2에 DY-ReLU를 적용하면 ImageNet top-1 정확도가 **72.0%에서 76.2%로 상승**하고 FLOPs 증가는 약 **5%** 수준이라고 보고한다.  

이 문제가 중요한 이유는 activation function이 네트워크의 비선형 표현력을 결정하는 핵심 요소이기 때문이다. 기존의 ReLU, Leaky ReLU, PReLU, Maxout 등은 모두 입력과 무관하게 같은 규칙을 적용한다. 하지만 convolution feature는 입력마다 분포와 의미가 달라질 수 있으므로, activation도 입력에 적응적으로 변하면 더 강한 표현력을 얻을 수 있다는 것이 이 논문의 문제의식이다. 저자들은 dynamic network, hypernetwork, SENet, dynamic convolution 계열의 흐름과 연결하면서도, 기존 방법들이 주로 **kernel weights** 를 바꾸는 데 집중했던 반면 자신들은 **activation 자체를 동적으로 만들었다**고 주장한다.

## 2. Core Idea

핵심 아이디어는 매우 명확하다. 기존 ReLU 계열은 입력 $x$ 가 주어졌을 때 항상 같은 piecewise linear 함수를 적용한다. DY-ReLU는 이를 다음과 같이 일반화한다.

* 입력 전체 $\mathbf{x}$ 로부터 hyper function이 파라미터 $\theta(\mathbf{x})$ 를 생성한다.
* 이 파라미터로 각 채널 또는 위치의 piecewise linear activation을 결정한다.
* 따라서 activation 함수는 고정된 함수 $f(x)$ 가 아니라, **입력 의존적 함수** $f_{\theta(\mathbf{x})}(x)$ 가 된다.

논문이 강조하는 직관은 다음과 같다. 이미지 내의 모든 feature element가 담고 있는 **global context** 를 요약해 activation 곡선을 조절하면, 같은 convolution output이라도 입력 샘플에 따라 더 적절한 비선형 변환을 할 수 있다. 이 방식은 네트워크의 깊이와 너비를 늘리지 않고도 모델 용량을 키우는 효과를 낸다. 특히 가벼운 CNN은 본래 capacity가 제한적이므로, dynamic activation의 이득이 더 크게 나타난다고 해석한다.  

또 하나의 중요한 아이디어는 DY-ReLU를 **dynamic and efficient Maxout** 으로 볼 수 있다는 점이다. 논문은 DY-ReLU가 Maxout처럼 여러 선형 조각의 최대값을 취하지만, 그 조각들의 기울기와 절편이 입력에 따라 동적으로 달라진다는 점, 그리고 계산량은 Maxout보다 훨씬 작으면서 성능은 더 좋다고 설명한다.  

## 3. Detailed Method Explanation

### 3.1 Dynamic activation의 일반 형태

논문은 dynamic activation을 먼저 일반적으로 정의한다. 입력 벡터 또는 텐서 $\mathbf{x}$ 가 주어졌을 때 activation은 단순한 $f(\mathbf{x})$ 가 아니라,

$$
f_{\theta(\mathbf{x})}(\mathbf{x})
$$

형태를 갖는다. 여기서 두 부분이 있다.

1. **hyper function** $\theta(\mathbf{x})$
   입력 전체로부터 activation 파라미터를 생성한다.

2. **activation function** $f_{\theta(\mathbf{x})}(\mathbf{x})$
   생성된 파라미터를 사용해 실제 비선형 변환을 수행한다.

즉, DY-ReLU의 본질은 “입력으로부터 activation을 생성하는 activation”이다.

### 3.2 Static ReLU에서 Dynamic ReLU로

기존 ReLU는 채널별 입력 $x_c$ 에 대해

$$
y_c = \max(x_c, 0)
$$

형태다. 논문은 이를 더 일반적인 piecewise linear 함수로 확장한다. 정적 버전의 일반화는 대략

$$
y_c = \max_k { a_c^k x_c + b_c^k }
$$

형태이며, 여기서 $a_c^k$, $b_c^k$ 는 고정 또는 학습되는 파라미터다. DY-ReLU는 이 파라미터들을 다시 입력 의존적으로 바꾼다. 즉,

$$
y_c = \max_k { a_c^k(\mathbf{x}) x_c + b_c^k(\mathbf{x}) }
$$

와 같은 식으로 이해할 수 있다. 이때 $a_c^k(\mathbf{x})$, $b_c^k(\mathbf{x})$ 는 hyper function이 모든 입력 요소를 보고 생성한다.

### 3.3 구현 직관

논문이 강조하는 구현상의 장점은 DY-ReLU가 **깊이도 너비도 늘리지 않는다**는 점이다. 대신 매우 작은 hyper function만 추가된다. 이 hyper function은 입력 feature의 global context를 압축해 activation 파라미터를 만들고, 실제 activation은 여전히 piecewise linear이므로 계산량이 크지 않다. 그래서 성능 향상 대비 연산량 증가가 매우 작다고 주장한다.

### 3.4 세 가지 변형: DY-ReLU-A/B/C

논문은 spatial, channel 차원에서 activation을 얼마나 공유할지에 따라 세 가지 변형을 제안한다.

* **DY-ReLU-A**: spatial and channel-shared
* **DY-ReLU-B**: spatial-shared, channel-wise
* **DY-ReLU-C**: spatial and channel-wise  

이 차이는 매우 중요하다.
이미지 분류에서는 보통 채널별 의미 차이가 크므로 **channel-wise** 인 B, C가 더 적합하다. 반면 keypoint detection처럼 공간적으로 민감한 태스크에서는 head 쪽에서 위치마다 다른 activation이 필요하므로 **spatial-wise** 인 C가 더 유리하다. 논문은 실제 ablation에서 이 차이를 명확히 보여 준다.

### 3.5 prior work와의 관계

논문은 DY-ReLU를 기존 activation/generalization과 다음처럼 구분한다.

* **Leaky ReLU / PReLU**: 음수 기울기를 바꾸지만 입력과 무관한 static 함수
* **Maxout**: 여러 선형 조각의 최대를 취하지만 동적이지 않고 계산량이 큼
* **Swish 등 NAS 기반 activation**: 새로운 정적 함수 탐색
* **Dynamic convolution / SENet / hypernetwork**: 입력 의존적이지만 초점이 주로 kernel weight나 channel reweighting에 있음

DY-ReLU는 이들과 달리 **activation function 그 자체가 입력에 따라 달라진다**는 점이 차별점이다.

## 4. Experiments and Findings

### 4.1 실험 대상

논문은 두 가지 대표 과제를 사용한다.

* **ImageNet classification**
* **COCO keypoint detection**

그리고 여러 backbone에 대해 평가한다.

* ResNet
* MobileNetV2
* MobileNetV3  

### 4.2 ImageNet classification

논문의 핵심 결과는 DY-ReLU가 세 가지 CNN 아키텍처에서 모두 static activation보다 낫다는 것이다. 특히 MobileNetV2에 DY-ReLU를 적용하면 top-1 정확도가 **72.0% → 76.2%** 로 크게 향상된다. MobileNetV3에서도 improvement가 보고되며, MobileNetV3-Small은 **2.3%**, MobileNetV3-Large는 **0.7%** 향상된다고 적고 있다. 저자들은 특히 **작은 모델일수록 improvement가 더 크다**고 해석한다. 예로 MobileNetV2 ×0.35, MobileNetV3-Small, ResNet-10 같은 경량 모델에서 이득이 더 크다고 설명한다.

이 결과는 DY-ReLU의 주장을 정면으로 뒷받침한다. 즉, 큰 모델에서 이미 충분한 표현력을 갖는 경우보다, **capacity가 부족한 lightweight CNN에서 dynamic activation이 표현력 부족을 더 잘 보완한다**는 것이다.

### 4.3 COCO keypoint detection

COCO keypoint detection에서도 DY-ReLU는 효과를 보인다. MobileNetV2에서 keypoint detection AP가 **3.5 AP** 상승했다고 논문은 요약한다. MobileNetV3-Large와 MobileNetV3-Small에서도 각각 **1.5 AP**, **3.6 AP** 향상이 보고된다.  

이 실험은 중요하다. 분류처럼 global semantics가 중요한 문제뿐 아니라, 위치 민감성이 큰 pose estimation에서도 DY-ReLU가 유효함을 보여 주기 때문이다.

### 4.4 변형별 ablation

A/B/C 세 변형에 대한 결과도 논문의 중요한 기여다.

* **Image classification** 에서는 channel-wise 변형인 **DY-ReLU-B, C** 가 더 좋다.
* **Keypoint detection** 에서는 backbone에는 **B, C** 가 더 좋고, head에서는 **C** 가 특히 중요하다.
* head에서 A나 B를 쓰면 baseline보다 더 나빠질 수도 있다고 논문은 지적한다. 이유는 spatially-shared hyper function이 픽셀 수준의 spatially sensitive task를 학습하기 어렵기 때문이라고 해석한다.

이 결과는 DY-ReLU가 단순히 “동적이면 무조건 좋다”는 논문이 아니라, **과제 유형에 따라 어떤 차원에서 동적으로 만들 것인가**까지 분석한 논문이라는 뜻이다.

### 4.5 계산량과 효율성

논문은 DY-ReLU가 일관되게 성능을 올리면서도 추가 계산량은 대체로 **약 5% 수준**에 머문다고 강조한다. 또한 Maxout보다 훨씬 적은 계산으로 더 나은 결과를 얻는다고 주장한다. 이는 DY-ReLU가 실용적 replacement로서 의미가 있다는 핵심 근거다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **아이디어가 간단하면서도 적용 범위가 넓다**는 점이다. 기존 네트워크의 ReLU나 h-swish를 DY-ReLU로 바꾸기만 해도 일관된 개선이 보고된다. 구조 전체를 재설계할 필요가 없고, 계산량 증가도 작다. 특히 MobileNet 같은 효율형 CNN에서 improvement가 크다는 점은 실용성이 높다.  

또 다른 강점은 activation 연구를 dynamic network 문맥과 잘 연결했다는 점이다. 기존 dynamic method들이 weight나 block selection에 집중했던 반면, DY-ReLU는 **activation을 dynamic하게 만드는 것만으로도 상당한 이득**을 보여 준다. 이는 activation function을 단순한 고정 비선형성으로 보지 않고, 입력 의존적 모듈로 재해석한 중요한 시도다.

### 한계

한계도 있다. 첫째, DY-ReLU는 activation 자체는 가볍지만 **hyper function이라는 추가 모듈**에 의존한다. FLOPs 증가는 작아도 완전히 공짜는 아니다. 둘째, A/B/C 중 어떤 변형이 좋은지는 task에 따라 달라지므로, 사용자는 여전히 어느 정도 설계 선택을 해야 한다. 셋째, visible 결과 기준으로 논문의 강점은 주로 CNN, 특히 효율형 비전 모델에 집중되어 있다. Transformer나 sequence model까지 일반화되는지는 이 논문만으로는 말할 수 없다.  

### 해석

비판적으로 보면, 이 논문의 진짜 메시지는 “ReLU를 더 좋은 함수로 바꿨다”가 아니다. 더 중요한 메시지는 **activation function도 입력에 따라 달라져야 할 수 있다**는 것이다. 즉, convolution kernel뿐 아니라 activation까지 conditionally modulated될 수 있다는 관점을 제시한다. 이후 dynamic network, conditional computation, adaptive modulation 계열을 이해할 때도 이 문제의식은 꽤 선구적이다.

## 6. Conclusion

이 논문은 입력마다 동일하게 작동하는 static rectifier의 한계를 지적하고, 입력 전체의 global context로 activation 파라미터를 생성하는 **Dynamic ReLU (DY-ReLU)**를 제안했다. DY-ReLU는 입력 의존적인 piecewise linear activation으로, 기존 네트워크의 깊이와 너비를 늘리지 않으면서도 표현력을 높인다. 또한 DY-ReLU-A/B/C라는 세 가지 변형을 통해 spatial, channel 차원의 공유 방식을 달리 설계했고, 이미지 분류와 keypoint detection에서 서로 다른 최적 선택이 있음을 보였다.  

실험적으로는 ResNet, MobileNetV2, MobileNetV3에서 일관된 성능 향상을 달성했으며, 특히 MobileNetV2에서 ImageNet top-1이 **72.0%에서 76.2%** 로 증가하고 COCO keypoint detection에서도 **3.5 AP** 향상을 기록했다. 계산량 증가는 약 **5%** 수준으로 작다. 따라서 이 논문은 lightweight CNN의 표현력 향상을 위해 **activation을 dynamic하게 만드는 것**이 매우 효과적일 수 있음을 보여 준 대표적인 작업으로 볼 수 있다.  
