# Mish: A Self Regularized Non-Monotonic Activation Function

## 1. Paper Overview

이 논문은 deep neural network의 activation function으로 **Mish**를 제안한다. Mish는 $f(x)=x\tanh(\mathrm{softplus}(x))$로 정의되며, 저자는 이를 **self-regularized, smooth, non-monotonic activation**이라고 설명한다. 핵심 목적은 ReLU의 단점인 음수 구간 정보 손실과 Dying ReLU 문제를 줄이면서도, Swish 계열의 장점인 smoothness와 작은 negative response 보존 특성을 유지하여 더 나은 최적화와 일반화를 얻는 것이다. 논문은 단순한 함수 제안에 그치지 않고, CIFAR-10, MNIST, ImageNet-1k, MS-COCO 등 여러 비전 벤치마크에서 ReLU, Leaky ReLU, Swish 등과 비교 실험을 수행해 Mish의 유효성을 주장한다.

이 문제가 중요한 이유는 activation function이 단순한 비선형 변환이 아니라, gradient flow, optimization landscape, 학습 안정성, 최종 정확도에 직접 영향을 주기 때문이다. 논문은 특히 Mish가 smooth한 출력/손실 지형을 만들어 더 쉬운 최적화를 유도하고, 결과적으로 분류와 객체 검출 모두에서 일관된 개선을 보인다고 주장한다.

## 2. Core Idea

논문의 중심 아이디어는 **Swish의 self-gating 성질을 유지하면서, 더 강한 regularization과 smoother gradient behavior를 유도하는 activation**을 설계하는 것이다. Mish는 입력 $x$와 비선형 게이트 $\tanh(\mathrm{softplus}(x))$를 곱하는 구조를 가진다:

$$
f(x)=x\tanh(\mathrm{softplus}(x)) = x\tanh(\ln(1+e^x))
$$

여기서 중요한 직관은 다음과 같다.

첫째, Mish는 **양의 방향으로는 unbounded**, **음의 방향으로는 bounded**이며, 완전히 0으로 잘라버리지 않고 작은 음수 값을 보존한다. 이는 ReLU처럼 음수 입력을 전부 죽이지 않기 때문에 정보 흐름 측면에서 유리하다. 둘째, 함수가 **smooth하고 non-monotonic**이어서, sharp transition이 많은 piecewise linear 함수보다 gradient가 더 부드럽게 전달될 수 있다. 셋째, 저자는 Mish의 1차 미분식에 등장하는 항이 일종의 **preconditioner-like behavior**를 하여 optimization을 쉽게 만든다고 해석한다.

논문이 주장하는 novelty는 “NAS로 발견된 Swish와 비슷한 효과를 보이지만, 보다 체계적 분석과 실험을 통해 설계된 함수”라는 점이다. 또한 단순 accuracy 향상뿐 아니라, 손실 지형이 더 smooth해지고 large/deep model에서 더 안정적이라는 해석을 함께 제시한다.

## 3. Detailed Method Explanation

### 3.1 Mish의 정의와 성질

Mish는 다음과 같이 정의된다.

$$
f(x)=x\tanh(\mathrm{softplus}(x))
$$

또는

$$
f(x)=x\tanh(\ln(1+e^x))
$$

논문은 Mish의 range를 대략 $[\approx -0.31,\infty)$로 설명한다. 즉, 음수 쪽은 제한되지만 양수 쪽은 제한되지 않는다. 이는 SELU류처럼 bounded-below 특성으로 regularization 효과를 어느 정도 제공하면서, 양수 영역에서 saturation을 피할 수 있다는 해석으로 이어진다.

### 3.2 미분과 gradient behavior

논문은 Mish의 1차 도함수를 Swish와의 관련성 속에서 설명한다. 먼저 다음 형태를 제시한다.

$$
f'(x)=\mathrm{sech}^2(\mathrm{softplus}(x))\cdot x\cdot \sigma(x) + \frac{f(x)}{x}
$$

또는 이를

$$
f'(x)=\Delta(x),\mathrm{swish}(x)+\frac{f(x)}{x}
$$

형태로 쓸 수 있다고 설명한다. 여기서 $\Delta(x)$는 논문이 강조하는 항이며, 저자는 이 항이 gradient를 더 smooth하게 만들어 optimization을 쉽게 하는 **preconditioner 유사 역할**을 할 수 있다고 해석한다. 다만 이 부분은 엄밀한 이론적 증명이라기보다, 실험과 도함수 모양에 기반한 해석에 가깝다. 즉, “왜 좋은가”에 대한 완전한 수학적 정당화보다는 **관찰 기반의 설명**이다.

논문 4페이지에는 Mish의 다른 형태의 도함수도 제시된다.

$$
f'(x)=\frac{e^x\omega}{\delta^2}
$$

여기서

$$
\omega = 4(x+1)+4e^{2x}+e^{3x}+e^x(4x+6),
\qquad
\delta = 2e^x + e^{2x} + 2
$$

이다. 이 식은 Mish가 연속적으로 미분 가능하다는 점, 즉 gradient-based optimization에서 singularity를 피할 수 있다는 점을 부각하는 데 사용된다.

### 3.3 왜 ReLU보다 나을 수 있는가

논문의 논리는 비교적 분명하다.

ReLU는 음수 입력을 0으로 보내므로 sparsity와 계산 효율 측면에서는 장점이 있지만, 음수 영역 gradient가 끊겨 **Dying ReLU** 문제가 발생할 수 있다. 반면 Mish는 음수 구간에서 완전히 죽지 않고 작은 음수 값을 유지한다. 또한 함수 프로파일이 smooth하여, 논문 4페이지 Figure 3의 랜덤 네트워크 출력 landscape 비교에서 ReLU는 sharp transitions를 보이는 반면 Mish는 더 smooth한 profile을 보인다. 저자는 이 smooth한 출력 landscape가 더 smooth한 loss landscape로 이어질 수 있다고 해석한다.

### 3.4 손실 지형 해석

논문 5페이지 Figure 4는 ResNet-20을 CIFAR-10에 200 epochs 학습한 뒤, ReLU/Mish/Swish의 loss landscape를 비교한다. 저자 해석에 따르면 Mish는 ReLU나 Swish보다 **더 smooth하고 better-conditioned된 loss landscape**, 그리고 **wider minima**를 보인다. 이는 일반화 성능 향상과 연결된다고 주장된다. 여기서 중요한 점은 이 시각화가 Mish의 효과를 설명하는 정성적 근거라는 점이다. 즉, Mish가 더 좋은 성능을 내는 이유를 단순 accuracy 표가 아니라 optimization geometry 관점에서도 설명하려고 한다.

### 3.5 구현과 효율성

Mish는 ReLU보다 계산이 복잡하다. TanH, SoftPlus, identity의 조합이기 때문이다. 논문은 실용적 구현에서 Softplus에 threshold 20을 두어 overflow를 막고 안정성을 높인다고 설명한다. 또한 PyTorch 기반 **Mish-CUDA** 구현을 제안하여 순수 Mish 대비 forward/backward 시간을 크게 줄였다고 보고한다. 표 6에 따르면 native Mish는 ReLU보다 훨씬 느리지만, Mish-CUDA는 overhead를 상당히 줄여 SoftPlus 수준에 가깝게 만든다. 따라서 “성능 향상은 있으나 계산량 증가가 따른다”는 trade-off를 인정하면서, 커스텀 CUDA 구현으로 이를 완화하려 한다.

## 4. Experiments and Findings

## 4.1 실험 구성

논문은 MNIST와 CIFAR-10에서 ablation을 수행하고, ImageNet-1k 분류와 MS-COCO 객체 검출까지 확장한다. 비교 대상은 ReLU, Leaky ReLU, Swish, GELU, ELU, SELU 등 다수의 activation이다. 평가지표로는 classification에서는 accuracy와 loss, detection에서는 mAP/AP50 등이 사용된다. 또한 일부 실험에서는 CutMix, Mosaic, Label Smoothing, SAT, DropBlock 등의 augmentation/regularization 기법도 함께 쓴다.

### 4.2 유사 함수와의 비교

초반 ablation에서 저자는 Swish와 유사한 여러 함수, 예를 들어 $\arctan(x)\mathrm{softplus}(x)$, $\tanh(x)\mathrm{softplus}(x)$, $x\log(1+\arctan(e^x))$, $x\log(1+\tanh(e^x))$ 등을 시험했다. CIFAR-10의 6-layer CNN 기준으로 Mish가 대체로 더 좋았고, 일부 함수는 성능이 비슷해도 deeper architecture에서 training divergence가 있었다고 서술한다. 이 실험은 “Mish가 우연히 잘 된 것이 아니라 비슷한 후보군 중에서 안정성과 성능을 함께 만족했다”는 메시지를 전달한다.

### 4.3 MNIST ablation

MNIST fully-connected network 깊이 증가 실험에서, 15 layer 이후 ReLU와 Swish의 정확도가 급격히 떨어진 반면 Mish는 더 높은 정확도를 유지했다. 이는 깊은 네트워크에서 Mish가 optimization difficulty를 덜 겪는다는 저자의 주장과 연결된다. 또 Gaussian noise를 입력에 추가한 실험에서도 Mish가 Swish와 ReLU보다 일관되게 낮은 test loss를 보였다. 마지막으로 CIFAR-10 initializer 비교에서도 여러 initialization 전략에서 Mish가 Swish보다 대체로 우수했다.

### 4.4 통계적 분석

CIFAR-10에서 SqueezeNet으로 23회 반복 실험한 통계 분석이 제시된다. 표 1에 따르면 Mish는 평균 정확도 $\mu_{acc}=87.48%$로 비교군 중 최고였고, 평균 손실은 $4.13%$로 두 번째로 낮으며, 정확도 표준편차도 낮은 편이었다. Swish는 $87.32%$, ReLU는 $86.66%$였다. 이 결과는 Mish가 단일 run의 우연이 아니라, 평균적으로도 강하다는 근거로 제시된다. 다만 ELU나 ISRU처럼 일부 지표에서 더 좋은 분산/손실을 보이는 함수도 있어, Mish가 모든 지표에서 압도적이라고 보기는 어렵다. 그래도 정확도 기준으로는 가장 좋은 결과를 낸다.

### 4.5 CIFAR-10 주요 결과

표 2에서 Mish는 다양한 아키텍처에서 ReLU와 Swish를 대체로 앞선다. 예를 들어 ResNet-20에서 Mish 92.02%, Swish 91.61%, ReLU 91.71%이며, WRN-10-2에서는 Mish 86.83%, Swish 86.56%, ReLU 84.56%다. EfficientNet-B0에서도 Mish 80.73%, Swish 79.37%, ReLU 79.31%로 보고된다. 논문은 이를 근거로 Mish가 architecture-specific trick이 아니라 비교적 넓은 범위의 vision architecture에서 잘 동작한다고 주장한다.

### 4.6 ImageNet-1k 결과

ImageNet-1k에서는 ResNet-18, ResNet-50, SpineNet-49, PeleeNet, CSP-ResNet-50, CSP-DarkNet-53, CSP-ResNext-50 등을 비교했다. ResNet-50에서 Mish는 Top-1 76.1%, ReLU는 75.2%, Swish는 75.9%였다. CSP-DarkNet-53에서는 Mish가 78.7% Top-1로 Leaky ReLU 77.8%보다 높다. 특히 CSP-ResNext-50에서 Swish가 크게 불안정한 결과를 보이는 반면, Mish는 Leaky ReLU보다 높은 성능을 보였다고 서술한다. 저자는 이를 근거로 “Swish는 큰 복잡 모델에서 일관적이지 않을 수 있지만 Mish는 더 robust하다”고 해석한다.

다만 이 부분은 해석에 주의가 필요하다. 여러 모델이 서로 다른 augmentation 설정을 사용하고 있고, 일부 칸은 누락되어 있어 완전히 대칭적인 controlled comparison이라고 보기는 어렵다. 그래도 논문이 보여주는 방향성은 분명하다. 즉, **Mish는 최소한 large-scale classification에서도 작은 개선 이상을 기대할 수 있다**는 것이다.

### 4.7 MS-COCO 객체 검출 결과

논문은 CSP-DarkNet-53 backbone과 YOLOv4 계열 detector에서도 Mish를 평가한다. 표 4에서는 CSP-DarkNet-53에서 ReLU 64.5% 대비 Mish 64.9%, 608x608 설정에서는 Mish 65.7%를 보고한다. 표 5의 YOLOv4 변형들에서는 AP50 기준으로 개선 폭이 더 분명하다. 예를 들어 YOLOv4pacsp-s에서 Leaky ReLU 54.2% 대비 Mish 56.3%, YOLOv4pacsp에서는 64.8% 대비 65.7%, YOLOv4pacsp-x에서는 66.1% 대비 67.4%다. 저자는 이를 평균적으로 0.9%~2.1%의 AP50 향상으로 요약한다.

이 결과는 Mish의 장점이 단순 image classification에만 국한되지 않고, detection backbone의 feature extraction 품질에도 도움을 줄 수 있음을 시사한다. 특히 논문 초록에서도 YOLOv4 + CSP-DarkNet-53에서 AP50이 2.1% 개선되었다고 강조한다.

### 4.8 계산 효율 비교

표 6은 ReLU, SoftPlus, Mish, Mish-CUDA의 FP16/FP32 forward/backward runtime을 비교한다. 원본 Mish는 ReLU보다 상당히 느리지만, Mish-CUDA는 FP16 forward 267.3µs, backward 345.6µs 수준으로 최적화되어 실용성이 올라간다. 즉, 논문은 성능 향상만이 아니라 deployment feasibility도 일부 고려했다. 그러나 여전히 ReLU만큼 단순하고 빠르지는 않다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **제안, 해석, 실험**이 비교적 고르게 갖춰져 있다는 점이다. 단순히 “새 activation 하나 제안”이 아니라, 도함수 해석, output/loss landscape 시각화, 다중 벤치마크, 통계 반복 실험, CUDA 최적화까지 포함한다. 따라서 독자는 Mish가 왜 효과적일 수 있는지에 대해 단순 empirical claim보다 풍부한 그림을 얻을 수 있다.

또 다른 강점은 **실험 범위**다. MNIST, CIFAR-10, ImageNet-1k, MS-COCO에 걸쳐 분류와 검출 모두에서 개선 사례를 제시한다. 특히 객체 검출 backbone 교체만으로 AP50 개선을 보인 점은 실제 CV 파이프라인에서 관심을 끌 수 있는 결과다.

### 한계

가장 큰 한계는 **이론적 설명의 엄밀성 부족**이다. $\Delta(x)$가 preconditioner처럼 동작한다는 설명은 흥미롭지만, 이는 정식 이론보다는 실험 관찰에 기반한 해석에 가깝다. 논문 스스로도 향후 이 regularizing term의 메커니즘을 더 이해할 필요가 있다고 인정한다.

둘째, 비교 실험이 완전히 공정한 controlled setting만으로 구성되지는 않는다. 일부 ImageNet 결과는 augmentation 사용 여부가 다르고, 비교표에서 결측값이 있으며, 훈련 framework나 레시피가 아키텍처마다 다르다. 따라서 “Mish가 항상 모든 activation보다 우월하다”는 강한 결론을 일반화하기는 어렵다. 더 정확한 평가는 동일 recipe, 동일 tuning budget, 동일 regularization 아래의 대규모 재현이 필요하다.

셋째, **계산 비용** 문제가 있다. Mish는 ReLU보다 분명히 비싸다. Mish-CUDA로 개선되지만, 기본적으로 값싼 piecewise linear activation을 대체하는 데 따른 latency/throughput trade-off는 남는다. 즉, deployment 환경에서는 accuracy gain이 그 비용을 상쇄하는지 따져야 한다.

### 해석

비판적으로 보면, Mish의 성공은 “non-monotonic + smooth + small negative preservation”이라는 최근 activation 설계 흐름의 연장선에 있다. Swish, GELU와 비슷한 방향이며, 특히 transformer 이전의 CV 문맥에서 ReLU류보다 부드러운 activation의 장점을 실험적으로 보여준 논문으로 볼 수 있다. 따라서 이 논문의 핵심 가치는 Mish 그 자체뿐 아니라, **activation function이 optimization geometry를 얼마나 바꿀 수 있는가**를 강조했다는 데 있다.

## 6. Conclusion

이 논문은 Mish를 새로운 activation function으로 제안하고, 이를 다양한 vision task에서 평가하여 ReLU, Leaky ReLU, Swish 대비 자주 더 좋은 결과를 보인다고 주장한다. 핵심 기여는 다음 세 가지로 요약할 수 있다. 첫째, $x\tanh(\mathrm{softplus}(x))$라는 간단하지만 expressive한 비선형 함수를 제안했다. 둘째, smooth/non-monotonic/bounded-below 성질이 gradient flow와 loss landscape에 유리할 수 있음을 해석적으로 제시했다. 셋째, CIFAR-10, ImageNet-1k, MS-COCO에서 광범위한 empirical improvement를 보고했다.

실무적으로는 “기존 backbone에서 activation만 바꿔도 성능을 조금 더 끌어올릴 수 있는가”라는 질문에 대한 유용한 답을 제공한다. 연구적으로는 activation design을 단순 heuristic이 아니라 optimization 관점에서 분석하는 흐름을 강화한 논문으로 의미가 있다. 다만 비용 증가와 이론적 미완성은 남아 있으며, 이 때문에 Mish는 “무조건 표준이 되는 함수”라기보다, 특정 task와 model에서 성능-비용 trade-off를 감수할 가치가 있는 선택지로 이해하는 편이 적절하다.
