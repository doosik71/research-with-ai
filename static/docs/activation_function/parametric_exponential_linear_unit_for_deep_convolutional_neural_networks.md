# Parametric Exponential Linear Unit for Deep Convolutional Neural Networks

## 1. Paper Overview

이 논문은 ELU(Exponential Linear Unit)의 핵심 장점인 음수 출력과 bias shift 완화 효과는 유지하면서, 기존 ELU에서 사람이 수동으로 정해야 했던 shape parameter를 **학습 가능한 형태**로 바꾸는 activation function인 **PELU (Parametric ELU)** 를 제안한다. 저자들은 CNN에서 activation function이 단순 비선형성이 아니라 최적화 안정성, 표현력, 수렴 속도, 최종 일반화 성능에 직접 영향을 준다고 보고, 각 layer가 자기에게 맞는 activation shape를 직접 배우도록 만드는 것이 더 합리적이라고 주장한다. 이를 MNIST, CIFAR-10/100, ImageNet에서 ResNet, NiN, Overfeat, All-CNN, VGG 등에 적용해 ELU 대비 개선을 보였다고 보고한다. 특히 NiN on ImageNet에서 상대 오류율 7.28% 개선을 강조하며, 추가 파라미터는 극히 적다고 주장한다.

이 문제가 중요한 이유는 activation의 형태가 네트워크의 gradient 전달과 hidden representation의 분포를 바꾸기 때문이다. ELU는 음수 영역을 통해 bias shift를 줄이는 장점이 있지만, 그 shape을 고정하면 네트워크와 layer마다 최적인 비선형 구조를 반영하기 어렵다. 논문은 이 한계를 “각 layer가 activation shape를 직접 학습”하게 함으로써 해결하려 한다.

## 2. Core Idea

핵심 아이디어는 간단하다. **기존 ELU를 그대로 쓰지 말고, 양수 구간의 기울기와 음수 구간의 포화점 및 decay 속도를 조절하는 파라미터를 학습하자**는 것이다. 저자들은 ELU의 음수 출력이 bias shift 완화에 유리하다는 점은 받아들이되, 특정 고정 shape가 모든 네트워크와 모든 layer에 최적일 리는 없다고 본다. 그래서 activation의 형태 자체를 layer별로 적응적으로 배우게 만든다.

논문의 novelty는 단순히 “ELU에 파라미터를 하나 더 붙였다”는 수준이 아니다. 저자들은 직접 ELU의 파라미터 $a$만 학습하면 $h=0$에서 미분 가능성이 깨질 수 있다는 점을 지적하고, 이를 피하기 위해 **양수/음수 양쪽을 함께 재매개변수화**한다. 이 설계 덕분에 함수가 fully differentiable하게 유지되고, 역전파 업데이트에 유리하다고 주장한다. 또 계산량은 ELU와 같은 수준이며, layer당 추가 파라미터는 2개뿐이라 실용성도 확보하려 한다.

## 3. Detailed Method Explanation

### 3.1 기본 ELU와 문제 설정

기존 ELU는 양수 구간에서는 identity, 음수 구간에서는 지수 함수 형태를 갖는다. 논문은 ELU가 음수 값을 허용하기 때문에 bias shift 완화에 도움을 줄 수 있다고 설명한다. 하지만 ELU의 핵심 parameter $a$는 보통 고정되어 있고, 다른 값을 직접 학습하면 $h=0$에서 미분 불연속 문제가 생길 수 있다. 즉, 단순히 “ELU의 음수 쪽 amplitude만 learnable로 만들자”는 접근은 깔끔하지 않다.

### 3.2 PELU 정의

논문이 제안하는 PELU는 다음과 같다.

$$
f(h)=
\begin{cases}
ch & \text{if } h \ge 0 \
a\left(\exp\left(\frac{h}{b}\right)-1\right) & \text{if } h < 0
\end{cases},
\qquad a,b,c>0
$$

여기서 각 파라미터의 역할은 다음과 같다.

* $c$: 양수 구간의 선형 기울기
* $b$: 음수 구간 exponential decay의 scale
* $a$: 음수 구간 saturation point를 조절

즉, PELU는 ELU의 음수 쪽 포화 정도만 바꾸는 것이 아니라, 양수 구간 기울기까지 포함해 activation 전체 shape를 조절한다. 논문 Figure 1 설명에 따르면, $a$가 커질수록 음수 포화점이 더 낮아지고, $b$가 커질수록 decay가 느려지며, $c$가 커질수록 양수 선형 구간의 기울기가 증가한다.

### 3.3 미분 가능성을 유지하는 제약

논문에서 중요한 설계 포인트는 **미분 가능성 보존**이다. 저자들은 ELU에서 단순히 $a$만 학습하면 $a \ne 1$일 때 $h=0$에서 미분 가능성이 깨진다고 지적한다. 이를 피하기 위해 positive side와 negative side를 함께 parameterize하여 함수 전체를 differentiable하게 유지하려고 한다. 이 점이 PReLU, APL, SReLU 같은 기존 parametric activation들과의 차별점으로 제시된다. 특히 SReLU나 APL과 달리 PELU는 non-differentiable kink를 여러 개 만들지 않는다는 점을 강조한다.

### 3.4 파라미터 수와 계산 복잡도

PELU는 각 layer마다 2개의 추가 파라미터만 늘어난다. 논문은 이를 전체 layer 수를 $L$이라 할 때 추가 파라미터 수가 $2L$이라고 정리한다. 즉, weight tensor 전체에 비하면 증가량이 매우 작다. 저자들은 이 때문에 PELU가 ELU와 거의 같은 계산적 부담으로 적용 가능하다고 주장한다. 특히 ImageNet 실험에서 NiN에 추가된 파라미터 비율이 0.0003%에 불과하다고 보고한다.

### 3.5 다른 parametric activation과의 비교 관점

저자들은 PELU를 다음 계열과 비교한다.

* **PReLU**: 음수 구간 slope만 학습
* **APL**: 여러 hinge의 weighted sum
* **Maxout**: 여러 affine의 max
* **SReLU**: 세 개의 선형 함수 조합

이 비교의 요지는 “PELU는 fully differentiable하면서도 expressive하고, 파라미터 수는 작다”는 것이다. Maxout처럼 파라미터를 크게 늘리지 않고, APL/SReLU처럼 비미분 지점을 늘리지도 않는다는 점을 장점으로 제시한다.

## 4. Experiments and Findings

### 4.1 실험 대상과 설정

논문은 MNIST, CIFAR-10/100, ImageNet 2012를 사용한다. 네트워크는 ResNet, NiN, All-CNN, Overfeat, VGG를 포함한다. 저자들은 비교의 공정성을 위해 가능한 한 activation만 교체하고 나머지 구조와 training framework는 크게 바꾸지 않았다고 설명한다.

### 4.2 CIFAR-10/100 결과

CIFAR-10/100에서는 110-layer ResNet을 중심으로 PELU, ELU, BN-ReLU, BN-PReLU 등을 비교한다. Figure 4와 Table 1 설명에 따르면, **PELU는 ELU보다 더 나은 convergence와 더 낮은 recognition error**를 보였다. 또한 저자들은 주기여가 ELU 대비 개선임을 분명히 하면서도, PReLU와 비교해도 CIFAR-100에서는 PELU가 더 유리했다고 설명한다.

### 4.3 Batch Normalization의 영향

논문에서 꽤 흥미로운 부분은 **BN을 activation 앞에 두면 PELU와 ELU 모두 성능이 나빠진다**는 관찰이다. ResNet-110에서 CIFAR-10/100 실험 결과, ELU는 BN 앞삽입 시 오류율이 크게 악화되었고, PELU도 악화되지만 상대적 악화 폭은 더 작았다. 구체적으로 ELU의 minimum median test error는 CIFAR-10에서 5.99%에서 10.39%로, CIFAR-100에서 25.08%에서 34.75%로 증가했다. PELU는 CIFAR-10에서 5.36%에서 5.85%, CIFAR-100에서 24.55%에서 25.38%로 증가했다. 즉, PELU가 BN의 악영향을 완전히 없애지는 못하지만, ELU보다는 덜 민감했다. 그럼에도 저자 결론은 분명하다. **PELU 앞에는 BN을 두지 말아야 한다.**

### 4.4 ImageNet 결과

ImageNet 2012 validation set에서 ResNet18, NiN, All-CNN, Overfeat를 평가한 결과, Figure 6 설명에 따르면 **모든 경우에서 PELU가 ELU보다 낮은 Top-1 error**를 기록했다. 특히 NiN은 PELU에서 36.06% error, ELU에서 40.40% error로, 상대 개선률이 7.29%였다. 논문은 이것이 단순히 파라미터 수 증가 때문이 아니라고 주장한다. NiN에는 추가 파라미터가 24개뿐이므로, 표현력 증가보다는 activation shape adaptation 자체가 효과를 냈다고 해석한다. 또한 regime #1처럼 learning rate와 decay가 더 공격적인 학습 조건에서 PELU의 이점이 더 크게 나타났다고 보고한다.

### 4.5 파라미터 구성 실험

논문은 PELU 파라미터화를 여러 방식으로 시도했고, 제안한 구성이 가장 좋은 성능을 보였다고 주장한다. 이 부분의 핵심은 “아무 parametric ELU나 다 좋은 것은 아니며, differentiability와 shape control을 동시에 만족하는 parameterization이 중요하다”는 점이다. 즉, 논문의 성과는 단순히 ELU에 자유도를 넣은 것보다, **어떤 방식으로 자유도를 넣느냐**에 상당히 좌우된다.

### 4.6 VGG에서 학습된 activation shape 관찰

저자들은 VGG에서 학습된 PELU 파라미터를 시각화해, layer마다 서로 다른 activation shape가 학습되었다고 보여준다. 예를 들어 어떤 layer에서는 $a$가 0.5 부근, 다른 layer에서는 2 부근으로 수렴했다. 이는 네트워크가 ReLU처럼 음수 출력이 0인 형태만 선호한 것이 아니라, **여러 layer에서 음수 saturation을 유지하는 activation**을 택했다는 뜻이다. 저자들은 이것을 bias shift 완화에 유리한 음수 출력의 중요성을 뒷받침하는 추가 실험 증거로 해석한다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 강점은 제안이 매우 간단하면서도 설계 논리가 분명하다는 점이다. ELU의 장점을 계승하면서 fixed shape라는 약점을 직접 겨냥했고, differentiability 문제를 회피하는 parameterization도 함께 제시했다. 또한 CIFAR와 ImageNet까지 포함해 여러 대표 CNN에서 ELU 대비 개선을 보여 “toy idea”에 그치지 않도록 했다.

또 하나의 강점은 **실제 학습 과정에서 layer별 activation shape가 달라진다**는 점을 관찰로 보여줬다는 것이다. 이는 PELU가 단순한 regularization 트릭이 아니라, 네트워크가 서로 다른 stage에서 서로 다른 비선형성을 원한다는 가설을 지지한다.

### 한계

반면 한계도 분명하다. 첫째, 왜 BN이 PELU/ELU 앞에서 성능을 악화시키는지에 대한 설명은 가설 수준에 머문다. Discussion에서 저자들은 positive scale invariance가 부족해 BN과 궁합이 나쁠 수 있다고 추정하지만, 이는 검증된 이론이 아니라 future work로 남겨 둔다.

둘째, 비교 기준이 주로 ELU 중심이다. 논문 당시로서는 자연스럽지만, 더 넓은 activation family와의 대규모 비교는 제한적이다. 또한 성능 향상이 activation shape adaptation 때문인지, 특정 optimizer/training regime와의 상호작용 때문인지는 완전히 분리되지 않았다. 실제로 논문 스스로도 regime #1에서 개선 폭이 더 크게 나타났다고 말한다.

셋째, activation을 layer-wise scalar parameter로 조절하는 접근은 가볍지만, channel-wise나 neuron-wise adaptation보다 표현력이 제한적일 수 있다. 논문은 parameter efficiency를 장점으로 내세우지만, 그만큼 세밀한 조정 능력에는 한계가 있을 수 있다. 이 부분은 논문에서 직접 깊게 논의되지는 않는다. 이는 본문을 바탕으로 한 해석이다.

### 해석

비판적으로 보면, 이 논문은 “좋은 activation은 고정 함수여야 하는가?”라는 질문에 대한 초기의 설득력 있는 답변 중 하나다. 즉, activation도 weight처럼 학습 대상이 되어야 한다는 관점이다. 이후 등장한 더 복잡한 learned activation이나 gating 계열과 비교하면 구조는 단순하지만, CNN 문맥에서 **작은 자유도만 줘도 의미 있는 이득을 얻을 수 있다**는 점을 잘 보여준다.

## 6. Conclusion

이 논문은 ELU를 learnable activation으로 확장한 **PELU**를 제안하고, 이를 다양한 CNN에 적용해 ELU 대비 일관된 성능 향상을 보였다고 보고한다. 핵심 기여는 다음과 같이 정리할 수 있다. 첫째, ELU를 fully differentiable하게 parameterize하는 실용적 방법을 제시했다. 둘째, layer별로 activation shape를 학습하게 하여 네트워크가 더 유연한 비선형 구조를 사용하도록 만들었다. 셋째, CIFAR-10/100과 ImageNet에서 empirical improvement를 보였다. 넷째, BN과의 상호작용, parameterization choice, learned shape visualization까지 함께 분석해 단순 benchmark 보고를 넘어서려 했다.

실무적으로는 ELU를 쓰는 CNN에서 큰 구조 변경 없이도 성능 향상을 노릴 수 있는 가벼운 대안으로 볼 수 있다. 연구적으로는 activation function을 고정 설계물이 아니라 **학습 가능한 함수 패밀리**로 보는 흐름을 강화한 작업으로 의미가 있다. 다만 BN과의 궁합 문제, 더 넓은 비교군 부족, 이론적 설명의 제한은 남아 있다.
