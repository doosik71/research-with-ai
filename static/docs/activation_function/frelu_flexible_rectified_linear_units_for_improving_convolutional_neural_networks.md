# FReLU: Flexible Rectified Linear Units for Improving Convolutional Neural Networks

## 1. Paper Overview

이 논문은 CNN에서 가장 널리 쓰이는 activation인 ReLU의 한계를 정면으로 다룬다. 저자들은 ReLU가 계산이 단순하고 gradient vanishing을 완화하는 장점이 있지만, 음수 입력을 전부 0으로 잘라 버리는 **zero-hard rectification** 때문에 negative information을 활용하지 못한다고 본다. 동시에 ELU류가 음수 값을 살려 zero-like property를 제공하긴 하지만, 지수 연산이 필요하고 batch normalization과의 궁합 문제가 있다는 점도 지적한다. 이를 해결하기 위해 저자들은 ReLU의 절단점을 고정값이 아니라 학습 가능한 형태로 바꾼 **FReLU (Flexible Rectified Linear Unit)** 를 제안한다. 논문의 핵심 주장은 FReLU가 ReLU의 sparsity와 계산 효율을 유지하면서도 음수 출력을 허용해 표현력을 확장하고, 그 결과 plain network와 residual network 모두에서 더 빠른 수렴과 더 높은 성능을 낸다는 것이다.  

이 문제가 중요한 이유는 activation function이 단순 비선형성 이상의 역할을 하기 때문이다. activation은 정보가 층 사이를 어떻게 통과하는지, feature가 어떤 분포를 가지는지, batch normalization과 어떤 상호작용을 하는지까지 좌우한다. 이 논문은 ReLU 계열의 장점을 최대한 보존하면서 negative output과 zero-like property를 더 싼 비용으로 도입하려는 시도라는 점에서 의미가 있다. 특히 ELU의 장점 일부를 지수 연산 없이 얻고, BN과도 잘 결합되도록 설계했다는 점이 논문의 실용적 포인트다.  

## 2. Core Idea

핵심 아이디어는 매우 단순하다. ReLU의 출력 절단점을 고정된 0에 두지 말고, **학습 가능한 bias 형태로 이동시키자**는 것이다. 논문은 먼저 다음 식으로 FReLU를 소개한다.

$$
\mathrm{frelu}(x)=\mathrm{relu}(x+a)+b
$$

여기서 $a$ 와 $b$ 는 learnable variable이다. 그런데 저자들은 activation이 보통 convolution 또는 linear layer 뒤에 붙는다는 점을 이용해, $a$ 는 이전 layer의 bias와 함께 흡수될 수 있다고 설명한다. 그래서 실제 구현은 더 단순한 형태가 된다.

$$
\mathrm{frelu}(x)=\mathrm{relu}(x)+b
$$

즉, FReLU는 사실상 **ReLU 출력에 layer-wise learnable offset $b_l$ 를 더하는 구조**다. 이때 forward pass는

$$
\mathrm{frelu}(x)=
\begin{cases}
x+b_l, & x>0 \
b_l, & x\le 0
\end{cases}
$$

가 된다. $b_l=0$이면 정확히 ReLU로 돌아간다.  

이 구조의 의미는 중요하다. ReLU는 각 unit당 사실상 두 상태, 즉 양수 전달 또는 0만 갖는다. 그런데 FReLU에서 $b<0$가 되면 음수 비활성 상태도 meaningful하게 생긴다. 논문은 이때 출력 상태 수가 늘어나며, 성공적으로 학습된 FReLU는 대체로 **음수 bias로 수렴**해 표현력이 좋아진다고 해석한다. 저자들은 이것을 ReLU 대비 더 풍부한 output state를 제공하는 메커니즘으로 본다.  

## 3. Detailed Method Explanation

### 3.1 FReLU의 수식과 동작

ReLU는

$$
\mathrm{relu}(x)=
\begin{cases}
x, & x>0 \
0, & x\le 0
\end{cases}
$$

이다. 반면 FReLU는 음수 영역에서도 완전히 0으로 떨어지지 않고, $b_l$ 라는 학습 가능한 상수 상태를 유지한다.

$$
\mathrm{frelu}(x)=
\begin{cases}
x+b_l, & x>0 \
b_l, & x\le 0
\end{cases}
$$

이 정의 덕분에 negative information을 일정 부분 유지하면서도, 음수 영역의 기울기는 여전히 0이라 ReLU의 sparsity 성질을 잃지 않는다. 이것이 PReLU/LReLU와의 중요한 차이다. PReLU/LReLU는 음수 영역에 slope를 남겨 zero gradient를 피하지만, 그 대신 sparsity가 줄어든다. FReLU는 **negative output은 허용하되 hard rectification 자체는 유지**한다.  

### 3.2 Backpropagation

논문에 따르면 backward pass는 매우 단순하다. 입력에 대한 gradient는 ReLU와 동일하다.

$$
\frac{\partial \mathrm{frelu}(x)}{\partial x}=
\begin{cases}
1, & x>0 \
0, & x\le 0
\end{cases}
$$

그리고 learnable parameter $b_l$ 에 대해서는

$$
\frac{\partial \mathrm{frelu}(x)}{\partial b_l}=1
$$

이다. 즉, FReLU는 gradient flow 측면에서 계산적으로 거의 ReLU와 다르지 않다. 이 점 때문에 지수 연산이 필요한 ELU보다 훨씬 가볍고, 기존 CNN에 쉽게 삽입 가능하다.

### 3.3 왜 음수 bias가 중요한가

논문은 FReLU가 학습 후 음수 값으로 가는 경향을 보인다고 말한다. $b<0$일 때, 음수 입력은 전부 같은 음수 상수 상태로 매핑되고, 양수 입력은 그 위로 선형적으로 펼쳐진다. 저자들은 이를 통해 ReLU보다 더 넓은 feature representation space를 확보할 수 있다고 주장한다. 특히 visualization 실험에서 FReLU를 쓴 경우 feature embedding이 ReLU보다 더 분리되어 보인다고 해석한다.  

### 3.4 기존 activation들과의 차이

논문은 FReLU를 ReLU, PReLU, ELU, SReLU와 비교한다.

* **ReLU**: sparsity와 효율은 좋지만 negative missing이 있다.
* **PReLU/LReLU**: 음수 기울기를 남겨 negative part를 살리지만 sparsity를 잃을 수 있다.
* **ELU**: negative saturation과 zero-like property를 제공하지만 지수 연산이 필요하고 BN과의 호환성이 떨어질 수 있다.
* **SReLU**: 더 복잡한 piecewise 선형 함수지만 구조와 파라미터가 상대적으로 무겁다.

FReLU는 이들 사이에서 “ReLU처럼 단순하지만 음수 상태와 zero-like 성질을 일부 확보한 middle ground”에 가깝다. 특히 ELU 대비 **지수 연산 없이**, PReLU 대비 **sparsity를 유지하며**, BN과도 더 잘 맞는다는 점을 장점으로 든다.  

### 3.5 초기화

저자들은 FReLU용 초기화도 따로 논의한다. 실험에서는 기본적으로 $b=-1$ 로 두고 시작했다고 밝힌다. 이는 음수 비활성 상태를 초기에 열어 두되, 학습이 이를 데이터에 맞게 조정하도록 하려는 선택으로 보인다. 논문은 적절한 initialization이 gradient vanishing 방지에 중요하다고 설명한다.  

## 4. Experiments and Findings

### 4.1 SmallNet on CIFAR-100

가장 먼저 저자들은 CIFAR-100에서 작은 CNN인 SmallNet으로 ReLU, ELU, FReLU를 비교한다. 논문은 FReLU가 **더 빠르게 수렴하고 더 높은 generalization performance**를 보였다고 정리한다. 이 실험은 핵심 메시지를 잘 보여 준다. 즉, FReLU의 이점이 대형 모델이나 특정 residual 구조에만 한정된 것이 아니라, 작은 plain CNN에서도 나타난다는 것이다.  

### 4.2 Batch Normalization과의 호환성

논문이 강하게 밀고 있는 결과 중 하나가 BN compatibility다. 저자들은 ELU가 BN과 함께 쓸 때 성능이 손상될 수 있지만, FReLU는 BN과 잘 결합된다고 주장한다. 실제 CIFAR 실험에서 BN+FReLU 조합이 좋은 결과를 보였고, residual network에서도 “ReLU를 FReLU로 단순 치환”했을 때는 성능이 좋아졌지만 ELU는 오히려 나빠졌다고 설명한다. 저자들은 이를 근거로 **FReLU가 ELU보다 BN과 더 호환적**이라고 해석한다.  

### 4.3 Residual Networks on CIFAR-10/CIFAR-100

Residual network 실험에서도 FReLU는 유효했다. 논문은 CIFAR-10과 CIFAR-100에서 residual bottleneck 구조를 사용해 비교했으며, 단순 치환 설정과 수정된 bottleneck 설정 모두에서 FReLU가 더 좋은 성능을 보였다고 쓴다. 특히 ELU는 residual+BN 설정에서 성능을 해쳤지만 FReLU는 개선을 가져왔다고 명시한다.  

### 4.4 MNIST feature visualization

MNIST의 LeNet++ 기반 시각화 실험은 정량 성능보다 표현력 해석에 초점을 둔다. 마지막 hidden layer를 2차원으로 두고 ReLU와 FReLU의 embedding을 비교했을 때, FReLU의 embedding이 더 분리되어 보였고, 정확도도 **97.8% 대 97.05%** 로 더 높았다. 저자들은 이를 “negative bias가 더 큰 feature representation space를 제공한다”는 정성적 증거로 본다.  

### 4.5 ImageNet

ImageNet 실험에서는 논문 표에서 ReLU가 53.00, PReLU가 52.20, ELU와 FReLU가 51.20으로 제시된다. 표 snippet만으로는 top-1/top-5의 정확한 metric label이 완전히 드러나지 않지만, 적어도 visible 결과 기준으로는 **FReLU가 ReLU보다 낮은 error를 기록했고 ELU와 동급 최고 수준**이었다고 볼 수 있다. 논문 결론부는 전반적으로 plain/residual network 모두에서 더 높은 성능을 보였다고 요약한다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 아이디어가 극도로 단순하다는 점이다. FReLU는 사실상 **ReLU + learnable bias** 수준이라 구현과 계산 비용이 거의 없다. 그럼에도 negative information, zero-like property, BN compatibility라는 세 가지 실용적 장점을 동시에 노린다. 또한 backward gradient가 ReLU와 같아 optimization 측면에서도 부담이 적다.  

또 다른 강점은 ELU와의 차별화다. ELU가 음수 영역과 zero-like property를 제공하지만 BN과의 결합이 까다롭고 지수 연산이 필요한 반면, FReLU는 더 가볍고 residual+BN 환경에서 실제로 더 잘 작동했다고 주장한다. 이는 CNN 실무 관점에서 꽤 설득력 있는 포인트다.

### 한계

한계도 분명하다. 첫째, FReLU는 음수 영역의 기울기를 열지 않기 때문에, negative region에서 gradient가 여전히 0이다. 즉, dead ReLU 문제를 PReLU처럼 직접 해결하는 방식은 아니다. 둘째, 핵심 자유도는 layer-wise bias 하나뿐이라 표현력 확장은 있지만, Dynamic ReLU나 더 유연한 piecewise activation들에 비하면 적응성은 제한적이다. 셋째, visible 실험 기준으로는 주로 CNN classification 중심이며, 다른 modality나 sequence model로의 일반화는 다루지 않는다.

### 해석

비판적으로 보면, 이 논문의 진짜 메시지는 “복잡한 activation이 꼭 필요한 것은 아니다”라는 점이다. 때로는 ReLU의 절단 구조는 유지한 채 **출력 기준점만 조금 이동**시켜도 표현력과 최적화 특성이 눈에 띄게 달라질 수 있다. 즉, activation 설계에서 slope만이 아니라 **baseline level** 자체도 중요한 설계 축이라는 점을 보여 준 논문으로 읽을 수 있다.

## 6. Conclusion

이 논문은 ReLU의 zero-hard rectification이 negative information을 버린다는 문제를 지적하고, 이를 해결하기 위해 **FReLU (Flexible Rectified Linear Unit)** 를 제안했다. FReLU는 본질적으로 ReLU 출력에 layer-wise learnable bias를 더하는 구조이며,

$$
\mathrm{frelu}(x)=
\begin{cases}
x+b_l, & x>0 \
b_l, & x\le 0
\end{cases}
$$

로 쓸 수 있다. 이 단순한 수정만으로 FReLU는 ReLU의 sparsity와 계산 효율을 유지하면서도 음수 출력 상태와 zero-like 성질을 일부 제공한다.

실험적으로 저자들은 CIFAR-10, CIFAR-100, ImageNet에서 FReLU가 plain network와 residual network 모두에서 빠른 수렴과 더 높은 성능을 보였다고 보고했다. 특히 residual+BN 환경에서 ELU보다 더 호환적이라는 점을 강하게 주장했다. 따라서 FReLU는 “복잡하지 않은 activation 개선”의 대표 예로, ReLU 계열 연구 흐름 안에서 실용성이 높은 변형으로 볼 수 있다.  
