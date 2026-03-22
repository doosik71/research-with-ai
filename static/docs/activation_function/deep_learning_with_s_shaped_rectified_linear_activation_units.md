# Deep Learning with S-shaped Rectified Linear Activation Units

## 1. Paper Overview

이 논문은 CNN에서 널리 쓰이는 ReLU 계열 activation function의 표현 한계를 문제로 삼는다. 저자들은 ReLU, Leaky ReLU, PReLU, maxout 같은 기존 rectified activation이 대체로 **convex 함수만 잘 표현**하거나, APL처럼 non-convex를 근사하더라도 불필요한 제약을 갖는다고 본다. 이를 해결하기 위해 저자들은 **S-shaped Rectified Linear Unit (SReLU)**를 제안한다. SReLU는 세 개의 piecewise linear segment와 네 개의 학습 가능한 파라미터로 구성되며, backpropagation으로 네트워크 전체와 함께 학습된다. 논문의 핵심 주장은 SReLU가 **convex와 non-convex 함수를 모두 유연하게 학습**하면서도, 비포화 piecewise linear 구조이기 때문에 ReLU류의 계산 효율과 gradient 전달 장점을 유지한다는 것이다.

이 문제가 중요한 이유는 activation function이 단순한 부품이 아니라, 네트워크의 표현력과 optimization dynamics를 직접 제한하기 때문이다. 저자들은 특히 deep CNN의 성공을 비포화 activation 덕분이라고 보면서도, 단순 ReLU류는 표현 가능한 비선형성의 형태가 제한적이라고 지적한다. 따라서 이 논문은 “더 깊은 모델”이나 “더 복잡한 block”이 아니라, **activation 자체를 더 유연하게 만들어 성능을 끌어올릴 수 있는가**라는 질문에 답한다.  

## 2. Core Idea

핵심 아이디어는 SReLU를 통해 하나의 neuron이 입력 크기에 따라 서로 다른 세 개의 선형 구간을 사용하게 만드는 것이다. 즉, 일반적인 ReLU처럼 한 번 꺾이는 구조가 아니라, **좌측 구간, 중앙 구간, 우측 구간**을 가진 S자 형태의 piecewise linear function을 학습한다. 이 구조 덕분에 SReLU는 단순한 convex rectifier가 아니라, 상황에 따라 non-convex 형태도 근사할 수 있다. 저자들은 이를 통해 기존 ReLU/LReLU/PReLU/maxout의 한계를 넘고자 한다.  

또 하나의 핵심은 영감의 출처다. 저자들은 psychophysics와 neural science의 두 기본 법칙인 **Weber-Fechner law**와 **Stevens law**를 동기로 제시한다. 전자는 로그 형태, 후자는 power-law 형태의 지각 반응을 설명하는데, 저자들은 SReLU의 양의 구간이 이런 형태를 piecewise linear로 근사하도록 설계될 수 있다고 본다. 그래서 SReLU는 단순히 “꺾이는 선을 하나 더 추가한 activation”이 아니라, **입력 강도에 대한 반응 곡선을 더 유연하게 모델링하는 activation**으로 제안된다.

## 3. Detailed Method Explanation

### 3.1 SReLU의 구조

논문에 따르면 SReLU는 세 개의 선형 함수 조합으로 정의되며, 네 개의 학습 파라미터로 제어된다. 직관적으로 쓰면 다음 형태다.

$$
h(x_i)=
\begin{cases}
a_i^r(x_i-t_i^r)+t_i^r, & x_i \ge t_i^r \
x_i, & t_i^l < x_i < t_i^r \
a_i^l(x_i-t_i^l)+t_i^l, & x_i \le t_i^l
\end{cases}
$$

여기서 $t_i^l, t_i^r$ 는 좌우 threshold, $a_i^l, a_i^r$ 는 좌우 기울기다. 중앙 구간에서는 unit slope를 유지하고, 좌우 구간에서는 학습된 slope로 입력을 압축하거나 확장한다. 이 단순한 구조만으로도 SReLU는 다양한 함수 모양을 만들 수 있다.  

### 3.2 왜 기존 activation보다 유연한가

ReLU, LReLU, PReLU는 모두 본질적으로 convex 계열로 해석된다. 논문은 SReLU의 파라미터를 특정 값으로 두면 ReLU/LReLU/PReLU가 SReLU의 special case가 된다고 설명한다. 예를 들어 적절한 $t_i^r, a_i^r, t_i^l, a_i^l$ 를 선택하면 ReLU 또는 LReLU/PReLU로 퇴화한다. 하지만 SReLU는 여기에 머무르지 않고, 파라미터를 자유롭게 학습시켜 **non-convex 함수까지 근사**할 수 있다.  

APL과의 차이도 중요하다. APL은 여러 hinge를 합쳐 non-convex를 근사할 수 있지만, 논문은 APL이 **오른쪽 끝 선형 구간의 slope와 bias를 사실상 고정**해 버린다고 비판한다. 반면 SReLU는 오른쪽 구간도 자유롭게 학습하므로, 큰 입력에 대해 출력 증가를 억제하거나 완만하게 만들 수 있다. 저자들은 이것이 Weber-Fechner law의 로그형 반응을 더 잘 흉내 내는 핵심 차이라고 주장한다.  

### 3.3 학습과 파라미터 수

SReLU는 backpropagation으로 네트워크와 함께 end-to-end로 학습된다. 중요한 점은 저자들이 **각 channel마다 독립적인 SReLU**를 둔다는 것이다. 그럼에도 추가 파라미터는 channel당 4개뿐이라 전체 오버헤드는 작다. 논문은 deep network 전체에서 SReLU에 필요한 파라미터 수가 매우 작고, 계산비용도 무시할 만하다고 강조한다.  

### 3.4 freezing 기반 초기화

이 논문의 실질적인 기여 중 하나는 초기화 방식이다. SReLU는 파라미터가 더 많기 때문에, threshold와 slope를 처음부터 잘못 잡으면 학습이 불안정해질 수 있다. 특히 층마다 activation 입력의 분포 크기가 크게 다르기 때문에 hand-tuned initialization은 어렵다고 저자들은 말한다. 이를 해결하기 위해 초반 몇 epoch 동안 SReLU를 **사전 정의된 Leaky ReLU로 퇴화시켜 frozen 상태로 두고**, 그 후 네트워크가 관찰한 실제 activation 분포를 바탕으로 적절한 초기값을 얻어 학습 가능 상태로 전환하는 “freezing” 방법을 제안한다.  

이 초기화 전략은 단순 편의 기능이 아니라 SReLU를 실제 deep network에 꽂아 넣을 수 있게 만드는 핵심 장치다. 논문은 특히 이 전략이 없으면 좋은 $t_i^r$ 와 $a_i^r$ 를 수동으로 주기 어렵다고 설명한다.

## 4. Experiments and Findings

### 4.1 실험 설정

저자들은 네 개의 데이터셋에서 실험한다.

* CIFAR-10
* CIFAR-100
* MNIST
* ImageNet

아키텍처는 CIFAR-10, CIFAR-100, MNIST에는 **Network in Network (NIN)**, ImageNet에는 **GoogLeNet**을 사용한다. 실험에서는 원래 네트워크의 ReLU만 SReLU로 교체하고, 나머지 구조와 하이퍼파라미터는 그대로 유지한다. validation set은 각 데이터셋 train의 20%를 랜덤 샘플링해 사용한다.  

이 설정은 논문의 주장에 잘 맞는다. 즉, 성능 향상이 네트워크 전체 재설계 때문이 아니라 **activation 교체 자체의 효과**임을 보여 주려는 설계다.

### 4.2 CIFAR-10 / CIFAR-100

논문은 SReLU가 CIFAR-10과 CIFAR-100에서 다른 activation보다 일관되게 좋은 결과를 냈다고 보고한다. 특히 저자들은 APL 대비 성능 차이를 강조한다. **데이터 증강과 제안한 initialization이 없는 조건에서도**, NIN + SReLU는 NIN + APL보다 CIFAR-10에서 **0.98%**, CIFAR-100에서 **3.04%** 더 좋았다고 서술한다. 이는 단순히 non-convex를 표현할 수 있다는 것만이 아니라, **오른쪽 구간을 자유롭게 학습하는 설계**가 실제 성능 차이로 이어졌다는 저자들의 핵심 증거다.  

또한 convergence curve도 제시되어 SReLU가 학습 과정에서 안정적이며 경쟁 activation 대비 우수한 추세를 보인다고 설명한다.

### 4.3 MNIST

MNIST에서도 NIN 기반 실험을 수행하며, 논문은 SReLU가 scale이 작은 데이터셋에서도 성능 향상을 보였다고 정리한다. 첨부된 본문 조각에는 MNIST 세부 수치표 전체가 완전하게 드러나지 않지만, 저자들의 결론은 MNIST 포함 네 개 데이터셋 전반에서 SReLU가 효과적이라는 것이다. 따라서 여기서는 구체적 수치보다 **일관된 개선 경향**이 더 중요한 메시지다.  

### 4.4 ImageNet

가장 강한 실험은 ImageNet이다. 저자들은 1000-class ImageNet에서 **GoogLeNet + SReLU**를 평가했고, baseline은 Caffe에서 공개된 원래 GoogLeNet이다. 결과적으로 SReLU를 적용한 GoogLeNet은 ReLU 기반 원본보다 **1.24% 향상**을 얻었다고 보고한다. 추가 파라미터는 **21.6K**뿐인데, 이는 원본 GoogLeNet의 약 **5M** 파라미터에 비해 매우 작다. 즉, 아주 작은 overhead로 의미 있는 대규모 성능 향상을 얻었다는 것이 논문의 강한 주장이다.

이 실험은 특히 중요하다. CIFAR 계열 개선은 toy-scale로 볼 수 있지만, ImageNet과 GoogLeNet에서 효과가 있다는 것은 SReLU가 단순 아이디어가 아니라 **실제 대형 CNN에도 꽂아 넣을 수 있는 activation**임을 보여 준다.

### 4.5 파라미터 학습 결과 해석

논문은 학습된 SReLU 파라미터도 분석한다. 일부 층에서는 $a^r$ 가 거의 1에 가깝고, 다른 층에서는 threshold가 커지며 보다 복잡한 함수 형태를 학습한다. 특히 높은 층에서 $t^r$ 가 커지는 현상은, 상위 층 입력 평균이 더 크기 때문에 그에 맞춰 activation이 다른 응답 곡선을 학습한 결과로 해석된다. 이는 SReLU가 실제로 층별로 다른 비선형 형태를 자동으로 선택하고 있음을 보여 준다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **표현력과 단순성의 균형**이다. SReLU는 세 구간 piecewise linear라는 아주 단순한 구조이지만, ReLU/PReLU보다 훨씬 넓은 함수 family를 표현한다. 게다가 maxout처럼 큰 파라미터 증가를 요구하지도 않는다. 이 때문에 deep CNN에 넣기 쉽고, 계산 효율도 좋다.  

또한 activation 설계를 단순 heuristic이 아니라 psychophysics law와 연결해 설명하려는 점도 흥미롭다. 물론 이 연결은 엄밀한 생물학 모델링이라기보다 영감 수준이지만, 적어도 “왜 오른쪽 구간도 adaptive해야 하는가”를 설득력 있게 설명하는 역할을 한다.  

### 한계

한계도 있다. 첫째, SReLU의 우수성은 activation 자체만이 아니라 **freezing initialization**에 부분적으로 의존한다. 즉, 이 방법은 함수 정의만 제시한 것이 아니라 초기화 트릭까지 포함한 패키지다. 둘째, 논문의 비교 대상은 당시 CNN activation들이 중심이며, 이후 등장한 ELU/GELU/Swish류와의 비교는 없다. 이는 논문의 시대적 한계다. 셋째, psychophysics law를 동기로 제시하지만, 그것이 실제 최적화 성능과 직접적으로 연결된다는 엄밀한 증명은 없다.  

### 해석

비판적으로 보면, 이 논문의 핵심 가치는 “S자 모양 activation 하나를 제안했다”에만 있지 않다. 더 중요한 메시지는 **activation의 양 끝 구간까지 학습 가능하게 만들면 deep CNN의 표현력이 실제로 좋아질 수 있다**는 점이다. ReLU 이후 activation research가 단순히 “negative slope를 열어 주는가”에서 “shape 전체를 얼마나 적응적으로 만들 수 있는가”로 확장되는 흐름 속에서, 이 논문은 꽤 중요한 중간 단계로 읽힌다.

## 6. Conclusion

이 논문은 deep CNN을 위한 새로운 activation인 **SReLU**를 제안했다. SReLU는 세 개의 piecewise linear 구간과 네 개의 학습 파라미터로 구성되며, ReLU/LReLU/PReLU보다 더 유연하게 **convex와 non-convex 함수 모두를 근사**할 수 있다. 또한 제안된 freezing 초기화 덕분에 기존 네트워크에 안정적으로 삽입할 수 있고, 실험에서는 NIN과 GoogLeNet을 사용해 CIFAR-10, CIFAR-100, MNIST, ImageNet 전반에서 성능 향상을 보였다. 특히 ImageNet에서는 적은 추가 파라미터로 1.24% 개선을 보고했다.  

실무적으로는, 이 논문은 “activation을 좀 더 expressive하게 만드는 것만으로도 대형 CNN 성능을 끌어올릴 수 있다”는 사례다. 연구적으로는 ReLU의 성공 이후, activation function을 더 유연한 learnable piecewise linear family로 확장한 중요한 작업으로 볼 수 있다.
