# Instance-sensitive Fully Convolutional Networks

- **저자**: Jifeng Dai, Kaiming He, Yi Li, Shaoqing Ren, Jian Sun
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1603.08678

## 1. 논문 개요

이 논문의 목표는 기존의 Fully Convolutional Network (FCN)가 잘 처리하던 semantic segmentation을 넘어, **개별 객체 인스턴스(instance) 단위의 segment proposal**을 end-to-end 방식으로 생성하는 것이다. 기존 FCN은 픽셀마다 “이 클래스인가?”를 예측하는 데는 강하지만, 같은 클래스의 물체가 여러 개 있을 때 그것들을 서로 다른 인스턴스로 분리하지 못한다. 예를 들어 이미지 안에 사람 두 명이 있으면, 둘을 모두 사람으로는 맞게 표시해도 “어느 픽셀이 어느 사람에 속하는가”는 구분하지 못한다.

논문이 다루는 연구 문제는 바로 이 지점이다. 즉, **fully convolutional한 구조를 유지하면서도 instance-aware한 mask proposal을 만들 수 있는가**가 핵심 질문이다. 이는 당시 instance segmentation 파이프라인들이 외부 segment proposal 기법이나, 큰 fully connected layer를 사용한 mask regression에 의존하던 흐름과 대비된다.

이 문제가 중요한 이유는, instance segmentation이 detection과 segmentation의 중간에 있는 매우 중요한 문제이기 때문이다. 실제 응용에서는 단순히 “여기 자동차 클래스가 있다”가 아니라 “이 자동차, 저 자동차”처럼 객체를 개별적으로 분리해야 한다. 논문은 이 문제를 해결하면서도 FCN의 장점인 **dense prediction, parameter sharing, end-to-end 학습, 효율성**을 유지하려고 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **instance-sensitive score maps**이다. 기존 FCN이 클래스별 score map 하나를 만드는 방식이었다면, 이 논문은 객체 내부의 **상대적 위치(relative position)** 를 나타내는 여러 개의 score map을 만든다. 예를 들어 $3 \times 3$ grid를 쓰면, 객체의 좌상단, 상단, 우상단, 좌중단, 중심, 우중단, 좌하단, 하단, 우하단에 해당하는 총 9개의 relative position classifier를 학습한다.

직관적으로 보면, 픽셀이 “사람인가?”만 묻는 대신 “이 픽셀은 어떤 객체의 오른쪽 부분인가?”, “왼쪽 위 부분인가?”를 묻는 것이다. 이렇게 하면 서로 붙어 있는 두 객체가 있어도, 각 픽셀이 어느 인스턴스의 어느 상대 위치에 대응되는지로 분리 가능성이 생긴다. 논문은 이 구조를 바탕으로 sliding window마다 해당 위치의 조각들을 모아 하나의 instance mask를 조립한다.

기존 접근과의 가장 중요한 차별점은 DeepMask와의 대비에서 드러난다. DeepMask는 각 window를 입력으로 받아 $m \times m$ 해상도의 mask를 직접 예측하기 위해 $m^2$ 차원의 큰 fully connected layer를 사용한다. 반면 이 논문은 **mask 해상도에 비례하는 고차원 파라미터 층이 없다**. 대신 자연영상의 **local coherence**를 활용하여, 인접한 window들 사이에서 비슷한 픽셀 예측이 재사용된다는 점을 구조적으로 반영한다. 즉, “고해상도 mask를 직접 회귀(regress)하는” 대신, “위치 의미를 갖는 저차원 score map들을 조립하는” 방식으로 문제를 푼다.

## 3. 상세 방법 설명

전체 시스템은 크게 두 가지 fully convolutional branch로 구성된다. 첫 번째 branch는 instance-sensitive score maps를 생성하고, 두 번째 branch는 각 sliding window가 실제 객체 인스턴스를 포함하는지를 나타내는 **objectness score**를 예측한다. 이후 assembling module이 첫 번째 branch의 출력들을 결합해 각 위치별 instance mask를 만든다.

### 3.1 Instance-sensitive score maps

논문은 상대적 위치를 $k \times k$ regular grid로 정의한다. 예를 들어 $k=3$이면 총 $k^2=9$개의 위치가 생긴다. 네트워크의 마지막 convolution layer는 바로 이 $k^2$개의 채널을 출력하며, 각 채널은 특정 상대 위치에 대한 score map이다. 각 픽셀은 단순한 semantic class classifier가 아니라, “어떤 인스턴스의 특정 위치인가 아닌가”를 판별하는 classifier 역할을 한다.

이 아이디어의 중요한 점은, 각 픽셀 예측이 여전히 low-dimensional classifier라는 것이다. 즉 FCN의 철학을 유지한다. DeepMask처럼 각 window마다 큰 mask vector 전체를 예측하는 것이 아니라, 모든 위치에서 공유되는 규칙적이고 구조적인 예측을 수행한다.

### 3.2 Assembling module

score map만으로는 아직 완성된 object instance가 아니다. 이를 실제 mask로 만드는 것이 assembling module이다. 논문은 feature map 위에서 해상도 $m \times m$의 sliding window를 움직인다. 그리고 각 window를 $k \times k$개의 sub-window로 나눈다. 각 sub-window는 그에 대응하는 relative position score map의 같은 영역 값을 그대로 복사해 온다.

즉, $k^2$개의 score map 각각에서 일부를 가져와 하나의 $m \times m$ mask를 조립한다. 예를 들어 오른쪽 위 sub-window는 “right-top” score map의 해당 부분에서 복사된다. 이렇게 하면 한 sliding window마다 하나의 instance candidate가 생성된다.

중요한 점은 이 assembling module이 **mask 해상도 $m \times m$와 관련은 있지만 학습 파라미터는 전혀 없다**는 것이다. 단순 copy-and-paste 연산이므로 계산량이 가볍고, mask 크기가 커져도 별도의 거대한 파라미터 층이 필요하지 않다.

### 3.3 Local coherence 해석

논문은 이 설계를 local coherence 관점에서 정당화한다. 자연영상에서는 인접한 두 sliding window가 거의 같은 장면을 보기 때문에, 어떤 픽셀의 예측은 window가 조금 움직여도 크게 달라지지 않는 경우가 많다. 이 방법에서는 인접 window가 같은 score map을 재사용하므로 동일 픽셀에 대한 예측이 상당 부분 공유된다. 반면 DeepMask의 sliding fully connected layer 방식은 window가 한 칸만 움직여도 같은 픽셀이 다른 채널에 의해 다시 예측되므로, 파라미터 낭비가 크고 일관성이 구조적으로 보장되지 않는다.

논문은 이 차이가 단순 계산 효율의 문제가 아니라, 특히 PASCAL VOC처럼 데이터가 작은 환경에서 **과적합 위험을 줄이는 핵심 설계 차이**라고 주장한다.

### 3.4 네트워크 구조

feature extractor로는 ImageNet으로 사전학습된 VGG-16을 사용한다. 다만 stride를 줄여 더 높은 해상도의 feature map을 얻기 위해, `pool4`의 stride를 2에서 1로 바꾸고 `conv5_1`부터 `conv5_3`까지는 hole algorithm을 적용한다. 그 결과 `conv5_3` feature map의 effective stride는 입력 이미지 기준 $s=8$ 픽셀이 된다.

instance branch에서는 먼저 $1 \times 1$의 512차원 convolution + ReLU를 적용하고, 그 다음 $3 \times 3$ convolution으로 $k^2$개의 instance-sensitive score maps를 출력한다. 실험에서 실제 mask 조립에 사용하는 window 크기는 feature map 기준 $m=21$이다.

objectness branch에서는 $3 \times 3$ 512차원 convolution + ReLU 뒤에 $1 \times 1$ convolution을 둔다. 이 마지막 층은 각 픽셀 위치를 중심으로 하는 sliding window가 instance인지 아닌지를 판별하는 logistic regression이다. 따라서 출력은 objectness score map이 된다.

### 3.5 학습 목표와 손실 함수

학습은 end-to-end로 수행된다. forward pass 후 전체 sliding window 중 일부만 샘플링하여 손실을 계산한다. 한 이미지당 256개의 sliding window를 random sampling하고, positive와 negative의 비율은 1:1로 맞춘다.

손실 함수는 다음과 같다.

$$
\sum_i \left( L(p_i, p_i^*) + \sum_j L(S_{i,j}, S_{i,j}^*) \right)
$$

여기서 $i$는 샘플링된 window 인덱스이다. $p_i$는 해당 window의 predicted objectness score이고, $p_i^*$는 그 window가 positive sample이면 1, 아니면 0이다. $S_i$는 assembling module로 조립된 predicted segment instance이고, $S_i^*$는 ground-truth segment instance이다. $j$는 window 내부 픽셀 인덱스이다. $L$은 logistic regression loss라고 논문이 명시한다.

즉 손실은 두 부분으로 나뉜다. 하나는 “이 window가 객체인가?”를 맞추는 objectness loss이고, 다른 하나는 “그 객체의 mask 픽셀들이 맞는가?”를 맞추는 mask loss이다. 중요한 점은 mask loss가 조립된 결과 $S_i$ 위에서 직접 계산된다는 점이다. 즉 assembling module도 학습 그래프 안에 포함된다.

### 3.6 학습 및 추론 절차

학습 시 입력 이미지는 arbitrary size를 받을 수 있고, 짧은 변 길이를 $600 \times 1.5^{\{-4,-3,-2,-1,0,1\}}$ 픽셀 중 하나로 랜덤하게 선택하는 scale jittering을 적용한다. 최적화는 SGD를 사용하며, 총 40k iteration을 수행한다. learning rate는 처음 32k iteration 동안 0.001, 마지막 8k 동안 0.0001이다. weight decay는 0.0005, momentum은 0.9다. 구현은 8-GPU이며, GPU당 1개 이미지와 그 이미지에서 샘플링한 256 windows를 처리한다.

추론 시에는 입력 이미지 전체에 대해 forward pass 한 번으로 instance-sensitive score maps와 objectness score map을 얻는다. 그 후 assembling module이 모든 위치에서 dense sliding을 수행해 instance mask를 생성한다. 각 mask는 objectness score를 함께 가진다. 여러 스케일에서 같은 과정을 반복한 뒤, 각 output mask를 binary mask로 thresholding한다. 이후 binary mask의 tight bounding box 간 IoU를 이용해 NMS를 수행하며, 임계값은 0.8이다. 최종적으로 상위 $N$개 proposal을 출력한다. 논문은 K40 GPU에서 이미지당 총 1.5초가 걸린다고 보고한다.

## 4. 실험 및 결과

## 4.1 PASCAL VOC 2012

PASCAL VOC 2012에서는 segmentation annotation을 활용해 train set으로 학습하고 validation set에서 평가했다. 평가는 predicted instance와 ground-truth instance 사이의 **mask-level IoU**를 사용한다. proposal quality는 DeepMask를 따라 **Average Recall (AR)** 로 측정하며, IoU threshold 0.5부터 1.0까지의 recall 곡선 면적을 쓴다. 제안 개수 제한에 따라 AR@10, AR@100, AR@1000을 보고한다.

### Relative position 수에 대한 ablation

$k^2$ 값을 바꿔 실험한 결과는 다음과 같다.

- $3^2$: AR@10 38.3, AR@100 49.2, AR@1000 52.1
- $5^2$: AR@10 38.9, AR@100 49.7, AR@1000 52.6
- $7^2$: AR@10 38.8, AR@100 49.7, AR@1000 52.7

이 결과는 성능이 $k$ 값에 매우 민감하지 않음을 보여준다. $k=3$만으로도 이미 강한 성능을 내며, 더 세밀한 위치 분할은 약간의 이득만 주고 $k=5$ 정도에서 거의 포화된다. 이후 실험에서는 $k=5$를 사용한다.

### DeepMask 방식과의 ablation 비교

논문은 공정 비교를 위해 PASCAL VOC에서 DeepMask와 유사한 baseline을 직접 구현한다. 이 baseline은 VGG-16 위에 추가 convolution을 두고, $14 \times 14$ feature map을 만든 뒤 512차원 fully connected layer와 $56^2$ 차원의 fully connected layer로 $56 \times 56$ mask를 예측한다. 이 두 fc layer의 파라미터 수는 53M이다.

비교 결과는 다음과 같다.

- `~DeepMask` (`crop 224x224`, sliding fc): AR@10 31.2, AR@100 42.9, AR@1000 47.0
- 제안 방법 (`crop 224x224`, fully conv): AR@10 37.4, AR@100 48.4, AR@1000 51.4
- 제안 방법 (`full image`, fully conv): AR@10 38.9, AR@100 49.7, AR@1000 52.6

즉, 같은 crop 기반 학습 조건에서도 제안 방법이 `~DeepMask`보다 크게 좋다. 전체 이미지를 fully convolutional하게 학습하면 성능이 더 좋아진다. 이는 제안 구조가 단순히 파라미터 효율적일 뿐 아니라, 학습 방식 자체에서도 이점을 가진다는 점을 시사한다.

파라미터 차이는 매우 크다. 논문에 따르면 제안 방법의 마지막 $k^2$-channel convolution layer는 약 0.1M 파라미터로, DeepMask mask generation fc layers 대비 약 1/500 규모다. 이는 논문의 핵심 주장, 즉 local coherence를 활용하면 고차원 mask regression 없이도 충분히 좋은 proposal을 만들 수 있다는 점을 강하게 뒷받침한다.

### 다른 proposal 방법과의 비교

PASCAL VOC validation set에서의 비교 결과는 다음과 같다.

- SS: AR@10 7.0, AR@100 23.5, AR@1000 43.3
- MCG: AR@10 18.9, AR@100 36.8, AR@1000 49.5
- `~DeepMask`: AR@10 31.2, AR@100 42.9, AR@1000 47.0
- MNC: AR@10 33.4, AR@100 48.5, AR@1000 53.8
- 제안 방법: AR@10 38.9, AR@100 49.7, AR@1000 52.6

bottom-up proposal인 SS, MCG보다 CNN 기반 방법들이 전반적으로 좋다. 제안 방법은 특히 **AR@10에서 38.9**로 가장 높아, 적은 수의 proposal만 사용할 때 매우 강하다는 점이 눈에 띈다. AR@100과 AR@1000에서는 MNC와 비슷한 수준이며, AR@1000에서는 MNC가 약간 높다. 즉, 제안 방법은 proposal ranking의 앞부분 품질이 특히 좋다고 해석할 수 있다.

### Instance semantic segmentation 평가

proposal 자체뿐 아니라 downstream classifier와 결합한 instance semantic segmentation도 평가한다. classifier 구조는 MNC의 stage 3를 사용하고, proposal은 본 논문 방법으로 생성한다. 학습은 joint training이 아니라 two-step training이다. proposal 수는 $N=300$을 사용한다.

결과는 다음과 같다.

- SDS + MCG: mAP@0.5 49.7, mAP@0.7 25.3
- Hypercolumn + MCG: mAP@0.5 60.0, mAP@0.7 40.4
- CFM + MCG: mAP@0.5 60.7, mAP@0.7 39.6
- MNC + MNC proposal: mAP@0.5 63.5, mAP@0.7 41.5
- MNC classifier + 제안 proposal: mAP@0.5 61.5, mAP@0.7 43.0

여기서 중요한 결과는 **mAP@0.7에서 43.0으로 최고 성능**을 기록했다는 점이다. IoU threshold가 높은 설정에서 성능이 더 좋다는 것은, proposal mask의 경계 품질이나 정밀한 localization이 우수하다는 뜻으로 해석할 수 있다. 반면 mAP@0.5는 MNC보다 낮다. 논문은 MNC가 joint training을 수행하는 반면, 본 실험은 two-step training이라는 점을 함께 언급한다.

## 4.2 MS COCO

MS COCO에서는 80k training images로 학습하고, DeepMask 논문을 따라 validation set 첫 5k images에서 평가한다. 동일한 multiple-scale 설정을 사용해 공정 비교를 시도한다.

결과는 다음과 같다.

- GOP: AR@10 2.3, AR@100 12.3, AR@1000 25.3
- Rigor: AR@100 9.4, AR@1000 25.3
- SS: AR@10 2.5, AR@100 9.5, AR@1000 23.0
- MCG: AR@10 7.7, AR@100 18.6, AR@1000 29.9
- DeepMask: AR@10 12.6, AR@100 24.5, AR@1000 33.1
- DeepMaskZoom: AR@10 12.7, AR@100 26.1, AR@1000 36.6
- 제안 방법: AR@10 16.6, AR@100 31.7, AR@1000 39.2

COCO처럼 더 어렵고 다양한 데이터셋에서도 제안 방법은 DeepMask와 DeepMaskZoom을 모두 안정적으로 앞선다. 특히 AR@100 기준으로 31.7을 기록해 DeepMaskZoom의 26.1보다 유의미하게 높다. 이는 제안한 fully convolutional, instance-sensitive 설계가 작은 데이터셋에서만 유리한 것이 아니라 대규모 데이터셋에서도 충분히 경쟁력이 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **instance segmentation proposal 문제를 FCN의 연장선 위에서 자연스럽게 재구성했다**는 점이다. 단순히 큰 출력층으로 mask를 직접 회귀하는 대신, relative position 기반 score map이라는 구조적 표현을 도입해 파라미터를 크게 줄였다. 이는 아이디어 측면에서도 깔끔하고, 실험적으로도 DeepMask 대비 훨씬 적은 파라미터로 더 좋은 성능을 보였다는 점에서 설득력이 있다.

두 번째 강점은 **local coherence를 명시적으로 활용한 설계 논리**다. 논문은 왜 fully convolutional한 instance mask prediction이 가능한지에 대해 단순 경험적 주장에 머무르지 않고, 인접 window 간 예측 재사용 가능성이라는 관점에서 설명한다. 이 설명은 DeepMask와의 구조 차이를 이해하는 데 매우 유용하다.

세 번째 강점은 실험 결과의 성격이다. PASCAL VOC에서는 적은 proposal 수에서 강하고, COCO에서는 대규모 데이터셋에서도 성능 우위를 보인다. 또한 downstream instance segmentation에서 mAP@0.7이 높다는 점은 proposal의 정밀도가 높다는 실질적 장점으로 읽힌다.

반면 한계도 분명하다. 첫째, 이 논문은 proposal 생성에 초점이 있으며, 최종적인 category-aware instance segmentation 전체 시스템을 완전히 새로 설계한 것은 아니다. classifier는 MNC stage 3를 가져와 결합한다. 따라서 “proposal 방법”으로서의 공헌은 명확하지만, end-to-end instance segmentation 전체 프레임워크로서의 완결성은 제한적이다.

둘째, relative position을 고정된 $k \times k$ grid로 정의하는 방식은 단순하고 효율적이지만, 복잡한 형상이나 비정형 객체에서 이 분할이 얼마나 최적인지는 논문이 깊게 분석하지 않는다. $k$에 대한 ablation은 있지만, 왜 특정 객체 구조에서 이 표현이 잘 동작하는지에 대한 이론적 분석은 제한적이다.

셋째, NMS가 mask IoU가 아니라 **tight bounding box의 box-level IoU**를 기반으로 수행된다는 점도 다소 단순하다. 이는 구현상 효율적이지만, 진짜 mask 중복을 더 정확히 반영하는 설계는 아니다. 논문은 이를 실용적 선택으로 사용하지만, 여기서 생기는 trade-off는 깊게 논의하지 않는다.

넷째, 논문은 qualitative figure를 제시하지만 failure mode를 체계적으로 분해하지는 않는다. 예를 들어 작은 객체, 심한 occlusion, 매우 밀집된 장면에서 relative position score map 방식이 어떤 실패를 보이는지 자세한 유형별 분석은 없다. 따라서 실제 적용 관점에서는 어떤 장면에서 특히 강하거나 약한지 추가 검토가 필요하다.

## 6. 결론

이 논문은 **InstanceFCN**이라는 구조를 통해, semantic segmentation용 FCN을 instance segment proposal로 확장하는 설계 원리를 제시한다. 핵심 기여는 relative position에 기반한 **instance-sensitive score maps**와, 이를 조립해 mask를 만드는 **parameter-free assembling module**이다. 이를 통해 mask 해상도에 비례하는 거대한 fully connected layer 없이도 강력한 instance proposal 성능을 달성했다.

실험적으로는 PASCAL VOC와 MS COCO 모두에서 경쟁력 있는 결과를 보였고, 특히 DeepMask 대비 훨씬 적은 파라미터로 더 좋은 성능을 보여 설계의 효율성과 타당성을 입증했다. 또한 높은 IoU 기준에서 우수한 instance segmentation 성능을 보여, 제안 방법이 단순 proposal recall뿐 아니라 정밀한 mask quality 측면에서도 의미가 있음을 보여준다.

향후 연구 관점에서 이 논문은, instance segmentation을 반드시 고차원 mask regression이나 region-wise fully connected prediction으로만 풀 필요는 없다는 점을 보여준다. 즉 **structured dense prediction**의 방향으로 instance-level 문제를 다시 설계할 수 있다는 가능성을 열었다. 실제 후속 연구에서는 이 아이디어가 position-sensitive representation, fully convolutional instance segmentation 계열 방법들로 이어질 수 있는 중요한 출발점 역할을 했다고 볼 수 있다.
