# Instance-aware Semantic Segmentation via Multi-task Network Cascades

- **저자**: Jifeng Dai, Kaiming He, Jian Sun
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1512.04412

## 1. 논문 개요

이 논문은 `instance-aware semantic segmentation`, 즉 이미지의 각 픽셀에 semantic category를 부여하는 것을 넘어서, 같은 category에 속한 서로 다른 개체(instance)까지 구분하는 문제를 다룬다. 예를 들어 여러 사람이나 여러 양이 붙어 있을 때, 단순히 모두를 `person` 또는 `sheep`로 칠하는 것이 아니라 각 개체별 mask를 따로 예측해야 한다.

저자들은 당시의 semantic segmentation 계열 방법들, 특히 FCN 계열 방법이 픽셀 단위 category 예측에는 강하지만 동일 category 내부의 개체 분리는 못 한다는 점을 문제로 본다. 반대로 instance segmentation을 수행하던 기존 CNN 기반 방법들은 대체로 외부 mask proposal 방법, 예를 들어 MCG 같은 느린 모듈에 의존했다. 이 경우 추론 속도가 매우 느리고, proposal 단계가 전체 정확도의 병목이 될 수 있다.

이 논문의 목표는 외부 mask proposal 없이, CNN만으로 빠르고 정확한 instance-aware semantic segmentation을 수행하는 것이다. 이를 위해 저자들은 문제를 세 개의 더 단순한 하위 과제로 나눈다. 첫째, object instance를 class-agnostic bounding box로 찾는다. 둘째, 각 box 안에서 class-agnostic mask를 예측한다. 셋째, 그 instance의 category를 분류한다. 논문은 이 세 단계를 하나의 `Multi-task Network Cascades (MNC)` 구조로 묶고, stage 간 인과적 의존성까지 고려하여 end-to-end로 학습하는 방법을 제안한다.

이 문제의 중요성은 COCO 같은 대규모 benchmark가 instance-aware segmentation을 핵심 과제로 채택하고 있었기 때문이다. 논문은 이 문제에 대해 당시 SOTA 정확도와 매우 빠른 추론 속도를 동시에 달성했다고 주장한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 instance segmentation을 한 번에 직접 푸는 대신, 서로 연관된 세 단계의 sub-task로 분해하고 이를 cascade로 연결하는 것이다. 이 분해는 다음과 같다.

첫 번째 단계는 instance를 box 형태로 제안하는 것이다. 두 번째 단계는 해당 box 안에서 foreground mask를 추정하는 것이다. 세 번째 단계는 그 instance가 어떤 category인지 분류하는 것이다. 저자들의 관점은 원래 문제를 곧바로 푸는 것보다 이 세 문제가 각각 더 단순하며 CNN이 다루기 더 쉽다는 것이다.

이 논문의 중요한 차별점은 단순한 multi-task learning이 아니라 `causal cascade`라는 점이다. 일반적인 multi-task learning에서는 여러 task가 shared feature를 같이 쓰지만 서로의 출력에 직접 의존하지 않는 경우가 많다. 반면 이 논문에서는 뒤 stage가 앞 stage의 출력에 의존한다. 예를 들어 mask stage는 stage 1의 predicted box를 입력으로 사용하고, classification stage는 stage 2의 predicted mask를 사용한다. 즉, stage 간에 명확한 인과 관계가 있다.

또 하나의 핵심은 이 구조를 단순히 순차적으로 따로 학습하지 않고, 이론적으로 정당한 방식으로 end-to-end 학습했다는 점이다. 문제는 box 좌표 자체가 네트워크 출력이므로, RoI feature 추출 과정이 박스 위치에 대해 미분 가능해야 한다는 것이다. 이를 해결하기 위해 저자들은 `differentiable RoI warping layer`를 제안한다. 이것이 있어야 뒤 stage의 loss가 앞 stage의 box prediction까지 gradient를 전달할 수 있다.

정리하면, 이 논문의 핵심 기여는 다음 세 가지로 압축된다. 문제를 box-mask-category의 cascade로 분해한 점, cascade 전체가 shared convolutional feature를 사용하도록 설계한 점, 그리고 stage 간 의존성을 고려해 single-step end-to-end 학습을 가능하게 한 점이다.

## 3. 상세 방법 설명

전체 시스템은 입력 이미지 하나를 받아 instance-aware segmentation 결과를 출력한다. 구조는 3-stage 기본형과, 이를 확장한 5-stage 버전이 있다. 기본 3-stage는 다음과 같다.

### Stage 1: Box-level instance regression

첫 번째 stage는 class-agnostic object proposal을 생성한다. 구조와 loss는 기본적으로 `Region Proposal Network (RPN)`을 따른다. 공유 convolutional feature map 위에 3x3 convolution을 하나 두고, 그 위에 두 개의 sibling 1x1 convolution을 둔다. 하나는 bounding box regression, 다른 하나는 objectness classification을 담당한다.

이 stage의 출력은 box 집합 $B=\{B_i\}$ 이고, 각 box는
$B_i=\{x_i, y_i, w_i, h_i, p_i\}$  
형태를 가진다. 여기서 $(x_i, y_i)$는 box 중심, $w_i, h_i$는 폭과 높이, $p_i$는 objectness probability이다.

이 단계의 loss는 다음처럼 쓴다.

$$
L_1 = L_1(B(\Theta))
$$

여기서 $\Theta$는 네트워크의 모든 학습 파라미터이다. 중요한 점은 $B$가 고정 proposal이 아니라 $\Theta$의 함수라는 것이다.

### Stage 2: Mask-level instance regression

두 번째 stage는 stage 1의 predicted box와 shared convolutional feature를 입력으로 받아, 각 box에 대한 pixel-level mask를 예측한다. 여기서도 mask는 아직 class-agnostic이다.

각 box에 대해 `RoI pooling`으로 고정 크기 feature를 뽑는다. 논문에서는 이 stage의 RoI feature 크기를 14x14로 둔다. 그 다음 두 개의 fully connected layer를 붙인다. 첫 번째 fc는 ReLU와 함께 feature 차원을 256으로 줄이고, 두 번째 fc는 $m \times m$ 해상도의 mask를 회귀한다. 논문에서는 $m=28$을 사용하므로 출력은 784차원 벡터이다.

이 출력의 각 원소는 binary logistic regression으로 학습되며, sigmoid를 거쳐 $[0,1]$ 범위의 값을 가진다. 즉 각 픽셀이 foreground일 확률을 예측하는 구조이다.

이 단계의 loss는 다음과 같다.

$$
L_2 = L_2(M(\Theta)\mid B(\Theta))
$$

여기서 $M=\{M_i\}$ 이고, 각 $M_i$는 해당 instance에 대한 $m^2$차원 mask prediction이다. 수식에서 드러나듯이, mask loss는 단순히 $M$에만 의존하지 않고 stage 1의 box output $B(\Theta)$에도 의존한다. 왜냐하면 어떤 box를 자르고 warp할지가 $B$에 의해 결정되기 때문이다.

### Stage 3: Categorizing instances

세 번째 stage는 shared feature, stage 1의 box, stage 2의 mask를 입력으로 받아 instance category를 분류한다.

먼저 stage 1 box로부터 다시 RoI feature를 추출한다. 이후 이 feature map에 stage 2에서 예측된 mask를 곱해 foreground에 집중된 feature를 만든다. 논문은 이를 다음과 같이 쓴다.

$$
F_i^{Mask}(\Theta)=F_i^{RoI}(\Theta)\cdot M_i(\Theta)
$$

여기서 $F_i^{RoI}$는 RoI pooled feature이고, $M_i(\Theta)$는 RoI 해상도에 맞게 resize된 predicted mask이며, $\cdot$는 element-wise product이다.

이후 mask-based pathway에서는 이 masked feature를 두 개의 4096차원 fc layer에 통과시킨다. 동시에 논문은 SDS 계열 방법을 따라 box-based pathway도 둔다. 이 경로에서는 mask를 곱하지 않은 원래 RoI feature를 두 개의 4096차원 fc layer에 통과시킨다. 마지막으로 두 pathway를 concatenate하고, 그 위에 $(N+1)$-way softmax classifier를 올려 $N$개 object class와 1개 background를 분류한다.

box-based pathway를 추가한 이유는, predicted mask가 거의 비어 있거나 background 영역일 때 mask-based feature만으로는 불안정할 수 있기 때문이다.

이 단계의 loss는 다음과 같다.

$$
L_3 = L_3(C(\Theta)\mid B(\Theta), M(\Theta))
$$

여기서 $C=\{C_i\}$ 는 각 instance의 category prediction이다.

### 전체 학습 목표

전체 cascade의 통합 loss는 다음과 같다.

$$
L(\Theta)=L_1(B(\Theta))+L_2(M(\Theta)\mid B(\Theta))+L_3(C(\Theta)\mid B(\Theta), M(\Theta))
$$

논문에서는 세 항의 가중치를 모두 1로 둔다. 이 식의 핵심은 전통적인 multi-task learning과 달리, 뒤 loss가 앞 stage 출력에 조건부로 의존한다는 점이다. 따라서 backpropagation 시 단순히 각 stage를 독립적으로 다룰 수 없다.

### Differentiable RoI Warping

이 논문의 핵심 기술적 기여가 여기 있다. 기존 Fast R-CNN의 RoI pooling은 precomputed proposal이 고정되어 있다고 가정하므로, gradient는 convolutional feature에 대해서만 흘리면 된다. 그러나 이 논문에서는 proposal box 자체가 네트워크의 출력이므로, loss가 box 좌표 $(x_i,y_i,w_i,h_i)$ 에 대해서도 미분 가능해야 한다.

이를 위해 저자들은 RoI pooling을 두 단계로 분해한다.

1. `RoI warping`: box 영역을 잘라 fixed-size feature map으로 bilinear interpolation
2. 그 다음 standard max pooling

RoI warping은 다음처럼 선형변환으로 표현된다.

$$
F_i^{RoI}(\Theta)=G(B_i(\Theta))F(\Theta)
$$

여기서 $F(\Theta)$는 full-image convolutional feature map이고, $G(B_i(\Theta))$는 box $B_i$에 따라 결정되는 crop-and-warp 연산이다.

공간 좌표 단위로 쓰면,

$$
F_i^{RoI}(u',v')=\sum_{(u,v)} G(u,v;u',v'\mid B_i)F(u,v)
$$

이며, bilinear interpolation을 사용할 때 $G$는 separable하게 쓸 수 있다. 논문은 가로축에 대해 다음을 제시한다.

$$
g(u,u' \mid x_i,w_i)=\kappa\left(x_i+\frac{u'}{W'}w_i-u\right)
$$

여기서 $\kappa(\cdot)=\max(0,1-|\cdot|)$ 는 bilinear interpolation kernel이다. 세로축도 동일하게 정의된다.

이 정의 덕분에 loss $L_2$ 또는 $L_3$가 box 좌표에 대해 gradient를 가질 수 있다. 논문은 이를 다음과 같이 표현한다.

$$
\frac{\partial L_2}{\partial B_i}
=
\frac{\partial L_2}{\partial F_i^{RoI}}
\frac{\partial G}{\partial B_i}
F
$$

즉, box 위치와 크기가 바뀌면 RoI feature가 어떻게 변하는지를 계산할 수 있고, 따라서 앞단 proposal stage까지 end-to-end로 학습할 수 있다.

### Masking layer와 gradient 흐름

Stage 3의 mask-based pathway는 $F_i^{RoI}$ 와 $M_i$ 의 element-wise product만 수행하므로, differentiable RoI warping이 마련된 뒤에는 이 부분도 자연스럽게 미분 가능하다. 따라서 category classification loss는 mask prediction과 box prediction 양쪽 모두로 gradient를 전달할 수 있다.

### 5-stage cascade 확장

논문은 기본 3-stage 외에 5-stage cascade도 제안한다. 아이디어는 stage 3에서 category classification과 함께 class-wise box regression도 수행하고, 이 regressed box를 새로운 proposal처럼 다시 stage 2와 stage 3에 넣는 것이다.

즉 추론 시 순서는 다음과 같다.

1. stage 1에서 box proposal 생성
2. stage 2에서 mask 예측
3. stage 3에서 category 분류 및 box refinement
4. refined box를 다시 입력으로 stage 4(= mask 재예측)
5. stage 5(= category 재분류)

논문은 이를 통해 training-time 구조와 inference-time 구조를 일치시키는 것이 중요하다고 본다. 실제로 5-stage 구조로 학습하면 5-stage inference와 더 잘 맞아 정확도가 올라간다고 보고한다.

### 학습 세부사항

논문에 명시된 구현 세부사항은 다음과 같다.

Stage 1에서는 약 $10^4$개의 regressed box가 나오는데, IoU 0.7의 NMS를 적용해 top 300 box만 다음 stage로 넘긴다. 학습 시에도 stage 2와 3의 forward/backward는 이 300 box 경로를 통해서만 전달된다.

positive/negative sample 정의도 stage별로 다르다. Stage 2에서는 proposal box와 가장 많이 겹치는 ground-truth mask를 찾고, IoU가 0.5보다 크면 positive로 간주해 mask regression loss를 부여한다. target mask는 proposal box와 ground-truth mask의 교집합을 잘라 $m \times m$ 으로 resize한 것이다.

Stage 3에서는 두 종류의 classifier를 둔다. 하나는 mask-level instance classifier이고 다른 하나는 box-level instance classifier이다. 학습 label을 정할 때 box IoU와 mask IoU를 함께 사용한다. 이는 proposal이 background이거나 실제 instance와 잘 맞지 않을 때 predicted mask가 불안정할 수 있기 때문이다. 논문은 box-level classifier score는 inference에는 사용하지 않는다고 명시한다.

학습은 ImageNet pretrained model로 convolution layer와 대응되는 4096차원 fc layer를 초기화하고, 나머지 추가 layer는 무작위 초기화한다. `image-centric` 학습을 사용하며, mini-batch당 이미지 1장을 사용한다. Stage 1에서는 256개의 sampled anchor, stage 2와 3에서는 64개의 sampled RoI를 사용한다. 학습률은 0.001로 32k iteration, 이후 0.0001로 8k iteration이다. 8개의 GPU를 사용하고, 입력 이미지는 short side를 600 pixels로 resize한다.

### 추론 및 후처리

논문은 3-stage로 학습한 모델이든 5-stage로 학습한 모델이든 모두 5-stage inference를 사용한다고 설명한다. 최종적으로 600개의 instance prediction을 얻는다. 300개는 stage 3 outputs, 나머지 300개는 stage 5 outputs이다.

이후 category score를 기준으로 box-level IoU 0.3의 NMS를 수행한다. 그리고 suppress되지 않은 instance마다, IoU 0.5 이상으로 겹쳐 suppress된 “similar” instance들을 모아 mask를 weighted averaging한다. 가중치는 classification score이다. 이를 저자들은 `mask voting`이라 부르며, 약 1% 정확도 향상을 가져온다고 보고한다.

## 4. 실험 및 결과

### PASCAL VOC 2012 실험 설정

논문은 PASCAL VOC 2012 train set으로 학습하고 validation set에서 평가한다. segmentation annotation은 Hariharan 등 [12]의 annotation을 사용한다. 평가지표는 `mean Average Precision for regions`, 즉 `mAP^r` 이며, IoU threshold 0.5와 0.7에서 평가한다.

### 학습 전략에 대한 ablation

가장 먼저 저자들은 서로 다른 학습 전략을 비교한다. 모든 비교는 같은 5-stage inference를 사용하고, 차이는 오직 training 방법에 있다.

VGG-16 기준으로 보면, feature sharing 없이 3개 stage를 따로 step-by-step 학습한 baseline은 `mAP^r@0.5 = 60.2%`를 얻는다. 이 결과 자체도 이미 경쟁력이 있는데, 이는 저자들이 제안한 task decomposition 자체가 유효하다는 뜻으로 해석할 수 있다.

그 다음, feature sharing은 하되 end-to-end는 하지 않고 step-by-step 방식으로 다시 학습한 경우는 `60.5%`이다. 즉 단순 feature sharing만으로는 큰 이득이 없다.

반면 논문이 제안한 방식대로 3-stage cascade를 single-step end-to-end 학습하면 `62.6%`로 오른다. 이는 구조는 같고 학습 방식만 다르므로, 정확도 향상이 end-to-end 학습과 unified optimization에서 왔음을 보여준다.

마지막으로 5-stage cascade를 end-to-end로 학습하면 `63.5%`가 된다. 즉 training-time cascade 구조를 inference-time usage와 일치시킨 것이 추가 성능 향상으로 이어진다.

이 결과는 ZF net과 VGG-16 둘 다에서 일관되게 나타난다고 보고한다.

### 기존 방법과의 비교

PASCAL VOC 2012 validation에서 기존 대표 방법들과 비교한 결과는 다음과 같다.

- SDS (AlexNet): `49.7%` at mAP^r@0.5, `25.3%` at mAP^r@0.7, `48s/img`
- Hypercolumn: `60.0%`, `40.4%`, `>80s/img`
- CFM: `60.7%`, `39.6%`, `32s/img`
- MNC: `63.5%`, `41.5%`, `0.36s/img`

즉 이 논문의 방법은 기존 최고 수준 대비 mAP^r@0.5에서 약 3% 높고, mAP^r@0.7에서도 더 높다. 동시에 추론 시간은 `0.36초/이미지`로, MCG proposal에 크게 의존하던 이전 방법들보다 약 두 자릿수 이상 빠르다. 논문은 이를 “two orders of magnitude faster”라고 요약한다.

세부 시간 분해를 보면 VGG-16 기준으로 convolution에 0.15초, stage 2에 0.01초, stage 3에 0.08초, stage 4에 0.01초, stage 5에 0.08초, 기타 후처리에 0.03초가 걸려 총 0.36초이다. shared feature를 쓰기 때문에 multi-stage 구조임에도 계산량이 크게 늘지 않는다.

### Object detection 성능

저자들은 box-level detection 성능도 측정한다. instance mask에서 tight bounding box를 추출해 object detection mAP를 계산했을 때, VOC 2012 test에서 `70.9%`를 얻는다. 이는 VOC 2012만으로 학습한 Fast R-CNN의 `65.7%`, Faster R-CNN의 `67.0%`보다 높다.

더 나아가, stage 3/5의 box regression layer가 직접 출력한 box를 사용하면 `73.5%`를 얻는다. 그리고 VOC 2007+2012 데이터를 함께 사용했을 때 `75.9%`까지 올라간다. 흥미로운 점은 이 논문이 원래 instance segmentation을 목표로 설계되었는데도 box detection에서도 강한 성능을 보인다는 것이다.

다만 2007 데이터에는 mask annotation이 없기 때문에, 논문은 이 경우 mask regression loss를 무시하고 box proposal 및 categorization stage만 해당 샘플로 학습했다고 설명한다.

### MS COCO 실험

COCO에서는 80개 category에 대해 instance-aware semantic segmentation을 평가한다. 논문은 trainval 80k+40k로 학습하고 test-dev에서 평가한다. COCO 표준 metric인 `mAP^r@[0.5:0.95]`와 PASCAL 스타일의 `mAP^r@0.5`를 함께 보고한다.

VGG-16 backbone에서는 `19.5% / 39.7%`를 얻고, ResNet-101으로 바꾸면 `24.6% / 44.3%`가 된다. 즉 COCO의 stricter metric인 `[0.5:0.95]` 기준으로 약 26% 상대 향상을 보인다. 이는 deeper representation이 MNC 구조에 자연스럽게 이득을 준다는 논문의 주장과 맞는다.

또한 저자들은 이 baseline 위에 global context modeling, multi-scale testing, ensemble을 더해 최종적으로 test-challenge에서 `28.2% / 51.5%`를 달성했고, 2015 COCO segmentation track 1위를 차지했다고 적고 있다.

다만 global context modeling이나 ensemble의 구체적 구조는 본문에 상세히 설명되어 있지 않다. 따라서 그 부분의 정확한 메커니즘은 이 논문 텍스트만으로는 분석할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 분해가 매우 자연스럽고 실용적이라는 점이다. instance segmentation을 box proposal, mask regression, category classification으로 나누면 각 단계가 더 잘 정의되고, 기존 object detection 및 segmentation 기술을 효과적으로 재사용할 수 있다. 실제로 baseline조차 높은 성능을 보였다는 점은 이 분해가 설계상 타당함을 뒷받침한다.

두 번째 강점은 cascade 전체를 정말로 end-to-end로 학습했다는 점이다. 단순히 여러 모듈을 연결한 시스템이 아니라, 뒤 stage의 loss가 앞 stage의 box prediction까지 gradient를 전달하도록 differentiable RoI warping을 설계했다. 이는 이론적으로도 깔끔하고, 실험적으로도 step-by-step 학습 대비 분명한 성능 향상을 보였다.

세 번째 강점은 속도와 정확도를 동시에 잡았다는 것이다. 당시의 경쟁 방법들이 mask proposal 단계 때문에 수십 초가 걸리던 상황에서, 이 방법은 0.36초 수준으로 대폭 빠르다. 외부 proposal에 의존하지 않고 shared convolutional feature를 재사용한 덕분이다. 실용적인 시스템 관점에서 매우 중요한 장점이다.

네 번째 강점은 확장성이다. 논문은 3-stage 기본형에서 끝나지 않고 5-stage로 자연스럽게 확장할 수 있음을 보였다. 또한 VGG-16에서 ResNet-101로 backbone을 바꾸었을 때 성능 향상을 쉽게 얻는다. 이는 구조가 특정 backbone이나 얕은 네트워크에 묶여 있지 않음을 시사한다.

한편 한계도 분명하다. 먼저, 각 stage가 class-agnostic box와 mask를 먼저 만들고 마지막에 category를 분류하는 구조이므로, category-specific shape prior를 초기에 활용하지 못한다. 이것이 항상 최선인지는 논문이 직접 분석하지 않는다.

또한 논문은 mask를 $28 \times 28$의 고정 해상도 벡터로 회귀한다. 이 방식은 계산 효율은 좋지만, 매우 정교한 경계 표현에는 한계가 있을 수 있다. 저자들도 결론에서 CRF 등을 이용한 boundary refinement는 future work라고 언급한다.

학습 구조 역시 개념적으로는 깔끔하지만 구현 난도가 높다. 특히 differentiable RoI warping, stage 간 gradient dependency, multi-loss routing, NMS 후 top 300 proposal 경로 처리 등은 단순한 detector보다 구현이 복잡하다. 논문은 Caffe에서 SGD로 구현했다고만 말하며, 세부 엔지니어링 난이도는 별도로 논의하지 않는다.

또한 stage 3에서 두 종류의 classifier를 사용하는 설계, positive/negative sample 정의, 5-stage inference 후 mask voting 같은 후처리는 성능에 기여하지만 전체 시스템을 더 복잡하게 만든다. 즉 구조적 단순함보다는 성능 지향적 설계에 가깝다.

마지막으로, COCO 최종 우승 결과에는 global context modeling, multi-scale testing, ensemble이 추가되었지만, 이 논문 본문만으로는 그 추가 요소들의 개별 기여를 분리해서 평가하기 어렵다. 따라서 논문의 핵심 기여는 어디까지나 MNC 자체와 그 end-to-end 학습에 두는 것이 정확하다.

비판적으로 보면, 이 논문은 이후 등장한 더 직접적인 instance segmentation 프레임워크, 예를 들어 detection과 mask prediction을 더 긴밀히 묶는 방식의 전조로 볼 수 있다. 다만 이 논문 자체는 아직 proposal, mask, classification을 분리된 stage로 두고 있어 구조가 다소 복합적이다. 그럼에도 당시 기준으로는 매우 합리적인 절충이었다고 평가할 수 있다.

## 6. 결론

이 논문은 instance-aware semantic segmentation을 위해 `Multi-task Network Cascades`라는 구조를 제안했다. 핵심은 문제를 box proposal, mask regression, category classification의 세 하위 과제로 나누고, 이들을 shared feature 위의 causal cascade로 연결한 뒤, differentiable RoI warping을 통해 end-to-end 학습을 가능하게 한 것이다.

실험적으로 이 방법은 PASCAL VOC에서 당시 최고 수준의 segmentation 성능을 달성했고, 기존 방법보다 압도적으로 빠른 추론 속도를 보였다. 또한 object detection 성능도 강력했고, COCO에서는 더 깊은 backbone과 결합해 높은 확장성을 보였다.

실제 적용 관점에서 이 연구는 외부 proposal 의존성을 줄이고, segmentation을 detector-like pipeline 안에 통합하려는 흐름을 강하게 밀어준 작업으로 볼 수 있다. 향후 연구 측면에서도, proposal과 mask prediction의 결합, differentiable region operation, multi-stage end-to-end optimization 같은 아이디어를 발전시키는 중요한 기반 역할을 했다고 평가할 수 있다.
