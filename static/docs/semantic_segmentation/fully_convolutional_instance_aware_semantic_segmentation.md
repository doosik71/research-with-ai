# Fully Convolutional Instance-aware Semantic Segmentation

- **저자**: Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Ji, Yichen Wei
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1611.07709

## 1. 논문 개요

이 논문은 `instance-aware semantic segmentation`을 위한 최초의 end-to-end fully convolutional 방법인 **FCIS**를 제안한다. 이 문제는 단순히 픽셀마다 semantic class를 예측하는 것이 아니라, 같은 class에 속하더라도 서로 다른 객체 인스턴스를 구분하여 각각의 mask를 만들어야 한다. 예를 들어 사람 여러 명이 붙어 있는 장면에서는 모든 `person` 픽셀을 하나로 칠하면 안 되고, 각 사람을 따로 분리해 검출하고 분할해야 한다.

저자들이 지적하는 핵심 문제는 기존 FCN이 본질적으로 `translation invariant`하다는 점이다. 즉 같은 형태의 패턴은 이미지 어디에 있든 비슷한 반응을 내기 때문에, “이 픽셀이 어떤 객체 인스턴스의 내부인가”처럼 ROI 내부의 상대적 위치에 따라 의미가 달라지는 문제를 직접 다루기 어렵다. 반면 instance-aware segmentation은 ROI 수준의 `translation-variant`한 처리가 필요하다.

기존 방법들은 보통 전체 이미지에 convolution을 적용한 뒤, ROI pooling으로 각 proposal의 feature를 잘라내고, 마지막에 fully connected layer나 별도 subnet으로 mask를 예측했다. 저자들은 이런 구조가 공간 정보를 훼손하고, 파라미터가 과도하며, ROI별 계산을 공유하지 못해 느리다고 비판한다. 이 논문은 이런 약점을 해결하기 위해 detection과 segmentation을 하나의 fully convolutional 구조 안에서 공동으로 처리하는 방법을 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **position-sensitive score maps**를 확장하여, 객체의 **classification/detection**과 **mask prediction**을 따로 하지 않고 **jointly and simultaneously** 수행하는 것이다. 기존 semantic segmentation의 FCN은 category마다 하나의 score map만 두지만, FCIS는 객체를 $k \times k$ 격자로 나누고 각 상대 위치에 대응되는 점수 지도를 만든다. 이렇게 하면 같은 픽셀이라도 어떤 ROI 안에서 어느 상대 위치에 놓였는지에 따라 다른 의미를 가질 수 있다.

이 논문이 기존 `InstanceFCN`과 다른 점은, 단순히 class-agnostic mask proposal만 만드는 것이 아니라 **inside/outside position-sensitive score maps**를 통해 detection과 segmentation을 함께 풀었다는 데 있다. 저자들은 각 픽셀에 대해 두 질문을 동시에 고려한다.

첫째, 이 픽셀이 어떤 object bounding box에 속하는가  
둘째, 그 box 안에서 실제 object mask 내부인가 아닌가

이를 위해 각 category마다 `inside`와 `outside` 점수를 두고, 픽셀 단위에서는 softmax로 mask 내부/외부를 가르고, detection 쪽에서는 max와 average pooling을 이용해 ROI 수준의 class likelihood를 만든다. 중요한 점은 이 과정에 ROI별 별도 학습 파라미터가 거의 없고, convolutional representation과 score maps가 두 작업 사이에서 완전히 공유된다는 것이다.

즉 이 논문의 차별점은 다음과 같이 정리할 수 있다. 기존 방법이 “공유 backbone + ROI별 별도 mask head”에 가까웠다면, FCIS는 “공유 backbone + 공유 score maps + 가벼운 ROI assembling” 구조를 사용한다. 그 결과 정확도와 속도를 동시에 개선한다.

## 3. 상세 방법 설명

전체 구조는 ResNet backbone, RPN, 그리고 FCIS head로 이루어진다. 저자들은 ResNet의 마지막 1000-way classification fully connected layer를 제거하고 convolutional layers만 유지한다. 최상단 feature map은 2048채널이며, 그 위에 $1 \times 1$ convolution을 추가해 1024채널로 줄인다.

원래 ResNet의 top feature stride는 32인데, 이것은 segmentation에는 너무 거칠다. 그래서 저자들은 `hole algorithm` 또는 `dilated convolution`을 사용해 conv5의 stride를 2에서 1로 줄이고 dilation을 2로 설정하여 최종 feature stride를 16으로 만든다. 이렇게 하면 해상도를 높이면서 receptive field는 유지할 수 있다.

ROI는 Faster R-CNN 계열의 **RPN**이 생성한다. RPN 역시 fully convolutional이고 conv4 feature를 공유한다. 그런 다음 conv5 feature 위에서 각 category에 대해 position-sensitive inside/outside score map을 만든다. 본문에 따르면 score map의 총 개수는 다음과 같다.

$$
2k^2(C+1)
$$

여기서 $C$는 object category 수이고, $+1$은 background category이며, 앞의 2는 `inside`와 `outside` 두 종류를 뜻한다. 실험에서는 기본적으로 $k=7$을 사용한다.

각 ROI가 주어지면, 이 ROI를 feature map 좌표계로 사영한 뒤 $k \times k$ 셀로 나눈다. 그 다음 각 셀 위치에 대응하는 score map에서 해당 영역을 가져와 assemble하면 ROI 내부의 pixel-wise score를 얻는다. 이 조립 과정은 ROI pooling처럼 강제로 fixed-size feature를 만드는 것이 아니라, ROI의 실제 aspect ratio를 유지한 채 score map에서 대응 영역을 가져오는 방식이므로 공간 왜곡이 적다.

논문은 detection과 segmentation을 다음처럼 하나의 inside/outside 표현으로 결합한다. 각 픽셀에 대해 세 가지 경우가 있다.

1. inside score가 높고 outside score가 낮다: detection+, segmentation+  
2. inside score가 낮고 outside score가 높다: detection+, segmentation-  
3. 둘 다 낮다: detection-, segmentation-

즉 `inside`와 `outside` 두 점수만으로 “객체에 속하는지”와 “mask 내부인지”를 동시에 표현한다. 본문 설명을 정리하면, segmentation은 픽셀 단위 softmax로 inside와 outside를 비교해 foreground probability를 만들고, detection은 픽셀별로 inside/outside 중 더 큰 쪽을 취해 objectness 성격의 값을 만든 뒤 ROI 전체에서 average pooling하여 category score를 만든다. 저자들은 이 두 작업이 같은 score maps를 공유하므로 더 본질적인 결합이 가능하다고 주장한다.

이를 아주 단순화해 쓰면, category $c$와 ROI $R$에 대해 픽셀 $p$의 mask 확률은 개념적으로

$$
P_{\text{fg}}(p,c \mid R)
=
\frac{\exp(s_{\text{in}}(p,c))}
{\exp(s_{\text{in}}(p,c))+\exp(s_{\text{out}}(p,c))}
$$

처럼 inside/outside softmax로 해석할 수 있다. 또한 ROI의 detection score는 픽셀별 detection likelihood를 평균내는 형태로 이해할 수 있다. 다만 이 식은 본문 설명을 이해하기 쉽게 정리한 것이며, 추출문에는 이 형태의 기호식이 직접 적혀 있지는 않다.

Bounding box refinement를 위해 bbox regression branch도 함께 둔다. 이 branch는 conv5 feature 위에 또 하나의 sibling $1 \times 1$ convolution을 두고, 총 $4k^2$ 채널을 예측한다. 즉 segmentation/classification과 box regression이 모두 shared convolutional feature 위에 올라간다.

학습 시 ROI는 nearest ground-truth object와의 box IoU가 0.5보다 크면 positive, 아니면 negative이다. 각 ROI에는 세 개의 loss가 들어간다.

1. $C+1$ categories에 대한 softmax detection loss  
2. ground-truth category에 대해서만 계산하는 softmax segmentation loss  
3. bbox regression loss

본문 각주에 따르면 segmentation loss는 ROI 안의 픽셀별 loss를 합한 뒤 ROI 크기로 정규화한다. 또한 segmentation loss와 bbox regression loss는 positive ROI에서만 유효하다. 이 구조는 detection과 segmentation을 실제로 공동 최적화한다는 점에서 논문의 핵심이다.

학습은 ImageNet pretrained model로 초기화하고 SGD를 사용한다. 입력 이미지는 shorter side가 600이 되도록 resize한다. 8 GPU를 사용해 학습하며, COCO에서는 iteration 수를 VOC보다 8배 늘린다. 또 per-ROI 계산 비용이 매우 작다는 장점을 활용해 **OHEM**을 적용한다. 구체적으로 한 이미지에서 RPN이 만든 300개의 ROI를 모두 forward한 뒤, loss가 큰 128개 ROI만 골라 backpropagation한다.

추론 시에는 RPN에서 상위 300개 ROI를 고르고, bbox regression을 거쳐 다시 300개 ROI를 얻어 총 600개 수준의 후보를 사용한다. 이후 NMS를 IoU 0.3으로 적용하고, 최종 mask는 **mask voting**으로 보정한다. 어떤 ROI의 최종 mask를 만들 때 IoU 0.5 이상인 ROI들의 같은 category mask를 classification score로 가중 평균한 뒤 binary mask로 만든다.

## 4. 실험 및 결과

### PASCAL VOC Ablation

PASCAL VOC 2012 train으로 학습하고 VOC 2012 validation에서 평가했다. 추가 instance mask annotation은 [14]를 사용했다. 평가지표는 mask-level IoU threshold 0.5와 0.7에서의 $mAP^r$이다.

가장 중요한 ablation은 네 가지 비교다. `naive MNC`, `InstFCN + R-FCN`, `FCIS (translation invariant)`, `FCIS (separate score maps)` 그리고 최종 `FCIS`를 비교했다.

결과는 다음과 같다.

- `naive MNC`: $mAP^r@0.5 = 59.1\%$, $mAP^r@0.7 = 36.0\%$
- `InstFCN + R-FCN`: $62.7\%$, $41.5\%$
- `FCIS (translation invariant)`: $52.5\%$, $38.5\%$
- `FCIS (separate score maps)`: $63.9\%$, $49.7\%$
- `FCIS`: $65.7\%$, $52.1\%$

이 결과는 세 가지를 명확히 보여준다. 첫째, `translation invariant`하게 만들면 성능이 크게 떨어진다. 즉 position-sensitive parameterization이 핵심이다. 둘째, segmentation과 classification에 score map을 따로 두는 것보다 joint formulation이 더 좋다. 셋째, 단순히 mask proposal용 InstFCN과 detection용 R-FCN을 이어 붙이는 것보다 end-to-end FCIS가 더 강하다.

### COCO에서의 MNC 비교

COCO에서는 80k+40k trainval 이미지로 학습하고 test-dev에 보고했다. 평가는 표준 COCO metric인 $mAP^r@[0.5:0.95]$와 전통적인 $mAP^r@0.5$를 사용했다.

ResNet-101 기준 결과는 다음과 같다.

- `MNC`, random sampling: $24.6\%$ / $44.3\%$
- `FCIS`, random sampling: $28.8\%$ / $48.7\%$
- `FCIS`, OHEM: $29.2\%$ / $49.5\%$

여기서 FCIS는 MNC 대비 $mAP^r@[0.5:0.95]$에서 **절대 4.2%p**, 상대적으로는 약 **17%** 향상되었다. 특히 큰 객체에서 향상이 더 컸다고 논문은 설명한다. 이는 ROI warping 없이 더 풍부한 spatial detail을 유지하는 구조적 장점과 일치한다.

속도 차이도 크다. Nvidia K40 기준 test time은 MNC가 1.37초/image, FCIS가 0.24초/image이다. 즉 약 6배 빠르다. train time도 FCIS가 훨씬 짧다. 저자들은 이것이 per-ROI 계산이 거의 공짜에 가깝기 때문이라고 본다. OHEM도 MNC에서는 계산비용 때문에 실질적으로 어렵지만, FCIS는 거의 추가 비용 없이 사용할 수 있었다.

### Backbone depth 영향

FCIS에 서로 다른 깊이의 ResNet을 사용한 결과는 다음과 같다.

- ResNet-50: $27.1\%$, 0.16초/image
- ResNet-101: $29.2\%$, 0.24초/image
- ResNet-152: $29.5\%$, 0.27초/image

즉 50에서 101로 갈 때는 성능 향상이 뚜렷하지만, 152에서는 거의 포화된다. 따라서 정확도-속도 균형 측면에서 ResNet-101이 실용적인 선택으로 보인다.

### COCO 2016 Challenge 결과

FCIS baseline만으로도 $mAP^r@[0.5:0.95] = 29.2\%$를 달성해 2015년 우승 방법인 MNC+++의 28.4%를 넘는다. 여기에 몇 가지 단순한 enhancement를 더했다.

- multi-scale testing: 32.0%
- horizontal flip: 32.7%
- multi-scale training: 33.6%
- 6-network ensemble: 37.6%

최종 37.6%는 2016년 2위였던 G-RMI의 33.8%보다 **절대 3.8%p**, 상대적으로 약 **11%** 높다. 논문은 이 결과를 통해 FCIS가 단순히 학술적으로 흥미로운 구조를 넘어서 실제 benchmark competition에서도 매우 강력하다는 점을 보여준다.

또한 인스턴스 마스크의 enclosing box를 detection box로 사용했을 때 COCO object detection에서도 $mAP^b@[0.5:0.95] = 39.7\%$를 기록했다고 보고한다. 즉 이 구조는 segmentation뿐 아니라 detection 성능도 우수하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 instance segmentation을 위해 ROI별 무거운 subnet이나 fully connected mask head에 의존하지 않고, 거의 모든 계산을 fully convolutional하게 공유했다는 점이다. 이것은 단순한 구현상의 편의가 아니라, 속도와 정확도 모두에 직접 연결된다. 특히 large object에서 성능 향상이 더 크다는 결과는 기존 ROI pooling 기반 방법의 공간 왜곡 문제를 실제로 줄였음을 뒷받침한다.

또 다른 강점은 detection과 segmentation을 별도 단계가 아니라 하나의 inside/outside score map 체계 안에서 공동 학습했다는 점이다. 저자들의 ablation은 이 joint formulation이 단순히 “가능하다” 수준이 아니라 실제로 separate score maps보다 더 좋다는 것을 보여준다. 또한 OHEM을 저비용으로 적용할 수 있다는 점도 실전 학습에서 중요한 장점이다.

한계도 있다. 첫째, 이 방법은 여전히 **RPN proposal 기반**이다. 즉 완전한 proposal-free 방식은 아니며, 최종 성능은 ROI quality에 어느 정도 의존한다. 둘째, 최종 추론 과정에 NMS와 mask voting 같은 후처리가 들어간다. 논문이 “end-to-end fully convolutional”을 강조하지만, 실제 inference pipeline 전체가 순수 convolution 연산만으로 끝나는 것은 아니다. 셋째, 본문에서 inside/outside joint formulation의 수학적 분석이나 이론적 정당화는 깊게 제시되지 않고, 주로 empirical evidence로 설득한다.

또한 이 논문은 당시 강력한 성능을 보였지만, 본문만 보면 occlusion이 심한 장면이나 매우 작은 객체, 또는 proposal 단계에서 놓친 객체에 대한 별도의 구조적 보완은 크지 않다. COCO small object에서의 수치가 large object보다 많이 낮은 점도 이런 한계를 간접적으로 보여준다. 다만 이것이 FCIS만의 약점인지, 당시 instance segmentation 전반의 어려움인지는 구분해서 볼 필요가 있다.

## 6. 결론

이 논문은 **instance-aware semantic segmentation을 위한 최초의 fully convolutional end-to-end 방법**으로서 의미가 크다. 핵심 기여는 position-sensitive score maps를 inside/outside joint formulation으로 확장하여, segmentation과 detection을 같은 표현 위에서 동시에 수행하게 만든 점이다. 이 설계는 ROI별 무거운 계산을 제거하고, 공간 정보를 더 잘 보존하며, 정확도와 속도를 함께 끌어올렸다.

실험적으로도 FCIS는 PASCAL VOC와 COCO에서 강력한 결과를 보였고, 특히 COCO 2016 segmentation challenge 우승으로 그 실용성을 입증했다. 이후 instance segmentation 연구는 Mask R-CNN처럼 더 단순하고 확장성 좋은 방향으로 발전했지만, FCIS는 “translation-invariant FCN만으로는 부족하며, instance task에는 relative position-aware representation이 중요하다”는 점을 분명하게 보여준 중요한 전환점으로 볼 수 있다.
