# Instance and Panoptic Segmentation Using Conditional Convolutions

이 논문은 instance segmentation과 panoptic segmentation에서 오랫동안 지배적이었던 **ROI 기반(Mask R-CNN 계열)** 접근을 정면으로 대체하려는 시도입니다. 저자들은 각 인스턴스마다 ROIAlign으로 특징을 잘라 고정된 mask head에 넣는 대신, **각 인스턴스에 조건부로 생성되는 동적 convolution filter**를 사용해 마스크를 직접 예측하는 **CondInst**를 제안합니다. 이를 통해 ROI cropping과 feature alignment를 없애고, 더 높은 해상도의 마스크를 더 단순한 fully convolutional 구조에서 생성할 수 있게 했습니다. 논문은 이 구조가 정확도뿐 아니라 속도에서도 강력하며, semantic branch를 붙이면 panoptic segmentation까지 자연스럽게 확장된다고 주장합니다.  

## 1. Paper Overview

이 논문이 해결하려는 핵심 문제는 다음과 같습니다. 기존의 top-performing instance segmentation 방법은 대부분 Mask R-CNN처럼 **두 단계 구조**를 따릅니다. 먼저 detector가 bounding box를 찾고, 그다음 각 인스턴스에 대해 ROIAlign으로 영역을 crop한 뒤, 고정된 mask head가 foreground/background segmentation을 수행합니다. 하지만 저자들은 이 방식이 세 가지 구조적 한계를 가진다고 지적합니다. 첫째, axis-aligned ROI는 불규칙한 형태의 객체에서 배경이나 다른 인스턴스를 과도하게 포함할 수 있습니다. 둘째, 하나의 고정된 mask head가 모든 인스턴스를 처리해야 하므로 큰 receptive field와 큰 용량이 필요해 계산량이 커집니다. 셋째, ROI를 고정 크기로 resize해야 하므로 출력 마스크 해상도가 제한되어 큰 객체의 경계 디테일이 손실될 수 있습니다.

이 문제가 중요한 이유는 instance segmentation이 단순 semantic segmentation보다 더 어렵기 때문입니다. 각 픽셀에 클래스만 붙이면 되는 것이 아니라, **같은 클래스 안에서도 서로 다른 개별 인스턴스를 구분**해야 합니다. FCN은 비슷한 appearance에는 비슷한 출력을 내는 경향이 있어서, 서로 닮은 두 사람 A와 B가 있는 장면에서 A의 마스크를 예측할 때 B를 배경으로 누르는 것이 어렵습니다. 기존에는 ROI가 바로 이 “특정 인스턴스에 attention을 주는 장치” 역할을 했는데, 저자들은 이 attention을 ROI가 아니라 **동적으로 생성된 instance-aware filter**로 대체하겠다는 발상을 제시합니다.

## 2. Core Idea

CondInst의 핵심 직관은 간단합니다.
**인스턴스별로 같은 mask head를 반복해서 쓰지 말고, 그 인스턴스에 맞는 mask head 파라미터 자체를 생성하자**는 것입니다. 즉, Mask R-CNN의 관점에서는 “입력 ROI는 바뀌지만 mask head는 고정”이었다면, CondInst에서는 “입력은 전체 feature map이고, mask head의 filter가 인스턴스마다 바뀐다”로 관점이 뒤집힙니다.  

구체적으로는 각 위치 $(x,y)$에서 예측된 인스턴스 후보에 대해 controller sub-network가 그 인스턴스를 위한 filter parameter $\boldsymbol{\theta}_{x,y}$를 생성합니다. 그리고 이 필터를 사용해 전체 고해상도 feature map 위에서 해당 인스턴스의 마스크를 추론합니다. 논문은 이 동적 필터가 인스턴스의 상대 위치, 모양, appearance 같은 특성을 encode하며, 결과적으로 해당 인스턴스에만 반응하도록 학습된다고 설명합니다. 이 때문에 ROI 없이도 “어느 인스턴스를 분할해야 하는가”라는 문제를 해결할 수 있습니다.

논문이 제시하는 novelty는 세 가지 층위에서 볼 수 있습니다. 첫째, **ROI-free instance segmentation**을 고성능으로 구현했다는 점입니다. 둘째, dynamic filters를 단순 capacity 증가 목적이 아니라 **instance-aware mask prediction**에 직접 연결했다는 점입니다. 셋째, 이 구조 덕분에 instance segmentation과 panoptic segmentation을 하나의 fully convolutional framework 안에 통합할 수 있다는 점입니다. 저자들은 이것이 정확도와 속도 모두에서 기존 강한 방법을 넘어서는 첫 새로운 framework라고 강조합니다.  

## 3. Detailed Method Explanation

### 3.1 전체 구조

CondInst는 backbone에서 얻은 feature map $C_3, C_4, C_5$를 FPN을 통해 $P_3$부터 $P_7$까지의 pyramid feature로 만들고, detection 쪽은 FCOS 스타일로 처리합니다. 즉, classification head는 각 위치의 class probability $\boldsymbol{p}_{x,y}$를 예측하고, 병렬로 center-ness 및 box head가 동작합니다. 동시에 controller가 각 위치별 인스턴스를 위한 동적 필터 파라미터 $\boldsymbol{\theta}_{x,y}$를 생성합니다. 이 구조는 ROI proposal을 crop해서 head에 넣는 방식이 아니라, detection과 mask generation이 모두 dense prediction의 연장선에 놓여 있습니다.

### 3.2 Bottom branch와 relative coordinates

논문에서 매우 중요한 구성은 **bottom branch**입니다. bottom branch의 출력 $\mathbf{F}_{bottom}$은 $P_3$와 같은 해상도를 가지며, 주로 $P_3$, $P_4$, $P_5$ 정보를 집약해 고해상도 mask prediction에 필요한 세밀한 spatial detail을 제공합니다. 여기에 인스턴스 중심 기준의 **relative coordinates**를 concatenate하여 $\tilde{\mathbf{F}}_{bottom}$을 만듭니다. 그리고 instance-aware mask head는 이 $\tilde{\mathbf{F}}_{bottom}$ 위에서 작동합니다.

이 relative coordinate는 왜 중요한가를 이해하는 것이 핵심입니다. ROI가 사라지면 모델은 “지금 이 전체 feature map 위에서 어느 객체를 분할해야 하는지”를 명시적으로 알려줄 장치가 필요합니다. relative coordinate는 각 픽셀이 타깃 인스턴스 중심으로부터 얼마나 떨어져 있는지를 제공해, 동적 필터가 동일한 appearance의 다른 인스턴스를 배경으로 억제하는 데 도움을 줍니다. 논문 후반 ablation도 relative coordinates가 인스턴스 구분에 중요하다고 보여줍니다. 반면 bottom features는 경계의 디테일 복원에 결정적이라고 설명합니다.

### 3.3 Dynamic mask head

CondInst의 mask head는 매우 작습니다. 논문 abstract에 따르면 예시 설정으로 **3개의 convolution layer, 각 8채널**만으로도 충분합니다. 중요한 이유는 이 head가 모든 인스턴스를 동시에 처리하는 보편적 predictor가 아니라, **특정 인스턴스 하나의 mask만 예측하는 전용 predictor**이기 때문입니다. 일반적인 Mask R-CNN처럼 큰 head가 넓은 문맥을 포괄적으로 인코딩할 필요가 줄어드는 것입니다.

개념적으로 표현하면, 각 인스턴스 $k$에 대해 controller가 생성한 파라미터 $\theta_k$가 있고, 이때 mask prediction은 다음처럼 볼 수 있습니다.

$$
\mathbf{M}_k = f*{\theta_k}\big(\tilde{\mathbf{F}}_{bottom}\big)
$$

여기서 $f_{\theta_k}$는 인스턴스 $k$에 맞춰 동적으로 생성된 작은 convolutional mask head입니다. 논문은 이 구조를 통해 마스크 head를 수십, 수백 개의 인스턴스에 대해 반복 적용하더라도 per-instance overhead가 매우 작다고 주장합니다. 실제로 최대 100개 인스턴스에 대해서도 mask 결과 계산이 5ms 미만의 추가 시간만 필요하다고 설명합니다.

### 3.4 ROI 제거의 의미

ROI 제거는 단순히 파이프라인을 줄이는 것 이상의 의미를 가집니다.
첫째, resize 과정이 없어져 large object의 boundary detail이 더 잘 보존됩니다.
둘째, ROI cropping/align이 빠지므로 inference time이 인스턴스 수에 덜 민감해집니다.
셋째, architecture가 fully convolutional이 되어 panoptic segmentation으로의 확장이 쉬워집니다.  

저자들의 해석에 따르면, Mask R-CNN이 ROI로 하던 “instance attention”을 CondInst는 **filter generation + relative coordinate**의 조합으로 수행합니다. 그리고 논문은 나중에 “generated filter가 실제로 무엇을 encode하는가?”를 분석하면서, relative coordinate만 넣어도 대략적인 contour를, bottom feature까지 넣으면 fine detail을 복원한다고 설명합니다. 즉 동적 필터는 일종의 **instance contour representation**처럼 작동한다고 볼 수 있습니다.

### 3.5 Panoptic segmentation 확장

CondInst는 semantic segmentation branch를 추가하면 panoptic segmentation으로 확장됩니다. 여기서 저자들은 단순히 branch 하나를 붙이는 것보다 중요한 차이를 지적합니다. panoptic segmentation에서는 한 픽셀이 최종적으로 하나의 label만 가져야 하므로, instance segmentation 원래 annotation과 달리 **겹치는 영역은 앞쪽 인스턴스 하나에만 할당**되어야 합니다. 따라서 panoptic 학습 시에는 instance target 자체를 panoptic annotation에 맞게 바꿔야 합니다. 이 점은 단순한 multi-task 결합이 아니라, task definition 차이를 반영한 설계라는 점에서 중요합니다.  

## 4. Experiments and Findings

### 4.1 정량적 성능

논문 abstract 수준에서도 COCO에서 기존 state-of-the-art를 능가한다고 말합니다. 실험 섹션 요약 결과를 보면 CondInst는 **Mask R-CNN 대비 일관되게 더 높은 mask AP**를 기록하며, Cityscapes에서도 같은 설정에서 Mask R-CNN보다 1%p 이상 높은 mask AP를 보였다고 설명합니다. 또한 panoptic segmentation에서도 Panoptic-FPN과 AdaptIS 같은 강한 baseline을 상당 폭 앞선다고 보고합니다.

논문이 특히 강조하는 포인트는 “새로운 구조가 더 단순한데도 더 좋다”는 점입니다. ROI를 제거하고 작은 동적 mask head를 쓰는 것이, 복잡한 two-stage paradigm보다 오히려 더 나은 성능을 낼 수 있다는 메시지입니다. 따라서 이 논문의 실험은 단순한 incremental improvement라기보다, **instance segmentation의 핵심 설계를 재정의**하는 주장에 가깝습니다.

### 4.2 속도와 계산량

CondInst는 FCOS detector 대비 전체적으로 약 10% 정도, 절대 시간으로는 **5ms 미만**의 추가 비용으로 최대 100개 인스턴스의 mask를 계산할 수 있다고 합니다. 즉 mask prediction 비용이 매우 작아서, 전체 inference time이 인스턴스 수에 크게 흔들리지 않습니다. 이는 Mask R-CNN처럼 각 ROI별 head 연산이 누적되는 구조와 뚜렷하게 대비됩니다.  

이 점은 논문의 실용적 가치와 직접 연결됩니다. 실제 장면에 사람이 많거나 객체 수가 많은 경우, ROI 기반 모델은 자연스럽게 후단 연산이 커집니다. CondInst는 mask head가 작고 fully convolutional flow 위에 올려져 있기 때문에, 객체 수 증가에 상대적으로 덜 민감합니다. 즉 **accuracy-speed tradeoff**에서 강점이 있습니다.

### 4.3 Ablation과 해석

논문은 bottom features와 relative coordinates가 각각 다른 역할을 한다는 것을 보여줍니다. relative coordinates만 있을 때는 거친 외곽(contour)은 잡히지만 세밀한 구조가 부족하고, bottom features가 추가되면 세부 디테일이 살아납니다. 반대로 bottom features만 있으면 세부 텍스처는 살아나도 여러 인스턴스를 구별하는 데 한계가 생길 수 있습니다. 결국 둘을 함께 쓰는 것이 가장 좋습니다.

또한 저자들은 generated filters가 encode하는 것이 무엇인지 해석하려고 시도합니다. 그 결론은, CondInst의 동적 필터는 단순한 class embedding이 아니라 **타깃 인스턴스의 shape/contour에 가까운 정보**를 담고 있다는 것입니다. 이것은 axis-aligned ROI에 비해 훨씬 유연한 표현입니다. 즉 ROI는 직사각형 attention이지만, CondInst filter는 더 자유로운 형태 기반 attention이라고 해석할 수 있습니다.

### 4.4 Real-time 및 panoptic 결과

논문은 real-time variant도 제시하며, R-50 기반 CondInst-RT가 YOLACT++ 대비 약 2% AP 높은 성능을 보인다고 설명합니다. 이는 CondInst의 철학이 단지 정확도 모델에만 국한되지 않고, 경량화된 실시간 계열로도 확장 가능함을 보여줍니다.  

Panoptic segmentation에서는 COCO 2018과 Cityscapes에서 경쟁력 있는 PQ를 보고합니다. 특히 CondInst는 별다른 복잡한 trick 없이도 Panoptic-FPN 방식으로 semantic branch를 붙이는 것만으로 strong baseline을 넘었다고 주장합니다. 저자들은 최종 panoptic 성능의 핵심 요인이 결국 **instance segmentation 품질**이라고 보고 있으며, CondInst의 강한 instance mask quality가 panoptic으로도 이어진다고 해석합니다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **문제 정의에 대한 관점 전환**입니다. 기존에는 ROI가 인스턴스 구분의 핵심 도구였는데, 이 논문은 그것을 dynamic filters로 대체했습니다. 이로써 fully convolutional instance segmentation이 실제로 ROI-based SOTA를 넘을 수 있음을 보여줬습니다.

둘째, 구조가 단순하면서도 실용적입니다. detection은 FCOS 스타일, mask는 dynamic head, panoptic은 semantic branch 추가라는 조합이 매우 직관적입니다. 복잡한 ROI assignment나 별도 alignment stage 없이도 동작하므로 구현과 추론 흐름이 깔끔합니다.

셋째, 마스크 해상도와 속도 면에서 명확한 이점이 있습니다. ROI resize가 없으므로 large object 경계가 더 잘 살아나고, 작은 mask head 덕분에 계산 비용이 안정적입니다. 이것은 “정확도만 높다”가 아니라 실제 배치 시스템에서 쓰기 좋다는 뜻입니다.  

### 한계

논문이 직접 길게 한계를 정리하진 않지만, 구조상 몇 가지는 분명합니다.

첫째, 모든 성능 향상이 dynamic filter 덕분인지, FCOS-style detector와 bottom branch 설계까지 포함한 전체 조합 덕분인지는 분리해서 보아야 합니다. 즉 CondInst는 깔끔하지만, 완전히 “filter 하나만의 승리”라고 보기엔 여러 설계 요소가 함께 움직입니다. 이 점은 논문도 ablation으로 일부 보여주지만, 완전히 분리되지는 않습니다.  

둘째, 동적 필터가 contour-like representation을 encode한다고 해도, 그 해석은 실험적 관찰에 가까우며 완전히 엄밀한 설명은 아닙니다. 저자들도 “generated filters가 무엇을 encode하는지 직관적으로 보기 어렵다”고 말합니다.

셋째, 실패 사례에서는 noisy annotation이나 occlusion이 문제로 남습니다. appendix snippet에 따르면 일부 오류는 COCO annotation noise와 severe occlusion에서 발생합니다. 즉 CondInst가 구조적으로 우수해도, 복잡한 장면 이해와 annotation quality 문제를 완전히 해결하는 것은 아닙니다.

### 비판적 해석

제 해석으로, 이 논문의 가장 큰 공헌은 “instance segmentation에서 정말 ROI가 필요한가?”라는 질문에 대해 설득력 있는 **아니오**를 제시했다는 점입니다. 이후 세대의 query-based segmentation이나 dynamic-parameterized mask prediction 계열 연구를 볼 때도, CondInst는 분명한 전환점처럼 읽힙니다. 즉 단순히 성능표 하나 좋아진 논문이 아니라, **mask prediction을 per-instance conditional function으로 보는 관점**을 강하게 밀어붙인 논문입니다.

다만 오늘 관점에서 보면, CondInst는 transformer/query-based mask methods가 본격화되기 직전 시기의 중요한 bridge 역할을 합니다. ROI-free이지만 여전히 dense detector 기반이고, instance conditioning을 filter generation으로 수행합니다. 그래서 후속 연구를 이해할 때도 역사적으로 매우 중요한 위치를 차지합니다. 이 평가는 논문의 구조와 실험 결과에 근거한 해석입니다.  

## 6. Conclusion

CondInst는 instance segmentation과 panoptic segmentation을 위해 **ROI를 없애고 dynamic conditional convolutions를 도입한 단순하지만 강력한 framework**입니다. 핵심은 각 인스턴스마다 controller가 mask head의 filter를 생성하고, relative coordinates와 bottom features를 함께 사용해 전체 고해상도 feature map 위에서 마스크를 직접 예측하는 것입니다. 이 설계는 ROI cropping, feature alignment, fixed-size resizing의 부담을 제거하며, 더 가벼운 mask head로 더 좋은 품질과 더 안정적인 추론 시간을 제공합니다.  

실무적으로는 instance segmentation을 더 단순하고 빠르게 만들 수 있는 길을 열었고, 연구적으로는 dynamic parameter generation이 dense prediction에서도 얼마나 강력할 수 있는지를 보여준 대표 사례입니다. 특히 panoptic segmentation으로의 확장까지 자연스럽다는 점에서, 이 논문은 단일 task 개선을 넘어 **unified segmentation architecture**의 방향성을 제시한 작업으로 볼 수 있습니다.
