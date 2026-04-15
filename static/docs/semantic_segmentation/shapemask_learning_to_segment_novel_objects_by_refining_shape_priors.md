# ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors

- **저자**: Weicheng Kuo, Anelia Angelova, Jitendra Malik, Tsung-Yi Lin
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1904.03239

## 1. 논문 개요

이 논문은 **instance segmentation이 새로운 카테고리(novel categories)에도 잘 일반화되도록 만드는 문제**를 다룬다. 기존의 대표적 방법들은 대부분 각 카테고리마다 정확한 pixel-level mask annotation을 많이 필요로 한다. 그런데 실제 환경에서는 새로운 물체 범주가 계속 등장하고, 모든 범주에 대해 mask를 수집하는 것은 비용이 매우 크다. 논문은 바로 이 지점, 즉 **mask annotation이 없는 카테고리에도 segmentation을 얼마나 잘 확장할 수 있는가**를 핵심 연구 문제로 삼는다.

저자들은 기존 detection-based 방법이 box를 기반으로 mask를 예측하기 때문에, 물체의 자세(pose)나 윤곽(shape)에 대한 중간 표현이 약하다고 본다. 반대로 grouping-based 방법은 class-agnostic한 성격이 있지만, 실제로는 semantic segmentation 같은 추가 supervision에 의존하는 경우가 많아 novel class generalization이 충분하지 않다고 지적한다. 이 논문은 이 둘의 장점을 일부 결합하는 방향으로, **shape prior와 instance embedding을 이용해 box를 점진적으로 mask로 정제(refine)** 하는 새로운 방법인 **ShapeMask**를 제안한다.

문제의 중요성은 분명하다. 자율주행, 로봇 조작 같은 실제 응용에서는 학습 시점에 등장하지 않았던 물체를 만나는 일이 흔하다. 이런 상황에서 instance segmentation이 특정 학습 클래스에만 강하고 새로운 물체에는 약하다면 활용성이 크게 제한된다. 따라서 이 논문은 단순히 COCO 성능을 높이는 것이 아니라, **“wild” 환경에서 더 넓은 visual world를 다룰 수 있는 segmentation**이라는 더 큰 목표를 겨냥하고 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **object box보다 더 풍부한 중간 표현으로 object shape를 먼저 추정하고, 이후 instance-specific appearance 정보로 세부 mask를 다듬는 것**이다. 즉, 처음부터 픽셀 단위 정밀 mask를 바로 예측하려 하지 않고, 다음과 같은 단계적 구조를 취한다.

첫째, detection box를 단순한 사각형으로 보지 않고, 학습 데이터의 mask들에서 추출한 **shape priors**의 조합으로 대략적인 물체 형태를 추정한다. 저자들의 직관은 서로 다른 카테고리라도 비슷한 shape를 공유하는 경우가 많다는 것이다. 예를 들어 horse와 zebra, apple과 orange처럼 외형 구조가 유사한 경우가 있으므로, shape를 매개 표현으로 학습하면 novel category에도 도움이 될 수 있다고 본다.

둘째, 이렇게 얻은 coarse shape는 object-like한 강한 단서를 제공하지만, 개별 인스턴스의 세부 appearance까지 담지는 못한다. 그래서 논문은 coarse mask 내부의 feature를 average pooling하여 **instance embedding**을 만들고, 이를 이용해 해당 인스턴스에 맞는 세밀한 mask로 다시 정제한다. 이 부분은 grouping-based 접근의 “같은 물체에 속하는 픽셀을 묶는다”는 발상을 가져온 것으로 볼 수 있다.

기존 접근과의 차별점은 분명하다. Mask R-CNN류 방법은 box 내부에서 직접 mask를 예측하는 경향이 강하고, novel category generalization에서는 여전히 fully supervised setup과 큰 격차가 있었다. 이 논문은 **shape prior를 명시적으로 학습하는 중간 단계**를 넣어 출력 공간을 더 구조화하고, **instance embedding으로 개체별 refinement**를 수행함으로써 일반화를 높인다. 또한 feature cropping 기반 설계 대신 단순 crop, jittered ground-truth training, one-stage detector(RetinaNet) 등을 사용해 **효율성까지 함께 추구**했다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 세 단계다. 먼저 detection box에서 **shape recognition**을 수행해 detection prior를 만든다. 다음으로 detection prior와 image feature를 결합해 **coarse mask**를 예측한다. 마지막으로 coarse mask 안의 feature를 이용해 **instance embedding 기반 refinement**를 수행해 최종 fine mask를 만든다.

### 3.1 Shape priors와 shape recognition

저자들은 학습 데이터의 instance mask들을 $32 \times 32$ 크기로 정규화한 뒤, 각 클래스별로 k-means clustering을 수행하여 $K$개의 centroid를 구한다. 이것이 **shape priors**다. 클래스별 설정에서는 총 prior 수가 $C \times K$이고, class-agnostic 설정에서는 모든 클래스를 합쳐 하나의 foreground로 보고 더 큰 $K$를 사용한다. 논문은 이 prior 집합을 $H = \{S_1, S_2, ..., S_K\}$로 둔다.

입력으로 detection box가 주어지면, 이를 먼저 binary heatmap $B$로 표현한다. 그리고 backbone/FPN feature map $X$에서 box 내부 feature를 average pooling하여 box embedding $x_{\text{box}}$를 만든다.

$$
x_{\text{box}} = \frac{1}{|B|} \sum_{(i,j)\in B} X(i,j)
$$

이 embedding을 이용해 각 shape prior의 가중치 $w_k$를 예측한다. 구체적으로는 선형층 $\phi$를 적용한 뒤 softmax로 정규화한다.

$$
w_k = \text{softmax}(\phi_k(x_{\text{box}}))
$$

최종 shape는 prior들의 가중합으로 만든다.

$$
S = \sum_{k=1}^{K} w_k S_k
$$

이 $S$를 detection box 크기에 맞게 resize하여 부드러운 heatmap 형태의 **detection prior** $S_{\text{prior}}$를 만든다. 학습은 ground-truth mask $S_{\text{gt}}$와의 pixel-wise MSE로 한다.

$$
L_{\text{prior}} = \text{MSE}(S_{\text{prior}}, S_{\text{gt}})
$$

이 단계의 의미는 중요하다. 단순 box는 물체의 내부 구조를 거의 담지 못하지만, learned shape prior는 “이 box 안에 어떤 물체 형태가 있을 법한가”를 훨씬 구체적으로 제시한다. 저자들은 이것이 비현실적인 broken mask를 줄이고, novel object에도 더 잘 일반화하게 만든다고 해석한다.

### 3.2 Coarse mask prediction

다음 단계에서는 $S_{\text{prior}}$를 image feature와 결합해 coarse mask를 만든다. 먼저 $S_{\text{prior}}$를 $1 \times 1$ convolution 함수 $g$로 feature 공간에 embedding한 뒤, 원래 feature $X$와 더해 prior-conditioned feature $X_{\text{prior}}$를 만든다.

$$
X_{\text{prior}} = X + g(S_{\text{prior}})
$$

이제 이 feature는 이미지 appearance와 shape prior 정보를 동시에 담고 있다. 이후 네 개의 convolution layer로 구성된 decoder $f$를 적용해 coarse mask를 예측한다.

$$
S_{\text{coarse}} = f(X_{\text{prior}})
$$

학습은 pixel-wise cross-entropy를 사용한다.

$$
L_{\text{coarse}} = \text{CE}(S_{\text{coarse}}, S_{\text{gt}})
$$

논문은 이 단계를 Mask R-CNN의 mask decoder와 유사하다고 설명하지만, 중요한 차이는 **shape prior가 decoding을 guide한다**는 점이다. 즉, decoder가 완전히 맨땅에서 mask를 구성하는 것이 아니라, 이미 object-like shape 힌트를 가진 상태에서 coarse prediction을 수행한다.

### 3.3 Instance embedding을 이용한 fine mask refinement

coarse mask는 대략적인 shape는 잘 주지만, 여전히 instance-specific한 세부 구조를 충분히 반영하지 못할 수 있다. 그래서 논문은 coarse mask 안의 feature를 이용해 해당 인스턴스를 대표하는 embedding $x_{\text{mask}}$를 만든다. 먼저 soft coarse mask를 binarize하고, 그 내부 위치의 $X_{\text{prior}}$를 average pooling한다.

$$
x_{\text{mask}} = \frac{1}{|S_{\text{coarse}}|} \sum_{(i,j)\in S_{\text{coarse}}} X_{\text{prior}}(i,j)
$$

그 다음 모든 픽셀 위치에서 이 embedding을 빼서 centered instance feature $X_{\text{inst}}$를 만든다.

$$
X_{\text{inst}}(i,j) = X_{\text{prior}}(i,j) - x_{\text{mask}}
$$

이 연산은 현재 관심 인스턴스 기준으로 feature를 conditioning하는 역할을 한다. 저자들의 설명대로, 모델이 “이 물체와 비슷한 appearance를 가진 픽셀은 무엇인가”를 더 단순한 저차원 표현으로 학습하도록 유도한다.

이후 coarse 단계와 유사한 decoder를 사용하되, 추가 upsampling layer를 넣어 더 높은 해상도의 **fine mask** $S_{\text{fine}}$를 출력한다. 학습 손실은 역시 cross-entropy다.

$$
L_{\text{fine}} = \text{CE}(S_{\text{fine}}, S_{\text{gt}})
$$

즉, 전체 구조는 shape prior로 거친 형상을 만들고, instance embedding으로 그것을 세밀하게 조정하는 2단 refinement 구조라고 볼 수 있다.

### 3.4 Class-agnostic learning과 generalization

novel category generalization을 위해 저자들은 mask branch를 **class-agnostic**하게 학습한다. box branch는 모든 class에 대한 detection score를 출력하지만, mask branch는 class를 알지 못한 채 foreground mask만 예측한다. class-agnostic 설정에서는 모든 클래스의 instance mask를 모아 shape prior를 구성한다. 테스트 시 novel object도 같은 foreground category처럼 처리한다.

이 방식의 장점은 transfer learning처럼 class-specific mask head를 새로 생성하는 것이 아니라, **shape와 grouping 원리를 직접 foreground 수준에서 학습한다**는 점이다. 논문은 이것이 novel categories에 더 강한 일반화를 만든다고 주장한다.

### 3.5 구현 세부사항

검출기는 one-stage detector인 **RetinaNet**을 사용한다. 입력 해상도는 $1024 \times 1024$이며, multiscale training을 적용한다. 학습 시에는 proposal 대신 **ground-truth box를 고정 개수 샘플링**하고, 여기에 Gaussian noise를 주어 jittered box를 만든다. 이는 테스트 시 imperfect detection을 더 잘 견디게 하기 위한 설계다. box 중심과 크기는 다음처럼 변형된다.

$$
(x'_c, y'_c) = (x_c + \delta_x w,\; y_c + \delta_y h)
$$

$$
(w', h') = (e^{\delta_w} w,\; e^{\delta_h} h)
$$

여기서 $\delta$들은 평균 0, 표준편차 0.1의 Gaussian noise다.

RoI feature는 FPN의 $P_3$부터 $P_5$까지 사용하며, box 크기에 따라 적절한 pyramid level을 선택한다.

$$
k = m - \lfloor \log_2 \frac{L}{\max(\text{box}_h, \text{box}_w)} \rfloor
$$

그리고 해당 level에서 box 중심 기준의 고정 크기 feature patch를 가져온다. 논문은 이 방식이 ROIAlign 같은 crop-and-resize 없이도 scale normalization을 가능하게 하며, TPU/GPU 친화적이라고 설명한다.

## 4. 실험 및 결과

### 4.1 실험 설정

주요 실험은 COCO에서 수행되며, 평가 지표는 표준 COCO instance segmentation metric인 AP, $AP_{50}$, $AP_{75}$, 그리고 object size별 AP다. 일반화 실험에서는 [20]을 따라 COCO category를 VOC와 Non-VOC로 나누고, 한쪽만 mask annotation을 제공한 뒤 다른 쪽에서 segmentation 성능을 평가한다. 즉, 학습 시 모든 category의 box는 보지만, mask는 일부 category에서만 본다.

### 4.2 Novel category generalization 결과

이 논문의 핵심 결과는 부분 지도(partially supervised) 설정에서 매우 강하다. ResNet-101-FPN backbone 기준으로 ShapeMask는:

- VOC에서 mask를 학습하고 Non-VOC에서 테스트할 때 **30.2 AP**
- Non-VOC에서 mask를 학습하고 VOC에서 테스트할 때 **33.3 AP**

를 기록했다. 같은 설정에서 기존 state-of-the-art인 Mask X R-CNN [20]은 각각 23.8 AP, 29.5 AP였으므로, ShapeMask는 **각각 6.4 AP, 3.8 AP 더 높다**.

더 중요한 것은 oracle과의 격차다. VOC→Non-VOC에서 ShapeMask는 oracle 대비 4.8 AP 차이, Non-VOC→VOC에서 7.6 AP 차이였고, Mask X R-CNN은 각각 10.6 AP, 9.6 AP 차이였다. 즉, ShapeMask는 단순히 수치가 더 높은 것뿐 아니라, **fully supervised upper bound에 더 가까운 일반화**를 보였다.

NAS-FPN backbone을 쓰면 성능은 더 올라간다. 이 경우 Mask X R-CNN 대비 **9.4 AP, 6.2 AP** 더 높았다고 보고한다. 이는 ShapeMask가 stronger backbone의 이점을 잘 흡수한다는 뜻이다.

정성적으로도 저자들은 ShapeMask가 novel category에 대해 더 complete하고 object-like한 mask를 예측한다고 보여준다. 특히 Mask R-CNN 계열이 broken pieces처럼 부분 조각만 예측하는 경우, ShapeMask는 전체 윤곽을 유지하는 경향이 있다고 해석한다.

### 4.3 적은 데이터에서의 일반화

저자들은 VOC mask만 이용해 학습하고 Non-VOC에서 평가하는 설정에서, 학습 데이터를 $1/2$부터 $1/1000$까지 줄여가며 실험했다. 결과적으로 ShapeMask는 **극단적으로 적은 데이터에서도 성능 저하가 완만**했다. 특히 전체 훈련 데이터의 **1/100만 사용해도**, 기존 state-of-the-art인 Mask X R-CNN이 전체 데이터를 쓴 경우보다 **2.0 AP 높았다**고 보고한다. 이는 이 방법이 sample efficiency 측면에서도 강하다는 근거로 제시된다.

### 4.4 Robotics 데이터로의 외부 일반화

논문은 COCO 밖의 robotics grasping 데이터에서도 정성적 실험을 수행한다. 이 데이터는 COCO에 없는 office object나 구조물이 포함되고, instance mask annotation은 없다. 저자들은 detector 성능 영향을 줄이기 위해 ground-truth box를 입력으로 사용하고 segmentation만 평가했다. 논문은 plush toy, document, tissue box 같은 novel object도 잘 분할한다고 보여준다.

다만 이 부분은 정량 표가 아니라 **정성적 시각화 중심**이며, 구체적인 수치 평가는 제시되지 않았다. 따라서 “외부 데이터에서도 잘 된다”는 주장은 시각적 예시로는 설득력이 있지만, 정량적 강도를 판단하려면 추가 실험이 더 필요하다.

### 4.5 Fully supervised instance segmentation 결과

논문은 일반화가 주목적이지만, fully supervised 설정에서도 경쟁력을 보인다고 주장한다. COCO test-dev2017에서 ResNet-101-FPN backbone 기준 ShapeMask는 **37.4 AP**를 기록했고, 같은 backbone의 Mask R-CNN은 **35.7 AP**였다. 즉 **1.7 AP 개선**이다.

더 강한 ResNet-101-NAS-FPN에서는 ShapeMask가 **40.0 AP**를 기록해, 표에 제시된 Mask R-CNN 37.1 AP와 MaskLab 37.3 AP보다 각각 **2.9 AP, 2.7 AP 높다**. PANet 42.0 AP보다는 낮지만, 논문은 자신들은 atrous convolution, deformable crop and resize, heavier head, mask refinement 같은 추가 기법을 쓰지 않았다고 강조한다. 즉, 비교적 단순한 구성으로도 충분히 강하다는 주장이다.

### 4.6 효율성과 강건성

효율 측면에서 ShapeMask는 훈련 시간이 매우 짧다. 논문은 TPU에서 **11시간**에 학습을 마치며, 이는 Mask R-CNN 계열보다 약 **4배 빠르다**고 주장한다. 추론 속도는 ResNet-101 모델 기준 이미지당 약 **150ms 수준(125ms GPU + 24ms CPU)** 으로 보고된다.

강건성 분석도 흥미롭다. 저자들은 테스트 시 detection box의 width와 height를 무작위로 줄여 localization error를 인위적으로 만든다. 이때 Mask R-CNN은 예측 mask가 box 내부에 제한되기 때문에 성능이 크게 떨어지지만, ShapeMask는 detection을 soft prior로만 사용하므로 상대적으로 안정적이다. 표 3에 따르면 jittered detection에서:

- Our Mask R-CNN: 36.4 → 29.0 AP
- ShapeMask: 37.2 → 34.3 AP
- jittering training을 추가한 ShapeMask: 37.2 → 35.7 AP

즉, ShapeMask는 같은 조건에서 Mask R-CNN보다 **5.3 AP 더 robust**했다.

### 4.7 Ablation

부분 지도 설정의 ablation 결과는 이 방법의 핵심이 무엇인지 잘 보여준다. shape prior도 instance embedding도 없는 baseline은 VOC→Non-VOC에서 **13.7 AP**, Non-VOC→VOC에서 **24.8 AP**였다. 여기에 shape prior만 넣으면 각각 **26.2 AP**, **29.4 AP**, embedding만 넣으면 **26.4 AP**, **30.6 AP**로 크게 상승한다. 둘을 함께 쓰면 최종적으로 **30.2 AP**, **33.3 AP**가 된다.

fully supervised ablation에서도 baseline 35.5 AP에서 shape prior 또는 embedding 각각이 성능을 올리고, 둘을 함께 쓰면 **37.2 AP**가 된다. 따라서 shape prior와 instance embedding은 모두 중요한 구성 요소이며, 상호 보완적이라는 것이 논문의 결론이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **novel category generalization 문제를 단순한 transfer trick이 아니라 구조적 표현 학습 문제로 다시 정의했다는 점**이다. shape prior를 중간 표현으로 넣어 box보다 풍부한 정보를 제공하고, instance embedding으로 개체 수준 refinement를 하는 설계는 논리적으로 일관되고 실험적으로도 잘 뒷받침된다. 특히 partially supervised setting에서 큰 AP 향상은 이 논문의 핵심 기여를 강하게 지지한다.

둘째 강점은 **효율성과 실용성**이다. 논문은 정확도뿐 아니라 TPU/GPU 친화적인 구현을 함께 제안한다. ROIAlign 대신 단순 crop, proposal-based training 대신 jittered ground-truth training, one-stage RetinaNet 사용 등은 모두 대규모 학습 환경에서 현실적인 장점이 있다. 또한 detection noise에 대한 robustness 분석도 실제 적용 측면에서 의미가 있다.

셋째, **모델 용량이 줄어도 성능 저하가 완만하다**는 결과도 중요하다. mask branch 채널 수를 크게 줄여도 경쟁력 있는 AP를 유지하는 것은, shape prior가 출력 공간을 구조화해주기 때문에 가능한 현상으로 해석할 수 있다.

반면 한계도 있다. 첫째, shape prior는 기본적으로 학습 데이터의 mask distribution에서 얻은 cluster centroid에 의존한다. 따라서 novel object가 기존 shape manifold와 크게 다르면 효과가 제한될 수 있다. 논문도 이 경우를 직접 분석하지는 않는다. 즉, “shape similarity”가 일반화의 핵심 가정인데, 이 가정이 어느 정도까지 성립하는지는 더 검증이 필요하다.

둘째, robotics 데이터에 대한 일반화 주장은 정량 평가 없이 정성 예시 중심이다. 시각적으로는 인상적이지만, domain shift 상황에서의 일반화 성능을 강하게 주장하려면 더 체계적인 benchmark가 필요하다.

셋째, class-agnostic mask branch가 novel category generalization에는 유리하지만, 매우 fine-grained한 category-specific shape 차이가 중요한 경우에는 한계가 있을 수 있다. 논문은 fully supervised에서도 경쟁력이 있다고 보이지만, 최고 성능 시스템과의 비교에서는 PANet 같은 더 강한 방법에 여전히 뒤처진다.

넷째, shape prior를 만들기 위해 k-means를 사용하고 고정된 prior bank를 구성하는 방식은 단순하고 해석 가능하지만, end-to-end로 더 유연한 latent shape model을 학습하는 방식에 비해 표현력이 제한될 가능성도 있다. 다만 이것은 논문이 직접 실험한 내용은 아니므로, 가능한 비판적 해석 수준에서만 말할 수 있다.

## 6. 결론

이 논문은 **instance segmentation의 일반화를 위해 shape prior와 instance embedding을 결합한 ShapeMask**를 제안했다. 핵심은 detection box를 바로 정답 mask로 바꾸려 하지 않고, 먼저 plausible한 object shape를 복원한 뒤, 해당 인스턴스의 appearance 정보로 정교하게 refinement하는 것이다. 이 구조는 novel category에 대해 특히 효과적이었고, COCO partially supervised setting에서 기존 방법보다 큰 폭의 성능 향상을 보였다.

동시에 ShapeMask는 fully supervised setting에서도 경쟁력 있는 정확도를 유지했고, detection noise에 더 robust하며, 학습과 추론 효율도 높았다. 따라서 이 연구의 의미는 단지 하나의 segmentation 모델 제안에 그치지 않는다. **instance segmentation에서 “shape라는 중간 표현”이 generalization과 efficiency를 동시에 개선할 수 있다**는 점을 보여준다는 데 더 큰 가치가 있다. 실제 응용에서는 새로운 물체를 계속 만나게 되므로, 이런 방향은 로보틱스나 오픈월드 인식 환경에서 특히 중요하며, 향후에는 더 유연한 shape prior 학습이나 더 강한 open-vocabulary segmentation으로 이어질 가능성이 크다.
