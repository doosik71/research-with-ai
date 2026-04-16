# Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers

- **저자**: Zhiqi Li, Wenhai Wang, Enze Xie, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo, Tong Lu
- **발표연도**: 본문 발췌문에 명시되지 않음
- **arXiv**: 본문 발췌문에 명시되지 않음

## 1. 논문 개요

이 논문은 panoptic segmentation을 위한 transformer 기반 프레임워크인 **Panoptic SegFormer**를 제안한다. Panoptic segmentation은 이미지의 모든 픽셀에 대해 semantic category를 부여하면서, `thing` 클래스에 대해서는 서로 다른 instance id까지 함께 예측해야 하는 문제이다. 다시 말해 semantic segmentation과 instance segmentation을 하나의 통합된 형식으로 다루는 과제다.

저자들이 겨냥한 핵심 문제는 기존 transformer 기반 panoptic segmentation 방법, 특히 DETR 계열 접근이 갖는 세 가지 한계이다. 첫째, 학습 수렴이 매우 느리다. 둘째, self-attention의 계산량 문제 때문에 encoder에서 높은 해상도와 multi-scale feature를 충분히 다루기 어렵다. 셋째, `thing`과 `stuff`를 동일한 query 집합과 동일한 방식으로 처리하는 설계가 panoptic segmentation의 본질에 꼭 맞지 않는다. 특히 `stuff`는 amorphous하고 countable instance가 없는 영역인데, 이를 `thing`과 같은 방식으로 box 중심으로 다루는 것은 비효율적일 수 있다.

이 문제가 중요한 이유는 panoptic segmentation이 장면 전체를 이해하는 핵심 과제이기 때문이다. 자율주행, 로보틱스, 장면 이해처럼 이미지 안의 모든 요소를 빠짐없이 이해해야 하는 응용에서, 단순한 detection이나 semantic segmentation만으로는 부족하다. 따라서 높은 정확도뿐 아니라, 효율적 학습, 안정적 추론, 그리고 `thing`과 `stuff`의 성질 차이를 반영한 구조 설계가 중요하다.

이 논문은 이러한 문제를 해결하기 위해 세 가지 핵심 요소를 제안한다. 하나는 **deeply-supervised mask decoder**, 둘째는 **query decoupling strategy**, 셋째는 **mask-wise merging** 기반의 후처리이다. 여기에 Deformable DETR의 deformable attention을 도입해 multi-scale feature를 더 효율적으로 처리한다. 논문에 따르면 이 조합으로 COCO에서 baseline DETR 대비 **6.2% PQ 향상**을 달성했고, COCO test-dev에서 **56.2% PQ**를 기록했다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 panoptic segmentation을 transformer로 수행할 때, 단순히 detection용 query-decoder 구조를 가져오는 것만으로는 충분하지 않다는 것이다. 저자들은 특히 세 가지 관찰에서 출발한다.

첫째, **mask decoder 내부 attention이 의미 있는 영역을 빨리 보도록 강하게 지도해 주는 것이 중요하다**고 본다. 기존 DETR류는 attention이 초기에는 거의 모든 위치를 고르게 바라보며, 학습이 오래 걸린다. Panoptic SegFormer는 attention map 자체가 mask prediction에 직접 연결되도록 설계하고, 각 decoder layer에 deep supervision을 걸어 attention이 더 이른 단계에서 target region에 집중하도록 만든다. 이 아이디어는 성능뿐 아니라 수렴 속도 개선에도 직접 연결된다.

둘째, **`thing`과 `stuff`를 같은 query set으로 처리하는 것은 suboptimal**하다는 점이다. `thing`은 개별 instance를 구분해야 해서 위치 단서와 instance-level reasoning이 중요하지만, `stuff`는 category-level dense region 예측이 핵심이다. 그래서 저자들은 query를 두 집합으로 분리한다. `thing query`는 bipartite matching을 통해 object-like target과 대응시키고, `stuff query`는 각 query가 특정 stuff class에 고정 대응되도록 한다. 이렇게 하면 한 query 내부에서 `thing`과 `stuff`가 서로 간섭하는 문제를 줄일 수 있다.

셋째, **pixel-wise argmax 후처리는 극단적인 mask logit 값 때문에 false positive를 만들 수 있다**는 점이다. 기존 방법은 각 픽셀마다 가장 큰 logit을 갖는 mask를 택하는데, 이 방식은 mask 전체 품질이나 class confidence를 잘 반영하지 못한다. 저자들은 대신 mask 단위로 우선순위를 정하고, 높은 confidence mask부터 canvas에 채워 넣는 **mask-wise merging**을 제안한다. 이때 confidence는 classification probability와 segmentation quality를 함께 사용한다.

기존 접근과의 차별점은 분명하다. DETR는 detection 중심 구조 위에 panoptic head를 얹었고, MaskFormer는 pixel decoder와 query-feature multiplication을 사용하지만, 이 논문은 **attention map 자체를 mask 생성의 중심 표현으로 삼고**, 여기에 **deep supervision**, **thing/stuff query 분리**, **mask-level conflict resolution**을 결합했다. 즉, transformer를 쓰되 panoptic segmentation의 구조적 특성에 맞게 decoder와 inference를 더 깊게 재설계한 것이 핵심 차별점이다.

## 3. 상세 방법 설명

전체 구조는 **backbone + transformer encoder + location decoder + mask decoder**로 이루어진다. 입력 이미지 $X \in \mathbb{R}^{H \times W \times 3}$를 backbone에 넣으면 마지막 세 stage의 feature map $C_3, C_4, C_5$를 얻는다. 이들의 해상도는 각각 입력 대비 $1/8$, $1/16$, $1/32$이다. 각 feature는 FC layer로 256 채널로 투영되고 flatten되어 token sequence로 바뀐다. 이후 이 multi-scale token들을 transformer encoder에 넣어 refinement한 feature $F$를 만든다.

여기서 encoder는 일반 self-attention 대신 **deformable attention**을 사용한다. 논문이 강조하는 이유는 명확하다. panoptic segmentation에는 고해상도와 multi-scale 정보가 중요한데, 일반 self-attention은 계산량이 너무 커서 이를 처리하기 어렵다. deformable attention을 쓰면 낮은 비용으로 multi-scale feature를 encoder에 포함시킬 수 있다.

그 다음 query 단계에서 이 논문의 중요한 설계가 나온다. 저자들은 query를 두 종류로 나눈다. **$N_{th}$개의 thing query**와 **$N_{st}$개의 stuff query**다. Thing query는 추가적인 **location decoder**를 거쳐 location-aware query가 되고, stuff query는 별도 location refinement 없이 mask decoder로 들어간다.

### Query Decoupling Strategy

이 전략의 목적은 `thing`과 `stuff`의 책임을 분리하는 것이다. 기존 방법은 하나의 query set이 `thing`과 `stuff` 모두를 jointly match했다. 하지만 저자들은 이것이 서로 다른 속성을 가진 target을 한 표현 공간에서 경쟁시키기 때문에 특히 $PQ^{st}$를 해친다고 본다.

Panoptic SegFormer에서는 `thing query`가 오직 thing만 예측하고, `stuff query`는 오직 stuff만 예측한다. 학습 시 `thing` ground truth는 Hungarian algorithm 기반 **bipartite matching**으로 할당된다. 반면 `stuff`는 **class-fixed assignment**를 사용해 각 stuff query가 하나의 stuff class에 일대일 대응한다. 이 설계 덕분에 thing/stuff 간 간섭을 줄이면서도, 최종 출력 형식은 category와 mask라는 동일한 형식을 유지할 수 있어 후처리는 통일적으로 처리할 수 있다.

### Location Decoder

Location decoder는 thing query에만 적용된다. 목적은 `thing` 인스턴스를 구별하는 데 중요한 위치 정보를 query에 주입하는 것이다. 입력은 randomly initialized된 $N_{th}$개의 thing query와 encoder가 만든 refined feature token이다. 출력은 **location-aware query**다.

학습 시 location decoder 위에는 auxiliary MLP head가 붙어 bounding box와 category를 예측하고, 이에 대해 detection loss $L_{det}$를 적용한다. 이 branch는 inference 때는 버릴 수 있다. 저자들은 이 decoder가 bounding box 대신 mask의 mass center를 예측하는 방식으로도 학습 가능하며, 이 경우에도 비슷한 성능을 낸다고 보고한다. 즉, 이 모듈의 본질은 detection 자체보다 **query가 위치 단서를 학습하도록 돕는 것**에 있다.

### Mask Decoder

Mask decoder는 최종 category와 mask를 예측하는 핵심 모듈이다. 입력 query $Q$는 location-aware thing query와 class-fixed stuff query를 합친 것이다. Key와 value는 encoder의 refined feature $F$에서 projection된다.

각 decoder layer는 attention map $A$와 refined query $Q_{refine}$를 만든다. 여기서 $A \in \mathbb{R}^{N \times h \times (L_1 + L_2 + L_3)}$이며, $N = N_{th} + N_{st}$, $h$는 attention head 수다. Category prediction은 각 layer의 refined query 위에 FC layer를 올려 수행한다. Thing query는 모든 thing class에 대한 확률을 예측하고, stuff query는 자기에게 고정된 stuff category의 확률만 예측한다.

Mask prediction은 attention map을 multi-scale spatial map으로 다시 재구성하는 방식으로 이뤄진다. 먼저 attention map을 세 해상도에 대응하는 $A_3, A_4, A_5$로 split하고 reshape한다.

$$
(A_3, A_4, A_5) = Split(A), \quad A_i \in \mathbb{R}^{\frac{H}{2^{i+2}} \times \frac{W}{2^{i+2}} \times h}
$$

그 다음 이들을 $H/8 \times W/8$ 해상도로 upsample한 뒤 channel 방향으로 concatenate한다.

$$
A_{fused} = Concat(A_3, Up_{\times 2}(A_4), Up_{\times 4}(A_5))
$$

이 fused attention map에서 $1 \times 1$ convolution으로 binary mask를 예측한다. 중요한 점은 이 head가 매우 가볍다는 것이다. 논문은 이 head가 약 200개의 파라미터만 가진다고 설명한다. 이렇게 가볍게 설계한 이유는 attention map이 곧바로 mask 정보와 연결되도록 해서, attention이 정말로 의미 있는 영역에 집중하도록 만들기 위해서다.

### Deep Supervision

이 mask decoder의 또 하나의 핵심은 **layer-wise deep supervision**이다. 각 decoder layer의 attention map과 classification output에 대해 supervision을 주기 때문에, attention module이 초반 layer부터 의미 있는 semantic region을 학습하게 된다. 논문은 이것이 수렴 가속의 핵심이라고 주장한다. 실제로 attention map 시각화에서도 deep supervision이 있을 때 앞선 layer부터 target object에 집중하는 현상을 보여준다.

### Loss Function

전체 손실은 다음과 같이 thing과 stuff 손실의 가중합으로 정의된다.

$$
L = \lambda_{things} L_{things} + \lambda_{stuff} L_{stuff}
$$

Thing 손실은 detection loss와 각 mask decoder layer의 classification, segmentation loss를 포함한다.

$$
L_{things} = \lambda_{det} L_{det} + \sum_{i}^{D_m} \left( \lambda_{cls} L^{i}_{cls} + \lambda_{seg} L^{i}_{seg} \right)
$$

여기서 $L^{i}_{cls}$는 Focal loss, $L^{i}_{seg}$는 Dice loss이며, $L_{det}$는 Deformable DETR의 detection loss이다. $D_m$은 mask decoder layer 수다.

Stuff 손실은 detection 항이 없고, 각 layer의 classification과 segmentation loss만 사용한다.

$$
L_{stuff} = \sum_{i}^{D_m} \left( \lambda_{cls} L^{i}_{cls} + \lambda_{seg} L^{i}_{seg} \right)
$$

즉, `thing`은 location decoder를 통해 detection supervision을 추가로 받고, `stuff`는 class-fixed query 위에서 dense mask/class supervision만 받는다.

### Mask-Wise Merging Inference

추론 시 저자들은 pixel-wise argmax 대신 **mask-wise merging**을 사용한다. 각 prediction마다 category $c$, confidence $s$, mask $m$이 있다고 할 때, confidence는 classification probability와 mask quality를 조합해 계산한다.

$$
s_i = p_i^{\alpha} \times \left( average\left( \mathbf{1}\{m_i[h,w] > 0.5\} \, m_i[h,w] \right) \right)^{\beta}
$$

여기서 $p_i$는 해당 prediction의 가장 높은 class probability이고, 두 번째 항은 threshold 0.5 이상인 영역 내부의 평균 mask logit으로서 segmentation quality를 나타낸다. $\alpha, \beta$는 두 항의 중요도를 조절하는 하이퍼파라미터다.

이후 confidence가 높은 mask부터 차례로 정렬해 non-overlap canvas에 채워 넣는다. confidence가 너무 낮은 prediction은 버리고, 이미 채워진 영역과 겹치는 부분은 제거한 뒤, 남는 유효 영역이 원래 mask 대비 충분히 크면 유지한다. 이 방식은 각 픽셀을 독립적으로 선택하는 대신, **mask 전체의 품질을 고려해 충돌을 해결**한다는 점이 핵심이다.

## 4. 실험 및 결과

논문은 COCO 2017과 ADE20K에서 panoptic segmentation 성능을 평가하고, COCO test-dev에서 instance segmentation 결과도 함께 제시한다. 또한 각 모듈의 기여를 확인하는 ablation study와, 자연적 corruption에 대한 robustness 실험도 수행한다.

### 데이터셋과 지표

COCO 2017은 118K train, 5K val 이미지를 가지며, 80개 thing class와 53개 stuff class를 포함한다. ADE20K는 100개 thing, 50개 stuff를 포함한다. 주요 평가지표는 panoptic segmentation의 **PQ**, 그리고 이를 `thing`과 `stuff`로 나눈 $PQ^{th}$, $PQ^{st}$이다. 추가로 SQ, RQ도 제시되며, instance segmentation에서는 $AP^{seg}$를 사용한다.

### COCO Panoptic Segmentation

COCO val에서 ResNet-50 backbone 기준 Panoptic SegFormer는 **49.6 PQ**를 기록했다. 이는 논문이 baseline으로 둔 DETR-R50의 **43.4 PQ**보다 **6.2%p** 높고, MaskFormer-R50의 **46.5 PQ**보다 **3.1%p** 높다. 같은 ResNet-50에서 24 epoch만 학습했는데도, 300 epoch 이상 학습한 MaskFormer나 325 epoch의 DETR보다 우수하다.

ResNet-101에서는 **50.6 PQ**, Swin-L에서는 **55.8 PQ**를 기록했고, COCO test-dev에서는 Swin-L backbone으로 **56.2 PQ**를 달성했다. 이는 표에 제시된 MaskFormer-Swin-L의 **53.3 PQ**, K-Net-Swin-L의 **55.2 PQ**보다 높다. PVTv2-B5 backbone으로도 **55.8 PQ**를 달성했는데, Swin-L보다 파라미터와 FLOPs가 훨씬 적다. 즉, 정확도뿐 아니라 효율 면에서도 장점이 있다.

### ADE20K 결과

ADE20K val에서는 ResNet-50 backbone으로 **36.4 PQ**를 달성했다. 이는 MaskFormer-R50의 **34.7 PQ**보다 **1.7%p** 높다. 논문은 이를 통해 제안 방법이 COCO에만 맞춘 특수한 설계가 아니라, 다른 panoptic dataset에도 일반화된다고 주장한다.

### Instance Segmentation 결과

Stuff query를 제거하면 instance segmentation 모델로 바로 바꿀 수 있다. COCO test-dev에서 ResNet-50 기준, crop 없이 **40.4 APseg**, crop 사용 시 **41.7 APseg**를 기록했다. 이는 QueryInst의 40.6, HTC의 39.7, K-Net의 38.6보다 경쟁력 있는 결과다. 저자들은 이 점을 통해 panoptic segmentation용 설계가 instance segmentation에도 유리한 표현을 학습한다고 해석한다.

### 모듈별 Ablation

모듈을 단계적으로 추가한 Table 5는 이 논문의 기여를 가장 직접적으로 보여준다. baseline DETR-R50이 **43.4 PQ**, 325 epoch, 247.5G FLOPs, 4.9 FPS인데, 여기에 mask-wise merging만 추가하면 **44.7 PQ**로 오른다. deformable attention을 쓰면 **47.3 PQ**가 되고, mask decoder를 도입하면 **48.5 PQ**, 마지막으로 query decoupling까지 적용하면 **49.6 PQ**가 된다. 최종 모델은 학습 epoch가 24로 줄고, FLOPs는 214.2G, 속도는 7.8 FPS로 개선된다.

### Location Decoder 효과

Location decoder layer 수를 0에서 6까지 바꾸며 실험한 결과, layer 수가 증가할수록 주로 $PQ^{th}$가 상승한다. 0 layer에서는 **47.0 PQ**, 6 layer에서는 **49.6 PQ**다. box 대신 mass center를 예측하는 box-free 모델도 **49.2 PQ**로 근접한 성능을 보여, 이 모듈의 핵심이 box 자체보다 위치 정보 학습임을 뒷받침한다.

### Mask-Wise Merging 효과

후처리 비교에서도 일관된 개선이 나타난다. DETR는 pixel-wise argmax 사용 시 **43.4 Mask PQ**, mask-wise merging 사용 시 **44.7**로 상승한다. Panoptic SegFormer도 pixel-wise argmax일 때 **48.4**, mask-wise merging일 때 **49.6**으로 오른다. Boundary PQ도 함께 개선된다. 논문은 pixel-wise argmax가 비정상적으로 큰 일부 pixel logit에 끌려 false positive를 만든 사례를 제시하며, mask-wise merging이 이를 완화한다고 설명한다.

### Query Decoupling 효과

Joint matching과 query decoupling을 비교한 Table 8은 이 논문의 핵심 주장을 직접 보여준다. Joint matching은 **48.5 PQ, 39.5 PQst, 37.7 APseg**이고, Query decoupling은 **49.6 PQ, 42.4 PQst, 39.5 APseg**다. 특히 $PQ^{st}$가 크게 오른다. 저자들은 한 query가 thing을 더 선호할수록 stuff 예측 precision이 낮아지는 경향도 분석해, 하나의 query set이 두 역할을 동시에 맡을 때 실제로 간섭이 발생한다고 주장한다.

### 수렴 속도와 Decoder Layer별 결과

Convergence curve에서는 deep supervision이 있는 Panoptic SegFormer가 **24 epoch만으로 49.6 PQ**에 도달하며 이후 큰 향상이 없다고 보고한다. 반면 D-DETR-MS는 더 오래 학습해야 한다. Decoder 중간 layer의 출력만 사용해도 성능이 크게 떨어지지 않는 점도 흥미롭다. 1번째 layer 출력만 써도 **48.8 PQ**이며, 속도는 **10.6 FPS**로 더 빠르다. 이는 mask decoder가 초반 layer에서도 충분히 강한 attention 표현을 학습한다는 뜻이다.

### Robustness to Natural Corruptions

COCO-C에서의 robustness 실험도 제시된다. ResNet-50 기준 clean 데이터에서 Panoptic SegFormer는 **50.0 PQ**, corruption 평균에서 **32.9 PQ**를 기록해 Panoptic FCN, MaskFormer, D-DETR보다 높다. Swin-L 기준으로도 MaskFormer-Swin-L의 corruption 평균 **41.7** 대비 Panoptic SegFormer-Swin-L은 **47.2**를 기록한다. 논문은 transformer backbone의 robustness뿐 아니라, convolution-based pixel decoder보다 자신들의 transformer-based mask decoder가 더 robust할 가능성을 제기한다. 다만 이것은 논문의 해석이며, 이를 완전히 분리해 검증한 별도 실험은 발췌문 기준으로는 충분히 제시되지 않았다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 panoptic segmentation의 구조적 특성을 transformer 설계에 맞게 세밀하게 반영했다는 점이다. 단순히 더 큰 backbone이나 더 많은 연산으로 성능을 올린 것이 아니라, 왜 기존 DETR류가 panoptic segmentation에 완전히 최적화되지 않았는지 분석하고, 그 분석을 decoder, query assignment, 후처리 수준에서 구체적인 설계로 연결했다. 특히 query decoupling은 `thing`과 `stuff`의 본질적 차이를 분명히 구조에 반영한 점에서 설득력이 높다.

또 다른 강점은 **학습 효율과 성능을 동시에 개선했다는 점**이다. 논문은 baseline DETR 대비 학습 epoch를 크게 줄이면서도 PQ를 크게 높였다. Deep supervision이 attention 학습을 빠르게 만든다는 주장은 convergence curve와 attention map 시각화를 통해 비교적 잘 뒷받침된다. 또한 mask-wise merging은 별도 학습 없이 inference 단계에서 성능을 높이는 실용적 개선이다.

실험도 비교적 폭넓다. COCO, ADE20K, instance segmentation, corruption robustness, 다양한 backbone, 다수의 ablation을 포함한다. 단순히 최종 수치만 제시하지 않고, 각 모듈이 어떤 지표에 영향을 주는지도 나눠 보여준다. 예를 들어 location decoder가 주로 $PQ^{th}$에 기여하고, query decoupling이 특히 $PQ^{st}$를 끌어올린다는 점이 분리되어 드러난다.

한계도 명확히 언급된다. 저자들은 deformable attention에 의존하기 때문에 속도가 아주 빠르지는 않으며, 더 큰 spatial shape의 feature를 다루기 어렵다고 말한다. 또한 small target에 약하다고 인정한다. 시각화 분석에서도 crowded scene이나 작은 물체가 많은 경우 recall이 낮을 수 있다고 적고 있다. 이는 panoptic segmentation의 실제 응용, 특히 자율주행처럼 작은 객체가 중요한 환경에서는 중요한 약점이다.

또 하나의 한계는 후처리의 성격이다. Mask-wise merging은 효과적이지만, fixed threshold에 의존해 mask를 binarize하고 confidence score의 정확도에 민감하다. 논문도 confidence score가 부정확하면 낮은 품질의 panoptic mask가 생성될 수 있다고 스스로 인정한다. 즉, 후처리 개선이 성능 상승에 크게 기여하는 만큼, 이 부분이 완전히 원리적으로 해결된 것은 아니다.

비판적으로 보면, 이 논문은 “완전한 unified segmentation pipeline”보다 task-specific design이 더 낫다고 주장하는데, 이는 실험적으로 상당히 설득력 있다. 다만 이 주장이 범용적으로 항상 맞는지까지는 이 논문만으로 단정하기 어렵다. 또한 robustness 향상이 mask decoder 설계에서 비롯된다는 해석도 흥미롭지만, backbone 효과와 head 효과를 더 엄밀하게 분리한 분석이 있었다면 더 강한 주장이 되었을 것이다.

## 6. 결론

이 논문은 transformer 기반 panoptic segmentation을 더 깊이 설계한 프레임워크인 Panoptic SegFormer를 제안했다. 핵심 기여는 세 가지로 요약된다. 첫째, **deeply-supervised mask decoder**를 통해 attention map을 직접 mask 예측에 연결하고 빠른 수렴과 높은 mask 품질을 달성했다. 둘째, **query decoupling strategy**로 `thing`과 `stuff`를 분리해 상호 간섭을 줄이고 특히 stuff segmentation 품질을 높였다. 셋째, **mask-wise merging inference**로 classification confidence와 segmentation quality를 함께 반영해 더 안정적으로 non-overlap panoptic 결과를 만들었다.

실험적으로는 COCO와 ADE20K에서 강한 성능을 보였고, 적은 학습 epoch로도 경쟁 방법을 능가했다. 특히 COCO test-dev에서 56.2 PQ를 기록한 점은 당시 transformer 기반 panoptic segmentation에서 매우 강한 결과로 제시된다. 또한 instance segmentation으로의 전환과 corruption robustness 측면에서도 좋은 결과를 보였다.

실제 적용 관점에서 보면, 이 연구는 panoptic segmentation을 위한 transformer 설계가 detection의 단순 확장이 아니라는 점을 분명히 보여준다. 향후 연구에서는 더 빠르고 고해상도에 강한 attention 구조, small object 처리 개선, 후처리의 threshold 의존성 완화가 중요한 후속 과제가 될 가능성이 크다. 동시에 이 논문이 제안한 “공통된 표현은 유지하되, 필요한 차이는 구조적으로 분리한다”는 관점은 이후 unified segmentation 연구에도 중요한 시사점을 준다.
