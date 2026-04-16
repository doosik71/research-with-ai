# Per-Pixel Classification is Not All You Need for Semantic Segmentation

- **저자**: Bowen Cheng, Alexander G. Schwing, Alexander Kirillov
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2107.06278

## 1. 논문 개요

이 논문은 semantic segmentation을 반드시 per-pixel classification으로 풀 필요가 없다는 문제의식에서 출발한다. 기존의 주류 semantic segmentation 방법은 이미지의 각 픽셀마다 클래스 분포를 예측하는 방식, 즉 per-pixel classification을 사용한다. 반면 instance segmentation이나 panoptic segmentation에서는 개별 영역 또는 마스크 단위로 예측하는 mask classification이 널리 쓰인다. 저자들은 이 둘이 본질적으로 다른 문제로 취급되어 서로 다른 모델 계열이 발전해 온 점을 문제로 본다.

논문의 핵심 연구 질문은 두 가지다. 첫째, mask classification만으로 semantic segmentation과 instance-level segmentation을 하나의 통일된 틀에서 다룰 수 있는가이다. 둘째, semantic segmentation에서도 mask classification이 기존 per-pixel classification보다 실제로 더 좋은 성능을 낼 수 있는가이다.

이 문제는 중요하다. segmentation 연구는 semantic, instance, panoptic으로 나뉘어 발전해 왔고, 각 태스크마다 아키텍처, loss, post-processing이 달라지는 경우가 많았다. 이런 분리는 모델 설계와 비교를 복잡하게 만든다. 저자들은 MaskFormer라는 단순한 mask classification 모델을 통해 동일한 모델, 동일한 loss, 동일한 training procedure로 semantic segmentation과 panoptic segmentation을 모두 처리할 수 있음을 보이려 한다. 특히 클래스 수가 많아질수록 per-pixel 방식보다 mask classification이 더 유리할 수 있다는 실험적 증거를 제시한다.

## 2. 핵심 아이디어

중심 아이디어는 segmentation을 “픽셀별 분류”가 아니라 “마스크 집합 예측(set prediction)”으로 보는 것이다. 즉 모델은 이미지 전체에 대해 $N$개의 binary mask를 예측하고, 각 mask마다 하나의 전역 class label을 붙인다. 이때 각 mask는 이미지의 어떤 영역을 담당하고, class prediction은 그 영역 전체의 의미를 결정한다.

이 관점의 장점은 partitioning과 classification을 분리한다는 데 있다. per-pixel classification은 각 위치에서 곧바로 클래스를 고르기 때문에, 같은 객체나 같은 semantic region을 하나의 일관된 단위로 모델링하기 어렵다. 반대로 mask classification은 먼저 “어떤 영역이 하나의 segment인가”를 마스크로 표현하고, 그 다음 그 영역 전체에 대해 클래스를 예측한다. 저자들은 이런 구조가 특히 클래스 수가 많을 때 더 강한 recognition 능력을 보인다고 해석한다. 논문에서도 “single class prediction per mask models fine-grained recognition better than per-pixel class predictions”라고 주장한다.

기존 접근과의 가장 중요한 차별점은 semantic segmentation에도 instance-style mask classification을 그대로 적용했다는 점이다. 기존의 DETR 계열 panoptic 방법은 box prediction이나 추가 auxiliary loss에 크게 의존했고, semantic segmentation은 여전히 per-pixel formulation이 주류였다. 반면 MaskFormer는 box head 없이 직접 mask를 예측하고, 기본적으로 각 prediction에 대해 classification loss와 binary mask loss만 사용한다. 즉 semantic과 panoptic을 정말 같은 방식으로 학습한다는 점이 차별적이다.

## 3. 상세 방법 설명

### 3.1 Per-pixel classification과 mask classification의 수식적 차이

per-pixel classification에서는 $H \times W$ 이미지의 각 픽셀마다 $K$개 클래스에 대한 확률분포를 예측한다. 논문은 이를 다음처럼 표현한다.

$$
y = \{p_i \mid p_i \in \Delta^K\}_{i=1}^{H \cdot W}
$$

여기서 $\Delta^K$는 $K$개 클래스에 대한 probability simplex이고, $p_i$는 $i$번째 픽셀의 클래스 분포이다. 학습은 모든 픽셀에 대해 cross-entropy를 적용한다.

$$
L_{\text{pixel-cls}}(y, y^{gt}) = \sum_{i=1}^{H \cdot W} - \log p_i(y_i^{gt})
$$

즉 각 픽셀의 정답 클래스 $y_i^{gt}$에 대한 negative log-likelihood를 모두 더하는 전형적인 방식이다.

반면 mask classification에서는 출력이 픽셀별 클래스가 아니라, $N$개의 예측 쌍 $\{(p_i, m_i)\}_{i=1}^N$이다. 여기서 $m_i \in [0,1]^{H \times W}$는 binary mask이고, $p_i \in \Delta^{K+1}$는 클래스 분포다. 중요한 점은 $K$개 실제 클래스 외에 추가로 “no object”인 $\emptyset$ 클래스가 들어간다는 것이다. 이 덕분에 예측 쿼리 수 $N$이 실제 정답 segment 수보다 많아도 남는 prediction을 $\emptyset$로 처리할 수 있다.

학습하려면 예측 집합과 정답 segment 집합 사이의 matching이 필요하다. 저자들은 DETR처럼 bipartite matching을 사용하되, box가 아니라 class와 mask를 이용해 matching cost를 계산한다. 예측 $z_i$와 정답 $z_j^{gt}$의 비용은 대략 다음과 같다.

$$
- p_i(c_j^{gt}) + L_{\text{mask}}(m_i, m_j^{gt})
$$

즉 정답 클래스 확률이 높을수록 좋고, mask가 정답과 잘 맞을수록 좋다.

최종 mask classification loss는 다음과 같다.

$$
L_{\text{mask-cls}}(z, z^{gt}) =
\sum_{j=1}^{N}
\left[
- \log p_{\sigma(j)}(c_j^{gt})
+
\mathbf{1}_{c_j^{gt} \neq \emptyset}
L_{\text{mask}}(m_{\sigma(j)}, m_j^{gt})
\right]
$$

여기서 $\sigma(j)$는 matching 결과이며, 각 정답 segment에 매칭된 예측 하나를 뜻한다. 쉽게 말하면 각 예측은 “이 마스크가 어떤 클래스인지”와 “이 마스크의 픽셀 형태가 맞는지”를 동시에 학습한다.

### 3.2 MaskFormer 아키텍처

MaskFormer는 크게 세 모듈로 구성된다.

첫째는 pixel-level module이다. backbone이 입력 이미지에서 저해상도 feature map $F \in \mathbb{R}^{C_F \times H/S \times W/S}$를 만든다. 그 다음 pixel decoder가 이를 점진적으로 upsampling하여 최종적으로 per-pixel embedding $E_{\text{pixel}} \in \mathbb{R}^{C_E \times H \times W}$를 생성한다. 논문에서는 이 pixel decoder로 가벼운 FPN 기반 구조를 사용한다. stride 32 feature부터 시작해서 2배씩 업샘플링하며, 같은 해상도의 backbone feature와 합친 뒤 $3 \times 3$ convolution, GroupNorm, ReLU로 융합한다. 최종 stride 4 feature에서 $1 \times 1$ convolution을 적용해 per-pixel embedding을 얻는다. 모든 feature map 채널 수는 256이다.

둘째는 transformer module이다. DETR의 decoder 구조를 그대로 사용한다. $N$개의 learnable query가 backbone feature를 attend하여 $Q \in \mathbb{R}^{C_Q \times N}$ 형태의 per-segment embedding을 만든다. 기본 설정은 6개의 decoder layer와 100개의 query이다. 이 query는 각기 하나의 예측 segment 후보를 담당한다.

셋째는 segmentation module이다. 각 query embedding에 대해 선형 분류기와 softmax를 적용해 클래스 분포 $p_i$를 예측한다. 동시에 2개의 hidden layer를 가진 MLP가 query embedding을 mask embedding $E_{\text{mask}} \in \mathbb{R}^{C_E \times N}$으로 바꾼다. 이 mask embedding과 per-pixel embedding의 dot product로 실제 마스크를 만든다. 논문에 제시된 식은 다음과 같다.

$$
m_i[h,w] = \text{sigmoid}\left(E_{\text{mask}}[:,i]^T \cdot E_{\text{pixel}}[:,h,w]\right)
$$

즉 각 픽셀 위치의 embedding과 $i$번째 query의 mask embedding이 얼마나 잘 맞는지를 점수로 계산하고, sigmoid로 0과 1 사이 확률로 바꿔 binary mask를 얻는다.

중요한 설계 선택은 mask들 사이에 softmax를 쓰지 않았다는 점이다. 저자들은 마스크들이 서로 배타적일 필요가 없다고 보고, 독립적인 sigmoid를 사용했다. 이는 여러 예측이 겹칠 수 있게 하며, 나중 inference에서 이를 조합한다.

### 3.3 손실 함수와 학습

MaskFormer는 각 prediction에 대해 classification loss와 mask loss를 사용한다. mask loss는 DETR과 유사하게 focal loss와 dice loss의 선형 결합이다.

$$
L_{\text{mask}}(m, m^{gt}) =
\lambda_{\text{focal}} L_{\text{focal}}(m, m^{gt}) +
\lambda_{\text{dice}} L_{\text{dice}}(m, m^{gt})
$$

논문에서 사용한 하이퍼파라미터는 $\lambda_{\text{focal}} = 20.0$, $\lambda_{\text{dice}} = 1.0$이다. 또한 classification loss에서 “no object” 클래스의 가중치는 0.1로 둔다.

학습 절차의 중요한 장점은 semantic segmentation과 panoptic segmentation에서 완전히 같다는 것이다. 바뀌는 것은 supervision의 형태뿐이다. semantic segmentation에서는 category region mask를 정답으로 주고, panoptic segmentation에서는 object instance mask를 정답으로 준다. 즉 같은 모델이 semantic-level annotation을 받으면 하나의 query가 같은 클래스의 여러 instance를 합친 semantic region을 예측하고, instance-level annotation을 받으면 여러 query가 각각 개별 instance를 담당하게 된다. appendix의 시각화도 이 차이를 보여 준다.

### 3.4 추론 방식

논문은 두 가지 inference를 제안한다.

일반적인 general inference는 각 픽셀을 가장 강한 probability-mask pair에 할당한다. 수식은 다음과 같다.

$$
\arg\max_{i : c_i \neq \emptyset} p_i(c_i) \cdot m_i[h,w]
$$

여기서 $c_i$는 $i$번째 prediction의 가장 가능성 높은 클래스다. 즉 어떤 픽셀을 특정 prediction에 할당하려면 그 prediction의 클래스 신뢰도도 높고, 해당 픽셀에서의 mask 값도 높아야 한다. semantic segmentation에서는 같은 클래스로 할당된 segment들을 합치고, instance-level task에서는 prediction index $i$를 유지해 서로 다른 instance를 구분한다.

semantic segmentation 전용 semantic inference는 더 부드러운 방식이다. 각 클래스 $c$에 대해 모든 query의 기여를 합산한 뒤 argmax를 취한다.

$$
\arg\max_{c \in \{1,\dots,K\}}
\sum_{i=1}^N p_i(c) \cdot m_i[h,w]
$$

즉 픽셀 하나를 단일 query에 강제로 할당하지 않고, 여러 mask prediction의 기여를 합쳐 클래스 점수를 만든다. 저자들은 semantic segmentation에서는 이 방식이 general inference보다 mIoU가 더 좋다고 보고한다. 다만 학습 단계에서 이런 per-pixel likelihood를 직접 최대화하면 성능이 나빠졌다고 하며, 그 이유로 gradient가 모든 query에 너무 고르게 퍼져 학습이 어려워진다고 추정한다.

## 4. 실험 및 결과

논문은 semantic segmentation과 panoptic segmentation 모두에서 실험한다. semantic segmentation 데이터셋으로는 Cityscapes(19 classes), Mapillary Vistas(65), ADE20K(150), COCO-Stuff-10K(171), ADE20K-Full(847)을 사용한다. panoptic segmentation은 COCO panoptic(133 categories)와 ADE20K panoptic(150 categories)을 사용한다. semantic은 표준 지표인 mIoU를 사용하고, 추가로 region-level 평가를 위해 $PQ^{St}$도 본다. panoptic은 PQ, $PQ^{Th}$, $PQ^{St}$를 사용한다.

ADE20K validation에서 MaskFormer는 semantic segmentation state of the art를 달성했다. Swin-L backbone과 ImageNet-22K pretraining을 사용할 때 multi-scale inference 기준 55.6 mIoU를 기록했고, 이는 당시 강한 baseline인 Swin-UperNet의 53.5 mIoU보다 2.1 높다. 같은 계열 backbone 대비 parameter와 FLOPs도 더 적다. 예를 들어 Swin-L 기준으로 MaskFormer는 212M parameters, 375G FLOPs이고, Swin-UperNet은 234M parameters, 647G FLOPs다. 즉 정확도뿐 아니라 효율 측면에서도 이득이 있었다.

CNN backbone에서도 개선이 나타난다. ADE20K에서 ResNet-101c 기준 MaskFormer는 48.1 mIoU(multi-scale)를 기록해 DeepLabV3+의 46.4보다 높다. 이는 단순히 backbone의 힘이 아니라 formulation 변화가 실제로 성능에 기여함을 보여 준다.

저자들이 특별히 강조하는 결과는 클래스 수가 많을수록 MaskFormer의 이점이 커진다는 점이다. Table 2에서 PerPixelBaseline+과 비교하면 Cityscapes(19 classes)에서는 mIoU 이득이 거의 없지만, ADE20K에서는 +2.6 mIoU, COCO-Stuff에서는 +2.9 mIoU, ADE20K-Full(847 classes)에서는 +3.5 mIoU까지 상승한다. $PQ^{St}$ 역시 각각 의미 있게 증가한다. 저자들은 이를 mask 단위의 전역 class prediction이 미세한 category recognition에 유리하기 때문이라고 해석한다.

Cityscapes에서는 standard mIoU 기준으로 최고 수준과 비슷하거나 약간 낮은 편이다. 예를 들어 R101c 기준 multi-scale 81.4 mIoU로 OCRNet 82.0보다 낮다. 하지만 $PQ^{St}$ 분석에서는 recognition quality인 $RQ^{St}$가 더 좋고, segmentation quality인 $SQ^{St}$는 약간 뒤처진다. 즉 클래스 구분은 잘하지만 마스크 경계 정밀도는 덜 유리할 수 있다는 뜻이다. 논문은 recognition이 쉬운 적은 클래스 데이터셋에서는 pixel-level mask quality가 더 중요한 병목이 된다고 해석한다.

Mapillary Vistas에서는 고해상도 이미지에도 잘 동작한다. R50 backbone만으로도 single-scale 53.1, multi-scale 55.4 mIoU를 기록해 DeepLabV3+와 HMSANet을 능가한다. 저자들은 Transformer decoder가 전역 문맥(global context)을 잘 포착하기 때문이라고 본다.

Panoptic segmentation에서도 결과가 강하다. COCO panoptic validation에서 Swin-L backbone의 MaskFormer는 52.7 PQ를 기록해 기존 Max-DeepLab의 51.1 PQ를 넘어선다. ResNet backbone으로 DETR와 비교해도 우세하다. 예를 들어 R50 + 6 encoder 기준 DETR는 43.4 PQ, MaskFormer는 46.5 PQ이다. 특히 $PQ^{St}$ 향상이 더 크다. 이는 stuff class를 bounding box로 표현하는 것이 부적절하다는 논문의 주장과 맞닿아 있다. box-based matching 대신 mask-based matching이 stuff 영역에 특히 유리하다는 것이다.

COCO test-dev에서도 Swin-L 기반 MaskFormer는 53.3 PQ로 Max-DeepLab 51.3을 넘어섰다. ADE20K panoptic validation에서도 R101 + 6 encoder 기준 35.7 PQ를 기록하며 경쟁력 있는 결과를 보인다.

ablation도 논문의 주장을 강하게 뒷받침한다. 먼저 formulation 비교에서 PerPixelBaseline+가 41.9 mIoU인데, 같은 fixed matching을 쓰는 MaskFormer-fixed는 43.7 mIoU를 얻는다. 즉 matching 방식이나 loss 차이보다도 per-pixel에서 mask classification으로 문제를 바꾼 것이 핵심이라는 뜻이다. bipartite matching을 쓰는 MaskFormer-bipartite는 44.2 mIoU와 33.4 $PQ^{St}$로 더 좋다.

query 수 실험에서는 100 queries가 가장 안정적으로 좋았다. 흥미롭게도 클래스 수가 150, 171, 847로 크게 달라도 최적 query 수는 비슷했다. 심지어 20 queries만 써도 per-pixel baseline보다 나은 경우가 있었다. 저자들은 하나의 query가 여러 클래스를 포착할 수 있다고 추정한다. 실제 appendix 그림에서도 query별로 담당하는 unique category 수가 균등하지 않음을 보인다.

decoder layer 수 실험에서는 semantic segmentation은 1-layer decoder만으로도 상당히 좋은 성능을 냈지만, panoptic segmentation은 여러 layer가 필요했다. 저자들은 많은 decoder layer가 중복 예측을 제거하는 데 도움을 준다고 해석한다. self-attention을 제거하면 semantic 성능 저하는 제한적이지만, panoptic에서는 성능 하락이 더 크다.

또한 DETR와의 비교에서 중요한 실험이 있다. 같은 box-based matching을 쓰도록 MaskFormer에 box head를 추가하면 DETR와 거의 비슷한 성능이 나온다. 반면 mask-based matching을 사용하면 PQ가 크게 오른다. 이는 성능 향상의 핵심이 “mask를 직접 맞추는 학습 목표”에 있음을 보여 준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 segmentation의 세부 태스크들을 하나의 간결한 formulation으로 통합했다는 점이다. semantic segmentation과 panoptic segmentation에 대해 동일한 모델, 동일한 loss, 동일한 training pipeline을 사용한다는 주장은 단순한 개념적 통합이 아니라, 실제 state-of-the-art 결과로 뒷받침된다. 특히 semantic segmentation에서 per-pixel classification이 사실상 표준이던 시점에, mask classification이 더 좋을 수 있음을 강하게 보여 준 점이 중요하다.

두 번째 강점은 설계가 비교적 단순하다는 점이다. DETR류 방법이 box prediction이나 복잡한 auxiliary loss에 기대는 것과 달리, MaskFormer는 box-free이며 classification loss와 mask loss만으로 end-to-end 학습된다. 또한 DETR의 per-query upsampling head보다 훨씬 계산량이 낮은 per-image FPN 기반 pixel decoder를 사용해 효율도 개선했다.

세 번째 강점은 large-vocabulary setting에서의 잠재력을 실험적으로 보여 준 점이다. ADE20K-Full 847 classes에서 per-pixel baseline보다 더 높은 성능과 더 낮은 training memory를 보였다는 결과는 실제 대규모 semantic ontology를 다루는 현실 문제에 유리할 가능성을 시사한다. 논문이 “number of masks”와 “number of classes”를 분리한 설계의 장점을 잘 보여 준다.

한계도 분명하다. 첫째, 클래스 수가 적고 recognition이 상대적으로 쉬운 데이터셋에서는 절대적 우위가 크지 않다. Cityscapes에서 mIoU 개선이 거의 없고, 저자들 스스로 recognition quality는 좋아도 mask quality는 다소 뒤처진다고 분석한다. 즉 MaskFormer의 강점은 “무엇인지 맞히는 것”에 더 가깝고, 매우 정교한 픽셀 경계 예측은 여전히 과제일 수 있다.

둘째, 왜 mask classification이 large-vocabulary에서 더 강한지에 대한 설명은 완전히 이론적으로 정립되어 있지 않다. 저자들은 “single class prediction per mask”가 fine-grained recognition에 더 적합하다고 가설을 제시하지만, 이것은 실험적 관찰에 기반한 해석이지 엄밀한 증명은 아니다.

셋째, query가 어떤 기준으로 여러 클래스를 묶어 표현하는지 명확한 구조적 해석은 제공되지 않는다. 논문은 query별 unique class 분포를 보여 주지만, 어떤 query가 어떤 의미론적 군집을 학습하는지 일관된 패턴은 관찰하지 못했다고 직접 밝힌다. 따라서 내부 표현 해석 가능성은 아직 낮다.

넷째, semantic segmentation에서 최적 inference는 task 자체보다 metric에 더 의존하는 면이 있다. 논문은 semantic inference가 mIoU에 유리하고, general inference가 $PQ^{St}$에 유리하다고 보인다. 이는 “완전한 통일”이라는 메시지에 약간의 조건을 단다. 모델과 학습은 통일되지만, 평가 형식에 따라 추론 방식은 달라질 수 있다.

## 6. 결론

이 논문은 semantic segmentation을 per-pixel classification으로만 볼 필요가 없으며, mask classification이 semantic segmentation과 panoptic segmentation을 아우르는 일반적인 segmentation paradigm이 될 수 있다고 주장한다. 제안된 MaskFormer는 backbone, pixel decoder, Transformer decoder, segmentation head로 이루어진 단순한 구조이지만, mask prediction과 global class prediction을 결합해 semantic과 instance-level supervision을 모두 같은 방식으로 처리한다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, semantic segmentation에도 mask classification을 본격적으로 적용해 per-pixel formulation의 대안으로 제시했다. 둘째, box-free이며 auxiliary loss 없이도 동작하는 간단한 통합 모델을 설계했다. 셋째, ADE20K와 COCO panoptic 등에서 state-of-the-art 혹은 그에 준하는 강력한 결과를 통해 이 접근의 실효성을 입증했다.

향후 관점에서 보면, 이 연구는 segmentation을 더 통합적으로 연구하게 만드는 출발점 역할을 한다. 특히 class vocabulary가 크거나 semantic과 instance 정보를 함께 다뤄야 하는 실제 응용에서 영향력이 크다. 이후 나온 여러 segmentation transformer 계열 연구들이 MaskFormer의 문제 재정의와 query-based mask prediction 관점을 확장했다는 점을 고려하면, 이 논문은 단순히 성능 향상만이 아니라 segmentation 문제를 보는 방식을 바꾼 중요한 논문으로 평가할 수 있다.
