# Deep Learning Markov Random Field for Semantic Segmentation

- **저자**: Ziwei Liu, Xiaoxiao Li, Ping Luo, Chen Change Loy, Xiaoou Tang
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1606.07230

## 1. 논문 개요

이 논문은 semantic segmentation을 위해 Markov Random Field (MRF)와 deep learning을 하나의 end-to-end 구조로 결합하는 방법을 제안한다. 저자들은 기존의 semantic segmentation이 pixel별 분류 성능을 높이기 위해 CNN을 활용하는 데는 성공했지만, 픽셀들 사이의 구조적 관계와 문맥 정보를 충분히 반영하는 pairwise modeling은 여전히 복잡하고 비효율적이었다고 본다. 특히 기존 MRF/CRF 기반 방법은 mean field (MF) inference를 여러 번 반복해야 하므로 학습과 추론 모두 느리고, CNN과 MRF를 함께 최적화하기도 어렵다.

논문의 핵심 문제는 다음과 같다. semantic segmentation에서는 각 픽셀을 독립적으로 분류하는 것만으로는 부족하고, 주변 픽셀과의 관계, object 간의 spatial context, 그리고 더 넓은 범위의 high-order interaction을 함께 고려해야 한다. 그러나 이런 관계를 정교하게 모델링할수록 inference가 비싸지고 end-to-end 학습이 어려워진다. 저자들은 이 문제를 해결하기 위해 Deep Parsing Network (DPN)를 제안하며, 이는 복잡한 MRF inference를 CNN의 feed-forward computation으로 근사한다.

이 문제가 중요한 이유는 semantic segmentation이 scene understanding, smart editing, automated driving 같은 실제 응용의 핵심 기술이기 때문이다. 픽셀 단위 예측이 조금만 불안정해도 object boundary가 흐려지거나, 문맥상 말이 안 되는 label 조합이 나올 수 있다. 따라서 단순한 분류 성능뿐 아니라 구조적 일관성과 문맥 이해를 함께 반영하는 모델이 필요하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 MRF의 unary term과 pairwise term을 전부 CNN 안으로 흡수해, 복잡한 graphical model inference를 사실상 네트워크의 한 번의 forward pass로 처리하는 것이다. 저자들은 VGG-16을 확장하여 unary term을 만들고, 그 뒤에 몇 개의 추가 레이어를 붙여 mean field update의 한 번의 iteration을 근사한다. 이로써 전통적인 iterative optimization 없이도 구조적 제약이 반영된 segmentation 결과를 낼 수 있게 한다.

기존 접근법과의 차별점은 세 가지로 요약된다. 첫째, 기존 deep structured model들은 mean field를 5회 이상, 많게는 10회 정도 반복해야 했지만, DPN은 한 번의 근사만으로도 높은 성능을 낸다. 둘째, 단순한 label co-occurrence만 보는 pairwise term이 아니라, mixture of local label contexts와 triple penalty를 통해 공간적 배치와 high-order relation을 함께 모델링한다. 예를 들어 ‘person’과 ‘table’이 같이 나올 수 있다는 것만이 아니라, 어느 상대 위치에서 같이 나오는지가 중요하다는 점을 반영한다. 셋째, pairwise term의 구조를 convolution receptive field로 바꾸어 표현하므로, 복잡한 graphical structure를 GPU 친화적으로 병렬화하기 쉽다.

논문이 말하는 중요한 직관은 “강한 unary classifier와 풍부한 pairwise context를 함께 학습하면, mean field를 여러 번 반복하지 않아도 충분히 좋은 근사를 만들 수 있다”는 것이다.

## 3. 상세 방법 설명

전체 모델은 energy-based MRF를 기반으로 한다. 논문은 라벨 변수 집합 $y$에 대한 에너지 함수를 다음과 같이 둔다.

$$
E(y)=\sum_{\forall i \in V}\Phi(y_i^u)+\sum_{\forall (i,j)\in E}\Psi(y_i^u,y_j^v)
$$

여기서 $\Phi(y_i^u)$는 voxel 또는 pixel $i$에 label $u$를 할당하는 unary cost이고, $\Psi(y_i^u,y_j^v)$는 두 위치 사이의 pairwise penalty이다. 논문은 image뿐 아니라 video까지 포괄하기 위해 3-D MRF 형식으로 설명하지만, 2-D image segmentation은 그 특수한 경우로 볼 수 있다.

Unary term은 다음과 같이 정의된다.

$$
\Phi(y_i^u)=-\ln p(y_i^u=1|I)
$$

즉 CNN이 예측한 픽셀별 class probability를 negative log probability 형태의 cost로 사용한다. 여기서 $p_i^u$는 위치 $i$에서 label $u$일 확률이다. 이 unary predictor는 VGG-16을 변형한 네트워크가 담당한다.

기존 pairwise term은 보통 다음과 같은 형태였다.

$$
\Psi(y_i^u,y_j^v)=\mu(u,v)d(i,j)
$$

여기서 $\mu(u,v)$는 label pair의 global compatibility이고, $d(i,j)$는 두 픽셀의 appearance와 position 차이를 반영하는 거리 함수이다. 하지만 저자들은 이 방식이 두 가지 한계가 있다고 지적한다. 하나는 spatial arrangement를 반영하지 못한다는 점이고, 다른 하나는 high-order interaction을 다루지 못한다는 점이다.

그래서 논문은 더 일반적인 pairwise term을 다음처럼 제안한다.

$$
\Psi(y_i^u,y_j^v)=\sum_{k=1}^{K}\lambda_k \mu_k(i,u,j,v)\sum_{\forall z \in N_j} d(j,z)p_z^v
$$

이 식은 두 부분으로 나뉜다. 첫 번째인 $\mu_k(i,u,j,v)$는 mixture of local label contexts를 나타낸다. 이것은 단순히 ‘person’과 ‘table’이 함께 나온다를 넘어서, 중심 위치 $i$에 있는 label $u$와 이웃 위치 $j$에 있는 label $v$가 어떤 상대 위치 관계에서 함께 등장하는지를 모델링한다. 또한 mixture 구조를 두어, 같은 label pair라도 여러 배치 패턴을 수용한다. 예를 들어 ‘person standing beside table’과 ‘person sitting behind table’ 같은 서로 다른 문맥을 다른 mixture component가 표현할 수 있다.

두 번째인 $\sum_{z \in N_j} d(j,z)p_z^v$는 triple penalty이다. 이것은 단순한 pair $(i,j)$만 보지 않고, $j$의 주변 이웃 $z$까지 끌어들여 $(i,j,z)$의 관계를 반영한다. 논문 설명대로라면, $(i,u)$와 $(j,v)$가 잘 맞는다면, $j$ 주변의 $(z,v)$와도 일관성이 있어야 한다는 뜻이다. 즉 local neighborhood 전체에 걸친 smoother한 구조를 강제한다.

추론은 mean field approximation으로 수행된다. 원래 MRF의 분포는

$$
P(y)=\frac{1}{Z}\exp\{-E(y)\}
$$

이고, 이를 fully-factorized distribution

$$
Q(y)=\prod_{\forall i \in V}\prod_{\forall u \in L} q_i^u
$$

로 근사한다. KL divergence를 최소화하면 mean field update 식이 유도되며, 일반적으로는

$$
q_i^u \propto \exp\left\{-\left(\Phi_i^u+\sum_{\forall j \in N_i}\sum_{\forall v \in L} q_j^v \Psi_{ij}^{uv}\right)\right\}
$$

가 된다. 논문이 제안한 pairwise term을 대입하면 update는 더 복잡해지지만, 저자들은 이를 두 단계 convolution으로 분해해 구현한다.

첫 번째 단계는 triple penalty를 계산하는 local convolution이다. 각 category별 probability map에 대해 큰 receptive field를 가진 3D local convolution을 적용해 주변 픽셀 정보로 smoothing한다. 논문 구현에서는 b12 레이어가 이 역할을 하며, 50×50×3 필터를 사용한다. 이 레이어는 위치마다 다른 filter를 쓰는 locally convolutional layer로, 공간 위치마다 다른 거리 패턴을 반영한다.

두 번째 단계는 local label context를 계산하는 global convolution이다. b13 레이어가 9×9×3×21 필터를 사용해 label context penalty를 계산하고, b14에서는 block min pooling으로 여러 mixture 중 penalty가 가장 작은 패턴을 선택한다. 이는 “가능한 여러 contextual pattern 중 현재 입력에 가장 잘 맞는 것”을 활성화하는 역할이다.

마지막으로 b15에서는 unary term과 pairwise penalty를 결합한다.

$$
o_{15}(i,u)=
\frac{\exp\{\ln(o_{11}(i,u))-o_{14}(i,u)\}}
{\sum_{u=1}^{21}\exp\{\ln(o_{11}(i,u))-o_{14}(i,u)\}}
$$

여기서 $o_{11}$은 unary prediction, $o_{14}$는 pairwise penalty에 해당한다. 직관적으로는 unary 확률을 기반으로 하되, 문맥적으로 맞지 않는 label에는 penalty를 더해 softmax로 다시 정규화하는 구조다.

아키텍처 측면에서 DPN은 VGG-16을 크게 두 방향으로 수정한다. 하나는 pooling 일부를 제거해 feature map resolution을 높이는 것이다. 다른 하나는 fully connected layer를 convolution layer로 바꿔 dense prediction이 가능하게 만드는 것이다. 그 위에 b12부터 b15까지의 추가 레이어를 붙여 MRF inference를 구현한다.

학습은 한 번에 전체를 jointly 학습하지 않고 incremental하게 진행한다. 먼저 unary term만 학습하고, 그 다음 triple penalty를 추가해 해당 파라미터만 학습하고, 이어서 label context 레이어를 추가해 학습한 뒤, 마지막에 전체를 joint fine-tuning한다. 논문은 이 방식이 처음부터 joint learning하는 것보다 더 안정적이라고 보고한다.

## 4. 실험 및 결과

실험은 PASCAL VOC 2012, Cityscapes, CamVid 세 데이터셋에서 수행되었다. VOC12와 Cityscapes는 이미지 segmentation, CamVid는 video segmentation 평가용이다. 주요 평가지표는 mean IoU (mIoU)이며, 추가로 tagging accuracy (TA), localization accuracy (LA), boundary accuracy (BA)도 제안해 분석한다.

VOC12에서 저자들은 먼저 ablation study를 통해 각 구성요소의 효과를 검증한다. triple penalty의 receptive field를 바꿔 본 결과, 50×50이 가장 좋았고 mIoU는 64.7%였다. 너무 작은 범위는 문맥을 충분히 포착하지 못하고, 너무 큰 범위는 오히려 과적합 또는 비효율을 초래하는 것으로 해석된다. label context는 1×1, 5×5, 9×9, 그리고 9×9 mixtures를 비교했으며, 9×9 mixtures가 가장 높은 66.5%를 기록했다. 이는 단순 global co-occurrence보다 local spatial context와 그 mixture가 실제로 중요하다는 증거다.

학습 단계를 따라가며 VOC12 validation에서 성능이 올라가는 것도 보여준다. unary term만 사용하면 평균 mIoU가 62.4%, 여기에 triple penalty를 더하면 64.7%, label context를 더하면 66.5%, 마지막 joint tuning 후에는 67.8%가 된다. 즉 각 구성요소가 누적적으로 성능을 올리고, 마지막 end-to-end fine-tuning이 추가 개선을 준다.

논문은 mean field iteration 수에 대한 비교도 제시한다. DPN은 한 번의 MF 근사만으로도 좋은 성능에 도달하는 반면, denseCRF류 방법은 5회 이상 반복해야 안정적으로 수렴한다. 저자들의 주장대로라면, 이는 inference cost를 크게 줄이는 핵심 장점이다.

VOC12 test set에서 DPN은 COCO pretraining 없이 74.1% mIoU, COCO pretraining을 사용한 DPN은 77.5% mIoU를 달성했다. 논문 표 기준으로 이는 당시 강력한 baseline들과 경쟁력 있는 성능이며, 일부 COCO pretraining 모델들과 비교해도 상당히 근접하거나 우수하다. 저자들은 특히 기존 RNN, DeepLab, Piecewise가 10회 수준의 MF iteration을 요구하는 데 비해, DPN이 단 한 번의 iteration만 사용한다는 점을 강조한다.

Cityscapes에서는 DPN이 66.8% mIoU를 기록했다. 논문 표 기준으로 Dilation10의 67.1% 다음으로 높은 수치다. 특히 road, building, vegetation, terrain, sky 같은 넓고 형태가 자유로운 클래스에서 강점을 보인다고 저자들은 해석한다. 이는 long-range 및 high-order pairwise term이 큰 구조를 더 잘 잡는다는 주장과 연결된다.

CamVid에서는 image-only DPN이 60.06% mIoU, spatial-temporal DPN이 60.25% mIoU를 달성했다. absolute gain은 크지 않지만, temporal relation을 넣었을 때 성능이 실제로 개선된다는 점을 보였다. 특히 tree, sky, road처럼 프레임 간 연속성이 강한 클래스에서 효과가 있었다고 설명한다. 또한 pole, sign 같은 가늘고 작은 객체에서도 기존 방법보다 좋은 결과를 보였다고 보고한다.

추가 분석으로 TA, LA, BA를 제시한 점도 흥미롭다. 논문에 따르면 label context를 추가한 단계에서 image-level tagging accuracy가 올라가고, triple penalty와 joint tuning은 localization 및 boundary accuracy를 개선한다. 즉 각 구성요소가 단순히 mIoU만 올리는 것이 아니라, 다른 종류의 segmentation 품질을 서로 다른 방식으로 향상시키는 것으로 해석할 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 structured prediction과 deep network를 매우 직접적으로 결합했다는 점이다. 기존에는 CNN이 unary score를 만들고, 그 위에 CRF/MRF를 별도 inference 모듈처럼 붙이는 경우가 많았는데, DPN은 mean field approximation 자체를 CNN 레이어로 재구성했다. 이 덕분에 end-to-end 학습, GPU 병렬화, 빠른 inference라는 세 가지를 동시에 노린다.

또 다른 강점은 pairwise modeling이 단순하지 않다는 점이다. 논문은 global label compatibility만 쓰는 대신, local label context와 triple penalty를 함께 써서 더 풍부한 문맥 정보를 encoding한다. 특히 “어떤 label이 같이 나오느냐”뿐 아니라 “어느 상대 위치에서 같이 나오느냐”를 mixture 형태로 다루는 것은 semantic segmentation에서 매우 자연스러운 확장이다.

실험도 비교적 충실하다. 단순 benchmark score만 제시하지 않고, receptive field, learning stage, MF iteration 수, temporal extension, per-class 분석, tagging/localization/boundary 분석까지 제공한다. 따라서 저자들의 주장이 어떤 요소에서 실제로 성립하는지 비교적 설득력 있게 보여준다.

한계도 분명하다. 첫째, 논문은 one-iteration MF approximation이 충분히 좋다고 주장하지만, 이것이 항상 모든 데이터셋과 장면 복잡도에서 성립하는지는 실험 범위 안에서만 확인된다. 즉 “왜 한 번이면 충분한가”에 대한 이론적 보장은 강하지 않고, 강한 unary와 잘 설계된 pairwise 덕분에 경험적으로 잘 작동했다는 쪽에 가깝다.

둘째, b12의 local convolution은 위치마다 다른 filter를 쓰는 구조라 계산량과 메모리 부담이 크다. 논문도 b12가 전체에서 가장 복잡한 레이어라고 인정하며, GPU 병렬화와 lookup table로 해결한다고 설명한다. 하지만 구조 자체는 상당히 무겁고, 이후 더 단순하고 효율적인 segmentation architecture들과 비교하면 확장성이 제한될 수 있다.

셋째, 일부 구현 선택은 데이터셋 특성에 꽤 의존적이다. 예를 들어 b12 filter를 RGB distance로 초기화하고, VOC12에서는 filter를 학습하지 않고 고정하는 식의 설계는 일반적인 end-to-end learnable model이라기보다 hand-crafted prior가 일부 섞인 형태다. 논문도 얼굴이나 사람처럼 shape가 더 regular한 데이터에서는 학습 가능성이 높다고 말하지만, 여기서는 일반 장면에 대해 fixed design을 사용했다.

넷째, failure case 분석에서 드러나듯 atypical pose, scale variation, illumination 변화에는 여전히 취약하다. object detector 같은 추가 unary potential이 도움이 될 수 있다고 저자들이 제안하지만, 이는 곧 현재 모델만으로는 해당 문제를 충분히 해결하지 못했다는 뜻이기도 하다.

비판적으로 보면, 이 논문은 “MRF를 CNN으로 푼다”는 개념적 기여가 크지만, 실제 모델은 VGG-16 기반 구조와 복잡한 custom layer에 많이 의존한다. 따라서 방법론의 아이디어 자체는 강하지만, 실용적 구현은 다소 무겁고 특정 시대의 아키텍처 설계 제약을 반영하고 있다.

## 6. 결론

이 논문은 semantic image/video segmentation을 위해 Deep Parsing Network (DPN)를 제안하고, MRF의 unary term과 풍부한 pairwise term을 CNN 내부에서 통합적으로 학습하고 추론하는 프레임워크를 제시한다. 핵심 기여는 VGG 기반 unary predictor 위에 triple penalty와 mixture of local label contexts를 추가해, mean field inference를 한 번의 feed-forward computation으로 근사했다는 점이다. 또한 기존 DeepLab, CRF-RNN류 모델의 pairwise formulation을 더 일반적인 형태로 포함할 수 있음을 보였다.

실험적으로는 VOC12, Cityscapes, CamVid에서 강한 성능을 보였고, 특히 반복적 MF inference 없이도 높은 segmentation 품질을 낸다는 점이 중요하다. 이 연구는 이후 semantic segmentation에서 structured context를 neural network 안으로 흡수하는 방향에 의미 있는 발판을 제공한다고 볼 수 있다. 실제 적용 측면에서는 자율주행, 장면 이해, 비디오 parsing처럼 문맥과 구조적 일관성이 중요한 문제에 가치가 있고, 향후 연구 측면에서는 더 가벼운 backbone, 더 효율적인 context modeling, 더 일반적인 spatio-temporal structured prediction으로 확장될 여지가 크다.
