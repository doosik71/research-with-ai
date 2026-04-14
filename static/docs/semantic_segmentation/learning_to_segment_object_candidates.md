# Learning to Segment Object Candidates

- **저자**: Pedro O. Pinheiro, Ronan Collobert, Piotr Dollár
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1506.06204

## 1. 논문 개요

이 논문은 object detection의 전처리 단계에서 사용되는 object proposal을 더 정확하고 더 적은 수의 후보로 생성하는 방법을 제안한다. 기존의 많은 proposal 방법은 bounding box를 만들거나, segmentation mask를 만들더라도 edges, superpixels, low-level segmentation 같은 수작업 기반 단서를 강하게 사용했다. 반면 이 논문은 이미지 patch를 입력받아, 그 patch 중앙에 적절한 크기의 완전한 객체가 들어 있는지를 판단하는 score와 그 객체의 class-agnostic segmentation mask를 동시에 예측하는 convolutional network를 학습한다.

연구 문제는 명확하다. 좋은 object proposal 방법은 높은 recall을 가져야 하고, 그 recall을 가능한 적은 수의 proposal로 달성해야 하며, proposal의 위치와 형태가 실제 객체에 최대한 잘 맞아야 한다. 이 문제는 detection 시스템 전체 성능과 속도에 직접 연결된다. proposal 품질이 높으면 이후 classifier나 detector는 더 적은 후보만 보고도 더 높은 정확도를 낼 수 있다. 저자들은 이 문제를 “low-level heuristic”이 아니라 대규모 segmented data와 ConvNet feature learning으로 직접 풀 수 있음을 보이고자 한다.

특히 이 논문은 MS COCO에서 학습한 모델이 PASCAL VOC와 COCO에서 강한 성능을 보이고, 심지어 학습 시 보지 못한 category에도 어느 정도 일반화된다는 점을 강조한다. 이는 proposal 생성이 특정 class recognition이 아니라 보다 일반적인 objectness와 foreground shape 추정 문제라는 관점을 뒷받침한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 하나의 공유 ConvNet backbone 위에 두 개의 branch를 얹어, 하나는 mask를, 다른 하나는 objectness score를 예측하도록 joint learning하는 것이다. 입력은 `3 × 224 × 224` patch이고, 출력은 “중앙 객체의 mask”와 “이 patch가 centered full object를 담고 있을 가능성”이다. 여기서 중요한 점은 category label을 예측하지 않는다는 것이다. 즉, 이 모델은 “무엇인지”보다 “객체인지, 그리고 어디까지가 객체인지”를 학습한다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, segmentation proposal을 생성하면서도 edges나 superpixels에 의존하지 않는다. 둘째, box proposal이 아니라 mask proposal을 직접 생성하므로 더 풍부한 형태 정보를 제공한다. 셋째, proposal 생성과 ranking을 별도의 단계가 아니라 하나의 네트워크 안에서 동시에 수행한다. 논문은 이를 통해 더 적은 proposal 수로 더 높은 average recall을 달성했다고 보고한다.

또 하나의 중요한 설계 직관은 segmentation branch가 전체 feature map을 보아야 한다는 점이다. semantic segmentation처럼 각 위치가 주변 지역만 보고 class를 예측하는 것이 아니라, 이 논문에서는 여러 객체가 patch 안에 있어도 오직 중앙의 하나의 객체만 분리해내야 한다. 그래서 각 output pixel classifier가 patch 전체 문맥을 활용할 수 있게 설계한다.

## 3. 상세 방법 설명

전체 시스템인 DeepMask는 shared feature extractor와 두 개의 task-specific branch로 구성된다. 공유 부분은 ImageNet classification으로 pretrain된 VGG-A를 기반으로 한다. 저자들은 원래 VGG-A의 fully connected layer들을 제거하고, 마지막 max-pooling layer도 제거한다. 그 결과 입력 이미지가 `3 × h × w`일 때 공유 feature map은 `512 × h/16 × w/16` 크기가 된다. 이렇게 한 이유는 segmentation에 필요한 spatial information을 최대한 보존하기 위해서다.

학습 데이터의 각 sample `k`는 `(x_k, m_k, y_k)`라는 triplet로 정의된다. 여기서 `x_k`는 RGB patch, `m_k`는 binary mask, `y_k ∈ {±1}`은 이 patch가 positive인지 negative인지를 나타낸다. positive sample은 두 조건을 만족해야 한다. 첫째, patch 안에 객체가 대략 중앙에 위치해야 한다. 둘째, 그 객체가 patch 내부에 완전히 포함되어 있고 적절한 scale range 안에 있어야 한다. 이때 `m_k`는 중앙의 단일 객체에 대해서만 positive 값을 가지며, negative patch에서는 mask를 사용하지 않는다. 즉, negative patch에 대해 “배경 mask”를 강제로 학습시키지 않는다.

Segmentation branch는 먼저 `1 × 1` convolution layer와 ReLU를 적용한 뒤, 최종적으로 각 pixel이 중앙 객체에 속하는지 판별하는 출력을 만든다. 하지만 각 output pixel classifier가 전체 feature map을 봐야 하므로, 단순한 local classifier는 receptive field가 부족하고, 완전한 fully connected classifier는 파라미터가 너무 많다. 이를 해결하기 위해 저자들은 classification layer를 두 개의 linear layer로 분해한다. 중간에 non-linearity가 없는 이 구조는 일종의 low-rank fully connected classifier로 볼 수 있다. 이 설계는 전체 문맥을 활용하면서도 파라미터를 크게 줄인다. 실제 출력은 `56 × 56`으로 만든 뒤 bilinear upsampling을 통해 `224 × 224`로 복원한다.

Scoring branch는 patch가 centered object 조건을 만족하는지를 예측한다. 구조는 `2 × 2` max-pooling 뒤에 두 개의 fully connected layer와 ReLU, dropout을 두고, 마지막 linear layer가 하나의 objectness score를 출력한다. 이 score는 patch 중심에 객체가 존재하고, scale도 적절하다는 것을 함께 반영한다.

논문의 joint loss는 segmentation과 score를 동시에 학습하도록 설계된다. 논문에서 제시한 식은 다음과 같다.

$$
L(\theta)=\sum_k \left( \frac{1+y_k}{2w_oh_o}\sum_{ij}\log\left(1+e^{-m_{ij}^k f^{ij}_{segm}(x_k)}\right) + \lambda \log\left(1+e^{-y_k f_{score}(x_k)}\right) \right)
$$

여기서 `f^{ij}_{segm}(x_k)`는 위치 `(i, j)`의 mask prediction이고, `f_{score}(x_k)`는 objectness score다. 첫 번째 항은 pixel-wise binary logistic regression이고, 두 번째 항은 patch-level binary logistic regression이다. `\frac{1+y_k}{2}` 항 때문에 `y_k = 1`인 positive sample에서만 segmentation loss가 활성화된다. 즉, negative patch에는 segmentation branch의 gradient를 흘리지 않는다.

저자들은 negative까지 segmentation branch에 넣는 대안, 즉 negative sample에 대해 모든 픽셀을 0으로 두고 학습하는 방식도 고려했지만, positive sample만으로 segmentation branch를 학습하는 것이 unseen category에 대한 generalization과 높은 object recall에 중요했다고 말한다. 이 부분은 이 논문의 핵심 중 하나다. segmentation branch는 “특정 class의 모양”보다 “중앙에 있는 하나의 객체 영역을 분리하는 일반적 능력”을 학습하도록 유도된다.

Full scene inference에서는 모델을 이미지 전체에 대해 여러 위치와 여러 scale에서 dense하게 적용한다. 이렇게 해야 학습 시 가정했던 “객체가 patch 안에 완전히 포함되고 대략 중앙에 있음” 조건을 실제 테스트 이미지의 각 객체에 대해 적어도 한 번은 만족시킬 수 있다. 이 방식은 sliding-window처럼 보이지만, 실제 계산은 convolutional하게 구현되어 효율적이다. VGG feature는 전체 이미지에 대해 한 번에 구하고, segmentation과 score branch도 convolution으로 계산한다.

한 가지 구현상 문제는 segmentation branch와 scoring branch의 출력 해상도가 다르다는 점이다. segmentation branch는 downsampling factor가 16이고, scoring branch는 추가 max-pooling 때문에 32다. 따라서 두 출력을 일대일로 대응시키기 위해, 저자들은 scoring branch에 interleaving trick을 적용해 출력 해상도를 두 배로 높인다.

구현 세부사항도 비교적 구체적으로 제시된다. positive canonical example은 객체가 정확히 중앙에 있고 최대 dimension이 128 pixel인 경우로 정의한다. 하지만 inference 시 객체는 완벽히 정렬되지 않으므로, 학습에서 translation `±16` pixel, scale deformation `2^{±1/4}`, horizontal flip을 적용해 jittering한다. negative example은 canonical positive에서 위치가 `±32` pixel 이상 벗어나거나 scale이 `2^{±1}` 이상 차이 나는 patch로 정의한다.

테스트 시에는 stride 16 pixel로 dense하게 위치를 옮기고, scale은 `2^{-2}`부터 `2^1`까지 `2^{1/2}` step으로 적용한다. scoring branch는 hidden unit 512와 1024를 갖는 두 fully connected layer를 사용하며 dropout rate는 0.5다. segmentation branch는 `1 × 1` conv 512 unit 뒤에 low-rank output layer를 거쳐 `56 × 56` mask를 출력한다. 전체 모델 파라미터 수는 약 75M이다. 학습은 SGD, batch size 32, momentum 0.9, weight decay 0.00005, learning rate 0.001로 진행되며, Nvidia Tesla K40m에서 약 5일 걸렸다고 한다. mask binarization threshold는 PASCAL에서 0.1, COCO에서 0.2를 사용했다.

## 4. 실험 및 결과

실험은 MS COCO 2014 validation set의 첫 5000장과 PASCAL VOC 2007 test set에서 수행된다. 학습은 COCO train set 약 8만 장, 총 50만 개에 가까운 segmented object를 사용한다. 평가는 segmentation proposal뿐 아니라, mask를 감싸는 bounding box를 취해 box proposal로도 수행한다.

평가 지표는 IoU와 AR(Average Recall)이다. IoU는 proposal과 ground-truth의 intersection-over-union이고, AR은 IoU 0.5에서 1.0 사이 구간에 대한 평균 recall이다. 저자들은 Hosang 등의 분석을 인용하여 AR이 실제 detector 성능과 강하게 상관된다고 설명한다. 비교 대상은 EdgeBoxes, SelectiveSearch, Geodesic, Rigor, MCG로, 당시 공개된 대표적인 proposal 방법들이다.

결과는 매우 강하다. COCO의 box proposal 기준으로 DeepMask는 `AR@10 = .153`, `AR@100 = .313`, `AR@1000 = .446`, `AUC = .233`을 기록한다. 이는 MCG의 `.101`, `.246`, `.398`, `.180`보다 전반적으로 높다. segmentation proposal 기준에서도 DeepMask는 `AR@10 = .126`, `AR@100 = .245`, `AR@1000 = .331`, `AUC = .023`으로, MCG의 `.077`, `.186`, `.299`, `.031`과 비교해 proposal 수가 적을 때 특히 우수하다. 논문은 “100개의 segmentation proposal만으로 DeepMask가 COCO에서 `AR = .245`를 달성하며, 다른 경쟁 방법들은 비슷한 성능을 내려면 거의 1000개 proposal이 필요하다”고 강조한다. 즉, 동일한 recall을 훨씬 적은 후보로 달성한다.

PASCAL VOC 2007 box proposal에서도 DeepMask는 `AR@10 = .337`, `AR@100 = .561`, `AR@1000 = .690`, `AUC = .433`으로, MCG의 `.232`, `.462`, `.634`, `.344`보다 뚜렷하게 높다. 이 결과는 proposal 수가 매우 적을 때 특히 차이가 커서, 후속 detector의 계산량 절감과 직접 연결된다.

객체 크기별 분석도 제시된다. COCO 객체를 small, medium, large로 나누어 보면 모든 방법이 small object에서 고전한다. DeepMask도 small object에서는 약하지만, 추가로 작은 scale에서 한 번 더 적용하는 DeepMaskZoom을 사용하면 특히 small object 성능이 올라간다. 다만 이 경우 inference time이 증가한다. 즉, 기본 모델은 작은 객체보다 중간 또는 큰 객체에 더 적합하고, multiscale 확대 적용이 보완책으로 제시된다.

Localization 분석에서는 recall을 다양한 IoU threshold에서 비교한다. DeepMask는 거의 모든 구간에서 높은 recall을 보이지만, 매우 높은 IoU에서는 일부 방법보다 약간 낮다. 저자들은 그 이유를 출력 mask가 downsampled resolution에서 예측된 뒤 upsampling되기 때문으로 해석한다. 더 높은 정밀 localization을 위해 multiscale approach나 skip connection이 도움이 될 수 있다고 언급하지만, 실제로 이 논문에서 그것을 구현하지는 않았다.

Generalization 실험은 이 논문의 중요한 메시지다. DeepMask20은 COCO의 80개 category 중 PASCAL의 20개 category에 속한 객체만 사용해 학습한 모델이다. 이를 80개 전체 category에 대해 평가했을 때 AR이 다소 떨어지지만, 여전히 기존 방법들을 이긴다. 더 흥미로운 것은 DeepMask20*인데, 이것은 segmentation branch는 20 category로 학습하되 scoring branch는 원래 DeepMask의 것을 사용한다. 이 경우 성능이 원본 DeepMask와 거의 같아진다. 저자들은 이를 통해 성능 저하의 원인이 segmentation branch가 아니라 scoring branch의 discriminative training 때문이라고 해석한다. 즉, segmentation branch는 본 적 없는 category에도 꽤 잘 일반화한다.

Architecture ablation도 수행된다. 저자들은 low-rank segmentation head 대신 각 `56 × 56` output pixel이 convolutional feature에 직접 fully connected되는 full-rank 구조(300M+ parameter)를 실험했는데, 이것이 오히려 최종 모델보다 약간 성능이 낮고 훨씬 느렸다. 이는 단순히 파라미터를 늘린다고 좋아지지 않으며, low-rank decomposition이 효율성과 일반화 양쪽에 유리함을 시사한다.

Detection 관점의 최종 검증도 있다. Fast R-CNN을 SelectiveSearch proposal과 DeepMask proposal 각각으로 돌려 비교했을 때, DeepMask proposal 100개만으로 `68.2%` mAP를 달성해, SelectiveSearch proposal 2000개를 쓴 경우의 `66.9%` mAP보다 높다. 500개 DeepMask proposal에서는 `69.9%` mAP까지 오른다. 저자들은 이를 통해 proposal 품질 향상이 실제 detector 성능 향상으로 이어진다는 점을 입증한다.

속도 측면에서는 COCO 이미지당 평균 1.6초, 더 작은 PASCAL 이미지에서는 1.2초가 걸린다고 보고한다. Geodesic이 약 1초로 조금 더 빠를 수 있으나, MCG의 약 30초와 비교하면 훨씬 빠르다. 또한 여러 scale을 하나의 batch로 병렬화하면 약 30% 추가 단축 가능하다고 한다. 다만 GPU가 필요하다는 점은 분명한 제약이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 segmentation proposal을 raw pixel에서 직접 학습하는 최초의 데이터 기반 딥러닝 접근 중 하나라는 점이다. 기존 segmentation proposal 방법들이 superpixel이나 edge에 크게 의존하던 시점에서, DeepMask는 그런 handcrafted 중간 표현 없이도 더 높은 recall을 달성했다. 또한 적은 proposal 수로 높은 AR을 내기 때문에 downstream detector의 효율을 실질적으로 개선한다. Fast R-CNN과 결합했을 때 20배 적은 proposal로도 더 높은 mAP를 얻은 결과는 그 실용적 가치를 잘 보여준다.

두 번째 강점은 joint design의 균형이다. segmentation과 scoring을 공유 backbone 위에서 함께 학습하여 계산을 줄이면서도, 출력은 mask와 ranking score를 동시에 제공한다. 특히 positive sample에 대해서만 segmentation loss를 주는 설계는 unseen category generalization에 효과적이었다. 이는 논문이 단순히 성능만 높은 것이 아니라, 왜 그런 설계가 필요한지에 대한 실험적 근거도 제시했다는 뜻이다.

세 번째 강점은 low-rank segmentation head의 설계다. 각 pixel classifier가 전체 문맥을 보아야 한다는 문제를 정면으로 다루면서도, full-rank fully connected보다 더 적은 파라미터와 더 나은 성능을 보였다. 이는 구조적 효율성 측면에서도 설계가 잘 되어 있음을 보여준다.

한편 한계도 분명하다. 첫째, very high IoU localization에서 약점이 있다. 저자들 스스로도 downsampled mask 예측과 upsampling 때문에 극도로 정밀한 경계는 덜 정확할 수 있다고 인정한다. 둘째, small object 성능이 낮다. DeepMaskZoom으로 개선할 수 있지만, 이는 추가 연산 비용을 요구한다. 셋째, inference가 convolutional하게 효율적이긴 하지만 여전히 GPU 의존적이며, 완전한 real-time 수준이라고 보기는 어렵다. 넷째, scoring branch는 학습된 category 분포의 영향을 받기 쉬워 unseen category에 낮은 score를 줄 수 있다. generalization 실험은 이 문제가 segmentation이 아니라 scoring에서 비롯된다는 점을 보여주지만, 동시에 시스템 전체 ranking 품질은 여전히 scoring branch에 민감하다는 뜻이기도 하다.

비판적으로 보면, 이 논문은 proposal generation이라는 문제에 매우 효과적인 해법을 제시했지만, object shape 품질을 더 높이기 위한 finer-resolution design, multi-scale feature fusion, boundary refinement 같은 방향은 아직 열려 있다. 또한 논문은 proposal 자체의 category-free 일반화는 잘 보여주지만, 이 proposal이 복잡한 downstream detector와 end-to-end로 결합될 때 어떤 추가 이점을 주는지는 제한적으로만 보여준다. 물론 저자들도 결론에서 detection과 더 긴밀하게 결합하는 것을 future work로 제시한다.

## 6. 결론

이 논문은 object proposal 생성, 특히 segmentation proposal 생성 문제를 deep learning으로 직접 푼 중요한 초기 작업이다. DeepMask는 하나의 ConvNet으로 class-agnostic mask와 objectness score를 공동 예측하고, 이를 이미지 전체에 dense하게 적용해 ranked segmentation proposals를 생성한다. 그 결과 COCO와 PASCAL에서 기존 state of the art를 큰 폭으로 능가했고, 적은 proposal 수만으로도 높은 recall을 달성했다.

실제 적용 관점에서도 의미가 크다. 좋은 proposal은 detector 전체 계산량을 줄이고 정확도를 높이는 핵심 요소인데, DeepMask는 Fast R-CNN과 결합했을 때 그 효과를 분명히 보여줬다. 향후 연구 관점에서는 region proposal network, mask-based detection, instance segmentation 계열 연구로 이어지는 중요한 연결고리로 볼 수 있다. 특히 “객체 후보를 handcrafted grouping이 아니라 learned dense prediction으로 생성할 수 있다”는 메시지는 이후 컴퓨터 비전 연구 흐름에 중요한 영향을 준 것으로 해석할 수 있다.
