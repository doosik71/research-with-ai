# Semantic Image Segmentation via Deep Parsing Network

- **저자**: Ziwei Liu, Xiaoxiao Li, Ping Luo, Chen Change Loy, Xiaoou Tang
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1509.02634

## 1. 논문 개요

이 논문은 semantic image segmentation에서 CNN의 강한 픽셀 분류 능력과 MRF/CRF의 구조적 제약을 하나의 네트워크 안에서 결합하려는 문제를 다룬다. 저자들은 기존 방법들이 두 방향으로 갈라져 있었다고 본다. 하나는 MRF/CRF의 pairwise term을 정교하게 설계하여 문맥, 장거리 상호작용, 고차 관계를 넣는 방식이지만 unary term이 약하고 추론 비용이 크다. 다른 하나는 CNN으로 unary prediction을 강하게 만들지만 pairwise modeling은 단순하거나, CNN과 MRF를 결합하더라도 mean field inference를 여러 번 반복해야 해서 학습과 추론이 무겁다.

이 논문의 핵심 목표는 이런 두 흐름의 장점을 동시에 취하는 것이다. 즉, 강한 CNN 기반 unary term과 풍부한 pairwise term을 함께 사용하되, 반복적인 mean field inference를 여러 번 수행하지 않고 단 한 번의 forward pass 안에서 근사적으로 해결하는 구조를 만드는 것이다. 이를 위해 저자들은 Deep Parsing Network, 즉 DPN을 제안한다.

이 문제가 중요한 이유는 semantic segmentation이 단순히 각 픽셀을 독립적으로 분류하는 문제가 아니기 때문이다. 실제 이미지에서는 객체 간 상대 위치, 함께 등장하는 패턴, 경계의 정교함, 주변 픽셀과의 구조적 일관성이 성능에 매우 큰 영향을 준다. 따라서 좋은 unary classifier만으로는 한계가 있고, 구조적 제약까지 효율적으로 결합하는 방법이 필요하다. DPN은 바로 이 지점을 겨냥해, 복잡한 pairwise structure를 CNN 연산으로 흡수하고 GPU 친화적으로 구현하려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 MRF의 mean field 업데이트 식을 convolution과 pooling으로 분해하여 CNN 내부 연산으로 구현하는 데 있다. 기존 CNN+CRF 계열 방법들은 보통 mean field를 여러 번 반복해야 했고, 그 과정을 backpropagation 안에 넣거나 recurrent form으로 풀었다. 반면 DPN은 unary term을 충분히 강하게 학습하고 pairwise term도 richer하게 설계한 뒤, mean field의 한 번 업데이트만으로도 높은 성능을 내도록 만든다.

이때 pairwise term의 설계가 중요하다. 저자들은 기존의 단순한 pairwise term이 대체로 label co-occurrence와 pixel similarity 정도만 반영한다고 비판한다. 예를 들어 “person”과 “table”이 함께 나올 수 있다는 사실은 배울 수 있어도, “person은 table의 아래쪽보다는 옆이나 위쪽에 나타난다” 같은 local spatial context는 충분히 반영하지 못한다. 또한 단순 pairwise relation만으로는 주변 픽셀 집합과의 higher-order interaction도 포착하기 어렵다.

그래서 DPN은 pairwise term 안에 두 가지를 넣는다. 첫째는 **mixture of local label contexts**이다. 이는 객체 라벨 간 호환성뿐 아니라 상대적 위치 패턴까지 학습한다. 둘째는 **high-order relation**, 논문 표현으로는 사실상 triple penalty이다. 어떤 픽셀 $i$와 이웃 픽셀 $j$의 관계만 보는 것이 아니라, $j$ 주변의 다른 픽셀들 $z$까지 포함해 세 점 관계를 반영한다. 이 설계 덕분에 DPN은 단순한 smoothness를 넘어서 보다 구조적인 segmentation prior를 갖게 된다.

또 하나의 중요한 주장으로, 저자들은 기존의 여러 deep structured model이 DPN의 특수한 경우라고 설명한다. 특정 설정, 예를 들어 mixture 수를 1로 두고 spatially varying context를 제거하면 DeepLab이나 CRF-RNN류에서 쓰는 pairwise form으로 환원된다는 것이다. 즉 DPN은 새로운 모델인 동시에 더 일반적인 표현 틀로 제시된다.

## 3. 상세 방법 설명

논문은 semantic segmentation을 MRF 에너지 최소화 문제로 둔다. 각 픽셀 $i$와 라벨 $u$에 대해 이진 latent variable $y_i^u \in \{0,1\}$를 두고, 에너지는 다음과 같다.

$$
E(y)=\sum_{\forall i \in V}\Phi(y_i^u)+\sum_{\forall i,j \in E}\Psi(y_i^u,y_j^v)
$$

여기서 $\Phi$는 unary term이고, $\Psi$는 pairwise term이다. unary term은 CNN이 예측한 픽셀별 라벨 확률로부터 다음처럼 정의된다.

$$
\Phi(y_i^u)=-\ln p(y_i^u=1 \mid I)
$$

즉 어떤 픽셀이 라벨 $u$일 확률이 높을수록 해당 unary cost는 낮다.

기존의 단순한 pairwise term은 보통

$$
\Psi(y_i^u,y_j^v)=\mu(u,v)d(i,j)
$$

처럼 쓴다. 여기서 $\mu(u,v)$는 라벨 $u, v$의 전역적 공출현 호환성을 나타내고, $d(i,j)$는 픽셀 외관이나 위치의 거리 함수다. 논문은 이것만으로는 spatial context와 higher-order relation을 충분히 표현하지 못한다고 본다.

그래서 DPN은 pairwise term을 다음과 같이 확장한다.

$$
\Psi(y_i^u,y_j^v)=\sum_{k=1}^{K}\lambda_k \mu_k(i,u,j,v)\sum_{\forall z \in N_j} d(j,z)p_z^v
$$

이 식은 해석적으로 두 부분으로 나뉜다.

첫째, $\mu_k(i,u,j,v)$는 **local label context**를 학습한다. 이것은 단순히 라벨 $u$와 $v$가 함께 등장하는지만 보지 않고, 픽셀 $i$와 $j$의 상대 위치까지 고려하여 “어떤 위치에서 어떤 라벨 조합이 얼마나 자연스러운가”를 나타낸다. 예를 들어 “person” 아래쪽에 “table”이 오는 패턴은 벌점을 크게 줄 수 있다.

둘째, $\sum_{z \in N_j} d(j,z)p_z^v$는 **triple penalty** 역할을 한다. 픽셀 $j$의 라벨만 보는 것이 아니라, $j$ 주변의 이웃 $z$들의 확률까지 합쳐서 “$j$ 근처에서도 같은 라벨 구조가 일관적인가”를 반영한다. 결국 이는 한 점과 한 점의 관계가 아니라 국소 neighborhood 전체와의 관계를 끌어들이는 셈이다.

MRF inference는 mean field approximation으로 푼다. fully factorized distribution $Q(y)=\prod_i \prod_u q_i^u$를 두고 free energy를 최소화하면 업데이트 식은

$$
q_i^u \propto \exp \left\{-\left(\Phi_i^u + \sum_{\forall j \in N_i}\sum_{\forall v \in L} q_j^v \Psi_{ij}^{uv}\right)\right\}
$$

형태가 된다. 여기에 위의 확장된 pairwise term을 대입하면 논문의 식 (7)이 된다. DPN의 핵심은 이 mean field 한 번 업데이트를 CNN layer들로 구현하는 것이다.

구체적으로 네트워크는 VGG-16을 기반으로 확장된다. 논문은 DPN을 15개 그룹으로 설명한다.

먼저 **b1-b11**은 unary term을 모델링한다. 기본 뼈대는 VGG-16이지만, segmentation을 위해 해상도를 올리도록 수정한다. 원래 VGG-16의 max pooling 일부를 제거하여 feature map 해상도 손실을 줄이고, fully connected layer를 convolution으로 바꾼다. 마지막 b11은 $512 \times 512 \times 21$ 크기의 unary label map을 출력한다. 각 채널은 하나의 클래스에 대한 픽셀별 확률 맵이다.

그 다음 **b12-b15**가 pairwise smoothing을 구현한다.

**b12**는 locally convolutional layer이다. 각 spatial position마다 다른 filter를 쓰되, 21개 클래스 채널에 대해 공유한다. 논문 식은 다음과 같다.

$$
o_{12}(j,v)=\text{lin}(k_{(j,v)} * o_{11}(j,v))
$$

여기서 $k_{(j,v)}$는 $50 \times 50 \times 1$ 필터이고, 이는 position $j$ 주변의 pixel distance를 반영한다. 이 층은 본질적으로 triple penalty를 구현한다. 즉 특정 클래스 $v$의 확률 맵을 그 주변 이웃과의 거리 기반으로 locally smoothing한다. 논문에서는 VOC12처럼 형태가 매우 다양한 데이터셋에서는 이 필터들을 Euclidean distance 기반으로 초기화하고, 실제 학습에서는 distance measure의 파라미터를 업데이트한다고 설명한다.

**b13**은 standard convolution layer이며, $9 \times 9 \times 21$ 필터 105개를 사용한다. 이는 논문에서 $K=5$인 mixture of label contexts를 구현한다. 각 클래스에 대해 5개의 contextual pattern을 두는 구조라서 총 출력 채널 수가 $21 \times 5 = 105$가 된다. 이 층은 “어떤 클래스 조합이 어떤 상대적 위치에서 자연스러운가”를 지역적으로 학습한다.

**b14**는 block min pooling이다. 105개 채널을 클래스당 5개씩 묶어 최소값을 취해서 다시 21채널로 줄인다. 의미적으로는 5개의 mixture component 중에서 penalty가 가장 작은, 즉 가장 잘 맞는 contextual pattern을 선택하는 역할이다.

**b15**는 unary output과 pairwise penalty를 결합하여 최종 확률을 만든다. 논문 식은 다음과 같다.

$$
o_{15}(i,u)=
\frac{\exp\{\ln(o_{11}(i,u))-o_{14}(i,u)\}}
{\sum_{u=1}^{21}\exp\{\ln(o_{11}(i,u))-o_{14}(i,u)\}}
$$

이 식은 unary probability에 pairwise penalty를 반영하여 다시 softmax normalization하는 형태다. 직관적으로는 unary가 기본 점수를 주고, pairwise context가 그 점수를 깎거나 보정한다.

학습은 한 번에 전부 하지 않고 **incremental learning**으로 진행한다. 첫 단계에서는 b1-b11만 학습하여 unary term을 먼저 안정화한다. 둘째 단계에서는 b12를 얹고 triple relation 관련 파라미터를 학습한다. 셋째 단계에서는 b13-b14를 추가하여 local label context를 학습한다. 마지막으로 전체를 joint fine-tuning한다. 저자들은 이 점진적 학습이 한 번에 전부 joint training하는 것보다 더 안정적이고 더 높은 정확도를 준다고 보고한다.

계산 측면에서도 논문의 메시지는 분명하다. DPN은 복잡한 mean field 연산을 convolution과 pooling으로 바꾸었기 때문에 GPU 병렬화가 쉽다. 특히 가장 무거운 b12도 미니배치 10장에서 약 30ms 정도라고 주장하고, 마지막 4개 층 전체가 75ms라고 보고한다. 직접 mean field 식을 반복 계산하는 dense CRF류 구현보다 훨씬 병렬화 친화적이라는 것이 저자들의 주장이다.

## 4. 실험 및 결과

실험은 PASCAL VOC 2012에서 수행되었다. 클래스는 20개 객체 클래스와 1개 배경 클래스, 총 21개이며, 학습 10,582장, 검증 1,449장, 테스트 1,456장을 사용한다. 기본 평가 지표는 mean Intersection-over-Union, 즉 mIoU이다.

흥미로운 점은 저자들이 mIoU만으로는 모델의 특성을 충분히 설명하기 어렵다고 보고 세 가지 추가 지표를 도입했다는 점이다. **Tagging Accuracy (TA)**는 이미지 수준 클래스 존재 여부를 얼마나 맞추는지, **Localization Accuracy (LA)**는 예측 segmentation으로부터 얻은 bounding box와 GT box 사이의 IoU를 재는 bIoU, **Boundary Accuracy (BA)**는 맞게 localize된 객체에 대해 경계 정밀도를 평가한다. 이 세 지표 덕분에 “왜 특정 클래스가 어려운가”를 좀 더 세분화해서 분석할 수 있다.

먼저 ablation study를 보면, **b12의 receptive field**는 $50 \times 50$가 가장 좋았다. baseline 63.4%에서 $10 \times 10$은 63.8%, $50 \times 50$은 64.7%, $100 \times 100$은 64.3%였다. 너무 작은 영역은 관계를 충분히 못 보고, 너무 큰 영역은 오히려 overfitting이나 부정확한 smoothing을 유발한다고 해석할 수 있다.

**b13의 receptive field와 mixture 효과**도 검증했다. $1 \times 1$은 64.8%, $5 \times 5$는 66.0%, $9 \times 9$는 66.3%, 그리고 $9 \times 9$ mixtures는 66.5%였다. 여기서 $1 \times 1$은 사실상 local spatial context 없이 전역적인 label compatibility만 배우는 경우인데, $5 \times 5$ 이상에서 성능이 꽤 오른다. 즉 spatial context 자체가 중요하다는 증거다. mixture를 추가하면 또 소폭 향상되는데, 이는 하나의 고정된 context보다 여러 패턴을 섞어 표현하는 것이 실제 장면 구조와 더 잘 맞는다는 뜻이다.

단계별 성능 향상도 명확하다. validation set에서 unary term만 쓴 경우 mIoU가 62.4%, 여기에 triple penalty를 더하면 64.7%, label contexts까지 더하면 66.5%, 마지막 joint tuning 후에는 67.8%가 된다. 논문은 각 단계가 이전 단계 대비 실제로 개선된다고 강조한다. 특히 triple penalty가 +2.3%, label contexts가 +1.8%, final joint tuning이 추가로 +1.3% 향상시켰다.

또한 저자들은 DPN이 **one-iteration mean field approximation**만으로도 충분히 높은 성능을 낸다고 보인다. 비교 대상으로 든 dense CRF는 5회 이상 반복해야 수렴하는 반면, DPN은 1회 근사만으로도 좋은 정확도에 도달한다고 한다. 이는 본 논문의 가장 중요한 실용적 주장 중 하나다.

추가 지표 분석도 의미 있다. TA는 label contexts를 넣는 단계에서 향상되는데, 이는 클래스 공출현과 context mixture가 이미지 수준 태그 예측에도 도움을 준다는 뜻이다. LA는 triple penalty와 final joint tuning에서 크게 오른다. 이는 long-range, high-order relation이 객체 위치 추정에 직접 기여한다는 해석과 맞는다. BA도 triple penalty 단계에서 개선되는데, 이는 pixel dissimilarity를 반영하는 구조가 경계 보존에 도움을 준다고 볼 수 있다.

클래스별 분석에서는 흥미로운 관찰이 나온다. 예를 들어 joint training이 대부분 클래스에 도움이 되지만, 매우 작은 동물 객체들, 예를 들어 bird, cat, cow에서는 오히려 약간 손해일 수 있다고 저자들은 말한다. 이는 전체 segmentation 정확도를 높이는 방향으로 학습하다 보니, 데이터셋에서 드물고 작은 객체는 매끄러운 결과를 위해 희생될 수 있다는 해석이다. 또 bike는 localization은 잘 되지만 boundary와 mIoU가 낮다고 분석한다. 즉 객체의 위치는 대략 맞히지만 복잡한 경계를 세밀하게 따는 것이 어렵다는 뜻이다.

최종 비교에서, VOC12만 사용한 설정에서는 DPN이 테스트셋에서 **74.1% mIoU**를 달성했다. COCO pre-training까지 사용한 DPN†는 단일 모델로 **77.5% mIoU**를 기록했다고 보고한다. 논문 시점 기준으로 이는 state-of-the-art라고 주장한다. 표에 따르면 VOC12만 사용한 조건에서는 FCN 62.2%, Zoom-out 69.6%, Piecewise 70.7%, DeepLab 71.6%, RNN 72.0%, WSSL 73.9%, DPN 74.1% 순으로 DPN이 가장 높다. COCO pre-training까지 포함한 비교에서도 DPN† 77.5%가 RNN† 74.7%, BoxSup† 75.2%보다 높다.

논문은 또한 정확도뿐 아니라 효율을 강조한다. 이전 방법들은 mean field를 대개 5회에서 10회 반복해야 했지만, DPN은 한 번만 수행하므로 구조적으로 큰 속도 이점이 있다고 주장한다. 다만 이 비교는 동일한 하드웨어와 완전히 동일한 구현 조건에서 직접 head-to-head로 보고된 것은 아니므로, 속도 이득의 절대값은 논문이 제시한 계산 분석 범위 내에서 해석하는 것이 안전하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 structured prediction을 CNN 내부 연산으로 흡수한 방식이 상당히 명확하다는 점이다. 단순히 “CNN 뒤에 CRF를 붙였다”가 아니라, mean field 업데이트를 어떻게 convolution/pooling으로 근사하는지 수식과 아키텍처 대응 관계를 비교적 설득력 있게 제시한다. 특히 b12, b13, b14, b15 각각이 식 (7)의 어느 부분에 대응하는지 설명이 잘 되어 있어, 모델 설계 의도가 분명하다.

두 번째 강점은 pairwise term의 표현력이 기존보다 풍부하다는 점이다. local label context와 triple penalty를 모두 포함해, 단순한 smoothness나 global co-occurrence를 넘어 spatial context와 higher-order effect를 모델링하려는 시도는 당시 segmentation 문맥에서 의미가 크다. 또한 기존 DeepLab, RNN류 모델이 DPN의 특수한 경우라고 연결하는 방식도 개념적으로 강하다.

세 번째 강점은 실험 설계가 단순 성능 비교에 그치지 않는다는 점이다. receptive field, mixture, incremental learning, MF iteration 수, TA/LA/BA 같은 보조 지표까지 통해 어떤 구성 요소가 무슨 정보를 담당하는지 분석하려고 한다. 이런 실험은 모델의 내부 역할을 이해하는 데 도움을 준다.

하지만 한계도 분명하다. 우선 DPN은 “한 번의 mean field iteration 근사로 충분하다”는 강한 주장을 한다. 실험적으로는 성능이 좋지만, 이것이 일반적으로 언제나 성립하는지는 논문만으로 확언하기 어렵다. 특히 더 복잡한 데이터셋, 더 많은 클래스, 더 큰 appearance variation에서 동일하게 유지되는지는 논문 말미에서도 future work로 남겨 두고 있다.

또한 b12는 각 spatial position마다 다른 local filter를 쓰는 locally convolution 구조인데, 이는 파라미터 구조와 메모리, 구현 복잡도 면에서 상당히 무거울 수 있다. 논문은 GPU 병렬화와 lookup table 기반 최적화를 제시하지만, 실제 시스템 설계 측면에서는 여전히 비용이 작은 구조라고 보기는 어렵다. 즉 반복 MF를 줄인 대신, 구조적으로 큰 locally convolution을 도입한 셈이다.

한편 higher-order relation을 강조하지만, 실제 구현은 triple penalty 형태로 제한되어 있다. 따라서 “high-order”라는 표현은 맞지만, 임의의 고차 clique를 일반적으로 다루는 수준까지는 아니다. 또한 b12의 필터는 VOC12의 경우 고정된 Euclidean distance 기반 초기화와 제한적 학습 해석을 가지므로, truly learned high-order interaction이라고 보기에는 다소 조심스러운 면도 있다.

실험 비교에도 주의점이 있다. 비교 대상 중 일부는 COCO pre-training을 사용하고 일부는 사용하지 않기 때문에, 표를 읽을 때 같은 조건끼리 비교해야 한다. 논문도 이를 구분하고 있지만, 최종 state-of-the-art 주장에서는 이 점을 항상 염두에 둘 필요가 있다. 또 시각적 비교는 FCN은 공개 모델로 재생성했지만 DeepLab은 논문 그림을 가져왔다고 적혀 있어, 정성 비교의 조건이 완전히 통일된 것은 아니다.

비판적으로 보면, DPN은 당시 기준으로 매우 창의적인 구조였지만, 이후 관점에서 보면 hand-crafted structural prior와 복잡한 custom layer 설계에 많이 의존한다. 즉 설계의 해석 가능성은 높지만, 구조의 단순성과 범용성은 다소 떨어질 수 있다. 그럼에도 논문 자체가 제시한 범위 안에서는 “복잡한 MRF 구조를 CNN 안으로 녹이는 방법”이라는 점에서 충분히 가치가 있다.

## 6. 결론

이 논문은 semantic segmentation을 위해 Deep Parsing Network를 제안하고, VGG-16 기반 unary term과 richer pairwise term을 하나의 CNN 안에서 end-to-end로 학습하는 방법을 제시했다. 특히 pairwise term에 mixture of local label contexts와 triple penalty를 포함시켜, 기존의 단순한 CRF pairwise보다 더 풍부한 공간적 문맥과 고차 상호작용을 다루려 했다. 그리고 이를 mean field 반복 추론 대신 한 번의 forward pass로 근사할 수 있도록 설계한 것이 핵심 기여다.

실험적으로는 PASCAL VOC 2012에서 강한 성능을 보였고, 특히 단일 DPN†가 테스트셋에서 77.5% mIoU를 달성했다고 보고했다. 또한 각 구성 요소가 tagging, localization, boundary에 어떤 영향을 주는지 분석하여 단순한 성능 보고를 넘는 통찰도 제공한다.

실제 적용 측면에서 이 연구는 “structured prediction과 deep network를 어떻게 계산 효율적으로 통합할 것인가”라는 중요한 방향을 보여준다. 향후 연구 관점에서는 더 큰 데이터셋, 더 많은 클래스, 더 다양한 장면 구조에서 이러한 one-pass structured inference가 얼마나 일반화되는지가 핵심 후속 과제가 될 것이다. 논문 자체도 이 점을 인정하고 있으며, 그 의미에서 DPN은 완성형 해답이라기보다 CNN과 graphical model의 결합을 더 깊게 밀어붙인 중요한 단계로 이해하는 것이 적절하다.
