# Exploring Context with Deep Structured models for Semantic Segmentation

- **저자**: Guosheng Lin, Chunhua Shen, Anton van den Hengel, Ian Reid
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1603.03183

## 1. 논문 개요

이 논문은 semantic segmentation, 즉 이미지의 모든 픽셀에 클래스 라벨을 부여하는 문제에서, 단순히 픽셀 또는 patch 자체의 시각 특징만 보는 것이 아니라 **context**를 명시적으로 활용하면 성능을 크게 높일 수 있다는 문제의식에서 출발한다. 저자들은 특히 두 종류의 공간적 문맥을 구분한다. 하나는 인접한 두 patch 사이의 의미적 관계를 뜻하는 **patch-patch context**이고, 다른 하나는 특정 patch와 그 주변의 더 넓은 배경 영역 사이의 관계를 뜻하는 **patch-background context**이다.

논문이 다루는 핵심 연구 문제는 다음과 같다. 기존 CNN 기반 segmentation은 강력하지만, 객체와 배경, 혹은 객체들 사이의 의미적 공존 관계를 충분히 직접 모델링하지 못한다. 예를 들어 car는 road 위에 있을 가능성이 높고, boat는 road 위에 있을 가능성이 낮다. 이런 관계는 segmentation에서 매우 중요한데, 기존의 많은 CRF 기반 후처리 방식은 주로 경계선을 날카롭게 만드는 local smoothness에 집중했지, 이런 의미적 compatibility 자체를 학습하지는 않았다.

이 문제는 중요하다. semantic segmentation은 자율주행, 로보틱스, 실내 장면 이해 같은 실제 응용의 기반 기술이기 때문이다. 특히 시각적으로 애매한 물체는 그 자체 외형만으로는 분류가 어렵고, 주변 맥락이 결정적 단서가 되는 경우가 많다. 따라서 이 논문은 CNN의 표현력과 CRF의 구조적 모델링 능력을 결합해, coarse prediction 단계부터 문맥 정보를 적극적으로 반영하려고 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **CNN으로 unary potential뿐 아니라 pairwise potential도 직접 학습하는 deep structured CRF**를 만들자는 것이다. 즉, 각 위치가 어떤 클래스일지를 예측하는 unary term만 쓰지 않고, 서로 연결된 두 위치가 어떤 클래스 쌍을 가지는 것이 자연스러운지를 CNN이 학습하도록 설계한다.

기존 DeepLab류의 dense CRF는 주로 Potts-model 기반 pairwise term을 사용해 색 대비에 따라 인접 픽셀을 부드럽게 연결하고 object boundary를 정제하는 역할을 했다. 반면 이 논문은 pairwise term의 목적이 다르다. 이들은 pairwise term을 coarse prediction 단계에서 사용해, “위쪽 patch가 sky이고 아래쪽 patch가 road일 가능성”, “어떤 patch 주변에 특정 물체가 함께 나타날 가능성” 같은 **semantic compatibility**를 학습한다. 즉, 단순 smoothness가 아니라 **의미적 관계 자체**를 모델링한다는 점이 차별점이다.

또 하나의 핵심은 patch-background context를 위해 **multi-scale image input**과 **sliding pyramid pooling**을 결합한 FeatMap-Net 구조를 사용한 것이다. 이는 feature map이 더 넓은 배경 정보를 담게 하여, 한 위치의 예측이 주변 장면의 전반적 구조를 반영하도록 만든다.

마지막으로, 이런 deep CRF는 학습이 비싸다는 문제가 있다. 일반적인 CRF 최대우도 학습은 각 SGD step마다 추론이 필요할 수 있는데, 이는 매우 비효율적이다. 저자들은 이를 해결하기 위해 **piecewise training**을 도입하여 global partition function을 직접 다루지 않고도 효율적으로 학습한다.

## 3. 상세 방법 설명

전체 시스템은 크게 네 부분으로 이해할 수 있다. 첫째, 입력 이미지를 받아 feature map을 만드는 **FeatMap-Net**이 있다. 둘째, 각 node의 클래스 점수를 내는 **Unary-Net**이 있다. 셋째, 연결된 두 node의 클래스 조합 점수를 내는 **Pairwise-Net**이 있다. 넷째, 이 unary/pairwise potential을 이용해 CRF inference를 수행하고, 이후 upsampling과 boundary refinement를 적용해 최종 segmentation을 만든다.

CRF graph 구성부터 보면, 입력 이미지를 FeatMap-Net에 통과시켜 낮은 해상도의 feature map을 만든 뒤, feature map의 각 spatial position을 하나의 CRF node로 둔다. 그리고 각 node는 미리 정의한 spatial range box 안의 다른 node들과 연결된다. 논문은 두 종류의 spatial relation을 사용한다. 하나는 중심 node 주변 이웃을 보는 **surrounding relation**, 다른 하나는 위아래 방향성을 반영하는 **above/below relation**이다. 실험에서는 range box 크기를 feature map 짧은 변 길이 $a$에 대해 $0.4a \times 0.4a$로 둔다.

CRF의 조건부 확률은 다음과 같이 정의된다.

$$
P(y|x)=\frac{1}{Z(x)}\exp[-E(y,x)]
$$

여기서 $E(y,x)$는 energy function이고, $Z(x)$는 partition function이다. energy는 unary potential과 pairwise potential의 합으로 구성된다.

$$
E(y,x)=\sum_{U\in \mathcal{U}}\sum_{p\in \mathcal{N}_U} U(y_p,x_p)+\sum_{V\in \mathcal{V}}\sum_{(p,q)\in \mathcal{S}_V} V(y_p,y_q,x_{pq})
$$

이 식의 의미는 간단하다. 각 위치 $p$가 어떤 클래스를 갖는 것이 얼마나 자연스러운지 unary term이 평가하고, 연결된 두 위치 $(p,q)$가 어떤 클래스 조합을 갖는 것이 얼마나 자연스러운지 pairwise term이 평가한다.

Unary potential은 다음처럼 정의된다.

$$
U(y_p,x_p;\theta_U)=-z_{p,y_p}(x;\theta_U)
$$

여기서 $z_{p,y_p}$는 Unary-Net의 출력이다. 즉, 특정 node $p$가 클래스 $y_p$일 때의 점수를 의미하며, 점수가 높을수록 해당 라벨이 더 적합하다는 뜻이다. 실제로는 FeatMap-Net의 feature map에서 node 위치의 feature vector를 뽑아 Unary-Net에 넣고, 클래스 수 $K$차원의 출력을 얻는다.

Pairwise potential은 다음과 같다.

$$
V(y_p,y_q,x_{pq};\theta_V)=-z_{p,q,y_p,y_q}(x;\theta_V)
$$

여기서 $z_{p,q,y_p,y_q}$는 node 쌍 $(p,q)$가 라벨 조합 $(y_p,y_q)$를 가질 때의 compatibility score이다. Pairwise-Net의 입력은 두 node feature를 concat한 edge feature이다. 출력 차원은 가능한 라벨 조합 수에 맞춰 $K^2$이다. 즉, 이 모델은 각 클래스 쌍에 대해 별도의 의미적 적합도를 학습한다.

논문은 **asymmetric pairwise potential**도 중요하게 다룬다. 예를 들어 above/below 관계는 대칭적이지 않다. sky가 road 위에 있는 것은 자연스럽지만, road가 sky 위에 있는 것은 그렇지 않다. 이를 위해 edge feature를 만들 때 $(p,q)$ 순서대로 feature를 concat한다. 따라서 일반적으로

$$
V(y_p,y_q,x_{pq}) \ne V(y_q,y_p,x_{qp})
$$

가 가능해지고, 공간적 방향성을 표현할 수 있다.

Patch-background context를 위한 FeatMap-Net은 3-scale 입력을 사용한다. 이미지를 $1.2$, $0.8$, $0.4$ 배 크기로 만든 뒤, 각 scale을 6개의 convolution block에 통과시킨다. 상위 5개 block은 scale 간 공유되고, 6번째 block은 scale별로 독립적이다. 이후 서로 다른 해상도의 feature map을 bilinear interpolation으로 맞춘 뒤 concat한다.

여기에 더해 **sliding pyramid pooling**을 적용한다. 각 scale의 feature map에 대해 $5\times5$와 $9\times9$ sliding max-pooling을 적용해 여러 크기의 배경 문맥을 요약하고, 이를 원래 feature map과 concat한다. 결과적으로 feature가 더 넓은 receptive field를 갖게 되며, patch가 주변 scene 구조를 더 잘 반영하게 된다.

네트워크 구성은 VGG-16을 기반으로 한다. 상위 5개 conv block은 VGG-16과 유사하고, 첫 번째 fully-connected layer를 convolution으로 바꾸어 포함한다. 다만 기존 FCN 계열이 2개의 FC layer를 convolution화하는 것과 달리, 이 논문은 하나만 옮겨와 효율을 높인다. 또 stride를 덜 공격적으로 줄이는 대신, 추가적인 Conv Block 6의 두 개 $3\times3$ convolution layer를 넣어 field-of-view 감소 문제를 보완한다.

예측은 두 단계로 이뤄진다. 먼저 coarse-level prediction 단계에서 contextual CRF에 대해 mean field approximation 기반 inference를 수행해 low-resolution score map을 얻는다. 논문은 3번의 mean field iteration을 사용했다. 그다음 prediction refinement 단계에서 이 score map을 bilinear upsampling하고, dense CRF를 적용해 경계를 정제한다. 즉, 이 논문은 **semantic pairwise CRF**와 **boundary refinement용 dense CRF**를 서로 다른 단계에서 함께 사용한다.

학습에서 가장 중요한 부분은 piecewise training이다. 원래 CRF의 negative log-likelihood는

$$
-\log P(y|x;\theta)=E(y,x;\theta)+\log Z(x;\theta)
$$

이며, $\log Z(x;\theta)$의 gradient 계산에는 전체 구조에 대한 추론이 필요하다. 이는 deep network 학습과 결합될 때 매우 비싸다. 그래서 논문은 global likelihood를 각 potential별 독립 likelihood의 곱으로 근사한다.

Unary piece는

$$
P_U(y_p|x)=\frac{\exp[-U(y_p,x_p)]}{\sum_{y_p'}\exp[-U(y_p',x_p)]}
$$

Pairwise piece는

$$
P_V(y_p,y_q|x)=\frac{\exp[-V(y_p,y_q,x_{pq})]}{\sum_{y_p',y_q'}\exp[-V(y_p',y_q',x_{pq})]}
$$

로 둔다. 그러면 전체 piecewise log-likelihood는 각 unary/pairwise 항의 합이 되어, softmax 수준의 정규화만 필요하고 global inference가 필요 없어진다. 이것이 이 논문의 효율성 핵심이다.

구현 측면에서는 pairwise edge 수가 너무 많아 계산량과 메모리 사용량이 커지므로, 각 node당 원래 수백 개 연결 대신 **5x5 regular grid 기반으로 24개 이웃만 샘플링**한다. 또한 Pairwise-Net 학습에서는 모든 edge를 한 번에 처리하지 않고, 예를 들어 2000개씩 여러 sub-iteration으로 나눠 업데이트하는 **asynchronous gradient update**를 사용한다.

## 4. 실험 및 결과

논문은 총 8개 dataset에서 평가한다. PASCAL VOC 2012, NYUDv2, PASCAL-Context, SIFT-flow, SUN-RGBD, KITTI, COCO, Cityscapes를 사용했다. 평가지표는 pixel accuracy, mean accuracy, IoU이며, IoU는 클래스별 intersection-over-union의 평균이다.

NYUDv2에서는 RGB-D 데이터셋이지만, 저자들은 depth를 사용하지 않고 RGB만으로 학습했다. 그럼에도 IoU 40.6을 얻어 FCN-32s의 29.2, FCN-HHA의 34.0보다 높았다. 이 결과는 문맥 모델링의 효과를 잘 보여준다. Ablation에서도 baseline fully convolution network가 IoU 30.5였고, sliding pyramid pooling을 넣으면 32.4, multi-scale을 추가하면 37.0, boundary refinement를 추가하면 38.3, 마지막으로 CNN contextual pairwise를 추가하면 40.6이 되었다. 즉, background context와 pairwise context가 각각 의미 있는 성능 향상을 만든다.

또한 단순히 unary network를 여러 개 앙상블한 것과 비교한 실험도 중요하다. 4개의 unary ensemble은 IoU 38.7이었지만, pairwise 1개와 unary 1개를 쓰는 CRF 모델은 38.9였다. 즉, 성능 향상이 단순한 모델 수 증가 때문이 아니라, **pairwise potential이 새로운 정보를 포착하기 때문**임을 보여준다.

PASCAL VOC 2012에서는 VOC 데이터만 쓴 경우 test set IoU 75.3을 달성했고, VOC+COCO를 사용하면 77.2가 된다. 여기에 middle-layer feature를 활용한 refinement network를 추가해 최종적으로 **77.8 IoU**를 달성했다고 보고한다. 이는 당시 최고 수준의 성능이었다. 클래스별 결과에서도 대부분의 카테고리에서 경쟁 방법보다 우수했다.

Cityscapes test set에서는 IoU 71.6을 기록해 FCN-8s 65.3, Dilation10 67.1, DeepLab-CRF 63.1보다 높았다. PASCAL-Context에서는 pixel accuracy 71.5, mean accuracy 53.9, IoU 43.3을 기록했다. SUN-RGBD에서도 depth 없이 RGB만으로 IoU 42.3을 얻어 기존 RGB 기반 Kendall et al.의 30.7을 크게 넘었다.

COCO에서는 baseline FullyConvNet이 IoU 37.2, refinement 추가가 41.3, 제안 방법이 46.8이었다. SIFT-flow에서는 FCN-16s의 IoU 39.5보다 높은 44.9를 기록했고, KITTI에서는 68.5, COCO pretraining을 추가한 ours+는 70.3이었다.

전체적으로 실험은 두 가지를 입증한다. 첫째, **patch-background context**를 위한 multi-scale + sliding pyramid pooling이 강력하다. 둘째, **patch-patch context**를 위한 CNN pairwise potential이 추가적인 향상을 만든다. 그리고 이 조합은 여러 데이터셋에서 일관되게 효과가 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은, 당시 많은 segmentation 방법이 CRF를 단지 경계선 보정용 후처리로 사용하던 것과 달리, **semantic relation을 직접 학습하는 pairwise potential**을 CNN으로 설계했다는 점이다. 이는 context를 좀 더 본질적으로 모델링하려는 시도다. 또한 asymmetric relation까지 자연스럽게 다룰 수 있게 설계한 점도 강점이다.

둘째, piecewise training을 통해 deep CRF 학습의 계산 문제를 실용적으로 해결했다. 실제로 이 논문은 pairwise term이 있는 구조를 대규모 데이터셋에 적용해 SOTA 수준 결과를 냈다. 이는 아이디어가 이론적 제안에 그치지 않고, 실제 학습 가능하도록 잘 다듬어졌음을 보여준다.

셋째, 실험이 매우 폭넓다. 8개 dataset에서 indoor, outdoor, street scene, object-centric segmentation을 모두 평가했고, 대부분에서 strong baseline 대비 일관된 향상을 보였다. 특히 일부 RGB-D 데이터셋에서 depth를 쓰지 않고도 강한 결과를 낸 점은 context modeling의 실효성을 잘 보여준다.

한계도 분명하다. 첫째, 최종 예측은 여전히 두 단계로 나뉜다. coarse prediction은 contextual CRF가 담당하지만, 최종 고해상도 결과는 bilinear upsampling과 dense CRF 혹은 별도 refinement network에 의존한다. 즉, 고해상도 boundary prediction까지 하나의 통합 구조로 완결되지는 않는다.

둘째, piecewise training은 효율적이지만, 본래의 global CRF likelihood를 직접 최적화하는 것은 아니다. 따라서 학습 objective가 근사적이며, 구조적 의존성을 완전히 반영한다고 보기 어렵다. 논문도 이를 효율성을 위한 practical approximation으로 제시한다.

셋째, pairwise connection sampling과 asynchronous update는 계산량을 낮추기 위한 현실적 설계지만, 동시에 모든 pairwise relation을 완전하게 활용하지 못한다는 뜻이기도 하다. 이 선택이 성능에 어떤 trade-off를 만드는지는 논문이 정성적으로는 설명하지만, 더 깊은 분석은 제한적이다.

넷째, 이 논문은 VGG-16 기반 구조와 low-resolution feature map에 의존한다. 따라서 이후 등장한 dilated convolution, encoder-decoder, transformer 기반 dense prediction 구조와 비교하면 표현력과 해상도 복원 방식에서 시대적 한계가 있다. 다만 이것은 후속 연구 관점의 비판이며, 논문 내부 근거만 놓고 보면 당시 설계로는 매우 설득력 있는 결과를 냈다.

## 6. 결론

이 논문은 semantic segmentation에서 context를 활용하는 두 축, 즉 **patch-patch context**와 **patch-background context**를 함께 탐구했다. 이를 위해 CNN 기반 unary/pairwise potential을 갖는 contextual deep CRF를 제안했고, multi-scale input과 sliding pyramid pooling으로 배경 문맥을 강화했으며, piecewise training으로 학습을 실용화했다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, pairwise potential을 단순 smoothness가 아니라 semantic compatibility 학습으로 확장했다. 둘째, deep structured model을 반복 추론 없이 효율적으로 학습하는 방법을 제시했다. 셋째, 여러 대규모 benchmark에서 강한 성능을 입증했다.

실제 적용 측면에서 이 연구는 장면 이해, 자율주행, 로보틱스처럼 공간 문맥이 중요한 문제에 의미 있는 영향을 줄 수 있다. 향후 연구 측면에서도, 이 논문은 segmentation에서 “문맥을 어떻게 구조적으로 학습할 것인가”라는 방향을 분명히 제시했다는 점에서 가치가 크다. 특히 후속의 더 강한 backbone, 더 정교한 refinement, 혹은 end-to-end structured prediction과 결합될 여지가 큰 연구라고 볼 수 있다.
