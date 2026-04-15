# Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation

- **저자**: Guosheng Lin, Chunhua Shen, Anton van den Hengel, Ian Reid
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1504.01013

## 1. 논문 개요

이 논문은 semantic segmentation, 즉 이미지의 모든 픽셀에 semantic label을 할당하는 문제를 다룬다. 저자들은 당시의 강력한 baseline이던 fully convolutional neural networks (FCNNs)가 좋은 성능을 내고 있지만, 주로 로컬한 appearance 정보에 크게 의존하고 있으며, 이미지 패치들 사이의 문맥적 관계를 충분히 직접 모델링하지 못한다고 본다. 특히 어떤 물체나 영역은 단독 appearance만으로는 애매할 수 있기 때문에, 주변 패치와의 관계나 더 넓은 배경과의 관계를 함께 활용하는 것이 중요하다고 주장한다.

논문이 해결하려는 핵심 연구 문제는 두 가지다. 첫째, 패치와 패치 사이의 semantic compatibility를 어떻게 명시적으로 모델링할 것인가이다. 둘째, 그런 구조적 모델을 deep network와 함께 학습할 때 발생하는 비싼 CRF inference 비용을 어떻게 줄여 실제 학습이 가능하도록 만들 것인가이다. 이를 위해 저자들은 CNN 기반 unary potential뿐 아니라 CNN 기반 pairwise potential을 갖는 contextual CRF를 제안하고, 학습 시에는 piecewise training을 사용해 전역 partition function 계산과 반복적인 inference를 피한다.

이 문제가 중요한 이유는 semantic segmentation이 scene understanding의 핵심 과제이기 때문이다. 예를 들어 자동차는 도로 위에 있을 가능성이 높고 하늘에 둘러싸여 있을 가능성은 낮다. 이런 관계는 단순 smoothing이 아니라 semantic relation의 문제이다. 논문은 이러한 관계를 explicit하게 넣는 것이 coarse prediction 자체를 더 좋게 만들 수 있다고 본다. 이는 기존의 dense CRF처럼 boundary refinement만 담당하는 방식과는 다른 방향의 개선이다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 contextual information을 두 종류로 나누어 각각 활용하는 것이다. 하나는 **patch-patch context**로, 서로 이웃한 두 image patch 사이의 semantic relation을 의미한다. 다른 하나는 **patch-background context**로, 하나의 patch가 더 넓은 배경 영역과 맺는 관계를 의미한다. 저자들은 전자를 CRF의 pairwise potential로, 후자를 multi-scale network와 sliding pyramid pooling으로 다룬다.

기존 CNN+CRF 계열 방법과의 가장 큰 차별점은 pairwise potential의 역할과 형태에 있다. DeepLab이나 CRF-RNN류의 방법은 주로 Potts-model 기반 pairwise potential을 사용해 local smoothness와 boundary sharpening을 유도한다. 반면 이 논문은 $K \times K$ label 조합 각각에 대해 CNN이 직접 compatibility score를 예측하는 **general pairwise potential**을 사용한다. 즉, "이웃은 비슷해야 한다"는 단순 smoothness가 아니라, "road 위에 car가 오는 것은 자연스럽다", "sky 아래에 car가 오는 것은 부자연스럽다" 같은 semantic relation을 학습하려는 것이다.

또 하나의 핵심은 학습 전략이다. 일반적인 CRF 최대우도 학습은 $\log Z(x;\theta)$와 그 gradient 때문에 매 SGD step마다 expensive inference가 필요하다. 논문은 이를 피하기 위해 piecewise training을 도입한다. 이 방식은 전체 CRF likelihood를 각 unary/pairwise potential별 local likelihood들의 곱으로 근사하므로, 전역 partition function 없이도 학습할 수 있다. 이 점이 실제 deep structured model을 효율적으로 학습할 수 있게 만드는 실질적 기여다.

## 3. 상세 방법 설명

전체 시스템은 크게 세 단계로 이해할 수 있다. 먼저 `FeatMap-Net`이 입력 이미지에서 저해상도 feature map을 만든다. 그 다음 이 feature map의 각 spatial location을 CRF의 node로 보고, 일정 범위 안의 node 쌍들을 edge로 연결해 contextual CRF를 구성한다. 마지막으로 unary potential과 pairwise potential을 각각 별도의 shallow network가 예측하고, mean field inference로 coarse prediction을 얻은 뒤, bilinear upsampling과 dense CRF로 최종 결과를 refinement한다.

CRF의 조건부 분포는 다음과 같이 정의된다.

$$
P(y|x)=\frac{1}{Z(x)}\exp[-E(y,x)]
$$

여기서 $E(y,x)$는 energy function이고, $Z(x)$는 partition function이다. 출력 $y$는 CRF graph의 모든 node label assignment를 뜻한다. 에너지는 unary term과 pairwise term의 합으로 구성된다.

$$
E(y,x)=\sum_{U\in\mathcal{U}}\sum_{p\in N_U} U(y_p,x_p) + \sum_{V\in\mathcal{V}}\sum_{(p,q)\in S_V} V(y_p,y_q,x_{pq})
$$

즉, 각 node에 대해 unary potential을 더하고, 연결된 node pair마다 pairwise potential을 더한다. 여기서 $x_p$는 node $p$에 해당하는 image region, $x_{pq}$는 edge $(p,q)$에 대응하는 두 region의 정보다.

### Unary potential

Unary potential은 `FeatMap-Net`의 각 위치 feature vector를 받아 `Unary-Net`이 각 class에 대한 score를 출력하는 구조다. 수식은 다음과 같다.

$$
U(y_p, x_p; \theta_U) = - z_{p,y_p}(x;\theta_U)
$$

여기서 $z_{p,y_p}$는 node $p$가 class $y_p$일 때의 score다. 음수 부호가 붙는 이유는 score가 클수록 해당 label이 더 그럴듯하므로 energy는 더 작아져야 하기 때문이다. 사실상 standard CNN classifier score를 energy framework 안에 넣은 형태라고 이해하면 된다.

### Pairwise potential

이 논문의 핵심은 pairwise potential이다. 연결된 두 node의 feature vector를 concatenation하여 edge feature를 만들고, 이를 `Pairwise-Net`에 넣어 가능한 모든 label pair에 대한 score를 출력한다. 수식은 다음과 같다.

$$
V(y_p, y_q, x_{pq}; \theta_V) = - z_{p,q,y_p,y_q}(x;\theta_V)
$$

여기서 $z_{p,q,y_p,y_q}$는 node pair $(p,q)$가 label pair $(y_p,y_q)$를 가질 때의 compatibility score다. 출력 차원은 $K \times K$이며, 이는 class 수가 $K$일 때 가능한 모든 label 조합에 대응한다. 이 구조는 Potts model처럼 "같으면 좋고 다르면 나쁘다" 수준의 단순한 형식이 아니라, label pair마다 별도 parameterization이 가능하므로 더 풍부한 semantic relation을 표현할 수 있다.

논문은 특히 비대칭 관계(asymmetric relation)도 모델링할 수 있음을 강조한다. 예를 들어 "above/below" 관계는 순서가 중요하므로 일반적으로

$$
V(y_p,y_q,x_{pq}) \neq V(y_q,y_p,x_{qp})
$$

가 되어야 한다. 이 방법은 입력 순서를 pairwise network에 그대로 반영하기 때문에, "node $p$가 node $q$ 위에 있다" 같은 방향성 있는 관계를 자연스럽게 다룰 수 있다.

### CRF graph 구성

CRF node는 feature map의 각 spatial location에 대응한다. pairwise edge는 한 node에서 일정 spatial range box 안에 있는 다른 node들에 연결하는 방식으로 만든다. 논문은 두 가지 spatial relation을 쓴다.

- `surrounding`
- `above/below`

즉, 모든 이웃을 하나로 보는 것이 아니라, 어떤 상대적 위치 관계인지에 따라 서로 다른 type의 pairwise potential을 쓴다. 실험에서는 range box 크기를 feature map의 짧은 변 길이 $a$에 대해 $0.4a \times 0.4a$로 둔다고 설명한다.

### Patch-background context: multi-scale + sliding pyramid pooling

저자들은 넓은 배경 문맥을 담기 위해 `FeatMap-Net` 자체도 강화한다. 입력 이미지를 3개 scale로 resize한 뒤, 각 scale을 convolution block 6개를 거쳐 feature map으로 변환한다. 사용한 scale은 $1.2$, $0.8$, $0.4$이다. 상위 5개 convolution block은 모든 scale이 공유하고, 마지막 6번째 block은 scale별 전용으로 둔다. 이렇게 한 뒤 해상도가 작은 두 feature map은 bilinear interpolation으로 가장 큰 feature map 크기에 맞춰 upsample하고, 세 feature map을 concatenate한다.

여기에 더해 sliding pyramid pooling을 적용한다. 논문은 전통적인 spatial pyramid pooling을 feature map 위에서 sliding manner로 수행했다고 설명한다. 실험에서는 $5 \times 5$와 $9 \times 9$ max-pooling window를 사용한 2-level pooling을 수행하고, 이렇게 얻은 pooled feature map들을 원래 feature map과 concatenate하여 최종 feature map을 만든다. 이 과정은 receptive field, 즉 field-of-view를 넓혀 큰 배경 영역 정보를 feature에 반영하는 역할을 한다.

### Prediction 절차

예측은 coarse-level prediction과 refinement stage로 나뉜다. 먼저 contextual CRF에서 node marginal을 추론한다.

$$
\forall p \in N : P(y_p|x)=\sum_{y \setminus y_p} P(y|x)
$$

하지만 그래프는 loopy이고 pairwise potential도 submodular하지 않아서 exact inference는 어렵다. 그래서 mean field approximation을 사용한다. 논문은 $Q(y)=\prod_{p \in N}Q_p(y_p)$ 형태의 factorized distribution으로 $P(y)$를 근사하고, 실험에서는 mean field iteration을 3번 수행했다고 명시한다.

이 coarse prediction은 입력 영상 크기의 $1/16$ 해상도다. 그래서 이후 coarse score map을 bilinear upsampling으로 입력 크기로 키운 뒤, dense CRF를 post-processing으로 적용해 경계를 정교화한다. 중요한 점은 이 dense CRF는 논문의 핵심 contextual pairwise CRF와 목적이 다르다는 것이다. 앞의 CRF는 semantic relation을 이용해 coarse prediction 자체를 좋게 만드는 역할이고, 뒤의 dense CRF는 pixel-level color contrast를 이용해 boundary를 sharpen하는 역할이다.

### 학습 목표와 piecewise training

일반적인 CRF 최대우도 학습의 negative log-likelihood는 다음과 같다.

$$
-\log P(y|x;\theta)=E(y,x;\theta)+\log Z(x;\theta)
$$

regularization을 포함한 전체 objective는

$$
\min_{\theta} \frac{\lambda}{2}\|\theta\|_2^2 + \sum_{i=1}^{N}\left[E(y^{(i)},x^{(i)};\theta)+\log Z(x^{(i)};\theta)\right]
$$

이다. 문제는 $\log Z(x;\theta)$의 gradient가 전체 label space에 대한 expectation을 포함한다는 점이다.

$$
\nabla_\theta \log Z(x;\theta)
=
- \mathbb{E}_{y \sim P(y|x;\theta)} \nabla_\theta E(y,x;\theta)
$$

semantic segmentation에서는 node 수가 많아 output space가 지수적으로 커지므로, 이 값을 직접 계산하는 것은 사실상 불가능하다. 게다가 CNN training은 수많은 SGD iteration이 필요하므로 매 step inference를 돌리는 것은 매우 비싸다.

이를 해결하기 위해 논문은 piecewise training을 사용한다. 전체 likelihood를 각 potential의 local likelihood 곱으로 근사한다.

$$
P(y|x)=\prod_{U\in\mathcal{U}}\prod_{p\in N_U} P_U(y_p|x)
\prod_{V\in\mathcal{V}}\prod_{(p,q)\in S_V} P_V(y_p,y_q|x)
$$

각 local likelihood는 다음과 같다.

$$
P_U(y_p|x)=\frac{\exp[-U(y_p,x_p)]}{\sum_{y'_p}\exp[-U(y'_p,x_p)]}
$$

$$
P_V(y_p,y_q|x)=\frac{\exp[-V(y_p,y_q,x_{pq})]}{\sum_{y'_p,y'_q}\exp[-V(y'_p,y'_q,x_{pq})]}
$$

따라서 최적화 문제는 다음처럼 바뀐다.

$$
\min_{\theta} \frac{\lambda}{2}\|\theta\|_2^2
-
\sum_{i=1}^{N}
\left[
\sum_{U\in\mathcal{U}}\sum_{p\in N_U^{(i)}} \log P_U(y_p|x^{(i)};\theta_U)
+
\sum_{V\in\mathcal{V}}\sum_{(p,q)\in S_V^{(i)}} \log P_V(y_p,y_q|x^{(i)};\theta_V)
\right]
$$

이제 전역 partition function이 사라지고, unary는 $K$개 class에 대한 softmax, pairwise는 $K^2$개 label pair에 대한 softmax처럼 계산할 수 있다. 즉, expensive global inference 없이 standard backpropagation으로 학습 가능해진다. 논문은 이 방법 덕분에 potential function들을 병렬적으로 학습할 수 있다는 점도 언급한다.

## 4. 실험 및 결과

논문은 네 개의 대표 semantic segmentation dataset에서 실험한다. NYUDv2, PASCAL VOC 2012, PASCAL-Context, SIFT-flow가 그 대상이다. 평가 지표는 intersection-over-union (IoU), pixel accuracy, mean accuracy를 사용한다. backbone의 초기화는 주로 VGG-16을 사용했고, top 5 convolution blocks와 6번째 block의 첫 convolution layer는 VGG-16으로 초기화하며, 나머지 층은 random initialization을 사용했다. 데이터 증강으로는 random scaling ($0.7$부터 $1.2$)과 horizontal flipping을 적용했다.

### NYUDv2

NYUDv2는 RGB-D indoor scene dataset이며, 논문은 depth를 사용하지 않고 RGB만으로 학습한다. 총 1449장 중 795장을 train, 654장을 test로 사용하고, 40개 class 설정을 따른다.

결과는 다음과 같다.

- Gupta et al. (RGB-D): IoU 28.6
- FCN-32s (RGB): IoU 29.2
- FCN-HHA (RGB-D): IoU 34.0
- 논문 방법 (RGB only): IoU 40.6

즉, depth 없이도 기존 RGB-D 기반 강한 baseline을 크게 넘는다. 이는 contextual modeling과 background context encoding이 상당한 효과를 냈음을 보여준다.

또한 ablation study가 매우 중요하다. NYUDv2에서 구성요소별 기여는 다음과 같다.

- FullyConvNet Baseline: IoU 30.5
- `+ sliding pyramid pooling`: IoU 32.4
- `+ multi-scales`: IoU 37.0
- `+ boundary refinement`: IoU 38.3
- `+ CNN contextual pairwise`: IoU 40.6

이 결과는 두 가지를 명확히 보여준다. 첫째, patch-background context를 위한 multi-scale과 sliding pyramid pooling만으로도 성능이 크게 오른다. 둘째, boundary refinement만으로 끝나는 것이 아니라, 제안한 CNN pairwise contextual term이 추가적으로 큰 이득을 준다. 즉, 논문의 핵심 주장인 "semantic relation을 coarse level에서 직접 모델링하는 것이 유효하다"는 점을 실험적으로 잘 뒷받침한다.

### PASCAL VOC 2012

PASCAL VOC 2012는 20개 object class와 background를 포함하는 대표 benchmark다. train 1464장, val 1449장, test 1456장이며, 일반적인 설정에 따라 extra annotated VOC images를 포함해 총 10582장의 train set으로 학습한다.

VOC 데이터만 사용했을 때 저자 방법은 test set에서 mean IoU 75.3을 기록한다. 이는 표에서 비교한 기존 방법들 중 최고 성능이다. 예를 들어 DPN이 74.1, DeconvNet이 72.5, CRF-RNN이 72.0, DeepLab이 71.6이다.

추가로 COCO 데이터를 함께 사용하면 77.2 IoU를 달성한다. 그리고 여기에 중간 계층 feature를 활용한 refinement convolution layers를 더해 coarse prediction을 보완하면 최종적으로 **78.0 IoU**를 달성한다. 논문은 이것이 당시 PASCAL VOC 2012에서 best reported result라고 주장한다.

카테고리별 결과도 제시되는데, VOC-only 설정에서 저자 방법은 DPN보다 20개 class 중 18개에서 더 좋았고, VOC+COCO 설정에서는 15개 class에서 더 좋았다. 모든 class에서 일관되게 최고인 것은 아니지만, 전체 평균과 다수 class에서 우세하다는 점이 중요하다.

### PASCAL-Context

PASCAL-Context는 object뿐 아니라 stuff label까지 포함한 더 어려운 dataset이다. 논문은 59개 class와 background를 더한 60-class setting을 사용하며, train 4998장, test 5105장이다.

결과는 다음과 같다.

- O2P: IoU 18.1
- CFM: IoU 34.4
- FCN-8s: IoU 35.1
- BoxSup: IoU 40.5
- 논문 방법: IoU 43.3

이 역시 큰 차이의 개선이며, 논문은 best reported result라고 밝힌다. 문맥 정보가 중요한 복잡한 장면 데이터셋에서 제안법의 효과가 더 잘 드러난다고 볼 수 있다.

### SIFT-flow

SIFT-flow는 33개 class를 가진 scene parsing dataset이다. 2688장 이미지 중 2488장을 train, 200장을 test로 사용하며, 이미지가 작아서 학습 시 2배 업스케일했다고 명시한다.

결과는 다음과 같다.

- FCN-16s: IoU 39.5
- 논문 방법: IoU 44.9

pixel accuracy도 88.1로 기존 방법들보다 높다. 여기서도 전체 scene context와 semantic relation이 도움이 되었음을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 CNN과 CRF를 단순히 결합한 것이 아니라, pairwise potential의 의미를 완전히 다르게 설계했다는 점이다. 기존의 많은 CNN+CRF segmentation 방법은 dense CRF를 boundary refinement 도구로 사용했다. 반면 이 논문은 pairwise potential을 통해 patch 간 semantic compatibility를 coarse prediction 단계에서 직접 학습한다. 이는 구조적으로 분명한 차별점이며, ablation과 여러 benchmark 결과가 그 효과를 뒷받침한다.

또 다른 강점은 piecewise training의 실용성이다. deep structured model은 원래 개념적으로 매력적이지만 inference 비용 때문에 실제 학습이 어렵다. 논문은 이 문제를 우회하기 위해 piecewise objective를 도입해 학습을 tractable하게 만든다. 이 점은 단지 성능 향상뿐 아니라, "이런 구조를 실제로 학습 가능한 시스템으로 만들었다"는 engineering contribution으로도 중요하다.

배경 문맥을 위한 multi-scale input과 sliding pyramid pooling 역시 강한 구성 요소다. 특히 NYUDv2 ablation에서 이 부분만으로도 큰 성능 향상이 나타난다. 다시 말해 이 논문의 성능은 pairwise CRF 하나만의 효과가 아니라, background context를 잘 담는 feature design과 결합되었기 때문에 나온 결과다.

한계도 분명하다. 첫째, piecewise training은 전역 likelihood의 근사이지 정확한 maximum likelihood 학습이 아니다. 따라서 학습 시 각 potential을 독립적으로 정규화하면, 전역적 상호작용을 충분히 반영하지 못할 가능성이 있다. 논문은 이 방법이 실용적이라고 설득하지만, 근사로 인해 어느 정도 최적성이 희생되는지는 별도로 정량 분석하지 않았다.

둘째, 최종 예측이 여전히 2단계 구조에 의존한다. coarse prediction은 contextual CRF로 만들지만, 최종 high-resolution 결과는 bilinear upsampling과 dense CRF post-processing에 의존한다. 논문도 스스로 인정하듯, deconvolution network나 skip connection, coarse-to-fine refinement 같은 더 정교한 high-resolution prediction 방법을 결합하면 더 좋아질 수 있다. 즉, 이 논문 자체의 강점은 coarse semantic reasoning에 있고, fine boundary localization은 상대적으로 외부 기법에 의존한다.

셋째, pairwise relation은 `surrounding`과 `above/below` 두 종류로 제한되어 있다. 이는 설계가 명확하다는 장점이 있지만, 문맥 관계의 종류를 수동으로 정해두었다는 뜻이기도 하다. 좌우 관계, 장거리 관계, object-level relation 등 더 다양한 구조를 자동으로 학습하는 방향과 비교하면 표현력의 한계가 있을 수 있다. 다만 이 점은 논문이 명시적으로 제시한 범위 안에서의 설계 선택이지, 본문에 없는 확장 가능성을 단정할 수는 없다.

넷째, 계산 비용 측면의 정량 비교가 본문 발췌에는 충분히 제시되어 있지 않다. piecewise training이 효율적이라고 설명하지만, 실제 학습 시간, 메모리 사용량, inference 속도에서 기존 방법 대비 어느 정도 절감되는지는 여기 제공된 텍스트만으로는 구체적으로 확인할 수 없다. 따라서 "효율적"이라는 표현은 방법론적 구조상 타당하지만, 정확한 비용 절감 폭은 본문에 명시된 범위 이상으로 해석하면 안 된다.

종합하면, 이 논문은 구조적 문맥 모델링을 segmentation에 효과적으로 끌어들인 강한 작업이지만, 학습은 근사적이고 최종 해상도 복원은 별도 refinement 모듈에 의존한다는 점에서 완전히 end-to-end한 고해상도 structured prediction 시스템이라고 보기는 어렵다.

## 6. 결론

이 논문은 semantic segmentation에서 문맥 정보를 더 잘 사용하기 위해, CNN 기반 unary potential과 CNN 기반 general pairwise potential을 갖는 deep CRF를 제안했다. 특히 patch-patch context를 explicit하게 모델링하고, patch-background context를 multi-scale input과 sliding pyramid pooling으로 포착한 것이 핵심이다. 학습 측면에서는 piecewise training을 도입해 반복적인 전역 inference 없이도 deep structured model을 효율적으로 최적화할 수 있게 했다.

실험적으로는 NYUDv2, PASCAL VOC 2012, PASCAL-Context, SIFT-flow에서 모두 매우 강한 결과를 보였고, 특히 PASCAL VOC 2012에서는 당시 최고 수준인 78.0 IoU를 보고했다. 따라서 이 연구의 주요 기여는 단순 boundary refinement를 넘어서, semantic relation 자체를 coarse prediction 단계에 통합했다는 데 있다.

향후 연구 관점에서도 의미가 크다. 이 논문은 deep network와 structured model의 결합이 단순 smoothing 수준에 머무르지 않고, richer semantic relation learning으로 확장될 수 있음을 보여준다. 실제 응용에서는 scene parsing, medical image segmentation, remote sensing segmentation처럼 문맥 관계가 중요한 문제에 유용할 가능성이 크다. 또한 이후의 attention, graph reasoning, transformer 기반 segmentation 방법들을 생각해 보면, 이 논문은 "패치 간 관계를 명시적으로 모델링한다"는 흐름의 초기이자 중요한 사례로 볼 수 있다.
