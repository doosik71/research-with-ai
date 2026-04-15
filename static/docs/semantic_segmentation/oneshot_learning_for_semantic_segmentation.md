# One-Shot Learning for Semantic Segmentation

- **저자**: Amirreza Shaban, Shray Bansal, Zhen Liu, Irfan Essa, Byron Boots
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1709.03410

## 1. 논문 개요

이 논문은 단 한 장, 혹은 매우 적은 수의 픽셀 단위 주석 이미지만 보고도 새로운 semantic class에 대한 segmentation mask를 예측하는 문제를 다룬다. 저자들은 이를 **one-shot semantic image segmentation**으로 정의한다. 일반적인 semantic segmentation은 학습 시점과 테스트 시점에 등장하는 클래스가 동일하다는 전제를 두지만, 이 논문은 그 전제를 깨고, 학습 때 보지 못한 새로운 클래스에 대해서도 단 하나의 support example만으로 분할해야 한다는 훨씬 더 어려운 설정을 제시한다.

연구 문제는 명확하다. support set $S=\{(I_i^s, Y_i^s(l))\}_{i=1}^k$가 주어졌을 때, query image $I^q$ 안에서 클래스 $l$에 해당하는 모든 픽셀을 찾아 binary mask를 예측하는 함수 $f(I^q, S)$를 학습하는 것이다. 여기서 중요한 점은 테스트 클래스 집합 $L_{test}$와 학습 클래스 집합 $L_{train}$이 겹치지 않는다는 것이다. 즉, 모델은 새로운 클래스를 직접 supervised training으로 본 적이 없다.

이 문제는 실제로 매우 중요하다. 픽셀 단위 annotation은 이미지 분류보다 훨씬 비싸고 시간이 오래 걸리기 때문에, 새로운 클래스를 segmentation하려면 보통 많은 수의 정답 mask가 필요하다. 만약 단 한 장의 annotated image만으로도 새로운 객체를 분할할 수 있다면, 데이터 수집 비용을 크게 줄일 수 있고, 로봇 비전, 의료영상, 산업 검수, 희귀 객체 인식처럼 라벨이 희소한 환경에서 매우 유용하다. 논문은 few-shot learning의 아이디어를 dense prediction 문제로 확장했다는 점에서 의미가 크다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **support image로부터 바로 segmentation model의 일부 파라미터를 생성하는 two-branch architecture**를 학습하는 것이다. 즉, support example를 단순히 feature matching의 기준점으로 쓰는 것이 아니라, 새로운 클래스에 맞는 classifier parameter를 생성하는 입력으로 사용한다.

구체적으로 한 branch는 support image와 그 mask를 받아서 클래스별 분류기 파라미터 $\{w,b\}$를 생성한다. 다른 branch는 query image에서 dense feature map을 뽑고, support branch가 만든 $\{w,b\}$를 이용해 각 위치의 feature를 binary classification한다. 그러면 query image의 각 pixel 또는 feature location이 target class인지 아닌지를 판별할 수 있다.

이 아이디어의 차별점은 다음과 같다.

첫째, 단순 fine-tuning과 다르다. 새로운 클래스가 들어올 때 segmentation network를 여러 step의 SGD로 다시 적응시키는 대신, support image를 한 번 forward pass하는 것만으로 필요한 classifier를 생성한다. 그래서 훨씬 빠르고, overfitting 위험도 줄어든다.

둘째, metric learning 기반 dense matching과도 다르다. Siamese 방식은 query의 각 픽셀을 support의 많은 픽셀과 비교해야 하므로 dense prediction 환경에서 계산량과 메모리 비용이 급격히 커진다. 반면 이 논문은 support에서 직접 classifier parameter를 만들기 때문에 query 쪽은 일반적인 FCN처럼 처리할 수 있다.

셋째, 이 구조는 meta-learning 관점에서 해석할 수 있다. 모델은 “새로운 클래스를 어떻게 표현할 것인가”를 support example로부터 학습하고, 그 표현을 classifier 형태로 즉시 적용한다. 즉, support set을 이용해 직접 분류기를 생성하는 **feed-forward one-shot learner**를 segmentation으로 확장한 형태다.

## 3. 상세 방법 설명

전체 구조는 두 개의 branch로 구성된다. 첫 번째는 **conditioning branch**, 두 번째는 **segmentation branch**이다.

conditioning branch는 support pair $S=(I^s, Y^s(l))$를 입력받아 dynamic parameter를 출력한다. 논문은 이를 다음과 같이 쓴다.

$$
w, b = g_\eta(S)
$$

여기서 $g_\eta$는 학습 가능한 함수이며, support image와 해당 클래스 mask를 보고 query image 분할에 사용할 logistic regression layer의 파라미터를 만든다.

segmentation branch는 query image $I^q$를 받아 dense feature volume을 만든다.

$$
F^q = \phi_\zeta(I^q)
$$

여기서 $F^q_{mn}$은 spatial location $(m,n)$의 feature vector이다. 이후 conditioning branch가 만든 $w,b$를 이용해 각 위치를 binary classification한다.

$$
\hat{M}_{mn}^q = \sigma(w^\top F_{mn}^q + b)
$$

여기서 $\sigma(\cdot)$는 sigmoid 함수이고, $\hat{M}_{mn}^q$는 query image의 해당 위치가 target class일 확률이다. 쉽게 말하면, query image에서 추출한 각 pixel-level feature에 대해 support example로부터 생성된 1x1 convolution classifier를 적용하는 셈이다. 최종적으로 이 low-resolution mask를 bilinear interpolation으로 원래 해상도로 upsampling하고, threshold 0.5를 적용해 binary mask를 만든다.

### conditioning branch

저자들은 $g_\eta$를 위해 VGG-16 기반 구조를 수정해서 사용한다. 여기서 중요한 설계는 **masking**과 **weight hashing**이다.

먼저 masking이다. support input을 단순히 4채널 이미지-마스크 쌍으로 넣지 않고, 원본 image를 target mask로 가려서 target object만 남긴 형태로 네트워크에 넣는다. 논문은 그 이유를 두 가지 경험적 관찰로 설명한다. 하나는 mask를 함께 주더라도 네트워크가 이미지 내 가장 큰 객체에 편향되는 경향이 있었다는 점이고, 다른 하나는 background 정보를 포함하면 생성되는 $\{w,b\}$의 분산이 커져 학습이 잘 수렴하지 않았다는 점이다. 즉, support example에서 target object만 최대한 분리해서 보여주는 것이 classifier generation에 유리했다.

다음은 weight hashing이다. VGG 마지막 층 출력은 1000차원인데, query branch의 conv-fc7 feature는 4096차원이고 bias까지 포함하면 $\{w,b\}$는 총 4097차원이 필요하다. 이 차원 변환을 일반 fully connected layer로 하면 파라미터 수가 커져 overfitting 위험이 높아진다. 이를 줄이기 위해 논문은 hashing trick 기반의 고정 변환을 사용한다.

보충자료에 따르면 입력 $x \in \mathbb{R}^m$를 출력 $\theta \in \mathbb{R}^d$로 확장할 때, 각 출력 좌표는 임의 hash function $\kappa(i)$와 sign function $\zeta(i)$를 이용해 다음처럼 정의된다.

$$
\theta(i) = x(p)\zeta(i), \quad p=\kappa(i)
$$

즉, 입력 벡터의 일부 계수를 여러 위치에 복제하되, 부호를 무작위로 뒤집어 covariance를 줄인다. 실제 구현은 고정 가중치 fully connected layer와 동일하다.

$$
W(i,j)=\zeta(i)\delta_j(\kappa(i))
$$

여기서 $\delta_j(\cdot)$는 discrete Dirac delta 역할을 한다. 이 층의 가중치는 학습되지 않고 고정된다. 논문은 이를 통해 output variance를 줄이고, 큰 fully connected layer가 초래할 overfitting을 피했다고 설명한다.

### segmentation branch

query image의 feature extractor $\phi_\zeta$는 FCN-32s 구조를 기반으로 하며, 마지막 prediction layer를 제거한 형태다. conv-fc7에서 4096채널 feature volume을 추출하고, 여기에 support에서 생성한 logistic classifier를 적용한다. 논문 후반부에서는 stride 8의 dilated-FCN도 실험하지만, 기본 방법은 FCN-32s 저해상도 설정이다.

### 학습 절차

학습은 직접 one-shot task를 시뮬레이션하면서 진행한다. 매 iteration마다 다음 순서로 sample을 만든다.

먼저 학습 데이터 $D_{train}$에서 하나의 이미지-정답쌍 $(I^q, Y^q)$를 뽑는다. 그 이미지에 등장하는 클래스들 중 하나 $l \in L_{train}$를 균등하게 선택하여, 해당 클래스의 binary mask $Y^q(l)$를 만든다. 그다음 같은 클래스 $l$이 존재하는 다른 이미지를 support pair로 뽑아 $S$를 구성한다. 이렇게 하면 “support image 한 장을 보고 query image 안의 같은 클래스를 segmentation”하는 one-shot episode가 만들어진다.

학습 목표는 query mask의 로그우도를 최대화하는 것이다.

$$
L(\eta,\zeta)=
\mathbb{E}_{S,I^q,M^q \sim D_{train}}
\left[
\sum_{m,n}\log p_{\eta,\zeta}(M_{mn}^q \mid I^q, S)
\right]
$$

실질적으로는 query의 각 위치에 대한 binary classification log-likelihood를 최적화하는 것이다. 논문은 SGD를 사용했고, learning rate는 $10^{-10}$, momentum은 $0.99$, batch size는 1이다. 또한 conditioning branch의 VGG 쪽은 더 빨리 overfit하기 때문에 그쪽 learning rate multiplier를 0.1로 낮췄다. 총 60k iteration에서 학습을 종료했다.

이 설정에서 주의할 점은 learning rate가 매우 작다는 것이다. 논문은 그 이유를 자세히 분석하지는 않았지만, 두 branch가 결합된 구조와 VGG 기반 파라미터 생성 branch의 민감도를 고려한 경험적 선택으로 보인다. 여기서 더 자세한 optimizer 튜닝 근거는 논문 본문에 없다.

### k-shot 확장

$k$-shot일 때 support set에는 $k$개의 labeled image가 들어 있다. 저자들은 이를 복잡하게 joint aggregation하지 않고, 각 support image를 독립적으로 conditioning branch에 넣어 $k$개의 classifier $\{w_i,b_i\}_{i=1}^k$를 만든다. 각각 query image에 대해 binary mask를 예측한 뒤, 최종 결과는 픽셀 단위 logical OR로 합친다.

이 설계의 해석은 논문에서 분명히 제시된다. 각 one-shot classifier는 precision은 높지만 recall이 낮은 경향이 있는데, 이는 support image 한 장이 클래스의 다양한 appearance를 충분히 담지 못하기 때문이다. 따라서 여러 support example가 서로 다른 appearance subset을 보완하도록 OR aggregation을 사용한다. 이 방법은 retraining이 필요 없고, support image 수가 바뀌어도 그대로 적용 가능하다는 장점이 있다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

논문은 새로운 benchmark인 **PASCAL-5$^i$**를 제안한다. 이는 PASCAL VOC 2012의 20개 클래스를 4개 fold로 나누어, 각 fold마다 5개 클래스를 test class로 두고 나머지 15개 클래스를 training class로 사용하는 방식이다.

fold별 test class는 다음과 같다.

- $i=0$: aeroplane, bicycle, bird, boat, bottle
- $i=1$: bus, car, cat, chair, cow
- $i=2$: diningtable, dog, horse, motorbike, person
- $i=3$: potted plant, sheep, sofa, train, tv/monitor

학습셋은 PASCAL VOC와 SDS의 training images에서 $L_{train}$ 클래스 중 하나 이상이 있는 이미지를 사용해 만든다. 이때 $L_{train}$에 속하지 않는 클래스 픽셀은 background로 바꾼다. 테스트셋은 PASCAL VOC validation set에서 $L_{test}$에 대해 같은 방식으로 구성한다. 그리고 논문은 테스트에서 1000개의 episode를 샘플링해 benchmark로 사용한다.

평가 지표는 binary segmentation에 맞춘 class-wise IoU와 그 평균인 meanIoU이다. 특정 클래스 $l$에 대해

$$
IoU_l = \frac{tp_l}{tp_l + fp_l + fn_l}
$$

이며, meanIoU는 이를 클래스 평균한 값이다. 이는 semantic segmentation에서 널리 쓰이는 meanIU를 binary one-vs-background 설정에 맞게 적용한 것이다.

### 비교 대상

비교 baseline은 다음과 같다.

첫째, **Base Classifiers**다. FCN-32s를 16-way segmentation으로 먼저 학습한 뒤, support와 query에서 fc7 dense feature를 뽑고, support feature로 1-NN이나 logistic regression classifier를 학습해 query를 분류한다.

둘째, **Fine-tuning**이다. 테스트 시점마다 support set으로 segmentation network를 다시 fine-tune한다. overfitting과 시간 문제를 줄이기 위해 fc6, fc7, fc8만 조정한다.

셋째, **Co-segmentation by Composition**이다. 이는 weak or unsupervised co-segmentation 계열 방법과의 비교를 위한 baseline이다.

넷째, **Siamese Network for Dense Matching**이다. support와 query의 dense feature를 비교하여 pixel-level matching을 수행하는 방식으로, one-shot image classification의 Siamese network를 segmentation에 맞게 확장했다.

### 정량 결과

1-shot 결과에서 제안 방법은 fold 평균 **40.8 meanIoU**를 기록했다. 주요 baseline 성능은 1-NN 32.6, logistic regression 31.4, fine-tuning 32.6, Siamese 31.4였다. 즉, 저자들이 강조하듯 제안 방법은 strongest baseline 대비 약 **25% 상대 향상**을 보였다. 절대 차이로도 약 8포인트 이상 우세하다.

fold별로 보면 PASCAL-5$^1$에서 55.3으로 특히 높았고, PASCAL-5$^0$와 PASCAL-5$^3$에서도 baseline보다 뚜렷하게 좋았다. 다만 PASCAL-5$^2$에서는 40.9로, 1-NN의 41.7보다 약간 낮다. 즉, 모든 fold에서 일관되게 최고는 아니지만 평균적으로는 가장 좋은 일반화 성능을 보인다.

5-shot 결과에서는 제안 방법이 평균 **43.9 meanIoU**를 기록했다. 1-NN은 40.0, logistic regression은 39.3, co-segmentation은 27.1이었다. 제안 방법은 1-shot보다 3.1포인트 상승했다. 이는 OR aggregation이 recall 보완에 실제로 도움이 되었음을 시사한다.

흥미로운 점은 5-shot에서도 향상 폭이 아주 크지는 않다는 것이다. 논문은 특히 고해상도 dilated-FCN에서는 1-shot 37.0, 5-shot 37.43으로 격차가 더 작았다고 보고한다. 저자들은 그 이유를 training이 1-shot task에 특화되어 있었기 때문이라고 해석한다. 즉, $k$-shot aggregation 자체는 간단히 확장되지만, 학습 자체를 multi-shot episodic training으로 설계한 것은 아니었다.

### 속도와 계산 비용

추론 시간 비교도 이 논문의 중요한 주장 중 하나다. 1-shot에서 제안 방법은 **0.19초**로, logistic regression 0.66초보다 약 3배 이상 빠르다. 5-shot에서는 제안 방법이 **0.21초**, logistic regression이 3.50초로 약 10배 이상 빠르다. fine-tuning과 Siamese는 약 5.5초 수준으로 더 느리다.

이 결과는 논문의 핵심 설계와 직접 연결된다. support로부터 classifier를 한 번의 forward pass로 생성하고, query에 대해서는 FCN feature extraction과 간단한 logistic classification만 수행하기 때문에, 테스트 시 iterative adaptation이 필요한 fine-tuning 계열보다 훨씬 빠르다.

### 사전학습 효과 분석

논문은 ImageNet pretraining이 test class generalization에 얼마나 기여하는지도 실험한다. 이를 위해 PASCAL 클래스와 겹치는 ImageNet 클래스를 제거한 **PASCAL-removed-ImageNet**(771 classes)로 사전학습한 AlexNet과 원래 1000-class ImageNet으로 사전학습한 AlexNet을 비교한다.

결과적으로 AlexNet-1000은 초기 수렴이 더 빠르지만, 학습이 충분히 진행된 뒤에는 AlexNet-771이 거의 동등한 성능을 보인다. 즉, test class와 겹치는 weak label이 pretraining에 없어도 제안 방법은 여전히 unseen category로 일반화할 수 있다. 저자들은 이것을 meta-learning 구조의 장점으로 해석한다.

반대로 logistic regression baseline은 1000-class pretraining이 771-class pretraining보다 더 좋았다. 이는 단순 feature-based baseline은 weak supervision overlap에 더 의존하고, meta-learning 방식은 그 의존성이 적다는 점을 보여준다.

### 정성 결과

논문과 보충자료의 qualitative figure들은 제안 방법이 support example에 따라 실제로 다른 클래스에 conditioning된다는 점을 보여준다. 같은 query image에 대해 cow support를 주면 cow를, car support를 주면 car를 예측하는 사례를 제시한다. 또한 5-shot일 때 1-shot보다 예측 mask가 더 완전해지는 예시도 제공한다.

다만 본문에 정성 결과의 실패 사례 분석은 충분히 포함되어 있지 않다. 그래서 어떤 상황에서 자주 실패하는지, 예를 들어 occlusion, small object, cluttered background, multiple instances에 약한지는 정성적으로 암시만 있을 뿐 체계적으로 분석되지는 않는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 매우 잘 맞물려 있다는 점이다. one-shot segmentation은 새로운 클래스를 위한 classifier를 빠르게 만들어야 하는데, 저자들은 이를 support-conditioned dynamic parameter prediction으로 정면 돌파했다. 이 구조는 dense prediction의 계산량 문제를 피해 가면서도, 새로운 클래스에 대한 적응을 feed-forward 방식으로 처리한다. 그래서 정확도뿐 아니라 추론 속도에서도 이점을 얻는다.

또 다른 강점은 **benchmark 제안**이다. PASCAL-5$^i$는 이후 few-shot segmentation 연구에서 널리 사용되는 표준 설정이 되었다. 단순히 방법만 제안한 것이 아니라, 이 문제를 재현 가능하게 측정하는 분할 규칙과 평가 프로토콜을 함께 제공했다는 점에서 영향력이 크다.

또한 논문은 few-shot learning에 반드시 매우 많은 class가 필요하다는 통념에 도전한다. PASCAL은 클래스가 20개뿐인데도, 학습 클래스와 테스트 클래스를 분리한 episodic training만으로 꽤 의미 있는 일반화가 가능하다는 결과를 보여준다. 이 점은 segmentation처럼 강한 annotation이 비싼 문제에서 meta-learning이 현실적으로도 가능하다는 근거가 된다.

반면 한계도 분명하다. 첫째, classifier head가 사실상 pixel-wise logistic regression에 가깝기 때문에 표현력이 제한적이다. query feature extractor는 고정된 feature를 만들고, support는 그 위에서 선형 결정경계를 생성하는 역할만 한다. 따라서 매우 복잡한 class variation이나 context-dependent appearance를 모델링하는 데에는 한계가 있을 수 있다.

둘째, $k$-shot 확장이 매우 단순하다. 여러 support example 사이의 관계를 jointly reason하지 않고, 각 결과를 OR로 합친다. 이는 구현은 쉬우나 support example 간 상호보완 정보를 학습적으로 통합하지 못한다. 논문도 이 부분을 명확히 인정하지는 않지만, 5-shot 성능 향상이 제한적인 점은 이러한 단순 aggregation의 한계를 시사한다.

셋째, 학습 설정과 하이퍼파라미터가 다소 경험적이다. 예를 들어 learning rate $10^{-10}$, branch별 learning rate multiplier, masking 설계 등은 실험적으로 정해졌지만 왜 이 값들이 구조적으로 적절한지에 대한 분석은 제한적이다. 실제 재현이나 확장 연구에서는 이 부분이 민감할 수 있다.

넷째, 평가 데이터셋 규모가 작다. PASCAL-5$^i$는 중요한 benchmark이지만 클래스 수가 적고 장면 다양성도 제한적이다. 따라서 이 결과만으로 매우 다양한 real-world category에 대한 강한 일반화를 보장한다고 보기는 어렵다. 논문도 larger-scale few-shot segmentation에 대한 실험은 제공하지 않는다.

다섯째, 논문은 unseen class object가 training image 안에 등장할 수 있고 그것이 background로 처리된다고 설명한다. 이는 실제 annotation 설정을 반영하는 합리적 선택이지만, 동시에 background label 안에 future test object가 섞인다는 뜻이기도 하다. 이런 weak negative signal이 학습에 어떤 영향을 주는지에 대한 정량 분석은 제공되지 않는다.

비판적으로 보면, 이 논문은 one-shot segmentation의 초기 방향을 제시하는 데 매우 성공적이지만, support-query interaction을 더 풍부하게 만드는 방향, multi-scale matching, prototype aggregation, decoder refinement 같은 후속 발전의 여지를 많이 남긴다. 즉, 완성형이라기보다 매우 강한 출발점에 가깝다.

## 6. 결론

이 논문은 one-shot semantic segmentation이라는 새로운 문제를 명확히 정의하고, support image로부터 query segmentation classifier의 파라미터를 생성하는 two-branch meta-learning 구조를 제안했다. 방법의 핵심은 support-conditioned dynamic parameter generation과 query FCN feature 위의 pixel-wise logistic classification이며, 이를 통해 unseen class에 대해서도 단 한 장의 strongly annotated image만으로 segmentation이 가능함을 보였다.

실험적으로는 PASCAL-5$^i$ benchmark에서 baseline들인 1-NN, logistic regression, fine-tuning, Siamese network보다 더 높은 meanIoU를 기록했고, 동시에 추론 속도도 크게 개선했다. 특히 weak label overlap이 없는 pretraining에서도 성능이 유지된다는 분석은, 단순 transfer learning이 아니라 meta-learning 자체가 unseen class generalization에 실질적으로 기여함을 보여준다.

실제 적용 측면에서 이 연구는 annotation이 희소한 환경에서 새로운 객체를 빠르게 분할해야 하는 문제에 중요한 가능성을 제시한다. 향후 연구에서는 더 강력한 support aggregation, 고해상도 refinement, transformer 기반 support-query interaction, 더 큰 benchmark로의 확장 등이 자연스러운 발전 방향이 될 것이다. 이 논문은 few-shot segmentation 연구의 초기 기반을 만든 대표적인 작업으로 평가할 수 있다.
