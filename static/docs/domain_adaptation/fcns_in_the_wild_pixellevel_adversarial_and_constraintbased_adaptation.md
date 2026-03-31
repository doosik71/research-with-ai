# FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation

* **저자**: Judy Hoffman, Dequan Wang, Fisher Yu, Trevor Darrell
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1612.02649](https://arxiv.org/abs/1612.02649)

## 1. 논문 개요

이 논문은 semantic segmentation에서의 **unsupervised domain adaptation** 문제를 다룬다. 구체적으로는 source domain에서는 픽셀 단위 정답 라벨이 존재하지만, target domain에는 라벨이 전혀 없는 상황에서, source에서 학습한 FCN 기반 segmentation 모델을 target에 잘 작동하도록 적응시키는 방법을 제안한다. 저자들의 핵심 문제의식은 매우 분명하다. semantic segmentation 모델은 같은 분포의 데이터에서는 높은 성능을 내지만, 도시가 바뀌거나 계절이 달라지거나, synthetic 이미지에서 real 이미지로 넘어가는 경우처럼 사람이 보기에는 비교적 비슷해 보이는 변화에도 성능이 크게 하락한다.

이 문제는 자율주행, 로보틱스, 지도화 같은 실제 응용에서 특히 중요하다. 픽셀 단위 annotation은 매우 비싸고 시간이 오래 걸리기 때문에, 새로운 도시나 날씨, 새로운 센서 환경이 등장할 때마다 dense label을 다시 만드는 것은 현실적으로 어렵다. 따라서 라벨이 있는 source domain의 정보를 라벨이 없는 target domain으로 이전하는 것은 실용적인 가치가 매우 크다.

논문의 주장에 따르면, 기존 domain adaptation 연구는 주로 image classification에 집중되어 있었고, semantic segmentation을 위한 본격적인 adaptation 방법은 거의 없었다. 이 논문은 그 공백을 메우며, semantic segmentation을 위한 **최초의 unsupervised domain adaptation 방법**을 제안한다고 위치를 잡고 있다. 또한 단순히 하나의 adaptation 전략만 쓰는 것이 아니라, **global alignment**와 **category-specific adaptation**을 결합해 전체 분포 차이와 클래스별 편향을 동시에 줄이려는 점이 핵심이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 semantic segmentation에서 domain shift가 하나의 원인으로만 생기지 않는다는 점에 있다. 저자들은 이를 크게 두 가지로 나눈다. 첫째는 source와 target 사이의 전반적인 appearance 차이, 즉 **global distribution shift**이다. 예를 들어 synthetic와 real 이미지 간 차이, 혹은 도시 전체의 조명/색감/질감 차이 같은 것이다. 둘째는 특정 클래스마다 나타나는 **category-specific shift**이다. 예를 들어 어떤 도시에서는 차의 분포와 크기가 다르고, 교통 표지판의 모양이나 빈도가 달라질 수 있다.

이 문제를 해결하기 위해 논문은 두 단계 성격의 적응 전략을 결합한다.

첫 번째는 **Fully Convolutional Domain Adversarial Learning**이다. 기존 adversarial domain adaptation은 주로 이미지 하나를 하나의 instance로 취급했는데, semantic segmentation에서는 그런 방식이 픽셀 수준 구조를 너무 많이 잃어버린다. 그래서 저자들은 FCN의 마지막 feature map에서 각 spatial unit이 담당하는 receptive field 영역을 하나의 adaptation 단위로 간주한다. 즉, image-level이 아니라 **region/pixel-level representation**을 source와 target 사이에서 구분 못 하도록 만드는 것이다.

두 번째는 **Constraint-based Category Adaptation**이다. target에는 라벨이 없지만, source 데이터에서 “클래스별로 이미지 안에서 대략 어느 정도 면적을 차지하는가”라는 통계를 계산할 수 있다. 예를 들어 road는 보통 화면에서 넓은 비율을 차지하고, traffic sign은 아주 작은 비율을 차지한다. 저자들은 이런 source의 spatial layout 통계를 target에 제약조건으로 옮겨와, target prediction이 너무 비현실적인 class coverage를 가지지 않도록 만든다. 이것은 weakly supervised segmentation의 constrained multiple instance learning 아이디어를 adaptation에 맞게 일반화한 것이다.

따라서 이 논문의 차별점은 다음과 같이 요약할 수 있다. 첫째, semantic segmentation에 맞는 **fully convolutional adversarial adaptation**을 제안했다는 점이다. 둘째, 단순한 global feature alignment만으로 끝내지 않고, source의 클래스별 장면 구성 통계를 target으로 옮기는 **class-aware constraint**를 추가했다는 점이다. 셋째, target annotation이 전혀 없는 완전 unsupervised setting에서 이를 수행한다는 점이다.

## 3. 상세 방법 설명

논문의 전체 학습 목표는 세 개의 손실 함수로 구성된다. 하나는 source supervision을 유지하기 위한 segmentation loss이고, 하나는 global domain alignment를 위한 adversarial loss이며, 마지막 하나는 target에서 category-specific adaptation을 위한 multiple instance constraint loss이다. 전체 목적함수는 다음과 같다.

$$
\mathcal{L}(I_{\mathcal{S}}, L_{\mathcal{S}}, I_{\mathcal{T}}) = \mathcal{L}_{seg}(I_{\mathcal{S}}, L_{\mathcal{S}}) + \mathcal{L}_{da}(I_{\mathcal{S}}, I_{\mathcal{T}}) + \mathcal{L}_{mi}(I_{\mathcal{T}}, \mathcal{P}_{L_{\mathcal{S}}})
$$

여기서 $\mathcal{L}_{seg}$는 source의 픽셀 라벨을 이용한 일반적인 semantic segmentation 손실이다. 이 항은 adaptation 과정에서도 source task 성능이 완전히 무너지지 않도록 하는 역할을 한다. $\mathcal{L}_{da}$는 source와 target의 representation을 전역적으로 맞추는 항이고, $\mathcal{L}_{mi}$는 target prediction이 source에서 관찰된 클래스별 공간 통계를 따르도록 유도하는 항이다.

기본 backbone은 dilated convolution 기반의 front-end FCN이며, VGG-16 구조를 기반으로 한다. 마지막 fully connected layer들을 convolution layer인 $fc6$, $fc7$, $fc8$로 변환하고, 마지막에 bilinear upsampling을 통해 입력과 동일한 해상도의 segmentation map을 만든다. 논문은 이 backbone 자체를 새로 제안하는 것이 아니라, 이 위에 adaptation objective를 얹는 방식이다.

### 3.1 Global Domain Alignment

저자들은 segmentation에서 domain alignment를 할 때, 이미지 전체를 한 덩어리로 보는 것은 적절하지 않다고 본다. segmentation은 픽셀마다 예측을 해야 하므로, adaptation도 pixel/region 수준에서 일어나야 한다. 그래서 마지막 prediction 직전의 feature map, 즉 $\phi_{\ell-1}(\theta, I)$의 각 spatial 위치 $(h,w)$에 해당하는 representation을 adaptation의 기본 단위로 사용한다.

domain classifier는 이 지역 표현이 source에서 왔는지 target에서 왔는지 판별하도록 학습된다. 표현을 다음과 같이 두자.

$$R_{hw}^{\mathcal{S}} = \phi_{\ell-1}(\theta, I_{\mathcal{S}})_{hw}$$
$$R_{hw}^{\mathcal{T}} = \phi_{\ell-1}(\theta, I_{\mathcal{T}})_{hw}$$

그리고 domain classifier의 출력 $p_{\theta_D}(x)$는 해당 representation이 source domain일 확률로 해석된다. 그러면 domain classifier의 loss는 source는 1로, target은 0으로 맞추는 이진 분류 loss가 된다.

$$
\mathcal{L}_{D} = -\sum_{I_{\mathcal{S}} \in \mathcal{S}} \sum_{h \in H} \sum_{w \in W} \log \left( p_{\theta_D}(R_{hw}^{\mathcal{S}}) \right) - \sum_{I_{\mathcal{T}} \in \mathcal{T}} \sum_{h \in H} \sum_{w \in W} \log \left( 1 - p_{\theta_D}(R_{hw}^{\mathcal{T}}) \right)
$$

이 식은 domain classifier 자체를 잘 학습시키는 목적이다. 반대로 feature extractor 쪽은 domain classifier를 헷갈리게 만들어야 하므로, source를 target처럼 보이게 하고 target을 source처럼 보이게 하는 inverse domain loss를 정의한다.

$$
\mathcal{L}_{Dinv} = -\sum_{I_{\mathcal{S}} \in \mathcal{S}} \sum_{h \in H} \sum_{w \in W} \log \left( 1 - p_{\theta_D}(R_{hw}^{\mathcal{S}}) \right) - \sum_{I_{\mathcal{T}} \in \mathcal{T}} \sum_{h \in H} \sum_{w \in W} \log \left( p_{\theta_D}(R_{hw}^{\mathcal{T}}) \right)
$$

그 다음 alternating optimization을 수행한다. 먼저 domain classifier 파라미터 $\theta_D$는 $\mathcal{L}_D$를 최소화하도록 업데이트하고,

$$
\min_{\theta_D} \mathcal{L}_D
$$

feature representation 파라미터 $\theta$는 다음 식을 최소화한다.

$$
\min_{\theta} \frac{1}{2}\left[\mathcal{L}_{D} + \mathcal{L}_{Dinv}\right]
$$

직관적으로 보면, domain classifier는 source와 target을 최대한 잘 구분하려 하고, feature extractor는 그 구분이 어렵도록 representation을 바꾼다. 결국 segmentation에 중요한 region-level feature가 domain invariant해지도록 만드는 것이다. classification adaptation과 비슷한 adversarial idea를 쓰지만, semantic segmentation에 맞게 **fully convolutional spatial representation**에 적용했다는 점이 중요하다.

### 3.2 Category Specific Adaptation

global alignment만으로는 충분하지 않다고 저자들은 본다. 예를 들어 전체적인 색감이나 texture는 맞춰졌더라도, 특정 클래스의 상대적 크기나 위치 분포는 여전히 domain마다 다를 수 있다. 그래서 저자들은 weakly supervised segmentation에서 쓰이던 constrained multiple instance learning을 domain adaptation용으로 확장한다.

핵심은 source domain의 label map으로부터 클래스별 면적 통계를 추출하는 것이다. source에서 클래스 $c$가 등장하는 이미지들에 대해, 그 클래스가 이미지 전체 픽셀 중 몇 퍼센트를 차지하는지 계산한다. 이 비율들의 분포로부터 세 개의 통계를 얻는다.

* $\alpha_c$: 하위 10% 경계
* $\delta_c$: 평균값
* $\gamma_c$: 상위 10% 경계

이 값들은 target 이미지에서 클래스 $c$가 등장했다면, 대략 어느 정도 픽셀 수를 가져야 하는지에 대한 prior가 된다. 따라서 target prediction map $p = \arg\max \phi(\theta, I_{\mathcal{T}})$에 대해 다음 제약을 둔다.

$$
\delta_c \leq \sum_{h,w} p_{hw}(c) \leq \gamma_c
$$

논문의 설명상, 실제 최적화에서는 lower bound에는 slack을 허용하지만 upper bound에는 slack을 두지 않는다. 이유는 어떤 클래스가 예상보다 작게 나오는 것은 예외적인 경우로 허용할 수 있지만, 하나의 클래스가 이미지 대부분을 과도하게 차지하는 현상은 segmentation 붕괴를 초래할 수 있기 때문이다.

여기서 중요한 점은 이 방식이 object class뿐 아니라 road, sky, vegetation 같은 stuff class에도 적용된다는 것이다. 즉, 단순히 객체 존재 여부가 아니라 **장면의 공간적 레이아웃 통계**를 target 쪽으로 옮기는 것이다.

그러나 이 constrained loss는 원래 image-level label이 있을 때 더 직접적으로 사용할 수 있다. 이 논문에서는 target에 image-level label조차 없으므로, 먼저 각 클래스가 target 이미지에 존재하는지를 pseudo image-level label 형태로 추정해야 한다. 이를 위해 현재 모델의 prediction을 사용한다. target 이미지에 대해 클래스 $c$로 예측된 픽셀 비율을 다음처럼 계산한다.

$$
d_c = \frac{1}{H \cdot W} \sum_{h \in H} \sum_{w \in W} (p_{hw} = c)
$$

그리고 다음 조건을 만족하면 클래스 $c$가 그 이미지에 존재한다고 간주한다.

$$
d_c > 0.1 \cdot \alpha_c
$$

즉, 현재 prediction에서 클래스 $c$가 차지하는 비율이, source에서 그 클래스가 실제로 등장할 때의 최소 규모의 10% 이상이면 “이 클래스가 존재한다”고 판단한다. 그런 뒤 존재한다고 판단된 클래스들에 대해 위의 size constraint를 걸어 target prediction을 refine한다.

저자들은 또 하나의 실용적인 조정도 추가한다. 큰 면적을 차지하는 클래스들만 손실을 지배해버리면 작은 클래스는 학습에 기여하기 어렵다. 그래서 source 분포의 하위 10% 경계 $\alpha_c$가 0.1보다 큰 클래스, 즉 보통 이미지에서 꽤 큰 비율을 차지하는 클래스들은 gradient를 0.1배로 down-weight한다. 이는 일종의 re-balancing으로 볼 수 있다.

정리하면, category-specific adaptation은 다음 흐름으로 이해할 수 있다. 먼저 현재 target prediction으로부터 어떤 클래스들이 이미지에 존재하는지 추정한다. 다음으로 source에서 얻은 클래스별 크기 통계를 이용해 타당한 pixel coverage 범위를 정한다. 마지막으로 이러한 제약을 만족하도록 target prediction을 다시 학습시킨다. 이 과정은 source의 dense supervision이 없는 target에서도 class-aware regularization을 제공한다.

## 4. 실험 및 결과

논문은 세 가지 서로 다른 강도의 domain shift를 다룬다. 첫째는 **large shift**인 synthetic-to-real adaptation이다. 둘째는 **medium shift**인 season-to-season adaptation이다. 셋째는 **small shift**인 city-to-city adaptation이다. 모든 실험에서 baseline은 dilated FCN front-end이며, 평가 지표는 **IoU**와 평균 값인 **mIoU**이다.

### 4.1 데이터셋과 실험 설정

Cityscapes는 고해상도 urban scene dataset이며, 34개 카테고리를 포함하고 train/val/test로 도시 단위 split이 되어 있다. 논문은 이 데이터셋을 real-world cross-city adaptation의 target 또는 evaluation domain으로 사용한다.

SYNTHIA는 synthetic urban scene dataset으로, 계절, 날씨, 조명 조건이 다양하다. season adaptation에서는 SYNTHIA-VIDEO-SEQUENCES의 Summer, Fall, Winter를 서로 다른 domain으로 보고 adaptation을 수행한다. synthetic-to-real에서는 SYNTHIA-RAND-CITYSCAPES를 source로 사용한다.

GTA5는 Grand Theft Auto V 게임 엔진에서 생성한 대규모 synthetic dataset이다. Cityscapes와 호환되는 라벨을 가지며, synthetic-to-real adaptation에서 source domain으로 사용된다.

또한 저자들은 BDDS라는 새로운 대규모 dash-cam dataset을 소개한다. 논문 시점에는 정량 평가가 완전히 준비되지는 않았고, 정성적 결과 위주로 adaptation 가능성을 보여준다.

### 4.2 Large Shift: Synthetic to Real

가장 큰 shift는 synthetic에서 real로 가는 경우다. 저자들은 GTA5 $\rightarrow$ Cityscapes, SYNTHIA $\rightarrow$ Cityscapes를 실험했다.

#### GTA5 $\rightarrow$ Cityscapes

baseline인 Dilation Frontend의 mIoU는 21.1이다. global alignment만 적용한 경우 25.5로 상승하며, global alignment와 category adaptation을 모두 적용하면 27.1까지 올라간다. 즉 전체적으로 **6.0%p의 절대적 mIoU 개선**이 있다.

클래스별로 보면 road가 31.9에서 70.4로 크게 뛰고, sidewalk는 18.9에서 32.4, person은 36.0에서 44.1, car는 67.1에서 70.4로 개선된다. vegetation과 terrain 같은 scene layout과 관련된 클래스도 좋아진다. 반면 train, bike, motorbike 같이 원래 성능이 매우 낮거나 sparse한 클래스는 개선 폭이 제한적이다.

논문은 이 결과를 바탕으로 large shift 상황에서는 global alignment의 기여가 특히 크다고 해석한다. 실제로 21.1에서 25.5로 올라가는 대부분의 향상은 GA에서 오고, CA는 그 위에 추가 이득을 더하는 형태다. 이는 synthetic와 real 간 차이가 우선 전체적인 appearance/feature 분포 차이로 많이 설명된다는 뜻으로 볼 수 있다.

#### SYNTHIA $\rightarrow$ Cityscapes

baseline mIoU는 14.7이고, GA only는 16.6, GA+CA는 17.0이다. GTA5보다 전체 수치가 낮은데, 이는 source 데이터 성격이나 클래스 구성, 혹은 원래 모델의 전이 가능성 차이 때문으로 해석할 수 있다. 그래도 adaptation이 일관되게 성능을 올린다는 점은 유지된다.

클래스별로 pole, traffic sign, vegetation, car 등의 개선이 보인다. person은 51.1에서 51.2로 거의 비슷하지만, car는 47.3에서 54.0으로 의미 있게 오른다. 다만 많은 클래스에서 절대 수치가 아직 낮아, 대규모 synthetic-real gap이 완전히 해결되지는 않았음을 보여준다.

### 4.3 Medium Shift: Cross Seasons Adaptation

SYNTHIA 내부에서 Summer, Fall, Winter를 서로 다른 domain으로 보고 적응한 결과가 Table 2에 정리되어 있다. 이 실험은 synthetic 내부이지만 appearance shift가 꽤 강한 예를 제공한다. 예를 들어 Fall에서 Winter로 갈 때 road와 sidewalk가 눈 때문에 크게 달라진다.

논문은 평균적으로 **약 3%p mIoU 향상**을 보고한다. 표의 마지막 평균 행을 보면 Before Adapt의 mIoU가 61.5이고, After Adapt가 64.2이므로 전체 평균에서 분명한 이득이 있다. 특히 Fall $\rightarrow$ Winter는 51.9에서 59.6으로 크게 상승하고, Winter $\rightarrow$ Summer는 59.5에서 62.5, Winter $\rightarrow$ Fall은 59.3에서 62.0으로 오른다. 반면 Summer $\rightarrow$ Fall이나 Summer $\rightarrow$ Winter처럼 이미 비교적 높거나 shift 특성이 다른 경우는 개선이 작거나 거의 없다.

저자들은 13개 클래스 중 12개에서 적응 후 더 높은 mIoU를 얻었다고 설명한다. 예외적으로 car 클래스는 계절 변화에 따른 외형 변화가 거의 없어서 큰 이득이 없다고 해석한다. 이는 논문의 qualitative 분석과도 맞물린다. 도로와 보도는 계절에 따라 눈으로 덮여 appearance가 크게 바뀌지만, 자동차는 비슷하게 렌더링되므로 adaptation의 필요성이 상대적으로 작다는 것이다.

이 실험은 제안 방식이 단순히 synthetic-real 같은 극단적 gap에만 쓰이는 것이 아니라, 더 미묘한 appearance shift에도 유효함을 보여준다.

### 4.4 Small Shift: Cross City Adaptation

Cityscapes train cities를 source로, Cityscapes val cities를 target으로 하는 실험은 상대적으로 작은 domain shift를 다룬다. baseline의 mIoU는 이미 64.0으로 높다. 여기에 GA only를 적용하면 67.6, GA+CA를 적용하면 67.8이 된다.

즉, 총 향상은 **3.8%p** 정도이며, 대부분의 이득은 global alignment에서 나온다. category-specific adaptation은 전체 mIoU 기준으로는 추가 이득이 크지 않다. 그러나 traffic light, rider, train 같은 일부 클래스에서는 세부적인 개선이 있다. 저자들은 이 현상을, train과 val 도시 간 차이가 주로 전체 장면 appearance 차이에서 비롯되고, 클래스별 appearance 차이는 상대적으로 크지 않기 때문이라고 해석한다.

실제로 city-to-city adaptation은 segmentation 자체의 기본 성능이 이미 높아서, 더 큰 향상보다는 object 내부 consistency를 개선하는 형태로 이득이 나타나는 것으로 보인다. 이 관찰은 논문의 두 adaptation 구성요소가 서로 다른 shift 규모에서 다르게 작동한다는 점을 잘 보여준다.

### 4.5 BDDS Adaptation

BDDS에 대해서는 정량적 평가 대신 정성적 결과를 제시한다. Cityscapes로 학습한 모델을 BDDS의 San Francisco 이미지에 그대로 적용하면 noisy prediction이나 잘못된 context labeling이 나타난다. adaptation 후에는 segmentation 결과가 더 깨끗해지고 장면 구조가 더 일관되게 보인다고 저자들은 설명한다.

다만 이 부분은 논문 본문에서도 annotation이 아직 완비되지 않아 정량 평가를 향후 수행할 예정이라고 명시한다. 따라서 BDDS 실험은 새로운 dataset의 필요성과 real-world transfer difficulty를 보여주는 보조적 증거로 이해하는 것이 적절하다. 정량 성능을 주장할 수 있을 정도의 완전한 실험은 아니다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation을 위한 unsupervised domain adaptation 문제를 정면으로 다뤘다는 점이다. 당시 기준으로 classification 중심의 adaptation 연구를 dense prediction으로 확장한 선구적 작업이라는 의미가 있다. 특히 segmentation에서는 픽셀별 예측과 장면 구조가 중요하므로, image-level adaptation을 그대로 가져오면 부족하다. 이 논문은 region-level adversarial alignment라는 설계를 통해 그 문제를 자연스럽게 다루었다.

둘째 강점은 global alignment와 category-specific adaptation을 분리해서 설계했다는 점이다. 많은 논문이 하나의 adaptation loss로 모든 문제를 해결하려 하지만, 이 논문은 전반적 feature 분포 차이와 클래스별 spatial prior 차이를 अलग해서 처리한다. 이 분해가 실험 결과와도 잘 맞는다. large shift에서는 GA의 기여가 크고, 일부 setting에서는 CA가 추가적인 안정화와 클래스별 개선을 제공한다.

셋째 강점은 target annotation이 전혀 없는 상황에서 source의 **class size statistics**를 활용해 target prediction을 제약한다는 아이디어다. 이는 weakly supervised segmentation의 constrained learning을 domain adaptation에 연결한 점에서 개념적으로 흥미롭다. 특히 segmentation이 단순 classification과 달리 scene layout을 가진다는 점을 잘 활용한 설계다.

다만 한계도 분명하다. 첫째, category-specific adaptation은 source의 클래스별 공간 비율 통계가 target에도 어느 정도 유효하다는 가정에 의존한다. 하지만 실제로는 도시 구조, 카메라 장착 위치, 도로 폭, 날씨, 교통 밀도 등에 따라 클래스 비율이 크게 달라질 수 있다. 예를 들어 고속도로와 도심, 혹은 미국 도시와 유럽 도시 사이에서는 road, building, traffic sign의 상대적 비율이 다를 수 있다. 따라서 source prior가 target에 잘 맞지 않으면 잘못된 제약이 될 위험이 있다.

둘째, pseudo image-level label 추정 역시 현재 모델의 prediction에 의존한다. 즉 초기 예측이 많이 틀리면, 존재하지 않는 클래스를 있다고 판단하거나 실제 존재하는 클래스를 놓칠 수 있다. 논문은 source 모델이 target에서도 chance보다 낫다는 가정을 명시하고 있는데, 이는 매우 중요한 전제다. 만약 초기 전이 성능이 너무 낮으면 constrained adaptation 자체가 불안정해질 수 있다.

셋째, adversarial alignment는 domain confusion을 높이지만, 항상 semantic class alignment까지 보장하는 것은 아니다. 즉 source와 target의 representation이 섞인다고 해서, 같은 의미 클래스끼리 잘 맞는다는 보장은 없다. 이 점을 보완하기 위해 category-specific adaptation이 들어가지만, 여전히 fine-grained class semantics 정렬 문제는 부분적으로 남아 있다.

넷째, 실험 결과를 보면 드문 클래스나 매우 어려운 클래스에서는 여전히 성능이 낮다. 예를 들어 train, motorbike, bike 같은 클래스는 synthetic-to-real setting에서 거의 0에 가까운 성능을 보이는 경우가 많다. 즉 방법이 전체적으로 개선을 주기는 하지만, long-tail 클래스 문제까지 충분히 해결하지는 못한다.

비판적으로 보면, 이 논문은 segmentation adaptation의 출발점으로서 매우 의미 있지만, 이후 연구에서 더 발전된 self-training, pseudo labeling, output-space adaptation, style transfer 기반 adaptation 등이 등장한 이유도 이해할 수 있다. 이 논문의 방식은 통계 prior와 adversarial alignment에 크게 의존하기 때문에, 클래스 의미 수준의 정렬이나 fine detail 복원에는 한계가 있다.

## 6. 결론

이 논문은 semantic segmentation을 위한 unsupervised domain adaptation의 초기이자 중요한 작업이다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, semantic segmentation에 맞는 **fully convolutional adversarial domain alignment**를 제안했다. 둘째, source domain에서 얻은 클래스별 공간 통계를 활용해 target prediction을 제약하는 **category-specific constrained multiple instance adaptation**을 도입했다. 셋째, synthetic-to-real, season-to-season, city-to-city라는 다양한 domain shift에서 실제로 mIoU 개선을 보였다.

이 연구의 의미는 단순히 성능 숫자에만 있지 않다. segmentation adaptation에서는 무엇을 정렬해야 하는지, 이미지 전체가 아니라 어떤 수준의 representation을 adaptation 단위로 삼아야 하는지, 그리고 target supervision이 전혀 없을 때 source의 어떤 구조적 정보를 옮길 수 있는지를 비교적 명확하게 보여준다. 특히 자율주행처럼 장면 구조와 공간 배치가 중요한 응용에서, 단순 feature alignment를 넘어 layout prior를 활용한다는 발상은 이후 연구에도 영향을 줄 수 있는 관점이다.

실제 적용 측면에서는, 새로운 도시나 날씨, 혹은 synthetic simulator에서 real environment로 옮겨갈 때 dense annotation 비용을 줄이면서 모델을 개선할 수 있다는 가능성을 보여준다. 향후 연구에서는 이 논문의 아이디어를 바탕으로 더 정교한 pseudo label 생성, class-wise alignment, output-space regularization, 혹은 image translation과 결합한 방식으로 확장할 수 있다. 따라서 이 논문은 semantic segmentation domain adaptation의 본격적인 출발점으로 평가할 수 있다.
