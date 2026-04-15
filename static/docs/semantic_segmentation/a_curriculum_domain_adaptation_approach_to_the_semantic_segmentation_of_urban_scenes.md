# A Curriculum Domain Adaptation Approach to the Semantic Segmentation of Urban Scenes

- **저자**: Yang Zhang, Philip David, Hassan Foroosh, Boqing Gong
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1812.09953

## 1. 논문 개요

이 논문은 synthetic urban-scene dataset으로 학습한 semantic segmentation 모델을 real urban-scene image에 더 잘 일반화시키기 위한 unsupervised domain adaptation 방법을 다룬다. 문제의 핵심은, synthetic 데이터는 pixel-level annotation을 거의 공짜에 가깝게 얻을 수 있지만, 실제 이미지와는 texture, lighting, viewpoint, color, scene layout 등의 차이로 인해 그대로 학습하면 real domain 성능이 크게 떨어진다는 점이다.

저자들은 특히 semantic segmentation이 단순 classification보다 훨씬 더 구조적인 예측 문제라는 점에 주목한다. 기존의 많은 domain adaptation 방법은 source와 target의 feature distribution을 맞추는 방향으로 접근하지만, 이 논문은 그 과정이 사실상 두 도메인에서 비슷한 $P(Y \mid X)$를 암묵적으로 가정한다고 비판한다. semantic segmentation에서는 출력이 고차원이고 픽셀 간 상호의존성이 매우 강하므로, 이런 가정이 잘 성립하지 않을 수 있다는 것이 저자들의 문제의식이다.

이 문제는 자율주행처럼 semantic segmentation이 핵심 역할을 하는 응용에서 특히 중요하다. 실제 도로 장면에 대한 dense annotation은 이미지 한 장당 1시간 이상이 걸릴 정도로 비싸기 때문에, synthetic data를 현실적으로 쓸 수 있으려면 domain gap을 줄이는 방법이 필요하다. 이 논문은 그 해법으로, 처음부터 어려운 pixel-wise adaptation을 직접 하지 말고 먼저 더 쉬운 과제들을 풀어 target domain의 구조적 속성을 추정한 뒤, 그것을 segmentation network 학습의 regularizer로 활용하는 curriculum learning 스타일의 접근을 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 “semantic segmentation 자체는 어렵지만, urban scene의 거친 구조적 속성은 상대적으로 쉽고 domain gap에도 덜 민감하다”는 것이다. 예를 들어 도로 장면에서는 road가 traffic sign보다 훨씬 많은 픽셀을 차지하고, building은 보통 image 상단과 좌우에 크게 분포하며, road와 sidewalk는 서로 일정한 공간 관계를 가진다. 저자들은 이런 통계적, 구조적 속성을 먼저 추정하면, target에 정답 라벨이 없더라도 segmentation network가 완전히 비상식적인 예측을 하지는 않도록 유도할 수 있다고 본다.

이를 위해 저자들은 두 종류의 “쉬운 과제”를 정의한다. 첫째는 이미지 전체에 대한 global label distribution 추정이다. 즉, 한 target 이미지에서 각 클래스가 전체 픽셀 중 어느 정도 비율을 차지할지를 추정한다. 둘째는 일부 신뢰도 높은 landmark superpixel에 대한 local label distribution 추정이다. 이는 단순히 “무엇이 얼마나 있는가”뿐 아니라 “어디에 무엇이 있는가”에 대한 공간적 힌트도 제공한다.

기존 feature-alignment 계열 방법과의 중요한 차이는, 이 논문이 domain-invariant feature를 직접 강제하지 않는다는 점이다. 대신 target domain에서 segmentation 결과가 만족해야 할 “필요조건”을 먼저 배우고, 그 조건을 만족하도록 network 출력을 posterior regularization 형태로 제어한다. 저자들은 이것이 structured prediction 문제에 더 잘 맞는 관점이라고 주장한다.

## 3. 상세 방법 설명

전체 방법은 두 단계가 결합된 형태다. source domain에서는 일반적인 supervised semantic segmentation을 수행하고, target domain에서는 정답이 없으므로 대신 추정된 label distribution과 network prediction이 일치하도록 학습한다. 즉, source는 pixel-level discriminative signal을 주고, target은 구조적 sanity check를 제공한다.

논문에서 target 이미지 $I_t \in \mathbb{R}^{W \times H}$의 pixel-wise label을 $Y_t \in \mathbb{R}^{W \times H \times C}$로 두고, 예측값을 $\hat{Y}_t(i,j,c)$로 둔다. ground-truth가 있다면 이미지 전체의 클래스 분포는 다음처럼 계산된다.

$$
p_t(c)=\frac{1}{WH}\sum_{i=1}^{W}\sum_{j=1}^{H}Y_t(i,j,c)
$$

이는 이미지 전체에서 클래스 $c$가 차지하는 픽셀 비율이다. 반면 network prediction으로부터 대응되는 분포 $\hat{p}_t$를 만들 때는 softmax 출력을 sharpen하기 위해 큰 상수 $K$를 사용한다.

$$
\hat{p}_t(c)=\frac{1}{WH}\sum_{i=1}^{W}\sum_{j=1}^{H}\left(\frac{\hat{Y}_t(i,j,c)}{\max_{c'}\hat{Y}_t(i,j,c')}\right)^K
$$

논문에서는 $K=6$을 사용했다. $K$가 너무 크면 numerical instability가 생겼다고 명시한다. 이후 이 벡터는 $\ell_1$ 정규화를 통해 합이 1인 valid distribution으로 만든다.

학습 목표는 source segmentation loss와 target property matching loss를 함께 최소화하는 것이다.

$$
\min \frac{\gamma}{|S|}\sum_{s\in S}L(Y_s,\hat{Y}_s)
+\frac{1-\gamma}{|T|}\sum_{t\in T}\sum_k \mathcal{C}(p_t^k,\hat{p}_t^k)
$$

여기서 $L$은 fully labeled source image에 대한 pixel-wise cross-entropy loss이고, $\mathcal{C}(p_t,\hat{p}_t)$는 target property distribution과 prediction-derived distribution 사이의 cross-entropy이다. 논문은 이를 $H(p_t)+KL(p_t,\hat{p}_t)$로 해석한다. $k$는 여러 종류의 property, 즉 global image distribution과 local superpixel distribution을 가리킨다. 핵심은 target 정답 mask 자체는 없지만, 그보다 쉬운 통계적 속성 $p_t^k$는 source로부터 학습한 별도 모델로 추정 가능하다는 점이다.

### Global label distribution 추정

이미지 전체의 label distribution을 추정하기 위해 저자들은 Inception-ResNet-v2의 average pooling output에서 1536차원 feature를 뽑고, 이를 입력으로 여러 방법을 비교한다. 가장 잘 동작한 것은 multinomial logistic regression과 nearest neighbors였다. 최종적으로는 logistic regression의 출력을 target image의 global label distribution으로 사용한다. source image에서는 실제 pixel annotation으로부터 계산된 distribution을 supervision으로 사용한다는 점이 중요하다. 즉, 일반적인 image classification이 아니라 “분포 회귀”에 가깝게 logistic regression을 학습한다.

이 global distribution은 네트워크에 “어떤 클래스가 얼마나 있어야 하는가”를 알려준다. 예를 들어 baseline이 road를 sidewalk로 과도하게 분류하는 경우, 이미지 수준 분포 regularization은 그런 disproportionate prediction을 줄이는 방향으로 작동한다.

### Landmark superpixel distribution 추정

global distribution만으로는 spatial information이 부족하므로, 저자들은 local superpixel 단위의 distribution도 사용한다. 먼저 linear spectral clustering으로 각 이미지를 100개의 superpixel로 나눈다. source domain에서는 각 superpixel에 dominant class를 부여할 수 있으므로, 이를 이용해 multi-class SVM을 학습한다.

target superpixel에 대해 SVM은 클래스와 confidence를 출력한다. 이때 모든 superpixel을 쓰지 않고, confidence가 높은 상위 30%만 landmark superpixel로 선택한다. 선택된 superpixel의 predicted class를 one-hot distribution으로 보고 regularization에 사용한다. 저자들은 모든 superpixel을 다 쓰면 regularization이 지나치게 강해져 source에서 배운 pixel-level discriminativeness를 해칠 수 있다고 설명한다.

superpixel feature는 단순 appearance만 쓰지 않는다. 기본 설정에서는 PASCAL CONTEXT로 pretrain된 FCN-8s의 pixel-wise 59-class score를 구한 뒤, 각 superpixel 내부에서 평균을 내고, 여기에 좌우 및 상하 인접 superpixel의 feature까지 이어붙여 총 295차원 feature를 만든다. 즉, local semantic context까지 포함한 표현이다. 논문은 이후 ablation에서 handcrafted feature와 VGG feature도 비교한다.

### Color constancy

논문은 별도의 preprocessing으로 color constancy도 제안한다. 이는 target real image의 color distribution을 source synthetic domain 쪽으로 보정해 color mismatch를 줄이는 방식이다. 저자들은 illumination 차이가 domain gap의 중요한 원인이라고 보고, gamut-based color constancy 방법을 사용한다. 논문은 세부 알고리즘은 기존 연구 [82]를 참고하라고 하며, 이 단계가 독립적으로 다른 domain adaptation 방법에도 추가될 수 있다고 주장한다. 실제 실험에서도 color constancy를 적용하면 대부분의 변형에서 성능이 더 좋아졌다.

## 4. 실험 및 결과

실험의 기본 target domain은 Cityscapes이며, source domain은 SYNTHIA와 GTA 두 synthetic dataset이다. Cityscapes는 19개 공식 평가 클래스를 갖는 real-world urban scene dataset이고, SYNTHIA는 Cityscapes와 짝지어 쓰기 위한 synthetic subset을 제공하며, GTA는 Grand Theft Auto V에서 추출한 더 대규모의 synthetic urban driving dataset이다. 평가 지표는 Cityscapes 공식 evaluation code의 IoU이며, 전체 클래스 평균인 mIoU가 핵심 수치다.

기본 segmentation backbone은 FCN-8s이고, 대부분의 실험에서 이를 사용한다. convolution layer는 VGG-19로 초기화하고 AdaDelta로 학습한다. 미니배치는 source 5장과 target 5장으로 구성된다. 추가로 더 강한 backbone인 ADEMXAPP에 대해서도 실험해 방법의 일반성을 보인다.

### Global distribution 추정 성능

먼저 target image의 global label distribution을 얼마나 잘 맞추는지 비교했다. SYNTHIA를 source, Cityscapes validation을 target으로 두고 $\chi^2$ distance를 측정했을 때, uniform distribution은 1.13, adaptation 없이 학습한 segmentation network의 예측 분포는 0.65, source 전체 평균 분포는 0.44, nearest neighbor는 0.33, logistic regression은 0.27이었다. 즉, logistic regression이 가장 정확하게 target 이미지의 클래스 비율을 추정했다. 이 결과를 바탕으로 이후 본 실험에서는 LR 기반 global distribution을 사용한다.

이 수치는 중요한 의미를 갖는다. 단순 source-only segmentation output의 분포보다, 아예 별도의 distribution estimator가 target의 클래스 비율을 더 잘 맞춘다는 뜻이다. 논문의 핵심 주장인 “쉬운 과제를 먼저 풀어 structured property를 얻는다”는 아이디어가 실제로 타당하다는 정량적 근거다.

### FCN-8s: SYNTHIA to Cityscapes

SYNTHIA에서 Cityscapes로 적응할 때, source-only baseline인 NoAdapt는 22.0 mIoU였다. color constancy만 적용한 NoAdapt(CC)는 22.6이었다. 여기에 global image distribution만 사용한 `Ours (I)`는 25.5, color constancy를 더한 `Ours (CC+I)`는 27.3으로 상승했다. superpixel regularization만 사용한 `Ours (SP)`는 28.1, `Ours (CC+SP)`는 28.9였다. 둘을 함께 쓴 `Ours (I+SP)`는 29.0, 최종적으로 color constancy까지 포함한 `Ours (CC+I+SP)`는 29.7 mIoU를 기록했다.

즉 baseline 22.0에서 최종 29.7로 약 7.7 포인트 향상되었다. 논문은 이것이 당시 비교 대상인 FCNs in the Wild보다도 더 큰 개선폭이라고 설명한다. 클래스별로 보면 road, building, vegetation, sky처럼 큰 영역 클래스에서 강점이 두드러지지만, pole, traffic sign, traffic light처럼 작은 물체는 여전히 어렵다.

### FCN-8s: GTA to Cityscapes

GTA에서 Cityscapes로 적응할 때도 비슷한 경향이 나타난다. NoAdapt는 22.3, NoAdapt(CC)는 26.2였다. `Ours (I)`는 23.1로 global distribution만으로는 제한적이었지만, color constancy를 더하면 `Ours (CC+I)`가 28.5로 크게 좋아진다. `Ours (SP)`는 27.8, `Ours (CC+SP)`는 30.2였다. `Ours (I+SP)`는 28.9, 최종 `Ours (CC+I+SP)`는 31.4 mIoU였다.

즉 GTA 실험에서도 baseline 22.3에서 31.4로 약 9.1 포인트 향상되었다. 특히 GTA는 SYNTHIA보다 클래스 호환성이 더 좋아 19개 전체 클래스를 쓸 수 있고, road, sidewalk, car, vegetation, sky 등에서 상당한 개선이 보인다. 반면 train 클래스는 최종 결과에서도 0.0 또는 매우 낮은 수준인데, 저자들은 이는 데이터셋 자체에서 train과 bus의 구분이 시각적으로 매우 애매하기 때문이라고 해석한다.

### Global과 superpixel의 상보성

논문은 global image distribution과 landmark superpixel distribution이 서로 다른 방식으로 도움을 준다고 분석한다. global distribution은 전체적인 class proportion을 맞추는 역할이 크고, superpixel은 spatial anchor로 작용해 어디를 어떻게 수정해야 하는지를 알려준다. 실제로 `Ours (I)`와 `Ours (SP)`를 합친 `Ours (I+SP)`가 단독 사용보다 더 높은 성능을 낸다. 이는 두 종류의 regularization이 상보적이라는 증거다.

### Confusion matrix 분석

혼동 행렬 분석에서 저자들은 building이 여러 클래스를 빨아들이는 현상을 관찰했다. 특히 pole, traffic sign, traffic light, fence, wall이 building으로 잘못 분류되는 경우가 많다. 이는 이 클래스들이 건물 주변에 자주 등장하고, 객체 크기가 작으며 intra-class variability도 크기 때문이라고 본다. 또한 bus와 train의 상호 혼동도 큰데, 이는 알고리즘 자체 문제라기보다 GTA와 Cityscapes의 데이터 표현이 시각적으로 매우 유사해서 생기는 한계라고 해석한다.

### Superpixel 표현과 개수에 대한 ablation

superpixel feature는 PASCAL CONTEXT 기반 semantic descriptor뿐 아니라 BOW-SIFT, Fisher Vector, VGG feature도 실험했다. 흥미롭게도 handcrafted feature를 써도 baseline보다 성능이 좋아졌고, VGG feature를 쓰면 landmark superpixel 분류 정확도가 더 올라가면서 SYNTHIA to Cityscapes에서 mIoU가 29.0에서 29.6으로 약간 향상되었다. 즉, superpixel representation은 중요하지만, 반드시 외부 semantic dataset이 필요한 것은 아니다.

superpixel 개수도 실험했는데, GTA to Cityscapes에서 이미지당 50, 100, 200, 400개를 비교한 결과, 성능은 대체로 300 근처까지 증가하다가 포화되는 경향을 보였다. 또 confidence 상위 일부 superpixel만 사용할 때 정확도가 매우 높았으며, 모든 superpixel을 다 쓰는 것은 적절하지 않다는 점도 확인했다.

### 더 강한 backbone에서의 결과

ADEMXAPP backbone을 사용한 실험에서도 동일한 경향이 재현되었다. baseline `ADEMXAPP (CC)`는 30.0 mIoU, 최종 `ADEMXAPP (CC+I+SP)`는 35.7 mIoU였다. 즉, backbone이 바뀌어도 방법이 그대로 적용되고 추가 이득을 준다. 이는 이 접근이 특정 네트워크 구조에 강하게 묶여 있지 않다는 논문의 주장과 일치한다.

### Synthetic data의 “시장 가치”

저자들은 synthetic data가 real annotation을 얼마나 대체할 수 있는지도 실험했다. SYNTHIA만으로 학습한 경우 22.0 mIoU였는데, 여기에 Cityscapes 학습 이미지 5장만 추가해도 33.8로 크게 상승했다. 반면 Cityscapes만으로 학습할 경우 적은 수의 real image만으로는 학습이 제대로 되지 않았고, 20% 수준인 약 450장 이상이 있어야 의미 있는 결과가 나왔다. 저자들은 이를 근거로 SYNTHIA dataset의 가치가 최소 수백 장의 정밀 real annotation에 해당한다고 해석한다. 또한 real data가 1000장 미만일 때는 synthetic data를 섞는 것이 계속 성능 향상에 도움이 되었다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 structured prediction 문제의 domain adaptation을 feature alignment와 다른 관점에서 본다는 점이다. target domain의 픽셀 정답을 직접 맞추기 어려울 때, 그보다 쉬운 구조적 속성을 먼저 추정해 regularizer로 쓰는 방식은 매우 설득력이 있다. 특히 global class proportion과 local landmark superpixel이라는 선택은 urban scene의 고유한 레이아웃 특성을 잘 이용한다. 또한 방법이 segmentation network의 architecture를 거의 바꾸지 않고 적용 가능하다는 점도 실용적이다.

실험적으로도 강점이 분명하다. SYNTHIA와 GTA 두 source dataset, FCN-8s와 ADEMXAPP 두 backbone에서 모두 개선이 나타났고, color constancy, superpixel 표현, superpixel granularity, confusion matrix, few-shot mixing까지 상당히 폭넓은 ablation을 제공한다. 단순히 “좋아졌다”가 아니라 어떤 요소가 왜 도움이 되는지 비교적 자세히 보여준다.

다만 한계도 분명하다. 첫째, 이 방법이 사용하는 target property는 본질적으로 거친 제약이다. 이미지 전체 분포는 픽셀별 세밀한 경계를 알려주지 못하고, superpixel anchor도 일부 landmark에만 적용된다. 따라서 작은 물체나 경계가 복잡한 클래스에서 성능 향상이 제한적이다. 실제 결과에서도 pole, traffic sign, traffic light, wall, fence 등은 여전히 낮은 IoU를 보인다.

둘째, target property 자체가 추정치이므로 오차가 누적될 수 있다. 논문은 특히 모든 superpixel을 다 쓰면 오히려 성능이 나빠진다고 보고하는데, 이는 잘못된 pseudo-property가 지나치게 강한 제약이 될 수 있음을 뜻한다. 따라서 이 방법은 regularization strength와 anchor selection 품질에 민감하다.

셋째, color constancy가 성능 향상에 크게 기여하는데, 이는 반대로 말하면 본 방법의 효과 일부가 segmentation-specific curriculum보다는 preprocessing에 의존한다는 뜻이기도 하다. 물론 저자들도 이를 독립 모듈로 제시하지만, 실제 최종 수치 해석에서는 분리해서 볼 필요가 있다.

넷째, 논문은 Section 5에서 이후 adversarial 방법들과의 상보성을 강조하지만, 그 비교는 대부분 논문 간 보고 수치 요약 또는 late fusion 수준이다. 즉, end-to-end joint training으로 얼마나 잘 결합되는지까지는 이 논문만으로 확정하기 어렵다. 또한 survey 섹션의 일부 비교는 서로 backbone, resolution, implementation이 달라 직접적인 공정 비교로 보기는 어렵다. 저자들 스스로도 subtle implementation difference가 결과에 큰 영향을 준다고 인정한다.

## 6. 결론

이 논문은 semantic segmentation의 unsupervised domain adaptation에서, source와 target의 feature를 직접 맞추는 대신 target output이 만족해야 할 구조적 속성을 먼저 배우고 그것으로 network를 regularize하는 curriculum domain adaptation 프레임워크를 제안한다. 구체적으로 global image-level label distribution과 local landmark superpixel distribution을 추정해, source의 supervised segmentation loss와 함께 target regularization loss로 결합한다.

실험 결과는 이 접근이 synthetic-to-real urban scene segmentation에서 실제로 효과적임을 보여준다. SYNTHIA to Cityscapes와 GTA to Cityscapes 모두에서 baseline 대비 유의미한 mIoU 향상이 있었고, backbone이 바뀌어도 성능 개선이 유지되었다. 또한 synthetic data가 소량의 real annotation을 상당 부분 대체하거나 보완할 수 있음을 정량적으로 보여준 점도 실용적 의미가 크다.

전체적으로 이 연구의 가치는, domain adaptation을 “특징 정렬”만의 문제로 보지 않고 “target에서 지켜야 할 출력 구조를 어떻게 주입할 것인가”라는 방향으로 확장했다는 데 있다. 이후의 self-training, output-space adaptation, structure-aware regularization 같은 흐름과도 자연스럽게 연결되는 아이디어이며, 실제 적용 측면에서도 synthetic data를 활용한 자율주행 perception 연구에 중요한 발판이 되는 논문이라고 볼 수 있다.
