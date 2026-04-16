# Context-Aware Mixup for Domain Adaptive Semantic Segmentation

- **저자**: Qianyu Zhou, Zhengyang Feng, Qiqi Gu, Jiangmiao Pang, Guangliang Cheng, Xuequan Lu, Jianping Shi, Lizhuang Ma
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2108.03557

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation (UDA)** 환경에서 **semantic segmentation** 성능을 높이기 위한 방법을 제안한다. 설정은 전형적이다. 소스 도메인에는 픽셀 단위 정답 라벨이 있고, 타깃 도메인에는 라벨이 없다. 목표는 소스에서 학습한 segmentation 모델이 타깃 도메인에서도 잘 작동하도록 만드는 것이다.

논문이 문제로 지적하는 핵심은, 기존 UDA 기반 segmentation 방법들이 주로 **pixel level**, **feature level**, **output level**에서 도메인 차이를 줄이는 데 집중해 왔지만, 정작 서로 다른 도메인에서도 비교적 안정적으로 공유되는 **context-dependency**, 즉 장면 안의 의미적 맥락 관계를 충분히 활용하지 못했다는 점이다. 예를 들어 도로 옆에 보도가 있고, 자전거 위에는 rider가 있으며, traffic sign은 pole 주변에 존재하는 식의 관계는 synthetic-to-real 도메인 전이에서도 꽤 유지된다. 저자들은 이런 문맥 정보를 무시하면 adaptation 과정에서 **category confusion**, **label contamination**, 심하면 **early performance degradation** 같은 부정적 전이가 발생한다고 본다.

이 문제는 실제로 중요하다. semantic segmentation은 자율주행, 장면 이해 같은 응용에서 핵심 기술이지만, 실제 데이터에 픽셀 단위 주석을 다는 비용이 매우 크다. 그래서 GTAV, SYNTHIA 같은 synthetic dataset으로 학습하고 Cityscapes 같은 real dataset으로 옮기는 UDA가 실용적으로 매우 중요하다. 이 논문은 그 과정에서 “도메인 간에 공유되는 문맥 구조”를 명시적 prior knowledge로 활용하면 성능을 더 끌어올릴 수 있다는 점을 보여주려 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **context-aware mixup**, 즉 문맥을 고려한 cross-domain mixing이다. 기존 mixup 계열 방법은 소스와 타깃 이미지를 섞어 학습 신호를 만들지만, 어떤 객체를 어디에 붙일지에 대한 맥락적 제약이 약하다. 그래서 건물 위로 하늘이 뚫려 보이거나, 사람 일부가 자동차 위에 이상하게 붙는 식의 비현실적 합성이 생길 수 있다. 저자들은 이런 문제를 줄이기 위해, 단순히 class mask로 섞는 것이 아니라 **문맥적으로 함께 나타나야 할 클래스들**을 묶어서 복사하고, 장면의 **공간적 위치 prior**까지 반영해 mixing mask를 만든다.

이를 위해 논문은 **CAMix**라는 프레임워크를 제안한다. 구성의 핵심은 두 가지다.

첫째는 **Contextual Mask Generation (CMG)** 이다. 이 모듈은 소스 도메인에서 얻은 **spatial prior**와 타깃 도메인의 클래스 계층 정보에서 정의한 **contextual relationship**를 이용해 binary contextual mask $M$을 만든다. 이 마스크는 어떤 영역을 타깃에서 가져와 소스 위에 붙일지를 정한다.

둘째는 **Significance-Reweighted Consistency (SRC) loss** 이다. 타깃 pseudo label은 본질적으로 불확실하기 때문에, 단순히 teacher-student consistency를 강하게 거는 것은 오히려 negative transfer를 유발할 수 있다. 저자들은 predictive entropy를 이용해 픽셀별 신뢰도를 나타내는 **significance mask**를 만들고, 이를 mixup 과정에도 반영하여 consistency loss를 가중한다. 즉, 더 믿을 만한 픽셀에는 consistency를 강하게, 불확실한 픽셀에는 약하게 적용한다.

기존 접근과의 차별점은 명확하다. 이 논문은 문맥을 feature space에서 암묵적으로 학습하려는 것이 아니라, **image/mask 수준에서 명시적 prior knowledge로 직접 활용**한다. 또한 adversarial learning이나 image-to-image translation, 다단계 self-training 없이도 **end-to-end**로 학습 가능하다는 점을 장점으로 내세운다.

## 3. 상세 방법 설명

전체 구조는 **student-teacher framework** 위에 올라간다. student 모델 $F_\theta$와, student의 exponential moving average (EMA)로 갱신되는 teacher 모델 $F_{\theta'}$가 있다. 소스 샘플 $(X^S, Y^S)$와 타깃 이미지 $X^T$를 받아, teacher가 만든 타깃 예측을 이용해 문맥 기반 혼합 샘플을 만들고, student는 소스 supervised loss와 mixed sample consistency loss를 함께 학습한다.

### 3.1 Contextual Mask Generation (CMG)

CMG는 문맥 마스크 $M$을 만드는 단계다. 여기에는 두 가지 prior가 들어간다.

하나는 **source spatial prior**이다. 저자들은 소스 데이터에서 클래스별 공간 빈도를 누적하여 $Q \in \mathbb{R}^{C \times H \times W}$ 형태의 spatial prior tensor를 만든다. 각 픽셀 위치마다 어떤 클래스가 나타날 가능성이 높은지를 담은 분포라고 볼 수 있다. 예를 들어 sky는 위쪽, road는 아래쪽에 많다. 이 prior는 teacher의 타깃 예측을 보정하는 데 사용된다. 논문 표기대로라면 spatially modulated target prediction은 다음처럼 표현된다.

$$
\hat{F}_{\theta'} \leftarrow Q \, F_{\theta'}(X^T)
$$

여기서 정확한 곱셈 구현 방식은 본문에 직관 수준으로만 제시되어 있고, element-wise인지 다른 형태인지는 명시적으로 자세히 풀어 쓰지 않았다. 다만 의미는 분명하다. teacher의 타깃 예측을 source spatial prior로 regularize한다는 것이다.

다른 하나는 **target contextual relationship**이다. Cityscapes의 coarse/fine category hierarchy를 이용해, 문맥적으로 연관된 fine classes를 meta-class group으로 묶는다. 논문은 예시로 다음 그룹들을 제시한다.

- Group I: pole, traffic sign, traffic light
- Group II: rider, motorcycle, bicycle
- Group III: road, sidewalk
- Group IV: building, wall, fence
- Group V: vegetation, terrain

이 중 실험에서는 **Group I + Group II**를 기본 meta-class list $m$으로 사용한다. 저자 주장에 따르면 문맥 prior가 너무 적으면 supervision이 부족하고, 너무 많으면 제약이 과해 local optimum에 빠질 수 있기 때문이다.

구체적인 마스크 생성 절차는 다음과 같다.

1. teacher의 spatially modulated prediction으로부터 pseudo label $\tilde{Y}^T$를 만든다.
2. 그 이미지에서 등장한 클래스 집합 $C$를 구한다.
3. $C$에서 절반 정도의 클래스를 랜덤 선택해 리스트 $c$를 만든다.
4. 선택된 클래스 $k$가 meta-class list 안에 있으면, 그와 의미적으로 연관된 클래스 $\tilde{k}$도 함께 추가한다.
5. 최종 클래스 리스트 $c$에 속하는 픽셀 위치를 1로 하여 binary mask $M$을 만든다.

수식은 다음과 같다.

$$
M(h,w)=
\begin{cases}
1, & \text{if } \tilde{Y}^T(h,w)\in c \\
0, & \text{otherwise}
\end{cases}
$$

핵심은, 단순히 클래스 일부를 랜덤하게 잘라 붙이는 것이 아니라, **문맥적으로 같이 다녀야 할 클래스들을 함께 가져오도록** 만드는 것이다.

### 3.2 Input-level과 Output-level Domain Mixup

생성된 contextual mask $M$은 세 가지 level에 쓰이는데, 먼저 input level과 output level이 있다.

입력 이미지 mixup은 다음과 같다.

$$
X^M = M X^T + (1-M) X^S
$$

즉, 마스크가 1인 영역은 타깃 이미지에서, 0인 영역은 소스 이미지에서 가져온다. 중요한 점은 이 논문이 기존 DACS류와 다르게 **target-to-source 방향**으로 mixing한다는 것이다. 타깃의 일부 semantic region을 소스 위에 붙이는 방식이다. 저자들은 이 방향이 spatial/context prior를 활용하기 더 적합하다고 주장한다.

teacher 파라미터는 EMA로 갱신된다.

$$
\Phi'_t = \alpha \cdot \Phi'_{t-1} + (1-\alpha)\cdot \Phi_t
$$

output level에서는 소스 정답 라벨 $Y^S$와 teacher가 만든 타깃 pseudo label $\hat{Y}^T = F_{\theta'}(X^T)$를 동일한 마스크 $M$으로 섞어 mixed label $Y^M$을 만든다.

$$
Y^M = M \hat{Y}^T + (1-M) Y^S
$$

이렇게 하면 입력 이미지와 supervision 신호가 같은 문맥 구조를 공유하게 된다.

### 3.3 Significance-mask Level Domain Mixup

이 논문의 또 다른 핵심은 **significance mask level**이다. 여기서는 teacher prediction의 신뢰도를 추정한 뒤, 그 신뢰도를 mixup과 consistency loss에 반영한다.

#### Stochastic forward passes

각 타깃 이미지 $X^T$에 대해 $L$개의 복사본을 만들고, 각 복사본에 random Gaussian noise를 넣어 여러 번 추론한다. 각 stochastic pass $l$의 픽셀별 class probability를 $P_l(h,w,c)$라 하면, 평균 predictive probability는 다음과 같다.

$$
\hat{P}(h,w,c)=\frac{1}{L}\sum_{l=1}^{L} P_l(h,w,c)(X_l^T)
$$

그 다음 predictive entropy를 계산한다.

$$
\zeta(h,w) = - \sum_{c=1}^{C} \hat{P}(h,w,c)\log(\hat{P}(h,w,c))
$$

entropy가 크면 그 픽셀 예측은 불확실하다고 해석한다.

#### Dynamic threshold

신뢰도 판정에는 고정 threshold 대신 학습 단계에 따라 변하는 dynamic threshold $R$을 쓴다.

$$
R = \beta + (1-\beta)\cdot e^{\gamma (1-t/t_{\max})^2} \cdot K^{sup}
$$

여기서 $t$는 현재 iteration, $t_{\max}$는 최대 iteration, $K^{sup}$는 entropy 값들의 상한, $\beta$는 초기 상태, $\gamma$는 threshold 변화 속도를 조절하는 하이퍼파라미터다. 논문은 기본값으로 $\beta=0.75$, $\gamma=-5$를 사용했다.

#### Significance mask

이 threshold를 이용해 타깃 significance mask $U^T$를 만든다.

$$
U^T = I(\zeta < R)
$$

즉 entropy가 threshold보다 작은, 비교적 신뢰할 수 있는 픽셀만 1이 된다. 소스는 정답 라벨이 있으므로 source significance mask $U^S$는 전부 1인 텐서로 둔다.

그 후 contextual mask $M$으로 두 significance mask를 섞는다.

$$
U^M = M U^T + (1-M) U^S
$$

이 $U^M$은 최종 consistency loss의 픽셀별 가중치 역할을 한다.

### 3.4 Significance-Reweighted Consistency (SRC) Loss

mixed image $X^M$에 대해 student prediction과 teacher-derived supervision 사이의 consistency를 강제하되, 그 강도를 $U^M$으로 조절한다. 논문 수식은 다음과 같다.

$$
L_{con}(f_{\theta'}, f_{\theta}) =
\frac{\sum_j \left(U^M \cdot CE(F_\theta(X^M), Y^M)\right)}{\sum_j U^M}
$$

여기서 $CE$는 cross-entropy loss다. 의미는 단순하다. mixed sample 전체에 대해 같은 강도로 consistency를 거는 대신, **믿을 수 있는 픽셀 위주로 평균낸 weighted CE**를 사용한다는 것이다.

저자들은 segmentation에서는 MSE나 KL보다 CE 기반 consistency가 더 적합하다고 보고, 실제 ablation에서도 이를 뒷받침한다.

### 3.5 전체 학습 목적함수와 추론

소스 supervision용 segmentation loss는 일반적인 cross-entropy다.

$$
L_{seg} =
-\sum_{h=1}^{H}\sum_{w=1}^{W}\sum_{c=1}^{C}
Y^S_{(h,w,c)} \log(P^S_{(h,w,c)})
$$

최종 loss는 다음과 같다.

$$
L_{total} = L_{seg} + \lambda_{con} L_{con}
$$

즉, 소스 정답 학습과 mixed sample consistency 학습을 함께 한다. 논문은 이 전체 프레임워크가 **fully end-to-end trainable**하다고 강조한다. 추론 시에는 student보다 일반적으로 약간 더 나은 성능을 보인다고 판단한 **teacher model만 사용**한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 두 가지 대표 benchmark를 사용한다.

- **GTAV $\rightarrow$ Cityscapes**
- **SYNTHIA $\rightarrow$ Cityscapes**

Cityscapes는 자율주행 장면 중심의 real dataset이며, train 2,975장, val 500장으로 구성된다. GTAV는 24,966장의 synthetic 이미지와 19개 클래스를 가진다. SYNTHIA는 9,400장의 annotated synthetic 이미지이며, 논문은 SYNTHIA-RAND-CITYSCAPES subset을 사용한다.

입력 전처리는 소스가 GTAV일 때 $1280 \times 720$, SYNTHIA일 때 $1280 \times 760$으로 resize 후, source/target 모두 $512 \times 512$ random crop을 적용한다.

백본은 두 종류를 사용한다.

- **DeepLabV2 + ResNet-101**
- **SegFormer (MiT-B5 encoder 기반, DAFormer 세팅)**

이 점은 중요하다. 논문은 단순히 한 모델에서만 잘 되는 트릭이 아니라, 오래된 CNN 계열과 최신 transformer 계열 모두에 plug-in 가능하다는 점을 보이려 한다.

### 4.2 주요 정량 결과

#### GTAV $\rightarrow$ Cityscapes

- **DACS baseline**: 52.1 mIoU
- **CAMix with DACS**: 55.2 mIoU

즉 **+3.1%p** 향상이다.

- **DAFormer baseline**: 68.3 mIoU
- **CAMix with DAFormer**: 70.0 mIoU

즉 **+1.7%p** 향상이다.

세부 클래스별로도 motorcycle, bicycle, traffic sign 등 여러 클래스에서 향상이 나타난다. 특히 DAFormer 조합에서는 train, bus, truck 같은 클래스에서도 개선이 보인다.

#### SYNTHIA $\rightarrow$ Cityscapes

- **DACS baseline**: 54.8 mIoU (13-class 기준)
- **CAMix with DACS**: 59.7 mIoU

즉 **+4.9%p** 향상이다.

- **DAFormer baseline**: 67.4 mIoU
- **CAMix with DAFormer**: 69.2 mIoU

즉 **+1.8%p** 향상이다.

이 결과는 CAMix가 상대적으로 약한 baseline뿐 아니라 강한 baseline에도 추가 개선을 제공함을 보여준다.

### 4.3 Domain mixup 방법들과의 비교

논문은 Mean Teacher를 공통 기반으로 두고 CowMix, CutMix, DACS, inverse DACS(iDACS), DAFormer, inverse DAFormer(iDAFormer)와 비교한다.

DeepLabV2 기준:

- Mean Teacher: 43.1
- CowMix: 48.3
- CutMix: 48.7
- DACS: 52.1
- iDACS: 51.5
- **CAMix: 55.2**

SegFormer 기준:

- Mean Teacher: 51.6
- CowMix: 58.9
- CutMix: 58.7
- DAFormer: 68.3
- iDAFormer: 62.4
- **CAMix: 70.0**

이 비교가 말해 주는 바는 분명하다. 단순히 “섞는다”는 사실만으로 충분한 것이 아니라, **어떻게 섞느냐**, 특히 **문맥을 지키며 섞느냐**가 중요하다는 것이다.

### 4.4 Ablation study

#### CMG 구성 요소

iDACS baseline 51.5에서 시작해,

- spatial prior (SP) 추가: 53.1
- contextual relationship (CR)까지 추가: 54.5
- SRC까지 모두 추가: 55.2

즉 CMG의 두 요소가 각각 의미 있는 기여를 하고, SRC가 마지막으로 추가적인 향상을 준다.

#### Level별 기여

Mean Teacher baseline은 GTAV에서 43.1, SYNTHIA에서 45.9다.

- input + output level mixup만 사용: 54.5 / 59.0
- 여기에 SigMask level까지 추가: 55.2 / 59.7

즉 input-output mixing이 큰 폭의 개선을 만들고, significance-mask level이 추가 향상을 제공한다. 저자들은 세 level이 서로 보완적이라고 해석한다.

#### SRC loss 효과

GTAV $\rightarrow$ Cityscapes에서:

- full CAMix with SRC: 55.2
- SRC 없이 MSE consistency: 44.5
- SRC 없이 일반 CE consistency: 54.2

여기서 중요한 메시지는 두 가지다. 첫째, segmentation consistency에는 MSE가 매우 부적절할 수 있다. 둘째, 단순 CE consistency보다도 **신뢰도 가중을 넣은 SRC가 더 좋다**.

#### Meta-class list 분석

Group I만 쓰면 68.8, Group I+II가 70.0으로 가장 좋다. 이후 Group III, IV, V를 더 넣을수록 오히려 성능이 떨어져 67.3까지 내려간다. 이는 저자 주장대로 문맥 prior가 너무 많으면 제약이 지나치게 강해질 수 있음을 시사한다.

### 4.5 정성적 결과와 학습 안정성

논문은 mixed sample 시각화에서 DACS는 label contamination과 category confusion이 많고, CAMix는 더 자연스러운 문맥 구조를 유지한다고 보인다. segmentation 결과 시각화에서도 road, sidewalk, truck, traffic sign 같은 클래스에서 DACS보다 더 안정적인 예측을 보였다고 주장한다.

또한 adaptation curve 분석에서는 기존 Mean Teacher 기반 consistency regularization이 학습 초기에 성능이 심하게 흔들리고 떨어지는 반면, SRC를 쓰면 그런 instability가 완화된다고 보고한다. 이는 이 논문의 주장인 “context + significance weighting이 negative transfer를 줄인다”는 점과 연결된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 분명하고, 제안한 해법이 그 문제에 잘 맞는다는 점이다. 저자들은 기존 UDA segmentation이 context를 충분히 활용하지 못한다고 지적하고, 실제로 이를 해결하기 위한 메커니즘을 mask generation, mixup direction, significance reweighting까지 일관되게 설계했다. 단순한 아이디어 제안에 그치지 않고, **입력-출력-SigMask 세 level**로 이어지는 구조적 설계를 제시한 점이 좋다.

또 다른 강점은 **실용성**이다. adversarial discriminator, image translation, multi-stage retraining 없이 end-to-end로 학습 가능하다는 점은 구현과 수렴 측면에서 분명한 장점이다. 게다가 DACS와 DAFormer 모두에 꽂아서 성능 향상을 보였기 때문에, 독립적인 새 프레임워크라기보다 기존 강한 UDA baseline을 강화하는 일반적 모듈로 볼 수 있다.

실험도 비교적 설득력 있다. 단순 SOTA 표만 제시한 것이 아니라, related mixup methods 비교, component ablation, loss ablation, meta-class ablation, hyperparameter 분석, qualitative 결과까지 포함하고 있어 논문이 주장하는 각 요소의 역할을 어느 정도 검증한다.

한편 한계도 있다. 첫째, 문맥 prior의 일부는 **Cityscapes의 계층 구조에 의존하는 수작업 meta-class design**이다. 즉 어떤 클래스들을 함께 묶을지 사람이 정한 부분이 있으며, 이 prior가 다른 데이터셋이나 다른 장면 유형에 얼마나 일반화되는지는 논문만으로는 충분히 확인되지 않는다.

둘째, spatial prior tensor $Q$와 contextual relation을 결합하는 과정은 직관은 명확하지만, 본문 설명만으로는 그 연산 방식이 완전히 세밀하게 명시되었다고 보기는 어렵다. 특히 $QF_{\theta'}(X^T)$의 구체적 구현 형태는 논문 텍스트만으로는 상세히 알기 어렵다.

셋째, SRC를 위해 stochastic forward pass를 여러 번 수행하므로 계산량이 증가한다. 논문은 $N=8$회 stochastic passes를 사용했다고 했지만, 이것이 실제 학습 시간과 메모리 비용에 얼마나 영향을 주는지는 정량적으로 자세히 보고하지 않았다.

넷째, 논문은 negative transfer 완화와 early degradation 감소를 강조하지만, 그 효과를 보여주는 정량 근거는 일부 curve와 ablation 중심이다. 더 다양한 backbone이나 더 넓은 도메인 시나리오에서 동일한 안정성 이득이 유지되는지는 추가 검증이 필요하다.

비판적으로 보면, 이 방법은 “문맥을 지키는 mixup”이라는 점에서 분명 설득력 있지만, 문맥 정의 자체가 결국 데이터셋 구조와 label taxonomy에 일부 의존한다. 따라서 완전히 데이터 주도적이고 일반적인 context modeling이라기보다는, **적절한 domain knowledge를 mixup 설계에 잘 녹여 넣은 방법**에 가깝다. 그러나 바로 그 점이 실제 성능 향상에 강하게 기여한 것으로 보인다.

## 6. 결론

이 논문은 domain adaptive semantic segmentation에서 기존 방법들이 충분히 활용하지 못했던 **cross-domain context-dependency**를 명시적 prior로 활용하는 **CAMix** 프레임워크를 제안한다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, source spatial prior와 target contextual relation을 이용해 contextual mask를 생성하는 **CMG**를 제안했다. 둘째, 그 마스크를 input, output, significance-mask 세 level의 mixup에 일관되게 적용했다. 셋째, predictive entropy 기반 significance mask로 consistency를 가중하는 **SRC loss**를 도입해 adaptation instability와 negative transfer를 줄이려 했다.

실험 결과는 이 접근이 단순한 아이디어 수준이 아니라 실제 성능 향상으로 이어진다는 점을 보여준다. GTAV $\rightarrow$ Cityscapes와 SYNTHIA $\rightarrow$ Cityscapes에서 DACS, DAFormer 같은 강한 baseline 위에 안정적인 개선을 얻었고, ablation도 각 구성 요소의 효과를 뒷받침한다.

실제 적용 측면에서는 synthetic-to-real segmentation, 특히 자율주행 장면 이해처럼 문맥 구조가 강한 문제에서 유용할 가능성이 크다. 향후 연구로는 meta-class를 수작업으로 정하지 않고 자동으로 학습하는 방향, 보다 일반적인 scene graph나 relational prior와 결합하는 방향, 그리고 다른 dense prediction task로의 확장이 자연스럽게 이어질 수 있다. 전체적으로 이 논문은 UDA segmentation에서 “무엇을 정렬할 것인가”뿐 아니라 “어떤 문맥을 보존하며 적응할 것인가”라는 질문을 전면에 놓았다는 점에서 의미가 있다.
