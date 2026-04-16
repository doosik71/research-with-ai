# Bootstrapping Semantic Segmentation with Regional Contrast

- **저자**: Shikun Liu, Shuaifeng Zhi, Edward Johns, Andrew J. Davison
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2104.04465

## 1. 논문 개요

이 논문은 semantic segmentation에서 라벨이 매우 적을 때도 성능을 높일 수 있는 contrastive learning 기반 보조 학습 방법인 **ReCo (Regional Contrast)**를 제안한다. 핵심 문제의식은 분명하다. semantic segmentation은 픽셀 단위 정답이 필요하기 때문에 annotation 비용이 매우 크고, 실제 응용에서는 충분한 pixel-level label을 확보하기 어렵다. 특히 Cityscapes처럼 정밀한 경계가 필요한 데이터셋에서는 한 장의 이미지를 라벨링하는 데도 큰 비용이 든다.

저자들은 기존 segmentation 모델이 대체로 인접 픽셀의 smoothness bias를 가지기 때문에 경계가 흐려지고, 드문 클래스나 작은 객체에서 오분류가 자주 발생한다고 본다. 더 나아가 오분류는 모든 클래스 사이에서 무작위로 일어나는 것이 아니라, 소수의 “헷갈리기 쉬운 클래스 쌍” 사이에서 집중적으로 발생한다고 관찰한다. 예를 들어 `rider`는 `building`보다 `person`으로 잘못 예측될 가능성이 훨씬 높다. 이 구조를 학습에 직접 반영하자는 것이 논문의 출발점이다.

논문의 목표는 segmentation network가 단순히 local context만 보는 것이 아니라, 데이터셋 전체에 걸친 **semantic class relationship**까지 활용하도록 만드는 것이다. 이를 위해 ReCo는 픽셀 표현 공간에서 같은 클래스는 더 가깝게, 혼동되기 쉬운 다른 클래스는 더 멀어지게 학습한다. 특히 모든 픽셀을 다 쓰지 않고도, **hard query와 informative negative key만 희소하게 샘플링**하여 메모리 증가를 최소화하는 것이 중요한 설계 포인트다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 semantic segmentation을 위한 pixel-level contrastive learning을, “모든 픽셀을 전부 비교하는 방식”이 아니라 **regional, class-aware, hard-sample-focused 방식**으로 재구성한 데 있다.

ReCo의 직관은 다음과 같다. 어떤 클래스 `c`에 속하는 픽셀 표현들이 있다면, 이들을 그 클래스의 평균 표현(class mean representation) 쪽으로 끌어당긴다. 동시에 query 클래스와 혼동되기 쉬운 다른 클래스의 표현은 더 강하게 밀어낸다. 즉, 단순히 positive/negative를 균일하게 다루지 않고, **현재 query 클래스와 실제로 헷갈리는 negative class를 더 자주 뽑는다**. 이 점이 일반적인 supervised contrastive learning보다 segmentation 문제에 더 맞게 조정된 부분이다.

기존 접근과의 차별점은 크게 세 가지다.

첫째, dense pixel representation을 사용하면서도 모든 픽셀을 쓰지 않는다. 고해상도 segmentation에서 전체 픽셀 contrastive learning은 메모리와 계산량이 매우 크다. ReCo는 전체 픽셀의 5% 미만만 사용하도록 설계되어 실용적이다.

둘째, negative sampling이 균일하지 않다. 클래스 간 평균 표현의 유사도를 이용해 pairwise class relation graph를 만들고, 이 분포를 기반으로 hard negative를 샘플링한다. 따라서 `person`을 학습할 때 `bicycle` 같은 관련 클래스가 더 자주 negative로 들어간다.

셋째, query도 무작위가 아니라 **낮은 confidence를 가진 hard pixel** 위주로 뽑는다. segmentation에서는 흔한 클래스가 픽셀 수를 압도하기 때문에 랜덤 샘플링을 하면 rare class와 boundary pixel이 충분히 학습되지 않는다. ReCo는 uncertainty가 큰 픽셀을 골라 여기에 contrastive supervision을 집중한다.

## 3. 상세 방법 설명

전체 구조는 기존 segmentation network 위에 얹는 보조 모듈 형태다. 기본 segmentation network $f_\theta$는 encoder $\phi$와 classification head $\psi_c$로 나뉜다. 여기에 학습 시에만 사용하는 representation head $\psi_r$를 추가하여, 각 픽셀에 대해 classification용 출력과 별개로 $m$차원의 dense representation을 만든다. 논문에서는 $m=256$을 사용했다. 추론 시에는 이 representation head를 제거하므로 inference overhead는 없다.

ReCo loss의 기본 형태는 픽셀 표현 $r_q$를 같은 클래스의 positive key와 가깝게, 다른 클래스의 negative key들과는 멀게 만드는 contrastive loss다. 저자들이 제시한 식은 다음과 같다.

$$
L_{\text{reco}} =
\sum_{c \in C}
\sum_{r_q \sim R_q^c}
-\log
\frac{\exp(r_q \cdot r_k^{c,+}/\tau)}
{\exp(r_q \cdot r_k^{c,+}/\tau)+\sum_{r_k^- \sim R_k^c}\exp(r_q \cdot r_k^-/\tau)}
$$

여기서 $C$는 현재 mini-batch에 존재하는 클래스 집합이고, $\tau$는 temperature이며 논문에서는 $\tau=0.5$를 사용했다. $R_q^c$는 클래스 $c$에 속하는 query pixel들의 표현 집합이고, $R_k^c$는 클래스 $c$가 아닌 negative key 표현 집합이다. positive key $r_k^{c,+}$는 클래스 $c$의 평균 표현이다. 즉,

$$
r_k^{c,+} = \frac{1}{|R_q^c|}\sum_{r_q \in R_q^c} r_q
$$

이다. 이 설계는 같은 클래스 내부의 픽셀 표현을 한 점으로 collapse시키자는 뜻은 아니고, class-wise 중심을 기준으로 representation space를 더 정돈하려는 목적에 가깝다.

하지만 모든 픽셀을 query/key로 쓰면 비현실적이므로, ReCo는 **active hard sampling**을 도입한다.

먼저 **active key sampling**이다. 클래스 간 관계 그래프 $G$를 만들고, 클래스 $p$와 $q$의 평균 표현 유사도를

$$
G[p,q] = r_k^{p,+} \cdot r_k^{q,+}
$$

로 정의한다. 그 뒤 query class $c$에 대해, negative class들 사이의 SoftMax 분포를 만든다. 이 분포가 높다는 것은 해당 클래스가 query class와 representation space에서 가깝고, 실제로 혼동될 가능성이 높다는 뜻이다. 따라서 negative key는 이 분포에 따라 샘플링한다. 결과적으로 쉬운 negative보다 hard negative를 더 많이 보게 된다.

다음은 **active query sampling**이다. 논문은 segmentation에서 흔한 클래스가 대부분의 픽셀을 차지하기 때문에 random query sampling은 비효율적이라고 본다. 그래서 각 픽셀의 예측 confidence $\hat y_q$를 기준으로, confidence가 threshold $\delta_s$ 이하인 픽셀을 hard query로 정의한다. 식으로 쓰면

$$
R_{q,\text{easy}}^c = \bigcup_{r_q \in R_q^c} 1(\hat y_q > \delta_s) r_q,
\quad
R_{q,\text{hard}}^c = \bigcup_{r_q \in R_q^c} 1(\hat y_q \le \delta_s) r_q
$$

이다. 논문에서는 $\delta_s=0.97$를 사용했다. 즉, 모델이 자신 없어하는 픽셀, 특히 경계나 rare class 주변 픽셀에 contrastive supervision을 집중한다.

학습 방식은 supervised와 semi-supervised에서 조금 다르다.

완전 supervised setting에서는 전체 loss가 단순하다.

$$
L_{\text{total}} = L_{\text{supervised}} + L_{\text{reco}}
$$

여기서 $L_{\text{supervised}}$는 일반적인 pixel-wise cross-entropy loss다.

semi-supervised setting에서는 Mean Teacher framework를 사용한다. student model $f_\theta$와 teacher model $f_{\theta'}$를 두고, teacher는 EMA로 업데이트된다.

$$
\theta_t' = \lambda \theta'_{t-1} + (1-\lambda)\theta_t
$$

논문에서는 $\lambda=0.99$다. teacher가 unlabeled image의 pseudo-label을 만들고, student는 augmented unlabeled image에 대해 그 pseudo-label을 학습한다. 단, pseudo-label이 부정확할 수 있으므로 unlabeled pixel 중에서도 confidence가 $\delta_w$ 이상인 픽셀만 ReCo에 사용한다. 논문에서 $\delta_w=0.7$이다.

이때 전체 loss는

$$
L_{\text{total}} = L_{\text{supervised}} + \eta \cdot L_{\text{unsupervised}} + L_{\text{reco}}
$$

이다. 여기서 $\eta$는 pseudo-label confidence가 일정 수준 이상인 픽셀 비율로 정의되며, 초반 학습에서 noisy pseudo-label이 전체 학습을 지배하지 않도록 한다.

정리하면 ReCo는 segmentation 자체를 바꾸는 새 backbone이나 decoder가 아니라, 기존 DeepLabV3+ 같은 모델 위에 쉽게 얹을 수 있는 **auxiliary representation learning loss**다. 실험에서는 query 256개, key 512개만으로도 좋은 성능을 냈고, 이는 concurrent work가 수만 개의 query/key를 쓰는 것에 비해 메모리 효율이 매우 높다는 주장으로 이어진다.

## 4. 실험 및 결과

실험은 Pascal VOC 2012, Cityscapes, SUN RGB-D에서 수행되었다. backbone과 segmentation architecture는 모두 **DeepLabV3+ with ResNet-101**로 통일했고, optimizer는 SGD, learning rate는 $2.5\times10^{-3}$, momentum은 $0.9$, weight decay는 $5\cdot10^{-4}$를 사용했다. 모든 데이터셋에서 40k iteration 학습했고, polynomial learning rate decay를 적용했다.

논문은 semi-supervised segmentation benchmark를 두 가지로 재설계한 점도 중요하다.

하나는 **Partial Dataset Full Labels**로, 일부 이미지만 fully labeled이고 나머지는 unlabeled인 일반적 semi-supervised setting이다. 다만 극소수 label에서도 모든 클래스가 최소한 나타나도록 샘플링 규칙을 정교하게 설계했다.

다른 하나는 **Partial Labels Full Dataset**으로, 모든 이미지가 주어지지만 각 이미지에서 클래스별로 매우 적은 수의 픽셀만 라벨이 있는 setting이다. 이 경우 클래스마다 한 픽셀 또는 일부 퍼센트만 라벨링하고, dilation으로 sparse annotation을 만든다. 이 setting은 “경계 정보가 거의 없는 상태에서 semantic completion을 얼마나 잘 하느냐”를 보는 더 까다로운 문제다.

비교 baseline으로는 supervised 학습과 함께 S4GAN, CutOut, CutMix, ClassMix를 같은 구조와 같은 split 위에서 직접 재구현했다. 이 점은 비교의 공정성을 높여 준다. 논문에서는 semi-supervised baseline 중 **ClassMix가 가장 강한 baseline**이었고, 그래서 주된 비교는 `ReCo + ClassMix` 형태로 이루어진다.

주요 결과를 보면, full label benchmark에서 ReCo는 supervised와 semi-supervised 모두에서 일관되게 성능을 높였다.

Pascal VOC에서:
- `all labels` supervised는 $77.79$ mIoU이고, `ReCo + Supervised`는 $78.39$이다.
- 200 labels 기준 `ClassMix`는 $67.95$, `ReCo + ClassMix`는 $69.81$이다.

Cityscapes에서:
- `all labels` supervised는 $70.48$, `ReCo + Supervised`는 $71.45$이다.
- 20 labels 기준 `ClassMix`는 $45.61$, `ReCo + ClassMix`는 $49.86$이다.

SUN RGB-D에서:
- `all labels` supervised는 $51.06$, `ReCo + Supervised`는 $52.01$이다.
- 50 labels 기준 `ClassMix`는 $28.42$, `ReCo + ClassMix`는 $29.65$이다.

특히 라벨이 가장 적은 few-label regime에서 improvement가 크다. 논문은 상대적 기준으로 약 5%에서 10% 수준의 향상을 강조한다. 예를 들어 Cityscapes 20 labels에서 $45.61 \to 49.86$은 절대값으로도 꽤 큰 차이이며, low-label segmentation에서는 의미 있는 개선이다.

기존 benchmark와의 비교에서도 강한 결과를 보였다. Pascal VOC의 PseudoSeg setting에서는 `ReCo + ClassMix`가 1/16 split에서 $64.78$, 1/8 split에서 $72.02$, 1/4 split에서 $73.14$, 1/2 split에서 $74.69$ mIoU를 기록했다. 논문은 이 결과를 통해 ReCo가 당시 state-of-the-art 수준이며, 특히 PseudoSeg 성능에 근접하거나 이를 넘는 수준을 **더 적은 라벨**로 달성할 수 있다고 주장한다.

Partial label setting에서도 개선은 유지되지만, full label setting보다는 폭이 작다. 예를 들어 Pascal VOC에서 1 pixel per class per image 설정에서 `ClassMix`는 $63.69$, `ReCo + ClassMix`는 $66.11$이다. Cityscapes에서는 같은 설정에서 $47.42 \to 49.66$이다. 저자들은 이 경우 ground-truth boundary 정보가 너무 희소해서 ReCo가 부정확한 supervision을 받을 수 있다고 해석한다.

정성적 결과에서도 ReCo의 특징이 분명하다. Cityscapes에서는 `person`, `bicycle` 같은 작은 객체 경계가 더 또렷하고, SUN RGB-D에서는 `lamp`, `pillow` 같은 작고 경계가 복잡한 객체의 segmentation이 더 선명하다. 완전히 헷갈리기 쉬운 클래스 쌍, 예를 들어 `table`과 `desk`, `window`와 `curtain` 같은 경우에도 클래스 자체의 혼동은 완전히 사라지지 않지만, ReCo는 적어도 object boundary를 더 날카롭게 유지한다.

ablation도 논문의 설득력을 높인다. query와 key 개수를 늘릴수록 성능이 좋아지지만 일정 수준 이후에는 수익이 감소한다. 중요한 점은 query 32개만 써도, 즉 전체 픽셀의 0.5% 미만만 사용해도 baseline 대비 유의미한 향상이 나온다는 것이다. 또한 hard query sampling이 핵심이라는 점도 Table 4에서 드러난다. random query/random key는 $46.56$, active query/random key는 $46.38$, easy query/random key는 $45.81$인데, active query와 active key를 모두 쓴 기본 설정은 $49.86$이다. 즉, 단순 랜덤 샘플링이 아니라 **어떤 픽셀을 고를지**가 성능의 핵심이다.

또 하나 흥미로운 실험은 feature bank 방식과의 비교다. 저자들은 concurrent work처럼 stored feature bank를 써도 성능은 비슷하지만($49.34$ vs. $49.86$), 속도는 느렸다고 보고한다. 이 결과는 mini-batch 기반 샘플링만으로도 전체 데이터 분포를 충분히 근사할 수 있다는 저자들의 주장에 힘을 실어 준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정에 맞는 contrastive learning 설계**를 했다는 점이다. 일반적인 image-level contrastive learning을 segmentation에 억지로 적용한 것이 아니라, pixel-level dense representation, class-mean positive, hard query, class-relation-aware negative sampling을 조합해 segmentation 특성에 맞게 재구성했다. 특히 rare class와 boundary pixel에 집중하게 만든 설계는 실제 segmentation 오류 양상과 잘 맞아떨어진다.

둘째 강점은 **실용성**이다. ReCo는 기존 segmentation architecture를 거의 그대로 유지한 채 representation head와 auxiliary loss만 추가하면 된다. inference time overhead가 없고, memory usage도 feature bank 방식보다 훨씬 작다. 논문이 “less than 5% of all available pixels”만 사용한다고 강조하는 이유가 여기에 있다.

셋째는 **낮은 라벨 수에서의 효과가 분명하다**는 점이다. 논문의 핵심 메시지는 많은 라벨이 있을 때 소폭 개선하는 기법이 아니라, 극소량의 label만 있을 때도 segmentation을 성립시키는 방향의 보조 학습이라는 데 있다. 실제 결과도 이 주장과 잘 맞는다.

넷째는 **해석 가능성**이다. class relation graph와 dendrogram을 통해 ReCo가 representation space를 더 disentangled하게 만든다는 점을 시각적으로 보여 준다. 이는 단순히 mIoU만 올랐다는 주장보다 한 단계 더 나아간 분석이다.

반면 한계도 분명하다.

첫째, 이 방법은 여전히 **pseudo-label 품질과 confidence threshold 선택**에 의존한다. semi-supervised setting에서는 unlabeled pixel 중 confidence가 높은 것만 사용하기 때문에, teacher 예측이 편향되면 ReCo 역시 그 편향을 강화할 가능성이 있다. 논문은 이를 thresholding으로 완화하지만, 이 문제가 근본적으로 해결된 것은 아니다.

둘째, partial label setting에서는 개선 폭이 줄어든다. 이는 ReCo가 클래스 구조를 잘 이용하더라도, 경계 정보가 지나치게 빈약하면 오히려 contrastive supervision의 기준 자체가 불안정해질 수 있음을 뜻한다. 저자들도 이 경우를 open research question으로 남긴다.

셋째, class mean representation을 positive key로 쓰는 설계는 간결하고 효율적이지만, 클래스 내부의 multi-modal structure를 충분히 반영하지 못할 수 있다. 예를 들어 `person` 클래스는 자세, 크기, 가림 정도에 따라 내부 다양성이 큰데, 단일 mean이 이를 모두 대표하는 데는 한계가 있다. 이 점은 논문에서 명시적으로 문제 삼지는 않지만, 방법론의 구조상 자연스럽게 제기되는 한계다.

넷째, 실험은 주로 DeepLabV3+ 기반에서 수행되었고, 다른 최신 segmentation backbone이나 transformer 기반 구조에 대한 분석은 이 논문 범위 밖이다. 따라서 “model-agnostic”이라는 주장은 방향성으로는 타당하지만, 본문 실험만으로 완전히 일반화되었다고 보기는 어렵다.

## 6. 결론

이 논문은 semantic segmentation을 위한 효율적인 pixel-level contrastive learning 프레임워크인 ReCo를 제안한다. 핵심 기여는 같은 클래스의 픽셀 표현을 class mean 쪽으로 모으고, query 클래스와 헷갈리기 쉬운 negative class를 더 적극적으로 밀어내는 contrastive objective를, hard query와 adaptive negative sampling으로 메모리 효율적으로 구현한 데 있다.

실험적으로 ReCo는 supervised와 semi-supervised segmentation 모두에서 일관된 향상을 보였고, 특히 라벨이 매우 적은 few-label regime에서 효과가 컸다. 또한 더 선명한 boundary와 더 나은 rare class 예측을 보여 주어, 단순한 지표 개선 이상의 실질적 장점을 입증했다.

실제 적용 측면에서도 의미가 있다. segmentation annotation 비용이 큰 자율주행, 로보틱스, 실내 장면 이해 같은 환경에서, ReCo는 기존 모델 위에 비교적 간단히 붙여 label efficiency를 높일 수 있는 방법이다. 향후에는 더 복잡한 class structure를 반영하는 positive 설계, video segmentation이나 video representation learning으로의 확장, 혹은 transformer 기반 dense prediction 모델과의 결합이 자연스러운 후속 연구 방향으로 보인다.
