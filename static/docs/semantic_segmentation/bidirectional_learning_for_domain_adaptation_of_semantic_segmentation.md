# Bidirectional Learning for Domain Adaptation of Semantic Segmentation

- **저자**: Yunsheng Li, Lu Yuan, Nuno Vasconcelos
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1904.10620

## 1. 논문 개요

이 논문은 semantic segmentation에서 synthetic source domain의 라벨 정보를 이용해, 라벨이 없는 real target domain으로 성능을 옮기는 unsupervised domain adaptation 문제를 다룬다. 구체적으로는 GTA5나 SYNTHIA처럼 자동으로 pixel-level annotation을 만들 수 있는 합성 데이터로 학습한 모델이, Cityscapes 같은 실제 도로 장면 데이터에서도 잘 동작하도록 만드는 것이 목표이다.

이 문제가 중요한 이유는 semantic segmentation은 픽셀 단위 라벨링이 필요해서 데이터 구축 비용이 매우 크기 때문이다. 합성 데이터는 저렴하게 많이 만들 수 있지만, 실제 이미지와는 조명, 질감, 스케일, 외형 통계가 달라서 그대로 학습하면 target domain에서 성능이 크게 떨어진다. 따라서 source와 target 사이의 domain gap을 줄이는 것이 핵심이다.

저자들은 기존 방법이 대체로 두 단계로 분리된다고 본다. 먼저 image-to-image translation으로 source 이미지를 target처럼 보이게 바꾸고, 그 다음 segmentation network에서 feature-level adaptation을 수행한다. 하지만 이 방식은 translation 품질이 나쁘면 뒤 단계가 그 실패를 복구하기 어렵다는 한계가 있다. 이 논문은 이 점을 해결하기 위해 translation model과 segmentation adaptation model이 서로를 반복적으로 개선하는 bidirectional learning framework를 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 image translation 모델 $F$와 segmentation adaptation 모델 $M$을 순차적으로 한 번만 학습하지 않고, 닫힌 고리(closed loop) 안에서 번갈아 업데이트하는 것이다. 저자들은 이를 두 방향의 학습으로 설명한다.

첫 번째는 forward direction, 즉 “translation-to-segmentation”이다. source 이미지를 target 스타일로 번역한 뒤, 그 결과를 이용해 segmentation adaptation model을 학습한다. 이는 기존 CyCADA나 DCAN 계열과 유사한 방향이다.

두 번째는 backward direction, 즉 “segmentation-to-translation”이다. 여기서는 이미 개선된 segmentation model이 다시 translation model을 더 좋게 만들도록 사용된다. 이를 위해 저자들은 segmentation network의 출력을 이용한 perceptual loss를 새로 도입한다. 이 loss는 번역 전후 이미지가 같은 semantic meaning을 유지하도록 강제한다. 즉, 외형은 target처럼 바뀌더라도 객체의 의미와 구조는 유지되어야 한다는 것이다.

또 하나의 핵심은 self-supervised learning, 정확히는 pseudo-label 기반 self-training을 segmentation adaptation에 결합한 점이다. target 이미지에서 모델이 높은 confidence를 보이는 픽셀만 pseudo label로 채택하고, 이들만 다시 segmentation loss로 학습한다. 저자들의 주장에 따르면 이렇게 하면 이미 잘 정렬된 영역은 supervised하게 고정되고, adversarial learning은 아직 정렬되지 않은 나머지 영역에 더 집중하게 된다.

기존 접근과의 차별점은, translation과 segmentation을 독립적인 순차 단계로 두지 않고 서로의 개선이 서로에게 되먹임되도록 설계했다는 점이다. 또한 self-supervised learning이 단독으로 쓰인 것이 아니라 adversarial adaptation 및 translation feedback과 결합된다는 점도 차별점이다.

## 3. 상세 방법 설명

전체 시스템은 크게 두 모듈로 구성된다.

첫째는 image-to-image translation model $F$이다. 이 모듈은 source image $S$를 target-like image $S'$로 바꾸고, 반대로 target image $T$를 source-like image $T'$로 바꾸는 양방향 변환을 수행한다. 구조적으로는 CycleGAN을 기반으로 하며, GAN loss와 reconstruction loss를 사용한다.

둘째는 segmentation adaptation model $M$이다. 이 모듈은 semantic segmentation network이며, translated source image $S'$와 target image $T$를 입력받아 domain gap을 줄이도록 학습된다. backbone으로는 DeepLab V2 with ResNet101 또는 FCN-8s with VGG16을 사용한다.

### 3.1 Forward direction: $F \rightarrow M$

먼저 $F$를 이용해 source 이미지를 target 스타일로 번역하여 $S' = F(S)$를 얻는다. 번역은 appearance만 바꾸고 semantic label은 유지된다고 가정하므로, $S'$는 여전히 source label $Y_S$를 사용할 수 있다.

이후 segmentation model $M$은 다음 loss로 학습된다.

$$
\mathcal{L}_M = \lambda_{adv}\mathcal{L}_{adv}(M(S'), M(T)) + \mathcal{L}_{seg}(M(S'), Y_S)
$$

여기서 $\mathcal{L}_{adv}$는 segmentation output 또는 feature representation 수준에서 source-like prediction과 target prediction의 분포 차이를 줄이기 위한 adversarial loss이다. $\mathcal{L}_{seg}$는 translated source에 대한 supervised segmentation loss이다. 즉, target에는 정답이 없기 때문에 처음에는 source 쪽만 직접 supervision이 가능하다.

### 3.2 Backward direction: $M \rightarrow F$

이 논문의 중요한 새 요소는 segmentation model이 translation model을 다시 개선한다는 점이다. 저자들은 일반적인 perceptual loss가 object recognition용 pretrained network의 feature를 쓰는 것과 달리, segmentation adaptation model $M$의 출력을 semantic feature처럼 사용한다.

translation model의 전체 loss는 다음과 같다.

$$
\mathcal{L}_F =
\lambda_{GAN}\big[\mathcal{L}_{GAN}(S', T) + \mathcal{L}_{GAN}(S, T')\big]
+ \lambda_{recon}\big[\mathcal{L}_{recon}(S, F^{-1}(S')) + \mathcal{L}_{recon}(T, F(T'))\big]
+ \mathcal{L}_{per}(M(S), M(S'))
+ \mathcal{L}_{per}(M(T), M(T'))
$$

이 식은 세 종류의 제약을 동시에 넣는다.

첫째, GAN loss는 번역된 이미지 분포가 상대 도메인 분포와 비슷해지게 만든다. 예를 들어 $S'$는 target 이미지처럼 보여야 한다.

둘째, reconstruction loss는 cycle consistency를 유지한다. 즉, $S \rightarrow S' \rightarrow S$로 되돌렸을 때 원본과 비슷해야 한다. 논문에서는 $L_1$ norm을 사용한다.

셋째, perceptual loss는 semantic consistency를 유지한다. 예를 들어 source image와 translated source image는 스타일은 달라도 같은 segmentation 구조를 가져야 하므로, $M(I_S)$와 $M(I'_S)$가 유사해야 한다. 구체식은 다음과 같다.

$$
\mathcal{L}_{per}(M(S), M(S')) =
\lambda_{per}\,\mathbb{E}_{I_S \sim S}\|M(I_S)-M(I'_S)\|_1
+ \lambda_{per\_recon}\,\mathbb{E}_{I_S \sim S}\|M(F^{-1}(I'_S)) - M(I_S)\|_1
$$

두 번째 항은 reconstruction된 이미지까지 semantic consistency를 유지하게 해서 학습을 더 안정화하려는 목적이다. target 쪽도 대칭적으로 같은 항을 둔다.

직관적으로 보면, segmentation model이 “이 픽셀은 road, building, car 같은 의미를 유지해야 한다”는 신호를 translation model에 제공하는 셈이다. 따라서 translation이 단순히 target 스타일로만 보이는 것이 아니라 segmentation에 유리한 방식으로 semantic structure를 보존하도록 학습된다.

### 3.3 Self-supervised learning for $M$

target에는 정답 라벨이 없지만, 모델이 예측한 결과 중 confidence가 높은 픽셀은 pseudo label로 활용할 수 있다. 이를 반영한 수정된 segmentation loss는 다음과 같다.

$$
\mathcal{L}_M =
\lambda_{adv}\mathcal{L}_{adv}(M(S'), M(T))
+ \mathcal{L}_{seg}(M(S'), Y_S)
+ \mathcal{L}_{seg}(M(T_{ssl}), \hat{Y}_T)
$$

여기서 $T_{ssl} \subset T$는 pseudo label을 사용할 수 있는 target subset이고, $\hat{Y}_T$는 pseudo label이다.

pseudo label은 각 픽셀에 대해 가장 큰 class probability를 택해 만든다.

$$
\hat{y}_T = \arg\max M(I_T)
$$

다만 모든 픽셀을 다 쓰지 않고, 최대 확률이 threshold보다 큰 픽셀만 선택한다. 이를 mask map으로 쓰면

$$
m_T = 1[\arg\max M(I_T) > threshold]
$$

이 된다. 최종적으로 target에 대한 segmentation loss는 mask가 1인 픽셀만 포함한다. 즉, 불확실한 예측은 버리고, 확실한 예측만 self-training에 사용한다.

source에 대한 segmentation loss는 일반적인 pixel-wise cross entropy이다.

$$
\mathcal{L}_{seg}(M(S'), Y_S)
=
-\frac{1}{HW}\sum_{H,W}\sum_{c=1}^{C}
1[c = y^S_{hw}]\log P^S_{hwc}
$$

target도 유사하지만 mask $m_T$가 곱해진다. 이 방식은 pseudo label noise를 줄이기 위한 설계이다.

### 3.4 학습 절차

논문은 두 개의 반복 구조를 사용한다.

외부 loop는 bidirectional learning을 담당한다. 즉,
1. 현재의 segmentation model을 이용해 translation model $F^{(k)}$를 학습하고,
2. 그 translation 결과로 segmentation model $M^{(k)}_0$를 학습한다.

내부 loop는 SSL을 담당한다. 즉,
1. 현재 segmentation model로 target pseudo label을 갱신하고,
2. 이를 이용해 segmentation model을 다시 학습한다.

논문의 Algorithm 1에 따르면, 바깥 반복 횟수를 $K$, SSL 내부 반복 횟수를 $N$으로 두고 번갈아 학습한다. 저자들은 실험적으로 $N=2$가 적절하다고 보고한다.

## 4. 실험 및 결과

### 데이터셋과 설정

실험은 synthetic-to-real urban scene segmentation에 맞춰 수행되었다.

source dataset으로는 GTA5와 SYNTHIA를 사용했다.
- GTA5: 24,966장, 해상도 $1914 \times 1052$, Cityscapes와 공통 19개 클래스 사용
- SYNTHIA-RAND-CITYSCAPES: 9,400장, 해상도 $1280 \times 760$, Cityscapes와 공통 16개 클래스 사용

target dataset은 Cityscapes이다.
- training set 2,975장을 adaptation용 target unlabeled set으로 사용
- validation set 500장을 test set처럼 사용
- testing set은 ground truth가 없어서 평가에 쓰지 못했다고 명시한다

segmentation backbone은 두 가지이다.
- DeepLab V2 + ResNet101
- FCN-8s + VGG16

translation model은 9-block CycleGAN을 사용한다. discriminator 구조와 optimizer, learning rate schedule, loss weight도 상세히 제시되어 있다. 예를 들어 translation에서는 $\lambda_{GAN}=1$, $\lambda_{recon}=10$, $\lambda_{per}=0.1$, $\lambda_{per\_recon}=10$을 사용했다. segmentation adaptation에서는 $\lambda_{adv}=0.001$ for ResNet101, $10^{-4}$ for FCN-8s를 사용했다.

### 4.1 Ablation: bidirectional learning 자체의 효과

Table 1에서 GTA5 $\rightarrow$ Cityscapes, DeepLab V2 + ResNet101 기준 결과는 다음과 같다.

- source only baseline $M^{(0)}$: 33.6 mIoU
- adversarial adaptation만 적용한 $M^{(1)}$: 40.9
- translation 후 segmentation만 한 $M^{(0)}(F^{(1)})$: 41.1
- 둘을 결합한 $M^{(1)}_0(F^{(1)})$: 42.7
- 한 번 더 반복한 $M^{(2)}_0(F^{(2)})$: 43.3

이 결과는 translation과 segmentation adaptation이 각자 독립적으로도 baseline보다 약 7% 이상 개선하지만, 둘을 결합하면 더 좋아지고, 한 번 더 bidirectional iteration을 수행하면 추가 개선이 있음을 보여준다. 즉, 상호 보완 효과가 존재한다는 것이 저자들의 주장이다.

### 4.2 Ablation: self-supervised learning의 효과

SSL을 추가하면 성능이 더 크게 오른다. GTA5 $\rightarrow$ Cityscapes, ResNet101 기준 Table 2에서

- $M^{(1)}_0(F^{(1)})$: 42.7
- $M^{(1)}_1(F^{(1)})$: 46.8
- $M^{(1)}_2(F^{(1)})$: 47.2

즉, 첫 번째 outer iteration 안에서 SSL만으로 4.5 mIoU 정도의 추가 향상이 생긴다. 두 번째 outer iteration에서도

- $M^{(2)}_0(F^{(2)})$: 44.3
- $M^{(2)}_1(F^{(2)})$: 47.6
- $M^{(2)}_2(F^{(2)})$: 48.5

까지 오른다.

저자들은 Figure 4를 통해 confidence threshold 0.9에서 선택되는 white mask 영역이 iteration이 진행될수록 커진다고 설명한다. 이는 segmentation model이 더 많은 target 픽셀에 대해 확신 있는 예측을 하게 되었고, 그래서 SSL에 더 많은 pseudo label을 사용할 수 있게 되었음을 뜻한다.

### 4.3 Hyperparameter 분석

threshold 선택은 pseudo label의 양과 질 사이의 trade-off 문제다. threshold가 낮으면 많은 픽셀을 쓰지만 noise가 늘고, 높으면 quality는 좋아지지만 사용할 픽셀이 줄어든다.

Figure 5와 Table 3을 바탕으로 저자들은 0.9를 최적 threshold로 선택했다.
- 0.95: 45.7
- 0.9: 46.8
- 0.8: 46.4
- 0.7: 45.9
- soft threshold: 44.9

즉, hard threshold 0.9가 가장 좋았다. soft weighting은 noise 영향을 충분히 완화하지 못한 것으로 해석한다.

SSL iteration 수 $N$에 대해서는 Table 4에서
- $N=0$에 해당하는 $M^{(1)}_0(F^{(1)})$: 42.7
- $N=1$: 46.8
- $N=2$: 47.2
- $N=3$: 47.1

이므로 $N=2$ 정도에서 거의 수렴한다고 본다. pixel ratio도 79%에서 81% 정도로 증가한 뒤 정체된다.

### 4.4 State-of-the-art 비교

#### GTA5 $\rightarrow$ Cityscapes

ResNet101 기준 Table 5에서 제안법은 48.5 mIoU를 기록했다. 주요 비교 대상은 다음과 같다.

- CyCADA: 42.7
- AdaptSegNet: 41.4
- DCAN: 41.7
- CLAN: 43.2
- Ours: 48.5

즉, 최고 경쟁 방법 대비 약 5.3 mIoU, CyCADA 대비는 5.8, DCAN 대비는 6.8 정도 높다. 논문 본문에서는 “about 6% improvement”라고 요약한다.

VGG16 기준에서도
- Curriculum: 28.9
- CBST: 30.9
- CyCADA: 35.4
- DCAN: 36.2
- CLAN: 36.6
- Ours: 41.3

으로 가장 높은 성능을 얻는다.

#### SYNTHIA $\rightarrow$ Cityscapes

ResNet101 기준 Table 6에서
- AdaptSegNet: 45.9
- CLAN: 47.8
- Ours: 51.4

VGG16 기준에서는
- FCN wild: 20.2
- Curriculum: 29.0
- CBST: 35.4
- DCAN: 35.4
- Ours: 39.0

이다.

저자들은 SYNTHIA와 Cityscapes 사이의 domain gap이 GTA5보다 더 커서 adaptation이 전반적으로 더 어렵고, 그 때문에 SSL confidence도 더 낮아질 수 있다고 해석한다. 그럼에도 여전히 기존 방법보다 최소 4% 정도 높은 성능을 얻는다고 주장한다.

### 4.5 Upper bound와의 격차

완전 감독 학습으로 target label을 사용한 oracle 성능도 제시한다.

- GTA5 $\rightarrow$ Cityscapes
  - ResNet101 upper bound: 65.1
  - VGG16 upper bound: 60.3
- SYNTHIA $\rightarrow$ Cityscapes
  - ResNet101 upper bound: 71.7
  - VGG16 upper bound: 59.5

제안법은 SOTA를 갱신했지만 oracle과는 여전히 최소 16.6 mIoU 이상의 차이가 있다. 저자들 스스로도 아직 성능 개선 여지가 크다고 인정한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation pipeline의 두 핵심 모듈, 즉 image translation과 segmentation adaptation을 서로 독립적인 전처리와 후처리처럼 보지 않고 상호작용하는 시스템으로 재구성했다는 점이다. 단순히 “source를 target처럼 바꾼 뒤 segmentation”이라는 일방향 구조에서 벗어나, segmentation model이 translation model을 semantic하게 규제하도록 만든 설계가 논문의 핵심 공헌이다.

또한 self-supervised learning을 adversarial adaptation과 결합한 점도 설득력이 있다. pseudo label을 모든 픽셀에 무작정 쓰지 않고, max probability threshold를 둬서 신뢰도 높은 픽셀만 활용하는 방식은 매우 직관적이며, ablation 결과도 실제 성능 향상을 뒷받침한다. 특히 첫 iteration에서 42.7에서 47.2까지 오른 결과는 SSL이 실질적으로 기여한다는 강한 증거다.

실험 측면에서도 강점이 있다. GTA5와 SYNTHIA 두 source dataset, Cityscapes target, ResNet101과 VGG16 두 backbone, 그리고 threshold와 iteration 수에 대한 ablation까지 포함해 비교적 폭넓게 검증했다. 단순히 최종 수치만 제시한 것이 아니라, 왜 0.9 threshold와 $N=2$를 쓰는지 데이터 기반으로 설명한 점도 좋다.

반면 한계도 분명하다. 첫째, 성능 향상의 원인이 여러 요소가 강하게 결합되어 있어서 각 성분의 독립적 기여도를 완전히 분리해 보기는 어렵다. 예를 들어 perceptual loss가 translation 품질에 정확히 어떤 시각적 변화를 유도하는지, 혹은 adversarial alignment와 SSL의 상대 기여가 어느 정도인지에 대한 더 세밀한 분석은 부족하다.

둘째, pseudo label 기반 SSL은 confidence calibration이 나쁘면 오히려 error reinforcement를 일으킬 수 있다. 논문은 threshold 0.9를 실험적으로 정했지만, 클래스 불균형이나 희귀 클래스에 대해서는 이 전략이 얼마나 안정적인지 충분히 분석하지 않았다. 실제로 표를 보면 일부 클래스, 예를 들어 train, motorbike 같은 클래스는 여전히 성능이 매우 낮거나 들쭉날쭉하다.

셋째, oracle과의 격차가 여전히 크다. GTA5 $\rightarrow$ Cityscapes, ResNet101에서도 48.5 대 65.1이므로 실제 적용 관점에서는 여전히 상당한 domain mismatch가 남아 있다. 논문의 기여는 “기존보다 낫다”는 데에는 분명하지만, 문제를 근본적으로 해결했다고 보기는 어렵다.

넷째, 논문에 제공된 텍스트 기준으로는 계산 비용이나 학습 시간, bidirectional iteration을 여러 번 돌렸을 때의 효율성 문제는 충분히 논의되지 않았다. translation model과 segmentation model을 반복적으로 갱신하는 구조는 실용적으로는 비용이 큰 편일 수 있다. 이 부분은 명확히 서술되지 않았다.

비판적으로 보면, 이 방법은 translation quality가 segmentation에 중요하다는 기존 직관을 더 잘 이용한 방법이라고 볼 수 있다. 다만 본질적으로는 pseudo-label self-training과 cycle-consistent translation, adversarial adaptation을 결합한 복합 시스템이므로, 각 구성요소의 안정적 튜닝이 성능에 크게 영향을 줄 가능성이 높다. 즉, 개념적으로는 강하지만 실제 재현성과 하이퍼파라미터 민감도는 추가 검증이 필요해 보인다.

## 6. 결론

이 논문은 semantic segmentation의 unsupervised domain adaptation 문제에서, image translation 모델과 segmentation adaptation 모델을 서로 반복적으로 개선하는 bidirectional learning framework를 제안한다. 여기에 target pseudo label을 사용하는 self-supervised learning과 segmentation-driven perceptual loss를 결합해, translation과 segmentation이 서로에게 유의미한 피드백을 주도록 설계했다.

실험 결과는 이 아이디어가 단순한 순차 학습보다 확실히 효과적임을 보여준다. GTA5 $\rightarrow$ Cityscapes와 SYNTHIA $\rightarrow$ Cityscapes 모두에서 당시 state-of-the-art를 큰 폭으로 넘었고, backbone이 달라도 일관된 개선을 보였다. 특히 translation이 segmentation을 돕고, 다시 segmentation이 translation을 돕는 폐루프 구조는 이후의 domain adaptation 연구에서도 충분히 확장 가능한 관점이다.

실제 적용 측면에서는 synthetic-to-real segmentation처럼 라벨링 비용이 큰 분야에서 의미가 크다. 향후 연구로는 pseudo label 품질 개선, 클래스별 confidence 제어, 더 강한 translation backbone, 그리고 oracle과의 큰 성능 격차를 줄이기 위한 구조적 개선이 중요할 것으로 보인다.
