# Bootstrapping Semi-supervised Medical Image Segmentation with Anatomical-aware Contrastive Distillation

- **저자**: Chenyu You, Weicheng Dai, Yifei Min, Lawrence Staib, James S. Duncan
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2206.02307

## 1. 논문 개요

이 논문은 의료 영상 분할에서 라벨이 매우 부족한 상황을 다루는 semi-supervised learning(SSL) 문제를 다룬다. 특히 기존 contrastive learning 기반 방법들이 흔히 가정하는 조건, 즉 labeled/unlabeled 데이터의 class distribution이 어느 정도 균형적이라는 가정이 실제 의료 영상에서는 잘 성립하지 않는다는 점을 문제로 삼는다. 실제 임상 데이터는 기관이나 병변 종류에 따라 클래스 불균형이 심하며, 이로 인해 경계가 흐려지거나 드문 class가 잘못 분할되는 문제가 자주 발생한다.

저자들은 이러한 문제를 해결하기 위해 **ACTION**이라는 framework를 제안한다. 이름은 **Anatomical-aware Contrastive Distillation**의 약자이다. 핵심 목적은 단순히 이미지 단위에서 representation을 맞추는 것이 아니라, 의료 영상의 해부학적 구조와 클래스 불균형을 함께 고려하면서 unlabeled data를 더 효과적으로 활용하는 것이다.

이 문제가 중요한 이유는 의료 영상 분할에서 pixel-level annotation이 매우 비싸고 전문 지식이 필요하기 때문이다. 충분한 라벨을 확보하기 어려운 상황에서 SSL이 현실적으로 매우 중요하지만, 기존 방법이 실제 의료 데이터의 불균형성과 해부학적 구조를 충분히 반영하지 못하면 임상 적용 가능성이 떨어진다. 이 논문은 바로 그 간극을 메우려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 두 가지 관찰에서 출발한다.

첫째, **모든 negative sample이 똑같이 negative는 아니다**. 기존 contrastive learning은 positive와 negative를 거의 이진적으로 구분하는 경우가 많다. 하지만 의료 영상에서는 서로 다른 이미지라도 유사한 해부학적 구조를 공유할 수 있기 때문에, 형식상 negative로 들어간 샘플 중에도 실제로는 query와 의미적으로 가까운 경우가 존재한다. 이런 false negative는 representation learning을 해칠 수 있다. 그래서 저자들은 teacher가 negative anchor들에 대해 soft probability distribution을 만들고, student가 그것을 distillation 방식으로 학습하도록 설계했다.

둘째, **의료 영상은 class imbalance가 심하다**. 저자들은 contrastive learning이 이런 imbalance 상황에서도 정말 잘 작동하는지 질문을 던진다. 이 문제를 위해 단순한 global representation learning에 그치지 않고, 데이터셋 전체 수준의 semantic relationship과 픽셀 주변의 local anatomical feature를 함께 학습하도록 구성했다. 그 결과 rare class나 경계 부근의 어려운 픽셀에 대해 더 분별력 있는 표현을 얻게 하려 한다.

기존 접근과의 차별점은 명확하다. 논문에 따르면 ACTION은 다음 점에서 기존 GCL, SCS류와 다르다.

첫째, student에 predictor를 추가하고 teacher를 EMA 기반 slow-moving average model로 두어 BYOL 스타일의 안정적인 distillation 구조를 만든다.  
둘째, logits 대신 **SoftMax probability**를 사용해 teacher-student 간 similarity distribution을 맞춘다.  
셋째, 같은 이미지의 두 augmentation만 비교하는 것이 아니라, **randomly chosen images를 anchor로 사용**하여 더 다양한 semantic/anatomical relation을 학습한다.  
넷째, 최종 fine-tuning 단계에서 **AnCo(Anatomical Contrast)**라는 anatomical-level contrastive loss를 도입해 hard pixel을 집중적으로 학습한다.

## 3. 상세 방법 설명

전체 구조는 3단계 학습 절차로 이루어진다.

1. **Global Contrastive Distillation Pre-training**
2. **Local Contrastive Distillation Pre-training**
3. **Anatomical Contrast Fine-tuning**

기본 backbone은 2D U-Net이며, encoder $E(\cdot)$와 decoder $D(\cdot)$로 구성된다. 전체 segmentation network를 $F(\cdot)$로 표기한다. ACTION은 BYOL 파이프라인을 기반으로 하지만, segmentation task에 맞게 global/local/anatomical contrast를 단계적으로 쌓아 올린다.

### Global Contrastive Distillation Pre-training

첫 번째 단계의 목적은 unlabeled data로부터 **global-level feature**를 학습하는 것이다. 입력 query image $q$에 두 가지 augmentation을 적용해 teacher 입력 $q_t$와 student 입력 $q_s$를 만든다. 동시에 unlabeled set에서 임의의 이미지들을 여러 개 뽑아 anchor set $\{x_j\}_{j=1}^n$를 만든다. 이 anchor는 같은 이미지의 augmentation이 아니라 다른 이미지들이므로, 더 다양한 semantic relationship을 제공한다.

teacher encoder와 projection head를 거쳐 query의 embedding $z_t$와 anchor embedding $a_j$들을 얻는다. 이후 query와 각 anchor 사이의 cosine similarity를 계산하고, temperature $\tau_t$를 사용해 teacher 확률분포를 만든다.

$$
p_t(j)=\frac{-\log \exp(\mathrm{sim}(z_t,a_j)/\tau_t)}{\sum_{i=1}^{n}\exp(\mathrm{sim}(z_t,a_i)/\tau_t)}
$$

student 쪽도 비슷하게 처리하되, representation collapse를 피하기 위해 predictor $H_{gp}$를 통과시킨 embedding $z_s^*$를 사용한다. student 분포는 다음과 같다.

$$
p_s(j)=\frac{-\log \exp(\mathrm{sim}(z_s^*,a_j)/\tau_s)}{\sum_{i=1}^{n}\exp(\mathrm{sim}(z_s^*,a_i)/\tau_s)}
$$

그 다음 teacher 분포와 student 분포 사이의 KL divergence를 최소화한다.

$$
L_{\mathrm{contrast}} = KL(p_t \Vert p_s)
$$

이 손실의 의미는 단순히 “이 둘은 positive, 저 둘은 negative”라고 강하게 나누는 것이 아니라, query가 여러 anchor와 맺는 **상대적 유사도 구조 자체**를 student가 모사하도록 만드는 것이다. 저자들이 말하는 “soft labeling on negatives”가 바로 이 부분이다.

### Local Contrastive Distillation Pre-training

두 번째 단계는 global feature만으로는 부족한 segmentation 문제의 특성을 반영하기 위해, encoder뿐 아니라 decoder까지 포함하여 **pixel-level local feature**를 학습하는 단계이다. 여기서는 labeled data와 unlabeled data를 모두 사용한다.

labeled data에 대해서는 supervised segmentation loss를 사용한다. 논문에 따르면 supervised loss는 **cross-entropy loss와 dice loss의 선형 결합**이다.

unlabeled data에 대해서는 global 단계와 유사한 teacher-student distillation을 local feature 수준에서 수행한다. query image에 augmentation을 적용하여 teacher와 student에 넣고, 출력 feature $f_t, f_s$를 local projection head에 통과시킨 뒤, student는 다시 predictor를 지난 representation으로 teacher와의 consistency를 맞춘다. 이때도 random sampled images를 함께 사용해 diversity를 높인다.

즉, 1단계가 이미지 전역 수준의 representation 정렬이라면, 2단계는 segmentation에 직접 중요한 **local structure와 pixel neighborhood 정보**를 더 잘 반영하도록 teacher-student contrastive distillation을 확장한 것이다.

### Anatomical Contrast Fine-tuning

세 번째 단계가 이 논문의 가장 중요한 부분이다. 저자들은 의료 영상에서 같은 조직은 서로 유사한 해부학적 특징을 공유하지만, 다른 조직은 class, appearance, spatial distribution이 다르다고 본다. 이 정보를 이용하기 위해 **AnCo**라는 새로운 unsupervised anatomical contrastive loss를 도입한다.

이를 위해 student network에 segmentation head와 병렬로 **representation decoder head $H_r$**를 추가한다. 이 head는 multi-layer hidden feature를 up-sampling하여 입력과 같은 spatial resolution의 dense feature로 만든 뒤, 각 픽셀 표현을 고차원 embedding으로 매핑한다. 여기서 query embedding $r_q$, positive key $r_k^+$, negative key $r_k^-$가 정의된다.

AnCo loss는 다음과 같다.

$$
L_{\mathrm{anco}}=
\sum_{c \in C}\sum_{r_q \sim R_q^c}
-\log
\frac{\exp(r_q \cdot r_k^{c,+}/\tau_{an})}
{\exp(r_q \cdot r_k^{c,+}/\tau_{an})+\sum_{r_k^- \sim R_k^c}\exp(r_q \cdot r_k^-/\tau_{an})}
$$

여기서 class $c$에 대해:

- $R_q^c$는 class $c$에 속하는 query embedding 집합
- $R_k^c$는 class $c$가 아닌 negative key embedding 집합
- $r_k^{c,+}$는 class $c$의 평균 표현, 즉 positive prototype

이다. 식 (5)는 이를 더 구체적으로 정의한다.

$$
R_q^c=\bigcup_{[m,n]\in P}\mathbf{1}(y^{[m,n]}=c)\,r^{[m,n]}
$$

$$
R_k^c=\bigcup_{[m,n]\in P}\mathbf{1}(y^{[m,n]}\neq c)\,r^{[m,n]}
$$

$$
r_k^{c,+}=\frac{1}{|R_q^c|}\sum_{r_q \in R_q^c} r_q
$$

직관적으로 말하면, 각 픽셀 query를 자기 class의 평균 표현에 가깝게 만들고, 다른 class의 표현과는 멀어지게 하는 것이다. 이 방식은 같은 class 내부의 일관성을 높이고 class 간 경계를 더 선명하게 만들도록 유도한다.

### Active Hard Sampling

contrastive learning은 보통 많은 positive/negative pair가 필요하지만, GPU memory 제약이 크다. 이를 해결하기 위해 저자들은 두 가지 active sampling을 넣는다.

첫째, **hard negative sampling**이다. 각 class prototype 간의 관계를 그래프 $G$로 계산한다.

$$
G[p,q] = r_k^{p,+}\cdot r_k^{q,+}, \quad p,q \in C,\ p\neq q
$$

이 값이 크면 두 class가 해부학적 또는 의미적으로 더 비슷하다는 뜻이다. 따라서 어떤 query class $c$에 대해, 모든 negative class를 동일하게 다루지 않고, $G[c,v]$를 SoftMax로 정규화한 분포에 따라 더 어려운 negative class에서 더 많이 sample한다. 이것은 “헷갈리기 쉬운 class를 더 자주 보여주는” 전략이다.

둘째, **hard query sampling**이다. rare class와 어려운 픽셀을 더 잘 배우기 위해, 예측 confidence $\hat{y}_q$와 threshold $\theta_s$를 사용해 easy query와 hard query를 나눈다.

$$
R_{q,\mathrm{easy}}^c=\bigcup_{r_q \in R_q^c}\mathbf{1}(\hat{y}_q>\theta_s)r_q
$$

$$
R_{q,\mathrm{hard}}^c=\bigcup_{r_q \in R_q^c}\mathbf{1}(\hat{y}_q\le \theta_s)r_q
$$

confidence가 낮은 픽셀일수록 경계나 소수 class에 속할 가능성이 높으므로, 이런 픽셀을 적극적으로 학습해 decision boundary를 더 정확하게 만든다.

### 학습 절차 요약

실제 학습은 다음처럼 진행된다.

첫째, Stage-i에서 unlabeled data만으로 global contrastive distillation을 학습한다.  
둘째, Stage-ii에서 labeled와 unlabeled data를 함께 사용해 local contrastive distillation과 supervised segmentation을 학습한다.  
셋째, Stage-iii에서 supervised segmentation loss, pseudo-label 기반 unsupervised cross-entropy loss, 그리고 $L_{\mathrm{anco}}$를 함께 사용해 fine-tuning한다.

논문에 따르면 inference 시에는 projection heads, predictor, representation decoder head는 제거되고 segmentation network만 사용된다.

## 4. 실험 및 결과

### 데이터셋과 설정

논문은 두 개의 benchmark dataset에서 실험한다.

첫째, **ACDC 2017**이다. 심장 cine MRI 200개 scan, 100명 환자 데이터이며, segmentation class는 LV, Myo, RV의 3개이다. 학습/검증/테스트는 140/20/60 scan으로 나눈다.

둘째, **LiTS**이다. contrast-enhanced 3D abdominal CT 131개 volume이며, class는 liver와 tumor의 2개이다. 첫 100개 volume을 학습, 나머지 31개를 테스트에 사용한다.

전처리는 intensity normalization 후, 모든 2D slice와 segmentation map을 $256\times256$ 해상도로 resample한다. 평가 지표는 3D segmentation 결과에 대해 **Dice coefficient (DSC)**와 **Average Surface Distance (ASD)**를 사용한다.

### 구현 세부 사항

모델은 PyTorch로 구현되었고, SGD optimizer를 사용한다. learning rate는 0.01, momentum은 0.9, weight decay는 0.0001, batch size는 6이다. GPU는 RTX 3090 두 장을 사용했다. Stage-i와 Stage-ii는 각각 100 epoch, Stage-iii는 200 epoch 학습한다. teacher/student temperature는 $\tau_t=0.01$, $\tau_s=0.1$로 두고, teacher는 momentum update $ \theta_t \leftarrow m\theta_t + (1-m)\theta_s $를 사용하며 $m=0.99$이다. memory bank size는 36이다.

Stage-iii에서는 mini-batch마다 query 256개, key 512개를 adaptively sample하고, student temperature는 0.5, confidence threshold는 $\theta_s=0.97$로 설정한다.

### ACDC 결과

ACDC에서 3 labeled와 7 labeled 두 설정 모두에서 ACTION이 기존 SSL 방법들을 크게 능가한다.

3 labeled 설정에서 평균 Dice/ASD는 다음과 같다.

- GCL: 70.6 / 2.24
- SCS: 73.6 / 5.37
- **ACTION: 87.5 / 1.12**

즉, 기존 최고 평균 Dice 73.6%를 87.5%까지 크게 끌어올렸다. 논문은 이것이 7 labeled를 사용한 기존 SSL 성능에 필적하거나 능가한다고 강조한다.

7 labeled 설정에서도:

- GCL: 87.0 / 0.751
- SCS: 84.2 / 2.01
- **ACTION: 89.7 / 0.736**

으로 최고 성능을 기록한다.

클래스별로 보면 RV와 Myo에서 향상이 특히 크다. 이는 class imbalance와 경계 문제를 더 잘 다룬다는 논문의 주장과 잘 맞는다. Figure 3에 따르면 시각적으로도 ACTION은 RV, Myo 등의 경계가 더 날카롭고 정확하다.

### LiTS 결과

LiTS에서도 5% labeled와 10% labeled 설정 모두에서 가장 좋은 결과를 낸다.

5% labeled에서 평균 Dice/ASD는:

- GCL: 63.3 / 20.1
- SCS: 61.5 / 28.8
- **ACTION: 66.8 / 17.7**

10% labeled에서:

- GCL: 65.0 / 37.2
- SCS: 64.6 / 33.9
- **ACTION: 67.7 / 20.4**

특히 tumor class에서 향상이 중요하다. 5% labeled에서 tumor Dice가 40.5로, GCL의 35.9와 SCS의 30.4보다 높다. tumor는 보통 liver보다 훨씬 어렵고 불균형이 심한 class이므로, 이 결과는 ACTION의 imbalance 대응이 실제로 의미 있음을 보여준다.

### Ablation Study

ACDC 3 labeled 설정에서 다양한 구성요소의 기여를 분석했다.

기본 Vanilla 모델은 Dice 60.6, ASD 6.64이다. 반면 전체 ACTION은 Dice 87.5, ASD 1.12이다. 중간 ablation 결과를 보면:

- `w/o RSI`: Dice 82.7, ASD 6.66
- `w/o Stage-ii`: Dice 86.4, ASD 1.69
- `w/o RSI + Stage-ii`: Dice 82.6, ASD 1.77
- `w/o Stage-iii`: Dice 76.7, ASD 2.91
- `w/o L_anco`: Dice 86.5, ASD 1.30
- `w/o L_unsup`: Dice 83.7, ASD 2.51

이 결과는 random sampled images, local contrastive distillation, anatomical contrast fine-tuning, pseudo-label 기반 unsupervised loss가 모두 성능에 기여함을 보여준다. 특히 Stage-iii 제거 시 성능 하락이 큰 점은, 단순 pre-training보다 anatomical contrast fine-tuning이 핵심이라는 점을 뒷받침한다.

### Augmentation Study

teacher/student에 weak 또는 strong augmentation을 다르게 적용한 실험도 수행했다. 가장 좋은 조합은 **teacher에 weak augmentation, student에 strong augmentation**이다. 이 설정에서 Dice 87.5, ASD 1.12를 얻었다. 이는 teacher가 안정적인 target을 제공하고 student가 더 강한 변형에 견디는 representation을 배우는 것이 효과적임을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 contrastive learning을 의료 영상의 실제 문제 구조에 맞게 다시 설계했다는 점이다. 단순히 unlabeled data를 많이 활용한다는 수준이 아니라, **false negative 문제**, **class imbalance**, **해부학적 구조의 유사성**, **memory 효율성**을 하나의 framework 안에서 함께 다룬다. 특히 teacher가 anchor와의 soft similarity distribution을 제공하고 student가 이를 distillation하는 구조는, “negative는 모두 동일하지 않다”는 문제의식을 비교적 직접적으로 반영한다.

또 다른 강점은 방법의 구조가 단계적으로 잘 정리되어 있다는 점이다. global feature 학습, local feature 학습, anatomical contrast fine-tuning이 서로 다른 역할을 수행하며, ablation 결과도 각 단계의 기여를 어느 정도 설득력 있게 보여준다. 또한 ACDC와 LiTS 모두에서 일관된 개선을 보였고, 특히 RV, Myo, tumor처럼 상대적으로 어려운 구조에서 성능 향상이 두드러진다.

실용적인 측면에서도 장점이 있다. 논문은 minimal additional memory footprint를 강조하며, 모든 픽셀을 다 contrastive pair로 쓰지 않고 active sampling으로 query/key를 일부만 선택한다. segmentation task에서 dense contrastive learning의 계산량 문제를 현실적으로 다루려 했다는 점이 긍정적이다.

한계도 분명히 있다. 첫째, 논문이 제안한 3-stage pipeline은 구조가 복잡하며, 학습 과정이 길다. Stage-i, ii, iii를 순차적으로 학습해야 하므로 실제 재현과 튜닝 비용이 낮다고 보기 어렵다. 둘째, 성능 향상은 명확하지만, 어떤 요소가 어떤 데이터셋에서 가장 본질적으로 효과를 냈는지에 대한 분해는 제한적이다. 예를 들어 imbalance 대응과 false negative 완화 중 어느 쪽이 더 핵심 요인인지까지는 본문만으로 완전히 분리되지 않는다.

셋째, 논문은 주로 두 benchmark dataset에서 결과를 제시하며, 더 다양한 modality나 장기/병변 조합, 또는 심한 domain shift가 있는 환경에서의 일반화는 여기 제공된 본문만으로는 확인할 수 없다. 넷째, pseudo-label 기반 unsupervised loss와 AnCo를 함께 사용하기 때문에, confidence threshold $\theta_s$나 sampling 수 같은 hyperparameter에 민감할 가능성이 있다. 다만 본문에서는 이에 대한 민감도 분석은 제한적으로만 제시된다.

비판적으로 보면, 이 방법의 핵심 주장은 “anatomical-aware”인데, 실제 anatomical prior를 명시적 해부학 모델로 넣기보다는 representation similarity와 class prototype 관계로 우회적으로 구현했다. 따라서 이름이 주는 인상만큼 명시적 anatomical modeling이 강한 것은 아니다. 그럼에도 불구하고 의료 영상에서 class 간 구조적 관계를 contrastive objective에 녹였다는 점은 충분히 의미 있다.

## 6. 결론

이 논문은 semi-supervised medical image segmentation에서 contrastive learning의 한계를 정면으로 다룬 연구이다. 저자들은 모든 negative sample을 동일하게 취급하는 기존 방식과, 실제 의료 영상의 심한 class imbalance 문제를 동시에 비판하고, 이를 해결하기 위해 **ACTION**이라는 anatomical-aware contrastive distillation framework를 제안했다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, teacher-student 기반의 **soft contrastive distillation**으로 false negative 문제를 완화했다. 둘째, global뿐 아니라 local feature 수준에서도 distillation을 수행해 segmentation에 필요한 구조 정보를 더 잘 학습했다. 셋째, **AnCo loss와 active hard sampling**을 통해 픽셀 수준에서 어려운 경계와 드문 class를 더 잘 학습하도록 만들었다.

실험적으로도 ACDC와 LiTS에서 기존 state-of-the-art SSL 방법들을 큰 폭으로 앞섰고, 특히 적은 labeled data와 불균형 class 상황에서 효과가 두드러졌다. 따라서 이 연구는 의료 영상처럼 annotation이 비싸고 class imbalance가 심한 환경에서 매우 실용적인 의미를 가진다. 앞으로 더 다양한 modality, 3D setting, domain shift 상황, 또는 더 단순한 학습 절차와 결합된다면, 실제 임상 분할 시스템의 강한 기반 방법으로 발전할 가능성이 크다.
