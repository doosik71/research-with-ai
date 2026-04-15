# ACTION++: Improving Semi-supervised Medical Image Segmentation with Adaptive Anatomical Contrast

- **저자**: Chenyu You, Weicheng Dai, Yifei Min, Lawrence Staib, Jas Sekhon, James S. Duncan
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2304.02689

## 1. 논문 개요

이 논문은 semi-supervised medical image segmentation에서 자주 나타나는 **class imbalance**, 특히 **long-tailed distribution** 문제를 더 잘 다루기 위한 방법인 **ACTION++**를 제안한다. 의료 영상에서는 background나 큰 장기처럼 픽셀 수가 많은 class가 있는 반면, 경계 영역이나 작은 구조물처럼 픽셀 수가 매우 적은 minority class도 많다. 이런 데이터 불균형은 segmentation 모델이 head class에는 잘 맞고 tail class에는 약해지는 문제를 만든다.

기존 semi-supervised segmentation 연구들은 주로 unlabeled data를 활용하는 방법에 집중했고, contrastive learning 기반 방법들도 대부분 unlabeled portion에서 representation을 잘 학습하는 데 초점을 맞췄다. 하지만 저자들은 **labeled data 내부에서도 이미 class distribution이 크게 불균형하다**는 점에 주목한다. 즉, supervised signal 자체도 minority class를 충분히 보호하지 못할 수 있다는 것이다. 따라서 이 논문은 unlabeled data뿐 아니라 labeled data에서도 class separation을 더 균형 있게 만들도록 contrastive learning을 개선하려고 한다.

문제의 중요성은 명확하다. 실제 의료 영상에서 놓치기 쉬운 작은 구조나 경계 부위는 임상적으로 중요한 경우가 많다. 따라서 단순히 평균 성능을 높이는 것보다, 적은 픽셀을 가진 class를 얼마나 안정적으로 구분하느냐가 실제 적용에서 중요하다. 이 논문은 바로 그 지점을 겨냥한다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 두 가지이다.

첫째, **supervised adaptive anatomical contrastive learning (SAACL)**를 도입한다. 기존 anatomical contrast는 주로 unlabeled data의 representation을 안정화하는 역할을 했지만, 이 논문은 labeled data에서도 class center를 더 명확하게 형성해야 한다고 본다. 이를 위해 각 class가 embedding space 안에서 한쪽으로 몰리지 않도록, **사전에 균등하게 퍼진 optimal class centers를 off-line으로 계산**해 둔다. 이후 학습 중에는 각 class의 feature가 그 class에 배정된 center 쪽으로 모이도록 유도한다. 이렇게 하면 head class가 representation space를 과도하게 점유하는 현상을 막고, tail class도 충분한 분리 공간을 확보할 수 있다.

둘째, contrastive loss의 temperature $\tau$를 고정값으로 두는 것이 long-tailed medical data에는 적절하지 않다고 주장한다. 저자들은 큰 $\tau$는 group-wise discrimination을 강화하고, 작은 $\tau$는 instance-level discrimination을 강화한다고 설명한다. 따라서 한 값만 계속 쓰면 둘 중 하나에 치우칠 수 있다. 이를 해결하기 위해 **anatomical-aware temperature scheduler (ATS)**를 제안하며, cosine schedule을 사용해 학습 중 $\tau$를 동적으로 바꾼다. 이 방식은 representation space를 더 isotropic하게 만들고, majority/minority class 모두에 대해 더 나은 분리를 유도한다.

기존 ACTION과의 차별점은 분명하다. ACTION이 anatomical-aware contrastive distillation 구조를 기반으로 했던 반면, ACTION++는 여기에 **labeled data에 대한 adaptive supervised contrast**와 **dynamic temperature scheduling**을 추가해 class imbalance를 더 직접적으로 다룬다.

## 3. 상세 방법 설명

ACTION++는 기본적으로 기존 **ACTION** 프레임워크를 backbone으로 사용한다. 전체 파이프라인은 세 단계로 구성된다.

첫 번째는 **global contrastive distillation pre-training**이고, 두 번째는 **local contrastive distillation pre-training**이며, 세 번째는 **anatomical contrast fine-tuning**이다. student-teacher 구조를 사용하며, teacher의 파라미터는 student의 exponential moving average로 업데이트된다.

### 3.1 기본 ACTION 파이프라인

unlabeled image에서 두 종류의 view를 만든다.

- augmented views: 같은 입력에서 서로 다른 augmentation으로 만든 두 view
- mined views: unlabeled pool에서 랜덤 샘플링한 추가 anchor-like sample들

이 view들을 student/teacher network에 통과시켜 encoder와 decoder feature를 얻고, projector와 predictor를 통해 embedding을 만든다. 이후 augmented view와 mined view 사이의 similarity distribution을 계산하고, student와 teacher의 분포가 비슷해지도록 KL divergence를 최소화한다.

논문에 제시된 instance discrimination loss는 다음과 같다.

$$
L_{\text{inst}} = KL(u_s \| u_t)
$$

여기서 $u_s$와 $u_t$는 각각 student와 teacher에서 계산된 similarity 기반 확률 분포이다. 본문에서는 다음과 같은 형태로 정의된다.

$$
u_s = \log \frac{\exp(\text{sim}(w_1, v_3)/\tau_s)}{\sum_{n=1}^{N}\exp(\text{sim}(w_1, v_{3n})/\tau_s)}, \quad
u_t = \log \frac{\exp(\text{sim}(w_2, v_3)/\tau_t)}{\sum_{n=1}^{N}\exp(\text{sim}(w_2, v_{3n})/\tau_t)}
$$

pre-training objective는 global/local $L_{\text{inst}}$와 supervised segmentation loss $L_{\text{sup}}$의 조합이다. $L_{\text{sup}}$는 Dice loss와 cross-entropy loss를 동일 가중치로 합한 것이라고 논문에 명시되어 있다.

기존 ACTION의 fine-tuning 단계에서는 **AnCo loss**를 사용한다. 각 class $c$에 대해, query feature는 같은 class 픽셀에서 가져오고, positive key는 그 class의 평균 representation이며, negative key는 다른 class의 representation이다. 식은 다음과 같다.

$$
L_{\text{anco}} =
\sum_{c \in C} \sum_{r_q \sim R_q^c}
- \log
\frac{\exp(r_q \cdot r_k^{c,+}/\tau_{\text{an}})}
{\exp(r_q \cdot r_k^{c,+}/\tau_{\text{an}}) + \sum_{r_k^- \sim R_k^c}\exp(r_q \cdot r_k^-/\tau_{\text{an}})}
$$

여기서 positive key $r_k^{c,+}$는 class $c$의 mean representation이다. fine-tuning objective는 $L_{\text{anco}}$, unsupervised cross-entropy loss $L_{\text{unsup}}$, supervised loss $L_{\text{sup}}$의 동일 가중치 조합이다.

### 3.2 SAACL: Supervised Adaptive Anatomical Contrastive Learning

이 논문의 핵심 개선은 바로 이 부분이다. SAACL은 세 단계로 구성된다.

#### 3.2.1 Anatomical Center Pre-computation

우선 각 class의 중심점(class center)을 embedding space에서 미리 계산한다. 이 center들은 $d$차원 unit sphere $S^{d-1}$ 위의 점들로 놓이며, 서로 최대한 균등하게 퍼지도록 만든다. 이를 위해 저자들은 다음 uniformity loss를 최소화한다.

$$
L_{\text{unif}}(\{\psi_c\}_{c=1}^{K}) =
\sum_{c=1}^{K}
\log \left(
\sum_{c'=1}^{K}
\exp(\psi_c \cdot \psi_{c'} / \tau)
\right)
$$

여기서 $\{\psi_c\}$는 class center들이다. gradient descent로 이 loss를 최소화해 optimal centers $\{\psi_c^\star\}$를 구한다. 논문은 $d \gg K$가 되도록 latent dimension을 잡으면 두 class center 사이의 minimum distance를 크게 하는 해를 찾기 좋다고 설명한다. 또한 이 해는 구 위에 내접한 regular simplex 형태의 배치를 이룬다고 언급한다.

중요한 점은 이 center 계산이 **데이터와 무관한 off-line 단계**라는 것이다. 즉, 학습 데이터의 class imbalance에 영향을 받지 않는다. 이것이 tail class를 보호하는 중요한 설계이다.

#### 3.2.2 Adaptive Allocation

사전에 계산한 center가 여러 개 있다고 해서, 어떤 center를 어떤 실제 class에 연결할지는 학습 중 결정해야 한다. 이 배정 문제는 원래 combinatorial optimization이지만, exhaustive search는 비현실적이므로 저자들은 batch 기반 empirical mean을 사용한 adaptive allocation을 제안한다.

batch $B = \{B_1, \dots, B_K\}$에서 class $c$의 empirical mean은 다음과 같이 정의된다.

$$
\phi_c(B) = \frac{\sum_{i \in B_c}\phi_i}{\left\|\sum_{i \in B_c}\phi_i\right\|_2}
$$

여기서 $\phi_i$는 sample $i$의 feature embedding이다. 이후 pre-computed center와 empirical mean 사이의 거리 합이 최소가 되도록 assignment $\pi^\star$를 구한다.

$$
\pi^\star = \arg\min_{\pi} \sum_{c=1}^{K} \|\psi_{\pi(c)}^\star - \phi_c\|_2
$$

실제로는 empirical mean을 moving average로 갱신한다.

$$
\phi_c \leftarrow (1-\eta)\phi_c + \eta \phi_c(B)
$$

즉, 현재 batch에서 본 class feature의 중심과 미리 계산해 둔 optimal center를 점진적으로 맞춰 가는 방식이다.

#### 3.2.3 Adaptive Anatomical Contrast

마지막으로, 각 픽셀 feature가 자신의 class center 주변으로 모이도록 supervised contrastive loss를 설계한다. batch의 픽셀-feature-label tuple을 $\{(\omega_i, \phi_i, y_i)\}_{i=1}^{n}$라고 할 때, loss는 다음과 같다.

$$
L_{\text{aaco}} =
-\frac{1}{n}\sum_{i=1}^{n}
\left(
\sum_{\phi_i^+}
\log
\frac{\exp(\phi_i \cdot \phi_i^+ / \tau_{sa})}
{\sum_{\phi_j}\exp(\phi_i \cdot \phi_j / \tau_{sa})}
+
\lambda_a
\log
\frac{\exp(\phi_i \cdot \nu_i / \tau_{sa})}
{\sum_{\phi_j}\exp(\phi_i \cdot \phi_j / \tau_{sa})}
\right)
$$

여기서 $\nu_i = \psi_{\pi^\star(y_i)}^\star$는 label $y_i$에 배정된 pre-computed class center이다.

이 식은 두 부분으로 이해할 수 있다.

첫 번째 항은 일반적인 supervised contrastive term이다. 같은 class의 픽셀 feature끼리는 가깝게, 다른 class와는 멀어지도록 만든다.

두 번째 항은 각 픽셀 feature를 해당 class의 pre-computed center 쪽으로 당기는 역할을 한다. 이 항이 바로 adaptive anatomical contrast의 핵심이다.

$\lambda_a$는 이 center matching 항의 강도를 조절하는 하이퍼파라미터이다.

결과적으로 SAACL은 단순히 같은 class끼리 뭉치게 하는 것을 넘어서, **각 class cluster가 embedding space 전체에 균형 있게 퍼지도록 강제**한다.

### 3.3 ATS: Anatomical-aware Temperature Scheduler

저자들은 temperature $\tau$를 고정하지 않고 학습 iteration에 따라 바꾸는 cosine schedule을 제안한다. 식은 다음과 같다.

$$
\tau_t = \tau^- + 0.5(1 + \cos(2\pi t / T))(\tau^+ - \tau^-)
$$

여기서 $t$는 현재 iteration, $T$는 총 iteration 수, $\tau^- < \tau^+$는 하한과 상한이다.

이 설계의 직관은 다음과 같다. contrastive learning에서 작은 $\tau$는 더 날카로운 구분을 만들고, 큰 $\tau$는 더 그룹 수준의 구조를 강조한다. long-tailed segmentation에서는 boundary나 rare structure를 구분하기 위한 fine-grained discrimination도 필요하고, 동시에 anatomical group 구조도 유지해야 한다. ATS는 이 둘을 번갈아 학습하게 하여 representation 품질을 높인다고 주장한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

논문은 두 개의 benchmark를 사용한다.

첫째, **LA dataset**은 100개의 gadolinium-enhanced MRI scan으로 구성되며, 80개를 training, 20개를 validation에 사용한다.

둘째, **ACDC dataset**은 100명 환자의 200 cardiac cine MRI scan으로 구성되며, segmentation 대상 class는 left ventricle (LV), myocardium (Myo), right ventricle (RV)이다. 데이터 분할은 70명 training, 10명 validation, 20명 testing이다.

두 데이터셋 모두 5% label과 10% label의 semi-supervised setting에서 평가한다.

구현 세부사항도 비교적 명확히 제시되어 있다. optimizer는 SGD, learning rate는 $1e\!-\!2$, momentum은 $0.9$, weight decay는 $0.0001$이다. 입력은 zero mean, unit variance로 정규화하고, augmentation은 rotation과 flip을 사용한다. ACTION 기반 구현이므로 대부분의 설정은 ACTION을 따른다. 새로 중요한 하이퍼파라미터는 $\lambda_a = 0.2$, latent dimension $d=128$, 그리고 temperature range $\tau^+=1.0$, $\tau^-=0.1$이다.

ACDC에서는 U-Net backbone, patch size $256 \times 256$, batch size 8을 썼고, pre-training 10K iteration, fine-tuning 20K iteration이다. LA에서는 V-Net backbone, crop size $112 \times 112 \times 80$, batch size 2를 사용했고, pre-training 5K, fine-tuning 15K iteration이다.

평가지표는 **Dice coefficient (DSC)**와 **Average Surface Distance (ASD)**이다. DSC는 높을수록 좋고, ASD는 낮을수록 좋다.

### 4.2 LA 결과

LA dataset에서 ACTION++는 5% label과 10% label 모두에서 가장 좋은 성능을 보인다.

5% labeled setting에서는:
- ACTION: DSC 86.6, ASD 2.24
- ACTION++: DSC 87.8, ASD 2.09

10% labeled setting에서는:
- ACTION: DSC 88.7, ASD 2.10
- ACTION++: DSC 89.9, ASD 1.74

즉, 기존 ACTION 대비 Dice는 상승하고 ASD는 감소했다. SS-Net, MC-Net, UAMT 등 다른 semi-supervised baseline보다도 우수하다. 저자들은 appendix의 qualitative result를 통해 boundary region에서도 더 sharp하고 정확한 segmentation을 보인다고 설명한다.

### 4.3 ACDC 결과

ACDC에서도 ACTION++는 모든 비교 방법을 능가한다.

5% labeled setting 평균 성능은:
- ACTION: 87.5 / 1.12
- ACTION++: **88.5 / 0.723**

세부 class별로도 RV, Myo, LV 모두 개선된다.
- RV: 85.4 / 0.915 → 86.9 / 0.662
- Myo: 85.8 / 0.784 → 86.8 / 0.689
- LV: 91.2 / 1.66 → 91.9 / 0.818

10% labeled setting 평균 성능은:
- ACTION: 89.7 / 0.736
- ACTION++: **90.4 / 0.592**

세부 class별 결과는:
- RV: 89.8 / 0.589 → 90.5 / 0.448
- Myo: 86.7 / 0.813 → 87.5 / 0.628
- LV: 92.7 / 0.804 → 93.1 / 0.700

논문은 특히 RV와 Myo 같은 어려운 구조에서도 경계를 더 정확히 복원한다고 해석한다. 이는 minority-like region이나 difficult boundary에 대한 robustness가 좋아졌다는 주장과 연결된다.

### 4.4 Ablation Study

LA 10% label setting에서 SAACL의 효과를 따로 검증한다.

- KCL: DSC 88.4, ASD 2.19
- CB-KCL: DSC 86.9, ASD 2.47
- SAACL: DSC 89.9, ASD 1.74
- SAACL (random assign): DSC 88.0, ASD 2.79
- SAACL (adaptive allocation): DSC 89.9, ASD 1.74

이 결과는 두 가지를 보여 준다. 첫째, 단순 contrastive variant보다 SAACL이 낫다. 둘째, pre-computed center를 단순 random assign하는 것보다 **adaptive allocation이 매우 중요**하다.

또한 ATS와 SAACL을 단계별로 비교한 ablation에서는:
- pre-training without ATS: 86.2 / 2.69
- pre-training with ATS: 88.1 / 2.44
- fine-tuning without SAACL/ATS: 89.0 / 2.06
- fine-tuning only with ATS: 89.3 / 1.98
- fine-tuning only with SAACL: 89.5 / 1.96
- fine-tuning with SAACL/ATS: **89.9 / 1.74**

즉, ATS와 SAACL은 각각도 유효하지만 같이 쓸 때 가장 좋다.

추가로 cosine boundary, cosine period, temperature variation 방식, $\lambda_a$에 대한 분석도 제공한다. 논문이 보고한 최적 설정은 $\tau^- = 0.1$, $\tau^+ = 1.0$, $T/\#iterations = 1.0$, cosine scheduler, $\lambda_a = 0.2$이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 실제적이라는 점이다. 많은 semi-supervised segmentation 연구가 unlabeled data 활용에만 집중하는 반면, 이 논문은 **labeled data 내부의 class imbalance 자체도 representation learning의 병목**이라는 점을 짚는다. 이는 long-tailed segmentation을 더 정면으로 다루는 접근이다.

또 다른 강점은 방법이 비교적 명확하고 설계 논리가 일관된다는 점이다. class center를 off-line으로 균등 배치한 뒤, 학습 중 adaptive allocation으로 실제 class와 연결하고, supervised contrastive objective로 그 center에 feature를 모이게 만드는 구조는 매우 직관적이다. 특히 center pre-computation이 데이터 분포에 의존하지 않는다는 점은 tail class 보호라는 목적과 잘 맞는다.

실험 결과도 강하다. ACDC와 LA 모두에서 기존 SOTA였던 ACTION보다 일관되게 개선되었고, DSC뿐 아니라 ASD에서도 좋아졌다. 특히 low-label setting인 5%에서도 성능 향상이 유지된다는 점은 label efficiency 주장과 잘 맞는다.

이론 분석을 추가한 점도 장점이다. Appendix A에서는 nearest neighbor classifier 관점에서 representation quality를 분석하며, 좋은 representation이 되기 위해서는 **positive alignment**는 작아야 하고, class divergence는 커야 한다고 설명한다. SAACL의 두 항이 각각 이 역할을 담당한다는 해석은 방법의 목적을 이론적으로 뒷받침한다.

반면 한계도 있다. 첫째, 논문은 Appendix에서 이론적 우수성을 언급하지만, 실제 본문에서는 그 수학적 조건들이 실험 상황에서 얼마나 직접적으로 성립하는지까지는 충분히 검증하지 않는다. 즉, 이론은 해석적 근거를 제공하지만 실전 학습 dynamics를 완전히 설명한다고 보기는 어렵다.

둘째, 성능 향상의 원인이 얼마나 SAACL 자체에서 오는지, 혹은 ACTION backbone의 강력함과 결합된 효과인지 분리해서 보기에는 제한이 있다. 논문은 ACTION 위에 개선을 쌓는 방식이므로, 전혀 다른 SSL backbone에서도 같은 수준의 개선이 유지되는지는 이 텍스트만으로는 확인할 수 없다.

셋째, 데이터셋 범위가 넓지 않다. ACDC와 LA는 대표적인 benchmark이지만, 결론에서도 저자들이 앞으로 foreground label이 더 많은 CT/MRI dataset에서 검증하겠다고 밝히고 있다. 즉, 다기관 데이터나 class 수가 더 많은 복잡한 segmentation 문제로의 일반화는 아직 충분히 보여 주지 않았다.

넷째, 논문은 t-SNE를 future work로 언급하지만, 현재 텍스트 기준으로는 representation space가 실제로 얼마나 균등하게 퍼졌는지에 대한 시각화 증거는 제한적이다. class center 기반 방법이라면 이러한 qualitative embedding analysis가 있었으면 더 설득력 있었을 것이다.

또한 일부 구현 세부는 ACTION을 따른다고만 하고 본문에서 모두 재기술하지 않는다. 예를 들어 local instance discrimination의 자세한 정의는 단순화를 위해 생략했다고 명시되어 있다. 따라서 완전한 재현을 위해서는 원문 ACTION 논문도 함께 참고할 필요가 있다.

## 6. 결론

이 논문은 semi-supervised medical image segmentation에서 long-tailed class imbalance를 더 잘 다루기 위해 **ACTION++**를 제안한다. 핵심 기여는 두 가지다. 하나는 labeled data에서도 class separation을 강화하기 위한 **SAACL**, 다른 하나는 contrastive learning의 temperature를 동적으로 바꾸는 **ATS**이다. SAACL은 embedding space에 균등하게 배치된 pre-computed class center를 기준으로 feature를 정렬하게 하여 head class 편향을 줄이고 tail class 분리를 돕는다. ATS는 group-level과 instance-level discrimination 사이의 균형을 학습 과정에서 조절한다.

실험적으로 ACTION++는 LA와 ACDC 모두에서 기존 SSL 방법들, 특히 ACTION을 꾸준히 앞선다. 따라서 이 연구는 단순한 성능 개선을 넘어서, 의료영상 segmentation에서 representation space를 어떻게 설계해야 minority class까지 안정적으로 다룰 수 있는지에 대한 하나의 분명한 방향을 제시한다. 실제 적용 측면에서는 적은 라벨 환경에서 중요한 작은 구조를 더 안정적으로 분할해야 하는 문제에 의미가 있고, 향후 연구 측면에서는 long-tailed segmentation, supervised contrastive learning, dynamic temperature scheduling의 결합을 더 확장하는 출발점이 될 가능성이 크다.
