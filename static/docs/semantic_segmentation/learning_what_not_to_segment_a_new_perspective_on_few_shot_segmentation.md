# Learning What Not to Segment: A New Perspective on Few-Shot Segmentation

- **저자**: Chunbo Lang, Gong Cheng, Binfei Tu, Junwei Han
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2203.07615

## 1. 논문 개요

이 논문은 few-shot segmentation(FSS)에서 기존 meta-learning 기반 방법들이 보이는 구조적 한계를 정면으로 다룬다. FSS의 목표는 매우 적은 수의 support annotation만 보고 query image에서 같은 범주의 물체를 분할하는 것이다. 그런데 기존 방법들은 대부분 few-shot classification에서 유래한 episodic meta-learning에 의존하며, base classes로 학습된 모델이 실제로는 class-agnostic하게 일반화되지 못하고 seen classes 쪽으로 편향된다는 문제가 있다. 저자들은 이 편향 때문에 novel class를 분할해야 할 때, query image 안의 base class 물체가 distractor로 작용하며 잘못 활성화되는 현상이 심해진다고 본다.

논문의 핵심 문제의식은 “무엇을 segment할 것인가”만 배울 것이 아니라, “무엇을 segment하지 말아야 하는가”도 명시적으로 배워야 한다는 점이다. 이를 위해 저자들은 기존 FSS 모델에 추가 branch를 붙여 query image 안의 base class 영역을 예측하게 하고, 이 정보를 이용해 meta learner의 coarse prediction을 보정하는 BAM(Base And Meta)이라는 구조를 제안한다. 즉, novel object를 직접 찾는 meta learner와, base object를 찾아 제거 신호를 제공하는 base learner를 함께 사용한다.

이 문제는 중요하다. FSS는 annotation 비용이 큰 semantic segmentation을 소량 데이터 환경으로 확장하려는 방향인데, 실제 응용에서는 novel class와 base-like distractor가 함께 등장하는 장면이 흔하다. 따라서 단순히 support-query correspondence를 강화하는 것만으로는 충분하지 않고, seen-class bias를 어떻게 완화할지에 대한 해법이 필요하다. 이 논문은 그 점에서 기존 FSS를 다른 각도에서 재해석한다.

## 2. 핵심 아이디어

중심 아이디어는 매우 직관적이다. 기존 FSS는 novel target을 찾는 데만 집중하지만, 실제로는 query image 안에 base class에 해당하는 혼동 유발 영역(confusable regions)이 존재하고, 이것이 meta learner의 오탐을 유발한다. 그렇다면 base dataset으로 잘 학습할 수 있는 강한 segmentation branch를 별도로 두고, “이 부분들은 base classes이므로 novel target이 아닐 가능성이 높다”는 정보를 제공하면 novel segmentation이 더 안정화될 수 있다.

이 논문의 차별점은 FSS generalization 문제를 support-query interaction module의 고도화가 아니라, base dataset의 활용 방식 자체를 재설계하는 관점에서 접근했다는 데 있다. 기존 연구들은 주로 prototype interaction, attention, graph reasoning, feature enrichment 등으로 support와 query 사이의 매칭을 정교하게 만들었다. 반면 이 논문은 meta learner의 편향 문제를 인정하고, 아예 base learner를 통해 base objects를 명시적으로 예측해 suppression signal로 사용한다.

또 하나의 중요한 아이디어는 meta learner가 support image 품질과 support-query scene gap에 민감하다는 점을 활용한 adjustment factor $\psi$이다. 저자들은 support와 query의 low-level feature로부터 Gram matrix를 계산하고, 그 차이의 Frobenius norm으로 장면 차이를 수치화한다. 이 값은 support-query 쌍이 얼마나 잘 맞는지를 나타내는 지표처럼 사용되며, meta learner의 예측을 얼마나 신뢰할지 조절하는 데 쓰인다. 결과적으로 BAM은 단순한 branch 추가가 아니라, base prediction과 support-query consistency를 함께 고려하는 ensemble framework가 된다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 base learner, meta learner, ensemble module의 세 부분으로 구성된다. 두 learner는 encoder backbone을 공유하지만 역할이 다르다. base learner는 query image 하나만 받아 base classes를 semantic segmentation 방식으로 예측한다. meta learner는 support-query 쌍을 받아 support mask가 가리키는 class를 query에서 분할한다. 이후 ensemble module이 두 결과를 합쳐 최종 few-shot prediction을 만든다.

### Base learner

base learner의 목적은 query image 안에서 base classes의 위치를 찾는 것이다. query image $x^q \in \mathbb{R}^{3 \times H \times W}$를 encoder $E$와 convolution block에 통과시켜 intermediate feature $f_b^q$를 얻는다.

$$
f_b^q = F_{\text{conv}}(E(x^q)) \in \mathbb{R}^{c \times h \times w}
$$

여기서 $F_{\text{conv}}$는 순차적 convolution 연산이고, $h \times w$는 추출된 feature map 중 가장 낮은 해상도이다. 이후 decoder $D_b$가 이 feature를 원래 해상도로 복원하면서 $(1 + N_b)$개 채널의 확률맵을 출력한다.

$$
p_b = \text{softmax}(D_b(f_b^q)) \in \mathbb{R}^{(1+N_b) \times H \times W}
$$

여기서 $N_b$는 base classes 수이다. PASCAL-5i에서는 15, COCO-20i에서는 60이라고 본문이 명시한다. 이 branch는 episodic training이 아니라 일반 semantic segmentation처럼 supervised learning으로 사전학습된다. 손실 함수는 cross-entropy loss이다.

$$
L_{\text{base}} = \frac{1}{n_{bs}} \sum_{i=1}^{n_{bs}} CE(p_{b;i}, m^q_{b;i})
$$

저자들이 base learner를 따로 pre-train한 이유도 중요하다. FSS 쪽 최신 방법들은 backbone을 고정해 generalization을 높이는 경향이 있지만, 일반 segmentation 모델은 backbone을 업데이트하며 discriminative feature를 학습하는 경향이 있다. 이 두 요구는 충돌할 수 있으므로, 저자들은 joint training 대신 two-stage training을 택한다.

### Meta learner

meta learner는 표준적인 FSS branch다. support set $S = \{x^s, m^s\}$와 query image $x^q$가 주어지면, support mask가 나타내는 class를 query에서 분할해야 한다. 저자들은 backbone의 block 2와 block 3 feature를 concat한 뒤 $1 \times 1$ convolution으로 256차원 feature로 줄인다.

$$
f_m^s = F_{1 \times 1}(E(x^s)) \in \mathbb{R}^{c \times h \times w}
$$

$$
f_m^q = F_{1 \times 1}(E(x^q)) \in \mathbb{R}^{c \times h \times w}
$$

그다음 masked average pooling(MAP)으로 support prototype $v^s$를 만든다.

$$
v^s = F_{\text{pool}}(f_m^s \odot I(m^s)) \in \mathbb{R}^c
$$

여기서 $I(m^s)$는 support mask를 feature 크기에 맞게 보간 및 확장한 것이고, $\odot$는 Hadamard product이다. 이 prototype은 support object의 class-specific cue를 담는다. 이후 $v^s$를 query feature $f_m^q$에 guidance로 주입하고 decoder $D_m$로 예측을 만든다.

$$
p_m = \text{softmax}(D_m(F_{\text{guidance}}(v^s, f_m^q))) \in \mathbb{R}^{2 \times H \times W}
$$

여기서 $F_{\text{guidance}}$는 저자 구현에서는 “expand & concatenate” 연산이다. 즉, prototype을 spatial하게 확장해 query feature와 결합한다. meta learner의 학습 손실은 binary cross-entropy이다.

$$
L_{\text{meta}} = \frac{1}{n_e} \sum_{i=1}^{n_e} BCE(p_{m;i}, m^q_i)
$$

### Ensemble module과 adjustment factor

이 논문의 가장 중요한 부분이다. 우선 base learner는 multi-class segmentation 결과를 내므로, 거기서 base foreground 확률을 모두 합쳐 few-shot task의 “base-class foreground” 확률맵 $p_b^f$를 만든다.

$$
p_b^f = \sum_{i=1}^{N_b} p_b^i
$$

이 값은 “query의 어느 위치가 base classes에 속할 가능성이 있는가”를 나타낸다.

다음으로 저자들은 support-query 쌍의 scene difference를 계산한다. low-level feature $f_{low}^s, f_{low}^q \in \mathbb{R}^{C_1 \times H_1 \times W_1}$를 reshape하여 $A^s \in \mathbb{R}^{C_1 \times N}$ 형태로 만든 뒤 Gram matrix를 계산한다.

$$
A^s = F_{\text{reshape}}(f_{low}^s) \in \mathbb{R}^{C_1 \times N}
$$

$$
G^s = A^s {A^s}^T \in \mathbb{R}^{C_1 \times C_1}
$$

query도 동일하게 $G^q$를 만든 뒤, 두 Gram matrix 차이의 Frobenius norm을 adjustment factor $\psi$로 정의한다.

$$
\psi = \|G^s - G^q\|_F
$$

직관적으로 $\psi$가 작으면 support-query 장면 차이가 작고, meta learner 예측을 더 신뢰할 수 있다. 반대로 $\psi$가 크면 support가 query를 잘 설명하지 못할 수 있으므로 meta prediction을 보수적으로 다뤄야 한다. 저자들은 이를 style transfer 문헌의 style loss 아이디어에서 가져왔다고 설명한다.

이후 meta learner의 background/foreground coarse prediction과 base learner의 예측을 결합한다.

$$
p_f^0 = F_{\text{ensemble}}(F_\psi(p_m^0), p_b^f)
$$

$$
p_f = p_f^0 \oplus F_\psi(p_m^1)
$$

여기서 $F_\psi$와 $F_{\text{ensemble}}$는 $1 \times 1$ convolution이며, $F_\psi$는 $\psi$를 이용해 meta learner 출력을 조정하고, $F_{\text{ensemble}}$는 base와 meta 결과를 합친다. 마지막 손실은 final prediction에 대한 BCE와 meta learner 자체의 BCE를 함께 쓴다.

$$
L = L_{\text{final}} + \lambda L_{\text{meta}}
$$

$$
L_{\text{final}} = \frac{1}{n_e} \sum_{i=1}^{n_e} BCE(p_i^q, m_i^q)
$$

논문에서는 $\lambda = 1.0$으로 고정했다. 이 구조는 단순히 base 결과를 덧붙이는 것이 아니라, meta learner를 보조 supervision으로 유지하면서 최종 ensemble prediction을 별도로 최적화하는 방식이다.

### K-shot 설정

$K > 1$일 때 기존 방법들은 여러 support prototype을 평균내는 경우가 많다. 그러나 저자들은 모든 support가 query에 똑같이 유용하다고 보기 어렵다고 주장한다. 어떤 support는 query와 장면 차이가 크기 때문에 적합하지 않을 수 있다. 그래서 각 support의 $\psi_i$를 모아 벡터 $\psi_t \in \mathbb{R}^K$를 만들고, 두 개의 fully connected layer로 fusion weight $\eta$를 생성한다.

$$
\eta = \text{softmax}(w_2^T \, \text{ReLU}(w_1^T \psi_t)) \in \mathbb{R}^K
$$

이 가중치로 support별 contribution을 조절한다. 즉, scene difference가 작은 support에 더 큰 비중을 준다.

### Generalized FSS로의 확장

표준 FSS는 novel class만 찾으면 되지만, generalized FSS는 query 안의 base class와 novel class를 모두 구분해야 한다. BAM은 base learner가 이미 base classes를 예측하므로 이 설정으로 자연스럽게 확장될 수 있다. 저자들은 threshold $\tau$를 이용해 final few-shot prediction과 base learner prediction을 합친다.

$$
\hat{m}_g^{(x,y)} =
\begin{cases}
1 & p_f^{1;(x,y)} > \tau \\
\hat{m}_b^{(x,y)} & p_f^{1;(x,y)} \le \tau \text{ and } \hat{m}_b^{(x,y)} \neq 0 \\
0 & \text{otherwise}
\end{cases}
$$

여기서 $\hat{m}_b$는 base learner의 argmax segmentation mask이다.

$$
\hat{m}_b = \arg\max(p_b)
$$

Supplementary에 따르면 generalized FSS에서 사용한 threshold는 $\tau = 0.9$이다.

## 4. 실험 및 결과

실험은 PASCAL-5i와 COCO-20i에서 수행되었다. 두 데이터셋 모두 4개 fold로 나누고 cross-validation 방식으로 평가한다. 각 fold에 대해 validation용으로 1,000개의 support-query pair를 랜덤 샘플링했다고 적혀 있다. 평가지표는 mean IoU(mIoU)와 foreground-background IoU(FB-IoU)다.

구현 측면에서 훈련은 두 단계다. 첫 단계에서는 base learner를 supervised segmentation으로 학습한다. PASCAL-5i에서는 100 epochs, COCO-20i에서는 20 epochs 동안 PSPNet 기반 base learner를 SGD로 학습한다. 둘째 단계에서는 base learner를 고정하고, shared encoder도 고정한 상태에서 meta learner와 ensemble module을 episodic learning으로 학습한다. meta learner는 PFENet 변형판이며, FEM 대신 ASPP를 사용해 복잡도를 줄였다. 결과는 서로 다른 random seed로 5회 실행한 평균이다.

정량 결과에서 BAM은 매우 강한 성능 향상을 보인다. PASCAL-5i에서 VGG16 backbone 기준 1-shot mIoU는 64.41, 5-shot은 68.76으로 보고되며, 기존 최고 성능보다 각각 4.71%p와 4.66%p 높다고 저자들은 강조한다. ResNet50 기준으로도 1-shot 67.81, 5-shot 70.91 mIoU를 달성했다. COCO-20i에서는 ResNet50 기준 1-shot 46.23, 5-shot 51.16 mIoU를 기록하여 HSNet 대비 각각 7.03%p, 4.26%p 앞선다고 설명한다.

PASCAL-5i의 FB-IoU 결과도 강력하다. VGG16에서 BAM은 1-shot 77.26, 5-shot 81.10을 기록했고, ResNet50에서는 1-shot 79.71, 5-shot 82.18을 달성했다. 이는 단순히 novel object boundary가 조금 나아진 정도가 아니라, foreground/background 판별 자체가 더 안정화되었음을 시사한다.

정성 결과에서도 저자들의 주장과 맞는 패턴이 보인다. baseline은 base class distractor를 novel target처럼 잘못 활성화하는 경향이 있는데, BAM은 이를 억제한다. 본문 설명에 따르면 person, sofa 같은 base-class distractor가 suppression되어 cat 같은 novel class localization이 더 정확해진다.

ablation study는 논문의 주장에 직접적인 근거를 제공한다. 먼저 two-stage training이 end-to-end joint training보다 성능이 좋았다. 이는 base learner와 meta learner가 backbone에 요구하는 학습 방식이 다르다는 저자 해석을 뒷받침한다. 또한 $L_{\text{meta}}$를 제거하면 성능이 약간 하락해, final ensemble loss만으로는 meta learner가 충분히 안정적으로 학습되지 않는다는 점을 보여준다.

ensemble module의 초기화도 중요했다. meta learner 쪽 weight를 1, base learner 쪽 weight를 0으로 두는 초기화가 random initialization보다 2.73% mIoU 더 좋았다. 이는 학습 초기에 meta learner를 기본 경로로 두고, base learner를 suppression 보조 신호로 점진적으로 사용하는 것이 유리함을 의미한다.

$\psi$를 계산할 때 어떤 low-level feature를 쓰는지도 분석했다. ResNet50에서 $B_2$ feature가 정확도와 FLOPs 사이의 trade-off가 가장 좋았다고 한다. 즉, 너무 낮은 층은 표현력이 부족하고, 너무 높은 층은 계산량이 크거나 목적에 맞지 않을 수 있다는 해석이 가능하다.

5-shot fusion에서도 제안한 reweighting이 기존 방법보다 낫다. Table 5에서 baseline 64.41 대비 Mask-OR는 65.15, Mask-Avg는 65.92, Feature-Avg는 66.83인데, 제안 방식은 68.76으로 가장 높다. 이는 support마다 query와의 적합도가 다르다는 논문의 주장을 잘 뒷받침한다.

support annotation을 bounding box로 바꿔도 성능이 꽤 유지된다. PASCAL-5i에서 pixel-wise label은 1-shot 64.41, 5-shot 68.76이고 bounding box는 62.25, 66.17이다. 성능은 다소 떨어지지만 완전히 붕괴하지는 않아서, 제안 방식이 annotation noise나 약한 supervision에도 어느 정도 견고함을 시사한다.

generalized FSS 결과도 보고한다. VGG16 기준 BAM은 1-shot에서 $mIoU_n = 43.19$, $mIoU_b = 67.03$, $mIoU_a = 61.07$이며, 5-shot에서는 각각 46.15, 67.02, 61.80이다. ResNet50에서는 1-shot 47.93, 72.72, 66.52와 5-shot 49.17, 72.72, 66.83을 기록했다. ensemble module이 novel classes뿐 아니라 전체 성능에도 도움을 준다고 저자들은 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정에 대한 관점 전환이 명확하고, 그 관점이 실제 성능 향상으로 이어졌다는 점이다. 많은 FSS 논문이 support-query interaction을 더 복잡하게 만드는 방향으로 갔던 반면, 이 논문은 seen-class bias와 distractor suppression이라는 더 근본적인 문제를 짚는다. “base class를 잘 찾아서 novel class segmentation을 돕는다”는 발상은 단순하지만 설득력이 높고, ablation도 그 효과를 비교적 분명하게 보여준다.

두 번째 강점은 방법이 의외로 범용적이라는 점이다. base learner, meta learner, ensemble module이라는 구조는 특정 backbone이나 특정 correlation block에 강하게 종속되지 않는다. 실제로 논문도 plain learner 두 개만으로 SOTA를 달성했다고 강조한다. generalized FSS로 자연스럽게 확장되는 것도 구조적 장점이다. 기존 FSS는 novel class만 찾는 설정에 머무는 경우가 많은데, BAM은 base class 예측 branch를 이미 가지고 있어 보다 현실적인 setting으로 확장하기 쉽다.

세 번째 강점은 scene difference를 수치화한 $\psi$의 도입이다. support-query mismatch는 FSS에서 흔한 문제인데, 이를 Gram matrix 기반 지표로 단순하고 효율적으로 반영한 점이 실용적이다. K-shot weighting에도 같은 값을 재사용해 일관된 설계를 만든 점도 좋다.

반면 한계도 있다. 첫째, 제안법은 base learner를 별도로 supervised pre-train해야 하므로 학습 절차가 2-stage이고, 시스템이 단일 meta learner보다 단순하다고 보기는 어렵다. 논문은 전체적으로 parameter-efficient하다고 주장하지만, 실무에서는 추가 branch와 사전학습 비용이 존재한다. 다만 정확한 inference latency 비교나 메모리 비교는 본문에 충분히 제시되지 않았다.

둘째, 방법의 성능 향상이 정말 “bias alleviation” 때문인지, 혹은 단순히 강한 base-class prior를 추가했기 때문인지는 완전히 분리되어 있지 않다. 논문은 정성 결과와 ablation으로 설득하지만, base learner가 잘못 base region을 예측할 때 생기는 failure case 분석은 충분히 다루지 않는다. 특히 novel class가 base class와 시각적으로 매우 유사한 경우, suppression이 과도해질 가능성은 이론적으로 존재하지만 논문은 이를 체계적으로 분석하지 않았다.

셋째, generalized FSS 확장은 흥미롭지만 비교 기준이 강한 편은 아니다. 본문에서는 “strong baseline results”를 제시한다고 하지만, generalized FSS 자체가 새롭게 제안된 setting에 가까워 보이며, 기존 방법들과의 폭넓은 비교는 없다. 따라서 이 부분은 중요한 확장 아이디어이긴 하지만, 아직 확립된 benchmark 경쟁 결과라고 보기는 어렵다.

넷째, $\psi$의 의미는 직관적이지만 완전히 task-specific한 이론적 정당화가 강한 것은 아니다. Gram matrix 차이는 style transfer에서 style discrepancy를 측정할 때 많이 쓰이지만, 이것이 FSS에서 support-query 적합도를 가장 잘 나타내는지에 대해서는 더 많은 비교가 가능했을 것이다. 예를 들어 feature distance, correlation-based discrepancy, learned quality score 등과의 체계적 비교는 본문에 없다.

## 6. 결론

이 논문은 few-shot segmentation의 일반화 문제를 “novel target을 더 잘 찾는 법”이 아니라 “base distractor를 명시적으로 제거하는 법”이라는 관점에서 다시 정의한다. 제안된 BAM은 base learner로 query image의 base class 영역을 찾고, meta learner의 novel segmentation 결과를 scene-difference-aware ensemble로 보정한다. 이 구조는 기존 meta-learning 기반 FSS의 seen-class bias를 완화하려는 직접적인 시도이며, PASCAL-5i와 COCO-20i에서 일관된 성능 향상을 보였다.

실제 기여는 세 가지로 요약할 수 있다. 첫째, base class suppression을 명시적으로 모델링한 새로운 FSS 관점을 제시했다. 둘째, Gram matrix 기반 adjustment factor $\psi$로 support-query mismatch를 반영했다. 셋째, 이 구조를 generalized FSS까지 확장해 base와 novel classes를 동시에 다루는 방향을 열었다.

향후 연구 관점에서도 의미가 있다. 이 논문은 FSS에서 단순히 support-query matching block을 더 복잡하게 만드는 것보다, base dataset이 초래하는 편향을 어떻게 제어할지 고민해야 한다는 메시지를 준다. 실제 응용에서는 query 안에 base-like distractor가 흔하므로, BAM의 아이디어는 의료영상, 원격탐사, 산업 결함 검출처럼 class confusion이 중요한 분야에도 확장 가능성이 있다. 다만 별도 pre-training과 failure case 분석 부족 같은 한계는 남아 있으므로, 이후 연구에서는 더 가벼운 통합 학습 방식이나 learned suppression mechanism으로 발전할 여지가 크다.
