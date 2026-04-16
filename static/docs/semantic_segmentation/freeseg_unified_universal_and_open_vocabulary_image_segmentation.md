# FreeSeg: Unified, Universal and Open-Vocabulary Image Segmentation

- **저자**: Jie Qin, Jie Wu, Pengxiang Yan, Ming Li, Ren Yuxi, Xuefeng Xiao, Yitong Wang, Rui Wang, Shilei Wen, Xin Pan, Xingang Wang
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2303.17225

## 1. 논문 개요

이 논문은 하나의 모델로 semantic segmentation, instance segmentation, panoptic segmentation을 모두 처리하면서, 학습 때 보지 못한 unseen class까지 텍스트 이름만으로 분할할 수 있는 unified open-vocabulary segmentation 문제를 다룬다. 저자들은 기존 open-vocabulary segmentation 연구가 주로 특정 task 하나에 맞춘 구조를 설계해 왔고, task를 바꾸면 다시 별도 모델을 학습해야 한다는 점을 핵심 문제로 본다.

이 문제는 실제 응용에서 매우 중요하다. 현실의 시각 시스템은 고정된 소수 클래스만 다루는 것이 아니라, 다양한 분할 과제와 훨씬 넓은 vocabulary를 동시에 처리해야 하기 때문이다. 기존 segmentation 데이터셋의 class 수는 수십에서 수백 수준이지만, 사람이 실제로 사용하는 개념 수는 훨씬 크다. 따라서 저자들은 segmentation의 통합성, 범용성, open-vocabulary 일반화를 한 번에 만족하는 프레임워크가 필요하다고 주장한다.

이를 위해 제안된 FreeSeg는 하나의 all-in-one 모델을 one-shot training으로 학습하고, 추론 시에도 동일한 architecture와 동일한 parameter로 여러 segmentation task를 처리한다. 논문의 핵심 주장은 단순히 여러 task를 묶는 데 그치지 않고, CLIP 기반의 텍스트-비전 정렬을 통해 arbitrary category까지 다룰 수 있게 했다는 점이다.

## 2. 핵심 아이디어

FreeSeg의 중심 아이디어는 segmentation을 두 단계로 나누는 것이다. 첫 번째 단계에서는 task에 독립적인 class-agnostic mask proposal을 생성하고, 두 번째 단계에서는 이 mask들에 대해 CLIP 기반 zero-shot classification을 수행한다. 이렇게 하면 mask 생성 자체는 범용적으로 유지하면서, category 인식은 텍스트 임베딩을 통해 유연하게 바꿀 수 있다.

기존 접근과의 가장 큰 차별점은 세 가지다. 첫째, semantic, instance, panoptic segmentation을 하나의 모델로 함께 학습한다. 둘째, seen class에만 학습하면서도 unseen class를 텍스트로 일반화한다. 셋째, 단순한 고정 prompt가 아니라 adaptive prompt learning을 사용해 task 정보와 class 정보를 함께 텍스트 표현에 주입한다.

저자들은 특히 task-aware 특성과 category-sensitive 특성을 동시에 잡는 것이 중요하다고 본다. 단순히 class name만 넣는 fixed prompt로는 semantic segmentation용 문맥과 instance segmentation용 문맥이 충분히 구분되지 않는다. 그래서 FreeSeg는 adaptive task prompt와 adaptive class prompt를 함께 학습하고, 여기에 semantic context interaction과 test time prompt tuning까지 추가해 cross-modal alignment를 더 강하게 만든다.

## 3. 상세 방법 설명

FreeSeg는 전체적으로 두 단계 구조를 가진다. 첫 단계의 mask proposal extractor는 이미지 $I$를 입력받아 시각 특징 $F_v \in \mathbb{R}^{N \times D}$와 class-agnostic mask $M \in \mathbb{R}^{N \times H \times W}$를 만든다. 여기서 $N$은 query 수, $D$는 feature 차원이다. 이 모듈은 논문 구현에서 Mask2Former를 기반으로 한다.

학습 시에는 multi-task label을 사용한다. 즉, semantic, instance, panoptic label이 모두 있을 수 있지만, 모든 task를 동시에 한 iteration에 직접 supervision하면 gradient conflict가 생길 수 있으므로, 각 iteration에서는 세 label 중 하나를 무작위로 골라 supervision한다. 마스크 학습 손실은 다음과 같다.

$$
L_{mask} = L_F(M, M^{gt}) + L_D(M, M^{gt})
$$

여기서 $L_F$는 Focal loss, $L_D$는 Dice loss이다. 즉, mask proposal extractor는 기본적으로 정확한 binary mask를 생성하도록 학습된다.

그 다음 classification 단계에서는 adaptive prompt learning으로 만든 텍스트 임베딩 $F_t \in \mathbb{R}^{C \times D}$와 visual concept $F_v$ 사이의 cosine similarity를 계산한다. 각 mask-query와 class 텍스트 사이 유사도는 다음과 같다.

$$
S(i,j) = \cos(F_v^i, F_t^j) = \frac{F_v^i \cdot F_t^j}{\|F_v^i\| \|F_t^j\|}
$$

이 similarity map $S$는 각 mask가 어떤 class에 해당하는지를 나타내며, 여기에 class label 기반 cross-entropy loss $L_{cla}$를 적용한다. 전체 학습 손실은 다음과 같다.

$$
L = L_{cla} + L_{mask}
$$

adaptive prompt learning은 이 논문의 핵심 설계다. adaptive task prompt $P_t$는 task 이름, 예를 들어 “semantic segmentation.”, “instance segmentation.”, “panoptic segmentation.”을 learnable vector들과 함께 CLIP text encoder에 넣어 task embedding $E_t$를 만든다.

$$
E_t = \Psi(P_t(t)), \quad t \in T
$$

adaptive class prompt $P_c$는 class name을 learnable vector들과 결합해 class embedding $E_c$를 만든다.

$$
E_c = \Psi(P_c(c)), \quad c \in C_{seen}
$$

이 둘을 concat하여 최종 multi-granularity text embedding을 만든다.

$$
F_t = Cat(E_c, E_t)
$$

이 구조의 의미는 단순하다. 같은 class라도 어떤 task에서 묻는지에 따라 필요한 표현이 다를 수 있으므로, class 정보와 task 정보를 함께 텍스트 표현에 넣자는 것이다. 논문은 이것이 unified model이 여러 task를 동시에 처리하는 데 중요하다고 설명한다.

semantic context interaction은 텍스트 정보를 decoder의 visual feature에 직접 주입하는 모듈이다. 논문은 cross-attention을 사용해 텍스트 임베딩과 multi-scale visual feature의 상관관계를 모델링한다. attention은 다음과 같이 정의된다.

$$
Attn(Q_z, K, V) = softmax\left(\frac{Q_z K^T}{\sqrt{d_k}}\right)V^T
$$

여기서

$$
Q_z = \phi_q(F_v^z), \quad K = \phi_k(F_t), \quad V = \phi_v(F_t)
$$

이다. $F_v^z$는 decoder의 $z$번째 레이어 visual feature다. 이렇게 얻은 attention 결과를 projection layer $H$로 보내 강화된 visual feature를 얻는다.

$$
\hat{F}_v^z = H\{Attn[\phi_q(F_v^z), \phi_k(F_t), \phi_v(F_t)]\}
$$

직관적으로 보면, visual feature가 현재 주어진 task/class 텍스트와 더 잘 맞도록 보정되는 과정이다. 저자들은 이것이 cross-modal alignment를 개선한다고 본다.

test time prompt tuning은 unseen class 일반화를 위한 추가 장치다. 테스트 시 unseen class에 대한 similarity score $S_u$만 따로 보고, entropy가 낮은 고신뢰 query만 골라 adaptive class prompt를 추가로 미세조정한다. 샘플별 entropy는 다음과 같다.

$$
entro = - \frac{1}{N_u} \sum_{i=1}^{N_u} s_i \log(s_i)
$$

여기서 $N_u$는 unseen class 수이고 $s_i$는 unseen class score다. entropy가 threshold $\tau$보다 낮은 샘플만 선택해 $S_u^*$를 만들고, 선택된 query에 대해 entropy loss를 최소화한다.

$$
L_{ent} = - \frac{1}{N_u K} \sum_{i=1}^{N_u} \sum_{j=1}^{K} s_{ij} \log(s_{ij})
$$

여기서 $K$는 선택된 query 수다. 핵심 아이디어는 test time adaptation을 이용해 unseen class에 대해 prompt를 더 잘 맞추는 것이다. 다만 논문은 이 과정의 최적화 세부 절차를 매우 상세히 풀어 쓰지는 않았고, threshold 선택 기준도 본문 발췌에서는 구체적으로 보이지 않는다.

## 4. 실험 및 결과

실험은 COCO, ADE20K, PASCAL VOC2012에서 수행되었다. COCO는 171개 category 중 156 seen, 15 unseen으로 나누었고, semantic/panoptic annotation을 통합해 사용했다. ADE20K는 150개 category 중 135 seen, 15 unseen으로 나누었다. VOC2012는 semantic segmentation만 평가하며 20 foreground class 중 15 seen, 5 unseen으로 나누었다.

평가 지표는 task별로 다르다. semantic segmentation은 seen/unseen에 대한 mIoU와 이들의 조화평균인 hIoU를 사용했다. instance segmentation은 mAP를 사용했다. panoptic segmentation은 PQ, SQ, RQ를 사용했다.

구현 측면에서 COCO 실험은 Mask2Former를 mask proposal extractor로, ResNet101을 backbone으로, CLIP의 ViT-B/16을 vision-language backbone으로 사용했다. 8개의 A100 GPU, AdamW optimizer, learning rate 0.0002, weight decay 0.0002, 총 60,000 iteration으로 학습했다. task prompt 크기는 $8 \times 512$, class prompt 크기는 $16 \times 512$라고 명시되어 있다.

semantic segmentation 결과를 보면 FreeSeg는 COCO unseen에서 49.1 mIoU, ADE20K unseen에서 28.6 mIoU를 기록했다. 이는 기존 최고 성능이던 ZSSeg 대비 각각 $+5.5$%p, $+8.3$%p 개선이다. VOC2012에서도 seen/unseen이 각각 91.8/82.6 mIoU로, ZSSeg보다 크게 높다. 특히 COCO에서 fully supervised baseline이 unseen 54.3 mIoU인데 FreeSeg는 49.1 mIoU라서, unseen class를 학습하지 않았음에도 완전지도학습과의 차이가 비교적 작다는 점을 강조한다.

instance segmentation에서는 COCO unseen 20.6 mAP, ADE20K unseen 15.4 mAP를 달성했다. COCO 기준으로 이전 최고 ZSI의 unseen 13.6 mAP보다 $+7.0$%p 높다. 다만 seen class에서는 ZSI가 더 높다. 저자들은 그 이유를 ZSI가 instance segmentation에 유리한 box-level supervision을 사용하기 때문이라고 설명한다. 반면 FreeSeg는 여러 segmentation task를 모두 포괄하는 더 일반적인 mask supervision을 사용한다.

panoptic segmentation에서는 개선 폭이 특히 크다. COCO unseen에서 FreeSeg는 29.8 PQ, 79.2 SQ, 37.6 RQ를 기록했고, ZSSeg 대비 각각 $+20.1$%p, $+7.5$%p, $+25.4$%p 높다. ADE20K unseen에서도 25.4 PQ, 75.2 SQ, 30.6 RQ로 가장 좋다. 저자들은 이것이 semantic segmentation 중심 방법이 다른 segmentation task로 일반화되기 어렵다는 점을 보여 준다고 해석한다.

cross-dataset generalization도 중요한 실험이다. COCO에서 학습한 모델을 ADE20K에 그대로 테스트하거나, ADE20K에서 학습한 모델을 COCO에 그대로 테스트했다. 이 설정에서는 target dataset의 모든 class를 unseen으로 본다. COCO→ADE20K에서 FreeSeg는 24.6 mIoU, 6.5 mAP, 16.3 PQ를 기록해 CLIP, LSeg+, OpenSeg, ZSSeg, MaskCLIP보다 전반적으로 좋았다. ADE20K→COCO에서도 21.7 mIoU, 6.6 mAP, 16.5 PQ로 최고 성능을 보였다. VOC로의 전이에서도 COCO→VOC는 91.9 mIoU, ADE20K→VOC는 80.1 mIoU였다.

ablation study도 비교적 설득력 있게 제시된다. 텍스트 guidance가 전혀 없는 기본 vision model은 unseen 성능이 semantic 4.9 mIoU, instance 0.7 mAP, panoptic 0.1 PQ 수준으로 매우 낮다. adaptive class prompt를 넣으면 unseen 성능이 크게 오른다. 여기에 adaptive task prompt, semantic context interaction, test time prompt tuning을 차례로 추가할수록 성능이 계속 좋아진다. 최종적으로 COCO에서 unseen semantic mIoU는 43.3, unseen instance mAP는 14.6, unseen panoptic PQ는 19.2까지 향상된다. 저자들은 이를 통해 각 모듈이 실제로 기여한다고 주장한다.

또한 multi-task 학습과 single-task 학습을 비교하면, 하나의 unified model로 학습한 multi-task setting이 unseen 성능에서 더 좋았다. 예를 들어 COCO에서 unseen semantic mIoU는 42.9에서 43.3으로, unseen instance mAP는 12.7에서 14.6으로, unseen panoptic PQ는 17.5에서 19.2로 증가했다. 저자들은 이것이 multi-task training이 일반화를 돕는다는 증거라고 본다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의 자체가 분명하고 야심차다는 점이다. open-vocabulary segmentation과 universal segmentation을 따로 보는 대신, 하나의 모델이 여러 segmentation task를 모두 처리하도록 묶어냈다. 특히 semantic, instance, panoptic segmentation을 모두 다루면서 unseen class까지 평가했다는 점은 실험 범위가 넓고 설득력이 있다.

두 번째 강점은 구조가 지나치게 복잡하지 않다는 점이다. Mask2Former 기반 proposal extractor와 CLIP 기반 zero-shot classifier라는 비교적 이해 가능한 구성 위에, adaptive prompt learning, semantic context interaction, test time prompt tuning을 추가했다. 각 모듈의 역할도 비교적 명확하다. adaptive prompt는 task/class 문맥 주입, context interaction은 visual-text alignment 강화, prompt tuning은 unseen 적응 강화라는 식으로 기능 분담이 잘 보인다.

세 번째 강점은 실제 결과다. 특히 unseen class에서의 semantic, instance, panoptic 성능 향상이 크고, panoptic segmentation에서의 개선 폭이 두드러진다. cross-dataset generalization 실험까지 포함해, 단순히 한 benchmark에 과적합된 방법이 아니라는 점을 보여 주려 한 것도 좋다.

반면 한계도 있다. 첫째, 이 논문은 one-shot unified training을 강조하지만, 실제 학습에서는 각 iteration마다 semantic/instance/panoptic label 중 하나만 랜덤 선택해 supervision한다. 즉, 완전히 동시에 최적화하는 형태는 아니며, gradient conflict를 완화하기 위한 타협적 설계다. 이것은 현실적인 선택이지만, “진정한 의미의 공동 최적화”와는 다르다.

둘째, instance segmentation의 seen 성능은 일부 specialized method보다 낮다. 논문도 이를 인정하며, 범용 mask supervision과 task-specific box supervision 사이의 trade-off가 있음을 시사한다. 따라서 FreeSeg는 모든 상황에서 절대적으로 최고인 모델이라기보다, 범용성과 unseen 일반화에서 특히 강한 모델로 보는 편이 정확하다.

셋째, test time prompt tuning의 실제 계산 비용과 안정성은 본문 발췌만으로는 충분히 평가하기 어렵다. 고신뢰 query를 골라 entropy minimization을 수행하는 방식은 이론적으로 타당하지만, threshold $\tau$ 설정, tuning step 수, 추론 속도 영향 같은 운영상 요소는 여기 제공된 텍스트만으로는 명확하지 않다. 논문 원문 일부가 생략되어 있어 그 부분을 더 읽지 않고는 단정할 수 없다.

넷째, 본문은 CLIP 기반 zero-shot 분류의 강점을 활용하지만, 동시에 CLIP 표현 품질에 상당히 의존한다. 즉, 텍스트-비전 사전학습 모델이 포착하지 못하는 미세한 category 구분이나 domain-specific 개념에서는 성능이 제한될 가능성이 있다. 다만 이 점은 논문이 직접 실험으로 충분히 파고들었다기보다, 접근법 자체의 구조적 가정에 가깝다.

## 6. 결론

이 논문은 unified open-vocabulary segmentation이라는 새로운 문제를 전면에 내세우고, 이를 해결하기 위한 FreeSeg 프레임워크를 제안했다. 핵심 기여는 하나의 모델, 하나의 architecture, 하나의 inference parameter setting으로 semantic, instance, panoptic segmentation을 모두 처리하면서, unseen class까지 텍스트 기반으로 일반화했다는 점이다. 이를 위해 two-stage segmentation 구조, adaptive task/class prompt, semantic context interaction, test time prompt tuning을 결합했다.

실험 결과는 이 접근이 단순한 개념 제안에 그치지 않음을 보여 준다. FreeSeg는 여러 benchmark에서 기존 task-specific 또는 단일-task open-vocabulary 방법보다 더 나은 unseen 성능을 보였고, cross-dataset generalization에서도 강했다. 실제 배포 관점에서도 모델 수를 줄이고 학습 비용을 약 3분의 2 수준 절감할 수 있다고 주장한다.

종합하면, 이 연구는 open-vocabulary segmentation을 개별 task 단위에서 보던 흐름을 넘어, 보다 통합적이고 실용적인 방향으로 확장한 작업이다. 앞으로의 후속 연구는 더 강한 vision-language backbone, 더 정교한 multi-task optimization, test-time adaptation 비용 감소 같은 방향에서 이 프레임워크를 발전시킬 가능성이 크다.
