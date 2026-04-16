# Rethinking Semi-Supervised Medical Image Segmentation: A Variance-Reduction Perspective

- **저자**: Chenyu You, Weicheng Dai, Yifei Min, Fenglin Liu, David A. Clifton, S. Kevin Zhou, Lawrence Staib, James S. Duncan
- **발표연도**: 2023
- **arXiv**: https://arxiv.org/abs/2302.01735

## 1. 논문 개요

이 논문은 semi-supervised medical image segmentation에서 contrastive learning(CL)을 다시 바라보면서, 기존 방법의 핵심 문제가 단순히 “좋은 positive/negative pair를 만들지 못한다”는 수준이 아니라, 픽셀 또는 복셀 샘플링 과정에서 발생하는 높은 분산(variance) 때문에 학습이 불안정해지고 representation collapse가 쉽게 일어난다는 점에 주목한다. 저자들은 이 관점에서 ARCO라는 새로운 semi-supervised contrastive learning 프레임워크를 제안한다.

연구 문제는 매우 명확하다. 의료 영상 분할에서는 라벨이 극히 적고 데이터가 class imbalance, long-tail distribution, 해부학적 다양성을 강하게 띠는 경우가 많다. 이때 기존 pixel-level contrastive learning은 모든 픽셀을 다루기 어렵기 때문에 일부 픽셀을 샘플링해 contrastive objective를 계산하는데, 이 샘플링이 충분히 안정적이지 않으면 서로 비슷한 해부학적 구조를 잘 구분하지 못하고, 특히 minority tail class에서 오분류가 쉽게 발생한다. 논문은 이런 현상이 결국 gradient estimator의 분산 증가와 연결되며, 학습 속도와 최종 segmentation 품질 모두를 해친다고 본다.

이 문제가 중요한 이유는 의료 영상 분할이 안전과 직접 연결되는 safety-critical task이기 때문이다. 단순히 평균 Dice가 조금 높아지는 것보다도, 매우 적은 라벨만으로도 안정적으로 수렴하고, 경계나 작은 병변처럼 어려운 구조를 덜 놓치며, minority class를 더 잘 구분하는 robust한 모델이 필요하다. 저자들은 이 논문에서 robustness를 segmentation accuracy뿐 아니라 faster convergence와 collapse 완화까지 포함하는 더 넓은 의미로 다룬다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 contrastive learning 자체를 새로 설계하기보다는, contrastive loss를 계산할 때 어떤 픽셀/복셀을 샘플링하느냐를 variance-reduction 관점에서 재설계하는 것이다. 저자들은 비슷한 해부학적 의미를 가지는 픽셀들이 공간적으로 어느 정도 묶여 있을 가능성이 높다고 보고, 이미지를 class별로 같은 크기의 grid로 나눈 뒤, 각 그룹 안에서 의미적으로 비슷한 픽셀을 높은 확률로 뽑는 방법을 사용한다.

이 아이디어는 두 가지 구체적 샘플링 기법으로 구현된다. 하나는 Stratified Group(SG)이고, 다른 하나는 Stratified-Antithetic Group(SAG)이다. SG는 전체 픽셀 집합을 여러 disjoint group으로 나눈 뒤 각 그룹에서 비례적으로 샘플링하는 방식이다. SAG는 여기에 더해 group 내부에서 대칭적인 위치의 픽셀을 함께 고려하도록 구성해 추가적인 분산 제어를 시도한다.

기존 접근과의 차별점은, 이 논문이 contrastive learning의 성능 향상을 “더 강한 augmentation”이나 “더 복잡한 memory bank 설계”가 아니라, gradient estimator의 분산을 줄이는 샘플링 이론으로 설명하고 정당화한다는 점이다. 특히 저자들은 medical segmentation에서 널리 쓰이는 MONA 프레임워크 위에 ARCO를 얹어, 기존 CL 구조와 양립 가능하면서도 더 robust하고 label-efficient한 학습을 달성할 수 있음을 보인다. 또한 SG가 naïve sampling(NS)보다 이론적으로 분산이 작거나 같고, 실질적으로는 거의 항상 더 작아진다고 증명한다.

## 3. 상세 방법 설명

ARCO는 MONA를 단순화한 두 단계 학습 절차를 따른다. 첫 번째 단계는 relational semi-supervised pre-training이고, 두 번째 단계는 anatomical contrastive fine-tuning이다. 학생(student) 네트워크와 교사(teacher) 네트워크가 있으며, teacher는 student의 exponential moving average(EMA)로 업데이트된다. backbone은 2D에서는 UNet, 3D에서는 VNet을 사용한다. 네트워크는 segmentation map뿐 아니라 representation map도 출력한다.

첫 번째 단계인 pre-training에서는 labeled data에 대해서 supervised loss $L_{\text{sup}}$를 사용하고, unlabeled data에 대해서는 augmented view와 mined view 사이의 관계를 이용한 instance discrimination loss를 학습한다. 논문은 global instance discrimination과 local instance discrimination을 모두 사용한다고 설명한다. 이 단계의 전체 손실은 다음처럼 구성된다.

$$
L = L_{\text{global inst}} + L_{\text{local inst}} + L_{\text{sup}}
$$

논문에서 제시한 unsupervised instance discrimination loss는 teacher와 student가 만든 relational similarity 분포 사이의 Kullback-Leibler divergence이다.

$$
L_{\text{inst}} = KL(u_s \| u_t)
$$

여기서 $u_s$와 $u_t$는 각각 student와 teacher branch에서 계산된 softmax 기반 relational similarity이다. 온도 하이퍼파라미터로 $\tau_s$, $\tau_t$가 사용된다.

두 번째 단계인 anatomical contrastive fine-tuning은 논문의 핵심이다. 저자들은 이 단계를 두 원칙으로 설명한다. 첫째는 tailness로, tail class의 hard pixel에 더 큰 중요도를 주는 것이다. 둘째는 diversity로, 다른 이미지들 사이의 해부학적 다양성을 충분히 반영하는 것이다.

이를 위해 pixel-level contrastive loss $L_{\text{contrast}}$를 정의한다. 각 class $c$에 대해 query representation $r_q$는 같은 class의 positive key $r_k^{c,+}$와 가깝게 만들고, 다른 class의 negative key들과는 멀어지도록 한다. 식은 다음과 같다.

$$
L_{\text{contrast}} =
\sum_{c \in C}
\sum_{r_q \sim R_q^c}
-\log
\frac{\exp(r_q \cdot r_k^{c,+}/\tau)}
{\exp(r_q \cdot r_k^{c,+}/\tau) + \sum_{r_k^- \sim R_k^c} \exp(r_q \cdot r_k^- / \tau)}
$$

여기서 $C$는 현재 mini-batch에 존재하는 class 집합이고, $\tau$는 temperature이다. $R_q^c$는 class $c$의 query 집합, $R_k^c$는 class $c$가 아닌 representation들의 negative key 집합이다. positive key는 class mean representation으로 정의된다.

$$
r_k^{c,+} = \frac{1}{|R_q^c|}\sum_{r_q \in R_q^c} r_q
$$

즉, 각 픽셀 representation을 그 class의 평균 표현에 가깝게 만들고 다른 class와는 멀어지게 하는 구조다. 이 방식은 dense prediction에 더 적합한 contrastive objective로 볼 수 있다.

diversity를 위해서는 FIFO memory bank를 이용해 $K$-nearest neighbors를 찾고, nearest neighbor loss $L_{\text{nn}}$를 사용해 inter-instance relationship을 활용한다. 또한 pseudo label 기반 unsupervised cross-entropy loss $L_{\text{unsup}}$도 포함된다. 따라서 fine-tuning 단계의 총 손실은 다음과 같다.

$$
L = L_{\text{sup}} + \lambda_1 L_{\text{contrast}} + \lambda_2 L_{\text{unsup}} + \lambda_3 L_{\text{nn}}
$$

논문에서 사용한 기본 하이퍼파라미터는 $\lambda_1 = 0.01$, $\lambda_2 = 1.0$, $\lambda_3 = 1.0$이다.

이 논문의 가장 중요한 방법론적 기여는 샘플링 추정기 자체를 다시 정의한 부분이다. 임의의 함수 $h : X \times P \to \mathbb{R}$에 대해 전체 픽셀 집합 $P$에 대한 aggregation function을

$$
H(x) = \frac{1}{|P|}\sum_{p \in P} h(x;p)
$$

로 두면, 기존 naïve sampling은 픽셀 부분집합 $D$를 뽑아서 평균을 계산하는 식으로 이를 근사한다. ARCO의 SG는 픽셀을 $M$개의 group $P_m$으로 나눈 뒤 각 그룹에서 샘플 $D_m$을 뽑아 다음처럼 추정한다.

$$
\hat H_{\text{SG}}(x;D) =
\frac{1}{M}\sum_{m=1}^M \frac{1}{|D_m|}\sum_{p \in D_m} h(x;p)
$$

SAG는 SG와 비슷하지만, 각 group 내부에서 중심 $c_m$에 대해 대칭적인 픽셀 쌍을 포함하도록 설계된다. 논문은 이 구조가 분산을 줄이는 데 유리하다고 본다.

이론적으로는 두 결과가 핵심이다. 첫째, SG는 proportional group size 조건에서 unbiased estimator다. 즉,

$$
\mathbb{E}[\hat H_{\text{SG}}(x)] = H(x)
$$

를 만족한다. 둘째, SG의 분산은 naïve sampling보다 크지 않다.

$$
\mathrm{Var}[\hat H_{\text{SG}}]
=
\mathrm{Var}[\hat H_{\text{NS}}]
-
\frac{1}{n}\sum_{m=1}^M
\left(
\mathbb{E}_{p \sim P_m}[h(x;p)] - \mathbb{E}_{p \sim P}[h(x;p)]
\right)^2
$$

오른쪽 마지막 항은 group 간 평균 차이를 반영하므로, 그룹들이 실제로 서로 다른 해부학적 특성을 담고 있다면 SG의 분산은 NS보다 엄밀하게 더 작아진다. 논문은 의료 영상에서는 이런 조건이 거의 항상 성립한다고 주장한다. SAG의 경우에는 같은 sample size에서

$$
\mathrm{Var}[\hat H_{\text{SAG}}] \le 2\,\mathrm{Var}[\hat H_{\text{SG}}]
$$

를 보인다. 즉, SG와 같은 규모의 분산 수준을 유지한다고 해석할 수 있다.

또 하나 중요한 이론 결과는 학습 수렴 분석이다. loss가 smooth하고 gradient estimate의 variance가 bounded라는 표준 가정 아래, SGD의 평균 gradient norm은

$$
\frac{1}{T}\sum_{t=1}^T \mathbb{E}\left[\|\nabla L(\theta_t)\|_2^2\right]
\le
C\left(\frac{1}{T} + \frac{\sigma_g}{\sqrt{T}}\right)
$$

를 만족한다. 여기서 $\sigma_g$는 gradient estimator의 표준편차에 해당한다. 따라서 샘플링으로 gradient variance를 줄이면 $\sigma_g$가 작아지고, 결과적으로 더 빠르고 안정적으로 수렴할 수 있다는 것이 저자들의 주장이다.

## 4. 실험 및 결과

실험은 매우 광범위하게 수행되었다. 의료 영상 분할에서는 5개 데이터셋을 사용했다. 2D 데이터셋은 ACDC, LiTS, MMWHS이고, 3D 데이터셋은 LA와 in-house MP-MRI이다. 추가로 일반 semantic segmentation 벤치마크인 Cityscapes, Pascal VOC 2012, SUN RGB-D에서도 실험했다. backbone은 2D에서 UNet, 3D에서 VNet이며, 라벨 비율은 주로 1%, 5%, 10%를 다뤘다.

평가 지표로 의료 영상에서는 Dice coefficient(DSC)와 Average Symmetric Surface Distance(ASD)를 사용했다. 일반 semantic segmentation에서는 mean IoU를 사용했다.

가장 대표적인 결과는 ACDC이다. 1% labeled setting에서 MONA의 평균 Dice는 82.6인데, ARCO-SAG는 84.9, ARCO-SG는 85.5를 기록했다. ASD는 MONA가 1.43, ARCO-SG가 0.947로 더 낮았다. 5% labeled에서는 MONA 86.9, ARCO-SG 88.7이었고, 10% labeled에서는 MONA 87.7, ARCO-SG 89.4였다. 즉, ARCO-SG는 모든 라벨 비율에서 가장 좋은 평균 Dice를 보였다.

LiTS에서도 비슷한 경향이 나타난다. 1% labeled에서 MONA의 평균 Dice는 62.2, ARCO-SG는 65.5였다. 5% labeled에서는 66.6 대 68.4, 10% labeled에서는 68.3 대 70.1이었다. 특히 lesion class처럼 어려운 소수 클래스에서 개선이 더 의미 있는 것으로 보인다.

MMWHS는 클래스 수가 7개로 더 어려운 설정인데, 여기서도 ARCO-SG의 이점이 두드러진다. 1% labeled에서 MONA의 평균 Dice는 82.2, ARCO-SG는 87.3이었다. 이는 상당히 큰 차이이며, ASD도 8.05에서 5.79로 줄었다. 5%와 10% labeled에서도 ARCO-SG는 각각 89.3, 89.4의 평균 Dice를 기록하며 가장 좋은 성능을 보였다.

3D 데이터셋에서도 개선이 유지된다. LA 데이터셋에서 1% labeled일 때 MONA는 72.8 Dice, ARCO-SG는 75.0 Dice를 기록했다. MP-MRI에서는 1% labeled에서 MONA가 91.3, ARCO-SG가 91.6으로 차이는 작지만, ASD는 5.31에서 6.60으로 오히려 SG가 더 높았다. 따라서 MP-MRI 1% setting에서는 Dice는 소폭 개선되지만 ASD 관점에서는 일관되게 우세하다고 보기는 어렵다. 반면 5%와 10% setting에서는 Dice가 92.5, 92.8까지 올라간다.

일반 semantic segmentation에서도 ReCo 대비 일관된 향상이 보고된다. 예를 들어 Pascal VOC 60 labels에서 ReCo + ClassMix는 57.1 IoU인데, ARCO-SG는 grid 설정에 따라 최대 59.6까지 오른다. Cityscapes 20 labels에서는 49.9에서 최대 53.7까지, SUN RGB-D 50 labels에서는 30.5에서 최대 38점대까지 올라간 경우가 있다. 이 결과는 ARCO의 샘플링 전략이 의료 영상에만 국한되지 않고 dense segmentation 전반에 적용 가능함을 시사한다.

정성적 결과에서도 ARCO는 경계가 더 선명하고 anatomical region의 shape consistency가 더 좋다고 보고된다. ACDC에서는 RV와 Myo의 경계가 더 분명했고, LiTS에서는 작은 lesion과 같은 tail-class 샘플에서 더 안정적인 경계를 보였다. MMWHS에서는 shape-consistent한 segmentation을 보여준다고 서술한다.

ablation study도 비교적 충실하다. ACDC 1% labeled에서 ARCO-SG 전체 모델은 Dice 85.5, ASD 0.947이다. 여기서 tailness와 연결된 $L_{\text{contrast}}$를 제거하면 Dice가 60.9로 급감하고, diversity와 연결된 $L_{\text{nn}}$를 제거하면 Dice가 79.3으로 떨어진다. 즉, contrastive tailness와 nearest-neighbor diversity 둘 다 중요하다. loss component 제거 실험에서도 $L_{\text{unsup}}$, $L_{\text{global inst}}$, $L_{\text{local inst}}$를 각각 제거할 때 모두 성능 저하가 나타난다.

data augmentation ablation에서는 random rotation, random cropping, horizontal flipping이 모두 도움이 되었고, 특히 horizontal flipping을 제거하거나 augmentation을 전혀 쓰지 않으면 성능이 크게 나빠졌다. 예를 들어 ARCO-SG는 full augmentation에서 Dice 85.5인데, no augmentation에서는 76.2까지 떨어진다.

학습 안정성 분석에서는 epoch에 따른 $L_{\text{contrast}}$ 곡선을 비교했을 때 SG가 가장 빠르게 loss가 감소하고 분산도 작았다. 논문은 이를 variance-reduction hypothesis와 일치하는 결과로 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semi-supervised medical segmentation에서 contrastive learning의 실패 원인을 sampling variance라는 관점에서 해석하고, 이를 이론과 실험으로 함께 밀어붙였다는 점이다. 단순히 성능을 높인 방법 논문이 아니라, 왜 특정 샘플링이 robust한지 수학적으로 설명하려고 시도한다. 특히 SG가 unbiased하면서 NS보다 분산이 작거나 같다는 정리는 논문의 핵심 설득력이다.

또 다른 강점은 방법이 비교적 단순하고 기존 프레임워크에 쉽게 삽입 가능하다는 점이다. 저자들은 ARCO를 MONA 위에 올렸지만, appendix의 결과를 보면 MoCo v2, SimCLR, BYOL, VICReg 등 다른 contrastive framework와도 결합 가능하다고 주장한다. 즉, 이 방법은 완전히 새로운 backbone이나 대규모 추가 모듈이 아니라 sampling rule의 변경이라는 점에서 practical하다.

실험 범위도 넓다. 5개 의료 영상 데이터셋, 3개 일반 segmentation 데이터셋, 다양한 label ratio, 2D/3D 세팅, 다수의 baseline을 포함해 결과를 제시한다. 특히 long-tail 성격이 강한 MMWHS나 lesion segmentation을 포함한 LiTS에서 개선 폭이 크다는 점은 논문의 주장과 잘 맞아떨어진다.

반면 한계도 있다. 첫째, 논문은 ARCO가 MONA의 단순화 버전 위에 구축되었다고 밝히며, 일부 복잡한 augmentation 전략은 제거했다고 한다. 이것이 오히려 분석에는 장점이지만, 실제로 원래의 MONA 전체 구성과 결합했을 때 항상 같은 개선을 보장하는지는 본문만으로는 완전히 분명하지 않다.

둘째, SAG에 대한 이론은 SG에 비해 다소 약하다. 논문은 SAG의 분산이 최악의 경우 SG의 2배 이하라고 보이지만, 실제 실험에서는 대체로 SG가 SAG보다 더 강한 결과를 보인다. 즉, SAG의 실질적 필요성이나 SG 대비 이점은 본문 기준으로는 아주 강하게 설득되지는 않는다.

셋째, “semantic similarity”를 group/grid 기반으로 근사하는 방식은 직관적이지만, 해부학적 구조가 항상 grid 내부에서 잘 묶인다고 보장되지는 않는다. 논문은 loss landscape 시각화와 경험적 결과로 이를 뒷받침하지만, 어떤 grid 크기나 grouping 방식이 최적인지에 대한 일반 이론은 충분히 제시되지 않았다. semantic segmentation 실험에서 9, 16, 25 grid를 비교한 정도가 전부이며, 더 적응적인 grouping이 필요한 상황도 있을 수 있다.

넷째, robustness를 이 논문은 “빠른 수렴, 작은 분산, collapse 완화, segmentation quality 향상”까지 넓게 포함해 사용한다. 그러나 adversarial robustness나 domain shift robustness처럼 다른 의미의 robustness는 다루지 않는다. 따라서 제목이나 서술만 보고 더 일반적인 robust learning을 기대하면 범위를 과대해석할 수 있다.

다섯째, 실제 임상 적용 측면에서 추론 시간 증가나 메모리 사용량 절감 정도를 정량적으로 자세히 제시하지는 않는다. 저자들은 minimal additional memory footprint라고 말하지만, 구체적 수치 비교는 제공된 텍스트에 없다.

## 6. 결론

이 논문은 semi-supervised medical image segmentation에서 contrastive learning의 핵심 병목을 variance reduction 문제로 재정의하고, 이를 해결하기 위해 ARCO라는 stratified group 기반 sampling framework를 제안했다. 핵심 기여는 SG와 SAG라는 두 샘플링 기법을 도입해 pixel/voxel-level contrastive loss의 gradient estimator 분산을 줄이고, 그 결과 더 안정적이고 label-efficient한 학습을 가능하게 했다는 점이다.

실험적으로는 ACDC, LiTS, MMWHS, LA, MP-MRI를 포함한 의료 영상 벤치마크와 일반 semantic segmentation 데이터셋에서 기존 SOTA semi-supervised 방법들을 일관되게 능가했다. 특히 극소량 라벨 환경에서 improvement가 더 두드러졌고, 이는 이 방법이 실제 의료 데이터처럼 라벨이 매우 부족한 상황에 적합함을 보여준다.

향후 연구 측면에서 이 논문은 중요한 출발점이 될 가능성이 크다. contrastive objective 자체를 복잡하게 만드는 대신, 샘플링과 추정기의 통계적 성질을 개선하는 방향이 dense prediction 문제에서 매우 유효할 수 있음을 보여주었기 때문이다. 실제 적용 측면에서도, 의료 영상처럼 long-tail class와 불안정한 경계가 문제되는 환경에서 보다 신뢰할 수 있는 semi-supervised segmentation 시스템을 설계하는 데 의미 있는 기반을 제공한다.
