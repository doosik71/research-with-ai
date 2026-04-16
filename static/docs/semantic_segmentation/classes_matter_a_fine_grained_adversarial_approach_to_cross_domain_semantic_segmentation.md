# Classes Matter: A Fine-grained Adversarial Approach to Cross-domain Semantic Segmentation

- **저자**: Haoran Wang, Tong Shen, Wei Zhang, Ling-Yu Duan, Tao Mei
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2007.09222

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation for semantic segmentation** 문제를 다룬다. 구체적으로는 라벨이 있는 source domain의 데이터로 학습한 segmentation 모델을 라벨이 없는 target domain에 적용할 때 발생하는 **domain shift**를 줄이는 것이 목표다. 예를 들어 GTA5나 SYNTHIA 같은 synthetic 데이터로 학습한 모델을 실제 도시 장면인 Cityscapes에 적용하면 성능이 크게 떨어지는데, 이 간극을 줄이려는 것이다.

기존의 많은 방법은 source와 target의 전체 feature distribution을 비슷하게 만드는 **global alignment**에 집중했다. 그러나 저자들은 이런 방식이 semantic class 구조를 충분히 반영하지 못한다고 지적한다. 즉, 전체적으로는 두 도메인이 가까워져도, 서로 다른 클래스의 feature들이 잘못 섞일 수 있다. semantic segmentation은 픽셀 단위로 클래스를 구분해야 하므로, 이런 class mismatch는 실제 성능 저하로 이어질 수 있다.

이 문제가 중요한 이유는 semantic segmentation이 자율주행, 도로 장면 이해 등 실제 응용과 직접 연결되기 때문이다. 현실에서는 대규모 pixel-level annotation을 만들기 어렵기 때문에 synthetic 데이터 활용이 매우 중요하다. 따라서 synthetic-to-real adaptation 또는 city-to-city adaptation 성능을 높이는 일은 실용적 가치가 크다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 단순히 “도메인이 source인지 target인지”만 구분하는 binary discriminator 대신, **도메인과 클래스 정보를 함께 다루는 fine-grained discriminator**를 사용하는 것이다. 저자들은 기존 adversarial adaptation이 $P(d \mid f)$를 모델링한다고 보고, 이를 더 세분화하여 $P(d, c \mid f)$를 직접 모델링하도록 바꾼다. 여기서 $d$는 domain, $c$는 class, $f$는 feature이다.

직관적으로 보면, 기존 adversarial learning은 “source feature와 target feature를 전반적으로 비슷하게 만들자”는 목표를 가진다. 반면 이 논문은 “같은 class에 속하는 source/target feature끼리 더 잘 맞추자”는 방향으로 문제를 다시 설계한다. 예를 들어 road는 road끼리, car는 car끼리 정렬되도록 유도하는 것이다. 이렇게 하면 전체 분포만 맞추다가 class boundary가 무너지는 문제를 줄일 수 있다.

이 논문의 차별점은 class 정보를 discriminator 안으로 **명시적으로 집어넣는다**는 데 있다. 기존 CLAN 같은 방법도 class-level alignment를 겨냥했지만, 저자들에 따르면 class 관계를 직접적으로 모델링하지는 않았다. 반면 FADA는 discriminator 출력 채널 자체를 class-aware하게 설계하고, binary domain label도 더 풍부한 형태의 **domain encoding**으로 일반화한다.

## 3. 상세 방법 설명

전체 segmentation 네트워크 $G$는 feature extractor $F$와 classifier $C$로 구성되며, $G = C \circ F$로 표현된다. source domain에는 라벨이 있는 데이터 $X_S = \{(x_i^{(s)}, y_i^{(s)})\}$가 있고, target domain에는 라벨이 없는 데이터 $X_T = \{x_j^{(t)}\}$만 있다. 두 도메인은 동일한 $K$개 semantic class를 공유한다고 가정한다.

기존 adversarial feature alignment에서는 binary domain discriminator $D$를 두고, source feature는 source로, target feature는 target으로 구분하도록 학습한다. 이때 discriminator는 다음 loss로 학습된다.

$$
L_D = - \sum_{i=1}^{n_s} (1-d)\log P(d=0 \mid f_i) - \sum_{j=1}^{n_t} d \log P(d=1 \mid f_j)
$$

여기서 source는 $d=0$, target은 $d=1$이다. segmentation network는 source 데이터에 대해 supervised segmentation loss를 받고, 동시에 target feature가 discriminator를 속이도록 adversarial loss를 받는다. source supervised loss는 다음과 같다.

$$
L_{seg} = - \sum_{i=1}^{n_s} \sum_{k=1}^{K} y_{ik}^{(s)} \log p_{ik}^{(s)}
$$

이 식은 표준 pixel-wise cross-entropy이며, source에서 semantic class를 잘 구분하는 feature를 배우게 한다. 이어서 target에 대해서는 discriminator가 source라고 믿도록 다음 adversarial loss를 사용한다.

$$
L_{adv} = - \sum_{j=1}^{n_t} \log P(d=0 \mid f_j)
$$

문제는 이 구조가 클래스별 정렬을 직접 보장하지 않는다는 점이다. 이를 해결하기 위해 저자들은 discriminator 출력 채널을 기존의 2개(source/target)에서 **$2K$개**로 확장한다. 즉, “source-road”, “source-car”, “target-road”, “target-car”처럼 domain-class 조합을 예측하는 구조로 바꾸는 것이다. 이때 discriminator는 $P(d, c \mid f)$를 모델링한다.

이 fine-grained discriminator를 학습시키기 위해 논문은 **domain encoding**을 도입한다. 기존 binary domain label은 source가 $[1, 0]$, target이 $[0,1]$이었다. 이를 $K$차원 class knowledge $a$를 사용해 source는 $[a; 0]$, target은 $[0; a]$로 일반화한다. 여기서 $a$는 classifier의 prediction에서 얻은 class-related signal이다. 즉, 단순히 “이 feature는 source다”가 아니라 “이 feature는 source이면서 class distribution은 이런 형태다”라는 더 풍부한 supervision을 준다.

이때 fine-grained discriminator loss는 다음과 같이 바뀐다.

$$
L_D =
- \sum_{i=1}^{n_s} \sum_{k=1}^{K} a_{ik}^{(s)} \log P(d=0, c=k \mid f_i)
- \sum_{j=1}^{n_t} \sum_{k=1}^{K} a_{jk}^{(t)} \log P(d=1, c=k \mid f_j)
$$

이 식의 의미는 각 feature를 특정 domain-class 조합으로 분류하도록 discriminator를 학습한다는 것이다. 여기서 중요한 점은 target에는 ground-truth class label이 없지만, classifier의 prediction 자체를 class knowledge로 사용한다는 점이다. 저자들은 source와 target이 semantic class를 공유하므로, source supervision으로 학습된 classifier의 예측이 target에서도 class 정보를 어느 정도 담고 있다고 본다.

이에 대응하는 adversarial loss는 다음과 같다.

$$
L_{adv} =
- \sum_{j=1}^{n_t} \sum_{k=1}^{K} a_{jk}^{(t)} \log P(d=0, c=k \mid f_j)
$$

이 loss는 target feature가 discriminator 입장에서 “target class-$k$”가 아니라 “source class-$k$”처럼 보이도록 만든다. 핵심은 단순히 source처럼 보이게 하는 것이 아니라, **class 관계를 보존한 채 source 쪽으로 정렬**한다는 것이다.

### Domain encoding을 만드는 방법

논문은 class knowledge $a$를 만드는 두 가지 일반적 방법을 제시한다.

첫째는 **one-hot hard labels**이다. classifier output $p_k$ 중 가장 큰 class 하나만 선택해 다음처럼 만든다.

$$
a_k =
\begin{cases}
1 & \text{if } k = \arg\max_k p_k \\
0 & \text{otherwise}
\end{cases}
$$

이 방식은 가장 확신이 높은 class만 사용한다. 실전에서는 noisy prediction의 영향을 줄이기 위해 confidence threshold보다 낮은 샘플은 무시할 수 있다고 설명한다.

둘째는 **multi-channel soft labels**이다. logits $z_k$와 temperature $T$를 사용해 다음처럼 정의한다.

$$
a_k = \frac{\exp(z_k/T)}{\sum_{j=1}^{K} \exp(z_j/T)}
$$

이 방식은 한 class만 고르지 않고 여러 class에 걸친 불확실성을 반영한다. 저자들은 soft label이 더 유연하고 성능도 더 좋았다고 보고한다. 또한 soft label이 특정 class에 과도하게 몰리는 것을 막기 위해 **confidence clipping**을 regularization으로 사용한다. 다만 clipping의 정확한 연산 형태는 제공된 본문에서 수식으로 자세히 쓰여 있지 않고, “threshold로 값을 잘라낸다”는 수준으로 설명되어 있다.

### 네트워크 구조와 학습 절차

전체 구조는 segmentation network와 fine-grained domain discriminator로 이루어진다. discriminator는 3개의 convolution layer를 사용하며 채널 수는 $\{256, 128, 2K\}$이고, kernel size는 $3 \times 3$, stride는 1이다. 마지막 층을 제외한 각 convolution 뒤에는 slope 0.2의 Leaky-ReLU가 붙는다.

학습은 먼저 source 데이터만으로 20k iteration pretraining을 수행한 뒤, 제안한 adaptation framework로 40k iteration fine-tuning을 한다. batch size는 8이며 source 4장, target 4장으로 구성된다. segmentation network는 SGD를 사용하고 momentum은 0.9, weight decay는 $10^{-4}$, 초기 learning rate는 $2.5 \times 10^{-4}$이다. discriminator는 Adam을 사용하며 $\beta_1 = 0.9$, $\beta_2 = 0.99$, 초기 learning rate는 $10^{-4}$이다. 두 경우 모두 poly learning rate policy를 사용하고, adversarial loss의 가중치 $\lambda_{adv}$는 0.001, temperature $T$는 1.8로 고정했다.

추론 시에는 adaptation용 discriminator는 제거하고, 적응이 끝난 segmentation network만 사용한다. 즉, 추가 비용 없이 inference가 가능하다는 장점이 있다.

## 4. 실험 및 결과

논문은 세 가지 대표적인 domain adaptation benchmark에서 실험한다. 첫째는 **Cityscapes $\rightarrow$ Cross-City**, 둘째는 **SYNTHIA $\rightarrow$ Cityscapes**, 셋째는 **GTA5 $\rightarrow$ Cityscapes**이다. Cityscapes는 실제 도시 장면 데이터셋이고, Cross-City는 여러 도시의 real-world street-view 데이터셋이다. SYNTHIA와 GTA5는 synthetic urban scene 데이터셋이다.

평가 지표는 각 class별 **IoU**와 전체 평균인 **mIoU**이다. IoU는

$$
IoU = \frac{TP}{TP + FP + FN}
$$

로 정의되며, semantic segmentation에서 표준적으로 쓰이는 지표다.

### Cityscapes $\rightarrow$ Cross-City

이 설정은 real-to-real adaptation으로, 다른 도시 간 장면 차이를 줄이는 문제다. 논문은 Rome, Rio, Tokyo, Taipei 네 도시에 대해 결과를 보고한다. 평균적으로 FADA는 source-only baseline 대비 **8.5%** 향상, 기존 최고 방법 대비 **2.25%** 향상을 달성했다고 설명한다.

표 1을 보면 DeepLab-v2 source model 대비 FADA는 네 도시 모두에서 mIoU를 개선한다. 예를 들어 Rome에서는 source DeepLab-v2가 50.9, AdaptSegNet이 53.8, FADA가 **54.7**이다. Rio에서는 48.2에서 **54.7**로 크게 향상되고, Tokyo에서는 42.8에서 **51.3**, Taipei에서는 39.6에서 **52.7**까지 오른다. 특히 person, rider, bike, mbike 같은 세부 클래스에서 눈에 띄는 향상이 보인다. 이는 class-aware alignment가 작은 객체나 혼동되기 쉬운 클래스에 특히 유리하다는 해석을 가능하게 한다.

### SYNTHIA $\rightarrow$ Cityscapes

이 설정은 synthetic-to-real adaptation으로 shift가 더 크다. VGG-16 backbone에서 source only의 mIoU는 25.6인데, feature-only baseline은 31.0, FADA는 **39.5**를 기록한다. mIoU*는 **46.0**이다. 저자들은 source model 대비 **16.4%** 향상이 있다고 요약한다.

ResNet-101 backbone에서는 source only가 33.5, feature-only baseline이 35.4, FADA가 **45.2**를 기록하며 mIoU*는 **52.5**이다. 기존 경쟁 방법들인 AdaptSegNet 46.7, CLAN 47.8, ADVENT 48.0과 비교해도 FADA가 가장 높다. 특히 Road, Build, Pole, TS, Car, Bus 같은 클래스에서 두드러진 개선이 보인다. 반면 Wall, Fence 같은 일부 클래스는 여전히 낮은 수치를 보이며, 어려운 클래스에 대해서는 adaptation이 완전히 해결되지 않았음을 보여준다.

### GTA5 $\rightarrow$ Cityscapes

이 역시 synthetic-to-real adaptation이며, 클래스 수가 19개로 더 많다. VGG-16 backbone 기준 source only는 28.3, feature-only baseline은 34.1, FADA는 **43.8**이다. ResNet-101 backbone 기준 source only는 36.8, feature-only baseline은 39.3, FADA는 **49.2**를 기록한다. 여기에 self distillation과 multi-scale testing을 추가한 FADA-MST는 **50.1**까지 올라간다.

기존 강한 baseline들과 비교하면, ResNet-101에서 AdaptPatch가 46.5, ADVENT가 45.5인데 FADA는 49.2, FADA-MST는 50.1이다. 즉, feature-level adaptation 계열 대비 대략 3~4% 이상의 실질적 개선이 있다. 클래스별로 보면 Road, Sidewalk, Building, Wall, Fence, Car, Bus, Bike 등에서 강한 성능을 보인다. 다만 Train 클래스는 여전히 매우 낮은 값이며, 저자들도 Cityscapes의 train 이미지가 GTA5의 bus와 더 비슷하다는 기존 지적을 언급한다. 즉, 일부 클래스는 시각적 대응 자체가 불안정해 adaptation 난도가 높다.

### Feature distribution 분석: CCD

이 논문의 중요한 실험적 포인트는 단순히 mIoU만 보여주는 데서 끝나지 않고, 정말로 class-level alignment가 잘 되었는지를 따로 분석했다는 점이다. 이를 위해 저자들은 **Class Center Distance (CCD)**를 제안한다.

$$
CCD(i) =
\frac{1}{K-1}
\sum_{j=1, j \ne i}^{K}
\frac{
\frac{1}{|S_i|}\sum_{x \in S_i} \|x - \mu_i\|_2^2
}{
\|\mu_i - \mu_j\|_2^2
}
$$

여기서 $\mu_i$는 class $i$의 중심, $S_i$는 class $i$에 속한 feature 집합이다. 이 값은 같은 클래스 내부가 얼마나 조밀한지와 다른 클래스와 얼마나 떨어져 있는지를 함께 반영한다. 값이 낮을수록 intra-class compactness는 높고 inter-class separation은 크다는 뜻이다.

논문은 AdaptSegNet, CLAN, FADA를 비교했을 때, FADA가 대부분의 클래스에서 더 낮은 CCD를 보였고 평균 CCD도 **1.1**로 가장 낮았다고 보고한다. AdaptSegNet은 1.9, CLAN은 1.3이다. 이 결과는 FADA가 단순히 target 성능만 올린 것이 아니라, 실제로 class structure를 더 잘 보존하며 정렬했다는 논문의 핵심 주장을 뒷받침한다.

### Ablation study

GTA5 $\rightarrow$ Cityscapes, ResNet-101 기준으로 ablation 결과를 제시한다. source only는 36.8이고, fine-grained adversarial training만 추가하면 **46.9**가 된다. 여기에 self distillation을 더하면 **49.2**, multi-scale testing까지 더하면 **50.1**이다. 즉, 가장 큰 기여는 명백히 fine-grained adversarial training 자체에서 나온다.

또한 hard label과 soft label 비교도 수행했다. GTA5에서는 baseline 39.4, hard labels 45.7, soft labels **46.9**이고, SYNTHIA에서는 baseline 35.4, hard labels 40.8, soft labels **41.5**이다. 따라서 두 방식 모두 이득이 있지만 soft label이 좀 더 우수하다.

confidence clipping threshold 실험에서는 GTA5 $\rightarrow$ Cityscapes에서 threshold 0.7일 때 46.2, 0.8일 때 46.3, 0.9일 때 **46.9**, 1.0일 때 45.7이다. 여기서 1.0은 clipping이 없는 경우이므로, 적절한 clipping이 noisy soft labels에 대한 regularization 역할을 한다는 해석이 가능하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법 설계가 매우 일관된다는 점이다. 저자들은 global alignment의 구조적 한계를 분명히 짚고, 이를 해결하기 위해 discriminator 자체를 class-aware하게 재설계했다. 단순히 loss weight를 바꾸는 수준이 아니라, $P(d \mid f)$를 $P(d,c \mid f)$로 확장해 문제를 더 직접적으로 다뤘다는 점이 설득력 있다.

또 다른 강점은 target label이 없는 상황에서도 classifier prediction을 이용해 class knowledge를 끌어낸다는 점이다. 이는 unsupervised adaptation 설정을 깨지 않으면서도 class-level supervision 비슷한 효과를 만들어낸다. 또한 inference 시 adaptation module이 제거되므로 실제 사용 비용이 크지 않다.

실험적으로도 강점이 분명하다. 세 가지 대표 benchmark에서 일관되게 strong baseline을 넘고, 단순 성능 표뿐 아니라 CCD 분석을 통해 “정말 class-level alignment가 개선되었는가”를 보여주려 했다. 특히 CCD는 이 논문의 핵심 주장과 맞물린 분석 도구로서 의미가 있다.

한계도 있다. 첫째, target domain의 class knowledge를 classifier prediction에 의존하므로, 초기에 prediction이 불안정하거나 심한 오분류가 있으면 잘못된 alignment 신호를 줄 가능성이 있다. 저자들도 hard label에서는 confidence thresholding, soft label에서는 clipping을 써서 이를 완화하려 했는데, 이는 곧 방법이 prediction noise에 민감할 수 있음을 시사한다.

둘째, 일부 어려운 클래스에서는 여전히 성능이 낮다. 예를 들어 SYNTHIA나 GTA5 실험에서 Wall, Fence, Train 같은 클래스는 적응 후에도 낮은 IoU를 보인다. 즉, class-level alignment가 모든 클래스에 동일하게 잘 작동하는 것은 아니다.

셋째, self distillation과 multi-scale testing이 최종 성능을 더 끌어올리지만, 제공된 본문에서는 이 두 요소의 구체적인 구현이 상세히 설명되어 있지 않다. 따라서 최종 최고 성능이 순수하게 FADA 설계만의 효과인지, 부가적인 training/inference trick의 영향이 어느 정도인지 세밀하게 분리해서 보려면 본문만으로는 정보가 조금 부족하다.

비판적으로 보면, 이 방법은 클래스 구조를 반영한 discriminator를 도입함으로써 global alignment의 약점을 상당히 잘 보완한다. 다만 target prediction을 supervision 신호처럼 재사용하는 구조이기 때문에, domain gap가 극단적으로 큰 상황에서는 self-reinforcing error의 위험이 있을 수 있다. 논문은 이 문제를 empirical하게 어느 정도 통제했지만, 이론적으로 얼마나 안정적인지는 본문에서 깊게 다루지 않는다.

## 6. 결론

이 논문은 cross-domain semantic segmentation에서 global adversarial alignment의 한계를 지적하고, 이를 해결하기 위해 **fine-grained adversarial learning framework (FADA)**를 제안했다. 핵심은 discriminator가 domain만 구분하는 것이 아니라 domain과 class를 함께 다루도록 바꾸고, classifier prediction으로부터 만든 domain encoding을 사용해 class-level feature alignment를 유도하는 것이다.

주요 기여는 세 가지로 요약할 수 있다. 첫째, semantic class 정보를 명시적으로 반영한 adversarial adaptation 구조를 제안했다. 둘째, CCD 분석을 통해 실제로 class-level alignment가 개선되었음을 보였다. 셋째, Cityscapes $\rightarrow$ Cross-City, SYNTHIA $\rightarrow$ Cityscapes, GTA5 $\rightarrow$ Cityscapes의 세 benchmark에서 강력한 성능을 달성했다.

실제 적용 측면에서도 의미가 있다. synthetic 데이터나 다른 도시 데이터에서 학습한 segmentation 모델을 현실 환경에 옮겨야 하는 자율주행, 도로 장면 이해 같은 문제에서 유용할 가능성이 크다. 또한 이후 연구에서는 이 논문의 아이디어를 확장해 더 정교한 class uncertainty 처리, pseudo-label 품질 개선, transformer 기반 segmentation backbone과의 결합 같은 방향으로 이어갈 수 있다. 전체적으로 이 논문은 domain adaptive semantic segmentation에서 **“클래스 구조를 무시하지 말아야 한다”**는 점을 명확히 보여준, 실험적으로도 강한 설득력을 가진 작업이다.
