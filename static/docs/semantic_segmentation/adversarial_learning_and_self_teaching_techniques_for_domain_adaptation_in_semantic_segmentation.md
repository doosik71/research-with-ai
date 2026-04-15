# Adversarial Learning and Self-Teaching Techniques for Domain Adaptation in Semantic Segmentation

- **저자**: Umberto Michieli, Matteo Biasetton, Gianluca Agresti, Pietro Zanuttigh
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/1909.00781

## 1. 논문 개요

이 논문은 semantic segmentation 모델을 synthetic dataset으로 학습한 뒤, label이 없는 real-world dataset에 잘 적응시키는 Unsupervised Domain Adaptation (UDA) 방법을 제안한다. 문제의 핵심은 synthetic 이미지에는 pixel-level annotation을 대량으로 만들 수 있지만, 실제 도로 장면에서는 정밀한 segmentation label을 만드는 비용이 매우 크다는 점이다. 따라서 synthetic-to-real adaptation이 가능해지면 자율주행용 장면 이해 시스템을 훨씬 저렴하게 구축할 수 있다.

논문이 다루는 연구 문제는 명확하다. GTA5나 SYNTHIA 같은 synthetic road scene으로 학습한 segmentation network는 Cityscapes나 Mapillary 같은 실제 데이터에 적용될 때 domain shift 때문에 성능이 크게 떨어진다. 이 논문은 target domain의 label 없이, source domain의 label과 target domain의 unlabeled image만으로 이 간극을 줄이려 한다.

저자들은 이 문제를 세 가지 학습 신호의 결합으로 해결한다. 첫째는 synthetic label을 사용하는 supervised cross-entropy loss, 둘째는 segmentation output이 ground-truth segmentation distribution처럼 보이도록 만드는 adversarial loss, 셋째는 discriminator가 신뢰할 만하다고 본 target prediction을 pseudo-label처럼 다시 활용하는 self-teaching loss이다. 특히 self-teaching 부분을 단순한 hard mask가 아니라 soft weighting, region growing, class frequency weighting으로 개선한 것이 이 논문의 핵심 기여다.

## 2. 핵심 아이디어

중심 아이디어는 discriminator를 단순히 “진짜/가짜” 판별기로만 쓰지 않고, pixel-wise confidence estimator로도 활용하는 것이다. 즉, discriminator가 어떤 위치의 segmentation 출력을 ground truth와 유사하다고 판단하면, 그 위치의 예측은 비교적 신뢰할 수 있다고 보고 self-training에 다시 쓴다.

기존 adversarial UDA나 semi-supervised segmentation과의 차이는 세 가지로 요약된다. 첫째, discriminator가 이미지 단위의 scalar가 아니라 pixel-level confidence map을 출력한다. 둘째, self-teaching에서 hard threshold로 선택된 픽셀만 쓰는 대신, discriminator 출력을 soft weight로 직접 loss에 반영한다. 셋째, confidence가 높은 seed 주변으로 region growing을 적용해 edge 영역과 작은 물체까지 pseudo-supervision을 확장한다. 논문은 이전 버전 [3] 및 Hung et al. [5]와 비교해, 이 개선이 small object와 rare class 성능 향상에 중요하다고 주장한다.

또 하나의 중요한 설계는 class frequency weighting이다. target에는 ground truth가 없으므로 rare class는 pseudo-label 기반 self-training 과정에서 쉽게 사라질 수 있다. 이를 막기 위해 source dataset에서 계산한 class frequency를 이용해 드문 클래스일수록 더 큰 가중치를 준다. 저자들은 이 설계가 pole, traffic light 같은 작은 구조물이 dominant class인 road, building 등에 흡수되는 문제를 완화한다고 설명한다.

## 3. 상세 방법 설명

전체 구조는 generator $G$와 discriminator $D$로 이루어진 adversarial framework이다. $G$는 semantic segmentation network이며, 실험에서는 DeepLab v2 with ResNet-101 backbone을 사용했다. $D$는 fully convolutional discriminator로, segmentation probability map 또는 one-hot ground truth segmentation map을 입력받아 각 픽셀마다 “ground truth처럼 보이는 정도”를 출력한다.

### Supervised segmentation loss

source domain의 labeled synthetic image $X_n^s$와 정답 $Y_n^s$에 대해서는 일반적인 multi-class cross-entropy를 사용한다.

$$
L_{G,1} = - \sum_{p \in X_n^s} \sum_{c \in C} Y_n^s(p)[c] \cdot \log(G(X_n^s)(p)[c])
$$

여기서 $p$는 픽셀 위치, $c$는 semantic class이다. 이 항은 synthetic label이 있는 경우에만 계산 가능하다. 즉, 모델이 기본적인 segmentation 능력을 source에서 배우는 역할을 한다.

### Discriminator와 adversarial loss

discriminator $D$는 ground truth segmentation map은 class 1, generator output은 class 0으로 보도록 학습된다. 중요한 점은 generator output이 source 이미지에서 왔든 target 이미지에서 왔든 모두 “generated”로 본다는 점이다. discriminator loss는 다음과 같다.

$$
L_D = - \sum_{p \in X_n^{s,t}} \log(1 - D(G(X_n^{s,t}))(p)) + \log(D(Y_n^s)(p))
$$

즉, generator output에는 낮은 score를, ground truth one-hot map에는 높은 score를 주도록 학습한다.

반대로 generator는 자신의 segmentation 결과가 discriminator 입장에서 ground truth처럼 보이도록 학습된다. 이것이 adversarial loss $L_{G,2}^{s,t}$이다.

$$
L_{G,2}^{s,t} = - \sum_{p \in X_n^{s,t}} \log(D(G(X_n^{s,t}))(p))
$$

이 loss는 source와 target 모두에서 사용된다. source에서는 labeled supervised learning을 보조하고, target에서는 label 없이도 generator가 “ground-truth-like output distribution”을 만들도록 압박한다. 논문의 관점에서 이는 output space alignment에 가까운 효과를 낸다.

### Self-teaching loss

세 번째 요소는 target real image에 대한 self-teaching이다. 우선 $G(X_n^t)$의 class-wise argmax를 취해 pseudo-label $\hat{Y}_n$를 만든다. 하지만 모든 픽셀을 동일하게 신뢰하지 않고, discriminator와 region growing에서 얻은 confidence $D_R(X_n^t)(p)$, 그리고 class weight $W_c^s$를 곱해 loss를 구성한다.

$$
L_{G,3} = - \sum_{p \in X_n^t} \sum_{c \in C} D_R(X_n^t)(p) \cdot W_c^s \cdot \hat{Y}_n(p)[c] \cdot \log(G(X_n^t)(p)[c])
$$

이 식의 의미는 간단하다. target prediction 중에서도 신뢰도가 높은 픽셀, 그리고 rare class로 예측된 픽셀에 더 큰 학습 신호를 준다는 것이다.

### Confidence mask와 region growing

먼저 discriminator output에 threshold $T_u$를 적용해 confident seed mask $m_{T_u}$를 만든다.

$$
m_{T_u}(p) =
\begin{cases}
1 & \text{if } D(G(X_n^t))(p) > T_u \\
0 & \text{otherwise}
\end{cases}
$$

그 다음, seed 픽셀 $p$가 class $c^*$로 예측되었다면, 인접 픽셀 $p'$에서 그 클래스의 generator probability가 충분히 높을 때 region을 확장한다. 조건은 $G(X_n^t)(p')[c^*] > T_R$이다. 이렇게 얻은 확장 마스크를 $m_{T_u}^R$라고 둔다.

최종 confidence map은 확장된 마스크 내부에서는 discriminator score를 유지하고, 바깥은 0으로 둔다.

$$
D_R(X_n^t) = m_{T_u}^R \cdot D(G(X_n^t))
$$

논문은 $T_u = 0.2$, $T_R = 1 - 10^{-5}$를 사용했다. 매우 높은 $T_R$를 사용하므로 region growing은 아무 곳이나 퍼지는 것이 아니라, 이미 매우 강한 class confidence를 가진 이웃에만 확장된다. 저자들은 이 설계가 edge 근처와 small object를 hard mask 방식보다 더 잘 보존한다고 설명한다.

### Class frequency weighting

class weighting은 source dataset에서 클래스별 픽셀 비율을 계산한 뒤, 그 보완값 $1 - \text{frequency}$를 weight로 쓴다.

$$
W_c^s = 1 - \frac{\sum_n |p \in X_n^s \wedge p \in c|}{\sum_n |p \in X_n^s|}
$$

자주 등장하는 클래스는 작은 가중치를, 드문 클래스는 큰 가중치를 받는다. target ground truth가 없기 때문에 target 통계를 쓸 수 없고, 대신 source의 class statistics를 이용한다는 점이 논문에 명시되어 있다.

### 최종 학습 목표와 절차

generator의 최종 loss는 세 항의 가중합이다.

$$
L_{\text{full}} = L_{G,1} + w^{s,t} L_{G,2}^{s,t} + w' L_{G,3}
$$

본문 후반과 Table VII를 종합하면 기본 하이퍼파라미터는 다음과 같다.

- GTA5에서 적응할 때: $w_s = 10^{-2}, w_t = 10^{-4}, w' = 10^{-3}$
- SYNTHIA에서 적응할 때: $w_s = 10^{-2}, w_t = 10^{-3}, w' = 10^{-1}$

또한 학습 초반 5000 step 동안은 $L_{G,3}$를 비활성화해 $w' = 0$으로 둔다. 이는 discriminator가 먼저 어느 정도 신뢰할 수 있는 confidence map을 만들 수 있게 하기 위한 장치다. 전체 학습은 20000 iteration 동안 수행된다.

discriminator 구조는 5개의 convolution layer로 구성되며, 모두 $4 \times 4$ kernel, stride 2, Leaky ReLU를 사용한다. 채널 수는 $64, 64, 128, 128, 1$이고, 마지막에 bilinear upsampling으로 입력 해상도에 맞춘다.

## 4. 실험 및 결과

실험은 synthetic source와 real target의 네 조합을 중심으로 이루어진다.

- GTA5 $\rightarrow$ Cityscapes
- SYNTHIA $\rightarrow$ Cityscapes
- GTA5 $\rightarrow$ Mapillary
- SYNTHIA $\rightarrow$ Mapillary

source는 labeled synthetic data, target은 unlabeled real training set을 쓴다. 평가는 real dataset의 validation split에서 mean Intersection over Union, 즉 mIoU로 수행한다.

### 데이터셋과 설정

GTA5는 24,966장의 synthetic urban scene을 포함하며, 이 중 23,966장을 학습, 1,000장을 validation으로 사용했다. Cityscapes와 호환되는 19개 클래스를 제공한다. SYNTHIA-RAND-CITYSCAPES는 9,400장의 synthetic image를 가지며, 이 중 9,300장을 학습, 100장을 validation으로 썼다. Cityscapes와 16개 클래스가 호환된다.

Cityscapes는 2,975장의 training image와 500장의 validation image를 갖는다. adaptation에는 training image를 label 없이 사용하고, 평가는 validation set에서 수행한다. Mapillary는 20,000장의 고해상도 image 중 18,000장을 unlabeled adaptation에, 2,000장을 validation 평가에 사용했다. label은 Cityscapes taxonomy에 맞춰 remap했다.

모든 training image는 메모리 제한 때문에 $750 \times 375$로 resize/crop했고, test는 원본 해상도에서 수행했다. generator는 SGD with momentum $0.9$, weight decay $10^{-4}$로 학습하고, discriminator는 Adam을 사용했다. learning rate는 둘 다 $10^{-4}$에서 시작해 polynomial decay로 $10^{-6}$까지 낮췄다.

### Cityscapes 결과

GTA5에서 Cityscapes로 적응할 때, 단순 supervised training만 하면 mIoU가 27.9%다. 제안 방법은 이를 33.3%로 끌어올려 5.4%p 향상을 보였다. 비교 대상 중 Hung et al. [5]는 29.0%, Zhang et al. [16]은 28.9%, Biasetton et al. [3]는 30.4%였다. 즉, 이 논문의 full method가 가장 높다.

세부적으로 보면 road는 45.3에서 81.0, building은 50.1에서 65.8, terrain은 21.1에서 33.0, car는 77.9에서 80.3으로 올랐다. 드문 클래스도 일부 개선되는데, rider는 1.7에서 6.2, motorcycle은 4.7에서 8.5로 증가했다. 다만 traffic sign은 0.7에서 0.2로, bicycle은 0.0으로 유지되어 모든 클래스가 고르게 좋아진 것은 아니다.

SYNTHIA에서 Cityscapes로 적응하는 더 어려운 설정에서는 supervised baseline이 25.4%, 제안 방법이 31.3%다. 향상폭은 5.9%p로 비슷하거나 조금 더 크다. Hung et al. [5]는 29.4%, Zhang et al. [16]은 29.0%, Biasetton et al. [3]는 30.2%였다. 여기서도 제안 방법이 최고다. 특히 road는 10.3에서 80.7, building은 35.5에서 75.0으로 크게 향상된다. 반면 sidewalk, wall, fence, traffic light 같은 일부 클래스는 여전히 매우 어렵다.

논문은 qualitative result에서도 차이를 보여 준다. supervised-only 모델은 road surface에 noise가 많고, pole이나 rider 같은 small object가 잘 사라진다. Hung et al. [5]는 road는 개선하지만 small structure가 약하고, 이전 방법 [3]는 edge와 사람 표현이 더 낫지만 일부 artifact가 남는다. 제안 방법은 road, sidewalk, terrain, pole 등에서 더 안정적인 모양과 cleaner output을 보인다고 설명한다.

### Mapillary 결과

GTA5에서 Mapillary로 적응할 때 supervised baseline은 32.7%, 제안 방법은 38.5%다. 향상폭은 5.8%p이며, Cityscapes 실험과 비슷한 규모다. Hung et al. [5]는 34.4%, Biasetton et al. [3]는 35.2%였다. 이 setting에서도 제안 방법이 가장 높다.

클래스별로 보면 road는 66.5에서 79.9, building은 46.1에서 73.4, terrain은 25.6에서 39.6, motorcycle은 10.1에서 24.8로 상승한다. 다만 pole은 24.8에서 20.9로 오히려 줄고, traffic light와 traffic sign은 매우 낮은 값에 머문다. 즉, 개선은 뚜렷하지만 모든 class가 일관되게 좋아지는 것은 아니다.

SYNTHIA에서 Mapillary로 적응할 때는 baseline이 26.6%, 제안 방법이 32.0%다. Hung et al. [5]는 27.0%, Biasetton et al. [3]는 28.2%였다. 이 설정은 역시 가장 어렵고, 저자들은 SYNTHIA의 road, sidewalk texture realism이 낮아 domain gap이 더 크다고 해석한다. 그럼에도 road는 14.7에서 57.6, building은 34.6에서 62.1로 크게 상승한다.

### Ablation study

ablation은 Cityscapes target에서 수행되었다. 핵심 관찰은 다음과 같다.

supervised loss $L_{G,1}$만 사용하면 mIoU는 GTA5 기준 27.9, SYNTHIA 기준 25.4다. adversarial loss $L_{G,2}$를 더하면 29.4, 27.4로 오른다. self-teaching $L_{G,3}$까지 넣으면 30.4, 30.2가 된다. 즉, adversarial alignment와 pseudo-label self-training이 각각 추가 이득을 제공한다.

여기에 region growing을 넣으면 32.6, 31.0으로 추가 상승한다. 특히 GTA5에서 +2.2%p가 붙는 것이 눈에 띈다. discriminator-based soft weighting만 넣은 경우는 32.8, 30.2로, GTA5에서는 이득이 크지만 SYNTHIA에서는 거의 효과가 없다. class weighting을 제거하면 33.1, 30.2가 되어 full model 33.3, 31.3보다 소폭 낮다. 즉, class weighting의 효과는 크지만 폭발적이지는 않고, 안정적으로 rare class 보호에 기여하는 정도로 해석할 수 있다.

하이퍼파라미터 ablation(Table VII)에서는 source adversarial weight $w_s$가 가장 민감한 변수로 보인다. 이 값을 10배 키우면 GTA5 기준 27.2, SYNTHIA 기준 23.3까지 떨어지고, 0.1배로 줄여도 30.8, 23.4로 하락한다. 반면 $w_t$, $w'$ 변화에는 상대적으로 덜 민감하다. 이는 source 쪽 adversarial regularization이 전체 학습 균형에 중요한 역할을 한다는 뜻이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 adversarial learning과 self-teaching을 느슨하게 병렬로 둔 것이 아니라, discriminator를 두 목적에 동시에 활용해 학습 신호를 정교하게 만든 점이다. discriminator가 output distribution alignment를 돕는 동시에 pseudo-label의 신뢰도를 측정하는 도구로도 쓰이기 때문에, 동일한 모듈이 두 단계의 adaptation을 연결한다. 설계가 비교적 단순하면서도 실험적으로 일관된 이득을 보여 준다는 점이 설득력 있다.

두 번째 강점은 region growing과 class frequency weighting의 조합이다. 논문은 기존 hard-mask self-training이 edge와 small object를 잘 버린다고 지적하고, 실제로 ablation에서 region growing이 특히 GTA5 기준으로 큰 향상을 만든다. class weighting도 rare class collapse를 막는 방향으로 작동하며, qualitative 결과와 클래스별 성능표가 이 주장을 어느 정도 뒷받침한다.

세 번째 강점은 다양한 dataset 조합에서 성능 향상이 반복된다는 점이다. Cityscapes와 Mapillary 모두에서, GTA5와 SYNTHIA 모두를 source로 썼을 때 약 5%p 내외의 개선을 보였다. 이는 특정 target에만 맞춘 기법이 아니라는 인상을 준다.

한편 한계도 분명하다. 첫째, 일부 클래스는 여전히 거의 학습되지 않는다. 예를 들어 traffic light, traffic sign, bicycle 등은 여러 설정에서 0에 가깝거나 매우 낮다. 즉, rare class 보존을 위해 weighting을 도입했지만, 실제 해결 정도는 제한적이다.

둘째, discriminator output을 reliability로 해석하는 가정은 직관적이지만 이론적으로 강하게 정당화되지는 않는다. 논문도 generator output이 빠르게 Dirac-like distribution에 가까워지므로 discriminator가 단순히 “soft vs one-hot” 차이만 보는 것은 아니라고 논의하지만, 이것이 충분한 설명인지는 별개다. 즉, confidence calibration 측면의 엄밀한 검증은 부족하다.

셋째, self-training은 본질적으로 초기 pseudo-label 품질에 의존한다. 이를 완화하기 위해 첫 5000 step 동안 $L_{G,3}$를 끄지만, domain gap이 매우 큰 경우 잘못된 pseudo-label이 reinforcement될 가능성은 남아 있다. SYNTHIA 실험에서 일부 클래스가 여전히 매우 불안정한 것은 이 한계를 시사한다.

넷째, 실험 비교의 일부는 backbone 차이가 있다. 예를 들어 [48], [16]은 다른 generator architecture를 사용하므로 완전한 apples-to-apples 비교는 아니다. 논문도 이를 암묵적으로 인정한다. 다만 [5], [3]처럼 더 직접적인 비교 대상에서도 성능이 앞선다는 점은 긍정적이다.

마지막으로, 본문에는 계산 비용이나 메모리 오버헤드에 대한 상세 분석이 많지 않다. 학습 시간이 약 20시간이라고만 제시되며, discriminator와 region growing이 얼마나 실용적인 추가 비용을 유발하는지는 깊게 분석되지 않는다.

## 6. 결론

이 논문은 synthetic-to-real semantic segmentation UDA를 위해 supervised learning, pixel-level adversarial learning, self-teaching을 결합한 프레임워크를 제안한다. 그중에서도 핵심은 discriminator를 confidence estimator로 재해석하고, 이를 soft weighting과 region growing으로 보강해 pseudo-label supervision의 품질을 높인 점이다. 또한 class frequency weighting을 통해 rare class 망각을 완화하려 했다.

실험적으로는 GTA5/SYNTHIA에서 Cityscapes/Mapillary로의 적응에서 일관된 mIoU 향상을 보였고, 이전 버전 및 유사한 self-training 기반 방법보다 더 좋은 결과를 냈다. 따라서 이 연구는 실제 자율주행 장면 이해에서 synthetic label을 더 효과적으로 활용하는 실용적 방향을 제시한다고 볼 수 있다.

향후 연구 측면에서는 논문이 직접 언급하듯 더 강한 backbone과의 결합, 그리고 더 realistic한 synthetic image를 생성하는 generative model과의 통합이 자연스러운 다음 단계다. 또한 rare class calibration, pseudo-label quality estimation, confidence의 이론적 정당화 같은 문제를 더 깊게 다룬다면 방법의 완성도가 높아질 것이다.
