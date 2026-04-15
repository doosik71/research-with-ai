# ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation

- **저자**: Tuan-Hung Vu, Himalaya Jain, Maxime Bucher, Matthieu Cord, Patrick Pérez
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1811.12833

## 1. 논문 개요

이 논문은 semantic segmentation에서의 unsupervised domain adaptation(UDA)를 다룬다. 구체적으로는 source domain에는 pixel-level annotation이 존재하지만, target domain에는 라벨이 없는 상황에서 target domain 성능을 높이는 것이 목표다. 논문이 특히 겨냥하는 대표적 문제는 synthetic-to-real setting으로, 예를 들어 GTA5나 SYNTHIA 같은 synthetic 데이터로 학습한 segmentation 모델을 실제 도시 주행 영상인 Cityscapes에 적용하면 성능이 크게 떨어지는 현상이다.

저자들은 이 성능 저하의 중요한 원인 중 하나를 prediction uncertainty의 차이로 본다. source domain에서만 학습한 모델은 source-like image에 대해서는 over-confident, 즉 low-entropy prediction을 내는 반면, target-like image에서는 under-confident, 즉 high-entropy prediction을 자주 만든다는 것이다. 논문의 핵심 문제의식은 여기서 출발한다. target image에 대해서도 source처럼 더 확신 있는 예측, 즉 낮은 entropy를 유도하면 domain gap을 줄일 수 있다는 주장이다.

이 문제는 실제 응용 측면에서 중요하다. 자율주행처럼 다양한 날씨, 조명, 도시 환경에서 안정적으로 동작해야 하는 시스템에서는 학습 데이터와 실제 배포 환경의 차이가 빈번하다. 그런데 실제 환경마다 대규모 pixel annotation을 수집하는 것은 매우 비싸므로, synthetic data와 unlabeled real data만으로 적응하는 UDA는 현실적인 가치가 크다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 target domain prediction의 entropy를 줄이는 것이다. 기존 UDA segmentation 연구는 주로 feature distribution alignment, output-space adversarial alignment, self-training, image translation 등에 집중해 왔다. 반면 이 논문은 prediction confidence 자체를 직접적인 학습 신호로 사용한다는 점이 특징이다.

저자들은 entropy minimization을 두 가지 방식으로 구현한다. 첫째는 target prediction의 entropy를 직접 줄이는 direct entropy minimization이고, 둘째는 weighted self-information 분포를 source와 target 사이에서 adversarial하게 맞추는 indirect entropy minimization이다. 후자는 단순히 각 픽셀의 불확실성을 줄이는 것뿐 아니라, semantic layout의 구조적 일관성까지 함께 적응시키려는 목적을 가진다.

또 하나의 중요한 포인트는 self-training과의 관계를 명시적으로 설명했다는 점이다. self-training은 confidence가 높은 target prediction만 pseudo-label로 사용하지만, 이 논문의 entropy loss는 그런 hard pseudo-label 대신 soft assignment 형태로 전체 예측 분포를 다룬다. 저자들의 해석에 따르면 entropy minimization은 pseudo-label cross-entropy의 부드러운 형태라고 볼 수 있다. 따라서 threshold scheduling 같은 복잡한 절차 없이도 작동할 수 있다.

## 3. 상세 방법 설명

전체적으로 모델은 기본 semantic segmentation network $F$ 위에 domain adaptation을 위한 추가 학습 분기를 붙여 학습한다. 기본 supervised 학습은 source image $x_s$와 source label $y_s$를 이용한 segmentation loss로 수행된다. 각 픽셀에 대해 class probability를 출력하는 softmax 기반 예측 맵 $P_x$를 만들고, source에서는 일반적인 pixel-wise cross-entropy를 사용한다.

source supervised loss는 다음과 같다.

$$
L_{\mathrm{seg}}(x_s, y_s)
=
-
\sum_{h=1}^{H}
\sum_{w=1}^{W}
\sum_{c=1}^{C}
y_s^{(h,w,c)} \log P_{x_s}^{(h,w,c)}.
$$

source만으로 학습할 때의 목적함수는 다음과 같다.

$$
\min_{\theta_F}
\frac{1}{|X_s|}
\sum_{x_s \in X_s}
L_{\mathrm{seg}}(x_s, y_s).
$$

### Direct entropy minimization

target domain에는 정답 라벨 $y_t$가 없으므로 supervised loss를 직접 쓸 수 없다. 여기서 저자들은 target prediction의 entropy를 직접 줄이는 loss를 도입한다. 각 픽셀 $(h,w)$에서 normalized Shannon entropy를 다음과 같이 정의한다.

$$
E_{x_t}^{(h,w)}
=
-\frac{1}{\log(C)}
\sum_{c=1}^{C}
P_{x_t}^{(h,w,c)} \log P_{x_t}^{(h,w,c)}.
$$

이 값은 $0$에서 $1$ 사이로 정규화되며, 특정 클래스 하나에 확신 있게 몰려 있으면 작고, 여러 클래스에 고르게 퍼져 있으면 크다. 전체 target image의 entropy loss는 모든 픽셀 entropy의 합이다.

$$
L_{\mathrm{ent}}(x_t)
=
\sum_{h,w}
E_{x_t}^{(h,w)}.
$$

최종적으로 source supervised loss와 target entropy loss를 함께 최소화한다.

$$
\min_{\theta_F}
\frac{1}{|X_s|}
\sum_{x_s}
L_{\mathrm{seg}}(x_s, y_s)
+
\lambda_{\mathrm{ent}}
\frac{1}{|X_t|}
\sum_{x_t}
L_{\mathrm{ent}}(x_t).
$$

직관적으로 보면, 이 loss는 target prediction이 애매하게 퍼지는 것을 막고, 더 뚜렷한 class assignment를 하도록 decision boundary를 target distribution의 low-density region 쪽으로 밀어낸다.

### Self-training과의 연결

논문은 이 entropy loss를 pseudo-label self-training과 비교한다. self-training에서는 confidence가 높은 픽셀 집합 $K$만 골라 pseudo-label $\hat y_t$를 만들고 cross-entropy를 적용한다.

$$
L_{\mathrm{seg}}(x_t, \hat y_t)
=
-
\sum_{(h,w)\in K}
\sum_{c=1}^{C}
\hat y_t^{(h,w,c)} \log P_{x_t}^{(h,w,c)}.
$$

저자들은 entropy loss가 이 hard pseudo-label 기반 목적함수의 soft-assignment 버전이라고 설명한다. 즉, 특정 클래스만 1로 두는 대신 전체 확률분포를 그대로 이용해 uncertainty 자체를 벌점으로 준다. 이 때문에 confidence threshold를 정교하게 조절해야 하는 부담이 줄어든다.

### Adversarial entropy minimization

직접 entropy를 줄이는 방식은 픽셀별 독립적 계산이므로 구조적 관계를 충분히 반영하지 못할 수 있다. 이를 보완하기 위해 저자들은 weighted self-information 공간에서 adversarial learning을 수행한다.

각 픽셀의 class score $P_x^{(h,w,c)}$에 대해 self-information은 $-\log P_x^{(h,w,c)}$이다. 논문은 여기에 class probability를 곱한 weighted self-information vector를 사용한다.

$$
I_x^{(h,w)}
=
-
P_x^{(h,w)} \cdot \log P_x^{(h,w)}.
$$

이 표현은 Shannon entropy를 클래스 축으로 펼쳐놓은 형태로 볼 수 있다. 즉, entropy는 각 클래스별 weighted self-information의 합 또는 기대값에 해당한다. 저자들은 이 $I_x$를 discriminator $D$의 입력으로 넣는다. discriminator는 입력이 source에서 왔는지 target에서 왔는지를 판별한다.

discriminator의 학습 목적은 다음과 같다.

$$
\min_{\theta_D}
\frac{1}{|X_s|}
\sum_{x_s}
L_D(I_{x_s}, 1)
+
\frac{1}{|X_t|}
\sum_{x_t}
L_D(I_{x_t}, 0).
$$

반대로 segmentation network는 target의 weighted self-information이 source처럼 보이도록 discriminator를 속이도록 학습된다.

$$
\min_{\theta_F}
\frac{1}{|X_t|}
\sum_{x_t}
L_D(I_{x_t}, 1).
$$

이를 source segmentation loss와 합치면 다음이 된다.

$$
\min_{\theta_F}
\frac{1}{|X_s|}
\sum_{x_s}
L_{\mathrm{seg}}(x_s, y_s)
+
\lambda_{\mathrm{adv}}
\frac{1}{|X_t|}
\sum_{x_t}
L_D(I_{x_t}, 1).
$$

이 방식의 핵심은 단순히 entropy를 낮추는 데 그치지 않고, source와 target의 structured output distribution을 weighted self-information 공간에서 정렬한다는 점이다. 저자들은 이것이 semantic layout의 유사성을 활용하는 구조적 적응이라고 설명한다.

### Class-ratio prior

entropy minimization은 일부 쉬운 클래스에 치우칠 수 있다. 이를 막기 위해 source label 전체에서 클래스별 pixel count histogram을 정규화한 class prior $p_s$를 만든다. 그리고 target prediction에서 특정 클래스의 기대 확률이 이 prior보다 지나치게 낮아지면 벌점을 준다.

$$
L_{\mathrm{cp}}(x_t)
=
\sum_{c=1}^{C}
\max\left(0, \mu p_s^{(c)} - \mathbb{E}_c(P_{x_t}^{(c)})\right).
$$

여기서 $\mu \in [0,1]$는 prior를 얼마나 느슨하게 적용할지를 정하는 완화 계수다. 저자들은 single target image의 class ratio가 source 전체 분포와 완전히 같을 필요는 없다고 보고, 강제 동일화가 아니라 minimum presence constraint에 가깝게 사용한다.

### 네트워크와 학습 설정

기본 segmentation architecture는 DeepLab-v2이며, ASPP를 마지막 feature 위에 적용한다. sampling rate는 $\{6,12,18,24\}$를 사용한다. backbone은 VGG-16과 ResNet-101 두 가지를 실험했다. ResNet-101에서는 conv4와 conv5의 multi-level output adaptation도 사용했다. adversarial discriminator는 DCGAN 스타일의 fully convolutional 구조이며, 4개의 convolution layer와 leaky-ReLU를 사용한다.

학습은 PyTorch로 구현되었고, segmentation network는 SGD, discriminator는 Adam으로 최적화했다. learning rate는 segmentation network에 $2.5 \times 10^{-4}$, discriminator에 $10^{-4}$를 사용했다. $\lambda_{\mathrm{ent}}$와 $\lambda_{\mathrm{adv}}$는 모두 $0.001$로 고정했다. 논문에 따르면 이 값들은 모델이나 데이터셋에 크게 민감하지 않았지만, 너무 크게 설정하면 entropy가 너무 빠르게 떨어져 몇몇 클래스에 편향될 수 있다.

## 4. 실험 및 결과

실험은 두 개의 대표적인 synthetic-to-real segmentation UDA benchmark에서 수행되었다.

첫째는 GTA5 $\rightarrow$ Cityscapes이다. source는 24,966장의 GTA5 synthetic image이며, Cityscapes와 공통인 19개 클래스를 사용한다. 둘째는 SYNTHIA $\rightarrow$ Cityscapes이며, SYNTHIA-RAND-CITYSCAPES split의 9,400장을 사용하고 16개 공통 클래스로 학습한다. 평가 시에는 prior work와 맞추기 위해 16-class와 13-class subset을 모두 보고한다. 두 설정 모두 target 쪽 학습에는 2,975장의 unlabeled Cityscapes training image를 사용하고, 평가는 500장의 Cityscapes validation set에서 mIoU로 수행한다.

### GTA5 $\rightarrow$ Cityscapes 결과

VGG-16 기반 모델에서, direct entropy minimization인 MinEnt는 mIoU 32.8을 기록했다. 이는 Self-Training 28.1, Self-Training + CB 30.9보다 높고, Adapt-SegMap 35.0에는 다소 못 미친다. 반면 adversarial entropy minimization인 AdvEnt는 36.1을 기록하여 비교 모델들을 넘어섰다.

ResNet-101 기반에서는 결과가 더 강하다. MinEnt는 42.3, entropy range를 선택적으로 사용한 MinEnt + ER은 43.1, AdvEnt는 43.8을 기록했다. 여기에 MinEnt와 AdvEnt의 ensemble은 45.5까지 올라간다. 이는 표에 제시된 Adapt-SegMap 42.4나 재학습한 Adapt-SegMap* 42.2보다 높다. 따라서 GTA5 $\rightarrow$ Cityscapes에서는 AdvEnt가 단일 모델 기준 최고 성능을 달성했고, 두 모델 결합은 추가 이득을 보여준다.

논문은 oracle과의 gap도 비교한다. ResNet-101 기준 oracle은 65.1 mIoU인데, AdvEnt는 -21.3 gap, ensemble은 -19.6 gap이다. 이는 다른 UDA 방법들보다 oracle에 더 가까운 결과다.

정성적으로도 source-only 모델은 noisy prediction과 높은 entropy activation을 보였고, adaptation 후에는 더 깨끗한 segmentation map과 더 낮은 entropy map이 나타났다고 보고한다. 특히 AdvEnt가 MinEnt보다 전반적으로 더 낮은 prediction entropy를 만든다고 서술한다.

### SYNTHIA $\rightarrow$ Cityscapes 결과

이 설정은 layout과 viewpoint 차이가 GTA5보다 더 커서, entropy minimization이 더 쉽게 class bias에 빠질 수 있다고 저자들은 해석한다.

VGG-16 기반에서 MinEnt는 16-class mIoU 27.5, 13-class mIoU 32.5를 기록했다. Self-Training은 각각 23.9와 27.8이므로 이보다 낫지만, Self-Training + CB의 35.4와 36.1에는 미치지 못한다. 여기에 class prior를 넣은 MinEnt + CP는 30.4와 35.4로 개선되고, AdvEnt + CP는 31.4와 36.6까지 올라간다.

ResNet-101 기반에서는 MinEnt가 38.1/44.2, AdvEnt가 40.8/47.6, AdvEnt + MinEnt ensemble이 41.2/48.0을 기록한다. 이는 Adapt-SegMap*의 39.6/45.8보다 높다. 따라서 이 데이터셋에서도 단일 모델 기준 AdvEnt가 최고이며, ensemble이 추가 향상을 제공한다.

oracle gap 비교에서도 ResNet-101 기준 AdvEnt는 oracle 71.7에 대해 -24.1 gap, ensemble은 -23.7 gap으로 기존 방법보다 더 작다.

### Entropy range 선택과 class prior의 효과

GTA5 $\rightarrow$ Cityscapes, ResNet-101 설정에서는 target image 안에서 entropy가 가장 높은 상위 30% 픽셀에 대해서만 entropy loss를 적용했을 때 성능이 0.8 mIoU 향상되었다. 저자들은 high-entropy pixel이 가장 confusing한 위치이지만, ResNet-101처럼 일반화 성능이 좋은 모델에서는 그 안에도 맞지만 확신이 낮은 예측이 많이 포함되어 있어, 이 구간을 집중적으로 정제하는 것이 유리하다고 해석한다. 반면 VGG-16에서는 이런 가정이 잘 성립하지 않았다고 본다.

SYNTHIA $\rightarrow$ Cityscapes에서는 class prior가 중요했다. 저자들은 viewpoint와 layout 차이 때문에 특정 클래스가 완전히 사라지거나 과도하게 억눌리는 degeneracy가 발생할 수 있다고 본다. 그래서 $\mu = 0.5$로 완화된 class-ratio prior를 사용해 모든 클래스가 일정 수준 이상 존재하도록 유도했고, 실제로 mIoU 향상을 얻었다.

### Object detection으로의 확장

논문은 appendix 수준이지만 UDA for object detection에도 entropy 기반 아이디어를 적용했다. 실험은 Cityscapes $\rightarrow$ Cityscapes Foggy에서 SSD-300 with VGG-16을 사용했다. detection에서는 각 anchor box에 대한 class distribution의 entropy를 계산하고, segmentation과 유사한 방식으로 entropy loss와 adversarial loss를 적용한다.

baseline SSD-300은 mAP 14.7이었고, MinEnt는 16.9, AdvEnt는 26.2를 기록했다. 논문은 Domain Adaptive Faster R-CNN의 27.6 mAP가 약간 더 높지만, 자신들은 더 단순한 SSD-300과 더 낮은 해상도 $300 \times 300$을 사용했음을 강조한다. 특히 baseline 대비 improvement는 AdvEnt에서 +11.5로, 비교 논문 [3]의 +8.8보다 더 크다고 주장한다. 다만 이 부분은 본 논문의 중심 실험은 아니고 preliminary result로 제시되었다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 아이디어가 단순하면서도 효과적이라는 점이다. target prediction entropy를 줄인다는 관찰은 직관적이고, 실제로 synthetic-to-real segmentation benchmark 두 개에서 강한 성능으로 이어졌다. 특히 direct entropy minimization은 구현 오버헤드가 거의 없고, discriminator를 학습하지 않아도 일정 수준 이상의 경쟁력 있는 성능을 낸다. 논문도 MinEnt의 장점으로 가벼운 학습 비용과 adversarial training보다 더 안정적인 학습을 직접 언급한다.

두 번째 강점은 direct 방식과 adversarial 방식을 명확히 분리하면서도 서로 보완적인 관계를 보여준 점이다. MinEnt는 픽셀 단위의 확신을, AdvEnt는 weighted self-information 공간의 구조적 정렬을 담당한다. 실제로 두 모델 ensemble이 항상 더 나은 결과를 내는 것은 두 접근이 서로 다른 정보를 학습했다는 간접적 증거다.

세 번째 강점은 self-training과 entropy minimization의 연결을 이론적으로 해석해 준 점이다. 단순히 “잘 된다” 수준이 아니라, entropy loss가 pseudo-label cross-entropy의 soft version이라는 관점을 제시해 방법의 의미를 더 명확하게 만들었다.

한편 한계도 분명하다. 첫째, entropy minimization은 잘못된 확신을 더 강화할 위험이 있다. 논문도 source-only 모델이 low-entropy이지만 완전히 틀린 prediction을 만들 수 있다고 인정한다. 따라서 entropy를 줄이는 것만으로는 항상 안전하지 않으며, 이것이 class prior나 structured adversarial alignment 같은 추가 장치가 필요한 이유다.

둘째, direct entropy minimization은 구조적 의존성을 무시한다. 논문이 직접 지적하듯, 픽셀별 독립 entropy 합은 scene topology나 semantic co-occurrence를 반영하지 못한다. 그래서 더 좋은 성능은 대체로 AdvEnt에서 나온다. 이는 direct 방식만으로는 adaptation의 본질적 어려움을 모두 해결하지 못한다는 뜻이다.

셋째, class prior는 source class distribution을 target에 어느 정도 투영하는 가정 위에 서 있다. 저자들은 $\mu$를 둬서 완화했지만, source와 target의 class ratio가 본질적으로 다른 환경에서는 이 prior가 오히려 편향을 줄 가능성도 있다. 논문은 이를 완전히 분석하지는 않았다.

넷째, detection 확장 실험은 유망하지만 예비적이다. 더 강한 detector가 아닌 SSD-300 기반 단일 결과만 제시되며, detection setting에 대한 체계적 ablation은 본문에 충분히 포함되지 않는다. 따라서 “segmentation에서 통했던 아이디어가 detection에도 일반적으로 강하다”라고까지 결론내리기는 이르다.

마지막으로, 논문은 왜 entropy minimization이 특정 클래스나 특정 데이터셋에서 잘 작동하는지에 대한 이론적 분석을 깊게 제공하지는 않는다. 예를 들어 GTA5와 SYNTHIA에서 다른 거동이 나타나는 이유는 합리적으로 해석하지만, 이는 주로 경험적 설명에 머문다.

## 6. 결론

이 논문은 semantic segmentation UDA에서 target prediction entropy를 줄이는 전략이 매우 효과적일 수 있음을 보여준다. 저자들은 direct entropy loss인 MinEnt와, weighted self-information 공간에서의 adversarial alignment인 AdvEnt를 제안했고, 두 방법 모두 synthetic-to-real benchmark에서 강한 성능을 냈다. 특히 AdvEnt는 구조적 적응까지 포함하면서 더 일관되게 좋은 결과를 보여 주었고, 두 방법의 ensemble은 추가 성능 향상을 달성했다.

실용적 관점에서 이 연구의 의미는 크다. synthetic data로 학습한 segmentation 모델을 실제 장면으로 옮길 때, 복잡한 image translation이나 pseudo-label scheduling 없이도 uncertainty 자체를 제어하는 방식으로 domain gap을 줄일 수 있음을 보여주기 때문이다. 또한 detection에 대한 예비 결과는 entropy-based adaptation이 segmentation에만 국한되지 않을 가능성을 시사한다.

종합하면, ADVENT는 UDA for semantic segmentation에서 entropy를 단순한 분석 도구가 아니라 직접적인 학습 목표로 승격시킨 논문이다. 방법은 비교적 간결하지만, 구조적 adversarial adaptation과 결합했을 때 강력한 성능을 보였고, 이후 output-space adaptation과 uncertainty-aware domain adaptation 연구를 이해하는 데 중요한 기준점이 되는 작업으로 볼 수 있다.
