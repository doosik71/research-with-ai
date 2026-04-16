# Context-aware Feature Generation for Zero-shot Semantic Segmentation

- **저자**: Zhangxuan Gu, Siyuan Zhou, Li Niu, Zihan Zhao, Liqing Zhang
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2008.06893

## 1. 논문 개요

이 논문은 `zero-shot semantic segmentation` 문제를 다룬다. 이 문제에서는 학습 시점에 `seen category`에 대해서만 픽셀 단위 정답이 주어지고, 테스트 시점에는 `unseen category`까지 함께 등장할 수 있다. 즉, 모델은 한 번도 픽셀 주석을 본 적 없는 새로운 카테고리까지 분할해야 한다. 이를 위해 논문은 카테고리 이름의 의미를 담은 `semantic word embedding`을 seen과 unseen 사이의 연결 고리로 사용한다.

문제의 핵심 어려움은, 일반적인 segmentation 모델이 본질적으로 학습 데이터에 등장한 클래스에 강하게 편향된다는 점이다. 기존 zero-shot segmentation 방법 중 하나인 SPNet은 시각 특징을 semantic embedding 공간으로 사상하지만, 테스트 시 seen 클래스 쪽으로 예측이 치우치는 문제가 심하다. 또 다른 계열인 ZS3Net은 unseen 클래스의 feature를 생성해서 classifier를 보정하려고 하지만, 생성 다양성이 낮고 문맥 정보 활용이 제한적이라는 문제가 있다.

이 논문은 이런 한계를 해결하기 위해 `CaGNet`을 제안한다. 핵심 발상은 segmentation의 픽셀 특징이 단순히 클래스 의미만으로 결정되지 않고, 주변 문맥에 매우 크게 의존한다는 점이다. 예를 들어 같은 `cat` 클래스 픽셀이라도 몸통 내부인지, 윤곽 근처인지, 주변에 쿠션이 있는지, 식물이 있는지에 따라 deep feature가 달라질 수 있다. 따라서 저자들은 semantic embedding만으로 feature를 만들지 않고, 픽셀별 contextual information을 함께 넣어 `context-aware`하고 더 다양한 feature를 생성하도록 설계했다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. unseen category를 분할하기 위해서는 classifier가 unseen 클래스 feature에도 익숙해져야 하는데, 이때 생성되는 feature가 단순히 “클래스 이름에 대응하는 평균적인 feature” 수준에 머물면 부족하다. 같은 클래스라도 문맥에 따라 feature가 달라져야 실제 segmentation에 도움이 된다. CaGNet은 바로 이 점을 반영해, 각 픽셀의 주변 문맥을 latent code로 인코딩하고, 이 latent code와 semantic word embedding을 함께 generator에 넣는다.

기존 ZS3Net과의 가장 중요한 차이는 두 가지다. 첫째, ZS3Net은 semantic embedding에 random noise를 붙여 다양한 feature를 만들려 하지만, 논문은 이런 방식이 `mode collapse`에 취약하다고 지적한다. generator가 noise를 무시해버리면 한 클래스당 feature 다양성이 거의 생기지 않는다. 반면 CaGNet은 실제 이미지에서 추출한 픽셀 문맥을 latent code로 사용하고, 이 latent code와 출력 feature 사이에 사실상 일대일 대응을 유도함으로써 더 다양한 feature 생성을 유도한다.

둘째, ZS3Net의 확장판인 ZS3Net (GC)은 object-level graph로 문맥을 반영하지만, 논문은 이것이 공간적 object arrangement 정도만 반영하는 제한된 문맥이라고 본다. CaGNet은 object-level이 아니라 `pixel-wise contextual information`을 사용한다. 즉, 객체 내부 위치, 자세, 인접 배경, 멀리 있는 배경 단서 등 더 세밀한 문맥을 다룬다. 또한 unseen 클래스의 explicit graph를 만들 필요 없이 latent code를 `N(0,1)`에서 샘플링할 수 있게 만들어, 실제 unseen 문맥 구조를 미리 알지 못해도 feature 생성이 가능하게 했다.

또 하나의 중요한 설계는 segmentation network와 feature generation network를 분리하지 않고 연결했다는 점이다. Contextual Module과 classifier를 공유하여, 실제 이미지에서 나온 feature와 생성된 fake feature가 같은 분류기 위에서 학습되도록 만들었다. 이 구조는 zero-shot segmentation이라는 목적에 더 직접적이다. classifier는 seen real feature뿐 아니라 seen/unseen synthetic feature도 함께 보면서 학습되므로 unseen 클래스에 대한 대응력이 좋아진다.

## 3. 상세 방법 설명

CaGNet은 크게 다섯 구성요소로 설명된다. segmentation backbone $E$, Contextual Module $CM$, feature generator $G$, discriminator $D$, classifier $C$이다. 기본 segmentation network는 backbone과 classifier로 이루어진다. 입력 이미지가 들어오면 backbone이 feature map을 만들고, classifier가 픽셀별 클래스를 예측한다. 여기에 저자들은 backbone 뒤에 $CM$을 삽입하고, 또 $CM$이 만든 contextual latent code를 이용해 generator가 feature를 생성하도록 한다.

### 전체 구조

입력 이미지 $I_n$에 대해 backbone $E$는 feature map $F_n \in \mathbb{R}^{h \times w \times l}$을 출력한다. 그 다음 $CM$은 이 $F_n$으로부터 픽셀별 문맥 정보를 추출하고, 두 가지를 만든다.

하나는 contextual latent code map $Z_n \in \mathbb{R}^{h \times w \times l}$이고, 다른 하나는 segmentation에 실제로 쓰일 향상된 feature map $X_n$이다. 이때 $X_n$은 단순한 $F_n$이 아니라, latent code를 residual attention처럼 반영한 결과이다. 논문은 이를 다음처럼 정의한다.

$$
X_n = F_n + F_n \odot \phi(Z_n)
$$

여기서 $\phi$는 sigmoid activation이고, $\odot$는 element-wise Hadamard product이다. 즉, $Z_n$은 generator 입력일 뿐 아니라 segmentation backbone feature를 보정하는 attention 역할도 한다. 다만 논문은 이 residual attention 자체의 성능 향상은 제한적이며, $CM$의 주된 목적은 feature generation을 돕는 것이라고 명시한다.

### Contextual Module

$CM$의 첫 번째 목적은 픽셀별 문맥을 모으는 것이다. 저자들은 pooling을 쓰지 않는다. 이유는 각 픽셀과 그 문맥 코드 사이의 대응을 유지해야 하기 때문이다. 대신 dilated convolution을 연속으로 사용해 receptive field를 넓히면서 spatial resolution은 유지한다.

구체적으로 $F_n$에 대해 세 개의 serial dilated convolution을 적용하여 세 개의 context map $\hat{F}_n^0, \hat{F}_n^1, \hat{F}_n^2 \in \mathbb{R}^{h \times w \times l}$를 만든다. 더 깊은 context map일수록 더 넓은 receptive field를 가지므로, 작은 범위 문맥부터 큰 범위 문맥까지 함께 표현할 수 있다.

그 다음 `context selector`가 등장한다. 논문은 픽셀마다 필요한 문맥 스케일이 다르다고 본다. 어떤 픽셀은 작은 범위 문맥만으로 충분하지만, 어떤 픽셀은 물체 자세나 멀리 있는 배경까지 필요할 수 있다. 이를 위해 세 context map을 이어붙인 후, $3 \times 3$ conv를 통해 3채널 scale weight map $A_n = [A_n^0, A_n^1, A_n^2] \in \mathbb{R}^{h \times w \times 3}$를 예측한다. 각 채널은 해당 스케일의 중요도를 픽셀별로 나타낸다.

이 가중치를 각 feature 채널 수만큼 복제해 $\hat{A}_n^k \in \mathbb{R}^{h \times w \times l}$를 만든 뒤, 세 context map을 픽셀별로 가중합 비슷한 방식으로 결합한다. 실제 구현은 concat 후 channel-wise weighting이다.

$$
[\hat{F}_n^0 \odot \hat{A}_n^0,\; \hat{F}_n^1 \odot \hat{A}_n^1,\; \hat{F}_n^2 \odot \hat{A}_n^2]
$$

이렇게 하면 픽셀마다 적절한 스케일의 문맥이 선택된다. 논문은 이것이 단순 attention과 닮아 보일 수 있으나, 동기와 세부 구현은 다르다고 설명한다.

### Contextual Latent Code

가중된 multi-scale context를 $1 \times 1$ conv에 넣어 각 픽셀마다 Gaussian distribution의 파라미터 $\mu_n^Z, \sigma_n^Z$를 얻는다. 각 픽셀 $i$에 대한 latent code $z_{n,i}$는 다음과 같이 샘플링된다.

$$
z_{n,i} = \mu_{n,i}^Z + \epsilon \sigma_{n,i}^Z
$$

여기서 $\epsilon \sim \mathcal{N}(0,1)$이다. 논문의 표기에는 $\epsilon$을 scalar로 적었지만, 일반적인 reparameterization 관점에서는 픽셀 표현 차원에 맞는 noise로 이해하는 편이 자연스럽다. 다만 이 부분을 논문이 엄밀하게 더 설명하지는 않는다.

이 latent code가 실제로 샘플 가능한 잠재공간이 되도록 하기 위해 KL divergence를 사용한다.

$$
L_{KL} = D_{KL}\big[\mathcal{N}(\mu_{n,i}^Z, \sigma_{n,i}^Z)\,\|\,\mathcal{N}(0,1)\big]
$$

즉, 학습 시에는 실제 이미지 문맥에서 유도된 latent code를 쓰고, 추론 또는 unseen feature generation 시에는 $N(0,1)$에서 샘플링해 쓸 수 있도록 만든다.

### Context-aware Feature Generator

seen category에 대한 학습 단계에서는, 각 픽셀의 정답 클래스 $c_{n,i}^s$에 대응하는 semantic word embedding $\bar{w}_{c_{n,i}^s}$를 가져와 픽셀별 embedding map $W_n^s$를 만든다. 이제 generator는 각 픽셀에 대해 contextual latent code와 semantic embedding을 함께 받아 fake feature를 생성한다.

$$
\tilde{x}_{n,i}^s = G(z_{n,i}, w_{n,i}^s)
$$

저자들의 핵심 가정은, semantic embedding $w_{n,i}^s$가 클래스 의미를 제공하고, $z_{n,i}$가 해당 픽셀의 문맥을 제공하면, generator가 실제 픽셀 feature $x_{n,i}^s$를 복원할 수 있어야 한다는 것이다. 그래서 reconstruction loss를 둔다.

$$
L_{REC} = \sum_{n,i} \|x_{n,i}^s - \tilde{x}_{n,i}^s\|_2^2
$$

이 손실은 논문의 매우 중요한 구성요소다. 실제 ablation에서도 제거 시 unseen 성능이 크게 하락한다.

### Classification Loss와 Adversarial Loss

classifier $C$는 segmentation용 분류를 담당한다. real feature에 대해 cross-entropy loss를 사용한다.

$$
L_{CLS} = - \sum_{n,i} y_{n,i}^s \log(C(x_{n,i}^s))
$$

여기서 $y_{n,i}^s$는 one-hot label이다.

또한 discriminator $D$는 real feature와 fake feature를 구분하도록 학습된다. 논문은 Least Squares GAN 스타일의 adversarial loss를 사용한다.

$$
L_{ADV} = \sum_{n,i} (D(x_{n,i}^s))^2 + \big(1 - D(G(z_{n,i}, w_{n,i}^s))\big)^2
$$

논문 서술상 target이 real=1, fake=0이라고 설명되어 있으므로 식의 부호나 항 배치는 일반적인 표기와 조금 혼동될 수 있다. 그러나 저자 의도는 real/fake feature 분포를 맞추는 adversarial regularization이다. 여기서는 논문에 적힌 식을 그대로 따르는 것이 안전하다.

### 학습 절차

최적화는 두 단계로 나뉜다.

첫 번째는 `training step`이다. 이 단계에서는 실제 이미지와 seen mask만 사용한다. 모든 모듈 $E, CM, G, D, C$를 함께 학습한다. 전체 목적함수는 다음과 같다.

$$
\min_{G,E,C,CM}\max_D \; L_{CLS} + L_{ADV} + \lambda_1 L_{REC} + \lambda_2 L_{KL}
$$

즉, discriminator는 구분 능력을 높이도록 maximize 방향으로, 나머지 모듈은 segmentation과 generation 품질을 함께 높이도록 minimize 방향으로 학습된다.

두 번째는 `finetuning step`이다. 여기서는 seen과 unseen 카테고리를 모두 포함하도록 semantic embedding map을 인위적으로 구성한다. 픽셀마다 seen 또는 unseen 클래스의 word embedding을 랜덤하게 배치해 $\tilde{W}_m^{s \cup u}$를 만든 후, latent code는 $\mathcal{N}(0,1)$에서 샘플링하여 fake feature map $\tilde{X}_m^{s \cup u}$를 생성한다.

이 단계에서는 real visual feature가 없으므로 backbone $E$와 $CM$은 고정한다. 오직 $G, D, C$만 업데이트한다. 목적함수는 다음과 같다.

$$
\min_{G,C}\max_D \; \tilde{L}_{CLS} + \tilde{L}_{ADV}
$$

즉, synthetic feature를 이용해 classifier가 unseen category에도 적응하도록 한다. 논문은 처음에 충분히 training step을 수행한 뒤, 이후에는 100 iteration마다 training과 finetuning을 번갈아 수행한다고 설명한다.

### 구현 세부

generator $G$는 hidden neuron 512개의 MLP이고, 각 층에 Leaky ReLU와 dropout을 사용한다. classifier $C$와 discriminator $D$는 두 개의 $1 \times 1$ conv 층으로 구성되며, 첫 번째 conv layer의 가중치를 공유한다. backbone은 ImageNet으로 pretrain된 ResNet-101 기반 Deeplabv2이다. semantic embedding은 SPNet 설정을 따라 word2vec 300차원과 fastText 300차원을 이어붙여 총 $d=600$ 차원으로 사용한다.

## 4. 실험 및 결과

### 데이터셋과 설정

논문은 세 가지 benchmark에서 평가한다. Pascal-Context, COCO-stuff, Pascal-VOC 2012이다. 각각 33개, 182개, 20개 카테고리 설정을 사용한다. seen/unseen split은 SPNet 설정을 따른다. COCO-stuff에서는 15개 unseen, Pascal-VOC에서는 5개 unseen, Pascal-Context에서는 4개 unseen을 지정했다. 학습 시 unseen 픽셀의 annotation은 무시된다.

평가 지표는 pixel accuracy, mean accuracy, mIoU, 그리고 `hIoU`이다. 특히 저자들은 overall 성능을 볼 때 단순 mIoU보다 hIoU를 더 중요하게 본다. 이유는 seen 클래스 성능이 훨씬 높아서 평균 mIoU가 unseen 성능 문제를 가릴 수 있기 때문이다. zero-shot segmentation에서는 seen/unseen 균형이 중요하므로 이 주장은 타당하다.

### 정량 결과

가장 중요한 결과는 Table 1이다. CaGNet은 세 데이터셋 모두에서 unseen 및 overall 성능에서 SPNet, ZS3Net보다 일관되게 우수하다.

Pascal-VOC에서 self-training 없이,
- SPNet의 hIoU는 0.0002,
- ZS3Net은 0.2874,
- CaGNet은 0.3972이다.

특히 unseen mIoU는
- SPNet 0.0001,
- ZS3Net 0.1765,
- CaGNet 0.2659로 상승한다.

Pascal-Context에서도 CaGNet은 unseen mIoU 0.1442, hIoU 0.2061로 ZS3Net의 unseen mIoU 0.0768, hIoU 0.1246보다 크게 낫다. COCO-stuff에서도 unseen mIoU가 0.1223으로 ZS3Net의 0.0953보다 높다.

이 결과는 논문의 핵심 주장을 강하게 뒷받침한다. seen 성능을 거의 유지하거나 약간 희생하면서 unseen 성능을 훨씬 크게 끌어올린다. 저자들도 SPNet이 일부 seen 성능에서 더 강한 이유를 “거의 모든 픽셀을 seen category로 밀어버리는 경향” 때문이라고 해석한다. zero-shot 문제에서는 이런 편향은 바람직하지 않다.

self-training을 붙였을 때도 개선이 유지된다. 예를 들어 Pascal-VOC에서 `CaGNet+ST`는 hIoU 0.4366, unseen mIoU 0.3031로 전체 최고 성능을 얻는다.

### ZS3Net 설정에서의 비교

보충자료에서는 ZS3Net이 사용한 다른 설정도 따라간다. Deeplabv3+ backbone, 300차원 word2vec, background를 seen으로 포함하는 설정이다. 여기서도 Pascal-VOC에서 hIoU가 37.9에서 50.9로, Pascal-Context에서 16.3에서 21.2로 상승한다. 따라서 성능 향상이 특정 설정에만 의존하는 것은 아니라는 점을 추가로 보여준다.

### Ablation Study

Table 2는 모듈별 효과를 보여준다. backbone과 classifier만 있으면 unseen mIoU가 0이다. generator와 discriminator를 넣으면 unseen mIoU가 0.1798로 올라간다. 여기에 $CM$까지 넣으면 0.2659가 된다. 즉, feature generation 자체도 중요하지만, 문맥을 넣는 것이 추가적인 핵심 개선 요인이다.

Table 3은 $CM$ 구조를 세밀하게 검증한다. 단순 $1 \times 1$ conv만 사용하면 hIoU가 0.3211이다. 일반 conv로 문맥을 넣으면 0.3298, dilated conv를 쓰면 0.3654, multi-scale 결합을 하면 0.3825, 최종적으로 context selector를 포함하면 0.3972가 된다. 즉,
1. contextual information 자체가 중요하고,
2. dilated conv 기반 multi-scale context가 더 좋으며,
3. 픽셀별 adaptive scale weighting이 추가 이득을 준다.

Table 4는 손실 함수의 역할을 보여준다. $L_{REC}$를 빼면 unseen mIoU가 0.1263까지 크게 떨어진다. $L_{ADV}$를 빼도 0.1979로 하락한다. $L_{KL}$ 제거 역시 0.2487로 감소한다. reconstruction loss가 특히 중요하다는 점이 두드러진다. 이는 논문의 직관, 즉 “semantic embedding + contextual code로 real feature를 재구성해야 한다”는 설계 철학과 잘 맞는다.

Table 5는 finetuning 시 seen:unseen feature 생성 비율 $r$를 다룬다. 자연 비율 $|C_s|:|C_u|$를 쓰면 unseen 성능이 낮고, $1:1$이 가장 좋다. unseen을 너무 많이 생성하는 `0:1`도 seen 성능을 많이 깎는다. 따라서 균형 있는 synthetic feature 공급이 중요하다는 것을 보여준다.

### 정성 결과

Figure 4와 보충자료의 시각화에서는 unseen object인 `train`, `tv monitor`, `potted plant`, `sheep`, `sofa` 등을 CaGNet이 더 잘 분할한다. 이는 단순 수치 향상뿐 아니라 실제 예측 품질 개선을 보여준다.

Figure 5는 문맥 없는 random latent code와, $CM$이 만든 contextual latent code를 비교한다. ground-truth semantic embedding과 latent code를 generator에 넣어 생성한 feature와 실제 feature의 reconstruction loss를 시각화했을 때, $CM$을 사용할수록 더 어두운 영역이 많아진다. 논문 해석대로라면, 문맥 정보가 feature generation 품질을 실제로 높인다는 직접적인 증거다.

Figure 6의 context selector 시각화도 흥미롭다. 판별적인 로컬 부위는 small-scale context를, 나머지 영역은 medium 또는 large-scale context를 더 선호한다. 또 작은 물체는 작은 scale, 큰 물체는 큰 scale을 선호하는 경향이 나타난다. 이는 설계 동기를 잘 뒷받침하는 qualitative evidence다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 zero-shot segmentation에서 왜 context가 필요한지 매우 설득력 있게 문제를 정식화했다는 점이다. 저자들은 단순히 “context가 중요하다”라고 주장하는 데서 그치지 않고, 같은 클래스 `cat` 안에서도 feature가 문맥에 따라 여러 cluster로 나뉜다는 시각화로 출발한다. 이 문제 정의가 method와 ablation, visualization까지 일관되게 이어진다.

두 번째 강점은 설계가 비교적 간결하면서도 효과적이라는 점이다. 기존 segmentation backbone에 $CM$을 삽입하고, generator 입력을 `semantic embedding + contextual latent code`로 바꾸는 방식은 구조적으로 과도하게 복잡하지 않다. 동시에 segmentation network와 generator를 classifier 차원에서 연결해 학습하는 설계도 합리적이다. 실험적으로도 세 데이터셋에서 일관된 개선을 보인다.

세 번째 강점은 ablation이 충분히 충실하다는 점이다. 모듈별 효과, $CM$의 변형, 각 손실 항의 역할, finetuning 비율, supplementary 설정까지 비교해 두었기 때문에, 논문의 핵심 주장인 “context-aware generation이 중요하다”는 점은 꽤 잘 검증된다.

반면 한계도 분명하다. 첫째, zero-shot segmentation 성능이 여전히 절대적으로 매우 높다고 보기는 어렵다. 예를 들어 Pascal-Context나 COCO-stuff에서 unseen mIoU는 개선되었지만 여전히 낮다. 즉, 방법의 상대적 우수성은 분명하지만 문제 자체는 여전히 매우 어렵다.

둘째, 생성된 feature가 실제로 unseen 클래스의 진짜 분포를 얼마나 잘 근사하는지는 완전히 명확하지 않다. 논문은 seen 클래스의 reconstruction과 adversarial regularization으로 이를 뒷받침하지만, unseen은 본질적으로 real feature가 없기 때문에 간접적 검증에 의존한다. 따라서 unseen feature generation의 품질은 여전히 강한 가정에 기대고 있다.

셋째, semantic embedding의 품질에 영향을 많이 받을 가능성이 있다. supplementary에서는 `train`이라는 단어가 여러 의미를 가져 embedding이 불정확할 수 있다고 직접 언급한다. 즉, 이 방법은 context를 잘 쓰더라도 category semantic representation 자체가 부정확하면 한계가 생긴다. 이런 문제는 zero-shot 계열 전반의 공통 약점이지만, 본 논문도 예외는 아니다.

넷째, adversarial loss 식과 latent sampling 표기는 다소 엄밀성이 부족한 부분이 있다. 예를 들어 $\epsilon$을 scalar로 적은 부분이나 LSGAN 식의 방향성 표기는 구현 수준에서 더 자세한 설명이 있었으면 좋았을 것이다. 다만 논문의 전체 아이디어를 손상시킬 정도의 문제는 아니다.

다섯째, 비교 실험에서 ZS3Net (GC)은 제외된다. 저자들은 graph context가 현재 설정에서 사용 불가능하고 실제 적용에서도 얻기 어렵다고 설명하지만, 결과적으로는 가장 강한 기존 context baseline과의 직접 비교가 빠져 있다. 이 부분은 독자가 약간 아쉽게 느낄 수 있다.

## 6. 결론

이 논문은 zero-shot semantic segmentation에서 unseen category를 더 잘 다루기 위해, semantic word embedding만이 아니라 `pixel-wise contextual information`까지 feature generation에 활용해야 한다고 주장하고, 이를 구현한 `CaGNet`을 제안한다. 핵심 기여는 세 가지로 정리할 수 있다. 첫째, 문맥을 latent code로 인코딩해 context-aware feature generation을 수행했다. 둘째, segmentation network와 feature generation network를 하나의 구조로 연결했다. 셋째, multi-scale context와 pixel-wise scale weighting을 수행하는 context selector를 설계했다.

실험 결과를 보면, CaGNet은 여러 benchmark에서 unseen 및 overall 성능, 특히 hIoU와 unseen mIoU를 꾸준히 개선한다. 따라서 이 연구는 zero-shot segmentation에서 단순 semantic transfer를 넘어서, 문맥 기반 feature synthesis가 중요하다는 점을 분명히 보여준다. 실제 응용 측면에서는 새로운 클래스에 대한 dense annotation 없이 segmentation 성능을 확장하려는 상황에서 의미가 있다. 향후 연구로는 더 강한 semantic representation, 더 정교한 latent modeling, 그리고 open-vocabulary segmentation과의 연결이 자연스러운 확장 방향으로 보인다.
