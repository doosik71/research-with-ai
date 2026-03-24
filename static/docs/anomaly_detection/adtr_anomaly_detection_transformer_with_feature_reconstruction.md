# ADTR: Anomaly Detection Transformer with Feature Reconstruction

- **저자**: Zhiyuan You, Kai Yang, Wenhan Luo, Lei Cui, Yu Zheng, Xinyi Le
- **발표연도**: 2022
- **arXiv**: <https://arxiv.org/abs/2209.01816v3>

## 1. 논문 개요

이 논문은 **정상 샘플만으로 학습하는 anomaly detection** 문제를 다룬다. 특히 산업 검사나 one-class classification 환경에서는 불량(anomaly) 샘플이 매우 적거나 아예 없는 경우가 많기 때문에, 정상 데이터의 분포만 학습한 뒤 테스트 시 정상과 다른 패턴을 찾아내는 방식이 중요하다. 저자들은 기존의 CNN 기반 reconstruction 방식이 가진 두 가지 근본적 한계를 지적한다. 첫째, 복원 대상이 raw pixel이기 때문에 의미론적 정보가 약하다. 둘째, CNN은 anomaly까지도 너무 잘 복원하는 경향이 있어 정상과 이상을 구분하기 어렵다.

이를 해결하기 위해 제안된 방법이 **ADTR (Anomaly Detection TRansformer)** 이다. 핵심은 픽셀을 복원하는 대신, ImageNet으로 사전학습된 CNN backbone이 추출한 **semantic feature**를 복원 대상으로 삼고, 그 복원을 **transformer**로 수행하는 것이다. 저자들의 주장은 분명하다. 사전학습 feature는 정상과 이상을 더 잘 구분할 수 있고, transformer의 query embedding 기반 attention 구조는 CNN보다 “입력을 그대로 복사하는 identical mapping”을 덜 학습하므로 anomaly reconstruction을 제한할 수 있다는 것이다.

이 문제는 실제적으로도 중요하다. 산업 현장에서는 초기에는 정상 샘플만 있고, 시간이 지나며 anomaly 샘플이 조금씩 생길 수 있다. 따라서 방법이 처음에는 normal-only setting에서 동작하고, 이후에는 일부 anomaly label이 있는 상황까지 자연스럽게 확장될 수 있어야 한다. 이 논문은 바로 그 점을 겨냥해, normal-only case뿐 아니라 image-level 또는 pixel-level anomaly label이 일부 있는 경우까지 포괄하는 unified framework를 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 크게 세 가지로 요약할 수 있다.

첫째, **reconstruction target을 pixel이 아니라 pre-trained feature로 바꾼다**. 기존 autoencoder류 방법은 입력 이미지 자체를 복원하려고 한다. 하지만 서로 다른 semantic class라도 pixel 수준에서는 비슷하게 보일 수 있다. 예를 들어 texture anomaly나 구조적 불일치는 raw pixel reconstruction만으로는 잘 드러나지 않을 수 있다. 반면, 대규모 데이터셋으로 사전학습된 backbone의 feature는 더 풍부한 semantic distinction을 담고 있으므로 정상과 이상을 더 잘 분리한다.

둘째, **reconstruction model로 CNN 대신 transformer를 쓴다**. 논문은 단순히 “transformer가 좋다”라고 주장하지 않는다. 왜 transformer가 anomaly detection에 유리한지에 대한 기제를 설명하려고 한다. 저자들은 CNN이 identity-like shortcut을 학습하기 쉬운 반면, transformer decoder의 **learnable query embedding**은 정상 feature와 정렬된 attention map을 만들어야 하므로, 정상 데이터로 학습된 query가 이상 feature에는 잘 맞지 않아 anomaly reconstruction이 실패하기 쉽다고 본다. 즉, normal과 anomaly 사이의 reconstruction gap을 더 크게 만들 수 있다는 논리다.

셋째, **normal-only와 anomaly-available setting을 하나의 프레임워크로 확장**한다. 실제 현장에서는 시간이 지날수록 anomaly sample이 조금씩 확보될 수 있다. 이때 ADTR를 그대로 확장해, pixel-level label이 있을 때는 anomaly 위치별로 “정상은 가깝게, 이상은 멀게” 만드는 push-pull loss를 사용하고, image-level label만 있을 때는 anomaly score가 큰 상위 $k$개 위치를 뽑아 이미지 단위 loss를 구성한다. 즉, 단순한 완전 비지도 모델이 아니라, weak supervision이나 limited supervision까지 자연스럽게 수용하는 구조가 이 논문의 차별점이다.

기존 접근과의 차별점은 명확하다. reconstruction 기반 방법과 비교하면 복원 대상 자체가 다르고, projection 기반 방법과 비교하면 feature difference map을 통해 localization까지 직접 수행한다. 또한 transformer를 anomaly detection에 사용한 기존 방법들이 주로 raw image reconstruction에 머물렀던 것과 달리, 이 논문은 **pre-trained feature reconstruction + query embedding의 역할 분석**까지 제시한다는 점에서 차별성이 있다.

## 3. 상세 방법 설명

### 전체 파이프라인

ADTR의 전체 구조는 세 단계로 이해할 수 있다.

먼저 **Embedding 단계**에서 frozen pre-trained CNN backbone이 입력 이미지로부터 multi-scale feature를 추출한다. 논문에서는 ImageNet pre-trained EfficientNet-B4를 사용한다. layer1부터 layer5까지의 feature를 같은 spatial size로 resize한 뒤 channel 방향으로 concatenate하여 하나의 multi-scale feature map을 만든다. 이를 다음과 같이 쓸 수 있다.

$$
\mathbf{f} \in \mathbb{R}^{C \times H \times W}
$$

이 feature map은 서로 다른 receptive field를 가진 여러 계층의 정보를 함께 담고 있으므로, 작은 결함과 큰 구조적 이상 모두에 민감해질 수 있다.

다음으로 **Reconstruction 단계**에서 이 feature map을 transformer로 복원한다. 우선 $\mathbf{f}$를 $H \times W$개의 feature token으로 나눈다. 계산량을 줄이기 위해 transformer에 넣기 전 $1 \times 1$ convolution으로 channel 수를 줄이고, transformer 출력 후 다시 $1 \times 1$ convolution으로 원래 channel 수로 복구한다. encoder는 표준 transformer encoder 구조이고, decoder는 **auxiliary learnable query embedding**을 입력으로 사용해 encoder output을 참조하며 reconstruction을 수행한다.

마지막으로 **Comparison 단계**에서 backbone이 뽑은 원본 feature와 transformer가 복원한 feature의 차이를 anomaly score로 사용한다. 중요한 점은 anomaly localization과 image-level anomaly detection이 모두 이 feature discrepancy 위에서 정의된다는 것이다.

### Embedding: multi-scale pre-trained features

저자들은 feature extractor로 EfficientNet-B4를 고정한 채 사용한다. Appendix에 따르면 layer1~layer5의 channel 수는 각각 24, 32, 56, 160, 448이고, 이를 합치면 총 720-channel feature map이 된다. MVTec-AD에서는 최종적으로 $32 \times 32 \times 720$, CIFAR-10에서는 $8 \times 8 \times 720$ feature map을 만든다. 본문에서는 실험 설명에서 feature map size를 $16 \times 16$으로 언급한 부분도 있으나, Appendix에서는 $32 \times 32$로 더 자세히 적혀 있다. 이 부분은 문서 내 서술이 완전히 일치하지 않으므로, 세부 설정은 Appendix 쪽이 더 구체적이라고 보는 것이 타당하다.

이 multi-scale 설계의 의미는 단순하다. 낮은 층의 feature는 local texture나 edge에 민감하고, 높은 층의 feature는 더 넓은 semantic context를 담는다. anomaly는 작은 scratch일 수도 있고, 물체의 방향이 뒤집힌 structural anomaly일 수도 있기 때문에, 여러 scale의 정보를 함께 쓰는 것이 합리적이다.

### Reconstruction: transformer로 feature token 복원

feature map $\mathbf{f}$를 token으로 나눈 뒤, 먼저 $1 \times 1$ convolution으로 720차원을 256차원으로 축소한다. 이후 token sequence를 transformer encoder에 통과시킨다. encoder는 standard multi-head self-attention, FFN, residual connection, normalization으로 이루어진다. Appendix 기준 encoder와 decoder 모두 4층, attention head는 8개다. FFN은 256 → 1024 → 256 구조를 사용한다.

decoder의 핵심은 **learnable query embedding**이다. 이 query embedding은 입력 token과 같은 크기의 learnable parameter이며, decoder는 이를 self-attention과 encoder-decoder attention을 통해 변환하여 reconstruction token을 생성한다. transformer는 permutation-invariant하기 때문에 learned positional embedding도 사용한다. 특히 Appendix에서는 positional embedding을 첫 layer 한 번만이 아니라 각 self-attention layer마다 더한다고 설명한다.

결국 transformer 출력은 다시 $1 \times 1$ convolution을 거쳐 원래 720차원으로 복구되고, reshape하여 reconstructed feature map $\hat{\mathbf{f}} \in \mathbb{R}^{C \times H \times W}$를 얻는다.

### Normal-only 학습 목표

정상 샘플만 있는 경우, 학습은 backbone feature와 reconstructed feature 사이의 평균 제곱 오차(MSE)로 진행된다.

$$
\mathcal{L}_{norm} = \frac{1}{H \times W} \lVert \mathbf{f} - \hat{\mathbf{f}} \rVert_2^2
$$

의미는 매우 직관적이다. 정상 데이터에 대해서는 transformer가 backbone feature를 잘 복원하도록 만든다. 테스트 시에는 정상 샘플에서는 reconstruction error가 작고, 이상 샘플에서는 reconstruction error가 커지기를 기대한다.

### 추론과 anomaly score

feature difference map은 다음처럼 정의된다.

$$
\mathbf{d}(i,u) = \mathbf{f}(i,u) - \hat{\mathbf{f}}(i,u)
$$

여기서 $i$는 channel index이고, $u$는 spatial position이다. 각 위치 $u$에 대한 anomaly score는 channel 방향 차이 벡터의 $L_2$ norm으로 계산한다.

$$
s(u) = \lVert \mathbf{d}(:,u) \rVert_2
$$

즉, 한 위치에서 backbone feature와 reconstructed feature가 많이 다르면 그 위치가 이상일 가능성이 높다고 본다. 이 $s(u)$를 anomaly score map으로 사용하면 localization이 가능하다.

이미지 단위 anomaly detection에서는 이 score map을 average pooling한 뒤, 그 최대값을 이미지 전체의 anomaly score로 사용한다고 설명한다. 정확한 pooling 방식의 세부는 본문에 간략히만 제시되며, 구체적 구현 디테일은 충분히 상세하게 쓰여 있지는 않다.

### 왜 transformer가 identical mapping을 막는가

논문의 가장 중요한 주장 중 하나는 transformer, 특히 **query embedding이 anomaly reconstruction을 덜 잘하게 만든다**는 것이다.

저자들은 먼저 CNN의 경우를 선형층으로 단순화해 설명한다. normal feature를 $\mathbf{x}^+ \in \mathbb{R}^{K \times C}$라고 하자. fully connected layer 기반 복원은 다음과 같이 쓸 수 있다.

$$
\hat{\mathbf{x}} = \mathbf{x}^+ \mathbf{w} + \mathbf{b}
$$

MSE loss로 $\hat{\mathbf{x}}$를 $\mathbf{x}^+$에 맞추다 보면, 모델은 쉽게 $\mathbf{w} \to \mathbf{I}$, $\mathbf{b} \to \mathbf{0}$인 shortcut, 즉 거의 identity mapping을 학습할 수 있다. 이렇게 되면 anomaly feature $\mathbf{x}^-$가 들어와도 비슷하게 잘 통과시켜 reconstruction error가 작아질 수 있다.

반면 transformer attention with learnable query는 다음과 같이 이상화해 표현된다.

$$
\hat{\mathbf{x}} = \text{softmax}\left(\mathbf{q}(\mathbf{x}^+)^T / C\right)\mathbf{x}^+
$$

여기서 $\mathbf{q}$는 learnable query embedding이다. 정상 feature를 잘 재구성하려면 attention map이 identity matrix와 비슷해져야 하는데, 이를 위해 query는 정상 샘플의 구조와 잘 맞아야 한다. 따라서 정상 샘플에 맞춰 학습된 query는 anomaly feature에는 잘 대응하지 못해 reconstruction이 덜 잘 될 가능성이 높다. 이 설명은 엄밀한 증명이라기보다는 **직관적 메커니즘 분석**에 가깝다. 그러나 ablation 결과에서 attention과 query embedding을 제거하면 성능이 CNN 수준으로 떨어진다는 점을 근거로, 저자들은 이 해석을 지지한다.

### Anomaly-available case로의 확장: ADTR+

논문은 일부 anomaly label이 있는 경우로도 확장된다.

#### Pixel-level label이 있는 경우

우선 feature difference map으로부터 pseudo-Huber 형태의 위치별 차이 값을 정의한다.

$$
\phi(u) = \left( \left( \frac{1}{C}\sum_i^C |\mathbf{d}(i,u)| \right)^2 + 1 \right)^{1/2} - 1
$$

이 $\phi(u)$는 anomaly difference map처럼 동작한다. 이를 이용해 pixel-level push-pull loss를 정의한다.

$$
\mathcal{L}_{px} = \frac{1}{HW}\sum_u (1-y(u))\phi(u) \alpha \log\left(1-\exp\left(-\frac{1}{HW}\sum_u y(u)\phi(u)\right)\right)
$$

여기서 $y(u)=0$이면 정상, $y(u)=1$이면 anomaly이다. 첫 번째 항은 정상 위치의 reconstruction difference를 줄이는 역할을 한다. 두 번째 항은 anomaly 위치에서 difference가 충분히 커지도록 유도한다. 즉, 정상은 pull, 이상은 push하는 구조다.

#### Image-level label만 있는 경우

이미지 전체가 anomaly라고 해도 모든 위치가 이상인 것은 아니다. 그래서 anomaly image의 모든 위치를 anomalous로 처리하면 학습이 혼란스러워진다. 이를 피하기 위해 저자들은 $\phi(u)$ 중에서 큰 값 상위 $k$개만 뽑아 평균을 내고, 이를 이미지 anomaly score $q$로 정의한다.

$$
q = \frac{1}{k} \sum \text{top}_k(\phi)
$$

그 다음 image-level loss를 다음처럼 정의한다.

$$
\mathcal{L}_{img} = (1-y)q \alpha y \log(1-\exp(-q))
$$

여기서 $y=0$이면 정상 이미지, $y=1$이면 anomaly 이미지다. 정상 이미지에서는 $q$를 작게 만들고, anomaly 이미지에서는 $q$를 크게 만들도록 학습한다. 핵심은 anomaly image 내부에서도 정말 이상한 부분만 반영되도록 top-$k$를 사용하는 점이다.

이 설계는 practical하다. pixel-level segmentation label이 있을 때는 정교한 위치별 supervision을 쓰고, image-level label만 있을 때는 weakly supervised 방식으로도 활용 가능하기 때문이다.

## 4. 실험 및 결과

## 실험 설정 전반

논문은 두 개의 대표 벤치마크에서 실험한다. 하나는 산업 이상 탐지 데이터셋 **MVTec-AD**, 다른 하나는 one-class classification 성격의 **CIFAR-10**이다. 또한 normal-only case와 anomaly-available case를 모두 평가한다.

MVTec-AD에서는 15개 category에 대해 anomaly detection과 anomaly localization을 함께 평가한다. normal-only case에서는 정상 샘플로만 학습하고 정상 및 이상 샘플로 테스트한다. anomaly-available case에서는 [22]를 따라 normal image에 confetti noise를 더한 synthetic anomaly를 사용한다.

CIFAR-10에서는 한 클래스를 정상으로 두고 나머지 클래스를 이상으로 보는 one-class classification 설정을 따른다. normal-only case에서는 특정 클래스의 train set으로 학습하고, test set에서 같은 클래스는 normal, 다른 클래스 샘플은 anomaly로 둔다. anomaly-available case에서는 CIFAR-100에서 같은 수의 이미지를 뽑아 irrelevant anomaly로 사용한다.

### MVTec-AD 결과

MVTec-AD에서 저자들은 anomaly localization과 image-level anomaly detection 모두를 평가한다.

anomaly localization의 metric은 **pixel-level AUROC**이다. 표 1에 따르면 ADTR는 평균 **97.2**, ADTR+는 **97.5**를 기록한다. 당시 강한 baseline으로 제시된 SPADE가 **96.0**, PSVDD가 **95.7**, FCDD가 **92.0**, KDAD가 **90.7**이므로, ADTR는 normal-only setting에서도 baseline을 앞서고, synthetic anomaly를 추가한 ADTR+는 이를 조금 더 개선한다.

특히 category별로도 전반적으로 높고 안정적이다. 예를 들어 carpet 98.7, leather 98.1, screw 99.3, zipper 97.2 등 대부분 범주에서 매우 높은 localization AUROC를 보인다. 단순 texture anomaly뿐 아니라 metal nut의 뒤집힘 같은 구조적 anomaly도 잡는 qualitative 사례를 제시한다. 이는 단순 pixel mismatch가 아니라 semantic feature discrepancy를 활용하는 설계의 강점을 보여주려는 것이다.

image-level anomaly detection metric은 **image-level AUROC**다. 표 2에서 ADTR는 평균 **96.4**, ADTR+는 **96.9**를 기록한다. 비교 대상인 TS가 **92.5**, PSVDD가 **92.1**, KDAD가 **87.7**, SPADE가 **85.5**이므로, 성능 격차가 꽤 크다. 저자들은 ADTR가 모든 baseline보다 적어도 3.9% 이상 높다고 주장한다. 실제 표 수치상 TS 대비 약 3.9pt 향상이다.

이 결과는 중요한 의미가 있다. 기존 강력한 baseline 중 하나인 teacher-student 계열보다도 높고, 단순 localization이 아니라 image-level detection에서도 강하다는 점에서, feature reconstruction gap 자체가 robust한 anomaly signal로 작동한다는 해석이 가능하다.

### CIFAR-10 결과

CIFAR-10에서는 image-level AUROC로 one-class classification 성능을 측정한다. 표 3에서 ADTR는 평균 **94.7**, ADTR+는 **96.1**을 기록한다. 비교 대상인 KDAD는 **87.2**, GT는 **82.3**, TS는 **82.0**, Loc-Glo는 **70.5**다. 즉, ADTR는 KDAD 대비 **7.5pt** 높고, anomaly-available case의 ADTR+는 여기에 **1.4pt**를 더 올린다.

카테고리별 수치도 매우 높다. 예를 들어 automobile 97.4, frog 97.4, horse 95.8, truck 96.7 등을 보이며, 동물/차량 등 semantic class 간 구분이 필요한 설정에서도 잘 작동한다. 저자들은 qualitative visualization을 통해 anomaly score가 객체 중심에 집중된다고 설명한다. 이는 배경 차이가 아니라 semantic object understanding에 기반해 이상 여부를 판단한다는 주장과 연결된다.

### Ablation Study

이 논문의 설득력은 ablation에서 상당 부분 나온다.

첫째, **attention과 auxiliary query embedding의 역할**이다. 표 4(a)에서 CNN은 94.4, attention 제거(w/o Attn)는 94.8, query 제거(w/o Query)는 94.2, full model(Attn+Query)은 97.2를 기록한다. 즉, transformer라는 이름 자체보다도 **attention + learnable query embedding의 조합**이 핵심이라는 메시지다. query를 제거하면 오히려 CNN보다 나빠진다는 점은 저자들의 “query가 identical shortcut을 막는다”는 주장과 일치한다.

둘째, **pixel reconstruction vs feature reconstruction**이다. 표 4(b)에서 pixel 복원은 91.3, feature 복원은 97.2다. 차이가 매우 크다. 이는 이 논문 전체의 첫 번째 주장, 즉 “semantic feature를 복원 대상으로 삼아야 한다”는 점을 강하게 뒷받침한다.

셋째, **backbone 비교**다. ResNet-18은 95.3, ResNet-34는 95.7, EfficientNet-B0는 96.4, EfficientNet-B4는 97.2를 기록한다. 즉 backbone 종류가 어느 정도 성능에 영향을 주지만, 전반적으로 여러 backbone과 잘 결합된다.

넷째, **multi-scale feature의 효과**다. 마지막 레이어만 쓰면 96.0, multi-scale을 쓰면 97.2다. 향상 폭은 크지 않지만 일관되다. anomaly가 다양한 크기와 성질을 갖는다는 점을 생각하면 자연스러운 결과다.

### 시각화 해석

논문은 feature difference vector $\mathbf{d}(:,u)$를 t-SNE로 시각화한다. 정상과 이상 샘플이 상당히 분리되고, 정상 샘플끼리는 잘 군집화된다고 보고한다. 이 시각화는 ADTR가 normal과 anomaly 사이의 **generalization gap**을 실제 feature discrepancy 공간에서 키운다는 해석을 뒷받침하는 보조 증거다. 다만 t-SNE는 정성적 도구이므로, 이것만으로 이론적 결론을 강하게 내리기는 어렵다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 잘 맞물린다는 점이다. anomaly detection에서 reconstruction 기반 방법의 약점을 정확히 짚고, 이를 단순히 backbone만 바꾸거나 loss만 손보는 수준이 아니라 **복원 대상(feature)과 복원 모델(transformer)을 동시에 재설계**하는 방식으로 해결하려고 한다. 특히 “왜 transformer가 anomaly reconstruction을 제한하는가”를 query embedding 관점에서 설명하려 한 점은 단순 성능 보고 이상의 기여다.

또 다른 강점은 실험 결과가 강하다 는 점이다. MVTec-AD와 CIFAR-10 두 벤치마크에서 normal-only와 anomaly-available setting 모두 높은 성능을 보인다. 특히 pixel reconstruction 대비 feature reconstruction의 큰 성능 차이, 그리고 query embedding의 중요성을 보여주는 ablation은 논문의 핵심 주장과 잘 연결된다.

실용성도 장점이다. 실제 산업 현장에서는 처음에는 정상 샘플만 있지만 시간이 지나며 일부 불량 샘플이 수집될 수 있다. 이 논문은 그런 현실을 반영해 ADTR에서 ADTR+로 확장 가능한 loss를 설계했다. synthetic anomaly나 irrelevant anomaly를 추가하는 간단한 방식으로도 성능을 더 끌어올릴 수 있다는 점은 응용 측면에서 매력적이다.

하지만 한계도 있다. 첫째, transformer가 왜 identity shortcut을 덜 학습하는지에 대한 설명은 **엄밀한 이론 증명보다는 직관적 분석**에 가깝다. attention 식을 통해 논리를 제시하지만, 실제 깊은 transformer 전체가 왜 anomaly reconstruction을 제한하는지에 대한 보다 엄밀한 이론은 없다. 따라서 이 주장은 흥미롭지만 완전히 입증되었다고 보기는 어렵다.

둘째, backbone이 강하게 사전학습된 EfficientNet-B4이고 multi-scale feature를 사용하는 만큼, 성능의 상당 부분이 **pre-trained feature quality**에 의존할 가능성이 있다. 즉, transformer reconstruction 그 자체의 공헌과 사전학습 backbone의 공헌을 완전히 분리해 해석하기는 어렵다. 논문은 여러 backbone ablation을 보여주지만, pre-training 없이 얼마나 버티는지는 다루지 않는다.

셋째, anomaly-available case에서 사용하는 synthetic anomaly나 external irrelevant anomaly는 실제 산업 불량과 분포가 다를 수 있다. 논문은 성능 향상을 보여주지만, 이 방식이 실제 현장의 소량 실불량 데이터에 대해 얼마나 일반적으로 잘 동작하는지는 텍스트만으로는 충분히 확인할 수 없다.

넷째, 계산 비용 측면의 논의가 거의 없다. multi-scale 720-channel feature를 tokenization해서 transformer encoder-decoder로 처리하는 방식은 CNN AE보다 연산량이 클 수 있다. 채널을 256으로 줄이긴 하지만, industrial deployment 관점에서 latency나 memory overhead는 중요한데 이에 대한 비교는 제공되지 않는다.

마지막으로, 일부 세부 설정은 본문과 Appendix 사이에 약간의 표기 차이가 있다. 예를 들어 MVTec-AD의 이미지 크기와 feature map 크기에 관한 설명이 본문과 부록에서 완전히 동일하지 않다. 큰 결론을 흔들 정도는 아니지만, 재현성 측면에서는 더 정교한 정리가 있었으면 좋았을 것이다.

## 6. 결론

이 논문은 anomaly detection에서 reconstruction 기반 접근을 한 단계 발전시킨 작업으로 볼 수 있다. 핵심 기여는 세 가지다. 첫째, raw pixel 대신 **pre-trained semantic feature**를 복원 대상으로 삼아 anomaly와 normal을 더 잘 구분하게 했다. 둘째, reconstruction model로 **transformer with query embedding**을 사용해 CNN의 identical mapping 경향을 완화하려 했다. 셋째, normal-only setting뿐 아니라 pixel-level 또는 image-level anomaly supervision이 일부 존재하는 상황까지 대응하는 **ADTR+ loss 설계**를 제안했다.

실험적으로도 이 방법은 MVTec-AD와 CIFAR-10에서 매우 강한 성능을 보이며, 특히 feature reconstruction과 query embedding의 중요성을 설득력 있게 보여준다. 따라서 이 연구는 산업 비전 검사, 불량 검출, one-class classification 같은 실제 응용에서 의미 있는 잠재력을 가진다. 또한 이후 연구에서는 이 논문의 방향을 따라, 더 강력한 pre-trained representation과 reconstruction bottleneck 설계를 결합하거나, transformer가 anomaly reconstruction을 제한하는 메커니즘을 더 엄밀히 분석하는 방향으로 확장할 수 있다.

전체적으로 보면, 이 논문은 단순히 transformer를 anomaly detection에 가져다 붙인 작업이 아니라, **무엇을 복원할 것인가**와 **어떻게 복원 실패를 유도할 것인가**를 함께 고민한 비교적 잘 설계된 방법론 논문이다. 제공된 텍스트 범위 안에서 판단할 때, 학술적 기여와 실험적 설득력이 모두 좋은 편에 속한다. 다만 발표연도와 arXiv URL은 제공된 텍스트에 명확히 적혀 있지 않아 여기서는 확정적으로 기재하지 않았다.
