# ADTR: Anomaly Detection Transformer with Feature Reconstruction

- **저자**: Zhiyuan You, Kai Yang, Wenhan Luo, Lei Cui, Yu Zheng, Xinyi Le
- **발표연도**: 2022
- **arXiv**: https://ar5iv.labs.arxiv.org/html/2209.01816v3

## 1. 논문 개요

본 논문의 목표는 사전학습된-feature reconstruction을 활용한 산업용 이상 탐지(anomaly detection) 시스템을 제안하는 것이다. 기존 CNN 기반 방법들이 원시 픽셀값을 복원 대상으로 사용하여 semantic 정보를 충분히 담지 못하며, CNN이 "identical mapping" 학습 경로를 선호하여 비정상 샘플도 잘 복원해버리는 근본적 한계가 있다.

논문에서는 먼저 비정상 샘플이 생산 라인에서 극히 부족한 현실적 상황을 가정한다. 전통적인 비지도 학습 방식은 정상 샘플만으로 모델을 학습시키고, 정상 샘플은 잘 복원되지만 비정상 샘플은 복원에 실패한다는 generalization gap에 의존한다. 그러나 픽셀 수준 복원의 semantic 표현력 한계와 CNN의 identical mapping 경향성이라는 두 가지 핵심 문제를 동시에 해결해야 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **Transformer 기반 Feature Reconstruction**이다. 구체적으로 두 가지 핵심 기여가 있다.

첫째, 사전학습된 CNN 백본(EfficientNet-B4)이 추출한 multi-scale feature를 복원 대상으로 사용한다. ImageNet에서 사전학습된 피처 추출기는 정상 샘플과 비정상 샘플을 구별할 수 있는 semantic 정보를 담고 있어, 픽셀값 기반 복원보다 훨씬 풍부한 표현력을 제공한다.

둘째, Transformer의 **auxiliary learnable query embedding**이 "identical mapping" shortcut을 방지한다. CNN은 가중치를 단위행렬($I$)로 쉽게 수렴시켜 정상/비정상 모두를 잘 복원할 수 있지만, Transformer에서 query embedding $\mathbf{q}$는 학습 과정에서 정상 샘플에만 적응하게 되어, 비정상 샘플에 대해 복원 실패를 유도한다.

## 3. 상세 방법 설명

### 전체 파이프라인

ADTR은 세 단계로 구성된다. 첫 번째 Embedding 단계에서 frozen 사전학습 EfficientNet-B4의 layer1~layer5에서 각각 추출한 피처맵을 동일한 크기로 리사이즈 후 concatenation하여 720채널의 multi-scale feature map $\mathbf{f} \in \mathbb{R}^{C \times H \times W}$를 생성한다. 두 번째 Reconstruction 단계에서 이 피처맵을 $H \times W$개의 feature token으로 분할하고, 1×1 convolution으로 720→256 채널로 차원 감소 후 Transformer 인코더-디코더에 입력한다. 디코더는 학습 가능한 query embedding $\mathbf{q} \in \mathbb{R}^{K \times C}$를 입력받아 self-attention과 cross-attention을 통해 feature token을 복원한다. 출력 후 1×1 convolution으로 256→720 채널을 복원한다. 세 번째 Comparison 단계에서 추출 피처와 복원 피처의 차이를 기반으로 이상 score map을 생성한다.

### 인코더 아키텍처

Transformer 인코더는 4개의 레이어로 구성되며, 각 레이어는 multi-head self-attention(8 heads), feed-forward network(FC 256→1024→256), residual connection, layer normalization을 포함한다.

### 디코더 아키텍처

Transformer 디코더도 4개의 레이어로 구성되며, 각 레이어는 self-attention 부분과 cross-attention 부분으로 나뉜다. Self-attention 부분은 multi-head self-attention과 residual connection+layer normalization을 포함하고, cross-attention 부분은 multi-head cross-attention(multi-scale feature tokens를 attend)과 feed-forward network, residual connection+layer normalization을 포함한다.

### 학습 목표 함수

**Normal-sample-only case (Equation 1):**

$$\mathcal{L}_{norm} = \frac{1}{H \times W} || \mathbf{f} - \hat{\mathbf{f}} ||_2^2$$

추출된 피처 $\mathbf{f}$와 복원된 피처 $\hat{\mathbf{f}}$ 사이의 MSE 손실을 사용한다.

**Anomaly-available case with pixel-level labels (Equations 6, 7):**

pseudo-Huber loss $\phi(u)$를 먼저 계산한다:

$$\phi(u) = \left( \left( \frac{1}{C} \sum_{i}^{C} |\mathbf{d}(i,u)| \right)^2 + 1 \right)^{\frac{1}{2}} - 1$$

여기서 $\mathbf{d}(i,u) = \mathbf{f}(i,u) - \hat{\mathbf{f}}(i,u)$는 피처 차분 맵이다.

Push-pull loss $\mathcal{L}_{px}$는 다음과 같다:

$$\mathcal{L}_{px} = \frac{1}{HW} \sum_{u}^{HW} (1 - \mathbf{y}(u)) \phi(u) - \alpha \log(1 - \exp(-\frac{1}{HW} \sum_{u}^{HW} \mathbf{y}(u) \phi(u)))$$

첫 번째 항은 정상 영역($\mathbf{y}(u)=0$)의 복원 피처를 원본 피처로 당겨오고(pull), 두 번째 항은 비정상 영역($\mathbf{y}(u)=1$)의 복원 피처를 원본에서 밀어낸다(push).

**Anomaly-available case with image-level labels (Equations 8, 9):**

비정상 샘플에는 정상/비정상 영역이 혼재할 수 있으므로, $\phi(u)$의 top-k 최대값의 평균을 이미지의 이상 점수로 사용한다:

$$q = \frac{1}{k} \sum \texttt{top\_k}(\phi)$$

이미지 수준 손실 함수는:

$$\mathcal{L}_{img} = (1-y) q - \alpha y \log(1 - \exp(-q))$$

### 추론 시 이상 탐지 및 위치 결정

피처 차분 벡터 $\mathbf{d}(:,u)$의 L2 노름으로 각 위치의 이상 score map을 생성한다:

$$s(u) = || \mathbf{d}(:,u) ||_2$$

이미지 수준 이상 점수로는 average pooled $s(u)$의 최대값을 사용한다.

### Transformer가 "Identical Mapping"을 방지하는 원리 (Section 3.2)

CNN의 완전연결층은 가중치 $\mathbf{w} \in \mathbb{R}^{C \times C}$, 바이어스 $\mathbf{b} \in \mathbb{R}^{C}$에 대해 복원식을 $\hat{\mathbf{x}} = \mathbf{x}^+ \mathbf{w} + \mathbf{b}$로 표현할 수 있다. MSE 손실 하에서 가중치를單位행렬 $\mathbf{w} \to \mathbf{I}$, $\mathbf{b} \to \mathbf{0}$로 간단히 수렴시켜 $\hat{\mathbf{x}} \approx \mathbf{x}^+$를 달성할 수 있으므로, 비정상 샘플 $\mathbf{x}^-$도 잘 복원하게 된다.

반면 Transformer의 attention 연산은:

$$\hat{\mathbf{x}} = \texttt{softmax}(\mathbf{q}(\mathbf{x}^+)^T / \sqrt{C}) \mathbf{x}^+$$

attention map이 단위행렬 $\mathbf{I}$에 근사하려면 query embedding $\mathbf{q}$가 정상 피처 $\mathbf{x}^+$에 highly related되어야 한다. $\mathbf{q}$가 정상 샘플에 맞춰 학습되면, 비정상 샘플 $\mathbf{x}^-$에 대해서는 적절한 attention pattern을 학습하지 못해 복원 실패를 야기한다. Ablation study(Section 4.4)에서 attention 레이어 제거 시 성능이 2.4%, query embedding 제거 시 3% 하락하여 CNN 수준으로 회귀하는 것이 확인되었다.

## 4. 실험 및 결과

### 데이터셋

**MVTec-AD**: 산업용 이상 탐지 데이터셋으로 15개 카테고리(5개 texture, 10개 object)를 포함한다. Pixel-level ground-truth와 image-level 레이블을 모두 제공한다. Normal-sample-only case에서는 정상 샘플만으로 학습하고 정상/비정상 테스트셋으로 평가한다. Anomaly-available case에서는 confetti 노이즈를 정상 샘플에 합성하여 비정상 샘플로 활용한다.

**CIFAR-10**: 10개 클래스의 분류 데이터셋으로, one-class 분류 설정으로 활용한다. 한 클래스의 샘플을 정상으로 학습하고, 다른 클래스들을 비정상으로 평가한다. Anomaly-available case에서는 CIFAR-100을 외부 비관련 데이터셋으로 활용한다.

### MVTec-AD 실험 결과

**Setup**: 이미지 크기 256×256, 피처맵 크기 16×16. Transformer 인코더/디코더 레이어 수 각각 4. EfficientNet-B4 layer1~5 피처 concatenation으로 720채널. 채널 감소량 256. AdamW optimizer, weight decay $1 \times 10^{-4}$, batch size 16. Normal-sample-only case: 500 epochs, lr $1 \times 10^{-4}$, 400 epoch에서 0.1 감소. Anomaly-available case: $\mathcal{L}_{px}$ 사용, $\alpha$=0.003, 먼저 normal-sample-only 모델 로드 후 300 epochs 추가 학습.

**Anomaly Localization (Pixel-level AUROC, Table 1)**: ADTR은 순수 정상 샘플만으로 학습 시 모든 baseline 중 최고 성능을 기록한 SPADE(96.0%)보다 1.2%p 높은 97.2%를 달성했다. Simple synthetic anomalies만으로 ADTR+는 97.5%로 추가 0.3%p 향상했다. Texture 카테고리(Carpet, Grid 등)와 Object 카테고리(Bottle, Cable, Capsule 등) 전반에서 일관되게 최고 또는 최고 수준의 성능을 보였다.

**Anomaly Detection (Image-level AUROC, Table 2)**: 이미지 수준 이상 탐지에서 ADTR은 96.4%를 달성하여 최고 baseline TS(92.5%)보다 3.9%p 앞서며, ADTR+는 96.9%로 0.5%p 추가 향상했다. 특히 Grid, Leather, Tile, Bottle, Hazelnut, Metal Nut 카테고리에서 100%에 가까운 성능을 보였다.

**Qualitative 결과**: 다양한 유형의 이상(텍스처 불규칙성, 색상 변화, 구조적 변형 등)을 높은 위치 결정 정확도로 탐지했다. 특히 Metal Nut 예시에서 플립된 정상 샘플(시각적 텍스처나 색상 변화가 없음)도 성공적으로 탐지하여, semantic 수준의 변화에 대한 민감도를 입증했다.

### CIFAR-10 실험 결과

**Setup**: 이미지 크기 32×32, 피처맵 크기 8×8. Batch size 128. Anomaly-available case에서 $\mathcal{L}_{img}$ 사용, $\alpha$=0.003, k=20.

**Anomaly Detection (Image-level AUROC, Table 3)**: Normal-sample-only case에서 ADTR은 94.7%를 달성하여 이전 최고 성능 KDAD(87.2%)보다 7.5%p 큰 폭으로 향상했다. Anomaly-available case에서 ADTR+는 96.1%로 1.4%p 추가 향상했다. 모든 10개 클래스에서 일관되게 최고 성능을 기록했으며, 특히 Automobile(98.0%), Ship(98.0%), Frog(98.0%) 클래스에서 매우 높은 성능을 보였다.

### Ablation Study (Table 4)

**Attention 및 Query Embedding**: CNN baseline(94.4%) 대비 Attention+Query 사용 시(97.2%) 2.8%p 향상. Attention 제거 시(94.8%) 또는 Query Embedding 제거 시(94.2%) 모두 CNN 수준으로 하락하여 두 구성 요소의 필수성을 입증했다.

**피처 vs. 픽셀 복원**: 픽셀 복원 시(91.3%) 대비 피처 복원 시(97.2%) 5.9%p 향상하여 사전학습 피처의 표현력 우수성을 확인했다.

**백본 선택**: ResNet-18(95.3%), ResNet-34(95.7%), EfficientNet-B0(96.4%), EfficientNet-B4(97.2%)로 모든 백본에서 양호한 성능을 보였으며, 백본이 커질수록 성능이 점진적으로 향상되었다.

**Multi-scale 피처**: Last-layer 피처만使用时(96.0%) 대비 multi-scale 피처使用时(97.2%) 1.2%p 향상하여 다양한 수준의 receptive field가 서로 다른 유형의 이상에 민감하게 반응함을 확인했다.

### Feature Difference Vector 시각화 (Figure 5)

t-SNE를 사용한 피처 차분 벡터 시각화에서 정상 샘플과 비정상 샘플 사이에 넓은 gap이 형성되었다. 정상 샘플은 파란색(낮은 이상 가능성)으로 잘 군집화되고, 비정상 샘플은 빨간색(높은 이상 가능성)으로 분포하여 ADTR이 큰 generalization gap을 만들어냄을 시각적으로 입증했다.

## 5. 강점, 한계

### 강점

첫째, CNN 기반 방법의 두 가지 근본적 한계를 동시에 해결했다. 사전학습 피처 reconstruction으로 semantic 표현력을 확보하고, Transformer의 query embedding 메커니즘으로 identical mapping 경로를 원천 차단한다. 둘째, unified framework로 normal-sample-only case와 anomaly-available case를 모두 지원한다. $\mathcal{L}_{norm}$, $\mathcal{L}_{px}$, $\mathcal{L}_{img}$ 세 가지 손실 함수의 조합으로 다양한 레이블 가용성 상황에 대응한다. 셋째, 다양한 백본(EfficientNet-B0~B4, ResNet-18/34)과 호환되어 실제 적용에서 유연성을 제공한다. 넷째, MVTec-AD와 CIFAR-10 모두에서 최신 성능을 달성했으며, 특히 CIFAR-10에서 KDAD 대비 7.5%p 향상은 의미 있는 격차를 보여준다.

### 한계 및 비판적 해석

첫째, Transformer 기반 접근으로 인해 CNN 기반 방법보다 계산 비용이 높다. 인코더-디코더 구조와 multi-head attention의 복잡성은 실시간 산업용 적용에 제약이 될 수 있다. 둘째, 사전학습 백본에 대한 의존성이 있다. EfficientNet-B4의 피처 추출 품질이 전체 성능에 영향을 미치며, 백본 변경 시 재학습이 필요할 수 있다. 셋째, 논문에서 밝힌 Transformer의 attention 메커니즘이 identical mapping을 방지한다는 설명은 이론적 깊이가 부족하다. Attention map이單位행렬에 근사하려면 query가 입력에 highly related해야 한다는 분석은 타당하지만, 이를 수학적으로 엄밀하게 증명한 것은 아니다. 넷째, anomaly-available case에서의 synthetic anomaly 생성 방식(confetti noise, CIFAR-100)은 실제 산업 환경의 복잡한 이상 패턴을 충분히 대표하지 못할 수 있다.

## 6. 결론

본 논문은 사전학습된 피처를 reconstruction 대상으로 사용하고 Transformer 기반 복원 모델을 채택한 ADTR(Anomaly Detection Transformer)을 제안했다. 사전학습 피처의 풍부한 semantic 정보와 Transformer attention 레이어의 query embedding이 만드는 generalization gap을 활용하여, 정상 샘플은 잘 복원하고 비정상 샘플은 복원 실패하는 특성을 체계적으로 확보했다. 또한 $\mathcal{L}_{norm}$, $\mathcal{L}_{px}$, $\mathcal{L}_{img}$ 손실 함수를 통해 normal-sample-only case와 anomaly-available case를 모두 지원하는 unified framework를 구축했다. MVTec-AD(97.2% pixel-level AUROC)와 CIFAR-10(94.7% image-level AUROC)에서 모두 최신 성능을 달성했다.

실제 산업용 이상 탐지 시스템에 직접 적용 가능하며, 사전학습 피처 추출기와 Transformer 복원 모델이라는 모듈화된 구조 덕분에 다양한 백본과 호환된다. 향후 연구에서는 실시간 처리를 위한 모델 경량화, 다양한 도메인으로의 전이 학습, 그리고 semi-supervised 또는 weakly-supervised 설정에서의 확장 가능성을 탐색할 수 있다.
