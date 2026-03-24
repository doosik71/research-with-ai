# Explainable Deep One-Class Classification

- **저자**: Philipp Liznerski, Lukas Ruff, Robert A. Vandermeulen, Billy Joe Franks, Marius Kloft, Klaus-Robert Müller
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2007.01760

## 1. 논문 개요

이 논문은 이미지 anomaly detection에서 널리 쓰이는 deep one-class classification을 **설명 가능하게(explainable)** 만드는 방법을 제안한다. 기존의 Deep SVDD나 그 변형들은 정상 데이터가 feature space의 특정 중심 근처로 모이도록 학습하고, 이상 샘플은 그 중심에서 멀어지도록 만든다. 이 접근은 anomaly score를 잘 만들 수는 있지만, 왜 어떤 이미지가 이상으로 판단되었는지, 이미지의 어느 부분이 그 판단에 기여했는지를 자연스럽게 보여주기 어렵다. 논문은 바로 이 지점을 문제로 삼는다.

저자들이 해결하려는 핵심 연구 문제는 다음과 같다. **“one-class classification 기반의 anomaly detector가 높은 성능을 유지하면서도, anomaly score와 직접 연결된 공간적 explanation heatmap을 동시에 만들 수 있는가?”** 기존의 gradient-based explanation은 사후적(a posteriori) 설명이라 노이즈가 많고, reconstruction 기반 autoencoder는 heatmap을 만들기 쉽지만 알려진 anomaly나 auxiliary anomaly를 학습에 자연스럽게 넣기 어렵다. 따라서 설명 가능성과 탐지 성능을 동시에 만족하는 방법이 필요하다.

이 문제가 중요한 이유는 anomaly detection이 실제로는 의료, 제조, 보안, 산업 검사처럼 설명이 매우 중요한 영역에서 사용되기 때문이다. 단순히 “이상이다”라고 판정하는 것만으로는 충분하지 않고, 어떤 국소 영역이 이상 판단의 근거인지 보여줘야 사람이 신뢰하고 후속 조치를 할 수 있다. 특히 제조 불량 검출에서는 heatmap 자체가 의사결정의 핵심 출력이다. 이 논문은 그런 응용을 염두에 두고, 설명이 score에 직접 결합된 구조를 설계했다는 점에서 의미가 크다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 매우 명확하다. **네트워크의 출력 자체를 anomaly heatmap으로 만들자**는 것이다. 이를 위해 저자들은 Fully Convolutional Network(FCN)를 사용하고, one-class objective를 이 FCN 출력에 직접 적용한다. 그러면 최종 출력은 더 이상 하나의 전역 feature vector가 아니라, 공간 구조를 보존하는 $u \times v$ 크기의 map이 된다. 이 map의 각 위치는 입력 이미지의 특정 receptive field와 대응되므로, 출력의 각 값이 입력 이미지의 어느 영역과 연결되는지 알 수 있다.

기존 DSVDD 계열은 보통 이미지 전체를 하나의 벡터로 매핑하므로, anomaly score는 얻을 수 있어도 어디가 이상인지 바로 해석하기 어렵다. 반면 FCDD는 fully connected layer를 제거하고 convolution/pooling만 사용하여, 출력 각 픽셀이 입력의 국소 영역에 대응되도록 한다. 그 결과 anomaly score를 만들기 위해 합산되는 값들이 곧 지역별 이상도(local anomaly evidence)가 된다.

또 하나의 차별점은 explanation이 **사후적으로 덧붙여지는 것이 아니라 모델 목적함수에 구조적으로 내장되어 있다는 점**이다. gradient saliency 같은 방법은 예측 후에 gradient를 계산해 설명을 만든다. 하지만 FCDD에서는 모델이 애초에 “heatmap 같은 출력을 만들고, 그 총합이 anomaly score가 되도록” 학습된다. 저자들은 이 점 때문에 FCDD heatmap이 gradient 기반 방식보다 더 구조적이고 덜 noisy하다고 주장한다.

추가로 이 방법은 auxiliary anomalies, 즉 Outlier Exposure(OE) 데이터를 쉽게 받아들일 수 있고, 더 나아가 소수의 실제 anomaly와 그 pixel-level ground-truth map까지 있을 경우 semi-supervised 방식으로 확장할 수 있다. 즉 이 논문은 explainable anomaly detection을 unsupervised와 semi-supervised 모두에 연결한다는 점에서도 설계가 유연하다.

## 3. 상세 방법 설명

### 3.1 배경: Deep one-class classification과 HSC

논문은 기존 one-class classification 계열, 특히 HSC(Hypersphere Classifier)를 출발점으로 삼는다. 기본 생각은 정상 샘플은 feature space의 중심 $\mathbf{c}$ 근처로 보내고, anomaly는 그 바깥으로 보내는 것이다. 논문에서 제시한 HSC objective는 다음과 같다.

$$
\min_{\mathcal{W}, \mathbf{c}} \frac{1}{n}\sum_{i=1}^{n} (1-y_i), h\big(\phi(X_i;\mathcal{W})-\mathbf{c}\big) - y_i \log\Big(1-\exp\big(-h(\phi(X_i;\mathcal{W})-\mathbf{c})\big)\Big)
$$

여기서 $y_i=0$이면 nominal, $y_i=1$이면 anomaly이다. $\phi$는 신경망, $\mathbf{c}$는 중심, $h$는 pseudo-Huber loss이다. pseudo-Huber loss는 다음처럼 정의된다.

$$
h(\mathbf{a}) = \sqrt{|\mathbf{a}|_2^2 + 1} - 1
$$

이 함수는 작은 값에서는 대략 quadratic penalty처럼 행동하고, 큰 값에서는 linear penalty에 가까워진다. 즉 이상치에 대해 지나치게 민감하지 않으면서도 중심으로 모으는 목적을 달성하는 robust loss 역할을 한다.

직관적으로 보면, 정상 데이터는 $h(\phi(X)-\mathbf{c})$가 작아지도록 학습되고, anomaly는 반대로 이 값이 커지도록 학습된다. 이 값 자체가 anomaly score의 기반이 된다.

### 3.2 Fully Convolutional Architecture로의 전환

기존 one-class 방법의 문제는 최종 출력이 전역 벡터라서 공간 해석이 어렵다는 점이다. 이를 해결하기 위해 FCDD는 네트워크를

$$
\phi: \mathbb{R}^{c \times h \times w} \to \mathbb{R}^{1 \times u \times v}
$$

형태로 구성한다. 즉 입력 이미지를 하나의 스칼라나 벡터가 아니라 **하나의 2D feature map**으로 변환한다. 이때 fully connected layer는 사용하지 않고 convolution과 pooling만 사용한다.

이 구조의 핵심은 receptive field다. 출력 map의 한 픽셀은 입력 이미지 전체가 아니라 제한된 영역만 본다. 또한 출력의 상대적 위치와 입력에서 대응되는 receptive field의 상대적 위치가 보존된다. 예를 들어 출력 map의 왼쪽 아래 값은 입력 이미지 왼쪽 아래 근처 영역에서 주로 영향을 받는다. 여러 convolution layer를 쌓으면 receptive field는 커지지만 여전히 공간적 대응관계는 유지된다.

이 점이 explainability의 기반이 된다. 출력 map의 특정 위치가 크다면, 입력 이미지의 해당 receptive field가 anomaly 판단에 많이 기여했다고 해석할 수 있다.

### 3.3 FCDD objective

FCDD에서는 FCN의 출력을 anomaly heatmap의 저해상도 버전으로 본다. 논문은 우선 출력 $\phi(X;\mathcal{W})$에 element-wise pseudo-Huber 형태를 적용하여 다음과 같은 map을 정의한다.

$$
A(X) = \sqrt{\phi(X;\mathcal{W})^2 + 1} - 1
$$

여기서 제곱, 제곱근, 감산은 모두 원소별(element-wise) 연산이다. 따라서 $A(X)\in\mathbb{R}^{u\times v}$의 각 값은 양수이며, 각 공간 위치의 anomaly contribution으로 해석된다.

이때 FCDD objective는 다음과 같다.

$$
\min_{\mathcal{W}} \frac{1}{n}\sum_{i=1}^{n} (1-y_i)\frac{1}{u\cdot v}|A(X_i)|_1 - y_i \log\Big(1-\exp\big(-\frac{1}{u\cdot v}|A(X_i)|_1\big)\Big)
$$

여기서 $|A(X)|_1$은 map의 모든 엔트리를 더한 값이다. 즉 출력 heatmap 전체의 총합이다.

이 식의 의미는 간단하다. 정상 샘플에 대해서는 $|A(X)|_1$을 작게 만들고, anomaly에 대해서는 크게 만들도록 학습한다. 따라서 전체 anomaly score는

$$
s(X)=|A(X)|_1
$$

로 둘 수 있다. 중요한 점은 이 score가 heatmap의 각 픽셀 값을 단순 합산해서 얻어진다는 것이다. 따라서 score에 크게 기여한 위치가 곧 “이상하다고 본 부위”가 된다. 이처럼 explanation과 score가 직접 연결되어 있다는 것이 FCDD의 가장 큰 구조적 장점이다.

### 3.4 Heatmap upsampling

문제는 $A(X)$가 $u \times v$의 저해상도 map이라는 점이다. 원본 입력은 $h \times w$이므로 실제 응용에서는 full-resolution heatmap이 더 유용하다. 하지만 anomaly detection에서는 보통 학습 중 pixel-level ground truth가 없기 때문에 supervised upsampling module을 따로 학습시키기 어렵다.

저자들은 이 문제를 receptive field의 성질을 이용해 해결한다. 각 출력 픽셀은 입력 이미지의 특정 receptive field 중심과 대응되며, 실제 receptive field의 영향은 중심에서 멀어질수록 Gaussian처럼 감소하는 경향이 있다는 기존 관찰을 사용한다. 그래서 각 출력 픽셀 값을 해당 receptive field 중심에 놓인 Gaussian blob으로 퍼뜨려 전체 heatmap을 만든다.

알고리즘적으로는 각 저해상도 heatmap 픽셀 $a$에 대해 그 receptive field 중심 $c$를 찾고, full-resolution map $A'$에 다음을 누적한다.

$$
A' \leftarrow A' + a \cdot G_2(c,\sigma)
$$

여기서 $G_2(c,\sigma)$는 중심 $c$를 갖는 2차원 Gaussian kernel이다. 이 전체 과정은 사실상 **고정 Gaussian kernel을 사용하는 strided transposed convolution**과 같다. stride는 네트워크의 cumulative stride, kernel size는 receptive field 범위에 맞춘다.

이 방식은 학습 없이도 설명 map을 자연스럽게 원해상도로 올릴 수 있다는 장점이 있다. 다만 $\sigma$는 경험적으로 정해야 하고, receptive field가 너무 크면 heatmap이 뭉개져 “blob”처럼 된다는 한계도 논문 부록에서 확인된다.

### 3.5 Semi-supervised pixel-wise objective

이 논문은 FCDD가 ground-truth anomaly map까지 활용할 수 있다고 확장한다. 특히 MVTec-AD에서는 defect별로 극소수의 실제 anomaly와 그 segmentation map이 있을 수 있는데, 이를 학습에 직접 넣는다.

이 경우 각 입력 $X_i$와 대응하는 pixel-level annotation $Y_i$가 있고, upsampled heatmap $A'(X_i)$를 사용한다. 논문은 다음과 같은 pixel-wise objective를 제시한다.

$$
\min_{\mathcal{W}} \frac{1}{n}\sum_{i=1}^{n} \left( \frac{1}{m}\sum_{j=1}^{m}(1-(Y_i)_j)A'(X_i)_j \right) - \log\left( 1-\exp\left( -\frac{1}{m}\sum_{j=1}^{m}(Y_i)_j A'(X_i)_j \right) \right)
$$

여기서 $m=h\cdot w$는 전체 픽셀 수이다. 식의 의미는 정상 픽셀 영역에서는 anomaly score가 낮아지도록 하고, 실제 anomaly로 표시된 픽셀 영역에서는 anomaly score가 높아지도록 한다는 것이다. 즉 이미지 단위 one-class objective를 픽셀 단위 supervision으로 세분화한 버전이라고 볼 수 있다.

이 확장은 reconstruction-based 방법보다 훨씬 자연스럽다. autoencoder는 재구성 오차가 설명이기는 하지만, “이 픽셀은 anomaly여야 한다”라는 지도 신호를 직접 목적함수에 통합하는 구조가 상대적으로 덜 자연스럽다. 반면 FCDD는 원래부터 heatmap을 score의 구성 요소로 사용하므로, pixel-level supervision으로 확장하는 것이 매우 직접적이다.

### 3.6 학습 구성과 구현 관찰

논문은 auxiliary anomaly를 batch 단위로 온라인 샘플링한다. 각 nominal 샘플은 50% 확률로 auxiliary anomaly로 대체되어, 충분히 큰 batch에서는 대체로 balanced batch가 형성된다. 데이터셋별로 optimizer와 augmentation이 조금씩 다르다. 예를 들어 CIFAR-10과 ImageNet 계열은 Adam을, Fashion-MNIST와 MVTec-AD는 SGD with Nesterov를 사용했다. ImageNet, MVTec-AD, Pascal VOC용 네트워크는 VGG11 스타일의 fully convolutional architecture를 사용한다.

논문이 부록에서 강조하는 중요한 실험적 관찰은 두 가지다. 첫째, receptive field가 너무 커지면 detection AUC는 크게 떨어지지 않지만 explanation 품질은 악화될 수 있다. 둘째, upsampling용 Gaussian variance $\sigma$도 heatmap 품질에 영향을 미친다. 즉 설명 가능성을 잘 확보하려면 단순히 분류 성능만 보는 것이 아니라 receptive field와 upsampling hyperparameter를 함께 고려해야 한다.

## 4. 실험 및 결과

### 4.1 표준 anomaly detection benchmark

논문은 먼저 Fashion-MNIST, CIFAR-10, ImageNet에서 one-vs-rest anomaly detection을 수행한다. 각 데이터셋에서 하나의 class를 nominal로 두고, 나머지 class는 test-time anomaly로 취급한다. 학습 시에는 nominal sample과 auxiliary anomaly(OE)를 사용한다. 평가 지표는 일반적인 anomaly detection 지표인 AUC(Area Under ROC Curve)다.

Fashion-MNIST에서는 각 class를 nominal로 번갈아 설정했고, OE로 EMNIST 또는 grayscale CIFAR-100을 사용했다. 저자들은 grayscale CIFAR-100이 EMNIST보다 평균적으로 약 3 AUC 포인트 더 낫다고 보고한다. 평균 AUC는 AE 0.82, DSVDD 0.93, GEO 0.94, FCDD 0.89로 보고되었다. 즉 이 단순 데이터셋에서는 FCDD가 최고 성능은 아니지만, explainability 제약을 두고도 상당히 경쟁력 있는 성능을 유지했다.

CIFAR-10에서는 CIFAR-100을 OE로 사용했다. 평균 AUC는 AE 0.59, DSVDD 0.65, GEO 0.86, GEO+ 0.90, Focal 0.87, OE 포함 GEO+ 0.96, Deep SAD 0.95, HSC 0.96, FCDD 0.95였다. 여기서는 FCDD가 사실상 최신 성능권에 매우 근접한다. 특히 FCN 구조로 제한되었음에도 HSC, Deep SAD와 거의 같은 수준이라는 점이 중요하다.

ImageNet 30개 class subset에서는 ImageNet22k에서 ImageNet1k class를 제거한 데이터를 OE로 사용했다. 평균 AUC는 AE 0.56, Focal 0.56, GEO+ 0.86, Deep SAD 0.97, HSC 0.97, FCDD 0.94였다. 여기서는 FCDD가 최상위권보다 약간 낮지만 여전히 매우 높은 성능을 보인다. 논문의 주장은 “설명 가능성을 위해 구조를 제한했음에도 state-of-the-art에 가깝다”는 것이며, 표 1은 이 주장을 대체로 뒷받침한다.

요약하면, 표준 benchmark에서는 FCDD가 detection 성능만 놓고 절대 최강이라고 하기는 어렵다. 그러나 **해석 가능한 heatmap을 내장한 구조**라는 추가 제약을 감안하면 성능 손실이 작다는 것이 핵심 메시지다.

### 4.2 정성적 heatmap 분석

저자들은 Fashion-MNIST와 ImageNet 예시를 통해 heatmap 품질을 비교한다. 예를 들어 Fashion-MNIST에서 nominal class가 “trousers”일 때, FCDD는 수평 성분을 anomalous하게 강조한다. 바지가 대체로 수직 구조를 가지므로, 이는 의미 있는 설명이다. ImageNet의 “acorn” class에서는 녹색/갈색 영역을 nominal하게 보고, 빨간 헛간이나 흰 눈처럼 class 문맥과 어긋나는 색을 anomalous하게 보는 경향을 보였다. 동시에 단순 색상뿐 아니라 초록색 caterpillar를 anomalous로 감지하는 등 어느 정도 semantic feature도 활용하는 모습이 제시된다.

또한 CIFAR-10에서 nominal class가 “airplane”일 때 OE의 양을 늘리면, FCDD heatmap이 bird, ship, truck 같은 이미지에서 주요 객체 자체로 점점 집중된다는 관찰을 제시한다. 이는 auxiliary anomaly가 충분할수록 모델이 더 본질적인 object-level 단서를 학습할 수 있음을 시사한다.

baseline explanation과 비교하면, gradient heatmap은 대체로 중앙에 뭉친 blob 형태를 보여 spatial context가 약하다고 평가된다. AE heatmap은 reconstruction error와 직접 연결되어 그럴듯한 explanation을 보이지만, OE나 labeled anomalies를 학습에 넣기 어렵다는 구조적 한계가 있다. 저자들은 전반적으로 FCDD heatmap이 gradient보다 덜 noisy하고, AE보다 더 유연하며, 실제 anomaly score와 직접 연결되어 일관적이라고 주장한다.

### 4.3 MVTec-AD: 제조 결함 검출

이 논문의 가장 강한 실험은 MVTec-AD다. 이 데이터셋은 15개 object class의 고해상도 제조 이미지와 pixel-level ground-truth anomaly map을 제공하므로, 탐지 성능뿐 아니라 explanation 자체의 정확도를 정량 평가할 수 있다. 저자들은 heatmap의 pixel score와 binary segmentation label을 비교해 pixel-wise AUC를 계산한다.

MVTec-AD에서는 일반 natural-image OE가 큰 도움이 되지 않는다고 본다. 제조 결함은 “완전히 다른 class의 물체”가 아니라 nominal object의 미세한 defect이기 때문이다. 그래서 논문은 colored blob을 삽입하는 “confetti noise” 형태의 synthetic anomalies를 사용한다. 이 선택은 defect의 local nature를 반영하려는 것이다.

unsupervised setting에서 FCDD의 pixel-wise mean AUC는 **0.92**로 보고되며, 이는 표 2 기준 기존 경쟁법들보다 높다. 비교 대상에는 AE-SS 0.86, AE-L2 0.82, AnoGAN 0.74, CNNFD 0.78, VEVAE 0.86, SMAI 0.89, GDR 0.89, P-NET 0.89 등이 포함된다. 평균뿐 아니라 표준편차도 FCDD가 0.04로 낮아 class 간 일관성도 좋다고 보고된다.

semi-supervised setting에서는 defect type마다 단 하나의 실제 anomalous sample과 그 ground-truth map만 학습에 추가한다. 클래스당 대략 3~8개의 anomaly sample만 사용하는 셈인데, 이 경우 FCDD는 **0.96** pixel-wise mean AUC까지 올라간다. 표준편차도 0.02로 더 줄어든다. 이 결과는 “소수의 실제 anomaly와 localization map만 있어도 설명 성능이 크게 개선된다”는 점을 보여준다.

이는 논문 전체에서 가장 설득력 있는 결과 중 하나다. 왜냐하면 이 논문의 본질적 기여가 단순 이미지-level detection이 아니라 **설명 가능한 localization**에 있기 때문이다. MVTec-AD에서의 우수한 pixel-wise AUC는 FCDD heatmap이 단순 시각화가 아니라 실제로 localization signal로 기능한다는 강한 근거다.

### 4.4 Clever Hans effect 분석

논문은 explainability의 실용적 가치를 보여주기 위해 “Clever Hans effect” 실험도 수행한다. PASCAL VOC의 horse 이미지에는 종종 왼쪽 아래 watermark가 포함되는데, 분류기가 이를 말(horse) 자체보다 더 중요한 단서로 학습하는 문제가 알려져 있다. 저자들은 이를 one-class setting으로 바꾸어, horse를 anomaly class로 두고 ImageNet을 nominal로 둔다. 직관적으로 정상적이라면 말의 몸체가 anomaly heatmap에서 강조되어야 한다.

하지만 결과를 보면 모델은 watermark, bar, fence, grid 같은 spurious feature에 높은 score를 주는 경우가 있었다. 즉 one-class classifier도 의미 있는 semantic concept 대신 데이터 편향이나 우연한 상관관계를 학습할 수 있다는 것이다. 흥미로운 점은, black-box detector였다면 이런 문제를 발견하기 어렵지만 FCDD는 heatmap으로 이를 드러낼 수 있다는 것이다.

이 실험은 성능 수치보다도 논문의 철학을 잘 보여준다. 설명 가능성은 단순히 “보기 좋은 heatmap”이 아니라, 모델이 무엇을 학습했는지 감시하고 잘못된 shortcut을 발견하는 도구가 될 수 있다는 메시지다.

### 4.5 부록 실험: receptive field와 Gaussian variance

부록 A에서는 receptive field 크기의 영향을 분석한다. CIFAR-10에서는 receptive field를 18에서 32까지 바꾸어도 mean AUC가 0.9328에서 0.9235 정도로만 조금 떨어진다. 즉 detection 성능은 크게 흔들리지 않는다. 그러나 MVTec-AD에서는 receptive field가 53일 때 pixel-wise mean AUC가 0.88인데, 243까지 커지면 0.75로 감소한다. 설명 heatmap이 퍼지고 뭉개지는 현상이 localization 성능을 해친다는 뜻이다.

부록 B에서는 upsampling의 Gaussian variance $\sigma$를 조정한다. MVTec-AD에서 $\sigma$를 4에서 16까지 바꾼 결과, mean AUC가 0.8567에서 시작해 14에서 0.9217로 가장 좋고, 16에서는 0.9208로 약간 낮아진다. 즉 upsampling도 단순 시각화 단계가 아니라 정량 성능에 실질적인 영향을 준다.

이 결과는 FCDD가 단순히 FCN을 쓰는 것만으로 충분하지 않고, **receptive field 설계와 heatmap upsampling까지 explanation 품질의 핵심 요소**라는 점을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 anomaly score와 explanation을 구조적으로 결합했다는 점이다. 많은 설명 기법은 예측 후 gradient나 perturbation으로 설명을 만든다. 반면 FCDD는 출력 map의 합이 곧 anomaly score가 되므로, 각 위치가 score에 기여하는 정도가 직접 해석된다. 이 설계는 설명의 일관성과 신뢰성을 높인다.

또 다른 강점은 실험 설계가 폭넓고 설득력 있다는 점이다. Fashion-MNIST, CIFAR-10, ImageNet으로 semantic anomaly detection을 보고, MVTec-AD로 subtle defect localization을 평가하며, Pascal VOC로 spurious feature 문제까지 분석한다. 특히 MVTec-AD에서 unsupervised 0.92, semi-supervised 0.96 pixel-wise mean AUC를 보인 것은 논문 기여를 강하게 뒷받침한다.

세 번째 강점은 semi-supervised 확장의 자연스러움이다. 소수의 true anomaly와 ground-truth map을 쉽게 받아들일 수 있다는 점은 산업 현장에서 중요하다. 실제로 완전 무감독보다는 소량의 라벨이 있는 경우가 많기 때문이다. 논문은 단 3–8개의 anomaly sample만으로도 큰 개선을 보였다.

반면 한계도 분명하다. 첫째, fully convolutional restriction 때문에 일부 benchmark에서는 최상위 detection 성능보다 약간 뒤처진다. 예를 들어 ImageNet에서는 Deep SAD와 HSC의 0.97보다 FCDD는 0.94다. 즉 explainability를 얻는 대가가 아주 없지는 않다.

둘째, heatmap upsampling은 학습 기반이 아니라 hand-crafted Gaussian transposed convolution에 의존한다. 이는 principled하다고 설명되지만, 결국 $\sigma$ 같은 hyperparameter를 경험적으로 맞춰야 한다. 즉 explanation 품질의 일부가 설계 선택에 민감하다.

셋째, 논문은 FCDD가 gradient 기반 설명보다 더 덜 noisy하다고 정성적으로 설득하지만, explanation의 “인과적 정당성”이나 “인간 해석과의 정량적 일치”를 폭넓게 검증한 것은 아니다. MVTec-AD에서는 segmentation AUC라는 강한 평가가 있지만, CIFAR-10이나 ImageNet 같은 semantic anomaly benchmark에서는 주로 시각적 설득에 의존한다.

넷째, Clever Hans 실험은 explainability의 필요성을 잘 보여주지만, 이런 spurious correlation을 **어떻게 체계적으로 교정할 것인지**는 후속 과제로 남는다. 논문도 데이터 정제나 확장 같은 실천적 해결책을 제안할 뿐, 자동적인 bias mitigation 방법을 제공하지는 않는다.

다섯째, 이 논문은 이미지 anomaly detection에 초점을 맞춘다. one-class classification 일반론을 다루지만, 제안 구조와 설명 방식은 사실상 2D spatial input에 최적화되어 있다. 텍스트나 시계열, 그래프 같은 다른 modality로의 일반화는 이 논문만으로는 판단하기 어렵다. 논문이 그 부분을 직접 검증했다고 말할 수는 없다.

종합하면, 이 논문은 “설명 가능한 anomaly detection”이라는 목표에 매우 직접적이고 우아한 구조를 제시했지만, 일부 성능 trade-off와 handcrafted upsampling에 대한 의존성은 남아 있다.

## 6. 결론

이 논문은 Fully Convolutional Data Description(FCDD)라는 explainable deep one-class classification 방법을 제안했다. 핵심은 fully convolutional architecture를 사용해 출력 자체를 저해상도 anomaly heatmap으로 만들고, 그 총합을 anomaly score로 사용하는 것이다. 이 설계 덕분에 anomaly detection과 localization explanation이 같은 표현 위에서 동시에 이루어진다.

실험적으로 FCDD는 표준 anomaly detection benchmark에서 state-of-the-art에 근접한 탐지 성능을 유지하면서도, MVTec-AD에서는 unsupervised 기준 새로운 최고 수준의 pixel-wise explanation 성능을 달성했다. 또한 소수의 실제 anomaly map을 추가로 사용하면 성능이 크게 향상됨을 보였다. 나아가 모델이 watermark 같은 spurious feature에 의존하는 경우도 heatmap으로 드러낼 수 있음을 보여주었다.

실제 적용 측면에서 이 연구는 제조 결함 검출, 의료 영상, 안전·보안 검사처럼 “이상 여부”뿐 아니라 “어디가 왜 이상인가”가 중요한 분야에 특히 의미가 크다. 향후 연구에서는 더 정교한 upsampling, 더 다양한 modality로의 확장, spurious feature를 자동으로 제어하는 학습 전략, 그리고 explanation의 robust/fairness 측면 분석이 중요할 것이다. 저자들도 설명과 anomaly score가 직접 결합된 구조가 사후적 설명보다 공격에 덜 취약할 가능성을 언급하며, 이에 대한 분석을 미래 과제로 남긴다.
