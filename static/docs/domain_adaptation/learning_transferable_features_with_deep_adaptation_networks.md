# Learning Transferable Features with Deep Adaptation Networks

* **저자**: Mingsheng Long, Yue Cao, Jianmin Wang, Michael I. Jordan
* **발표연도**: 2015
* **arXiv**: [https://arxiv.org/abs/1502.02791](https://arxiv.org/abs/1502.02791)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation**을 위한 딥러닝 방법인 **Deep Adaptation Network (DAN)** 을 제안한다. 문제 설정은 다음과 같다. source domain에는 라벨이 있는 데이터가 충분히 있지만, target domain에는 라벨이 없거나 매우 적다. 그런데 두 도메인의 데이터 분포가 다르기 때문에, source에서 학습한 분류기를 target에 그대로 적용하면 성능이 크게 떨어진다. 논문은 이 문제를 **deep feature의 transferability 저하** 관점에서 설명한다.

저자들의 핵심 문제의식은 매우 분명하다. 일반적으로 CNN의 하위 층은 비교적 일반적인 특징을 학습하므로 다른 도메인으로도 잘 옮겨갈 수 있다. 반면 상위 층, 특히 fully connected layer에 가까워질수록 원래 학습한 데이터셋과 작업에 특화된 표현이 형성되므로, 도메인이 달라지면 이 표현은 잘 맞지 않게 된다. 즉, 딥네트워크는 강력한 표현을 만들지만, 그 표현이 도메인 차이를 더 또렷하게 드러내는 방향으로 작동할 수도 있다. 이로 인해 target domain에서의 risk가 커질 수 있다.

따라서 이 논문의 목표는 **깊은 네트워크의 task-specific layer들에서 source와 target의 분포 차이를 명시적으로 줄여**, 더 **transferable한 feature**를 학습하는 것이다. 이를 위해 저자들은 각 상위 layer의 hidden representation을 **RKHS (reproducing kernel Hilbert space)** 에 임베딩하고, source와 target의 평균 임베딩(mean embedding)을 **MK-MMD (multiple kernel maximum mean discrepancy)** 로 맞춘다. 논문은 이를 통해 기존의 단일 층 정렬, 단일 커널 정렬보다 더 효과적인 domain adaptation을 달성했다고 주장한다.

이 문제는 당시 매우 중요했다. 기존의 shallow transfer learning은 도메인 정렬을 하더라도 표현력이 부족했고, 반대로 deep model은 표현력이 강하지만 domain bias를 완전히 제거하지 못했다. 이 논문은 그 둘을 연결하여, **deep representation learning과 distribution matching을 결합한 구조적 방법**을 제시했다는 점에서 의미가 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 다음 한 문장으로 요약할 수 있다. **딥네트워크의 상위 표현층들에서 source와 target의 분포를 여러 커널로 정렬하면, task-specific feature의 transferability를 회복할 수 있다.**

기존 접근과의 차별점은 크게 두 가지다.

첫째, **multi-layer adaptation**이다. 기존 DDC 같은 방법은 네트워크의 한 층만 정렬했다. 그러나 저자들은 transferability가 한 층에서만 무너지는 것이 아니라 여러 상위 층에 걸쳐 점진적으로 떨어진다고 본다. 그래서 $fc6$부터 $fc8$까지 여러 층에 대해 동시에 adaptation regularization을 걸어야 한다고 주장한다. 이는 단순히 feature extractor 일부만 맞추는 것이 아니라, representation과 classifier에 가까운 부분까지 함께 조정한다는 뜻이다.

둘째, **multi-kernel adaptation**이다. MMD는 어떤 kernel을 쓰느냐에 따라 분포 차이를 얼마나 잘 감지하고 줄일 수 있는지가 달라진다. 단일 Gaussian kernel 하나만 쓰면 특정 스케일의 차이만 잘 잡고 다른 차이는 놓칠 수 있다. DAN은 여러 bandwidth를 가진 Gaussian kernel들을 convex combination으로 묶어 사용하고, 그 조합 계수 $\beta$를 최적화한다. 이로써 낮은 차수와 높은 차수의 통계적 차이를 더 잘 포착하려고 한다.

즉, 이 논문은 단순히 “deep feature에 MMD를 추가했다” 수준이 아니다. 저자들이 말하는 본질은 **어느 층을 정렬할 것인가**와 **어떤 kernel family로 정렬할 것인가**를 함께 설계한 데 있다. 이 때문에 DAN은 단일층 단일커널 방식보다 더 강한 적응 성능을 보였다고 해석할 수 있다.

## 3. 상세 방법 설명

### 전체 파이프라인과 네트워크 구조

DAN은 AlexNet을 기반으로 한다. 저자들은 ImageNet으로 사전학습된 AlexNet을 가져와 domain adaptation에 맞게 수정한다. 구조적 선택은 다음과 같다.

하위 convolution layer인 $conv1$부터 $conv3$까지는 일반적인 특징을 학습한다고 보고 **freeze**한다. 중간 층인 $conv4$와 $conv5$는 약간 domain-specific해질 수 있으므로 **fine-tuning**한다. 상위 fully connected layer인 $fc6$, $fc7$, $fc8$는 원래 task에 강하게 맞춰져 있어 transferability가 크게 떨어진다고 보고, 이 층들에는 **MK-MMD regularizer**를 넣어 source와 target의 hidden representation 분포를 맞춘다.

논문이 보여주는 직관은 매우 분명하다. lower layer는 건드리지 않거나 약하게 조정하고, upper layer는 강하게 adaptation한다. 이것이 DAN의 “deep adaptation”이라는 이름의 핵심이다.

### 기본 분류 목적함수

CNN의 기본 분류 학습은 cross-entropy loss를 최소화하는 것이다. 논문에서는 이를 다음과 같이 쓴다.

$$
\min_{\Theta} \frac{1}{n_a}\sum_{i=1}^{n_a} J(\theta(x_i^a), y_i^a)
$$

여기서 $\Theta$는 네트워크 전체 파라미터이고, $J$는 cross-entropy loss이다. $x_i^a, y_i^a$는 라벨이 있는 데이터인데, unsupervised adaptation에서는 사실상 source의 라벨 데이터만 들어간다. 즉, 기본 분류기는 source supervision으로 학습된다.

### MK-MMD: 분포 차이를 측정하는 방법

핵심 regularizer는 **MMD**이다. MMD는 두 분포 $p$와 $q$의 차이를 RKHS에서 평균 임베딩 간 거리로 측정한다. 논문에서는 squared MK-MMD를 다음과 같이 정의한다.

$$
d_k^2(p,q) \triangleq \left| \mathbb{E}_{p}[\phi(x^s)] - \mathbb{E}_{q}[\phi(x^t)] \right|_{\mathcal{H}_k}^{2}
$$

쉽게 말하면, source 표본과 target 표본을 kernel feature space로 보낸 뒤 그 평균이 얼마나 다른지를 보는 것이다. 값이 0이면 두 분포가 같다고 볼 수 있다. 이 점이 domain adaptation에 매우 유용하다. target에 라벨이 없어도, 최소한 **분포를 source와 비슷하게 맞추는 방향**으로 representation을 학습할 수 있기 때문이다.

### Multi-kernel 구성

하나의 kernel만 쓰지 않고 여러 kernel을 결합한다. 논문은 다음과 같은 kernel family를 둔다.

$$
\mathcal{K} \triangleq \left\{ k=\sum_{u=1}^{m}\beta_u k_u : \sum_{u=1}^{m}\beta_u = 1,\ \beta_u \ge 0 \right\}
$$

즉, 여러 PSD kernel $k_u$를 convex combination으로 합친다. $\beta_u$는 각 kernel의 가중치다. 이 설정은 서로 다른 bandwidth를 가진 Gaussian kernel들을 함께 사용하게 해 주며, 분포 차이를 여러 스케일에서 포착할 수 있게 한다.

### 최종 학습 목적함수

DAN의 전체 목적함수는 분류 loss와 여러 층의 MK-MMD regularizer를 함께 최소화하는 형태이다.

$$
\min_{\Theta} \frac{1}{n_a}\sum_{i=1}^{n_a} J(\theta(x_i^a), y_i^a) + \lambda \sum_{\ell=l_1}^{l_2} d_k^2(\mathcal{D}_s^\ell, \mathcal{D}_t^\ell)
$$

논문 구현에서는 $l_1=6$, $l_2=8$로 두어 $fc6$부터 $fc8$까지 adaptation한다. 여기서 $\mathcal{D}_s^\ell$와 $\mathcal{D}_t^\ell$는 각각 $\ell$번째 layer에서의 source와 target hidden representation 집합이다.

이 식은 의미가 명확하다. 첫 번째 항은 source classification을 잘 하도록 만들고, 두 번째 항은 source와 target의 layer-wise representation이 비슷해지도록 강제한다. $\lambda$는 둘 사이의 trade-off를 조절한다. $\lambda$가 너무 작으면 adaptation이 약해지고, 너무 크면 discriminative power가 떨어질 수 있다.

### 왜 여러 층을 동시에 맞추는가

논문은 이 점을 중요한 설계 철학으로 제시한다. 한 층만 정렬하면 다른 상위 층들에 남아 있는 dataset bias를 제거하지 못할 수 있다. 특히 $fc6$, $fc7$, $fc8$는 각각 서로 다른 수준의 추상화와 분류 결정을 담당하므로, 이들 전체에서 정렬이 일어나야 보다 일관된 domain invariance를 만들 수 있다.

저자들은 더 나아가 multi-layer adaptation이 marginal distribution뿐 아니라 conditional distribution 차이까지 완화하는 데 도움이 될 수 있다고 해석한다. 다만 이것은 엄밀한 conditional alignment 알고리즘을 별도로 설계했다기보다는, representation과 classifier 근처 층을 같이 맞추면 결과적으로 그런 효과가 기대된다는 논지에 가깝다.

### 선형 시간 MK-MMD 추정

일반적인 MMD 계산은 pairwise kernel 계산이 필요해서 $O(n^2)$ 비용이 든다. 이는 딥러닝의 mini-batch SGD와 잘 맞지 않는다. DAN의 중요한 실용적 기여는 **linear-time unbiased estimator**를 사용해 이 문제를 해결한 점이다.

논문은 표본들을 네 개씩 묶은 quad-tuple
$$
z_i = (x_{2i-1}^s, x_{2i}^s, x_{2i-1}^t, x_{2i}^t)
$$
을 정의하고, 다음과 같은 함수로 MMD를 추정한다.

$$
g_k(z_i) \triangleq k(x_{2i-1}^s, x_{2i}^s) + k(x_{2i-1}^t, x_{2i}^t) - k(x_{2i-1}^s, x_{2i}^t) k(x_{2i}^s, x_{2i-1}^t)
$$

그리고 전체 MMD는 이 항들의 평균으로 추정한다.

$$
d_k^2(p,q)=\frac{2}{n_s}\sum_{i=1}^{n_s/2} g_k(z_i)
$$

이렇게 하면 계산량이 $O(n)$으로 내려가므로 mini-batch SGD에 쉽게 넣을 수 있다. 저자들은 이것이 기존 MMD 기반 deep adaptation 방법보다 더 scalable하다고 강조한다.

### SGD 업데이트

mini-batch에서 각 layer 파라미터 $\Theta^\ell$의 gradient는 분류 loss gradient와 MK-MMD gradient의 합으로 계산된다.

$$
\nabla_{\Theta^\ell} = \frac{\partial J(z_i)}{\partial \Theta^\ell} + \lambda \frac{\partial g_k(z_i^\ell)}{\partial \Theta^\ell}
$$

즉, 일반적인 supervised backpropagation에 더해, source-target 차이를 줄이는 방향의 gradient가 같이 들어간다. 이 구조 덕분에 네트워크는 분류를 잘하면서도 representation이 domain-invariant해지도록 학습된다.

논문은 Gaussian kernel에 대해 chain rule로 미분을 전개한 예시도 제시한다. 식 (6)은 특정 kernel 항이 weight matrix $W^\ell$에 대해 어떻게 미분되는지 보여준다. 핵심은 kernel 값이 두 feature vector 사이 거리의 함수이므로, source feature와 target feature가 멀면 이를 줄이는 방향의 gradient가 생긴다는 점이다.

### 커널 가중치 $\beta$ 학습

DAN은 네트워크 파라미터 $\Theta$만 학습하는 것이 아니라, multi-kernel의 계수 $\beta$도 최적화한다. 목적은 **두 표본 검정의 test power를 높이고 Type II error를 줄이는 것**이다. 논문은 다음 최적화 문제를 제시한다.

$$
\max_{k \in \mathcal{K}} d_k^2(\mathcal{D}_s^\ell,\mathcal{D}_t^\ell)\sigma_k^{-2}
$$

여기서 $\sigma_k^2$는 추정량의 분산이다. 결국 분포 차이를 잘 드러내면서도 안정적인 kernel 조합을 찾으려는 것이다. 이 문제는 최종적으로 다음의 quadratic program으로 바뀐다.

$$
\min_{\mathbf{d}^{\mathsf T}\beta = 1,\ \beta \ge 0} \beta^{\mathsf T}(Q+\varepsilon I)\beta
$$

논문은 $\varepsilon = 10^{-3}$를 사용한다. 이렇게 얻어진 $\beta$는 여러 Gaussian kernel의 적절한 혼합 비율을 결정한다.

학습은 전체적으로 **alternating optimization**이다. 하나의 단계에서는 mini-batch SGD로 $\Theta$를 업데이트하고, 다른 단계에서는 QP를 풀어 $\beta$를 업데이트한다. 저자들은 두 단계 모두 선형 시간 복잡도로 처리 가능하다고 설명한다.

### 이론 분석

논문은 target risk upper bound도 제시한다. 핵심 정리는 다음 형태다.

$$
\epsilon_t(\theta) \le \epsilon_s(\theta) + 2d_k(p,q) + C
$$

이 식의 의미는 단순하다. target error는 source error와 domain discrepancy에 의해 상한이 결정된다. 따라서 $d_k(p,q)$, 즉 source-target 간 MK-MMD를 줄이면 target risk upper bound를 줄일 수 있다. 논문은 Ben-David 계열의 domain adaptation theory와 kernel embedding theory를 연결해 이 주장을 뒷받침한다.

물론 이 정리는 매우 일반적인 형태의 bound이며, 실제 딥네트워크 최적화가 이 bound를 얼마나 tight하게 반영하는지는 별개의 문제다. 그럼에도 논문 수준에서는 “왜 MMD를 줄이는 것이 adaptation에 도움이 되는가”를 설명하는 중요한 이론적 근거로 기능한다.

## 4. 실험 및 결과

### 데이터셋과 작업

실험은 두 개의 표준 domain adaptation benchmark에서 수행된다.

첫 번째는 **Office-31**이다. 세 도메인 Amazon (A), Webcam (W), DSLR (D)로 이루어져 있고, 총 31개 카테고리가 있다. 논문은 여섯 개 transfer task인 $A \rightarrow W$, $D \rightarrow W$, $W \rightarrow D$, $A \rightarrow D$, $D \rightarrow A$, $W \rightarrow A$를 평가한다.

두 번째는 **Office-10 + Caltech-10**이다. Office와 Caltech-256에서 공통된 10개 클래스만 사용한 데이터셋이며, 여섯 개 transfer task를 더 만든다. 이 데이터셋은 transfer task 수가 많아 dataset bias를 더 균형 있게 보기 위해 사용된다.

또한 DDC와 직접 비교하기 위해 일부 Office-31 task에 대해 **semi-supervised adaptation** 실험도 수행한다. 여기서는 target domain에 카테고리당 3개의 labeled sample을 허용한다.

### 비교 방법

비교 대상은 TCA, GFK 같은 shallow transfer learning 방법과, CNN, LapCNN, DDC 같은 deep learning 기반 방법이다. 또한 DAN 내부 ablation으로 다음 변형들을 평가한다.

* **DAN7**: 한 개 hidden layer만 adaptation
* **DAN8**: 다른 한 개 hidden layer만 adaptation
* **DAN_SK**: multi-layer이지만 single-kernel MMD 사용
* **DAN**: multi-layer + multi-kernel 전체 모델

이 실험 설계는 논문의 주장을 직접 검증하기 위한 것이다. 즉, “multi-layer가 중요한가?”, “multi-kernel이 중요한가?”를 분리해서 본다.

### 구현 세부사항

AlexNet을 ImageNet 사전학습 모델로 초기화하고, $conv1$-$conv3$는 고정, $conv4$-$conv5$와 $fc6$-$fc7$은 fine-tune, $fc8$은 새로 학습한다. $fc8$은 scratch에서 학습하므로 learning rate를 하위 층보다 10배 크게 둔다. SGD with momentum 0.9를 사용한다.

MMD 기반 방법들은 Gaussian kernel을 사용하고, DAN에서는 bandwidth를 여러 스케일로 바꾼 Gaussian kernel family를 사용한다. $\lambda$는 source classifier와 two-sample classifier를 함께 고려하는 validation 절차로 선택한다고 설명한다.

### Office-31 결과

Table 1의 평균 정확도는 다음과 같다.

* TCA: 27.3
* GFK: 27.8
* CNN: 70.1
* LapCNN: 69.5
* DDC: 70.6
* DAN7: 71.1
* DAN8: 71.3
* DAN_SK: 71.5
* **DAN: 72.9**

가장 중요한 점은 DAN이 평균 정확도에서 가장 높다는 것이다. 특히 어려운 task에서 향상이 눈에 띈다. 예를 들어 $A \rightarrow W$에서는 CNN 61.6, DDC 61.8에 비해 DAN이 68.5로 크게 높다. 반면 $D \rightarrow W$와 $W \rightarrow D$처럼 source-target이 이미 비슷한 쉬운 task에서는 향상이 작거나 거의 없다. 이는 논문 해석과도 맞는다. 도메인 차이가 작으면 adaptation 여지가 크지 않다.

### Office-10 + Caltech-10 결과

Table 2의 평균 정확도는 다음과 같다.

* TCA: 44.6
* GFK: 41.0
* CNN: 84.0
* LapCNN: 83.9
* DDC: 84.6
* DAN7: 85.4
* DAN8: 86.4
* DAN_SK: 85.5
* **DAN: 87.3**

여기서도 DAN이 가장 좋은 성능을 보인다. 특히 $C \rightarrow W$에서 DAN은 92.0으로, DDC의 85.5보다 크게 높다. 전체적으로 multi-layer와 multi-kernel의 조합이 일관되게 이득을 준다는 논문 주장이 수치적으로 뒷받침된다.

### Semi-supervised 결과

Table 3에서도 DAN은 DDC보다 높은 평균 성능을 보인다.

* classic unsupervised protocol 평균: DDC 81.2, DAN 84.9
* semi-supervised protocol 평균: DDC 91.9, DAN 93.1

즉, target에 소량의 라벨이 있어도 DAN의 이점은 유지된다.

### 결과 해석

논문이 실험을 통해 강조하는 메시지는 세 가지다.

첫째, **deep feature 자체가 shallow transfer learning보다 훨씬 강하다**. CNN 계열이 TCA나 GFK보다 크게 높다. 이는 좋은 표현 자체의 중요성을 다시 보여준다.

둘째, **단순 semi-supervised regularization은 domain discrepancy 문제를 해결하지 못한다**. LapCNN이 CNN보다 거의 좋아지지 않는 점이 이를 보여준다.

셋째, **DDC의 단일층 단일커널 정렬은 한계가 있다**. DAN의 변형 실험은 multi-layer만 써도, multi-kernel만 써도 각각 성능 향상이 있음을 보이고, 둘을 합친 전체 DAN이 최고 성능을 얻는다. 따라서 논문 핵심 주장인 “multi-layer adaptation”과 “multi-kernel MMD”의 필요성이 실험적으로 설득력을 가진다.

### 정성 분석

논문은 t-SNE 시각화도 제시한다. task $C \rightarrow W$에서 DAN feature는 DDC feature보다 다음 두 측면에서 더 좋다고 해석한다.

하나는 target sample들이 클래스별로 더 잘 분리된다는 점이다. 다른 하나는 source와 target의 같은 클래스들이 더 잘 정렬된다는 점이다. 이는 source classifier가 target에도 더 잘 작동할 수 있게 해 준다.

또한 $\mathcal{A}$-distance 근사도 측정한다. 흥미롭게도 deep feature는 raw feature보다 domain discrepancy를 더 크게 만들 수 있는데, 이는 deep feature가 클래스뿐 아니라 도메인도 더 잘 구별하게 되기 때문이다. DAN은 이런 deep feature의 장점을 유지하면서도 CNN feature보다 더 낮은 domain discrepancy를 보인다고 논문은 주장한다.

### 파라미터 민감도

$\lambda$에 대한 실험에서는 성능이 종 모양의 curve를 보인다. $\lambda$가 너무 작으면 domain alignment가 부족하고, 너무 크면 분류 성능을 해칠 수 있다. 이는 DAN이 단순 regularization이 아니라 **분류력과 domain invariance 사이의 균형 문제**라는 점을 잘 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의와 방법이 매우 잘 맞물린다**는 점이다. 저자들은 deep feature가 위로 갈수록 덜 transferable하다는 기존 분석을 받아들여, 실제로 상위 층들만 선택적으로 adaptation한다. 즉, 네트워크의 표현 특성에 대한 경험적 통찰이 모델 설계로 자연스럽게 연결된다.

또 다른 강점은 **방법의 구조적 완성도**다. 단순히 MMD loss를 넣는 것이 아니라, 여러 층에서 적용하고, 여러 kernel을 조합하고, 이를 효율적으로 학습할 수 있도록 linear-time estimator까지 도입했다. 이 때문에 DAN은 개념적 기여와 실용적 기여를 동시에 가진다.

세 번째 강점은 **ablation이 비교적 설득력 있다**는 점이다. DAN7, DAN8, DAN_SK를 따로 비교함으로써 multi-layer와 multi-kernel이 각각 중요하다는 메시지를 실험으로 뒷받침한다. 이는 단순히 “새 모델이 더 잘 나왔다”보다 훨씬 강한 논증이다.

반면 한계도 분명하다.

첫째, 이 방법은 본질적으로 **marginal distribution matching**에 가까운 접근이다. 논문은 multi-layer adaptation이 conditional distribution 차이에도 도움을 줄 수 있다고 해석하지만, class-conditional alignment를 직접적으로 강제하는 구조는 아니다. 따라서 클래스별 분포가 크게 엇갈리는 경우에는 한계가 있을 수 있다.

둘째, adaptation 대상 층의 선택인 $fc6$-$fc8$은 설득력은 있지만, 엄밀한 자동 선택 기준은 없다. 논문 결론에서도 general-to-specific 경계가 어디인지 principled way로 정하는 문제를 미래 과제로 남긴다. 즉, 어느 층을 freeze하고 어느 층을 adapt할지는 여전히 경험적 설계에 의존한다.

셋째, 실험은 당시 표준 벤치마크 기준으로 충분하지만, 데이터셋 규모가 크지 않고 도메인 종류도 제한적이다. 따라서 매우 복잡한 실제 domain shift에 대해서도 같은 수준의 효과가 유지되는지는 이 논문만으로는 판단하기 어렵다.

넷째, 이론적 분석은 target risk bound를 제시하지만, 실제 딥네트워크의 비선형적 학습 동역학을 정밀하게 설명하는 수준은 아니다. 즉, 이론은 방법의 방향성을 정당화하지만, 성능 향상을 완전히 설명하는 강한 보장은 아니다.

비판적으로 보면, DAN은 이후 등장한 domain adversarial 방법들처럼 representation 학습과 domain confusion을 end-to-end 경쟁적으로 푸는 방식보다 직접적이지 않을 수 있다. 그러나 이 논문 시점에서는 **딥네트워크 내부 여러 층에 distribution matching을 체계적으로 도입했다는 점 자체가 매우 선구적**이었다고 평가할 수 있다.

## 6. 결론

이 논문은 domain adaptation에서 deep feature의 transferability 문제를 정면으로 다루며, **Deep Adaptation Network (DAN)** 라는 구조를 제안했다. DAN은 AlexNet의 상위 task-specific layer들에서 source와 target의 hidden representation을 **MK-MMD** 로 정렬하고, 이를 **multi-layer** 및 **multi-kernel** 방식으로 수행한다. 또한 linear-time unbiased estimator를 사용하여 deep learning에 적합한 학습 절차를 제공한다.

주요 기여를 정리하면 다음과 같다. 첫째, deep network의 여러 상위 층에 걸쳐 domain adaptation regularization을 적용했다. 둘째, 단일 kernel이 아니라 여러 Gaussian kernel의 조합을 최적화하여 더 강한 distribution matching을 수행했다. 셋째, 이 구조가 기존 shallow transfer learning과 기존 deep adaptation 방법보다 더 좋은 성능을 보인다는 것을 표준 벤치마크에서 보여주었다.

실제 적용 측면에서 이 연구는 이후의 deep domain adaptation 분야에 중요한 기반을 제공했다. 특히 “좋은 feature를 학습하는 것”과 “도메인 차이를 줄이는 것”을 하나의 네트워크 목적함수 안에서 함께 다루는 관점은 이후 많은 연구로 이어졌다. 향후 연구에서는 어떤 층을 얼마나 adapt할지 자동으로 결정하는 문제, convolution layer까지 adaptation을 확장하는 문제, 그리고 class-conditional alignment를 더 직접적으로 수행하는 문제가 중요한 후속 과제가 될 수 있다.

전체적으로 볼 때, 이 논문은 domain adaptation에서 **깊은 표현의 층별 transferability**를 명확히 문제화하고, 그것을 **통계적 분포 정렬**로 해결한 대표적인 초기 딥러닝 논문으로 평가할 수 있다.
