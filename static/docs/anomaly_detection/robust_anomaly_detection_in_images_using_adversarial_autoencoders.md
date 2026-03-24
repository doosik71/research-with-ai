# Robust Anomaly Detection in Images using Adversarial Autoencoders

- **저자**: Laura Beggel, Michael Pfeiffer, Bernd Bischl (Technical University of Munich)
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1901.06355

## 1. 논문 개요

이 논문은 이미지 anomaly detection에서 널리 쓰이는 autoencoder 기반 방법이, **학습 데이터에 소량의 anomaly가 섞여 있을 때 매우 취약해진다**는 문제를 정면으로 다룬다. 기존 접근은 정상 데이터만으로 학습된 autoencoder가 정상 이미지는 잘 복원하고 비정상 이미지는 잘 복원하지 못할 것이라는 가정에 기대고 있다. 하지만 실제 환경에서는 학습 데이터가 완전히 깨끗하다고 보장하기 어렵다. 품질 검사에서는 불량 샘플이 소량 섞일 수 있고, 의료나 감시 데이터에서도 annotation 오류나 검수 비용 문제 때문에 완전한 clean set을 확보하기 어렵다.

논문의 핵심 문제의식은 단순하다. **학습 목표가 전체 학습 샘플의 reconstruction error를 줄이는 것이라면, 학습을 오래 할수록 anomaly까지도 점점 더 잘 복원하게 된다**는 것이다. 그러면 정상과 비정상의 reconstruction error 분포가 겹치게 되고, 결국 anomaly detection 성능이 악화된다. 저자들은 이것이 단순한 구현상의 문제가 아니라, autoencoder 학습 목표 자체에서 비롯되는 구조적 한계라고 본다.

이를 해결하기 위해 저자들은 adversarial autoencoder(AAE)를 이용해 latent space에 prior distribution을 강제하고, 이 latent likelihood를 anomaly 판단에 활용한다. 더 나아가 latent space에서 잠재적 anomaly를 찾아 학습 과정에서 제외하거나 역으로 분리되도록 다시 학습시키는 **Iterative Training Set Refinement (ITSR)**라는 절차를 제안한다. 목표는 contaminated training set에서도 정상 데이터 manifold를 더 견고하게 학습하는 것이다.

이 문제는 실용적으로 매우 중요하다. anomaly detection은 대개 anomaly가 희귀하고 형태도 다양해서 supervised learning으로 풀기 어렵다. 따라서 정상 데이터만으로 모델을 학습하는 one-class 혹은 unsupervised 접근이 현실적인데, 이때 학습 데이터 contamination에 강한 방법은 실제 배포 가능성을 크게 높여 준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 두 단계로 이해할 수 있다.

첫째, **reconstruction error만으로 anomaly를 판단하는 것은 부족하다**는 점이다. 표준 autoencoder에서는 anomaly도 충분히 학습되면 잘 복원될 수 있다. 따라서 저자들은 AAE를 사용해 latent representation $z=f(x)$가 특정 prior $p(z)$를 따르도록 만들고, 입력 샘플의 latent code가 이 prior 아래에서 얼마나 그럴듯한지, 즉 **likelihood**를 추가적인 anomaly signal로 사용한다. 직관적으로 정상 데이터는 prior의 고밀도 영역으로 매핑되고, anomaly는 저밀도 영역으로 밀려날 가능성이 크다.

둘째, 이 latent-space signal을 단순히 테스트 시점 점수로만 쓰지 않고, **학습 데이터 정제(training set refinement)**에 활용한다는 점이 이 논문의 더 중요한 기여다. 즉, latent space에서 1-class SVM으로 “정상 다수 집합”을 추정하고, 그 바깥에 놓인 샘플들을 잠재적 anomaly 후보로 간주한다. 그다음 이 후보들의 학습 기여를 제거하거나 약화하면서 autoencoder를 다시 학습하면, 모델이 정상 데이터에 더 집중하게 된다. 이후 retraining 단계에서는 일부 anomaly 후보에 음수 가중치를 주어, 정상과 anomaly 사이의 reconstruction 및 latent likelihood 분리를 더 강하게 만든다.

기존 AAE anomaly detection 연구와의 차별점은, 단순히 latent prior를 두는 데 그치지 않고 다음 세 가지를 함께 결합했다는 데 있다. 첫째, reconstruction error와 latent likelihood의 결합 기준을 사용한다. 둘째, latent space에서 anomaly 후보를 탐지해 학습 세트를 반복적으로 정제한다. 셋째, 마지막 retraining 단계에서 anomaly 후보의 reconstruction을 의도적으로 나쁘게 만들어 separability를 강화한다. 저자들은 이 조합이 contamination에 특히 강하다고 주장한다.

## 3. 상세 방법 설명

### 3.1 기본 autoencoder와 한계

표준 autoencoder는 encoder $f:\mathcal{X}\to\mathcal{Z}$와 decoder $g:\mathcal{Z}\to\mathcal{X}$로 구성되며, 입력 이미지 $x\in\mathbb{R}^n$를 latent code $z=f(x)$로 압축한 뒤 다시 $x'=g(z)$로 복원한다. 학습 목표는 reconstruction loss $L(x,x')$를 최소화하는 것이다. 논문에서는 pixelwise mean squared error나 image space에서의 Euclidean distance를 대표적인 예로 든다.

테스트 시에는 reconstruction error가 큰 샘플을 anomaly로 판정한다. 즉, 임계값 $T_{rec}$를 두고

$$
L(y, g(f(y))) > T_{rec}
$$

이면 anomaly로 본다. $T_{rec}$는 학습 데이터 reconstruction error 분포에서 최대값이나 높은 percentile, 예를 들어 90th percentile로 정한다.

문제는 이 방식이 **학습 데이터 전체의 reconstruction을 잘하게 만드는 방향으로 최적화**된다는 데 있다. 학습 데이터에 anomaly가 소량 섞여 있더라도, 초기에는 정상 샘플이 다수이므로 정상 복원이 먼저 좋아진다. 하지만 시간이 지나 정상 샘플의 error가 줄어 gradient가 작아지면, 아직 error가 큰 anomaly 샘플이 더 큰 gradient를 만들어 업데이트를 주도하게 된다. 그러면 후반에는 anomaly reconstruction error가 빠르게 감소하고, 결국 정상과 anomaly의 error 차이가 사라진다. 논문은 MNIST에서 정상 95%, anomaly 5%인 오염 학습셋으로 이 현상을 실험적으로 보여 준다.

저자들은 early stopping도 근본 해법이 아니라고 본다. 너무 일찍 멈추면 정상 클래스 자체를 충분히 정확히 모델링하지 못하고, 너무 오래 학습하면 anomaly까지 잘 복원한다. 모델 용량을 줄이는 것도 정상 복원 품질까지 떨어뜨려 threshold가 높아지고 false negative 위험이 생긴다.

### 3.2 Adversarial Autoencoder의 도입

AAE는 일반 autoencoder에 adversarial training을 추가해 latent space에 원하는 prior $p(z)$를 강제한다. encoder가 만든 latent code와 prior에서 샘플링한 벡터를 discriminator가 구분하도록 학습하고, encoder는 discriminator를 속이도록 학습한다. 이 과정을 통해 latent distribution이 지정한 prior에 가까워지도록 유도된다.

논문에서 중요한 직관은 다음과 같다. 정상 데이터가 주로 형성하는 데이터 분포 $p_{data}(x)$를 잘 따르는 샘플은 prior의 고확률 영역에 놓이는 latent code를 가질 가능성이 높다. 반대로 anomaly는 다음 둘 중 하나일 가능성이 있다.

첫째, 복원이 잘 안 되어 reconstruction error가 커진다.
둘째, 복원이 어느 정도 되더라도 latent code가 prior의 저확률 영역에 놓인다.

따라서 AAE에서는 reconstruction error뿐 아니라 latent likelihood $p(f(x))$도 anomaly score가 된다.

### 3.3 Likelihood-based Anomaly Detection

AAE로부터 얻은 latent code $\hat z=f(\hat x)$에 대해 prior 아래 likelihood $p(\hat z)$를 계산하고, 이것이 작으면 anomaly로 판정한다. 임계값 $T_{prior}$는 학습셋 전체의 latent likelihood 분포를 보고 정한다. 논문에서는 이상적으로는 예상 anomaly rate $\alpha$에 해당하는 percentile을 사용할 수 있다고 말하지만, 실제로는 encoder 근사 오차와 finite sample bias 때문에 고정된 설정이 더 안정적일 수 있다고 설명한다. 실험에서는 $10$th percentile을 사용했다.

최종적으로 AAE의 anomaly 판정은 reconstruction 기준과 likelihood 기준을 단순 OR 결합한다. 즉, 새 샘플 $y$에 대해 다음 둘 중 하나를 만족하면 anomaly로 본다.

$$
L(y, y') > T_{rec}
$$

또는

$$
p(f(y)) < T_{prior}.
$$

저자들은 reconstruction error와 likelihood를 2차원 특징으로 보고 그 위에서 다시 1-class SVM을 돌리는 대안도 시도했지만 성능 향상은 없었다고 명시한다. 즉, 단순 결합이 가장 실용적이었다는 뜻이다.

Fashion-MNIST에서 정상 T-shirt, anomaly Pullover인 clean setting 실험에서는 reconstruction만 쓸 때와 likelihood만 쓸 때 Balanced Accuracy(BAcc)가 각각 0.72, 0.73 정도였고, 둘을 결합하면 0.80으로 좋아졌다고 보고한다. 이것은 두 신호가 서로 보완적이라는 점을 보여 준다.

### 3.4 ITSR: Iterative Training Set Refinement

이 논문의 핵심 방법은 ITSR이다. 전제는 AAE가 unimodal prior, 예를 들어 Gaussian prior를 쓰면 정상 샘플이 latent space 중심부에 모이고, anomaly는 상대적으로 바깥쪽 저밀도 영역에 위치할 가능성이 높다는 것이다. 특히 정상 클래스가 비교적 homogeneous한 품질 검사 문제에서는 이 가정이 더 잘 맞는다.

저자들은 latent representation 위에서 1-class SVM을 학습해 정상 다수 영역을 추정한다. 1-class SVM은 anomaly 비율의 상한을 의미하는 $\nu$를 하이퍼파라미터로 받는다. 실험에서는 true anomaly rate를 모르더라도 고정값 $\nu=0.02$를 사용한다. 커널은 RBF이다.

훈련 샘플 $x_i$ 각각에는 가중치 $w_i$를 부여하고, 가중 reconstruction loss를 다음과 같이 정의한다.

$$
L_w = \sum_{i=1}^{N} w_i , L\bigl(x_i, g(f(x_i))\bigr).
$$

논문은 이 가중치를 adversarial training에도 동일하게 사용한다고 설명한다. 즉, 샘플별로 학습에 얼마나 기여할지를 직접 조절한다.

ITSR은 세 단계로 이루어진다.

#### (1) Pretraining

먼저 전체 학습셋으로 AAE를 일정 epoch 동안 학습한다. 이때 모든 샘플의 가중치는 $w_i=1$이다. 목적은 latent space에 어느 정도 구조가 형성되도록 하는 것이다. 이 단계가 끝나야 1-class SVM이 의미 있는 경계를 찾을 수 있다.

#### (2) Detection and Refinement

pretraining으로 얻은 latent representation에 대해 1-class SVM을 적용해 anomaly 후보 집합 $\hat A$를 찾는다. 이 후보들에는

$$
w_i = 0
$$

을 부여하여 이후 학습에서 제외한다. 그런 다음 reduced training set $X\setminus \hat A$로 다시 짧게 학습한다. 이 탐지와 재학습 과정을 $d$번 반복한다. 반복하면서 정상 manifold에 대한 모델은 점점 더 정교해지고, anomaly 후보는 학습에서 배제되므로 그 복원 품질이 개선될 유인이 줄어든다.

#### (3) Re-training

이 단계는 단순 배제를 넘어 **분리도(separation)**를 의도적으로 높이기 위한 절차다. anomaly 후보 전체에 음수 가중치를 주면, 해당 샘플들의 reconstruction error를 키우는 방향으로 학습을 유도할 수 있다. 그러나 detection 단계에서 false positive가 포함될 수 있으므로, 후보 전부에 무조건 음수 가중치를 주는 것은 위험하다.

그래서 저자들은 anomaly 후보 집합 $\hat A$ 안에서 reconstruction error 분포를 보고 retraining threshold $T_{retrain}$을 정의한다. 본문 텍스트에는 표기가 다소 불분명하지만, 핵심 의도는 **후보 중에서도 reconstruction error가 충분히 큰 샘플만 더 강하게 anomaly 방향으로 밀어내고, 그렇지 않은 샘플은 단순 제외 상태에 두는 것**이다. 논문에서는 $80$th percentile을 사용한다.

구체적으로, 후보 샘플 $x_i\in \hat A$에 대해 reconstruction error가 $T_{retrain}$보다 크면

$$
w_i = w_{anomaly} < 0
$$

로 두고, 그렇지 않으면

$$
w_i = 0
$$

으로 둔다. 이로써 이미 anomaly일 가능성이 큰 샘플은 더 나쁘게 복원되도록 밀어내고, false positive로 잘못 잡힌 정상 유사 샘플은 과도하게 훼손하지 않도록 한다.

이 설계의 결과는 다음과 같다. 정상 데이터는 latent high-density region과 낮은 reconstruction error를 유지하고, anomaly는 latent low-density region으로 밀리거나, 혹은 high-density region에 있더라도 reconstruction error가 높아지도록 유도된다. 즉, 논문이 원하는 “둘 중 하나는 반드시 anomaly 신호를 주는 구조”를 더 명확히 만든다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

실험은 MNIST와 Fashion-MNIST에서 수행되었다. 두 데이터셋 모두 원래 train-test split을 사용하며, 학습 샘플은 총 60,000개, 테스트 샘플은 10,000개이다.

MNIST에서는 digit ‘0’을 정상, digit ‘2’를 anomaly로 정의했다. Fashion-MNIST에서는 T-shirt를 정상으로 고정하고, anomaly를 두 경우로 나누었다. 하나는 Boot로, 비교적 쉬운 경우이고, 다른 하나는 Pullover로, T-shirt와 시각적으로 더 유사해 어려운 경우다.

학습 데이터 contamination 비율은 $\alpha\in\{5\%, 1\%, 0.1\%\}$로 설정했다. 즉, 정상 클래스가 다수이고 anomaly 클래스가 극소량 섞이도록 구성했다. 이는 실제 anomaly detection 상황과 잘 맞는 설정이다.

### 4.2 평가 방식

평가 지표는 True Positive Rate(TPR), True Negative Rate(TNR), 그리고 Balanced Accuracy(BAcc)이다. anomaly를 positive event로 보고,

$$
BAcc = \frac{TPR + TNR}{2}
$$

로 정의한다. 저자들은 anomaly detection에서 F1 score가 자주 쓰이지만, F1은 true negative 성능을 반영하지 못하므로 본 문제에서는 적절하지 않다고 설명한다. contamination 상황에서는 false alarm과 missed anomaly 사이의 균형이 중요하므로 BAcc를 선택한 것이다.

또한 테스트셋을 두 가지로 나누어 평가한다. 하나는 **observed anomalies**, 즉 학습 중 contamination에 실제로 섞여 있던 anomaly 클래스와 같은 종류를 포함한 테스트셋이고, 다른 하나는 **unobserved anomalies**, 즉 학습 때 전혀 보지 못한 새로운 anomaly 클래스들을 포함한 테스트셋이다. 이 구분은 매우 중요하다. ITSR이 학습셋에 섞여 있던 anomaly를 배제하는 방식이므로, observed anomaly에 특히 강해질 것으로 기대되기 때문이다.

### 4.3 모델 구조와 학습 세부사항

기본 AE의 encoder와 decoder는 각각 fully connected layer 2개, 각 층 1000 유닛으로 구성된다. 활성함수는 대부분 ReLU이고, encoder 최종층은 linear, decoder 최종층은 sigmoid다. latent dimension은 일반 AE에서는 32이다.

AAE에서는 latent space를 2차원으로 줄이고, discriminator를 추가한다. discriminator 역시 fully connected 2층, 각 1000 유닛이며 마지막 층은 sigmoid를 사용한다. encoder 각 층 뒤에는 batch normalization을 적용한다. prior는 2차원 Gaussian $p(z)=[\mathcal{N}(0,1)]^2$이다.

학습 epoch는 AE의 경우 MNIST에서 4000, Fashion-MNIST에서 10000이며 Adam optimizer를 사용한다. AAE는 1000 epoch 학습한다.

ITSR은 동일한 AAE 구조를 사용하고, pretraining 500 epoch 후에 detection/refinement를 $d=10$회 반복하며 각 반복마다 100 epoch씩 학습한다. retraining은 MNIST에서 1000 epoch, Fashion-MNIST에서 500 epoch 진행한다. 1-class SVM은 RBF kernel, $\nu=0.02$를 사용한다.

AAE와 ITSR에서 anomaly detection용 threshold는 reconstruction error의 90th percentile과 likelihood의 10th percentile을 쓴다. 일반 AE는 reconstruction threshold만 사용한다.

### 4.4 핵심 결과

논문이 보여 주는 가장 중요한 결과는, contamination이 있는 학습셋에서 **ITSR이 AE와 AAE보다 consistently 더 높은 BAcc를 보인다**는 점이다.

특히 contamination rate가 5%일 때, observed anomaly 기준으로 MNIST에서는 ITSR이 기존 AE보다 BAcc를 30% 이상 개선했다고 보고한다. Fashion-MNIST에서도 전반적으로 AAE보다 20% 이상 높은 경우가 많았고, 가장 어려운 T-shirt vs Pullover 설정에서는 AAE 대비 약 30% 개선이 있었다고 한다.

표 1의 수치를 보면 이 경향이 더 명확하다.

관찰된 anomaly 유형에 대해:

- MNIST, $\alpha=1%$: AE 0.69, AAE 0.91, ITSR 0.94
- Fashion-MNIST T-shirt vs Boot, $\alpha=1%$: AE 0.74, AAE 0.89, ITSR 0.92
- Fashion-MNIST T-shirt vs Pullover, $\alpha=1%$: AE 0.74, AAE 0.70, ITSR 0.81

또한 $\alpha=0.1%$처럼 anomaly가 훨씬 더 희귀한 경우에도 ITSR은 여전히 유리했다. 예를 들어 T-shirt vs Pullover observed anomaly에서 AE 0.73, AAE 0.71, ITSR 0.80이었다. 이는 contamination이 매우 작더라도 refinement 과정이 의미 있는 이점을 준다는 뜻이다.

unobserved anomaly 유형에 대해서는 ITSR의 개선 폭이 observed anomaly보다 작다. 이는 자연스러운 결과다. 새로운 anomaly는 학습 과정 자체를 오염시키지 않았기 때문에, ITSR의 직접적인 이득은 제한적이다. 그럼에도 논문은 ITSR이 unobserved anomaly에 대해서도 기존 방법과 동등하거나 약간 더 낫다고 보고한다. 예를 들어 MNIST $\alpha=1%$에서 AE 0.69, AAE 0.91, ITSR 0.93이고, Fashion-MNIST T-shirt vs Pullover $\alpha=1%$에서 AE 0.74, AAE 0.78, ITSR 0.81이다.

### 4.5 결과 해석

결과 해석에서 중요한 포인트는 ITSR이 **정상 샘플 reconstruction은 크게 손상시키지 않으면서 anomaly reconstruction error를 상승시킨다**는 점이다. 논문 Fig. 5(d)-(f)에 따르면, refinement 전후를 비교했을 때 정상 클래스 reconstruction error는 거의 변하지 않지만 anomaly 클래스 reconstruction error는 크게 증가한다. 이것은 단순히 전체 모델을 나쁘게 만든 것이 아니라, 정상 manifold에는 더 집중하고 anomaly에는 덜 민감하도록 모델이 재구성되었음을 의미한다.

또한 latent likelihood와 reconstruction error의 조합이 더 명확해진다. 즉, prior high-density region에 잘못 들어온 anomaly는 reconstruction error가 커지고, reconstruction이 상대적으로 되는 anomaly는 latent likelihood가 낮아지는 식으로 상호 보완 구조가 강해진다. 이 점이 AAE 단독보다 ITSR이 더 강한 이유로 해석된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제를 매우 현실적으로 정의하고, 그 원인을 학습 dynamics 수준에서 설명했다는 점**이다. 단순히 “contamination에 강한 새 모델”을 제안하는 것이 아니라, 왜 표준 autoencoder가 contamination에서 실패하는지 설명한다. 특히 학습 초반에는 정상 다수 클래스가 업데이트를 지배하지만, 후반에는 error가 큰 anomaly 샘플이 gradient를 크게 만들어 anomaly reconstruction이 개선된다는 설명은 설득력이 높다.

두 번째 강점은 AAE의 latent prior를 anomaly scoring뿐 아니라 **training data refinement**에 연결했다는 것이다. 이는 test-time detector 설계와 training-time robustness 확보를 하나의 틀로 묶어 준다. reconstruction error와 latent likelihood의 결합도 매우 직관적이며, 각 신호의 약점을 다른 신호가 보완하는 구조를 잘 보여 준다.

세 번째 강점은 observed anomaly와 unobserved anomaly를 분리해 평가했다는 점이다. contamination robust method는 자칫 training에 섞인 anomaly에만 과적응할 수 있는데, 논문은 새로운 anomaly에 대해서도 성능이 나빠지지 않음을 확인하려 했다. 이 실험 설계는 방법의 실제 유용성을 평가하는 데 적절하다.

하지만 한계도 분명하다.

가장 먼저, 이 방법은 **정상 데이터가 latent space에서 비교적 하나의 응집된 구조를 이룬다**는 가정에 크게 기대고 있다. 논문도 품질 검사처럼 normal class가 homogeneous한 경우를 특히 적합한 응용으로 언급한다. 반대로 정상 데이터 자체가 매우 heterogeneous하면 1-class SVM이 정상의 외곽 영역을 anomaly 후보로 잘못 많이 잡을 가능성이 있다. 저자도 $\nu$와 retraining threshold 설정이 데이터의 homogeneity에 따라 달라진다고 인정한다.

둘째, 방법은 여러 하이퍼파라미터에 의존한다. 예를 들어 $\nu$, refinement 반복 횟수 $d$, pretraining epoch 수, retraining threshold percentile 등이 있다. 논문은 true anomaly rate를 몰라도 $\nu=0.02$로 꽤 robust했다고 주장하지만, 다양한 데이터셋에서 이 값이 항상 잘 작동할지는 논문 텍스트만으로는 확실히 판단하기 어렵다.

셋째, 실험이 모두 MNIST와 Fashion-MNIST 같은 비교적 단순한 grayscale benchmark에 집중되어 있다. 논문은 산업 품질 검사나 의료 영상 같은 실제 응용 가능성을 강조하지만, 제공된 텍스트만 보면 더 복잡한 자연 이미지나 실제 산업 데이터셋 검증은 제시되지 않았다. 따라서 실전 대규모 데이터에 대한 일반화는 추가 검증이 필요하다.

넷째, retraining 단계의 수식 표기는 텍스트 추출본에서 일부 혼동이 있다. 특히 threshold 계산식에서 $L(x, f(g(x)))$처럼 일반적인 reconstruction 표기와 다른 부분이 보이는데, 이는 원문 수식 렌더링 또는 추출 과정의 오류일 가능성이 있다. 그러나 본문 설명의 의도는 reconstruction error percentile 기반으로 anomaly 후보 중 일부만 음수 가중치로 재학습한다는 점으로 이해된다. 이처럼 제공된 텍스트 기준으로는 몇몇 식의 정확한 원문 표기를 완전히 복원하기 어렵다.

비판적으로 보면, ITSR은 본질적으로 “latent space에서 잘 분리되면 refinement가 성공한다”는 가정 위에 세워진다. 따라서 latent prior를 잘 맞추지 못하거나 anomaly가 normal manifold 깊숙이 섞여 들어가는 경우에는 효과가 제한될 수 있다. 논문도 likelihood 단독 기준이 불완전하다고 인정하며, local cluster anomaly가 high-likelihood를 가질 수 있다고 말한다. 결국 이 방법의 성능은 AAE latent organization의 품질에 상당 부분 좌우된다.

## 6. 결론

이 논문은 autoencoder 기반 이미지 anomaly detection의 중요한 약점을 짚는다. 즉, **학습 데이터에 anomaly가 조금만 섞여 있어도, 충분히 오래 학습한 autoencoder는 anomaly까지 잘 복원하게 되어 탐지 성능이 무너질 수 있다**는 점이다. 저자들은 이를 해결하기 위해 adversarial autoencoder를 기반으로 latent prior를 부과하고, reconstruction error와 latent likelihood를 함께 활용하는 anomaly criterion을 제시했다. 더 나아가 latent space에서 anomaly 후보를 탐지하여 학습셋을 반복적으로 정제하고, 최종적으로 anomaly 후보의 reconstruction을 더 나쁘게 만드는 ITSR 절차를 제안했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, reconstruction error와 latent likelihood를 결합한 anomaly scoring. 둘째, 1-class SVM을 이용한 latent-space anomaly candidate detection. 셋째, iterative refinement와 retraining을 통해 contaminated training set에서도 정상 manifold 중심의 표현을 학습하게 만드는 training procedure다.

실험 결과는 이 방법이 특히 **학습 중 실제로 섞여 있던 anomaly 유형에 대해 큰 개선**을 보이며, 동시에 학습 중 보지 못한 새로운 anomaly에 대해서도 성능을 해치지 않음을 보여 준다. 따라서 이 연구는 “완전히 깨끗한 정상 데이터만으로 학습해야 한다”는 autoencoder anomaly detection의 실무적 부담을 줄여 준다.

실제 적용 측면에서 이 연구는 품질 검사처럼 anomaly rate가 매우 낮고 정상 클래스가 비교적 일관된 환경에서 특히 유망하다. 향후에는 더 복잡한 이미지 도메인, 시계열이나 spectrogram 같은 다른 고차원 데이터, 그리고 semi-supervised 혹은 active learning과의 결합으로 확장될 가능성이 있다. 논문 자체도 이러한 확장 가능성을 언급하고 있으며, 제공된 텍스트 기준으로 볼 때 ITSR은 단순한 benchmark 개선이 아니라 **contamination-robust one-class representation learning**이라는 더 넓은 방향성을 제시한 작업으로 평가할 수 있다.
