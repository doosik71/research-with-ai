# DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection

- **저자**: Hadi Hojjati, Narges Armanfard
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2106.05410

## 1. 논문 개요

이 논문은 one-class anomaly detection, 즉 정상 데이터만으로 학습하여 비정상 샘플을 검출하는 문제를 다룬다. 저자들이 집중한 핵심 문제는 기존의 deep SVDD 계열 방법, 특히 DSVDD가 보여준 장점은 유지하면서도 훈련 과정에서 발생하는 **hypersphere collapse** 문제를 해결하는 것이다. DSVDD는 입력을 latent representation으로 보낸 뒤, 그 표현들이 작은 hypersphere 안에 모이도록 학습한다. 그러나 이 구조는 잘못하면 모든 입력을 거의 같은 점으로 보내는 trivial solution으로 수렴할 수 있고, 이를 피하려고 기존 방법은 hypersphere center를 고정하고 bias를 제거하는 등 강한 제약을 둬야 했다.

이 논문은 이러한 제약이 실제 성능과 적용성을 제한한다고 본다. 그래서 autoencoder와 SVDD를 결합한 **DASVDD**를 제안한다. 이 방법은 encoder가 만든 latent space에서 정상 샘플들이 작은 hypersphere 안에 모이도록 하면서도, decoder가 원래 입력을 잘 복원하도록 동시에 요구한다. 즉, “정상 데이터의 공통 구조를 압축된 표현으로 모으되, 그 표현이 충분히 의미 있어 복원도 가능해야 한다”는 두 목적을 함께 최적화한다.

문제의 중요성은 분명하다. anomaly detection은 의료, 산업 모니터링, 침입 탐지, 음향 이상 탐지, 시계열 분석 등 매우 다양한 영역에서 쓰인다. 그런데 실제 응용에서는 비정상 데이터의 종류를 미리 충분히 수집하기 어려운 경우가 많다. 따라서 정상 데이터만으로 robust하게 학습하는 방법은 실용성이 높다. 이 논문은 특정 입력 형식에 강하게 의존하는 vision 전용 self-supervised 기법과 달리, 이미지 외에도 speech, 의료 tabular data, intrusion detection, industrial sound까지 포괄하는 **general anomaly detection** 방법을 목표로 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. 기존 AE 기반 anomaly detection은 주로 reconstruction error만으로 이상 여부를 판단한다. 하지만 autoencoder는 경우에 따라 anomaly도 잘 복원할 수 있어서 검출력이 떨어질 수 있다. 반대로 DSVDD는 latent space에서 정상 데이터를 한 점 주변에 모으는 방향으로 anomaly score를 직접 줄이지만, trivial collapse 문제가 있다. DASVDD는 이 둘의 장점을 결합한다.

구체적으로는 다음과 같은 직관에 기반한다. 정상 데이터라면 encoder가 만든 latent representation이 hypersphere center 근처에 모여야 하고, 동시에 decoder가 그 latent code로부터 원래 입력을 잘 복원해야 한다. 반면 anomaly는 이 두 조건 중 적어도 하나를 만족하기 어렵다. 즉, 정상 데이터와 다른 본질적 구조를 가지므로 latent space에서 중심에서 멀어지거나, 복원 오차가 커지거나, 혹은 둘 다 발생할 가능성이 높다.

기존 접근과의 가장 중요한 차별점은 두 가지다.

첫째, **anomaly score 자체를 훈련 목표로 직접 사용한다**는 점이다. reconstruction error만 최소화하는 것이 아니라, reconstruction error와 latent-center distance를 합친 score를 정상 데이터에서 낮추도록 학습한다. 이것은 “복원을 잘하는 표현”과 “정상 클래스의 compact한 표현”을 동시에 요구한다.

둘째, **hypersphere center $c$를 trainable parameter로 둔다**는 점이다. 기존 DSVDD는 collapse를 피하기 위해 center를 미리 고정해야 했지만, DASVDD는 reconstruction term 덕분에 trivial solution이 더 이상 최적해가 되지 않으므로 $c$를 자유롭게 학습할 수 있다고 주장한다. 이 점은 모델 자유도를 높이고, 데이터에 맞는 더 좋은 중심을 찾게 해 준다는 점에서 중요하다.

또한 저자들은 단순히 loss를 합치는 데서 끝나지 않고, network parameters와 center를 번갈아 학습하는 **iterative training strategy**를 제안한다. 이는 각 파라미터군의 수치 범위와 학습 dynamics가 다르기 때문에, 한 optimizer로 한 번에 학습하면 한쪽 loss가 지나치게 지배하는 현상을 줄이기 위한 설계다.

## 3. 상세 방법 설명

DASVDD는 크게 두 부분으로 구성된다. 하나는 autoencoder이고, 다른 하나는 latent representation 위에서 작동하는 SVDD이다. 입력 $x$가 들어오면 encoder $h(\cdot)$가 이를 latent vector $z$로 변환하고, decoder $g(\cdot)$가 $z$를 다시 복원하여 $\hat{x}$를 만든다.

즉,
$$
z = h(x; \theta_e), \qquad \hat{x} = g(z; \theta_d)
$$
이다. 여기서 $\theta_e$와 $\theta_d$는 각각 encoder와 decoder의 파라미터다.

논문은 anomaly score를 다음처럼 정의한다.
$$
S(x) = |\hat{x} - x|^2 + \gamma |z - c^*|^2
$$

이를 풀어 쓰면,
$$
S(x) = |g(h(x;\theta_e^*);\theta_d^*) - x|^2 + \gamma |h(x;\theta_e^*) - c^*|^2
$$
가 된다.

이 식의 의미는 직관적이다. 첫 번째 항은 **reconstruction error term**으로, 입력이 정상 데이터의 구조를 잘 따를수록 작아진다. 두 번째 항은 **SVDD term**으로, latent representation이 hypersphere center에 가까울수록 작아진다. $\gamma$는 두 항의 상대적 중요도를 조절하는 hyperparameter다.

정상 데이터만으로 학습할 때의 목적함수는 배치 평균 anomaly score를 최소화하는 것이다.
$$
\min_{\theta_e,\theta_d,c} \frac{1}{n} \sum_{i=1}^{n} \left( |g(h(x_i;\theta_e);\theta_d)-x_i|^2 + \gamma |h(x_i;\theta_e)-c|^2 \right)
$$

필요하면 여기에 weight decay도 추가할 수 있다고 설명한다. 핵심은 첫 번째 항이 trivial collapse를 막는 역할을 한다는 점이다. 만약 모든 가중치가 0이 되어 모든 입력이 같은 latent point로 매핑되면, DSVDD의 두 번째 항만 있는 경우에는 오히려 쉽게 loss가 작아질 수 있다. 하지만 DASVDD에서는 decoder가 정상 입력을 잘 복원해야 하므로, 그런 collapse는 reconstruction error를 크게 만들어 더 이상 좋은 해가 아니다. 저자들의 주장은 바로 이 점 때문에 bias를 0으로 강제하거나 center를 고정하지 않아도 된다는 것이다.

### 학습 절차

논문은 network parameter와 hypersphere center를 동시에 한 optimizer로 학습하는 대신, 한 epoch 안에서 두 단계를 나누는 방식을 제안한다.

먼저 각 배치의 $\kappa$ 비율 샘플로 encoder/decoder 파라미터를 학습한다. 이때 $c$는 고정한다. 그다음 나머지 $(1-\kappa)$ 비율 샘플로는 network를 고정하고 center $c$만 업데이트한다. 알고리즘 관점에서 보면, 이는 alternating optimization이다.

이 설계의 이유는 간단하다. $\theta_e, \theta_d$와 $c$는 값의 범위도 다르고, gradient dynamics도 다르다. 따라서 동시에 같은 방식으로 업데이트하면 reconstruction term과 SVDD term 중 하나가 지나치게 우세해질 수 있다. 저자들은 이런 불안정을 막기 위해 $c$는 adaptive learning rate를 갖는 optimizer로, network는 Adam으로 학습할 것을 권장한다.

또한 network가 고정되어 있을 때 center $c$의 최적값은 배치 latent representation의 평균으로 계산됨을 보인다.
$$
c = \frac{1}{|B|}\sum_{i=1}^{|B|} h(x_i;\theta_e)
$$

이 식은 매우 중요하다. 이는 center 학습이 복잡한 별도 최적화 문제라기보다, latent points의 중심을 따라가는 방식으로 이해할 수 있음을 보여준다. 다시 말해 DASVDD는 “정상 데이터 latent cloud의 중심”을 점진적으로 조정해 가는 모델이라고 볼 수 있다.

### $\gamma$ 선택 전략

이 논문에서 실용적으로 눈에 띄는 부분은 $\gamma$를 자동으로 정하는 heuristic이다. reconstruction term과 SVDD term은 데이터셋과 네트워크 구조에 따라 수치 스케일이 크게 다를 수 있으므로, 저자들은 두 항의 비율에 비례하도록 $\gamma$를 잡자고 제안한다.
$$
\gamma = \frac{1}{N} \sum_{i=1}^{N} \frac{|\hat{x}_i - x_i|^2}{|z_i - c^*|^2}
$$

문제는 훈련 전에는 최적 파라미터 $\theta_e^*, \theta_d^*, c^*$를 알 수 없다는 점이다. 그래서 실제 구현에서는 초기 random weights와 초기 center를 사용해 위 비율을 근사하고, 이를 여러 번 반복한 평균값을 최종 $\gamma$로 사용한다.
$$
\gamma = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{N} \sum_{i=1}^{N} \frac{ |g(h(x_i;\theta_e^{(0)}),\theta_d^{(0)}) - x_i|^2 }{ |h(x_i;\theta_e^{(0)}) - c^{(0)}|^2 }
$$

여기서 $T$는 서로 다른 random initialization 횟수다. 이 방법은 이론적으로 최적의 $\gamma$를 보장하는 것은 아니지만, 수작업 tuning 없이 적당한 균형값을 얻기 위한 practical heuristic으로 제안되었다.

### 구조적 해석

이 방법을 다른 anomaly detector와 비교해 보면, DASVDD는 단순 AE보다 “latent compactness”를 강하게 요구하고, DSVDD보다 “의미 있는 representation”을 강하게 요구한다. 따라서 이 모델의 성공 여부는 두 목적의 균형에 달려 있다. reconstruction이 너무 강하면 AE처럼 anomaly까지 복원할 수 있고, SVDD term이 너무 강하면 DSVDD처럼 collapse 위험이 커진다. 논문 전체는 이 균형을 잘 잡는 방법을 제안한 것으로 이해하면 된다.

## 4. 실험 및 결과

논문은 7개 공개 benchmark 데이터셋에서 DASVDD를 평가했다. 데이터 유형은 매우 다양하다. 이미지 데이터셋으로는 MNIST, CIFAR-10, Fashion-MNIST를 사용했고, 비이미지 영역으로는 ODDS Speech, PIMA, intrusion detection 데이터셋인 AWID 3, 그리고 industrial acoustic recordings인 MIMII를 사용했다. 이 구성이 중요한 이유는 논문의 핵심 주장인 “입력 타입에 구애받지 않는 general anomaly detection”을 검증하기 때문이다.

MNIST, CIFAR-10, Fashion-MNIST에서는 각 클래스 하나를 정상으로 두고 나머지 아홉 클래스를 anomaly로 두는 one-class setup을 사용했다. 이는 image anomaly detection 문헌에서 흔히 쓰이는 실험 프로토콜이다. 나머지 Speech, PIMA, AWID 3, MIMII는 원래부터 정상/비정상 라벨이 있는 데이터로 그대로 사용했다. 평가지표는 ROC-AUC이며, 각 실험을 10회 반복해 평균 AUC를 보고했다.

비교 대상은 전통적 방법 3개와 deep anomaly detection 방법 8개로 상당히 넓다. 전통적 baseline은 OCSVM, KDE, Isolation Forest이고, deep baseline은 AE, VAE, DAGMM, AnoGAN, PixCNN, AND, OCGAN, DSVDD다. 즉, 단순 복원 기반, density/generative 기반, latent autoregressive 기반, adversarial 기반, 그리고 deep SVDD 기반까지 폭넓게 비교했다.

### 주요 정량 결과

가장 중요한 결과는 Table II의 평균 성능이다.

MNIST 평균 AUC는 DASVDD가 **97.7**로 가장 높다. DSVDD는 94.8, OCGAN은 97.5, AND는 96.7이다. 즉 MNIST에서는 이미 강한 baseline들이 많지만, DASVDD가 평균적으로 최고 수준을 기록했다.

CIFAR-10 평균 AUC는 DASVDD가 **66.5**로 가장 높다. 절대 수치는 낮지만, 이 데이터셋 자체가 훨씬 어렵다. OCGAN 65.7, KDE 64.9, DSVDD 64.8보다 높다. 논문은 이 결과를 통해 CIFAR-10처럼 배경이 복잡하고 class 다양성이 큰 이미지에서도, 비록 domain-specific vision 모델만큼은 아니더라도 general method로서 경쟁력 있는 성능을 보였다고 해석한다.

Fashion-MNIST 평균 AUC는 DASVDD가 **92.6**으로 최고다. OCGAN 91.2, OCSVM 90.7, IF 90.6, DSVDD 84.8보다 높다. 특히 class 1인 Trouser에서 DASVDD는 99.0으로 매우 높다. 저자들은 이 클래스가 다른 클래스들과 시각적 유사성이 적어서 anomaly boundary가 상대적으로 더 명확하다고 해석한다.

비이미지 데이터셋에서도 DASVDD는 모두 최고 평균 성능을 기록했다. Speech에서는 **62.4**, PIMA에서는 **72.2**, AWID 3에서는 **96.3**, MIMII에서는 **67.8**을 얻었다. 절대 성능이 높지 않은 Speech에서도 chance 수준에 가까운 전통 방법들보다 낫고, AWID 3와 같은 intrusion detection에서는 매우 높은 정확도를 달성했다.

이 결과는 논문의 핵심 주장을 뒷받침한다. DASVDD는 특정 입력 종류에 특화된 augmentation이나 pretext task 없이도, 이미지와 비이미지 전반에서 안정적으로 성능을 낸다.

### 결과 해석

논문은 몇 가지 흥미로운 해석을 제시한다.

먼저 MNIST보다 Fashion-MNIST가 더 어렵다고 본다. 그 이유는 Fashion-MNIST의 intra-class variance가 더 크기 때문이다. 예를 들어 T-shirt와 Shirt, Sandal과 Sneaker처럼 서로 비슷한 클래스들이 있어 near out-of-class sample을 anomaly로 구별하기 어렵다.

또 CIFAR-10은 훨씬 더 어렵다. 이유는 객체 클래스의 다양성뿐 아니라 배경이 복잡하고, 단순 fully connected AE가 이 복잡한 시각 패턴을 비지도 방식으로 충분히 잘 구조화하기 어렵기 때문이다. 저자들은 augmentation을 사용하면 더 좋아질 수 있지만, 이는 general-purpose method라는 논문 범위를 벗어나므로 포함하지 않았다고 설명한다.

반면 AWID 3나 MIMII처럼 비이미지 데이터셋에서도 좋은 성능이 나온 점은 중요한 메시지다. 많은 최신 anomaly detection 방법이 이미지 전용 변환이나 self-supervised pretext task에 강하게 의존하는 반면, DASVDD는 그런 장치 없이도 여러 모달리티에서 일정 수준 이상 성능을 낸다.

### 고정 center와 학습 center 비교

논문은 추가 실험으로 center $c$를 고정했을 때와 학습 가능하게 두었을 때를 비교했다. 그 결과 trainable $c$를 사용한 DASVDD가 더 낫다. MNIST에서는 97.7% 대 95.2%, CIFAR-10에서는 66.5% 대 64.7%, Fashion-MNIST에서는 92.6% 대 89.7%였다. 이는 center를 고정하면 가능한 해 공간이 줄어들어 suboptimal solution으로 이어질 수 있다는 저자들의 주장을 잘 뒷받침한다.

### 하이퍼파라미터 분석

$\gamma$에 대해서는 극단적으로 큰 값을 주면 성능이 크게 떨어진다. 이는 reconstruction term의 영향이 사라지고 SVDD 항이 지나치게 지배하는 경우다. 반면 논문이 제안한 automatic selection 전략으로 얻은 값은 여러 수동 설정값과 비슷하거나 더 나은 ROC를 보였다. 저자들은 이것이 완전히 최적인 보장은 없지만, 적어도 실용적인 기본값으로 충분히 쓸 만하다고 주장한다.

$T$는 $\gamma$를 추정할 때 random initialization을 몇 번 평균낼지 정하는 값이다. 실험상 $T$를 늘리면 성능이 개선되지만, 약 $T=10$ 이후부터는 개선 폭이 줄어든다. 따라서 정확도와 계산량 사이의 균형점으로 $T=10$을 선택했다.

$\kappa$는 한 배치에서 network와 center를 각각 학습할 샘플 비율을 정한다. $\kappa=1$이면 사실상 center를 고정하는 것과 유사해지고 성능이 떨어진다. 반대로 $\kappa$가 너무 작으면 AE 학습에 쓰는 샘플이 부족해져 reconstruction 학습이 약해진다. 논문은 $\kappa=0.9$가 좋은 절충점이라고 보고한다.

### 수렴 분석과 latent size 분석

수렴 그래프를 보면 초반에는 SVDD loss가 빠르게 줄고, 이후에는 reconstruction loss가 상대적으로 더 지배적이 된다. 이는 center를 학습할 때 큰 learning rate와 decay를 사용하는 설계와 잘 맞는다. 초반에는 “중심을 빨리 잡고”, 이후에는 “복원 구조를 세밀하게 맞춘다”는 식으로 해석할 수 있다.

또 latent representation 크기를 $32$에서 $2048$까지 바꿔도 성능이 크게 출렁이지 않았다는 결과는 인상적이다. 특히 overcomplete autoencoder에서도 성능이 유지된다는 점은, latent regularization 역할을 하는 SVDD term이 단순 identity mapping으로 흐르는 것을 어느 정도 막아 준다는 해석과 연결된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 해법이 잘 맞물려 있다는 점이다. DSVDD의 대표적 약점인 hypersphere collapse를 해결하고 싶다는 문제의식이 분명하고, 그 해결책으로 reconstruction error를 anomaly score와 training objective에 직접 포함시키는 설계가 자연스럽다. 단순 regularizer 하나 더 추가한 정도가 아니라, 모델의 학습 목표 자체를 다시 정의했다는 점이 강점이다.

두 번째 강점은 **trainable center**를 허용했다는 점이다. 기존 deep SVDD류 방법에서 center 고정은 불가피한 제약처럼 받아들여졌는데, DASVDD는 이를 풀어냈고 실험으로도 성능 향상을 보여줬다. 이는 모델 표현력을 넓히는 실질적 기여다.

세 번째 강점은 **범용성**이다. 이미지뿐 아니라 speech, tabular, network, industrial audio까지 평가했고, 모든 데이터셋에서 평균적으로 최고 성능을 냈다. 특정 domain-specific transformation에 의존하지 않는다는 주장이 실험적으로도 설득력을 가진다.

네 번째 강점은 **실용적 하이퍼파라미터 설계**다. $\gamma$를 자동 추정하는 heuristic, $\kappa$를 통한 alternating update, 그리고 center optimizer를 따로 두는 설계는 실제 구현 관점에서 꽤 유용하다. 단순히 아이디어만 제시한 것이 아니라, 실제로 훈련이 안정되도록 꽤 세심하게 설계했다.

반면 한계도 분명하다.

첫째, 이 방법이 hypersphere collapse를 피한다는 설명은 매우 설득력 있지만, 엄밀한 전역 최적화 관점의 보장은 아니다. 논문도 사실상 reconstruction term이 trivial solution을 불리하게 만든다는 직관과 실험에 의존한다. 즉, 모든 조건에서 collapse가 절대 불가능하다는 강한 이론적 증명은 제시하지 않는다.

둘째, $\gamma$ 선택 heuristic은 실용적이지만 경험적이다. 논문도 스스로 인정하듯, 이 방법이 항상 최적의 $\gamma$를 찾는다고 주장하지 않는다. 따라서 데이터셋이나 아키텍처가 크게 달라지면 별도 튜닝이 필요할 수 있다.

셋째, 네트워크가 전부 fully connected AE 기반이라는 점은 특히 이미지 데이터에서 표현력 한계를 만든다. CIFAR-10 성능이 그 예다. 논문은 generality를 위해 domain-specific 구조를 피했지만, 그 대가로 복잡한 시각 패턴에 대한 feature extraction 능력은 제한된다.

넷째, anomaly score가 reconstruction과 latent-center distance의 단순 가중합이라는 구조는 이해하기 쉽지만, anomaly 유형에 따라 두 신호의 중요도가 크게 다를 수 있다. 어떤 데이터에서는 reconstruction이 더 중요하고, 다른 데이터에서는 compact latent clustering이 더 중요할 수 있다. 논문은 이 상호작용을 heuristic $\gamma$로만 다루고 있으며, 더 정교한 adaptive weighting은 다루지 않는다.

다섯째, 논문은 많은 baseline과 비교했지만, self-supervised visual anomaly detection 계열의 매우 강한 최신 image-specific 모델과는 직접 비교하지 않는다. 물론 저자들은 공정성을 위해 general-purpose 모델끼리 비교했다고 설명하지만, “최고 성능”이라는 의미는 general anomaly detection 범주 안에서 이해해야 한다.

비판적으로 보면, DASVDD는 “AE + SVDD”의 결합이 단순해 보이지만 실제로는 학습 안정성과 목적함수 설계가 핵심이다. 따라서 이 논문의 진짜 기여는 단순 결합이 아니라, **그 결합이 collapse 없이 작동하도록 만드는 objective와 optimization scheme**에 있다. 이 점을 정확히 읽어야 한다.

## 6. 결론

이 논문은 one-class anomaly detection에서 DSVDD의 장점과 AE의 장점을 결합한 **DASVDD**를 제안했다. 핵심 기여는 reconstruction error와 latent-space SVDD term을 함께 사용하는 anomaly score를 정의하고, 이를 정상 데이터에서 직접 최소화하도록 학습했다는 점이다. 이를 통해 기존 deep SVDD가 겪던 hypersphere collapse 문제를 완화하고, center를 trainable parameter로 둘 수 있게 만들었다.

또한 alternating optimization, $\gamma$ 자동 선택 heuristic, 그리고 다양한 데이터 유형에서의 광범위한 실험을 통해 이 방법이 단순한 아이디어 수준이 아니라 실제로 안정적으로 작동하는 anomaly detector임을 보여주었다. 특히 이미지 외의 speech, intrusion detection, industrial sound, medical/tabular data에서도 일관된 성능을 보였다는 점은 실용적 가치가 크다.

실제 적용 측면에서 이 연구는 “특정 모달리티 전용 기법이 아닌 범용 anomaly detector”를 설계할 때 좋은 출발점이 될 수 있다. 향후 연구에서는 convolutional encoder나 transformer encoder 같은 더 강한 representation learner와 결합하거나, $\gamma$를 자동으로 더 정교하게 조절하는 방법, 혹은 sample-wise adaptive scoring을 도입하는 방향으로 확장할 여지가 크다. 논문 자체도 이 부분을 후속 연구 과제로 남겨 두고 있다.

전체적으로 보면, DASVDD는 새로운 anomaly detection 원리를 제시했다기보다, 기존 AE와 SVDD의 결합을 매우 설득력 있게 재설계하여 **범용성, 안정성, 성능** 사이의 균형을 잘 잡은 논문이라고 평가할 수 있다. 특히 deep SVDD류 방법을 실제로 더 쓰기 쉽게 만들었다는 점에서 학술적·실용적 의미가 있다.
