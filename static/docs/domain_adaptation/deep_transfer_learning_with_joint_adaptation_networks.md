# Deep Transfer Learning with Joint Adaptation Networks

* **저자**: Mingsheng Long, Han Zhu, Jianmin Wang, Michael I. Jordan
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1605.06636](https://arxiv.org/abs/1605.06636)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 다룬다. 즉, 라벨이 있는 source domain 데이터와 라벨이 없는 target domain 데이터가 있을 때, 두 도메인의 분포 차이 때문에 source에서 학습한 분류기가 target에서 성능이 떨어지는 문제를 해결하고자 한다. 기존의 deep transfer learning 방법들은 주로 feature의 **marginal distribution**만 맞추는 데 집중했지만, 이 논문은 실제로 더 중요한 차이가 **입력 특징과 출력 라벨의 joint distribution** $P(X,Y)$ 수준에서 발생할 수 있다고 본다.

논문의 핵심 목표는 deep network 내부의 여러 task-specific layer에서 나타나는 activation들을 이용해 source와 target의 **joint distribution**을 직접 가깝게 만드는 것이다. 저자들은 이를 위해 **Joint Adaptation Networks (JAN)** 를 제안하고, 여러 층의 activation에 대해 joint distribution discrepancy를 측정하는 **Joint Maximum Mean Discrepancy (JMMD)** 를 도입한다. 이 방식은 단순히 각 층을 독립적으로 맞추는 것이 아니라, 여러 층 사이의 상호작용을 함께 반영한다는 점에서 기존 방법보다 더 강력한 적응 능력을 기대할 수 있다.

이 문제가 중요한 이유는 실제 응용에서 도메인 차이가 단순한 feature shift만으로 설명되지 않는 경우가 많기 때문이다. 예를 들어 source와 target에서 같은 물체라도 촬영 환경, 배경, 클래스 빈도, 라벨 결정 경계가 함께 달라질 수 있다. 이런 경우 feature 분포만 맞추면 충분하지 않고, feature와 classifier output이 함께 만들어내는 joint structure까지 적응해야 한다. 이 논문은 바로 이 지점을 겨냥해 deep domain adaptation의 한계를 확장하려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. deep network의 상위 층으로 갈수록 representation은 점점 더 task-specific해지고, 따라서 도메인 간 차이도 이 상위 층들에 남아 있게 된다. 저자들은 AlexNet 기준으로 $fc6$, $fc7$, $fc8$, ResNet 기준으로 $pool5$와 $fc$ 같은 상위 층들을 **domain-specific layers**로 보고, 이 층들의 activation joint distribution을 source와 target 사이에서 정렬해야 한다고 주장한다.

기존 접근과의 차별점은 다음과 같다. DAN 같은 방법은 여러 층에 대해 각각 MMD를 걸어 각 층의 **marginal distribution**을 독립적으로 맞춘다. 그러나 JAN은 여러 층의 activation을 하나의 joint random variable처럼 보고, 이들의 **결합 분포 전체**를 맞춘다. 즉, 어떤 feature layer의 분포와 classifier layer의 출력 분포를 따로따로 보는 것이 아니라, 이들이 함께 만들어내는 구조를 본다. 논문은 이것이 실제로 $P(X,Y)$의 차이를 더 잘 반영한다고 본다.

또 하나의 핵심은, 단순한 kernel-based discrepancy만 쓰는 것이 아니라 이를 더 강하게 만들기 위해 **adversarial training**을 결합한 **JAN-A**를 제안했다는 점이다. 기본 JAN은 비모수적 kernel embedding 기반이라 안정적이고 간단하지만, 복잡한 고차원 분포 차이를 구분하는 능력이 부족할 수 있다. 이를 보완하기 위해 저자들은 JMMD를 계산하기 전 여러 fully-connected layer를 추가하고, 이 부분은 source와 target을 더 잘 구분하도록 **maximize**하며, feature extractor는 이를 다시 줄이도록 **minimize**하는 min-max 학습을 수행한다.

## 3. 상세 방법 설명

논문의 방법은 크게 세 단계로 이해할 수 있다. 첫째, ImageNet으로 사전학습된 CNN을 source와 target 데이터에 동시에 통과시킨다. 둘째, 상위 task-specific layers의 activation들을 수집한다. 셋째, source label에 대한 분류 loss와 함께 source/target activation joint distribution 차이를 나타내는 JMMD를 최소화한다.

### 3.1 Hilbert space embedding과 MMD의 준비 개념

논문은 먼저 kernel mean embedding을 설명한다. 어떤 분포 $P(X)$를 RKHS 상의 한 점으로 표현하는 기본 식은 다음과 같다.

$$
\mu_X(P) \triangleq \mathbb{E}_X[\phi(X)] = \int_\Omega \phi(x), dP(x)
$$

여기서 $\phi(x)$는 kernel이 암묵적으로 정의하는 feature map이다. 직관적으로는, 분포 전체를 평균 feature vector 하나로 표현하는 것이다. 실제 분포를 모르더라도 샘플 ${x_i}_{i=1}^n$이 있으면 경험적 추정량은 다음과 같다.

$$
\widehat{\mu}_X = \frac{1}{n}\sum_{i=1}^{n}\phi(x_i)
$$

MMD는 두 분포 $P$와 $Q$의 mean embedding 거리로 정의된다. 논문이 제시한 식은 다음과 같다.

$$
D_{\mathcal H}(P,Q) \triangleq \sup_{f\in\mathcal H}
\left(\mathbb{E}_{X^s}[f(X^s)] - \mathbb{E}_{X^t}[f(X^t)]\right)
$$

그리고 universal RKHS에서는 이것이 embedding distance와 동치가 된다.

$$
D_{\mathcal H}(P,Q)=|\mu_{X^s}(P)-\mu_{X^t}(Q)|_{\mathcal H}^2
$$

즉, 두 분포가 같으면 MMD는 0이 된다.

### 3.2 Joint distribution embedding

이 논문의 진짜 핵심은 single variable 분포가 아니라 여러 변수의 **joint distribution**을 embedding하는 것이다. 여러 층의 activation을 각각 $Z^1,\dots,Z^m$이라고 하면, joint distribution의 embedding은 tensor product Hilbert space에서 다음처럼 정의된다.

$$
\mathcal C_{X^{1:m}}(P)
\triangleq
\mathbb E_{X^{1:m}}
\left[
\bigotimes_{\ell=1}^{m}\phi^\ell(X^\ell)
\right]
$$

이 식은 여러 변수의 feature map을 tensor product로 묶은 뒤 평균을 취한 것이다. 쉽게 말하면, 각 층을 독립적으로 보지 않고 여러 층이 동시에 어떤 패턴을 이루는지를 표현하는 방식이다. 이 구조 덕분에 "feature layer에서는 이 패턴이 나오고 classifier layer에서는 저 출력이 나온다"는 식의 결합 관계까지 반영할 수 있다.

### 3.3 Joint Maximum Mean Discrepancy (JMMD)

source와 target의 domain-specific layers 집합을 $\mathcal L$라고 할 때, 저자들은 이들 activation joint distribution의 차이를 다음처럼 정의한다.

$$
D_{\mathcal L}(P,Q) \triangleq \left| \mathcal C_{Z^{s,1:|\mathcal L|}}(P) - \mathcal C_{Z^{t,1:|\mathcal L|}}(Q) \right|_{\otimes_{\ell=1}^{|\mathcal L|}\mathcal H^\ell}^{2}
$$

이것이 **JMMD**이다. MMD가 한 층의 분포 차이를 보는 것이라면, JMMD는 여러 층 activation들의 joint distribution 차이를 보는 것이다.

논문이 제시한 경험적 추정량은 다음과 같다.

$$
\begin{aligned}
\widehat D_{\mathcal L}(P,Q) = & \; \frac{1}{n_s^2}\sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{\ell\in \mathcal L} k^\ell(z_i^{s\ell},z_j^{s\ell}) \\
& + \frac{1}{n_t^2}\sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{\ell\in \mathcal L} k^\ell(z_i^{t\ell},z_j^{t\ell}) \\
& - \frac{2}{n_s n_t}\sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{\ell\in \mathcal L} k^\ell(z_i^{s\ell},z_j^{t\ell})
\end{aligned}
$$

중요한 점은 각 층 커널의 **곱**이 등장한다는 것이다. 이 때문에 한 층의 similarity는 다른 층의 상태에 의해 가중되며, 결과적으로 여러 층 사이의 상호작용이 반영된다. 논문은 이것이 기존 MMD 방식과의 본질적 차이라고 설명한다.

### 3.4 최종 학습 목표

기본 JAN은 source classification loss와 JMMD regularization을 함께 최소화한다.

$$
\min_f \frac{1}{n_s}\sum_{i=1}^{n_s} J(f(x_i^s), y_i^s) + \lambda \widehat D_{\mathcal L}(P,Q)
$$

여기서 $J(\cdot,\cdot)$는 cross-entropy loss이고, $\lambda > 0$는 분류 성능과 도메인 정렬 사이의 trade-off를 조절하는 하이퍼파라미터이다.

이 식의 의미는 간단하다. source에서 분류를 잘 하면서도, network 상위 층의 joint distribution이 source와 target에서 최대한 비슷해지도록 representation을 학습한다는 것이다. 결과적으로 classifier는 source에 맞춰 학습되지만, feature/classifier activation 구조 자체가 target에도 더 잘 맞도록 바뀐다.

### 3.5 Linear-time JMMD 추정

위 식은 쌍별 비교가 들어가므로 quadratic complexity를 가진다. 이는 mini-batch SGD에 비효율적이므로 논문은 linear-time unbiased estimate를 도입한다. 논문에 제시된 형태는 다음과 같다.

$$
\begin{aligned}
\widehat D_{\mathcal L}(P,Q) = & \; \frac{2}{n}\sum_{i=1}^{n/2} \left( \prod_{\ell\in \mathcal L} k^\ell(z_{2i-1}^{s\ell}, z_{2i}^{s\ell}) + \prod_{\ell\in \mathcal L} k^\ell(z_{2i-1}^{t\ell}, z_{2i}^{t\ell}) \right) \\
& - \frac{2}{n}\sum_{i=1}^{n/2} \left( \prod_{\ell\in \mathcal L} k^\ell(z_{2i-1}^{s\ell}, z_{2i}^{t\ell}) + \prod_{\ell\in \mathcal L} k^\ell(z_{2i-1}^{t\ell}, z_{2i}^{s\ell}) \right)
\end{aligned}
$$

이 식은 mini-batch 내에서 source/target 샘플을 짝지어 계산하므로 계산량이 선형이 된다. 덕분에 back-propagation으로 자연스럽게 최적화할 수 있다. 논문은 source와 target에서 같은 수의 샘플을 mini-batch에 넣어 bias를 줄인다고 설명한다.

### 3.6 Adversarial Joint Adaptation Network (JAN-A)

저자들은 kernel-based MMD/JMMD가 안정적이라는 장점은 있지만, 고차원 복잡한 분포 차이를 충분히 잘 구분하지 못할 수 있고, 어떤 kernel에서는 gradient가 약해질 수 있다고 지적한다. 이를 해결하기 위해 JMMD 앞에 여러 fully-connected layer를 추가하고, 이 부분의 파라미터 $\theta$는 source/target 차이를 더 크게 만들도록 학습한다. 반면 feature extractor $f$는 다시 그 차이를 줄이도록 학습한다.

최종 목표는 다음 min-max 문제가 된다.

$$
\min_f \max_\theta \frac{1}{n_s}\sum_{i=1}^{n_s} J(f(x_i^s), y_i^s) + \lambda \widehat D_{\mathcal L}(P,Q;\theta)
$$

직관적으로 보면, $\theta$는 “source와 target을 가장 잘 구분하는 joint discrepancy 측정기”를 만들고, $f$는 그 측정기조차 구분하지 못하도록 representation을 바꾸는 역할을 한다. 논문은 이것이 RevGrad의 domain discriminator와 유사한 adversarial idea를 공유하지만, 로지스틱 회귀 기반 domain classifier 대신 **JMMD 자체를 adversary**로 사용한다는 점이 다르다고 설명한다.

### 3.7 네트워크 구성과 학습 절차

AlexNet 기반 JAN에서는 $\mathcal L={fc6, fc7, fc8}$, ResNet 기반 JAN에서는 $\mathcal L={pool5, fc}$를 사용한다. lower layers는 비교적 일반적이라 transferable하다고 보고 별도 adaptation을 하지 않는다. 모든 convolution/pooling layer는 fine-tuning하며, classifier layer는 새로 학습하므로 learning rate를 다른 층보다 10배 크게 둔다.

학습은 mini-batch SGD with momentum 0.9로 수행한다. learning rate는 training progress $p$에 따라 다음과 같이 감소시킨다.

$$
\eta_p = \frac{\eta_0}{(1+\alpha p)^\beta}
$$

논문에서는 $\eta_0=0.01$, $\alpha=10$, $\beta=0.75$를 사용했다. 또한 adaptation factor는 고정하지 않고 점진적으로 증가시킨다.

$$
\lambda_p=\frac{2}{1+\exp(-\gamma p)}-1
$$

여기서 $\gamma=10$이다. 저자들은 이 점진적 스케줄이 초기 noisy activation으로 인한 불안정을 줄이고, hyperparameter sensitivity를 완화한다고 설명한다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 두 개의 대표적인 domain adaptation benchmark에서 실험한다.

첫째는 **Office-31**이다. 총 4,652장 이미지와 31개 클래스가 있으며, Amazon (A), Webcam (W), DSLR (D)의 세 도메인으로 구성된다. 논문은 여섯 가지 전이 과제 $A\to W$, $D\to W$, $W\to D$, $A\to D$, $D\to A$, $W\to A$를 평가했다.

둘째는 **ImageCLEF-DA**이다. Caltech-256 (C), ImageNet ILSVRC 2012 (I), Pascal VOC 2012 (P)에서 공통 12개 클래스를 뽑아 각 도메인 600장으로 맞춘 데이터셋이다. 여섯 가지 전이 과제 $I\to P$, $P\to I$, $I\to C$, $C\to I$, $C\to P$, $P\to C$를 사용했다. 이 데이터셋은 각 도메인 크기가 같아 Office-31보다 더 균형 잡힌 실험 환경을 제공한다.

비교 대상은 TCA, GFK, AlexNet/ResNet baseline, DDC, DAN, RevGrad, RTN 등이다. 평가 지표는 target domain classification accuracy이며, 세 번의 random experiment 평균과 standard error를 보고한다. 모델 선택은 transfer cross-validation으로 수행했다. MMD 계열 방법과 JAN은 Gaussian kernel을 쓰고 bandwidth는 training data의 median pairwise squared distance로 설정했다.

### 4.2 Office-31 결과

Office-31에서 AlexNet 기반 평균 정확도는 다음과 같다.

* AlexNet baseline: 70.1
* DAN: 72.9
* RTN: 73.7
* RevGrad: 74.3
* JAN: 76.0
* JAN-A: 76.3

즉, AlexNet 기반에서는 JAN이 기존 강력한 baseline보다 평균 약 1.7%p 이상 높고, JAN-A는 JAN보다도 조금 더 높다.

특히 어려운 전이 과제에서 개선이 두드러진다. 예를 들어 $D\to A$에서 AlexNet 기반 성능은 RevGrad 53.4, JAN 58.3, JAN-A 57.5이고, $W\to A$에서는 RevGrad 51.2, JAN 55.0, JAN-A 56.3이다. 논문은 이런 과제가 source와 target 차이가 크고 source 크기가 더 작기 때문에 더 어렵다고 설명한다. 반면 쉬운 과제인 $D\to W$, $W\to D$에서는 이미 baseline들이 매우 높아서 성능 차이가 작다.

ResNet 기반 평균 정확도는 다음과 같다.

* ResNet baseline: 76.1
* DAN: 80.4
* RTN: 81.6
* RevGrad: 82.2
* JAN: 84.3
* JAN-A: 84.6

여기서도 JAN 계열이 최고 성능을 기록한다. 예를 들어 $A\to W$에서 RevGrad 82.0, JAN 85.4, JAN-A 86.0이고, $W\to A$에서 RevGrad 67.4, JAN 70.0, JAN-A 70.7이다. 논문은 이를 통해 joint distribution adaptation이 매우 깊은 backbone에서도 여전히 필요하다고 주장한다.

### 4.3 ImageCLEF-DA 결과

ImageCLEF-DA에서는 개선 폭이 Office-31보다 다소 작지만 여전히 JAN이 가장 좋다.

AlexNet 기반 평균 정확도는 다음과 같다.

* AlexNet baseline: 73.9
* DAN: 76.9
* RTN: 77.9
* JAN: 79.3

ResNet 기반 평균 정확도는 다음과 같다.

* ResNet baseline: 80.7
* DAN: 82.5
* RTN: 83.9
* JAN: 85.8

예를 들어 ResNet 기반에서 $P\to I$는 RTN 85.8, JAN 88.0이고, $C\to I$는 RTN 85.9, JAN 89.5이다. 논문은 ImageCLEF-DA에서 도메인 크기가 균형 잡혀 있어 Office-31보다 개선폭이 작았으며, 이를 통해 domain size difference도 분포 shift에 영향을 줄 수 있다고 해석한다.

### 4.4 정성 분석과 추가 분석

논문은 t-SNE 시각화도 제시한다. DAN과 JAN으로 학습된 ResNet activation을 비교했을 때, JAN에서 target category들이 source classifier 기준으로 더 또렷하게 분리되어 보인다고 설명한다. 이는 multilayer joint distribution adaptation이 실제 representation geometry를 더 잘 정리해 준다는 정성적 근거다.

또한 **$\mathcal A$-distance**를 사용해 도메인 간 분포 차이를 측정한다. JAN feature에서 $\mathcal A$-distance가 CNN이나 DAN보다 더 작게 나타났으며, 이는 JAN이 도메인 gap을 더 효과적으로 줄였음을 의미한다. 다만 논문은 $\mathcal A$-distance가 joint distribution discrepancy 자체를 직접 재지는 못한다고 인정한다.

이를 보완하기 위해 저자들은 JMMD 값도 직접 측정한다. $fc7$ feature와 $fc8$ label-related output을 이용해 계산한 결과, JAN activation의 JMMD가 CNN과 DAN보다 더 작았다. 이는 JAN이 실제로 joint distribution shift를 줄인다는 논문의 핵심 주장과 직접 연결된다.

파라미터 민감도 분석에서는 $\lambda \in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1}$ 범위를 실험했고, 정확도가 종 모양 곡선처럼 처음 증가하다가 다시 감소했다. 이는 classification loss와 adaptation loss 사이의 균형이 중요함을 보여준다. 너무 작은 $\lambda$는 adaptation이 부족하고, 너무 큰 $\lambda$는 source discriminative learning을 해칠 수 있다는 뜻으로 읽힌다.

수렴 분석에서는 JAN이 비모수적 JMMD 덕분에 빠르게 수렴하고, JAN-A는 RevGrad와 비슷한 속도를 유지하면서도 더 높은 정확도를 달성한다고 보고한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 deep domain adaptation에서 **joint distribution adaptation**을 전면적으로 제안했다는 점이다. 기존 연구가 feature marginal alignment에 머물렀던 반면, JAN은 “상위 layer들의 activation joint structure”를 맞추어야 한다는 명확한 문제 설정을 제시한다. 이는 단순한 기법 추가가 아니라, 도메인 차이를 바라보는 관점을 한 단계 확장한 것이다.

또 다른 강점은 수학적 설계와 실제 구현이 잘 연결된다는 점이다. joint distribution embedding이라는 이론적 도구를 가져오되, linear-time estimator를 도입해 mini-batch SGD와 back-propagation으로 바로 학습할 수 있게 만들었다. 이 때문에 이론적으로는 joint distribution을 다루면서도, 실용적으로는 일반적인 deep learning 파이프라인에 통합 가능하다.

실험적으로도 설득력이 있다. Office-31과 ImageCLEF-DA, AlexNet과 ResNet, 그리고 다양한 strong baseline을 모두 비교했고, 대부분의 transfer task에서 SOTA를 기록했다. 특히 어려운 task에서 개선폭이 크다는 점은 논문의 주장을 잘 뒷받침한다.

다만 한계도 분명하다. 첫째, 이 방법은 결국 **상위 layer activation joint distribution이 원래의 $P(X,Y)$를 잘 대리한다**는 가정 위에 서 있다. 논문은 이를 직관적으로 설명하지만, 이 surrogate relation이 언제 정확하고 언제 불완전한지에 대한 이론적 분석은 깊지 않다.

둘째, target domain이 unlabeled이므로 conditional shift를 직접 관측할 수 없다. JAN은 이를 joint activation alignment로 우회하지만, 이것이 모든 종류의 label shift나 class prior shift를 충분히 다루는지는 논문만으로는 확정하기 어렵다. 특히 심한 label distribution mismatch 상황에서 어떤 동작을 보일지는 명확히 분석되지 않았다.

셋째, kernel choice와 layer selection에 여전히 의존한다. 논문은 Gaussian kernel과 특정 상위 층 집합을 사용했지만, 어떤 kernel과 어떤 layer 조합이 최적인지는 문제별로 달라질 수 있다. JAN-A는 richer function class를 통해 이를 완화하려 하지만, adversarial optimization의 안정성 문제를 완전히 제거했다고 보기는 어렵다. 논문은 수렴이 양호하다고 보고하지만, 더 큰 규모 데이터나 더 복잡한 task에서의 안정성은 추가 검증이 필요하다.

넷째, 실험은 당시 표준 벤치마크에서는 충분히 강하지만, 현대적 대규모 비전 적응 문제나 open-set / partial / universal domain adaptation 같은 더 어려운 설정까지 다루지는 않는다. 따라서 논문의 공헌은 매우 중요하지만, 적용 범위는 실험적으로는 비교적 전통적 unsupervised domain adaptation 설정에 제한되어 있다.

## 6. 결론

이 논문은 deep transfer learning에서 중요한 전환점을 제시한다. 핵심 기여는 source와 target의 **marginal feature distribution**이 아니라, 여러 task-specific layer activation의 **joint distribution**을 정렬해야 한다는 문제의식을 명확히 제시하고, 이를 실제로 학습 가능한 형태인 **JMMD**로 공식화한 데 있다. 또한 linear-time 추정과 back-propagation 가능한 구조를 통해 end-to-end 학습을 가능하게 했고, adversarial extension인 JAN-A까지 제안해 분포 구분력을 더 강화했다.

실제 성능 면에서도 JAN과 JAN-A는 Office-31과 ImageCLEF-DA에서 강력한 baseline을 지속적으로 넘어섰다. 특히 어려운 도메인 전이에서 성능 향상이 크다는 점은, joint distribution adaptation이 단순한 이론적 제안이 아니라 실제 transferability 향상에 실질적으로 기여함을 보여준다.

향후 연구 관점에서 보면, 이 논문은 이후의 domain adaptation 연구에 두 가지 중요한 방향을 남긴다. 하나는 deep network 내부의 여러 표현 수준을 **독립이 아니라 상호작용 구조**로 바라보는 관점이고, 다른 하나는 kernel discrepancy와 adversarial learning을 결합하는 방향이다. 실제 응용에서도 source와 target 사이의 차이가 복합적일 때, 단순 feature alignment보다 더 구조적인 alignment가 필요하다는 점을 일깨워 준다는 의미가 있다. 따라서 JAN은 deep unsupervised domain adaptation의 중요한 기반 논문으로 볼 수 있다.
