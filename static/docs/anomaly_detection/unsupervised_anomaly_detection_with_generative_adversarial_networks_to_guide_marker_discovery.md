# Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery

- **저자**: Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs
- **발표연도**: 2017
- **arXiv**: http://arxiv.org/abs/1703.05921v1

## 1. 논문 개요

이 논문은 의료영상에서 **정상 데이터만으로 정상 해부학적 변이를 학습한 뒤**, 학습된 정상 분포에 잘 맞지 않는 샘플을 이상(anomaly)으로 탐지하는 방법을 제안한다. 저자들은 특히 안과 영역의 **retina SD-OCT(spectral-domain optical coherence tomography)** 영상에서, 질환 마커를 모두 사전에 정의하고 주석을 붙이는 기존 supervised 접근의 한계를 문제로 제기한다. 실제 임상에서는 이미 알려진 병변만 찾는 것보다, 데이터 안에 아직 명시적으로 정의되지 않은 이상 패턴을 발견하는 능력이 중요하다.

이 논문의 핵심 문제는 다음과 같이 정리할 수 있다. 첫째, 병변 주석을 대규모로 수집하는 비용이 매우 크다. 둘째, supervised 학습은 애초에 정의된 병변 vocabulary에 갇히기 쉽다. 셋째, 새로운 형태의 병리적 변화나 미세한 이상은 사전 라벨 없이 탐색적으로 찾기 어렵다. 이를 해결하기 위해 저자들은 **건강한 영상 패치만으로 정상 manifold를 학습**하고, 테스트 영상이 이 manifold 위에 얼마나 잘 투영될 수 있는지를 기준으로 이상 여부를 판단한다.

논문의 의의는 단순히 “정상/비정상 이진 분류기”를 만든 데 있지 않다. 저자들은 생성모델을 이용해 “정상이라면 어떻게 보였어야 하는가”를 복원하고, 원본 입력과 생성된 정상 근사 영상 사이의 차이를 이용해 **이상 점수(anomaly score)**와 **이상 위치(local anomalous region)**를 함께 얻고자 했다. 즉, 단순 판별을 넘어 임상적으로 해석 가능한 형태의 이상 탐지를 시도한 점이 중요하다.

## 2. 핵심 아이디어

중심 아이디어는 **GAN으로 정상 영상의 분포를 학습한 다음, 테스트 이미지를 latent space로 역매핑하여 가장 유사한 정상 샘플을 찾고, 그 과정에서 얻는 오차를 이상의 증거로 사용하는 것**이다. 논문은 이 전체 프레임워크를 **AnoGAN**이라고 부른다.

기존 anomaly detection과 비교했을 때 이 논문의 차별점은 두 가지다. 첫째, autoencoder나 one-class SVM 기반 방법보다 **GAN의 생성력과 discriminator의 표현력**을 함께 활용해 정상 해부학적 다양성을 더 풍부하게 모델링하려 했다. 둘째, 단순히 generator만 학습하는 데서 끝나지 않고, **새로운 입력 영상을 latent space의 한 점으로 찾는 mapping 절차 자체를 별도로 설계**했다. GAN은 본래 $z \rightarrow x$ 방향의 생성만 제공하고, $x \rightarrow z$ 역함수는 자동으로 주어지지 않기 때문에 이 문제가 핵심이다.

특히 저자들은 기존 image inpainting 연구를 참고하면서도, 단순히 discriminator를 속이는 방향이 아니라 **discriminator의 중간 feature representation을 이용한 feature matching 기반 discrimination loss**를 제안했다. 이로써 테스트 이미지를 latent space에 투영할 때, scalar decision 하나에 의존하지 않고 더 풍부한 정상성(normality) 정보를 반영하도록 만들었다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

전체 파이프라인은 크게 세 단계로 구성된다.

먼저 전처리 단계에서 retina OCT 볼륨으로부터 retinal area를 추출하고, 층 구조를 기준으로 flattening을 수행한 뒤, 2D patch를 잘라 intensity normalization을 적용한다. 학습에는 **정상(healthy) OCT 패치만 사용**된다.

그다음 GAN 학습 단계에서는 generator $G$와 discriminator $D$를 adversarial하게 학습한다. generator는 latent vector $z$를 입력받아 정상처럼 보이는 영상 $G(z)$를 생성하고, discriminator는 입력 영상이 실제 정상 패치인지 generator가 만든 합성 패치인지를 구분한다. 이를 통해 generator는 정상 영상 manifold를, discriminator는 정상성에 관한 feature representation을 학습하게 된다.

마지막 테스트 단계에서는 임의의 query image $x$가 주어졌을 때, 이와 가장 비슷한 정상 샘플을 생성하는 latent code $z$를 iterative optimization으로 찾는다. 이렇게 얻은 $G(z)$는 입력 $x$의 “정상 근사본”처럼 작동한다. 따라서 $x$와 $G(z)$의 차이가 크면 클수록 이상일 가능성이 높다고 본다.

### 3.2 GAN 기반 정상 manifold 학습

논문은 DCGAN 계열 구조를 사용한다. generator는 strided convolution 계열의 decoder 구조이며, discriminator는 CNN으로 구성된다. 학습 목표는 표준 GAN의 minimax 게임이다.

$$
\min_G \max_D \, V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

여기서 discriminator는 실제 정상 패치에는 높은 확률을, 생성 패치에는 낮은 확률을 부여하도록 학습된다. 반대로 generator는 discriminator가 생성 영상을 실제처럼 판단하도록 유도된다. 저자들의 목적은 class label을 학습하는 것이 아니라, **정상 anatomy의 다양성을 포괄하는 생성 모델**을 확보하는 데 있다.

논문은 GAN이 정상 데이터의 manifold를 충분히 잘 근사할 경우, 정상 입력은 그 manifold 위의 적절한 점으로 설명될 수 있지만 비정상 입력은 그렇지 못하다고 본다. 따라서 이상 탐지의 핵심은 “입력이 정상 manifold 위에서 얼마나 잘 재구성되는가”가 된다.

### 3.3 새로운 이미지를 latent space로 매핑하는 절차

GAN 학습이 끝난 뒤에도 바로 anomaly detection이 가능한 것은 아니다. generator는 $z \rightarrow x$만 제공하고 $x \rightarrow z$는 제공하지 않기 때문이다. 그래서 query image $x$에 대해, 가장 유사한 정상 생성 영상 $G(z)$를 만들 수 있는 latent code $z$를 iterative backpropagation으로 찾는다.

초기 $z$는 latent prior에서 랜덤 샘플링한 뒤, loss를 줄이는 방향으로 반복 업데이트한다. 이때 사용되는 loss는 두 구성요소의 가중합이다.

첫 번째는 **residual loss**이다. 이는 입력 영상과 생성 영상의 픽셀 수준 차이를 나타낸다.

$$
\mathcal{L}_R(z) = \sum |x - G(z)|
$$

이 값이 작다는 것은 생성된 정상 근사 영상이 입력과 시각적으로 유사하다는 뜻이다. 정상 영상이라면 이 값이 상대적으로 작을 가능성이 높다.

두 번째는 **discrimination loss**이다. 기존 연구는 discriminator의 최종 scalar output을 활용했지만, 이 논문은 그보다 더 풍부한 정보를 담는 **중간 layer feature**를 사용한다. 논문은 discriminator의 중간 표현을 $f(\cdot)$라고 둘 때, feature matching 기반 discrimination loss를 다음과 같이 정의한다.

$$
\mathcal{L}_D(z) = \sum |f(x) - f(G(z))|
$$

이 정의의 직관은 명확하다. 입력 영상과 생성 영상이 discriminator의 feature space에서도 비슷해야, 단순히 겉모양만 비슷한 것이 아니라 **학습된 정상 분포의 통계적/표현적 구조**에도 잘 부합한다고 볼 수 있다.

최종적으로 latent mapping을 위한 전체 loss는 다음과 같다.

$$
\mathcal{L}(z) = (1-\lambda)\mathcal{L}_R(z) + \lambda \mathcal{L}_D(z)
$$

논문에서는 $\lambda = 0.1$을 사용했다. 중요한 점은 이 최적화 과정에서 **업데이트되는 것은 오직 $z$뿐**이며, 학습이 끝난 $G$와 $D$의 파라미터는 고정된다는 것이다.

### 3.4 이상 점수와 이상 영역 탐지

이상 점수는 latent mapping 마지막 단계에서의 residual score와 discrimination score를 조합해 계산된다. 논문은 이를 다음과 같이 정의한다.

$$
A(x) = (1-\lambda) R(x) + \lambda D(x)
$$

여기서 $R(x)$와 $D(x)$는 각각 최종 iteration에서의 residual loss와 discrimination loss에 대응한다. 값이 클수록 입력이 정상 데이터 분포에 잘 맞지 않는다고 해석한다.

또한 저자들은 픽셀 수준 residual image를 이용해 **이상 위치를 시각화**한다. 즉, 단순히 patch-level anomaly score만 제공하는 것이 아니라, 원본 영상과 정상 근사 영상의 차이를 overlay하여 retinal fluid 같은 병변이 어느 위치에서 두드러지는지도 보여준다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

실험은 retina의 고해상도 **SD-OCT volume** 데이터에서 수행되었다. 학습에는 retinal fluid가 없는 **건강한 270개 OCT volume**에서 추출한 2D 패치를 사용했고, 전처리 후 총 **1,000,000개의 training patch**를 구성했다. 테스트는 별도의 **정상 10개 case와 병리 10개 case**에서 추출한 총 **8192개 patch**로 수행되었다.

병리 사례에는 retinal fluid가 포함되어 있었으며, 전문가가 만든 voxel-level annotation이 존재했다. 이 주석은 오직 평가용으로만 쓰였고, 학습이나 latent mapping에는 사용되지 않았다. 이미지 단위 평가는 “patch 안에 retinal fluid annotation이 한 픽셀이라도 있으면 positive”라는 기준으로 수행되었다.

구현 측면에서는 Radford 등의 DCGAN 설계를 기반으로 generator에 4개의 fractionally-strided convolution layer, discriminator에 4개의 convolution layer를 사용했고, 학습은 20 epoch 동안 Adam으로 수행되었다. 테스트 시 query image의 latent mapping을 위해 **500회의 backpropagation step**을 사용했다.

### 4.2 평가 항목

논문은 세 가지를 본다. 첫째, 모델이 정상적인 medical patch를 실제처럼 생성할 수 있는지에 대한 정성 평가다. 둘째, anomaly score, residual score, discrimination score 각각을 이용했을 때의 **image-level anomaly detection 성능**이다. 셋째, residual image가 실제 병변 위치와 얼마나 잘 맞는지, 그리고 기존에 주석되지 않은 추가 이상까지 탐지하는지를 확인한다.

또한 비교 실험으로는 adversarial convolutional autoencoder(aCAE) 기반 대안과, 기존 inpainting 계열의 reference discrimination score를 사용하는 방식, 그리고 discriminator output만 직접 사용하는 방식이 포함되었다.

### 4.3 주요 결과

정성적으로는, 정상 입력 패치에 대해서는 AnoGAN이 생성한 $G(z)$가 원본과 매우 유사한 형태를 보였고, 병리 패치에 대해서는 intensity와 texture 측면에서 뚜렷한 차이를 드러냈다. 이는 모델이 정상 anatomy manifold를 어느 정도 유의미하게 학습했음을 시사한다.

정량적으로는 AnoGAN이 가장 높은 성능을 보였다. 표 1에 따르면 image-level anomaly detection에서 **AnoGAN의 AUC는 0.89**였고, precision은 **0.8834**, recall은 **0.7277**, sensitivity는 **0.7279**, specificity는 **0.8928**이었다. 비교 대상으로 제시된 aCAE는 **AUC 0.73**, reference score 방식은 **AUC 0.72**, discriminator output 직접 사용 방식은 **AUC 0.88**이었다. 즉, 제안한 방법은 discriminator-only 방식과 비슷한 수준까지 강력했으며, 전체적으로 가장 높은 AUC를 기록했다.

논문이 특별히 강조하는 부분은 **제안한 feature matching 기반 discrimination loss가 reference discrimination loss보다 우수했다**는 점이다. 저자들은 이는 단순 scalar decision보다 intermediate representation이 정상성 판별에 더 유용한 정보를 제공하기 때문이라고 해석한다.

또 하나 중요한 결과는 residual overlay가 retinal fluid뿐 아니라 **hyperreflective foci(HRF)** 같은 추가 병변도 탐지했다는 것이다. 이 부분은 논문의 핵심 주장과 연결된다. 즉, 정답 라벨에 포함된 이상만 찾는 supervised detector가 아니라, **정상 분포에서 벗어난 새로운 패턴 자체를 표시**할 가능성을 보여준 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **정상 데이터만으로 학습한 생성모델을 이용해 병리적 이상을 탐지한다는 문제 설정을 설득력 있게 구현했다는 점**이다. 의료영상에서는 라벨링 비용과 병변 정의의 불완전성이 매우 크므로, 정상 데이터 중심의 unsupervised 혹은 one-class 접근은 실제 가치가 높다. 또한 GAN의 generator와 discriminator를 모두 활용하여 anomaly score를 구성하고, residual image로 국소 영역까지 시각화한 점은 임상 해석 가능성 측면에서도 의미가 있다.

둘째 강점은 **latent mapping 문제를 독립적인 핵심 요소로 다루고, feature matching 기반 discrimination loss를 도입해 이를 개선했다는 점**이다. 이 설계는 단순히 GAN을 anomaly detection에 가져다 쓴 수준을 넘어, 테스트 시 inference 절차까지 논리적으로 확장한 기여라고 볼 수 있다.

셋째, 실험 결과가 단순 수치 비교에 그치지 않고 실제 OCT 병변 위치와 residual의 대응 관계를 보여주며, 주석되지 않은 HRF까지 검출한 사례를 제시한 것은 방법의 잠재력을 잘 드러낸다.

반면 한계도 분명하다. 가장 실용적인 한계는 **추론 속도**다. 테스트마다 latent code를 500 step backpropagation으로 최적화해야 하므로, 직접 encoder를 두는 방식보다 느리다. 논문도 aCAE가 direct mapping 덕분에 runtime 측면의 이점이 있음을 언급한다.

또한 평가가 retinal OCT와 retinal fluid 중심으로 이루어졌기 때문에, 다른 의료영상 모달리티나 더 복잡한 병리 분포에서도 동일하게 잘 작동하는지는 아직 불확실하다. 논문 스스로도 정량 평가가 일부 anomaly class에 한정되어 있어, false positive가 실제로는 “새로운 이상”일 가능성을 완전히 반영하지 못한다고 인정한다.

비판적으로 보면, discriminator output만 직접 사용하는 방식도 AUC 0.88로 강력했기 때문에, 전체 시스템에서 iterative latent mapping이 가져오는 실질적 이득이 모든 상황에서 압도적인지는 추가 검증이 필요하다. 그럼에도 불구하고 이 논문은 단순 성능 최적화보다 **새로운 이상 후보(marker candidate)를 발굴하는 프레임워크**라는 관점에서 읽는 것이 더 적절하다.

## 6. 결론

이 논문은 GAN을 이용해 정상 해부학적 변이를 학습하고, 새로운 입력이 그 정상 manifold에 얼마나 잘 맞는지를 바탕으로 이상을 탐지하는 **AnoGAN**을 제안했다. 핵심 기여는 정상 데이터만으로 학습된 DCGAN, query image를 latent space로 투영하는 iterative mapping 절차, 그리고 residual loss와 feature matching 기반 discrimination loss를 결합한 anomaly score 설계에 있다.

실험에서는 retina SD-OCT 데이터에서 retinal fluid를 포함한 병리 패치를 효과적으로 구분했으며, AUC 0.89의 성능과 함께 HRF 같은 추가 이상도 탐지할 수 있음을 보였다. 이 결과는 방법이 단순한 supervised lesion detector를 넘어, **라벨되지 않은 새로운 병리적 패턴을 탐색하는 도구**로 활용될 가능성을 보여준다.

후속 연구 관점에서 이 논문은 매우 중요한 출발점이다. 이후 많은 연구들이 AnoGAN의 느린 latent optimization 문제를 encoder 기반 추론으로 개선하거나, GAN 대신 VAE·flow·diffusion 등 다양한 생성모델로 확장했다. 그럼에도 이 논문은 “정상 생성 모델을 이용한 이상 탐지”라는 흐름을 의료영상 분야에서 분명하게 제시한 대표적 초기 작업으로 평가할 수 있다.
