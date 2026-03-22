# Deep Bayesian Active Learning with Image Data

첨부된 ar5iv HTML 원문을 바탕으로 정리한 상세 분석 보고서다. 이 논문은 **deep learning을 active learning에 실용적으로 접목**하는 초기 대표작 중 하나로, 특히 **Bayesian CNN + uncertainty-based acquisition**을 결합해 고차원 이미지 데이터에서도 active learning이 작동할 수 있음을 보여준다. 논문의 문제의식은 단순하다. active learning은 본질적으로 “적은 라벨로 잘 배워야 하는” 설정인데, 일반적인 deep learning은 반대로 “많은 라벨 데이터”를 요구하고, 또 uncertainty를 제대로 표현하지 못하는 경우가 많다. 저자들은 이 간극을 **Bayesian deep learning**, 구체적으로 **MC dropout을 이용한 Bayesian convolutional neural network**로 메우고자 한다.  

## 1. Paper Overview

이 논문이 다루는 핵심 문제는 **이미지처럼 고차원 입력을 가지는 active learning**이다. 전통적 active learning은 주로 SVM, Gaussian process, graph/kernel 기반 방법에 기대어 왔고, 이미지와 같은 고차원 데이터에서는 확장성이 떨어졌다. 반면 CNN은 이미지에 매우 강하지만, active learning이 요구하는 “작은 데이터에서의 안정적 학습”과 “예측 uncertainty 표현”에는 약점이 있었다. 저자들은 이 지점을 정확히 찌르며, **Bayesian CNN을 통해 uncertainty를 표현하고, 그 uncertainty를 acquisition function에 넣어 효율적으로 라벨링 대상을 고르는 프레임워크**를 제안한다.

왜 이 문제가 중요한가도 분명하다. 의료 영상, 피부 병변 이미지, 진단 보조 시스템 같은 실제 문제에서는 라벨링이 비싸고 전문가 시간이 필요하다. 따라서 무작위로 데이터를 더 모으는 대신, **가장 정보량이 큰 샘플부터 선택적으로 라벨링**하면 비용과 시간을 크게 줄일 수 있다. 저자들은 실제로 MNIST뿐 아니라 **피부암 병변 이미지(ISIC 2016)** 실험까지 포함해, 이 접근이 toy setting을 넘어서 실제 응용 가능성이 있음을 보이려 한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 크게 두 축이다.

첫째, **CNN을 Bayesian model로 바꿔 uncertainty를 추정**한다. 저자들은 기존 kernel 기반 active learning 대신, 이미지에 특화된 CNN을 쓰되 이를 Bayesian CNN으로 해석해 예측 분포의 불확실성을 얻는다. 이를 위해 dropout을 단순 regularizer가 아니라 **approximate variational inference** 도구로 해석하고, test time에도 dropout을 켠 채 여러 번 forward pass를 수행하는 **MC dropout**을 사용한다. 이렇게 하면 하나의 deterministic prediction이 아니라 posterior predictive distribution에 대한 근사값을 얻을 수 있다.

둘째, 이렇게 얻은 uncertainty를 이용해 **어떤 unlabeled image를 다음에 라벨링할지 결정**한다. 논문은 BALD, Max Entropy, Variation Ratios, Mean STD 같은 acquisition function을 비교하며, 이 중 특히 **BALD**와 **Variation Ratios**가 효과적이라고 보인다. 즉, “모델이 가장 헷갈려하는 샘플” 혹은 “모델 posterior 관점에서 정보 이득이 큰 샘플”을 골라 라벨을 요청하는 전략이다. 이 접근의 novelty는 deep model을 그냥 분류기로 쓰는 데 그치지 않고, **Bayesian approximation을 통해 active learning의 핵심 자원인 uncertainty를 복원했다는 점**이다.  

## 3. Detailed Method Explanation

### 3.1 Bayesian CNN의 기본 설정

논문은 CNN의 가중치 집합을 $\omega = {W_1, \dots, W_L}$라고 두고, 이 위에 prior $p(\omega)$를 둔다. 분류 문제에서는 출력이 다음과 같이 softmax likelihood로 정의된다.

$$
p(y=c \mid \mathbf{x}, \omega) = \mathrm{softmax}(\mathbf{f}^{\omega}(\mathbf{x}))
$$

즉, 입력 이미지 $\mathbf{x}$에 대해 네트워크가 logits를 출력하고, 이를 softmax로 바꿔 클래스 확률로 해석한다. 여기서 핵심은 가중치 $\omega$가 고정 파라미터가 아니라 확률변수라는 점이다. 따라서 최종 예측은 하나의 네트워크 출력이 아니라 posterior over weights를 적분한 predictive distribution이 된다.

### 3.2 MC dropout을 통한 posterior predictive 근사

정확한 Bayesian posterior $p(\omega \mid \mathcal{D}_{train})$를 계산하는 것은 어렵기 때문에, 논문은 dropout 기반 variational approximation을 사용한다. 저자들이 제시한 predictive distribution은 다음과 같은 형태다.

$$
p(y=c \mid \mathbf{x}, \mathcal{D}_{train})
= \int p(y=c \mid \mathbf{x}, \omega), p(\omega \mid \mathcal{D}_{train}), d\omega
$$

이를 tractable한 variational distribution $q_\theta^*(\omega)$로 근사하고,

$$
p(y=c \mid \mathbf{x}, \mathcal{D}_{train})
\approx \int p(y=c \mid \mathbf{x}, \omega), q*\theta^*(\omega), d\omega
$$

다시 이를 Monte Carlo 평균으로 계산한다.

$$
p(y=c \mid \mathbf{x}, \mathcal{D}_{train})
\approx \frac{1}{T} \sum*{t=1}^T p(y=c \mid \mathbf{x}, \hat{\omega}_t)
$$

여기서 $\hat{\omega}_t \sim q*\theta^*(\omega)$는 dropout mask가 적용된 서로 다른 샘플에 해당한다. 직관적으로는 **test time dropout을 켠 채 여러 번 예측하고 평균을 내는 것**이다. 이 과정이 epistemic uncertainty를 반영해 준다.

### 3.3 왜 Bayesian uncertainty가 중요한가

저자들은 active learning이 잘 작동하려면 **“무엇을 모르는지”를 모델이 알아야 한다**고 본다. deterministic CNN도 softmax probability는 낼 수 있지만, 그것은 calibration이 좋지 않을 수 있고 model uncertainty를 반영하지 않는다. 논문은 Bayesian CNN과 deterministic CNN을 비교해, 같은 acquisition function을 쓰더라도 **Bayesian 모델의 uncertainty propagation이 더 유의미한 acquisition을 만든다**고 주장한다. 즉, active learning 성능 향상의 핵심은 단순히 CNN을 썼기 때문이 아니라, **Bayesian treatment를 통해 uncertainty를 더 잘 측정했기 때문**이라는 것이다.  

### 3.4 Acquisition function들

논문은 여러 acquisition function을 비교한다.

* **BALD**: Bayesian Active Learning by Disagreement. 모델 파라미터 posterior에 대해 예측 불확실성과 정보 이득을 본다.
* **Max Entropy**: predictive entropy가 큰 샘플을 고른다.
* **Variation Ratios**: ensemble/MC 샘플들 사이에서 mode class의 빈도가 낮은 샘플을 고른다.
* **Mean STD**: 예측 확률들의 표준편차를 기준으로 삼는 방식.
* **Random**: baseline.

실험 결과에서 Mean STD는 Random과 비슷하게 부진했고, BALD, Variation Ratios, Max Entropy가 훨씬 낫다. 특히 MNIST에서는 **Variation Ratios가 BALD와 Max Entropy보다 약간 더 빨리 좋은 성능에 도달**하는 것으로 보인다고 저자들은 해석한다.  

### 3.5 이미지 active learning을 위한 전체 파이프라인

전체 절차는 다음처럼 이해하면 된다.

1. 작은 initial labeled set으로 Bayesian CNN을 학습한다.
2. unlabeled pool의 각 샘플에 대해 MC dropout으로 여러 번 예측한다.
3. acquisition function(BALD, Variation Ratios 등)을 계산한다.
4. 값이 큰 샘플들을 골라 oracle에게 라벨을 요청한다.
5. 새 라벨을 training set에 추가하고 모델을 다시 학습한다.
6. 이를 반복한다.

이 구조 자체는 active learning의 표준 loop이지만, 논문의 차별점은 **이미지에 특화된 CNN 표현력**과 **Bayesian uncertainty estimation**을 동시에 가져왔다는 점이다.

## 4. Experiments and Findings

### 4.1 MNIST에서 acquisition function 비교

MNIST 실험은 initial training set 20개, validation 100개, test 10K로 설정하고, 매 acquisition step마다 10개 샘플을 새로 고르는 방식으로 진행된다. 저자들은 acquisition 과정을 100회 반복하고, 실험 자체는 3회 반복해 평균을 냈다. 이 비교에서 **Random과 Mean STD는 뚜렷하게 성능이 낮았고**, BALD, Variation Ratios, Max Entropy가 훨씬 효율적이었다. 저자들은 특히 **Variation Ratios가 약간 더 빠르게 높은 정확도에 도달**한다고 서술한다.

표 1의 메시지도 강력하다. MNIST에서 **5% test error**에 도달하기 위해 필요한 labeled image 수는 BALD 335, Variation Ratios 295, Max Entropy 355, Mean STD 695, Random 835였다. 즉, 같은 수준의 성능을 얻기 위해 Random 대비 Variation Ratios는 **절반 이하의 라벨 수**만 요구한다. 이것이 논문이 말하는 “data efficiency”다.  

### 4.2 모델 uncertainty의 중요성

논문은 acquisition 함수가 좋아도, 그 uncertainty 자체가 믿을 만하지 않으면 active learning이 무너진다고 본다. 그래서 Bayesian CNN 대신 deterministic CNN에 BALD, Variation Ratios, Max Entropy를 적용해 비교한다. 결론은 **Bayesian uncertainty를 제대로 사용한 쪽이 훨씬 낫다**는 것이다. 저자들은 이를 통해 active learning 개선이 단순 모델 용량 증가 때문이 아니라, **Bayesian 모델이 confidence를 더 제대로 측정하기 때문**임을 보이려 한다.  

### 4.3 기존 이미지 active learning 기법과 비교

저자들은 고전적인 이미지 active learning 접근인 **MBR**(Zhu et al. 2003, Gaussian random field 기반)과도 비교한다. MBR은 raw image에 대한 RBF kernel similarity graph를 사용해 expected classification error를 줄이는 방향으로 unlabeled point를 선택한다. 그런데 binary MNIST 실험에서, 저자들은 **랜덤 acquisition + CNN조차 MBR보다 낫다**고 보고한다. 심지어 MBR의 kernel 부분을 CNN으로 바꿔도 성능 개선이 크지 않았다고 말한다. 이는 이 논문이 kernel 방식보다 **specialised image model 자체가 중요하다**는 점을 강조하는 대목이다.  

### 4.4 Semi-supervised learning과 비교

흥미롭게도 저자들은 자신들의 active learning 시스템을 semi-supervised learning 기법과도 비교한다. 1000 labeled MNIST 샘플만 사용하는 조건에서 제안 방식은 **1.64% test error**를 기록했고, 논문에서 언급한 DGN은 **2.40%**, Ladder Network는 **1.53%**였다. 여기서 중요한 점은 Ladder Network와 DGN은 **나머지 unlabeled training set 전체를 함께 사용**한다는 것이다. 반면 이 논문의 active learning은 오직 acquired 1000 labeled example만을 사용한다. 즉, 순수 active learning 세팅만으로도 semi-supervised strong baseline에 꽤 근접한 성능을 보인다는 점이 인상적이다.  

### 4.5 피부암 진단(ISIC 2016) 실험

실제 응용으로는 **melanoma diagnosis**가 사용된다. 데이터는 ISIC Archive 기반 ISBI 2016 task training split이며, 총 900장의 dermoscopic lesion image가 있고, 이 중 **727 benign / 173 malignant**로 클래스 불균형이 심하다. 모델은 이전 공개 코드 기반의 VGG16 fine-tuning 방식이며, 마지막 1000-way layer를 2-class output으로 바꾸고, 그 앞의 두 개 4096-unit fully connected layer 뒤에 dropout 0.5를 둔다. 즉, 실험은 완전히 새로운 네트워크를 설계했다기보다, **강한 vision backbone 위에 Bayesian uncertainty 추정 구조를 얹는 방식**이다.  

이 실험에서 성능 평가는 accuracy보다 **AUC**를 사용한다. 저자들은 average precision이 데이터 imbalance 때문에 misleading할 수 있어 AUC가 더 informative하다고 설명한다. 또한 이 setting에서는 MNIST와 달리 Variation Ratios가 실패하는데, 그 이유는 malignant probability가 benign 이미지에서도 약간씩 높게 나와서 acquisition 값이 거의 동일해지기 때문이라고 해석한다. 그래서 lesion 실험에서는 **uniform baseline과 BALD**만 비교한다. 결과적으로 BALD는 더 많은 positive example을 더 빨리 찾아내고, AUC도 더 빠르게 개선한다. 저자들은 특히 BALD가 **aleatoric uncertainty가 큰 noisy point보다 epistemic uncertainty가 큰 sample을 선호해** 더 효율적인 acquisition을 했다고 해석한다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 첫 번째 강점은 시기적 중요성이다. 지금 보면 MC dropout 기반 uncertainty는 익숙하지만, 당시에는 **deep learning과 active learning을 실용적으로 연결한 설득력 있는 사례**가 많지 않았다. 논문은 이미지용 specialised model(CNN)과 Bayesian uncertainty를 결합함으로써, “active learning은 저차원/kernel 기반에서만 된다”는 인식을 실제로 깨뜨렸다.

두 번째 강점은 **단순하지만 구현 가능한 Bayesian approximation**을 썼다는 점이다. 완전한 Bayesian deep net inference는 매우 어렵지만, dropout을 variational approximation으로 해석하면 기존 딥러닝 도구체인 안에서도 نسب적으로 쉽게 실험할 수 있다. 이 때문에 논문은 개념 논문을 넘어서 재현성과 실용성을 동시에 가진다.

세 번째 강점은 **실제 의료영상 응용까지 확장**했다는 점이다. MNIST만으로 끝났다면 방법론 소개에 그쳤을 수 있는데, melanoma diagnosis라는 현실적 태스크까지 포함해 “라벨이 비싼 영역에서 정말 의미가 있는가”를 보여준다.  

### 한계

첫 번째 한계는, uncertainty 추정이 **MC dropout approximation**에 강하게 의존한다는 점이다. 이는 practical하지만 posterior approximation의 질이 언제나 충분하다고 보장되지는 않는다. 특히 acquisition quality가 uncertainty estimation quality에 직접 의존하므로, dropout 기반 추정이 부정확하면 전체 active learning loop가 흔들릴 수 있다. 이 한계는 논문이 implicit하게 안고 있는 구조적 제약이다.

두 번째 한계는 실험 비용이다. 저자들은 acquisition function의 순수 효과를 보기 위해 **매 acquisition step마다 모델을 초기 pre-trained weight로 reset하고 다시 수렴까지 학습**했다고 설명한다. 이는 공정한 비교에는 좋지만 실제 운영에서는 상당히 비싸다. 논문은 melanoma 실험 하나에만도 **20시간**이 걸렸다고 말하며, 이는 실시간 혹은 반복적 human-in-the-loop 환경에는 부담이 크다.  

세 번째 한계는 acquisition function의 **dataset dependence**다. MNIST에서는 Variation Ratios가 매우 잘 작동하지만, melanoma 데이터에서는 사실상 실패한다. 이는 uncertainty-based acquisition이 항상 보편적으로 강한 것이 아니라, 데이터 분포와 calibration 특성에 따라 크게 달라질 수 있음을 시사한다.  

### 해석

비판적으로 보면, 이 논문의 가장 중요한 기여는 “최고의 acquisition function을 하나 발명했다”기보다, **deep model에서 uncertainty를 어떻게 operationalize할 것인가**를 active learning 맥락에서 명확히 보여준 데 있다. 즉, BALD나 Variation Ratios 자체보다도, **Bayesian deep learning이 active learning의 핵심 병목인 uncertainty estimation 문제를 풀 수 있다**는 메시지가 더 본질적이다. 이후 등장한 많은 deep active learning 연구들이 ensemble, Bayesian approximation, evidential uncertainty, calibration 등을 계속 탐색하게 된 출발점 중 하나라고 볼 수 있다.

## 6. Conclusion

이 논문은 **Bayesian convolutional neural network와 MC dropout을 이용해 이미지 active learning을 실용적으로 구현**한 작업이다. 핵심 기여는 세 가지다. 첫째, 고차원 이미지 데이터에서도 uncertainty-aware active learning이 가능함을 보였다. 둘째, BALD, Variation Ratios, Max Entropy 같은 acquisition function을 deep Bayesian setting에서 비교하고, random baseline 대비 큰 라벨 효율 개선을 확인했다. 셋째, MNIST뿐 아니라 피부암 진단 같은 실제 의료 이미지 문제에서도 BALD가 유의미한 이점을 보인다는 것을 시연했다.

실무적으로 이 논문은 “라벨링이 비싸고, 이미지가 많고, uncertainty가 중요한” 문제에서 특히 의미가 있다. 연구적으로는 이후 deep active learning 분야가 발전하는 데 중요한 초석 역할을 했다고 평가할 수 있다. 다만 computational cost와 dropout approximation의 한계, acquisition 함수의 데이터 의존성은 후속 연구가 보완해야 할 부분으로 남는다.
