# Continual Classification Learning Using Generative Models

* **저자**: Frantzeska Lavda, Jason Ramapuram, Magda Gregorova, Alexandros Kalousis
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1810.10612](https://arxiv.org/abs/1810.10612)

## 1. 논문 개요

이 논문은 **continual learning** 환경에서 분류 모델이 순차적으로 여러 태스크를 학습할 때 발생하는 **catastrophic forgetting** 문제를 해결하는 것을 목표로 한다. 구체적으로는, 과거 태스크의 원본 데이터에 더 이상 접근할 수 없고, 과거 태스크별 모델을 따로 저장하지도 않는 조건에서, 새로운 태스크를 계속 배우면서도 이전 태스크의 분류 성능을 유지하려는 문제를 다룬다.

저자들은 이를 위해 **VAE(Variational Autoencoder)**의 encoder와 decoder에 classifier를 결합한 잠재변수 기반 분류 모델을 제안한다. 핵심은 단순히 reconstruction만 잘하는 생성모델이 아니라, 입력 $x$와 라벨 $y$를 **공동으로 모델링하는 joint likelihood $\log p(x,y)$** 를 최적화하는 방식으로 분류와 생성을 함께 학습한다는 점이다. 이를 위해 저자들은 $\log p(x,y)$에 대한 새로운 variational lower bound를 유도한다.

이 문제가 중요한 이유는 continual learning의 핵심 제약 때문이다. 실제 환경에서는 데이터가 한 번에 모두 주어지지 않고 시간에 따라 순차적으로 도착하는 경우가 많다. 이때 일반적인 신경망은 모든 태스크를 한꺼번에 학습하면 여러 태스크를 처리할 수 있지만, 태스크를 하나씩 순서대로 학습하면 이전 태스크 성능이 급격히 무너진다. 논문은 이러한 구조적 한계를 완화하기 위해, 과거 데이터를 저장하지 않고도 **과거 태스크를 “생성”해서 재학습에 활용하는 방식**을 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **생성모델의 lifelong generative capability를 분류 문제에 확장**하는 것이다. 저자들은 기존의 lifelong generative modeling 연구를 바탕으로, **student-teacher architecture**를 사용한다. teacher는 과거 태스크 분포의 요약을 내부에 보존하고 있으며, 과거 원본 데이터가 사라진 뒤에도 이전 태스크의 입력-라벨 쌍을 생성할 수 있다. student는 현재 새로 들어온 태스크 데이터와 teacher가 생성한 과거 태스크 데이터를 함께 사용하여 학습한다.

이 설계의 직관은 명확하다. catastrophic forgetting은 새 태스크를 학습할 때 이전 태스크의 분포를 더 이상 보지 못하기 때문에 발생한다. 그렇다면 이전 데이터를 직접 저장하지 않더라도, 이전 데이터를 닮은 샘플을 생성해서 현재 학습에 섞어주면 forgetting을 줄일 수 있다. 저자들은 이 직관을 단순한 replay 수준에 그치지 않고, **분류와 생성이 동시에 가능한 latent variable model** 위에 올려 정식화했다.

기존 접근과 비교했을 때 차별점은 세 가지 정도로 정리할 수 있다.

첫째, dynamic architecture 계열처럼 태스크별 과거 모델을 계속 보존하지 않는다. 논문은 teacher 안에 과거 분포의 요약만 유지하고, 과거 태스크별 별도 모델 저장을 피한다.

둘째, regularization 기반 방법들처럼 “이전 태스크에 중요한 파라미터를 덜 바꾸자”는 제약만 사용하는 것이 아니다. 대신 과거 태스크 데이터를 직접 생성해 현재 student가 다시 학습하게 만든다. 즉, 파라미터 보존이 아니라 **분포 replay**에 가깝다.

셋째, VCL 같은 variational continual learning 계열과 비교해, 이 논문은 생성과 분류를 동시에 하나의 모델 안에서 다루며, 코어셋 저장이나 task-specific head에 의존하지 않는 continual learning 조건을 더 엄격하게 지키려 한다고 주장한다.

## 3. 상세 방법 설명

### 3.1 기본 잠재변수 모델

논문은 라벨이 있는 데이터 쌍 $(x,y)$를 다룬다. 각 입력 $x$에는 잠재변수 $z$가 대응되며, 이 $z$를 통해 입력과 라벨이 함께 설명된다. 저자들이 사용하는 결합분포는 다음과 같이 factorization된다.

$$
p(x,y,z) = p(x|z)p(y|z)p(z)
$$

여기서 중요한 가정은 **조건부 독립성**이다. 즉, $z$가 주어지면 $x$와 $y$는 서로 독립이라고 본다.

$$
p(x,y|z) = p(x|z)p(y|z)
$$

이 가정은 “좋은 잠재표현 $z$가 입력의 정보와 분류에 필요한 정보를 함께 담고 있다”는 해석과 연결된다. decoder $p(x|z)$는 입력을 복원하고, classifier $p(y|z)$는 라벨을 예측한다.

### 3.2 posterior approximation과 새로운 variational bound

원래 posterior는 $p(z|x,y)$이지만, 테스트 시점에는 $y$를 알 수 없으므로 직접 사용할 수 없다. 그래서 논문은 VAE와 유사하게 근사 posterior로 $q_\phi(z|x)$를 사용한다. 이는 분류 시점에도 사용할 수 있는 형태다.

저자들은 다음 KL divergence를 최소화하는 관점에서 출발한다.

$$
D_{KL}(q_\phi(z|x)|p(z|x,y)) = - E_{q_\phi(z|x)}[\log p(x,y,z)-\log q_\phi(z|x)] + \log p(x,y)
$$

여기서 $\log p(x,y)$는 상수이므로, 결국 다음 항을 최대화하면 된다.

$$
L(x,y)=E_{q_\phi(z|x)}[\log p(x,y,z)-\log q_\phi(z|x)]
$$

그리고 이를 다시 정리하면,

$$
\log p(x,y) = L(x,y) + D_{KL}(q_\phi(z|x)|p(z|x,y))
$$

가 되어, KL divergence가 항상 0 이상이므로 $L(x,y)$가 $\log p(x,y)$의 lower bound임을 알 수 있다.

즉, 논문의 핵심 이론 기여는 **VAE의 ELBO를 $\log p(x)$가 아니라 $\log p(x,y)$에 대해 확장한 새로운 variational bound**를 제시한 데 있다.

### 3.3 bound의 해석: ELBO + classification term

조건부 독립성 가정을 사용하면 이 bound는 다음처럼 해석된다.

$$
L(x,y) = E_{q_\phi(z|x)}[\log p(x|z)] - D_{KL}(q_\phi(z|x)|p(z)) + E_{q_\phi(z|x)}[\log p(y|z)]
$$

이 식은 매우 중요하다. 왜냐하면 세 부분으로 나뉘기 때문이다.

첫 번째와 두 번째 항

$$
E_{q_\phi(z|x)}[\log p(x|z)] - D_{KL}(q_\phi(z|x)|p(z))
$$

은 표준 VAE의 **ELBO**이다. 즉, 입력을 잘 복원하고 posterior가 prior에서 너무 멀어지지 않도록 하는 생성모델 목적함수다.

세 번째 항

$$
E_{q_\phi(z|x)}[\log p(y|z)]
$$

은 latent variable $z$로부터 올바른 라벨을 예측하도록 만드는 **classification loss**이다.

따라서 이 모델은 “좋은 reconstruction을 위한 잠재공간”과 “좋은 classification을 위한 잠재공간”을 따로 만드는 대신, **하나의 공통 latent representation** $z$가 두 역할을 동시에 하도록 만든다. 논문이 반복해서 강조하는 포인트도 바로 이것이다.

### 3.4 student-teacher continual learning 구조

모델은 teacher와 student 두 네트워크로 구성된다. 둘 다 다음 요소를 가진다.

* encoder: $q_{\phi^m}(z|x)$
* decoder: $p_{\theta_x^m}(x|z)$
* classifier: $p_{\theta_y^m}(y|z)$

여기서 $m \equiv t,s$는 각각 teacher와 student를 의미한다.

teacher의 역할은 과거 태스크 기억 유지다. 과거 태스크의 실제 데이터는 더 이상 없지만, teacher는 과거 분포에서 샘플 $(\tilde{x}, \tilde{y})$를 생성할 수 있다. student는 현재 태스크의 실제 데이터 $(x,y)$와 teacher가 생성한 과거 태스크 데이터 $(\tilde{x}, \tilde{y})$를 함께 학습한다.

학습 절차는 다음처럼 이해할 수 있다.

현재 태스크가 끝나고 다음 태스크가 시작될 때, student의 최신 파라미터가 teacher로 전달된다. 이후 새 태스크 학습 중에는 teacher가 예전 태스크 데이터의 대용 샘플을 생성하고, student는 그것과 새 데이터를 함께 사용해 분류기와 생성기를 다시 학습한다. 이렇게 하면 과거 태스크 데이터를 직접 저장하지 않고도, 과거 분포를 간접적으로 다시 경험할 수 있다.

### 3.5 최종 학습 목표

논문은 student가 단순히 joint likelihood bound만 최적화하는 것이 아니라, 추가적인 regularization 항도 넣는다고 설명한다. 최종 loss는 식 (5)로 제시된다.

$$
E_{q_{\phi^s}(z|x^s)} \left[ \log p_{\theta_x^s}(x|z^s) + \log p_{\theta_y^s}(y|z^s) \right] -
D_{KL}(q(z|x)|p(z)) - D_{KL}[q_{\phi^s}(z^s|\tilde{x})|q_{\phi^t}(z^t|\tilde{x})] - L_I(z,\tilde{x})
$$

이 식을 구성요소별로 쉽게 풀어보면 다음과 같다.

첫 번째 부분은 현재 student가 입력을 잘 복원하고, 라벨도 잘 맞추도록 하는 항이다. 즉 생성과 분류를 동시에 학습한다.

두 번째 부분은 VAE의 일반적인 KL regularization으로, posterior $q(z|x)$가 prior $p(z)$에서 너무 멀어지지 않도록 한다.

세 번째 부분은 teacher가 생성한 과거 샘플 $\tilde{x}$에 대해, student의 posterior 표현이 teacher의 posterior 표현과 비슷해지도록 강제하는 항이다. 이것은 **과거 태스크의 latent representation을 유지**하게 도와 훈련을 빠르게 하고 안정화하려는 목적을 가진다.

네 번째의 negative information gain regularizer $L_I(z,\tilde{x})$는 teacher가 생성한 데이터와 latent representation 사이의 정보 관련 regularization으로 소개되지만, 제공된 텍스트에는 이 항의 정확한 정의나 계산 방식이 자세히 나와 있지 않다. 따라서 이 항이 구체적으로 어떤 수식으로 구현되었는지는 본문 추출 텍스트만으로는 확인할 수 없다.

### 3.6 왜 이 접근이 타당한가

논문은 추가로 다음 관계를 언급한다.

$$
\frac{p(z|x,y)}{p(z|x)} = \frac{p(y|z)}{p(y|x)}
$$

그리고 $z$가 $x$를 분류에 충분히 잘 요약한다면 $p(y|z) \approx p(y|x)$라고 보고, 결과적으로 $p(z|x,y)$와 $p(z|x)$가 매우 유사해질 수 있다고 설명한다. 다시 말해, 테스트 시점에서 $y$ 없이도 $q_\phi(z|x)$만으로 충분히 좋은 추론이 가능하도록 만드는 논리적 근거를 제시한 것이다.

이 부분은 엄밀한 정리라기보다, 왜 $q(z|x,y)$가 아니라 $q(z|x)$를 써도 합리적인지에 대한 **직관적 정당화**로 이해하는 것이 적절하다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 제안한 방법이 실제로 catastrophic forgetting을 줄이는지 보기 위해 continual classification 실험을 수행한다. 첫 번째 주요 실험은 **permuted MNIST**에서 진행된다. 각 태스크는 숫자 0부터 9까지의 10-way classification이지만, 이미지의 픽셀 순서를 태스크마다 서로 다른 고정 permutation으로 섞는다. 즉, 라벨 공간은 같지만 입력 분포가 태스크마다 달라진다.

실험에서는 총 5개 태스크를 순차적으로 학습한다. 원본 MNIST 하나와, 서로 다른 4개의 random permutation 태스크다. 중요한 조건은 각 태스크 학습이 끝나면 그 태스크 데이터에는 더 이상 접근할 수 없다는 점이다. 이는 continual learning 제약을 반영한다.

학습은 mini-batch 256으로 수행하고, classification accuracy 기준 early stopping을 사용한다.

### 4.2 비교 대상

비교 모델은 두 가지다.

첫 번째 baseline은 **vae-cl**이다. 이것은 논문이 제안한 variational bound와 classifier를 사용하지만, teacher-student architecture는 없는 모델이다. 다시 말해 joint generative-discriminative VAE이긴 하지만 continual replay 메커니즘이 없다. 따라서 단순한 joint model만으로 forgetting을 막을 수 있는지를 비교하는 기준이 된다.

두 번째 baseline은 **EWC(Elastic Weight Consolidation)**를 이 문제에 맞게 변형한 방식이다. 논문 설명에 따르면 이 baseline에서도 teacher는 과거 분포의 요약을 유지하는 데 사용되지만, teacher가 student를 위해 데이터를 생성하지는 않는다. 대신 teacher와 student의 파라미터 차이에 대해 Fisher 정보 기반 regularization을 건다.

식으로는 대략 다음 형태다.

$$
\sum_i F_i(\psi_i^s - \psi_i^t)^2
$$

여기서 $\psi^m=[\theta^m,\phi^m]$는 모델 파라미터이며, $F_i$는 Fisher diagonal 추정량이다. 즉, 과거 태스크에 중요했던 파라미터를 너무 많이 바꾸지 않게 하는 전형적인 EWC 철학이다.

### 4.3 평가 지표

논문은 두 가지 관점에서 성능을 본다.

하나는 **average test classification accuracy**이고, 다른 하나는 **average test negative reconstruction ELBO**이다. 전자는 분류를 얼마나 잘 유지하는지, 후자는 생성모델로서 reconstruction 품질을 얼마나 유지하는지를 본다.

즉, 이 논문은 단순히 분류만 잘하는 continual learner가 아니라, **분류와 생성 둘 다에서 forgetting을 줄이는지**를 평가한다.

### 4.4 permuted MNIST 결과

논문 설명에 따르면, naive한 vae-cl은 MNIST에서 첫 번째 permuted task로 넘어가자마자 성능이 크게 떨어진다. 이는 joint VAE-classifier 구조만으로는 continual learning이 거의 해결되지 않음을 보여준다.

EWC는 vae-cl보다는 덜 심하게 성능이 감소하지만, 여전히 이전 태스크를 잊는다. 즉, 파라미터 중요도 기반 regularization만으로는 충분하지 않다는 결과다.

반면 제안 방법인 **CCL-GM**은 태스크 수가 늘어나도 높은 평균 분류 정확도와 낮은 평균 negative reconstruction ELBO를 유지한다. 저자들은 이를 근거로, 자신들의 방법이 **classification과 generation을 동시에 연속적으로 유지**할 수 있다고 주장한다.

다만 제공된 추출 텍스트에는 Figure 3의 정확한 수치값이 들어 있지 않다. 따라서 “얼마나 몇 퍼센트 더 높았는지” 같은 정량적 차이를 정확히 재현할 수는 없다. 텍스트 수준에서 확인 가능한 것은, **상대적 추세상 CCL-GM이 가장 안정적이었다**는 점이다.

### 4.5 두 번째 실험: 서로 다른 데이터셋 시퀀스

추가 실험으로 저자들은 세 개의 태스크 시퀀스를 사용한다.

* MNIST
* FashionMNIST
* 하나의 MNIST permutation

이 설정은 첫 번째 실험보다 더 이질적인 분포 이동을 포함한다. 결과는 Figure 4로 제시되며, 여기서도 제안 방법이 baseline보다 좋았다고 서술한다. 저자들은 이를 통해 CCL-GM이 단지 permuted MNIST 같은 구조적 toy setting뿐 아니라, 서로 다른 시각 데이터 분포가 섞인 경우에도 forgetting을 줄일 잠재력이 있음을 보여준다고 주장한다.

여기 역시 구체적인 수치표는 제공된 텍스트에 없으므로, 실험 결과의 절대값이나 통계적 유의성 여부까지는 판단할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **생성과 분류를 하나의 잠재변수 모델 안에서 함께 정식화**했다는 점이다. 많은 continual learning 논문이 분류 성능 유지에만 집중하는 반면, 이 논문은 reconstruction 능력까지 함께 유지하는 방향으로 문제를 설정했다. 따라서 latent representation이 단순한 discriminative feature가 아니라, 과거 데이터 분포까지 포착하는 더 풍부한 표현이 되도록 유도한다.

또 다른 강점은 **새로운 variational bound의 해석이 비교적 자연스럽다**는 점이다. 식 (4)는 기존 VAE의 ELBO에 classification term을 더한 형태로 읽히기 때문에, 왜 이런 목적함수가 필요한지 이해하기 쉽다. 저자들 스스로도 기존 분류용 VAE가 classification term을 다소 ad-hoc하게 붙였다고 비판하면서, 자신들의 접근은 $\log p(x,y)$를 직접 다루는 점에서 더 원리적이라고 주장한다.

구조적 측면에서도 장점이 있다. teacher가 과거 데이터 분포를 요약해 생성해주기 때문에, 원본 데이터를 저장하지 않고도 replay 효과를 낼 수 있다. 이는 저장 제약이나 개인정보 이슈가 있는 환경에서 개념적으로 매력적이다.

하지만 한계도 분명하다.

첫째, 실험이 아직 **preliminary** 수준이다. 논문 본문에서도 그렇게 표현하며, 주된 실험이 permuted MNIST와 MNIST/FashionMNIST 같은 비교적 단순한 벤치마크에 머물러 있다. 더 복잡한 실제 continual vision task에서 얼마나 잘 동작하는지는 본문만으로 판단하기 어렵다.

둘째, 생성 품질에 크게 의존하는 구조라는 점이 잠재적 약점이다. teacher가 과거 태스크 데이터를 잘 생성하지 못하면, student는 왜곡된 replay data로 학습하게 된다. 논문은 reconstruction ELBO를 함께 제시하지만, 생성 샘플의 실제 품질이나 label fidelity에 대한 깊은 분석은 제공된 텍스트에는 충분하지 않다.

셋째, 최종 loss에 포함된 **negative information gain regularizer** $L_I(z,\tilde{x})$가 텍스트에 충분히 설명되어 있지 않다. 따라서 이 항이 정확히 어떤 역할을 하는지, 실제 기여도가 큰지, ablation으로 검증되었는지 판단하기 어렵다. 이 부분은 방법론 설명의 투명성 측면에서 아쉽다.

넷째, baseline 비교가 제한적이다. 논문은 VCL을 관련연구에서 언급하지만, 제공된 실험 텍스트에는 직접적인 비교 결과가 없다. 또한 task-specific head나 small memory buffer를 허용하는 강력한 최신 기법들과의 비교도 보이지 않는다. 따라서 “state of the art 수준으로 얼마나 경쟁력 있는가”까지는 본문만으로 단정하기 어렵다.

비판적으로 보면, 이 논문은 **생성 replay + joint latent model**이라는 흥미로운 방향을 제시하지만, 그 효과를 입증하는 실험 범위는 아직 넓지 않다. 따라서 이 연구는 완성형 솔루션이라기보다는, continual classification을 generative modeling 관점으로 확장하는 **유망한 초기 제안**으로 보는 것이 적절하다.

## 6. 결론

이 논문은 continual classification 문제를 해결하기 위해, VAE 기반 생성모델에 classifier를 결합하고 $\log p(x,y)$에 대한 새로운 variational bound를 유도한 **CCL-GM**을 제안한다. 이 모델은 공통 latent variable $z$를 통해 입력 reconstruction과 label prediction을 동시에 학습하며, student-teacher 구조를 통해 과거 태스크 데이터를 생성해 replay함으로써 catastrophic forgetting을 줄이려 한다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, 분류와 생성을 동시에 다루는 joint variational objective를 제안했다. 둘째, 과거 데이터를 저장하지 않고도 teacher 생성 샘플을 통해 continual learning을 수행했다. 셋째, permuted MNIST 및 간단한 다중 데이터셋 시퀀스에서 baseline보다 더 안정적으로 성능을 유지하는 초기 결과를 보였다.

실제 적용 관점에서 보면, 이 연구는 데이터 저장이 어렵거나 과거 태스크 원본 접근이 제한된 상황에서 의미가 있다. 또한 향후 연구에서는 더 강력한 생성모델, 더 복잡한 비전 벤치마크, 더 충실한 ablation study와 결합될 경우, generative replay 기반 continual learning의 중요한 출발점이 될 수 있다. 특히 “좋은 latent representation이 분류와 생성 모두를 동시에 지지해야 한다”는 이 논문의 관점은 이후 연구에도 충분히 영향을 줄 수 있는 아이디어다.
