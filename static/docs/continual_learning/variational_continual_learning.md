# Variational Continual Learning

* **저자**: Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, Richard E. Turner
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1710.10628](https://arxiv.org/abs/1710.10628)

## 1. 논문 개요

이 논문은 continual learning, 즉 데이터가 시간에 따라 순차적으로 들어오고 과거 태스크를 다시 전부 보지 못하는 상황에서, 신경망이 새로운 태스크를 학습하면서도 예전 태스크를 잊지 않도록 하는 방법을 제안한다. 논문의 핵심 제안은 **Variational Continual Learning (VCL)**이며, 이는 Bayesian inference의 온라인 업데이트 관점을 신경망의 variational inference에 직접 연결한 방법이다.

연구 문제는 명확하다. 일반적인 딥러닝 모델은 새로운 태스크를 순차적으로 학습할 때 기존에 학습한 내용을 급격히 잊어버리는 **catastrophic forgetting** 문제가 심하다. 이는 continual learning의 가장 대표적인 난제다. 기존 연구들은 주로 파라미터가 많이 바뀌지 않도록 규제항을 두는 방식, 예를 들면 EWC나 SI 같은 방법을 사용했지만, 이들은 대체로 하이퍼파라미터에 민감하고, 이전 지식을 얼마나 강하게 유지할지 사람이 조정해야 하며, 불확실성 추정도 제한적이다.

이 논문은 이러한 문제를 “원래 Bayesian inference가 연속 학습에 자연스럽게 맞는 틀”이라는 관점에서 다시 본다. 이미 관측한 데이터로 얻은 posterior를 다음 단계의 prior처럼 사용하고, 새 데이터의 likelihood를 곱해서 posterior를 갱신하면 된다. 즉, continual learning을 별도의 특수 문제로 보기보다, **온라인 근사 Bayesian 추론** 문제로 재해석한다. 이 관점은 이론적으로 매우 자연스럽고, 과거 지식을 “중요한 파라미터를 덜 바꾸게 하는 정규화”로 근사하는 기존 접근보다 더 원리적이다.

또한 이 논문은 단지 분류기 같은 discriminative model뿐 아니라 VAE 기반 deep generative model에도 같은 프레임워크를 적용했다는 점이 중요하다. 당시 continual learning 연구는 주로 분류 문제에 집중되어 있었는데, 이 논문은 생성 모델까지 범위를 넓혀 방법의 일반성을 보여준다.

## 2. 핵심 아이디어

논문의 중심 직관은 다음과 같다. 새로운 데이터셋 $\mathcal{D}_T$가 도착했을 때 우리가 진짜로 원하는 것은

$$
p(\boldsymbol{\theta} \mid \mathcal{D}_{1:T}) \propto p(\boldsymbol{\theta} \mid \mathcal{D}_{1:T-1}) , p(\mathcal{D}_T \mid \boldsymbol{\theta})
$$

와 같은 posterior 업데이트이다. 즉, 이전까지의 posterior가 과거 데이터의 요약 역할을 하고, 여기에 현재 데이터의 likelihood를 곱하면 된다. continual learning은 본질적으로 이 재귀식 하나로 표현된다.

문제는 신경망에서 정확한 posterior가 계산 불가능하다는 점이다. 그래서 논문은 이전 posterior를 tractable한 variational distribution $q_{t-1}(\boldsymbol{\theta})$로 근사해 두고, 새 태스크가 오면 이를 prior처럼 사용해 새 posterior $q_t(\boldsymbol{\theta})$를 구한다. 이것이 VCL이다. 즉, 기존 Bayesian neural network의 variational inference를 batch setting에서 online/continual setting으로 옮긴 셈이다.

기존 접근과의 차별점은 크게 세 가지다.

첫째, EWC나 SI처럼 “이전 태스크에서 중요했던 파라미터는 덜 바꾸자”는 휴리스틱을 직접 설계하지 않는다. 대신 이전 posterior 전체를 유지한다. 이 posterior는 평균만이 아니라 분산도 포함하므로, 어떤 파라미터가 확실히 중요한지, 어떤 파라미터는 아직 불확실한지까지 표현할 수 있다.

둘째, 목적함수에 별도의 task-specific regularization coefficient $\lambda$를 두지 않는다. EWC, SI, LP는 논문에서도 여러 $\lambda$를 탐색해 최적값을 골랐지만, VCL은 objective 자체가 하이퍼파라미터 없이 정해진다. 온라인 상황에서 검증셋으로 $\lambda$를 튜닝하기 어렵다는 점을 생각하면 이것은 실용적으로도 강한 장점이다.

셋째, 반복 근사에서 발생할 수 있는 정보 손실을 줄이기 위해 **coreset**이라는 작은 episodic memory를 결합한다. 이는 각 태스크에서 소수의 대표 샘플만 저장해 두고, 예측 직전에 그 정보까지 다시 posterior에 반영하는 방식이다. 따라서 VCL은 pure Bayesian update와 rehearsal-like memory를 자연스럽게 결합한 구조를 갖는다.

## 3. 상세 방법 설명

### 3.1 Bayesian continual learning의 기본 재귀

논문은 먼저 연속적으로 도착하는 데이터셋 $\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_T$에 대해 posterior가 다음처럼 재귀적으로 갱신된다고 설명한다.

$$
p(\boldsymbol{\theta} \mid \mathcal{D}_{1:T})
\propto
p(\boldsymbol{\theta}) \prod_{t=1}^{T} p(\mathcal{D}_t \mid \boldsymbol{\theta})
\propto
p(\boldsymbol{\theta} \mid \mathcal{D}_{1:T-1}) p(\mathcal{D}_T \mid \boldsymbol{\theta})
$$

이 식의 의미는 단순하다. 과거 데이터 전체를 다시 저장하지 않아도, 이전 posterior만 유지하면 새 데이터가 도착했을 때 업데이트할 수 있다는 것이다. continual learning에 매우 이상적인 형태다.

하지만 신경망에서는 $p(\boldsymbol{\theta} \mid \mathcal{D}_{1:T})$를 정확히 구할 수 없으므로, 논문은 이를 variational approximation $q_t(\boldsymbol{\theta})$로 대체한다.

### 3.2 VCL의 variational projection

VCL은 각 단계에서 다음 최적화 문제를 푼다.

$$
q_t(\boldsymbol{\theta}) =
\arg\min_{q \in \mathcal{Q}}
\mathrm{KL}
\Bigg(q(\boldsymbol{\theta});\Big|;
\frac{1}{Z_t} q_{t-1}(\boldsymbol{\theta}) p(\mathcal{D}_t \mid \boldsymbol{\theta})
\Bigg)
$$

여기서 $\mathcal{Q}$는 근사 posterior family이고, 논문에서는 주로 mean-field Gaussian family를 사용한다. $q_0(\boldsymbol{\theta}) = p(\boldsymbol{\theta})$로 시작한다.

이 식은 해석이 중요하다. 새로운 posterior를 구할 때, 이전 posterior $q_{t-1}$를 그대로 prior처럼 사용하고, 현재 데이터 likelihood를 곱한 뒤, 그것에 가장 가까운 분포를 variational family 안에서 찾는 것이다. 즉, batch variational inference를 매 단계 온라인으로 반복하는 구조다.

이 식에서 normalizing constant $Z_t$는 실제로 계산할 필요가 없다. KL 최소화 문제에서 상수항으로 사라지기 때문이다.

### 3.3 반복 근사 오차와 coreset memory

논문은 중요한 한계를 스스로 지적한다. 각 단계에서 posterior를 근사하고, 그 근사 posterior를 다시 다음 단계의 입력으로 쓰기 때문에 오차가 누적될 수 있다. 특히 Monte Carlo approximation까지 쓰면 더 많은 정보가 소실될 수 있다. 이 문제가 결국 forgetting으로 이어질 수 있다.

이를 완화하기 위해 coreset을 도입한다. coreset은 이전 태스크들에서 고른 소수의 대표 샘플 집합 $C_t$이다. 논문은 이를 episodic memory와 유사한 개념으로 본다.

핵심 아이디어는 전체 posterior를 두 부분으로 나누는 것이다.

$$
p(\boldsymbol{\theta} \mid \mathcal{D}_{1:t})
\propto
p(\boldsymbol{\theta} \mid \mathcal{D}_{1:t} \setminus C_t); p(C_t \mid \boldsymbol{\theta})
\approx
\tilde{q}_t(\boldsymbol{\theta}), p(C_t \mid \boldsymbol{\theta})
$$

여기서 $\tilde{q}_t(\boldsymbol{\theta})$는 현재 coreset에 포함되지 않은 데이터들만 반영한 posterior 근사다. 이 분포는 다음과 같이 재귀적으로 업데이트된다.

$$
\tilde{q}_t(\boldsymbol{\theta}) =
\mathrm{proj}
\big(
\tilde{q}_{t-1}(\boldsymbol{\theta})
,
p(\mathcal{D}_t \cup C_{t-1} \setminus C_t \mid \boldsymbol{\theta})
\big)
$$

그리고 예측 직전에는 현재 coreset likelihood를 다시 반영해 최종 분포 $q_t(\boldsymbol{\theta})$를 만든다.

$$
q_t(\boldsymbol{\theta}) =
\mathrm{proj}
\big(
\tilde{q}_t(\boldsymbol{\theta}) p(C_t \mid \boldsymbol{\theta})
\big)
$$

즉, propagation에는 non-coreset 데이터와 coreset에서 빠져나온 데이터가 쓰이고, prediction 직전에는 현재 coreset 정보가 다시 들어간다. 이 구조는 “작은 기억 장치로 과거 태스크 정보를 refresh한다”는 효과를 낸다.

논문은 coreset 선택 방법으로 두 가지를 예시한다. 하나는 random sampling이고, 다른 하나는 입력 공간 전체를 잘 덮도록 점들을 고르는 **K-center** 방식이다.

### 3.4 discriminative model에 대한 적용

논문은 먼저 deep fully-connected classifier에 VCL을 적용한다. continual learning의 태스크 구조에 따라 두 가지 네트워크 구조를 구분한다.

하나는 **single-head network**로, 모든 태스크가 동일한 출력 공간을 공유하거나 단순한 distribution shift만 있을 때 사용한다. 다른 하나는 **multi-head network**로, 입력 쪽은 shared parameters를 쓰고, 태스크마다 다른 output head를 둔다. Split MNIST 같은 설정은 multi-head가 적합하다.

파라미터 전체를 $\boldsymbol{\theta}$라고 하고, 이에 대해 mean-field Gaussian posterior를 둔다.

$$
q_t(\boldsymbol{\theta}) = \prod_{d=1}^{D} \mathcal{N}(\theta_{t,d}; \mu_{t,d}, \sigma_{t,d}^2)
$$

태스크 $k$를 아직 보지 않았다면 그 태스크의 head 파라미터 posterior는 prior 그대로 유지된다. 따라서 새로운 태스크가 등장할 때 model structure를 점진적으로 확장하기 쉽다.

분류 모델의 학습 objective는 online variational free energy 또는 online marginal likelihood lower bound 형태로 쓸 수 있다.

$$
\mathcal{L}^{t}_{\mathrm{VCL}}(q_t(\boldsymbol{\theta})) =
\sum_{n=1}^{N_t}
\mathbb{E}_{\boldsymbol{\theta} \sim q_t(\boldsymbol{\theta})}
\left[
\log p(y_t^{(n)} \mid \boldsymbol{\theta}, \mathbf{x}_t^{(n)})
\right] - \mathrm{KL}(q_t(\boldsymbol{\theta}) | q_{t-1}(\boldsymbol{\theta}))
$$

이 식은 직관적으로 두 항의 균형이다. 첫 번째 항은 현재 태스크 데이터를 잘 설명하도록 만들고, 두 번째 항은 새 posterior가 이전 posterior에서 너무 멀어지지 않도록 한다. 중요한 점은 이 KL이 단순한 가중치 decay가 아니라, 이전 posterior 분산을 고려한 분포 수준의 제약이라는 점이다.

논문에 따르면 KL 항은 Gaussian 사이에서 closed-form으로 계산할 수 있다. 반면 기대 log-likelihood는 신경망 때문에 닫힌형태가 아니므로 Monte Carlo로 근사하고, gradient 계산에는 **local reparameterization trick**을 사용한다.

### 3.5 generative model에 대한 적용

논문은 VAE에도 VCL을 적용한다. 일반적인 VAE는 관측 $\mathbf{x}$와 latent variable $\mathbf{z}$에 대해

$$
p(\mathbf{x} \mid \mathbf{z}, \boldsymbol{\theta}) p(\mathbf{z})
$$

의 구조를 갖는다. batch VAE는 보통 파라미터 $\boldsymbol{\theta}$를 point estimate처럼 학습하지만, continual learning에서는 파라미터 uncertainty가 중요하므로 그것만으로는 부족하다고 본다.

기존 VAE objective는 다음 lower bound이다.

$$
\mathcal{L}_{\mathrm{VAE}}(\boldsymbol{\theta}, \boldsymbol{\phi}) =
\sum_{n=1}^{N} \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}^{(n)} \mid \mathbf{x}^{(n)})}
\left[
\log \frac{p(\mathbf{x}^{(n)} \mid \mathbf{z}^{(n)}, \boldsymbol{\theta}) p(\mathbf{z}^{(n)})}{q_{\boldsymbol{\phi}}(\mathbf{z}^{(n)} \mid \mathbf{x}^{(n)})}
\right]
$$

VCL에서는 $\boldsymbol{\theta}$에 대해 posterior $q_t(\boldsymbol{\theta})$를 유지하고, 각 태스크마다 encoder 파라미터 $\boldsymbol{\phi}$를 따로 둔다. 이때 objective는 다음처럼 된다.

$$
\mathcal{L}^{t}_{\mathrm{VCL}}(q_t(\boldsymbol{\theta}), \boldsymbol{\phi}) =
\mathbb{E}_{q_t(\boldsymbol{\theta})}
\Bigg\lbrace
\sum_{n=1}^{N_t} \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}_t^{(n)} \mid \mathbf{x}_t^{(n)})}
\left[
\log \frac{p(\mathbf{x}_t^{(n)} \mid \mathbf{z}_t^{(n)}, \boldsymbol{\theta}) p(\mathbf{z}_t^{(n)})}{q_{\boldsymbol{\phi}}(\mathbf{z}_t^{(n)} \mid \mathbf{x}_t^{(n)})}
\right]
\Bigg\rbrace - \mathrm{KL}(q_t(\boldsymbol{\theta}) | q_{t-1}(\boldsymbol{\theta}))
$$

즉, VAE의 ELBO 바깥에 parameter posterior update를 위한 KL regularization이 붙는 구조다. 이 방식은 생성 모델에서도 과거 태스크의 생성 능력을 유지하도록 돕는다.

논문은 generative multi-head 구조에 대해 두 가지 가능성을 언급한다. 하나는 latent $\mathbf{z}$에서 intermediate representation $\mathbf{h}$로 가는 부분을 태스크별 head로 두고, $\mathbf{h}$에서 관측 $\mathbf{x}$로 가는 lower-level network를 공유하는 방식이다. 다른 하나는 그 반대다. 논문은 실험적으로 첫 번째 구조가 더 적절하다고 보고 그것만 사용한다. 이유는 공통적인 low-level primitive, 예를 들면 필기 획 같은 구조를 shared part가 담당하고, 태스크별 차이는 high-level part가 담당하는 편이 transfer에 유리하기 때문이다.

### 3.6 기존 방법들과의 관계

논문은 EWC, LP, SI를 모두 하나의 큰 틀에서 비교한다. 이들 방법은 대체로 다음 형태의 regularized objective를 쓴다.

$$
\mathcal{L}^{t}(\boldsymbol{\theta}) =
\sum_{n=1}^{N_t} \log p(y_t^{(n)} \mid \boldsymbol{\theta}, \mathbf{x}_t^{(n)})
-\frac{1}{2}
\lambda_t
(\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})^\intercal
\Sigma_{t-1}^{-1}
(\boldsymbol{\theta} - \boldsymbol{\theta}_{t-1})
$$

여기서 $\Sigma_{t-1}^{-1}$는 파라미터 중요도를 나타내는 대각 행렬 비슷한 역할을 하고, $\lambda_t$는 과거 정보를 얼마나 강하게 유지할지 조절하는 하이퍼파라미터다.

VCL은 겉보기에는 이와 비슷하게 “현재 데이터 적합도 - 이전 posterior로부터의 이탈 비용” 구조를 가지지만, 실제로는 분포 전체를 업데이트한다는 점이 다르다. 따라서 평균뿐 아니라 분산도 유지되고, 학습과 추론 모두에서 posterior averaging이 가능하다. 논문은 이것이 uncertainty estimation 측면에서 Laplace나 MAP 계열보다 더 낫다고 주장한다.

## 4. 실험 및 결과

논문은 세 가지 discriminative task와 두 가지 generative task에서 VCL을 평가한다. 비교 대상은 주로 **EWC**, **diagonal Laplace Propagation (LP)**, **Synaptic Intelligence (SI)**이다. 중요한 점은 비교 방법들은 $\lambda$를 탐색해서 최적화했지만, VCL은 objective가 하이퍼파라미터 프리라는 점이다.

### 4.1 Permuted MNIST

이 실험에서는 각 태스크가 MNIST 이미지에 서로 다른 고정 permutation을 적용한 데이터셋이다. 이는 continual learning에서 매우 널리 쓰이는 benchmark다. 모든 방법에 대해 두 개의 hidden layer, 각 100 units, ReLU를 사용하는 fully-connected single-head network를 썼다. VCL은 세 가지 버전으로 평가된다. coreset 없는 VCL, random coreset VCL, K-center coreset VCL이다. coreset은 태스크당 200개 샘플을 선택했다.

결과는 매우 강하다. 10개 태스크 후 평균 정확도는 VCL이 약 **90%** 이고, EWC는 **84%**, SI는 **86%**, LP는 **82%** 다. coreset을 결합하면 성능이 더 좋아져 random과 K-center 모두 약 **93%** 정도를 달성한다.

이 결과가 의미하는 바는 분명하다. 단순히 예전 파라미터를 보존하려는 정규화 방식보다, posterior uncertainty를 직접 유지하는 VCL이 장기적인 태스크 축적에 더 유리하다는 것이다. 또한 coreset 자체만으로는 성능이 낮지만, VCL과 결합하면 추가 개선이 있다는 점도 중요하다. 즉, “베이지안 업데이트 + 소규모 기억”의 조합이 상호보완적이라는 뜻이다.

논문은 coreset 크기 효과도 분석한다. 태스크당 coreset 크기를 5,000개까지 늘리면 10개 태스크 후 정확도가 **95.5%** 로 올라간다. 이는 vanilla VCL의 90%보다 상당히 높다. coreset 크기가 커질수록 성능이 향상되지만, 충분히 커지면 포화되는 모습도 보인다. 이는 당연히 coreset이 거의 원래 태스크를 대표할 정도로 커지면 replay 효과가 커지기 때문이다.

### 4.2 Split MNIST

Split MNIST는 숫자 쌍을 이진 분류 태스크로 나눈 설정이다. 태스크 순서는 0/1, 2/3, 4/5, 6/7, 8/9 이다. 이 경우 출력 구조가 태스크별로 분리되므로 multi-head network를 사용한다. hidden layer는 두 개, 각 256 units, ReLU를 사용했다. coreset은 태스크당 40개 샘플이다.

이 실험에서 VCL은 EWC와 LP를 크게 앞서고, SI보다는 약간 낮다. 5개 태스크 후 평균 정확도는 VCL이 **97.0%**, EWC가 **63.1%**, SI가 **98.9%**, LP가 **61.2%** 다. coreset을 추가하면 VCL은 약 **98.4%** 까지 오른다.

즉, 이 실험에서는 SI가 가장 강력한 baseline이었고, VCL은 그와 거의 비슷한 수준까지 올라간다. 중요한 것은 VCL이 별도의 $\lambda$ 튜닝 없이 이 수준을 달성했다는 점이다. 또한 batch VI로 full dataset 전체를 학습한 upper bound와 비교해도 상당히 근접한 성능을 보인다.

### 4.3 Split notMNIST

이 실험은 Split MNIST보다 어렵다. notMNIST는 A-J 문자 이미지 데이터셋이고, 태스크는 A/F, B/G, C/H, D/I, E/J의 다섯 개 이진 분류다. 네 개의 hidden layer, 각 150 units를 사용하는 더 깊은 네트워크를 사용한다.

결과는 VCL이 여전히 강력하지만, SI와 경쟁적인 수준이다. 5개 태스크 후 평균 정확도는 VCL이 **92.0%**, EWC가 **71%**, SI가 **94%**, LP가 **63%** 다. random coreset을 추가하면 VCL은 **96%** 까지 향상된다.

이는 데이터가 더 복잡해지고 네트워크가 깊어져도 VCL이 안정적으로 동작함을 보여준다. 동시에 SI가 일부 분류 설정에서는 여전히 매우 강한 baseline이라는 점도 보여준다.

### 4.4 Deep Generative Models: MNIST / notMNIST generation

논문은 continual learning을 VAE 기반 생성 모델로 확장한 점을 큰 기여로 내세운다. 각 태스크는 한 클래스씩 순차적으로 들어온다. MNIST는 digit 0, 1, ..., 9가 순서대로 도착하고, notMNIST도 A-J가 순서대로 도착한다. 생성기와 태스크별 모듈은 한 개의 hidden layer와 500 units를 사용하고, latent $z$ 차원은 50, intermediate representation $h$ 차원은 500이다. encoder는 task-specific이다.

비교 방법은 naive online VAE, LP, EWC, SI, VCL이다. naive online learning은 catastrophic forgetting이 너무 심해서 수치 결과조차 생략되었다. 즉, 가장 최근 태스크만 생성하게 된다.

정성적 샘플을 보면 LP, EWC, SI, VCL은 이전 태스크를 어느 정도 기억하지만, **SI와 VCL이 시각 품질이 가장 좋다**고 보고한다.

정량 평가는 두 지표로 했다. 하나는 importance sampling 기반 **test log-likelihood (test-LL)**이고, 높을수록 좋다. 다른 하나는 **classifier uncertainty**로, 생성된 샘플을 별도의 분류기에 넣었을 때 해당 태스크의 one-hot 정답 분포와 예측 분포 간 KL divergence를 측정한다. 낮을수록 생성 품질이 좋다.

논문은 LP와 EWC가 비슷한 성능을 보인다고 말한다. 둘 다 사실상 유사한 $\Sigma_t$ 구조를 쓰기 때문이라는 해석이다. EWC는 SI보다 많이 뒤처지고, VCL은 **SI와 비슷하거나 약간 더 낫다**고 보고한다. 특히 장기 기억 측면에서 VCL이 더 우수하여 전체적인 test-LL와 classifier uncertainty에서 좋은 결과를 낸다. MNIST에서는 digit 0에서 digit 1로 넘어갈 때 LP와 EWC 성능이 크게 무너지는 경향도 관찰된다. 또한 SI는 task 7 이후 high test-LL을 유지하지 못했다고 기술한다.

이 부분은 논문의 매우 중요한 포인트다. 분류 성능뿐 아니라 **생성 능력의 보존**이라는 더 어려운 문제에서도 VCL이 효과적이라는 근거를 제공한다.

### 4.5 실험 해석

전체적으로 보면, VCL은 분류 문제에서는 대부분 EWC와 LP를 크게 앞서고, SI와는 비슷하거나 상황에 따라 조금 낮거나 높다. 하지만 생성 문제까지 포함하면 VCL의 일반성과 안정성이 더 돋보인다. 논문은 이를 통해 VCL이 “state-of-the-art performance”를 보였다고 주장한다. 엄밀히 보면 모든 벤치마크에서 항상 1등은 아니지만, 다양한 문제를 하나의 원리적 프레임워크로 다루며 매우 강한 성능을 보였다는 해석은 타당하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 continual learning을 ad hoc regularization 문제가 아니라 **온라인 Bayesian 추론 문제**로 재정의했다는 점이다. 이 재해석은 이론적으로 깔끔하고, 왜 이전 지식을 유지해야 하는지, 왜 어떤 파라미터는 덜 바꾸고 어떤 파라미터는 더 바꿔도 되는지를 불확실성 관점에서 설명할 수 있게 해준다.

두 번째 강점은 방법의 일반성이다. 논문은 discriminative classifier와 VAE 기반 generative model 모두에 같은 프레임워크를 적용했다. 많은 continual learning 방법이 분류기 중심으로 설계된 것과 비교하면 상당히 넓은 적용 범위다.

세 번째 강점은 objective가 하이퍼파라미터 프리라는 점이다. EWC, LP, SI는 $\lambda$ 선택에 따라 성능이 크게 달라질 수 있고, 실제 논문도 baseline 성능을 위해 extensive hyper-parameter search를 했다고 명시한다. 반면 VCL은 이런 조정 없이도 경쟁력 있는 성능을 낸다.

네 번째 강점은 coreset memory를 매우 자연스럽게 결합했다는 점이다. 단순 replay가 아니라 variational posterior update 과정과 연결해 설명했기 때문에, 메모리 기반 보강이 이론적 틀 안에 잘 들어간다.

하지만 한계도 분명하다.

첫째, 실제 posterior를 mean-field Gaussian으로 근사한다. 이는 계산 효율은 좋지만, 파라미터 간 상관관계를 무시한다. continual learning에서 중요한 공분산 구조를 놓칠 가능성이 있다. 논문 부록에서도 선형 회귀 예시를 통해 diagonal Gaussian 근사에서 batch와 sequential solution이 달라질 수 있음을 보여준다. 즉, 근사 자체의 표현력 한계가 존재한다.

둘째, 반복 근사 오차 누적 문제가 근본적으로 해결된 것은 아니다. coreset이 이를 완화하지만, 이는 결국 일부 데이터를 저장하는 방식이다. 따라서 strict한 의미의 “데이터를 전혀 다시 보지 않는” setting에서는 coreset 없는 VCL의 성능이 제한될 수 있다.

셋째, task boundary가 비교적 명확한 실험을 다룬다. Permuted MNIST나 Split MNIST처럼 각 태스크가 분리되어 순차적으로 도착하는 설정은 연구용 benchmark로는 적절하지만, 실제 환경의 gradual drift나 unknown task identity 문제와는 다르다. 논문은 general continual learning을 이야기하지만, 실험이 그 전체를 모두 포괄한다고 보기는 어렵다.

넷째, generative model 부분에서 encoder $\boldsymbol{\phi}$는 task-specific으로 두고 있으며, 이를 공유하는 문제는 “조사하지 않았다”고 명시한다. 즉, 생성 모델에서 truly shared continual representation learning까지는 아직 가지 못했다.

다섯째, 논문은 더 유연한 구조 학습, 예를 들면 새로운 태스크가 오면 네트워크 구조를 자동으로 확장하는 문제를 흥미로운 방향으로 언급하지만 실제로 다루지 않는다. 따라서 모델 구조는 사전에 안다고 가정한다.

비판적으로 보면, 이 논문은 원리적으로 매우 설득력이 있지만, 실험 벤치마크가 당시 continual learning 연구의 전형적인 소형 이미지 문제에 머물러 있다. 따라서 대규모 비전이나 복잡한 실제 연속 환경에서 같은 장점이 유지되는지는 이 논문만으로는 판단할 수 없다. 또한 “state-of-the-art”라는 표현은 당시 기준에서는 상당히 타당하지만, Split MNIST처럼 일부 설정에서는 SI가 더 높은 수치를 보였으므로 모든 경우 절대적 우위라고 말하는 것은 조심해야 한다.

## 6. 결론

이 논문은 continual learning을 위한 매우 중요한 방향을 제시한다. 핵심 기여는 **Variational Continual Learning (VCL)**이라는 프레임워크를 통해, 온라인 variational inference를 신경망 continual learning에 적용하고, 이를 분류 모델과 생성 모델 모두에 확장했다는 점이다. 이전 posterior를 다음 단계 prior처럼 사용하는 Bayesian 업데이트 원리를 실제 딥러닝 학습 objective로 구현했고, 여기에 coreset 기반 episodic memory를 결합해 forgetting을 더 줄였다.

방법론적으로 보면, VCL의 목적함수는 현재 데이터에 대한 적합도와 이전 posterior로부터의 KL 제약을 동시에 포함한다. 이는 EWC류의 정규화 방식과 겉보기에는 비슷하지만, 실제로는 파라미터 uncertainty를 명시적으로 유지한다는 점에서 더 풍부하다. 실험적으로도 VCL은 Permuted MNIST, Split MNIST, Split notMNIST, 그리고 VAE 기반 생성 실험에서 강력한 성능을 보였으며, 특히 생성 모델까지 포함한 일반성이 인상적이다.

향후 연구 측면에서 이 논문은 몇 가지 중요한 길을 연다. 더 정교한 posterior family를 사용해 mean-field 한계를 줄일 수 있고, 더 나은 episodic memory나 coreset selection을 결합할 수도 있다. 또한 reinforcement learning, active learning, sequential decision making과 같이 데이터가 본질적으로 순차적으로 들어오는 문제에서 VCL은 매우 자연스러운 기반 기술이 될 수 있다.

종합하면, 이 논문은 continual learning 분야에서 “Bayesian uncertainty를 유지하며 순차 학습하자”는 방향을 신경망 수준에서 실질적으로 구현한 대표적인 작업이다. 이후 Bayesian continual learning 계열 연구의 중요한 출발점으로 볼 수 있다.
