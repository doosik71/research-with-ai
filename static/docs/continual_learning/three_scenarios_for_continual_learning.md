# Three scenarios for continual learning

* **저자**: Gido M. van de Ven, Andreas S. Tolias
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1904.07734](https://arxiv.org/abs/1904.07734)

## 1. 논문 개요

이 논문은 continual learning에서 자주 발생하는 catastrophic forgetting 문제를 다룬다. 저자들은 최근 많은 방법들이 제안되었지만, 서로 다른 평가 프로토콜 때문에 방법 간 성능 비교가 공정하게 이루어지지 못하고 있다고 지적한다. 같은 데이터셋을 사용하더라도 테스트 시점에 task identity가 주어지는지, 주어지지 않는다면 모델이 그것을 직접 추론해야 하는지에 따라 문제의 난이도와 적합한 방법이 크게 달라진다는 것이 논문의 출발점이다.

논문의 핵심 목표는 두 가지다. 첫째, continual learning 실험을 세 가지 시나리오로 구조화하여 평가 체계를 정리하는 것이다. 둘째, 이 세 시나리오 각각에서 대표적인 기존 방법들을 동일한 조건에서 폭넓게 비교하여, 어떤 접근이 어떤 조건에서 실제로 유효한지 밝히는 것이다.

저자들이 제안하는 세 시나리오는 다음과 같다. Task-IL(Task-Incremental Learning)은 테스트 시 task identity가 주어지는 경우다. Domain-IL(Domain-Incremental Learning)은 task identity가 주어지지 않지만, 현재 입력이 어느 task인지 명시적으로 맞힐 필요는 없는 경우다. Class-IL(Class-Incremental Learning)은 task identity가 주어지지 않을 뿐 아니라, 지금까지 본 모든 class 중 무엇인지까지 구별해야 하는 경우다. 저자들은 이 세 구분이 단순한 실험 설정 차이가 아니라, 실제로 방법론의 성패를 가르는 중요한 요인임을 실험으로 보인다.

이 문제의 중요성은 명확하다. continual learning은 여러 작업을 순차적으로 익혀야 하는 실제 지능 시스템의 핵심 능력인데, 평가 설정이 정리되지 않으면 어떤 방법이 진짜 강한지 알기 어렵다. 따라서 이 논문은 새로운 알고리즘을 제안하는 논문이라기보다, 분야의 평가 체계를 정리하고 강한 baseline을 재평가하는 성격의 논문이다. 그럼에도 불구하고, 결론은 매우 실질적이다. 특히 task identity를 추론해야 하는 Class-IL에서는 regularization 기반 방법이 거의 실패하고, replay 계열 접근이 사실상 필요하다는 점을 강하게 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 continual learning 문제를 “하나의 문제”로 다루지 말고, 테스트 시 모델에게 어떤 정보가 주어지는지를 기준으로 서로 다른 문제군으로 분리해야 한다는 것이다. 기존 문헌에서는 흔히 multi-headed와 single-headed 설정 정도로만 구분하거나, 아예 이런 차이를 명시하지 않은 채 성능을 비교하는 경우가 많았다. 저자들은 이런 방식이 충분히 정교하지 않다고 본다.

저자들의 설계 직관은 다음과 같다. catastrophic forgetting 자체는 공통 문제이지만, 테스트 시 task identity가 주어지면 task-specific component를 사용할 수 있고, 출력 공간도 task마다 분리할 수 있으므로 문제가 크게 단순해진다. 반대로 task identity가 주어지지 않으면 모델은 같은 파라미터와 같은 출력 공간에서 더 많은 간섭을 견뎌야 한다. 특히 Class-IL에서는 단순히 예전 task를 잊지 않는 것만으로는 부족하고, 이전 class들과 현재 class들을 하나의 통합된 분류 문제로 다뤄야 한다. 이 차이가 regularization, replay, exemplar 기반 방법의 유효성을 크게 바꾼다.

기존 접근과의 가장 중요한 차별점은 “single-headed vs multi-headed”보다 더 본질적인 기준을 제시했다는 점이다. 저자들은 output layer 구조만으로 시나리오를 구분하는 것은 불충분하다고 주장한다. multi-headed는 task identity를 사용하는 한 구현 방식일 뿐이고, single-headed라 해도 hidden layer 등 다른 방식으로 task identity를 활용할 수 있다. 따라서 진짜 중요한 것은 네트워크 구조 자체가 아니라, 모델이 평가될 때 어떤 정보가 허용되는지다.

또한 저자들은 split MNIST와 permuted MNIST라는 익숙한 두 프로토콜조차도 각각 하나의 고정된 문제로 봐서는 안 된다고 보여준다. 예를 들어 split MNIST는 보통 multi-headed split MNIST 또는 single-headed split MNIST로 쓰이지만, 사실 Domain-IL 방식으로도 정의할 수 있다. 즉, 동일한 데이터 시퀀스라도 평가 질문을 어떻게 던지느냐에 따라 서로 다른 continual learning 문제가 된다. 이 점이 이 논문의 가장 교육적이면서도 영향력 있는 메시지다.

## 3. 상세 방법 설명

이 논문은 새로운 모델을 하나 제안하기보다는, 세 가지 continual learning 시나리오를 정의하고 여러 기존 방법을 동일한 네트워크와 학습 조건에서 비교한다. 따라서 방법 설명은 두 부분으로 나누어 이해하는 것이 좋다. 첫째는 시나리오 자체의 정의이고, 둘째는 비교 대상이 되는 continual learning 전략들의 구조다.

### 3.1 세 가지 시나리오의 정의

#### Task-IL

Task-IL에서는 테스트 시 task identity가 제공된다. 따라서 모델은 “지금 task 3을 풀고 있다”는 사실을 알고 있다. 이 경우 각 task마다 별도의 출력 헤드(multi-headed output)를 둘 수 있고, 심지어 task마다 다른 subnet을 쓰는 것도 가능하다. 예를 들어 split MNIST에서 현재 task가 $(0,1)$ 분류인지 $(2,3)$ 분류인지 알려준다면, 모델은 해당 두 class만 구분하면 된다. 이 설정은 세 시나리오 중 가장 쉽다.

#### Domain-IL

Domain-IL에서는 테스트 시 task identity가 주어지지 않는다. 그러나 모델은 현재 입력이 어느 task인지 명시적으로 맞힐 필요는 없다. 단지 현재 입력에 대해 올바른 label만 내면 된다. 예를 들어 permuted MNIST에서는 각 task마다 다른 pixel permutation이 적용되지만, 결국 예측해야 하는 것은 digit label이다. 입력 분포가 바뀌는 상황에서 같은 문제를 풀어야 하는 형태에 가깝다.

#### Class-IL

Class-IL은 가장 어렵다. 테스트 시 task identity가 없고, 동시에 모델은 지금까지 등장한 모든 class들 중 정답이 무엇인지 맞혀야 한다. split MNIST를 예로 들면, 더 이상 “현재 task는 $(4,5)$ 중 하나”라는 힌트가 없다. 모델은 0부터 9까지 전체 digit 중 어느 것인지 구분해야 한다. 따라서 이전 class와 현재 class를 한 출력 공간에서 함께 다뤄야 한다.

### 3.2 비교한 방법군

저자들은 방법을 크게 task-specific, regularization, replay, exemplar 기반으로 나눈다.

#### (1) Task-specific components

대표 방법은 XdG(Context-dependent Gating)이다. 각 task마다 hidden unit의 일부를 비활성화하여 task별로 서로 다른 subnet을 사용하게 만든다. 즉, 어떤 task에서는 특정 hidden node들을 0으로 막아 사용하지 않는다. 이 방식은 각 task가 서로 다른 파라미터 부분공간을 쓰게 하므로 간섭을 줄인다. 그러나 테스트 시 올바른 gating pattern을 선택하려면 task identity가 반드시 필요하다. 따라서 이 방법은 본질적으로 Task-IL에만 적용 가능하다.

#### (2) Regularized optimization

대표 방법은 EWC, Online EWC, SI다. 이 계열의 공통 아이디어는 이전 task에 중요했던 파라미터를 새 task 학습 중 크게 바꾸지 못하게 하는 것이다.

전체 loss는 다음 형태다.

$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{current}}+\lambda \mathcal{L}_{\text{regularization}}
$$

여기서 $\mathcal{L}_{\text{current}}$는 현재 task의 분류 loss이고, $\mathcal{L}_{\text{regularization}}$은 과거 task를 보존하기 위한 penalty다. $\lambda$는 두 항의 균형을 조절한다.

##### EWC

EWC는 각 task가 끝난 뒤 Fisher Information의 대각 원소를 이용해 파라미터 중요도를 추정한다. 직관적으로, 어떤 파라미터가 이전 task의 예측 확률에 큰 영향을 주었다면 그것은 중요한 파라미터이므로 이후 task에서 덜 움직여야 한다.

EWC의 정규화 항은 다음과 같다.

$$
\mathcal{L}^{(K)}_{\text{regularization}_{\text{EWC}}}(\boldsymbol{\theta}) =
\sum_{k=1}^{K-1}
\left(
\frac{1}{2}
\sum_{i=1}^{N_{\text{params}}}
F_{ii}^{(k)}
(\theta_i-\hat{\theta}_i^{(k)})^2
\right)
$$

여기서 $\hat{\theta}_i^{(k)}$는 task $k$ 학습 직후의 파라미터 값이고, $F*{ii}^{(k)}$는 task $k$에 대한 Fisher Information의 대각 성분이다. 즉, 중요한 파라미터일수록 과거 값에서 멀어질 때 더 큰 페널티를 받는다.

##### Online EWC

기존 EWC는 task 수가 늘수록 quadratic penalty term도 계속 늘어나 계산과 메모리 비용이 커진다. Online EWC는 이를 줄이기 위해 이전 task들의 Fisher를 누적합으로 요약해 하나의 penalty만 유지한다.

$$
\mathcal{L}^{(K)}_{\text{regularization}_{\text{oEWC}}} =
\sum_{i=1}^{N_{\text{params}}}
\tilde{F}_{ii}^{(K-1)}
(\theta_i-\hat{\theta}_i^{(K-1)})^2
$$

여기서 $\tilde{F}_{ii}^{(K-1)}$는 과거 task들의 Fisher를 decay와 함께 누적한 값이다.

##### SI

SI(Synaptic Intelligence)는 Fisher를 따로 계산하지 않고, 실제 학습 경로에서 각 파라미터가 loss 감소에 얼마나 기여했는지를 누적해 중요도를 추정한다. 학습 중 parameter update와 gradient를 곱해 contribution을 누적하고, 이를 task 전체에서 해당 파라미터가 얼마나 변했는지로 정규화한다.

핵심 누적량은 다음과 같다.

$$
\omega_i^{(k)} =
\sum_{t=1}^{N_{\text{iters}}}
\left(\theta_i[t^{(k)}]-\theta_i[(t-1)^{(k)}]\right)
\frac{-\delta \mathcal{L}_{\text{total}}[t^{(k)}]}{\delta \theta_i}
$$

이후 중요도는

$$
\Omega_i^{(K-1)} =
\sum_{k=1}^{K-1}
\frac{\omega_i^{(k)}}{(\Delta_i^{(k)})^2+\xi}
$$

로 정의되고, 최종 regularization term은

$$
\mathcal{L}^{(K)}_{\text{regularization}_{\text{SI}}} =
\sum_{i=1}^{N_{\text{params}}}
\Omega_i^{(K-1)}
(\theta_i-\hat{\theta}_i^{(K-1)})^2
$$

가 된다.

이 세 방법 모두 본질적으로 “이전 task에 중요했던 파라미터를 보존하자”는 접근이다. 하지만 논문 실험 결과에 따르면, 이 접근만으로는 특히 Class-IL을 해결하지 못한다.

#### (3) Replay-based methods

Replay 계열은 새 task를 학습할 때 과거 task를 대표하는 pseudo-data를 함께 사용한다. 저자들은 이 계열이 시나리오 전반에서 가장 강력하다고 본다.

전체 loss는 현재 task loss와 replay loss의 가중합이다.

$$
\mathcal{L}_{\text{total}} =
\frac{1}{N_{\text{tasks so far}}}\mathcal{L}_{\text{current}} +
\left(1-\frac{1}{N_{\text{tasks so far}}}\right)\mathcal{L}_{\text{replay}}
$$

즉, 시간이 갈수록 현재 task 비중은 줄고 과거 task 보존 비중은 상대적으로 커진다.

##### LwF

LwF(Learning without Forgetting)는 과거 task 데이터를 저장하거나 생성하지 않고, 현재 task의 입력을 이전 모델로 통과시켜 soft target을 만든 뒤 distillation으로 과거 지식을 유지하려 한다. 즉, 현재 task 이미지 자체를 과거 모델의 출력 분포와 함께 replay처럼 사용한다.

Distillation에서 soft target은

$$
\tilde{y}_c = p_{\hat{\boldsymbol{\theta}}^{(K-1)}}^{T}(Y=c|\boldsymbol{x})
$$

이고, loss는

$$
\mathcal{L}_{\text{distillation}}(\boldsymbol{x},\tilde{\boldsymbol{y}};\boldsymbol{\theta}) =
-T^2 \sum\_{c=1}^{N\_{\text{classes}}}
\tilde{y}_c \log p_{\boldsymbol{\theta}}^{T}(Y=c|\boldsymbol{x})
$$

이다. 여기서 $T$는 temperature다. 이 논문에서는 $T=2$를 사용했다.

##### DGR

DGR(Deep Generative Replay)은 별도의 generative model을 두어 과거 task의 입력을 생성한다. 생성된 샘플에 대해, 이전 classifier가 예측한 hard target을 붙여 replay한다.

##### DGR+distill

이 방법은 DGR과 LwF를 결합한 형태다. 생성된 샘플을 replay하되, label을 hard target이 아니라 soft target으로 준다. 논문 실험에서는 이 조합이 매우 강력하게 나타난다.

#### (4) iCaRL

iCaRL은 exemplar를 저장하고, 학습 시 replay와 distillation을 쓰며, 테스트 시에는 nearest-class-mean 분류를 사용하는 방법이다. 이 논문에서는 Class-IL에만 적용했다.

iCaRL의 학습용 sigmoid 출력은 class별 binary probability를 낸다.

$$
p_{\boldsymbol{\theta}}^{c}(\boldsymbol{x}) =
\frac{1}{1+e^{-\boldsymbol{w}_c^T \psi*{\boldsymbol{\phi}}(\boldsymbol{x})}}
$$

이전 class에는 이전 모델의 출력 확률을 soft target으로, 현재 class에는 hard target을 사용하여 다음 loss를 최적화한다.

$$
\mathcal{L}_{\text{iCaRL}}(\boldsymbol{x},\boldsymbol{\bar{y}};\boldsymbol{\theta}) =
-\sum\_{c=1}^{N_{\text{classes so far}}}
\left[
\bar{y}_c \log p_{\boldsymbol{\theta}}^c(\boldsymbol{x})
+(1-\bar{y}_c)\log(1-p_{\boldsymbol{\theta}}^c(\boldsymbol{x}))
\right]
$$

테스트 시에는 softmax classifier 대신 exemplar들의 feature 평균을 class prototype처럼 사용한다. 입력 $\boldsymbol{x}$에 대해 가장 가까운 class mean을 찾는다.

$$
y^\* =
\operatorname*{argmin}_{c=1,\ldots,N*{\text{classes so far}}}
\left|
\psi_{\boldsymbol{\phi}}(\boldsymbol{x})-\boldsymbol{\mu}_c
\right|
$$

즉, representation space에서 prototype classification을 수행한다.

### 3.3 생성 모델(VAE)

DGR과 DGR+distill을 위해 별도의 VAE를 사용했다. encoder와 decoder는 모두 2-layer MLP이고 latent dimension은 100이다. 생성 모델의 loss는 reconstruction term과 latent regularization term의 합이다.

$$
\mathcal{L}_{\text{generative}}(\boldsymbol{x};\boldsymbol{\phi},\boldsymbol{\psi}) =
\mathcal{L}_{\text{recon}}(\boldsymbol{x};\boldsymbol{\phi},\boldsymbol{\psi})
+\mathcal{L}_{\text{latent}}(\boldsymbol{x};\boldsymbol{\phi})
$$

latent term은 VAE의 KL 형태이며,

$$
\mathcal{L}_{\text{latent}}(\boldsymbol{x};\boldsymbol{\phi}) =
\frac{1}{2}
\sum_{j=1}^{100}
\left(
1+\log((\sigma_j^{(\boldsymbol{x})})^2)-(\mu_j^{(\boldsymbol{x})})^2-(\sigma_j^{(\boldsymbol{x})})^2
\right)
$$

reconstruction term은 binary cross entropy다.

이 생성 모델도 continual learning 방식으로 순차 학습되며, 이전 task의 생성 샘플을 다시 replay하여 generator 자체도 forgetting을 줄인다. 즉, classifier만 replay하는 것이 아니라 generator도 replay로 유지한다.

### 3.4 실험 프로토콜과 학습 설정

저자들은 모든 방법에 대해 동일한 분류기 backbone을 썼다. split MNIST에서는 hidden layer 두 개, 각각 400 unit의 MLP를 사용했고, permuted MNIST에서는 각각 1000 unit을 사용했다. hidden activation은 ReLU다.

* split MNIST: 5개 task, 각 task는 2-way classification
* permuted MNIST: 10개 task, 각 task는 10-way classification

학습은 ADAM optimizer를 사용했다. split MNIST는 task당 2000 iteration, learning rate는 0.001이다. permuted MNIST는 task당 5000 iteration, learning rate는 0.0001이다. iteration마다 current batch 128개를 사용했고, replay가 있을 경우 replay batch 128개를 추가했다.

중요한 점은 시나리오에 따라 active output node가 다르다는 것이다. Task-IL에서는 현재 task의 output head만 활성화된다. Domain-IL에서는 항상 같은 output head 전체가 활성화된다. Class-IL에서는 지금까지 본 모든 class의 output node가 활성화된다. 이 차이가 실제 난이도를 결정하는 핵심 요소다.

## 4. 실험 및 결과

### 4.1 데이터셋과 작업 설정

#### Split MNIST

원래의 MNIST 10개 class를 5개의 task로 분할했다. 각 task는 두 개 digit만 구분하는 2-class classification이다. 예를 들어 $(0,1)$, $(2,3)$ 식으로 task가 구성된다. 입력 이미지는 28x28 grayscale 그대로 사용했다.

#### Permuted MNIST

각 task는 모든 10개 digit을 분류하지만, task마다 픽셀 permutation이 다르다. 원래 이미지를 32x32로 zero padding한 뒤, 1024개 픽셀에 무작위 permutation을 적용했다. 따라서 semantic label은 같지만 입력 분포가 크게 달라진다.

### 4.2 비교 대상과 지표

비교한 방법은 다음과 같다.

* baseline: None(fine-tuning), Offline(joint training)
* task-specific: XdG
* regularization: EWC, Online EWC, SI
* replay: LwF, DGR, DGR+distill
* replay + exemplars: iCaRL

성능 평가는 모든 task에 대한 평균 test accuracy로 보고되며, 각 실험은 서로 다른 random seed로 20회 반복되었다. 표에는 mean과 SEM이 함께 제시된다.

### 4.3 Split MNIST 결과

Split MNIST에서는 세 시나리오 간 난이도 차이가 매우 뚜렷했다.

Task-IL에서는 거의 모든 방법이 매우 잘 작동했다. Offline은 $99.66%$, XdG는 $99.10%$, EWC는 $98.64%$, Online EWC는 $99.12%$, SI는 $99.09%$, LwF는 $99.57%$, DGR은 $99.50%$, DGR+distill은 $99.61%$였다. 즉, task identity가 주어질 때는 regularization이나 replay 모두 높은 성능을 낸다.

하지만 Domain-IL로 가면 양상이 달라진다. None은 $59.21%$로 크게 떨어진다. EWC는 $63.95%$, Online EWC는 $64.32%$, SI는 $65.36%$로 regularization 계열은 제한적인 개선만 보인다. LwF는 $71.50%$로 그보다 낫지만 여전히 부족하다. 반면 DGR은 $95.72%$, DGR+distill은 $96.83%$로 매우 강력하다. 즉, task identity가 없기만 해도 replay의 가치가 확연히 드러난다.

가장 중요한 것은 Class-IL 결과다. None은 $19.90%$로 사실상 random 수준이다. EWC는 $20.01%$, Online EWC는 $19.96%$, SI는 $19.99%$로 regularization 계열이 완전히 무너진다. LwF도 $23.85%$에 그친다. 반면 DGR은 $90.79%$, DGR+distill은 $91.79%$, iCaRL은 $94.57%$를 달성한다. 이 표는 논문의 가장 강한 메시지를 담고 있다. task identity를 추론해야 하는 Class-IL에서는 regularization만으로는 거의 해결이 안 되고, replay 또는 exemplar 저장이 사실상 필요하다는 것이다.

저자들이 흥미롭게 지적하는 점은, split MNIST에서는 현재 task 입력을 replay하는 LwF조차 regularization보다 낫다는 점이다. 예를 들어 현재 2와 3을 학습할 때, 그 이미지들에 대해 이전 모델의 출력을 맞추는 것만으로도 0과 1을 완전히 잊는 것을 일부 완화할 수 있었다는 뜻이다.

### 4.4 Permuted MNIST 결과

Permuted MNIST에서는 Task-IL과 Domain-IL의 차이가 split MNIST보다 작다. 이는 output layer에서 task identity를 쓰는 것만으로는 permutation 정보의 이점을 충분히 활용하지 못했기 때문이라고 저자들은 해석한다.

Task-IL 결과를 보면 None은 $81.79%$, EWC는 $94.74%$, Online EWC는 $95.96%$, SI는 $94.75%$, DGR은 $92.52%$, DGR+distill은 $97.51%$다. LwF는 오히려 $69.84%$로 낮다. Offline은 $97.68%$다.

Domain-IL 결과도 유사하다. None은 $78.51%$, EWC는 $94.31%$, Online EWC는 $94.42%$, SI는 $95.33%$, DGR은 $95.09%$, DGR+distill은 $97.35%$다. 즉, permuted MNIST에서는 regularization도 꽤 잘 작동한다. 입력 분포는 달라지지만 label semantics는 동일하므로, split MNIST의 Class-IL처럼 출력 공간 충돌이 심하지 않기 때문으로 이해할 수 있다.

그러나 Class-IL에서는 다시 regularization이 실패한다. None은 $17.26%$, EWC는 $25.04%$, Online EWC는 $33.88%$, SI는 $29.31%$로 낮다. LwF도 $22.64%$다. 반면 DGR은 $92.19%$, DGR+distill은 $96.38%$, iCaRL은 $94.85%$를 달성한다. 즉, permuted MNIST에서도 결론은 같다. Class-IL은 replay 없이는 어렵다.

LwF가 permuted MNIST에서 잘 안 되는 이유에 대해 저자들은 현재 task 입력과 이전 task 입력이 랜덤 permutation 때문에 서로 거의 상관이 없기 때문이라고 해석한다. split MNIST에서는 현재 이미지가 이전 class 정보 보존에 어느 정도 도움을 줄 수 있었지만, permuted MNIST에서는 task 간 입력 구조가 너무 달라 그런 효과가 약하다.

### 4.5 부록의 보충 결과

부록 B에서는 permuted MNIST의 Task-IL에서 task identity를 hidden layer에 넣으면 성능이 더 좋아진다는 것을 보였다. 예를 들어 EWC는 output layer에서 task-ID를 쓰는 경우 $94.74%$였지만, XdG와 결합해 hidden layer에 task-ID를 반영하면 $96.94%$가 된다. None도 $81.79%$에서 $90.41%$로 오른다. 이는 permuted MNIST에서 task identity 정보가 정말 쓸모없었던 것이 아니라, 기존 구현에서 그것을 충분히 활용하지 못했던 것임을 보여준다.

부록 C에서는 exact replay, 즉 실제 저장된 과거 데이터를 replay하는 방법을 exemplar budget에 따라 비교했다. Class-IL에서는 class당 단 1개 exemplar만 저장해도 regularization 기반 방법보다 좋아질 수 있었지만, generative replay 수준에 도달하려면 훨씬 많은 저장 메모리가 필요했다. 특히 permuted MNIST에서는 50,000개를 저장해도 DGR+distill보다 일관되게 낮았다고 보고한다. 즉, 생성 기반 replay가 메모리 효율 측면에서 강력할 가능성을 시사한다.

부록 D에서는 hyperparameter grid search에 대한 논의가 나온다. 저자들은 continual learning에서 validation set을 반복적으로 활용하는 hyperparameter tuning이 엄밀히 말해 “각 task를 한 번만 본다”는 원칙을 위반할 수 있다고 비판한다. 그럼에도 본 논문에서는 비교의 공정성을 위해 grid search를 수행했다. 특히 EWC가 Task-IL split MNIST에서 경쟁력 있게 나온 이유는, 기존보다 훨씬 넓은 범위의 큰 regularization strength를 탐색했기 때문이라고 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 continual learning 분야에서 매우 널리 인용되는 평가 관점을 명확하고 설득력 있게 정리했다는 점이다. 단순히 방법 몇 개를 비교한 수준이 아니라, 왜 서로 다른 논문들의 결과가 직접 비교되기 어려운지 그 구조적 이유를 정확히 짚는다. 특히 “task identity가 주어지는가”, “주어지지 않는다면 추론해야 하는가”라는 구분은 이후 continual learning 문헌 전반에 큰 영향을 준 분류 체계다.

또 다른 강점은 비교 실험의 통제 수준이다. 저자들은 동일한 backbone, 비슷한 학습 조건, 동일한 task protocol 위에서 여러 방법을 폭넓게 비교했다. 이로 인해 방법 간 차이를 비교적 명확하게 해석할 수 있다. 또한 split MNIST와 permuted MNIST를 모두 세 시나리오로 재구성하여, “데이터셋 자체”보다 “평가 질문”이 더 중요하다는 점을 잘 보여준다.

세 번째 강점은 결론이 매우 실용적이라는 점이다. Class-IL에서 regularization 기반 방법이 완전히 무너지고 replay 기반 방법만이 의미 있는 성능을 낸다는 결과는, 이후 연구자들이 어떤 문제를 진짜 어려운 continual learning 문제로 볼 것인지에 중요한 기준을 제공했다. 다시 말해, Task-IL에서 잘 되는 것만으로는 강한 continual learner라고 보기 어렵다는 메시지를 준다.

하지만 한계도 분명하다. 가장 큰 한계는 실험이 전부 MNIST 계열에 머문다는 점이다. 저자들 자신도 인정하듯, MNIST는 생성이 쉬운 데이터다. 따라서 generative replay가 이렇게 잘 되는 것이 복잡한 자연 이미지 분포나 대규모 비전 문제에서도 그대로 유지될지는 알 수 없다. 즉, “replay가 필요하다”는 결론은 상당히 설득력 있지만, “generative replay가 일반적으로 충분하다”는 결론까지 확장하기는 어렵다.

또한 이 논문은 task boundary가 명확히 존재하는 설정만 다룬다. 현실에서는 task가 명확히 구분되지 않거나 점진적으로 변하는 경우가 많다. 저자들도 이런 경우 자신들의 시나리오가 더 이상 직접 적용되지 않는다고 인정한다. 따라서 이 논문의 분류 체계는 분명 유용하지만, 모든 lifelong learning 상황을 포괄하는 것은 아니다.

추가로, hyperparameter selection의 공정성 문제를 비판하면서도 결국 grid search를 사용했다는 점은 약간의 긴장을 남긴다. 다만 저자들은 이를 숨기지 않고 명시적으로 토론하며, 오히려 continual learning 평가에서 validation protocol 자체가 중요한 연구 주제임을 드러낸다.

비판적으로 해석하면, 이 논문은 새로운 이론이나 모델보다는 “문제 재정의와 재평가”에 더 가까운 작업이다. 그러나 그것이 약점만은 아니다. 분야가 혼란스러울수록, 이런 구조화 작업은 새로운 알고리즘 못지않게 중요하다. 실제로 이 논문은 왜 continual learning에서 strong baseline과 evaluation protocol이 중요한지를 잘 보여주는 대표 사례다.

## 6. 결론

이 논문은 continual learning을 Task-IL, Domain-IL, Class-IL의 세 시나리오로 나누어 체계적으로 정리하고, 대표적인 방법들을 동일 조건에서 비교함으로써 분야의 평가 기준을 명확하게 만든다. 핵심 기여는 단순히 “세 가지 이름”을 붙인 것이 아니라, 이 세 시나리오가 실제 난이도와 방법의 성패를 크게 바꾼다는 것을 실험적으로 설득력 있게 보인 데 있다.

실험 결과를 종합하면, Task-IL은 비교적 쉬워서 다양한 방법이 잘 작동한다. Domain-IL은 더 어려우며, 특히 split MNIST에서는 replay의 중요성이 커진다. 가장 중요한 것은 Class-IL로, 여기서는 regularization 기반 접근이 거의 실패하고 replay 기반 접근만이 높은 성능을 보인다. 따라서 task identity를 모르는 현실적 continual learning 문제를 다루려면, 과거 경험을 어떤 형태로든 재생성하거나 저장하여 활용하는 메커니즘이 사실상 핵심이라는 결론에 도달한다.

실제 적용 관점에서 이 연구는 매우 중요하다. 앞으로 continual learning 시스템을 설계할 때, 먼저 자신이 풀고자 하는 문제가 Task-IL인지, Domain-IL인지, Class-IL인지 명확히 해야 하며, 그에 따라 적절한 baseline과 방법을 선택해야 한다는 기준을 제공한다. 향후 연구에서도 이 논문의 분류 체계는 더 복잡한 데이터셋, 더 현실적인 환경, 불명확한 task boundary, 대규모 생성 모델 기반 replay 등으로 확장되어 계속 중요한 역할을 할 가능성이 크다. 특히 오늘날의 대규모 멀티모달 모델과 online adaptation 문제를 생각하면, 이 논문의 핵심 질문인 “테스트 시 task identity가 무엇이며, 모델이 무엇을 스스로 추론해야 하는가”는 여전히 매우 현재적인 질문이다.
