# Optimal Transport for Domain Adaptation

* **저자**: Nicolas Courty, Rémi Flamary, Devis Tuia, Alain Rakotomamonjy
* **발표연도**: 2015
* **arXiv**: [https://arxiv.org/abs/1507.00504](https://arxiv.org/abs/1507.00504)

## 1. 논문 개요

이 논문은 **unsupervised domain adaptation** 문제를 **optimal transport (OT)** 관점에서 해결하는 방법을 제안한다. 기본 상황은 다음과 같다. source domain에는 라벨이 있는 학습 데이터가 있고, target domain에는 라벨이 없으며, 두 도메인의 데이터 분포가 다르다. 이 분포 차이 때문에 source에서 학습한 분류기를 target에 그대로 적용하면 성능이 떨어진다. 논문의 핵심 목표는 source와 target의 분포를 잘 정렬시키는 **transportation plan**을 학습하고, 그 결과로 source 샘플을 target 쪽 표현으로 옮긴 뒤 그 위에서 분류기를 학습하는 것이다.

문제의 중요성은 매우 크다. 컴퓨터 비전에서는 조명, 센서, 배경, 촬영 조건 변화 때문에 같은 클래스라도 feature distribution이 달라진다. 음성, 원격탐사, 일반 시각 인식에서도 유사한 drift가 발생한다. 기존 연구도 공통 latent space를 찾거나 도메인 간 통계량을 맞추는 방식으로 이 문제를 다루었지만, 이 논문은 **각 source 샘플을 target 샘플들과의 관계 속에서 locally transport**하는 관점을 취한다. 즉, 전역적인 선형 projection이 아니라, 데이터 자체의 기하 구조를 활용해 샘플 단위로 적응시키는 접근이다.

저자들은 특히 다음과 같은 점을 강조한다. 첫째, OT는 empirical distribution 사이의 거리를 직접 계산할 수 있어 분포 추정의 부담이 적다. 둘째, support가 겹치지 않아도 의미 있는 거리와 매칭을 정의할 수 있다. 셋째, source의 class label 정보를 regularization에 넣으면 단순한 분포 정렬이 아니라 **분류에 유리한 정렬**을 만들 수 있다. 이 논문은 바로 그 점을 이용해, 분포 일치와 class structure 보존을 동시에 만족하려는 domain adaptation 프레임워크를 구성한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 다음 한 문장으로 요약할 수 있다. **source 분포를 target 분포로 옮기는 최소 비용의 운송 계획을 찾고, 그 운송 과정이 class 구조를 최대한 보존하도록 regularization을 추가하자**는 것이다.

기존의 많은 domain adaptation 방법은 공통 feature space를 찾는 데 집중했다. 예를 들어 평균을 맞추거나, 상관관계를 맞추거나, subspace를 정렬하는 방식이다. 하지만 이런 방법은 대개 전체 도메인에 동일한 전역 변환을 적용한다. 반면 이 논문은 **source의 각 샘플이 target의 여러 샘플로 얼마나 mass를 보내는지**를 나타내는 coupling matrix $\gamma$를 학습한다. 이 coupling은 단순한 거리 기반 매칭이 아니라, 두 분포의 주변합(marginal)을 동시에 만족하는 확률적 대응 관계다.

이 아이디어가 중요한 이유는 다음과 같다.

첫째, OT는 데이터 간의 geometric structure를 직접 활용한다. source 샘플과 target 샘플이 feature space에서 얼마나 떨어져 있는지를 비용으로 두고, 전체 비용이 최소가 되도록 매칭을 찾는다. 따라서 단순한 평균 정렬보다 더 정교한 대응을 만들 수 있다.

둘째, 이 논문은 단순 OT에 머물지 않고 source 라벨을 regularization에 넣는다. 그 결과, 같은 target 샘플이 서로 다른 클래스의 source 샘플로부터 mass를 많이 받는 상황을 억제할 수 있다. 즉, “분포는 맞췄지만 클래스가 섞이는” 나쁜 정렬을 줄이려는 것이다.

셋째, neighborhood structure를 보존하는 Laplacian regularization도 제안한다. 이는 source에서 비슷한 샘플들이 transport 후에도 비슷하게 남도록 유도한다. 따라서 클래스 정보뿐 아니라 국소 구조도 유지하려는 설계다.

넷째, semi-supervised setting도 자연스럽게 확장된다. target에 소수의 라벨이 있으면, 라벨이 맞지 않는 source-target 매칭에 무한대 비용을 주어 아예 금지할 수 있다. 이 확장은 매우 직관적이고 elegant하다.

결국 이 논문은 domain adaptation을 “좋은 공통 표현 찾기”가 아니라, **분포 사이의 구조적 이동 문제**로 재해석했다는 점에서 차별적이다.

## 3. 상세 방법 설명

### 3.1 문제 설정

논문은 source domain과 target domain의 두 분포를 둔다. source에는 샘플과 라벨 $(\mathbf{X}_s, \mathbf{Y}_s)$가 있고, target에는 샘플 $\mathbf{X}_t$만 있다. source와 target의 joint distribution은 각각 $\mathbf{P}_s(\mathbf{x}^s, y)$, $\mathbf{P}_t(\mathbf{x}^t, y)$이며, 입력 공간에 대한 marginal distribution을 각각 $\mu_s$, $\mu_t$라고 둔다.

저자들은 도메인 간 차이가 어떤 미지의 변환 $\mathbf{T}:\Omega_s \to \Omega_t$로 설명될 수 있다고 가정한다. 그리고 이 변환이 conditional distribution을 보존한다고 둔다.

$\mathbf{P}_s(y|\mathbf{x}^s) = \mathbf{P}_t(y|\mathbf{T}(\mathbf{x}^s))$

즉, 적절한 변환 $\mathbf{T}$를 적용하면 source 샘플의 label semantics가 target에서도 유지된다는 뜻이다. 따라서 좋은 adaptation은 결국 **$\mu_s$를 $\mu_t$로 보내는 transport map**을 찾는 문제로 바뀐다.

### 3.2 Monge와 Kantorovich 관점

가장 이상적인 목표는 source 분포를 target 분포로 보내는 변환 $\mathbf{T}$ 중에서 이동 비용이 가장 작은 것을 찾는 것이다. Monge formulation은 다음과 같다.

$$\mathbf{T}_0 = \arg\min_{\mathbf{T}} \int_{\Omega_s} c(\mathbf{x}, \mathbf{T}(\mathbf{x})) d\mu(\mathbf{x}) \quad \text{s.t. } \mathbf{T}_{\#}\mu_s = \mu_t$$

여기서 $c(\mathbf{x}, \mathbf{T}(\mathbf{x}))$는 이동 비용이고, $\mathbf{T}_{\#}\mu_s = \mu_t$는 $\mathbf{T}$가 source 분포를 정확히 target 분포로 보낸다는 뜻이다.

하지만 이 문제는 직접 풀기 어렵다. 그래서 논문은 Kantorovich relaxation을 사용한다. 여기서는 deterministic map $\mathbf{T}$ 대신, source와 target 사이의 **probabilistic coupling** $\gamma$를 찾는다.

$$\gamma_0 = \arg\min_{\gamma \in \Pi} \int_{\Omega_s \times \Omega_t} c(\mathbf{x}^s, \mathbf{x}^t), d\gamma(\mathbf{x}^s, \mathbf{x}^t)$$

$\gamma$는 source와 target의 joint distribution처럼 동작하며, 주변합이 각각 $\mu_s$, $\mu_t$가 되도록 제약된다. 즉, source의 각 점이 target의 어느 점들로 얼마나 질량을 보내는지를 나타낸다.

논문은 비용 함수로 squared Euclidean distance를 사용한다.

$$c(\mathbf{x}, \mathbf{y}) = |\mathbf{x} - \mathbf{y}|_2^2$$

즉, Wasserstein-$2$ 거리 기반 OT를 사용한다.

### 3.3 이산(discrete) OT 문제

실제로는 분포를 샘플로만 볼 수 있으므로, empirical distribution을 사용한다.

$$\mu_s = \sum_{i=1}^{n_s} p_i^s \delta_{\mathbf{x}_i^s}, \quad \mu_t = \sum_{i=1}^{n_t} p_i^t \delta_{\mathbf{x}_i^t}$$

여기서 $\delta_{\mathbf{x}}$는 Dirac mass이고, $p_i^s$, $p_j^t$는 각 샘플의 질량이다. 이때 coupling matrix $\gamma \in \mathbb{R}_+^{n_s \times n_t}$는 다음 제약을 만족해야 한다.

$$\gamma \mathbf{1}_{n_t} = \mu_s,\quad \gamma^T \mathbf{1}_{n_s} = \mu_t$$

즉, 각 source 샘플에서 나가는 총 mass와 각 target 샘플로 들어오는 총 mass가 각각 주어진 empirical distribution과 일치해야 한다. 최종 최적화는 다음 선형계획 문제가 된다.

$$\gamma_0 = \arg\min_{\gamma \in \mathcal{B}} \langle \gamma, \mathbf{C} \rangle_F$$

여기서 $\mathbf{C}(i,j)=|\mathbf{x}_i^s-\mathbf{x}_j^t|_2^2$는 비용 행렬이다. 직관적으로는 “source의 질량을 target으로 옮기되 총 이동거리가 최소가 되게 하라”는 문제다.

### 3.4 Entropy regularization

기본 OT는 해가 sparse해지기 쉽고 계산 비용도 크다. 이를 완화하기 위해 논문은 Cuturi의 entropy regularization을 채택한다.

$$\gamma_0^\lambda = \arg\min_{\gamma \in \mathcal{B}} \langle \gamma, \mathbf{C} \rangle_F + \lambda \Omega_s(\gamma)$$

여기서

$$\Omega_s(\gamma) = \sum_{i,j} \gamma(i,j)\log \gamma(i,j)$$

이다. 이 항은 coupling의 entropy를 높여서 너무 sparse한 transport plan을 부드럽게 만든다. $\lambda$가 커질수록 각 source 점의 mass가 더 많은 target 점으로 분산된다. 반대로 $\lambda$가 작으면 원래의 sharp한 OT 해에 가까워진다.

이 regularization의 또 다른 장점은 **Sinkhorn-Knopp algorithm**을 사용할 수 있게 해준다는 점이다. 즉, regularized OT는 훨씬 빠르게 계산할 수 있다.

### 3.5 OT-based sample mapping

$\gamma$를 구한 뒤에는 source 샘플을 실제로 target 쪽으로 옮겨야 한다. 논문은 barycentric mapping을 사용한다.

각 source 샘플 $\mathbf{x}_i^s$의 transport 결과 $\widehat{\mathbf{x}}_i^s$는 다음으로 정의된다.

$$\widehat{\mathbf{x}}_i^s = \arg\min_{\mathbf{x}\in\mathbb{R}^d} \sum_j \gamma_0(i,j), c(\mathbf{x}, \mathbf{x}_j^t)$$

비용 함수가 squared $\ell_2$ distance이면 이는 결국 target 샘플들의 **가중 평균**이 된다. 전체 행렬 형태로는 다음과 같이 쓸 수 있다.

$$\widehat{\mathbf{X}}_s = \mathbf{T}_{\gamma_0}(\mathbf{X}_s) = \operatorname{diag}(\gamma_0\mathbf{1}_{n_t})^{-1}\gamma_0\mathbf{X}_t$$

만약 source와 target의 marginal이 uniform이면 더 단순하게

$$\widehat{\mathbf{X}}_s = n_s \gamma_0 \mathbf{X}_t$$

로 쓸 수 있다.

이 단계가 domain adaptation에서 매우 중요하다. OT는 단지 분포 간 거리를 재는 데서 끝나지 않고, 실제로 **transported source samples**를 만들어 준다. 이후 분류기는 이 옮겨진 source 샘플들 위에서 학습된다.

### 3.6 이론적 논의: 정확한 affine transformation 복원

논문은 특정 조건에서 OT가 source와 target의 affine transformation을 정확히 복원할 수 있다는 정리를 제시한다. 조건은 대략 다음과 같다. source 샘플들이 서로 구별되고, 모든 샘플의 가중치가 균등하며, target이 source의 affine transform $\mathbf{x}_i^t = \mathbf{A}\mathbf{x}_i^s + \mathbf{b}$로 만들어졌고, $\mathbf{A}$가 positive definite이며, 비용이 squared $\ell_2$ distance인 경우다.

이때 optimal transport solution은 각 source 샘플을 정확히 해당 target 샘플로 보낸다. 이 결과는 적어도 단순한 affine drift에 대해서는 OT가 **정확한 정렬을 복원할 수 있는 이론적 근거**를 준다.

다만 논문은 더 일반적인 비선형 경우에 대한 강한 보장은 여기서 제시하지 않는다.

### 3.7 Class-aware regularization

기본 OT는 label 정보를 사용하지 않는다. 하지만 source에는 라벨이 있으므로, 이를 transport 단계에서 활용하는 것이 이 논문의 핵심 기여다.

전체 목적식은 다음과 같다.

$$\min_{\gamma \in \mathcal{B}} \langle \gamma, \mathbf{C} \rangle_F + \lambda \Omega_s(\gamma) + \eta \Omega_c(\gamma)$$

여기서 $\Omega_c(\gamma)$가 class-based regularizer다.

#### (1) Group-lasso regularization

의도는 간단하다. 한 target 샘플이 여러 source 클래스들로부터 mass를 받는 것을 줄이고, 가능하면 **같은 클래스 source들로부터만 mass를 받게 하자**는 것이다.

이를 위해 논문은 다음 regularizer를 제안한다.

$$\Omega_c(\gamma) = \sum_j \sum_{cl} |\gamma(\mathcal{I}_{cl}, j)|_2$$

여기서 $\mathcal{I}_{cl}$은 source에서 class $cl$에 속하는 샘플들의 인덱스 집합이다. $\gamma(\mathcal{I}_{cl}, j)$는 target 샘플 $j$에 대해 특정 클래스에서 들어오는 coupling 값들의 벡터다.

이 regularizer는 각 target column별로 클래스 그룹 단위 sparsity를 유도한다. 즉, 한 target 샘플 column에서 여러 클래스가 동시에 활성화되는 것을 줄인다. 결과적으로 transported source samples가 class-wise로 더 일관된 정렬을 얻도록 돕는다.

이 방식은 사실상 label proportion이 source와 target에서 크게 다르지 않다는 가정이 있을 때 더 잘 맞는다. 논문도 $P_s(y)=P_t(y)$가 이상적이라고 말한다. 다만 작은 편차는 실험적으로 크게 문제되지 않는다고 설명한다.

#### (2) Laplacian regularization

두 번째 regularizer는 label뿐 아니라 **source 내부의 neighborhood structure**도 보존하려는 목적을 가진다. source에서 비슷한 두 샘플은 transport 후에도 비슷해야 한다는 직관이다.

source similarity matrix를 $\mathbf{S}_s$라고 하면 regularizer는 다음과 같다.

$$\Omega_c(\gamma) = \frac{1}{N_s^2}\sum_{i,j} S_s(i,j)|\widehat{\mathbf{x}}_i^s - \widehat{\mathbf{x}}_j^s|_2^2$$

즉, source에서 유사한 쌍은 transport 후에도 멀어지지 않도록 패널티를 준다. 또 클래스 구조를 더 보존하기 위해 서로 다른 클래스 쌍에 대해서는 $S_s(i,j)=0$으로 둘 수 있다.

uniform marginal의 경우 이 식은 graph Laplacian을 사용해 다음처럼 쓸 수 있다.

$$\Omega_c(\gamma) = \operatorname{Tr}(\mathbf{X}_t^\top \gamma^\top \mathbf{L}_s \gamma \mathbf{X}_t)$$

여기서 $\mathbf{L}_s = \operatorname{diag}(\mathbf{S}_s\mathbf{1}) - \mathbf{S}_s$다.

더 나아가 target similarity matrix $\mathbf{S}_t$가 있으면 대칭형 regularizer도 쓸 수 있다.

$$\Omega_c(\gamma) = (1-\alpha)\operatorname{Tr}(\mathbf{X}_t^\top \gamma^\top \mathbf{L}_s \gamma \mathbf{X}_t) + \alpha \operatorname{Tr}(\mathbf{X}_s^\top \gamma \mathbf{L}_t \gamma^\top \mathbf{X}_s)$$

이 항은 source와 target 양쪽의 local geometry를 함께 반영한다.

### 3.8 Semi-supervised extension

target에 소수의 라벨이 있다면, 라벨이 맞지 않는 source-target 매칭을 아예 금지할 수 있다. 이를 위해 논문은 다음 항을 사용한다.

$$\Omega_{semi}(\gamma) = \langle \gamma, \mathbf{M} \rangle$$

여기서 $\mathbf{M}(i,j)=0$ if $y_i^s = y_j^t$ 또는 target 샘플 $j$의 라벨이 미지수이고, 그렇지 않으면 $+\infty$다. 결국 이는 원래 cost matrix $\mathbf{C}$에 라벨 불일치 매칭에 대한 무한대 비용을 더하는 것과 같다.

이 방식의 장점은 매우 분명하다. 별도 복잡한 모델 없이도, 몇 개의 target 라벨만 있으면 transport plan 자체가 더 class-consistent하게 바뀐다.

### 3.9 최적화 알고리즘

논문은 regularized OT를 풀기 위해 **generalized conditional gradient (GCG)** 알고리즘을 제안한다. 일반 형태는 다음 문제를 다룬다.

$$\min_{\gamma\in\mathcal{B}} f(\gamma)+g(\gamma)$$

여기서 논문은

$$f(\gamma)=\langle \gamma,\mathbf{C}\rangle_F + \eta \Omega_c(\gamma)$$
$$g(\gamma)=\lambda \Omega_s(\gamma)$$

로 둔다.

GCG는 전체 목적식을 다 선형화하지 않고, 일부 $f$만 선형화한다. 그러면 각 iteration에서 다음과 같은 서브문제를 푼다.

$\gamma^\star = \arg\min_{\gamma\in\mathcal{B}} \langle \gamma, \mathbf{C} + \eta \nabla \Omega_c(\gamma^k)\rangle_F + \lambda \Omega_s(\gamma)$

이 서브문제는 다시 entropy-regularized OT 꼴이므로 Sinkhorn-Knopp으로 빠르게 풀 수 있다. 이후 선형 탐색으로 step size를 정해 업데이트한다.

이 설계의 핵심은 다음과 같다. class regularizer나 Laplacian regularizer를 넣으면 원래 문제는 단순한 OT가 아니게 되는데, GCG를 쓰면 각 반복에서 여전히 **풀기 쉬운 regularized OT 서브문제**만 처리하면 된다. 즉, 이론과 계산 모두를 연결해 주는 구조다.

## 4. 실험 및 결과

### 4.1 실험 대상 방법

논문은 네 가지 OT 변형을 비교한다.

* **OT-exact**: 기본 discrete OT
* **OT-IT**: entropy regularized OT
* **OT-GL**: group-lasso regularized OT
* **OT-Laplace**: Laplacian regularized OT

추가로 예전 비볼록 regularizer 기반 방법인 **OT-LpL1**도 일부 표에 포함된다.

비교 대상 baseline으로는 SVM, DA-SVM, PBDA, PCA, GFK, TSL, JDA 등이 사용된다. 논문 텍스트에 따르면 분류기는 실험 종류에 따라 Gaussian-kernel SVM 또는 1NN이 쓰였다.

### 4.2 Two moons toy experiment

이 실험은 source로 두 개의 entangled moons를 두고, target은 이를 일정 각도만큼 회전시킨 데이터다. 회전 각도가 커질수록 adaptation 난이도가 커진다. 이 문제는 낮은 차원, 비선형 구조를 가지므로 subspace alignment 계열 방법이 약한 상황이다.

실험 세팅은 source 각 클래스당 150개, target도 동일 수이며, 최종 테스트는 target 분포를 따르는 1000개 샘플로 진행된다. 10회 반복 평균 error rate를 비교한다.

결과를 보면 OT 계열이 전반적으로 기존 방법보다 우수하다. 예를 들어 회전이 $20^\circ$일 때:

* SVM(no adapt.): 0.104
* PBDA: 0.094
* OT-exact: 0.028
* OT-IT: 0.007
* OT-GL: 0.000
* OT-Laplace: 0.000

회전이 $40^\circ$일 때도:

* PBDA: 0.225
* OT-exact: 0.109
* OT-IT: 0.102
* OT-GL: 0.013
* OT-Laplace: 0.062

특히 중간 각도까지는 OT-GL이 거의 최적 수준이다. 회전이 커져 $70^\circ$, $90^\circ$가 되면 모든 방법이 어려워지지만, OT 기반 방법은 여전히 강한 성능을 유지한다. 다만 $90^\circ$에서는 error가 약 0.5 수준으로 수렴하는데, 이는 데이터 분포만 보면 라벨이 뒤집힌 회전도 비슷하게 설명될 수 있기 때문이라고 논문은 해석한다. 즉, 분포 정렬만으로는 label inversion ambiguity를 완전히 해소할 수 없는 경우다.

이 실험은 논문의 핵심 주장을 잘 보여준다. OT는 비선형 도메인 변형에 대해서도 유연하게 동작하며, class-aware regularization이 들어가면 더 좋은 정렬을 만든다.

### 4.3 실제 visual adaptation 데이터셋

논문은 세 종류의 실제 시각 인식 문제를 사용한다.

첫째, **digits**: USPS와 MNIST
둘째, **faces**: PIE 데이터셋의 네 가지 pose domain
셋째, **objects**: Office-Caltech 데이터셋의 Amazon, Caltech, Webcam, DSLR

Object recognition에서는 두 종류의 feature를 쓴다. 하나는 800차원 SURF histogram이고, 다른 하나는 DeCAF의 fully connected 6층과 7층에서 뽑은 4096차원 deep feature다.

실험 프로토콜은 다음과 같다. 각 source domain에서 클래스당 20개 샘플(DSLR은 8개)을 랜덤 선택하고, target은 validation/test로 나눠 하이퍼파라미터를 검증한 뒤 테스트 정확도를 평가한다. 10회 반복 평균 정확도를 보고한다. unsupervised DA이지만 validation set에서 성능을 보고 파라미터를 고르는 설정이다.

### 4.4 SURF feature 결과

#### Digits

USPS→MNIST, MNIST→USPS의 평균을 보면:

* 1NN: 48.66
* GFK: 52.56
* JDA: 57.30
* OT-exact: 49.96
* OT-IT: 59.20
* OT-Laplace: 61.07
* OT-LpL1: 64.11
* OT-GL: 63.90

digits에서는 OT regularized 방법이 매우 강하다. 특히 OT-GL, OT-LpL1이 가장 좋다. 기본 OT-exact는 오히려 강하지 않은데, 이는 단순 분포 정렬만으로는 클래스 보존이 충분하지 않음을 보여준다.

#### Faces (PIE)

PIE 12개 도메인 쌍 평균은:

* 1NN: 26.22
* PCA: 34.55
* GFK: 26.15
* TSL: 36.10
* JDA: 56.69
* OT-exact: 50.47
* OT-IT: 54.89
* OT-Laplace: 56.10
* OT-LpL1: 55.45
* OT-GL: 55.88

이 경우 평균 최고는 JDA다. 논문은 PIE가 클래스 수가 68개로 많고, JDA의 EM-like iterative target refinement가 이점이 되었을 가능성을 언급한다. 즉, OT 방법이 전반적으로 강하지만 모든 상황에서 최강은 아니라는 점도 솔직하게 보여준다.

#### Objects (Office-Caltech, SURF)

12개 도메인 쌍 평균은:

* 1NN: 28.47
* PCA: 37.98
* GFK: 39.21
* TSL: 42.97
* JDA: 44.34
* OT-exact: 36.69
* OT-IT: 42.30
* OT-Laplace: 43.20
* OT-LpL1: 46.42
* OT-GL: 47.70

여기서는 OT-GL이 평균 최고다. 예를 들어 C→A, C→D, A→W, D→W 등 여러 쌍에서 OT-GL이 최고 정확도를 기록한다. 논문은 특히 object와 digit recognition에서 OT-GL이 baseline보다 뚜렷하게 좋다고 강조한다.

### 4.5 DeCAF deep feature 결과

흥미로운 점은 deep feature를 써도 domain shift가 완전히 사라지지 않는다는 것이다. Office-Caltech에 대해 DeCAF layer 6, 7 결과를 제시한다.

Layer 6 평균:

* DeCAF baseline: 65.20
* JDA: 86.72
* OT-IT: 83.64
* OT-GL: 88.18

Layer 7 평균:

* DeCAF baseline: 75.93
* JDA: 87.11
* OT-IT: 87.53
* OT-GL: 88.11

즉, deep feature 자체가 baseline을 크게 올리지만, OT adaptation을 추가하면 더 올라간다. 몇몇 도메인 쌍에서는 20%p 이상 개선이 보인다. 예를 들어 D→A나 A→W 같은 경우가 그렇다. 이 결과는 “좋은 deep representation이 있어도 여전히 distribution non-stationarity가 남아 있다”는 메시지를 준다. 또한 7층 feature가 6층보다 꼭 크게 유리하지 않다는 관찰도 제시한다. 저자들은 이 차이가 크지 않은 이유로, OT가 이미 도메인 정렬의 상당 부분을 대신하고 있을 수 있다고 해석한다.

### 4.6 Semi-supervised domain adaptation 결과

Office-Caltech SURF feature에서 target에 클래스당 3개 라벨을 준다. 여기서 두 설정을 비교한다.

첫째, **Unsupervised + labels**: 먼저 unsupervised OT로 적응하고, 이후 분류기 학습에서만 target 라벨 사용
둘째, **Semi-supervised**: OT transport plan 계산 자체에 target 라벨을 반영

평균 성능은 다음과 같다.

* Unsupervised + labels, OT-IT: 41.8
* Unsupervised + labels, OT-GL: 46.6
* Semi-supervised, OT-IT: 54.8
* Semi-supervised, OT-GL: 55.6
* MMDT [28]: 52.5

즉, target 라벨을 단지 classifier에만 쓰는 것보다, **transport plan 계산 단계에 직접 반영하는 것**이 훨씬 효과적이다. Semi-supervised OT-GL은 평균 55.6으로 비교 대상 MMDT보다 높다. 이는 이 논문의 semi-supervised extension이 단순한 부록 수준이 아니라 실제로 유의미한 성능 향상을 준다는 것을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation을 optimal transport로 정식화하면서, 단순 분포 정렬을 넘어 **class-preserving transport**까지 연결했다는 점이다. 기본 OT 위에 entropy regularization, group-lasso regularization, Laplacian regularization을 얹어 문제를 분류 친화적으로 설계했다. 이 구성은 수학적으로도 비교적 명확하고, 실험적으로도 효과가 잘 드러난다.

또 다른 강점은 방법이 매우 일반적이라는 것이다. 특정 분류기나 feature extractor에 강하게 묶이지 않는다. 실제로 논문은 toy data, hand-crafted feature, deep feature 모두에서 실험했고, 대부분의 경우 긍정적인 개선을 보였다. 특히 deep feature에서도 추가 개선이 가능하다는 결과는 이 접근이 representation learning과 경쟁하기보다 보완할 수 있음을 보여준다.

계산 측면에서도 의미가 있다. 원래 OT는 계산 비용이 큰 편인데, entropy regularization과 Sinkhorn-Knopp, 그리고 GCG를 결합해 실용적인 스케일로 확장했다. 논문이 실세계 이미지 데이터셋에서 여러 도메인 쌍 실험을 수행했다는 점은 적어도 당시 기준으로 실용성이 있었음을 시사한다.

하지만 한계도 분명하다.

첫째, 방법의 핵심 가정은 도메인 차이가 어떤 transportable transformation으로 설명될 수 있다는 것이다. 이는 많은 경우 직관적이지만, 복잡한 semantic shift나 class prior shift가 심한 상황에서는 충분하지 않을 수 있다. 논문도 group-lasso regularizer가 잘 작동하려면 이상적으로는 $P_s(y)=P_t(y)$가 유지되는 것이 바람직하다고 인정한다.

둘째, 기본 OT는 분포 정렬 문제이기 때문에, 분포만으로는 label ambiguity를 해결할 수 없는 경우가 있다. two moons의 $90^\circ$ 회전 예시는 이를 잘 보여준다. 이 경우 분포는 맞지만 라벨 반전도 비슷하게 설명될 수 있어 성능이 0.5 error 수준으로 내려간다. 즉, 분포 정렬과 label consistency는 같은 문제가 아니다.

셋째, PIE face 실험에서는 JDA가 더 좋은 결과를 보인 경우가 많다. 이는 OT 기반 방법이 모든 상황에서 가장 좋다고 말할 수는 없다는 뜻이다. 특히 클래스 수가 매우 많고 target pseudo-label refinement가 중요한 경우에는 다른 방식이 더 유리할 수 있다.

넷째, 논문은 regularizer 선택이 성능에 중요하다고 말하면서도, 어떤 상황에서 어떤 regularizer가 더 적합한지에 대한 일반 이론은 충분히 제시하지 않는다. group-lasso와 Laplacian 모두 좋은 결과를 보이지만, 데이터 특성별 선택 기준은 명확하지 않다.

다섯째, 본문에는 out-of-sample extension이나 매우 큰 데이터셋에서의 계산 자원 요구량에 대한 세부 분석은 제한적이다. transported source samples 위에서 1NN이나 SVM을 사용하는 구조는 설명되어 있지만, 완전히 새로운 test sample이 들어올 때 어떤 방식으로 가장 효율적으로 처리하는지는 이 논문 텍스트만으로는 충분히 상세하지 않다.

종합하면, 이 논문은 매우 강한 아이디어와 실험적 설득력을 갖지만, 분포 정렬이 모든 adaptation 문제를 해결해 주는 것은 아니며, regularizer 선택과 가정의 타당성이 성능에 큰 영향을 미친다.

## 6. 결론

이 논문은 **optimal transport를 domain adaptation의 중심 도구로 사용한 대표적 연구**다. 저자들은 source와 target의 empirical distribution 사이에서 최소 비용 transportation plan을 학습하고, 그 coupling으로 source 샘플을 target 쪽으로 barycentric mapping한 뒤 분류기를 학습하는 전체 프레임워크를 제안했다.

여기에 더해, source label 정보를 transport 단계에 직접 반영하는 두 가지 regularization을 제안했다. 하나는 target 샘플이 가능한 한 같은 클래스 source들로부터만 mass를 받게 만드는 **group-lasso regularization**이고, 다른 하나는 유사한 source 샘플들이 transport 후에도 유사하게 유지되도록 하는 **Laplacian regularization**이다. 또한 target에 소량의 라벨이 있을 때 이를 OT cost에 직접 넣는 semi-supervised extension도 제시했다.

실험적으로는 toy problem, digits, faces, objects, deep feature adaptation까지 폭넓게 검증했고, 전반적으로 기존 방법보다 우수하거나 경쟁력 있는 성능을 보였다. 특히 OT-GL은 digits와 object recognition에서 강한 평균 성능을 보였고, semi-supervised 확장도 실질적인 향상을 입증했다.

이 연구의 실제적 의미는 크다. 이후 domain adaptation, distribution alignment, Wasserstein learning, 그리고 deep OT 기반 방법들에 중요한 기반을 제공했기 때문이다. 특히 “도메인 적응은 공통 subspace를 찾는 것만이 아니라, 분포를 어떻게 이동시킬 것인가의 문제”라는 관점 전환을 제시했다는 점이 중요하다. 향후 연구에서는 더 복잡한 regularizer, 물리적 제약을 반영한 transport, multi-domain adaptation 같은 확장이 자연스럽게 이어질 수 있다.

전체적으로 보면, 이 논문은 OT를 단순 거리 계산 도구가 아니라 **label-aware distribution matching framework**로 확장한 작업이며, domain adaptation 연구 흐름에서 매우 영향력 있는 기여로 평가할 수 있다.
