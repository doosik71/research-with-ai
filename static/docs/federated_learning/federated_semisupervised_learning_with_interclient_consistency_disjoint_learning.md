# FEDERATED SEMI-SUPERVISED LEARNING WITH INTER-CLIENT CONSISTENCY & DISJOINT LEARNING

Wonyong Jeong et al.(2021)

## 🧩 Problem to Solve

기존의 Federated Learning (FL) 접근 방식은 클라이언트가 모델 학습을 위한 모든 데이터를 완전히 레이블링된 상태로 보유하고 있다고 가정한다. 그러나 현실적인 환경에서는 데이터 레이블링에 높은 비용이 들거나, 전문 지식이 요구되어 주석 작업이 어렵기 때문에, 클라이언트 측에서 얻은 데이터가 부분적으로만 레이블링되어 있거나 아예 레이블이 없는 경우가 많다. 이러한 레이블 부족 현상은 실세계 FL 애플리케이션의 중요한 제약 조건으로 작용한다.

본 논문은 이러한 레이블 부족 문제를 다루는 새로운 실용적인 FL 문제인 Federated Semi-Supervised Learning (FSSL)을 정의하고 해결하고자 한다. 특히, 레이블된 데이터의 위치에 따라 두 가지 필수 시나리오를 고려한다.

1. **Labels-at-Client 시나리오**: 각 클라이언트가 자체적으로 레이블된 데이터와 레이블 없는 데이터를 모두 보유하는 일반적인 경우를 다룬다. 예를 들어, 사용자가 자신의 개인 데이터를 부분적으로만 주석을 달 수 있는 상황이다.
2. **Labels-at-Server 시나리오**: 레이블된 데이터는 서버에만 존재하고, 클라이언트들은 레이블 없는 데이터만 보유하는 더 도전적인 경우를 다룬다. 이는 의료 영상 진단이나 체육 자세 평가와 같이 전문 지식이 필요한 레이블링 작업이 중앙 서버에서만 가능하고, 동시에 환자 기록과 같은 민감한 데이터는 개인 정보 보호 문제로 인해 클라이언트 외부로 공유될 수 없는 시나리오를 반영한다.

문제의 중요성은 데이터 프라이버시, 보안 및 접근 권한이라는 FL의 핵심 가치를 유지하면서도, 레이블링 비용과 전문성 부족이라는 실질적인 제약을 극복하여 FL의 적용 범위를 확장하는 데 있다. 기존의 FL과 Semi-Supervised Learning (SSL)을 단순히 결합하는 방식은 여러 클라이언트에서 학습된 모델 간의 이질적인 데이터 분포 지식을 충분히 활용하지 못하고, 특히 Labels-at-Server 시나리오에 적용하기 어렵거나, 레이블된 데이터로부터 학습된 지식을 잊어버리는(forgetting) 문제를 야기할 수 있다.

따라서 이 논문의 목표는 위 두 가지 FSSL 시나리오에서 효율적으로 작동하며, 기존 방법의 한계를 극복하는 새로운 FL 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Federated Semi-Supervised Learning (FSSL)이라는 실용적인 문제를 제기하고, 이를 효과적으로 해결하기 위한 "Federated Matching (FedMatch)"이라는 새로운 방법론을 제시하는 데 있다. FedMatch의 중심적인 직관과 설계 아이디어는 다음과 같다.

1. **FSSL 문제 정의 및 시나리오 제시**: 레이블 부족 문제를 겪는 FL 시나리오를 FSSL로 명명하고, 레이블된 데이터의 위치에 따라 `Labels-at-Client`와 `Labels-at-Server` 두 가지 현실적인 시나리오를 명확히 정의한다. `Labels-at-Server` 시나리오는 기존 SSL 접근 방식에 새로운 도전 과제를 제시한다.
2. **새로운 Inter-Client Consistency Loss 도입**: 여러 클라이언트에서 학습된 모델들 간의 예측 일관성을 강제하는 새로운 손실 함수를 제안한다. 이는 기존 SSL의 데이터 증강 기반 일관성 정규화와는 달리, 서로 다른 모델 간의 지식 공유를 촉진하여 이질적인 데이터 분포 환경에서 모델의 견고성을 향상시키는 것을 목표로 한다. 서버는 각 클라이언트와 유사한 다른 클라이언트 모델(헬퍼 에이전트)을 선별하여 일관성 학습을 돕는다.
3. **Disjoint Learning을 위한 Parameter Decomposition**: 모델 파라미터 $\theta$를 두 가지 독립적인 파라미터, 즉 지도 학습용 $\sigma$와 비지도 학습용 $\psi$로 분해하는 아이디어를 제시한다($\theta = \sigma + \psi$).
    * 이러한 분해는 지도 학습과 비지도 학습 태스크 간의 간섭(inter-task interference)을 최소화하여, 모델이 레이블된 데이터로부터 얻은 "신뢰할 수 있는 지식"을 잊어버리는 현상을 방지한다.
    * $\psi$에 희소성(sparsity) 정규화를 적용하고 클라이언트-서버 간에 파라미터의 "차이(differences)"만 전송함으로써 통신 비용을 크게 줄인다.
    * `Labels-at-Server` 시나리오와 같이 지도 학습이 서버에서만 이루어지고 클라이언트에서는 비지도 학습만 필요한 경우, 학습 과정을 유연하게 분리할 수 있도록 한다.
4. **광범위한 실험적 검증**: 제안하는 FedMatch가 `Labels-at-Client` 및 `Labels-at-Server` 시나리오 모두에서, IID(독립동일분포) 및 Non-IID(비독립동일분포), 스트리밍 데이터 환경을 포함한 다양한 설정에서 기존의 로컬 SSL 방법론 및 FL과 SSL의 단순 결합 베이스라인들을 크게 능가함을 입증한다. 또한, FedMatch가 훨씬 더 낮은 통신 비용을 달성함을 보여준다.

## 📎 Related Works

본 논문은 Federated Learning (FL)과 Semi-Supervised Learning (SSL)이라는 두 가지 주요 분야의 관련 연구들을 소개하고, 기존 접근 방식의 한계와 FedMatch의 차별점을 명확히 제시한다.

### Federated Learning (FL)

* **기존 FL 접근 방식**: FL은 여러 클라이언트가 협력하여 글로벌 모델을 학습하는 분산 학습 패러다임이다. 데이터는 각 로컬 클라이언트에만 비공개적으로 접근 가능하며, 클라이언트 간 데이터 공유 없이 모델 파라미터의 업데이트를 중앙 서버에 집계하는 방식으로 진행된다.
  * **FedAvg (McMahan et al., 2017)**: 로컬 학습된 가중치를 로컬 훈련 데이터 크기에 따라 가중 평균하는 기본적인 집계 방식이다.
  * **FedProx (Li et al., 2018)**: 클라이언트가 글로벌 가중치에 대한 근접 정규화(proximal regularization)를 수행하면서 로컬 업데이트를 균등하게 평균한다.
  * **FedMA (Wang et al., 2020)**: 은닉 계층의 요소들을 유사한 특징 추출 서명으로 매칭하여 평균한다.
  * **PFNM (Yurochkin et al., 2019)**: 베이지안 비모수적 방법을 활용한 집계 정책을 제안한다.
* **FL의 확장**: 로컬 지식 평균화 외에도, FL을 다양한 영역으로 확장하려는 노력이 진행되고 있다. 예를 들어, `Yoon et al., 2020a`는 `Yoon et al., 2020b`에서 제안된 파라미터 분해 기법을 활용하여 FL 환경에서의 지속 학습(continual learning)을 연구했다.
* **레이블 부족 문제에 대한 초기 관심**: 최근 `Jin et al., 2020`, `Guha et al., 2019`, `Albaseer et al., 2020` 등에서 FL에서 레이블된 데이터의 희소성을 다루는 문제에 대한 관심이 부상하고 있다.

### Semi-Supervised Learning (SSL)

* **일관성 정규화 (Consistency Regularization)**: SSL 분야에서 가장 인기 있는 접근 방식 중 하나로, 입력 인스턴스의 변환이 클래스 의미에 영향을 미치지 않는다는 가정을 기반으로 한다. 모델 출력이 다른 입력 섭동(perturbations)에 대해 동일하도록 강제한다.
  * **Sajjadi et al., 2016**: 확률적 변환 및 섭동을 통한 정규화 방법을 제안했다.
  * **확장된 기법**: 적대적 섭동(Miyato et al., 2018), 드롭아웃(Srivastava et al., 2014), 데이터 증강(French et al., 2018)을 통해 입력에 섭동을 가한다.
  * **UDA (Xie et al., 2019) 및 ReMixMatch (Berthelot et al., 2019a)**: 약한(weak) 및 강한(strong) 두 가지 증강 세트를 사용하여, 약하게 증강된 예제와 강하게 증강된 예제 간의 일관성을 강제한다.
  * **FixMatch (Sohn et al., 2020)**: 약-강 증강 쌍 간의 일관성 강제 외에도, 신뢰도 임계값(confidence thresholding)을 통해 모델 예측에 대한 의사 레이블(pseudo-label) 정제를 수행한다.
* **기타 SSL 기법**:
  * **엔트로피 최소화 (Entropy Minimization, Grandvalet & Bengio, 2004)**: 레이블 없는 데이터에 대해 분류기가 낮은 엔트로피를 예측하도록 강제한다.
  * **Pseudo-Label (Lee, 2013)**: 레이블 없는 데이터에 대한 높은 신뢰도 예측으로부터 원-핫 레이블을 생성하고 이를 표준 교차 엔트로피 손실의 훈련 목표로 사용한다.
  * **MixMatch (Berthelot et al., 2019c)**: 레이블 없는 데이터에 대한 목표 분포를 날카롭게(sharpening) 하여 생성된 의사 레이블을 더욱 정제한다.

### 기존 접근 방식과의 차별점

* **Naive FL + SSL의 한계**: 본 논문은 기존 FL 알고리즘(FedAvg, FedProx)과 최신 SSL 방법론(UDA, FixMatch)을 단순히 결합하는 것이 FSSL 문제를 완전히 해결하지 못함을 지적한다. 이러한 단순 결합은 여러 모델에서 학습된 이질적인 데이터 분포 지식을 충분히 활용하지 못하며, 특히 `Labels-at-Server` 시나리오와 같은 특정 설정에는 적용하기 어렵다. 또한, 단일 파라미터로 레이블된 데이터와 레이블 없는 데이터를 동시에 학습할 때, 레이블된 데이터로부터 얻은 지식을 잊어버리는 `inter-task interference` 문제를 겪을 수 있다.
* **FedMatch의 차별성**: FedMatch는 이러한 한계를 극복하기 위해 두 가지 핵심 요소를 도입한다.
    1. **Inter-Client Consistency Loss**: 기존 SSL의 데이터 증강 기반 일관성 정규화에서 한 단계 더 나아가, *여러 클라이언트 모델 간*의 예측 일관성을 직접적으로 강제한다. 이는 클라이언트 간의 지식 교환을 통해 모델의 견고성을 높인다.
    2. **Parameter Decomposition for Disjoint Learning**: 모델 파라미터를 지도 학습용 $\sigma$와 비지도 학습용 $\psi$로 분리하여 학습하는 기법이다. 이는 `inter-task interference`를 효과적으로 방지하고, `Labels-at-Server`와 같이 지도 학습과 비지도 학습이 물리적으로 분리될 수 있는 시나리오에 유연하게 대응하며, 통신 비용을 줄이는 데 기여한다.

이러한 독창적인 구성 요소들을 통해 FedMatch는 FSSL의 복잡한 시나리오를 효과적으로 다루며, 기존 FL 및 SSL 연구의 한계를 넘어서는 새로운 패러다임을 제시한다.

## 🛠️ Methodology

본 논문에서 제안하는 Federated Matching (FedMatch) 알고리즘은 크게 두 가지 핵심 구성 요소인 Inter-Client Consistency Loss와 Parameter Decomposition for Disjoint Learning을 통해 Federated Semi-Supervised Learning (FSSL) 문제를 해결한다.

### 1. Inter-Client Consistency Loss

기존의 일관성 정규화(consistency regularization)는 동일한 입력에 대한 다른 섭동(perturbation)의 예측이 일관되도록 강제한다. FedMatch는 이에 더해 **여러 클라이언트에서 학습된 모델들 간의 예측 일관성**을 정규화하는 새로운 `inter-client consistency loss`를 제안한다.

* **핵심 아이디어**: 각 로컬 클라이언트 모델이 다른 클라이언트 모델(헬퍼 에이전트)의 예측과 일관된 예측을 생성하도록 유도한다.
* **손실 함수**: 이는 로컬 모델 $l$의 예측 $p_{\theta_l}(y|u)$과 서버에서 선택된 헬퍼 에이전트 $h_j$의 예측 $p^*_{\theta_{h_j}}(y|u)$ 간의 KL 발산(Kullback-Leibler divergence)을 최소화함으로써 달성된다. 여기서 $u$는 레이블 없는 인스턴스이다.
    $$ \frac{1}{H} \sum_{j=1}^{H} \text{KL}[p^*_{\theta_{h_j}}(y|u) || p_{\theta_l}(y|u)] $$
    여기서 $p^*_{\theta_h}(y|u)$는 헬퍼 에이전트의 파라미터가 고정된 상태($*$ denotes frozen parameters)임을 의미한다. $H$는 사용되는 헬퍼 에이전트의 수이다.
* **전체 일관성 정규화 항 $\Phi(\cdot)$**: 데이터 레벨 일관성 정규화(예: FixMatch)와 `inter-client consistency loss`를 결합한다.
    $$ \Phi(\cdot) = \text{CrossEntropy}(\hat{y}, p_{\theta_l}(y|\pi(u))) + \frac{1}{H} \sum_{j=1}^{H} \text{KL}[p^*_{\theta_{h_j}}(y|u) || p_{\theta_l}(y|u)] $$
    여기서 $\pi(u)$는 `RandAugment`와 같은 강한 데이터 증강을 거친 레이블 없는 인스턴스이며, $\hat{y}$는 의사 레이블(pseudo-label)이다.
* **Agreement-based Pseudo-Label $\hat{y}$**:
  * 로컬 모델 $p^*_{\theta_l}(y|u)$과 $H$개의 헬퍼 에이전트 $p^*_{\theta_{h_j}}(y|u)$의 예측을 모두 원-핫 인코딩된 레이블 $\mathbf{1}(\cdot)$로 변환한 후, 이들의 합계에서 가장 많은 합의(agreement)를 이룬 클래스를 선택하여 생성한다.
  * 낮은 신뢰도의 예측은 신뢰도 임계값 $\tau$ 이하일 경우 의사 레이블 생성에서 제외된다.
    $$ \hat{y} = \text{Max}( \mathbf{1}(p^*_{\theta_l}(y|u)) + \sum_{j=1}^{H} \mathbf{1}(p^*_{\theta_{h_j}}(y|u)) ) $$
* **헬퍼 에이전트 선정**: 서버는 각 라운드에서 활성화된 클라이언트에 대해, 다른 클라이언트 모델 중 가장 관련성이 높은 $H$개의 모델을 헬퍼 에이전트로 선정한다. 모델의 관련성은 서버에 위치한 임의의 입력 $a$(예: 랜덤 가우시안 노이즈)에 대한 각 모델의 예측 $m_l = p_{\theta_l}(m|a)$을 기반으로 KD-Tree를 구축하여 가장 가까운 이웃을 찾는 방식으로 결정된다. 헬퍼 에이전트는 일반적으로 10 라운드마다 전송된다.

### 2. Parameter Decomposition for Disjoint Learning

기존 SSL은 단일 파라미터 세트로 레이블된 데이터와 레이블 없는 데이터를 동시에 학습하는데, 이는 `Labels-at-Server` 시나리오에 적용하기 어렵고, 레이블된 데이터로부터 학습된 지식을 잊어버리는 `inter-task interference`를 유발할 수 있다. FedMatch는 이러한 문제를 해결하기 위해 모델 파라미터 $\theta$를 두 개의 파라미터인 $\sigma$(지도 학습용)와 $\psi$(비지도 학습용)로 분해한다 ($\theta = \sigma + \psi$).

* **지도 학습 손실 $L_s(\sigma)$**: 레이블된 데이터 $S$에 대해 $\psi$를 고정한 상태로 $\sigma$를 최적화한다.
    $$ \text{minimize } L_s(\sigma) = \lambda_s \text{CrossEntropy}(y, p_{\sigma+\psi^*}(y|x)) $$
    여기서 $x, y$는 레이블된 데이터셋 $S$에서 가져오며, $\lambda_s$는 가중치 하이퍼파라미터이다.
* **비지도 학습 손실 $L_u(\psi)$**: 레이블 없는 데이터 $U$에 대해 $\sigma$를 고정한 상태로 $\psi$를 최적화한다.
    $$ \text{minimize } L_u(\psi) = \lambda_{\text{ICCS}} \Phi_{\sigma^*+\psi}(\cdot) + \lambda_{L_2} ||\sigma^* - \psi||^2_2 + \lambda_{L_1} ||\psi||_1 $$
    여기서 $\lambda_{\text{ICCS}}, \lambda_{L_2}, \lambda_{L_1}$은 각 항의 학습 비율을 제어하는 하이퍼파라미터이다.
  * $\Phi_{\sigma^*+\psi}(\cdot)$는 위에서 설명한 일관성 정규화 항이며, $\sigma$는 고정된 값 $\sigma^*$로 사용된다.
  * $L_2$ 정규화 항 $||\sigma^* - \psi||^2_2$는 $\psi$가 $\sigma^*$로부터 너무 멀리 벗어나지 않도록 하여, $\sigma$가 학습한 지식을 유지하도록 돕는다.
  * $L_1$ 정규화 항 $||\psi||_1$은 $\psi$를 희소(sparse)하게 만들어 통신 비용 절감에 기여한다.
* **Parameter Decomposition의 이점**:
    1. **레이블된 데이터의 신뢰할 수 있는 지식 보존**: 지도 학습과 비지도 학습 간의 태스크 간 간섭을 효과적으로 방지하여 모델이 레이블된 데이터로부터 학습한 지식을 잊어버리는 것을 막는다.
    2. **통신 비용 감소**: $\psi$의 희소성 덕분에 통신량을 줄일 수 있다. 또한, 각 파라미터에 대해 학습된 지식의 차이($\Delta\psi = \psi_l^r - \psi_G^r$, $\Delta\sigma = \sigma_l^r - \sigma_G^r$)만을 희소 행렬로 전송하여 클라이언트-서버 및 서버-클라이언트 통신 비용을 모두 최소화한다.
    3. **분리된 학습 (Disjoint Learning)**: `Labels-at-Server` 시나리오와 같이 지도 학습이 서버에서, 비지도 학습이 클라이언트에서 이루어지는 경우에도 모델의 학습 절차를 유연하게 분리할 수 있다.

### 3. FSSL 시나리오별 학습 절차

#### **가. Labels-at-Client 시나리오 (Algorithm 1, Figure 3)**

* **문제 정의**: 클라이언트가 전체 데이터의 일부(예: 5%)만 레이블링하고 나머지는 레이블 없는 상태로 보유한다.
* **학습 절차**:
    1. **서버**: $\sigma_0, \psi_0$를 초기화하고, 각 라운드 $r$마다 $A$개의 클라이언트를 선택한다.
    2. **클라이언트**: 선택된 각 클라이언트 $l_a^r$는 서버로부터 $\sigma_r, \psi_r$ 및 헬퍼 에이전트 $\psi_{1:H}^r$를 받는다. 로컬 클라이언트는 자신의 레이블된 데이터 $S_{l_a}$에서 $\sigma$를, 레이블 없는 데이터 $U_{l_a}$에서 $\psi$를 학습한다 (각각 다른 파라미터를 고정한 상태로).
    3. **서버**: 클라이언트로부터 학습된 $\sigma_{l_a}^r, \psi_{l_a}^r$를 받아 집계하고($\sigma_{r+1} \leftarrow \frac{1}{A} \sum \sigma_{l_a}^r$, $\psi_{r+1} \leftarrow \frac{1}{A} \sum \psi_{l_a}^r$), 다음 라운드를 위해 클라이언트에 전송한다. 또한, 헬퍼 에이전트 선정을 위해 모델 임베딩을 업데이트하고 KD-Tree를 생성한다.

#### **나. Labels-at-Server 시나리오 (Algorithm 2, Figure 4)**

* **문제 정의**: 레이블된 데이터 $S_G$는 서버에만 존재하고, 클라이언트들은 오직 레이블 없는 데이터 $U_{l_a}$만 보유한다.
* **학습 절차**:
    1. **서버**: $\sigma_0, \psi_0$를 초기화한다. 각 라운드 $r$마다 서버는 자신의 레이블된 데이터 $S_G$에서 $\sigma$를 직접 학습한다.
    2. **클라이언트**: 서버는 학습된 $\sigma_{r+1}$과 $\psi_r$ 및 헬퍼 에이전트 $\psi_{1:H}^r$를 선택된 클라이언트에 전송한다. 각 클라이언트 $l_a^r$는 오직 자신의 로컬 레이블 없는 데이터 $U_{l_a}$에서 $\psi$만 학습한다 ($\sigma$는 고정).
    3. **서버**: 클라이언트로부터 학습된 $\psi_{l_a}^r$를 받아 집계하고($\psi_{r+1} \leftarrow \frac{1}{A} \sum \psi_{l_a}^r$), 다음 라운드를 위해 클라이언트에 전송한다. 이 시나리오에서는 클라이언트가 레이블된 데이터에 대한 파라미터 $\Delta\sigma$를 서버에 전송할 필요가 없으므로 C2S 통신 비용이 더욱 효율적이다.

## 📊 Results

본 논문은 제안하는 FedMatch 방법론을 Labels-at-Client 및 Labels-at-Server 두 가지 시나리오에서 Batch-IID, Batch-NonIID(클래스 불균형), Streaming-NonIID(클래스 불균형 데이터 스트림)의 세 가지 태스크에 걸쳐 광범위하게 검증한다.

### 실험 설정

* **데이터셋**:
  * Batch 태스크(IID 및 NonIID): CIFAR-10 (60,000개 인스턴스). 100개 클라이언트, 각 클래스당 5개의 레이블된 인스턴스(총 50개)를 클라이언트(Labels-at-Client) 또는 서버(Labels-at-Server)에 할당하고, 나머지는 레이블 없는 데이터로 사용한다.
  * Streaming 태스크(NonIID): Fashion-MNIST (70,000개 인스턴스). 10개 클라이언트, 각 클래스당 5개의 레이블된 인스턴스를 할당한다. 데이터 스트림은 클래스 불균형 분포로 각 클라이언트에 분할된다.
  * **COVID-19 Radiography Dataset (Appendix)**: 실제 세계 데이터셋에 대한 실험도 수행하여 COVID-19, 일반, 바이러스성 폐렴 X-레이 이미지 분류에서 FedMatch의 성능을 검증한다.
* **베이스라인**:
  * **Local-SL, Local-UDA, Local-FixMatch**: 로컬에서만 학습하고 지식을 공유하지 않는 지도/반지도 학습 모델.
  * **FedAvg/FedProx-SL**: FedAvg 또는 FedProx 프레임워크와 결합된 지도 학습 모델. (모든 데이터에 완전한 레이블이 있다고 가정한 상한선 역할)
  * **FedAvg/FedProx-UDA, FedAvg/FedProx-FixMatch**: FedAvg 또는 FedProx 프레임워크와 UDA/FixMatch를 단순히 결합한 naive한 Fed-SSL 모델.
* **평가 지표**: 글로벌 모델 정확도(Accuracy), 평균 통신 비용(Server-to-Client Cost: S2C Cost, Client-to-Server Cost: C2S Cost).
* **네트워크 아키텍처**: ResNet-9를 기본 백본으로 사용하며, AlexNet-Like 아키텍처에서도 추가 실험을 수행한다.
* **훈련 세부 사항**: SGD 옵티마이저, 적응형 학습률 감소, L2 가중치 감소 정규화. 모든 모델에 동일한 하이퍼파라미터 설정 적용 (FixMatch 및 FedMatch의 의사 레이블링 임계값 $\tau=0.85$).

### 주요 실험 결과

1. **Batch-IID 및 NonIID 태스크 성능 (Table 1, Figure 5)**
    * **FedMatch의 우수성**: 모든 태스크 및 시나리오에서 FedMatch가 모든 naive Fed-SSL 베이스라인을 능가한다. 특히 Labels-at-Server 시나리오에서 naive 결합 모델들은 `forgetting issue`로 인해 성능 저하를 겪지만, FedMatch는 일관되고 견고한 성능을 보여준다.
    * **Non-IID 환경에서의 강점**: 클래스 불균형(Non-IID) 태스크에서 베이스라인 모델의 성능이 1-3%p 하락하는 반면, FedMatch는 일관된 성능을 유지하며, `inter-client consistency`가 불균형 태스크에서 효과적임을 시사한다.
    * **통신 효율성**: FedMatch는 베이스라인 대비 S2C 및 C2S 통신 비용을 크게 절감한다 (예: Labels-at-Server Batch-IID에서 S2C 45%, C2S 22%). 이는 파라미터 분해 및 차이 전송 기법 덕분이다.

2. **Streaming-NonIID 태스크 성능 (Table 2)**
    * **FedMatch의 우수성**: Labels-at-Client 시나리오에서 FedMatch는 Local-SSL 및 naive Fed-SSL 모델보다 4-15%p 높은 성능을 보인다.
    * **Labels-at-Server 시나리오의 특징**: FedProx-SL의 성능은 Labels-at-Client 시나리오에 비해 약 5%p 감소한다. 이는 스트리밍 환경에서 모델이 새로운 데이터에 충분히 훈련되지 못하기 때문으로 추정된다. 반면, Fed-SSL 모델들은 일관된 의사 레이블을 활용하여 이 한계를 극복하고 개선된 성능을 얻는다. FedMatch는 이 태스크에서도 가장 우수한 성능을 보이며, 통신 비용도 크게 절감된다.

3. **요소별 효과 분석 (Ablation Study, Figure 6)**
    * **Inter-Client Consistency Loss (ICCL)의 효과 (Figure 6a)**: ICCL을 제거하면 성능이 약간 하락하지만, FedMatch는 여전히 베이스라인 모델보다 우수하다. 이는 파라미터 분해 기법 자체의 효과를 보여준다. ICCL은 헬퍼 에이전트를 통한 모델 간 일관성 강화를 통해 성능을 더욱 향상시킨다.
    * **Parameter Decomposition의 효과 (Figure 6b)**: $\sigma$와 $\psi$ 중 어느 하나라도 제거하면 성능이 크게 하락하며, 특히 레이블된 데이터에서 필수적인 지식을 포착하는 $\sigma$를 제거했을 때 성능 저하가 더 크다. 이는 지도 학습과 비지도 학습 간의 `knowledge interference`가 존재하며, 파라미터 분해가 이를 효과적으로 다룸을 입증한다.
    * **Inter-Task Interference 해결 (Figure 6c)**: FedMatch는 레이블된 데이터셋에 대한 학습된 지식을 효과적으로 보존하는 반면, 다른 베이스라인 모델들은 `knowledge interference`로 인해 성능 저하를 겪는다.
    * **레이블 수에 따른 성능 변화 (Figure 6d)**: 레이블당 인스턴스 수가 증가함에 따라 FedMatch는 일관된 성능 향상을 보인다. 반면, FedProx-UDA/FixMatch 같은 베이스라인은 때때로 레이블 수가 증가함에도 성능이 저하되는 현상을 보여, FedMatch가 레이블된 및 레이블 없는 데이터를 더 효과적으로 활용함을 시사한다.

4. **COVID-19 Radiography Dataset 실험 (Figure 9 in Appendix)**
    * 실제 세계 데이터셋에서도 FedMatch는 FedProx-UDA/FixMatch 대비 Labels-at-Client 및 Labels-at-Server 시나리오 모두에서 4-10%p의 상당한 성능 향상을 보이며, 더 빠르고 안정적인 학습 곡선을 나타낸다.

5. **백본 아키텍처 및 클라이언트 참여율 (Table 6 in Appendix)**
    * AlexNet-Like와 같은 더 작고 다른 아키텍처에서도 FedMatch는 naive Fed-SSL 모델들보다 우수한 성능을 유지하며, 방법론의 일반화 가능성을 보여준다.
    * 라운드당 참여 클라이언트 비율이 증가하면 모든 모델의 성능이 약간 향상되는 경향을 보이지만, FedMatch의 성능은 여전히 모든 베이스라인을 일관되게 능가한다.

## 🧠 Insights & Discussion

본 논문은 Federated Semi-Supervised Learning (FSSL)이라는 실용적이고 중요한 문제를 심도 있게 다루며, `FedMatch`라는 혁신적인 해결책을 제시한다. 이 연구에서 도출할 수 있는 통찰과 논의 사항은 다음과 같다.

### 논문에서 뒷받침되는 강점

1. **현실 문제 해결 능력**: 기존 FL의 비현실적인 '완전 레이블링' 가정을 극복하고, 레이블 부족이라는 현실적인 제약을 FSSL이라는 명확한 문제로 정의한다. 특히 `Labels-at-Server` 시나리오는 실제 의료, 산업 분야에서 데이터 프라이버시와 전문 지식의 필요성이라는 두 가지 난제를 동시에 해결할 수 있는 중요한 접근 방식이다.
2. **효과적인 지식 활용**:
    * **Inter-Client Consistency Loss**: 이질적인 데이터 분포를 가진 여러 클라이언트 모델 간의 예측 일관성을 강제함으로써, 각 로컬 모델이 더 넓은 범위의 집단 지식을 활용하고 견고성을 높이는 데 기여한다. 이는 Non-IID 환경에서 특히 효과적임을 실험 결과가 입증한다. 기존 SSL의 데이터 증강 기반 일관성 정규화와 차별화되는 강점이다.
    * **Parameter Decomposition**: $\sigma$와 $\psi$로의 파라미터 분해는 지도 학습과 비지도 학습 간의 `inter-task interference`를 효과적으로 차단하여, 레이블된 데이터로부터 얻은 핵심 지식의 `catastrophic forgetting`을 방지한다. 이는 naive한 Fed-SSL 모델들이 `Labels-at-Server` 시나리오에서 겪는 치명적인 성능 저하를 FedMatch가 회피할 수 있는 주된 이유이다.
3. **통신 효율성**: $\psi$에 대한 $L_1$ 정규화를 통한 희소성 유도 및 클라이언트와 서버 간 파라미터의 "차이"만을 전송하는 전략은 FL의 고질적인 문제인 통신 비용을 대폭 절감한다. 특히 `Labels-at-Server` 시나리오에서는 클라이언트가 $\Delta\sigma$를 전송할 필요가 없어 더욱 효율적이다.
4. **광범위한 실험적 증명**: IID, Non-IID, 스트리밍 환경, 두 가지 FSSL 시나리오, 다양한 백본 아키텍처, 그리고 실제 세계 데이터셋(COVID-19)에 걸친 철저한 실험을 통해 FedMatch의 우수한 성능과 견고성, 그리고 실용적 가치를 강력하게 뒷받침한다.

### 한계, 가정 또는 미해결 질문

1. **헬퍼 에이전트 선정의 복잡성 및 비용**: 서버는 각 클라이언트에 대해 관련성 높은 $H$개의 헬퍼 에이전트를 찾기 위해 모든 클라이언트 모델의 임베딩을 관리하고 KD-Tree를 구축해야 한다. 본 논문은 이러한 과정이 효율적이라고 주장하지만, 대규모 클라이언트 환경에서 이 과정 자체가 서버에 상당한 계산 및 저장 부담을 줄 수 있으며, 10 라운드마다 헬퍼 에이전트를 전송하는 빈도나 $H$ 값의 최적화에 대한 심층적인 분석은 부족하다.
2. **의사 레이블링 임계값 $\tau$의 민감도**: 의사 레이블링에 사용되는 신뢰도 임계값 $\tau=0.85$는 모든 FixMatch 및 FedMatch 실험에 고정되어 있다. 이 임계값의 선택이 모델 성능에 미치는 영향이나 동적으로 조절하는 방법에 대한 분석은 제시되지 않는다.
3. **통신 효율성의 비트 레벨 구현**: "실제 비트 레벨 압축 기술은 본 연구 범위를 넘어선 구현 문제"라고 언급하며, "정보량" 감소에 초점을 맞추었음을 밝힌다. 파라미터 차이를 전송할 때 "거의 변경되지 않은 값을 요소별로 버린다"고 설명하지만, 구체적인 임계값 설정(1e-5 ~ 5e-5)의 근거나 그로 인한 정보 손실의 실제 영향에 대한 정량적 분석은 제한적이다.
4. **프라이버시 보장**: Federated Learning의 주요 목표 중 하나인 프라이버시 보호에 대해, FedMatch는 클라이언트 데이터를 직접 공유하지 않으므로 기본적인 프라이버시는 유지한다. 그러나 헬퍼 에이전트 선정 과정에서 클라이언트 모델의 "예측(predictions)"을 서버에서 취합하고, 이 예측을 통해 다른 클라이언트 모델에 대한 정보를 간접적으로 노출할 수 있는 잠재적인 프라이버시 문제는 명확히 다루지 않는다.

### 논문에 근거한 간략한 비판적 해석 및 논의사항

`FedMatch`는 FSSL이라는 중요하고 도전적인 문제에 대한 매우 견고하고 효과적인 해결책을 제시한다. 특히 파라미터 분해와 `inter-client consistency`라는 두 가지 핵심 아이디어는 FSSL의 고유한 난제들(예: `inter-task interference`, `Labels-at-Server` 시나리오의 비유연성, 통신 비용)을 정면으로 돌파한다. 논문은 `naive FL+SSL` 결합이 왜 실패하는지를 명확히 보여주고, `FedMatch`가 이러한 한계를 어떻게 극복하는지 상세한 실험을 통해 설득력 있게 증명한다.

비판적으로 보자면, 헬퍼 에이전트 선정 과정의 중앙 집중식 특성(서버가 모든 모델 임베딩을 관리)이 대규모 FL 환경에서 병목 현상이나 확장성 문제를 야기할 수 있는 가능성, 그리고 민감한 하이퍼파라미터(예: $\tau$, 통신 절감 임계값)에 대한 추가적인 강건성 분석이 있다면 연구의 완성도를 더욱 높일 수 있었을 것이다. 그럼에도 불구하고, `FedMatch`는 현실적인 제약 조건 하에서 딥러닝 모델을 분산 학습하는 방법을 탐구하는 데 있어 중요한 진전을 이루었으며, 향후 FSSL 연구의 강력한 기반을 제공할 것으로 기대된다.

## 📌 TL;DR

이 논문은 레이블 부족 문제가 심각한 현실 세계의 분산 학습 시나리오인 **Federated Semi-Supervised Learning (FSSL)**을 소개하고, `Labels-at-Client`와 `Labels-at-Server`의 두 가지 핵심 상황을 정의한다. 제안하는 방법론인 **Federated Matching (FedMatch)**은 두 가지 주요 기여를 통해 이 문제를 해결한다. 첫째, **Inter-Client Consistency Loss**를 도입하여 여러 클라이언트 모델 간의 예측 일관성을 강제함으로써 이질적인 데이터 환경에서 모델의 견고성을 향상시킨다. 둘째, **Parameter Decomposition for Disjoint Learning**을 통해 모델 파라미터를 지도 학습용 $\sigma$와 비지도 학습용 $\psi$로 분리하여 학습한다. 이는 지도-비지도 학습 간의 간섭(interference)을 방지하고, 레이블된 데이터로부터의 지식 보존을 가능하게 하며, 동시에 $\psi$의 희소성 및 파라미터 차이 전송을 통해 통신 비용을 크게 절감한다. 광범위한 실험 결과에 따르면, FedMatch는 모든 시나리오와 태스크(IID, Non-IID, 스트리밍)에서 기존의 로컬 SSL 및 FL과 SSL을 단순히 결합한 베이스라인보다 훨씬 우수한 성능과 통신 효율성을 보인다. 이 연구는 레이블이 희소한 환경에서의 FL 적용 가능성을 확장하고, 향후 온디바이스 학습과 같은 분야에서 완전히 레이블 없는 데이터 스트림에 적응하는 모델 연구에 중요한 기반을 제공할 잠재력이 있다.
