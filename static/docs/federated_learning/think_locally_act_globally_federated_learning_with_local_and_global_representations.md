# Think Locally, Act Globally: Federated Learning with Local and Global Representations

Paul Pu Liang, Terrance Liu, Liu Ziyin, Nicholas B. Allen, Randy P. Auerbach, David Brent, Ruslan Salakhutdinov, Louis-Philippe Morency (2020)

## 🧩 Problem to Solve

이 논문은 분산된 개인 데이터를 활용하여 기계 학습 모델을 훈련하는 연합 학습(Federated Learning) 환경에서 발생하는 주요 문제들을 해결하고자 한다. 기존 연합 학습은 각 장치(device)의 데이터를 비공개로 유지하면서 모델 파라미터(parameter)나 업데이트(update)만 통신하여 전역 모델(global model)을 훈련한다. 그러나 대규모 모델에 대한 최근 요구는 효율적인 파라미터 통신에 병목 현상을 초래하여 연합 학습의 확장성을 저해한다.

문제의 중요성은 다음과 같다:

1. **확장성(Scalability) 및 통신 효율성(Communication Efficiency)**: 모델 크기가 커질수록 통신해야 할 파라미터 수가 증가하여 무선 네트워크와 같은 기존 통신 채널을 통해 효율적으로 전송하기 어렵다. 이는 실제 환경에서 연합 학습을 배포하는 데 큰 어려움이 된다.
2. **데이터 이질성(Heterogeneity)**: 실제 데이터는 다양한 소스에서 발생하며, 종종 non-i.i.d. 분포를 보인다. 특히 훈련 과정에서 관찰되지 않은 새로운 장치의 데이터 분포에 대해 기존 전역 모델이 잘 일반화(generalize)되지 못하는 문제가 있다.
3. **프라이버시(Privacy) 및 공정성(Fairness)**: 장치 내 데이터는 민감한 정보를 포함할 수 있으며, 데이터 표현(representation)에서 이러한 속성이 유출될 위험이 있다. 모델이 인종, 나이, 성별과 같은 보호된 속성을 예측하지 못하도록 공정한 표현을 학습하는 것이 중요하다.

이 논문의 목표는 이러한 문제들을 해결하기 위해 통신 효율적이며, 이질적인 데이터를 처리할 수 있고, 공정한 표현 학습을 보장하는 새로운 연합 학습 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

이 논문의 핵심 아이디어는 각 장치에서 **컴팩트한 지역 표현(compact local representations)을 공동으로 학습**하고, 이 지역 표현을 기반으로 **전역 모델을 훈련**하는 새로운 연합 학습 알고리즘인 LG-FEDAVG (Local Global Federated Averaging)를 제안하는 것이다.

이 접근 방식의 중심적인 직관과 설계 아이디어는 다음과 같다:

* **지역 모델(Local Model)과 전역 모델(Global Model)의 분리**: 각 장치는 원본 데이터를 저차원의 유용한 지역 표현으로 변환하는 지역 모델을 갖는다. 전역 모델은 이 지역 표현 위에서 작동하므로, 전역 모델 자체는 더 작아질 수 있어 통신해야 할 파라미터 및 업데이트 수를 크게 줄인다. 이는 통신 효율성 문제를 직접적으로 해결한다.
* **분산 및 장치 간 분산 감소**: 이론적으로, 지역 및 전역 모델의 조합이 데이터 분산(data variance)과 장치 분포 간 분산(device distribution variance)을 모두 줄인다는 것을 보인다. 이는 순수하게 지역 모델만 사용하거나 순수하게 전역 모델만 사용하는 경우보다 더 최적이라는 것을 의미한다.
* **유연성 및 이질성 처리**: 지역 모델은 각 장치의 특이한 데이터 분포(이질적인 데이터)를 처리하고, 훈련 중에 본 적 없는 새로운 장치 데이터에도 특화된 인코더(encoder)를 통해 유연하게 대응할 수 있다.
* **공정성 학습**: 지역 모델을 수정하여 인종, 나이, 성별과 같은 보호된 속성을 모호하게 만드는 공정한 표현을 학습할 수 있다. 이는 장치 내 데이터의 프라이버시를 보호하는 데 필수적인 기능이다.

요약하자면, LG-FEDAVG는 `Think Locally, Act Globally`라는 철학을 구현하여, 각 장치에서 데이터를 효율적으로 요약한 후, 이 요약된 정보를 기반으로 전역적인 의사결정을 내림으로써, 기존 연합 학습의 통신 병목 현상, 데이터 이질성, 프라이버시/공정성 문제를 동시에 해결하는 일반적인 프레임워크를 제공한다.

## 📎 Related Works

이 논문은 연합 학습(Federated Learning)의 맥락에서 다양한 관련 연구들을 검토하고, LG-FEDAVG가 이들과 어떻게 차별화되는지 설명한다.

### 연합 학습 (Federated Learning)

* **기존 연구**: 연합 학습은 대규모 분산 네트워크에서 개인 데이터를 사용하여 모델을 훈련하는 패러다임으로, 이질적인 데이터 소스와 다양한 학습 목표를 다룬다. 최근 연구들은 연합 학습의 효율성 개선, 원샷(one-shot) 학습, 현실적인 벤치마크 제안, 지역 및 전역 데이터 분포 간의 불일치(data mismatch) 감소에 초점을 맞추고 있다.
* **한계 및 차별점**: 기존의 특정 알고리즘들이 이질적인 데이터를 다루기 위해 제안되었지만, LG-FEDAVG는 새로운 장치에서 발생하는 이질적인 데이터를 처리하고, 통신 복잡성을 줄이며, 공정한 표현 학습을 보장할 수 있는 **더 일반적인(more general) 프레임워크**라는 점에서 차별화된다. 본 논문은 LG-FEDAVG가 이질적인 환경에서 기존 베이스라인(baseline)보다 우수한 성능을 보임을 입증한다.

### 분산 학습 (Distributed Learning)

* **기존 연구**: 분산 학습은 데이터 분할 및 업데이트 집계(aggregation)의 이론과 실제를 다룬다는 점에서 연합 학습과 유사하다. 통신 효율성 개선을 위해 데이터 및 모델 희소화(sparsifying), 효율적인 그레디언트(gradient) 방법 개발, 업데이트 압축(compressing) 등의 연구가 진행되었다.
* **한계 및 차별점**: 연합 학습은 분산 학습과 달리 **개인 데이터**와 **non-i.i.d. 분포**라는 추가적인 제약 조건에 직면한다. 본 논문의 접근 방식(지역 및 전역 모델)과 기존 압축 기술은 상호 보완적이므로, LG-FEDAVG에 압축 기술을 적용하여 추가적인 효율성 향상을 기대할 수 있다.

### 표현 학습 (Representation Learning)

* **기존 연구**: 표현 학습은 데이터로부터 생성 및 판별 작업에 유용한 특징(feature)을 학습하는 것을 목표로 한다. 최근에는 적대적 훈련(adversarial training)을 사용하여 인구 통계나 성별과 같은 개인 속성에 대해 유익하지 않은(not informative) 공정한 표현(fair representations)을 학습하는 연구가 주목받고 있다. 차등 프라이버시(differential privacy) 역시 개인 정보의 프라이버시 영향을 제한하는 관련 연구 분야이다.
* **한계 및 차별점**: LG-FEDAVG는 이질적인 데이터 및 공정성을 위한 연합 학습의 최신 발전 사항을 확장한다. 특히, LG-FEDAVG는 중간 단계에서 **공정한 지역 표현**을 학습할 수 있도록 유연한 구조를 제공한다는 점에서 기존 연구들과 차별점을 갖는다.

종합적으로, LG-FEDAVG는 기존 연합 학습 및 관련 분야의 문제점과 한계를 인식하고, 지역 표현 학습과 전역 모델 훈련을 결합함으로써 통신 효율성, 데이터 이질성 처리, 그리고 공정한 표현 학습이라는 세 가지 핵심 과제를 동시에 해결하는 새로운 일반화된 프레임워크를 제시한다.

## 🛠️ Methodology

LG-FEDAVG (Local Global Federated Averaging)는 각 장치에서의 지역 표현 학습과 모든 장치에 걸친 전역 모델 학습을 엔드-투-엔드(end-to-end) 방식으로 결합한다. 이 방법론은 크게 지역 표현 학습, 전역 집계, 그리고 추론 단계로 구성된다. 이론적 분석과 공정한 표현 학습을 위한 확장도 제시된다.

### 전체 파이프라인 또는 시스템 구조

LG-FEDAVG는 각 지역 장치(device)에서 원본 데이터($\mathbf{X}_m, \mathbf{Y}_m$)로부터 고수준의 컴팩트한 특징 $\mathbf{H}_m$을 추출하는 **지역 모델 $\Phi_m(\cdot; \theta_{\Phi_m})$**을 학습한다. 이 지역 표현 $\mathbf{H}_m$은 저차원이며 예측에 유용한 정보를 담고 있다. 이후 **전역 모델 $g(\cdot; \theta_g)$**은 원본 데이터가 아닌 이 저차원 지역 표현 $\mathbf{H}_m$을 입력으로 받아 예측을 수행한다. 전역 서버(global server)는 각 장치에서 업데이트된 전역 모델의 사본을 취합하여 새로운 전역 모델 $\theta_g$를 생성한다. 이 과정은 그림 1에 요약되어 있다.

### 각 주요 구성 요소 및 역할

* **지역 장치 (Local Device) $m$**:
  * **지역 모델 $\Phi_m(\cdot; \theta_{\Phi_m})$**: 각 장치 $m$에 특화된 모델로, 원본 데이터 $\mathbf{X}_m$를 입력으로 받아 저차원의 지역 표현 $\mathbf{H}_m = \Phi_m(\mathbf{X}_m; \theta_{\Phi_m})$을 추출한다. 이 모델의 파라미터 $\theta_{\Phi_m}$는 해당 장치에만 존재한다.
  * **전역 모델 사본 $g(\cdot; \theta_g^m)$**: 서버로부터 받은 전역 모델의 사본으로, 지역 표현 $\mathbf{H}_m$을 입력으로 받아 최종 예측 $\hat{\mathbf{Y}}_m = g(\mathbf{H}_m; \theta_g^m)$을 수행한다. 이 사본의 파라미터 $\theta_g^m$는 지역적으로 업데이트된다.
  * **학습 목표**: 지역 모델과 전역 모델 사본은 전체 손실 함수 $\mathcal{L}_g^m(\theta_{\Phi_m}, \theta_g^m)$를 최소화하기 위해 함께 업데이트된다.

* **전역 서버 (Global Server)**:
  * **전역 모델 $g(\cdot; \theta_g)$**: 모든 장치의 지역 표현에 대해 작동하는 중앙 모델이다. 각 라운드(round)마다 장치들로부터 업데이트된 $\theta_g^m$를 받아 집계하여 새로운 전역 모델 $\theta_g$를 생성한다.
  * **집계(Aggregation) 함수 AGG**: 각 장치에서 반환된 전역 모델 파라미터 사본 $\theta_g^{(t+1)m}$들을 가중 평균하여 새로운 전역 모델 $\theta_g^{(t+1)}$을 생성한다 (예: FEDAVG).

### 지역 표현 학습 (Local Representation Learning)

각 장치 $m$의 데이터 $(\mathbf{X}_m, \mathbf{Y}_m)$에 대해 학습되는 표현 $\mathbf{H}_m$은 다음 세 가지 조건을 만족해야 한다.

1. **저차원(Low-dimensional)**: 원본 데이터 $\mathbf{X}_m$에 비해 저차원이어야 한다.
2. **유용한 특징(Useful Features)**: 전역 모델의 예측에 유용한 $\mathbf{X}_m$의 중요한 특징을 포착해야 한다.
3. **과적합 방지(Prevent Overfitting)**: 전역 데이터 분포와 일치하지 않을 수 있는 장치 데이터에 과적합되지 않아야 한다.

이 논문은 다양한 지역 학습 방법들을 제안한다 (그림 1(a) ~ 1(c)):

* **(a) 지도 학습 (Supervised Learning)**: 지역 표현 $\mathbf{H}_m$이 최종 레이블 $\mathbf{Y}_m$을 예측하는 데 유용하도록 학습된다.
* **(b) 비지도 오토인코더 학습 (Unsupervised Autoencoder Learning)**: 보조 모델 $\alpha_m$이 $\mathbf{H}_m$을 사용하여 원본 데이터 $\mathbf{X}_m$를 재구성(reconstruct)하도록 학습된다.
* **(c) 자기 지도 학습 (Self-supervised Learning)**: 보조 모델 $\alpha_m$이 $\mathbf{H}_m$을 사용하여 퍼즐 조각 맞추기(jigsaw solving)와 같은 보조 레이블(auxiliary labels) $\mathbf{Z}_m$을 예측하도록 학습된다.
* **(d) 공정한 표현 학습 (Fair Representation Learning)**: (Appendix B.1에서 상세히 설명) 보조 적대적 모델(adversarial model) $\alpha_m$이 $\mathbf{H}_m$으로부터 보호된 속성 $\mathbf{P}_m$을 예측하지 못하도록 지역 모델 $\Phi_m$이 훈련된다.

### 훈련 목표, 손실 함수, 추론 절차 또는 알고리즘 흐름

**학습 절차**:
훈련은 전역 모델 라운드(round)와 각 장치에서의 지역 에포크(epoch)로 구성된다.

#### **Algorithm 1: LG-FEDAVG**

**서버 실행(Server executes):**

1. 전역 모델 $\theta_g$와 $M$개의 지역 모델 $\theta_{\Phi_m}$을 초기화한다.
2. 각 라운드 $t=1, 2, \dots$ 에 대해:
    a.  $M$개 클라이언트 중 무작위로 $m=\max(C \cdot M, 1)$개의 클라이언트 집합 $S_t$를 선택한다.
    b.  선택된 각 클라이언트 $m \in S_t$는 병렬로 `ClientUpdate` 함수를 실행한다.
    c.  업데이트된 전역 파라미터 $\theta_g^{(t+1)m}$를 받아 $\theta_g^{(t+1)} = \sum_{m=1}^M \frac{N_m}{N} \theta_g^{(t+1)m}$와 같이 집계한다.

**클라이언트 $m$ 실행(ClientUpdate($m, \theta_g^m$) run on client $m$):**

1. 지역 데이터 $(\mathbf{X}_m, \mathbf{Y}_m)$를 배치($\mathcal{B}$)로 나눈다.
2. 각 지역 에포크에 대해:
    a.  각 배치 $(\mathbf{X}, \mathbf{Y}) \in \mathcal{B}$에 대해:
        i.   추론: $\mathbf{H} = \Phi_m(\mathbf{X}; \theta_{\Phi_m})$, $\hat{\mathbf{Y}} = g(\mathbf{H}; \theta_g^m)$
        ii.  **지역 모델 업데이트**: $\theta_{\Phi_m} \leftarrow \theta_{\Phi_m} - \eta_{\theta_{\Phi_m}} \nabla_{\theta_{\Phi_m}} \mathcal{L}_g^m(\theta_{\Phi_m}, \theta_g^m)$
        iii. **전역 모델 사본 업데이트**: $\theta_g^m \leftarrow \theta_g^m - \eta_{\theta_g^m} \nabla_{\theta_g^m} \mathcal{L}_g^m(\theta_{\Phi_m}, \theta_g^m)$
3. 업데이트된 전역 파라미터 $\theta_g^m$를 서버에 반환한다.

**주요 손실 함수**:

장치 $m$에서의 전체 손실은 지역 모델 파라미터 $\theta_{\Phi_m}$와 전역 모델 파라미터 $\theta_g^m$의 함수이며 다음과 같이 정의된다:
$$ \mathcal{L}_g^m(\theta_{\Phi_m}, \theta_g^m) = \mathbb{E}_{\mathbf{x} \sim \mathbf{X}_m, \mathbf{y} \sim \mathbf{Y}_m | \mathbf{x}} \left[ -\log \sum_{\mathbf{h}} p_{\theta_g^m}(\mathbf{y}|\mathbf{h}) p_{\theta_{\Phi_m}}(\mathbf{h}|\mathbf{x}) \right] \quad (1) $$
이 손실 함수는 지역 모델과 전역 모델의 훈련을 동기화(synchronize)한다. 지역 모델은 지역 데이터에 유연하게 적합할 수 있지만, 이 손실은 모든 장치에서 학습된 지역 표현이 전역 모델에 의해 동시에 예측될 수 있도록 규제(regularizer) 역할을 한다. 즉, 지역 모델이 지역 데이터에 과적합되면 전역 모델이 모든 지역 표현으로부터 잘 예측할 수 없으므로 손실 값이 높아진다.

**추론 절차 (Inference at Test Time)**:

새로운 테스트 샘플 $\ddot{\mathbf{x}}$가 주어졌을 때 두 가지 설정이 고려된다:

1. **지역 테스트 (Local Test)**: 테스트 데이터가 어느 장치에 속하는지 아는 경우 (예: 개인화된 텍스트 완성 모델). 해당 장치에서 훈련된 지역 모델 $\Phi_m^*$을 사용하여 최적의 일치를 얻는다.
2. **새로운 테스트 (New Test)**: 테스트 시점에 새로운 데이터 분포를 가진 새로운 장치가 나타날 수 있는 경우. 이 경우, 각 지역 모델을 전역 데이터 분포의 다른 관점(view)으로 간주하고, $\ddot{\mathbf{x}}$를 모든 훈련된 지역 모델 $\Phi_m^*$에 통과시킨 후 그 출력을 **앙상블(ensemble)**하여 가장 가능성 있는 클래스를 선택한다.

### 이론적 분석 (Theoretical Analysis)

이론적 분석은 선형 모델(linear model) 설정을 가정하여 LG-FEDAVG가 왜 효과적인지 설명한다.

* **학생-교사 설정 (Student-Teacher Setting)**: 목표는 교사 네트워크 $f_u$가 생성한 목표를 예측하는 네트워크 $\hat{f}_w$를 훈련하는 것이다.
* **데이터 생성 모델**: 각 장치 $m$의 레이블은 전역 특징 벡터 $\mathbf{v}$와 지역 특징 $\mathbf{r}_m$으로 구성된 지역 교사 가중치 $\mathbf{u}_m = \mathbf{v} + \mathbf{r}_m$에 의해 생성된다. $\mathbf{r}_m \sim \mathcal{N}(0, \rho^2 \mathbf{I})$는 장치 분산(device variance)을 나타내며, $\rho^2$가 높을수록 장치별 개인화가 강하다. 훈련 목표는 노이즈 $\epsilon \sim \mathcal{N}(0, \sigma^2)$에 의해 손상되며, $\sigma^2$는 데이터 분산(data variance)을 나타낸다.

**일반화 오차(Generalization Error) 분해**:
일반화 손실 $\mathcal{E}$는 다음과 같이 분해될 수 있다 (Theorem 1):
$$ \mathcal{E} = \mathbb{E}_{\mathbf{x}, \mathbf{r}_m, \epsilon} [\hat{\mathcal{E}}_m] = \text{Var}[\hat{f}] + b^2 \quad (5) $$
여기서 $\text{Var}[\hat{f}]$는 모델의 분산(variance)이고, $b^2$는 모델의 편향(bias)이다. 편향 항은 전역 파라미터를 학습할 때 발생하며, 분산 항은 지역 및 전역 파라미터 추정치의 분산에서 비롯된다.

**LG-FEDAVG의 앙상블 해석**:
LG-FEDAVG는 지역 모델과 전역 모델의 앙상블로 볼 수 있다: $f(\mathbf{x}; \hat{\mathbf{v}}, \hat{\mathbf{u}}_m) = \alpha f_{\hat{\mathbf{u}}_m}(\mathbf{x}) + (1-\alpha) f_{\hat{\mathbf{v}}}(\mathbf{x})$. 여기서 $\alpha \in [0,1]$는 두 모델 간의 보간(interpolation) 가중치이다.
Proposition 1에 따르면:
$$ \mathcal{E} = (1 - \alpha)^2 \delta^2 + \text{Var}[\hat{f}] \quad (6) $$
여기서 $\delta^2$는 지역 변화로 인한 지역 및 전역 특징 간의 불일치(discrepancy)를 측정한다. 선형 설정에서 이는 다음과 같이 확장될 수 있다 (Corollary 0, Equation 18 in Appendix):
$$ \mathcal{E} = (1 - \alpha)^2 \left(\frac{M-1}{M}\right) \rho^2 + \text{Var}[\hat{f}] \quad (7) $$

**베이스라인 분석**:

* **순수 지역 모델($\alpha=1$)**: 일반화 오차 $\mathcal{E}(f_\Lambda) = \frac{d}{N} \sigma^2$. 데이터 분산만 제어하며, 장치 분산에 영향을 받지 않는다.
* **순수 전역 모델($\alpha=0$)**: 일반화 오차 $\mathcal{E}(f_g) = \frac{M-1}{M} \rho^2 + \frac{d}{MN} \sigma^2$. 데이터 분산을 더 효율적으로 제어하지만, 장치 분산($O(\rho^2)$ 항)에 취약하다.

**LG-FEDAVG 분석 (Theorem 2)**:
$f_\alpha(\mathbf{x}; \hat{\mathbf{v}}, \hat{\mathbf{u}}_m) = \alpha f_\Lambda(\mathbf{x}; \hat{\mathbf{u}}_m) + (1-\alpha) f_g(\mathbf{x}; \hat{\mathbf{v}})$에 대한 일반화 오차 $\mathcal{E}(f_\alpha)$는 다음과 같다:
$$ \mathcal{E}(f_\alpha) = \alpha^2 \frac{d}{N} \sigma^2 + (1-\alpha)^2 \frac{M-1}{M} \rho^2 + (1-\alpha^2) \frac{d}{MN} \sigma^2 $$
**최적의 $\alpha$ (Corollary 1)**:
일반화 오차를 최소화하는 최적의 $\alpha^*$는 다음과 같다:
$$ \alpha^* = \frac{\rho^2}{\rho^2 + \frac{d}{N}\sigma^2} $$
$\sigma^2, \rho^2 > 0$일 때, $\mathcal{E}(f_{\alpha^*}) < \mathcal{E}(f_\Lambda)$이고 $\mathcal{E}(f_{\alpha^*}) < \mathcal{E}(f_g)$이다. 이는 LG-FEDAVG가 순수 지역 모델 또는 순수 전역 모델보다 더 나은 일반화 성능을 달성함을 의미한다. $\rho^2$가 클 때(높은 장치 분산), 지역 모델에 더 큰 가중치($\alpha^*$ 증가)를 두어 지역 데이터 분포를 잘 모델링해야 한다. 반대로 $\sigma^2$가 클 때(높은 데이터 분산), 더 많은 데이터를 사용하는 전역 모델에 가중치($\alpha^*$ 감소)를 두어야 한다.

### 공정한 표현 학습 (Fair Representation Learning)

(Appendix B.1, B.2) 지역 모델을 훈련하여 보호된 속성 $\mathbf{P}_m$에 대한 정보를 제거하는 기법이다.

* **목표**: 지역 모델 $\Phi_m$이 학습한 표현 $\mathbf{H}_m$이 보호된 속성 $\mathbf{P}_m$과 통계적으로 독립이 되도록 학습하는 것이다. 즉, $p(\Phi_m(\mathbf{x}; \theta_{\Phi_m}) = \mathbf{h} | \mathbf{p}) = p(\Phi_m(\mathbf{x}; \theta_{\Phi_m}) = \mathbf{h} | \ddot{\mathbf{p}})$ for all $\mathbf{p}, \ddot{\mathbf{p}} \in \mathcal{P}$ 이다.
* **적대적 훈련 (Adversarial Training)**: 이를 위해 지역 모델 $\Phi_m$은 보조 적대적 모델 $\alpha_m$과 대립하는 방식으로 훈련된다.
  * $\alpha_m$은 지역 표현 $\mathbf{h}$로부터 $\mathbf{P}_m$을 최대한 정확하게 예측하도록 훈련된다.
  * $\Phi_m$은 $\alpha_m$이 $\mathbf{P}_m$을 예측하는 것을 방해하여, $\mathbf{H}_m$이 $\mathbf{P}_m$에 대한 정보를 담지 않도록 훈련된다.
* **미니맥스 최적화(Minimax Optimization)**: 지역 모델 $\Phi_m$, 전역 모델 $g$의 지역 사본, 그리고 적대적 모델 $\alpha_m$은 다음과 같은 미니맥스 목적 함수를 해결함으로써 공동으로 훈련된다:
    $$ \min_{\{\theta_{\Phi_m}, \theta_g^m\}} \max_{\theta_{\alpha_m}} \left[ \mathcal{L}_g^m(\theta_{\Phi_m}, \theta_g^m) - \lambda \mathcal{L}_{\alpha_m}(\theta_{\Phi_m}, \theta_{\alpha_m}) \right] \quad (40) $$
    여기서 $\mathcal{L}_{\alpha_m}$은 적대적 손실이고, $\lambda$는 예측 모델과 적대적 모델 간의 균형을 조절하는 하이퍼파라미터이다.
* **이론적 정당성 (Proposition 3)**: 적절한 손실 함수와 미니맥스 해법을 통해, 최적의 지역 모델 $\Phi_m$은 최적의 분류기(classifier)인 동시에 보호된 속성 $\mathbf{P}_m$에 대해 불변(invariant)인 핵심량(pivotal quantity)이 된다는 것을 증명한다.

## 📊 Results

LG-FEDAVG는 이론적 분석 검증, 통신 효율성, 이질적 데이터 처리, 개인화된 예측, 그리고 공정한 표현 학습 등 다양한 측면에서 평가되었다.

### 1. 이론적 분석 검증 (Verifying Theoretical Analysis)

* **합성 데이터 (Synthetic Data)**:
  * **데이터셋**: $d=20, M=100$, 장치당 2000개 훈련 샘플, 1000개 테스트 샘플. 특징 $\mathbf{x} \sim \mathcal{U}[-1.0, 1.0]$, 교사 가중치 $\mathbf{u}_m = \mathbf{v} + \mathbf{r}_m$, $\mathbf{v} \sim \mathcal{U}[0.0, 1.0]$, $\mathbf{r}_m \sim \mathcal{N}(\mathbf{0}_d, \rho^2 \mathbf{I}_d)$ (장치 분산). 레이블 $y = \mathbf{u}_m^\top \mathbf{x} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ (데이터 분산).
  * **작업**: 평균 테스트 오차(average test error) 측정.
  * **주요 결과**:
    * 지역 모델이 더 잘 수행되는 경우 ($\sigma=1.5, \rho=0.1$, 높은 장치 분산): LG-FEDAVG의 $\alpha$-보간법이 극단적인 경우(순수 지역 또는 순수 전역)보다 더 나은 성능을 보인다 (그림 2(a)).
    * 전역 모델이 더 잘 수행되는 경우 ($\sigma=1.5, \rho=0.06$, 낮은 장치 분산): 역시 LG-FEDAVG의 $\alpha$-보간법이 더 나은 성능을 보인다 (그림 2(b)).
    * 다양한 분산 설정(Appendix C.1, 그림 4)에서도 $\alpha$-보간법은 최적의 극단(cases 1, 4)에 가깝거나, 두 극단보다 우수(cases 2, 3)하다.
  * **결과의 중요성**: LG-FEDAVG의 이론적 분석(지역 및 전역 모델의 앙상블이 데이터 및 장치 분산을 모두 줄임)이 합성 데이터에서 검증되었다.

* **CIFAR-10**:
  * **데이터셋**: CIFAR-10, highly non-i.i.d. 설정. 각 장치에 $s \in \{2,3,4,5,10\}$개 클래스 할당. $s$ 값은 장치 분산을 시뮬레이션한다 ( $s=2$가 가장 높은 장치 분산, $s=10$이 i.i.d. 분할).
  * **작업**: 이미지 분류.
  * **주요 결과**: LG-FEDAVG는 `local only` 및 `FEDAVG`보다 일관되게 우수한 성능을 보인다 (그림 2(c)). 장치 분산이 증가할수록 성능 격차가 더 커진다.
  * **결과의 중요성**: 심층 네트워크(deep networks)를 사용한 복잡한 이미지 분류 문제에서도 지역 모델이 높은 장치 분산을 처리하는 데 효과적이라는 이론적 예측이 지지되었다.

### 2. 모델 성능 및 통신 효율성 (Model Performance & Communication Efficiency)

* **CIFAR-10**:
  * **데이터셋**: CIFAR-10, 각 장치에 두 개 클래스의 예시만 무작위 할당(highly unbalanced).
  * **작업**: 이미지 분류.
  * **기준선**: `FEDAVG`, `Local only`, `MTL` (Multi-Task Learning).
  * **지표**: `Local Test Accuracy`, `New Test Accuracy`, `Parameters Communicated`.
  * **모델**: LeNet-5. LG-FEDAVG는 두 개의 컨볼루션 레이어를 전역 모델로 사용, 파라미터 수를 4.48% (2872/64102)로 줄인다.
  * **주요 결과 (표 1, 표 8)**:
    * `Local Test`에서 LG-FEDAVG는 `FEDAVG`와 `MTL`을 유의미하게 능가한다 (예: 91.07% vs 58.99% for 2 classes/device). 이는 지역 모델이 장치별 데이터 분포를 더 잘 모델링하기 때문이다.
    * `New Test`에서 LG-FEDAVG는 `FEDAVG`와 유사한 성능을 달성하면서도 통신 파라미터 수를 약 50% 절감한다. 동일한 수의 파라미터를 통신할 때는 더 우수한 성능을 보인다.
    * `Local only`와 `MTL`보다도 LG-FEDAVG가 더 좋은 성능을 보이며, 엔드-투-엔드 훈련 전략의 효과를 입증한다.
  * **결과의 중요성**: LG-FEDAVG는 모델 성능을 유지하면서 통신 비용을 대폭 줄일 수 있음을 보여주었다.

* **MNIST (Appendix C.2.1, 표 6)**:
  * **데이터셋**: MNIST, non-i.i.d. 분할 (2-10개 클래스/장치).
  * **모델**: 마지막 두 레이어를 전역 모델로 사용, 파라미터 수를 15.79% (99,978/633,226)로 줄인다.
  * **주요 결과**: CIFAR-10과 유사하게, LG-FEDAVG는 `Local Test`에서 `FEDAVG`보다 우수하고, `New Test`에서 유사한 성능을 보이며 통신 파라미터 수를 약 50% 절감한다.

* **VQA (Visual Question Answering) (표 2)**:
  * **데이터셋**: VQA (0.25M 이미지, 0.76M 질문, 10M 답변), non-i.i.d. 장치 분할.
  * **모델**: LSTM 및 ResNet-18 unimodal 인코더를 지역 모델로, early fusion을 수행하는 모델을 전역 모델로 사용. 전역 모델 파라미터 수를 9.53% (5149200/54042572)로 줄인다.
  * **주요 결과**: LG-FEDAVG는 `FEDAVG`보다 높은 `Local Test Accuracy` (40.94% vs 40.02%)를 달성하면서도, 통신 파라미터 수를 약 30% 절감한다.
  * **결과의 중요성**: 대규모 멀티모달(multimodal) 벤치마크에서도 통신 비용 절감과 함께 강력한 성능을 유지함을 보여주었다.

### 3. 모바일 데이터로부터 개인화된 기분 예측 학습 (Learning Personalized Mood Predictors from Mobile Data)

* **데이터셋**: MAPS (Mobile Assessment for the Prediction of Suicide), 13-18세 청소년 14명의 실제 모바일 키보드 로거 데이터 (6개월), 572개 샘플. 기분 점수(1-100)를 5개 빈(bin)으로 분류.
* **작업**: 개인화된 기분 예측 (5-way 분류).
* **모델**: Bi-LSTM 인코더 위에 MLP 분류기. LG-FEDAVG는 지역 및 전역 모델 레이어의 다양한 분할($\alpha \in \{0.2, 0.4, 0.6, 0.8\}$)을 실험한다.
* **주요 결과 (그림 2(d))**: LG-FEDAVG의 $\alpha$-분할 모델은 순수 지역 모델 또는 순수 전역 모델보다 뛰어난 성능을 보인다.
* **결과의 중요성**: 높은 장치 분산을 가진 실제 개인화된 모바일 데이터에서, LG-FEDAVG는 개인화된 표현과 모든 장치 간의 통계적 강점 공유를 모두 활용하여 우수한 성능을 달성한다. 이는 이론적 발견과 일치한다.

### 4. 온라인 환경에서의 이질적 데이터 처리 (Heterogeneous Data in an Online Setting)

* **데이터셋**: MNIST (i.i.d. 및 non-i.i.d. 100개 장치에 분할), 여기에 90도 회전된 MNIST 이미지를 포함하는 **새로운 장치**를 추가 (그림 6).
* **작업**: 이미지 분류.
* **기준선**: `FEDAVG`, `FEDPROX` (이질적 데이터를 위해 설계된 방법).
* **지표**: `Normal` MNIST 및 `Rotated` MNIST에 대한 정확도.
* **주요 결과 (표 3)**:
  * `FEDAVG`는 미세 조정(fine-tuning) 없이(`C=0.0`) **파국적 망각(catastrophic forgetting)** 현상을 겪는다. 회전된 MNIST에서는 92%의 성능을 보이지만, 일반 MNIST에서는 32%로 크게 하락한다. 미세 조정(`C=0.1`) 후에야 두 데이터셋 모두에서 성능이 향상되지만, 이는 더 많은 통신 비용을 발생시킨다.
  * LG-FEDAVG는 지역 모델을 통해 파국적 망각을 완화한다. 회전된 MNIST에서 93%의 성능을 달성하는 동시에, 일반 MNIST에서 97%의 성능을 유지한다. 이는 `FEDAVG`와 `FEDPROX` 모두를 능가하는 결과이다.
* **결과의 중요성**: LG-FEDAVG가 새로운 데이터 분포를 가진 이질적인 장치에 대해 온라인 환경에서 효과적으로 적응하고, 기존 `FEDAVG`의 주요 문제점인 파국적 망각을 완화할 수 있음을 입증한다.

### 5. 공정한 표현 학습 (Learning Fair Representations)

* **데이터셋**: UCI adult dataset. 개인 속성(인종, 성별 등)을 기반으로 소득($>$50K/년)을 예측.
* **작업**: 소득 예측과 동시에 인종 및 성별과 같은 보호된 속성에 대해 불변(invariant)한 표현 학습.
* **모델**: 지역 모델에 적대적 훈련을 적용하여 보호된 속성 정보를 제거한다.
* **기준선**: `FEDAVG`, `LG-FEDAVG` (적대적 페널티 없음), `LG-FEDAVG+Adv` (적대적 훈련 적용).
* **지표**: `Classifier Accuracy`, `Classifier AUC`, `Adversary AUC`. 분류기 지표는 100%에 가까울수록 좋고, 적대자 지표는 50%에 가까울수록 좋다 (보호된 속성을 예측하지 못함을 의미).
* **주요 결과 (표 4)**:
  * `LG-FEDAVG+Adv`는 보호된 속성을 예측할 수 없는 공정한 지역 표현(`~50% adversary AUC`)을 학습한다.
  * 이러한 공정성 달성에도 불구하고, 전역 분류기 정확도는 `~2-4%` 정도의 작은 하락만 발생한다.
* **결과의 중요성**: LG-FEDAVG의 유연한 구조가 연합 학습 환경에서 중요한 프라이버시 목표인 공정한 표현 학습을 효과적으로 가능하게 하며, 이는 최소한의 성능 저하로 달성될 수 있음을 보여준다.

## 🧠 Insights & Discussion

LG-FEDAVG는 연합 학습의 핵심 과제인 통신 효율성, 데이터 이질성, 개인화 및 공정성 문제를 혁신적인 방식으로 해결하는 강력한 프레임워크를 제시한다.

**논문에서 뒷받침되는 강점**:

1. **통신 효율성**: 지역 모델이 원시 데이터를 저차원의 유용한 표현으로 압축하기 때문에, 전역 모델은 더 적은 수의 파라미터로 작동할 수 있다. 이는 통신되는 파라미터 및 업데이트 수를 크게 줄여 통신 병목 현상을 완화하고 효율적인 훈련을 가능하게 한다.
2. **이론적 기반**: 지역 및 전역 모델의 조합이 데이터 분산과 장치 분포 간 분산을 모두 줄여준다는 명확한 이론적 분석을 선형 모델 설정에서 제공한다. 이는 두 극단(순수 지역 또는 순수 전역)보다 최적의 일반화 성능을 달성할 수 있음을 설명한다.
3. **이질성 데이터 처리 및 개인화**: 실제 모바일 데이터 및 회전된 MNIST 데이터 실험을 통해, LG-FEDAVG가 이질적인 데이터 분포와 훈련 중 접수되지 않은 새로운 장치 데이터를 효과적으로 처리하며, 개인화된 예측 모델을 학습하는 능력을 입증한다. 특히 `FEDAVG`의 `catastrophic forgetting` 문제를 완화하는 효과를 보여준다.
4. **공정한 표현 학습**: 지역 모델에 적대적 훈련을 통합하여 보호된 속성을 모호하게 만드는 공정한 표현을 학습할 수 있음을 보여준다. 이는 민감한 데이터 환경에서 프라이버시를 보존하는 중요한 기능이며, 분류 성능에 미미한 영향만 주면서 달성된다.
5. **유연한 아키텍처**: 지역 모델과 전역 모델 간의 레이어 분할($\alpha$-split)을 통해 다양한 시나리오에 맞는 최적의 균형을 찾을 수 있는 유연성을 제공한다.

**한계, 가정 또는 미해결 질문**:

1. **선형 모델 이론의 확장**: 현재의 이론적 분석은 선형 모델에 기반하고 있다. 이것이 심층 신경망(deep neural networks)에도 그대로 적용될 수 있다는 경험적 증거는 제시하지만, 비선형 모델에 대한 엄격한 이론적 증명은 추가 연구가 필요하다.
2. **$\alpha$ 값의 결정**: 이론적으로는 최적의 $\alpha^*$가 도출되지만, 실제 심층 네트워크에서 $\alpha$는 레이어 분할(split)로 해석된다. 이 분할 지점을 동적으로 학습하거나 데이터 및 장치 특성에 따라 최적화하는 방법에 대한 심층적인 연구가 필요하다.
3. **보조 모델의 필요성**: 비지도 학습, 자기 지도 학습, 적대적 학습 등 특정 지역 표현 학습 패러다임은 추가적인 보조 모델(`$\alpha_m$`)을 필요로 한다. 이는 지역 장치에 추가적인 계산 복잡성 및 메모리 요구 사항을 부과할 수 있다.
4. **보안 및 프라이버시의 정량화**: 논문은 공정성 측면에서 프라이버시 개선을 다루지만, 통신되는 모델 업데이트가 제3자나 중앙 서버에 민감한 정보를 노출할 수 있는 보안 위험에 대해서는 명시적인 해결책보다는 향후 연구 방향으로 언급한다. 보안-프라이버시-성능 간의 트레이드오프를 정량화하는 연구가 필요하다.
5. **데이터 불균형 및 사회적 편향**: 데이터 불균형으로 인한 편향 증폭(bias amplification) 위험을 인정하며, 적대적 훈련 외의 다른 편향 제거 방법(debiasing methods) 적용 가능성을 제안한다. 이는 LG-FEDAVG 프레임워크 내에서 추가적으로 탐구될 수 있는 중요한 영역이다.

**논문에 근거한 간략한 비판적 해석 및 논의사항**:
LG-FEDAVG는 기존 연합 학습의 근본적인 한계였던 '파라미터 통신 비용'과 '이질적 데이터 처리' 문제를 지역 표현 학습을 통해 효과적으로 우회하는 영리한 접근 방식이다. 특히 이론적 분석이 심층 신경망의 경험적 결과와 밀접하게 일치한다는 점은 모델 설계의 견고함을 시사한다. 그러나 `optimal $\alpha$`에 대한 이론은 선형 모델에 한정되며, 실제 환경에서 최적의 $\alpha$ 또는 레이어 분할 지점을 결정하는 것은 여전히 실용적인 과제로 남아 있다. 또한, 공정한 표현 학습은 중요하지만, `~4%`의 분류 정확도 하락은 일부 고위험 응용 분야에서는 수용하기 어려운 트레이드오프일 수 있다. 전반적으로, 이 연구는 연합 학습 분야에 중요한 진전을 가져왔으며, 특히 자원 제약이 있는 장치와 이질적인 데이터 환경에서 실용적인 해결책을 제시하는 잠재력을 가진다. 향후 연구는 이 프레임워크의 유연성을 활용하여 압축 기술 통합, 동적 아키텍처 탐색, 그리고 다양한 편향 및 보안 위협에 대한 더욱 강력한 방어 메커니즘을 개발하는 방향으로 나아갈 수 있을 것이다.

## 📌 TL;DR

이 논문은 분산된 개인 데이터를 이용한 연합 학습의 주요 문제점(통신 비용, 데이터 이질성, 프라이버시/공정성)을 해결하기 위한 `LG-FEDAVG` (Local Global Federated Averaging)라는 새로운 알고리즘을 제안한다. 핵심 아이디어는 각 장치에서 **컴팩트한 지역 표현(local representations)을 학습**하고, 이 표현들을 기반으로 **더 작은 전역 모델(global model)을 훈련**하는 것이다.

**주요 기여 사항**:

* **통신 효율성**: 지역 모델이 원시 데이터를 저차원 표현으로 압축하여 전역 모델의 크기를 줄임으로써, 통신되는 파라미터 수를 대폭 감소시킨다.
* **이론적 정당성**: 지역 및 전역 모델의 앙상블이 데이터 분산(data variance)과 장치 분포 간 분산(device distribution variance)을 모두 효과적으로 줄여, 순수 지역 또는 순수 전역 모델보다 더 나은 일반화 성능을 달성한다는 것을 선형 모델에 대한 이론적 분석을 통해 증명한다.
* **이질성 및 개인화 처리**: 실제 모바일 데이터 및 회전된 MNIST 데이터셋에 대한 실험을 통해, LG-FEDAVG가 이질적인 데이터 분포와 훈련 중에 보지 못한 새로운 장치 데이터를 효과적으로 처리하고, 개인화된 예측 모델을 학습할 수 있음을 입증한다. 특히 기존 연합 학습의 `catastrophic forgetting` 문제를 완화한다.
* **공정한 표현 학습**: 지역 모델에 적대적 훈련(adversarial training)을 적용하여 인종, 성별 등 보호된 속성 정보를 모호하게 만드는 공정한 표현을 학습할 수 있으며, 이는 분류 성능에 최소한의 영향만 미치면서 달성된다.

**이 연구의 중요성**:
LG-FEDAVG는 연합 학습의 실질적인 적용을 가로막던 주요 장벽들을 낮추는 데 기여한다. 특히, IoT 장치, 모바일 헬스케어, 자율 주행 등 대규모 분산 환경에서 개인 정보 보호가 필수적인 시나리오에서 이 연구는 다음과 같은 중요한 역할을 할 가능성이 있다:

* **실제 적용 가능성 증대**: 통신 제약이 있는 환경에서도 고성능 모델 훈련을 가능하게 하여, 연합 학습의 실제 배포를 촉진한다.
* **프라이버시 및 윤리적 고려**: 데이터 이질성 및 공정성 문제 해결을 통해, 개인화된 서비스 제공과 동시에 사용자 프라이버시 및 사회적 편향 완화를 위한 중요한 기반을 제공한다.
* **향후 연구 방향 제시**: 압축 기술, 동적 모델 아키텍처 학습, 다양한 편향 제거 방법 통합 등 LG-FEDAVG 프레임워크의 유연성을 활용한 다양한 확장 연구를 위한 길을 열어준다.
