# A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder

- **저자**: Daehyung Park, Yuuna Hoshi, Charles C. Kemp
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1711.00614

## 1. 논문 개요

본 논문의 목표는 로봇 보조 급식(robot-assisted feeding) 작업 중 발생하는 비정상적인 실행(anomaly)을 실시간으로 감지하는 멀티모달 이상 감지기를 제안하는 것이다. 연구의 핵심 동기는 장애인이 필요한 일상생활 활동에서 로봇의 도움所提供的 assistance가 구조적 복잡성, 작업 가변성, 센서 불확실성으로 인해 실패할 수 있으며, 이러한 실패를 감지하지 못하면 잠재적 위험이 발생할 수 있다는 점이다.

연구 문제는 고차원적이고 이질적인(heterogeneous) 멀티모달 센서 신호를 효과적으로 융합하여 다양한 유형의 이상을 감지하는 것이다. 기존의 접근법은 특징 선택이나 차원 축소 후 분류기를 적용하는 방식이었으나, 이러한 압축된 표현은 이상 감지에 필요한 정보를 잃어버릴 수 있고, 수작업 특징 공학은 상당한 엔지니어링 노력과 도메인 전문 지식을 필요로 한다.

## 2. 핵심 아이디어

본 논문의 중심 아이디어는 LSTM 기반 변이 오토인코더(LSTM-VAE)를 사용하여 멀티모달 센서 신호를 잠재 공간(latent space)으로 투영하고, 이로부터 기대되는 분포를 재구성하는 reconstruction-based 접근 방식을 제안하는 것이다. LSTM-VAE는 시계열 데이터의 시간적 의존성을 모델링하면서 VAE의 생성 모델 capabilities을 결합한다.

기존 접근 방식과의 주요 차별점은 다음과 같다:

첫째, denoising autoencoding criterion을 적용하여 identity function 학습을 방지하고 표현 능력을 개선한다. 둘째, progress-based prior를 도입하여 태스크 실행의 진행 상황에 따라 prior distribution의 중심을 변화시킨다. 이는 시계열 데이터의 시간적 의존성을 잠재 공간의 분포에 반영하기 위한 것이다. 셋째, state-based threshold를 도입하여 고정 임계값보다 더 tight한 결정 경계를 달성하고 false alarm을 줄인다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인 구조

논문에서 제안하는 시스템은 크게 세 단계로 구성된다. 첫 번째 단계에서는 정상적인 실행(non-anomalous execution) 데이터로 LSTM-VAE를 학습시킨다. 두 번째 단계에서는 검증 데이터로부터 잠재 공간 표현과 해당하는 anomaly score를 추출하고, 이들을 사용하여 Support Vector Regression(SVR) 기반의 기대 anomaly score 추정기를 학습한다. 세 번째 단계에서는 온라인 실시간 감지 과정에서 현재 관측치의 anomaly score가 상태 기반 임계값을 초과하면 이상으로 판단한다.

### 3.2 LSTM-VAE 아키텍처

LSTM-VAE는 인코더(encoder)와 디코더(decoder)로 구성된다. 인코더는 시간 $t$에서의 멀티모달 입력 $\mathbf{x}_t$를 LSTM에 통과시킨 후, 두 개의 선형 모듈을 통해 잠재 변수 $\mathbf{z}_t$의 평균 $\mu_{\mathbf{z}_t}$와 공분산 $\Sigma_{\mathbf{z}_t}$을 추정한다. 이 때 $\mathbf{z}_t$는 사후 분포 $q_\phi(\mathbf{z}_t|\mathbf{x}_t)$로부터 샘플링된다.

디코더는 샘플링된 $\mathbf{z}_t$를 입력으로 받아 LSTM을 통과시키고, 최종 출력으로 재구성된 분포의 평균 $\mu_{\mathbf{x}_t}$와 공분산 $\Sigma_{\mathbf{x}_t}$를 산출한다.

### 3.3 손실 함수: Denoising Variational Lower Bound

입력에 가우시안 노이즈를 추가하는 denoising autoencoding criterion을 적용한다. 손상된 입력 $\tilde{\mathbf{x}} = \mathbf{x} + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}})$를 사용하여, 손실 함수는 다음과 같이 정의된다:

$$\mathcal{L}_{\text{dvae}} = -D_{\text{KL}}(\tilde{q}_\phi(\mathbf{z}_t|\mathbf{x}_t)||p_\theta(\mathbf{z}_t)) + \mathbb{E}_{\tilde{q}_\phi(\mathbf{z}_t|\mathbf{x}_t)}[\log p_\theta(\mathbf{x}_t|\mathbf{z}_t)]$$

첫 번째 항은 근사 사후 분포와 prior 사이의 KL 발산을 최소화하여 잠재 변수를 정규화한다. 두 번째 항은 재구성 손실을 최소화한다.

### 3.4 Progress-based Prior

기존 VAE가 정적인 표준 정규분포 $\mathcal{N}(0,1)$을 prior로 사용하는 것과 달리, 본 논문에서는 태스크 진행 상황에 따라 변화하는 prior $p(\mathbf{z}_t)$를 도입한다. 구체적으로, prior의 중심을 $\mathcal{N}(\mu_p, \Sigma_p)$로 설정하고, $\mu_p$를 태스크 시작 시 $p_1$에서 종료 시 $p_T$로 선형적으로 변화시킨다. 이를 통해 시계열 데이터의 시간적 의존성을 잠재 공간 분포에 반영한다.

정규분포 간 KL 발산은 다음과 같이 계산된다:

$$D_{\text{KL}}(\mathcal{N}(\mu_{\mathbf{z}_t},\Sigma_{\mathbf{z}_t})||\mathcal{N}(\mu_p, 1)) = \frac{1}{2}\left(\text{tr}(\Sigma_{\mathbf{z}_t}) + (\mu_p - \mu_{\mathbf{z}_t})^T(\mu_p - \mu_{\mathbf{z}_t}) - D - \log|\Sigma_{\mathbf{z}_t}|\right)$$

재구성 항은 다변량 가우시안 분포를 가정하여 다음과 같이 유도된다:

$$\mathbb{E}_{\tilde{q}_\phi(\mathbf{z}_t|\mathbf{x}_t)}[\log p_\theta(\mathbf{x}_t|\mathbf{z}_t)] = -\frac{1}{2}(\log|\Sigma_{\mathbf{x}_t}| + (\mathbf{x}_t - \mu_{\mathbf{x}_t})^T\Sigma_{\mathbf{x}_t}^{-1}(\mathbf{x}_t - \mu_{\mathbf{x}_t}) + D\log(2\pi))$$

### 3.5 Anomaly Score 및 감지 과정

Anomaly score는 관측치의 음의 로그 우도로 정의된다:

$$f_s(\mathbf{x}_t, \phi, \theta) = -\log p(\mathbf{x}_t; \mu_{\mathbf{x}_t}, \Sigma_{\mathbf{x}_t})$$

높은 score는 입력이 LSTM-VAE에 의해 잘 재구성되지 못했음을 의미한다.

감지 과정은 다음과 같은 조건으로 수행된다:

$$\begin{cases} \text{anomaly}, & \text{if } f_s(\mathbf{x}_t, \phi, \theta) > \eta \\ \neg\text{anomaly}, & \text{otherwise} \end{cases}$$

### 3.6 State-based Threshold

고정 임계값 대신 상태 기반 임계값을 도입한다. 잠재 공간 표현 $\mathbf{z}$와 해당하는 anomaly score $s$의 관계를 SVR로 모델링하여, 기대 anomaly score $\hat{f}_s(\mathbf{z})$를 추정한다. 최종 임계값은 $\eta = \hat{f}_s(\mathbf{z}) + c$로 정의되며, 여기서 상수 $c$는 감도의 민감도를 조절한다.

## 4. 실험 및 결과

### 4.1 데이터셋

실험에는 Georgia Tech Healthcare Robotics Lab에서 수집한 1,555회의 로봇 보조 급식 실행 데이터가 사용되었다. 24명의 건강한 피험자로부터 수집된 이 데이터셋은 352회의 학습/테스트 데이터(160개 이상, 192개 정상)와 1,203회의 사전 학습용 데이터(모두 정상)로 구성된다.

멀티모달 센서 신호는 5개 센서 유형에서 수집된 17차원 신호를 사용하였다: 사운드 에너지(1차원), 힘/토크 센서(3차원), 관절 토크(7차원), 숟가락 위치(3차원), 입 위치(3차원).

12가지 대표적인 이상 유형이 fault tree analysis를 통해 정의되었다: 사용자 접촉, 공격적 식사, 사용자에 의한 utensil 충돌, 사용자 소리, 얼굴 가림, 사용자에 의한 utensil 미스, 도달 불가 위치, 환경 충돌, 환경 소음, 시스템 결함에 의한 utensil 미스, 시스템 결함에 의한 utensil 충돌, 시스템 동결.

### 4.2 비교 대상(Baseline Methods)

5가지 기준선 방법과 성능을 비교하였다: Random(무작위 분류기), OSVM(One-class SVM), HMM-GP(Hidden Markov Model with Gaussian Process), AE(Autoencoder), EncDec-AD(LSTM-based Encoder-Decoder).

### 4.3 주요 정량적 결과

Leave-one-person-out cross-validation을 통해 평가된 결과, LSTM-VAE는 모든 비교 방법 중 최고 성능을 달성하였다.

4가지 수작업 특징(feature)을 사용한 경우, LSTM-VAE의 AUC는 0.8564로, 차선책인 HMM-GP(0.8121)보다 0.044 높았다.

17개의 원시 센서 신호를 사용한 경우, LSTM-VAE의 AUC는 0.8710으로, 차선책인 EncDec-AD(0.8075)보다 0.064 높았다. 특히 이 결과는 수작업 특징을 사용한 경우보다도 향상된 성능을 보여, 특징 공학 노력 없이 고차원 멀티모달 신호를 직접 활용할 수 있음을 입증하였다.

### 4.4 정성적 결과

재구성 성능 시각화에서, 정상 실행에서는 관측값과 재구성된 분포의 평균이 유사한 패턴을 보인 반면, 이상 실행(얼굴-숟가락 충돌)에서는 누적 힘(accumulated force)의 패턴이 크게 편향되어 anomaly score가 점진적으로 증가하였다. State-based threshold는 고정 임계값보다 tight한 결정 경계를 달성하여, 더 낮은 false alarm rate로 더 높은 true positive rate를 달성하였다.

## 5. 강점, 한계

### 5.1 강점

본 논문의 강점은 다음과 같이 뒷받침된다. 첫째, 원시 고차원 멀티모달 센서 신호를 특징 추출 없이 직접 활용하여 이상을 감지할 수 있으며, 이는 수작업 특징 공학의 부담을 크게 줄인다. 논문의 실험 결과에서 17개 원시 신호가 4개 수작업 특징보다 높은 AUC를 달성한 것이 이를 뒷받침한다. 둘째, LSTM-VAE가 시계열 데이터의 시간적 의존성을 효과적으로 모델링하여, sliding window 기반 방법의 한계(창 간 의존성 무시, 창 내에 이상 미포함 가능성)를 극복한다. 셋째, denoising criterion과 progress-based prior의 도입이 모델의 일반화能力和 재구성 품질을 향상시킨다. 넷째, state-based threshold가 태스크 상태에 따라 감지 민감도를 적응적으로 조절하여 false alarm을 줄인다.

### 5.2 한계 및 가정

본 연구의 한계와 가정은 다음과 같다. 첫째, 모델은 정상 실행 데이터만으로 학습되므로, 학습 시没见过한 유형의 이상에 대해서는 감지 성능이 저하될 수 있다. 둘째, 사전 학습 데이터와 다른 음식/도구를 사용하는 경우 일반화 성능이 제한될 수 있다. 논문에서 사용된 17개 센서 신호의 차원이 상대적으로 낮아, 더 복잡한 시나리오에서 확장성이 검증되지 않았다. 셋째, SVR 기반 상태 기반 임계값 학습은 추가적인 모델과 하이퍼파라미터 튜닝을 필요로 한다. 넷째, 20Hz의 샘플링 속도는 빠른 이상에 대한 반응성을 제한할 수 있다.

## 6. 결론

본 논문의 주요 기여는 세 가지로 요약된다. 첫째, LSTM-VAE를 활용한 멀티모달 이상 감지 프레임워크를 제안하여, 고차원 이질적 센서 신호의 잠재 공간 모델링과 재구성 기반 이상 감지를 실현하였다. 둘째, denoising criterion과 progress-based prior를 도입하여 시계열 데이터의 특성을 반영한 학습을 달성하였다. 셋째, state-based threshold를 통해 상태에 따른 적응적 결정 경계를 설정하여 감지 민감도와 false alarm 사이의 균형을 조절할 수 있게 하였다.

향후 연구에 대한 가능성으로, Assistive robotics에서의 실시간 안전 시스템으로의 확장, 다양한 유형의 조작 작업으로의 일반화, 그리고 더 깊은 아키텍처나 attention 메커니즘 도입을 통한 모델 개선이 있다. 본 연구는 장애인의 안전한 로봇 보조를 위한 핵심 기술로서, 실제 환경에서의 적용 가능성을 시사한다.
