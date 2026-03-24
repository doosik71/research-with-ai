# Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective

- **저자**: Zexin Wang, Changhua Pei, Minghua Ma, Xin Wang, Zhihan Li, Dan Pei, Saravan Rajmohan, Dongmei Zhang, Qingwei Lin, Haiming Zhang, Jianhui Li, Gaogang Xie
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2402.02820

## 1. 논문 개요

이 논문은 **univariate time series (UTS)** 에 대한 **비지도 이상 탐지(unsupervised anomaly detection)** 문제를 다룬다. 특히 웹 시스템이나 클라우드 시스템처럼 수많은 KPI 시계열을 실시간으로 감시해야 하는 환경에서, 이상을 빨리 감지해 후속 진단과 복구 절차를 시작하는 것이 매우 중요하다는 실무적 배경 위에서 출발한다.

저자들은 기존의 비지도 시계열 이상 탐지 방법을 크게 예측 기반(prediction-based)과 복원 기반(reconstruction-based)으로 나눈다. 예측 기반 방법은 다음 값을 맞추는 데 초점이 있기 때문에, 학습 데이터에 섞여 있는 이상 패턴까지 따라가 버릴 위험이 있다. 반면 VAE 기반 복원 방법은 입력 윈도우를 latent space로 압축한 뒤 다시 복원하면서 “정상 패턴”을 재구성하는 데 강점이 있어 이상 탐지에 적합하다. 그러나 저자들은 기존 VAE 계열 방법들이 실제로는 **긴 주기의 heterogeneous periodic pattern** 과 **짧은 주기의 상세한 trend** 를 동시에 잘 복원하지 못한다고 지적한다.

논문이 제기하는 핵심 연구 문제는 다음과 같다. **기존 VAE 기반 이상 탐지 모델은 왜 정상 시계열의 주기성과 세부 변동을 충분히 복원하지 못하는가? 그리고 이 한계를 frequency 관점에서 어떻게 개선할 수 있는가?** 저자들은 이 문제를 단순히 모델 용량 부족이나 구조 선택의 문제로 보지 않고, **시간 영역에서만 학습할 때 중요한 frequency-domain 정보가 충분히 반영되지 않는다**는 관점으로 다시 해석한다.

이 문제의 중요성은 매우 크다. 이상 탐지에서는 이상 지점에서만 재구성 오차가 커야 하는데, 정상 구간에서도 재구성이 부정확하면 false positive가 증가하고 전체 F1이 크게 떨어진다. 즉, “이상은 잘 무시했지만 정상도 잘 못 복원하는” 모델은 실제 운영 환경에서 탐지기로서 한계가 있다. 이 논문은 바로 이 지점을 파고들어, 정상 시계열 복원의 정밀도를 높임으로써 이상 탐지 자체를 개선하려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **frequency information을 Conditional Variational Autoencoder (CVAE)의 condition으로 넣어 주면, 기존 VAE가 놓치던 정상 시계열의 구조적 정보를 더 잘 복원할 수 있다**는 것이다. 저자들은 특히 frequency 정보를 두 수준으로 나누어 본다. 하나는 전체 윈도우의 전반적 주기 패턴을 반영하는 **global frequency**, 다른 하나는 마지막 시점 주변의 세부적 변화와 국소 패턴을 반영하는 **local frequency** 이다. 이 둘을 동시에 사용하면 긴 주기 패턴과 짧은 주기 패턴을 함께 포착할 수 있다고 본다.

기존 접근과의 차별점은 크게 세 가지로 요약할 수 있다.

첫째, 단순히 CVAE를 쓰는 것이 아니라 **time stamp 같은 기존 조건 정보 대신 frequency 자체를 condition으로 사용**한다는 점이다. 논문은 timestamp condition이 heterogeneous periodic pattern을 충분히 설명하지 못한다고 본다. 시간 인덱스는 sparse하고 정보량이 제한적이며, 실제 패턴의 모양 차이를 직접적으로 전달하지 못한다는 것이다.

둘째, frequency 정보를 하나의 단일 벡터로만 쓰지 않고, **GFM(Global Frequency Module)** 과 **LFM(Local Frequency Module)** 으로 나누어 설계했다는 점이다. GFM은 전체 윈도우의 평균적이고 전역적인 주기 정보를 제공하고, LFM은 작은 sub-window들에서 추출한 주파수 특성을 target attention으로 가중 결합해 마지막 지점과 관련된 국소 정보를 강조한다.

셋째, 단지 모듈 하나를 추가한 수준이 아니라, 학습과 테스트 절차까지 frequency 관점에 맞게 조정했다는 점이다. 저자들은 **CM-ELBO**, **missing data injection**, **masking the last point**, 그리고 **이상 데이터 증강**을 함께 사용해, 이상이 섞인 비지도 학습 환경에서도 모델이 정상 패턴 복원에 집중하도록 만든다.

이 논문이 전달하는 직관은 비교적 명확하다. 정상 시계열의 세밀한 모양은 시간 영역에서 바로 보이지 않을 수 있지만, frequency 영역에서는 특정 성분의 부재나 약화로 나타난다. 그렇다면 복원 모델이 이 frequency 정보를 condition으로 활용하면, 정상 패턴을 더 정확하게 복원할 수 있고, 결과적으로 이상 점수도 더 분리된다는 것이다.

## 3. 상세 방법 설명

### 3.1 문제 정의와 기본 프레임

논문은 시계열 $\mathbf{x} = [x_0, x_1, x_2, \dots, x_t]$ 와 라벨 시계열 $\mathbf{L} = [l_0, l_1, l_2, \dots, l_t]$ 를 정의한다. 여기서 $x_i \in \mathbb{R}$, $l_i \in {0,1}$ 이다. 이상 탐지의 목표는 각 시점 $x_i$ 에 대해, 그 이전 데이터 $[x_0, x_1, \dots, x_{i-1}]$ 를 활용해 $l_i$ 를 예측하는 것이다. 실제 모델은 윈도우 단위 입력을 받아 **마지막 포인트가 이상인지** 판단하는 구조를 택한다.

저자들이 제안한 FCVAE는 크게 세 단계로 구성된다.

첫째는 **data preprocessing** 이다. 표준화, 결측/이상치 보정, 그리고 새롭게 제안한 anomaly-focused data augmentation을 수행한다.
둘째는 **training** 이다. CVAE 구조 위에 frequency condition을 얹고, CM-ELBO와 missing data injection, last-point masking을 사용해 학습한다.
셋째는 **testing** 이다. 마지막 포인트를 missing으로 간주한 뒤 MCMC 기반 missing imputation으로 정상적인 복원값을 얻고, reconstruction probability를 anomaly score로 사용한다.

### 3.2 VAE와 CVAE의 수식적 배경

논문은 먼저 VAE의 ELBO와 DONUT의 modified ELBO(M-ELBO)를 소개한다. DONUT은 윈도우 안의 이상치나 결측치가 재구성 손실에 미치는 영향을 줄이기 위해 indicator $\alpha_w$ 와 비율 $\beta$ 를 사용한다. 논문에 제시된 식은 다음과 같다.

$$
\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \sum_{w=1}^{W}\alpha_w \log p_\theta(x_w|\mathbf{z}) + \beta \log p_\theta(\mathbf{z}|\mathbf{x}) - \log q_\phi(\mathbf{z}|\mathbf{x}) \right]
$$

여기서 $\alpha_w = 1$ 이면 해당 위치가 정상 또는 결측이 아닌 값이고, $\alpha_w = 0$ 이면 이상 또는 결측의 영향을 제거하려는 뜻이다. $\beta = (\sum_{w=1}^W \alpha_w)/W$ 는 유효한 포인트 비율이다.

CVAE는 여기에 조건 변수 $\mathbf{c}$ 를 추가한 형태다. 논문에서 제시한 CVAE objective는 다음과 같다.

$$
\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z},\mathbf{c}) + \log p_\theta(\mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c}) \right]
$$

FCVAE는 바로 이 $\mathbf{c}$ 에 frequency-derived condition을 넣는 방식이다. 즉, latent variable $\mathbf{z}$ 만으로 복원하지 않고, 현재 입력 윈도우의 global/local frequency feature를 함께 제공하여 더 정교한 reconstruction을 유도한다.

### 3.3 FCVAE의 전체 구조

논문은 FCVAE를 encoder, decoder, 그리고 조건 추출 블록(condition extraction block)으로 나눈다. 조건 추출 블록에는 **GFM** 과 **LFM** 이 포함된다. 핵심 계산은 식 (3)으로 정리된다.

$$
\mu,\sigma = Encoder(\mathbf{x}, LFM(\mathbf{x}), GFM(\mathbf{x}))
$$

$$
\mathbf{z} = Sample(\mu,\sigma)
$$

$$
\mu_{\mathbf{x}}, \sigma_{\mathbf{x}} = Decoder(\mathbf{z}, LFM(\mathbf{x}), GFM(\mathbf{x}))
$$

즉, encoder와 decoder 모두가 단순 입력 $\mathbf{x}$ 뿐 아니라 $LFM(\mathbf{x})$, $GFM(\mathbf{x})$ 를 함께 받는다. 이 구조는 CVAE의 condition이 인코딩 단계와 디코딩 단계 모두에 영향을 준다는 점에서 중요하다. 논문은 frequency 정보를 input에 덧붙이는 방식(FVAE)보다, condition으로 넣는 CVAE 구조가 더 효과적이라고 실험적으로 주장한다.

### 3.4 GFM: Global Frequency Module

GFM은 전체 윈도우의 **전역 주파수 특성**을 추출한다. 입력 윈도우 전체에 FFT를 적용한 뒤, dense layer와 dropout을 거쳐 global frequency embedding을 만든다. 수식은 다음과 같다.

$$
f_{global} = Dropout(Dense(\mathcal{F}(\mathbf{x})))
$$

여기서 $\mathcal{F}$ 는 FFT다. 저자들의 설명에 따르면, 전체 주파수 스펙트럼에는 noise나 anomaly 때문에 생긴 긴 tail 성분도 포함되어 있으므로, 단순 FFT 결과를 그대로 쓰지 않고 linear layer로 유용한 성분을 필터링한다. 그리고 Fedformer를 참고해 dropout을 넣어, 일부 주파수 정보가 빠진 경우도 복원할 수 있도록 학습시킨다.

GFM의 역할은 전체 윈도우가 보이는 큰 주기 구조, 예를 들어 길고 느린 periodic pattern을 제공하는 것이다. 이는 특히 서로 비슷하지만 완전히 같지는 않은 heterogeneous periodic pattern을 복원하는 데 도움이 된다.

### 3.5 LFM: Local Frequency Module과 target attention

논문의 가장 흥미로운 설계는 LFM이다. 저자들은 GFM만으로는 윈도우 전체의 평균적인 frequency만 반영되므로, **마지막 지점의 국소적인 변화**를 충분히 반영하기 어렵다고 본다. 특히 실제 이상 탐지는 마지막 포인트를 판정하는 문제이므로, 마지막 포인트 부근의 local pattern이 중요하다.

이를 위해 전체 윈도우를 sliding window로 여러 개의 작은 sub-window $\mathbf{x}_{sw}$ 로 나눈다. 각 sub-window에 대해 FFT와 dense projection을 적용해 frequency feature를 만든다. 그리고 **가장 최근 sub-window** 를 query $Q$ 로, 나머지 sub-window를 key $K$ 와 value $V$ 로 사용하여 target attention을 계산한다. 논문에 제시된 식은 다음과 같다.

$$
\mathbf{x}_{sw} = SlidingWindow(\mathbf{x})
$$

$$
Q = Select(Dense(\mathcal{F}(\mathbf{x}_{sw})))
$$

$$
K, V = Dense(\mathcal{F}(\mathbf{x}_{sw}))
$$

$$
f_{local} = Dropout\left( FeedForward\left( \sigma(Q \cdot K^\top)\cdot V \right) \right)
$$

여기서 $\sigma$ 는 softmax이다. 직관적으로는, 마지막 포인트가 포함된 최신 sub-window와 **frequency pattern이 유사한 과거 sub-window들** 에 더 큰 가중치를 주어 local frequency representation을 만든다. 이 때문에 LFM은 단순 평균보다 훨씬 선택적으로 정보를 추출할 수 있다.

논문은 attention mechanism이 왜 필요한지도 별도로 검증한다. 단순히 latest window만 쓰거나, 모든 sub-window를 average pooling하는 방식은 사전에 어떤 window가 중요한지 알 수 없기 때문에 성능이 떨어진다. 반면 attention은 현재 query와 관련성이 큰 window에 더 높은 weight를 부여할 수 있으므로 local condition의 품질을 높인다.

### 3.6 학습 절차: CM-ELBO, last-point masking, data augmentation

FCVAE의 학습에서 중요한 것은 단순 CVAE 학습이 아니라, 이상이 섞여 있는 비지도 시계열 학습 환경에 맞는 objective를 쓰는 점이다. 이를 위해 저자들은 DONUT의 M-ELBO를 CVAE에 맞게 확장한 **CM-ELBO** 를 사용한다.

$$
\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c})} \left[ \sum_{w=1}^{W}\alpha_w \log p_\theta(x_w|\mathbf{z},\mathbf{c}) + \beta \log p_\theta(\mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c}) \right]
$$

이 식의 의미는, reconstruction term은 유지하되 이상이나 결측이 의심되는 위치의 영향은 약화하고, latent regularization은 계속 유지한다는 것이다. 저자들은 이 기법이 비지도 환경에서 특히 중요하다고 본다.

또 하나의 핵심은 **masking the last point** 이다. 논문은 마지막 포인트가 이상일 경우, 시간 영역에서는 단일 outlier일 뿐이지만 frequency 영역으로 가면 전체 주파수 성분이 흔들릴 수 있다고 설명한다. 특히 이 논문은 마지막 포인트를 판정 대상으로 삼기 때문에, 그 포인트가 condition 추출 자체를 오염시키면 문제가 커진다. 그래서 frequency condition을 추출할 때는 마지막 포인트를 0으로 masking한다. 이 설계는 논문의 문제 설정과 아주 밀접하게 연결되어 있다.

데이터 증강도 특이하다. 일반적인 time series augmentation은 정상 패턴을 다양화하는 방향이 많지만, 이 논문은 모든 시계열을 한 모델에 함께 학습시키므로 정상 패턴 다양성은 이미 충분하다고 본다. 대신 **이상 데이터 augmentation** 에 집중한다. pattern mutation은 서로 다른 두 윈도우를 이어붙여 경계 지점을 이상으로 만드는 방식이고, value mutation은 일부 포인트를 임의의 비정상 값으로 바꾸는 방식이다. 이런 synthetic anomaly를 넣어주면, 비지도 상황에서도 CM-ELBO가 이상치에 덜 민감한 복원을 배우는 데 도움이 된다고 주장한다.

### 3.7 테스트 절차와 anomaly score

테스트 단계에서는 MCMC 기반 missing imputation을 사용한다. 마지막 포인트를 missing으로 두고, 모델이 그것을 “정상적인 값”으로 복원하도록 만든다. 이는 학습 때 last-point masking을 했던 설계와 잘 맞물린다. 마지막 포인트의 reconstruction probability를 바탕으로 anomaly score를 정의하는 식은 다음과 같다.

$$
AnomalyScore = - \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x},\mathbf{c})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z},\mathbf{c}) \right]
$$

즉, reconstruction log-likelihood가 낮을수록 anomaly score는 커진다. 정상 패턴은 높은 확률로 재구성되고, 이상 패턴은 낮은 확률로만 설명되므로 이상 점수가 상승하는 구조다.

## 4. 실험 및 결과

### 4.1 데이터셋, 비교 대상, 평가 지표

실험은 네 개의 데이터셋에서 수행되었다. **Yahoo**, **KPI**, **WSD**, **NAB** 이며, 모두 시계열 이상 탐지에서 널리 쓰이는 벤치마크들이다. 논문은 특히 WSD와 KPI를 웹 서비스 운영 맥락의 실제 KPI 데이터로 소개한다.

비교 기준선은 전통 통계 방법부터 최신 딥러닝 방법까지 폭넓게 포함한다. 구체적으로 **SPOT**, **SRCNN**, **TFAD**, **DONUT**, **Informer**, **Anomaly-Transformer**, **AnoTransfer**, **VQRAE** 와 비교한다. 이 구성은 통계 기반, 지도학습 기반, 예측 기반, VAE 기반 복원 방법을 모두 아우른다는 점에서 비교가 비교적 넓다.

평가 지표는 point-wise F1이 아니라 **best F1** 과 **delay F1** 을 사용한다. 이는 운영자가 연속된 이상 구간을 조기에 포착하는 데 더 관심이 있다는 점을 반영한다. best F1은 가능한 threshold를 모두 탐색한 뒤 point adjustment strategy를 적용해 계산하고, delay F1은 허용 지연 구간 내에서 이상 세그먼트를 얼마나 빨리 잡았는지를 반영한다. 데이터셋마다 이상 세그먼트 길이가 달라, Yahoo는 delay 3, 나머지는 7 또는 NAB의 경우 150으로 설정했다.

또한 중요한 구현 설정으로, **모든 실험은 완전 비지도 조건** 에서 수행되었다. 실제 레이블은 훈련에 사용하지 않았고, 데이터셋 내 모든 곡선에 대해 하나의 모델을 학습했다.

### 4.2 전체 성능

표 1에 따르면 FCVAE는 네 데이터셋 모두에서 가장 좋은 성능을 달성했다.

Yahoo에서는 FCVAE가 best F1 **0.857**, delay F1 **0.842** 를 기록했다. 주요 경쟁 모델 중 TFAD가 best F1 0.805, delay F1 0.802였고, Informer가 0.707 / 0.671이었다. 따라서 Yahoo에서는 FCVAE의 이점이 꽤 뚜렷하다.

KPI에서는 FCVAE가 best F1 **0.927**, delay F1 **0.835** 를 기록했다. 여기서는 Informer가 best F1 0.918, delay F1 0.822로 매우 강한 기준선이지만, FCVAE가 근소하게 앞선다. 즉, KPI에서는 절대적인 큰 격차보다도, 이미 강한 baseline 위에서 추가 개선을 달성했다는 점이 의미 있다.

WSD에서는 FCVAE가 best F1 **0.831**, delay F1 **0.631** 로 가장 높다. 특히 delay F1에서 기존 방법들과 차이가 크게 난다. 예를 들어 TFAD는 0.455, Informer는 0.393, AnoTransfer는 0.379, Anomaly-Transformer는 0.137이다. 논문이 주장하는 “주파수 변화나 세밀한 패턴 왜곡에 더 강하다”는 점이 이 데이터셋에서 더 잘 드러난다고 볼 수 있다.

NAB에서는 FCVAE가 best F1 **0.976**, delay F1 **0.917** 를 기록했다. 다만 NAB에서는 Informer 0.973 / 0.892, Anomaly-Transformer 0.971 / 0.911, AnoTransfer 0.965 / 0.871 등도 매우 강하다. 따라서 이 데이터셋에서는 개선 폭이 아주 크다기보다, 최고 성능을 안정적으로 달성했다는 해석이 더 적절하다.

논문은 이를 요약해, best F1 기준으로 각각 **6.45%, 0.98%, 14.14%, 0.31%**, delay F1 기준으로 **4.98%, 1.58%, 38.68%, 0.65%** 향상되었다고 주장한다. 특히 WSD에서 delay F1 개선이 매우 크다는 점은, FCVAE가 연속 이상 구간을 더 빠르게 포착한다는 메시지를 강하게 뒷받침한다.

### 4.3 조건 정보 비교: 왜 frequency가 중요한가

논문은 CVAE에서 어떤 종류의 condition이 가장 유효한지 비교한다. 비교 대상은 timestamp, time-domain information, frequency-domain information이다. 결과적으로 **frequency condition이 가장 우수**했다고 보고한다.

이 결과는 논문의 핵심 주장을 직접 지지한다. timestamp는 정보량이 제한적이고 one-hot encoding이 필요해 sparse하다. time-domain 정보를 다시 condition으로 넣는 것은 이미 입력 자체가 시간 영역 값이므로 중복 정보일 가능성이 높다. 반면 frequency information은 입력이 직접 제공하지 못하는 보완적 prior로 작용해 reconstruction을 돕는다. 즉, 논문은 “CVAE를 쓰는 것”보다 더 구체적으로 “무엇을 condition으로 쓰는가”가 중요하며, 그 답이 frequency라고 본다.

### 4.4 FVAE와의 비교: frequency를 어떻게 쓰는가가 중요하다

논문은 frequency를 입력과 함께 VAE에 넣는 **FVAE** 와도 비교한다. 결과는 FCVAE가 FVAE보다 우수했다. 저자들의 해석은 두 가지다.

첫째, CVAE는 구조적으로 condition을 encoder와 decoder 모두에 반영하기 때문에 단순 VAE보다 조건 정보를 더 잘 활용한다.
둘째, FVAE는 frequency를 추가 정보로 넣기는 하지만, decoder에서 그 정보를 충분히 활용하지 못한다. 따라서 “frequency를 쓴다”는 사실 자체보다, **frequency를 condition으로 활용하는 CVAE 구조** 가 핵심이라는 주장이다.

이 비교는 꽤 중요하다. 왜냐하면 이 논문의 성능 향상이 단지 FFT를 넣어서 생긴 것인지, 아니면 CVAE적 설계 덕분인지를 분리해 보여 주기 때문이다.

### 4.5 GFM과 LFM의 효과

논문은 GFM만 쓰는 경우, LFM만 쓰는 경우, 둘 다 쓰는 경우를 비교한다. 대부분 데이터셋에서 GFM 또는 LFM 단독 사용도 기본 VAE보다 낫고, **둘을 함께 사용할 때 가장 좋다**고 보고한다. 이는 global frequency와 local frequency가 서로 중복이라기보다 상보적이라는 뜻이다.

다만 NAB에서는 GFM의 정보와 현재 시점의 값 사이 불일치가 생길 수 있다고 논문이 언급한다. NAB는 데이터가 자주 oscillation하기 때문에, 전역 frequency 요약이 현재 시점 판단에 완전히 맞지 않을 수 있다는 것이다. 그럼에도 전체적으로는 GFM+LFM 조합이 가장 우수했다.

이 결과는 논문이 제안한 decomposition이 단순한 모듈 추가가 아니라 실제로 역할 분담이 된다는 점을 보여 준다. GFM은 장기 구조를, LFM은 마지막 지점 관련 local structure를 담당한다고 해석할 수 있다.

### 4.6 Attention mechanism의 역할

LFM의 성능 향상이 단지 작은 윈도우를 쓰기 때문인지, 아니면 attention 자체 때문인지를 확인하기 위해, 논문은 attention을 제거한 변형도 실험했다. 하나는 latest small window만 사용하는 방식이고, 다른 하나는 average pooling 방식이다. 결과는 둘 다 원래 FCVAE보다 못했다.

저자들은 attention이 각 small window의 중요도를 사전에 알 수 없는 상황에서, query와 관련된 sub-window에 더 높은 weight를 주는 것이 핵심이라고 해석한다. Figure 10의 사례 설명도 이 해석과 맞닿아 있다. 최신 window와 가장 유사한 5번째 window가 실제로 attention heatmap에서 가장 높은 값을 받았다고 한다. 이는 target attention이 단순 averaging이 아니라, **현재 탐지 대상과 닮은 과거 local frequency pattern을 선택적으로 참조** 한다는 점을 보여준다.

### 4.7 핵심 학습 기법의 ablation

표 2는 세 가지 학습 기법의 중요도를 보여 준다. FCVAE 완전형과 비교해 보면:

- **w/o data augment**: Yahoo 0.841, KPI 0.825, WSD 0.626, NAB 0.904
- **w/o mask last point**: Yahoo 0.835, KPI 0.830, WSD 0.534, NAB 0.877
- **w/o CM-ELBO**: Yahoo 0.690, KPI 0.757, WSD 0.435, NAB 0.897
- **FCVAE**: Yahoo 0.842, KPI 0.835, WSD 0.631, NAB 0.917

이 결과에서 가장 두드러지는 것은 **CM-ELBO 제거 시 성능이 가장 크게 떨어진다**는 점이다. 특히 Yahoo와 WSD에서 낙폭이 크다. 이는 비지도 이상 탐지에서 이상/결측 오염을 견디는 학습 objective가 매우 중요하다는 저자들의 주장과 일치한다.

또한 **mask last point** 도 중요하다. 특히 WSD에서 0.631에서 0.534로 큰 하락이 나타난다. 이는 마지막 포인트의 이상이 frequency condition 전체를 오염시킬 수 있다는 논문의 문제의식이 실제로 타당함을 보여 준다.

데이터 증강의 효과는 상대적으로 작지만 일관되게 긍정적이다. 즉, 이상 synthetic sample을 도입하는 것이 비지도 학습 안정성에 보조적 역할을 하는 것으로 보인다.

### 4.8 파라미터 민감도와 실제 적용

논문은 KPI와 WSD에서 조건 embedding dimension, window size, missing data injection rate, data augmentation rate를 바꿔 보며 민감도 실험을 수행했다. 구체적인 수치나 최적값은 본문 발췌에 자세히 적혀 있지 않지만, 저자들은 다양한 설정에서도 안정적이고 우수한 성능을 보였다고 주장한다. 따라서 실무 적용 시 하이퍼파라미터에 과도하게 민감하지 않다는 메시지를 전달한다.

더 나아가 FCVAE는 실제 대규모 클라우드 시스템에 적용되었다고 한다. 하루에 billions of time series points가 생성되는 시스템에서, 24GB 메모리의 3090 GPU 기준으로 **1195.7 points/second** 의 inference efficiency를 보였고, 기존 detector 대비 **F1 10.9%**, **delay F1 11.1%** 향상을 달성했다. 온라인 성능은 legacy detector 기준 F1 0.66, F1*0.63에서 FCVAE가 F1 0.73, F1* 0.69로 향상된 것으로 제시된다.

이 결과는 논문이 단순 벤치마크 성능 향상에 머무르지 않고, 실제 운영 환경에서도 속도와 정확도 측면에서 의미 있는 성과를 보였음을 보여 준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **VAE 기반 시계열 이상 탐지의 실패 원인을 frequency 관점에서 다시 해석했다는 점** 이다. 많은 논문이 더 복잡한 구조를 제안하는 데 그치지만, 이 논문은 왜 기존 복원 모델이 정상 구간의 미세한 패턴을 놓치는지를 “missing frequency information”이라는 설명으로 연결한다. 그리고 그 가설에 맞춰 condition 설계, global/local 분해, attention, 학습 objective까지 일관되게 구성했다.

또 다른 강점은 **실험 설계가 가설 검증형** 이라는 점이다. 단순히 최종 모델 성능만 보여 주지 않고, timestamp vs time vs frequency condition 비교, FVAE vs FCVAE 비교, GFM/LFM 분리, attention 제거, CM-ELBO와 mask-last-point 제거 등의 ablation을 통해 각 요소의 역할을 비교적 설득력 있게 제시한다. 이 때문에 모델 설계 선택이 임의적이라기보다 문제 정의와 연결되어 있다는 인상을 준다.

실용성도 강점이다. 완전 비지도 조건에서 학습하고, 모든 곡선을 하나의 모델로 학습하며, 실제 대규모 클라우드 시스템에 배포해 성능과 추론 효율을 보고했다는 점은 AIOps나 운영 이상 탐지 분야에서 가치가 크다.

다만 한계도 분명하다. 첫째, 논문은 **univariate time series** 에 초점을 둔다. 실제 시스템 운영에서는 multivariate dependency가 매우 중요한데, FCVAE가 다변량 상황에서도 그대로 통할지는 이 논문만으로는 알 수 없다. 관련 문제는 논문에 직접적으로 해결되지 않는다.

둘째, frequency 정보가 항상 유리한 것은 아닐 수 있다. 논문도 NAB에서 GFM의 정보와 현재 시점 값의 불일치 가능성을 언급한다. 즉, 급격한 non-stationary 변화나 불규칙한 구조에서는 frequency prior가 오히려 현재 상태를 덜 정확히 반영할 수 있다. 이 문제를 얼마나 일반적으로 해결할 수 있는지는 아직 남아 있다.

셋째, 제시된 수식과 설명은 핵심 구조를 전달하지만, 실제 구현 세부는 일부 생략되어 있다. 예를 들어 encoder/decoder의 구체적 네트워크 유형, latent dimension의 구체 설정, FFT 결과의 어떤 성분을 어떻게 정규화하는지, MCMC 추론의 반복 횟수 같은 세부 구현은 발췌 텍스트만으로는 충분히 알 수 없다. 따라서 재현 시에는 공개 코드가 중요할 것이다.

넷째, anomaly score를 reconstruction probability로 두고 마지막 포인트를 missing 처리하는 전략은 이 논문의 문제 설정과 잘 맞지만, 탐지 대상이 마지막 포인트 하나가 아니라 구간 전체이거나 online latency 요구가 다른 환경에서는 설계가 다르게 필요할 수 있다. 즉, 이 방법은 “window의 마지막 포인트 판정”에 최적화된 구조이지, 모든 형태의 이상 탐지 문제에 보편적이라고 보기는 어렵다.

비판적으로 보면, 이 논문은 VAE 계열 방법의 성능을 크게 개선했지만, 근본적으로는 여전히 reconstruction-based anomaly detection의 틀 안에 있다. 따라서 정상 패턴의 다양성이 아주 크거나, 이상이 정상과 거의 같은 재구성 가능성을 갖는 경우에는 한계가 남을 수 있다. 논문은 prediction-based 방법보다 mixed anomaly-normal training data에 더 강하다고 주장하지만, 이 역시 데이터 특성에 따라 상대적일 수 있다.

## 6. 결론

이 논문은 비지도 UTS 이상 탐지에서 VAE 기반 접근을 다시 강화하기 위해, **frequency information을 CVAE의 condition으로 도입한 FCVAE** 를 제안한다. 핵심 기여는 단순히 FFT를 추가한 것이 아니라, **GFM으로 global frequency**, **LFM으로 local frequency**, 그리고 **target attention으로 마지막 시점 관련 정보 선택** 을 결합해 정상 복원의 정확도를 높였다는 점이다. 여기에 **CM-ELBO**, **masking the last point**, **anomaly-focused data augmentation** 을 더해 비지도 환경에서도 강한 탐지기를 구성했다.

실험 결과는 네 개의 공개 데이터셋과 실제 대규모 클라우드 시스템에서 모두 우수한 성능을 보여 준다. 특히 WSD에서의 delay F1 개선과 실제 운영 환경에서의 online improvement는, 이 방법이 단순 학술적 기법이 아니라 실전 배치 가능한 접근임을 시사한다.

향후 연구 관점에서 보면, 이 논문은 최소 두 가지 중요한 방향을 연다. 하나는 **frequency-aware anomaly detection** 이라는 설계 원리를 더 일반화하는 방향이다. 다른 하나는 이를 multivariate setting, concept drift가 더 심한 환경, 또는 다른 generative model과 결합하는 방향이다. 실제 적용 측면에서는 웹 서비스, 클라우드 운영, KPI 모니터링처럼 주기성과 국소 패턴이 동시에 중요한 환경에서 매우 유용할 가능성이 크다.

전체적으로 이 논문은 “VAE는 이미 오래된 방법”이라는 인식을 뒤집고, **어떤 정보를 condition으로 주고 어떤 구조로 추출하느냐에 따라 여전히 state-of-the-art 수준의 성능을 낼 수 있다**는 점을 설득력 있게 보여 준다. 특히 정상 복원의 실패 원인을 frequency 관점에서 분석하고, 이를 구조적 개선으로 연결했다는 점에서 의미 있는 작업이다.
