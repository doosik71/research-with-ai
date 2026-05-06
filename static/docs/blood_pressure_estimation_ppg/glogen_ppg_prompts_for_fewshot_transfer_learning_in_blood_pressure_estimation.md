# GloGen: PPG prompts for few-shot transfer learning in blood pressure estimation

* **저자**: Taero Kim, Hyeonjeong Lee, Minseong Kim, Kwang-Yong Kim, Kyu Hyung Kim, Kyungwoo Song
* **발표연도**: 2024
* **학술지**: Computers in Biology and Medicine, Volume 183, Article 109216
* **DOI**: [https://doi.org/10.1016/j.compbiomed.2024.109216](https://doi.org/10.1016/j.compbiomed.2024.109216)

## 1. 논문 개요

이 논문은 PPG(Photoplethysmogram) 신호만을 사용하여 수축기 혈압(Systolic Blood Pressure, SBP)과 이완기 혈압(Diastolic Blood Pressure, DBP)을 추정하는 문제를 다룬다. 논문의 핵심 목표는 단순히 평균적인 혈압 예측 오차를 낮추는 것이 아니라, 사전학습된 PPG 기반 혈압 추정 모델을 매우 적은 target task 데이터만으로 새로운 데이터셋에 안정적으로 전이시키고, 특히 저혈압(hypotension) 및 고혈압(hypertension)과 같은 고위험 혈압 그룹에서도 예측 성능을 유지하도록 만드는 것이다.

혈압은 심혈관 건강을 평가하는 핵심 생체 지표이다. 기존 cuff 기반 혈압 측정 방식은 신뢰할 수 있지만 장비가 번거롭고, 연속적이고 일상적인 혈압 모니터링에는 불편하다. 반면 PPG는 혈액량 변화나 혈류 속도 변화에 따른 빛 흡수 정도를 측정하는 생체 신호로, 웨어러블 기기에서 비침습적으로 측정할 수 있다. 따라서 PPG를 이용한 혈압 추정은 실시간, 연속적, 비침습적 건강 모니터링을 가능하게 할 수 있다는 점에서 의료 및 디지털 헬스케어 분야에서 중요한 연구 주제이다.

논문이 제기하는 주요 연구 문제는 다음과 같다. 첫째, PPG 기반 혈압 추정 모델은 전체 평균 성능은 괜찮아 보일 수 있지만, 혈압 구간별로 보면 성능 편차가 크다. 특히 정상 혈압군보다 저혈압군과 고혈압군에서 예측 오차가 커지는 경향이 있다. 그러나 실제 의료 현장에서는 바로 이러한 고위험군에서 더 정확한 혈압 추정이 필요하다. 둘째, 의료 데이터는 충분한 양의 target task 데이터를 확보하기 어렵다. 새로운 병원, 새로운 센서, 새로운 환자 집단에 모델을 적용하려면 해당 target domain의 데이터가 필요하지만, 의료 데이터의 수집 비용과 개인정보 문제 때문에 충분한 데이터를 확보하기 어렵다. 셋째, 기존 transfer learning 방식인 fine-tuning이나 linear probing은 few-shot 환경에서 불안정할 수 있으며, 특정 혈압 그룹에 편향된 사전학습 모델을 그대로 이어받을 위험이 있다.

이를 해결하기 위해 논문은 **GloGen(Global Prompt and Prompt Generator)** 이라는 few-shot transfer learning 프레임워크를 제안한다. GloGen은 입력 PPG 신호에 직접 prompt를 더하는 방식으로 사전학습 모델을 target task에 적응시킨다. 이때 모든 샘플에 공통적으로 적용되는 **Global Prompt(GP)** 와 각 샘플마다 다르게 생성되는 **Instance-wise Prompt(IP)** 를 함께 사용한다. 또한 IP가 혈압 그룹별로 충분히 다양하게 생성되도록 **Variance Penalty(VP)** 라는 regularization을 도입한다.

이 논문의 중요성은 PPG 기반 혈압 추정에서 평균 성능뿐 아니라 혈압 그룹별 robustness를 명시적으로 다룬다는 점에 있다. 많은 기존 연구가 전체 test set에 대한 평균 MAE만 보고하는 데 비해, 이 논문은 혈압 그룹별 MAE와 group average MAE를 함께 제시하여 모델이 특정 혈압 구간에 편향되지 않는지를 평가한다. 이는 실제 의료 AI 시스템의 안전성과 공정성 측면에서 중요한 관점이다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 PPG 신호를 직접 수정하는 prompt를 학습함으로써, 사전학습된 혈압 추정 모델이 새로운 target dataset에서도 잘 작동하도록 만드는 것이다. 여기서 prompt는 자연어 처리에서의 textual prompt가 아니라, 입력 PPG 신호와 같은 차원을 갖는 learnable signal이다. 즉, 원본 PPG 신호 $x_i$에 학습된 보정 신호를 더하여 모델이 더 잘 해석할 수 있는 형태의 입력으로 바꾸는 방식이다.

기존 PPG 기반 혈압 추정 연구는 주로 모델 아키텍처를 개선하는 방향으로 진행되었다. 예를 들어 1D convolution 기반 모델, spectro-temporal network, attention 기반 recurrent model 등이 사용되었다. 반면 이 논문은 backbone model 자체를 직접 바꾸기보다는, 입력 신호에 prompt를 추가하여 사전학습 모델을 target task에 적응시킨다. 논문은 이 방식이 모델 구조에 상대적으로 덜 의존적이며, embedding을 얻을 수 있는 모델이라면 적용 가능하다고 설명한다.

GloGen의 핵심 설계는 두 종류의 prompt를 동시에 사용하는 dual-prompt learning이다. 첫 번째는 **Global Prompt(GP)** 이다. GP는 모든 입력 PPG 신호에 동일하게 더해지는 trainable parameter이다. 이는 target dataset에 공통적으로 존재하는 PPG 패턴, 센서 특성, 도메인 차이 등을 보정하는 역할을 한다. 즉 GP는 target task 전체에 공유되는 특성을 학습하는 prompt라고 볼 수 있다.

두 번째는 **Instance-wise Prompt(IP)** 이다. IP는 모든 샘플에 동일하게 적용되지 않고, 각 입력 PPG 신호의 embedding을 바탕으로 Prompt Generator가 샘플마다 생성한다. 이는 개별 환자 또는 개별 신호의 특성을 반영하기 위한 personalized prompt이다. 특히 저혈압이나 고혈압처럼 사전학습 데이터에서 충분히 대표되지 못했을 가능성이 있는 그룹에 대해, 각 샘플에 맞는 보정 신호를 생성하여 예측 robustness를 높이는 것이 목적이다.

이 dual-prompt 구조의 직관은 다음과 같다. 혈압 추정에서 모든 PPG 신호가 공유하는 일반적인 형태적 특징도 중요하지만, 같은 혈압 구간 내에서도 개인차, 측정 환경, 센서 특성, waveform 형태 차이가 존재한다. 따라서 하나의 공통 prompt만으로는 충분하지 않고, 모든 신호를 독립적으로 처리하는 instance-specific prompt만으로도 target domain 전체의 공통 특성을 안정적으로 반영하기 어렵다. GloGen은 GP와 IP를 동시에 사용하여 dataset-level adaptation과 instance-level adaptation을 함께 수행한다.

또 하나의 중요한 아이디어는 **Variance Penalty(VP)** 이다. 의료 데이터는 혈압 그룹별 분포가 불균형하다. 예를 들어 BCG dataset에서는 Hypo 그룹이 18개뿐인 반면 Normal 그룹은 1466개이다. 이러한 imbalance에서 사전학습 모델은 특정 다수 그룹에 최적화될 수 있고, Prompt Generator도 비슷한 prompt만 생성하는 방향으로 학습될 수 있다. 이를 방지하기 위해 논문은 BP group별 IP 평균이 서로 충분히 달라지도록 VP를 설계한다. 즉 VP는 IP가 혈압 그룹별로 구분되는 다양한 보정 신호를 만들도록 유도한다.

기존 transfer learning 방식과 비교하면, 이 논문의 차별점은 다음과 같다. Fine-tuning은 모델 전체를 업데이트하므로 few-shot 환경에서 과적합되기 쉽고, linear probing은 마지막 회귀층만 바꾸므로 domain shift에 충분히 대응하지 못할 수 있다. GloGen은 feature extractor를 기본적으로 고정한 상태에서 prompt와 일부 regression layer를 학습하기 때문에 parameter-efficient adaptation에 가깝다. 또한 단순히 전체 MAE를 줄이는 것이 아니라 group average MAE와 high-risk group 성능을 함께 고려한다는 점에서 의료적 의미가 크다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

![Fig. 1. Overview of GloGen model.](https://ars.els-cdn.com/content/image/1-s2.0-S0010482524013015-gr1.jpg)

GloGen의 전체 흐름은 다음과 같다. 먼저 입력 PPG 신호 $x_i$가 주어진다. 이 신호는 encoder $h$를 통과하여 embedding $z_i$로 변환된다. 이 embedding은 Prompt Generator $g_\phi$에 입력되어 해당 샘플에 맞는 Instance-wise Prompt $x_i^{IP}$를 생성한다. 동시에 모든 샘플에 공통적으로 적용되는 Global Prompt $x^{GP}$가 학습된다. 최종적으로 원본 PPG 신호에 GP와 IP를 더한 $x_i^{GloGen}$이 사전학습된 혈압 추정 모델 $f$에 입력되고, 모델은 SBP와 DBP를 예측한다.

논문에서 사전학습 모델은 다음과 같이 표현된다.

$$
f = f_{\theta}^{pen} \circ f_{\varphi}^{reg}
$$

여기서 $f_{\theta}^{pen}$은 마지막 layer 직전까지의 feature extractor 또는 penultimate network를 의미하고, $f_{\varphi}^{reg}$는 SBP와 DBP를 예측하는 regression layer를 의미한다. 논문은 기본적으로 $f_{\theta}^{pen}$을 고정하고, Prompt Generator, Global Prompt, 그리고 regression layer를 target task에 맞게 학습한다.

이 구조의 중요한 특징은 backbone feature extractor를 크게 변경하지 않는다는 점이다. 따라서 사전학습 모델이 이미 학습한 PPG representation을 보존하면서, target dataset에 필요한 보정만 prompt를 통해 수행한다.

### 3.2 Global Prompt와 Instance-wise Prompt

입력 PPG 신호를 $x_i$라고 하자. GP와 IP는 모두 입력 PPG 신호와 같은 차원을 가진다. 이는 prompt가 원본 신호에 직접 더해질 수 있도록 하기 위함이다.

Global Prompt는 모든 샘플에 동일하게 적용되는 학습 가능한 벡터이다. 논문에서는 이를 $x^{GP}$로 나타낸다. 이 prompt는 target dataset 전체에 공유되는 특징을 학습한다. 예를 들어 target dataset의 센서 특성, waveform scale 차이, preprocessing 차이 등 전체 입력 분포에 공통적으로 나타나는 차이를 보정하는 역할을 할 수 있다. 다만 논문은 GP가 구체적으로 어떤 생리학적 의미를 갖는지까지는 직접 해석하지 않는다.

Instance-wise Prompt는 각 샘플마다 다르게 생성된다. Prompt Generator는 encoder가 만든 embedding을 받아 PPG 신호와 같은 길이의 prompt를 출력한다.

$$
x_i^{IP} = g_\phi \circ h(x_i) = g_\phi(z_i)
$$

여기서 $g_\phi$는 Prompt Generator이고, $h$는 encoder이다. 기본 설정에서 $z_i$는 사전학습 모델의 feature extractor $f_{\theta}^{pen}$이 생성한 embedding이다. Trigger vector를 사용하는 경우에는 $z_i$가 embedding과 trigger vector를 concat한 형태가 된다.

Prompt Generator의 구조는 deconvolution layer, 즉 transposed convolution layer, batch normalization, ReLU activation으로 구성된다. 논문은 Prompt Generator를 encoder-decoder 구조에서 decoder에 해당한다고 설명한다. 입력 embedding은 상대적으로 압축된 representation이고, Prompt Generator는 이를 원래 PPG 신호와 같은 차원의 prompt로 복원한다.

### 3.3 최종 GloGen 입력 구성

GloGen이 생성하는 최종 입력은 원본 PPG 신호, GP, IP의 합으로 구성된다.

$$
x_i^{GloGen} = x_i + \lambda \cdot x^{GP} + \gamma \cdot x_i^{IP}
$$

여기서 $\lambda$는 Global Prompt의 영향력을 조절하는 계수이고, $\gamma$는 Instance-wise Prompt의 영향력을 조절하는 계수이다. 이 두 값은 hyperparameter로 탐색된다.

이 수식의 의미는 직관적으로 다음과 같다. 원본 PPG 신호 $x_i$는 그대로 유지하되, target dataset 전체에 공통적인 보정 항 $\lambda \cdot x^{GP}$를 더하고, 각 샘플에 맞는 개인화 보정 항 $\gamma \cdot x_i^{IP}$를 추가한다. 따라서 GloGen은 입력 신호를 사전학습 모델이 더 잘 해석할 수 있는 형태로 재구성한다.

이 방식은 data augmentation과도 유사한 면이 있다. 다만 일반적인 augmentation이 hand-crafted transformation을 적용하는 반면, GloGen의 prompt는 target task의 supervised signal, 즉 SBP와 DBP label을 이용해 직접 학습된다.

### 3.4 Encoder 선택과 Trigger Vector

GloGen은 Prompt Generator에 입력할 embedding을 만드는 방식으로 두 가지 선택지를 제시한다.

첫 번째는 사전학습 모델의 feature extractor를 encoder로 사용하는 것이다.

$$
h = f_{\theta}^{pen}
$$

이 방식의 장점은 사전학습 모델이 이미 학습한 PPG representation을 활용할 수 있다는 점이다. 사전학습 데이터가 충분하고 모델이 안정적으로 학습되었다면, 이 embedding은 prompt 생성에 유용한 정보를 포함할 수 있다. 그러나 사전학습 모델이 특정 혈압 그룹에 편향되어 있거나 target dataset과 분포 차이가 크다면, embedding 자체가 prompt 생성에 적합하지 않을 수 있다.

두 번째는 PCA encoder를 사용하는 것이다. 이 경우 few-shot training data로 PCA projection matrix를 만들고, test data를 같은 PCA space에 projection하여 embedding을 얻는다. 논문은 PCA encoder가 사전학습 모델의 bias가 강할 때 대안이 될 수 있으며, inference 속도 측면에서도 장점이 있다고 설명한다. 다만 제공된 본문에서는 PCA encoder를 사용한 구체적인 성능 비교표가 별도로 자세히 제시되지는 않는다.

논문은 추가적으로 **trigger vector** $Z_{trigger}$를 도입한다. 이는 randomly initialized trainable vector이며, 각 instance마다 달라지는 것이 아니라 target task 전체에서 공유된다. Trigger vector를 사용하는 경우 Prompt Generator의 입력은 다음과 같은 형태가 된다.

$$
z_i \in [Z_{embed}, Z_{trigger}]
$$

여기서 $Z_{embed}$는 encoder가 만든 instance-specific embedding이고, $Z_{trigger}$는 target task에 필요한 추가 정보를 학습하기 위한 공유 벡터이다.

GP와 trigger vector는 모두 모든 샘플에 공유된다는 점에서 비슷해 보일 수 있지만, 역할이 다르다. GP는 최종 PPG 입력에 직접 더해지는 signal-level prompt이다. 반면 trigger vector는 Prompt Generator의 입력 embedding에 추가되는 latent-level vector이다. 즉 GP는 입력 신호를 직접 수정하고, trigger vector는 IP를 더 잘 생성하기 위한 조건 정보로 사용된다.

### 3.5 학습 목표와 MSE Loss

GloGen은 SBP와 DBP를 직접 예측하는 regression task로 학습된다. 기본 손실 함수는 Mean Squared Error(MSE)이다.

$$
L_{MSE} = \frac{1}{N} \sum_i \left( f(x_i^{GloGen}) - y_i \right)^2
$$

여기서 $N$은 training data의 개수이고, $y_i$는 입력 PPG 신호 $x_i$에 대응하는 ground-truth SBP와 DBP이다. 수식에서는 간단히 하나의 예측값처럼 표현되어 있지만, 실제 task는 SBP와 DBP를 함께 추정하는 회귀 문제이다.

학습 중에는 기본적으로 $f_{\theta}^{pen}$은 freeze된다. 즉 feature extractor는 업데이트하지 않는다. 대신 다음 요소들이 업데이트된다.

첫째, Prompt Generator $g_\phi$가 업데이트된다. 둘째, Global Prompt $x^{GP}$가 업데이트된다. 셋째, regression layer $f_{\varphi}^{reg}$가 target task에 맞게 업데이트된다. Trigger vector를 사용하는 설정에서는 $Z_{trigger}$도 업데이트된다. PCA encoder를 사용하는 경우에는 target task training data를 이용해 PCA projection matrix를 구성한다.

이러한 학습 방식은 full fine-tuning보다 업데이트되는 파라미터 수가 적고, few-shot 환경에서 과적합 위험을 줄이는 방향으로 설계되어 있다.

### 3.6 Prompt Normalization과 Clipping

논문은 prompt가 원본 PPG 신호보다 지나치게 큰 값을 가지면 원본 신호 정보가 가려질 수 있다고 지적한다. 예를 들어 $x^{GP}$나 $x_i^{IP}$의 magnitude가 너무 커지면, 모델은 실제 PPG waveform보다 prompt에 의해 왜곡된 신호를 보게 된다. 이는 학습 불안정성과 예측 오류로 이어질 수 있다.

이를 방지하기 위해 논문은 training dataset의 statistics를 이용한 normalization을 도입한다. 입력 PPG training set을 $X_{tr}$라고 할 때, 전체 training signal의 최솟값과 최댓값을 각각 $\min(X_{tr})$, $\max(X_{tr})$로 정의한다. 어떤 prompt 벡터 $x_i^P$에 대해 정규화된 prompt는 다음과 같이 계산된다.

$$
\hat{x}_i^P = \frac{ \max(X_{tr}) - \min(X_{tr}) }{ \max(x_i^P) - \min(x_i^P) } \left( x_i^P - \min(x_i^P) \right) + \min(X_{tr})
$$

여기서 $x_i^P$는 $x^{GP}$, $x_i^{IP}$ 또는 $x_i^{GloGen}$ 중 하나일 수 있다. 이 normalization은 prompt의 값 범위를 training PPG signal의 값 범위와 맞춤으로써, prompt가 원본 신호를 과도하게 압도하지 않도록 한다.

또한 논문은 clipping도 hyperparameter로 고려한다. Clipping은 최종 GloGen 입력 값이 training set의 signal range를 벗어나지 않도록 제한한다.

$$
\hat{x}_{ij}^{GloGen} = \begin{cases}
\min(X_{tr}) & \text{if } x_{ij}^{GloGen} < \min(X_{tr}) \newline
x_{ij}^{GloGen} & \text{if } \min(X_{tr}) \le x_{ij}^{GloGen} \le \max(X_{tr}) \newline
\max(X_{tr}) & \text{if } \max(X_{tr}) < x_{ij}^{GloGen}
\end{cases}
$$

이 clipping은 prompt learning의 안정성을 높이기 위한 장치이다. 논문은 어떤 prompt를 normalize할지, clipping을 적용할지를 hyperparameter로 설정한다.

### 3.7 Variance Penalty

Variance Penalty는 이 논문의 방법론에서 가장 중요한 regularization이다. 의료 데이터는 혈압 그룹별 데이터 수가 불균형하기 때문에, 사전학습 모델이 특정 BP group에 편향될 가능성이 높다. 예를 들어 Normal group 데이터가 압도적으로 많으면 모델은 Normal group에 대해서는 비교적 좋은 성능을 보이지만 Hypo 또는 Hyper2 group에서는 큰 오차를 낼 수 있다.

GloGen은 IP가 혈압 그룹별로 서로 다른 특성을 반영하도록 만들기 위해 VP를 도입한다. 혈압 그룹 index를 $k$라고 하고, 그룹 $k$에 속한 PPG 신호의 수를 $G_k$라고 하자. IP는 길이 $D$의 시계열 prompt이며, $x_{ij,k}^{IP}$는 그룹 $k$에 속한 $i$번째 샘플의 IP에서 $j$번째 time step 값을 의미한다.

먼저 그룹 $k$ 내 IP의 time step별 평균을 계산한다.

$$
\bar{x}_{j,k}^{IP} = \frac{1}{G_k} \sum_i x_{ij,k}^{IP}
$$

그 다음 전체 그룹 평균을 다음과 같이 정의한다.

$$
\mu_j^{IP} = \frac{1}{K} \sum_k \bar{x}_{j,k}^{IP}
$$

여기서 $K$는 BP group의 수이다. 논문에서는 Hypo, Normal, Prehyper, Hyper2의 네 그룹을 사용하므로 $K=4$이다.

Variance Penalty는 각 그룹의 IP 평균이 전체 그룹 평균에서 얼마나 떨어져 있는지를 측정한다.

$$
VP = \frac{1}{D} \cdot \frac{1}{K} \sum_j \sum_k \left( \mu_j^{IP} - \bar{x}_{j,k}^{IP} \right)^2
$$

이 값이 크다는 것은 혈압 그룹별 IP 평균이 서로 더 다르다는 뜻이다. 논문은 같은 BP group에 속한 PPG 신호들이 서로 더 유사하고, 다른 BP group에 속한 신호들은 더 다를 것이라는 가정을 바탕으로, BP group별 IP가 서로 구분되도록 유도한다.

그러나 VP를 무작정 크게 만들기 위해 $L_{MSE} - VP$를 직접 최소화하면 학습이 불안정해진다. VP만 커지고 실제 혈압 예측을 위한 prompt 학습이 제대로 이루어지지 않을 수 있기 때문이다. 이를 해결하기 위해 논문은 margin $m$과 ReLU를 사용한다.

최종 objective는 다음과 같다.

$$
L = L_{MSE} + \alpha \cdot ReLU(m - VP)
$$

여기서 $\alpha$는 MSE와 VP regularization 사이의 상대적 영향력을 조절하는 hyperparameter이고, $m$은 VP가 최소한 어느 정도 이상 커지도록 유도하는 margin이다.

이 loss의 의미는 명확하다. 만약 $VP$가 $m$보다 작으면 $ReLU(m - VP)$가 양수가 되어 penalty가 발생한다. 따라서 모델은 IP 다양성을 높이도록 학습된다. 반면 $VP$가 이미 $m$ 이상이면 penalty가 0이 되어, 더 이상 무리하게 VP를 키우지 않고 MSE를 줄이는 데 집중할 수 있다. 이 설계는 prompt diversity와 학습 안정성 사이의 균형을 잡기 위한 실용적인 방법이다.

### 3.8 추론 절차

추론 시에는 입력 PPG 신호 $x_i$가 encoder를 통과하여 embedding $z_i$를 만든다. Prompt Generator는 이 embedding을 바탕으로 IP를 생성한다. 학습된 GP와 생성된 IP를 원본 PPG 신호에 더해 $x_i^{GloGen}$을 만들고, 이를 사전학습 모델에 입력하여 SBP와 DBP를 예측한다.

중요한 점은 testing 단계에서는 BP group label을 사용하지 않는다는 것이다. BP group 정보는 training 중 VP를 계산하기 위해 사용된다. 하지만 inference 단계에서는 입력 PPG 신호와 모델 예측만 사용한다. 따라서 실제 적용 환경에서 사전에 환자의 혈압 그룹을 알고 있어야 하는 방식은 아니다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 PPG 신호로 SBP와 DBP를 예측하는 regression task를 수행한다. Few-shot transfer learning을 평가하기 위해 BCG, Sensors, UCI라는 세 benchmark dataset을 재구성한다. 이 데이터셋들은 기존 연구에서 PPG 기반 비침습 혈압 추정 benchmark로 사용된 데이터셋이다.

각 데이터 instance는 SBP와 DBP 기준에 따라 네 개 BP group 중 하나로 분류된다. 논문은 지나치게 높거나 낮은 SBP/DBP 값을 outlier로 제거한다. 혈압 그룹 기준은 다음과 같다.

| BP group | SBP 범위(mmHg) | DBP 범위(mmHg) |
| :------: | :------------: | :------------: |
|   Hypo   |     80–90      |     40–60      |
|  Normal  |     90–120     |     60–80      |
| Prehyper |    120–140     |     80–90      |
|  Hyper2  |    140–180     |     90–120     |

SBP와 DBP가 서로 다른 그룹 기준을 만족하는 경우에는 더 높은 BP group으로 instance를 배정한다. 예를 들어 SBP는 Normal 기준이지만 DBP가 Prehyper 기준이면 해당 instance는 Prehyper로 분류된다.

데이터셋별 group distribution은 매우 불균형하다. BCG dataset은 총 3053개 instance를 포함하며 Hypo는 18개, Normal은 1466개, Prehyper는 1274개, Hyper2는 295개이다. Sensors dataset은 총 10829개 instance를 포함하며 Hypo는 78개, Normal은 2849개, Prehyper는 3890개, Hyper2는 4012개이다. UCI dataset은 총 400457개 instance를 포함하며 Hypo는 3881개, Normal은 127042개, Prehyper는 135632개, Hyper2는 133902개이다. 추가로 MIMIC dataset은 총 1070081개 instance를 포함하며 large-scale pretraining 실험에 사용된다.

Few-shot 설정에서는 target task training set에서 각 BP group별로 5개 또는 10개 샘플만 선택한다. 즉 5-shot setting에서는 총 20개, 10-shot setting에서는 총 40개의 training sample만 사용되는 구조이다. Validation set은 target task의 validation fold에서 group별 5-shot으로 구성한다. Test set은 전체 test set을 사용하여 평가한다.

Backbone architecture로는 ResNet1D를 사용한다. ResNet1D는 이미지용 ResNet의 2D convolution을 1D convolution으로 바꾼 구조이며, time-series data 처리에 적합하다. Residual block, skip connection, batch normalization, ReLU activation을 포함한다. 논문은 여러 transfer learning 방법을 공정하게 비교하기 위해 모든 실험에서 동일한 ResNet1D architecture를 사용한다.

### 4.2 비교 대상

논문은 GloGen을 다음 세 baseline과 비교한다.

**Scratch**는 target task few-shot data만 사용하여 모델을 처음부터 학습하는 방식이다. Few-shot 환경에서는 데이터가 매우 적기 때문에 일반적으로 성능이 낮을 가능성이 높다.

**LP(Linear Probing)** 는 사전학습 모델의 마지막 regression layer를 random initialization한 뒤, 이 마지막 layer만 target task에 맞게 학습하는 방식이다. Feature extractor는 고정된다. 이 방식은 안정적일 수 있지만 target domain shift를 충분히 보정하지 못할 수 있다.

**FT(Fine-Tuning)** 는 사전학습 모델 전체를 target task에 맞게 fine-tuning하는 방식이다. 모델 전체를 업데이트하므로 표현력이 높지만, few-shot 환경에서는 과적합되거나 특정 group에 편향될 가능성이 있다.

**GloGen**은 제안 방법이다. 사전학습 모델의 feature extractor를 기본적으로 freeze한 상태에서 GP, Prompt Generator, regression layer 등을 학습한다.

### 4.3 평가 지표

논문은 Mean Absolute Error(MAE)를 주요 평가 지표로 사용한다. 전체 test distribution에 대한 평균 MAE를 $L_{Data}$로 정의한다.

$$
L_{Data} = \frac{1}{N} \sum_{i=1}^{N} |f(x_i) - y_i|
$$

여기서 $N$은 test set 전체 sample 수이다. GloGen을 평가할 때 $x_i$는 prompt가 적용된 PPG 신호를 의미한다.

그러나 $L_{Data}$는 test set distribution의 imbalance 영향을 받는다. 예를 들어 Normal과 Prehyper sample이 많고 Hypo sample이 적다면, 전체 평균 MAE가 좋아도 Hypo group 성능이 나쁠 수 있다. 이를 보완하기 위해 논문은 group average MAE를 정의한다.

$$
L_{Group} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{G_k} \sum_{i=1}^{G_k} |f(x_i^k) - y_i^k|
$$

여기서 $G_k$는 group $k$에 속한 test sample 수이고, $K$는 group 수이다. 이 지표는 각 BP group에 동일한 weight를 부여하므로, minority group 성능을 더 잘 반영한다. 논문은 $L_{Group}$을 robustness 평가 지표로 사용한다.

또한 논문은 실제 혈압값과 예측 혈압값 사이의 관계를 보기 위해 coefficient of determination인 $R^2$도 보고한다. 다만 표의 $R^2$ 값은 여러 경우에서 음수로 나타나며, 이는 few-shot transfer setting이 매우 어렵고 baseline 예측이 단순 평균 예측보다 나쁠 수 있음을 보여준다.

### 4.4 주요 정량 결과

![Fig. 2. Performance by BP group.](https://ars.els-cdn.com/content/image/1-s2.0-S0010482524013015-gr2.jpg)

논문은 BCG, Sensors, UCI 세 데이터셋 사이의 가능한 모든 pre-trained dataset과 target dataset 조합을 평가한다. 즉 BCG에서 사전학습 후 Sensors로 전이, BCG에서 사전학습 후 UCI로 전이, Sensors에서 사전학습 후 BCG로 전이, Sensors에서 사전학습 후 UCI로 전이, UCI에서 사전학습 후 BCG로 전이, UCI에서 사전학습 후 Sensors로 전이하는 여섯 가지 조합을 실험한다. 각 조합에서 5-shot과 10-shot setting을 모두 평가한다.

전체적으로 GloGen은 대부분의 경우에서 가장 낮은 total average MAE와 total group average MAE를 기록한다. 이는 GloGen이 단순히 전체 평균 성능만 개선한 것이 아니라, 혈압 그룹별 robustness도 함께 개선했음을 의미한다.

예를 들어 **BCG에서 사전학습하고 Sensors로 전이한 경우**를 보면, 5-shot setting에서 Scratch의 total average MAE는 46.35이고 LP는 38.94, FT는 33.32, GloGen은 32.47이다. Group average MAE에서도 Scratch는 37.25, LP는 34.79, FT는 31.56, GloGen은 30.20으로 가장 낮다. 10-shot setting에서도 GloGen은 total average MAE 31.71, group average MAE 29.76으로 가장 우수하다.

**BCG에서 사전학습하고 UCI로 전이한 경우**에도 GloGen은 5-shot에서 total average MAE 27.75, group average MAE 30.23을 기록하여 전체적으로 가장 좋은 결과를 보인다. 10-shot에서는 LP가 Avg SBP에서 약간 더 낮은 값을 보이지만, GloGen은 total average MAE 25.74와 $R^2=-0.30$으로 전반적인 성능과 상관 측면에서 강한 결과를 보인다.

**Sensors에서 사전학습하고 BCG로 전이한 경우**는 GloGen의 장점이 특히 두드러진다. 5-shot setting에서 Scratch는 total average MAE 25.35, LP는 22.06, FT는 30.28인 반면 GloGen은 19.97로 가장 낮다. Group average MAE도 GloGen이 27.92로 가장 낮으며, FT는 42.02로 크게 악화된다. 이는 full fine-tuning이 few-shot 환경에서 오히려 불안정할 수 있음을 보여준다.

**Sensors에서 사전학습하고 UCI로 전이한 경우**에도 GloGen은 5-shot에서 total average MAE 26.58, group average MAE 29.40을 기록하며 baseline보다 우수하다. 10-shot에서는 FT가 group average MAE 28.85로 GloGen의 29.27보다 약간 낮지만, GloGen은 total average MAE 26.03과 $R^2=-0.40$으로 전체 성능과 균형 측면에서 경쟁력 있는 결과를 보인다.

**UCI에서 사전학습하고 BCG로 전이한 경우** 5-shot setting에서 GloGen은 total average MAE 23.56, group average MAE 29.44로 가장 우수하다. 10-shot setting에서도 GloGen은 total average MAE 20.17로 Scratch의 20.71보다 약간 좋고, group average MAE도 30.40으로 경쟁력 있는 성능을 보인다.

**UCI에서 사전학습하고 Sensors로 전이한 경우** 5-shot setting에서 GloGen은 total average MAE 30.44, group average MAE 29.47을 기록하여 모든 baseline보다 우수하다. 10-shot에서도 GloGen은 total average MAE 29.22, group average MAE 28.50으로 가장 낮다.

이러한 결과는 GloGen이 dataset pair가 바뀌어도 비교적 일관되게 좋은 성능을 보인다는 점을 뒷받침한다. 특히 target task 데이터가 group별 5개 또는 10개에 불과한 상황에서도 robust adaptation이 가능하다는 점이 논문의 핵심 실험적 주장이다.

### 4.5 BP group별 성능 분석

논문은 Hypo와 Hyper2를 high-risk group(HG), Normal과 Prehyper를 low-risk group(LG)으로 묶어 분석한다. Figure 2에 따르면 대부분의 방법에서 HG의 MAE가 LG보다 높다. 이는 저혈압과 고혈압 구간의 예측이 정상 또는 전고혈압 구간보다 어렵다는 것을 의미한다.

GloGen은 일반적으로 LG 성능을 유지하면서 HG 성능을 개선하는 경향을 보인다. 논문은 FT가 일부 경우 HG에서 좋은 성능을 보일 수 있지만, 이때 LG 성능이 악화되는 trade-off가 발생한다고 설명한다. 예를 들어 FT가 Hypo와 Normal에서는 낮은 MAE를 보이더라도 Prehyper와 Hyper2에서 성능이 나빠지는 경우가 있다. 반면 GloGen은 GP, IP, VP를 통해 전체 test distribution의 일반화 성능과 BP group 간 robustness의 균형을 더 잘 관리한다.

이 결과는 GloGen의 실질적 의미를 잘 보여준다. 의료 AI 모델은 평균 오차만 낮으면 충분하지 않다. 실제로 위험도가 높은 환자군에서 성능이 낮으면 임상적 가치가 떨어질 수 있다. GloGen은 group average MAE를 개선함으로써 고위험군 성능을 평가하고 개선하려는 방향성을 명확히 제시한다.

### 4.6 GloGen Prompt의 역할 분석

논문은 Figure 4를 통해 GloGen prompt가 실제로 PPG 신호를 어떻게 바꾸는지 분석한다. 같은 SBP와 DBP 값을 갖더라도 PPG waveform은 측정 환경, 개인 특성, 센서 차이 등으로 인해 서로 다를 수 있다. 논문은 training set과 test set에서 SBP/DBP가 유사한 두 PPG 신호를 비교하고, GloGen prompt를 추가한 후 두 신호 간 cosine similarity가 증가하는 사례를 제시한다.

이 결과는 GloGen prompt가 unseen PPG 신호를 training set의 유사한 혈압 신호와 더 비슷한 형태로 변환하는 역할을 할 수 있음을 시사한다. 특히 GP는 같은 BP group 내부뿐 아니라 서로 다른 BP group 사이에서도 shared feature를 학습하여 PPG 신호의 공통 구조를 보정하는 데 기여한다고 해석된다.

또한 UMAP visualization에서는 IP들이 혈압 그룹별로 구분되는 cluster를 형성한다. 논문은 IP가 HG와 LG를 구분하고, HG 내부에서도 Hypo와 Hyper2가 구별되는 패턴을 보인다고 설명한다. 이는 Prompt Generator가 단순히 무작위적인 보정 신호를 생성하는 것이 아니라, BP group과 관련된 정보를 반영하는 prompt를 생성하고 있음을 뒷받침한다.

다만 이 해석은 시각화와 similarity 분석에 기반한 것이다. 논문은 IP의 특정 waveform 패턴이 어떤 생리학적 의미를 갖는지까지는 설명하지 않는다. 따라서 prompt의 생리학적 해석 가능성은 추가 연구가 필요한 부분이다.

### 4.7 Ablation Study

논문은 GloGen의 구성 요소가 각각 성능에 어떤 영향을 미치는지 ablation study를 통해 분석한다. 주요 분석 대상은 GP normalization, GP, IP, VP이다. 실험은 UCI에서 사전학습하고 Sensors를 target dataset으로 사용하는 5-shot setting에서 수행된다.

전체 GloGen은 total average MAE 30.44, group average MAE 29.47을 기록한다. GP normalization을 제거하면 total average MAE가 33.40, group average MAE가 31.26으로 악화된다. 이는 prompt normalization이 학습 안정성과 성능에 중요하다는 것을 보여준다.

GP를 제거한 경우 total average MAE는 34.14, group average MAE는 31.97로 악화된다. 이는 target dataset 전체에 공유되는 보정 정보가 필요하다는 점을 의미한다. IP를 제거한 경우 total average MAE는 32.72, group average MAE는 31.54가 된다. 이는 instance-specific prompt가 개별 신호와 BP group 특성을 반영하는 데 중요하다는 점을 보여준다.

VP를 제거한 경우 total average MAE는 33.84이고 group average MAE는 30.90이다. 흥미롭게도 VP를 제거해도 일부 DBP group metric은 나쁘지 않지만, 전체적으로는 full GloGen보다 낮은 성능을 보인다. 이는 VP가 IP diversity와 group robustness를 향상시키는 데 기여하지만, metric별 효과는 세부적으로 다를 수 있음을 보여준다.

Figure 3에서는 $\lambda$와 $\gamma$의 크기, 즉 GP와 IP의 영향력에 따른 성능 변화를 분석한다. GP 또는 IP가 없는 경우보다 두 prompt를 적절히 함께 사용할 때 average MAE와 group average MAE가 모두 좋아진다. 이는 GloGen의 dual-prompt design이 단순한 구성 요소 추가가 아니라 실제로 상호보완적인 역할을 한다는 것을 뒷받침한다.

![Fig. 3. Ablation Study on $\gamma$ and $\lambda$ with UCI dataset pre-training and Sensors Dataset as the target.](https://ars.els-cdn.com/content/image/1-s2.0-S0010482524013015-gr3.jpg)

### 4.8 Large-scale Pretraining 실험

논문은 MIMIC dataset으로 사전학습한 모델을 VitalDB dataset으로 전이하는 추가 실험을 수행한다. 이는 GloGen이 작은 benchmark dataset뿐 아니라 대규모 의료 데이터로 학습된 pretrained model에도 적용 가능한지를 검증하기 위한 실험이다.

MIMIC과 VitalDB는 서로 다른 환자 정보를 갖는 non-overlapping dataset으로 설명된다. 또한 VitalDB의 training/validation set과 test set도 subject overlap이 발생하지 않도록 설정했다고 논문은 명시한다. 이는 의료 데이터 실험에서 patient leakage를 방지하기 위한 중요한 설정이다.

5-shot setting에서 GloGen은 total average MAE 23.41을 기록하여 Scratch 25.63, LP 25.69, FT 25.00보다 우수하다. Group average MAE도 GloGen이 30.69로 가장 낮다. 10-shot setting에서도 GloGen은 total average MAE 23.78, group average MAE 29.48을 기록하여 대부분의 baseline보다 우수하다. 단, 10-shot setting의 average DBP MAE에서는 Scratch가 9.10으로 GloGen의 9.29보다 약간 낮다. 논문도 이 예외를 언급하면서, 전반적으로는 GloGen이 large-scale pretrained model에서도 일반화 성능과 group robustness를 개선한다고 설명한다.

이 실험은 GloGen의 practical relevance를 강화한다. 실제 의료 AI에서는 대규모 공용 데이터셋이나 병원 데이터로 사전학습된 모델을 새로운 병원 또는 새로운 장비에 적응시키는 상황이 많다. GloGen은 target dataset이 매우 작을 때도 이러한 전이가 가능함을 보여준다.

### 4.9 Classification 실험

논문은 GloGen이 SBP/DBP regression뿐 아니라 BP group classification에도 적용 가능함을 보인다. 이 실험에서는 Sensors dataset으로 사전학습한 모델을 BCG dataset의 few-shot classification task에 적용한다. 평가 지표는 accuracy, group accuracy, precision, recall, F1-score이다.

5-shot setting에서 GloGen은 accuracy 43.5, group accuracy 43.0, precision 57.7, recall 43.0을 기록하여 대부분의 지표에서 가장 좋은 성능을 보인다. 다만 F1-score는 Scratch가 29.5이고 GloGen이 28.2로 Scratch보다 낮다. 10-shot setting에서는 Scratch가 accuracy 42.9로 GloGen의 42.3보다 약간 높지만, GloGen은 group accuracy 40.8과 recall 40.8로 baseline보다 좋은 group-balanced 성능을 보인다.

이 결과는 GloGen이 단순히 연속값 회귀에만 맞춘 방법이 아니라, PPG 신호 transformation을 통해 BP-related representation을 개선할 수 있음을 시사한다. 다만 classification 실험의 F1-score에서는 모든 setting에서 완전히 우월하지는 않으므로, classification task에서의 효과는 regression 결과보다 다소 제한적으로 해석할 필요가 있다.

### 4.10 Failure Case 분석

논문은 실패 사례로 PPG signal의 peak 수가 모델 성능에 큰 영향을 미친다는 점을 제시한다. BCG dataset 일부 샘플을 peak 수에 따라 나누어 분석했을 때, peak 수가 8개인 신호에서는 Scratch, LP, FT, GloGen 모두 매우 높은 MAE를 보인다.

예를 들어 UCI에서 사전학습하고 BCG 5-shot으로 fine-tuning한 경우, peak 수가 4개인 샘플에서 GloGen의 MAE는 $35.4 \pm 4.7$이고, peak 수가 5개인 샘플에서는 $7.4 \pm 6.3$으로 매우 낮다. 그러나 peak 수가 8개인 샘플에서는 GloGen의 MAE가 $77.8 \pm 15.3$으로 크게 증가한다. 다른 baseline들도 peak 수 8개에서 매우 높은 오차를 보인다.

이는 GloGen도 비정상적이거나 복잡한 waveform pattern에 대해서는 한계를 가진다는 점을 보여준다. 논문은 peak 수가 많은 신호가 왜 어려운지에 대해 자세한 생리학적 분석이나 noise 원인 분석을 제공하지는 않는다. 따라서 이 부분은 향후 연구에서 motion artifact, signal quality assessment, abnormal waveform detection 등과 함께 다루어야 할 문제로 보인다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정 문제를 단순한 평균 성능 경쟁이 아니라 few-shot transfer learning과 group robustness 관점에서 재정의했다는 점이다. 실제 의료 환경에서는 정상 혈압군에 대한 평균 오차보다 저혈압 및 고혈압 고위험군에서의 안정적 성능이 중요하다. 논문은 Hypo, Normal, Prehyper, Hyper2로 그룹을 나누고, group average MAE를 통해 모델의 robustness를 정량적으로 평가한다. 이는 기존 연구에서 자주 간과되던 부분이다.

두 번째 강점은 dual-prompt design이다. GP는 target dataset 전체의 공통 특성을 학습하고, IP는 개별 신호 또는 BP group 특성을 반영한다. 이 구조는 직관적으로 타당하며, ablation study에서도 GP와 IP가 모두 필요하다는 점이 뒷받침된다. 특히 PPG 신호처럼 개인차와 측정 환경 차이가 큰 생체 신호에서는 shared adaptation과 personalized adaptation을 함께 고려하는 설계가 적절하다.

세 번째 강점은 Variance Penalty이다. VP는 prompt diversity를 명시적으로 유도하여 BP group별 IP가 서로 구분되도록 만든다. 단순히 MSE만 줄이는 prompt generator는 majority group에 맞는 비슷한 prompt를 생성할 가능성이 있다. VP는 이러한 편향을 완화하고 underrepresented BP group에 대한 robustness를 높이려는 장치이다. 또한 margin과 ReLU를 사용해 VP maximization의 불안정성을 줄인 점도 실용적인 설계로 볼 수 있다.

네 번째 강점은 few-shot setting이 매우 엄격하다는 점이다. 논문은 group별 5개 또는 10개 sample만 사용하는 상황을 설정한다. 의료 데이터에서는 target dataset을 대규모로 수집하기 어렵기 때문에, 이러한 실험 설정은 현실적인 의미가 있다. 특히 MIMIC에서 사전학습하고 VitalDB로 전이하는 실험은 large-scale pretraining과 small target adaptation이라는 실제적 시나리오를 반영한다.

다섯 번째 강점은 모델 구조에 대한 의존성이 비교적 낮다는 점이다. 논문에서는 ResNet1D를 사용했지만, 방법론 자체는 입력 prompt를 추가하고 embedding 기반 Prompt Generator를 사용하는 구조이므로, embedding을 얻을 수 있는 다른 PPG model에도 확장 가능하다고 주장한다. 이는 향후 다양한 backbone에 적용할 가능성을 제공한다.

### 5.2 한계

첫 번째 한계는 prompt의 생리학적 해석 가능성이 부족하다는 점이다. 논문은 prompt가 PPG 신호 간 cosine similarity를 증가시키고, IP가 BP group별로 cluster를 형성한다는 분석을 제공한다. 그러나 특정 prompt pattern이 실제 혈류, 혈관 탄성, pulse transit 관련 특성과 어떤 관계를 갖는지는 설명하지 않는다. 의료 AI에서 interpretability가 중요한 만큼, GP와 IP가 어떤 physiological meaning을 갖는지 분석하는 후속 연구가 필요하다.

두 번째 한계는 backbone 다양성 검증이 제한적이라는 점이다. 논문은 모든 주요 실험에서 ResNet1D를 사용한다. 이는 공정한 비교에는 유리하지만, GloGen이 Transformer 기반 time-series model, temporal convolution network, recurrent model, spectro-temporal model 등 다양한 architecture에서도 동일하게 효과적인지는 본문만으로는 확인할 수 없다. 논문은 model-agnostic 가능성을 주장하지만, 실험적 검증은 제한적이다.

세 번째 한계는 hyperparameter 의존성이 크다는 점이다. GloGen은 learning rate, weight decay, $\lambda$, $\gamma$, margin $m$, $\alpha$, normalization 대상, clipping 여부 등 여러 hyperparameter를 grid search로 탐색한다. Few-shot target data가 매우 적은 상황에서는 validation set도 작기 때문에, hyperparameter 선택이 불안정할 수 있다. 논문은 validation group average MAE를 사용한다고 설명하지만, 작은 validation set에서 선택된 hyperparameter의 안정성은 추가 검증이 필요하다.

네 번째 한계는 signal quality 문제에 대한 대응이 부족하다는 점이다. Failure case 분석에서 peak 수가 많은 PPG 신호에 대해 모든 모델이 큰 오차를 보였다. 실제 웨어러블 환경에서는 motion artifact, poor contact, sensor noise, irregular pulse 등으로 인해 품질이 낮은 PPG가 자주 발생할 수 있다. GloGen은 robust transfer learning에는 효과적이지만, signal quality degradation 자체를 해결하는 방법은 아니다.

다섯 번째 한계는 real-time deployment 관점의 분석이 부족하다는 점이다. 논문은 GloGen이 real-time non-invasive BP estimation에 잠재력이 있다고 언급하지만, Prompt Generator의 latency, 메모리 사용량, 웨어러블 기기에서의 연산 비용은 구체적으로 분석하지 않는다. GP만 사용하는 경우는 가볍지만, IP를 생성하려면 encoder와 Prompt Generator를 거쳐야 하므로 실제 edge device 적용 시 비용 분석이 필요하다.

여섯 번째 한계는 demographic factor나 sensor/device variability에 대한 별도 분석이 없다는 점이다. 기존 연구에서 혈압 추정 성능은 질병, 인종, 성별 등 subgroup에 따라 달라질 수 있다고 언급되지만, 이 논문은 주로 BP group 기준으로 robustness를 평가한다. 데이터셋 내 demographic subgroup에 대한 fairness나 cross-device generalization은 본문에서 다루지 않는다.

### 5.3 비판적 해석

이 논문은 PPG 기반 혈압 추정에서 중요한 문제를 잘 포착한 연구이다. 특히 평균 MAE 중심 평가의 한계를 지적하고, group average MAE를 통해 high-risk BP group에서의 성능을 분석했다는 점은 의료 AI 관점에서 매우 적절하다. 또한 prompt learning을 생체 시계열 신호에 적용하여, 모델 전체를 fine-tuning하지 않고 입력 신호를 학습적으로 재구성하는 접근은 흥미롭고 실용적이다.

다만 GloGen의 개선이 prompt 자체의 생리학적 의미에서 비롯된 것인지, 아니면 few-shot regularization 및 input perturbation의 효과인지에 대해서는 더 깊은 분석이 필요하다. UMAP과 cosine similarity 분석은 설득력 있는 정성적 근거를 제공하지만, prompt가 어떤 waveform component를 보정하는지, 예를 들어 systolic peak, dicrotic notch, pulse width, amplitude ratio 등과 어떤 관련이 있는지는 명확하지 않다.

또한 FT가 일부 경우 성능이 나쁘게 나타나는 것은 few-shot 환경의 과적합 때문일 가능성이 크지만, FT baseline의 세부 regularization, early stopping, layer-wise learning rate, partial fine-tuning 등 다양한 fine-tuning 전략과 비교했는지는 본문에서 충분히 제시되지 않는다. 따라서 GloGen이 모든 가능한 transfer learning 전략보다 우월하다고 일반화하기보다는, 논문에서 비교한 Scratch, LP, FT baseline 대비 few-shot BP estimation transfer에서 우수하다고 해석하는 것이 정확하다.

## 6. 결론

이 논문은 PPG 신호 기반 혈압 추정을 위한 few-shot transfer learning 프레임워크인 GloGen을 제안한다. GloGen은 Global Prompt와 Instance-wise Prompt를 함께 사용하여 target dataset의 공통 특성과 개별 신호 특성을 동시에 반영한다. 또한 Variance Penalty를 통해 혈압 그룹별 IP가 서로 다양하게 생성되도록 유도하여, 저혈압 및 고혈압과 같은 high-risk group에서의 robustness를 개선한다.

실험 결과는 GloGen이 BCG, Sensors, UCI 사이의 다양한 cross-dataset transfer setting에서 대부분 Scratch, Linear Probing, Fine-Tuning보다 낮은 average MAE와 group average MAE를 달성함을 보여준다. MIMIC에서 사전학습하고 VitalDB로 전이하는 large-scale pretraining 실험에서도 GloGen은 전반적으로 우수한 성능을 보인다. 또한 regression뿐 아니라 BP group classification에서도 group-balanced 성능 개선 가능성을 보인다.

이 연구의 주요 기여는 다음과 같이 요약할 수 있다. 첫째, PPG 기반 혈압 추정에 prompt learning을 적용한 few-shot transfer learning 프레임워크를 제안했다. 둘째, GP와 IP를 결합한 dual-prompt 구조를 통해 shared adaptation과 personalized adaptation을 함께 수행했다. 셋째, VP를 도입하여 prompt diversity와 BP group robustness를 명시적으로 강화했다. 넷째, 평균 성능뿐 아니라 혈압 그룹별 성능을 평가하여 의료적으로 중요한 high-risk group robustness 문제를 다루었다.

향후 연구에서는 prompt의 생리학적 해석, 다양한 backbone model에서의 검증, signal quality-aware GloGen, real-time wearable deployment, demographic subgroup robustness 분석이 필요하다. 그럼에도 불구하고 이 논문은 PPG 기반 비침습 혈압 추정에서 few-shot transfer learning과 robust prompt learning을 결합한 의미 있는 연구이며, 의료 시계열 신호에 prompt-based adaptation을 적용하는 중요한 사례로 평가할 수 있다.
