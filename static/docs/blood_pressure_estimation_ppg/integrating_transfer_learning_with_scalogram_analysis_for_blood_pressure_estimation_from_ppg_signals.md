# Integrating transfer learning with scalogram analysis for blood pressure estimation from PPG signals

* **저자**: Shyamala Subramanian, Sashikala Mishra, Shruti Patil, Maheshkumar H. Kolekar, Fernando Ortiz-Rodriguez
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호만을 이용하여 비침습적으로 혈압을 추정하기 위한 deep learning 및 transfer learning 기반 방법을 제안한다. 연구의 핵심 목표는 PPG 시계열을 직접 모델에 넣는 대신, CWT(Continuous Wavelet Transform)를 사용하여 PPG 신호를 scalogram이라는 2차원 time-frequency representation으로 변환하고, ImageNet 등에서 사전학습된 CNN 모델을 feature extractor로 활용한 뒤, 추출된 deep feature를 Random Forest 회귀기에 입력하여 SBP(Systolic Blood Pressure)와 DBP(Diastolic Blood Pressure)를 예측하는 것이다.

논문이 다루는 연구 문제는 연속적이고 비침습적인 혈압 추정을 어떻게 정확하고 효율적으로 수행할 것인가이다. 전통적인 cuff 기반 혈압 측정은 임상적으로 널리 사용되지만 연속 모니터링에는 적합하지 않고, 반복 측정 시 불편함이 크다. 침습적 ABP(Arterial Blood Pressure) 측정은 정확하지만 입원 환자나 중환자실 환경에 제한된다. 반면 PPG는 광학 센서를 통해 말초 혈류량 변화를 측정할 수 있어 스마트워치와 같은 웨어러블 기기에 쉽게 통합될 수 있다. 따라서 PPG로부터 혈압을 정확하게 추정할 수 있다면, 고혈압 및 심혈관 질환의 조기 발견과 장기 모니터링에 실질적인 도움이 될 수 있다.

이 논문의 중요성은 두 가지 측면에 있다. 첫째, PPG 신호를 time-frequency domain에서 해석하기 위해 Morlet wavelet 기반 scalogram을 사용함으로써, 단순한 time-domain feature나 hand-crafted feature에 의존하지 않는다. 둘째, VGG16, ResNet50, InceptionV3, NASNetLarge, InceptionResNetV2, ConvNeXtTiny 같은 pretrained CNN을 feature extractor로 사용하여 처음부터 대규모 CNN을 학습시키지 않고도 deep feature를 얻는다. 이는 데이터가 제한적이고 계산 자원이 제한될 수 있는 biomedical signal processing 상황에서 실용적인 접근이다.

논문은 MIMIC-II 기반 UCI cuffless blood pressure estimation dataset을 사용한다. PPG와 ABP 신호를 8초 길이의 non-overlapping window로 분할하고, 각 PPG segment에서 scalogram을 생성한다. ABP segment에서는 SBP와 DBP reference 값을 추출한다. 실험 결과 ConvNeXtTiny와 VGG16 기반 feature가 가장 좋은 성능을 보였으며, 특히 ConvNeXtTiny는 SBP MAE 2.95 mmHg, SD 4.11 mmHg, DBP MAE 1.66 mmHg, SD 2.60 mmHg를 달성했다. 저자들은 이 성능이 AAMI 기준과 BHS Grade A 기준을 만족한다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG 신호를 1차원 시간 신호로만 보지 않고, 시간과 주파수 정보를 동시에 담은 이미지 형태의 scalogram으로 변환한 뒤, 이미지 인식 분야에서 성능이 검증된 pretrained CNN을 feature extractor로 활용하는 것이다. PPG는 심장 박동에 따른 혈류량 변화가 반영된 비정상(non-stationary) 생리 신호이다. 혈압과 관련된 정보는 특정 시점의 파형 모양뿐 아니라 시간에 따라 변하는 주파수 성분에도 포함될 수 있다. 따라서 time-frequency representation은 혈압 추정에 유용할 수 있다.

기존 전통적 방법들은 PTT(Pulse Transit Time), PWV(Pulse Wave Velocity), PWA(Pulse Wave Analysis), morphology feature 등을 수작업으로 정의하고 regression model에 입력하는 방식이 많았다. 이러한 방법은 생리적으로 해석 가능하다는 장점이 있지만, 여러 센서가 필요하거나 feature engineering이 복잡하고, 개인차나 신호 품질 변화에 취약할 수 있다. 반면 end-to-end deep learning 방식은 raw PPG에서 feature를 자동으로 학습하지만, 충분한 데이터와 큰 계산 자원이 필요하고, 학습 안정성이나 overfitting 문제가 발생할 수 있다.

이 논문은 두 접근의 중간 지점에 있다. PPG 신호를 CWT로 scalogram화하여 time-frequency structure를 보존하고, pretrained CNN을 사용하여 고차원 feature를 추출한다. 이후 Random Forest가 이 feature와 혈압값 사이의 비선형 관계를 학습한다. 즉, CNN은 직접 혈압 회귀를 수행하는 것이 아니라 feature extractor 역할을 하고, 최종 회귀는 Random Forest가 담당한다.

논문에서 강조하는 차별점은 다음과 같다. 첫째, Morlet wavelet 기반 CWT를 사용하여 PPG 신호의 시간 및 주파수 정보를 동시에 보존한다. 둘째, pretrained CNN을 활용하여 feature extraction 비용을 낮추고, 처음부터 대규모 모델을 학습할 필요를 줄인다. 셋째, Random Forest를 최종 회귀기로 사용하여 복잡한 feature pattern을 robust하게 처리하고 overfitting을 완화한다. 넷째, 단순한 regression metric뿐 아니라 AAMI와 BHS라는 임상 기준으로 성능을 평가한다.

핵심 직관은 “PPG의 혈압 관련 정보는 time-frequency image 안에 구조적으로 나타나며, pretrained CNN은 이러한 구조에서 유용한 deep feature를 추출할 수 있다”는 것이다. 특히 ConvNeXtTiny와 VGG16이 좋은 성능을 보인 것은, scalogram의 시각적 패턴이 복잡한 CNN 구조보다 비교적 안정적인 convolutional feature extractor와 잘 맞았기 때문일 가능성이 있다. 다만 논문은 왜 특정 pretrained model이 더 잘 작동하는지에 대한 깊은 ablation이나 feature-level 해석은 충분히 제공하지 않는다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

논문의 전체 시스템은 PPG 신호 입력에서 SBP와 DBP 예측값을 출력하는 회귀 파이프라인이다. 구성은 다음 단계로 이루어진다.

첫째, MIMIC-II 기반 UCI dataset에서 PPG와 ABP 신호를 불러온다. 둘째, PPG와 ABP를 1000 sample 길이의 non-overlapping segment로 나눈다. sampling rate가 125 Hz이므로 하나의 segment 길이는 8초이다. 셋째, PPG segment에 band-pass filtering을 적용한다. 논문은 PPG에서 중요한 정보가 0.1 Hz부터 8 Hz 사이에 존재한다고 보고, 이 구간을 사용한다. 넷째, 각 PPG segment에 CWT를 적용하여 128개의 scale을 갖는 scalogram을 생성한다. 다섯째, scalogram을 pretrained CNN의 입력 크기에 맞게 resize하고 RGB 형태로 변환한다. 여섯째, pretrained CNN에서 top classifier를 제거하고 deep feature를 추출한다. 일곱째, flatten된 feature vector를 Random Forest regressor에 입력하여 SBP와 DBP를 예측한다. 마지막으로 MAE, SD, Pearson correlation coefficient, AAMI, BHS 기준으로 평가한다.

이 파이프라인은 다음과 같이 요약할 수 있다.

$$
PPG \ segment
\rightarrow CWT \ scalogram
\rightarrow pretrained \ CNN \ feature
\rightarrow Random \ Forest
\rightarrow (\hat{SBP}, \hat{DBP})
$$

### 3.2 데이터셋과 segment 구성

논문은 UCI Machine Learning Repository에서 제공되는 cuffless blood pressure estimation dataset을 사용한다. 이 데이터셋은 MIMIC-II waveform database의 subset이며, 총 12,000개 record를 포함한다. 각 record에는 PPG, ABP, ECG 신호가 동기화되어 포함되어 있고 sampling frequency는 125 Hz이다. 이 연구에서는 PPG와 ABP만 사용하며, ECG는 사용하지 않는다.

ABP는 invasive radial arterial catheterization으로 측정된 arterial blood pressure waveform이므로 reference standard로 사용된다. 즉 모델은 PPG만 입력으로 사용하지만, 정답 SBP와 DBP는 ABP waveform에서 추출된다.

논문은 각 record를 1000 sample 단위로 나눈다. segment duration은 다음과 같이 계산된다.

$$
\text{Duration} = \frac{\text{No.\ of Samples}}{\text{Sampling Frequency}} = \frac{1000}{125} = 8\ \text{seconds}
$$

실험에는 3000개의 PPG segment가 사용되며, 각 segment는 1000 sample 길이를 가진다. 각 PPG segment로부터 하나의 scalogram이 생성되므로 총 3000개의 scalogram이 생성된다. 각 scalogram의 초기 크기는 128 by 1000이다. 여기서 128은 CWT scale 수이고, 1000은 시간축 sample 수이다. 각 ABP segment에서는 대응되는 SBP와 DBP reference value가 추출되며, 최종 target은 3000 by 2 형태의 SBP/DBP 값이다.

### 3.3 PPG filtering과 ABP target 추출

PPG 신호는 0.1 Hz부터 8 Hz까지의 band-pass filter를 거친다. 논문은 Fourier Transform 분석을 통해 이 주파수 범위가 PPG의 주요 정보를 포함한다고 설명한다. PPG에는 baseline level을 반영하는 DC component와 심장 박동에 따른 혈액량 변화가 반영된 AC component가 있다. 혈압 추정에 필요한 정보는 주로 이러한 주파수 범위 안에서 나타난다고 본다.

ABP segment에서 SBP와 DBP를 추출하는 방식은 비교적 단순하다. 각 ABP segment를 1초 단위 또는 일정 기간으로 나누고, 각 구간에서 maximum value를 SBP 후보, minimum value를 DBP 후보로 사용한다. 이후 segment 내 여러 period의 maximum 평균을 SBP, minimum 평균을 DBP로 정의한다. 개념적으로는 다음과 같이 표현할 수 있다.

$$
SBP =
\frac{1}{K}
\sum_{k=1}^{K}
\max(ABP_k)
$$

$$
DBP =
\frac{1}{K}
\sum_{k=1}^{K}
\min(ABP_k)
$$

여기서 $K$는 segment 내에서 혈압값을 추출한 period 수이다. 이 방식은 ABP waveform이 reference blood pressure를 직접 포함한다는 점을 활용한다. 다만 논문에는 peak detection 품질 관리, 이상 ABP segment 제거, outlier handling에 대한 자세한 설명은 제한적으로 제시되어 있다.

### 3.4 Continuous Wavelet Transform과 scalogram 생성

이 논문의 핵심 전처리는 CWT이다. CWT는 입력 신호와 wavelet basis function 사이의 유사도를 scale과 shift에 따라 계산한다. Fourier Transform이 전체 신호를 complex exponential로 분해하는 데 비해, CWT는 시간 위치와 scale 정보를 함께 제공하므로 비정상 생리 신호 분석에 적합하다.

논문에서 제시한 CWT 식은 다음과 같다.

$$
C(a,b;f(t),\psi(t)) =
\int_{-\infty}^{\infty}
f(t)\frac{1}{a}\psi^{*}
\left(
\frac{t-b}{a}
\right)dt
$$

여기서 $f(t)$는 입력 PPG 신호, $\psi(t)$는 wavelet basis, $\psi^{*}(t)$는 complex conjugate, $a$는 scale, $b$는 shift를 의미한다. Scale $a$는 주파수 해상도와 관련되고, shift $b$는 시간 위치와 관련된다. CWT 결과는 시간과 scale의 2차원 함수가 되며, 이를 이미지처럼 표현한 것이 scalogram이다.

논문은 Poisson wavelet, Mexican Hat wavelet, Morlet wavelet을 설명하지만, 실제 사용한 것은 Morlet wavelet이다. Morlet wavelet은 Gaussian window와 complex sinusoid를 결합한 형태이며, PPG처럼 주기적이고 oscillatory한 생리 신호 분석에 적합하다고 설명된다. 논문에서 제시된 Morlet wavelet은 다음과 같다.

$$
\psi_{Mor}(t) =
e^{j2\pi ft}e^{-\frac{t^2}{2\sigma^2}}
$$

Gaussian width $\sigma$는 다음과 같이 정의된다.

$$
\sigma =
\frac{n}{2\pi f}
$$

여기서 $n$은 wavelet cycle 수이며 time-frequency precision의 trade-off를 조절한다. 논문은 PPG의 주요 주파수 범위가 0.5 Hz부터 10 Hz 정도이고 EEG와 유사한 생리 신호라는 점을 고려하여 $n=3$을 선택했다고 설명한다.

CWT를 적용한 후 각 PPG segment는 128 by 1000 크기의 scalogram으로 변환된다. 이후 pretrained CNN 모델의 입력 요구사항에 맞게 resize된다. 예를 들어 VGG16, ResNet50, ConvNeXtTiny는 224 by 224 by 3 입력을 요구하고, NASNetLarge는 331 by 331 by 3, InceptionResNetV2는 299 by 299 by 3 입력을 요구한다. 원래 scalogram은 grayscale 2D array이므로 RGB 3-channel 형태로 변환된다.

### 3.5 Pretrained CNN을 이용한 feature extraction

논문은 여섯 개 pretrained CNN을 비교한다. 이들은 모두 혈압 회귀 모델로 end-to-end fine-tuning되는 것이 아니라, scalogram에서 deep feature를 추출하는 feature extractor로 사용된다. top layer는 제거하고, convolutional feature map을 flatten하여 Random Forest에 입력한다. 논문 설명에 따르면 대부분의 pretrained layer는 frozen 상태이며, top 또는 dense layer만 조정되는 방식으로 설명되어 있으나, 실제 feature extraction 과정에서는 include_top=false로 pretrained model의 convolutional feature를 얻는 방식이 중심이다.

VGG16은 3 by 3 convolution과 max pooling을 반복하는 단순하고 안정적인 구조이다. 입력은 224 by 224 by 3이고, feature output은 7 by 7 by 512이다. Flatten하면 25,088차원의 feature vector가 된다.

ResNet50은 residual connection을 사용하여 깊은 CNN을 안정적으로 학습할 수 있는 구조이다. 입력은 224 by 224 by 3이고, output feature는 7 by 7 by 2048이다. Flatten하면 100,352차원이 된다.

InceptionV3는 서로 다른 kernel size를 가진 병렬 convolution path를 통해 multi-scale feature를 추출한다. 입력은 224 by 224 by 3으로 설명되어 있으며, output은 5 by 5 by 2048이고 flatten feature는 51,200차원이다.

NASNetLarge는 neural architecture search 기반 모델이며, 입력 크기는 331 by 331 by 3이다. output은 11 by 11 by 4032로 매우 크며, flatten feature는 487,872차원이다. 이는 Random Forest 입력으로 상당히 고차원이다.

InceptionResNetV2는 Inception module과 residual connection을 결합한 구조이다. 입력은 299 by 299 by 3이고, output은 8 by 8 by 1536이다. Flatten feature는 98,304차원이다.

ConvNeXtTiny는 modern convolutional network 계열로, 비교적 작은 구조이면서도 높은 성능과 효율성을 목표로 한다. 입력은 224 by 224 by 3이고, output은 7 by 7 by 768이다. Flatten feature는 37,632차원이다. 실험 결과 ConvNeXtTiny가 가장 좋은 성능을 보인다.

### 3.6 Random Forest 회귀

Pretrained CNN에서 얻은 flatten feature vector는 Random Forest regressor에 입력된다. Random Forest는 여러 decision tree를 bootstrap sample로 학습시키고, 각 tree의 예측을 평균하여 최종 예측을 만든다. 논문에서는 estimator 수를 100으로 설정하고, random state는 42로 사용한다. Validation은 10-fold cross-validation을 사용한다.

학습 데이터는 다음과 같이 표현된다.

$$
D =
{(x_k, y_k)}_{k=1}^{m}
$$

여기서 $x_k$는 pretrained CNN에서 추출한 feature vector이고, $y_k$는 SBP 또는 DBP target value이다. 각 tree는 bootstrapped dataset $D_b$에서 학습된다.

Decision tree의 한 region $R_j$에서 예측값은 해당 region에 속한 target value의 평균이다.

$$
c_j =
\frac{1}{|R_j|}
\sum_{x_k \in R_j}
y_k
$$

새로운 입력 $x_k$가 특정 region $R_j$에 속하면, 단일 tree의 예측은 다음과 같이 표현된다.

$$
f_b(x_k) =
\sum_{j=1}^{F}
c_j I(x_k \in R_j)
$$

여기서 $I(x_k \in R_j)$는 indicator function이다.

$$
I(x_k \in R_j) =
\begin{cases}
1, & x_k \in R_j \
0, & otherwise
\end{cases}
$$

Random Forest의 최종 예측은 모든 tree 예측의 평균이다.

$$
f(x_k) =
\frac{1}{B}
\sum_{b=1}^{B}
f_b(x_k)
$$

여기서 $B$는 tree 수이며, 이 논문에서는 $B=100$이다. Random Forest는 여러 tree의 평균을 사용하므로 variance를 줄이고 overfitting을 완화하는 장점이 있다. 또한 feature importance를 계산할 수 있어 어떤 feature가 예측에 기여했는지 분석할 가능성이 있지만, 이 논문에서는 feature importance에 대한 구체적인 해석 결과는 제시되지 않는다.

### 3.7 평가 지표와 임상 기준

논문은 MAE와 SD를 주요 성능 지표로 사용한다. MAE는 예측값과 실제값 사이의 평균 절대 오차이다.

$$
MAE =
\frac{1}{n}
\sum_{k=1}^{n}
|y_k - \hat{y}_k|
$$

SD는 오차 분포의 표준편차를 나타낸다.

$$
SD =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(x_i-\mu)^2
}
$$

여기서 $x_i$는 error sample, $\mu$는 error의 평균이다. 논문은 SD를 AAMI 기준과 연결하여 해석한다.

AAMI 기준은 예측 혈압과 기준 혈압의 mean error가 5 mmHg 이내이고, standard deviation이 8 mmHg 이하일 것을 요구한다. 논문은 모든 모델 또는 주요 모델이 이 기준을 만족한다고 보고한다.

BHS 기준은 절대 오차가 5, 10, 15 mmHg 이하인 sample 비율에 따라 Grade A, B, C, D를 부여한다. Grade A는 각각 60%, 85%, 95% 이상을 요구한다.

$$
Grade \ A:
\quad
P(|error| \le 5) \ge 60%,
\quad
P(|error| \le 10) \ge 85%,
\quad
P(|error| \le 15) \ge 95%
$$

논문은 ConvNeXtTiny와 VGG16이 SBP와 DBP 모두에서 BHS Grade A를 달성했다고 보고한다. 다른 모델들은 DBP에서는 Grade A에 해당하지만 SBP에서는 Grade B 또는 Grade A에 근접한다고 설명된다.

또한 Pearson correlation coefficient $r$도 사용된다. 이는 예측값과 실제값 사이의 선형 관계를 측정한다.

$$
r =
\frac{
\sum (X_i-\bar{X})(Y_i-\bar{Y})
}{
\sqrt{
\sum (X_i-\bar{X})^2
\sum (Y_i-\bar{Y})^2
}
}
$$

여기서 $X_i$와 $Y_i$는 각각 실제값과 예측값 또는 두 비교 변수이며, $\bar{X}$와 $\bar{Y}$는 평균이다. 논문에서는 ConvNeXtTiny와 VGG16이 다른 모델보다 높은 correlation을 보인다고 보고한다.

## 4. 실험 및 결과

### 4.1 실험 환경

논문에서 사용한 실험 환경은 Windows 11, Python 3.6, Jupyter Notebook, TensorFlow 2.10.0이다. 하드웨어는 Intel Core i5-12500H CPU, NVIDIA GeForce RTX 3050 GPU, 8 GB RAM이다. Random Forest는 estimator 수 100, random state 42로 설정되었고, validation은 10-fold cross-validation을 사용했다. 논문 표에는 epoch 수 30도 제시되어 있으나, pretrained CNN을 feature extractor로 사용하고 Random Forest를 회귀기로 사용하는 구조에서 epoch가 정확히 어떤 단계의 학습에 적용되었는지는 명확하게 설명되어 있지 않다. 이 부분은 논문 설명만으로는 다소 불분명하다.

### 4.2 모델별 정량 결과

여섯 개 transfer learning model과 Random Forest 조합의 결과는 Table 4에 제시된다.

ConvNeXtTiny가 가장 좋은 성능을 보인다. SBP에 대해 ME 0.64 mmHg, MAE 2.95 mmHg, SD 4.11 mmHg를 달성했고, DBP에 대해 ME 0.47 mmHg, MAE 1.66 mmHg, SD 2.60 mmHg를 달성했다. 이는 AAMI 기준인 ME 5 mmHg 이하, SD 8 mmHg 이하를 만족한다. BHS 기준에서도 SBP의 경우 5 mmHg 이하 error 비율이 81.33%, 10 mmHg 이하가 97.33%, 15 mmHg 이하가 99.33%로 Grade A를 만족한다. DBP는 5 mmHg 이하 92.67%, 10 mmHg 이하 98.67%, 15 mmHg 이하 100%로 역시 Grade A이다.

VGG16은 두 번째로 좋은 성능을 보인다. SBP에 대해 ME 0.24, MAE 3.94, SD 5.19를 기록했고, DBP에 대해 ME 0.08, MAE 2.56, SD 3.80을 기록했다. BHS 기준에서도 SBP는 5 mmHg 이하 68%, 10 mmHg 이하 96.67%, 15 mmHg 이하 98.67%로 Grade A이고, DBP는 84%, 95%, 100%로 Grade A이다.

ResNet50은 SBP MAE 4.66, SD 5.75, DBP MAE 2.39, SD 3.63을 보인다. DBP에서는 좋은 성능을 보이지만, SBP의 5 mmHg 이하 비율은 58.67%로 Grade A 기준인 60%에 약간 못 미친다. 따라서 논문은 SBP에서는 Grade B 또는 Grade A에 근접한 수준으로 해석하고, DBP에서는 Grade A로 본다.

InceptionV3는 SBP MAE 5.21, SD 6.70, DBP MAE 3.20, SD 4.72를 보인다. NASNetLarge는 SBP MAE 5.19, SD 6.34, DBP MAE 3.06, SD 4.27이다. InceptionResNetV2는 SBP MAE 5.11, SD 6.40, DBP MAE 2.74, SD 4.10이다. 이들 모델은 DBP에서는 BHS Grade A를 만족하지만, SBP에서는 5 mmHg 이하 비율이 60%에 미치지 못해 Grade B 수준으로 해석된다. 그래도 SD는 모두 8 mmHg 이하이므로 AAMI 기준 측면에서는 비교적 양호하다.

### 4.3 Pearson correlation 결과

논문은 예측값과 reference 값의 Pearson correlation coefficient도 보고한다. VGG16은 SBP에서 0.879, DBP에서 0.729를 기록한다. ConvNeXtTiny는 SBP에서 0.826, DBP에서 0.831을 기록한다. 두 모델 모두 비교적 높은 선형 상관성을 보인다.

반면 ResNet50은 SBP 0.589, DBP 0.342이고, InceptionV3는 SBP 0.399, DBP 0.195, NASNetLarge는 SBP 0.427, DBP 0.199, InceptionResNetV2는 SBP 0.448, DBP 0.214이다. 이 값들은 MAE와 SD 측면에서는 임상 기준을 어느 정도 만족하더라도, 예측값과 실제값의 선형적 추세 추적 능력은 상대적으로 약할 수 있음을 시사한다.

특히 ConvNeXtTiny는 SBP와 DBP 모두에서 균형 잡힌 correlation을 보인다. VGG16은 SBP correlation이 가장 높지만 DBP correlation은 ConvNeXtTiny보다 낮다. 따라서 논문이 ConvNeXtTiny를 가장 신뢰할 수 있는 모델로 강조하는 것은 MAE, SD, BHS 비율, correlation의 균형을 고려한 결과로 볼 수 있다.

### 4.4 BHS 및 AAMI 기준 해석

AAMI 기준 관점에서 ConvNeXtTiny와 VGG16은 SBP와 DBP 모두 평균 오차와 표준편차 요구사항을 만족한다. ConvNeXtTiny는 SBP SD 4.11, DBP SD 2.60으로 매우 낮은 dispersion을 보인다. VGG16도 SBP SD 5.19, DBP SD 3.80으로 허용 범위 안에 있다.

BHS 기준 관점에서 ConvNeXtTiny와 VGG16은 SBP와 DBP 모두 Grade A를 만족한다. ConvNeXtTiny의 SBP 5 mmHg 이하 비율 81.33%는 기준인 60%를 훨씬 상회한다. DBP의 경우 92.67%가 5 mmHg 이하로 매우 높은 비율을 보인다. 이는 DBP 추정이 SBP보다 상대적으로 쉬웠음을 시사한다. 실제로 모든 모델에서 DBP MAE가 SBP MAE보다 낮다.

ResNet50, InceptionV3, NASNetLarge, InceptionResNetV2도 DBP에서는 Grade A 기준을 만족하지만, SBP에서는 5 mmHg 이하 비율이 54.67%부터 58.67% 사이로 Grade A의 60% 기준에 약간 미치지 못한다. 그러나 10 mmHg 이하와 15 mmHg 이하 비율은 높은 편이므로, 매우 큰 오차가 많다기보다는 작은 오차 기준에서 ConvNeXtTiny와 VGG16보다 덜 정밀하다고 볼 수 있다.

### 4.5 기존 방법과의 비교

논문은 기존 deep learning 및 feature-based 혈압 추정 연구들과 성능을 비교한다. 비교 대상에는 visibility graph와 AlexNet/Ridge regression, temporal feature 기반 BiLSTM, LSTM-based autoencoder, CNN-LSTM, CNN-LSTM multitask learning, GRNN, attention residual improved U-Net, EEMD-TCN 등이 포함된다.

제안 방법 중 ConvNeXtTiny는 SBP MAE 2.95, DBP MAE 1.66으로 기존 다수 연구보다 낮은 오차를 보인다. VGG16도 SBP MAE 3.94, DBP MAE 2.56으로 경쟁력 있는 성능이다. 논문은 특히 ConvNeXtTiny와 VGG16이 AAMI와 BHS Grade A를 모두 만족한다는 점을 강조한다.

다만 비교 해석에는 주의가 필요하다. 비교 연구들은 사용한 데이터셋, subject 수, split 방식, segment 수, preprocessing, validation protocol이 서로 다를 수 있다. 예를 들어 어떤 연구는 소수 subject를 사용했고, 어떤 연구는 PPG와 ECG를 함께 사용했다. 이 논문도 3000개 segment를 사용하지만, subject-wise split 여부, record leakage 방지 여부, train-test subject overlap 여부는 명확하게 설명되어 있지 않다. 따라서 표면적인 MAE 비교만으로 완전한 우위를 단정하기는 어렵다. 논문은 임상 기준 충족을 강조하지만, 일반화 성능을 엄격히 검증하려면 subject-wise 또는 external dataset validation이 추가로 필요하다.

### 4.6 결과의 의미

실험 결과는 scalogram-based transfer learning이 PPG 기반 혈압 추정에서 유효할 수 있음을 보여준다. CWT scalogram은 PPG 신호의 시간적 변화와 주파수 성분을 동시에 담고 있으며, pretrained CNN은 이러한 2D pattern에서 혈압과 관련된 feature를 추출할 수 있다. Random Forest는 이 feature와 혈압값 사이의 비선형 mapping을 안정적으로 학습한다.

특히 ConvNeXtTiny의 성능은 흥미롭다. ConvNeXtTiny는 상대적으로 modern CNN 구조이면서도 output feature dimension이 NASNetLarge보다 훨씬 작다. NASNetLarge는 매우 큰 feature vector를 생성하지만 성능은 ConvNeXtTiny보다 낮다. 이는 feature dimension이 크다고 해서 혈압 추정에 유리한 것은 아니며, scalogram의 구조에 적합한 feature extractor가 중요함을 시사한다.

VGG16의 좋은 성능도 주목할 만하다. VGG16은 오래된 단순한 구조이지만, uniform convolution과 pooling 구조가 scalogram feature 추출에 안정적으로 작동했을 가능성이 있다. 복잡한 Inception 계열이나 NASNetLarge가 더 좋은 성능을 내지 못한 점은 biomedical signal scalogram task에서 모델 복잡도와 성능이 항상 비례하지 않음을 보여준다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 신호를 scalogram으로 변환하여 time-frequency 정보를 보존했다는 점이다. PPG는 비정상 생리 신호이므로, 단순 time-domain 분석이나 전체 주파수 분석만으로는 혈압 관련 동적 변화를 충분히 포착하기 어렵다. CWT는 시간과 주파수를 동시에 표현하므로 PPG waveform의 국소적 변화와 주기적 패턴을 함께 반영할 수 있다.

두 번째 강점은 transfer learning을 활용하여 계산 비용과 학습 데이터 요구량을 줄였다는 점이다. Pretrained CNN을 feature extractor로 사용하면, 처음부터 대규모 CNN을 학습하지 않고도 고차원 feature representation을 얻을 수 있다. 이는 의료 데이터가 제한적일 때 특히 유용하다.

세 번째 강점은 여러 pretrained CNN architecture를 비교했다는 점이다. VGG16, ResNet50, InceptionV3, NASNetLarge, InceptionResNetV2, ConvNeXtTiny를 동일한 pipeline에서 평가함으로써, 어떤 종류의 CNN feature extractor가 scalogram 기반 혈압 추정에 적합한지 탐색했다. ConvNeXtTiny와 VGG16이 우수하다는 결과는 향후 모델 설계의 참고점이 될 수 있다.

네 번째 강점은 Random Forest를 최종 회귀기로 사용하여 deep feature와 혈압 사이의 비선형 관계를 robust하게 모델링했다는 점이다. Random Forest는 ensemble 방식으로 variance를 줄이고 overfitting을 완화할 수 있으며, 작은 dataset에서도 비교적 안정적인 성능을 낼 수 있다.

다섯 번째 강점은 AAMI와 BHS 기준을 모두 사용해 임상적 의미를 강조했다는 점이다. 단순히 MAE만 제시하는 것이 아니라, error threshold별 sample 비율을 통해 BHS Grade를 계산하고 AAMI의 ME 및 SD 기준을 확인한다. 이는 혈압 추정 알고리즘의 임상적 신뢰성 평가에 중요하다.

### 5.2 한계

첫 번째 한계는 train-test split 방식의 엄격성이 충분히 명확하지 않다는 점이다. 논문은 10-fold cross-validation을 사용한다고 설명하지만, subject-wise split인지 segment-wise split인지 명확하지 않다. 만약 같은 subject 또는 같은 record에서 나온 segment가 training과 test에 동시에 포함된다면, model이 subject-specific pattern을 학습하여 성능이 과대평가될 수 있다. PPG 기반 혈압 추정 분야에서는 window-level random split이 data leakage를 유발할 수 있다는 문제가 잘 알려져 있으므로, 이 부분은 중요한 한계이다.

두 번째 한계는 데이터 규모가 제한적이라는 점이다. 원 dataset은 12,000 record를 포함하지만, 실제 실험에는 3000개의 PPG segment가 사용된다. 3000개의 scalogram은 pretrained feature extraction과 Random Forest 학습에는 가능할 수 있지만, 다양한 환자군과 혈압 분포를 대표하기에는 충분하지 않을 수 있다. 특히 논문은 external dataset validation을 수행하지 않는다.

세 번째 한계는 ABP target 추출 및 신호 품질 관리가 충분히 상세하지 않다는 점이다. ABP의 maximum과 minimum을 이용해 SBP와 DBP를 추출하지만, artifact, abnormal waveform, peak detection error, saturation, arrhythmia, motion artifact 등에 대한 체계적 제거 기준이 충분히 제시되어 있지 않다. 혈압 추정 성능은 label quality에 매우 민감하므로 이 부분은 중요하다.

네 번째 한계는 pretrained CNN feature의 해석 가능성이 낮다는 점이다. 논문은 scalogram에서 deep feature를 추출한다고 설명하지만, 어떤 time-frequency 영역이 SBP 또는 DBP 추정에 중요한지 분석하지 않는다. Random Forest feature importance를 활용하거나 Grad-CAM과 같은 saliency 분석을 적용했다면 생리적 해석 가능성이 높아졌을 것이다.

다섯 번째 한계는 model comparison이 architecture input size와 feature dimension 차이에 영향을 받을 수 있다는 점이다. NASNetLarge는 매우 큰 feature vector를 생성하지만 성능은 낮고, ConvNeXtTiny는 상대적으로 작은 feature vector로 좋은 성능을 낸다. 그러나 feature normalization, dimensionality reduction, Random Forest hyperparameter tuning이 각 feature extractor에 최적화되었는지는 명확하지 않다.

여섯 번째 한계는 “computational cost를 줄인다”는 주장에 비해 실제 inference latency, memory usage, feature extraction time이 정량적으로 제시되지 않았다는 점이다. Pretrained CNN과 Random Forest 조합은 end-to-end CNN보다 학습 비용은 낮을 수 있지만, scalogram 생성과 CNN feature extraction 자체가 웨어러블 실시간 처리에는 부담이 될 수 있다. 특히 CWT와 image resizing, CNN inference, Random Forest prediction을 모두 포함한 pipeline의 real-time feasibility는 추가 검증이 필요하다.

일곱 번째 한계는 clinical deployment와 관련된 calibration 문제, population shift, sensor shift를 다루지 않는다는 점이다. MIMIC-II 기반 dataset은 ICU 환경의 invasive ABP reference를 포함하지만, 실제 wearable PPG는 센서 위치, 피부색, 움직임, ambient light, device hardware에 따라 신호 특성이 달라진다. 논문은 이러한 외부 일반화 문제를 실험적으로 평가하지 않는다.

### 5.3 비판적 해석

이 논문은 PPG 기반 혈압 추정에서 scalogram과 pretrained CNN을 결합한 실용적 pipeline을 제안했다는 점에서 의미가 있다. 특히 ConvNeXtTiny와 VGG16의 성능은 매우 인상적이며, AAMI와 BHS 기준을 만족했다는 점은 강한 결과처럼 보인다. 그러나 이 결과를 실제 임상 적용 가능성으로 곧바로 해석하기에는 주의가 필요하다.

가장 중요한 검증 포인트는 subject-wise generalization이다. 혈압 추정 모델이 실제로 새로운 사람에게 잘 작동하려면, training subject와 test subject가 겹치지 않아야 한다. 제공된 텍스트만으로는 이 부분이 명확히 보장되지 않는다. 따라서 보고된 낮은 MAE는 segment overlap 또는 subject leakage의 영향을 받았을 가능성을 배제할 수 없다.

또한 Random Forest가 매우 고차원 deep feature를 입력으로 받는 구조이기 때문에, sample 수가 충분하지 않을 경우 feature space에서 과적합이 발생할 수 있다. 10-fold cross-validation은 평균 성능 추정에는 도움이 되지만, fold 분할 단위가 segment라면 leakage 문제를 해결하지 못한다.

그럼에도 이 논문의 접근법은 연구 방향으로 가치가 있다. CWT scalogram은 PPG의 time-frequency feature를 잘 보존할 수 있고, pretrained CNN을 활용하는 방식은 생리 신호를 image-like representation으로 변환해 computer vision model을 적용하는 하나의 효율적 전략이다. 향후 subject-wise external validation, signal quality control, model interpretability, real-time deployment cost 분석이 추가된다면 더 설득력 있는 혈압 추정 프레임워크가 될 수 있다.

## 6. 결론

이 논문은 PPG 신호 기반 비침습 혈압 추정을 위해 scalogram analysis와 transfer learning을 결합한 방법을 제안한다. 연구의 핵심은 PPG를 Morlet wavelet 기반 CWT로 변환하여 time-frequency scalogram을 만들고, 이를 pretrained CNN 모델에 입력해 deep feature를 추출한 뒤, Random Forest regression으로 SBP와 DBP를 예측하는 것이다.

실험 결과, ConvNeXtTiny와 VGG16이 가장 좋은 성능을 보였다. ConvNeXtTiny는 SBP MAE 2.95 mmHg, SD 4.11 mmHg, DBP MAE 1.66 mmHg, SD 2.60 mmHg를 달성했다. VGG16은 SBP MAE 3.94 mmHg, SD 5.19 mmHg, DBP MAE 2.56 mmHg, SD 3.80 mmHg를 달성했다. 두 모델은 AAMI 기준을 만족하고, BHS Grade A도 달성한 것으로 보고된다. Pearson correlation에서도 ConvNeXtTiny와 VGG16은 다른 모델보다 높은 상관성을 보였다.

논문의 주요 기여는 PPG 기반 혈압 회귀 문제에서 scalogram을 활용한 transfer learning pipeline을 제안하고, 여러 pretrained CNN feature extractor를 비교했으며, Random Forest와 결합하여 임상 기준 기반 평가를 수행했다는 점이다. 특히 ConvNeXtTiny가 좋은 성능을 보인 것은 최신 경량 CNN 구조가 time-frequency biomedical signal representation에서도 유용할 수 있음을 시사한다.

실제 적용 측면에서 이 연구는 웨어러블 기반 연속 혈압 모니터링의 가능성을 보여준다. PPG는 스마트워치와 같은 장치에서 쉽게 수집할 수 있고, cuffless monitoring에 적합하다. 따라서 이 방법이 충분히 일반화되고 실시간 구현 가능성이 검증된다면, 고혈압 조기 탐지와 심혈관 질환 관리에 활용될 수 있다.

그러나 현재 논문만으로는 임상 적용 가능성을 확정하기 어렵다. subject-wise split 여부, external dataset validation, signal quality control, data leakage 방지, real-time computational cost, 모델 해석 가능성 등이 추가로 검증되어야 한다. 향후 연구에서는 더 큰 규모의 다양한 환자군, 다른 센서 환경, 실제 웨어러블 데이터, multimodal physiological signal, subject-independent validation을 포함해야 한다.

종합하면, 이 논문은 PPG 혈압 추정에서 Morlet scalogram과 pretrained CNN을 결합하는 유망한 방법을 제시한다. ConvNeXtTiny와 VGG16의 성능은 특히 주목할 만하다. 다만 보고된 높은 정확도가 실제 사용자와 임상 환경에서도 유지되는지 확인하려면 더 엄격한 일반화 평가가 필요하다.
