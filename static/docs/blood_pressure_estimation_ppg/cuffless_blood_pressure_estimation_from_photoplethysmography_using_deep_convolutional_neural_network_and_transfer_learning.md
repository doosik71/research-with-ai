# Cuffless blood pressure estimation from photoplethysmography using deep convolutional neural network and transfer learning

* **저자**: Hüseyin Murat Koparır, Özkan Arslan
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 PPG(Photoplethysmography) 신호와 그 미분 신호를 이용하여 cuffless, non-invasive 방식으로 혈압을 추정하는 프레임워크를 제안한다. 구체적으로는 PPG 원 신호, 1차 미분 신호인 VPG(Velocity Plethysmography), 2차 미분 신호인 APG(Acceleration Plethysmography)의 waveform을 2차원 이미지로 변환하고, 사전 학습된 CNN 모델을 transfer learning 방식으로 활용하여 deep feature를 추출한다. 이후 RFE(Recursive Feature Elimination)를 통해 가장 구별력 있는 feature subset을 선택하고, 여러 machine learning 및 deep learning regression 모델을 사용하여 SBP, DBP, MBP를 추정한다.

논문의 연구 문제는 기존 PPG 기반 혈압 추정 연구가 주로 hand-crafted morphological feature에 의존한다는 점에서 출발한다. PPG waveform은 motion artifact, sensor contact, baseline drift, amplitude variability 등으로 쉽게 왜곡되며, 이 경우 peak, foot, dicrotic notch 같은 명시적 형태 특징을 안정적으로 추출하기 어렵다. 따라서 저자들은 waveform의 형태 정보를 이미지로 보존한 뒤, 이미지 인식 분야에서 강력한 feature extractor로 검증된 pre-trained CNN을 사용하면 더 풍부하고 안정적인 deep feature를 얻을 수 있다고 본다.

혈압 모니터링은 고혈압, 만성 심부전, 뇌졸중, 신장 질환 등 심혈관계 질환의 조기 탐지와 관리에 매우 중요하다. 기존 invasive 방식은 정확하지만 감염과 출혈 위험이 있고, cuff-based non-invasive 방식은 연속 측정이 어렵고 white coat syndrome, 불규칙 측정 등의 문제가 있다. 따라서 PPG 기반 cuffless BP estimation은 웨어러블 환경에서 연속적이고 편리한 혈압 모니터링을 가능하게 하는 중요한 연구 주제이다.

이 논문은 MIMIC-II dataset의 PPG와 invasive ABP 신호를 사용한다. ABP waveform에서 SBP와 DBP를 계산하고, MBP 또는 MAP는 SBP와 DBP로부터 산출한다. 논문은 최종적으로 VPG waveform image, DenseNet121 기반 deep feature, Bi-GRU regression model의 조합이 가장 우수한 성능을 보였으며, 제안 모델이 AAMI 기준을 만족하고 BHS protocol에서 Grade A를 달성했다고 보고한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG 기반 혈압 추정을 전통적인 1차원 signal feature engineering 문제로만 다루지 않고, waveform image 기반 deep feature extraction 문제로 재구성하는 것이다. 기존 연구들은 PPG 한 주기에서 systolic upstroke time, diastolic time, dicrotic notch 위치, area, slope, pulse width 등 다양한 morphological feature를 수작업으로 추출하였다. 하지만 이러한 방법은 신호 품질이 낮거나 waveform이 불명확하면 feature extraction 자체가 불안정해진다.

저자들은 PPG, VPG, APG의 시간 파형을 2-D image로 변환하면 waveform의 전체적인 형태 정보를 보존할 수 있고, pre-trained CNN이 이를 통해 사람이 직접 설계하지 않은 deep feature를 자동으로 추출할 수 있다고 본다. Transfer learning을 사용하면 대규모 의료 신호 이미지 데이터를 처음부터 학습하지 않아도, ImageNet 등에서 학습된 CNN의 feature extraction 능력을 활용할 수 있다.

또 다른 핵심 아이디어는 deep feature를 모두 사용하는 대신 RFE를 통해 최적의 feature subset을 선택한다는 점이다. CNN에서 추출된 feature는 고차원일 가능성이 높으며, 이 중 일부는 혈압 추정에 불필요하거나 중복될 수 있다. RFE는 regression model의 성능에 기여도가 낮은 feature를 반복적으로 제거하여 더 간결하고 구별력 있는 feature set을 만든다. 논문은 최종적으로 24개의 feature를 사용한다고 설명한다.

방법론적으로 이 논문은 세 가지 요소의 조합을 탐색한다. 첫째, 입력 신호 유형으로 PPG, VPG, APG를 비교한다. 둘째, deep feature extractor로 여러 pre-trained CNN 모델을 비교하며, 제공된 텍스트에서는 DenseNet121이 최적 조합에 포함된 것으로 보고된다. 셋째, regression model로 다양한 machine learning 및 deep learning 방법을 비교하며, 최종적으로 Bi-GRU가 가장 좋은 성능을 보인다고 설명한다.

기존 접근과의 차별점은 calibration-free 설정을 지향한다는 점에도 있다. Calibration-based 방식은 각 subject의 일부 데이터를 학습에 포함하고 나머지를 테스트하는 방식이어서 성능이 높게 나올 수 있지만, 새로운 subject에 대한 일반화 능력이 제한될 수 있다. 저자들은 calibration-free high-performance BP estimation model을 개발했다고 명시한다. 다만 제공된 텍스트만으로는 subject-independent split의 세부 구현 방식이 완전히 확인되지는 않는다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

제안 프레임워크는 여섯 단계로 구성된다. 첫째, MIMIC-II dataset에서 PPG와 ABP 신호를 가져온다. 둘째, PPG 신호에 noise removal, baseline correction, normalization 등의 전처리를 수행한다. 셋째, PPG와 그 미분 신호인 VPG, APG의 waveform을 2-D image로 변환한다. 넷째, 사전 학습된 CNN 모델을 transfer learning 방식으로 사용하여 waveform image에서 deep feature를 추출한다. 다섯째, random forest를 kernel로 사용하는 RFE algorithm을 적용하여 가장 중요한 feature만 선택한다. 여섯째, 선택된 feature를 여러 regression model에 입력하여 SBP, DBP, MBP를 추정하고, statistical metric과 AAMI, BHS 기준으로 평가한다.

이 구조의 중요한 특징은 PPG 신호를 직접 1-D regression model에 넣는 대신, waveform image representation과 CNN feature extractor를 중간 표현으로 사용한다는 점이다. 즉, PPG 기반 혈압 추정을 “생체 신호 회귀” 문제와 “이미지 기반 feature extraction” 문제를 결합한 형태로 설계하였다.

### 3.2 데이터셋

논문은 PhysioNet에서 제공되는 MIMIC-II dataset을 사용한다. 이 dataset은 다양한 연령과 성별을 가진 수천 명 환자로부터 측정된 physiological waveform을 포함하며, ECG, invasive ABP, PPG 등을 제공한다. 제공된 텍스트에 따르면 dataset은 942명의 환자 기록을 포함하고, 네 개의 section으로 나뉘며, 각 section은 3000개 record를 가져 총 12,000개 signal sample로 구성된다. 모든 signal은 125 Hz로 동시에 sampling되었다.

이 연구는 single-sensor 기반 접근을 제안하므로 ECG는 사용하지 않고 PPG와 reference ABP만 사용한다. ABP는 arterial catheter로 invasive하게 측정된 신호이므로, ground truth 혈압 산출에 사용된다. PPG는 fingertip sensor로 non-invasive하게 측정될 수 있는 신호이다.

각 record는 일부가 1000 sample로 제한되어 있고 많은 신호가 끝부분에서 왜곡되어 있으므로, 논문은 각 신호의 처음 1000 sample만 사용한다. 125 Hz sampling rate에서 1000 sample은 8초에 해당한다. 또한 문헌 비교를 위해 극단적으로 높은 혈압 record와 낮은 혈압 record를 제외한다. 제외 기준은 SBP가 180 mmHg 이상이거나 DBP가 130 mmHg 이상인 경우, 또는 SBP가 80 mmHg 이하이거나 DBP가 60 mmHg 이하인 경우이다. 이러한 제한을 적용한 뒤 최종적으로 942명 subject로부터 기록된 7014개의 quality signal이 확보되었다고 설명한다.

ABP waveform에서 최고 peak는 SBP, valley는 DBP로 사용된다. MBP 또는 MAP는 다음 식으로 계산된다.

$$
MBP = \frac{2 \times DBP + SBP}{3}
$$

최종 dataset의 혈압 분포는 제공된 표에 따르면 SBP는 80.25–179.96 mmHg, 평균 129.79 mmHg, 표준편차 20.87 mmHg이다. DBP는 60.00–129.13 mmHg, 평균 72.13 mmHg, 표준편차 10.63 mmHg이다. MBP는 66.76–139.80 mmHg, 평균 91.35 mmHg, 표준편차 11.51 mmHg이다.

### 3.3 신호 전처리

MIMIC-II dataset의 physiological signal은 noise와 distortion을 포함할 수 있으므로, 논문은 세 가지 주요 전처리를 수행한다. 첫째는 filtering, 둘째는 baseline deviation removal, 셋째는 normalization이다.

Noise reduction을 위해 DWT(Discrete Wavelet Transform)를 사용한다. 구체적으로 Daubechies 8(db8) wavelet function을 사용하여 신호를 10 level로 분해한다. 이후 남아 있을 수 있는 interference를 제거하고 PPG 신호를 band-limited signal로 만들기 위해 3rd order Butterworth band-pass filter를 적용한다. Cutoff frequency는 0.5–40 Hz로 설정된다.

PPG waveform에서 peak와 foot을 안정적으로 해석하려면 baseline이 일정해야 한다. 그러나 실제 PPG에서는 sensor, respiration, physical motion 등의 영향으로 foot 위치가 baseline에서 벗어나거나 amplitude drift가 발생할 수 있다. 이를 보정하기 위해 linear detrending을 수행한다.

또한 PPG amplitude는 subject별로 크게 다를 수 있고, VPG와 APG에서 얻어지는 amplitude scale도 서로 다를 수 있다. 따라서 min–max normalization을 사용하여 모든 PPG signal을 0–1 범위로 정규화한다. 이 normalization은 feature가 amplitude variation에 의해 과도하게 영향을 받지 않도록 하는 목적을 가진다.

다만 amplitude normalization은 PPG amplitude 자체가 가지는 생리학적 정보를 일부 제거할 수 있다는 점도 고려해야 한다. 논문은 waveform image 기반 deep feature를 통해 형태 정보를 보존하려고 하지만, absolute amplitude와 혈압 사이의 관계가 얼마나 유지되는지는 제공된 텍스트에서 별도로 분석되지 않는다.

### 3.4 PPG, VPG, APG waveform image 생성

논문은 PPG 원 신호뿐 아니라 PPG의 1차 미분 신호인 VPG와 2차 미분 신호인 APG를 사용한다. VPG는 PPG waveform의 변화율을 나타내므로 혈류 변화 속도와 관련된 정보를 더 강조한다. APG는 PPG의 acceleration 성분을 나타내며, 혈관 탄성, 반사파, 심혈관계 상태와 관련된 형태 정보를 더 민감하게 드러낼 수 있다.

이 연구의 중요한 설계는 PPG, VPG, APG의 time waveform을 2-D image로 변환하는 것이다. 논문은 waveform image가 정보를 보존하도록 변환된다고 설명하지만, 제공된 텍스트에서는 이미지 생성의 구체적인 해상도, 축 설정, line thickness, color channel 구성, background 처리, scaling 방식 등은 완전히 제시되지 않았다. 따라서 본 보고서에서는 waveform이 이미지로 변환되어 CNN 입력으로 사용된다는 점까지만 확정적으로 설명할 수 있다.

이 image-based representation은 CNN이 waveform의 전체 형태, peak pattern, slope pattern, curvature, derivative-based shape variation 등을 공간적 패턴으로 인식하게 만든다. 이는 handcrafted feature extraction이 불안정한 상황에서 유용할 수 있다.

### 3.5 Transfer learning 기반 deep feature extraction

CNN은 convolution, pooling, normalization 등의 layer를 통해 이미지의 local pattern과 high-level representation을 추출한다. 이 논문은 pre-trained CNN model의 feature extraction part를 사용하여 waveform image에서 deep feature를 자동으로 추출한다.

CNN의 핵심 연산은 kernel convolution이다. 제공된 텍스트에서 수식은 중간에 끊겨 있지만, 일반적으로 CNN의 2-D convolution은 입력 image $f$와 kernel $h$를 사용하여 feature map $G$를 다음과 같이 계산한다.

$$
G[m,n] = \sum_j \sum_k h[j,k] f[m-j,n-k]
$$

여기서 $f$는 입력 image, $h$는 convolution kernel, $G$는 output feature map이다. CNN은 여러 convolution filter를 통해 waveform image의 다양한 local pattern을 추출한다. Transfer learning에서는 이러한 filter와 layer가 대규모 이미지 데이터에서 이미 학습되어 있으므로, 작은 biomedical dataset에서도 feature extractor로 활용될 수 있다.

논문은 여러 pre-trained CNN 모델을 사용한 것으로 보이며, abstract에서는 DenseNet121 기반 feature가 최적 성능을 제공했다고 밝힌다. DenseNet121은 dense connection 구조를 사용하여 이전 layer의 feature를 이후 layer가 직접 활용할 수 있도록 설계된 CNN architecture이다. 이러한 구조는 feature reuse를 촉진하고 gradient flow를 개선하여, 복잡한 image pattern의 표현에 강점을 가진다.

### 3.6 RFE 기반 feature selection

CNN에서 추출된 deep feature는 고차원일 가능성이 높다. 고차원 feature를 모두 regression model에 입력하면 계산 비용이 커지고, 불필요한 feature가 overfitting을 유발할 수 있다. 이를 해결하기 위해 논문은 RFE(Recursive Feature Elimination)를 적용한다.

RFE는 feature subset을 점진적으로 줄여가며 모델 성능에 가장 덜 기여하는 feature를 제거하는 wrapper-type feature selection 방법이다. 이 논문에서는 random forest machine learning method를 kernel 또는 estimator로 사용한다고 설명한다. Random forest는 feature importance를 제공할 수 있으므로, 각 feature의 상대적 중요도를 바탕으로 덜 중요한 feature를 제거하는 데 활용될 수 있다.

논문은 RFE를 통해 최적의 24개 feature set을 선택했다고 밝힌다. 이 선택된 feature set은 이후 machine learning 및 deep learning regression method의 입력으로 사용된다. Feature selection은 모델의 계산 복잡도를 낮추고, BP estimation에 더 직접적으로 관련된 feature만 남겨 성능을 높이는 역할을 한다.

### 3.7 Regression model과 목표 변수

모델의 목표 변수는 SBP, DBP, MBP이다. SBP와 DBP는 ABP waveform에서 직접 얻고, MBP는 앞서 제시한 식으로 계산한다. 따라서 각 input waveform image에서 추출된 feature는 세 가지 regression target 중 하나를 예측하는 데 사용된다.

논문은 선택된 deep feature를 여러 machine learning 및 deep learning regression method에 입력한다고 설명한다. Abstract에서는 최종적으로 Bi-GRU(Bidirectional Gated Recurrent Unit)가 가장 좋은 BP estimation performance를 제공했다고 보고한다.

GRU는 recurrent neural network의 한 종류로, update gate와 reset gate를 사용하여 sequential dependency를 학습한다. Bi-GRU는 forward direction과 backward direction의 GRU를 함께 사용하여 입력 feature sequence의 양방향 context를 반영한다. 이 논문에서 CNN deep feature가 어떻게 Bi-GRU의 sequence input으로 구성되는지는 제공된 텍스트만으로 완전히 명확하지 않다. 다만 저자들은 Bi-GRU가 deep feature 기반 regression에서 가장 높은 성능을 보였다고 설명한다.

## 4. 실험 및 결과

### 4.1 실험 구성

실험은 MIMIC-II dataset에서 추출한 7014개의 quality signal을 사용한다. 입력 유형으로는 PPG, VPG, APG waveform image가 사용된다. Feature extractor로는 pre-trained CNN 모델들이 사용되며, 제공된 텍스트에서 최적 조합은 DenseNet121로 보고된다. Feature selection은 RFE로 수행되고, 최종적으로 24개 feature가 선택된다. Regression model로는 여러 machine learning 및 deep learning model이 사용되며, 최종적으로 Bi-GRU가 가장 우수한 성능을 보인다.

평가에는 statistical metric, visual analytical tool, international gold standard가 사용된다. Statistical metric으로는 일반적으로 ME(mean error), MAE(mean absolute error), RMSE(root mean square error), STD(standard deviation of error), correlation 등으로 평가했을 가능성이 있으나, 제공된 텍스트에서는 정확한 전체 metric list와 결과 table이 포함되어 있지 않다. 따라서 구체적 수치는 본 보고서에서 단정하지 않는다.

국제 기준으로는 AAMI와 BHS가 사용된다. AAMI 기준은 일반적으로 혈압 측정 장치의 평균 오차와 표준편차가 일정 기준 이하인지 평가한다. BHS protocol은 예측 오차가 5, 10, 15 mmHg 이하에 들어오는 누적 비율을 바탕으로 Grade A, B, C 등을 부여한다. 논문은 SBP, DBP, MBP estimation model이 BHS 기준에서 Grade A를 달성하고 AAMI standard를 만족한다고 보고한다.

### 4.2 주요 결과

제공된 abstract에 따르면 가장 좋은 성능은 VPG input image, DenseNet121 deep feature, Bi-GRU algorithm의 조합에서 얻어졌다. 이는 PPG 원 신호보다 1차 미분 신호인 VPG가 혈압 관련 정보를 더 잘 드러냈다는 것을 의미할 수 있다. VPG는 waveform의 상승과 하강 속도, peak 주변 변화율, dicrotic notch 부근의 동적 변화 등을 강조하기 때문에 SBP, DBP, MBP estimation에 유용한 feature를 제공할 수 있다.

DenseNet121이 좋은 성능을 보인 이유는 dense connection을 통해 다양한 level의 feature를 효율적으로 재사용하기 때문으로 해석할 수 있다. 혈압 추정에 필요한 waveform image feature는 단순 edge나 local shape뿐 아니라 전체 waveform pattern과 derivative pattern을 포함할 수 있는데, DenseNet 계열은 이러한 multi-level representation을 잘 보존할 수 있다.

Bi-GRU가 좋은 성능을 보였다는 점은 선택된 deep feature들 사이에 순차적 또는 구조적 dependency가 존재할 수 있음을 시사한다. Bidirectional recurrent model은 앞뒤 방향의 정보를 함께 반영하므로, 단순 regression model보다 feature 간 관계를 더 잘 학습했을 가능성이 있다.

논문은 제안 모델이 AAMI standard를 만족하고, BHS protocol에서 SBP, DBP, MBP 모두 Grade A를 달성했다고 보고한다. 이는 제안 프레임워크가 단순히 평균 오차가 낮은 수준을 넘어, 혈압 측정 장치 평가 기준에서도 우수하다고 주장하는 근거가 된다. 다만 제공된 텍스트에는 실제 ME, STD, MAE, RMSE 수치가 포함되어 있지 않으므로, 본 보고서에서는 “논문이 그렇게 보고한다”는 수준으로만 기술한다.

### 4.3 기존 연구와의 비교

논문은 MIMIC-II dataset을 사용한 established studies와 비교했으며, comparative analysis에서 제안 방법이 기존 기법보다 우수하다고 주장한다. 기존 방법들은 주로 PPG morphological feature, time-domain feature, frequency-domain feature, statistical feature 등을 수작업으로 추출하고 SVR, random forest, AdaBoost, feedforward neural network, LSTM, GRU 등을 적용하였다.

제안 방법의 차별점은 수작업 feature가 아니라 waveform image에서 추출된 deep feature를 사용한다는 점이다. 또한 RFE로 feature selection을 수행하고 Bi-GRU로 regression을 수행한다. 이러한 조합이 기존 handcrafted feature 기반 모델보다 높은 성능을 제공했다는 것이 논문의 핵심 실험 결과이다.

다만 제공된 텍스트에는 비교 대상 논문의 구체적 수치, 실험 protocol의 동일성, calibration-based 여부, subject split 방식 등이 충분히 포함되어 있지 않다. 따라서 “기존 연구보다 우수하다”는 주장은 원문 전체의 결과 table을 확인해야 더 엄밀하게 평가할 수 있다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 PPG 기반 혈압 추정에서 waveform image와 transfer learning을 결합했다는 점이다. 기존의 handcrafted feature 방식은 peak, foot, dicrotic notch가 명확해야 하고, 신호 품질이 낮으면 feature extraction이 실패할 수 있다. 반면 제안 방법은 waveform 전체를 이미지로 표현하고 CNN이 deep feature를 자동으로 추출하므로, 더 풍부한 형태 정보를 활용할 가능성이 있다.

두 번째 강점은 PPG뿐 아니라 VPG와 APG를 모두 고려했다는 점이다. PPG의 1차, 2차 미분은 혈류 변화 속도와 가속도에 해당하는 형태 정보를 강조한다. 특히 최적 결과가 VPG에서 나왔다는 점은 derivative signal이 혈압 추정에 중요한 정보를 제공할 수 있음을 보여준다.

세 번째 강점은 feature selection을 포함했다는 점이다. Transfer learning으로 추출된 deep feature는 고차원이고 중복될 수 있다. RFE를 통해 24개 feature로 줄이면 계산 효율이 좋아지고, 불필요한 feature로 인한 overfitting 가능성을 줄일 수 있다.

네 번째 강점은 AAMI와 BHS 같은 국제 기준을 사용해 평가했다는 점이다. 많은 BP estimation 연구가 MAE나 RMSE만 보고하는 데 비해, 혈압 측정 장치의 실제 적합성을 판단하려면 AAMI/BHS 평가가 중요하다. 논문은 제안 모델이 BHS Grade A와 AAMI 기준을 만족한다고 보고하므로, 실용적 관점의 평가를 포함한다는 장점이 있다.

다섯 번째 강점은 calibration-free model을 목표로 했다는 점이다. Calibration-free 방식은 새로운 subject에 대한 일반화 능력을 평가하는 데 더 적합하다. 실제 wearable BP monitoring에서는 사용자의 calibration 부담을 줄이는 것이 중요하므로, calibration-free 접근은 실용적 의미가 크다.

### 5.2 한계

첫 번째 한계는 제공된 텍스트 기준으로 dataset split 방식이 완전히 명확하지 않다는 점이다. 논문은 calibration-free model을 개발했다고 주장하지만, train/test split이 subject-independent로 엄격히 수행되었는지, 또는 record-level split인지 제공된 부분만으로는 확인할 수 없다. MIMIC-II의 같은 subject에서 나온 여러 8초 segment가 train과 test에 동시에 들어가면 성능이 과대평가될 수 있다.

두 번째 한계는 극단적 혈압 record를 제외했다는 점이다. 논문은 SBP가 180 mmHg 이상 또는 80 mmHg 이하, DBP가 130 mmHg 이상 또는 60 mmHg 이하인 record를 제외한다. 이는 기존 연구와의 비교를 위한 선택이라고 설명하지만, 실제 임상적으로는 극단적 고혈압과 저혈압이 매우 중요하다. 따라서 모델이 hypertensive crisis 또는 hypotension 상황에서 잘 작동하는지는 이 연구만으로 평가하기 어렵다.

세 번째 한계는 waveform image 변환 방식의 세부 설명이 제공된 텍스트에서 부족하다는 점이다. PPG/VPG/APG를 이미지로 만들 때 이미지 크기, scaling, line rendering 방식, anti-aliasing, color channel, axis 제거 여부 등이 CNN feature에 영향을 줄 수 있다. 이러한 설정은 재현성에 중요하지만, 제공된 부분에서는 확인되지 않는다.

네 번째 한계는 transfer learning이 원래 natural image domain에서 학습된 CNN을 biomedical waveform image에 적용하는 방식이라는 점이다. ImageNet pre-trained CNN의 low-level filter는 edge나 texture 추출에는 유용할 수 있지만, PPG waveform image의 생리학적 의미와 직접적으로 맞춰진 것은 아니다. 따라서 transfer learning feature가 어떤 생리학적 패턴을 포착하는지 해석 가능성이 제한될 수 있다.

다섯 번째 한계는 실제 wearable 환경의 motion artifact나 long-term monitoring에 대한 검증이 제공된 텍스트에서는 확인되지 않는다는 점이다. MIMIC-II는 ICU 환경에서 측정된 신호이며, 웨어러블 환경의 손목 PPG, 움직임, sensor pressure 변화, 피부색, 온도 변화 등과는 차이가 있다. 따라서 병원 데이터에서 좋은 성능이 실제 일상 환경으로 곧바로 이어진다고 보기는 어렵다.

여섯 번째 한계는 ABP에서 계산된 SBP/DBP를 ground truth로 사용하는 것은 타당하지만, cuffless BP device validation 관점에서는 독립적 임상 프로토콜 검증이 필요하다는 점이다. 논문이 AAMI/BHS 기준을 만족한다고 보고하더라도, 실제 표준 검증은 subject 수, 측정 조건, reference device, BP range distribution 등 엄격한 조건을 요구한다. 제공된 텍스트만으로는 이러한 조건을 모두 충족했는지 판단하기 어렵다.

### 5.3 비판적 해석

이 연구는 PPG 기반 혈압 추정에서 feature engineering을 CNN transfer learning으로 대체하려는 실용적 시도이다. 특히 VPG image와 DenseNet121 feature가 좋은 성능을 보였다는 결과는, waveform의 derivative representation이 혈압 추정에 중요한 정보를 제공할 수 있음을 시사한다. 이는 기존 PPG morphology 연구에서 slope, upstroke time, derivative peak 등이 중요하게 다뤄진 것과도 일관된다.

그러나 제안 방법은 PPG를 이미지로 바꾸고 natural image CNN으로 feature를 추출하기 때문에, 모델의 해석 가능성이 제한될 수 있다. 의료 신호 분석에서는 단순히 높은 정확도뿐 아니라 어떤 waveform 특성이 혈압 추정에 기여하는지 설명하는 것이 중요하다. RFE로 feature selection을 수행했지만, 선택된 deep feature가 실제로 어떤 생리학적 의미를 갖는지에 대한 해석은 제공된 텍스트에서 확인되지 않는다.

또한 극단적 혈압 sample을 제외한 점은 성능을 안정적으로 만들 수 있지만, 임상적으로 중요한 high-risk group 성능을 평가하지 못하게 한다. 실제 hypertension monitoring system은 정상 범위뿐 아니라 고혈압, 저혈압, 급격한 혈압 변화 상황에서 robust해야 한다. 따라서 향후 연구에서는 극단적 혈압 범위를 포함한 검증과 group-wise performance 분석이 필요하다.

## 6. 결론

이 논문은 PPG 기반 cuffless blood pressure estimation을 위해 CNN transfer learning, RFE feature selection, Bi-GRU regression을 결합한 새로운 프레임워크를 제안한다. PPG, VPG, APG waveform을 2-D image로 변환하고, pre-trained CNN을 사용해 deep feature를 자동 추출한 뒤, RFE로 24개의 핵심 feature를 선택하여 SBP, DBP, MBP를 추정한다.

제공된 텍스트에 따르면 최적 조합은 VPG input image, DenseNet121 deep feature, Bi-GRU regression model이며, 제안 모델은 AAMI standard를 만족하고 BHS protocol에서 Grade A를 달성한다. 이는 waveform image 기반 deep feature가 기존 handcrafted morphological feature보다 더 효과적일 수 있음을 보여준다.

이 연구의 주요 기여는 PPG 기반 혈압 추정에서 transfer learning 기반 deep feature extraction을 적극적으로 활용했다는 점, derivative waveform인 VPG의 유용성을 보였다는 점, RFE를 통해 계산 복잡도를 낮추고 feature subset을 최적화했다는 점이다. 또한 calibration-free 고성능 모델을 목표로 했다는 점에서 실제 적용 가능성을 고려한 연구로 볼 수 있다.

다만 실제 임상 적용을 위해서는 subject-independent split의 명확한 검증, 극단적 혈압 범위 포함, waveform image 변환 방식의 재현성 강화, deep feature의 생리학적 해석, wearable 환경에서의 long-term validation이 추가로 필요하다. 이러한 보완이 이루어진다면, 본 연구의 접근은 웨어러블 기반 고혈압 위험군 모니터링이나 병원 밖 연속 혈압 추정 시스템 개발에 중요한 기반이 될 수 있다.
