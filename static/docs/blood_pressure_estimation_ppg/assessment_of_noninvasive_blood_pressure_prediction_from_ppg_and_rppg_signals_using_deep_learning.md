# Assessment of Non-Invasive Blood Pressure Prediction from PPG and rPPG Signals Using Deep Learning

* **저자**: Fabian Schrumpf, Patrick Frenzel, Christoph Aust, Georg Osterhoff, Mirco Fuchs
* **발표연도**: 2021

## 1. 논문 개요

이 논문은 PPG와 rPPG 신호를 이용한 deep learning 기반 non-invasive blood pressure prediction이 실제로 얼마나 신뢰할 수 있는지를 체계적으로 평가한 연구이다. 논문의 목적은 새로운 최고 성능의 혈압 예측 모델을 제안하는 것이 아니라, 이미 널리 사용되는 neural network architecture들이 PPG 및 rPPG 기반 혈압 예측에서 어떤 조건에서 성능이 좋아 보이고, 어떤 조건에서 실제 일반화 성능이 떨어지는지를 분석하는 데 있다.

PPG는 손가락 clip sensor 등으로 쉽게 측정할 수 있는 photoplethysmography 신호이고, rPPG는 camera 기반으로 얼굴 피부의 미세한 색 변화에서 유도되는 remote photoplethysmography 신호이다. 두 신호 모두 혈류량 변화와 관련된 pulse wave를 포함하므로, cuffless blood pressure estimation의 후보 신호로 활발히 연구되어 왔다. 특히 rPPG는 신체 접촉 없이 표준 RGB camera로 측정할 수 있기 때문에, 병원 내 비접촉 모니터링, 원격 진료, 모바일 헬스케어, 감염 위험이 있는 환경에서 매우 매력적인 기술이다.

그러나 저자들은 기존 연구의 평가 방식에 근본적인 문제가 있다고 지적한다. 많은 논문이 전체 test set의 mean absolute error, 즉 MAE만 보고하고, 이 값이 낮으면 AAMI 또는 BHS 기준에 근접한다고 주장한다. 하지만 혈압 데이터는 균등하게 분포하지 않는다. 대부분의 sample은 정상 또는 특정 중간 혈압 범위에 몰려 있고, 저혈압이나 고혈압처럼 임상적으로 중요한 구간은 상대적으로 적다. 이 경우 모델은 실제로 혈압을 잘 추정하지 못하더라도, training distribution의 mode 근처 값을 주로 예측함으로써 전체 평균 MAE를 낮게 만들 수 있다.

따라서 이 논문의 핵심 연구 문제는 “PPG나 rPPG로 혈압을 예측할 수 있는가”가 아니라, “PPG/rPPG 기반 deep learning 혈압 예측 모델이 subject-independent 조건과 전체 혈압 범위에서 임상적으로 신뢰 가능한가”이다. 이를 위해 저자들은 입력 segment 구성 방식, window length, derivative 사용 여부, mixed versus non-mixed train/test split, 혈압 bin별 MAE, personalization, rPPG transfer learning을 모두 평가한다.

논문의 결론은 매우 신중하다. PPG 기반 neural network는 일부 조건에서 mean regressor보다 낮은 error를 보이지만, subject가 완전히 분리된 non-mixed setting에서는 성능 개선 폭이 작고, 혈압 분포의 양끝, 즉 sample이 적은 저혈압 및 고혈압 구간에서 error가 크게 증가한다. 반대로 subject가 train/test에 섞이는 mixed setting에서는 MAE가 크게 낮아지지만, 이는 실제 신규 환자에 대한 일반화 성능을 과대평가할 가능성이 크다. rPPG 기반 예측은 transfer learning과 personalization을 적용해도 여전히 challenging하며, 특히 clinical application에 필요한 정확도에는 미치지 못한다.

이 논문은 PPG 및 rPPG 기반 cuffless BP estimation 연구에서 평가 설계가 얼마나 중요한지를 보여주는 methodological assessment 성격의 논문이다. 저자들은 향후 연구에서 subject-aware split, full BP range evaluation, bin-wise error analysis, mean regressor baseline, personalization 여부를 명확히 보고해야 한다고 강조한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 deep learning 기반 혈압 예측 모델의 낮은 평균 오차가 반드시 실제 생리학적 일반화 능력을 의미하지는 않는다는 것이다. 특히 PPG 기반 혈압 예측에서는 subject-specific pulse morphology가 매우 강하게 나타난다. 나이, 혈관 탄성, 심혈관 질환, 약물 복용, 피부 상태, sensor contact pressure, 측정 장비, ICU 환경 등이 모두 PPG waveform shape에 영향을 준다. 따라서 같은 subject의 일부 sample이 training set에 있고 다른 sample이 test set에 있으면, 모델은 새로운 사람의 혈압을 예측하는 것이 아니라 이미 본 subject의 morphology를 기억한 상태에서 비슷한 sample을 예측하는 상황이 될 수 있다.

이를 검증하기 위해 저자들은 mixed dataset과 non-mixed dataset을 비교한다. Mixed dataset에서는 subject affiliation을 무시하고 sample을 무작위로 train, validation, test set에 나눈다. 이 경우 같은 subject의 데이터가 training과 test에 동시에 들어갈 수 있다. Non-mixed dataset에서는 subject 단위로 완전히 분리하여, 특정 subject는 training, validation, test 중 하나에만 포함된다. 실제 임상 적용에서는 새로운 환자의 데이터를 예측해야 하므로 non-mixed setting이 훨씬 현실적이다.

실험 결과 mixed setting에서는 MAE가 크게 낮아졌다. 예를 들어 ResNet은 mixed dataset에서 SBP MAE 7.7 mmHg, DBP MAE 4.4 mmHg를 기록했지만, non-mixed dataset에서는 SBP MAE 16.4 mmHg, DBP MAE 8.5 mmHg로 크게 악화되었다. 이는 sample-level random split이 성능을 상당히 낙관적으로 보이게 만들 수 있음을 보여준다.

두 번째 핵심 아이디어는 평균 MAE 대신 혈압 범위별 error를 봐야 한다는 것이다. 저자들은 혈압을 10 mmHg 폭의 bin으로 나누고 각 bin에서 MAE를 별도로 계산했다. 그 결과 sample이 많은 100–130 mmHg SBP 구간과 50–70 mmHg DBP 구간에서는 error가 낮았지만, sample이 적은 낮은 혈압 및 높은 혈압 구간에서는 error가 크게 증가했다. 이는 neural network가 전체 loss를 줄이기 위해 데이터가 많은 구간에 예측을 집중하는 경향, 즉 distribution mode로 bias되는 경향을 가진다는 것을 의미한다.

세 번째 핵심 아이디어는 personalization의 필요성이다. PPG morphology가 subject마다 크게 다르다면, 완전히 population-level model만으로는 충분한 정확도를 얻기 어렵다. 저자들은 test subject 데이터의 일부, 구체적으로 첫 20% 또는 무작위 20%를 사용해 pre-trained model을 fine-tuning하는 personalization을 수행했다. 그 결과 PPG와 rPPG 모두에서 prediction error가 유의하게 감소했다. 다만 personalization strategy 자체, 즉 처음 20%를 쓰는지 무작위 20%를 쓰는지는 큰 차이를 만들지 않았다.

네 번째 핵심 아이디어는 rPPG 기반 혈압 예측의 가능성과 한계를 동시에 평가하는 것이다. rPPG는 PPG와 유사한 pulse waveform을 제공하지만, motion artifact, illumination variation, ROI instability, skin tone, camera noise 등의 영향을 훨씬 크게 받는다. 저자들은 PPG에서 학습한 network를 rPPG에 transfer learning으로 적용하고, rPPG 데이터로 final layer를 fine-tuning했다. Fine-tuning은 성능을 크게 개선했지만, rPPG 기반 BP prediction은 여전히 subject별 편차가 크고 임상 기준을 만족하기 어렵다고 평가된다.

## 3. 상세 방법 설명

이 연구의 전체 방법은 크게 PPG 데이터 준비, rPPG 데이터 준비, neural network architecture 선택, 입력 segment parameterization 평가, PPG 기반 BP prediction, PPG personalization, rPPG transfer learning 및 rPPG personalization으로 구성된다.

### 3.1 데이터셋

논문은 두 종류의 PPG 데이터셋과 하나의 rPPG 데이터셋을 사용한다.

첫 번째 PPG 데이터셋은 MIMIC-A이다. 이는 Kaggle에 공개된 MIMIC-III subset으로, PPG, ECG, ABP signal을 포함하는 12,000개의 record로 구성되어 있다. 이 데이터는 이미 extensive preprocessing이 적용되어 signal quality가 비교적 양호하고 compact하다는 장점이 있다. 그러나 subject affiliation, 즉 각 sample이 어떤 subject에 속하는지 알 수 없다. 따라서 이 데이터셋은 최종 성능 평가에는 적절하지 않고, 입력 segment 길이와 cropping strategy를 평가하는 초기 실험에만 사용된다.

두 번째 PPG 데이터셋은 MIMIC-B이다. 이는 Slapničar et al. 연구에서 제공된 script를 이용해 MIMIC-III database에서 다운로드한 더 큰 데이터셋이다. 원문은 4,000개의 record와 약 150 million 또는 결과부에서 약 170 million sample 규모의 PPG-ABP signal pair pool을 언급한다. 이 데이터셋은 실제 성능 평가에 사용된다. 저자들은 특정 subject가 지나치게 많은 sample을 제공하여 dataset imbalance를 만드는 문제를 방지하기 위해, 각 subject가 dataset에 기여할 수 있는 sample 수를 2,000개로 제한했다. 이는 subject 수를 충분히 유지하면서도 특정 subject가 training, validation, test 성능을 지배하지 않도록 하기 위한 절충이다.

rPPG 데이터는 Leipzig University Hospital에서 clinical study로 수집되었다. 연구는 University of Leipzig ethics committee의 승인을 받았고, subject들은 서면 동의했다. 총 50명의 수술 예정 환자가 등록되었으며, 수술 후 ICU로 옮겨진 뒤 얼굴과 상반신 video가 촬영되었다. 촬영 장비는 IDS UI-3040CP industrial USB camera이고 frame rate는 32 fps이다. 각 영상은 약 2시간 길이이며, ground truth BP는 bedside monitor에서 1분 간격으로 수집되었다. 이후 motion artifact, frequent movement, insufficient lighting이 심한 subject는 제외되었다.

### 3.2 Neural network architecture

저자들은 네 가지 neural network architecture를 비교했다. 이 논문이 CVPR workshop version보다 확장된 부분 중 하나가 LSTM architecture 추가이다.

첫 번째 모델은 AlexNet 기반 CNN이다. AlexNet은 원래 image classification용 CNN architecture이지만, 이 연구에서는 PPG time series를 입력으로 받고 SBP와 DBP 두 값을 출력하도록 수정되었다. 원래 classification layer는 linear activation을 가진 regression layer로 교체되었다.

두 번째 모델은 ResNet 기반 CNN이다. ResNet은 깊은 neural network에서 gradient가 소실되는 문제를 residual connection으로 완화한다. 이 연구에서는 ResNet도 AlexNet과 마찬가지로 1D PPG signal regression에 맞게 수정했다. 입력 dimension은 raw PPG만 사용할 때 $N_{samp} \times 1$이고, raw PPG와 1차 및 2차 derivative를 함께 사용할 때는 $N_{samp} \times 3$이다.

세 번째 모델은 Slapničar et al.이 제안한 spectrotemporal residual network이다. 이 구조는 PPG waveform과 그 1차 및 2차 derivative를 병렬로 처리하는 architecture로, PPG 기반 BP prediction에 특화되어 있다. 저자들은 이 모델을 일반 CNN인 AlexNet, ResNet과 비교하기 위해 포함했다.

네 번째 모델은 bidirectional LSTM 기반 recurrent neural network이다. LSTM은 sequence 내부의 long-term dependency를 학습하기 위해 feedback connection과 gating mechanism을 사용하는 구조이다. 본 논문에서는 PPG time series 내 temporal dependency가 혈압과 관련될 수 있다고 보고 LSTM을 사용했다. 이 LSTM architecture는 첫 번째 convolutional layer, 세 개의 LSTM layer, 하나의 dense layer로 구성된다. Convolutional layer는 32개 filter, kernel size 5, stride 1, ReLU activation을 사용한다. LSTM layer의 hidden unit 수는 원문에 64, 64, 32로 제시되어 있으며, 마지막 dense layer는 128 neuron으로 구성된다. 최종 출력은 SBP와 DBP이다.

### 3.3 PPG signal processing 및 입력 segment 생성

저자들은 먼저 PPG 입력 segment를 어떻게 구성할지 평가했다. 두 가지 방식이 비교되었다.

첫 번째는 const_time 방식이다. PPG와 ABP signal을 1, 2, 5, 7, 9, 11, 13, 15, 17, 20초 길이의 고정 시간 window로 나눈다. 이 방식은 단순하지만, window의 시작과 끝에서 heartbeat cycle이 중간에 잘릴 수 있다. 이렇게 되면 PPG waveform의 phase discontinuity가 생기고, 모델이 pulse morphology를 안정적으로 학습하기 어렵다.

두 번째는 const_beats 방식이다. 저자들은 PPG에서 spectral amplitude가 가장 큰 component를 찾아 heart rate를 추정한 뒤, 각 segment가 정수 개수의 PPG wave, 즉 complete beat를 포함하도록 나눈다. 이후 각 PPG wave의 duration을 1초 또는 125 sample로 맞추어 resampling한다. 예를 들어 $N_P=4$개의 PPG wave를 포함하는 window는 $4 \times 125=500$ sample로 resampling된다. 이 방식은 complete pulse cycle을 유지하지만, heart rate 차이를 resampling으로 제거하므로 absolute temporal information 일부를 잃는다.

Ground truth SBP와 DBP는 ABP segment에서 peak detection algorithm으로 systolic peak와 diastolic peak를 검출한 뒤, 각 segment 내부 peak들의 median으로 계산한다. Physiological plausibility check도 적용된다. SBP가 75–165 mmHg 범위를 벗어나거나 DBP가 40–80 mmHg 범위를 벗어나면 제외하고, median heart rate가 50–140 bpm 범위를 벗어나는 window도 제거한다.

저자들은 raw PPG뿐 아니라 1차 derivative와 2차 derivative를 함께 사용하는 multivariate input도 평가했다. 기존 연구에서는 derivative가 vascular state나 waveform 변화율 정보를 반영할 수 있다고 보고되었기 때문이다. 그러나 본 연구에서는 derivative 사용이 전반적인 성능 향상을 제공하지 않았고, 이후 분석에서는 단순성을 위해 derivative를 사용하지 않았다.

### 3.4 SNR 계산과 window length 결정

rPPG는 PPG보다 noise가 크기 때문에, window length를 정할 때 SNR을 함께 고려했다. 저자들은 de Haan et al.이 제안한 방식으로 SNR을 계산했다. Pulse rate에 해당하는 spectral peak와 그 first harmonic 주변의 에너지를 $E_P$로 계산하고, 그 외 spectrum 영역의 에너지를 $E_S$로 계산한다. SNR은 다음과 같이 정의된다.

$$
SNR = 10 \log_{10}\frac{E_P}{E_S}
$$

SNR threshold는 -7 dB로 설정되었고, rPPG segment의 SNR이 -7 dB보다 낮으면 제외했다. 너무 짧은 window에서는 SNR 계산이 불안정하고, 너무 긴 window에서는 training sample 수가 줄어든다. 저자들은 PPG prediction 성능과 rPPG SNR/sample 수 trade-off를 고려해 최종적으로 7초 window length를 선택했다.

### 3.5 PPG 기반 학습 및 mixed/non-mixed dataset 비교

MIMIC-B dataset은 optimal cropping strategy와 window length를 적용해 나뉘었다. PPG signal에는 4th order Butterworth band-pass filter가 적용되었고 cutoff frequency는 0.5 Hz와 8 Hz이다. 이후 모든 PPG window의 SNR을 계산하고, SNR이 -7 dB 미만인 window는 제거했다. 각 window는 zero mean과 unit variance로 normalization되었다.

저자들은 두 가지 dataset split을 만들었다.

첫 번째는 non-mixed dataset이다. Training, validation, test set을 subject-basis로 완전히 분리한다. Training에는 3,750명, validation과 test에는 각각 625명이 사용되며, 각각 1,000,000개, 250,000개, 250,000개의 sample이 추출된다. 이 설정에서는 test subject가 training 중 절대 등장하지 않으므로, 실제 신규 환자에 대한 generalization을 평가할 수 있다.

두 번째는 mixed dataset이다. Sample pool에서 750명의 subject를 선택하고 각 subject가 2,000개의 sample을 제공하도록 한 뒤, subject affiliation을 무시하고 training, validation, test set을 무작위로 나눈다. 이 경우 같은 subject의 sample이 training과 test에 동시에 존재할 수 있다. 저자들은 이를 통해 subject leakage가 모델 성능에 어떤 영향을 주는지 분석했다.

학습은 TensorFlow 2.4.1과 Python 3.8로 수행되었다. Optimizer는 Adam이고 learning rate는 $\alpha=0.001$이다. Loss는 Euclidean loss로 제시되어 있으며, 최대 200 epoch까지 학습하되 validation loss가 10 epoch 동안 개선되지 않으면 early stopping을 적용했다. 최종 test에는 validation MAE가 가장 낮은 모델을 사용했다.

평가 지표는 MAE이다.

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|
$$

여기서 $\hat{y}_i$는 예측 혈압값이고, $y_i$는 reference 혈압값이다. 저자들은 전체 MAE뿐 아니라 혈압을 10 mmHg 폭의 bin으로 나누어 bin-wise MAE도 계산했다.

### 3.6 Mean regressor baseline

논문은 mean regressor를 baseline으로 사용한다. Mean regressor는 training set의 평균 SBP와 평균 DBP를 항상 예측하는 단순한 모델이다. 복잡한 neural network가 mean regressor보다 크게 우수하지 않다면, 모델이 PPG waveform에서 일반화 가능한 혈압 관련 정보를 충분히 학습했다고 보기 어렵다.

이 baseline은 특히 혈압 데이터가 불균형할 때 중요하다. 데이터가 정상 범위에 몰려 있으면 평균값만 출력해도 전체 MAE가 생각보다 낮아질 수 있기 때문이다. 따라서 mean regressor와의 비교는 deep learning model이 단순히 distribution mode를 따라가는지, 아니면 실제 waveform morphology를 활용하는지 판단하는 데 도움이 된다.

### 3.7 PPG 기반 personalization

저자들은 non-mixed test set에서 20명의 subject를 무작위로 선택하고, 각 subject에 대해 pre-trained model을 fine-tuning하는 personalization을 수행했다. 두 가지 personalization strategy가 비교되었다.

첫 번째는 random personalization이다. Test subject의 전체 measurement에서 20%의 sample을 무작위로 뽑아 fine-tuning에 사용한다. 두 번째는 first personalization이다. Test subject의 measurement 시작 부분에서 처음 20%의 sample을 사용한다. 후자는 실제 상황에서 환자가 병원에 입원할 때 cuff-based measurement를 이용해 calibration을 수행한 뒤 이후 non-invasive prediction을 사용하는 scenario와 더 유사하다.

비교 가능성을 유지하기 위해, test에는 두 personalization strategy 모두에서 겹치지 않는 동일한 나머지 sample subset을 사용했다. 통계적 유의성은 Kolmogorov-Smirnov test로 평가했다.

### 3.8 rPPG 기반 prediction 및 transfer learning

rPPG에서는 subject의 forehead와 cheeks ROI를 수동으로 labeling했다. 이후 POS, 즉 Plane-Orthogonal-to-Skin algorithm으로 skin pixel에서 rPPG pulse wave를 추출했다. 추출된 rPPG signal은 시각적으로 검사되었고, motion artifact나 lighting 문제가 심한 subject는 제외되었다.

남은 rPPG data는 heart rate 기반으로 windowing되며, 각 window에는 7개의 heartbeat가 포함된다. PPG와 동일하게 resampling하고, SNR이 -7 dB 미만인 window는 제거했다. Ground truth BP는 bedside monitor에서 얻었다.

rPPG dataset은 크기가 작기 때문에 처음부터 network를 학습하지 않았다. 대신 PPG로 pre-trained된 network를 가져와 대부분의 weight를 freezing하고 final layer만 rPPG data로 fine-tuning했다. 이는 PPG와 rPPG가 유사한 pulse waveform property를 공유한다는 transfer learning 가정에 기반한다.

rPPG 실험은 leave-two-out cross-validation 방식으로 수행되었다. 17명 중 15명은 fine-tuning에 사용하고, 1명은 validation, 1명은 test에 사용했다. 또한 rPPG에서도 personalization을 평가했다. Test subject의 처음 20% 또는 random 20% sample을 training에 추가하고, 나머지 sample로 성능을 평가했다.

## 4. 실험 및 결과

### 4.1 입력 segment 구성 결과

AlexNet과 ResNet을 사용해 MIMIC-A dataset에서 input segment 구성 방식을 평가한 결과, const_HR 또는 const_beats 방식이 const_time 방식보다 낮은 prediction error를 보였다. 이는 complete heartbeat cycle을 포함하도록 segment를 구성하는 것이 고정 시간 window로 pulse를 중간에 자르는 것보다 유리하다는 의미이다. Paired t-test 결과 이 차이는 통계적으로 유의했다($p < 0.01$).

Derivative 사용은 일반적인 성능 향상을 제공하지 않았다. Raw PPG에 1차 및 2차 derivative를 추가한 const_HR_derivative와 const_time_derivative는 univariate PPG input보다 일관되게 더 좋지 않았다. AlexNet의 SBP MAE에서만 약간의 개선이 있었지만, 전체적인 결론으로는 derivative는 필수적이지 않았다.

Window length에 대해서는 PPG 기반 prediction error가 segment length에 따라 뚜렷하게 변하지 않았다. 저자들은 긴 segment에서 pulse morphology variability가 커져 error가 증가할 것으로 예상했지만, 세 번의 반복 실험에서는 거의 비슷한 error가 나타났다. 그러나 계산 비용 때문에 충분히 많은 반복 실험을 하지 못했으므로, 이 결과에 대한 강한 통계적 결론은 제시하지 않았다.

rPPG에서는 SNR과 sample 수의 trade-off가 중요했다. 너무 짧은 segment는 SNR 계산이 부정확하고, 너무 긴 segment는 usable sample 수를 줄인다. SNR threshold -7 dB를 기준으로 분석한 결과, 저자들은 이후 실험에서 7초 segment length를 사용했다.

### 4.2 MIMIC-B 데이터셋의 sample imbalance 분석

저자들은 MIMIC-B의 4,000명 subject에서 약 170 million sample pool을 구성했다. Figure 4의 histogram은 subject별 sample 수가 몇백 개에서 500,000개 이상까지 매우 넓게 분포함을 보여준다. 특정 subject가 지나치게 많은 sample을 제공하면, 특히 mixed split에서 그 subject의 데이터가 training, validation, test에 동시에 들어갈 수 있고, 모델 성능이 과대평가될 수 있다.

이를 방지하기 위해 각 subject의 contribution을 2,000 sample로 제한했다. 이는 subject imbalance를 줄이는 중요한 preprocessing decision이다. Figure 5는 mixed dataset과 non-mixed dataset의 SBP/DBP distribution이 매우 유사하고 동일한 혈압 범위를 포괄함을 보여준다. 즉 두 dataset은 혈압 분포 측면에서는 comparable하지만, subject separation 여부만 다르다. 따라서 mixed와 non-mixed 성능 차이는 주로 subject leakage 여부에서 비롯된다고 해석할 수 있다.

### 4.3 PPG 기반 BP prediction: mixed vs non-mixed

Table 1과 Figure 6은 mixed와 non-mixed dataset에서 neural architecture들의 전체 MAE를 보여준다. 가장 중요한 결과는 mixed dataset에서 모든 neural network의 MAE가 크게 낮아진다는 것이다.

Non-mixed dataset에서 각 모델의 SBP MAE는 AlexNet 16.6 mmHg, ResNet 16.4 mmHg, Slapničar model 16.8 mmHg, LSTM 16.4 mmHg이다. DBP MAE는 AlexNet 8.7 mmHg, ResNet 8.5 mmHg, Slapničar model 8.8 mmHg, LSTM 8.6 mmHg이다. Mean regressor는 SBP 19.6 mmHg, DBP 9.8 mmHg이다. Neural network들은 mean regressor보다 통계적으로 낮은 MAE를 보였지만, 실용적으로는 개선 폭이 크지 않다.

Mixed dataset에서는 성능이 훨씬 좋아진다. ResNet은 SBP MAE 7.7 mmHg, DBP MAE 4.4 mmHg로 가장 좋은 성능을 기록했다. AlexNet은 SBP 8.8 mmHg, DBP 4.9 mmHg이고, Slapničar model은 SBP 12.9 mmHg, DBP 7.5 mmHg, LSTM은 SBP 11.6 mmHg, DBP 6.7 mmHg이다. Mean regressor는 mixed에서도 SBP 19.6 mmHg, DBP 9.9 mmHg로 큰 변화가 없다.

이 결과는 subject-aware split의 중요성을 강하게 보여준다. Mixed setting에서는 모델이 동일 subject의 다른 sample을 test에서 보게 될 수 있으므로, subject-specific morphology를 사실상 기억하거나 활용할 수 있다. 반면 non-mixed setting에서는 완전히 새로운 subject에 일반화해야 하므로 error가 크게 증가한다. 실제 clinical application은 non-mixed setting에 가깝기 때문에, mixed result를 근거로 모델의 임상 적용 가능성을 주장하는 것은 위험하다.

### 4.4 혈압 bin별 MAE 분석

Figure 7은 혈압 범위를 10 mmHg bin으로 나누어 MAE를 계산한 결과를 보여준다. Non-mixed dataset에서는 error가 혈압 bin에 따라 크게 달라진다. 모든 architecture에서 SBP 100–130 mmHg, DBP 50–70 mmHg 근처에서 가장 낮은 error를 보였다. 이 구간은 training set에서 sample 수가 가장 많은 구간이다.

반대로 혈압 분포의 양끝, 즉 낮은 SBP/DBP 또는 높은 SBP/DBP 구간에서는 MAE가 크게 증가한다. 이는 neural network가 전체 loss를 줄이기 위해 sample이 많은 혈압 범위에 예측을 집중하는 경향을 보인다는 의미이다. 즉 모델은 PPG morphology로부터 전체 혈압 범위에 대해 균등하게 강건한 mapping을 학습했다기보다, training distribution의 mode 주변을 잘 맞추도록 최적화된 것이다.

Mixed dataset에서는 이러한 BP range dependence가 훨씬 약해지고, 특히 혈압 분포의 tail에서 MAE가 크게 줄어든다. 그러나 저자들은 이를 실제 성능 향상으로 해석하지 않는다. 오히려 train/test set에 동일 subject의 sample이 섞이면 tail 구간에서도 subject-specific pattern을 활용할 수 있어 성능이 과대평가된다고 해석한다.

임상적으로 이 결과는 매우 중요하다. 혈압 예측 장치는 정상 혈압 범위뿐 아니라 저혈압 및 고혈압처럼 intervention이 필요한 구간에서 정확해야 한다. 그러나 본 논문의 non-mixed 결과는 바로 이 중요한 구간에서 error가 커진다는 점을 보여준다. 저자들은 이러한 이유로 평균 MAE만 보고하는 것은 clinical relevance를 판단하기에 충분하지 않다고 주장한다.

### 4.5 AAMI 및 BHS 기준과의 관계

저자들은 자신들의 mean performance를 기존 연구와 비교하기 위해 전체 평균 MAE도 평가했다. 그러나 논문은 어떤 non-mixed PPG 기반 결과도 관련 BHS 및 AAMI 기준을 만족하지 못했다고 설명한다. 원문은 관련 기준에서 혈압 측정 장치가 허용 가능한 error, 예를 들어 $BP < 10$ mmHg 오차를 85% 이상 확률로 제공해야 한다고 언급한다.

특히 저혈압 및 고혈압 범위의 높은 MAE는 clinical application에서 치명적이다. 실제 병원에서는 정상 범위보다 hypo- 및 hypertensive range에서의 빠르고 정확한 detection이 더 중요하기 때문이다. 따라서 모델이 정상 범위에서만 낮은 error를 보이고 극단 범위에서 큰 error를 보인다면, 전체 평균 MAE가 어느 정도 낮더라도 임상적으로 부적합할 수 있다.

저자들은 subject-specific split을 고려한 Slapničar et al.의 결과와 자신들의 결과가 일치한다고 설명한다. 반면 subject-based split을 명확히 하지 않은 일부 기존 연구들이 훨씬 낮은 error를 보고했는데, 본 논문의 mixed/non-mixed 비교는 이러한 낮은 error가 method 자체의 우수성보다 train/test split violation에서 비롯되었을 가능성을 제기한다.

### 4.6 PPG 기반 personalization 결과

PPG personalization은 non-mixed test set의 20명 subject를 대상으로 수행되었다. Table 1에 따르면 personalization 전 pre-personalization 상태에서 SBP MAE는 AlexNet 15.8 mmHg, ResNet 16.2 mmHg, Slapničar model 15.2 mmHg, LSTM 15.7 mmHg이다. DBP MAE는 각각 10.1, 9.8, 9.8, 9.9 mmHg이다.

Random 20% personalization을 적용하면 성능이 크게 좋아진다. SBP MAE는 AlexNet 11.8 mmHg, ResNet 13.0 mmHg, Slapničar model 10.8 mmHg, LSTM 8.5 mmHg로 감소한다. DBP MAE는 각각 6.0, 6.3, 5.8, 4.5 mmHg로 감소한다.

First 20% personalization도 비슷한 개선을 보인다. SBP MAE는 AlexNet 12.2 mmHg, ResNet 12.3 mmHg, Slapničar model 11.1 mmHg, LSTM 9.0 mmHg이고, DBP MAE는 각각 6.1, 5.8, 5.9, 4.6 mmHg이다.

이 결과는 personalization이 PPG 기반 BP prediction에서 상당히 효과적임을 보여준다. 특히 LSTM은 personalization 후 가장 낮은 SBP 및 DBP MAE를 보였다. 그러나 personalization은 test subject의 일부 데이터를 사용한다는 점에서 완전한 calibration-free prediction은 아니다. 실제 사용에서는 초기 cuff-based measurement 또는 subject-specific calibration이 필요할 수 있다.

흥미롭게도 random 20%와 first 20% strategy 사이의 차이는 크지 않았다. 이는 임상적으로 긍정적인 결과이다. 실제 사용에서는 미래 데이터를 무작위로 사용할 수 없기 때문에, 환자가 입원하거나 device를 착용한 초기 구간의 data를 calibration에 사용하는 first 20% scenario가 더 현실적이다. 이 strategy가 random strategy와 유사한 성능을 보인다는 것은 personalization의 practical feasibility를 뒷받침한다.

### 4.7 rPPG 기반 prediction 결과

rPPG 실험은 PPG로 pre-trained된 network를 camera-derived rPPG signal에 transfer하는 방식으로 진행되었다. Table 2는 rPPG 기반 MAE를 fine-tuning 전후로 보여준다.

Fine-tuning 전, 즉 PPG로 학습된 모델을 rPPG에 바로 적용했을 때 error는 매우 컸다. SBP MAE는 AlexNet 28.1 mmHg, ResNet 28.9 mmHg, Slapničar model 29.6 mmHg, LSTM 33.5 mmHg였다. DBP MAE는 각각 13.8, 13.3, 11.5, 12.4 mmHg였다. 이는 PPG와 rPPG가 유사한 pulse waveform을 공유하더라도, 직접적인 domain transfer는 어렵다는 것을 보여준다.

rPPG data로 final layer를 fine-tuning하면 SBP MAE가 크게 감소한다. Fine-tuning 후 without personalization에서 SBP MAE는 AlexNet 14.0 mmHg, ResNet 14.1 mmHg, Slapničar model 14.8 mmHg, LSTM 13.6 mmHg이다. DBP MAE는 각각 11.0, 11.2, 10.3, 10.3 mmHg이다. 즉 rPPG fine-tuning은 특히 SBP error를 크게 줄였지만, DBP error 개선은 상대적으로 작았다.

Personalization을 적용하면 일부 architecture에서 추가 개선이 나타났다. First 20% personalization에서 ResNet의 SBP MAE는 12.7 mmHg로 fine-tuning only의 14.1 mmHg보다 낮아졌다. DBP MAE는 10.8 mmHg였다. 그러나 모든 architecture에서 일관된 큰 개선이 나타난 것은 아니다. 예를 들어 AlexNet은 first 20% personalization 후 SBP MAE가 14.2 mmHg로 fine-tuning only 14.0 mmHg보다 오히려 약간 커졌다. Slapničar model과 LSTM도 SBP 기준으로는 personalization 후 약간 악화된 값이 보고된다.

Table 2에서 random 20% personalization 결과가 fine-tuning only와 동일하게 제시되어 있는 점은 원문 표 기준으로는 다소 해석이 필요하다. 텍스트에서는 personalization strategy 간 통계적 차이를 분석했다고 설명하지만, 표에서는 random 20% 결과가 without personalization과 동일한 수치로 나타난다. 보고서에서는 원문 표에 근거해 수치를 그대로 해석하되, random strategy가 추가 개선을 명확히 보여주지는 않는다고 정리하는 것이 타당하다.

Figure 9는 rPPG fine-tuning이 전반적으로 MAE를 유의하게 줄였음을 보여준다. 그러나 subject별 결과를 보면 일부 subject는 fine-tuning 후 개선이 뚜렷했지만, 4명의 subject는 대부분 architecture에서 개선이 없었다. 이는 rPPG 기반 BP prediction이 subject와 recording condition에 매우 민감함을 의미한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PPG/rPPG 기반 혈압 예측 연구의 평가 bias를 매우 체계적으로 보여준다는 점이다. 저자들은 단순히 평균 MAE를 보고하지 않고, mixed와 non-mixed dataset을 직접 비교하여 subject leakage가 성능을 얼마나 과대평가할 수 있는지 실험적으로 입증했다. ResNet의 경우 SBP MAE가 non-mixed 16.4 mmHg에서 mixed 7.7 mmHg로 크게 낮아지는 결과는, sample-level random split이 얼마나 위험한지를 명확히 보여준다.

두 번째 강점은 혈압 bin별 error analysis이다. 전체 평균 MAE는 정상 혈압 범위의 많은 sample에 의해 지배될 수 있다. 반면 bin-wise MAE는 임상적으로 중요한 저혈압 및 고혈압 구간에서 모델이 어떻게 동작하는지 보여준다. 본 논문은 sample이 적은 혈압 구간에서 error가 크게 증가한다는 점을 제시함으로써, 기존 연구에서 보고된 낮은 평균 오차의 임상적 의미를 재검토하게 만든다.

세 번째 강점은 mean regressor baseline을 사용했다는 것이다. Mean regressor는 단순하지만 강력한 sanity check 역할을 한다. 복잡한 neural network가 training set 평균을 출력하는 모델보다 실질적으로 많이 개선되지 않는다면, 해당 neural network가 생리학적으로 의미 있는 waveform-to-BP mapping을 충분히 학습했다고 보기 어렵다. 이 baseline 비교는 논문의 비판적 설득력을 높인다.

네 번째 강점은 personalization을 PPG와 rPPG 모두에서 평가했다는 점이다. 기존 많은 연구가 calibration-free 성능을 강조하는 반면, 이 논문은 subject-specific morphology 차이를 고려할 때 personalization이 현실적인 개선 방향일 수 있음을 보여준다. 특히 first 20% calibration strategy는 실제 사용 scenario와 어느 정도 맞닿아 있다.

다섯 번째 강점은 rPPG를 단순 가능성 차원이 아니라 실제 clinical video dataset으로 평가했다는 점이다. 병원 환경에서는 lighting variation, movement, ROI instability가 흔히 발생한다. 저자들은 rPPG signal 일부가 artifact로 인해 사용할 수 없었다고 명시하며, camera-based BP prediction의 현실적 어려움을 솔직하게 다룬다.

하지만 한계도 분명하다. 첫째, 저자들은 state-of-the-art 성능을 달성하기 위한 exhaustive hyperparameter tuning을 수행하지 않았다. 이는 논문의 목적이 평가 방법론 분석에 있기 때문이지만, 결과적으로 사용된 architecture들의 성능이 PPG/rPPG BP prediction의 최적 성능을 의미하지는 않는다. 더 정교한 domain-invariant architecture, subject adaptation method, attention mechanism, self-supervised pretraining이 적용되면 성능이 개선될 가능성은 남아 있다.

둘째, MIMIC 데이터의 subject metadata가 제한적이다. 나이, 성별, 질환, 약물, 혈관 상태, measurement equipment 같은 정보가 충분하지 않으면, 왜 특정 subject나 혈압 bin에서 error가 커지는지 분석하기 어렵다. 저자들도 age group별 training이 성능을 높일 수 있다고 언급하지만, MIMIC-III에서는 해당 정보가 충분히 제공되지 않았다고 설명한다.

셋째, rPPG 데이터의 규모가 작다. 50명 subject가 등록되었지만, motion artifact와 lighting 문제로 상당수가 제외되었고, 최종 fine-tuning에는 17명 수준의 subject가 사용되었다. 이 규모로는 rPPG 기반 BP prediction의 일반화 가능성을 확정하기 어렵다. 특히 final layer만 fine-tuning했기 때문에, 더 많은 rPPG 데이터가 있을 때 deeper layer fine-tuning이나 end-to-end training이 어떤 결과를 낼지는 검증되지 않았다.

넷째, rPPG ground truth BP의 temporal resolution이 1분 단위이다. rPPG signal은 연속 video에서 window 단위로 추출되지만, reference BP는 bedside monitor에서 1분 간격으로 얻는다. 혈압이 시간에 따라 변할 수 있다는 점을 고려하면, label noise 또는 temporal misalignment가 발생할 수 있다. 원문은 이 문제를 깊게 분석하지 않는다.

다섯째, rPPG ROI selection이 수동으로 이루어졌다. Forehead와 cheeks ROI를 수동 labeling했기 때문에, 실제 자동 시스템에서는 ROI tracking, 얼굴 움직임, illumination 변화에 따른 추가 오류가 발생할 수 있다. 저자들은 향후 PhysNet 같은 end-to-end rPPG extraction network를 사용하면 ROI placement 문제를 줄일 수 있다고 제안한다.

여섯째, 본 논문은 PPG morphology와 sensor contact pressure의 관계가 중요한 문제라고 지적하지만, 이를 직접 실험하지는 않는다. Sensor contact pressure는 PPG waveform shape에 큰 영향을 주며, cuffless BP estimation에서 매우 중요한 confounder가 될 수 있다. 이 문제는 향후 실험적으로 더 검증되어야 한다.

## 6. 결론

이 논문은 PPG 및 rPPG 기반 deep learning 혈압 예측 연구에 대해 매우 중요한 평가 기준을 제시한다. 저자들은 AlexNet, ResNet, Slapničar et al. model, LSTM이라는 네 가지 neural architecture를 사용하여 PPG 기반 SBP/DBP prediction을 평가하고, rPPG 기반 transfer learning 및 personalization까지 분석했다.

핵심 결론은 세 가지로 요약할 수 있다. 첫째, PPG segment는 complete heartbeat cycle을 포함하도록 구성하는 것이 고정 시간 window보다 유리하며, derivative 추가는 일관된 성능 향상을 제공하지 않았다. 둘째, subject-aware split을 적용하지 않고 sample-level random split을 사용하면 성능이 크게 과대평가된다. Mixed dataset에서는 MAE가 크게 낮아졌지만, 이는 동일 subject의 data가 train/test에 동시에 포함되는 leakage 효과로 해석된다. 실제 신규 환자 예측에 가까운 non-mixed setting에서는 MAE가 훨씬 커지고, 특히 저혈압 및 고혈압 구간에서 error가 크게 증가했다. 셋째, personalization은 PPG와 rPPG 모두에서 error를 줄일 수 있지만, 이는 calibration-free prediction이 아니라 subject-specific calibration이 필요하다는 뜻이기도 하다.

이 연구는 PPG-only 또는 rPPG-only morphology 기반 BP prediction이 현재 단계에서 임상 적용에 필요한 강한 요구사항을 만족하기 어렵다고 결론짓는다. 특히 rPPG 기반 camera-only blood pressure estimation은 기술적으로 매력적이지만, signal quality, motion artifact, lighting variation, subject-specific morphology 문제 때문에 더욱 어렵다.

논문의 가장 중요한 기여는 “모델이 낮은 평균 MAE를 보였는가”보다 “그 MAE가 어떤 데이터 분할과 어떤 혈압 범위에서 얻어진 것인가”가 훨씬 중요하다는 점을 명확히 보여준 것이다. 향후 연구는 subject-independent train/test split, full BP range bin-wise evaluation, balanced dataset construction, demographic metadata 활용, sensor contact pressure 통제, personalization protocol, rPPG artifact robustness를 반드시 고려해야 한다.

종합하면, 이 논문은 cuffless BP estimation 분야에서 deep learning model의 성능을 더 엄격하고 임상적으로 타당하게 평가해야 한다는 강력한 메시지를 제공한다. PPG와 rPPG 기반 혈압 예측은 여전히 중요한 연구 방향이지만, 현재의 morphology-only approach만으로는 real-world clinical deployment를 보장하기 어렵고, 더 큰 데이터셋, 더 정교한 subject adaptation, domain-invariant feature learning, 그리고 실환경 검증이 필요하다.
