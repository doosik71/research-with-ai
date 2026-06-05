# Photoplethysmogram (PPG)-Based Blood Pressure Estimation using Vision Transformer Networks: A Deep Learning Approach

* **저자**: Sri Harsha Darbhasayanam, Noorjahan, Ashoka Reddy Komalla, Hassaan Ali Areeb Mohammad
* **발표연도**: 2024

## 1. 논문 개요

이 논문은 Photoplethysmography, 즉 PPG 신호를 이용하여 blood pressure, BP를 추정하기 위해 Vision Transformer, ViT를 적용한 연구이다. 기존의 PPG 기반 혈압 추정은 Pulse Transit Time, PTT와 같이 PPG와 Electrocardiography, ECG를 함께 사용하는 방식에서 출발했지만, 최근에는 착용 편의성과 시스템 단순화를 위해 PPG 단일 신호만으로 SBP와 DBP를 추정하려는 방향으로 발전하고 있다. 이 논문은 이러한 흐름 속에서 자연 이미지 분류 분야에서 발전한 pre-trained Vision Transformer를 PPG 기반 혈압 추정에 적용하고, 같은 조건에서 pre-trained ResNet50과 비교한다.

논문의 연구 문제는 PPG 신호를 image-like representation으로 변환했을 때, ViT가 PPG의 시간적·주파수적 패턴을 학습하여 systolic blood pressure, SBP와 diastolic blood pressure, DBP를 예측할 수 있는가이다. 저자들은 PPG 신호를 두 가지 방식으로 이미지화한다. 첫 번째는 PPG segment를 그대로 time-domain waveform plot으로 저장하는 방식이고, 두 번째는 Continuous Wavelet Transform, CWT를 이용해 time-frequency scalogram으로 변환하는 방식이다. 이후 이 이미지들을 pre-trained ViT와 pre-trained ResNet50에 입력하여 fine-tuning을 수행한다.

이 문제는 임상 및 wearable healthcare 관점에서 중요하다. 혈압은 cardiovascular health를 평가하고 진단 및 치료 결정을 내리는 데 핵심적인 생체 지표이다. 전통적인 청진 기반 혈압 측정은 Korotkoff sound를 식별해야 하므로 훈련과 표준화된 절차가 필요하고, oscillometric device는 편리하지만 장치별 알고리즘 차이와 신뢰성 문제가 있다. 또한 cuff 기반 방식은 지속 측정이 어렵고, white-coat syndrome이나 신체 활동 중 측정 어려움 같은 한계가 있다. PPG는 optical radiation을 이용해 말초 조직의 blood volume 변화를 측정하므로 cuff 없이 연속 모니터링이 가능하다는 장점이 있다.

논문은 PPG 기반 BP estimation에서 traditional machine learning이 manual feature engineering에 의존하고, CNN이나 LSTM 같은 deep learning 모델이 PPG 패턴 학습에 활용되어 왔다는 점을 설명한다. 여기에 더해 transformer architecture는 self-attention을 통해 장거리 의존성을 학습할 수 있으므로, PPG 신호의 복잡한 temporal dependency를 포착하는 데 잠재력이 있다고 본다. 다만 이 논문은 PPG를 직접 1차원 sequence로 transformer에 입력하는 것이 아니라, PPG를 이미지로 변환한 뒤 image pre-trained ViT를 fine-tuning한다는 특징이 있다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 PPG 신호를 시각적 표현으로 변환한 뒤, 자연 이미지에서 사전학습된 Vision Transformer를 transfer learning 방식으로 fine-tuning하여 혈압을 회귀 예측하는 것이다. PPG는 본질적으로 1차원 시간 신호이지만, time-domain plot이나 CWT scalogram으로 변환하면 2차원 이미지 형태가 된다. ViT는 이미지를 patch 단위 token sequence로 보고 self-attention을 수행하는 모델이므로, 이러한 변환을 통해 biomedical signal analysis 문제를 image-based regression 문제로 다룰 수 있다.

기존 traditional ML 방식은 PPG waveform에서 peak, valley, pulse width, amplitude, rising time, falling time, frequency-domain feature 등을 수작업으로 추출해야 한다. 이러한 방식은 feature selection과 preprocessing에 많은 전문 지식이 필요하고, 신호 품질이 낮거나 noise가 있는 경우 성능이 떨어질 수 있다. CNN, LSTM, ResNet 같은 deep learning 모델은 raw 또는 전처리된 신호에서 feature를 자동으로 학습할 수 있지만, CNN은 주로 local pattern에 강하고, LSTM은 sequence dependency를 다루지만 장기 의존성 학습과 학습 안정성 측면에서 한계가 있을 수 있다.

ViT의 차별점은 self-attention을 통해 모든 patch 또는 token 사이의 관계를 직접 모델링한다는 점이다. 이미지의 경우 이는 멀리 떨어진 영역 사이의 관계를 학습하는 데 유리하다. PPG scalogram의 경우 시간축과 주파수축에서 서로 다른 위치의 패턴이 혈압과 관련될 수 있으므로, self-attention이 유용할 수 있다. 예를 들어 특정 시간 구간의 pulse morphology, 특정 frequency band의 에너지 변화, pulse 간 반복 구조 등이 함께 고려될 수 있다.

또 다른 핵심은 pre-trained model을 활용한 fine-tuning이다. 논문에서 사용한 ViT는 `google/ViT-base-patch16-224-in21k`로, 21,000개 class의 자연 이미지 데이터에서 사전학습된 모델이다. 혈압 추정용 PPG 데이터는 일반적으로 자연 이미지 데이터보다 훨씬 작기 때문에, 처음부터 ViT를 학습하면 overfitting 위험이 크고 많은 데이터가 필요하다. 저자들은 pre-trained ViT의 일반적인 visual representation 능력을 활용하고, PPG 이미지와 SBP, DBP label에 맞게 fine-tuning함으로써 제한된 biomedical dataset에서도 학습이 가능하다고 본다.

논문은 또한 CWT scalogram과 time-domain waveform plot이라는 두 입력 표현을 비교한다. CWT는 PPG 신호의 시간적 변화와 주파수 성분을 동시에 표현하므로, 단순한 waveform plot보다 더 풍부한 정보를 제공할 가능성이 있다. 실험 결과에서도 CWT scalogram이 time-domain waveform보다 약간 더 좋은 성능을 보인다. 이는 PPG 기반 혈압 추정에서 time-frequency representation이 의미 있는 feature representation이 될 수 있음을 시사한다.

다만 논문에서 제안한 방식은 PPG 신호 자체를 위한 custom transformer architecture라기보다는, PPG를 이미지로 바꾸어 기존 image pre-trained ViT를 적용하는 transfer learning 접근이다. 따라서 이 연구의 기여는 “ViT를 PPG 신호에 직접 맞춘 새로운 구조 설계”보다는 “PPG image representation과 pre-trained ViT fine-tuning의 적용 가능성 평가”에 가깝다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 데이터 획득, PPG 신호 전처리, PPG의 이미지 변환, Hugging Face dataset 구성, pre-trained ViT 및 ResNet50 fine-tuning, SBP와 DBP 예측 성능 평가로 구성된다.

데이터는 MIMIC-III Waveform Database Matched Subset에서 가져온 11,787개 segment로 구성된다. 각 segment는 arterial blood pressure, ABP 신호와 이에 대응하는 finger PPG 신호를 포함한다. 데이터는 `.mat` 파일 형식으로 저장되어 있으며, 각 파일에는 raw PPG structure와 그에 대응하는 computed characteristics가 포함된다. 각 segment는 15초 길이이고, sampling frequency는 125 Hz이다. 따라서 각 PPG segment는 이론적으로 $15 \times 125 = 1875$개의 sample point를 가진다. 논문은 각 PPG segment와 그에 대응하는 mean SBP 및 mean DBP 값을 Python으로 추출했다고 설명한다.

전처리 단계에서는 먼저 band-pass Butterworth filter를 적용한다. Cut-off frequency는 0.5 Hz와 45 Hz이다. 이 필터는 PPG 신호에서 너무 낮은 주파수의 baseline drift와 너무 높은 주파수의 noise를 제거하기 위한 것이다. 이후 Savitzky–Golay filter를 적용한다. 이때 window size는 7이고 polynomial degree는 3이다. Savitzky–Golay filter는 신호의 local polynomial fitting을 통해 smoothing을 수행하므로, PPG waveform의 형태를 어느 정도 유지하면서 noise를 줄이는 데 사용될 수 있다.

전처리된 PPG 신호는 두 가지 방식으로 시각화된다. 첫 번째 방식은 time-domain representation이다. PPG segment data point를 Matplotlib으로 직접 plot하고, sampling rate 125 Hz를 기준으로 time index를 구성한다. 이렇게 생성된 waveform plot은 이미지로 저장된다. 이 방법은 PPG의 pulse shape, amplitude variation, rising phase, falling phase 같은 형태적 정보를 이미지 형태로 전달한다.

두 번째 방식은 Continuous Wavelet Transform, CWT를 이용한 scalogram 생성이다. CWT는 시간에 따른 frequency content 변화를 표현하는 time-frequency transform이다. PPG 신호는 시간에 따라 pulse morphology와 frequency content가 변할 수 있으므로, CWT scalogram은 단순 time-domain plot보다 더 풍부한 정보를 포함할 가능성이 있다. 논문은 CWT-generated scalogram을 이미지로 저장하여 모델 입력으로 사용한다. 다만 제공된 추출 텍스트에서는 CWT에 사용한 wavelet 종류, scale range, normalization 방식은 명시되어 있지 않다. 따라서 scalogram 생성의 세부 재현성에는 추가 정보가 필요하다.

각 이미지와 이에 대응하는 mean SBP, mean DBP 값을 연결하기 위해 CSV 파일이 생성된다. CSV에는 이미지 경로와 해당 이미지의 SBP, DBP 값이 포함된다. 이후 Hugging Face builder script를 사용하여 두 종류의 dataset을 만든다. 하나는 time-domain waveform image dataset이고, 다른 하나는 CWT scalogram image dataset이다. 각 dataset은 image, mean SBP, mean DBP 세 feature를 포함하며, Apache Arrow format으로 구성된다. 이후 Google Colab 환경에서 dataset을 로드하고 train-test split을 수행한다. 다만 추출 텍스트에는 train-test split 비율, subject-wise split 여부, validation set 구성 여부가 명확히 제시되어 있지 않다.

모델의 중심은 pre-trained Vision Transformer이다. ViT는 입력 이미지를 고정 크기의 non-overlapping patch로 나누고, 각 patch를 linear projection하여 token embedding으로 변환한다. 논문에서 사용한 모델은 `google/ViT-base-patch16-224-in21k`이다. 이 모델은 224×224 RGB image를 기본 입력으로 사용하고, patch size는 16×16이다. 따라서 한 이미지는 $14 \times 14 = 196$개의 patch로 나뉘며, 각 patch는 token으로 변환된다. 각 token에는 patch 위치 정보를 제공하기 위해 positional encoding이 더해진다.

ViT encoder block은 self-attention과 feed-forward neural network, FFN으로 구성된다. Self-attention은 각 token이 다른 token들과 얼마나 관련 있는지 계산하여, 이미지 전체의 전역적인 관계를 반영한다. PPG scalogram의 경우 이는 특정 시간·주파수 위치의 pattern이 다른 위치의 pattern과 어떻게 연결되는지를 학습하는 방식으로 해석할 수 있다. FFN은 각 token representation에 position-wise transformation을 적용하고, GELU activation을 사용하여 비선형 feature를 학습한다. 각 sublayer에는 layer normalization과 residual connection이 적용되어 학습 안정성과 gradient flow를 개선한다.

Self-attention의 일반적인 구조는 query, key, value를 사용하여 각 token 사이의 attention weight를 계산하는 방식이다. 논문 텍스트에는 수식이 직접 제시되어 있지 않지만, ViT의 핵심 연산은 다음과 같은 scaled dot-product attention으로 이해할 수 있다.

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

여기서 $Q$는 query, $K$는 key, $V$는 value이고, $d_k$는 key vector의 dimension이다. 이 식은 각 token이 다른 token들을 얼마나 참고할지 계산하고, 그 가중합으로 새로운 representation을 만든다. 이 논문에서 중요한 점은 PPG 이미지의 특정 patch가 전체 waveform 또는 scalogram의 다른 patch와 관계를 맺으면서 혈압 예측에 필요한 feature를 구성할 수 있다는 것이다.

사용한 ViT의 세부 구조는 12 attention heads, hidden size 768, feed-forward intermediate size 3072, GELU activation을 포함한다. Dropout은 사용하지 않았다고 설명된다. Layer normalization의 epsilon 값은 $1e^{-12}$이고, query, key, value operation에 bias가 활성화되어 있다. 이 모델은 자연 이미지에서 사전학습되었지만, fine-tuning을 통해 PPG image regression에 맞게 조정된다.

Fine-tuning 과정에서는 Hugging Face library를 사용한다. ViT image processor를 통해 dataset image를 모델 입력 형식에 맞게 변환하고, collate function을 정의하여 batch processing을 수행한다. Optimizer로는 Adam을 사용한다. Adam은 gradient의 1차 moment와 2차 moment를 사용하여 parameter별 adaptive learning rate를 적용하는 optimization algorithm이다. 또한 ReduceLROnPlateau scheduler를 사용하여 validation performance가 정체될 때 learning rate를 동적으로 조정한다. Epoch 수, learning rate, batch size, input size 등 여러 hyperparameter를 바꾸어 실험했다고 설명하지만, 최종적으로 사용된 정확한 hyperparameter 값은 제공된 텍스트에 명확히 제시되어 있지 않다.

학습 목표는 SBP와 DBP를 예측하는 regression이다. 논문은 평가 지표로 Mean Squared Error, MSE와 Mean Absolute Error, MAE를 사용한다. MSE는 예측 오차를 제곱하여 평균한 값이다.

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

여기서 $y_i$는 실제 SBP 또는 DBP 값이고, $\hat{y}_i$는 모델 예측값이다. MSE는 큰 오차에 더 큰 penalty를 주기 때문에, 혈압 예측에서 큰 예측 실패를 줄이는 데 의미가 있다.

MAE는 절대 오차의 평균이다.

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

MAE는 평균적으로 몇 mmHg 정도 틀리는지를 직관적으로 보여준다. 혈압 추정 연구에서는 MAE가 가장 많이 보고되는 지표 중 하나이다.

비교 모델로는 pre-trained ResNet50이 사용된다. 구체적으로 Hugging Face의 `microsoft/resnet-50` 모델을 같은 dataset에 fine-tuning한다. ResNet50은 convolutional neural network 기반 모델이며, residual connection을 통해 깊은 network의 학습을 안정화한다. ViT와 ResNet50을 비교함으로써, transformer 기반 image representation이 CNN 기반 image representation보다 PPG 기반 혈압 추정에서 유리한지 확인하려는 목적이 있다.

## 4. 실험 및 결과

실험은 두 종류의 입력 표현과 두 종류의 모델을 비교하는 구조이다. 입력 표현은 CWT-generated scalogram과 PPG time-domain waveform image이고, 모델은 pre-trained ViT와 pre-trained ResNet50이다. 각 모델은 SBP와 DBP를 각각 예측하며, 성능은 MSE와 MAE로 보고된다.

먼저 ViT를 CWT-generated scalogram dataset에 fine-tuning한 결과, SBP 예측에서 MSE 411.7237, MAE 16.3227 mmHg를 기록했다. DBP 예측에서는 MSE 108.0452, MAE 8.0827 mmHg를 기록했다. 이 결과는 CWT scalogram이 PPG의 time-frequency information을 포함하고 있으며, ViT가 이를 이용해 혈압을 어느 정도 예측할 수 있음을 보여준다.

ViT를 PPG time-domain waveform dataset에 fine-tuning한 결과는 SBP MSE 476.9947, MAE 17.5984 mmHg이고, DBP MSE 108.6173, MAE 8.0907 mmHg이다. CWT scalogram과 비교하면 SBP 성능은 scalogram이 더 좋다. SBP MAE는 scalogram에서 16.3227 mmHg, waveform에서 17.5984 mmHg로 약 1.28 mmHg 차이가 난다. DBP는 scalogram 8.0827 mmHg, waveform 8.0907 mmHg로 거의 동일하다. 이는 CWT scalogram이 SBP 예측에는 조금 더 유용한 정보를 제공하지만, DBP 예측에서는 두 표현의 차이가 작다는 것을 의미한다.

논문은 ViT가 학습 초기에 빠르게 수렴하고 이후 안정적인 performance level을 유지한다고 설명한다. 이는 pre-trained model이 새로운 PPG image dataset에 빠르게 적응했음을 나타낸다. 또한 learning rate, batch size, input size 등 training parameter를 바꾸거나 여러 번 retraining했을 때 성능 변동이 대체로 ±0.5 범위에 머물렀다고 설명한다. 이 결과는 모델 학습이 비교적 안정적이라는 주장으로 이어진다. 다만 구체적인 반복 실험 횟수, 표준편차, confidence interval은 제공되지 않으므로 통계적 안정성은 제한적으로만 확인할 수 있다.

입력 이미지 resolution을 높이는 실험도 수행되었지만, 성능이 크게 향상되지는 않았다. 논문은 이 결과를 통해 단순히 더 높은 해상도를 사용하는 것이 혈압 예측 성능을 자동으로 높이지는 않는다고 해석한다. 이는 PPG image representation에서 중요한 것은 픽셀 수 자체보다, 혈압과 관련된 feature가 어떻게 표현되고 추출되는지라는 점을 시사한다.

ResNet50 비교 결과는 ViT보다 약간 더 좋다. ResNet50을 CWT-generated scalogram dataset에 fine-tuning한 경우, SBP MSE 352.6488, MAE 15.6113 mmHg를 기록했다. DBP는 MSE 89.1689, MAE 7.2730 mmHg이다. 이는 ViT scalogram 결과인 SBP MAE 16.3227, DBP MAE 8.0827보다 모두 낮은 오차이다. 따라서 이 실험에서는 ResNet50이 ViT보다 CWT scalogram 기반 혈압 추정에서 더 좋은 성능을 보인다.

ResNet50을 PPG time-domain waveform dataset에 fine-tuning한 경우, SBP MSE 435.4197, MAE 16.4068 mmHg이고, DBP MSE 104.0477, MAE 7.5323 mmHg이다. 이 역시 ViT waveform 결과인 SBP MAE 17.5984, DBP MAE 8.0907보다 좋다. 다만 논문은 두 모델의 성능이 비교적 comparable하다고 표현한다. 실제 수치상으로는 ResNet50이 모든 조건에서 더 낮은 MAE와 MSE를 보이므로, 이 실험만 놓고 보면 ViT가 ResNet50을 능가했다고 말하기는 어렵다.

두 입력 표현을 비교하면, ViT와 ResNet50 모두 CWT scalogram에서 time-domain waveform보다 좋은 성능을 보이는 경향이 있다. ResNet50의 경우 CWT scalogram은 SBP MAE 15.6113, DBP MAE 7.2730이고, waveform은 SBP MAE 16.4068, DBP MAE 7.5323이다. ViT에서도 같은 경향이 나타난다. 따라서 이 논문에서 가장 좋은 조합은 ResNet50 + CWT scalogram이며, 그 다음이 ViT + CWT scalogram이다.

실험 결과의 중요성은 두 가지로 볼 수 있다. 첫째, PPG를 image representation으로 변환한 뒤 pre-trained image model을 fine-tuning하는 접근이 어느 정도 작동한다는 점이다. ViT와 ResNet50 모두 SBP와 DBP를 예측할 수 있으며, 특히 CWT scalogram은 time-domain waveform보다 더 informative한 표현으로 보인다. 둘째, ViT가 적용 가능성은 보였지만 ResNet50보다 우수하다는 증거는 제한적이다. 논문은 ViT의 promise와 adaptability를 강조하지만, 정량 결과는 ResNet50이 약간 더 우수하다.

혈압 추정 정확도 관점에서 보면, 이 논문의 결과는 아직 clinical-grade 수준과는 거리가 있다. SBP MAE가 가장 좋은 경우에도 15.6113 mmHg이고, ViT의 SBP MAE는 16.3227 mmHg 또는 17.5984 mmHg이다. DBP MAE는 가장 좋은 경우 7.2730 mmHg이고, ViT는 약 8.08 mmHg이다. 일반적인 cuffless BP estimation에서 임상적 사용 가능성을 논의하려면 AAMI, BHS, ISO 등 표준 기준에 따른 mean error, standard deviation, error threshold 비율이 필요하지만, 이 논문은 MSE와 MAE만 제시한다. 따라서 임상 적용 가능성은 제한적으로만 평가할 수 있다.

## 5. 강점, 한계

이 논문의 강점은 Vision Transformer를 PPG 기반 혈압 추정에 적용하려는 시도 자체에 있다. ViT는 computer vision에서 발전한 모델이지만, 최근 biomedical image analysis와 ECG, EEG 등의 signal-derived image analysis에도 확장되고 있다. 이 논문은 PPG를 time-domain plot과 CWT scalogram으로 변환하여 ViT fine-tuning을 수행함으로써, transformer-based transfer learning이 PPG 기반 BP estimation에도 적용될 수 있음을 탐색한다.

두 번째 강점은 CWT scalogram과 time-domain waveform이라는 두 가지 입력 표현을 비교했다는 점이다. 단순 waveform plot은 PPG의 시간적 형태를 보여주지만, CWT scalogram은 시간과 주파수 정보를 동시에 제공한다. 실험 결과 CWT scalogram이 두 모델 모두에서 더 좋은 성능을 보여, PPG 기반 혈압 추정에서 time-frequency representation이 유용할 수 있음을 보여준다.

세 번째 강점은 ViT와 ResNet50을 같은 조건에서 비교했다는 점이다. 논문이 ViT를 제안하지만, CNN 기반 강력한 baseline인 ResNet50도 함께 fine-tuning하여 비교한다. 이 비교는 transformer가 기존 CNN 대비 어느 정도 경쟁력이 있는지 판단하는 데 도움이 된다. 비록 ResNet50이 수치상 더 좋은 성능을 보였지만, ViT가 유사한 수준의 성능을 보였다는 점은 향후 transformer 기반 모델 개선의 출발점이 될 수 있다.

네 번째 강점은 Hugging Face 기반 fine-tuning pipeline을 사용했다는 점이다. Dataset을 Apache Arrow format으로 구성하고, image processor와 pre-trained model을 활용하는 방식은 재사용성과 확장성이 높다. Biomedical signal을 image representation으로 변환해 image pre-trained model을 적용하는 workflow는 다른 생체신호 분석에도 확장될 수 있다.

그러나 한계도 매우 명확하다. 첫째, 모델 성능이 임상적으로 충분하지 않다. 가장 좋은 결과인 ResNet50 + CWT scalogram에서도 SBP MAE는 15.6113 mmHg이다. ViT의 SBP MAE는 16.3227 mmHg 또는 17.5984 mmHg이다. 이러한 오차는 혈압 진단이나 치료 의사결정에 사용하기에는 크다. 논문도 향후 model architecture refinement와 dataset expansion이 필요하다고 언급한다.

둘째, ViT가 ResNet50보다 우수하다는 실험적 근거는 없다. 논문은 ViT의 promise를 강조하지만, 제공된 표의 모든 조건에서 ResNet50이 ViT보다 낮은 MSE와 MAE를 기록한다. 따라서 이 논문의 결론은 “ViT가 ResNet50보다 뛰어나다”가 아니라, “ViT도 PPG image 기반 BP estimation에 적용 가능하지만, 현재 실험에서는 ResNet50이 약간 더 우수하다”로 정리하는 것이 정확하다.

셋째, PPG 신호를 이미지로 변환하는 방식이 정보 손실 또는 불필요한 복잡성을 유발할 수 있다. Time-domain plot은 원래 1차원 signal을 시각화한 이미지이므로, plot style, axis, line thickness, scaling, resolution 같은 요소가 모델 입력에 영향을 줄 수 있다. 이러한 시각화 요소는 생리학적 정보가 아니라 rendering artifact일 수 있다. CWT scalogram은 더 의미 있는 time-frequency representation이지만, wavelet 종류와 scale 설정에 따라 결과가 달라질 수 있다. 논문에서는 이러한 세부 설정이 충분히 설명되지 않는다.

넷째, 자연 이미지에 사전학습된 ViT가 PPG scalogram에 얼마나 잘 맞는지에 대한 분석이 부족하다. `google/ViT-base-patch16-224-in21k`는 자연 이미지 21,000개 class에 대해 학습된 모델이다. 자연 이미지의 texture, edge, object part representation이 PPG waveform이나 scalogram의 physiological pattern과 얼마나 잘 transfer되는지는 명확하지 않다. 논문은 fine-tuning으로 성능을 얻었지만, scratch training, self-supervised pretraining on PPG, signal-specific transformer와의 비교는 제공하지 않는다.

다섯째, 데이터 분할 방식이 불명확하다. 논문은 train-test split을 수행했다고만 설명하며, subject-wise split인지 여부를 명시하지 않는다. MIMIC-III 기반 PPG segment는 같은 환자에서 여러 segment가 나올 수 있다. 만약 동일 환자의 segment가 train set과 test set에 동시에 포함되면, 모델이 환자별 특징을 학습하여 실제 새로운 환자에 대한 일반화 성능보다 높은 결과를 보일 수 있다. 혈압 추정 연구에서 subject-independent validation은 매우 중요하므로, 이 부분은 큰 한계이다.

여섯째, target label이 mean SBP와 mean DBP라는 점도 주의해야 한다. 논문은 각 segment의 corresponding mean SBP와 DBP 값을 사용한다고 설명한다. 그러나 혈압은 beat-to-beat로 변할 수 있고, 15초 segment 안에서도 ABP waveform의 peak와 trough가 여러 번 존재한다. Mean SBP와 mean DBP를 어떻게 계산했는지, abnormal beat를 제거했는지, label noise가 어느 정도인지가 명확하지 않다.

일곱째, 평가 지표가 제한적이다. MSE와 MAE만 보고되며, mean error, standard deviation, Bland–Altman analysis, BHS 기준의 5/10/15 mmHg error threshold, AAMI 기준 만족 여부 등은 제공되지 않는다. 따라서 모델이 특정 방향으로 과대추정 또는 과소추정하는지, 오차 분포가 안정적인지, 임상 기준에 얼마나 근접한지는 판단하기 어렵다.

여덟째, 논문에서 fine-tuning hyperparameter 정보가 충분하지 않다. Epoch 수, batch size, learning rate, scheduler setting, train-test split ratio, image normalization, regression head 구조, loss function 설정 등이 명확히 제시되어야 재현성이 높아진다. 제공된 텍스트에서는 여러 trial을 수행했다고만 설명한다.

비판적으로 보면, 이 논문은 ViT를 PPG 기반 혈압 추정에 적용한 초기 탐색 연구로 의미가 있다. 하지만 현재 결과만으로 ViT가 이 문제에 특별히 적합하다고 결론내리기는 어렵다. 오히려 ResNet50이 더 좋은 성능을 보였고, 전반적인 MAE도 높다. 따라서 이 논문의 가치는 “최종 성능”보다는 “PPG signal-to-image representation과 pre-trained vision model fine-tuning의 가능성을 실험적으로 확인한 것”에 있다.

## 6. 결론

이 논문은 PPG 기반 blood pressure estimation을 위해 pre-trained Vision Transformer를 fine-tuning하는 방법을 제안하고, CWT scalogram과 time-domain waveform image라는 두 가지 PPG image representation을 비교했다. 또한 pre-trained ResNet50을 같은 조건에서 fine-tuning하여 ViT와 성능을 비교했다. 데이터는 MIMIC-III Waveform Database Matched Subset에서 가져온 11,787개의 15초 PPG segment와 대응하는 mean SBP, DBP label로 구성된다.

실험 결과, ViT는 CWT scalogram에서 SBP MAE 16.3227 mmHg, DBP MAE 8.0827 mmHg를 기록했고, time-domain waveform에서는 SBP MAE 17.5984 mmHg, DBP MAE 8.0907 mmHg를 기록했다. ResNet50은 CWT scalogram에서 SBP MAE 15.6113 mmHg, DBP MAE 7.2730 mmHg를 기록하여 가장 좋은 성능을 보였고, time-domain waveform에서도 ViT보다 낮은 오차를 보였다. 따라서 CWT scalogram이 waveform image보다 더 유용한 입력 표현으로 보이며, ResNet50은 본 실험 조건에서 ViT보다 약간 더 우수했다.

이 연구의 주요 기여는 ViT 기반 transfer learning을 PPG-derived image에 적용하여 BP estimation 가능성을 탐색했다는 점이다. 또한 time-frequency representation인 CWT scalogram이 PPG 혈압 추정에서 유용할 수 있음을 보여준다. 그러나 현재 성능은 clinical-grade BP estimation에는 부족하며, ViT가 ResNet50을 능가하지도 못했다. 따라서 본 연구는 완성된 혈압 추정 시스템이라기보다, transformer-based biomedical signal analysis의 초기 적용 사례로 해석하는 것이 적절하다.

향후 연구에서는 subject-wise validation, 더 큰 dataset, PPG-specific self-supervised pretraining, 1D signal transformer와의 비교, CWT parameter ablation, advanced data augmentation, clinical standard 기반 평가가 필요하다. 또한 PPG를 단순 plot image로 변환하는 방식보다, raw PPG sequence나 scalogram tensor를 생리학적 의미를 보존하도록 입력하는 architecture가 더 적합할 수 있다. 이러한 보완이 이루어진다면 ViT와 transformer 계열 모델은 PPG 기반 cuffless BP estimation에서 더 의미 있는 역할을 할 가능성이 있다.
