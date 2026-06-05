# Deep-Learning-Based Blood Pressure Estimation Using Multi-Channel Photoplethysmogram and Finger Pressure with Attention Mechanism

* **저자**: Jehyun Kyung, Joon-Young Yang, Jeong-Hwan Choi, Joon-Hyuk Chang, Sangkon Bae, Jinwoo Choi, Younho Kim
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 multi-channel photoplethysmogram, 즉 multi-channel PPG와 finger pressure signal을 이용하여 cuffless blood pressure, BP를 추정하는 deep-learning-based system을 제안한다. 기존 PPG 기반 cuffless BP estimation은 일반적으로 단일 PPG channel을 사용하기 때문에 손가락이 sensor 위에 놓이는 위치, 손가락 혈관 구조, sensor contact 상태에 따라 성능이 크게 달라질 수 있다. 본 연구는 이러한 문제를 줄이기 위해 넓은 field of view, FOV를 갖는 multi-channel PPG sensor를 직접 개발하고, 각 channel의 중요도를 attention mechanism으로 학습하여 SBP와 DBP를 예측한다.

연구 문제는 finger pressure 방식에서 PPG 측정 위치 변화가 혈압 추정 성능에 미치는 영향을 줄이는 것이다. Finger pressure 방식은 사용자가 손가락으로 PPG sensor를 점진적으로 누를 때 혈관 압박에 따라 PPG envelope가 변하는 현상을 이용한다. 이는 cuff-based oscillometric method에서 cuff pressure와 oscillometric waveform, OMW를 이용하는 방식과 유사한 생리적 신호를 얻는다는 장점이 있다. 그러나 기존 smartphone-based 또는 single-channel PPG 기반 finger pressing 방식은 PPG sensor가 혈관을 잘 포함하는 optical path를 형성하지 못하면 예측 정확도가 크게 떨어질 수 있다.

이 문제는 실제 mobile 또는 wearable device 적용에서 중요하다. 사용자가 매번 손가락을 정확히 같은 위치와 같은 압력 패턴으로 sensor에 올려놓기 어렵기 때문이다. 특히 손가락에는 양쪽 측면과 손톱 끝 주변에 작은 동맥들이 복잡하게 분포하므로, 단일 PPG sensor만으로 항상 혈관을 잘 통과하는 optical path를 만들기 어렵다. 따라서 본 논문은 9-channel PPG sensor를 통해 여러 위치의 PPG 신호를 동시에 측정하고, neural network가 각 subject 또는 각 sample에서 어떤 channel이 더 유용한지 학습하도록 설계한다.

논문에서 제안한 multi-channel system은 최종적으로 SBP에서 ME ± STD 0.43 ± 9.35 mmHg, DBP에서 0.21 ± 7.72 mmHg를 달성했다. 또한 single-channel best model과 비교했을 때 SBP standard deviation은 9.94에서 9.35로 약 6% 상대 개선되었고, DBP standard deviation은 8.10에서 7.72로 약 4.7% 개선되었다. Pearson correlation coefficient도 SBP에서 0.84에서 0.86으로, DBP에서 0.76에서 0.80으로 향상되었다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 “PPG channel을 하나로 고정하지 말고, 여러 위치에서 동시에 측정한 뒤 attention mechanism이 중요한 channel을 adaptive하게 선택하도록 하자”는 것이다. 기존 single-channel PPG 기반 BP estimation에서는 sensor 위치가 잘못되면 낮은 quality의 PPG가 입력되어 모델 성능이 크게 떨어진다. 반면 9-channel PPG sensor를 사용하면 손가락의 여러 위치에서 동시에 optical signal을 얻을 수 있고, 어떤 channel이 혈압 추정에 유용한지 모델이 학습할 수 있다.

두 번째 핵심 아이디어는 finger pressure signal을 함께 사용하는 것이다. 본 연구에서 사용자는 40초 동안 손가락으로 sensor를 점진적으로 누른다. 이때 PPG signal은 압력 증가에 따른 혈관 폐색 및 pulse amplitude 변화를 반영하고, force sensor는 사용자가 가한 압력의 시간 변화를 측정한다. 이러한 signal pair는 cuff-based oscillometric measurement에서 cuff pressure와 oscillometric wave가 함께 사용되는 것과 유사한 정보를 제공한다. 즉 단순히 안정 상태의 PPG waveform만 보는 것이 아니라, progressive pressure에 따른 PPG 변화 양상을 학습한다.

세 번째 핵심 아이디어는 handcrafted feature extraction을 피하고 end-to-end CNN-based feature extractor를 사용한 것이다. 기존 PPG 기반 혈압 추정 연구에서는 pulse height, pulse width, derivative point, time interval 등 engineered feature를 사람이 설계하는 경우가 많았다. 그러나 이러한 feature는 연령, 질병, 약물, 혈관 상태, sensor 위치에 따라 안정적으로 추출하기 어렵다. 본 논문은 raw PPG, PPG envelope, derivative signal, finger pressure signal을 CNN에 입력하여 BP estimation에 필요한 latent feature를 자동으로 학습한다.

네 번째 핵심 아이디어는 attention weight 분석을 통해 physiological 또는 sensor-level 해석 가능성을 일부 제공한다는 점이다. 논문은 hypertension, hypotension, normotension group에서 attention weight 분포가 다르게 나타남을 보였다. 예를 들어 SBP attention에서는 hypertension data에서 2번 channel, hypotension data에서 4번 channel의 attention weight가 크게 나타났다. Normotensive data는 attention weight가 비교적 고르게 분포했다. 이는 multi-channel PPG와 attention mechanism이 단순 성능 향상뿐 아니라 혈압 상태에 따라 다른 channel 정보를 활용한다는 점을 시사한다.

## 3. 상세 방법 설명

### 3.1 Sensor 설계와 데이터 수집

논문은 finger pressure 기반 multi-channel PPG sensor를 직접 개발했다. Sensor는 535 nm green LED 3개, 850 nm infrared, IR LED 3개, photodetector, PD 9개로 구성된다. LED와 PD는 finger blood vessel structure를 고려하여 배치되었다. Multi-channel PPG의 FOV는 5 mm × 4.5 mm이고, 전체 sensor 크기는 12 mm × 7.5 mm이다. 이 크기는 손가락이 sensor 전체를 덮을 수 있도록 설계된 것이다.

실험에서는 IR LED 기반 9-channel PPG signal을 43 Hz sampling rate로 측정했다. PPG sensor는 실험 지지대 위의 button 형태로 구성되었고, 그 아래에는 commercial force sensor가 배치되어 손가락이 가하는 force signal을 측정했다. 9-channel PPG와 force signal은 synchronize되어 built-in mainboard의 analog-to-digital converter로 전달되었다.

데이터는 두 기관에서 수집되었다. Dataset1은 러시아 MONIKI Hospital에서 수집되었고, 290명 participant의 1,450 case를 포함한다. Dataset2는 한국 Samsung Medical Center에서 수집되었고, 186명 participant의 865 case를 포함한다. 전체적으로 476명, 2,315 case가 수집되었다. 각 case에는 40초 길이의 synchronized 9-channel PPG와 force signal이 포함되며, reference BP는 두 명의 medical staff가 auscultation 방식으로 측정한 값의 평균이다.

임상 시험 설계는 ISO 81060-2 기준을 따랐다고 설명된다. 이 기준은 non-invasive sphygmomanometer의 clinical investigation과 관련된 subject requirement와 reference measurement 절차를 포함한다. 각 subject에 대해 5회 측정이 수행되었고, 측정 사이에는 안정화를 위해 최소 5분 휴식이 주어졌다. 측정 시 subject는 index finger를 미리 표시된 guideline 위에 놓고, computer screen에 표시된 pressure increase guide를 보면서 40초 동안 sensor를 점진적으로 눌렀다.

### 3.2 Dataset split과 학습 설정

Dataset1과 Dataset2는 서로 다른 방식으로 사용되었다. Dataset1은 training과 validation에만 사용되었다. 구체적으로 Dataset1의 290명 중 183명은 training, 107명은 validation에 사용되었다. Dataset2는 training과 testing에 사용되었으며, participant overlap이 없도록 5-fold cross-validation을 수행했다. Dataset2의 186명 participant를 5개 fold로 나누고, 매번 1개 fold를 test set으로, 나머지 fold를 training set으로 사용했다. 중요한 점은 training, validation, test split에서 participant overlap이 없도록 구성했다는 것이다. 이는 같은 사람의 data가 train과 test에 동시에 들어가 성능이 과대평가되는 문제를 줄이는 장점이 있다.

학습에는 Adam optimizer가 사용되었다. Hyperparameter는 $\beta_1 = 0.9$, $\beta_2 = 0.999$, learning rate 0.005, mini-batch size 64이다. Generalization을 개선하기 위해 $\ell_2$ regularization scale 0.005와 dropout rate 0.3을 사용했다. 이러한 설정은 clinical dataset 규모가 매우 크지 않은 상황에서 overfitting을 줄이기 위한 목적이다.

### 3.3 Signal preprocessing과 입력 구성

Raw PPG와 finger pressure signal에는 noise component가 포함되어 있으므로 filtering과 파생 signal 생성이 수행된다. Raw PPG signal은 각 multi-channel signal에 대해 0.8–8 Hz cut-off frequency를 갖는 band-pass filter를 통과한다. 이 대역은 PPG의 pulse component를 보존하면서 저주파 baseline drift와 고주파 noise를 줄이기 위한 선택으로 이해할 수 있다.

Filtered PPG signal을 $X_p$라고 할 때, PPG envelope $X_e$는 peak detection과 interpolation을 통해 계산된다. Envelope는 finger pressure가 증가하면서 PPG amplitude가 어떻게 변하는지, 즉 oscillometric-like pattern을 반영한다. 또한 PPG와 envelope signal에서 각각 first-order temporal derivative와 second-order temporal derivative를 계산한다. 이는 PPG waveform의 slope, curvature, 변화율 정보를 모델에 제공하기 위함이다.

Finger pressure signal은 raw force signal에서 얻어지며, 0.2 Hz cut-off frequency의 low-pass filter를 통과한다. 이 force signal은 매우 느리게 증가하는 finger pressure guide와 실제 누름 정도를 반영하므로 low-pass filtering이 적절하다. 이후 PPG envelope signal의 maximum point를 기준으로 좌우 5초 구간을 잘라 finger pressure segment를 구성한다.

최종적으로 세 종류의 입력이 만들어진다. 첫 번째 입력 $X_1$은 filtered PPG와 그 first-order derivative, second-order derivative를 channel axis로 concatenate한 것이다.

$$
X_1 = X_p \oplus \Delta X_p \oplus \Delta^2 X_p
$$

두 번째 입력 $X_2$는 PPG envelope와 그 first-order derivative, second-order derivative를 concatenate한 것이다.

$$
X_2 = X_e \oplus \Delta X_e \oplus \Delta^2 X_e
$$

세 번째 입력 $X_3$는 filtered finger pressure signal이다.

$$
X_3 = X_f
$$

논문 Table 2에 따르면 $X_1$과 $X_2$는 각각 1720 × 3 dimension을 가지며, $X_3$는 215 × 1 dimension을 가진다. 이 입력 길이 차이는 PPG-related signal과 finger pressure signal의 sampling 또는 segmentation 처리 차이에서 비롯된 것으로 보인다.

### 3.4 CNN-based feature extractor

제안 system은 각 PPG channel마다 CNN-based feature extractor를 학습한다. 하나의 single-channel BP estimator는 세 개의 parallel input stream을 가진다. 각 stream은 $X_1$, $X_2$, $X_3$ 중 하나를 입력받는다. 논문은 각 stream을 $C(\cdot): X_i \rightarrow Z_i$로 표현한다. 여기서 $X_i$는 입력 signal이고, $Z_i$는 추출된 feature이다.

각 input stream은 먼저 convolution, batch normalization, ReLU activation을 적용한 뒤 max pooling을 수행한다. 이후 residual connection을 포함한 CNN block 세 개가 쌓인다. 각 CNN block은 convolution, batch normalization, ReLU의 반복 구조를 가지며, dropout도 포함된다. 마지막으로 average pooling layer가 각 feature stream의 정보를 집약한다. 세 stream에서 얻어진 feature $Z_1$, $Z_2$, $Z_3$는 concatenate되어 하나의 feature $Z$를 형성한다.

$$
Z = Z_1 \oplus Z_2 \oplus Z_3
$$

Concatenated feature $Z$는 fully connected layer와 sigmoid nonlinearity를 통과하여 16-dimensional latent feature가 된다. 이 latent feature는 특정 PPG channel에서 BP estimation에 필요한 정보를 압축한 representation이다. Residual connection은 깊은 CNN feature extractor를 학습할 때 vanishing gradient 문제를 완화하기 위해 사용된다.

Single-channel feature extractor를 학습할 때 마지막 output layer는 estimated BP $\hat{y}$를 출력한다. Training objective는 reference BP $y_i$와 estimated BP $\hat{y}_i$ 사이의 mean squared error, MSE를 최소화하는 것이다.

$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

여기서 $N$은 sample 수이고, $y_i$는 reference BP, $\hat{y}_i$는 estimated BP이다. SBP와 DBP는 각각 별도의 모델로 학습된 것으로 설명된다. 또한 9개 PPG channel 각각에 대해 feature extraction model이 학습되므로, 최종적으로 9개의 latent feature가 생성된다. 이들을 모으면 $Z \in R^{16 \times 9}$가 되며, 이는 attention mechanism의 입력으로 사용된다.

### 3.5 Multi-channel attention mechanism

Attention mechanism은 각 PPG channel의 latent feature가 BP estimation에 얼마나 중요한지를 adaptive하게 가중한다. 사용자의 finger position과 finger characteristic이 다르기 때문에, 어떤 channel이 좋은 PPG 정보를 담는지는 사람마다 또는 sample마다 달라질 수 있다. 따라서 고정된 channel을 선택하는 대신 attention layer가 channel-wise importance를 학습한다.

각 PPG channel $i$의 latent feature를 $Z_i$라고 할 때, attention layer는 single-layer perceptron $s(\cdot)$을 통해 score $S_i$를 계산한다.

$$
S_i = s(\omega Z_i + b)
$$

여기서 $\omega$와 $b$는 trainable weight와 bias이다. 이후 softmax function을 통해 attention weight $W_i$를 계산한다.

$$
W_i = softmax(S_i) = \frac{\exp(S_i)}{\sum_i \exp(S_i)}
$$

이 $W_i$는 각 channel의 중요도를 확률값처럼 나타낸다. Attention-weighted feature $Z' \in R^{16 \times 1}$는 각 channel feature와 attention weight의 weighted summation으로 계산된다.

$$
Z' = \sum_i W_i Z_i
$$

마지막 output layer는 이 attention-weighted feature $Z'$를 사용하여 최종 BP를 추정한다. Attention mechanism 역시 reference BP와 estimated BP 사이의 MSE loss를 최소화하도록 학습된다. 논문은 SBP와 DBP estimation을 위해 attention-based model을 별도로 학습했다고 설명한다.

이 구조의 의미는 단순한 hard channel selection이 아니라 soft adaptive combination이다. 즉 특정 subject에서 한 channel만 선택하는 것이 아니라, 9개 channel의 latent feature를 attention weight에 따라 연속적으로 조합한다. 이 점은 이후 실험에서 top-2 또는 top-3 channel만 사용하는 hard selection보다 전체 9-channel attention이 더 우수한 결과를 보인다는 점으로 검증된다.

## 4. 실험 및 결과

### 4.1 평가 지표

논문은 mean error, ME, standard deviation of error, STD, Pearson correlation coefficient $r$을 사용하여 BP estimation performance를 평가한다. ME는 예측값과 reference 사이의 평균적인 bias를 나타내고, STD는 error의 변동성을 나타낸다. 혈압 추정 연구에서는 ME가 0에 가까워도 STD가 크면 individual sample에서 오차가 클 수 있으므로, ME와 STD를 함께 보는 것이 중요하다. Pearson correlation coefficient는 예측 BP와 reference BP 사이의 선형 관계를 나타낸다.

논문은 AAMI criterion도 논의한다. AAMI 기준에서는 BP estimation error가 대체로 $5 \pm 8$ mmHg 범위에 있어야 한다고 설명된다. 논문 Discussion에 따르면 제안 모델은 DBP에 대해서는 AAMI criterion을 만족했고, SBP도 기준에 근접했다.

### 4.2 Single-channel BP estimation 결과

논문은 먼저 9개 PPG channel 각각을 단독으로 사용할 때의 SBP와 DBP estimation performance를 분석한다. 이는 single-channel PPG sensor가 finger position에 얼마나 민감한지 보여주기 위한 실험이다.

SBP estimation에서는 2번 channel이 가장 좋은 single-channel 성능을 보였다. 2번 channel의 평균 성능은 ME ± STD 1.90 ± 9.94 mmHg이고, correlation coefficient는 0.84였다. 반면 7번 channel은 ME ± STD 2.01 ± 10.99 mmHg, correlation 0.80으로 가장 나쁜 편이었다. 논문은 SBP performance에서 2번 channel과 7번 channel 사이의 relative performance gap이 약 9.6%라고 설명한다.

DBP estimation에서는 3번 channel이 가장 좋은 single-channel 성능을 보였다. 3번 channel의 평균 성능은 ME ± STD 0.75 ± 8.10 mmHg이고, correlation은 0.76이었다. 반면 6번 channel이나 7번 channel 등은 더 큰 STD와 낮은 correlation을 보였다. 논문은 DBP performance에서 best channel과 worst channel 사이에 약 3.7% 차이가 있다고 설명한다.

이 결과는 중요한 메시지를 갖는다. 9개 PPG channel은 동시에 측정되었지만, channel 위치에 따라 BP estimation accuracy가 달라졌다. 즉 손가락이 sensor 위에 놓이는 위치와 finger vascular structure 차이 때문에 single-channel PPG는 모든 사용자에게 일관된 품질을 제공하기 어렵다. 이 결과가 multi-channel sensor와 attention mechanism의 필요성을 뒷받침한다.

### 4.3 Attention mechanism의 성능 개선

Table 5는 best single-channel estimator와 proposed multi-channel attention estimator를 비교한다. SBP에서는 best single-channel인 channel 2가 평균 ME ± STD 1.90 ± 9.94 mmHg, correlation 0.84를 보였다. Proposed attention mechanism은 평균 ME ± STD 0.43 ± 9.35 mmHg, correlation 0.86을 달성했다. STD 기준으로 약 6% 상대 개선이다.

DBP에서는 best single-channel인 channel 3이 평균 ME ± STD 0.75 ± 8.10 mmHg, correlation 0.76을 보였다. Proposed attention mechanism은 평균 ME ± STD 0.21 ± 7.72 mmHg, correlation 0.80을 달성했다. STD 기준으로 약 4.7% 상대 개선이다.

이 결과는 attention mechanism이 단순히 여러 channel을 사용하는 데 그치지 않고, channel-wise latent feature를 효과적으로 결합하여 혈압 추정 오차를 줄인다는 것을 보여준다. 또한 correlation coefficient도 SBP와 DBP 모두에서 증가했으므로, 예측값이 reference BP 변화 추세를 더 잘 따라간다고 볼 수 있다.

### 4.4 Attention weight 분석

논문은 attention weight를 hypertension, hypotension, normotension group으로 나누어 분석했다. Figure 3은 SBP와 DBP estimation task 각각에 대해 9개 channel의 average attention weight를 bar graph로 보여준다.

흥미로운 결과는 hypertension과 hypotension data에서 특정 channel의 attention weight가 상대적으로 크게 나타난다는 점이다. SBP attention mechanism에서는 hypertension data에서 2번 channel의 attention weight가 크고, hypotension data에서는 4번 channel의 attention weight가 컸다. 또한 hypertension data에서 가장 큰 attention weight를 받은 channel이 hypotension data에서는 상대적으로 낮은 attention weight를 갖는 경향도 관찰되었다. DBP attention mechanism에서도 유사한 경향이 나타났다. 반면 normotensive data에서는 attention weight가 상대적으로 고르게 분포했다.

이 결과는 attention mechanism이 모든 sample에서 같은 channel을 사용하는 것이 아니라, BP state에 따라 다른 channel 정보를 활용할 수 있음을 보여준다. 논문은 이러한 결과가 multi-channel PPG sensor와 attention mechanism이 hypertensive 및 hypotensive user를 구분하는 데 효과적으로 사용될 수 있음을 시사한다고 해석한다.

### 4.5 Attention 방법 변경 실험

논문은 attention mechanism의 효과를 더 확인하기 위해, 전체 9개 channel을 soft attention으로 결합하는 방식과 attention weight가 높은 top-2 또는 top-3 channel만 사용하는 hard selection 방식을 비교했다.

SBP에서 top-2 attention weight channel만 사용한 경우 평균 ME ± STD는 1.92 ± 10.10 mmHg, correlation 0.83이었다. Top-3만 사용한 경우는 0.72 ± 9.63 mmHg, correlation 0.85였다. Proposed full attention mechanism은 0.43 ± 9.35 mmHg, correlation 0.86으로 가장 좋았다.

DBP에서도 top-2는 0.53 ± 8.04 mmHg, correlation 0.76이고, top-3는 0.18 ± 7.87 mmHg, correlation 0.79였다. Proposed full attention mechanism은 0.21 ± 7.72 mmHg, correlation 0.80으로 STD와 correlation 기준에서 가장 좋았다. DBP의 ME만 보면 top-3가 0.18로 slightly lower이지만, STD와 correlation을 함께 보면 full attention이 더 안정적인 성능을 보인다.

이 실험은 attention weight가 높은 channel 몇 개만 고정적으로 사용하는 hard selection보다, 모든 channel을 soft하게 조합하는 방식이 더 효과적임을 보여준다. 즉 attention weight는 단순 ranking 용도가 아니라, multi-channel latent feature를 adaptive하게 조합하기 위한 continuous weighting으로 사용하는 것이 중요하다.

### 4.6 입력 signal combination 분석

Table 7은 single-channel BP estimation model에서 어떤 입력 조합이 좋은지 비교한다. 논문은 세 가지 입력 조합을 비교한다. $E+F$는 PPG envelope 및 envelope derivative 관련 signal과 finger pressure를 사용하는 경우이고, $P+F$는 filtered PPG 및 PPG derivative 관련 signal과 finger pressure를 사용하는 경우이다. $P+E+F$는 PPG, PPG envelope, finger pressure signal을 모두 사용하는 경우이다.

SBP에서 $E+F$는 평균 ME ± STD 1.72 ± 11.29 mmHg, correlation 0.76이고, $P+F$는 -0.08 ± 11.06 mmHg, correlation 0.79였다. $P+E+F$는 1.90 ± 9.94 mmHg, correlation 0.84로 가장 좋은 STD와 correlation을 보였다. DBP에서도 $E+F$는 0.39 ± 8.62 mmHg, correlation 0.69, $P+F$는 0.77 ± 8.55 mmHg, correlation 0.71, $P+E+F$는 0.75 ± 8.10 mmHg, correlation 0.76으로 가장 좋았다.

이 결과는 PPG waveform 자체, PPG envelope, finger pressure를 함께 사용할 때 가장 풍부한 정보를 제공한다는 것을 보여준다. 특히 envelope signal은 pressure-induced amplitude change를 반영하고, PPG derivative는 waveform shape 변화를 반영하며, force signal은 압력 입력 조건을 반영하므로 세 정보가 상보적으로 작동한다고 해석할 수 있다.

### 4.7 Scatter plot과 Bland-Altman plot 분석

논문은 proposed BP estimation system의 최종 결과를 scatter plot과 Bland-Altman plot으로 제시한다. Scatter plot에서 SBP와 DBP의 Pearson correlation coefficient는 각각 0.86과 0.80이었다. 이는 predicted BP가 reference BP와 비교적 강한 양의 상관관계를 가짐을 의미한다.

Bland-Altman plot에서는 대부분의 SBP와 DBP sample이 limits of agreement 안에 포함되었다고 설명된다. Bland-Altman plot은 prediction error가 혈압 수준에 따라 systematic하게 증가하거나 감소하는지 확인하는 데 중요하다. 논문은 대부분의 data가 agreement limit 안에 있으므로 proposed system의 prediction이 reference measurement와 비교적 일치한다고 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 sensor hardware와 deep learning algorithm을 함께 설계했다는 점이다. 많은 BP estimation 연구가 공개 데이터셋의 PPG 또는 ECG waveform만을 사용하여 algorithm을 제안하는 반면, 본 연구는 finger pressure 방식에 맞는 9-channel PPG sensor와 force sensor를 직접 설계하고, 해당 sensor에서 얻은 data에 적합한 CNN-attention model을 구성했다. 이는 실제 cuffless BP measurement device 개발 관점에서 의미가 크다.

두 번째 강점은 single-channel PPG의 위치 민감성을 실험적으로 보여주고, 이를 multi-channel attention으로 완화했다는 점이다. 논문은 channel별 성능 차이가 SBP와 DBP 모두에서 존재함을 정량적으로 제시했다. 특히 SBP에서는 best channel과 worst channel 간 STD 차이가 꽤 크다. 이 결과는 single-channel PPG sensor가 사용자 finger placement에 따라 불안정할 수 있음을 보여주며, multi-channel design의 필요성을 뒷받침한다.

세 번째 강점은 participant overlap을 피한 validation design이다. Dataset1은 training과 validation에 사용되고, Dataset2는 5-fold cross-validation으로 training과 testing에 사용되며, fold 간 participant가 겹치지 않도록 구성되었다. 생체신호 기반 BP estimation에서 같은 subject의 data가 train과 test에 동시에 들어가면 성능이 과대평가될 수 있는데, 본 연구는 적어도 participant-level overlap을 피하려는 설계를 명시한다.

네 번째 강점은 attention weight 분석을 통해 모델이 단순 black-box로만 동작하지 않음을 일부 보여준 점이다. Hypertension, hypotension, normotension에 따라 channel attention distribution이 달라졌다는 결과는 model behavior를 이해하는 데 도움이 된다.

그러나 한계도 분명하다. 첫째, 데이터셋 크기가 deep learning 관점에서 충분히 크지 않다. 전체 subject는 476명이고 case는 2,315개이다. 논문도 MIMIC online waveform database나 University of Queensland Vital Signs dataset에 비해 dataset이 작다고 인정한다. Neural network 기반 BP prediction model은 dataset size에 민감하므로, 더 많은 data가 수집되면 성능이 향상될 가능성이 있다고 논문은 설명한다.

둘째, 데이터는 static condition에서만 수집되었다. 실제 cuffless BP monitoring에서는 inter-individual variation뿐 아니라 intra-individual BP variation도 중요하다. 운동, 스트레스, 자세 변화, 수면, 약물, 시간대 변화에 따른 혈압 변화를 추적할 수 있어야 한다. 그러나 이 연구는 static condition에서만 BP를 측정했고, 개인 내부의 다양한 BP 변화는 충분히 고려하지 못했다.

셋째, Dataset1과 Dataset2 사이에 domain mismatch가 존재한다. 두 dataset은 서로 다른 병원, 국가, 측정 환경, 일부 sensor specification 차이를 가진다. 논문은 이러한 domain difference 때문에 target dataset에서 높은 accuracy를 기대하기 어렵다고 설명한다. Regularization과 dropout, 5-fold cross-validation을 사용했지만, domain adaptation을 적용하면 더 나은 성능을 얻을 수 있을 것이라고 제안한다.

넷째, SBP 성능은 AAMI criterion에 “close”하지만 완전히 만족한다고 단정하기 어렵다. Discussion에서 저자들은 DBP는 AAMI criterion을 만족했고, SBP도 기준에 가까웠다고 표현한다. 최종 SBP STD가 9.35 mmHg로 AAMI 기준에서 흔히 언급되는 8 mmHg보다 크기 때문이다. 따라서 SBP 추정은 아직 의료기기 수준의 엄격한 정확도를 만족하기 위해 개선이 필요하다.

다섯째, 모델은 demographic information을 사용하지 않는다. Age, gender, height, weight 등은 혈압 및 PPG morphology에 영향을 줄 수 있다. 논문은 사용자가 demographic information을 입력해야 하는 번거로움을 줄이기 위해 이를 사용하지 않았다고 설명한다. 이는 user convenience 측면에서는 장점이지만, accuracy 향상 가능성을 일부 포기한 설계이기도 하다.

여섯째, finger pressure 기반 방식은 사용자의 조작에 의존한다. 사용자는 40초 동안 화면 guide를 보며 점진적으로 sensor를 눌러야 한다. 이는 cuff보다 간단할 수 있지만, 완전 passive wearable monitoring과는 다르다. 사용자가 일정한 방식으로 pressure를 증가시키지 못하거나 손가락 움직임이 크면 signal quality가 떨어질 수 있다.

비판적으로 보면, 이 연구는 cuffless BP estimation에서 sensor placement 문제를 정면으로 다룬 점이 강하다. Multi-channel PPG와 attention mechanism은 실제 device variation에 대응하는 현실적인 접근이다. 그러나 최종 error, 특히 SBP STD가 여전히 높은 편이고, static condition data에 기반하기 때문에 실제 continuous monitoring system으로 사용하려면 더 많은 subject, 다양한 혈압 변화, domain adaptation, user behavior robustness 검증이 필요하다.

## 6. 결론

이 논문은 multi-channel PPG sensor와 finger pressure signal을 이용한 cuff-free, calibration-free BP estimation system을 제안했다. 제안 sensor는 9-channel PPG와 force signal을 동시에 측정하며, 사용자가 손가락으로 sensor를 점진적으로 누르는 동안 pressure-induced PPG 변화를 수집한다. 제안 algorithm은 CNN-based feature extractor로 PPG, PPG envelope, derivative signal, finger pressure signal에서 latent feature를 추출하고, multi-channel attention mechanism으로 9개 PPG channel의 feature를 adaptive하게 결합하여 SBP와 DBP를 예측한다.

실험 결과, proposed attention-based multi-channel system은 SBP에서 ME ± STD 0.43 ± 9.35 mmHg, correlation 0.86을 달성했고, DBP에서 0.21 ± 7.72 mmHg, correlation 0.80을 달성했다. Best single-channel model과 비교해 SBP와 DBP 모두에서 STD와 correlation이 개선되었다. Channel별 성능 차이와 attention weight 분석은 손가락 위치와 channel 선택이 BP estimation에 큰 영향을 미친다는 점을 보여주며, multi-channel attention이 이러한 문제를 완화할 수 있음을 뒷받침한다.

이 연구의 주요 기여는 single-channel PPG 기반 finger pressure BP estimation의 위치 민감성 문제를 multi-channel sensing과 attention mechanism으로 해결하려 했다는 점이다. 또한 handcrafted feature 없이 raw 및 derived signal을 CNN으로 처리하는 end-to-end deep learning 구조를 제안했고, participant overlap이 없는 cross-validation으로 모델을 평가했다.

향후 연구에서는 더 큰 clinical dataset 수집, intra-individual BP variation 포함, motion 및 real-world usage condition 검증, domain adaptation 적용, SBP accuracy 개선, 실제 smartphone 또는 smartwatch form factor에서의 usability 평가가 필요하다. 이러한 보완이 이루어진다면, 본 연구의 multi-channel PPG와 attention 기반 finger pressure 방식은 cuffless BP monitoring을 mobile device에서 구현하는 유망한 방향이 될 수 있다.
