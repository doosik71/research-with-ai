# Continuous Cuffless Blood Pressure Monitoring Using Photoplethysmography-Based PPG2BP-Net for High Intrasubject Blood Pressure Variations

* **저자**: Jingon Joung, Chul-Woo Jung, Hyung-Chul Lee, Moon-Jung Chae, Hae-Sung Kim, Jonghun Park, Won-Yong Shin, Changhyun Kim, Minhyung Lee, Changwoo Choi
* **발표연도**: 2023

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 이용하여 cuff 없이 continuous blood pressure, 즉 연속 혈압을 추정하는 PPG2BP-Net을 제안한다. 논문의 핵심 목표는 기존 PPG 기반 cuffless BP estimation 연구들이 충분히 해결하지 못한 subject-independent 환경과 high intrasubject blood pressure variation 문제를 다루는 것이다. 여기서 subject-independent란 학습에 사용된 사람과 테스트에 사용된 사람이 완전히 분리되어 있다는 의미이며, high intrasubject BP variation은 한 사람 안에서도 calibration 시점의 혈압과 이후 target 혈압 사이의 차이가 크게 변한다는 의미이다.

논문은 혈압 측정에서 continuous, comfortable, convenient, accurate를 합쳐 C3A라는 요구를 강조한다. 기존 cuff-based BP measurement는 비교적 신뢰도 높은 정확도를 달성할 수 있지만, 장시간 연속 측정이 어렵고, cuff size와 position에 민감하며, white-coat hypertension이나 masked hypertension처럼 측정 상황에 따른 문제가 발생할 수 있다. 반면 cuffless BP measurement는 착용성과 연속성 면에서 유리하지만, 실제로 높은 정확도를 확보하기 어렵다. 특히 PPG 기반 방법은 단일 wearable sensor로 측정할 수 있어 편리하지만, PPG waveform은 자세, 운동, 마취, 혈관 반응, 생리적 변화 등 다양한 요인에 의해 변동성이 크기 때문에 안정적인 혈압 추정이 어렵다.

이 논문이 다루는 연구 문제는 “초기 calibration BP와 calibration PPG가 주어진 상황에서, 새로운 subject의 이후 PPG segment로부터 크게 변동하는 SBP와 DBP를 정확히 추정할 수 있는가”이다. 저자들은 이를 위해 calibration segment와 target segment를 비교하는 comparative paired 1D-CNN 구조를 설계하였다. 모델은 calibration PPG와 target PPG를 각각 동일한 구조와 동일한 parameter를 공유하는 1D-CNN에 통과시키고, 두 feature와 그 차이를 이용하여 target segment의 SBP와 DBP를 예측한다. 또한 calibration SBP와 DBP는 별도의 multilayer perceptron, 즉 MLP를 통해 numerical calibration feature로 변환되어 최종 예측에 함께 사용된다.

논문은 대규모 실제 수술실 데이터에 기반한다는 점에서도 중요하다. 서울대학교병원에서 2016년부터 2019년까지 4년간 수집된 25,779 surgical cases의 raw vital waveforms에서 전처리를 통해 4,185명의 clean independent subjects를 선별하였다. 이 중 2,978명은 training, 410명은 validation, 797명은 holdout test에 사용되었다. 이는 많은 기존 PPG 기반 혈압 추정 연구가 subject-dependent 평가, 소규모 subject 수, 낮은 intrasubject BP variation 조건에서 검증되었다는 한계를 보완하려는 설계이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 calibration-based cuffless BP estimation에서 단순히 PPG waveform 하나로 혈압을 예측하는 것이 아니라, calibration PPG와 target PPG 사이의 변화 관계를 직접 학습해야 한다는 것이다. 일반적인 calibration-based BP estimation은 초기 calibration BP를 기준점으로 사용한다. 그러나 한 사람의 혈압이 시간이 지나면서 크게 변하는 경우, 모델이 calibration BP에 지나치게 의존하면 이후 target BP 변화를 제대로 추정하지 못할 수 있다. PPG2BP-Net은 calibration segment와 target segment를 paired input으로 넣고, 두 PPG feature의 absolute difference까지 학습함으로써 calibration 시점 대비 target 시점에서 PPG가 어떻게 변했는지를 반영한다.

두 번째 핵심 아이디어는 subject-independent modeling이다. 많은 기존 연구에서는 같은 subject의 PPG segment가 train과 test에 동시에 포함되는 subject-dependent 방식이 사용되었다. 이 경우 모델이 사람 고유의 waveform pattern을 기억할 가능성이 있어, 실제 새로운 사용자에게 적용했을 때의 성능을 과대평가할 수 있다. 본 논문은 training, validation, test subject를 배타적으로 분리하여 모델이 never-seen subject에 대해 동작하도록 평가하였다. 이는 실제 cuffless BP monitoring device가 새로운 사용자에게 적용되는 상황과 더 가까운 설정이다.

세 번째 핵심 아이디어는 high intrasubject BP variation을 정량화하기 위해 SDS, 즉 standard deviation of subject-calibration centring이라는 새로운 metric을 제안한 것이다. 기존의 혈압 분포 SD는 전체 subject와 segment의 BP 분산을 나타내지만, calibration-based estimator에서 중요한 것은 각 subject 안에서 calibration BP로부터 target BP가 얼마나 멀리 변하는가이다. SDS는 각 subject의 segment BP에서 해당 subject의 calibration BP를 뺀 값을 중심으로 표준편차를 계산한다. 따라서 calibration 이후 혈압이 크게 변하는 상황이 얼마나 어려운 평가 조건인지 더 직접적으로 보여준다.

네 번째 핵심 아이디어는 충분히 큰 subject 수와 엄격한 preprocessing을 결합한 점이다. 저자들은 25,779 surgical cases에서 abnormal cases, abnormal segments, sparse BP range, 부족한 segment 수를 제거하고, 각 subject가 50개에서 100개의 clean segment를 갖도록 balancing하였다. 이를 통해 특정 subject가 학습에 과도하게 영향을 미치는 것을 줄이고, subject-wise batch construction을 통해 다양한 subject의 calibration-target pair를 학습하도록 했다.

기존 접근 방식과의 차별점은 명확하다. 이 논문은 ECG를 사용하지 않고 PPG만 사용하지만, 단순 PPG-to-BP regression이 아니라 calibration PPG와 target PPG의 비교 구조를 사용한다. 또한 subject-independent holdout test를 대규모 subject에서 수행하고, SDS를 통해 calibration 기반 모델의 실제 난이도를 평가한다. 저자들은 이 조합이 C3A cuffless BP estimation device 설계를 위한 현실적인 검증 방향이라고 주장한다.

## 3. 상세 방법 설명

PPG2BP-Net의 전체 방법은 데이터 수집, preprocessing, SDS metric 계산, subject-wise paired training batch 구성, comparative paired 1D-CNN 기반 feature extraction, calibration BP를 처리하는 MLP, 그리고 최종 fully connected layer를 통한 SBP와 DBP 예측으로 구성된다.

### 3.1 데이터 수집과 전체 파이프라인

데이터는 서울대학교병원 수술실에서 Vital Recorder를 이용해 수집되었다. ABP와 PPG waveforms는 TramRac-4A 장비로 측정되었고, ABP 및 NIBP SBP/DBP numeric data는 Solar 8000M 장비로 측정되었다. 일부 마취 관련 정보는 Primus와 Orchestra 장비에서 수집되었다. Vital Recorder는 여러 마취 장비의 high-resolution time-synchronised physiological data를 자동 기록하는 도구이다. Raw waveform은 500 Hz sampling frequency로 저장되었다.

연구에서는 adult patients, 즉 18세 이상 90세 이하이며 intraoperative ABP monitoring이 수행된 수술 case가 포함되었다. Retrospective study design으로 진행되었고, 논문에 따르면 SNUH IRB 승인을 받았으며 informed consent requirement는 면제되었다.

### 3.2 Data preprocessing

전처리는 크게 abnormal surgical case elimination, downsampling and segmentation, abnormal segment elimination, normalization, balancing the number of segments로 구성된다.

먼저 abnormal surgical case elimination에서는 두 조건이 사용되었다. T1은 subject의 기본 정보가 비정상적이지 않아야 한다는 조건이다. 예를 들어 $10 \le weight \le 100$ kg, $100 \le height \le 200$ cm, $18 \le age \le 100$ years, nonpregnant 조건을 만족해야 한다. T2는 operation time log, PPG, ABP 등 essential information이 존재해야 한다는 조건이다. T1 또는 T2를 위반한 case는 제거되었다. 이 단계에서 T1 위반으로 469개 case, T2 위반으로 8,040개 case가 제거되었고, 17,271개 clean cases가 남았다.

다음으로 downsampling and segmentation을 수행하였다. 원본 ABP와 PPG는 500 Hz로 sampling되었지만, training complexity를 줄이기 위해 50 Hz로 downsampling되었다. 이후 각 waveform은 10초 segment로 나뉘었으며, 각 segment는 500 points를 포함한다. Non-overlapped segmentation을 사용하여 수집 데이터의 정보를 최대한 중복 없이 활용하려 했다.

Abnormal segment elimination에서는 movement artifact, device not wearing, 비정상 ABP fluctuation, irregular pulse 등의 영향을 줄이기 위해 segment-level 조건을 적용하였다. T3은 PPG와 ABP segment가 null value 없이 valid data를 포함하고 최소 하나 이상의 non-zero data를 가져야 한다는 조건이다. T4는 typical SBP range에 관한 조건으로, average SBP가 $70 \le average\ SBP \le 180$ mmHg 범위에 있어야 한다. 이 조건을 만족하지 않는 segment는 제거되었다. 이 때문에 제안 모델의 예측 가능한 혈압 범위는 기본적으로 70 mmHg에서 180 mmHg 사이로 제한된다. 논문은 이 범위를 확장하려면 70 mmHg 미만과 180 mmHg 초과의 충분하고 신뢰할 수 있는 데이터를 추가로 학습해야 한다고 명시한다.

Normalization 단계에서는 A-line SBP와 DBP를 전체 training set의 mean과 standard deviation으로 standardization하였다. 이 과정은 학습 안정성과 정확도를 높이기 위한 것이다.

마지막으로 balancing the number of segments에서는 각 subject가 학습에 공정하게 기여하도록 segment 수를 제한하였다. T5 조건은 abnormal segment elimination 이후 남은 clean PPG와 ABP segment 수가 50개 이상이어야 한다는 것이다. 50개 미만 segment를 가진 subject는 제거되었다. 반대로 100개보다 많은 clean segment를 가진 subject는 임의로 100개만 선택하였다. 따라서 최종 subject는 50개에서 100개의 clean segment를 가지며, 특정 subject가 지나치게 많은 segment로 training을 지배하지 않도록 하였다.

최종적으로 25,779 surgical cases에서 4,221 cleaned cases가 얻어졌고, 이 중 4,185 clean independent subjects가 분석에 사용되었다. 이들은 training 2,978명, validation 410명, test 797명으로 나뉘었다.

### 3.3 SDS metric

이 논문의 중요한 방법론적 기여는 SDS metric이다. 일반적인 SD는 모든 subject와 segment의 BP deviation을 계산하지만, calibration-based BP estimator에서 핵심은 calibration BP에서 target BP가 얼마나 벗어나는지이다. 이를 위해 논문은 subject-calibration centring ABP를 다음과 같이 정의한다.

$$
s_{i,n}=x_{i,n}-x_{i,c}
$$

여기서 $x_{i,n}$은 subject $i$의 $n$번째 segment ABP이고, $x_{i,c}$는 subject $i$의 calibration에 사용된 ABP이다. 즉 $s_{i,n}$은 특정 subject의 target segment BP가 calibration BP에서 얼마나 벗어났는지를 나타낸다.

기존 SD는 다음과 같이 정의된다.

$$
SD=\sqrt{\frac{1}{\sum_i N_i-1}\sum_i\sum_{n=1}^{N_i}(x_{i,n}-\bar{x})^2}
$$

반면 SDS는 calibration-centred value $s_{i,n}$에 대해 다음과 같이 정의된다.

$$
SDS=\sqrt{\frac{1}{\sum_i N_i-1}\sum_i\sum_{n=1}^{N_i}(s_{i,n}-\bar{s})^2}
$$

여기서 $\bar{x}$와 $\bar{s}$는 각각 전체 $x_{i,n}$과 $s_{i,n}$의 평균이다.

$$
\bar{x}=\frac{1}{\sum_i N_i}\sum_i\sum_{n=1}^{N_i}x_{i,n}
$$

$$
\bar{s}=\frac{1}{\sum_i N_i}\sum_i\sum_{n=1}^{N_i}s_{i,n}
$$

SDS는 calibration-based BP estimator의 design difficulty를 나타내는 지표로 해석된다. SDS가 낮으면 target BP가 calibration BP 주변에서 크게 벗어나지 않는다는 뜻이므로, 모델이 실제로 PPG에서 혈압 변화를 잘 추정하지 못하더라도 calibration BP를 거의 그대로 출력하는 것만으로 좋은 SD error를 얻을 수 있다. 저자들은 이를 overqualified issue라고 부른다. 반대로 SDS가 높으면 calibration BP와 target BP 사이의 차이가 크므로, 모델이 PPG 변화로부터 실제 혈압 변화를 추정해야 한다. 따라서 높은 SDS 조건에서 좋은 성능을 보이는 것이 더 강한 검증이다.

논문은 또 다른 문제로 nonregenerative issue를 제시한다. 어떤 모델이 높은 inter-subject BP variation에서는 좋은 성능을 보일 수 있지만, 실제 새로운 subject의 high intrasubject BP variation을 잘 예측한다는 보장은 없다. SDS는 이 문제를 더 명확히 측정하기 위한 지표이다.

### 3.4 Subject-wise batch construction

PPG2BP-Net은 subject-wise batch construction 방식으로 학습된다. Training batch size는 64 segments로 설정되었다. 한 batch를 구성할 때 2,978명의 training subjects 중 64명의 independent subjects를 무작위로 선택한다. 각 subject에서 target segment $(x_{j,sub},p_{j,sub})$ 하나와 calibration segment $(x_{i,sub},p_{i,sub})$ 하나를 선택한다. 여기서 $x$는 PPG segment, $p$는 corresponding BP value를 의미하는 것으로 해석된다.

이러한 random subject and segment selection은 training subject들이 동일한 weight로 학습에 기여하도록 하며, balancing the number of segments와 유사한 목적을 갖는다. 또한 모델이 다양한 calibration-target 조합을 학습하게 하여, calibration BP와 target BP 사이의 변화 관계를 더 일반적으로 포착하도록 한다. Learning rate와 epoch 수는 stochastic하게 결정되지만, initial learning rate는 0.0001이고 maximum epoch는 1000으로 설정되었다.

### 3.5 PPG2BP-Net architecture

PPG2BP-Net은 comparative paired 1D-CNNs, MLP, final fully connected layer로 구성된다.

가장 핵심적인 부분은 comparative paired 1D-CNN 구조이다. Calibration PPG segment와 target PPG segment는 각각 $1 \times 500$ vector이다. 두 segment는 동일한 구조와 동일한 parameter를 공유하는 두 개의 1D-CNN에 입력된다. 이 shared network 구조는 calibration PPG와 target PPG에서 같은 방식으로 feature를 추출하도록 하며, 두 waveform의 feature 차이를 일관되게 비교할 수 있게 한다.

각 1D-CNN은 네 개의 hidden layer group으로 구성된다. 각 hidden group은 convolutional layer, batch normalization layer, ReLU layer를 포함한다. Convolutional layer는 PPG waveform에서 시간 방향의 local pattern과 hidden nonlinear feature를 추출한다. Batch normalization은 hidden layer input distribution을 안정화하여 convergence speed와 learning performance를 개선한다. ReLU는 비선형성을 제공하고 학습을 빠르게 한다.

네 번째 hidden layer group 이후에는 average pooling layer가 사용된다. Average pooling은 feature의 essential information을 유지하면서 network complexity를 줄인다. 이후 dropout layer에서 training 중 output data의 30%가 무작위로 zero 처리된다. Dropout rate는 0.3이며, overfitting을 줄이고 generalization을 개선하기 위한 것이다.

각 1D-CNN output은 FCL과 batch normalization을 거쳐 feature representation이 된다. 최종 FCL module에는 네 종류의 정보가 입력된다. 첫째는 calibration PPG에서 추출한 1D-CNN feature이고, 둘째는 target PPG에서 추출한 1D-CNN feature이다. 셋째는 두 feature의 absolute difference이다. 넷째는 calibration SBP와 DBP를 MLP로 처리한 numerical calibration feature이다.

MLP는 calibration SBP와 calibration DBP numerical value를 별도로 fully connected layer에 통과시켜 feature를 추출한다. 각 FCL 뒤에는 batch normalization과 ReLU가 따른다. SBP와 DBP 각각에서 추출된 feature는 concatenation되어 최종 FCL module의 입력 중 하나가 된다. 이렇게 함으로써 모델은 calibration BP의 수치적 기준값과 PPG waveform 변화 정보를 함께 활용한다.

최종 FCL에서는 calibration PPG feature, target PPG feature, 두 feature의 difference, calibration BP feature를 모두 concatenate한다. 이후 fully connected layer, batch normalization, ReLU를 거쳐 마지막 fully connected layer에서 target SBP와 DBP를 출력한다.

이 구조의 직관은 분명하다. 단순히 target PPG만 보고 혈압을 추정하는 것이 아니라, “이 subject의 calibration 시점 PPG와 BP는 이랬고, 현재 PPG는 이렇게 달라졌으니, 현재 BP는 calibration BP에서 이 정도 변했을 것이다”라는 관계를 CNN feature space에서 학습하는 것이다.

### 3.6 Validation and test procedure

Validation과 test에서는 두 개의 calibration segment를 사용한다. 각 subject의 첫 번째와 두 번째 PPG, SBP, DBP segment가 calibration segment로 사용되고, 나머지 independent segment가 target으로 사용된다. Target segment에 대한 estimated SBP와 DBP는 첫 번째 calibration segment 기반 prediction과 두 번째 calibration segment 기반 prediction의 평균값이다.

논문은 re-calibration을 고려하지 않았다고 명시한다. Re-calibration은 BP estimation accuracy를 향상시킬 수 있지만, continuous, comfortable, convenient한 C3 BP estimation의 목적과 맞지 않을 수 있기 때문이다. 따라서 이 연구는 초기 calibration만으로 이후 target BP를 추정하는 어려운 설정을 택했다.

## 4. 실험 및 결과

### 4.1 데이터셋 특성

전체 raw data는 25,779 surgical cases에서 수집되었다. 전처리 후 4,185 clean independent subjects가 최종 사용되었다. Training set은 2,978 subjects와 229,323 segments, validation set은 410 subjects와 31,152 segments, Whole test set은 797 subjects와 60,060 segments로 구성된다.

Whole test set 외에도 세 개의 test subset이 구성되었다. ABP-20m set은 A-line insertion 이후 20분이 지난 뒤 수집된 segment가 10개 이상 있는 subject만 포함한다. 이는 A-line insertion 직후 약 20분 동안 ABP waveform이 불안정할 수 있다는 rationale에 따른 것이다. NIBP-c set은 45초 이내 평균 A-line SBP/DBP와 noninvasive BP, 즉 NIBP의 차이가 10 mmHg 이하인 segment만 허용한다. 이는 intra-measurement zeroing이나 transducer issue로 악화된 abnormal test subject를 제거하기 위한 조건이다. ABP &NIBP set은 ABP-20m과 NIBP-c의 교집합이다.

ABP-20m, NIBP-c, ABP &NIBP subset의 subject 수는 각각 629명, 104명, 86명이다. AAMI 기준은 test sample size가 최소 85명이어야 하므로, 세 subset 모두 이 기준을 만족한다. 특히 ABP &NIBP set은 86명으로 기준을 가까스로 넘는다.

혈압 분포를 보면 training set의 SBP와 DBP 평균 및 SD는 각각 111.84 ± 17.68 mmHg, 61.61 ± 11.04 mmHg이다. Whole test set은 SBP 112.07 ± 17.18 mmHg, DBP 61.72 ± 10.92 mmHg이다. 즉 전체적으로 평균 혈압은 수술 중 성인 환자 데이터로서 비교적 낮은 DBP와 중간 수준 SBP 분포를 보인다.

SDS 측면에서는 training set의 SBP SDS가 19.750 mmHg, DBP SDS가 11.748 mmHg로 높다. Whole test set의 SDS도 SBP 19.807 mmHg, DBP 11.627 mmHg로 매우 높다. ABP-20m subset의 SDS는 SBP 15.375 mmHg, DBP 8.745 mmHg이다. 저자들은 이러한 높은 SDS가 모델이 calibration BP에서 크게 변하는 target BP를 추정해야 하는 어려운 조건임을 보여준다고 설명한다.

### 4.2 PPG2BP-Net의 혈압 추정 성능

PPG2BP-Net은 2,978명의 training subjects로 학습되었고, 797명의 Whole test subjects와 세 subset에서 평가되었다. 평가 지표는 mean error, 즉 ME, SD of error, MAE이며, AAMI와 BHS 기준으로 성능을 판단하였다.

AAMI 기준은 test subject 수가 85명 이상이고, ME가 ±5 mmHg 이내이며, error SD가 8 mmHg 이하일 것을 요구한다. Whole test set에서는 DBP는 AAMI 기준을 만족하지만, SBP의 error SD가 10.263 mmHg로 8 mmHg를 초과하였다. NIBP-c set에서도 SBP error SD가 9.807 mmHg로 기준을 초과하였다. 반면 ABP-20m과 ABP &NIBP set에서는 SBP와 DBP 모두 AAMI 기준을 만족하였다.

ABP-20m set에서 PPG2BP-Net의 SBP error는 ME ± SD 기준으로 0.209 ± 7.509 mmHg였고, DBP error는 0.150 ± 4.549 mmHg였다. MAE는 SBP 5.525 mmHg, DBP 3.282 mmHg였다. 이 subset은 A-line insertion 이후 20분이 지난 안정적인 ABP waveform을 사용했다는 점에서 신뢰도가 높다고 논문은 해석한다.

ABP &NIBP set에서는 SBP error가 0.977 ± 6.969 mmHg, DBP error가 0.519 ± 4.379 mmHg였다. MAE는 SBP 5.238 mmHg, DBP 3.183 mmHg였다. 이 subset은 ABP-20m 조건과 NIBP consistency 조건을 모두 만족하는 subject들로 구성되므로, 가장 엄격하게 정제된 test subset으로 볼 수 있다. 이 결과는 AAMI 기준을 만족할 뿐 아니라, BHS Grade A도 모든 category에서 달성하였다.

Whole test set에서는 SBP error가 -0.231 ± 10.263 mmHg, MAE 7.991 mmHg였고, DBP error가 0.062 ± 6.252 mmHg, MAE 4.789 mmHg였다. DBP는 BHS Grade A를 달성했지만, SBP는 BHS 기준에서 각 error threshold별로 Grade D, C, C 수준이었다. 이는 전체 test set에는 A-line insertion 직후 불안정한 ABP waveform이나 transducer-related issue가 포함되었을 가능성이 있음을 시사한다.

NIBP-c set에서도 DBP는 좋은 성능을 보였지만, SBP는 SD 9.807 mmHg로 AAMI 기준을 만족하지 못했다. 논문은 Whole과 NIBP-c set의 SDS가 상대적으로 더 크다는 점에 주목하며, SBP estimation이 DBP estimation보다 더 어렵다고 설명한다. 실제로 모든 set에서 DBP error SD는 SBP보다 낮으며, BHS 기준에서도 DBP가 더 안정적으로 Grade A를 보인다.

BHS 기준은 error가 5 mmHg 이하, 10 mmHg 이하, 15 mmHg 이하인 비율을 기준으로 Grade A, B, C를 부여한다. ABP &NIBP set에서 SBP error는 5 mmHg 이하 60.0%, 10 mmHg 이하 85.9%, 15 mmHg 이하 95.6%로 모두 Grade A 조건을 만족한다. DBP error는 5 mmHg 이하 78.0%, 10 mmHg 이하 96.6%, 15 mmHg 이하 99.2%로 역시 Grade A이다. ABP-20m set에서는 SBP가 BHS 기준에서 Grade B 수준이지만, DBP는 Grade A이다.

### 4.3 비교 연구

논문은 기존 PPG-based cuffless BP estimation systems와 PPG2BP-Net을 비교하였다. 비교 대상에는 DBN-RBM, ANN, SVR, DTR, RFR, AdaboostR, CNN-LSTM, LRCN, CNN, RFPASN, Concat-CNN 등이 포함된다. 비교 표에는 modeling and experiment 방식, data source, training subject 수, validation/test subject 수, subject exclusiveness, SBP/DBP error가 정리되어 있다.

저자들은 일부 기존 연구가 AAMI 기준을 만족하지 못했으며, 일부 연구는 AAMI 기준을 만족하더라도 subject-dependent 방식으로 평가되어 실제 새로운 subject에 대한 성능이 보장되지 않는다고 지적한다. 또한 몇몇 연구는 training 및 validation subject 수가 매우 적어 결과가 misleading할 수 있다고 설명한다.

특히 2021년 CNN 기반 연구는 subject-independent 방식과 1,620명 규모의 UCI DB from MIMIC II dataset을 사용하여 비교적 유사한 설정으로 평가되었다. 하지만 저자들은 해당 데이터셋의 SDS가 SBP 7.509 mmHg, DBP 4.127 mmHg로 본 연구 데이터셋의 SDS, 즉 SBP 19.750 mmHg, DBP 11.748 mmHg보다 훨씬 낮다고 분석한다. 이는 그 데이터셋에서 subject 내부 BP variation이 상대적으로 낮았으며, calibration BP를 거의 그대로 유지하는 단순한 estimator도 SD error 기준을 쉽게 만족할 수 있음을 의미한다. 저자들은 이를 overqualified issue로 설명한다.

PPG2BP-Net은 subject-independent in-house operation room data에서 2,987 training subjects, 410 validation subjects, 797 Whole test subjects를 사용했다는 점에서 대규모 subject-independent 검증이라는 장점이 있다. 다만 Whole set의 SBP error SD는 AAMI 기준을 만족하지 못하므로, 논문의 가장 강한 성능 주장은 ABP-20m 및 ABP &NIBP subset에 기반한다는 점을 구분해야 한다.

### 4.4 주요 결과의 해석

이 논문의 결과는 몇 가지 중요한 의미를 가진다. 첫째, calibration-based PPG BP estimation에서 평가 데이터의 intrasubject BP variation을 반드시 고려해야 한다. 단순히 전체 BP SD가 높다고 해서 calibration 이후 혈압 변화가 큰 것은 아니며, calibration BP에서 target BP가 얼마나 벗어나는지가 실제 난이도에 더 중요하다. SDS는 이 문제를 정량화한다.

둘째, A-line insertion 직후의 ABP waveform reliability가 결과에 큰 영향을 줄 수 있다. Whole test set에서는 SBP error SD가 기준을 초과했지만, A-line insertion 이후 20분이 지난 ABP-20m set에서는 AAMI 기준을 만족하였다. 이는 ground truth ABP 자체의 신뢰도 관리가 cuffless BP estimation 평가에서 매우 중요하다는 점을 보여준다.

셋째, SBP estimation은 DBP estimation보다 어렵다. 모든 test set에서 SBP error SD와 MAE가 DBP보다 높으며, BHS 기준에서도 SBP의 grade가 더 낮다. 이는 SBP가 intraoperative intervention, vascular tone, cardiac output, pulse waveform 변화 등에 더 민감하기 때문일 수 있지만, 논문은 이를 생리학적으로 깊게 정량 분석하지는 않는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 대규모 subject-independent evaluation을 수행했다는 점이다. 4,185명의 clean independent subjects를 사용하고, training, validation, test subject를 완전히 분리한 것은 PPG 기반 BP estimation 연구에서 매우 중요한 설계이다. Subject-dependent split은 모델이 개인 고유 waveform을 기억하여 성능을 과대평가할 위험이 큰데, 이 논문은 실제 새로운 사용자에게 적용되는 상황을 더 가깝게 모사한다.

두 번째 강점은 high intrasubject BP variation을 명시적으로 평가했다는 점이다. 단순히 BP range나 전체 SD를 제시하는 데 그치지 않고, SDS라는 metric을 제안하여 calibration-based estimator의 난이도를 정량화하였다. 특히 기존 UCI DB from MIMIC II dataset의 SDS가 낮다는 분석을 통해, 일부 기존 연구의 좋은 성능이 실제 high intrasubject variation 상황에서는 재현되지 않을 수 있음을 지적한다. 이는 cuffless BP estimation 연구의 평가 방법론 측면에서 의미 있는 기여이다.

세 번째 강점은 calibration PPG와 target PPG를 비교하는 paired 1D-CNN 구조이다. 이 구조는 calibration-based BP estimation의 본질을 잘 반영한다. Calibration BP와 calibration PPG는 subject-specific baseline을 제공하고, target PPG는 현재 상태를 반영한다. 두 waveform feature와 그 차이를 함께 사용하면, 단순한 absolute BP prediction보다 calibration 대비 변화량을 더 잘 학습할 수 있다.

네 번째 강점은 데이터 전처리와 test subset 구성이 비교적 현실적이고 신중하다는 점이다. 저자들은 A-line insertion 직후 ABP waveform이 불안정할 수 있다는 점을 고려하여 ABP-20m subset을 만들었고, NIBP와 A-line BP 차이가 큰 segment를 제거하는 NIBP-c subset도 구성하였다. 이는 ground truth ABP의 품질이 모델 평가에 영향을 줄 수 있음을 고려한 설계이다.

다섯 번째 강점은 PPG만을 사용하면서도 cuffless BP monitoring의 실제 device scenario를 염두에 두었다는 점이다. ECG나 다중 sensor를 요구하지 않으므로 wearable ring, watch, wristlet 등 단일 optical sensor 기반 장치로 확장할 가능성이 있다. 논문은 24-hour continuous measurement, BP variability assessment, nocturnal BP monitoring 등 실제 서비스 가능성을 언급한다.

그러나 한계도 명확하다. 첫째, 모델은 calibration-based 방식이다. 즉 초기 calibration PPG, SBP, DBP가 필요하다. 논문은 re-calibration을 사용하지 않았다는 점에서 C3 조건을 고려했지만, 실제 사용자가 calibration을 어떻게 수행해야 하는지, calibration BP를 어떤 certified cuff-based device로 얻을지, calibration 오류가 이후 추정에 얼마나 영향을 미치는지는 충분히 분석되지 않았다.

둘째, 예측 가능한 SBP 범위가 전처리 조건 T4에 의해 $70 \le average\ SBP \le 180$ mmHg로 제한된다. 논문은 이 범위를 확장하려면 70 mmHg 미만과 180 mmHg 초과의 충분하고 신뢰할 수 있는 데이터를 추가 학습해야 한다고 명시한다. 따라서 extreme hypotension이나 severe hypertension 상황에서 모델의 성능은 제공된 텍스트만으로는 알 수 없다.

셋째, 가장 강한 성능 결과는 정제된 subset에서 나온다. Whole test set에서는 SBP error SD가 10.263 mmHg로 AAMI 기준을 만족하지 못한다. ABP-20m과 ABP &NIBP subset에서는 기준을 만족하지만, 이는 A-line insertion 이후 안정화된 segment 또는 NIBP와의 consistency 조건을 만족하는 subset이다. 따라서 전체 수술실 데이터의 모든 상황에서 모델이 AAMI 기준을 만족한다고 해석해서는 안 된다.

넷째, 데이터는 수술실 intraoperative setting에서 수집되었다. 환자들은 anesthetization 이후 또는 surgery before/around context의 physiological state를 가질 수 있으며, 일반적인 일상생활 wearable monitoring 환경과는 다르다. 논문은 C3A cuffless device와 daily user 적용 가능성을 논의하지만, 실제 ambulatory environment에서의 motion artifact, sensor displacement, exercise, sleep posture, skin contact 변화에 대한 직접 검증은 제공되지 않았다.

다섯째, PPG2BP-Net의 code는 private asset으로 공개되지 않는다. 데이터 일부는 VitalDB에서 접근 가능하다고 하지만, 논문에 사용된 전체 pipeline과 code가 완전히 공개되지 않는다면 외부 연구자가 결과를 재현하기 어렵다. 의료 AI 연구에서는 reproducibility가 중요한데, 이 부분은 한계로 볼 수 있다.

여섯째, subject-independent split은 강점이지만, 같은 병원, 같은 수술실 환경, 유사한 장비 조건에서 수집된 in-house data라는 점도 고려해야 한다. 다른 병원, 다른 PPG sensor, 다른 sampling condition, 다른 patient population에서 동일 성능이 유지되는지는 아직 명확하지 않다. 논문도 clinical test를 통한 추가 verification이 fidelity를 높일 것이라고 언급한다.

마지막으로, 모델이 어떤 PPG waveform feature를 실제로 BP estimation에 사용했는지에 대한 explainability 분석은 부족하다. CNN 기반 모델은 feature extraction을 자동으로 수행하지만, 의료 응용에서는 어떤 waveform pattern이 SBP 또는 DBP 변화와 연결되는지 이해하는 것이 중요하다. 본 논문은 architecture와 성능 검증에 집중하며, learned feature의 생리학적 해석은 제한적이다.

## 6. 결론

이 논문은 PPG 기반 cuffless continuous BP monitoring에서 subject-independent evaluation과 high intrasubject BP variation 문제를 본격적으로 다룬 연구이다. 제안한 PPG2BP-Net은 calibration PPG와 target PPG를 비교하는 paired 1D-CNN 구조, calibration SBP/DBP를 처리하는 MLP, 그리고 최종 FCL을 결합하여 target SBP와 DBP를 추정한다. 이 구조는 calibration-based BP estimation에서 중요한 “baseline 대비 변화”를 직접 학습하려는 설계로 이해할 수 있다.

논문의 주요 기여는 세 가지로 요약된다. 첫째, 25,779 surgical cases에서 전처리한 4,185명의 clean independent subjects를 사용하여 대규모 subject-independent 학습 및 holdout test를 수행하였다. 둘째, calibration-based estimator의 실제 난이도를 평가하기 위해 SDS metric을 제안하였다. 셋째, ABP-20m 및 ABP &NIBP test subset에서 AAMI 기준과 BHS 기준을 만족하는 성능을 보였다.

정량적으로 ABP-20m set에서 PPG2BP-Net은 SBP error 0.209 ± 7.509 mmHg, DBP error 0.150 ± 4.549 mmHg를 달성하였다. ABP &NIBP set에서는 SBP error 0.977 ± 6.969 mmHg, DBP error 0.519 ± 4.379 mmHg를 달성했으며, 이 subset에서는 BHS Grade A도 모든 category에서 만족하였다. 특히 ABP-20m set의 SDS가 SBP 15.375 mmHg, DBP 8.745 mmHg로 높다는 점에서, calibration BP로부터 비교적 큰 혈압 변화가 있는 조건에서도 유의미한 성능을 보였다고 평가할 수 있다.

이 연구는 실제 cuffless BP monitoring device 개발에서 평가 프로토콜이 얼마나 중요한지를 잘 보여준다. 단순히 낮은 MAE나 SD error만 제시하는 것이 아니라, subject-independent split, 충분한 subject 수, calibration 이후 intrasubject variation, ground truth ABP의 안정성까지 고려해야 한다는 점을 강조한다. 이러한 관점은 향후 PPG 기반 BP estimation 연구의 benchmark 설계에도 중요한 시사점을 제공한다.

다만 실제 일상 wearable device로 확장하기 위해서는 추가 검증이 필요하다. 모델은 초기 calibration을 필요로 하며, 예측 가능한 BP range가 제한되어 있고, 전체 test set에서는 SBP가 AAMI 기준을 만족하지 못했다. 또한 데이터는 수술실 환경에서 수집되었기 때문에 motion artifact가 많은 ambulatory setting, 다양한 sensor, 장기간 home monitoring 환경에서의 성능은 아직 확인되지 않았다.

종합하면, PPG2BP-Net은 PPG 기반 cuffless BP estimation 연구에서 한 단계 더 엄격한 평가 조건을 제시하고, calibration-based paired CNN 구조와 SDS metric을 통해 high intrasubject BP variation 문제를 정면으로 다룬 의미 있는 연구이다. 이 접근은 향후 C3A cuffless BP estimation device, 24-hour BP monitoring, nocturnal BP monitoring, BP variability assessment, cardiovascular event early prediction으로 발전할 가능성이 있다.
