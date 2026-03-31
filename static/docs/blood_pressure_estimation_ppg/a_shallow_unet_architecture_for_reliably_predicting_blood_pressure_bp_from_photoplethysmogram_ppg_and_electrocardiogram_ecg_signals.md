# A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals

Sakib Mahmud, Nabil Ibtehaz, Amith Khandakar, Anas Tahir, Tawsifur Rahman, Khandaker Reajul Islam, Md Shafayet Hossain, M. Sohel Rahman, Mohammad Tariqul Islam, Muhammad E. H. Chowdhury

## 🧩 해결하고자 하는 문제

* 심혈관 질환은 전 세계 사망 원인의 주요인이며, 이를 진단하고 치료하기 위해 지속적인 혈압(BP) 모니터링이 필수적입니다.
* 현재 병원에서 사용되는 대부분의 연속 혈압 측정 방식은 침습적이거나, 비침습적인 커프(cuff) 기반 방식은 연속 모니터링에 부적합합니다.
* 광혈류측정(PPG) 및 심전도(ECG)와 같은 비침습적으로 수집 가능한 신호를 활용하여 혈압을 예측하려는 노력이 있었지만, 딥러닝 모델을 사용한 혈압 예측 성능에는 여전히 개선의 여지가 있으며, 특히 대규모 데이터셋에서 강력하고 경량의 모델을 통해 수축기 혈압(SBP)과 이완기 혈압(DBP)을 정확하게 예측하는 것이 필요합니다.

## ✨ 주요 기여

* **얕은 U-Net 기반 특징 추출:** PPG 및 ECG 신호에서 혈압 예측에 최적화된 잠재 특징을 추출하기 위해 매우 얕은(1-레벨) 1D U-Net 인코더를 오토인코더로 활용하는 새로운 파이프라인을 제안했습니다.
* **최고 수준의 성능 달성:** MIMIC-II 대규모 데이터셋(UCI Dataset)에서 SBP와 DBP 예측 모두 British Hypertension Society (BHS) Grade A 기준을 충족하며, 기존 연구 대비 최고 수준의 평균 절대 오차(MAE)를 달성했습니다. (독립 테스트 세트: SBP $\text{MAE} = 2.333 \text{ mmHg}$, DBP $\text{MAE} = 0.713 \text{ mmHg}$)
* **경량 모델:** 제안된 얕은 U-Net 모델은 약 0.55M의 매개변수를 가지는 경량 모델로, 컴퓨팅 및 메모리 자원이 제한된 환경(예: 웨어러블 기기)에 배포하기에 적합합니다.
* **우수한 일반화 성능:** MIMIC-II 데이터셋으로 학습된 모델이 외부 BCG 데이터셋에서도 높은 성능을 유지하며 뛰어난 일반화 능력을 입증했습니다. (외부 데이터셋: SBP $\text{MAE} = 2.728 \text{ mmHg}$, DBP $\text{MAE} = 1.166 \text{ mmHg}$)
* **ECG 대체 가능성:** ECG 신호 없이도 PPG 및 PPG의 1차, 2차 미분값(VPG, APG)만을 사용하여 유사한 높은 성능을 달성할 수 있음을 보여주어 하드웨어 복잡성을 줄일 수 있는 가능성을 제시했습니다.

## 📎 관련 연구

* **전통적인 머신러닝 (ML):** Support Vector Regressor (SVR), Adaptive Boosting (AdaBoost), Random Forest, Gradient Boosting (GradBoost), Gaussian Process Regression (GPR), Artificial Neural Network (ANN), Recurrent Neural Network (RNN) 기반 Long Short-Term Memory (LSTM) 등이 PPG 단독 또는 PPG와 ECG 조합으로 혈압 예측에 사용되었습니다.
* **딥러닝 기반 접근 방식:**
  * **CNN 기반:** Spectro-Temporal ResNets [26]은 PPG 신호와 그 파생물의 스펙트로그램을 사용하여 혈압을 예측했습니다.
  * **U-Net 활용:** U-Net 아키텍처는 PPG를 동맥 혈압(ABP) 파형으로 변환하는 signal-to-signal 변환에 사용되거나(Athaya et al. [27], Ibtehaz et al. [13]), 의료 영상 분할과 같은 다양한 1D, 2D, 3D 문제에 적용되었습니다.
  * **특징 추출:** 일반 CNN으로 PPG에서 특징을 추출한 후 LSTM 모델을 사용하여 혈압을 예측하는 연구(Esmaelpoor et al. [40])도 있었습니다. 본 연구는 U-Net의 인코더 부분을 특징 추출에 활용하여 성능을 개선합니다.

## 🛠️ 방법론

* **데이터셋:**
  * **MIMIC-II 데이터셋 (UCI Repository):** 942명의 환자로부터 얻은 PPG, ABP, ECG 신호 12,000개 인스턴스 (샘플링 레이트 125 Hz).
  * **BCG 데이터셋 (외부 검증):** 40명의 피험자로부터 얻은 BCG, ECG, PPG, ABP 신호 (샘플링 레이트 1000 Hz, 125 Hz로 다운샘플링).
* **데이터 전처리:**
  * **신호 분할:** 1024개 샘플 길이로 신호 분할.
  * **기준선 표류 보정:** MATLAB의 `movmin`, `polyfit`, `polyval` 함수를 사용하여 기준선 표류 제거.
  * **정규화:** PPG 및 ECG는 각 신호별로 0~1 범위로 정규화, ABP는 전체 데이터셋의 최소-최대값을 사용하여 전역적으로 Min-Max 정규화.
  * **PPG 미분값:** PPG의 1차(VPG) 및 2차(APG) 미분값을 계산하고, 필터를 적용하여 고주파 왜곡을 제거하며 지연을 보정.
  * **품질 낮은 신호 제거:** 극단적인 혈압 값, 비정상적인 혈압 범위, 이중 피크 또는 비균일한 피크를 가진 왜곡된 신호를 제거 (약 25% 제거).
* **혈압 예측 파이프라인:** U-Net 기반 오토인코더를 통한 특징 추출 단계와 머신러닝 기반 회귀 분석 단계로 구성.
  * **특징 추출 (U-Net 기반 오토인코더):**
    * 입력 신호 (PPG, ECG, VPG, APG 조합)를 받아 ABP 파형을 출력하도록 U-Net을 훈련하여 네트워크가 PPG 및 ECG 신호의 패턴을 ABP 파형에 매핑하는 특징 공간을 학습하도록 유도합니다.
    * U-Net의 인코더 부분 끝에 완전 연결(Dense) MLP 계층을 추가하여 특징을 추출합니다.
    * 훈련 매개변수: 배치 크기 = 64, 에포크 수 = 100, 조기 종료 = 15, 손실 함수 = 평균 제곱 오차(MSE), 최적화 도구 = Adam.
  * **회귀 분석 (전통적 ML 기법):** 추출된 특징을 사용하여 혈압(SBP, DBP)을 예측하기 위해 MLP, SGD, SVR, XgBoost, GradBoost, AdaBoost, K-Nearest Neighbor, Random Forest 등의 알고리즘을 사용합니다.
* **실험 설정:**
  * **실험 1 (UCI 데이터셋 내 훈련 및 테스트):** U-Net 인코더의 깊이(1-4), 폭(32-256), 커널 크기(3-11), 입력 채널 수(1-4개 조합)를 변화시키며 최적 아키텍처를 탐색하고, 다양한 회귀 기법을 비교합니다.
  * **실험 2 (외부 BCG 데이터셋 검증):** UCI 데이터셋으로 훈련된 모델을 BCG 데이터셋에 테스트하고, BCG 데이터셋에 대한 5-Fold 교차 검증을 수행합니다.
* **평가 지표:**
  * **주요 지표:** 평균 절대 오차(MAE).
  * **임상 표준:** British Hypertension Society (BHS) 표준 (Grade A, B, C) 및 Association for the Advancement of Medical Instrumentation (AAMI) 표준.
  * **고혈압 분류:** Normotension, Prehypertension, Hypertension에 대한 Precision, Recall, F1-Score.
  * **통계 분석:** 선형 회귀, Pearson 상관 계수(PCC), Bland-Altman 플롯.

## 📊 결과

* **최적 U-Net 아키텍처:**
  * **깊이:** 가장 얕은 **1-레벨** 인코더가 최상의 성능을 보였습니다. 깊이가 깊어질수록 성능이 저하되었습니다.
  * **폭 및 특징 수:** 인코더 폭 **128**, 추출된 특징 수 **1024개**일 때 최적의 성능을 달성했습니다.
  * **커널 크기:** 커널 크기 **3**일 때 가장 좋은 성능을 보였습니다.
  * **입력 채널 수:** PPG, VPG, APG, ECG **4개 채널**을 모두 사용했을 때 가장 우수했습니다 (SBP $\text{MAE} = 2.333$, DBP $\text{MAE} = 0.713$). PPG와 그 미분값(VPG, APG) 3개 채널만 사용해도 ECG를 포함한 2개 채널(PPG, ECG)과 유사하거나 더 나은 성능을 보였습니다.
* **회귀 기법:** Multi-Layer Perceptron (MLP)이 다른 전통적인 머신러닝 기법보다 뛰어난 성능을 보였습니다.
* **BHS 및 AAMI 표준:**
  * UCI 데이터셋에서 SBP 및 DBP 예측 모두 **BHS Grade A**를 달성했으며, DBP는 거의 100%의 예측이 Grade A 기준을 충족했습니다.
  * AAMI 표준도 SBP 평균 오차 0.09 mmHg, 표준 편차 0.94 mmHg; DBP 평균 오차 -0.019 mmHg, 표준 편차 2.876 mmHg로 기준을 크게 상회하며 충족했습니다.
* **고혈압 분류:** DBP 예측의 전반적인 정확도는 98.95%, SBP는 94.14%였습니다. Normotension 그룹에서 가장 높은 분류 성능을 보였습니다.
* **통계 분석:** SBP 및 DBP 예측은 실측값(ground truths)과 각각 0.991 및 0.996의 높은 Pearson 상관 계수를 보였습니다. Bland-Altman 플롯에서도 대부분의 오차는 5 mmHg 범위 내에 있었고, 오차 크기는 혈압 범위에 걸쳐 일정하게 유지되었습니다.
* **외부 데이터셋 검증 (BCG):** UCI 데이터셋으로 훈련된 모델이 BCG 데이터셋에서 SBP $\text{MAE} = 2.728 \text{ mmHg}$, DBP $\text{MAE} = 1.166 \text{ mmHg}$를 기록하며 우수한 일반화 성능을 입증했습니다.

## 🧠 통찰 및 논의

* 본 연구는 오토인코더로 활용된 U-Net 아키텍처가 PPG 및 ECG 신호에서 혈압 예측에 최적화된 특징을 효율적으로 추출할 수 있음을 입증했습니다. 특히, U-Net의 가장 얕은 버전이 대규모 MIMIC-II 데이터셋에서 최고 수준의 SBP 및 DBP 예측 성능을 달성할 수 있음을 보여주었습니다.
* 제안된 모델은 약 0.55M의 매개변수로 매우 경량이며, 이는 컴퓨팅 및 메모리 자원이 제한된 웨어러블 장치와 같은 환경에 배포하는 데 매우 적합합니다.
* 외부 BCG 데이터셋에 대한 성공적인 일반화 능력은 대규모 일반 데이터셋을 활용한 얕은 오토인코더 기반 특징 추출 방식의 견고함을 보여줍니다.
* ECG 신호 없이 PPG와 그 미분값(VPG, APG)만으로도 유사하게 높은 성능을 달성할 수 있다는 점은 하드웨어 설계 및 장치 구현을 간소화할 수 있는 중요한 통찰을 제공하며, ECG 센서가 없는 단일 PPG 센서 기반의 혈압 모니터링 장치 개발 가능성을 열어줍니다.
* 현재 모델은 움직임 인공물(motion artifacts) 처리를 염두에 두고 설계되지 않아 웨어러블 장치에 직접 적용하기에는 어려움이 있을 수 있습니다. 그러나 움직임 인공물 보정 기술이 적용된다면, 원격 모니터링 서버나 모바일 애플리케이션에서 실시간 비침습 혈압 모니터링을 위한 강력한 솔루션이 될 수 있습니다.

## 📌 요약 (TL;DR)

심혈관 질환 진단에 필수적인 혈압의 연속적, 비침습적 모니터링을 위해, 본 연구는 PPG 및 ECG 신호로부터 혈압(SBP 및 DBP)을 정확하게 예측하는 경량 딥러닝 파이프라인을 제안합니다. 핵심은 **얕은(1-레벨) 1D U-Net 인코더를 오토인코더로 사용하여** PPG 및 ECG 신호에서 **최적의 잠재 특징을 추출**하는 것입니다. 추출된 특징은 MLP 회귀 모델에 입력되어 혈압을 예측합니다. MIMIC-II 대규모 데이터셋에서 훈련 및 테스트한 결과, SBP $\text{MAE} = 2.333 \text{ mmHg}$, DBP $\text{MAE} = 0.713 \text{ mmHg}$로 **BHS Grade A**를 달성했으며, 이는 **기존 최고 성능을 뛰어넘는 결과**입니다. 모델은 약 0.55M의 매개변수로 **매우 경량**이며, 외부 데이터셋에서도 우수한 일반화 성능을 보였습니다. 또한, ECG 신호 없이 PPG와 그 미분값만으로도 유사한 성능을 달성할 수 있어 하드웨어 간소화 가능성을 제시합니다. 이 파이프라인은 리소스 제약이 있는 환경에서의 비침습적 혈압 모니터링에 적합합니다.
