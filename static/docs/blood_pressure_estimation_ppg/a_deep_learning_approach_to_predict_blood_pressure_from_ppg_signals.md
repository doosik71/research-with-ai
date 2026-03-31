# A Deep Learning Approach to Predict Blood Pressure from PPG Signals

Ali Tazarv, Marco Levorato

## 🧩 Problem to Solve

- 고혈압은 전 세계적으로 주요 사망 원인 중 하나이며, 뇌졸중, 심근경색, 심부전 등 다양한 심혈관 질환의 위험을 증가시킵니다.
- 기존의 커프(cuff) 기반 혈압(BP) 측정 장치는 지속적인 모니터링에 비용이 많이 들고 불편하며, 특히 일상생활 환경에서는 사용하기 어렵습니다.
- 따라서 비침습적이고 저렴하며 지속적으로 혈압을 모니터링할 수 있는 대안적인 방법의 개발이 필요합니다.
- 기존 광혈류측정(PPG) 신호 기반 BP 추정 방법들은 주로 사전 정의된 특징(predefined features)에 의존하여, 원시 PPG 신호 내의 중요한 정보(특히 피험자별(subject-specific) 패턴)를 놓치거나 PPG의 시계열(time-series) 특성을 충분히 활용하지 못하는 한계가 있습니다.

## ✨ Key Contributions

- **고급 개인화된 데이터 기반 접근 방식 제안**: 3계층 심층 신경망을 사용하여 PPG 신호에서 BP를 추정하는 개인화된 데이터 기반 접근 방식을 제시합니다.
- **자동 특징 추출(Automatic Feature Extraction)**: Convolutional Neural Network (CNN) 계층을 사용하여 PPG 신호에서 BP 추정에 가장 중요한 특징을 비선형적이고 효율적인 방식으로 자동으로 학습하고 추출합니다. 이는 수동 특징 추출 방식의 한계를 극복합니다.
- **시간적 의존성 포착(Temporal Dependency Capture)**: 추출된 특징들 사이의 장기적인 시간적 상호 의존성을 원활하고 자동화된 방식으로 포착하기 위해 Long-Short Term Memory (LSTM) 모델을 채택합니다.
- **최첨단 성능 달성**: 두 가지 표준 병원 데이터셋(MIMIC-II 및 UQVSD)에서 수축기 혈압(SBP) 및 이완기 혈압(DBP) 값에 대해 이전 연구들을 능가하는 낮은 평균 절대 오차(MAE)와 절대 오차 표준 편차(SDAE)를 달성했습니다.
- **임상 표준 충족**: British Hypertension Society (BHS) 표준에서 SBP와 DBP 모두 **Grade A**를 획득했으며, Association for the Advancement of Medical Instrumentation (AAMI) 표준도 통과했습니다.

## 📎 Related Works

- **PPG 신호 기반 BP 추정**: PPG 신호와 대동맥 압력(Aortic Pressure) 파형 간의 유사성 및 상관관계에 대한 연구가 활발히 진행되었으며, PPG 신호로부터 SBP 및 DBP를 추정하는 다양한 방법들이 제안되었습니다.
- **맥파 전달 시간(Pulse Transit Time, PTT) 기반 방법**: [6], [7], [8] 등의 연구에서는 심장에서 PPG 신호가 기록되는 손목까지 혈압파가 전달되는 시간인 PTT를 사용하여 BP를 추정합니다. 이는 심전도(ECG)와 PPG 신호 간의 시간차를 측정합니다.
- **전체 신호 기반 특징 추출 및 회귀**: [9]에서는 PPG 신호의 한 주기 내 모든 샘플 포인트를 개별 특징으로 간주하고, 주성분 분석(PCA)을 통해 특징 수를 줄인 후 회귀 알고리즘을 적용했습니다.
- **사전 정의된 특징 및 오토인코더**: [10]에서는 PPG 신호의 특정 특징(예: 곡선 아래 면적, 특정 지점의 시간 길이)을 수동으로 정의하고, 오토인코더를 사용하여 이러한 특징을 압축한 다음 피드포워드 신경망으로 BP를 추정했습니다.
- **기존 방법의 한계점**:
  - 대부분의 기존 방법은 사전 정의된 특징에 의존하여, BP 추정에 유용할 수 있는 신호 내의 미묘한 세부 정보(피험자마다 다를 수 있는 패턴 포함)를 놓칠 수 있습니다.
  - PCA와 같은 차원 축소 절차는 BP 추정 작업에 최적화된 것이 아니라 일반적인 데이터 압축 방법입니다.
  - 이러한 접근 방식은 PPG 신호의 시계열 특성을 효과적으로 활용하지 못합니다.

## 🛠️ Methodology

제안된 방법은 각 피험자에 대해 BP 추정에 최적화된 특징을 자동으로 추출하고, 고급 신경망 아키텍처를 사용하여 데이터의 시간적 종속성을 포착합니다.

1. **신호 전처리 (Signal Pre-processing)**:
   - **필터링**: PPG 신호에 $0.1$~$8 \text{Hz}$ 대역 통과 필터를 적용하여 고주파 노이즈와 DC 오프셋을 제거합니다. 대동맥 압력(ABP) 신호에는 $5 \text{Hz}$ 저역 통과 필터를 적용하여 급격한 피크를 제거합니다.
   - **윈도우 분할**: PPG 및 ABP 신호를 $8 \text{s}$ 길이의 윈도우로 분할하며, $2 \text{s}$ 간격으로 이동하여 $6 \text{s}$의 오버랩을 생성합니다.
   - **정규화**: 각 PPG 윈도우 신호를 평균 $0$, 단위 분산으로 스케일링하여 모델 입력으로 사용합니다.
   - **BP 값 추출**: 각 ABP 윈도우에서 최대값을 SBP, 최소값을 DBP로 해석합니다.
2. **딥러닝 모델 (Deep Learning Model)**: Convolutional Neural Network (CNN), Long-Short Term Memory (LSTM) 네트워크, Multi-Layer Perceptron (MLP)으로 구성된 3계층 심층 신경망을 사용합니다.
   - **CNN 계층 (자동 특징 추출)**:
     - 필터 크기 $15$의 1-D 필터링 작업으로 시작합니다.
     - 활성화 함수로 Rectified Linear Unit (RELU)을 사용합니다.
     - 배치 정규화(Batch Normalization) 계층을 거칩니다.
     - 풀링 크기 $s_P = 4$인 Max-Pooling 계층을 통해 특징을 압축하고 과적합을 방지합니다.
     - $0.1$의 드롭아웃(Dropout) 비율을 가진 드롭아웃 계층을 적용하여 과적합을 추가로 방지합니다. 이 계층은 BP 추정에 가장 유익한 특징을 학습하고 추출하는 역할을 합니다.
   - **LSTM 계층 (시계열 분석)**:
     - 두 개의 동일한 LSTM 모듈을 직렬로 연결하여 구성됩니다.
     - 각 LSTM 모듈은 $64$개의 유닛($n_U$)을 가집니다.
     - 은닉 상태 및 출력 데이터에는 $tanh$를, 망각, 입력, 출력 게이트에는 $hard\text{-}sigmoid$를 활성화 함수로 사용합니다. 이 계층은 입력 데이터의 장기적인 시간 의존성을 학습하고 기울기 소실 문제를 해결합니다.
   - **MLP 계층 (최종 예측)**: LSTM 계층의 출력을 받아 SBP 및 DBP 값을 예측하는 데 사용됩니다.
3. **훈련 및 평가**:
   - **구현**: Keras 2.2.4와 Tensorflow 1.3.1 백엔드를 사용하여 Python 3.6.3 환경에서 구현되었습니다.
   - **최적화**: Adam 옵티마이저를 사용하며, 배치 크기는 $20$으로 설정합니다. 모든 하이퍼파라미터는 그리드 탐색(grid search)을 통해 최적화되었습니다.
   - **검증 전략**: Leave-one-window-out 검증 방식을 사용합니다. $8 \text{s}$ 시간 윈도우 하나를 테스트 샘플로 사용하고, 나머지 데이터로 모델을 훈련합니다. $6 \text{s}$의 윈도우 오버랩을 고려하여 테스트 세트 양쪽의 세 시간 윈도우는 훈련에 사용하지 않아 훈련 데이터와 테스트 데이터의 완전한 분리를 보장합니다.
   - **개인화된 모델**: 각 피험자에 대해 모델을 독립적으로 훈련하고 테스트하여 개인화된 예측을 수행합니다.

## 📊 Results

제안된 모델은 MIMIC-II 및 UQVSD 두 가지 데이터셋에서 이전 연구들을 능가하는 뛰어난 성능을 입증했습니다.

1. **평균 절대 오차 (MAE) 및 절대 오차 표준 편차 (SDAE)**:

   - **MIMIC-II 데이터셋 (20명)**:
     - SBP: $\text{MAE} = 3.70 \text{mmHg}$, $\text{SDAE} = 3.07 \text{mmHg}$
     - DBP: $\text{MAE} = 2.02 \text{mmHg}$, $\text{SDAE} = 1.76 \text{mmHg}$
   - **UQVSD 데이터셋 (49개 측정)**:
     - SBP: $\text{MAE} = 3.91 \text{mmHg}$, $\text{SDAE} = 4.78 \text{mmHg}$
     - DBP: $\text{MAE} = 1.99 \text{mmHg}$, $\text{SDAE} = 2.45 \text{mmHg}$
   - 이러한 결과는 비슷한 데이터셋을 사용한 기존 방법들보다 우수합니다.

2. **BHS (British Hypertension Society) 표준**:

   - 제안된 모델은 SBP와 DBP 모두에서 **Grade A**를 획득했습니다. 이는 PPG 신호만을 사용하여 BP를 예측하는 방법 중 최초로 두 지표 모두에서 Grade A를 달성한 것으로, [9] (DBP Grade A, SBP Grade C 미만) 및 [24] (DBP Grade A, SBP Grade B) 등 이전 연구를 능가합니다.
   - **MIMIC-II**: SBP ($\le 5 \text{mmHg} - 77\%(\text{A})$, $\le 10 \text{mmHg} - 92\%(\text{A})$, $\le 15 \text{mmHg} - 96\%(\text{A})$), DBP ($\le 5 \text{mmHg} - 93\%(\text{A})$, $\le 10 \text{mmHg} - 97\%(\text{A})$, $\le 15 \text{mmHg} - 99\%(\text{A})$)
   - **UQVSD**: SBP ($\le 5 \text{mmHg} - 75\%(\text{A})$, $\le 10 \text{mmHg} - 92\%(\text{A})$, $\le 15 \text{mmHg} - 96\%(\text{A})$), DBP ($\le 5 \text{mmHg} - 92\%(\text{A})$, $\le 10 \text{mmHg} - 98\%(\text{A})$, $\le 15 \text{mmHg} - 99\%(\text{A})$)

3. **AAMI (Association for the Advancement of Medical Instrumentation) 표준**:

   - 평균 오차(ME)는 $5 \text{mmHg}$ 미만, 오차 표준 편차(SD)는 $8 \text{mmHg}$ 미만이어야 하는 AAMI 표준 기준을 SBP와 DBP 모두에서 충족합니다.
   - **MIMIC-II**: SBP ($\text{ME}=0.21$, $\sigma=6.27$), DBP ($\text{ME}=0.24$, $\sigma=3.40$)
   - **UQVSD**: SBP ($\text{ME}=0.52$, $\sigma=6.16$), DBP ($\text{ME}=0.20$, $\sigma=3.15$)
   - UQVSD 데이터에서 [24]와 비교하여 더 나은 오차 값(더 낮은 평균 및 $\sigma$)을 보입니다.

4. **오차 분포 및 Bland-Altman 플롯**:
   - 예측 오차는 대부분 작았으며, SBP의 경우 $20 \text{mmHg}$ 이상 벗어나는 경우는 $1\%$ 미만이었습니다. DBP의 경우 이러한 극단적인 오차는 발생하지 않았습니다.
   - Bland-Altman 플롯은 BP 값이 낮을수록(덜 활동적일 때) 예측의 신뢰도가 더 높음을 보여주며, 이는 PPG 신호의 노이즈가 적을 때 더 정확하다는 점을 시사합니다.

## 🧠 Insights & Discussion

- **비침습적이고 연속적인 BP 모니터링의 실현 가능성**: 이 연구는 단일 채널 PPG 신호만을 사용하여 고정밀 혈압 추정이 가능함을 입증하여, 기존 커프 기반 장치의 한계를 극복하고 일상생활에서의 연속적인 BP 모니터링의 중요한 발전을 제시합니다.
- **딥러닝 아키텍처의 효율성**: CNN을 통한 최적 특징의 자동 추출과 LSTM을 통한 시간적 종속성 학습의 조합이 기존 방법론보다 훨씬 뛰어난 성능을 이끌어냈습니다. 이는 복잡한 생체 신호 분석에 딥러닝이 효과적임을 보여줍니다.
- **개인화된 모델의 중요성**: 각 피험자에 대해 독립적으로 모델을 훈련함으로써, 모델이 피험자별로 다른 미묘한 생체학적 특성을 학습하고 이를 통해 예측 정확도를 높일 수 있음을 시사합니다.
- **실용적인 적용 가능성**: PPG 센서는 소형, 저비용이며 스마트워치와 같은 최신 웨어러블 기기에 이미 통합되어 있어, 제안된 방법은 상업용 제품 및 의료 응용 분야에 즉시 적용될 수 있는 높은 잠재력을 가지고 있습니다.
- **미래 연구 과제**: 모델 실행 및 지속적인 업데이트에 필요한 처리 능력은 IoT 아키텍처의 엣지(edge) 단에서 처리하기에는 부담이 될 수 있습니다. 따라서 원시 PPG 데이터를 클라우드로 전송하고, 클라우드에서 예측 결과를 사용자 기기로 다시 보내는 실시간 클라우드 기반 아키텍처의 개발이 필요합니다.

## 📌 TL;DR

이 논문은 고혈압 관리의 핵심인 지속적인 혈압(BP) 모니터링을 위해 단일 채널 광혈류측정(PPG) 신호를 활용하는 딥러닝 기반의 혁신적인 접근 방식을 제안합니다. Convolutional Neural Network (CNN)를 통해 PPG 신호에서 BP 예측에 최적화된 특징을 자동으로 추출하고, Long-Short Term Memory (LSTM) 네트워크를 사용하여 이러한 특징 간의 시간적 상관관계를 정교하게 포착합니다. MIMIC-II 및 UQVSD 두 가지 표준 병원 데이터셋을 사용한 실험 결과, 제안된 모델은 이전 연구들 대비 현저히 낮은 MAE와 SDAE를 달성했으며, BHS 표준에서 SBP와 DBP 모두 Grade A를 획득하고 AAMI 표준을 충족하는 최첨단 성능을 입증했습니다. 이는 웨어러블 기기를 통한 비침습적이고 고정밀 혈압 모니터링의 상업적 적용 가능성을 강력하게 시사합니다.
