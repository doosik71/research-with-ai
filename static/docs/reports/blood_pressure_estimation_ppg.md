# Blood Pressure Estimation using PPG

## 서론

### 1. 연구 배경

본 보고서는 PPG(Photoplethysmogram) 신호를 기반으로 동맥혈압을 추정하는 연구들을 체계적으로 분석한다. PPG 신호를 활용한 혈압 추정 기술은 커프 기반 측정의 불편함과 침습적 방법을 극복하여 비침습적이고 연속적인 모니터링을 가능하게 한다. 분석 대상 논문들은 PPG 단독 입력, PPG+ECG 혼합 입력, 파형 변환, 머신러닝/딥러닝 회귀, LLM 기반 접근 방식 등 다양한 방식으로 혈압 추정 문제를 해결해 왔으며, 입력 신호 구성, 모델 아키텍처, 학습 패러다임, 배포 관점 등 4 차원에서 분류된다. 연구의 흐름은 PPG 단독 입력에서 PPG+ECG 혼합 입력으로, 전통적 머신러닝에서 딥러닝과 LLM 으로 진화하고 있으며, 종단 간 모델 개발과 프라이버시 보호, 엣지 디바이스 배포, 개인화 학습 등 실세계 적용을 위한 기술적 고민이 지속적으로 반영되고 있다.

### 2. 문제의식 및 분석 필요성

PPG 기반 혈압 추정 분야에서 다양한 방법론과 모델 아키텍처가 제안되었지만, 연구 체계가 산재해 있어 비교·분석이 필요하다. 입력 신호 구성, 모델 아키텍처, 학습 패러다임 등 여러 차원에서 제안된 접근법을 체계적으로 정리하여, 각 방법론의 특징과 성능 차이를 이해하고 연구의 발전 방향을 파악할 필요가 있다. 특히 MIMIC 데이터셋 위주의 성능 평가가 외부 데이터셋에서는 성능 저하를 야기하는 데이터셋 편향 문제, 평가 지표 불일치, 일반화 한계 등 실험 설계의 한계점도 명확히 파악해야 한다.

### 3. 보고서의 분석 관점

본 보고서는 3 가지 주요 축으로 분석을 수행한다. 첫째, **연구체계 분류**를 통해 PPG 단독/혼합 입력, 딥러닝/머신러닝/LLM 회귀, 중앙/분산 학습 등 연구 분류 체계를 수립한다. 둘째,**방법론 분석**을 통해 U-Net 기반 파형 재구성, 시계열 변환 GAN, 전통적 머신러닝 회귀, LLM 기반 특징 통합 등 방법론 계열을 구분하고, 공통 문제 설정, 핵심 설계 패턴, 구조/모델 차이, 적용 대상 차이, 복잡도 및 확장성을 비교한다. 셋째,**실험결과 분석**을 통해 평가 지표 달성 현황, 데이터셋별 성능 차이, 성능 경향, ablation study, 민감도 분석, 일반화 검증 등의 결과를 정리하고, 실험 설계의 한계점과 결과 해석 경향도 검토한다.

### 4. 보고서 구성

- **1 장**: 연구체계 분류 - 입력 신호 구성, 모델 아키텍처, 학습 패러다임, 배포 관점에서 PPG 기반 혈압 추정 연구들을 체계적으로 분류하고, PPG 단독/혼합 입력, 회귀/딥러닝/LLM, 중앙/분산 학습 등 4 차원의 분류 체계를 제시한다.

- **2 장**: 방법론 분석 - U-Net 기반 파형 재구성, GAN 기반 변환, 전통적 머신러닝 회귀, LLM 기반 특징 통합 등 주요 방법론 계열을 구분하며, 공통 처리 구조, 핵심 설계 패턴, 구조/모델 차이, 적용 대상 차이, 복잡도 및 확장성 차원을 비교 분석한다.

- **3 장**: 실험결과 분석 - 평가 지표 달성 현황, 데이터셋별 성능 차이, 성능 경향, ablation study, 민감도 분석, 일반화 검증 등의 결과를 정립하고, 실험 설계의 한계점과 결과 해석 경향을 종합 정리한다.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

분석 대상 논문들은 PPG(Photoplethysmogram) 신호를 기반으로 동맥혈압 (ABP) 을 추정하는 연구로, 다음과 같은 기준과 원칙에 따라 분류되었다.

1. **주요 입력 신호**: PPG 단독 입력, PPG+ECG 혼합 입력, 신호 파형 자체 활용 여부
2. **모델 아키텍처**: 딥러닝 기반 (U-Net, CNN, LSTM, GAN 등), 머신러닝 회귀 (ARMA, SVR 등), LLM 기반
3. **접근 방식**: 파형 변환 (시계열 변환), 특징 공학 + 회귀, 종단 간 (end-to-end) 학습
4. **학습 패러다임**: 중앙 집중식 학습, 분산 학습 (Federated Learning), 개인별 모델 학습

분류는 각 논문이 PPG 신호 처리와 혈압 추정 문제 해결에 접근하는 주된 관점을 중심으로 수행되었으며, 일부 논문은 여러 범주에 적용 가능하나 가장 대표적인 1 개 범주에만 배정되었다.

### 2. 연구 분류 체계

#### 2.1 입력 신호 구성에 따른 분류

PPG 신호와 다른 생체 신호 (ECG 등) 의 유무에 따라 분류한다.

##### 2.1.1 PPG 단독 입력 기반 혈압 추정

PPG 신호만을 입력으로 사용하여 ABP 또는 혈압 값을 추정하는 연구들이다.

| 분류                                             | 논문명                                                                                                                                                        | 분류 근거                                                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| PPG 단독 입력 > PPG 단일 신호 기반 회귀          | Continuous PPG-Based Blood Pressure Monitoring Using Multi-Linear Regression (2020)                                                                           | 보조 센서 없이 단일 PPG 신호만으로 혈압을 추정하는 PWA 기반 MLR 회귀 모델                                            |
| PPG 단독 입력 > PPG 단독 입력을 통한 파형 재구성 | PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks (2020)             | PPG 신호 단독 입력을 통한 ABP 파형 재구성을 위해 근사 - 정제 2 단계 네트워크 구조를 활용                             |
| PPG 단독 입력 > U-Net 기반 PPG-to-ABP 파형 변환  | BP-Net: Efficient Deep Learning for Continuous Arterial Blood Pressure Estimation using Photoplethysmogram (2021)                                             | PPG 신호를 직접 입력받아 ABP 파형을 재구성하는 종단 간 U-Net 모델                                                    |
| PPG 단독 입력 > SCSA 기반 스펙트럼 및 특징 결합  | Schrödinger Spectrum based Continuous Cuff-less Blood Pressure Estimation using Clinically Relevant Features from PPG Signal and its Second Derivative (2023) | PPG 신호의 Schrödinger 연산자 기반 스펙트럼 분해 (SCSA) 와 SDPPG 형태학적 특징을 결합한 잡음 내성 커프리스 혈압 추정 |
| PPG 단독 입력 > 시계열 변환 및 프라이버시 보호   | Estimation of Continuous Blood Pressure from PPG via a Federated Learning Approach (2021)                                                                     | 페더레이티드 러닝을 통한 PPG 에서 ABP 로의 시계열 변환 프레임워크                                                    |
| PPG 단독 입력 > ARMA 기반 시변 동적 시스템       | Arterial Blood Pressure Feature Estimation Using Photoplethysmography (2018)                                                                                  | PPG 신호의 피크/트로프를 입력으로 하는 시변 ARMA 모델을 통한 혈압 연속 추정                                          |

##### 2.1.2 PPG + ECG 혼합 입력 기반 혈압 추정

PPG 신호와 ECG 신호를 함께 입력으로 사용하여 혈압을 추정하는 연구들이다.

| 분류                                                | 논문명                                                                                                                                            | 분류 근거                                                                                                    |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| PPG + ECG 입력 > CNN-LSTM 하이브리드                | A Deep Learning Approach to Predict Blood Pressure from PPG Signals (2021)                                                                        | PPG 신호의 시계열적 특성을 CNN 의 자동 특징 추출 능력과 LSTM 의 장기 의존성 포착을 결합                      |
| PPG + ECG 입력 > 오토인코더 특징 추출 파이프라인    | A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals (2021) | 얕은 U-Net 오토인코더를 특징 추출기, 전통적 ML 알고리즘을 회귀기로 조합한 하이브리드 파이프라인              |
| PPG + ECG 입력 > 파형 기반 ANN-LSTM 계층적 네트워크 | Cuffless Blood Pressure Estimation from Electrocardiogram and Photoplethysmogram Using Waveform Based ANN-LSTM Network (2018)                     | 심전도 및 광혈류측정 신호의 파형 정보를 직접 활용하여 자동 특징 추출과 시간 영역 학습이 결합된 계층적 신경망 |

#### 2.2 모델 아키텍처 및 접근 방식에 따른 분류

기존 특징 공학 기반 모델과 딥러닝 기반 모델, LLM 기반 모델로 세분화된 분류를 제시한다.

##### 2.2.1 딥러닝 기반 회귀 모델

CNN, U-Net, LSTM, GAN 등의 딥러닝 아키텍처를 사용하여 회귀 문제를 푸는 연구들이다.

| 분류                                          | 논문명                                                                                                                                            | 분류 근거                                                                                                                                                                          |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 딥러닝 회귀 > 근사 - 정제 2 단계 U-Net        | PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks (2020) | PPG 신호 단독 입력을 통한 ABP 파형 재구성을 위해 근사 - 정제 2 단계 네트워크 구조를 활용하며, 기존 ECG 의존적 방법과 수동적 특징 추출 방식을 개선한 딥러닝 기반 혈압 모니터링 연구 |
| 딥러닝 회귀 > CNN-LSTM 하이브리드 개인화 예측 | A Deep Learning Approach to Predict Blood Pressure from PPG Signals (2021)                                                                        | PPG 신호의 시계열적 특성을 CNN 의 자동 특징 추출 능력과 LSTM 의 장기 의존성 포착을 결합하여 개인화된 혈압 예측 정확도 극대화하는 딥러닝 기반 생체신호 분석 연구                    |
| 딥러닝 회귀 > U-Net 종단 간 학습              | BP-Net: Efficient Deep Learning for Continuous Arterial Blood Pressure Estimation using Photoplethysmogram (2021)                                 | PPG 신호를 직접 입력받아 ABP 파형을 재구성하는 종단 간 U-Net 모델로서, 명시적 특징 공학을 제거한 엣지 디바이스 배포 가능한 커프리스 혈압 추정 시스템에 속함                        |
| 딥러닝 변환 > 시계열 변환 GAN                 | Estimation of Continuous Blood Pressure from PPG via a Federated Learning Approach (2021)                                                         | CycleGAN 을 시계열 데이터에 변형한 T2T-GAN 아키텍처 생성자 (LSTM 기반), 판별자 (1D CNN) 사용                                                                                       |

##### 2.2.2 전통적 머신러닝 회귀 모델

딥러닝 대신 ARMA, SVR, CatBoost, 다중 선형 회귀 등의 전통적 머신러닝 기법을 사용하는 연구들이다.

| 분류                               | 논문명                                                                                                                                                        | 분류 근거                                                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 전통적 회귀 > 다중 선형 회귀 (MLR) | Continuous PPG-Based Blood Pressure Monitoring Using Multi-Linear Regression (2020)                                                                           | 보조 센서 없이 단일 PPG 신호만으로 혈압을 추정하는 PWA 기반 MLR 회귀 모델                                                             |
| 전통적 회귀 > CatBoost 회귀        | Schrödinger Spectrum based Continuous Cuff-less Blood Pressure Estimation using Clinically Relevant Features from PPG Signal and its Second Derivative (2023) | PPG 신호의 Schrödinger 연산자 기반 스펙트럼 분해 (SCSA) 와 SDPPG 형태학적 특징을 결합한 잡음 내성 커프리스 혈압 추정 프레임워크       |
| 전통적 회귀 > 시변 ARMA 모델       | Arterial Blood Pressure Feature Estimation Using Photoplethysmography (2018)                                                                                  | PPG 신호의 피크/트로프를 입력으로 하는 시변 ARMA 모델을 통해 ECG 없이도 정상 호흡 및 숨 참기 조건에서 혈압을 연속적으로 추정하는 연구 |

##### 2.2.3 LLM 기반 혈압 추정

대규모 언어 모델을 활용한 최근 접근법이다.

| 분류                                          | 논문명                                                                                        | 분류 근거                                                                                                                                                                                     |
| --------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM 기반 > 프롬프트 엔지니어링 및 지시어 튜닝 | Large Language Models for Cuffless Blood Pressure Measurement From Wearable Biosignals (2024) | LLM 을 기존 머신러닝 모델 (AdaBoost, Decision Tree) 과 비교 대상으로 사용하여 커프리스 BP 추정 성능을 평가하는 시스템, 프롬프트 문맥 강화 (BP 도메인 지식 + 사용자 정보) 전략으로 정확도 개선 |

#### 2.3 학습 패러다임 및 시스템 관점 분류

학습 방식 (중앙 집중식, 분산 학습) 과 배포 관점 (엣지 디바이스, 클라우드 기반) 에 따라 분류한다.

| 분류                                          | 논문명                                                                                                            | 분류 근거                                                                                                                                                   |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 분산 학습 > 페더레이티드 러닝 프라이버시 보호 | Estimation of Continuous Blood Pressure from PPG via a Federated Learning Approach (2021)                         | 환자 프라이버시 보호와 연속 혈압 모니터링의 실세계 적용 가능성에 초점을 둔 연구                                                                             |
| 엣지 배포 > 종단 간 경량화 시스템             | BP-Net: Efficient Deep Learning for Continuous Arterial Blood Pressure Estimation using Photoplethysmogram (2021) | PPG 신호를 직접 입력받아 ABP 파형을 재구성하는 종단 간 U-Net 모델로서, 명시적 특징 공학을 제거한 엣지 디바이스 배포 가능한 커프리스 혈압 추정 시스템에 속함 |

### 3. 종합 정리

분석 대상 논문들은 PPG 신호를 활용한 혈압 추정이라는 공통 목표를 향해 다양한 접근 방식으로 발전해 왔으며, PPG 단독 입력에서 PPG+ECG 혼합 입력으로, 전통적 머신러닝에서 딥러닝과 LLM 으로 진화하는 흐름이 관찰된다. 연구 체계는 입력 신호 구성 (단독/혼합), 모델 아키텍처 (회귀/딥러닝/LLM), 학습 패러다임 (중앙/분산), 배포 관점 (엣지/클라우드) 등 4 개 차원에서 체계적으로 구분된다. 전반적으로 PPG 신호 자체의 정보량을 최대한 활용하는 종단 간 모델 개발과 프라이버시 보호, 엣지 디바이스 배포, 개인화 학습 등 실세계 적용을 위한 기술적 고민이 지속적으로 반영되고 있는 연구 지형이 드러난다.

## 2 장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

### 1.1 공통 문제 정의

| 항목          | 내용                                                                    |
| ------------- | ----------------------------------------------------------------------- |
| **핵심 문제** | PPG(Photoplethysmogram) 신호 기반 연속 혈압 추정                        |
| **입력**      | PPG 파형 (단일 또는 다중 채널), 일부 논문: ECG                          |
| **출력**      | 수축기/이완기 혈압 (SBP/DBP), 평균동맥압 (MAP), ABP 파형                |
| **목표**      | 커프 기반 측정의 불편함과 침습적 방법을 극복한 비침습적·연속적 모니터링 |

### 1.2 공통 처리 구조

```text
[전처리] → [특징 추출/변환] → [모델/회귀] → [후처리] → [혈압 값]
   ↓              ↓                ↓            ↓
 필터링         자동/수동 특징     딥러닝/      정규화/
 정규화         (PPG/SDPPG 등)   머신러닝     보정/캘리브레이션
```

## 2. 방법론 계열 분류

### 2.1 U-Net 기반 파형 재구성 계열

**계열 정의**:

- 인코더-디코더 구조를 활용한 시계열 신호 변환
- ABP 파형을 '생성'하는 파이프라인 중심 접근
- 깊이 지도 학습 (Depth Supervision) 으로 정밀도 향상

**공통 특징**:

| 특징   | 설명                                    |
| ------ | --------------------------------------- |
| 구조   | 인코더 → 디코더 (U-Net 아키텍처)        |
| 학습   | MSE/MAE 기반 회귀 손실, Adam 옵티마이저 |
| 출력   | 재구성된 ABP 파형 → 파형 기반 통계 추출 |
| 데이터 | 윈도우 단위로 시퀀스 처리 (8~10 초)     |

**해당 논문**:

| 방법론 계열            | 논문명                                                                  | 핵심 특징                                                                                 |
| ---------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| U-Net 기반 파형 재구성 | PPG2ABP (2020)                                                          | 2 단계 계단식 네트워크 (Approximation Network + Refinement Network), 깊이 지도 학습 U-Net |
| U-Net 기반 파형 재구성 | A Shallow U-Net Architecture (2021)                                     | 얕은 1-레벨 U-Net 인코더 오토인코더 + Dense MLP 특징 추출 + 다양한 회귀 분석기            |
| U-Net 기반 파형 재구성 | BP-Net (2021)                                                           | Inception-Residual 블록, 자체 지도 사전 학습, 종단 간 학습                                |
| U-Net 기반 파형 재구성 | Cuffless Blood Pressure Estimation Using Waveform Based ANN-LSTM (2018) | ANN 하위 계층 (특징 추출) + 양방향 LSTM (시간 의존성)                                     |

### 2.2 시계열 변환 GAN 계열 (T2T-GAN)

**계열 정의**:

- 생성-판별자 구조를 활용한 시계열 간 변환
- CycleGAN 개념을 시계열 데이터 (PPG ↔ ABP) 에 적용
- 양방향 변환 (ABP→PPG, PPG→ABP) 가능

**공통 특징**:

| 특징   | 설명                                           |
| ------ | ---------------------------------------------- |
| 구조   | 2 방향 생성자/판별자 (G_PA, G_AP, D_P, D_A)    |
| 생성자 | LSTM 스택 (2 층, 50 은닉 유닛)                 |
| 판별자 | 1D CNN (4 층, Max Pooling, Sigmoid)            |
| 손실   | MSE + L1-cyc (Cycle consistency) + L1-identity |
| 학습   | 분산 페더레이티드 러닝 (클라이언트-서버 구조)  |

**해당 논문**:

| 방법론 계열     | 논문명                                                                         | 핵심 특징                                                     |
| --------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| 시계열 변환 GAN | Estimation of Continuous Blood Pressure from PPG via Federated Learning (2021) | T2T-GAN 프레임워크, 양방향 변환 + 분산 학습 + 프라이버시 보호 |

### 2.3 전통적 머신러닝 회귀 계열

**계열 정의**:

- 신호 특징 공학 (Feature Engineering) 을 통한 선형/비선형 회귀 모델
- 데이터 중심 접근 (Data-driven) 보다 규칙/모델 중심
- 작은 데이터셋에서도 일반화 가능

**공통 특징**:

| 특징 | 설명                                                  |
| ---- | ----------------------------------------------------- |
| 입력 | 수동 추출된 특징 벡터 (PPG 파형, 미분, 파형 속성)     |
| 모델 | 다중 선형 회귀 (MLR), SVR, 랜덤 포레스트, CatBoost 등 |
| 특징 | PPG 파형 특징점, 이차미분 특징, 파형 기하학적 속성    |
| 학습 | 중첩 K-겹 교차 검증, 베이지안 최적화                  |

**해당 논문**:

| 방법론 계열          | 논문명                                                                              | 핵심 특징                                                     |
| -------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 전통적 머신러닝 회귀 | Continuous PPG-Based Blood Pressure Monitoring Using Multi-Linear Regression (2020) | 27 개 특징 (PPG + 이차미분) + MLR 모델 + 레코드별 오프셋 보정 |
| 전통적 머신러닝 회귀 | Schrödinger Spectrum based Continuous Cuff-less Blood Pressure Estimation (2023)    | SCSA 신호 재구성 + 38 개 특징 + SVR/CatBoost/XGBoost          |
| 전통적 머신러닝 회귀 | Arterial Blood Pressure Feature Estimation Using Photoplethysmography (2018)        | 5 차 시변 ARMA 모델 + 최소 제곱 + 피크/트로프 매핑            |

### 2.4 LLM 기반 특징 통합 계열

**계열 정의**:

- 대형 언어 모델을 도메인 지식 통합 도구로 활용
- 텍스트 프롬프트에 특징 벡터 임베딩하여 추론
- 보정 기반 (Calibration-based) 출력

**공통 특징**:

| 특징   | 설명                                              |
| ------ | ------------------------------------------------- |
| 입력   | 31 가지 생리학적 특징 (평균) → 텍스트 프롬프트    |
| 모델   | LLaMA3-8B, Qwen2-7B, Gemma-7B 등 오픈 소스 LLM    |
| 후처리 | $\alpha$ 가중치로 LLM 출력 + 기준 BP 값 선형 조합 |
| 장점   | 도메인 지식 직접 통합, 데이터 양 부족 보완        |

**해당 논문**:

| 방법론 계열        | 논문명                                                                                        | 핵심 특징                                                         |
| ------------------ | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| LLM 기반 특징 통합 | Large Language Models for Cuffless Blood Pressure Measurement From Wearable Biosignals (2024) | 생체 신호 특징을 텍스트로 임베딩, LLM 지시어 튜닝, 보정 공식 적용 |

## 3. 핵심 설계 패턴 분석

### 3.1 인코더-디코더 구조 (End-to-End Waveform Reconstruction)

| 논문                   | 구조                                       | 설명                                                                                          |
| ---------------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| PPG2ABP (2020)         | Approximation Network → Refinement Network | Approximation Network(1D U-Net) 가 ABP 파형을 근사 → Refinement Network(MultiResUNet) 가 정제 |
| BP-Net (2021)          | 인코더-디코더 U-Net                        | Inception-Residual 블록, 잔차 연결로 기울기 흐름 개선                                         |
| A Shallow U-Net (2021) | 얕은 1-레벨 인코더 오토인코더              | 오토인코더로 ABP 파형 복원 → 인코더 말단 Dense 계층으로 특징 추출                             |

### 3.2 특징 추출 파이프라인

```text
원시 신호
   ↓
[전처리: 필터링, 정규화, 웨이블릿/스케일 재구성]
   ↓
[특징 추출: 수동(PPG 특징점, 이차미분) 또는 자동(CNN/LSTM)]
   ↓
[모델: MLR, SVR, GAN, LLM 등]
   ↓
[출력: SBP/DBP/MAP 또는 ABP 파형]
```

### 3.3 오프셋/캘리브레이션 전략

| 논문                                                  | 전략                                   | 설명                                             |
| ----------------------------------------------------- | -------------------------------------- | ------------------------------------------------ |
| Continuous PPG-Based Blood Pressure Monitoring (2020) | 제로-평균 (MIMIC I) / 첫 커프 값 (QUD) | 레코드별 오프셋 분리 학습, 평균 제곱 오차 최소화 |
| Estimation of Continuous Blood Pressure (2021)        | 1 분 캘리브레이션                      | 생성된 파형에서 편향 제거                        |
| Large Language Models (2024)                          | α 가중치 보정                          | SBP_cal = SBP_free·α + BaseSBP·(1-α)             |

### 3.4 시계열 변환 전략

| 논문                                           | 전략           | 설명                                                 |
| ---------------------------------------------- | -------------- | ---------------------------------------------------- |
| Estimation of Continuous Blood Pressure (2021) | 양방향 변환    | ABP→PPG 와 PPG→ABP 모두 가능, Cycle consistency 손실 |
| Cuffless Blood Pressure Estimation (2018)      | 파형 분할 기반 | 256 샘플 리샘플링, 비균일 파형 분할, ANN-LSTM 계층   |

## 4. 방법론 비교 분석

### 4.1 문제 접근 방식 차이

| 접근 방식                | 장점                                                   | 한계                                  | 해당 논문                         |
| ------------------------ | ------------------------------------------------------ | ------------------------------------- | --------------------------------- |
| **파형 재구성 (U-Net)**  | ABP 파형 전체 생성, 위상 지연 극복, 심박동별 추정 가능 | 모델 복잡도, GPU 요구, 데이터 양 필요 | PPG2ABP, BP-Net, A Shallow U-Net  |
| **GAN 기반 변환**        | 양방향 변환 가능, 생성-판별자 균형, 분산 학습          | 학습 안정성, 하이퍼파라미터 민감      | Estimation via Federated Learning |
| **수동 특징 + MLR**      | 데이터 효율성, 엣지 배포 가능, 해석 용이               | 특징 공학 부담, 자동화 어려움         | MLR, ARMA, SCSA                   |
| **자동 특징 (CNN-LSTM)** | 특징 공학 불필요, 최적화 가능성                        | 데이터 양 필요, 모델 블랙박스         | CNN-LSTM-MLP, ANN-LSTM            |
| **LLM 기반**             | 도메인 지식 통합, 설명 가능성, 데이터 효율             | 컴퓨팅 자원, 추론 지연                | Large Language Models             |

### 4.2 구조/모델 차이

| 차원            | U-Net 계열               | GAN 계열                      | MLR 계열              | LLM 계열           |
| --------------- | ------------------------ | ----------------------------- | --------------------- | ------------------ |
| **입력 형태**   | 윈도우 PPG (8~10 초)     | 윈도우 PPG                    | 특징 벡터             | 특징 벡터 (텍스트) |
| **모델 유형**   | CNN/U-Net/LSTM           | GAN (LSTM 생성자, CNN 판별자) | MLR, SVR, Tree Model  | LLM (Transformer)  |
| **학습 데이터** | 대규모 (MIMIC II/III)    | 중규모, 분산                  | 소규모 (28 레코드 등) | 특징 데이터        |
| **추론 시간**   | ms 단위 (엣지 배포 가능) | 중간                          | 빠름                  | 느림 (LLM)         |

### 4.3 적용 대상 차이

| 적용 대상                          | 적합한 방법론        | 이유                           |
| ---------------------------------- | -------------------- | ------------------------------ |
| **대규모 데이터셋** (MIMIC II/III) | U-Net 계열, CNN-LSTM | 모델 용량 요구, 과적합 방지    |
| **소규모/특수 데이터셋**           | MLR, ARMA, SCSA      | 데이터 효율성, 오프셋 보정     |
| **엣지 디바이스**                  | MLR, U-Net (경량)    | 낮은 지연 시간, 계산 효율성    |
| **도메인 지식 통합**               | LLM 계열             | 텍스트 프롬프트로 지식 인코딩  |
| **프라이버시 보호**                | 페더레이티드 GAN     | 분산 학습, 중앙 서버 무 데이터 |

### 4.4 복잡도 및 확장성

| 측면               | U-Net 계열                   | GAN 계열                    | MLR 계열         | LLM 계열              |
| ------------------ | ---------------------------- | --------------------------- | ---------------- | --------------------- |
| **모델 파라미터**  | ~0.55M (얕은 U-Net)          | ~2 층 LSTM (50 유닛)        | 최소 (회귀 계수) | 최대 (Transformer)    |
| **학습 시간**      | 중간 (100 epoch 등)          | 중간 (10 라운드)            | 빠름             | 중간 (지시어 튜닝)    |
| **확장성**         | 데이터 증가에 따라 성능 향상 | 분산 학습 확장 가능         | 데이터 독립적    | 도메인 지식 확장 가능 |
| **하이퍼파라미터** | 중간 (Batch, LR, Epoch)      | 중간 (λ_c, λ_i, 클라이언트) | 최소             | 중간 (프롬프트, α)    |

## 5. 방법론 흐름 및 진화

### 5.1 초기 접근 (2018-2019)

| 특징               | 설명                         | 대표 논문                                         |
| ------------------ | ---------------------------- | ------------------------------------------------- |
| **모델 중심**      | ARMA 등 통계 모델, 회귀 분석 | Arterial Blood Pressure Feature Estimation (2018) |
| **특징 공학 의존** | PPG 파형 특징점 수동 추출    | Cuffless Blood Pressure Using ANN-LSTM (2018)     |
| **단일 파형**      | 심박동 단위 데이터           |                                                   |

### 5.2 발전 단계 (2020-2021)

| 특징            | 설명                                    | 대표 논문                                      |
| --------------- | --------------------------------------- | ---------------------------------------------- |
| **파형 재구성** | U-Net 아키텍처 도입, ABP 파형 전체 생성 | PPG2ABP, BP-Net, A Shallow U-Net               |
| **자동 특징**   | CNN-LSTM 파이프라인, 특징 공학 불필요   | A Deep Learning Approach (2021)                |
| **GAN 적용**    | CycleGAN 개념 시계열 변환에 적용        | Estimation of Continuous Blood Pressure (2021) |

### 5.3 최신 경향 (2023-2024)

| 특징             | 설명                               | 대표 논문                                      |
| ---------------- | ---------------------------------- | ---------------------------------------------- |
| **하이브리드**   | 신호 재구성 (SCSA) + 머신러닝 특징 | Schrödinger Spectrum (2023)                    |
| **LLM 통합**     | 도메인 지식 텍스트 프롬프트로 통합 | Large Language Models (2024)                   |
| **페더레이티드** | 분산 학습, 프라이버시 보호         | Estimation of Continuous Blood Pressure (2021) |

## 6. 종합 정리

본 분야의 방법론은 크게 **파형 재구성 계열 (U-Net)**과**시계열 변환 계열 (GAN)**과**전통적 머신러닝 계열**의 세 가지 축으로 구분된다. U-Net 계열은 인코더-디코더 구조를 통해 ABP 파형을 '생성'하며, GAN 계열은 양방향 변환과 분산 학습을 특징으로 한다. 전통적 머신러닝 계열은 수동 특징 공학과 회귀 모델에 의존하며, 최근에는 LLM을 도메인 지식 통합 도구로 활용하는 새로운 접근이 등장했다. 초기에는 ARMA 등 통계 모델과 특징 공학에 의존했으나, U-Net 도입으로 파형 재구성 정확도가 향상되었으며, 최근에는 LLM과 페더레이티드 러닝이 데이터 효율성과 프라이버시 보호를 동시에 가능하게 하고 있다.

## 3장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

**1.1 데이터셋 유형**

| 데이터셋            | 유형        | 샘플링 주파수          | 주요 활용                |
| ------------------- | ----------- | ---------------------- | ------------------------ |
| MIMIC-I             | 임상        | 125 Hz                 | ECG/PPG 기반 모델 학습   |
| MIMIC-II            | 임상        | 125 Hz                 | 대규모 인스턴스 (12,000) |
| MIMIC-III           | 임상        | 1024 샘플              | 웨이블릿 전처리          |
| QUD                 | 외부 데이터 | 100 Hz                 | 보정 전 성능 평가        |
| BCG                 | 외부 데이터 | 1000→125 Hz 다운샘플링 | 일반화 성능 검증         |
| PhysioNet Cuff-Less | 훈련/검증   | 125 Hz                 | 페더레이티드 학습        |
| UCI 웨어러블        | 대규모      | -                      | 1,272 명 데이터셋        |

**1.2 평가 환경**

| 환경 유형       | 활용 사례                                 |
| --------------- | ----------------------------------------- |
| 임상 환경       | MIMIC 데이터셋 (ICU 환자)                 |
| 시뮬레이션 환경 | in-silico PWD, 잡음 주입 (SNR 10 dB 이상) |
| 실환경/웨어러블 | UCI 웨어러블, IEEE 웨어러블 데이터셋      |

**1.3 비교 방식**

| 비교 대상          | 비교 유형                            |
| ------------------ | ------------------------------------ |
| 커프 기반 측정     | 황금 표준으로 사용                   |
| 기존 머신러닝 모델 | AdaBoost, SVR, XGBoost 등            |
| 딥러닝 모델        | CNN, LSTM, U-Net 기반 방법           |
| 선구적 연구들      | 전통적 매개변수 기반 (PWV, PTT, PAT) |

**1.4 주요 평가 지표**

| 지표 | 단위        | 기준                                          |
| ---- | ----------- | --------------------------------------------- |
| MAE  | mmHg        |                                               |
| BHS  | Grade A/B/C | Grade A: 평균 오차 $\le$ 5, 표준 편차 $\le$ 8 |
| AAMI | ME/SD       | ME $\lt$ 5mmHg, SD $\lt$ 8mmHg                |
| PCC  | -           | 상관 계수                                     |

### 2. 주요 실험 결과 정렬

| 논문 제목 (연도)                                                                                                                                              | 데이터셋/환경                                                                | 비교 대상                                                | 평가 지표                                       | 핵심 결과                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PPG2ABP: Translating Photoplethysmogram (PPG) Signals to Arterial Blood Pressure (ABP) Waveforms using Fully Convolutional Neural Networks (2020)             | MIMIC-III PPG 및 ABP 동시 신호 (8.192 초 세그먼트)                           | 기존 침습적/커프 기반 방법, 전통적 매개변수, 선구적 연구 | MAE, BHS, AAMI, PCC, Bland-Altman 95% 일치 한계 | ABP 파형 MAE 4.604±5.043 mmHg, DBP MAE 3.449±6.147 mmHg, MAP MAE 2.310±4.437 mmHg, SBP MAE 5.727±9.162 mmHg; BHS Grade A(DBP, MAP), Grade B(SBP); AAMI 만족 (DBP, MAP) |
| Continuous PPG-Based Blood Pressure Monitoring Using Multi-Linear Regression (2020)                                                                           | MIMIC-I(28 고품질 레코드), QUD                                               | 커프 기반, PWV 기반, PWA 기반                            | MAE, SDE, ME, 절대 오차 비율, PCC               | MIMIC-I: SBP MAE 6.10 mmHg, DBP MAE 4.65 mmHg; QUD: SBP MAE 6.70 mmHg, DBP MAE 6.64 mmHg; QUD에서 ISO/ANSI/AAMI 기준 충족 (5±8 mmHg)                                   |
| Estimation of Continuous Blood Pressure from PPG via a Federated Learning Approach (2021)                                                                     | 훈련/검증=Cuff-Less(125Hz), 테스트=QUD 마취 환자 데이터 (100Hz)              | 페더레이티드 vs 비페더레이티드 방식                      | MAP 오차, AAMI 기준, DTW, RMSE, PCC             | 캘리브레이션 전 MAP 오차 -4.23mmHg(σ=23.7), 캘리브레이션 후 MAP 오차 2.54mmHg(σ=23.7); 평균 오차 AAMI 기준 충족, 표준 편차 미달                                        |
| A Deep Learning Approach to Predict Blood Pressure from PPG Signals (2021)                                                                                    | MIMIC-II(20 명), UQVSD(49 개 측정)                                           | 기존 PPG 기반 BP 추정 연구                               | MAE, SDAE, BHS, AAMI                            | MIMIC-II: SBP MAE 3.70/SDAE 3.07, DBP MAE 2.02/SDAE 1.76; UQVSD: SBP MAE 3.91/SDAE 4.78, DBP MAE 1.99/SDAE 2.45; BHS Grade A 달성, AAMI 표준 충족                      |
| A Shallow U-Net Architecture for Reliably Predicting Blood Pressure (BP) from Photoplethysmogram (PPG) and Electrocardiogram (ECG) Signals (2021)             | MIMIC-II(942 명, 12,000 인스턴스), 외부 BCG 데이터셋(40 명)                  | 기존 머신러닝/딥러닝 (SVR, LSTM, CNN, 깊은 U-Net)        | MAE, BHS, AAMI, PCC                             | 1-레벨 인코더+4 채널: SBP MAE=2.333, DBP MAE=0.713(BHS Grade A); 외부 BCG: SBP MAE=2.728, DBP MAE=1.166; ECG 없이 PPG+VPG+APG로도 유사 성능                            |
| BP-Net: Efficient Deep Learning for Continuous Arterial Blood Pressure Estimation using Photoplethysmogram (2021)                                             | MIMIC II Waveform(127,260 개 10 초 에피소드)                                 | Athaya et al., Ibtehaz et al., Harfiya et al.            | BHS, AAMI, MAE, 추론 시간                       | SBP MAE 5.16mmHg, DBP MAE 2.89mmHg; BHS Grade A(DBP/MAP), Grade B(SBP); AAMI 기준 DBP/MAP 통과, Raspberry Pi 4에서 4.25ms/초 추론                                      |
| Schrödinger Spectrum based Continuous Cuff-less Blood Pressure Estimation using Clinically Relevant Features from PPG Signal and its Second Derivative (2023) | UCI MIMIC-II, MIMIC-III, Queensland, in-silico PWD, IEEE 웨어러블, 잡음 주입 | SVR, XGBoost, CatBoost, LightGBM                         | MAE, PCC(r), AAMI, BHS                          | CatBoost: SBP MAE 5.37±5.56 mmHg, DBP MAE 2.96±3.13 mmHg(SBP r=0.89, DBP r=0.85, MAP r=0.86); AAMI 기준 충족, BHS Grade A                                              |
| Large Language Models for Cuffless Blood Pressure Measurement From Wearable Biosignals (2024)                                                                 | 1,272 명 대규모 웨어러블 생체 신호 데이터셋                                  | AdaBoost, Decision Tree, Zero-baseline                   | ME±SDE, MAE                                     | LLaMA3-8B: SBP ME±SDE 0.00±9.25 mmHg, MAE 7.08; Qwen2-7B: DBP ME±SDE 1.29±6.37 mmHg, MAE 5.19; AdaBoost 대비 SBP ME 감소 1.39→0.00 mmHg, DBP ME 감소 1.62→1.29 mmHg    |
| Cuffless Blood Pressure Estimation from Electrocardiogram and Photoplethysmogram Using Waveform Based ANN-LSTM Network (2018)                                 | MIMIC I(39 명 ICU 환자)                                                      | P-AdaBoost, W-AdaBoost, DeepRNN-4L                       | MAE, RMSE, AAMI, BHS, PCC, Bland-Altman         | M=32 시퀀스: SBP MAE 0.93/RMSE 1.26, DBP MAE 0.52/RMSE 0.73; SBP r=0.9986, DBP r=0.9975; AAMI 및 BHS Grade A 표준 만족                                                 |
| Arterial Blood Pressure Feature Estimation Using Photoplethysmography (2018)                                                                                  | 15 명 건강한 피험자, Finapres NOVA, Nellcor OxiMax N-600x                    | 커프 기반, Finapres, 압평 측정법, PTT 기반, PPG 회귀     | rMSE, 모델 일관성 (ANOVA)                       | 정상 호흡: SBP 3.95~4.69 mmHg, DBP 2.97~3.50 mmHg, MAP 3.03~3.35 mmHg(rMSE<5 mmHg); 숨 참기에서도 SBP 4.19~4.94, DBP 2.88~3.29 유지                                    |

### 3. 성능 패턴 및 경향 분석

**3.1 평가 지표 달성 현황 비교**

| 평가 기준                  | 달성 논문                                                                                                                                                             | 달성 수 | 미달 논문                                                 | 미달 수 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------------------------- | ------- |
| AAMI 평균 오차 (ME<5 mmHg) | PPG2ABP(2020), Continuous PPG MLR(2020), Federated(2021), Deep Learning CNN-LSTM(2021), Shallow U-Net(2021), Schrödinger Spectrum(2023), BP-Net(2021), ANN-LSTM(2018) | 8       | Schrödinger Spectrum(2023)-정상 SBP 범주 약간 벗어남      | 1       |
| AAMI 표준 편차 (SD<8 mmHg) | PPG2ABP(2020), Continuous PPG MLR(2020)                                                                                                                               | 2       | Federated(2021), BP-Net(2021), Schrödinger Spectrum(2023) | 3       |
| BHS Grade A                | PPG2ABP(2020), Deep Learning CNN-LSTM(2021), Shallow U-Net(2021), Schrödinger Spectrum(2023)                                                                          | 4       | BP-Net(2021)-SBP Grade B, Continuous PPG MLR(2020)        | 2       |

**3.2 데이터셋별 성능 차이 분석**

| 데이터셋               | SBP MAE 범위     | DBP MAE 범위     | MAP MAE 범위     | 평균 SBP MAE |
| ---------------------- | ---------------- | ---------------- | ---------------- | ------------ |
| MIMIC-II/I 기반        | 0.93 ~ 5.73 mmHg | 0.52 ~ 6.15 mmHg | 2.31 ~ 3.35 mmHg | 2.57 mmHg    |
| QUD                    | 6.70 mmHg        | 6.64 mmHg        | -                | 6.67 mmHg    |
| 대규모 웨어러블 (2024) | 0.00 mmHg (ME)   | 1.29 mmHg (ME)   | -                | 0.00 mmHg    |

**3.3 성능 경향 분석**

- **MIMIC 데이터셋 기반 결과 일관성**: MIMIC-I, MIMIC-II, MIMIC-III 데이터를 사용한 연구들에서 SBP MAE는 0.93~5.73 mmHg 범위, DBP MAE는 0.52~6.15 mmHg 범위에서 비교적 일관된 성능을 보인다.

- **외부 데이터셋 성능 변동**: QUD 데이터셋을 사용한 연구에서는 SBP MAE 6.10~6.70 mmHg로 MIMIC 데이터셋 대비 약 4~5 배 높은 오차율을 보인다.

- **모델 아키텍처별 성능**:
  - 얕은 아키텍처 (Shallow U-Net, 1-레벨 인코더): SBP MAE 2.333 mmHg 달성
  - 심층 CNN-U-Net (BP-Net): SBP MAE 5.16 mmHg 달성
  - 전통적 회귀 (CatBoost/SVR): SBP MAE 5.37 mmHg 달성
  - CNN-LSTM 하이브리드: SBP MAE 3.70~3.91 mmHg 달성

- **보정 효과**: 캘리브레이션/오프셋 보정 적용 시 MAP 오차가 -4.23mmHg에서 2.54mmHg로 개선 (Federated(2021)).

### 4. 추가 실험 및 검증 패턴

**4.1 Ablation Study 패턴**

| 연구                        | ablation 유형                                    | 검증 방식                                 | 주요 결과                                                      |
| --------------------------- | ------------------------------------------------ | ----------------------------------------- | -------------------------------------------------------------- |
| Shallow U-Net (2021)        | 인코더 깊이 (1-4), 폭 (32-256), 커널 크기 (3-11) | 각 파라미터 조합별 성능 비교              | 1-레벨 인코더+4 채널 입력이 가장 강력                          |
| Shallow U-Net (2021)        | 입력 채널 (1-4), ECG 포함 여부                   | PPG 단독 vs PPG+VPG+APG vs PPG+ECG        | ECG 없이 PPG+VPG+APG로도 유사 성능                             |
| Schrödinger Spectrum (2023) | 특징 유형 (PPG/SDPPG vs SCSA 스펙트럼)           | 특징 중요도 분석                          | BW_66, PIR_p, T_cb 등 PPG/SDPPG 기반 특징이 SCSA 특징보다 중요 |
| LLM (2024)                  | 프롬프트 구성 (기본/BP 도메인/사용자 정보)       | 단계별 프롬프트 성능 비교                 | 도메인 지식 + 사용자 정보 통합이 정확도 향상 핵심              |
| LLM (2024)                  | 학습 데이터 비율                                 | 학습 데이터 30%만으로 80% 베이스라인 능가 | 데이터 효율성 우수                                             |

**4.2 민감도 분석 패턴**

| 연구                      | 분석 대상     | 분석 방법                    | 결과                                               |
| ------------------------- | ------------- | ---------------------------- | -------------------------------------------------- |
| Federated (2021)          | 데이터셋 편향 | 훈련/테스트 데이터셋 간 차이 | 데이터셋 간 환자 특성 차이가 표준 편차 높은 주원인 |
| Schrödinger (2023)        | 잡음 내성     | SNR 10 dB 이상 잡음 주입     | SDPPG 특징이 잡음 환경에서 더 견고                 |
| ANN-LSTM (2018)           | 시퀀스 길이   | M=10/32 비교                 | 긴 시퀀스(M=32)에서 성능 향상                      |
| Continuous PPG MLR (2020) | 임상 상태     | 고혈압 그룹 성능 분석        | 고혈압 그룹에서 성능 저하 (PPG 파형 왜곡)          |

**4.3 일반화 검증 패턴**

| 연구                      | 일반화 접근 방식 | 검증 데이터                        | 결과                                 |
| ------------------------- | ---------------- | ---------------------------------- | ------------------------------------ |
| Continuous PPG MLR (2020) | 레코드 기반 학습 | 28 고품질 레코드 중 041+427만 사용 | 다양한 피험자에서 우수한 일반화 성능 |
| ANN-LSTM (2018)           | 외부 데이터      | MIMIC I(39 명)                     | 다양한 임상 상태에서의 검증          |
| Shallow U-Net (2021)      | 외부 데이터셋    | MIMIC-II(942 명) + BCG(40 명)      | 외부 데이터셋에서도 성능 유지        |
| Federated (2021)          | 분산 학습        | 20 클라이언트                      | 분산 환경에서의 학습/추론            |

### 5. 실험 설계의 한계 및 비교상의 주의점

**5.1 데이터셋 의존성**

| 문제 유형            | 구체적 사례                             | 영향                                      |
| -------------------- | --------------------------------------- | ----------------------------------------- |
| MIMIC 데이터셋 편향  | 대부분의 연구가 MIMIC-I/II/III에만 의존 | 외부 데이터셋 (QUD) 에서 성능 급격히 저하 |
| 단일 환자 테스트     | Federated(2021) 테스트 데이터는 1 명만  | 일반화 한계 명확                          |
| 제한된 인구집단      | ANN-LSTM(2018), Arterial(2018)          | 젊은 건강한 피험자, 다양한 인구군 미포함  |
| 샘플링 주파수 불일치 | 100 Hz, 125 Hz, 1024 샘플 등            | 교차 비교 어려움                          |

**5.2 비교 조건의 불일치**

| 문제 유형        | 구체적 사례                               | 영향               |
| ---------------- | ----------------------------------------- | ------------------ |
| 평가 지표 차이   | 일부 연구는 BHS, 일부는 AAMI, 일부는 모두 | 직접 비교 어려움   |
| 데이터 분할 방식 | 10-fold, Leave-one-window-out 등          | 검증 프로토콜 차이 |
| 학습 에포크      | 300 에포크 vs 조기 종료 (patience=20)     | 과적합 가능성 차이 |
| 전처리 차이      | 웨이블릿, DWT 등                          | 공정성 비교 어려움 |

**5.3 평가 지표의 한계**

| 지표     | 한계                | 개선 방향                  |
| -------- | ------------------- | -------------------------- |
| MAE      | 극단 값에 민감      | 양자화 오차 함께 고려 필요 |
| AAMI BHS | 표준 편차 기준 미달 | 95% 일치 한계 추가         |
| PCC      | 방향성만 측정       | 절대 오차와 함께 필요      |

**5.4 일반화 한계**

- **MIMIC 데이터셋**: ICU 환자를 대상으로 했으며, 일반 인구집단으로 확장 시 성능 저하 예상
- **외부 데이터셋 의존성**: QUD 등 외부 데이터셋 사용 시 성능 변동 심화
- **임상 환경 vs 웨어러블**: 임상 환경에서 측정한 데이터를 웨어러블 기기로 재현할 경우 운동량자 등 환경 요인 고려 필요

### 6. 결과 해석의 경향

**6.1 저자들의 공통적 해석 경향**

| 해석 유형        | 구체적 내용                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| 성능 향상 강조   | "뛰어난 정확도", "유사 성능", "경쟁력 있는 정확도" 등 긍정적 어조 사용 |
| 실용성 강조      | "웨어러블 기기 배포 가능성", "상용화 잠재력", "실시간 구현 가능" 등    |
| 한계 인정 제한적 | "추가 전처리 필요", "개인 맞춤형 보정 탐색 필요" 등으로 제한적 언급    |
| 표준 준수 강조   | "AAMI/BHS 표준 충족" 결과 해석의 핵심 요소로 사용                      |

**6.2 해석과 관찰 결과 분리**

| 구분 | 관찰 결과              | 저자 해석                                |
| ---- | ---------------------- | ---------------------------------------- |
| 성능 | MIMIC-QUD 간 성능 차이 | "데이터셋 간 환자 특성 차이가 원인"      |
| 성능 | 고혈압 그룹 성능 저하  | "PPG 파형 왜곡 때문"                     |
| 성능 | AAMI 표준 편차 미달    | "표준 편차 높은 주원인 데이터셋 간 편향" |
| 성능 | SBP 추정부 분산 높음   | "높은 분산으로 인해 성능 한계"           |

**6.3 해석 경향 분석**

- **과장된 긍정적 표현**: "뛰어난", "유리하다", "완벽한" 등의 표현보다는 구체적 수치 ("SBP MAE 3.70 mmHg") 를 사용한 객관적 서술
- **한계 인정의 선택성**: 실제 성능 저하 (QUD 성능) 는 "데이터셋 편향"으로, 기술적 한계 (계산 효율성) 는 인정하나 "추가 연구余地"로 포장
- **기술적 혁신 강조**: "자동 특징 추출", "분산 학습", "LLM 기반" 등 기술적 특징을 결과 해석의 핵심으로 사용

### 7. 종합 정리

전체 실험 결과 분석은 PPG 기반 혈압 추정 분야에서 MIMIC 데이터셋을 중심으로 일관된 성능 패턴이 관찰되며, SBP MAE는 0.93~5.73 mmHg 범위, DBP MAE는 0.52~6.15 mmHg 범위로 비교적 일관된 성능을 보인다. 외부 데이터셋 (QUD) 에서는 성능이 MIMIC 대비 2~4 배 높은 오차를 보이며, 이는 데이터셋 편향과 일반화 한계 문제를 명확히 드러낸다. 모델 아키텍처 (얕은 U-Net, CNN-LSTM, CatBoost 등) 간 성능 차이는 제한적이며, 얕은 아키텍처가 대규모 데이터에서 더 강력한 성능을 보인다. 보정/캘리브레이션은 평균 오차 개선에는 효과적이지만 표준 편차 개선에는 한계가 있다. 평가 지표 측면에서는 AAMI 평균 오차 기준 대부분 충족하나 표준 편차 기준은 일부 연구에서 미달하며, BHS 등급은 모델과 데이터셋에 따라 Grade A/B/C에서 변동된다. 추가 실험에서 ablation study, 민감도 분석, 일반화 검증은 방법론의 강건성을 입증하는 핵심 도구로 사용되며, ablation study는 얕은 아키텍처의 효율성을, 민감도 분석은 데이터셋 편향의 영향을, 일반화 검증은 다양한 환경에서의 성능을 확인한다. 실험 설계의 한계로는 데이터셋 편향, 평가 지표 불일치, 외부 데이터셋 성능 변동, 일반화 제한이 주된 문제이며, 향후 연구에서는 더 다양한 인구집단, 표준화된 평가 프로토콜, 외부 데이터셋 성능 개선이 필수적이다.
