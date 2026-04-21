# Anomaly Detection

## 서론

### 1. 연구 배경

본 보고서는 이상 탐지 (Anomaly Detection) 분야를 체계적으로 분석한 문헌 연구 결과이다. 이상 탐지는 정상 데이터의 패턴을 학습하고, 이를 학습하지 못한 데이터를 이상으로 판별하는 문제 설정을 기반으로 하며, One-Class 또는 Unsupervised 설정에서 정상 분포 학습과 이상 판별을 동시에 수행하는 다양한 방법론이 존재한다.

분석 대상 논문들은 (1) 이상 탐지 목적함수의 정의 방식 (재구성/분류/생성/하이브리드), (2) 학습 데이터의 라벨링 정도 (비지도/준감독/완전감독/강화학습), (3) 대상 데이터 형태 (이미지/시계열/멀티모달/고차원 수치), (4) 모델 아키텍처 (AE/GAN/Transformer/SVM/FCN 등) 를 기준으로 15 개 대분류와 여러 하위 범주로 체계화되어 있다. 재구성 기반 방법 (VAE, Autoencoder, Perceptual Loss 등) 이 가장 많은 논문을 차지하며, 생성 모델 (GAN), 분류기 기반 (Self-supervised classification, Deviation Network), 하이브리드 (AE+SVDD, Transformer, Normality+Deviation) 접근법도 활발히 연구된 것으로 나타났다. 학습 데이터 설정에 따라 준감독/완전감독 방법 (DevNet, Reinforcement Learning, SVDDneg) 과 효율성/실시간성 (OCSVM 근사, HTM 예측) 을 개선하는 연구도 포함되어 있으며, 공정성 견고성 (Adversarial defense, Contamination robust), 설명 가능성 (FCN explanation), 메모리/상태 (Memory module, state tracking) 등의 보조적 고려사항도 반영되었다. 도메인 적용 관점에서 의료이미지, 산업비전 (MVTec-AD), 물처리시스템, 로봇 피딩 등 특수 환경 탐지 논문이 포괄되며, 다중모달 (5 종 센서 융합), 그룹/집단, 능동학습 기반 탐지 등도 연구 영역으로 포함되었다.

### 2. 문제의식 및 분석 필요성

이상 탐지 연구는 방법론적 다양성과 응용 환경의 특수성 때문에 체계적인 비교·분석이 필요하다. 다양한 방법론 계열 (재구성 기반, 판별기 기반, 생성 모델 기반, SVDD 기반, 하이브리드, 특수 설계) 이 각각 고유한 장단점과 트레이드오프를 가지고 있으며, 데이터셋 규모, 오염 비율, 평가 목표에 따라 적합한 설계가 달라진다. 또한 평가 지표 (ROC AUC, AUROC, AUC-PR, mAUROC 등) 와 데이터셋 (의료 영상, 이미지, 시계열, IoT 트래픽 등) 간의 불일치로 인해 방법론 간 직접 비교가 어려운 현실에서, 본 보고서는 이러한 비교 조건의 불일치를 인식하고 방법론별 특성과 적용 조건을 명확히 제시한다.

### 3. 보고서의 분석 관점

본 보고서는 세 가지 관점에서 문헌을 분석한다: (1) **연구체계 분류** - 이상 탐지 방법론의 분류 기준 (이상 점수 생성 방식, 학습 데이터 설정, 대상 데이터 형태, 모델 구조) 을 수립하고 주요 계열을 체계화한다; (2)**방법론 분석** - One-class 설정의 공통 문제 구조, 방법론 계열별 공통 특징, 핵심 설계 패턴 (Loss Function, Latent Space Regularization, Architecture), 방법론 간 비교와 진화 경향을 분석한다; (3)**실험결과 분석** - 평가 지표, 데이터셋 유형, 성능 패턴, 데이터 효율성, 성능 한계, 결과 해석의 경향 등을 종합적으로 정리한다.

### 4. 보고서 구성

- **1 장. 연구체계 분류**: 이상 탐지 논문들을 방법론적 특징 (재구성/분류/생성/하이브리드 모델, 학습 데이터 설정, 데이터 형태, 모델 구조) 을 기준으로 15 개 대분류와 하위 범주로 체계화하며, 재구성 기반, 생성 모델 기반, 분류기 기반, 하이브리드, 시계열/문맥 기반, 준감독/완전감독, 공정성/견고성, 설명 가능 탐지, 그룹/집단 탐지, 메모리/상태 기반, 효율성/실시간, 의료/산업 특수 도메인, 다중모달, 능동학습 기반, survey 등 주요 범주를 정리한다.

- **2 장. 방법론 분석**: One-Class Anomaly Detection 의 공통 문제 구조 (정상 분포 학습, 이상 판별) 와 6 가지 주요 계열 (재구성 기반, 판별기 기반, 생성 모델 기반, SVDD 기반, 하이브리드, 특수 설계 및 확장) 을 분석한다. 또한 핵심 설계 패턴 (One-class 설정, Loss Function, Latent Space Regularization, Architecture Patterns) 과 각 계열 간 비교, 트레이드오프, 진화 경향을 종합적으로 정리한다.

- **3 장. 실험결과 분석**: 평가 지표 (ROC AUC, AUROC, AUC-PR 등), 데이터셋 유형 (의료 영상, 이미지, 시계열, IoT), 성능 패턴 (feature reconstruction 의 우위, contamination 환경에서의 강인성), 성능 한계 (데이터셋 의존성, 평가 지표 한계, 계산 비용 분석 부재), 결과 해석 경향 등을 종합적으로 분석하고, 데이터 규모, 오염 비율, 평가 목표에 따른 방법론 선택 지침을 제시한다.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 분류체계는 논문들의 **(1) 이상 점수 생성 방식**(재구성/생성/분류기/하이브리드),**(2) 학습 데이터 설정**(비지도/준감독/완전감독/강화학습),**(3) 대상 데이터 형태**(이미지/시계열/멀티모달/고차원 수치),**(4) 모델 구조**(AE/GAN/Transformer/SVM/FCN 등) 를 종합적으로 고려하여 수립되었다. 중복 분류를 최소화하기 위해 각 논문은 가장 핵심적인 방법론적 차별점에 따라 단수 분류로 배치되었으며, 유사한 접근을 가진 논문들을 논리적으로 집합한 계층적 분류체계를 구성하였다.

### 2. 연구 분류 체계

#### 2.1 재구성 기반 탐지 (Reconstruction-based Detection)

정상 데이터의 특징적 구조를 복원 문제로 모델링하여 재구성 오류를 이상 점수로 활용하는 기법들.

| 분류                                        | 논문명                                                                                           | 분류 근거                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| 재구성 기반 > VAE 기반                      | A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based VAE (2017)          | LSTM-VAE 를 통한 고차원 멀티모달 시계열의 확률적 재구성 |
| 재구성 기반 > Autoencoder 기반              | Anomaly Detection for a Water Treatment System Using Unsupervised Machine Learning (2017)        | LSTM 기반 조건부 생성 모델 (DAE) 의 재구성 오류 점수화  |
| 재구성 기반 > AE+Perceptual Loss            | Anomaly Detection in Medical Imaging with Deep Perceptual Autoencoders (2020)                    | Perceptual Loss 기반 AE 로 재구성 손실을 최적화         |
| 재구성 기반 > Dual AE                       | Anomaly Detection with Adversarial Dual Autoencoders (2019)                                      | dual AE 의 generator/discriminator 재구성 오류 점수화   |
| 재구성 기반 > Hybrid AE+SVDD                | DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection (2021)            | reconstruction error 와 latent-distance 가중합          |
| 재구성 기반 > Discriminative Reconstruction | DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection (2021) | reconstruction sub-network 와 discriminative U-Net 결합 |

#### 2.2 생성 모델 기반 탐지 (Generative Model-based Detection)

GAN 기반 생성 모델을 통한 정상 분포 학습 및 재구성/분류 기반 이상 탐지.

| 분류                          | 논문명                                                                                               | 분류 근거                                                               |
| ----------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 생성 모델 기반 > GAN 정상분포 | Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (2017) | DCGAN 으로 정상 분포 학습 후 iterative latent mapping 에 의한 이상 탐지 |

#### 2.3 분류기 기반 탐지 (Classification-based Detection)

normality scoring 또는 one-class classification 과함수를 통한 이상 판별.

| 분류                                         | 논문명                                                        | 분류 근거                                                                |
| -------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 분류기 기반 > Self-supervised Classification | Deep Anomaly Detection Using Geometric Transformations (2018) | Dirichlet 기반 transformation discriminator 로 normality score           |
| 분류기 기반 > End-to-End One-Class NN        | Anomaly Detection using One-Class Neural Networks (2018)      | OC-SVM 목적함수와 neural representation 의 end-to-end 학습               |
| 분류기 기반 > Deviation Network              | Deep Anomaly Detection with Deviation Networks (2019)         | Z-score 기반 deviation loss 로 anomaly score 직접 최적화                 |
| 분류기 기반 > Deep Fair SVDD                 | Towards Fair Deep Anomaly Detection (2020)                    | encoder 가 sensitive attribute 를 예측하지 못하도록 adversarial training |

#### 2.4 하이브리드 모델 탐지 (Hybrid Model Detection)

다양한 모델 기법을 결합한 통합 프레임워크.

| 분류                                                 | 논문명                                                                                                              | 분류 근거                                               |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| 하이브리드 모델 > AE+RFF+OC-SVM                      | Scalable and Interpretable One-class SVMs with Deep Learning and Random Fourier features (2018)                     | RFF mapping 으로 AE bottleneck 에 OC-SVM joint learning |
| 하이브리드 모델 > Transformer+Feature Reconstruction | ADTR: Anomaly Detection Transformer with Feature Reconstruction (2022)                                              | EfficientNet feature 를 transformer query 로 복원       |
| 하이브리드 모델 > Normality+Deviation                | Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection (2019) | memory module 과 reconstruction entropy combined loss   |

#### 2.5 시계열/문맥 기반 탐지 (Temporal/Contextual Detection)

시간적/주파수적 정보를 반영한 시계열 이상 탐지.

| 분류                  | 논문명                                                                                        | 분류 근거                                                 |
| --------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 시계열 > HTM 예측모델 | Real-Time Anomaly Detection for Streaming Analytics (2016)                                    | HTM 기반 sparse representation 과 branching sequence 예측 |
| 시계열 > 주파수 관점  | Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective (2024) | GFM+LFM 을 활용한 global/local frequency condition        |

#### 2.6 준감독/완전 감독 탐지 (Semi-supervised/Supervised Detection)

일부 이상 레이블이나 partial label 을 활용한 탐지.

| 분류                       | 논문명                                                                                                      | 분류 근거                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 준감독 > partially labeled | Toward Deep Supervised Anomaly Detection: Reinforcement Learning from Partially Labeled Anomaly Data (2021) | 레이블된 이상과 비레이블 데이터를 강화학습으로 공동 최적화    |
| 준감독 > known/unknown     | Toward Supervised Anomaly Detection (2014)                                                                  | SVDDneg/SSAD 로 양/음 라벨 모두 반영하는 semi-supervised 학습 |
| 준감독 > DevNet            | Deep Anomaly Detection with Deviation Networks (2019)                                                       | 30 개 labeled anomaly 를 Gaussian prior 로 활용               |

#### 2.7 공정성/견고성 탐지 (Fair/Robust Detection)

민감 속성 편향 제거 또는 adversarial 환경 견고성.

| 분류                          | 논문명                                                                   | 분류 근거                                           |
| ----------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------- |
| 견고성 > Adversarial defense  | Adversarially Robust One-class Novelty Detection (2021)                  | Vector-PCA+Spatial-PCA 의 latent space PCA defense  |
| 견고성 > Contamination robust | Robust Anomaly Detection in Images using Adversarial Autoencoders (2019) | AAE 로 latent prior 부과 및 training set refinement |

#### 2.8 설명 가능 탐지 (Explainable Detection)

model output 을 직접 anomaly heatmap 으로 활용.

| 분류                        | 논문명                                           | 분류 근거                                 |
| --------------------------- | ------------------------------------------------ | ----------------------------------------- |
| 설명 가능 > FCN explanation | Explainable Deep One-Class Classification (2020) | FCN output map 합을 anomaly score 로 정의 |

#### 2.9 그룹/집단 탐지 (Group/Collective Detection)

그룹 전체 분포가 이상인지 판별.

| 분류                     | 논문명                                                      | 분류 근거                                           |
| ------------------------ | ----------------------------------------------------------- | --------------------------------------------------- |
| 집단탐지 > Group anomaly | Group Anomaly Detection using Deep Generative Models (2018) | VAE/AAE 를 그룹 분포의 잠재 통계 평균화 틀에 정식화 |

#### 2.10 메모리/상태 기반 탐지 (Memory/State-based Detection)

memory module, state tracking, progress adaptation 등을 활용한 탐지.

| 분류        | 논문명                                                                                                              | 분류 근거                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 메모리 기반 | Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection (2019) | hard shrinkage 기반 sparse addressing 으로 memory 모듈    |
| 상태 기반   | A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based VAE (2017)                             | state-based thresholding 으로 progress-adaptive threshold |
| 상태 기반   | Anomaly Detection using One-Class Neural Networks (2018)                                                            | progress-based prior 로 monotonic sequence 가정           |

#### 2.11 효율성/실시간 탐지 (Efficient/Real-time Detection)

확장성 및 실시간 성능 최적화.

| 분류               | 논문명                                                                            | 분류 근거                                                       |
| ------------------ | --------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 효율성 > IoT OCSVM | An Efficient One-Class SVM for Anomaly Detection in the Internet of Things (2021) | Nyström 근사와 Gaussian Sketching 으로 대규모 IoT 데이터셋 확장 |
| 실시간 > Streaming | Real-Time Anomaly Detection for Streaming Analytics (2016)                        | 연속 학습과 multi-model ensemble 로 실시간 스트림 처리          |

#### 2.12 의료/산업 특수 도메인 (Domain-specific Detection)

특정 도메인 (의료/물처리/산업비전/로봇 피딩) 에 특화된 탐지.

| 분류         | 논문명                                                                                               | 분류 근거                          |
| ------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 의료이미지   | Anomaly Detection in Medical Imaging with Deep Perceptual Autoencoders (2020)                        | Camelyon16/NIH 의료이미지 벤치마크 |
| 의료이미지   | Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (2017) | retina SD-OCT 영상 분석            |
| 물처리시스템 | Anomaly Detection for a Water Treatment System Using Unsupervised Machine Learning (2017)            | SWaT 물 처리 테스트베드            |
| 산업비전     | DRAEM - A discriminatively trained reconstruction embedding for surface anomaly detection (2021)     | MVTec AD 산업 표면 검사            |
| 산업비전     | ADTR: Anomaly Detection Transformer with Feature Reconstruction (2022)                               | MVTec-AD 벤치마크 15 카테고리      |
| 로봇피딩     | A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based VAE (2017)              | robot-assisted feeding 시스템      |

#### 2.13 다중모달 탐지 (Multimodal Detection)

여러 센서/모달리티 융합을 통한 탐지.

| 분류     | 논문명                                                                                  | 분류 근거                                                            |
| -------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 다중모달 | A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based VAE (2017) | RGB-D camera, force/torque, joint encoders 등 5 종 센서 17 차원 신호 |

#### 2.14 능동학습 기반 탐지 (Active Learning-based Detection)

전문가 피드백을 통한 레이블 효율적 학습.

| 분류     | 논문명                                            | 분류 근거                                              |
| -------- | ------------------------------------------------- | ------------------------------------------------------ |
| 능동학습 | Deep Active Learning for Anomaly Detection (2018) | UAI 레이어를 통해 정상/이상 데이터를 능동학습으로 분리 |

#### 2.15 survey

Anomaly detection 방법론에 대한 체계적 정리.

| 분류   | 논문명                                                                         | 분류 근거                                                                 |
| ------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| survey | A Survey on Anomaly Detection for Technical Systems using LSTM Networks (2021) | LSTM 기반 이상 탐지 방법론을 체계적 정리 (Regular/Encoder-Decoder/Hybrid) |

### 3. 종합 정리

분석 대상 논문들은 (1) 이상 탐지 목적함수의 정의 방식 (재구성/분류/생성/하이브리드), (2) 학습 데이터의 라벨링 정도 (비지도/준감독/완전감독), (3) 대상 데이터 형태 (이미지/시계열/멀티모달/고차원 수치), (4) 모델 아키텍처 (AE/GAN/Transformer/SVM/FCN) 를 기준으로 15 개 대분류와 여러 하위 범주로 체계화되었다. 재구성 기반 방법 (VAE, Autoencoder, Perceptual Loss 등) 이 가장 많은 논문을 차지하며, 생성 모델 (GAN), 분류기 기반 (Self-supervised classification, Deviation Network), 하이브리드 (AE+SVDD, Transformer, Normality+Deviation) 접근법도 활발히 연구된 것으로 나타났다. 학습 데이터 설정에 따라 준감독/완전감독 방법 (DevNet, Reinforcement Learning, SVDDneg) 과 효율성/실시간성 (OCSVM 근사, HTM 예측) 을 개선하는 연구도 포함되어 있으며, 공정성 견고성 (Adversarial defense, Contamination robust), 설명 가능성 (FCN explanation), 메모리/상태 (Memory module, state tracking) 등의 보조적 고려사항도 반영되었다. 도메인 적용 관점에서 의료이미지, 산업비전 (MVTec-AD), 물처리시스템, 로봇 피딩 등 특수 환경 탐지 논문이 포괄되며, 다중모달 (5 종 센서 융합), 그룹/집단, 능동학습 기반 탐지 등도 연구 영역으로 포함되었다.

## 2 장. 방법론 분석

### 1. 공통 문제 설정 및 접근 구조

전체 논문들은 **One-Class Anomaly Detection**이라는 공통된 문제 설정을 기반으로 한다:

| 구분            | 내용                                                  |
| --------------- | ----------------------------------------------------- |
| **문제 유형**   | One-class / Unsupervised Anomaly Detection            |
| **학습 데이터** | 정상 (normal) 데이터만 사용                           |
| **목표**        | 학습 분포와 현저히 다른 입력을 이상(anomaly)으로 판별 |
| **출력**        | 이상 점수 (anomaly score) 또는 이진 판정              |
| **판정 기준**   | 임계값 (threshold) 또는 분위수 (quantile) 기반        |

#### 방법론적 공통 구조

```text
학습 단계
├─ 정상 분포/패턴 학습
│  ├─ 결정 경계 학습 (SVM 기반)
│  ├─ 분포 모델링 (VAE, GAN 기반)
│  └─ 판별기 학습 (Discriminative classifier)
│
테스트 단계
├─ 입력 → 분포/모델 → 재구성/점수 생성
├─ 재구성 오차 또는 판정 거리 계산
└─ 임계값 비교 → 이상/정상 판정
```

### 2. 방법론 계열 분류

다양한 논문들을 방법론적 접근 방식에 따라 6 개 계열로 분류한다:

#### (1) 재구성 기반 (Reconstruction-based)

**계열 정의**: Autoencoder 기반 구조를 통해 정상 패턴의 재구성 능력을 학습하고, 재구성 오차를 이상 점수로 사용하는 접근법.

**공통 특징**:

- Encoder-decoder 구조로 정상 분포 학습
- 재구성 오차 (L2, MSE, Perceptual Loss 등) 를 이상 점수
- One-class 설정에서 trivial solution (모든 입력 재구성) 방지 기법 포함

**해당 논문**:

| 방법론 계열          | 논문명                                                                                    | 핵심 특징                                                     |
| -------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Reconstruction-based | Memorizing Normality to Detect Anomaly (2019)                                             | Memory-augmented AE 로 정상 prototype 만 조회하여 복원        |
| Reconstruction-based | Robust Anomaly Detection in Images using Adversarial Autoencoders (2019)                  | AAE 잠재 공간 정제 + 1-class SVM 으로 오염 견고성             |
| Reconstruction-based | Anomaly Detection using One-Class Neural Networks (2018)                                  | OC-NN 은 one-class objective 가 representation 학습 직접 통제 |
| Reconstruction-based | Anomaly Detection for a Water Treatment System Using Unsupervised Machine Learning (2017) | DNN(LSTM) + one-class SVM 하이브리드 비교                     |

#### (2) 판별기 기반 (Discriminative)

**계열 정의**: 정상 클래스를 판별기 (classifier) 로 학습하고, 판별점 (discriminative score) 을 이상 점수로 사용하는 접근법.

**공통 특징**:

- 정상 데이터만 정상 클래스로 레이블
- Softmax 출력 벡터 또는 decision boundary 를 판정 기준
- Reconstruction error 대신 판별기 출력 직접 활용

**해당 논문**:

| 방법론 계열    | 논문명                                                              | 핵심 특징                                        |
| -------------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| Discriminative | Deep Anomaly Detection Using Geometric Transformations (2018)       | Self-labeled multi-class 분류기 + Dirichlet 분포 |
| Discriminative | Scalable and Interpretable One-class SVMs with Deep Learning (2018) | AE bottleneck + RFF-OC-SVM end-to-end            |
| Discriminative | Explainable Deep One-Class Classification (2020)                    | FCN 출력과 anomaly score 동일화                  |

#### (3) 생성 모델 기반 (Generative)

**계열 정의**: GAN 또는 VAE 기반 생성 모델을 학습하고, 잠재 공간 매핑 또는 재구성 실패를 이상으로 탐지하는 접근법.

**공통 특징**:

- GAN: 정상 manifold 학습 후 latent mapping, feature matching
- VAE: ELBO 최대화 + reconstruction likelihood 기반 scoring
- Adversarial training 또는 feature matching 추가

**해당 논문**:

| 방법론 계열 | 논문명                                                                                               | 핵심 특징                                              |
| ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Generative  | Anomaly Detection with Adversarial Dual Autoencoders (2019)                                          | GAN 생성기+판별기 모두 AE 로 학습 안정성               |
| Generative  | Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (2017) | AnoGAN: latent mapping + feature matching loss         |
| Generative  | Group Anomaly Detection using Deep Generative Models (2018)                                          | VAE/AAE 기반 group-level distribution 기반 이상성 검출 |

#### (4) SVDD 기반 (Support Vector Data Description)

**계elas**: 정상 데이터의 저차원 잠재 표현을 단일 hypersphere 로 요약하는 접근법.

**공통 특징**:

- Latent space 중심 c와 반지름 R 학습
- 이상 점수 = ‖z − c‖²
- Trainable center 또는 trainable SVDD 로 trivial solution 방지

**해당 논문**:

| 방법론 계열 | 논문명                                                          | 핵심 특징                                      |
| ----------- | --------------------------------------------------------------- | ---------------------------------------------- |
| SVDD-based  | Toward Supervised Anomaly Detection (2014)                      | SVDD + labeled constraints 로 margin 확장      |
| SVDD-based  | DASVDD: Deep Autoencoding Support Vector Data Descriptor (2021) | AE 복원 능력 + SVDD compactness hybrid         |
| SVDD-based  | Adversarially Robust One-class Novelty Detection (2021)         | PrincipaLS(Vector-PCA+Spatial-PCA)净化         |
| SVDD-based  | Elliptic stable envelopes (2016)                                | 기하학적 stable envelope + elliptic cohomology |

#### (5) 하이브리드 (Hybrid)

**계열 정의**: 서로 다른 방법론 (e.g., AE+VAE, AE+GAN, VAE+GAN, AE+SVM) 을 결합한 접근법.

**공통 특징**:

- 한 방법은 표현 학습, 다른 방법은 판정 기준 역할
- Joint loss 또는 alternating optimization 으로 최적화
- 장점 조합: 표현 학습능력 + 판정 신뢰성

**해당 논문**:

| 방법론 계열 | 논문명                                                                        | 핵심 특징                                                 |
| ----------- | ----------------------------------------------------------------------------- | --------------------------------------------------------- |
| Hybrid      | Anomaly Detection in Medical Imaging with Deep Perceptual Autoencoders (2020) | Perceptual Loss + Progressive Growing + weakly supervised |
| Hybrid      | Adversarially Robust One-class Novelty Detection (2021)                       | AE + PrincipaLS(Vector-PCA+Spatial-PCA)净化               |
| Hybrid      | DRAEM - A discriminatively trained reconstruction embedding (2021)            | Reconstruction + Discrimination joint modeling            |

#### (6) 특수 설계 및 확장 (Specialized Design & Extensions)

**계열 정의**: 특정 문제 설정 (group-level, streaming, streaming, streaming, frequency, fairness) 에 맞춘 특수 설계.

**공통 특징**:

- Group-level 이상성: aggregation function 으로 전체 집합 이상성 판별
- Streaming: HTM 기반 online learning + rolling 통계
- Frequency: GFM+LFM 으로 주파수 조건화
- Fairness: Adversarial discriminator 로 민감 속성 제거

**해당 논문**:

| 방법론 계열 | 논문명                                                               | 핵심 특징                                             |
| ----------- | -------------------------------------------------------------------- | ----------------------------------------------------- |
| Specialized | Group Anomaly Detection using Deep Generative Models (2018)          | f, g, d 함수로 group-level 이상성 판별                |
| Specialized | Real-Time Anomaly Detection for Streaming Analytics (2016)           | HTM 예측 + rolling 평균 + Q-function tail probability |
| Specialized | Revisiting VAE for Unsupervised Time Series Anomaly Detection (2024) | GFM+LFM 으로 주파수 조건화 + CVAE                     |
| Specialized | Towards Fair Deep Anomaly Detection (2020)                           | Adversarial discriminator 로 sensitive attribute 제거 |

### 3. 핵심 설계 패턴 분석

다양한 계열 간에도 반복적으로 나타나는 설계 패턴을 추출:

#### (1) One-class 설정과 Training Strategy

| 패턴                              | 내용                                                         | 적용 논문 예시                                             |
| --------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| 정상 데이터만 학습                | One-class 또는 unsupervised 설정에서 normal-only 데이터 사용 | Reconstruction-based, Discriminative, Generative 계열 모두 |
| Threshold 결정                    | Quantile(v-quantile) 또는 validation set 기반 임계값 설정    | OC-NN, DPLAN, SVDD 기반 모두                               |
| Active Learning / Semi-supervised | 소량 labeled data 로 모델 개선                               | Deep Active Learning, Toward Supervised, DPLAN             |

#### (2) Loss Function Design

| 패턴                      | 내용                                                | 적용 논문 예시                  |
| ------------------------- | --------------------------------------------------- | ------------------------------- |
| Reconstruction Loss       | L2, MSE, Perceptual Loss 등으로 복원 오차           | AE, AAE, MemAE, DRAEM           |
| Adversarial Loss          | Generator-Discriminator minimax 게임                | AnoGAN, AAE, Fair SVDD, Dual AE |
| Hinge Loss / Pseudo-Huber | One-class objective 위한 margin 기반 손실           | OC-NN, SVDD                     |
| Joint Loss                | Reconstruction + Discrimination + Regularization 합 | DRAEM, Scalable OC-SVM          |

#### (3) Latent Space Regularization

| 패턴                   | 내용                                            | 적용 논문 예시     |
| ---------------------- | ----------------------------------------------- | ------------------ |
| Prior Distribution     | Isotropic Gaussian prior 또는learnable prior    | LSTM-VAE, VAE 계열 |
| PCA-based Purification | Vector-PCA + Spatial-PCA 로 latent purification | PrincipaLS         |
| Memory Module          | Normal prototype memory 로 복원 경로 제한       | MemAE              |
| RFF Mapping            | Random Fourier Feature 로 kernel 근사           | Scalable OC-SVM    |

#### (4) Architecture Patterns

| 패턴                     | 내용                                          | 적용 논문 예시            |
| ------------------------ | --------------------------------------------- | ------------------------- |
| Encoder-Decoder          | Standard AE 구조로 reconstruction             | Reconstruction-based 계열 |
| U-Net / Skip Connections | Feature preservation 으로 segmentation 정확도 | DRAEM, Medical Imaging    |
| FCN / Receptive Field    | 공간 구조 보존                                | Explainable ODC           |
| Multi-scale Feature      | Pyramid feature 로 semantic 정보 활용         | ADTR                      |

### 4. 방법론 비교 분석

각 계열 간 차이점과 트레이드오프:

| 비교 차원     | 재구성 기반                           | 판별기 기반                 | 생성 모델 기반              | SVDD 기반              |
| ------------- | ------------------------------------- | --------------------------- | --------------------------- | ---------------------- |
| **문제 접근** | Normal 분포 복원                      | 정상/비정상 분류            | Normal manifold 학습        | Latent compact cluster |
| **주요 신호** | $\vert\vert\hat{x} − x\vert\vert^2$   | Softmax / decision boundary | Reconstruction chain error  | $‖z − c‖^2$            |
| **복잡도**    | 중간 (Encoder+Decoder)                | 낮음 (Classifier)           | 높음 (Adversarial dynamics) | 낮음 (Hypersphere)     |
| **확장성**    | 이미지/시계열 모두 가능               | 대규모 데이터 효율적        | GPU 메모리 요구 큼          | 중간                   |
| **Weakness**  | Trivial solution, Over-reconstruction | Normal-only class imbalance | GAN 불안정성                | Hypersphere collapse   |

**Trayd-off 분석**:

1. **재구성 vs 판별**:
   - 재구성: Over-reconstruction 문제 있으나 일반성 좋음
   - 판별: Reconstruction 없이 판정 가능으나 decision boundary 학습 어려움

2. **GAN vs VAE**:
   - GAN: 복잡한 manifold 학습 가능하나 학습 불안정
   - VAE: 안정적 학습으나 manifold 표현 제한적

3. **End-to-End vs Two-Stage**:
   - Two-stage (AE+OC-SVM): Modular 이나 표현 학습과 판정 분리
   - End-to-End (Hybrid): Joint loss 로 표현 학습과 판정 통합

### 5. 방법론 흐름 및 진화

시간 흐름에 따른 방법론 발전 경향:

| 시기          | 주요 접근법                                  | 특징                                                 |
| ------------- | -------------------------------------------- | ---------------------------------------------------- |
| **2014-2017** | SVDD, One-class SVM, 기본 AE                 | 수학적 최적화 기반, 전통적 one-class 방법            |
| **2017-2019** | AnoGAN, GAN 기반, Deep AE                    | 생성 모델 도입, feature matching 추가                |
| **2018-2021** | OC-NN, MemAE, PrincipaLS, Dual AE            | One-class objective 직접화, latent purification      |
| **2020-2024** | Fairness, Frequency conditioning, End-to-End | 특수 문제 (fairness, frequency) 대응, joint learning |

**진화 방향**:

1. **문제 설정 확장**:
   - Normal-only → Semi-supervised → End-to-end anomaly score 학습

2. **Objective 통합**:
   - Reconstruction + Discrimination + Regularization → Joint objective

3. **특수 문제 대응**:
   - Streaming, Group-level, Fairness, Frequency conditioning

### 6. 종합 정리

본 장에서 분석한 방법론 지형은 **두 차원 축**으로 구조화된다:

| 축                           | 구분             | 내용                                    |
| ---------------------------- | ---------------- | --------------------------------------- |
| **분포 모델링 vs 판별 학습** | 재구성/생성 모델 | 정상 분포 학습 후 재구성/latent mapping |
|                              | 판별기           | Decision boundary 또는 softmax 판별점   |
| **분산 가정**                | 명시적           | Gaussian prior, PCA 기반 정화           |
|                              | 암묵적           | Data-driven representation 학습         |

**방법론 계층**:

1. **기저**: SVDD, OC-SVM (수학적 최적화 기반)
2. **확장**: AE, GAN, VAE (표현 학습능력 추가)
3. **통합**: Hybrid, End-to-End (joint objective)
4. **특화**: Fairness, Streaming, Group-level, Frequency

전체적으로는 **Reconstruction ↔ Discrimination ↔ Generation**이라는 삼각 관계를 중심으로 한 방법론적 지형도이며, 각 접근법은**One-class 설정의 제약 하에서 Normal 분포 학습과 Anomaly 판별을 동시에 달성**하는 방법을 모색하고 있다.

*참고: 논문 제목은 (연도) 형식으로 표기하며, 제공된 방법론 문서 모음 외의 내용은 포함하지 않습니다.*

## 3장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 1.1 주요 평가 지표

각 논문들은 다음과 같은 평가 지표를 주로 사용하였다:

| 평가 지표               | 사용 빈도 | 주요 활용 분야                     |
| ----------------------- | --------- | ---------------------------------- |
| **ROC AUC**             | 16 건     | 의료 이미지, 일반 이미지 이상 탐지 |
| **AUROC**               | 14 건     | 이미지 기반 anomaly detection      |
| **AUC-PR**              | 9 건      | 불균형 데이터셋 기반 실험          |
| **Precision/Recall/F1** | 6 건      | 실시간 이상 탐지, 사이버보안       |
| **mAUROC**              | 5 건      | 다중 데이터셋 평균 성능 평가       |
| **AUPRC/AUROC**         | 5 건      | 클래스 불균형 상황                 |
| **NAB score**           | 1 건      | 스트리밍 환경 이상 탐지            |
| **F1-score/delay F1**   | 4 건      | 시계열 이상 탐지                   |

#### 1.2 평가 환경 유형

| 환경 유형                       | 포함 논문                                                                                                                   | 특징                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **실제 IoT 트래픽**             | An Efficient One-Class SVM... (2021)                                                                                        | 스마트 홈/시티/건물 인프라           |
| **물리 시스템**                 | Anomaly Detection for a Water Treatment... (2017)                                                                           | SWaT 물 처리 CPS                     |
| **의료 영상**                   | Anomaly Detection in Medical Imaging... (2020), Group Anomaly Detection... (2018), Unsupervised Anomaly Detection... (2017) | Camelyon16, NIH, BRATS, retinal OCT  |
| **시계열 데이터**               | VAE for Unsupervised Time Series... (2024), A Multimodal Anomaly Detector... (2017)                                         | Yahoo, KPI, WSD, NAB, PR2            |
| **이미지/이미지-비이미지 혼합** | Adversarially Robust One-class Novelty... (2021), ADTR... (2022)                                                            | MNIST/CIFAR/Fashion-MNIST + MVTec-AD |
| **네트워크/사이버보안**         | Deep Anomaly Detection Using Geometric... (2018), Real-Time Anomaly... (2016)                                               | FIRST, KDDCUP99, NAB                 |

#### 1.3 비교 방식

| 비교 유형                               | 사용 논문                                                    | 비교 대상                         |
| --------------------------------------- | ------------------------------------------------------------ | --------------------------------- |
| **Baseline 방법**                       | 대부분의 논문                                                | AE, VAE, SVM, Isolation Forest 등 |
| **SOTA 방법**                           | Anomaly Detection in Medical Imaging... (2020)               | AnoGAN, GANomaly, DeepSVDD 등     |
| **기존 GAN 방법**                       | Anomaly Detection using Adversarial Dual Autoencoders (2019) | AnoGAN, EGBAD, GANomaly, IGMM-GAN |
| **전통적 ML vs 딥러닝**                 | Anomaly Detection for a Water Treatment... (2017)            | DNN vs SVM                        |
| **Reconstruction 기반 vs Feature 기반** | ADTR... (2022)                                               | pixel vs feature reconstruction   |
| **Supervised vs Unsupervised**          | DPLAN... (2021), Deep Active Learning... (2018)              | 레이블 유무 비교                  |

### 2. 주요 실험 결과 정렬

#### 2.1 이미지/이미지-비이미지 데이터셋별 성능

**MVTec-AD 벤치마크 (image-level AUROC)**

| 논문                              | 방법                          | AUROC             | 상대 성능                           |
| --------------------------------- | ----------------------------- | ----------------- | ----------------------------------- |
| DRAEM (2021)                      | reconstruction + segmentation | 98.0              | +2.5 point vs 기존 최고 95.5        |
| Explainable Deep One-Class (2020) | FCN 구조                      | 0.92 (pixel-wise) | SOTA 근접                           |
| ADTR+ (2022)                      | feature reconstruction        | 96.9              | +0.9 vs ADTR 96.4                   |
| ADTR (2022)                       | feature reconstruction        | 96.4              | +6.2 vs TS 92.5, +7.6 vs SPADE 96.0 |
| TS (2022)                         | teacher-student               | 92.5              | +4.5 vs SPADE 96.0                  |
| SPADE (2022)                      | pixel reconstruction          | 96.0              | 베이스라인                          |

**MNIST/CIFAR 기반 일반 이미지 데이터셋 (AUROC)**

| 논문                                                         | 방법             | MNIST               | CIFAR-10                  |
| ------------------------------------------------------------ | ---------------- | ------------------- | ------------------------- |
| Adversarially Robust One-class Novelty (2021)                | PrincipaLS       | 0.706 (clean 0.973) | 0.578                     |
| Anomaly Detection using One-Class Neural Networks (2018)     | OC-NN            | 87~99               | 자동차 62, 새 64, 사슴 67 |
| Deep Anomaly Detection with Deviation Networks (2019)        | DevNet           | N/A                 | AUC-PR 0.574              |
| Anomaly Detection using Adversarial Dual Autoencoders (2019) | ADAE             | 0.858 (평균)        | 0.610 (평균)              |
| DSVDD (2021)                                                 | trainable center | 97.7%               | 66.5%                     |
| Anomaly Detection in Medical Imaging... (2020)               | DPA              | N/A                 | 83.9% (CIFAR10)           |

**Fashion-MNIST 기반 성능**

| 논문                                          | 방법             | AUC               |
| --------------------------------------------- | ---------------- | ----------------- |
| Explainable Deep One-Class (2020)             | FCN 구조         | 0.82              |
| Adversarially Robust One-class Novelty (2021) | PrincipaLS       | 0.613 (PGD 0.613) |
| DASVDD (2021)                                 | trainable center | 92.6%             |

#### 2.2 데이터셋 유형별 성능 패턴

**의료 영상 (ROC AUC)**

| 데이터셋    | 방법   | AUROC | 향상 정도            |
| ----------- | ------ | ----- | -------------------- |
| Camelyon16  | DPA    | 93.4% | +2.8% vs 기존 최첨단 |
| NIH         | DPA    | 92.6% | +5.2% vs 기존 최첨단 |
| UCSD-Ped2   | MemAE  | 0.941 | +0.024 vs AE 0.917   |
| CUHK Avenue | MemAE  | 0.833 | +0.023 vs AE 0.810   |
| retinal OCT | AnoGAN | 0.89  | +0.16 vs aCAE 0.73   |

**시계열 데이터 (F1)**

| 데이터셋 | 방법  | best F1 | delay F1 |
| -------- | ----- | ------- | -------- |
| Yahoo    | FCVAE | 0.857   | 0.842    |
| KPI      | FCVAE | 0.927   | 0.835    |
| WSD      | FCVAE | 0.831   | 0.631    |
| NAB      | FCVAE | 0.976   | 0.917    |

#### 2.3 데이터 오염/불균형 상황에서의 강인성

**학습셋 오염에 대한 적응 (BAcc)**

| 논문                                     | 오염율 | AE   | AAE  | ITSR (최고)          |
| ---------------------------------------- | ------ | ---- | ---- | -------------------- |
| Anomaly Detection using Robust... (2019) | 5%     | 0.69 | 0.91 | 0.94 (MNIST)         |
| (2019)                                   | 5%     | 0.74 | 0.70 | 0.81 (Fashion-MNIST) |

**불균형 데이터셋 (AUPRC)**

| 논문                                 | 데이터셋    | 방법     | AUPRC  |
| ------------------------------------ | ----------- | -------- | ------ |
| Scalable and Interpretable... (2018) | ForestCover | AE-1SVM  | 0.1976 |
| Scalable and Interpretable... (2018) | Shuttle     | AE-1SVM  | 0.9483 |
| Scalable and Interpretable... (2018) | KDDCup99    | AE-1SVM  | 0.5115 |
| Scalable and Interpretable... (2018) | USPS        | AE-1SVM  | 0.8024 |
| Scalable and Interpretable... (2018) | MNIST       | CAE-1SVM | 0.0885 |

#### 2.4 데이터 효율성 및 소량 레이블 상황

| 논문                           | 레이블 데이터                   | 성능 지표    | 향상                |
| ------------------------------ | ------------------------------- | ------------ | ------------------- |
| DevNet (2019)                  | 30개 labeled anomaly (0.005~1%) | AUC-PR 0.574 | +9% vs REPEN        |
| DPLAN (2021)                   | 소수 레이블 이상치              | AUC-PR       | +7% vs DevNet       |
| Deep Active Learning... (2018) | 0.17% 이상 (Covtype)            | AUC          | 최상 (유저에이블로) |
| SSAD (2014)                    | 15% labeled data                | AUC          | 포화 달성           |

### 3. 성능 패턴 및 경향 분석

#### 3.1 공통적인 성능 개선 패턴

**Pattern 1: Feature reconstruction이 pixel reconstruction보다 우수**

- ADTR (2022) 에서 feature reconstruction(97.2)이 pixel reconstruction(91.3) 대비 5.9% 향상
- semantic discrepancy 를 활용한 normal/anomaly 분리가 더 명확함

**Pattern 2: Discriminator의 reconstruction error가 anomaly score 로 더 효과적**

- ADAE (2019) 에서 discriminator reconstruction error 가 generator reconstruction error 보다 clearer anomaly separation
- generator 가 normal 데이터 잘 복원하는 반면, discriminator 가 normal vs anomaly 분리가 더 명확함

**Pattern 3: Reconstruction quality + center optimization 이 SOTA**

- DASVDD (2021) 가 reconstruction error + latent-center distance 동시 최적화로 SOTA
- MNIST 97.7%, CIFAR-10 66.5%, Fashion-MNIST 92.6% 등 일관된 최고 성능

**Pattern 4: Sparse addressing 및 memory 구조가 재구성 억제에 핵심**

- MemAE (2019) 가 hard shrinkage 와 entropy regularization 으로 normal prototype 만 복원
- 7 벤치마크에서 일관된 +2~3% AUC/F1 향상

#### 3.2 특정 조건에서만 성능 향상되는 경우

**조건 1: 데이터셋 규모에 따른 방법론 차이**

| 데이터 규모     | 우세 방법                |
| --------------- | ------------------------ |
| 소규모 (M<5000) | 기존 GAD 방법 (OCSMM 등) |
| 대규모 (M>5000) | VAE/AAE 기반 DGM         |

*Group Anomaly Detection using Deep Generative Models (2018)*

**조건 2: 공격 강도에 따른 defense 성능**

| 공격 유형 | defense    | mAUROC                  |
| --------- | ---------- | ----------------------- |
| PGD       | PrincipaLS | 0.706                   |
| FGSM      | PrincipaLS | N/A (PGD 대비 우수)     |
| black-box | PrincipaLS | 0.706 (5 데이터셋 평균) |

*Adversarially Robust One-class Novelty Detection (2021)*

**조건 3: 오염 비율 증가 시 robustness**

| 오염율 | 방법         | 상대 성능                                  |
| ------ | ------------ | ------------------------------------------ |
| 0~2%   | DevNet       | +9% vs REPEN                               |
| 10%    | DPLAN        | +11% vs Deep SAD                           |
| 20%    | DevNet/DPLAN | contamination 환경에서도 일관된 robustness |

*Deep Anomaly Detection with Deviation Networks (2019), DPLAN (2021)*

#### 3.3 논문 간 상충되는 결과

**Trade-off: 계산 비용 vs 성능**

| 방법            | 계산 비용         | 성능                     |
| --------------- | ----------------- | ------------------------ |
| one-class SVM   | 낮음 (30 분 학습) | Precision 0.925          |
| DNN (LSTM 기반) | 높음 (2 주 학습)  | Precision 0.983, F 0.803 |

*Anomaly Detection for a Water Treatment System Using Unsupervised Machine Learning (2017)*

**Trade-off: 설명 가능성 vs 최상 성능**

| 설정            | 설명 가능성                  | 성능                      |
| --------------- | ---------------------------- | ------------------------- |
| FCN 구조        | 높음 (gradient보다 덜 noisy) | AUC 0.92 (SOTA 0.97 대비) |
| 일반 deep model | 낮음                         | AUC 0.97                  |

*Explainable Deep One-Class Classification (2020)*

**Trade-off: fairness vs AUC**

| 방법           | fairness (p%-rule) | AUC 손실 |
| -------------- | ------------------ | -------- |
| Deep Fair SVDD | 80% rule 만족      | 작음     |
| 일반 SVDD      | 낮음               | -        |

*Toward Fair Deep Anomaly Detection (2020)*

#### 3.4 데이터셋 또는 환경에 따른 성능 차이

**이미지 vs 비이미지 데이터셋**

| 데이터셋 유형                | DASVDD 성능 | DASVDD 제한점                        |
| ---------------------------- | ----------- | ------------------------------------ |
| 이미지 (MNIST/CIFAR)         | 97.7%/66.5% | fully connected AE 기반 표현력 제한  |
| 비이미지 (ODDS Speech, PIMA) | 경쟁력 입증 | 이미지 외부 modalities 에서도 경쟁력 |
| 대규모 이미지 (AWID)         | 96.3%       |                                      |

*DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection (2021)*

**시계열 vs 일반 데이터**

| 방법                | 시계열                | 일반 이미지 |
| ------------------- | --------------------- | ----------- |
| VAE 기반            | 제한적 (univariate만) | 우수        |
| frequency condition | 필수적                | 불필요      |

*Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective (2024)*

### 4. 추가 실험 및 검증 패턴

#### 4.1 ablation study 패턴

**Pattern: 구성 요소별 기여도 분리**

| 논문          | ablation 요소              | 영향                        |
| ------------- | -------------------------- | --------------------------- |
| ADTR (2022)   | attention + query 제거     | 성능 하락 (97.2→94.4)       |
| MemAE (2019)  | w/o Shrinkage, w/o Entropy | UCSD-Ped2 0.941→0.932/0.937 |
| DevNet (2019) | Rep, Linear, 3HL           | 3HL가 가장 우수             |
| DPA (2020)    | Perceptual Loss 제거       | Camelyon16 87.9%→21.1%      |
| FCVAE (2024)  | CM-ELBO 제거               | 가장 큰 낙폭                |

**Pattern: 하이퍼파라미터 민감도 분석**

| 논문                                                         | 파라미터                   | 영향                         |
| ------------------------------------------------------------ | -------------------------- | ---------------------------- |
| DPA (2020)                                                   | Weakly-supervised labeling | 0.5% 미만으로 재현 가능      |
| Anomaly Detection using Adversarial Dual Autoencoders (2019) | latent dim                 | MNIST 32, CIFAR 128          |
| DRAEM (2021)                                                 | architecture               | Disc.-only 93.9 vs full 98.0 |
| Deep Anomaly Detection with Deviation Networks (2019)        | labeled anomaly 수         | 5~120 개 변조                |

#### 4.2 조건 변화 실험

**변화 1: contamination 비율 변조**

| 논문                                                  | 오염 범위                | 결과                                                       |
| ----------------------------------------------------- | ------------------------ | ---------------------------------------------------------- |
| DRAEM (2021)                                          | synthetic anomaly 일반성 | synthetic anomaly가 realistic할 필요 없음                  |
| Deep Anomaly Detection with Deviation Networks (2019) | 0~20%                    | contamination 환경에서도 robust                            |
| Adversarially Robust One-class Novelty (2021)         | 5%, 1%, 0.1%             | observed anomaly 에서 우수, unobserved 에도 성능 저하 없음 |

**변화 2: receptive field 크기**

| 논문                              | RF 크기 | AUROC 변화     |
| --------------------------------- | ------- | -------------- |
| Explainable Deep One-Class (2020) | 53→243  | 0.88→0.75 감소 |

#### 4.3 transfer learning 검증

**Pattern: known/unknown task별 성능 차이**

| 논문                                    | 설정                  | 성능 차이                           |
| --------------------------------------- | --------------------- | ----------------------------------- |
| A Survey on Anomaly Detection... (2021) | known vs unknown task | transfer learning 의 성능 차이 분석 |

**Pattern: pre-trained backbone 활용**

| 논문                                 | backbone        | 활용 방법                        |
| ------------------------------------ | --------------- | -------------------------------- |
| ADTR (2022)                          | EfficientNet-B4 | pre-trained feature 복원 대상    |
| Scalable and Interpretable... (2018) | AE 초기화       | Xavier initialization, fine-tune |

### 5. 실험 설계의 한계 및 비교상의 주의점

#### 5.1 비교 조건의 불일치

| 한계                  | 영향                            |
| --------------------- | ------------------------------- |
| 평가 지표 불일치      | AUC, AUROC, AUC-PR, mAUROC 혼용 |
| 데이터셋 불일치       | 각 논문별 고유 데이터셋 사용    |
| baseline 불일치       | 방법 간 직접 비교 어려움        |
| hyperparameter 불일치 | 공정한 비교 제한                |

*A Survey on Anomaly Detection for Technical Systems using LSTM Networks (2021)*

#### 5.2 데이터셋 의존성

| 데이터셋 유형                | 일반화 한계                           |
| ---------------------------- | ------------------------------------- |
| robot-assisted feeding (PR2) | able-bodied adult만 포함, task 한정   |
| SWaT 물 처리                 | 단일 테스트베드, 공격 데이터가 인위적 |
| 의료 이미지                  | 각 의료 도메인별 일반화 어려움        |
| 네트워크 intrusion           | 실제 고장 데이터 아님                 |

#### 5.3 평가 지표의 한계

| 지표      | 한계                                     |
| --------- | ---------------------------------------- |
| ROC AUC   | threshold selection 영향, cost 반영 안됨 |
| AUC-PR    | 불균형 데이터셋 편향, 계산 복잡성        |
| NAB score | anomaly 주변 window 설정 의존적          |
| delay F1  | 지연 시간 정의에 민감                    |

*Adversarially Robust One-class Novelty Detection (2021)*

#### 5.4 계산 비용 분석 부재

| 방법                   | 계산 비용 분석 상태                               |
| ---------------------- | ------------------------------------------------- |
| DNN vs SVM             | 30 분 vs 2 주 학습 (DNN 학습 8 시간 vs SVM 10 분) |
| deep model 대 majority | 계산 비용 비교 부재                               |
| 실시간 적용 가능성     | 대부분 명시적 검증 안됨                           |

### 6. 결과 해석의 경향

#### 6.1 저자들이 결과를 어떻게 해석하는지

**해석 1: "state-of-the-art" 또는 "압도적 성능"**

| 표현            | 사용 사례                          |
| --------------- | ---------------------------------- |
| "압도적 성능"   | GANomaly 대비 ADAE의 CIFAR-10 성능 |
| "SOTA"          | DRAEM의 MVTec-AD 성능              |
| "일관된 우수성" | 다중 데이터셋에서의 성능           |
| "+X% 향상"      | 상대적 성능 차이 강조              |

**해석 2: "task-specific property"**

| 방법       | property                                    |
| ---------- | ------------------------------------------- |
| PrincipaLS | novelty detection 의 task-specific defense  |
| DPA        | 약 0.5% 미만 레이블된 이상 예제로 재현 가능 |
| MemAE      | normal prototype 만 학습한 메모리 구조      |
| DevNet     | data-efficient learning                     |

**해석 3: "메커니즘 증명"**

| 논문   | 메커니즘                                         |
| ------ | ------------------------------------------------ |
| ADTR   | query embedding 이 identical mapping 제한        |
| ADTR   | feature reconstruction 의 semantic discrepancy   |
| DAEM   | reconstruction 과 segmentation 의 joint learning |
| AnoGAN | residual overlay 로 추가 병변 탐지               |

#### 6.2 실제 관찰 결과와의 구분

| 해석                         | 실제 결과                            |
| ---------------------------- | ------------------------------------ |
| "frequency condition 필수적" | Yahoo 등 다양한 dataset 에서 우수    |
| "task-specific defense"      | novelty detection 의 property 활용   |
| "joint learning 효과적"      | end-to-end 신호로 일반화             |
| "sparse addressing 핵심"     | normal prototype 으로 이상 샘플 치환 |

**주의:** 해석은 종종 "기존 방법 대비 우수함"을 강조하면서도, 실제 수치는 다음과 같은 제한사항을 가진다:

- "압도적 성능" → MVTec 98.0 (기존 95.5 대비)
- "일관된 우수성" → 48 개 시나리오, 10 개 이상 데이터셋
- "재현 가능" → 약 0.5% 미만 레이블 사용

### 7. 종합 정리

전체 실험 결과는 다음과 같은 주요 패턴을 제시한다:

**평가 방식:** ROC AUC 와 AUROC 이 주된 지표이며, AUC-PR 은 불균형 데이터셋에서 보조 지표로 사용된다. 계산 비용 분석은 일부 논문 (SVM vs DNN) 제외하고 체계적으로 이루어지지 않는다.

**성향 개선 조건:** (1) feature reconstruction 이 pixel reconstruction 보다, (2) discriminator reconstruction error 가 generator 보다, (3) reconstruction + center 최적화 조합이 각각 더 나은 성능을 보인다. (4) 데이터 규모가 클수록 VAE/AAE 기반 DGM 이 유리하며, 소규모에서는 기존 GAD 방법이 우위다.

**결과 일관성:** MemAE, DASVDD, FCVAE, DPA 와 같이 일관된 성능을 보인 방법은 7 개 이상 데이터셋에서 일관된 성능을 보인다. 반면 OC-NN, GAD는 데이터셋에 따른 편차가 크다.

**조건별 성능:** (1) 데이터셋 규모 (M<5000 소규모→기존 GAD, M>5000 대규모→DGM), (2) 오염 환경 (DevNet/DPLAN 0~20% 까지 robust), (3) 공격 강도 (PGD 대비 black-box 포함), (4) data efficiency (소량 레이블에서 DevNet 75-88% 성능으로 최고)

**유리한 조건:** (1) 이미지 데이터셋에서 reconstruction + segmentation joint learning, (2) 시계열에서 frequency condition 필수, (3) 소량 레이블에서 active learning 및 DQN 강화 학습, (4) 메모리 구조를 활용한 sparse addressing

이 실험 결과들은 이상 탐지 방법론이 데이터셋 규모, 오염 비율, 평가 목표에 따라 다른 설계가 필요함을 보여준다.
