# AI Healthcare

## 서론

### 1. 연구 배경

본 보고서는 의료 인공지능 (AI) 연구의 기술적·응용적 차원을 체계적으로 파악하기 위해 연구 체계 분류, 방법론 분석, 실험 결과 분석에 대한 종합적 문헌 정리를 목표로 한다. 의료 AI 연구는 (1) 오픈소스 프레임워크 및 생성 모델 표준화 같은 생태계 인프라 구축, (2) 연합학습과 분산 학습을 통한 협업·프라이버시 보존, (3) 생성적 데이터 증강·시계열 예측·멀티모달 융합 같은 데이터 중심 접근, (4) 해석 가능성·보안·지식 기반 학습 같은 신뢰성 강화, (5) 의료 영상·자연어·강화학습 등 응용 도메인을 포괄하는 다층적 영역으로 확장되어 왔으며, 각 영역 내에서 방법론 (GAN, Transformer, FL 등) 과 의료 데이터 특성 (시계열, 멀티모달, 희소성) 이 결합된 복합적 설계 전략이 강조된다.

### 2. 문제의식 및 분석 필요성

의료 데이터의 이질적 형식 (임상 텍스트, 의료 영상, 시계열, omics 등) 과 특수성 (프라이버시 제약, 데이터 편차, 라벨 희소성, 임상적 해석 가능성 요구) 은 단일 모델 중심 접근법의 한계를 드러냈다. 따라서 의료 AI 를 **학습 패러다임** (비지도·자기지도·supervised·foundation model) 과**구조적 접근** (중앙집중식·분산·멀티모달 융합) 으로 구분하여 체계화할 필요가 있다. 또한 일반-purpose foundation model 의 의료 도메인 적용에서 발생할 수 있는 구조적 한계 (온톨로지 의존성, temporal sequence 처리, 희소 질환 대응) 를 명시하고, 데이터 부족 문제 해결을 위한 필수 전략 (FL, transfer learning, few-shot learning) 을 규명해야 한다.

### 3. 보고서의 분석 관점

본 보고서는 문헌을 다음 세 관점에서 분석한다.

- **연구체계 분류**: 20 편의 논문을 기술적 접근 방식 (생성·강화·분산 학습), 데이터·시스템 구조 (시계열/멀티모달/분산/희소), 응용 목적 (진단·데이터 프라이버시·시스템 효율·해석 가능성) 기준 20 개 분류로 구분.
- **방법론 분석**: 16 계열 (Foundation Model, Federated Learning, Multimodal Fusion, XAI, Graph/Ontology 등) 과 9 가지 설계 패턴 (Pre-training→Adaptation, Representation→Prediction, Multi-Modal Fusion, Federated 등) 으로 재구성.
- **실험결과 분석**: 의료 영상 생성 성능 (FID 0.005~8.83), EHR 예측 모델 (AUC +8~15%), FL 통신 효율성 (traffic 99.8% 절감), hardware EDP 비교 등 주요 실험 결과를 정렬하여 성능 패턴과 상충 관계 (privacy↔accuracy, communication↔performance) 도 제시.

### 4. 보고서 구성

- **1 장 (연구체계 분류)**: 20 개의 연구 분류로 의료 AI 연구 지형도 정립. 오픈소스 프레임워크 (MONAI), 연합학습 (FL/HFL/VFL), 생성 모델 (GAN/Diffusion), 시계열 예측, 멀티모달 융합 (DIKW), 해석 가능성 (XAI), 지식 그래프, 보안 강화 등 범주별 대표 논문과 분류 근거 제시.
- **2 장 (방법론 분석)**: 16 계열 방법론과 9 가지 설계 패턴을 통해 의료 AI 연구의 방법론적 지형도 분석. Foundation Model 기반 파이프라인, 분산/협력 학습, 멀티모달 융합, XAI, 하드웨어 적응 등 계열별 공통 구조와 핵심 특징 정리.
- **3 장 (실험 결과 분석)**: 주요 데이터셋 (MIMIC, CheXpert, CPRD 등) 기반 평가 결과를 정렬하여 의료 영상 생성 성능, EHR 예측 모델 성능, FL 통신 효율성, hardware 플랫폼 비교 등 주요 실험 결과와 성능 패턴, 상충 관계, 일반화 한계 제시.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 보고서의 연구 분류는 의료 인공지능 연구의 기술적·응용적 차원을 체계적으로 파악하기 위해 다음과 같은 원칙에 따라 수립되었다.

첫째, **기술적 접근 방식**으로 분류한다. 각 논문이 어떤 기계학습 패러다임(예: 생성 모델, 강화 학습, 분산 학습 등)을 기반으로 하는지를 주요 분류 기준으로 삼는다.

둘째, **데이터·시스템 구조**를 고려한다. 의료 데이터의 특수성(시계열, 멀티모달, 분산, 희소성)과 이에 따른 시스템 설계 전략(예: EHR 내부 구조 활용, 멀티모달 융합, 생성적 데이터 증강)이 적용되는 범주를 구분한다.

셋째, **응용 목적**에 따른 축을 고려한다. 진단·예측, 데이터 프라이버시 보호, 시스템 효율성 향상, 해석 가능성 등 최종 목표로서의 차원이 다른 범주 설정이 가능하다.

넷째, **분류 중복 최소화**를 위해 각 논문의 가장 대표적인 1 개 범주에 배치한다. 연구가 여러 범주에 동시에 속할 수 있으나 본 분류 체계에서는 분석의 일관성을 위해 한 가지 범주로 통합한다.

### 2. 연구 분류 체계

#### 2.1 의료 AI 프레임워크 및 오픈소스 생태계

의료 AI 연구의 인프라·플랫폼 수준을 연구하는 논문으로, 의료 영상/데이터 특화 소프트웨어 개발 환경, 표준 인터페이스, 재현성·확장성 지원 인프라에 집중한다.

| 분류                                        | 논문명                                                                             | 분류 근거                                                                                                       |
| ------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 프레임워크 개발 > 오픈소스 의료 AI 플랫폼   | MONAI: An open-source framework for deep learning in healthcare (2022)             | 의료 영상 특화 transform, metadata-aware tensor, deterministic caching 을 일관된 PyTorch 호환 프레임워크로 제공 |
| 프레임워크 개발 > 의료영상 생성 모델 표준화 | Generative AI for Medical Imaging: extending the MONAI Framework (2023)            | 확산되는 생성 모델을 공통 인터페이스와 모듈형 구성으로 통합한 표준화된 프레임워크 구축                          |
| 프레임워크 개발 > 의료 영상 딥러닝 개요     | Deep Learning for Medical Image Processing: Overview, Challenges and Future (2017) | 의료영상 딥러닝의 기술적 아키텍처와 의료 적용 장벽을 함께 다루며 CNN이 핵심 모델임을 명시하는 survey 형 개관    |

#### 2.2 분산·연합 학습 및 협업 학습

의료 기관 간 데이터 공유 제약, 개인정보 보호 규정, 분산 데이터 환경에서 협업 학습 및 프라이버시 보호를 위한 설계·알고리즘을 연구한다.

| 분류                                  | 논문명                                                                                                 | 분류 근거                                                                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 연합 학습 > IoMT 프라이버시 보전      | Federated Learning for Privacy Preservation in Smart Healthcare Systems: A Comprehensive Survey (2022) | IoMT 환경에서 FL 기반 privacy preservation 을 위한 체계적 survey 로, DP, encryption, blockchain, DT 등 보강 설계 공간과 threat/model 분류를 포함                                 |
| 연합 학습 > 헬스케어 데이터 분산 구조 | Federated Learning for Smart Healthcare: A Survey (2021)                                               | 헬스케어 데이터 분산 형태 (sample/feature space) 에 따른 FL 유형 (HFL/VFL/FTL) 과 advanced design 차원 (resource/secure/incentive/personalized) 을 체계화한 헬스케어 중심 survey |
| 연합 학습 > IoT 의료 기기 경량화      | A Federated Learning Framework for Healthcare IoT devices (2020)                                       | 자원 제약된 의료 IoT 환경에서 DNN 분할 (computation offload) 과 sparse 통신 (traffic reduction) 을 동시에 최적화한 공학 중심 FL 프레임워크                                       |
| 분산 학습 > 의료영상 협업 학습        | Split Learning for collaborative deep learning in healthcare (2019)                                    | raw data sharing 및 label sharing 없이 collaborative training 이 가능한 split learning 구조를 DR 과 CheXpert 에서 입증                                                           |

#### 2.3 생성 모델 및 데이터 증강

의료 데이터 부족, 불균형, 프라이버시 문제 해결을 위해 생성적 모델을 활용한 데이터 증강·synthetic 생성·모델 공격 연구를 포함한다.

| 분류                                       | 논문명                                                                                                                               | 분류 근거                                                                                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 생성 모델 > EHR synthetic 생성             | A review of Generative Adversarial Networks for Electronic Health Records: applications, evaluation measures and data sources (2022) | 이 논문을 structured EHR synthetic data 연구의 응용·평가·프라이버시 삼각 구도 제시 및 evaluation taxonomy 확립을 위한 methodological checklist 로 분류 |
| 생성 모델 > GAN 기반 데이터 증강           | Boosting Deep Learning Risk Prediction with Generative Adversarial Networks for Electronic Health Records (2017)                     | 라벨이 제한된 시계열 의료 데이터에서 생성 모델을 직접적인 예측 성능 향상 도구로 활용한 low-resource setting 연구                                       |
| 생성 모델 > 자연어 생성                    | Natural Language Generation for Electronic Health Records (2018)                                                                     | 구조화 EHR 이산 변수를 입력으로 받아 자연어 chief complaint 를 생성하는 encoder-decoder 접근 제시 및 생성 결과의 역학적 유효성 실증                    |
| 생성 모델 > 생성적 모델 기반 의료영상 생성 | Generative AI for Medical Imaging: extending the MONAI Framework (2023)                                                              | 확산되는 생성 모델을 공통 인터페이스와 모듈형 구성으로 통합한 표준화된 프레임워크 구축                                                                 |
| 생성 모델 > 의료영상 생성 모델 표준화      | Generative AI for Medical Imaging: extending the MONAI Framework (2023)                                                              | 확산되는 생성 모델을 공통 인터페이스와 모듈형 구성으로 통합한 표준화된 프레임워크 구축                                                                 |
| 생성 모델 > 의료영상 조작 공격             | Jekyll: Attacking Medical Image Diagnostics using Deep Generative Models (2021)                                                      | patient identity 유지하면서 disease condition 만 교체하는 controlled image-to-image translation 을 통한 의료영상 자체 조작 공격 구현                   |

#### 2.4 시계열 분석 및 예측

의료 기록의 시간적 특성 (irregular timing, sequential dependence) 을 활용한 예측·이상 탐지·치료 전략 연구를 포함한다.

| 분류                                     | 논문명                                                                                           | 분류 근거                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 시계열 예측 > EHR 시계열 표현학습        | DeepCare: A Deep Dynamic Memory Model for Predictive Medicine (2016)                             | 의료 기록을 단순 시계열이 아닌, 시간 간격과 개입이 미래 상태를 바꾸는 동적 메모리 문제로 정식화하고 이를 end-to-end 신경망으로 구현                                      |
| 시계열 예측 > EHR 표현학습               | MiME: Multilevel Medical Embedding of Electronic Health Records for Predictive Healthcare (2018) | EHR 내부의 다층 구조 (방문-진단-치료) 를 representation 학습과 보조 예측 과제에 동시 활용하여 외부 지식 의존 없이 데이터 부족 환경에서 견고한 성능 달성                  |
| 시계열 예측 > 시계열 시계열 예측         | Time Series Prediction Using Deep Learning Methods in Healthcare (2021)                          | 의료 구조화 시계열 예측 문헌을 10 개 기술 축으로 재분류하며, 각 축에서 현재 best practice 와 연구 공백을 체계적으로 정리한 meta 연구                                     |
| 시계열 이상탐지 > 탐지 원리 중심 분류    | Deep Learning for Time Series Anomaly Detection: A Survey (2022)                                 | 시계열 이상탐지 문제를 탐지 원리 (예측/재구성/하이브리드) 와 anomaly 유형 (temporal/intermetric/univariate/multivariate) 의 구조적 특성을 함께 고려하는 통합 survey 체계 |
| 시계열 예측 > 시계열 EHR 예측 프레임워크 | BEHRT: Transformer for Electronic Health Records (2019)                                          | EHR 시퀀스를 문서처럼 모델링한 Transformer 기반 사전학습 모델로, 의료 특화 temporal embedding (age/segment) 결합 및 대규모 multi-label 질환 예측 실험                    |

#### 2.5 멀티모달 융합 및 정보 통합

이질적 형식과 의미를 가진 다중 모달리티 의료 데이터를 통합하고 의미 축적을 위한 융합 기법 및 프레임워크 연구 포함한다.

| 분류                                  | 논문명                                                                                                         | 분류 근거                                                                                                                                                                 |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 멀티모달 융합 > DIKW 융합 프레임워크  | A Survey of Multimodal Information Fusion for Smart Healthcare: Mapping the Journey from Data to Wisdom (2023) | 이 논문은 멀티모달 의료 데이터 융합 기법을 단순 목록이 아닌 DIKW 계층을 따른 의미 축적 과정으로 재구성한 서베이로, 분류 시 "의미 생성 과정 관점의 융합 설계" 범주에 적합  |
| 멀티모달 융합 > 감정 인식 세 모달리티 | Emotion Recognition for Healthcare Surveillance Systems Using Neural Networks: A Survey (2021)                 | 의료 감시 맥락에서 세 모달리티와 세 단계 파이프라인으로 체계화된 survey 논문이며, 심리상태 조기 탐지와 elder care 에 적용 가능한 감정 인식 기술의 기술적·실천적 토대 제공 |
| 멀티모달 융합 > 멀티모달 의료 영상    | A survey on attention mechanisms for medical applications: are we moving towards better algorithms? (2022)     | 이 논문은 attention mechanism 에 대한 낙관적 서사를 문헌 정리와 직접 실험을 통해 검증하는 비판적 survey 로, 의료 AI 에서의 성능·복잡도·해석가능성 관계를 규명한 연구      |

#### 2.6 해석 가능성 및 투명성 (Explainable AI)

딥러닝의 블랙박스 성질을 극복하고 의료 현장에서clinician 의 신뢰 획득, 규제 대응을 위한 설명 가능성 기법 및 평가 체계를 연구한다.

| 분류                                    | 논문명                                                                                                     | 분류 근거                                                                                                                                                                                            |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 해석 가능성 > Attribution method 체계화 | Explainable Deep Learning in Healthcare: A Methodological Survey from an Attribution View (2021)           | 본 논문은 의료 헬스케어 DL 모델의 해석 가능성 방법을 attribution 관점에서 체계적 분류 체계와 방법론적 비교 가이드로 제공하며, 다양한 XAI 기법의 장단점과 적용 시나리오를 구조화한 방법론 중심 survey |
| 해석 가능성 > 주의력 메커니즘 비판      | A survey on attention mechanisms for medical applications: are we moving towards better algorithms? (2022) | 이 논문은 attention mechanism 에 대한 낙관적 서사를 문헌 정리와 직접 실험을 통해 검증하는 비판적 survey 로, 의료 AI 에서의 성능·복잡도·해석가능성 관계를 규명한 연구                                 |

#### 2.7 지식 그래프·온톨로지 통합

의료 도메인 지식 (진단 코드 계층, 지식 그래프, 온톨로지) 을 표현 학습이나 예측 모델에 통합하여 희소성, 해석 가능성, 일반화 성능을 향상시키는 연구 포함한다.

| 분류                                    | 논문명                                                                          | 분류 근거                                                                                                                                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 지식 그래프 > 의료 온톨로지 표현학습    | GRAM: Graph-based Attention Model for Healthcare Representation Learning (2016) | 의료 지식 그래프를 representation layer 에 내장한 적응형 smoothing 방식을 통해 희소 데이터 환경에서 ontology-aware 예측 가능                                                                  |
| 지식 그래프 > 의료 코딩 통합 프레임워크 | A Unified Review of Deep Learning for Automated Medical Coding (2022)           | 이 논문은 의료 코딩 모델을 encoder, deep connections, decoder, auxiliary information 의 네 축으로 분해하는 통합 프레임워크를 제시하며, 구조적 설계 선택에 따른 모델 비교를 체계화한 리뷰 연구 |
| 지식 그래프 > 희소 질환 표현보강        | GRAM: Graph-based Attention Model for Healthcare Representation Learning (2016) | 의료 지식 그래프를 representation layer 에 내장한 적응형 smoothing 방식을 통해 희소 데이터 환경에서 ontology-aware 예측 가능                                                                  |

#### 2.8 보안을 강화한 의료 AI

의료 AI 시스템의 취약성 분석, adversarial attack 방어, privacy-preserving 기술, IoMT 보안 아키텍처를 연구한다.

| 분류                                | 논문명                                                                            | 분류 근거                                                                                                                                                                                  |
| ----------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 보안 > H-IoT 계층 보안 체계         | Machine Learning for Healthcare-IoT Security: A Review and Risk Mitigation (2024) | 四层 계층 구조와 26 가지 공격 유형을 매핑하고 ML 기반 anomaly/intrusion detection, authentication/access control, routing attack mitigation 등을 완화 전략으로 연결한 H-IoT 보안 종합 리뷰 |
| 보안 > 의료 ML 파이프라인 취약점    | Secure and Robust Machine Learning for Healthcare Applications: A Survey (2020)   | 본 논문은 의료 ML 파이프라인의 각 단계별 취약점을 구조화하고, security threat taxonomy 와 secure ML 솔루션 분류 체계를 제시하는 종합 survey                                                |
| 보안 > 생성 모델 기반 의료영상 공격 | Jekyll: Attacking Medical Image Diagnostics using Deep Generative Models (2021)   | patient identity 유지하면서 disease condition 만 교체하는 controlled image-to-image translation 을 통한 의료영상 자체 조작 공격 구현                                                       |

#### 2.9 의료 영상 분석

의료 영상 (X-ray, MRI, CT, ultrasound, histology 등) 의 표현학습, 분류, 분할, 등록, 생성 등 다양한 태스크에 딥러닝 기법을 적용하는 연구 포함한다.

| 분류                                   | 논문명                                                                                                     | 분류 근거                                                                                                                                                                                                      |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 의료영상 > 의료영상 처리 개요          | Deep Learning for Medical Image Processing: Overview, Challenges and Future (2017)                         | 의료영상 딥러닝의 기술적 아키텍처와 의료 적용 장벽을 함께 다루며 CNN이 핵심 모델임을 명시하는 survey 형 개관                                                                                                   |
| 의료영상 > 비지도 딥러닝 의료영상 분석 | A Tour of Unsupervised Deep Learning for Medical Image Analysis (2018)                                     | 이 논문은 새로운 방법론 제안이 아닌 비지도 딥러닝 문헌의 체계적 정리와 model family 중심 분류체계를 제공하며, 의료영상 분석에서 unsupervised learning 의 초기 탐색과 응용 지형도를 보여주는 survey paper       |
| 의료영상 > 심전도 부정맥 탐지          | A Federated Learning Framework for Healthcare IoT devices (2020)                                           | 자원 제약된 의료 IoT 환경에서 DNN 분할 (computation offload) 과 sparse 통신 (traffic reduction) 을 동시에 최적화한 공학 중심 FL 프레임워크                                                                     |
| 의료영상 > 주의력 메커니즘 의료영상    | A survey on attention mechanisms for medical applications: are we moving towards better algorithms? (2022) | 이 논문은 attention mechanism 에 대한 낙관적 서사를 문헌 정리와 직접 실험을 통해 검증하는 비판적 survey 로, 의료 AI 에서의 성능·복잡도·해석가능성 관계를 규명한 연구                                           |
| 의료영상 > 강화학습 의료영상 응용      | Deep reinforcement learning in medical imaging: A literature review (2021)                                 | 의료 영상 문제를 세 가지 유형 (parametric image analysis, optimization, miscellaneous application) 으로 분류하고, 각 유형 내에서의 DRL 적용 패턴과 model-free 알고리즘의 지배적 사용 여부를 근거로 연구 체계화 |
| 의료영상 > 생성 모델 의료영상 생성     | Generative AI for Medical Imaging: extending the MONAI Framework (2021)                                    | 확산되는 생성 모델을 공통 인터페이스와 모듈형 구성으로 통합한 표준화된 프레임워크 구축                                                                                                                         |

#### 2.10 자연어 처리 및 텍스트 분석

의료 텍스트 (임상노트, EHR 구조화 필드, radiology report 등) 를 이해·생성·코딩하는 NLP 기반 기술과 encoder-decoder 프레임워크를 연구한다.

| 분류                            | 논문명                                                                                            | 분류 근거                                                                                                                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NLP > 의료 텍스트 통합 리뷰     | Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions (2024) | 의료 AI 분야를 언어, 영상, 생명정보학, 멀티모달 네 관점에서 통합 분류하는 종합 survey 로, pre-training 과 adaptation 전략, 데이터 자원, 병목 요소를 체계적으로 정리                           |
| NLP > EHR 자연어 생성           | Natural Language Generation for Electronic Health Records (2018)                                  | 구조화 EHR 이산 변수를 입력으로 받아 자연어 chief complaint 를 생성하는 encoder-decoder 접근 제시 및 생성 결과의 역학적 유효성 실증                                                           |
| NLP > 의료 코딩 통합 프레임워크 | A Unified Review of Deep Learning for Automated Medical Coding (2022)                             | 이 논문은 의료 코딩 모델을 encoder, deep connections, decoder, auxiliary information 의 네 축으로 분해하는 통합 프레임워크를 제시하며, 구조적 설계 선택에 따른 모델 비교를 체계화한 리뷰 연구 |
| NLP > EHR 시퀀스 모델링         | BEHRT: Transformer for Electronic Health Records (2019)                                           | EHR 시퀀스를 문서처럼 모델링한 Transformer 기반 사전학습 모델로, 의료 특화 temporal embedding (age/segment) 결합 및 대규모 multi-label 질환 예측 실험                                         |

#### 2.11 강화학습 및 순차적 의사결정

의료 환경에서 순차적 치료를 설계하거나 치료 정책을 최적화하기 위한 강화학습·오프라인 RL 접근법을 연구한다.

| 분류                             | 논문명                                                                                                      | 분류 근거                                                                                                                                                                            |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 강화학습 > I-Health 시스템 RL    | Reinforcement Learning for Intelligent Healthcare Systems: A Comprehensive Survey (2021)                    | 헬스케어 알고리즘보다 시스템 전체 인프라 (sensing–networking–edge–clinical) 에서 RL 이 배치 가능한 위치를 지도하는 survey                                                            |
| 강화학습 > 오프라인 RL 모델 선택 | Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings (2021) | OPE 를 validation proxy 로 사용하여 offline RL 의 model selection pipeline 을 구성하고, 네 가지 OPE 방법의 ranking quality 와 computational cost trade-off 를 체계적으로 분석한 연구 |

#### 2.12 확률적 머신러닝 및 불확실성

의료 데이터의 불확실성, 누락, 검열, 분포 변화 등을 명시적으로 다루는 확률적 프레임워크 및 모델링 기법을 연구한다.

| 분류                             | 논문명                                               | 분류 근거                                                                                                                                                      |
| -------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 확률적 ML > 의료 데이터 불확실성 | Probabilistic Machine Learning for Healthcare (2021) | 예측 정확도 중심의 deterministic 접근이 아닌, 의료 데이터의 분포와 불확실성을 명시적으로 다루는 probabilistic framework 를 의료 AI 설계에 적용하는 survey 논문 |

#### 2.13 비지도·자기지도 학습

의료 데이터 라벨 부족 문제를 극복하고 표현 학습·자기지도 학습·GAN 등을 활용한 비지도 학습 프레임워크를 연구한다.

| 분류                                      | 논문명                                                                                           | 분류 근거                                                                                                                                                                                                |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 비지도 학습 > 비지도 딥러닝 의료영상      | A Tour of Unsupervised Deep Learning for Medical Image Analysis (2018)                           | 이 논문은 새로운 방법론 제안이 아닌 비지도 딥러닝 문헌의 체계적 정리와 model family 중심 분류체계를 제공하며, 의료영상 분석에서 unsupervised learning 의 초기 탐색과 응용 지형도를 보여주는 survey paper |
| 비지도 학습 > EHR 내부 구조 자기지도 학습 | MiME: Multilevel Medical Embedding of Electronic Health Records for Predictive Healthcare (2018) | EHR 내부의 다층 구조 (방문-진단-치료) 를 representation 학습과 보조 예측 과제에 동시 활용하여 외부 지식 의존 없이 데이터 부족 환경에서 견고한 성능 달성                                                  |
| 비지도 학습 > 시계열 이상 탐지            | Deep Learning for Time Series Anomaly Detection: A Survey (2022)                                 | 시계열 이상탐지 문제를 탐지 원리 (예측/재구성/하이브리드) 와 anomaly 유형 (temporal/intermetric/univariate/multivariate) 의 구조적 특성을 함께 고려하는 통합 survey 체계                                 |

#### 2.14 앙상블·메타 학습

단일 모델의 성능 한계를 극복하기 위한 앙상블·stacked generalization, basic classifier+meta-learner 조합을 의료 분류 문제에 적용한다.

| 분류                            | 논문명                                                                                                                  | 분류 근거                                                                                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 앙상블 > Stacking Meta-learning | Deep Neural Network Based Ensemble learning Algorithms for the healthcare system (diagnosis of chronic diseases) (2021) | 의료 tabular classification 도메인에서 basic ML algorithms 보다 ensemble stacking+neural network meta-learner 조합이 높은 진단 성능과 강건성을 보이는 실증적 연구 |

#### 2.15 지식 증류 및 해석 가능한 모델

딥러닝의 예측력을 유지하면서 GBT 기반 규칙성 모델로 surrogate modeling 하며 interpretable phenotype 추출 및 knowledge distillation 기법을 연구한다.

| 분류                         | 논문명                                                                                | 분류 근거                                                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 지식 증류 > Interpretable ML | Distilling Knowledge from Deep Networks with Applications to Healthcare Domain (2015) | 의료 도메인에서 deep model 의 성능을 GBT 기반 surrogate 모델로 재표현하며, post-hoc explanation 이 아닌 proxy modeling 방식으로 interpretable phenotype 을 추출 |

#### 2.16 의료 IoT 및 하드웨어 가속화

의료 IoT 디바이스·웨어러블의 연산·전력·통신 효율성 및 edge 하드웨어 가속화 기술을 연구한다.

| 분류                                       | 논문명                                                                                                     | 분류 근거                                                                                                                                              |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IoT/하드웨어 > IoT 연산/통신 최적화        | A Federated Learning Framework for Healthcare IoT devices (2020)                                           | 자원 제약된 의료 IoT 환경에서 DNN 분할 (computation offload) 과 sparse 통신 (traffic reduction) 을 동시에 최적화한 공학 중심 FL 프레임워크             |
| IoT/하드웨어 > 의료용 딥러닝 하드웨어 선택 | Hardware Implementation of Deep Network Accelerators Towards Healthcare and Biomedical Applications (2021) | 이 논문은 의료용 딥러닝을 edge 로 구현할 때의 하드웨어 선택지를 CMOS/FPGA/memristive/SNN 네 가지 관점에서 체계적으로 비교 분석한 tutorial/review paper |

#### 2.17 기초 모델 (Foundation Model) 통합

대규모 의료 데이터와 멀티모달 데이터를 pre-training 한 foundation model 과 그 adaptation 전략을 체계화하는 종합 survey 포함한다.

| 분류                             | 논문명                                                                                            | 분류 근거                                                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 기초 모델 > 의료 HFM 통합 survey | Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions (2024) | 의료 AI 분야를 언어, 영상, 생명정보학, 멀티모달 네 관점에서 통합 분류하는 종합 survey 로, pre-training 과 adaptation 전략, 데이터 자원, 병목 요소를 체계적으로 정리 |
| 기초 모델 > 의료 FM survey       | A Comprehensive Survey of Foundation Models in Medicine (2024)                                    | 의료 AI 의 통합 관점에서 foundation models 의 다양한 domain 적용과 학습 전략 (SSL, transfer learning) 을 체계적으로 분류 정리                                       |

#### 2.18 시스템·응용 중심 통합 연구

특정 응용 영역 (감정 인식, EEG-BCI, 만성질환 분류) 을 중심으로 시스템 전체의 hardware→algorithm→application 통합 관점에서 정리한 종합 survey 포함한다.

| 분류                                  | 논문명                                                                                                                                                                        | 분류 근거                                                                                                                                                                                    |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 시스템/응용 > 의료 감시 감정 인식     | Emotion Recognition for Healthcare Surveillance Systems Using Neural Networks: A Survey (2021)                                                                                | 의료 감시 맥락에서 세 모달리티와 세 단계 파이프라인으로 체계화된 survey 논문이며, 심리상태 조기 탐지와 elder care 에 적용 가능한 감정 인식 기술의 기술적·실천적 토대 제공                    |
| 시스템/응용 > EEG-BCI 통합 프레임워크 | EEG-based Brain-Computer Interfaces (BCIs): A Survey of Recent Studies on Signal Sensing Technologies and Computational Intelligence Approaches and Their Applications (2020) | 이 논문은 EEG-BCI 연구가 hardware usability, artefact handling, domain adaptation, interpretability, representation learning 등 다층적 요소의 균형 발전 위에서 진행된다는 통합적 관점을 제공 |
| 시스템/응용 > 만성질환 ensemble 분류  | Deep Neural Network Based Ensemble learning Algorithms for the healthcare system (diagnosis of chronic diseases) (2021)                                                       | 의료 tabular classification 도메인에서 basic ML algorithms 보다 ensemble stacking+neural network meta-learner 조합이 높은 진단 성능과 강건성을 보이는 실증적 연구                            |

#### 2.19 의료 딥러닝 종합 서베이

EHR 분석·딥러닝 아키텍처·응용 과제 전반을 응용과 기술 축 쌍중심으로 정리한 종합 서베이 포함한다.

| 분류                            | 논문명                                                                                                               | 분류 근거                                                                                                                                                                                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 종합 서베이 > EHR 딥러닝 서베이 | Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis (2018) | 이 논문은 EHR 딥러닝 연구를 "임상응용 (5 가지 영역) × 기술적 접근 (5 가지 아키텍처)" 이란 쌍중심 분류체계로 구조화했으며, representation learning 을 중심축으로 표현학습 기반 예측을 강조하는 점이 후속 분류에서 의료 딥러닝 연구 위치 판단의 기준이 됨 |

#### 2.20 구조화된 문헌 고찰 및 Meta 연구

PRISMA 지침에 따른 체계적 리뷰·meta 분석·systematic review 방법을 적용한 연구 포함한다.

| 분류             | 논문명                                                                                    | 분류 근거                                                                                                                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 체계적 리뷰      | Time Series Prediction Using Deep Learning Methods in Healthcare (2021)                   | 의료 구조화 시계열 예측 문헌을 10 개 기술 축으로 재분류하며, 각 축에서 현재 best practice 와 연구 공백을 체계적으로 정리한 meta 연구                                                                               |
| 체계적 문헌 고찰 | Reliable and Resilient AI and IoT-based Personalised Healthcare Services: A Survey (2022) | 다중 건강 조건 간의 상호의존성을 고려한 CPHS 개념을 제안하며 시스템 아키텍처와 요구사항 엔지니어링에 초점을 맞춘 체계적 survey 논문으로, 의료 IoT 시스템 설계를 위한 신뢰성·복원력·개인화의 통합적 프레임워크 제시 |

### 3. 종합 정리

위 분류 체계를 바탕으로 분석한 의료 AI 연구 지형은 크게 (1) 의료 AI 생태계 인프라 (오픈소스 프레임워크, 생성 모델 표준화, 기술 개요), (2) 분산 및 협업 학습 (연합학습 프라이버시·IoT 연산 효율화, 의료영상 분산 협업), (3) 데이터 중심 접근 (생성적 증강, 시계열 표현·예측·이상탐지, 멀티모달 융합), (4) 모델 신뢰성 (해석 가능성, 보안 강화, 지식 기반 학습), (5) 응용 도메인 (의료영상, 자연어, 강화학습·확률적 ML, 의료 IoT) 의 5 가지 축으로 통합되는데, 각 축 내부에서는 방법론 (GAN, Transformer, FL, attention 등) 과 의료 데이터 특성 (시계열, 멀티모달, 희소성) 이 결합된 복합적 설계 전략이 강조된다.

## 2장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

| 차원          | 공통 구조                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| **데이터**    | 이질적 의료 데이터: 임상 텍스트 (EHR), 의료 영상 (X-ray/CT/MRI), 시계열 (vital signs), 시계열, omics, EEG 등 |
| **목표**      | 질병 예측, 진단 보조, 의사결정 지원, 위험 평가, synthetic data 생성, representation learning                 |
| **패러다임**  | Pre-training → Adaptation / Unsupervised Representation → Supervised Fine-tuning / Centralized → Federated   |
| **핵심 제약** | 프라이버시 (PII), 데이터 편차, label scarcity, clinical interpretability, real-time inference                |
| **입력→출력** | Raw clinical data → Latent representation → Downstream task prediction                                       |

## 2. 방법론 계열 분류

## (1) Foundation Model 기반 파이프라인

**계열 정의**: 대규모 데이터를 자기지도/비지도 방식으로 사전 학습한 후, down-stream medical task 에 fine-tuning 또는 adaptation 을 적용하는 구조.

**공통 특징**:

- Pre-training 단계: GL/CL/HL/SL 등 다양한 objective
- Adaptation 단계: FT/AT/PE (Fine-tuning/Adapter Tuning/Prompt Engineering)
- Encoder-only/Decoder-only/Dual-encoder architecture

| 방법론 계열      | 논문명                                                                                                                               | 핵심 특징                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Foundation Model | Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions (2024)                                    | 4 가지 하위 분야 (language/vision/bio/multimodal), 2 단계 프레임워크 (pre-training/adaptation), instruction tuning |
| Foundation Model | A Comprehensive Survey of Foundation Models in Medicine (2024)                                                                       | Transformer/attention/SSL/contrastive learning/RL, transfer learning 패러다임                                      |
| Foundation Model | A review of Generative Adversarial Networks for Electronic Health Records: applications, evaluation measures and data sources (2022) | GAN 기반 생성 (medGAN/medWGAN/RCGAN), adversarial objective, downstream task 학습                                  |
| Foundation Model | Boosting Deep Learning Risk Prediction with Generative Adversarial Networks for Electronic Health Records (2017)                     | ehrGAN 기반 semi-supervised data augmentation, SSL-GAN                                                             |
| Foundation Model | DeepCare: A Deep Dynamic Memory Model for Predictive Medicine (2016)                                                                 | modified LSTM, time-decayed multiscale pooling, dynamic memory                                                     |
| Foundation Model | Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis (2018)                 | MLP/CNN/RNN/AE/RBM 등 다양한 representation learning                                                               |
| Foundation Model | BEHRT: Transformer for Electronic Health Records (2019)                                                                              | MLM pre-training, temporal embedding (age/segment/position)                                                        |

## (2) Federated/Decentralized Learning 계열

**계열 정의**: 중앙 집중형 학습의 프라이버시·확장성 문제를 해결하기 위해 raw data 공유 없이 기관/장치 간 협력 학습 구조.

**공통 특징**:

- FedAvg 기반 aggregation
- HFL/VFL/FTL 세 가지 분산 패러다임
- DP/HE/blockchain/security mechanism 통합

| 방법론 계열        | 논문명                                                                                                                                                                        | 핵심 특징                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Federated Learning | A Federated Learning Framework for Healthcare IoT devices (2020)                                                                                                              | network decomposition, sparse activation/gradient 통신                   |
| Federated Learning | Federated Learning for Smart Healthcare: A Survey (2021)                                                                                                                      | HFL/VFL/FTL 분류, resource-aware/security-aware/personalized FL          |
| Federated Learning | Federated Learning for Privacy Preservation in Smart Healthcare Systems: A Comprehensive Survey (2022)                                                                        | DP/HE/blockchain/DRL/GAN 등 확장 설계                                    |
| Federated Learning | Split Learning for collaborative deep learning in healthcare (2019)                                                                                                           | U-shaped split config, raw data/sharing 없이 feature-level collaboration |
| Federated Learning | EEG-based Brain-Computer Interfaces (BCIs): A Survey of Recent Studies on Signal Sensing Technologies and Computational Intelligence Approaches and Their Applications (2020) | domain adaptation (subject-to-subject/session-to-session)                |

## (3) Multimodal Fusion 계열

**계열 정의**: 이질적 의료 데이터 (EHR/imaging/genomic/environmental/behavioral) 를 융합하여 임상적 의사결정 파이프라인을 구성하는 구조.

**공통 특징**:

- Data→Information→Knowledge→Wisdom (DIKW) 계층
- Early/Late/Hybrid fusion architecture
- Attention/transfer learning/cross-modal alignment

| 방법론 계열       | 논문명                                                                                                         | 핵심 특징                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Multimodal Fusion | A Survey of Multimodal Information Fusion for Smart Healthcare: Mapping the Journey from Data to Wisdom (2023) | DIKW 프레임워크, 5 단계 융합 (Data/Info/Know/Wisdom), attention/transfer learning |
| Multimodal Fusion | Emotion Recognition for Healthcare Surveillance Systems Using Neural Networks: A Survey (2021)                 | SER/FER/AVR 세 모달리티, late fusion                                              |

## (4) Attention/Mechanism Analysis 계열

**계열 정의**: attention mechanism 의 구조적 역할과 해석 가능성, 성능 영향을 체계적으로 분석하는 방법론.

**공통 특징**:

- Multi-level/single-level/soft/hard/local 분류
- Post-hoc explanation 및 시각 분석
- Survey + case study 결합

| 방법론 계열        | 논문명                                                                                                     | 핵심 특징                                                                         |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Attention Analysis | A survey on attention mechanisms for medical applications: are we moving towards better algorithms? (2022) | 6 가지 taxonomy (channel/spatial/temporal/branch), post-hoc explanation 시각 분석 |

## (5) Unsupervised Representation Learning 계열

**계열 정의**: 레이블 불일용 데이터에서 latent representation/feature extraction 을 수행하는 unsupervised 모델 계열.

**공통 특징**:

- Reconstruction/density estimation/dimensionality reduction
- AE/RBM/DBN/DBM/GAN 등 생성 모델

| 방법론 계열           | 논문명                                                                             | 핵심 특징                                                                    |
| --------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Unsupervised Learning | A Tour of Unsupervised Deep Learning for Medical Image Analysis (2018)             | AE/RBM/DBN/DBM/GAN taxonomy, reconstruction loss, VLB                        |
| Unsupervised Learning | Deep Learning for Medical Image Processing: Overview, Challenges and Future (2017) | AE/CNN/RNN/DBN/DBM/VAE 등 6 가지 아키텍처, semi-supervised 방향              |
| Unsupervised Learning | Deep Learning for Time Series Anomaly Detection: A Survey (2022)                   | Reconstruction-based (AE/VAE/GAN), 정상 패턴 학습 후 오차 기반 anomaly score |

## (6) Generative Modeling 계열

**계열 정의**: 의료 데이터를 학습하여 synthetic sample, missing value imputation, anomaly detection, image translation 등의 태스크 수행.

**공통 특징**:

- Generator/Discriminator adversarial structure
- Cycle consistency, identity preservation
- Latent diffusion/VAE/GAN 구조

| 방법론 계열       | 논문명                                                                                                                               | 핵심 특징                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| Generative Models | Generative AI for Medical Imaging: extending the MONAI Framework (2023)                                                              | Diffusion/LDM/autoregressive/SPADE, FID/MMD/MS-SSIM metrics     |
| Generative Models | Jekyll: Attacking Medical Image Diagnostics using Deep Generative Models (2021)                                                      | CycleGAN unpaired translation, healthy→diseased style injection |
| Generative Models | A review of Generative Adversarial Networks for Electronic Health Records: applications, evaluation measures and data sources (2022) | EHR 특화 GAN, fidelity/utility/privacy trade-off                |
| Generative Models | Boosting Deep Learning Risk Prediction with Generative Adversarial Networks for Electronic Health Records (2017)                     | ehrGAN, semi-supervised data augmentation                       |

## (7) Ensemble/Stacking 계열

**계열 정의**: 여러 기본 분류기의 예측을 결합하여 일반화 성능을 높이는 meta-learning 구조.

**공통 특징**:

- Base classifier ensemble
- Neural network meta-learner stacking
- Bagging/boosting/AdaBoost

| 방법론 계열       | 논문명                                                                                                                  | 핵심 특징                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Ensemble Learning | Deep Neural Network Based Ensemble learning Algorithms for the healthcare system (diagnosis of chronic diseases) (2021) | Base classifier ensemble + neural network meta-learner, stacking |

## (8) Knowledge Distillation 계열

**계열 정의**: 고 성능 deep model 의 출력을 interpretable 모델이 모사하여 해석 가능성 확보.

**공통 특징**:

- Teacher-Student 구조
- Soft target/hidden feature mimicry
- GBT 기반 interpretable surrogate

| 방법론 계열            | 논문명                                                                                | 핵심 특징                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Knowledge Distillation | Distilling Knowledge from Deep Networks with Applications to Healthcare Domain (2015) | Teacher (DNN/LSTM/SDA) → Student (GBT), soft target loss, interpretable mimic learning |

## (9) Explainable AI (XAI) 계열

**계열 정의**: DNN 예측 결과에 대한 feature contribution 및 causal reasoning 을 attribution 관점에서 체계화.

**공통 특징**:

- Back-propagation/attention/perturbation/game theory 기반
- Local/global explanation
- Integrated Gradients/LRP/DeepLIFT

| 방법론 계열 | 논문명                                                                                                             | 핵심 특징                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| XAI         | Explainable Deep Learning in Healthcare: A Methodological Survey from an Attribution View [Advanced Review] (2021) | attribution taxonomy, baseline dependence, counterfactual generation |

## (10) Graph/Ontology Integration 계열

**계열 정의**: 의료 ontology/Knowledge DAG 를 representation learning 내부에 통합.

**공통 특징**:

- Ontology DAG 구조 활용
- Self-attention + ancestor-attention mixture
- Rare/COMMON code adaptive weighting

| 방법론 계열    | 논문명                                                                                           | 핵심 특징                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| Graph/Ontology | GRAM: Graph-based Attention Model for Healthcare Representation Learning (2016)                  | Knowledge DAG, self-attention + ancestor-attention, ontology-aware representation |
| Graph/Ontology | BEHRT: Transformer for Electronic Health Records (2019)                                          | Segment embedding, temporal encoding, multi-label classification                  |
| Graph/Ontology | MiME: Multilevel Medical Embedding of Electronic Health Records for Predictive Healthcare (2018) | (문서 단절됨)                                                                     |

## (11) Hardware/Edge Implementation 계열

**계열 정의**: 의료 환경에 적합한 하드웨어 (CMOS/FPGA/memristor/SNN) 와 edge intelligence 구조.

**공통 특징**:

- Systolic array/crossbar/spike/event 기반 연산
- Quantization/online adaptation
- Memory/energy efficiency

| 방법론 계열   | 논문명                                                                                                     | 핵심 특징                                                       |
| ------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Hardware/Edge | Hardware Implementation of Deep Network Accelerators Towards Healthcare and Biomedical Applications (2021) | CMOS/memristor/FPGA/SNN 네 가지 기술 축, EDP/latency/power 비교 |

## (12) Offline RL/OPE 계열

**계열 정의**: historical data 만으로 offline policy ranking 및 deployment selection 수행.

**공통 특징**:

- OPE (FQE/WIS/AM) estimator
- Two-stage selection scheme
- Validation-proxy policy ranking

| 방법론 계열 | 논문명                                                                                                      | 핵심 특징                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Offline RL  | Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings (2021) | OPE estimator, FQE/WIS ranking metric, offline deployment |

## (13) Deep Reinforcement Learning (DRL) 계열

**계열 정의**: MDP 프레임워크 기반 순차적 의사결정 문제 (navigation/registration/optimization) 에 DRL 적용.

**공통 특징**:

- MDP (state/action/reward/discount factor)
- Q-learning/Actor-critic
- Experience replay/replay buffer

| 방법론 계열 | 논문명                                                                     | 핵심 특징                                          |
| ----------- | -------------------------------------------------------------------------- | -------------------------------------------------- |
| Deep RL     | Deep reinforcement learning in medical imaging: A literature review (2021) | MDP formalism, multi-scale Q-network, actor-critic |

## (14) Probabilistic/Bayesian 계열

**계열 정의**: 데이터 누락·검열·분포 변화를 확률적 모델로 다룸.

**공통 특징**:

- Negative log-likelihood loss
- Missing value modeling, censoring-aware likelihood
- Uncertainty quantification

| 방법론 계열   | 논문명                                               | 핵심 특징                                                              |
| ------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| Probabilistic | Probabilistic Machine Learning for Healthcare (2021) | 불확실성 quantification, missing/censoring-aware modeling, calibration |

## (15) Security/Privacy 계열

**계열 정의**: 의료 ML 파이프라인의 보안·프라이버시·강건성 위협을 계층별/단계별로 분석·대응.

**공통 특징**:

- Attack taxonomy (influence/violation/specificity)
- Defense-in-depth, DP/HE/blockchain
- Perception/network/cloud/application 계층

| 방법론 계열      | 논문명                                                                                                 | 핵심 특징                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| Security/Privacy | Machine Learning for Healthcare-IoT Security: A Review and Risk Mitigation (2024)                      | 4 계층 공격 분석, 26 가지 공격 유형, ML 기반 IDS           |
| Security/Privacy | Federated Learning for Privacy Preservation in Smart Healthcare Systems: A Comprehensive Survey (2022) | DP/HE/blockchain/DRL, gradient poisoning/Byzantine 방어    |
| Security/Privacy | Secure and Robust Machine Learning for Healthcare Applications: A Survey (2020)                        | pipeline 단계별 취약점, threat taxonomy, poisoning defense |
| Security/Privacy | Jekyll: Attacking Medical Image Diagnostics using Deep Generative Models (2021)                        | adversarial attack, unpaired style injection               |

## (16) Time Series/Representation 계열

**계열 정의**: 시계열 의료 데이터 (이벤트/측정값) 의 불규칙성/결측/long-term dependency 처리.

**공통 특징**:

- RNN/LSTM/GRU/temporal encoding
- Missing value handling, masking
- Medical ontology embedding

| 방법론 계열 | 논문명                                                                  | 핵심 특징                                                                         |
| ----------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Time Series | Time Series Prediction Using Deep Learning Methods in Healthcare (2021) | 10 개 기술 축 (representation/missing/temporal/attention/ontology), RNN 계열 우위 |

## (17) System/Framework 계열

**계열 정의**: 의료 AI 연구/개발/배포를 위한 통합 플랫폼/프레임워크.

**공통 특징**:

- Medical image transforms/metadata-aware tensor
- Distributed training/sliding window inference
- Open-source/open API

| 방법론 계열      | 논문명                                                                  | 핵심 특징                                                         |
| ---------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| System/Framework | MONAI: An open-source framework for deep learning in healthcare (2022)  | MetaTensor/transforms/losses/engines, deterministic preprocessing |
| System/Framework | Generative AI for Medical Imaging: extending the MONAI Framework (2023) | MONAI 기반 diffusion/autoregressive/SPADE 확장                    |

## 3. 핵심 설계 패턴 분석

## (1) Pre-training → Adaptation 패턴

| 단계             | 목적                                                              | 기법                                                             |
| ---------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Pre-training** | 대규모 unlabeled/structured data 에서 general representation 학습 | GL/CL/HL/SL, MLM, NSP, contrastive learning, SSL                 |
| **Adaptation**   | Downstream medical task 로 전이                                   | Fine-tuning/Adapter Tuning/Prompt Engineering/Instruction Tuning |

**해당 논문**: Foundation Model, BEHRT, BERT family (encoder-only)

## (2) Representation Learning → Prediction 패턴

| 단계               | 목적                               | 기법                                                     |
| ------------------ | ---------------------------------- | -------------------------------------------------------- |
| **Representation** | Raw clinical code → Latent vector  | Embedding layer, AE, MLM, ontology-aware attention       |
| **Prediction**     | Prediction head 로 downstream task | Multi-label classification, risk prediction, phenotyping |

**해당 논문**: BEHRT, Deep EHR, GRAM, MiME, DeepCare |

## (3) Unsupervised/Reconstruction → Downstream 패턴

| 단계             | 목적                                    | 기법                                              |
| ---------------- | --------------------------------------- | ------------------------------------------------- |
| **Unsupervised** | Reconstruction error/density estimation | AE/VAE/GAN, reconstruction loss, KL divergence    |
| **Downstream**   | Representation 을 활용                  | Feature extraction, clustering, denoising support |

**해당 논문**: Deep Learning for Medical Image Processing, A Tour of Unsupervised Deep Learning, Deep Learning for TS Anomaly Detection |

## (4) Multi-Modal Fusion 패턴

| 단계                 | 목적                               | 기법                                    |
| -------------------- | ---------------------------------- | --------------------------------------- |
| **Data Fusion**      | 수집/정렬/전처리/feature selection | Early fusion, feature-level aggregation |
| **Info Fusion**      | Deep architectures/attention       | Hybrid fusion, cross-modal alignment    |
| **Knowledge Fusion** | Clinical knowledge/CDSS            | Rule-based/Knowledge graph integration  |
| **Wisdom**           | Decision support                   | Predictive/preventive intervention      |

**해당 논문**: Multimodal Information Fusion, Emotion Recognition |

## (5) Federated/Decentralized 패턴

| 단계               | 목적                    | 기전                                          |
| ------------------ | ----------------------- | --------------------------------------------- |
| **Initialization** | Server 초기 model 배포  | Client selection, task definition             |
| **Local Training** | Client 로컬 데이터 학습 | Local gradient/parameter update               |
| **Aggregation**    | Server 가 업데이트 집계 | FedAvg/data-size-weighted averaging           |
| **Iteration**      | 수렴 조건 반복          | Stop criteria: convergence/accuracy threshold |

**해당 논문**: Federated Learning papers (A Federated Learning Framework, Federated Learning Survey), Split Learning |

## (6) Ensemble/Stacking 패턴

| 단계              | 목적                                 | 기법                                    |
| ----------------- | ------------------------------------ | --------------------------------------- |
| **Base Training** | 기본 분류기 개별 학습                | Logistic Regression/Naive Bayes/KNN/CNN |
| **Meta-Learning** | Base 예측 출력을 입력한 meta-learner | Neural network stacking, DeepNN_SG      |

**해당 논문**: Deep Neural Network Based Ensemble learning Algorithms |

## (7) Knowledge Distillation 패턴

| 단계                 | 목적                             | 기법                               |
| -------------------- | -------------------------------- | ---------------------------------- |
| **Teacher Training** | 고 성능 DNN 학습                 | Cross-entropy loss                 |
| **Distillation**     | Soft target/hidden feature mimic | GBT surrogate, soft target loss    |
| **Interpretation**   | Tree rule/feature importance     | Interpretable phenotype extraction |

**해당 논문**: Distilling Knowledge from Deep Networks |

## (8) Attribution/XAI 패턴

| 단계              | 목적                      | 기법                                          |
| ----------------- | ------------------------- | --------------------------------------------- |
| **Attribution**   | Feature contribution 계산 | Gradient/Relevance (Integrated Gradients/LRP) |
| **Visualization** | Heatmap/relevance map     | Post-hoc explanation                          |
| **Evaluation**    | 설명 신뢰도 평가          | Causal vs. non-causal, counterfactual         |

**해당 논문**: Explainable Deep Learning in Healthcare |

## (9) Hardware Adaptation 패턴

| 단계                   | 목적                           | 기법                          |
| ---------------------- | ------------------------------ | ----------------------------- |
| **Model Quantization** | Fixed-point/weight mapping     | FPGA/memristor compatibility  |
| **Inference**          | Parallel MAC/event propagation | Systolic array/crossbar/spike |
| **Online Adaptation**  | Patient-specific tuning        | SNN on-chip adaptation        |

**해당 논문**: Hardware Implementation of Deep Network Accelerators |

## 4. 방법론 비교 분석

## (1) 문제 접근 방식 차이

| 접근 방식            | 계열                 | 장점                                        | 단점                                       |
| -------------------- | -------------------- | ------------------------------------------- | ------------------------------------------ |
| **Foundation Model** | FM 기반 파이프라인   | General representation, few-shot adaptation | 대규모 데이터 필요, computational cost     |
| **Federated**        | 분산 학습            | Privacy-by-design, data locality            | Communication overhead, heterogeneous data |
| **Fusion**           | Multimodal           | Comprehensive clinical insight              | Complex integration, alignment cost        |
| **Unsupervised**     | Reconstruction-based | Unlabeled data 활용, pre-training           | Reconstruction ≠ semantic meaning          |
| **Ensemble**         | Stacking             | Robustness, generalization                  | Computational overhead                     |

## (2) 구조/모델 차이

| 축           | FM 계열                         | FL 계열                    | Fusion 계열                                    |
| ------------ | ------------------------------- | -------------------------- | ---------------------------------------------- |
| **Model**    | Encoder-only/Decoder-only/Dual  | FedAvg/Horizontal/Vertical | Early/Late/Hybrid                              |
| **Training** | Self-supervised/Semi-supervised | Distributed SGD            | Multi-objective (reconstruction + adversarial) |
| **Data**     | Centralized                     | Decentralized              | Multi-modal                                    |

## (3) 적용 대상 차이

| 적용 대상         | FM                        | FL                  | Fusion                   | Unsupervised                |
| ----------------- | ------------------------- | ------------------- | ------------------------ | --------------------------- |
| **Data type**     | Text/Image/omics          | IoMT/health sensors | EHR+Imaging+Genomic      | Medical imaging/TSAD        |
| **Clinical task** | Diagnosis/risk prediction | Remote monitoring   | Clinical decision        | Representation pre-training |
| **Environment**   | Clinical AI platform      | IoMT/edge devices   | Smart healthcare systems | Medical image analysis      |

## (4) 복잡도 및 확장성 차이

| 계열             | Scalability                 | Privacy                      | Computational cost              | Data requirement |
| ---------------- | --------------------------- | ---------------------------- | ------------------------------- | ---------------- |
| **FM**           | High (multi-modal scaling)  | Low (centralized)            | High                            | Very high        |
| **FL**           | Medium (client-dependent)   | High (by design)             | Medium (communication overhead) | Distributed      |
| **Fusion**       | Medium (modality-dependent) | Medium (requires protection) | Medium-high                     | Multi-modal data |
| **Unsupervised** | High                        | Medium                       | Medium                          | High (unlabeled) |
| **Ensemble**     | High                        | Low                          | High                            | High             |

## 5. 방법론 흐름 및 진화

## 초기 접근 (2015-2018): Representation Learning 중심

| 시기     | 주요 방법론                                                               | 특징                                                |
| -------- | ------------------------------------------------------------------------- | --------------------------------------------------- |
| **2015** | Knowledge Distillation (Distilling Knowledge from Deep Networks)          | Teacher-Student, interpretable surrogate            |
| **2016** | DeepCare (Deep Dynamic Memory), GRAM (Graph-based Attention)              | LSTM modification, ontology integration             |
| **2017** | Boosting with GAN, Medical Image Processing survey                        | GAN for data augmentation, AE/CNN/RNN taxonomy      |
| **2018** | Deep EHR, MiME, Natural Language Generation, Unsupervised Learning survey | EHR representation, encoder-decoder, AE/RBM/DBN/GAN |

**기법 특징**:

- 수작업 feature representation 또는 embedding
- RNN/LSTM 기반 시계열 처리
- GAN 을 생성/증강 도구로 사용

## 발전된 구조 (2019-2021): Model Architecture Integration

| 시기     | 주요 방법론                                                 | 특징                                                |
| -------- | ----------------------------------------------------------- | --------------------------------------------------- |
| **2019** | BEHRT (Transformer for EHR), Split Learning                 | Transformer for EHR, decentralized training         |
| **2020** | Federated Learning survey, Attention survey, EEG-BCI survey | FL 기본 구조, attention taxonomy, domain adaptation |
| **2021** | XAI survey, Probabilistic ML, Deep RL, Ensemble, Security   | XAI taxonomy, uncertainty, DRL/MDP, OPE             |

**기법 특징**:

- Transformer attention integration
- Federated learning 기본 설계
- XAI attribution taxonomy
- Probabilistic modeling for uncertainty

## 최근 경향 (2022-2024): Integration & Standardization

| 시기     | 주요 방법론                                                                               | 특징                                                         |
| -------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **2022** | MONAI framework, Security survey, Deep TSAD survey, FL/Privacy survey, GAN for EHR survey | Standardization, security taxonomy, unified framework        |
| **2023** | Multimodal Fusion, Generative AI (MONAI extension)                                        | DIKW framework, diffusion/autoregressive integration         |
| **2024** | Foundation Model survey, Security ML for H-IoT                                            | Unified taxonomy, comprehensive security, instruction tuning |

**기법 특징**:

- 오픈소스 프레임워크 표준화 (MONAI)
- Foundation model 통합 분류 체계
- Security threat taxonomy 통합
- Multimodal integration 강화

## 6. 종합 정리

**전체 방법론 지형**은 두 가지 축으로 구조화된다.

**1 축 (Learning Paradigm)**: Unsupervised → Semi-supervised → Supervised → Foundation Model adaptation. 초기 연구는 unsupervised representation 학습 (AE/RBM/GAN) 에 집중했고, 발전 과정에서는 supervised fine-tuning 과 foundation model pre-training/adaptation 으로 전환되었다.

**2 축 (Architecture/Structure)**: Centralized → Federated/Decentralized → Multi-modal Fusion. 중앙 집중식 학습에서 분산/분할/융합 구조로 진화했다.

**방법론 계열 간 관계**: Foundation model 이 핵심 back-bone 를 제공하고, federated learning 이 데이터 분산 환경에서 구현 방식, multimodal fusion 이 임상적 통합을 담당하며, XAI/security/hardware 가 robustness 와 interpretability 를 보완하는 모듈화 구조를 이룬다.

**재귀적 패턴**: Pre-training → Adaptation, Teacher → Student, Teacher → Feature → Surrogate 로 반복되는 설계 언어가 관찰되며, 각 단계마다 clinical interpretability 와 privacy 요구를 반영한 확장 (adapter/adapter tuning/DP/HE) 이 추가된다.

**최종 요약**: 본 장은 28 편의 논문 문헌을 분석하여 의료 AI 방법론을 16 계열로 분류하고, 설계 패턴 9 개, 진화 흐름 3 단계로 재구성했다. 방법론은 데이터→representation→prediction 파이프라인으로 공통되나, 학습 패러다임 (unsupervised/foundation/federated) 과 구조적 접근 (centralized/fusion/XAI) 으로 세분화된다.

사용자가 제공한 30 편의 논문의 실험 결과 자료를 바탕으로 "3 장. 실험 결과 분석"을 작성합니다. 자료를 먼저 분석한 후 구조화된 분석 문서를 생성하겠습니다.

## 3 장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 1.1 주요 데이터셋 유형

전체 논문의 2/3 가 다음 핵심 데이터셋들을 활용했습니다.

| 데이터셋 유형 | 대표 사례                                         | 활용 분야           |
| ------------- | ------------------------------------------------- | ------------------- |
| 의료 영상     | MIMIC-CXR, CheXpert, TCIA, ADNI, UK Biobank       | 분류/분할/등록      |
| EHR 텍스트    | MIMIC-III/IV, CPRD, VUMC, New York Emergency      | 예측/코딩/생성      |
| 시계열        | 2002-2013 병원 EMR, Heart Failure/Diabetes cohort | 위험 예측           |
| 공개 벤치마크 | UCI Diabetes/Heart/Breast, Diabetic Retinopathy   | 진단                |
| 시뮬레이션    | COOJA simulator, OpenAI Gym Lunar                 | IoT 보안/OFFLINE RL |

#### 1.2 평가 환경

| 환경 유형              | 사례                              | 활용 목적                 |
| ---------------------- | --------------------------------- | ------------------------- |
| 실시간 환경            | hand-gesture recognition, EEG BCI | edge inference            |
| 시뮬레이션             | sepsis treatment, COOJA IoT       | RL/보안 정책 평가         |
| 시뮬레이션/실환경 혼합 | CAT08, spine/heart dataset        | registration/segmentation |
| 비의료 synthetic       | MNIST/Fashion-MNIST               | FL 기초 연구 (infancy)    |

#### 1.3 비교 방식

- **baseline 대비 성능**: Deepr/RETAIN (BEHRT), CNN/SVM (의료 영상), Random Forest/LSTM (DeepCare)
- **의료 특화 vs 일반-purpose**: MedCLIP vs CLIP, MedSAM vs 일반 SAM
- **FL 유형 간**: FedAvg vs custom FL (network decomposition + sparsification)
- **하드웨어 플랫폼**: Loihi/ODIN/Jetson/FPGA/memristive

#### 1.4 주요 평가 지표

| 지표 유형   | 대표 지표                                                      | 사용 빈도 |
| ----------- | -------------------------------------------------------------- | --------- |
| 분류 성능   | AUC, accuracy, F1, sensitivity/specificity, precision@k        | 매우 높음 |
| 재현 품질   | FID, MS-SSIM, reconstruction error                             | 생성 모델 |
| 보안/강건성 | accuracy degradation, false negative rate, attack success rate | 보안 연구 |
| 효율성      | inference energy, latency, EDP, communication traffic          | edge/IoT  |
| 해석 가능성 | faithfulness, class-discriminativeness, saliency map quality   | XAI       |

### 2. 주요 실험 결과 정렬

#### 2.1 의료 영상 생성 모델 성능 비교

| 논문명                                      | 데이터셋                    | 평가 지표            | 핵심 결과                                  |
| ------------------------------------------- | --------------------------- | -------------------- | ------------------------------------------ |
| Generative AI for Medical Imaging (2023)    | MIMIC-CXR/CSAW-M/UK Biobank | FID/MS-SSIM/Recons   | 2D X-ray FID 8.83, 3D MRI FID 0.005        |
| A Tour of Unsupervised Deep Learning (2018) | ABIDE/TCIA/DDSM 등          | reconstruction error | GAN 기반 retina/chest X-ray synthesis 성공 |

#### 2.2 EHR 기반 예측 모델 성능

| 논문명                 | 데이터셋                      | 비교 대상            | 평가 지표    | 핵심 결과                         |
| ---------------------- | ----------------------------- | -------------------- | ------------ | --------------------------------- |
| DeepCare (2016)        | Diabetes/Mental health cohort | RF/RNN/LSTM/DeepCare | F-score      | RF 71.4% → DeepCare 79.0% (+7.6p) |
| BEHRT (2019)           | CPRD/HES 160 만 환자          | Deepr/RETAIN         | APS          | +8.0~10.8% 상대 향상              |
| Deep EHR survey (2018) | MIMIC/TCGH 등                 | traditional ML       | AUC/PR-AUC   | temporal 모델이 static 보다 우세  |
| GRAM (2016)            | CCS ICD-9 hierarchy           | RNN                  | AUC/accuracy | rare disease 기준 +10%            |
| MiME (2018)            | Sutter Health 3 만 환자       | Med2Vec/MLP          | PR-AUC       | 작은 데이터셋 대비 +15%           |

#### 2.3 FL/분산 학습 통신 효율성

| 논문명                              | 환경                         | 비교 대상                     | 평가 지표             | 핵심 결과                           |
| ----------------------------------- | ---------------------------- | ----------------------------- | --------------------- | ----------------------------------- |
| Federated Learning Framework (2020) | PhysioNet 2017/16-64 devices | vanilla SGD/FedAvg/SplitNN    | traffic/accuracy loss | FedAvg 2.72GB → 12.8MB (99.8% 절감) |
| Split Learning (2019)               | DR/CheXpert                  | centralized vs split learning | AUROC/accuracy        | 3→50 client 급격한 하락 방지        |

#### 2.4 하드웨어 플랫폼 성능 비교 (hand-gesture recognition)

| 플랫폼                  | 정확도 | 에너지    | 지연    | EDP       |
| ----------------------- | ------ | --------- | ------- | --------- |
| Loihi (SNN)             | 96.0%  | 1104.5 µJ | 7.75 ms | 8.6       |
| ODIN (SNN)              | 89.4%  | 37.4 µJ   | -       | 0.42      |
| memristive (presilicon) | 96.2%  | 4.83 µJ   | -       | 4.83×10⁻⁶ |
| Jetson Nano (GPU)       | 95.4%  | 32,100 µJ | -       | -         |
| FPGA                    | 94.8%  | 31,200 µJ | -       | -         |

#### 2.5 의료 영상 분류 성능 (CNN 기반)

| 작업                 | 데이터셋             | 모델      | 성능                    |
| -------------------- | -------------------- | --------- | ----------------------- |
| diabetic retinopathy | EyePACS-1/Messidor-2 | Gulshan   | sens 97.5% / spec 93.4% |
| malaria detection    | microscopy           | GoogLeNet | AUC 98.66%              |
| breast imaging       | mammography          | CNN+SVM   | AUC 88%                 |
| neuroimaging         | ADNI/PET/MRI         | LeNet-5   | 96.86%                  |

### 3. 성능 패턴 및 경향 분석

#### 3.1 공통 성능 개선 패턴

**데이터 규모 효과**:

- DeepCare: 작은 데이터셋일수록 structure-aware 모델 (MiME/GRAM) 의 상대적 우위 확대
- GRAM: rare code(rare disease)일수록 상위 ontology 의존성 증가로 성능 향상
- MiME: EHR 데이터가 적을수록 baseline 대비 성능 격차 확대

**Temporal modeling 의 우위**:

- Deep EHR survey: temporal prediction 모델이 static classification 일관되게 우세
- BEHRT: MLM pretraining 으로 질환 간 장거리 상관관계 포착 가능

#### 3.2 데이터셋/환경에 따른 성능 차이

| 환경 조건                           | 영향                    | 사례                     |
| ----------------------------------- | ----------------------- | ------------------------ |
| rare diagnosis/code                 | baseline 의 한계 극대화 | GRAM (+10%), MiME (+15%) |
| temporal sequence                   | static 대비 우위        | Deep EHR survey          |
| 3D vs 2D 영상                       | FID 0.005 vs 8.83       | Generative AI (2023)     |
| data shift (temporal/institutional) | performance degradation | Secure ML survey         |

#### 3.3 상충 관계 (trade-off)

| trade-off                      | 설명                              | 사례                        |
| ------------------------------ | --------------------------------- | --------------------------- |
| privacy vs accuracy            | FL 에서 trade-off 존재            | Federated Learning (2021)   |
| communication vs accuracy      | traffic 절감이 accuracy 소폭 수용 | Federated Learning (2020)   |
| interpretability vs complexity | tree model 과 deep model 균형     | Distilling Knowledge (2015) |
| fidelity vs utility            | synthetic data 품질 대 실용성     | EHR GAN survey (2022)       |

### 4. 추가 실험 및 검증 패턴

#### 4.1 ablation study

| 연구                               | ablation 방식                                    |
| ---------------------------------- | ------------------------------------------------ |
| DeepCare (2016)                    | pooling strategy(sum/multiscale/parametric time) |
| Natural Language Generation (2018) | decoding(beam search k=3/5/10)                   |
| Boosting GAN (2017)                | ρ parameter (0~1)                                |
| Split Learning (2019)              | client 수 3/10/20/50                             |

#### 4.2 일반화 전략

| 전략               | 적용 사례                                  |
| ------------------ | ------------------------------------------ |
| few-shot/zero-shot | 의료 코딩, GRAM rare code                  |
| domain adaptation  | BCI calibration reduction (90%)            |
| transfer learning  | ImageNet pretrained feature transfer       |
| human-in-the-loop  | active learning, human-grounded evaluation |

#### 4.3 robustness 검증

| 검증 유형                    | 사례                                     |
| ---------------------------- | ---------------------------------------- |
| temporal/institutional shift | Secure ML survey                         |
| adversarial attack           | Jekyll deepfake, XAI robustness          |
| distribution shift           | Secure ML survey performance degradation |
| multi-center validation      | Split Learning, FL                       |

### 5. 실험 설계의 한계 및 비교 주의점

#### 5.1 비교 조건의 불일치

- FL 연구: 비의료 MNIST 기반 평가가 헬스케어 데이터셋 부재
- hardware 비교: memristive은 presilicon/network-specific 가정
- EHR 데이터: MIMIC 기반 편향으로 기관 일반화 제한

#### 5.2 데이터셋 의존성

- MIMIC-centric: 77% 논문이 MIMIC 기반 (Time Series Prediction)
- public/private 이질성: 기관별 ICD/CPT/LOINC/RxNorm 코드 체계 차이
- 시뮬레이션 환경: COOJA/OpensAI Gym 임상 검증 부재

#### 5.3 평가 지표 불일치

- unified metric 부재: TSAD 에서 evaluation metric/thresholding 다양성
- point-level vs range-level metric: TSAD 비교 왜곡
- single-label vs multi-label: CheXpert 5 task 평균AUROC

#### 5.4 일반화 한계

- 2017-2021 년 시점 중심: foundation model/transformer 미반영
- single-task 중심: single-lead ECG, specific disease
- clinical deployment: workflow/redesign/reimbursement 규제 문제

### 6. 결과 해석의 경향

#### 6.1 저자 해석의 공통 경향

1. **의료 특화화 필수성 강조**: 일반-purpose model 의 의료 데이터 한계 → medical-specific design 필요
2. **structure-aware 모델링**: EHR 의 진단-치료 관계, ontology hierarchy 활용이 성능 개선 핵심
3. **temporal modeling**: patient trajectory modeling 이 static classification 으로 패러다임 이동
4. **multimodal 융합**: DIKW 계층에서 data→wisdom 으로 융합, 단순 concatenation 이 아닌 semantic alignment

#### 6.2 해석과 관찰 결과 구분

| 해석 주장                       | 관찰 결과                                              |
| ------------------------------- | ------------------------------------------------------ |
| FL 로 privacy 유지 가능         | traffic 99.8% 절감, accuracy loss 2% 미만 (조건부)     |
| GAN 은 synthetic data 생성 가능 | FID 1.91~8.83, MS-SSIM 0.43~0.92 (modality 의존)       |
| DRL 은 search 문제 해결         | accuracy +20%, success rate +11%, inference 7x 속도    |
| XAI 는 trust calibration        | method 간 sensitivity/faithfulness 차이, ensemble 필요 |

### 7. 종합 정리

의료 AI 실험 결과들은 **데이터 구조화 (의료 특화화)**와**Temporal modeling**이 핵심 성과 축임을 보여줍니다. 일반-purpose foundation model(200 편 논문 분석)보다 의료 특화 모델 (MedCLIP, MedSAM, GatorTronGPT) 이 의료 도메인에서 구조적 이점 (ontology, temporal sequence, rare code 대응) 을 입증합니다. FL 은 privacy 를 보장하면서 centralized 성능의 90% 이상을 유지하나, 실제 헬스케어 데이터셋 기반 연구는 infancy 단계입니다. 생성 모델 (diffusion, GAN) 은 2D/3D 모달리티 일관성 입증 (FID 0.005~8.83) 하지만 synthetic data 의 clinical utility 는 metric 한계를 넘어 실제 outcome 과 연계해야 합니다. 하드웨어 측면에서는 spiking/neuromorphic 하드웨어 (Loihi/ODIN/memristive) 가 always-on monitoring 에 적합 (EDP 0.42~8.6) 하지만 CMOS/GPU가 현재 실용적 주류입니다. 핵심 시사점은:**의료 데이터의 구조적 특성 (temporal, ontology, rare code) 을 명시적으로 반영하는 모델 설계**가 단순 모델 크기 경쟁보다 성능 개선의 핵심이며,**FL, transfer learning, few-shot learning**이 data 부족 문제를 해결하는 필수 전략입니다.
