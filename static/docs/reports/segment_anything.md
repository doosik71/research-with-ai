# Segment Anything

## 서론

### 1. 연구 배경

Segment Anything는 대규모 SA-1B 데이터셋(1,100 만 이미지, 11 억 마스크)으로 사전 훈련된 파운데이션 모델로서 제로샷 분할 성능을 입증한 기술이다. 본 보고서는 SAM 생태계에 포함된 68 편 핵심 논문을 체계적으로 분석하며, 기초 모델 구축(7 편), 도메인 적응/특수 응용(31 편), 효율화 및 경량화(13 편), 비디오 분할 및 추적(8 편), 프롬프트 엔지니어링(6 편), 종합 조사(6 편) 등 주요 연구 범주를 정리한다.

### 2. 문제의식 및 분석 필요성

초기 SAM 의 제로샷 일반화 능력은 다양한 도메인 적용 가능성을 보장했으나, 의료 영상, 은폐 장면, 저품질 이미지, 원격탐사 등 특정 도메인에서는 도메인 간 격차가 존재한다. 또한 경량화 모델, 자동 프롬프트 생성, 시공간 확장 등 다양한 방법론이 등장하면서 성능과 효율성 간의 트레이드오프 관계가 중요해졌다. 이러한 방법론적 다양성과 실험 결과들의 상충되는 관점들을 체계적으로 비교·분석할 필요가 있다.

### 3. 보고서의 분석 관점

본 보고서는 연구 대상 논문을 세 가지 관점으로 정리한다. 첫째는 **연구체계 분류**로, 주요 연구 목적, 기술적 접근 방식, 적용 대상, 연구 기여도 등의 원칙에 따라 논문을 6 개 범주로 분류한다. 둘째는**방법론 분석**으로, 공통 문제 설정, 계열별 상세 분석, 핵심 설계 패턴, 학습 전략 등 방법론적 기법을 체계적으로 비교한다. 셋째는**실험결과 분석**으로, 주요 실험 결과 정렬, 성능 패턴, 비교 조건의 불일치, 결과 해석 경향 등을 종합하여 평가한다.

### 4. 보고서 구성

- **1장 연구체계 분류**: 주요 연구 목적, 기술적 접근 방식, 적용 대상, 연구 기여도 등의 기준에 따라 기초 모델 및 파운데이션 모델 연구, 도메인 적응 및 특수 응용 연구, 효율화 및 경량화 연구, 비디오 분할 및 추적 확장 연구, 프롬프트 엔지니어링 및 전략 연구, 종합/조사와 리뷰 연구로 논문을 체계적으로 분류한다.

- **2장 방법론 분석**: 공통 문제 설정 및 접근 구조, 주요 계열 분류, 핵심 설계 패턴 분석, 방법론 비교 분석, 방법론 흐름 및 진화 등을 다룬다. 인코더/프롬프트 처리/디코더/학습/손실 함수 패턴, 도메인 적응/2D vs 3D 확장/효율성/프롬프트 효율성 비교 등을 분석한다.

- **3장 실험결과 분석**: 주요 데이터셋 유형, 평가 환경 분류, 비교 대상 및 평가 방식, 주요 평가 지표를 정리하며, 주요 실험 결과 정렬, 성능 패턴 및 경향 분석, ablation study, 민감도 분석, 조건 변화 실험 결과를 종합한다.

## 1장. 연구체계 분류

### 1.1 연구 분류 체계 수립 기준

본 연구 분류 체계는 논문들을 다음과 같은 원칙에 따라 체계적으로 분류한다.

- **주요 연구 목적**에 따른 분류 (기초 모델 구축, 도메인 적응, 효율화 등)
- **기술적 접근 방식**에 따른 분류 (프롬프트, 아키텍처, 데이터 중심 등)
- **적용 대상**에 따른 분류 (의료 영상, 비디오, 원격탐사, 엣지 디바이스 등)
- **연구 기여도**에 따른 분류 (샘플 논문, 실증 연구, 종합 조사를 구분)

### 1.2 연구 분류 체계

#### 2.1 기초 모델 및 파운데이션 모델 연구

| 분류                            | 논문명                                                                        | 분류 근거                                                     |
| ------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 기초 모델 > 원본 모델           | Segment Anything (2023)                                                       | 프롬프트 기반 세그멘테이션 태스크 정의 및 SA-1B 데이터셋 구축 |
| 기초 모델 > 종합 조사           | A Survey on Segment Anything Model (2023)                                     | SAM 및 SAM 2 아키텍처에 대한 체계적 문헌 검토                 |
| 기초 모델 > 영상 및 비전을 넘어 | A Comprehensive Survey on Segment Anything Model for Vision and Beyond (2023) | 이미지/비디오/3D/멀티모달 도메인 확장 가능성 분석             |
| 기초 모델 > 2D/3D 의료 영상     | Segment Anything model 2: an application to 2D and 3D medical images (2024)   | SAM 2 의 의료 영상 2D/3D 적용 평가                            |
| 기초 모델 > SAM 2               | SAM 2: Segment Anything in Images and Videos (2024)                           | 스트리밍 메모리 기반 이미지/비디오 통합 분할 모델             |
| 기초 모델 > SAM 3               | SAM 3: Segment Anything with Concepts (2025)                                  | 텍스트/이미지 프롬프트 개념 분할 (PCS) 작업                   |
| 기초 모델 > SAM 2 종합          | SAM 2 for Image and Video Segmentation: A Comprehensive Survey (2025)         | SAM 2 의 강점/약점 도메인별 종합 분석                         |
| 기초 모델 > 비디오 분할         | Segment Anything for Videos: A Systematic Survey (2024)                       | 비디오 분할 도메인 체계적 분류                                |

#### 2.2 도메인 적응 및 특수 응용 연구

| 분류                    | 논문명                                                         | 분류 근거                                          |
| ----------------------- | -------------------------------------------------------------- | -------------------------------------------------- |
| 도메인 적응 > 의료 영상 | Segment Anything Model for Medical Image Analysis (2023)       | 의료 영상 데이터셋별 제로샷 분할 성능 평가         |
| 도메인 적응 > 의료 영상 | Medical SAM Adapter (2023)                                     | PEFT 기반 2D→3D 의료 영상 도메인 적응              |
| 도메인 적응 > 의료 영상 | Input Augmentation with SAM (2023)                             | SAM 출력을 활용한 의료 영상 입력 증강              |
| 도메인 적응 > 의료 영상 | Customized SAM for Medical Image Segmentation (2023)           | LoRA 기반 효율적 파인튜닝 및 의미론적 분할         |
| 도메인 적응 > 의료 영상 | How to Efficiently Adapt SAM to Medical Images (2023)          | 프롬프트-없는 예측 헤드 및 AutoSAM 제안            |
| 도메인 적응 > 의료 영상 | Polyp-SAM (2023)                                               | 전이 학습 전략 비교 및 다기관 일반화               |
| 도메인 적응 > 의료 영상 | AdaptiveSAM (2023)                                             | 바이어스 튜닝 및 텍스트 어파인 레이어              |
| 도메인 적응 > 의료 영상 | SAM-Med2D (2023)                                               | 10 개 모달리티/31 개 장기 프롬프트 통합 지원       |
| 도메인 적응 > 의료 영상 | SA-Med2D-20M Dataset (2023)                                    | 1,970 만 개 마스크 대규모 의료 분할 데이터셋 구축  |
| 도메인 적응 > 의료 영상 | Segment Anything for Brain Tumor Segmentation (2023)           | BraTS 2D MRI 슬라이스에서의 제로샷 평가            |
| 도메인 적응 > 의료 영상 | Segment Anything in Medical Images and Videos (2024)           | 11 개 모달리티 SAM2 벤치마킹 및 임상 배포          |
| 도메인 적응 > 의료 영상 | Is SAM 2 Better than SAM in Medical Image Segmentation? (2024) | 영상 양식별 성능 비교 및 음성 프롬프트 효과        |
| 도메인 적응 > 의료 영상 | S-SAM: SVD-based Fine-Tuning (2024)                            | 특이값 미세 조정 및 텍스트 프롬프트 분할           |
| 도메인 적응 > 의료 영상 | Segment anything, from space? (2023)                           | 항공/위성 이미지에서의 SAM 일반화 평가             |
| 도메인 적응 > 원격탐사  | RSAM-Seg (2024)                                                | 원격 감지 이미지 시맨틱 분할 및 자동 프롬프트 생성 |
| 도메인 적응 > 원격탐사  | ROS-SAM (2025)                                                 | 원격 탐사 비디오 움직는 객체 분할 및 제로샷        |
| 도메인 적응 > 위장 장면 | SAM Struggles in Concealed Scenes (2023)                       | 위장/결함/병변 은폐 장면에서의 한계 분석           |
| 도메인 적응 > 위장 장면 | Segment Anything Is Not Always Perfect (2023)                  | 농업/제조/헬스케어 등 실제 응용 평가               |
| 도메인 적응 > 영상 품질 | RobustSAM (2024)                                               | 블러/노이즈/악천후 등 저품질 이미지에 강인성       |
| 도메인 적응 > 품질 저하 | Segment anything without supervision (2024)                    | 감독 없이 Divide-and-Conquer 전략으로 생성         |
| 도메인 적응 > 3D 분할   | Segment Anything in 3D with Radiance Fields (2023)             | 2D SAM 을 래디언스 필드와 결합한 3D 분할           |
| 도메인 적응 > 3D 분할   | Gaussian Grouping (2023)                                       | 3D 가우시안 스플래팅 기반 2D-3D 전이 및 편집       |
| 도메인 적응 > 3D 분할   | SANeRF-HQ (2023)                                               | SAM-NeRF 통합 고품질 3D 분할 프레임워크            |
| 도메인 적응 > 3D 의료   | A Federated Learning-Friendly Approach (2024)                  | FL 기반 3D 의료 영상 연합 학습 적응                |

#### 2.3 효율화 및 경량화 연구

| 분류                    | 논문명                                                 | 분류 근거                                       |
| ----------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| 효율화 > 경량 인코더    | Faster Segment Anything (2023)                         | CNN/YOLOv8 기반 50 배 빠른 실시간 솔루션        |
| 효율화 > 경량 인코더    | Faster Segment Anything Mobile (2023)                  | 디커플드 증류 및 5M 파라미터 TinyViT            |
| 효율화 > 경량 인코더    | MobileSAM (2023)                                       | 경량 인코더/디코더 결합 최적화 및 모바일 배포   |
| 효율화 > 경량 인코더    | MobileSAMv2 (2023)                                     | SegEvery 병목 해결 및 YOLOv8 바운딩 박스 샘플링 |
| 효율화 > 경량 인코더    | TinySAM (2023)                                         | 지식 증류/양자화/계층적 추론을 통한 효율화      |
| 효율화 > 효율적 SAM     | EfficientSAM (2023)                                    | SAM 이미지 인코더 특징 재구성 자기 지도 학습    |
| 효율화 > 엣지 배포      | EdgeSAM (2023)                                         | 온-디바이스 프롬프트 증류 및 CNN 아키텍처       |
| 효율화 > 모델 압축      | PTQ4SAM (2024)                                         | SAM 특화 PTQ 프레임워크 및 이중 모드 분포 처리  |
| 효율화 > 효율 변형 종합 | On Efficient Variants of Segment Anything Model (2024) | 효율 변형 모델 체계적 분류 및 하드웨어별 평가   |
| 효율화 > 경량 디코더    | HQ-SAM (2023)                                          | 최소 수정 고품질 분할 및 경량 적응 전략         |
| 효율화 > 경량 디코더    | PA-SAM (2024)                                          | 디코더 내 프롬프트 어댑터 병렬 연결             |
| 효율화 > 경량 디코더    | Stable SAM (2023)                                      | 플러그인 방식 저품질 프롬프트 환경 안정화       |

#### 2.4 비디오 분할 및 추적 확장 연구

| 분류                           | 논문명                                                     | 분류 근거                                      |
| ------------------------------ | ---------------------------------------------------------- | ---------------------------------------------- |
| 비디오 분할 > SAM+XMem         | Track Anything (2023)                                      | SAM 과 XMem 통합 단일 추론 대화형 추적         |
| 비디오 분할 > 통합 프레임워크  | Segment and Track Anything (2023)                          | SAM+DeAOT+Grounding-DINO 통합 비디오 분할      |
| 비디오 분할 > 2단계 파이프라인 | Tracking Anything with Decoupled Video Segmentation (2023) | 이미지-시간 전파 구조 분리 및 클립 내 합의     |
| 비디오 분할 > 점 추적          | Segment Anything Meets Point Tracking (2023)               | 점 기반 상호작용 및 비디오 분할 확장           |
| 비디오 분할 > 고품질 추적      | Tracking Anything in High Quality (2023)                   | VMOS+MR 2 단계 파이프라인 통합 추적 프레임워크 |
| 비디오 분할 > 움직임 분할      | Moving Object Segmentation: All You Need Is SAM (2024)     | SAM 과 광학 흐름 결합 Frame/Sequence 레벨 처리 |
| 비디오 분할 > 제로샷 추적      | Segment Anything Meets Point Tracking (2023)               | 비디오에서 제로샷 일반화 및 점 전파            |
| 비디오 분할 > VOS 챌린지       | HQTrack (2023)                                             | VOTS2023 공동 추적 AUC 0.615 달성              |

#### 2.5 프롬프트 엔지니어링 및 전략 연구

| 분류                          | 논문명                                                    | 분류 근거                                            |
| ----------------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| 프롬프트 > 텍스트 융합        | Deep Instruction Tuning (2024)                            | 텍스트 시맨틱 투사 기반 깊은 융합                    |
| 프롬프트 > 텍스트 융합        | Deep Instruction Tuning for Segment Anything Model (2024) | 레이어 단위 텍스트 시맨틱 공간 투사                  |
| 프롬프트 > 프롬프트 생성      | AI-SAM (2023)                                             | AI-Prompter 자동 프롬프트 생성기 및 휴리스틱 손실    |
| 프롬프트 > 프롬프트 품질      | Stable SAM (2023)                                         | 저품질 프롬프트 환경에서 안정성 및 동적 라우팅       |
| 프롬프트 > 밀집/희소 프롬프트 | PA-SAM (2024)                                             | 밀집/희소 프롬프트 보상 최적화 및 하드 포인트 마이닝 |
| 프롬프트 > 점 프롬프트        | SAM-Med2D (2023)                                          | 점 프롬프트 상호작용 효율성과 프롬프트 다양성        |
| 프롬프트 > 프롬프트 의존성    | Segment Anything (2023)                                   | 점/박스/텍스트/마스크 4 가지 프롬프트 모드           |

#### 2.6 종합/조사와 리뷰 연구

| 분류      | 논문명                                                         | 분류 근거                                      |
| --------- | -------------------------------------------------------------- | ---------------------------------------------- |
| 종합 조사 | Segment Anything Struggles in Concealed Scenes (2023)          | SAM 의 은폐 장면에서의 취약점 정량 정성 분석   |
| 종합 조사 | Segment Anything Is Not Always Perfect (2023)                  | 다양한 도메인 성능 평가 및 한계 식별           |
| 종합 조사 | Segment Anything for Brain Tumor Segmentation (2023)           | 프롬프트 전략 및 파인튜닝 영향 비교 분석       |
| 종합 조사 | A Federated Learning-Friendly Approach (2024)                  | 연합 학습 컨텍스트에서의 효율성 비교           |
| 종합 조사 | Is SAM 2 Better than SAM in Medical Image Segmentation? (2024) | 의료 영상 양식별 모델 비교 및 프롬프트 전략    |
| 종합 조사 | ROS-SAM (2025)                                                 | 원격 탐사 도메인 성능 분석 및 도메인 적응 평가 |

### 2.7 종합 정리

연구 대상 논문을 체계적으로 분류한 결과, **기초 모델 구축 (7 편)**,**도메인 적응/특수 응용 (31 편)**,**효율화 및 경량화 (13 편)**,**비디오 분할 및 추적 (8 편)**,**프롬프트 엔지니어링 (6 편)**,**종합 조사 (6 편)** 등 6 개의 주요 범주로 구분됩니다. 기초 모델 연구는 SAM 의 제로샷 일반화 능력을 입증하고, 도메인 적응 연구는 의료/비디오/원격탐사 등 다양한 도메인에서의 적용 가능성을 탐구하며, 효율화 연구는 엣지 배포 및 실시간 처리 문제를 해결합니다. 비디오 분할 연구는 시간적 일관성과 추적 능력을 확보하고, 프롬프트 연구는 자동화와 다양한 입력 전략을 탐구합니다.

## 2장. 방법론 분석

### 1. 공통 문제 설정 및 접근 구조

#### 1.1 기본 태스크 정의

모든 논문들이 다루는 공통 태스크는 **프롬프트 기반 세그멘테이션 (Prompt-based Segmentation)**이다. 입력은 다음과 같은 형태를 가지며, 출력은 유효한 객체 분할 마스크이다.

| 입력 유형   | 형태                          | 처리 목적         |
| ----------- | ----------------------------- | ----------------- |
| 시각적 입력 | 이미지 (2D/3D), 비디오 프레임 | 객체 특징 추출    |
| 프롬프트    | 점, 박스, 텍스트, 마스크      | 관심 영역 지정    |
| 도메인 정보 | 자연/의료/항공/위성 등        | 세그멘테이션 대상 |

출력으로는 분할 마스크와 IoU 예측이 제공되며, **제로샷 (Zero-shot)** 또는**전이학습** 방식으로 수행된다.

#### 1.2 일반적인 처리 파이프라인

모든 SAM 기반 모델은 다음 3 단계로 구성된다.

```text
[입력 이미지] → [프롬프트 입력] → [분할 예측]
```

- **이미지 인코딩**: ViT/Hiera 기반으로 고해상도 임베딩 생성
- **프롬프트 인코딩**: 점/박스/텍스트/마스크 임베딩 생성
- **마스크 디코딩**: 교차 어텐션을 통한 마스크 예측 및 IoU 계산

### 2. 방법론 계열 분류

#### 2.1 주요 계열 분류

| 방법론 계열                      | 논문명                                                                                     | 핵심 특징                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| **Zero-shot Segmentation**       | Segment Anything (2023)                                                                    | 대규모 SA-1B 데이터셋으로 사전 훈련, 다양한 프롬프트에서 제로샷 일반화 |
| **Domain Adaptation**            | Medical SAM Adapter (2023), SAM-Med2D (2023), AdaptiveSAM (2024)                           | 도메인 간 격차 해소, 어댑터/PEFT 기법 활용                             |
| **Efficient Variant**            | FastSAM (2023), MobileSAMv2 (2023), TinySAM (2023), EfficientSAM (2023)                    | CNN 기반 또는 경량화로 속도 최적화, 지식 증류/양자화 활용              |
| **3D Extension**                 | Segment Anything in 3D with Radiance Fields (2023), Gaussian Grouping (2023), SAM 2 (2024) | 2D→3D 전이, 래디언스 필드/가우시안 통합, 시공간 전파                   |
| **Video Segmentation**           | Track Anything (2023), Segment and Track Anything (2023), SAM 2 (2024)                     | 시퀀셜 일관성, 메모리 뱅크, 시간적 전파                                |
| **Medical Specialization**       | SA-Med2D-20M Dataset (2023), SAM-Med2D-20M (2023), SAM2 Medical (2024)                     | 의료 영상 데이터셋 구축, 도메인 특화 적응                              |
| **Robustness Enhancement**       | Stable-SAM (2023), RobustSAM (2024), HQ-SAM (2023)                                         | 프롬프트 불안정성 완화, 품질 저하 저항, 고품질 분할                    |
| **Prompt Engineering**           | PA-SAM (2024), AI-SAM (2023), MobileSAMv2 (2023)                                           | 프롬프트 생성 최적화, 하드 포인트 마이닝, 자동 프롬프트                |
| **Multimodal Fusion**            | Segment and Caption Anything (2023), Deep Instruction Tuning (2024)                        | 텍스트 명령, 지역 캡셔닝 통합                                          |
| **Self-Supervised/Unsupervised** | Segment Anything without Supervision (2024), Segment Anything in 3D (2023)                 | 수동 주석 없이 자기 지도 학습, 2D→3D 전이                              |

#### 2.2 계열별 상세 분석

##### 2.2.1 Zero-shot Segmentation 계열

**정의**: 대규모 데이터셋으로 사전 훈련된 파운데이션 모델을 새로운 도메인에서 제로샷으로 적용

**공통 특징**:

- ViT 기반 이미지 인코더 고정
- SA-1B 데이터셋으로 대규모 사전 훈련
- 프롬프트 인코더와 마스크 디코더만 미세 조정
- IoU 기반 최적 마스크 선택

**관련 논문**:

- Segment Anything (2023)
- SAM Struggles in Concealed Scenes (2023)
- Can SAM Count Anything? (2023)
- Segment Anything Is Not Always Perfect (2023)

##### 2.2.2 Domain Adaptation 계열

**정의**: 도메인 간 격차 (자연→의료/위성 등) 를 극복하는 모델 적응

**공통 특징**:

- SAM 인코더 고정 + 어댑터/PEFT 적용
- 3D 공간 정보 통합 (의료 영상)
- 도메인 특화 어댑터 학습
- 파라미터 효율적 미세 조정 (2% 미만)

**관련 논문**:

- Medical SAM Adapter (2023)
- SAM-Med2D (2023)
- AdaptiveSAM (2024)
- SA-Med2D-20M Dataset (2023)

##### 2.2.3 Efficient Variant 계열

**정의**: 계산 비용 절감 및 실시간 배포 가능하도록 모델 경량화

**공통 특징**:

- CNN 기반 백본 사용 (YOLOv8, RepViT 등)
- 지식 증류 (Teacher-Student)
- 양자화/가지치기 압축
- 경량 ViT (TinyViT, EfficientViT)

**관련 논문**:

- FastSAM (2023)
- MobileSAMv2 (2023)
- TinySAM (2023)
- EfficientSAM (2023)
- PTQ4SAM (2024)

##### 2.2.4 3D Extension 계열

**정의**: 2D 분할 지식을 3D/비디오 영역으로 확장

**공통 특징**:

- 래디언스 필드/NeRF/Gaussian Splatting 통합
- 시공간 메모리 뱅크
- 2D→3D 전이 학습
- 프레임 간 전파 전략

**관련 논문**:

- Segment Anything in 3D with Radiance Fields (2023)
- Gaussian Grouping (2023)
- SAM 2 (2024)
- Tracking Anything with Decoupled Video Segmentation (2023)

##### 2.2.5 Video Segmentation 계열

**정의**: 시간적 일관성 있는 다중 프레임 세그멘테이션

**공통 특징**:

- 스트리밍 메모리 아키텍처
- 시퀀셜 프레임 처리
- 객체 추적 모듈 (DeAOT, XMem 등)
- 메모리 뱅크 기반 전파

**관련 논문**:

- Track Anything (2023)
- Segment and Track Anything (2023)
- Tracking Anything in High Quality (2023)
- SAM 2 (2024)

##### 2.2.6 Medical Specialization 계열

**정의**: 의료 영상 특화 세그멘테이션 모델

**공통 특징**:

- 3D 슬라이스 처리
- 다양한 모달리티 (CT/MRI/초음파)
- 2000만 개 의료 마스크 데이터셋
- 의료 영상 정규화 전처리

**관련 논문**:

- Segment Anything for Medical Image Analysis (2023)
- SA-Med2D-20M Dataset (2023)
- SAM-Med2D (2023)
- SAM2 Medical (2024)

##### 2.2.7 Robustness Enhancement 계열

**정의**: 프롬프트 불안정성, 품질 저하, 저조도 등 환경 변화에 강건한 모델

**공통 특징**:

- 어텐션 교정 (DSP, DRP 플러그인)
- 품질 저하 제거 모듈
- 고품질 특징 융합
- 강건한 학습 전략

**관련 논문**:

- Stable-SAM (2023)
- RobustSAM (2024)
- HQ-SAM (2023)

##### 2.2.8 Prompt Engineering 계열

**정의**: 프롬프트 자동 생성 및 최적화 기법

**공통 특징**:

- 하드 포인트 마이닝
- 자동 프롬프트 생성 (AI-Prompter)
- 프롬프트 샘플링 전략
- 조밀/희소 프롬프트 변환

**관련 논문**:

- PA-SAM (2024)
- AI-SAM (2023)
- MobileSAMv2 (2023)

##### 2.2.9 Multimodal Fusion 계열

**정의**: 텍스트 명령과 시각적 입력을 결합한 세그멘테이션

**공통 특징**:

- BERT 텍스트 임베딩
- 시각-언어 통합
- 지역 캡셔닝
- 텍스트 특징 믹서

**관련 논문**:

- Segment and Caption Anything (2023)
- Deep Instruction Tuning (2024)

##### 2.2.10 Self-Supervised/Unsupervised 계열

**정의**: 수동 주석 없이 자기 지도 학습

**공통 특징**:

- Divide-and-Conquer 파이프라인
- 자기 훈련 데이터 증강
- 클러스터링 기반 초기화
- CRF/CascadePSP 정제

**관련 논문**:

- Segment Anything without Supervision (2024)
- Segment Anything in 3D with Radiance Fields (2023)

### 3. 핵심 설계 패턴 분석

#### 3.1 인코더 패턴

| 패턴 유형            | 설명                           | 관련 논문                     |
| -------------------- | ------------------------------ | ----------------------------- |
| ViT 기반 고정 인코더 | MAE 사전 훈련 ViT-H/L/B 고정   | 대부분의 논문                 |
| 경량 ViT 인코더      | TinyViT, EfficientViT 대체     | FastSAM, MobileSAMv2, TinySAM |
| CNN 기반 인코더      | YOLOv8, RepViT 등 CNN 백본     | FastSAM, EdgeSAM, MobileSAMv2 |
| LoRA 어댑터          | 인코더 바이패스 + LoRA 학습    | SAM-Med2D, FLAP-SAM           |
| SVD 미세 조정        | 특이값 분해를 통한 인코더 최적 | S-SAM                         |

#### 3.2 프롬프트 처리 패턴

| 패턴 유형         | 설명                         | 관련 논문           |
| ----------------- | ---------------------------- | ------------------- |
| Sparse 프롬프트   | 점/박스 임베딩               | 기본 SAM 구조       |
| Dense 프롬프트    | 컨볼루션 임베딩              | SAM-Med2D           |
| Text 프롬프트     | CLIP 텍스트 임베딩           | Grounding-DINO 통합 |
| Auto 프롬프트     | AI-Prompter 자동 생성        | AI-SAM, RobustSAM   |
| Hard Point Mining | 고난이도 영역 샘플 집중 추출 | PA-SAM              |

#### 3.3 디코더 패턴

| 패턴 유형          | 설명                  | 관련 논문         |
| ------------------ | --------------------- | ----------------- |
| Transformer 디코더 | 경량 트랜스포머 블록  | 기본 SAM 구조     |
| CNN 기반 디코더    | YOLACT 프로토타입     | FastSAM           |
| Decoupled 디코더   | 공간/시간 분리 전파   | Tracking Anything |
| 2D-3D 통합         | 3D 복셀/가우시안 표현 | 3D SAM 계열       |

#### 3.4 학습 패턴

| 패턴 유형            | 설명                          | 관련 논문                            |
| -------------------- | ----------------------------- | ------------------------------------ |
| Zero-shot 학습       | 사전 훈련 모델 직접 적용      | 기본 SAM 구조                        |
| PEFT                 | LoRA, Adapter 기반 미세 조정  | SAM-Med2D, Medical SAM Adapter       |
| Teacher-Student 증류 | 고모델에서 저모델로 지식 전달 | EfficientSAM, FastSAM                |
| 양자화               | Post-training quantization    | PTQ4SAM, TinySAM                     |
| Self-training        | 자기 지도 데이터 증강         | Segment Anything without Supervision |

#### 3.5 손실 함수 패턴

| 손실 함수         | 용도               | 관련 논문         |
| ----------------- | ------------------ | ----------------- |
| Focal Loss        | 클래스 불균형 처리 | 기본 SAM          |
| Dice Loss         | 오버랩 최적화      | 기본 SAM          |
| MSE Loss          | IoU 예측           | 기본 SAM          |
| Cross-Entropy     | 마스크 분류        | 기본 SAM, Medical |
| Consistency Loss  | 품질 저하 제거     | RobustSAM         |
| Ray-Pair RGB Loss | 경계 정확도        | SANeRF-HQ         |

### 4. 방법론 비교 분석

#### 4.1 도메인 적응 방식 비교

| 적응 전략               | 장점                        | 한계                | 관련 논문             |
| ----------------------- | --------------------------- | ------------------- | --------------------- |
| **Zero-shot**           | 추가 학습 불필요, 빠른 적용 | 도메인 간 성능 저하 | SAM (2023)            |
| **PEFT (LoRA/Adapter)** | 2% 파라미터만 업데이트      | 어댑터 설계 복잡성  | Medical SAM Adapter   |
| **Full Fine-tuning**    | 최대 성능                   | 높은 계산 비용      | SA-Med2D-20M          |
| **지식 증류**           | 효율성 극대화               | Teacher 모델 필요   | EfficientSAM, FastSAM |

#### 4.2 2D vs 3D 확장 비교

| 확장 전략             | 장점                  | 한계              | 관련 논문         |
| --------------------- | --------------------- | ----------------- | ----------------- |
| **래디언스 필드**     | 3D 기하학적 정보 통합 | 고품질 NeRF 필요  | 3D SAM (2023)     |
| **가우시안 스플래팅** | 고품질 실시간 3D      | 정적 장면 제한    | Gaussian Grouping |
| **시공간 전파**       | 메모리 기반 일관성    | 메모리 오버헤드   | SAM 2 (2024)      |
| **2D 슬라이스**       | 단순 처리             | 3D 구조 정보 손실 | SAM-Med3D         |

#### 4.3 효율성 전략 비교

| 전략                       | 가속화율      | 정확도 유지 | 복잡도 | 관련 논문   |
| -------------------------- | ------------- | ----------- | ------ | ----------- |
| **CNN 인코더**             | 50배 감소     | 유사        | 중간   | FastSAM     |
| **TinyViT**                | 60 배 감소    | 유사        | 낮음   | MobileSAMv2 |
| **양자화**                 | 8 bit → 4 bit | 일부 저하   | 낮음   | PTQ4SAM     |
| **Knowledge Distillation** | 16 배 감소    | 유사        | 낮음   | EdgeSAM     |

#### 4.4 프롬프트 효율성 비교

| 전략                      | 프롬프트 수   | 선택 효율 | 관련 논문   |
| ------------------------- | ------------- | --------- | ----------- |
| **Grid Search**           | ~320 개       | 낮음      | 기본 SAM    |
| **Auto Prompt**           | 자동 생성     | 높음      | AI-SAM      |
| **Hard Point Mining**     | 최소          | 높음      | PA-SAM      |
| **Object-aware Sampling** | 320 개 (포화) | 매우 높음 | MobileSAMv2 |

### 5. 방법론 흐름 및 진화

#### 5.1 초기 접근 (2019-2023 초기)

초기 SAM (2023) 은 ViT 기반 이미지 인코더, 프롬프트 인코더, 트랜스포머 디코더로 구성된 기본 아키텍처를 제시했다. 이 시기의 방법론은 다음과 같은 특징을 보였다:

- **대규모 데이터셋 의존**: SA-1B 데이터셋으로 사전 훈련
- **Zero-shot 일반화**: 새로운 도메인에서 직접 적용
- **단일 프레임 처리**: 2D 이미지 중심
- **모호성 인식**: 여러 유효한 마스크 동시 예측

#### 5.2 발전된 구조 (2023 중기)

2023 년 중반 이후 다음과 같은 진화가 이루어졌다:

| 진화 방향       | 구체적 개선              | 대표 논문                   |
| --------------- | ------------------------ | --------------------------- |
| **도메인 적응** | 의료/항공/위성 영상 적용 | Medical SAM Adapter, 3D SAM |
| **효율성**      | CNN/경량화로 속도 개선   | FastSAM, MobileSAMv2        |
| **비디오 확장** | 시공간 일관성 확보       | Track Anything, SAM 2       |
| **강건성**      | 프롬프트/품질 저하 대응  | Stable-SAM, RobustSAM       |
| **3D 전이**     | 2D→3D 확장               | Gaussian Grouping, SAM 2    |

#### 5.3 최근 경향 (2024-2025)

최근 (2024-2025) 에 나타난 주요 경향:

- **SAM 2 아키텍처**: 메모리 어텐션 및 스트리밍 메모리 통합
- **Multimodal 통합**: 텍스트/이미지/비디오 통합
- **의료 영상 특화**: 3D 슬라이스 처리, 다양한 모달리티
- **자동 프롬프트**: AI-Prompter, 자동 생성
- **연합 학습**: FLAP-SAM 등 분산 학습
- **개념 분할**: SAM 3 의 다중 인스턴스 지원

#### 5.4 기술 진화 타임라인

```text
2019
  └─ Towards Segmenting Anything That Moves (움직임 기반 분할)

2023
  ├── Q1~Q2: SAM 원본 제시, 의료/항공 적용
  ├── Q3~Q4: FastSAM, MobileSAM 등 효율성 개선
  └─ 3D 확장, 비디오 분할 기술 발전

2024
  ├── Q1: SAM 2 아키텍처 발표 (메모리 뱅크)
  ├── Q2: 의료 영상 특화 (SAM2 Medical)
  ├── Q3: 효율성 지속 개선 (EdgeSAM, TinySAM)
  └─ Q4: SAM 3 개념 분할 소개

2025
  └─ SAM 3: 개념 기반 다중 인스턴스 분할
```

### 6. 종합 정리

#### 6.1 방법론 지형도

본 문서가 분석한 SAM 관련 방법론들은 크게 **5 가지 축**으로 분류된다:

| 축           | 분류 기준   | 주요 계열                               |
| ------------ | ----------- | --------------------------------------- |
| **도메인**   | 적용 대상   | Zero-shot, 도메인 적응, 의료, 원격 탐사 |
| **효율성**   | 계산 자원   | CNN 기반, 경량화, 양자화, 증류          |
| **차원**     | 데이터 형태 | 2D, 3D, 비디오                          |
| **강건성**   | 환경 조건   | Robust, 고품질, 저조도 대응             |
| **프롬프트** | 입력 방식   | Sparse, Dense, Auto, Multimodal         |

#### 6.2 핵심 통찰

1. **Zero-shot의 한계와 극복**: SAM 의 초기 Zero-shot 접근은 도메인 간 격차에 취약하여 PEFT, 지식 증류, 도메인 특화 어댑터가 등장했다.

2. **2D→3D 진화**: 초기 2D 기반 모델에서 래디언스 필드, 가우시안 스플래팅, 시공간 메모리를 통한 3D/비디오 확장이 이루어졌다.

3. **효율성의 트레이드오프**: CNN 기반, 양자화, 경량화는 속도 개선을 가져오지만 정확도 일부 저하를 수반하며, 최근 모델들은 이 트레이드오프를 완화한다.

4. **프롬프트의 진화**: 초기 수동 프롬프트에서 자동 프롬프트, 텍스트 프롬프트로 발전하며 Multimodal 융합이 이루어졌다.

5. **의료 영상 적응**: 의료 영상은 도메인 격차가 가장 큰 문제이며, 이를 위해 어댑터, LoRA, 3D 슬라이스 처리 등 다양한 기법이 개발되었다.

#### 6.3 방법론 지평

SAM 생태계는 **파운데이션 모델의 제로샷 일반화**를 기반으로 하며, 각 계열은 다음과 같은 문제를 해결한다:

- **Zero-shot 계열**: "모든 도메인에서 즉시 작동하는가?"
- **도메인 적응 계열**: "특정 도메인에 최적화하는가?"
- **효율성 계열**: "실시간 배포 가능한가?"
- **3D/비디오 계열**: "시간/3D 차원 처리하는가?"
- **강건성 계열**: "불리한 환경에서도 작동하는가?"
- **프롬프트 계열**: "효율적이고 자동화된가?"

이러한 축들이 서로 교차하며 다양한 모델 조합이 가능해진다. 예를 들어 **MobileSAMv2**는 효율성 (CNN/경량) + 프롬프트 효율성 (박스 샘플링) 을 결합하고,**SAM 2**는 3D/비디오 + 메모리 어텐션을 통합한다.

#### 6.4 결론

SAM 관련 방법론들은 초기 파운데이션 모델 중심에서 **도메인 적응**,**효율성**,**차원 확장**,**강건성**,**Multimodal 통합** 등 다양한 차원에서 발전했다. 각 계열은 서로 다른 트레이드오프를 관리하며, 응용 목적에 따라 적절한 조합을 선택할 수 있는 풍부한 지형도를 제공한다.

## 3 장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

Segment Anything 관련 연구들에서 공통적으로 관찰되는 평가 구조를 정리한다.

**주요 데이터셋 유형**

| 데이터셋 유형           | 구체적인 예시                                         | 용도                        |
| ----------------------- | ----------------------------------------------------- | --------------------------- |
| 대규모 일반 이미지      | SA-1B(1,100 만 이미지, 11 억 마스크)                  | SAM 사전 학습               |
| 일반 객체 분할 벤치마크 | COCO, LVIS, ADE20K, DIS                               | 인스턴스/의미론적 분할 평가 |
| 의료 영상               | ACDC, BraTS, CholecSeg8k, MoNuSeg, GLAS 등 46 개 이상 | 의료 도메인 적응            |
| 위장/은폐 장면          | CAMO, COD10K, NC4K, 위장 동물/산업 결함/의료 병변     | 도메인 적응 한계 평가       |
| 원격 탐사/항공          | Solar, Inria, DeepGlobe, SpaceNet 2 등 8 개 이상      | 항공 이미지 일반화          |
| 비디오 분할             | DAVIS-2016/2017, YouTube-VOS, BURST, VOTS2023         | 비디오 객체 분할            |
| 3D 장면                 | NVOS, Replica, ScanNet, LERF-Mask                     | 3D 분할/래디언스 필드       |
| 천체 데이터             | 달 (DEM, LROC), 화성 (Mars Express), 포보스           | 천체 탐지/크레이터 탐지     |

**평가 환경 분류**

| 환경 유형          | 구체적 사례                                       |
| ------------------ | ------------------------------------------------- |
| Zero-shot 설정     | SAM 기본 모델, SAM-Med2D, SAM 2 초기 평가         |
| Few-shot 설정      | FSC-147 3-shot/1-shot, 1-10% 데이터 비율 시나리오 |
| Fine-tuning 설정   | MedSAM, Polyp-SAM, RSAM-Seg 등 도메인 적응        |
| Interactive/Oracle | 점/박스 프롬프트, 반복적 프롬프트 시뮬레이션      |

**비교 대상 및 평가 방식**

| 비교 유형         | 대상 모델/메서드                                  |
| ----------------- | ------------------------------------------------- |
| 기존 분할 모델    | U-Net, nnUNet, SegGPT, OneFormer                  |
| SAM 파생 모델     | MobileSAM, FastSAM, EfficientSAM, TinySAM, HQ-SAM |
| 비디오 분할 모델  | XMem, DeAOT, VMT, SAM+XMem++, Cutie               |
| 의료 전용 모델    | MedSAM, SAM-Med2D, SAMed, SAM 2                   |
| Baseline 상호작용 | RITM, SimpleClick, FocalClick, Human Oracle       |

**주요 평가 지표**

| 지표                        | 용도                      |
| --------------------------- | ------------------------- |
| mIoU/mBIoU                  | 평균 IoU, 경계 정밀도     |
| Dice Similarity Coefficient | 의료 영상 분할            |
| AR@K/Box AR                 | 바운딩 박스/마스크 Recall |
| J&F (Joint-and-F1)          | 비디오 객체 분할          |
| MAE/RMSE                    | 카운팅/오차 평가          |
| AUC                         | 추적 성능 평가            |
| F1/F-score                  | 종합 성능 지표            |

### 2. 주요 실험 결과 정렬

핵심 결과를 비교 가능하도록 정리한다.

| 논문명                                                    | 데이터셋/환경                                 | 비교 대상                          | 평가 지표                      | 핵심 결과                                                                               |
| --------------------------------------------------------- | --------------------------------------------- | ---------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------- |
| Segment Anything (2023)                                   | 23 개 세그멘테이션 벤치마크                   | RITM, ViTDet-H, 인간 주석가        | mIoU, AR@1000                  | 23 개 벤치마크에서 mIoU 로 RITM 을 제압, 모호성 인식에서 RITM 능가                      |
| SAM Struggles in Concealed Scenes (2023)                  | CAMO, COD10K, 의료/산업 데이터셋              | CamoFormer-P, HitNet               | $E_{\phi}$, $F_{\beta}$        | COD10K 에서 $E_{\phi}$ 가 ViT-H 대비 13.8% 낮음, CAMO 에서 25.6% 낮음                   |
| SAM on Different Real-world Applications (2023)           | 8 개 벤치마크 (DUTS, COD10K, CDS2K 등)        | VST, ICONet, EDNet, HSNet 등       | MAE, IoU                       | CDS2K 에서 SAM(H) MAE 0.265 vs DGNet 0.089 (17.6% 차이)                                 |
| Deep learning universal crater detection using SAM (2023) | 달·화성·포보스 DEM/위성 이미지                | U-Net, CNN, SVM                    | 탐지 정확도, 오탐/미탐         | 다양한 천체에서 크레이터 탐지 성공, 낮고 해상도에서 작은 크레이터 누락                  |
| Can SAM Count Anything? (2023)                            | FSC-147, MS-COCO                              | GMN, CFOCNet, LaoNet               | MAE, RMSE                      | MS-COCO: MAE 3.87(vs LaoNet 1.73); FSC-147 1-shot: MAE 36.68(vs LaoNet 17.11)           |
| Medical Image Analysis: An Experimental Study (2023)      | 19 개 의료 데이터셋 (28 개 작업)              | RITM, SimpleClick, FocalClick      | IoU, 점프롬프트 오라클         | 평균 0.4595(vs 비교 대상 0.2124), 박스 모드 4 평균 0.6542                               |
| Segment Anything in 3D with Radiance Fields (2023)        | NVOS, SPIn-NeRF, Replica                      | OmniSeg3D, ISRF, MVSeg             | mIoU, 추론 시간                | NVOS 에서 92.2% mIoU(OmniSeg3D 91.7%, ISRF 83.8%), 2 초 추론                            |
| Medical SAM Adapter (2023)                                | BTCV, BraTs, 안저/초음파 등 17 태스크         | vanilla SAM, MedSAM, nnUNet        | Dice, HD95                     | BTCV: vanilla SAM 대비 34.8% Dice 향상, Swin-UNetr 대비 2.9%p                           |
| Track Anything (2023)                                     | DAVIS-2016/2017                               | XMem, Interactive VOS              | J&F                            | DAVIS-2016: 88.4 J&F(클릭 초기화 + 단일 추론), 다중 객체 분리 성공                      |
| Input Augmentation with SAM (2023)                        | Polyp, MoNuSeg, GlaS                          | HSNet, U-Net, P-Net                | Dice, AJI, F-score             | Polyp 에서 HSNet+SAMAug 가 Dice 점수 상승, MoNuSeg 에서 AJI 대폭 개선                   |
| Segment anything, from space? (2023)                      | 8 개 항공 이미지 벤치마크                     | U-Net, RITM                        | IoU, 예측 IoU                  | 도로 클래스에서 IoU < 0.10 으로 완전 실패, 상호작용 시 RITM 대비 50 포인트 차이         |
| HQ-SAM (2023)                                             | 10 개 데이터셋 (8 개 zero-shot)               | SAM, MobileSAM, CRF 후처리         | mIoU, mBIoU, FPS               | 고정밀 데이터셋에서 평균 mIoU 79.5→89.1 개선, mBIoU 71.1→81.8                           |
| Fast Segment Anything (2023)                              | SA-1B, BSDS500, COCO, LVIS                    | SAM-H, ViTDet, YOLOv8-seg          | AP, AR@1000, latency           | SAM-H 대비 50 배 빠름 (40ms vs 2099ms), COCO bbox AR@1000 +1.2                          |
| How to Efficiently Adapt SAM to Medical Images (2023)     | ACDC 심장 MRI (100 환자)                      | 스크래치 UNet, SimCLR UNet         | Dice, ASSD                     | 1 개 레이블 볼륨으로 AutoSAM Dice 39.32(UNet 대비 2 배), LV Dice 0                      |
| MobileSAM (2023)                                          | SA-1B (1% 증류), 모바일 벤치마크              | 원본 SAM, FastSAM, 경량 ViT        | mIoU, 추론 속도, 파라미터      | 9.66M 파라미터 (원본 60 배 작음), 10ms/이미지, mIoU 0.7447                              |
| Polyp-SAM (2023)                                          | 5 개 대장 내시경 데이터셋                     | U-Net, HSNet, Polyp-PVT            | Dice, mIoU                     | CVC-ColonDB 89.4% Dice, 다기관 일반화 성공, ViT-B가 ViT-L 을 능가하는 경우 있음         |
| SAM-PT (2023)                                             | DAVIS, YouTube-VOS, UVO                       | SegGPT, DINO, XMem                 | J&F, 주석 노력                 | DAVIS2017 평균 79.4(SegGPT 75.6), DAVIS2016 84.3, UVO 에서 XMem 능가                    |
| HQTrack (2023)                                            | VOTS2023, DAVIS, VIPSeg                       | DeAOT, VMOS, SAM                   | AUC, IoU                       | VOTS2023 종합 2 위, 공동 추적 AUC 0.566 vs 개별 0.552, 기본 대비 3.9%p ↑                |
| AdaptiveSAM (2023)                                        | Endovis17/18, Cholec-Seg8k, 초음파            | UNet, MedSAM, AutoSAM              | Dice, IoU, 평균 정밀도/재현율  | EV17: SAM-ZS 대비 68% Dice 향상, 희귀 클래스에서 개별 성능 평가                         |
| SAM-Med2D (2023)                                          | 460 만 의료 영상, 9 개 MICCAI 검증            | SAM, FT-SAM                        | Dice                           | 바운딩 박스: 79.30% (SAM 61.63%, FT-SAM 73.56% 대비), 1 점 프롬프트: 70.01%(SAM 18.94%) |
| Decoupled Video Segmentation (2023)                       | VIPSeg, BURST, DAVIS, UVO                     | 종단 간 모델                       | VPQ, J&F                       | 희귀 클래스에서 종단 간 대비 60% VPQ 개선, 클립 내 합의로 시간적 일관성 확보            |
| SAM for Brain Tumor Segmentation (2023)                   | BraTS2019 (4 클래스)                          | 다양한 프롬프트 전략               | Dice, HD, ASSD                 | 10 점 프롬프트가 2 점 우수, 1 박스 + 1 점(0.8029)이 1 박스(0.7823)보다 우수             |
| Gaussian Grouping (2023)                                  | LERF-Mask, Replica, ScanNet                   | LERF, SA3D, Panoptic Lifting       | mIoU, mBIoU, FPS               | LERF-Mask 에서 2 배 성능, Replica 에서 140 FPS(vs 10 FPS)                               |
| EfficientSAM (2023)                                       | ImageNet, COCO, ADE20K                        | SAM ViT-H, FastSAM/MobileSAM       | Top-1 accuracy, AP, mIoU       | SAM-T 대비 ~20 배 작고 ~20 배 빠름, COCO 대비 6.5 AP 향상                               |
| Segment and Caption Anything (2023)                       | Visual Genome                                 | GRiT, SAM+Image Captioner          | CIDEr-D, METEOR, SPICE         | GRiT 대비 CIDEr-D 149.8 달성, LLAMA-3B 디코더에서 최적                                  |
| SANeRF-HQ (2023)                                          | Mip-NeRF 360, LERF, LLFF                      | SA3D, ISRF, SAN                    | mIoU, Acc, 경계 정확도         | Mip-NeRF 360 에서 mIoU 91.0%, 모든 데이터셋에서 압도                                    |
| AI-SAM (2023)                                             | ACDC, Synapse, 위장, 그림자 감지              | MedSAM, SimpleClick, AdaptiveClick | DICE, PCM, OCM                 | ACDC 에서 바운딩 박스 프롬프트 사용 시 SOTA, AI-Prompter 점수 92.06%→64.05%             |
| EdgeSAM (2023)                                            | SA-1K, COCO, iPhone 14, 2080Ti                | SAM, MobileSAM, EfficientSAM       | FPS, GFLOPs, mIoU              | iPhone 14 에서 38.7 FPS (최초의 30 FPS 이상), COCO mIoU +2.3                            |
| MobileSAMv2 (2023)                                        | LVIS, SA-1B 하위 집합                         | SAM, MobileSAM, FastSAM            | mask AR@K, 처리 시간           | `AR@100`: 42.5% vs 38.9%, 6464ms → 97ms, 16 배 속도 향상                                |
| TinySAM (2023)                                            | COCO, LVIS, DOORS 등                          | FastSAM, MobileSAM, Q-TinySAM      | AP, mIoU, MACs, 지연           | FastSAM 대비 AP+4%, MobileSAM MACs 동일시 AP+1.3%, MACs -9.5%, 지연 -25%                |
| SA-Med2D-20M Dataset (2023)                               | 460 만 영상, 1,970 만 마스크                  | U-Net, nnU-Net, SAM                | 데이터셋 공개                  | 219 개 고유 레이블, CT/MR 압도적 비중, 긴 꼬리 문제 존재                                |
| Stable Segment Anything Model (2023)                      | MS COCO, SGinW, 노이즈 박스                   | SAM, DT-SAM, PT-SAM, HQ-SAM        | mask mIoU, mBIoU, mSF          | 노이즈 박스에서 SAM 급격히 하락 vs Stable-SAM 일관된 최고 성능                          |
| RSAM-Seg (2024)                                           | DeepGlobe, 클라우드/들판/건물/도로            | 원본 SAM, U-Net, DeepLab 계열      | F1, 전체 정확도                | 클라우드 시나리오 36.7% 향상, 10% 데이터로 U-Net 수준, Ground Truth 누락 영역 식별      |
| PA-SAM (2024)                                             | HQSeg-44K, COCO, SegInW                       | HQ-SAM, BOFT-SAM                   | mIoU, mBIoU, AP                | HQSeg-44K 에서 mIoU 2.1%, mBIoU 2.7% 향상, 제로샷 COCO AP +0.4%                         |
| DIT-SAM (2024)                                            | RefCOCO, RefCOCO+, RefCOCOg                   | 기본 SAM, E-DIT                    | oIoU, mIoU, P@K                | L-DIT 74.73% mIoU(E-DIT 71.46% 대비), 69.97% oIoU(기본 57.98% 대비)                     |
| Moving Object Segmentation (2024)                         | DAVIS16/17, YTVOS, MoCA                       | 기존 VOS 방법들                    | fIoU, MOS                      | 10% 이상 능가하며 최첨단 성능, FlowI-SAM+FlowP-SAM 결합 시 추가 향상                    |
| RobustSAM (2024)                                          | Robust-Seg, MSRA10K, LVIS 등                  | SAM, HQ-SAM, AirNet+SAM            | IoU, PA, Dice                  | 모든 품질 저하 시나리오에서 최상위 성능, 15 가지 유형에서 견고성 입증                   |
| Without Supervision (2024)                                | 7 개 데이터셋 (COCO, LVIS 등)                 | CutLER, U2Seg, SOHES, SAM          | AR, AP, MaxIoU                 | 전체 이미지 분할에서 SAM 대비 평균 AR 11.0% 향상, small object AR +16.2%                |
| PTQ4SAM (2024)                                            | MS-COCO, ADE20K, DOTA-v1.0                    | MinMax, OMSE, QDrop 등             | AP, mIoU, mAP, 가속비          | W6A6 에서 무손실, W4A4 에서 QDrop 대비 5.1~6.3% 우위, 3.9 배 가속, 4.9 배 저장          |
| Federated Learning-Friendly PEFT (2024)                   | Fed-KiTS2019, Fed-IXI, Prostate MRI           | FullFT, AttnFT, SAMed              | Dice, 통신 비용, 파라미터      | FullFT 대비 ~48x 통신 비용 감소, Dice ~6% 향상, ~2.8x 파라미터 효율성                   |
| SAM 2 (2024)                                              | 9 개 프롬프트 비디오, 17 개 VOS, 37 개 제로샷 | SAM+XMem++, Cutie, XMem++          | mIoU, J&F, FPS                 | 이미지: 58.9 mIoU, VOS: 76.8/77.0 J&F(기존 약 60 압도), 6 배 빠름                       |
| SAM2 Medical Images and Videos (2024)                     | 11 개 모달리티 공공 데이터셋                  | SAM1, MedSAM, SAM-Med3D            | Dice, NSD                      | SAM2-Base 3D CT/MRI 현저히 개선, 3D 초음파 0.8537 DSC, 전이 학습 후 DSC 3.5~45.62%      |
| Is SAM 2 Better than SAM? (2024)                          | 11 개 의료 데이터셋 (24 개 장기-양식)         | SAM                                | DICE                           | CT/초음파에서 저조, MRI 일부 동등/약간 더 나은, 음성 프롬프트 추가 시 SAM 2 더 큰 향상  |
| S-SAM (2024)                                              | CholecSeg8k, 초음파, ChestXDet 등 5 개        | Zero-shot, MedSAM, LoRA 등         | Dice, 매개변수 비율            | SOTA 대비 6-7% 향상, SAM 대비 99.6%, LoRA 대비 50% 적은 매개변수                        |
| SAM2 Survey (2024)                                        | 21 개 의료 데이터셋, 17 개 VOS                | SAM 기반 모델, SOTA                | Dice, NSD, J&F, J, F           | 3D 분할에서 전략적 활용 중요, 2D 대비 3D 성능 격차, 실시간성 저하                       |
| On Efficient Variants Survey (2024)                       | COCO, LVIS, UVO, SGinW                        | EfficientViT-SAM, NanoSAM          | #Params, FLOPs, EER, FPS, mIoU | EfficientViT-SAM-L0: GPU 30x, CPU 50x 빠름, NanoSAM: Jetson Nano 최고 처리량            |
| ROS-SAM (2025)                                            | SAT-MTB, iSAID, NPWS 등                       | SAM, SAM2, HQ-SAM                  | IoU, BIoU                      | SAT-MTB: 50.54%(SAM 대비 13% 향상), iSAID 73.22%, 제로샷 성능                           |
| SAM 3 (2025)                                              | SA-Co, ODinW13, LVIS, CountBench 등           | OWLv2, GLEE, T-Rex2                | cgF₁, pHOTA, MAE               | LVIS AP 48.8(vs 38.5), SA-Co/Gold cgF₁ 74%(인간 수준), CountBench MAE 0.12              |
| Towards Segmenting Anything That Moves (2019)             | FBMS, DAVIS-Moving, YTVOS-Moving              | SOTA 방법, 2-스트림 기법           | F-measure, 오탐지 패널티       | FBMS: 6.4% 개선, DAVIS-Moving: 77.9%(기존 42.3%), YTVOS: 완만한 향상                    |

### 3. 성능 패턴 및 경향 분석

여러 논문을 묶어서 분석한다.

**가) 공통적으로 나타나는 성능 개선 패턴**

1. **프롬프트 모드의 영향**
    - 박스 프롬프트가 점 프롬프트보다 일반적으로 성능이 높음 (의료 영상: 평균 0.6542 vs 0.1136~0.9118)
    - 음성 프롬프트 추가 시 성능 향상 (특히 SAM 2: 경계선 유실 보완, 일반화 능력 향상)
    - 반복적 프롬프트는 두 모델 모두에서 IoU 개선

2. **경량화 모델의 효율성 유지**
    - 경량화 모델 (MobileSAM, FastSAM, EfficientSAM, TinySAM) 이 원본 SAM 대비 mIoU 0.71~0.74 유지하면서 20~50 배 속도 향상
    - EfficientViT-SAM-L0: GPU 30x, CPU 50x 빠름
    - MobileSAMv2: 16 배 속도 향상, AR@100 성능 동등

3. **의료 도메인 적응**
    - 어댑터 기반 미세 조정 (SAM-Med2D, MedSAM, Polyp-SAM) 이 원본 SAM 대비 17~35%p Dice 향상
    - 2~4% 파라미터 업데이트만으로 SOTA 달성 (SAM-Med2D: 17.67%p, Medical SAM Adapter: 34.8%)
    - LoRA 기반 적응 (SAM-Med2D: 랭크 4 최적, SAM 2: 메모리 뱅크 활용)

4. **도메인 적응 데이터의 영향**
    - 의료 데이터셋 (SA-Med2D-20M: 1,970 만 마스크) 이 자연 영상 대비 도메인 격차 극복
    - few-shot 설정 (1~10% 데이터) 에서도 기존 모델 대비 우수한 성능 (RSAM-Seg: 10% 데이터로 U-Net 수준)
    - 도메인 특화 사전 지식 (임베딩/고주파) 이 자동화 및 성능 개선

**나) 특정 조건에서만 성능이 향상되는 경우**

| 조건                      | 성능 향상 관찰          | 근거                                       |
| ------------------------- | ----------------------- | ------------------------------------------ |
| 저대비 영상 (CCT, 초음파) | 음성 프롬프트 추가 시   | SAM 2: 음성 프롬프트 추가 시 향상 폭 더 큼 |
| 밀집 객체/작은 객체       | 10 점 프롬프트          | 2 점보다 우수, 20/30 점 성능 감소          |
| 의료 영상 점 프롬프트     | SAM-Med2D               | 1 점 프롬프트에서 SAM 대비 압도적 우위     |
| 3D 분할                   | 양방향 전파             | 평균 IoU 0.0874→0.2383(266% 향상)          |
| 엣지 디바이스             | 프롬프트-인-더-루프 KD  | 인코더-온리 KD 대비 성능 향상              |
| 긴 비디오                 | 재초기화 ($h=8$ 프레임) | 추적기 오류 복구 효과 확인                 |

**다) 논문 간 상충되는 결과**

1. **SAM 2 vs SAM**
    - 의료 영상 2D 분할: SAM 2 가 일관되게 저조 (CT/초음파), MRI 일부 동등/약간 더 나은
    - 3D CT/MRI: SAM 2 가 현저히 개선, 3D 초음파 0.8537 DSC

2. **ViT-B vs ViT-L/H (의료 영상)**
    - Polyp-SAM: ViT-B 가 ViT-L 을 능가하는 데이터셋 존재
    - 일반 이미지: ViT-B→L→H 확장 시 성능 포화 경향

3. **Box vs Point 프롬프트 (의료)**
    - 박스 프롬프트가 점 프롬프트보다 우수 (의료 영상)
    - 제로샷 이미지 분할에서 점 프롬프트 58.9 mIoU vs Box 더 높음

**라) 데이터셋 또는 환경에 따른 성능 차이**

| 환경                  | SAM 의 성능 변화 | 설명                        |
| --------------------- | ---------------- | --------------------------- |
| 은폐 장면 (위장/병변) | 13.8~25.6% 격차  | COS 모델 대비 현저히 저조   |
| 항공 이미지 (도로)    | IoU < 0.10       | 인스턴스 분할 부적절        |
| 천체 이미지           | 다양한 성공      | 해상도/천체별 편차          |
| 의료 영상             | 도메인 격차 존재 | 17~35%p Dice 격차 (적응 전) |
| 저품질 이미지         | SAM 급격히 하락  | 노이즈 박스 등              |

### 4. 추가 실험 및 검증 패턴

여러 논문을 묶어 공통 실험 패턴을 분석한다.

**가) ablation study**

1. **구성 요소별 기여도 분석**
    - RobustSAM: 각 모듈 (AMFG, AOTG, ROT) 기여도 확인
    - PA-SAM: 조밀/희소 프롬프트, Gumbel top-k point sampler 효과
    - HQ-SAM: HQ-Output Token, 3 계층 MLP, 글로벌-로컬 특징 융합

2. **프롬프트 전략**
    - SAM for Brain Tumor: 10 점 > 2 점, 1 박스 + 1 점 > 1 박스
    - AdaptiveSAM: 희귀 클래스 개별 성능 평가
    - SAM-Med2D: 1 점 vs 바운딩 박스 비교

**나) 민감도 분석**

1. **프롬프트 품질 영향**
    - Stable-SAM: 노이즈 박스, 1/3/10 포인트 프롬프트 영향도 분석
    - SAM-Med2D: 1 점 프롬프트가 바운딩 박스보다 압도적

2. **데이터 양 영향**
    - RSAM-Seg: 1%~70% 데이터 비율별 성능 곡선
    - AutoSAM: 레이블 10 개 미만 시 우위, 10 개 초과 시 이점 소멸

**다) 조건 변화 실험**

1. **백본 크기 비교**
    - Medical SAM Adapter: ViT-B vs ViT-L vs ViT-H
    - EfficientSAM: ViT-Tiny/Small/Base 학생 모델

2. **적응 전략 비교**
    - SAM-Med2D: FT-SAM vs 어댑터 삽입 (어댑터 더 효율적)
    - EdgeSAM: 디코더 프리징 vs 미세조정 (미세조정 더 좋음)

### 5. 실험 설계의 한계 및 비교상의 주의점

여러 논문을 묶어 분석한다.

**가) 비교 조건의 불일치**

| 불일치 유형    | 구체적 사례                   | 영향               |
| -------------- | ----------------------------- | ------------------ |
| 평가 지표 차이 | mIoU vs Dice vs MAE           | 수직 비교 어려움   |
| 프롬프트 설정  | 점/박스/Everything 모드       | 공정성 문제        |
| 데이터셋 편향  | 자연 이미지 중심 vs 의료 영상 | 도메인 격차        |
| 평가 환경      | Zero-shot vs Fine-tuning      | 실제 성능 과소평가 |

**나) 데이터셋 의존성**

- SA-1B 의 자연 이미지 훈련으로 의료/항공/위장 장면에서 성능 저하
- 의료 데이터셋 (SA-Med2D-20M) 이 자연 영상 대비 도메인 격차 극복
- 460 만 영상 1,970 만 마스크에서 CT/MR 압도적 비중

**다) 일반화 한계**

- 제로샷 설정에서 전문 분야 미세 분류 일반화 저조 (SAM 3)
- 단일 프롬프트로 시맨틱/파놉틱 분할 어려움
- 3D 분할에서 2D 대비 성능 격차 존재

**라) 평가 지표의 한계**

- mIoU 만으로는 경계 품질 평가 불가 (mBIoU 필요)
- Dice 만으로는 경계 정밀도 고려 안됨 (HD95 필요)
- 비디오 분할에서 J&F 는 지역 정확도+전체 정확도만 평가

### 6. 결과 해석의 경향

여러 논문을 묶어서 저자들의 해석 경향을 분석한다.

**가) 공통 해석 경향**

1. **파운데이션 모델로서의 평가**
    - "GPT-3 for CV"로 평가 (SAM 기본 논문)
    - Zero-shot 전이 능력을 강조
    - 프롬프트 엔지니어링만으로도 다양한 태스크 적응

2. **제한점 인정**
    - 저대비/작은/전문적 객체에서 성능 저조
    - 도메인 적응 필요 (의료/산업)
    - 프롬프트 품질 의존성

3. **적응 전략 제안**
    - 어댑터/LoRA/PEFT 를 통한 효율적 적응
    - 도메인 특화 사전 지식 통합
    - 작은 파라미터 업데이트로 SOTA 달성

**나) 실제 관찰 결과와의 구분**

| 해석                     | 실제 결과                                      |
| ------------------------ | ---------------------------------------------- |
| "SAM 은 파운데이션 모델" | 23 개 벤치마크에서 RITM 제압, 제로샷 성능 입증 |
| "의료 영상에서 한계"     | 17~35%p Dice 격차, 적응 필요                   |
| "경량화 모델 효율성"     | 20~50 배 속도 향상, mIoU 유지                  |
| "음성 프롬프트 효과"     | 경계선 유실 보완, 일반화 능력 향상             |

### 7. 종합 정리

Segment Anything 관련 실험 결과 전체를 종합하면, SAM 이 방대한 SA-1B(11 억 마스크) 로 훈련된 파운데이션 모델로서 제로샷 분할 성능을 입증했으나, 특정 도메인 (의료, 은폐 장면, 저대비 영상) 에서는 도메인 격차가 존재하고 이를 위해 적응 전략 (어댑터, LoRA, 도메인 특화 데이터) 이 필수임이 확인됨. 경량화 모델 (MobileSAM, FastSAM, EfficientSAM) 이 속도 20~50 배 향상하면서 mIoU 0.71~0.74 를 유지해 모바일/엣지 배포가 가능해졌고, 프롬프트 모드 (박스 > 점 > Everything) 와 음성 프롬프트, 반복적 프롬프트 등이 성능에 중요한 영향을 미침. 의료 영상에서는 어댑터 기반 미세 조정이 2~4% 파라미터로 17~35%p Dice 향상을 달성하고, 비디오 분할에서는 SAM 2 의 시간적 일관성 처리 능력이 기존 VOS 모델 대비 우수하나 실시간 적용에는 제한이 있음. 최종적으로 SAM 은 범용 분할 모델로서 강력한 제로샷 능력을 보유하지만, 특정 도메인/작업에서는 추가 적응과 최적화가 필요하며, 효율성과 성능 사이 균형이 연구자들의 주요 관심사임.
