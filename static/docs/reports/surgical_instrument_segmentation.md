# 서론

## 1. 연구 배경

수술 도구 분할 (Surgical Instrument Segmentation) 은 내시경 영상을 기반으로 수술 도구를 정밀하게 식별하고 세분화하는 컴퓨터 비전 태스크이다. 본 보고서는 2016 년부터 2025 년까지 발표된 총 38 편의 논문을 대상으로 분석하며, 데이터 효율성 (합성 데이터 생성), 지도/비지도/메타 학습 방법론, 시간적 정보 통합, 비전-언어 모델 융합, 파운데이션 모델 활용, 실시간 효율성 최적화, specializied 태스크, 응용 연구 등 11 가지 주요 범주로 연구 체계를 분류하였다. 특히 2017 년 ResNet 기반 초기 접근부터 2024 년 SAM 기반 제로샷 적응까지 8 년 간의 기술 진화 흐름을 포괄적으로 검토한다.

## 2. 문제의식 및 분석 필요성

수술 도구 분할 연구들은 각자의 방법론적 관점에서 독립적으로 발전해 왔으나, 다음과 같은 체계적 정리가 필요한 문제의식이 존재한다.

첫째, 연구들의 공통 문제 정의와 접근 구조가 명확히 정립되지 않은 상태이다. 내시경 영상에서 도구 분할을 위한 3 단계 파이프라인 (데이터 전처리→특징 처리→분할 예측) 과 주요 계열 (Multi-Angle, Attention-based, Temporal Flow, Instance-based, SAM 기반 등 27 계열) 은 존재하나, 방법론 간의 구조적 차이와 트레이드오프 관계를 체계화한 비교 분석이 부족하다.

둘째, 평가 지표와 성능 결과의 해석에 일관성이 없는 상황에서, IoU/mIoU 중심 평가 (주요 지표), Dice/DSC 보조 지표, FPS 처리 속도, FLOPs 계산 효율성 등 다양한 지표를 혼용한 상태이며, 교차 데이터셋 일반화 시 13.8%p~25%p 하락 등 데이터셋 의존성 문제가 지속적 관찰된다.

셋째, 실시간 처리 (30 FPS 기준), 실제 임상 환경 일반화 (EndoVis→RoboTool 도메인 격차), 완전 비지도 학습 (83.77% vs 지도 89.61%) 등 실용적 제약 조건에서의 성능 한계에 대한 종합적 평가가 결여되어 있다.

## 3. 보고서의 분석 관점

본 보고서는 다음과 같은 3 차원적 관점에서 문헌을 정리한다.

| 분석 축       | 주요 내용                                                                                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 연구체계 분류 | 데이터 효율성, 지도/비지도/메타 학습, 시간적 통합, 비전-언어 모델, foundation 모델, 실시간 최적화, specialized 태스크, 응용 연구 등 11 범주별 체계적 분류 |
| 방법론 분석   | 27 계열의 기술적 구현, 모듈 분리/단계적 처리/학습 방식 패턴, 계열별 상세 분석, 방법론 지형 지도 (데이터 의존성/시간 정보/적응 방식 3 축)                  |
| 실험결과 분석 | 주요 논문별 성능 정렬, 성능 패턴 (일반화, 실시간, 효율성) 분석, ablation study 및 조건 변화 실험 패턴, 결과 해석의 경향과 한계점                          |

## 4. 보고서 구성

본 보고서는 총 3 장으로 구성되어 있다.

- **1 장 (연구체계 분류)**: 제공된 38 편의 논문을 문제 정의, 접근 방식, 시스템 구조 3 개 기준에서 11 가지 범주로 분류하며, 초기 CNN 기반 접근부터 최근 SAM/Transformer 기반 적응까지의 진화 흐름과 각 계열별 핵심 기술을 정리한다.

- **2 장 (방법론 분석)**: 모든 연구의 공통 문제 설정 (입력, 처리, 출력 3 단계) 과 27 계열 방법론의 상세 분석을 수행한다. 모듈 분리 (Encoder/Decoder/Attention/Memory/Loss), 단계적 처리 (2~3 단계 아키텍처), 학습/데이터 활용 패턴을 비교하고, 방법론 간 트레이드오프와 진화 추세 (수동 주석→프롬프트→제로샷, CNN→Transformer→SAM) 를 체계화한다.

- **3 장 (실험결과 분석)**: 12 개 주요 데이터셋과 9 가지 평가 지표를 기준으로 성능 비교를 수행한다. 주요 논문별 정량적 결과 정리, 성능 패턴 (데이터셋 의존성, 학습 전략, 실시간 처리, 효율성), ablation study 및 민감도 분석 패턴을 검증하며, 결과 해석의 경향과 데이터셋 의존성, 도메인 격차, 평가 지표 한계 등 연구의 한계점을 종합적으로 논의한다.

## 1 장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 분류 체계는 제공된 논문 요약문을 **문제 정의 (데이터 희소성, 시간적 일관성, 실시간성)**,**접근 방식 (지도/비지도/메타 학습)**,**시스템 구조 (단일 프레임/시간적 통합/다중모달)** 관점에서 분석하여 수립하였다. 각 논문은 가장 지배적인 연구 관점을 중심으로 대분류와 하위 범주로 배치되었으며, 일부 논문은 여러 범주에 동시에 해당할 수 있으나 대표성 있는 1 개 범주에만 배정하였다.

### 2. 연구 분류 체계

#### 2.1 데이터 효율성 및 합성 기반 접근

데이터 수집 비용 감소, 합성 데이터 생성, 도메인 일반화를 목적으로 하는 연구들이다.

| 분류                                                   | 논문명                                                                                            | 분류 근거                                                                     |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 데이터 효율성 및 합성 기반 접근 > 합성 데이터 생성     | Image Compositing for Segmentation of Surgical Tools without Manual Annotations (2021)            | 크로마 키 합성으로 반합성 데이터를 실시간 생성하는 데이터 생성/증강 관점 연구 |
| 데이터 효율성 및 합성 기반 접근 > 합성 데이터 생성     | Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need (2022)        | 단일 배경과 소수 전경 이미지를 증강/블렌딩하여 합성 데이터셋 생성             |
| 데이터 효율성 및 합성 기반 접근 > 데이터 효율성 일반화 | SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation (2023) | SAM 의 수술 도메인 일반화 성능과 프롬프트 의존성을 실증 평가한 연구           |

#### 2.2 지도 학습 기반 방법론

주석 데이터를 활용한 전통적인 지도 학습 프레임워크를 개선한 연구들이다.

| 분류                                                 | 논문명                                                                                                                               | 분류 근거                                                                          |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| 지도 학습 기반 방법론 > 회전 불변성 및 경계면 정밀화 | Towards Better Surgical Instrument Segmentation in Endoscopic Vision: Multi-Angle Feature Aggregation and Contour Supervision (2020) | 다각도 특징 집계와 윤곽선 감독으로 회전 불변성과 경계면 분할 정밀화                |
| 지도 학습 기반 방법론 > 전역 컨텍스트 및 다중 스케일 | BARNet: Bilinear Attention Network with Adaptive Receptive Fields for Surgical Instrument Segmentation (2020)                        | 2 차 통계 전역 컨텍스트와 다중 스케일 특징을 통한 조명/스케일 변화 강건성          |
| 지도 학습 기반 방법론 > 어텐션 기반 특징 융합        | RASNet: Segmentation for Tracking Surgical Instruments in Surgical Videos Using Refined Attention Segmentation Network (2019)        | 어텐션 기반 특징 융합과 전이 학습으로 정밀 분할 및 분류 동시 수행                  |
| 지도 학습 기반 방법론 > 어텐션 기반 특징 융합        | RAUNet: Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments (2019)                                   | 백내장 수술 환경의 반사와 클래스 불균형 해결을 위한 어텐션 모듈과 하이브리드 손실  |
| 지도 학습 기반 방법론 > 경량 어텐션 네트워크         | Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments (2019)                               | MobileNetV2 인코더와 경량 디코더, 어텐션 퓨전 블록 통합으로 실시간 성능 달성       |
| 지도 학습 기반 방법론 > residual learning            | Deep Residual Learning for Instrument Segmentation in Robotic Surgery (2017)                                                         | ResNet 기반 FCN 구조로 high-resolution_dense prediction 및 binary→multi-class 확장 |
| 지도 학습 기반 방법론 > 사전 학습 인코더 활용        | Automatic Instrument Segmentation in Robot-Assisted Surgery Using Deep Learning (2018)                                               | pretrained encoder (VGG/ResNet)를 활용한 U-Net/TernausNet 의 실용성 입증           |
| 지도 학습 기반 방법론 > 시맨틱 인스턴스 분할         | ISINet: An Instance-Based Approach for Surgical Instrument Segmentation (2020)                                                       | 시계열 인스턴스 추적과 시간적 사전지식 활용으로 공간/시간 일관성 확보              |
| 지도 학습 기반 방법론 > 실시간 인스턴스 분할         | Real-time Instance Segmentation of Surgical Instruments using Attention and Multi-scale Feature Fusion (2021)                        | YOLACT++ 단일 단계 아키텍처에 MSFF 와 CBAM 통합한 실시간 분할                      |
| 지도 학습 기반 방법론 > 인스턴스 세분화 프레임워크   | From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments (2022)                                      | 3 단계 아키텍처에 분류 모듈 도입, 마스크 어텐션과 메트릭 학습 (arc loss)           |

#### 2.3 비지도 및 반지도 학습 기반 방법론

수동 주석 의존성 해소를 위한 비지도/반지도 학습 연구들이다.

| 분류                                                               | 논문명                                                                                               | 분류 근거                                                                    |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 비지도/반지도 학습 기반 방법론 > 준지도 모션 플로우                | Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video (2020) | 모션 플로우 기반 준지도 데이터 증강으로 어노테이션 비용 감소                 |
| 비지도/반지도 학습 기반 방법론 > 앵커 생성 및 확산                 | Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion (2020)    | 수작업 큐로 의사 레이블 생성, 의미론적 확산 손실로 점진적 해결               |
| 비지도/반지도 학습 기반 방법론 > 완전 비지도 Teacher-Proxy-Student | FUN-SIS: a Fully UNsupervised approach for Surgical Instrument Segmentation (2022)                   | 생성적 적대 학습과 노이즈 특성 분석을 통한 완전 비지도 Teacher-Proxy-Student |

#### 2.4 시간적 정보 통합 및 메모리 기반 방법론

비디오 시퀀스의 시간적 일관성을 활용하거나 메모리 메커니즘을 활용한 연구들이다.

| 분류                                           | 논문명                                                                                               | 분류 근거                                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 시간적 정보 통합 > 모션 플로우 기반 프레임워크 | Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video (2020) | 듀얼 모션 기반 프레임워크로 비디오 시퀀스 시간적 일관성 학습          |
| 시간적 정보 통합 > 시계열 인스턴스 추적        | ISINet: An Instance-Based Approach for Surgical Instrument Segmentation (2020)                       | FlowNet2 와 IoU 기반 시간적 추적, 6 프레임 정보 활용                  |
| 시간적 정보 통합 > 의미론적 확산               | Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion (2020)    | 프레임 간 특징 유사성을 통한 모호 영역 점진적 확산 해결               |
| 시간적 정보 통합 > 듀얼 메모리 아키텍처        | Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video (2021) | ELA 와 AGA 를 통한 지역/전역 시공간 정보 통합, 능동적 메모리 업데이트 |
| 시간적 정보 통합 > temporal consistency module | MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation (2024)                     | MViT 기반 temporal consistency module로 비디오 문맥 통합              |

#### 2.5 메타 학습 및 도메인 적응

새로운 도메인에 빠르게 적응하는 메타 학습/도메인 적응 연구들이다.

| 분류                                              | 논문명                                                                                                                         | 분류 근거                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| 메타 학습 및 도메인 적응 > 온라인 적응 프레임워크 | One to Many: Adaptive Instrument Segmentation via Meta Learning and Dynamic Online Adaptation in Robotic Surgical Video (2021) | MAML 기반 오프라인 훈련과 동적 온라인 적응, 노이즈 인식 그래디언트 게이트 |

#### 2.6 비전-언어 모델 및 다중모달 융합

텍스트/음성 등 비전 외 정보를 활용한 분할 연구들이다.

| 분류                                                                 | 논문명                                                                                                                            | 분류 근거                                                                       |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 비전-언어 모델 및 다중모달 융합 > 텍스트 프롬프트 분할               | Text Promptable Surgical Instrument Segmentation with Vision-Language Models (2023)                                               | CLIP 기반 VLM 을 이용한 텍스트 프롬프트로 새로운 도구에 대한 적응적 분할        |
| 비전-언어 모델 및 다중모달 융합 > surgical visual question answering | Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery (2023) | GVLE-LViT 로 이종 모달리티 효율 융합 및 위치 기반 답변 성능 향상                |
| 비전-언어 모델 및 다중모달 융합 > 음성 의도 이해                     | ASI-Seg: Audio-Driven Surgical Instrument Segmentation with Surgeon Intention Understanding (2024)                                | 음성 명령을 intention 신호로 삼아 다중모달 융합과 대조학습 프롬프트 인코더 결합 |

#### 2.7 파운데이션 모델 및 SAM 기반 연구

Segment Anything Model 이나 파운데이션 모델 활용 연구들이다.

| 분류                                                  | 논문명                                                                                            | 분류 근거                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 파운데이션 모델 및 SAM 기반 연구 > SAM 실증 평가      | SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation (2023) | SAM 의 수술 도메인 적용 가능성과 일반화 한계를 다각도로 평가한 실증 연구          |
| 파운데이션 모델 및 SAM 기반 연구 > SurgicalSAM 튜닝   | SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation (2023)                   | 프로토타입 기반 클래스 프롬프트 인코더를 도입한 SAM 의 효율적 튜닝                |
| 파운데이션 모델 및 SAM 기반 연구 > SAM2 실시간 최적화 | Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning (2024)    | cosine similarity 기반 동적 프레임 제거로 SAM2 의 실시간 처리 및 메모리 효율 개선 |

#### 2.8 실시간 및 효율성 최적화

실시간 처리 속도와 계산 효율성 확보에 중점을 둔 연구들이다.

| 분류                                              | 논문명                                                                                                        | 분류 근거                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 실시간 및 효율성 최적화 > 경량화 모델             | Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments (2019)        | MobileNetV2 인코더와 경량 디코더, 어텐션 퓨전 블록으로 39.49 fps 달성       |
| 실시간 및 효율성 최적화 > ELA/AGA 메모리 통합     | Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video (2021)          | ELA 와 AGA 를 통한 지역/전역 정보 통합으로 EndoVis17/18 에서 평균 26ms 추론 |
| 실시간 및 효율성 최적화 > YOLACT++ 실시간 분할    | Real-time Instance Segmentation of Surgical Instruments using Attention and Multi-scale Feature Fusion (2021) | MSFF 와 CBAM 통합, 도메인 타겟 데이터 증강으로 69 fps→24 fps 실시간 달성    |
| 실시간 및 효율성 최적화 > Efficient Frame Pruning | Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning (2024)                | cosine similarity 기반 프레임 유사성 제거로 vanilla SAM2 대비 3 배 FPS 향상 |

#### 2.9 Tip Detection 및 Keypoint Detection

수술 도구 끝 부분 (tip) 의 위치를 찾는 specialized 태스크 연구들이다.

| 분류                                                                    | 논문명                                                                                                | 분류 근거                                                                                        |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Tip Detection 및 Keypoint Detection > segmentation-driven tip detection | ToolTipNet: A Segmentation-Driven Deep Learning Baseline for Surgical Instrument Tip Detection (2025) | segmentation module 과 tip detection 모듈 분리, segmentation 기반 input 만 사용하는 tip detector |

#### 2.10 응용 및 체계적 검토

문헌 분석, 체계적 검토, 데이터셋 생태계 분석, 응용 연구를 포함한 연구들이다.

| 분류                                                | 논문명                                                                                                                                                    | 분류 근거                                                                                |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 응용 및 체계적 검토 > 시스템 통합 개념적 프레임워크 | Perspectives on Surgical Data Science (2016)                                                                                                              | surgical data science 의 개념적 프레임워크와 응용 영역을 체계화하는 meta-perspective     |
| 응용 및 체계적 검토 > 체계적 문헌 분석              | Methods and datasets for segmentation of minimally invasive surgical instruments in endoscopic images and videos: A review of the state of the art (2023) | 시각 기반 분할 분야의 체계적 문헌 분석으로 데이터셋 생태계와 방법론 분류를 포괄적 문서화 |
| 응용 및 체계적 검토 > 다목적 DL 모델 평가           | Deep Learning for Surgical Instrument Recognition and Segmentation in Robotic-Assisted Surgeries: A Systematic Review (2024)                              | 로봇 수술 분야에서 수술 도구 감지 및 분할에 초점을 둔 다목적 DL 모델 평가                |
| 응용 및 체계적 검토 > 데이터셋 기반 성능 비교       | SAR-RARP50: Segmentation of surgical instrumentation and Action Recognition on Robot-Assisted Radical Prostatectomy Challenge (2023)                      | 공개 다중 모드 수술 데이터셋 기반 단일/다중 작업 학습 성능 비교 연구                     |
| 응용 및 체계적 검토 > 응용 중심 실증 분석           | Identifying Surgical Instruments in Laparoscopy Using Deep Learning Instance Segmentation (2025)                                                          | 소량 의료 영상 데이터를 기반으로 binary/multi-class 과제를 분리하여 실증 분석            |
| 응용 및 체계적 검토 > 도메인 특화 데이터셋          | Automatic Instrument Segmentation in Robot-Assisted Surgery Using Deep Learning (2018)                                                                    | 제한된 의료 영상 데이터 환경에서 pretrained encoder 활용 실용성 입증                     |

#### 2.11 심층적 시간 모델링 및 비디오 기반 분할

심층적인 시간적 정보를 활용한 복잡한 비디오 분할 연구들이다.

| 분류                                             | 논문명                                                                                               | 분류 근거                                                                   |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 심층적 시간 모델링 > 모션 플로우 보상 프레임워크 | Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video (2020) | Frame-Label 공동 전파, ConvLSTM 로 시간적 일관성 강화                       |
| 심층적 시간 모델링 > 시공간 지식 통합 아키텍처   | Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video (2021) | BottleneckLSTM 과 Non-local 결합, 능동적 메모리 업데이트로 시공간 지식 통합 |

#### 2.12 heatmap 회귀 및 밀집 예측 기반 추적

heatmap regression 기반의 dense prediction과 추적 연구들이다.

| 분류                                         | 논문명                                                                               | 분류 근거                                                                      |
| -------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| heatmap 회귀 및 밀집 예측 > 좌표 회귀 재정의 | Concurrent Segmentation and Localization for Tracking of Surgical Instruments (2017) | landmark localization 을 좌표 회귀 대신 heatmap regression 으로 재정의 및 통합 |

### 3. 종합 정리

본 분류 체계는 surgical instrument segmentation 분야 연구를 **데이터 효율성, 지도/비지도/메타 학습, 시간적 정보 통합, 비전-언어 모델, foundation 모델, 실시간 최적화, specialized 태스크, 응용 연구** 등 11 가지 주요 범주로 구성하였으며, 이는 총 38 편의 논문을 포괄적으로 분류하였다. 2016 년부터 2025 년까지의 연구 흐름을 살펴보면, 초기에는 residual learning 과 어텐션 기반 U-Net 을 활용한 지도 학습이 주류를 이루었으나, 이후에는 데이터 효율성을 높이기 위한 합성 데이터 생성과 SAM 기반 foundation 모델 활용, 텍스트/음성 등 비전 정보를 활용한 프롬프트 기반 적응, 시간적 일관성을 강화한 메모리/모션 플로우 기반 접근법이 활발히 등장하였다. 또한 실시간 처리 속도 확보를 위한 경량화 및 효율성 최적화 연구, tip detection 과 같은 specialized 태스크에 대한 연구도 꾸준히 진행되고 있으며, 다중 작업 학습과 체계적 문헌 분석을 통한 응용 연구도 확대되고 있는 것으로 확인된다.

## 2 장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

## 1.1 문제 정의의 공통성

모든 연구에서 공통되는 핵심 문제: **내시경 영상에서 수술 도구의 정확한 분할 (Segmentation)

| 구성 요소         | 기술적 표현                                   | 역할                              |
| ----------------- | --------------------------------------------- | --------------------------------- |
| 입력 (Input)      | 내시경 영상 \(I\), 프레임 시퀀스              | 도구, 조직, 배경 시각 정보        |
| 처리 (Processing) | 딥러닝 모델, 특징 추출, 예측                  | 시각적 패턴 → 클래스/마스크 매핑  |
| 출력 (Output)     | 분할 맵 \(\Sigma\), 이진/유형/인스턴스 마스킹 | 픽셀 단위 또는 인스턴스 단위 식별 |

## 1.2 방법론적 접근의 공통 구조

전체 연구의 **3 단계 파이프라인** 구조:

```text
[데이터 수집] → [특징 처리] → [분할 예측]
   ↓                ↓             ↓
[이미지/비디오] → [특징 추출] → [마스크 생성]
```

| 단계             | 주요 기술              | 목적                    |
| ---------------- | ---------------------- | ----------------------- |
| 1. 데이터 전처리 | 회전, 증강, 블렌딩     | 학습 데이터 품질 향상   |
| 2. 특징 처리     | CNN/Transformer/어텐션 | 공간적·시간적 정보 통합 |
| 3. 분할 예측     | U-Net/FCN/Decoder      | 픽셀 단위 예측 생성     |

## 2. 방법론 계열 분류

## 2.1 방법론 계열 목록

| 방법론 계열                         | 논문 제목 (연도)                                                                                                                     | 핵심 특징                               |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| **Multi-Angle Feature Aggregation** | Towards Better Surgical Instrument Segmentation in Endoscopic Vision: Multi-Angle Feature Aggregation and Contour Supervision (2020) | 회전-병렬 특징 처리, 다각도 특징 평균화 |
| **Attention-based Receptive Field** | BARNet: Bilinear Attention Network with Adaptive Receptive Fields for Surgical Instrument Segmentation (2020)                        | 2 차 어텐션 분포, 적응적 수용장         |
| **Temporal Flow-based**             | Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video (2020)                                 | 모션 플로우, 시간적 일관성 활용         |
| **Instance-based**                  | ISINet: An Instance-Based Approach for Surgical Instrument Segmentation (2020)                                                       | 인스턴스 후보, IoU 매칭                 |
| **Anchor Generation**               | Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion (2020)                                    | 앵커 생성, 의미론적 확산                |
| **Synthetic Data**                  | Image Compositing for Segmentation of Surgical Tools without Manual Annotations (2021)                                               | 합성 데이터셋, mix-blend                |
| **Meta-Learning**                   | One to Many: Adaptive Instrument Segmentation via Meta Learning and Dynamic Online Adaptation in Robotic Surgical Video (2021)       | MAML, 온라인 적응                       |
| **Dual-Memory**                     | Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video (2021)                                 | 지역/전역 메모리, ELA/AGA               |
| **Single-Stage Real-time**          | Real-time Instance Segmentation of Surgical Instruments using Attention and Multi-scale Feature Fusion (2021)                        | YOLACT++, MSFF, CBAM                    |
| **Optical Flow-based**              | FUN-SIS: a Fully UNsupervised approach for Surgical Instrument Segmentation (2022)                                                   | 생성적 적대적 학습, 광학 흐름 분할      |
| **Background-Only**                 | Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need (2022)                                           | 단일 배경 이미지, 합성 데이터           |
| **3-Stage Classification**          | From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments (2022)                                      | S3Net, 3 단계 분류                      |
| **Transformer-based**               | MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation (2023)                                                     | Mask2Former,Temporal consistency        |
| **Text-Promptable**                 | Text Promptable Surgical Instrument Segmentation with Vision-Language Models (2023)                                                  | CLIP, 텍스트 프롬프팅                   |
| **Detection-free VQA**              | Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery (2023)    | GVLE, detection-free                    |
| **Zero-shot Prompt**                | SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation (2023)                                    | SAM, LoRA 파인튜닝                      |
| **Prototype-based**                 | SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation (2023)                                                      | 프로토타입, 대조 학습                   |
| **Audio-Driven**                    | ASI-Seg: Audio-Driven Surgical Instrument Segmentation with Surgeon Intention Understanding (2024)                                   | 음성 의도 인식, intention-aware         |
| **Frame-Pruned**                    | Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning (2024)                                       | 효율적 프레임 프루닝, memory pruning    |
| **Part-Level Detection**            | ToolTipNet: A Segmentation-Driven Deep Learning Baseline for Surgical Instrument Tip Detection (2025)                                | part-level mask, heatmap regression     |
| **Instance-based (Mask R-CNN)**     | Identifying Surgical Instruments in Laparoscopy Using Deep Learning Instance Segmentation (2025)                                     | Mask R-CNN                              |
| **Residual Baseline**               | Deep Residual Learning for Instrument Segmentation in Robotic Surgery (2017)                                                         | ResNet-101, dilated convolution         |
| **Concurrent Learning**             | Concurrent Segmentation and Localization for Tracking of Surgical Instruments (2017)                                                 | 동시 분할/위치 추정                     |
| **U-Net Encoder-Decoder**           | Automatic Instrument Segmentation in Robot-Assisted Surgery Using Deep Learning (2018)                                               | U-Net, skip connection                  |
| **Attention Fusion**                | RASNet: Segmentation for Tracking Surgical Instruments in Surgical Videos Using Refined Attention Segmentation Network (2019)        | AFM, 어텐션 퓨전                        |
| **Augmented Attention**             | RAUNet: Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments (2019)                                   | AAM, CEL-Dice Loss                      |
| **Lightweight**                     | Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments (2019)                               | 경량 디코더, 어텐션 퓨전                |

## 2.1 계열별 상세 분석

### (1) Multi-Angle Feature Aggregation 계열

**계열 정의**: 내시경 영상의 시각적 불변성을 확보하기 위해 이미지의 회전 변형을 인코더 수준에서 병렬 처리하는 방법

**공통 특징**:

- 입력 이미지에 대한 회전 세트 병렬 처리
- 다각도 특징 맵 평균화
- 공유 인코더를 통한 병렬 특징 추출

| 핵심 요소 | 기술적 구현                                | 역할                             |
| --------- | ------------------------------------------ | -------------------------------- |
| 회전 변형 | $I, \{\phi_k\}$                            | $[0^\circ, 360^\circ]$ 균일 간격 |
| 특징 집계 | $H_{MA} = \frac{1}{N_A} \sum H_{\phi_k}^A$ | 회전 불변성 확보                 |
| 손실 함수 | $L = L_S + L_C$                            | 분할 + 윤곽선                    |

**해당 논문**:

- Towards Better Surgical Instrument Segmentation in Endoscopic Vision: Multi-Angle Feature Aggregation and Contour Supervision (2020)

### (2) Attention-based Receptive Field 계열

**계열 정의**: 어텐션 메커니즘을 통해 전역 컨텍스트와 지역 정보를 통합하며 수용장을 동적으로 조절하는 방법

**공통 특징**:

- 전역 디스크립터 생성 및 분배
- 채널·스페이스 어텐션 동시 적용
- 적응적 수용장 조절

| 핵심 요소   | 기술적 구현            | 역할                |
| ----------- | ---------------------- | ------------------- |
| 전역 어텐션 | $D \times D$ 어텐션 맵 | 전역 의존성 모델링  |
| 어텐션 융합 | 채널·스페이스 병합     | 지역/전역 정보 통합 |
| 가중치 적용 | $S \otimes P$          | 수용장 조절         |

**해당 논문**:

- BARNet: Bilinear Attention Network with Adaptive Receptive Fields for Surgical Instrument Segmentation (2020)
- Real-time Instance Segmentation of Surgical Instruments using Attention and Multi-scale Feature Fusion (2021)
- RASNet: Segmentation for Tracking Surgical Instruments in Surgical Videos Using Refined Attention Segmentation Network (2019)
- RAUNet: Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments (2019)
- Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments (2019)

### (3) Temporal Flow-based 계열

**계열 정의**: 시간적 모션 정보를 활용하여 프레임 간 일관성을 유지하고 분할 예측을 개선하는 방법

**공통 특징**:

- 광학 흐름 계산 및 활용
- 시간적 일관성 제약
- 프레임 전파 및 공동 전파

| 핵심 요소   | 기술적 구현           | 역할                |
| ----------- | --------------------- | ------------------- |
| 모션 플로우 | FlowNet2.0 기반       | 프레임 간 변형 예측 |
| 프레임 워핑 | 빌리니어 인터폴레이션 | 시간 전파           |
| 시간적 제약 | 쿼드러플렛 형태       | 일관성 유지         |

**해당 논문**:

- Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video (2020)
- FUN-SIS: a Fully UNsupervised approach for Surgical Instrument Segmentation (2022)

### (4) Instance-based 계열

**계열 정의**: 인스턴스 단위 분할을 위해 객체 후보 추출 후 시간적·공간적 일관성 모듈로 통합하는 방법

**공통 특징**:

- 인스턴스 후보 생성
- IoU 기반 매칭
- 중복 예측 활용

| 핵심 요소     | 기술적 구현         | 역할           |
| ------------- | ------------------- | -------------- |
| 인스턴스 추출 | Mask R-CNN          | 후보 영역 생성 |
| IoU 매칭      | $IoU(U_i) = IoU(0)$ | 시간적 일관성  |
| 클래스 할당   | 가중 최빈값         | 객체 식별      |

**해당 논문**:

- ISINet: An Instance-Based Approach for Surgical Instrument Segmentation (2020)
- From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments (2022)

### (5) Anchor Generation 계열

**계열 정의**: 수작업 주석 없이 앵커 생성 및 확산을 통해 초기 지도 신호를 생성하는 비지도 학습 방법

**공통 특징**:

- 수작업 큐 계산 및 융합
- 긍정/부정 앵커 생성
- 의미론적 확산 손실

| 핵심 요소 | 기술적 구현              | 역할             |
| --------- | ------------------------ | ---------------- |
| 큐 계산   | 색상, 객체성, 위치       | 초기 의사 레이블 |
| 앵커 융합 | 쿼드러플렛               | 초기 지도 생성   |
| 확산 손실 | $L_{fg/dif}, L_{bg/dif}$ | 모호 영역 전파   |

**해당 논문**:

- Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion (2020)

### (6) Synthetic Data 계열

**계열 정의**: 수동 주석 없이 실제 성능을 달성하기 위해 합성 데이터를 생성하고 학습하는 방법

**공통 특징**:

- 전경/배경 분리 및 합성
- 확률적 mix-blend
- GrabCut 후처리

| 핵심 요소 | 기술적 구현    | 역할           |
| --------- | -------------- | -------------- |
| 전경/배경 | $X_F, X_B$     | 블렌딩 입력    |
| mix-blend | $0.2-0.8$ 확률 | 무한 변형 생성 |
| 후처리    | GrabCut        | 이진화         |

**해당 논문**:

- Image Compositing for Segmentation of Surgical Tools without Manual Annotations (2021)
- Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need (2022)

### (7) Meta-Learning 계열

**계열 정의**: 소량의 주석으로 다양한 도메인에 빠르게 적응할 수 있도록 사전 학습된 메타 학습자 구성하는 방법

**공통 특징**:

- MAML 기반 외부/내부 루프
- 노이즈 인식 그래디언트 게이트
- 유사 마스크 품질 평가

| 핵심 요소 | 기술적 구현          | 역할          |
| --------- | -------------------- | ------------- |
| 외부 루프 | $\theta^*, \alpha^*$ | 도메인 적응   |
| 내부 루프 | $K \le 5$ 반복       | 빠른 적응     |
| 게이트    | 그래디언트 게이트    | 노이즈 필터링 |

**해당 논문**:

- One to Many: Adaptive Instrument Segmentation via Meta Learning and Dynamic Online Adaptation in Robotic Surgical Video (2021)

### (8) Dual-Memory 계열

**계iel 정의**: 지역적 시간 의존성과 전역적 의미 상관관계를 분리하여 통합하는 2 메모리 네트워크

**공통 특징**:

- 지역 메모리 (인접 프레임)
- 전역 메모리 (능동적 선택)
- ELA/AGA 병렬 파이프라인

| 핵심 요소   | 기술적 구현     | 역할             |
| ----------- | --------------- | ---------------- |
| 지역 메모리 | BottleneckLSTM  | 인접 프레임 처리 |
| 전역 메모리 | AGA 프레임 선별 | 장기 의존성      |
| 집계        | Non-local       | 시공간 통합      |

**해당 논문**:

- Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video (2021)

### (9) Single-Stage Real-time 계열

**계iel 정의**: 단일 단계 분할 네트워크에 어텐션과 다중 스케일 특징 융합을 통합하여 실시간 처리를 구현하는 방법

**공통 특징**:

- YOLACT++ 기반
- 다중 스케일 특징 융합
- 앵커 최적화

| 핵심 요소 | 기술적 구현                        | 역할          |
| --------- | ---------------------------------- | ------------- |
| MSFF      | $F_{MS} = conv([F_0, \dots, F_4])$ | 전역 컨텍스트 |
| CBAM      | 채널·스페이스 어텐션               | 특징 정제     |
| 앵커      | Differential Evolution             | 최적화        |

**해당 논문**:

- Real-time Instance Segmentation of Surgical Instruments using Attention and Multi-scale Feature Fusion (2021)

### (10) Optical Flow-based 계열

**계iel 정의**: 생성적 적대적 학습을 통해 광학 흐름 이미지를 생성하고 이를 분할에 활용하는 방법

**공통 특징**:

- shape-priors 기반 GAN
- Teacher-Proxy-Student 학습
- 예측 불가능성 활용

| 핵심 요소 | 기술적 구현                   | 역할             |
| --------- | ----------------------------- | ---------------- |
| Teacher   | Cycle-GAN + Cycle-consistency | 광학 흐름 분할   |
| Proxy     | Unet11 (소형)                 | 의사 레이블 학습 |
| Student   | Unet16/20 (표준)              | 최종 분할        |

**해당 논문**:

- FUN-SIS: a Fully UNsupervised approach for Surgical Instrument Segmentation (2022)

### (11) Background-Only 계열

**계iel 정의**: 단일 배경 이미지와 소수의 전경 이미지만을 사용하여 고품질 합성 데이터셋을 생성하는 방법

**공통 특징**:

- 단일 배경 $X_b^p$
- 연쇄 증강 혼합
- 동적 클래스 균형

| 핵심 요소   | 기술적 구현    | 역할          |
| ----------- | -------------- | ------------- |
| 배경 풀     | $X_b^p$        | 합성 기반     |
| AugMix      | Soft/Hard      | 데이터 다양성 |
| 클래스 균형 | Beta/Dirichlet | 균형 유지     |

**해당 논문**:

- Rethinking Surgical Instrument Segmentation: A Background Image Can Be All You Need (2022)

### (12) 3-Stage Classification 계열

**계iel 정의**: 기존 2 단계 인스턴스 분할 모델을 3 단계로 재구조화하여 분류 정확도를 개선하는 방법

**공통 특징**:

- Stage 1: 바운딩 박스 제안
- Stage 2: 마스크 예측
- Stage 3: 마스크 어텐션 분류

| 핵심 요소 | 기술적 구현      | 역할        |
| --------- | ---------------- | ----------- |
| RPN       | 초기 박스 제안   | 객체 후보   |
| FCN       | 다중 스케일 특징 | 마스크 생성 |
| MSMA      | 마스크 어텐션    | 배경 제거   |

**해당 논문**:

- From Forks to Forceps: A New Framework for Instance Segmentation of Surgical Instruments (2022)

### (13) Transformer-based 계열

**계iel 정의**: Transformer 아키텍처와 어텐션 메커니즘을 통해 공간적·시간적 정보를 통합하는 방법

**공통 특징**:

- Mask2Former baseline
- Deformable attention
- Temporal context 통합

| 핵심 요소  | 기술적 구현   | 역할          |
| ---------- | ------------- | ------------- |
| Deformable | Pixel decoder | 피델리티      |
| Temporal   | MViT encoder  | 시간적 일관성 |
| Region Set | N=100 mask    | 예측 통합     |

**해당 논문**:

- MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation (2023)

### (14) Text-Promptable 계열

**계iel 정의**: 텍스트 프롬프트를 입력으로 받아 다양한 도구 유형에 동적으로 적응하는 분할 방법

**공통 특징**:

- CLIP 이미지/텍스트 인코딩
- 어텐션-컨볼루션 프롬프팅
- 프롬프트 혼합 (MoP)

| 핵심 요소 | 기술적 구현       | 역할           |
| --------- | ----------------- | -------------- |
| CLIP      | ViT + Transformer | 시각-언어 특징 |
| MSFA      | FPN 융합          | 다중 스케일    |
| HiAR      | 어려운 영역 강화  | 집중 학습      |

**해당 논문**:

- Text Promptable Surgical Instrument Segmentation with Vision-Language Models (2023)

### (15) Detection-free VQA 계열

**계iel 정의**: 객체 감지기 없이 전역 장면 이해를 통해 질문에 대한 위치 기반 답변을 예측하는 방법

**공통 특징**:

- ResNet18 시각 추출
- ViT 인코더
- GVLE 게이트

| 핵심 요소  | 기술적 구현          | 역할           |
| ---------- | -------------------- | -------------- |
| GVLE       | 게이트 노드 $\alpha$ | 이종 특징 융합 |
| FFN        | 3 층 위치 파악       | 위치 예측      |
| End-to-End | 공동 학습            | 효율성         |

**해당 논문**:

- Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery (2023)

### (16) Zero-shot Prompt 계열

**계iel 정의**: SAM의 제로샷 분할 능력을 활용하고 LoRA를 통해 도메인 적응하는 방법

**공통 특징**:

- ViTb 이미지 인코더 (고정)
- LoRA 적응 ($r=4$)
- 프롬프트 기반 분할

| 핵심 요소 | 기술적 구현    | 역할        |
| --------- | -------------- | ----------- |
| SAM       | ViTb 고정      | 제로샷 기반 |
| LoRA      | $r=4$, 18.28MB | 도메인 적응 |
| 프롬프트  | 바운딩/포인트  | 클래스 분할 |

**해당 논문**:

- SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation (2023)

### (17) Prototype-based 계열

**계iel 정의**: 클래스 프로토타입을 임베딩으로 사용하고 대조 학습을 통해 클래스 간 판별력을 높이는 방법

**공통 특징**:

- 프로토타입 기반 프롬프트
- Dense/Sparse 임베딩 병렬
- InfoNCE 대조 손실

| 핵심 요소    | 기술적 구현   | 역할          |
| ------------ | ------------- | ------------- |
| 프로토타입   | 클래스 임베딩 | 프롬프트 생성 |
| Dense/Sparse | 병렬 임베딩   | 다양성        |
| InfoNCE      | 대조 학습     | 판별력        |

**해당 논문**:

- SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation (2023)

### (18) Audio-Driven 계열

**계iel 정의**: 외과의의 음성 의도를 인식하여 특정 도구만 선택적으로 분할하는 intention-aware 분할 방법

**공통 특징**:

- Mel spectrogram 오디오 인코딩
- Intent classifier
- Contrastive prompt 학습

| 핵심 요소   | 기술적 구현     | 역할              |
| ----------- | --------------- | ----------------- |
| Audio       | Mel spectrogram | 의도 인식         |
| Intent      | classifier      | target/non-target |
| Contrastive | prompt 학습     | 의도 기반         |

**해당 논문**:

- ASI-Seg: Audio-Driven Surgical Instrument Segmentation with Surgeon Intention Understanding (2024)

### (19) Frame-Pruned 계열

**계iel 정의**: Surgical SAM 2를 기반으로 동적 메모리 bank 관리와 효율적 프레임 프루닝을 적용하는 방법

**공통 특징**:

- Cosine similarity 기반 프루닝
- ViT-Small backbone
- Dynamic memory bank

| 핵심 요소         | 기술적 구현 | 역할          |
| ----------------- | ----------- | ------------- |
| Cosine Similarity | 메모리 정제 | 중복 제거     |
| ViT-Small         | lightweight | 효율성        |
| EFP               | 메모리 관리 | 메모리 최적화 |

**해당 논문**:

- Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning (2024)

### (20) Part-Level Detection 계열

**계iel 정의**: part-level segmentation 마스크를 직접 입력으로 사용하여 tool tip 위치를 예측하는 segmentation-driven 방법

**공통 특징**:

- part-level mask 입력
- multi-scale feature fusion
- mask-guided attention

| 핵심 요소      | 기술적 구현      | 역할      |
| -------------- | ---------------- | --------- |
| HRNet          | multi-resolution | 형상 정보 |
| Mask Attention | gripper-part     | 공간 집중 |
| Heatmap        | argmax           | 위치 예측 |

**해당 논문**:

- ToolTipNet: A Segmentation-Driven Deep Learning Baseline for Surgical Instrument Tip Detection (2025)

### (21) Instance-based (Mask R-CNN) 계열

**계iel 정의**: Mask R-CNN 아키텍처를 활용한 인스턴스 단위 분할 및 식별 방법

**공통 특징**:

- Region Proposal Network
- FCN 단계별 예측
- ResNet-101 백본

| 핵심 요소 | 기술적 구현    | 역할      |
| --------- | -------------- | --------- |
| RPN       | 후보 영역      | 객체 제안 |
| FCN       | 분류/위치/mask | 다중 예측 |
| NMS       | 사후 처리      | 중복 제거 |

**해당 논문**:

- Identifying Surgical Instruments in Laparoscopy Using Deep Learning Instance Segmentation (2025)
- ISINet: An Instance-Based Approach for Surgical Instrument Segmentation (2020)

### (22) Residual Baseline 계열

**계iel 정의**: ResNet-101 의 residual learning 과 FCN 구조를 결합하여 고해상도 분할을 구현하는 기본선 방법

**공통 특징**:

- ResNet-101 분류 백본
- Downsampling 감소
- Dilated convolution

| 핵심 요소  | 기술적 구현     | 역할        |
| ---------- | --------------- | ----------- |
| ResNet-101 | residual unit   | 깊이 표현   |
| Dilated    | Receptive field | 정보 보존   |
| Bilinear   | 업샘플링        | 해상도 복원 |

**해당 논문**:

- Deep Residual Learning for Instrument Segmentation in Robotic Surgery (2017)

### (23) Concurrent Learning 계열

**계iel 정의**: 단일 네트워크에서 분할과 위치 추정을 동시 수행하는 concurrent dense prediction 방법

**공통 특징**:

- 공유 encoder-decoder
- Segmentaion/Localization branch
- Long-range skip connection

| 핵심 요소          | 기술적 구현          | 역할      |
| ------------------ | -------------------- | --------- |
| Shared Decoder     | SL                   | 공통 처리 |
| Heatmap Regression | Gaussian             | 위치 예측 |
| Concat             | Localization feature | 정보 통합 |

**해당 논문**:

- Concurrent Segmentation and Localization for Tracking of Surgical Instruments (2017)

### (24) U-Net Encoder-Decoder 계열

**계iel 정의**: U-Net 계열 구조와 사전 학습 인코더를 결합하여 픽셀 단위 클래스 예측을 수행하는 기본 구조

**공통 특징**:

- Contracting/Expanding path
- Skip connection
- Pretrained encoder

| 핵심 요소 | 기술적 구현       | 역할      |
| --------- | ----------------- | --------- |
| U-Net     | 3×3 conv, pooling | 기본 구조 |
| Encoder   | VGG/ResNet        | 사전 학습 |
| Dice Loss | IoU 제약          | 형태 최적 |

**해당 논문**:

- Automatic Instrument Segmentation in Robot-Assisted Surgery Using Deep Learning (2018)
- RAUNet: Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments (2019)

### (25) Attention Fusion 계열

**계iel 정의**: 단순 skip connection 대신 어텐션 퓨전으로 고수준 전역 컨텍스트를 활용하는 방법

**공통 특징**:

- ResNet-50 인코더
- AFM 어텐션 모듈
- 가중 합 손실

| 핵심 요소    | 기술적 구현 | 역할          |
| ------------ | ----------- | ------------- |
| ResNet-50    | 인코더      | 특징 추출     |
| AFM          | 어텐션 퓨전 | 전역 컨텍스트 |
| Weighted Sum | 손실        | 불균형 해결   |

**해당 논문**:

- RASNet: Segmentation for Tracking Surgical Instruments in Surgical Videos Using Refined Attention Segmentation Network (2019)

### (26) Augmented Attention 계열

**계iel 정의**: 기존 어텐션 모듈 대비 적은 파라미터 증가로 성능 향상, 로그 Dice 손실로 클래스 불균형 해결하는 방법

**공통 특징**:

- ResNet34 인코더
- AAM 증강된 어텐션
- CEL-Dice 손실

| 핵심 요소 | 기술적 구현        | 역할        |
| --------- | ------------------ | ----------- |
| ResNet34  | 인코더             | 사전 학습   |
| AAM       | 1×1 conv, BN, ReLU | 어텐션 생성 |
| LogDice   | 손실               | 불균형      |

**해당 논문**:

- RAUNet: Residual Attention U-Net for Semantic Segmentation of Cataract Surgical Instruments (2019)

### (27) Lightweight 계열

**계iel 정의**: 경량 인코더 (MobileNetV2) 와 경량 어텐션 디코더로 구성되어 효율적인 실시간 분할을 수행하는 방법

**공통 특징**:

- MobileNetV2 인코더
- Depthwise Separable Conv
- Focal Loss

| 핵심 요소   | 기술적 구현    | 역할      |
| ----------- | -------------- | --------- |
| MobileNetV2 | 경량 인코더    | 효율성    |
| Depthwise   | Separable conv | 계산 비용 |
| Focal Loss  | $\gamma=6$     | 불균형    |

**해당 논문**:

- Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments (2019)

## 2.2 계열 분류 요약

**전체 계열 수**: 27 가지

| 분류 기준   | 계열 분포                                                   |
| ----------- | ----------------------------------------------------------- |
| 데이터 소스 | 수동 주석, 합성, 비지도, 메타 학습, 음성, 텍스트            |
| 시간 정보   | 단일 프레임, 동시 학습, 광학 흐름, 메모리, temporal context |
| 모델 구조   | CNN, U-Net, ResNet, Transformer, SAM                        |
| 적응 방식   | 사전 학습, 파인튜닝, 온라인 적응, 제로샷                    |
| 실시간 여부 | 실시간, 실시간, 오프라인                                    |

## 3. 핵심 설계 패턴 분석

## 3.1 모듈 분리 패턴

**패턴 설명**: 주요 기능적 모듈을 분리하여 재사용성 및 확장성을 확보하는 구조

| 모듈      | 역할           | 활용 논문                     |
| --------- | -------------- | ----------------------------- |
| Encoder   | 특징 추출      | ResNet, VGG, ViT 등 모든 모델 |
| Decoder   | 해상도 복원    | U-Net, FCN 계열               |
| Attention | 특징 정제      | CBAM, BAM, AFM 등             |
| Memory    | 시간적 의존성  | ELA, AGA, Local/Global        |
| Loss      | 목적 함수 최적 | Dice, CE, Log-IoU             |

## 3.2 단계적 처리 패턴

**패턴 설명**: 복잡한 작업을 단계별로 나누어 처리하는 구조

```text
예시: S3Net (3 단계)
  ↓ Stage 1: Box Proposal (RPN)
  ↓ Stage 2: Mask Prediction (FCN + NMS)
  ↓ Stage 3: Classification (MSMA)
```

| 단계화 유형 | 예시                                 | 논문               |
| ----------- | ------------------------------------ | ------------------ |
| 2 단계      | Proposal → Classification            | MATIS (2023)       |
| 3 단계      | Proposal → Mask → Classification     | S3Net (2022)       |
| Multi-stage | Fine-tuning → Adaptation → Inference | SurgicalSAM (2023) |

## 3.3 학습 방식 패턴

**지도 학습**:

- Cross Entropy, Dice Loss
- Hybrid 손실 ($CE + Dice$)
- Arc Loss (메트릭 학습)

**비지도/약지도 학습**:

- Anchor Generation
- Semantic Diffusion
- Teacher-Proxy-Student
- Self-supervised (FlowNet)

**전이 학습**:

- ImageNet 사전 학습
- COCO pretraining
- Kinetics-400 temporal pretraining

**메타 학습**:

- MAML 기반 외부/내부 루프
- Gradient gating for noise filtering

## 3.4 데이터 활용 패턴

**합성 데이터 생성**:

- Chroma key
- mix-blend
- AugMix 연쇄

**시간적 정보 활용**:

- 인접 프레임 전파
- 광학 흐름 워핑
- Memory bank

**주석 효율성**:

- 제로샷 (SAM 기반)
- Few-shot (메타 학습)
- Noisy label filtering

## 4. 방법론 비교 분석

## 4.1 문제 접근 방식 차이

| 접근 방식           | 장점               | 단점             |
| ------------------- | ------------------ | ---------------- |
| **Multi-Angle**     | 회전 불변성 확보   | 계산 비용 증가   |
| **Attention-based** | 전역 컨텍스트 활용 | 파라미터 증가    |
| **Temporal Flow**   | 시간적 일관성      | 광학 흐름 의존성 |
| **Synthetic Data**  | 수동 주석 불필요   | 도메인 격차      |
| **Meta-Learning**   | 빠른 적응          | 소량 데이터 의존 |
| **Zero-shot**       | 프롬프트 기반 적응 | 프롬프트 의존성  |

## 4.2 구조/모델 차이

| 구분        | CNN 기반      | Transformer 기반 | SAM 기반      |
| ----------- | ------------- | ---------------- | ------------- |
| 특징 표현   | 지역적        | 전역적           | 프롬프트 기반 |
| 시간 정보   | LSTM/Flow     | Self-attention   | Memory bank   |
| 실시간      | ✓ (경량화)    | △ (계산 비용)    | △ (파인튜닝)  |
| 데이터 효율 | △ (전이 학습) | ✓ (사전 학습)    | ✓ (제로샷)    |

## 4.3 적용 대상 차이

| 대상             | 적합한 방법              |
| ---------------- | ------------------------ |
| 단일 프레임 분할 | U-Net, CNN 계열          |
| 실시간 처리      | 경량화, Single-stage     |
| 다양한 도구 적응 | Meta-learning, Zero-shot |
| 시간적 일관성    | Flow-based, Memory-based |
| 의도 기반 분할   | Audio-driven             |
| 텍스트 프롬프트  | Vision-Language          |

## 4.4 트레이드오프 분석

| 트레이드오프            | 최적화 방향             | 방법론 예시             |
| ----------------------- | ----------------------- | ----------------------- |
| 정확도 ↔ 속도           | 경량화, 프루닝          | LWANet, SurgicalSAM2    |
| 수동 주석 ↔ 데이터 효율 | 합성, 비지도            | Synthetic, Unsupervised |
| 사전 학습 ↔ 일반성      | 도메인 적응             | LoRA, Meta-learning     |
| 계산 비용 ↔ 성능        | 메모리 최적화           | EFP, Frame Pruning      |
| 프롬프트 필요 ↔ 편의성  | 제로샷 vs 사전 프롬프트 | SAM vs Traditional      |

## 5. 방법론 흐름 및 진화

## 5.1 초기 접근 (2017-2019)

**주요 특징**: 기본 CNN 구조 (U-Net, ResNet) 와 전이 학습 기반 방법

| 특징   | 설명                |
| ------ | ------------------- |
| 모델   | FCN, U-Net, ResNet  |
| 손실   | Cross Entropy, Dice |
| 데이터 | 수동 주석 필수      |
| 실시간 | △ (계산 비용)       |

## 5.2 발전된 구조 (2020-2021)

**주요 특징**: 어텐션 메커니즘 도입, 시간적 정보 통합

| 발전 방향     | 기술                              |
| ------------- | --------------------------------- |
| 특징 통합     | Multi-Angle Feature Aggregation   |
| 전역 컨텍스트 | Bilinear Attention, Global Memory |
| 시간적 일관성 | FlowNet, ConvLSTM                 |
| 데이터 효율   | Synthetic, Semi-supervised        |

## 5.3 최근 경향 (2022-2025)

**주요 특징**: Transformer 아키텍처, 텍스트/음성 프롬프트, SAM 기반

| 발전 방향     | 기술                          |
| ------------- | ----------------------------- |
| 프롬프트 기반 | Text, Class, Prototype        |
| 제로샷        | SAM, LoRA                     |
| 효율성        | Frame Pruning, Lightweight    |
| 멀티모달      | Audio-driven, Vision-Language |

## 5.4 진화 축

```text
2017-2019: CNN 기반 기본 구조
  ↓
2020-2021: 어텐션, 시간적 정보, 합성 데이터
  ↓
2022-2023: Transformer, 메타 학습, 프롬프트
  ↓
2024-2025: SAM 기반, 제로샷, 멀티모달
```

**진화 추세**:

- 수동 주석 의존 → 프롬프트 기반 → 제로샷
- 단일 프레임 → 시간적 통합 → 메모리 효율
- CNN → CNN+Attention → Transformer → SAM
- 고 계산량 → 경량화 → 실시간 최적화

## 6. 종합 정리

## 6.1 방법론 지형 지도

수술 도구 분할 방법론은 다음 3 축으로 구조화될 수 있다:

| 축                | 분류 기준   | 주요 계열                          |
| ----------------- | ----------- | ---------------------------------- |
| **데이터 의존성** | 주석 필요량 | 수동 주석 → 합성 → 비지도 → 제로샷 |
| **시간 정보**     | 시간 활용도 | 단일 → 동시 → 흐름 → 메모리        |
| **적응 방식**     | 도메인 적응 | 사전 → 전이 → 메타 → 프롬프트      |

## 6.2 방법론 분류

```text
┌─────────────────────────────────────────────┐
│    Surgical Instrument Segmentation         │
│              Methodologies                  │
├─────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Multi-Angle │  │ Transformer-based    │  │
│  │ Feature     │  │ Vision-Language      │  │
│  │ Aggregation │  │ Text-Promptable      │  │
│  └─────────────┘  └──────────────────────┘  │
│                                             │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Attention   │  │ SAM-based            │  │
│  │ Network     │  │ Prototype/Zero-shot  │  │
│  └─────────────┘  └──────────────────────┘  │
│                                             │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │ Temporal    │  │ Synthetic            │  │
│  │ Flow        │  │ Data-driven          │  │
│  └─────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────┘
```

## 6.3 공통성 및 차이점 요약

**공통성**:

- 인코더-디코더 기본 구조
- Dice/CE 손실 함수
- 전이 학습 및 데이터 증강

**차이점**:

- 정보 통합 방식 (어텐션 vs Transformer vs Flow)
- 데이터 의존성 (수동 주석 vs 합성 vs 제로샷)
- 적응 전략 (사전 학습 vs 메타 학습 vs 프롬프트)

**반복 패턴**:

- 효율성 ↔ 정확도 트레이드오프 관리
- 데이터 효율성 증대 경향
- 프롬프트 기반 적응의 등장

## 6.4 방법론 지형의 핵심 축

방법론 분석의 주요 축은:

1. **데이터 효율성 축**: 수동 주석 → 합성 데이터 → 비지도 → 제로샷
2. **시간 정보 축**: 단일 프레임 → 동시 처리 → 흐름 → 메모리
3. **적응 메커니즘 축**: 사전 학습 → 전이 학습 → 메타 학습 → 프롬프트
4. **아키텍처 축**: CNN → CNN+Attention → Transformer → SAM

## 3 장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

**주요 데이터셋**

| 데이터셋                | 사용 논문 수 | 용도                                  |
| ----------------------- | ------------ | ------------------------------------- |
| EndoVis-2017            | 20+          | 주요 벤치마크 (이진/도구별/유형 분할) |
| EndoVis-2018            | 15+          | 도구 부분 분할 벤치마크               |
| Cata7                   | 8+           | 백내장 수술 도구 분할                 |
| Sinus-Surgery (C/L)     | 1+           | 회전 불변성 평가                      |
| EndoVis-2015/2019       | 3+           | 초기/후기 벤치마크                    |
| RoboTool                | 2+           | 실제 임상 테스트셋                    |
| Cholec80                | 2+           | 도메인 간 일반화                      |
| ISIC-2016               | 1+           | 피부 병변 분확 (확장성 검증)          |
| SAR-RARP50              | 1+           | RARP 수술 분할                        |
| DAVIS2016               | 1+           | 일반 객체 분할 벤치마크               |
| STRAS/Cholec80/RandSurg | 1+           | 다양한 수술 환경                      |
| Kvasir Instrument       | 1+           | 추가 공개 데이터셋                    |

**평가 환경**

| 환경 유형              | 사용 빈도      | 특징                |
| ---------------------- | -------------- | ------------------- |
| 공개 데이터셋          | 90%+           | 재현성 확보         |
| 시뮬레이션 (돼지 모델) | 5%+            | 실제 수술 환경 근사 |
| 비공개 데이터셋        | 5%             | 재현성 저해         |
| 교차 검증              | 4-fold, 5-fold | 일반화 검증         |

**주요 평가 지표**

| 지표                    | 사용 빈도 | 측정 항목                   |
| ----------------------- | --------- | --------------------------- |
| IoU/mIoU                | 100%      | 분할 정확도 (주요 지표)     |
| Dice/mDice              | 90%+      | 중첩 계수                   |
| DSC                     | 40%       | Dice Similarity Coefficient |
| Sensitivity/Specificity | 30%       | 이진 분할 성능              |
| Balanced Accuracy       | 15%       | 불균형 데이터 성능          |
| FPS                     | 25%+      | 실시간 처리 속도            |
| FLOPs/Paras             | 10%+      | 계산 효율성                 |
| RMSE                    | 20%       | Tip 위치 정확도             |
| ChIoU/IoU_NB            | 15%+      | 과제 특화 지표              |

**비교 대상 방식**

- **Baseline 방법**: FCN-8s, U-Net, LinkNet, PSPNet
- **SOTA 방법**: TernausNet, MF-TAPNet, Dual-MF, MATIS
- **Cross-dataset 일반화**: EndoVis17→2018, EndoVis18→2017
- **Zero-shot 일반화**: SAM 기반 모델 비교

### 2. 주요 실험 결과 정렬

**시맨틱 분할 성능 비교 (주요 논문)**

| 논문명                                                 | 데이터셋/환경             | 비교 대상                                 | 평가 지표            | 핵심 결과                                                           |
| ------------------------------------------------------ | ------------------------- | ----------------------------------------- | -------------------- | ------------------------------------------------------------------- |
| Towards Better Surgical Instrument Segmentation (2020) | Sinus-Surgery/EndoVis2017 | DeepLabV3+, MFF, ICNet 등                 | mDSC, mIOU           | MAFA+윤곽선: mDSC 93.1% 달성, 회전 불변성 개선 (mIOU 분산 10%→2.0%) |
| BARNet (2020)                                          | Cata7/EndoVis2017         | PSPNet, DeepLabV2, TFNet 등               | mIOU, mDice          | Cata7: mIOU 97.47%, EndoVis2017: 64.30% (1 위)                      |
| Learning Motion Flows (2020)                           | EndoVis2017 MICCAI        | Self-training, Re-color, MF-TAPNet, UA-MT | IoU, Dice            | UA-MT 대비 IoU +2.68%, 완전 지도 능가                               |
| ISINet (2020)                                          | EndoVis2017/2018          | TernausNet, MF-TAPNet                     | IoU                  | 기본: 2 배, 완전: 3 배 TernausNet 대비 향상                         |
| Unsupervised via Anchor Generation (2020)              | EndoVis2017               | 반지도 (50%), 완전지도 (100%)             | IoU, Dice            | 0% 주석: IoU 0.71, 50%: IoU 0.80 달성                               |
| Image Compositing (2021)                               | EndoVis2017/RoboTool      | Trivial, Gaussian, Laplacian blend        | mIoU                 | mix-blend+GrabCut: RoboTool 68.1%, EndoVis2017 83.3%                |
| One to Many (2021)                                     | EndoVis17→18/타겟         | Fine-Tuning, 준지도, 자기지도             | IoU, Dice            | EndoVis18: IoU 73.3%, FT 시간 25s→1.5s                              |
| Efficient Global-Local Memory (2021)                   | EndoVis17/18              | ToolNet, TDNet 등 SOTA                    | mDice, mIoU, FPS     | EndoVis17: mDice 61.03%, FPS 38                                     |
| Real-time Instance Segmentation (2021)                 | ROBUST-MIS                | 챌린지 상위 팀 모델                       | MI_DSC, MI_NSD, FPS  | MI_DSC 44.7%, MI_NSD 48.9%, 69 FPS                                  |
| FUN-SIS (2022)                                         | EndoVis2017/DAVIS2016     | MF-TAPNet (89.61%), AGSD (71.47%)         | IoU                  | 83.77% IoU (MF-TAPNet 대비 -5.84% 포인트)                           |
| Background Image All You Need (2022)                   | EndoVis2017/2018          | 전체 데이터셋 훈련 모델                   | DSC                  | DSC 73.51% (전체 데이터 83.15% 대비)                                |
| From Forks to Forceps (2022)                           | EndoVis2017/2018          | ISINet, TraSeTR 등 18 개 SOTA             | ChIoU, mcIoU         | EndoVis2017: ChIoU 72.54, ISINet 대비 +30%                          |
| MATIS (2023)                                           | EndoVis2017/2018          | TraSeTR, CNN per-pixel 분류               | mIoU                 | 2017: 71.36, 2018: 84.26 (SOTA 76.20 초과)                          |
| Text Promptable (2023)                                 | EndoVis2017/2018          | TernausNet, ISINet, MATIS, SAM            | Ch_IoU, mc_IoU       | 기존 모든 방법 대비 개선                                            |
| Surgical-VQLA (2023)                                   | EndoVis-18/17-VQLA        | VisualBERT, VisualBERT ResMLP             | Accuracy, GIoU       | 객체 감지 제거: 8 배 속도 향상 (150.6 FPS)                          |
| SAM Meets Robotic Surgery (2023)                       | EndoVis2017/2018          | 지도학습 모델, SAM 공식                   | IoU                  | EndoVis17 이진 IoU 89.19% 달성                                      |
| SurgicalSAM (2023)                                     | EndoVis2018/2017          | MaskTrack+SAM, MATIS Frame                | mcIoU, Challenge IoU | mcIoU: 기존 SAM 기반/SOTA 모두 능가                                 |
| Surgical SAM 2 (2024)                                  | EndoVis17/18              | SAM2, SurgicalSAM, SOTA                   | Dice, J&F, FPS       | 29 FPS → 86 FPS (3 배), Dice 85.1→87.3                              |
| ASI-Seg (2024)                                         | EndoVis2018/2017          | SurgicalSAM, S3Net                        | IoU, mcIoU           | 2018 semantic: 82.37% (SurgicalSAM 80.33% 대비 +2.04%p)             |
| ToolTipNet (2025)                                      | Simulated/SurgPose        | Yang et al. (hand-crafted)                | RMSE, accuracy       | Simulated: RMSE 3.73, accuracy 0.959                                |
| Identifying Surgical Instruments (2025)                | Laparoscopy               | 증강/무증강, COCO pretrained              | AP50, AR1            | Binary: AP50 0.820, Multi-class: 0.613 (증강)                       |

**인스턴스 분할 성능 비교**

| 논문명                    | 데이터셋    | 평가 지표 | 비교 결과                           |
| ------------------------- | ----------- | --------- | ----------------------------------- |
| ISINet (2020)             | EndoVis2017 | IoU       | TernausNet 대비 2 배 향상           |
| Real-time Instance (2021) | ROBUST-MIS  | MI_DSC    | 챌린지 최고 팀 대비 +13.7%p         |
| S3Net (2022)              | EndoVis2017 | ChIoU     | ISINet 대비 +30%, TraSeTR 대비 +20% |
| MATIS (2023)              | EndoVis2017 | mIoU      | SOTA (TraSeTR 76.20) 대비 71.36     |
| SAM Meets (2023)          | EndoVis2017 | IoU       | 이진: 89.19%, Instrument: 88.20%    |
| SurgicalSAM (2023)        | EndoVis2018 | mcIoU     | oracle 대비 +20.07% 개선            |
| ASI-Seg (2024)            | EndoVis2018 | IoU       | SurgicalSAM 대비 +2.04%p            |

### 3. 성능 패턴 및 경향 분석

**1) 데이터셋 의존성 및 일반화 결과**

| 비교 유형            | 데이터셋              | 성능 영향   | 관찰 결과                                                                   |
| -------------------- | --------------------- | ----------- | --------------------------------------------------------------------------- |
| 교차 데이터셋 일반화 | EndoVis2017→2018      | -15~-25%p   | RoboTool 학습→EndoVis2017: 80.5%, EndoVis2017→RoboTool: 66.6% (13.8%p 하락) |
| 도메인 간 일반화     | Cholec80 (도메인)     | 정성적 평가 | 실제 데이터 일반화 어려움 (기구-조직 상호작용 차이)                         |
| 도구 확장성          | 2017→2017(+2000 도구) | +1.7%p DSC  | 새로운 도구 추가 시 성능 향상 (EndoVis2017: 75.69%→85.72%)                  |
| 시간적 정보 유무     | 단일 프레임 vs 시퀀스 | -20~30%p    | 시퀀스 정보 활용 시 성능 향상 (Dual-MF: UA-MT 대비 IoU +2.68%)              |

**2) 학습 전략별 성능**

| 전략        | 대표 논문                                | 성능                         | 특징                                         |
| ----------- | ---------------------------------------- | ---------------------------- | -------------------------------------------- |
| 완전 지도   | TernausNet, ISINet                       | IoU 89~90%                   | 데이터 주석 병목, 일반화 어려움              |
| 준지도      | Learning Motion Flows (2020)             | UA-MT 대비 IoU +2.68%        | 희소 주석 (10~20%) 에서 완전 지도 능가       |
| 반지도      | Unsupervised (2020), FUN-SIS (2022)      | 0% 주석: IoU 0.71, 50%: 0.80 | 0% 주석에서 80% 주석까지 점진적 향상         |
| 완전 비지도 | FUN-SIS (2022), Image Compositing (2021) | 83.77% (완전 비지도)         | 수동 주석 없이 83.77% 달성, 도메인 격차 감소 |
| 합성 데이터 | Background Image (2022)                  | 73.51% DSC                   | 실제 전체 데이터 83.15% 대비 85% 수준        |

**3) 실시간 처리 능력**

| 분류             | 최대 FPS    | 대표 논문                             | 속도-정확도 균형         |
| ---------------- | ----------- | ------------------------------------- | ------------------------ |
| 고속 (≥30 FPS)   | 182~174 FPS | Islam et al.(2019): 174 FPS (시맨틱)  | 정확도↓ (이진 분할 중심) |
| 중속 (20~40 FPS) | 38~49 FPS   | Efficient Global-Local Memory: 38 FPS | mDice 61%, MI_DSC 44.7%  |
| 저속 (<20 FPS)   | 4~15 FPS    | MATIS Frame: 15 FPS                   | mIoU↑ (84.26%)           |

**4) 모델 복잡도 및 효율성**

| 모델            | 파라미터       | FLOPs       | 추론 시간 | 비고                 |
| --------------- | -------------- | ----------- | --------- | -------------------- |
| MATIS Full      | ~8M            | -           | -         | Transformer 기반     |
| SurgicalSAM     | 4.65M          | -           | -         | 대조 프로토타입 학습 |
| Surgical SAM 2  | 1.08 GB 메모리 | -           | 86 FPS    | Fine-tuning 후       |
| LWANet          | 2.06M          | 3.39 GFLOPs | 39.49 FPS | 경량 네트워크        |
| SAM 2 (vanilla) | 3.10 GB 메모리 | -           | 29.10 FPS | Fine-tuning 전       |

### 4. 추가 실험 및 검증 패턴

**1) Ablation Study 공통 패턴**

| 검증 대상     | 검증 방식                    | 주요 발견                                             |
| ------------- | ---------------------------- | ----------------------------------------------------- |
| 모듈별 기여도 | ELA+AGA 분해 실험            | ELA 단독: +5.74%, AGA 단독: +5.13%, 모두 적용: +7.38% |
| 손실 함수     | CEL-Dice (α=0.2) vs CE       | CEL-Dice로 클래스 불균형 해결                         |
| 데이터 구성   | Synthetic-A/B/C 3 단계       | Synthetic-C+AugMix-Hard: EndoVis2018 73.51% DSC       |
| 메모리 설계   | ELA vs ConvLSTM vs Non-local | ELA: FLOPS 절반, 성능 유사                            |
| 프롬프트 전략 | 단일 vs 혼합 프롬프트        | MoP: 단일 대비 성능 향상                              |
| 프롬프트 유형 | Dense/Sparse, 양/음          | 모두 핵심 기여 입증                                   |
| 손실 제거     | MSFA/HiAR 제거 실험          | MSFA 제거: 성능 하락, HIAR: 결정적 역할               |

**2) 민감도 분석**

| 조건 변화 | 검증 논문            | 관찰 결과                                   |
| --------- | -------------------- | ------------------------------------------- |
| 주석 비율 | 10%/20%/30%          | 모든 비율에서 점진적 성능 향상              |
| 프레임 수 | n=5 vs n=10          | 최근 5 프레임이 충분, 더 많을수록 성능 저하 |
| 해상도    | 480×480 vs 1280×1024 | 640×544 입력 시 ~42 FPS                     |
| 클래스 수 | 7 vs 20              | 새로운 2000 도구 추가 시 +1.7%p             |
| 도메인    | EndoVis2017→2018     | Cross-dataset: 66.6%~83.3%                  |

**3) 조건 변화 실험**

| 실험 유형     | 조건             | 결과                                               |
| ------------- | ---------------- | -------------------------------------------------- |
| 프롬프트 품질 | 고품질 vs 없음   | 고품질: 89.19%, 없음: 급격히 저하                  |
| 데이터 손상   | JPEG/노이즈/블러 | JPEG/노이즈가 가장 큰 영향, 블러: 대의적 성능 저하 |
| 프레임 손실   | EFP 적용 전/후   | EndoVis17: 소폭 저하, EndoVis18: 향상              |
| 도메인        | EndoVis→RoboTool | 일반화 실패: 81.6%→66.6%                           |

### 5. 실험 설계의 한계 및 비교상의 주의점

**1) 데이터셋 의존성**

| 문제                   | 설명                       | 영향                  |
| ---------------------- | -------------------------- | --------------------- |
| EndoVis-2017 과다 사용 | 62 개 논문 중 20+ 논문     | 특정 데이터셋 편향    |
| 비공개 데이터셋        | 40 개 비공개 데이터셋 사용 | 재현성 저해           |
| ex-vivo 중심           | 돼지 모델 기반 데이터      | 실제 임상 환경과 차이 |
| 인간 데이터 부족       | 대부분 돼지/시뮬레이션     | 일반화 능력 불확실    |

**2) 비교 조건의 불일치**

| 문제             | 설명                             | 영향                |
| ---------------- | -------------------------------- | ------------------- |
| 손실 함수 차이   | CE/Dice/Cross-Entropy/Dice 혼합  | 공정한 비교 어려움  |
| 학습 설정 불명확 | 학습률, 배치 크기, 에포크 불일치 | 재현 어려움         |
| 아블레이션 부재  | 일부 논문: 모듈별 기여도 없음    | 성능 원인 파악 불가 |
| 상이한 평가 지표 | IoU/DSC/MI_DSC 혼용              | 직접 비교 불가      |

**3) 일반화 한계**

| 한계             | 설명                             | 예시                       |
| ---------------- | -------------------------------- | -------------------------- |
| 도메인 격차      | EndoVis→RoboTool: 13.8%p 하락    | 도구-조직 상호작용 차이    |
| 실시간 기준 미달 | 24 FPS 모델 (<30 FPS)            | 실시간 처리 기준 미충족    |
| 클래스 불균형    | Shaft(44.9%) vs Background(>95%) | 일부 클래스 과소대표       |
| 도구 확장성      | 소량 클래스 (36~90 개)           | 얇은/유사 외형 기구 어려움 |

**4) 평가 지표의 한계**

| 지표            | 문제                         | 대안 필요               |
| --------------- | ---------------------------- | ----------------------- |
| IoU/Dice        | 클래스 불균형 민감           | Balanced Accuracy       |
| 단일 지표       | 다중 지표 (IoU/Dice) 필요    | 두 지표 동시 보고       |
| FPS 보고 부재   | 32 개 논문 중 속도 보고 적음 | 모든 논문에서 속도 필수 |
| 데이터셋 불일치 | 훈련/테스트 데이터셋 차이    | 교차 데이터셋 검증      |

### 6. 결과 해석의 경향

**1) 저자들의 공통 해석 경향**

| 해석 유형     | 대표 문구                           | 실제 관찰                                     |
| ------------- | ----------------------------------- | --------------------------------------------- |
| SOTA 주장     | "압승", "최고", "압도적"            | 데이터셋 의존성 강, 교차 검증 시 하락         |
| 모듈 기여도   | "핵심 역할", "결정적"               | Ablation 실험 부재 시 과대평가                |
| 일반화 가능성 | "강건성 입증", "실제 데이터 일반화" | Cross-dataset: 66.6%~83.3% (13.8%p~21%p 하락) |
| 데이터 효율성 | "소량 주석으로", "합성 데이터로"    | 73.51%→83.15% (실제 대비 85% 수준)            |
| 실시간 처리   | "실시간 달성", "FPS 30+"            | 24~38 FPS (30 FPS 기준: 일부 미달)            |

**2) 해석 과대평가 사례**

| 해석                 | 실제 결과                                            | 설명                          |
| -------------------- | ---------------------------------------------------- | ----------------------------- |
| "실시간 처리"        | 38~49 FPS (일부 24 FPS)                              | 30 FPS 기준: 일부 모델 미달   |
| "실제 데이터 일반화" | EndoVis→RoboTool: 13.8%p 하락                        | 도메인 격차 존재              |
| "완전 지도 능가"     | 준지도 (10~20% 주석) 에서 완전 지도 (100% 주석) 초과 | 희소 주석 환경에서만 가능     |
| "완전 비지도 SOTA"   | 83.77% vs 지도 (89.61%): -5.84% 포인트               | 지도 학습에 근접하지만 불완전 |
| "프롬프트 강건성"    | 고품질 프롬프트: 89.19%, 없음: 급격히 저하           | 프롬프트 품질 의존성 강       |

**3) 성공 요인 해석**

| 방법론        | 해석                 | 검증                          |
| ------------- | -------------------- | ----------------------------- |
| MAFA          | "회전된 시각적 단서" | mIOU 분산 10%→2.0% 로 입증    |
| Dual-Memory   | "시공간 정보 통합"   | ELA+AGA: +7.38% 입증          |
| SAM 튜닝      | "도메인 격차 해소"   | SurgicalSAM: SOTA 능가 입증   |
| Frame Pruning | "redundancy 제거"    | Fine-tuning 병행 시 3 배 속도 |
| Text Prompt   | "오픈-셋 분할"       | 새로운 도구 적응성 입증       |

### 7. 종합 정리

**평가 방식**
전체 20 여 논문이 IoU/mIoU 를 주된 지표로 사용 (90%+), Dice/DSC 를 보조 지표로 활용 (80%+). 처리 속도 (FPS) 는 실시간 처리 능력을 평가하는 데 사용 (25%+) 이며, FLOPs/파라미터 수는 계산 효율성 평가를 위해 사용 (10%+). 손실 함수 조합 (CE+Dice, Hybrid Loss) 은 클래스 불균형 완화에 활용되고, ablation study 를 통한 모듈별 기여도 검증이 일반적인 검증 방식이다.

**성능이 향상되는 조건**

1) **시간적 정보 활용**: 준지도/비지도 학습에서 단일 프레임 대비 시퀀스 정보 활용이 성능 향상 (IoU +2.68%~6.12%).
2) **데이터 효율성**: 합성 데이터, 소량 주석 환경에서 성능 유지 (EndoVis2018 73.51% DSC, 10~20% 주석에서 완전 지도 능가).
3) **프롬프트 품질**: 고품질 프롬프트에서 제로샷 일반화 가능 (89.19% IoU), 프롬프트 없음 시 급격한 성능 저하.
4) **도메인 간 일반화**: EndoVis2017→2018: -15~-25%p 하락, 실제 데이터 일반화 어려움.
5) **실시간 처리**: ELA+AGA 두 메모리 모듈 병행 시 38 FPS 달성 (30 FPS 기준 충족), EFP 적용 시 3 배 속도 향상 (29→86 FPS).

**결과 일관성**

- **일관성 있는 결과**: MAFA, Dual-Memory, SAM 튜닝 등 특정 아키텍처의 모듈별 기여도 일관성 있음 (ELA:+5.74%, AGA:+5.13%).
- **일관성 없는 결과**: 교차 데이터셋 일반화에서 EndoVis2017→RoboTool: 81.6%→66.6% (13.8%p 하락) 등 데이터셋 의존성.
- **데이터셋 의존성**: EndoVis2017 과다 사용 (62 개 논문 중 20+), 비공개 데이터셋 (40 개) 은 재현성 저해.

**결론**
Surgical Instrument Segmentation 분야에서 성능은 데이터셋, 학습 전략, 평가 지표에 크게 의존한다. EndoVis2017 의 압도적 사용으로 특정 데이터셋 편향이 존재하며, 교차 데이터셋 일반화 시 13.8%~25%p 하락이 일관되게 관찰된다. 지도 학습은 높은 정확도 (89~90%) 를 달성하나 주석 병목 문제가 있고, 완전 비지도 학습은 83.77% 로 지도 학습에 근접하나 완전한 성능은 달성하지 못한다. 실시간 처리는 30 FPS 기준의 일부 모델이 미달하며, 도메인 일반화 능력은 여전히 개선이 필요하다.
