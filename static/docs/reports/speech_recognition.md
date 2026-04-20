# Speech Recognition

## 서론

### 1. 연구 배경

본 보고서는 2010 년 이후 음성 인식 (Speech Recognition) 분야에서 발표된 30 여 편의 핵심 논문을 체계적으로 분석하여, 음성 인식 연구의 기술적 진보 경로와 방법론적 발전을 종합적으로 조망합니다. 연구의 범위는 음성 인식 모델의 아키텍처 설계, 학습 방법론, 처리 전략, 그리고 다양한 환경 및 도메인 적용이라는 4 차원 축을 중심으로 구성됩니다. BLSTM-CTC 에서 시작하여 Transformer, Conformer, 하이브리드 아키텍처로 진화하는 모델 구조의 변천, 자기 지도/준지도 학습에서 약한 지도 학습으로 확장되는 학습 패러다임의 전환, 스트리밍 처리 및 효율성 최적화, 그리고 잡음 강건성부터 온디바이스 적용까지의 환경적 요구사항 대응까지를 포함합니다.

### 2. 문제의식 및 분석 필요성

음성 인식 연구는 초기 HMM-GMM 기반 시스템에서 딥러닝 기반 엔드투엔드 모델로 전환되는 과정에서 구조적, 방법론적 변화를 겪었습니다. 그러나 연구 분야가 다양해지면서 각 논문별로 적용되는 아키텍처, 학습 전략, 평가 방식이 상이하여 전체적인 기술 발전 경로를 파악하기에는 정보가 분산되어 있습니다. 특히 모델 성능, 계산 효율성, 데이터 효율성, 제로샷 전이 능력, 환경 강건성 등 여러 목표가 상충하는 상황에서 어떤 설계가 실제 적용에 적합한지는 단순한 벤치마크 수치 비교만으로 판단하기 어렵습니다. 본 보고서는 이러한 맥락에서 기술적 진보와 방법론적 전환을 통합적으로 정리하며, 관련 연구의 비교 분석을 통해 음성 인식 시스템 설계에 실질적으로 활용 가능한 통찰을 제시하고자 합니다.

### 3. 보고서의 분석 관점

본 보고서는 다음과 같은 3 개의 주요 관점에서 문헌을 분석합니다.

**연구체계 분류**: 음성 인식 연구의 핵심 기여 요소를 4 차원 축 (모델 아키텍처, 학습 방법, 처리 전략, 적용 환경) 으로 구분하여 체계화합니다.

**방법론 분석**: 아키텍처 구조, 학습 전략, 디코딩 방식에 따른 차이와 트레이드오프를 비교 분석하며, 아키텍처의 진화 패턴, 성능과 자원 효율성의 절충 관계, 데이터 규모에 따른 성능 향상을 체계적으로 정리합니다.

**실험결과 분석**: 주요 데이터셋과 평가 환경별 실험 결과를 정렬하고, 성능 패턴과 한계를 함께 고찰합니다.

### 4. 보고서 구성

본 보고서의 구성은 다음과 같습니다:

- **1 장. 연구체계 분류**: 모델 아키텍처, 학습 방법론, 처리 전략, 적용 환경이라는 4 차원 축을 기준으로 60 편의 논문을 체계적으로 분류합니다. 모델 아키텍처 설계 연구, 자기/준지도 학습 연구, 스트리밍 및 효율성 최적화 연구, 도메인 특화 및 환경 강건성 연구로 구분하여 각 분야의 대표적 논문을 정리합니다.

- **2 장. 방법론 분석**: 분류된 연구들의 방법론적 공통점과 차이점을 분석합니다. 인코더-디코더, 트랜스듀서, 하이브리드 등 핵심 설계 패턴과 학습 패러다임을 비교하며, 성능-효율성 트레이드오프와 데이터 효율성 경향에 대한 종합적 통찰을 제공합니다.

- **3 장. 실험결과 분석**: LibriSpeech, TIMIT, KsponSpeech 등 주요 데이터셋을 기준으로 벤치마크 결과를 정리하고, 아키텍처 진화, 자기지도 학습 효과, 데이터 규모 영향 등 성능 패턴과 일반화 한계를 고찰합니다. 실험 설계 시 주의해야 할 비교 조건의 불일치와 평가 지표의 한계도 함께 논의합니다.

## 1장. 연구체계 분류

### 1.1 연구 분류 체계 수립 기준

본 연구체계는 음성 인식 연구의 **모델 아키텍처**,**학습 방법**,**처리 전략**,**적용 환경**이라는 4 차원 축을 기준으로 설계되었습니다. 먼저 각 논문의 핵심 기여 요소를 분석한 뒤, 가장 대표적인 한 축에 할당하는 방식으로 범주화를 수행합니다. 특정 논문이 여러 범주에 걸쳐도 본 연구는 가장 핵심적인 기여를 기준으로 분류하되, 일부 논문에 대해서는 분류 근거를 명확히 밝히지 않고 연구 대상의 기술적 특징을 간략히 서술하는 보수적 원칙을 따릅니다.

### 1.2 연구 분류 체계

#### 1.2.1. 모델 아키텍처 및 구조적 설계

이 대분류는 음성 인식 모델의 핵심 구성 요소를 설계하거나 새로운 아키텍처를 제안한 연구를 포괄합니다. RNN, CNN, Transformer, Conformer 등 모델 구조 자체가 주된 초점이 되며, 하이브리드 설계, 스트리밍 지원, 효율성 개선 등이 포함됩니다.

| 분류                                              | 논문명                                                                                                                   | 분류 근거                                                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| 1.2.1.1 > BLSTM-CTC                               | Phoneme recognition in TIMIT with BLSTM-CTC (2008)                                                                       | 단일 BLSTM 순환 신경망에 CTC 손실을 적용하여 기존 최고 성능 시스템과 통계적으로 동등한 성능 입증       |
| 1.2.1.2 > RNN-Transducer                          | Sequence Transduction with Recurrent Neural Networks (2012)                                                              | CTC를 확장하여 입력-출력 및 출력-출력 의존성을 공동으로 학습하는 RNN 기반 종단 간 시퀀스 변환 아키텍처 |
| 1.2.1.3 > LAS                                     | Listen, Attend and Spell (2015)                                                                                          | 어텐션 기반 디코더를 통한 종단 간 학습과 문자 단위 시퀀스 생성 모델 제안                               |
| 1.2.1.4 > Deep Speech-2                           | Deep Speech 2: End-to-End Speech Recognition in English and Mandarin (2015)                                              | RNN 인코더-디코더 아키텍처에 배치 정규화와 커리큘럼 학습을 적용한 대규모 엔드투엔드 시스템             |
| 1.2.1.5 > Wav2Letter                              | Wav2Letter: an End-to-End ConvNet-based Speech Recognition System (2016)                                                 | 음소 정렬 없이 글자 단위 직접 전사를 위한 스트라이딩 컨볼루션 기반 ConvNet 아키텍처                    |
| 1.2.1.6 > Joint CTC-Attention (2016)              | Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning (2016)                                 | CTC와 어텐션 디코더를 공유 인코더로 통합한 다중 작업 학습 아키텍처로 정렬 문제 해결                    |
| 1.2.1.7 > Self-Attention Networks                 | Self-Attention Networks for Connectionist Temporal Classification in Speech Recognition (2019)                           | CTC 프레임워크에 완전한 셀프-어텐션 인코더를 통합한 하이브리드 아키텍처                                |
| 1.2.1.8 > Transformers with Convolutional Context | Transformers with convolutional context for ASR (2019)                                                                   | 인코더 합성곱 레이어와 디코더 컨볼루션으로 Transformer의 위치 정보 처리 및 최적화 안정화               |
| 1.2.1.9 > Self-Attention Transducers              | Self-Attention Transducers for End-to-End Speech Recognition (2019)                                                      | RNN을 자기-어텐션 블록으로 대체하고 청크-플로우 메커니즘을 통한 병렬화 및 장기 의존성 개선             |
| 1.2.1.10 > SA-Transducer                          | Transformer-Transducer: End-to-End Speech Recognition with Self-Attention (2019)                                         | 인과 컨볼루션과 문맥 제한으로 트랜스포머를 트랜스듀서로 변환한 하이브리드 아키텍처                     |
| 1.2.1.11 > Very Deep Transformers                 | Very Deep Self-Attention Networks for End-to-End Speech Recognition (2019)                                               | 순환 구조 없이 순수 자기 주의 기반으로 깊은 ASR 네트워크 구축 및 확률적 잔차 연결 전략                 |
| 1.2.1.12 > Conformer                              | Conformer: Convolution-augmented Transformer for Speech Recognition (2020)                                               | 트랜스포머의 전역적 어텐션과 CNN 의 지역적 컨볼루션을 샌드위치 형태로 통합한 하이브리드 블록           |
| 1.2.1.13 > Streaming Transformer                  | Streaming automatic speech recognition with the transformer model (2020)                                                 | Look-ahead 프레임 기반 지연 시간 제어와 CTC-TA 공동 훈련을 통한 스트리밍 Transformer 아키텍처          |
| 1.2.1.14 > Conv-Transformer Transducer            | Conv-Transformer Transducer: Low Latency, Low Frame Rate, Streamable End-to-End Speech Recognition (2020)                | Transducer 프레임워크와 인터리브드 컨볼루션을 결합한 단방향 Transformer 아키텍처                       |
| 1.2.1.15 > Transformer Transducer (Y-Model)       | Transformer Transducer: One Model Unifying Streaming and Non-streaming Speech Recognition (2020)                         | 마지막 레이어에 가변 문맥을 적용하여 단일 모델로 스트리밍/비스트리밍 통합하는 Y-모델 아키텍처          |
| 1.2.1.16 > Efficient Conformer                    | Efficient conformer: Progressive downsampling and grouped attention for automatic speech recognition (2021)              | 프로그레시브 다운샘플링과 그룹화된 어텐션 메커니즘을 통한 계산 복잡도 감소 아키텍처                    |
| 1.2.1.17 > U2 Architecture                        | WeNet: Production Oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit (2021)                      | CTC와 AED 를 결합한 통합 모델로 스트리밍/비스트리밍 단일화가 가능한 U2 아키텍처                        |
| 1.2.1.18 > Speech Swin-Transformer                | Speech Swin-Transformer: Exploring a Hierarchical Transformer with Shifted Windows for Speech Emotion Recognition (2024) | Shifted Windows 기법을 통해 패치 병합과 계층적 특징 표현을 구현한 Transformer 아키텍처                 |
| 1.2.1.19 > Online Hybrid CTC/Attention            | Online Hybrid CTC/Attention End-to-End Automatic Speech Recognition Architecture (2023)                                  | 하이브리드 CTC/어텐션 아키텍처의 모든 오프라인 구성요소를 스트리밍 대안으로 대체한 완전 온라인 솔루션  |
| 1.2.1.20 > Efficient Conformer AV                 | Audio-Visual Efficient Conformer for Robust Speech Recognition (2023)                                                    | 효율적 패치 어텐션과 중간 CTC 잔여 모듈을 통해 오디오-시각 모달리티 융합 아키텍처                      |
| 1.2.1.21 > Fast Conformer                         | Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition (2023)                                  | 8 배 다운샘플링과 전역 어텐션 전략을 통한 긴 오디오 시퀀스 처리 아키텍처                               |
| 1.2.1.22 > Whisper Streaming                      | Turning Whisper into Real-Time Transcription System (2023)                                                               | Whisper 모델의 실시간 스트리밍 기능을 구현하는 스트리밍 전사 정책 아키텍처                             |
| 1.2.1.23 > U2++                                   | WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit (2022)                                                  | 양방향 어텐션 디코더 기반 U2++ 프레임워크와 WFST 기반 LM 통합 아키텍처                                 |
| 1.2.1.24 > U2++ IRIS                              | End-to-End Integration of Speech Recognition, Speech Enhancement, and Self-Supervised Learning Representation (2022)     | Conv-TasNet 기반 SE 모듈, WavLM 기반 SSLR 모듈, Transformer 기반 ASR 모듈을 결합한 IRIS 아키텍처       |
| 1.2.1.25 > Speech Swin-Transformer                | Speech Swin-Transformer: Exploring a Hierarchical Transformer with Shifted Windows for Speech Emotion Recognition (2024) | Shifted Windows 기법을 통해 계층적 특징 표현을 구현한 Transformer 아키텍처                             |
| 1.2.1.26 > CATT                                   | Context-Aware Transformer Transducer for Speech Recognition (2021)                                                       | 훈련 시 통합된 컨텍스트 바이어싱 네트워크를 통한 희귀 단어 인식 개선 아키텍처                          |
| 1.2.1.27 > Two-Pass Whisper                       | Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding (2025)                                           | U2(Unified Two-pass) 구조로 Whisper를 스트리밍 ASR 모델로 재구성한 아키텍처                            |
| 1.2.1.28 > SLM                                    | Language Model Can Listen While Speaking (2024)                                                                          | 발화 채널과 듣기 채널을 동시에 갖춘 종단 간 단일 모델인 대화형 음성 언어 모델 아키텍처                 |
| 1.2.1.29 > Speech-LLM                             | Chain-of-Thought Prompting for Speech Translation (2024)                                                                 | 인코더-디코더 구조의 Speech-LLM(Megatron-T5 기반)을 활용한 음성 번역 시스템                            |

#### 1.2.2. 학습 방법론 및 자기/준지도 전략

이 대분류는 모델 아키텍처를 설계하기보다 학습 패러다임, 손실 함수, 데이터 활용 전략이 연구의 핵심이 되는 연구들을 포괄합니다. 자기 지도 학습 (SSL), 준지도 학습, 전이 학습, 강화 학습, 다중 작업 학습 등의 방법이 포함됩니다.

| 분류                          | 논문명                                                                                                               | 분류 근거                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1.2.2.1 > SSL                 | Deep Representation Learning in Speech Processing: Challenges, Recent Advances, and Future Trends (2020)             | 음성 처리 3 대 응용 (ASR/SR/SER) 전반에 걸친 표현 학습 기술의 아키텍처와 학습 기법 차원에서 체계적으로 분류·조망 |
| 1.2.2.2 > Semi-Supervised     | Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition (2020)                               | 대규모 ASR 모델에서 레이블 없는 데이터를 활용하기 위한 자기 지도/준지도 하이브리드 학습 파이프라인               |
| 1.2.2.3 > Unsupervised SE     | Unsupervised Speech Enhancement with speech recognition embedding and disentanglement losses (2021)                  | 혼합 손실 함수 구조를 통해 실제 잡음 데이터의 도메인 불일치를 해소하는 준지도 학습 접근                          |
| 1.2.2.4 > Wav2vec-C           | Wav2vec-C: A Self-supervised Model for Speech Representation Learning (2021)                                         | VQ-VAE 의 일관성 개념을 자기 지도 학습에 접목한 모델로, 코드북 활용률 향상을 위한 일관성 손실 추가               |
| 1.2.2.5 > Wav2Letter SSL      | Listening while Speaking: Speech Chain by Deep Learning (2017)                                                       | ASR과 TTS를 상호 연결한 폐쇄 루프 아키텍처와 자기 지도 학습 전략을 통해 레이블링되지 않은 데이터 활용            |
| 1.2.2.6 > Data Augmentation   | Self-Training for End-to-End Speech Recognition (2019)                                                               | 시퀀스-투-시퀀스 모델의 오류에 특화된 필터링 기법과 샘플 앙상블을 결합한 자기 훈련 접근법                        |
| 1.2.2.7 > SSLR                | End-to-End Integration of Speech Recognition, Speech Enhancement, and Self-Supervised Learning Representation (2022) | SE/ASR 모듈을 사전 훈련 모델로 초기화한 후 SE/ASR만 공동 미세 조정하는 종단 간 통합 구조                         |
| 1.2.2.8 > Multi-Task          | Unified Speech-Text Pre-training for Speech Translation and Recognition (2022)                                       | 다중 양식 데이터를 통합하는 다중 작업 학습 프레임워크로 하위 작업 간 간섭을 완화                                 |
| 1.2.2.9 > Weak Supervision    | Robust Speech Recognition via Large-Scale Weak Supervision (2022)                                                    | 방대한 약한 지도 학습 데이터를 활용한 스케일링이 자기 지도 학습 기법보다 효과적임을 입증                         |
| 1.2.2.10 > Federated Learning | Federated Learning for ASR based on Wav2vec 2.0 (2023)                                                               | SSL 사전 학습 모델을 FL 환경에 통합한 ASR 연구로, 데이터 프라이버시 보호와 성능 개선을 달성                      |
| 1.2.2.11 > Continual Learning | Continual Learning in Machine Speech Chain Using Gradient Episodic Memory (2024)                                     | TTS 생성 의사 샘플을 에피소드 메모리로 활용하여 GEM 알고리즘의 그라디언트 제약을 수행                            |
| 1.2.2.12 > Federated Learning | Federated Learning for ASR based on Wav2vec 2.0 (2023)                                                               | SSL 사전학습 모델을 FL 환경에 통합한 ASR 연구로, 데이터 프라이버시 보호와 성능 개선을 달성                       |
| 1.2.2.13 > Multi-Task         | Deep Transfer Learning for Automatic Speech Recognition: Towards Better Generalization (2023)                        | DTL 기법을 귀납적/전이적/적대적 등으로 구조적 분류한 포괄적 조사 연구                                            |
| 1.2.2.14 > SSL                | Deep Representation Learning in Speech Processing: Challenges, Recent Advances, and Future Trends (2020)             | 음성 처리 3 대 응용 (ASR/SR/SER) 전반에 걸친 표현 학습 기술의 아키텍처와 학습 기법 차원에서 체계적으로 분류·조망 |
| 1.2.2.15 > Semi-Supervised    | Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition (2020)                               | 대규모 ASR 모델에서 레이블 없는 데이터를 활용하기 위한 자기 지도/준지도 하이브리드 학습 파이프라인               |
| 1.2.2.16 > Unsupervised SE    | Unsupervised Speech Enhancement with speech recognition embedding and disentanglement losses (2021)                  | 혼합 손실 함수 구조를 통해 실제 잡음 데이터의 도메인 불일치를 해소하는 준지도 학습 접근                          |
| 1.2.2.17 > Wav2vec-C          | Wav2vec-C: A Self-supervised Model for Speech Representation Learning (2021)                                         | VQ-VAE 의 일관성 개념을 자기 지도 학습에 접목한 모델로, 코드북 활용률 향상을 위한 일관성 손실 추가               |
| 1.2.2.18 > SSL                | Listening while Speaking: Speech Chain by Deep Learning (2017)                                                       | ASR과 TTS 를 상호 연결한 폐쇄 루프 아키텍처와 자기 지도 학습 전략을 통해 레이블링되지 않은 데이터 활용           |
| 1.2.2.19 > Self-Training      | Self-Training for End-to-End Speech Recognition (2019)                                                               | 시퀀스-투-시퀀스 모델의 오류에 특화된 필터링 기법과 샘플 앙상블을 결합한 자기 훈련 접근법                        |
| 1.2.2.20 > SSLR               | End-to-End Integration of Speech Recognition, Speech Enhancement, and Self-Supervised Learning Representation (2022) | SE/ASR 모듈을 사전 훈련 모델로 초기화한 후 SE/ASR 만 공동 미세 조정하는 종단 간 통합 구조                        |
| 1.2.2.21 > Multi-Task         | Unified Speech-Text Pre-training for Speech Translation and Recognition (2022)                                       | 다중 양식 데이터를 통합하는 다중 작업 학습 프레임워크로 하위 작업 간 간섭을 완화                                 |
| 1.2.2.22 > Weak Supervision   | Robust Speech Recognition via Large-Scale Weak Supervision (2022)                                                    | 방대한 약한 지도 학습 데이터를 활용한 스케일링이 자기 지도 학습 기법보다 효과적임을 입증                         |
| 1.2.2.23 > Federated Learning | Federated Learning for ASR based on Wav2vec 2.0 (2023)                                                               | SSL 사전 학습 모델을 FL 환경에 통합한 ASR 연구로, 데이터 프라이버시 보호와 성능 개선을 달성                      |
| 1.2.2.24 > Continual Learning | Continual Learning in Machine Speech Chain Using Gradient Episodic Memory (2024)                                     | TTS 생성 의사 샘플을 에피소드 메모리로 활용하여 GEM 알고리즘의 그라디언트 제약을 수행                            |
| 1.2.2.25 > Multi-Task         | Deep Transfer Learning for Automatic Speech Recognition: Towards Better Generalization (2023)                        | DTL 기법을 귀납적/전이적/적대적 등으로 구조적 분류한 포괄적 조사 연구                                            |

#### 1.2.3. 처리 전략 및 최적화 기법

이 대분류는 모델 아키텍처나 학습 방법론 자체보다는 시스템 효율성, 지연 시간, 스트리밍 처리, 계산 복잡도 감소 등 운영 전략 및 최적화 기법에 초점을 맞춘 연구를 포함합니다.

| 분류                           | 논문명                                                                                                                   | 분류 근거                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 1.2.3.1 > CTC/Decoding         | Phoneme recognition in TIMIT with BLSTM-CTC (2008)                                                                       | 단일 BLSTM-CTC 신경망이 기존 가장 성능의 앙상블 시스템과 통계적으로 동등한 음소 오류율 달성         |
| 1.2.3.2 > Online/Streaming     | FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization (2020)                                   | 트랜스듀서의 시퀀스 수준 (전향-후향 확률) 최적화에 직접 적용되는 정규화 방법을 통해 지연 감소       |
| 1.2.3.3 > Streaming Efficiency | Developing Real-time Streaming Transformer Transducer for Speech Recognition on Large-scale Dataset (2020)               | 어텐션 수용 필드를 청크 단위로 제한하여 트랜스포머 레이어 깊이에 따른 지연 시간 선형 증가 문제 해결 |
| 1.2.3.4 > Data Augmentation    | SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition (2019)                                   | ASR 도메인에서 신경망 특징 입력에 직접 적용되는 시간/주파수 마스킹 기반 데이터 증강 기법            |
| 1.2.3.5 > Domain Adaptation    | A Study of Enhancement, Augmentation, and Autoencoder Methods for Domain Adaptation in Distant Speech Recognition (2018) | 레이블 요구 수준과 성능 효율성을 기준으로 도메인 적응 방법론의 데이터 효율성 스펙트럼을 제시        |
| 1.2.3.6 > Hybrid Tokenizer     | Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding (2025)                                           | U2(Unified Two-pass) 구조로 Whisper 를 스트리밍 ASR 모델로 재구성한 아키텍처                        |
| 1.2.3.7 > Streaming Whisper    | Turning Whisper into Real-Time Transcription System (2023)                                                               | Whisper 모델의 실시간 스트리밍 기능을 구현하는 스트리밍 전사 정책 아키텍처                          |
| 1.2.3.8 > Two-Pass             | Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding (2025)                                           | U2 구조로 CTC partial hypothesis 와 attention rescoring 을 통한 데이터 효율적 아키텍처              |
| 1.2.3.9 > Domain Adaptation    | A Study of Enhancement, Augmentation, and Autoencoder Methods for Domain Adaptation in Distant Speech Recognition (2018) | 레이블 요구 수준과 성능 효율성을 기준으로 도메인 적응 방법론의 데이터 효율성 스펙트럼을 제시        |
| 1.2.3.10 > Streaming           | Streaming automatic speech recognition with the transformer model (2020)                                                 | Look-ahead 프레임 기반 지연 시간 제어와 CTC-TA 공동 훈련을 통한 스트리밍 Transformer 아키텍처       |
| 1.2.3.11 > Hybrid Tokenizer    | Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding (2025)                                           | U2 구조로 CTC partial hypothesis 와 attention rescoring 을 통한 데이터 효율적 아키텍처              |
| 1.2.3.12 > Streaming Whisper   | Turning Whisper into Real-Time Transcription System (2023)                                                               | Whisper 모델의 실시간 스트리밍 기능을 구현하는 스트리밍 전사 정책 아키텍처                          |
| 1.2.3.13 > Domain Adaptation   | A Study of Enhancement, Augmentation, and Autoencoder Methods for Domain Adaptation in Distant Speech Recognition (2018) | 레이블 요구 수준과 성능 효율성을 기준으로 도메인 적응 방법론의 데이터 효율성 스펙트럼을 제시        |

#### 1.2.4. 적용 환경 및 도메인 특화

이 대분류는 특정 환경 조건 (잡음, 멀티모달, 저자원, 화자 인식 등) 또는 도메인 (음성 번역, 감정 인식, 시각 음성 인식 등) 에 특화된 응용 연구를 포함합니다.

| 분류                                | 논문명                                                                                                 | 분류 근거                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| 1.2.4.1 > Environmental Robustness  | Deep Learning for Environmentally Robust Speech Recognition: An Overview of Recent Developments (2017) | 채널 수와 ASR 처리 단계별(DNN 매핑/마스킹, 모델 적응, 통합 훈련, 빔포밍/포스트필터)의 이중 차원 분류 체계                 |
| 1.2.4.2 > Multi-channel             | Multichannel End-to-end Speech Recognition (2017)                                                      | 음성 향상과 인식을 단일 미분 가능 그래프로 연결하고 ASR 목표에 따라 공동 최적화하는 프레임워크                            |
| 1.2.4.3 > Multi-modal               | Audio-Visual Efficient Conformer for Robust Speech Recognition (2023)                                  | 효율적 패치 어텐션과 중간 CTC 잔여 모듈을 통해 오디오-시각 모달리티의 효과적인 융합과 강건성 확보                         |
| 1.2.4.4 > Speaker Recognition       | A Machine of Few Words -- Interactive Speaker Recognition with Reinforcement Learning (2020)           | 강화 학습을 통해 화자 인식 문제를 대화형 상호작용 게임으로 정형화하고 질문자 모듈이 개인화된 단어 선택을 통한 정확도 개선 |
| 1.2.4.5 > Speaker Diarization       | End-to-End Speaker Diarization Conditioned on Speech Activity and Overlap Detection (2021)             | 확률적 연쇄 법칙 기반 하위 태스크 의존성 모델링을 통해 SAD 와 OD 를 효율적으로 결합하는 EEND 프레임워크                   |
| 1.2.4.6 > Low-resource              | Speech Synthesis as Augmentation for Low-resource ASR (2020)                                           | 저자원 환경에서 음성 합성을 ASR 데이터 증강원천으로 활용한 실험적 연구로, 세 가지 합성기 접근법 비교                      |
| 1.2.4.7 > Low-resource ASR          | Robust Speech Recognition via Large-Scale Weak Supervision (2022)                                      | 방대한 약한 지도 학습 데이터를 활용한 스케일링이 자기 지도 학습 기법보다 효과적임을 입증                                  |
| 1.2.4.8 > Speech Translation        | Chain-of-Thought Prompting for Speech Translation (2024)                                               | ASR 가설을 LLM 입력으로 주입하는 CoT 프롬프팅 방식으로 음성 번역 성능을 개선                                              |
| 1.2.4.9 > Visual Speech Recognition | Large-Scale Visual Speech Recognition (2018)                                                           | 음소 예측 단계와 단어 디코딩 단계를 분리하여 어휘 확장 유연성을 제공하는 대규모 VSR 시스템                                |
| 1.2.4.10 > Speech Translation       | Unified Speech-Text Pre-training for Speech Translation and Recognition (2022)                         | 다중 양식 데이터를 통합하는 다중 작업 학습 프레임워크로 하위 작업 간 간섭을 완화                                          |
| 1.2.4.11 > Low-resource TTS         | Exploring Speech Enhancement for Low-resource Speech Synthesis (2023)                                  | 잡음 제거를 통한 ASR 코퍼스 재활용을 저자원 TTS 학습에 적용하는 파이프라인 연구                                           |

### 1.3 종합 정리

본 연구체계는 음성 인식 연구의 진화 경로를 4 차원 축으로 구분하여 체계화했습니다. 모델 아키텍처 측면에서는 BLSTM-CTC 에서 시작해 RNN 트랜스듀서, Self-Attention, Conformer, 트랜스포머 하이브리드 등으로 발전했으며, 학습 방법론 측면에서는 지도 학습 중심에서 자기 지도, 준지도, 연합 학습, 강화 학습 등 다양한 패러다임이 도입되었습니다. 처리 전략에서는 데이터 증강, 스트리밍 최적화, 도메인 적응 기법들이 발전되었고, 적용 환경 측면에서는 환경 강건성, 멀티모달, 저자원, 화자 인식 등 다양한 도메인이 개척되었습니다. 이러한 4 차원 축은 음성 인식 연구의 기술적 진보와 응용 영역 확장을 포괄적으로 반영하고 있습니다.

**문서 개요:**

제공된 30 여 편의 논문 (2010 년부터 2025 년까지) 을 체계적으로 분석하여 다음과 같이 구성했습니다:

- **6 개의 주요 계열 분류:** 트랜스듀서, 인코더-디코더 어텐션, Transformer, 자기 지도 학습, 효율성 최적화, 데이터 증강/준지도, 멀티모달, 연속 학습 등
- **30 개의 핵심 설계 패턴:** 인코더-디코더, 트랜스듀서, 하이브리드, 멀티모달, 자기 지도 학습, 빔 서치 등
- **방법론 비교 분석:** 아키텍처 구조, 학습 전략, 디코딩 방식별 차이점과 트레이드오프
- **시간적 진화 흐름:** 초기 RNN 기반 → Transformer 도입 → 하이브리드 통합 → 효율성 최적화 → 다양한 활용
- **종합 표:** 각 계열별 대표 논문, 핵심 특징, 차별점 정리

## 3장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 주요 데이터셋 유형

제공된 실험 결과들을 분석하면 다음과 같은 데이터셋 유형들이 주로 사용되었습니다.

| 데이터셋 분류    | 대표 데이터셋            | 용도                 | 특징                                   |
| ---------------- | ------------------------ | -------------------- | -------------------------------------- |
| 표준 평가 코퍼스 | LibriSpeech              | ASR 성능 벤치마크    | 960시간 (clean+other), 9600시간 (full) |
|                  | TIMIT                    | 음소 인식, 화자 지문 | 6300 문장, 630 화자, 39 음소           |
|                  | Switchboard              | 하이브리드 ASR 비교  | 300 시간                               |
| 한국어 ASR       | KsponSpeech              | 한국어 ASR 표준      | 1,000시간 대화 코퍼스                  |
|                  | AISHELL                  | 중국어 ASR           | AISHELL-1/2 등                         |
|                  | FLEURS                   | 저자원 언어          | 75 언어 포함                           |
| 다국어/다음소    | VoxPopuli, MLS, TED-LIUM | 제로샷 전이, 다국어  |                                        |
| 잡음/강건성      | CHiME-4                  | 단일 채널 잡음 환경  | 시뮬레이션 및 실제 잡음                |
|                  | DNS 2020                 | 음성 향상 평가       | 테스트셋 기준                          |
| 대규모 ASR       | 65,000시간 MS            | 실시간 스트리밍      | 180만 단어                             |
|                  | 680,000시간 약한 지도    | 제로샷 OOD 강건성    | 다국어 다중 작업                       |

#### 평가 환경 유형

| 환경 분류   | 유형           | 대표 논문                                  | 주요 특징                         |
| ----------- | -------------- | ------------------------------------------ | --------------------------------- |
| 온디바이스  | Google Pixel   | E2E Speech Recognition Survey (2023)       | Pixel 4/5/6, <10ms 지연, TPU 사용 |
|             | 엣지/모바일    | Quantization for Whisper (2025), WeNet 2.0 | INT8 양자화, ARM 플랫폼           |
| 온디바이스  | Android        | WeNet (2021)                               | RTF 2배 감소 (0.251→0.114)        |
| 스트리밍    | 실시간 전사    | Turning Whisper into Real-Time (2023)      | 3.3초 지연, LocalAgreement 정책   |
|             | 대화형         | HKUST 만다린 전화                          | 200 시간, 속도 섭동               |
| 잡음 강건성 | 단일 채널 잡음 | End-to-End Integration (2022)              | 3.92% WER 달성                    |
|             | 다중 채널      | Multichannel E2E SR (2017), AVEC (2023)    | CHiME-4 5채널, AMI 8채널          |
|             | OOD 평가       | Weak Supervision (2022)                    | 12 개 영어 ASR 데이터셋           |

#### 비교 대상 및 방식

| 비교 유형                | 대상                           | 설명                                      |
| ------------------------ | ------------------------------ | ----------------------------------------- |
| 전통적 vs 딥러닝         | HMM-GMM vs DNN/RNN/Transformer | 60 년 ASR 역사에서 WER 지속적 감소        |
| 하이브리드 vs 엔드투엔드 | HMM-DNN vs CTC/Attention       | 엔드투엔드 통합 최적화 장점               |
| 아키텍처 간              | RNN vs Transformer             | 병렬화 효율성, 장기 의존성 처리           |
|                          | Transformer vs Conformer       | CNN 지역적 특징 vs 어텐션 전역적 상호작용 |
|                          | 스트리밍 vs 비스트리밍         | 지연 시간 vs 정확도 절충                  |
| 사전학습 vs 미세조정     | Self-supervised vs Fine-tuning | Wav2Vec 2.0, HuBERT 등 SSL 효과           |
| 도메인 적응              | 지도 vs 준지도 vs 자기지도     | 데이터 희소성 극복                        |

#### 주요 평가 지표

| 지표 | 의미                   | 사용 빈도        | 범위        |
| ---- | ---------------------- | ---------------- | ----------- |
| WER  | Word Error Rate        | 가장 일반적      | 1.4%~93%    |
| CER  | Character Error Rate   | 문자 단위 오차율 | 2%~40%      |
| PER  | Phoneme Error Rate     | 음소 단위 오차율 | 1.6%~40%    |
| DER  | Diarization Error Rate | 화자 분할 오차율 | 8.53%~9.39% |
| BLEU | 번역 품질              | 음성 번역 (ST)   | 29.1~76.8   |
| F1   | 화자 식별 정확도       | ISR 성능         | 46%~98%     |

### 2. 주요 실험 결과 정렬

#### LibriSpeech 벤치마크 결과 비교 (test-clean / test-other)

| 논문 (연도)                       | Model               | test-clean WER  | test-other WER | 비교 대상              | 핵심 결과                                  |
| --------------------------------- | ------------------- | --------------- | -------------- | ---------------------- | ------------------------------------------ |
| SpecAugment (2019)                | LAS-6-1280          | 6.8% / 21.1%    | 7.5%           | 이전 하이브리드        | LM 없이 6.8%, LM + 얕은 융합 5.8%          |
| FastEmit (2020)                   | 트랜스듀서          | 3.1%            | 7.5%           | Constrained Alignments | 4.4%→3.1% (1.3%p 개선)                     |
| Conformer (2020)                  | Conformer(L)        | 1.9%            | 3.9%           | Transformer Transducer | 언어 모델 사용 시 SOTA                     |
| Efficient Conformer (2021)        | Efficient Conformer | 3.57%           | 8.99%          | Conformer CTC          | 13M 파라미터, Conformer 대비 추론 29% 단축 |
| FastEmit (2020)                   | 트랜스듀서          | 3.1%            | 7.5%           | Constrained Alignments | 4.4%→3.1% (1.3%p 개선)                     |
| Efficient Conformer (2021)        | Efficient Conformer | 3.57%           | 8.99%          | Conformer CTC          | 2.9 배 낮은 GMACs                          |
| Conformer (2020)                  | Conformer(L)        | 1.9%            | 3.9%           | Transformer Transducer | 언어 모델 사용 시 SOTA                     |
| FastEmit (2020)                   | 트랜스듀서          | 3.1%            | 7.5%           | Constrained Alignments | 4.4%→3.1% (1.3%p 개선)                     |
| Efficient Conformer (2021)        | Efficient Conformer | 3.57%           | 8.99%          | Conformer CTC          | 2.9 배 낮은 GMACs                          |
| Conformer (2020)                  | Conformer(L)        | 1.9%            | 3.9%           | Transformer Transducer | 언어 모델 사용 시 SOTA                     |
| End-to-End Integration (2022)     | WavLM + SE+ASR      | 2.03%           | 3.92%          | 하이브리드 최상        | 37.1% 개선, WavLM alone 4.90%              |
| End-to-End Integration (2022)     | IRIS(WavLM+SE+ASR)  | 2.03%           | 3.92%          | 하이브리드 최상        | 6.25% 대비 37.1% 개선                      |
| Transformers Survey (2023)        | Conformer + LM      | 1.8%            | 3.7%           | 고전 HMM-GMM           | 고전 대비 0.5~1.2%p 개선                   |
| Robust SR Weak Supervision (2022) | Whisper(680k h)     | 2.5%            | N/A            | SOTA ASR               | 미세 조정 없이 2.5% 달성                   |
| Robust SR Weak Supervision (2022) | 12 개 OOD 평균      | 55.2% 오류 감소 | -              | 제로샷 전이            |                                            |
| Deep Transfer Learning (2023)     | 교차-언어 DTL       | 38.6% WER 감소  | -              | 일반 전이              |                                            |
| Deep Transfer Learning (2023)     | 적대적 훈련         | 23% WER 감소    | -              | 도메인 불일치          |                                            |
| AVECS (2023)                      | AV+Neural LM        | 1.8%            | 2.3%           | 오디오 단독            | 4배 적은 훈련 단계로 수렴                  |
| Efficient Conformer (2023)        | Efficient Conformer | 4.99%           | 5.19%          | Conformer              | 2.8 배 빠른 추론 속도                      |
| WeNet 2.0 (2022)                  | U2++ + LM           | -               | 8.42%          | ESPnet                 | LibriSpeech test_other                     |
| WeNet 2.0 (2022)                  | U2++ + LM           | -               | 5%             | -                      | AISHELL-1 에서 n-gram LM                   |

#### 스트리밍 ASR 성능 비교

| 논문 (연도)                           | 지연 시간           | WER (clean/other)  | RTF           | 핵심 기술                  |
| ------------------------------------- | ------------------- | ------------------ | ------------- | -------------------------- |
| Streaming Transformer (2020)          | 2190ms              | 2.8% / 7.2%        | -             | ε^enc=3, ε^dec=18          |
| Transformer Transducer (2020)         | 34초 lookahead      | 4.86%              | 0.07 (2.4s)   | 가변 문맥 훈련             |
| Conv-Transformer Transducer (2020)    | 140ms 룩어헤드      | 3.5% / 8.3%        | -             | 67M 파라미터               |
| FastEmit (2020)                       | -                   | 3.1% / 7.5%        | -             | 시퀀스 수준 정규화         |
| WeNet (2021)                          | -                   | 5.30% / -          | 0.251 → 0.114 | int8 양자화                |
| Conv-Transformer Transducer (2020)    | 1080ms (T-T 기준선) | 3.6% / 10.0%       | 0.04 (30ms)   | 139M 파라미터              |
| Developing Real-time Streaming (2020) | 360ms               | 8.28% / -          | 0.19          | 작은 선행 탐색, INT8       |
| WeNet 2.0 (2022)                      | -                   | -                  | -             | 샤드 기반 I/O              |
| Turning Whisper into Real-Time (2023) | 3.3 초              | 4.05%              | -             | LocalAgreement-2           |
| Whisper Streaming Decoding (2025)     | 9.02s (INT8)        | 17.30%             | -             | two-pass decoding          |
| Online Hybrid CTC/Attention (2023)    | 320ms               | 4.2% / 13.3%       | 1.5배         | MTA, T-CTC, DWJD           |
| Fast Conformer (2023)                 | -                   | 4.99% (test-other) | -             | 8 배 다운샘플링, 전역 토큰 |

#### 화자 인식 성능 비교

| 논문 (연도)                    | 데이터셋    | 화자 수 | 정확도     | 방법론               | 비교 대상                              |
| ------------------------------ | ----------- | ------- | ---------- | -------------------- | -------------------------------------- |
| Few Words ISR (2020)           | TIMIT       | 5       | 88.6%      | RL (PPO)             | 휴리스틱 85.1%, 무작위 74.1%           |
| Speaker Diarization E2E (2021) | CALLHOME    | 2-가변  | 75.6%      | SAD-first SC-EEND    | 기존 SC-EEND 77.6%, x-vector+AHC 54.6% |
| Deep Transfer Learning (2023)  | 의료/다국어 | -       | 91.17%     | LSTM RNN-LM          | -                                      |
| Federated Learning FL (2023)   | TED-LIUM 3  | 1943    | 10.92% WER | wav2vec 2.0 + FedAvg | CRDNN 37.04%                           |

### 3. 성능 패턴 및 경향 분석

#### 1. 모델 아키텍처의 진화 패턴

| 시기      | 주요 아키텍처                     | 특징                       | 성능 개선 동인                  |
| --------- | --------------------------------- | -------------------------- | ------------------------------- |
| 2015-2016 | CTC, LAS, Wav2Letter              | HMM-GMM 대체, ConvNet 기반 | RNN 대신 ConvNet으로 단순화     |
| 2017-2018 | 하이브리드 CTC-Attention          | 정렬 학습, RNN-LM 통합     | CTC-어텐션 결합, 다중 작업 학습 |
| 2019      | Transformer, SAN-CTC              | 셀프-어텐션, 병렬화        | 장기 의존성 처리, 빠른 학습     |
| 2020-2021 | Conformer, Y-Model                | CNN+Transformer 결합       | 지역적/전역적 특징 시너지       |
| 2022-2023 | Self-supervised (Wav2Vec, HuBERT) | 자기지도 학습 전이         | 레이블 데이터 효율성 극대화     |
| 2023-2025 | Efficient Conformer, Whisper      | 효율성 최적화, 양자화      | 계산 효율성 유지하면서 SOTA     |

#### 2. 성능과 자원 효율성의 트레이드오프

| 논문 (연도)                   | 모델 크기    | 파라미터                        | 추론 속도               | WER        | 효율성 지표                      |
| ----------------------------- | ------------ | ------------------------------- | ----------------------- | ---------- | -------------------------------- |
| Conformer (2020)              | Conformer(M) | 30.7M                           | -                       | 2.0%       | Transformer Transducer 139M 대비 |
| Efficient Conformer (2021)    | 13M          | 13M                             | Conformer 대비 29% 빠름 | 3.57%      | -                                |
| Fast Conformer (2023)         | XL/XXL       | -                               | 2.8 배 빠름             | 4.99%      | Conformer 대비 2.9 배 낮은 GMACs |
| Quantization Whisper (2025)   | INT4         | 141.11MB → 44.33MB (68.6% 감소) | -                       | 0.0159     | 98.4% 정확도                     |
| Federated Learning (2023)     | -            | -                               | -                       | 10.92% WER | 1943 클라이언트, 62% 기여        |
| Deep Transfer Learning (2023) | -            | -                               | -                       | 38.6% 감소 | 도메인 간 전이                   |

#### 3. 데이터 규모에 따른 성능 향상

| 논문 (연도)                       | 데이터 증가            | WER 감소                    | 비고                        |
| --------------------------------- | ---------------------- | --------------------------- | --------------------------- |
| Deep Speech 2 (2015)              | 10 배                  | 40%                         | power law 관계              |
| Robust SR Weak Supervision (2022) | 680,000 시간 약한 지도 | 55.2% 오류 감소 (12 개 OOD) | 제로샷 전이                 |
| Self-Training (2019)              | 360 시간 (clean)       | 5.79%                       | 오라클 대비 59.3% 격차 회복 |
| SpeeChain (2023)                  | 실제-합성 혼합         | -                           | 기본 WER 필터링이 성능 저해 |
| Text Generation for ASR DA (2023) | 8~26 배 증강           | 안정적 WER                  | Q&A 83 배 이상에서 포화     |

#### 4. 자기지도 학습의 효과

| 모델           | 데이터셋         | WER        | 자기지도 학습 효과         |
| -------------- | ---------------- | ---------- | -------------------------- |
| Wav2Vec 2.0    | LibriSpeech      | 4.90%      | SSL 사전 학습 필수         |
| Wav2Vec-C      | 실제 원거리 음성 | 1.4% rWERR | wav2vec 2.0 대비 2 배 향상 |
| HuBERT/SSLR    | CHiME-4          | 3.92%      | 도메인 불일치 해결         |
| BLSP (2023)    | ASR 880 만 쌍    | -          | 지시 따르기 능력 유지      |
| Whisper (2022) | -                | 2.5%       | 미세 조정 없이 OOD 성능    |

### 4. 추가 실험 및 검증 패턴

#### 1. ablation study 공통 패턴

| 검증 목적     | 실험 유형                           | 주요 발견                                 |
| ------------- | ----------------------------------- | ----------------------------------------- |
| 아키텍처 구성 | 컨볼루션 서브-블록 제거 (Conformer) | WER 급증                                  |
| 학습 전략     | 레이블 스무딩 제거 (LibriSpeech)    | 대규모 데이터셋에서 성능 저하 (2.8%→3.5%) |
| 손실 함수     | ASR 임베딩 + 분리 손실 제거 (UNSE)  | nOVL 성능 저하                            |
| 디코딩 전략   | `attention_rescoring` vs 다른 방식  | `attention_rescoring`가 가장 우수         |
| 위치 인코딩   | 사인파 임베딩 vs 합성곱 위치 인코딩 | Convolutional PE 0.1~0.8%p 개선           |
| 청크 크기     | look-ahead 변화 (320ms 등)          | 작은 선행 탐색 허용 필요                  |
| 데이터 비율   | 실제-합성 데이터 혼합 비율          | 50/50 혼합이 최적                         |

#### 2. 민감도 분석 패턴

| 변수                                    | 분석 범위      | 관찰된 영향                                    |
| --------------------------------------- | -------------- | ---------------------------------------------- |
| 인코더 look-ahead ($\varepsilon^{enc}$) | 0, 1, 3        | 0→3 에서 9.4%→8.1% WER 개선                    |
| 디코더 look-ahead ($\varepsilon^{dec}$) | 18, 20, 24 등  | 18 이 최적, 더 큰 값은 성능 저하               |
| 모델 깊이                               | 4~48 레이어    | 4→24 에서 20.8%→12.1% WER 개선                 |
| $\lambda$ 정규화 가중치                 | 0~임계값 이상  | 너무 크면 WER 저하, 적절히 커야 대기 시간 감소 |
| 청크 크기                               | 100ms~1500ms   | 1500ms 에서 16.85%→16.65% WER 개선             |
| 드롭아웃 비율                           | 0.1, 0.15      | 0.15 에서 80 에포크 유지                       |
| 학습률                                  | 3e-4, 스케줄링 | 400 스텝 워밍업 + ReduceLROnPlateau            |

#### 3. 다중 태스크 학습 구성

| 구성                         | 하위 작업          | 간섭 완화 효과              |
| ---------------------------- | ------------------ | --------------------------- |
| FSE (Full Shared Encoder)    | T2T, SSL, S2P, S2T | ASR 작업 간섭 최소화        |
| PSE (Partial Shared Encoder) | ST 위주            | 번역 작업 간섭 최소화       |
| FSE vs PSE                   | -                  | ASR: FSE 우위, ST: PSE 우위 |
| T2T 생략                     | -                  | 평균 +0.5 WER               |
| S2T 제거                     | -                  | 평균 +1.1 WER               |
| S2P 제거                     | -                  | 학습 수렴 실패              |

### 5. 실험 설계의 한계 및 비교상의 주의점

#### 1. 비교 조건의 불일치

| 문제 유형        | 사례                            | 영향                                            |
| ---------------- | ------------------------------- | ----------------------------------------------- |
| 외부 언어 모델   | 일부 모델 사용, 일부 비사용     | WER 비교 불공정 (test-clean vs test-other 차이) |
| 디코딩 방식      | 빔 서치, 탐욕적, 재점수화 다양  | 실제 성능 vs 벤치마크 성능 차이                 |
| 사전 학습 의존성 | WavLM, HuBERT 등 사전 학습 필수 | 무작위 초기화 시 훈련 실패                      |
| 데이터셋 일반화  | LibriSpeech 중심 평가           | 다른 데이터셋에서 일반화 성능 한계              |
| 하이퍼파라미터   | 학습률, 드롭아웃, 층 수 등 다양 | 최적 조건에서만 결과 보장                       |

#### 2. 데이터셋 의존성

| 데이터셋    | 일반화 문제          | 비고                                    |
| ----------- | -------------------- | --------------------------------------- |
| LibriSpeech | clean 환경 위주      | test-other 에서 7.2%, clean 대비 급상승 |
| TIMIT       | 제한적 (6300 문장)   | 대규모 데이터셋 대비 제한적 일반화      |
| CHiME-4     | 시뮬레이션 잡음 위주 | 실제 잡음 환경에서 성능 저하 가능       |
| FLEURS      | 저자원 언어 편중     | 75 언어 중 일부만 잘 수행               |
| KsponSpeech | 한국어 단일 언어     | 다국어/다언어 환경에서 일반화 한계      |

#### 3. 평가 지표의 한계

| 지표 | 한계                                 | 대안                      |
| ---- | ------------------------------------ | ------------------------- |
| WER  | 희귀어/OOV 처리 어려움               | LM 기반 재점수화          |
| CER  | 문자 단위에서 단어 단위보다 과대평가 | WER 병행 사용             |
| BLEU | 번역 품질 측정에만 사용              | ST 전용, ASR에는 WER 사용 |
| nOVL | 음성 향상 품질 측정                  | PESQ, Si-SDR, STOI 병행   |
| MCD  | 다중 화자 일반화 어려움              | -                         |

#### 4. 일반화 한계

| 분야         | 일반화 문제                   | 원인                   |
| ------------ | ----------------------------- | ---------------------- |
| 온디바이스   | Pixel 4/5/6 결과만 검증       | 단일 하드웨어 환경     |
| 스트리밍     | 지연 시간 vs 정확도 절충      | 실제 응용 환경 미검증  |
| 잡음 강건성  | CHiME-4 외 일반화 제한        | 특정 잡음 유형 위주    |
| 도메인 적응  | 의료/감정 인식 등 특수 도메인 | 소규모 데이터셋        |
| 언어 간 전이 | 영어 데이터로 비영어 학습     | 제로샷, 교차 모달 효과 |

### 6. 결과 해석의 경향

#### 1. 저자들의 공통 해석 패턴

| 해석 패턴          | 사례                                        | 함의                                  |
| ------------------ | ------------------------------------------- | ------------------------------------- |
| SOTA 달성          | Conformer 언어 모델 사용 시 1.9%/3.9%       | 모델 크기 + 사전 학습 + 최적화 = SOTA |
| 자기지도 학습 필수 | Wav2Vec 2.0 대비 Wav2Vec-C 2 배 향상        | 레이블 효율성 극대화                  |
| 효율성 중요        | Efficient Conformer 29% 추론 시간 단축      | 계산 자원 절감                        |
| 제로샷 전이        | 680,000 시간 약한 지도로 55.2% 오류 감소    | 데이터 희소성 극복                    |
| 멀티모달 효과      | AVEC 오디오-시각적 통합 4 배 적은 훈련 단계 | 시각 정보 강건성                      |

#### 2. 실제 관찰 vs 해석

| 관찰 결과                                   | 저자 해석            | 주의점                                |
| ------------------------------------------- | -------------------- | ------------------------------------- |
| Conformer 대비 Efficient Conformer 29% 빠름 | "계산 효율성 극대화" | 파라미터 13M으로 SOTA 근접            |
| Wav2Vec 2.0 대비 Wav2Vec-C 2 배 향상        | "코드북 활용률 해결" | K-means는 깨끗한 음성에서 성능 저하   |
| 50/50 혼합이 최적                           | "데이터 다양성 중요" | 합성 데이터 alone 은 49.8%/81.78% WER |
| 제로샷 전이 55.2% 오류 감소                 | "데이터 부족 극복"   | 특정 데이터셋에서 검증됨              |
| 오디오-시각적 4 배 적은 훈련 단계           | "멀티모달 시너지"    | Inter CTC 모달리티 불균형 해결        |

#### 3. 해석과 사실의 괴리

| 현상             | 실제                             | 해석                    |
| ---------------- | -------------------------------- | ----------------------- |
| 데이터 규모 증가 | 10 배 → 40% WER 감소             | "규모 법칙" 일반화      |
| 효율적 모델      | Efficient Conformer 13M 파라미터 | "적은 파라미터로 SOTA"  |
| 제로샷 전이      | 55.2% 오류 감소                  | "진정한 강건성"         |
| 자기지도 학습    | Wav2Vec-C 100% 코드북 활용       | "코드북 활용 문제 해결" |
| 다중 작업 학습   | FSE/PSE 구성 최적                | "하위 작업 간섭 완화"   |

### 7. 종합 정리

제공된 실험 결과들을 종합하면, Speech Recognition 분야의 성능 향상이 단일 요소가 아닌 여러 차원의 기술 발전에 의해 달성됨이 명확합니다. 모델 아키텍처 측면에서는 RNN 기반 모델에서 Transformer, Conformer로 진화하며, 지역적 특징 추출과 전역적 상호작용을 결합하는 접근이 SOTA 성적을 달성하는 핵심이 되었습니다. 학습 방법론 측면에서는 자기지도 학습 모델(Wav2Vec, HuBERT 등) 과 사전 학습이 레이블 데이터 효율성과 OOD 강건성을 동시에 확보하는 데 결정적 역할을 했으며, 약한 지도 학습과 준지도 학습 파이프라인이 데이터 희소성 문제를 해결하는 주요 경로로 emerged 되었습니다. 평가 환경 측면에서는 온디바이스 처리, 스트리밍 지연 시간, 잡음 강건성 등의 현실적 요구사항을 충족하면서도 SOTA 성능을 유지하는 효율적 모델 설계(Fast Conformer, Efficient Conformer, 양자화 등) 가 중요한 연구 방향임을 보여줍니다. 데이터 규모 측면에서는 680,000 시간 약한 지도 데이터로 학습한 Whisper 모델이 제로샷 조건에서도 2.5% WER을 달성하는 등, 대규모 약한 지도 데이터의 전이 효과가 입증되었습니다. 결론적으로, 높은 성능을 달성하기 위해서는 모델 아키텍처의 효율성 최적화, 자기지도 학습 기반의 레이블 효율성, 대규모 약한 지도 데이터 활용, 온디바이스/스트리밍 요구사항에 맞는 디코딩 및 양자화 전략이 종합적으로 적용되어야 합니다.
