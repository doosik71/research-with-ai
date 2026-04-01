# FAST CONFORMER WITH LINEARLY SCALABLE ATTENTION FOR EFFICIENT SPEECH RECOGNITION

Dima Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar, Vahid Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar, Jagadeesh Balam, Boris Ginsburg

## 🧩 Problem to Solve

Conformer 기반 모델은 음성 처리 태스크에서 지배적인 아키텍처가 되었지만, 자체 어텐션 레이어의 입력 시퀀스 길이에 대한 이차 시간 및 메모리 복잡도로 인해 긴 오디오 시퀀스 처리와 모델 확장에 심각한 제약이 있습니다. 또한 기존 컨볼루션 전용 ASR 모델에 비해 더 많은 연산 및 메모리를 사용합니다. 이 연구는 이러한 Conformer의 효율성 및 확장성 문제를 해결하는 것을 목표로 합니다.

## ✨ Key Contributions

- **새로운 다운샘플링 스키마를 통한 Conformer 재설계:** 제안된 Fast Conformer(FC)는 기존 Conformer보다 추론 속도가 2.8배 빠르고, 2.9배 적은 연산량으로 유사한 WER(Word Error Rate)을 달성합니다.
- **10억 개 파라미터로 확장 가능:** 핵심 아키텍처 변경 없이 10억 개 이상의 파라미터로 스케일링을 지원하며, ASR 벤치마크에서 최첨단(SOTA) 정확도를 달성합니다.
- **긴 오디오 시퀀스 처리 능력 강화:** 전역 어텐션을 제한된 컨텍스트 어텐션과 전역 토큰(global token)으로 대체하여, 11시간 길이의 오디오를 단일 포워드 패스로 처리할 수 있게 함으로써 긴 형식의 음성 전사(transcription)를 가능하게 합니다.
- **다양한 음성 처리 태스크에서의 우수성:** Speech Translation(ST) 및 Spoken Language Understanding(SLU) 태스크에서도 기존 Conformer보다 정확도와 속도 면에서 뛰어난 성능을 보입니다.
- **오픈 소스 공개:** Fast Conformer 모델과 학습 레시피를 NVIDIA NeMo를 통해 오픈 소스로 공개했습니다.

## 📎 Related Works

- **Conformer [1]:** 지역적 특징을 위한 깊이별 컨볼루션 레이어와 전역 컨텍스트를 위한 자체 어텐션 레이어를 결합한 RNNT 기반 ASR 모델.
- **Quartznet [3]:** 컨볼루션 전용 ASR 모델로, Conformer보다 효율적이지만 성능이 떨어질 수 있습니다.
- **EfficientConformer [9]:** 점진적 다운샘플링과 그룹화된 어텐션을 사용하여 효율성을 높인 Conformer 변형.
- **Squeezeformer [8] & Uconv-Conformer [10]:** 점진적 다운샘플링과 U-Net 구조를 결합하여 시간 해상도를 조정하는 Conformer 변형.
- **Longformer [5]:** 긴 문서 처리를 위해 제한된 컨텍스트 어텐션과 전역 토큰을 사용하는 Transformer 모델.
- **HuBERT [23] & Wav2Vec 2.0 [30]:** 자체 지도 학습(Self-Supervised Learning, SSL)을 통해 음성 표현을 학습하는 모델로, FC 모델의 확장 및 사전 훈련에 활용되었습니다.

## 🛠️ Methodology

1. **새로운 다운샘플링 스키마 (8배 감소):**
   - 기존 Conformer의 4배(10ms $\to$ 40ms)에서 8배(10ms $\to$ 80ms)로 다운샘플링 비율을 시작 인코더에서 증가시켜 후속 어텐션 레이어의 연산 비용을 4배 줄입니다.
   - 기존 컨볼루션 서브샘플링 레이어를 깊이별 분리 가능 컨볼루션($\text{depthwise separable convolutions}$)으로 대체합니다.
   - 다운샘플링 블록의 컨볼루션 필터 수를 512개에서 256개로 줄입니다.
   - 컨볼루션 커널 크기를 31에서 9로 줄입니다.
   - CTC 손실 함수의 제약을 우회하기 위해 문자 토큰화 대신 SentencePiece BPE(Byte Pair Encoding) 토큰화(128~1024 토큰)를 사용합니다.
2. **긴 형식 오디오 전사 (Limited Context Attention + Global Token):**
   - 표준 멀티 헤드 어텐션 레이어를 학습 후(post-training) 제한된 컨텍스트 어텐션으로 대체합니다. 각 토큰 주변의 고정 크기 윈도우 내에서만 어텐션을 수행합니다.
   - 하나의 전역 어텐션 토큰(global attention token)을 추가하여, 이 토큰은 다른 모든 토큰에 어텐션하고 다른 모든 토큰은 이 토큰에 어텐션하도록 합니다.
   - Longformer에서 도입된 오버랩 청크(overlapping chunks) 방식을 활용하여 효율적인 연산을 가능하게 합니다.
3. **Fast Conformer 모델 스케일링:**
   - L (120M), XL (600M), XXL (1.1B) 세 가지 모델 크기를 설계했습니다.
   - Conformer 블록과 상대 어텐션을 포함한 핵심 아키텍처는 모델을 확장해도 변경하지 않았습니다.
   - XL에서 XXL로 확장 시, 자체 지도 학습(SSL) 기반의 사전 훈련(Wav2Vec 2.0 기반)을 활용하여 학습 안정화와 높은 학습률을 가능하게 했습니다.

## 📊 Results

- **ASR 성능 및 효율성:**
  - 인코더 추론 속도는 Conformer 대비 2.8배 빨라졌고, 연산량(GMACs)은 약 2.9배 감소(예: RNNT 모델의 경우 143.2 $\to$ 48.7)했습니다.
  - LibriSpeech, MLS, MCV, WSJ 벤치마크에서 기존 Conformer와 유사하거나 약간 더 좋은 WER을 달성했습니다 (예: LibriSpeech test-other WER: Conformer-RNNT 5.19% $\to$ Fast Conformer-RNNT 4.99%).
  - EfficientConformer, SqueezeFormer보다 계산 효율성 및 WER 측면에서 우수한 성능을 보였습니다.
- **긴 형식 오디오 처리:**
  - 제한된 컨텍스트 어텐션과 전역 토큰을 사용한 Fast Conformer는 A100 GPU에서 최대 675분(11시간 이상)의 오디오를 단일 패스로 처리할 수 있습니다 (기존 Conformer는 15분). 이는 45배 개선된 수치입니다.
  - TED-LIUM v3 및 Earnings-21 벤치마크에서 WER을 크게 개선했습니다 (예: TED-LIUM v3 WER: Conformer 9.18% $\to$ Fast Conformer + Limited Context + Global Token 7.51%).
- **Speech Translation (ST):**
  - Transformer 디코더와 결합 시 Conformer보다 1.66배 빠른 추론 속도와 더 높은 BLEU 점수(31.41 vs 31.02)를 달성했습니다.
  - RNNT 디코더와 결합 시에도 1.84배 빠른 속도와 높은 BLEU 점수(27.94 vs 23.28)를 기록했습니다.
- **Spoken Language Understanding (SLU):**
  - Conformer 기반 모델에 비해 1.1배 빠른 추론 속도를 보였으며, Intent Accuracy와 SLURP-F1에서 유사하거나 약간 더 좋은 성능을 달성했습니다 (예: Intent Acc. 90.68% vs 90.14%).
- **모델 스케일링:**
  - FC-XXL (1.1B 파라미터) 모델은 Conformer-XL보다 적은 GMACs (441 vs 686)로 LibriSpeech test-other에서 2.52%의 낮은 WER을 달성하여 SOTA 성능을 보였습니다.
  - 추가 4만 시간의 데이터셋(ASR Set++)으로 학습 시 XL 및 XXL 모델의 정확도와 잡음 강건성이 향상되었습니다.

## 🧠 Insights & Discussion

- Fast Conformer는 혁신적인 8배 다운샘플링 스키마를 통해 연산 효율성을 대폭 개선하면서도 Conformer의 핵심 장점인 정확도를 유지하거나 향상시켰습니다. 특히, 인코더 시작 단계에서 높은 다운샘플링을 적용하여 기존 점진적 다운샘플링의 연산 불균형 문제를 해결했습니다.
- BPE 토큰화의 도입은 8배 다운샘플링이 CTC 손실 함수의 길이 제약을 우회하고 효율성을 높이는 데 중요한 역할을 합니다.
- 제한된 컨텍스트 어텐션과 전역 토큰의 조합은 긴 오디오 시퀀스 처리의 한계를 극복하는 효과적인 전략임을 입증했습니다. 이는 실제 환경에서 긴 형식의 회의록 작성 등 다양한 활용 가능성을 제시합니다.
- Fast Conformer 아키텍처는 수십억 개의 파라미터로 쉽게 확장 가능하며, SSL 사전 훈련과 대규모 데이터셋 활용을 통해 성능과 잡음 강건성을 더욱 향상시킬 수 있음을 보여주었습니다.
- ASR 외 ST, SLU와 같은 다양한 음성 처리 태스크에서도 Fast Conformer의 효율성 이점이 확인되었습니다. 특히 SLU에서는 자동회귀(autoregressive) 디코더의 비용 때문에 인코더의 속도 향상이 전체 속도에 미치는 영향이 ASR만큼 크지 않을 수 있다는 점은 향후 연구 방향을 제시합니다.

## 📌 TL;DR

- **문제:** Conformer 모델의 높은 연산 비용과 자체 어텐션으로 인한 긴 오디오 시퀀스 처리 및 모델 확장성 한계.
- **제안 방법:** 8배 다운샘플링 스키마, 깊이별 분리 가능 컨볼루션, 학습 후(post-training) 제한된 컨텍스트 어텐션과 전역 토큰을 활용한 Fast Conformer를 제안.
- **주요 결과:** 기존 Conformer 대비 2.8배 빠른 추론 속도와 2.9배 낮은 연산량으로 SOTA ASR 정확도를 유지. 최대 11시간 길이의 오디오를 단일 패스로 처리 가능하며, 10억 개 파라미터로 확장 가능. ASR, ST, SLU 태스크 전반에서 우수한 성능과 효율성을 입증.
