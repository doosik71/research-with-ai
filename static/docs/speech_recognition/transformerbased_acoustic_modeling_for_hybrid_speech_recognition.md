# TRANSFORMER-BASED ACOUSTIC MODELING FOR HYBRID SPEECH RECOGNITION

Yongqiang Wang, Abdelrahman Mohamed, Duc Le, Chunxi Liu, Alex Xiao, Jay Mahadeokar, Hongzhao Huang, Andros Tjandra, Xiaohui Zhang, Frank Zhang, Christian Fuegen, Geoffrey Zweig, Michael L. Seltzer

---

## 🧩 Problem to Solve

기존 음향 모델(Acoustic Models, AMs)에서 널리 사용되는 순환 신경망(RNNs)은 장기적인 시간 의존성 모델링에 어려움이 있고, 순환 특성 때문에 음성 신호 병렬 처리가 어렵다는 한계를 가지고 있습니다. 이 논문은 이러한 RNN의 한계를 극복하고 하이브리드 음성 인식(Hybrid Speech Recognition) 시스템에서 음향 모델링의 성능을 향상시키기 위해, 자연어 처리 분야에서 성공을 거둔 Transformer 아키텍처를 적용하는 것을 목표로 합니다.

## ✨ Key Contributions

- 하이브리드 음성 인식 시스템을 위한 Transformer 기반 음향 모델을 제안하고 평가했습니다.
- 다양한 위치 임베딩(Positional Embedding) 방법(사인파, 프레임 스태킹, 합성곱 임베딩)을 비교하고, 합성곱 임베딩이 가장 효과적임을 보여주었습니다.
- 깊은 Transformer 모델 학습을 가능하게 하는 반복 손실(iterated loss) 기법을 사용하여 모델의 깊이를 심화하고 성능을 크게 향상시켰습니다.
- 표준 4-gram 언어 모델(LM)을 사용했을 때 Librispeech 벤치마크에서 기존의 최고 하이브리드 시스템보다 19%에서 26%의 상대적인 단어 오류율(Word Error Rate, WER) 감소를 달성했습니다.
- 신경망 LM(NNLM) 재점수화(rescoring)와 결합하여 Librispeech에서 최신(State-of-the-Art, SOTA) 성능을 달성했습니다.
- 더 큰 내부 데이터셋에서도 Transformer 기반 음향 모델의 우수성을 확인했습니다.
- 제한된 우측 문맥(limited right context)을 사용하는 스트리밍 애플리케이션 가능성에 대한 예비 연구를 수행했습니다.

## 📎 Related Works

- **심층 학습 기반 ASR:** 초기 RNN (LSTM), TDNN, FSMN, CNN 등의 다양한 신경망 아키텍처가 음향 모델링에 탐구되었습니다.
- **Self-Attention 및 Transformer:** Vaswani et al. (2017)의 "Attention is All You Need"에서 제안된 Self-Attention 및 Transformer는 NLP 분야에서 뛰어난 성능을 보였습니다.
- **ASR 분야의 Self-Attention 및 Transformer:** 주로 시퀀스-투-시퀀스(sequence-to-sequence) 아키텍처에서 Self-Attention 및 Transformer가 사용되었으며, 일부 예외적으로 음향 모델링에 Self-Attention 레이어만 사용된 연구도 있었습니다.
- **위치 임베딩:** Sperber et al. (2018)는 시퀀스-투-시퀀스 모델에서 다양한 위치 임베딩 방법을 연구했습니다. Mohamed et al. (2019)는 ASR을 위해 합성곱 문맥(convolutional context)을 가진 Transformer를 제안했습니다.
- **깊은 신경망 학습:** Tjandra et al. (2020)은 깊은 Transformer 네트워크에서 이중 특징 표현(double feature presentation)과 반복 손실(iterated loss)을 사용했습니다. Al-Rfou et al. (2019)도 깊은 Self-Attention 모델 학습을 위해 유사한 손실 함수를 사용했습니다.

## 🛠️ Methodology

1. **하이브리드 아키텍처:** 음향 인코더(Acoustic Encoder)가 입력 시퀀스 $x_1, \dots, x_T$를 고수준 임베딩 벡터 $z_1, \dots, z_T$로 인코딩합니다. 이 벡터들은 HMM(Hidden Markov Model)의 상태(senone 또는 chenone)에 대한 사후 확률 분포를 생성하며, 이는 어휘집(lexicon) 및 언어 모델(LM)과 결합되어 최적의 가설을 찾습니다. 본 연구에서는 이 인코더를 Transformer로 대체합니다.

2. **Transformer 음향 모델:**

   - **Self-Attention 및 Multi-Head Attention (MHA):**
     - 쿼리(Query) $W_q x_t$와 키(Key) $W_k x_\tau$의 내적을 통해 어텐션 분포 $\alpha_{t\tau}$를 계산합니다.
     - 값(Value) $W_v x_\tau$를 어텐션 가중치로 가중 평균하여 출력 임베딩 $z_t$를 얻습니다.
     - MHA는 여러 어텐션 헤드를 병렬로 적용하고 그 출력을 연결하여 선형 변환합니다.
   - **Transformer 아키텍처:** 각 Transformer 레이어는 MHA 서브 레이어와 완전 연결 피드포워드 네트워크(FFN)로 구성됩니다. 잔차 연결(residual connections)과 드롭아웃(dropouts)이 적용됩니다. Layer Normalization은 MHA 및 FFN 전에 적용되며, 세 번째 Layer Normalization은 레이어를 완전히 우회하는 것을 방지합니다. 활성화 함수로 GELU를 사용합니다.
   - **위치 임베딩(Positional Embedding, PE):**
     - **Sinusoid PE:** 사인파 함수를 사용하여 입력 $x_t$에 절대 위치 정보를 주입합니다.
     - **Frame Stacking:** 현재 프레임과 이후 8개 프레임을 스태킹하고 stride-2 샘플링을 통해 상대 위치 정보를 인코딩합니다.
     - **Convolutional Embedding:** 두 개의 VGG 블록(3x3 커널의 합성곱 레이어 2개, ReLu, 풀링 레이어)을 Transformer 레이어 앞에 사용하여 단거리 스펙트럼-시간 패턴과 암묵적인 상대 위치 정보를 학습합니다. (가장 좋은 성능)
   - **깊은 Transformer 학습:** 깊은 Transformer 모델의 학습 불안정 문제를 해결하기 위해 반복 손실(iterated loss) 기법을 사용합니다. 일부 중간 Transformer 레이어의 출력에도 보조 크로스 엔트로피(CE) 손실을 계산하고, 이를 최종 손실 함수에 가중치 0.3으로 보간하여 적용합니다.

3. **실험 설정:**
   - **데이터셋:** Librispeech (960시간) 및 내부 영어 비디오 ASR 데이터셋 (13.7K 시간).
   - **특징:** 80차원 로그 멜-필터 뱅크 특징, 10ms 프레임 시프트, 20ms 프레임 레이트.
   - **증강:** Speed Perturbation 및 SpecAugment (LD 정책).
   - **최적화:** Adam optimizer, 선형 웜업(warm-up) 학습률 스케줄.
   - **모델 크기:** 12-레이어 Transformer ($d_i=768$, 약 90M 파라미터), 20-레이어 Transformer ($d_i=768$, 약 149M 파라미터), 24-레이어 Transformer ($d_i=512$, 약 81M 파라미터). 비교 대상 BLSTM 모델 ($94M \sim 163M$ 파라미터).
   - **훈련:** 최대 10초 길이로 음성 분할(segmentation), 오버피팅 방지를 위한 SpecAugment 필수.

## 📊 Results

- **위치 임베딩 효과:**
  - Convolutional PE 방식이 Sinusoid PE, Frame Stacking, None 방식보다 Librispeech test-clean 및 test-other 세트에서 가장 낮은 WER을 기록하며 가장 좋은 성능을 보였습니다 (예: test-other 6.46%).
- **Transformer vs. BLSTM:**
  - 유사한 파라미터 수를 가진 Transformer 기반 모델(vggTrf)이 BLSTM 기반 모델(vggBLSTM)보다 test-clean에서 2-4%, test-other에서 7-11% 더 낮은 WER을 보이며 일관되게 우수한 성능을 나타냈습니다.
  - 가장 큰 vggTrf (768, 20) 모델은 vggBLSTM (1000, 6)보다 Librispeech test-other에서 6.10% 대 6.63%로 더욱 우수했습니다.
- **반복 손실(Iterated Loss) 효과:**
  - 반복 손실을 사용하지 않으면 20레이어보다 깊은 모델(예: 24레이어 Transformer)은 수렴하지 않거나 성능 향상이 미미했습니다.
  - 반복 손실을 적용한 vggTrf (512, 24) 모델은 vggTrf (768, 12) 기본 모델 대비 test-clean에서 7%, test-other에서 13% WER을 감소시켰습니다.
- **최종 성능 및 SOTA 달성:**
  - sMBR 훈련을 추가한 vggTrf (512, 24) 모델은 표준 4-gram LM을 사용했을 때 Librispeech test-clean에서 2.60%, test-other에서 5.59%의 WER을 달성하며, 이전 최고 하이브리드 시스템보다 각각 19%, 26% 상대적 성능 향상을 이루었습니다.
  - Transformer LM 재점수화와 결합하여 test-clean 2.26%, test-other 4.85%로 Librispeech 벤치마크에서 SOTA를 달성했습니다.
- **제한된 우측 문맥:** 12-레이어 Transformer 모델에 추론 시 제한된 우측 문맥(limited right context)을 강제 적용했을 때, 문맥이 줄어들수록 WER이 증가했지만, 충분히 큰 문맥에서는 여전히 합리적인 성능을 보였습니다.
- **대규모 데이터셋 실험:** 내부 영어 비디오 ASR 데이터셋에서도 vggTrf (768, 12) 모델이 vggBLSTM (800, 5) 모델보다 4.0-7.6% 낮은 WER을 기록하며 우수성을 입증했습니다.

## 🧠 Insights & Discussion

- **Transformer의 우수성:** 이 연구는 Transformer 아키텍처가 하이브리드 음성 인식의 음향 모델링에서 BLSTM과 같은 강력한 RNN 기반 모델을 능가할 수 있음을 명확하게 보여주었습니다. 특히 Librispeech 벤치마크에서 SOTA 성능을 달성하며 그 잠재력을 입증했습니다.
- **깊은 모델 학습의 중요성:** 반복 손실(iterated loss) 기법은 깊은 Transformer 모델의 효과적인 학습에 필수적이며, 모델의 깊이를 늘리는 것이 성능 향상에 기여한다는 것을 확인했습니다.
- **위치 정보 인코딩:** 합성곱 계층을 통한 위치 임베딩이 단순한 사인파 또는 프레임 스태킹보다 효과적이며, 이는 음성 신호의 스펙트럼-시간 패턴과 상대적 위치 정보 학습의 중요성을 시사합니다.
- **한계점 및 향후 연구:**
  - **높은 계산 비용:** Transformer의 입력 시퀀스 길이에 대한 계산 비용이 제곱으로 증가하는 특성은 실제 스트리밍 ASR 애플리케이션에 적용하는 데 큰 제약이 됩니다.
  - **스트리밍 적용:** 현재 연구는 완전 문맥(full context)을 주로 사용하며, 제한된 우측 문맥에 대한 예비 연구는 추론 시 여전히 긴 미래 정보(look-ahead window)를 필요로 합니다. 스트리밍 제약 조건 하에서 Transformer 기반 음향 모델을 심층적으로 연구할 필요가 있습니다.
  - **Recurrence 대체 효과:** Transformer의 우수성이 Self-Attention의 재귀(recurrence) 대체에서 오는 것인지, 아니면 다른 모델링 기법에서 오는 것인지에 대한 심층적인 분석은 이루어지지 않았습니다.

## 📌 TL;DR

이 논문은 하이브리드 음성 인식의 음향 모델링에 Transformer를 적용하여 기존 RNN의 한계를 극복하고자 했습니다. 합성곱 위치 임베딩과 깊은 네트워크 훈련을 위한 반복 손실을 사용하여, Transformer 기반 음향 모델이 Librispeech 벤치마크에서 BLSTM 대비 크게 우수한 성능을 보였으며, 신경망 LM과 결합하여 SOTA WER을 달성했습니다. 다만, 긴 시퀀스에 대한 계산 비용과 스트리밍 애플리케이션 적용 가능성은 향후 연구 과제로 남아 있습니다.
