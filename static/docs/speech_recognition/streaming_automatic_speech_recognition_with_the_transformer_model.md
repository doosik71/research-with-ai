# STREAMING AUTOMATIC SPEECH RECOGNITION WITH THE TRANSFORMER MODEL

Niko Moritz, Takaaki Hori, Jonathan Le Roux

## 🧩 Problem to Solve

기존의 인코더-디코더 기반 시퀀스-투-시퀀스 모델, 특히 Transformer 아키텍처는 음성 인식(ASR)에서 최첨단(SOTA) 성능을 달성했지만, 일반적으로 전체 음성 발화를 입력으로 요구하여 오프라인 ASR 작업에만 사용 가능했습니다. 이로 인해 출력이 각 단어가 발화된 직후에 생성되어야 하는 스트리밍 ASR 애플리케이션에는 적용하기 어려웠습니다. 본 논문의 핵심 문제는 Transformer 모델의 강력한 성능을 유지하면서도 실시간 스트리밍 환경에서 동작하는 ASR 시스템을 개발하는 것입니다.

## ✨ Key Contributions

- **스트리밍 Transformer ASR 시스템 제안:** Transformer 기반의 종단 간(end-to-end) 스트리밍 ASR 시스템을 성공적으로 제안했습니다.
- **인코더에 시간 제한 자기 주의(Time-Restricted Self-Attention) 적용:** 인코더의 컨텍스트를 고정된 미래 프레임으로 제한하여 인코더의 지연 시간(latency)을 제어했습니다.
- **디코더에 Triggered Attention (TA) 메커니즘 적용:** 인코더-디코더 주의 메커니즘에 TA 개념을 도입하여 디코더가 스트리밍 방식으로 동작할 수 있도록 했습니다.
- **CTC와의 공동 훈련 및 디코딩:** Transformer 모델을 CTC(Connectionist Temporal Classification) 목적 함수와 공동으로 훈련하고, CTC-TA 공동 디코딩 알고리즘을 사용하여 훈련 및 디코딩 결과를 최적화했습니다.
- **최고 성능 달성:** LibriSpeech 데이터셋의 "clean" 테스트에서 2.8%, "other" 테스트에서 7.2%의 단어 오류율(WER)을 달성하여, 저자들이 아는 한 이 작업에 대한 최고의 스트리밍 종단 간 ASR 결과를 기록했습니다.

## 📎 Related Works

- **기존 ASR 시스템:**
  - 하이브리드 HMM(Hidden Markov Model) 기반 ASR [1, 2]
  - 종단 간 ASR: CTC [4], RNN 트랜스듀서 (RNN-T) [5, 7, 8], 주의 기반 인코더-디코더 [6, 9]
- **스트리밍 주의 기반 ASR:**
  - Neural Transducer (NT) [10]
  - Monotonic Chunkwise Attention (MoChA) [11]
  - Triggered Attention (TA) [12, 14]: 본 연구에서 활용.
- **신경망 아키텍처:**
  - RNN 기반 (LSTM, BLSTM, LC-BLSTM, PTDLSTM) [15]
  - Transformer 모델 [16]: 기계 번역에서 제안되었으며 ASR에 적용 [17].
- **공동 훈련 및 디코딩:**
  - 하이브리드 CTC/주의 아키텍처 [13, 17]
  - CTC-주의 기반 모델을 위한 프레임 동기식 원패스 디코딩 [14]
- **기타 기술:**
  - 시간 제한 자기 주의(Time-restricted self-attention) [19]
  - SentencePiece [24], SpecAugment [25]

## 🛠️ Methodology

본 논문은 Transformer 기반 ASR 시스템을 스트리밍 환경에 맞게 변형하기 위해 인코더와 디코더에 각각 다른 접근 방식을 적용합니다.

1. **스트리밍 인코더: 시간 제한 자기 주의(Time-Restricted Self-Attention)**

   - 인코더는 두 개의 CNN 레이어 모듈 ${\small ENC} CNN$과 $E$개의 자기 주의 레이어 스택 ${\small ENC} SA$로 구성됩니다.
   - ${\small ENC} CNN$은 입력 음향 특징 $X$의 프레임 양을 4배 감소시킵니다.
   - ${\small ENC} SA$ 레이어의 자기 주의 메커니즘에 **시간 제한**을 적용하여 미래 컨텍스트를 $\varepsilon^{enc}$ 프레임으로 제한합니다. 이는 $x^{E}_{1:n} = {\small ENC} SA_{tr}(x^{0}_{1:n+\varepsilon^{enc}})$와 같이 표현됩니다.
   - $\varepsilon^{enc}$는 각 인코더 레이어에서 사용되는 look-ahead 프레임 수를 나타내며, 총 인코더 지연 시간은 $E \times \varepsilon^{enc} \times 40 \text{ms}$가 됩니다.

2. **스트리밍 디코더: Triggered Attention (TA)**

   - 디코더의 인코더-디코더 주의 메커니즘은 TA 개념 [12, 14]을 사용하여 스트리밍 방식으로 작동합니다.
   - TA 훈련은 인코더 상태 시퀀스 $X_E$와 레이블 시퀀스 $Y$ 간의 정렬(alignment)을 요구하며, 이는 보조 CTC 목적 함수 $p_{ctc}(Y|X_E)$를 통해 강제 정렬(forced alignment)하여 얻습니다.
   - 디코더는 과거 인코더 프레임과 고정된 $\varepsilon^{dec}$ look-ahead 프레임에만 주의(attention)를 기울입니다.
   - Triggered attention 목적 함수는 $p_{ta}(Y|X_E) = \prod_{l=1}^{L} p(y_l|y_{1:l-1},x^{E}_{1:\nu_l})$로 정의되며, 여기서 $\nu_l = n'_{l} + \varepsilon^{dec}$이고 $n'_{l}$은 CTC 강제 정렬에서 레이블 $y_l$의 첫 번째 발생 위치입니다.
   - 디코더 지연 시간은 $\varepsilon^{dec} \times 40 \text{ms}$입니다.

3. **공동 훈련:**

   - CTC 모델과 Triggered Attention 모델은 다중 목적 손실 함수 $L = -\gamma \log p_{ctc} - (1-\gamma) \log p_{ta}$를 사용하여 공동 훈련됩니다. ($γ$는 0.3으로 설정됨)

4. **공동 CTC-Triggered Attention 디코딩:**
   - 프레임 동기식 원패스 디코딩 알고리즘 [14]을 사용하며, 이는 CTC prefix 빔 서치 알고리즘 [20]을 Triggered Attention 디코더와 통합하여 확장한 것입니다.
   - 디코딩 과정은 CTC prefix 점수 $p_{prfx}$, TA 점수 $p_{ta}$, RNN 언어 모델(LM) 점수 $p_{LM}$를 결합하여 최종 공동 점수 $p_{joint}$를 계산합니다.
     $$p_{joint}(\mathcal{l}) = \lambda \log p_{prfx}(\mathcal{l}) + (1-\lambda) \log p_{ta}(\hat{\mathcal{l}}) + \alpha \log p_{LM}(\mathcal{l}) + \beta |\mathcal{l}|$$
   - $K$개의 가장 확률 높은 prefix를 선택하고 빔 폭 $θ_1$, $θ_2$를 사용하여 가지치기(pruning)를 수행합니다.

## 📊 Results

- **데이터셋:** LibriSpeech (960시간 훈련 데이터)
- **모델 설정:** small 및 large Transformer 모델 사용. (Large 모델: $d_{model}=512, d_h=8$)
- **기준선 (오프라인) 성능:** 대형 Transformer 모델, RNN-LM, SpecAugment를 사용하여 _전체 시퀀스_ CTC-attention 디코딩 시 test-clean 2.7%, test-other 6.1% WER 달성.
- **스트리밍 Transformer 성능:**
  - **인코더 look-ahead ($\varepsilon^{enc}$) 효과:** $\varepsilon^{enc}$ 증가 시 WER이 크게 개선됩니다. 예를 들어, CTC 빔 서치에서 test-other WER이 $\varepsilon^{enc}=0$일 때 9.4%에서 $\varepsilon^{enc}=3$일 때 8.1%로 감소합니다.
  - **디코더 look-ahead ($\varepsilon^{dec}$) 효과:** 공동 CTC-TA 디코딩은 CTC prefix 빔 서치보다 일관되게 WER을 개선합니다. $\varepsilon^{dec}$를 18프레임으로 늘리면 더 좋은 성능을 보이며, 이는 전체 시퀀스 CTC-attention 디코딩 결과에 근접합니다.
  - **최고 스트리밍 결과:**
    - $\varepsilon^{enc}=3$ (인코더 지연 1440ms) 및 $\varepsilon^{dec}=18$ (디코더 지연 720ms) 설정에서 test-clean 2.8%, test-other 7.2% WER 달성. 총 지연 시간은 약 2190ms (${\small ENC} CNN$ 30ms + ${\small ENC} SA$ 1440ms + DECTA 720ms).
    - $\varepsilon^{enc}=1$ (인코더 지연 480ms) 및 $\varepsilon^{dec}=18$ (디코더 지연 720ms) 설정에서 test-clean 3.0%, test-other 7.8% WER 달성. 총 지연 시간은 약 1230ms로, 정확도와 지연 시간 사이의 좋은 절충점을 제공합니다.
- **SOTA 달성:** 제안된 시스템은 LibriSpeech에서 완전한 스트리밍 종단 간 ASR 시스템으로서 가장 낮은 WER을 달성했습니다.

## 🧠 Insights & Discussion

- Transformer 모델의 스트리밍 ASR 적용 가능성을 성공적으로 입증했으며, 특히 시간 제한 자기 주의와 Triggered Attention의 조합이 핵심적인 역할을 했습니다.
- 인코더와 디코더의 look-ahead 프레임 수($\varepsilon^{enc}, \varepsilon^{dec}$)를 조절하여 ASR 정확도와 시스템 지연 시간 사이에 유연한 균형점을 찾을 수 있음을 보여주었습니다. look-ahead 값을 늘리면 정확도가 향상되지만 지연 시간도 증가합니다.
- 공동 CTC-TA 디코딩 방식이 CTC prefix 빔 서치 단독보다 우수한 성능을 보였으며, look-ahead 프레임 수가 충분할 경우 오프라인 전체 시퀀스 디코딩 성능에 근접할 수 있음을 확인했습니다.
- LibriSpeech (word-piece 출력)에서는 디코딩 알고리즘의 DCOND 및 ACOND 조건이 WER에 큰 영향을 미치지 않았지만, WSJ (문자 수준 출력)와 같은 다른 작업에서는 개선 효과가 있음을 언급하여 출력 레이블 유형에 따른 민감성을 시사했습니다.
- 향후 연구에서는 래티스(lattice) 기반 CTC-TA 디코딩 구현을 통해 중간 CTC prefix 빔 서치 결과를 출력하여 사용자 인지 지연 시간(perceived latency)을 더욱 줄일 수 있을 것으로 기대되며, 이에 대한 심층적인 연구가 필요함을 밝혔습니다.

## 📌 TL;DR

본 논문은 Transformer 모델을 활용한 스트리밍 종단 간 ASR 시스템을 제안합니다. 인코더에는 **시간 제한 자기 주의**를, 디코더에는 **Triggered Attention** 개념을 적용하여 Transformer의 지연 시간 문제를 해결했습니다. CTC와 공동 훈련하고 프레임 동기식 원패스 디코딩 알고리즘을 사용하여, LibriSpeech 데이터셋에서 test-clean 2.8%, test-other 7.2%의 WER을 달성하며 스트리밍 종단 간 ASR의 새로운 최고 성능을 기록했습니다. 이 시스템은 look-ahead 프레임 조절을 통해 정확도와 지연 시간 간의 효과적인 절충점을 제공합니다.
