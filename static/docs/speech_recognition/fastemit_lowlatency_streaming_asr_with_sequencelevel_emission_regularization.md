# FASTEMIT:LOW-LATENCY STREAMING ASR WITH SEQUENCE-LEVEL EMISSION REGULARIZATION

Jiahui Yu, Chung-Cheng Chiu, Bo Li, Shuo-yiin Chang, Tara N. Sainath, Yanzhang He, Arun Narayanan, Wei Han, Anmol Gulati, Yonghui Wu, Ruoming Pang

## 🧩 Problem to Solve

스트리밍 자동 음성 인식(ASR) 시스템은 음성 입력이 들어오는 즉시 가능한 한 빠르고 정확하게 단어를 인식해야 합니다. 하지만 기존의 스트리밍 ASR 모델들은 시퀀스 트랜스듀서(sequence transducer) 목표 함수를 단순히 최대화할 경우, 더 나은 예측을 위해 더 많은 미래 문맥을 사용하려는 경향이 있어 상당한 방출 지연(emission delay)을 초래합니다.

기존의 지연 감소 방법들(예: Early and Late Penalties, Constrained Alignments)은 토큰별(per-token) 또는 프레임별(per-frame) 확률 예측에 페널티를 부여하여 지연을 줄였지만, 이는 시퀀스 수준(sequence-level)에서 확률을 최적화하는 트랜스듀서의 방식과 일치하지 않아 상당한 정확도 저하(accuracy regression)를 겪습니다. 또한, 이러한 방법들은 추가적인 음성-단어 정렬(speech-word alignment) 정보나 높은 계산 비용을 요구합니다.

이 논문은 이러한 정확도 저하 없이 스트리밍 ASR의 방출 지연을 효과적으로 줄이는 방법을 제안합니다.

## ✨ Key Contributions

- **시퀀스 수준 방출 정규화(Sequence-Level Emission Regularization) 방법인 FastEmit 제안**: 기존의 토큰/프레임 기반 정규화와 달리 트랜스듀서의 전향-후향(forward-backward) 시퀀스 확률에 직접 적용됩니다.
- **정렬 정보 불필요**: 기존 방법과 달리 음성-단어 정렬 정보를 필요로 하지 않아 "플러그 앤 플레이" 방식으로 쉽게 적용 가능합니다.
- **최소한의 하이퍼파라미터**: 트랜스듀서 손실과 정규화 손실의 균형을 맞추는 단 하나의 하이퍼파라미터 $\lambda$만 도입합니다.
- **추가 훈련/서빙 비용 없음**: FastEmit 적용에 따른 추가적인 훈련 또는 서빙(serving) 비용이 발생하지 않습니다.
- **다양한 트랜스듀서 모델에 적용 가능**: RNN-Transducer, Transformer-Transducer, ConvNet-Transducer, Conformer-Transducer 등 다양한 최신 스트리밍 ASR 네트워크에 효과적으로 적용됨을 입증했습니다.
- **상당한 대기 시간 감소 및 정확도 향상**: Voice Search 테스트 세트에서 이전 기술 대비 150~300ms의 대기 시간 감소와 함께 더 나은 정확도를 달성했습니다. LibriSpeech에서는 90번째 백분위수 대기 시간을 210ms에서 30ms로 줄이는 동시에 WER을 4.4%/8.9%에서 3.1%/7.5%로 향상시켰습니다.

## 📎 Related Works

- **Early and Late Penalties [1]**: 문장의 끝(</s>) 예측을 VAD(Voice Activity Detector)가 제공하는 합리적인 시간 창 내에서 강제하여 지연을 줄이는 방법입니다.
- **Constrained Alignments [2, 3]**: 기존 음성 모델에서 생성된 음성-텍스트 정렬 정보를 기반으로 각 단어에 페널티를 확장하여 지연을 줄이는 방법입니다.
  - **한계**: 이 두 가지 정규화 방법은 토큰별 또는 프레임별 확률 예측에 독립적으로 페널티를 부과하여 시퀀스 수준의 트랜스듀서 최적화와 일관성이 없어 정확도 저하를 겪습니다. 또한 정렬 정보를 필요로 합니다.
- **Second-pass Listen, Attend and Spell (LAS) rescorer [16, 17] 및 Minimum Word Error Rate (MWER) 훈련 [18]**: 정확도 저하를 완화하기 위한 방법들이지만, 훈련 및 서빙 시 무시할 수 없는 계산 비용을 동반합니다.

## 🛠️ Methodology

1. **트랜스듀서 최적화의 문제점 분석**:

   - 트랜스듀서는 입력 시퀀스 $x = (x_{1}, ..., x_{T})$와 출력 시퀀스 $y = (y_{1}, ..., y_{U})$ 사이의 확률적 정렬을 학습합니다.
   - 이를 위해 출력 공간 $Y$를 '공백 토큰' $\emptyset$으로 확장하여 $\bar{Y} = Y \cup \emptyset$으로 만듭니다.
   - 트랜스듀서는 목표 시퀀스의 로그 확률을 최대화하는 것을 목표로 합니다: $L = -\log P(\hat{y}|x) = -\log \sum_{a \in B^{-1}(\hat{y})} P(a|x)$.
   - 여기서 $B$는 정렬 격자(alignment lattice) $a$에서 $\emptyset$ 토큰을 제거하는 함수입니다.
   - 트랜스듀서의 최적화 과정에서 어휘 토큰 $y \in Y$와 공백 토큰 $\emptyset$은 로그 확률이 최대화되는 한 동일하게 취급됩니다. 이는 '아무것도 출력하지 않음'을 의미하는 공백 토큰이 많아져 방출 지연을 야기할 수 있습니다.

2. **FastEmit 정규화 제안**:

   - 트랜스듀서의 전향 변수 $\alpha(t,u)$와 후향 변수 $\beta(t,u)$를 사용하여 특정 노드 $(t,u)$를 통과하는 모든 완전한 정렬의 확률 $P(A_{t,u}|x) = \alpha(t,u)\beta(t,u)$를 두 가지 경로로 분해합니다.
     - 공백 예측 경로: $\alpha(t,u)b(t,u)\beta(t+1,u)$
     - 레이블 예측 경로: $\alpha(t,u)\hat{y}(t,u)\beta(t,u+1)$
   - FastEmit는 레이블 예측 경로의 확률을 추가로 최대화하도록 손실 함수를 수정합니다. 새로운 정규화된 트랜스듀서 손실 $\tilde{L}$은 다음과 같습니다:
     $$ \tilde{L} = -\log \sum*{(t,u):t+u=n} (P(A*{t,u}|x) + \lambda \tilde{P}(A*{t,u}|x)) $$
     여기서 $\tilde{P}(A*{t,u}|x) = \alpha(t,u)\hat{y}(t,u)\beta(t,u+1)$는 '레이블 예측' 경로의 확률이며, $\lambda$는 트랜스듀서 손실과 정규화 손실의 균형을 맞추는 하이퍼파라미터입니다.

3. **그레디언트 해석**:
   - 이 새로운 손실 $\tilde{L}$에 대한 그레디언트 계산은 다음과 같습니다:
     $$ \frac{\partial \tilde{L}}{\partial \hat{y}(t,u)} = (1+\lambda)\frac{\partial L}{\partial \hat{y}(t,u)} $$
        $$ \frac{\partial \tilde{L}}{\partial b(t,u)} = \frac{\partial L}{\partial b(t,u)} $$
   - 이는 레이블 토큰을 방출하는 그레디언트가 스트리밍 ASR 네트워크로 '더 높은 학습률'로 역전파되고, 공백 토큰 방출 그레디언트는 동일하게 유지된다는 것을 의미합니다.
   - FastEmit는 개별 토큰이나 프레임 예측 확률이 아닌 정렬 확률에 기반하므로 시퀀스 수준 방출 정규화로 간주됩니다.

## 📊 Results

- **LibriSpeech 결과**:

  - FastEmit는 ContextNet 및 Conformer 모델에 적용 시 PR(Partial Recognition) 대기 시간을 약 200ms 크게 감소시켰습니다. 일부 모델에서는 심지어 음성보다 먼저 가설을 방출하여 음의 PR 대기 시간을 기록했습니다.
  - ContextNet-L 모델의 경우, WER을 TestClean에서 4.4%에서 3.1%로, TestOther에서 8.9%에서 7.5%로 향상시켰습니다. 90번째 백분위수 PR 대기 시간은 210ms에서 30ms로 감소했습니다.
  - 정확도 향상은 주로 삭제 오류(deletion errors) 감소에 기인합니다. LibriSpeech와 같은 장문(long-form) 음성 인식에서 FastEmit의 조기 방출 장려가 장문 RNN-T의 그레디언트 소실 문제에 도움이 됩니다.

- **하이퍼파라미터 $\lambda$ 연구**:

  - $\lambda$ 값이 커질수록 스트리밍 모델의 PR 대기 시간은 감소합니다.
  - 그러나 $\lambda$가 특정 임계값을 넘어서면 정규화가 너무 강해져 WER이 저하되기 시작합니다.
  - $\lambda$는 WER-대기 시간 간의 유연한 균형을 제공합니다.

- **MultiDomain 대규모 실험 결과 (Voice Search)**:
  - FastEmit는 RNN-T, Transformer-T, Conformer-T 모델에 적용되어 효과를 입증했습니다.
  - 기존 방법(Constrained Alignment, MaskFrame)과 비교했을 때, FastEmit는 150~300ms의 대기 시간 감소를 달성하면서도 더 나은 정확도를 보여주었습니다.
  - 예를 들어, RNN-T에서 Constrained Alignment (CA)가 WER을 6.0%에서 6.7%로 증가시킨 반면, FastEmit는 6.0%에서 6.2%로 소폭 증가에 그치면서도 PR 대기 시간을 크게 줄였습니다.
  - Voice Search와 같은 단문(short-query) 대화형 음성에서는 조기 방출이 오류 증가로 이어질 수 있으나, FastEmit는 모든 기술 중 가장 우수한 WER-대기 시간 트레이드오프를 제공합니다.

## 🧠 Insights & Discussion

- **시퀀스 수준 최적화의 중요성**: FastEmit는 트랜스듀서의 시퀀스 수준 최적화 방식과 일치하는 정규화를 적용함으로써, 기존의 토큰/프레임 기반 정규화에서 발생했던 정확도 저하 문제를 효과적으로 해결했습니다. 이는 트랜스듀서 모델에 더욱 적합한 접근 방식임을 시사합니다.
- **범용성 및 효율성**: 정렬 정보가 필요 없고, 하이퍼파라미터가 적으며, 추가 비용이 없어 다양한 트랜스듀서 기반 스트리밍 ASR 모델에 쉽고 효율적으로 적용할 수 있습니다. 이는 실제 시스템 배포에 큰 이점으로 작용합니다.
- **WER 및 대기 시간 트레이드오프 관리**: 하이퍼파라미터 $\lambda$를 통해 대기 시간과 WER 사이의 균형을 유연하게 조절할 수 있습니다. 애플리케이션의 요구 사항에 따라 최적의 $\lambda$를 선택할 수 있습니다.
- **문맥에 따른 효과 분석**: LibriSpeech와 같은 장문 읽기 음성에서는 조기 방출이 삭제 오류를 줄여 WER을 개선하는 긍정적인 효과를 보였습니다. 반면, Voice Search와 같은 단문 대화형 음성에서는 조기 방출이 약간의 WER 증가를 가져올 수 있지만, 전반적으로 다른 지연 감소 방법들보다 훨씬 우수한 대기 시간-정확도 트레이드오프를 제공했습니다.
- **한계**: $\lambda$ 값이 너무 클 경우 WER이 저하될 수 있다는 점은 정규화 강도 조절의 중요성을 보여줍니다. 이는 모든 유형의 데이터셋에서 무조건적인 정확도 향상을 보장하지 않을 수 있음을 의미합니다.

## 📌 TL;DR

FastEmit는 스트리밍 ASR에서 정확도 저하 없이 방출 지연을 줄이기 위해 트랜스듀서의 시퀀스 수준 손실 함수에 직접 적용되는 새로운 방출 정규화 방법입니다. '공백 토큰' 대신 '레이블 토큰'의 예측을 장려하도록 손실 함수를 수정함으로써, FastEmit는 정렬 정보나 추가 계산 비용 없이 다양한 트랜스듀서 모델의 대기 시간을 획기적으로 줄이고 WER도 개선했습니다. LibriSpeech에서는 대기 시간을 200ms 감소시키고 WER을 1% 이상 향상시켰으며, MultiDomain 데이터셋에서도 이전 방법들보다 뛰어난 WER-대기 시간 트레이드오프를 달성했습니다.
