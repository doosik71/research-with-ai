# Jasper: An End-to-End Convolutional Neural Acoustic Model

Jason Li, Vitaly Lavrukhin, Boris Ginsburg, Ryan Leary, Oleksii Kuchaiev, Jonathan M. Cohen, Huyen Nguyen, Ravi Teja Gadde

## 🧩 Problem to Solve

기존 자동 음성 인식(ASR) 시스템은 음향 모델, 발음 모델, 언어 모델 등 여러 개별적으로 학습된 구성 요소로 이루어져 복잡성이 높았습니다. 이 논문은 이러한 복잡성을 줄이고, 복잡한 구성 요소 없이 단일 DNN(심층 신경망)으로 평면 학습(flat-start training)을 가능하게 하는 End-to-End(E2E) ASR 시스템의 성능을 향상시키는 것을 목표로 합니다. 특히, 1D 컨볼루션 레이어를 기반으로 한 E2E 모델의 깊이와 용량을 확장하여 LibriSpeech와 같은 벤치마크에서 최첨단(SOTA) 결과를 달성하고자 합니다.

## ✨ Key Contributions

- **계산 효율적인 End-to-End 컨볼루션 신경망 음향 모델 제시:** Jasper 아키텍처는 GPU 학습 및 추론에 최적화된 1D 컨볼루션, 배치 정규화(Batch Normalization), ReLU, 드롭아웃(Dropout), 잔차 연결(Residual Connections)만을 사용하여 깊고 확장 가능한 모델을 제안합니다.
- **정규화 및 활성화 함수 효과 입증:** ReLU와 배치 정규화 조합이 테스트된 다른 조합보다 성능이 우수하며, 깊은 모델의 수렴을 위해 잔차 연결이 필수적임을 실험적으로 보였습니다.
- **새로운 계층별 최적화 도구 NovoGrad 도입:** Adam 최적화 도구와 유사하지만, 두 번째 모멘트(second moments)를 가중치별이 아닌 계층별로 계산하여 메모리 사용량을 줄이고 수치적 안정성을 높인 NovoGrad를 제안합니다.
- **LibriSpeech 테스트 세트에서 SOTA WER 달성:** 외부 신경망 언어 모델(Transformer-XL)과 빔 서치(beam-search) 디코더를 사용하여 LibriSpeech test-clean에서 2.95%의 WER(Word Error Rate)을 달성하여 최첨단 성능을 갱신했습니다.

## 📎 Related Works

- **전통적인 ASR 시스템:** Hidden Markov Model (HMM)과 신경망을 결합한 하이브리드 시스템 [1, 2, 3, 4].
- **End-to-End ASR:** Connectionist Temporal Classification (CTC) 손실 [5], 심층 컨볼루션 신경망을 이용한 E2E 시스템 [6].
- **Wav2letter:** 1D 컨볼루션 레이어를 사용하는 E2E ConvNet 기반 음성 인식 시스템 [7]. Liptchinsky et al. [8]은 Wav2letter를 개선하여 모델 깊이를 늘리고 Gated Linear Units (GLU) [9], Weight Normalization [10] 및 Dropout을 추가했습니다.
- **최적화 도구:** Adam 최적화 도구 [15], ND-Adam [29], AdamW [30].
- **정규화 기법:** Batch Normalization [11], Weight Normalization [10], Layer Normalization [18], 시퀀스 마스킹(Sequence Masking) [20].
- **잔차 연결 및 밀집 연결:** ResNet의 잔차 연결, DenseNet [16], DenseRNet [17].
- **언어 모델:** N-gram 언어 모델 [24], 신경망 언어 모델 (RNN, LSTM) [21, 22, 23], Transformer-XL [12].
- **데이터 증강:** Speed perturbation [32], SpecAugment [28].

## 🛠️ Methodology

Jasper는 End-to-End ASR을 위한 컨볼루션 신경망 기반의 음향 모델입니다.

1. **아키텍처 (JasperBxR):**

   - 멜 필터뱅크(mel-filterbank) 특징을 입력으로 받아 프레임당 문자 확률 분포를 출력합니다.
   - $B$개의 블록과 각 블록 내 $R$개의 서브 블록으로 구성됩니다.
   - **서브 블록:** 1D 컨볼루션, 배치 정규화, ReLU, 드롭아웃의 순서로 구성됩니다. 추론 시에는 GPU 최적화를 위해 드롭아웃이 제거되고 배치 정규화, ReLU, 잔차 합산이 컨볼루션과 융합됩니다.
   - **블록:** 모든 서브 블록은 동일한 수의 출력 채널을 가집니다.
   - **추가 컨볼루션 블록:** 전처리(1개) 및 후처리(3개)를 위한 4개의 추가 컨볼루션 블록이 있습니다.

2. **잔차 연결 (Residual Connections):**

   - 블록 입력은 1x1 컨볼루션과 배치 정규화 계층을 거쳐 마지막 서브 블록의 배치 정규화 출력에 더해집니다. 이 합산 결과가 활성화 함수와 드롭아웃을 거쳐 블록의 최종 출력이 됩니다.
   - **Dense Residual (DR):** DenseNet [16], DenseRNet [17]에서 영감을 받아 각 컨볼루션 블록의 출력을 모든 후속 블록의 입력에 더하는 방식 (DenseNet/DenseRNet은 연결(concatenation)을 사용하지만 Jasper DR은 덧셈(addition)을 사용)입니다. 깊은 모델 학습에 필수적임을 확인했습니다.

3. **정규화 및 활성화 함수 선택:**

   - Batch Norm, Weight Norm, Layer Norm과 ReLU, Clipped ReLU, Leaky ReLU, GLU, GAU 등 다양한 조합을 실험했습니다.
   - 작은 모델(Jasper5x3)에서는 Layer Norm + GAU가, 큰 모델(Jasper10x4)에서는 **Batch Norm + ReLU**가 가장 좋은 성능을 보여 최종 아키텍처로 채택되었습니다.
   - 시퀀스 패딩(padding) 값으로 인한 문제 해결을 위해 Layer Norm 및 Batch Norm 계산 시 시퀀스 마스킹(masking)을 적용했습니다.

4. **언어 모델 (Language Model):**

   - 빔 서치 디코딩 시 음향 점수와 언어 모델 점수를 모두 활용합니다.
   - 통계적 N-gram 언어 모델과 신경망 Transformer-XL 언어 모델 [12]을 사용했습니다.
   - 최종 SOTA 결과는 빔 폭(beam width) 2048의 빔 서치와 외부 Transformer-XL LM을 사용한 재점수화(rescoring)를 통해 얻었습니다.

5. **NovoGrad 최적화 도구:**
   - SGD (Momentum 포함) 또는 NovoGrad를 사용했습니다.
   - NovoGrad는 Adam [15]과 유사하지만, 두 번째 모멘트 $v_{l}^{t}$를 가중치별이 아닌 **계층별(per layer)**로 계산합니다:
     $$v_{l}^{t} = \beta_{2} \cdot v_{l}^{t-1} + (1-\beta_{2}) \cdot ||g_{l}^{t}||^{2} \tag{1}$$
   - $v_{l}^{t}$는 그래디언트 $g_{l}^{t}$의 스케일을 조정하는 데 사용됩니다:
     $$m_{l}^{t} = \beta_{1} \cdot m_{l}^{t-1} + \frac{g_{l}^{t}}{\sqrt{v_{l}^{t} + \epsilon}} \tag{2}$$
   - L2-정규화(weight decay $d \cdot w^{t}$)가 사용되는 경우:
     $$m_{l}^{t} = \beta_{1} \cdot m_{l}^{t-1} + \frac{g_{l}^{t}}{\sqrt{v_{l}^{t} + \epsilon}} + d \cdot w^{t} \tag{3}$$
   - 최종 가중치 업데이트:
     $$w^{t+1} = w^{t} - \alpha_{t} \cdot m^{t} \tag{4}$$
   - 메모리 소비를 줄이고 수치적 안정성을 향상시킵니다. NovoGrad 사용 시 LibriSpeech dev-clean에서 WER이 4.00%에서 3.64%로 9% 상대적 개선을 보였습니다.

## 📊 Results

- **LibriSpeech (test-clean):**
  - 외부 신경망 언어 모델(Transformer-XL)을 사용한 빔 서치 디코더로 2.95% WER 달성 (SOTA).
  - 언어 모델 없는 Greedy 디코더로 3.86% WER.
  - Time/Frequency 마스크를 사용한 Jasper DR 10x5 + Transformer-XL 모델은 2.84% WER 달성.
- **Wall Street Journal (WSJ):**
  - Jasper 10x3 모델이 Transformer-XL LM과 함께 nov93에서 9.3%, nov92에서 6.9% WER을 기록하며 경쟁력 있는 결과를 보였습니다.
- **Hub5’00 (대화체 음성):**
  - Jasper DR 10x5 모델이 Transformer-XL LM과 함께 Switchboard (SWB)에서 7.8%, Callhome (CHM)에서 16.2% WER을 기록했습니다. SWB에서는 좋은 결과를 보였으나 CHM에서는 개선의 여지가 있습니다.
- **NovoGrad 효과:** Jasper DR 10x5 모델에서 SGD 대비 LibriSpeech dev-clean WER을 4.00%에서 3.64%로 감소시켜 9%의 상대적 개선을 보였습니다.
- **정규화 및 활성화:** 깊은 모델에서는 Batch Norm + ReLU가 다른 조합보다 우수하며, 잔차 연결 없이는 깊은 모델이 수렴하지 못함을 확인했습니다.

## 🧠 Insights & Discussion

- **깊고 단순한 컨볼루션 모델의 잠재력:** Jasper는 1D 컨볼루션, 배치 정규화, ReLU, 드롭아웃, 잔차 연결 등 표준 구성 요소만을 사용하여 깊고 확장 가능한 아키텍처를 구축함으로써 End-to-End ASR에서 최첨단 성능을 달성할 수 있음을 입증했습니다. 이는 복잡한 아키텍처 요소 없이도 높은 효율성과 성능을 확보할 수 있다는 중요한 통찰을 제공합니다.
- **잔차 연결의 중요성:** 모델 깊이를 5x3 이상으로 확장할 때 잔차 연결이 학습 수렴에 필수적이라는 점을 명확히 보여주었습니다. 특히, Dense Residual 구조는 DenseNet과 유사한 효과를 보이면서도 파라미터 성장 인자 튜닝이 필요 없어 더 실용적입니다.
- **NovoGrad의 효율성:** 제안된 NovoGrad 최적화 도구는 Adam과 유사한 성능을 보이면서도 계층별 모멘트 계산을 통해 메모리 효율성을 높이고 수치적 안정성을 개선하여 깊은 신경망 학습에 유리합니다.
- **언어 모델의 영향:** 외부 신경망 언어 모델(Transformer-XL)이 WER 개선에 강력한 영향을 미치며, 언어 모델의 품질(퍼플렉서티)과 WER 사이에 강한 상관관계가 있음을 보여주었습니다.
- **GPU 최적화:** Jasper 서브 블록은 GPU 추론에 최적화되도록 설계되어, 배치 정규화, ReLU, 잔차 합산 등을 컨볼루션과 융합하여 단일 GPU 커널로 처리할 수 있습니다.
- **한계 및 향후 연구:** 현재 접근 방식이 더 깊은 모델과 더 큰 데이터셋에도 계속 확장될 수 있을지 탐구하는 것이 향후 연구 과제입니다. 또한, 더욱 정교한 정규화 기법, 데이터 증강, 손실 함수, 언어 모델 및 최적화 전략을 탐색할 수 있는 좋은 기준선 역할을 합니다.

## 📌 TL;DR

Jasper는 1D 컨볼루션, 배치 정규화, ReLU, 드롭아웃 및 잔차 연결로 구성된 계산 효율적인 End-to-End 컨볼루션 신경망 음향 모델입니다. 이 모델은 새로운 계층별 최적화 도구인 NovoGrad와 Transformer-XL 언어 모델과의 결합을 통해 LibriSpeech test-clean에서 2.95%의 WER로 최첨단 성능을 달성했습니다. 깊은 모델의 학습 수렴을 위해 잔차 연결이 필수적이며, 배치 정규화와 ReLU의 조합이 효과적임을 보였습니다.
