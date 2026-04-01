# CONTEXT-AWARE TRANSFORMER TRANSDUCER FOR SPEECH RECOGNITION

Feng-Ju Chang, Jing Liu, Martin Radfar, Athanasios Mouchtaris, Maurizio Omologo, Ariya Rastrow, Siegfried Kunzmann

## 🧩 Problem to Solve

최신 종단 간(End-to-End, E2E) 자동 음성 인식(ASR) 시스템은 훈련 데이터에 자주 나타나지 않는 희귀 단어(예: 개별화된 장치 이름, 특정 엔티티)를 인식하는 데 어려움을 겪습니다. 이러한 희귀 단어의 인식 정확도를 향상시키기 위해 추론 시 맥락(contextual) 정보를 활용하는 효과적인 방법이 필요합니다.

## ✨ Key Contributions

- **새로운 컨텍스트 인식 Transformer Transducer (CATT) 네트워크 제안**: 기존 Transformer 기반 ASR 시스템에 맥락 신호를 통합하여 희귀 단어 인식 성능을 향상시킵니다.
- **멀티헤드 어텐션 기반 컨텍스트 바이어싱 네트워크 도입**: ASR 서브 네트워크와 함께 공동으로 훈련되며, 맥락 구문의 중요도를 학습합니다.
- **다양한 컨텍스트 인코딩 기법 탐색**: BLSTM 기반 모델과 사전 훈련된 BERT 기반 모델을 활용하여 맥락 데이터를 인코딩하며, 특히 BERT 기반 인코더의 우수성을 입증했습니다.
- **어텐션 쿼리 방식 비교**: 오디오 임베딩 단독 사용과 오디오 및 레이블 임베딩 동시 사용을 통해 맥락 임베딩에 대한 어텐션을 측정하고, 후자의 경우 추가적인 성능 향상을 보였습니다.
- **SOTA 성능 달성**: 사내 원거리 음성 데이터셋에서 기준 Transformer Transducer 및 기존 딥 컨텍스트 모델(C-LAS) 대비 각각 24.2% 및 19.4%의 WER 감소를 달성했습니다.

## 📎 Related Works

- **E2E ASR 시스템**: Connectionist Temporal Classification (CTC) [1], Listen-Attend-Spell (LAS) [2], Recurrent Neural Network Transducer (RNN-T) [3], Transformer [4] 등은 하이브리드 HMM-DNN 시스템에 비해 우수한 성능을 보여주었습니다.
- **E2E ASR의 희귀 단어 인식 한계**: 훈련 데이터의 부족으로 엔티티 이름이나 개인화된 단어 인식에 어려움이 있습니다 [9–11].
- **기존 맥락 활용 방법**:
  - **훈련 후 통합(Post-training integration)**: WFST(Weighted Finite State Transducer) [12]를 통한 Shallow Fusion [18] 또는 딥 퓨전(Deep Fusion) [20]을 통해 외부 언어 모델(LM)을 통합합니다.
  - **훈련 중 통합(During-training integration)**: Contextual LAS (C-LAS) [9]는 LAS에 편향 인코더와 위치 인식 어텐션을 추가하여 개인화된 단어에 대한 재평가를 수행합니다. Contextual RNN-T (C-RNN-T) [15]도 유사한 어텐션 메커니즘을 RNN-T에 적용했습니다. 음성 정보도 C-LAS 개선에 활용되었습니다 [11, 27].
- **Transformer 기반 모델**: Transformer [28] 및 Transformer Transducer [28–31]는 최신 ASR 모델로 자리매김했습니다.

## 🛠️ Methodology

1. **Transformer Transducer (기반 모델)**:
   - **오디오 인코더 ($f_{enc}$)**: 오디오 특징 $x$를 입력받아 오디오 임베딩 $h^{AE}_{t} \in \mathbb{R}^{d_{a} \times 1}$를 생성하는 스택형 자기-어텐션 Transformer 레이어.
   - **레이블 인코더 ($f_{pred}$)**: 이전 $L_1$개의 비공백(non-blank) 토큰 $y$를 사용하여 레이블 임베딩 $h^{LE}_{u} \in \mathbb{R}^{d_{l} \times 1}$를 생성하는 스택형 Transformer 네트워크.
   - **조인트 네트워크**: 오디오 인코더 출력과 레이블 인코더 출력을 결합하여 $z_{t,u} = \phi(U h^{AE}_{t} + V h^{LE}_{u} + b_1)$를 생성한 후, 선형 및 소프트맥스 레이어를 통해 출력 레이블에 대한 확률 분포 $p(y|t,u) = Softmax(W z_{t,u} + b_2)$를 산출합니다.
2. **Context-Aware Transformer Transducer (CATT)**: 기본 Transformer Transducer에 두 가지 구성 요소를 추가합니다 (Fig. 1 (b) 참조).
   - **컨텍스트 인코더 ($f_{context}$)**: 화자가 제공한 개인화된 장치 이름, 설정, 위치 등 맥락 구문 $w_k$를 고정 차원 벡터 $h^{CE}_{k} \in \mathbb{R}^{d_{c} \times 1}$로 인코딩합니다.
     - **BLSTM 기반**: BLSTM의 마지막 상태를 임베딩으로 사용하며, 모델과 함께 처음부터 훈련됩니다.
     - **BERT 기반**: 사전 훈련된 SmallBERT [35] 모델을 사용하며, 강력한 의미론적 사전 지식을 제공합니다. 이 인코더는 훈련 중에 고정될 수 있습니다.
   - **멀티헤드 어텐션(MHA) 기반 컨텍스트 바이어싱 레이어**:
     - 맥락 임베딩 $h^{CE}_{k}$와 발화 간의 관련성을 학습하여, 엔티티나 개인화된 단어에 더 많은 주의를 기울이도록 합니다.
     - **쿼리($Q_{cb}$)**: 오디오 인코더 출력 $X = [h^{AE}_{1},...,h^{AE}_{T}]^>$를 단독으로 사용하거나, 오디오 및 레이블 인코더 출력 $Y = [h^{LE}_{1},...,h^{LE}_{U}]^>$를 함께 사용하여 생성합니다 (Fig. 1 (c), (d) 참조).
     - **키($K_{cb}$) 및 값($V_{cb}$)**: 컨텍스트 임베딩 $C = [h^{CE}_{1},...,h^{CE}_{K}]^>$로부터 생성됩니다.
     - **크로스 어텐션 계산**: $H_{cb} = Softmax(\frac{Q_{cb}(K_{cb})^>}{\sqrt{d}})V_{cb}$.
     - **결합기(Combiner)**: 컨텍스트 인식 행렬 $H_{cb}$를 LayerNorm, 연결(concatenation), 피드포워드 투영(feed-forward projection) 레이어를 통해 오디오 (또는 오디오 + 레이블) 임베딩과 융합하여 컨텍스트 인식 임베딩 $H_{CA}$를 생성합니다.
     - 최종적으로 이 컨텍스트 인식 임베딩을 조인트 네트워크에 공급하여 정렬 학습을 개선합니다.

## 📊 Results

- **비컨텍스트 기반 모델 및 Shallow Fusion 대비 성능**:
  - CATT (SmallBERT-CE)는 baseline Transformer Transducer (T-T) 대비 `Personalized` 테스트 세트에서 4.7% WERR을 달성했으며, Shallow Fusion (SF)의 1.6%를 능가했습니다 (14M 모델).
  - 22M 모델에서는 CATT (SmallBERT-CE)가 T-T 대비 15.3% WERR을 달성하여 SF의 3%보다 훨씬 우수했습니다.
  - CATT와 SF를 결합하면 `Personalized` 및 `Common` 테스트 세트 모두에서 WERR이 추가로 향상되었습니다 (예: 14M 모델에서 6.3% 및 6.5%).
  - Shallow Fusion은 가중치에 민감하여 성능 저하가 발생할 수 있으나, CATT는 일관되게 WER을 개선했습니다.
- **컨텍스트 인코더 비교**: BERT 기반 컨텍스트 인코더는 BLSTM 기반 인코더보다 지속적으로 우수한 성능을 보였으며, 이는 강력한 의미론적 사전 지식의 중요성을 강조합니다.
- **어텐션 쿼리 방식 비교**: 오디오 임베딩과 레이블 임베딩을 모두 사용하여 컨텍스트에 어텐션하는 방식(audio+label-Q)이 오디오 임베딩만 사용하는 방식(audio-Q)보다 더 큰 성능 향상을 가져왔습니다 (예: 22M SmallBERT-CE 모델에서 `Personalized` 테스트 세트의 WERR이 audio-Q의 13.6%에서 audio+label-Q의 24.2%로 증가).
- **컨텍스트 엔티티 수에 따른 견고성**: CATT의 WERR은 다양한 수의 컨텍스트 구문에 대해 일관되게 유지되어, 관련 없는 컨텍스트에 대한 모델의 견고성을 입증했습니다.
- **Contextual LAS (C-LAS) 대비 성능**:
  - 동일한 BLSTM 컨텍스트 인코더를 사용할 경우 CATT는 C-LAS 대비 약 3-6%의 상대적 WER 개선을 보였습니다.
  - BERT 기반 컨텍스트 임베딩을 활용한 CATT는 22M 모델에서 최대 8.1%의 상대적 개선을 달성했습니다 (audio-Q).
  - 오디오와 레이블 임베딩을 모두 쿼리로 사용하는 CATT (audio+label-Q)는 22M 모델에서 C-LAS 대비 최대 21.7%의 상대적 개선을 보여 가장 우수한 성능을 기록했습니다.

## 🧠 Insights & Discussion

- CATT는 멀티헤드 크로스-어텐션 메커니즘을 통해 맥락 데이터의 관련성을 효과적으로 학습하고, 이를 ASR 모델에 주입하여 희귀 단어 인식 성능을 크게 향상시킵니다.
- 사전 훈련된 BERT 모델을 컨텍스트 인코더로 활용하는 것은 강력한 의미론적 사전 지식을 활용하여 BLSTM 기반 인코더보다 훨씬 우수한 성능을 제공함을 확인했습니다. 이는 맥락 표현의 풍부함이 중요함을 시사합니다.
- 오디오 임베딩뿐만 아니라 레이블 임베딩까지 쿼리로 사용하여 맥락에 어텐션하는 것은 추가적인 성능 향상을 가져옵니다. 이는 맥락 정보가 입력 오디오와 출력 토큰 예측 모두와 연관되어 있으며, 이들을 함께 보정할 때 더 나은 정렬 학습이 가능하다는 가설을 뒷받침합니다.
- CATT는 shallow fusion과 달리 부스팅 가중치에 민감하지 않고 일관된 WER 개선을 제공하여 더 안정적인 솔루션임을 보여줍니다.
- 어텐션 가중치 시각화를 통해 CATT가 실제 엔티티에 높은 어텐션 가중치를 부여하여 맥락을 효과적으로 활용하고 있음을 알 수 있습니다.

## 📌 TL;DR

Transformer Transducer ASR 시스템이 희귀 단어 인식에 어려움을 겪는 문제를 해결하기 위해, 본 논문은 **CATT(Context-Aware Transformer Transducer)**를 제안합니다. CATT는 **멀티헤드 어텐션 기반의 컨텍스트 바이어싱 네트워크**를 사용하여 오디오 및 레이블 임베딩과 컨텍스트 임베딩 간의 관계를 학습합니다. **사전 훈련된 BERT 모델**을 컨텍스트 인코더로 활용하고 오디오 및 레이블 임베딩을 모두 컨텍스트 어텐션 쿼리로 사용함으로써, 기존 Transformer Transducer, shallow fusion, 그리고 C-LAS 대비 **최대 24.2%의 WER 감소**를 달성하여 희귀 단어 인식 성능을 크게 향상시켰습니다.
