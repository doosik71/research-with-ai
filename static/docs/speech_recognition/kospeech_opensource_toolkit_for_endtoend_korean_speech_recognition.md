# KoSpeech: Open-Source Toolkit for End-to-End Korean Speech Recognition

Soohwan Kim, Seyoung Bae, Cheolhwang Won, Suwon Park

## 🧩 Problem to Solve

기존의 자동 음성 인식(ASR) 오픈소스 툴킷들은 대부분 영어와 같은 비한국어에 초점을 맞추고 있어, 한국어 음성 인식 연구를 위한 진입 장벽이 높았습니다. 특히, AI Hub에서 KsponSpeech라는 1,000시간 분량의 한국어 대화 코퍼스를 공개했음에도 불구하고, 해당 코퍼스에 대한 표준화된 전처리 방법과 모델 성능을 비교할 수 있는 **명확한 베이스라인 모델이 부재**했습니다. 이로 인해 연구자들이 모델 성능을 서로 비교하기 어려운 문제가 있었습니다.

## ✨ Key Contributions

- **한국어 종단간(End-to-End) ASR 툴킷 KoSpeech 공개:** PyTorch 기반의 모듈화되고 확장 가능한 오픈소스 툴킷을 제공하여 한국어 음성 인식 연구를 위한 가이드라인을 제시합니다.
- **KsponSpeech 코퍼스 전처리 방법론 제안:** KsponSpeech 코퍼스의 특수 토큰 처리, 선택적 전사(transcript) 방식 (발음 전사 vs. 표기 전사), 다양한 기본 단위(문자, 서브워드, 음소) 지원 등 세부적인 전처리 방법을 제시합니다.
- **LAS(Listen, Attend and Spell) 기반 베이스라인 모델 제공:** 다양한 훈련 하이퍼파라미터를 쉽게 사용자 정의할 수 있는 LAS 기반 베이스라인 모델을 제시하며, KsponSpeech 코퍼스에서 10.31%의 CER(Character Error Rate)을 달성했습니다.
- **다양한 음향 모델 구성 요소 실험:** CNN 특징 추출기(Deep Speech 2, VGG Net), 다양한 어텐션 메커니즘(Dot-Product, Additive, Multi-Head, Location-Aware), 최적화 기법(Scheduled Sampling, Label Smoothing, Learning Rate Scheduling) 등을 KoSpeech 툴킷 내에서 실험하고 결과를 제시합니다.
- **EEE(Easy to use, Easy to read, Easy to expand) 원칙 기반 설계:** KoSpeech가 사용하기 쉽고, 코드를 읽기 쉬우며, 확장이 용이하도록 설계되었음을 강조합니다.

## 📎 Related Works

- **종단간 음성 인식(End-to-End ASR):** 기존의 복잡한 DNN-HMM 기반 하이브리드 시스템의 단점을 극복하기 위해 제안되었으며, 단일 시스템으로 전체 과정을 처리합니다. LAS [7]와 같은 모델들이 대표적입니다.
- **Sequence-to-Sequence 프레임워크:** 가변 길이 입출력 시퀀스 문제를 해결하기 위해 제안되었으며, 기계 번역 [27, 28], 이미지 캡셔닝 [29, 30] 등 다양한 분야에 적용되었습니다. 음성 인식에서도 직접적인 응용이 가능합니다 [11, 12].
- **어텐션 메커니즘:** 디코더 RNN이 출력 토큰을 생성할 때 더 많은 정보를 제공하여 모델 성능을 크게 향상시킵니다 [10, 12].
- **대규모 공개 음성 코퍼스:** LibriSpeech [18], Wall Street Journal [19], Switchboard [20], CallHome [21] 등 과거와 현재의 벤치마크 데이터셋들이 ASR 모델 발전에 기여했습니다. 하지만 대부분이 영어 또는 비한국어 데이터셋에 집중되어 있었습니다.
- **KsponSpeech 코퍼스:** AI Hub에서 2018년에 공개한 1,000시간 분량의 한국어 대화 코퍼스로, 한국어 음성 인식 연구의 진입 장벽을 낮추는 데 기여했습니다.
- **데이터 증강 기법 (SpecAugment):** 입력 오디오의 로그 멜 스펙트로그램에 직접 적용하여 추가 데이터 생성 없이 모델 성능을 향상시키는 방법입니다 [43].

## 🛠️ Methodology

KoSpeech는 전처리, 음향 모델 구축, 최적화 등 종단간 ASR 시스템의 전반적인 부분을 포괄합니다.

1. **전사(Transcript) 전처리:**

   - **통계 분석:** 62만 개 이상의 스크립트 파일 중 시퀀스 길이가 100이 넘는 긴 문장을 제외하여 훈련 속도 향상 및 메모리 사용량 감소.
   - **특수 토큰 처리:** ETRI 전사 규칙에 따른 배경 소음, 발화자의 숨소리 등 ASR에 불필요한 토큰 및 의미 없는 감탄사 제거.
   - **선택적 전사:** 원본 스크립트의 표기(정확한) 표현과 발음(음성) 표현 중 선택 가능. 일반적으로 발음 표현을 채택하며, 숫자 표현 등 특정 경우에는 옵션으로 선택할 수 있도록 함.
   - **가변 기본 단위:** 문자, 서브워드(SentencePiece 또는 KoBERT 토크나이저), 음소(hgtk) 단위로 전처리 지원.

2. **음성 신호 처리:**

   - **무음 제거:** KsponSpeech 오디오 파일의 무음 구간(30dB 임계치)을 제거하여 훈련 속도를 높이고 불필요한 정보를 줄임.
   - **특징 추출:**
     - **Windowing:** 음성 신호를 10~25ms 프레임으로 분할하고 해밍 윈도우 적용.
     - **Spectrogram:** 시간 변화에 따른 주파수 스펙트럼의 시각적 표현. STFT(Short-Time Fourier Transform)를 통해 계산.
     - **Mel Spectrogram:** 인간의 비선형적인 주파수 지각 특성을 반영하여 멜 스케일로 변환된 스펙트로그램.
     - **Log scaled spectrogram:** 인간이 저전력 수준에서 강도 차이를 더 잘 인지하는 특성을 반영하여 로그 스케일 적용.
     - **MFCC (Mel-Frequency Cepstral Coefficients) 및 Mel 필터 뱅크:** 음향 모델의 여기(excitation)와 공명(formant) 분리를 목표로 하며, 멜 스케일에서 로그 파워 스펙트럼에 선형 코사인 변환을 적용하여 계산.
   - **SpecAugment:** 입력 오디오의 로그 멜 스펙트로그램에 직접 적용되는 데이터 증강 기법. 주파수 마스킹($f$개의 연속 주파수 채널 마스킹)과 시간 마스킹($t$개의 연속 시간 스텝 마스킹)을 포함.

3. **음향 모델 (LAS 기반):**

   - **Listener (인코더):** 음성 특징을 고수준 표현으로 인코딩.
     - **CNN 특징 추출기:** Deep Speech 2 (DS2) 및 VGG Net [23, 24]의 초기 CNN 레이어를 사용.
     - **RNN:** CNN 추출기 후 3개의 양방향 LSTM(BLSTM) 레이어(각 방향 512 유닛) 사용.
   - **Speller (디코더):** 인코더 출력과 어텐션 메커니즘을 사용하여 문자 확률 분포를 예측.
     - 2개의 단방향 LSTM(각 1024 유닛)과 2개의 투영(projection) 레이어 사용.
     - **잔여 연결(Residual connection)** [26]을 사용하여 vanishing gradient 문제 방지 및 디코더 RNN 정보 유지.
   - **어텐션 메커니즘:** 인코더 출력과 디코더 은닉 상태 간의 정렬을 학습.
     - Scaled Dot-Product Attention: 쿼리와 키의 내적 후 $\sqrt{d_k}$로 스케일링.
     - Additive Attention: 쿼리와 키 벡터를 연결하고 단일 은닉 계층 피드포워드 네트워크 사용.
     - Location-Aware Attention: 이전 어텐션 분포를 고려하여 음성 인식에 적합 [12].
     - Multi-Head Attention (MHA): 여러 어텐션 헤드를 병렬로 실행하여 다양한 어텐션 분포 생성 [26]. (베이스라인 모델에 4개 헤드 사용)

4. **최적화:**
   - **Scheduled Sampling:** 훈련 초반에는 정답 레이블(teacher forcing)을 사용하고, 점차 모델 예측을 다음 입력으로 사용하는 확률을 증가시켜 훈련과 추론 간의 불일치(discrepancy)를 줄임. (에포크당 2% 감소, 최소 0.8까지)
   - **Label Smoothing:** 모델 예측의 자신감을 낮추고, 높은 엔트로피를 장려하여 모델의 적응성을 높이는 정규화 기법. (epsilon 0.1 사용)
   - **Learning Rate Scheduling:** Adam 옵티마이저 [48] 사용. 400 스텝 동안 0에서 3e-04까지 학습률 워밍업 후, 검증 손실이 개선되지 않을 경우 학습률 감소(PyTorch의 `ReduceLROnPlateau` 사용).

## 📊 Results

- **최종 성능 (베이스라인 모델):** KsponSpeech 코퍼스(음향 모델만 사용)에서 **10.31%의 CER**을 달성했습니다.
- **특징 비교:**
  - 가장 좋은 성능은 필터 뱅크(filter bank) (80차원)에서 10.31% CER을 기록했습니다.
  - 로그 멜 스펙트로그램, 로그 스펙트로그램, MFCC 순으로 성능이 저하되었습니다.
- **CNN 추출기 비교:**
  - VGG Net 추출기가 Deep Speech 2 추출기보다 더 나은 성능을 보였습니다. (VGG: 10.31% vs. DS2: 12.39%)
- **어텐션 메커니즘 비교:**
  - Multi-Head Attention이 가장 우수한 성능을 보였습니다. (10.31%)
  - 그 다음으로 Location-Aware, Additive, Scaled Dot-Product 순으로 성능을 보였습니다.
- **디코딩 방식:**
  - 빔 서치(Beam search)가 탐욕적 서치(Greedy search)보다 평균적으로 2% 높은 CER을 기록했습니다. 이는 외부 언어 모델의 부재 때문으로 분석됩니다.

## 🧠 Insights & Discussion

- **음향 특징의 중요성:** 로그 스펙트로그램은 더 많은 암묵적인 정보를 포함하지만, 발음 정보를 명시적으로 표현하는 데는 필터 뱅크가 더 효과적일 수 있다는 가설을 제시합니다. MFCC의 성능 저하는 차원 축소의 영향으로 해석됩니다.
- **디코딩 메커니즘의 한계:** 외부 언어 모델 부재 시 빔 서치가 탐욕적 서치보다 성능이 낮게 나올 수 있음을 발견했습니다. 이는 빔 서치 디코딩 단계에서 더 정교한 재평가(rescoring) 시스템이 필요함을 시사합니다.
- **최적화 기법의 영향:** 음향 모델 자체만큼이나 Scheduled Sampling, Label Smoothing, Learning Rate Scheduling과 같은 최적화 기법의 적용이 모델 성능에 큰 영향을 미 미친다는 점을 강조합니다.
- **KoSpeech의 활용 가치:** 한국어 음성 인식 연구를 시작하려는 연구자들에게 KsponSpeech 코퍼스를 활용한 베이스라인 모델과 전처리 방법을 제공함으로써 중요한 가이드라인 역할을 할 수 있습니다.

**향후 계획:**

- 외부 언어 모델과의 얕은 융합(shallow fusion) [45] 추가.
- 문자 외에 음소(grapheme), 서브워드(sub-word) 등 더 다양한 인식 단위 지원.
- 학습 시간 단축을 위해 트랜스포머(Transformer) 구조 추가.

## 📌 TL;DR

KoSpeech는 한국어 음성 인식 연구의 진입 장벽을 낮추기 위해 KsponSpeech 코퍼스에 대한 **표준 전처리 및 베이스라인 모델이 부재한 문제**를 해결하고자 합니다. 본 논문은 **LAS 기반의 종단간 한국어 ASR 툴킷 KoSpeech**를 제안하며, **KsponSpeech 전처리 방법**과 **VGG Net 추출기 및 Multi-Head Attention을 활용한 베이스라인 모델**을 제시합니다. 이 모델은 KsponSpeech에서 **10.31%의 CER**을 달성했으며, 다양한 특징, 인코더, 어텐션 메커니즘, 최적화 기법에 대한 실험 결과를 제공하여 한국어 음성 인식 연구의 가이드라인이 되기를 기대합니다.
