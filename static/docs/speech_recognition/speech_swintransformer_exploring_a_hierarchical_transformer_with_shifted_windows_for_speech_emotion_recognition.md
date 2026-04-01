# SPEECH SWIN-TRANSFORMER: EXPLORING A HIERARCHICAL TRANSFORMER WITH SHIFTED WINDOWS FOR SPEECH EMOTION RECOGNITION

Yong Wang, Cheng Lu, Hailun Lian, Yan Zhao, Björn W. Schuller, Yuan Zong, Wenming Zheng

## 🧩 Problem to Solve

- 음성 감정 인식(SER)은 인간의 음성에서 감정 범주를 분석하고 인식하는 것을 목표로 합니다.
- 감정 정보는 단어, 구, 발화 등 다양한 스케일의 음성 특징에 걸쳐 분포되어 있습니다.
- 기존 Transformer 모델들은 음성 특징 내의 장거리 의존성 포착에는 능숙하지만, 패치 간의 상호 관계, 특히 패치 경계의 정보 집계에는 취약하며, 이는 다중 스케일 감정 특징 표현에 있어 핵심적인 단서가 됩니다.

## ✨ Key Contributions

- 음성 감정 인식을 위한 새로운 계층적 Transformer 모델인 "Speech Swin-Transformer"를 제안합니다.
- Shifted Windows를 활용한 계층적 특징 표현을 통해 다중 스케일 감정 특징을 효과적으로 통합합니다.
- 로컬 윈도우 Transformer(Local Windows Transformer)를 도입하여 각 세그먼트 패치 내의 프레임 간 지역적 감정 정보를 탐색합니다.
- Shifted Windows Transformer를 설계하여 로컬 윈도우 간의 연결을 모델링하고 세그먼트 패치 경계 근처의 패치 간 상관관계를 보완합니다.
- 패치 병합(Patch Merging) 연산을 사용하여 Transformer의 수용 필드를 프레임 레벨에서 세그먼트 레벨로 확장하며 계층적 음성 표현을 구성합니다.
- 제안된 모델이 IEMOCAP 및 CASIA 데이터셋에서 기존 최신(state-of-the-art) 방법들보다 우수한 성능을 달성함을 입증했습니다.

## 📎 Related Works

- **초기 SER 연구:** 저수준 기술자(LLD) 특징(예: 기본 주파수, MFCC)과 전통적인 머신러닝 모델(예: GMM, HMM, SVM)을 결합했습니다.
- **딥러닝 기반 SER:** CNN, LSTM, Transformer와 같은 딥 신경망(DNN) 방법들이 뛰어난 성능을 달성했습니다.
- **Transformer 기반 SER:**
  - Wang et al. [9]: CNN 및 LSTM으로 시간적 감정 정보를 추출한 후 Transformer 레이어를 쌓아 특징 추출을 강화했습니다.
  - Lu et al. [10] (LGFA): 지역 및 전역 감정 특징을 융합하기 위해 프레임 레벨 Transformer를 세그먼트 레벨 Transformer 내에 중첩시켰습니다.
  - Wang et al. [11] (TF-Transformer): 시간-주파수 도메인에서 각각 지역 감정 특징을 추출하고, 여러 Transformer를 통해 전역 감정 표현으로 통합했습니다.
- **컴퓨터 비전 분야 Transformer:**
  - Vision Transformer (ViT) [21]
  - Transformer in Transformer (TNT) [22]
  - Swin-Transformer [12]: Shifted Windows를 사용하는 계층적 비전 Transformer로, 본 연구에 영감을 주었습니다.

## 🛠️ Methodology

제안하는 Speech Swin-Transformer는 4개의 스테이지로 구성되며, 각 스테이지는 주로 세 가지 모듈을 포함합니다: Local Windows Transformer, Shifted Windows Transformer, 그리고 Patch Merging 모듈입니다.

1. **입력 처리:**

   - 음성에서 추출된 로그-멜 스펙트로그램 특징 $x \in \mathbb{R}^{b \times c \times f \times d}$을 모델의 입력으로 사용합니다. 여기서 $b$는 배치 크기, $c$는 채널 수, $f$는 멜 필터 뱅크 수, $d$는 음성 프레임 수입니다.
   - 모델의 계산 복잡도를 줄이기 위해 시간 도메인에서 $N$개의 블록으로 스펙트로그램을 분할하고, 이를 선형 투영하여 첫 번째 Transformer의 입력 $\hat{x}' \in \mathbb{R}^{b \times h \times e}$로 변환합니다. 여기서 $h=f \times d/N$이며, $e$는 임베딩 차원입니다.

2. **Local Windows Transformer:**

   - 주요 역할은 각 시간 도메인 윈도우 내에서 MSA (Multihead Self-Attention) 메커니즘을 통해 감정적 상관관계를 계산하는 것입니다.
   - 입력 특징 $\hat{x}'$을 크기 $[f, t]$의 윈도우로 분할하여 $M$개의 시간 도메인 블록을 얻고 ($M=d/(t \times N)$), 각 윈도우 내에서 MSA를 수행합니다.
   - 처리 과정은 다음과 같습니다:
     $$l = \text{WP}(\text{LN}(\hat{x}'))$$
     $$m = \text{WM}(\text{MSA}(l)) + \hat{x}'$$
     $$s = \text{MLP}(\text{LN}(m)) + m$$
     여기서 $\text{LN}(\cdot)$은 Layer Normalization, $\text{WP}(\cdot)$은 Window Partition, $\text{MSA}(\cdot)$는 Multihead Self-Attention, $\text{WM}(\cdot)$은 Window Merging, $\text{MLP}(\cdot)$은 Multi-Layer Perceptron입니다.

3. **Shifted Windows Transformer:**

   - Local Windows Transformer가 윈도우 간의 연결을 무시하는 문제를 해결하기 위해 설계되었습니다.
   - 윈도우 분할 전에 추가적인 시프트(shift) 연산을 수행하며, 이동 단계는 윈도우 시간 도메인 길이의 절반으로 설정됩니다. 이를 통해 윈도우 간의 연결에 집중할 수 있습니다.
   - 처리 과정은 다음과 같습니다:
     $$\hat{l} = \text{WP}(\text{SF}(\text{LN}(s)))$$
     $$\hat{m} = \text{WM}(\text{MSA}(\hat{l})) + s$$
     $$\hat{s} = \text{MLP}(\text{LN}(\hat{m})) + \hat{m}$$
     여기서 $\text{SF}(\cdot)$는 Swin Transformer와 유사한 윈도우 이동을 나타냅니다.

4. **Patch Merging Module:**

   - 다운샘플링 연산을 통해 특징 해상도를 줄이고 채널 수를 조정하여 계산 복잡도를 줄이고 계층적 구조를 형성합니다.
   - 특징의 시간 도메인과 주파수 도메인에서 일정 간격으로 요소를 선택하여 전체 텐서로 연결합니다. 채널 차원은 원래 크기의 4배가 된 후, 완전 연결 계층을 통해 원래 크기의 2배로 조정됩니다.
   - 처리 과정은 다음과 같습니다:
     $$\hat{s}' = \text{PM}(\hat{s})$$
     여기서 $\text{PM}(\cdot)$은 Patch Merging을 나타냅니다.

5. **최종 분류:**
   - 마지막 스테이지의 Shifted Windows Transformer 출력은 패치 병합 모듈을 거치지 않고 정규화, 평균 풀링, 평탄화되어 특징 $y \in \mathbb{R}^{b \times 8e}$를 얻습니다.
   - 이 특징 $y$를 완전 연결 계층 $(\text{FC}(\cdot))$에 통과시켜 예측된 감정 확률을 계산하고, Softmax 함수를 사용하여 최종 감정을 결정합니다. 모델은 예측된 감정 레이블과 실제 감정 레이블 $z \in \mathbb{R}^{b \times k}$ 간의 교차 엔트로피 손실($\text{CrossEntropyLoss}(\cdot)$)을 최소화하여 최적화됩니다.

## 📊 Results

- **실험 데이터베이스:** IEMOCAP (4가지 감정: 분노, 기쁨, 중립, 슬픔) 및 CASIA (6가지 감정: 분노, 기쁨, 공포, 슬픔, 중립, 놀람) 데이터셋을 사용했습니다.
- **실험 프로토콜:** Leave-One-Speaker-Out (LOSO) 교차 검증을 사용했으며, 평가 지표는 가중 평균 재현율(WAR)과 비가중 평균 재현율(UAR)을 사용했습니다.
- **주요 결과:**
  - **IEMOCAP 데이터셋:** 제안된 Speech Swin-Transformer는 WAR 75.22%, UAR 65.94%를 달성하여 최신 방법 중 가장 높은 성능을 보였습니다. 특히 '분노', '중립', '슬픔' 감정에서 높은 인식률(75% 이상)을 보였습니다. '기쁨' 감정의 인식률은 24%로 낮았고, 주로 '중립'으로 오분류되는 경향을 보였습니다.
  - **CASIA 데이터셋:** WAR 54.33%, UAR 54.33%를 달성하여 역시 최신 방법 중 최고 성능을 기록했습니다. (샘플 균형으로 WAR과 UAR이 동일함). '분노', '중립', '슬픔' 감정에서 68% 이상의 정확도를 보였습니다. '공포', '기쁨', '놀람' 감정은 40% 미만의 낮은 인식률을 보였습니다.
- **계층적 특징 맵 시각화:** 모델의 깊은 스테이지(3단계 및 4단계)에서 네트워크가 '슬픔'과 같은 낮은 각성 감정의 표현과 밀접하게 관련된 중-저 주파수 범위에 집중하는 것을 시각적으로 확인했습니다. 이는 제안된 방법이 계층적 음성 감정 표현을 효과적으로 얻음을 보여줍니다.

## 🧠 Insights & Discussion

- Speech Swin-Transformer는 계층적 Transformer 아키텍처를 기반으로 하여 다양한 수준의 감정 특징을 효과적으로 통합함으로써 뛰어난 SER 성능을 달성했습니다.
- 모델의 계층적 구조는 스펙트로그램의 일반적인 특징(얕은 레이어)에서부터 특정 주파수 범위 내의 의미론적으로 풍부한 감정 정보(깊은 레이어)에 이르기까지 특징을 학습할 수 있게 합니다.
- 감정별 인식 성능의 차이는 데이터셋 불균형(IEMOCAP의 '기쁨' 샘플 부족)과 감정의 각성(arousal) 및 원자가(valence) 유사성(예: '기쁨'/'중립'은 높은 원자가, '공포'/'슬픔'은 낮은 원자가, '기쁨'/'놀람'/'분노'는 높은 각성)과 관련이 있습니다. 이는 미묘한 감정을 구분하는 데 있어 존재하는 어려움을 시사합니다.
- **한계점 및 향후 연구:** 향후 연구는 주파수 도메인과 시간-주파수 도메인 모두에서 패치 경계 모델링에 초점을 맞출 것입니다.

## 📌 TL;DR

- **문제:** 음성 감정 인식(SER)에서 다중 스케일 감정 특징을 효과적으로 집계하고 패치 간의 의존성을 처리하는 것이 중요합니다.
- **방법:** 본 논문에서는 Shifted Windows를 활용한 계층적 Transformer인 Speech Swin-Transformer를 제안합니다. 이 모델은 Local Windows Transformer로 윈도우 내 감정 정보를, Shifted Windows Transformer로 윈도우 간 연결을 모델링하며, Patch Merging을 통해 프레임-세그먼트 레벨의 계층적 특징 학습 및 수용 필드 확장을 수행합니다.
- **핵심 발견:** IEMOCAP (WAR 75.22%, UAR 65.94%) 및 CASIA (WAR 54.33%, UAR 54.33%) 데이터셋에서 기존 최신(SOTA) 모델을 능가하는 성능을 달성하여, 계층적 감정 특징 포착의 효과를 입증했습니다.
