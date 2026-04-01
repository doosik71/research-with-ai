# Conformer: 음성 인식을 위한 컨볼루션 증강 트랜스포머

Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang

## 🧩 Problem to Solve

최근 자동 음성 인식(ASR) 분야에서 트랜스포머(Transformer) 모델은 전역적인(global) 내용 기반 상호작용 포착에 능하지만, 지역적인(local) 미세 특징 패턴 추출에는 제한적입니다. 반면 컨볼루션 신경망(CNN)은 지역적 특징을 효과적으로 활용하지만, 전역적 정보를 포착하기 위해서는 더 많은 계층이나 파라미터가 필요합니다. 이 연구는 음성 시퀀스의 지역적 및 전역적 의존성을 모두 효과적이고 파라미터 효율적인 방식으로 모델링하기 위해 컨볼루션 신경망과 트랜스포머를 어떻게 결합할 것인가 하는 문제를 다룹니다.

## ✨ Key Contributions

- **Conformer 아키텍처 제안**: 음성 인식을 위해 컨볼루션 모듈을 트랜스포머에 증강한 새로운 모델인 Conformer를 제안합니다.
- **최고 수준(SOTA) 성능 달성**: LibriSpeech 벤치마크에서 외부 언어 모델 없이 2.1%/4.3% (test/testother), 외부 언어 모델과 함께 1.9%/3.9%의 WER을 달성하여 기존 트랜스포머 및 CNN 기반 모델을 능가합니다.
- **파라미터 효율성 입증**: 10M 파라미터의 작은 모델로도 2.7%/6.3%의 경쟁력 있는 성능을 보여주며, 30M 파라미터 모델은 139M 파라미터의 이전 SOTA Transformer Transducer를 능가합니다.
- **종합적인 Ablation Study**: 어텐션 헤드 수, 컨볼루션 커널 크기, 활성화 함수, 피드포워드 계층 배치, 컨볼루션 모듈 추가 전략 등 각 구성 요소가 성능 향상에 기여하는 방식을 심층적으로 분석합니다.
- **새로운 블록 구조**: Macaron-Net에서 영감을 받아 두 개의 하프-스텝 잔차 연결 피드포워드 모듈 사이에 멀티-헤드 셀프-어텐션과 컨볼루션 모듈을 배치하는 독창적인 Conformer 블록 구조를 제시합니다.

## 📎 Related Works

- **RNN 기반 ASR**: 음성 시퀀스의 시간적 의존성 모델링에 효과적인 기존 ASR 시스템 [1, 2, 3, 4, 5].
- **트랜스포머(Transformer)**: 셀프-어텐션 기반 아키텍처로 긴 거리의 상호작용 포착 및 높은 훈련 효율성을 가짐 [6, 7, 19, 24]. Transformer-XL [20]의 상대적 위치 인코딩 기법 활용.
- **CNN 기반 ASR**: 지역적 수용 필드(receptive field)를 통해 점진적으로 지역적 컨텍스트를 포착하는 데 성공적 [8, 9, 10, 11, 12]. ContextNet [10]은 Squeeze-and-Excitation 모듈로 더 긴 컨텍스트를 포착하려 시도. QuartzNet [9].
- **컨볼루션과 셀프-어텐션 결합 연구**: 둘을 함께 사용하는 것이 개별적으로 사용하는 것보다 성능 향상에 기여함이 입증 [14, 15, 16, 17]. Wu et al. [17]은 멀티-브랜치 아키텍처를 제안.
- **Macaron-Net**: 트랜스포머 블록의 피드포워드 계층을 두 개의 하프-스텝 피드포워드 계층으로 대체하는 아이디어를 제시 [18].

## 🛠️ Methodology

Conformer는 Convolution Subsampling Layer와 여러 개의 Conformer Block으로 구성된 인코더를 사용합니다.

### 1. Conformer Block 구조 (Macaron-Net 스타일)

Conformer 블록은 두 개의 Feed-Forward 모듈이 Multi-Headed Self-Attention 모듈과 Convolution 모듈을 감싸는 형태로 구성됩니다. 수식적으로는 다음과 같습니다:

- $\tilde{x}_i = x_i + \frac{1}{2} \text{FFN}(x_i)$
- $x'_i = \tilde{x}_i + \text{MHSA}(\tilde{x}_i)$
- $x''_i = x'_i + \text{Conv}(x'_i)$
- $y_i = \text{Layernorm}(x''_i + \frac{1}{2} \text{FFN}(x''_i))$
  여기서 $\text{FFN}$은 Feed Forward 모듈, $\text{MHSA}$는 Multi-Head Self-Attention 모듈, $\text{Conv}$는 Convolution 모듈을 의미합니다.

### 2. Multi-Headed Self-Attention (MHSA) 모듈

- **상대적 위치 인코딩(Relative Positional Encoding)**: Transformer-XL [20]에서 제안된 상대적 사인파 위치 인코딩 스키마를 사용하여 다양한 입력 길이에 대한 일반화 능력을 향상합니다.
- **Pre-norm 잔차 유닛(Pre-norm Residual Units)**: 더 깊은 모델의 학습 및 정규화를 돕기 위해 드롭아웃과 함께 사용됩니다 [21, 22].

### 3. Convolution 모듈

- **Gating Mechanism**: 포인트와이즈 컨볼루션(Pointwise Convolution)과 GLU(Gated Linear Unit) 활성화로 시작합니다 [17, 23].
- **1-D Depthwise Convolution**: 게이팅 메커니즘 다음에 1차원 Depthwise Convolution 계층이 이어집니다.
- **Batchnorm & Swish**: 컨볼루션 직후 배치 정규화(Batchnorm)와 Swish 활성화 함수 [25]를 적용하여 깊은 모델 학습을 돕습니다.

### 4. Feed Forward (FFN) 모듈

- **두 개의 선형 변환**: 두 개의 선형 변환과 그 사이에 비선형 활성화 함수(Swish)를 포함합니다.
- **하프-스텝 잔차(Half-step Residuals)**: Macaron-Net [18]에서 영감을 받아 하프-스텝 잔차 가중치를 사용합니다.
- **Pre-norm 잔차 유닛**: MHSA 모듈과 마찬가지로 pre-norm 잔차 유닛과 드롭아웃을 사용합니다.

### 5. Encoder 및 Decoder

- **Encoder**: 입력은 먼저 Convolution Subsampling 계층을 거쳐 여러 Conformer 블록을 통과합니다.
- **Decoder**: 모든 모델에서 단일 LSTM 계층 디코더를 사용합니다.

### 6. 훈련 설정

- **데이터**: LibriSpeech [26] 데이터셋을 사용하고 80채널 필터뱅크 특징을 추출합니다. SpecAugment [27, 28]를 데이터 증강에 활용합니다.
- **정규화**: 각 잔차 유닛에 드롭아웃 [29]을 적용하고, 변동 노이즈(variational noise) [5, 30] 및 $L_2$ 정규화를 사용합니다.
- **최적화**: Adam optimizer [31]와 트랜스포머 학습률 스케줄 [6]을 사용합니다.
- **외부 LM**: LibriSpeech 960h 텍스트 코퍼스를 포함하여 훈련된 3계층 LSTM 언어 모델(LM)을 얕은 융합(shallow fusion)에 사용합니다.

## 📊 Results

- **LibriSpeech 벤치마크 성능**:
  - **Conformer(L) (118.8M 파라미터)**:
    - 언어 모델 없음: test-clean 2.1% / test-other 4.3% WER
    - 언어 모델 사용: test-clean 1.9% / test-other 3.9% WER (SOTA 달성)
  - **Conformer(M) (30.7M 파라미터)**: 언어 모델 사용 시 test-clean 2.0% / test-other 4.3% WER로, 139M 파라미터의 Transformer Transducer [7]를 능가합니다.
  - **Conformer(S) (10.3M 파라미터)**: ContextNet(S) [10] (10.8M 파라미터)와 비교하여 test-other에서 0.7% 더 좋은 성능을 보입니다.
- **Ablation Study 주요 결과**:
  - **컨볼루션 서브-블록**: Conformer 블록에서 가장 중요한 특징입니다. 컨볼루션 모듈을 제거하면 WER이 크게 증가합니다.
  - **Macaron-style FFN**: 단일 FFN보다 Macaron-style FFN 쌍을 사용하는 것이 더 효과적입니다.
  - **활성화 함수**: Swish 활성화 함수는 ReLU보다 Conformer 모델에서 더 빠른 수렴을 이끌어냅니다.
  - **모듈 배치**: 컨볼루션 모듈을 셀프-어텐션 모듈 다음에 배치하는 것이 가장 좋은 성능을 보입니다. 병렬 배치나 셀프-어텐션 이전에 배치하는 것은 성능 저하를 초래합니다.
  - **컨볼루션 종류**: Depthwise Convolution이 Lightweight Convolution보다 우수합니다.
  - **어텐션 헤드 수**: 8~16개의 어텐션 헤드를 사용할 때 가장 좋은 성능을 보입니다.
  - **컨볼루션 커널 크기**: 커널 크기 32에서 가장 좋은 성능을 달성합니다 (커널 크기 $\{3, 7, 17, 32, 65\}$ 실험 중).

## 🧠 Insights & Discussion

- Conformer는 트랜스포머의 전역적 상호작용 모델링 능력과 CNN의 지역적 특징 추출 능력을 효과적으로 결합함으로써 음성 인식에서 뛰어난 성능을 발휘합니다.
- Macaron-Net에서 영감을 받은 블록 구조, 특히 하프-스텝 잔차 연결을 사용하는 두 개의 피드포워드 계층이 어텐션 및 컨볼루션 모듈을 감싸는 "샌드위치" 구조가 모델의 성능 향상에 핵심적인 역할을 합니다.
- 컨볼루션 모듈의 설계(포인트와이즈 컨볼루션, GLU, 1-D Depthwise Convolution, Batchnorm, Swish)와 배치(셀프-어텐션 이후)는 음성 인식 태스크에 최적화되어 있습니다.
- 종합적인 ablation study는 각 구성 요소의 중요성을 밝혀내고 Conformer의 설계 선택이 성능 향상에 미치는 영향을 명확히 보여줍니다.
- Conformer는 적은 파라미터 수로도 이전 SOTA 모델들을 능가하는 파라미터 효율성을 입증하여, 실제 배포 환경에서의 활용 가능성을 높입니다. 이는 복잡한 음성 시퀀스에서 전역적 및 지역적 의존성을 모두 포착하는 것이 파라미터 효율성에 중요하다는 가설을 뒷받침합니다.

## 📌 TL;DR

Conformer는 음성 시퀀스의 지역적 및 전역적 의존성을 효과적으로 모델링하기 위해 트랜스포머의 전역적 어텐션과 CNN의 지역적 컨볼루션을 결합한 새로운 ASR 아키텍처입니다. Macaron-Net에서 영감을 받은 "샌드위치" 구조의 Conformer 블록을 제안하며, LibriSpeech 벤치마크에서 SOTA WER을 달성함과 동시에 높은 파라미터 효율성을 보여줍니다. 핵심은 셀프-어텐션 다음에 컨볼루션 모듈을 배치하고, 하프-스텝 잔차 연결을 가진 두 개의 피드포워드 모듈이 이들을 감싸는 구조입니다.
