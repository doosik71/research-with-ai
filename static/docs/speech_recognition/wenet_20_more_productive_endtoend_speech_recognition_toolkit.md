# WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit

Binbin Zhang, Di Wu, Zhendong Peng, Xingchen Song, Zhuoyuan Yao, Hang Lv, Lei Xie, Chao Yang, Fuping Pan, Jianwei Niu

## 🧩 Problem to Solve

이 논문은 생산 환경(production environment)에 최적화된 종단간(End-to-End, E2E) 음성 인식(ASR) 툴킷인 WeNet의 성능을 향상시키고 다양한 생산 요구사항을 충족하는 것을 목표로 합니다. 기존 WeNet은 스트리밍 및 비스트리밍 디코딩을 단일 모델로 처리하는 U2(Unified Two-pass) 프레임워크와 내장 런타임을 제공했지만, ASR 성능 향상 및 실질적인 생산 시나리오에서의 활용성 개선이 필요했습니다. 특히, 대규모 데이터셋 처리, 강력한 언어 모델 통합, 사용자 맞춤형 문맥 편향(contextual biasing) 등의 기능이 부족했습니다.

## ✨ Key Contributions

WeNet 2.0의 주요 개선사항 및 기여는 다음과 같습니다.

- **U2++ 프레임워크:** 양방향 어텐션 디코더를 포함하는 U2++ 프레임워크를 제안하여 공유 인코더의 표현 능력을 향상시키고 재채점(rescoring) 단계에서 성능을 개선했습니다. 이는 미래 문맥 정보까지 활용합니다.
- **생산 지향 언어 모델 솔루션:** n-gram 기반 언어 모델과 WFST(Weighted Finite State Transducer) 기반 디코더를 WeNet 2.0에 통합하여 풍부한 텍스트 데이터를 생산 시나리오에 효과적으로 활용할 수 있도록 했습니다.
- **통합 문맥 편향 프레임워크:** 사용자별 문맥 정보(예: 연락처 목록)를 활용하여 ASR 정확도를 높이고 빠른 적응 능력을 제공하는 통합 문맥 편향 프레임워크를 설계했습니다. 이는 LM 사용 여부에 관계없이 작동합니다.
- **통합 I/O (UIO) 시스템:** 대규모 데이터셋의 효율적인 모델 훈련을 지원하기 위해 통합 I/O 시스템을 설계했습니다. 이는 다양한 저장 매체와 데이터 규모에 대해 단일 인터페이스를 제공합니다.

## 📎 Related Works

- **기존 E2E ASR 모델:** Connectionist Temporal Classification (CTC) [2], Recurrent Neural Network Transducer (RNN-T) [3, 4, 5], Attention Based Encoder-Decoder (AED) [6, 7, 8, 9].
- **E2E ASR 툴킷:** ESPnet [16] 및 SpeechBrain [17] (연구 지향적).
- **WeNet 1.0:** 생산 지향적 E2E ASR 툴킷으로, Transformer [18] 및 Conformer [19] 기반 E2E 모델의 생산 문제를 다루고, CTC/AED 결합 구조와 U2 프레임워크를 채택.
- **기타 기술:** Kaldi [21] (디코딩 알고리즘 및 코드 재사용), blank frame skipping [22] (디코딩 속도 향상), TFRecord (Tensorflow) [26] 및 AIStore [27] (대규모 데이터 I/O 설계 영감).
- **문맥 편향 기술:** [23, 24, 25] 등에서 전통 및 E2E 시스템에 대한 연구.

## 🛠️ Methodology

### U2++ 프레임워크

U2++는 스트리밍 및 비스트리밍 모드를 통합하는 양방향 어텐션 디코더를 갖춘 통합 2-패스 CTC/AED 프레임워크입니다.

- **구성:**
  1. **Shared Encoder:** 음향 특징을 모델링하며 제한된 오른쪽 문맥만 고려하여 지연 시간을 균형 있게 유지합니다 (Transformer 또는 Conformer 레이어로 구성).
  2. **CTC Decoder:** 음향 특징과 토큰 단위 간의 프레임 레벨 정렬 정보를 모델링합니다 (선형 레이어로 구성).
  3. **Left-to-Right Attention Decoder (L2R):** 과거 문맥 정보를 나타내기 위해 토큰 시퀀스를 좌에서 우로 모델링합니다.
  4. **Right-to-Right Attention Decoder (R2L):** 미래 문맥 정보를 나타내기 위해 역순 토큰 시퀀스를 우에서 좌로 모델링합니다.
- **훈련:** 결합된 CTC 및 AED 손실을 사용하여 훈련됩니다:
  $$L_{combined}(x,y) = \lambda L_{CTC}(x,y) + (1-\lambda)(L_{AED}(x,y))$$
  여기서 AED 손실 $L_{AED}(x,y)$는 L2R 및 R2L 디코더의 기여도를 조절하는 하이퍼파라미터 $\alpha$를 포함합니다:
  $$L_{AED}(x,y) = (1-\alpha)L_{L2R}(x,y) + \alpha(L_{R2L}(x,y))$$
  스트리밍 및 비스트리밍 모드를 통합하기 위해 동적 청크 마스킹(dynamic chunk masking) 전략이 사용됩니다.
- **디코딩:** 첫 번째 패스에서는 CTC 디코더가 스트리밍 모드로 N-best 결과를 생성하고, 두 번째 패스에서는 L2R 및 R2L 어텐션 디코더가 이 결과를 재채점하여 최종 결과를 얻습니다.

### 생산 지향 언어 모델 솔루션

풍부한 텍스트 데이터 활용을 위해 n-gram LM을 통합합니다.

- **LM 미사용 시:** CTC prefix beam search를 사용하여 N-best 후보를 얻습니다.
- **LM 사용 시:** WeNet 2.0은 n-gram LM (G), 어휘집 (L), 종단간 모델링 CTC 토폴로지 (T)를 WFST 기반 디코딩 그래프 (TLG)로 컴파일합니다:
  $$TLG=T \circ \text{min}(\text{det}(L \circ G))$$
  그 후 CTC WFST beam search를 사용하여 N-best 후보를 얻고, 마지막으로 Attention Rescoring 모듈이 최적의 후보를 찾습니다. Kaldi의 디코딩 알고리즘과 코드를 재사용하며, blank frame skipping 기술로 속도를 향상시킵니다.

### 문맥 편향 프레임워크

사용자별 문맥 정보를 활용하여 ASR 정확도를 향상시킵니다.

- **구현:** 미리 알려진 편향 구문(biasing phrases) 집합이 있을 때, 동적으로 문맥 WFST 그래프를 구축합니다.
  1. 각 편향 단위(LM-free의 E2E 모델링 단위 또는 with-LM의 어휘 단어)는 부스트된 점수와 함께 해당 아크에 순차적으로 배치됩니다.
  2. 각 중간 상태에는 부분 일치 시 부스트 점수를 제거하는 특수 실패 아크(negative accumulated boosted score)가 추가됩니다.
- **디코딩 중:** 빔 서치 결과가 문맥 WFST 그래프를 통해 편향 단위와 일치하면 즉시 부스트 점수가 추가됩니다:
  $$y^* = \text{arg max}_y \log P(y|x) + \lambda \log P_C(y)$$
  여기서 $P_C(y)$는 편향 점수이고 $\lambda$는 문맥 LM의 영향을 제어하는 하이퍼파라미터입니다.

### 통합 I/O (UIO) 시스템

대규모 데이터셋 훈련 시 발생하는 메모리 부족 및 느린 훈련 속도 문제를 해결합니다.

- **작은 데이터셋:** 샘플 수준의 무작위 접근(random access) 기능을 유지하며 직접 로드합니다.
- **큰 데이터셋:** 각 샘플 집합(예: 1000개 샘플)과 메타데이터를 GNU tar를 사용하여 더 큰 샤드(shard)로 묶습니다.
  - 샤드 파일은 메모리를 절약하여 OOM 문제를 해결합니다.
  - 훈련 중 메모리 내에서 온더플라이(on-the-fly) 압축 해제가 수행되며, 동일한 압축 샤드 내의 데이터는 순차적으로 읽혀 데이터 접근 속도를 향상시킵니다.
  - 동시에 다른 샤드들은 무작위로 읽혀 데이터의 전역 무작위성을 보장합니다.
- **저장 매체:** 로컬 디스크 및 분산 스토리지(S3, OSS, HDFS 등)의 샤드 로딩을 지원합니다.
- 데이터 처리를 위한 체인 작업(chain operations)이 설계되어 확장 가능하고 디버깅이 용이합니다.

## 📊 Results

- **U2++:** 다양한 ASR 코퍼스(AISHELL-1, AISHELL-2, LibriSpeech, WenetSpeech, GigaSpeech)에서 U2 대비 최대 10%의 상대적 에러율 감소를 달성했습니다. 이는 일관되게 우수한 성능을 보여줍니다.
- **N-gram Language Model:** AISHELL-1에서 5%의 이득, LibriSpeech `test_other`에서 8.42%의 우수한 개선을 보였습니다. 전반적으로 세 코퍼스에서 일관된 이득을 보여 E2E 시스템에 효과적으로 통합됨을 입증했습니다.
- **Contextual Biasing:** `test_p` (긍정 테스트 세트)에서 에러율을 크게 감소시켰으며, 부스트 점수가 클수록 개선 폭이 컸습니다. 적절한 부스트 점수를 설정하면 `test_p`의 성능을 향상시키면서 `test_n` (부정 테스트 세트)의 성능 저하를 방지할 수 있음을 보여주었습니다. LM 사용 여부에 관계없이 효과적이었습니다.
- **UIO:** AISHELL-1에서 raw 모드와 유사한 정확도를 유지하면서 샤드 모드에서 훈련 속도를 약 9.27% 향상시켰습니다. WenetSpeech와 같은 대규모 데이터셋에서는 raw 모드로는 훈련이 매우 느려 비교가 어렵지만, 샤드 모드로 훈련한 결과가 ESPnet과 유사한 성능을 보여 UIO의 효율성을 입증했습니다.

## 🧠 Insights & Discussion

WeNet 2.0은 생산 지향적 E2E ASR 툴킷으로서 상당한 성능 향상과 함께 여러 중요한 기능을 도입하여 실제 배포 환경의 요구사항을 충족합니다. U2++는 양방향 문맥 정보를 활용하여 모델의 표현 능력을 극대화했고, n-gram LM 통합은 풍부한 텍스트 데이터의 가치를 ASR 시스템으로 가져왔습니다. 문맥 편향 기능은 사용자 맞춤형 적응성을 제공하여 ASR 정확도에 결정적인 영향을 미치며, UIO 시스템은 대규모 데이터셋 훈련의 효율성을 혁신적으로 개선했습니다.

이러한 개선 사항들은 WeNet이 단순한 연구 툴킷을 넘어, 엔터프라이즈급 ASR 시스템 구축에 필수적인 요소들을 제공함으로써 "더욱 생산적인" 툴킷이 되었음을 시사합니다.

**제한사항 및 향후 과제:** 논문에서는 명시적인 제한사항을 언급하지는 않았으나, WeNet 3.0에서는 비지도 자기 학습, 온디바이스(on-device) 모델 탐색 및 최적화 등 생산 수준 ASR을 위한 추가적인 특성에 중점을 둘 것이라고 언급했습니다. 이는 현재 WeNet 2.0이 다루지 못하는 영역이나 미래에 중요해질 부분들을 암시합니다.

## 📌 TL;DR

WeNet 2.0은 생산 지향적 종단간(E2E) 음성 인식 툴킷으로, 기존 WeNet의 한계를 극복하기 위해 네 가지 주요 업데이트를 제시합니다. 첫째, 양방향 어텐션 디코더를 활용하는 **U2++ 프레임워크**로 ASR 성능을 최대 10% 향상시켰습니다. 둘째, **n-gram 언어 모델과 WFST 기반 디코더**를 통합하여 풍부한 텍스트 데이터 활용을 가능하게 했습니다. 셋째, 사용자별 문맥 정보를 활용하는 **통합 문맥 편향 프레임워크**를 통해 ASR 정확도를 크게 높였습니다. 넷째, 대규모 데이터셋 훈련을 위한 효율적인 **통합 I/O (UIO) 시스템**을 설계하여 메모리 및 속도 문제를 해결했습니다. 이 모든 개선을 통해 WeNet 2.0은 다양한 코퍼스에서 상당한 성능 향상을 달성하며 더욱 생산적인 E2E ASR 툴킷으로 거듭났습니다.
