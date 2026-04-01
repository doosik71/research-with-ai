# Speech Recognition by Machine: A Review

M.A.Anusuya, S.K.Katti

## 🧩 Problem to Solve

이 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템의 주요 연구 과제인 **다양한 화자, 환경, 맥락 변화에 따른 정확도와 견고성 문제를 해결**하고, 지난 60년간 ASR 연구의 발전 과정을 기술적 관점에서 조망하며 핵심적인 발전 사항과 남은 도전 과제들을 제시합니다. 궁극적으로 ASR 기술의 광범위한 배포를 가로막는 실제적 한계를 극복하는 것을 목표로 합니다.

## ✨ Key Contributions

- **ASR 기술의 포괄적인 역사적 개관:** 지난 60년간의 ASR 연구 및 개발의 주요 테마와 발전을 연대순으로 정리하여 기술적 관점을 제공합니다.
- **ASR 시스템 설계 요소 분석:** 음성 클래스 정의, 음성 표현, 특징 추출 기법, 음성 분류기, 데이터베이스 및 성능 평가 등 ASR 시스템 설계에 필요한 핵심 요소들을 논의합니다.
- **주요 ASR 접근 방식 비교:** 음향-음성학적 접근, 패턴 인식 접근(템플릿 기반, 확률적 모델-HMM), 인공 지능 접근(지식 기반, 연결주의-ANN, SVM)을 상세히 설명하고 그 장단점을 비교합니다.
- **핵심 기술 및 발전 동향 요약:** Dynamic Time Warping (DTW), Vector Quantization (VQ), Hidden Markov Models (HMM), Neural Networks, Support Vector Machine (SVM) 등 핵심 기술의 원리와 적용을 설명합니다.
- **성능 평가 지표 정의:** 단어 오류율 (WER) 및 단어 인식률 (WRR)과 같은 ASR 시스템의 성능 측정 방법을 제시합니다.
- **인간과 기계의 음성 인식 격차 논의:** 기계 음성 인식의 한계와 인간 음성 처리 간의 격차를 줄이기 위한 미래 연구 방향을 제시합니다.
- **주요 음성 데이터베이스 소개:** Resource Management (RM), TIMIT, SWITCHBOARD, ATIS 등 ASR 연구에 사용되는 다양한 음성 데이터베이스의 특징을 설명합니다.

## 📎 Related Works

이 논문은 ASR 분야의 광범위한 연구를 다루며 다음을 포함한 수많은 선행 연구들을 언급합니다:

- **초기 ASR 연구:** Bell Labs (1920년대부터), RCA Laboratories (1950년대), University College (Fry & Denes, 1959), MIT Lincoln Laboratories (Forgie & Forgie, 1959).
- **템플릿 매칭 및 DTW:** Vintsyuk (1960년대, Dynamic Programming), Sakoe & Chiba (1970년대, DTW).
- **연속 음성 인식 초기 연구:** Reddy (1960년대, Carnegie Mellon University).
- **확률적 모델링 (HMM):** IBM (F. Jelinek, L.R. Bahl, R.L. Mercer 등, 1970년대-1980년대), L.R. Rabiner (HMM 튜토리얼, 1989), J. Ferguson (HMM, 1980).
- **인공 신경망 (ANN):** R.P. Lippmann (1987), A. Weibel 등 (1989).
- **DARPA 프로그램:** Speech Understanding Project (1970년대, CMU Heresay, Harpy), SPHINX (CMU), BYBLOS (BBN), ATIS (Air Travel Information Service), EARS (Effective Affordable Reusable Speech-to-Text).
- **판별 학습:** Minimum Classification Error (MCE) 및 Generalized Probabilistic Descent (GPD), Maximum Mutual Information (MMI).
- **강건한 음성 인식:** Maximum Likelihood Linear Regression (MLLR), Parallel Model Combination (PMC), Signal Bias Removal (SBR).
- **현대적 기술:** Variational Bayesian (VB) 추정, Conditional Random Fields (CRF), Support Vector Machines (SVM).

## 🛠️ Methodology

이 논문은 ASR 시스템을 구축하는 데 사용되는 다양한 접근 방식과 기술을 검토합니다.

1. **ASR의 기본 모델**:

   - **확률적 모델**을 기반으로 음향 관측 시퀀스 $A$로부터 가장 가능성 있는 단어 시퀀스 $W$를 디코딩하는 것을 목표로 합니다:
     $$\hat{W} = \arg \max_{W} P(A|W)P(W)$$
     여기서 $P(A|W)$는 **음향 모델(Acoustic Model)**, $P(W)$는 **언어 모델(Language Model)**입니다.
   - ASR 시스템은 일반적으로 프론트엔드(음향 분석), 모델 유닛, 언어 모델 유닛, 검색 유닛으로 구성됩니다.

2. **음성 인식 유형**:

   - **단어 분리(Isolated Words)**: 각 발화 사이에 일시 정지가 필요한 단일 단어 또는 구 인식.
   - **연결 단어(Connected Words)**: 최소한의 일시 정지를 허용하며 여러 단어를 연결하여 인식.
   - **연속 음성(Continuous Speech)**: 자연스럽게 말하는 문장을 인식 (컴퓨터 받아쓰기).
   - **자발적 음성(Spontaneous Speech)**: "음", "아" 같은 비유창성(disfluency)이나 발화 간의 경계 모호성을 처리.

3. **주요 접근 방식**:

   - **음향-음성학적 접근(Acoustic-Phonetic Approach)**: 음성 신호를 음소(phonemes) 단위로 분할하고 음향적 특성을 기반으로 라벨링한 후, 언어적 제약을 사용하여 단어 시퀀스를 결정.
   - **패턴 인식 접근(Pattern Recognition Approach)**:
     - **템플릿 기반(Template Based)**: 미리 저장된 참조 패턴(템플릿)과 들어온 음성 패턴을 직접 비교. 시간적 변동성을 보정하기 위해 **Dynamic Time Warping (DTW)** 사용.
     - **확률적(Stochastic)**: 음성 신호의 불확실성을 확률 모델로 다룸. **Hidden Markov Model (HMM)**이 가장 널리 사용되며, 은닉 상태의 시계열적 변화와 관측 값의 스펙트럼 변화를 모델링.
     - **Vector Quantization (VQ)**: 데이터 축소 기법으로, 음성 특징 벡터를 코드북의 대표 벡터로 양자화하여 효율적인 모델 표현 및 비교를 가능하게 함.
   - **인공 지능 접근(Artificial Intelligence Approach)**: 음향-음성학적 및 패턴 인식 아이디어를 결합. 언어학적, 음성학적, 스펙트로그램 정보와 전문가 규칙을 활용.
     - **연결주의 접근(Connectionist Approaches / ANN)**: 인공 신경망을 사용하여 음성 패턴 간의 복잡한 비선형 관계를 학습. 병렬 분산 처리와 대규모 학습 데이터를 통해 분류기 성능을 최적화.
     - **Support Vector Machine (SVM)**: 고정 길이 데이터 벡터에 대한 판별 학습 분류기로, 선형 및 비선형 분리 초평면을 사용하여 최대 마진을 찾음으로써 분류 성능을 향상.

4. **특징 추출(Feature Extraction)**:

   - 음성 신호의 컴팩트한 표현을 위해 수행되며, 일반적으로 세 단계를 거침: 음성 분석 (스펙트럼 포락선 특징), 정적 및 동적 특징으로 확장, 더 견고한 벡터로 변환.
   - 주요 방법: Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Independent Component Analysis (ICA), Linear Predictive Coding (LPC), Cepstral Analysis, Mel-frequency Cepstral Coefficients (MFCCs), Wavelet 변환, 스펙트럼 차감법(Spectral Subtraction), 켑스트럼 평균 차감법(Cepstral Mean Subtraction).

5. **분류기(Classifiers)**:
   - **유사성 기반**: 템플릿 매칭, 최소 거리 분류기, 최근접 평균 분류기, VQ, 학습 벡터 양자화 (LVQ), 1-NN (1-Nearest Neighbor).
   - **확률 기반**: 베이즈 의사결정 규칙, 최대 사후 확률 (MAP), 최대 우도(Maximum Likelihood).

## 📊 Results

이 논문은 ASR 시스템의 성능을 **단어 오류율(Word Error Rate, WER)** 및 **단어 인식률(Word Recognition Rate, WRR)**로 평가하며, 지난 60년간의 기술 발전과 그에 따른 성능 향상을 연대순으로 제시합니다.

- **성능 측정**:

  - **단어 오류율 (WER)**: $$WER = \frac{S+D+I}{N}$$
    여기서 $S$는 치환(Substitutions) 수, $D$는 삭제(Deletions) 수, $I$는 삽입(Insertions) 수, $N$은 참조 단어 시퀀스의 총 단어 수입니다.
  - **단어 인식률 (WRR)**: $$WRR = \frac{H}{N}$$
    여기서 $H$는 올바르게 인식된 단어 수($N-(S+D)$)입니다.

- **기술 발전 요약 (주요 마일스톤)**:

  - **1920-1960년대**: 초기 음성 인식 기계(Radio Rex), Bell Labs의 단일 화자 숫자 인식 시스템, 스펙트럼 공명 및 논리 회로 활용.
  - **1960-1970년대**: DTW(Dynamic Time Warping) 개념 도입(Vintsyuk, Sakoe & Chiba), 연속 음성 인식의 기초 마련(Reddy).
  - **1970-1980년대**: 고립 단어(Isolated Word) 인식 기술 상용화, 선형 예측 코딩(LPC) 기반 거리 측정, IBM의 대규모 어휘 음성 인식 연구, AT&T Bell Labs의 화자 독립적 인식 연구, DARPA의 Speech Understanding Project (Heresay, Harpy 시스템).
  - **1980-1990년대**: 연결 단어(Connected Word) 인식 연구 활발, HMM(Hidden Markov Model)이 지배적인 통계적 모델링 방법으로 부상, 인공 신경망(Neural Networks) 재도입, DARPA의 대규모 어휘 연속 음성 인식 프로젝트 (SPHINX, BYBLOS).
  - **1990-2000년대**: 패턴 인식이 오차 최소화 최적화 문제로 전환 (판별 학습: MCE, MMI), 강건한 음성 인식 기법(MLLR, PMC, SMAP), 음성 데이터베이스 확장(ATIS, Switchboard), HMM의 다양한 개선(가중 HMM, 부분 공간 투영).
  - **2000-2009년**: Variational Bayesian 추정, 능동 학습, LVCSR(Large Vocabulary Continuous Speech Recognition) 성능 향상, 자발적 음성 인식 연구(CSJ 말뭉치), 조건부 랜덤 필드(CRF), 다중 모달(Multimodal) 음성 인식, 잡음 환경에서의 강건성 향상(SLDMs, 특징 강화), 데이터 기반 접근법 (복합어 설계, EMD).

- **전반적인 성과**: 지난 수십 년간 ASR은 템플릿 매칭에서 통계적 모델링, 판별 학습, 강건한 특징 추출, 대규모 데이터베이스 활용 등을 통해 상당한 발전을 이루었으며, WER을 크게 감소시켰습니다. 그러나 여전히 인간의 음성 인식 능력에 비해 취약한 부분이 많습니다.

## 🧠 Insights & Discussion

- **성장과 한계**: 지난 60년간 ASR 기술은 신호 처리 알고리즘, 아키텍처, 하드웨어의 발전과 함께 고립 단어 인식에서 연속 및 자발적 음성 인식으로, 소규모 어휘에서 대규모 어휘로, 깨끗한 음성에서 잡음/전화 음성 인식으로 진화하며 놀라운 발전을 이루었습니다. 그러나 여전히 인간이 보이는 인식 정확도 및 견고성에는 크게 미치지 못하는 '성능 격차'가 존재합니다. 특히 환경 조건, 화자 변동성, 맥락의 다양성 등 실제 환경에서의 견고성은 중요한 도전 과제로 남아 있습니다.
- **미래 방향**: ASR 시스템은 자연스러운 인간-기계 대화를 위해 필요한 지식을 효율적으로 표현, 저장, 검색하는 방법을 발전시켜야 합니다. 음향-음성학, 음성 지각, 언어학, 음향심리학 분야의 연구를 통해 인간 음성 처리 메커니즘에 대한 더 깊은 이해를 바탕으로 기술적 돌파구를 마련하는 것이 중요합니다.
- **근본적인 질문**: ASR 분야는 여전히 음성 통신의 본질, 인간-기계 통신에서의 관련성, 음성 과학과 기술의 통합, 음성 단위 및 표현의 최적화 등 20가지 핵심 질문에 대한 명확한 답을 찾지 못하고 있습니다. 이러한 질문에 대한 탐구가 ASR의 다음 단계 발전을 이끌 것입니다.
- **지속적인 중요성**: 음성 인식은 기계 지능 분야에서 가장 흥미로운 영역 중 하나이며, 인간-기계 상호작용의 핵심 요소로서 사회에 큰 기술적 영향을 미치고 있습니다. 앞으로도 이 분야는 계속해서 번성할 것으로 기대됩니다.

## 📌 TL;DR

- **Problem:** 자동 음성 인식(ASR)은 다양한 화자, 환경, 맥락에서 인간 수준의 정확도와 견고성을 달성하는 데 큰 어려움을 겪고 있습니다.
- **Method:** 이 논문은 지난 60년간 ASR 연구의 역사적 발자취를 포괄적으로 검토하고, 음향-음성학적 접근, 패턴 인식(템플릿 기반 및 HMM), 인공 지능(신경망, SVM) 등 주요 접근 방식과 특징 추출, 분류기 기술의 발전을 상세히 분석합니다.
- **Findings:** ASR은 초기 템플릿 매칭 시스템에서 HMM 및 판별 학습 기반 통계 모델로 발전하며 대규모 어휘, 연속/자발적 음성 인식에서 상당한 성능 향상(WER 감소)을 이루었습니다. 그러나 인간과 기계의 음성 인식 성능 격차는 여전히 존재하며, 특히 복잡한 환경과 자연스러운 대화 처리 능력 향상이 미래 연구의 핵심 과제로 남아 있습니다.
