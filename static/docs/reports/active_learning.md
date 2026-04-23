# Active Learning

## 서론

### 1. 연구 배경

본 보고서는 Active Learning 분야에서 수행된 주요 연구들을 체계적으로 분석 및 정리한 내용을 담고 있다. 총 18 편 논문 (8 편 핵심 논문, 4 편 불확실성 추정 관련, 2 편 통합 프레임워크 관련, 2 편 survey, 2 편 streaming/human-in-the-loop 관련) 을 대상으로 공통 주제, 문제 정의, 적용 대상, 접근 방식, 시스템 구성 관점에서 검토하여 연구 분류 체계를 수립했다. 논문들은 문제 정의 관점 (uncertainty estimation, batch selection, concept drift 등), 접근 방식 관점 (Bayesian, geometric, RL, evidential 등), 시스템 구성 관점, 적용 대상 관점 (이미지, 회귀, 스트림, crowd-sourced 등) 으로 범주화되어 있다.

### 2. 문제의식 및 분석 필요성

Active Learning 분야의 논문들은 공통된 문제 설정 (제한된 라벨링 예산, unlabeled pool 에서 informative subset 선택) 에서 출발하지만, 각 계열마다 고유한 설계 철학과 최적화 대상 (uncertainty, diversity, geometric coverage, evidence, sequential policy, domain adaptation) 을 가진다. 다양한 방법론이 각기 다른 접근 방식과 설계 철학을 공유하며, 체계적인 분류와 비교 분석이 필요한 상황이다. 또한 평가 환경 (실험실 데이터셋, 시뮬레이션, crowd-sourcing, streaming), 평가 지표 (accuracy, RMSE, F1 등), 데이터셋 의존성 등 비교 조건에 불일치가 존재하기 때문에, 연구 결과들을 올바르게 이해하고 비교할 수 있는 체계적인 프레임워크가 요구된다.

### 3. 보고서의 분석 관점

본 보고서는 세 가지 주요 축으로 분석을 진행한다. 첫째, 연구체계 분류는 논문들을 9 개 주요 범주 (불확실성 추정, 배치 선택, 온라인/스트림 기반 선택, 고유한 동기를 통한 탐색, 강화학습 기반 선택, 인간 - 기계 협업, Crowd-Sourcing 최적화, Survey 및 벤치마크) 로 체계화한다. 둘째, 방법론 분석은 6 가지 계열 (Kernel 기반 프레임워크, Distribution Matching, Geometric/Core-Set, Evidential/Bayesian 계측, Sequential/Reinforcement, Domain-Specific) 으로 그룹화하며, 핵심 설계 패턴과 트레이드오프를 분석한다. 셋째, 실험결과 분석은 평가 지표와 성능 패턴, 데이터셋 의존성, 검증 방법론 등을 종합하여 결과를 해석한다.

### 4. 보고서 구성

- **1 장 연구체계 분류**: 논문들을 문제 정의, 접근 방식, 시스템 구성, 적용 대상 관점에서 9 개 주요 범주로 체계화하고, 불확실성 추정 (Bayesian, uncertainty sampling, improvements), 배치 선택 (framework, representative subset, aggressive sampling), 온라인/스트림 기반 선택, 고유한 동기, 강화학습 기반 선택, 인간 - 기계 협업, crowd-sourcing 최적화, survey/벤치마크 등 주요 범주를 정리한다.

- **2 장 방법론 분석**: 공통 문제 설정 및 접근 구조를 설명하고, 6 가지 방법론 계열 (Kernel 기반 프레임워크, Distribution Matching, Geometric/Core-Set, Evidential/Bayesian 계측, Sequential/Reinforcement, Domain-Specific) 의 특징과 설계 패턴을 분석한다. Acquisition function, Batch selection 최적화, 모델 설계 패턴과 계열 간 차이점, 트레이드오프를 비교하며, 방법론 진화 흐름과 선택 가이드를 제시한다.

- **3 장 실험결과 분석**: 평가 구조, 주요 데이터셋 유형, 평가 환경, 공통 평가 지표를 설명하고, 주요 실험 결과를 정리한다. 성능 개선 패턴, 데이터셋 의존성, 논문 간 상충되는 결과를 분석하며, 실험 설계의 한계와 비교 주의점을 논의한다. 추가 실험 및 검증 방법 패턴을 소개하고, 저자들의 결과 해석 경향과 실제 결과의 구분점을 제시한다.

## 1 장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 분류 체계는 논문들의 공통 주제, 문제 정의, 적용 대상, 접근 방식, 시스템 구성 관점에서 검토하여 수립되었습니다. 각 논문은 다음과 같은 기준으로 범주화되었습니다:

- **문제 정의 관점**: 연구 대상의 핵심 문제 (uncertainty estimation, batch selection, concept drift 등) 를 기준으로 분류
- **접근 방식 관점**: 사용된 핵심 방법론 (Bayesian, geometric, RL, evidential 등) 에 따라 분류
- **시스템 구성 관점**: active learning pipeline 의 구성 요소 (query strategy, model architecture 등) 를 반영하여 분류
- **적용 대상 관점**: 적용 대상 (이미지, 회귀, 스트림, crowd-sourced 등) 을 기준으로 분류

각 논문은 가장 대표적인 1 개 범주에 배치되었으며, 중복 범주는 명시되지 않습니다.

### 2. 연구 분류 체계

#### 2.1 불확실성 추정 (Uncertainty Estimation)

이 범주에는 모델의 불확실성을 직접 추정하거나 개선하는 방법론을 연구한 논문들이 포함됩니다.

##### 2.1.1 Bayesian Uncertainty

| 분류                                            | 논문명                                                                                              | 분류 근거                                                                                            |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 2.1.1 Bayesian Uncertainty > MC-Dropout         | What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? (2017)                 | MC-dropout 기반 epistemic uncertainty 근사 및 aleatoric/epistemic 불확실성 분리 모델링               |
| 2.1.1 Bayesian Uncertainty > Bayesian CNN       | Deep Bayesian Active Learning with Image Data (2017)                                                | MC dropout 기반 uncertainty estimation 과 acquisition function(BALD 등) 결합                         |
| 2.1.1 Bayesian Uncertainty > Evidential         | DEAL: Deep Evidential Active Learning for Image Classification (2020)                               | Dirichlet density 파라미터 출력하는 evidential formulation 통한 single forward pass uncertainty 추정 |
| 2.1.1 Bayesian Uncertainty > On-Device Bayesian | ActiveHARNet: Towards On-Device Deep Bayesian Active Learning for Human Activity Recognition (2019) | on-device deep Bayesian active learning 프레임워크로 incremental adaptation 방식                     |
| 2.1.1 Bayesian Uncertainty > Ensemble           | A Survey on Deep Active Learning: Recent Advances and New Frontiers (2024)                          | ensemble 기반 uncertainty sampling을 DAL taxonomy 의 한 축으로 포함                                  |

##### 2.1.2 Uncertainty Sampling

| 분류                                      | 논문명                                             | 분류 근거                                                                  |
| ----------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------- |
| 2.1.2 Uncertainty Sampling > Thresholding | Active Learning for Data Streams: A Survey (2023)  | stationary/classification 설정에서 uncertainty thresholding 이 대표적 방법 |
| 2.1.2 Uncertainty Sampling > MinExpError  | Active Learning for Crowd-Sourced Databases (2014) | bootstrap 기반의 Uncertainty 와 MinExpError active learning ranker 제안    |

##### 2.1.3 Uncertainty Improvements

| 분류                                             | 논문명                                                                              | 분류 근거                                         |
| ------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| 2.1.3 Uncertainty Improvements > Softmax         | DEAL: Deep Evidential Active Learning for Image Classification (2020)               | softmax 대신 Dirichlet density 파라미터 직접 출력 |
| 2.1.3 Uncertainty Improvements > Loss Modulation | What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? (2017) | loss attenuation 으로 loss formulation 수정       |

#### 2.2 배치 선택 (Batch Selection)

이 범주에는 batch 단위의 샘플 선택 문제를 연구한 논문들이 포함됩니다.

##### 2.2.1 Batch Selection Framework

| 분류                                                    | 논문명                                                                            | 분류 근거                                                                                         |
| ------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| 2.2.1 Batch Selection Framework > Kernel Framework      | A Framework and Benchmark for Deep Batch Active Learning for Regression (2022)    | 커널·변환·선택법 조합으로 BMDAL 을 모듈화한 통합 프레임워크                                       |
| 2.2.1 Batch Selection Framework > Distribution Matching | Deep Active Learning: Unified and Principled Method for Query and Training (2019) | distribution matching 문제로 redefining, uncertainty/diversity 통합한 batch query 정책            |
| 2.2.1 Batch Selection Framework > Gradient Embedding    | Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (2019)     | gradient embedding 단일 표현으로 uncertainty/diversity 통합, 별도 hyperparameter tuning 필요 없음 |

##### 2.2.2 Representative Subset Selection

| 분류                                                         | 논문명                                                                 | 분류 근거                                                                                                          |
| ------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 2.2.2 Representative Subset Selection > Core-Set             | Active Learning for Convolutional Networks: A Core-Set Approach (2018) | batch acquisition 의 본질을 "uncertainty minimization"이 아닌 "representative subset selection"으로 재정의         |
| 2.2.2 Representative Subset Selection > Coordinated Matching | Batch Active Learning via Coordinated Matching (2012)                  | validated sequential policy 의 k-step 행동을 Monte Carlo simulation 으로 근사, batch selection problem 으로 재정의 |
| 2.2.2 Representative Subset Selection > k-Center             | Active Learning for Convolutional Networks: A Core-Set Approach (2018) | core-set 선택 문제를 $k$-Center 최적화로 환원                                                                      |

##### 2.2.3 Aggressive Sampling

| 분류                                         | 논문명                                                                 | 분류 근거                                                                                                    |
| -------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 2.2.3 Aggressive Sampling > Target-Dependent | Efficient Active Learning of Halfspaces: An Aggressive Approach (2012) | target-dependent bound 와 margin-dependent approximation guarantee 를 통한 near-optimal aggressive selection |

#### 2.3 온라인/스트림 기반 선택 (Online/Streaming Selection)

이 범주에는 개념 드리프트나 데이터 스트림 환경에서 active learning 을 연구한 논문들이 포함됩니다.

##### 2.3.1 Data Stream Active Learning

| 분류                                                    | 논문명                                            | 분류 근거                                                                                |
| ------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 2.3.1 Data Stream Active Learning > Stationary/Drifting | Active Learning for Data Streams: A Survey (2023) | 분포 안정성 (stationary/drifting), 처리 방식 (single-pass/batch) 으로 4 가지 범주로 분류 |

#### 2.4 고유한 동기를 통한 탐색 (Intrinsic Motivation)

이 범주에는 모델의 능력 발전이나 내부 동기를 바탕으로 샘플을 선택하는 방법론을 연구한 논문들이 포함됩니다.

##### 2.4.1 Intrinsic Motivation & Task Space Exploration

| 분류                                                                             | 논문명                                                                                           | 분류 근거                                                                                                                  |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| 2.4.1 Intrinsic Motivation & Task Space Exploration > Task Space Goal Generation | Active Learning of Inverse Models with Intrinsically Motivated Goal Exploration in Robots (2013) | task space 에서 goal 능동적으로 생성·선택, competence progress 기반 관심도 측정, intrinsic motivation 기반 active learning |

#### 2.5 강화학습 기반 선택 (Reinforcement Learning Based)

이 범주에는 active learning 정책 자체를 RL 문제로 정식화하거나 강화학습 기법을 활용한 논문들이 포함됩니다.

##### 2.5.1 RL Policy Learning

| 분류                                          | 논문명                                                                      | 분류 근거                                                                      |
| --------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 2.5.1 RL Policy Learning > Deep Q-Network     | Learning how to Active Learn: A Deep Reinforcement Learning Approach (2017) | active learning 정책을 MDP 로 정식화, Deep Q-Network 를 통한 정책 학습 및 전이 |
| 2.5.1 RL Policy Learning > Multi-Armed Bandit | Active Learning for Data Streams: A Survey (2023)                           | multi-armed bandit 접근이 exploration-exploitation trade-off 제공              |

#### 2.6 인간-기계 협업 (Human-in-the-Loop)

이 범주에는 인간의 피드백이나 라벨링을 활용하는 active learning 시스템을 연구한 논문들이 포함됩니다.

##### 2.6.1 Human Feedback Integration

| 분류                                                       | 논문명                                              | 분류 근거                                                                                                                    |
| ---------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 2.6.1 Human Feedback Integration > Online Human Preference | Deep Active Learning for Dialogue Generation (2016) | 오프라인 supervised 모델을 online 단계에서 human preference 를 암묵적 reward 신호로 학습하는 incremental supervised learning |

#### 2.7 Crowd-Sourcing 최적화 (Crowd-Sourcing Optimization)

이 범주에는 crowd-sourced 환경의 active learning, 특히 라벨러 예산 분배 및 병렬 처리에 초점을 맞춘 논문들이 포함됩니다.

##### 2.7.1 Crowd-Sourced Active Learning System

| 분류                                                             | 논문명                                             | 분류 근거                                                                                                                                             |
| ---------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.7.1 Crowd-Sourced Active Learning System > Practical Framework | Active Learning for Crowd-Sourced Databases (2014) | crowd-sourced DB 시스템의 practical 요구사항 (generality, black-box, batching, parallelism, noise management) 을 모두 만족하는 first practical method |

#### 2.8 Survey 및 벤치마크 (Survey & Benchmark)

이 범주에는 분야 전체를 종합 분석하거나 공개 벤치마크를 제공하는 논문들이 포함됩니다.

##### 2.8.1 Survey

| 분류                                           | 논문명                                                                     | 분류 근거                                                                                                       |
| ---------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 2.8.1 Survey > DAL Taxonomy                    | A Survey on Deep Active Learning: Recent Advances and New Frontiers (2024) | DAL 방법론을 5 개의 관점 (annotation/query/model paradigm/training) 으로 체계화하고 연구 흐름 종합 분석         |
| 2.8.1 Survey > Online Active Learning Taxonomy | Active Learning for Data Streams: A Survey (2023)                          | online active learning 분야를 공통 설계 축 (uncertainty, drift, latency, evaluation) 으로 재해석, taxonomy 수립 |

##### 2.8.2 Benchmark

| 분류                                   | 논문명                                                                         | 분류 근거                                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| 2.8.2 Benchmark > Regression Benchmark | A Framework and Benchmark for Deep Batch Active Learning for Regression (2022) | 15 개 tabular regression 데이터셋, 15 번 acquisition step, 256 개씩 배치 선택한 대규모 공개 벤치마크 제공 |

### 3. 종합 정리

본 연구 분류 체계는 8 편 논문(Active Learning for Convolutional Networks(2018), Active Learning for Data Streams(2023), Batch Active Learning via Coordinated Matching(2012), Efficient Active Learning of Halfspaces(2012), Deep Active Learning for Dialogue Generation(2016), Active Learning of Inverse Models(2013), Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds(2019), Active Learning for Crowd-Sourced Databases(2014)) 을 핵심 문제 정의에 따라 9 개의 범주로 체계화했습니다. 또한 4 편 논문(Bayesian Deep Learning(2017), Deep Bayesian Active Learning(2017), ActiveHARNet(2019), DEAL(2020)) 은 불확실성 추정을 핵심 문제로, 2 편 논문(Deep Active Learning: Unified(2019), A Framework and Benchmark(2022)) 은 배치 선택의 통합 프레임워크로, 2 편 논문(Active Learning for Data Streams, Deep Active Learning for Dialogue Generation) 은 각각 online/streaming 설정과 human-in-the-loop 학습을, 2 편 논문(Survey on Deep Active Learning(2024), A Survey on Active Learning for Data Streams(2023)) 은 분야 종합 분석으로 분류됩니다. 이 체계는 active learning 의 핵심 문제 (불확실성, 배치 선택, online 처리, 인간의 참여, crowd-optimization) 를 6 가지 주요 관점으로 조직하며, 향후 연구 방향 (uncertainty 통합, hybrid strategy, foundation model adaptation) 과 연구 지형의 명확한 구조를 제공합니다.

## 2장. 방법론 분석

### 1. 공통 문제 설정 및 접근 구조

제공된 논문들은 모두 **제한된 라벨링 예산**(labeling budget)이라는 공통된 문제 설정에서 출발한다. 이는**unlabeled pool**에서**informative subset**을 선택하는**acquisition** 문제로 수식화된다.

#### 1.1 문제의 공통 구조

| 구성 요소 | 설명                                                                                                       |
| --------- | ---------------------------------------------------------------------------------------------------------- |
| **입력**  | Unlabeled pool ($\mathcal{D}_u$), Labeled seed set ($\mathcal{D}_l$), Current model ($\mathcal{M}_\theta$) |
| **처리**  | Acquisition function 평가 → Batch selection → Oracle 질의                                                  |
| **출력**  | Informative batch $\mathcal{B} \subset \mathcal{D}_u$                                                      |
| **목표**  | 최소 라벨링 예산으로 일반화 성능 극대화                                                                    |

#### 1.2 접근 방식의 공통 구조

전체 논문들은 다음과 같은 **3 단계의 공통 파이프라인**을 공유한다:

```text
1. Uncertainty/Diversity Estimation
   ↓
2. Selection/Optimization (Batch 구성)
   ↓
3. Oracle & Model Update
```

**모델 중심 vs 시스템 중심**으로 구분할 수 있다:

| 모델 중심                   | 시스템 중심                    |
| --------------------------- | ------------------------------ |
| CNN, Bayesian CNN, NN 수정  | System architecture 설계       |
| Acquisition function 최적화 | Budget allocation, Parallelism |

### 2. 방법론 계열 분류

제공된 논문들을 핵심 설계 관점에서 **6 가지 주요 계열**로 그룹화한다.

#### (계열 1) Kernel 기반 프레임워크 계열

**계열 정의**: Base kernel 과 kernel transformation 으로 회귀·분류 문제를 재정의, selection method 로 배치 선택

| 방법론 계열            | 논문명               | 핵심 특징                                               |
| ---------------------- | -------------------- | ------------------------------------------------------- |
| Kernel-based Framework | BMDAL (2022)         | base kernel → transformation → selection 의 3 단계 조합 |
|                        | Deep Bayesian (2017) | Kernel 기반 대신 CNN + MC dropout                       |
|                        | BADGE (2019)         | Gradient embedding 공간에서 k-means++ 선택              |

- **공통 특징**:
  - Similarity-based representation 활용
  - Uncertainty/diversity 를 embedding/gradient 로 근사
  - k-center, k-means 등 clustering 기반 선택

#### (계열 2) Distribution Matching 계열

**계열 정의**: Active learning 을 distribution matching 문제로 정식화, GAN-style adversarial training 을 통해 querying/training 통합

| 방법론 계열           | 논문명                      | 핵심 특징                                      |
| --------------------- | --------------------------- | ---------------------------------------------- |
| Distribution Matching | Unified DML (2019)          | Wasserstein distance, min-max adversarial 구조 |
|                       | Deep Active Learning (2016) | 오프라인+온라인 2 단계 학습                    |

- **공통 특징**:
  - Labeled/unlabeled representation alignment
  - Uncertainty-diversity 를 단일 목적함수에서 통합
  - GAN-style critic과 feature extractor 구조

#### (계열 3) Geometric/Core-Set 계열

**계uel 정의**: Feature space 전체를 geometric coverage 관점에서 대표 subset 으로 축소, $k$-Center 문제로 환원

| 방법론 계열        | 논문명                      | 핵심 특징                             |
| ------------------ | --------------------------- | ------------------------------------- |
| Geometric/Core-Set | Core-Set Approach (2018)    | $k$-Center 문제, geometric bound, MIP |
|                    | Efficient Halfspaces (2012) | Version space volume 최소화, greedy   |

- **공통 특징**:
  - Feature space 전체를 고르게 커버
  - Geometric bound 및 convex hull 기반
  - Lipschitz continuity 보장

#### (계열 4) Evidential/Bayesian 계측 계열

**계uel 정의**: Softmax 대신 Dirichlet evidence 를 학습, belief mass 와 uncertainty mass 동시 추정, single forward pass 로 획득

| 방법론 계열 | 논문명                        | 핵심 특징                                      |
| ----------- | ----------------------------- | ---------------------------------------------- |
| Evidential  | DEAL (2020)                   | Dirichlet density, Softsign activation, KL reg |
|             | Bayesian Uncertainties (2017) | Aleatoric/Epistemic 분리, heteroscedastic      |
|             | ActiveHARNet (2019)           | Dropout-based Bayesian uncertainty             |

- **공통 특징**:
  - Evidence vector 직접 학습
  - Aleatoric(데이터 노이즈)/Epistemic(모델 ignorance) 구분
  - Single forward pass 로 uncertainty 획득

#### (계uel 5) Sequential/Reinforcement 계열

**계uel 정의**: Active learning 을 MDP 로 정식화, RL 정책이나 sequential policy simulation 으로 선택 전략 학습

| 방법론 계열   | 논문명                            | 핵심 특징                                    |
| ------------- | --------------------------------- | -------------------------------------------- |
| Sequential/RL | Reinforcement (2017)              | DQN policy network, reward shaping           |
|               | Batch Active Learning (2012)      | Sequential policy 시뮬레이션, BCM 최적화     |
|               | Deep Batch Active Learning (2016) | Curriculum-like, semi-supervised integration |

- **공통 특징**:
  - Sequential policy 를 batch 로 확장
  - Reward shaping(Accuracy improvement) 기반
  - Monte Carlo simulation 으로 분포 추정

#### (계uel 6) Domain-Specific 계열

**계uel 정의**: 특정 도메인(streaming, robot, dialogue, mobile) 에 최적화된 specialized 방법론

| 방법론 계열 | 논문명                | 핵심 특징                                    |
| ----------- | --------------------- | -------------------------------------------- |
| Streaming   | Data Streams (2023)   | Single-pass, concept drift 대응, prequential |
|             | ActiveHARNet (2019)   | On-device, incremental update                |
|             | Inverse Models (2013) | Task space exploration, competence progress  |
|             | Dialogue (2016)       | Human-in-the-loop, online adaptation         |
|             | Crowd-Sourced (2014)  | Bootstrap, partitioning, parallelism         |

- **공통 특징**:
  - Domain-specific 제약 조건 (resource, streaming, crowd)
  - Specialized evaluation(regime, drift detection)
  - Domain-aware acquisition strategy

### 3. 핵심 설계 패턴 분석

#### 3.1 Acquisition Function 설계 패턴

| 패턴 유형        | 구조                                | 대표 논문                        |
| ---------------- | ----------------------------------- | -------------------------------- |
| Uncertainty-only | Max entropy, variance, MC dropout   | Deep Bayesian (2017)             |
| Diversity-only   | k-center, k-means++, representative | Core-Set (2018)                  |
| Hybrid           | Wasserstein, gradient embedding     | Unified DML (2019), BADGE (2019) |
| Evidential       | Dirichlet belief/uncertainty        | DEAL (2020)                      |
| Evidential       | Aleatoric/Epistemic                 | Bayesian Uncertainties (2017)    |

#### 3.2 Batch Selection 최적화 패턴

| 최적화 유형  | 알고리즘                  | 문제 환원                    |
| ------------ | ------------------------- | ---------------------------- |
| Greedy       | Supermodular minimization | BCM objective                |
| Clustering   | k-means++ initialization  | Gradient embedding           |
| Optimization | k-MMM matching            | $k$-Center 문제              |
| Simulation   | Monte Carlo trajectory    | Sequential policy 시뮬레이션 |

#### 3.3 모델 설계 패턴

| 설계 축          | 옵션                                  | 논문 예                               |
| ---------------- | ------------------------------------- | ------------------------------------- |
| **Backbone**     | CNN, Bayesian CNN, LSTM, Transformer  | Deep Bayesian (2017), Dialogue (2016) |
| **Uncertainty**  | MC dropout, Evidential, Ensemble      | DEAL (2020), ActiveHARNet (2019)      |
| **Learning**     | Fine-tuning, Re-training, Incremental | DEAL (2020), ActiveHARNet (2019)      |
| **Architecture** | Deterministic, Bayesian, Hybrid       | DEAL (2020), Deep Bayesian (2017)     |

### 4. 방법론 비교 분석

#### 4.1 계열 간 차이점

| 차원          | Kernel 기반                     | Distribution Matching  | Geometric          | Evidential           |
| ------------- | ------------------------------- | ---------------------- | ------------------ | -------------------- |
| **문제 접근** | Similarity-based                | Distribution alignment | Geometric coverage | Evidence modeling    |
| **구조**      | 3 단계 (kernel/trans/selection) | Min-max adversarial    | Geometric bound    | Dirichlet inference  |
| **적용 대상** | Regression, Classification      | Classification         | CNN, Halfspace     | Image classification |
| **복잡도**    | Low-Medium                      | Medium                 | Medium             | Low                  |
| **확장성**    | Pool-size 대비 우수             | Unlabeled pool 필요    | Feature space 의존 | Single pass 장점     |

#### 4.2 트레이드오프 분석

| 트레이드오프                  | A 편향                                                           | B 편향                                                                |
| ----------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| **Uncertainty vs Diversity**  | Uncertainty-only: OOD 잘 감지<br>단 redundancy 문제              | Diversity-only: Representative<br>단 계산 비용 증가                   |
| **Simple vs Accurate**        | Simple (entropy, MC dropout): Fast<br>단 정확도 제한             | Complex (Wasserstein, k-MMM): Accurate<br>단 계산 비용 높음           |
| **Deterministic vs Bayesian** | Deterministic: Computationally cheap<br>단 uncertainty 표현 낮음 | Bayesian: Full uncertainty<br>단 inference 비용 높음                  |
| **Single vs Multi-stage**     | Single-stage: Simple pipeline<br>단 limited optimization         | Multi-stage (offline+online): Better adaptation<br>단 complexity 증가 |

#### 4.3 방법론 선택 가이드

| 시나리오                   | 권장 계열                |
| -------------------------- | ------------------------ |
| Pool 기반, regression      | Kernel-based Framework   |
| Image classification       | Evidential               |
| Streaming environment      | Streaming 계열           |
| Resource-constrained       | Evidential (single pass) |
| Diversity 중시             | Geometric/Core-Set       |
| Uncertainty-diversity 통합 | Distribution Matching    |

### 5. 방법론 흐름 및 진화

#### 5.1 시간 흐름에 따른 진화

| 시기                         | 주요 경향                          | 대표적 논문                                                                          |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------ |
| **초기**(2012-2013)          | Version space, simple uncertainty  | Efficient Halfspaces (2012)<br>Inverse Models (2013)                                 |
| **Bayesian 도입**(2016-2017) | MC dropout, epistemic/aleatoric    | Deep Active Learning (2016)<br>Deep Bayesian (2017)<br>Bayesian Uncertainties (2017) |
| **Batch 최적화**(2012-2019)  | BCM, k-means++, gradient           | Batch Active Learning (2012)<br>Core-Set (2018)<br>BADGE (2019)                      |
| **Evidential**(2019-2020)    | Dirichlet, single forward pass     | Unified DML (2019)<br>DEAL (2020)                                                    |
| **Taxonomy/Meta**(2023-2024) | Stream survey, field-wide overview | Data Streams (2023)<br>Survey (2024)                                                 |

#### 5.2 방법론 진화의 동인

1. **Uncertainty 표현 개선**: MC dropout → Evidential → Aleatoric/Epistemic 분리
2. **Batch 효율성 요구**: Sequential → Batch → Coordinated matching
3. **도메인 확장**: Image → Dialogue → Robot → Streaming
4. **시스템 설계**: Model-only → System-aware → Domain-specific

#### 5.3 접근 방식의 변화

```text
Early: Simple uncertainty maximization
  ↓
Mid: Uncertainty-diversity trade-off
  ↓
Recent: Unified frameworks (Evidential, Distribution matching)
```

### 6. 종합 정리

제공된 논문들을 통해 Deep Active Learning 의 방법론 지형을 다음과 같이 재구성할 수 있다:

1. **계열 구조**: 6 가지 주요 계열로 분류되며, 각 계열은 고유한 설계 철학과 최적화 대상 (uncertainty, diversity, geometric coverage, evidence, sequential policy, domain adaptation) 을 가진다.

2. **공통적 구조**: Acquisition → Selection → Oracle 의 3 단계 파이프라인이 공유되며, 이는 model design, query strategy, update rule 로 구현된다.

3. **핵심 패턴**:
   - Uncertainty: MC dropout, Evidential, Gradient embedding
   - Diversity: k-center, k-means++, Representative
   - Optimization: Greedy, Clustering, Adversarial, RL policy

4. **진화 경향**: 단순 uncertainty maximization 에서 uncertainty-diversity 통합, deterministic 에서 Bayesian 으로, simple model 에서 system-aware design 으로 진화.

5. **트레이드오프 축**:
   - Uncertainty vs Diversity
   - Simple vs Accurate  
   - Deterministic vs Bayesian
   - Single-stage vs Multi-stage

이 지형은 Active Learning 이 **uncertainty 표현**,**batch 최적화**,**도메인 적응**라는 3 가지 축으로 발전해 왔음을 보여준다. 각 계열은 서로 다른 trade-off 를 제공하며, 적용 시나리오에 따라 선택 전략이 결정된다.

## 3 장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 주요 데이터셋 유형

| 데이터셋 유형      | 대표적 데이터셋                                             | 평가 환경         |
| ------------------ | ----------------------------------------------------------- | ----------------- |
| Tabular regression | 15 개 대규모 tabular regression 데이터셋                    | 회귀 실험         |
| 이미지 분류        | MNIST, CIFAR-10, ISIC 2016, 3 개 이미지 분류 데이터셋       | 실험실 환경       |
| 의료 영상          | pediatric pneumonia chest X-ray (1,500/200/1,400)           | 실험실/의료 환경  |
| NLP/NER            | CoNLL2002/2003(en/de/es/nl), Cornell Movie Dialogs Corpus   | 실험실 환경       |
| 스트리밍           | stationary/drifting/evolving data stream, spam filtering 등 | 시뮬레이션/실환경 |
| 로봇 시뮬레이션    | 7/15/30 DOF redundant arm, 사족보행 로봇, 낚싯줄 제어       | 시뮬레이션        |
| crowd-sourced      | 3 개 Amazon Mechanical Turk, 15 개 UCI KDD                  | crowd-sourcing    |

#### 평가 환경 분류

| 환경 유형            | 사용 사례                                  | 비고                   |
| -------------------- | ------------------------------------------ | ---------------------- |
| 실험실/실제 데이터셋 | MNIST, CIFAR-10, ISIC 등 공개 데이터셋     | 표준화된 벤치마크      |
| 시뮬레이션           | 로봇 inverse kinematics, synthetic dataset | 제어된 환경            |
| crowd-sourcing       | Amazon Mechanical Turk                     | crowd noise 고려       |
| 스트리밍             | 실시간 데이터 흐름                         | holdout vs prequential |

#### 공통 평가 지표

| 지표                           | 사용 용도           | 주요 논문                 |
| ------------------------------ | ------------------- | ------------------------- |
| **accuracy**                   | 최종 학습 성능 평가 | почти 모든 논문           |
| **test error (%)**             | 모델 일반화 오차    | Bayesian CNN 2017         |
| **RMSE / MAE / Maximum error** | 회귀 성능 지표      | BMDAL 2022                |
| **label complexity**           | 라벨링 효율성       | Efficient Halfspaces 2012 |
| **F1 score**                   | NER 성능 평가       | Deep RL 2017              |
| **AUC / average precision**    | 의료 영상 분류      | ISIC 2016                 |
| **inference efficiency**       | on-device 추론 효율 | ActiveHARNet 2019         |
| **learning curve**             | 라벨 수 대비 성능   | BCM 2012, DEAL 2020       |

### 2. 주요 실험 결과 정렬

#### 회귀 및 이미지 분류 주요 결과

| 논문명               | 데이터셋/환경                                   | 비교 대상                                                                       | 평가 지표                                       | 핵심 결과                                                                                                                         |
| -------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| BMDAL (2022)         | 15 개 tabular regression, 20 개 random split    | random, BALD, BatchBALD, BAIT, ACS-FW, Core-Set, FF-Active, BADGE, LCMD         | RMSE, MAE, maximum error                        | LCMD 가 모든 지표에서 SOTA, NTK kernel 은 모든 selection method 에서 accuracy 향상                                                |
| DEAL (2020)          | MNIST 58k, CIFAR-10 48k, pediatric pneumonia    | softmax margin, entropy, least confidence, MC-Dropout, BADGE, random sampling   | test accuracy, 목표 라벨 수, runtime            | MNIST ResNet 대비 +1.06%, CIFAR-10 ResNet 대비 Deep Ensemble +0.51%, MNIST random 대비 280 장 절감, pneumonia random 대비 -243 장 |
| Core-Set (2018)      | 3 개 이미지 분류 데이터셋                       | random, uncertainty-based (entropy, Bayesian, DB), Bayesian AL, Wang & Ye       | classification accuracy                         | 모든 설정에서 제안법 우세, weakly supervised 에서 격차 최대, MIP 가 greedy 보다 약간 우월                                         |
| BCM (2012)           | 8 개 UCI 이진 분류 데이터셋                     | Fisher Information, Maximum Uncertainty top-k, Random, Sequential max entropy   | classification accuracy                         | batch size 20 에서 가장 우위, 일부 데이터셋 (Ionosphere, Breast) 에서 sequential 능가                                             |
| BADGE (2019)         | 다양한 환경 (architecture, batch size, dataset) | uncertainty-only, diversity-only, 기존 strong baseline                          | expected error 감소율, 성능 일관성              | 별도 tuning 없이 fixed parameter 로 작동, uncertainty-diversity trade-off 자연스럽게 수행                                         |
| Deep Bayesian (2017) | MNIST, ISIC 2016 (900 장)                       | Random, Mean STD, Deterministic CNN, MBR/Gaussian RF, DGN, Ladder Network, BALD | test error (%), labeled sample 수, AUC          | MNIST Random 대비 Variation Ratios 65% 감소 (835→295 labeled), ISIC에서 BALD AUC 개선                                             |
| ActiveHARNet (2019)  | HAR/fall detection 2 개 데이터셋                | deterministic classifier user-independent, HARNet                               | inference efficiency, accuracy, labeling burden | incremental learning 중 라벨 획득량 60% 감소 (두 데이터셋 모두)                                                                   |

#### 라벨링 비용 절감 결과

| 논문명              | 절감 효과                                                                       | 상대 비교 기준                 |
| ------------------- | ------------------------------------------------------------------------------- | ------------------------------ |
| DEAL (2020)         | MNIST 280 장 (-34.15%), CIFAR-10 6,800 장 (-24.29%), pneumonia 243 장 (-34.52%) | random 대비                    |
| ActiveHARNet (2019) | 60% 감소                                                                        | user-independent baseline 대비 |
| BMDAL (2022)        | SOTA 달성                                                                       | LCMD 가 기존 BMAL 방법들 우위  |

### 3. 성능 패턴 및 경향 분석

#### 공통 성능 개선 패턴

1. **Hybrid 전략의 우위**
   - BMDAL (2022): 좋은 selection rule + 좋은 kernel choice + 효율적 근사의 조합이 효과적
   - BADGE (2019): 별도 tuning 없이 uncertainty-diversity trade-off 자연스럽게 수행
   - Survey (2024): hybrid query 전략 (serial-form, criteria-selection, parallel-form) 관찰

2. **Foundation Model 결합 효과**
   - Survey (2024): labeled 데이터 10~20% 만 fine-tuning해도 full labeling 대비 5~10 배 효율적 (foundation model 시대)
   - 미래 전선: large pre-trained model 과 결합된 universal framework

3. **Batch selection 의 일관된 우위**
   - Core-Set (2018): batch CNN 학습 환경에서 representative coverage 가 uncertainty 보다 중요
   - BCM (2012): batch size 20 에서 학습 곡선 우위 명확
   - BMDAL (2022): batch deep active learning 특화 방법론

#### 데이터셋 의존성 및 환경 영향

| 조건                        | 성능 영향                     | 구체적 결과                                           |
| --------------------------- | ----------------------------- | ----------------------------------------------------- |
| **데이터 양 증가**          | epistemic 감소/aleatoric 일정 | Aleatoric/Epistemic 2017                              |
| **weakly supervised**       | core-set 접근 이점 확대       | Core-Set 2018                                         |
| **batch size 변화**         | 성능 일관성 변화              | BADGE 2019 (batch size, dataset 다양하게 변하는 환경) |
| **malignant 클래스 불균형** | acquisition function 실패     | Deep Bayesian 2017 (ISIC 에서 Variation Ratios 실패)  |
| **concept drift**           | prequential evaluation 필요   | Active Learning Streams 2023                          |

#### 논문 간 상충되는 결과

1. **Uncertainty-only 와 diversity-only 비교**
   - BADGE (2019): uncertainty-only 의 redundancy 문제 (상호 중복 샘플 선택), diversity-only 의 비정보성 문제
   - Deep Bayesian (2017): BALD 가 noisy point 보다 epistemic uncertainty 큰 sample 선호 (더 효율적)

2. **Sequential vs Batch trade-off**
   - BCM (2012): 일부 데이터셋 (Ionosphere, Breast) 에서 sequential 능가
   - Core-Set (2018): batch setting 에서 representative coverage 중요

3. **Evaluation protocol 의존성**
   - Active Learning Streams 2023: drifting stream 은 prequential, stationary 은 holdout 더 적합
   - Deep Bayesian 2017: ISIC 에서 variation ratios 실패 (malignant probability 가 benign 이미지에서도 높아)

### 4. 추가 실험 및 검증 패턴

#### 주요 검증 방법 패턴

| 검증 유형                 | 사용 방법                                       | 대표 논문                                                                 |
| ------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------- |
| **Ablation study**        | uncertainty vs diversity component 분리         | BADGE 2019                                                                |
| **Random baseline**       | 무작위 선택 vs 제안법 비교                      | 거의 모든 논문                                                            |
| **Random split 반복**     | 20 회/100 회 반복으로 통계적 유의성 확보        | BMDAL 2022 (20 개 random split), Deep Bayesian 2017 (100 회 반복)         |
| **Sensitivity analysis**  | hyperparameter 변화, dataset scale              | Efficient Halfspaces 2012 (MNIST 변형), Deep RL 2017 (learning rate 튜닝) |
| **Condition variation**   | architecture choice, batch size, dataset 다양화 | BADGE 2019                                                                |
| **Cross-domain transfer** | cross-lingual NER, cold-start                   | Deep RL 2017                                                              |
| **k-fold CV**             | stopping 결정                                   | Crowd-ER 2014                                                             |

#### 학습률 튜닝 실험 (Deep RL 2017)

| 학습률 | 안정성 | one-shot adaptation | 비고 |
| ------ | ------ | ------------------- | ---- |
| 0.005  | 균형   | 균형                | 최적 |
| 0.01   | -      | -                   | -    |
| 0.05   | -      | -                   | -    |

### 5. 실험 설계의 한계 및 비교 주의점

#### 비교 조건의 불일치

| 문제 유형                   | 구체적 사례                         | 영향                              |
| --------------------------- | ----------------------------------- | --------------------------------- |
| **Evaluation metric 차이**  | accuracy, RMSE, F1 등 도메인별 지표 | 직접 비교 어려움                  |
| **Labeled budget 차이**     | 초기 20~256 개, batch size 10~20 등 | 성능 비교 시 맥락 필요            |
| **Model architecture 차이** | LeNet, ResNet, VGG16 등             | 모델 자체 영향과 AL 영향 분리困難 |
| **Random seed 차이**        | 10, 20, 100 회 반복                 | 통계적 유의성 확보 정도 불일치    |

#### 데이터셋 의존성

| 논문명                       | 데이터셋 제한                            | 일반화 한계                              |
| ---------------------------- | ---------------------------------------- | ---------------------------------------- |
| BMDAL (2022)                 | tabular regression 에 집중               | 이미지/딥 설정 일반화 제한               |
| Core-Set (2018)              | 이미지 분류 데이터셋 3 개                | 일반화 주장에 구체적인 근거 부족         |
| Deep Bayesian (2017)         | MNIST, ISIC 2016                         | 의료 영상 한계                           |
| Crowd-ER (2014)              | 분류 task 특화                           | regression/missing-item future work      |
| ActiveHARNet (2019)          | 공개 데이터셋 중심                       | 다양한 센서 구성/장기 실환경 증거 제한적 |
| Active Learning Streams 2023 | deep learning 기반 streaming 성숙도 낮음 | 실용성 한계                              |

#### 평가 지표의 한계

| 지표           | 한계                                               | 대안 제안                       |
| -------------- | -------------------------------------------------- | ------------------------------- |
| accuracy       | 단일 지표로 모든 성능 포착 불가                    | RMSE, MAE, F1 등 다중 지표      |
| AUC            | medical image 분류에서 average precision 대체      | 클래스 불균형 상황에서의 한계   |
| test error (%) | 오차 절대값만 보고 상대적 효율성 부족              | labeled sample 수와 함께 분석   |
| runtime        | acquisition time vs labeling cost trade-off 불명확 | single forward pass 효율성 고려 |
| cold-start     | held-out data 필요                                 | 기본 알고리즘에 제약            |

### 6. 결과 해석의 경향

#### 저자들의 공통 해석 경향

1. **Framework 통합 강조**
   - BMDAL (2022): "회귀 특성에 맞는 kernelized formulation 이 유리"
   - Survey (2024): DAL 을 dataset distillation/pruning/augmentation 과는 다른 "sample selection paradigm" 으로 재정의
   - Deep Unified 2019: "이론적 틀에서 도출된 query criterion 이 실제로 성능과 효율성을 동시에 개선"

2. **Foundation model 강조**
   - Survey (2024): "foundation model 과의 결합이 future 전선"
   - Universal framework 를 통한 방법 통합

3. **Uncertainty quality 강조**
   - DEAL (2020): "uncertainty quality 는 높지만 acquisition time 은 낮아 효율적"
   - Bayesian uncertainty estimation 이 incremental learning 이나 OOD 감지에 필수

4. **Robustness 및 general-purpose 강조**
   - BADGE (2019): "best baseline 과 동등하거나 우수한 일관성 있는 성능"
   - 별도 hyperparameter 없이 fixed parameter 로 작동

5. **Survey 는 "지도" 역할 강조**
   - Survey (2024): survey 는 "연구 gap 탐색 도구", research guidance 역할
   - method 간 direct apples-to-apples 비교는 어려움

#### 해석과 실제 결과 구분

| 해석 주장                | 실제 관찰 결과                                                  | 구분                     |
| ------------------------ | --------------------------------------------------------------- | ------------------------ |
| LCMD 가 SOTA             | RMSE/MAE 기준 SOTA 달성                                         | 결과 일관                |
| Hybrid 전략 우수         | hybrid 는 serial-form, criteria-selection, parallel-form 세분화 | 일부 방법론적 차이 있음  |
| Foundation model 효율성  | 10~20% labeled 이 5~10 배 효율적                                | 수치적 근거 명확         |
| Batch selection 일관성   | batch size 20 에서 batch 우위, 일부 데이터셋 sequential 우위    | 데이터셋 의존성          |
| Uncertainty quality 필요 | DEAL 에서 +1.01%~+1.51% 정확도 향상                             | 일관된 결과              |
| cold-start 문제          | pretrained model 기반 PAL 이 baseline 보다 우수                 | 해결 가능 사례           |
| concept drift 대응       | prequential evaluation 필요                                     | evaluation protocol 의존 |

### 7. 종합 정리

전체 실험 결과는 deep active learning 이 단일 방법론이 아닌 framework 통합과 조건별 최적화 필요성을 보여준다. 회귀 작업에서는 LCMD 와 sketched NTK 결합이 SOTA 를 달성하고, 이미지 분류에서는 uncertainty-quality 와 diversity 의 균형 (BADGE 등) 이 핵심이며, batch selection 이 일관된 우위를 보인다. Foundation model 시대에는 10~20% labeled 데이터로 5~10 배 효율성을 보이는 pre-trained 모델 활용이 future direction 이다. 평가 지표는 domain 에 따라 accuracy, RMSE, F1 등 다양하게 사용되며, cross-domain transfer 와 cold-start 문제를 해결하기 위한 cold-start/pretrained 모델 연구가 지속된다. 데이터셋 의존성 (medical, tabular, image 등) 이 성능에 큰 영향을 미치고, evaluation protocol(holdout vs prequential) 은 방법 설정과 일치해야 함을 주의해야 한다.
