# Few-shot Learning

## 서론

### 1. 연구 배경

이 보고서는 Few-Shot Learning(FSL) 분야에서 제시된 다양한 연구들을 체계적으로 분석하고 정리한 것인데, 구체적인 범위는 다음과 같다:

**문제 정의**: Few-shot classification, segmentation, object detection, dialogue 이해 등 다양한 태스크를 대상으로 한 연구들을 분류

**방법론**: Meta-learning 기반 접근 (metric-based, optimization-based, model-based), Self-supervised learning, Cross-domain adaptation, Incremental learning, Zero-shot/Knowledge Graph 기반 방법 등 다양한 방법론을 포함

**연구 성격**: 알고리즘 제안형 연구, Survey 논문, 재평가/분석형 연구 등을 구분하여 체계화

### 2. 문제의식 및 분석 필요성

Few-Shot Learning 분야는 2017 년 Meta-SGD 이후 급격히 발전하며 여러 방법론적 접근이 등장했는데, 이를 체계적으로 비교·분석할 필요가 있다. 그러나 관련 논문들은 다음과 같은 이유로 비교가 어렵다:

- **방법론적 차이**: Metric-based(prototype/distance), Optimization-based(second-order meta-learning), Model-based(external memory) 등 접근 방식마다 근본적으로 다른 학습 구조를 가짐
- **실험 설정의 다양성**: Standard, Cross-domain, Semi-supervised, Incremental 등 다양한 설정에서 결과를 보고하며, 동일한 태스크라도 데이터셋, backbone, evaluation protocol 차이가 큼
- **Survey 논문 부재**: 방법론적 분류 체계나 연구 지형도를 제시하는 종합적인 survey 는 드물며, 각 연구마다 제한된 비교만 수행

따라서 Few-Shot Learning 연구의 지형도를 명확히 하고, 방법론의 장단점, 적용 범위, 성능 경향을 체계적으로 정리하는 분석이 필요하다.

### 3. 보고서의 분석 관점

이 보고서는 다음과 같은 세 가지 관점에서 문헌을 정리한다:

**1 차원: 연구체계 분류**  
연구 분류 체계 수립 기준 (문제 정의 중심, 방법론적 접근 관점, 시스템 구성 관점, 도메인 특성 반영, 연구 성격 구분) 을 적용하여 논문들을 체계적으로 대분류하고 하위 범주로 세분화한다.

**2 차원: 방법론 분석**  
공통 문제 설정 (input, processing, output) 과 구조적 패턴을 분석하고, 방법론 계열별 (Metric/optimization/memory/augmentation/transductive/innovative) 특징, 장단점, 복잡도, 학습/추론 방식을 비교한다.

**3 차원: 실험결과 분석**  
주요 데이터셋, 평가 환경, baseline, 평가 지표를 정리하고, 실험 결과 (few-shot classification, cross-domain, semi-supervised 등) 를 정렬하여 성능 패턴, 경향, 데이터셋 의존성, 평가 지표 편차 등을 분석한다.

### 4. 보고서 구성

보고서는 총 3 장으로 구성되어 있으며, 각 장은 다음과 같은 관점에서 분석을 담당한다:

**1 장. 연구체계 분류**: Few-Shot Learning 연구를 문제 정의 (classification, segmentation, detection 등), 방법론 (baseline 재발견, metric-based, optimization-based 등), 연구 성격 (survey, algorithmic 등) 의 세 가지 기준으로 체계화한다. Metric-based, Optimization-based, Model-based, Self-supervised, Incremental, Cross-domain, Fine-grained, Task-Specific, Zero-shot/KG 기반 등 다양한 하위 범주로 논문들을 분류하고 survey 논문을 방법론적 분류 체계 제시 역할로 별도 정리한다.

**2 장. 방법론 분석**: Few-Shot Learning 파이프라인 (pre-training, adaptation, inference) 의 공통 구조를 분석하고, 방법론 계열별 (Metric, Optimization, Model, Augmentation, Transductive, Innovative) 구조적 특징, 학습 방식, 추론 방식, 장단점, 복잡도를 비교한다. 핵심 설계 패턴 (Base-Novel Separation, Two-Stage Paradigm, Teacher-Student Framework 등), Loss function, Optimization strategy, Regularization 패턴을 분석하며, 계열 간 차이점, 적용 대상, 확장성을 체계적으로 정리한다.

**3 장. 실험결과 분석**: 주요 데이터셋, 평가 환경, baseline, 평가 지표를 정리하고, few-shot classification, cross-domain, semi-supervised, incremental, segmentation/detection, KBQA 등 다양한 태스크별 실험 결과를 정렬하여 성능을 비교한다. 성능 개선 패턴, 조건별 성능 편차, 평가 지표 의존성, 비교 조건 불일치, 데이터셋 의존성, 해석 경향 등을 분석하고, 실험 결과 해석 시 주의사항을 제시한다.

## 1 장. 연구체계 분류

### 1.1. 연구 분류 체계 수립 기준

이 보고서에서 논문들을 분류할 때는 다음과 같은 기준을 적용하였다:

1. **문제 정의 중심 분류**: 논문이 해결하려는 근본적인 few-shot 학습 문제 (classification, segmentation, object detection, dialogue 등) 를 기준으로 대분류를 수립
2. **방법론적 접근 관점**: 해당 논문이 제안하는 주요 기술적 접근 방식 (meta-learning, self-supervised learning, fine-tuning, knowledge graph 등) 을 기준으로 하위 범주 설정
3. **시스템 구성 관점**: few-shot learning 의 전체 파이프라인 (pre-training, adaptation, inference) 에서 담당하는 역할에 따른 분류
4. **도메인 특성 반영**: 표준 설정, cross-domain, incremental, semi-supervised, generalized 등 다양한 실험 설정을 구분하여 체계화
5. **연구 성격 구분**: 알고리즘 제안형 연구, survey 논문, 재평가/분석형 연구 등 연구 목적에 따른 분류

### 1.2. 연구 분류 체계

#### 1.2.1. Few-Shot Classification 방법론

이미지/텍스트 등 few-shot 설정에서의 학습 대상 클래스 분류에 초점을 맞춘 연구들

##### 1.2.1.1. Baseline 와 재평가 연구

simple fine-tuning 기반 baseline 의 강점 재발견과 재평가 연구

| 분류                                      | 논문명                                                                                   | 분류 근거                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Few-Shot Classification > Baseline 재발견 | Revisiting Fine-tuning for Few-shot Learning (2019)                                      | 적응형 최적화 설정만으로도 complex meta-learning 대비 강력한 baseline 입증 |
| Few-Shot Classification > Baseline 재발견 | Partial Is Better Than All: Revisiting Fine-tuning Strategy for Few-shot Learning (2021) | 층별 부분 학습률 탐색으로 fixed fine-tuning 전략 일반화 및 개선            |

##### 1.2.1.2. Metric-Based Methods

prototype 및 embedding similarity 기반 metric 공간 구성 방식

| 분류                                   | 논문명                                                                             | 분류 근거                                                                         |
| -------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Few-Shot Classification > Metric-based | Few-Shot Learning with Geometric Constraints (2020)                                | feature 와 weight 의 $l_2$ 정규화를 통한 geometric constraint 기반 metric control |
| Few-Shot Classification > Metric-based | Concept Learners for Few-Shot Learning (2020)                                      | 개념별로 분리된 metric space 와 prototype 을 학습하는 concept decomposition       |
| Few-Shot Classification > Metric-based | Meta-learning approaches for few-shot learning: A survey of recent advances (2023) | metric-learning (prototype/attention) 기반 방법의 체계적 survey                   |
| Few-Shot Classification > Metric-based | On Episodes, Prototypical Networks, and Few-shot Learning (2020)                   | metric-based learner 에서 episodic training 의 불필요성과 NCA 비교                |

##### 1.2.1.3. Optimization-Based Methods

meta-initialization/meta-update 규칙을 학습하는 최적화 기반 접근법

| 분류                                         | 논문명                                                                            | 분류 근거                                                                                 |
| -------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Few-Shot Classification > Optimization-based | Meta-SGD: Learning to Learn Quickly for Few-Shot Learning (2017)                  | 업데이트 규칙 자체 (초기화, gradient 방향, learning rate) 를 최적화 대상으로 함           |
| Few-Shot Classification > Optimization-based | Meta-learning for Semi-Supervised Few-Shot Classification (2018)                  | few-shot episode 내부에서 unlabeled set 을 prototype refinement 에 활용하는 meta-learning |
| Few-Shot Classification > Optimization-based | Self-Promoted Supervision for Few-Shot Transformer (2022)                         | teacher 를 통한 pseudo label 생성으로 ViT token dependency 학습 촉진                      |
| Few-Shot Classification > Optimization-based | Self-Supervised Prototypical Transfer Learning for Few-Shot Classification (2020) | prototype 기반 self-supervised embedding 학습 후 fine-tuning 으로 adaptation              |

##### 1.2.1.4. Model-Based Methods

외부 메모리나 model architecture 를 활용한 학습 방식

| 분류                                  | 논문명                                                                                             | 분류 근거                                                                    |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Few-Shot Classification > Model-based | XtarNet: Learning to Extract Task-Adaptive Representation for Incremental Few-Shot Learning (2020) | representation 자체를 meta-learning 의 대상으로 삼아 base-novel balance 개선 |
| Few-Shot Classification > Model-based | An Overview of Deep Learning Architectures in Few-Shot Learning Domain (2020)                      | external memory/memo-learner 를 활용한 models-based 접근법 정리              |
| Few-Shot Classification > Model-based | Learning Generative Models across Incomparable Spaces (2019)                                       | GW distance 를 사용하여 서로 다른 공간의 pairwise distance 구조 비교         |

##### 1.2.1.5. Self-Supervised Learning 기반 방법

unsupervised label 없이 representation 학습하는 방식

| 분류                                      | 논문명                                                                            | 분류 근거                                                                         |
| ----------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Few-Shot Classification > Self-Supervised | Self-Supervised Learning For Few-Shot Image Classification (2019)                 | SSL 로 사전학습된 대형 embedding 을 meta-learning 과 결합하여 representation 개선 |
| Few-Shot Classification > Self-Supervised | When Does Self-supervision Improve Few-shot Learning? (2019)                      | self-supervision 을 few-shot meta-learning 의 auxiliary regularizer 로 재해석     |
| Few-Shot Classification > Self-Supervised | Self-Supervised Prototypical Transfer Learning for Few-Shot Classification (2020) | label 없는 source domain 에서 prototype-based embedding 학습 후 transfer          |

##### 1.2.1.6. Incremental/Class-Incremental Methods

기존 클래스 지식 보존하면서 새로운 클래스 추가하는 방식

| 분류                                  | 논문명                                                                | 분류 근거                                                                      |
| ------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Few-Shot Classification > Incremental | Few-Shot One-Class Classification via Meta-Learning (2020)            | FS-OCC 문제를 one-class classification 으로 정식화하고 incremental learning    |
| Few-Shot Classification > Incremental | On the Soft-Subnetwork for Few-shot Class Incremental Learning (2022) | soft subnetwork 와 major-minor 역할 분리로 forgetting 과 overfitting 동시 완화 |
| Few-Shot Classification > Incremental | Few-Shot Learning with Geometric Constraints (2020)                   | base 성능 유지하면서 novel class 추가하는 incremental learning                 |

##### 1.2.1.7. Domain Adaptation 및 Cross-Domain 연구

도메인 이동 상황에서의 일반화 능력 개선 연구

| 분류                                   | 논문명                                                                                                     | 분류 근거                                                                          |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Few-Shot Classification > Cross-Domain | A Closer Look at Few-shot Classification (2019)                                                            | cross-domain 설정 (mini-ImageNet→CUB) 에서 simple fine-tuning 의 강점 입증         |
| Few-Shot Classification > Cross-Domain | Few-Shot Learning with Geometric Constraints (2020)                                                        | cross-domain 에서 feature 와 weight 의 geometric normalization 이 도메인 이동 완화 |
| Few-Shot Classification > Cross-Domain | A Comprehensive Survey of Few-shot Learning: Evolution, Applications, Challenges, and Opportunities (2022) | 도메인 차이로 인한 성능 급락 문제와 다양한 cross-domain 벤치마크                   |

##### 1.2.1.8. Fine-Grained 및 Task-Specific 연구

세밀한 클래스 구별이나 특수한 설정을 다루는 연구

| 분류                                    | 논문명                                                                                           | 분류 근거                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Few-Shot Classification > Fine-Grained  | Task Discrepancy Maximization for Fine-grained Few-Shot Classification (2022)                    | fine-grained setting 에서 클래스별 구별력 있는 채널을 동적으로 재가중                |
| Few-Shot Classification > Task-Specific | A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)                   | meta-learning 의 4 가지 주요 범주 (black-box, metric-based, layered, Bayesian)       |
| Few-Shot Classification > Task-Specific | Language Models as Few-Shot Learner for Task-Oriented Dialogue Systems (2020)                    | TOD 의 NLU, DST, ACT, NLG 모듈에서 few-shot priming 기반 성능 평가                   |
| Few-Shot Classification > Task-Specific | FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering (2023) | KBQA 에서 LLM 을 program translator 로 활용하는 synthetic data 생성 및 self-training |

#### 1.2.2. Few-Shot Segmentation 및 Object Detection

segmentation mask 나 bounding box 검출을 대상으로 하는 few-shot 연구

| 분류                                        | 논문명                                                                                             | 분류 근거                                                                                           |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Few-Shot Segmentation > Meta-Learning-Free  | Few-Shot Segmentation Without Meta-Learning: A Good Transductive Inference Is All You Need? (2021) | meta-learning 없이 support-supervision 과 query-aware regularization 으로 transductive inference    |
| Few-Shot Segmentation > Meta-Learning       | Meta-learning approaches for few-shot learning: A survey of recent advances (2023)                 | segmentation 도 domain 에 포함되며 metric/memory/learning 기반 분류                                 |
| Few-Shot Object Detection > Self-Supervised | A Survey of Self-Supervised and Few-Shot Object Detection (2021)                                   | FSOD 와 SSOD 를 unified low-data detection 프레임워크로 통합, pretraining 범위/adaptation 방식 분류 |

#### 1.2.3. Survey 및 Comprehensive Review 논문

few-shot learning 연구 지형도를 체계적으로 정리하는 survey 논문

| 분류                             | 논문명                                                                                                     | 분류 근거                                                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Survey > Methodological Taxonomy | Learning from Few Examples: A Summary of Approaches to Few-Shot Learning (2022)                            | meta-learning 기반 방법 (metric/optimization/model-based) 과 non-meta-learning baseline 분류                                     |
| Survey > Methodological Taxonomy | Meta-learning approaches for few-shot learning: A survey of recent advances (2023)                         | algorithm mechanics 관점에서 metric/memory/learning 기반 세분화                                                                  |
| Survey > Methodological Taxonomy | A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)                             | black-box/metric-based/layered/Bayesian 네 가지 주요 범주로 체계화                                                               |
| Survey > Challenge-Centered      | A Comprehensive Survey of Few-shot Learning: Evolution, Applications, Challenges, and Opportunities (2022) | "도전 과제 중심"의 새로운 taxonomy 를 제시하고 데이터 증강/전이학습/메타러닝/멀티모달 축으로 체계화                              |
| Survey > Comprehensive           | Zero-shot and Few-shot Learning with Knowledge Graphs: A Comprehensive Survey (2021)                       | KG 기반 zero-shot/few-shot 학습 6 패러다임 (mapping/data augmentation/propagation/class feature/optimization/transfer) 으로 분류 |
| Survey > Survey                  | An Overview of Deep Learning Architectures in Few-Shot Learning Domain (2020)                              | data augmentation, similarity metric, external memory, meta-level initialization 네 가지 아키텍처 계열                           |
| Survey > Survey                  | Meta-learning approaches for few-shot learning: A survey of recent advances (2023)                         | meta-learning 연구들의 체계적 분류와 benchmark 정리                                                                              |
| Survey > Survey                  | A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)                             | meta-learning 의 최근 흐름과 다양한 접근법을 네 가지 범주로 정리                                                                 |

#### 1.2.4. Semi-Supervised Few-Shot Learning

label 없는 데이터 를 활용하는 few-shot 방법론

| 분류                                       | 논문명                                                                                                     | 분류 근거                                                                                 |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Semi-Supervised FSL > Unlabeled Refinement | Meta-learning for Semi-Supervised Few-Shot Classification (2018)                                           | few-shot episode 내부에서 unlabeled set 을 prototype refinement 에 활용하는 명시적 방법론 |
| Semi-Supervised FSL > Unlabeled Learning   | A Comprehensive Survey of Few-shot Learning: Evolution, Applications, Challenges, and Opportunities (2022) | semi-supervised setting 과 transductive 설정을 별도 taxon 으로 제시                       |

#### 1.2.5. Zero-Shot 및 Knowledge Graph 기반 방법

전혀 새로운 클래스나 KG 구조를 활용한 learning 방법

| 분류                  | 논문명                                                                               | 분류 근거                                                      |
| --------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Zero-Shot > KG-Based  | Zero-shot and Few-shot Learning with Knowledge Graphs: A Comprehensive Survey (2021) | KG 가 seen/unseen 간 지식 전이를 매개하는 구조적 원천으로 활용 |
| Zero-Shot > Zero-Shot | Language Models as Few-Shot Learner for Task-Oriented Dialogue Systems (2020)        | zero-shot 설정에서도 성능 달성 (GrailQA, WebQSP 등)            |

### 1.3. 종합 정리

이 보고서는 few-shot learning 의 다양한 연구들을 **문제 정의** (classification, segmentation, object detection 등),**방법론** (baseline 재발견, metric-based, optimization-based, model-based, self-supervised, incremental, domain adaptation 등),**연구 성격** (algorithmic, survey, re-evaluation) 의 세 가지 관점에서 체계화하였다. 핵심적으로 meta-learning 기반 방법을 metric/optimization/model 기반 세 가지 관점으로 분류한 동시에, self-supervised learning, cross-domain adaptation, incremental learning, fine-grained classification, zero-shot learning, KG 기반 방법 등 다양한 접근법을 별도 하위 범주로 구분하였다. 특히 survey 논문들은 방법론적 분류 체계를 제시하는 역할을 하므로 별도의 category 로 정리하였으며, 이 분류 체계는 few-shot learning 의 연구 지형도에서 방법론적 기둥, 응용 영역, 연구 흐름을 명확히 드러내는 틀을 제공한다.

## 2장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

## 1.1 공통 문제 정의

모든 few-shot 학습 논문들이 해결하려는 핵심 문제는 다음과 같이 정의된다:

| 구분     | 정의                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------- |
| **입력** | Base class 데이터 ($X_b$), few-shot support set ($X_n$, N-way K-shot), unlabeled 데이터 (선택적), Knowledge Graph (선택적) |
| **처리** | Meta-training, few-shot adaptation, transductive inference                                                                 |
| **출력** | Query set의 분류 결과, bounding box 예측, segmentation mask, SPARQL 프로그램                                               |

## 1.2 공통 구조적 패턴

```text
┌─────────────────────────────────────────────────────────┐
│            Few-Shot Learning 공통 파이프라인            │
├─────────────────────────────────────────────────────────┤
│  Stage 1: Pre-training / Base Learning                  │
│    - External data 또는 base class 데이터로 학습        │
│    - Feature extractor 또는 detector 학습               │
│    - Self-supervised pretraining (선택적)               │
├─────────────────────────────────────────────────────────┤
│  Stage 2: Few-shot Adaptation                           │
│    - Episode-based training 또는 non-episodic           │
│    - Metric-based: prototype/distance 계산              │
│    - Optimization-based: gradient adaptation            │
│    - Model-based: memory access                         │
├─────────────────────────────────────────────────────────┤
│  Stage 3: Inference                                     │
│    - Query set 적용 및 예측                             │
│    - Test-time adaptation 또는 transductive inference   │
└─────────────────────────────────────────────────────────┘
```

## 2. 방법론 계열 분류

## 2.1 Metric-Based Methodology (도표 기반 접근)

클래스를 embedding 공간에서의 prototype 또는 support sample과의 거리로 표현하는 계열.

| 방법론 계열               | 논문명                                                                        | 핵심 특징                                                                                           |
| ------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Metric-Based Learning** | Few-Shot Learning with Geometric Constraints (2020)                           | Feature/weight l2 normalization, cosine similarity 기반 prototype 분류, base-novel coexistence 구조 |
|                           | Learning from Few Examples (2022)                                             | Embedding 함수 $g$, squared Euclidean distance, nearest centroid                                    |
|                           | Concept Learners for Few-Shot Learning (2020)                                 | Concept별 분리된 embedding space, concept mask로 masking, Euclidean distance metric                 |
|                           | On Episodes, Prototypical Networks, and Few-shot Learning (2020)              | Prototype 계산 $c_k = \frac{1}{ / S_k / }\sum f_\theta(s_i)$, nearest centroid                      |
|                           | Task Discrepancy Maximization for Fine-grained Few-Shot Classification (2022) | SAM/QAM 모듈로 채널별 가중치 동적 계산, cosine similarity                                           |
|                           | Few-Shot Object Detection (2021)                                              | RepMet(prototype-based), BYOL style cosine similarity maximization                                  |

### 공통 특징

- **구조적 특징**: Support set의 embedding을 평균화하거나 개별 sample로 prototype 생성 → Query embedding과 거리 계산
- **학습 방식**: Episodic training 또는 non-episodic 배치 학습
- **추론 방식**: Cosine similarity, Euclidean distance, log-probability 기반 분류
- **장점**: 구현 간단, domain shift에 강건, 계산 효율적

## 2.2 Optimization-Based Methodology (최적화 기반 접근)

MAML을 중심으로 한 second-order meta-learning 및 gradient adaptation 계열.

| 방법론 계열                     | 논문명                                                           | 핵심 특징                                                                       |
| ------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Optimization-Based Learning** | Learning from Few Examples (2022)                                | MAML-style adaptation, inner/outer loop 최적화                                  |
|                                 | Meta-learning approaches for few-shot learning (2023)            | Inner loop(task-specific), outer loop(meta-initialization)                      |
|                                 | Meta-SGD: Learning to Learn Quickly for Few-Shot Learning (2017) | Element-wise update vector $\alpha$, $\theta' = \theta - \alpha \circ \nabla L$ |
|                                 | Revisiting Fine-tuning for Few-shot Learning (2019)              | Full network 또는 selective fine-tuning, gradient adaptation                    |
|                                 | Meta-learning for Semi-Supervised Few-Shot Classification (2018) | Unlabeled data를 prototype refinement에 활용, episodic meta-training            |
|                                 | Self-Promoted Supervision for Few-Shot Transformer (2022)        | Teacher-pseudo label + fine-tuning                                              |
|                                 | Learning Generative Models across Incomparable Spaces (2019)     | Minimax objective, alternating generator/adversary update                       |

### 공통 특징

- **구조적 특징**: Inner-loop adaptation + Outer-loop meta-update 의双层 최적화
- **학습 방식**: Second-order derivative 활용 ($g_{\text{MAML}} = g_2 - \alpha H_2 g_1 - \alpha H_1 g_2 + O(\alpha^2)$), Adam optimizer
- **추론 방식**: Adaptation 단계에서 몇 번의 gradient step 수행
- **장점**: Task-specific 초기화 학습, unseen task에 적응力强

## 2.3 Model-Based Methodology (모델 기반 접근)

외부 메모리 구조나 메모리 접근 메커니즘을 활용한 계열.

| 방법론 계열               | 논문명                                                                        | 핵심 특징                                                                 |
| ------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Memory-Based Learning** | Learning from Few Examples (2022)                                             | External memory, content-based addressing                                 |
|                           | Meta-learning approaches for few-shot learning (2023)                         | MANN, SNAIL, CNPs (memory-based subcategory)                              |
|                           | Language Models as Few-Shot Learner for Task-Oriented Dialogue Systems (2020) | In-context learning, zero-shot priming                                    |
|                           | An Overview of Deep Learning Architectures (2020)                             | NTM, MANN(controller + external memory), Meta Networks(fast/slow weights) |

### 공통 특징

- **구조적 특징**: External memory 모듈, attention 메커니즘, fast/slow weights
- **학습 방식**: Memory access 패턴 학습, sequence-to-sequence generation
- **추론 방식**: Attention weight를 통한 memory 읽기, top-K similarity
- **장점**: Few-shot 데이터에서도 semantic knowledge 활용, zero-shot 가능

## 2.4 Data Augmentation Methodology (데이터 증강 기반 접근)

인공적 증강 또는 self-supervised 학습을 통한 계열.

| 방법론 계열                    | 논문명                                                                            | 핵심 특징                                                          |
| ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Data Augmentation Learning** | Learning from Few Examples (2022)                                                 | Siamese Networks(contrastive loss), hand-crafted augmentation 규칙 |
|                                | An Overview of Deep Learning Architectures (2020)                                 | Siamese 구조, contrastive loss                                     |
|                                | Self-Supervised Learning For Few-Shot Image Classification (2019)                 | AMDIM(Augmented Multiscale Deep InfoMax), MI maximization          |
|                                | Self-Supervised Prototypical Transfer Learning for Few-Shot Classification (2020) | ProtoCLR(prototype contrastive loss), large batch 사용             |
|                                | When Does Self-supervision Improve Few-shot Learning? (2019)                      | Jigsaw/rotation auxiliary task, representation regularizer         |
|                                | Self-Promoted Supervision for Few-Shot Transformer (2022)                         | Patch-level pseudo label, BGF/SCA 보정                             |

### 공통 특징

- **구조적 특징**: Augmented view 생성, Siamese 구조, contrastive loss
- **학습 방식**: NCE 기반, InfoNCE, Mutual information maximization
- **추론 방식**: Pre-trained representation 활용, prototype-initialized classifier
- **장점**: Label 없이도 representation 학습, overfitting 감소

## 2.5 Transductive Inference Methodology (추론 중심 접근)

Meta-learning 없이 test-time inference만으로 성능을 달성하는 계열.

| 방법론 계열                | 논문명                                             | 핵심 특징                                                                  |
| -------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------- |
| **Transductive Inference** | Few-Shot Segmentation Without Meta-Learning (2020) | RePRI, KL divergence regularizer, region proportion estimation, 3항목 loss |
|                            | Learning from Few Examples (2022)                  | Non-meta-learning 기반 접근 분류                                           |
|                            | Language Models as Few-Shot Learner (2020)         | Zero-shot priming, no gradient updates                                     |

### 공통 특징

- **구조적 특징**: Base training → Transductive inference
- **학습 방식**: Query pixel 분포 actively 활용, gradient descent로 classifier 업데이트
- **추론 방식**: Test-time optimization, temperature scaling
- **장점**: Meta-learning 복잡도 제거, 구현 단순, 적은 compute

## 2.6 Other Innovative Methodologies (기타 혁신적 접근)

기존 범주에 들어가지 않는 독특한 방법론 계열.

| 방법론 계열             | 논문명                                                                                   | 핵심 특징                                                                             |
| ----------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Program Translation** | FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot KBQA (2023)                      | KB에서 executable program 자동 샘플링, LLM 번역, execution-guided self-training       |
| **Soft Subnetwork**     | On the Soft-Subnetwork for Few-shot Class Incremental Learning (2022)                    | Major/minor subnetwork, soft mask ($m \in [0,1]$), base session → incremental session |
| **Gromov-Wasserstein**  | Learning Generative Models across Incomparable Spaces (2019)                             | GW distance, pairwise distance 보존, entropy/normalization regularization             |
| **Evolutionary Search** | Partial Is Better Than All: Revisiting Fine-tuning Strategy for Few-Shot Learning (2021) | Genetic algorithm로 층별 fine-tuning 전략 탐색                                        |

### 공통 특징

- **구조적 특징**: Domain-specific 설계 (KBQA, FSCIL, GAN)
- **학습 방식**: Self-training loop, evolutionary optimization
- **추론 방식**: Execution-guided, soft inference, structure-style disentangled

## 3. 핵심 설계 패턴 분석

## 3.1 구조적 설계 패턴

### Pattern A: Base-Novel Separation (기초-신규 분리 구조)

```text
┌────────────────────────────────────────┐
│         Base-Novel Separation          │
├────────────────────────────────────────┤
│  Base Class → Pre-training             │
│       ↓                                │
│  Novel Class → Few-shot Adaptation     │
│       ↓                                │
│  Test: Base + Novel Score Aggregation  │
└────────────────────────────────────────┘
```

**적용 논문**:

- Few-Shot Learning with Geometric Constraints (2020)
- Revisiting Fine-tuning for Few-shot Learning (2019)
- Partial Is Better Than All (2021)

### Pattern B: Two-Stage Paradigm (두 단계 학습)

```text
┌───────────────────────────────────────────────┐
│        Two-Stage Learning Paradigm            │
├───────────────────────────────────────────────┤
│  Stage A: Self-Supervised Pre-training        │
│         (Large backbone, unlabeled data)      │
│       ↓                                       │
│  Stage B: Episodic Meta-learning              │
│         (Few-shot, prototype initialization)  │
└───────────────────────────────────────────────┘
```

**적용 논문**:

- Self-Supervised Learning For Few-Shot Image Classification (2019)
- Self-Supervised Prototypical Transfer Learning (2020)
- Few-Shot Object Detection (2021)

### Pattern C: Teacher-Student Framework (교사-학생 프레임워크)

```text
┌─────────────────────────────────────────────┐
│       Teacher-Student Self-Training         │
├─────────────────────────────────────────────┤
│  Teacher Model → Synthetic Data Generation  │
│       ↓                                     │
│  Student Model → Fine-tuning                │
│       ↓                                     │
│  Distribution Alignment → Iterative Update  │
└─────────────────────────────────────────────┘
```

**적용 논문**:

- Self-Promoted Supervision for Few-Shot Transformer (2022)
- FlexKBQA (2023)
- When Does Self-supervision Improve Few-shot Learning? (2019)

### Pattern D: Multi-Dimension Learning (다차원 학습)

```text
┌────────────────────────────────────────────┐
│     Multi-Dimension Concept Learning       │
├────────────────────────────────────────────┤
│  Concept Mask → Multiple Concept Learners  │
│       ↓                                    │
│  Concept-Specific Embedding → Combination  │
│       ↓                                    │
│  Prototype-Based Classification            │
└────────────────────────────────────────────┘
```

**적용 논문**:

- Concept Learners for Few-Shot Learning (2020)

### Pattern E: Task-Adaptive Mixing (태스크 적응적 혼합)

```text
┌────────────────────────────────────────────────────┐
│        Task-Adaptive Representation Mixing         │
├────────────────────────────────────────────────────┤
│  Base Feature (Pretrained) + Novel Feature (Meta)  │
│       ↓                                            │
│  MergeNet (Mixture Weight) → Combined Feature      │
│       ↓                                            │
│  TconNet (Task-Specific Classifier)                │
└────────────────────────────────────────────────────┘
```

**적용 논문**:

- XtarNet (2020)

## 3.2 학습 방식 패턴

### Loss Function Patterns

| Loss 유형               | 적용 논문                                                        | 목적                                     |
| ----------------------- | ---------------------------------------------------------------- | ---------------------------------------- |
| Cross-Entropy           | Almost all classification papers                                 | Standard classification loss             |
| Contrastive Loss        | Siamese, AMDIM, ProtoCLR                                         | Embedding space alignment                |
| KL Divergence           | RePRI, Few-Shot Segmentation                                     | Transductive inference regularization    |
| InfoNCE                 | Self-Supervised Learning (2019)                                  | Mutual information maximization          |
| GW Loss                 | Learning Generative Models across Incomparable Spaces (2019)     | Pairwise distance structure preservation |
| Negative Log-Likelihood | Meta-learning for Semi-Supervised Few-Shot Classification (2018) | Prototype refinement                     |

### Optimization Strategy Patterns

| 전략                   | 적용 논문                         | 특징                                           |
| ---------------------- | --------------------------------- | ---------------------------------------------- |
| Inner/Outer Loop       | MAML, Meta-SGD, MetaNet           | Second-order meta-update                       |
| Element-wise Update    | Meta-SGD (2017)                   | $\alpha \circ \nabla L$로 방향/크기 제어       |
| Selective Fine-tuning  | Partial Is Better Than All (2021) | Evolutionary search로 층별 전략 탐색           |
| Test-Time Optimization | RePRI, Transductive Inference     | Query pixel 분포 활용                          |
| One-Class Adaptation   | OC-MAML                           | Inner-loop one-class only, outer-loop balanced |

### Regularization Patterns

| Regularization        | 적용 논문                                           | 목적                          |
| --------------------- | --------------------------------------------------- | ----------------------------- |
| Weight Imprinting     | Revisiting Fine-tuning (2019)                       | Classifier 초기화             |
| Geometric Constraints | Few-Shot Learning with Geometric Constraints (2020) | WCFC, AWS로 class 응집도 향상 |
| Entropy Minimization  | RePRI                                               | Distribution sharpening       |
| Normalization         | Almost all metric-based                             | Feature/weight 스케일 안정화  |

## 4. 방법론 비교 분석

## 4.1 계열 간 차이점

```text
┌────────────────────────────────────────────────────────┐
│           방법론 계열 비교 (Trade-off 분석)            │
├────────────────────────────────────────────────────────┤
│                                                        │
│  [Metric-Based]                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Pros: 구현 간단, 계산 효율적, domain shift 강건  │  │
│  │ Cons: 고차원 embedding space 학습 한계           │  │
│  │ 복잡도: O(N·K)                                   │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  [Optimization-Based]                                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Pros: Unseen task에 적응力强, fast/slow weight   │  │
│  │ Cons: Second-order derivative 계산 비용 큼       │  │
│  │ 복잡도: O(N·K·iterations)                        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  [Model-Based]                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Pros: Zero-shot 가능, semantic knowledge 활용    │  │
│  │ Cons: 메모리 구조 설계 복잡, inference 느림      │  │
│  │ 복잡도: O(|Memory|·|Query|)                      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  [Data Augmentation]                                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Pros: Overfitting 감소, representation 질 향상   │  │
│  │ Cons: Large batch 필요, domain similarity 가정   │  │
│  │ 복잡도: O(batch_size)                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  [Transductive Inference]                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Pros: Meta-learning 없음, 구현 간단, query-aware │  │
│  │ Cons: Test-time computation, 1-way 설정 제한     │  │
│  │ 복잡도: O(iterations)                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  [Innovative]                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Pros: Domain-specific 최적화                     │  │
│  │ Cons: 일반화 어려움, KB/program 등 특수한 가정   │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## 4.2 구조적/모델적 차이점

| 비교 관점     | Metric-Based          | Optimization-Based  | Model-Based      | Data Augmentation   |
| ------------- | --------------------- | ------------------- | ---------------- | ------------------- |
| **구조**      | Prototype embedding   | 双层 최적화 구조    | External memory  | Siamese/Contrastive |
| **학습 단계** | Prototype 계산 + 거리 | Inner/outer loop    | Memory access    | Augmentation + loss |
| **적응 방식** | Distance 기반         | Gradient adaptation | Attention/memory | Representation      |
| **추론**      | Cosine/Euclidean      | Gradient steps      | Memory access    | Pre-trained         |

## 4.3 적용 대상 차이점

| 접근법                 | 적용 대상                   | 데이터 요구               |
| ---------------------- | --------------------------- | ------------------------- |
| Metric-Based           | General FSL, FSOD           | Base class 데이터 필수    |
| Optimization-Based     | General FSL, one-class      | Meta-train 데이터 필수    |
| Model-Based            | Sequence modeling, dialogue | Context/pre-trained model |
| Data Augmentation      | SSL-friendly task           | Unlabeled data 필요       |
| Transductive Inference | Segmentation                | Query image 필수          |

## 4.4 복잡도 및 확장성

| 접근법                 | 계산 복잡도   | 메모리    | 데이터 효율성                     |
| ---------------------- | ------------- | --------- | --------------------------------- |
| Metric-Based           | O(N·K)        | O(C·d)    | O(base_data)                      |
| Optimization-Based     | O(N·K·I)      | O(\theta) | O(meta_data)                      |
| Model-Based            | O(            | Mem       | · / Q / ) / O( / Mem / ) / O(ctx) |
| Data Augmentation      | O(batch)      | O(ndf)    | O(unlabeled)                      |
| Transductive Inference | O(iterations) | O(d)      | O(base)                           |

## 5. 방법론 흐름 및 진화

## 5.1 초기 접근 (2017-2019)

```text
┌─────────────────────────────────────────────────────────┐
│            초기 Few-Shot Learning 접근 방식             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Meta-SGD (2017)                                        │
│  └─ First-order meta-learning (initialization, update)  │
│                                                         │
│  Revisiting Fine-tuning (2019)                          │
│  └─ Simple baseline 재평가, weight imprinting           │
│                                                         │
│  A Closer Look (2019)                                   │
│  └─ Baseline++(cosine similarity), domain shift 평가    │
│                                                         │
│  Self-Supervised Learning (2019)                        │
│  └─ AMDIM로 SSL pretraining 도입                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**특징**:

- 단순 fine-tuning 의 재평가
- Meta-learning 기본 개념 정립
- SSL 초기 도입

## 5.2 발전된 구조 (2020)

```text
┌────────────────────────────────────────────────────────────┐
│        발전된 Few-Shot Learning 구조적 다양성              │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Metric-Based 정립                                         │
│  └─ Concept Learners, Few-Shot with Geometric Constraints  │
│                                                            │
│  Optimization-Based 확장                                   │
│  └─ OC-MAML, Meta-learning for Semi-Supervised             │
│                                                            │
│  Transductive Inference 발견                               │
│  └─ RePRI (few-shot segmentation without meta-learning)    │
│                                                            │
│  Multi-Dimension 도입                                      │
│  └─ Concept mask, task-adaptive mixing                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**특징**:

- Metric/optimization 기반 방법론 성숙
- Transductive inference로 meta-learning 없이도 성능 달성
- Multi-dimensional representation (concept, task-adaptive) 등장

## 5.3 최근 경향 (2021-2023)

```text
┌───────────────────────────────────────────────────────────────┐
│              최근 Few-Shot Learning 방법론 진화               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Self-Supervised 심화                                         │
│  └─ Self-Promoted Supervision, Patch-level pseudo label       │
│                                                               │
│  Knowledge Graph Integration                                  │
│  └─ FlexKBQA, ZSL/FSL with KG                                 │
│                                                               │
│  Transformer Integration                                      │
│  └─ Vision Transformer few-shot 개선                          │
│                                                               │
│  Incremental Learning                                         │
│  └─ SoftNet (soft subnetwork), XtarNet (TAR)                  │
│                                                               │
│  Program Translation                                          │
│  └─ KB에서 program 자동 생성, execution-guided self-training  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**특징**:

- Self-supervised 학습 심화 (patch-level, location-specific)
- Knowledge Graph 및 KG embedding 활용
- Transformer 아키텍처 few-shot 환경 적용
- Incremental learning (forgetting 방지) 연구 활성화
- LLM 을 활용한 program translation 접근

## 6. 종합 정리

## 6.1 방법론 지형도

Few-shot learning 방법론은 **세 가지 축**으로 구조화된다:

```text
                │
  ┌─────────────┼─────────────┐
  │             │             │
DATA          MODEL         OPTIMIZATION
Augmentation  Memory        Gradient Adaptation
  │             │             │
  └─────────────┼─────────────┘
                │
  ┌─────────────┼─────────────┐
  │             │             │
INFER-        TRANS-        KNOW-
ENCE          DUCTIVE       LEDGE
  │             │             │
  └─────────────┴─────────────┘
```

## 6.2 방법론 계보

```text
                    ┌──────────────────────┐
                    │  Meta-Learning       │
                    │  (Inner/Outer Loop)  │
                    └──────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────┴─────────┐  ┌────────┴─────────┐  ┌────────┴─────────┐
│ Metric-Based     │  │ Optimization     │  │ Memory-Based     │
│ (Embedding)      │  │ (Gradient)       │  │ (External Memory)│
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ • ProtoNet       │  │ • MAML           │  │ • MANN           │
│ • MatchingNet    │  │ • Meta-SGD       │  │ • SNAIL          │
│ • RelationNet    │  │ • MetaNet        │  │ • CNPs           │
│ • Concept Learner│  │ • OC-MAML        │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
        ┌──────────────────────┼─────────────────────┐
        │                      │                     │
┌───────┴──────┐        ┌──────┴──────┐    ┌─────────┴─────────┐
│ Data         │        │ Self-       │    │ Transductive      │
│ Augmentation │        │ Supervised  │    │ Inference         │
├──────────────┤        └─────────────┘    ├───────────────────┤
│  • Siamese   │                           │  • RePRI          │
│  • AMDIM     │                           │  • Few-Shot Seg.  │
│  • ProtoCLR  │                           │  • ViT patch      │
└──────────────┘                           └───────────────────┘
        │                                           │
        └───────────────────────────────────────────┘
                              │
                    ┌─────────┴───────────┐
                    │ Knowledge Graph     │
                    │ Program Translation │
                    │ Incremental         │
                    │ Soft Subnetwork     │
                    └─────────────────────┘
```

## 6.3 방법론 요약

Few-shot learning 방법론은 크게 **데이터 증강**,**메트릭 기반**,**최적화 기반**,**모델 기반**의 네 가지 축으로 나뉜다. 데이터 증강 방법은 unlabeled 데이터로 representation 학습, 메트릭 기반 방법은 embedding space 의 prototype/distance로 클래스 표현, 최적화 기반 방법은 second-order meta-learning 으로 adaptation 학습, 모델 기반 방법은 external memory 및 sequence modeling을 활용한다. Transductive inference는 meta-learning 없이 test-time inference만으로 성능을 달성하는 접근이며, 최근에는 Knowledge Graph, Program Translation, Vision Transformer, Incremental Learning 등 다양한 응용 분야로 확장되고 있다.

## 3 장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 1.1 주요 데이터셋 유형

| 데이터셋 유형        | 대표 데이터셋                                                  | 사용 목적                              |
| -------------------- | -------------------------------------------------------------- | -------------------------------------- |
| 이미지 분류 벤치마크 | mini-ImageNet, CUB-200-2011, Omniglot, CIFAR-FS                | Few-shot image classification 평가     |
| 세분화 벤치마크      | PASCAL-5<sup>i</sup>, COCO-20<sup>i</sup>, Pascal VOC, MS COCO | Few-shot segmentation/detection 평가   |
| 도메인 적응 벤치마크 | CDFSL (mini-ImageNet → CropDiseases 등)                        | Cross-domain/generalization 평가       |
| KG 기반 벤치마크     | GrailQA, WebQSP, KQA Pro, FB15k-237-OWE                        | KBQA 및 KG completion 평가             |
| 세분화 벤치마크      | PASCAL VOC, MS COCO, LVIS v0.5, BSCD-FSL                       | Object detection/instance segmentation |

#### 1.2 평가 환경 유형

| 환경 유형             | 대표 사례                                | 특징                               |
| --------------------- | ---------------------------------------- | ---------------------------------- |
| 표준 벤치마크         | mini-ImageNet, Omniglot, tieredImageNet  | 5-way K-shot 설정 (K=1/5/10/20/30) |
| Cross-domain 설정     | mini-ImageNet → CUB/CropDiseases/EuroSAT | 도메인 이동성 평가                 |
| Domain shift 설정     | COCO 학습 → PASCAL 테스트                | 일반화 강건성 평가                 |
| Semi-supervised 설정  | Labeled 10%/Unlabeled 90%                | Unlabeled data 활용 효과 평가      |
| Non-transductive 설정 | BN 통계 개별 계산                        | Transductive inference 비교        |
| Incremental 설정      | Base+Novel, Multi-session                | Forgetting 과 Novel 학습 균형 평가 |

#### 1.3 주요 비교 대상 (Baseline)

| Baseline 유형      | 대표 방법                          | 역할                               |
| ------------------ | ---------------------------------- | ---------------------------------- |
| Metric-based       | MatchingNet, ProtoNet, RelationNet | 거리 기반 metric 공간 구성         |
| Optimization-based | MAML, Meta-SGD, Reptile            | Second-order meta-update           |
| Fine-tuning        | Baseline++, Fixed, Full fine-tune  | Supervised pretraining + fine-tune |
| Self-supervised    | SSL pretraining, ProtoCLR          | Representation robustness          |
| Transductive       | RePRI, ReSim                       | Test-time adaptation               |
| Knowledge-based    | Pangu, LLM-ICL                     | KG/LLM 기반 zero-shot/few-shot     |

#### 1.4 평가 지표 정리

| 지표         | 사용 태스크           | 계산 방식                        |
| ------------ | --------------------- | -------------------------------- |
| Accuracy (%) | Image classification  | 600 episode 평균 ± std, 95% CI   |
| mIoU         | Segmentation          | Classwise IoU 평균 (5 runs 평균) |
| AP/nAP       | Object detection      | COCO-style AP@IoU0.5~0.95        |
| mAP50        | Instance segmentation | 1-shot mAP50                     |
| EM/F1        | KBQA                  | Exact Match, F1 score            |
| BLEU         | NLG                   | Prefix design BLEU               |
| Recall@K     | Interpretability      | Concept recall@20                |

### 2. 주요 실험 결과 정렬

#### 2.1 Few-shot Classification 표준 벤치마크 비교 (5-way 1-shot/5-shot)

| 논문                            | 데이터셋                        | Baseline        | Proposed               | 개선 결과                                        |
| ------------------------------- | ------------------------------- | --------------- | ---------------------- | ------------------------------------------------ |
| Baseline++ (2019)               | mini-ImageNet 5-shot            | Baseline        | Baseline++             | 66.43±0.63 (MatchingNet: 63.48, ProtoNet: 64.24) |
| TDM (2022)                      | CUB cropped 1-shot              | ProtoNet 62.90% | TDM 69.94%             | +7.0 pp                                          |
| Concept Learners (2020)         | CUB-200 1-shot                  | ProtoNet        | COMET                  | 67.9% (+9.5 pp vs strongest baseline)            |
| SSL (2019)                      | mini-ImageNet 1-shot            | ProtoNet        | Mini80-SSL             | 64.03% vs 43.92% (+20.11 pp)                     |
| Self-Promoted (2022)            | tieredImageNet 1-shot           | NesT 72.93%     | Visformer 72.99%       | +0.06%                                           |
| Self-Promoted (2022)            | CIFAR-FS 1-shot                 | Meta-Baseline   | NesT+local supervision | 78.17% (+2.5% vs SoTA)                           |
| SimpleShot (2022)               | mini-ImageNet 5-shot            | MAML 63.15%     | SimpleShot             | 81.5% (+8.35 pp)                                 |
| Self-Supervised Transfer (2020) | mini-ImageNet 5-shot            | Pre+Linear      | ProtoTransfer          | 77.22% vs 74.31% (+2.91 pp)                      |
| SimpleShot (2022)               | Omniglot 5-shot                 | Baseline 55.31% | SimpleShot             | 81.5%                                            |
| Revisiting FT (2019)            | mini-ImageNet 5-shot (high-res) | ProtoNet 68.20% | VGG-16+Normalized All  | 74.50% (+6.3 pp)                                 |

#### 2.2 Cross-domain 성능 비교

| 논문                   | Setting                           | Domain               | Proposed | Baseline          | 개선             |
| ---------------------- | --------------------------------- | -------------------- | -------- | ----------------- | ---------------- |
| Closer Look (2019)     | mini-ImageNet→CUB 5-shot          | Baseline             | Baseline | 65.57             | Baseline++ 62.02 |
| Revisiting FT (2019)   | High-res cross-domain             | ResNet-50+Normalized | 74.88    | MTL (61.2)        | +13.68           |
| Self-Supervised (2019) | mini-ImageNet→CropDiseases 5-shot | Mini80-SSL           | 97.38%   | Supervised 67.13% | +30.25           |
| Self-Supervised (2019) | →EuroSAT 5-shot                   | Mini80-SSL           | 90.43%   | Supervised 67.13% | +23.30           |
| Self-Supervised (2019) | →ISIC 50-shot                     | Mini80-SSL           | 66.15%   | Pre+Linear 66.48% | ≈                |
| ProtoCLR (2020)        | ChestX 5-shot                     | Mini80-SSL           | 26.71%   | Pre+Linear 25.97% | +0.74            |
| ProtoCLR (2020)        | ISIC 50-shot                      | Mini80-SSL           | 66.15%   | Pre+Linear 66.48% | ≈                |

#### 2.3 Semi-supervised Few-shot Comparison

| 논문                                 | 데이터셋               | 1-shot Supervised | Proposed            | 개선   |
| ------------------------------------ | ---------------------- | ----------------- | ------------------- | ------ |
| Meta-Learning Semi-Supervised (2018) | Omniglot 1-shot        | 94.62%            | Masked Soft k-Means | 97.30% |
| Meta-Learning Semi-Supervised (2018) | mini ImageNet 1-shot   | 43.61%            | Masked Soft k-Means | 50.41% |
| Meta-Learning Semi-Supervised (2018) | tiered ImageNet 1-shot | 46.52%            | Masked Soft k-Means | 52.39% |

#### 2.4 Self-supervised Pretraining 효과

| 논문                   | 데이터셋      | Setting | Proposed   | Baseline   | 개선      |
| ---------------------- | ------------- | ------- | ---------- | ---------- | --------- |
| Self-Supervised (2019) | mini-ImageNet | 1-shot  | Mini80-SSL | Supervised | +20.11 pp |
| Self-Supervised (2019) | mini-ImageNet | 5-shot  | Mini80-SSL | Supervised | +14.02 pp |
| Self-Supervised (2019) | CUB-200-2011  | 1-shot  | Mini80-SSL | Supervised | +26.75 pp |
| Self-Supervised (2019) | CUB-200-2011  | 5-shot  | Mini80-SSL | Supervised | +9.70 pp  |

#### 2.5 Incremental Learning 결과 비교

| 논문           | 데이터셋              | Session       | Proposed           | Baseline    | 개선                   |
| -------------- | --------------------- | ------------- | ------------------ | ----------- | ---------------------- |
| SoftNet (2022) | CIFAR-100             | Final Session | SoftNet            | HardNet/cRT | 46.63 vs 46.31/45.28   |
| SoftNet (2022) | miniImageNet          | 60th session  | SoftNet ($c=80\%$) | cRT         | 50.48 vs 44.85 (+5.63) |
| XtarNet (2020) | tieredImageNet 5-shot | Joint         | XtarNet            | AAN         | 69.58 vs 65.52 (+4.06) |

#### 2.6 Object Detection/Segmentation Few-shot 결과

| 논문             | 데이터셋             | Setting | Proposed             | Baseline | 개선          |
| ---------------- | -------------------- | ------- | -------------------- | -------- | ------------- |
| Meta-DETR (2021) | MS COCO              | 30-shot | DETReg               | —        | 30.0 nAP      |
| Meta-DETR (2021) | MS COCO              | 30-shot | PASCAL VOC unlabeled | —        | 45.5 AP       |
| HSNet (2022)     | COCO                 | 1-shot  | HSNet                | —        | 66.2 mean IoU |
| HSNet (2022)     | COCO                 | 5-shot  | HSNet                | —        | 70.4 mean IoU |
| RePRI (2020)     | PASCAL-5<sup>i</sup> | 1-shot  | RePRI                | PFENet   | 60.8 vs 59.1  |
| RePRI (2020)     | PASCAL-5<sup>i</sup> | 5-shot  | RePRI                | PFENet   | 61.9 vs 66.8  |
| RePRI (2020)     | PASCAL-5<sup>i</sup> | 10-shot | RePRI                | PFENet   | 62.1 vs 68.2  |

#### 2.7 KBQA/Language Model Few-shot 결과

| 논문                  | 데이터셋 | Setting  | Proposed | Baseline    | 개선                    |
| --------------------- | -------- | -------- | -------- | ----------- | ----------------------- |
| FlexKBQA (2023)       | GrailQA  | 25-shot  | FlexKBQA | Pangu       | EM: 62.8 vs 56.1        |
| FlexKBQA (2023)       | WebQSP   | 100-shot | FlexKBQA | Pangu       | F1: 60.6 vs 54.5        |
| FlexKBQA (2023)       | KQA Pro  | 100-shot | FlexKBQA | LLM-ICL     | Acc: 46.83 vs 31.72     |
| GPT-2 Dialogue (2020) | MultiWoZ | 10-shot  | GPT-2 XL | Fine-tuning | NLU Acc: 73.0% (Intent) |

#### 2.8 Fine-tuning Strategy 비교

| 논문                    | 데이터셋                    | Strategy         | Proposed       | Baseline        | 개선             |
| ----------------------- | --------------------------- | ---------------- | -------------- | --------------- | ---------------- |
| Partial Transfer (2021) | mini-ImageNet 1-shot        | Full Fine-tuning | P-Transfer     | Baseline++      | 64.21% vs 64.21% |
| Partial Transfer (2021) | mini-ImageNet 1-shot        | Fixed            | P-Transfer     | Fixed           | 64.21% vs 64.21% |
| Partial Transfer (2021) | mini-ImageNet 5-shot        | Full Fine-tuning | P-Transfer     | Meta-Baseline   | +1.12%           |
| Partial Transfer (2021) | CUB 5-shot                  | Fixed            | P-Transfer     | Fixed           | +0.55%           |
| Revisiting FT (2019)    | mini-ImageNet 5-shot (high) | BN+FC            | Normalized All | ProtoNet 68.20% | 74.50% (+6.3 pp) |

### 3. 성능 패턴 및 경향 분석

#### 3.1 공통적으로 나타나는 성능 개선 패턴

| 패턴                                                   | 설명                                                           | 대표 사례                         |
| ------------------------------------------------------ | -------------------------------------------------------------- | --------------------------------- |
| SSL pretraining 이 supervised 보다 우세                | Mini80-SSL 대비 supervised +20.11 pp (1-shot)                  | Self-Supervised (2019)            |
| Deep backbone 에서 알고리즘 차이 감소                  | ResNet-34 에서 모든 방법 간 격차 축소                          | Closer Look (2019)                |
| Cross-domain 에서 fine-tuning 이 meta-learning 이 우세 | mini-ImageNet→CUB 에서 Baseline 65.57% vs meta-learning 51-62% | Closer Look (2019)                |
| Unlabeled data 활용으로 cross-domain 성능 향상         | CropDiseases 에서 +30.25 pp                                    | Self-Supervised (2019)            |
| SimpleShot 같은 단순 method 가 complex method 와 경쟁  | MAML 대비 SimpleShot +8.35 pp                                  | Learning from Few Examples (2022) |

#### 3.2 특정 조건에서만 성능이 향상되는 경우

| 조건                      | 향상되는 방법            | 설명                                              |
| ------------------------- | ------------------------ | ------------------------------------------------- |
| Fine-grained 태스크       | TDM (+7%), COMET (+9.5%) | SAM/QAM 모듈이 discriminative channel 강조        |
| Low-data regime (1-shot)  | COMET, SimpleShot        | Few-shot 환경일수록 regularizer 필요              |
| Domain shift 환경         | RePRI, P-Transfer, SSL   | Normalized classifier, SSL, partial transfer 효과 |
| Class diversity 낮을 때   | ProtoCLR                 | Class diversity 낮을 때 ProtoTransfer 강점        |
| Unlabeled data 가 있을 때 | SSL, Semi-supervised     | Unlabeled data 가 cross-domain 성능 기여          |

#### 3.3 논문 간 상충되는 결과

| 상충 관계                    | 사례                                                       | 원인                                                   |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| meta-learning vs fine-tuning | meta-learning 51-62% vs Baseline 65-80% (cross-domain)     | Domain shift 에서 meta-learning 의 generalization 한계 |
| shallow vs deep backbone     | shallow 에서 Baseline++ 강, deep 에서 알고리즘 차이 줄어듦 | Backbone depth 에 따라 알고리즘 효과 변화              |
| domain-specific vs general   | domain-specific SSL +20 pp, 일반화 시 성능 편차            | SSL 효과 domain 의존성                                 |

#### 3.4 데이터셋 또는 환경에 따른 성능 차이

| 데이터셋             | 특징             | 선호 방법                 | 평균 성능 차이   |
| -------------------- | ---------------- | ------------------------- | ---------------- |
| Omniglot             | 단순한 시각 패턴 | 모든 method 99.9% 포화    | -                |
| mini-ImageNet        | 일반 이미지      | ProtoNet, SimpleShot, SSL | SSL +14~20 pp    |
| CUB-200              | Fine-grained     | COMET, TDM, ProtoNet+SAM  | COMET +9.5 pp    |
| CIFAR-FS             | Small imagenet   | Meta-SGD, ProtoNet        | -                |
| PASCAL-5<sup>i</sup> | Segmentation     | RePRI, Transductive       | RePRI +5-10 mIoU |

#### 3.5 평가 지표별 성능 편차

| 지표                             | 편차 원인                 | 영향                                      |
| -------------------------------- | ------------------------- | ----------------------------------------- |
| 1-shot vs 5-shot                 | Support set 다양성        | 1-shot 에서 simple method 더 효과적       |
| low-res vs high-res              | Feature representation    | High-res 에서 fine-tuning 더 강력         |
| single vs cross-domain           | Distribution gap          | Cross-domain 에서 meta-learning 한계      |
| transductive vs non-transductive | Inference-time adaptation | Transductive 가 shot 수 증가 시 이점 확대 |

### 4. 추가 실험 및 검증 패턴

#### 4.1 Ablation study 패턴

| 연구 유형       | 사례                                        | 검증 목표                 |
| --------------- | ------------------------------------------- | ------------------------- |
| Module ablation | TDM: SAM vs QAM vs Both                     | 모듈별 기여도             |
| Ablation        | Self-Promoted: teacher only vs +supervision | Local supervision 효과    |
| Ablation        | RePRI: CE → CE+H → CE+H+KL                  | Regularizer 단계별 효과   |
| Ablation        | Revisiting FT: All vs BN+FC vs FC           | Parameter update 전략     |
| Ablation        | P-Transfer: Fixed vs P-Transfer             | Partial vs Full fine-tune |

#### 4.2 민감도 분석 패턴

| 분석 대상         | 사례                          | 결과                     |
| ----------------- | ----------------------------- | ------------------------ |
| Hyperparameter    | P-Transfer: P, I, lr 후보     | Evolutionary search 필요 |
| Batch size        | ProtoCLR: 5 → 50              | +5.36 pp (1-shot)        |
| Concept number    | COMET: 1-1500 개              | 일관된 성능 향상         |
| Dropout rate      | WCFC: Dropout+Strong backbone | 효과 최대화              |
| Teacher drop rate | Self-Promoted: pdpr=0.5       | 최적                     |

#### 4.3 조건 변화 실험 패턴

| 변화 유형          | 사례                                     | 목적                     |
| ------------------ | ---------------------------------------- | ------------------------ |
| Resolution 변화    | Revisiting FT: 84x84 vs 224x224          | High-res 의 이점         |
| Domain 변화        | Self-Supervised: mini-ImageNet→각 도메인 | Cross-domain 강건성      |
| Labeled ratio 변화 | Meta-Learning SS: 10%/90% vs 40%/60%     | Unlabeled data 비율 영향 |
| Split 변화         | P-Transfer: cross-domain split           | Domain gap 영향          |
| Shot 수 변화       | RePRI: 1-shot→10-shot                    | Shot 수 의존성           |

#### 4.4 Validation/Confidence Interval 사용

| 연구                 | CI 사용 | 목적                  |
| -------------------- | ------- | --------------------- |
| Closer Look (2019)   | ± std   | 성능 변동성 측정      |
| Revisiting FT (2019) | 95% CI  | 통계적 유의성         |
| Self-Promoted (2022) | 95% CI  | 성능 신뢰도           |
| On Episodes (2020)   | 95% CI  | Hyperparameter 민감도 |

### 5. 실험 설계의 한계 및 비교상의 주의점

#### 5.1 비교 조건의 불일치

| 불일치 유형         | 사례                                     | 영향                     |
| ------------------- | ---------------------------------------- | ------------------------ |
| Backbone 차이       | meta-learning 원 논문 각기 다른 backbone | 직접 비교 불가           |
| Split 차이          | meta-val/meta-test 분할 차이             | 결과 편차 발생           |
| Implementation      | 구현 세부사항 (optimizer, lr, etc.)      | 원 논문 보고값 비교 제한 |
| Evaluation protocol | episodic vs non-episodic                 | 성능 비교 어려움         |
| Hyperparameter      | 각 연구마다 별도 설정                    | 과적합 가능성            |

#### 5.2 데이터셋 의존성

| 의존성 유형         | 사례                               | 일반화 문제                        |
| ------------------- | ---------------------------------- | ---------------------------------- |
| mini-ImageNet 의존  | 대부분의 실험이 mini-ImageNet 중심 | 다른 이미지 domain 에서 성능 변화  |
| Fine-grained 필요성 | CUB 에서만 TDM 효과                | Coarse-grained task 에서 제한적    |
| Simple-shot 일반화  | mini-ImageNet 중심                 | 다른 벤치마크에서도 효과 확인 필요 |

#### 5.3 평가 지표의 한계

| 한계                     | 설명                                           |
| ------------------------ | ---------------------------------------------- |
| Accuracy 중심            | Cross-domain/generalization 능력 반영 안 됨    |
| 단일 지표 집중           | Detection/Segmentation 등 다양한 태스크별 지표 |
| CI 미사용                | 일부 survey 에서 CI 없이 평균 정확도만 보고    |
| Statistical significance | 일부 연구에서 t-test 등 유의성 검정 없음       |

#### 5.4 결과 해석 시 주의사항

| 주의점                  | 설명                                          |
| ----------------------- | --------------------------------------------- |
| 원 논문 보고값          | Survey 특이: 재현 실험 아님                   |
| Synthetic data bias     | FlexKBQA: synthetic 단독 EM 32.7 vs real 61.4 |
| Transductive assumption | oracle case 의존적 결과                       |
| Few-shot vs zero-shot   | Zero-shot/few-shot 설정 구분 필요             |

### 6. 결과 해석의 경향

#### 6.1 저자들의 공통 해석 경향

| 해석 경향              | 사례                   | 해석                                     |
| ---------------------- | ---------------------- | ---------------------------------------- |
| Baseline 재평가        | Revisiting FT (2019)   | Few-shot 에서 fine-tuning 이 충분히 강력 |
| SSL regularizer        | When Does SSL? (2019)  | SSL 은 auxiliary regularizer 역할        |
| Simple method 강력     | SimpleShot (2022)      | 복잡한 meta-learning 불필요              |
| Cross-domain 한계      | Closer Look (2019)     | Meta-learning domain shift 취약          |
| Unlabeled data 필요    | Self-Supervised (2019) | Cross-domain 에서 unlabeled 필요         |
| Local supervision 효과 | Self-Promoted (2022)   | ViT 의 few-shot generalization 개선      |
| Transductive inference | RePRI (2020)           | Meta-learning 없이도 경쟁력              |
| Meta-learning 한계     | Closer Look (2019)     | Meta-learning 필요성 의문 제기           |

#### 6.2 해석과 실제 관찰 결과 구분

| 연구                 | 관찰 결과                                | 저자 해석                                  | 실제 관찰 vs 해석 |
| -------------------- | ---------------------------------------- | ------------------------------------------ | ----------------- |
| Closer Look (2019)   | Baseline++ 66.43% vs MatchingNet 63.48%  | Baseline++ intra-class variation 감소 효과 | 일치              |
| Revisiting FT (2019) | Normalized All 74.50% vs ProtoNet 68.20% | Fine-tuning 이 meta-learning 보다 강력     | 일치              |
| SSL (2019)           | Mini80-SSL +20.11 pp                     | SSL robustness 향상                        | 일치              |
| Self-Promoted (2022) | +2.5% vs SoTA                            | ViT few-shot 친화적                        | 일치              |
| FlexKBQA (2023)      | Synthetic 32.7 vs Real 61.4              | Synthetic data 품질 낮음                   | 일치              |
| On Episodes (2020)   | NCA consistently 우위                    | Support/query 분리 불필요                  | 일치              |

#### 6.3 해석적 주의점

| 주의점       | 설명                                       |
| ------------ | ------------------------------------------ |
| 해석적 근거  | 이론적 분석보다 경험적 비교 위주           |
| 일반화 주장  | 특정 데이터셋에서의 성능만으로 일반화 주의 |
| 방법론 비교  | 원 논문 구현 차이로 인한 비교 한계         |
| Setting 차이 | Few-shot vs zero-shot 구분 모호            |

### 7. 종합 정리

Few-shot learning 실험 결과들은 몇 가지 일관된 패턴을 보여준다. 먼저 평가 구조는 주로 5-way K-shot(1/5/10/20/30) 설정에서 accuracy 중심이지만, 태스크별로 segmentation(mIoU), detection(AP), KBQA(EM/F1) 등 도메인 특화 지표 사용된다. 데이터셋 의존성은 명확하며, mini-ImageNet 기반 결과와 cross-domain 결과 간 격차가 크다. SSL pretraining 이 supervised 대비 일관되게 우위(평균 +14~20 pp)이고, cross-domain 환경에서는 unlabeled data 활용이 critical factor 로 작용한다. Meta-learning 이 domain shift 환경에서는 fine-tuning baseline 보다 제한적임을 여러 논문이 보여준다. SimpleShot 과 같은 단순 방법들이 복잡한 메타러닝과 경쟁력이 있음을 반복적으로 입증한다. Transductive inference 는 shot 수 증가 시 이점을 확대하며, test-time adaptation 이 학습 구조보다 중요할 수 있음을 시사한다. Fine-grained 태스크는 local descriptor/attention 기반 접근이 유리하고, coarse-grained 에서는 일반 method 가 유효하다. Partial fine-tuning 과 normalized classifier 는 cross-domain 환경에서 domain shift 적응에 효과적이다. 그러나 실험 간 비교는 backbone/split/implementation 차이로 인해 엄밀성이 제한되며, 원 논문 보고값 집합이므로 재현되지 않은 결과들이 많다. 결론적으로 few-shot 환경에서는 regularizer(SSL/Transductive) 활용이 일반화 성능에 기여하며, 단순한 baseline 도 복잡한 메타러닝 전략보다 강력할 수 있음이 종합된다.
