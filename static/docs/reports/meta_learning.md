# Meta-Learning

## 서론

### 1. 연구 배경

본 보고서는 meta-learning 분야의 핵심 논문들을 체계적으로 분석하여 연구 지형을 정리한다. 제공된 6 편의 논문은 meta-learning 방법론의 다양성과 적용 영역을 세 가지 범주로 구분한다: Survey 및 방법론 분류 연구, 특정 접근법 심화 연구, 연구 인프라 및 재현성 개선 연구이다. 각 논문은 bilevel optimization 프레임워크, black-box/metric-based/layered/Bayesian 최적화 전략, few-shot adaptation 과 hyperparameter transfer 등의 관점에서 분석된다.

### 2. 문제의식 및 분석 필요성

Meta-learning 연구는 새로운 작업 (new task) 에 대한 빠른 적응 (adaptation) 을 목표로 한다.然而, 다양한 방법론이 존재하며, 각 계열의 장단점과 적용 가능성을 명확히 구분하는 체계적 정리가 필요하다. 특히 기존 연구들이 초점을 맞춘 최적화 전략과 구조적 접근법의 차이를 체계적으로 분류하여, 연구자나 실무자가 적절한 방법론을 선택할 수 있는 기준을 제시하는 것이 본 보고서의 목적이다. 또한 재현성 문제와 인프라 부족이라는 meta-learning 연구 생태계의 병목 현상을 해결하기 위한 논의도 포함한다.

### 3. 보고서의 분석 관점

본 보고서는 meta-learning 연구 체계를 세 가지 축으로 정리한다: (1) 연구체계 분류: survey 논문과 방법론 분류 체계를 중심으로 연구 유형을 정립한다, (2) 방법론 분석: bilevel optimization 프레임워크와 최적화 전략 (black-box, metric-based, Bayesian, learned optimizer 등) 을 비교분석한다, (3) 실험결과 분석: data-중심 벤치마크, cross-domain 일반화, 성능 패턴 및 한계를 경험적으로 평가한다. 각 축은 상호보완적이며, 방법론의 이론적 기반과 실제 성능을 연결하는 통합적 시각을 제공한다.

### 4. 보고서 구성

- **1 장**: 연구체계 분류. Survey 논문, 특정 접근법 심화 연구, 연구 인프라 개선 논문을 범주화하고 분류 근거를 제시한다. 세 축 (Meta-Representation, Meta-Optimizer, Meta-Objective) 과 접근 방식 (black-box, metric-based, layered, Bayesian, bilevel optimization 등) 을 통해 방법론적 다양성을 정리한다.
- **2 장**: 방법론 분석. bilevel optimization 프레임워크와 four major families (black-box adaptation, metric-based learning, layered approach, Bayesian inference) 를 비교한다. 또한 initialization+adaptation 구조, meta-parameter 학습, task similarity modeling 등 설계 패턴과 각 계열의 장단점 및 트레이드오프를 분석한다.
- **3 장**: 실험결과 분석. few-shot 이미지 분류와 회귀 등의 주요 벤치마크에서 관찰된 성능 패턴을 정리한다. Transduction 설정의 효과, K-shot 변화에 따른 성능 변화, cross-domain 일반화의 한계 등 경험적 통찰을 제공하며, 결과 해석의 경향과 비교상의 주의점을 명시한다.

## 1장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 분석은 제공된 meta-learning 분야 논문 요약문들을 체계적으로 분류하기 위해 다음과 같은 기준과 원칙을 적용한다.

**분류 원칙:**

1. **연구 초점 중심**: 각 논문의 핵심 기여와 방법론적 초점을 기준으로 범주화
2. **단일 배치 원칙**: 논문이 여러 범주에 적합하더라도 가장 대표적인 1개 범주에만 배치
3. **연구 도구 구분**: 알고리즘 제안이 아닌 연구 인프라/도구 제공 논문을 별도로 분류
4. **Survey 논문 특화**: 방법론 분류와 체계화를 주요 목적으로 하는 survey 논문을 독립 범주로 구성

**분류 관점:**

- 연구 유형: 방법론 제안, survey/taxonomy, 연구 인프라
- 접근 방식: black-box, metric-based, layered, Bayesian, bilevel optimization 등
- 응용 목표: few-shot adaptation, hyperparameter transfer, reproducibility 개선

### 2. 연구 분류 체계

#### 2.1 Survey 및 방법론 분류 연구

이 범주는 meta-learning 의 방법론적 다양성과 분류 체계를 체계적으로 정리하는 survey 논문을 포함한다. 이러한 연구들은 meta-learning 의 연구 지형을 이해하는 기초 틀을 제공한다.

**2.1.1 방법론 분류 체계**
meta-learning 의 다양한 접근법을 체계적으로 분류하고 비교하는 연구들이다. black-box, metric-based, layered, Bayesian 등 method taxonomy 관점에서 알고리즘들을 계층화하거나, 세 축 (Meta-Representation, Meta-Optimizer, Meta-Objective) 으로 분해하는 분류 체계를 제안한다.

| 분류                         | 논문명                                                                         | 분류 근거                                                                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Survey > 방법론 분류 체계 1  | A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020) | black-box/adaptation rule 학습, metric-based/similarity metric, layered/initialization, Bayesian/posterior inference 라는 4 가지 관점에서 메타러닝 방법론을 체계화                                           |
| Survey > 세 축 분류 체계     | Meta-Learning in Neural Networks: A Survey (2020)                              | meta-learning 을 세 축 (무엇을 학습할지, 어떻게 최적화할지, 왜 학습할지) 으로 구조화하고 bilevel optimization formalism 을 제시하며 few-shot/many-shot, narrow/diverse task distribution 등 응용 영역 체계화 |
| Survey > meta-data 표현 체계 | Meta-Learning: A Survey (2018)                                                 | 과거 학습 경험을 meta-data 형태로 재사용하여 새 과제 적응 효율을 높이는 방법론 전체로 정의하며, meta-data 표현 유형과 태스크 유사성 정의 방식에 따라 방법론 계층화                                           |

#### 2.2 특정 접근법 심화 연구

이 범주는 survey 를 통한 일반적 분류를 넘어, 특정 접근법 (예: optimizer 기반, initialization 기반) 에 대한 심층 분석과 방법론적 통찰을 제공하는 논문을 포함한다.

**2.2.1 Optimizer 기반 메타러닝**
SGD 의 파라미터 업데이트 규칙 자체를 학습하여 adaptation 효율을 개선하는 optimizer-style meta-learning 연구들이다. initialization, update direction, learning rate 를 joint 로 최적화하는 joint optimization 관점에서 접근한다.

| 분류                               | 논문명                                                           | 분류 근거                                                                                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Optimizer 기반 > SGD 파라미터 학습 | Meta-SGD: Learning to Learn Quickly for Few-Shot Learning (2017) | Meta-SGD 는 initialization, update direction, learning rate 를 모두 학습하는 optimizer 기반 meta-learning 방법으로, few-shot adaptation 의 core constraint 를 직접 해결 |

#### 2.3 연구 인프라 및 재현성 개선 연구

이 범주는 새로운 알고리즘 제안을 목적으로 하지 않으며, meta-learning 연구 생태계의 인프라 구축과 재현성 문제를 해결하기 위한 도구, 라이브러리, 표준화 방안을 제공하는 논문을 포함한다.

**2.3.1 meta-learning 연구 라이브러리 및 인프라**
meta-learning 알고리즘 구현과 실험을 위한 공통 인프라를 제공하고, 연구 생산성과 재현성을 개선하는 소프트웨어 도구 개발 연구들이다.

| 분류                             | 논문명                                                   | 분류 근거                                                                                                                                                                                                   |
| -------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 인프라 > study-aiding 라이브러리 | learn2learn: A Library for Meta-Learning Research (2020) | 이 논문은 알고리즘 제안이 아닌, meta-learning 연구의 prototyping 과 reproducibility 병목을 해결하기 위한 소프트웨어 인프라로서, 공통 low-level 루틴과 standardized benchmark 를 제공하는 study-aiding paper |

### 3. 종합 정리

본 분석에 포함된 논문 6 편은 meta-learning 분야의 연구 지형을 세 가지 축으로 구성된다. 첫 번째 축은 Survey 및 방법론 분류 연구로, meta-learning 의 방법론적 다양성을 체계화하고 분류 체계를 수립하는 survey 논문 3 편이 이에 해당한다. 두 번째 축은 특정 접근법 심화 연구로, optimizer 기반 meta-learning 에 대한 심층 분석을 제공한다. 세 번째 축은 연구 인프라 및 재현성 개선 연구로, meta-learning 연구 생태계의 생산성과 재현성을 높이는 소프트웨어 라이브러리를 개발한다. 이 세 가지 범주는 서로 보완적이며, meta-learning 의 방법론적 이해, 특정 접근법에 대한 심화 연구, 연구 생태계 인프라 구축이라는 서로 다른 차원의 기여를 통해 연구 체계의 다면적 특성을 드러낸다.

## 2장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

### 1.1 기본 문제 설정

본 문헌들에 공통적으로 다루는 핵심 문제는 **new task 에 대한 빠른 적응 (adaptation)**이며, 이를 위한**meta-learning 프레임워크**를 구축하는 것为目标.

| 문제 유형                | 일반화 목적                              | 적용 맥락              |
| ------------------------ | ---------------------------------------- | ---------------------- |
| Out-of-sample adaptation | unseen task 에 대한 빠른 적응            | few-shot learning 환경 |
| Task generalization      | 다양한 task 분포 `𝒯∼p(𝒯)` 에 대한 일반화 | multi-task learning    |

### 1.2 Bilevel Optimization 프레임워크

대부분의 논문은 **bilevel optimization** 구조로 방법론을 formalize 한다.

```text
Level 1 (Outer Loop):  Meta-parameter ω* 학습
  ↓
Level 2 (Inner Loop):  Task-specific 모델 θ(i) 적응
```

- **Inner loop**: 각 task 의 support set 에서 task-specific solution 생성
- **Outer loop**: query set 성능을 기반으로 meta-parameter 업데이트
- **Target task**: meta-parameter 로 initialized 모델을 amortized inference 로 적용

### 1.3 입력 - 출력 구조

| 구성 요소 | 역할                                                                      |
| --------- | ------------------------------------------------------------------------- |
| Input     | Task distribution, support set (inner learner), query set (outer learner) |
| Output    | Meta-parameter `ω*`, task-specific parameters `θ*(i)`, adaptation 성능    |

## 2. 방법론 계열 분류

### 2.1 방법론 계열 1: Black-box Adaptation

**계열 정의**: task-specific adaptation 을 최적화 전략 (optimization strategy) 으로 간주하여 black-box 모델로 적응하는 계열

**공통 특징**:

- meta-parameter 로 initialized 모델에 대해 gradient descent 로 적응
- Second-order gradient 또는 first-order 근사 사용
- task similarity 가 전제됨

**해당 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: MAML 은 second-order gradient 기반 initialization 학습
- *Meta-SGD: Learning to Learn Quickly for Few-Shot Learning (2017)*: MAML 은 outer loop 에 second-order derivative 포함

### 2.2 방법론 계열 2: Metric-based Learning

**계열 정의**: class/proposal embedding 과 similarity metric 을 학습하여 task 간 일반화 하는 접근

**공통 특징**:

- Class centroid 기반 metric learning
- Embedding 및 similarity metric 을 명시적으로 학습
- Transductive inference 경우 support-query 그래프로 전파

**해당 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: ProtoNet 은 class centroid 기반 metric learning

### 2.3 방법론 계열 3: Layered Approach

**계열 정의**: base learner 와 meta-learner 를 역할 분리하여 계층적 구조로 구성하는 계열

**공통 특징**:

- Base learner: inner loop 에서 task-specific 모델 학습
- Meta-learner: outer loop 에서 meta-parameter 학습
- 독립 축: meta-representation, meta-optimizer, meta-objective

**해당 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: layered approach 에서 base learner 와 meta-learner 역할 분리
- *Meta-Learning in Neural Networks: A Survey (2020)*: 세 가지 독립 축 (representation, optimizer, objective) 으로 방법론 분해

### 2.4 방법론 계열 4: Bayesian Inference

**계열 정의**: meta-parameter 를 확률분포 (posterior distribution) 로 추론하는 Bayesian 접근

**공통 특징**:

- Parameter posterior inference 수행
- Laplace approximation 또는 SVGD 기반
- Uncertainty estimate 가능

**해당 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: Bayesian 은 posterior distribution inference (Laplace/SVGD)

### 2.5 방법론 계열 5: Meta-feature 기반 접근

**계열 정의**: 과제의 meta-data 를 특징화하여 과거 경험 재사용하는 접근

**공통 특징**:

- Meta-data 수집: 모델 평가값, task 메타특성, 사전학습 모델
- Meta-feature 추출, 차원 축소
- Task similarity 를 측정하여 경험 재사용

**해당 논문**:

- *Meta-Learning: A Survey (2018)*: meta-data 의 형태 (평가값, task 속성, 사전학습 모델) 에 따라 세 가지 범주

### 2.6 방법론 계열 6: Learned Optimizer

**계열 정의**: optimizer hyperparameters (update coefficients, step size 등) 를 학습 가능한 파라미터로 최적화하는 접근

**공통 특징**:

- Differentiable optimization 기반
- Learnable optimizer 파라미터
- Gradient transform 기반 adaptive update

**해당 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: Meta-SGD 는 step size까지 학습
- *Meta-Learning in Neural Networks: A Survey (2020)*: learned optimizer 의 meta-gradient 최적화
- *Meta-SGD: Learning to Learn Quickly for Few-Shot Learning (2017)*: Meta-SGD는 initialization, update direction, learning rate 를 jointly 학습

### 2.7 방법론 계열 비교汇总表

| 방법론 계열           | 논문명                              | 핵심 특징                                      |
| --------------------- | ----------------------------------- | ---------------------------------------------- |
| Black-box adaptation  | MAML (2020)                         | Second-order gradient 기반 initialization 학습 |
| Metric-based learning | ProtoNet (2020)                     | Class centroid 기반 metric learning            |
| Layered approach      | Base/Meta-learner separation (2020) | Base learner 와 meta-learner 역할 분리         |
| Bayesian inference    | BMAML (2020)                        | Particle ensemble posterior                    |
| Meta-feature 기반     | Surrogate model (2018)              | 순위 추천, performance 예측, Gaussian Process  |
| Learned optimizer     | Meta-SGD (2017)                     | Initialization + Update direction jointly 학습 |

## 3. 핵심 설계 패턴 분석

### 3.1 Pattern 1: Initialization + Adaptation 구조

| 구성 요소                       | 역할                                         |
| ------------------------------- | -------------------------------------------- |
| Initialization (meta-parameter) | Base learner 를 시작하는 초기값              |
| Adaptation (inner loop)         | Target task 에서 몇 단계 gradient descent    |
| Evaluation (outer loop)         | Validation set 에서 generalization 성능 평가 |

**적용 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: support set 에서 adaptation → query set 에서 평가
- *Meta-Learning in Neural Networks: A Survey (2020)*: Inner loop 생성 → Outer loop 업데이트 → Target task 적용

### 3.2 Pattern 2: Meta-parameter Learning

| 학습 대상           | 설명                                             |
| ------------------- | ------------------------------------------------ |
| Meta-representation | 학습할 representation 구조                       |
| Meta-optimizer      | 최적화 방법 (gradient descent, RL, evolutionary) |
| Meta-objective      | 학습 목표 (accuracy, uncertainty 등)             |

**적용 논문**:

- *Meta-Learning in Neural Networks: A Survey (2020)*: 세 가지 독립 축 (representation, optimizer, objective)

### 3.3 Pattern 3: Task Similarity Modeling

**기작**:

- Meta-train/task distribution 에서 유사한 task 샘플링
- Task similarity 가 전제되며, similarity 를 측정할 수 있는 meta-data 가 필요
- Support-query 간 관계 모델링 (transductive inference)

**적용 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: task similarity 가 가정되며, meta-train 분포에서 일반화
- *Meta-Learning: A Survey (2018)*: 과거 과제와 새 과제 사이에 유사성이 존재하며, 유사성을 측정할 수 있는 meta-data 가 수집 가능해야 효과적

### 3.4 Pattern 4: Memory and Attention Retrieval

**구성 요소**:

- Memory: 과거 task 경험 저장
- Attention: 관련 경험에 선택적 접근
- Retrieval: 가장 관련성 높은 경험 추출

**적용 논문**:

- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: memory and attention retrieval 구성 요소

### 3.5 Pattern 5: Amortized Inference

**기작**:

- Meta-parameter 로 initialized 모델을 forward pass 로 생성
- 별도의 적응 없이 prediction 수행 가능
- Computational efficient

**적용 논문**:

- *Meta-Learning in Neural Networks: A Survey (2020)*: Amortized inference 로 prediction 수행

## 4. 방법론 비교 분석

### 4.1 문제 접근 방식 차이

| 차원        | Black-box/Metric-based     | Layered/Bayesian                  |
| ----------- | -------------------------- | --------------------------------- |
| 일반화 전략 | 최적화 규칙 or metric 학습 | Explicit posterior or 계층적 구조 |
| 적응 방식   | Optimization rule          | Model initialization + adaptation |
| 불확실성    | Uncertainty estimate 없음  | Bayesian inference 로 가능        |

### 4.2 구조/모델 차이

| 구조 유형         | 모델 구성             | 계산 복잡도           |
| ----------------- | --------------------- | --------------------- |
| Black-box         | Second-order gradient | O(n²) (Hessian 기반)  |
| Metric-based      | Embedding + metric    | O(n) (linear scan)    |
| Bayesian          | Particle ensemble     | O(n) (sampling 기반)  |
| Learned optimizer | Gradient transform    | O(k·d) (k parameters) |

### 4.3 적용 대상 차이

| 적용 대상                   | 적합한 계열             |
| --------------------------- | ----------------------- |
| Classification              | Metric-based (ProtoNet) |
| Continuous control          | Black-box (MAML-style)  |
| Hyperparameter search       | Meta-feature 기반       |
| Uncertainty-aware inference | Bayesian                |

### 4.4 트레이드오프 분석

| 트레이드오프 | Black-box              | Meta-feature               |
| ------------ | ---------------------- | -------------------------- |
| 일반화 범위  | Task 분포에 국한       | Meta-data 기반 재사용 가능 |
| 계산 비용    | Second-order 계산 부담 | Meta-model 학습 비용       |
| 설명 가능성  | Low (black-box)        | Meta-feature 설명 가능     |

## 5. 방법론 흐름 및 진화

### 5.1 초기 접근 (2017)

**Meta-SGD (2017)**:

- Initialization 과 update rule 을 jointly 학습하는 첫 번째 시도
- Element-wise product 연산자 (∘) 를 사용한 파라미터별 update 스케일 학습
- Few-shot learning 설정에서 one-step adaptation 목표

```text
θ' = θ - α ∘ ∇Ltrain(θ)
α는 θ와 동일한 차원의 벡터로 각 파라미터별 update 스케일과 방향 결정
```

### 5.2 발전된 구조 (2018-2020)

**Survey papers**:

- *Meta-Learning: A Survey (2018)*: 네 가지 주요 계열 (평가값, 메타피처, 사전학습, learned optimizer) 체계화
- *Meta-Learning in Neural Networks: A Survey (2020)*: Bilevel optimization 프레임워크로 세 가지 독립 축 formalize
- *A Comprehensive Overview and Survey of Recent Advances in Meta-Learning (2020)*: Black-box, metric-based, layered, Bayesian 네 가지 범주 정리

### 5.3 인프라 발전 (2020)

**learn2learn (2020)**:

- Meta-learning 연구의 프로토타이핑 복잡성과 재현성 문제 해결
- Common 라이브러리 및 표준화된 인터페이스 제공
- TaskDataset, MetaDataset 추상화
- PyTorch 호환성 유지

## 6. 종합 정리

### 6.1 방법론 지형도

Meta-learning 방법론은 **두 가지 축**으로 분류된다:

1. **최적화 전략 축**: Black-box adaptation, Metric learning, Bayesian inference, Meta-feature 기반
2. **계층 구조 축**: Layered (base+meta-learner), Bilevel optimization

### 6.2 공통 핵심 요소

모든 방법론이 공유하는 요소:

- **Task similarity modeling**: Meta-train 분포에서 일반화
- **Adaptation mechanism**: Few-shot 데이터로 모델 조정
- **Generalization goal**: Unseen task 에 대한 out-of-sample adaptation

### 6.3 방법론 분류 요약

| 분류 기준   | 유형                                          | 논문의 관점                             |
| ----------- | --------------------------------------------- | --------------------------------------- |
| 최적화 전략 | Black-box, Metric, Bayesian                   | A Comprehensive Overview (2020)         |
| 최적화 전략 | Optimization, Meta-feature, Learned optimizer | Meta-Learning: A Survey (2018)          |
| 구조        | Bilevel, Layered                              | Meta-Learning in Neural Networks (2020) |
| 구현        | Library-based                                 | learn2learn (2020)                      |

### 6.4 진화적 흐름

```text
2017 (Meta-SGD)
    ↓
2018 (Survey: 3 meta-data 범주)
    ↓
2020 (Meta-Learning in NN: Bilevel 3 축)
    ↓
2020 (Comprehensive Survey: 4 계열 체계화 + learn2learn 라이브러리)
```

본 장에서 분석한 방법론들은 **meta-learning**이라는 공통 프레임워크 내에서, 다양한 최적화 전략과 구조적 접근을 통해**few-shot adaptation** 문제를 해결한다. 초기에는 단일 학습 규칙 (initialization) 에 집중되었으나, 점차**optimization strategy**,**representation**,**uncertainty**,**infrastructure** 등으로 확장되어 체계화되어 왔다.

## 3장. 실험결과 분석

## 1. 평가 구조 및 공통 실험 설정

### 1.1 주요 데이터셋 유형

| 데이터셋 유형                    | 구체적인 데이터셋                                                    | 용도                    |
| -------------------------------- | -------------------------------------------------------------------- | ----------------------- |
| Few-shot image classification    | miniImageNet, tieredImageNet, CIFAR-10/100, CUB-200                  | 시각적 학습 일반화 평가 |
| Few-shot classification (simple) | Omniglot, CelebA                                                     | 기본 성능 벤치마킹      |
| Few-shot vision tasks            | CIFAR-FS, FC100                                                      | 다양한 few-shot 설정    |
| Reinforcement learning           | Atari, PTB, Meta-Dataset, Sonic, CoinRun, ProcGen, Meta-World, PHYRE | meta-RL 평가            |
| Navigation control               | 2D particle navigation, robotics control                             | 제어 및 탐색 작업       |

### 1.2 평가 환경

| 환경 유형       | 구체적 환경                              | 특징                       |
| --------------- | ---------------------------------------- | -------------------------- |
| 인위적 벤치마크 | miniImageNet, Omniglot 등                | Few-shot 설정으로 표준화됨 |
| 시뮬레이션 환경 | 2D particle navigation, MetaWorld, PHYRE | 제어/탐색 작업             |
| 실환경          | robotics control                         | 실제 로봇 제어             |
| meta-test shift | 넓은 meta-test 환경                      | 일반화 능력 평가           |

### 1.3 비교 방식

| 비교 유형                 | 설명                                           |
| ------------------------- | ---------------------------------------------- |
| Baseline 비교             | random initialization, default 설정 등         |
| Cross-method 비교         | 계열 내부 다른 방법 비교 (MAML vs ProtoNet 등) |
| Cross-domain 비교         | domain adaptation 기법 평가                    |
| Transductive vs inductive | 유도 방식 비교                                 |

### 1.4 주요 평가 지표

| 지표                                       | 사용 빈도 | 측정 내용             |
| ------------------------------------------ | --------- | --------------------- |
| Classification accuracy (mean ± std/error) | 매우 높음 | 분류 정확도           |
| Few-shot accuracy                          | 높음      | few-shot 설정 정확도  |
| MSE                                        | 중간      | 회귀 성능             |
| F1                                         | 중간      | 분류 F1 점수          |
| Adaptation speed                           | 낮음      | 적응 속도             |
| Cross-domain generalization gap            | 낮음      | 도메인 간 일반화 격차 |

## 2. 주요 실험 결과 정렬

### 2.1 few-shot 이미지 분류 성능 비교 (miniImageNet)

| 논문                           | 설정         | 방법       | 정확도 | 비교 대상               |
| ------------------------------ | ------------ | ---------- | ------ | ----------------------- |
| Black-box WRN (2020)           | 5-way 1-shot | WRN-70-16  | 73.74% | -                       |
| DAPNA (2020)                   | 5-way 1-shot | DAPNA      | 83.62% | MetaOptNet-SVM-val      |
| DAPNA (2020)                   | 5-way 1-shot | DAPNA      | 84.07% | MetaOptNet-SVM-trainval |
| MetaOptNet-SVM-val (2020)      | 5-way 1-shot | MetaOptNet | -      | -                       |
| MetaOptNet-SVM-trainval (2020) | 5-way 1-shot | MetaOptNet | 80.00% | AM3+TRAML               |
| AM3+TRAML (2020)               | 5-way 1-shot | AM3+TRAML  | 79.54% | -                       |
| LEO (2020)                     | 5-way 1-shot | LEO        | 61.76% | MetaOptNet-SVM-trainval |
| WRN (2020)                     | 5-way 1-shot | WRN-16-2   | 62.71% | WRN-70-16               |
| WRN (2020)                     | 5-way 1-shot | WRN-16-2   | 75.32% | WRN-70-16               |
| LEO (2020)                     | 5-way 5-shot | LEO        | 93.28% | MetaOptNet-SVM-val      |
| LEO (2020)                     | 5-way 5-shot | LEO        | 94.67% | MetaOptNet-SVM-trainval |
| ProtoNet (2020)                | 5-way 5-shot | ProtoNet   | 91.01% | MetaOptNet-SVM-val      |
| ProtoNet (2020)                | 5-way 5-shot | ProtoNet   | 93.16% | MetaOptNet-SVM-trainval |
| MetaOptNet-SVM-val (2020)      | 5-way 5-shot | MetaOptNet | 91.27% | AM3+TRAML               |
| MetaOptNet-SVM-trainval (2020) | 5-way 5-shot | MetaOptNet | 93.07% | AM3+TRAML               |
| AM3+TRAML (2020)               | 5-way 5-shot | AM3+TRAML  | 92.20% | -                       |

### 2.2 Omniglot 5-way 1-shot 성능

| 논문            | 정확도 |
| --------------- | ------ |
| Meta-SGD (2017) | 99.53% |

### 2.3 Omniglot multi-way performance

| 논문            | 설정          | 정확도 |
| --------------- | ------------- | ------ |
| Meta-SGD (2017) | 20-way 5-shot | 98.97% |

### 2.4 MiniImagenet 5-way 1-shot 성능 차이

| 논문                 | 방법      | 정확도 | 차이 원인                    |
| -------------------- | --------- | ------ | ---------------------------- |
| Black-box WRN (2020) | WRN-70-16 | 73.74% | -                            |
| Black-box WRN (2020) | WRN-16-2  | 75.32% | -                            |
| LEO (2020)           | LEO       | 61.76% | MetaOptNet-SVM-trainval 대비 |

### 2.5 회귀 성능 비교 (MAML vs Meta-SGD)

| 설정                              | MAML | Meta-SGD | 개선 |
| --------------------------------- | ---- | -------- | ---- |
| Regression 5-shot                 | 1.13 | 0.90     | 0.23 |
| sine curve regression (K=5,10,20) | -    | -        | -    |

### 2.6 MiniImagenet 5-way 5-shot 성능 (상위)

| 순위 | 방법                          | 정확도 |
| ---- | ----------------------------- | ------ |
| 1    | LEO (MetaOptNet-SVM-trainval) | 94.67% |
| 2    | LEO (MetaOptNet-SVM-val)      | 93.28% |
| 3    | MetaOptNet-SVM-trainval       | 93.07% |
| 4    | ProtoNet                      | 93.16% |
| 5    | MetaOptNet-SVM-val            | 91.27% |

## 3. 성능 패턴 및 경향 분석

### 3.1 공통적인 성능 개선 패턴

#### 3.1.1 Feature extractor 의 영향

```text
패턴: Feature extractor 의 quality 가 성능 결정에 영향

관찰:
- Black-box WRN 비교: WRN-16-2 (75.32%) > WRN-70-16 (73.74%)
- MetaOptNet-SVM-val vs MetaOptNet-SVM-trainval: trainval 가 consistently 더 높음
- AM3+TRAML 가 AM3+val 보다 낮음 (transduction 방식 차이)
```

#### 3.1.2 Transduction 설정의 영향

```text
패턴: Train-val split 에 따라 성능 편차 발생

관찰:
- MetaOptNet-SVM-val: 91.27% → MetaOptNet-SVM-trainval: 93.07% (+1.8%)
- AM3+TRAML: 92.20% → AM3+val: - (상당 차이 존재)
- LEO: 93.28% (val) → 94.67% (trainval) (+1.39%)
```

#### 3.1.3 K-shot 설정에 따른 성능 변화

```text
패턴: Few-shot 수 증가에 따른 성능 향상

관찰:
- 1-shot: 61.76%~94.67%
- 5-shot: 91.01%~94.67%
- 1-shot → 5-shot: 대부분의 방법에서 2~5% 성능 향상
```

### 3.2 특정 조건에서만 성능이 향상되는 경우

#### 3.2.1 Task-dependent design 의 효과

```text
조건: Task-dependent 설계가 적용될 때

관찰:
- Black-box 방법은 task-dependent 설계가 없을 경우
- Convex solver 기반 방법은 uncertainty modeling 강점
- Bayesian 계열은 accuracy보다 uncertainty modeling 강점
```

#### 3.2.2 Cross-domain 일반화의 한계

```text
조건: 넓은 meta-test shift 환경

관찰:
- few-shot vision 에서 성능 입증되나 cross-domain 에서 일반화 어려움
- Meta-learning 은 narrow task family 에서 성능 입증되나 broad task distribution 에서 부족
- "보편적 learning-to-learn 시스템"은 아님
```

### 3.3 논문 간 상충되는 결과

```text
상충: Transduction vs Val 방식

관찰:
- MetaOptNet-SVM-trainval vs MetaOptNet-SVM-val: consistently trainval 높음
- AM3+TRAML vs AM3+val: 상당 차이 (비교 조건 불일치)
- 완전한 apples-to-apples 비교 아님
```

### 3.4 데이터셋에 따른 성능 차이

| 데이터셋     | 1-shot 성능 범위 | 5-shot 성능 범위 |
| ------------ | ---------------- | ---------------- |
| miniImageNet | 61.76%~94.67%    | 91.01%~94.67%    |
| Omniglot     | 98.97%~99.53%    | -                |

## 4. 추가 실험 및 검증 패턴

### 4.1 Ablation study 패턴

| 검증 유형          | 방법                  | 목적                           |
| ------------------ | --------------------- | ------------------------------ |
| Backbone 영향 분석 | WRN-16-2 vs WRN-70-16 | Feature extractor quality 영향 |
| Transduction 설정  | val vs trainval       | 데이터 분할 방식 영향          |
| Task similarity    | -                     | cold-start 완화 효과 검증      |

### 4.2 민감도 분석

```text
검증: Hyperparameter 및 설정 민감도

- One-step adaptation 만 사용 (Meta-SGD)
- Multi-step adaptation 에 의존하지 않음
- Inner/outer loop 구조 분석
```

### 4.3 조건 변화 실험

| 실험 유형    | 조건 변화                   | 비교 대상             |
| ------------ | --------------------------- | --------------------- |
| K-shot 변화  | 1-shot → 5-shot             | few-shot setting 영향 |
| N-way 변화   | 5-way → 20-way              | task complexity 영향  |
| Domain shift | narrow → broad distribution | generalization gap    |

## 5. 실험 설계의 한계 및 비교상의 주의점

### 5.1 비교 조건의 불일치

```text
문제점: 완전한 apples-to-apples 비교 어려움

구체적 예:
- feature extractor 차이
- loss function 차이
- transduction 방식 차이
- train/val split 방식 차이
- dataset 구성 차이
```

### 5.2 데이터셋 의존성

```text
문제점: narrow task family 의존성

관찰:
- few-shot vision 에서 성과 있음
- wide task distribution 에서 일반화 어려움
- cross-domain 에서 performance gap
- 계산 비용 증가
```

### 5.3 일반화 한계

```text
문제점: meta-generalization 어려움

구체적 관찰:
- meta-learning 은 diverse task distribution 처리 어려움
- computation cost 및 scalable design 과제로 남음
- "보편적 학습 시스템" 아님
```

### 5.4 평가 지표의 한계

```text
문제점: accuracy 중심 평가

구체적 제한:
- Bayesian 접근은 accuracy보다 uncertainty modeling 강점
- 계산 비용 측정 부족
- adaptation speed 측정 제한적
```

## 6. 결과 해석의 경향

### 6.1 저자들의 공통 해석 경향

#### 6.1.1 Algorithm-centric 접근의 제한

```text
해석: 특정 알고리즘보다 설계 요소가 중요

관찰:
- DAPNA가 최고(84.07%)지만 단일 방법론이 아님
- feature extractor, task-dependent design, convex solver, transduction 등 설계 요소 조합이 성능 결정
- Bayesian 접근은 accuracy보다 uncertainty modeling 강점
```

#### 6.1.2 Infrastructure 과 algorithm 분리

```text
해석: 인프라가 재현성의 핵심

관찰:
- meta-learning은 아이디어보다 software infrastructure 가 병목
- common abstraction과 standardized API 제공 필요
- prototyping 과 reproducibility 병목 해소
```

#### 6.1.3 Survey 기반의 체계적 정리

```text
해석: 경험적 통찰의 통합

관찰:
- unified 프레임워크로 구조화
- OpenML 실험 기반 통찰
- 각 방법론의 경험적 성능 경향과 condition-dependence 정리
```

### 6.2 해석과 실제 관찰 결과의 구분

```text
해석 (저자 주장):
"single method가 압도적 승자" 아님 → 실제 관찰: 설계 요소 조합이 중요

해석 (저자 주장):
"보편적 learning-to-learn 시스템" 아님 → 실제 관찰: narrow task family 에서 성과, broad 에서 한계

해석 (저자 주장):
"few-shot 에서 sample efficiency" → 실제 관찰: cross-domain 에서 generalization gap 존재
```

## 7. 종합 정리

meta-learning 방법론들의 실험 결과는 특정 알고리즘의 성능 차이보다는 설계 요소 (feature extractor, task-dependent design, transduction, convex solver) 의 조합에 크게 의존함을 보여준다. few-shot 이미지 분류에서 1-shot(61.76%~94.67%)에서 5-shot(91.01%~94.67%)으로 성능이 상승하는 패턴이 관찰되나, 이러한 성능 개선은 narrow task family 에 제한된다. cross-domain 또는 broad task distribution 환경에서는 generalization gap 이 명확히 관찰되며, meta-learning 은 여전히 scalable design 과 computation cost 문제를 해결해야 한다. MetaOptNet-SVM-trainval, DAPNA, LEO 등 train-val 설정의 효과를 보여주는 결과들은 transduction 의 중요성을 시사하지만, 비교 조건의 불일치로 인한 완전한 apples-to-apples 평가의 어려움도 존재한다. Bayesian 접근의 강점은 accuracy 향상보다 uncertainty modeling 에 있고, Meta-SGD 는 initialization, update direction, learning rate 를 동시 학습함으로써 few-shot 회귀/분류/강화학습 모두에서 one-step adaptation만으로 높은 성능을 입증한다.
