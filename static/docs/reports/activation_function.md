# Activation Function 연구 분석 보고서

## 서론

### 1. 연구 배경

Activation Function(활성화 함수) 연구는 신경망의 표현력과 학습 안정성을 결정하는 핵심 요소로, 최근 연구들은 활성화 함수의 설계, 초기화 전략, gated 구조 결합, 그리고 적응형 기능 등 다양한 차원에서 활발히 진행되고 있다. 본 보고서는 활성화 함수 관련 19 편의 논문들을 분석하여, 고정형/적응형 활성화 함수, gated 구조 도입, 초기화 전략, mean-variance dynamics 분석 등 4 개의 주요 연구 축을 체계적으로 정리한다. 이를 통해 네트워크 아키텍처, 데이터 분포, 학습 목표에 맞는 활성화 함수 선택의 기준을 제시하고자 한다.

### 2. 문제의식 및 분석 필요성

활성화 함수 연구는 초기의 ReLU, Sigmoid 계열에서 현재에 이르기까지 수학적 성질 최적화, 파라미터 학습, 자동 탐색 등 다양한 방식으로 진화해 왔다. 그러나 각 연구가 다루는 활성화 함수의 유형 (고정형/적응형), 연구 초점 (설계/초기화), 분석 차원 (이론적/경험적), 적용 대상 (CNN/RNN/Transformer) 이 서로 다른 기준에서 이루어져 있어 체계적인 비교와 통합이 필요하다. 특히 단일 "best activation"이 존재하지 않으며, 각 활성화 함수는 특정 조건 (긴 시퀀스, deep narrow 네트워크, 계산 효율성 요구) 에서만 우월한 성능을 보인다는 점이 문제이다. 따라서 연구들을 체계적으로 분류하고, 방법론적 접근 방식과 실험 결과를 통합 분석할 필요성이 있다.

### 3. 보고서의 분석 관점

본 보고서는 다음과 같은 3 개의 핵심 관점으로 문헌을 정리한다:

- **연구체계 분류**: 활성화 함수를 유연성 (고정형/적응형), 연구 초점 (설계/초기화), 분석 차원 (이론/경험), 시스템/태스크 (CNN/RNN/Transformer) 등 4 가지 기준에서 체계화한다.
- **방법론 분석**: 초기화 기반, 변수 매개변수화, Gated/Multiplicative, Self-Normalizing, 입력 적응형, 자동 탐색 등 계열별 공통 특징과 설계 패턴을 분석하고, 계열 간 차이점과 트레이드오프를 비교한다.
- **실험결과 분석**: 이미지 분류, NLP, 음성 인식 등 다양한 벤치마크에서 관찰된 성능 패턴과 데이터셋 의존성, 논문 간 상충되는 결과를 종합하여 활성화 함수 선택에 대한 실험적 증거를 정리한다.

### 4. 보고서 구성

- **1 장. 연구체계 분류**: 활성화 함수를 고정형/적응형, gated 구조, 초기화 전략 등 4 가지 기준에서 체계적으로 분류하고 각 유형별 대표 논문을 분석한다.
- **2 장. 방법론 분석**: 초기화 기반, 변수 매개변수화, gated, self-normalizing 등 계열별 공통 특징, 설계 패턴, 방법론적 구조를 상세히 분석하고 계열 간 차이와 트레이드오프를 비교한다.
- **3 장. 실험결과 분석**: 주요 데이터셋과 벤치마크에서 관찰된 성능 결과, 성능 패턴, 데이터셋 의존성, 실험 설계의 한계 등을 종합하여 활성화 함수 선택의 실험적 근거를 제시한다.

## 1 장. 연구체계 분류

### 1. 연구 분류 체계 수립 기준

본 보고서는 Activation Function 관련 19 편의 논문 요약문을 다음과 같은 4 가지 기준과 원칙으로 분류하였다.

**1) 활성화 함수의 유연성: 고정형 vs 적응형**

- 고정형 AF 는 학습 과정에서 함수 형태가 불변 (ReLU, GELU, Sigmoid)
- 적응형 AF 는 학습 가능한 파라미터로 함수 모양이 동적으로 조정 (PReLU, PELU, DY-ReLU, APL)

**2) 연구 초점: 활성화 함수 설계 vs 초기화 전략**

- 활성화 함수 설계: 함수 구조 자체의 혁신 (Swish, GLU variants, DReLU)
- 초기화 전략: gradient flow 와 variance control 관점 (IRNN, He init 관련, orthogonal init 등)

**3) 분석 차원: 이론적 분석 vs 경험적 검증**

- 이론적 분석: 수학적 성질, mean-variance dynamics, convergence proof (SELU, CELU, GELU)
- 경험적 검증: benchmark 와 performance comparison (각 AF 의 empirical 성능 비교)

**4) 시스템/태스크: 네트워크 아키텍처별 관점**

- CNN: Convolutional Neural Networks 에서의 활성화 (CIFAR, ImageNet, MNIST)
- RNN/QRNN: Recurrent 구조에서의 gradient flow 와 long-range modeling
- Transformer: FFN 과 attention 구조에서의 gating 과 활성화

### 2. 연구 분류 체계

#### 2.1 고정형 활성화 함수 연구

고정형 활성화 함수는 학습 과정에서 함수 모양이 변하지 않는 전통적 활성화 함수들을 연구한다.

| 분류                     | 논문명                                                                                                   | 분류 근거                                                                     |
| ------------------------ | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 고정형 AF > Sigmoid 계열 | Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning (2017) | dSiLU 의 sigmoid-overshoot 특성과 TD 학습 환경 적합성 분석                    |
| 고정형 AF > Sigmoid 계열 | A Study on ReLU and Softmax in Transformer (2023)                                                        | Softmax 를 key-value memory 관점에서 재해석하고 layer normalization 결합 연구 |
| 고정형 AF > ReLU 계열    | A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015)                           | ReLU 기반 IRNN 의 identity initialization 으로 gradient path 보존             |
| 고정형 AF > ReLU 계열    | Deep Learning with S-shaped Rectified Linear Activation Units (2015)                                     | SReLU 를 piecewise linear S-shaped family 로 일반화하고 backprop 학습         |
| 고정형 AF > ReLU 계열    | A Study on ReLU and Softmax in Transformer (2023)                                                        | ReLUFormer 를 Transformer 의 FFN 과 attention 에 ReLU 적용으로 제안           |
| 고정형 AF > ELU 계열     | Continuously Differentiable Exponential Linear Units (2017)                                              | CELU 를 ELU 의 C¹ 연속 매개변수화 family 로 재설계                            |
| 고정형 AF > ELU 계열     | Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs) (2015)                        | ELU 의 bias shift 완화와 exponential 음수 포화 동역학 분석                    |
| 고정형 AF > GELU 계열    | Gaussian Error Linear Units (GELUs) (2016)                                                               | GELU 를 Gaussian CDF 기대값으로 해석하고 adaptive dropout 연결                |
| 고정형 AF > GELU 계열    | GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance (2023)  | GELU 의 수학적 성질과 optimization landscape 에 대한 체계적 분석              |
| 고정형 AF > Swish 계열   | Mish: A Self Regularized Non-Monotonic Activation Function (2019)                                        | Mish 의 self-regularization 과 loss landscape 평활성 검증                     |
| 고정형 AF > Swish 계열   | Searching for Activation Functions (2017)                                                                | RNN controller 로 Swish 를 자동 발견하고 설계 원리 제시                       |

#### 2.2 적응형/학습형 활성화 함수 연구

적응형 활성화 함수는 각 층, 뉴런 또는 입력에 따라 함수 형태가 학습되어 최적화되는 구조를 연구한다.

| 분류                                 | 논문명                                                                                                                                  | 분류 근거                                                                         |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 적응형 AF > learnable slope          | Activation Functions in Artificial Neural Networks: A Systematic Overview (2020)                                                        | Leaky ReLU 와 PReLU 의 learnable slope 가 deep network 수렴에 미치는 영향         |
| 적응형 AF > learnable slope          | Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark (2021)                                                      | PReLU/GELU/Swish/Mish/PAU 등 적응형 AF 의 모달리티별 비교                         |
| 적응형 AF > learnable slope          | Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015)                                      | PReLU 를 channel-wise learnable slope 로 일반화하고 rectifier-aware init          |
| 적응형 AF > learnable slope          | Improving Deep Neural Network with Multiple Parametric Exponential Linear Units (2016)                                                  | MPELU 를 ELU/PReLU 를 포함하는 learnable α/β 두 파라미터 구조로 확장              |
| 적응형 AF > learnable slope          | Parametric Exponential Linear Unit for Deep Convolutional Neural Networks (2016)                                                        | PELU 를 양수/음수 구간 모두 learnable (c, a, b) 삼 파라미터로 매개변수화          |
| 적응형 AF > learnable slope          | FReLU: Flexible Rectified Linear Units for Improving Convolutional Neural Networks (2017)                                               | FReLU 를 ReLU 의 cutoff 를 learnable bias 로 이동시키는 구조로 제안               |
| 적응형 AF > learnable shape          | Learning Activation Functions to Improve Deep Neural Networks (2014)                                                                    | APL 뉴런별 learnable hinge 파라미터로 piecewise linear non-convex activation 학습 |
| 적응형 AF > learnable shape          | Activation Ensembles for Deep Neural Networks (2017)                                                                                    | 각 뉴런이 여러 활성화의 convex combination 을 스스로 선택하는 activation ensemble |
| 적응형 AF > learnable shape          | Dynamic ReLU (2020)                                                                                                                     | DY-ReLU 를 입력 전체의 global context 를 읽는 hyper function 으로 동적 생성       |
| 적응형 AF > learnable shape          | Three Decades of Activations: A Comprehensive Survey of 400 Activation Functions for Neural Networks (2024)                             | PReLU, TAAF, trainable spline/rational, blended activation 등 적응형 AF 체계화    |
| 적응형 AF > parameterized activation | Activation Functions: Comparison of Trends in Practice and Research for Deep Learning (2018)                                            | 연구와 실제 간 AF 선택 간극 정리 (PReLU 등 learnable 활성화는 연구, ReLU 는 실제) |
| 적응형 AF > parameterized activation | How important are activation functions in regression and classification? A survey, performance comparison, and future directions (2022) | 적응형 활성화 함수가 거의 모든 문제에서 고전적 대응형을 능월                      |
| 적응형 AF > parameterized activation | A Survey on Activation Functions and their relation with Xavier and He Normal Initialization (2020)                                     | ReLU/PReLU 의 zero-centered 가 아니며 negative input 구간 분산 감소 보정          |

#### 2.3 Gated Activation 및 Transformer 구조 연구

Transformer 와 gated architecture 에서 활성화 함수가 gate 와 결합하여 표현력을 확장하는 구조를 연구한다.

| 분류                         | 논문명                                                                                                                      | 분류 근거                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Gated AF > GLU variants      | GLU Variants Improve Transformer (2020)                                                                                     | GLU/Bilinear/ReGLU/GEGLU/SwiGLU 등 gate 와 content projection 곱셈 결합 |
| Gated AF > gated convolution | Language Modeling with Gated Convolutional Networks (2016)                                                                  | GLU 를 gated tanh 계열보다 우수하며 linear path 제공으로 gradient 전달  |
| Gated AF > dual rectifier    | Dual Rectified Linear Units (DReLUs): A Replacement for Tanh Activation Functions in Quasi-Recurrent Neural Networks (2017) | DReLU 를 두 ReLU 차로 정의하고 QRNN tanh 대체와 signed output 제공      |
| Gated AF > gated convolution | Language Modeling with Gated Convolutional Networks (2016)                                                                  | GLU 를 gated tanh 계열보다 우수하며 linear path 제공으로 gradient 전달  |
| Gated AF > gated convolution | Language Modeling with Gated Convolutional Networks (2016)                                                                  | GLU 를 gated tanh 계열보다 우수하며 linear path 제공으로 gradient 전달  |

#### 2.4 활성화 초기화 및 수렴 안정성 연구

활성화 함수와 연계한 초기화 전략, gradient flow, mean-variance dynamics 를 분석하여 학습 안정성을 확보하는 연구를 포함한다.

| 분류                          | 논문명                                                                                             | 분류 근거                                                                           |
| ----------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 초기화 > gradient path 보존   | A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015)                     | recurrent weight 를 identity matrix 로 초기화하여 gradient 가 시간축 따라 일정 전달 |
| 초기화 > rectifier-aware init | Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015) | He initialization 으로 rectifier 계열에 적합한 2/n 분산 스케일 제시                 |
| 초기화 > dying ReLU 완화      | Dying ReLU and Initialization: Theory and Numerical Examples (2019)                                | RAI(Randomized Asymmetric Initialization) 로 born-dead 확률 70-90% 감소             |
| 초기화 > self-normalizing     | Self-Normalizing Neural Networks (2017)                                                            | SELU 를 mean-variance dynamics 로 fixed point 수렴으로 설계하고 LeCun init 조합     |
| 초기화 > bias shift 해소      | Improving Deep Network Learning by Exponential Linear Units (ELUs) (2015)                          | ELU 가 unit-wise Fisher 정보를 통해 natural gradient 보정 방향 유도                 |
| 초기화 > dead ReLU 해결       | A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015)                     | scaled identity(\α I) 로 task memory horizon 에 맞춰 gradient 보존                  |
| 초기화 > activation-centering | Activation Functions in Artificial Neural Networks: A Systematic Overview (2020)                   | zero-centered 활성화가 학습 안정성에 핵심이며 Xavier/He init 관계 설명              |

### 3. 종합 정리

Activation Function 연구는 고정형/적응형 AF 설계, gated 구조 도입, 초기화 전략, mean-variance dynamics 분석 등 4 개의 주요 축으로 조직화된다. 고정형 AF 연구는 ReLU/GELU/Swish/ELU/SELU 등의 수학적 성질과 최적화 특성을 체계적으로 분석하며, 적응형 AF 연구는 learnable slope, learnable shape, parameterized activation 을 통해 task 와 데이터에 맞춘 최적화를 가능하게 한다. Gated AF 는 Transformer 와 gated convolution 에서 gate 와 활성화의 곱셈적 결합으로 표현력과 gradient 흐름을 동시에 확보한다. 초기화 및 안정성 연구는 gradient path 보존, dying ReLU 완화, self-normalization 등을 통해 깊은 네트워크 학습의 안정성을 제공한다. 결과적으로 활성화 함수 연구는 고정/적응, gated/convolutional, theory/empirical 이나 initialization/architecture 등 다양한 차원에서 신경망의 표현력과 학습 동역학을 제어하는 핵심 요소임을 입증한다.

## 2 장. 방법론 분석

## 1. 공통 문제 설정 및 접근 구조

### 1.1 전체 논문들이 다루는 공통 문제

활성화 함수 (Activation Function, AF) 에 대한 연구는 다음과 같은 공통 문제 설정을 공유한다:

| 문제 영역            | 핵심 질문                                                |
| -------------------- | -------------------------------------------------------- |
| **비선형성 부여**    | 뉴런 출력을 제한하거나 변형하여 네트워크의 표현력을 확보 |
| **경사 흐름 안정화** | 역전파 시 gradient 가 적절한 크기로 전파되도록 보장      |
| **학습 안정성**      | 초기화, 수렴 속도와 관련된 최적화 문제 완화              |
| **계산 효율성**      | 연산 복잡도 증가 없이 성능 개선                          |

### 1.2 방법론 관점의 공통 구조

```text
┌───────────────────────────────────────────────────────────┐
│                    활성화 함수 설계 프레임워크            │
├───────────────────────────────────────────────────────────┤
│  입력 (x) → 활성화 함수 f(·) → 출력 (y)                   │
│                                                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ 선형 변환   │ →  │ 활성화      │ →  │ 비선형      │    │
│  │ (Weight)    │    │ 변환        │    │ 출력        │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
└───────────────────────────────────────────────────────────┘
```

#### 공통적 특징

- **입력 → 처리 → 출력**의 일원 구조
- **비선형 변환**이 필수 (선형만으로는 깊은 네트워크 학습 불가)
- **초기화와의 관계**: 활성화 함수의 성질에 따른 적절한 초기화 필요

## 2. 방법론 계열 분류

### 2.1 계열 분류 체계

제공된 논문들을 방법론적 접근 방식으로 다음과 같이 분류한다:

| 계열명                        | 계열 정의                                                    | 해당 논문 (연도)                                                                                                                                                                                                                                               |
| ----------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **초기화 기반 계열**          | 활성화 함수의 성질에 맞춰 최적화된 가중치 초기화 전략과 결합 | A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015), Delving Deep into Rectifiers (2015), A Survey on Activation Functions and their relation with Xavier and He Normal Initialization (2020), Dying ReLU and Initialization (2019) |
| **변수 매개변수화 계열**      | 활성화 함수의 파라미터를 학습 가능한 형태로 전환             | Parametric Exponential Linear Unit (2016), Learning Activation Functions to Improve Deep Neural Networks (2014), PReLU (2015), PDELU, PAU                                                                                                                      |
| **Gated/Multiplicative 계열** | 곱셈적 게이트 구조로 표현력 및 선택적 정보 흐름 제어         | GLU Variants Improve Transformer (2020), Language Modeling with Gated Convolutional Networks (2016), Dual Rectified Linear Units (2017)                                                                                                                        |
| **Self-Normalizing 계열**     | 네트워크의 mean/variance가 자동으로 안정화되도록 설계        | Self-Normalizing Neural Networks (2017), Fast and Accurate Deep Network Learning by Exponential Linear Units (2015)                                                                                                                                            |
| **입력 적응형 계열**          | 입력에 따라 동적으로 활성화 함수 형태를 조정                 | Dynamic ReLU (2020), FReLU (2017), Weighted Sigmoid Gate Unit (2018)                                                                                                                                                                                           |
| **자동 탐색 계열**            | 자동 탐색이나 진화 알고리즘으로 새로운 활성화 함수 발견      | Searching for Activation Functions (2017), Evolutionary Optimization of Deep Learning Activation Functions (2020)                                                                                                                                              |
| **Gated Convolutional 계열**  | convolutive 구조에서 gating 을 결합한 접근                   | Language Modeling with Gated Convolutional Networks (2016)                                                                                                                                                                                                     |
| **Convex Hull 계열**          | Convex 또는 piecewise linear 함수로 근사                     | Maxout Networks (2013), APL (2014), SReLU (2015)                                                                                                                                                                                                               |

### 2.2 계열별 상세 분석

#### (1) 초기화 기반 계열

**계열 정의**: 활성화 함수와 최적화된 가중치 초기화를 함께 제안하며, 초기화 전략이 네트워크 안정성에 핵심

**공통 특징**:

| 특징                   | 설명                                                               |
| ---------------------- | ------------------------------------------------------------------ |
| **분산 보존**          | 층별 variance가 일정하게 유지되도록 설계                           |
| **초기화-활성화 매칭** | ReLU/Rectifier 계열 → He 초기화, Tanh/Sigmoid 계열 → Xavier 초기화 |
| **비대칭성 고려**      | zero-hard rectification이 variance를 감소시킴을 반영               |

**핵심 구성 요소**:

- **He Normal Initialization**: $w \sim \mathcal{N}(0, \frac{2}{n_{in}})$ (ReLU 계열)
- **Xavier Initialization**: $w \sim U[-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}]$
- **Scaled Identity**: $W_{hh} = \alpha I$ (RNN 초기화)

**해당 논문 목록**:

- A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015)
  - IRNN: $W_{hh} = I$ (identity matrix)
  - Gradient가 시간 축을 따라 일정하게 유지됨
- Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015)
  - Rectifier-aware Initialization: ReLU에서 활성의 절반이 0이 됨을 반영
- A Survey on Activation Functions and their relation with Xavier and He Normal Initialization (2020)
  - 활성화 함수-초기화 매칭 권장사항
- Dying ReLU and Initialization: Theory and Numerical Examples (2019)
  - RAI: Randomized Asymmetric Initialization
  - Symmetric initialization 에서 dying probability 높음

**방법론적 해석**:

초기화 전략 선택 = f(활성화 함수의 분산 보존 성질)

- **Rectifier 계열**: $f(x) = \max(0,x)$ → 음수 입력이 0 으로 잘림 → 분산 감소 → 더 큰 초기화 분산 필요
- **Sigmoidal 계열**: bounded, zero-centered → Xavier 초기화 적합
- **Zero-mean 가용**: ReLU 계열은 mean이 양수 → bias shift 발생 → mean centering 필요 (ELU 등)

#### (2) 변수 매개변수화 계열

**계열 정의**: 활성화 함수의 파라미터를 학습 가능한 변수로 전환하여 데이터/레이어별 최적 형태 학습

**공통 특징**:

| 특징                        | 설명                                       |
| --------------------------- | ------------------------------------------ |
| **Parameetric Learning**    | slope, scale, offset 등 파라미터 학습 가능 |
| **End-to-End Optimization** | 기존 backpropagation으로 동시 학습         |
| **파라미터 효율성**         | 전체 weight 대비 매우 작은 추가 파라미터   |

**핵심 구성 요소**:

| 함수      | 정의                                             | 파라미터                                   |
| --------- | ------------------------------------------------ | ------------------------------------------ |
| **PReLU** | $f(x) = x (x>0), a x (x\le 0)$                   | 음수 구간 slope (channel-wise 또는 shared) |
| **PELU**  | 양수/음수 구간 모두 매개변수화                   | $a, b, c$ (각 layer 당 2 개)               |
| **MPELU** | $f(x) = x (x>0), \alpha(e^{\beta x}-1) (x\le 0)$ | $\alpha, \beta$ (channel-wise)             |

**해당 논문 목록**:

- Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015) - PReLU
- Parametric Exponential Linear Unit for Deep Convolutional Neural Networks (2016) - PELU
- Improving Deep Neural Network with Multiple Parametric Exponential Linear Units (2016) - MPELU
- Learning Activation Functions to Improve Deep Neural Networks (2014) - APL

**방법론적 구조**:

기존 고정 활성화 함수 $f(x)$ → 학습 가능한 $f(\theta, x)$

- AP L: $h_i(x) = \max(0,x) + \sum_{s=1}^M a_i^s \max(0, -x+b_i^s)$
- PELU: 양수/음수 구간 모두 scale factor 학습
- MPELU: ELU 와 PReLU 의 하이브리드

**학습 전략**:

1. **Weight Initialization**: 제안된 analytic initialization 적용
2. **Parameter Learning**: $\theta$를 backpropagation 으로 최적화
3. **Regularization**: 학습 파라미터에 appropriate regularization 추가

#### (3) Gated/Multiplicative 계열

**계열 정의**: element-wise 곱셈적 게이트 구조로 정보를 선택적으로 활성화하거나 억제

**공통 특징**:

| 특징                      | 설명                           |
| ------------------------- | ------------------------------ |
| **Multiplicative Gating** | $\sigma(xW) \otimes xV$ 형태   |
| **Dual Path**             | content path 와 gate path 분리 |
| **Gate Function**         | 주로 sigmoid/tanh 기반         |

**핵심 구성 요소**:

GLU 계열: output = content_path $\otimes$ activation(gate_path)

| 함수         | 수식                                          | 특징            |
| ------------ | --------------------------------------------- | --------------- |
| **GLU**      | $\sigma(XV + c) \otimes XW + b$               | 기본 gated 구조 |
| **ReGLU**    | $\text{ReLU}(\sigma(XV + c)) \otimes XW + b$  | ReLU 게이트     |
| **GEGLU**    | $\text{GELU}(\sigma(XV + c)) \otimes XW + b$  | GELU 게이트     |
| **SwiGLU**   | $\text{Swish}(\sigma(XV + c)) \otimes XW + b$ | Swish 게이트    |
| **Bilinear** | $\sigma(XW_1)\otimes XW_2 + b$                | bilinear form   |

**해당 논문 목록**:

- GLU Variants Improve Transformer (2020)
- Language Modeling with Gated Convolutional Networks (2016)
- Dual Rectified Linear Units (2017) - DReLU, DELU

**방법론적 구조**:

```text
content_path = xW
gate_path = σ(xV + b)
output = content_path ⊗ gate_path
```

**이론적 근거**:

- GLU 는 content path 자체를 선형 변환 → gradient 전달이 더 직접적
- tanh 기반 gate 는 양쪽 경로 모두 nonlinearity 개입 → GLU 보다 gradient 흐름 유리

#### (4) Self-Normalizing 계열

**계열 정의**: 명시적 정규화 없이도 네트워크의 mean/variance 가 zero-mean, unit-variance로 자동 수렴하도록 설계

**공통 특징**:

| 특징                     | 설명                                             |
| ------------------------ | ------------------------------------------------ |
| **Fixed Point Property** | mean/variance mapping 이 stable fixed point 가짐 |
| **Variance Control**     | exploding/vanishing variance 방지                |
| **No Explicit Norm**     | batch normalization 등 명시적 정규화 불필요      |

**핵심 구성 요소**:

- **SELU**: $\text{SELU}(x) = \lambda \begin{cases} \alpha(e^{\alpha x}-1) & x>0 \\ \alpha x & x\le 0 \end{cases}$
- **Scale Factor**: $\alpha, \lambda$의 고정된 값 필요
- **Mean-Variance Mapping**: 한 층에서 다음 층으로의 mean/variance 전이 관계

**해당 논문 목록**:

- Self-Normalizing Neural Networks (2017)
- Fast and Accurate Deep Network Learning by Exponential Linear Units (2015)

**방법론적 구조**:

```text
(μ, ν) → g(μ, ν) = (μ̃, ν̃) → fixed point (0, 1) 근처 수렴
```

- **Theorem 1**: mapping 이 stable and attracting fixed point 가짐
- **Theorem 2**: exploding variance 방지 (variance 가 크면 감소)
- **Theorem 3**: vanishing variance 방지 (variance 가 작으면 증가)

**초기화 조건**:

- LeCun normal initialization 필요
- SELU 의 $\alpha, \lambda$값을 정확히 맞춰야 fixed point 성질 확보

#### (5) 입력 적응형 계열

**계열 정의**: 입력 데이터의 특성 (global context, channel/statistics)에 따라 활성화 함수 파라미터를 동적으로 생성

**공통 특징**:

| 특징                         | 설명                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| **Dynamic Parameter**        | 입력 $\mathbf{x}$에 따라 파라미터 $\theta(\mathbf{x})$생성 |
| **Global Context**           | 전체 입력에서 파라미터 생성                                |
| **Computational Efficiency** | 작은 hyper function 추가만으로도 효과                      |

**핵심 구성 요소**:

| 함수             | 수식                                                                                          | 특징                        |
| ---------------- | --------------------------------------------------------------------------------------------- | --------------------------- |
| **Dynamic ReLU** | $f_{\theta(\mathbf{x})}(\mathbf{x}) = \max_k \{ a_c^k(\mathbf{x}) x_c + b_c^k(\mathbf{x}) \}$ | 입력에 따라 slope 조정      |
| **FReLU**        | $\text{ReLU}(x) + b_l$                                                                        | layer-wise learnable offset |
| **WiG**          | $\mathbf{x} \odot \sigma(\mathbf{W}_g\mathbf{x} + \mathbf{b}_g)$                              | vector-input gating         |

**해당 논문 목록**:

- Dynamic ReLU (2020)
- FReLU: Flexible Rectified Linear Units for Improving Convolutional Neural Networks (2017)
- Weighted Sigmoid Gate Unit for an Activation Function of Deep Neural Network (2018)

**변형 전략** (Dynamic ReLU):

| 변형          | Spatial      | Channel      | 적용                            |
| ------------- | ------------ | ------------ | ------------------------------- |
| **DY-ReLU-A** | shared       | shared       | general                         |
| **DY-ReLU-B** | shared       | channel-wise | image classification, backbone  |
| **DY-ReLU-C** | channel-wise | channel-wise | head, spatially sensitive tasks |

**방법론적 구조**:

```text
hyper_function(x) → θ(x) → activation f_θ(x)(x)
```

#### (6) 자동 탐색 계열

**계열 정의**: 자동 탐색 (exhaustive search, reinforcement learning, evolutionary algorithm) 으로 새로운 활성화 함수 발견

**공통 특징**:

| 특징                        | 설명                                     |
| --------------------------- | ---------------------------------------- |
| **Compositional Search**    | unary/binary operator 조합으로 함수 생성 |
| **Validation-based Reward** | 성능 지표로 탐색 방향 유도               |
| **Discovery**               | 인간 설계가 아닌 데이터 기반 발견        |

**핵심 구성 요소**:

| 요소                | 설명                                                            |
| ------------------- | --------------------------------------------------------------- |
| **Core Unit**       | 두 입력 각각에 unary function 적용 후 binary function 으로 결합 |
| **Search Space**    | 수십억 개의 후보 포함                                           |
| **Reward Function** | validation accuracy 또는 loss                                   |

**해당 논문 목록**:

- Searching for Activation Functions (2017) - Swish 발견
- Evolutionary Optimization of Deep Learning Activation Functions (2020)

**탐색 전략**:

- **Mutation**: 노드 무작위 교체
- **Crossover**: subtree 교환
- **Fitness Function**: loss-based > accuracy-based > random search

#### (7) Gated Convolutional 계열

**계열 정의**: convolution 구조에서 gating 을 결합하여 문맥 길이를 효율적으로 커버

**공통 특징**:

| 특징                     | 설명                                            |
| ------------------------ | ----------------------------------------------- |
| **Causal Convolution**   | 과거 token 만 보는 causal 구조                  |
| **Hierarchical Feature** | 깊은 convolution stack으로 receptive field 확장 |
| **Gradient Flow**        | chain 구조보다 vanishing gradient 부담 감소     |

**핵심 구성 요소**:

$h_l(X) = (X * W + b) \otimes \sigma(X * V + c)$

| 구성 요소        | 역할                                      |
| ---------------- | ----------------------------------------- |
| **Content path** | $X * W + b$ - 내용 정보 생성              |
| **Gate path**    | $\sigma(X * V + c)$ - gate로 content 조절 |
| **Convolution**  | causal temporal convolution               |

**해당 논문 목록**:

- Language Modeling with Gated Convolutional Networks (2016)

**성능 비교**:
$$
\text{GLU} > \text{bilinear} > \text{linear}
$$

### 2.3 계열별 비교 표

| 계열                 | 핵심 아이디어             | 추가 파라미터       | 수렴 보장           | 주요 제약     |
| -------------------- | ------------------------- | ------------------- | ------------------- | ------------- |
| 초기화 기반          | 최적화된 초기화 전략      | 없음                | 초기화 조건 충족 시 | 초기화 민감도 |
| 변수 매개변수화      | 학습 가능한 파라미터      | 1-2 개/layer        | 학습 안정성 향상    | 파라미터 튜닝 |
| Gated/Multiplicative | 곱셈적 게이트             | projection mat      | 구조적 안정성       | 파라미터 증가 |
| Self-Normalizing     | 자동 mean/variance 안정화 | fixed parameter     | 고정조건 충족 시    | SELU 고정값   |
| 입력 적응형          | 동적 파라미터 조정        | hyper function      | 입력에 따른 적응    | 계산 오버헤드 |
| 자동 탐색            | 데이터 기반 발견          | 생성 결과           | 데이터에 따른 성능  | 탐색 비용     |
| Gated Convolutional  | convolution + gate        | convolution weights | 문맥 커버           | causal 제약   |

## 3. 핵심 설계 패턴 분석

### 3.1 패턴: 활성화 함수의 수학적 성질

#### (1) 수학적 성질 카테고리

| 성질                   | 설명           | 최적화 영향             |
| ---------------------- | -------------- | ----------------------- |
| **Differentiability**  | 미분 가능성    | gradient flow           |
| **Boundedness**        | 출력 범위 제한 | output range            |
| **Zero-centered**      | mean ≈ 0       | bias shift 완화         |
| **Smoothness**         | 연속적 전이    | gradient stability      |
| **Bounded Derivative** | 도함수 bounded | gradient exploding 방지 |

#### (2) 성질별 대표 함수

| 성질               | 함수                    | 수식               |
| ------------------ | ----------------------- | ------------------ |
| Bounded            | Sigmoid, Tanh, Softsign | sigmoid, tanh 계열 |
| Zero-centered      | Tanh, ELU, SELU         | zero-mean 출력     |
| Unbounded          | ReLU, Leaky ReLU, GELU  | unbounded positive |
| Bounded Derivative | CELU, Mish              | derivative ≤ 1     |

### 3.2 패턴: 비선형성의 정도

#### (1) Hard vs Soft Gating

| 유형        | 함수              | Gating 방식               |
| ----------- | ----------------- | ------------------------- |
| Hard gating | ReLU, Leaky ReLU  | $x\cdot \mathbb{I}_{x>0}$ |
| Soft gating | GELU, Swish, Mish | 연속적 weighting          |
| Adaptive    | PReLU, PELU       | 학습 가능 slope           |

**Hard gating**: ReLU 와 같이 sign 기반 hard threshold
**Soft gating**: GELU 의 $x\Phi(x)$처럼 입력 크기에 따라 부드럽게 weighting

### 3.3 패턴: 정규화 전략

#### (1) 정규화 유형

| 유형                    | 적용 위치        | 효과                      |
| ----------------------- | ---------------- | ------------------------- |
| **Layer Normalization** | Softmax 뒤에     | FFN 과 memory 동등화      |
| **Batch Normalization** | ReLU/PReLU/ELU   | 분산/평균 안정화          |
| **None**                | Self-Normalizing | 고정점 성질로 자동 안정화 |
| **Regularization Loss** | Attention 분포   | entropy 기반 정규화       |

#### (2) ReLUFormer 구조

```text
FFN: ReLU + LN → global key-value memory
SAN: ReLU + scale + reg → local key-value memory
```

### 3.4 패턴: Convex Hull 근사

#### (1) Convex Approximation 계열

| 함수       | Convexity        | 근사 가능 함수               |
| ---------- | ---------------- | ---------------------------- |
| **Maxout** | Convex           | convex function              |
| **APL**    | Piecewise convex | non-convex 가능 (hinge 추가) |
| **SReLU**  | Mixed            | convex 와 non-convex 모두    |

**Maxout**: $k$개의 affine response 중 max 선택 → universal convex approximator
**APL**: $\max(0,x) + \sum a^s\max(0, -x+b^s)$ → non-convex 근사 가능

### 3.5 패턴: Gated Information Flow

#### (1) Gate 구조

$$
content = xW \\
gate = \sigma(xV + b) \\
output = content \otimes gate
$$

**GLU 의 장**: content path 선형 → gradient 전달 직접적
**GLU 의 단**: parametric 증가 (hidden dim → 2/3 로 조정)

## 4. 방법론 비교 분석

### 4.1 계열 간 차이점

#### (1) 문제 접근 방식 차이

| 계열             | 접근 방식          | 장단점                                   |
| ---------------- | ------------------ | ---------------------------------------- |
| 초기화 기반      | 초기화-활성화 매칭 | 초기화 조건 명확, but 초기화 민감        |
| 변수 매개변수화  | 파라미터 학습      | 데이터 적응, but 파라미터 튜닝 필요      |
| Gated            | 정보 선택적 통과   | 표현력 향상, but 계산 복잡도             |
| Self-Normalizing | 자동 안정화        | explicit norm 불필요, but 고정 조건 엄격 |
| 입력 적응형      | 동적 조정          | 입력 적응, but computation 오버헤드      |
| 자동 탐색        | 데이터 발견        | 새로운 함수 발견, but 탐색 비용          |

#### (2) 구조/모델 차이

**파라미터 구조**:

```text
기존: f(x) → 0 추가 파라미터
매개변수화: f(θ, x) → 1-2 개/layer
Gated: content ⊗ gate → 2× projection
```

**계산 구조**:

```text
Hard: piecewise linear (ReLU, Leaky ReLU)
Soft: smooth (GELU, Swish, Mish)
Gated: content ⊗ gate (GLU 계열)
```

#### (3) 적용 대상 차이

| 계열             | 주요 적용             | 제한 사항        |
| ---------------- | --------------------- | ---------------- |
| 초기화 기반      | RNN, general FFN      | 초기화 조건 엄격 |
| 변수 매개변수화  | CNN, deep network     | BN 호환성        |
| Gated            | Transformer FFN, Conv | 파라미터 증가    |
| Self-Normalizing | Deep FFN              | 고정 조건 충족   |
| 입력 적응형      | CNN, classification   | 계산 오버헤드    |
| 자동 탐색        | CIFAR 등 benchmark    | 탐색 비용        |

### 4.2 트레이드오프 분석

#### (1) 표현력 vs 계산 효율성

| 함수      | 표현력    | 계산 비용             | Trade-off          |
| --------- | --------- | --------------------- | ------------------ |
| **ReLU**  | 중간      | 매우 낮음             | 계산 효율성 우선   |
| **GELU**  | 높음      | 높음 (soft)           | 표현력 우선        |
| **GLU**   | 높음      | 매우 높음 (2$\times$) | 표현력 vs 파라미터 |
| **Mish**  | 높음      | 높음                  | 계산 효율성 희생   |
| **MPELU** | 매우 높음 | 중간                  | 표현력 vs 파라미터 |

**Trade-off 관계**:

```text
계산 효율 ↑ ↓ 표현력
```

#### (2) 고정 vs 적응형

| 측면          | 고정형 (Fixed) | 적응형 (Adaptive) |
| ------------- | -------------- | ----------------- |
| **초기 학습** | 안정적         | 데이터 의존       |
| **일반화**    | task 의존적    | task 적응적       |
| **파라미터**  | 0              | 1-2 개/layer      |
| **계산**      | 빠름           | 느림              |

**적응형의 장단점**:

- 장점: 데이터 분포 변화에 적응, task-specific 최적
- 단점: 파라미터 증가, 계산 오버헤드, 초기화 민감

#### (3) Hard vs Soft Gating

| 측면               | Hard            | Soft              |
| ------------------ | --------------- | ----------------- |
| **Gradient**       | 불연속성 있음   | 부드러운 gradient |
| **Dying**          | Dying ReLU 가능 | 완화 (Mish 등)    |
| **계산**           | 빠름            | 느림              |
| **Regularization** | limited         | self-regularized  |

### 4.3 활성화 함수별 성능 비교

#### (1) 이미지 분류 벤치마크 (CIFAR-10/100)

| 활성화 함수 | CIFAR-10 (error) | CIFAR-100 (error) | 특성                 |
| ----------- | ---------------- | ----------------- | -------------------- |
| ReLU        | baseline         | baseline          | 기본값               |
| PReLU       | -0.2%            | -0.5%             | 채널별 slope         |
| ELU         | -0.3%            | -0.7%             | bias shift 완화      |
| **MPELU**   | **-0.5%**        | **-1.0%**         | combined 장점        |
| GELU        | -0.4%            | -0.8%             | soft gating          |
| **SELU**    | -0.6%            | -1.2%             | explicit norm 불필요 |

**주의**: SELU 는 fixed point 조건 충족 필요

#### (2) Transformer 벤치마크 (Machine Translation)

| 활성화 함수 | BLEU     | 비고            |
| ----------- | -------- | --------------- |
| ReLU        | baseline | 기본값          |
| **GELU**    | +1.5%    | BERT, GPT 표준  |
| **SwiGLU**  | +2.0%    | FFN gate 구조   |
| **GLU**     | +1.8%    | gated structure |

### 4.4 Activation Function - Initialization 매트릭스

| 활성화 함수 | 권장 초기화  | 고정 조건          | 비고                 |
| ----------- | ------------ | ------------------ | -------------------- |
| ReLU        | He Normal    | -                  | 기본값               |
| Leaky ReLU  | He ($1+a^2$) | -                  | 음수 구간 slope 고정 |
| **PReLU**   | He ($1+a^2$) | -                  | slope 학습           |
| **ELU**     | LeCun Normal | -                  | bias shift 완화      |
| **SELU**    | LeCun Normal | fixed point        | explicit norm 불필요 |
| **CELU**    | LeCun Normal | bounded derivative | ELU 일반화           |
| **MPELU**   | Analytic     | -                  | combined 장점        |
| **GELU**    | LeCun Normal | -                  | BERT 표준            |
| **Mish**    | He/LeCun     | -                  | self-regularized     |
| **SiLU**    | LeCun Normal | -                  | SiLU = Swish         |

## 5. 방법론 흐름 및 진화

### 5.1 초기화 중심 → 표현력 중심

#### (1) 초기 단계 (2013-2015)

**주요 연구 방향**:

1. **Maxout (2013)**: convex hull 기반 universal approximator
2. **ReLU (2013)**: computational simplicity 강조
3. **ReLU initialization (2015)**: He initialization 제안
4. **ELU (2015)**: bias shift 완화
5. **IRNN (2015)**: RNN 초기화 전략

**특징**: 초기화 전략이 네트워크 안정성 결정

#### (2) 발전 단계 (2016-2019)

**주요 연구 방향**:

1. **PReLU (2015)**: 채널별 slope 학습
2. **APL (2014)**: non-convex 근사
3. **PELU/ELU/MPELU (2016)**: 파라미터 학습
4. **CELU (2017)**: bounded derivative
5. **SELU/Self-Normalizing (2017)**: 자동 정규화
6. **Mish (2019)**: self-regularized
7. **Dying ReLU Theory (2019)**: dying problem 이론화

**특징**: 파라미터 학습과 automatic stability 연구 확대

#### (3) 최근 단계 (2020-2024)

**주요 연구 방향**:

1. **Dynamic ReLU (2020)**: 입력 적응형 활성화
2. **GLU variants (2020)**: Transformer FFN 개선
3. **Evolutionary Search (2020)**: 자동 탐색
4. **GELU Analysis (2023)**: 수학적 분석 심화
5. **ReLUFormer (2023)**: Transformer 재해석
6. **300 Activation Survey (2024)**: 체계적 분류

**특징**: 자동 탐색, 일반화 성능 최적, 체계적 분류

### 5.2 활성화 함수 진화 경로

```text
고정 함수 → 파라미터 학습 → 적응형 → 자동 탐색
```

#### (1) 고정 함수 단계

- **Sigmoid/Tanh → ReLU → ELU → CELU → GELU**
- 수학적 성질 최적화 (bounded, smooth, zero-centered)

#### (2) 파라미터 학습 단계

- **PReLU → PELU → MPELU → APL**
- slope, scale 등 파라미터 학습으로 데이터 적응

#### (3) 적응형 단계

- **Dynamic ReLU → FReLU → WiG**
- 입력에 따라 파라미터 동적 조정

#### (4) 자동 탐색 단계

- **Searching for Activation (2017) → Swish**
- **Evolutionary Optimization → 일반화 함수**
- 데이터 기반 함수 발견

### 5.3 활성화 함수와 네트워크 아키텍처 진화

| 시대                        | 대표 아키텍처            | 대표 활성화 함수       | 관계                |
| --------------------------- | ------------------------ | ---------------------- | ------------------- |
| **Shallow (2010-2013)**     | MLP, CNN                 | Sigmoid, Tanh          | fixed activation    |
| **Deep (2014-2017)**        | Deep CNN, ResNet         | ReLU, ELU, PReLU       | He initialization   |
| **Very Deep (2018-2020)**   | ResNet-152+, Wide ResNet | Swish, GELU, Mish      | adaptive activation |
| **Transformer Era (2020-)** | BERT, GPT, ViT           | GELU, SwiGLU           | gated FFN           |
| **Efficient Era (2021-)**   | MobileNet, EfficientNet  | ReLU, Hard Swish, SiLU | efficiency-first    |

## 6. 종합 정리

### 6.1 방법론 지형 요약

제공된 논문들의 방법론을 다음 네 가지 축으로 요약할 수 있다:

```text
┌─────────────────────────────────────────────────────────────┐
│                    활성화 함수 방법론 지형                  │
├─────────────────────────────────────────────────────────────┤
│  제 1 축: 초기화 전략                                       │
│    ├─ 최적화된 초기화 (He, Xavier)                          │
│    ├─ 비대칭성 반영 (dying ReLU)                            │
│    └─ 자동 안정화 (self-normalizing)                        │
├─────────────────────────────────────────────────────────────┤
│  제 2 축: 표현력                                            │
│    ├─ Convex approximation (Maxout, APL)                    │
│    ├─ Soft gating (GELU, Swish)                             │
│    └─ Gated structure (GLU, SwiGLU)                         │
├─────────────────────────────────────────────────────────────┤
│  제 3 축: 적응도                                            │
│    ├─ 파라미터 학습 (PReLU, MPELU)                          │
│    ├─ 동적 생성 (Dynamic ReLU)                              │
│    └─ 자동 탐색 (Swish, general functions)                  │
├─────────────────────────────────────────────────────────────┤
│  제 4 축: 정규화 전략                                       │
│    ├─ Hard/Soft regularization                              │
│    ├─ Explicit normalization (Batch/Layer)                  │
│    └─ Implicit normalization (Self-Normalizing)             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 핵심 통찰

1. **초기화-활성화 매칭 필수**: 활성화 함수의 수학적 성질에 맞는 초기화 전략이 네트워크 안정성 결정
2. **표현력-계산 Trade-off**: 고정형 (ReLU) vs 적응형 (MPELU) 간의 trade-off 존재
3. **자동화 경향**: 수동 설계 → 파라미터 학습 → 자동 탐색으로 진화
4. **Gated 구조의 부상**: Transformer 시대에 gated structure(FFN 내 gate) 중요성 증가

### 6.3 결론

활성화 함수 연구는 다음 세 가지 핵심 질문에 답하기 위한 방법론적 진화 과정을 거쳤다:

1. **어떤 활성화 함수가 좋은가?**: 수학적 성질 (bounded, smooth, zero-centered) 과 최적화 역학 (vanishing gradient, bias shift) 관점
2. **어떻게 초기화해야 하는가?**: 활성화 함수의 분산 보존 성질에 맞는 초기화 전략 필요
3. **어떻게 개선할 수 있는가?**: 파라미터 학습, 적응형, 자동 탐색 등 다양한 접근

이러한 방법론적 다양성은 단일 "best activation"이 존재하지 않음을 보여준다. 대신, 네트워크 아키텍처, 데이터 분포, 학습 목표에 맞는 적절한 활성화 함수 선택이 필요하다.

## 3 장. 실험결과 분석

## 1. 평가 구조 및 공통 실험 설정

### 1.1 주요 데이터셋 유형

| 데이터셋                             | 도메인      | 활용 논문 수 | 특징                         |
| ------------------------------------ | ----------- | ------------ | ---------------------------- |
| **CIFAR-10/100**                     | 이미지 분류 | 18 개        | 표준 벤치마크, 32×32 이미지  |
| **ImageNet**                         | 이미지 분류 | 15 개        | 1000 클래스, 대규모 벤치마크 |
| **MNIST**                            | 이미지 분류 | 14 개        | 간단한 손글씨 숫자           |
| **STL-10**                           | 이미지 분류 | 4 개         | 소규모 이미지, 연구용        |
| **German-English**                   | 번역        | 3 개         | 문서/문장 수준               |
| **TIMIT**                            | 음성 인식   | 3 개         | ASR 표준 벤치마크            |
| **SVHN**                             | 이미지 분류 | 1 개         | 대용량 집합식 학습           |
| **UCI repository**                   | 다양한 분류 | 121 개       | 기계학습 표준 벤치마크       |
| **Tox21**                            | 약물 발견   | 1 개         | 12,000 화합물 분류           |
| **Atari 2600**                       | 강화학습    | 1 개         | 12 게임 서브셋               |
| **WikiText-103/Google Billion Word** | NLP         | 2 개         | 언어 모델링                  |

### 1.2 평가 환경 유형

| 환경 유형           | 주요 활용 분야    | 대표 논문                 |
| ------------------- | ----------------- | ------------------------- |
| **실험/시뮬레이션** | 대부분의 벤치마크 | CIFAR, ImageNet, MNIST 등 |
| **실환경**          | 거의 없음         | -                         |
| **혼합 환경**       | 음성 인식, 번역   | TIMIT, WMT 번역           |

### 1.3 비교 방식

| 비교 유형          | 활용 빈도 | 비고                   |
| ------------------ | --------- | ---------------------- |
| **Baseline 대비**  | 매우 높음 | ReLU, Sigmoid, Tanh 등 |
| **SOTA 비교**      | 중간      | 최근 SOTA 와 비교      |
| **함수 간 비교**   | 높음      | ReLU vs Swish 등       |
| **Ablation Study** | 중간      | 구성 요소 제거 실험    |

### 1.4 주요 평가 지표

| 지표 유형     | 대표 지표                   | 활용도    |
| ------------- | --------------------------- | --------- |
| **분류 성능** | Top-1/Top-5 error, Accuracy | 매우 높음 |
| **번역/NLP**  | BLEU, Perplexity            | 높음      |
| **음성 인식** | CER, WER                    | 중간      |
| **강화학습**  | Average Score, P-value      | 낮음      |
| **검출/분할** | AP, mAP                     | 중간      |
| **일반화**    | Loss, Validation Error      | 높음      |
| **계산 효율** | FLOPs, Convergence Speed    | 중간      |

## 2. 주요 실험 결과 정렬

### 2.1 이미지 분류 성능 (CIFAR-10/100/ImageNet)

| 논문명                                                                                                   | 데이터셋/환경                            | 비교 대상                      | 평가 지표                         | 핵심 결과                                                 |
| -------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------ | --------------------------------- | --------------------------------------------------------- |
| A Study on ReLU and Softmax in Transformer (2023)                                                        | Europarl7 En-De                          | vanilla Transformer, Sparsemax | BLEU                              | 1024 길이의 긴 시퀀스에서 ReLUFormer 1.15 BLEU 향상       |
| Activation Ensembles for Deep Neural Networks (2017)                                                     | MNIST, ISOLET, CIFAR-100, STL-10         | 6 개 activation set            | Classification error              | MNIST 에서 ReLU 가 가장 자주 우세                         |
| Activation Functions in Artificial Neural Networks: A Systematic Overview (2021)                         | -                                        | 수학적 분석                    | -                                 | ReLU, softsign 를 practice standard 로 권장               |
| Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark (2021)                       | CIFAR10/100, German-English, LibriSpeech | 200 개+ AF                     | Accuracy, BLEU, CER               | CIFAR10 MobileNet: Softplus 91.05% (ReLU 90.10%)          |
| Continuously Differentiable Exponential Linear Units (2017)                                              | -                                        | ELU vs CELU                    | C¹ 연속성, Derivative boundedness | CELU 가 ELU 의 모든 단점 해결                             |
| Deep Learning with S-shaped Rectified Linear Activation Units (2015)                                     | CIFAR-10/100, ImageNet                   | ReLU 기반 NIN/GoogLeNet        | Test Error                        | ImageNet: SReLU 1.24% 향상, 추가 파라미터 21.6K           |
| Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (2015)       | ImageNet 2012                            | ReLU 기반 GoogLeNet            | Top-1/5 error                     | PReLU-net: 4.94% (인간 성능 5.1% 최초 극복)               |
| Learning Activation Functions to Improve Deep Neural Networks (2014)                                     | CIFAR-10/100, Higgs boson                | 5 개 ensemble, baseline        | Classification error              | CIFAR-10: 7.51%, CIFAR-100: 30.83%                        |
| Mish: A Self Regularized Non-Monotonic Activation Function (2019)                                        | MNIST, CIFAR-10, ImageNet-1k             | ReLU, Swish, GELU              | Top-1 accuracy                    | CIFAR-10: 87.48% (ReLU 86.66%), ImageNet ResNet-50: 76.1% |
| Improving Deep Learning by Inverse Square Root Linear Units (2017)                                       | MNIST                                    | ReLU, ELU                      | Accuracy, Cross-entropy loss      | ISRLU 가 training speed 및 generalization 개선            |
| Improving Deep Neural Network with Multiple Parametric Exponential Linear Units (2016)                   | CIFAR-10/100, ImageNet                   | ReLU, PReLU, ELU               | Test Error                        | CIFAR-100 MPELU: 18.81% (Pre-ResNet 대비 더 낮음)         |
| FReLU: Flexible Rectified Linear Units for Improving Convolutional Neural Networks (2017)                | CIFAR-10/100, ImageNet                   | ReLU, ELU, PReLU               | Accuracy, Convergence             | CIFAR-100: 97.8% (MNIST), ImageNet: 51.20%                |
| Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning (2017) | Tetris, Atari 2600                       | ReLU, sigmoid, SiLU            | Average Score, BLEU               | stochastic SZ-Tetris: dSiLU 1 위 (ReLU 대비 20% 향상)     |
| Weighted Sigmoid Gate Unit for an Activation Function of Deep Neural Network (2018)                      | CIFAR-10/100, Yang91                     | ReLU, ELU, Swish               | Validation Accuracy, PSNR         | CIFAR-10/100: WiG 최고 성능, Denoising: PSNR 일관된 개선  |
| Parametric Exponential Linear Unit for Deep Convolutional Neural Networks (2016)                         | MNIST, CIFAR-10/100, ImageNet            | ELU, BN-ReLU, BN-PReLU         | Test Error                        | ImageNet NiN: PELU 36.06% vs ELU 40.40%                   |
| Searching for Activation Functions (2017)                                                                | CIFAR-10/100, ImageNet, WMT              | ReLU, Swish                    | Top-1 Accuracy, BLEU              | CIFAR-10 Swish: ReLU 일관되게 능가, WMT: +0.6 BLEU        |
| Evolutionary Optimization of Deep Learning Activation Functions (2020)                                   | CIFAR-10/100                             | ReLU, Swish                    | Validation accuracy               | ReLU/Swish 모두 이긴 new activation 발견                  |
| Gaussian Error Linear Units (GELUs) (2016)                                                               | MNIST, CIFAR-10/100, TIMIT               | ReLU, ELU                      | Log loss, Test Error              | CIFAR: shallow/deep 모두 outperforms, Twitter POS: 12.57% |
| GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance (2023)  | CIFAR-10/100, STL-10                     | Various AF                     | Accuracy                          | 모든 데이터셋에서 superior performance                    |
| Dual Rectified Linear Units (DReLUs) (2017)                                                              | Sentiment, LM, QRNN                      | tanh-QRNN, LSTM                | Perplexity, Classification        | 8 층 QRNN: tanh-QRNN 대비 SOTA, 계산량 증가               |

### 2.2 음성 인식/번역 성능

| 논문명                                                                                                   | 데이터셋/환경     | 비교 대상           | 평가 지표             | 핵심 결과                                     |
| -------------------------------------------------------------------------------------------------------- | ----------------- | ------------------- | --------------------- | --------------------------------------------- |
| A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015)                           | TIMIT             | 2/5-layer LSTM/iRNN | Train/Test Error      | 5-layer bidirectional: LSTM 28.5%, iRNN 28.9% |
| A Study on ReLU and Softmax in Transformer (2023)                                                        | Europarl7 En-De   | vanilla Transformer | BLEU                  | ReLUFormer: 1.15 BLEU 향상 (길이 1024)        |
| Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark (2021)                       | German-English    | Various AF          | BLEU                  | Tanh 20.93%, SELU 20.85%, SRS 20.66%          |
| Language Modeling with Gated Convolutional Networks (2016)                                               | WikiText-103, GBW | GCNN, LSTM          | Perplexity            | GCNN-8: 38.1, GCNN-10: 31.9 vs LSTM 39.8      |
| Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning (2017) | Atari 2600        | DQN, double DQN     | Normalized Mean Score | DQN 대비 +232%, double DQN 대비 +161%         |

### 2.3 계산 효율 및 수렴 속도

| 논문명                                                                                       | 환경           | 비교 대상            | 평가 지표                               | 핵심 결과                                                     |
| -------------------------------------------------------------------------------------------- | -------------- | -------------------- | --------------------------------------- | ------------------------------------------------------------- |
| Dynamic ReLU (2020)                                                                          | ImageNet, COCO | Static activation    | Top-1 accuracy, AP, FLOPs               | MobileNetV2: +4.2% accuracy, FLOPs 증가 ~5%                   |
| Continuously Differentiable Exponential Linear Units (2017)                                  | -              | ELU vs CELU          | Derivative boundedness                  | CELU: bounded derivative (최대 1), ELU 는 큰 α 에서 exploding |
| Improving Deep Learning by Inverse Square Root Linear Units (2017)                           | MNIST          | ReLU, ELU            | Training speed, Accuracy                | ISRLU: training error 더 빠르게 감소                          |
| Activation Functions: Comparison of Trends in Practice and Research for Deep Learning (2018) | ImageNet 계열  | Research vs Practice | Adoption rate                           | 연구: 다양한 AF 제안, 실무: ReLU/Softmax 주도                 |
| Improving Deep Neural Network with Multiple Parametric Exponential Linear Units (2016)       | CIFAR-10 NIN   | ReLU, PReLU, ELU     | Classification performance, Convergence | MPELU: 더 나은 convergence 와 classification                  |
| Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark (2021)           | -              | Various AF           | 학습 시간                               | PDELU/SRS: 학습 시간 크게 증가                                |

## 3. 성능 패턴 및 경향 분석

### 3.1 공통적으로 나타나는 성능 개선 패턴

| 개선 패턴                              | 관찰된 논문                             | 비고                                      |
| -------------------------------------- | --------------------------------------- | ----------------------------------------- |
| **ReLU 대비 adaptive activation 우위** | Adaptive activation 과 Rowdy activation | 거의 모든 데이터셋에서 개선               |
| **Smooth activation 의 일반화 우위**   | GELU, Swish, Mish, CELU                 | NLP, ASR, vision 모두                     |
| **층별 activation 차별화**             | Deep & Narrow 네트워크, APL             | 깊은 네트워크에서 layer-dependent 선택    |
| **Data-adaptive activation 의 성능**   | APL unit, PELU, Dynamic ReLU            | dataset 특성에 맞춰 학습                  |
| **Gating 구조 결합 시 성능 향상**      | GLU 계열, Gated Convolution             | Transformer, gated architecture 에서      |
| **Scale-similarity 달성 시 안정성**    | CELU, SReLU                             | parameter 조정 없이 family 전체 일괄 조정 |

### 3.2 특정 조건에서만 성능이 향상되는 경우

| 조건                         | 성능 개선 활성화 함수   | 관찰된 논문                                       |
| ---------------------------- | ----------------------- | ------------------------------------------------- |
| **긴 시퀀스 (길이>512)**     | ReLU, Softmax 대비 ReLU | A Study on ReLU and Softmax in Transformer (2023) |
| **깊고 좁은 네트워크**       | RAI, DReLU              | Dying ReLU and Initialization (2019)              |
| **경량 모델**(MobileNet)     | Dynamic ReLU, Swish     | Dynamic ReLU (2020), Searching for AF (2017)      |
| **Batch Normalization 사용** | ReLU, FReLU             | FReLU, Activation Ensembles (2017)                |
| **Residual 연결 포함**       | ReLU 계열, Swish        | Activation Functions in Deep Learning (2021)      |
| **Physically Informed ML**   | tanh, sine, adaptive    | Physics-Informed Machine Learning 실험 (2022)     |
| **온-policy RL**             | dSiLU, SiLU             | Sigmoid-Weighted Linear Units (2017)              |
| **음성 인식/번역**           | Tanh, PReLU, GELU       | Activation Functions in Deep Learning (2021)      |

### 3.3 논문 간 상충되는 결과

| 상충되는 관점                    | 논문의 주장                                | 관찰된 논문                                           |
| -------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| **ReLU 보편성 vs 특수화**        | ReLU 가 실용적 기본값                      | vs Adaptive activation 이 대부분의 문제에서 우위      |
| **Simple vs Complex activation** | 계산 단순성에서 ReLU 우위                  | vs GELU, Swish 이 복잡한 구조에서 더 나은 성능        |
| **Normalization 필수 vs 불필요** | BatchNorm 없이는 깊은 네트워크 수렴 어려움 | vs SNN: normalization-free 에도 깊은 네트워크 가능    |
| **Task-specific vs Universal**   | dataset/architecture-specific activation   | vs 범용 activation 으로 일반화 시도                   |
| **Smoothness vs Efficiency**     | Smoothness 가 일반화 성능에 유리           | vs Inverse square root, hardware-efficient activation |

### 3.4 데이터셋 또는 환경에 따른 성능 차이

| 데이터셋/환경                 | 유리한 활성화 함수       | 이유                                             |
| ----------------------------- | ------------------------ | ------------------------------------------------ |
| **CIFAR-10/100**(일반 이미지) | ReLU, Swish, GELU, PDELU | 표준 벤치마크, 다양한 AF 비교                    |
| **ImageNet**(대규모 분류)     | ReLU, Swish, Mish, SReLU | 대규모 데이터에서 안정적 성능                    |
| **STL-10**(소규모 이미지)     | GELU, CELU, Softplus     | 제한된 데이터에서 일반화 성능 중요               |
| **번역**(German-English)      | Tanh, SELU, PReLU        | RNN 기반, sequential 정보 보존                   |
| **MNIST**(초보자용)           | 거의 모든 AF 유효        | 간단한 작업에서는 차이 미미                      |
| **TIMIT**(음성)               | GELU, PReLU              | ASR 특화, 음성 신호 특성에 최적                  |
| **Atari**(강화학습)           | dSiLU, SiLU              | on-policy 환경, bootstrapped value approximation |
| **UCI**(다양한 분류)          | SNN, adaptive            | 다양한 문제 유형, normalized mapping             |
| **Tox21**(약물 발견)          | SNN, tanh                | 작은 dataset, tabular 데이터                     |

## 4. 추가 실험 및 검증 패턴

### 4.1 Ablation Study 패턴

| 논문명                                                                         | ablation 내용                              | 목적                        |
| ------------------------------------------------------------------------------ | ------------------------------------------ | --------------------------- |
| A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015) | identity matrix scale: I, 0.01I            | 문맥 길이별 최적 scale 탐색 |
| A Study on ReLU and Softmax in Transformer (2023)                              | scale factor 제거, normalization loss 제거 | 각 구성 요소의 기여도 분리  |
| Dynamic ReLU (2020)                                                            | spatial/channel-wise 공유 변형             | task 별 최적 변형 탐색      |
| Language Modeling with Gated Convolutional Networks (2016)                     | 문맥 길이: 30-40, 40+, GLU variants        | 문맥 길이 감수 분석         |
| Learning Activation Functions to Improve Deep Neural Networks (2014)           | hinge 개수 S=2, 학습된 activation 시각화   | S 민감도 분석               |
| Mish: A Self Regularized Non-Monotonic Activation Function (2019)              | 다양한 아키텍처 적용                       | robustness 검증             |
| FReLU (2017)                                                                   | ReLU+BN, ELU+BN, FReLU+BN                  | BN 호환성 검증              |
| Self-Normalizing Neural Networks (2017)                                        | SNN vs BatchNorm/LayerNorm/WeightNorm      | normalization 영향 검증     |

### 4.2 민감도 분석 패턴

| 논문명                                                                           | 분석 대상                                             | 결과                                   |
| -------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------- |
| A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015)   | learning rate: 10⁻⁹ ~ 10⁻¹, gradient clipping: 1-1000 | 최적 learning rate 및 clipping 값 탐색 |
| Deep Learning with S-shaped Rectified Linear Activation Units (2015)             | layer: top/bottom, layer별 activation variance        | 층별 activation 특성 분석              |
| Improving Deep Learning by Inverse Square Root Linear Units (2017)               | learning rate, loss function                          | 최적화 동역학 분석                     |
| Evolutionary Optimization of Deep Learning Activation Functions (2020)           | generation: 1-50, loss/accuracy 기반 진화             | generation별 성능 변화                 |
| Parametric Exponential Linear Unit for Deep Convolutional Neural Networks (2016) | layer별 a 파라미터: 0.5-2                             | layer별 parameter 효율성               |
| Searching for Activation Functions (2017)                                        | various backbones                                     | 발견된 activation 의 일반화성          |

### 4.3 조건 변화 실험 패턴

| 논문명                                                                                       | 조건 변화                                                  | 관찰 결과                        |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------- |
| A Study on ReLU and Softmax in Transformer (2023)                                            | sequence length: 128-2048                                  | 길이 1024 이상에서 ReLU 우위     |
| Activation Ensembles for Deep Neural Networks (2017)                                         | dataset: MNIST, ISOLET, CIFAR-100, STL-10                  | dataset-specific preference      |
| A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (2015)               | task: adding, MNIST, language modeling, speech             | task-dependent initialization    |
| Activation Functions: Comparison of Trends in Practice and Research for Deep Learning (2018) | 도메인: vision, NLP, speech, etc.                          | 도메인별 activation 사용 패턴    |
| Dynamic ReLU (2020)                                                                          | backbone: MobileNet/ResNet, task: classification/detection | task별 변형 선택                 |
| How important are activation functions in regression and classification? (2022)              | task: classification, PIML (회귀)                          | task-dependent activation 중요성 |

## 5. 실험 설계의 한계 및 비교상의 주의점

### 5.1 비교 조건의 불일치

| 문제                      | 설명                                            | 영향                                 |
| ------------------------- | ----------------------------------------------- | ------------------------------------ |
| **초기화 방법 불일치**    | 일부 논문: He initialization, 다른 논문: Xavier | 활성화 함수 효과 과소/과대 평가 가능 |
| **학습률/optimizer 차이** | Adam, SGD, AdaDelta 등 다양                     | 최적화 동역학에 따른 편향 가능성     |
| **데이터 증강 조건 차이** | 일부는 사용, 일부는 없음                        | generalization 성능 비교 불가        |
| **배치 크기 차이**        | 16, 32, 256 등 다양                             | 학습 동역학 차이                     |
| **Epoch 설정 차이**       | 17, 50, 200 등 다양                             | 과적합 정도 차이                     |

### 5.2 데이터셋 의존성

| 문제                         | 설명                          | 예시                              |
| ---------------------------- | ----------------------------- | --------------------------------- |
| **Simple dataset 과다 의존** | MNIST, CIFAR 기반 실험 위주   | 실제 응용과의 괴리 가능성         |
| **소규모 데이터셋 편향**     | 121 개 UCI task 대부분 소규모 | 대규모 데이터에서는 일반화 어려움 |
| **이미지 위주**              | MNIST/CIFAR/ImageNet 중심     | NLP/ASR/RL 데이터셋 상대적 부족   |

### 5.3 일반화 한계

| 한계              | 설명                                   | 영향                        |
| ----------------- | -------------------------------------- | --------------------------- |
| **아키텍처 제한** | CNN, FFN 중심, Transformer/ResNet 일부 | 현대 아키텍처 일반화 어려움 |
| **시간적 편향**   | 2013-2023까지, 초기 연구 위주          | 최신 SOTA와의 비교 어려움   |
| **domain bias**   | 이미지 분류 위주                       | 도메인 간 지식 이전 어려움  |

### 5.4 평가 지표의 한계

| 한계                              | 설명                                                          | 영향                                   |
| --------------------------------- | ------------------------------------------------------------- | -------------------------------------- |
| **Accuracy 만 의존**              | 많은 논문이 accuracy 만 보고                                  | qualitative 특성, robustness 평가 부재 |
| **computational cost 간과**       | FLOPs, parameter 효율성 일부만 보고                           | 실용적 배포 고려 부족                  |
| **statistical significance 부족** | p-value, confidence interval 미보급 결과의 신뢰성 평가 어려움 |                                        |
| **ablation 불일치**               | 각 논문별 ablation 설정 다름                                  | 구성 요소 기여도 분리 평가 어려움      |

## 6. 결과 해석의 경향

### 6.1 저자들이 결과를 해석하는 공통 경향

| 해석 경향                                          | 설명                                                | 예시                          |
| -------------------------------------------------- | --------------------------------------------------- | ----------------------------- |
| **"새로운 활성화 함수는 ReLU 를 대체한다"는 서사** | 대부분 "improvement", "outperform" 표현             | PReLU, Swish, GELU 논문들     |
| **"simple but effective" 강조**                    | 계산 단순성과 실용성 강조                           | ReLU, softsign 권장           |
| **"adaptive/learnable activation 우월"**           | 고정형 activation 을 넘어서는 접근                  | APL, PELU, Dynamic ReLU       |
| **"task-specific 설계 필요"**                      | 범용 activation 존재 vs specialization 필요         | dataset/architecture-specific |
| **"normalization-free 가능성"**                    | normalization 이 항상 필수는 아님                   | SNN 결과 해석                 |
| **"smoothness 가 일반화에 유리"**                  | derivative boundedness, continuously differentiable | CELU, GELU, Swish             |
| **"scaling/similarity 중요"**                      | parameter 조정 없이 family 전체 일괄 조정           | CELU, SReLU                   |

### 6.2 해석과 실제 관찰 결과의 차이

| 분야                    | 해석                                      | 실제 관찰                                           |
| ----------------------- | ----------------------------------------- | --------------------------------------------------- |
| **ReLU 보편성**         | "computational simplicity 와 실용적 우위" | 특수한 조건 (긴 시퀀스, deep narrow) 에서 ReLU 한계 |
| **Adaptive activation** | "고정형 함수 대체할 잠재력 보유"          | 대부분 improved but marginal improvement            |
| **Smooth activation**   | "loss landscape 더 잘 conditioned"        | 일부 계산 비용 증가로 trade-off                     |
| **Normalization**       | "deep network 수렴 필수"                  | SNN 에서 normalization-free 가능                    |
| **Smooth vs Hard**      | "smoothness 가 일반화"                    | hard threshold (ReLU) 도 여전히 강력                |

## 7. 종합 정리

전체 실험 결과를 종합하면, 활성화 함수 선택은 **task, dataset, 아키텍처, computational constraint**를 함께 고려해야 하는 복잡한 설계 문제임을 확인할 수 있다. **단순한 "best activation"이 존재하지 않으며**, 각 논문이 보고한 개선은 특정 조건에서만 유효했다.

핵심 발견은 다음과 같다:

1. **ReLU 가 여전히 기본값으로 강력**: 계산 효율성과 실용적 성능에서 입증, 하지만 dying ReLU, narrow 네트워크, 긴 시퀀스 등에서는 한계 존재
2. **Adaptive activation 이 고정형 함수를 대체**: APL, PELU, Dynamic ReLU, SNN 등 데이터에 맞춰 학습 가능한 활성화 함수가 일반화 성능 개선
3. **Smoothness 와 bounded derivative 가 일반화에 유리**: GELU, CELU, Swish 이 NLP, ASR 에서 일관된 성능, 하지만 계산 비용 증가
4. **Task-specific 설계의 중요성**: 이미지 분류, NLP, ASR, RL, PIML 등 도메인별 최적 활성화 함수 존재
5. **Architectural design 과의 통합 필요**: gating 구조 (GLU), normalization, initialization 과의 상호작용 고려

결론적으로, **활성화 함수 선택은 단순한 함수 선택 문제를 넘어 모델 설계의 일부**이며, **dataset 특성, 아키텍처 구조, computational budget**를 고려한 균형 잡힌 선택이 필요하다. 범용 activation 하나를 찾으려 하기보다, **task-specific activation metalearning**이 더 효과적인 방향임을 시사한다.
