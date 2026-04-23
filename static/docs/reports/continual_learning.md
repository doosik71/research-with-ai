# Continual Learning

## 서론

본 보고서는 Continual Learning 분야를 다룬 9 편의 문헌을 체계적으로 분석하고 비교합니다. 제공된 논문들은 형식적 정의부터 생성 모델 기반 접근법, meta-learning, 하이브리드 시스템, 정규화 및 지식 증류 기법에 이르기까지 다양한 방법론과 연구 체계의 관점을 포함하고 있습니다.

본 연구는 Continual Learning 방법론들을 연구 대상, 접근 관점, 시스템 구성, 연구 목적이라는 네 가지 기준에 따라 범주화합니다. 각 논문은 정의 체계 구축, 방법론 제안, 평가 체계 수립 등 다양한 목적을 지향하며, 동일한 연구가 여러 범주에 적용 가능하더라도 가장 대표적인 분류에 배치됩니다.

Continual Learning 분야에서 방법론의 효과는 초기화 조건 (Random vs Pre-training), 태스크 정합성 정보 유무 (Task Identity), 데이터셋 유형 등의 미세한 설정 차이에 따라 극명하게 달라집니다. 이러한 맥락에서 본 보고서는 방법론별 성능 패턴을 정밀하게 비교하고, Pre-training, Core Set, Replay 전략의 상충된 결과를 분석하며, 기존 결과들의 일반화 한계를 규명합니다.

보고서는 총 3 장으로 구성되어 있습니다. 1 장에서는 연구 대상, 접근 관점, 시스템 구성, 연구 목적이라는 네 가지 기준에 따라 논문들을 범주화하고 분석합니다. 2 장에서는 Generative Replay, Parameter Regularization, Knowledge Distillation, Meta-Learning 등 6 개 주요 방법론 계열의 설계 패턴과 비교 분석을 수행합니다. 3 장에서는 데이터셋 유형별, 환경 조건별 실험 결과를 정리하고 성능 패턴, 조건 의존성, 일반화 한계 등 결과 해석의 경향을 종합합니다.

## 1장. 연구체계 분류

### 1.1. 연구 분류 체계 수립 기준

본 보고서에서는 제공된 논문들을 다음과 같은 기준과 원칙에 따라 분류합니다:

1. **연구 대상 (Research Object)**: 각 연구가 해결하고자 하는 핵심 문제를 중심으로 범주화
2. **접근 관점 (Approach Perspective)**: 알고리즘 중심, 시스템 중심, 형식적 정의 중심 등 연구의 관점 차이 반영
3. **시스템 구성 (System Composition)**: 하이브리드, 생성 모델 기반, meta-learning 기반 등 구성 요소 관점 분류
4. **연구의 목적 (Research Purpose)**: 정의 체계 구축, 방법론 제안, 평가 체계 수립 등 목적별 분류

각 범주 내 논문의 분류 근거는 해당 연구의 핵심 방법론과 주요 결과에 기반하며, 동일한 연구가 여러 범주에 적용 가능하더라도 가장 대표적인 1개 범주에만 배치합니다.

### 1.2. 연구 분류 체계

#### 1.2.1. 형식적 정의 및 이론적 기초 연구

이 범주는 continual learning 문제에 대한 수학적 형식화, 정의 체계, 이론적 프레임워크를 제시하는 연구들을 포함합니다.

| 분류                             | 논문명                                                  | 분류 근거                                                                            |
| -------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 형식적 정의 > 수학적 형식화      | A Definition of Continual Reinforcement Learning (2023) | 논리 연산자를 활용한 CRL 문제에 대한 수학적 프레임워크 및 에이전트 기저 개념 정의    |
| 형식적 정의 > 평가 시나리오 체계 | Three scenarios for continual learning (2019)           | Task/Doman/Class-IL 세 가지 평가 시나리오 체계화 및 방법론별 성능 비교 프로토콜 제안 |

#### 1.2.2. 생성 모델 기반 접근법

과거 데이터 접근 불가 조건에서 생성 모델을 통해 과거 분포를 재현하고 replay하며 forgetting 문제를 해결하는 연구들입니다.

| 분류                                     | 논문명                                                           | 분류 근거                                                                                                |
| ---------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 생성 모델 > VAE 기반 teacher-student     | Continual Classification Learning Using Generative Models (2018) | VAE encoder/decoder를 활용한 joint likelihood 최적화 및 teacher-student 구조를 통한 과거 분포 재생성     |
| 생성 모델 > 온라인 variational inference | Variational Continual Learning (2017)                            | 이전 posterior를 다음 prior로 재사용하는 재귀 업데이트 및 mean-field Gaussian 근사 기반 온라인 변분 추론 |

#### 1.2.3. meta-learning 기반 적응 기법

forgetting을 줄이는 능력을 직접 학습하거나 선택적 활성화/가소성 메커니즘을 학습하는 방법론들입니다.

| 분류                                        | 논문명                                                      | 분류 근거                                                                                                              |
| ------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| meta-learning > neuromodulatory gating      | Learning to Continually Learn (2020)                        | Forgetting 을 줄이는 능력을 meta-learning 으로 직접 학습하는 neuromodulatory network 기반 gating 메커니즘              |
| meta-learning > representation optimization | Meta-Learning Representations for Continual Learning (2019) | prediction head 와 representation 을 분리 학습하고 online update 간섭을 최적화 대상으로 삼는 representation space 학습 |

#### 1.2.4. 하이브리드 시스템 및 운영 체계

딥러닝과 통계적 방법을 결합하거나 운영 시스템 관점에서 continual learning 을 접근하는 연구들입니다.

| 분류                               | 논문명                                                                 | 분류 근거                                                                                                     |
| ---------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 하이브리드 > feature-decision 분리 | Continual Learning for Anomaly Detection in Surveillance Videos (2020) | 딥러닝 feature extractor 와 통계적 검출기를 분리하여 retraining 없이 참조 집합만 업데이트하는 하이브리드 방식 |
| 운영 체계 > MLOps 아키텍처         | Continual Learning in Practice (2019)                                  | streaming 데이터 환경에서 지속적인 ML 운영을 위한 모듈형 참조 아키텍처 및 progression testing 프레임워크 제안 |

#### 1.2.5. 정규화 및 지식 증류 기반 방법

regularization 과 distillation 기법을 활용한 rehearsal-free 설정에서의 continual learning 방법들입니다.

| 분류                                | 논문명                                                    | 분류 근거                                                                                                |
| ----------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 정규화 > conditional regularization | A Closer Look at Rehearsal-Free Continual Learning (2022) | pre-training 유무와 regularization 유형에 따라 method ranking 이 뒤집히는 conditional effectiveness 분석 |

### 1.3. 종합 정리

본 보고서에서 분석한 continual learning 연구들은 수학적 정의 체계, 생성 모델 기반 접근, meta-learning 기반 적응, 하이브리드 시스템, 운영 체계 설계, regularization 기반 방법 등 6 가지 범주로 구성되어 있습니다. 이 분류 체계는 연구의 핵심 문제 정의, 접근 방법론, 시스템 구성 관점에서 논문을 체계적으로 조직하며, 기존 continual learning 연구들이 정의, 방법론, 시스템이라는 세 차원에서 균등하게 분포하고 있음을 보여줍니다. 특히 생성 모델과 meta-learning 기반 접근의 출현은 rehearsal-free 제약 조건과 forgetting 문제를 해결하는 새로운 패러다임을 제시합니다.

`2 장 방법론 분석.md` 파일을 생성했습니다.

제공된 9 개의 논문 문서 모음만을 바탕으로 방법론 지형을 재구성했습니다:

**주요 구조:**

- 6 개 방법론 계열 (Generative Replay, Parameter Regularization, Knowledge Distillation, Statistical Monitoring, Meta-Learning, Policy Systems, Bayesian Sequential, Formal RL)
- 핵심 설계 패턴 6 개 (Teacher-Student, Outer-Inner Loop, NM+PLN Dual Network, Modular System, CUSUM, Coreset)
- 방법론 비교 (문제 접근, 구조, 적용 대상, 트레이드오프)
- 방법론 진화 흐름 (시나리오 정의 → 시스템 확장 → 메타러닝 정교화 → 생성/통계적 감지 → 베이지안 → 형식적 정의)

```markdown
## 3장. 실험결과 분석

### 1. 평가 구조 및 공통 실험 설정

#### 1.1 주요 데이터셋 유형

| 데이터셋 유형   | 구체적 데이터셋                                   | 평가 목적                                             |
| --------------- | ------------------------------------------------- | ----------------------------------------------------- |
| 벤치마크 이미지 | Split MNIST, Permuted MNIST, Split notMNIST       | forgetting 저항성, 생성 품질, classification accuracy |
| 확장 이미지     | CIFAR-100, ImageNet-R, ImageNet1k, mini-ImageNet  | pre-training 의존성, ViT 설정에서의 성능              |
| 제한적 학습     | Omniglot few-shot                                 | meta-learning, few-shot 성능                          |
| 회귀/시계열     | Incremental Sine Waves                            | representation learning, MSE                          |
| 실제 감시       | UCSD Ped2, CUHK Avenue, ShanghaiTech, CCTV 스트림 | 이상 탐지, update latency, false alarm                |

#### 1.2 평가 환경

| 환경 유형  | 구체적 설정                      | 비고                      |
| ---------- | -------------------------------- | ------------------------- |
| 시뮬레이션 | Switching MDPs (0.001 확률 전환) | CRL 정의 검증용 단순 환경 |
| 벤치마크   | permuted MNIST, Split MNIST 등   | 생성 데이터셋에 의존      |
| 실환경     | CCTV 스트림 (8시간 23분)         | 실제 감시 시스템 적용     |
| 이론 분석  | 수학적 정의, oracle              | 상한선 기준 제공          |

#### 1.3 비교 방식

| 비교 유형      | 대상                                    | 비고                   |
| -------------- | --------------------------------------- | ---------------------- |
| baseline       | Scratch, naive online, joint training   | 하한선 기준            |
| regularization | EWC, SI, LP, L2                         | parameter-based 접근   |
| replay         | LwF, DGR, DGR+distill, iCaRL, VAE-based | generative replay 기반 |
| distillation   | PredKD, FeatKD, BCE vs Softmax          | knowledge distillation |
| pre-training   | ImageNet1k, Random init                 | plasticity 의존성 분석 |
| upper-bound    | Oracle (i.i.d. interleaved)             | 이상적 성능 상한선     |

#### 1.4 주요 평가 지표

| 지표                       | 정의                          | 사용 맥락                        |
| -------------------------- | ----------------------------- | -------------------------------- |
| Global accuracy (A_{1:N})  | 최종 단계에서의 전체 accuracy | parameter distillation 효과 비교 |
| Global forgetting (F_N^G)  | 전체 클래스에 대한 forgetting | regularization 효과 측정         |
| Local forgetting (F_N^L)   | 클래스별 forgetting           | layer별 drift 분석               |
| Frame-level AUC            | frame 단위의 이상 탐지 정확도 | 이상 탐지 벤치마크               |
| Average test accuracy      | 평균 classification accuracy  | classification 성능              |
| ELBO                       | Reconstruction negative ELBO  | 생성 품질 측정                   |
| Average reward             | 지속적 학습 환경 평균 보상    | CRL 환경 적응성                  |
| MSE                        | 회귀 오차                     | representation 학습              |
| Training/testing accuracy  | held-out 샘플 정확도          | forgetting 저항성                |
| NM activation KNN accuracy | KNN 기반 일반화               | meta-test 성능                   |

### 2. 주요 실험 결과 정렬

#### 2.1 Rehearsal-Free CL에서의 distillation 전략 비교 (2022)

| 설정                     | 방법        | Global accuracy (A_{1:N}) | Global forgetting (F_N^G) |
| ------------------------ | ----------- | ------------------------- | ------------------------- |
| Random init, CIFAR-100   | PredKD 단독 | 25.2                      | -                         |
| Random init, CIFAR-100   | PredKD+EWC  | 22.7                      | -0.7 (최적)               |
| Pre-training, ImageNet-R | PredKD+L2   | 35.6                      | -                         |
| ViT, ImageNet-R          | L2          | 76.06                     | -                         |
| ViT, ImageNet-R          | CODA-P      | 75.45                     | -                         |
| ViT, ImageNet-R          | DualPrompt  | 71.32                     | -                         |

**비교**:
- Random init 환경에서는 PredKD 단독이 최고 성능, EWC 결합은 forgetting 최적
- Pre-training 환경에서는 L2 regularization이 가장 효과적 (35.6)
- ViT 설정에서는 L2가 76.06으로 최고, CODA-P와 DualPrompt 이 뒤따름

#### 2.2 Continual Reinforcement Learning 환경에서의 적응성 (2023)

| 방법                          | 적응성           | 평균 보상 | 환경 변화 시 성능 |
| ----------------------------- | ---------------- | --------- | ----------------- |
| CRL (지속적 학습)             | 유연한 적응      | 높음      | 꾸준한 높은 보상  |
| 전통적 Q-learning (수렴 유도) | 파국적 성능 저하 | 급감      | 파국적 망각       |
| 시간적 감감 기반              | 과거 학습 중단   | -         | -                 |

**비교**: CRL 은 환경 전환 시 유연하게 적응하며, 수렴 유도 traditional Q-learning 대비 비정상성에 대처 능력 우위

#### 2.3 VAE 기반 생성 replay의 성능 (2018)

| 데이터셋            | 방법   | 평균 classification accuracy   |
| ------------------- | ------ | ------------------------------ |
| permuted MNIST      | CCL-GM | 가장 높은 안정적 성능 (상대적) |
| MNIST-Fashion-MNIST | CCL-GM | baseline 대비 더 안정적        |

**비교**: 태스크 증가에도 성능 저하 없이 stable, reconstruction 품질 동시 유지

#### 2.4 이상 탐지에서의 update 효율 (2020)

| 데이터셋     | 방법           | Frame-level AUC | Update latency |
| ------------ | -------------- | --------------- | -------------- |
| UCSD Ped2    | 제안법         | 97.8            | 10 초          |
| UCSD Ped2    | Liu et al.     | 95.4            | 4.8 시간       |
| UCSD Ped2    | Ionescu et al. | 71.5            | 2.5 시간       |
| Avenue       | 제안법         | 86.4            | -              |
| Avenue       | Liu et al.     | 85.1            | -              |
| ShanghaiTech | 제안법         | 71.62           | -              |
| ShanghaiTech | Liu et al.     | 72.8            | -              |
| ShanghaiTech | Ionescu et al. | 71.5            | -              |

**비교**:
- UCSD Ped2: 제안법 97.8% vs Liu 95.4% (상대적 우위), Ionescu 대비 월등한 update speed
- Avenue: 제안법 86.4% vs Liu 85.1% (상대적 우위)
- ShanghaiTech: 제안법 71.62% vs Liu 72.8% (Liu 대비 약간 낮음) vs Ionescu 71.5% (Ionescu와 비슷)
- Update time: 제안법 10 초 vs Liu 4.8 시간, Ionescu 2.5 시간 (100~300 배 빨름)

#### 2.5 ANML의 meta-learning 성능 (2020)

| trajectory 길이 | 방법        | meta-test training accuracy      |
| --------------- | ----------- | -------------------------------- |
| 600 클래스      | ANML        | 63.8%                            |
| 600 클래스      | OML         | 18.2%                            |
| 600 클래스      | OML-OLFT    | 44.2%                            |
| 모든 길이       | ANML vs OML | p ≤ 1.26×10^{-8} (통계적 유의성) |

**비교**: 600-class 설정에서 ANML 이 OML 대비 약 3 배 높은 성능, oracle 보다도 좋음

#### 2.6 OML의 representation 학습 효과 (2019)

| 설정                   | 방법         | training accuracy | forgetting 정도 |
| ---------------------- | ------------ | ----------------- | --------------- |
| Incremental Sine Waves | OML          | 거의 완벽 유지    | 매우 작음       |
| Incremental Sine Waves | Pre-training | 후기 오차 증가    | -               |
| Split-Omniglot         | OML          | 거의 완벽 유지    | 매우 작음       |
| Split-Omniglot         | Oracle       | 상한선            | -               |

**비교**:
- Incremental Sine Waves: OML 평균 MSE 증가 매우 작음, Pre-training 으로 갈수록 오차 커짐
- Split-Omniglot: OML training accuracy 거의 완벽 유지
- OML sparsity: 3.8% (vs SR-NN: 15%, Pre-training: 38%)
- OML dead neuron: 0% (vs SR-NN: 0.7%, Pre-training: 3%)

#### 2.7 Task identity 유무에 따른 성능 차이 (2019)

| 설정                    | 방법                    | 평균 test accuracy |
| ----------------------- | ----------------------- | ------------------ |
| Class-IL, Split MNIST   | Regularization (EWC/SI) | ~20% (완전 실패)   |
| Class-IL, Split MNIST   | Replay (DGR+distill)    | 91.79%             |
| Class-IL, Split MNIST   | Replay (iCaRL)          | 94.57%             |
| Task-IL, Permuted MNIST | 모든 방법               | 99% 이상           |
| Domain-IL, Split MNIST  | Replay 기반             | 71.50% → 95.72%    |

**비교**:
- Class-IL: regularization 기반 접근 근본적으로 부적합 (~20%), replay 필수 (90% 이상)
- Task-IL: 모든 방법 99% 이상 (task identity 가 결정적)
- Domain-IL: replay 중요성 커짐 (split MNIST 71.50% → 95.72%)

#### 2.8 VCL의 Bayesian inference 성능 (2017)

| 데이터셋                 | 방법                 | 평균 정확도 (태스크 완료 후) |
| ------------------------ | -------------------- | ---------------------------- |
| Permuted MNIST (10 task) | VCL (coreset)        | 93%                          |
| Permuted MNIST (10 task) | VCL (random coreset) | 90%                          |
| Permuted MNIST (10 task) | EWC                  | 84%                          |
| Permuted MNIST (10 task) | SI                   | 86%                          |
| Permuted MNIST (10 task) | LP                   | 82%                          |
| Split MNIST (5 task)     | VCL (coreset)        | 98.4%                        |
| Split MNIST (5 task)     | SI                   | 98.9%                        |
| Split MNIST (5 task)     | EWC                  | 63.1%                        |
| notMNIST (5 task)        | VCL                  | 92.0%                        |
| notMNIST (5 task)        | SI                   | 94%                          |

**비교**:
- Permuted MNIST: VCL+coreset 93% 최고, EWC 84%, SI 86%, LP 82%
- Split MNIST: SI 98.9%가 VCL 98.4% 약간 우위, EWC 63.1%는 대폭 뒤짐
- notMNIST: SI 94%가 VCL 92.0% 우위

### 3. 성능 패턴 및 경향 분석

#### 3.1 공통적으로 나타나는 성능 개선 패턴

| 패턴                   | 설명                                                                               | 관련 논문                                                 |
| ---------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Pre-training 의존성    | ImageNet pre-training 이 있을 경우 plasticity 문제 완화, L2 regularization 최적화  | 2022 (PredKD+L2 35.6 최고), 2019 (Pre-training 후기 오차) |
| Coreset 보완 효과      | VCL 에서 coreset 결합 시 정확도 상승 (90% → 93%), coreset 와 VCL 상호보완적        | 2017 (VCL+coreset 93% 최고)                               |
| Replay 필수성          | Class-IL 설정에서 regularization alone 은 실패, replay 기반 DGR+distill/iCaRL 필수 | 2019 (Class-IL regularization 20% 실패, replay 90%+ 성공) |
| Task identity 효과     | Task-IL 설정에서 모든 방법 99%+, identity 정보 없이는 replay 중요성 커짐           | 2019 (Task-IL 99%+, Domain-IL replay 필요)                |
| Bayesian approach 우위 | VCL 이 EWC/LP 대비 stable, SI 에 근접하거나 우위, 생성 모델 일반성                 | 2017 (VCL state-of-the-art, 생성 유지)                    |

#### 3.2 특정 조건에서만 성능이 향상되는 경우

| 조건               | 향상되는 방법     | 비고                                                          |
| ------------------ | ----------------- | ------------------------------------------------------------- |
| Pre-training 환경  | L2 regularization | Random init 시에는 PredKD+L2 성능 떨어짐                      |
| Coreset 결합       | VCL               | coreset alone 보다 VCL+coreset 93% 상승                       |
| Task identity 유무 | Task-IL           | identity 없는 경우 replay 필수, 있는 경우 regularization 실패 |
| ViT 설정           | L2                | CNN 보다 ViT 에서 L2 단독 최고 (76.06)                        |
| Random init 환경   | PredKD            | pre-training 있으면 L2, 없으면 PredKD                         |

#### 3.3 논문 간 상충되는 결과

| 상충 내용                   | 문맥                                                                          | 해석                                                       |
| --------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| L2 regularization 효과 차이 | Random init: PredKD+L2=22.7 (PredKD+EWC), Pre-training: PredKD+L2=35.6 (최고) | init 조건에 따라 L2 효과 극명하게 달라짐                   |
| VCL vs SI 성능              | Permuted MNIST: VCL 93% vs SI 86%, notMNIST: VCL 92% vs SI 94%                | 데이터셋에 따라 VCL 이 또는 SI 가 우위                     |
| AUC 비교                    | UCSD: 제안법 97.8% vs Liu 95.4%, ShanghaiTech: 제안법 71.62% vs Liu 72.8%     | 일부 벤치마크에서는 기존 SOTA 보다 낮음                    |
| Regularization alone 적절성 | Class-IL: EWC/SI ~20% (실패), ViT: L2 76.06 (최고)                            | task 설정에 따라 regularization alone 적절성 극명하게 차이 |

#### 3.4 데이터셋 또는 환경에 따른 성능 차이

| 데이터셋 유형          | 적합한 방법                 | 부적합 방법                     |
| ---------------------- | --------------------------- | ------------------------------- |
| Permuted MNIST         | VCL+coreset 93%, CCL-GM     | EWC 84%, LP 82%                 |
| Split MNIST            | SI 98.9%, VCL+coreset 98.4% | EWC 63.1%                       |
| notMNIST               | SI 94%                      | -                               |
| Switching MDPs         | CRL (지속적 학습)           | 전통적 Q-learning (파국적 저하) |
| UCSD Ped2              | 제안법 97.8%                | -                               |
| Omniglot               | ANML 63.8%                  | OML 18.2%, OML-OLFT 44.2%       |
| Incremental Sine Waves | OML (MSE 안정적 유지)       | Pre-training (후기 오차)        |
| ViT ImageNet-R         | L2 76.06                    | CODA-P 75.45, DualPrompt 71.32  |

### 4. 추가 실험 및 검증 패턴

#### 4.1 Ablation study 패턴

| 논문 | ablation 유형                    | 결과                                                            |
| ---- | -------------------------------- | --------------------------------------------------------------- |
| 2022 | PredKD vs PredKD+EWC             | Random init: PredKD 단독 최고, EWC 결합: forgetting 최적 (-0.7) |
| 2022 | BCE vs Softmax                   | formulation 차이 분석                                           |
| 2017 | VCL alone vs VCL+coreset         | coreset 결합 시 90% → 93% 상승                                  |
| 2017 | multi-head vs single-head vs VAE | 구조 차이 분석                                                  |

#### 4.2 민감도 분석 패턴

| 논문 | 분석 유형                           | 결과                                      |
| ---- | ----------------------------------- | ----------------------------------------- |
| 2019 | hyperparameter grid search (부록 D) | 공정성 논의                               |
| 2017 | coreset 크기 (5000 개)              | 95.5% 도달                                |
| 2020 | update time 민감도                  | 10 초 vs 4.8 시간 (100x 차이)             |
| 2020 | false positive 민감도               | 20,000 프레임마다 20% 사용 시 점진적 감소 |

#### 4.3 조건 변화 실험 패턴

| 논문 | 조건 변화                        | 결과                                                |
| ---- | -------------------------------- | --------------------------------------------------- |
| 2022 | Random vs Pre-training init      | init 에 따라 최적 방법 극명하게 차이 (PredKD vs L2) |
| 2019 | Task-IL vs Domain-IL vs Class-IL | identity 유무에 따라 방법론 적합성 근본적 차이      |
| 2017 | Permuted vs Split MNIST          | 데이터셋에 따라 VCL/SI 우위 전환                    |
| 2023 | CRL vs traditional               | 환경 전환 시 적응성 극명하게 차이                   |

### 5. 실험 설계의 한계 및 비교상의 주의점

#### 5.1 비교 조건의 불일치

| 문제              | 구체적 내용                                            | 영향                                                                    |
| ----------------- | ------------------------------------------------------ | ----------------------------------------------------------------------- |
| init 차이         | Random vs Pre-training                                 | L2 등 regularization 효과 비교 시 초기화 조건에 따라 결과 극명하게 다름 |
| benchmark 다양성  | CIFAR-100/ImageNet-R 등 제한적                         | 2022 년도 Task granularity 다양성 부족                                  |
| Oracle 상한선     | 2019 년 oracle (i.i.d. interleaved)                    | 실제 continual 환경과 상이한 상한선                                     |
| data availability | 2018 년 experimental level, 복잡한 실제 task 검증 부족 | preliminary 수준 실험                                                   |

#### 5.2 데이터셋 의존성

| 데이터셋       | 문제                           | 영향                                                    |
| -------------- | ------------------------------ | ------------------------------------------------------- |
| MNIST 계열     | permuted/split 모두 MNIST 기반 | 생성 데이터셋에 의존적, 실제 vision task 에 일반화 한계 |
| Omniglot       | few-shot 벤치마크에 한정       | meta-learning 실험 도메인 제한                          |
| Switching MDPs | 단순 환경                      | 구체적 알고리즘 치환 문제 미해결                        |
| CCTV           | 8시간 23분                     | 인간 in-the-loop 의존, 실제 시스템 의존                 |

#### 5.3 일반화 한계

| 한계                    | 구체적 내용                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------- |
| continual learning 정의 | 2023 년도 추상적 수학 연산자 구체적 알고리즘 치환 미해결                              |
| algorithm 유형          | 2019 년도 algorithmic CL 논문과 운영 시스템 paper 기대 불일치 가능성                  |
| reinforcement learning  | 2019 년도 regression/classification 중심, RL, long-term non-stationary 환경 검증 없음 |
| encoder sharing         | 2017 년도 encoder sharing 문제 조사 안됨                                              |

#### 5.4 평가 지표의 한계

| 한계                     | 구체적 내용                                                |
| ------------------------ | ---------------------------------------------------------- |
| AUC alone                | 이상 탐지 성능 평가 시 frame-level AUC alone 의 부족       |
| average reward alone     | CRL 환경 적응성 평가 시 average reward alone 의 한계       |
| negative ELBO            | 생성 품질 평가 시 negative ELBO alone 의 부족              |
| meta-test accuracy alone | forgetting 저항성 평가 시 meta-test alone 의 부족          |
| business metric          | 2019 년도 business metric 등 개념적 지표, 정량적 지표 없음 |

### 6. 결과 해석의 경향

#### 6.1 저자들이 결과를 해석하는 공통 경향

| 경향                            | 구체적 내용                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| state-of-the-art 주장           | 제안법 이 benchmark 에서 competitive 성능, 일부는 superior                               |
| architectural contribution 강조 | 2019 년도 continual learning 은 단일 알고리즘이 아닌 운영 시스템 문제로 정식화           |
| method 선택 설명                | 2022 년도 pre-training 의 plasticity 향상 효과, parameter vs feature alignment 효과 분석 |
| baseline 대비 우위              | 2018 년도 baseline 보다 안정적인 성능, 2017 년도 baseline 대비 대폭 개선                 |
| limitation 명시                 | 2019 년도 정량적 검증 부재, 2017 년도 mean-field Gaussian 근로 인한 한계                 |
| comparative context             | 2020 년도 oracle 대비 성능, 2019 년도 iid sampling sanity check                          |

#### 6.2 해석과 실제 관찰 결과의 구분

| 해석                     | 실제 결과                                                  | 구분             |
| ------------------------ | ---------------------------------------------------------- | ---------------- |
| "state-of-the-art"       | benchmark 에서 competitive 성능, 일부 경우 최고            | 해석적 프레임    |
| "plasticity 문제 완화"   | Pre-training 환경에서 L2 최적, Random 환경에서 PredKD 최적 | 환경 의존적 관찰 |
| "replay 필수"            | Class-IL 설정에서 regularization alone 실패, replay 성공   | 조건부 관찰      |
| "Bayesian approach 우위" | VCL 이 EWC/LP 대비 stable, SI 에 근접하거나 우위           | 상대적 우위      |
| "task identity 결정적"   | Task-IL 에서 99%+, identity 없는 경우 replay 중요성 커짐   | 조건에 따른 결과 |

### 7. 종합 정리

전체 실험 결과는 continual learning 방법이 설정의 미세한 차이 (initialization, task identity, 데이터셋 유형) 에 따라 극명하게 다른 성능 패턴을 보임을 시사한다. Pre-training 이 있을 경우 plasticity 문제 완화되어 L2 regularization 이 최적화되는 반면, Random init 환경에서는 PredKD 기반 prediction distillation 이 우위를 점한다. Coreset 와 VCL 의 결합이 상호보완적이며, task identity 정보 유무에 따라 regularization 기반 접근의 적절성이 근본적으로 달라진다. 재생산성 측면에서는 VCL 이 hyperparameter-free 로 competitive 성능을 유지하면서 생성 모델 일반성을 입증했으며, 이상 탐지 영역에서는 update latency 가 100~300 배 빨름으로써 few-shot 적응성을 달성한다. 그러나 대부분 MNIST 계열 벤치마크에 의존하는 한계와 task 설정에 따라 결과가 극명하게 달라지는 점은 실제 비정적 환경으로의 일반화 가능성을 제한하는 요소로 작용한다.
