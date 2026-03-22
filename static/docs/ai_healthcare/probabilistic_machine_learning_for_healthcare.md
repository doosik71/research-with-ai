# Probabilistic Machine Learning for Healthcare

## 1. Paper Overview

이 논문은 **의료 데이터 분석에서 probabilistic machine learning이 왜 중요한가**를 체계적으로 설명하는 서베이 논문이다. 저자들은 의료 데이터가 단순히 예측 정확도만으로 다뤄질 수 있는 대상이 아니라, 본질적으로 **불확실성(uncertainty), 누락(missingness), 검열(censoring), 분포 변화(data shift)**를 동반하는 관측 데이터라고 본다. 따라서 단일 점추정(point estimate)만 내놓는 deterministic model보다, 결과의 분포와 불확실성을 함께 표현하는 probabilistic model이 의료 의사결정에 더 적합하다고 주장한다. 특히 이 논문은 예측 모델 단계에서의 calibration과 missing data 문제뿐 아니라, latent variable을 이용한 phenotyping, clinical use case를 위한 generative model, treatment planning을 위한 reinforcement learning까지 probabilistic 관점을 확장해 설명한다.  

이 문제가 중요한 이유는 의료에서 “평균적으로 맞는 예측”만으로는 충분하지 않기 때문이다. 논문 서두의 예처럼, 같은 평균 생존시간을 갖는 두 환자군이라도 분산이 다르면 환자와 의료진의 계획은 완전히 달라질 수 있다. 즉 의료에서는 “무엇이 가장 가능성 높은가?”뿐 아니라 “얼마나 불확실한가?”, “어떤 경우들이 가능한가?”가 중요하다. 저자들은 probabilistic model이 이러한 정보를 제공함으로써 환자 상담, 치료 계획, 모델 유지보수, 의료 AI의 신뢰성 향상에 기여한다고 본다.

## 2. Core Idea

이 논문의 중심 아이디어는 **의료 머신러닝을 예측 문제로만 보지 말고, 데이터 생성 분포(data generating distribution)를 명시적으로 다루는 문제로 보자**는 것이다. 일반적인 deterministic model은 입력 $\mathbf{x}$에 대해 출력 $y$의 평균적 예측값이나 점추정값을 주지만, probabilistic model은 $p_\theta(y \mid \mathbf{x})$ 자체를 학습함으로써 평균뿐 아니라 분산, tail behavior, 신뢰도까지 함께 다룰 수 있다. 이 차이는 의료에서 특히 중요하다. 왜냐하면 의료 데이터는 관측이 불완전하고, 시간이 지나며 환경이 바뀌며, 실제 임상 행동은 예측값 그 자체보다도 risk distribution에 더 민감하기 때문이다.

또한 이 논문은 probabilistic model을 단순히 “확률 출력이 있는 classifier” 수준으로 한정하지 않는다. 저자들은 probabilistic perspective가 의료 AI 파이프라인 전반에 걸쳐 작동한다고 본다. 예를 들어 predictive model 단계에서는 missing values, censoring, calibration, uncertainty, data shift, fairness 문제를 다루는 데 쓰이고, 더 나아가 latent variable model을 이용해 질병 phenotype를 추론하거나, generative model로 임상 시뮬레이션과 데이터 생성을 하거나, RL에서 치료 정책을 배우는 데도 활용될 수 있다고 정리한다. 즉, 이 논문의 novelty는 새로운 알고리즘이 아니라 **확률적 관점을 의료 AI 전체에 관통하는 공통 언어로 재배치**한 데 있다.  

## 3. Detailed Method Explanation

이 논문은 새로운 단일 알고리즘을 제안하는 논문이 아니라 **review + conceptual framework** 성격의 논문이다. 따라서 “방법”은 하나의 모델 구조나 loss function보다, probabilistic machine learning이 의료 문제를 어떻게 재구성하는지에 대한 설명으로 이해하는 것이 적절하다.

### 3.1 Deterministic model과 probabilistic model의 구분

논문은 가장 먼저 deterministic model과 probabilistic model의 차이를 수식으로 설명한다. deterministic model은 $g_\theta(\mathbf{x})$처럼 입력 feature $\mathbf{x}$를 받아 응답 $y$의 예측값을 내놓는 함수이며, 예를 들어 squared error를 최소화하여 학습할 수 있다.

$$
\mathbb{E}_{\mathbf{x}, y \sim F}\left[\left(g*\theta(\mathbf{x}) - y\right)^2\right]
$$

이 경우 잘 학습된 모델은 대체로 $y$의 기대값에 가까운 값을 준다. 하지만 응답 분포 자체는 표현하지 못한다. 반면 probabilistic model은 $p_\theta(y \mid \mathbf{x})$처럼 **조건부 확률분포 자체**를 학습하며, 로그우도를 최대화하는 식으로 학습할 수 있다.

$$
\mathbb{E}_{\mathbf{x}, y \sim F}\left[\log p*\theta(y \mid \mathbf{x})\right]
$$

이렇게 학습된 모델은 평균값뿐 아니라 분산과 다른 통계량까지 제공할 수 있다. 논문이 강조하는 핵심은 의료에서 바로 이 차이가 임상적 의미를 만든다는 점이다. 예측 평균이 같더라도 분산이 다르면 의사결정은 달라진다.

### 3.2 Predictive model pipeline에서의 확률적 접근

논문의 3장은 의료 예측 모델 구축 과정에서 probabilistic model이 어떤 문제를 다루는지 정리한다. TOC 기준으로 다루는 핵심 항목은 다음 여섯 가지다.

* Missing Values
* Censoring
* Calibration
* Uncertainty
* Data Shift
* Fairness

이 구성 자체가 이 논문의 중요한 설계다. 즉, probabilistic ML은 단지 “uncertainty를 뽑아내는 모델”이 아니라, **의료 예측 파이프라인의 여러 실패 모드에 대응하는 방법론 집합**으로 제시된다.

#### (a) Missing Values

논문은 의료 데이터가 본질적으로 observational data이기 때문에 누락값이 매우 흔하다고 설명한다. 예를 들어 당뇨 환자의 longitudinal visit 기록에서는 모든 biomarker가 매 방문마다 측정되지 않을 수 있고, 환자마다 방문 횟수도 다르다. 이때 단순히 complete case만 남기면 표본 수가 지나치게 줄거나 편향이 생길 수 있다. probabilistic model은 **missingness mechanism 자체를 모델링**할 수 있기 때문에 이 문제를 더 잘 다룰 수 있다.

논문은 두 방향을 구분한다. 예측 성능이 가장 중요할 때는 observation indicator를 feature로 직접 넣어 missingness 자체를 신호로 사용할 수 있고, parameter estimation이 중요할 때는 posterior predictive distribution을 사용하는 imputation, 특히 multiple imputation using chained equations 같은 접근이 중요하다고 설명한다. 또한 MCAR, MAR, MNAR 구분을 명확히 하면서, 특히 MNAR는 추가 가정 없이 일반적인 imputation으로 해결되지 않는다고 지적한다. 이 부분은 실무적으로 매우 중요하다. 의료 데이터에서 결측은 “실수로 빠진 값”이 아니라, 종종 **측정되지 않았다는 사실 자체가 환자 상태와 연결**되기 때문이다.

#### (b) Censoring

의료 라벨은 종종 미래의 사건에 달려 있기 때문에 censoring 문제가 흔하다. 생존시간, 질병 발생 시점, 재입원 시점 같은 outcome은 모든 환자에게 완전히 관측되지 않는다. 논문은 probabilistic survival model이 censoring interval에 들어갈 확률을 적분으로 계산함으로써 censored observation을 직접 다룰 수 있다고 설명한다. 예를 들어 censoring time이 $c$인 환자의 likelihood는 다음처럼 표현된다.

$$
\int_c^\infty p(a \mid \mathbf{x}), da
$$

즉, event time을 단순 회귀로 예측하는 대신 survival function 자체를 모델링하는 것이 더 타당하다는 것이다. 논문은 deep survival analysis와 Brier score 기반 평가, inverse probability of censoring adjustment, interval censoring 문제까지 언급하며, censoring을 uncertainty modeling 문제로 본다.

#### (c) Calibration

논문 초록과 목차에서 calibration을 핵심 challenge로 명시한다. 의료 risk score는 단순히 순위가 잘 맞는 것만으로 부족하고, 예측 확률이 실제 발생 빈도와 맞아야 한다. 즉 20% risk라고 예측한 환자군에서 실제로 약 20%가 사건을 겪어야 한다. 논문은 probabilistic model이 이런 risk estimate의 calibration을 다루는 데 적합하다고 본다. 이는 임상에서 threshold-based decision making이 많기 때문에 매우 중요하다. calibration이 안 된 모델은 AUC가 높아도 실제 치료 경로를 왜곡할 수 있다.  

#### (d) Uncertainty, Data Shift, Fairness

목차 수준에서 보면 논문은 uncertainty, data shift, fairness를 각각 독립된 subsection으로 다룬다. 이 배치는 저자들의 시각을 잘 보여준다. uncertainty는 의료 AI가 “모르는 것을 아는 것”을 가능하게 하고, data shift는 시간·병원·인구집단 변화에 따른 분포 이동을 탐지하고 대응하는 데 중요하며, fairness는 서로 다른 환자 집단에서 risk estimation이 어떻게 달라지는지를 점검하는 문제다. 이 셋을 probabilistic perspective로 묶는 것은, 의료 AI를 단순 정확도 경쟁이 아니라 **distribution-aware modeling**으로 보려는 논문의 철학과 맞닿아 있다.

### 3.3 Latent Variable Model을 이용한 Phenotyping

논문 4장은 **phenotyping with latent variables**를 다룬다. 세부 항목은 다음과 같다.

* Phenotypic Matching
* Uncovering Hidden Phenotypes
* Semi-supervised Phenotyping with Anchors

이는 probabilistic ML이 예측 문제를 넘어, **관측된 EHR나 임상 feature 뒤에 숨어 있는 질병 subtype, 환자군 구조, 잠재 상태를 추론**하는 데 쓰일 수 있음을 보여준다. latent variable model은 관측되지 않는 phenotype를 $\mathbf{z}$ 같은 숨은 변수로 두고, 관측 데이터와의 결합분포를 모델링한다. 저자들이 이 섹션을 별도로 둔 것은 의료에서 label이 불완전하거나 거칠고, 실제 임상적으로 의미 있는 subgroup이 표면에 드러나지 않는 경우가 많기 때문이다.

### 3.4 Generative Model

논문 5장은 **generative models**를 다룬다. 서론에서도 probabilistic model이 simulation과 scientific discovery, 예를 들어 drug development에 쓰일 수 있다고 밝힌다. generative model의 핵심 역할은 단순 classification이 아니라, **임상 데이터 분포를 모델링하고 샘플을 생성하며, counterfactual-like reasoning이나 simulation을 가능하게 하는 것**이다. 의료에서는 synthetic data generation, trajectory simulation, clinical scenario generation 같은 응용과 연결될 수 있다. 이 논문은 generative modeling을 의료 예측의 보조 수단이 아니라 독립적 임상 도구로 제시한다는 점이 인상적이다.

### 3.5 Reinforcement Learning for Treatment Planning

논문 6장은 probabilistic 관점이 **treatment planning을 위한 reinforcement learning**에도 적용된다고 설명한다. 세부 항목은 다음과 같다.

* Model-based RL
* Stochastic Policies
* Partial Observability

이 구성은 매우 의료적이다. 치료는 한 번의 분류가 아니라, 시간에 따라 상태가 변하는 환자에게 순차적으로 intervention을 선택하는 문제다. 따라서 RL은 본질적으로 probabilistic transition과 partial observability를 다루는 sequential decision problem이 된다. 논문은 diabetes와 sepsis management에서 probabilistic approach가 유망함을 서론에서 언급하고, 본문에서는 model-based RL, stochastic policy, partial observability 같은 요소를 통해 그 이유를 설명하는 구조로 보인다. 즉, 이 논문은 probabilistic ML의 범위를 supervised prediction에만 한정하지 않고, **의료 의사결정 최적화 문제 전체**로 확장한다.

## 4. Experiments and Findings

이 논문은 새로운 모델을 제안하고 benchmark 실험을 수행하는 실험 논문이 아니라 **review article**이다. 따라서 datasets, baselines, metrics를 통일된 조건에서 직접 비교하는 구조는 아니다. 이 논문의 실질적 산출물은 정량적 SOTA가 아니라, **어떤 의료 문제에서 probabilistic modeling이 왜 필요한지에 대한 구조화된 지도**다.  

논문이 실제로 보여주는 주요 발견은 다음과 같이 요약할 수 있다.

첫째, 의료에서는 deterministic prediction만으로는 불충분하며, **분포와 불확실성의 명시적 모델링이 실제 임상 의미를 가진다**.
둘째, probabilistic ML은 predictive model 단계의 missingness, censoring, calibration 문제 해결에 특히 유용하다.
셋째, probabilistic perspective는 phenotyping, generative simulation, treatment planning처럼 예측을 넘어서는 영역에서도 강력하다.
넷째, probabilistic model은 더 많은 정신적·계산적 비용이 들 수 있지만, 최근 toolkit 발전으로 이 부담이 줄어들고 있다고 평가한다.

즉, 이 논문의 “findings”는 성능 수치라기보다 **probabilistic machine learning이 의료 전반에서 갖는 역할 분류와 설계 논리**라고 보는 것이 맞다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **probabilistic ML을 의료 문제의 본질적 요구와 직접 연결했다는 점**이다. 많은 개론서가 확률 모델을 일반적 수학 도구로 설명하는 데 그치지만, 이 논문은 missing data, censoring, calibration, uncertainty, phenotype discovery, treatment planning 같은 구체적 의료 난제에 하나씩 매핑한다. 그래서 독자는 “왜 의료에서 확률이 필요한가”를 추상적 차원이 아니라 실무적 맥락에서 이해하게 된다.  

두 번째 강점은 **범위의 균형감**이다. predictive model pipeline의 문제를 다루면서도, latent variable model, generative model, RL까지 포함해 probabilistic ML의 외연을 자연스럽게 확장한다. 이 덕분에 이 논문은 단순 통계 개론이 아니라, probabilistic healthcare ML 전반의 conceptual survey로 기능한다.

세 번째 강점은 **의료 의사결정과 uncertainty를 강하게 연결한다는 점**이다. 평균값이 아닌 분포를 주는 모델이 환자·의료진의 planning에 더 유용하다는 서론의 예시는 매우 설득력 있다. 이는 explainability와는 다른 차원의 임상적 유용성이다.

### Limitations

한편 한계도 분명하다.

첫째, 이 논문은 review이므로 **개별 방법론의 수학적 깊이나 empirical benchmark 비교는 제한적**이다. 예를 들어 어떤 calibration method가 어떤 의료 조건에서 더 좋은지, 어떤 latent variable approach가 어떤 phenotype task에 더 적합한지에 대한 정량적 비교는 논문의 중심이 아니다.

둘째, 논문 자체가 broad survey이기 때문에 **구현 수준의 지침은 상대적으로 적다**. probabilistic perspective가 왜 중요한지는 잘 설명하지만, 실제 hospital deployment나 production monitoring에서 어떤 exact workflow를 택해야 하는지는 후속 문헌이 더 필요하다.

셋째, probabilistic model의 계산 비용과 인지적 복잡성도 논문이 직접 인정하는 한계다. 저자들은 probabilistic model이 non-probabilistic approach보다 더 큰 mental and computational labor를 요구할 수 있다고 언급한다.

### Critical Interpretation

비판적으로 보면, 이 논문은 “확률모형이 의료에서 유용하다”는 점을 설득력 있게 보여주지만, 어떤 독자에게는 다소 **프레임워크 중심**으로 느껴질 수 있다. 즉, 최신 deep probabilistic architecture를 자세히 배우기 위한 논문이라기보다는, **의료 AI를 확률적 사고로 다시 정렬하는 논문**에 가깝다. 그럼에도 이 점이 דווקא 장점이기도 하다. 의료 데이터는 실제로 noisy, incomplete, censored, shifting, unfair할 수 있기 때문에, probabilistic perspective는 단순한 옵션이 아니라 의료 AI의 기본 태도가 되어야 한다는 메시지를 준다.  

## 6. Conclusion

이 논문은 probabilistic machine learning이 의료 데이터 분석에서 갖는 의미를 **예측, phenotype discovery, 생성, 순차적 치료계획**이라는 네 축으로 정리한 포괄적 review다. 핵심 기여는 다음과 같다. 첫째, deterministic model과 probabilistic model의 차이를 의료적 의사결정 맥락에서 명확히 설명했다. 둘째, predictive pipeline의 missing values, censoring, calibration, uncertainty, data shift, fairness 문제를 probabilistic perspective로 재해석했다. 셋째, latent variable model을 통한 phenotyping, generative model, RL-based treatment planning까지 확장해 probabilistic ML의 의료 응용 지형을 넓게 제시했다.  

실무적으로 이 논문은 “어떤 모델이 최고 성능인가”를 알려주는 논문보다, **의료 AI를 설계할 때 무엇을 distribution으로 보고 무엇을 uncertainty로 봐야 하는가**를 알려주는 논문이다. 따라서 EHR modeling, survival analysis, risk score design, phenotype discovery, clinical decision support를 연구하거나 설계하는 사람에게 특히 유용하다.
