# Explainable AI – the Latest Advancements and New Trends

Bowen Long, Enjie Liu SMember, Renxi Qiu, Yanqing Duan

## 🧩 Problem to Solve

최근 인공지능(AI) 기술은 다양한 분야에서 뛰어난 성과를 보이고 있으나, 특히 신경망 알고리즘의 복잡성으로 인해 AI 시스템의 의사결정 과정을 이해하기 어렵다는 "블랙박스" 문제가 발생한다. 이러한 이해 부족은 AI 시스템에 대한 신뢰 부족으로 이어지며, 이는 공정성, 책임성, 사회적 수용 측면에서 중요한 제약으로 작용한다. 따라서 AI의 결정에 대한 이유를 인간이 명확히 이해할 수 있도록 하는 '설명 가능성(Explainability)'을 확보하는 것이 시급한 과제이다.

## ✨ Key Contributions

- 다양한 국가 및 지역에서 제안된 신뢰할 수 있는 AI(Trustworthy AI)를 위한 윤리적 원칙과 발전 동향을 종합적으로 조사하였다.
- AI의 설명 가능성을 달성하기 위한 최신 연구(기술 및 기법)를 심층적으로 분류하고 분석하였다. 이는 크게 범위 기반(전역/지역)과 순서 기반(모델링 전/중/후) 접근 방식으로 나뉜다.
- 설명 가능한 AI를 위한 새로운 트렌드를 식별하고, 특히 AI의 설명 가능성과 메타 추론(Meta-reasoning) 개념 간의 강력한 연관성을 강조하였다.
- 신뢰할 수 있는 자율 시스템(Trustworthy Autonomous Systems), 도메인 무작위화(Domain Randomization), 그리고 대규모 언어 모델(LLMs)이 설명 가능한 AI 시스템의 미래를 위한 유망한 접근 방식이 될 수 있음을 제시하였다.

## 📎 Related Works

- **신뢰할 수 있는 AI(Trustworthy AI)의 부상:** 2010년대 중반부터 AI 시스템의 윤리적, 투명성, 안전한 배포에 대한 우려가 커지면서 이 개념이 주목받기 시작했다.
- **표준화 및 윤리 가이드라인:**
  - **IEEE ECPAIS (Ethics Certification Program for Autonomous and Intelligent Systems):** 자율 및 지능 시스템의 윤리 인증 프로그램을 제시하였다 [2].
  - **EU AI HLEG (High-level Expert Group on AI):** 신뢰할 수 있는 AI의 윤리 가이드라인을 수립하고, '인간 자율성 존중', '위해 방지', '공정성', '설명 가능성'이라는 네 가지 기본 원칙과 일곱 가지 구체적인 요구사항을 제시하였다 [7, 13].
  - **다양한 국가의 노력:** 미국, 일본, 캐나다, 호주, 뉴질랜드, 한국, 중국 등 여러 국가에서도 AI 윤리 원칙 및 신뢰성 프레임워크를 발표하였다 (표 1).
- **DARPA XAI (Explainable AI) 프로그램:** 미국 국방고등연구계획국(DARPA)이 인간 사용자가 AI 시스템의 행동을 더 쉽게 이해할 수 있도록 AI 시스템을 개발하기 위해 2016년 시작한 프로그램이다 [5].
- **AI 해석 가능성 연구 초기:** AI의 해석 가능성에 대한 연구는 1970년대 중반부터 시작되었으며 [17], 이후 다양한 기계 학습 해석 가능성 방법론이 제시되었다 [19].

## 🛠️ Methodology

이 논문은 AI 설명 가능성 기술을 포괄적으로 검토하고, 이를 다음 두 가지 주요 관점에서 체계적으로 분류 및 분석한다.

- **1. 범위 기반(Range-based) 해석 접근 방식:**

  - **전역적 해석 가능성(Global Interpretability):** 모델 전체의 논리와 동작 방식을 이해하는 데 중점을 둔다. 일반적으로 모델의 파라미터가 투명하거나 통계학적 지식 및 사전 경험이 필요한 모델에서 사용된다 (예: 의사결정 트리, 선형/로지스틱 회귀, 베이즈 규칙 기반 알고리즘, SVM, 퍼지 인지 맵, 스파이크 신경망 등).
  - **지역적 설명 가능성(Local Explainability):** 특정 개별 예측이나 단일 결정에 대한 이유를 설명하는 데 중점을 둔다.
    - **모델 독립적(Model-agnostic) 방법:** $\text{LIME}$ (Locally Interpretable Model-agnostic Explanations) [23], $\text{SHAP}$ (SHapley Additive exPlanations) [24] 등이 대표적이다.
    - **특징 기여도 기반:** 예측에 대한 각 입력 특징의 중요도를 파악한다.
    - **반사실적 설명(Counterfactual Explanations):** 예측 결과를 바꾸기 위해 입력에 최소한의 어떤 변화가 필요했는지를 제시하여 이해를 돕는다.

- **2. 순서 기반(Sequence-based) 해석 접근 방식:**
  - **모델링 전(Pre-modelling):** 모델 개발 프로세스가 시작되기 전에 설명 가능성을 고려한다.
    - **데이터 수집 및 분류:** 데이터 증강($\text{data augmentation}$) [30-32], 프로토타입 네트워크($\text{prototype network}$)를 사용한 데이터 유사성 측정 [33-35].
    - **설계에 의한 설명 가능성(By design):** 모델 자체를 본질적으로 해석 가능하도록 설계한다 (예: $\text{Bayesian Rule Lists (BRL)}$ [37], 일반화된 합산 기반 모델 [38], 선형 모델 [40]).
  - **모델링 중(In-modelling):** 모델 학습 및 동작 중에 본질적으로 설명을 제공하는 방법을 개발한다.
    - **모델 특정 해석 방법:** 특정 모델 아키텍처에 내재된 해석 가능성 요소를 활용한다 (예: 컨볼루션 신경망의 그래디언트 및 활성화 분석 [13, 42], 신경망 구조 단순화 [63]).
    - **주의 기반 자체 해석(Attentional self-interpretation):** 모델이 예측을 수행하면서 중요한 부분에 '주의(attention)'를 기울이도록 하여, 예측과 해석을 동시에 제공한다 (예: $\text{Visual Question Answering (VQA)}$ [58], $\text{Human-In-Loop (HIL)}$ [59], 다중 모달 해석 [60]).
  - **모델링 후(Post-modelling):** 모델이 학습된 후, 그 동작이나 예측 결과를 해석하는 모델 독립적(model-independent) 또는 사후(post-hoc) 접근 방식이다.
    - **시각화(Visualization):** 대리 모델(Surrogate models) [69], 부분 종속성 플롯($\text{Partial Dependence Plot (PDP)}$) [71] 및 개별 조건부 기대치($\text{Individual Conditional Expectation (ICE)}$) [73]와 같은 제어 변수 그래프, 인터랙티브 방법 [76-79]을 통해 모델 동작을 시각적으로 표현한다.
    - **지식 추출(Knowledge Extraction):** 신경망에서 학습된 지식을 규칙 추출($\text{rule extraction}$) [80] 또는 모델 증류($\text{model distillation}$) [82, 83]를 통해 이해 가능한 형태로 변환한다.
    - **영향 수준 방법(Impact level methods):** 입력 또는 내부 구성 요소의 변화가 모델 성능에 미치는 영향 정도를 측정하여 특징 또는 모듈의 중요도를 평가한다 (예: 민감도 분석($\text{sensitivity analysis}$) [88], 계층적 관련성 전파($\text{Layered Correlation Propagation (LRP)}$) [91], 특징 중요도 측정($\text{feature importance metrics}$) [92-94]).
    - **사례 기반 샘플(Instance-based samples):** 데이터셋에서 대표적인 샘플을 선택하여 모델을 설명한다. 주로 반사실적 해석($\text{counterfactual interpretation}$) [96, 97]을 사용한다.

## 📊 Results

이 서베이 논문의 주요 결과는 설명 가능한 AI(XAI) 분야의 현재 상태를 체계적으로 정리하고, 미래 연구 방향을 제시한 것이다.

- **XAI 기술의 종합적인 분류 및 분석:** AI 설명 가능성을 달성하기 위한 수많은 기술과 기법들을 범위 기반(전역/지역) 및 순서 기반(모델링 전/중/후)이라는 명확한 분류 체계로 제시하였다. 이를 통해 연구자들은 각 접근 방식의 장단점을 이해하고, 특정 시나리오에 맞는 적절한 XAI 기법을 선택할 수 있는 가이드라인을 얻을 수 있다.
- **신뢰할 수 있는 AI의 윤리적 프레임워크 제시:** 전 세계 주요 국가 및 기관의 AI 윤리 가이드라인을 분석하여 신뢰할 수 있는 AI의 핵심 윤리 원칙(인간 자율성, 위해 방지, 공정성, 설명 가능성)과 그를 충족시키기 위한 구체적인 요구사항(인간 감독, 견고성, 프라이버시, 투명성, 다양성, 사회/환경적 복지, 책임성)을 도출하였다.
- **XAI의 새로운 트렌드 식별:** 기존 XAI 방법론의 한계를 지적하며, '보상 중심의 설명 가능성(reward-driven explainability)'을 위한 메타 추론(Meta-reasoning)의 개념을 새로운 연구 방향으로 제안하였다. 이는 AI 시스템의 복잡한 내부 동작을 직접 해석하기보다, 보상 공간에서의 논리적 추론을 통해 설명 가능성을 높이는 접근 방식이다.
- **융합적 접근의 잠재력 강조:** 신뢰할 수 있는 자율 시스템(Trustworthy Autonomous Systems), 도메인 무작위화(Domain Randomization), 그리고 대규모 언어 모델(LLMs)이 XAI의 발전에 핵심적인 역할을 할 수 있음을 강조하며, 이들 간의 통합적 접근이 미래의 해석 가능한 AI 시스템을 구현하는 데 중요한 시너지를 창출할 수 있음을 보여주었다.

## 🧠 Insights & Discussion

- **XAI의 궁극적 목표와 메타 추론:** 현재 XAI 방법론은 학습과 추론 간의 복잡한 상호작용으로 인해 설명 가능성이 모호해지는 한계가 있다. 이를 극복하기 위해 본 논문은 "추론에 대한 추론"이라는 메타 추론($\text{meta-reasoning}$) [136] 개념을 설명 가능성과 연결한다. 메타 추론은 보상 공간($\text{reward space}$)에 문제를 투영함으로써 복잡성을 줄이고 관찰 가능성을 높여 설명 가능성을 향상시킬 수 있다. 이는 지상 수준($\text{ground level}$)의 데이터 중심 모델의 한계를 극복하고, 계산 모델 및 자원의 효율적인 사용에 초점을 맞춘다 [114].
- **신뢰할 수 있는 자율 시스템 및 도메인 무작위화:** 신뢰할 수 있는 자율 시스템($\text{Trustworthy Autonomous Systems}$) [119]은 AI가 보상 공간에 직접 노출되어 행동을 설계에 따라 설명하고 검증할 수 있게 한다. 또한, 도메인 무작위화($\text{Domain Randomization}$) [120-131]는 훈련 데이터에 의도적인 무작위 노이즈를 추가하거나 시뮬레이션 환경을 다양화하여 모델의 견고성과 일반화 능력을 향상시킨다. 이는 '현실 격차($\text{reality gap}$)'를 줄이고, 네트워크 오류의 영향을 최소화하여 설명 가능성이 실제 보상에 집중하도록 돕는다.
- **대규모 언어 모델(LLMs)의 역할:** 최근 $\text{LLM}$의 발전은 XAI에 새로운 지평을 열었다. 자연어 이해 및 생성 능력은 메타 추론을 조율하고, 관찰을 기반으로 AI 시스템에 가장 적합한 설명을 제공하는 데 필수적이다 [132-134]. $\text{Chain-of-Thoughts}$ [132]나 $\text{Tree-of-Thoughts}$ [133]와 같은 방법은 $\text{LLM}$을 추론기로 활용하여 문제 해결 과정을 설명하지만, 다양한 작업에서 일관된 성능을 유지하기 어렵다는 한계가 있다. 메타 추론은 이러한 한계를 완화하여 $\text{LLM}$이 상황과 작업 요구사항에 따라 설명 전략을 조절할 수 있도록 한다 [135]. $\text{SHAP}$ 값과 $\text{LLM}$의 통합 [137], 또는 $\text{Markov logic networks (MLNs)}$와 외부 지식의 결합 [138]과 같은 접근 방식은 $\text{LLM}$이 XAI에서 더욱 중요한 역할을 할 수 있음을 시사한다.

## 📌 TL;DR

AI의 '블랙박스' 문제로 인한 신뢰성 부족을 해결하고자, 본 논문은 기존의 Explainable AI (XAI) 기술들을 범위(전역/지역)와 순서(모델링 전/중/후) 기준으로 체계적으로 분류하고 각 기법의 특징을 분석한다. 나아가, '추론에 대한 추론'인 메타 추론($\text{Meta-reasoning}$)을 보상 공간($\text{reward space}$)에 투영하는 새로운 XAI 연구 트렌드를 제시하며, 신뢰할 수 있는 자율 시스템($\text{Trustworthy Autonomous Systems}$), 도메인 무작위화($\text{Domain Randomization}$), 그리고 대규모 언어 모델($\text{LLMs}$)의 통합이 미래의 설명 가능한 AI 시스템을 구현하는 데 핵심적인 역할을 할 것임을 강조한다.
