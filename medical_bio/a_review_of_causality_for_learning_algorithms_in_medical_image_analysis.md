# A Review of Causality for Learning Algorithms in Medical Image Analysis

## 논문 메타데이터

- **제목**: A Review of Causality for Learning Algorithms in Medical Image Analysis
- **저자**: Athanasios Vlontzos, Daniel Rueckert, Bernhard Kainz
- **arXiv 공개 연도**: 2022
- **최신 arXiv 버전 날짜**: 2022-11-26
- **저널**: Machine Learning for Biomedical Imaging
- **권/연도**: 1 (2022)
- **Paper ID**: 2022:028
- **arXiv ID**: 2206.05498
- **DOI**: 10.48550/arXiv.2206.05498
- **arXiv URL**: https://arxiv.org/abs/2206.05498
- **PDF URL**: https://arxiv.org/pdf/2206.05498v2
- **저널 URL**: https://www.melba-journal.org/papers/2022:028.html

## 연구 배경 및 문제 정의

이 논문은 의료영상 AI가 실험실 수준에서는 높은 성능을 보여도 실제 임상 현장으로 번역되는 비율이 낮다는 문제에서 출발한다. 저자들은 그 원인 중 하나를 모델이 `상관관계`와 `인과관계`를 구분하지 못한 채 학습한다는 점에서 찾는다. 의료영상 데이터는 병원, 장비, 인구집단, 라벨링 방식, 데이터 수집 절차에 따라 강한 편향과 도메인 시프트를 가지므로, 단순한 경험적 위험 최소화만으로는 실제 임상 환경에서 안정적으로 동작하기 어렵다는 것이다.

논문이 제시하는 핵심 문제는 다음과 같다.

- 의료영상 모델은 종종 병원 식별자, 스캐너 특성, 환자군 차이 같은 spuriously correlated signal을 학습한다.
- 이런 현상은 임상 배치 후 성능 붕괴나 위험한 오진으로 이어질 수 있다.
- 기존 의료영상 ML은 proof-of-concept에서 deployment로 너무 빠르게 이동하며, 강건성과 적응성을 다루는 중간 단계가 부족하다.
- 인과 분석은 이런 간극을 메우는 도구가 될 수 있지만, 의료영상 분야에서는 아직 채택이 제한적이다.

저자들은 이를 Technology Readiness Levels 관점으로 재해석하며, 특히 임상 번역 직전 단계에서 causally robust algorithm이 필요하다고 주장한다.

## 핵심 기여

이 논문은 새 알고리즘을 제안하지 않는 리뷰다. 대신 의료영상 학습 알고리즘에 대한 인과적 사고의 필요성을 구조적으로 정리한다.

핵심 기여는 다음과 같다.

1. 의료영상 ML의 임상 번역 실패를 인과적 관점에서 설명한다.
2. Structural Causal Models와 Potential Outcomes 같은 인과 추론 기초 개념을 의료영상 독자에게 연결한다.
3. 의료영상 분야의 causality 관련 연구를 `causal discovery`, `causal reasoning`, `causal representation learning`, `causal fairness`, `reinforcement learning` 등으로 분류해 정리한다.
4. 인과 분석이 domain shift, confounding, dataset bias, fairness 문제를 줄이는 데 어떤 역할을 할 수 있는지 논의한다.

즉, 이 논문의 주된 가치는 "의료영상 알고리즘을 더 똑똑하게 만드는 법"보다 "왜 현재 방식이 임상에서 깨지는지, 그리고 인과성이 그 틈을 어떻게 메울 수 있는지"를 보여주는 데 있다.

## 이론적 배경 정리

## 1. Structural Causal Models

논문은 먼저 Pearl 계열 인과 추론의 중심 개념인 Structural Causal Model과 Directed Acyclic Graph를 소개한다. 여기서 변수 간 화살표는 단순 상관이 아니라 원인-결과 방향을 나타내며, intervention은 특정 변수의 생성 메커니즘을 외생적으로 바꾸는 연산으로 해석된다.

저자들이 강조하는 핵심은 다음과 같다.

- 단순 연관성 분석만으로는 모델이 무엇을 진짜 원인으로 삼는지 알 수 없다.
- 의료영상에서 중요한 질문은 관찰된 패턴뿐 아니라 "개입하면 결과가 어떻게 달라질까"다.
- counterfactual reasoning은 특정 환자 또는 상황에서 "다른 조건이었다면 어떻게 되었을까"를 묻는 도구다.

논문은 Abduction-Action-Prediction과 Twin Network를 counterfactual 추론의 대표적 틀로 정리한다.

## 2. Potential Outcomes

다른 인과 추론 축으로 Rubin의 Potential Outcomes framework도 소개한다. 여기서는 치료 여부가 달랐을 때 같은 환자가 어떤 결과를 보였을지를 가정적으로 비교한다.

이 프레임의 의료영상 맥락상 장점은 다음과 같다.

- 치료 효과 추정과 같이 intervention 중심 질문을 다루기 쉽다.
- average treatment effect, individual treatment effect, propensity score 같은 개념이 임상 연구와 자연스럽게 연결된다.
- 관찰 데이터 기반 의료영상 연구를 임상 의사결정과 접목하는 통로가 된다.

즉, SCM이 구조적 원인 모형을 강조한다면, Potential Outcomes는 처치 효과 평가와 의사결정에 가까운 해석을 제공한다.

## 논문이 제시하는 분류 체계

## 1. Causal Discovery

이 섹션은 데이터로부터 원인 구조를 추정하려는 연구를 다룬다. 저자들은 constraint-based, score-based, optimization-based 방법을 정리하고, 의료영상 및 시각 데이터에서 causal graph를 추정하려는 초기 시도를 소개한다.

핵심 포인트는 다음과 같다.

- causal discovery는 과학적 설명력 측면에서 매력적이지만 강한 가정에 의존한다.
- 의료영상 데이터는 관측되지 않은 교란변수, 작은 표본, 고차원 입력 때문에 문제를 더 어렵게 만든다.
- 따라서 직접적인 full causal graph 복원보다 제한된 구조 추론이나 representation level causal discovery가 현실적일 수 있다.

이 섹션은 즉시 실용화된 방법을 보여준다기보다, 의료영상 AI가 해석 가능한 원인 구조로 이동할 수 있는 가능성을 제시한다.

## 2. Causal Reasoning and Counterfactuals

논문은 인과 추론의 가장 실용적 측면으로 counterfactual reasoning을 강조한다. 예를 들어 특정 병변이 없었더라면 예측이 어떻게 달라졌을지, acquisition setting이 달랐더라면 segmentation 결과가 어떻게 바뀌었을지 같은 질문이 여기에 속한다.

저자들의 관점에서 counterfactual 기반 접근은 다음 장점이 있다.

- spurious feature와 disease-relevant feature를 분리하는 데 도움
- data augmentation을 넘어 causally plausible sample generation 가능
- clinical explanation과 intervention planning에 가까운 질의 수행 가능

의료영상에서는 특히 병변 제거/삽입, modality 변형, 해부학 변화 시뮬레이션 같은 방향과 연결된다.

## 3. Causal Representation Learning

이 리뷰에서 가장 현대적인 축은 causal representation learning이다. 저자들은 좋은 표현이 단순히 predictive한 것만이 아니라, domain shift가 바뀌어도 유지되는 invariant causal factor를 포착해야 한다고 본다.

의료영상에서 이 아이디어가 중요한 이유는 다음과 같다.

- 병원이나 장비가 바뀌어도 질병 기전은 바뀌지 않는 경우가 많다.
- 따라서 causal feature를 학습하면 domain generalization에 유리할 수 있다.
- 반대로 shortcut feature에 의존한 representation은 외부 검증에서 쉽게 무너진다.

이 논문은 직접 방법을 제안하지는 않지만, representation learning과 invariance, disentanglement, robustness 사이의 연결고리를 비교적 선명하게 제시한다.

## 4. Fairness and Bias

저자들은 causality가 fairness 문제에도 중요하다고 본다. 의료영상 AI는 성별, 인종, 연령, 병원, 보험 접근성 같은 변수와 얽혀 있으며, 단순 성능 최적화는 특정 하위집단에 불리한 결과를 만들 수 있다.

인과 관점의 장점은 다음과 같다.

- 어떤 변수 경로를 허용할지와 차단할지를 명시적으로 논의할 수 있다.
- confounder와 mediator를 구분해 더 정교한 편향 분석이 가능하다.
- fairness를 단순 group metric이 아닌 구조적 원인 문제로 볼 수 있다.

의료영상 공정성 연구가 아직 제한적이라는 점을 감안하면, 이 부분은 미래 방향 제시에 가깝다.

## 5. Reinforcement Learning and Decision Making

논문은 causality를 단순 예측 문제를 넘어 sequential decision making과도 연결한다. 의료에서는 진단뿐 아니라 추적 검사, 치료 선택, 개입 시점 결정이 중요하므로, 장기 결과를 고려하는 인과적 decision making이 필요하다는 것이다.

다만 이 축은 의료영상 자체보다는 broader medical AI와 더 강하게 연결돼 있으며, survey 내에서도 상대적으로 짧게 다뤄진다.

## 실험 설정과 결과

이 논문은 리뷰 논문이므로 자체 실험을 수행하지 않는다. 따라서 특정 데이터셋에서 수치 우위를 보이는 방식의 결과 분석은 불가능하다.

대신 이 논문이 제공하는 실질적 결과는 다음과 같은 수준의 종합 판단이다.

- causality는 domain shift와 spurious correlation 문제를 줄일 잠재력이 있다.
- 의료영상 분야에서 실제로 causality를 모델 설계에 사용한 연구 수는 아직 적다.
- 특히 임상 downstream validation이나 deployment-level 연구는 매우 제한적이다.

즉, 이 논문의 결론은 "인과성이 중요하다"는 선언에 머물지 않고, "중요하지만 아직 field-wide uptake는 낮다"는 다소 냉정한 평가를 포함한다.

## 강점

## 1. 의료영상 임상 번역 실패를 구조적으로 설명한다

단순히 dataset bias가 있다고 말하는 수준을 넘어, 왜 proof-of-concept 성능이 임상 강건성으로 이어지지 않는지 TRL 관점으로 정리한 점이 설득력 있다.

## 2. 인과 추론과 의료영상을 연결하는 입문 문헌으로 유용하다

Structural Causal Models, intervention, counterfactual, potential outcomes 같은 개념을 의료영상 응용 맥락으로 옮겨 놓았다는 점에서 좋은 브리지 문헌이다.

## 3. 기술보다 문제 설정을 개선한다

이 논문은 새 네트워크를 제시하지 않지만, 의료영상 연구자에게 "무엇을 예측할 것인가"뿐 아니라 "무엇을 원인으로 볼 것인가"를 묻게 만든다.

## 한계와 비판적 검토

## 1. 정량적 비교가 거의 없다

survey 논문이라 해도 각 방법이 실제로 얼마나 외부 일반화와 임상 번역을 개선하는지에 대한 정량 메타분석은 부족하다.

## 2. 인과성의 실제 적용 난이도를 상대적으로 덜 드러낸다

인과 그래프 설계, 교란변수 관측 가능성, identifiability, intervention 정의 등은 실제 의료영상 연구에서 매우 어렵다. 논문은 이를 언급하지만, 실무 장벽의 크기를 완전히 체감하게 하지는 않는다.

## 3. 의료영상-specific 방법론 정리는 아직 초기적이다

분류 체계는 유익하지만, 당시까지 의료영상에서 causality를 본격 활용한 논문 수 자체가 많지 않아 각 범주의 사례 깊이는 제한적이다.

## 4. 2022년 이후 급증한 foundation model 시대 흐름은 반영하지 못한다

대형 비전 모델, multimodal medical foundation model, prompt-based adaptation과 causality의 접점은 이후 더 커졌지만, 이 논문은 시점상 이를 포괄하지 못한다.

## 실무적 및 연구적 인사이트

이 논문은 의료영상 연구자가 validation 성능만으로 모델을 신뢰하면 안 된다는 점을 강하게 상기시킨다. 특히 병원 변경, 장비 변경, 인구집단 변경, annotation policy 변경 같은 현실 변화에 모델이 어떻게 반응하는지는 본질적으로 인과 문제에 가깝다. 따라서 의료영상 AI의 다음 단계는 더 큰 모델만 만드는 것이 아니라, 어떤 feature가 질병의 원인 구조와 연결되는지, 어떤 변수는 교란인지, 어떤 성능 저하는 distribution shift 때문인지 구조적으로 분석하는 방향으로 가야 한다.

후속 연구 방향도 비교적 분명하다.

- domain generalization과 causality의 결합
- counterfactual data generation을 이용한 robust training
- fairness와 causality의 결합
- 의료 foundation model의 spurious feature 분석
- 관찰 데이터 기반 임상 의사결정 지원에서 treatment effect estimation 활용

## 종합 평가

`A Review of Causality for Learning Algorithms in Medical Image Analysis`는 의료영상 알고리즘의 성능 향상 기법을 소개하는 논문이 아니라, 왜 현재의 고성능 모델이 임상에서 자주 무너지는지를 인과적 시각으로 해석하는 리뷰다. 핵심 메시지는 간단하다. 의료영상 ML의 진짜 병목은 모델 크기보다도 causal robustness의 부족일 수 있다는 것이다.

따라서 이 논문은 구체적 구현 레시피를 주는 문헌이라기보다, 의료영상 AI 연구 질문 자체를 재구성하게 만드는 개념적 문헌으로 읽는 것이 적절하다. 인과 기반 방법의 사례 수는 아직 적지만, 임상 번역과 강건성 문제를 진지하게 다루려면 여전히 중요한 기준 문헌이다.
