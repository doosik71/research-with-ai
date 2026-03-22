# Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings

이 논문은 healthcare와 같은 high-stakes domain에서 **offline reinforcement learning(offline RL)**을 사용할 때, 실제로 가장 어려운 문제 중 하나인 **model selection** 을 정면으로 다룬다. 핵심 문제는 supervised learning처럼 validation set에서 성능을 직접 측정할 수 없다는 점이다. RL에서는 validation performance를 알려면 학습된 policy를 실제 환경에 배치해 봐야 하는데, 의료 환경에서는 여러 미검증 치료 정책을 실제 환자에게 시험하는 것이 비현실적이거나 위험하다. 저자들은 이 공백을 메우기 위해 **off-policy evaluation(OPE)**를 validation proxy로 사용하는 training-validation pipeline을 제안하고, 어떤 OPE가 실제 model ranking에 유용한지, 또 계산 비용은 얼마나 드는지를 sepsis treatment task에서 체계적으로 분석한다. 특히 **FQE(Fitted Q Evaluation)**가 가장 좋은 ranking을 주지만 계산 비용이 크다는 점을 보이고, 이를 보완하기 위해 **WIS로 후보를 먼저 거르고 FQE로 최종 선택하는 2-stage selection** 을 제안한다.

## 1. Paper Overview

이 논문의 문제의식은 매우 실무적이다. healthcare에서 RL은 sepsis treatment, ICU hypotension management, ventilator weaning처럼 순차적 의사결정 문제를 다루는 데 유망하지만, 실제로는 online interaction이 불가능한 경우가 많다. 따라서 historical observational data만으로 policy를 학습하는 **offline RL** 이 필요하다. 그런데 clinical decision making은 상태 공간이 고차원이고 연속적인 경우가 많아서 neural network 같은 function approximator를 써야 하며, 이 경우 overfitting을 막기 위한 **hyperparameter tuning과 model selection** 이 필수다. 그럼에도 당시까지 offline RL에는 널리 받아들여진 training-validation framework가 없었다.  

저자들이 제기하는 핵심 연구 문제는 다음과 같다. “실제 environment rollout 없이, historical data만으로 여러 candidate policy 중 어느 것이 deployment에 가장 적합한지 어떻게 고를 것인가?” 이 질문은 의료처럼 실험 비용이 매우 크고 잘못된 선택이 환자 위해로 이어질 수 있는 영역에서 특히 중요하다. 기존 연구가 simulator가 있는 benchmark에서는 실제 return으로 model selection을 하거나, 아예 default hyperparameter만 쓰는 경우가 많았다는 점을 감안하면, 이 논문은 offline RL의 현실적 사용 장벽을 직접 겨냥한 연구라고 볼 수 있다.  

논문의 중요성은 단순히 “새 OPE 하나를 제안했다”는 데 있지 않다. 오히려 저자들은 이미 널리 알려진 OPE 기법들을 **model selection 관점** 에서 다시 비교한다. 즉, “절대적인 value estimate가 얼마나 정확한가”보다 “여러 후보 policy를 얼마나 잘 순위화(rank)할 수 있는가”를 더 중요하게 본다. 이는 실제 hyperparameter search와 early stopping에서 훨씬 직접적인 기준이다.  

## 2. Core Idea

이 논문의 중심 아이디어는 **OPE를 validation performance의 대리 지표(proxy)**로 사용해 offline RL의 training-validation pipeline을 구성하는 것이다. supervised learning에서는 validation set accuracy가 model selection 기준이 되지만, offline RL에서는 learned policy를 실제 환경에서 실행할 수 없으므로 그 역할을 OPE가 대신해야 한다. 저자들은 이 단순한 발상을 실제로 작동하게 만들기 위해, OPE estimator의 ranking quality, 추가 hyperparameter, auxiliary model 필요성, computational overhead까지 함께 평가한다.  

핵심적으로 논문은 네 가지 popular OPE method를 비교한다. 그리고 이들 사이의 trade-off를 명확히 드러낸다. 저자들의 주요 결론은 다음과 같다.
FQE는 candidate policy ranking을 가장 잘해 주지만 계산 비용이 높다.
WIS는 비교적 단순하고 계산이 덜 들지만 ranking accuracy는 FQE보다 낮다.
AM은 상대적으로 robust하지만 전체 성능은 떨어진다.
따라서 단일 estimator만 고집하기보다, **빠른 estimator로 후보를 pruning하고 정확한 estimator로 최종 선택하는 2-stage scheme** 이 더 실용적이다.

이 아이디어가 기존 연구와 구분되는 지점은, OPE를 단순 evaluation 용도가 아니라 **model selection pipeline의 구성 요소** 로 다룬다는 점이다. 다시 말해 “이 policy의 value는 얼마인가?”보다 “이 OPE를 써서 hyperparameter search를 하면 실제로 더 좋은 policy를 고를 수 있는가?”가 논문의 질문이다. healthcare용 offline RL에서는 이 차이가 매우 크다.

## 3. Detailed Method Explanation

### 3.1 Offline RL problem setup

논문은 MDP를
$$
\mathcal{M}=(\mathcal{S},\mathcal{A},p,r,\mu_0,\gamma)
$$
로 두고, 목표 policy $\pi$의 성능을
$$
v(\pi)=J(\pi;\mathcal{M})=\mathbb{E}_{s\sim \mu_0}[V^\pi(s)]
$$
로 정의한다. 여기서
$$
V^\pi(s)=\mathbb{E}_{\pi}\mathbb{E}_{\mathcal{M}}\left[\sum*{t=1}^{\infty}\gamma^{t-1}r_t \mid s_1=s\right]
$$
이다. offline RL에서는 environment와 상호작용하지 못하고, 관측 데이터셋 $\mathcal{D}_{obs}={s_i,a_i,r_i,s_i'}_{i=1}^N$ 만 이용해 policy를 학습한다. 따라서 실제 목표는 이 데이터만으로 좋은 policy를 배우는 것뿐 아니라, 여러 learned policy 중 가장 좋은 policy를 고르는 것이다.  

### 3.2 왜 supervised-style validation이 불가능한가

supervised learning에서는 validation set에 넣어 예측 성능을 측정하면 되지만, RL에서 validation performance는 policy를 environment에 rollout해 봐야 알 수 있다. simulator가 있는 benchmark라면 가능하지만, healthcare에서는 환자에게 여러 학습 정책을 시험해 볼 수 없다. 이 때문에 저자들은 **OPE를 validation proxy** 로 사용해야 한다고 주장한다. 다만 OPE도 완벽하지 않으며, counterfactual을 추정해야 해서 본질적으로 어렵다. 즉, historical data에서 관찰되지 않은 행동을 했을 때 어떤 결과가 나왔을지를 추정해야 한다.  

### 3.3 OPE-based model selection framework

논문이 제안하는 framework는 개념적으로 단순하다.

1. training data로 여러 candidate policy를 학습한다.
2. 별도의 validation data에 대해 각 candidate policy의 value를 OPE로 추정한다.
3. OPE score를 기준으로 policy를 rank한다.
4. 가장 높은 validation-OPE score를 가진 policy를 최종 선택한다.

이 과정에서 중요한 것은 OPE가 단순 value estimation보다 **ranking accuracy** 를 가져야 한다는 점이다. 한 estimator가 절대값은 biased하더라도 후보 간 순서를 잘 맞추면 model selection에는 유용할 수 있다. 논문은 바로 이 ranking quality를 정량적으로 비교한다.  

### 3.4 비교한 OPE 방법과 practical issue

논문은 네 가지 popular OPE 방법을 분석하며, 공통적으로 **추가 hyperparameter** 와 **auxiliary model** 이 필요할 수 있음을 강조한다. 이 점이 실제론 매우 중요하다. offline RL model을 고르기 위해 OPE를 쓰는데, OPE 자체도 별도의 model selection 문제를 만든다면 전체 파이프라인이 복잡해진다. 또한 auxiliary model fitting 때문에 계산량이 크게 늘어난다. 저자들이 논문 전체에서 반복해서 강조하는 practical consideration이 바로 이것이다.  

대표적으로 본문과 표 설명에서 드러나는 auxiliary hyperparameter는 **policy softening parameter $\varepsilon$** 와 **evaluation horizon $H$** 다. 본 논문의 continuous-state main experiment에서는 FQE/FQI를 neural network function approximator로 학습했고, OPE 적용 시 $\varepsilon=0.01$, $H=20$을 사용했다고 명시한다. 즉, OPE는 “그냥 한 번 계산”하는 도구가 아니라, 실제로는 별도 설계 choices가 필요한 절차다.  

### 3.5 Two-stage selection

논문의 가장 실용적인 기여는 2-stage selection이다. 아이디어는 매우 직관적이다.
먼저 계산이 빠르지만 덜 정확한 OPE로 low-quality policy를 제거한다.
그 다음 남은 promising subset에 대해서만 계산이 비싸지만 더 정확한 OPE를 적용한다.

이 논문에서는 **WIS로 상위 subset을 고르고, FQE로 최종 선택** 하는 구성이 중심이다. subset size $\alpha$ 는 사전에 정해야 하며, 실험에서는 다양한 크기를 시험한다. 본문과 Figure 3 설명에 따르면, 전체 후보 $K=96$ 중 중간 크기 subset, 예를 들어 **24개 정도** 를 남기는 설정이 regret과 computation의 균형이 가장 좋았다.  

이 방식의 핵심은 “정확한 estimator를 모두에게 돌릴 필요가 없다”는 것이다. 즉, FQE의 정확도는 유지하면서 계산 부담을 줄일 수 있다. healthcare처럼 compute budget도 중요하고 반복적 model search가 필요한 영역에서는 매우 실용적인 전략이다.  

## 4. Experiments and Findings

논문은 sepsis treatment task를 사용해 실험한다. candidate policy 생성에는 FQI를 사용했고, discrete state space와 continuous state space 모두를 고려한다. 특히 continuous setting이 main experiment이며, neural architecture selection과 early stopping 같은 실제 model selection 시나리오를 반영한다. implementation detail로는 neural network 기반 FQI를 최대 50 iteration까지 학습하고, training/validation에 각각 10,000 episode를 사용하며 10회 replication run을 수행했다고 명시한다.  

가장 중요한 결과는 **FQE가 가장 좋은 validation ranking** 을 제공했다는 점이다. abstract와 conclusion 모두에서 이 메시지가 일관되게 반복된다. 저자들은 FQE가 sample size와 behavior policy 변화 같은 다양한 데이터 조건에서도 가장 robust한 편이라고 정리한다. 따라서 ranking quality만 보면 FQE가 최선의 선택이다.  

하지만 FQE의 약점도 명확하다. **계산 비용이 높다.** auxiliary model fitting과 inference 때문에 candidate set 전체에 FQE를 적용하면 비용이 커진다. 이 때문에 저자들은 정확도와 계산 효율 사이의 trade-off를 explicit하게 논의하고, 2-stage selection을 제안한다. Figure 3 설명에 따르면, subset size 24 정도의 two-stage selection이 regret과 normalized computational cost 사이에서 가장 좋은 균형을 보였다.  

Sensitivity analysis도 논문의 중요한 부분이다. 저자들은 세 가지 현실적 가정을 흔들어 본다.
첫째, OPE auxiliary hyperparameter의 선택.
둘째, validation data의 behavior policy가 충분히 exploratory하지 않은 경우.
셋째, validation sample size가 줄어드는 경우.

결과적으로 **AM은 $H$에 대해 robust하지만 전반적으로 underperform** 하고, **WIS와 FQE는 reasonable한 hyperparameter 선택 아래 낮은 regret** 을 유지한다. 또 validation data가 덜 exploratory하거나 behavior policy가 혼합되어 있는 challenging condition에서도, 제안한 two-stage scheme은 FQE와 비슷하거나 더 나은 regret을 유지하면서 계산량을 줄였다.  

추가로 논문은 “그냥 FQI training value나 TD error로 선택하면 되지 않느냐”는 가능성도 간접적으로 부정한다. appendix 결과에 따르면 이런 단순 기준은 낮은 ranking correlation과 높은 regret을 보여, 실제 validation performance 대용으로는 부적절하다. 즉, offline RL model selection에서는 OPE가 필요하다는 것이 실험적으로도 뒷받침된다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **문제 설정이 현실적** 이라는 점이다. offline RL 연구에서는 종종 algorithm 그 자체에 집중하고, 실제로는 어떻게 hyperparameter tuning을 하고 최종 모델을 고를지에 대한 논의가 빠진다. 이 논문은 바로 그 practical gap을 다룬다. 특히 healthcare처럼 online validation이 불가능한 domain에서 무엇이 현실적인 pipeline인지 구체적으로 보여준다는 점이 강하다.  

두 번째 강점은 **ranking quality와 computational cost를 함께 본다** 는 점이다. 단순히 가장 정확한 estimator를 찾는 데서 멈추지 않고, 정확하지만 비싼 FQE와 상대적으로 저렴한 WIS를 조합해 더 실용적인 방법을 설계한다. 이런 관점은 high-stakes domain에서 반복 실험이 필요한 실제 연구 워크플로우와 잘 맞는다.  

세 번째 강점은 **sensitivity analysis의 충실함** 이다. OPE hyperparameter, behavior policy misspecification, validation sample size 감소 같은 현실적 조건 변화를 따져 본 덕분에, 단일 benchmark 숫자보다 훨씬 유용한 실무적 결론을 제공한다.  

한계도 분명하다. 우선 실험은 **simulated sepsis treatment task** 중심이다. healthcare inspiration은 강하지만, 실제 병원 deployment의 복잡한 observational bias, partial observability, clinician policy heterogeneity, reward misspecification을 모두 포괄하지는 못한다. 따라서 결론은 “실제 임상에 바로 적용 가능한 절대 법칙”이라기보다, healthcare-style offline RL에서 유효한 강한 실천 지침으로 보는 편이 맞다.  

또한 이 논문이 해결하지 못한 메타 문제도 있다. OPE를 model selection에 쓰면, OPE 자체가 auxiliary hyperparameter와 auxiliary model selection을 요구한다. 즉, model selection 문제를 완전히 없앤 것이 아니라, 더 잘 구조화한 것이다. 논문은 이를 숨기지 않고 오히려 전면적으로 드러낸다는 점에서 정직하지만, 동시에 이것이 남는 어려움이기도 하다.  

비판적으로 해석하면, 이 논문은 “FQE가 최고다”가 전부가 아니다. 더 중요한 메시지는 **offline RL에서 model selection은 반드시 evaluation-as-ranking 문제로 다시 생각해야 한다** 는 점이다. 그리고 좋은 실무 파이프라인은 정확도만이 아니라 compute budget, validation data quality, auxiliary tuning burden까지 함께 고려해야 한다는 교훈을 준다.  

## 6. Conclusion

이 논문은 healthcare와 같은 offline, high-stakes 환경에서 offline RL의 model selection을 어떻게 수행할지에 대한 매우 실용적인 가이드를 제공한다. 핵심 아이디어는 **OPE를 validation performance의 proxy로 사용** 하는 것이며, 네 가지 popular OPE 방법을 model ranking, 추가 hyperparameter, auxiliary model, computational cost 관점에서 비교한다. 실험 결과, **FQE가 가장 정확한 policy ranking을 제공** 하지만 계산 비용이 높고, 이를 보완하기 위해 **WIS로 후보를 pruning한 뒤 FQE로 최종 선택하는 2-stage approach** 가 성능과 효율 사이의 좋은 균형을 보였다.  

이 연구가 중요한 이유는, offline RL의 성능을 높이는 새로운 algorithm을 제안했다기보다, **실제로 어떻게 좋은 policy를 고를 것인가** 라는 배치 직전의 문제를 해결하려 했기 때문이다. 특히 healthcare RL 연구가 종종 default hyperparameter나 simulator-only evaluation에 의존해 온 점을 생각하면, 이 논문은 보다 공정한 알고리즘 비교와 더 현실적인 deployment 준비를 가능하게 하는 기반 연구라고 볼 수 있다. 향후 실제 clinical observational data, 더 복잡한 behavior policy mixture, partial observability 문제와 결합될 때 더욱 의미가 커질 것이다.  
