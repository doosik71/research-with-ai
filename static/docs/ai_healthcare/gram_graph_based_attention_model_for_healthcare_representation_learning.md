# GRAM: Graph-based Attention Model for Healthcare Representation Learning

이 논문은 의료 예측에서 deep learning이 겪는 두 가지 핵심 문제, 즉 **data insufficiency**와 **interpretability**를 동시에 다루기 위해 제안된 모델이다. 저자들은 의료 ontology의 계층 구조를 활용해 각 의료 개념을 자기 자신과 조상(ancestor) 개념의 가중 결합으로 표현하는 **GRAM (Graph-based Attention Model)**을 제안한다. 핵심은 희귀 코드일수록 더 상위의 일반 개념에 의존하고, 자주 관측되는 코드일수록 leaf 개념 자체를 더 강하게 반영하도록 attention이 자동으로 조절된다는 점이다. 이 방식은 희소 데이터 환경에서 예측 성능을 높이면서도, 표현이 의료 지식 구조와 정렬되도록 만든다.  

## 1. Paper Overview

이 논문이 푸는 문제는 단순히 “EHR에서 더 정확한 예측을 하자”가 아니다. 저자들은 의료 AI에서 deep learning이 잘 안 풀리는 이유를 두 가지로 정리한다. 첫째는 **표본 부족**이다. 실제 의료 데이터에서는 rare disease나 희귀 코드가 많고, 단일 의료기관 데이터만으로는 충분한 표본을 확보하기 어렵다. 둘째는 **해석 가능성**이다. 모델이 학습한 representation이 의료 ontology와 전혀 맞지 않으면, 임상적으로 신뢰하기 어렵다. GRAM은 이 두 문제를 동시에 풀기 위해, EHR 자체만이 아니라 ICD/CCS 같은 의료 ontology의 parent-child 구조를 representation learning에 주입한다.

이 문제가 중요한 이유는 의료 데이터의 long-tail 특성 때문이다. 흔한 질환은 학습이 되지만, 희귀 질환이나 자주 등장하지 않는 진단 코드는 RNN 같은 flexible model에서도 충분히 학습되지 않는다. 저자들은 실제로 기존 RNN이 rare disease onset prediction에서 비효율적이었다고 지적한다. 따라서 의료 ontology를 활용해 상위 개념으로부터 통계적 strength를 빌려오는 전략은 매우 자연스럽고 실용적이다.

논문은 두 개의 sequential diagnosis prediction task와 하나의 heart failure prediction task에서 GRAM을 평가한다. 초록 수준의 핵심 결과는 다음과 같다. **희귀 질환 예측에서는 기본 RNN 대비 최대 10% 높은 정확도**, 그리고 **heart failure prediction에서는 10배 적은 학습 데이터로도 약 3% 더 높은 AUC**를 달성했다. 또한 learned representation이 의료 ontology와 더 잘 정렬되고, attention도 직관적으로 해석 가능하다고 주장한다.

## 2. Core Idea

GRAM의 핵심 아이디어는 “의료 코드 하나를 하나의 embedding으로만 보지 말고, **그 코드와 그 조상 개념들의 mixture로 표현하자**”는 것이다. 예를 들어 어떤 leaf diagnosis code가 매우 희귀하다면, 그 코드 자체의 embedding만으로는 학습이 불안정할 수 있다. 대신 그 code의 parent, grandparent 같은 더 일반적인 개념은 더 많은 샘플을 공유하므로 더 안정적으로 학습된다. GRAM은 이 점을 attention mechanism으로 구현한다.  

즉, 의료 개념 $c_i$의 최종 표현 $\mathbf{g}_i$는 단순한 base embedding $\mathbf{e}_i$가 아니라, 자기 자신과 ancestor들의 base embedding을 attention으로 가중합한 결과다. 논문 그림 설명에 따르면, leaf node는 실제 EHR 코드이고 non-leaf node는 더 일반적인 상위 개념이며, 최종 표현은 이들을 조합해 만들어진다. 그 뒤 이 embedding matrix를 이용해 visit vector를 visit representation으로 바꾸고, 이를 예측 모델에 넣는다.

이 구조의 novelty는 두 가지다. 첫째, ontology를 단순 feature engineering이나 post-hoc regularizer로 쓰지 않고, **attention 기반 representation learning 내부에 직접 통합**했다는 점이다. 둘째, ontology 사용 강도를 고정하지 않고, **데이터 양과 계층 구조에 따라 adaptive하게 조절**한다는 점이다. 저자들이 반복해서 강조하는 것도 바로 “rare code면 ancestor에 더 의존하고, frequent code면 leaf 자체를 더 신뢰한다”는 직관이다.  

## 3. Detailed Method Explanation

### 3.1 데이터와 ontology의 형식화

논문은 EHR를 환자별 visit sequence로 본다. 전체 의료 코드 집합을 $\mathcal{C}$라고 하면, 각 visit $V_t$는 그 부분집합이고, binary visit vector $\mathbf{x}_t \in {0,1}^{|\mathcal{C}|}$로 표현할 수 있다. 또한 의료 ontology는 leaf node가 실제 코드, non-leaf node가 상위 개념인 directed acyclic graph(DAG) $\mathcal{G}$로 표현된다. 즉, 모델은 EHR의 순차 구조와 ontology의 계층 구조를 동시에 다룬다.

### 3.2 Knowledge DAG와 attention

GRAM에서 각 node $c_i$는 먼저 base embedding $\mathbf{e}_i \in \mathbb{R}^m$를 가진다. 하지만 실제 예측에 쓰는 것은 base embedding이 아니라 final representation $\mathbf{g}_i$다. 이 final representation은 해당 leaf concept와 ancestor concept들의 embedding을 attention으로 결합해 만든다. 저자 설명에 따르면, 희귀한 코드일수록 ancestor가 더 큰 weight를 받고, 충분히 자주 등장하는 코드일수록 leaf 자체가 더 큰 weight를 받는다. 이는 ontology를 통해 rare code representation을 보강하는 메커니즘이다.  

직관적으로 보면, 이는 의료 ontology를 통한 **adaptive smoothing**이다. 일반적인 임베딩에서는 희귀 코드는 noisy한 vector가 되기 쉽지만, GRAM은 상위 개념과의 관계를 활용해 representation을 안정화한다. 중요한 점은 이 과정이 hand-crafted rule이 아니라 **attention으로 end-to-end 학습**된다는 것이다. 논문은 이를 통해 robust하면서도 ontology-aligned한 representation을 얻는다고 설명한다.

### 3.3 전체 예측 파이프라인

논문의 Figure 1 설명에 따르면, 최종 concept embeddings로 이루어진 matrix $\mathbf{G}$를 이용해 visit vector $\mathbf{x}_t$를 visit representation $\mathbf{v}_t$로 변환한 뒤, 이를 neural predictive model에 넣어 $\hat{\mathbf{y}}_t$를 예측한다. 즉 GRAM은 그 자체가 최종 classifier라기보다, **ontology-aware representation layer + downstream predictive model** 구조로 이해하는 것이 맞다.

이 때문에 GRAM의 공헌은 특정 예측기 하나보다 “의료 ontology를 활용하는 representation learning 방식”에 더 가깝다. 논문에서도 sequential diagnosis prediction과 HF prediction이라는 서로 다른 task에 같은 representation idea를 적용한다. 그래서 GRAM은 task-specific trick이라기보다 healthcare representation learning framework로 해석할 수 있다.

### 3.4 End-to-end training

저자들은 attention generation과 predictive model을 함께 학습한다고 설명한다. 즉, ontology attention은 사전 정의된 rule이 아니라 prediction loss를 통해 같이 업데이트된다. 이 점이 중요하다. 단순히 ontology 기반 embedding을 미리 만들고 고정하는 것이 아니라, 실제 예측 성능에 유리하도록 attention이 조절되기 때문이다.

### 3.5 초기화(initialization)

논문은 ontology 정보 외에도 효과적인 initialization 기법을 제안한다. 실험 비교에서 `GRAM+`는 `GRAM`과 동일하지만 basic embedding $\mathbf{e}_i$를 별도 초기화 전략으로 초기화한 버전이다. 저자 해석에 따르면 이 initialization은 co-occurrence 정보를 더 많이 주입하므로, 순차적 diagnosis prediction에서는 더 유리할 수 있지만 HF prediction에서는 개선이 제한적일 수 있다. 즉 initialization 효과는 데이터셋보다는 **task nature**에 더 민감하다.  

## 4. Experiments and Findings

### 4.1 실험 구성

논문은 세 가지 task를 평가한다.

* 두 개의 sequential diagnoses prediction
* 하나의 heart failure onset prediction

Knowledge DAG로는 CCS multi-level diagnoses hierarchy를 사용했고, ICD-9 hierarchy도 시험했지만 성능은 유사했다고 한다. 데이터는 train/validation/test를 0.75/0.10/0.15로 분할했다.

### 4.2 예측 성능

논문의 핵심 메시지는 GRAM이 **희귀 질환일수록 더 큰 이점**을 보인다는 것이다. 초록과 서론에 따르면, 기본 RNN 대비 rare disease prediction에서 최대 10% 더 높은 정확도를 달성했다. 또한 HF prediction에서는 training data를 10배 적게 쓰고도 약 3% 더 높은 AUC를 얻었다. 이는 ontology를 representation에 통합한 방식이 low-resource setting에서 특히 강력하다는 것을 보여준다.

Table 2 설명에 따르면, sequential diagnosis prediction에서는 label frequency percentile별 성능을 보고하며, 0–20 구간이 가장 희귀한 diagnoses, 80–100 구간이 가장 흔한 diagnoses다. 이 구성 자체가 저자들의 문제의식과 맞닿아 있다. 즉, 전체 평균보다도 **rare label 구간에서 얼마나 이기는가**가 핵심 평가 포인트다.

HF prediction에서는 AUC를 보고하고, training data size를 변화시키며 scalability도 평가한다. 논문 본문 요약에 따르면, GRAM은 적은 데이터로도 성능을 유지하거나 개선해, ontology가 사실상 sample efficiency를 높이는 역할을 했다고 볼 수 있다.  

### 4.3 Interpretable representation

질적 평가에서 저자들은 GRAM과 GRAM+가 learned representation을 medical knowledge DAG와 더 잘 정렬시킨다고 주장한다. 특히 co-occurrence만 쓰는 방식이나 supervised prediction만으로는 이런 해석 가능한 구조를 얻기 어렵다고 말한다. representation visualization에서 GRAM은 비슷한 의료 개념들을 더 구조적으로 묶는다고 설명한다.  

이 점은 의료 AI 맥락에서 중요하다. 단순히 정확도만 높은 black box가 아니라, embedding space 자체가 ontology와 맞아떨어지면 임상적 설득력이 올라간다. 논문은 이걸 단순 주장으로 끝내지 않고, representation visualization과 attention 사례 분석까지 제공한다.

### 4.4 Attention behavior 분석

논문은 Figure 4를 통해 attention behavior를 해석한다. 여기서 가장 흥미로운 점은 **frequency와 sibling structure에 따라 attention 분포가 달라진다**는 것이다. 예를 들어 `Other pneumothorax (ICD9 512.89)`처럼 매우 드물고 sibling 수가 적은 경우에는 높은 ancestor에서 대부분의 정보가 온다. 반면 `Temporomandibular joint disorders & articular disc disorder (ICD9 524.63)`처럼 드물지만 sibling 수가 많은 경우에는 parent가 더 큰 attention을 받아 더 많은 샘플을 집계하면서도, leaf도 어느 정도 attention을 받아 형제들과 구별되도록 한다.  

또한 `Unspecified essential hypertension (ICD9 401.9)`처럼 매우 자주 관측되는 질환은 leaf node에 강한 attention이 주어진다. 저자들은 이것이 논리적이라고 설명한다. 충분히 자주 본 질병은 그 자체 representation을 강하게 신뢰해도 되기 때문이다. 이 attention behavior는 GRAM이 단순히 ontology를 고정적으로 섞는 모델이 아니라, **관측 빈도와 ontology 구조를 동시에 고려하는 adaptive model**이라는 점을 잘 보여준다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

가장 큰 강점은 문제 설정과 방법이 아주 잘 맞물린다는 점이다. 의료 데이터는 long-tail 분포를 가지므로 희귀 코드 representation이 약해지기 쉽다. GRAM은 이를 ontology ancestor attention으로 자연스럽게 보완한다. 또한 이 과정이 end-to-end 학습으로 이루어져, 수작업 규칙보다 유연하다.

두 번째 강점은 interpretability다. 단순히 attention을 붙였다고 해석 가능하다고 주장하는 것이 아니라, 실제로 representation이 의료 ontology와 더 잘 정렬되고, rare/frequent disease에 따라 attention이 직관적으로 달라진다는 qualitative evidence를 제공한다. 의료 분야에서는 이 점이 성능 향상만큼 중요하다.  

세 번째는 sample efficiency다. HF prediction에서 10배 적은 학습 데이터로도 더 좋은 AUC를 달성했다는 결과는, ontology가 사실상 데이터 부족을 완화하는 inductive bias로 작동했음을 시사한다.

### Limitations

한계도 있다. 첫째, ontology 품질에 크게 의존한다. CCS나 ICD hierarchy가 잘 정리돼 있으므로 효과가 있었지만, ontology가 부정확하거나 실제 환자 상태와 잘 맞지 않으면 attention mixture도 왜곡될 수 있다.

둘째, 이 논문은 주로 **parent-child hierarchy**에 초점을 둔다. 저자들도 SNOMED-CT처럼 더 복잡한 관계형 ontology가 있지만, 이 연구에서는 parent-child만 사용한다고 명시한다. 따라서 richer relation graph까지 fully 활용한 것은 아니다.

셋째, GRAM은 concept representation을 강하게 다루지만, 현대 기준에서 보면 transformer나 graph neural network 계열과의 비교는 없다. 물론 이는 논문 시기를 감안해야 한다. 2016년 맥락에서는 매우 선구적이지만, 오늘날 기준으로는 sequence modeling backbone 자체를 더 강하게 바꿔볼 여지가 있다.

### Critical interpretation

비판적으로 보면, GRAM의 진짜 기여는 단순히 “attention을 썼다”가 아니라, **의료 ontology를 representation uncertainty의 보정 장치로 사용했다**는 데 있다. 희귀한 개념일수록 상위 개념으로 일반화하고, 흔한 개념일수록 leaf specificity를 유지한다는 아이디어는 의료 지식과 통계적 학습을 잘 연결한다. 이 점에서 GRAM은 후속 의료 representation learning 연구의 중요한 출발점으로 볼 수 있다.

## 6. Conclusion

GRAM은 의료 예측에서 deep learning의 두 난제인 **data insufficiency**와 **interpretability**를 동시에 겨냥한 모델이다. 핵심은 각 의료 개념을 자기 자신과 ontology ancestor의 attention-weighted combination으로 표현하는 것이다. 이렇게 하면 rare concept는 상위 개념의 일반성을 활용해 더 안정적으로 표현되고, frequent concept는 leaf specificity를 유지할 수 있다.

실험적으로 GRAM은 두 개의 sequential diagnosis prediction과 하나의 HF prediction에서 기본 RNN 대비 개선을 보였고, 특히 희귀 질환 예측과 적은 데이터 환경에서 강점을 드러냈다. representation은 ontology와 더 잘 정렬되었고, attention behavior도 직관적으로 해석 가능했다. 따라서 이 논문은 의료 AI에서 **knowledge-aware representation learning**의 대표적 초기 연구로 볼 수 있다.  

## Source

Canonical arXiv URL: `https://arxiv.org/abs/1611.07012`

```json
{"title": "GRAM: Graph-based Attention Model for Healthcare Representation Learning", "author": "Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F. Stewart, Jimeng Sun", "year": 2016, "url": "https://arxiv.org/abs/1611.07012", "summary": "gram_graph_based_attention_model_for_healthcare_representation_learning.md", "slide": ""}
```
