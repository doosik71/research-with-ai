# A Survey on Deep Active Learning: Recent Advances and New Frontiers

## 1. Paper Overview

이 논문은 **Deep Active Learning(DAL)** 분야를 체계적으로 정리한 survey입니다. Active Learning은 적은 수의 라벨만으로도 높은 성능을 얻기 위해, 모델이 “가장 정보가 많은 샘플”을 선택하고 인간 annotator(oracle)가 그 샘플에 라벨을 부여하는 **human-in-the-loop** 학습 방식입니다. 저자들은 기존 survey가 특정 태스크나 도메인에 치우치거나, 최근 딥러닝 기반 발전과 새로운 학습 패러다임을 충분히 반영하지 못했다고 보고, 이를 보완하기 위해 **DAL의 정의, 주요 baseline, 데이터셋, 방법론 taxonomy, 응용 분야, 도전 과제와 미래 방향**을 한 번에 묶어 제시합니다. 특히 이 논문은 pool-based DAL에 초점을 맞추고, annotation type, query strategy, deep model architecture, learning paradigm, training process의 **5개 축 taxonomy**를 중심으로 분야 전체를 재구성합니다.  

DAL이 중요한 이유는 딥러닝의 성공이 대규모 human-annotated dataset에 크게 의존하는데, 실제 라벨링은 시간·비용·노동 측면에서 병목이 되기 때문입니다. 논문은 DAL이 이러한 병목을 줄이면서도 competitive performance를 유지하도록 돕는 핵심 접근이라고 설명합니다. 또한 GNN, CNN, Transformer, CLIP, GPT 같은 최신 딥러닝/사전학습 모델의 발전이 DAL의 성능과 적용 범위를 더 넓혀 왔다고 봅니다.

---

## 2. Core Idea

이 논문의 핵심은 **새로운 DAL 알고리즘 하나를 제안하는 것이 아니라, DAL 연구 전체를 해부하는 “지도(map)”를 제공하는 것**입니다. 그 지도는 두 가지 층위로 구성됩니다.

첫째, 저자들은 DAL을 단순히 “불확실한 샘플을 고르는 기법”으로 보지 않고,

* 어떤 형태의 annotation을 요구하는가
* 어떤 query strategy를 쓰는가
* 어떤 neural architecture와 결합되는가
* 어떤 learning paradigm과 섞이는가
* 어떤 training process를 가지는가

라는 다차원 관점에서 재분류합니다. 이 덕분에 DAL을 uncertainty sampling의 확장으로만 보는 좁은 시각에서 벗어나, **딥러닝 시대의 데이터 선택 전략**으로 재해석합니다.

둘째, 이 survey는 단순 나열이 아니라 **방법 간 trade-off와 실전 지침**을 추출합니다. 예를 들어 저자들은 uncertainty-based, representative-based, Bayesian-based, influence-based, hybrid query가 각각 장단점이 있으며, 대규모 pre-trained model 시대에는 Pre-training + Fine-tuning, curriculum-like training, semi-supervised integration이 점점 더 중요해진다고 강조합니다. 또한 DAL의 한계로 pipeline 문제, task 전이 문제, noisy oracle과 outlier 문제를 체계적으로 묶어냅니다.  

요약하면, 이 논문의 핵심 기여는 다음과 같습니다.

1. 최신 DAL 논문들을 광범위하게 수집·선별해 survey의 기반을 단단히 마련함
2. DAL 방법을 5개 관점에서 taxonomy화함
3. 응용, 한계, 향후 과제를 같은 프레임 안에서 연결함
4. “어떤 DAL 방법을 언제 써야 하는가”에 대한 해석적 가이드를 제공함  

---

## 3. Detailed Method Explanation

### 3.1 DAL의 기본 정의와 파이프라인

논문은 DAL의 일반 절차를 다음과 같이 설명합니다.

1. 큰 unlabeled pool $\mathcal{D}_{\textbf{pool}}$ 이 존재한다.
2. 초기 샘플 집합 $\mathcal{Q}_0$ 를 선택해 초기 training set $\mathcal{D}_{\textbf{train}}^0$ 를 만든다.
3. 모델 $\mathcal{M}_0$ 를 학습한다.
4. query function $\alpha$ 를 사용해 unlabeled pool에서 informative sample batch를 선택한다.
5. oracle이 선택된 샘플에 라벨을 부여한다.
6. 이를 training set에 추가해 모델을 재학습 또는 fine-tuning한다.
7. 정해진 iteration 수 $T$ 또는 stopping condition까지 반복한다.

즉, DAL의 본질은 **모든 데이터에 라벨이 있다고 가정하는 데이터 선택이 아니라**, 라벨이 없는 pool에서 **어떤 샘플을 먼저 라벨링할지 결정하는 sequential decision process**입니다. 이 점이 dataset distillation, pruning, augmentation, curriculum learning과 닮았으면서도 본질적으로 다른 지점입니다. 저자들은 특히 다른 data-centric 방법들이 대체로 전체 라벨 접근권을 가정하는 반면, DAL은 **selection 시점에는 라벨이 없다**는 점을 강조합니다.

---

### 3.2 논문의 survey methodology

이 survey는 단순 문헌 소개가 아니라, 비교적 엄격한 논문 수집·필터링 절차를 사용합니다. 저자들은 다양한 keyword 조합으로 Google Scholar, Scopus, Semantic Scholar, Web of Science를 검색했고, 검색 기간은 **2013년 1월부터 2023년 3월까지**입니다. 각 query당 최대 200편을 수집해 총 10,000편을 모았고, 중복 제거 후 3,967편, 초록 수동 점검 후 1,273편, 최종 정제 후 405편을 systematic analysis 대상으로 삼았으며, 최종적으로 **220편을 중점적으로 요약·논의**합니다. 이 과정은 survey의 범위를 넓히는 동시에, 질 낮거나 비관련 논문을 배제하려는 시도로 볼 수 있습니다.  

이 methodology는 survey 논문으로서 꽤 강점이 있습니다. 단순히 “대표 논문 몇 편”만 모은 것이 아니라, 명시적 search/filtering pipeline을 제시했기 때문에 독자가 survey coverage를 어느 정도 신뢰할 수 있습니다. 다만 여전히 최종 선정에는 수동 검사와 저널/컨퍼런스 영향력 판단이 개입되므로, 완전한 객관성보다는 **정교한 but curated survey**에 가깝습니다.  

---

### 3.3 DAL taxonomy: 5개의 축

#### (1) Annotation Type

논문은 annotation type을 hard, soft, hybrid, explanatory, random/multi-agent annotation 등으로 나눕니다. 이 분류는 “샘플을 어떤 방식으로 라벨링하느냐”를 중심으로 합니다. 이는 DAL을 단순 query 문제에서 annotation design 문제까지 확장해서 바라보게 해 줍니다.

#### (2) Query Strategy

Query strategy는 논문 전체에서 가장 중심적인 축 중 하나입니다. 저자들은 query 전략을 크게

* uncertainty-based
* representative-based
* influence-based
* Bayesian-based
* hybrid

로 나눕니다. 특히 hybrid는 다시 serial-form, criteria-selection, parallel-form 등으로 세분화합니다. 예를 들어 DUAL은 iteration마다 density-based와 uncertainty-based selector 사이를 전환하고, parallel-form hybrid는 여러 query criterion을 weighted sum이나 multi-objective optimization으로 통합합니다. 이는 실제 DAL에서 “한 기준만으로는 부족하다”는 점을 잘 보여줍니다.  

#### (3) Deep Model Architecture

모델 구조 측면에서는 RNN, CNN, GNN, pre-trained method로 구분합니다. 이는 같은 query strategy라도 backbone architecture에 따라 효과가 달라질 수 있음을 반영합니다. 예를 들어 NLP와 CV, graph domain에서 informative sample의 정의가 달라질 수 있기 때문에, query function은 model representation과 강하게 얽혀 있습니다.

#### (4) Learning Paradigm

저자들은 DAL이 curriculum learning, continual learning, imitation learning, multi-task learning, meta learning, transfer learning, semi-supervised learning 등과 결합되는 흐름을 중요하게 다룹니다. 예를 들어 imitation learning은 expert-like query policy를 학습하는 방식이고, multi-task DAL은 여러 task classifier의 uncertainty를 함께 고려합니다. 이 부분은 DAL이 더 이상 독립적 기법이 아니라, **다른 학습 패러다임과 결합해 확장되는 메타 프레임워크**가 되고 있음을 보여줍니다.  

#### (5) Training Process

Training process는 traditional training, curriculum-learning-based training, pre-training & fine-tuning(Pre+FT)로 정리됩니다. 저자들은 특히 대형 pre-trained model 시대에는 Pre+FT가 DAL의 실질적 활용성과 잘 맞는다고 봅니다. 이는 샘플 선택의 가치가 “작은 모델을 처음부터 학습하는 것”보다 “거대한 foundation model을 적은 라벨로 fine-tune하는 것”에서 더 커질 수 있다는 시각입니다.  

---

### 3.4 주요 baseline 정리

논문은 DAL의 대표 baseline들을 연대기적으로 소개합니다. 예를 들어

* **BCBA**: Bayesian neural network와 Monte Carlo dropout을 활용
* **DBAL**: 고차원 이미지 분류를 위한 uncertainty-based query
* **CEAL**: high-confidence sample에는 pseudo-label을 붙이고, uncertain sample만 인간이 라벨링
* **ESNN**: ensemble 기반 uncertainty 측정으로 robustness 향상
* **CoreSet**: batch selection에서 전체 데이터 분포를 잘 덮는 샘플 선택
* **BatchBALD**: mutual information 기반 batch informativeness 추정  

또한 meta/transfer 계열에서는

* **LAL**: downstream task를 위한 query strategy regressor 학습
* **MAML + DAL**: meta-learned initialization으로 active learner 초기화
* **DLER**: high-resource에서 low-resource로 transferable model 설계
* **AADA**: domain alignment, uncertainty, diversity를 함께 고려합니다.

이 정리는 DAL 발전사를 “uncertainty 중심 초기 방법 → batch/diversity 보완 → semi-supervised/meta/transfer/pretrained 확장”이라는 흐름으로 이해하게 해 줍니다.

---

## 4. Experiments and Findings

### 4.1 이 논문의 “실험”은 무엇인가

이 논문은 새로운 DAL method를 제안하고 benchmark에서 실험하는 논문이 아니라 survey이므로, 전통적인 의미의 method-vs-baseline 실험은 없습니다. 대신 저자들은 **수집한 대규모 문헌을 바탕으로 empirical trend와 field-level finding**을 제시합니다. 그러므로 이 논문의 실험적 결과는 개별 수치가 아니라, **DAL 분야 전체에서 반복적으로 관찰되는 패턴을 요약한 메타 수준의 finding**이라고 보는 것이 맞습니다.  

### 4.2 주요 발견

논문이 도출한 중요한 관찰은 다음과 같습니다.

첫째, 저자들은 **대규모 pre-trained language model과 DAL의 결합 가능성**을 강조합니다. survey에 따르면 일부 연구에서는 전체 라벨 데이터셋 대신 **10~20% 정도의 labeled sample만으로 fine-tuning해도 더 나은 성능을 내거나, full labeled training보다 5~10배 효율적일 수 있다**고 정리합니다. 이는 foundation model 시대에 DAL의 실용성이 커지고 있음을 의미합니다.

둘째, semi-supervised learning과 DAL의 결합은 매력적이지만, 동시에 **pseudo-label error와 outlier가 악순환을 일으킬 수 있는 위험**이 있습니다. 즉, uncertain하거나 open-set 샘플을 잘못 pseudo-labeling하면 이후 query-selection과 retraining이 더 왜곡될 수 있습니다. 저자들은 이것을 중요한 open problem으로 봅니다.  

셋째, 실제 응용에서 어떤 DAL 방법이 최적인지 고르려면 많은 비교 실험이 필요해 **실무 적용 비용이 크다**고 지적합니다. 그래서 다양한 downstream task에 잘 맞는 **universal framework**에 대한 수요가 크다고 주장합니다.  

넷째, 응용 분야 측면에서 DAL은 NLP, CV, Data Mining 등 여러 도메인에 적용됐지만, NLP에서는 여전히 **classification 중심**이고 **generative task**(예: summarization, QA)는 상대적으로 연구가 부족하다고 봅니다. 저자들은 generation task에서 “무엇이 informative sample인가”를 정의하는 것이 더 어렵기 때문에 추가 연구가 필요하다고 말합니다.

---

### 4.3 Applications

논문은 DAL의 응용을 NLP, CV, DM 등으로 정리합니다. 예를 들어 CV에서는 semantic segmentation, object detection, pose estimation 같은 비용이 큰 annotation task에서 DAL이 labeling cost를 줄이는 데 특히 유용하다고 봅니다. object detection에서는 uncertainty와 diversity를 결합한 hybrid 전략, partially labeled/unlabeled sample을 활용하는 active sample mining, adversarial instance classifier discrepancy를 이용한 uncertainty estimation 등이 소개됩니다. pose estimation에서도 diverse/representative sample selection이나 uncertainty-based 전략이 유효하다고 설명합니다.  

---

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **범위와 구조화 수준**입니다. 단순히 대표 논문을 요약하는 데서 끝나지 않고, 검색·필터링 절차를 명시했으며, DAL을 5개 관점 taxonomy로 정리해 독자가 분야를 빠르게 파악할 수 있게 합니다. 또한 baselines, datasets, applications, challenges를 따로따로 두지 않고 한 프레임 안에서 연결했기 때문에, 초심자에게는 입문 지도, 연구자에게는 research gap 탐색 도구가 됩니다.  

또 하나의 강점은 **최근 패러다임 변화 반영**입니다. 초기 AL survey들이 uncertainty sampling 위주였다면, 이 논문은 pre-trained model, semi-supervised integration, curriculum/continual/meta/transfer learning까지 포괄합니다. 그래서 “DAL이 어디까지 확장되었는가”를 이해하는 데 도움이 큽니다.

### Limitations

반면 한계도 분명합니다.

첫째, 이 논문은 survey이기 때문에 각 방법의 수치적 비교를 동일 실험 조건에서 직접 검증하지 않습니다. 따라서 독자가 “내 문제에 어떤 방법이 실제로 제일 좋은가”를 바로 알 수는 없습니다.
둘째, 분류 체계가 매우 넓어지는 대신, 개별 하위 방법들의 수학적 세부사항이나 재현 세부설정은 깊게 다루지 못합니다.
셋째, 최종 논문 선정과 영향력 평가에 수동적 판단이 들어가므로, survey coverage가 완전히 중립적이라고 보긴 어렵습니다.
넷째, 저자들 스스로 인정하듯 generative task, stopping strategy, cold-start, cross-domain transfer, noisy oracle, class distribution mismatch 같은 문제는 여전히 풀리지 않았습니다.

### Interpretation

제 해석으로는, 이 논문은 DAL을 단순한 sample selection trick이 아니라 **foundation model 시대의 데이터 효율화 프레임워크**로 재정의하려는 survey입니다. 특히 “Pre+FT”, “few-shot/one-shot with large models”, “semi-supervised coupling”, “universal framework”에 대한 강조를 보면, 저자들은 DAL의 미래를 작은 CNN 실험이 아니라 **large model adaptation 문제**와 연결해서 보고 있습니다. 이 점은 현재의 데이터 중심 AI 흐름과 잘 맞습니다.  

---

## 6. Conclusion

이 논문은 Deep Active Learning 분야를 넓고 깊게 정리한 최신 survey로서, DAL의 정의, 파이프라인, 중요한 baseline과 dataset, 5개 축 taxonomy, 응용 분야, 그리고 핵심 도전 과제를 한 번에 제공합니다. 핵심 메시지는 분명합니다.

* DAL은 라벨 비용을 줄이면서 competitive performance를 달성하려는 데이터 중심 딥러닝의 핵심 기법이다.
* DAL 연구는 uncertainty sampling을 넘어, hybrid strategy, pre-trained model, semi-supervised integration, meta/transfer learning으로 빠르게 확장되고 있다.
* 실용적 병목은 여전히 annotation cost, stopping rule, cold-start, cross-domain transfer, noisy oracle, outlier, scalability/generalizability에 있다.
* 앞으로는 universal DAL framework, generation task용 DAL, large pre-trained model과의 결합이 중요한 연구 전선이 될 가능성이 크다.

이 논문은 DAL을 처음 공부하는 사람에게는 매우 좋은 입문서이고, 이미 연구 중인 사람에게는 **어디가 비어 있고 다음에 무엇을 풀어야 하는지**를 보여주는 지형도 역할을 합니다.

## Source

[https://arxiv.org/abs/2405.00334](https://arxiv.org/abs/2405.00334)
