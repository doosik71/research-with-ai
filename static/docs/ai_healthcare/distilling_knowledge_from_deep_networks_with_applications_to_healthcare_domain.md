# Distilling Knowledge from Deep Networks with Applications to Healthcare Domain

이 논문은 의료 예측에서 자주 충돌하는 두 목표, 즉 **높은 예측 성능**과 **임상적 해석 가능성**을 동시에 달성하려는 시도다. 저자들은 deep model 자체를 직접 해석하려 하기보다, 먼저 강한 deep model을 학습한 뒤 그 출력을 **Gradient Boosting Trees(GBT)**가 모사하도록 하는 **Interpretable Mimic Learning** 프레임워크를 제안한다. 핵심 주장은 “deep network의 예측력을 유지하면서도, tree 기반 모델의 규칙성과 feature importance를 통해 해석 가능한 phenotype을 얻을 수 있다”는 것이다.  

## 1. Paper Overview

이 논문이 다루는 문제는 Computational Phenotyping과 clinical prediction에서 매우 직접적이다. EHR가 빠르게 축적되면서 질병 패턴을 데이터 기반으로 발견할 기회가 커졌지만, 실제 의료 현장에서는 **단순히 잘 맞는 모델보다 설명 가능한 모델**이 더 중요할 수 있다. 논문은 기존 deep learning이 phenotype prediction에서 우수한 성능을 내더라도, 임상의가 왜 그런 결정을 내렸는지 이해하기 어렵다는 점을 핵심 문제로 설정한다.

저자들은 이 문제를 “deep model을 더 투명하게 만들자”가 아니라, **deep model의 지식을 해석 가능한 다른 모델로 distill하자**는 방향으로 푼다. 즉, teacher는 SDA나 LSTM 같은 deep network이고, student는 GBT다. 이 접근은 일반적인 knowledge distillation 아이디어를 의료 domain에 맞게 변형한 것이다.

이 문제가 중요한 이유는 의료 의사결정이 단순 자동화가 아니라 **설명 책임(accountability)**과 밀접하기 때문이다. 논문도 healthcare에서는 interpretability가 단순히 바람직한 성질이 아니라 사실상 **필수적**이라고 강조한다. 임상가는 새 시스템을 채택할 때 성능뿐 아니라 어떤 신호가 근거인지 확인하고 싶어 한다.  

## 2. Core Idea

논문의 핵심 아이디어는 매우 명확하다. 먼저 deep model이 복잡한 정적/시계열 EHR 데이터에서 강한 predictive representation을 학습한다. 그다음 이 deep model의 출력을 이용해, 보다 해석 가능한 GBT가 그 행동을 **mimic**하도록 학습한다. 저자들은 이 전체 과정을 **Interpretable Mimic Learning**이라 부른다.

이 논문에서 novelty는 두 가지다.

첫째, mimic learning을 단순 compression 목적이 아니라 **해석 가능한 clinical phenotype 추출**을 위해 사용했다는 점이다. 기존 mimic learning은 주로 작은 모델로 성능을 복제하는 데 초점이 있었는데, 이 논문은 student로 GBT를 택해 tree structure, decision rule, feature importance까지 확보하려 한다.

둘째, teacher의 knowledge를 student에게 전달하는 방식을 두 가지로 나눠 실험한다. 하나는 deep model의 **hidden feature**를 추출해서 다른 classifier와 연결하는 방식이고, 다른 하나는 teacher가 내는 **soft target**을 student가 직접 모사하는 방식이다. 논문은 서로 다른 deep architecture에서 이 둘의 효과가 어떻게 다른지도 함께 비교한다.  

결국 이 논문의 중심 직관은 다음과 같다.
**“deep network가 학습한 지식은 black box 안에 묻어둘 필요가 없고, tree 기반 모델로 옮겨오면 성능을 상당 부분 유지하면서도 임상적으로 해석 가능한 규칙으로 바꿀 수 있다.”**

## 3. Detailed Method Explanation

### 3.1 입력 데이터 형식

논문은 각 샘플이 정적 변수와 시간축 변수 모두를 가진다고 가정한다. 정적 변수 개수를 $Q$, 시간 길이를 $T$, 시계열 변수 수를 $P$라고 두면, 전체 입력은 시간축을 펼쳐(flatten) 정적 변수를 붙인 벡터로 표현된다.

$$
\mathbf{X} \in \mathbb{R}^{D}, \quad D = TP + Q
$$

즉, 일부 모델은 전체 longitudinal record를 하나의 큰 feature vector로 처리하고, 일부 모델은 time-series 구조를 보존해 순차 모델에 넣는다. 이 설정은 이후 DNN/SDA와 LSTM의 차이를 이해하는 데 중요하다.

### 3.2 Teacher model 1: Feedforward Network / DNN

논문은 기본 multilayer feedforward network를 먼저 설명한다. 각 layer의 변환은 다음과 같다.

$$
\mathbf{X}^{(l+1)} = f^{(l)}(\mathbf{X}^{(l)}) = s^{(l)}\left(\mathbf{W}^{(l)}\mathbf{X}^{(l)} + \mathbf{b}^{(l)}\right)
$$

여기서 $s^{(l)}$는 sigmoid, tanh, ReLU 같은 nonlinear activation이다. 최상단 prediction layer는 cross-entropy loss로 학습되며, 중간 hidden activation은 단순 내부 상태가 아니라 downstream mimicry에 사용할 수 있는 learned feature로 간주된다.

이 논문에서 DNN은 단지 baseline이 아니다. “deep hidden representation이 clinical phenotype-like signal을 담고 있는가”를 보는 teacher 역할을 한다.

### 3.3 Teacher model 2: Stacked Denoising Autoencoder (SDA)

SDA는 구조적으로 feedforward network와 유사하지만, prediction loss 대신 **입력 복원(reconstruction)**을 목표로 한다. decoder 쪽의 복원식은 다음과 같이 정리된다.

$$
\mathbf{Z}^{(l)} = s^{(l)}\left((\mathbf{W}^{(l)})^T \mathbf{Z}^{(l+1)} + \mathbf{b}_d^{(l)}\right)
$$

그리고 encoder 최상단 표현을 얻은 뒤, 그 위에 logistic prediction layer를 추가해 실제 예측 task를 푼다. Denoising을 사용한다는 것은 일부 입력을 훼손한 뒤 원래 입력을 복원하도록 학습한다는 뜻이며, 이는 noisy clinical data에 대해 보다 robust한 feature를 학습하기 위한 장치다.

의미상 SDA는 raw flattened EHR에서 latent structure를 추출하고, 이 latent representation이 phenotype discovery에 유용할 수 있다는 가정 위에 있다.

### 3.4 Teacher model 3: LSTM

논문은 시계열 구조를 더 직접 반영하기 위해 LSTM도 teacher로 사용한다. LSTM은 temporal variables만을 대상으로 입력 시퀀스를 순차적으로 읽고, hidden states를 시간 순으로 만든다. 이후 sequence output 위에 prediction layer를 올려 분류를 수행한다. LSTM에서 mimic에 사용할 feature를 뽑을 때는 **flattened sequence output**을 사용한다고 논문이 명시한다.

이 부분은 중요하다. 논문 후반 해석에서도 LSTM은 다른 flatten-based 모델과 달리 temporal dependency를 직접 다루기 때문에, mimic 관점에서 hidden feature를 그대로 넘기는 것이 항상 최선은 아닐 수 있다고 본다. 즉, time structure를 한 번 펼쳐버리면 LSTM의 장점 일부가 사라질 수 있다.

### 3.5 Student model: Gradient Boosting Trees

논문이 해석 가능한 mimic model로 선택한 것은 GBT다. 이유는 다음과 같다.

* tree 구조와 decision rule이 비교적 해석 가능하다
* boosting을 통해 단일 decision tree보다 더 강한 성능을 낼 수 있다
* original deep model의 동작을 꽤 잘 근사할 수 있다

구현 세부로는 shrinkage rate 0.1, boosting stage 최대 100, 각 tree depth 3을 사용한다. 개별 tree를 얕게 유지해 과도한 복잡성을 줄이고, boosting ensemble로 성능을 확보하는 설계다.

### 3.6 Interpretable Mimic Learning의 두 파이프라인

논문은 GBTmimic을 두 방식으로 구성한다. 스니펫상 Figure 2로 설명되며, 본질은 다음과 같이 요약할 수 있다.

#### Pipeline 1: Deep feature extraction + classifier

먼저 deep network의 hidden representation을 feature로 추출하고, 여기에 Logistic Regression 같은 별도 classifier를 붙인다. 이때 mimic model 이름에 `LR`이 붙는 경우가 많다. 예를 들어 `GBTmimic-LR-SDA`는 SDA에서 얻은 feature와 logistic layer의 결과를 활용해 GBT가 mimic하도록 만든 모델로 이해할 수 있다.

이 방식은 “teacher의 internal representation이 얼마나 informative한가”를 보려는 관점이다.

#### Pipeline 2: Soft target mimicry

두 번째는 teacher deep model이 내는 soft probability target을 직접 mimic하는 것이다. 이것은 고전적인 knowledge distillation에 더 가깝다. hard label보다 soft target은 클래스 간 유사도나 teacher의 불확실성 정보를 담고 있기 때문에 student에게 richer signal을 전달할 수 있다. 논문도 mimic learning과 dark knowledge 계열 배경을 그대로 따른다.

### 3.7 모델 이름의 해석

논문에서 자주 등장하는 표기는 다음처럼 읽으면 된다.

* `GBTmimic-LSTM`: LSTM teacher를 GBT가 모사
* `GBTmimic-LR-SDA`: SDA의 learned feature + LR prediction을 바탕으로 GBT mimic
* `DTmimic-*`: GBT 대신 Decision Tree를 student로 사용한 비교군

즉, 이 논문은 “해석 가능한 mimic”이라는 큰 아이디어 아래서, teacher와 student 조합을 체계적으로 비교한다.

## 4. Experiments and Findings

### 4.1 실험 설정

실험은 real-world clinical time-series dataset에서 수행되며, 논문은 두 classification task를 강조한다.

* `MOR`: mortality prediction
* `VFD`: ventilator-free days 관련 분류 과제

또한 모든 알고리즘은 **5 random trials의 5-fold cross validation**으로 평가한다. DNN/SDA는 hidden layer 2개와 prediction layer 1개를 사용하고, 50 epochs SGD로 학습한다. LSTM은 temporal feature만 사용하며 50 epochs RMSprop으로 학습한다.  

이 설정은 논문의 결론을 해석할 때 중요하다. 즉, 단순히 한 번 잘 나온 결과가 아니라 반복 CV 평균을 본다는 점에서 결과의 안정성을 어느 정도 확보하려 했다.

### 4.2 핵심 정량 결과

논문이 가장 강하게 주장하는 실험 결과는 다음이다.

* deep models는 standard machine learning baseline보다 대체로 더 낫다
* 하지만 **Interpretable Mimic Learning은 deep model과 비슷하거나 더 좋은 성능**을 낼 수 있다
* MOR task에서는 `GBTmimic-LR-SDA`가 최고 성능
* VFD task에서는 `GBTmimic-LR-DNN`이 최고 성능

즉, teacher가 deep model이라고 해서 student가 반드시 성능 손실을 감수해야 하는 것은 아니며, 오히려 특정 조합에서는 student가 더 나은 generalization을 보일 수 있다는 것이 논문의 메시지다.

### 4.3 LSTM 관련 해석

논문은 흥미롭게도, DNN이나 SDA에서는 learned feature + Logistic Regression 조합이 원래 deep model보다 더 좋아질 수 있었지만, **LSTM에서는 그런 경향이 약하다**고 설명한다. 저자들의 해석은 합리적이다. LSTM은 temporal dependency 자체를 활용해 예측하는데, 이 시계열 구조를 flatten한 feature로 바꾸는 순간 그 장점이 일부 사라질 수 있다는 것이다.

이 결과는 단순 성능 비교 이상으로 중요하다. 즉, “모든 deep representation이 똑같이 distillable하지는 않다”는 뜻이다. sequence teacher의 내부 지식은 feedforward teacher보다 student에게 옮기기 더 까다로울 수 있다.

### 4.4 예시 수치

보이는 테이블 조각에서도 몇 가지 수치가 확인된다. 예를 들어 `GBTmimic-LR-LSTM`는 한 실험 표에서 AUC 0.7565, 다른 설정 표에서 0.7263 수준이 보이며, `DTmimic-LR-LSTM`는 이보다 더 낮은 수치를 보인다. 이는 적어도 LSTM teacher를 mimic할 때 **GBT가 단순 DT보다 더 강한 student**라는 점과 일치한다. 다만 제공된 표 일부는 잘려 있어 전체 metric 정의와 모든 비교군을 완전하게 복원하기는 어렵다. 따라서 수치 해석은 논문 본문이 직접 요약한 수준까지만 신뢰하는 것이 적절하다.  

### 4.5 해석 가능성 평가

논문은 단지 AUC만 비교하지 않고, 해석 가능성도 따로 본다. GBT와 GBTmimic이 학습한 top useful features를 비교하고, MOR task에서 중요한 tree를 시각화해 original GBT와 mimic tree가 **공통 feature와 유사한 rule**을 공유함을 보여준다고 설명한다. 저자들은 이런 feature와 decision rule이 healthcare experts에 의해 평가될 수 있으며, 모델 이해와 의사결정 개선에 도움을 준다고 주장한다.  

이 대목이 논문의 실질적 공헌을 잘 보여준다. 이 논문의 목적은 단순 압축이 아니라, **deep model의 성능을 임상의가 읽을 수 있는 규칙 형태로 번역**하는 데 있다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 문제 정의가 정확하다는 점이다. 의료에서 해석 가능성은 부가 옵션이 아니라 필수 제약인데, 저자들은 이를 정면으로 다룬다. 그리고 deep model을 직접 억지로 설명하려 하기보다, **teacher-student distillation**이라는 우회 경로를 택해 practical solution을 제안했다.  

또 다른 강점은 실험 설계다. SDA, DNN, LSTM처럼 서로 다른 deep teacher를 비교하고, soft target과 hidden feature 활용도 함께 점검한다. 따라서 “mimic learning이 된다” 정도가 아니라, **어떤 teacher에서 어떤 distillation 방식이 더 잘 먹히는가**를 어느 정도 보여준다.  

마지막으로, 논문은 해석 가능성을 말로만 주장하지 않고 feature importance와 decision tree 예시를 제시한다. 의료 현장에서 실제로 사용할 수 있는 규칙 형태를 얻는다는 점이 중요하다.

### Limitations

한계도 분명하다.

첫째, mimic model이 해석 가능하다고 해도, 그것이 곧 **teacher deep model의 내부 reasoning을 완전히 복원했다**는 뜻은 아니다. student는 teacher의 입력-출력 관계를 근사할 뿐이며, deep network의 내부 causal mechanism을 보장하지는 않는다.

둘째, 논문의 해석 가능성은 주로 tree rule과 feature importance 수준이다. 의료적으로 얼마나 타당한 phenotype인지에 대한 검증은 일부 전문가 검토에 기대고 있으며, 보다 체계적인 clinical validation은 제한적이다.

셋째, 데이터셋이 하나의 real-world clinical time-series dataset 중심으로 보이며, 저자들 스스로 더 큰 데이터셋(MIMIC-II 등)과 다른 structured deep model로 확장할 계획을 future work로 제시한다. 즉, generalizability는 후속 검증이 필요하다.

### Critical interpretation

비판적으로 보면, 이 논문의 진짜 기여는 “GBT가 deep model을 따라할 수 있다”가 아니다. 더 본질적으로는 **해석 가능성과 성능의 trade-off를 직접 줄이는 구조적 제안**이라는 점에 있다. 오늘날의 XAI 관점으로 보면 이 연구는 post-hoc explanation보다는 **surrogate modeling**에 가깝다. 즉, black box를 억지로 들여다보지 않고, 강한 모델을 해석 가능한 모델로 재표현하는 전략이다.

현대 기준에서는 더 강한 distillation, calibration, uncertainty, transformer teacher 등으로 확장 여지가 많다. 그러나 논문 시점에서는 healthcare domain에서 매우 실용적이고 선구적인 문제 설정이었다고 볼 수 있다.

## 6. Conclusion

이 논문은 의료 예측에서 강력한 deep learning 모델과 임상적으로 수용 가능한 해석 가능성을 연결하기 위해 **Interpretable Mimic Learning**을 제안했다. teacher로 SDA와 LSTM 같은 deep model을 두고, student로 GBT를 학습시켜 성능을 유지하면서도 해석 가능한 rule과 feature를 얻는 방식이다. 논문은 실제 clinical time-series 데이터에서 mimic 방법이 deep model과 비슷하거나 더 나은 성능을 낼 수 있음을 보였고, 특히 GBT 기반 mimic이 decision tree보다 더 강력한 student라는 점도 시사한다.  

실무적으로 이 연구는 “의료 AI는 반드시 black box일 필요가 없다”는 메시지를 준다. 연구적으로는 distillation을 단순 모델 압축이 아니라 **interpretable phenotype discovery**에 연결한 초기 사례로서 의미가 있다. 이후의 healthcare XAI, surrogate modeling, clinically grounded distillation 연구의 출발점 중 하나로 읽을 수 있다.  
