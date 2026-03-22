# BEHRT: Transformer for Electronic Health Records

## 1. Paper Overview

이 논문은 전자의무기록(EHR, Electronic Health Records)을 이용해 환자의 **미래 질환 발생을 다중 과제(multitask) 방식으로 예측**하는 Transformer 기반 모델 **BEHRT**를 제안한다. 저자들은 기존 EHR 예측 모델들이 순차성은 다루더라도 장기 의존성, 방문 간 불규칙한 시간 간격, 다양한 의료 개념의 이질성, 그리고 대규모 질병 동시 예측 문제를 충분히 다루지 못한다고 보고, NLP에서 성공한 BERT/Transformer 구조를 EHR에 맞게 변형해 적용했다. 이 모델은 약 **160만 명 환자 데이터**로 학습·평가되었고, **301개 질환**의 미래 발생 예측에서 기존 딥러닝 EHR 모델 대비 **Average Precision Score 기준 절대 8.0~10.8% 개선**을 보였다고 보고한다.  

이 연구가 중요한 이유는, 의료 현장에서 많은 질환이 증상이 뚜렷해진 뒤에야 진단된다는 문제를 완화하려는 데 있다. 조기 예측이 가능하면 조기 개입, 질병 관리, 자원 배분 효율화에 도움이 된다. 논문은 단순히 성능 개선에 그치지 않고, attention을 통해 **질병 trajectory(질병 진행 경로)**와 질환 간 관계를 해석하는 가능성까지 보여주려 한다.

## 2. Core Idea

핵심 아이디어는 **환자의 EHR를 자연어 문서처럼 취급**하는 것이다. 논문은

* 진단 코드(diagnosis)를 “단어(word)”
* 한 번의 방문(visit)을 “문장(sentence)”
* 환자의 전체 병력을 “문서(document)”

처럼 보고, 여기에 **multi-head self-attention**, **positional encoding**, **masked language modeling(MLM)**을 적용한다. 즉, BERT의 문맥적 표현 학습 방식을 의료 시계열 데이터로 옮긴 셈이다.

BEHRT의 차별점은 단순히 BERT를 가져다 쓴 것이 아니라, EHR 특성을 반영해 **4개 embedding**을 결합했다는 점이다.

1. **disease embedding**: 질병 코드 자체의 의미
2. **position embedding**: 시퀀스 내 상대적 위치
3. **age embedding**: 해당 진단/방문 시점의 나이
4. **segment embedding**: 방문 경계 구분

이 중 **age**와 **segment**는 원래 NLP용 BERT에는 없는 요소로, 의료 데이터의 시간성·방문 구조를 반영하기 위한 BEHRT 고유 설계다.  

요약하면, 이 논문의 novelty는 다음처럼 볼 수 있다.

* EHR 예측을 **RNN/CNN 중심에서 Transformer 기반으로 전환**
* 단순 위치 정보 외에 **age/visit segment**를 구조적으로 포함
* MLM pretraining을 통해 **보편적 EHR representation**을 학습
* 이후 이를 **다질환 예측(multi-label disease prediction)**과 질병 phenomapping에 활용

## 3. Detailed Method Explanation

### 3.1 데이터와 문제 설정

논문은 영국의 **CPRD(Clinical Practice Research Datalink)**와 **HES(Hospital Episode Statistics)** 연계 데이터를 사용한다. CPRD는 영국 GP(primary care) 중심의 장기 종단 기록이며, HES는 병원 입원·외래·응급 기록을 포함한다. 저자들은 이 조합이 대규모 환자군의 longitudinal EHR 학습에 매우 적합하다고 설명한다.

전처리 과정은 대략 다음과 같다.

* 시작: 약 **800만 명 환자**
* HES linkage 가능, CPRD quality standard 충족 환자만 유지
* 최소 **5회 이상 방문 기록**이 있는 환자만 유지
* 최종적으로 약 **160만 명**을 학습/평가에 사용

질병 코드는 primary care의 Med Code와 hospital care의 ICD-10을 정렬 가능한 공통 질병 체계로 매핑하고, 최종적으로 **301개 diagnosis code** 집합을 만든다. 이를 논문은 $D={d_i}_{i=1}^{G}$, $G=301$ 로 둔다.

### 3.2 EHR 시퀀스 표현

각 환자 $p$의 병력은 방문들의 순서열로 표현된다. 방문 시퀀스를

$$
V_p = {\mathbf{v}_p^1, \mathbf{v}_p^2, \ldots, \mathbf{v}_p^{n_p}}
$$

처럼 두고, 각 방문 $\mathbf{v}_p^j$ 는 그 방문에 포함된 여러 진단 코드들의 집합이다. 논문은 실제 모델 입력을 만들기 위해 방문을 시간순으로 정렬한 뒤, 문장 분리와 유사한 특수 토큰을 추가한다.

$$
V_p = {CLS, \mathbf{v}_p^1, SEP, \mathbf{v}_p^2, SEP, \ldots, \mathbf{v}_p^{n_p}, SEP}
$$

여기서 `CLS`는 병력 시작, `SEP`는 방문 경계를 나타낸다. 이 표현은 BERT의 문장 입력 형식을 EHR에 맞춘 것이다.  

### 3.3 왜 Transformer인가

저자들은 EHR 모델링의 네 가지 핵심 과제를 제시한다.

* 과거/현재/미래 개념 사이의 복잡한 비선형 상호작용
* 장기 의존성(long-term dependencies)
* 가변 길이의 이질적 의료 개념 표현
* 방문 간 불규칙한 시간 간격

기존 RNN은 긴 시퀀스에서 gradient vanishing/exploding 문제가 있고, CNN은 넓은 수용영역 확보를 위해 깊은 계층이 필요하다. 반면 BEHRT는 **feed-forward + self-attention** 구조라서 **전체 시퀀스를 병렬적으로 보고**, 긴 거리 관계를 직접 학습할 수 있다는 논리를 편다.

### 3.4 BEHRT의 입력 표현

BEHRT의 각 입력 토큰 표현은 4개 embedding의 합으로 구성된다.

$$
\mathbf{e} = \mathbf{e}_{disease} + \mathbf{e}_{position} + \mathbf{e}_{age} + \mathbf{e}_{segment}
$$

논문에 이 식이 명시적으로 적혀 있지는 않지만, Figure 3 설명과 본문 서술상 실제 설계는 이 네 표현의 결합으로 이해하는 것이 정확하다.

각 embedding의 역할은 다음과 같다.

**Disease embedding**
질병 자체의 의미를 담는다. 특정 질환의 과거 이력이 미래 질환의 중요한 위험 신호가 될 수 있으므로 가장 핵심적인 표현이다. 멀티모비디티 패턴이나 trajectory를 담는 기반이 된다.

**Position embedding**
Transformer는 recurrence가 없으므로 순서를 직접 알 수 없다. 이를 보완하기 위해 논문은 원래 Transformer의 positional encoding 규칙을 따른다. 저자들은 의료 시퀀스 길이 분포 불균형 때문에 학습형(trainable)보다 **pre-determined positional encoding**을 사용했다고 설명한다.

**Age embedding**
BEHRT의 매우 중요한 의료 특화 요소다. 나이는 역학에서 강력한 위험 인자이며, 각 진단에 해당 시점 나이를 붙이면 단순 순서뿐 아니라 “환자가 몇 살 때 어떤 사건이 있었는가”라는 **절대적 시간성**을 제공한다. 이는 환자 간 비교 가능성을 높인다.

**Segment embedding**
방문 단위 경계를 구분한다. BERT의 sentence A/B와 유사하지만, 여기서는 visit separation을 위한 구조적 힌트다. 즉 모델이 어떤 진단들이 같은 방문에서 함께 발생했는지, 어떤 진단들이 다른 방문에 속하는지 구분하도록 돕는다.

### 3.5 Pre-training: MLM

BEHRT는 BERT와 유사하게 **Masked Language Modeling** 으로 사전학습된다. 질병 토큰의 **15%를 랜덤 선택**한 후,

1. **80%**는 `[MASK]` 로 교체
2. **10%**는 랜덤 질병 코드로 교체
3. **10%**는 그대로 유지

하는 전략을 사용한다. 즉 주변 문맥을 기반으로 가려진 질병을 복원하게 하여 bidirectional contextual representation을 학습한다.

이 접근의 의미는, BEHRT가 단지 다음 토큰 예측만 하는 것이 아니라 **좌우 문맥을 모두 활용하는 환자 병력 표현기(universal EHR feature extractor)**로 동작한다는 점이다. 논문도 MLM 후의 BEHRT를 downstream task에 작은 추가 학습만으로 적용 가능한 범용 표현 학습기로 해석한다.

### 3.6 Downstream prediction

사전학습된 BEHRT 출력 위에 **단일 feed-forward classifier layer** 를 올려 미래 질환 예측을 수행한다. 논문은 이를 **multi-task / multi-label classification** 으로 설정하며, 질환별로 별도 모델을 만드는 대신 **하나의 예측 모델**이 여러 질환을 동시에 다루도록 한다.  

논문이 강조하는 장점은 다음과 같다.

* 질환 간 관계를 공유 표현으로 학습
* 대규모 질환군(301개)으로 확장 가능
* attention을 통해 장기적 질환 연관성 시각화 가능
* 나중에 medication, tests, interventions 등 다른 개념도 embedding 하나 추가하는 방식으로 확장 가능  

## 4. Experiments and Findings

### 4.1 데이터셋과 비교 기준

실험은 CPRD-HES 기반 대규모 EHR에서 수행되며, 핵심 다운스트림 문제는 **가까운 미래의 질환 예측**이다. 논문은 기존 deep EHR baselines로 대표적으로 **Deepr** 와 **RETAIN** 을 언급하고, 이들과 성능을 비교한다.  

### 4.2 주요 결과

가장 중요한 결과는, BEHRT가 **301개 질환 예측**에서 기존 state-of-the-art deep EHR 모델 대비 **Average Precision Score 기준 절대 8.0~10.8% 향상**을 보였다는 점이다. 이는 abstract와 conclusion 모두에서 반복해서 강조된다.  

또한 appendix figure 설명에 따르면 질환별 precision/AUROC 비교에서도 BEHRT가 전반적으로 Deepr, RETAIN보다 우상단에 위치해 더 좋은 성능을 보인다. 즉 개선이 특정 질환에만 국한되지 않고 **전반적 disease-wise trend**에서도 관찰된다.

### 4.3 실험이 보여주는 것

이 실험이 시사하는 바는 세 가지다.

첫째, **Transformer-style bidirectional pretraining** 이 EHR에서도 유효하다는 점이다. NLP에서 성공한 문맥 표현 학습이 의료 시계열에도 일반화될 수 있음을 보인다.

둘째, 단순히 diagnosis sequence만 넣는 것이 아니라 **age와 visit structure** 를 명시적으로 넣는 것이 실질적 도움이 된다는 주장이다. 논문은 이를 통해 모델이 “환자가 젊은 나이에 X,Y 질환을 겪고, 갑자기 방문 빈도가 늘면서 새로운 질환이 나타나는 패턴” 같은 care process까지 표현할 수 있다고 해석한다.

셋째, attention 분석을 통해 장거리 질환 관계를 포착할 수 있음을 시각적으로 제시한다. 예를 들어 한 사례에서는 류마티스 관절염과 훨씬 뒤 시점의 Enthesopathies/synovial disorders 사이 강한 연관이 관찰되었다고 설명한다. 이는 최근 사건만 보는 것이 아니라 먼 과거-미래 관계를 포착한다는 질적 근거다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

가장 큰 강점은 **설계의 정합성**이다. 논문은 EHR를 단순 벡터가 아니라 구조적 시퀀스로 보고, Transformer의 장점을 의료 데이터 문제에 자연스럽게 맞춰 넣었다. 특히 age/segment embedding은 단순 “BERT를 의료에 적용했다” 수준을 넘어선 실질적 도메인 적응이라고 볼 수 있다.

두 번째 강점은 **스케일**이다. 160만 명 규모, 301개 질환이라는 설정은 단일 질환 예측보다 훨씬 현실적인 멀티태스크 환경이다. 이 정도 규모에서 8% 이상의 absolute APS improvement는 의미가 크다.

세 번째 강점은 **확장성**이다. 논문은 현재는 diagnosis와 age 중심이지만, medication, measurements, interventions 등을 embedding 추가만으로 넣을 수 있다고 주장한다. 즉 구조적으로 multimodal EHR foundation model 방향의 초기 형태로 읽을 수 있다.

### Limitations

첫째, 논문 본문 기준으로 실제 사용 입력은 **풍부한 EHR 전체가 아니라 주로 diagnosis와 age subset** 이다. 저자들도 full richness를 다 쓴 것은 아니라고 인정한다. 따라서 결과는 “Transformer가 EHR 전체를 이해했다”기보다 “진단 중심 표현만으로도 큰 성능 향상이 가능하다”에 더 가깝다.  

둘째, attention 기반 해석 가능성은 흥미롭지만, 이것이 곧바로 **causal clinical interpretability** 를 보장하지는 않는다. 논문은 disease trajectory mapping 가능성을 제시하지만, attention이 의학적 원인성을 의미한다고 주장하지는 않는다. 이 부분은 독자가 과대해석하지 않아야 한다. 이 평가는 논문 서술을 바탕으로 한 해석이다.

셋째, 논문은 강한 성능 개선을 보여주지만, 본문에서 확인 가능한 범위 내에서는 calibration, subgroup fairness, external validation 같은 현대 의료 AI 평가 항목은 중심적으로 다뤄지지 않는다. 따라서 실제 배치(clinical deployment)까지는 추가 검증이 필요하다. 이 점은 논문에 명시적 상세가 부족하므로 추정이 아니라 **보고 범위의 한계**로 보는 것이 적절하다.

### Interpretation

이 논문은 오늘날 관점에서 보면 “EHR용 BERT/Transformer” 계열의 초기 대표작으로 읽힌다. 특히 sequence modeling을 RNN에서 self-attention으로 옮기고, 도메인 특화 temporal/context embedding을 결합했다는 점에서 이후의 의료 foundation model 흐름을 예고하는 성격이 있다. 동시에 아직은 diagnosis-centric formulation에 머물러 있어, 후속 연구의 확장 여지가 크다.  

## 6. Conclusion

BEHRT는 EHR를 문서형 시퀀스로 재구성하고, BERT식 bidirectional pretraining과 self-attention을 통해 미래 질환을 예측하는 모델이다. 핵심 기여는 다음 세 가지로 요약할 수 있다.

* **Transformer/BERT를 EHR 문제에 성공적으로 이식**
* **age, position, segment, disease embedding** 을 결합한 의료 특화 입력 설계
* 대규모 실험에서 **기존 deep EHR baselines 대비 유의미한 성능 향상** 제시  

실무적으로는 조기 위험 예측, 질병 trajectory 분석, 범용 EHR representation 학습의 출발점으로 의미가 있다. 연구적으로는 이후 등장한 다양한 의료 Transformer, foundation model, multimodal EHR representation 연구의 중요한 선행 사례로 볼 수 있다. 다만 실제 임상 적용 관점에서는 더 다양한 입력 모달리티, 외부 검증, 해석 가능성 검증이 추가되어야 한다.
