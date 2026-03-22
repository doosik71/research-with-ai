# AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML

## 1. Paper Overview

이 논문은 자연어로 주어진 사용자 요구사항만으로 데이터 검색, 전처리, 모델 탐색, 하이퍼파라미터 최적화, 코드 생성, 배포까지 이어지는 **전체 AutoML 파이프라인**을 자동화하는 LLM 기반 멀티에이전트 프레임워크인 **AutoML-Agent**를 제안한 논문이다. 저자들은 기존 AutoML이 전문적인 설정과 코딩 역량을 요구하고, 최근의 LLM 기반 접근도 파이프라인의 일부 단계만 다루는 경우가 많다고 본다. 이에 따라 이 논문은 **full-pipeline**, **task-agnostic**, **deployment-ready**라는 세 가지 목표를 동시에 만족하는 시스템을 설계하고자 한다.

이 논문이 다루는 핵심 문제는 두 가지이다. 첫째는 **전체 파이프라인 계획(planning)의 복잡성**이다. 데이터 유형, 전처리 방식, 모델 구조, 최적화할 하이퍼파라미터가 서로 강하게 얽혀 있기 때문에, 부분 문제를 따로 잘 푼다고 해서 전체적으로 좋은 해법이 되지 않는다. 둘째는 **정확한 구현의 어려움**이다. LLM이 복잡한 ML 파이프라인 코드를 자율 생성할 때 누락된 의존성, 불완전한 코드, 잘못된 구현이 발생할 수 있으며, 사용자 요구가 모호할수록 문제가 커진다. 논문은 이 두 문제를 해결하기 위해 검색 기반 계획 수립과 다단계 검증을 결합한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 단일 LLM에게 “전체 파이프라인을 한 번에 만들어라”라고 요구하는 대신, **역할이 분리된 여러 에이전트가 협력하도록 구조를 설계**하는 데 있다. 이를 통해 문제를 더 작은 하위 문제로 나누고, 각 에이전트가 자신에게 맞는 역할만 수행하게 한다. 이 구조의 중심에는 Agent Manager가 있으며, Prompt Agent, Data Agent, Model Agent, Operation Agent를 조정한다. 각 에이전트는 서로 다른 전문성을 갖고 있고, Manager가 전체 계획과 검증을 담당한다.

저자들이 기존 연구 대비 새롭다고 주장하는 지점은 크게 세 가지이다. 첫째, **Retrieval-Augmented Planning(RAP)**을 통해 외부 지식을 검색하고, 그 지식을 바탕으로 여러 후보 계획을 생성해 탐색 폭을 넓힌다는 점이다. 둘째, 각 계획을 에이전트 역할에 맞게 분해한 뒤 **training-free prompting 기반으로 병렬 실행**하여 탐색 효율을 높인다는 점이다. 셋째, **request verification, execution verification, implementation verification**의 세 단계 검증을 통해 코드 생성 이전과 이후 모두에서 오류를 줄인다는 점이다. 논문은 표 1에서 자사 방법이 planning, verification, full pipeline, task-agnostic, training-free search, retrieval을 모두 갖춘 유일한 조합이라고 정리한다.

## 3. Detailed Method Explanation

### 3.1 전체 시스템 구조

AutoML-Agent의 전체 흐름은 크게 **초기화, 계획, 실행**의 세 단계로 나뉜다. 먼저 Agent Manager가 사용자 입력을 받고, 그것이 충분히 명확한지 확인한다. 그다음 Prompt Agent가 사용자 요구를 표준화된 JSON 형태로 파싱한다. 이후 Agent Manager가 RAP를 사용해 여러 개의 end-to-end 계획을 생성한다. 실행 단계에서는 Data Agent와 Model Agent가 각각 데이터 관련 계획과 모델 관련 계획을 수행하고, 그 결과를 Manager가 검증한다. 마지막으로 가장 적절한 해를 Operation Agent가 실제 코드로 구현하고, 구현 결과가 다시 검증되면 배포 가능한 모델이 사용자에게 반환된다.

이 구조의 중요한 점은 **계획과 구현을 분리**했다는 점이다. 기존 일부 에이전트 시스템은 계획을 세운 뒤 바로 코드를 생성하거나, 반대로 탐색 과정에서 실제 학습을 반복해 비용이 커진다. 이 논문은 먼저 계획 수준에서 충분히 탐색하고 검증한 뒤에, 그 결과를 바탕으로 실제 코드를 작성하게 한다. 즉, “좋은 계획을 고른 뒤 구현한다”는 점이 구조의 핵심이다.

### 3.2 에이전트 구성

**Agent Manager(Amgr)**는 사용자와 다른 에이전트 사이의 핵심 인터페이스이다. 이 에이전트는 계획 생성, 작업 분배, 결과 검증, 피드백 반영, 전체 상태 추적을 담당한다. 실제로 시스템의 중심 제어기 역할을 한다고 이해하면 된다.

**Prompt Agent(Ap)**는 사용자 요구를 구조화된 JSON으로 파싱한다. 논문은 이 작업을 위해 Ap를 instruction-tuning했다고 설명한다. 파싱 결과는 user, problem, dataset, model, knowledge, service 같은 상위 키를 포함한다. 예를 들어 problem에는 downstream task, application domain, 정확도나 latency 같은 제약이 들어가고, service에는 target device나 inference engine 관련 정보가 들어간다. 이것은 이후 단계의 에이전트들이 동일한 표현을 공유하도록 하는 공통 인터페이스 역할을 한다.

**Data Agent(Ad)**는 데이터 검색, 전처리, 증강, 데이터 특성 분석을 담당한다. 사용자가 데이터셋을 직접 제공하지 않으면 HuggingFace나 Kaggle 같은 저장소에서 API 호출을 통해 후보 데이터셋을 찾고, 찾은 데이터셋의 메타데이터를 프롬프트에 추가한 뒤 분석을 진행한다. 논문은 이를 실제 코드를 수행하는 것이 아니라 “pseudo data analysis” 형태의 prompting 기반 실행으로 설명한다. 즉, Data Agent는 계획 단계에서 데이터 파이프라인의 적절성을 추론하고 요약하는 역할을 맡는다.

**Model Agent(Am)**는 모델 검색, HPO, profiling, candidate ranking을 담당한다. 이 에이전트는 Data Agent의 결과를 참고하여 어떤 모델 계열이 적절한지, 어떤 하이퍼파라미터를 우선적으로 조정해야 하는지, 성능과 비용은 어떠한지를 추정한다. 또한 top-k 유망 모델을 반환하게 하여 이후 검증 단계가 그중 가장 유망한 해를 고르도록 설계되어 있다.

**Operation Agent(Ao)**는 선택된 계획을 바탕으로 실제 실행 가능한 코드를 생성한다. 이때 논문은 full-pipeline skeleton code를 제공하고, Ao는 그것을 기반으로 구체적 구현과 runtime debugging까지 수행한다. 최종적으로 Ao가 생성한 코드가 실제로 배포 가능한 수준인지 implementation verification을 거쳐 확인된다.

### 3.3 Prompt Parsing

논문은 Prompt Agent가 임의의 텍스트를 바로 안정적으로 구조화하지 못할 수 있다고 보고, 이를 위해 별도의 instruction tuning을 수행한다. 생성된 응답은 단순 자유 텍스트가 아니라 JSON 객체이며, 각 항목은 AutoML 파이프라인의 서로 다른 측면을 표현한다. 이 설계는 중요하다. 사용자 요구에는 종종 정확도, latency, 데이터 특성, 원하는 모델군, 배포 환경 등이 한꺼번에 섞여 있기 때문이다. 이들을 구조화해야 이후 planning과 verification이 가능해진다.

이 모듈의 의의는 단순한 파서가 아니라 **멀티에이전트 협업을 위한 공통 표상(common representation)** 을 만든다는 점이다. 즉, 시스템 전체가 같은 요구사항 해석을 공유하므로, 각 에이전트가 다른 가정을 하며 엇갈리는 문제를 줄일 수 있다. 이는 뒤에서 설명할 verification 단계의 기준점 역할도 한다.

### 3.4 Retrieval-Augmented Planning (RAP)

RAP는 이 논문의 가장 핵심적인 설계이다. 논문은 계획 집합을 $P={p_1,\dots,p_P}$로 두고, 사용자 요구 $R$과 LLM이 내재적으로 가진 지식, 그리고 외부 API로 검색한 최신 지식을 결합하여 여러 개의 end-to-end 계획을 생성한다고 설명한다. 형식적으로는 $P = Amgr(RAP(R))$로 적고 있다. 여기서 중요한 것은 **단일 계획이 아니라 여러 계획을 독립적으로 생성**한다는 점이다.

이렇게 하는 이유는 full-pipeline AutoML에서는 정답이 하나로 수렴하지 않기 때문이다. 같은 문제라도 데이터 처리 전략, 모델 계열, HPO 범위, 배포 전략에 따라 서로 다른 좋은 해가 존재할 수 있다. RAP는 외부 검색으로 최신 논문, 웹 지식, 사례 요약을 가져와 계획 후보를 확장하고, 이를 통해 더 나은 파이프라인을 탐색할 수 있게 한다. 동시에 계획들을 독립적으로 만들기 때문에 이후 병렬화도 가능하다. 논문은 이러한 retrieval 기반 계획 수립이 제약 조건이 있는 환경에서 특히 효과적이라고 주장한다.

### 3.5 Prompting-Based Plan Execution

계획이 생성되면 각 계획 $p_i$는 그대로 실행되지 않고, 먼저 에이전트 역할에 맞는 하위 작업으로 분해된다. 논문은 이를 **Plan Decomposition(PD)**라고 부른다. Data Agent에 대해서는 $s_i^d = PD(R, Ad, p_i)$처럼 정의하고, 이후 $O_i^d = Ad(s_i^d)$를 얻는다. Model Agent는 데이터 분석 결과까지 참고하므로 $s_i^m = PD(R, Am, p_i, O_i^d)$처럼 정의된다. 이후 $O_i^m = Am(s_i^m)$를 얻는다.

여기서 중요한 점은 이 단계가 실제 대규모 학습을 돌리는 탐색이 아니라 **prompting 기반의 training-free 실행**이라는 점이다. 저자들은 실제 학습 기반 탐색이 너무 느리고 비싸다고 본다. 따라서 LLM의 in-context reasoning과 외부 지식 검색을 사용해 유망한 데이터 처리 방식과 모델 후보를 빠르게 좁히고, 그 결과를 요약해 verification과 code generation에 넘긴다. 이 방식은 엄밀한 NAS나 MCTS 기반 탐색보다 계산량은 적지만, 계획의 다양성과 최신성은 확보하려는 절충으로 볼 수 있다.

### 3.6 Multi-Stage Verification

논문은 세 가지 검증 단계를 둔다. 첫째는 **Request Verification(ReqVer)**이다. 사용자 요청이 AutoML 문제로서 충분히 명확한지 확인하고, 부족하면 추가 정보를 요청한다. 이는 특히 비전문가 사용자가 불완전한 프롬프트를 넣을 가능성이 높다는 점을 고려한 설계이다.

둘째는 **Execution Verification(ExecVer)**이다. Data Agent와 Model Agent가 산출한 결과 집합 $O={(O_i^d, O_i^m)}_{i=1}^P$ 중에서 사용자 요구를 만족하는 후보가 있는지 확인한다. 통과한 해만 실제 구현 단계로 보낸다. 이 과정은 탐색 비용을 줄이는 역할도 한다. 왜냐하면 유망하지 않은 후보까지 모두 코드로 구현하지 않기 때문이다.

셋째는 **Implementation Verification(ImpVer)**이다. Operation Agent가 생성한 코드를 실제 실행 및 컴파일 결과까지 포함해 검증한다. 만약 요구조건을 만족하지 못하면 실패 사례를 기록하고, Manager가 그 정보를 반영해 계획을 다시 수정한다. 즉, 이 시스템은 단순한 일회성 생성이 아니라 **실패 피드백을 이용하는 수정 루프**를 가진다. 이는 LLM 코드 생성의 취약점을 보완하는 장치이다.

## 4. Experiments and Findings

### 4.1 실험 설정

논문은 이미지, 텍스트, 표형(tabular), 시계열, 그래프를 포함하는 **5개 데이터 모달리티와 7개 downstream task, 총 14개 데이터셋**에서 실험한다. 예를 들어 image classification에는 Butterfly Image와 Shopee-IET, text classification에는 Ecommerce Text와 Textual Entailment, graph에서는 Cora와 Citeseer가 포함된다. 평가 지표는 task에 따라 Accuracy, F1, RMSLE, RI 등을 사용한다.

평가 지표는 단순 task 성능만이 아니다. 논문은 **SR(success rate)**, **NPS(normalized performance score)**, **CS(comprehensive score)**를 사용한다. 여기서
$$
CS = 0.5 \times SR + 0.5 \times NPS
$$
이다. 즉, 좋은 모델을 찾는 것뿐 아니라 **실제로 코드가 실행되고 배포까지 가능한지**를 함께 평가한다. 이는 이 논문의 문제 설정이 “모델 추천”이 아니라 “deployable full pipeline 생성”이기 때문에 적절한 설계이다.

비교 대상은 Human Models, AutoGluon, GPT-3.5, GPT-4, DS-Agent이다. 구현 측면에서는 Prompt Agent에 Mixtral-8x7B를 사용하고, 나머지 에이전트와 LLM baseline에는 GPT-4o 계열을 사용한다. 또한 RAP의 기본 계획 수는 $P=3$, candidate model 수는 $k=3$으로 설정한다.

### 4.2 주요 결과

논문은 **constraint-aware setting**에서 AutoML-Agent가 특히 강하다고 보고한다. 성공률 측면에서 평균 SR이 87.1%라고 설명하며, 이는 제약조건이 있는 실제 환경에서 계획 기반 검색과 검증이 효과적이라는 근거로 제시된다. 저자들은 retrieval을 통해 어떤 제약에 집중해야 하는지 파악할 수 있었기 때문이라고 해석한다.

Downstream performance 측면에서도 AutoML-Agent는 NPS에서 다른 agent들뿐 아니라 Human Models까지 능가한다고 주장한다. 특히 constraint-aware 환경 전반에서 가장 좋은 결과를 보였다고 설명한다. 이는 RAP가 단지 검색 보조가 아니라, 제약 조건에 맞는 파이프라인을 찾는 데 실제로 도움이 된다는 논지와 연결된다.

종합 점수인 CS에서도 AutoML-Agent는 전반적으로 다른 baseline보다 우수하다. 흥미로운 관찰로, 일반-purpose LLM은 tabular classification/regression 같은 고전적 과제에서는 어느 정도 잘 작동하지만, 더 복잡한 과제에서는 DS-Agent나 AutoML-Agent 같은 구조적 프레임워크가 훨씬 낫다고 논문은 해석한다. 이는 복잡한 과제일수록 planning, decomposition, verification의 가치가 커진다는 것을 시사한다.

### 4.3 추가 분석

Ablation 결과는 이 논문의 설계 의도를 잘 보여준다. **RAP만 사용**하면 성능이 떨어지고, 경우에 따라 runnable model도 만들지 못한다. 여기에 **plan decomposition**을 추가하면 downstream 성능은 좋아지지만, 여전히 일부 과제에서는 code verification이 부족해 실패한다. 반면 **multi-stage verification까지 포함한 full framework**는 planning과 coding을 모두 개선하여 가장 좋은 결과를 만든다. 즉, 이 논문에서 제안한 각 부품은 독립적으로도 의미가 있지만, 함께 결합될 때 효과가 극대화된다.

계획 수 $P$에 대한 hyperparameter study에서는 plan 수가 success rate에는 큰 영향을 주지 않지만, NPS와 CS에는 영향을 준다고 보고한다. 그러나 plan 수를 무조건 늘린다고 좋아지지는 않는다. 비슷한 계획이 여러 개 생성되어 다양성이 줄어들 수 있기 때문이다. 그래서 저자들은 기본값을 $P=3$으로 선택한다. 이 판단은 “무조건 많이 탐색”보다 “적당한 다양성과 비용의 균형”을 중시한 것이다.

노이즈 강건성 실험에서는 planning 전에 또는 planning 직전에 가짜/무관한 정보를 주입해도 성능 저하가 크지 않았다고 보고한다. 저자들은 error correction과 multi-stage verification이 노이즈 영향을 완화한다고 해석한다. Prompt sensitivity 실험에서는 system prompt의 세부 문구보다 **역할이 명확하게 정의되었는지**가 더 중요하다고 분석한다. 이는 멀티에이전트 시스템 설계에서 persona 자체보다 역할 구분과 인터페이스 정의가 더 중요하다는 흥미로운 시사점이다.

SELA와의 비교에서는 AutoML-Agent가 약 **8배 빠른 탐색 시간**을 보였고, 평균 점수도 0.612 대 0.599로 비슷하거나 더 낫다고 보고한다. 또한 단일 모델 탐색 비용이 평균 약 **525초**, **0.30달러** 수준이라고 분석한다. 이 결과는 이 논문이 training-based search보다 prompting-based search를 택한 이유를 정당화한다. 성능 차이는 크지 않은데 비용 차이는 상당하기 때문이다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **문제 정의가 명확하고, 시스템 설계가 그 문제 정의와 잘 맞물린다**는 점이다. 이 논문은 단순히 “LLM으로 AutoML을 해보자”가 아니라, full pipeline, task-agnostic, deployment-ready라는 명확한 목표를 세운다. 그리고 이를 위해 parsing, planning, decomposition, verification을 계층적으로 설계한다. 즉, 제안된 각 모듈이 왜 필요한지가 비교적 분명하다.

둘째 강점은 **실용성에 대한 집착**이다. 많은 에이전트 논문이 reasoning 품질이나 툴 사용 능력 자체를 강조하는 반면, 이 논문은 실제 코드 성공률과 배포 가능성까지 평가한다. 또한 비용과 시간을 별도로 분석하여, 성능만이 아니라 현실적인 사용 가능성을 함께 논의한다. 이는 AutoML이라는 응용 영역에 잘 맞는 평가 방식이다.

셋째 강점은 **retrieval과 verification을 결합한 구조**이다. 최신 외부 지식을 가져오는 것만으로는 충분하지 않고, 그 결과를 여러 단계에서 걸러내야 실제 구현 품질이 올라간다는 점을 실험적으로 설득한다. 특히 ablation 결과는 decomposition과 verification이 왜 필요한지를 잘 보여준다.

반면 한계도 분명하다. 첫째, 이 시스템은 여전히 **강한 backbone LLM**에 크게 의존한다. 부록에서 저자들은 소형 모델이 복잡한 planning과 code generation에서 충분히 작동하지 않았다고 설명한다. 이는 시스템의 일반성보다는 강력한 기반 모델의 능력에 일정 부분 기대고 있음을 의미한다.

둘째, 새로운 유형의 ML task에 대해서는 **skeleton code 부재 시 hallucination 위험**이 커질 수 있다고 저자들도 인정한다. 즉, task-agnostic을 표방하지만 실제로는 실험에서 다룬 범주의 supervised/unsupervised pipeline에 상대적으로 잘 맞춰져 있다. 강화학습이나 추천시스템처럼 개발 절차가 크게 다른 영역은 추가 에이전트와 도메인별 모듈이 필요하다. 이 점에서 “완전한 범용 AutoML agent”라기보다는 “상당히 넓은 범위를 커버하는 구조화된 AutoML agent”로 보는 편이 더 정확하다.

셋째, training-free 검색은 효율성 측면에서 유리하지만, 반대로 **실제 학습 기반 탐색의 엄밀성을 일부 포기한 선택**이기도 하다. 논문은 효율성과 실용성을 우선한 것으로 보이며, 이는 합리적이다. 다만 매우 민감한 설계 공간이나 미세한 하이퍼파라미터 차이가 중요한 영역에서는 실제 학습 기반 탐색이 여전히 더 나을 가능성이 있다. 이 논문은 그 trade-off를 명시적으로 택한 연구라고 해석할 수 있다.

종합적으로 보면, 이 논문은 “LLM이 AutoML을 대신할 수 있는가”를 묻기보다, “LLM을 어떻게 구조화하면 실용적인 AutoML assistant가 될 수 있는가”를 다룬다. 그 답으로 멀티에이전트 역할 분화, retrieval-augmented planning, 다단계 verification을 제시한다는 점에서 설계적 기여가 크다.

## 6. Conclusion

이 논문은 full-pipeline AutoML이라는 까다로운 문제를 대상으로, LLM의 장점을 단순 생성이 아니라 **구조화된 협업**으로 끌어낸 연구이다. Prompt parsing으로 요구를 구조화하고, RAP로 여러 계획을 만들고, 역할별 plan decomposition으로 실행 효율을 높이며, multi-stage verification으로 구현 신뢰성을 높인다는 설계가 논문의 핵심 기여이다. 실험에서도 다양한 모달리티와 과제에 대해 높은 성공률과 좋은 종합 성능을 보였고, 비용 및 시간 분석까지 제시하여 실용적 타당성을 함께 논의한다.

향후 이 연구가 중요한 이유는 두 가지이다. 하나는 비전문가도 자연어만으로 비교적 복잡한 ML 파이프라인을 만들 수 있는 방향을 보여준다는 점이다. 다른 하나는 LLM 기반 시스템이 단일 거대 모델의 일회성 출력보다, **역할 분화와 검증 루프를 갖춘 시스템 설계**를 통해 훨씬 더 신뢰할 만한 결과를 낼 수 있음을 보여준다는 점이다. 따라서 이 논문은 AutoML 자체뿐 아니라, 향후 AI agent 기반 소프트웨어 엔지니어링과 실험 자동화 연구에도 중요한 참고점이 될 가능성이 크다.
