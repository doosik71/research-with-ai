# MiME: Multilevel Medical Embedding of Electronic Health Records for Predictive Healthcare

첨부하신 논문은 EHR 데이터가 가진 **계층적 구조(patient → visits → diagnosis objects → treatment codes)** 자체를 학습에 적극 활용해, 데이터가 많지 않은 의료기관 환경에서도 더 견고한 예측 성능을 내도록 설계한 방법인 **MiME**를 제안합니다. 핵심 문제의식은 기존 딥러닝 기반 EHR 예측 모델이 대규모 데이터에 강하게 의존하고, 그 한계를 보완하기 위해 외부 의료 ontology를 쓰더라도 실제 병원별 비표준 코드 체계 때문에 곧바로 적용하기 어렵다는 점입니다. 이에 저자들은 외부 지식 대신 EHR 내부에 이미 존재하는 **진단-처치 관계와 방문 구조**를 활용해 representation을 학습하고, heart failure prediction과 sequential disease prediction에서 성능 향상을 보였다고 주장합니다. 특히 heart failure 예측에서는 작은 데이터셋일수록 이점이 더 커졌고, 가장 작은 데이터셋에서 best baseline 대비 PR-AUC 상대 향상 15%를 보고합니다.  

## 1. Paper Overview

이 논문이 풀고자 하는 문제는 “왜 의료 예측 모델은 실제 현장에서는 잘 안 먹히는가?”에 가깝습니다. 연구 커뮤니티에서는 EHR 기반 risk prediction, diagnosis, subtyping 등에서 딥러닝이 강력한 결과를 보였지만, 실제 다수의 healthcare system은 대형 병원 수준의 데이터 규모를 확보하지 못합니다. 저자들은 이런 **data insufficiency**가 특히 희귀 질환이나 복잡한 의료 서비스에서 치명적이라고 봅니다. 또한 ontology 기반 보강은 이론적으로 좋지만, 약물/검사/처치 코드가 병원마다 다르게 관리되는 현실 때문에 일반화가 어렵다고 지적합니다.

이 문제의 중요성은 의료 데이터가 단순한 bag-of-codes가 아니라는 점에 있습니다. 한 방문에서 특정 diagnosis가 나오면 그에 상응하는 medication/procedure가 뒤따르는 구조가 있으며, 이런 관계는 환자 상태를 이해하는 데 매우 중요합니다. 그런데 기존 많은 방법은 이를 평평하게(flatten) 처리해 코드 집합으로만 본다는 것이 저자들의 비판입니다. MiME는 바로 이 지점을 공략합니다.  

## 2. Core Idea

MiME의 중심 아이디어는 매우 명확합니다.

첫째, EHR를 단순한 방문 시퀀스가 아니라 **다층 구조(multilevel structure)**로 본다는 점입니다. 환자 기록은 방문들의 시퀀스이고, 각 방문 안에는 여러 진단 객체가 있으며, 각 진단 객체는 다시 대응되는 치료 코드 집합을 가집니다. 즉, “진단과 처치가 어떤 국소 구조를 이루는가”를 representation의 기본 단위로 삼습니다.

둘째, 이 구조를 표현으로만 쓰는 것이 아니라, **보조 예측 과제(auxiliary prediction tasks)**와 함께 학습한다는 점입니다. 저자들은 진단 객체 표현 $\mathbf{o}^{(t)}_i$ 로부터 해당 diagnosis code 자체와 연결된 treatment code들을 예측하게 만듭니다. 이 auxiliary task는 외부 라벨이 필요 없고, EHR 안에 내재된 구조를 그대로 supervision으로 활용합니다. 즉, main task만으로 representation을 간접적으로 배우는 것이 아니라, “이 진단 객체 표현은 적어도 자기 diagnosis와 treatment를 설명할 수 있어야 한다”는 구조적 제약을 넣습니다.  

셋째, novelty는 “의료 ontology를 외부에서 주입하는 대신, EHR 내부의 diagnosis-treatment coupling을 구조적으로 모델링한다”는 데 있습니다. 기존 방법 중 Med2Vec은 co-occurrence 중심, GRAM은 external knowledge 중심인데, MiME는 **inherent EHR structure** 자체를 supervision과 representation 양쪽에 모두 사용합니다.  

## 3. Detailed Method Explanation

### 3.1 데이터 표현 단위

논문은 한 환자의 $t$번째 방문을 $\mathcal{V}^{(t)}$ 로 두고, 그 안에 여러 diagnosis object $\mathcal{O}^{(t)}_i$ 가 있다고 정의합니다. 각 diagnosis object는 하나의 diagnosis code $d_i^{(t)}$ 와, 그 진단에 연결된 treatment code 집합 $\mathcal{M}_i^{(t)}$ 를 가집니다. 치료 코드는 medication과 procedure를 포함합니다. 방문 표현은 $\mathbf{v}^{(t)}$, 진단 객체 표현은 $\mathbf{o}^{(t)}_i$ 로 표기됩니다. 또한 auxiliary prediction은 $p(d_i^{(t)}|\mathbf{o}^{(t)}_i)$ 및 $p(m*{i,j}^{(t)}|\mathbf{o}^{(t)}_i)$ 형태로 정의됩니다.

이 정의 자체가 중요합니다. MiME는 “방문 안의 코드들”을 곧바로 합치는 대신, **진단 객체라는 중간 표현 단위** 를 둠으로써 diagnosis-specific treatment relation을 보존합니다.

### 3.2 방문-진단-치료의 3단계 구조

논문은 MiME를 **visit level, diagnosis level, treatment level** 의 top-down 구조로 설명합니다. Eq. (1), Eq. (2), Eq. (3)이 각각 이 세 수준에 대응한다고 밝힙니다. 방문 임베딩 $\mathbf{v}^{(t)}$ 는 여러 진단 객체 임베딩의 합으로 만들어지고, 각 진단 객체는 diagnosis code와 그에 연결된 treatment interaction을 반영해 구성됩니다.  

핵심은 “방문 표현 = 진단 객체들의 집계”이고, “진단 객체 표현 = diagnosis 자체 + diagnosis-treatment interaction”이라는 분해입니다. 이 구조 덕분에 어떤 처치가 어떤 진단과 연결되었는지가 보존됩니다. flatten 방식에서는 방문 내 코드 공출현만 남고, 어느 처치가 어느 진단과 연결되었는지가 사라집니다.

### 3.3 Treatment interaction: Eq. (3)의 의미

논문에서 비교적 선명하게 드러나는 수식은 interaction 함수입니다.

$$
g(d_i, m_{i,j}) = \sigma(\mathbf{W}_m r(d_i)) \odot r(m*{i,j})
$$

여기서 $r(d_i)$ 와 $r(m_{i,j})$ 는 각각 diagnosis code와 treatment code의 embedding이고, $\sigma(\mathbf{W}_m r(d_i))$ 는 diagnosis에 의해 조절되는 gate처럼 작동하며, treatment embedding과 element-wise product $\odot$ 로 결합됩니다. 저자들은 이 formulation이 bilinear pooling 계열 아이디어에서 영감을 받았다고 설명합니다.

직관적으로 보면, 같은 treatment라도 어떤 diagnosis와 결합되느냐에 따라 의미가 달라집니다. 예를 들어 단순 co-occurrence만 쓰면 “acetaminophen이 fever와 같이 나왔다”와 “acetaminophen이 다른 diagnosis와 같이 나왔다”를 충분히 구분하지 못할 수 있는데, MiME는 diagnosis-conditioned gating을 통해 이 상호작용을 더 세밀하게 모델링합니다.

### 3.4 Skip-connections의 역할

논문은 Eq. (1), Eq. (2)에서 등장하는 $F$, $G$ 를 skip-connection으로 해석합니다. 이는 vanishing gradient를 줄이고 representation learning을 안정화하기 위한 장치입니다. 부록에서는 실제로 MiME의 여러 비선형 부분에 ReLU를 주로 사용하고, 특정 식에서는 sigmoid를 regularization 효과를 위해 사용했다고 설명합니다. 또한 단순히 sigmoid MLP에 skip connection을 더한다고 항상 좋아지는 것은 아니었다고 밝힙니다.  

즉, MiME의 성능은 단순한 구조적 아이디어만이 아니라, 이를 안정적으로 학습시키는 residual-style 설계에도 의존합니다.

### 3.5 Auxiliary task의 기능

MiME의 중요한 축은 auxiliary task입니다. 진단 객체 표현 $\mathbf{o}^{(t)}_i$ 로부터 해당 diagnosis code와 각 treatment code를 예측하게 함으로써, 표현이 main prediction task에만 편향되지 않도록 만듭니다. 저자들은 이를 통해 learned visit representation이 “target task에만 특화된 벡터”가 아니라, 보다 general-purpose한 구조적 지식을 담는다고 주장합니다.  

이 설계는 의료 데이터처럼 label scarcity가 심한 환경에서 특히 의미가 있습니다. 본 과제 라벨이 부족해도, 방문 내부의 진단-처치 관계는 거의 항상 존재하기 때문에 self-supervision에 가까운 형태로 representation 품질을 높일 수 있습니다.

### 3.6 전체 파이프라인

전체적으로는 다음 흐름으로 이해할 수 있습니다.

1. 각 visit를 diagnosis object들의 집합으로 분해한다.
2. 각 diagnosis object에서 diagnosis embedding과 diagnosis-conditioned treatment interaction을 합쳐 object embedding을 만든다.
3. object embedding들을 합쳐 visit embedding을 만든다.
4. patient 수준에서는 visit sequence를 이용해 downstream prediction을 수행한다.
5. 동시에 각 object embedding에 대해 diagnosis/treatment auxiliary prediction을 수행해 구조적 regularization을 건다.

heart failure prediction처럼 sequence 전체가 중요한 과제에서는 방문 시퀀스를 환자 표현으로 변환해야 하며, 논문은 patient representation $\mathbf{h}$ 를 정의하고 sequence model을 사용합니다. 부록에서는 순차 처리에 GRU cell size 128을 사용했다고 기술합니다.  

## 4. Experiments and Findings

### 4.1 데이터와 과제

실험 데이터는 Sutter Health의 EHR이며, heart failure 예측을 위해 구성되었습니다. 총 30,764명의 50~85세 senior patient가 포함되었고, diagnosis, medication, procedure 코드가 사용되었습니다. diagnosis는 ICD9 기반 388개 category, medication은 99개 group, procedure는 1,824개 category로 그룹화되었고, 맞지 않는 코드는 자체 category를 형성했습니다.

heart failure prediction 과제에서는 18개월 observation record를 바탕으로 이후 1년 안에 첫 HF 진단이 발생하는지를 예측합니다. 30,764명 중 3,414명이 case, 27,350명이 control입니다. 이 class imbalance 때문에 저자들은 ROC-AUC보다 PR-AUC를 특히 중요하게 봅니다.  

또 다른 과제는 sequential disease prediction입니다. 제공된 스니펫만으로 세부 metric 정의 전체를 완전하게 확인할 수는 없지만, 부록 결과에서는 rare/frequent diagnosis group에 대해 Precision@k 형태의 평가가 사용되었고, 희귀 질환군을 포함한 다양한 frequency setting에서 성능을 비교했습니다.

### 4.2 비교 대상

스니펫에서 확인되는 baseline에는 **Med2Vec** 와 **GRAM** 이 포함됩니다. Med2Vec은 EHR 기반 unsupervised visit representation의 대표 예시이고, GRAM은 ICD9 tree 같은 external domain knowledge를 주입하는 대표 예시입니다. 논문 전체에는 이 외에도 raw / 여러 MLP / sequence baselines가 더 포함되어 있는 것으로 보이지만, 현재 확인 가능한 스니펫에서는 Med2Vec과 GRAM이 가장 명시적입니다.

### 4.3 핵심 결과

가장 강한 메시지는 heart failure prediction에서의 성능 향상입니다. 논문은 MiME가 **모든 데이터 크기 설정에서 baseline을 일관되게 이겼고**, 특히 작은 데이터셋일수록 격차가 더 컸다고 보고합니다. 가장 작은 데이터셋에서는 best baseline 대비 PR-AUC 상대 향상 15%가 나타났습니다. 이는 저자들의 핵심 주장, 즉 “MiME가 작은 데이터 환경에서 특히 유리하다”를 직접 뒷받침합니다.  

또한 varying data size 실험에서 E1, E2처럼 더 작은 데이터셋에서 MiME와 baselines의 차이가 더 크게 벌어진다고 설명합니다. 반면 일부 baseline은 데이터 크기에 따라 성능이 들쭉날쭉했고, 예를 들어 tanh-MLP는 작은 데이터에서 경쟁력이 있지만 큰 데이터에서는 덜 강했고, relu-MLP는 반대 양상을 보였습니다. 저자들은 이를 activation function과 regularization의 상호작용으로 해석합니다.  

visit complexity를 달리한 실험에서도 MiME는 작은 데이터셋 $D_1, D_2, D_3$ 에서 diagnosis-treatment 관계를 포착해 robust한 성능을 낸다고 보고합니다. 이는 단순히 샘플 수가 작을 때만이 아니라, 방문 구조의 복잡성이 높을 때도 MiME의 구조적 modeling이 도움이 됨을 시사합니다.

Sequential disease prediction에서는 rarest diagnosis group을 제외하면 MiME가 baseline을 능가했고, 일부 설정에서는 best baseline 대비 최대 11.6% 상대 향상을 보였습니다. 다만 가장 희귀한 코드군에서는 Med2Vec이 더 좋았다고 보고합니다. 즉, MiME의 장점은 “완전히 데이터가 거의 없는 극단적 희귀코드”보다는, 어느 정도 학습 샘플이 존재하지만 구조적 modeling이 필요한 영역에서 더 잘 나타난다고 볼 수 있습니다.

### 4.4 저자들의 해석

저자들은 MiME가 PR-AUC에서 특히 강한 이유를, HF prediction이 **positive sample을 정확히 골라내는 능력** 을 더 요구하기 때문이라고 설명합니다. Med2Vec 같은 방법은 같은 방문에서 자주 같이 나오는 코드를 잘 묶어 ROC-AUC에서는 경쟁력이 있을 수 있지만, diagnosis-treatment의 미묘한 상호작용을 잃기 때문에 PR-AUC에서는 약하다고 해석합니다. 이 설명은 MiME의 구조적 설계가 단지 “표현이 더 복잡하다”가 아니라, 실제로 clinical discriminative signal과 더 맞닿아 있음을 보여줍니다.  

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 문제 설정과 방법론이 잘 맞물린다는 점입니다. 데이터 부족 문제를 해결하고 싶다면 일반적으로 더 많은 외부 지식을 넣거나 사전학습을 강화하는 방향을 생각하기 쉽습니다. 그러나 MiME는 외부 ontology의 한계를 짚고, 오히려 EHR 내부에 이미 존재하는 구조를 이용합니다. 이 선택은 의료 현장의 비표준 코드 체계라는 현실과 잘 맞습니다. 또한 auxiliary task를 통해 구조적 prior를 representation에 자연스럽게 주입한 점도 설득력이 있습니다.  

두 번째 강점은 성능 향상이 특히 **작은 데이터셋** 에서 두드러진다는 점입니다. 이는 논문의 주장과 정확히 일치하는 empirical evidence입니다. 단지 평균 성능이 좋은 모델이 아니라, 왜 이 모델이 필요한지와 어디서 가장 유용한지가 비교적 분명하게 드러납니다.  

하지만 한계도 분명합니다. 우선 MiME는 diagnosis-treatment linkage가 비교적 명확히 정의되는 EHR 구조를 전제로 합니다. 실제 병원 데이터에서는 ordering/association이 항상 깔끔하지 않을 수 있고, 어떤 처치가 어느 진단에 대응되는지 애매한 경우도 많습니다. 논문은 이런 linkage가 어떻게 생성되었는지 완전한 일반론을 제시하기보다는, 실험 데이터셋의 구조를 활용합니다. 따라서 다른 시스템으로 이전할 때 preprocessing 파이프라인의 질이 성능에 큰 영향을 줄 가능성이 있습니다. 이 부분은 논문에서 강하게 일반화하지는 않지만, 실용 관점에서는 중요한 제약입니다.

또한 sequential disease prediction에서 가장 희귀한 코드군에서는 Med2Vec이 더 나았다는 결과는, MiME가 모든 저자원 상황에서 절대적으로 우월하다는 뜻은 아니라는 점을 보여줍니다. 극단적으로 sparse한 regime에서는 pre-trained co-occurrence embedding 계열이 더 유리할 수 있습니다.

비판적으로 보면, MiME는 구조적 inductive bias를 잘 설계한 모델이지만, 그 bias가 언제나 옳다고 보장되지는 않습니다. 예를 들어 어떤 처치가 diagnosis보다 provider habit나 administrative coding에 더 많이 좌우되는 경우, diagnosis-conditioned interaction이 noise를 끌어올 수도 있습니다. 논문은 전반적으로 긍정적 결과를 제시하지만, 실패 사례나 구조 정보가 오히려 해가 되는 조건에 대한 분석은 제한적입니다. 이 점은 후속 연구에서 더 검토할 가치가 있습니다.

## 6. Conclusion

MiME는 EHR를 단순한 코드 집합이 아니라 **진단-처치 관계를 가진 다층 구조** 로 보고, 그 구조를 representation learning과 auxiliary supervision에 동시에 반영한 방법입니다. 결과적으로 heart failure prediction과 sequential disease prediction에서 강한 성능을 보였고, 특히 데이터가 작을수록 더 큰 이점을 보였습니다.

실무적으로는, 대규모 사전학습이나 외부 ontology 정합이 어려운 병원 환경에서 “내부 EHR 구조를 최대한 활용하는 예측 모델”이라는 방향성을 제시했다는 점에서 의미가 큽니다. 연구적으로도, 의료 데이터에서의 hierarchical inductive bias와 self-/auxiliary supervision의 결합이 왜 중요한지를 설득력 있게 보여준 논문이라고 볼 수 있습니다.
