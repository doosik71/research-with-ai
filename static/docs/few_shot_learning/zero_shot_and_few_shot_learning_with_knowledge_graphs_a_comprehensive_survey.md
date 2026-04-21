# Zero-shot and Few-shot Learning with Knowledge Graphs: A Comprehensive Survey

- **저자**: Jiaoyan Chen, Yuxia Geng, Zhuo Chen, Jeff Z. Pan, Yuan He, Wen Zhang, Ian Horrocks, Huajun Chen
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2112.10006

## 1. 논문 개요

이 논문은 labeled sample이 부족한 상황에서 **Knowledge Graph (KG)** 를 활용해 **zero-shot learning (ZSL)** 과 **few-shot learning (FSL)** 을 어떻게 개선할 수 있는지를 폭넓게 정리한 survey이다. 저자들은 2021년 12월 기준으로 수집한 90편이 넘는 관련 논문을 검토하면서, KG가 단순한 부가 정보가 아니라 클래스 간 관계, 상식, 계층 구조, ontology, relation schema 등을 제공하는 중요한 지식 원천이라고 본다.

연구 문제는 명확하다. 대부분의 딥러닝 모델은 충분한 labeled data를 전제로 하지만, 실제 환경에서는 새 클래스가 계속 등장하거나 annotation 비용이 너무 높아서 그런 가정이 자주 깨진다. ZSL은 학습 단계에서 전혀 보지 못한 unseen class를 예측해야 하고, FSL은 unseen class당 아주 적은 labeled sample만 주어진다. 이때 seen class에서 unseen class로 무엇을 어떻게 transfer할지, 그리고 그 과정에서 auxiliary information을 어떻게 사용할지가 핵심 문제다.

이 논문이 중요한 이유는 두 가지다. 첫째, 기존 survey들이 ZSL이나 FSL 자체는 정리했지만, **KG 관점에서 방법론을 체계적으로 분류하고 비교한 작업은 부족했다**. 둘째, KG 활용은 computer vision, NLP, KG completion까지 넓게 퍼져 있는데, 이 논문은 방법론, KG construction, benchmark, open problem을 한 자리에서 연결해 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **KG-aware ZSL/FSL를 “KG를 어떻게 쓰는가”라는 관점으로 재구성하는 것**이다. 단순히 ZSL/FSL 일반론을 반복하는 것이 아니라, KG가 class hierarchy, commonsense, textual description, ontology schema, logical rule을 제공할 때 모델이 그 지식을 어떤 방식으로 흡수하는지를 기준으로 패러다임을 나눈다.

ZSL에서는 크게 네 가지 패러다임을 제안한다. **mapping-based**, **data augmentation**, **propagation-based**, **class feature**이다. FSL에서는 여기에 더해 few-shot sample 활용이 중요한 만큼 **optimization-based** 와 **transfer-based** 를 추가한다. 이 분류는 기존의 “classifier-based vs instance-based” 같은 구분보다 KG의 역할을 더 직접적으로 드러낸다.

또 하나의 차별점은, 저자들이 KG를 매우 넓게 정의한다는 점이다. RDF triple 기반 graph뿐 아니라 WordNet 같은 semantic network, taxonomy, weighted graph, Horn rule, Datalog rule, SWRL rule까지 KG의 실질적 범주 안에 둔다. 즉, 이 survey에서 KG는 단지 DB 형태의 지식 저장소가 아니라, **symbolic structure를 가진 외부 지식 전반**을 뜻한다.

## 3. 상세 방법 설명

논문은 먼저 ZSL과 FSL을 공통된 형식으로 정의한다. supervised classification은 training set $D_{tr}=\{(x,y)\mid x\in X, y\in Y\}$ 에서 함수 $f:x\to y$ 를 학습해 test sample의 label을 맞히는 문제다. ZSL에서는 training class 집합 $Y_s$ 와 test class 집합 $Y_u$ 가 서로 겹치지 않는다. 즉,

$$
D_{tr}=\{(x,y)\mid x\in X_s, y\in Y_s\}, \quad
D_{te}=\{(x,y)\mid x\in X_u, y\in Y_u\}, \quad
Y_s \cap Y_u = \varnothing
$$

이다. FSL에서는 unseen class $Y_u$ 에 대해 소수의 labeled sample로 이루어진 $D_{few}$ 가 추가된다.

논문이 반복해서 쓰는 핵심 함수는 세 가지다. 입력 인코더 $g:x\to \mathbf{x}$, 클래스 인코더 $h:y\to \mathbf{y}$, 그리고 필요시 점수 함수 $f':(x,y)\to s$ 이다. 여기서 $\mathbf{x}$ 와 $\mathbf{y}$ 는 각각 input feature와 class representation이다. KG는 주로 $h$ 를 더 잘 정의하거나, $g$ 와 $h$ 를 연결하는 구조를 제공하는 데 쓰인다.

ZSL의 **mapping-based** 패러다임은 입력과 클래스가 같은 공간에서 비교 가능하도록 만드는 방식이다.  
Input Mapping은 $M$ 만 학습해 $\mathbf{x}$ 를 class space로 보낸다.  
Class Mapping은 $M'$ 만 학습해 $\mathbf{y}$ 를 input space로 보낸다.  
Joint Mapping은 둘 다 학습해 중간 공간에서 비교한다.  
이때 예측은 보통 cosine similarity나 Euclidean distance처럼 거리 기반으로 수행된다. 논문은 이 방식이 image classification, entity typing, relation extraction, event extraction 등에 널리 쓰였다고 설명한다.

**Data Augmentation** 패러다임은 unseen class용 sample 또는 feature를 생성하는 접근이다. rule-based 방식은 논리 규칙으로 새 triple이나 sample을 유도하는 형태인데, survey에 따르면 실제로는 거의 KG completion 쪽에만 쓰인다. 더 중요한 것은 GAN 기반 생성 방식이다. 이때 generator는 unseen class embedding $\mathbf{y}_u$ 와 noise $z\sim \mathcal{N}(0,1)$ 를 입력받아 pseudo feature $\hat{\mathbf{x}}_u$ 를 만든다. discriminator는 real feature와 fake feature를 구분하며, 추가로 classification loss $L_{cls}$ 나 regularization loss $L_R$ 를 둘 수 있다. 저자들은 KG embedding이 class semantics를 풍부하게 담으면 생성 feature의 품질도 좋아진다고 정리한다.

**Propagation-based** 패러다임은 graph 위에서 정보를 퍼뜨리는 방식이다. 대표적으로 seen class classifier의 parameter $p(y_s)$ 를 graph node에 놓고 unseen class 쪽으로 전파해 $p(y_u)$ 를 추정한다. GCN, ResGCN, attention-equipped GCN, multi-relational GCN 등이 여기에 속한다. 이 접근은 WordNet이나 task-specific KG 위에서 image classifier parameter를 transfer하는 데 자주 쓰인다. 다른 하위 유형으로는 class belief propagation이 있는데, multi-label setting에서 seen class belief를 unseen class belief로 전달하는 방식이다.

**Class Feature** 패러다임은 입력과 클래스의 표현을 직접 결합해 점수를 계산한다. 즉, $f'(g(x), h(y))\to s$ 구조다. text feature fusion에서는 entity description, relation text, class name 같은 텍스트를 BERT, fastText, CNN, BiLSTM 등으로 encode한 뒤 triple classification이나 QA를 수행한다. multi-modal feature fusion에서는 image/question feature와 KG/class feature를 결합한다. 최근 pre-trained language model의 발전 때문에 이 계열이 특히 KG completion과 QA에서 강해지고 있다고 논문은 본다.

FSL에서는 위 방식들이 일부 그대로 확장되지만, few-shot sample 자체를 적극 활용하는 두 패러다임이 추가된다. **Optimization-based** 는 episode-based meta-learning이다. 각 task $t$ 에 대해 support set $D^t_{support}$ 와 query set $D^t_{query}$ 를 만들고, meta learner $F_\theta$ 가 support로부터 빠르게 적응하도록 $\theta$ 를 학습한다. MAML, Bayesian meta-learning, relation-meta learner가 대표적이다. **Transfer-based** 는 seen KG에서 학습한 GNN이나 logical rule을 unseen KG 또는 새 sub-KG로 직접 옮기는 방식이다. GraIL, DRUM 같은 방법이 여기에 포함된다.

## 4. 실험 및 결과

이 논문은 새로운 모델을 제안하는 논문이 아니라 survey이므로, 하나의 통일된 실험을 수행하지는 않는다. 대신 task별 benchmark와 대표 성과를 구조적으로 정리한다. 분야는 크게 computer vision, NLP, KG completion으로 나뉜다.

Computer vision에서는 zero-shot image classification이 가장 큰 비중을 차지한다. 대표 benchmark는 ImageNet, AwA2, CUB, NUS-WIDE, 그리고 KG가 보강된 ImNet-A/ImNet-O이다. WordNet hierarchy, ConceptNet commonsense, attribute, textual label, OWL logical relation 같은 지식이 auxiliary KG로 사용된다. 논문은 최근 state-of-the-art 성능이 단순 attribute나 text만 쓰는 방식보다 **KG를 명시적으로 활용하는 방법들**, 특히 propagation-based나 graph-enhanced generation 방식에서 자주 나온다고 정리한다. 다만 survey 특성상 전체 방법을 동일 조건에서 수치 비교한 표를 중심으로 제시하는 것이 아니라, task별 자원과 대표 경향을 정리하는 데 초점을 둔다.

NLP 쪽에서는 entity typing, relation extraction, event extraction, text classification, QA가 다뤄진다. 예를 들어 BBN, OntoNotes, Wikipedia는 fine-grained entity typing에, NYT10과 WEB19는 relation extraction에, ACE05는 event extraction에 사용된다. 여기서는 KG entity/relation/type embedding을 label space로 활용하는 mapping-based 접근과, pre-trained language model에 KG 정보를 얹는 class feature 방식이 중요하다. QA에서는 SocialIQa, CommonsenseQA, QASC, ARC, OpenBookQA 등이 등장하며, ConceptNet, ATOMIC, WorldTree 등의 graph 지식이 필요하다.

KG completion은 이 survey에서 특히 비중 있게 다뤄진다. unseen entity 문제에서는 FB15k-237-OWE, DBpedia50k/500k, Wikidata5M, FB20k 같은 benchmark가 소개된다. unseen relation 문제에서는 NELL-ZS, Wiki-ZS, NELL-One, Wiki-One이 핵심이다. 여기서는 entity description 기반 inductive embedding, relation description 기반 generation, few-shot relation meta-learning, neighbor aggregation 기반 embedding propagation 등이 주류다. 저자들은 KG completion이 “KG를 auxiliary 정보로 쓰는 문제”이기도 하지만, 동시에 “KG 자체가 zero-shot/few-shot 대상”이 되는 독특한 영역이라고 강조한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 정리의 폭과 구조다. 저자들은 단순히 논문 목록을 나열하지 않고, KG construction, KG type, task, benchmark, method paradigm을 서로 연결했다. 특히 ZSL과 FSL을 따로따로 보지 않고, 어느 방법이 ZSL에서 FSL로 자연스럽게 확장되는지, 어떤 방법은 few-shot sample 없이는 성립하지 않는지를 비교한 점이 유용하다.

또 다른 강점은 KG의 범위를 ontology, taxonomy, commonsense graph, logical rule까지 넓혀 본 점이다. 이 덕분에 WordNet 기반 class hierarchy부터 OWL schema, relation domain/range, Horn rule, KG textual description까지 하나의 프레임 안에서 설명할 수 있다. 실제 연구자 입장에서는 “내가 가진 외부 지식이 KG인가?”보다 “이 구조를 어떻게 학습에 연결할 수 있는가?”가 더 중요하므로, 이런 넓은 정의가 실용적이다.

한계도 분명하다. 첫째, survey이기 때문에 개별 방법의 실험 설정 차이를 완전히 통제하지 못한다. 논문도 benchmarking의 부족을 직접 지적한다. 둘째, 저자들이 제안한 분류는 유용하지만, 어떤 방법은 mapping-based이면서 동시에 meta-learning을 쓰고, 또 다른 방법은 class feature와 propagation 성격을 같이 가진다. 즉, 분류가 완전히 mutually exclusive하지는 않다. 셋째, “KG quality” 문제를 중요하게 언급하지만, 실제 어떤 지식이 얼마나 도움이 되는지에 대한 정량 분석은 기존 문헌 전반에서 부족하다고 인정한다.

비판적으로 보면, 이 논문은 매우 포괄적이지만 survey의 특성상 깊이가 방법별로 균일하지는 않다. 예를 들어 생성 모델 기반 방법은 가능성이 크다고 평가하면서도 실제 문헌 수가 적어 비교 분석이 제한적이다. 또한 benchmark 정리는 풍부하지만, 각 benchmark의 split 설계가 얼마나 공정한지, label leakage나 ontology alignment 비용이 어느 정도인지에 대한 비판적 검토는 상대적으로 더 확장될 여지가 있다.

## 6. 결론

이 논문은 KG-aware ZSL/FSL를 다루는 매우 포괄적인 survey로, **KG construction**, **방법론 패러다임**, **응용 분야**, **benchmark**, **open problem**을 하나의 큰 지도처럼 정리한다. 핵심 메시지는 분명하다. sample shortage 문제에서 KG는 단순한 보조 정보가 아니라, seen과 unseen 사이를 연결하는 구조적 지식의 매개체다.

실제로 이 연구 흐름은 앞으로도 중요할 가능성이 크다. 이유는 새 클래스, 새 relation, 새 domain이 계속 등장하는 환경에서 완전한 재학습은 비효율적이기 때문이다. 특히 ontology-aware reasoning, generative augmentation, meta-learning, inductive KG completion, pre-trained language model과 symbolic KG의 결합은 앞으로도 유망한 방향으로 보인다. 논문 자체도 마지막에 지적하듯, 앞으로의 핵심 과제는 **더 좋은 KG를 자동 또는 반자동으로 만들기**, **few-shot sample과 KG를 더 정교하게 결합하기**, 그리고 **공정하고 체계적인 benchmarking을 확립하기**이다.
