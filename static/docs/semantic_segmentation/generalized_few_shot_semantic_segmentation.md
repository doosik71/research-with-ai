# Generalized Few-shot Semantic Segmentation

- **저자**: Zhuotao Tian, Xin Lai, Li Jiang, Shu Liu, Michelle Shu, Hengshuang Zhao, Jiaya Jia
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2010.05210

## 1. 논문 개요

이 논문은 semantic segmentation을 few-shot 조건에서 더 실제적인 방식으로 다루기 위해, 기존의 Few-Shot Semantic Segmentation(FS-Seg)을 확장한 **Generalized Few-Shot Semantic Segmentation (GFS-Seg)** 설정을 제안한다. 핵심 문제의식은 기존 FS-Seg 평가 방식이 실제 사용 환경과 다르다는 점이다. 기존 FS-Seg는 테스트 시 query image에 들어 있는 novel class를 support set이 정확히 알려준다는 강한 전제를 둔다. 즉, 모델은 “이 이미지에서 찾아야 할 클래스가 무엇인지”를 support image와 mask를 통해 미리 전달받는다.

논문은 이 전제가 실제 응용에서 지나치게 강하다고 지적한다. 실제 semantic segmentation 시스템은 입력 이미지만 보고, 그 안에 어떤 class가 들어 있는지 미리 모르는 상태에서 base class와 novel class를 함께 예측해야 한다. 또한 기존 FS-Seg는 주로 novel class만 평가하지만, 실제 장면에는 이미 충분히 학습된 base class들도 계속 등장한다. 따라서 기존 FS-Seg 성능만으로는 “새 클래스 몇 장만 보고 기존 클래스와 새 클래스를 동시에 잘 분할할 수 있는가”를 판단하기 어렵다.

이 문제를 해결하기 위해 저자들은 GFS-Seg라는 새로운 benchmark를 정의한다. 이 설정에서는 먼저 base classes로 일반 segmentation 모델을 학습하고, 이후 novel classes에 대해 소수의 support sample만 받아 novel classifier를 등록한 뒤, 최종 평가에서는 support sample 없이 query image만 입력으로 받아 **base와 novel 전체 클래스 집합**에 대해 segmentation을 수행해야 한다. 논문은 이 설정이 실용적인 few-shot segmentation에 더 가깝다고 주장하며, 기존 대표 FS-Seg 방법들이 이 환경에서 크게 성능이 떨어진다는 점을 실험으로 보인다.

이와 함께 저자들은 GFS-Seg를 풀기 위한 baseline과, 이를 개선하는 **Context-Aware Prototype Learning (CAPL)** 방법을 제안한다. CAPL의 목표는 단순 prototype 결합만으로는 놓치기 쉬운 문맥 정보(context)를 support와 query 양쪽에서 활용하여 base prototype과 novel prototype을 더 잘 조정하는 것이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 두 가지다.

첫째, **문제 설정 자체를 바꿔야 한다**는 점이다. 기존 FS-Seg는 query-support pair 기반 episodic setting에 강하게 묶여 있어서, 모델이 “주어진 support class를 foreground로 찾는 능력”에 집중하게 된다. 하지만 실제로는 query image 안에 어떤 class가 있는지 모르는 상태에서 여러 base/novel class를 동시에 구분해야 한다. 논문은 이 차이가 단순한 평가 조건 차이가 아니라, 모델이 학습하는 decision boundary 자체를 바꾸는 중요한 차이라고 본다. 그래서 GFS-Seg는 novel class generalization 능력뿐 아니라, base class를 유지하면서 novel class를 함께 인식하는 능력을 본다.

둘째, **semantic segmentation에서는 context가 특히 중요하다**는 점을 few-shot setting에 본격적으로 반영한다. 저자들은 prototype learning 자체는 few-shot classification이나 FS-Seg에서 이미 잘 알려져 있지만, GFS-Seg에서는 그것만으로 부족하다고 본다. 예를 들어 base class인 `Dog`는 학습 중 `People`과의 문맥은 익혔을 수 있지만, novel class인 `Sofa`가 support에서 `Dog`와 함께 자주 나타난다면 이런 새 co-occurrence 정보가 base prototype에 반영될 필요가 있다. 또 support set만으로 얻은 문맥은 제한적이므로, 실제 query image의 내용에 맞춰 classifier를 동적으로 조정할 필요도 있다.

이 관찰에서 CAPL이 나온다. CAPL은 두 단계로 context를 활용한다.

- **Support Contextual Enrichment (SCE)**: support sample에 함께 등장한 base class 정보를 이용해 base classifier를 보정한다.
- **Dynamic Query Contextual Enrichment (DQCE)**: 각 query image에서 임시 prediction을 만들고, 그 결과로부터 query-specific semantic hint를 추출해 classifier를 동적으로 다시 보정한다.

즉, CAPL은 “support에서 얻은 사전 문맥”과 “query에서 얻은 현재 문맥”을 모두 classifier에 주입하는 방식이다. 구조를 크게 바꾸지 않고 normal semantic segmentation model의 classifier/prototype 구성만 바꾸므로, FCN, PSPNet, DeepLab 같은 기존 segmentation backbone/decoder에 비교적 일반적으로 붙일 수 있다는 점도 차별점이다.

## 3. 상세 방법 설명

### 3.1 GFS-Seg 문제 설정

논문에서 GFS-Seg는 세 단계로 정의된다.

첫 번째는 **base class learning phase**이다. 여기서는 일반 semantic segmentation처럼 base classes $C_b$에 대해 충분한 labeled data로 모델을 학습한다.

두 번째는 **novel class registration phase**이다. 이 단계에서는 novel classes $C_n$ 각각에 대해 매우 적은 수의 support sample, 즉 $K$-shot 예제를 받아 novel class prototype을 만든다. 이 단계는 테스트 이미지마다 반복되는 것이 아니라, novel classes를 한 번 등록하는 과정이다.

세 번째는 **evaluation phase**이다. 여기서는 query image만 입력으로 받아, 그 안에 어떤 class가 들어 있는지 모르는 상태에서 base와 novel 전체 집합 $C_b \cup C_n$에 대해 픽셀 단위 예측을 해야 한다. 따라서 주요 평가지표는 novel class만의 mIoU가 아니라, 전체 클래스 평균인 **total mIoU**이다.

이 설정은 기존 FS-Seg와 본질적으로 다르다. FS-Seg에서는 support가 query에 있는 target class를 명시적으로 알려주므로, 사실상 “주어진 class를 foreground/background로 나누는 문제”에 가깝다. 반면 GFS-Seg는 실제 multi-class segmentation에 더 가깝다.

### 3.2 Prototype Learning 기반 baseline

논문은 먼저 prototype learning 관점에서 baseline을 구성한다. FS-Seg에서 각 novel class $c_i$의 prototype $p_i$는 support image feature를 해당 class mask로 average pooling해서 만든다. 식은 다음과 같다.

$$
p_i = \frac{1}{K} \ast \sum_{j=1}^{K}
\frac{\sum_{h,w} [m_i^j \circ F(s_i^j)]_{h,w}}
{\sum_{h,w}[m_i^j]_{h,w}}
$$

여기서 $F$는 feature extractor, $m_i^j$는 support sample의 binary mask, $\circ$는 Hadamard product이다. 쉽게 말하면, support image에서 해당 class가 있는 픽셀 feature만 평균내어 class 대표 벡터를 만드는 것이다.

하지만 GFS-Seg에서는 novel class만이 아니라 base class도 함께 예측해야 한다. 모든 base sample을 다시 feature extractor에 넣어 base prototype을 만드는 것은 비효율적이므로, 저자들은 일반 segmentation model의 classifier weight 자체를 **base prototype**으로 해석한다. 즉, classifier가 $N_b \times d$ 크기라면, 이것을 $N_b$개의 base class prototype 집합 $P_b \in \mathbb{R}^{N_b \times d}$로 본다.

그 다음 support로부터 만든 novel prototype $P_n \in \mathbb{R}^{N_n \times d}$를 이어 붙여 전체 classifier $P_{all}$을 만든다. 최종 예측은 dot product 대신 cosine similarity를 사용한다. query image의 픽셀 feature $F(q_{x,y})$와 각 prototype $p_i$의 cosine similarity를 구하고 softmax로 class를 결정한다.

$$
O_{x,y} = \arg\max_i
\frac{\exp(\alpha \phi(F(q_{x,y}), p_i))}
{\sum_{p_i \in P_{all}} \exp(\alpha \phi(F(q_{x,y}), p_i))}
$$

여기서 $\phi$는 cosine similarity이고, $\alpha$는 scale factor로 실험에서 10으로 고정되었다. 저자들은 일반 classifier의 dot product scale이 prototype averaging과 잘 맞지 않기 때문에 cosine similarity가 더 적합하다고 설명한다.

### 3.3 CAPL: Support Contextual Enrichment (SCE)

SCE는 **novel class registration phase**에서 작동한다. support sample 안에는 novel class뿐 아니라 몇몇 base class도 함께 나타날 수 있다. 저자들은 이 base class들의 support feature를 이용해 기존 base classifier를 문맥적으로 보정한다.

우선 support에 등장한 base class $c_{b,i}$에 대해 support-derived prototype $p_{b,i}^{sup}$를 만든다.

$$
p_{b,i}^{sup} =
\frac{
\sum_{u=1}^{N_n}\sum_{j=1}^{K}\sum_{h,w}[m_{i,u}^j \circ F(s_u^j)]_{h,w}
}{
\sum_{u=1}^{N_n}\sum_{j=1}^{K}\sum_{h,w}[m_{i,u}^j]_{h,w}
}
$$

이 식은 novel support images 안에서 base class $c_{b,i}$가 나타난 부분들의 feature를 모두 평균내는 것이다. 그 뒤 기존 classifier weight $p_{b,i}^{cls}$와 support prototype $p_{b,i}^{sup}$를 adaptive하게 섞는다.

$$
p_{b,i} = \gamma_i^{sup} \ast p_{b,i}^{cls} + (1 - \gamma_i^{sup}) \ast p_{b,i}^{sup}
$$

여기서 $\gamma_i^{sup} = G_{sup}(p_{b,i}^{cls}, p_{b,i}^{sup})$ 이며, $G_{sup}$는 두 prototype의 관계를 보고 mixing weight를 내는 함수이다. 논문에 따르면 실험적으로 $G_{sup}$는 **2-layer MLP**가 가장 잘 작동했다.

의미는 직관적이다. 기존 classifier는 base dataset 전반에서 학습된 안정적인 class 표현이고, support-derived prototype은 이번 novel registration 상황에서 얻은 새로운 co-occurrence 문맥을 반영한다. SCE는 이 둘을 데이터 의존적으로 결합해 base prototype을 업데이트한다.

### 3.4 CAPL: Dynamic Query Contextual Enrichment (DQCE)

SCE만으로는 support set의 문맥이 너무 제한적일 수 있다. support는 적은 수의 예제이므로, 그 문맥이 모든 query image에 잘 맞는다고 보장할 수 없다. 이를 보완하기 위해 저자들은 **DQCE**를 제안한다.

DQCE는 evaluation 시 각 query image마다 동적으로 classifier를 조정한다. 우선 원래 base classifier로 query feature $F(q)$에 대해 임시 prediction $y_{qry}$를 만든다. 그 후 이 예측을 soft weighting으로 사용해 query 내부의 class-wise representative feature를 계산한다.

$$
p_b^{qry} = Softmax(y_{qry}^t) \times F(q)
$$

이 식은 각 base class마다 query feature map에서 “그 class일 가능성이 높은 위치들”을 가중 평균해 임시 class prototype을 만드는 과정이다. 다만 이 값은 ground truth가 아니라 임시 prediction에서 나온 것이므로 noisy할 수 있다. 그래서 역시 신뢰도 weight $\gamma_i^{qry}$가 필요하다.

$$
p_{b,i}^{dyn} = \gamma_i^{qry} \ast p_{b,i}^{cls} + (1 - \gamma_i^{qry}) \ast p_{b,i}^{qry}
$$

여기서 $\gamma_i^{qry} = G_{qry}(p_{b,i}^{cls}, p_{b,i}^{qry})$ 이다. 논문은 실험적으로 $G_{qry}$에는 **cosine similarity**가 더 적합하다고 보고한다. 이유는 query-based prototype은 잡음이 많을 수 있으므로, MLP처럼 쉽게 높은 신뢰도를 주는 방식보다 similarity 기반 보수적 판단이 더 낫기 때문이다.

마지막으로 SCE로 얻은 prototype과 DQCE로 얻은 dynamic prototype을 결합해 최종 CAPL base prototype을 만든다.

$$
p_{b,i}^{capl} = p_{b,i} + p_{b,i}^{dyn}
$$

이렇게 만든 $P_b^{capl}$을 novel prototype $P_n$과 concatenation하여 최종 classifier $P_{all}^{capl}$을 만들고, 이를 이용해 base와 novel classes를 모두 예측한다.

### 3.5 학습 절차

논문에서 중요한 부분은, 위 식들을 테스트 때만 적용하면 feature space와 classifier weight space가 맞지 않아 잘 동작하지 않는다는 점이다. 즉, 단순히 support/query feature 평균을 기존 classifier weight와 섞으려면, feature extractor가 그런 결합이 가능하도록 학습되어야 한다.

이를 위해 저자들은 학습 중에 **Fake Support**, **Fake Query**, **Fake Novel**, **Fake Context**라는 구성을 만든다. 배치의 절반 정도를 Fake Support로 뽑고, 거기에 등장한 base classes 중 일부를 Fake Novel, 나머지를 Fake Context로 지정한다. 이름은 novel/context이지만 실제로는 모두 base classes다. 다만 테스트 시 novel registration과 context enrichment가 일어나는 상황을 흉내 내기 위해 이렇게 역할을 부여한다.

- Fake Novel class는 novel class처럼 prototype로 classifier weight를 대체한다.
- Fake Context class는 support에 함께 등장하는 base class처럼 기존 weight와 support prototype을 섞는다.
- 나머지 class는 원래 classifier weight를 유지한다.

논문의 식 (8), (9)는 이를 형식화한 것이다. 최종적으로 이렇게 만든 context-aware classifier $P_b^{capl}$로 prediction을 수행하고, **표준 cross-entropy loss**를 최소화한다. 즉, 손실 함수 자체는 특별하지 않고, classifier를 구성하는 방식과 학습 스킴이 핵심이다.

## 4. 실험 및 결과

### 4.1 기존 FS-Seg 모델을 GFS-Seg에서 평가

논문은 먼저 대표적인 FS-Seg 모델들인 CANet, PFENet, SCL, PANet을 GFS-Seg setting에서 평가한다. 여기서 핵심은 support/query 구조를 유지한 원래 방식이 아니라, base prototype과 novel prototype을 모두 넣어 실제 GFS-Seg처럼 전체 class를 예측하도록 inference를 수정했다는 점이다.

Pascal-5i에서 ResNet-50 backbone 기준 결과를 보면, relation-based FS-Seg 모델들인 CANet, PFENet, SCL은 total mIoU가 1-shot/5-shot에서 약 7 정도로 매우 낮다. PANet은 상대적으로 낫지만 total mIoU가 26.97 / 28.74에 그친다. 반면 같은 prototype 기반 계열에 CAPL을 붙인 **PANet + CAPL**은 51.60 / 53.30으로 크게 올라간다. 더 나아가 일반 segmentation backbone에 CAPL을 붙인 **DeepLab-V3 + CAPL**은 53.77 / 56.59, **PSPNet + CAPL**은 54.38 / 55.72를 기록한다.

이 결과는 기존 FS-Seg 모델이 GFS-Seg에서 성능이 급락함을 보여준다. 저자들은 그 원인으로 다음을 든다.

- FS-Seg의 episodic 학습은 foreground vs background 구분에 치우쳐 있다.
- query에 어떤 class가 들어 있는지 support가 알려준다는 전제가 GFS-Seg에는 없다.
- 일부 방법은 backbone을 고정한 채 학습해 복잡한 multi-class labeling 상황에 잘 적응하지 못한다.

즉, GFS-Seg는 단순히 “더 어려운 테스트셋”이 아니라, 모델의 문제 정의 자체를 바꾸는 설정이라는 점이 실험으로 뒷받침된다.

### 4.2 CAPL 구성요소 ablation

PSPNet + ResNet-50 기반 Pascal-5i GFS-Seg 실험에서, baseline은 1-shot total mIoU 49.54, 5-shot 51.12이다. 여기에 DQCE만 추가하면 52.55 / 54.80까지 오른다. SCE만 추가해도 51.63 / 53.35 정도로 개선된다. 둘을 합친 **CAPL**은 54.38 / 55.72로 가장 좋다. 따라서 support context와 query context가 상호보완적이라는 결론이 나온다.

또한 weighting 함수 선택도 비교했다. 논문은 DQCE에서는 cosine similarity가, SCE에서는 MLP가 더 적합하다고 보고한다. DQCE의 query-derived prototype은 noisy할 수 있으므로 similarity 기반 보수적 weighting이 더 안전하고, SCE의 support-derived prototype은 ground-truth mask를 사용해 비교적 신뢰도가 높으므로 MLP가 적당한 중간 값을 내며 기존 정보와 새 문맥 정보를 균형 있게 섞는 것이 유리하다고 해석한다.

### 4.3 학습 전략의 중요성

논문은 CAPL의 성능이 단순히 테스트 시 prototype을 섞는 기법 때문만이 아니라, 이를 가능하게 하는 학습 전략 덕분이라는 점도 보여준다.

- `Baseline+`는 Fake Novel replacement만 적용한 변형 baseline이다.
- `CAPL-Tr`는 CAPL식 학습만 하고 테스트 시 enrichment를 하지 않는다.
- `CAPL-Te`는 테스트 시 enrichment만 하고 학습은 baseline처럼 한다.

결과를 보면 `CAPL-Tr`는 baseline보다 소폭 개선되지만 제한적이고, `CAPL-Te`는 오히려 novel mIoU가 크게 떨어진다. 예를 들어 1-shot에서 `CAPL-Te`의 novel mIoU는 7.00으로 baseline 14.55보다 훨씬 낮다. 반면 full CAPL은 18.85로 가장 높다. 이 결과는 feature와 classifier weight의 정렬(alignment)을 학습 단계에서 미리 만들어 두지 않으면, 테스트 시 contextual enrichment가 제대로 작동하지 않음을 보여준다.

### 4.4 FS-Seg에도 적용 가능성

흥미롭게도 저자들은 CAPL이 GFS-Seg만을 위한 방법이 아니라, FS-Seg의 extreme case에도 적용 가능하다고 주장한다. Table 4에서 PANet과 PFENet에 CAPL을 붙여 Pascal-5i와 COCO-20i에서 평가한 결과, baseline 대비 상당한 향상이 있다.

예를 들어 ResNet-50 기준:

- `CAPL (PANet)`은 Pascal-5i에서 60.6 / 66.1, COCO-20i에서 38.0 / 47.3
- `CAPL (PFENet)`은 Pascal-5i에서 62.2 / 67.1, COCO-20i에서 39.8 / 48.3

ResNet-101 기준 `CAPL (PFENet)`은 COCO-20i에서 42.8 / 50.4로 더 높은 수치를 보인다. 다만 Pascal-5i에서는 HSNet이 더 높다. 논문 말미에서도 저자들은 CAPL이 dense spatial reasoning 자체를 새로 설계한 방법은 아니므로, hyper-correlation을 직접 활용하는 HSNet 같은 방법보다 항상 우위에 있는 것은 아니라고 명시한다.

### 4.5 정성적 결과

Figure 4의 시각화에서 baseline 대비 SCE, 그리고 SCE+DQCE를 적용했을 때 prediction이 점진적으로 개선되는 예시를 보여준다. 제공된 텍스트에는 개별 이미지별 세부 설명은 없지만, 저자들은 SCE와 DQCE가 baseline의 오류를 줄이고 더 정교한 segmentation을 만든다고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정에 대한 비판과 재정의가 설득력 있다**는 점이다. 단지 기존 방법에 새로운 module을 얹은 수준이 아니라, FS-Seg의 평가 전제가 실제 응용과 다르다는 점을 명확히 짚고, GFS-Seg라는 더 현실적인 benchmark를 제안했다. 이는 few-shot segmentation 연구가 무엇을 실제로 해결하고 있는지 다시 묻게 만드는 기여다.

또 다른 강점은 **방법이 비교적 단순하고 일반적**이라는 점이다. CAPL은 backbone 구조를 대폭 바꾸지 않고 classifier/prototype 형성과 학습 절차를 조정하는 방식이므로, PSPNet, DeepLab, PANet, PFENet 등 여러 기반 모델에 적용 가능하다. 논문이 제시한 실험도 이 점을 잘 뒷받침한다. 특히 GFS-Seg뿐 아니라 FS-Seg에도 경쟁력 있는 성능 향상을 보인 점은 방법의 범용성을 보여준다.

세 번째 강점은 **context를 support와 query 양쪽에서 분리해 다룬 설계가 명확하다**는 점이다. support에서 얻는 co-occurrence prior와 query에서 얻는 image-specific context는 성격이 다르며, 이를 SCE와 DQCE로 나눈 것은 직관적이다. ablation도 이 설계를 비교적 잘 정당화한다.

하지만 한계도 분명하다. 가장 먼저, CAPL은 본질적으로 **classifier/prototype 보정 방식**이지, query-support 간의 정교한 dense spatial matching을 직접 수행하는 방법은 아니다. 저자들 스스로도 결론에서 HSNet과 비교하며 이 점을 인정한다. 그래서 class 수가 적고 정밀한 상호상관 구조가 중요한 FS-Seg 조건에서는 항상 최상위 성능을 보이지 않는다.

또한 DQCE는 query에서 임시 prediction을 바탕으로 prototype을 만드는 구조이므로, 초기에 잘못된 prediction이 들어가면 noisy prototype이 생성될 위험이 있다. 논문은 이를 reliability weight로 완화하지만, 이 위험이 완전히 사라진다고 보기는 어렵다. 특히 매우 어려운 query나 base class confusion이 큰 상황에서 얼마나 안정적인지는 제공된 텍스트만으로는 자세히 판단하기 어렵다.

학습 절차도 다소 복잡하다. Fake Support, Fake Novel, Fake Context를 구성해 feature-classifier alignment를 유도하는 방식은 합리적이지만, 구현 관점에서는 baseline보다 학습 설계가 복잡해진다. Supplementary에 구현 세부가 있다고만 되어 있고, 본문 텍스트만으로는 sampling 전략의 민감도나 학습 안정성에 대한 정량 분석은 충분히 제시되지 않았다.

또 하나의 한계는 benchmark의 실용성 주장과 별개로, **실제 배포 환경에서 novel registration이 얼마나 자주 일어나고 얼마나 비용이 드는지**에 대한 논의는 제한적이라는 점이다. 논문은 등록이 “한 번” 이루어진다고 설명하지만, novel class가 점진적으로 계속 추가되는 continual setting까지 직접 다루는 것은 아니다. 따라서 GFS-Seg는 FS-Seg보다 현실적이지만, 여전히 실제 운영 환경 전체를 포괄한다고 보기는 어렵다.

## 6. 결론

이 논문은 few-shot semantic segmentation 연구에서 중요한 방향 전환을 제시한다. 기존 FS-Seg가 “지원된 novel class만 찾는 문제”에 가까웠다면, 저자들은 이를 **base와 novel을 함께 다루는 generalized setting**으로 확장해 더 실용적인 GFS-Seg를 제안했다. 그리고 이 설정에서 기존 FS-Seg 방법들이 크게 약화된다는 점을 실험으로 보인 뒤, support 문맥과 query 문맥을 함께 활용하는 **Context-Aware Prototype Learning (CAPL)**을 통해 강한 baseline을 구축했다.

방법론적으로 CAPL의 핵심 기여는 classifier를 고정된 weight 집합이 아니라, support와 query에 따라 조정 가능한 context-aware prototype 집합으로 본 데 있다. 이 관점은 semantic segmentation에서 context가 중요하다는 기존 통찰을 few-shot/generalized few-shot 조건으로 자연스럽게 확장한 것이다.

실제 적용 측면에서도 의미가 있다. 구조를 크게 바꾸지 않고 기존 segmentation model에 붙일 수 있기 때문에, 소량의 새 클래스 예제를 빠르게 등록하면서도 기존 클래스 성능을 유지하려는 응용에 유용할 가능성이 있다. 동시에 이 연구는 향후 few-shot segmentation이 단순 episodic novel-class matching을 넘어서, **open-world에 가까운 multi-class dense prediction 문제**로 나아가야 함을 보여주는 출발점으로 볼 수 있다.
