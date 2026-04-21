# Few-Shot Learning with Geometric Constraints

- **저자**: Hong-Gyu Jung, Seong-Whan Lee
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2003.09151

## 1. 논문 개요

이 논문은 이미 많은 데이터로 학습된 **base categories** 분류기에, 매우 적은 수의 예시만 가진 **novel categories**를 추가하는 few-shot learning 문제를 다룬다. 논문이 설정한 핵심 상황은 단순히 novel category만 맞히는 것이 아니라, **base와 novel을 함께 분류해야 한다**는 점이다. 즉, 기존에 잘 학습된 base category 성능을 유지하면서도, 1-shot 또는 5-shot처럼 극히 적은 novel example만으로 새 클래스를 정확히 구분해야 한다.

저자들은 이 문제가 어려운 이유를 두 가지로 정리한다. 첫째, base와 novel 모두에서 높은 성능이 필요하다. 둘째, novel category를 몇 개 안 되는 샘플로 fine-tuning하면, 원래 base categories를 위해 잘 형성된 feature space가 망가질 수 있다. 실제로 논문은 softmax 기반의 단순 fine-tuning이 base feature space를 오염시키는 예를 그림으로 보여준다.

이 문제는 실제 응용에서도 중요하다. 현실에서는 모든 새 클래스를 위해 대규모 데이터를 다시 수집하고 전체 모델을 재학습하는 것이 비싸고 비효율적이다. 따라서 기존 모델의 지식을 유지하면서 새로운 클래스를 빠르게 추가할 수 있는 방법은 지속적 학습과 실제 배포 시스템 관점에서 의미가 크다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **feature와 classifier weight를 모두 정규화하여 hypersphere 위의 각도 정보만 다루고**, 그 위에서 novel category를 위한 fine-tuning을 **두 개의 geometric constraint**로 제어하자는 것이다.

첫 번째 제약은 **Weight-Centric Feature Clustering (WCFC)** 이다. 이 제약은 novel class의 feature들이 그 클래스의 classifier weight 주변에 모이도록 만든다. 직관적으로 말하면, 같은 novel class 샘플들이 feature space에서 한 군데에 잘 뭉치게 하여 intra-class variation을 줄인다.

두 번째 제약은 **Angular Weight Separation (AWS)** 이다. 이 제약은 novel class weight가 base class weight 및 다른 novel class weight와 너무 가깝지 않도록 각도상으로 밀어낸다. 즉, 클래스 중심점들끼리 충분히 떨어지게 하여 class 간 혼동을 줄이려는 목적이다.

기존 방법들과의 차별점은 분명하다. prior work 중 일부는 base category에서 잘 학습된 activation이나 weight를 이용해 novel class의 weight를 예측한다. 그러나 이 논문은 그런 방식과 달리, **novel category의 feature space 자체를 명시적으로 제어**한다. 또한 base category 샘플을 다시 쓰지 않고도, novel category 소수 샘플만으로 fine-tuning하되 base feature space를 최대한 보존하려고 한다는 점이 특징이다.

## 3. 상세 방법 설명

전체 프레임워크는 **2-stage training**으로 구성된다.

첫 번째 단계에서는 base categories만 사용해 네트워크를 학습한다. 네트워크는 feature extractor와 classifier로 구성되며, classifier의 weight와 feature를 모두 $l_2$ normalization 한다. 따라서 분류 점수는 벡터 크기가 아니라 cosine similarity에 의해 결정된다. 이때 분류 loss는 다음과 같다.

$$
L_{cls} = -\frac{1}{M}\sum_{i=1}^{M}\log \frac{e^{s\tilde{f}_i^T \tilde{w}_{y_i}^B}}{\sum_{j=1}^{n_B} e^{s\tilde{f}_i^T \tilde{w}_{j}^B}}
$$

여기서 $\tilde{f}_i$와 $\tilde{w}_j^B$는 정규화된 feature와 base weight이고, $s$는 learnable scale parameter이다. 저자들은 cosine similarity 값의 범위가 $[-1,1]$로 너무 좁아 gradient가 약해질 수 있기 때문에 $s$를 도입한다고 설명한다. 이 단계에서는 base class feature들도 각 class weight 근처에 위치하도록 WCFC를 함께 사용한다.

두 번째 단계에서는 novel categories를 추가한다. 여기서 중요한 점은 **base를 위한 하위 convolution block은 고정(freeze)** 하고, **상위 convolution block만 복제해서 novel category 학습용으로 fine-tune**한다는 것이다. 그리고 novel classifier를 별도로 만든다. 즉, base branch와 novel branch가 완전히 같은 파라미터를 공유하는 단순 구조가 아니라, 하위층은 공유하되 상위층은 novel 적응을 위해 분리한다.

테스트 시에는 입력 이미지를 base용 경로와 novel용 경로 모두에 통과시키고, base classifier score와 novel classifier score를 합친 뒤 가장 큰 점수를 택한다. 식으로는 다음과 같다.

$$
\arg\max_k (S_k^{Both})
$$

여기서 $S^{Both} = S^{Base} \cup S^{Novel}$ 이다.

이제 핵심 loss를 보면, 먼저 **WCFC**는 다음과 같이 정의된다.

$$
L_{WCFC} = \sum_{i=1}^{C_{Novel}} -\log \big( \cos \theta_{g(f_i), \tilde{w}_i^N} \big)
$$

논문은 $g(\cdot)$를 두 방식으로 정의한다. 첫 번째는 각 클래스 feature 평균을 정규화한 것이고,

$$
g(f_i) = \frac{\bar{f}_i}{\|\bar{f}_i\|}
$$

두 번째는 각 샘플 feature를 먼저 정규화한 뒤 합산해서 다시 정규화한 것이다.

$$
g(f_i) = \frac{\sum_i \tilde{f}_i}{\|\sum_i \tilde{f}_i\|}
$$

저자들은 두 번째 방식이 magnitude 측면에서 더 많은 자유도를 가지며 few-shot novel categories에 더 유리했다고 보고한다. 직관적으로 이 loss는 “novel class의 대표 feature 방향”과 해당 class weight 방향이 최대한 일치하도록 만든다.

다음으로 **AWS**는 class weight들 사이의 angular separation을 강제한다. 먼저 weight 간 cosine similarity를

$$
u_{ij} =
\begin{cases}
\cos \theta_{\tilde{w}_i, \tilde{w}_j^N}, & \text{if } \tilde{w}_i \not\equiv \tilde{w}_j^N \\
0, & \text{otherwise}
\end{cases}
$$

로 두고, margin $m$보다 가까운 경우만 벌점을 준다. loss는 다음과 같다.

$$
L_{AWS} = \frac{\sum_{i,j} -\log(-u_{ij}\cdot 1_M(u_{ij}) + 1)}{\sum_{i,j} 1_M(u_{ij})}
$$

여기서 $M = \{u_{ij}\mid u_{ij}>m, \forall i,j\}$ 이고, $1_M(u_{ij})$는 indicator function이다. 의미는 간단하다. novel weight가 base weight 또는 다른 novel weight와 cosine similarity 기준으로 너무 가까우면 벌점을 주고, 충분히 멀어지면 더 이상 AWS를 적용하지 않는다.

최종 objective는 다음과 같다.

$$
L_{total} = \gamma L_{cls} + \alpha L_{WCFC} + \beta L_{AWS}
$$

논문에서는 기본적으로 $\gamma \alpha \beta = 111$, 즉 세 loss를 모두 사용한다. 저자 설명에 따르면 cross-entropy는 분류 자체를 담당하고, WCFC는 같은 class 내부 응집도를 높이며, AWS는 class 간 분리를 촉진한다. 이 세 요소가 함께 작동해야 novel category의 discriminative feature를 만들면서도 base feature space 훼손을 줄일 수 있다.

## 4. 실험 및 결과

논문은 두 데이터셋에서 실험했다. 첫 번째는 **miniImageNet**, 두 번째는 **Bharath & Girshick’s dataset**으로 불리는 ImageNet subset이다.

miniImageNet은 64 train, 16 validation, 20 test category로 구성되며 각 category당 600개의 $84 \times 84$ 이미지를 가진다. 이 논문은 기존 meta-learning 방식과 달리, 64개 base category를 충분한 데이터로 먼저 학습하고, 그 뒤 novel category를 few-shot으로 추가한다. feature extractor로는 C64F, C64F-Dropout, ResNetS를 사용했다.

miniImageNet 결과를 보면, ResNetS를 사용할 때 제안 방법이 가장 강하다. 예를 들어 **5-way 5-shot**에서 제안법은 novel accuracy가 **78.00 ± 0.61**, both accuracy가 **68.16**, base accuracy가 **79.78**이다. **5-way 1-shot**에서는 novel accuracy가 **58.52 ± 0.82**, both accuracy가 **56.05**, base accuracy가 **79.78**이다. 특히 Gidaris and Komodakis [19]와 비교하면 ResNetS에서 both accuracy가 크게 높아진다. 이는 단순히 novel weight를 잘 생성하는 수준이 아니라, novel feature를 더 discriminative하게 재배치한 효과로 해석할 수 있다.

또한 ablation으로 fine-tuning 없이 novel weight를 정규화된 feature 합으로만 두었을 때 성능이 떨어졌다. 예를 들어 ResNetS에서 5-way 5-shot both accuracy는 **47.11**로, 제안법의 **68.16**보다 훨씬 낮다. 이 결과는 “좋은 초기 weight만으로는 부족하고 feature extractor의 일부 fine-tuning이 중요하다”는 논문의 주장을 뒷받침한다.

ImageNet subset에서는 ResNet10과 ResNet10-Dropout을 사용했다. 여기서는 top-5 accuracy를 보고한다. **Proposed-Dropout**은 모든 shot 설정에서 좋은 성능을 보였고, 특히 “Both” 기준으로 $k=1,2,5,10,20$ shot에서 각각 **60.39, 67.44, 74.22, 77.32, 79.05**를 기록했다. “Both with prior”에서도 각각 **58.91, 65.84, 72.71, 75.96, 77.88**로 보고된다. 논문은 이 결과가 hallucination 기반 데이터 생성이나 base training examples 재사용 없이도 얻어진 성능이라는 점을 강조한다.

논문은 또한 **incremental learning** 실험도 수행했다. ResNet10-Dropout으로 1-shot부터 20-shot까지 novel examples가 점진적으로 늘어나는 상황에서 계속 fine-tuning했더니, 단순 20-shot 학습보다 더 좋은 결과가 나왔다. 예를 들어 Proposed-Dropout의 novel / both / both with prior 성능이 각각 **78.84 / 79.05 / 77.88**인 반면, incremental learning은 **79.87 / 79.44 / 78.59**였다. 이는 실제 시스템에서 새로운 샘플이 순차적으로 들어올 때 점진적 업그레이드가 가능함을 시사한다.

정성적 분석도 제시된다. t-SNE 시각화에서는 fine-tuning 전 novel feature가 넓게 퍼져 있지만, fine-tuning 후에는 base category 사이의 적절한 위치에 더 응집력 있게 놓인다. Grad-CAM 시각화에서는 fine-tuning 후 네트워크가 객체의 중요한 부분에 더 집중하고 배경 노이즈를 덜 본다고 설명한다.

추가로 single-stage fine-tuning과의 비교도 중요하다. base와 novel이 같은 feature extractor를 직접 공유하면서 마지막 block과 classifier를 few-shot으로 그냥 fine-tune하면 base 성능이 크게 무너진다. miniImageNet의 ResNetS에서 single-stage fine-tuning은 5-way 5-shot both accuracy가 **58.62**, base accuracy가 **64.58**로 제안법의 **68.16**, **79.78**보다 훨씬 낮다. ImageNet subset에서도 base accuracy 평균이 single-stage는 **90.45**, 제안법은 **93.55**로 차이가 난다. 이는 저자들의 문제 설정, 즉 “few-shot novel 학습이 기존 base representation을 망가뜨릴 수 있다”는 문제의식이 실험적으로 타당하다는 뜻이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot learning을 단지 novel-only recognition으로 보지 않고, **base와 novel의 공존 문제**로 정식화했다는 점이다. 이는 실제 분류 시스템에 더 가깝다. 또한 method 설계가 비교적 명확하다. hypersphere 상에서 feature와 weight를 다루고, 하나는 class 내부 응집도를 높이고 다른 하나는 class 간 분리를 보장하는 식으로 목적이 분리되어 있어 해석 가능성이 좋다.

또 다른 강점은 실험 설계가 단순 성능 보고에 그치지 않는다는 점이다. 저자들은 ablation, single-stage fine-tuning 비교, margin sensitivity, loss 조합 비교, feature space 시각화까지 제시하며 제안 요소의 역할을 분석한다. 특히 AWS가 both-category 성능에 중요하다는 분석은 방법의 핵심이 무엇인지 이해하는 데 도움이 된다.

하지만 한계도 분명하다. 논문이 직접 인정하듯이, 방법의 성능은 **network capacity**에 많이 의존한다. ResNetS처럼 base feature가 이미 잘 군집화되고 class 간 분리가 좋은 경우에는 큰 이득을 보지만, C64F나 ResNet10처럼 feature geometry가 덜 이상적이면 이득이 제한적이다. 즉, 이 방법은 빈약한 backbone을 근본적으로 보완하는 접근은 아니다.

또한 이 방법은 base와 novel에 대해 상위 block을 분리하고, base branch는 고정한 채 novel branch만 적응시키는 구조를 택한다. 이것은 catastrophic forgetting을 피하는 데는 유리하지만, 동시에 base와 novel representation을 완전히 공동 최적화하지는 못한다는 의미이기도 하다. 저자들도 더 나은 성능을 위해서는 base examples를 다시 함께 사용해 양쪽 feature를 조정할 수 있지만, 그러면 class imbalance와 데이터 생성 문제가 생긴다고 말한다.

비판적으로 보면, 이 논문은 geometric constraint의 효과를 잘 보였지만, 왜 특정 backbone에서 더 크게 듣는지에 대한 설명은 주로 경험적 분석에 머문다. 또한 few-shot setting에서 top-layer duplication이 계산량과 메모리 측면에서 얼마나 부담되는지는 본문에서 깊게 다루지 않았다. 하이퍼파라미터와 optimizer는 appendix에서 정리되어 있지만, 실무적으로 얼마나 튜닝 민감한지에 대한 더 체계적인 보고는 부족하다. 다만 margin sensitivity 실험은 성능이 완전히 불안정하지는 않다는 점을 보여준다.

## 6. 결론

이 논문은 base categories로 충분히 학습된 분류기에 few-shot novel categories를 추가하는 문제를 다루며, 이를 위해 **WCFC**와 **AWS**라는 두 geometric constraint를 제안했다. 핵심은 feature와 weight를 정규화하여 각도 기반 공간에서 다루고, novel feature는 자기 class weight에 가깝게 모으고, novel weight는 base 및 다른 novel weight와 충분히 멀어지게 만드는 것이다.

실험 결과는 특히 강한 backbone과 Dropout을 사용할 때 이 접근이 매우 효과적임을 보여준다. 제안법은 단순 fine-tuning처럼 base 성능을 무너뜨리지 않으면서, novel category에 대해서도 경쟁력 있는 혹은 state-of-the-art 수준의 성능을 달성했다. 따라서 이 연구는 실제 분류 시스템에서 새로운 클래스를 점진적으로 추가해야 하는 상황에 의미 있는 방향을 제시한다.

향후 연구 방향으로는 논문이 직접 언급하듯, few-shot 상황에서 더 강력한 **example generation** 또는 데이터 augmentation 기법과 결합하는 것이 중요해 보인다. 이 방법 자체는 feature geometry를 잘 제어하는 틀을 제공하므로, 더 나은 샘플 생성 기술과 결합될 경우 실제 적용 범위가 더 넓어질 가능성이 있다.
