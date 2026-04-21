# On Episodes, Prototypical Networks, and Few-Shot Learning

- **저자**: Steinar Laenen, Luca Bertinetto
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2012.09831

## 1. 논문 개요

이 논문은 few-shot learning(FSL)에서 널리 쓰이는 **episodic learning**이 정말 필요한지, 특히 **Matching Networks**와 **Prototypical Networks**처럼 에피소드 내부에서 비모수적(nonparametric) 분류를 수행하는 방법들에 대해 비판적으로 검토한다. 일반적인 episodic learning은 각 mini-batch를 support set과 query set으로 나누어, 테스트 시나리오와 비슷한 작은 학습 문제를 반복적으로 만들며 학습한다. 기존 메타러닝 연구에서는 이것이 중요하다고 여겨졌지만, 저자들은 이런 구조가 오히려 훈련 데이터 활용을 비효율적으로 만들 수 있다고 주장한다.

연구 문제는 명확하다. **에피소드 기반 학습이 metric-based few-shot learner에서 실제로 성능상 이점을 주는가, 아니면 단지 관습적으로 유지되어 온 설계인가?** 저자들은 이 질문을 Prototypical Networks(PNs), Matching Networks(MNs), 그리고 이들과 밀접하게 연결되는 **Neighbourhood Component Analysis (NCA)**의 관계를 통해 분석한다. 핵심 결론은, 적어도 이 계열의 방법에서는 episodic learning이 필수적이지 않을 뿐 아니라, 많은 경우 **성능을 떨어뜨리고 하이퍼파라미터 민감도만 높인다**는 것이다.

이 문제는 중요하다. 최근 few-shot learning에서는 복잡한 meta-learning 방법보다 단순한 embedding pretraining baseline이 더 강한 성능을 보이는 경우가 많았는데, 이 논문은 그 원인 중 하나로 **support/query 분리 자체의 비효율성**을 지목한다. 즉, 이 논문은 단지 새로운 방법을 제안하는 것이 아니라, few-shot learning 커뮤니티에서 거의 기본 전제로 받아들여진 훈련 방식의 타당성을 다시 묻는 작업이다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 단순하다. **PNs와 MNs는 test time에 episode-specific parameter adaptation을 하지 않기 때문에, 굳이 training time에 support/query로 배치를 강제 분할할 이유가 약하다**는 것이다. 이 방법들은 결국 embedding space에서 거리 기반으로 분류하므로, 배치 안의 예제들 사이의 가능한 많은 pairwise relation을 활용하는 편이 자연스럽다. 그런데 episodic learning은 어떤 샘플은 support, 어떤 샘플은 query로 역할을 고정해 버리므로, 실제로는 학습에 쓸 수 있었던 많은 거리 정보를 버리게 된다.

저자들이 강조하는 차별점은, 기존 연구들이 “train/test conditions should match”라는 직관에 따라 episodic training을 정당화해 온 반면, 이 논문은 **nonparametric metric-based learner에서는 그 직관이 반드시 성립하지 않는다**고 보였다는 점이다. 특히 PNs와 MNs의 episodic loss를 제거하고 NCA 형태의 non-episodic loss로 바꾸면, 모델은 더 단순해지고 하이퍼파라미터도 줄어들며, 성능도 좋아진다.

논문은 episodic learning을 단순히 “불필요하다”라고만 말하지 않는다. 더 강하게, **episodic learning은 배치 내 거리쌍(pairwise distances)을 임의로 일부 버리는 것과 거의 같은 효과를 낸다**고 주장한다. 즉, episodic training의 본질적 기여가 있다기보다, 학습 신호를 줄이는 제약으로 작동한다는 해석이다.

## 3. 상세 방법 설명

논문은 먼저 episodic learning을 형식화한다. 하나의 episode는 label subset $L$을 샘플링한 뒤, 각 클래스에서 support set $S$와 query set $Q$를 뽑아 구성된다. 여기서 `ways`는 클래스 수 $w = |L|$, `shots`는 클래스당 support 샘플 수 $n = |S_k|$, query 수는 $m = |Q_k|$이다. episodic training은 다음과 같이 쓸 수 있다.

$$
\arg \max_\theta \; \mathbb{E}_{L \sim \hat{E}} \mathbb{E}_{S \sim L, Q \sim L}
\left(
\sum_{(q_i, y_i)\in Q}
\log P_\theta(y_i \mid q_i, S, \rho)
\right).
$$

여기서 $f_\theta$는 이미지 embedding을 만드는 신경망이고, $\rho$는 episode 내부에서 추가로 적응되는 파라미터를 뜻한다. 중요한 점은, 이 논문이 다루는 PNs와 MNs는 사실상 **$\rho = \emptyset$**, 즉 episode별 inner adaptation이 없다는 점이다. 저자들은 바로 이 점 때문에 episodic setup의 필요성이 약하다고 본다.

### Prototypical Networks

PNs에서는 support set의 각 클래스 $k$에 대해 prototype을 만든다.

$$
c_k = \frac{1}{|S_k|} \sum_{(s_i, y_i)\in S_k} f_\theta(s_i).
$$

그 다음 query 샘플이 자기 클래스 prototype에 가깝고 다른 클래스 prototype과는 멀어지도록 다음 loss를 최소화한다.

$$
L_{\text{PNs}} =
-\frac{1}{|Q|}
\sum_{(q_i, y_i)\in Q}
\log
\left(
\frac{\exp(-\|f_\theta(q_i)-c_{y_i}\|^2)}
{\sum_{k'} \exp(-\|f_\theta(q_i)-c_{k'}\|^2)}
\right).
$$

쉽게 말하면, support set으로 클래스 중심점을 만들고 query가 올바른 중심점에 붙도록 학습한다.

### Matching Networks

MNs는 prototype으로 평균내지 않고, support set의 개별 샘플들과 직접 비교한다.

$$
L_{\text{MNs}} =
-\frac{1}{|Q|}
\sum_{(q_i, y)\in Q}
\log
\left(
\frac{\sum_{s_j \in S_y} \exp(-\|f_\theta(q_i)-f_\theta(s_j)\|^2)}
{\sum_{s_k \in S} \exp(-\|f_\theta(q_i)-f_\theta(s_k)\|^2)}
\right).
$$

즉, query가 support set 내 같은 클래스 샘플들에 높은 확률 질량을 주도록 만든다. 원 논문은 cosine distance를 썼지만, 이 논문은 Euclidean distance가 더 잘 작동했다고 보고 모든 실험에서 이를 사용했다.

### NCA와의 연결

핵심은 NCA loss가 PNs/MNs와 구조적으로 매우 가깝다는 점이다. NCA에서는 support와 query를 나누지 않고, batch의 모든 샘플을 대칭적으로 다룬다.

$$
L_{\text{NCA}} =
-\frac{1}{|B|}
\sum_{i=1}^{b}
\log
\left(
\frac{\sum_{j \neq i,\, y_i = y_j} \exp(-\|z_i-z_j\|^2)}
{\sum_{k \neq i} \exp(-\|z_i-z_k\|^2)}
\right),
$$

여기서 $z_i = f_\theta(x_i)$이다. 이 loss는 같은 클래스 간 거리를 줄이고 다른 클래스 간 거리를 늘리도록 embedding을 학습한다. support/query 분리가 없으므로, batch 안의 더 많은 positive/negative pair를 모두 활용할 수 있다.

저자들은 세 방법의 차이를 세 가지로 정리한다.

첫째, **PNs/MNs는 query와 support 사이 거리만 사용하지만, NCA는 batch 내부의 모든 거리쌍을 사용한다.**  
둘째, **PNs만 prototype을 만든다.**  
셋째, episodic sampling은 일부 샘플이 더 자주 뽑히는 구조를 만들 수 있지만, NCA는 보통 standard supervised learning처럼 epoch마다 데이터를 한 번씩 순회한다.

### 데이터 효율성 분석

논문의 중요한 분석은, episodic learning이 얼마나 많은 pair를 버리는지 수식으로 보였다는 점이다. 배치에서 클래스 수가 $w$, 클래스당 support/query 수가 각각 $n, m$일 때, NCA가 추가로 활용할 수 있는 pair 수는 support/query 분리 때문에 크게 증가한다. 논문은 빠진 training signal이 대략 $O(w^2(m^2+n^2))$ 규모로 커진다고 설명한다.

부록의 유도에 따르면, positive pair 수는 NCA가 PNs/MNs보다 항상 크거나 같고, negative pair 수는 항상 더 많다. 직관적으로는, episodic setup에서는 “query는 support만 본다”는 제약 때문에 **support-support**, **query-query** 관계가 학습에 사용되지 않는다. 저자들은 이것이 성능 차이의 주된 원인이라고 본다.

### 평가 시 few-shot 분류 방식

학습된 embedding $f_\theta$를 이용해 평가할 때 논문은 세 가지 분류 방식을 비교한다.

첫째는 $k$-NN이다.  
둘째는 클래스 centroid에 가장 가까운 클래스를 택하는 **nearest centroid**이다.  
셋째는 support 샘플들에 대한 softmax 확률을 계산해 클래스별로 합치는 **soft assignments**이다.

논문은 실험적으로 nearest centroid가 가장 잘 작동했다고 보고, 기본 평가 방식으로 주로 이를 사용한다. 저자들은 soft assignments가 기대보다 약한 이유를 calibration 문제로 추정하지만, 이는 논문의 해석이며 확정적 증명은 아니다.

## 4. 실험 및 결과

실험은 **mini ImageNet**, **CIFAR-FS**, **tiered ImageNet**에서 수행되었다. backbone은 ResNet-12 변형을 사용했고, feature centering과 normalization을 적용했다. 평가는 표준적인 5-way, 15-query, 1-shot 또는 5-shot 설정에서 이루어졌고, 각 설정마다 10,000개 episode로 평가했다. 또한 모델마다 서로 다른 3개의 random seed로 학습하여 총 30,000개 episode 기준으로 95% confidence interval을 계산했다.

### 에피소드 하이퍼파라미터 민감도

먼저 저자들은 PNs와 MNs가 episodic hyperparameter인 $\{w, n, m\}$에 매우 민감하다는 점을 보여준다. batch size를 128, 256, 512로 바꾸고, $m+n$을 8, 16, 32로 바꿔가며 비교했을 때, 같은 모델이라도 episode 구성에 따라 성능 차이가 크게 난다. 반면 NCA는 사실상 batch size만 정하면 되므로 훨씬 단순하다.

CIFAR-FS와 mini ImageNet validation 결과에서 **NCA는 모든 batch size와 모든 episodic configuration보다 consistently 더 좋은 성능**을 보였다. 또한 원래 PNs/MNs 논문에서 사용한 episode 설정보다, 저자들이 별도로 탐색해 찾은 더 좋은 episode 설정이 존재했다. 이는 기존에 널리 사용되던 episodic setup이 최적이 아니었음을 뜻한다.

### 랜덤 pair sub-sampling과의 비교

가장 설득력 있는 실험 중 하나는, NCA에서 배치 내 거리쌍 중 일부만 랜덤하게 남기고 나머지를 버리는 실험이다. 이렇게 하면 pair를 적게 쓰는 상황을 인위적으로 만들 수 있다. 결과적으로, **PNs와 MNs의 성능 점들이 “pair를 일부만 쓰는 NCA” 곡선 위 또는 그 근처에 놓였다.**

이 결과는 episodic learning이 특별한 일반화 이득을 주기보다, 단순히 쓸 수 있는 거리쌍을 줄이는 효과와 거의 같다는 주장을 뒷받침한다. 즉, support/query 분리가 “학습을 더 잘되게 하는 구조적 inductive bias”라기보다, “batch 내 정보를 덜 쓰는 제약”처럼 보인다는 것이다.

### Ablation study

저자들은 PNs와 NCA의 차이를 세부적으로 분해하는 ablation을 수행했다.

첫째, replacement sampling 여부는 영향이 거의 없었다. 즉, episodic sampling의 “with replacement” 자체는 핵심 요인이 아니었다.  
둘째, prototype을 없애는 ablation은 5-shot 성능을 약간 떨어뜨렸지만, 그 효과는 크지 않았다. 따라서 **prototype 생성 자체가 주된 문제는 아니다.**  
셋째, support/query 분리를 없애고 더 많은 pair를 쓰게 하면 성능이 개선되었다.  
마지막으로, support/query 분리 제거와 prototype 제거 등을 함께 적용해 NCA에 가까운 형태로 만들면, **PNs가 잃었던 성능이 거의 완전히 회복되었다.**

이 결과는 성능 열세의 가장 중요한 원인이 support/query 역할 분리라는 논문의 주장을 강하게 지지한다.

### 최근 방법들과의 비교

테스트셋 비교(Table 2)에서 저자들의 NCA 변형은 매우 단순함에도 강한 결과를 보였다.

mini ImageNet에서 NCA nearest centroid는 1-shot 62.55%, 5-shot 78.27%를 기록했다.  
CIFAR-FS에서는 1-shot 72.49%, 5-shot 85.15%였다.  
tiered ImageNet에서는 1-shot 68.35%, 5-shot 83.20%였다.

이는 같은 표에 있는 여러 episodic meta-learning 방법과 경쟁 가능하거나 일부 경우 더 나은 수치다. 또한 SimpleShot 같은 strong non-episodic baseline과 비교해도 충분히 경쟁력 있다. 특히 저자들의 “best episode”로 튜닝한 PNs/MNs보다도 NCA가 자주 더 강했다.

부록 결과도 흥미롭다. NCA로 학습한 embedding의 test-time classifier를 비교했을 때, nearest centroid가 soft assignments보다 더 잘 작동했다. 또한 intermediate layer feature를 concatenate하거나 support set에서 추가 최적화를 수행하면 약간 더 나아질 수 있음을 보였지만, 본 논문의 핵심 기여는 거기에 있지 않다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은, few-shot learning에서 너무 당연하게 받아들여졌던 설계를 **정면으로 해부했다는 점**이다. 단순히 “더 좋은 방법”을 던지는 것이 아니라, 왜 기존 방법이 비효율적인지를 loss 구조와 pair 수 계산으로 설명한다. 특히 PNs, MNs, NCA의 관계를 한 프레임 안에서 정리하고, support/query 분리가 실제 training signal을 어떻게 줄이는지 보여준 점이 강하다.

또 다른 강점은 실험 설계가 비교적 정직하다는 점이다. 저자들은 PNs/MNs에 대해서도 episode hyperparameter를 탐색해 가능한 좋은 설정을 찾아준 뒤 NCA와 비교했다. 즉, 일부러 약한 baseline을 세운 것이 아니라는 점을 강조한다. 또한 여러 데이터셋, 여러 batch size, 여러 평가 방식, 그리고 ablation을 폭넓게 수행해 주장의 재현성과 설득력을 높였다.

한편 한계도 있다. 가장 중요한 것은 이 논문의 결론이 **metric-based nonparametric few-shot learner**에 강하게 맞추어져 있다는 점이다. 저자들도 명시하듯, MAML처럼 episode 내부에서 실제 parameter adaptation이 일어나는 방법까지 일반화할 수는 없다. 즉, “episodic learning 전체가 불필요하다”가 아니라, **적어도 $\rho=\emptyset$인 계열에서는 그러하다**는 것이 더 정확한 해석이다.

또한 성능 향상의 원인을 상당 부분 pair 수 증가로 설명하지만, 논문 스스로도 부록에서 인정하듯 이것만으로 모든 차이를 완전히 설명하지는 못한다. positive/negative balance, batch 안 클래스 수, 학습 중 class diversity 같은 요소도 함께 작용한다. 따라서 “episode가 안 좋은 이유는 오직 pair 수 때문”이라고 단정하면 과도한 단순화가 된다.

비판적으로 보면, 논문의 문제 제기는 매우 강력하지만, 이 결과가 실제 few-shot evaluation protocol 자체에 어떤 더 넓은 이론적 함의를 갖는지까지는 깊게 다루지 않는다. 예를 들어 support/query separation이 있는 알고리즘과 없는 알고리즘의 일반화 차이를 이론적으로 비교하는 분석은 제공되지 않는다. 저자도 관련 bound가 없는 점을 future work 방향으로 언급한다.

## 6. 결론

이 논문은 few-shot learning에서 널리 사용되는 episodic learning이, 적어도 Prototypical Networks와 Matching Networks 같은 **비모수적 metric-based 방법**에서는 필수적이지 않으며 오히려 불리할 수 있음을 보였다. support/query 분리는 배치 내에서 활용 가능한 pairwise distance를 크게 줄이고, 그 결과 성능 저하와 하이퍼파라미터 민감도를 초래한다. 이에 비해 NCA는 같은 문제를 더 단순한 방식으로 다루면서도 더 많은 학습 신호를 활용할 수 있다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, episodic hyperparameter의 민감성과 비효율성을 실험적으로 보였다. 둘째, PNs/MNs와 NCA의 관계를 통해 성능 차이의 핵심 원인이 support/query 분리임을 ablation으로 밝혔다. 셋째, 단순한 NCA 기반 학습이 여러 대표 FSL benchmark에서 충분히 강한 baseline이 될 수 있음을 보였다.

실제 적용 측면에서 이 연구는 의미가 크다. few-shot learning 시스템을 설계할 때, 무조건 episodic training을 도입하기보다 **문제 구조가 정말 episode adaptation을 필요로 하는지** 먼저 따져야 함을 시사한다. 향후 연구에서도 복잡한 meta-learning 구조를 추가하기 전에, batch 전체를 더 효율적으로 쓰는 non-episodic metric learning 관점이 강력한 출발점이 될 가능성이 크다.
