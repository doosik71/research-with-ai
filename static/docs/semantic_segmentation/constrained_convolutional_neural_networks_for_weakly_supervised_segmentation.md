# Constrained Convolutional Neural Networks for Weakly Supervised Segmentation

- **저자**: Deepak Pathak, Philipp Krahenbuhl, Trevor Darrell
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1506.03648

## 1. 논문 개요

이 논문은 pixel-level annotation 없이, 오직 image-level tag 같은 약한 supervision만으로 semantic segmentation을 학습하는 문제를 다룬다. 즉, 학습 이미지에 대해 "car가 있다", "person이 있다" 같은 클래스 존재 정보만 주어졌을 때, 각 픽셀에 어떤 클래스를 할당해야 하는지를 예측하는 CNN을 학습하는 것이 목표다.

연구 문제의 핵심은 weak supervision이 너무 거칠다는 점이다. segmentation은 본질적으로 structured prediction 문제인데, 기존 CNN 기반 방법들은 보통 각 픽셀의 정답이 필요하다. 그러나 실제로는 pixel annotation 비용이 매우 크기 때문에, 더 싸게 얻을 수 있는 image-level label, bounding box, 대략적인 object size 같은 약한 정보만으로 학습할 수 있다면 확장성이 크게 좋아진다.

저자들은 이런 약한 정보를 “출력 분포에 대한 선형 제약(linear constraints)”으로 표현한다. 예를 들어 이미지에 car가 있다고 하면 car로 분류된 픽셀이 어느 정도는 존재해야 하고, 이미지에 없는 클래스는 예측되지 않아야 한다. 논문의 주장은, 이런 제약을 CNN 출력에 직접 걸기보다, 제약을 만족하는 latent distribution $P(X)$를 하나 두고 CNN 출력 $Q(X \mid \theta)$가 그 분포를 따르도록 학습하면 최적화가 쉬워지고 성능도 좋아진다는 것이다.

이 문제의 중요성은 분명하다. semantic segmentation 같은 dense labeling 작업은 fully supervised 데이터 구축 비용이 매우 높다. 따라서 image-level tag 수준의 supervision만으로도 실용적인 성능을 낼 수 있다면, 더 많은 클래스와 더 큰 데이터셋으로 확장하기 쉬워진다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “약한 supervision을 직접적인 정답으로 쓰지 말고, 출력 공간이 만족해야 할 제약 조건으로 바꾸자”는 것이다. 그리고 CNN의 출력을 그 제약을 만족하는 latent label distribution 쪽으로 끌어당기는 방식으로 학습한다.

보다 구체적으로, 각 이미지에 대해 CNN은 픽셀별 class probability를 출력한다. 그런데 이 출력은 weak label 제약을 바로 만족하지 않을 수 있다. 저자들은 이때 제약을 만족하는 latent distribution $P(X)$를 먼저 구하고, CNN 출력 분포 $Q(X \mid \theta)$와 $P(X)$ 사이의 KL divergence를 최소화하도록 네트워크를 업데이트한다.

차별점은 다음과 같다.

기존 MIL 계열 접근은 보통 “적어도 하나의 positive instance가 있어야 한다” 같은 약한 규칙을 활용하지만, initialization에 민감하고 나쁜 local optimum에 빠지기 쉽다. 반면 이 논문은 다양한 형태의 선형 제약을 통합적으로 표현할 수 있고, 이를 만족하는 분포를 intermediate target으로 사용함으로써 최적화를 더 안정적으로 만든다.

또한 Papandreou 등의 EM-Adapt가 adaptive bias로 클래스 존재 여부를 반영했다면, 이 논문은 그것을 더 일반화한 constrained optimization 관점으로 본다. 즉 EM-Adapt는 suppression/foreground 제약의 특수한 경우로 해석될 수 있지만, CCNN은 background area 제약, size 제약 등 더 풍부한 선형 제약을 직접 다룰 수 있다.

## 3. 상세 방법 설명

### 3.1 기본 설정

이미지 $I$의 픽셀 라벨 집합을 $X = \{x_0, \dots, x_n\}$라고 두고, 각 $x_i$는 label set $L = \{1, \dots, m\}$ 중 하나를 가진다. CNN은 각 픽셀에 대해 softmax 분포를 예측하며, 전체 분포를 독립 marginals의 곱으로 모델링한다.

$$
Q(X \mid \theta, I) = \prod_i q_i(x_i \mid \theta, I)
$$

각 픽셀의 확률은 다음과 같다.

$$
q_i(x_i \mid \theta, I) = \frac{1}{Z_i}\exp(f_i(x_i; \theta, I))
$$

여기서 $f_i(x_i; \theta, I)$는 네트워크가 출력한 score이고, $Z_i$는 softmax 정규화 상수다. score가 클수록 해당 라벨일 가능성이 높다.

보통 fully supervised learning에서는 각 픽셀의 ground truth label이 있으므로 cross entropy를 바로 최소화하면 된다. 하지만 weak supervision에서는 픽셀 정답이 없으므로, 대신 출력이 만족해야 할 제약만 주어진다.

### 3.2 원래의 constrained formulation

논문은 CNN 출력의 vectorized form을 $\tilde Q_I$라고 두고, 각 이미지마다 다음과 같은 선형 제약을 건다.

$$
A_I \tilde Q_I \ge \tilde b_I \quad \forall I
$$

여기서 $A_I$와 $\tilde b_I$는 해당 이미지에 대해 어떤 클래스가 존재하는지, 배경이 어느 정도 있어야 하는지, 특정 클래스 크기가 큰지 작은지 같은 정보를 제약식으로 표현한 것이다.

문제는 이 제약이 출력 $Q$에 대해서는 convex해도, 네트워크 파라미터 $\theta$에 대해서는 일반적으로 non-convex라는 점이다. deep network를 직접 이 constrained objective로 최적화하는 것은 어렵다.

### 3.3 latent distribution 도입

이를 해결하기 위해 저자들은 latent distribution $P(X)$를 도입한다. 핵심은 제약을 CNN 출력 $Q$가 아니라 $P$에 걸고, $Q$는 $P$를 따라가게 만드는 것이다.

최적화 문제는 다음과 같다.

$$
\min_{\theta, P} D(P(X)\|Q(X \mid \theta))
$$

subject to

$$
A\tilde P \ge \tilde b,\quad \sum_X P(X)=1
$$

여기서 $D(P\|Q)$는 KL divergence다. 논문 본문 표현대로 쓰면,

$$
D(P(X)\|Q(X \mid \theta)) = \sum_X P(X)\log P(X) - \mathbb{E}_{X \sim P}[\log Q(X \mid \theta)]
$$

즉, 제약을 만족하는 분포 $P$를 찾고, 네트워크 출력 $Q$가 그 분포와 최대한 가까워지도록 한다.

저자들은 제약이 satisfiable하다면 이 문제와 원래 문제는 동치라고 설명한다. 또한 최적의 $P(X)$는 independent marginals의 곱 형태로 factorize될 수 있다고 보였다. 이 점이 매우 중요하다. 원래 structured distribution 전체를 다룰 필요 없이, 픽셀별 marginal $p_i(x_i)$만 계산하면 되기 때문이다.

### 3.4 latent distribution 최적화

$\theta$를 고정하면, $P$에 대한 문제는 convex optimization이 된다. 저자들은 dual form을 유도하고 projected gradient ascent로 최적화한다. dual objective는 다음과 같다.

$$
L(\lambda) = \lambda^\top \tilde b - \sum_{i=1}^n \log \sum_{l \in L}\exp\big(f_i(l;\theta) + A_{i;l}^\top \lambda\big)
$$

여기서 $\lambda \ge 0$는 inequality constraints의 dual variable이다.

이때 latent marginal은

$$
p_i(x_i) = \frac{1}{Z_i}\exp\big(f_i(x_i;\theta) + A_{i;x_i}^\top \lambda\big)
$$

형태가 된다. 직관적으로는, 어떤 제약이 아직 만족되지 않으면 그 제약에 대응하는 dual variable $\lambda$가 커지고, 그 결과 각 픽셀 확률 $p_i$가 제약을 더 잘 만족하도록 조정된다.

논문에 따르면 이 dual 최적화는 보통 50 iteration 이내에 수렴하며 효율적이다.

### 3.5 SGD 단계

이제 $P$를 고정하면, 남는 것은 표준 cross entropy 학습과 같다.

$$
L(\theta) = - \sum_i \sum_{x_i} p_i(x_i)\log q_i(x_i \mid \theta)
$$

gradient는 다음과 같다.

$$
\frac{\partial L(\theta)}{\partial \tilde f_i(x_i)} = \tilde q_i(x_i \mid \theta) - \tilde p_i(x_i)
$$

즉, ordinary supervised learning에서 one-hot label을 쓰는 대신, 여기서는 constrained optimization으로 얻은 soft target distribution $p_i$를 사용해 학습한다.

논문은 이 과정을 alternating optimization으로 설명한다.

1. 현재 CNN 출력 $Q^{(t)}$에서 시작한다.
2. 제약을 만족하는 가장 가까운 latent distribution $P^{(t)}$를 구한다.
3. SGD로 네트워크를 업데이트해서 $Q^{(t+1)}$가 $P^{(t)}$를 따르도록 만든다.

이 과정을 반복하면 overall objective가 감소한다. 이론적으로는 $P$를 고정한 채 SGD를 여러 번 해야 하지만, 실제로는 매 SGD step마다 새 $P$를 추론해도 잘 동작했고 더 빨리 수렴했다고 보고한다.

### 3.6 slack variable을 이용한 완화

현실적으로는 모든 제약이 동시에 만족되지 않을 수 있다. 이를 위해 저자들은 slack variable $\xi$를 도입한다.

$$
\min_{\theta, P, \xi} D(P(X)\|Q(X \mid \theta)) + \beta^\top \xi
$$

subject to

$$
A\tilde P \ge \tilde b - \xi,\quad \sum_X P(X)=1,\quad \xi \ge 0
$$

여기서 $\xi$는 제약 위반을 허용하는 양이고, $\beta$는 그것에 대한 penalty다. supplementary material에 따르면 dual에서는 이 효과가

$$
0 \le \lambda \le \beta
$$

라는 상한으로 나타난다. 즉, 제약을 절대적으로 강제하지 않고, 너무 어려운 제약은 일부 완화할 수 있게 한다. 논문은 이 slack 덕분에 여러 제약이 충돌할 때도 robust하고, 파라미터에 대한 민감도도 낮아진다고 주장한다.

### 3.7 weak semantic segmentation을 위한 구체적 제약

논문은 semantic segmentation에 맞춰 몇 가지 선형 제약을 정의한다.

첫째는 suppression constraint이다. 이미지에 없는 클래스는 예측되지 않도록 한다.

$$
\sum_{i=1}^n p_i(l) \le 0 \quad \forall l \notin L_I
$$

즉, image-level tag에 없는 클래스 $l$은 전체 픽셀에서 확률 질량이 0이어야 한다.

둘째는 foreground constraint이다. 이미지에 존재하는 클래스는 어느 정도는 나타나야 한다.

$$
a_l \le \sum_{i=1}^n p_i(l) \quad \forall l \in L_I
$$

실험에서는 보통 $a_l = 0.05n$으로 두고 slack weight는 $\beta = 2$를 사용했다. 이는 단순한 MIL의 “적어도 한 픽셀은 positive”보다 더 일반적이다.

셋째는 background constraint이다. 배경 픽셀 비율에 하한과 상한을 둔다.

$$
a_0 \le \sum_{i=1}^n p_i(0) \le b_0
$$

여기서 $l=0$은 background label이다. 논문은 $a_0 = 0.3n$, $b_0 = 0.7n$이 잘 작동했다고 보고한다. 이 제약은 모든 foreground의 총 면적을 간접적으로 조절한다.

넷째는 size constraint이다. 어떤 클래스가 이미지의 10%보다 큰지 작은지라는 1-bit 정보를 이용한다. 큰 클래스는 lower bound를 더 크게 잡아주고, 작은 클래스에는 upper bound를 둔다.

$$
\sum_{i=1}^n p_i(l) \le b_l
$$

작은 객체 클래스에 대해서는 실험상 $b_l < 0.01n$ 정도가 tight한 값보다 약간 나은 성능을 보였다고 한다.

이러한 제약들은 모두 image-level tag를 더 구조적으로 활용하도록 해준다. 단순히 “존재 여부”만 반영하는 것이 아니라, “없는 것은 억제하고, 있는 것은 어느 정도 나오게 하며, 배경 비율도 적절히 유지하라”는 식으로 출력 공간을 구체적으로 제한한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 설정

실험은 PASCAL VOC 2012 semantic segmentation에서 수행되었다. 20개 object class와 background class가 있으며, 학습은 VOC 2012 train set과 Hariharan et al.의 추가 데이터셋을 합친 총 10,582장으로 진행되었다. validation set은 1,449장이다.

평가 지표는 class-wise intersection over union, 즉 IoU 혹은 Jaccard Index다. 각 클래스에 대해 정답과 예측의 겹침 정도를 측정하며, mean IoU가 주요 비교 지표다.

네트워크는 VGG-16 기반 FCN이다. ImageNet 1K로 pretrained된 모델을 사용했고, fully connected layer를 convolution으로 바꿨다. 마지막 fc8은 Pascal VOC의 21개 클래스 출력으로 교체했다. 중요한 점은, 다른 약지도 학습 방법들과 달리 마지막 classifier layer를 ImageNet에서 가져오지 않고 random Gaussian으로 초기화했다는 것이다. 저자들은 CCNN이 이런 랜덤 초기화에도 안정적이라고 강조한다.

학습은 batch size 1, momentum 0.99, initial learning rate $10^{-6}$, 총 60,000 iteration으로 수행되었다. learning rate는 20,000 iteration마다 0.1배로 줄였다. 단일 이미지에 대한 constrained optimization은 CPU single core에서 30ms 미만이라고 한다.

추론 시에는 optional하게 fully connected CRF를 후처리로 적용한다.

### 4.2 image-level tags만 사용한 경우

Table 1에서 VOC 2012 validation set 기준으로 weakly supervised segmentation을 비교했다.

주요 mean IoU는 다음과 같다.

- MIL-FCN: 24.9
- MIL-Base: 17.8
- MIL-Base w/ ILP: 32.6
- EM-Adapt w/o CRF: 32.0
- EM-Adapt: 33.8
- CCNN w/o CRF: 33.3
- CCNN: 35.3

즉, CRF를 포함한 최종 성능에서 CCNN은 35.3 mIoU로 당시 비교 대상 중 가장 높다. CRF 없이도 33.3으로 EM-Adapt w/o CRF의 32.0보다 높다.

클래스별로 보면 bottle, car, cat, dog, motorbike, person, plant, tv 등에서 꽤 강한 수치를 보인다. 예를 들어 person은 40.7, car는 47.1, cat은 48.0, dog는 44.5다. 물론 fully supervised 수준과는 차이가 크지만, image-level tag만으로 이 정도 dense prediction을 학습했다는 점이 핵심이다.

저자들은 기존 MIL 계열 방법이 final classifier initialization에 민감한 반면, CCNN은 arbitrary random initialization도 감당할 수 있다고 주장한다. 이는 실용적으로 의미가 크다. weak supervision에서는 초깃값에 크게 의존하면 재현성과 안정성이 떨어지기 때문이다.

### 4.3 추가 supervision을 넣은 경우

논문은 약간 더 강한 supervision이 들어오면 성능이 얼마나 오르는지도 본다.

첫 번째는 random crop의 image tag다. Papandreou et al.의 EM-Adapt와 비슷하게, 원본 이미지와 segmentation mask의 random crop에서 weak label을 계산하는 방식이다. 이것은 정확한 pixel supervision은 아니지만, 공간적 힌트를 어느 정도 준다.

Table 3에 따르면 validation set에서

- EM-Adapt, tags with random crops: 34.3
- CCNN, tags with random crops: 34.4

이고, CRF 없이 보면

- EM-Adapt: 36.0
- CCNN: 36.4

로 약간 앞선다. 저자들은 이 setting에서는 차이가 크게 나지 않는데, random crop이 사실상 background upper bound와 비슷한 공간적 정보를 제공하기 때문이라고 해석한다.

두 번째는 size supervision이다. 각 클래스에 대해 “이 객체가 이미지의 10% 이상인가 아닌가”라는 1-bit 정보만 추가한다. 이 단순한 정보로 성능이 크게 오른다. Table 3에서는 validation set에서

- CCNN, tags with object sizes: 40.5
- CRF 없이: 42.4

라고 보고한다. 표의 열 이름과 본문 설명 사이에 약간의 표기 혼선이 있지만, 중요한 점은 size bit 하나만 더해도 성능이 크게 향상된다는 것이다.

Table 2의 test set 결과도 이를 보여준다.

- CCNN w/ tags: 35.6
- CCNN w/ size: 43.3
- CCNN w/ size (CRF tuned): 45.1

즉, 단순한 tags-only에서 35.6이던 것이 size bit 추가만으로 43.3까지 오른다. 이는 weak supervision에서도 어떤 형태의 구조적 prior가 들어가면 segmentation이 크게 개선된다는 점을 보여준다.

### 4.4 fully supervised 방법과의 비교

Table 2에는 fully supervised SOTA와도 비교가 있다.

- SDS: 51.6
- FCN-8s: 62.2
- TTIC Zoomout: 64.4
- DeepLab-CRF: 66.4

이에 비해 weakly supervised인 CCNN w/ size (CRF tuned)는 45.1이다. 완전 감독 방식과는 여전히 큰 격차가 있다. 따라서 이 논문은 fully supervised를 대체했다고 보기보다는, weak supervision에서 당시 상당히 강한 baseline을 제시한 것으로 이해하는 것이 정확하다.

### 4.5 bounding box 제약

Discussion에서는 bounding box constraint도 실험한다. 박스 내부 픽셀의 75%가 특정 label을 가지도록 하고, 박스 바깥에서는 그 label을 억제한다. 이때 IoU가 54%까지 오른다고 보고한다. 이는 단순히 box 내부 전체를 해당 클래스라고 학습한 baseline 52.3%보다 높지만, bounding box 안의 segmentation 정보를 더 잘 활용하는 sophisticated method들인 58.5%~62.0% 수준에는 미치지 못한다.

### 4.6 파라미터 민감도와 semi-supervised setting

supplementary material의 ablation에 따르면 제약 파라미터를 바꿔도 평균 정확도 표준편차가 0.73%에 불과하다. 저자들은 이 robustness를 slack variable 덕분이라고 해석한다.

또한 fully supervised image를 일부 추가하는 semi-supervised setting에서도 모델이 추가 supervision을 잘 활용한다고 한다. 다만 제공된 텍스트에는 Figure 5의 수치값이 상세히 적혀 있지 않으므로, 그래프의 정확한 수치 변화는 여기서 단정할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 weak supervision을 매우 일반적인 “선형 제약” 형태로 정식화했다는 점이다. suppression, foreground, background, size, bounding box 등 서로 다른 형태의 supervision을 하나의 틀 안에서 처리할 수 있다. 이는 특정 heuristic에 강하게 묶인 방법보다 훨씬 유연하다.

또 다른 강점은 최적화 설계가 깔끔하다는 점이다. CNN 출력에 직접 제약을 거는 대신, 제약을 만족하는 latent distribution $P$를 도입하고, 네트워크는 그 분포를 따라가게 한다. 이 때문에 복잡한 constrained deep optimization을 convex step과 SGD step의 교대 최적화로 풀 수 있다. supplementary material까지 포함하면, factorization과 dual derivation도 비교적 설득력 있게 제시되어 있다.

실험적으로도 의미가 있다. image-level tag만으로 기존 weakly supervised segmentation보다 높은 성능을 얻었고, object size라는 매우 약한 추가 정보 1-bit만으로 성능이 크게 상승했다. 이는 weak supervision의 질보다 “출력 공간에 대한 구조적 제약”이 중요할 수 있음을 보여준다.

한계도 분명하다. 첫째, fully supervised 방법과의 격차는 여전히 크다. 예를 들어 test set에서 CCNN w/ size (CRF tuned)는 45.1 mIoU인데, DeepLab-CRF는 66.4다. 따라서 실제 high-quality segmentation이 필요한 응용에서는 당시 기준으로도 대체재가 아니라 약지도 학습의 강한 출발점에 가깝다.

둘째, 제약 설계에 task-specific prior가 들어간다. 예를 들어 background 비율을 $30\%$~$70\%$로 두거나, foreground 최소 비율을 $5\%$로 두는 설정은 Pascal VOC에 잘 맞을 수 있지만 다른 도메인에서는 달라질 수 있다. 논문은 slack 덕분에 민감도가 낮다고 보였지만, 그래도 어떤 제약이 유효한지는 문제별 설계가 필요하다.

셋째, 출력 분포를 픽셀 독립 marginals의 곱으로 두는 구조는 계산을 단순화하지만, segmentation의 공간적 구조를 직접 모델링하지는 않는다. 실제 spatial consistency는 주로 후처리 CRF에 의존한다. 즉, 본 방법의 핵심은 학습 시 제약 주입이지, segmentation 자체의 구조적 modeling은 아니다.

넷째, bounding box supervision에서는 더 정교한 방법들보다 낮다. 저자들도 박스 내부 segmentation information을 더 강하게 활용하는 방법들이 더 높은 성능을 낸다고 인정한다. 이는 CCNN의 generality가 강점이지만, 특정 supervision type에 최적화된 specialized method보다 항상 강한 것은 아니라는 뜻이다.

비판적으로 보면, 이 논문은 “weak supervision을 제약으로 표현하는 아이디어”와 “그 제약을 latent distribution으로 다루는 최적화 프레임워크” 자체가 핵심 기여다. 따라서 성능 숫자만이 아니라, 약지도 학습을 다루는 관점의 전환이 더 중요한 논문이라고 볼 수 있다.

## 6. 결론

이 논문은 weakly supervised semantic segmentation을 위해 CCNN이라는 constrained optimization framework를 제안했다. 핵심은 image-level tag나 object size 같은 약한 supervision을 출력 분포에 대한 선형 제약으로 바꾸고, 그 제약을 만족하는 latent distribution $P$를 구한 뒤 CNN 출력 $Q$가 이를 따르도록 학습하는 것이다.

방법론적으로는 KL divergence 기반 objective, dual optimization을 통한 latent distribution 추정, 그리고 SGD와의 교대 최적화가 중심이다. 이 구조 덕분에 arbitrary linear constraints를 CNN 학습에 자연스럽게 통합할 수 있다.

실험적으로는 PASCAL VOC 2012에서 image-level tag만으로도 당시 weakly supervised segmentation의 state of the art를 달성했고, object size에 대한 1-bit 정보만 추가해도 성능이 크게 향상됨을 보였다. 이는 dense annotation이 부족한 환경에서 매우 실용적인 메시지다.

향후 연구 관점에서 이 논문은 중요한 기반 역할을 한다. 약한 supervision을 손실 함수의 ad hoc heuristic으로 넣는 대신, 제약식과 latent target distribution으로 다루는 방식은 이후의 weakly supervised, semi-supervised, constrained learning 연구로 확장될 수 있다. 실제 응용에서도 annotation 비용을 줄이면서 structured prediction 모델을 학습하려는 상황에 유용한 출발점을 제공한다.
