# PANet: Few-Shot Image Semantic Segmentation with Prototype Alignment

- **저자**: Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, Jiashi Feng
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/1908.06391

## 1. 논문 개요

이 논문은 **few-shot semantic segmentation** 문제를 다룬다. 즉, 새로운 클래스에 대해 픽셀 단위 정답이 거의 없는 상황에서도, 몇 장의 annotated support image만 보고 query image를 정확히 분할하는 모델을 만드는 것이 목표다. 기존 semantic segmentation 모델은 대량의 dense annotation을 필요로 하고, 학습 중 보지 못한 새로운 클래스에 대한 일반화 성능이 약하다는 문제가 있다. 이 논문은 바로 그 지점을 겨냥한다.

저자들은 이 문제를 기존의 “support 정보를 네트워크 내부 파라미터로 녹여 넣는 방식”이 아니라, **metric learning 관점**에서 다시 본다. support set에서 각 클래스의 대표 벡터인 **prototype**을 만들고, query image의 각 픽셀이 어떤 prototype에 가장 가까운지를 계산해 segmentation을 수행한다. 핵심은 segmentation을 복잡한 parametric decoder에 맡기지 않고, embedding space 안에서의 **비모수적(non-parametric) 거리 기반 분류**로 푼다는 점이다.

이 문제가 중요한 이유는 분명하다. 실제 환경에서는 모든 새로운 클래스마다 충분한 pixel-wise annotation을 수집하기 어렵다. 따라서 적은 예시만으로도 새로운 객체를 분할할 수 있는 방법은 데이터 효율성과 실제 적용성 측면에서 매우 중요하다. 논문은 이 문제에 대해 단순하지만 강한 구조를 제시하고, 특히 support 정보를 더 잘 활용하기 위한 **Prototype Alignment Regularization (PAR)** 를 추가해 성능을 끌어올린다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 support image로부터 각 클래스의 **compact하고 discriminative한 prototype**을 만들고, query image의 각 위치를 이 prototype들과 직접 비교해서 분할하는 것이다. 다시 말해, “이 픽셀은 어떤 클래스의 대표 표현과 가장 닮았는가?”를 계산하는 방식이다. 이는 Prototypical Networks를 dense prediction 문제로 확장한 형태라고 볼 수 있다.

기존 few-shot segmentation 방법들은 대체로 support와 query feature를 합치고, 그 위에 별도의 segmentation module 또는 decoder를 얹어 결과를 낸다. 저자들은 이런 방식이 두 가지 한계를 가진다고 본다. 첫째, support에서 지식을 뽑아내는 과정과 query를 분할하는 과정이 뒤섞여 있어 일반화가 약해질 수 있다. 둘째, support annotation은 종종 단순 masking 용도로만 쓰이고, “학습을 직접 감독하는 신호”로 충분히 활용되지 않는다.

이를 해결하기 위해 PANet은 두 가지를 분리한다. 하나는 **prototype extraction**, 다른 하나는 **non-parametric metric learning**이다. 이렇게 하면 support 정보가 class-specific representation으로 정리되고, segmentation은 그 representation과 query pixel 간 거리 계산으로 수행된다. 여기에 더해 저자들은 **PAR**를 도입한다. query를 예측한 뒤, 그 예측 mask를 이용해 query 쪽 prototype을 다시 만들고, 이번에는 반대로 support를 segment하게 하여 support-query prototype이 더 잘 정렬되도록 유도한다. 이 역방향 학습이 PANet의 가장 중요한 차별점이다.

## 3. 상세 방법 설명

### 3.1 문제 설정

훈련 클래스 집합 $C_{seen}$ 과 테스트 클래스 집합 $C_{unseen}$ 은 겹치지 않는다. 훈련과 테스트는 모두 여러 개의 **episode**로 구성되며, 각 episode는 support set $S_i$ 와 query set $Q_i$ 를 포함한다. 각 episode는 $C$-way $K$-shot segmentation task를 이룬다. support에는 각 클래스당 $K$개의 이미지-마스크 쌍이 있고, query는 같은 클래스 집합에 속한 이미지들로 이루어진다.

모델은 support에서 클래스별 prototype을 추출한 뒤, 이를 이용해 query를 분할한다. 훈련 과정 전체가 episode 기반으로 진행되므로, 모델은 특정 클래스에 과적합되기보다 “적은 예시로 새로운 클래스를 빠르게 분할하는 방법”을 배우게 된다.

### 3.2 전체 파이프라인

PANet의 흐름은 다음과 같다.

먼저 support image와 query image를 **shared backbone**으로 feature map으로 바꾼다. 논문에서는 VGG-16을 backbone으로 사용했다. 첫 5개 convolution block을 유지하고, `maxpool4`의 stride를 1로 바꾸어 해상도 손실을 줄였으며, `conv5`는 dilation 2의 dilated convolution으로 바꿔 receptive field를 넓혔다.

그 다음 support feature map에 대해 mask를 적용하여 foreground class prototype과 background prototype을 계산한다. 이후 query feature map의 각 spatial location에서 각 prototype까지의 거리를 계산하고, softmax를 통해 각 위치의 클래스 확률을 얻는다. 결국 각 픽셀은 가장 가까운 prototype의 클래스로 할당된다.

이 구조의 장점은 분명하다. decoder나 추가 refinement 모듈이 없고, prototype 계산과 예측이 모두 feature map 위에서 이루어지므로 구조가 단순하다. 또한 PAR은 학습 시에만 사용되므로, 추론 비용은 늘어나지 않는다.

### 3.3 Prototype learning

prototype은 support feature map에서 mask에 해당하는 위치들의 평균으로 계산된다. 저자들은 **late fusion**을 채택한다. 즉, 이미지를 먼저 backbone에 통과시켜 feature map을 만든 뒤, 그 위에서 mask를 적용한다. 이는 입력 단계에서 이미지를 잘라내는 early fusion보다 shared feature extractor의 입력 일관성을 더 잘 유지한다고 본다.

클래스 $c$의 prototype은 다음과 같이 계산된다.

$$
p_c = \frac{1}{K}\sum_k \frac{\sum_{x,y} F^{(x,y)}_{c,k}\mathbf{1}[M^{(x,y)}_{c,k}=c]}{\sum_{x,y}\mathbf{1}[M^{(x,y)}_{c,k}=c]}
$$

여기서 $F_{c,k}^{(x,y)}$ 는 support image의 feature map에서 위치 $(x,y)$의 feature vector이고, $\mathbf{1}[\cdot]$ 는 indicator function이다. 식의 의미는 간단하다. 각 support image에서 해당 클래스 마스크 내부 feature들을 평균내고, 다시 $K$개 support에 대해 평균을 낸 것이다.

background prototype은 episode에 포함된 클래스 집합 $C_i$ 에 속하지 않는 위치들을 모아 평균내어 만든다.

$$
p_{bg} = \frac{1}{CK}\sum_{c,k}\frac{\sum_{x,y} F^{(x,y)}_{c,k}\mathbf{1}[M^{(x,y)}_{c,k}\notin C_i]}{\sum_{x,y}\mathbf{1}[M^{(x,y)}_{c,k}\notin C_i]}
$$

즉, foreground뿐 아니라 background도 하나의 prototype으로 명시적으로 모델링한다.

### 3.4 Non-parametric metric learning

query segmentation은 각 위치 feature와 prototype 간의 거리 비교로 이루어진다. query feature map을 $F_q$, prototype 집합을 $P=\{p_c \mid c\in C_i\}\cup\{p_{bg}\}$ 라고 하면, 위치 $(x,y)$에서 클래스 $j$일 확률은 다음과 같다.

$$
\tilde{M}^{(x,y)}_{q;j} =
\frac{\exp(-\alpha d(F^{(x,y)}_q, p_j))}
{\sum_{p_j\in P}\exp(-\alpha d(F^{(x,y)}_q, p_j))}
$$

여기서 $d$는 거리 함수이고, 논문은 cosine distance를 사용한다. 저자들은 squared Euclidean distance보다 cosine distance가 더 안정적이고 성능도 더 좋았다고 보고한다. $\alpha$는 softmax에 들어가기 전 거리값 스케일을 조절하는 상수이며, 실험적으로 $20$으로 고정했다.

최종 예측 마스크는 가장 높은 확률을 갖는 클래스를 고르는 방식이다.

$$
\hat{M}^{(x,y)}_q = \arg\max_j \tilde{M}^{(x,y)}_{q;j}
$$

학습 손실은 query에 대한 pixel-wise cross-entropy loss이다.

$$
L_{seg} =
-\frac{1}{N}\sum_{x,y}\sum_{p_j\in P}
\mathbf{1}[M^{(x,y)}_q=j]\log \tilde{M}^{(x,y)}_{q;j}
$$

여기서 $N$은 spatial location 수이다. 결국 이 손실을 줄이도록 backbone이 학습되면, embedding space 안에서 같은 클래스는 가깝고 다른 클래스는 멀어지는 방향으로 prototype이 정리된다.

### 3.5 Prototype Alignment Regularization (PAR)

이 논문의 가장 중요한 추가 설계는 PAR이다. 아이디어는 다음과 같다. support에서 만든 prototype으로 query를 잘 분할했다면, query의 예측 mask를 사용해 다시 만든 prototype도 support를 잘 분할할 수 있어야 한다. 즉, support-derived prototype과 query-derived prototype이 같은 embedding space 안에서 일관되게 정렬되어야 한다.

구체적으로는 다음 순서다.

먼저 support로부터 prototype을 만들고 query를 분할한다. 그다음 query feature와 **예측된 query mask**를 이용해 새로운 prototype 집합 $\bar{P}$를 만든다. 그리고 이 query-derived prototype으로 원래 support image들을 다시 분할한다. 이 역방향 segmentation 결과를 support ground truth와 비교해 추가 손실 $L_{PAR}$를 계산한다.

논문이 제시한 support 분할 확률은 다음과 같다.

$$
\tilde{M}^{(x,y)}_{c,k;j} =
\frac{\exp(-\alpha d(F^{(x,y)}_{c,k}, \bar{p}_j))}
{\sum_{\bar{p}_j\in\{\bar{p}_c,\bar{p}_{bg}\}}\exp(-\alpha d(F^{(x,y)}_{c,k}, \bar{p}_j))}
$$

최종 손실은

$$
L = L_{seg} + \lambda L_{PAR}
$$

로 주어진다. 논문에서는 $\lambda=1$을 사용했고, 다른 값도 큰 차이를 만들지 않았다고 한다.

여기서 주의할 점이 있다. 논문에 인용된 Eq. (7)의 표기는 문맥상 약간 어색하다. 설명상으로는 “query-derived prototype으로 support를 분할한 결과를 support ground truth와 비교”해야 하는데, 식 표기에는 query mask 기호가 반복되어 있다. 따라서 보고서 기준으로는 **문맥상 support에 대한 cross-entropy loss를 뜻하는 것으로 해석하는 것이 타당하지만, 식 표기 자체는 본문에서 다소 불일치한다**고 보는 것이 안전하다.

### 3.6 약한 주석으로의 일반화

PANet은 dense mask뿐 아니라 scribble, bounding box 같은 약한 annotation에도 바로 적용 가능하다고 주장한다. 그 이유는 prototype 계산이 “주어진 annotation 영역의 평균 feature”라는 단순한 형식이기 때문이다. annotation이 완전하지 않더라도 대표 feature를 뽑아낼 수 있으면 동작한다는 것이다.

또한 late fusion 구조이기 때문에 annotation이 갱신되었을 때 backbone을 다시 통과할 필요 없이 feature map 위에서 바로 반영할 수 있어, interactive segmentation 가능성도 언급한다. 다만 이 부분은 미래 연구 방향으로만 제시되고 실제 실험은 포함되지 않는다.

## 4. 실험 및 결과

### 4.1 실험 설정

주요 평가는 **PASCAL-5$^i$** 와 **MS COCO**에서 이루어졌다. PASCAL-5$^i$는 PASCAL VOC 2012와 SBD augmentation을 기반으로 하며, 20개 클래스를 4개 split으로 나누고, 3개 split으로 학습한 뒤 남은 1개 split으로 평가하는 cross-validation 방식을 사용한다. MS COCO 역시 80개 클래스를 4개 split으로 나누어 같은 프로토콜을 따른다.

평가 지표는 두 가지다. **mean-IoU**는 foreground 클래스별 IoU를 구해 평균한 값이고, **binary-IoU**는 모든 foreground 클래스를 하나로 합쳐 foreground와 background의 IoU를 계산한 값이다. 저자들은 클래스 구분 능력을 더 잘 반영하는 mean-IoU를 더 중요한 지표로 본다.

구현 측면에서 backbone은 ImageNet 사전학습 VGG-16을 사용했고, 입력 이미지는 $(417,417)$로 리사이즈했다. 학습은 SGD, momentum 0.9, 총 30,000 iteration, 초기 learning rate $10^{-3}$, 10,000 iteration마다 0.1배 감소, weight decay 0.0005, batch size 1로 설정했다.

### 4.2 PASCAL-5$^i$ 결과

PASCAL-5$^i$에서 PANet은 1-way few-shot segmentation 기준으로 다음 성능을 보였다.

- 1-shot mean-IoU: **48.1%**
- 5-shot mean-IoU: **55.7%**

이는 논문이 비교한 기존 최고 성능인 SG-One의 46.3%, 47.1%보다 높다. 특히 5-shot에서는 **8.6%p**라는 큰 차이로 앞선다. binary-IoU 기준으로도 PANet은 1-shot **66.5%**, 5-shot **70.7%**로 최고 성능을 기록했다.

흥미로운 점은 **shot 수 증가에 따른 성능 향상 폭**이다. 기존 방법들은 1-shot에서 5-shot으로 가도 mean-IoU 향상이 3.1%p 이하였는데, PANet은 **7.6%p** 향상되었다. 저자들은 이를 support 정보를 더 효과적으로 활용했기 때문이라고 해석한다. 즉, support example이 늘어날수록 prototype 품질이 좋아지고, metric learning 구조가 그 이점을 실제 성능 향상으로 잘 연결한다는 주장이다.

또한 파라미터 수도 적다. 표에 따르면 PANet은 **14.7M parameters**로, OSLSM의 272.6M, co-FCN의 34.2M보다 훨씬 작다. 단순한 구조가 성능뿐 아니라 모델 크기 측면에서도 이점이 있음을 보여준다.

### 4.3 Multi-way few-shot 결과

2-way few-shot segmentation에서도 PANet은 큰 폭으로 앞선다. PASCAL-5$^i$에서 2-way 1-shot, 2-way 5-shot mean-IoU는 각각 **45.1%**, **53.1%**로 보고되었다. 비교 대상으로 제시된 PL [4]의 binary-IoU는 42.7%, 43.7% 수준이며, 논문은 PANet이 multi-way setting에서도 20% 이상 큰 개선을 보인다고 서술한다. 이 결과는 단순히 “foreground vs background”만 잘 나누는 것이 아니라, 여러 novel foreground class 사이의 구분까지 상대적으로 잘 수행함을 시사한다.

### 4.4 MS COCO 결과

MS COCO는 클래스 수가 더 많고 더 어렵다. 이 데이터셋에서 PANet은 1-way 기준

- 1-shot mean-IoU: **20.9%**
- 5-shot mean-IoU: **29.7%**
- 1-shot binary-IoU: **59.2%**
- 5-shot binary-IoU: **63.5%**

를 기록했다. 비교 대상 A-MCG [8] 대비 binary-IoU 기준으로 1-shot에서 **7.2%p**, 5-shot에서 **8.2%p** 개선되었다고 논문은 설명한다. 절대 성능 자체는 PASCAL보다 낮지만, 더 많은 클래스와 더 높은 복잡성을 가진 COCO에서 일관되게 개선을 보였다는 점이 중요하다.

### 4.5 PAR 분석

PAR의 효과는 별도 분석으로 제시된다. PASCAL-5$^i$에서 PANet without PAR는 1-shot **47.2%**, 5-shot **54.9%**였고, PAR를 넣은 PANet은 1-shot **48.1%**, 5-shot **55.7%**였다. 절대 향상폭은 크지 않지만 일관되게 좋아진다.

저자들은 PAR이 실제로 prototype alignment를 개선했는지도 확인했다. 1-way 5-shot, split-1의 1,000개 episode를 대상으로 support prototype과 query prototype 간 Euclidean distance를 재보니, PAR이 있을 때 평균 거리가 **32.2**, 없을 때 **42.6**이었다. 즉, embedding space에서 support와 query의 prototype이 더 가깝게 정렬되었다는 것이다.

또 하나의 관찰은 **학습 수렴 속도**다. 논문은 training loss curve를 통해 PAR이 없는 경우보다 있는 경우가 더 빨리 수렴하고 더 낮은 loss에 도달했다고 말한다. 이는 support 정보가 단순 masking을 넘어 실제 regularization signal로 사용되기 때문으로 해석할 수 있다.

### 4.6 약한 annotation 실험

support annotation을 dense mask 대신 scribble이나 bounding box로 바꿔도 PANet은 비교적 잘 동작했다.

- Dense: 1-shot **48.1%**, 5-shot **55.7%**
- Scribble: 1-shot **44.8%**, 5-shot **54.6%**
- Bounding box: 1-shot **45.1%**, 5-shot **52.8%**

dense annotation이 가장 좋기는 하지만, 성능 저하는 생각보다 크지 않다. 특히 5-shot에서 scribble은 dense와 1.1%p 차이밖에 나지 않는다. bounding box는 noise가 더 많아 5-shot에서 scribble보다 2%p 낮다. 이는 prototype 기반 구조가 annotation noise에 어느 정도 견고하다는 점을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 방법이 **단순하면서도 설계 논리가 명확하다**는 점이다. support에서 class prototype을 만들고 query pixel을 prototype과 비교한다는 구조는 few-shot 문제의 본질과 잘 맞는다. support-query feature를 복잡하게 합치거나 별도 decoder를 두지 않아도 성능이 높고, 파라미터 수도 적다. 실험 결과도 이 주장을 잘 뒷받침한다.

두 번째 강점은 **support 정보를 더 잘 활용하는 방식**을 제시했다는 것이다. 기존 방법에서 support annotation은 종종 단순 masking 용도였는데, PANet은 PAR을 통해 support와 query 사이의 embedding consistency를 학습하도록 만든다. 이 아이디어는 구조적으로도 자연스럽고, 실제로 alignment distance 감소와 성능 향상으로 연결된다.

세 번째 강점은 **weak annotation에 대한 확장성**이다. dense mask가 아니어도 support에서 prototype을 뽑을 수 있기 때문에 scribble, bounding box에도 적용 가능하다. 이는 실제 annotation 비용을 줄일 수 있다는 점에서 실용적 의미가 있다.

반면 한계도 분명하다. 먼저, 논문 스스로 qualitative failure case에서 인정하듯이, 이 모델은 각 위치를 비교적 독립적으로 예측하므로 **unnatural patches** 같은 결과가 생길 수 있다. 즉, spatial consistency를 강하게 보장하는 구조가 없다. decoder나 CRF 같은 refinement가 없다는 점이 장점이기도 하지만, 동시에 한계이기도 하다.

또한 prototype 하나로 클래스를 대표시키는 구조는 클래스 내부 변이가 큰 경우 불리할 수 있다. 논문도 chair와 table처럼 비슷한 appearance를 가진 객체를 구분하지 못하는 사례를 제시한다. 이는 **single prototype representation의 표현력 한계**를 시사한다.

실험 해석에서도 주의할 점이 있다. 논문은 1,000개 episode만으로는 평가가 불안정하다고 보고 5개 random seed 평균을 사용했는데, 이는 오히려 좋은 실험 태도이지만 동시에 few-shot segmentation 평가가 **sampling variance에 민감**하다는 뜻이기도 하다. 따라서 절대적 수치 차이는 episode sampling 방식에 어느 정도 영향받을 수 있다.

마지막으로, 본문에서 Eq. (7)의 표기는 설명과 완전히 깔끔하게 맞아떨어지지 않는다. 핵심 아이디어 이해에는 큰 문제가 없지만, 수식 수준의 엄밀함은 다소 부족해 보인다. 또한 interactive segmentation 가능성은 언급만 있고 실험 검증은 없다.

## 6. 결론

이 논문은 few-shot semantic segmentation을 위해 **prototype-based metric learning**을 전면에 내세운 PANet을 제안했다. support set으로부터 foreground와 background prototype을 계산하고, query의 각 픽셀을 prototype과의 거리로 분류하는 단순한 구조를 사용한다. 여기에 **Prototype Alignment Regularization**을 더해 support-query 간 embedding consistency를 강화함으로써 일반화 성능을 높였다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, 복잡한 parametric decoder 없이도 강한 few-shot segmentation 성능을 내는 단순하고 효과적인 설계를 제시했다. 둘째, PAR을 통해 support 정보를 학습 과정에서 더 적극적으로 활용하는 새로운 regularization을 도입했다. 셋째, dense annotation뿐 아니라 scribble, bounding box 같은 약한 annotation에도 비교적 잘 일반화되는 가능성을 보여주었다.

실제 적용 측면에서도 의미가 있다. 새로운 객체 클래스를 빠르게 분할해야 하지만 충분한 pixel annotation이 없는 환경, 또는 annotation 비용을 줄여야 하는 환경에서 유용할 가능성이 크다. 향후 연구로는 multiple prototypes, spatial refinement, stronger backbone, 그리고 interactive setting 검증 같은 방향이 자연스럽게 이어질 것이다. 전체적으로 이 논문은 few-shot segmentation을 보다 직접적이고 해석 가능한 방식으로 푼 대표적인 prototype 기반 접근으로 볼 수 있다.
