# CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning

- **저자**: Chi Zhang, Guosheng Lin, Fayao Liu, Rui Yao, Chunhua Shen
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1903.02351

## 1. 논문 개요

이 논문은 **few-shot semantic segmentation** 문제를 다룬다. 목표는 새로운 클래스가 주어졌을 때, 그 클래스에 대해 픽셀 단위 라벨이 거의 없는 상황에서도 segmentation을 수행할 수 있는 모델을 만드는 것이다. 일반적인 semantic segmentation 모델은 대규모 pixel-wise annotation이 필요하고, 학습 시 정의된 클래스 집합 밖의 새로운 클래스에는 바로 일반화하기 어렵다. 저자들은 이러한 한계를 해결하기 위해 **class-agnostic segmentation** 관점에서, 적은 수의 support example만으로 query image의 새로운 클래스를 분할하는 방법을 제안한다.

연구 문제는 다음과 같이 정리할 수 있다. 학습 시 본 적 없는 클래스 $c \notin C_{train}$가 테스트 시 등장할 때, support set에 포함된 몇 장의 annotation된 이미지와 query image만 보고 query image 안의 해당 클래스를 segmentation해야 한다. 이는 단순한 이미지 분류보다 훨씬 어렵다. 이유는 segmentation이 이미지 전체에 대한 structured prediction이기 때문이다. 즉, 단지 “이 클래스가 있는가”를 묻는 것이 아니라, **어느 픽셀이 그 클래스에 속하는가**를 정확히 예측해야 한다.

이 문제가 중요한 이유는 명확하다. 실제 응용에서는 새로운 클래스가 계속 등장하며, 클래스가 추가될 때마다 많은 pixel annotation을 다시 수집하는 것은 비용이 매우 크다. 따라서 적은 annotation으로 새로운 객체 클래스를 빠르게 분할할 수 있는 방법은 데이터 효율성과 실제 활용성 측면에서 매우 중요하다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 가지다. 첫째, support image와 query image를 단순히 고수준 category feature로 비교하지 않고, CNN의 **중간 수준(middle-level) feature**를 이용해 query의 각 위치와 support의 foreground representation을 **dense comparison**하는 것이다. 저자들은 lower-level feature는 edge나 color처럼 너무 저수준이고, higher-level feature는 학습 클래스에 특화된 category semantics가 강해 unseen class에 일반화되기 어렵다고 본다. 그래서 unseen class에도 재사용 가능한 object part 수준 표현을 활용하려고 한다.

둘째, 초기 dense comparison 결과만으로는 객체 전체를 정확히 복원하기 어렵기 때문에, 이를 점진적으로 보정하는 **iterative optimization**을 도입한다. 저자들의 관찰에 따르면, dense comparison은 객체 전체를 완전히 잡아내지 못하더라도 적어도 객체의 위치에 대한 강한 단서를 제공한다. 그러므로 이 초기 mask를 반복적으로 refinement하면 더 정교한 segmentation map을 만들 수 있다.

기존 접근과의 차별점도 분명하다. 이전 few-shot segmentation 방법들은 support branch와 query branch가 서로 다른 구조를 가지거나, 1-shot 결과를 단순 평균 또는 OR 연산으로 합치는 경우가 많았다. 반면 이 논문은 두 branch가 backbone을 공유하고, support foreground의 global representation과 query feature map을 직접 비교하는 **dense comparison module (DCM)**을 제안한다. 또한 $k$-shot 설정에서는 단순 평균 대신 **attention mechanism**으로 여러 support example의 기여도를 학습적으로 결정한다.

## 3. 상세 방법 설명

전체 시스템은 크게 **Dense Comparison Module (DCM)**과 **Iterative Optimization Module (IOM)**으로 구성된다. 1-shot 설정을 기준으로 설명하면, support image와 support mask가 하나 주어지고, query image가 입력된다. DCM은 support와 query 사이의 feature comparison을 수행해 초기 segmentation 단서를 만들고, IOM은 이를 반복적으로 refinement한다. 5-shot 같은 $k$-shot 설정에서는 여러 support example의 정보를 attention으로 결합한다.

### Dense Comparison Module

DCM의 첫 부분은 feature extractor이다. backbone으로는 **ResNet-50**을 사용하며, ImageNet pretraining을 이용한다. 저자들은 ResNet의 네 개 block 중 **block2와 block3의 feature를 선택**해 사용한다. block4는 너무 high-level semantic에 치우치고 채널 수도 커 few-shot setting에서 최적화가 어렵다고 본다. 또한 block2 이후에는 **dilated convolution**을 사용하여 spatial resolution을 유지한다. 결과적으로 block2와 block3에서 나온 feature map을 concatenate한 뒤, $3 \times 3$ convolution으로 256차원으로 인코딩한다.

중요한 점은 support image 전체를 대표하는 feature를 만들 때, 단순 global average pooling이 아니라 **foreground area에 대해서만 average pooling**을 수행한다는 것이다. support image에는 배경이나 다른 객체가 있을 수 있기 때문에, support mask를 이용해 foreground에 해당하는 feature만 남기고 평균을 낸다. 구현상으로는 support mask를 feature map 크기로 bilinear downsampling한 뒤, feature와 element-wise multiplication을 수행해 background feature를 0으로 만든다. 이후 foreground 영역에 대한 sum pooling을 하고 foreground area로 나누어 평균 feature vector를 얻는다.

이 support foreground vector는 query feature map의 모든 spatial location에 복제되어 concatenate된다. 이후 convolution block을 통과시키면, query의 각 위치가 support foreground와 얼마나 잘 대응되는지를 반영하는 dense comparison feature가 생성된다. 이 과정은 metric learning의 아이디어를 dense prediction에 맞게 확장한 형태라고 볼 수 있다.

### Iterative Optimization Module

저자들은 dense comparison만으로는 객체의 일부만 잘 찾고 전체 mask를 완성하지 못하는 경우가 많다고 본다. 이를 해결하기 위해 IOM을 사용한다. IOM은 현재 iteration의 feature와 이전 iteration에서 예측된 mask를 받아 더 나은 mask를 만든다.

이전 mask를 단순히 channel-wise concatenation으로 넣으면 첫 iteration에는 mask가 없기 때문에 feature distribution mismatch가 생긴다고 저자들은 지적한다. 그래서 residual form을 사용한다. 핵심 식은 다음과 같다.

$$
M_t = x + F(x, y_{t-1})
$$

여기서 $x$는 DCM의 출력 feature이고, $y_{t-1}$는 이전 iteration의 predicted mask이며, $M_t$는 현재 iteration의 refined feature이다. 함수 $F(\cdot)$는 $x$와 $y_{t-1}$를 concatenate한 뒤, 256개의 필터를 갖는 두 개의 $3 \times 3$ convolution block을 통과시키는 연산이다. 즉, 이전 mask 정보를 residual correction 형태로 feature에 주입한다.

그 위에는 두 개의 일반 residual block을 더 쌓고, 이후 **ASPP (Atrous Spatial Pyramid Pooling)**를 적용한다. ASPP는 서로 다른 receptive field를 통해 multi-scale context를 수집하기 위한 모듈이다. 이 논문에서는 네 개의 병렬 branch를 사용한다.

- atrous rate 6의 $3 \times 3$ convolution
- atrous rate 12의 $3 \times 3$ convolution
- atrous rate 18의 $3 \times 3$ convolution
- image-level global average pooling 후 $1 \times 1$ convolution

이 네 branch의 출력을 concatenate하고, 다시 $1 \times 1$ convolution으로 256채널로 fuse한다. 마지막으로 $1 \times 1$ convolution을 이용해 foreground와 background에 대한 score map을 만든 뒤, softmax를 적용해 confidence map을 얻는다. 이 confidence map은 다음 IOM으로 다시 입력된다. 추론 시에는 초기 prediction 후 **총 4번 refinement**를 수행한다.

훈련 시에는 IOM이 이전 mask에 과적합되는 것을 막기 위해, 이전 epoch에서 얻은 predicted mask를 넣기도 하고 빈 mask를 넣기도 한다. 구체적으로 $y_{t-1}$를 확률 $p_r$로 empty mask로 reset한다. 논문에서는 $p_r = 0.7$을 사용한다. 저자들은 이를 whole-mask dropout의 확장으로 해석한다.

### k-shot을 위한 Attention Mechanism

$k$-shot 설정에서는 여러 support example이 주어진다. 기존 방법은 각 support로부터 얻은 결과를 단순 평균하거나 mask 수준에서 OR 연산을 하는 경우가 많았다. 이 논문은 이를 개선하기 위해 attention을 사용한다.

각 support sample마다 dense comparison 결과와 함께 별도의 attention branch를 둔다. attention branch는 두 개의 convolution block으로 구성된다. 첫 번째는 256개의 $3 \times 3$ filter와 max pooling, 두 번째는 하나의 $3 \times 3$ convolution과 global average pooling으로 이루어진다. 이 branch의 출력이 해당 support sample의 weight $\lambda_i$가 된다. 이후 softmax로 정규화한다.

$$
\hat{\lambda}_i = \frac{e^{\lambda_i}}{\sum_{j=1}^{k} e^{\lambda_j}}
$$

최종적으로 서로 다른 support sample에서 나온 feature를 $\hat{\lambda}_i$를 가중치로 하여 weighted sum한다. 즉, 모델이 어떤 support example이 현재 query를 segmentation하는 데 더 유용한지 스스로 학습한다.

### Bounding Box Annotation 설정

저자들은 support set annotation 비용을 더 줄이기 위해, support에 pixel-wise mask 대신 **bounding box annotation**만 주는 설정도 실험한다. 이 경우 bounding box 내부 전체를 foreground로 간주한다. 엄밀한 mask 대신 box를 사용하면 support feature 안에 배경 잡음이 더 포함되지만, 저자들은 DCM이 이런 잡음에 어느 정도 견딜 수 있는지 확인하고자 했다.

### 학습 절차와 목적 함수

논문에 따르면 전체 네트워크는 end-to-end로 학습된다. 학습은 **episodic paradigm**을 따른다. 각 episode는 support set $S = \{(x_s^i, y_s^i(c))\}_{i=1}^k$와 query pair $(x_q, y_q(c))$로 구성된다. 여기서 $y_s^i(c)$와 $y_q(c)$는 클래스 $c$에 대한 binary mask이다.

손실 함수는 output map의 모든 spatial location에서 계산한 **cross-entropy loss의 평균**이다. 최적화는 SGD를 사용하며, PASCAL-5$^i$에서는 mini-batch 4 episode, COCO subset에서는 mini-batch 8 episode를 사용한다. 학습 epoch는 200, learning rate는 0.0025이다.

## 4. 실험 및 결과

### 평가 설정과 지표

논문은 PASCAL-5$^i$와 COCO subset에서 실험한다. PASCAL-5$^i$는 PASCAL VOC 2012의 20개 클래스를 4개 split으로 나누고, 각 실험에서 3개 split으로 학습하고 1개 split으로 테스트한다. 테스트 시에는 1000개의 support-query pair를 랜덤 샘플링한다.

평가 지표는 두 가지를 다룬다. 하나는 **meanIoU**, 다른 하나는 **FB-IoU**이다. 저자들은 meanIoU를 주 지표로 선호한다. 이유는 클래스별 sample 수 불균형이 크고, background IoU는 객체를 놓쳐도 높게 나올 수 있어 segmentation 성능을 제대로 반영하지 못할 수 있기 때문이다. 이 선택은 few-shot binary segmentation의 본질을 생각하면 타당한 주장이다.

### PASCAL-5i 결과

PASCAL-5$^i$에서 제안 방법 CANet은 매우 큰 성능 향상을 보인다.

meanIoU 기준으로:

- 1-shot: **55.4%**
- 5-shot: **57.1%**

비교 대상 OSLSM은 각각 40.8%, 43.9%이므로, CANet은 1-shot에서 **14.6%p**, 5-shot에서 **13.2%p** 향상된다. few-shot segmentation 분야에서는 상당히 큰 개선폭이다.

FB-IoU 기준으로도:

- 1-shot: **66.2%**
- 5-shot: **69.6%**

으로, 기존 방법들인 OSLSM, co-FCN, PL보다 모두 높다. 논문이 특히 강조하는 부분은, 제안한 방법이 단지 특정 split에서만 좋은 것이 아니라 4개 split 전반에서 일관되게 우수하다는 점이다.

정성적 결과에서는 같은 query image에 대해 support example을 바꾸면 서로 다른 클래스를 올바르게 segment하는 사례를 보여준다. 이는 모델이 query image 자체만 보고 fixed category prediction을 하는 것이 아니라, 실제로 support-conditioned segmentation을 수행하고 있음을 뒷받침한다.

### Bounding Box Annotation 결과

support set에 pixel-wise mask 대신 bounding box만 사용할 경우 meanIoU는 다음과 같다.

- pixel-wise label: **54.0%**
- bounding box: **52.0%**

성능 하락은 2.0%p에 그친다. 즉, support annotation을 크게 단순화해도 성능이 비교적 잘 유지된다. 이는 support box 안의 배경 노이즈가 있더라도 foreground global representation이 어느 정도 유효하게 작동함을 보여준다.

### Ablation Study

#### 비교 feature 선택

ResNet의 어떤 block feature를 사용할지에 대한 ablation에서, 단일 block만 사용할 경우 block3가 가장 좋았고, 여러 block을 조합할 경우 **block2 + block3**가 최고 성능인 **51.2% meanIoU**를 기록했다. block4를 포함하면 오히려 성능이 떨어졌다. 이는 저자들의 주장대로, unseen class 일반화에는 중간 수준 representation이 더 적합하다는 실험적 근거가 된다.

또한 backbone을 VGG16으로 바꿔도 multi-scale evaluation에서 **54.3%**를 기록했다. ResNet50 기반 55.4%보다 1.1%p 낮지만, 여전히 기존 SOTA를 크게 앞선다. 즉, 핵심 기여는 backbone 자체보다는 제안 구조에 있음을 시사한다.

#### Iterative Optimization Module 효과

IOM의 효과를 보기 위해 초기 prediction만 사용하는 CANet-Init과 DenseCRF 후처리를 비교했다.

- CANet-Init: **51.2%**
- CANet-Init + DenseCRF: **51.9%**
- CANet: **54.0%**

즉, IOM은 초기 prediction 대비 **2.8%p** 향상된다. DenseCRF는 경계 refinement에는 도움이 될 수 있지만, 위치를 잘못 잡은 경우 false positive를 더 확장시키는 문제가 있다고 저자들은 설명한다. 반면 IOM은 학습 가능한 방식으로 객체 영역을 채우고 불필요한 영역을 제거할 수 있다.

#### 5-shot fusion 방식 비교

5-shot에서 여러 support를 합치는 방법 비교 결과는 다음과 같다.

- 1-shot baseline: 54.0%
- Feature average: 55.0%
- Mask average: 54.5%
- Mask OR: 53.4%
- Attention: **55.8%**

attention이 가장 좋고, 1-shot 대비 **1.8%p** 향상을 보인다. 특히 mask OR는 오히려 성능을 떨어뜨렸다. 이는 단순 rule-based fusion보다 support quality를 학습적으로 구분하는 방식이 더 효과적임을 보여준다.

#### Multi-scale evaluation

query image를 $[0.7, 1.0, 1.3]$ 배율로 추론한 뒤 평균하는 multi-scale evaluation도 실험했다. PASCAL-5$^i$에서 1-shot은 **1.4%p**, 5-shot은 **1.3%p** 추가 향상이 있었다.

### COCO 결과

COCO에서는 전체 80개 클래스를 모두 쓰지 않고, 40 train / 20 validation / 20 test의 subset을 구성해 실험했다. 이는 계산 비용 문제 때문이라고 논문에 명시되어 있다.

1-shot COCO 결과는 다음과 같다.

- CANet-Init: **42.2%**
- CANet: **46.3%**
- CANet + multi-scale: **49.9%**

여기서도 IOM이 **4.1%p** 개선을 보이며, multi-scale evaluation이 추가로 **3.3%p** 향상시킨다.

5-shot COCO 결과는 다음과 같다.

- Feature-Avg: 48.9%
- Mask-Avg: 49.2%
- Mask-OR: 46.2%
- Attention: **49.7%**
- Attention + multi-scale: **51.6%**

COCO처럼 더 어려운 데이터셋에서도 attention 기반 fusion이 가장 높은 성능을 보인다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot segmentation을 위해 설계한 구조가 매우 명확하고, 각 구성 요소의 역할이 잘 분리되어 있다는 점이다. DCM은 support-conditioned dense matching을 담당하고, IOM은 초기 localization 단서를 정교한 mask로 바꾸는 refinement를 담당한다. 또한 $k$-shot attention까지 포함해 1-shot과 5-shot 모두에 대해 일관된 개선을 보인다. 무엇보다 PASCAL-5$^i$에서 기존 방법 대비 10%p가 넘는 큰 향상은 실질적인 기여로 볼 수 있다.

또 다른 강점은 **bounding box support** 설정을 실험했다는 점이다. 많은 few-shot segmentation 논문이 pixel-accurate support mask를 전제로 하지만, 이 논문은 annotation cost라는 실제 문제를 더 밀어붙여 생각했다. box annotation만으로도 성능 저하가 비교적 작다는 결과는 응용 가능성을 높인다.

한편 한계도 있다. 첫째, support representation이 foreground global average pooled vector 하나에 크게 의존하기 때문에, support object 내부의 복잡한 part structure나 multiple modes를 충분히 표현하지 못할 수 있다. 논문도 dense comparison만으로는 객체 일부만 매칭되는 문제가 있다고 인정하고, 이를 IOM으로 보완한다. 즉, 초기 매칭 자체의 한계가 구조 안에 내재해 있다.

둘째, 방법은 class-agnostic generalization을 목표로 하지만, 실제 성능 검증은 여전히 PASCAL-5$^i$와 제한된 COCO subset에 기반한다. 특히 COCO 실험은 원본 전체가 아니라 subset이다. 따라서 더 대규모이거나 분포가 다른 환경에서 어느 정도 일반화되는지는 이 논문만으로는 충분히 판단하기 어렵다.

셋째, refinement를 4회 반복하는 inference 구조는 성능 향상에 기여하지만, 그만큼 계산량과 추론 복잡성이 증가할 수 있다. 논문은 runtime이나 실제 latency 분석은 제공하지 않는다. 따라서 실시간 응용 가능성은 본문만으로는 평가하기 어렵다.

넷째, 논문은 support box annotation의 가능성을 보여주지만, weak annotation의 종류를 더 다양하게 탐색하지는 않는다. 예를 들어 scribble, point, extreme point와 같은 annotation에 대해서는 이 논문 안에서 실험적 검증이 없다. 따라서 “annotation cost를 얼마나 더 줄일 수 있는가”는 아직 열린 질문으로 남는다.

비판적으로 보면, 이 논문의 핵심 개선은 매우 효과적이지만, support-query 관계를 더욱 구조적으로 모델링하는 방향, 예를 들어 part-level alignment나 spatial correspondence를 명시적으로 다루는 방향까지는 가지 못했다. 그럼에도 당시 few-shot segmentation 맥락에서는 단순하면서도 강력한 설계로 높은 성능을 달성했다는 점이 중요하다.

## 6. 결론

이 논문은 few-shot semantic segmentation을 위해 **CANet**이라는 class-agnostic segmentation framework를 제안한다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, support foreground와 query feature를 중간 수준 representation에서 조밀하게 비교하는 **Dense Comparison Module**을 설계했다. 둘째, 초기 segmentation 결과를 반복적으로 개선하는 **Iterative Optimization Module**을 도입해 mask 품질을 높였다. 셋째, $k$-shot 상황에서 여러 support example을 효과적으로 결합하기 위해 **attention mechanism**을 제안했다.

실험 결과는 제안 방법의 효과를 강하게 뒷받침한다. PASCAL-5$^i$에서 1-shot 55.4%, 5-shot 57.1% meanIoU를 기록하며 기존 방법을 큰 폭으로 넘어섰고, COCO subset에서도 같은 경향을 보였다. 또한 support에 bounding box만 사용해도 성능이 크게 무너지지 않는다는 점은 실제 annotation 비용 절감 측면에서 의미가 있다.

종합하면, 이 연구는 few-shot segmentation에서 단순한 metric-style matching을 넘어서, **dense matching + learnable refinement + support fusion**이라는 조합이 매우 효과적임을 보여준 작업이다. 이후 연구에서는 보다 정교한 correspondence modeling, stronger backbone, transformer 기반 matching 등이 등장하겠지만, 이 논문은 few-shot segmentation의 초기 발전 과정에서 매우 중요한 기준점을 제공한 것으로 볼 수 있다.
