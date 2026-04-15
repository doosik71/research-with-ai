# Adaptive Prototype Learning and Allocation for Few-Shot Segmentation

- **저자**: Gen Li, Varun Jampani, Laura Sevilla-Lara, Deqing Sun, Jonghyun Kim, Joongkyu Kim
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2104.01893

## 1. 논문 개요

이 논문은 few-shot segmentation에서 support image 몇 장만 보고 query image의 새로운 클래스 객체를 분할하는 문제를 다룬다. 핵심 문제의식은 기존 prototype 기반 방법이 보통 support object를 하나의 prototype으로 평균 내어 표현한다는 점이다. 이런 방식은 계산은 단순하지만, 물체의 다양한 part, scale, shape, appearance variation을 하나의 벡터에 압축해버리기 때문에 표현력이 부족해질 수 있다. 특히 support와 query 사이에 자세 변화, 가림(occlusion), 크기 차이가 크면 single prototype은 어떤 부분을 대표하는지 모호해진다.

논문은 이 한계를 해결하기 위해 object 전체를 하나의 평균 벡터로 표현하지 않고, support object 내부를 feature similarity에 따라 여러 개의 representative region으로 나눈 뒤, 각 region에서 prototype을 만들고, query의 각 위치마다 가장 적절한 prototype을 다르게 할당하는 방법을 제안한다. 즉, prototype의 개수 자체도 고정하지 않고 object 크기에 따라 바꾸며, query pixel마다 참조하는 prototype도 다르게 한다.

이 문제는 중요하다. few-shot segmentation에서는 테스트 시 학습에 없던 새로운 클래스를 처리해야 하므로, 특정 클래스에 강하게 맞춰진 복잡한 matching보다는 일반화 가능한 representation이 필요하다. 기존 affinity learning 방식은 spatial information은 잘 보존하지만 dense matching 때문에 overfitting 위험과 계산량 부담이 있다. 반면 prototype learning은 더 간결하고 강건하지만 지나친 정보 압축이 문제다. 이 논문은 그 중간 지점을 노린다. 즉, prototype learning의 효율성과 robustness를 유지하면서도, single prototype의 정보 손실 문제를 완화하려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 가지다. 첫째, support object를 하나의 prototype으로 요약하지 말고, superpixel-like clustering을 통해 여러 개의 content-adaptive prototype으로 표현하자는 것이다. 둘째, query image의 모든 위치에 동일한 prototype을 복제해서 붙이지 말고, 각 위치에서 가장 잘 맞는 prototype을 선택적으로 배정하자는 것이다.

이를 위해 저자들은 두 모듈을 제안한다.

첫 번째는 **Superpixel-guided Clustering (SGC)** 이다. support feature map에서 foreground mask 내부의 feature만 남긴 뒤, feature similarity와 위치 정보를 바탕으로 iterative clustering을 수행한다. 이렇게 얻은 superpixel centroid를 prototype으로 사용한다. 중요한 점은 이 과정이 논문 설명상 parameter-free, training-free라는 것이다. 즉, 별도의 학습 가능한 clustering 모듈을 두지 않고 feature space에서 직접 centroid를 계산한다.

두 번째는 **Guided Prototype Allocation (GPA)** 이다. query feature의 각 위치와 각 prototype 사이의 cosine similarity를 계산한 뒤, 위치별로 가장 유사한 prototype을 선택한다. 그 결과 query의 위치마다 다른 prototype이 guide feature로 할당된다. 이 과정은 object part visibility나 occlusion에 적응적으로 대응하기 위한 설계다. 예를 들어 query에 객체의 일부만 보이면, 그 부분과 잘 맞는 prototype만 쓰도록 유도한다.

기존 방법과의 차별점은 명확하다. 기존 prototype 방법들은 대개 masked average pooling으로 하나의 prototype만 만들거나, multiple prototype을 만들더라도 query 위치별 relevance를 충분히 반영하지 못했다. 논문은 prototype의 **생성 단계**와 **사용 단계** 모두를 adaptive하게 만들었다는 점을 차별점으로 내세운다. 생성 단계에서는 object 크기와 내부 구조에 따라 prototype 수와 영역이 달라지고, 사용 단계에서는 query의 각 pixel이 다른 prototype을 참조한다.

## 3. 상세 방법 설명

전체 네트워크는 **Adaptive Superpixel-guided Network (ASGNet)** 이다. support image와 query image는 shared CNN backbone을 통과해 feature를 추출한다. support feature는 SGC를 거쳐 여러 prototype으로 바뀌고, query feature는 GPA를 통해 이 prototype들과 매칭된다. 이후 feature enrichment module과 FPN-like top-down structure를 이용해 multi-scale 정보를 결합하고, 최종 segmentation prediction을 만든다.

### 3.1 문제 설정

few-shot segmentation에서는 학습 클래스와 테스트 클래스가 겹치지 않는다. 논문은 이를 다음처럼 정의한다.

$$
S_{train} \cap S_{test} = \varnothing
$$

각 episode에서 입력은 query image $I^Q$ 와, 같은 클래스 $c$ 를 가지는 $K$개의 support pair $(I^S_i, M^S_i)$ 이다. 목표는 query mask $\tilde{M}^Q$ 를 예측해 ground-truth mask $M^Q$ 에 가깝게 만드는 것이다.

### 3.2 Superpixel-guided Clustering (SGC)

SGC의 목적은 support object 내부에서 여러 개의 prototype을 뽑는 것이다. 입력은 support feature $F_s \in \mathbb{R}^{c \times h \times w}$, support mask $M_s \in \mathbb{R}^{h \times w}$, 그리고 초기 superpixel seed이다.

먼저 feature map에 각 픽셀의 absolute coordinate를 붙여 positional information을 추가한다. 이후 support mask를 적용해 foreground 위치의 feature만 남긴다. 이렇게 압축된 feature를 $F'_s \in \mathbb{R}^{c \times N_m}$ 로 둔다. 여기서 $N_m$ 은 foreground pixel 수다.

거리 함수는 feature distance와 spatial distance를 함께 고려한다.

$$
D = \sqrt{(d_f)^2 + (d_s/r)^2}
$$

여기서 $d_f$ 와 $d_s$ 는 각각 feature와 좌표의 Euclidean distance이고, $r$ 은 weighting factor다.

이후 iterative clustering을 수행한다. 각 iteration $t$ 에서 pixel $p$ 와 superpixel centroid $i$ 사이의 association은 다음과 같이 계산된다.

$$
Q^t_{pi} = e^{-D(F'_p, S^{t-1}_i)} = e^{- \lVert F'_p - S^{t-1}_i \rVert_2}
$$

직관적으로는 feature와 위치가 비슷한 pixel일수록 특정 centroid에 더 크게 할당된다. 그 다음 centroid는 weighted sum으로 업데이트된다.

$$
S^t_i = \frac{1}{Z^t_i} \sum_{p=1}^{N_m} Q^t_{pi} F'_p
$$

여기서 정규화 상수는

$$
Z^t_i = \sum_p Q^t_{pi}
$$

이다.

최종적으로 얻은 $S_i$ 들이 support object의 multiple prototype이 된다. 논문의 해석에 따르면 이 prototype은 object 전체 평균이 아니라, 비슷한 특성을 가지는 part-level representation이다.

초기 seed 배치도 중요하다. 일반 SLIC처럼 전체 이미지를 grid로 나누는 대신, foreground mask 내부에만 균등하게 seed를 뿌려야 한다. 이를 위해 저자들은 MaskSLIC의 아이디어를 빌려 foreground 영역 내부에 반복적으로 seed를 배치한다. 이런 초기화 덕분에 소수 iteration만으로도 수렴이 빨라진다고 설명한다.

### 3.3 Guided Prototype Allocation (GPA)

SGC로 만든 prototype을 query에 어떻게 쓸지가 GPA의 역할이다. 기존 방식처럼 prototype 하나를 query spatial size로 broadcast하면 모든 위치에 같은 guidance를 주게 된다. GPA는 이 점을 바꾼다.

각 query 위치 $(x,y)$ 의 feature $F^q_{x,y}$ 와 prototype $S_i$ 사이의 cosine similarity를 계산한다.

$$
C^i_{x,y} = \frac{S_i \cdot F^q_{x,y}}{\lVert S_i \rVert \cdot \lVert F^q_{x,y} \rVert}, \quad i \in \{1,2,\dots,N_{sp}\}
$$

이 similarity map은 두 갈래로 사용된다.

첫 번째 branch에서는 각 위치에서 가장 유사한 prototype index를 고른다.

$$
G_{x,y} = \arg\max_{i \in \{0,\dots,N_{sp}\}} C^i_{x,y}
$$

이렇게 얻은 $G$ 는 guide map이며, 각 위치마다 어떤 prototype을 참조할지 알려준다. guide map에 따라 prototype을 배치하면 pixel-wise guide feature $F_G$ 가 만들어진다.

두 번째 branch에서는 모든 prototype에 대한 similarity를 합쳐 probability map $P$ 를 만든다.

마지막으로 query 원본 feature $F_Q$, guide feature $F_G$, probability map $P$ 를 channel 방향으로 concat하고, $1 \times 1$ convolution을 적용해 refined query feature를 얻는다.

$$
F'_Q = f(F_Q \oplus F_G \oplus P)
$$

여기서 $\oplus$ 는 channel-wise concatenation이고, $f(\cdot)$ 는 $1 \times 1$ convolution이다.

직관적으로 보면 GPA는 "query의 각 위치가 support object의 어느 part와 가장 닮았는가"를 선택적으로 반영하는 모듈이다. 이 덕분에 보이지 않는 part의 prototype이 강제로 영향을 주는 문제를 줄일 수 있다.

### 3.4 적응성(adaptability)

논문은 적응성을 두 차원에서 설명한다.

첫째, SGC는 object scale에 적응한다. prototype 개수는 foreground 크기에 따라 달라진다.

$$
N_{sp} = \min\left(\left\lfloor \frac{N_m}{S_{sp}} \right\rfloor, N_{max}\right)
$$

여기서 $N_m$ 은 support mask 내부 pixel 수, $S_{sp}$ 는 superpixel seed 하나가 평균적으로 담당하는 면적이며 실험에서는 100으로 설정했다. 객체가 매우 작으면 $N_{sp}=0$ 또는 1이 되어 사실상 masked average pooling과 유사하게 동작한다. 반대로 객체가 크면 더 많은 prototype을 쓴다. 다만 계산량과 overfitting을 막기 위해 최대 개수 $N_{max}$ 를 둔다.

둘째, GPA는 object shape과 occlusion에 적응한다. query에서 어떤 part만 보이는 경우, 그 위치마다 가장 잘 맞는 prototype을 고르기 때문에 일부 part가 가려진 장면에서도 더 유연하게 동작한다.

### 3.5 K-shot 확장

기존 k-shot few-shot segmentation에서는 여러 support feature를 평균하거나 attention 기반으로 fusion하는 경우가 많다. 논문은 이 방식들이 계산량 대비 이득이 작다고 본다. ASGNet은 각 support image에서 SGC를 독립적으로 수행한 뒤, 모든 support의 prototype을 모아서 하나의 후보 집합으로 사용한다.

$$
S = (S_1, S_2, \dots, S_k)
$$

전체 prototype 수는

$$
N_{sp} = \sum_{i=1}^{k} N^i_{sp}
$$

이다.

즉, 여러 support image를 feature 차원에서 뭉개지 않고, prototype candidate pool을 넓힌 다음 GPA가 query 위치별로 더 좋은 선택을 하도록 설계했다. 논문은 이것이 추가 계산비용 없이 5-shot 성능 향상에 특히 효과적이라고 주장한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 지표

평가는 **Pascal-5$^i$** 와 **COCO-20$^i$** 에서 수행되었다. Pascal-5$^i$ 는 PASCAL VOC 2012와 SBD annotation을 포함한 20개 클래스를 4개 split으로 나눠 cross-validation한다. 추론 시 1000개의 support-query pair를 샘플링해 평가한다.

COCO-20$^i$ 는 MSCOCO 80개 클래스를 4개 split으로 나눠 같은 방식으로 평가한다. 이 데이터셋은 훨씬 더 크고 다양해서 더 어려운 벤치마크로 간주된다. 안정적인 평가를 위해 20,000개의 pair를 샘플링한다.

평가 지표는 주로 **mIoU** 를 사용하고, 비교의 일관성을 위해 **FB-IoU** 도 함께 보고한다.

### 4.2 구현 세부 사항

backbone은 ResNet을 사용했고, block2와 block3 feature를 concat해 feature map을 만든다. Pascal-5$^i$ 에서는 200 epoch, COCO-20$^i$ 에서는 50 epoch 학습한다. optimizer는 SGD이며, 학습률과 batch size는 데이터셋에 따라 다르게 설정했다. SGC iteration은 training에서 10, inference에서 5로 설정했다.

중요한 세부 사항 하나는 cosine similarity의 분산을 키우기 위해 support와 query feature 앞의 ReLU를 제거했다는 점이다. 저자 설명에 따르면 이렇게 하면 similarity 값의 범위가 $[0,1]$ 이 아니라 $[-1,1]$ 이 되어 prototype 간 구분이 더 잘 된다.

### 4.3 Ablation Study

#### prototype 개수의 영향

$N_{max}$ 를 1, 3, 5, 7, 9로 바꿔 1-shot Pascal-5$^i$ 를 평가했다. $N_{max}=1$ 은 사실상 single prototype이다. 평균 mIoU는 각각 58.74, 58.85, 59.29, 59.20, 58.88이었다. 가장 좋은 평균은 $N_{max}=5$ 에서 나왔다. 즉 prototype 수를 너무 적게 두면 표현력이 부족하고, 너무 많게 두면 오히려 overfitting이나 불필요한 prototype 때문에 성능이 소폭 떨어진다.

또한 adaptive하게 $N_{sp}$ 를 정하는 것이 fixed 5개보다 조금 더 좋았다. fixed일 때 평균 59.05, adaptive일 때 59.29였다. 성능 차이는 크지 않지만 계산량을 줄이면서 성능도 약간 개선된 점이 의미 있다고 저자들은 본다.

#### SGC와 GPA의 효과

논문은 prototype generation과 matching을 분리해 분석했다. baseline은 PFENet 스타일의 single prototype이다. 여기에 SGC를 넣어 multiple prototype을 만들고, 이후 GPA를 붙여 allocation을 수행한다.

표 3에 따르면:
- baseline(MAP + Expand): 1-shot 58.96, 5-shot 60.19
- SGC + Expand: 1-shot 58.31, 5-shot 61.24
- SGC + GPA: 1-shot 59.29, 5-shot 63.94

이 결과는 중요한 해석을 준다. SGC만 추가하고 matching이 단순하면 1-shot에서는 오히려 성능이 떨어진다. 논문은 single support에서 나온 여러 prototype이 서로 너무 비슷해져 cosine distance만으로는 잘 구분되지 않기 때문이라고 해석한다. 하지만 GPA까지 넣으면 prototype selection이 위치별로 이루어지므로 성능이 크게 오른다. 특히 5-shot에서 63.94로 매우 큰 향상이 나온다.

계산량 측면에서도 GPA는 prototype expansion보다 훨씬 유리하다. 표에 따르면 dense expansion 방식은 $K \times 8.5G + 0.5G$ FLOPs 증가가 들지만, GPA는 $K \times 5.5M + 0.9G$ 수준이다. 즉 성능뿐 아니라 효율도 개선된다.

#### K-shot fusion

5-shot에서 attention fusion, feature average, 제안 방식의 성능을 비교했다. Pascal-5$^0$ 실험에서:
- 1-shot baseline: 58.84
- Attention: 60.69
- Feature average: 62.10
- Ours: 63.66

제안 방식은 1-shot 대비 4.82% 향상으로 가장 컸고, 추가 계산량도 없다고 보고한다. 반면 attention은 계산량이 42.6G FLOPs 더 들면서 성능 향상은 1.85에 그친다. 이는 multiple support를 prototype pool로 유지한 뒤 GPA로 선택하게 하는 설계가 매우 효과적임을 보여준다.

### 4.4 SOTA 비교

#### Pascal-5$^i$

ResNet-101 backbone 기준으로 ASGNet은 1-shot 평균 mIoU 59.31, 5-shot 64.36을 기록했다. PFENet은 각각 60.10, 61.40이었다. 즉 1-shot에서는 거의 비슷하거나 약간 낮지만, 5-shot에서는 2.96 정도가 아니라 표 기준으로 기존 최고 대비 **2.40%** 향상이라고 저자들이 요약한다. 논문 본문의 claim은 5-shot 성능 개선에 초점이 있다.

FB-IoU에서도 ASGNet(RN-101)은 1-shot 71.7, 5-shot 75.2를 기록했다. PFENet은 73.3, 73.9였으므로, 1-shot은 낮지만 5-shot은 더 높다. 특히 ASGNet(RN-50)은 파라미터 수가 10.4M으로 비교적 작으면서 5-shot FB-IoU 74.2를 달성했다. 저자들은 trainable parameter 수가 적다는 점도 장점으로 강조한다.

정성 결과에서도 support와 query 사이 pose와 appearance 변화가 큰 경우에도 비교적 정확한 segmentation을 보인다고 보고한다.

#### COCO-20$^i$

COCO-20$^i$ 에서는 성능 향상이 더 크다. ASGNet(ResNet-50)은 1-shot mIoU 34.56, 5-shot 42.48을 기록했다. RPMMs는 30.58, 35.52였고, PFENet은 32.40, 37.40이었다. 특히 5-shot에서는 PFENet 대비 5.08, RPMMs 대비 6.96 정도 더 높다. 논문은 이 결과를 통해 ASGNet이 더 복잡하고 어려운 데이터셋에서도 잘 작동한다고 주장한다.

Supplementary의 split별 결과에서도 COCO-20$^i$ 1-shot과 5-shot의 모든 split에서 최고 성능을 냈다고 보고한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 prototype learning의 약점을 아주 구체적으로 짚고, 그 약점에 맞춘 구조를 제안했다는 점이다. "하나의 prototype은 너무 거칠다"는 문제 제기가 직관적일 뿐 아니라, 실제로 SGC와 GPA라는 두 모듈로 나누어 해결하고, ablation에서도 각 모듈의 역할을 분리해 보여준다. 특히 "multiple prototype만으로는 부족하고 allocation까지 adaptive해야 한다"는 실험 결과가 설계의 타당성을 뒷받침한다.

또 다른 강점은 효율성이다. 저자들은 ASGNet이 lightweight하며, parameter 수가 적고, 특히 k-shot 확장에서 추가 계산비용 없이 큰 성능 향상을 얻는다고 주장한다. 5-shot COCO에서 개선폭이 크다는 점도 실용적으로 중요하다. few-shot segmentation에서는 support가 늘어날 때 처리 비용이 급격히 커지는 방법들이 많은데, 본 논문은 support 정보를 prototype pool로 다루어 이를 비교적 깔끔하게 해결했다.

적응성에 대한 설명도 설득력이 있다. SGC는 object 크기에 따라 prototype 수를 조절하고, GPA는 query 위치마다 다른 prototype을 선택한다. 따라서 scale variation, shape variation, occlusion에 대응하려는 설계 의도가 비교적 분명하다. Supplementary의 similarity map 시각화도 각 prototype이 object의 서로 다른 part를 나타낸다는 해석과 일치한다.

다만 한계도 있다. 첫째, 1-shot setting에서는 일관된 압도적 우위를 보이지 않는다. Pascal-5$^i$ 의 경우 PFENet보다 1-shot mIoU가 오히려 약간 낮다. 즉 이 방법의 강점은 주로 5-shot에서 크게 나타난다. 이는 proposal의 핵심 이점이 다양한 prototype 후보를 많이 확보할 때 더 잘 드러난다는 뜻으로 읽힌다.

둘째, SGC 자체만으로는 충분하지 않다. Table 3에서 SGC + Expand는 1-shot에서 baseline보다 나쁘다. 즉 "multiple prototype"이라는 아이디어만으로 성능이 자동으로 좋아지는 것은 아니며, 이를 어떻게 query에 할당하느냐가 매우 중요하다. 다시 말해 전체 기여는 SGC와 GPA의 결합에 있고, SGC 단독의 효용은 제한적일 수 있다.

셋째, superpixel clustering의 안정성이나 하이퍼파라미터 민감도는 완전히 해소되지 않았다. 예를 들어 $S_{sp}$ 를 100으로 경험적으로 정했다고만 되어 있고, 다른 값에 대한 분석은 제공되지 않았다. 또한 $N_{max}$ 가 너무 크면 성능이 떨어진다는 사실은 보였지만, 왜 어떤 데이터셋이나 backbone에서 최적 값이 같은지에 대한 이론적 설명은 없다.

넷째, 논문은 clustering이 training-free이고 parameter-free라고 설명하지만, 실제 전체 시스템은 backbone feature quality에 강하게 의존한다. unseen class generalization이 잘 되는 이유가 SGC/GPA의 공인지, backbone과 enrichment 구조의 영향인지 완전히 분리되어 있지는 않다. 이 부분은 ablation으로 일부 보이지만 완전히 분해되었다고 보긴 어렵다.

다섯째, 논문은 severe occlusion에 강하다고 정성적으로 보여주지만, occlusion robustness를 별도의 정량 실험으로 검증하지는 않았다. 따라서 "occlusion에 강하다"는 주장은 시각화와 설계 직관에는 근거하지만, 독립적인 benchmark 수준의 정량 검증까지 제시된 것은 아니다.

## 6. 결론

이 논문은 few-shot segmentation에서 single prototype representation의 한계를 해결하기 위해, support object를 여러 개의 adaptive prototype으로 나누어 표현하고, query의 각 위치에 가장 적합한 prototype을 동적으로 배정하는 **ASGNet** 을 제안했다. 핵심 구성은 feature-space superpixel clustering 기반의 **SGC** 와 위치별 prototype selection을 수행하는 **GPA** 이다.

실험 결과를 보면 이 방법은 특히 5-shot setting에서 강력하다. Pascal-5$^i$ 와 COCO-20$^i$ 모두에서 당시 state-of-the-art를 능가했고, 특히 COCO-20$^i$ 5-shot에서 큰 폭의 개선을 보였다. 이는 support example이 늘어날수록 다양한 prototype 후보를 유지하고 query 위치별로 선택하는 전략이 효과적임을 시사한다.

실제 적용 측면에서는, 적은 annotated support로 새로운 객체 클래스를 빠르게 분할해야 하는 상황에서 의미가 있다. 또한 향후 연구에서는 이 아이디어를 더 발전시켜, prototype 수 결정의 자동화, 더 정교한 allocation mechanism, stronger backbone과의 결합, 또는 video/object tracking 같은 다른 dense prediction 문제로 확장할 여지가 있다. 논문이 제시한 핵심 메시지는 분명하다. few-shot segmentation에서 중요한 것은 단지 prototype을 만드는 것이 아니라, **얼마나 적응적으로 만들고, 얼마나 적절하게 할당하느냐** 이다.
