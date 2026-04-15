# Video Object Segmentation with Joint Re-identification and Attention-Aware Mask Propagation

- **저자**: Xiaoxiao Li, Chen Change Loy
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1803.04242

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation 문제를 다룬다. 구체적으로는 첫 프레임에서 주어진 object mask를 바탕으로, 이후 모든 프레임에서 각 객체를 정확하게 분할하고 추적하는 것이 목표다. 특히 DAVIS 2017처럼 한 비디오 안에 여러 instance가 동시에 존재하는 경우를 핵심 과제로 본다.

저자들이 강조하는 연구 문제는 크게 두 가지다. 첫째, 여러 객체가 서로 가리거나 완전히 occlusion되면 단순한 temporal propagation만으로는 객체를 계속 추적하기 어렵다. 둘째, 객체의 scale과 pose가 프레임마다 크게 변하면 첫 프레임 mask만을 template로 삼는 방식은 일반화가 잘 되지 않는다. 즉, 어떤 방법은 “연속성”에는 강하지만 occlusion 후 재등장에 약하고, 다른 방법은 “재탐색”은 가능하지만 appearance variation에 약하다.

이 문제는 실제 비디오 분석에서 매우 중요하다. 비디오 속 객체는 배경 clutter, 상호 가림, 자세 변화, 크기 변화가 동시에 발생하기 때문이다. 따라서 여러 객체를 장기간 안정적으로 segment하고 다시 찾아내는 능력은 video understanding의 핵심 기반 기술이라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 video object segmentation을 두 개의 상호보완적 기능으로 나누고, 이를 하나의 end-to-end 네트워크로 결합하는 것이다. 그 두 기능은 다음과 같다. 하나는 re-identification(Re-ID)으로, occlusion 등으로 사라졌다가 다시 나타난 객체를 찾아내는 역할이다. 다른 하나는 recurrent mask propagation(Re-MP)으로, 일단 신뢰할 수 있는 시작 mask가 있으면 시간축으로 앞뒤로 전파하며 객체를 연속적으로 추적하는 역할이다.

이 설계의 직관은 명확하다. Re-ID는 “어디서 다시 시작해야 하는가”를 결정하고, Re-MP는 “찾아낸 시작점에서 주변 프레임으로 어떻게 퍼뜨릴 것인가”를 담당한다. 저자들은 이를 천에 염료를 찍고 주변으로 퍼뜨리는 비유로 설명하며, 그래서 모델 이름을 DyeNet이라고 붙였다.

기존 접근과의 차별점은 세 가지로 요약할 수 있다. 첫째, template matching 계열과 temporal propagation 계열을 단순히 후처리 수준에서 결합한 것이 아니라 하나의 통합 프레임워크로 joint learning한다. 둘째, Re-ID와 Re-MP를 한 번만 수행하는 것이 아니라 iterative하게 반복하면서 template set을 확장한다. 셋째, propagation 과정에 attention mechanism을 넣어 distractor나 다른 객체의 영향을 줄인다. 특히 template expansion은 첫 프레임 template만 고정적으로 쓰는 기존 방법의 약점을 줄이는 핵심 장치다.

## 3. 상세 방법 설명

### 전체 파이프라인

입력은 비디오 시퀀스 $\{I_1, \dots, I_N\}$와 첫 프레임의 object masks이다. 각 프레임 $I_i$는 feature extractor $N_{feat}$를 통과하여 feature map $f_i$를 만든다.

본문의 표기대로 이는 다음과 같다.

$$
f_i = N_{feat}(I_i)
$$

feature extractor는 ResNet-101을 backbone으로 사용하며, `conv1`부터 `conv4_x`까지만 공통 backbone으로 사용한다. 해상도를 높이기 위해 `conv4_x`의 stride를 줄이고 dilated convolution을 사용하여 feature map 해상도를 입력의 $1/8$ 수준으로 유지한다. 이 feature는 Re-ID와 Re-MP가 함께 공유하므로 계산을 아낄 수 있다.

전체 추론은 iterative하게 진행된다.

1. 첫 프레임 mask들을 초기 template로 사용한다.
2. Re-ID 모듈이 전체 비디오에서 template와 유사한 후보 mask를 찾아 high-confidence starting point를 만든다.
3. Re-MP 모듈이 이 starting point를 기준으로 시간축 양방향으로 mask를 전파해 tracklet을 만든다.
4. confidence가 높은 예측 결과를 새 template로 추가한다.
5. 더 이상 high-confidence object를 찾지 못할 때까지 반복한다.

저자들에 따르면 첫 iteration에서 DAVIS 2017 기준 약 25%의 mask를 starting point로 찾고, 세 번 반복하면 약 33%까지 증가한다. 여기서 중요한 점은 모든 프레임을 Re-ID로 직접 찾는 것이 아니라, propagation이 잘 작동할 수 있는 “좋은 시작점”만 확보해도 된다는 것이다.

### Re-identification 모듈

Re-ID 모듈의 목적은 비디오 전역에서 target object를 다시 찾는 것이다. 각 프레임마다 Region Proposal Network(RPN)가 object proposal bounding box $\{b^i_1, \dots, b^i_M\}$를 생성한다. proposal을 쓰는 이유는 anchor 크기가 다양하므로 scale variation에 대응하기 쉽기 때문이다.

각 proposal에 대해 feature map $f_i$에서 RoIAlign을 적용해 고정 크기 $m \times m$의 feature를 추출한다. 이 feature는 두 개의 shallow subnet으로 들어간다.

첫 번째는 mask network다. 이것은 proposal 내부의 주 객체에 대한 $m \times m$ binary mask를 예측한다. 두 번째는 re-identification network다. 이것은 proposal feature를 $L2$-normalized 256차원 embedding space로 투영해 mask feature를 만든다. template 역시 같은 공간으로 투영된다.

그 다음 candidate와 template 사이 cosine similarity를 계산한다. similarity가 threshold $\rho_{reid}$보다 크면 그 candidate의 mask를 starting point로 채택한다. 즉, Re-ID는 bounding box detection 자체보다 “template와 비슷한 instance를 높은 precision으로 골라내는 것”에 초점을 둔다.

이 모듈에서 template expansion이 중요하다. 첫 프레임 template만 쓰면 자세 변화나 크기 변화가 큰 경우 매칭이 어려워진다. 하지만 iteration이 진행되며 새롭게 confident하게 찾은 mask를 template에 추가하면, 나중에는 다른 pose나 scale의 object도 더 잘 검색할 수 있다. 실험에서도 iteration이 거듭될수록 성능이 올라간다.

### Recurrent Mask Propagation 모듈

Re-MP는 starting point에서 출발해 mask를 인접 프레임으로 전파하는 모듈이다. 저자들은 이를 recurrent neural network로 모델링했다. 설명은 forward propagation 기준으로 제시되지만 backward도 같은 방식이다.

어떤 instance $k$에 대해 $i$번째 프레임에서 찾은 mask $\hat{y}$가 있다고 하자. 이후 $(j-1)$번째 프레임까지 propagation했다고 할 때, $j$번째 프레임의 mask $y_j$를 예측하는 식은 다음과 같다.

$$
h_j = N_R(h_{(j-1)\rightarrow j}, x_j)
$$

$$
y_j = N_O(h_j)
$$

여기서 $h_j$는 hidden state이고, $N_R$은 recurrent function, $N_O$는 output function이다.

이 식의 의미를 풀어 쓰면 다음과 같다.

먼저 $(j-1)$ 프레임과 $j$ 프레임 사이 optical flow $F_{(j-1)\rightarrow j}$를 FlowNet 2.0으로 계산한다. 이전 프레임의 mask $y_{j-1}$를 optical flow로 warping하여 현재 프레임에서의 예상 위치 $y_{(j-1)\rightarrow j}$를 얻는다. 이 warped mask의 bounding box를 현재 프레임에서 해당 객체가 있을 위치로 간주한다. 그리고 현재 프레임 feature $f_j$에서 그 box 영역을 RoIAlign으로 잘라 $x_j$를 얻는다.

동시에 이전 hidden state $h_{j-1}$도 optical flow로 warp하여 $h_{(j-1)\rightarrow j}$를 만든다. 이렇게 하면 과거 기억과 현재 프레임 정보를 spatially aligned한 상태로 합칠 수 있다. 결국 $x_j$와 warped hidden state를 합쳐 현재 hidden state $h_j$를 갱신하고, 여기서 mask $y_j$를 출력한다.

즉, 이 모듈은 단순히 이전 mask만 넘기는 것이 아니라, 과거의 richer한 hidden representation 전체를 다음 프레임으로 전달한다. 이 점이 기존 단순 propagation 계열보다 강한 이유다.

### Attention mechanism

저자들은 propagation 과정의 중요한 문제로 distractor를 지적한다. bounding box 안에는 target object 외에도 다른 객체나 배경 clutter가 들어올 수 있다. 이 경우 hidden state만으로 바로 mask를 예측하면 잘못된 영역까지 target으로 segment할 수 있다.

이를 해결하기 위해 attention distribution $a_j \in \mathbb{R}^{m \times m \times 1}$를 도입한다. warped hidden state $h_{(j-1)\rightarrow j}$를 convolution layer와 softmax에 통과시켜 spatial attention map을 만든 뒤, 이를 현재 hidden state $h_j$에 채널 방향으로 곱해 target region에 집중하도록 한다. 그 후 향상된 hidden state로부터 $y_j$를 생성한다.

논문의 질적 예시에서는 attention이 없는 경우 다른 개나 사람에게 instance id가 잘못 번지는 현상이 나타난다. attention을 넣으면 target object 중심으로 propagation이 훨씬 안정적이다. 저자들은 이것이 mask propagation에 attention을 도입한 첫 시도라고 주장한다.

### Tracklet 생성과 연결

각 starting point는 forward와 backward propagation을 통해 하나의 tracklet을 만든다. 다만 서로 다른 starting point가 실제로는 같은 객체의 같은 구간을 설명할 수도 있다. 이런 중복 계산을 줄이기 위해 starting point를 template similarity 순으로 정렬한 뒤, 이미 만들어진 tracklet과 mask overlap이 높은 경우는 skip한다.

이후 여러 tracklet을 greedy 방식으로 linking하여 최종 mask tube를 만든다. 우선 template와 가장 similarity가 높은 tracklet을 각 template에 먼저 할당하고, 나머지 tracklet은 모순이 없으면 상위 tracklet과 합친다. 저자들은 더 정교한 방법으로 conditional random field 등을 미래 과제로 언급하지만, 본 논문에서는 greedy linking만으로도 충분히 잘 동작했다고 말한다.

### 학습 목표와 학습 절차

전체 loss는 다음과 같다.

$$
L = L_{reid} + \lambda (L_{mask} + L_{remp})
$$

여기서 $L_{reid}$는 Re-ID network의 loss이며, 논문에 따르면 Online Instance Matching(OIM) loss를 따른다. $L_{mask}$는 Re-ID 모듈 안 mask network의 pixel-wise segmentation loss이고, $L_{remp}$는 recurrent mask propagation 모듈의 pixel-wise segmentation loss다. $\lambda$는 이 손실들의 scale을 맞추기 위한 가중치다.

초기화는 semantic segmentation network의 weight를 사용한다. 메모리 한계 때문에 `conv1`부터 `conv4_20`까지는 training 중 freeze한다. Re-ID subnet은 추가로 ImageNet으로 pretrain한다. 이후 DAVIS training set에서 joint training을 수행한다. 논문에 명시된 설정은 24k iterations, mini-batch 32장(8개 비디오에서 각 4프레임), momentum $0.9$, weight decay $5\times10^{-4}$, 초기 learning rate $10^{-3}$이며 8k iteration마다 10배씩 감소시킨다.

online training의 경우에는 테스트 비디오 첫 프레임을 기반으로 합성 비디오를 만들어 training set에 추가하는 방식을 따른다. 이 부분은 [16]의 절차를 따른다고만 되어 있고, 본문에 세부 합성 방식은 자세히 적혀 있지 않다.

## 4. 실험 및 결과

### 데이터셋과 평가 지표

논문은 DAVIS 2016, DAVIS 2017, SegTrack v2, YouTubeObjects에서 실험한다.

DAVIS 2016은 single-object segmentation 중심의 데이터셋이고, DAVIS 2017은 multi-object annotation이 추가되어 훨씬 어렵다. SegTrack v2는 저해상도 14개 비디오, YouTubeObjects는 약 126개 비디오 subset을 사용한다.

평가 지표는 DAVIS 2017에서는 region similarity인 $J$, boundary accuracy인 $F$, 그리고 이 둘의 평균인 $G$를 사용한다. 다른 세 데이터셋에서는 mIoU를 사용한다.

또한 학습 모드를 offline training과 online training으로 나눈다. offline은 test annotation을 전혀 쓰지 않는 방식이다. online은 첫 프레임 annotation을 활용해 모델을 적응시키는 방식이며, per-dataset online training과 per-video online training으로 더 나뉜다. DyeNet은 online training 없이도 상당한 성능을 보이고, online training을 하면 더 좋아진다.

### Ablation: Re-MP 모듈 효과

저자들은 먼저 Re-ID 없이 Re-MP만으로 실험하여 propagation 모듈 자체의 효과를 본다. 비교 대상은 MSK [26]이며, 공정 비교를 위해 동일한 ResNet-101 backbone으로 재구현했다.

DAVIS 2017 validation에서 결과는 다음과 같다.

- MSK: $G$-mean 65.3
- Re-MP without attention: $G$-mean 67.5
- Re-MP with attention: $G$-mean 69.1

즉, recurrent structure와 RoIAlign 기반 foreground focus만으로도 기존 propagation보다 좋아지고, attention을 추가하면 다시 1.6포인트 향상된다. 논문은 이를 통해 Re-MP가 단순 이전 mask propagation보다 더 많은 historical information을 유지하고, attention이 distractor 억제에 실질적으로 기여한다고 해석한다.

### Ablation: Re-ID와 template expansion 효과

Re-ID 실험에서는 similarity threshold $\rho_{reid}$를 바꾸며 precision-recall trade-off를 분석한다. $\rho_{reid}$가 낮아지면 첫 iteration에서 더 많은 starting point를 찾으므로 recall은 올라가지만 precision이 낮아질 수 있다. 반대로 threshold가 높으면 시작점은 적지만 더 믿을 만하다.

중요한 점은 template expansion 덕분에 iteration이 진행될수록 $G$-mean이 대체로 증가한다는 사실이다. 예를 들어 $\rho_{reid}=0.7$일 때 1회차 73.2, 2회차 74.1, 이후도 74.1 수준을 유지한다. 저자들은 전체적으로 약 3번 iteration이면 충분하다고 본다. 최종적으로는 $\rho_{reid}=0.7$을 이후 실험의 기본값으로 사용한다.

이 결과는 template expansion이 실제로 첫 프레임 template 의존성을 낮추고, pose/scale variation이 있는 재등장 객체 검색을 도와준다는 주장을 뒷받침한다.

### Ablation: 전체 DyeNet 구성요소 효과

DAVIS 2017 test-dev에서 단계적으로 모듈을 추가한 결과는 다음과 같다.

- MSK baseline: $G$-mean 51.7
- Re-MP without attention: 58.0
- Re-MP with attention: 61.0
- Full DyeNet (+Re-ID): 68.2
- DyeNet offline only: 62.5

이 결과는 두 가지를 보여준다. 첫째, attention-aware Re-MP만으로도 baseline 대비 큰 폭의 향상이 있다. 둘째, Re-ID를 더하면 7.2포인트가 추가로 상승하여 occlusion 이후 재탐색이 실제로 결정적이라는 점이 드러난다. 또한 online training이 없을 때도 62.5라는 경쟁력 있는 성능을 보인다는 점도 강조된다.

### 속성별 분석

저자들은 object size, scale variation, occlusion, pose variation별로 성능 증가 양상을 분석했다. 논문 설명에 따르면 가장 큰 영향을 주는 요인은 object size와 occlusion이다. scale variation은 pose variation보다 성능에 더 큰 영향을 준다.

세부 해석은 다음과 같다.

Re-MP는 작은 객체를 추적하는 데 도움이 되며, partial occlusion 상황에서 distractor에 덜 흔들린다. 반면 heavy occlusion에서는 propagation만으로는 부족하므로 Re-ID가 빠진 객체를 다시 찾는 데 크게 기여한다. 또한 template expansion 덕분에 pose variation이 큰 경우에도 Re-ID가 잘 작동한다.

### 벤치마크 결과

DAVIS 2017 test-dev에서 DyeNet의 결과는 다음과 같다.

- DyeNet offline: $J=60.2$, $F=64.8$, $G=62.5$
- DyeNet online: $J=65.8$, $F=70.5$, $G=68.2$

이는 표에 제시된 기존 방법들인 OnAVOS, LucidTracker, VS-ReID보다 높은 수치다. 특히 VS-ReID의 $G=66.1$보다 높은 68.2를 달성했다고 보고한다.

다른 데이터셋에서도 strong result를 보인다.

- DAVIS 2016: offline 84.7 mIoU, online 86.2 mIoU
- SegTrack v2: offline 78.3, online 78.7
- YouTubeObjects: offline 74.9, online 79.6

저자들은 특히 DAVIS 2017으로 학습한 모델을 SegTrack v2와 YouTubeObjects에 별도 offline 학습 없이 적용했음에도 높은 성능을 보인 점을 들어 generalization과 transferability를 강조한다. YouTubeObjects에서는 offline prediction이 일부 ground-truth annotation보다 더 좋아 보이며, 성능 손실이 annotation bias 때문일 수 있다고 주장한다. 다만 이 부분은 정량적 증거보다는 저자들의 해석에 가깝다.

### 속도 분석

속도도 논문의 중요한 주장 중 하나다. 기존 강한 방법들은 online training과 post-processing 때문에 느리다. 예를 들어 논문이 인용한 수치에 따르면 OnAVOS는 DAVIS 2016 val에서 약 13초/frame이 걸리고, LucidTracker는 대량의 online training과 후처리가 필요하다.

반면 DyeNet은 다음과 같은 속도를 보인다.

- offline DyeNet: 2.4 FPS, DAVIS 2016에서 84.7 mIoU
- 2k per-dataset online training 후: 0.43 FPS, 86.2 mIoU

또한 VS-ReID의 추론 속도는 DAVIS에서 약 3초/frame이며, DyeNet은 그보다 약 7배 빠르다고 본문에서 주장한다. 이 주장은 공유 feature extraction과 효율적인 inference 절차 덕분이라는 설명과 연결된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제를 매우 현실적으로 분해했다는 점이다. video object segmentation의 실패 원인을 단순히 segmentation capacity 부족으로 보지 않고, “연속 추적 실패”와 “재등장 객체 검색 실패”라는 두 문제로 나눈 뒤 이를 구조적으로 결합했다. 그래서 Re-MP와 Re-ID가 서로 명확히 다른 실패 모드를 보완한다.

또 다른 강점은 iterative template expansion이다. 첫 프레임 annotation만을 고정 template로 쓰는 방식은 multi-instance, large variation 상황에서 근본적인 약점이 있는데, 이 논문은 이를 간단하지만 효과적인 반복 갱신 전략으로 해결한다. ablation 결과도 이 설계의 효과를 잘 보여준다.

attention-aware propagation도 설득력 있는 구성이다. propagation에서 bounding box 내부 distractor는 실제로 빈번한 문제인데, attention을 추가해 spatial focus를 강화한 점은 구조적으로 타당하며 정량·정성 결과에서도 이점이 확인된다.

또한 end-to-end joint learning이 가능하고, online training 없이도 경쟁력 있는 성능과 빠른 속도를 보인다는 점은 실용성 측면에서 강하다. 당시 DAVIS 2017 benchmark 기준 state-of-the-art를 달성했다는 점도 분명한 성과다.

반면 한계도 있다. 첫째, Re-ID는 여전히 proposal quality에 의존한다. RPN이 target object를 충분히 잘 제안하지 못하면 이후 단계도 시작점을 잃을 수 있다. 논문은 RPN을 별도로 학습한다고 했지만, proposal 품질이 전체 시스템에 미치는 영향은 깊게 분석하지 않았다.

둘째, propagation에서 optical flow 품질이 중요하다. Re-MP는 mask와 hidden state를 flow-guided warping하므로, optical flow가 부정확한 경우 위치 추정이 흔들릴 가능성이 있다. 하지만 논문은 optical flow 오차에 대한 민감도 분석을 제공하지 않는다.

셋째, tracklet linking은 greedy 방식이다. 저자들도 이를 임시적이고 단순한 방법으로 제시하며, 더 정교한 linking 방법을 future work로 남긴다. 따라서 복잡한 다중 객체 interaction이나 모호한 충돌 상황에서는 전역 최적화 관점에서 한계가 있을 수 있다.

넷째, online training이 없는 경우에도 강하지만 최고 성능은 여전히 online adaptation에 기대고 있다. 즉, 완전히 training-free한 inference만으로 최고 성능을 내는 구조는 아니다.

다섯째, loss 함수의 각 항과 하이퍼파라미터 $\lambda$의 구체적 설정, attention 계산 세부, linking 과정의 threshold 등은 본문에 충분히 상세히 서술되지 않았다. 따라서 재현성 측면에서 일부 구현 의존적인 부분이 남아 있다. 이는 논문이 틀렸다는 뜻이 아니라, 제공된 본문만으로는 완전한 재현이 쉽지 않다는 의미다.

비판적으로 보면, 이 논문의 성능 향상은 분명하지만 구성요소가 적지 않게 복합적이다. ResNet backbone, RPN, RoIAlign, FlowNet 2.0, Re-ID embedding, recurrent propagation, attention, iterative template expansion, greedy linking이 모두 함께 작동한다. 따라서 어느 정도는 강한 기존 컴포넌트를 잘 조합한 시스템 논문 성격도 있다. 그럼에도 불구하고 Re-ID와 attention-aware Re-MP를 하나의 반복형 프레임워크로 통합한 설계는 충분히 독창적이고, ablation이 그 기여를 비교적 명확히 보여준다.

## 6. 결론

이 논문은 multi-instance video object segmentation에서 핵심 난제인 occlusion, pose variation, scale variation, distractor 문제를 동시에 다루기 위해 DyeNet을 제안했다. 핵심 기여는 크게 세 가지다. 첫째, re-identification과 temporal mask propagation을 end-to-end로 결합한 unified framework를 제시했다. 둘째, iterative template expansion을 통해 appearance variation에 더 강한 Re-ID를 구현했다. 셋째, attention-aware recurrent mask propagation을 통해 distractor에 강한 propagation을 달성했다.

실험적으로는 DAVIS 2017 test-dev에서 $G$-mean 68.2를 기록하며 당시 state-of-the-art를 달성했고, DAVIS 2016, SegTrack v2, YouTubeObjects에서도 매우 강한 성능을 보였다. 또한 online training 없이도 경쟁력 있는 정확도와 빠른 추론 속도를 보인다는 점에서 practical value가 크다.

향후 연구 관점에서 이 논문은 “tracking-like continuity”와 “retrieval-like re-identification”을 함께 다뤄야 video segmentation이 강해진다는 방향을 명확히 보여준다. 실제 적용 측면에서도 장면 내 여러 객체가 반복적으로 가려졌다 나타나는 surveillance, robotics, video editing, autonomous driving과 같은 문제에 중요한 기반 아이디어를 제공한다고 볼 수 있다.
