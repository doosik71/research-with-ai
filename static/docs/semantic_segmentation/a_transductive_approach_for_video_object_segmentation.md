# A Transductive Approach for Video Object Segmentation

- **저자**: Yizhuo Zhang, Zhirong Wu, Houwen Peng, Stephen Lin
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2004.07193

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation(VOS) 문제를 다룬다. 문제 설정은 간단하다. 비디오의 첫 프레임에서 target object의 mask가 주어졌을 때, 이후 모든 프레임에서 같은 객체를 분할해야 한다. 겉보기에는 단순하지만, 실제로는 객체의 deformation, occlusion, scale 변화, 빠른 motion, 다중 객체 간 혼동 때문에 매우 어렵다.

저자들이 제기하는 핵심 문제의식은 기존 강한 성능의 VOS 방법들이 공정한 비교가 어렵다는 점이다. 당시 많은 방법은 optical flow, instance segmentation, object re-identification, tracking 같은 외부 모듈을 함께 사용했고, 이 모듈들은 별도 데이터셋에서 사전학습되었다. 따라서 성능 향상이 정말 VOS 알고리즘 자체에서 온 것인지, 아니면 외부 task의 prior에서 온 것인지 분리해 보기 어렵다.

이 논문은 이런 복잡성을 줄이기 위해, 추가 모듈 없이도 강력하게 동작하는 단순한 transductive approach를 제안한다. 핵심은 첫 프레임의 label을 이후 프레임으로 직접 “전파”하되, 단순히 인접 프레임만 보는 것이 아니라 과거 여러 프레임에 걸친 dense한 long-term dependency를 활용하는 것이다. 저자들은 이를 통해 성능과 속도를 동시에 얻고, 향후 연구를 위한 강한 baseline을 만들고자 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 video object segmentation을 transductive inference 문제로 재해석하는 것이다. 즉, label이 있는 첫 프레임과 label이 없는 나머지 프레임들을 하나의 spatio-temporal volume으로 보고, 이 안에서 비슷한 픽셀끼리는 같은 label을 가져야 한다는 smoothness 가정을 이용해 label을 전파한다.

기존 propagation 기반 방법들은 보통 두 가지 방식 중 하나였다. 첫째, 첫 프레임만 reference로 삼아 먼 프레임에 label을 전달한다. 둘째, 첫 프레임과 직전 프레임 정도만 참고하는 sparse하고 local한 propagation을 쓴다. 저자들은 이런 방식이 deformation이나 occlusion이 심할 때 drifting 문제를 일으킨다고 본다. 중간에 appearance가 크게 바뀌면, 짧은 범위의 전파만으로는 객체를 안정적으로 다시 찾기 어렵기 때문이다.

이 논문의 차별점은 unlabeled temporal structure 전체를 더 적극적으로 쓴다는 점이다. 현재 프레임을 예측할 때, 과거 프레임들 전반에 대해 similarity graph를 구성하고 label을 holistic하게 전파한다. 다만 모든 과거 프레임을 쓰면 계산량이 너무 커지므로, 최근 프레임은 dense하게, 먼 과거는 sparse하게 sampling하는 전략을 사용한다. 이로써 최근 motion 정보와 장기 appearance 정보를 동시에 반영한다.

또 하나의 중요한 차별점은 구현의 단순성이다. 저자들에 따르면 이 방법은 추가 데이터셋, 특수한 memory architecture, optical flow module, re-ID module 없이, pretrained ResNet-50 하나로 학습 및 추론이 가능하다. inference도 feed-forward와 dot product 위주로 구성되어 매우 빠르다.

## 3. 상세 방법 설명

### 3.1 Transductive inference의 기본 공식

저자들은 먼저 일반적인 semi-supervised classification의 transductive regularization framework를 소개한다. labeled data와 unlabeled data가 함께 있을 때, unlabeled sample의 예측 label $\hat{y}$는 다음 에너지를 최소화하는 방향으로 정해진다.

$$
Q(\hat{y}) = \sum_{i,j}^{n} w_{ij}\left\| \frac{\hat{y}_i}{\sqrt{d_i}} - \frac{\hat{y}_j}{\sqrt{d_j}} \right\|^2 + \mu \sum_{i=1}^{l} \|\hat{y}_i - y_i\|^2
$$

여기서 $w_{ij}$는 sample $x_i$와 $x_j$의 similarity이고, $d_i = \sum_j w_{ij}$는 degree이다. 첫 번째 항은 smoothness constraint로, 서로 비슷한 sample은 같은 label을 갖도록 강제한다. 두 번째 항은 fitting constraint로, 이미 주어진 labeled sample의 정답에서 벗어나지 않도록 한다.

최종 목표는 다음 최적화 문제를 푸는 것이다.

$$
\hat{y} = \arg\min Q(y)
$$

이 문제는 normalized similarity matrix $S = D^{-1/2}WD^{-1/2}$를 이용하여 다음과 같은 iterative update로 풀 수 있다고 설명한다.

$$
\hat{y}^{(k+1)} = \alpha S\hat{y}^{(k)} + (1-\alpha)y^{(0)}
$$

여기서 $\alpha = \mu / (\mu + 1)$이고, 보통 $\alpha = 0.99$를 사용한다. 직관적으로 보면, 현재 예측값을 similarity graph를 따라 주변 sample들로부터 반복적으로 보정해 나가는 과정이다.

### 3.2 이를 online VOS로 바꾸는 방법

비디오에서는 모든 프레임이 동시에 주어지는 것이 아니라 시간 순서대로 들어오므로, 위 식을 그대로 쓰기 어렵다. 또한 픽셀 수가 매우 많아서 전체 비디오에 대한 거대한 similarity matrix를 직접 계산하는 것도 비현실적이다.

그래서 저자들은 online inference에 맞게 다음과 같이 단순화한다.

$$
\hat{y}^{(t+1)} = S_{1:t \rightarrow t+1}\hat{y}^{(t)}
$$

즉, 현재 시점 $t+1$의 픽셀 label은 과거 $1 \sim t$ 프레임의 픽셀들과의 similarity를 통해 계산한다. 첫 프레임 이후에는 새 supervised label이 주어지지 않으므로, 일반 semi-supervised 식에 있던 prior term은 현재 프레임에서는 빠진다.

이 식은 현재 프레임 픽셀과 과거 프레임 픽셀 사이에서 smoothness term만 최소화하는 것과 같은 의미를 가진다.

$$
Q_{t+1}(\hat{y}) = \sum_i \sum_j w_{ij}\left\| \frac{\hat{y}_i}{\sqrt{d_i}} - \frac{\hat{y}_j}{\sqrt{d_j}} \right\|^2
$$

여기서 $i$는 현재 target frame의 픽셀이고, $j$는 과거 모든 프레임의 픽셀이다.

### 3.3 Label propagation과 similarity metric

실제 VOS 성능은 similarity metric $W$가 얼마나 좋은지에 크게 의존한다. 저자들은 similarity를 appearance term과 spatial term의 곱으로 정의한다.

$$
w_{ij} = \exp(f_i^T f_j)\cdot \exp\left(-\frac{\|loc(i)-loc(j)\|^2}{\sigma^2}\right)
$$

여기서 $f_i, f_j$는 CNN이 만든 pixel embedding이고, $loc(i)$는 픽셀의 spatial location이다.

이 식은 두 가지 의미를 가진다.

첫째, appearance가 비슷한 픽셀일수록 $f_i^T f_j$가 커져 affinity가 커진다. 즉 같은 객체의 비슷한 부분이 시간적으로 멀리 떨어져 있어도 연결될 수 있다.

둘째, 위치가 가까운 픽셀일수록 spatial prior 때문에 affinity가 커진다. 이는 segmentation boundary를 더 매끄럽게 만들고, 완전히 엉뚱한 곳으로 label이 튀는 것을 줄여준다.

### 3.4 Frame sampling 전략

모든 과거 프레임을 참조하면 너무 비싸므로, 저자들은 최근 40프레임 중 총 9프레임만 sampling한다. 구체적으로는 다음과 같다.

- target frame 바로 이전의 4개 프레임은 연속적으로 사용한다.
- 나머지 36프레임에서는 5개 프레임을 sparse하게 뽑는다.

이 전략의 목적은 분명하다. 최근 프레임들은 local motion을 잘 반영하고, 먼 과거 프레임들은 long-term appearance cue를 제공한다. 저자들은 이것이 efficiency와 effectiveness 사이에서 좋은 균형이라고 주장한다.

### 3.5 Simple motion prior

저자들은 temporal distance가 멀수록 spatial continuity가 약해진다고 본다. 그래서 spatial Gaussian의 폭을 시간 거리별로 다르게 둔다.

- local, dense reference frame에는 $\sigma = 8$
- distant reference frame에는 $\sigma = 21$

이 설계는 직관적이다. 가까운 프레임은 객체 위치가 크게 변하지 않았을 가능성이 높으므로 spatial prior를 강하게 두고, 먼 프레임은 위치가 많이 달라질 수 있으므로 spatial prior를 느슨하게 둔다. 논문에서는 이 단순한 motion prior만으로도 drifting을 줄이는 데 효과가 있다고 보고한다.

### 3.6 Appearance embedding 학습

pixel embedding은 2D CNN으로 학습한다. target pixel $x_i$에 대해, 과거 프레임의 reference pixel $x_j$들과의 similarity를 softmax로 정규화하여 label을 예측한다.

$$
\hat{y}_i = \sum_j \frac{\exp(f_i^T f_j)}{\sum_k \exp(f_i^T f_k)} \cdot y_j
$$

즉, target pixel은 과거 픽셀들의 label을 similarity 가중 평균한 값으로 예측된다. 이 식은 사실상 memory retrieval 또는 attention과 매우 유사한 구조지만, 논문은 이를 transductive label propagation의 관점에서 설명한다.

학습은 target frame의 모든 픽셀에 대해 standard cross-entropy loss를 사용한다.

$$
L = -\sum_i \log P(\hat{y}_i = y_i \mid x_i)
$$

즉, embedding이 잘 학습되면 같은 객체의 픽셀끼리는 feature space에서 가깝고, 다른 객체나 배경 픽셀은 멀어져서 label propagation이 정확해진다.

### 3.7 구현 세부사항

백본은 ResNet-50이다. 출력 해상도를 높게 유지하기 위해 3번째와 4번째 residual block의 stride를 1로 바꾸고, 마지막에 $1 \times 1$ convolution을 붙여 256차원 embedding을 만든다. 최종 feature stride는 8이다.

학습은 ImageNet pretrained weight로 시작한다. DAVIS 2017에서는 240 epochs, Youtube-VOS에서는 30 epochs 학습한다. augmentation은 random flipping과 $256 \times 256$ random cropping이다. optimizer는 SGD, 초기 learning rate는 $0.02$, scheduler는 cosine annealing이다. 학습은 4개의 Tesla P100 GPU에서 batch size 16으로 약 16시간 걸린다고 적혀 있다.

추론 시에는 480p 원본 해상도에서 feature를 뽑고, 최대 40프레임의 embedding history를 cache한다. 프레임당 계산은 backbone forward 한 번과, 현재 embedding과 과거 embedding 사이의 dot product 정도이므로 매우 빠르다. 또한 계산량이 객체 수에 따라 증가하지 않는다고 설명한다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 지표

실험은 DAVIS 2017과 Youtube-VOS에서 수행되었다.

DAVIS 2017은 150개 비디오 시퀀스로 구성되며, multiple objects, severe deformation, prolonged occlusion, fast motion 등 어려운 경우가 많다. 모든 프레임에 high-definition annotation이 있다.

Youtube-VOS는 4453개의 training sequence와 474개의 validation sequence를 가진 대규모 데이터셋이다. 94개의 일상 object category를 포함한다. 다만 frame rate가 DAVIS보다 낮아 5 fps 수준이라고 설명한다.

평가는 표준 VOS 지표인 $J$, $F$, 그리고 그 평균인 $J\&F$를 쓴다.

- $J$: region similarity, 즉 mean IoU 계열의 객체 전체 영역 정확도
- $F$: contour accuracy, 즉 경계 품질
- $J\&F$: 둘의 평균

Youtube-VOS에서는 추가로 seen / unseen object 분리 평가도 사용한다.

### 4.2 Ablation study

가장 중요한 ablation은 dense local and global dependencies의 효과를 보는 것이다. Table 2에 따르면, 과거를 더 길게 보고, 특히 최근 프레임을 촘촘하게 샘플링할수록 성능이 좋아진다.

예를 들어 DAVIS 2017 validation에서 mean $J$ 기준으로 보면, 단순한 설정보다 sparse long-range sampling과 motion prior를 함께 쓴 경우가 가장 좋다. 표의 최고 값은 `train / tracking = 9 frames / sparse + motion`에서 $69.9$이다. 이는 첫 프레임이나 짧은 최근 히스토리만 보는 것보다 long-term dependency가 실제로 도움이 된다는 근거다.

흥미로운 점은 embedding 학습 자체는 “너무 긴” temporal range가 항상 좋은 것은 아니라는 점이다. 저자들은 매우 긴 범위를 학습에 넣으면 비디오 전체를 거의 다 덮게 되어 training variation이 줄고 generalization이 나빠질 수 있다고 해석한다. 즉, inference에서는 긴 history가 유리하지만, training에서는 적절한 temporal diversity가 중요하다는 뜻이다.

또한 transferred representation 실험도 포함된다. ImageNet pretrained model을 DAVIS에 추가 학습하지 않아도 mean $J$가 $54.8\%$까지 나온다. 심지어 self-supervised pretraining인 InstDisc나 MoCo로도 꽤 경쟁력 있는 결과가 나온다. 이는 저자들의 transductive inference mechanism 자체가 강력하다는 주장에 힘을 실어준다.

simple motion prior는 약 $1\%$ 정도 추가 개선을 주었다고 보고한다. 수치는 크지 않지만, 설계가 매우 단순하다는 점을 생각하면 의미 있는 개선이다.

### 4.3 DAVIS 2017 결과

DAVIS 2017 validation set에서 TVOS는 다음 성능을 보였다.

- $J = 69.9$
- $F = 74.7$
- $J\&F = 72.3$
- 속도 = 37 fps

이 결과는 같은 범주의 non-finetuning propagation 기반 방법들과 비교할 때 매우 강하다. Table 3에서 STM이 $71.6$의 $J\&F$를 보이는데, TVOS가 약간 더 높다. FEELVOS는 $69.1$, RGMP는 $66.7$이다. 즉, 추가 모듈 없이도 propagation 계열에서는 매우 경쟁력 있는 수준이다.

finetuning 기반의 PReMVOS는 validation에서 $77.8$로 더 높지만, 이 방법은 optical flow, proposal, re-identification 등 여러 외부 모듈과 추가 데이터셋을 사용한다. 따라서 저자들은 직접적인 apples-to-apples 비교는 어렵다고 강조한다.

DAVIS 2017 test-dev에서는 TVOS가

- $J = 58.8$
- $F = 67.4$
- $J\&F = 63.1$
- 속도 = 37 fps

를 기록했다. 여기서는 DyeNet($68.2$), CNN-MRF($67.5$)보다 낮다. 저자들은 test-dev의 분포가 validation과 꽤 다르고, 같은 category 객체들 사이의 prolonged occlusion이 많아서 re-identification 모듈이 있는 방법이 더 유리하다고 해석한다. 이 설명은 논문 안에 명시되어 있지만, 정확히 어느 정도가 데이터 분포 차이 때문인지는 추가 실험 없이 단정할 수는 없다.

### 4.4 Youtube-VOS 결과

Youtube-VOS validation에서 TVOS는 overall $67.8$을 기록한다. 세부적으로는 다음과 같다.

- Overall: $67.8$
- Seen: $67.1$ / $69.4$
- Unseen: $63.0$ / $71.6$

표의 형식상 seen/unseen에 대해 $J$, $F$가 나뉘어 제시된 것으로 보이지만, 본문에 각 항목 라벨이 충분히 자세히 풀려 있지는 않다. 다만 overall 성능만 봐도 당시 이전 방법들을 대체로 앞서며, STM의 heavy pretraining 버전만이 훨씬 높은 $79.4$를 기록한다.

또한 DAVIS에서 학습한 모델을 Youtube-VOS에 바로 적용했을 때도 $67.4$의 overall score를 얻는다. 이는 dataset transfer/generalization이 꽤 좋음을 보여준다.

### 4.5 속도와 temporal stability

이 논문의 큰 장점 중 하나는 속도다. TVOS는 single Titan Xp GPU에서 37 fps로 동작한다고 보고한다. 이는 Table 3, 4 기준으로 다른 고성능 방법보다 압도적으로 빠르다. 예를 들어 PReMVOS는 0.03 fps, OnAVOS는 0.08 fps, STM은 6.25 fps 수준이다. 따라서 이 논문은 “실시간 또는 online에 가까운 VOS baseline”이라는 위치를 분명히 가진다.

저자들은 temporal stability에 대한 정식 정량평가는 제공하지 않지만, per-frame IoU 곡선을 통해 qualitative 비교를 한다. Figure 6에 따르면 PreMVOS는 object identity를 자주 바꾸거나, 갑자기 객체를 놓치는 경우가 있는 반면, TVOS는 더 temporally smooth한 예측을 보인다고 주장한다. 다만 이것은 정량 metric이 아니라 사례 기반 관찰이므로, 일반적인 결론으로 확대할 때는 주의가 필요하다.

### 4.6 Optical flow와의 관계

논문은 자신들의 pixel association이 optical flow와 얼마나 비슷한지도 간단히 분석한다. 인접 두 프레임에서 normalized similarity $s_{ij}$를 이용해 displacement의 가중합으로 pseudo-flow를 계산한다.

$$
\Delta d_i = \sum_j s_{ij}\Delta d_{ij}
$$

시각화 결과, 객체 영역에서는 꽤 말이 되는 flow가 나오지만 background에서는 noise가 많다. 저자들은 optical flow에서 흔히 쓰는 spatial smoothness constraint를 추가로 넣어보았지만, flow는 더 부드러워져도 VOS 성능은 오히려 consistently 나빠졌다고 말한다. 즉, 이 모델이 correspondence를 일부 학습하긴 하지만, 그것을 optical flow처럼 강하게 regularize하는 것이 segmentation 목표에는 반드시 유리하지 않다는 해석이 가능하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법론이 매우 정직하고 단순하다는 점이다. optical flow, proposal, re-ID 같은 외부 모듈 없이도 경쟁력 있는 성능을 보여 주며, 따라서 방법 자체의 기여를 보기 쉽다. 또한 transductive inference라는 고전적 semi-supervised learning 관점으로 VOS를 설명해 이론적 직관도 분명하다.

두 번째 강점은 long-term dependency를 효율적으로 다루는 설계다. 최근 프레임은 dense, 먼 과거는 sparse하게 쓰는 전략은 구현이 단순하면서도 실제로 occlusion 이후 재검출과 temporal consistency에 도움을 준다. ablation도 이 점을 비교적 설득력 있게 뒷받침한다.

세 번째 강점은 속도다. 37 fps는 논문에 제시된 비교군들 대비 매우 빠르며, online processing이 가능하다는 점에서 실제 응용 가능성이 높다. 또한 객체 수에 따라 계산량이 크게 늘지 않는다는 점도 practical하다.

한편 한계도 분명하다. 첫째, test-dev 성능이 validation보다 크게 떨어지고, 특히 re-identification module이 있는 방법들보다 불리하다. 이는 장기 occlusion이나 동일 category 다중 객체가 많은 상황에서 단순 similarity propagation만으로는 identity 유지가 충분하지 않을 수 있음을 시사한다.

둘째, temporal stability에 대한 주장은 주로 qualitative evidence에 의존한다. Figure 6은 인상적이지만, 정량 metric 없이 “더 안정적이다”라고 일반화하기에는 근거가 제한적이다.

셋째, sampling, motion prior, embedding 차원 등 여러 설계가 경험적으로는 잘 동작하지만, 왜 최적인지에 대한 이론적 분석은 깊지 않다. 예를 들어 왜 40프레임 history와 9-frame sampling이 가장 적절한지, 더 긴 범위에서 어떤 조건에서 일반화가 깨지는지까지는 논문이 명확히 설명하지 않는다.

넷째, 논문은 STM과의 차이를 강조하지만, dense long-term memory 활용이라는 고수준 아이디어는 상당히 가깝다. 저자들도 이를 인정한다. 따라서 이 논문의 핵심 기여는 완전히 새로운 메커니즘이라기보다, transductive formulation과 단순하고 빠른 구현에 있다고 보는 편이 더 정확하다.

## 6. 결론

이 논문은 semi-supervised video object segmentation을 transductive inference 관점에서 재정의하고, long-term spatio-temporal dependency를 활용하는 간단한 label propagation 방법을 제안한다. 핵심은 픽셀 embedding 기반 similarity graph를 이용해 첫 프레임의 annotation을 이후 프레임으로 전파하되, 최근 프레임과 먼 과거 프레임을 함께 참조하여 local motion과 global appearance를 동시에 반영하는 것이다.

결과적으로 TVOS는 추가 모듈이나 복잡한 architecture 없이도 DAVIS 2017 validation에서 $72.3$의 $J\&F$, test-dev에서 $63.1$, Youtube-VOS validation에서 $67.8$을 달성했고, 동시에 37 fps라는 매우 빠른 속도를 보여준다. 최고 성능만 놓고 보면 일부 복잡한 방법이나 heavy pretraining 방법에 못 미치지만, 단순성, 공정성, 속도, 재현 가능성 면에서 매우 가치 있는 baseline이다.

실제 적용 측면에서는 online video editing, robotics, surveillance처럼 빠른 처리와 안정적인 object tracking이 필요한 환경에서 의미가 있다. 향후 연구 측면에서는 이 방법 위에 더 강한 re-identification, memory selection, uncertainty handling, multi-object identity preservation 기법을 얹는 방향이 자연스럽다. 즉, 이 논문은 “복잡한 외부 모듈 없이도 어디까지 가능한가”를 분명히 보여 준 기준점으로 볼 수 있다.
