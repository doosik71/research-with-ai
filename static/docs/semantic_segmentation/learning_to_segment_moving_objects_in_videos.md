# Learning to Segment Moving Objects in Videos

- **저자**: Katerina Fragkiadaki, Pablo Arbeláez, Panna Felsen, Jitendra Malik
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1412.6504

## 1. 논문 개요

이 논문은 비디오에서 움직이는 물체를 분할하는 문제를 다룬다. 목표는 각 비디오에서 moving object를 잘 포함하는 영역들을 제안하고, 그중 좋은 후보를 높은 순위로 정렬한 뒤, 이를 시간 축으로 확장하여 spatio-temporal tube 형태의 물체 분할 결과를 얻는 것이다. 저자들은 이 문제를 단순한 bottom-up motion segmentation이나 특정 클래스 detector를 사용하는 tracking 문제로 보지 않고, “object proposal + objectness ranking”이라는 정적 이미지 검출의 성공적인 패러다임을 비디오의 moving object segmentation으로 옮겨오려 한다.

연구 문제는 다음과 같이 정리할 수 있다. 첫째, 한 프레임 안에서 움직이는 물체를 잘 덮는 segment proposal을 어떻게 만들 것인가. 둘째, optical flow 기반 경계는 종종 실제 물체 경계와 어긋나는데, 이런 noisy motion cue를 어떻게 활용할 것인가. 셋째, 한 프레임에서 얻은 segmentation을 여러 프레임에 걸친 tube로 어떻게 안정적으로 확장할 것인가. 넷째, 수많은 후보 중 실제 moving object를 잘 담는 후보를 어떻게 랭킹할 것인가.

이 문제가 중요한 이유는, 비디오 이해에서 moving object segmentation은 추적, 행동 분석, 장면 이해의 핵심 중간 표현이기 때문이다. 특히 monocular, uncalibrated video 환경에서는 깊이 정보나 다중 시점 정보 없이 appearance와 motion만으로 물체를 분리해야 하므로 난도가 높다. 저자들은 VSB100과 Moseg라는 당시 대표적인 비디오 segmentation benchmark에서 적은 수의 proposal만으로도 높은 IoU를 달성하는 것이 중요하다고 본다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 단계로 요약된다. 먼저 각 프레임에서 optical flow boundary를 이용해 moving object proposal을 생성한다. 그 다음 RGB와 optical flow를 함께 입력받는 CNN 기반 Moving Objectness Detector, 즉 MOD를 사용해 proposal을 점수화하고, 상위 proposal을 dense point trajectory 기반 random walk로 시간축 방향으로 확장하여 tube를 만든다.

이 설계의 핵심 직관은 static boundary와 motion boundary가 서로 다른 실패 모드를 가진다는 점이다. 정적 이미지 기반 segmentation은 사람 옷의 무늬나 내부 contour 같은 강한 내부 경계 때문에 물체를 조각내기 쉽다. 반대로 motion은 물체 내부에서 비교적 부드럽게 유지되므로 이러한 내부 경계를 억제하고 object-level grouping에 유리할 수 있다. 저자들은 optical flow를 static boundary와 억지로 결합하기보다, flow boundary를 독립적으로 segmentation의 입력으로 사용해 별도의 proposal pool을 만드는 편이 더 낫다고 주장한다.

기존 접근과의 차별점도 분명하다. bottom-up supervoxel이나 trajectory clustering 방법은 학습 없이 color/motion similarity만 활용하는 경향이 있었고, tracking-by-detection 계열은 car, pedestrian 같은 특정 카테고리 detector에 의존했다. 이 논문은 학습 기반 moving objectness를 도입하되, 특정 객체 클래스에 묶이지 않고 “움직이는 물체”라는 더 넓은 개념을 학습한다. 또한 단일 segmentation을 산출하려 하기보다 proposal set을 만들고 이를 ranking하는 접근을 취한다는 점에서 object proposal 계열과 더 가깝다.

## 3. 상세 방법 설명

전체 파이프라인은 다음 순서로 구성된다. 각 프레임에서 optical flow를 계산하고, 그 크기(magnitude) 영상에 structured forest boundary detector를 적용하여 flow boundary를 얻는다. 이 경계 지도를 이용해 multiple figure-ground segmentation을 수행하여 per-frame Moving Object Proposals, 즉 MOPs를 생성한다. 동시에 static proposal도 함께 고려할 수 있다. 그런 다음 MOD가 각 proposal이 moving object를 얼마나 잘 포함하는지 점수화한다. 이후 상위 proposal을 dense point trajectory 공간으로 투영하고, trajectory 간 motion affinity 위에서 random walker를 수행해 foreground/background label을 확산시킨다. 마지막으로 trajectory cluster를 supervoxel에 투영하여 pixel-level spatio-temporal tube를 얻고, tube lifespan 전반의 score를 합산해 최종 ranking을 만든다.

프레임 단위 proposal 생성에서 중요한 점은 optical flow boundary를 직접 segmentation 입력으로 쓴다는 것이다. 저자들은 Brox and Malik의 large displacement optical flow를 사용하고, flow magnitude를 3채널로 복제한 뒤 Dollár and Zitnick의 structured forest boundary detector를 적용한다. detector는 원래 static image boundary용으로 학습되었지만, 논문은 이를 flow magnitude에도 효과적으로 적용할 수 있다고 보고 재학습은 하지 않았다. 이유는 flow bleeding의 정도가 장면과 배경 텍스처에 따라 크게 달라져 detector를 혼란스럽게 할 수 있다고 판단했기 때문이다.

이 boundary map 위에서 Geodesic Object Proposals(GOP) [22] 방식을 사용해 다중 figure-ground segmentation을 수행한다. GOP는 randomized seed placement와 shortest path 기반 superpixel 분류를 통해 다양한 segment proposal을 만든다. 여기서 static boundary 대신 flow boundary를 사용하면, 물체 내부의 불필요한 경계는 약해지고 물체 외곽 motion discontinuity는 강화되어 moving object를 더 잘 덮는 proposal을 생성할 수 있다.

Moving Objectness Detector는 dual-pathway CNN이다. 하나의 경로는 RGB 이미지를, 다른 경로는 optical flow field를 입력받는다. flow 입력은 $x$ displacement, $y$ displacement, flow magnitude의 3채널로 구성된다. 각 경로의 네트워크 구조는 거의 동일하며, convolution, ReLU, max pooling, normalization, fully connected layer로 구성된다. 논문이 제시한 각 stack의 구조는 다음과 같다.

$$
\begin{aligned}
& C(7,96,2) - RL - P(3,2) - N - C(5,384,2) - RL - P(3,2) - N \newline
& - C(3,512,1) - RL - C(3,512,1) - RL - C(3,384,1) - RL  \newline
& - P(3,2) - FC(4096) - RL - D(0.5) - FC(4096) - RL
\end{aligned}
$$

두 경로의 `relu7` feature를 concatenate한 뒤, 마지막 layer가 입력 bounding box와 ground-truth segment 사이의 IoU를 회귀(regression)하도록 학습된다. 즉 이 네트워크는 단순한 binary classification이 아니라 “이 proposal이 실제 moving object segmentation과 얼마나 잘 맞는가”를 연속값으로 예측한다. 논문은 이 점이 proposal ranking에 유리하다고 본다.

초기 가중치는 Girshick 등의 200-category detection network에서 가져온다. 이는 moving object category가 ImageNet detection set에 충분히 포함되어 있고, detection network가 classification network보다 objectness 개념을 더 잘 담고 있을 것이라는 판단 때문이다. 이후 VSB100과 Moseg의 training set에서 모은 moving object box와 background box를 이용해 fine-tuning한다. 학습은 Caffe에서 SGD with momentum으로 수행했다. 정확한 learning rate, batch size, epoch 수 등은 제공된 본문에 명시되어 있지 않다.

tube proposal 생성 단계에서는 dense point trajectory를 사용한다. trajectory는 optical flow field를 연결하여 만들고, forward-backward consistency가 깨지면 종료한다. 따라서 occlusion, dis-occlusion, 저텍스처 구간에서는 trajectory가 짧아질 수 있다. trajectory 집합을 $T$, trajectory 수를 $n = |T|$라고 두고, 각 trajectory 쌍 사이의 motion similarity를 affinity matrix $A \in [0,1]^{n \times n}$로 정의한다. affinity는 두 trajectory의 maximum velocity difference에 기반하며, 시간적으로 겹치고 공간적으로 60 pixel 이내인 trajectory 쌍에 대해서만 계산한다.

어떤 per-frame MOP가 frame $t_i$에서 검출되면, 그 프레임을 통과하는 trajectory들을 foreground 또는 background로 표시한다. 이를 통해 label vector $x \in \{0,1\}^n$를 정의하며, foreground trajectory set을 $F$, background set을 $B$, labeled trajectory set을 $M = F \cap B$, unlabeled trajectory set을 $U = T \setminus M$이라 둔다. 논문은 random walker energy를 최소화한다.

$$
\min_x \frac{1}{2} x^T L x \quad \text{subject to } x_B = 0, \; x_F = 1
$$

여기서 $L = \text{Diag}(A1_n) - A$는 unnormalized Laplacian이다. 이 목적함수는 결국 affinity가 큰 trajectory끼리는 유사한 label을 갖도록 만든다. 즉

$$
x^T L x \equiv \sum_{i,j} A_{ij}(x_i - x_j)^2
$$

를 최소화하는 형태이므로, motion이 비슷한 trajectory는 같은 foreground/background 쪽으로 묶이게 된다.

이산 label 대신 연속값 $x \in [0,1]^n$로 relax하면, unlabeled trajectory에 대해 다음 선형 시스템을 얻는다.

$$
L_U x_U = -L_{MU}^T x_M
$$

논문은 이 닫힌형 해를 직접 풀기보다 normalized affinity matrix를 이용한 반복 diffusion으로 근사한다.

$$
x' = \text{Diag}(A1_n)^{-1} A x
$$

저자들은 약 50번 diffusion이면 충분하다고 보고한다. 이렇게 얻은 trajectory label을 supervoxel에 투영해 pixel-level tube를 복원한다. supervoxel의 weight는 trajectory cluster와의 IoU로 정하고, weighted average를 thresholding해 binary spatio-temporal segmentation을 만든다. 이 단계는 초기 MOP의 경계가 다소 어긋나 있었더라도 최종 tube에서 sharper boundary를 회복하도록 도와준다.

ranking은 tube 전체 수명 동안의 box score를 합산하여 계산한다. 평균이 아니라 합을 쓰는 이유는 더 긴 tube를 선호하기 위해서라고 명시한다. 또한 다양한 tube proposal 사이의 중복을 줄이기 위해 soft non-max suppression 성격의 score diversification을 사용했다고 설명한다.

## 4. 실험 및 결과

실험은 VSB100과 Moseg 두 benchmark에서 수행되었다. VSB100은 YouTube에서 수집된 100개 시퀀스로 구성되며, 40개 training, 60개 testing으로 나뉜다. 고해상도이고 object motion이 매우 미묘하거나 강하게 articulated한 경우가 많으며, parade, cycling race, beach volley, ballet, salsa dancing 같은 crowded scene도 포함한다. 저자들은 이 중 moving object segmentation과 관련된 “rigid and non-rigid motion subtasks”에 집중한다. Moseg는 59개 비디오로 구성되며 상대적으로 uncluttered scene이 많고, 평균적으로 한두 개 정도의 moving object가 등장한다.

motion segmentation 평가에서는 제안 방법 전체와 기존 방법 [12], [39], [28]을 비교한다. 성능 지표는 비디오별 proposal 수 대비 ground-truth spatio-temporal segment와의 평균 IoU이다. 저자들의 full method는 MOP tubes와 multiscale trajectory clusters의 union으로 구성된다. 결과적으로 이 방법은 VSB100과 Moseg 모두에서 어떤 proposal 수 구간에서도 baseline보다 높은 ground-truth coverage를 보였다. 특히 VSB100처럼 articulated motion과 clutter가 강한 어려운 데이터셋에서 이득이 더 중요하게 나타난다. 논문은 VSB100에서 64개에서 1000개 tube proposal만으로도 ground-truth object의 약 55%에서 65%를 포착한다고 서술한다.

per-frame static segmentation 관점에서도 MOP의 효과를 측정했다. 사용된 지표는 average best overlap, coverage, detection rate at 50%, detection rate at 70%, 그리고 tube lifespan 전체에서 최고 overlap을 보는 anytime-best 변형들이다. 결과 표를 보면 VSB100에서 GOP 단독은 detection rate 50%가 60.34, 70%가 26.12인데, GOP+MOP를 사용하면 각각 66.48, 31.50으로 오른다. 즉 50% 기준에서 약 6%, 70% 기준에서 약 5% 향상이다. Moseg에서는 GOP 단독의 detection rate 70%가 64.54이고 GOP+MOP는 70.21로 약 5% 상승한다. 저자들은 GOP의 proposal 수를 늘리는 것만으로는 이런 개선이 나오지 않으며, GOP는 약 2500 proposals/frame 부근에서 saturation된다고 설명한다. 이는 motion boundary 기반 proposal이 static proposal과 상보적임을 보여준다.

proposal ranking 실험에서는 VSB100에서 per-frame MOP와 spatio-temporal tube를 대상으로 MOD의 효과를 분석한다. 비교 대상은 dual-pathway regression CNN, dual-pathway classification CNN, image-only CNN, flow-only CNN, center-surround saliency, LSDA 기반 static objectness detector다. 결과는 dual-pathway regression CNN이 가장 좋은 ranking 성능을 보인다는 것이다. dual-pathway classification CNN도 근접하지만, 회귀 방식이 더 낫다. 이는 proposal의 quality를 연속적인 IoU 점수로 학습하는 것이 ranking에 더 적합하다는 해석이 가능하다. 또한 segmentation mask를 직접 입력으로 쓰지 않고 bounding box만 사용했는데, 저자들은 under-segmentation과 over-segmentation을 판단하려면 주변 context가 중요하기 때문이라고 설명한다.

failure case 분석도 포함된다. VSB100에서는 큰 motion이나 완전한 occlusion 때문에 tube가 시간적으로 잘리는 temporal fragmentation이 주요 실패 원인이다. 저자들은 appearance가 비슷한 tube를 후처리로 링크하는 단계가 추가되면 도움이 될 수 있지만, 방법의 단순성을 위해 넣지 않았다고 말한다. Moseg에서는 trajectory cluster를 pixel tube로 매핑하는 과정에서 약간의 background leakage가 남는 경우가 많았고, 특히 camel처럼 가는 다리를 가진 동물에서 문제가 두드러졌다고 한다.

계산 시간도 제시된다. single CPU 기준으로 optical flow는 이미지당 평균 16초, MOP 생성은 700x1000 이미지당 4초, 각 MOP의 trajectory embedding 투영은 70000 trajectories 기준 2초, supervoxel 계산은 frame당 7초, 70000 trajectories의 motion affinity 계산은 비디오당 15초가 든다고 한다. 저자들은 여러 단계가 병렬화 가능하다고 설명하지만, 전체 end-to-end 처리 시간이 얼마나 되는지는 명시하지 않았다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 moving object segmentation을 proposal generation과 learned objectness ranking으로 재정의했다는 점이다. 이는 기존의 pure bottom-up segmentation이나 category-specific tracking 사이의 중간 지점을 잘 포착한 설계다. 특히 flow boundary를 static boundary와 섞으려 하지 않고, 독립적인 proposal source로 취급한 판단이 실험적으로도 설득력 있다. cluttered scene에서 static proposal의 saturation을 motion proposal이 넘어서게 만든 점은 논문의 핵심 실증 결과다.

또 다른 강점은 per-frame segmentation과 temporal propagation을 서로 다른 표현으로 분리한 점이다. 프레임 수준에서는 경계 기반 figure-ground segmentation을 사용하고, 시간 확장에서는 dense trajectory affinity 위의 diffusion을 사용한다. 이 조합은 한 프레임에서 motion이 강하게 드러나는 “lucky frame”에서 시작한 segmentation을 motion이 약한 프레임까지 퍼뜨릴 수 있게 한다. 논문이 말하듯 motion은 opportunistic cue인데, 이 점을 잘 활용한 구조다.

MOD 역시 중요한 기여다. RGB와 flow를 함께 보는 dual-pathway CNN이 image-only나 flow-only보다 낫고, classification보다 IoU regression이 더 낫다는 결과는 이후 proposal ranking 계열 연구와도 잘 맞는 방향성이다. “moving objectness”를 특정 클래스 detector가 아니라 범주 독립적 개념으로 학습했다는 점도 실용적이다.

한계도 분명하다. 첫째, optical flow 품질에 강하게 의존한다. 논문 스스로도 flow bleeding, occlusion, dis-occlusion, 저텍스처 영역이 문제라고 설명한다. 둘째, trajectory가 짧아지거나 끊기는 상황에서 temporal fragmentation이 쉽게 발생한다. 셋째, 최종 pixelization 단계에서 supervoxel projection이 배경 leakage를 일으킬 수 있다. 특히 얇은 구조를 가진 물체에서 이 문제가 심하다. 넷째, ranking에 bounding box를 사용하기 때문에 mask 자체의 형상을 직접 모델링하지는 않는다.

비판적으로 보면, 이 방법은 당시로서는 매우 강력하지만 여러 복잡한 모듈의 조합이라는 성격이 강하다. optical flow, boundary detection, proposal generation, CNN ranking, trajectory graph diffusion, supervoxel projection이 차례로 연결되기 때문에 어느 한 부분의 품질 저하가 전체 성능에 영향을 줄 수 있다. 또한 제공된 본문 기준으로는 학습 세부 설정, 데이터 증강, proposal 수 선택 기준 등의 구현 디테일이 충분히 상세하지 않아 재현성 관점에서는 추가 정보가 필요하다. 다만 이런 점은 본문 발췌 범위의 한계일 수 있으며, 논문에서 명확히 주어지지 않은 부분은 여기서 추측할 수 없다.

## 6. 결론

이 논문은 비디오 moving object segmentation을 위해 motion boundary 기반 proposal, learned moving objectness ranking, trajectory affinity 기반 spatio-temporal propagation을 결합한 방법을 제안했다. 핵심 기여는 세 가지로 요약된다. 첫째, optical flow boundary 위의 multiple segmentation으로 per-frame MOP를 만든 점. 둘째, RGB와 flow를 함께 사용하는 MOD로 proposal을 효과적으로 정렬한 점. 셋째, dense trajectory 위의 random walk를 통해 프레임 단위 segment를 tube로 확장한 점이다.

실험적으로는 VSB100과 Moseg에서 당시 대표적 supervoxel 및 trajectory clustering 기반 방법보다 더 높은 ground-truth coverage를 보였고, static proposal에 MOP를 더하면 moving object detection rate가 유의미하게 향상됨을 보였다. 따라서 이 연구는 video segmentation과 tracking 사이를 잇는 방향을 제시한 작업으로 볼 수 있다. 실제 적용 측면에서는 비디오 객체 추출, 전경-배경 분리, 후속 tracking 및 video understanding 파이프라인의 전처리로 의미가 크며, 향후 연구 측면에서는 learned proposal scoring, multimodal appearance-motion modeling, long-term temporal linking 같은 주제의 기반이 되는 논문이라고 평가할 수 있다.
