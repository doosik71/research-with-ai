# Improving Semantic Segmentation via Video Propagation and Label Relaxation

- **저자**: Yi Zhu, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1812.01593

## 1. 논문 개요

이 논문은 semantic segmentation에서 가장 큰 병목 중 하나인 **pixel-wise annotation 비용**을 줄이면서도 성능을 높이는 방법을 다룬다. 특히 Cityscapes처럼 비디오 시퀀스 기반으로 수집되었지만 일부 프레임에만 사람이 라벨을 붙인 데이터셋에서는, 라벨이 없는 중간 프레임이 많다. 저자들은 이 점에 주목하여, **video prediction/reconstruction 모델이 학습한 motion transformation을 이용해 새로운 image-label pair를 합성**하고, 이를 segmentation 학습에 활용한다.

연구 문제는 명확하다. semantic segmentation은 고성능을 위해 많은 정답 라벨이 필요하지만, 경계까지 정교하게 주석을 다는 작업은 매우 비싸고 어렵다. 기존에는 coarse annotation, synthetic data, teacher model 기반 pseudo-label, optical flow 기반 label propagation 등이 쓰였지만, 각각 annotation 품질, domain gap, teacher 성능 한계, alignment 오류 같은 문제가 있었다. 이 논문은 이런 한계를 줄이기 위해 **비디오 예측 모델이 배운 변형을 라벨 전파에 재사용**하고, 더 나아가 **image와 label을 함께 propagation**해서 misalignment를 줄이는 전략을 제안한다.

이 문제가 중요한 이유는 자율주행과 같은 실제 응용에서 segmentation 데이터셋 구축 비용이 매우 크기 때문이다. 논문은 단순히 데이터 수를 늘리는 것이 아니라, 실제로 학습 가능한 품질의 augmentation sample을 생성하는 방법을 제안하고, 그 결과 Cityscapes, CamVid, KITTI에서 모두 강한 성능 향상을 보인다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 가지다.

첫째, **video prediction 또는 video reconstruction 모델이 예측한 motion vectors를 이용해 미래 프레임뿐 아니라 미래 라벨도 함께 생성**하는 것이다. 보통 video model은 다음 프레임을 만들기 위해 현재 프레임의 픽셀을 어디로 이동시킬지 학습한다. 저자들은 이 이동 정보를 이미지에만 쓰지 않고, 현재의 semantic label map에도 동일하게 적용해 미래 시점의 label map을 만든다. 이렇게 하면 사람이 직접 라벨링하지 않은 프레임에도 합성 라벨을 붙여 학습 샘플을 늘릴 수 있다.

둘째, **joint image-label propagation**이다. 기존의 label propagation은 보통 전파된 label $\tilde{L}_{t+k}$를 실제 미래 이미지 $I_{t+k}$와 짝지어 학습한다. 그런데 motion 추정이 완벽하지 않으면 label과 image가 미세하게 어긋난다. segmentation은 픽셀 단위 정합이 매우 중요하므로 이 misalignment가 성능을 갉아먹는다. 저자들은 이를 해결하기 위해 이미지도 같이 전파해서 $(\tilde{I}_{t+k}, \tilde{L}_{t+k})$를 한 쌍으로 사용한다. 같은 transformation으로 이미지와 라벨을 동시에 이동시키므로 서로 정렬이 잘 맞는다.

여기에 더해, 객체 경계에서 annotation 자체가 애매하거나 propagation artifact가 생기는 문제를 해결하기 위해 **boundary label relaxation**을 도입한다. 핵심은 경계 픽셀에서 정답을 단일 class one-hot으로 강제하지 않고, 주변 $3 \times 3$ 이웃에 존재하는 클래스들의 합집합을 정답으로 허용하는 것이다. 즉, 경계에서 class A인지 B인지 애매하면 $P(A \cup B)$를 높이도록 학습한다. 이는 hard boundary supervision을 완화하여 noisy label에 더 강인한 학습을 가능하게 만든다.

기존 접근과의 차별점은 논문이 직접 명시한다. optical flow 대신 **self-supervised video prediction이 학습한 motion vectors**를 사용하고, 단순 label warping이 아니라 **joint propagation**으로 alignment 문제를 줄였으며, 경계 처리에서도 uncertainty를 logit 분포나 edge cue로 우회적으로 다루는 대신 **label space 자체를 완화**했다는 점이 핵심 차별점이다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같이 이해할 수 있다. 입력은 비디오 프레임 시퀀스 $I \in \mathbb{R}^{n \times W \times H}$와 일부 프레임에만 존재하는 semantic label $L \in \mathbb{R}^{m \times W \times H}$이다. 여기서 $m \le n$이다. 각 라벨된 프레임 $(I_i, L_i)$에 대해, 논문은 forward 혹은 backward 방향으로 $k$ step propagation을 수행하여 새로운 training pair를 만든다. 이 과정을 통해 데이터셋을 최대 $2k+1$배까지 확장할 수 있다고 설명한다.

### Video prediction 기반 propagation

논문은 vector-based video prediction 모델을 사용한다. 이 모델은 직접 픽셀을 생성하는 대신, 현재 프레임의 각 픽셀이 미래 프레임에서 어디로 이동할지를 나타내는 motion vector $(u,v)$를 예측한다. 미래 프레임은 다음 식으로 주어진다.

$$
\tilde{I}_{t+1} = T\big(G(I_{1:t}, F_{2:t}), I_t\big)
$$

여기서 $G$는 3D CNN이며, 과거 프레임 $I_{1:t}$와 연속 프레임 간 optical flow $F_{2:t}$를 입력으로 받아 motion vector를 예측한다. $T$는 bilinear sampling 연산으로, 마지막 입력 프레임 $I_t$를 예측된 motion vector에 따라 warping한다.

중요한 점은 이 motion vector가 optical flow 그 자체는 아니라는 것이다. 논문은 optical flow는 disocclusion 상황에서 정의되지 않거나 foreground duplication, hole, border stretch 문제를 만들 수 있다고 설명한다. 반면 video prediction이 학습한 motion vector는 미래 프레임 재구성을 목표로 하므로 이런 현상을 더 잘 다룬다고 주장한다.

이제 같은 motion vector를 label map에도 적용하면 미래 라벨을 만들 수 있다.

$$
\tilde{L}_{t+1} = T\big(G(I_{1:t}, F_{2:t}), L_t\big)
$$

즉, 영상 예측을 위해 학습된 transformation을 semantic label propagation에도 재사용한다.

### Joint image-label propagation

단순 label propagation은 새 학습 샘플을 $(I_{i+k}, \tilde{L}_{i+k})$로 만든다. 그러나 motion 추정 오차가 있으면 실제 프레임과 전파된 라벨이 안 맞을 수 있다. 논문은 특히 pole, pedestrian leg 같은 thin structure에서 misalignment가 두드러진다고 시각적으로 보인다.

이를 줄이기 위해 제안한 것이 joint propagation이다. 이미지와 라벨을 모두 같은 transformation으로 전파해서 $(\tilde{I}_{i+k}, \tilde{L}_{i+k})$를 사용한다. 이렇게 하면 왜곡이 있더라도 image와 label이 함께 왜곡되기 때문에 서로의 정합이 유지된다. 저자들은 이를 일종의 data augmentation으로 해석한다. 랜덤 회전이나 스케일 조정처럼 image-label 쌍 전체에 같은 변환을 거는 것과 비슷하지만, 여기서 쓰는 변환은 **future frame prediction을 위해 학습된 motion-aware transformation**이라는 점이 다르다.

또한 논문은 backward propagation도 사용한다. 즉, 미래뿐 아니라 과거 방향으로도 샘플을 합성해 데이터셋을 더 확장한다. 실험에서는 $k = \pm 1, \pm 2, \pm 3, \pm 4, \pm 5$를 사용했다.

### Video reconstruction

논문은 실제 미래 프레임을 알고 있는 상황에서는 prediction보다 reconstruction이 더 좋을 수 있다고 본다. video prediction은 과거만 보고 미래를 예측하지만, video reconstruction은 과거와 미래를 모두 보고 현재 프레임을 어떻게 warp하면 목표 프레임을 잘 설명할 수 있을지 학습한다. 식은 다음과 같다.

$$
\hat{I}_{t+1} = T\big(G(I_{1:t+1}, F_{2:t+1}), I_t\big)
$$

여기서는 미래 프레임 $I_{t+1}$도 입력에 포함된다. 이로부터 얻은 transformation으로 미래 라벨 $\hat{L}_{t+1}$도 생성한다. 논문의 주장대로라면 reconstruction은 미래 정보를 직접 보기 때문에 prediction보다 더 정확한 motion parameter를 줄 가능성이 높다.

부록에 따르면 video prediction 모델의 구체적 구조는 SDC-Net 계열의 fully convolutional U-Net이다. encoder 10층, decoder 6층이며 skip connection을 가진다. prediction 모델 입력은 $I_{t-1}, I_t, F_t$의 총 8채널이고, reconstruction 모델은 여기에 $I_{t+1}, F_{t+1}$를 더해 13채널 입력을 사용한다. 출력은 2채널 motion vector다.

### Boundary label relaxation

이 논문의 또 다른 핵심은 경계 픽셀 처리다. 저자들은 가장 어려운 픽셀들이 객체 경계에 있으며, annotation도 경계에서는 pixel-perfect하지 않다고 지적한다. propagation을 거치면 경계 왜곡은 더 심해진다. 그래서 경계에서 기존 one-hot cross-entropy를 그대로 쓰면, 사실상 애매한 정답을 지나치게 강하게 강제하게 된다.

논문은 경계 픽셀을 “서로 다른 라벨을 가진 이웃이 있는 픽셀”로 정의한다. 예를 들어 어떤 픽셀이 클래스 A와 B의 경계라면, 기존 방식은 A 또는 B 하나만 정답으로 둔다. 하지만 저자들은 $P(A \cup B)$를 최대화하자고 제안한다. A와 B는 mutually exclusive이므로

$$
P(A \cup B) = P(A) + P(B)
$$

가 된다. 이를 일반화해서, 한 픽셀 주변 $3 \times 3$ window 안에 존재하는 클래스 집합을 $N$이라고 두면, 경계 loss는

$$
L_{\text{boundary}} = - \log \sum_{C \in N} P(C)
$$

가 된다.

이 식의 의미는 직관적이다. 경계 픽셀에서 모델이 주변 클래스들 중 하나를 강하게 예측하면 충분하다고 보는 것이다. 만약 $|N| = 1$이면 일반적인 one-hot cross-entropy와 동일해진다. 따라서 이 방법은 기존 loss를 완전히 대체하는 것이 아니라, 경계에서만 label space를 완화하는 방식이다.

논문은 이 기법이 두 가지에 모두 도움이 된다고 주장한다. 첫째, 사람 주석의 경계 노이즈에 강해진다. 둘째, propagation으로 생기는 경계 artifact에도 강해진다. 실험 결과에서도 propagation 길이가 길어질수록 relaxation의 이점이 커졌다.

### Segmentation 학습 설정

semantic segmentation 모델은 DeepLabV3+ 기반이며 output stride는 8이다. ablation에서는 ResNeXt50 backbone, 최종 제출에는 WideResNet38 backbone을 사용했다. 학습은 SGD와 polynomial learning rate policy를 사용하며, 초기 learning rate는 0.002, power는 1.0, momentum은 0.9, weight decay는 0.0001이다. synchronized batch normalization을 사용했고, batch size는 16, GPU는 8개의 V100이다. 학습 epoch는 Cityscapes 180, CamVid 120, KITTI 90이다.

추가로 두 가지 강한 baseline 기법을 사용했다. 하나는 **Mapillary Vistas pre-training**이고, 다른 하나는 **class uniform sampling**이다. 후자는 희소 클래스가 epoch마다 비슷한 빈도로 crop에 등장하도록 centroid 기반 sampling을 섞는 방식이다. Cityscapes에서는 coarse annotation 20K도 이 sampling 전략에 맞춰 일부 클래스 보강용으로 사용했다.

## 4. 실험 및 결과

논문은 Cityscapes, CamVid, KITTI 세 데이터셋에서 평가했고, 기본 지표는 모두 mIoU이다. KITTI에서는 추가적으로 IoU class, iIoU class, IoU category, iIoU category도 보고한다.

### Cityscapes

Cityscapes는 2975 train, 500 val, 1525 test의 fine annotation과 20K coarse annotation을 포함하며, 해상도는 $1024 \times 2048$이다. 클래스는 19개 semantic class와 void class다.

먼저 baseline 강화 실험에서, 기본 baseline 76.60% mIoU에 대해 Mapillary pre-training을 적용하면 78.32%로 $+1.72$%p 향상되었고, class uniform sampling을 추가하면 79.46%로 다시 $+1.14$%p 향상되었다. 이후의 모든 비교는 이 강한 baseline 위에서 이루어진다.

#### Label Propagation vs Joint Propagation

video prediction 기반 motion vector를 사용하여 비교했을 때, joint propagation은 모든 propagation length에서 label propagation보다 좋았다. 가장 좋은 결과는 $\pm 1$에서 나왔고, label propagation은 79.79%, joint propagation은 80.26%였다. baseline 79.46% 대비 각각 $+0.33$%p, $+0.80$%p 향상이다. 이는 alignment가 실제로 중요하다는 논문의 주장을 잘 뒷받침한다.

#### Video Prediction vs Video Reconstruction

joint propagation을 고정하고 비교하면, video reconstruction이 video prediction보다 모든 propagation length에서 우수했다. reconstruction + joint propagation은 $\pm 1$에서 80.54%를 기록해 baseline 대비 $+1.08$%p 향상되었다. propagation이 $\pm 4$ 이상으로 길어지면 성능이 다시 떨어졌는데, 논문은 이것을 propagation 품질 저하로 해석한다.

#### Boundary Label Relaxation 효과

boundary label relaxation은 이 논문의 중요한 포인트다. video reconstruction 기반에서 relaxation을 쓰지 않으면 최고 성능은 $\pm 1$의 80.54%였다. relaxation을 추가하면 최고점이 $\pm 3$에서 81.35%로 올라간다. 즉, $+0.81$%p 향상이며, 더 긴 propagation에서도 성능 저하를 늦춰준다.

더 중요한 결과는 **propagation을 전혀 쓰지 않아도** relaxation이 큰 효과를 낸다는 점이다. $k=0$일 때 baseline 79.46%가 relaxation으로 80.85%가 된다. 즉, 이 기법은 propagation artifact 보정뿐 아니라 일반적인 segmentation annotation 경계 모호성에도 도움이 된다고 볼 수 있다.

#### Learned Motion Vectors vs Optical Flow

논문은 FlowNet2 optical flow와 비교한다. 정성적으로는 optical flow가 occlusion에서 dragging, doubling artifact를 만든다고 보여준다. 정량적으로도 learned motion vectors가 모든 propagation length에서 더 좋았고, 특히 $\pm 1$, $\pm 4$, $\pm 5$에서는 FlowNet2 기반 결과가 baseline보다도 낮았다. 이 실험은 “왜 optical flow 대신 video model 기반 motion을 쓰는가”에 대한 직접적인 근거다.

#### 최종 성능

Cityscapes validation에서 baseline 79.5%, VRec with JP 80.5%, 여기에 label relaxation까지 더하면 81.4%였다. test set에서는 최종적으로 **83.5% mIoU**를 기록했고, 비교 표에서 제시된 DeepLabV3+, InPlaceABN, DRN-CRL 등을 넘어섰다. 클래스별로는 pole, traffic light/sign, person, rider, bicycle 같은 small/thin object에서 개선폭이 컸다. 논문은 합성 샘플이 이런 클래스의 variation을 늘려 일반화에 도움을 준다고 해석한다.

### CamVid

CamVid는 701장의 densely annotated 이미지로 구성되며, 표준 분할은 367 train, 101 val, 233 test다. 총 32개 클래스가 있지만 일반적으로는 11개 클래스 평가가 주로 쓰인다.

CamVid에서는 Cityscapes에서 학습한 video reconstruction 모델을 **fine-tuning 없이 그대로 사용**해 증강 샘플을 만들었다고 명시한다. 이는 제안 방식의 일반화 가능성을 보여주려는 설정으로 보인다.

결과적으로 single-scale 기준 **81.7% mIoU**, multi-scale 기준 **82.9% mIoU**를 기록했다. 비교 대상 중 VideoGCRF가 75.2%, DenseDecoder가 70.9%였으므로 큰 폭의 향상이다. 또한 같은 WideResNet38 backbone을 사용하되 augmented sample과 label relaxation 없이 학습한 baseline은 79.8%였고, 제안 기법을 모두 쓰면 81.7%가 되어 **1.9%p 향상**이 있었다. 클래스별로는 11개 중 8개 클래스에서 최고 성능을 기록했는데, 특히 작은 구조물이나 thin object 클래스에서 개선이 두드러졌다고 한다.

### KITTI

KITTI semantic segmentation은 200 train, 200 test로 매우 작은 데이터셋이며, 이미지 해상도는 $375 \times 1242$이다. 논문은 데이터가 적기 때문에 200 training image에 대해 10-split cross validation fine-tuning을 수행했다고 설명한다. test 제출은 알고리즘당 한 번만 가능하기 때문에, 전체 training set에서 mIoU가 가장 좋은 모델을 택해 제출했다.

최종 결과는 **72.83 IoU class**, **48.68 iIoU class**, **88.99 IoU category**, **75.26 iIoU category**였다. 논문은 특히 mIoU 관점에서 이전 최고 성능인 MapillaryAI [10]의 69.56보다 **3.3%p 높다**고 강조한다. 더 흥미로운 점은 MapillaryAI가 ROB Challenge 2018 우승 엔트리이며 **5개 모델 앙상블**인데, 본 논문은 **single model**로 이를 넘었다는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 아이디어가 단순한 augmentation 수준을 넘어서, **비디오 기반 데이터셋의 구조를 직접 활용하는 실용적인 학습 전략**으로 연결된다는 점이다. 라벨이 없는 인접 프레임이 많은 데이터셋에서 사람 annotation 없이 학습 샘플을 늘릴 수 있고, 실제로 세 개의 대표 벤치마크에서 일관된 향상을 보였다. 특히 Cityscapes에서 joint propagation과 boundary relaxation의 기여를 분리해 보여준 ablation이 비교적 설득력 있다.

또 다른 강점은 **misalignment 문제를 정면으로 다뤘다는 점**이다. 단순히 label만 warping하면 될 것처럼 보이지만, 실제 segmentation 학습에서는 image-label 정합이 매우 중요하다. joint propagation은 이 문제를 매우 직접적으로 해결한다. 또한 boundary label relaxation은 수식도 간단하고 구현도 비교적 쉬우면서, propagation이 없는 baseline에서도 성능 향상을 보여 범용성이 있음을 시사한다.

실험 설계도 비교적 충실하다. label propagation 대 joint propagation, prediction 대 reconstruction, learned motion vector 대 optical flow, relaxation 유무, accumulated/non-accumulated까지 단계적으로 비교했다. 따라서 논문의 핵심 주장이 어느 부분에서 성능 향상을 만드는지 추적이 가능하다.

반면 한계도 분명하다. 첫째, 방법의 효과는 결국 **propagation 품질**에 의존한다. 논문도 propagation이 길어질수록 artifact가 쌓여 성능이 떨어진다고 인정한다. relaxation이 이를 완화하긴 하지만, 무한정 긴 전파가 가능하다는 뜻은 아니다. 둘째, 제안 방식은 비디오 기반 데이터셋에 자연스럽게 적용되지만, **독립 이미지 데이터셋에는 직접 적용하기 어렵다**. 논문도 joint propagation이 raw frame과 label 모두 부족한 다른 상황에도 쓸 수 있다고 언급하지만, 그 가능성을 실험으로 충분히 검증하지는 않았다.

셋째, boundary label relaxation은 경계 ambiguity를 잘 처리하지만, 반대로 보면 경계 supervision을 느슨하게 만드는 것이므로 아주 정밀한 boundary localization이 필요한 경우 어떤 trade-off가 생기는지는 논문에서 깊게 다루지 않는다. 넷째, 최종 성능 향상에는 Mapillary pre-training, class uniform sampling, multi-scale inference 같은 강한 recipe도 함께 들어간다. 따라서 제안 기법 자체의 순수 기여는 ablation으로 어느 정도 확인되지만, 최종 SOTA 성능이 전적으로 propagation과 relaxation만의 효과라고 해석하면 과장될 수 있다.

또한 failure case 분석에서 car vs truck, person vs rider, wall vs fence, terrain vs vegetation 같은 class confusion이 여전히 존재하고, reflection이나 건물 내부 인물 모형을 사람으로 예측하는 사례도 나온다. 즉, appearance 기반으로는 맞아 보여도 **context reasoning**은 여전히 부족하다. 논문도 이런 실패를 인정한다.

## 6. 결론

이 논문은 semantic segmentation에서 annotation 비용 문제를 해결하기 위해, **video prediction/reconstruction 기반 label synthesis**, **joint image-label propagation**, 그리고 **boundary label relaxation**이라는 세 가지 축의 방법을 제안한다. 핵심은 비디오 모델이 배운 motion transformation을 이용해 새로운 image-label training pair를 만들고, 경계에서는 단일 정답을 강제하지 않는 방식으로 noisy supervision에 강한 학습을 만드는 것이다.

실험적으로는 Cityscapes 83.5%, CamVid 82.9%, KITTI 72.8%의 강한 성능을 보였고, 특히 KITTI에서는 단일 모델로 기존 강한 앙상블을 넘었다. 이는 제안 방식이 단순한 보조 기법이 아니라 실제 benchmark 성능을 끌어올리는 실용적 방법임을 보여준다.

향후 연구 관점에서 이 논문은 두 가지 의미가 있다. 하나는 video-based augmentation이 semantic segmentation 학습에서 매우 유효하다는 점이고, 다른 하나는 boundary supervision을 더 유연하게 설계하는 것이 label noise 문제를 완화할 수 있다는 점이다. 논문 마지막에서도 저자들은 향후 learned kernel을 활용한 soft label relaxation 같은 방향을 제안한다. 실제 적용 측면에서는 자율주행처럼 비디오 기반 수집은 쉽지만 dense annotation은 비싼 환경에서 특히 가치가 큰 접근이라고 볼 수 있다.
