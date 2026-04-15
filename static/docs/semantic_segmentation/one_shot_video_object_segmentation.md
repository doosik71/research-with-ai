# One-Shot Video Object Segmentation

- **저자**: S. Caelles, K.-K. Maninis, J. Pont-Tuset, L. Leal-Taixé, D. Cremers, L. Van Gool
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1611.05198

## 1. 논문 개요

이 논문은 **semi-supervised video object segmentation** 문제를 다룬다. 즉, 비디오의 첫 프레임에서 객체의 foreground mask가 한 번만 주어졌을 때, 이후 모든 프레임에서 같은 객체를 background와 분리해내는 것이 목표다. 논문은 이를 위해 **OSVOS (One-Shot Video Object Segmentation)** 라는 방법을 제안한다.

핵심 문제의식은 분명하다. 기존의 video object segmentation 방법들은 대체로 optical flow, trajectory, frame-to-frame mask propagation 같은 **시간적 일관성(temporal consistency)** 에 강하게 의존했다. 이런 접근은 프레임 간 변화가 완만할 때는 효과적일 수 있지만, occlusion, abrupt motion, long-term disappearance 같은 상황에서는 쉽게 실패하고, 한 프레임의 오류가 뒤로 계속 전파되는 문제가 있다.

이 논문은 그 전제를 뒤집는다. 저자들은 “정말로 비디오 segmentation에서 시간적 연결을 강하게 모델링해야만 하는가?”라는 질문을 던지고, 충분히 강한 object appearance model을 학습하면 **각 프레임을 독립적으로 처리해도** 결과가 안정적일 수 있다고 주장한다. 이것이 중요한 이유는 명확하다. 프레임 독립 처리 방식은 occlusion에 더 강하고, 순차 처리에 묶이지 않으며, error propagation이 없고, 병렬 처리도 가능하다. 실제로 논문은 DAVIS validation에서 기존 최고 성능 68.0%를 79.8%로 크게 넘어선다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **점진적 특수화(progressive specialization)** 이다. 저자들은 segmentation 능력을 세 단계로 옮겨간다.

첫째, ImageNet으로 학습된 base CNN은 일반적인 semantic representation을 이미 갖고 있다. 둘째, 이 네트워크를 video segmentation 데이터로 다시 학습시켜 “foreground object란 무엇인가”라는 더 구체적인 개념을 익히게 한다. 셋째, 테스트 시점에 주어진 단 하나의 annotated frame으로 다시 fine-tuning하여, “이 비디오에서 찾아야 할 바로 그 객체”에 맞는 모델로 바꾼다. 즉, “It is an object”에서 “It is this particular object”로 이동하는 구조다.

기존 방식과의 가장 큰 차별점은 **mask propagation을 하지 않는다는 점**이다. OSVOS는 이전 프레임의 예측 결과를 다음 프레임으로 전달하지 않는다. optical flow도 필수가 아니다. 각 프레임은 동일한 객체 모델에 대해 독립적으로 segmentation된다. 그럼에도 불구하고 논문은 deep network가 충분히 정확한 object model을 학습하면 temporal coherence가 별도의 명시적 제약 없이도 자연스럽게 나타난다고 보인다.

또 하나의 중요한 아이디어는 **one-shot adaptation** 이다. 보통 deep learning 기반 segmentation은 많은 annotation을 필요로 하지만, 이 논문은 테스트 시점에 단 한 장의 labeled frame만으로 특정 객체 instance에 적응할 수 있음을 보여준다. 이것이 논문의 제목에 있는 one-shot의 의미다.

## 3. 상세 방법 설명

전체 파이프라인은 세 단계로 이해하면 된다.

먼저 **base network** 단계가 있다. 저자들은 VGG 기반 네트워크를 시작점으로 사용한다. 이 네트워크는 원래 ImageNet image labeling용으로 학습되어 있으므로, 그대로는 segmentation을 잘 하지 못한다.

그 다음 **parent network** 단계가 있다. 여기서는 DAVIS training set의 binary mask를 사용해 네트워크를 offline training한다. 이 단계의 목적은 특정 객체가 아니라, 일반적인 foreground/background 분리를 학습하는 것이다. 예를 들어 객체의 흔한 형태, foreground와 background의 차이 같은 보편적 구조를 익힌다.

마지막으로 **test network** 단계가 있다. 테스트 비디오의 첫 프레임과 그에 대한 ground-truth mask가 주어지면, parent network를 이 한 장에 대해 추가 fine-tuning한다. 이렇게 얻은 모델을 이용해 이후 프레임들을 segmentation한다. 중요한 점은 이후 프레임들을 순차적으로 의존해서 처리하는 것이 아니라, **모든 프레임을 독립적으로** 처리한다는 것이다.

네트워크 구조는 Fully Convolutional Network(FCN) 계열이다. 저자들은 VGG의 fully connected layers를 제거해 dense prediction이 가능하도록 바꾸고, 각 stage의 마지막 convolutional feature map을 skip connection으로 끌어와 결합한다. 깊은 층의 semantic 정보와 얕은 층의 spatial detail을 함께 써서, 입력 이미지와 같은 해상도의 segmentation map을 출력한다. 이 설계는 localization accuracy를 높이면서도 parameter 수를 줄여 적은 annotation에서도 학습 가능하게 만든다.

손실 함수는 pixel-wise binary cross-entropy다. 각 픽셀 $j$에 대해 label $y_j \in \{0,1\}$를 가지며, 전체 손실은 다음과 같이 정의된다.

$$
L(W) = - \sum_j y_j \log P(y_j=1 \mid X; W) + (1-y_j)\log(1 - P(y_j=1 \mid X; W))
$$

논문은 이를 positive pixel 집합 $Y^+$와 negative pixel 집합 $Y^-$로 나누어 다음처럼 다시 쓴다.

$$
L(W) = - \sum_{j \in Y^+} \log P(y_j=1 \mid X; W) - \sum_{j \in Y^-} \log P(y_j=0 \mid X; W)
$$

여기서 $P(\cdot)$는 마지막 layer activation에 sigmoid를 적용해 얻는다.

foreground/background 픽셀 수가 불균형할 수 있으므로, 저자들은 class-balanced loss를 사용한다. 수정된 손실은 다음과 같다.

$$
L_{mod} = - \beta \sum_{j \in Y^+} \log P(y_j=1 \mid X) - (1-\beta)\sum_{j \in Y^-} \log P(y_j=0 \mid X)
$$

여기서 $\beta = |Y^-| / |Y|$이다. 즉, negative pixel 비율을 반영해 positive와 negative의 손실 기여를 조정한다. 이는 foreground가 상대적으로 적은 binary segmentation에서 중요한 장치다.

학습 절차도 논문에 비교적 구체적으로 제시되어 있다. offline training에서는 SGD with momentum 0.9를 사용하고, DAVIS training masks로 50,000 iteration 학습한다. data augmentation은 mirroring과 zooming을 사용하며, learning rate는 $10^{-8}$에서 시작해 점차 감소시킨다. online fine-tuning은 테스트 시점에 특정 객체의 첫 프레임 한 장만으로 진행되며, 10초에서 10분까지의 학습 시간을 두고 speed-accuracy trade-off를 실험한다.

논문은 contour localization을 더 개선하기 위한 **boundary snapping**도 제안한다. 첫 번째 방법은 **Fast Bilateral Solver (FBS)** 를 이용해 예측 mask를 image edge에 맞추는 것이다. 이 방식은 빠르며 프레임당 약 60 ms가 추가된다. 하지만 단순한 color gradient를 기준으로 정렬하기 때문에 semantic contour를 잘 반영하지 못할 수 있다.

이를 보완하기 위해 두 번째 방법으로 **contour branch**를 둔 two-stream FCN 구조를 제안한다. 하나의 branch는 foreground segmentation을 하고, 다른 branch는 scene 전체의 contour를 검출한다. contour branch는 foreground object에만 맞춰 online tuning할 필요가 없기 때문에 offline only training이 가능하다. 저자들은 shared layers로 joint training하면 오히려 결과가 나빠졌다고 보고하며, 두 branch를 분리해 학습했다. 이후 contour branch 출력으로 UCM(Ultrametric Contour Map) 기반 superpixel을 만들고, foreground mask와의 majority voting으로 최종 segmentation을 얻는다. 이 방법은 FBS보다 느려서 약 400 ms/frame이 들지만 더 정확하다.

## 4. 실험 및 결과

주요 실험은 **DAVIS** validation set에서 수행되었다. DAVIS는 50개의 full-HD video sequence와 각 프레임의 pixel-level annotation을 제공한다. 평가 지표는 세 가지다. 첫째, region similarity $J$는 intersection over union(IoU)이다. 둘째, contour accuracy $F$는 경계 품질을 측정한다. 셋째, temporal instability $T$는 mask의 시간적 불안정성을 측정한다. 이 논문에서는 특히 $J$와 $F$를 중심으로 성능을 강조한다.

state-of-the-art 비교에서 OSVOS는 매우 강한 결과를 보인다. DAVIS validation에서 $J$ mean은 **79.8**로, OFL의 **68.0**보다 11.8 point 높고, BVS의 **60.0**보다 19.8 point 높다. $F$ mean도 **80.6**으로 OFL의 **63.4**, BVS의 **58.8**보다 크게 앞선다. 이는 단순한 소폭 개선이 아니라, 당시 기준으로는 상당히 큰 성능 도약이다.

ablation study는 이 논문의 설계가 왜 필요한지를 잘 보여준다. boundary snapping을 제거하면 $J$가 79.8에서 77.4로 2.4 point 하락한다. parent network pretraining을 제거하면 64.6으로 떨어져 15.2 point 손해를 본다. one-shot online fine-tuning을 제거하면 52.5로 떨어져 27.3 point 하락한다. 둘 다 제거하고 ImageNet raw CNN만 쓰면 $J=17.6$으로 사실상 무작위 수준이라고 저자들은 해석한다. 즉, **generic foreground prior를 배우는 offline training**과 **특정 객체에 맞추는 online adaptation**이 둘 다 핵심이다.

에러 분석도 흥미롭다. 저자들은 false positive를 object boundary에서 가까운 경우와 먼 경우로 나누고, false negative와 함께 분석했다. OSVOS의 주요 오류는 false negative가 더 많았고, boundary snapping은 특히 false positive를 줄이는 데 효과적이었다. 이는 contour refinement가 주로 경계와 배경 오검출 정리에 기여함을 뜻한다.

속도 측면에서 OSVOS는 매우 유연하다. 논문은 480p 프레임을 **102 ms/frame**에 처리할 수 있다고 보고한다. fine-tuning 시간까지 포함하면 정확도와 속도 사이에 trade-off가 있으며, 빠른 설정에서는 **181 ms/frame, 71.5%**, 느리지만 더 정확한 설정에서는 **7.83 s/frame, 79.7%** 수준을 제시한다. 또, 만약 특정 객체를 미리 알고 있어 fine-tuning을 사전에 해둘 수 있다면, 테스트 시에는 거의 단순 forward pass만 수행하면 된다.

추가 supervision 실험도 실용적으로 중요하다. 처음에는 annotation이 0개인 zero-shot 상태에서 $J=58.5$이고, 1개 annotated frame을 쓰면 79.8로 급상승한다. 2개면 84.6, 3개면 85.9, 4개면 86.9, 5개면 87.5, 모든 프레임을 쓰면 88.7이다. 즉, **첫 한 장의 annotation이 가장 큰 이득을 주고**, 소수의 추가 annotation으로도 성능이 계속 개선된다. 이는 rotoscoping 같은 실제 작업 환경에서 매우 유용하다.

논문은 DAVIS 속성별 분석도 제공한다. appearance change, deformation, fast motion, motion blur, occlusion 등의 challenge가 있는 시퀀스에 대해 OSVOS는 모든 속성에서 가장 높은 품질을 보였고, 속성이 있을 때와 없을 때의 성능 하락폭도 가장 작았다. 이는 단순 평균 성능뿐 아니라 robustness 측면에서도 강하다는 의미다.

tracking 관점의 평가도 수행했다. segmentation mask를 bounding box로 바꿔 VOT 스타일로 평가했을 때, OSVOS는 MDNET보다 모든 IoU threshold에서 높은 성능을 보였다. 예를 들어 overlap threshold 0.9에서는 OSVOS가 49.6%, MDNET이 14.7%였다. 다만 이 평가는 segmentation을 tracking으로 환산한 부가 실험이며, 논문의 본래 초점은 segmentation이다.

Youtube-Objects에서도 실험했으며, mean IoU는 **78.3**으로 OFL의 **77.6**보다 약간 높다. 차이는 DAVIS만큼 크지 않은데, 논문은 이 데이터셋이 occlusion과 motion 변화가 덜 심해서 temporal consistency를 강하게 쓰는 방법에 상대적으로 유리하기 때문이라고 설명한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 해결 방식이 매우 명확하다는 점이다. 저자들은 semi-supervised video object segmentation에서 당연시되던 temporal propagation 중심 사고를 버리고, **object model 자체를 강하게 학습하면 프레임 독립 처리만으로도 충분하다**는 관점을 설득력 있게 보여준다. 그리고 그 주장을 DAVIS에서 큰 성능 향상으로 입증했다.

또 다른 강점은 구조가 비교적 단순하다는 점이다. optical flow, CRF, 복잡한 sequence optimization 없이도 강한 성능을 낸다. pipeline이 간단하므로 구현과 해석이 쉬우며, error propagation이 없는 것도 큰 장점이다. occlusion 이후 다시 객체가 보이면 recovery가 가능하다는 점 역시 propagation 기반 방식보다 실용적이다.

one-shot adaptation의 실용성도 강점이다. 첫 프레임 하나만 annotation하면 된다는 설정은 실제 영상 편집, rotoscoping, annotation tooling에 매우 적합하다. 추가 annotation을 점진적으로 넣어 품질을 높일 수 있다는 점도 현업 사용성을 높인다.

반면 한계도 분명하다. 첫째, 이 방법은 프레임을 독립 처리하므로, temporal information을 명시적으로 활용하지 않는다. 논문은 이것이 장점이라고 주장하지만, 반대로 말하면 video-specific motion cue를 적극 활용하지 않기 때문에 특정 상황에서는 temporal modeling이 가능한 방법보다 불리할 수 있다. 실제로 temporal instability $T$ 지표에서는 OSVOS가 가장 좋은 값은 아니다. 예를 들어 DAVIS 표에서 OFL의 $T$ mean은 21.7이고, OSVOS는 37.6이다. 즉, segmentation 품질은 훨씬 좋지만, 시간적 매끄러움 지표 자체가 최고는 아니다. 이는 “프레임 독립 처리인데도 temporal coherence가 나온다”는 주장과 완전히 모순되지는 않지만, temporal smoothness를 직접 최적화한 것은 아니라는 점을 보여준다.

둘째, 테스트 시 online fine-tuning이 필요하다는 점은 비용이다. per-frame inference는 빠르지만, 좋은 성능을 얻으려면 sequence마다 fine-tuning 시간이 추가된다. 논문은 이를 trade-off로 제시하지만, 실시간 완전 자동 시스템에서는 여전히 부담일 수 있다.

셋째, contour snapping 모듈은 정확도를 올리지만 속도를 희생한다. learned contour 기반 snapping은 400 ms/frame의 추가 비용이 든다. 따라서 최고 성능을 쓰려면 pipeline이 완전히 lightweight하다고 보기는 어렵다.

넷째, 논문은 single-object setting을 중심으로 설명한다. 제공된 본문 기준으로는 multi-object segmentation을 일반화해 어떻게 처리하는지 상세히 논의하지 않는다. 따라서 여러 유사 객체가 동시에 등장할 때의 구체적 한계는 추가 annotation 실험의 camel 사례를 통해 간접적으로만 드러난다. 이 경우 OSVOS는 유사한 두 객체를 혼동했고, 추가 annotation으로 해결했다. 즉, **appearance가 매우 비슷한 distractor object**가 있으면 한 장 annotation만으로는 충분하지 않을 수 있다.

다섯째, 논문은 강력한 성능을 보였지만, 그 성공의 일부는 DAVIS 기반 parent network pretraining에 의존한다. 표 4에 따르면 약 200장의 annotated training image만으로도 상당한 성능을 얻지만, 완전히 데이터가 없는 환경에서 가능한 방법은 아니다. 즉, “one-shot”은 완전 무학습이 아니라, **강한 사전학습 + 소량 instance adaptation** 위에서 성립한다.

## 6. 결론

이 논문은 video object segmentation에서 매우 영향력 있는 관점을 제시한다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, ImageNet pretraining, DAVIS offline training, test-time one-shot fine-tuning을 결합해 특정 객체 instance에 빠르게 적응하는 OSVOS를 제안했다. 둘째, frame-to-frame propagation 없이도 각 프레임을 독립적으로 segmentation하여 높은 정확도와 실용적 안정성을 달성했다. 셋째, DAVIS에서 기존 방법 대비 큰 폭의 성능 향상을 보이며, 추가 annotation을 통한 progressive refinement도 가능함을 보였다.

실제 적용 측면에서 이 연구는 annotation assistance, video editing, rotoscoping, surveillance 같은 분야에 직접적인 의미가 있다. 특히 “한 장만 라벨링하고 전체 비디오를 분할한다”는 설정은 사람 작업을 크게 줄일 수 있다. 연구적으로도 이 논문은 이후의 many-shot, few-shot, online adaptation, mask propagation, memory-based video segmentation 연구들에 중요한 출발점 역할을 했다고 볼 수 있다.

정리하면, OSVOS는 “비디오 segmentation은 반드시 temporal propagation 중심이어야 한다”는 당시의 통념에 강하게 도전한 논문이다. 제공된 원문 기준에서 볼 때, 저자들은 그 주장을 단순 아이디어 수준이 아니라 명확한 구조, 손실 함수, 학습 절차, 정량 실험으로 뒷받침했다. 다만 test-time fine-tuning 비용과 temporal smoothness를 직접 최적화하지 않는 점은 남는 한계로 볼 수 있다. 그럼에도 이 논문은 one-shot adaptation과 per-frame segmentation의 결합이 얼마나 강력할 수 있는지를 설득력 있게 보여준 대표적인 작업이다.
