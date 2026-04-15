# Learning Video Object Segmentation from Static Images

- **저자**: Anna Khoreva, Federico Perazzi, Rodrigo Benenson, Bernt Schiele, Alexander Sorkine-Hornung
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1612.02646

## 1. 논문 개요

이 논문은 video object segmentation을 “guided instance segmentation” 문제로 다시 정식화한다. 문제 설정은 다음과 같다. 비디오의 첫 프레임 또는 소수의 프레임에서 특정 객체 인스턴스에 대한 annotation이 주어졌을 때, 같은 객체를 이후 모든 프레임에서 정확히 분할해야 한다. 기존 방법들은 주로 box tracking과 segmentation을 결합하거나, 첫 프레임의 마스크를 그래프, CRF, GrabCut류 기법으로 시공간적으로 전파하는 방식이 많았다.

저자들의 핵심 주장은, 이 문제를 풀기 위해 반드시 densely annotated video training data가 필요한 것은 아니라는 점이다. 오히려 static image의 segmentation annotation만으로도 충분히 강한 video object segmentation 모델을 학습할 수 있다고 보인다. 이는 데이터 수집 비용 측면에서 중요하다. 픽셀 단위 video annotation은 매우 비싸고 규모를 키우기 어렵기 때문이다.

논문이 다루는 중요성은 분명하다. video editing, movie production, object-centric video understanding 같은 응용에서 특정 객체를 정확히 분할하는 기술은 매우 중요하다. 그런데 기존 방식은 종종 전역 최적화, long-range graph, optical flow 등에 크게 의존하며 계산량이 크고 세부 경계 처리도 어렵다. 이 논문은 한 프레임씩 처리하는 feed-forward convnet 기반 방식으로도 경쟁력 있는 결과를 낼 수 있음을 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 현재 프레임 $t$의 객체를 분할할 때, 이전 프레임 $t-1$의 객체 마스크 추정치를 추가 입력으로 넣어 convnet이 “어떤 객체를 분할해야 하는지”를 알게 하는 것이다. 즉 일반적인 semantic segmentation 네트워크를, 특정 객체 인스턴스를 따라가는 instance-aware segmentation 네트워크처럼 사용한다.

구체적으로는 RGB 이미지에 이전 프레임 마스크를 붙여 4채널 입력(RGB+mask)으로 만들고, 이로부터 현재 프레임의 정교한 마스크를 예측한다. 이때 이전 프레임 마스크는 정확한 정답일 필요가 없고, 대략적인 위치와 형태만 알려주는 rough estimate여도 충분하다고 주장한다. 논문은 이 네트워크를 사실상 “mask refinement network”로 해석한다.

또 하나의 핵심은 offline learning과 online learning의 결합이다. offline 단계에서는 static image만 이용해, 거칠고 왜곡된 입력 마스크로부터 올바른 객체 마스크를 복원하도록 학습한다. online 단계에서는 테스트할 비디오의 첫 프레임 annotation만으로 추가 fine-tuning하여, 해당 비디오의 특정 객체 외형에 더 특화되도록 만든다. 저자들은 이 조합이 일반적인 instance segmentation 능력과 특정 객체에 대한 specialization 사이의 균형을 만들어 준다고 설명한다.

기존 접근과의 차별점은 몇 가지가 있다. 첫째, superpixel, object proposal, box tube 같은 중간 표현 없이 바로 픽셀 단위 마스크를 추정한다. 둘째, 긴 시퀀스 전체를 동시에 최적화하지 않고 frame-by-frame으로 처리한다. 셋째, offline 학습에 비디오가 아니라 static image만 사용한다. 넷째, full mask annotation뿐 아니라 bounding box annotation에서도 동작하도록 확장할 수 있다.

## 3. 상세 방법 설명

전체 파이프라인은 단순하다. 프레임 $t$에서의 입력은 현재 RGB 이미지와 이전 프레임의 마스크 추정치다. 이전 프레임 마스크를 그대로 쓰지 않고 dilation으로 coarsening하여 세부 윤곽을 뭉개고, 이를 현재 프레임의 추가 입력 채널로 사용한다. 네트워크는 이 rough mask와 현재 이미지를 보고 현재 프레임의 refined mask를 출력한다.

### 3.1 오프라인 학습

기본 backbone으로는 DeepLabv2-VGG를 사용한다. 원래 semantic segmentation용 네트워크를 3채널 RGB 입력에서 4채널 RGB+mask 입력으로 확장한다. 첫 번째 convolution layer의 추가 mask 채널 가중치는 Gaussian initialization을 사용한다. 네트워크는 VGG16 ImageNet pretrained model로 초기화한다.

오프라인 학습의 핵심은 “이전 프레임에서 넘어온 마스크가 완벽하지 않다”는 테스트 상황을 훈련에서 모사하는 것이다. 이를 위해 정답 segmentation mask를 변형하여 거친 입력 마스크를 인위적으로 만든다. 사용된 변형은 다음 세 가지다.

첫째, affine transformation이다. object size 기준으로 $\pm 5\%$ random scaling, 위치 기준으로 $\pm 10\%$ translation을 적용한다. 이는 프레임 간 객체의 작은 이동과 크기 변화를 모사한다.

둘째, non-rigid deformation이다. thin-plate splines를 사용하며, 5개의 control point를 두고 각 점을 x, y 방향으로 원래 mask width와 height의 $\pm 10\%$ 범위에서 랜덤 이동시킨다. 이는 객체의 비강체 변형을 반영한다.

셋째, coarsening이다. dilation 반경 5픽셀을 적용해 경계를 뭉개고 더 “blob-like”한 마스크로 만든다. 이는 이전 프레임에서 convnet이 낸 imperfect mask를 흉내 내기 위한 것이다.

이 과정을 통해 한 장의 annotated image로부터 여러 개의 plausible previous-frame mask를 생성한다. 논문은 약 11,282장의 static image를 사용했고, 데이터셋은 ECSSD, MSRA10K, SOD, PASCAL-S이다. 각 이미지에서 두 개의 다른 입력 mask를 생성한다.

학습은 SGD를 사용한다. mini-batch는 10, 초기 learning rate는 $0.001$, polynomial learning policy를 사용하며 momentum은 $0.9$, weight decay는 $0.0005$, 총 20k iterations 동안 학습한다.

### 3.2 온라인 학습

오프라인 학습만으로도 성능이 나오지만, 저자들은 online fine-tuning이 성능 향상에 중요하다고 본다. 테스트 시 첫 프레임의 annotation을 추가 학습 데이터로 사용한다. 첫 프레임 이미지에 대해 flipping, rotation, mask deformation을 적용해 약 $10^3$개의 training sample을 만든 뒤, 오프라인 학습된 네트워크를 200 iterations 동안 fine-tuning한다.

중요한 점은 MDNet류 추적기처럼 일부 domain-specific layer만 바꾸는 것이 아니라, 이 논문은 모든 convolutional 및 fully connected layer를 fine-tuning한다는 것이다. 이를 통해 네트워크 가중치 일부가 해당 비디오의 특정 객체 appearance를 반영하게 된다. 저자 설명대로, 이 과정은 객체의 외형 변화에 일반화할 수 있는 성질과, 특정 객체에 집중할 수 있는 성질을 동시에 확보하려는 시도다.

### 3.3 추론 방식

오프라인 학습만 사용할 경우 추론은 간단하다. 프레임 $t-1$의 예측 마스크를 dilation하여 coarse mask를 만들고, 이를 프레임 $t$의 RGB 이미지와 함께 네트워크에 넣어 현재 마스크를 얻는다. 저자들은 optical flow로 mask를 warp하는 방법도 실험했지만, flow 오차가 이득을 상쇄했다고 말한다. 따라서 기본 시스템은 단순히 이전 마스크를 거칠게 복사해서 다음 프레임을 안내하는 방식이다.

### 3.4 변형 모델

`MaskTrack Box`는 첫 프레임 annotation이 segmentation mask가 아니라 bounding box일 때를 위한 변형이다. 첫 프레임에서는 box rectangle을 입력 마스크로 쓰도록 별도 학습된 convnet을 사용하고, 그 다음 프레임부터는 표준 `MaskTrack`으로 넘어간다.

`MaskTrack + Flow`는 optical flow magnitude를 추가 신호로 사용하는 방식이다. EpicFlow, Flow Fields, convolutional boundaries를 이용해 optical flow를 계산한 뒤, flow magnitude를 3채널 이미지처럼 복제하여 두 번째 네트워크 입력으로 사용한다. 하나는 RGB, 다른 하나는 flow magnitude를 받아 각각 출력 score를 만들고, 이를 평균하여 최종 결과를 낸다. 흥미로운 점은 flow 전용으로 별도 구조를 설계하지 않고, 원래 RGB용 모델을 그대로 재사용한다는 것이다. 논문은 flow magnitude가 대략 gray-scale object처럼 보여 객체 shape 정보를 담기 때문이라고 설명한다.

### 3.5 수식 관련 설명

논문 본문에는 명시적인 손실 함수 식이나 목적 함수 식이 제시되어 있지 않다. 따라서 정확한 형태의 loss를 재구성해서 적는 것은 적절하지 않다. 다만 DeepLabv2 기반 pixel labeling network를 사용한다고 했으므로, 프레임 단위 픽셀별 foreground/background 분류를 학습하는 segmentation objective로 이해하는 것이 자연스럽다. 그러나 손실의 정확한 수학적 형태는 논문 발췌문에 명시되지 않았다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 프로토콜

평가는 DAVIS, YoutubeObjects, SegTrack-v2 세 데이터셋에서 수행한다.

DAVIS는 50개 비디오, 총 3,455 프레임으로 구성되며, 매 프레임 pixel-level mask가 제공된다. YoutubeObjects는 10개 object category의 126개 비디오, 20,000프레임 이상을 포함한다. SegTrack-v2는 14개 시퀀스, 24개 객체, 947프레임으로 이루어지며, 여러 객체가 있는 경우 instance별로 별도 문제로 취급한다.

평가 지표는 mIoU(mean intersection-over-union), 즉 Jaccard Index의 평균이다. DAVIS는 benchmark code를 사용하며 첫 프레임과 마지막 프레임을 제외하고 평가한다. YoutubeObjects와 SegTrack-v2는 첫 프레임만 제외한다.

### 4.2 Ablation study

가장 중요한 ablation 결과는 다음과 같다.

기본 `MaskTrack`는 DAVIS에서 74.8 mIoU를 기록한다. 여기에 optical flow를 더한 `MaskTrack + Flow`는 78.4, CRF까지 추가한 `MaskTrack + Flow + CRF`는 80.3까지 올라간다. 즉 추가적인 motion cue와 post-processing이 확실한 이득을 준다.

하지만 논문이 특히 강조하는 것은 offline/online 학습과 mask deformation의 중요성이다. online fine-tuning을 제거하면 69.9로 약 4.9 포인트 감소한다. 반대로 offline training 없이 online fine-tuning만 하면 57.6으로 크게 떨어진다. 즉 첫 프레임만으로 specialization하는 것은 충분하지 않고, static image로부터 배운 일반적 guided instance segmentation 능력이 필수적이다.

training data를 11k에서 5k로 줄여도 73.2로 감소폭은 작다. 이는 데이터 양이 아주 극단적으로 중요하지는 않지만, 더 많아지면 성능 향상이 가능하다고 해석할 수 있다. 흥미롭게도 video data로 offline training했을 때 72.0으로 오히려 약간 나빠진다. 저자들은 이것이 기존 video segmentation dataset의 규모가 작고 다양성이 부족하며, benchmark 간 domain shift가 존재하기 때문이라고 해석한다. 즉 이 논문에서는 static image training이 단순한 차선책이 아니라 실제로도 매우 효과적이다.

mask deformation은 결정적이다. deformation을 완전히 제거하면 성능이 17.1까지 붕괴한다. dilation 제거는 72.4, non-rigid deformation 제거는 73.3이므로 각각도 의미 있는 기여를 한다. 결국 이 방법은 “rough previous mask를 입력으로 넣는 테스트 상황”을 얼마나 잘 시뮬레이션하느냐에 크게 의존한다.

입력 채널 ablation도 흥미롭다. full segment 대신 box를 쓰면 69.6으로 떨어지지만 여전히 괜찮은 성능이다. mask 입력 자체를 제거해도 72.5가 나온다. 이는 online fine-tuning으로 특정 객체 appearance를 모델링하는 것 자체가 강력하다는 뜻이다. 다만 최종적으로는 mask guidance가 추가 이득을 준다.

### 4.3 단일 프레임 annotation 결과

첫 프레임에 segmentation mask가 주어지는 표준 설정에서 `MaskTrack`는 DAVIS 74.8, YoutubeObjects 71.7, SegTrack-v2 67.4 mIoU를 기록한다. 표에 따르면 DAVIS에서는 ObjFlow 71.4, BVS 66.5보다 높고, YoutubeObjects에서는 ObjFlow 70.1, BVS 59.7보다 높으며, SegTrack-v2에서는 ObjFlow 67.5와 거의 비슷하고 TRS 69.1보다는 낮다.

즉 세 데이터셋 전체를 통틀어 항상 최고는 아니지만, 같은 모델과 같은 파라미터를 그대로 쓰면서 전반적으로 매우 경쟁력 있는 성능을 보인다. 저자들이 강조하듯, 이 결과는 global optimization도 없고 optical flow도 없는 순수 frame-by-frame feed-forward 시스템으로 얻은 것이다.

`MaskTrack Box`도 인상적이다. 첫 프레임에 box annotation만 주었을 때 DAVIS 73.7, YoutubeObjects 69.3, SegTrack-v2 62.4를 기록한다. 이는 full segment supervision보다 약간 낮지만 여전히 상위권에 속한다. 실용적으로는 box annotation 비용이 훨씬 낮을 수 있으므로 의미가 크다.

dataset-specific tuning을 허용하면 성능은 더 오른다. `MaskTrack + Flow + CRF`는 DAVIS에서 80.3 mIoU를 달성했고, YoutubeObjects에서는 72.6, SegTrack-v2에서는 70.3까지 간다고 supplementary가 설명한다. 다만 논문은 optical flow가 데이터셋 간 일관성이 부족해, 주 결과에서는 고정 파라미터의 순수 `MaskTrack`를 중심으로 제시한다.

### 4.4 Attribute-based evaluation

DAVIS 속성별 분석에서 `MaskTrack`는 background clutter, deformation, dynamic background, occlusion, fast motion, motion blur, appearance change, out-of-view 같은 다양한 난이도에서 전반적으로 강한 성능을 보인다. 특히 fast motion과 motion blur에서 spatio-temporal connection 기반 방법들의 전형적 실패 사례를 더 잘 처리한다고 주장한다.

camera-shake에서는 ObjFlow가 약간 더 좋지만, 대체로 `MaskTrack`는 거의 모든 attribute subset에서 강한 편이다. optical flow와 CRF를 추가하면 low resolution, scale variation, appearance change 같은 경우에서 더 강해진다. supplementary의 표를 보면 예를 들어 low resolution에서 `MaskTrack`는 0.60, `MaskTrack + Flow`는 0.75, `MaskTrack + Flow + CRF`는 0.77로 크게 오른다.

### 4.5 다중 프레임 annotation

이 논문은 첫 프레임 하나만 쓰는 경우를 넘어, 여러 프레임에 annotation이 있는 경우도 실험한다. 방법은 forward와 backward 두 방향으로 모델을 실행하고, 각 프레임에 대해 시간적으로 가장 가까운 annotated frame에서 전파된 결과를 선택하는 방식이다. online fine-tuning도 첫 프레임만이 아니라 모든 annotated frame을 사용한다.

결과적으로, segmentation mask annotation을 여러 프레임에 주면 성능이 빠르게 향상한다. DAVIS에서 전체 프레임의 약 10%만 annotation해도 mIoU 0.86에 도달하며, 20% quantile이 0.81이다. 저자들은 이를 “전체 프레임의 80%가 IoU 0.8 이상”으로 해석하며, 많은 응용에 충분한 수준이라고 본다. 반면 baseline인 nearest annotated mask copy는 같은 조건에서 0.64에 그친다.

box annotation의 경우도 10% 정도만 추가해도 mean IoU와 30% quantile이 대략 0.8 근처에 도달한다. 이후에는 성능이 포화되는 경향이 있다. 저자들은 box만으로는 본질적으로 제공되는 shape 정보가 제한적이기 때문이라고 해석한다. 그래도 moving from 1% to 3% to 4% annotation에서 품질이 급격히 오르는 점은, 소수의 추가 annotation만으로도 시스템이 충분히 이득을 얻는다는 사실을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 재정의가 명확하고 설득력 있다는 점이다. video object segmentation을 복잡한 시공간 최적화 문제로만 보지 않고, “rough guidance가 있는 per-frame instance segmentation”으로 바꿔 본 것이 핵심 기여다. 이 재정의 덕분에 static image annotation만으로도 학습할 수 있고, 기존 semantic segmentation backbone을 그대로 활용할 수 있다.

또 다른 강점은 실험적으로 중요한 설계 요소를 잘 분리해 검증했다는 점이다. offline/online training 각각의 역할, mask deformation의 필요성, box supervision 가능성, optical flow와 CRF의 추가 이득 등을 체계적으로 보여준다. 특히 deformation 제거 시 17.1 mIoU까지 무너지는 결과는 이 방법이 왜 동작하는지를 잘 설명해 준다.

실용성도 강점이다. 논문은 기본 시스템이 DAVIS 기준 평균 프레임당 약 12초가 걸린다고 보고하며, ObjFlow의 프레임당 약 2분보다 훨씬 빠르다고 주장한다. 또한 첫 프레임 segment뿐 아니라 box, 나아가 다중 프레임 annotation까지 유연하게 처리할 수 있다.

한편 한계도 분명하다. 첫째, 기본 방식은 이전 프레임의 마스크 추정에 의존하므로, 장기적인 누적 오차 가능성을 완전히 제거하지는 못한다. 논문은 rough mask refinement 덕분에 오류 누적을 줄였다고 보지만, 완전히 해소했다고 말하지는 않는다.

둘째, optical flow는 보조 정보로 유효하지만 brittle하다고 직접 인정한다. 데이터셋마다 다른 tuning이 필요하고, failure mode 때문에 고정 파라미터의 단일 시스템으로 쓰기 어렵다. 이는 motion cue 통합 방식이 아직 충분히 안정적이지 않음을 뜻한다.

셋째, 논문이 쓰는 backbone은 DeepLabv2-VGG로, 당시에는 강력했지만 temporal modeling 자체가 구조에 내장되어 있지는 않다. 저자들도 미래 연구로 temporal dimension과 global optimization의 통합을 제안한다. 즉 현재 방식은 비디오 문제를 프레임별 문제로 단순화해 성과를 냈지만, 진짜 temporal reasoning을 깊게 활용하는 방식은 아니다.

넷째, 손실 함수나 확률적 모델에 대한 수학적 기술이 본문 발췌에선 충분히 자세하지 않다. 네트워크 구조와 학습 절차는 명확하지만, 정확한 objective formulation을 논문이 상세 식으로 전개하지 않았기 때문에 이론적 이해보다는 시스템적 설계와 경험적 검증에 더 무게가 실려 있다.

비판적으로 보면, 이 논문의 성과는 매우 인상적이지만 일부 비교는 “고정 파라미터의 범용성”과 “데이터셋별 튜닝 성능”이 혼재되어 있다. 저자들은 이를 정직하게 구분해 설명하지만, 어떤 비교는 실전 설정의 fairness를 세밀하게 따져볼 필요가 있다. 또한 SegTrack-v2에서는 최고 성능이 아니므로, 모든 조건에서 일관되게 SOTA라고 보기는 어렵다. 그럼에도 세 데이터셋 전체에 걸친 안정성과 단순성은 높은 가치가 있다.

## 6. 결론

이 논문은 video object segmentation을 guided instance segmentation으로 재해석하고, 이전 프레임의 마스크를 추가 입력으로 사용하는 per-frame convnet 구조를 제안했다. 핵심은 static image만으로 offline training을 하고, 테스트 시 첫 프레임 annotation으로 online fine-tuning하여 특정 객체 appearance에 적응시키는 것이다. 이 조합으로 별도의 대규모 video annotation 없이도 경쟁력 있는 성능을 달성했다.

주요 기여는 세 가지로 정리할 수 있다. 첫째, static image 기반 학습만으로 video object segmentation이 가능함을 보였다. 둘째, mask guidance와 online specialization을 결합한 단순한 frame-by-frame 시스템이 강한 baseline이 될 수 있음을 입증했다. 셋째, box annotation이나 다중 프레임 annotation 같은 실용적 설정까지 포괄했다.

향후 연구 관점에서도 의미가 크다. 이 논문은 labeling convnet을 video object segmentation에 본격적으로 연결하는 초기 방향을 제시했고, 이후 더 강한 backbone, temporal modeling, memory mechanism, global optimization, better motion fusion으로 확장될 수 있는 출발점을 마련했다. 실제 적용 측면에서도, 적은 annotation 비용으로 높은 품질을 얻을 수 있다는 점에서 video editing, annotation propagation, semi-automatic content creation에 중요한 가능성을 보여준다.
