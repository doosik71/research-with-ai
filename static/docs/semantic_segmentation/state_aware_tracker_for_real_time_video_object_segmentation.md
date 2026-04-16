# State-Aware Tracker for Real-Time Video Object Segmentation

- **저자**: Xi Chen, Zuoxin Li, Ye Yuan, Gang Yu, Jianxin Shen, Donglian Qi
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2003.00482

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation, 즉 첫 프레임의 object mask만 주어진 상태에서 이후 모든 프레임의 같은 객체를 분할해야 하는 문제를 다룬다. 이 문제의 핵심 난점은 시간이 지나면서 대상 객체의 pose, scale, appearance가 크게 변할 수 있고, occlusion, truncation, fast motion, 심지어 search region 밖으로 벗어나는 상황까지 발생한다는 점이다. 단일 이미지 분할보다 훨씬 어렵고, 단순히 첫 프레임 정보를 계속 복사해 쓰는 방식으로는 긴 시퀀스에서 안정적으로 버티기 어렵다.

저자들은 기존 방법들이 영상의 성질, 특히 inter-frame consistency와 temporal context를 충분히 활용하지 못한다고 본다. 어떤 방법은 각 프레임을 거의 독립적으로 처리해 비디오 정보를 낭비하고, 어떤 방법은 이전 프레임 정보를 full image 수준에서 전파해 비효율적이며, 또 어떤 방법은 객체 상태가 바뀌어도 고정된 propagation 전략만 사용해 장기 시퀀스에서 불안정해진다고 지적한다. 또한 첫 프레임 또는 바로 이전 프레임만 참조하는 방식은 대상 객체의 holistic representation을 만들기 어렵다고 주장한다.

이 문제를 해결하기 위해 제안된 것이 SAT(State-Aware Tracker)이다. SAT는 VOS를 단순한 프레임별 segmentation이 아니라 “state estimation과 target modeling이 반복되는 연속 과정”으로 재해석한다. 저자들의 목표는 정확도만 높이는 것이 아니라, 실시간에 가까운 속도와 안정적인 장기 추적/분할을 동시에 달성하는 것이다. 논문에 따르면 SAT는 DAVIS2017 validation set에서 72.3%의 $J \& F$ mean과 39 FPS를 달성하여 정확도와 속도 사이에서 균형 잡힌 성능을 보인다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 두 가지이다. 첫째, 객체를 전체 이미지가 아니라 하나의 tracklet으로 다루는 것이다. 즉 매 프레임 전체 장면을 대상으로 분할하는 대신, 대상 객체 주변의 search region을 잘라내고 그 안에서 segmentation과 localization을 수행한다. 이렇게 하면 연산량이 줄고 distractor를 줄일 수 있어 효율성과 안정성이 동시에 개선된다.

둘째, 객체의 현재 상태를 추정하고 그 상태에 따라 시스템이 스스로 동작 방식을 바꾸도록 만드는 것이다. 저자들은 이를 estimation-feedback mechanism이라 부른다. 현재 프레임의 segmentation 결과를 보고 “지금은 normal state인가, abnormal state인가”를 판단한 뒤, 그 결과를 다시 다음 프레임 처리 방식에 반영한다. 이 피드백은 두 개의 loop로 구현된다.

하나는 Cropping Strategy Loop이다. 객체가 잘 보이는 normal state에서는 segmentation mask로부터 얻은 box를 사용하고, truncation이나 fast motion처럼 abnormal state에서는 tracker의 regression box를 사용한다. 다른 하나는 Global Modeling Loop이다. 각 프레임에서 얻은 객체 feature를 누적해 global representation을 만들되, state score가 높을 때만 강하게 반영해서 잘못된 프레임이 전역 표현을 망치지 않도록 한다.

기존 접근과의 차별점은, tracking과 segmentation을 느슨하게 이어붙인 것이 아니라 상태 인식(state awareness)을 중심으로 두 과정을 하나의 상호작용하는 파이프라인으로 묶었다는 점이다. 또한 online fine-tuning 대신 동적으로 업데이트되는 global feature를 사용해 semi-supervised 설정에서 필요한 target-specific representation을 더 효율적으로 구축한다.

## 3. 상세 방법 설명

전체 추론 흐름은 논문 표현대로 `Segmentation - Estimation - Feedback`의 세 단계로 요약된다. 먼저 Joint Segmentation Network가 현재 프레임에서 객체 mask를 예측한다. 다음으로 State Estimator가 이 mask 결과를 평가해 현재 상태가 정상인지 비정상인지 판단한다. 마지막으로 Feedback 단계에서 이 판단을 이용해 다음 프레임의 crop 전략과 global representation 업데이트 방식을 조절한다.

Joint Segmentation Network는 세 종류의 feature를 결합한다. 첫 번째는 saliency encoder feature이다. 이 branch는 객체 주변의 비교적 작은 영역을 crop한 뒤 큰 해상도로 확대해 입력받는다. 저자 설명대로 이렇게 하면 distractor가 줄고, 객체의 세부 정보를 보존한 class-agnostic feature를 얻을 수 있다. backbone으로는 shrinked ResNet-50을 사용한다.

두 번째는 similarity encoder feature이다. 이 branch는 현재 프레임의 더 큰 search region과 첫 프레임의 target region을 함께 입력받아 feature correlation을 계산한다. 여기서 얻는 correlated feature는 현재 후보가 초기 객체와 얼마나 appearance similarity를 가지는지 알려 주며, saliency encoder가 가지기 어려운 instance-level discrimination을 보완한다. 구현은 SiamFC++를 따르고 backbone은 AlexNet이다.

세 번째는 Global Modeling Loop에서 유지되는 global feature이다. 이는 시간 축 전체에서 누적된 대상 객체 표현으로, 장기 시퀀스에서 appearance variation에 더 강한 guidance 역할을 한다. 최종적으로 이 세 feature를 element-wise addition으로 합쳐 강한 high-level feature를 만든다. 이후 bilinear interpolation으로 upsample하고, saliency encoder의 low-level feature와 순차적으로 concatenate하여 decoder가 세밀한 contour를 가진 mask를 복원한다. 저자들의 의도는 high-level에서는 discrimination과 robustness를, low-level에서는 clean detail을 확보하는 것이다.

State Estimator는 현재 segmentation 결과를 바탕으로 상태 점수 $S_{state}$를 계산한다. 논문은 상태를 normal state와 abnormal state의 두 범주로 나눈다. well-presented 객체는 보통 예측 confidence가 높고 mask가 한 덩어리로 응집되어 있다. 반면 truncated 상태에서는 mask가 여러 조각으로 흩어질 수 있고, occluded 또는 사라진 경우에는 confidence가 낮아진다. 이를 반영해 두 개의 점수를 정의한다.

첫 번째는 confidence score $S_{cf}$이다.

$$
S_{cf} = \frac{\sum_{i,j} P_{i,j} \cdot M_{i,j}}{\sum_{i,j} M_{i,j}}
$$

여기서 $P_{i,j}$는 픽셀 $(i, j)$에서의 mask prediction score이고, $M$은 예측된 binary mask이다. 즉 foreground로 판정된 픽셀들에 대해 평균적으로 얼마나 자신 있게 예측했는지를 나타낸다.

두 번째는 concentration score $S_{cc}$이다.

$$
S_{cc} = \frac{\max(\{|R_c^1|, |R_c^2|, \cdots, |R_c^n|\})}{\sum_1^n |R_c^i|}
$$

여기서 $|R_c^i|$는 예측된 binary mask의 $i$번째 connected region의 픽셀 수이다. 즉 전체 foreground 면적 중 가장 큰 connected component가 차지하는 비율이다. 한 덩어리로 잘 모여 있으면 값이 높고, 여러 조각으로 분리되면 낮아진다.

최종 state score는 두 값을 곱해 얻는다.

$$
S_{state} = S_{cf} \times S_{cc}
$$

그리고 $S_{state} > T$이면 normal state, 아니면 abnormal state로 판단한다. 논문에서는 grid search로 $T = 0.85$를 사용했다고 명시한다.

Cropping Strategy Loop는 이 state score를 기반으로 다음 프레임 crop을 위한 bounding box 생성 방식을 바꾼다. normal state에서는 예측 binary mask의 largest connected region을 택하고, 그 minimal bounding box를 사용한다. 논문은 이를 mask-box라고 볼 수 있다. 이 방식은 객체가 잘 보이는 경우 contour에 더 밀착된 box를 주므로 위치가 더 정확하고, search region도 더 작아 distractor에 강하다.

반대로 abnormal state에서는 similarity encoder 뒤에 붙은 regression head가 box를 예측한다. 이 regression head는 SiamFC++ 방식에 기반하고, 예측된 위치, scale, ratio에 temporal smoothness를 적용한다. 이 regression-box는 더 넓은 검색 영역을 전제로 하므로 fast motion 상황에서 객체를 다시 찾기 쉽고, truncation 시에도 객체 전체를 포함하는 box를 줄 가능성이 높다. occlusion이나 disappear 상황에서도 temporal smoothness 덕분에 완전히 엉뚱한 위치로 튀지 않고 비교적 합리적인 추정을 유지할 수 있다. 논문의 주장은, 두 전략 중 하나만 고집하면 항상 문제가 생기지만, 상태에 따라 switching하면 정확성과 안정성을 동시에 얻을 수 있다는 것이다.

Global Modeling Loop는 시간에 따라 target object의 global representation을 누적 업데이트한다. 프레임 $t$에서 binary mask를 얻은 뒤, 이를 이용해 배경을 제거한 image patch를 만들고, 별도의 feature extractor(shrinked ResNet-50)로 high-level feature $F_t$를 추출한다. 이 feature를 이전 global representation $G_{t-1}$와 융합해 새 표현 $G_t$를 만든다.

$$
G_t = (1 - S_{state} \cdot \mu) \cdot G_{t-1} + S_{state} \cdot \mu \cdot F_t
$$

여기서 $\mu$는 업데이트 step length이며 논문에서는 0.5를 사용한다. 이 식의 의미는 단순하다. 현재 프레임이 신뢰할 만하면, 즉 $S_{state}$가 크면 현재 feature를 많이 반영한다. 반대로 occlusion, disappearance, poor segmentation처럼 신뢰가 낮으면 현재 feature의 영향력을 줄인다. 따라서 global representation이 이상한 프레임 때문에 오염되는 것을 완화할 수 있다. 이 전역 표현은 다시 Joint Segmentation Network에 입력되어 이후 프레임 segmentation을 돕는다. 저자들은 이것이 장기적인 visual variant에 robust한 target modeling을 가능하게 한다고 본다.

학습은 2단계로 이루어진다. 먼저 similarity encoder와 regression head를 object tracking datasets에서 SiamFC++ 방식으로 학습한다. 그다음 similarity encoder와 regression head의 weight를 고정한 채 전체 파이프라인을 학습한다. segmentation 쪽 데이터로는 COCO, DAVIS2017 training set, YouTube-VOS training set을 사용한다. 손실 함수는 stride 4의 binary mask prediction에 cross-entropy loss를 적용하고, stride 8과 stride 16의 auxiliary loss를 각각 0.5, 0.3 가중치로 추가한다. optimizer는 momentum 0.9의 SGD이고, batch size 16, 8 GPUs, synchronized batch normalization을 사용한다. 학습은 총 20 epochs이며 처음 2 epochs는 warm-up, 이후 18 epochs는 cosine annealing learning rate를 쓴다.

## 4. 실험 및 결과

저자들은 DAVIS2016, DAVIS2017, YouTube-VOS에서 실험했다고 밝힌다. DAVIS2017은 multi-object VOS 설정이며, 각 객체별 probability map을 예측한 뒤 concat 후 softmax aggregation으로 최종 결과를 만든다. 주요 평가지표는 $J \& F$, $J_M$, $F_M$, 그리고 시간에 따른 성능 저하를 나타내는 $J_D$이다. 속도는 single RTX 2080Ti에서의 forward pass FPS로 측정했다고 명시되어 있다.

가장 중요한 결과는 DAVIS2017 validation set이다. SAT는 $72.3$의 $J \& F$, $68.6$의 $J_M$, $76.0$의 $F_M$, 그리고 39 FPS를 기록했다. 논문이 비교한 offline 계열 방법 중 FEELVOS는 $71.5$에 2.2 FPS, AGAME은 $70.0$에 14.3 FPS, RGMP는 $66.7$에 7.7 FPS, RANet은 $65.7$에 30 FPS, SiamMask는 $56.4$에 35 FPS였다. 즉 SAT는 FEELVOS나 AGAME보다 더 높은 정확도를 내면서도 훨씬 빠르고, SiamMask나 RANet처럼 빠른 계열보다도 훨씬 높은 segmentation 성능을 보인다. 또한 $J_D = 13.6$으로 표에 제시된 많은 offline 방법보다 decay가 낮아 시간이 지나도 덜 무너진다고 해석할 수 있다.

추가로 저자들은 ResNet-18 기반의 빠른 버전인 Ours-Fast도 제시한다. 이 버전은 DAVIS2017에서 $69.5$의 $J \& F$와 60 FPS를 달성한다. 즉 약간의 정확도 손실로 더 높은 실시간성을 얻는 변형도 가능함을 보여 준다.

DAVIS2016 validation set에서는 SAT가 $83.1$의 $J \& F$, 39 FPS를 기록했다. FEELVOS의 81.7, RGMP의 81.8, SiamMask의 69.8보다 좋고, online fine-tuning 없이도 상당히 강한 결과를 보인다. 다만 STM은 89.3으로 더 높다. 논문은 STM이 더 많은 training data와 더 긴 training time을 요구한다고 덧붙인다. 즉 SAT는 최고 정확도 자체보다는 speed-accuracy trade-off를 강점으로 내세운다.

YouTube-VOS에서는 overall $G = 63.6$, seen category에서 $J_s = 67.1$, $F_s = 70.2$, unseen category에서 $J_u = 55.3$, $F_u = 61.7$을 기록한다. 표 기준으로 SiamMask, RGMP, S2S보다 전반적으로 좋으며, offline 방식 중 경쟁력 있는 성능이다. 그러나 여기서도 STM이 79.4로 훨씬 높다. 따라서 이 논문의 강점은 절대 최고 성능보다 효율성에 있다.

Ablation study는 이 논문의 설계를 이해하는 데 특히 중요하다. 가장 단순한 Naive Seg baseline은 $48.1$의 $J \& F$에 그친다. 여기에 tracker를 결합한 Track-Seg baseline은 $61.6$으로 크게 상승한다. similarity encoder의 correlated feature를 추가하면 $63.9$가 되어 2.3% 향상된다. Global Modeling Loop를 추가하면 $68.7$로 다시 4.8% 향상된다. 마지막으로 Cropping Strategy Loop까지 포함한 full SAT는 $72.3$이 되어 추가로 3.6% 상승한다. 이 수치는 각 구성 요소가 단순한 부가 장치가 아니라 실제로 기여함을 보여 준다.

Global Modeling Loop에 대한 세부 ablation도 설득력이 있다. first frame only를 쓰면 69.7, first + previous frame을 쓰면 71.1인데, 제안된 global representation은 72.3이다. 즉 전역 표현을 누적 구축하는 방식이 단순 참조 방식보다 낫다. no Score Weight는 71.1로 떨어져 state score 기반 weighting이 중요함을 보여 준다. no Mask Filter는 66.7, concat Mask는 66.5까지 떨어지므로 배경을 명시적으로 제거하는 설계가 핵심임을 알 수 있다. 저자들의 해석은 foreground의 high-level semantic feature는 서로 보완적이지만 background는 프레임마다 바뀌므로 누적하면 오히려 해가 된다는 것이다.

Cropping Strategy Loop에 대한 분석도 흥미롭다. DAVIS2017 validation set 3923 frames 중 2876 frames, 즉 74%를 normal state로 판정했고 1047 frames, 즉 26%를 abnormal state로 판정했다. 이는 설계 의도와 맞아떨어진다. 대부분의 보통 프레임은 정밀한 mask-box를 쓰고, 상대적으로 적지만 중요한 어려운 프레임에서는 regression-box로 복구하는 구조다.

Upper-bound 분석에서는 global modeling에 ground truth mask를 쓰면 +1.7%, crop box에 ground truth box를 쓰면 +1.8%, 둘 다 쓰면 총 +5.2% 향상되어 77.5의 $J \& F$를 얻는다. 이는 두 loop가 아직 더 개선될 여지가 있는 연구 포인트임을 보여 준다. 즉 저자들이 제안한 방향 자체는 맞지만 state estimation, crop generation, global update 품질이 더 좋아지면 추가 이득이 가능하다는 뜻이다.

연산량 비교에서도 SAT의 효율성이 강조된다. 표에 따르면 Ours-Fast는 약 12 Gflops, Ours는 약 13 Gflops이고, SiamMask는 약 16 Gflops, RANet과 AGAME은 65 Gflops를 넘는다. 논문은 이를 backbone과 input resolution 설계 덕분이라고 설명한다. similarity encoder는 큰 입력이지만 AlexNet을 쓰고, saliency encoder는 shrinked ResNet-50을 쓰며, Global Modeling Loop는 $129 \times 129$로 축소된 이미지만 사용한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 재정의가 명확하고, 그에 맞는 구조가 실제 ablation으로 잘 뒷받침된다는 점이다. 저자들은 VOS를 단순 segmentation이 아니라 상태 인식이 필요한 순차적 추정 문제로 보고, 그 관점을 Cropping Strategy Loop와 Global Modeling Loop라는 구체적 메커니즘으로 연결했다. 각 구성 요소가 실제로 성능을 올린다는 수치가 제시되어 있어 설계 타당성이 비교적 잘 설명된다.

또 다른 강점은 효율성과 정확도의 균형이다. SAT는 최고 정확도 방법은 아니지만, 실시간에 가까운 속도에서 기존 빠른 모델들보다 더 강한 성능을 보인다. 실제 적용에서는 최고 성능보다 처리 속도와 안정성이 더 중요한 경우가 많으므로, 이 지점은 분명한 실용적 가치가 있다. 특히 full-image propagation 대신 tracklet 기반 처리로 연산을 줄이고 distractor를 줄인 점은 공학적으로 설득력 있다.

Global Modeling Loop 역시 강점이다. online fine-tuning처럼 테스트 시 무거운 optimization을 수행하지 않으면서도, 시간에 따라 객체 표현을 점진적으로 보강하는 방식을 제시했다. state score로 update 세기를 조절하는 단순한 식도 해석 가능성이 좋고 구현 부담이 작다.

한편 한계도 분명하다. 우선 이 방법은 state estimator의 품질에 크게 의존한다. $S_{cf}$와 $S_{cc}$는 직관적이지만 비교적 hand-crafted한 규칙이며, state를 normal/abnormal의 이진 분류로 단순화한다. 실제 영상에서는 truncation, deformation, illumination change, heavy distractor, partial visibility가 섞여 더 복잡한 상태 공간을 형성할 수 있는데, 이 논문은 이를 세밀하게 모델링하지 않는다.

또한 global representation 업데이트 방식은 단순한 exponential moving average 형태다. 계산 효율은 높지만, 어떤 과거 프레임을 얼마나 오래 기억할지, appearance drift를 어떻게 제어할지에 대한 더 정교한 메커니즘은 없다. upper-bound 결과에서 여전히 5.2%의 개선 여지가 있다는 점도 현재 loop들이 완전하지 않음을 보여 준다.

정확도 측면에서는 STM 같은 더 강한 메모리 기반 방법보다 낮다. 논문은 이를 인정하고 속도-정확도 균형을 장점으로 제시하지만, 만약 최고 segmentation accuracy가 최우선인 응용이라면 SAT가 최선은 아닐 수 있다. 또 논문 추출 텍스트만 기준으로 보면 state estimator의 threshold $T=0.85$가 데이터셋 전반에 얼마나 일반화되는지, multi-object 상황에서 object 간 상호 간섭이 얼마나 문제가 되는지에 대한 추가 분석은 충분히 제시되지 않았다.

비판적으로 보면, SAT는 “tracking과 segmentation의 통합”을 주장하지만 실제 구조상 similarity encoder + regression head는 tracker 역할이 강하고, saliency decoder는 segmentation 역할이 강하다. 물론 feedback으로 tightly coupled되어 있는 점은 의미 있지만, 완전히 새로운 원리라기보다 tracking-based VOS를 state-aware하게 정교화한 시스템에 가깝다. 그럼에도 불구하고 이 결합 방식이 실제 성능 향상으로 이어졌다는 점은 긍정적이다.

## 6. 결론

이 논문은 semi-supervised VOS에서 정확도와 속도를 동시에 만족시키기 위해 SAT(State-Aware Tracker)를 제안한다. 핵심은 객체를 tracklet 단위로 다루어 효율적인 local processing을 수행하고, segmentation 결과로부터 state score를 계산해 crop 전략과 global representation 업데이트를 적응적으로 바꾸는 것이다. 이를 통해 장기 시퀀스에서 더 안정적이고 robust한 segmentation을 얻고자 했다.

실험 결과는 이 접근이 실제로 유효함을 보여 준다. SAT는 DAVIS2017에서 39 FPS와 72.3의 $J \& F$를 달성해 강한 speed-accuracy trade-off를 보였고, ablation을 통해 correlated feature, Global Modeling Loop, Cropping Strategy Loop의 기여도도 확인했다. 최고 정확도 자체는 일부 대형 모델에 미치지 못하지만, 실시간성, 계산 효율, 장기 안정성 측면에서 충분히 의미 있는 기여라고 볼 수 있다.

향후 연구 관점에서 보면, 이 논문은 두 방향을 열어 둔다. 하나는 더 정교한 state estimation과 switching policy 설계이고, 다른 하나는 더 강건한 global target representation 학습이다. 실제 응용에서는 실시간 video understanding, robotics, AR, surveillance처럼 빠르면서도 안정적인 object segmentation이 필요한 영역에 유용할 가능성이 크다.
