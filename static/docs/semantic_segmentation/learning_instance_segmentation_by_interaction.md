# Learning Instance Segmentation by Interaction

- **저자**: Deepak Pathak, Yide Shentu, Dian Chen, Pulkit Agrawal, Trevor Darrell, Sergey Levine, Jitendra Malik
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1806.08354

## 1. 논문 개요

이 논문은 로봇이 환경과 직접 상호작용하면서, 사람의 정답 마스크 없이도 시각 장면을 개별 object instance로 분할하는 방법을 배우게 할 수 있는지를 다룬다. 핵심 목표는 class-agnostic instance segmentation, 즉 특정 semantic class에 묶이지 않고 “이 장면에서 서로 다른 개체가 무엇인가”를 분리하는 능력을 self-supervised manner로 학습하는 것이다.

연구 문제는 분명하다. 기존의 강력한 instance segmentation 시스템은 보통 ImageNet 사전학습과 COCO 같은 대규모 human annotation에 크게 의존한다. 하지만 실제 embodied agent나 robot은 미리 정의된 class에만 의존해서는 안 되고, 새로운 물체와 새로운 배경에서도 스스로 적응해야 한다. 이 논문은 segmentation을 수동적 인식 과제가 아니라, agent가 자신의 가설을 행동으로 시험하고 수정하는 능동적 과정으로 재정의한다.

이 문제가 중요한 이유는 두 가지다. 첫째, 실제 세계의 agent는 annotation 없이도 새로운 환경에서 object-centric representation을 형성해야 한다. 둘째, 이런 representation은 단지 segmentation 자체를 위한 것이 아니라 이후 manipulation, control, physical reasoning으로 이어지는 기반 표현이 될 수 있다. 논문은 이 점을 보여주기 위해 downstream rearrangement task까지 평가한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 매우 단순하면서 강하다. agent가 어떤 픽셀 집합을 “하나의 물체일 것”이라고 가정한 뒤, 그 위치를 실제로 grasp해서 옮겨 보면, 진짜 물체였다면 그 픽셀 집합과 관련된 영역이 움직이고, 배경이었다면 거의 변화가 없을 것이다. 즉, “together move, together belong”이라는 common fate 원리를 robot interaction으로 실험하는 셈이다.

구체적으로는 현재 segmentation model이 생성한 object hypothesis 중 하나를 선택해 pick-and-place를 수행하고, 상호작용 전후 이미지 차이(frame difference)로 pseudo-mask를 만든다. 그 pseudo-mask를 다시 segmentation network 학습에 사용한다. 이렇게 하면 perception이 action을 만들고, action이 다시 perception supervision을 만든다.

기존 접근과의 차별점은 세 가지로 볼 수 있다. 첫째, 대규모 human mask annotation 없이 active interaction만으로 instance segmentation을 학습한다. 둘째, 단순히 비디오를 수동적으로 관찰하는 것이 아니라 agent가 스스로 어떤 object hypothesis를 시험할지 결정한다. 셋째, self-generated supervision이 심하게 noisy하다는 점을 전제로 하고, 이를 다루기 위한 **Robust Set Loss**를 제안한다. 이 손실은 픽셀별 정답을 정확히 맞추게 강제하는 대신, 예측 마스크와 noisy target mask가 집합 수준에서 충분한 overlap, 특히 IoU/Jaccard 기준을 만족하도록 유도한다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. Sawyer robot이 테이블 위 객체들을 관찰하고, 현재 이미지 $I_t$에서 segmentation network가 여러 object hypothesis $\{s_1^t, s_2^t, \dots, s_K^t\}$를 만든다. 여기서 각 $s_i^t$는 binary mask이다. 그중 하나를 무작위로 골라 grasp 후보로 삼고 pick-and-place를 수행한다. 상호작용 전후 이미지 차이에서 움직인 영역을 추출해 pseudo ground-truth mask를 만든다. 이 결과가 positive example이면 mask network와 scoring network를 업데이트하고, 비어 있거나 충분히 움직이지 않으면 negative example으로 사용한다.

논문은 DeepMask 프레임워크를 segmentation backbone으로 사용한다. 이 구조는 두 부분으로 나뉜다. 하나는 **scoring network**로, 어떤 patch의 중심이 foreground object에 속하는지 판단한다. 다른 하나는 **mask network**로, 해당 patch가 object를 포함한다고 판단되면 실제 segmentation mask를 예측한다. 입력 이미지는 여러 scale로 resize되고, $192 \times 192$ patch를 stride 16으로 슬라이딩하며 처리한다. backbone feature extractor로는 ResNet-18을 사용했고, batch size 32에 SGD with momentum으로 학습했다.

로봇 상호작용은 pick-and-place primitive로 이루어진다. pick은 arena 평면 위 2D 위치와 gripper rotation으로 parameterize된다. grasp 중간 시점의 이미지 $I'_{t+1}$와 최종 이미지 $I_{t+1}$를 포함해 세 장의 이미지를 촬영한다. difference image는 주로 $I_t$와 $I'_{t+1}$ 사이에서 계산하며, pick location 주변 $240 \times 240$ 지역에 한정해 noise를 줄인다. 연결된 변화 영역 중 픽셀 수가 1000 이상이면 positive example로 간주하고 해당 mask를 supervision으로 사용한다. 그렇지 않으면 background로 본다. 추가적으로 $I'_{t+1}$와 $I_{t+1}$ 쌍도 사용해 학습 예시를 늘린다.

이 논문의 핵심 기술적 기여는 **Robust Set Loss (RSL)**이다. 문제의식은 명확하다. interaction으로 만든 pseudo-mask는 자주 틀린다. grasp 실패로 일부만 움직이거나, 가까운 두 물체가 함께 움직여 두 개가 하나처럼 보이거나, 그림자와 조명 변화가 false mask를 만들 수 있다. 일반적인 pixel-wise cross entropy는 이런 noisy mask를 그대로 맞추려 하기 때문에 학습을 불안정하게 만들고 일반화를 방해한다.

저자들은 segmentation을 픽셀 독립 예측이 아니라 “픽셀 집합(set)”의 문제로 본다. 중요한 것은 각 픽셀이 완벽히 맞는지가 아니라, 예측 마스크와 noisy target이 전체적으로 충분히 겹치는가이다. 이를 위해 latent target mask $\hat{X}$를 직접 최적화 변수로 두고, 네트워크 출력 $Q(X \mid \theta, I)$가 이 latent target과 가깝도록 하되, latent target은 noisy mask $M_I$와 충분한 IoU를 만족하도록 제약을 건다.

최종 목적식은 다음과 같다.

$$
\min_{\theta, \hat{X}, \xi}
-\sum_i \log q_i(\hat{x}_i) + \lambda^T \xi
$$

subject to

$$
IoU(\hat{X}, M_I) \ge b - \xi,\quad \xi \ge 0
$$

여기서 $\hat{X}$는 latent discrete mask, $M_I$는 interaction으로 얻은 noisy mask, $\xi$는 slack variable, $b$는 요구되는 최소 IoU threshold이다. 직관적으로 말하면, 네트워크가 noisy mask를 한 픽셀도 틀리지 않게 모사할 필요는 없고, 적어도 일정 수준 이상 겹치기만 하면 된다. $b=1$이면 사실상 exact fitting이고, $b<1$이면 noise에 대한 margin을 허용하는 셈이다.

최적화는 alternating 방식으로 수행된다. 먼저 현재 네트워크 출력에 bias를 가감하여 IoU 제약을 만족하는 latent mask $\hat{X}$를 근사적으로 찾고, 그다음 이 $\hat{X}$를 ground truth처럼 사용해 네트워크를 업데이트한다. supplementary에 따르면 이 inner optimization은 이미지당 1ms 이하, batch 32 기준 약 0.35초 정도로 빠르다.

또 하나 중요한 점은 학습 초기 bootstrapping이다. 완전히 무지한 상태의 agent는 object hypothesis를 엉뚱하게 만들 것이고, 대부분 background만 집게 될 가능성이 높다. 이를 막기 위해 저자들은 기존 robotic pushing dataset을 이용해 [19]의 방법으로 물체 이동 기반 mask를 자동 추출하고, 이 데이터로 네트워크를 self-supervised pretraining한다. 즉, 초기에는 passive observation으로 시작하고, 이후 active interaction으로 발전시키는 구조다.

## 4. 실험 및 결과

실험 환경은 Sawyer robot, 평평한 wooden arena, 그리고 서로 다른 위치의 4개 카메라로 구성된다. 배경 texture는 쉽게 교체 가능하도록 설계되었고, 매 순간 4개에서 8개의 물체가 arena에 놓였다. 학습에는 36개 training objects, validation에는 8개, test에는 15개 또는 supplementary figure 기준 16개가 쓰였으며, 배경도 train/val/test가 분리되었다. 논문 본문에는 train background 24개, val 6개, test 10개라고 되어 있다. 로봇은 평균 분당 3회 상호작용했고, 총 50,000회 이상의 interaction을 수행했다. 또한 25회 상호작용마다 자동 reset을 수행해 데이터 상관성을 줄였다.

평가 지표는 standard mean Average Precision, 즉 mAP이며 IoU threshold 0.3과 0.5에서 비교했다. 비교 대상은 bottom-up proposal 방식인 GOP, 그리고 COCO 70만 개 이상의 strong mask supervision과 ImageNet pretraining을 사용하는 DeepMask이다. GOP와 DeepMask에는 domain-specific tuning도 적용한 버전을 별도로 비교했다.

핵심 정량 결과는 Table 1에 요약되어 있다. IoU 0.3 기준으로 GOP는 10.9, GOP tuned는 23.6, DeepMask는 44.5, DeepMask tuned는 61.8이다. 저자들의 self-supervised 기본 방법은 41.1 ± 2.4, **Robust Set Loss**를 추가하면 45.9 ± 2.1로 오른다. 즉, strong supervision 없이도 DeepMask의 기본 버전과 비슷하거나 약간 높은 수준까지 도달하며, 최소한 bottom-up GOP는 큰 차이로 앞선다. 반면 IoU 0.5에서는 Ours 16.0 ± 2.6, Ours + RSL 22.5 ± 1.3으로 개선되지만 DeepMask tuned 47.3에는 여전히 큰 격차가 있다. 이는 고품질 정밀 mask에서는 noisy pseudo-label의 한계가 남아 있음을 보여준다.

흥미로운 점은 소량의 clean annotation을 추가했을 때다. 1470장의 training image, 총 7946 object instances에 대한 human-labeled masks를 추가한 `Ours + Human`은 IoU 0.3에서 43.1 ± 2.6, IoU 0.5에서 21.1 ± 2.6을 보인다. 즉, 소량의 고품질 라벨이 noisy self-supervision을 보완할 수 있음을 보여준다. 다만 IoU 0.5에서는 RSL을 쓴 purely self-supervised setting이 오히려 약간 더 높게 나온다. 논문은 이 점을 깊게 분석하지는 않지만, 적어도 RSL이 noise handling에 꽤 중요한 역할을 한다는 것은 분명하다.

Figure 3(a)는 interaction 데이터가 늘어날수록 성능이 꾸준히 상승함을 보여준다. 특히 약 50K interaction 이후에는 GOP tuned를 확실히 넘는다. 중간중간 성능 하락 구간이 있는데, 저자들은 이를 active system 특유의 non-stationary data distribution 때문이라고 해석한다. 새로운 background가 도입되면 기존 모델이 잠시 overfit된 상태가 드러나 성능이 떨어지고, 이후 그 환경에서 더 상호작용하며 다시 적응한다는 설명이다.

Figure 3(b)는 active interaction의 질이 시간이 갈수록 좋아진다는 점을 보여준다. 저자들은 held-out test 환경에서 agent가 생성한 object hypothesis의 recall을 서로 다른 precision threshold에서 측정한다. 시간이 지날수록 recall이 꾸준히 증가하므로, agent는 더 많은 데이터만 모으는 것이 아니라 더 효율적인 실험, 즉 더 좋은 object hypothesis를 생성하는 방향으로 개선되고 있다고 해석할 수 있다. 이는 passive data collection과의 중요한 차별점이다.

정성적 결과에서도 의미 있는 패턴이 보인다. Figure 5에서 저자들의 방법은 새로운 객체와 새로운 배경에서 reasonably good masks를 생성하며, GOP보다 recall이 훨씬 높다. DeepMask와 질적으로 비슷한 수준의 예도 많다. 다만 주요 failure mode는 작은 disconnected pixels가 섞인 jittery mask 생성이다. Figure 4는 interaction 수가 늘어날수록 false positive background segmentation이 줄고 object mask 질이 좋아지는 과정을 시각적으로 보여준다.

일반화 분석도 흥미롭다. 저자들은 네 가지 조건, 즉 training object/training background, training object/test background, test object/training background, test object/test background를 비교했다. IoU 0.3에서는 새로운 background로 바뀔 때 성능 하락이 더 커서, 모델이 object 쪽 일반화는 상대적으로 더 잘하고 background 변화에는 더 민감하다는 결과가 나온다. 그러나 IoU 0.5에서는 반대 경향이 나온다. 즉, 거친 수준의 objectness는 새로운 object에도 비교적 잘 일반화되지만, mask 경계 품질까지 엄격히 요구하면 object shape 다양성이 더 중요해진다고 볼 수 있다.

마지막으로 downstream task인 object rearrangement를 평가했다. 현재 이미지와 목표 이미지가 주어졌을 때, segmentation으로 물체 후보를 뽑고, 각 segment를 crop한 뒤 AlexNet feature로 current-target 간 object matching을 수행한다. 그다음 pick-and-place로 목표 위치에 가까워질 때까지 옮긴다. 로봇은 최대 10번 상호작용할 수 있다. 논문은 qualitative 결과만 제시하며, 성공 사례도 있지만 실패도 적지 않다고 설명한다. 저자들은 대부분의 실패가 segmentation 자체보다 feature matching이나 grasping failure에서 온다고 주장하며, explicit instance segmentation을 쓰지 않는 inverse-model baseline보다 더 낫다고 보고한다. 다만 이 downstream 평가는 정량성이 약하고, segmentation 기여를 정확히 분리해서 측정한 것은 아니다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 instance segmentation을 annotation 기반 정적 인식 문제에서, interaction 기반 self-supervised learning 문제로 전환했다는 점이다. 단순히 “라벨 없이 학습했다”가 아니라, agent의 행동이 supervision source가 되는 구조를 실제 로봇 시스템으로 구현했다는 점에서 의미가 크다. 또한 50K 이상의 interaction 데이터와 1700장 규모의 human-labeled benchmark를 공개해 후속 연구 기반을 제공했다는 점도 실용적 기여다.

두 번째 강점은 **Robust Set Loss**다. segmentation noise는 본질적으로 set-level 문제인데, 이를 pixel-wise robust loss로 단순 처리하지 않고 IoU 제약 기반 latent target optimization으로 다룬 점이 논리적으로 잘 맞는다. 실제로 성능 평균을 끌어올리고 분산도 줄였다는 결과가 Table 1에서 확인된다.

세 번째 강점은 learned segmentation의 utility를 downstream control task에 연결했다는 점이다. 비록 rearrangement 실험은 완전히 정교하진 않지만, object-centric representation이 manipulation에 쓸모 있다는 논지에는 설득력을 준다.

반면 한계도 분명하다. 첫째, pseudo-mask 품질이 여전히 충분히 높지 않다. IoU 0.3에서는 인상적이지만 IoU 0.5에서는 strong supervision과 차이가 크다. 이는 정확한 경계 품질이나 복잡한 scene에서의 정교한 segmentation에는 아직 부족하다는 뜻이다.

둘째, 상호작용 파이프라인이 완전히 end-to-end learned system은 아니다. motion planning과 pick-and-place procedure는 hand-engineered이며, grasp orientation도 PCA 기반 heuristic을 사용한다. 논문의 목적이 interaction으로 segmentation을 배우는 것임은 맞지만, “agent가 전부 스스로 배운다”는 수준은 아니다.

셋째, 실험 도메인이 tabletop manipulation에 제한되어 있다. 배경 texture를 바꾸고 novel objects를 평가했지만, 여전히 controlled arena 환경이며 카메라 네 대, 고정된 manipulation primitive, 비교적 제한된 object distribution 안에서의 검증이다. 실제 복잡한 실내 환경이나 clutter가 심한 장면으로 일반화된다고 보기는 어렵다.

넷째, downstream rearrangement 실험은 정량 평가가 부족하다. 논문은 많은 실패가 feature matching이나 grasping failure 때문이라고 말하지만, 실제로 segmentation 품질이 control success에 얼마나 기여했는지 분해된 실험은 부족하다. 따라서 “segmentation이 downstream control에 실질적으로 얼마나 충분한가”는 아직 열린 질문으로 남는다.

다섯째, 학습 초기 bootstrap을 위해 별도 pushing dataset과 [19] 방식의 passive self-supervision을 사용한다. 이것도 사람이 라벨링하지는 않지만, 완전한 zero-prior active learning은 아니다. 즉, 학습이 완전히 interaction만으로 시작되는 것은 아니다.

비판적으로 보면, 이 논문은 segmentation 성능 자체로는 당시 strong supervised SOTA를 넘지 못하지만, 그보다 더 중요한 방향 전환을 제시했다. object segmentation을 “주어진 데이터셋에서 잘 맞히는 문제”가 아니라 “agent가 세계를 분해하는 기본 단위를 스스로 형성하는 문제”로 본 점이 핵심 가치다. 따라서 이 논문은 최종 성능보다 연구 방향성과 문제 설정에서 더 큰 의미를 가진다.

## 6. 결론

이 논문은 로봇이 환경과 상호작용하며 self-supervised하게 class-agnostic instance segmentation을 학습할 수 있음을 보였다. agent는 현재 segmentation hypothesis를 기반으로 물체를 집으려 시도하고, 그 결과 생기는 motion cue로 pseudo-mask를 얻어 다시 segmentation network를 개선한다. 이 과정에서 noisy supervision 문제를 해결하기 위해 **Robust Set Loss**를 제안했고, 실제로 성능 향상과 안정성 개선을 보였다.

정량적으로는 bottom-up GOP를 크게 앞서고, 일부 조건에서는 strong supervision으로 학습된 DeepMask 기본 버전에 근접하는 결과를 얻었다. 특히 대규모 human annotation 없이 novel objects와 backgrounds에 일반화하는 segmentation을 얻었다는 점이 중요하다. 또한 learned segmentation이 rearrangement 같은 downstream visuomotor task에 활용될 가능성도 확인했다.

향후 연구 측면에서 이 논문은 세 가지 방향에 영향을 줄 수 있다. 첫째, interaction-driven perception learning의 확대다. 둘째, noisy self-supervision을 다루는 set-level loss 설계의 발전이다. 셋째, object-centric representation을 perception과 control의 공통 인터페이스로 삼는 embodied AI 연구다. 정리하면, 이 논문은 성능 숫자만으로 평가하기보다, “행동을 통해 object를 배운다”는 학습 패러다임을 실제 로봇 실험으로 설득력 있게 제시했다는 점에서 중요한 역할을 한다.
