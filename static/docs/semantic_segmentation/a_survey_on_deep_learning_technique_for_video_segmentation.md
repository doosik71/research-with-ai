# A Survey on Deep Learning Technique for Video Segmentation

- **저자**: Tianfei Zhou, Fatih Porikli, David J. Crandall, Luc Van Gool, Wenguan Wang
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2107.01153

## 1. 논문 개요

이 논문은 새로운 video segmentation 알고리즘 하나를 제안하는 연구가 아니라, 딥러닝 기반 video segmentation 분야 전체를 체계적으로 정리한 **survey paper**이다. 저자들은 비디오 분할을 크게 두 축으로 나눈다. 하나는 **Video Object Segmentation (VOS)** 로, 의미 카테고리를 미리 알지 못해도 비디오 안의 지배적인 foreground object를 분리하는 문제이다. 다른 하나는 **Video Semantic Segmentation (VSS)** 로, car, person, road처럼 미리 정의된 semantic category를 각 픽셀에 할당하는 문제이다.

논문이 다루는 연구 문제는 단순히 “어떤 모델이 더 정확한가”가 아니다. 저자들은 video segmentation을 구성하는 **문제 정의, taxonomy, inference mode, learning paradigm, 대표 알고리즘 계보, 데이터셋, 평가 지표, 성능 비교, 향후 연구 방향**까지 한 흐름으로 정리한다. 특히 VOS 내부에서도 automatic, semi-automatic, interactive, language-guided처럼 서로 다른 세팅이 섞여 사용되던 점을 분명히 구분하고, VSS 역시 semantic, instance, panoptic으로 세분화한다.

이 문제가 중요한 이유는 응용 범위가 매우 넓기 때문이다. 논문은 autonomous driving, robotics, surveillance, movie production, augmented reality, video conferencing 등을 예로 든다. 또한 이미지 분할보다 video segmentation이 더 어려운 이유는 시간축이 추가되기 때문이다. 빠른 motion, occlusion, deformation, appearance change, long-term consistency, latency 제약 같은 요소가 동시에 작동한다. 따라서 이 survey는 단순 정리 이상의 가치가 있으며, 새로 진입하는 연구자가 분야 지형을 이해하는 데 매우 유용하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 video segmentation 연구를 단일 문제처럼 보지 않고, **출력 공간과 인간 개입 정도, 학습 방식**에 따라 구조적으로 분해해 보는 것이다. 저자들은 우선 입력 공간 $X$ 와 출력 공간 $Y$ 를 두고, 이상적인 비디오-분할 함수 $f^*: X \to Y$ 를 학습하는 문제로 video segmentation을 정식화한다. 그런 뒤 $Y$ 가 binary foreground/background이면 VOS, multi-class semantic parsing이면 VSS로 본다.

또 하나의 중요한 아이디어는 **inference mode** 와 **learning paradigm** 을 분리해서 설명하는 것이다. 예를 들어 VOS에서 “unsupervised video segmentation”이라는 말은 전통적으로 사람이 입력을 주지 않는 automatic inference를 뜻해 왔지만, machine learning의 “unsupervised learning”과 쉽게 혼동된다. 저자들은 이런 용어 혼란을 지적하면서, inference 관점에서는 **automatic / semi-automatic / interactive**, 학습 관점에서는 **supervised / unsupervised(self-supervised) / weakly supervised**로 구분해야 한다고 주장한다. 이 정리는 분야 전체를 이해하는 데 매우 중요하다.

기존 survey와의 차별점은 범위와 시대성에 있다. 저자들에 따르면 기존 survey는 대체로 deep learning 이전 시기를 다루거나, foreground/background segmentation처럼 좁은 범위만 다루었다. 반면 이 논문은 modern deep learning era 이후의 주요 방법들을 폭넓게 포괄하고, VOS와 VSS를 함께 다루며, 20개 데이터셋과 여러 benchmark를 직접 비교한다. 즉, 이 survey의 핵심 기여는 “개별 모델 설명”보다도 **분야 전체의 공통 구조를 드러내는 통합 관점**에 있다.

## 3. 상세 방법 설명

이 논문은 survey이므로 하나의 파이프라인이나 하나의 loss function을 제안하지 않는다. 대신 분야 전체를 설명하기 위한 방법론적 틀을 제시한다.

먼저 문제를 다음처럼 일반화한다. 입력 공간 $X$ 와 출력 공간 $Y$ 에 대해 목표는 이상적 함수 $f^*$ 를 근사하는 것이다. supervised setting에서는 훈련 샘플 $\{(x_n, y_n)\}$ 이 주어지고, 경험적 위험 최소화는 다음처럼 쓸 수 있다.

$$
\tilde f \in \arg\min_{f \in F} \frac{1}{N}\sum_n \varepsilon(f(x_n), z(x_n))
$$

여기서 $F$ 는 hypothesis space이고, $\varepsilon$ 는 error function이다. fully supervised일 때는 보통 $z(x_n)=f^*(x_n)=y_n$ 이다. 반면 unsupervised/self-supervised에서는 $z(x_n)$ 가 사람이 준 정답이 아니라 cross-frame consistency 같은 비디오의 내재적 구조에서 얻은 pseudo label이 된다. weakly supervised에서는 tag, box, scribble처럼 더 싸게 얻을 수 있는 annotation이 $Z$ 역할을 한다. 이 수식은 survey 전체에서 공통되는 학습 관점을 제공한다.

이후 저자들은 VOS와 VSS를 각각 세분화한다.

VOS는 사람 개입 정도에 따라 세 갈래로 나뉜다. **AVOS**는 입력이 비디오 $V$ 만 있는 경우로, $X \equiv V$ 이다. 사람의 초기화 없이 foreground object를 찾는다. **SVOS**는 첫 프레임 mask, bounding box, scribble, 혹은 language description 같은 제한된 human input이 함께 주어지며, $X = V \times M$ 으로 본다. **IVOS**는 분석 과정 중 여러 차례 사용자가 scribble 같은 제약을 계속 넣는 구조이며, $X = V \times S$ 로 정리한다. 이 정리는 실용적이다. 왜냐하면 AVOS는 자동화에 적합하지만 원하는 임의 대상을 고르기 어렵고, SVOS와 IVOS는 유연하지만 human-in-the-loop 비용이 들기 때문이다.

AVOS 내부에서는 딥러닝 방법이 다시 다섯 갈래 정도로 정리된다. 초기 방법은 optical flow나 proposal, handcrafted prior 위에 얕은 neural module을 얹는 수준이었다. 그 다음으로 pixel instance embedding을 먼저 만들고 clustering과 label propagation으로 foreground/background를 나누는 방식이 등장했다. 이후 mainstream는 end-to-end 방식으로 이동한다. 여기에는 짧은 시간 구간만 쓰는 two-stream network나 recurrent model이 있고, 더 최근에는 Siamese correlation, co-attention, memory, Transformer 등을 이용해 장기 문맥을 인코딩하는 방향으로 진화했다. 저자들의 관점에서 최근 강한 AVOS는 단순히 인접 프레임만 보는 것이 아니라, 여러 프레임 사이의 대응과 전역 맥락을 적극적으로 활용한다.

SVOS는 첫 프레임 supervision을 어떻게 사용하는지에 따라 분류된다. **Online fine-tuning based** 방법은 테스트 시 주어진 첫 mask로 모델을 다시 미세조정한다. OSVOS 계열이 대표적이며, target-specific adaptation이 강점이지만 테스트 시간이 길고 하이퍼파라미터 설계가 까다롭다. **Propagation-based** 방법은 이전 프레임의 mask를 현재 프레임으로 전파한다. 이 방식은 설계가 직관적이지만 occlusion이나 drift가 누적되면 에러 전파가 심해질 수 있다. **Matching-based** 방법은 첫 프레임 또는 과거 프레임의 target feature를 외부 memory 혹은 embedding space에 명시적으로 저장해 두고, 각 픽셀을 similarity 기반으로 분류한다. 논문은 이 계열을 가장 유망한 방향으로 본다. 특히 STM(space-time memory network) 이후 많은 상위권 방법이 memory-augmented architecture를 채택했다고 정리한다. 이 설명은 survey 전체에서 매우 중요한 부분이다. 실제로 저자들은 SVOS의 성능 향상이 implicit target encoding에서 explicit memory matching으로 옮겨간 역사와 맞물려 있다고 본다.

IVOS는 대체로 **interaction-propagation** 구조를 따른다. 사용자가 한 프레임에 scribble을 주면 interactive image segmentation module이 먼저 해당 프레임 마스크를 만들고, propagation module이 이를 전체 비디오로 퍼뜨린다. 이후 사용자가 다시 수정 scribble을 주면 반복한다. 최근 방법은 매 라운드마다 전체 비디오를 다시 무겁게 추론하지 않도록, 공통 feature encoder를 한 번만 돌리고 그 위에 가벼운 interaction branch와 propagation branch를 얹는 구조를 사용한다. 이 설계는 “사람이 조금씩 고쳐 가면서 빠르게 반응받고 싶다”는 IVOS의 실제 사용 시나리오와 맞닿아 있다.

LVOS는 language-guided VOS로, 시각 특징과 문장 표현을 어떻게 결합하는지가 핵심이다. survey는 이를 **dynamic convolution**, **capsule routing**, **attention-based** 세 부류로 나눈다. dynamic convolution은 텍스트에서 생성된 filter로 visual feature를 변환하는 방식인데, 문장 표현의 작은 변형에 민감할 수 있다는 한계를 논문은 지적한다. attention-based 방법은 더 전역적인 visual-text correlation을 포착한다.

VSS 쪽은 semantic, instance, panoptic으로 갈라진다. **Video Semantic Segmentation** 에서는 두 흐름이 있다. 하나는 여러 프레임의 정보를 모아 더 정확한 분할을 하는 방향이고, 다른 하나는 keyframe이나 feature reuse를 이용해 더 빠르게 만드는 방향이다. 전자는 optical flow-guided feature aggregation, recurrent propagation, CRF reasoning 같은 기법을 사용하고, 후자는 keyframe selection, partial execution, feature warping, knowledge distillation 등을 활용한다. 논문은 이 두 방향이 정확도와 속도라는 상충 목표를 각각 최적화한다고 본다.

**VIS**는 detection, segmentation, tracking을 동시에 해야 하므로 더 복잡하다. survey는 이를 track-detect, clip-match, propose-reduce, segment-as-a-whole 네 패러다임으로 구분한다. 앞의 세 방식은 대체로 “부분 결과를 만든 뒤 나중에 연결하는” 구조라서 merge 과정에서 오류 누적이 발생할 수 있다. 반면 Transformer 기반의 segment-as-a-whole은 전체 시퀀스를 직접 예측하는 더 우아한 접근으로 소개된다.

**VPS**는 모든 foreground instance tracklet과 background region을 함께 다루며, semantic label까지 부여하는 가장 포괄적인 설정이다. 이 경우 단일 프레임 panoptic segmentation을 비디오로 확장하면서 temporal feature fusion과 cross-frame association이 추가된다.

## 4. 실험 및 결과

이 논문은 survey이지만, 단순 문헌 나열에 그치지 않고 주요 benchmark를 모아 성능 비교까지 제공한다. 데이터셋 부분에서는 총 20개의 대표 benchmark를 정리한다. VOS 계열에서는 Youtube-Objects, FBMS 59, DAVIS 16, DAVIS 17, YouTube-VOS가 핵심이며, LVOS에서는 A2D Sentence, J-HMDB Sentence, DAVIS 17-RVOS, Refer-Youtube-VOS가 소개된다. VSS 쪽에서는 CamVid, CityScapes, NYUDv2, VSPW가 중요하고, VIS는 YouTube-VIS, KITTI MOTS, MOTSChallenge, BDD100K, OVIS, VPS는 VIPER-VPS와 Cityscapes-VPS가 대표적이다. 이 정리는 데이터셋 규모, annotation 유형, 용도를 한눈에 비교하게 해 준다.

평가 지표도 분야별로 정리한다. object-level AVOS와 SVOS에서는 보통 region similarity $J$ 와 boundary accuracy $F$ 를 쓴다. $J$ 는 IoU이며 다음과 같다.

$$
J = \frac{|\hat Y \cap Y|}{|\hat Y \cup Y|}
$$

여기서 $\hat Y$ 는 예측 마스크, $Y$ 는 ground truth이다. $F$ 는 boundary precision $P_c$ 와 recall $R_c$ 의 조화평균이다.

$$
F = \frac{2P_cR_c}{P_c + R_c}
$$

AVOS에서는 여기에 temporal stability $T$ 도 사용한다. IVOS는 시간 제약이 중요하므로 AUC와 $J@60$ 를 사용한다. LVOS는 overall IoU, mean IoU, Precision@K, mAP를 쓴다. VSS는 IoU가 기본이며, Cityscapes에서는 coarse한 category와 finer class 두 수준의 IoU를 모두 보고한다. VPS는 video panoptic quality인 VPQ를 사용하며, 시간 창 길이 $k$ 에 따라 clip 단위 tube matching을 수행한 뒤 평균낸다.

정량 결과를 보면, 이 survey가 작성될 당시 object-level AVOS의 DAVIS 16 validation에서는 **RTNet** 이 $J=85.6$ 으로 가장 강한 성능을 보였다. instance-level AVOS의 DAVIS 17 validation에서는 **UnOVOST** 가 $J\&F=67.9$ 로 최고였다. SVOS의 DAVIS 17 validation에서는 **LCM** 과 **RMNet** 이 모두 $J\&F=83.5$ 수준으로 최상위였고, **EGMN**, **SST**, **GIEL**, **CFBI**, **STM** 등도 모두 memory-based 계열로 강세를 보였다. 이 결과는 저자들의 “memory-augmented matching 구조가 SVOS의 주류”라는 분석을 뒷받침한다.

IVOS의 DAVIS 17 validation에서는 **MiVOS** 가 AUC 84.9, $J@60$ 85.4로 가장 좋았다. LVOS의 A2D Sentence test에서는 **CST** 가 mean IoU 56.1과 mAP 39.9로 가장 좋은 성능을 보였다. VSS의 Cityscapes validation에서는 **EFC** 가 IoU class 83.5로 최고였다. VIS의 YouTube-VIS validation에서는 **Propose-Reduce** 가 mAP 47.6으로 가장 강했고, Transformer 기반 **VisTR** 역시 큰 향상을 보여 주었다. VPS의 Cityscapes-VPS test에서는 **ViP-DeepLab** 이 종합 VPQ 62.5로 가장 높았다.

저자들이 실험 파트에서 특히 강조하는 해석은 두 가지이다. 첫째, 여러 분야에서 성능은 많이 올랐지만 **재현성 문제가 심각하다**는 점이다. 코드, segmentation mask, 실험 설정, backbone 차이가 통일되어 있지 않아 공정 비교가 어렵다고 지적한다. 둘째, 정확도는 많이 보고되지만 **실행 시간과 메모리 사용량 보고가 매우 부족하다**는 점이다. 이는 실제 시스템 배치, 특히 mobile, autonomous driving처럼 자원이 제한된 환경에서는 큰 문제라고 본다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 범위가 넓으면서도 구조가 명확하다는 점이다. 단순히 많은 논문을 나열하지 않고, VOS와 VSS를 상위 축으로 두고 다시 AVOS, SVOS, IVOS, LVOS, VIS, VPS 등으로 분기시켜 독자가 전체 지형을 이해하게 만든다. 또한 inference mode와 learning paradigm을 구분하고, 용어 혼란까지 바로잡으려 한다는 점이 학술적으로 유익하다.

둘째 강점은 survey이면서도 실제 benchmark 표를 풍부하게 제공한다는 점이다. 단순 서술형 review가 아니라 데이터셋, 대표 모델, 핵심 architecture, optical flow 사용 여부, 학습 데이터, 성능 수치까지 표로 정리해 놓았다. 따라서 “어느 분야에서 어떤 계열이 주류가 되었는가”를 빠르게 파악할 수 있다. 예를 들어 SVOS에서 memory-based matching이 강하고, VIS에서 Transformer와 sequence-level reasoning이 부상하고 있다는 흐름이 표와 본문에서 모두 드러난다.

셋째 강점은 미래 연구 방향 제안이 비교적 구체적이라는 점이다. long-term video segmentation, open-world video segmentation, annotation-efficient learning, adaptive computation, neural architecture search, sub-field 간 협력 같은 제안은 단순한 구호가 아니라 현재 benchmark와 기술 흐름의 한계를 바탕으로 나온다.

한편 한계도 분명하다. 먼저 이 논문은 survey이기 때문에 새로운 통합 모델이나 이론을 제안하지 않는다. 따라서 독자가 “어떤 단일 알고리즘을 구현해야 하는가”를 직접 얻기는 어렵다. 둘째, 논문이 정리한 성능 비교는 매우 유용하지만, 저자들 스스로 인정하듯 서로 다른 코드베이스와 최적화 수준 때문에 runtime 비교는 엄밀하지 않다. 즉 표의 FPS는 참고용이지 완전한 apples-to-apples 비교는 아니다. 셋째, 이 survey는 2022년 시점까지의 흐름을 정리한 것으로 보이며, 이후 급속히 발전한 large-scale video foundation model이나 더 강한 Transformer 계열 흐름은 포함되지 않는다. 이것은 논문의 잘못이라기보다 작성 시점의 한계이다.

비판적으로 보면, survey의 분류 체계는 매우 유용하지만 일부 경계는 실제로 겹친다. 예를 들어 SVOS의 propagation, matching, online fine-tuning은 실제 최신 모델 안에서 혼합되어 쓰이는 경우가 많다. 저자들도 이를 알고 있지만, 분류를 위해 대표 메커니즘 중심으로 정리한다. 따라서 이 taxonomy는 “엄밀한 상호배타 분류”라기보다 **주된 설계 철학에 따른 정리**로 이해하는 것이 적절하다.

## 6. 결론

이 논문은 deep learning 기반 video segmentation 분야를 매우 폭넓고 체계적으로 정리한 대표적인 survey이다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, video segmentation을 VOS와 VSS라는 큰 틀 아래에서 다시 세부 task로 정리했다. 둘째, 150개 이상의 모델과 20개 데이터셋을 구조적으로 비교했다. 셋째, 성능 benchmark와 함께 재현성, 속도, 메모리, annotation cost, open-world generalization 같은 앞으로의 핵심 문제를 선명하게 드러냈다.

실제 적용 측면에서 이 survey는 매우 유용하다. 비디오 편집, 자율주행, 로보틱스, 인터랙티브 도구, 멀티모달 질의 기반 분할 등 서로 다른 응용이 왜 서로 다른 task setting을 요구하는지 이해하게 해 준다. 향후 연구 측면에서는 long-term reasoning, open-world robustness, annotation-efficient learning, adaptive computation이 중요하다는 점을 설득력 있게 보여 준다. 즉 이 논문은 한 시대의 결과를 정리하는 데 그치지 않고, video segmentation이 앞으로 어디로 가야 하는지를 제시하는 로드맵 역할도 한다.
