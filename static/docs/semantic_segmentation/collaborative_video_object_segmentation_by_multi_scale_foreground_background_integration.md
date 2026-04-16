# Collaborative Video Object Segmentation by Multi-Scale Foreground-Background Integration

- **저자**: Zongxin Yang, Yunchao Wei, Yi Yang
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2010.06349

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation(VOS)를 다룬다. 문제 설정은 첫 번째 프레임에서 주어진 object mask를 바탕으로, 이후 전체 비디오 프레임에서 같은 객체를 정확히 분할하는 것이다. 저자들은 기존 embedding-based VOS 방법들이 주로 foreground object의 특징을 잘 찾는 데 집중해 왔고, background는 상대적으로 덜 중요하게 취급했다고 지적한다. 하지만 실제 비디오에는 foreground와 비슷한 모양이나 같은 category의 객체가 background에 함께 존재하는 경우가 많기 때문에, background를 제대로 모델링하지 않으면 foreground와 background를 혼동하는 문제가 쉽게 발생한다.

논문의 핵심 목표는 foreground만 잘 찾는 것이 아니라, foreground와 그에 대응하는 background를 함께 학습하여 두 영역을 더 잘 구분할 수 있는 embedding을 만들고, 이를 통해 더 정확하고 더 강건한 VOS를 구현하는 것이다. 이를 위해 저자들은 CFBI(Collaborative video object segmentation by Foreground-Background Integration)와 확장판 CFBI+를 제안한다. 특히 CFBI+는 multi-scale matching과 Atrous Matching을 추가하여 정확도뿐 아니라 연산 효율도 개선하려고 한다.

이 문제가 중요한 이유는 VOS가 augmented reality, self-driving, video instance segmentation, interactive video object segmentation 같은 여러 응용의 기반 기술이기 때문이다. 또한 기존 고성능 방식들 중 일부는 test-time fine-tuning이나 대규모 simulated data pre-training에 크게 의존했는데, 이는 실사용에서 속도나 학습 복잡도 측면의 부담이 크다. 이 논문은 그런 추가 장치 없이도 강한 성능을 보인다는 점을 강조한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 foreground와 background를 경쟁적인 관계로 두고 함께 embedding하는 것이다. 저자들의 직관은 단순하다. foreground를 잘 분할하려면 foreground를 잘 찾는 것만으로는 부족하고, 무엇이 background인지를 명확히 아는 것도 똑같이 중요하다. 특히 비슷한 객체가 장면 안에 여러 개 있을 때, foreground-only matching은 쉽게 혼동될 수 있다. 논문은 이런 문제를 “background confusion”으로 설명한다.

이를 해결하기 위해 CFBI는 두 가지 수준에서 foreground-background integration을 수행한다. 첫째, pixel-level matching에서 foreground pixel뿐 아니라 background pixel도 reference frame과 previous frame에서 함께 사용한다. 둘째, instance-level attention을 통해 foreground/background, first/previous frame에서 얻은 요약 정보를 channel-wise attention 형태로 prediction network에 주입한다. 즉, 세밀한 local correspondence는 pixel-level에서 처리하고, 더 큰 receptive field와 object-scale context는 instance-level guidance로 보완한다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, background embedding을 명시적으로 도입했다는 점이다. 둘째, pixel-level matching과 instance-level attention을 함께 사용해 다양한 object scale에 대응한다는 점이다. 셋째, CFBI+에서는 multi-scale matching과 Atrous Matching을 결합하여, 고해상도 matching의 장점을 유지하면서도 계산량을 줄이는 구조를 제안한다는 점이다. 논문은 이러한 설계가 간단하면서도 효과적인 baseline이 될 수 있다고 주장한다.

## 3. 상세 방법 설명

전체 구조는 첫 번째 프레임 $t=1$, 이전 프레임 $t=T-1$, 현재 프레임 $t=T$를 함께 사용하는 형태다. 먼저 backbone network가 각 프레임에서 pixel-level embedding을 추출한다. 그다음 첫 프레임과 이전 프레임의 embedding을 mask를 기준으로 foreground와 background로 분리한다. 이후 이 정보는 두 갈래로 사용된다. 하나는 pixel-level matching이고, 다른 하나는 instance-level attention이다. 마지막으로 collaborative ensembler가 이들을 종합해 현재 프레임의 segmentation을 예측한다.

### Collaborative Pixel-level Matching

논문은 기존 FEELVOS의 global/local matching을 확장하되, foreground와 background를 모두 고려한다. 현재 프레임의 pixel $p$와 reference 또는 previous frame의 pixel $q$ 사이의 거리는 embedding $e_p, e_q$를 이용해 정의된다. 중요한 점은 foreground pixel과 background pixel에 대해 서로 다른 bias를 둔다는 것이다.

$$
D(p, q) =
\begin{cases}
1 - \frac{2}{1 + \exp(\|e_p - e_q\|^2 + b_B)} & \text{if } q \in B_t \\
1 - \frac{2}{1 + \exp(\|e_p - e_q\|^2 + b_F)} & \text{if } q \in F_t
\end{cases}
$$

여기서 $b_B$와 $b_F$는 각각 background bias와 foreground bias이며 학습 가능한 변수다. 이 식의 의미는 단순한 embedding distance를 쓰는 것이 아니라, foreground와 background 사이의 거리 해석 방식을 분리해서 학습하겠다는 것이다. 논문은 이것이 foreground distance와 background distance를 अलग-अलग 취급하게 만들어 성능 향상에 기여한다고 본다.

global matching은 현재 프레임의 pixel이 첫 프레임의 특정 object foreground pixel 집합과 얼마나 가까운지를 본다.

$$
G_o(p) = \min_{q \in P_{1,o}} D(p, q)
$$

반대로 relative background에 대한 global matching도 정의한다.

$$
\bar{G}_o(p) = \min_{q \in \bar{P}_{1,o}} D(p, q)
$$

여기서 $P_{1,o}$는 첫 프레임에서 object $o$의 foreground pixel 집합이고, $\bar{P}_{1,o}$는 그 object 기준 background pixel 집합이다. 즉 현재 pixel이 해당 object foreground와 닮았는지, 아니면 그 object의 background와 닮았는지를 둘 다 본다.

local matching은 이전 프레임과 현재 프레임 사이의 temporal continuity를 이용한다. 그런데 논문은 object motion이 비디오마다 크게 다르기 때문에, 하나의 fixed local window만 쓰는 것은 충분하지 않다고 지적한다. 그래서 여러 개의 window size를 동시에 사용하는 multi-local matching을 제안한다.

$$
ML_o(p, K) = \{L_o(p, k_1), L_o(p, k_2), \dots, L_o(p, k_n)\}
$$

여기서 각 $L_o(p, k)$는 반경 $k$ 안에서 이전 프레임의 object pixel들과의 최소 거리를 뜻한다.

$$
L_o(p, k) =
\begin{cases}
\min_{q \in P^{p,k}_{T-1,o}} D(p, q) & \text{if } P^{p,k}_{T-1,o} \neq \emptyset \\
1 & \text{otherwise}
\end{cases}
$$

background에 대해서도 같은 방식으로 multi-local matching을 정의한다. 이 설계의 목적은 느리게 움직이는 object와 빠르게 움직이는 object를 모두 안정적으로 처리하는 것이다. 논문은 큰 window의 intermediate result를 작은 window 계산에 재사용해서 추가 계산량을 크게 늘리지 않는다고 설명한다.

또한 현재 프레임 feature뿐 아니라, 이전 프레임의 embedding과 mask도 함께 concat한다. 저자들은 previous mask는 FEELVOS에서 이미 효과가 입증되었고, previous embedding을 추가하면 $J \& F$ 기준 약 0.5% 성능 향상이 있었다고 보고한다.

### Collaborative Instance-level Attention

pixel-level matching만으로는 큰 물체나 넓은 영역의 구조적 정보를 충분히 처리하기 어렵고, pixel-wise diversity 때문에 noise가 생길 수 있다. 이를 보완하기 위해 논문은 instance-level attention을 추가한다.

방법은 비교적 간단하다. 첫 프레임과 이전 프레임의 embedding을 foreground와 background로 나눈 뒤, 각각에 대해 channel-wise average pooling을 수행한다. 그러면 총 네 개의 instance-level embedding vector가 생긴다.

- 첫 프레임 foreground
- 첫 프레임 background
- 이전 프레임 foreground
- 이전 프레임 background

이 네 벡터를 concat해서 하나의 collaborative instance-level guidance vector를 만든다. 이 벡터는 object 전체 수준의 정보를 요약한 것이라고 볼 수 있다. 이후 이 벡터를 fully connected layer와 non-linearity를 거쳐 gate로 만들고, collaborative ensembler 내부의 각 Res-Block 입력 채널을 channel-wise로 조절한다. 구조적으로는 SE-Net류의 attention과 유사하지만, 논문은 두 개 FC 대신 한 개 FC가 더 좋았다고 명시한다.

이 attention의 의미는 “현재 segmentation head가 어떤 채널을 더 강조해야 하는가”를 object-level context로 알려주는 것이다. 저자들은 instance-level 정보가 큰 receptive field를 제공하므로 local ambiguity를 줄이는 데 유용하다고 설명한다.

### Collaborative Ensembler

CFBI의 최종 prediction head는 collaborative ensembler(CE)다. 이 모듈은 pixel-level과 instance-level 정보를 합쳐 최종 segmentation mask를 예측한다. 설계는 ResNet과 DeepLab 계열의 아이디어를 따른다. downsample-upsample 구조를 사용하고, 세 stage의 Res-Block과 ASPP(Atrous Spatial Pyramid Pooling), Decoder로 구성된다.

Stage 1, 2, 3의 Res-Block 개수는 각각 2, 3, 3개다. 일부 convolution에는 dilated convolution을 사용하여 receptive field를 효율적으로 키운다. Stage 2와 3의 시작에서는 stride 2 downsampling을 적용한다. 이후 ASPP와 Decoder를 통해 더 넓은 context를 보고, low-level backbone feature와 결합해 세밀한 경계를 복원한다.

논문은 이 구조가 foreground/background, pixel-level/instance-level 관계를 명시적 규칙으로 결합하기보다, 넓은 receptive field를 가진 segmentation head가 이를 암묵적으로 통합하게 만든다고 설명한다.

### CFBI+: Multi-scale Matching과 Atrous Matching

CFBI는 stride 4 수준의 고해상도 matching을 사용하기 때문에 정확하지만 메모리와 연산 비용이 크다. 이를 개선하기 위해 CFBI+는 multi-scale matching을 도입한다. backbone으로부터 stride $4, 8, 16$의 세 가지 scale feature를 뽑고, FPN으로 이를 융합한 뒤 각 scale에서 matching을 수행한다. 각 scale의 출력은 CE의 대응 stage로 들어간다.

핵심은 coarse-to-fine 전략이다. 작은 scale에서는 semantic이 풍부하고 큰 scale에서는 spatial detail이 좋으므로, 각 scale의 장점을 결합해 더 robust한 matching을 만든다. 동시에 채널 수를 scale별로 다르게 두어 계산량을 조절한다. CFBI+에서는 embedding channel이 stride 4, 8, 16에서 각각 32, 64, 128이다.

그런데 해상도가 높아질수록 matching 계산량은 매우 빠르게 증가한다. 이를 줄이기 위해 Atrous Matching(AM)을 제안한다. 아이디어는 모든 reference pixel을 다 쓰지 않고, 간격 $l$을 두고 샘플링한 pixel subset만 matching에 사용하는 것이다. global matching의 atrous form은 다음과 같다.

$$
G_o^l(p) = \min_{q \in P_{1,o}^l} D(p, q)
$$

여기서 $P_{1,o}^l$는 $l$ 간격으로 샘플링된 object pixel 집합이다. local matching도 비슷하게 atrous neighborhood 안에서만 최소 거리를 구한다.

$$
L_o^l(p, k) =
\begin{cases}
\min_{q \in P_{T-1,o}^{l,p,k}} D(p, q) & \text{if } P_{T-1,o}^{l,p,k} \neq \emptyset \\
1 & \text{otherwise}
\end{cases}
$$

논문에 따르면 referred pixel 수가 $l^2$배 줄어들기 때문에, matching 계산 복잡도는 원래의 $1/l^2$ 수준이 된다. 특히 stride 4의 가장 큰 scale에서 2-atrous matching을 사용해 효율을 크게 높였다고 한다. 저자들은 이 방법이 plug-and-play 형태이며, CFBI test stage에도 붙일 수 있다고 설명한다.

### 학습 절차와 구현 세부

backbone은 DeepLabv3+ 기반이지만 Xception-65 대신 dilated ResNet-101을 사용한다. ImageNet과 COCO로 pre-train된 backbone을 사용하고, backbone의 BN 파라미터는 training 중 freeze한다.

학습에서 두 가지 추가 전략이 중요하다. 첫째는 balanced random-crop이다. VOS 데이터는 background pixel 비율이 매우 높기 때문에, 일반 random crop을 쓰면 foreground가 거의 없는 crop이 자주 생긴다. 이를 막기 위해 첫 프레임 crop에 충분한 foreground pixel이 포함될 때까지 crop을 반복한다. 목적은 모델이 background에 편향되는 것을 완화하는 것이다.

둘째는 sequential training이다. 한 번의 SGD iteration에서 연속된 프레임 시퀀스를 사용하고, 첫 step에서는 previous mask로 ground truth를 쓰지만 이후 step에서는 이전 예측 결과를 다음 입력의 previous mask로 사용한다. 이는 training과 inference의 불일치를 줄이기 위한 것이다. FEELVOS는 training에서 항상 GT previous mask를 사용했지만, 논문은 prediction-generated previous mask를 training에도 사용하는 것이 실제 inference와 더 잘 맞고 성능도 더 좋다고 주장한다.

loss는 bootstrapped cross-entropy를 사용하며, hardest 15% pixel만 고려한다. data augmentation은 flipping, scaling, balanced random-crop이다. training resolution은 480p로 맞추고, test 때도 최대 $1.3 \times 480p$로 resize한다. 논문에는 learning rate, step 수, batch size, GPU 수 등도 자세히 제시되어 있지만, 이는 성능 재현에 필요한 구현 정보이지 핵심 알고리즘 자체는 아니다.

## 4. 실험 및 결과

실험은 YouTube-VOS, DAVIS 2016, DAVIS 2017에서 수행되었다. 평가 지표는 $J$ score와 $F$ score, 그리고 그 평균인 $J \& F$다. $J$는 prediction mask와 GT mask의 IoU 평균이고, $F$는 boundary similarity 평균이다. 즉 $J$는 영역 정확도, $F$는 경계 품질을 본다.

### YouTube-VOS

YouTube-VOS는 대규모 multi-object VOS 데이터셋이며 unseen category가 포함되어 generalization 평가에 특히 중요하다. 논문에 따르면 validation 2018 split에서 CFBI+는 추가적인 fine-tuning, simulated data pre-training, post-processing 없이도 $J \& F = 82.0\%$를 달성했다. 더 강한 training schedule을 적용한 `CFBI+ 2×`는 $82.8\%$까지 올라간다. multi-scale + flip evaluation을 쓰면 $83.3\%$까지 향상된다.

논문은 이 결과가 당시 state-of-the-art를 능가한다고 주장한다. 특히 KMNVOS가 simulated data를 사용하고도 $81.4\%$ 수준인데, CFBI+는 simulated data 없이 이를 넘는다. 또한 STMVOS는 simulated data가 없으면 성능이 크게 떨어진다고 비교한다. 저자들은 이를 통해 CFBI+의 일반화 능력과 학습 효율성을 강조한다.

Testing 2019 split에서도 `CFBI+ 2×`는 $82.9\%$를 기록하며, challenge 상위권 결과보다 높았다고 보고한다. 흥미로운 점은 개선이 seen category보다 unseen category에서 더 크게 나타났다는 것이다. 논문은 이를 background-aware embedding과 multi-scale matching이 generalization에 실제로 도움이 된다는 근거로 해석한다.

### DAVIS 2017

DAVIS 2017은 multi-object segmentation benchmark다. validation split에서 `CFBI+ (Y)`는 $82.9\%$를 기록해 KMNVOS와 EGMN을 근소하게 앞선다. 여기서 `(Y)`는 YouTube-VOS를 추가 학습 데이터로 사용했음을 뜻한다. multi-scale evaluation을 적용하면 $84.5\%$까지 올라간다.

속도 측면에서도 의미 있는 비교가 있다. CFBI+는 multi-object inference 속도가 약 0.18초/frame 수준으로 보고되며, 논문은 backbone feature를 object별로 따로 계산하지 않고 공유하기 때문에 이전 방법보다 효율적이라고 설명한다. testing split에서도 480p 또는 600p 설정에서 강한 결과를 보인다. 특히 600p에서 `CFBI+`는 $78.0\%$를 달성한다.

### DAVIS 2016

DAVIS 2016은 single-object benchmark다. `CFBI+ (Y)`는 validation에서 $89.9\%$를 기록한다. 이는 simulated data를 사용하는 KMNVOS의 $90.5\%$보다 약간 낮지만, FEELVOS의 $81.7\%$보다는 크게 높다. 저자들은 DAVIS 데이터가 작기 때문에 simulated data가 overfitting 완화에 유리할 수 있다고 인정한다. 즉 이 설정에서는 완전한 승리가 아니라, 추가 데이터 사용 여부에 따라 비교 구도가 다르다는 점을 논문도 사실상 보여준다.

### Ablation Study

이 논문의 강점 중 하나는 ablation이 비교적 체계적이라는 점이다.

background embedding ablation에서는 foreground/background integration의 효과가 분명하게 드러난다. 전체 CFBI는 DAVIS 2017 validation에서 $74.9\%$인데, background mechanism을 모두 제거하면 $70.9\%$로 크게 떨어진다. pixel-level matching에서 background를 제거하면 $73.0\%$, instance-level attention에서 background를 제거하면 $72.3\%$다. 또한 foreground/background bias $b_F, b_B$를 제거하면 $72.8\%$다. 이 결과는 background를 단순 보조 정보가 아니라 핵심 구성요소로 사용해야 한다는 논문의 주장을 직접 뒷받침한다.

Atrous Matching ablation에서는 atrous factor $l=2$가 가장 실용적인 trade-off로 제시된다. global matching에서 $l=1$일 때 성능은 $81.4\%$, 속도는 0.29초/frame이고, $l=2$일 때 성능은 거의 유지된 채 $81.3\%$, 속도는 0.15초/frame으로 대폭 빨라진다. 반면 $l$을 더 크게 하면 성능 하락이 커진다. 즉 AM은 특히 global matching의 효율 개선에 효과적이라고 볼 수 있다.

multi-scale matching ablation에서도 CFBI+ 설계의 이유가 잘 드러난다. stride 4 matching은 가장 정확하지만 느리고, stride 16 matching은 빠르지만 성능이 크게 떨어진다. 세 scale을 함께 쓰고 atrous matching을 조합한 CFBI+는 0.25초/frame의 속도로 $82.0\%$를 달성해, 단일 scale 대비 더 좋은 accuracy-efficiency 균형을 보인다.

기타 ablation에서도 multi-local windows, sequential training, collaborative ensembler, balanced random-crop, instance-level attention이 모두 성능 향상에 기여한다. 예를 들어 sequential training을 제거하면 $74.9\% \rightarrow 73.3\%$, balanced random-crop을 제거하면 $74.9\% \rightarrow 72.8\%$, instance-level attention을 제거하면 $74.9\% \rightarrow 72.7\%$로 성능이 감소한다.

### 정성적 결과

정성적 비교에서는 CFBI+가 CFBI보다 비슷한 사람들 사이 경계를 더 정확하게 나누고, 작은 object도 놓치지 않는 사례가 제시된다. 또한 많은 비슷한 대상이 있는 장면, occlusion이 심한 장면, 작은 object가 있는 장면에서 비교적 안정적으로 동작한다고 보고한다. 다만 강한 halo blur 때문에 motorbike 전체를 제대로 분할하지 못하는 failure case도 함께 제시한다. 이는 논문이 한계를 완전히 숨기지 않았다는 점에서 긍정적이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 잘 맞물린다는 점이다. “foreground만 보지 말고 background도 함께 보자”는 문제 인식이 단순한 직관 수준에 머무르지 않고, distance metric, matching map, instance-level guidance, training crop 전략까지 일관되게 이어진다. 그리고 ablation 결과가 실제로 이 설계를 지지한다. 특히 background embedding 제거 시 성능이 크게 떨어지는 점은 논문의 핵심 주장에 매우 직접적인 근거가 된다.

두 번째 강점은 pixel-level과 instance-level을 동시에 사용하는 multi-scale representation이다. pixel-level matching은 세밀한 경계와 texture를 다루는 데 유리하고, instance-level attention은 larger context를 제공해 ambiguity를 줄인다. 이 두 수준을 결합한 설계는 비디오 segmentation에서 자연스럽고, 실제로 작은 object와 큰 object 모두에서 강건성을 높이는 방향으로 작동한다.

세 번째 강점은 accuracy와 efficiency를 동시에 고려했다는 점이다. 많은 VOS 연구가 성능 향상에 집중하면서 연산량이 급격히 증가하는 반면, 이 논문은 Atrous Matching을 통해 matching cost를 직접 줄이는 방향까지 제안했다. $l=2$에서 성능 저하가 작고 속도 향상이 큰 결과는 실용성이 높다.

한편 한계도 분명하다. 첫째, 논문은 memory-network 기반 방법처럼 장기 시계열 정보를 명시적으로 저장하는 구조는 아니다. 첫 프레임과 직전 프레임을 주로 사용하므로, 매우 긴 시간 간격이나 severe appearance change 상황에서 얼마나 안정적인지는 실험만으로 완전히 해소되지 않는다. 논문은 sequence training을 사용하지만, 장기 memory 자체를 학습하는 것은 아니다.

둘째, failure case에서 보이듯 강한 blur, halo, 심한 appearance degradation에는 여전히 취약하다. 이는 matching-based 방법이 근본적으로 appearance similarity에 의존하기 때문일 가능성이 크다. 논문은 이 한계를 인정하지만, 이를 해결하는 별도 메커니즘은 제시하지 않는다.

셋째, DAVIS 2016에서는 simulated data를 사용하는 KMNVOS보다 약간 낮은 성능을 보인다. 즉 “simulated data 없이도 항상 최고”라고 일반화하기는 어렵다. 이 논문이 강한 결과를 보여주는 것은 맞지만, 데이터가 적은 setting에서는 외부 시뮬레이션 데이터의 효과가 여전히 크다는 점도 함께 읽어야 한다.

넷째, 수식과 구조는 비교적 명확하지만, 일부 설계 선택은 경험적이다. 예를 들어 왜 특정 window set을 골랐는지, 왜 FC layer 하나가 두 개보다 좋은지, 왜 hardest 15% bootstrapped loss가 최적인지에 대한 이론적 설명은 제한적이다. 이는 실험 논문으로서는 자연스럽지만, 원리적 이해 측면에서는 추가 분석 여지가 있다.

마지막으로, 논문에 명시되지 않은 범위를 넘어서 일반화하면 안 된다. 예를 들어 CFBI+가 tracking이나 interactive editing에도 도움이 될 수 있다고 conclusion에서 언급하지만, 본문 실험은 어디까지나 semi-supervised VOS benchmark에 집중되어 있다. 따라서 관련 분야로의 확장 가능성은 저자들의 전망이지, 이 논문이 직접 검증한 사실은 아니다.

## 6. 결론

이 논문은 semi-supervised video object segmentation에서 foreground만이 아니라 background도 동등하게 중요하다는 관점을 전면에 내세우고, 이를 실제 네트워크 설계로 구현한 CFBI와 CFBI+를 제안한다. 핵심 기여는 foreground-background collaborative embedding, pixel-level matching과 instance-level attention의 결합, 그리고 CFBI+에서의 multi-scale matching과 Atrous Matching이다. 이 조합을 통해 저자들은 별도의 fine-tuning, simulated data pre-training, post-processing 없이도 YouTube-VOS와 DAVIS 계열 benchmark에서 매우 강한 성능을 보고한다.

실제 의미는 분명하다. 이 연구는 VOS에서 background modeling이 단순 부가 요소가 아니라 성능을 크게 좌우하는 핵심 요소일 수 있음을 보여준다. 또한 고해상도 dense matching의 정확도와 계산 효율 사이의 trade-off를 multi-scale + atrous 방식으로 다루는 아이디어는 이후의 segmentation, tracking, video understanding 연구에도 확장 가능성이 있다. 논문 자체도 CFBI/CFBI+를 solid baseline으로 제안하고 있으며, 제공된 실험 결과를 보면 실제로 당시 embedding-based VOS 방법론을 한 단계 정리하고 확장한 작업으로 평가할 수 있다.
