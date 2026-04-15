# CNN in MRF: Video Object Segmentation via Inference in A CNN-Based Higher-Order Spatio-Temporal MRF

- **저자**: Linchao Bao, Baoyuan Wu, Wei Liu
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1803.09453

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation 문제를 다룬다. 즉, 비디오의 첫 프레임에서 관심 객체의 mask가 주어졌을 때, 이후 모든 프레임에서 해당 객체를 픽셀 단위로 정확히 분할하는 것이 목표다. 여기서 중요한 점은 객체의 semantic class가 미리 알려져 있지 않은 class-agnostic setting이라는 것이다. 따라서 사람, 개, 차량처럼 특정 범주에 한정된 detector에 의존하지 않고, 첫 프레임에 주어진 대상 그 자체를 계속 추적하고 분할해야 한다.

저자들은 기존 방법의 두 계열이 서로 보완적이지만 분리되어 있다고 본다. 하나는 OSVOS 같은 CNN 기반 방법으로, 객체의 appearance와 shape를 잘 활용하지만 시간 축의 일관성을 충분히 모델링하지 못한다. 다른 하나는 MRF 같은 spatio-temporal graph 기반 방법으로, 시간적 연결성과 구조적 제약은 잘 다루지만 표현력이 제한적이다. 이 논문의 핵심 문제의식은 이 둘을 제대로 결합해, 객체의 외형 정보를 잘 이해하면서도 프레임 간 temporal consistency를 principled하게 강제하는 모델을 만드는 것이다.

이 문제는 video editing, summarization, action understanding 등 다양한 응용과 연결되며, 특히 DAVIS 2017처럼 occlusion, distractor, fast motion, object interaction이 심한 데이터셋에서는 단순한 appearance-only 접근만으로 성능이 급격히 떨어진다. 저자들은 이런 어려운 상황에서 공간적 object prior와 시간적 label propagation을 동시에 활용할 수 있는 새 MRF를 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 spatial dependency를 더 이상 단순한 pairwise smoothness로 두지 않고, 한 프레임 전체의 mask 품질을 평가하는 CNN으로 higher-order spatial potential을 정의하는 것이다. 전통적인 MRF는 주로 인접 픽셀 쌍 사이의 pairwise potential을 두어 부드러운 분할을 유도했는데, 이런 방식은 복잡한 객체 형태, 부분 누락 복원, 전체적인 objectness를 표현하기 어렵다. 저자들은 특정 객체용으로 fine-tuning된 mask refinement CNN이 “거친 mask를 더 좋은 mask로 고치고, 이미 좋은 mask는 유지한다”는 성질을 갖는다고 보고, 이를 spatial prior로 사용한다.

즉, 한 프레임의 mask $x_c$가 주어졌을 때 CNN 출력 $g_{\text{CNN}}(x_c)$와 원래 mask의 차이가 작으면 그 mask는 “CNN이 보기에 이미 좋은 mask”라고 해석한다. 반대로 차이가 크면 refinement가 많이 필요하므로 에너지가 높다. 이렇게 하면 한 프레임 안의 모든 픽셀 사이의 복잡한 관계를 CNN이 암묵적으로 모델링하게 된다.

기존 CNN 기반 video object segmentation 방법과의 차별점은 CNN을 단순히 per-frame predictor로 쓰는 것이 아니라, MRF inference 내부에 CNN 기반 spatial prior를 직접 삽입했다는 점이다. 반대로 기존 MRF/CRF와의 차별점은, hand-crafted pairwise 또는 제한적인 higher-order potential 대신 object-specific CNN이 훨씬 풍부한 spatial 구조를 표현한다는 점이다. 이 논문의 기여는 “CNN + MRF를 붙였다”는 수준이 아니라, CNN을 MRF의 higher-order potential과 inference 과정 자체에 통합한 데 있다.

## 3. 상세 방법 설명

전체 모델은 비디오 전체 픽셀 위에 정의된 spatio-temporal MRF다. 각 픽셀 $i$에 대해 이진 라벨 변수 $X_i \in \{0,1\}$를 두고, 전체 labeling $x$의 posterior를 최대화하는 MAP inference를 수행한다. 일반적인 MRF 형식은 다음과 같다.

$$
p(x \mid D) = \frac{1}{Z} \prod_c \psi_c(x_c \mid D)
= \frac{1}{Z} \prod_c \exp\{-E_c(x_c \mid D)\}
$$

따라서 최종 목표는 전체 에너지의 합을 최소화하는 것이다.

$$
x^* = \arg\min_x \sum_c E_c(x_c \mid D)
$$

논문에서 정의한 전체 에너지는 다음과 같다.

$$
E(x) = \sum_{i \in V} E_u(x_i) + \sum_{(i,j)\in N_T} E_t(x_i, x_j) + \sum_{c \in S} E_s(x_c)
$$

여기서 $E_u$는 unary energy, $E_t$는 temporal energy, $E_s$는 spatial energy다.

Unary term은 픽셀별 foreground/background likelihood에 기반한다.

$$
E_u(x_i) = -\theta_u \log p(X_i = x_i)
$$

이 likelihood는 실험에서는 OSVOS의 response map과 motion prior를 결합해 얻는다. 즉, 기본적인 per-pixel foreground confidence는 appearance 기반 segmentation 네트워크가 제공한다.

Temporal dependency는 optical flow로 정의된다. 각 픽셀은 인접 프레임들뿐 아니라 최대 2프레임 떨어진 위치까지 연결될 수 있다. 구체적으로 프레임 $t$에 대해 $\{t \leftrightarrow t-2\}, \{t \leftrightarrow t-1\}, \{t \leftrightarrow t+1\}, \{t \leftrightarrow t+2\}$ 링크를 만들고, forward-backward consistency check로 신뢰할 수 없는 flow는 제거한다. temporal energy는 다음과 같다.

$$
E_t(x_i, x_j) = \theta_t w_{ij} (x_i - x_j)^2
$$

이 식은 optical flow로 대응된 두 픽셀이 같은 라벨을 갖도록 유도한다. $w_{ij}$는 temporal connection의 신뢰도이며, 연결이 믿을 만할수록 라벨 불일치에 더 큰 패널티를 준다.

가장 중요한 spatial energy는 프레임 전체를 하나의 clique로 보고 정의한다. 전통적인 higher-order MRF보다 훨씬 강한 설정이다. 이상적으로는 정답 mask $x_c^*$가 있다면,

$$
f(x_c) = \|x_c - x_c^*\|_2^2
$$

처럼 정의해 좋은 mask에 낮은 에너지를 줄 수 있다. 하지만 실제로는 $x_c^*$를 모르므로, 논문은 refinement CNN $g_{\text{CNN}}(\cdot)$으로 이를 근사한다.

$$
f(x_c) = \|x_c - g_{\text{CNN}}(x_c)\|_2^2
$$

그리고 spatial energy는

$$
E_s(x_c) = \theta_s f(x_c)
$$

로 정의된다. 이 정의의 직관은 분명하다. 좋은 mask라면 refinement CNN을 통과해도 거의 바뀌지 않아야 한다. 반대로 나쁜 mask라면 CNN이 그것을 많이 수정하려 들 것이므로 에너지가 커진다. 결국 CNN은 object-specific appearance와 shape prior를 encode하는 역할을 한다.

문제는 이런 spatial term 때문에 exact inference가 사실상 불가능하다는 점이다. 각 frame clique가 매우 고차원이고, 매 에너지 평가마다 CNN forward pass가 들어가므로 계산량이 커진다. 이를 해결하기 위해 저자들은 auxiliary variable $y$를 도입해 문제를 분리한다.

$$
\hat{E}(x,y) =
\sum_{i \in V} E_u(x_i)
+ \sum_{(i,j)\in N_T} E_t(x_i, x_j)
+ \frac{\beta}{2}\|x-y\|_2^2
+ \sum_{c \in S} E_s(y_c)
$$

여기서 $\beta$는 $x$와 $y$가 서로 가깝게 유지되도록 하는 penalty다. 이후 alternating optimization을 수행한다.

첫 번째 단계는 $y$를 고정하고 $x$를 업데이트하는 temporal fusion step이다.

$$
x^{(k)} \leftarrow \arg\min_x \hat{E}(x, y^{(k-1)})
$$

이 단계에서는 사실상 temporal term과 unary term, 그리고 $y$에 대한 regularization만 고려한다. 저자들은 이를 큰 quadratic integer program으로 정확히 풀지 않고, ICM(Iterated Conditional Modes)으로 근사한다. 각 픽셀을 하나씩 업데이트하면서 나머지를 고정하고, 이 과정을 $L$번 반복한다.

두 번째 단계는 $x$를 고정하고 $y$를 업데이트하는 mask refinement step이다.

$$
y_c^{(k)} \leftarrow \arg\min_{y_c}
\left\{
\frac{\beta}{2}\|x_c^{(k)} - y_c\|_2^2 + E_s(y_c)
\right\}
$$

이 최적화 역시 CNN 기반 항 때문에 비선형이고 non-convex하다. 논문은 이를 직접 풀지 않고, 다음처럼 단순화한다.

$$
y_c^{(k)} \leftarrow g_{\text{CNN}}(x_c^{(k)})
$$

즉 refinement CNN의 forward pass 자체를 spatial inference의 근사 연산으로 본다. 저자들은 이 근사가 DAVIS 2017 validation에서 3000개 이상 프레임 중 99%에서 목적함수를 증가시키지 않았다고 보고한다. 결과적으로 알고리즘은 “temporal fusion으로 이웃 프레임 정보로 mask를 넓히고, mask refinement CNN으로 object-like한 형태로 정제하는” 과정을 반복한다.

CNN $g_{\text{CNN}}$는 MaskTrack과 유사하게 4채널 입력(RGB + coarse mask)을 받고 refined mask를 출력한다. 학습은 두 단계로 이루어진다. 먼저 공개된 segmentation data로 offline training을 수행하고, 테스트 비디오마다 첫 프레임의 ground-truth mask로 online fine-tuning을 수행한다. 이 과정으로 CNN은 특정 객체의 appearance에 적응된다. 따라서 occlusion 후 재등장한 객체도 어느 정도 복원할 수 있다고 주장한다.

## 4. 실험 및 결과

실험은 주로 DAVIS 2017 validation/test-dev, DAVIS 2016, Youtube-Objects, SegTrack v2에서 수행되었다. 평가 지표는 DAVIS에서 주로 region similarity인 IoU 기반 $J$와 contour accuracy인 $F$를 사용한다. 표에서는 Global mean, Region $J$ mean/recall, Contour $F$ mean/recall이 제시된다.

구현 측면에서 초기 likelihood와 초기 mask는 OSVOS를 기반으로 만든다. 다만 OSVOS는 유사한 객체가 많을 때 false positive가 발생하기 쉬우므로, 저자들은 단순 선형 motion model로 타깃의 위치를 예측한 뒤 Gaussian weighting을 적용해 response map을 보정한다. 또한 이전 프레임에서 optical flow로 warp된 response와 현재 response를 pixel-wise max로 합쳐 초기 labeling을 만든다.

Mask refinement CNN은 DeepLab 기반 Caffe 구현을 사용하며, backbone은 4채널 입력으로 수정한 VGG-Net이다. 중간 pooling layer에서 최종 output layer로 skip connection을 추가해 multi-level feature fusion을 한다. 입력은 객체 주변 crop을 $513 \times 513$로 resize한 것이다. Offline training은 DAVIS 2017 training set에서 50K iteration, batch size 10, learning rate $10^{-4}$로 수행했다. 테스트 비디오마다 first-frame GT를 이용해 2K iteration fine-tuning을 진행한다. Optical flow는 주로 FlowNet2를 사용하고, 오류 시 TV-L1 GPU 구현으로 대체한다.

Ablation study가 이 논문의 설계를 이해하는 데 매우 중요하다. baseline은 OSVOS에 linear motion model을 추가한 버전이며, validation set에서 Global mean이 0.596이다. Temporal fusion만 수행하면 오히려 0.589~0.590으로 baseline보다 나빠진다. 이는 temporal propagation만으로는 잘못된 labeling을 이웃 프레임에 퍼뜨릴 수 있음을 보여준다. 반면 mask refinement만 반복하면 0.640에서 0.649까지 올라 약 5%p 개선된다. 그러나 가장 큰 성능 향상은 TF와 MR을 함께 쓸 때 나오며, 3회 반복에서 0.706, 5회 반복에서 0.707까지 올라 baseline 대비 약 11.1%p 향상된다. 이 결과는 temporal fusion과 mask refinement가 상호보완적이라는 논문의 핵심 주장을 강하게 뒷받침한다. TF는 누락된 부분을 neighboring frame으로부터 되살리지만 거칠고 false positive를 만들 수 있고, MR은 그 거친 결과를 object-specific shape로 정제한다.

DAVIS 2017 test-dev에서는 제안 방법이 Global mean 0.675를 기록했다. 이는 당시 challenge 상위 엔트리였던 apata[26]의 0.666, lixx[31]의 0.661보다 높다. 논문이 강조하는 점은 이 결과가 model ensembling, multi-scale testing, dedicated detector 없이 달성되었다는 것이다. 비교 대상으로 제시된 OnAVOS는 0.528, OSVOS는 0.505로, DAVIS 2017의 난이도 상승이 appearance-only 모델에 얼마나 불리한지 보여준다.

다른 데이터셋에서도 성능은 강하다. DAVIS 2016에서는 $J=0.842$로 OnAVOS의 0.857보다는 약간 낮지만 LucidTracker 0.837, OSVOS 0.798보다 높다. Youtube-Objects에서는 0.784로 OSVOS 0.783, OnAVOS 0.774보다 높고, SegTrack v2에서는 0.771로 LucidTracker 0.768, MaskTrack 0.703보다 좋다. 논문은 이를 통해 제안한 접근이 특정 벤치마크에만 맞춘 기법이 아니라 다양한 semi-supervised VOS 환경에서 강력하다고 주장한다.

런타임 측면에서는 inference 자체는 빠르다. Optical flow는 한 번만 계산하면 되고, temporal fusion step은 매우 가볍다. Mask refinement는 CNN forward pass이므로 frame당 fraction of a second 수준이다. 다만 실제 전체 파이프라인의 큰 비용은 video마다 수행되는 online data augmentation과 CNN fine-tuning이다. 저자들에 따르면 data augmentation에 약 1시간, refinement CNN online training에 약 1시간, OSVOS online training에 약 20분이 걸린다. 따라서 추론 알고리즘 자체는 효율적이지만, 전체 시스템은 여전히 online adaptation 비용이 크다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 spatial prior를 CNN으로 정의해 MRF의 표현력을 크게 끌어올렸다는 점이다. 전통적인 pairwise MRF는 국소 smoothness는 줄 수 있지만 object-level shape prior는 거의 담지 못한다. 반면 이 논문은 object-specific refinement CNN을 spatial energy로 사용해, 한 프레임 전체 마스크의 “객체다움”을 훨씬 강하게 모델링한다. 이는 occlusion 이후 재등장, 누락된 파트 복원, 복잡한 형상 유지 같은 어려운 상황에서 유리하다.

두 번째 강점은 temporal cue와 appearance cue의 결합 방식이 heuristic averaging이 아니라 비교적 명확한 에너지 최소화 프레임워크 위에 있다는 점이다. 물론 inference는 근사적이지만, 적어도 설계 의도는 분명하다. Ablation에서도 TF 단독, MR 단독보다 TF+MR이 훨씬 강하다는 결과가 나와 각 구성 요소의 역할이 잘 설명된다.

세 번째 강점은 당시 강한 benchmark 성능이다. 특히 DAVIS 2017 test-dev에서 detector나 ensemble 없이 challenge 상위권 시스템을 넘었다는 점은 의미가 있다. 논문이 주장하는 “CNN representation + MRF temporal propagation”의 조합이 실제 난도 높은 setting에서 유효함을 보여준다.

반면 한계도 분명하다. 첫째, spatial energy 자체가 엄밀한 확률적 potential이라기보다 CNN refinement 연산에 기반한 heuristic한 surrogate다. $E_s(x_c)=\theta_s\|x_c-g_{\text{CNN}}(x_c)\|_2^2$는 직관은 좋지만, 이것이 어떤 명확한 generative 또는 probabilistic model에서 유도된 것은 아니다. 또 step 2 최적화를 CNN forward pass로 대체하는 근사는 실험적으로 잘 동작한다고만 제시되며, 강한 이론적 보장은 없다.

둘째, 성능의 상당 부분이 online fine-tuning에 의존한다. 각 비디오마다 first-frame mask를 이용해 OSVOS와 refinement CNN을 모두 재학습하므로, 실시간성이나 대규모 배치 처리에는 불리하다. 논문도 런타임의 주된 비용이 online training이라고 명시한다. 따라서 실제 응용에서 계산 자원이 제한된 경우 부담이 크다.

셋째, optical flow 품질에 여전히 영향을 받는다. 논문은 forward-backward consistency check와 confidence weighting으로 이를 완화하지만, temporal fusion은 기본적으로 flow 기반 대응이 틀리면 잘못된 정보를 전파할 위험이 있다. Ablation에서 TF 단독이 성능을 떨어뜨린 점이 이를 잘 보여준다.

넷째, 다중 객체 처리 방식은 상대적으로 단순하다. 여러 객체를 각각 독립적으로 처리한 뒤 overlap blob을 에너지 최소 기준으로 할당하는 방식인데, 객체 간 상호작용을 공동으로 모델링한 것은 아니다. 따라서 multi-object competition을 보다 구조적으로 푸는 설계는 아니다.

비판적으로 보면, 이 논문은 “엄밀한 CNN-MRF 이론”보다 “강한 empirical system design”에 더 가깝다. 그러나 그 설계가 임의적이지만은 않고, higher-order spatial prior를 CNN으로 두겠다는 발상이 분명하고 실험적 검증도 충분하다는 점에서 학술적 가치가 있다.

## 6. 결론

이 논문은 video object segmentation을 위해 CNN 기반 higher-order spatial potential과 optical-flow 기반 temporal connection을 함께 갖는 새로운 spatio-temporal MRF를 제안했다. 핵심은 프레임 전체 mask의 품질을 refinement CNN으로 평가하고, inference를 temporal fusion과 mask refinement의 반복 과정으로 근사했다는 점이다. 이로써 객체 appearance/shape prior와 temporal consistency를 동시에 활용할 수 있었다.

실험적으로는 DAVIS 2017 같은 어려운 벤치마크에서 당시 매우 강한 성능을 보였고, 단순한 temporal propagation이나 appearance-only CNN보다 두 정보원의 결합이 더 효과적임을 보여주었다. 특히 detector나 ensemble 없이 높은 성능을 달성했다는 점은 class-agnostic video object segmentation의 일반성 측면에서 의미가 있다.

향후 연구 관점에서 이 논문은 CNN을 단순 predictor가 아니라 structured inference의 일부로 집어넣는 방향을 제시한다. 오늘날의 end-to-end video segmentation, diffusion-based refinement, transformer-based memory model과 직접 같은 계열은 아니지만, “강한 neural prior를 structured optimization에 삽입한다”는 아이디어는 여전히 중요한 주제다. 다만 실제 적용에서는 online adaptation 비용과 optical flow 의존성을 줄이는 방향으로 발전할 필요가 있다.
