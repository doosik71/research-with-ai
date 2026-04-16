# Collaborative Video Object Segmentation by Foreground-Background Integration

- **저자**: Zongxin Yang, Yunchao Wei, Yi Yang
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2003.08333

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation(VOS) 문제를 다룬다. 이 문제는 비디오의 첫 프레임에서 주어진 object mask를 바탕으로, 이후 모든 프레임에서 같은 객체를 정확히 분할하는 것이 목표다. 저자들은 기존 방법들이 주로 foreground object의 feature embedding만 학습하고, background는 충분히 활용하지 않는다는 점을 핵심 한계로 지적한다.

논문의 출발점은 단순하다. 어떤 객체를 잘 분할하려면 그 객체를 잘 찾는 것만으로는 충분하지 않고, 무엇이 그 객체가 아닌지도 함께 잘 알아야 한다. 특히 비디오 장면에는 비슷한 모양의 다른 객체가 자주 등장한다. 예를 들어 양 여러 마리, 사람 여러 명, 자동차 여러 대가 동시에 있는 경우 foreground만 기준으로 matching하면 배경이나 다른 객체와 혼동하기 쉽다. 저자들은 이를 background confusion 문제로 설명하며, foreground와 background를 대등하게 다뤄야 더 안정적인 embedding을 학습할 수 있다고 주장한다.

이 문제는 실제 응용 측면에서도 중요하다. VOS는 augmented reality, autonomous driving, video editing, video instance segmentation 같은 다운스트림 문제와 밀접하게 연결된다. 따라서 정확도뿐 아니라 test-time fine-tuning 없이 빠르게 동작하는 구조가 중요하다. 이 논문은 복잡한 simulated data, fine-tuning, post-processing 없이도 높은 성능을 내는 간결한 구조를 제안한다는 점에서 의미가 있다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 foreground와 background를 함께 embedding하고, 이를 pixel-level과 instance-level 두 수준에서 통합하여 segmentation을 수행하는 것이다. 저자들은 이 접근을 CFBI(Collaborative video object segmentation by Foreground-Background Integration)라고 부른다.

중심 직관은 다음과 같다. 좋은 foreground embedding은 단순히 같은 객체 픽셀끼리 가깝게 만드는 것만으로 충분하지 않다. 그 객체의 background와는 충분히 멀어지도록, 즉 contrastive하게 학습되어야 한다. 이 논문은 reference frame의 foreground뿐 아니라 해당 object의 relative background도 함께 matching에 사용함으로써, foreground와 background가 협력적으로 예측을 돕는 구조를 설계했다.

기존 FEELVOS류 방법과 비교했을 때 차별점은 크게 네 가지다. 첫째, background embedding을 명시적으로 도입한다. 둘째, pixel-level matching뿐 아니라 instance-level attention을 함께 사용한다. 셋째, local matching을 하나의 고정 window가 아니라 여러 크기의 window로 수행해 object motion 속도 변화에 더 robust하게 만든다. 넷째, 이렇게 얻은 foreground/background, pixel/instance 정보를 collaborative ensembler가 통합해 최종 예측을 만든다.

즉, 이 논문의 기여는 단순히 matching 공식을 조금 바꾸는 데 있지 않다. foreground-background 통합, multi-scale local matching, instance-level attention, 그리고 이를 받아들이는 segmentation head를 하나의 일관된 framework로 묶었다는 데 있다.

## 3. 상세 방법 설명

전체 파이프라인은 첫 프레임 $t=1$, 직전 프레임 $t=T-1$, 현재 프레임 $t=T$를 입력으로 사용한다. backbone network는 각 프레임에서 pixel-level embedding을 추출한다. 이후 첫 프레임과 이전 프레임의 embedding을 mask를 이용해 foreground와 background로 분리한다. 이 분리된 정보는 두 경로로 사용된다. 하나는 pixel-level matching이고, 다른 하나는 instance-level attention이다. 마지막으로 collaborative ensembler가 이 정보들을 결합해 현재 프레임의 segmentation mask를 예측한다.

### Collaborative Pixel-level Matching

저자들은 먼저 foreground/background를 구분하는 새로운 pixel distance를 정의한다. 현재 프레임의 픽셀 $p$와 과거 프레임의 픽셀 $q$에 대한 embedding을 각각 $e_p$, $e_q$라고 할 때, 거리 함수는 다음과 같다.

$$
D_t(p, q) =
\begin{cases}
1 - \frac{2}{1 + \exp(\|e_p - e_q\|^2 + b_B)} & \text{if } q \in B_t \\
1 - \frac{2}{1 + \exp(\|e_p - e_q\|^2 + b_F)} & \text{if } q \in F_t
\end{cases}
$$

여기서 $B_t$는 background pixel 집합, $F_t$는 foreground pixel 집합이다. $b_B$와 $b_F$는 각각 background bias와 foreground bias로, trainable parameter이다. 핵심은 foreground 거리와 background 거리를 동일하게 취급하지 않고, 네트워크가 둘의 차이를 따로 학습하도록 했다는 점이다.

이 거리 함수를 바탕으로 먼저 global matching을 수행한다. 현재 프레임의 픽셀 $p$와 첫 프레임의 object $o$에 속한 foreground 픽셀들 사이의 최소 거리를 foreground global matching으로 정의한다.

$$
G_{T,o}(p) = \min_{q \in P_{1,o}} D_1(p, q)
$$

반대로 object $o$의 relative background 픽셀들과의 최소 거리를 background global matching으로 정의한다.

$$
\overline{G}_{T,o}(p) = \min_{q \in \overline{P}_{1,o}} D_1(p, q)
$$

논문 본문에서는 같은 기호 $G_{T,o}(p)$가 foreground/background 식에 반복되어 표기되어 있는데, 문맥상 하나는 foreground global matching, 다른 하나는 background global matching을 의미한다. 즉, 표기상 혼동은 있으나 의도는 명확하다.

다음은 local matching이다. 기존 FEELVOS는 이전 프레임과 현재 프레임 사이를 하나의 고정된 local window에서 matching했다. 그러나 object motion은 느릴 수도 있고 빠를 수도 있으므로, 고정 window는 제한적이다. 그래서 이 논문은 여러 크기의 neighborhood $K = \{k_1, k_2, ..., k_n\}$를 사용해 multi-local matching을 수행한다.

foreground multi-local matching은 다음과 같이 정의된다.

$$
ML_{T,o}(p, K) = \{L_{T,o}(p, k_1), L_{T,o}(p, k_2), ..., L_{T,o}(p, k_n)\}
$$

여기서 각 window size $k$에 대해,

$$
L_{T,o}(p, k) =
\begin{cases}
\min_{q \in P^{p,k}_{T-1,o}} D_{T-1}(p, q) & \text{if } P^{p,k}_{T-1,o} \neq \emptyset \\
1 & \text{otherwise}
\end{cases}
$$

이다. $P^{p,k}_{T-1,o}$는 이전 프레임에서 object $o$의 픽셀 중 현재 픽셀 $p$ 주변 $k$ window 내부에 있는 픽셀들이다. background에 대해서도 같은 방식으로 multi-local matching을 정의한다. 이렇게 하면 object가 천천히 움직이는 경우에는 작은 window가, 빠르게 움직이는 경우에는 큰 window가 도움이 된다.

최종적으로 pixel-level matching branch의 출력은 현재 프레임 embedding, 이전 프레임 embedding, 이전 프레임 mask, foreground/background global matching map, foreground/background multi-local matching map을 모두 concatenate한 feature이다. 저자들은 이전 프레임 embedding을 추가로 넣는 것이 성능을 약 $0.5\%$ 개선한다고 보고했다.

### Collaborative Instance-level Attention

pixel-level matching은 세밀한 위치 정합에는 강하지만, large-scale object나 전역적인 문맥을 파악하는 데는 한계가 있다. 이를 보완하기 위해 저자들은 instance-level attention을 도입한다.

첫 프레임과 이전 프레임의 embedding을 각각 foreground와 background로 나눈 뒤, 각 그룹에 대해 channel-wise average pooling을 수행한다. 그러면 총 네 개의 instance-level embedding vector가 생성된다. 즉, 첫 프레임 foreground, 첫 프레임 background, 이전 프레임 foreground, 이전 프레임 background에서 각각 하나씩 vector를 만든다. 이 네 vector를 concatenate하여 collaborative instance-level guidance vector를 만든다.

이 guidance vector는 collaborative ensembler 내부의 각 Res-Block 입력에 대해 channel-wise gate처럼 작동한다. 구조는 SE-Net과 유사하지만, 논문은 두 개의 FC layer 대신 하나의 FC layer와 비선형 활성화를 쓰는 편이 더 좋았다고 설명한다. 이 attention은 각 채널의 중요도를 조절함으로써, pixel-level 정보만으로는 해결하기 어려운 지역적 ambiguity를 줄이는 역할을 한다.

### Collaborative Ensembler

최종 segmentation head 역할을 하는 것이 collaborative ensembler(CE)이다. 이 모듈은 foreground/background와 pixel-level/instance-level 정보를 함께 받아 최종 mask를 예측한다. 구조적으로는 ResNet과 DeepLab 계열 설계에서 영감을 받아 downsample-upsample 구조를 사용한다.

CE는 세 stage의 Res-Block과 ASPP(Atrous Spatial Pyramid Pooling), 그리고 decoder로 구성된다. Stage 1, 2, 3의 Res-Block 수는 각각 2, 3, 3개다. dilated convolution을 사용해 receptive field를 키우며, 각 stage 내 $3 \times 3$ convolution의 dilation rate는 Stage 1에서는 1, 2이고, 다른 stage에서는 1, 2, 4이다. Stage 2와 3의 시작에서 stride 2 downsampling이 들어간다. 이후 ASPP와 decoder를 통해 receptive field를 더 넓히고, low-level backbone feature와 결합해 세밀한 예측을 복원한다.

요약하면, CE는 단순한 segmentation head가 아니라, 여러 수준의 matching과 attention에서 나온 정보를 받아 큰 receptive field에서 통합하는 집계기 역할을 한다.

### 학습 절차와 구현 세부사항

학습에서는 두 가지 실용적 기법이 추가된다. 첫째는 balanced random-crop이다. 비디오 분할 데이터는 background 픽셀이 훨씬 많아서 일반 random crop을 쓰면 foreground가 거의 없는 crop이 자주 나온다. 이는 모델이 background 쪽으로 편향되게 만든다. 저자들은 첫 프레임 crop 안에 foreground 픽셀이 충분히 포함될 때까지 crop을 다시 뽑는 방식으로 이를 완화했다.

둘째는 sequential training이다. FEELVOS는 한 step만 예측하고 이전 mask로 ground-truth를 사용하지만, 실제 추론 시에는 이전 step의 예측 mask가 다음 step 입력으로 들어간다. 이 논문은 학습 때도 연속된 프레임 시퀀스를 사용해, 첫 예측에는 ground-truth previous mask를 쓰고 그 이후에는 모델의 직전 예측을 previous mask로 사용한다. 즉, inference 조건과 학습 조건의 차이를 줄이려는 설계다.

backbone은 DeepLabv3+ 기반 dilated ResNet-101이며, ImageNet과 COCO로 pre-train되었다. pixel embedding은 stride 4의 depth-wise separable convolution으로 추출한다. loss는 hardest 15% pixel만 사용하는 bootstrapped cross-entropy loss다. YouTube-VOS 학습 시 learning rate는 0.01, 100,000 step, batch size는 GPU당 4 videos이고 2개의 Tesla V100을 사용했다. DAVIS에서는 learning rate 0.006, 50,000 step, batch size는 GPU당 3 videos를 사용했다. multi-local matching window는 $K=\{2,4,6,8,10,12\}$다.

## 4. 실험 및 결과

평가는 DAVIS 2016, DAVIS 2017, YouTube-VOS 세 benchmark에서 수행되었다. 지표는 $J$와 $F$, 그리고 그 평균인 $J \& F$이다. $J$는 prediction mask와 ground-truth mask 사이의 average IoU이고, $F$는 boundary similarity measure다. 논문은 official evaluation server 또는 공식 도구를 사용했다고 명시한다.

### YouTube-VOS 결과

YouTube-VOS는 대규모 multi-object VOS benchmark이며 unseen category가 포함되어 generalization 평가에 중요하다. 논문에 따르면 validation 2018 split에서 CFBI는 별도의 fine-tuning, simulated data, ensemble 없이 평균 $81.4\%$를 기록했다. 세부적으로 seen/unseen에 대해 $J$와 $F$를 포함한 모든 metric에서 당시 비교 대상보다 우수했다. 특히 STMVOS의 $79.4\%$보다 2.0% 높다. multi-scale과 flip을 test-time augmentation으로 추가하면 $82.7\%$까지 올라간다.

testing 2019 split에서도 single-model인 CFBI가 평균 $81.5\%$, multi-scale 버전이 $82.2\%$를 기록한다. 논문은 이것이 challenge 상위권 결과들과 비교해 unseen 및 average metric에서 강한 경쟁력을 보인다고 설명한다. 특히 model ensemble 없이 얻은 수치라는 점을 강조한다.

### DAVIS 2016 결과

DAVIS 2016은 single-object benchmark다. YouTube-VOS를 추가 학습 데이터로 사용한 CFBI는 validation set에서 $J \& F = 89.4\%$를 기록했다. 이는 STMVOS의 $89.3\%$보다 소폭 높다. multi-scale 및 flip을 추가하면 표에서는 $90.7\%$로 기재되어 있지만, 본문 서술에서는 $90.1\%$라고 적혀 있다. 이 부분은 논문 내부 표와 본문 사이에 불일치가 있다. 따라서 안전하게 말하면, test-time augmentation이 성능을 추가로 끌어올렸다는 점은 분명하지만 정확한 최종 수치는 논문 텍스트 내에서 일관되지 않다.

또한 FEELVOS와 비교했을 때 setting이 유사함에도 성능은 크게 높고, 추론 속도도 충분히 빠르다. 표에 따르면 FEELVOS는 $81.7\%$, 0.45초/frame이고, CFBI는 $89.4\%$, 0.18초/frame이다. 즉, 정확도는 크게 향상되면서 속도도 경쟁력 있다.

### DAVIS 2017 결과

DAVIS 2017은 multi-object extension으로 더 어렵다. validation split에서 CFBI는 YouTube-VOS 추가 학습과 함께 $81.9\%$를 기록해 STMVOS의 $81.8\%$를 근소하게 앞선다. multi-scale augmentation을 사용하면 $83.3\%$까지 오른다. testing split에서는 CFBI가 $74.8\%$, multi-scale 버전이 $77.5\%$로 보고되며, STMVOS보다 2.6% 높다고 설명한다.

정성적 결과에서는 large motion, occlusion, blur, similar objects가 동시에 있는 어려운 상황에서도 잘 작동한다고 주장한다. 예를 들어 crowded flock에서 여러 양을 추적하거나, occlusion 이후 사람과 개를 잘 분할한 예시를 제시한다. 반면 실패 사례도 제시하는데, 서로 매우 비슷하고 가까운 두 사람의 손을 분리하지 못한 경우가 있다. 이는 appearance similarity와 motion blur가 동시에 작용했기 때문이라고 해석한다.

### Ablation Study

ablation은 DAVIS 2017 validation set에서 수행되었다.

background embedding의 효과는 매우 크다. foreground/background 메커니즘을 모두 제거하면 성능이 $74.9\%$에서 $70.9\%$로 크게 감소한다. pixel-level matching에서 background를 제거하면 $73.0\%$, instance-level attention에서 background를 제거하면 $72.3\%$가 된다. 즉, background 정보는 두 branch 모두에 중요하지만, 특히 pixel-level matching에서 더 민감하게 작용한다. 또한 foreground/background bias $b_F$, $b_B$를 제거하면 $72.8\%$로 감소한다. 이는 foreground 거리와 background 거리를 분리해 학습하는 설계가 실제로 효과적임을 보여준다.

다른 구성요소에 대한 ablation에서도 모든 항목이 의미 있는 기여를 한다. multi-local window 제거 시 $73.8\%$, sequential training 제거 시 $73.3\%$, collaborative ensembler 제거 시 $73.3\%$, balanced random-crop 제거 시 $72.8\%$, instance-level attention 제거 시 $72.7\%$로 떨어진다. baseline(FEELVOS reproduction)은 $68.3\%$다. 따라서 CFBI의 개선은 하나의 요인 때문이 아니라, background 통합, 다중 local matching, instance-level guidance, 학습 전략이 함께 작동한 결과라고 보는 것이 타당하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 명확하고, 제안 방식이 그 문제를 직접 겨냥한다는 점이다. 저자들은 background confusion이라는 실제적인 실패 원인을 제시하고, 이를 foreground-background integration이라는 구조적 설계로 풀었다. 단순히 성능만 높인 것이 아니라 왜 그런 설계가 필요한지를 직관적 예시와 ablation으로 설득한다.

두 번째 강점은 pixel-level과 instance-level 정보를 결합한 점이다. pixel-level matching은 세밀한 detail에 강하고, instance-level attention은 넓은 receptive field의 문맥을 제공한다. 이 둘을 collaborative ensembler가 통합함으로써 작은 구조와 큰 구조를 동시에 다루도록 했다. 논문에 제시된 ablation은 이 조합이 단일 branch보다 낫다는 점을 잘 보여준다.

세 번째 강점은 실용성이다. 논문은 fine-tuning, simulated data, post-processing 없이도 strong baseline을 제시한다. 당시 강력한 경쟁 방법 중 일부는 elaborate training이나 memory mechanism, 추가 데이터 의존성이 강했는데, CFBI는 비교적 단순한 구조로 이를 대체한다. 속도 면에서도 single-object 기준 약 5 FPS 수준이라고 주장한다.

반면 한계도 있다. 첫째, 비슷한 외형을 가진 객체가 매우 가깝게 붙어 있고 motion blur까지 심한 경우 여전히 실패한다. 논문 스스로도 judo 사례에서 한 손을 놓치는 failure case를 제시한다. 즉, background를 도입해도 fine-grained instance separation이 완전히 해결된 것은 아니다.

둘째, 성능 향상의 일부는 training strategy와 augmentation에 의존한다. balanced random-crop과 sequential training은 합리적이지만, 순수한 모델 구조 개선과 별도로 성능에 영향을 준다. 따라서 제안 방법의 효과를 해석할 때 backbone, crop 정책, sequential training의 기여를 함께 봐야 한다.

셋째, 수식 및 결과 서술에서 일부 표기 혼동이 있다. 예를 들어 foreground/background global matching 식의 기호 표기는 문맥상 분리되어야 하지만 본문에서는 동일 기호처럼 보인다. 또한 DAVIS 2016 multi-scale 결과가 표와 본문에서 다르게 적혀 있다. 논문의 핵심 주장을 뒤집는 수준의 문제는 아니지만, 세부 재현이나 엄밀한 인용에서는 주의가 필요하다.

넷째, 저자들은 collaborative integration이 foreground와 background를 contrastive하게 만든다고 설명하지만, 별도의 contrastive loss를 직접 정의한 것은 아니다. 실제로는 distance metric과 matching 구조를 통해 간접적으로 그런 효과를 유도한다. 따라서 “contrastive”라는 표현은 직관적 설명에 가깝고, 엄밀한 대조학습 프레임워크로 읽으면 안 된다.

## 6. 결론

이 논문은 semi-supervised VOS에서 foreground만이 아니라 background도 함께 embedding하고 matching해야 한다는 관점을 전면에 내세운다. 이를 위해 foreground-background global/local matching, instance-level attention, collaborative ensembler, balanced random-crop, sequential training을 포함한 CFBI framework를 제안했다. 결과적으로 DAVIS 2016, DAVIS 2017, YouTube-VOS에서 당시 state-of-the-art 수준의 성능을 달성했다.

이 연구의 핵심 기여는 두 가지로 요약할 수 있다. 하나는 background를 명시적으로 모델링하면 video object segmentation이 더 안정적이고 일반화 잘 되는 문제로 바뀐다는 점을 보여준 것이다. 다른 하나는 pixel-level detail과 instance-level context를 함께 사용하면 object scale 변화와 motion variation에 더 robust해진다는 점을 실험적으로 입증한 것이다.

실제 적용 측면에서도 이 논문은 의미가 있다. 복잡한 fine-tuning이나 거대한 메모리 구조 없이도 강한 성능을 낸다는 점에서, video editing, tracking, interactive segmentation 같은 응용 시스템의 실시간 또는 준실시간 구성에 참고할 가치가 있다. 향후 연구에서는 이 아이디어를 memory-based framework, transformer 기반 matching, 더 강한 temporal modeling과 결합해 확장할 수 있을 것으로 보인다.
