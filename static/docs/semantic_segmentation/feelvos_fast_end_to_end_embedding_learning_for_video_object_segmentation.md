# FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation

- **저자**: Paul Voigtlaender, Yuning Chai, Florian Schroff, Hartwig Adam, Bastian Leibe, Liang-Chieh Chen
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1902.09513

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation(VOS)를 더 실용적으로 만들기 위한 방법인 FEELVOS를 제안한다. 문제 설정은 첫 번째 프레임에서 하나 이상의 객체 마스크가 주어졌을 때, 이후 모든 프레임에서 해당 객체들을 자동으로 분할하는 것이다. 저자들은 기존의 강력한 VOS 방법들이 높은 정확도를 달성하더라도, 실제 사용 측면에서는 여러 한계를 가진다고 지적한다. 특히 첫 프레임에서 별도의 fine-tuning이 필요하거나, 여러 개의 네트워크와 후처리 모듈을 조합해 시스템이 복잡하며, 프레임당 처리 시간이 길다는 점이 핵심 문제다.

이 논문이 다루는 연구 문제는 분명하다. 첫 프레임 annotation만을 이용하면서도, 추가 fine-tuning 없이, 단일 네트워크 기반으로, 빠르고 강한 성능을 내는 multi-object VOS를 어떻게 구현할 것인가이다. 저자들은 이를 위해 네 가지 설계 목표를 명시한다. 첫째 simple, 둘째 fast, 셋째 end-to-end, 넷째 strong이다. 특히 DAVIS 2017 validation에서 $65\%$ 이상의 $J \& F$를 목표로 제시한다.

이 문제가 중요한 이유는 VOS가 video editing, robotics, self-driving cars 같은 실제 응용에 직접 연결되기 때문이다. 정확도만 높고 속도가 느리거나 test-time adaptation이 필요한 방법은 실서비스나 실시간 시스템에서 쓰기 어렵다. 따라서 이 논문은 단순히 성능을 조금 높이는 것이 아니라, “실제로 쓸 수 있는 VOS”라는 관점에서 방법을 재설계한 점이 중요하다.

## 2. 핵심 아이디어

FEELVOS의 중심 아이디어는 pixel-wise embedding을 직접 최종 분류 결과로 쓰지 않고, segmentation network 내부에 주는 “soft guidance”로 사용하는 것이다. 기존 PML이나 VideoMatch 계열 방법은 embedding space에서 nearest neighbor matching을 수행한 뒤, 그 매칭 결과를 segmentation 결정에 더 직접적으로 연결했다. 이 경우 잘못된 match가 곧바로 segmentation error로 이어질 수 있다.

반면 FEELVOS는 embedding 기반 matching을 최종 판단기가 아니라 힌트로 사용한다. 현재 프레임의 각 픽셀에 대해 첫 프레임과의 global matching, 이전 프레임과의 local matching으로 distance map을 만들고, 이것을 backbone feature 및 이전 프레임 예측과 함께 segmentation head에 넣는다. 그러면 네트워크는 noisy한 matching 결과를 보정하며 최종 segmentation을 학습할 수 있다. 이 점이 논문의 가장 중요한 차별점이다.

또 다른 핵심은 previous frame과의 matching을 global이 아니라 local하게 제한한 것이다. 첫 프레임과는 객체가 크게 움직였을 수 있으므로 전체 범위에서 match해야 하지만, 바로 이전 프레임과는 일반적으로 움직임이 작기 때문에 local window 안에서만 match하는 것이 더 정확하고 효율적이라는 직관을 사용한다. 논문의 ablation 결과도 이 local previous-frame matching이 매우 중요하다고 보여준다.

마지막으로, variable number of objects를 end-to-end로 다루기 위해 dynamic segmentation head를 도입한 점도 핵심이다. 객체별로 segmentation head를 공유 가중치로 반복 적용해 object-wise logits를 만든 뒤, 이를 stack하고 softmax를 적용한다. 이 구조 덕분에 객체 수가 달라도 하나의 통일된 multi-object segmentation 학습이 가능하다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 먼저 backbone으로 DeepLabv3+를 사용해 현재 프레임에서 stride 4의 feature map을 추출한다. 그 위에 embedding layer를 추가해 각 픽셀마다 semantic embedding vector를 뽑는다. 이후 각 객체마다 두 종류의 distance map을 만든다. 하나는 첫 프레임 ground truth 객체 픽셀과의 global matching distance map이고, 다른 하나는 이전 프레임 예측 객체 픽셀과의 local matching distance map이다. 여기에 이전 프레임의 object probability map과 backbone feature를 더해 dynamic segmentation head로 보내고, 최종적으로 각 픽셀에 대해 객체 posterior를 예측한다.

논문은 픽셀 $p$와 $q$의 embedding distance를 다음과 같이 정의한다.

$$
d(p, q) = 1 - \frac{2}{1 + \exp(\|e_p - e_q\|^2)}
$$

여기서 $e_p$와 $e_q$는 각각 픽셀 $p$, $q$의 embedding vector이다. 이 거리는 항상 $0$에서 $1$ 사이이며, 동일한 embedding이면 $0$, 매우 멀면 $1$에 가까워진다. 즉 작은 값일수록 두 픽셀이 embedding space에서 유사하다는 뜻이다.

첫 프레임으로부터 정보를 가져오는 global matching은 객체 $o$에 대해 현재 프레임 픽셀 $p$가 첫 프레임의 객체 $o$ 픽셀 집합 $P_{1,o}$와 얼마나 가까운지를 본다.

$$
G_{t,o}(p) = \min_{q \in P_{1,o}} d(p, q)
$$

이 값은 “현재 픽셀이 첫 프레임의 객체 $o$와 얼마나 비슷한가”를 나타내는 soft cue이다. 배경도 하나의 객체처럼 동일하게 취급한다. 논문은 이 global distance map이 대체로 유용하지만 noisy하고 false positive를 포함할 수 있다고 직접 설명한다. 그래서 이를 바로 label assignment에 쓰지 않고, 뒤쪽 convolutional head가 refinement하도록 설계했다.

이전 프레임과의 matching은 tracking과 appearance change 대응을 위한 것이다. 이전 프레임 예측에서 객체 $o$에 속한 픽셀 집합을 $P_{t-1,o}$라고 할 때, 단순 global previous-frame matching은 다음처럼 쓸 수 있다.

$$
\hat{G}_{t,o}(p) =
\begin{cases}
\min_{q \in P_{t-1,o}} d(p, q) & \text{if } P_{t-1,o} \neq \emptyset \\
1 & \text{otherwise}
\end{cases}
$$

하지만 실제 FEELVOS는 이것보다 local previous-frame matching을 사용한다. 현재 픽셀 $p$ 주변의 local neighborhood $N(p)$만 허용하고, 그 안에 있는 이전 프레임 객체 픽셀들로만 최근접 거리를 구한다. $P^{p}_{t-1,o} := P_{t-1,o} \cap N(p)$라고 하면,

$$
L_{t,o}(p) =
\begin{cases}
\min_{q \in P^{p}_{t-1,o}} d(p, q) & \text{if } P^{p}_{t-1,o} \neq \emptyset \\
1 & \text{otherwise}
\end{cases}
$$

로 정의된다. 이 방식은 이전 프레임과 현재 프레임 사이의 motion이 작다는 가정을 활용한다. 논문에 따르면 local matching은 global previous-frame matching보다 더 정확하고 계산 효율도 좋다.

dynamic segmentation head는 객체별로 한 번씩 실행되며, 입력은 네 가지다. 첫째 $G_{t,o}(\cdot)$, 둘째 $L_{t,o}(\cdot)$, 셋째 이전 프레임에서 객체 $o$의 posterior probability map, 넷째 공유 backbone feature다. 이 head는 몇 개의 convolution layer로 객체 하나에 대한 1-channel logits map을 만든다. 모든 객체에 대해 얻은 logits를 채널 방향으로 쌓은 뒤 softmax를 적용하고, cross entropy loss로 학습한다. 중요한 점은 객체마다 달라지는 입력 차원이 많지 않음에도, matching map과 previous prediction이 충분히 강한 cue가 되어 좋은 분할이 가능하다는 것이다.

학습 절차는 비교적 단순하다. 각 training step에서 비디오들을 샘플링하고, 각 비디오에서 세 프레임을 고른다. 하나는 reference frame, 나머지 인접한 두 프레임은 previous/current 역할을 한다. loss는 current frame에만 적용한다. 학습 시 previous frame 관련 입력에는 예측값 대신 ground truth를 사용한다. 즉 local matching도 ground truth mask 기반으로 하고, previous frame prediction feature도 정답 객체에는 1, 나머지에는 0으로 둔다. 논문은 RGMP와 달리 synthetic data generation이나 backpropagation through time이 필요 없다는 점을 단순성의 근거로 강조한다.

추론은 프레임별 단일 forward pass로 수행된다. 테스트 비디오의 첫 프레임에서 embedding을 추출한 뒤, 이후 프레임들을 순차적으로 처리한다. 현재 프레임에서 embedding과 backbone feature를 계산하고, 첫 프레임과 global matching, 이전 프레임과 local matching을 수행한 뒤, dynamic segmentation head를 각 객체에 대해 실행한다. 이전 프레임 prediction feature로는 one-hot이 아니라 직전 프레임에서 예측된 soft probability map을 사용한다. 마지막에는 pixel-wise argmax로 segmentation mask를 얻는다.

구현 세부사항도 비교적 명확하다. backbone은 Xception-65 기반 DeepLabv3+이며, embedding dimension은 100이다. dynamic segmentation head는 channel 256의 depthwise separable convolution 4개와 마지막 $1 \times 1$ conv로 구성된다. receptive field를 크게 하기 위해 depthwise conv의 kernel size를 $7 \times 7$로 사용했다. global matching 계산 비용을 줄이기 위해 학습 시 첫 프레임 reference pixel을 객체당 최대 1024개로 subsampling한다. local matching window는 stride-4 feature 상에서 $k=15$다. 손실 함수는 hardest 15% pixel만 쓰는 bootstrapped cross entropy loss이며, DAVIS 2017 train과 YouTube-VOS train을 함께 사용해 학습한다.

## 4. 실험 및 결과

실험은 DAVIS 2016 validation, DAVIS 2017 validation, DAVIS 2017 test-dev, 그리고 YouTube-Objects에서 수행되었다. DAVIS 2016은 single-object, DAVIS 2017은 multi-object 중심이며, YouTube-Objects는 sparse annotation 기반 단일 객체 데이터셋이다. 평가는 DAVIS 표준 지표를 따른다. $J$는 region similarity, 즉 mIoU이고, $F$는 contour accuracy다. 최종 $J \& F$는 둘의 평균이다.

가장 중요한 결과는 DAVIS 2017 validation에서 FEELVOS가 fine-tuning 없는 방법들 중 최고 성능을 달성했다는 점이다. FEELVOS는 $J=69.1$, $F=74.0$, $J \& F=71.5$를 기록했다. 이는 당시 강한 non-finetuning baseline인 RGMP의 $66.7$보다 $4.8$ 포인트 높다. YouTube-VOS 없이 학습한 버전도 $69.1$의 $J \& F$를 얻어 여전히 RGMP보다 높다. 반면 fine-tuning 기반이면서 매우 복잡한 PReMVOS는 $77.8$로 더 높지만, 프레임당 37.6초로 매우 느리다. FEELVOS는 0.51초/frame이라 속도-정확도 균형이 훨씬 좋다.

DAVIS 2017 test-dev에서도 FEELVOS는 $J \& F=57.8$로 RGMP의 $52.9$보다 $4.9$ 포인트 높다. 논문은 이때 RGMP와 FEELVOS의 런타임이 거의 비슷해진다고 설명하는데, FEELVOS는 객체 수 증가에 상대적으로 덜 민감하기 때문이다. 즉 multi-object setting에서 구조적 장점이 드러난다.

DAVIS 2016 validation에서는 FEELVOS가 $J \& F=81.7$을 기록했다. 이는 RGMP의 $81.8$과 거의 비슷한 수준이다. 하지만 RGMP는 simulated data에 크게 의존하고, simulated data 없이 학습하면 $68.8$까지 떨어진다. 따라서 저자들은 FEELVOS가 더 단순한 학습 절차로 comparable한 성능을 낸다고 해석한다.

YouTube-Objects에서는 FEELVOS가 $J=82.1$을 달성한다. 표에 따르면 이는 OSVOS의 $78.3$, OnAVOS의 $80.5$보다 높다. 다만 논문은 이 데이터셋의 평가 프로토콜이 일관되지 않아 일부 기존 결과는 직접 비교가 어려울 수 있다고 명시한다. 따라서 이 결과는 유망하지만, 엄밀한 직접 비교에는 주의가 필요하다.

ablation study는 이 논문의 방법적 기여를 이해하는 데 매우 중요하다. 기본 설정은 first-frame global matching, previous-frame local matching, previous-frame prediction feature를 모두 사용하는 경우이며 $J \& F=69.1$이다. 여기서 previous-frame local matching을 previous-frame global matching으로 바꾸면 $64.2$로 거의 5포인트 하락한다. 이는 local restriction이 단순한 속도 최적화가 아니라 정확도에도 중요함을 뜻한다.

previous-frame matching을 완전히 제거하면 $54.9$로 크게 떨어진다. 즉 이전 프레임과의 embedding matching은 사실상 핵심 구성 요소다. previous-frame prediction feature도 제거하면 $52.6$까지 더 떨어진다. 반대로 previous-frame local matching만 살리고 previous-frame prediction을 제거하면 $63.3$인데, 이는 prediction feature보다 local matching이 시간 정보 전달에 더 강력하다는 것을 보여준다. 마지막으로 first-frame global matching을 제거하면 $56.1$로 성능이 크게 하락한다. 즉 첫 프레임 정보와 이전 프레임 정보는 둘 다 중요하고, 둘의 역할이 상보적이라는 것이 논문의 결론이다.

정성적 결과에서는 큰 motion, 잘린 초기 객체, 복잡한 장면에서도 상당히 안정적인 segmentation을 보여준다. 하지만 비슷하게 생긴 여러 물고기가 동시에 있는 장면에서는 일부 객체를 놓치고, 고양이 등 첫 프레임에서 충분히 보이지 않은 texture 영역은 누락되기도 한다. 흥미로운 점은 논문이 이런 오류 후에도 이후 프레임에서 recover할 수 있는 경우가 있다고 언급한다는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 방법론적 균형이다. FEELVOS는 단일 네트워크, no fine-tuning, single forward pass per frame이라는 실용적 조건을 유지하면서도 strong baseline들을 능가하는 성능을 달성했다. 특히 embedding을 직접 label decision에 쓰지 않고 convolutional refinement의 입력으로 쓰는 설계는 noisy matching 문제를 완화하는 매우 설득력 있는 아이디어다. 또한 previous-frame local matching과 dynamic segmentation head는 각각 temporal propagation과 multi-object handling을 깔끔하게 해결한다.

두 번째 강점은 end-to-end multi-object 학습을 체계적으로 구현했다는 점이다. 많은 기존 방법이 객체별로 전체 네트워크를 반복 실행하거나 heuristic merging에 의존했는데, FEELVOS는 shared backbone과 object-wise lightweight head를 통해 이 문제를 더 일관된 형태로 다룬다. 이는 객체 수가 많아지는 상황에서 계산과 구조 양면에서 장점이 있다.

세 번째 강점은 학습 절차의 단순성이다. synthetic data generation, BPTT, 복잡한 online adaptation 없이도 높은 성능을 낸다. 실험에서도 DAVIS 2017에서 non-finetuning SOTA를 달성했고, 속도-정확도 trade-off가 우수하다. 논문이 제시한 “practical usability”라는 목표와 결과가 잘 맞아떨어진다.

한계도 분명하다. 첫째, FEELVOS는 fine-tuning 기반 최고 성능 방법인 PReMVOS에는 미치지 못한다. 즉 실용성과 최고 정확도 사이의 trade-off를 택한 방법이다. 둘째, 유사한 외형의 여러 객체가 동시에 존재할 때 tracking이 흔들릴 수 있다. 논문이 fish sequence 사례에서 이를 직접 보여준다. 셋째, 첫 프레임에 충분히 관찰되지 않은 appearance는 이후 segmentation에 불리할 수 있다. 이는 first-frame global matching에 크게 의존하는 구조의 자연스러운 약점이다.

또 하나의 주의점은 학습 중 previous-frame 입력에 ground truth를 사용한다는 점이다. 추론 시에는 model prediction을 사용하므로 train-test gap이 존재한다. 논문은 이로 인한 큰 문제를 보고하지 않지만, 더 긴 시퀀스에서 error accumulation이 얼마나 심각한지는 이 논문만으로는 충분히 분석되지 않는다. 또한 embedding 자체에 대한 직접 loss 없이 segmentation loss만으로 학습하는 방식이 일반화 측면에서 항상 최선인지도 본문만으로는 단정할 수 없다.

비판적으로 보면, FEELVOS의 기여는 “완전히 새로운 VOS 패러다임”보다는 기존 아이디어들, 특히 embedding matching과 mask propagation을 실용적인 구조로 재조합하고 정제한 데 있다. 그러나 그 재구성이 매우 잘 설계되어 있고, ablation으로 각 선택의 타당성을 분명히 보였다는 점에서 연구적 가치는 충분하다.

## 6. 결론

FEELVOS는 semi-supervised video object segmentation에서 실용성 중심의 강력한 기준점을 제시한 논문이다. 이 방법의 주요 기여는 첫 프레임과의 global matching, 이전 프레임과의 local matching, 그리고 dynamic segmentation head를 결합해, no fine-tuning 조건에서 빠르고 정확한 multi-object VOS를 end-to-end로 달성한 데 있다. 특히 embedding을 최종 결정기가 아니라 segmentation network의 내부 단서로 사용한 점이 핵심이다.

실험적으로도 FEELVOS는 DAVIS 2017에서 fine-tuning 없는 방법들 중 새로운 state of the art를 달성했고, 속도-정확도 trade-off 측면에서 매우 경쟁력 있다. 최고 정확도 자체는 더 복잡하고 느린 방법이 앞설 수 있지만, 실제 응용 가능성까지 고려하면 FEELVOS의 설계 철학은 매우 중요하다.

향후 연구 관점에서 이 논문은 두 가지 방향에 의미가 있다. 하나는 practical VOS, 즉 빠르고 단순하면서도 충분히 강한 시스템 설계의 기준을 세웠다는 점이다. 다른 하나는 learned matching cue와 convolutional refinement를 결합하는 구조가 이후 video segmentation, tracking, temporal correspondence 문제들에도 확장될 가능성을 보여주었다는 점이다. 전체적으로 FEELVOS는 “현실적으로 쓸 수 있는 VOS”를 향한 중요한 진전으로 평가할 수 있다.
