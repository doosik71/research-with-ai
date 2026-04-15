# YouTube-VOS: Sequence-to-Sequence Video Object Segmentation

- **저자**: Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, Thomas Huang
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1809.00461

## 1. 논문 개요

이 논문은 video object segmentation을 위해 두 가지 기여를 함께 제시한다. 첫째는 대규모 데이터셋인 **YouTube-VOS**를 새로 구축한 것이고, 둘째는 이 데이터셋을 활용해 **sequence-to-sequence** 방식으로 장기적인 spatial-temporal 정보를 학습하는 segmentation 모델을 제안한 것이다. 논문이 다루는 문제는, 첫 프레임의 object mask 하나만 주어졌을 때 이후 전체 비디오 프레임에서 같은 객체를 정확히 분할하는 것이다.

저자들은 기존 방법들의 핵심 한계를 두 가지로 본다. 하나는 많은 방법이 사실상 static image segmentation 틀을 비디오에 가져다 쓰고 있다는 점이다. 예를 들어 첫 프레임에서 online fine-tuning을 한 뒤 각 프레임을 거의 독립적으로 처리하거나, 직전 프레임 정보 정도만 약하게 사용하는 경우가 많다. 다른 하나는 temporal consistency를 활용하려는 방법조차 optical flow나 motion segmentation 같은 외부 pretrained model에 의존한다는 점이다. 이런 방식은 비디오 segmentation 자체에 최적화된 end-to-end 학습이 아니므로 근본적으로 suboptimal하다고 본다.

문제의 중요성은 분명하다. video object segmentation은 객체 추적, 비디오 편집, 증강현실 같은 응용에 직접 연결된다. 그런데 현실 비디오는 appearance change, occlusion, camera motion, fast motion이 빈번해서 장기적인 시간 정보를 잘 활용하지 못하면 쉽게 성능이 무너진다. 저자들은 기존 데이터셋 규모가 너무 작아서 이런 장기 시계열 모델을 제대로 학습시키기 어렵다고 보고, 이를 해결하기 위해 대규모 데이터셋을 직접 만든다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **장기적인 spatial-temporal dependency를 직접 학습하려면, 그에 맞는 충분히 큰 비디오 segmentation 데이터셋과 sequence model이 함께 필요하다**는 것이다. 그래서 저자들은 3,252개 YouTube clip, 78개 category, 6,048개 object, 133,886개 annotation으로 구성된 YouTube-VOS를 구축하고, 그 위에서 **ConvLSTM 기반 sequence-to-sequence segmentation model**을 end-to-end로 학습한다.

기존 접근과의 차별점은 목적 함수 수준에서 드러난다. 기존 방법은 대체로 $P(\bar{y}_t \mid x_0, y_0, x_t)$ 또는 $P(\bar{y}_t \mid x_0, y_0, x_t, x_{t-1})$ 같은 형태에 가깝다. 즉, 첫 프레임 정보나 직전 프레임 정보를 중심으로 본다. 반면 이 논문은 전체 과거 프레임을 사용하는
$P(\bar{y}_t \mid x_0, x_1, \dots, x_t, y_0)$
를 목표로 삼는다. 이는 “현재 프레임을 잘 보라”보다 “지금까지의 object history를 기억하라”에 더 가깝다.

또 하나 중요한 차별점은, temporal cue를 외부 optical flow 모델에서 가져오지 않고 **모델 내부 memory state**로 처리한다는 점이다. 초기 프레임과 초기 mask를 바탕으로 객체의 appearance, location, scale 정보를 hidden state에 넣고, 이후 각 프레임을 보며 이 메모리를 갱신한다. 즉, 객체를 단순히 매 프레임 다시 찾는 것이 아니라, 시간에 따라 “기억하면서 추적하고 분할하는” 구조다.

## 3. 상세 방법 설명

전체 구조는 네 부분으로 구성된다: **Initializer**, **Encoder**, **ConvLSTM**, **Decoder**. 논문 Figure 2의 흐름을 글로 풀면 다음과 같다.

먼저 시간 $t=0$에서 첫 프레임 $x_0$와 초기 object mask $y_0$가 주어진다. 이 둘을 channel-wise로 결합해 **Initializer**에 넣는다. Initializer는 이 입력으로부터 ConvLSTM의 초기 memory state $c_0$와 hidden state $h_0$를 만든다. 논문은 이 초기 상태가 객체의 appearance, location, scale을 담도록 설계되었다고 설명한다.

그 다음 각 시점 $t$에서 현재 RGB frame $x_t$는 **Encoder**를 통해 feature map $\tilde{x}_t$로 변환된다. 이 feature map과 직전 시점의 state $(c_{t-1}, h_{t-1})$를 ConvLSTM에 넣으면, 새 state $(c_t, h_t)$가 계산된다. 이 $h_t$는 지금까지 본 프레임들의 누적 정보를 반영한 spatial-temporal representation으로 해석할 수 있다. 마지막으로 **Decoder**가 $h_t$를 full-resolution mask $\hat{y}_t$로 복원한다.

논문이 제시한 학습식은 다음과 같다.

$$
c_0, h_0 = Initializer(x_0, y_0)
$$

$$
\tilde{x}_t = Encoder(x_t)
$$

$$
c_t, h_t = ConvLSTM(\tilde{x}_t, c_{t-1}, h_{t-1})
$$

$$
\hat{y}_t = Decoder(h_t)
$$

$$
L = -(y_t \log(\hat{y}_t)) + ((1-y_t)\log(1-\hat{y}_t))
$$

마지막 식은 binary cross-entropy loss를 뜻한다. 표기상 일반적인 BCE의 전체 합 또는 평균 형태가 축약되어 적혀 있으며, 핵심은 예측 mask $\hat{y}_t$와 정답 mask $y_t$ 사이의 픽셀 단위 이진 분류 손실을 사용한다는 점이다.

모델 구조를 조금 더 구체적으로 보면, Initializer와 Encoder는 모두 **VGG-16** backbone을 사용한다. 모든 convolution layer와 첫 fully connected layer를 사용하고, fully connected layer는 $1 \times 1$ convolution으로 바꿔 fully convolutional하게 만든다. Initializer는 그 위에 추가 convolution 두 개를 두어 각각 $c_0$, $h_0$를 생성한다. Encoder는 추가 convolution 한 개를 두어 feature를 만든다. ConvLSTM 내부 convolution은 모두 512개의 $3 \times 3$ filter를 사용한다. gate output에는 sigmoid, state output에는 tanh 대신 **ReLU**를 사용했는데, 논문은 empirically ReLU가 더 좋았다고 보고한다. forget gate bias는 1로 초기화한다.

Decoder는 다섯 개의 upsampling layer를 가지며, filter 수는 512, 256, 128, 64, 64 순이다. 마지막에 sigmoid를 가진 출력층이 binary mask를 만든다. 초기화는 backbone의 VGG-16 부분만 pretrained VGG-16을 쓰고, 나머지는 Xavier initialization을 사용한다.

학습 절차도 중요한데, 매 iteration마다 임의의 training video에서 객체 하나와 길이 $T=5 \sim 11$ 프레임 구간을 뽑는다. 모든 RGB frame과 annotation은 $256 \times 448$로 resize한다. 초기 학습 단계에서는 annotation이 있는 프레임만 사용하여 모든 시점에 대해 loss를 계산한다. 학습이 안정되면 annotation이 없는 중간 프레임도 넣되, 그런 프레임은 loss를 0으로 둔다. 이는 YouTube-VOS가 30fps 원본에서 **5프레임마다 annotation**, 즉 6fps sampling rate로 주석화되었기 때문이다. optimizer는 Adam, 초기 learning rate는 $10^{-5}$, 80 epoch에서 수렴했다고 적혀 있다.

추론 단계에서 이 모델의 장점은 **offline-trained model만으로도 unseen category에 바로 적용 가능**하다는 것이다. 기존 강한 baseline들은 test video마다 수백 iteration online fine-tuning을 하는 경우가 많지만, 이 모델은 online learning 없이도 경쟁력 있는 결과를 보인다. 다만 저자들은 성능을 더 높이기 위해 online learning도 추가 실험한다.

online learning은 test 시점에 $(x_0, y_0)$에서 affine transformation을 적용해 synthetic pair $\{(x_0, y_0), (x_1, y_1)\}$를 만든 뒤, 이를 이용해 **Initializer, Encoder, Decoder**만 fine-tuning하는 방식이다. **ConvLSTM parameter는 고정**한다. 저자들의 논리는 ConvLSTM은 category-specific appearance가 아니라 장기 temporal dependency 자체를 모델링하므로 고정하는 것이 적절하다는 것이다.

## 4. 실험 및 결과

### YouTube-VOS 데이터셋

YouTube-VOS는 논문의 가장 큰 기반 자산이다. 논문 제출 시점 기준으로 3,252개 video clip, 78개 category, 6,048개 object, 133,886개 annotation을 포함한다. clip 길이는 보통 3초에서 6초이며, 동물, 차량, 액세서리, 일반 사물, 사람의 활동을 포함한다. 사람 activity는 단일 “person” category로 뭉치지 않고 tennis, skateboarding, motorcycling, surfing 등으로 나눈다. activity video에서는 사람과 상호작용 객체를 함께 annotation한다.

데이터 구성 방식도 실용적이다. YouTube-8M에서 후보 영상을 가져오고, shot detection으로 clip을 자른 뒤, 지나치게 어둡거나 흔들리거나 blurry하거나 scene transition이 포함된 clip을 제외한다. 각 clip마다 최대 5개 객체를 annotator가 정밀 경계 기반으로 라벨링한다. 기존 데이터셋이 frame-by-frame dense annotation인 것과 달리, 이 데이터셋은 **5프레임 간격 주석**을 택한다. 저자들은 인접 5프레임 사이의 시간 상관성이 충분히 높기 때문에 이 전략이 annotation budget 대비 영상 수와 객체 수를 크게 늘리는 데 유리하다고 주장한다.

데이터 split은 training 2,796개, validation 134개, test 322개다. test는 다시 **seen / unseen**으로 나뉜다. unseen category는 ant, bull riding, butterfly, chameleon, flag, jellyfish, kangaroo, penguin, slopestyle, snail의 10개다. 평가는 DAVIS와 동일하게 region similarity $J$와 contour accuracy $F$를 사용한다.

### YouTube-VOS 비교 실험

비교 대상은 SegFlow, OSMN, MaskTrack, OSVOS, OnAVOS이며, 모두 저자들이 YouTube-VOS training set으로 재학습했다. 결과의 핵심은 다음과 같다.

저자 방법은 **online learning 없이도** seen/unseen에서 각각 $J$ mean 60.9/60.1, $F$ mean 64.2/62.3을 기록한다. 이는 OSVOS의 59.1/58.8보다 낫고, 다른 baseline보다도 전반적으로 우수하다. 더욱 중요하게는 저자 모델은 **장기 temporal modeling만으로** 이 성능을 달성한다.

**online learning을 추가한 버전**은 seen/unseen에서 각각 $J$ mean 66.9/66.8, $F$ mean 74.1/72.3으로 더 상승한다. 논문은 특히 $J$ mean에서 이전 최고 성능 방법인 OSVOS 대비 약 8% absolute improvement라고 강조한다. 또한 contour accuracy와 decay rate에서도 큰 차이로 앞선다.

흥미로운 관찰도 있다. DAVIS에서 강했던 OnAVOS가 YouTube-VOS에서는 기대만큼 강하지 않다. 저자들은 복잡한 motion과 큰 appearance variation 때문에 online adaptation이 자주 실패한다고 해석한다. 시간에 따른 $J$ mean 곡선을 보면, 초반 몇 프레임에서는 online-learning 기반 방법이 강할 수 있지만, 시간이 지날수록 저자 모델의 성능 저하가 더 느리다. 이는 “초기 appearance 기억”보다 “장기 기억 유지”가 더 중요한 구간에서 제안 방식이 이점을 가진다는 해석과 맞아떨어진다.

unseen category 일반화도 중요하다. 대부분 방법은 seen이 unseen보다 약간 높지만, 차이는 크지 않다. 저자 모델은 online learning 없이도 seen/unseen 성능 차가 작다. 이는 spatial-temporal modeling이 unseen category에도 비교적 잘 일반화한다는 근거로 제시된다.

### DAVIS 2016 결과

DAVIS 2016 validation set에서는 먼저 YouTube-VOS pretrained model을 DAVIS train 30 videos로 200 epoch fine-tuning한 뒤 평가한다. 결과는 다음과 같다.

저자 방법의 **online learning 없는 버전**은 mean IoU 76.5%, 속도 0.16초/frame이다. 이는 MaskTrack 79.7%, OSVOS 79.8%보다는 조금 낮지만, 이 둘은 online learning 및 추가 요소를 활용하며 속도는 10초/frame 이상이다. 저자들은 이를 바탕으로 자사 모델이 **60배 정도 빠르면서도 비슷한 수준**이라고 주장한다.

**online learning 추가 버전**은 mean IoU 79.1%, 속도 9초/frame이다. 즉, 정확도는 더 올라가지만 실시간성은 줄어든다. OnAVOS의 85.7%에는 미치지 못하지만, 논문의 핵심 메시지는 DAVIS 단일 benchmark 최고점이 아니라, **large-scale data와 end-to-end temporal model의 유효성**을 입증하는 데 있다.

### 데이터 규모에 대한 분석

논문에서 가장 설득력 있는 부분 중 하나는 dataset scale ablation이다. YouTube-VOS training set의 25%, 50%, 75%, 100%로 각각 학습한 결과, 25%만 쓰면 성능이 크게 떨어진다. 예를 들어 100% 학습 시 $J$ mean이 60.9/60.1인데, 25% 학습 시 46.7/40.1이다. 특히 unseen 성능 하락이 더 크다. 저자들은 이를 근거로 “기존 모든 비디오 segmentation 데이터셋을 합쳐도 규모가 부족하다”고 주장한다.

DAVIS 2016만으로 학습한 별도 실험도 같은 메시지를 준다. 30개 DAVIS train video만으로 처음부터 학습하면 mean IoU가 51.3%, 기존 소형 데이터셋을 더해 총 192개로 늘려도 51.9%에 그친다. DeepLab pretrained encoder를 사용해도 45.6%로 오히려 더 낮다. 논문은 이 결과를 바탕으로 **spatial-temporal feature는 static image representation에서 쉽게 transfer되지 않으며**, 충분한 비디오 데이터가 반드시 필요하다고 결론짓는다.

### 모델 변형 실험

Initializer를 제거하고 object mask 자체를 hidden state로 직접 쓰는 변형은 seen/unseen에서 $J$ mean 45.1/38.6으로 크게 나빠진다. 이는 단순 mask만으로는 localization과 object representation에 필요한 정보가 부족하다는 뜻이다. 즉, 첫 프레임 RGB 정보와 mask를 함께 encoding하는 Initializer가 핵심이라는 점을 보여준다.

또 다른 변형은 이전 시점 segmentation mask를 Encoder 입력에 추가하는 방식이다. 이는 MaskTrack과 비슷한 아이디어이며 error drift 가능성이 있다. 학습 시에는 먼저 teacher forcing으로 이전 정답 mask를 넣고, 이후에는 model prediction으로 바꾸는 curriculum learning을 사용한다. 이 변형의 성능은 seen/unseen에서 $J$ mean 59.4/60.7로 원래 모델과 비슷하다. 즉, 이전 mask를 명시적으로 넣는 것이 절대적으로 필요하지는 않지만, 적절한 학습 전략이 있으면 비슷한 수준으로 동작할 수 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의와 해결 수단이 정합적**이라는 점이다. “장기 temporal modeling이 필요하지만 데이터가 부족하다”는 진단을 하고, 실제로 대규모 데이터셋을 만든 뒤 그 위에서 ConvLSTM 기반 모델을 학습한다. 데이터셋과 알고리즘이 따로 노는 것이 아니라 서로를 뒷받침한다.

둘째, 제안 모델은 video object segmentation을 truly sequential problem으로 다룬다. 첫 프레임과 직전 프레임만 참고하는 수준이 아니라, hidden state를 통해 전체 이력을 반영한다는 점에서 개념적으로 깔끔하다. 실험에서도 긴 구간으로 갈수록 decay가 완만하다는 결과가 이 설계를 지지한다.

셋째, 실용적인 장점도 있다. online learning 없이도 강한 성능을 보이며, DAVIS 기준 0.16초/frame으로 빠르다. 이는 test-time fine-tuning에 크게 의존하던 당시 방법들과 비교해 실질적인 장점이다.

반면 한계도 분명하다. 먼저 논문이 다루는 설정은 **첫 프레임 mask가 주어지는 semi-supervised video object segmentation**이다. 완전 자동 segmentation 문제는 아니다. 따라서 응용 범위는 넓지만 입력 조건이 강하다는 점을 분명히 봐야 한다.

또한 YouTube-VOS는 5프레임 간격 annotation이므로 dense per-frame supervision이 아니다. 저자들은 이것이 충분하다고 주장하지만, 빠른 motion이나 짧은 순간 occlusion 복원에 어떤 영향을 주는지는 이 논문만으로 완전히 평가되었다고 보기 어렵다. 이 부분은 논문이 장점으로 제시하지만, 반대로 세밀한 temporal supervision 부족이라는 관점에서도 볼 수 있다.

모델 측면에서는 ConvLSTM 기반 구조가 장기 정보를 기억한다고 하지만, 정확히 어떤 종류의 temporal dependency를 얼마나 안정적으로 보존하는지는 정성적 설명 중심이다. 예를 들어 긴 시퀀스에서 state saturation이나 memory bottleneck이 어느 정도인지에 대한 분석은 없다. 또한 multi-object interaction이 강한 상황에서 객체별 identity 보존 문제가 얼마나 잘 해결되는지도 본문에서 깊게 분석되지는 않는다.

비판적으로 보면, 논문의 가장 강한 기여는 사실상 **dataset contribution**과 그로 인한 학습 가능성 증명에 있다. 모델 자체는 ConvLSTM encoder-decoder라는 점에서 완전히 새로운 계열이라기보다, 기존 sequence-to-sequence 아이디어를 segmentation에 맞게 적용한 형태다. 물론 이 점이 논문의 가치를 낮추는 것은 아니지만, 알고리즘적 독창성보다 데이터와 문제 설정의 적절성이 더 핵심적이라는 해석이 가능하다.

## 6. 결론

이 논문은 video object segmentation에서 장기 spatial-temporal modeling이 중요하다는 문제의식을 바탕으로, 대규모 데이터셋 **YouTube-VOS**와 이를 활용한 **sequence-to-sequence ConvLSTM segmentation model**을 함께 제안했다. 제안 모델은 첫 프레임과 초기 mask로부터 객체 상태를 초기화하고, 이후 프레임들을 순차적으로 읽으면서 hidden state를 갱신해 segmentation을 수행한다. 손실은 binary cross-entropy로 단순하지만, 핵심은 이 전체 구조를 end-to-end로 학습했다는 점이다.

실험 결과는 두 가지 메시지를 분명히 준다. 첫째, large-scale video segmentation data가 있어야 이런 temporal model이 제대로 학습된다. 둘째, 그렇게 학습된 모델은 기존의 static-image 중심 또는 optical-flow 의존 방법보다 더 안정적으로 object segmentation을 전파할 수 있다. 실제 적용 측면에서는 online learning 없이도 상당한 성능과 빠른 추론 속도를 보여 practical value가 있다. 향후 연구 측면에서는 이 논문이 이후의 대규모 video understanding, memory-based video segmentation, 그리고 dataset-driven temporal modeling 연구의 기반 역할을 할 가능성이 크다.

결국 이 논문의 핵심 기여는 “비디오 segmentation을 정말 비디오답게 풀기 위해 필요한 데이터와 학습 틀을 제시했다”는 데 있다. 논문 본문에 근거하면, 저자들은 단순히 더 좋은 benchmark score를 보고한 것이 아니라, 이 분야의 학습 패러다임을 small-data, image-centric 방식에서 large-scale, sequence-centric 방식으로 이동시키려는 분명한 방향을 제시했다고 볼 수 있다.
