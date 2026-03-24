# Recurrent Neural Networks for Semantic Instance Segmentation

* **저자**: Amaia Salvador, Míriam Bellver, Víctor Campos, Manel Baradad, Ferran Marques, Jordi Torres, Xavier Giro-i-Nieto
* **발표연도**: 2017
* **arXiv**: <https://arxiv.org/abs/1712.00617>

## 1. 논문 개요

이 논문은 semantic instance segmentation을 **순차적(sequence) 예측 문제**로 재정의하고, 하나의 이미지로부터 객체들을 **하나씩 순서대로** binary mask와 class probability의 쌍으로 생성하는 end-to-end recurrent model을 제안한다. 기존의 다수 방법은 object proposal을 먼저 대량으로 만들고, 각 proposal마다 분류와 mask 예측을 수행한 뒤, 마지막에 non-maximum suppression이나 별도의 filtering/post-processing으로 중복 예측을 제거하는 구조를 사용했다. 이 논문은 그러한 우회 과정을 거치지 않고, 처음부터 최종 목표인 “이미지 안의 각 객체를 분리된 mask와 class label로 출력하는 일” 자체를 직접 학습하려는 시도라는 점에서 의미가 크다.

연구 문제는 분명하다. semantic instance segmentation의 출력은 고정 길이가 아니라 이미지마다 객체 수가 달라지는 **variable-length output**이다. 일반적인 feedforward network는 고정된 크기의 출력을 만들기 쉽지만, 객체 수가 장면마다 달라지는 문제를 직접 다루기에는 부자연스럽다. 저자들은 이 점을 해결하기 위해 Recurrent Neural Network, 정확히는 ConvLSTM 기반 decoder를 사용하여 이미지 한 장에서 길이가 가변적인 객체 시퀀스를 생성하도록 설계했다.

문제의 중요성은 두 가지 측면에서 드러난다. 첫째, instance segmentation은 detection보다 더 정밀한 픽셀 수준 이해를 요구하며, 자율주행, 로보틱스, 장면 이해 등 실제 응용에서 핵심적인 과제다. 둘째, 당시 강력한 성능을 내던 방법들이 proposal-based pipeline에 크게 의존하고 있었기 때문에, 진정한 의미의 end-to-end 학습이 가능한 구조를 제시하는 것은 방법론적으로도 중요한 진전이다. 이 논문은 성능 향상만이 아니라, **출력 순서를 포함한 장면 탐색 방식 자체를 학습한다**는 관점까지 제시한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 semantic instance segmentation을 “동시에 모든 픽셀을 한 번에 분류하는 문제”가 아니라, **장면 속 객체들을 차례대로 발견하고 분할하는 문제**로 보는 데 있다. 사람의 시각 탐색이 정적인 장면에서도 순차적으로 이뤄진다는 점에서 영감을 받아, 모델 역시 이미지 전체를 한 번에 끝내려 하기보다 이전까지 찾은 객체 정보를 recurrent hidden state에 축적하면서 다음 객체를 예측하도록 만들었다.

핵심 직관은 다음과 같다. 하나의 이미지 안에는 여러 객체가 있고, 어떤 객체를 먼저 찾느냐에 따라 이후의 분할이 쉬워질 수 있다. proposal-based method는 후보를 많이 뽑은 뒤 고르는 방식이라 중복 예측이 많고 후처리가 필수적이지만, 이 논문의 모델은 스스로 “다음에 무엇을 찾을지”와 “언제 멈출지”를 학습한다. 즉, 출력 자체를 final prediction으로 바로 사용할 수 있도록 설계되었다.

기존 sequential method와의 차별점도 명확하다. 논문에서 비교하는 이전 sequential instance segmentation 방법들은 대체로 **class-agnostic mask**만 생성하거나, segmentation 외에 별도의 semantic labeling module이 필요했다. 반면 이 논문은 각 time step에서 **binary mask, bounding box, class probability, objectness score**를 함께 예측한다. 따라서 객체 분할과 분류를 하나의 recurrent system 안에서 통합적으로 수행한다. 또한 입력 쪽에서도 foreground/background mask나 angle map 같은 사전 가공 정보에 의존하지 않고, RGB image pixel로부터 직접 예측하는 구조라는 점이 강점이다.

## 3. 상세 방법 설명

전체 구조는 **encoder-decoder architecture**이다. encoder는 ResNet-101 backbone을 사용하여 이미지로부터 다중 해상도의 convolutional feature를 추출하고, decoder는 여러 개의 ConvLSTM을 계층적으로 연결한 recurrent upsampling network로 구성된다. semantic segmentation 계열 모델처럼 encoder의 intermediate feature를 decoder에 skip connection으로 전달하지만, 여기서는 decoder가 recurrent하므로 각 time step마다 서로 다른 hidden state를 바탕으로 다른 객체를 예측하게 된다.

입력 이미지를 $x$라 하고, 정답을 객체들의 집합 $y = {y_1, \dots, y_n}$라 하자. 모델은 이를 순서가 있는 예측 시퀀스 $\hat{y} = (\hat{y}*1, \dots, \hat{y}*{\hat{n}})$로 출력한다. 각 time step $t$의 예측은 다음 네 요소로 구성된다.

$$
\hat{y}_t = {\hat{y}^m_t,; \hat{y}^b_t,; \hat{y}^c_t,; \hat{y}^s_t}
$$

여기서 $\hat{y}^m_t$는 binary mask, $\hat{y}^b_t$는 bounding box 좌표, $\hat{y}^c_t$는 class probability vector, $\hat{y}^s_t$는 objectness score이다. $\hat{y}^s_t$는 추론 시점에서 stopping criterion 역할을 한다. 즉, 모델은 객체를 하나 예측할 때마다 “계속 객체가 남아 있는가”까지 함께 판단한다.

### Encoder

encoder는 ImageNet으로 pretrained된 ResNet-101을 사용하며, 마지막 pooling layer와 classification layer를 제거한 뒤 convolutional feature extractor로 쓴다. 이미지 $x \in \mathbb{R}^{h \times w \times 3}$가 들어오면 여러 단계의 feature map이 나온다.

$$
F = encoder(x) = [f_0, f_1, f_2, f_3, f_4]
$$

논문 설명에 따르면 $f_0$는 가장 깊은 block의 출력이고, $f_4$는 입력에 가장 가까운 얕은 block의 출력이다. 즉, 깊은 feature는 semantic abstraction이 강하고, 얕은 feature는 공간적 detail이 더 풍부하다. 이 둘을 함께 쓰기 위해 skip connection이 필요하다.

### Decoder

decoder는 여러 개의 ConvLSTM layer를 위계적으로 쌓은 형태다. 각 ConvLSTM은 현재 time step의 입력과 이전 time step의 hidden state를 함께 사용하므로, 이미 찾은 객체에 대한 정보가 hidden state에 남아 다음 객체 예측에 반영된다.

$i$번째 ConvLSTM의 $t$시점 hidden state는 다음과 같이 계산된다.

$$
h_{i,t} = ConvLSTM_i \left( [B_2(h_{i-1,t}) \mid S_i],; h_{i,t-1} \right)
$$

여기서 $B_2$는 bilinear upsampling by 2이고, $h_{i-1,t}$는 이전 ConvLSTM layer의 출력, $S_i$는 encoder feature $f_i$를 convolution으로 projection한 side output이다. $[\cdot \mid \cdot]$는 concatenation을 뜻한다. 즉, decoder는 이전 recurrent layer의 업샘플된 출력과 encoder의 같은 해상도 feature를 합쳐 다음 출력을 만든다. 이는 semantic segmentation에서 흔히 쓰는 U-Net류 skip connection과 유사하지만, recurrent decoder 내부에 삽입되어 있다는 점이 다르다.

가장 첫 번째 recurrent block은 이전 layer 출력이 없으므로 다음과 같이 쓴다.

$$
h_{0,t} = ConvLSTM_0(S_0, h_{0,t-1})
$$

이 구조의 의미는 분명하다. 깊은 encoder feature로 대략적인 객체 의미를 파악하고, 상위 해상도로 갈수록 얕은 feature를 다시 합쳐 세밀한 경계를 복원한다. 저자들은 첫 두 ConvLSTM layer의 channel dimension을 $D$로 두고, 그 이후 layer는 이전 layer보다 절반씩 줄였다. 모든 ConvLSTM은 $3 \times 3$ kernel을 사용했는데, 이는 [16]에서 사용된 $1 \times 1$ ConvLSTM보다 receptive field가 넓어 멀리 떨어진 instance 관계를 모델링하는 데 더 유리하다고 본다.

최종 mask는 마지막 decoder output에 대해 $1 \times 1$ convolution과 sigmoid를 적용해 입력과 같은 해상도의 binary mask로 얻는다. 반면 bounding box, class, stop prediction은 fully connected layer 세 개로 예측한다. 이때 입력은 모든 ConvLSTM hidden state를 max pooling한 뒤 concat한 벡터 $h_t$이다. 즉, segmentation mask는 공간 정보를 유지하는 convolutional branch에서 만들고, box/class/stop은 전역 요약 표현에서 만든다.

### 학습 목표

학습은 multi-task loss로 이뤄진다. 핵심은 정답 객체 집합에는 고정된 순서가 없기 때문에, 모델의 출력 시퀀스를 정답 객체들과 어떻게 매칭할지를 먼저 정해야 한다는 점이다. 저자들은 이를 위해 **Hungarian matching**을 사용한다. 비용 함수는 mask 간 soft IoU loss이다.

segmentation loss의 기본이 되는 soft IoU는 다음과 같다.

$$
sIoU(\hat{y}, y) = 1 - \frac{\langle \hat{y}, y \rangle}{|\hat{y}|_1 + |y|_1 - \langle \hat{y}, y \rangle}
$$

이 식은 hard threshold를 쓰지 않고 연속값 mask에 대해 IoU에 해당하는 손실을 계산한다는 뜻이다. 예측 mask 시퀀스와 정답 mask 집합 사이의 최적 매칭을 Hungarian algorithm으로 찾고, 매칭 행렬 $\delta$를 이용해 segmentation loss를 계산한다.

$$
L_m(\hat{y}^m, y^m, \delta) = \sum_{t=1}^{\hat{n}} \sum_{t'=1}^{n} sIoU(\hat{y}^m_t, y^m_{t'}) , \delta_{t,t'}
$$

여기서 $\delta_{t,t'} = 1$이면 예측 $t$와 정답 $t'$가 매칭되었다는 뜻이다. 모델 출력 수가 정답 수보다 많을 때, $t > n$인 초과 예측은 gradient를 무시한다고 명시한다.

classification loss $L_c$는 매칭된 예측-정답 쌍에 대해 categorical cross entropy를 적용한다. detection loss $L_b$는 매칭된 bounding box 간 mean squared error다. stop loss $L_s$는 각 시점의 objectness $\hat{y}^s_t$와 “아직 남은 객체가 있는지”를 나타내는 $\mathbf{1}_{t \le n}$ 사이의 binary cross entropy다. 따라서 모델은 초반 step에서는 높은 stop score를, 모든 객체를 찾은 뒤에는 낮은 score를 내도록 학습된다.

총 손실은 다음과 같은 weighted sum이다.

$$
L = L_m + \alpha L_b + \lambda L_c + \gamma L_s
$$

논문은 이 loss term들을 학습 초기에 한 번에 다 켜지 않고, training이 진행되면서 순차적으로 추가했다고 설명한다. 또한 Cityscapes와 CVPPP처럼 객체 수가 많은 데이터셋에서는 curriculum learning을 사용했다. 처음에는 두 개 객체만 예측하게 학습하고, validation loss가 plateau에 도달하면 예측해야 할 객체 수를 하나씩 늘렸다. 이는 긴 시퀀스를 한 번에 학습하기 어려운 recurrent model의 최적화를 돕기 위한 장치로 해석할 수 있다.

### 추론 절차

추론 시 모델은 time step마다 하나의 객체 mask, class, box, stop score를 낸다. stop score가 stopping criterion 역할을 하므로, 객체가 더 이상 없다고 판단하면 시퀀스 생성을 종료한다. proposal-based method와 달리, 별도의 non-maximum suppression이나 mask filtering 과정은 필요하지 않다. 이것이 이 논문의 가장 중요한 설계 철학이다.

## 4. 실험 및 결과

논문은 서로 다른 난이도와 객체 수를 갖는 세 데이터셋에서 모델을 평가한다. 이는 sequential model이 짧은 시퀀스와 긴 시퀀스에서 각각 어떻게 동작하는지 보기 위한 의도로 보인다.

Pascal VOC 2012는 20개 카테고리, 이미지당 평균 2.3개 객체로 상대적으로 객체 수는 적지만 장면 구성과 배치 다양성이 크다. 추가 annotation을 사용해 학습하고 original validation set 1,449장으로 평가했다. CVPPP Plant Leaf Segmentation은 단일 카테고리이지만 이미지당 11~20개, 평균 16.2개의 leaf가 있어 시퀀스가 길다. 학습은 A1 subset 128장, 평가는 test 33장이다. Cityscapes는 8개 카테고리의 street-view 데이터로, 학습 2,975장, validation 500장, test 1,525장이며 학습 이미지당 평균 17.5개 객체, 최대 120개까지 존재해 가장 어렵다.

입력 해상도는 Pascal VOC에서 $256 \times 256$, Cityscapes에서 $256 \times 512$, CVPPP에서 $500 \times 500$이다. 평가지표는 CVPPP에서 SBD와 DiC를, Pascal VOC와 Cityscapes에서는 IoU threshold별 AP를 사용했다.

### Sequential method와의 비교

논문이 가장 직접적으로 강조하는 비교는 기존 sequential method [16], [17]와의 비교다. Table 1에 따르면 Pascal VOC person category에서 제안 방법은 $AP_{50}=60.7$을 기록해 [16]의 46.6, [16]+CRF의 50.1보다 크게 높다. 이는 recurrent formulation 자체뿐 아니라, class prediction까지 포함한 end-to-end 구조와 hierarchical recurrent decoder의 효과를 시사한다.

CVPPP에서는 제안 방법이 [16]보다 확실히 좋다. SBD는 74.7, DiC는 1.1로 보고된다. 다만 [17]이 CVPPP에서는 더 높은 SBD 84.9를 기록한다. 논문은 이 차이를 해석하면서 [17]이 입력 전처리와 다단계 학습, 더 강한 supervision을 사용한다고 설명한다. 즉, 제안 모델은 더 “순수한” end-to-end 설정이지만, 그만큼 특정 데이터셋에서 성능이 밀릴 수 있다는 점을 솔직히 인정한다.

Cityscapes에서는 제안 방법이 [17]과 비슷한 수준이지만, 비순차적 최신 방법들에는 미치지 못한다. Table 1에서 전체 AP는 7.8, $AP_{50}$은 17.0으로 제시되며, car의 경우 $AP_{50}=45.7$로 [17]의 41.9보다 높지만, small and less frequent object에서는 열세다. 예를 들어 bike와 motorbike 같은 작은 객체에서 성능이 낮다. 논문은 이를 global-scale mask prediction의 한계로 해석한다. [17]은 bounding box를 먼저 잡고 지역적으로 segmentation하기 때문에 작은 객체에 강한 반면, 이 논문의 방법은 매 시점에 이미지 전체를 보며 하나의 mask를 생성하므로 작은 물체의 세밀한 localization이 어렵다는 설명이다.

### 비순차적 state-of-the-art와의 비교

Pascal VOC에서 제안 방법은 오래된 proposal-based method들, 예를 들어 SDS나 Chen et al. 같은 초기 접근보다 높은 성능을 보인다. 그러나 R2-IOS, PFN, Arnab et al., MPA 같은 강한 비순차적 모델과 비교하면 낮은 IoU threshold에서는 뒤처진다. 흥미로운 점은 높은 threshold, 예를 들어 $AP_{80}$에서는 37.8로 일부 방법들과 비교적 근접하거나 경쟁적인 면을 보인다는 것이다. 이는 proposal 수를 많이 내는 방식보다 최종 mask 자체를 직접 최적화하는 구조가, 일부 조건에서는 더 정교한 mask를 만들 수 있음을 시사한다.

### Ablation study

Ablation study는 모델 설계의 각 부분이 실제로 중요했는지를 보여준다. encoder를 VGG16에서 ResNet-50, ResNet-101로 바꾸면 성능이 올라간다. 이는 더 강한 feature extractor가 순차 예측의 기초 표현 품질을 결정한다는 당연하지만 중요한 결과다.

skip connection 방식은 concat이 가장 좋았다. R101 + concat + 5 recurrent layers가 $AP_{50}=57.0$, person에서 60.7을 보인 반면, skip을 완전히 제거하면 $AP_{50}=53.8$, person 51.3으로 의미 있는 하락이 있다. 즉, low-level feature가 mask refinement에 필수적이라는 점이 드러난다.

decoder depth를 줄였을 때 성능이 전반적으로 감소하는 것도 중요하다. recurrent layer 수를 5에서 4, 3, 2, 1로 줄이면 성능이 떨어진다. 특히 논문이 강조하는 흥미로운 결과는, skip connection 없는 5-layer decoder와 skip connection 없는 1-layer decoder가 거의 비슷한 수준이라는 점이다. 이는 recurrent depth만 늘려서는 충분하지 않고, encoder side output을 함께 써야 그 깊이가 의미를 가진다는 뜻이다.

### Error analysis

오탐(false positive) 분석에서는 localization error가 가장 큰 비중을 차지한다. 이는 모델이 객체를 완전히 놓치는 것보다, 위치와 경계를 부정확하게 잡는 문제가 더 크다는 뜻이다. 시간 step이 뒤로 갈수록 IoU가 낮아지는 현상도 보고된다. 이는 recurrent model이 긴 시퀀스를 다룰 때 hidden state에 더 많은 정보를 압축해 유지해야 하므로 점차 정보 병목이 심해지는 현상으로 해석된다.

또한 false negative를 크기별로 분석했을 때, Cityscapes에서는 97%, Pascal VOC에서는 38%가 이미지의 1% 미만을 차지하는 작은 객체였다. 평균 IoU 역시 객체가 클수록 높다. 다시 말해 이 모델은 **큰 객체에는 비교적 강하고 작은 객체에는 약하다**. 이는 전역 해상도에서 하나씩 mask를 생성하는 구조의 자연스러운 약점이다.

### Object sorting pattern 분석

이 논문의 독특한 실험은 객체 예측 순서 자체를 분석한 부분이다. 모델은 ground truth 순서를 강제하지 않았고, Hungarian matching으로 가장 잘 맞는 순서를 학습한다. 그럼에도 불구하고 예측 시퀀스는 무작위가 아니라 일관된 sorting pattern을 보인다.

저자들은 right-to-left, bottom-to-top, large-to-small이라는 세 가지 predefined strategy와 모델의 예측 순서를 Kendall tau로 비교했다. Pascal VOC에서는 right-to-left와의 상관이 비교적 높고, Cityscapes에서는 반대 방향인 left-to-right 성향이 드러난다. bottom-to-top, large-to-small과도 일정한 상관이 있다. 이는 모델이 이미지 내용을 바탕으로 나름의 scanpath를 학습했다는 뜻이다.

또한 encoder activation과 예측 순서의 상관을 분석한 결과, 학습 전보다 학습 후 상관이 증가했다. 특히 Pascal VOC와 Cityscapes에서는 마지막 encoder block, CVPPP에서는 뒤에서 두 번째 block과의 상관이 크다. 이는 recurrent decoder로 들어가는 encoder feature 안에 단순한 appearance 정보만이 아니라 “어떤 객체를 어떤 순서로 꺼낼지”에 대한 정보까지 암묵적으로 인코딩된다는 해석을 가능하게 한다. 이 부분은 단순 성능 비교를 넘어, recurrent visual reasoning의 내부 동작을 탐구했다는 점에서 이 논문의 학술적 개성을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic instance segmentation을 truly end-to-end하게 재구성하려 했다는 점이다. proposal-based pipeline이 가진 중복 예측과 후처리 의존성을 제거하고, variable-length output을 sequence modeling으로 자연스럽게 처리했다. 또한 binary mask와 class probability를 동시에 출력하는 recurrent architecture를 제안하여, 기존 class-agnostic sequential method보다 한 단계 더 완전한 형태의 semantic instance segmentation을 다뤘다.

둘째 강점은 모델 분석의 깊이다. 많은 논문이 성능 수치만 제시하고 끝나지만, 이 논문은 prediction order, object sorting pattern, encoder activation correlation까지 분석한다. 이는 모델이 실제로 어떤 방식으로 장면을 “탐색”하는지 이해하려는 시도이며, 단순 engineering 결과를 넘어 연구적 흥미를 높인다.

셋째, architecture design도 설득력이 있다. semantic segmentation에서 입증된 encoder-decoder와 skip connection 구조를 recurrent decoder와 결합해, sequence modeling과 dense prediction을 자연스럽게 이어 붙였다. Hungarian matching을 이용해 순서를 강제하지 않고도 학습할 수 있게 한 점도 잘 설계된 부분이다.

반면 한계도 분명하다. 가장 큰 한계는 작은 객체와 긴 시퀀스 처리다. 논문 자체가 인정하듯이, 객체 수가 많아질수록 뒤쪽 step의 mask 품질이 나빠지고, 작은 객체를 잘 못 잡는다. 이는 recurrent hidden state에 너무 많은 정보를 담아야 하는 구조적 병목과 전역 해상도 기반 mask prediction의 한계 때문이다.

또 다른 한계는 당시 최신 비순차적 방법 대비 절대 성능이 충분히 높지 않다는 점이다. 특히 Cityscapes에서 state-of-the-art와 큰 차이가 난다. 이 점은 “개념적으로 아름답고 end-to-end하다”는 장점이 실제 benchmark 경쟁력으로 바로 이어지지는 않았음을 보여준다.

비판적으로 보면, 이 모델은 proposal-free end-to-end라는 철학은 매우 매력적이지만, 실제로는 detection-style local refinement가 갖는 장점을 완전히 대체하지 못한다. 작은 객체나 복잡한 crowded scene에서는 지역적 고해상도 처리의 이점이 크기 때문이다. 또한 stopping criterion을 objectness score 하나에 의존하는 방식은 sequence error propagation 가능성을 내포한다. 앞에서 잘못된 예측을 하거나 너무 일찍 멈추면 이후 모든 결과에 영향을 줄 수 있다. 다만 논문은 이 점을 실험적으로 완전히 분해해 보여주지는 않는다. 따라서 이 구조의 이론적 매력과 실제 확장성 사이에는 일정한 간극이 있다고 볼 수 있다.

## 6. 결론

이 논문은 semantic instance segmentation을 sequence prediction 문제로 보고, ConvLSTM 기반 recurrent decoder를 사용하여 이미지에서 객체를 하나씩 분할하고 분류하는 end-to-end 모델을 제안했다. 각 step에서 mask, box, class, stop score를 함께 출력하고, Hungarian matching과 multi-task loss를 통해 variable-length output을 학습한다는 점이 핵심 기여다.

실험적으로는 Pascal VOC와 CVPPP에서 기존 sequential method 대비 경쟁력 있는 결과를 보였고, Cityscapes에서는 한계를 드러냈지만 recurrent approach가 실제 urban scene benchmark에도 적용 가능함을 보여줬다. 무엇보다 이 논문은 “객체를 어떤 순서로 찾는가”라는 관점을 모델 해석의 대상으로 끌어들였다는 점에서 흥미롭다.

실제 적용 측면에서는, 이 논문의 구조가 곧바로 최고 성능 시스템으로 이어지지는 않더라도, proposal-free instance segmentation과 sequence-based visual reasoning을 연결하는 중요한 시도라고 평가할 수 있다. 향후 연구에서는 더 높은 입력 해상도, 더 강한 memory capacity, better decoder design, 혹은 transformer류 구조와 결합된 set/sequence prediction 관점으로 발전할 수 있는 출발점 역할을 한다. 즉, 이 논문은 당시 성능 최강 모델이라기보다, **instance segmentation을 end-to-end sequence modeling으로 다루는 방향성을 선명하게 제시한 연구**로 보는 것이 가장 적절하다.
