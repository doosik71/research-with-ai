# Dual Convolutional LSTM Network for Referring Image Segmentation

- **저자**: Linwei Ye, Zhi Liu, Yang Wang
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2001.11561

## 1. 논문 개요

이 논문은 **referring image segmentation** 문제를 다룬다. 입력으로 이미지와 자연어 표현(referring expression)이 주어졌을 때, 그 문장이 가리키는 대상 객체를 픽셀 단위로 분할하는 것이 목표다. 예를 들어 “man with back to camera”, “guy on left”, “man in yellow shirt”처럼 같은 장면 안에서도 서로 다른 속성, 위치, 관계 표현으로 특정 객체를 지정할 수 있어야 한다.

이 문제는 일반적인 semantic segmentation이나 instance segmentation보다 어렵다. 이유는 미리 정해진 클래스 집합만 다루는 것이 아니라, 자연어 문장이 포함하는 **객체명, 속성, 위치 관계, 맥락 정보**를 함께 이해해야 하기 때문이다. 또한 단순히 bounding box를 찾는 것이 아니라 정확한 segmentation mask를 생성해야 하므로, 언어 이해와 정밀한 공간 정보 보존이 동시에 필요하다.

저자들은 기존 방법의 두 한계를 지적한다. 첫째, 문장 전체를 하나의 벡터로 압축한 뒤 시각 특징과 결합하는 방식은 단어별 세밀한 정보를 잃기 쉽다. 둘째, 단어를 순차적으로 처리하더라도 모든 단어를 동일하게 취급하면, 긴 문장에서 중요한 단어와 덜 중요한 단어를 구분하기 어렵다. 이를 해결하기 위해 저자들은 **dual ConvLSTM 기반 encoder-decoder 구조**를 제안한다.

## 2. 핵심 아이디어

핵심 아이디어는 두 가지다. 첫째, 문장을 단어 단위로 처리하면서도 각 단어의 중요도를 계산하는 **word attention**을 사용해, 중요한 단어가 multimodal interaction에 더 크게 기여하도록 만든다. 둘째, encoder에서 만들어진 멀티레벨 multimodal feature들을 decoder에서 다시 순차적으로 통합하면서 **spatial attention**으로 중요한 공간 영역에 집중해 segmentation mask를 정교하게 복원한다.

이 논문의 차별점은 ConvLSTM을 단순히 시계열 데이터 처리용으로 쓰는 것이 아니라, **이미지의 spatial structure를 유지한 채 언어의 sequential structure를 결합하는 도구**로 재해석했다는 점이다. Encoder의 ConvLSTM(E-ConvLSTM)은 단어 시퀀스를 따라가며 이미지와 언어의 상호작용을 누적하고, decoder의 ConvLSTM(D-ConvLSTM)은 여러 CNN 레벨의 표현을 순차적으로 정제한다. 즉, 하나는 “문장을 따라 객체를 찾는 과정”, 다른 하나는 “여러 해상도의 특징을 따라 mask를 다듬는 과정”을 담당한다.

## 3. 상세 방법 설명

전체 구조는 크게 **multimodal feature encoder**와 **multi-level segment decoder**로 나뉜다.

먼저 언어 표현은 단어 단위로 처리된다. 문장을 $\{w_1, w_2, \dots, w_L\}$라고 할 때, 각 단어는 embedding을 거친 뒤 bidirectional LSTM을 통과하여 문맥을 반영한 단어 표현 $h_l$를 얻는다.

$$
e_l = embedding(w_l)
$$

$$
\overrightarrow{h_l} = \overrightarrow{LSTM}(e_l, \overrightarrow{h_{l-1}})
$$

$$
\overleftarrow{h_l} = \overleftarrow{LSTM}(e_l, \overleftarrow{h_{l+1}})
$$

$$
h_l = [\overrightarrow{h_l}, \overleftarrow{h_l}]
$$

이후 각 단어에 대해 attention weight $a_l$를 계산한다. 이는 두 개의 선형층과 softmax를 통해 얻어진다.

$$
a_l = \frac{\exp(W_{La}\tanh(W_{Lb}h_l))}{\sum_{k=1}^{L}\exp(W_{La}\tanh(W_{Lb}h_k))}
$$

그리고 단어 표현은 이 attention으로 재가중된다.

$$
r_l = a_l h_l
$$

이 단계의 의미는 간단하다. “left”, “yellow”, “keyboard”, “front”처럼 실제 객체 식별에 중요한 단어는 큰 가중치를 받고, 관사나 상대적으로 덜 중요한 단어는 작은 가중치를 받는다.

시각 특징은 DeepLab-101 backbone에서 추출된다. CNN feature map의 각 spatial location에는 시각 정보뿐 아니라 8차원의 **spatial feature**도 추가된다. 이 spatial feature는 좌표와 크기 정보를 정규화하여 담고 있으며, 상대적 위치 표현을 처리하는 데 쓰인다. 결과적으로 시각 특징 맵은 $V \in \mathbb{R}^{W \times H \times (C_v+8)}$ 형태가 된다.

각 단어의 attentive representation $r_l$는 이 feature map의 모든 spatial cell에 tile되어 결합되고, 이렇게 해서 단어별 multimodal feature map $M_l^e$가 만들어진다. 즉, 각 시점은 “현재 단어 + 전체 이미지 공간 구조”의 결합 표현이다.

그 다음 E-ConvLSTM이 이 단어별 multimodal feature들을 순차적으로 읽는다.

$$
(H_l^e, C_l^e) = ConvLSTM_E(M_l^e, H_{l-1}^e, C_{l-1}^e)
$$

이 논문의 중요한 설계는 cell state 업데이트를 수정했다는 점이다. 일반적인 ConvLSTM과 달리, 저자들은 단어 attention $a_l$를 memory update에 직접 넣는다.

$$
C_l^e = a_l \times i_l^e g_l^e + (1-a_l) \times f_l^e C_{l-1}^e
$$

이 식의 의미는 명확하다. 현재 단어가 중요하면 $a_l$가 커지고, 현재 입력에서 들어오는 정보 $i_l^e g_l^e$가 더 강하게 반영된다. 반대로 중요하지 않은 단어이면 과거 기억 $f_l^e C_{l-1}^e$ 쪽에 더 의존한다. 그래서 긴 문장에서도 중요한 단어 중심으로 객체를 점차 localization할 수 있다.

Encoder의 마지막 hidden state $H_L^e$는 해당 CNN 레벨에서의 multimodal representation이 된다. 논문은 이를 하나의 레벨만 쓰지 않고, 여러 CNN 레벨에서 각각 생성한다. 구체적으로 DeepLab-101의 `Res5`, `Res4`, `Res3`에 대해 encoder를 적용해 $\{M_1^d, M_2^d, M_3^d\}$를 얻는다.

Decoder 쪽에서는 먼저 각 레벨 feature에 대해 **spatial attention**을 적용한다.

$$
\tilde{M}_s^d = \sigma(W_s * M_s^d + b_s)\odot M_s^d
$$

여기서 $7 \times 7$ convolution을 사용해 비교적 넓은 receptive field에서 중요한 공간 영역을 강조한다. 저자들의 설명에 따르면 고수준 feature의 spatial attention은 주로 객체의 중심 영역에 집중하고, 저수준 feature의 attention은 더 넓고 세밀한 경계 정보를 반영한다.

그 뒤 D-ConvLSTM이 고수준에서 저수준으로 멀티레벨 feature를 순차적으로 받아 refinement를 수행한다.

$$
(H_s^d, C_s^d) = ConvLSTM_D(M_s^d, H_{s-1}^d, C_{s-1}^d)
$$

최종 hidden state $H_S^d$는 마지막 convolution을 거쳐 픽셀별 foreground probability map $y$를 만든다. 학습은 binary cross entropy loss를 사용한다.

$$
Loss = -\frac{1}{\Omega}\sum_{n=1}^{\Omega}\left(\hat{y}^{(n)}\log y^{(n)} + (1-\hat{y}^{(n)})\log(1-y^{(n)})\right)
$$

즉, 이 모델은 encoder에서 “문장을 따라 객체를 찾고”, decoder에서 “여러 수준의 특징을 따라 mask를 다듬는” 구조라고 이해하면 된다.

## 4. 실험 및 결과

실험은 네 개의 공개 데이터셋에서 수행되었다: **Google-Ref**, **UNC**, **UNC+**, **Referit**.  
Google-Ref는 평균 문장 길이가 8.43 단어로 더 길고 복잡한 표현이 많아 특히 어렵다. UNC와 UNC+는 MS COCO 기반이며, UNC+는 위치 정보 사용이 제한되어 appearance와 context에 더 의존해야 한다. Referit은 object뿐 아니라 “sky”, “ground” 같은 stuff도 포함한다.

평가 지표는 두 가지다. 하나는 평균 **IoU**, 다른 하나는 **Precision@X**이다. Prec@X는 예측 mask의 IoU가 임계값 $X$ 이상인 샘플 비율을 뜻하며, 논문에서는 $0.5$부터 $0.9$까지 사용했다.

구현 측면에서 backbone은 DeepLab-101이며, 입력 이미지는 $320 \times 320$으로 맞추고 문장 길이는 최대 20단어로 제한했다. `Res5`, `Res4`, `Res3`의 spatial resolution은 모두 $40 \times 40$으로 유지된다. 단어 embedding 차원은 1000, bidirectional LSTM의 hidden size는 방향당 500이라 최종 단어 표현 차원 $C_r$는 1000이다. E-ConvLSTM의 cell size는 1000, D-ConvLSTM의 cell size는 500이다. optimizer는 Adam이며 초기 learning rate는 0.00025, weight decay는 0.0005이고, learning rate schedule은 polynomial decay를 사용했다. 추가로 DCRF post-processing도 실험했다.

Ablation 결과는 제안한 각 구성요소가 실제로 성능 향상에 기여함을 보여준다. UNC validation set에서:

- `E-ConvLSTM(w/o word attention)`의 IoU는 **46.70**
- `E-ConvLSTM`은 **50.50**
- `E-ConvLSTM + D-ConvLSTM(w/o spatial attention)`은 **56.27**
- 전체 모델 `E-ConvLSTM + D-ConvLSTM`은 **58.62**
- 여기에 `DCRF`를 더하면 **59.04**

즉, **word attention**, **multi-level decoder**, **spatial attention**, **DCRF**가 모두 누적적으로 성능을 올린다. 특히 encoder-only 구조에서 decoder를 넣었을 때의 향상 폭이 크므로, multi-level refinement의 효과가 상당히 중요하다는 것을 알 수 있다.

정량 비교에서도 전반적으로 당시 state-of-the-art보다 우수한 성능을 보인다. IoU 기준으로 본문 표에 따르면:

- Google-Ref val: **41.32**로 가장 높음
- UNC val/testB: **58.62 / 56.23**
- UNC+ val/testB: **44.18 / 39.43**
- Referit test: **63.75**

DCRF를 적용하면 일부 수치가 더 오른다. 예를 들어 Google-Ref val은 **41.77**, UNC val은 **59.04**, Referit test는 **63.92**가 된다.

특히 Google-Ref에서 향상이 두드러졌다고 저자들은 강조한다. 이 데이터셋은 문장이 더 길고 복잡하므로, 중요한 단어에 더 집중하는 E-ConvLSTM의 장점이 크게 드러난다는 해석이다. 실제로 문장 길이를 구간별로 나눠 비교한 결과에서도, 제안 모델은 모든 길이 구간에서 우수했고 특히 긴 문장에서 이득이 더 컸다.

정성적 결과에서도 제안 모델은 흰 차처럼 내부 대비가 큰 객체, 비슷한 외형의 여러 객체 중 특정 속성과 위치를 가진 객체, 복잡한 관계 표현이 있는 경우 등에서 더 정확한 마스크를 생성했다. 시각화 결과는 E-ConvLSTM이 단어를 하나씩 보면서 점차 관련 영역을 좁혀 가고, D-ConvLSTM이 고수준 feature에서 거칠게 찾은 객체를 저수준 feature로 경계를 다듬는 과정을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **언어의 순차성**과 **이미지의 공간성**을 동시에 보존하면서 상호작용을 설계했다는 점이다. 단순 concat이나 문장 전체 임베딩 기반 접근보다, 어떤 단어가 실제로 중요한지를 명시적으로 반영한다. 또한 attention을 단순 출력 가중치로만 쓰지 않고, ConvLSTM의 **cell memory update 자체에 삽입**했다는 점이 설계적으로 깔끔하고 문제 의식과 잘 맞는다.

또 다른 강점은 decoder 설계다. 많은 segmentation 문제에서 멀티레벨 feature refinement가 중요하지만, 이 논문은 단순 skip connection이나 fusion이 아니라 ConvLSTM으로 레벨 간 순차 정제를 수행한다. 따라서 고수준 semantics와 저수준 boundary detail의 상호 보완을 구조적으로 활용한다. Ablation 결과도 이 설계의 유효성을 뒷받침한다.

실험도 비교적 충실하다. 네 개의 공개 데이터셋에서 평가했고, word attention, spatial attention, decoder, DCRF 각각의 기여를 분리해서 보여주었다. 또한 긴 표현에서의 성능 차이와 attention visualization까지 제시해, 제안 기법이 왜 효과적인지를 단순 숫자 이상으로 설명한다.

한계도 분명하다. 첫째, 모델은 여전히 DeepLab-101 backbone과 ConvLSTM 계열 구조에 강하게 의존하므로 계산량이 적지 않을 가능성이 높다. 그러나 논문은 연산량, 추론 속도, 메모리 사용량을 자세히 보고하지 않는다. 둘째, failure case에서 보이듯 **언어 모호성**, **심한 occlusion**, **미세한 part-level 구분**, **내부 gap이 있는 객체 형상**에는 약하다. 셋째, attention 시각화가 모델 동작을 설명하는 데 도움은 되지만, attention이 반드시 정확한 causal explanation이라고 보장되는 것은 아니다. 다만 이 부분은 논문이 직접 주장한 범위를 넘는 해석이므로, 논문 자체는 주로 성능과 시각화를 근거로 설명하고 있다.

또한 제안 모델은 단어 attention을 scalar로 사용해 각 단어의 전역 중요도를 정한다. 이는 직관적이지만, 단어 중요도가 공간 위치에 따라 달라질 수 있는 상황까지 세밀하게 표현하는지는 논문에서 다루지 않는다. 예를 들어 “left man with yellow hat” 같은 문장에서 어떤 단어의 중요성은 이미지의 특정 위치와 더 강하게 연결될 수 있는데, 본 논문은 이런 finer-grained cross-modal attention보다는 recurrent multimodal memory 설계에 무게를 둔다.

## 6. 결론

이 논문은 referring image segmentation을 위해 **Dual ConvLSTM encoder-decoder** 구조를 제안했다. Encoder의 E-ConvLSTM은 단어 sequence를 따라 이미지와 언어를 결합하면서, **word attention을 cell state에 반영**해 중요한 단어 중심으로 multimodal interaction을 수행한다. Decoder의 D-ConvLSTM은 여러 CNN 레벨의 encoded feature를 **spatial attention**과 함께 순차적으로 정제해 더 정확한 segmentation mask를 생성한다.

실험적으로는 네 개의 공개 데이터셋에서 강한 성능을 보였고, 특히 긴 문장과 복잡한 설명이 많은 Google-Ref에서 유의미한 이득을 확인했다. 이 연구의 의미는 단순히 당시 성능 향상에만 있지 않다. 자연어 기반 객체 localization과 segmentation에서, **언어 순차 처리와 공간 feature refinement를 한 구조 안에서 결합하는 방식**이 효과적이라는 점을 잘 보여준다. 이후 더 강한 vision-language backbone이나 transformer 기반 구조로 확장되더라도, “중요 단어를 중심으로 공간적 mask를 점진적으로 형성한다”는 이 논문의 관점은 여전히 유의미한 설계 원리로 볼 수 있다.
