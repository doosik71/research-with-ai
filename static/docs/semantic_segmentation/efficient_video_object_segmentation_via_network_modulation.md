# Efficient Video Object Segmentation via Network Modulation

- **저자**: Linjie Yang, Yanran Wang, Xuehan Xiong, Jianchao Yang, Aggelos K. Katsaggelos
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1802.01218

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation, 그중에서도 첫 프레임의 object mask 하나만 주어졌을 때 이후 모든 프레임에서 같은 객체를 분할하는 문제를 다룬다. 기존의 대표적 접근법은 먼저 일반적인 segmentation network를 학습한 뒤, 테스트 시점에 첫 프레임의 annotation을 이용해 해당 네트워크를 수백 번 gradient descent로 fine-tuning하여 특정 객체에 적응시키는 방식이었다. 이 계열은 정확도는 높지만, 매 비디오마다 추가 최적화가 필요하므로 매우 느리다.

논문의 핵심 목표는 이 느린 one-shot fine-tuning을 대체하는 것이다. 저자들은 segmentation network 자체를 반복적으로 업데이트하지 않고, 별도의 meta network가 한 번의 forward pass만으로 대상 객체에 맞는 modulation parameter를 생성하여 원래의 segmentation network를 즉시 적응시키는 방법을 제안한다. 논문은 이를 **network modulation**이라고 부른다.

이 문제가 중요한 이유는 video understanding, interactive video editing, augmented reality, video advertisement 같은 실제 응용에서 속도와 정확도가 동시에 중요하기 때문이다. 기존 fine-tuning 기반 방법은 실제 서비스나 실시간 처리에 부담이 크고, optical flow 같은 부가 모듈을 쓰는 경우 계산량이 더 커진다. 따라서 적응 속도를 획기적으로 줄이면서도 성능을 유지하는 것은 실용적으로 매우 큰 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “객체별 segmentation behavior를 네트워크 전체 재학습으로 얻지 말고, intermediate feature를 조절하는 소수의 modulation parameter로 표현하자”는 것이다. 즉, segmentation network는 공통 backbone으로 유지하고, 특정 객체의 appearance와 위치 정보만을 입력으로 받는 별도 modulator가 각 계층의 feature map을 조정하도록 만든다.

저자들은 객체 분할에 필요한 정보를 두 종류로 나눈다. 하나는 객체의 category, color, shape, texture 같은 **visual appearance**이고, 다른 하나는 비디오에서 객체가 연속적으로 움직인다는 점을 반영하는 **spatial prior**이다. 이에 따라 두 개의 modulator를 둔다.

첫째, **visual modulator**는 첫 프레임에서 잘라낸 target object image를 보고 channel-wise scale parameter를 생성한다. 이것은 segmentation network가 “어떤 객체를 봐야 하는가”를 알려주는 역할을 한다. 둘째, **spatial modulator**는 이전 프레임의 예측 mask에서 만든 Gaussian heatmap을 입력으로 받아 spatial bias를 생성한다. 이것은 “어디를 봐야 하는가”를 알려주는 역할을 한다.

기존 접근과의 가장 큰 차이는 적응 방식이다. OSVOS나 MaskTrack 같은 방법은 테스트 시 대상 비디오마다 network weight를 직접 fine-tuning한다. 반면 이 논문은 테스트 시 weight optimization 없이 modulation parameter만 생성해 넣는다. 따라서 adaptation 비용이 매우 작고, 논문은 fine-tuning 기반 방법 대비 약 $70\times$ 빠르면서도 유사한 정확도를 달성했다고 보고한다.

## 3. 상세 방법 설명

전체 시스템은 세 부분으로 구성된다. 하나는 실제 segmentation을 수행하는 **main segmentation network**이고, 나머지 둘은 **visual modulator**와 **spatial modulator**이다. segmentation network는 VGG16 기반 FCN에 hyper-column 구조를 사용한다. modulator들은 이 주 네트워크의 여러 convolution layer 뒤에 삽입된 modulation layer에 들어갈 parameter를 생성한다.

### Conditional Batch Normalization에서 출발한 modulation

저자들은 접근의 출발점을 Conditional Batch Normalization(CBN)으로 설명한다. CBN에서는 외부 입력으로부터 scale과 bias를 만들어 feature map을 조정한다. 논문은 이를 다음과 같이 적는다.

$$
y_c = \gamma_c x_c + \beta_c
$$

여기서 $x_c$와 $y_c$는 $c$번째 채널의 입력 및 출력 feature map이고, $\gamma_c$와 $\beta_c$는 controller network가 생성하는 scale과 bias이다.

논문은 이 아이디어를 일반화해서, 각 convolution layer 뒤에 **modulation layer**를 둔다. 구조는 겉보기에는 위 식과 같지만, 두 modulator가 역할을 나눠 갖는다. visual modulator가 channel-wise scale $\gamma_c$를 만들고, spatial modulator가 spatial bias $\beta_c$를 만든다.

$$
y_c = \gamma_c x_c + \beta_c
$$

여기서 중요한 차이는 $\gamma_c$는 채널마다 하나의 scalar이고, $\beta_c$는 2차원 위치별 값을 가지는 matrix라는 점이다. 즉 visual modulation은 채널 중요도를 조절하고, spatial modulation은 위치 prior를 feature map에 직접 주입한다.

### Visual modulator

visual modulator의 입력은 첫 프레임 annotation으로부터 추출한 target object image, 즉 논문에서 말하는 **visual guide**이다. 이 네트워크는 VGG16을 사용하며, 마지막 classification layer를 modulation parameter 개수에 맞게 바꾼다. 출력은 segmentation network의 여러 modulation layer에 대응하는 channel-wise scale들의 집합이다.

이 모듈의 의도는 segmentation network가 특정 객체 instance에 집중하도록 만드는 것이다. 예를 들어 같은 cat 계열 객체들은 비슷한 modulation parameter를, 완전히 다른 객체는 다른 parameter를 만들어야 한다. 논문은 실제로 이 출력들이 object appearance와 잘 상관되는 embedding을 형성한다고 시각화 결과로 보여준다.

이 설계의 장점은 ImageNet 등 대규모 객체 인식 사전학습 지식을 활용할 수 있다는 점이다. 저자들은 visual modulator와 segmentation network 모두를 ImageNet pretrained VGG16으로 초기화한다.

### Spatial modulator

spatial modulator의 입력은 object의 prior location이다. 저자들은 이전 프레임의 예측 mask에서 중심과 표준편차를 계산한 뒤, 이를 2차원 Gaussian heatmap으로 인코딩한다. 이 heatmap을 **spatial guide**라고 부른다.

spatial modulator는 이 heatmap을 각 계층 feature map 해상도에 맞게 downsample한 뒤, 채널별 scale-and-shift를 적용해 bias map을 만든다. 논문은 이를 다음과 같이 쓴다.

$$
\beta_c = \tilde{\gamma}_c m + \tilde{\beta}_c
$$

여기서 $m$은 해당 계층 해상도의 downsampled Gaussian heatmap이고, $\tilde{\gamma}_c$와 $\tilde{\beta}_c$는 $c$번째 채널에 대한 scale과 shift이다. 구현은 $1 \times 1$ convolution으로 매우 효율적으로 처리한다.

이 설계는 이전 mask를 직접 다음 프레임 입력으로 넣는 MaskTrack와 비슷해 보이지만, 실제로는 훨씬 더 거친 위치 정보만 사용한다. 저자들의 주장은 명확하다. 이전 frame의 정확한 foreground mask를 너무 강하게 쓰면 모델이 mask 자체에 과적합되어 error propagation이 심해질 수 있다. 반면 Gaussian 수준의 거친 위치 prior는 객체 위치와 크기 정도만 주고, 나머지는 RGB 정보로 다시 판단하게 하므로 일반화와 안정성에 유리하다는 것이다.

### 어느 계층을 modulation하는가

직관적으로는 모든 convolution layer 뒤에 modulation을 넣을 수 있지만, 저자들은 early layer에 modulation을 넣으면 성능이 나빠진다고 보고한다. 초기 계층은 low-level feature를 추출하므로 scale-and-shift 변화에 지나치게 민감하다는 해석을 제시한다. 그래서 실제 구현에서는 VGG16의 첫 네 개 convolution layer를 제외한 나머지에만 modulation을 적용했고, 총 9개의 modulation layer를 사용했다.

### 학습 데이터와 학습 절차

이 논문은 static image도 적극적으로 사용한다. 이유는 video segmentation dataset의 object category 수가 적기 때문이다. modulator가 다양한 객체에 대응하는 embedding을 배우려면 더 넓은 category coverage가 필요하므로, 저자들은 MS-COCO를 활용한다. 이미지 크기의 $3\%$보다 큰 객체들만 골라 총 217,516개 객체를 학습에 사용했다고 명시한다.

visual guide는 annotated mask로 객체를 crop한 뒤, 배경 픽셀을 mean image 값으로 채우고, $224 \times 224$로 resize한다. 추가로 최대 $10\%$ scaling과 $10^\circ$ rotation augmentation을 적용한다. spatial guide는 mask의 평균과 표준편차를 계산해 Gaussian으로 만들고, 최대 $20\%$ shift와 $40\%$ scaling augmentation을 준다. FCN 입력 전체 이미지는 한 변 길이를 320, 400, 480 중 하나로 랜덤 선택한 정사각형으로 사용한다.

학습은 balanced cross-entropy loss를 사용한다. optimizer는 Adam이고, mini-batch size는 8이다. 먼저 learning rate $10^{-5}$로 10 epoch, 이어서 $10^{-6}$로 5 epoch 학습한다. 이후 DAVIS 2017 같은 video segmentation dataset으로 추가 fine-tuning할 수 있는데, 이때는 각 frame마다 같은 비디오 안의 foreground object를 visual guide로 랜덤 선택하여 appearance variation에 더 강하게 만든다. 이 단계는 20 epoch, learning rate $10^{-6}$로 학습한다.

## 4. 실험 및 결과

논문은 DAVIS 2016, DAVIS 2017, YoutubeObjects에서 평가한다. 비교 대상은 OFL, BVS 같은 전통적 방법과 PLM, MaskTrack, OSVOS, VPN, SFL, ConvGRU 등 deep learning 기반 방법이다. 저자들은 optical flow나 CRF는 공정 비교를 어렵게 하므로 가능하면 그것들을 제외한 버전을 중심으로 비교했다고 설명한다.

### DAVIS 2016 및 YoutubeObjects

DAVIS 2016과 YoutubeObjects에서는 mean IU를 사용한다. 결과를 보면, fine-tuning이 없는 빠른 방법들 중에서는 제안 방법이 가장 좋은 축에 속한다. 구체적으로 Ours(Stage 1)는 DAVIS 2016에서 72.2, Ours(Stage 1&2)는 74.0을 기록했다. YoutubeObjects에서는 각각 66.4와 69.0이다.

비교해 보면 MaskTrack-B는 DAVIS 63.2, OSVOS-B는 52.5로 제안 방법보다 크게 낮다. fine-tuning을 사용하는 MaskTrack은 69.8, SFL은 74.8, OSVOS는 79.8인데, 제안 방법은 이들 중 일부와 비교해 비슷하거나 더 좋고, 특히 PLM과 MaskTrack보다 우수하다. OSVOS는 더 높은 정확도를 보였지만, 논문은 boundary snapping이 mean IU에 추가로 2.4% 기여한다고 언급한다.

속도 측면이 이 논문의 핵심인데, 제안 방법의 평균 속도는 0.14초/frame 수준으로 보고된다. 같은 표에서 MaskTrack은 12초/frame, OSVOS는 10초/frame, SFL은 7.9초/frame이다. 따라서 저자들이 말한 대로 fine-tuning 기반 강한 baseline 대비 대략 $50\times$에서 $70\times$ 정도 빠르다. visual modulator는 비디오 전체에서 한 번만 계산하면 되고, spatial modulator는 프레임마다 계산하지만 오버헤드는 무시할 수 있을 정도로 작다고 설명한다.

### DAVIS 2017

DAVIS 2017은 다중 객체가 등장하고 유사한 객체도 많아서 더 어렵다. 여기서는 DAVIS 공식 지표인 region similarity $J$와 contour accuracy $F$의 mean, recall, decay를 사용한다. 논문은 OSVOS와 MaskTrack의 single-network, add-on free 버전과 비교했다.

결과는 다음과 같은 해석이 가능하다. OSVOS-B는 $J$ mean 18.5, MaskTrack-B는 35.3으로 매우 낮다. 제안 방법은 fine-tuning 없이 $J$ mean 52.5, $F$ mean 57.1을 기록해 이들보다 큰 폭으로 높다. full OSVOS는 $J$ mean 55.1, MaskTrack은 51.2인데, 제안 방법은 이들과 비슷한 수준이다. 특히 decay 측면에서는 제안 방법이 더 낮다. 예를 들어 $J$ decay는 Ours 21.5, OSVOS 28.2, MaskTrack 28.3이다. 이는 시간이 흐를수록 성능 저하가 더 적다는 뜻이다.

논문은 이 현상을 이렇게 해석한다. OSVOS와 MaskTrack은 첫 프레임에 대한 fine-tuning 덕분에 초반에는 강하지만, 시간이 지나며 객체 자세와 appearance가 바뀌면 그 적응이 오히려 일반화를 방해할 수 있다. 반면 제안 방법은 modulation parameter를 통해 appearance embedding을 학습하므로 pose와 appearance 변화에 더 robust하다는 것이다.

정성적 결과에서도 두 가지 장점을 강조한다. MaskTrack과 비교하면 boundary가 더 정확한 편인데, 이는 spatial prior를 거칠게 주어 모델이 이전 mask 대신 이미지 cue를 더 적극적으로 사용하기 때문이라고 해석한다. OSVOS와 비교하면 여러 비슷한 객체가 있는 장면에서 더 잘 작동하는데, 이는 spatial modulator가 tracking capability를 주기 때문이라고 설명한다. 또한 camel, pigs처럼 MS-COCO에 없는 unseen category에도 잘 작동하는 사례를 제시한다.

### Modulation parameter 시각화

논문은 visual modulator가 실제로 의미 있는 embedding을 학습하는지 확인하기 위해 10개 category, 100개 object instance의 modulation parameter를 2차원 공간에 시각화한다. 그 결과, 같은 category 객체는 대체로 가까이 모이고, 시각적으로 비슷한 category끼리도 서로 인접하게 나타난다. 예를 들어 cat과 dog, car와 bus는 가까우며, bicycle과 dog, bus와 horse는 멀다. 이는 modulator 출력이 appearance 정보를 반영하고 있음을 보여준다.

또한 계층별 modulation 강도도 분석한다. visual modulator의 $\gamma_c$에 대한 표준편차를 보면 deeper layer로 갈수록 변화량이 커진다. 저자들은 이것이 high-level semantic feature가 뒤쪽 계층에 존재하기 때문에 객체별 적응도 주로 후반부 계층에서 크게 일어난다는 근거라고 본다.

spatial modulator의 $\tilde{\gamma}_c$를 보면 상당한 sparsity가 나타난다. 마지막 convolution layer를 제외하면 약 60% parameter가 0이고, 마지막 `conv5_3`에서는 약 70% feature map이 spatial guide와 상호작용한다. 이 결과는 공간 prior가 초반부터 강하게 작동하는 것이 아니라, feature extraction이 어느 정도 진행된 뒤 후반부에서 강하게 결합된다는 점을 시사한다.

### Ablation study

DAVIS 2017에서 mean IU로 ablation을 수행했다. 먼저 add-on 효과를 보면, CRF를 붙이면 54.4로 1.9 상승하고, online fine-tuning을 100 iteration만 추가해도 60.8로 8.3 상승한다. 저자들은 이 수치가 OSVOS의 1000 iteration보다도 5.7 높다고 적는다. 즉 제안 방법은 기본적으로 빠른데, 필요하면 소량 fine-tuning으로 더 강하게 만들 수도 있다.

모듈 중요도도 분명하다. visual modulator를 제거하면 mean IU가 33.0으로 크게 떨어지고, spatial modulator를 제거하면 40.1이 된다. 둘 다 중요하지만 visual guide 쪽이 더 핵심이라는 결론이다.

데이터 augmentation도 성능에 큰 영향을 준다. random crop을 없애면 1.9 감소, visual guide augmentation을 없애면 추가로 1.1 감소한다. 가장 큰 타격은 spatial guide augmentation 제거인데, 이 경우 35.6까지 급락한다. 저자들은 spatial prior를 perturbation 없이 주면 모델이 위치 정보에 과도하게 의존해 실제 비디오의 움직임 변화에 취약해진다고 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 매우 일관적이라는 점이다. 기존 one-shot video object segmentation의 병목이 “테스트 시 반복 fine-tuning”이라는 점을 정확히 짚고, 이를 meta-learning 스타일의 modulation으로 바꾸어 계산량을 근본적으로 줄였다. 단순히 빠르기만 한 것이 아니라, DAVIS 2016, DAVIS 2017, YoutubeObjects에서 fine-tuning 없는 방법보다 큰 폭으로 좋고, fine-tuning 기반 강한 방법과도 비슷한 수준에 도달한 점은 설득력이 있다.

두 번째 강점은 visual cue와 spatial cue의 역할 분담이 명확하다는 점이다. visual modulator는 “무엇을 찾을지”, spatial modulator는 “어디를 찾을지”를 담당한다. 이 구조 덕분에 유사 객체가 많은 장면에서는 spatial prior가 tracking에 도움을 주고, appearance 변화가 있는 장면에서는 visual embedding이 일반화에 도움을 준다. ablation에서도 두 모듈의 기여가 수치로 확인된다.

세 번째 강점은 분석이 비교적 충실하다는 점이다. 단순 benchmark 결과뿐 아니라 modulation parameter의 시각화, 계층별 modulation 강도 분석, sparsity 관찰 등을 통해 왜 이 방법이 동작하는지에 대한 내부 해석을 제공한다. 특히 deeper layer에서 modulation이 더 크게 작동한다는 관찰은 방법 설계와 잘 맞물린다.

한편 한계도 분명하다. 첫째, 절대적인 최고 정확도를 항상 달성한 것은 아니다. 예를 들어 DAVIS 2016에서는 OSVOS가 더 높은 성능을 보인다. 즉 속도-성능 trade-off를 크게 개선했지만, 최고 성능 자체를 완전히 갱신했다고 보기는 어렵다.

둘째, spatial prior는 이전 프레임 예측에 의존하므로, error propagation을 완전히 제거한 것은 아니다. 저자들은 coarse Gaussian prior가 exact mask보다 안정적이라고 주장하지만, 여전히 이전 예측이 크게 틀리면 다음 프레임 성능이 영향을 받을 가능성은 남아 있다. 다만 논문은 이것을 직접 정량적으로 깊게 분석하지는 않는다.

셋째, backbone과 설계 선택이 VGG16 기반 FCN에 많이 묶여 있다. modulation layer를 어디에 넣는지, 왜 첫 네 계층은 제외하는지에 대한 실험은 제시되지만, 더 현대적인 backbone이나 다른 segmentation architecture로의 일반성은 본문에서 충분히 검증되지 않는다. OSVOS-M, MaskTrack-M 실험이 일부 일반성을 보이기는 하나, visual modulator만 추가한 수준이다.

넷째, 논문은 meta-learning과 few-shot learning의 관점에서 의미를 강조하지만, 보다 이론적인 관점에서 “왜 적은 modulation parameter만으로 full fine-tuning을 상당 부분 대체할 수 있는가”에 대한 설명은 경험적 수준에 머문다. 이는 당시 흐름상 자연스럽지만, 방법의 표현력 한계를 엄밀히 분석한 것은 아니다.

## 6. 결론

이 논문은 one-shot video object segmentation에서 매우 비싼 online fine-tuning을 대체하기 위해, **network modulation**이라는 빠른 적응 메커니즘을 제안한다. 핵심은 annotated object의 appearance에서 channel-wise scale을 만들고, 이전 프레임 위치 prior에서 spatial bias를 만들어 segmentation network의 intermediate feature를 조정하는 것이다. 이렇게 하면 반복 최적화 없이도 특정 객체에 즉시 적응할 수 있다.

실험 결과를 보면 이 접근은 단순한 속도 개선 이상의 의미가 있다. fine-tuning 없는 기존 방법보다 훨씬 강하고, fine-tuning 기반 강한 baseline과도 비슷한 수준의 성능을 내면서 속도는 수십 배 빠르다. 또한 modulation parameter 분석을 통해 이 방법이 객체 appearance와 spatial tracking 정보를 실제로 의미 있게 인코딩한다는 점도 보여준다.

실제 적용 측면에서 이 연구는 실시간 또는 대규모 video processing 환경에서 특히 가치가 크다. 또한 저자들이 결론에서 언급하듯이, modulation 기반 적응은 video object segmentation 외에도 visual tracking, image stylization 같은 다른 few-shot 또는 conditional adaptation 문제로 확장될 가능성이 있다. 따라서 이 논문은 “테스트 시 weight fine-tuning” 중심이던 당시 video segmentation 흐름에서, 더 빠르고 구조적으로 우아한 meta-adaptation 방향을 제시한 중요한 작업으로 볼 수 있다.
