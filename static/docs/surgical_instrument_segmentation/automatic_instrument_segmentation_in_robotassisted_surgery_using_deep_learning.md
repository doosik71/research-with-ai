# Automatic Instrument Segmentation in Robot-Assisted Surgery Using Deep Learning

- **저자**: Alexey A. Shvets, Alexander Rakhlin, Alexandr A. Kalinin, Vladimir I. Iglovikov
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1803.01207

## 1. 논문 개요

이 논문은 robot-assisted surgery 환경에서 수술 도구를 영상으로부터 자동으로 분할하는 문제를 다룬다. 더 구체적으로는 수술 영상의 각 픽셀에 대해 그것이 instrument인지 background인지 구분하는 binary segmentation과, instrument의 각 부분이나 서로 다른 도구 종류를 구분하는 multi-class segmentation을 동시에 다룬다.

연구 문제의 핵심은 수술 장면이 매우 복잡하다는 점이다. 실제 수술 영상에는 shadow, specular reflection, blood에 의한 occlusion, camera lens fogging, 그리고 조직 배경의 복잡한 질감 변화가 존재한다. 이런 환경에서는 단순한 색상 기반 또는 texture 기반 기법만으로는 instrument의 정확한 위치를 안정적으로 찾기 어렵다. 그러나 instrument segmentation은 이후 단계인 tracking, pose estimation, intra-operative guidance의 입력으로 매우 중요하므로, 정밀한 pixel-wise segmentation이 필수적이다.

논문은 이 문제를 해결하기 위해 deep learning 기반 segmentation 모델들을 적용하고 비교한다. 특히 U-Net 계열 구조를 중심으로, encoder를 ImageNet pretrained backbone으로 교체한 TernausNet과 LinkNet 변형을 사용해 성능을 개선하고, 당시 기준 state-of-the-art 수준의 결과를 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 의료 영상 segmentation에서 강력한 encoder-decoder 구조를 robotic surgery instrument segmentation에 맞게 적용하고, 특히 pretrained encoder를 활용해 제한된 데이터셋에서도 더 좋은 성능을 얻는 것이다.

기본 직관은 다음과 같다. segmentation 문제에서는 한편으로는 장면의 큰 문맥 정보를 이해해야 하고, 다른 한편으로는 도구 경계를 정밀하게 복원해야 한다. U-Net 계열 구조는 contracting path에서 문맥 정보를 추출하고, expanding path에서 해상도를 복원하며, skip connection으로 encoder의 고해상도 특징을 decoder에 전달한다. 이 구조는 localization과 semantic understanding을 동시에 달성하는 데 적합하다.

기존 접근과의 차별점은 두 가지로 볼 수 있다. 첫째, 단순한 handcrafted feature 기반 방법이 아니라 end-to-end deep network를 사용한다. 둘째, 원래의 U-Net뿐 아니라 pretrained VGG11, VGG16, ResNet34를 encoder로 사용하는 구조를 실험해, 데이터가 적은 상황에서 pretrained representation이 실제로 성능 개선에 기여함을 보여준다. 논문은 특히 TernausNet-16이 binary segmentation과 parts segmentation에서 가장 좋은 성능을 낸다고 보고한다.

## 3. 상세 방법 설명

전체 파이프라인은 비교적 단순한 end-to-end semantic segmentation 시스템이다. 입력은 수술 비디오 프레임이고, 출력은 입력과 동일한 해상도의 segmentation mask이다. 각 픽셀에 대해 background 또는 특정 class의 확률을 예측한 뒤, 이를 thresholding 또는 class assignment로 최종 mask로 변환한다.

### 데이터셋과 라벨 구조

학습 데이터는 da Vinci Xi surgical system으로 획득한 고해상도 stereo camera 영상으로 구성된다. 논문에 따르면 training set은 `8 x 225-frame sequences`이며, 각 프레임은 RGB 형식의 `1920 x 1080` 해상도를 가진다. 원본 카메라 이미지를 얻기 위해 `(320, 28)` 위치부터 crop하여 `1280 x 1024` 이미지를 추출한다. ground truth label은 left frame에만 존재하므로, 실제 학습에는 left channel만 사용한다.

라벨은 두 가지 방식으로 사용된다.

첫째, binary segmentation에서는 모든 instrument pixel을 하나의 foreground class로 보고 background와 구분한다.

둘째, multi-class segmentation에서는 두 설정이 있다.
하나는 instrument의 part를 구분하는 설정으로, rigid shaft, articulated wrist, claspers 등의 부분을 분리한다.
다른 하나는 instrument type을 구분하는 설정으로, left/right prograsp forceps, monopolar curved scissors, large needle driver, miscellaneous 등을 class로 둔다.

논문에는 part label 값이 `(10, 20, 30, 40, 0)`으로 인코딩된다고 적혀 있는데, 추출 텍스트 품질상 정확히 몇 개의 전경 part class가 최종 실험에 쓰였는지는 약간 흐릿하다. 다만 Figure 설명에서는 `3 classes: rigid shaft, articulated wrist and claspers`라고 명시되어 있어, parts segmentation의 핵심 대상은 이 세 부분으로 이해하는 것이 자연스럽다.

### 네트워크 구조

논문은 네 가지 모델을 비교한다.

첫 번째는 기본 U-Net이다. U-Net은 encoder-decoder 구조를 가지며, encoder에서는 convolution과 pooling을 반복해 feature map 해상도를 줄이고 channel 수를 늘린다. decoder에서는 upsampling과 convolution으로 해상도를 복원한다. 그리고 같은 단계의 encoder feature를 skip connection으로 decoder에 연결해 세밀한 경계 정보를 복원한다.

두 번째와 세 번째는 TernausNet-11, TernausNet-16이다. 이는 U-Net-like 구조이지만 encoder로 각각 pretrained VGG11, VGG16을 사용한다. 논문 설명에 따르면 VGG11은 7개의 convolutional layer와 5개의 max-pooling으로 구성되며, 모든 convolution kernel은 `3 x 3`이다. TernausNet-16 역시 유사한 구조를 가지되 더 깊은 encoder를 사용한다. pretrained encoder의 목적은 ImageNet에서 학습된 일반 시각 특징을 활용해, 적은 의료 영상 데이터에서도 더 안정적인 feature extraction을 가능하게 하는 데 있다.

네 번째는 LinkNet-34이다. 이 모델은 encoder로 pretrained ResNet34를 사용한다. 초기 블록은 `7 x 7` convolution과 stride 2, 이후 max-pooling으로 downsampling을 수행한다. 그 뒤에는 residual block들이 이어진다. 논문 설명에 따르면 각 residual block의 첫 convolution은 stride 2로 downsampling을 수행하고, 나머지는 stride 1을 사용한다. decoder는 여러 decoder block으로 구성되고, encoder의 대응 feature를 decoder에 더하는 방식으로 정보를 전달한다. 각 decoder block은 `1 x 1` convolution으로 filter 수를 4배 줄인 뒤, batch normalization과 transposed convolution으로 upsampling한다.

즉, U-Net과 TernausNet은 skip connection을 통한 feature concatenation 중심의 encoder-decoder 계열이고, LinkNet은 residual encoder와 더 가벼운 decoder를 활용해 빠른 추론 속도를 노린 구조라고 이해할 수 있다.

### 학습 목표와 손실 함수

평가 지표로는 Jaccard index, 즉 IoU를 사용한다. 논문은 두 집합 $A$와 $B$에 대해 Jaccard index를 다음처럼 정의한다.

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

이를 픽셀 단위의 예측 문제로 확장해, 정답 라벨 $y_i$와 예측 확률 $\hat{y}_i$를 사용한 형태로 표현한다.

$$
J = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i \hat{y}_i}{y_i + \hat{y}_i - y_i \hat{y}_i} \right)
$$

이 식은 segmentation mask와 prediction이 얼마나 겹치는지를 직접 반영한다. 다만 실제로는 segmentation을 픽셀 단위 classification 문제로도 볼 수 있으므로, 논문은 classification loss $H$도 함께 사용한다. binary segmentation에서는 binary cross entropy, multi-class segmentation에서는 categorical cross entropy를 사용한다.

최종 loss는 다음과 같다.

$$
L = H - \log J
$$

이 손실 함수의 의미는 직관적이다. $H$는 각 픽셀의 class probability를 정확하게 맞추도록 만들고, $-\log J$는 전체 mask 차원에서 prediction과 ground truth의 overlap이 커지도록 유도한다. 즉, 픽셀 단위 정확도와 mask 수준의 형태적 일치를 동시에 최적화하려는 설계이다.

### 추론 절차

모델 출력은 입력 이미지와 같은 크기의 확률 맵이다. binary segmentation에서는 validation set에서 선택한 threshold `0.3`을 적용한다. 확률이 0.3 미만이면 0, 0.3 이상이면 255로 변환해 최종 binary mask를 만든다.

multi-class segmentation에서는 비슷한 절차를 사용하되, 각 픽셀에 대해 class별 예측을 바탕으로 해당 정수 label을 할당한다. 논문은 구체적인 multi-class post-processing 세부 규칙까지는 길게 설명하지 않으며, 별도의 복잡한 후처리보다는 네트워크 출력 자체를 주로 사용한 것으로 보인다.

## 4. 실험 및 결과

### 실험 설정

논문은 세 가지 task를 평가한다.

첫째, binary segmentation이다. instrument 전체를 foreground로 간주한다.

둘째, parts segmentation이다. instrument의 부분 구조를 여러 class로 나눈다.

셋째, instrument segmentation이다. 서로 다른 종류의 instrument를 여러 class로 분류한다. 표와 본문을 보면 이 경우 `7 classes` 기준으로 평가한 것으로 보인다.

평가 지표는 IoU와 Dice coefficient이며, 추가로 inference time도 측정했다. 추론 속도는 `1280 x 1024` 해상도 이미지 한 장 기준으로, `NVIDIA GTX 1080 Ti` GPU에서 측정했다고 적혀 있다.

### 정량 결과

표에 따르면 binary segmentation에서 가장 좋은 모델은 TernausNet-16이다. 성능은 $IoU = 0.836$, $Dice = 0.901$이다. 이는 논문 시점 기준으로 최고 수준의 결과라고 주장한다.

parts segmentation에서도 TernausNet-16이 가장 좋다. 결과는 $IoU = 0.655$, $Dice = 0.760$이다. 이는 instrument 전체를 foreground로만 구분하는 것보다 훨씬 어려운 문제인데도 상당한 성능을 보인다.

반면 instrument type segmentation에서는 성능이 눈에 띄게 낮아진다. 가장 좋은 모델은 TernausNet-11이며, $IoU = 0.346$, $Dice = 0.459$를 기록한다. 여기서는 더 깊은 TernausNet-16보다 TernausNet-11이 약간 더 좋았다.

이 차이는 논문에서도 명시적으로 설명한다. instrument class 수가 많고, 일부 class는 training set에서 등장 빈도가 매우 적기 때문에 데이터 부족의 영향을 크게 받았다는 것이다. 즉, model capacity만 늘린다고 해결되지 않고 class imbalance와 sample scarcity가 중요한 병목이라는 해석이다.

### 속도 비교

속도 측면에서는 LinkNet-34가 가장 빠르다. 논문은 binary segmentation에서 이 모델이 약 `88 ms` 정도가 걸려, TernausNet보다 두 배 이상 빠르다고 말한다. 정확도는 TernausNet 계열보다 다소 낮지만, 실시간성이나 경량성 관점에서는 장점이 있다.

즉, 성능만 보면 TernausNet-16이 가장 강력하고, 속도까지 고려하면 LinkNet-34가 유리한 선택지다. 이 비교는 실제 수술 보조 시스템처럼 latency가 중요한 응용에서 의미가 있다.

### 결과의 해석

이 실험은 pretrained encoder의 효과를 상당히 설득력 있게 보여준다. 단순 U-Net 대비 TernausNet 계열은 binary와 parts segmentation에서 뚜렷한 성능 향상을 보인다. 특히 의료 영상처럼 데이터가 적고 annotation 비용이 큰 환경에서, 자연영상으로 pretraining된 backbone이 실제로 유용하다는 점을 실증한다.

동시에 논문은 class granularity가 높아질수록 문제 난도가 급격히 상승한다는 점도 보여준다. instrument 전체를 foreground로 보는 binary task는 비교적 성공적으로 해결되지만, instrument type까지 세분화하면 데이터 부족으로 인해 성능이 크게 떨어진다. 이는 이후 연구에서 더 큰 데이터셋, class balancing, stronger augmentation, temporal modeling이 필요함을 시사한다.

## 5. 강점, 한계

이 논문의 강점은 먼저 문제 설정이 매우 실용적이라는 점이다. robotic surgery에서 instrument segmentation은 tracking과 pose estimation의 기반이 되는 핵심 문제이며, 임상적 활용 가능성이 분명하다. 단순히 benchmark 성능만 높이려는 작업이 아니라, 실제 시스템의 perception module과 직접 연결되는 문제를 다룬다.

둘째, 방법이 과도하게 복잡하지 않다. U-Net, TernausNet, LinkNet이라는 비교적 표준적이고 재현 가능한 구조를 사용하면서도, pretrained encoder와 적절한 loss 설계로 강한 성능을 끌어냈다. 특히 $L = H - \log J$ 형태의 손실은 segmentation에서 픽셀 분류 정확도와 overlap quality를 동시에 고려하려는 실용적 선택이다.

셋째, binary, parts, instrument type segmentation을 함께 평가해 문제 난이도별 특성을 비교한 점이 좋다. 이를 통해 어떤 설정에서는 성능이 충분히 높고, 어떤 설정에서는 데이터 부족이 병목인지가 분명하게 드러난다. 또한 속도 비교까지 포함해 accuracy-speed tradeoff를 제시한 점도 응용 측면에서 의미가 있다.

반면 한계도 분명하다. 가장 큰 한계는 데이터셋 규모가 작다는 점이다. 논문 자체도 이를 인정한다. 특히 instrument type segmentation에서는 드물게 등장하는 class가 있어 학습이 어렵다. 따라서 보고된 성능은 모델 자체의 한계라기보다 데이터 부족의 영향을 강하게 받은 결과일 수 있다.

또 다른 한계는 temporal information을 사용하지 않는다는 점이다. 입력은 비디오이지만, 본 논문 모델은 프레임 단위 image segmentation으로 보인다. 수술 영상은 연속적인 motion과 temporal consistency가 매우 중요하므로, 시계열 정보를 활용하면 occlusion이나 blur 상황에서 더 강건해질 가능성이 있다. 그러나 논문은 이러한 temporal modeling을 다루지 않는다.

또한 실험은 challenge dataset 중심으로 구성되어 있어 일반화 성능을 넓게 검증했다고 보기는 어렵다. 예를 들어 다른 수술 종류, 다른 병원, 다른 조명 조건, 다른 기기 세팅에서의 domain shift 문제는 여기서 다뤄지지 않는다.

비판적으로 보면, 논문의 핵심 기여는 완전히 새로운 architecture 제안이라기보다 기존 segmentation architecture를 surgical instrument 문제에 효과적으로 적용하고 비교 평가한 데 있다. 따라서 방법론적 novelty는 상대적으로 제한적일 수 있다. 그러나 응용 문제에서의 강한 empirical contribution과 실용성은 충분히 가치가 있다.

## 6. 결론

이 논문은 robotic surgery 영상에서 surgical instrument를 자동 분할하기 위해 U-Net 계열의 deep learning 모델들을 적용하고, 특히 pretrained encoder를 사용하는 TernausNet과 LinkNet이 강력한 성능을 낼 수 있음을 보여준다. binary segmentation에서는 TernausNet-16이 매우 높은 성능을 기록했고, instrument parts segmentation에서도 가장 우수한 결과를 냈다. 반면 instrument type segmentation은 데이터 부족으로 인해 여전히 어려운 문제임을 확인했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, robotic instrument segmentation에 대한 end-to-end deep learning 파이프라인을 제시했다. 둘째, pretrained encoder 기반 구조가 제한된 의료 영상 데이터에서도 효과적임을 보였다. 셋째, 정확도뿐 아니라 추론 속도까지 비교해 실제 시스템 적용 관점의 통찰을 제공했다.

이 연구는 실제 수술 보조 시스템에서 instrument detection, tracking, pose estimation으로 이어지는 비전 모듈의 기반 기술로 중요할 가능성이 크다. 향후에는 더 큰 데이터셋, class imbalance 완화, temporal modeling, domain generalization 같은 방향으로 발전할 수 있다. 즉, 이 논문은 robotic surgery perception에서 실용적인 segmentation 기준선을 제시한 초기이자 중요한 작업으로 볼 수 있다.
