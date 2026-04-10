# Deep Residual Learning for Instrument Segmentation in Robotic Surgery

* **저자**: Daniil Pakhomov, Vittal Premachandran, Max Allan, Mahdi Azizian, Nassir Navab
* **발표연도**: 2017
* **arXiv**: <https://arxiv.org/abs/1703.08580>

## 1. 논문 개요

이 논문은 로봇 보조 최소침습수술(RMIS) 영상에서 수술 도구를 픽셀 단위로 분할하는 문제를 다룬다. 구체적으로는 내시경 카메라 영상에서 각 픽셀이 수술 도구인지 배경인지를 구분하는 **binary segmentation**뿐 아니라, 도구의 서로 다른 부분까지 나누는 **multi-class segmentation**으로 문제를 확장한다.

논문이 해결하려는 핵심 문제는 수술 장면의 시각적 복잡성 때문에 도구 분할이 매우 어렵다는 점이다. 실제 수술 영상에는 그림자, 강한 specular reflection, 연기, 혈액, 가림(occlusion), 복잡한 조직 텍스처가 동시에 존재한다. 따라서 단순한 색 기반 규칙이나 얕은 모델만으로는 일반화가 어렵다. 이 문제는 단순히 영상 처리 성능의 문제가 아니라, 수술 중 도구 추적, 자세 추정, 그리고 해부학 정보나 수술 계획 정보를 영상 위에 겹쳐 보여주는 augmented overlay 시스템의 핵심 전처리 단계라는 점에서 중요하다. 도구를 정확히 분할하지 못하면 overlay가 도구를 가려버리거나, 추적 시스템이 오동작할 수 있다.

이 논문의 목표는 크게 두 가지다. 첫째, 기존의 FCN 기반 binary segmentation 성능을 더 향상시키는 것이다. 둘째, 도구 전체를 하나의 클래스처럼 다루는 데서 나아가, 도구의 shaft와 manipulator 같은 부분을 따로 분할하는 multi-class setting까지 확장하는 것이다. 저자들은 이를 위해 **ResNet-101 기반 FCN**, **dilated convolution**, **reduced downsampling**, **bilinear interpolation**을 결합한 구조를 제안한다.

![그림 1: RMIS 시술에서 추출한 예시 프레임으로, 복잡한 조명과 색상 분포를 보여주며 기구 분할이 매우 어려운 문제임을 나타낸다.](https://ar5iv.labs.arxiv.org/html/1703.08580/assets/x1.png)

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 원래 이미지 분류에 강한 **deep residual network**, 특히 **ResNet-101**을 semantic segmentation 문제에 맞게 fully convolutional하게 바꾸고, segmentation에서 중요한 공간 해상도를 유지하도록 네트워크 구조를 수정하는 데 있다.

기존 분류용 CNN은 여러 단계의 stride와 pooling 때문에 feature map의 해상도가 크게 줄어든다. 예를 들어 원래의 ResNet은 출력이 입력보다 $32\times$ 축소된다. 이런 구조는 이미지 전체를 하나의 클래스로 분류할 때는 괜찮지만, 픽셀 단위 예측이 필요한 segmentation에서는 경계가 너무 거칠어진다. 기존 FCN 계열은 deconvolution이나 skip architecture로 이 문제를 보완했지만, 저자들은 다른 방향을 택한다. 마지막 두 번의 downsampling을 제거해 출력 feature map의 해상도를 높이고, 그에 따라 receptive field와 사전학습 가중치의 의미가 무너지지 않도록 **dilated convolution**을 적용한다.

즉, 이 논문의 차별점은 다음과 같이 요약할 수 있다. 첫째, robotic instrument segmentation에 **ResNet-101의 residual learning**을 본격적으로 적용했다. 둘째, 단순히 FCN으로 바꾸는 데서 멈추지 않고, **downsampling 감소 + dilated convolution + bilinear upsampling** 조합을 통해 고해상도 dense prediction을 구현했다. 셋째, 기존 연구가 주로 binary segmentation에 머물렀던 반면, 이 논문은 **tool part segmentation**까지 실험해 multi-class 기준선을 제시했다.

## 3. 상세 방법 설명

논문이 푸는 문제는 입력 영상 $I$의 각 픽셀에 대해 $C$개 클래스 중 하나를 할당하는 것이다. binary segmentation에서는 $C=2$로 background와 tool을 구분하고, multi-class segmentation에서는 $C=3$으로 background, shaft, manipulator를 구분한다. 각 학습 샘플은 RGB 이미지 $I_i \in \mathbb{R}^{h \times w \times 3}$와, one-hot encoded annotation $A_i \in {0,1}^{h \times w \times C}$로 구성된다.

![그림 3: 단순한 CNN이 FCN으로 변환되는 과정. (a)는 입력 이미지의 특정 픽셀(빨간 점)을 중심으로 한 패치에 CNN을 적용해 클래스 점수 벡터(조작기, 샤프트, 배경)를 출력하는 예이다. (b)는 완전연결층을 1×1 합성곱 층으로 변환하여 네트워크를 완전 합성곱 구조로 바꾸고, 이를 통해 밀집 예측(dense prediction)이 가능해짐을 보여준다.
(c)는 다운샘플링을 줄이고 dilated convolution을 적용한 뒤, bilinear interpolation으로 업샘플링하여 픽셀 단위 예측을 수행하는 구조를 나타낸다.](https://ar5iv.labs.arxiv.org/html/1703.08580/assets/img/final_test_main_pic.png)

전체 파이프라인은 다음과 같은 흐름으로 이해하면 된다. 먼저 ImageNet 등에서 학습된 **ResNet-101 classification backbone**을 가져온다. 그 다음 classification network를 segmentation이 가능하도록 **Fully Convolutional Network**로 변환한다. 이후 segmentation 해상도를 높이기 위해 네트워크 후반부의 downsampling을 줄이고, 그로 인해 바뀐 feature spacing을 보정하기 위해 뒤따르는 convolution들을 dilated convolution으로 바꾼다. 마지막으로 낮은 해상도의 class score map을 bilinear interpolation으로 원래 영상 크기까지 올려서 픽셀별 예측을 만든다. 학습은 이 최종 예측과 정답 마스크 사이의 pixel-wise cross-entropy를 최소화하는 방식으로 end-to-end 수행된다.

### 3.1 Residual learning의 역할

논문은 ResNet의 핵심 단위를 residual unit으로 설명한다. 일반적인 CNN 블록은 입력 $x_l$를 convolution과 비선형 함수에 통과시켜 다음 출력을 만든다.

$$
y_l = g(x_l, w_l)
$$

$$
x_{l+1} = f(y_l)
$$

여기서 $g(\cdot,\cdot)$는 convolution 연산, $w_l$는 해당 층의 필터 파라미터, $f(\cdot)$는 ReLU 같은 비선형 함수다.

반면 residual unit은 원하는 매핑 $H(x_l)$을 직접 학습하는 대신, 입력과의 차이인 residual function $F(x_l, W_l)$을 학습한다.

$$
y_l = h(x_l) + F(x_l, W_l)
$$

$$
x_{l+1} = f(y_l)
$$

이 논문에서는 $h(x_l)=x_l$인 identity mapping을 사용한다. 즉, 블록의 출력은 “입력 그대로의 경로”와 “추가로 배워야 할 변화량”의 합으로 표현된다. 이런 구조는 깊은 네트워크에서 optimization을 더 쉽게 만들어 주며, 논문은 바로 이 점 때문에 100층이 넘는 ResNet-101을 segmentation backbone으로 선택한다.

직관적으로 말하면, 네트워크는 “완전히 새로운 표현 전체”를 매번 다시 만드는 대신 “현재 표현에서 무엇을 더 고치면 되는지”를 배우게 된다. 이는 깊은 네트워크를 안정적으로 학습시키고, 더 강한 feature extractor를 제공한다.

### 3.2 ResNet-101을 FCN으로 변환

분류용 CNN은 보통 마지막에 global average pooling이나 fully connected layer를 두어 이미지 전체에 대한 하나의 클래스 score를 낸다. 하지만 segmentation에서는 공간 위치별 예측이 필요하므로, 네트워크를 **fully convolutional**하게 바꿔야 한다.

논문은 이를 위해 ResNet-101의 최종 average pooling layer를 제거하고, fully connected layer를 $1 \times 1$ convolution으로 바꾼다. 그러면 네트워크는 입력 크기에 상관없이 동작하면서, 각 공간 위치마다 클래스 score를 내는 구조가 된다. 이것이 FCN의 기본 아이디어다.

다만 이렇게만 바꾸면 출력 해상도는 여전히 낮다. 원래 ResNet은 여러 stride-2 convolution으로 인해 feature map이 입력보다 $32\times$ 작아진다. 예를 들어 입력이 큰 영상이어도 마지막 score map은 매우 조밀하지 못하다. 논문은 이 문제를 해결하기 위해 마지막 두 downsampling layer의 stride를 2가 아니라 1로 바꾼다. 그 결과 출력 해상도는 입력 대비 $8\times$ 축소 수준으로 개선된다. 이후 bilinear interpolation으로 원래 크기로 올린다.

### 3.3 Dilated convolution의 역할

문제는 stride를 줄이면 원래 사전학습된 필터가 기대하던 feature spacing이 달라진다는 점이다. 원래는 더 거칠게 downsample된 feature map 위에서 학습된 convolution이, 이제 더 조밀한 feature map 위에서 그대로 적용되면 receptive field의 의미가 달라질 수 있다. 이를 해결하기 위해 논문은 **dilated convolution**을 사용한다.

1차원에서 dilated convolution은 다음처럼 표현된다.

$$
y[i] = \sum_{k=1}^{K} x[i + r \cdot k] , w[k]
$$

여기서 $r$은 dilation rate다. 일반 convolution이 인접한 위치를 연속적으로 보는 반면, dilated convolution은 간격을 두고 샘플링한다. 그래서 파라미터 수를 늘리지 않으면서도 더 넓은 receptive field를 유지할 수 있다.

이 논문에서는 어떤 downsampling을 제거한 이후, 그 뒤에 오는 convolution들을 적절한 dilation rate로 바꾼다. 이렇게 하면 두 가지 장점이 있다. 첫째, 더 높은 해상도의 feature map을 유지할 수 있다. 둘째, 원래 classification network에서 학습된 가중치를 비교적 자연스럽게 재사용할 수 있다. 즉, segmentation에 맞게 공간 해상도는 올리되, 분류 backbone의 강력한 표현력을 잃지 않으려는 설계다.

### 3.4 최종 예측과 학습 목표

네트워크가 출력하는 것은 해상도가 줄어든 class score map이다. 논문은 이것을 **bilinear interpolation**으로 입력 크기까지 업샘플링한다. 이때 논문은 deconvolution layer를 따로 학습하기보다는, 단순한 bilinear interpolation을 사용한다. 저자들의 관점에서는 핵심은 업샘플링 자체를 복잡하게 만드는 것이 아니라, 그 전 단계의 feature map 해상도를 충분히 높이는 것이다.

학습 목표는 **normalized pixel-wise cross-entropy loss**다. 각 픽셀 위치에서 예측한 클래스 확률 분포와 정답 one-hot label 사이의 cross-entropy를 계산하고, 이를 전체 픽셀에 대해 평균내어 최소화한다. 논문은 loss의 정확한 식을 전개하지는 않았지만, 의미상으로는 각 픽셀 $p$에 대해 정답 클래스 $c$의 로그 확률 $-\log \hat{y}_{p,c}$를 더하는 일반적인 semantic segmentation 목적함수로 이해하면 된다.

최적화는 **Adam**을 사용하며, 학습률은 $10^{-4}$이다. 저자들은 다섯 개의 learning rate를 grid search한 결과 이 값이 validation에서 가장 좋았다고 밝힌다. Adam의 나머지 하이퍼파라미터는 원 논문에서 권장한 기본값을 사용했다고 적고 있다. 배치 크기, epoch 수, 데이터 augmentation 여부 같은 세부 학습 설정은 제공된 텍스트에는 명시되어 있지 않다.

## 4. 실험 및 결과

실험은 **MICCAI Endoscopic Vision Challenge Robotic Instruments dataset**에서 수행되었다. 훈련 데이터는 ex-vivo 환경의 45초 길이 2D stereo image sequence 네 개로 구성되며, Large Needle Driver 도구가 포함되어 있다. 각 픽셀은 background, shaft, articulated head 중 하나로 라벨링되어 있다. 테스트 데이터는 비슷한 배경을 가진 15초 길이 시퀀스 네 개와, 학습에 없는 도구가 포함된 1분 길이 2-instrument 시퀀스 두 개를 포함한다. 저자들은 이 데이터가 occlusion과 articulation을 포함한다고 설명한다. 즉, 단순한 정적 배경이 아니라 실제로 어려운 수술 장면 변형이 존재한다는 뜻이다.

![그림 4: 본 방법의 정성적 결과. (a)와 (c)는 데이터셋에서 가져온 예시 이미지 프레임. (b)는 (a) 이미지에 대한 이진 분할 결과. (d)는 (c) 이미지에 대한 다중 클래스 분할 결과.](https://ar5iv.labs.arxiv.org/html/1703.08580/assets/img/binary_segmentation_updated.png)

### 4.1 Binary segmentation 결과

binary segmentation에서는 기존 state-of-the-art인 FCN-8s 기반 방법과 비교한다. 표에 따르면 이전 방법은 sensitivity 87.8%, specificity 88.7%, balanced accuracy 88.3%를 기록했고, 본 논문 방법은 sensitivity 85.7%, specificity 98.8%, balanced accuracy 92.3%를 기록했다.

이 결과의 의미는 단순히 평균 점수가 조금 오른 정도가 아니다. sensitivity는 다소 낮아졌지만 specificity가 크게 올랐고, 그 결과 **balanced accuracy가 약 4%p 향상**되었다. balanced accuracy는 positive와 negative 양쪽을 균형 있게 보는 지표이므로, 배경과 도구 픽셀의 비율이 불균형할 수 있는 segmentation 문제에서 의미가 크다. 즉, 이 모델은 도구를 조금 더 놓칠 수는 있어도, 배경을 도구로 잘못 분류하는 오류를 크게 줄였고, 전체적으로 더 균형 잡힌 성능을 보였다고 해석할 수 있다.

논문은 이 결과를 통해 residual network와 dilated convolution 기반 FCN 설계가 기존 FCN-8s보다 더 강력하다고 주장한다. 다만 제공된 표만으로는 통계적 유의성 검정이나 영상별 분산은 확인할 수 없다. 따라서 “항상 안정적으로 우월하다”기보다, 적어도 이 데이터셋과 평가 프로토콜에서는 유의미한 개선이 관찰되었다고 보는 것이 정확하다.

### 4.2 Multi-class segmentation 결과

multi-class segmentation에서는 세 클래스, 즉 Manipulator(C1), Shaft(C2), Background(C3)에 대해 **IoU(Intersection over Union)**를 보고한다. 영상별 결과는 다음과 같은 경향을 보인다.

Video 1부터 6까지 전체 평균 IoU는 각각 81.4, 83.7, 81.6, 72.3, 74.9, 72.2이다. 클래스별로 보면 background인 C3는 모든 비디오에서 95% 이상으로 매우 높다. 반면 manipulator인 C1은 대체로 70%대 후반에서 80%대 초반, shaft인 C2는 44.9에서 70.2 사이로 가장 어렵다.

이 결과는 중요한 해석 포인트를 준다. background 클래스가 매우 높은 것은 장면 대부분이 배경이고, 배경이 상대적으로 안정적 패턴을 가지기 때문일 수 있다. 반면 도구의 세부 파트, 특히 shaft와 manipulator의 경계는 반사광, 관절 운동, 가림 등에 따라 분리가 더 어렵다. 실제로 C2가 가장 낮은 것은 shaft가 길고 얇으며, 반사와 motion blur에 취약하기 때문일 가능성이 있지만, 이것은 논문이 직접 원인 분석을 상세히 제공하지는 않는다. 따라서 “shaft가 구조적으로 가장 어려웠다”는 수준까지는 말할 수 있지만, 정확한 실패 원인을 논문이 체계적으로 분석했다고 보기는 어렵다.

저자들은 자신들이 이 데이터셋에서 **multi-class segmentation 결과를 처음 보고한 것**이라고 주장한다. 따라서 이 실험은 단순 비교 우위라기보다, 이후 연구를 위한 baseline 성격도 가진다. 정성적 결과 그림에서는 binary와 multi-class 예측 예시를 제시하며, 모델이 도구 전체뿐 아니라 도구 파트까지 시각적으로 구분할 수 있음을 보인다. 다만 제공된 텍스트만으로는 실패 사례나 boundary error의 전형적인 패턴까지는 확인할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은, 당시 강력한 backbone인 ResNet-101을 robotic instrument segmentation에 효과적으로 이식했다는 점이다. 단순히 backbone만 교체한 것이 아니라, segmentation에 필요한 해상도 문제를 해결하기 위해 reduced downsampling과 dilated convolution을 결합했다. 이 설계는 이후 semantic segmentation 분야에서 널리 쓰인 DeepLab류 아이디어와도 자연스럽게 연결되며, 의료영상의 특수 과제에 이를 잘 적용했다는 의미가 있다.

또 다른 강점은 문제 정의의 확장성이다. 기존 연구가 주로 tool-vs-background의 binary 설정에 머물렀다면, 이 논문은 shaft와 manipulator를 구분하는 multi-class segmentation으로 나아갔다. 이는 후속 작업인 pose estimation, articulation analysis, part-aware tracking에 직접적으로 더 유용하다. 즉, 단순 성능 향상뿐 아니라 문제 자체를 실제 응용에 더 가깝게 만들었다.

실험적으로도 balanced accuracy 4%p 향상은 분명한 성과다. 특히 specificity의 큰 향상은 수술 장면에 정보를 overlay할 때 잘못된 도구 마스크로 인한 시각적 간섭을 줄이는 데 도움이 될 수 있다. 또한 multi-class 결과를 공개적으로 제시해 이후 연구 비교 기준을 제공했다는 점도 의미가 있다.

하지만 한계도 분명하다. 첫째, 데이터셋 규모가 작다. 훈련용 시퀀스가 제한적이고 ex-vivo 환경 중심이기 때문에, 실제 임상 환경의 다양성을 얼마나 잘 반영하는지는 불명확하다. 연기, 혈액, 조명 변화가 있다고는 하지만, in-vivo 수술 장면까지 일반화된다고 결론내리기는 어렵다.

둘째, 학습 설정과 구현 세부사항이 충분히 자세하지 않다. 예를 들어 data augmentation, batch size, 학습 iteration 수, validation split 방식, class imbalance 처리 여부 등은 제공된 텍스트에 명확히 나오지 않는다. 따라서 재현성을 엄밀히 평가하기는 어렵다.

셋째, 비교 범위가 넓지 않다. binary segmentation에서는 FCN-8s 기반 이전 방법과 비교했지만, 다른 backbone이나 다른 upsampling 전략, 혹은 CRF/post-processing 같은 요소와의 ablation은 제공되지 않는다. 그래서 성능 향상이 정확히 어느 설계 요소에서 얼마나 비롯되었는지 분리해서 보기 어렵다. 예를 들어 “ResNet backbone 자체의 효과”와 “dilated convolution에 의한 효과”를 개별적으로 분석한 실험은 이 텍스트에는 없다.

넷째, multi-class segmentation 결과는 제시되지만, 여기에 대한 baseline 비교가 없다. 저자들이 최초 보고라고 주장하는 만큼 이해할 수는 있지만, 절대적인 난도나 개선 폭을 상대 비교로 해석하기는 어렵다. 또한 unseen instrument가 포함된 테스트에서 어느 정도 강건한지, 도구 종류 변화에 대한 일반화 능력이 어느 수준인지도 정량적으로 깊게 분석되지는 않는다.

비판적으로 보면, 이 논문은 “강한 backbone + segmentation 친화적 구조 수정”이라는 설계가 유효함을 잘 보여 주지만, 실패 사례 분석과 ablation 측면에서는 다소 제한적이다. 따라서 방법의 핵심은 분명하지만, 왜 특정 클래스에서 약한지, 어떤 구성 요소가 가장 중요한지에 대한 설명은 후속 연구가 더 보완해야 할 부분이다.

## 6. 결론

이 논문은 robotic surgery 영상에서 수술 도구를 분할하기 위해 **ResNet-101 기반 fully convolutional architecture**를 제안하고, 해상도 손실 문제를 해결하기 위해 **stride reduction**, **dilated convolution**, **bilinear interpolation**을 결합했다. 그 결과 binary tool segmentation에서 기존 방법 대비 balanced accuracy를 약 4%p 향상시켰고, 더 나아가 tool part를 구분하는 multi-class segmentation 결과를 제시했다.

핵심 기여는 세 가지로 정리할 수 있다. 첫째, deep residual learning을 robotic instrument segmentation에 효과적으로 적용했다. 둘째, segmentation 해상도 문제를 구조적으로 해결해 더 정밀한 예측을 가능하게 했다. 셋째, binary segmentation을 넘어 multi-class tool-part segmentation의 실험 기준선을 마련했다.

실제 적용 측면에서 이 연구는 수술 도구 추적, 자세 추정, augmented reality overlay, 수술 보조 인터페이스 등과 직접 연결될 수 있다. 또한 향후 연구에서는 이 논문의 구조를 출발점으로 더 큰 데이터셋, 더 다양한 도구 종류, temporal modeling, skip connection 강화, attention, transformer 기반 segmentation 등으로 확장할 수 있다. 제공된 텍스트 범위 안에서 보면, 이 논문은 robotic surgery vision에서 “고전적 color-feature 기반 분할”에서 “강력한 deep semantic segmentation”으로 넘어가는 중요한 전환점 중 하나로 평가할 수 있다.
