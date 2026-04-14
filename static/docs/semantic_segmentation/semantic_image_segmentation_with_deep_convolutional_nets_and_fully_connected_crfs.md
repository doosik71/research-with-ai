# Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs

- **저자**: Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1412.7062

## 1. 논문 개요

이 논문은 semantic image segmentation, 즉 이미지의 각 픽셀에 semantic class label을 부여하는 문제를 다룬다. 저자들은 당시 image classification과 object detection에서 매우 강력한 성능을 보이던 Deep Convolutional Neural Networks (DCNNs)를 segmentation에 적용하되, DCNN의 약점인 경계 localization 문제를 보완하기 위해 fully connected Conditional Random Field (CRF)를 결합한 시스템을 제안한다. 이 시스템이 바로 초기 형태의 **DeepLab**이다.

논문이 제기하는 핵심 문제는 다음과 같다. DCNN은 고수준 인식에는 강하지만, repeated max-pooling과 striding 때문에 출력 해상도가 낮아지고, spatial invariance가 강해져 객체의 정확한 경계를 맞추는 데 불리하다. 즉, “무엇이 있는지”는 잘 맞추지만 “정확히 어디까지가 그 객체인지”는 덜 정확하다. semantic segmentation에서는 픽셀 단위로 정밀한 위치 정보가 필요하므로, 이 약점은 매우 중요하다.

이 문제의 중요성은 분명하다. segmentation은 자율주행, 로봇 비전, 장면 이해 같은 응용에서 단순한 분류보다 훨씬 더 세밀한 공간적 이해를 요구한다. 논문은 DCNN의 recognition power와 probabilistic graphical model의 boundary refinement 능력을 결합하면, 정확도와 경계 정밀도를 동시에 얻을 수 있음을 보이려 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 두 부분으로 구성된다.

첫째, classification용 VGG-16을 dense prediction에 맞게 재구성한다. 이를 위해 fully connected layer를 convolutional layer로 바꾸고, 단순히 매우 거친 stride 32 출력만 쓰지 않고, **atrous algorithm** 또는 **hole algorithm**을 사용해 더 조밀한 stride 8의 score map을 효율적으로 계산한다. 이 부분은 dense semantic labeling을 가능하게 하는 계산 구조의 핵심이다.

둘째, 이렇게 얻은 DCNN의 coarse한 class score를 최종 결과로 쓰지 않고, 이를 unary term으로 사용하는 **fully connected CRF**를 추가한다. 이 CRF는 모든 픽셀 쌍 사이의 pairwise interaction을 고려하면서도, Gaussian filtering 기반 mean-field approximation 덕분에 계산 가능하다. 저자들의 주장은 단순 smoothing이 아니라, DCNN이 놓치는 세밀한 boundary를 복구하는 데 fully connected CRF가 특히 효과적이라는 점이다.

기존 접근과의 차별점도 분명하다. 당시 많은 방법은 region proposal, superpixel, 또는 bottom-up segmentation 결과에 먼저 의존한 뒤 DCNN으로 region을 분류했다. 이런 방식은 초기에 잘못 분할되면 이후 단계에서 회복하기 어렵다. 반면 이 논문은 이미지 전체에서 직접 dense prediction을 만들고, segmentation refinement는 나중에 CRF로 수행한다. 또한 local CRF가 아니라 fully connected CRF를 사용해 장거리 관계와 edge-sensitive refinement를 동시에 활용한다.

## 3. 상세 방법 설명

전체 파이프라인은 비교적 단순하다. 입력 이미지를 VGG-16 기반 DCNN에 넣어 coarse class score map을 계산하고, 이를 bilinear interpolation으로 원래 해상도에 가깝게 upsample한 뒤, fully connected CRF로 정제하여 최종 segmentation을 얻는다. 논문 그림 설명대로 보면, 구조는 “DCNN coarse score map → bilinear interpolation → fully connected CRF → final output”의 흐름이다.

### DCNN의 dense prediction 변환

저자들은 ImageNet으로 pretrain된 VGG-16을 segmentation용으로 바꾼다. 마지막 1000-class classifier를 21-class classifier로 바꾸는데, 이는 PASCAL VOC 2012의 20 foreground classes와 1 background class에 대응한다. fully connected layers를 convolution으로 바꾸어 전체 이미지에 sliding-window처럼 적용한다.

하지만 이렇게만 하면 출력 stride가 32라 너무 거칠다. 이를 해결하기 위해 마지막 두 max-pooling layer 이후의 subsampling을 제거하고, 그 뒤 convolution이 더 넓은 receptive field를 유지하도록 **holes**를 넣는 방식으로 계산한다. 실제 구현은 filter에 0을 직접 삽입하기보다, feature map을 sparse하게 sampling하는 식으로 설명된다. 이 방법 덕분에 근사 없이 원하는 subsampling rate에서 dense feature map을 효율적으로 계산할 수 있다.

논문은 이를 Caffe의 `im2col` 함수 수정으로 구현했다고 밝힌다. 결과적으로 8-pixel stride의 output map을 얻을 수 있으며, 테스트 시 bilinear interpolation으로 원본 해상도로 올린다. 저자들은 DCNN score map이 비교적 smooth하기 때문에, 이 interpolation이 실제로 충분히 잘 작동한다고 설명한다.

### 학습 목표

DCNN 학습의 loss는 output map의 각 spatial position에 대한 cross-entropy loss의 합이다. 즉, stride 8로 subsample된 출력 위치 각각에 대해 정답 label과 예측 확률을 비교한다. 논문은 모든 위치와 label을 동일 가중치로 취급한다고 명시한다. target은 ground-truth label을 8배 subsample한 것이다.

정리하면, 픽셀 단위 분류를 위해 사용하는 기본 목표는 각 위치에서의 multinomial classification loss이며, 전체 loss는 그 합이다. 논문은 이를 standard SGD로 학습했다고 설명한다.

### receptive field 제어와 속도 개선

논문은 dense segmentation에서 receptive field 크기를 명시적으로 제어하는 것도 중요하다고 본다. 원래 VGG-16의 첫 fully connected layer는 convolution으로 바꾸면 spatial size가 $7 \times 7$이 되어 계산량이 크고 receptive field도 매우 크다. 저자들은 이 층을 단순 decimation으로 $4 \times 4$ 또는 $3 \times 3$으로 줄여 계산 비용을 줄인다.

이 조정은 두 가지 효과를 낸다. 하나는 계산 속도 개선이고, 다른 하나는 receptive field의 재조정이다. 논문은 이후 실험에서 kernel size와 input stride를 조절해 different Field-Of-View (FOV)를 비교하며, 더 적은 파라미터와 더 빠른 속도로도 높은 성능이 가능함을 보인다.

### Fully connected CRF

DCNN 출력만으로는 경계가 흐리기 때문에, 저자들은 fully connected CRF를 사용한다. CRF의 energy function은 다음과 같다.

$$
E(x) = \sum_i \theta_i(x_i) + \sum_{ij} \theta_{ij}(x_i, x_j)
$$

여기서 $x_i$는 픽셀 $i$의 label이다. unary potential은 DCNN이 예측한 label probability에서 온다.

$$
\theta_i(x_i) = - \log P(x_i)
$$

즉, DCNN이 어떤 픽셀을 특정 class라고 높게 믿을수록 그 label의 unary cost는 낮아진다.

pairwise potential은 Potts model 기반이며, 모든 픽셀 쌍 $(i, j)$에 대해 정의된다.

$$
\theta_{ij}(x_i, x_j) = \mu(x_i, x_j) \sum_{m=1}^{K} w_m k_m(f_i, f_j)
$$

여기서 $\mu(x_i, x_j)$는 $x_i \neq x_j$일 때 1, 같으면 0이다. 즉, label이 다를 때만 penalty를 준다. kernel $k_m$은 픽셀 feature 사이의 Gaussian kernel이다. 논문에서 실제로 사용한 kernel은 다음 두 항의 합이다.

$$
w_1 \exp \left( - \frac{\|p_i - p_j\|^2}{2\sigma_\alpha^2} - \frac{\|I_i - I_j\|^2}{2\sigma_\beta^2} \right)
+
w_2 \exp \left( - \frac{\|p_i - p_j\|^2}{2\sigma_\gamma^2} \right)
$$

첫 번째 항은 position $p$와 color intensity $I$를 모두 사용하므로 bilateral term이다. 공간적으로 가깝고 색도 비슷한 픽셀은 같은 label을 갖도록 유도한다. 두 번째 항은 position만 사용하여 보다 일반적인 smoothness prior를 준다. $\sigma_\alpha$, $\sigma_\beta$, $\sigma_\gamma$는 각각 kernel scale을 조절하는 hyperparameter이다.

이 모델의 중요한 점은 fully connected라는 것이다. 즉, 이웃 픽셀만 연결하는 것이 아니라 이미지의 모든 픽셀 쌍을 고려한다. 일반적으로는 계산이 매우 비싸지만, 논문은 Krähenbühl & Koltun (2011)의 mean-field inference를 사용한다. mean-field approximation $b(x) = \prod_i b_i(x_i)$ 아래에서 message passing이 feature space에서의 Gaussian filtering 형태로 바뀌기 때문에, high-dimensional filtering 기법으로 빠르게 계산할 수 있다. 논문은 Pascal VOC 이미지에서 평균 0.5초 미만이라고 보고한다.

### Multi-scale prediction

논문은 추가로 multi-scale feature도 실험한다. 입력 이미지와 첫 네 개의 max-pooling layer 출력에 각각 작은 2-layer MLP를 붙인다. 구체적으로 첫 층은 128개의 $3 \times 3$ convolution, 둘째 층은 128개의 $1 \times 1$ convolution이다. 이렇게 얻은 feature map들을 main network의 마지막 layer feature map과 concatenate하여 softmax에 넣는다. 따라서 총 $5 \times 128 = 640$개의 추가 channel이 생긴다.

이 단계에서는 새로 추가된 가중치만 학습하고, 기존 network parameter는 Section 3에서 학습한 값을 유지한다. 저자들은 multi-scale feature가 localization을 개선하긴 하지만, fully connected CRF만큼 큰 효과는 아니라고 정리한다.

## 4. 실험 및 결과

실험은 주로 PASCAL VOC 2012 semantic segmentation benchmark에서 수행된다. 이 데이터셋은 20개 foreground class와 1개 background class로 이루어진다. 원래 split은 train 1,464장, val 1,449장, test 1,456장이며, Hariharan et al. (2011)의 extra annotation을 사용해 train set을 10,582장으로 확장한다. 성능 평가는 21개 클래스 평균 pixel Intersection-over-Union, 즉 mean IOU로 측정한다.

### 학습 설정

DCNN과 CRF는 joint end-to-end가 아니라 **piecewise training**으로 따로 학습한다. 먼저 VGG-16을 ImageNet pretrained initialization으로 가져와 pixel classification에 맞게 fine-tuning한다. mini-batch는 20 이미지, 초기 learning rate는 0.001이며 마지막 classifier layer는 0.01을 사용한다. 2000 iteration마다 learning rate를 0.1배로 줄인다. momentum은 0.9, weight decay는 0.0005이다.

CRF는 DCNN unary를 고정한 상태에서 hyperparameter를 validation subset 100장으로 cross-validation한다. 논문은 $w_2 = 3$, $\sigma_\gamma = 3$을 default로 두고, $w_1$, $\sigma_\alpha$, $\sigma_\beta$를 coarse-to-fine 방식으로 탐색했다고 쓴다. mean-field iteration 수는 모든 실험에서 10으로 고정한다.

### Validation set 결과

기본 DeepLab은 val set에서 mean IOU 59.80%를 기록한다. 여기에 fully connected CRF를 붙인 DeepLab-CRF는 63.74%로 약 4%p 향상된다. 논문은 이 향상이 상당히 크다고 강조한다. 단순히 noisy prediction을 조금 다듬는 수준이 아니라, 실제로 object boundary를 훨씬 더 정교하게 복원했기 때문이다.

multi-scale feature를 추가한 DeepLab-MSc는 61.30%를 기록하며, 기본 모델보다 약 1.5%p 개선된다. 여기에 CRF까지 더한 DeepLab-MSc-CRF는 65.21%가 된다. 즉, multi-scale feature도 효과가 있으나, 성능 향상의 주된 원천은 CRF라고 보는 것이 논문 내용과 일치한다.

이후 Field-Of-View를 키운 변형들이 더 좋아진다. DeepLab-CRF-7x7과 DeepLab-CRF-LargeFOV는 둘 다 67.64%를 기록한다. 가장 높은 val 성능은 DeepLab-MSc-CRF-LargeFOV의 68.70%이다.

### Field-Of-View 실험

Table 2는 정확도와 효율성의 trade-off를 잘 보여준다.

DeepLab-CRF-7x7은 첫 fully connected layer kernel size가 $7 \times 7$, input stride가 4이며 receptive field가 224이고, parameter 수는 134.3M이다. mean IOU는 67.64%지만 training speed는 1.44 img/sec로 느리다.

DeepLab-CRF는 kernel size $4 \times 4$, input stride 4, receptive field 128, parameter 65.1M이며 63.74%를 기록한다. 같은 $4 \times 4$ kernel이라도 input stride를 8로 키운 DeepLab-CRF-4x4는 receptive field가 224가 되며 67.14%로 올라간다. 이는 receptive field 확대가 중요함을 보여준다.

가장 실용적인 결과는 DeepLab-CRF-LargeFOV이다. kernel size $3 \times 3$, input stride 12, 마지막 두 layer channel 수를 4096에서 1024로 줄여 parameter를 20.5M까지 낮췄다. 그런데도 mean IOU는 67.64%로 DeepLab-CRF-7x7과 동일하며, training speed는 4.84 img/sec로 훨씬 빠르다. 즉, 큰 성능 손실 없이 더 효율적인 모델 구성이 가능함을 보인다.

### 경계 주변 성능

논문은 object boundary 근처에서의 정확도를 따로 측정한다. val set의 `void` label을 이용해 boundary 주변 narrow band, 즉 trimap을 만들고 그 안에서의 pixel accuracy와 mean IOU를 계산한다. Figure 5에 따르면 multi-scale feature와 CRF 모두 boundary 성능을 개선하고, 특히 DeepLab-MSc-CRF가 가장 좋다. 이는 제안 방법이 단순히 내부 영역을 잘 맞추는 것이 아니라, 실제로 경계 근처 localization을 개선한다는 주장과 연결된다.

### Test set 결과와 SOTA 비교

PASCAL VOC 2012 test set에서 DeepLab-CRF는 66.4%, DeepLab-MSc-CRF는 67.1% mean IOU를 기록한다. 이는 당시 다른 최신 방법들보다 높다. 비교 대상인 MSRA-CFM은 61.8%, FCN-8s는 62.2%, TTI-Zoomout-16은 64.4%이다.

large FOV를 적용한 DeepLab-CRF-7x7과 DeepLab-CRF-LargeFOV는 모두 70.3%를 달성한다. 최종 최고 성능은 DeepLab-MSc-CRF-LargeFOV의 71.6%이다. 논문 abstract에서도 이 수치를 test set state-of-the-art 결과로 강조한다.

클래스별 성능 표를 보면, 최종 모델은 특히 `aero`, `bike`, `boat`, `bus`, `car`, `cat`, `dog`, `horse`, `person`, `sheep` 등 여러 클래스에서 강한 성능을 보인다. 다만 클래스별 상세 원인 분석은 논문 본문에 충분히 제시되어 있지 않으므로, 특정 클래스에서 왜 좋아졌는지를 이 텍스트만으로 단정할 수는 없다.

### 속도와 재현성

논문은 dense DCNN이 modern GPU에서 약 8 fps로 동작한다고 주장하고, fully connected CRF mean-field inference는 평균 0.5초가 걸린다고 쓴다. 또한 Caffe 기반 구현과 코드, 설정 파일, 학습된 모델을 공개했다고 명시한다. 이는 당시 기준으로는 정확도뿐 아니라 실용성도 중요한 강점이었다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation에서 DCNN의 표현력과 CRF의 구조적 정제를 매우 설득력 있게 결합했다는 점이다. 저자들은 단순히 성능 숫자만 올린 것이 아니라, 왜 DCNN alone이 localization에 약한지, 왜 fully connected CRF가 그 약점을 메울 수 있는지를 개념적으로도 명확히 설명한다. 실험적으로도 CRF가 약 4%p 수준의 큰 향상을 주고, trimap 실험을 통해 경계 근처 개선까지 보였다는 점이 강하다.

두 번째 강점은 계산 효율을 함께 다뤘다는 점이다. atrous algorithm을 통해 dense score computation을 가능하게 하고, receptive field와 parameter 수를 조정해 large FOV를 더 효율적으로 구현했다. 특히 134.3M parameter 모델과 비슷한 성능을 20.5M parameter 모델로 달성한 것은 중요한 engineering contribution이다.

세 번째 강점은 구조의 단순성이다. 논문 스스로도 강조하듯, 시스템은 essentially DCNN + CRF의 두 모듈로 이루어진다. 각 구성 요소는 이미 알려진 기법이지만, 이 조합과 구체적 구현이 semantic segmentation에 매우 효과적이라는 점을 보여줬다. 이후 DeepLab 계열 연구의 출발점이 되었다는 점에서도 의미가 크다.

한계도 분명하다. 먼저, 이 논문은 DCNN과 CRF를 end-to-end로 joint training하지 않는다. piecewise training으로 unary를 고정한 뒤 CRF parameter를 따로 맞춘다. 저자들도 discussion에서 이를 향후 개선 과제로 인정한다. 따라서 전체 시스템이 완전히 통합된 최적화는 아니다.

또한 multi-scale feature의 효과는 존재하지만 CRF에 비해 상대적으로 제한적이다. 즉, 내부 feature hierarchy를 더 깊게 활용하는 방식 자체는 초기 단계에 머물러 있다. boundary refinement의 많은 부분을 여전히 post-processing 성격의 CRF에 의존한다고 볼 수 있다.

마지막으로, failure case가 Figure 7 마지막 세 줄에 존재한다고 저자들이 직접 언급하지만, 이 텍스트에는 그 실패 유형을 자세히 분석한 설명은 없다. 따라서 어떤 경우에 system이 체계적으로 실패하는지, 예를 들어 thin structure, occlusion, rare pose, similar appearance confusion 등으로 일반화해서 말하는 것은 이 자료만으로는 적절하지 않다. 논문이 보여주는 것은 성능 향상과 대표 qualitative result이지, 실패 원인의 깊은 해부는 아니다.

## 6. 결론

이 논문은 semantic segmentation에서 초기 DeepLab 프레임워크를 제시하며, 세 가지 핵심 기여를 한다. 첫째, VGG-16을 atrous convolution 기반 dense predictor로 효율적으로 변환했다. 둘째, DCNN의 coarse한 score map을 unary로 사용하고 fully connected CRF로 정제하여 정밀한 object boundary를 복원했다. 셋째, PASCAL VOC 2012에서 당시 state-of-the-art 성능인 71.6% mean IOU를 달성했다.

이 연구의 중요성은 단순한 benchmark 향상에 그치지 않는다. 이후 semantic segmentation 연구에서 atrous convolution, large field-of-view, DCNN+CRF 조합은 매우 큰 흐름을 형성했다. 특히 “recognition은 CNN, localization refinement는 structured model”이라는 관점은 이후 end-to-end structured prediction, CRF-RNN, 그리고 더 발전된 DeepLab 계열로 이어지는 중요한 연결고리가 되었다. 따라서 이 논문은 semantic segmentation의 현대적 설계를 정립한 대표적인 전환점 중 하나로 볼 수 있다.
