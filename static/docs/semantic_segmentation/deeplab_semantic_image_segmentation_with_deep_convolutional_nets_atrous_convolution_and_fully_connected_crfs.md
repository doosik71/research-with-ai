# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

- **저자**: Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1606.00915

## 1. 논문 개요

이 논문은 semantic image segmentation, 즉 이미지의 모든 픽셀에 semantic class label을 할당하는 문제를 다룬다. 저자들은 기존 image classification용 DCNN을 segmentation에 그대로 쓰면 성능이 제한되는 세 가지 핵심 이유를 짚는다. 첫째, 반복적인 max-pooling과 striding 때문에 feature map의 해상도가 크게 줄어든다. 둘째, 실제 이미지에는 다양한 크기의 객체가 섞여 있으므로 single-scale 처리만으로는 충분하지 않다. 셋째, classification에 유리한 spatial invariance가 segmentation에서는 object boundary를 흐리게 만들어 localization accuracy를 떨어뜨린다.

논문의 목표는 이 세 문제를 하나의 실용적인 시스템으로 해결하는 것이다. 이를 위해 제안된 DeepLab은 크게 세 가지 축으로 구성된다. 첫째, **atrous convolution**으로 feature resolution을 높이면서 receptive field도 넓힌다. 둘째, **ASPP (Atrous Spatial Pyramid Pooling)** 로 multi-scale context를 포착한다. 셋째, **fully connected CRF**를 후처리로 붙여 object boundary를 정교하게 복원한다.

이 문제가 중요한 이유는 semantic segmentation이 자율주행, 장면 이해, 로보틱스, 의료 영상 등에서 핵심 역할을 하기 때문이다. 단순히 물체가 있는지만 아는 것이 아니라 정확히 어디까지가 물체인지 알아야 하는 응용에서는 boundary quality와 dense prediction accuracy가 매우 중요하다. 이 논문은 이후 segmentation 계열 연구에서 사실상 표준 구성요소가 된 atrous convolution과 ASPP의 대표적 출발점이라는 점에서도 중요하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 classification용 backbone을 segmentation에 맞게 “덜 다운샘플링된 dense predictor”로 바꾸고, 그 위에 multi-scale context와 structured refinement를 결합하는 것이다. 저자들은 deconvolution layer를 새로 학습하는 복잡한 경로 대신, convolution filter 자체를 듬성듬성 적용하는 **atrous convolution**을 사용해 더 조밀한 feature map을 얻는다. 이 방식은 parameter 수나 각 위치당 연산량을 늘리지 않고도 receptive field를 키울 수 있다는 점이 핵심이다.

기존 접근과의 차별점은 세 가지로 정리된다. 첫째, 단순한 fully convolutionalization을 넘어서 output stride를 직접 제어한다. 둘째, multi-scale 처리를 위해 여러 크기의 입력 이미지를 모두 통과시키는 비싼 방식 대신, 하나의 feature layer 위에서 서로 다른 atrous rate를 병렬로 적용하는 ASPP를 제안한다. 셋째, boundary refinement를 위해 short-range CRF나 superpixel 기반 후처리 대신, 모든 픽셀 쌍을 연결하는 fully connected CRF를 사용해 long-range dependency와 edge-aware smoothing을 동시에 수행한다.

즉, 이 논문은 “고해상도 dense feature extraction + multi-scale context aggregation + boundary-aware structured prediction”을 하나의 간단한 cascade로 묶은 것이 핵심이다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 먼저 VGG-16 또는 ResNet-101 같은 image classification network를 fully convolutional network로 바꾼다. 그 다음 마지막 몇 개의 downsampling 단계의 stride를 제거하고, 뒤따르는 convolution들을 atrous convolution으로 치환해 더 촘촘한 feature response를 계산한다. 이 score map은 원본 해상도보다 보통 8배 작은 해상도에서 계산되며, 이후 bilinear interpolation으로 원본 크기로 업샘플링된다. 마지막으로 fully connected CRF가 이 coarse prediction을 refinement하여 경계를 다듬는다.

### Atrous Convolution

논문은 1차원 atrous convolution을 다음과 같이 정의한다.

$$
y[i] = \sum_{k=1}^{K} x[i + r \cdot k] w[k]
$$

여기서 $r$은 **rate**이며, 표준 convolution은 $r=1$인 특수한 경우다. 직관적으로는 filter tap 사이에 구멍(hole)을 넣는 방식이다. 그래서 같은 개수의 filter parameter를 유지하면서도 더 넓은 영역을 본다.

논문은 atrous convolution의 중요한 효과를 두 가지로 설명한다.

첫째, **dense feature extraction**이다. 원래 분류용 네트워크는 보통 output stride가 32인데, 마지막 pooling 또는 stride convolution의 stride를 1로 바꾸고 이후 layer에 atrous convolution을 적용하면 output stride를 8까지 줄일 수 있다. DeepLab은 전체를 원본 해상도로 계산하지는 않고, atrous convolution으로 feature density를 높인 뒤 bilinear interpolation을 붙이는 hybrid 전략을 사용한다. 이는 accuracy와 efficiency 사이의 절충이다.

둘째, **field-of-view enlargement**다. $k \times k$ filter에 rate $r$을 쓰면 effective kernel size는

$$
k_e = k + (k - 1)(r - 1)
$$

가 된다. 따라서 큰 kernel을 직접 쓰지 않고도 더 큰 context를 볼 수 있다. 예를 들어 VGG-16의 `fc6`를 $3 \times 3$ kernel과 $r=12$로 구성해 큰 receptive field를 확보한 DeepLab-LargeFOV variant를 만든다.

### ASPP: Atrous Spatial Pyramid Pooling

논문은 object scale variation을 처리하기 위해 두 전략을 비교한다. 하나는 multi-scale input processing이다. 입력 이미지를 여러 scale로 리사이즈해 각각 DCNN에 통과시킨 뒤 score map을 upsample하고 position-wise max fusion을 한다. 성능은 좋아지지만 비용이 크다.

이를 대신하는 것이 ASPP다. ASPP는 하나의 feature layer 위에 여러 개의 parallel atrous convolution branch를 놓고, 서로 다른 rate를 적용한다. 예를 들어 VGG-16 기반 실험에서는 rate가 $\{6, 12, 18, 24\}$ 인 branch들을 병렬로 둔다. 이렇게 하면 각 branch는 서로 다른 effective field-of-view를 가지게 되고, 결과적으로 작은 객체와 큰 객체, 그리고 주변 context를 동시에 포착할 수 있다.

중요한 점은 ASPP가 원래 spatial pyramid pooling의 “여러 scale의 정보를 모은다”는 발상을 feature resampling 대신 atrous convolution rate 변화로 구현했다는 것이다. 즉, 연산 효율을 유지하면서 multi-scale representation을 만든다.

### Fully Connected CRF

DCNN의 coarse score map은 object 존재와 대략적 위치는 잘 잡지만, 경계가 흐리고 얇은 구조에 약하다. 이를 보완하기 위해 논문은 fully connected CRF를 후처리로 사용한다. CRF energy는 다음과 같다.

$$
E(x) = \sum_i \theta_i(x_i) + \sum_{ij} \theta_{ij}(x_i, x_j)
$$

여기서 unary term은 DCNN이 준 class probability로부터 온다.

$$
\theta_i(x_i) = -\log P(x_i)
$$

pairwise term은 다음과 같다.

$$
\theta_{ij}(x_i, x_j) =
\mu(x_i, x_j)
\left[
w_1 \exp \left(
-\frac{||p_i-p_j||^2}{2\sigma_\alpha^2}
-\frac{||I_i-I_j||^2}{2\sigma_\beta^2}
\right)
+
w_2 \exp \left(
-\frac{||p_i-p_j||^2}{2\sigma_\gamma^2}
\right)
\right]
$$

여기서 $\mu(x_i, x_j)$는 Potts model 형태로, $x_i \neq x_j$일 때 1이고 같으면 0이다. 첫 번째 Gaussian kernel은 위치 $p$와 색상 $I$를 함께 고려하는 **bilateral kernel**이고, 두 번째는 위치만 고려하는 spatial kernel이다. 전자는 색이 비슷하고 위치가 가까운 픽셀끼리 같은 label을 갖도록 유도하고, 후자는 일반적인 smoothness를 준다.

이 모델의 장점은 fully connected graph인데도 mean-field approximation 하에서 inference를 빠르게 할 수 있다는 점이다. 논문은 Pascal VOC 이미지에서 평균 0.5초 이하의 CPU 시간으로 수행된다고 보고한다. 학습 측면에서는 DCNN과 CRF를 end-to-end로 함께 학습하지 않고, 먼저 DCNN을 학습한 뒤 unary를 고정하고 CRF hyperparameter를 cross-validation으로 맞춘다. 즉, 이 논문 버전의 CRF는 **post-processing module**이다.

### 학습 절차

학습은 ImageNet pretrained VGG-16 또는 ResNet-101을 segmentation에 fine-tuning하는 방식이다. 마지막 classifier를 task-specific class 수에 맞게 바꾸고, loss는 output map의 각 spatial position에 대한 cross-entropy 합이다. labeled pixel은 동일 가중치로 취급하고 unlabeled pixel은 무시한다. target label은 output stride 8에 맞춰 subsample된다. 최적화는 SGD를 사용한다.

논문은 learning rate policy도 비교한다. VGG-16 LargeFOV에서 기존 step schedule보다 polynomial decay, 즉 learning rate를 $(1 - \frac{iter}{iter_{max}})^{power}$ 꼴로 줄이는 **poly policy**가 더 낫다고 보고한다. 이것은 이후 segmentation 연구에서 널리 채택되는 설정이다.

## 4. 실험 및 결과

논문은 네 개의 데이터셋에서 실험한다. PASCAL VOC 2012, PASCAL-Context, PASCAL-Person-Part, Cityscapes다. 기본 평가지표는 mean Intersection-over-Union, 즉 mIOU다.

### PASCAL VOC 2012

이 데이터셋은 20개 foreground class와 1개 background class로 구성된다. train 1,464장, val 1,449장, test 1,456장이며, Hariharan 등이 제공한 extra annotation을 포함하면 `trainaug`는 10,582장이다.

초기 conference version에서 VGG-16 기반 DeepLab-LargeFOV는 `fc6`의 field-of-view를 조절하면서 성능과 속도를 비교했다. 원래 VGG-16을 직접 옮긴 $7 \times 7$, rate 4 설정은 CRF 후 67.64% mIOU였지만 느렸다. 반면 $3 \times 3$, rate 12 설정의 DeepLab-LargeFOV는 비슷한 성능을 유지하면서 훨씬 적은 parameter와 더 빠른 학습 속도를 보였다. 즉, 큰 kernel을 직접 쓰는 대신 atrous rate를 키우는 것이 효율적이라는 점을 실험으로 보여준다. 또한 CRF는 모든 variant에서 약 3~5% absolute gain을 제공했다.

이후 개선된 버전에서는 세 가지 변화가 있었다. 첫째, poly learning policy를 도입해 VGG-16 LargeFOV의 val 성능을 step policy 대비 크게 끌어올렸다. 둘째, ASPP를 도입했다. VGG-16 기반에서 baseline LargeFOV는 CRF 전후로 65.76% / 69.84%, ASPP-S는 66.98% / 69.73%, ASPP-L은 68.96% / 71.57%였다. 작은 rate 집합보다 큰 rate 집합의 ASPP-L이 더 효과적이었다. 셋째, backbone을 ResNet-101로 교체하고, multi-scale inputs, MS-COCO pretraining, random scaling augmentation을 추가했다.

ResNet-101 기반 실험에서 단순 모델만으로도 68.72%를 얻어 VGG-16 LargeFOV 65.76%보다 명확히 좋았다. 여기에 multi-scale fusion을 넣으면 71.27%, COCO pretraining까지 하면 73.28%, augmentation까지 하면 74.87%, LargeFOV와 ASPP를 더하면 76%대 중반으로 올라간다. 최종적으로 CRF 후 val에서 77.69%, test에서 **79.7% mIOU**를 기록했다. 논문 시점 기준 leaderboard 최고 성능이었다.

또한 boundary-focused trimap 분석에서 ResNet-101은 VGG-16보다 경계 부근 정확도가 더 높았고, 저자들은 residual identity mapping이 intermediate feature 활용 측면에서 hyper-column과 유사한 효과를 줄 수 있다고 해석한다. 다만 이것은 논문의 정식 증명이 아니라 저자들의 해석이다.

### PASCAL-Context

이 데이터셋은 object뿐 아니라 stuff를 포함하는 dense scene labeling 벤치마크다. 59개 frequent class와 background 1개를 사용하고, train 4,998장, val 5,105장이다.

VGG-16 LargeFOV는 CRF 전 37.6%, 후 39.6%였다. ResNet-101 기반 DeepLab은 39.6%로 시작해 multi-scale과 COCO pretraining, augmentation, ASPP, CRF를 차례로 더하면서 최종 **45.7% mIOU**를 기록했다. 이는 표에 제시된 다른 강한 baseline들인 FCN-8s, CRF-RNN, ParseNet, BoxSup, Context, VeryDeep보다 높다. 저자들은 특히 non-linear pairwise term 없이도 당시 state-of-the-art를 넘어섰다고 강조한다.

### PASCAL-Person-Part

이 실험은 object-level segmentation이 아니라 person의 semantic part segmentation이다. 데이터는 PASCAL VOC 2010의 person annotation을 사용하며, head, torso, upper/lower arms, upper/lower legs, background로 총 7개 class를 구성한다. 사람을 포함한 이미지 1,716장을 train, 1,817장을 val에 사용한다.

ResNet-101 기반 DeepLab alone은 58.90%로, 기존 VGG-16 기반 DeepLab-Attention 56.39%보다 높다. multi-scale input을 넣으면 63.10%, COCO pretraining으로 64.40%까지 올라간다. 흥미롭게도 이 데이터셋에서는 LargeFOV나 ASPP가 오히려 도움이 되지 않았다. 표에서는 해당 옵션 적용 시 62%대로 내려간다. 즉, 큰 context가 항상 part segmentation에 유리한 것은 아니라는 점을 보여준다. 최종적으로 dense CRF를 붙인 모델은 **64.94% mIOU**를 기록했고, concurrent work인 Graph LSTM의 60.16%보다 높았다.

### Cityscapes

Cityscapes는 50개 도시의 street scene에서 수집한 고해상도 segmentation dataset이며, train 2,975장, val 500장, test 1,525장이다. 평가는 19개 semantic label에 대해 수행된다.

이 데이터셋은 해상도가 $2048 \times 1024$로 매우 높아서 GPU memory가 큰 제약이다. 논문은 처음에는 이미지를 2배 downsample해서 실험했지만, full resolution을 쓰는 것이 val 성능을 1.8~1.9% 정도 개선한다고 보고한다. inference 시에는 이미지를 overlapped region으로 나누어 처리한다.

VGG-16 기반보다 ResNet-101 기반이 확실히 좋고, multi-scale input은 메모리 제약으로 사용하지 않았다. 대신 deeper network, augmentation, LargeFOV, ASPP, CRF의 효과를 본다. ResNet-101 alone은 66.6%, LargeFOV를 더하면 69.2%, ASPP까지 하면 70.4%, augmentation과 CRF를 포함한 최종 val 성능은 **71.4%**다. test set에서는 **70.4%**를 보고한다. 당시 다른 strong method들과 비교해 상위권 성능이다.

### Failure Modes

논문은 실패 사례도 직접 제시한다. bicycle, chair 같은 얇고 복잡한 구조의 boundary는 여전히 잘 복원하지 못한다. CRF도 unary가 충분히 confident하지 않으면 세밀한 구조를 복구하지 못한다. 저자들은 encoder-decoder 구조가 이런 문제를 완화할 수 있을 것이라고 언급하지만, 이는 미래 과제로 남겨둔다. 따라서 이 논문이 boundary를 크게 개선했지만, 고해상도 세부 구조를 완전히 해결한 것은 아니다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation의 핵심 병목을 아주 명확한 모듈 설계로 해결했다는 점이다. atrous convolution은 해상도 보존과 receptive field 확장을 동시에 만족시키며, ASPP는 multi-scale context aggregation을 효율적으로 구현한다. fully connected CRF는 coarse DCNN output을 실제 usable한 segmentation map으로 다듬는 데 매우 효과적이다. 논문은 이 세 요소를 개별 아이디어 수준이 아니라 실제 benchmark 성능 개선으로 연결해 보였다.

또 다른 강점은 설계의 실용성이다. 논문은 “복잡한 새 네트워크를 처음부터 만들기”보다 기존 ImageNet backbone을 재활용하는 방향을 택했다. bilinear interpolation을 사용하고 CRF를 후처리로 두는 점도 당시 기준으로 구현 부담이 낮았다. 속도, 정확도, 단순성의 균형을 분명히 의식한 논문이다.

실험적으로도 설득력이 강하다. 단순히 한 데이터셋에서만 잘 되는 것이 아니라 PASCAL VOC 2012, PASCAL-Context, PASCAL-Person-Part, Cityscapes에서 일관되게 강한 성능을 보였다. 또한 field-of-view, learning rate policy, ASPP 설정, backbone 교체, multi-scale fusion, COCO pretraining, augmentation 등 각 요소의 기여를 분리해서 보여준다.

한계도 분명하다. 첫째, CRF는 end-to-end로 joint training되지 않고 post-processing으로 쓰인다. 따라서 DCNN unary와 CRF pairwise가 완전히 통합된 최적화는 아니다. 둘째, 얇은 구조나 세밀한 boundary는 여전히 어렵다. 논문 스스로 bicycle, chair 등에서 failure case를 인정한다. 셋째, multi-scale input fusion은 효과적이지만 계산 비용이 크다. ASPP가 대안이긴 하지만 모든 데이터셋에서 항상 이득을 주는 것은 아니며, person-part 실험에서는 오히려 성능이 떨어졌다. 넷째, 논문은 ResNet이 boundary localization을 더 잘하는 이유를 identity mapping과 hyper-column 유사 효과로 해석하지만, 이는 엄밀히 분석된 결론이라기보다 관찰 기반의 추정이다.

비판적으로 보면, DeepLab은 당시 강력한 해법이지만 encoder-decoder 기반의 explicit high-resolution reconstruction 경로를 갖지 않는다. 그래서 coarse feature에서 시작한 뒤 interpolation과 CRF로 보정하는 방식의 한계가 남아 있다. 이후 연구에서 skip connection, decoder, feature fusion이 강화된 이유도 여기에 있다.

## 6. 결론

이 논문은 semantic segmentation을 위해 image classification DCNN을 효과적으로 재구성하는 방법을 제시했다. 핵심 기여는 세 가지다. 첫째, **atrous convolution**으로 resolution을 높이면서 receptive field를 키우는 dense feature extraction 방식을 정립했다. 둘째, **ASPP**를 통해 multi-scale object와 context를 효율적으로 다루는 구조를 제안했다. 셋째, **fully connected CRF**를 결합해 object boundary localization을 크게 향상시켰다.

실험적으로 DeepLab은 여러 대표 데이터셋에서 당시 state-of-the-art를 달성하거나 갱신했다. 특히 PASCAL VOC 2012 test에서 79.7% mIOU를 기록한 결과는 이 접근이 단순한 아이디어 제안이 아니라 실제 경쟁력 있는 시스템임을 보여준다.

실제 적용과 향후 연구 측면에서도 이 논문은 매우 중요하다. atrous convolution과 ASPP는 이후 semantic segmentation뿐 아니라 detection, instance segmentation 등 다양한 dense prediction 문제로 확장되었다. 동시에 이 논문이 남긴 한계, 예를 들어 얇은 구조 복원과 high-resolution detail 문제는 이후 encoder-decoder 구조, skip fusion, end-to-end structured prediction 연구로 이어졌다. 즉, DeepLab은 하나의 강력한 방법이면서 동시에 이후 segmentation 연구의 기준점을 만든 논문이라고 평가할 수 있다.
