# Fully Convolutional Networks for Semantic Segmentation

- **저자**: Jonathan Long, Evan Shelhamer, Trevor Darrell
- **발표연도**: 2015
- **arXiv**: https://arxiv.org/abs/1411.4038

## 1. 논문 개요

이 논문은 semantic segmentation, 즉 이미지의 모든 픽셀에 대해 클래스 라벨을 예측하는 문제를 다룬다. 저자들의 핵심 목표는 기존 image classification용 convolutional network를 픽셀 단위 예측이 가능한 구조로 바꾸고, 이를 end-to-end로 학습하여 별도의 복잡한 후처리 없이도 높은 성능을 내는 것이다. 논문 제목의 Fully Convolutional Network, 즉 FCN은 입력 크기가 고정되지 않아도 되고, 입력 이미지 전체를 한 번에 처리해서 그에 대응되는 spatial output map을 만든다는 점이 핵심이다.

이 연구가 다루는 문제는 당시 semantic segmentation 연구의 중요한 병목과 직결되어 있었다. 기존 방법들은 patchwise training, superpixel, proposal, random field 기반 refinement 같은 복잡한 보조 절차에 크게 의존했다. 반면 semantic segmentation은 본질적으로 “무엇인지(what)”를 알아내는 semantic 정보와 “어디에 있는지(where)”를 알아내는 spatial 정보가 동시에 필요하다. 저자들은 classification network가 이미 풍부한 semantic hierarchy를 학습하고 있다는 점에 주목하여, 이를 dense prediction 문제로 자연스럽게 확장할 수 있다고 본다.

논문이 중요한 이유는 두 가지다. 첫째, classification backbone을 segmentation에 직접 이식하는 단순하고 일반적인 방법론을 제시했다. 둘째, coarse한 high-level feature와 fine한 low-level feature를 결합하는 skip architecture를 통해 semantic segmentation에서 정확도와 세부 경계 복원을 동시에 개선했다. 이 논문은 이후의 encoder-decoder, feature pyramid, U-Net류 구조에 큰 영향을 준 출발점으로 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 기존 classification convnet을 “fully convolutional”하게 재해석하면, 임의 크기의 입력에 대해 spatially dense한 출력을 만들 수 있다는 것이다. 원래 classification network의 fully connected layer는 고정 크기 입력을 전제로 하지만, 저자들은 이를 입력 전체를 덮는 큰 convolution으로 해석한다. 그러면 원래는 하나의 class score vector만 내던 네트워크가, 이제는 여러 위치에 대해 class score map을 출력하는 구조가 된다.

하지만 이렇게 얻은 출력은 stride가 커서 매우 coarse하다. 예를 들어 마지막 prediction layer의 stride가 32라면, 업샘플링을 하더라도 경계가 뭉개지고 작은 구조를 복원하기 어렵다. 이를 해결하기 위해 저자들은 깊은 층의 semantic information과 얕은 층의 fine appearance information을 결합하는 skip architecture를 제안한다. 즉, 깊은 층은 “무엇인지”를 잘 알지만 위치가 거칠고, 얕은 층은 “어디인지”를 더 잘 보존한다는 점을 이용한다.

기존 접근과의 차별점은 분명하다. 논문에서 직접 비교하듯이, 이전 방법들은 작은 모델, patchwise training, superpixel이나 CRF류 후처리, multi-scale pyramid, shift-and-stitch 같은 복잡한 구성요소를 많이 썼다. 반면 이 논문은 분류용 대형 네트워크를 segmentation용 FCN으로 바꾼 뒤, whole-image training과 in-network upsampling, skip connection만으로 state-of-the-art를 달성했다. 저자들은 이를 “더 단순하면서도 더 빠르고 더 정확한” 방향으로 제시한다.

## 3. 상세 방법 설명

논문은 먼저 convolutional layer의 일반적 형식을 정리한다. 어떤 층의 위치 $(i,j)$에서 출력 $y_{ij}$는 주변 입력 영역의 함수로 계산된다.

$$
y_{ij} = f_{ks}\left(\{x_{si+\delta i,\, sj+\delta j}\}_{0 \le \delta i,\delta j \le k}\right)
$$

여기서 $k$는 kernel size, $s$는 stride이다. convolution, pooling, activation 같은 연산이 모두 이런 local operator의 조합으로 표현된다는 점을 강조한다. 이런 형태의 층만으로 이루어진 네트워크는 입력 크기에 상관없이 대응되는 출력 지도를 만들 수 있으며, 저자들은 이를 fully convolutional network라고 부른다.

또한 loss가 출력 spatial location들에 대한 합으로 표현될 때,

$$
\ell(x;\theta) = \sum_{ij} \ell'(x_{ij};\theta)
$$

whole-image 단위로 계산한 gradient는 결국 각 spatial 위치의 loss gradient를 모두 더한 것과 같다. 따라서 patch를 하나씩 따로 처리하는 대신 이미지 전체를 한 번에 convolution으로 처리하면, receptive field들이 많이 겹치는 상황에서 훨씬 효율적으로 학습할 수 있다. 이것이 patchwise training보다 whole-image fully convolutional training이 계산적으로 유리한 이유다.

### 3.1 분류 네트워크를 dense predictor로 바꾸는 방식

저자들은 AlexNet, VGG16, GoogLeNet 같은 대표적인 classification network를 segmentation용으로 바꾼다. 방법은 간단하다.

첫째, 마지막 classifier layer를 제거한다.  
둘째, fully connected layers를 convolution layers로 바꾼다.  
셋째, 클래스 수만큼 채널을 갖는 $1 \times 1$ convolution을 붙여 각 spatial 위치에서 class score를 출력하게 한다. PASCAL VOC에서는 background를 포함해 21개 클래스이므로 채널 수 21의 $1 \times 1$ convolution을 사용한다.  
넷째, coarse한 출력을 deconvolution layer로 bilinear upsampling하여 원본 이미지 크기로 복원한다.

이렇게 바꾼 기본 모델이 FCN-32s다. 이름의 32는 최종 prediction이 원본보다 32배 coarse한 stride를 가진다는 뜻이다. 이 모델만으로도 VGG16 기반 FCN은 validation mean IU 56.0을 기록해 당시 strong baseline을 넘어서는 수준을 보였다.

### 3.2 Shift-and-stitch와 filter rarefaction

논문은 OverFeat에서 사용된 shift-and-stitch도 설명한다. 출력 stride가 $f$이면, 입력을 $(x,y)$만큼 여러 번 shift하여 총 $f^2$번 네트워크를 돌리고, 나온 coarse output들을 interlace하면 더 dense한 prediction을 얻을 수 있다. 저자들은 이것이 결국 filter rarefaction, 즉 stride를 줄인 대신 filter 내부를 띄엄띄엄 채운 형태의 필터와 동등하다고 해석한다.

필터 확장은 다음과 같이 표현된다.

$$
f'_{ij} =
\begin{cases}
f_{i/s,\,j/s} & \text{if } s \text{ divides both } i \text{ and } j \\
0 & \text{otherwise}
\end{cases}
$$

하지만 저자들은 이 방식이 계산 효율과 효과 면에서 skip connection과 learned upsampling보다 낫지 않다고 보고, 최종 모델에는 사용하지 않는다.

### 3.3 Upsampling as backwards strided convolution

coarse output을 dense pixel prediction으로 연결하기 위해 논문은 deconvolution을 사용한다. 저자들의 설명에 따르면 upsampling factor $f$는 일종의 fractional stride convolution처럼 볼 수 있고, 이를 backwards convolution, 즉 deconvolution으로 구현할 수 있다. 중요한 점은 이 연산이 네트워크 안에서 differentiable하므로 pixelwise loss로부터 end-to-end 학습이 가능하다는 것이다.

최종 업샘플링 필터는 bilinear interpolation으로 고정할 수도 있지만, 중간 upsampling layer는 bilinear로 초기화한 뒤 학습되도록 둔다. 즉, 단순한 고정 보간이 아니라 데이터에 맞는 업샘플링 방식을 네트워크가 배울 수 있게 만든다.

### 3.4 Skip architecture: FCN-32s, FCN-16s, FCN-8s

이 논문의 가장 중요한 구조적 기여는 skip architecture다. FCN-32s는 최종 깊은 층의 prediction만 사용하므로 semantic 정보는 강하지만 공간 해상도가 낮다. 이를 보완하기 위해 더 얕은 층의 정보를 결합한다.

FCN-16s는 `conv7`에서 얻은 stride 32 prediction을 $2\times$ upsample한 뒤, `pool4`에서 만든 stride 16 prediction과 더한다. `pool4` 위에도 별도의 $1\times1$ convolution을 올려 class score를 만든다. 이후 두 출력을 합친 결과를 다시 원본 크기로 upsample한다. 새로 추가된 `pool4` 경로의 파라미터는 처음에는 0으로 초기화하여, 기존 FCN-32s의 예측을 망가뜨리지 않은 상태에서 시작한다.

FCN-8s는 여기에 한 단계 더 나아가, `pool3`의 stride 8 prediction까지 결합한다. 즉, `conv7`의 coarse semantic 정보, `pool4`의 중간 해상도 정보, `pool3`의 finer location 정보를 함께 사용한다. 논문은 이를 통해 예측 경계와 세부 구조가 개선됨을 Figure 4로 보여준다.

실제로 성능도 점진적으로 올라간다. subset validation에서 FCN-32s는 mean IU 59.4, FCN-16s는 62.4, FCN-8s는 62.7을 기록했다. 개선 폭 자체는 마지막 단계에서 작지만, 시각적으로는 더 부드럽고 세밀한 segmentation을 보였다고 설명한다.

### 3.5 학습 설정

학습은 SGD with momentum으로 수행한다. minibatch는 20 images이고, learning rate는 모델별로 다르게 선택했다. AlexNet은 $10^{-3}$, VGG16은 $10^{-4}$, GoogLeNet은 $5\times10^{-5}$를 사용했다. momentum은 $0.9$, weight decay는 $5\times10^{-4}$ 또는 $2\times10^{-4}$다. bias에 대해서는 learning rate를 두 배로 주었다.

loss는 per-pixel multinomial logistic loss를 사용한다. ground truth에서 ambiguous 혹은 difficult로 마스킹된 픽셀은 학습에서 제외한다. 또한 class scoring convolution layer는 0으로 초기화했으며, 저자들은 random initialization이 더 빠른 수렴이나 더 나은 성능을 주지 않았다고 보고한다. dropout은 원래 classifier net에 있던 위치에 그대로 포함했다.

논문은 fine-tuning의 중요성도 강하게 보여준다. 마지막 classifier만 학습하는 경우는 전체 fine-tuning 성능의 약 70% 수준에 그쳤다. 반면 전체 네트워크를 backpropagation으로 fine-tuning해야 충분한 성능이 나왔다. 저자들은 base classification net을 처음부터 다시 학습하는 것은 시간상 현실적이지 않다고 명시한다.

### 3.6 Patchwise training과 whole-image training 비교

논문은 patch sampling의 이점도 직접 실험한다. 기존 연구들은 class imbalance나 spatial correlation 문제를 줄이기 위해 patch를 랜덤 샘플링해 학습하는 경우가 많았다. 저자들은 whole-image training이 사실상 이미지 내 겹치는 큰 patch들을 한 번에 처리하는 것과 같다고 본다. 또 loss의 spatial term 일부만 랜덤하게 무시하면 patch sampling과 유사한 효과를 낼 수 있다고 설명한다.

하지만 실험 결과, sampling은 convergence를 의미 있게 개선하지 못했고, 오히려 batch당 더 많은 이미지를 처리해야 해서 wall-clock time은 더 오래 걸렸다. 따라서 논문은 whole-image unsampled training이 더 효율적이라고 결론 내린다.

## 4. 실험 및 결과

논문은 PASCAL VOC, NYUDv2, SIFT Flow에서 평가한다. 평가 지표는 pixel accuracy, mean accuracy, mean IU, frequency weighted IU 네 가지다. 혼동행렬 기반으로 정의되며, $n_{ij}$를 클래스 $i$가 클래스 $j$로 예측된 픽셀 수, $t_i=\sum_j n_{ij}$를 클래스 $i$의 전체 픽셀 수라고 할 때, mean IU는 다음과 같다.

$$
\text{mean IU} = \frac{1}{n_{cl}} \sum_i \frac{n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}
$$

frequency weighted IU는 다음과 같다.

$$
\left(\sum_k t_k\right)^{-1}\sum_i \frac{t_i n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}
$$

논문은 특히 mean IU를 주요 지표로 사용한다. 다만 Appendix A에서 저자들 스스로 mean IU가 fine-scale accuracy를 충분히 반영하지 못한다고 지적한다. 예를 들어 coarse prediction만으로도 mean IU가 꽤 높을 수 있으며, pixel-perfect prediction이 아니어도 state-of-the-art를 넘을 수 있다는 것이다.

### 4.1 PASCAL VOC

가장 핵심적인 결과는 PASCAL VOC 2011/2012 test에서 나온다. 이전 강력한 방법인 SDS는 VOC2012 test에서 mean IU 51.6, VOC2011 test에서 52.6을 기록했다. 반면 FCN-8s는 VOC2012 test에서 62.2, VOC2011 test에서 62.7을 기록했다. 이는 논문이 주장하듯 이전 state-of-the-art 대비 약 20% relative improvement다.

추론 속도 차이도 매우 크다. SDS는 약 50초가 걸리는 반면, FCN-8s는 약 175ms다. 논문은 이를 convnet only 기준 114배, 전체 기준 286배 빠르다고 설명한다. 단순히 정확도만 오른 것이 아니라 실제 사용 측면에서도 큰 이점이 있다는 뜻이다.

기본 backbone 비교에서도 VGG16이 가장 강했다. PASCAL VOC 2011 validation 기준으로 FCN-AlexNet은 mean IU 39.8, FCN-VGG16은 56.0, FCN-GoogLeNet은 42.5였다. classification 정확도와 segmentation 성능이 반드시 같은 비율로 대응하지 않음을 보여주는 결과이기도 하다. 논문은 특히 자체 구현한 GoogLeNet이 classification에서는 나쁘지 않았지만 segmentation에서는 VGG16만큼 좋은 결과를 내지 못했다고 말한다.

### 4.2 Skip architecture의 효과

skip connection의 효과는 매우 직접적으로 검증된다. subset validation 기준으로 FCN-32s-fixed는 mean IU 45.4, FCN-32s는 59.4, FCN-16s는 62.4, FCN-8s는 62.7이다. 즉, classifier layer만 학습하는 것보다 전체 fine-tuning이 크게 중요했고, 거기에 `pool4`, `pool3`의 finer feature를 결합하면서 점진적으로 성능이 상승했다.

논문은 시각적 예시도 함께 제시한다. FCN-32s는 물체의 대략적 위치와 클래스는 잘 잡지만 경계가 거칠다. FCN-16s와 FCN-8s로 갈수록 fine structure가 살아나고, 인접한 객체 간 경계를 더 잘 나누며, 가려짐(occlusion) 상황에서도 비교적 안정적인 segmentation을 보인다고 설명한다.

### 4.3 NYUDv2

NYUDv2는 RGB-D indoor scene dataset으로 40-class semantic segmentation task를 사용한다. train 795장, test 654장 분할을 따른다. 이 데이터셋에서는 depth 활용 방식에 따른 비교가 흥미롭다.

RGB만 사용한 FCN-32s는 mean IU 29.2였다. RGB-D를 입력 채널 수준에서 early fusion한 모델은 30.5로 약간 상승했지만 큰 이득은 아니었다. 저자들은 meaningful gradient를 깊은 층까지 전달하기 어렵기 때문일 수 있다고 해석한다. 반면 Gupta et al.의 HHA depth encoding을 활용하면 late fusion 방식이 더 잘 작동했다. RGB-HHA late fusion FCN-32s는 mean IU 32.8, 이를 FCN-16s로 개선한 모델은 34.0을 기록했다. 이는 Gupta et al.의 28.6을 넘어서는 결과다.

이 결과는 depth를 단순히 raw channel로 추가하는 것보다, 구조화된 representation인 HHA와 별도 스트림을 결합하는 방식이 더 효과적일 수 있음을 보여준다.

### 4.4 SIFT Flow

SIFT Flow는 33개 semantic categories와 3개 geometric categories를 가진 데이터셋이다. 논문은 semantic segmentation과 geometric prediction을 동시에 하는 two-headed FCN-16s를 사용한다. 즉, 하나의 shared representation 위에 semantic head와 geometric head를 각각 붙이고, 각자의 loss를 동시에 학습한다.

결과적으로 semantic segmentation에서는 FCN-16s가 pixel accuracy 85.2, mean accuracy 51.7, mean IU 39.5를 기록했고, geometric prediction accuracy는 94.3이었다. 논문은 이 joint model이 semantic, geometric 두 과제 모두에서 각각 따로 학습한 모델과 동등한 성능을 내면서도, 학습과 추론 비용은 거의 한 모델 수준이라고 설명한다. 이는 FCN이 multi-task dense prediction에도 자연스럽게 확장 가능함을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제를 매우 단순하고 일반적인 방식으로 재정의했다는 점이다. semantic segmentation을 patch classification의 집합처럼 다루지 않고, image classification network를 dense predictor로 바꾸는 방식으로 접근했다. 이 덕분에 whole-image training, end-to-end optimization, 빠른 inference가 동시에 가능해졌다. 논문은 단순히 아이디어 차원에 머무르지 않고 AlexNet, VGG16, GoogLeNet에 실제로 적용해 비교했고, 여러 데이터셋에서 일관되게 강한 결과를 보였다.

둘째 강점은 skip architecture다. 깊은 층과 얕은 층을 결합해 semantic richness와 spatial precision의 균형을 맞추는 설계는 이후 semantic segmentation 구조의 표준적 발상으로 자리 잡았다. 특히 FCN-32s에서 FCN-16s, FCN-8s로 이어지는 개선 과정은 왜 multi-resolution fusion이 필요한지 매우 설득력 있게 보여준다.

셋째 강점은 복잡한 추가 장치 없이 성능과 속도를 동시에 개선했다는 점이다. 당시 강한 경쟁 방법들이 region proposal, superpixel, refinement module에 의존하던 것과 달리, 이 논문은 FCN 자체만으로 더 높은 mean IU와 훨씬 빠른 추론 속도를 얻었다. 이는 방법론의 실용성과 재사용성을 크게 높였다.

한계도 분명하다. 우선 기본 출력이 매우 coarse해서, 결국 bilinear initialization과 skip fusion에 의존해 공간 해상도를 회복해야 한다. 논문 자체도 FCN-8s에서 improvement가 점차 diminishing returns를 보인다고 말한다. 즉, 더 낮은 층까지 계속 fuse한다고 해서 무한정 좋아지는 구조는 아니다.

또 다른 한계는 경계 품질과 작은 구조 복원 능력에 대한 한계다. 논문은 mean IU 중심으로 성능을 보고하지만, Appendix A에서 스스로 mean IU가 fine-scale accuracy를 충분히 반영하지 못한다고 지적한다. 이는 곧 FCN이 mean IU에서는 매우 강해도, 실제 경계 정밀도나 얇은 구조 복원에는 여전히 제한이 있을 수 있음을 뜻한다.

또한 저자들은 일부 대안적 설계가 잘 작동하지 않았다고 솔직히 밝힌다. 예를 들어 pooling stride를 줄여 더 촘촘한 출력을 얻으려 하면, `fc6`에 해당하는 큰 $14\times14$ 필터가 필요해져 계산량과 학습 난도가 증가했다. shift-and-stitch도 제한적 실험에서는 layer fusion보다 cost-to-improvement ratio가 나빴다. 즉, 이 논문이 제안한 설계가 당시로서는 가장 합리적이었지만, dense prediction 자체를 근본적으로 고해상도화한 것은 아니었다.

비판적으로 보면, 이 논문은 segmentation을 위한 representation learning의 방향을 크게 진전시켰지만, decoder를 깊게 설계하거나 boundary-aware loss를 도입하는 식의 후속 발전까지는 다루지 않는다. 또한 class imbalance는 “필요 없었다”고 보고하지만, 이는 사용한 데이터셋과 설정에서의 관찰이지 일반적 결론으로 확대하기는 어렵다. 논문에 명시된 범위를 넘어서는 해석은 주의해야 한다.

## 6. 결론

이 논문은 modern classification convnet을 fully convolutional form으로 바꾸고, 이를 semantic segmentation에 맞게 fine-tuning하면 dense prediction을 효율적으로 수행할 수 있음을 보여주었다. 여기에 deconvolution 기반 upsampling과 skip architecture를 더해 coarse semantic 정보와 fine spatial 정보를 결합함으로써, PASCAL VOC, NYUDv2, SIFT Flow에서 강한 성능을 달성했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, classification network를 dense prediction network로 재해석하는 일반적 프레임워크를 제시했다. 둘째, whole-image end-to-end training이 patchwise training보다 효율적이면서도 효과적임을 실험적으로 보였다. 셋째, FCN-16s와 FCN-8s로 대표되는 multi-resolution skip fusion이 segmentation 품질을 실질적으로 개선함을 입증했다.

실제 적용 측면에서도 이 연구는 중요하다. 높은 정확도와 빠른 추론 속도를 동시에 보였기 때문에, 이후 실시간 혹은 대규모 dense prediction 시스템의 기반이 되기 적합했다. 더 넓게 보면, 이 논문은 semantic segmentation 전용의 새로운 네트워크 패러다임을 정립했고, 이후 encoder-decoder, feature pyramid, U-Net 계열 연구의 직접적인 출발점 역할을 했다고 평가할 수 있다.
