# A Survey on Deep Learning-based Architectures for Semantic Segmentation on 2D Images

- **저자**: Irem Ulku, Erdem Akagündüz
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/1912.10230

## 1. 논문 개요

이 논문은 2D 이미지 semantic segmentation, 즉 이미지의 모든 픽셀에 semantic class label을 부여하는 문제를 다루는 **survey paper**이다. 저자들은 특히 지난 10여 년 동안 급격히 발전한 **deep learning 기반 방법들**에 초점을 맞추어, 분야의 발전 흐름을 정리하고 기술적 핵심 과제를 분석한다. 단순히 방법들을 나열하는 것이 아니라, 왜 특정 구조가 등장했고 어떤 한계를 해결하려 했는지를 시간 순서에 따라 해석하는 것이 이 논문의 중심 목표이다.

연구 문제는 크게 두 층위로 볼 수 있다. 첫째, semantic segmentation 자체의 문제는 “무엇이 있는가”뿐 아니라 “어디에 있는가”를 픽셀 수준으로 정확히 찾아야 한다는 점에서 일반 분류보다 훨씬 어렵다. 둘째, survey 차원에서의 문제는 이미 방대한 문헌이 존재하는 상황에서, 이들을 단순 범주화가 아니라 **기술적 진화의 맥락**에서 재구성하는 일이다. 저자들은 이를 위해 방법들을 세 시기로 나눈다. 즉, **pre- and early deep learning era**, **fully convolutional era**, **post-FCN era**이다.

이 문제가 중요한 이유도 논문 전반에서 분명히 제시된다. semantic segmentation은 자율주행차, 드론, 로봇 보조 수술, 인간 친화형 로봇, 군사 시스템 등에서 핵심 모듈이다. 이들 응용에서는 단순한 객체 존재 판단만으로는 부족하고, 경계와 위치를 세밀하게 복원해야 한다. 따라서 semantic segmentation의 성능은 실제 시스템의 안전성과 직결된다. 저자들은 특히 fine-grained localisation과 global context integration이 여전히 핵심 난제로 남아 있다고 본다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 semantic segmentation 연구를 단순히 모델 이름별로 정리하지 않고, **분야가 해결하려고 했던 기술적 병목**을 중심으로 재배치하는 데 있다. 저자들이 강조하는 대표 병목은 다음과 같다. 첫째, pooling과 hierarchical feature extraction 때문에 발생하는 **fine-grained localisation 손실**이다. 둘째, fully convolutional 구조가 본질적으로 약한 **global context modeling** 문제이다. 셋째, 객체 크기 변화와 입력 해상도 변화에 대한 **scale invariance** 문제이다.

이 논문은 FCN(Fully Convolutional Network)을 semantic segmentation 역사에서 명확한 전환점으로 둔다. pre-deep learning 시기와 early deep learning 시기에는 graphical models, handcrafted features, fully connected classification network의 개조형 구조 등이 활용되었지만, 실제로 dense pixel prediction을 효율적으로 수행하는 구조적 해법은 FCN 이후에 본격화되었다고 평가한다. 이후 등장한 대부분의 state-of-the-art 방법은 사실상 FCN의 확장으로 해석될 수 있다는 것이 저자들의 관점이다.

기존 survey와의 차별점으로 저자들은 세 가지를 내세운다. 첫째, 2D visible imagery에 범위를 제한해 깊이 있는 논의를 가능하게 했다. 둘째, 공개 데이터셋, leaderboard, 성능 지표까지 함께 다뤄 실무적 맥락을 제공했다. 셋째, 무엇보다도 **chronological evolution**, 즉 기술이 어떤 한계를 해결하며 다음 단계로 이동했는지를 논리적으로 보여주려 했다. 이 점에서 이 논문은 단순 개론보다 “연구 흐름의 해석”에 더 가깝다.

## 3. 상세 방법 설명

이 논문은 새로운 모델을 제안하는 논문이 아니라 survey이므로, 하나의 통합 파이프라인이나 단일 loss function을 제시하지 않는다. 대신 분야 전체의 방법론을 구조적으로 설명한다. 따라서 이 섹션에서는 논문이 제시한 **분류 체계와 핵심 설계 요소**를 중심으로 정리해야 한다.

### 3.1 문제 설정과 평가 관점

semantic segmentation은 이미지의 각 픽셀에 class label을 부여하는 문제다. 여기서 중요한 것은 단순 classification이 아니라 localisation까지 포함된다는 점이다. 즉, 어떤 클래스인지 맞히는 것과 동시에 그 클래스가 정확히 어느 픽셀들에 해당하는지 맞혀야 한다.

논문은 성능 평가를 accuracy와 computational complexity로 나눈다. accuracy 측면에서는 Pixel Accuracy, Mean Pixel Accuracy, IoU, mIoU, FwIoU, Precision/Recall 기반 지표, F-score, AP, Hausdorff Distance 등을 설명한다. 특히 semantic segmentation에서 가장 많이 쓰이는 지표는 **IoU와 그 변형들, 그리고 AP**라고 정리한다.

대표 식들은 다음과 같다.

Pixel Accuracy와 Mean Pixel Accuracy는 다음과 같이 정의된다.

$$
PA = \frac{\sum_{j=1}^{k} n_{jj}}{\sum_{j=1}^{k} t_j}, \qquad
mPA = \frac{1}{k}\sum_{j=1}^{k}\frac{n_{jj}}{t_j}
$$

여기서 $n_{jj}$는 class $j$에 대해 정답도 예측도 $j$인 픽셀 수, 즉 true positive에 해당하고, $t_j$는 실제로 class $j$인 전체 픽셀 수이다.

IoU는 특정 클래스에서 예측과 정답의 교집합을 합집합으로 나눈 값이며, class-wise average를 취하면 mIoU가 된다.

$$
mIoU = \frac{1}{k}\sum_{j=1}^{k}\frac{n_{jj}}{n_{ij}+n_{ji}+n_{jj}}, \quad i \ne j
$$

이 식의 직관은 간단하다. 분자는 맞춘 픽셀 수이고, 분모는 맞춘 픽셀에 더해 false positive와 false negative까지 모두 포함하므로, Pixel Accuracy보다 더 엄격하다. 논문은 바로 이 이유 때문에 IoU가 semantic segmentation에서 더 informative하다고 설명한다.

Precision과 Recall은 다음처럼 주어진다.

$$
Precision = \frac{n_{jj}}{n_{ij}+n_{jj}}, \qquad
Recall = \frac{n_{jj}}{n_{ji}+n_{jj}}, \quad i \ne j
$$

그리고 F-score는

$$
F\text{ score} = \frac{2 \times Precision \times Recall}{Precision + Recall}
$$

로 정의된다. 논문은 PRC 기반 지표가 false positive와 false negative의 영향을 더 잘 구분한다는 점을 강조한다.

Hausdorff Distance는 분할 경계의 가장 큰 오차를 보려는 지표다. 논문은 다음 식들을 제시한다.

$$
hd(X,Y)=\max_{x \in X}\min_{y \in Y}\|x-y\|_2
$$

$$
hd(Y,X)=\max_{y \in Y}\min_{x \in X}\|x-y\|_2
$$

$$
HD(X,Y)=\max(hd(X,Y), hd(Y,X))
$$

즉 두 픽셀 집합 사이에서 가장 멀리 떨어진 대응 오차를 본다. 이는 경계 품질을 따로 보고 싶을 때 의미가 있다.

### 3.2 데이터셋과 문제 환경

논문은 semantic segmentation 데이터셋을 크게 **general-purpose**와 **urban street**로 나눈다. general-purpose 쪽의 대표는 PASCAL VOC, COCO, ADE20K이고, urban street 쪽의 대표는 Cityscapes이다.

PASCAL VOC는 20 foreground class와 1 background class를 가진 가장 고전적이고 널리 사용된 benchmark로 설명된다. COCO는 20만 장 규모의 이미지, 150만 object instance, 80 object category를 가지는 대규모 데이터셋으로, 많은 연구에서 pre-training 또는 fine-tuning에 활용된다. ADE20K는 150 semantic category를 포함하는 장면 중심 데이터셋이다. 자율주행 계열에서는 Cityscapes가 가장 영향력 있는 benchmark로 제시된다.

저자들은 데이터 규모와 품질이 deep learning에서 결정적이며, 특히 semantic segmentation에서는 dense annotation의 비용이 매우 높기 때문에 public leaderboard가 연구 방향을 강하게 이끈다고 본다.

### 3.3 시기별 방법론 진화

#### 3.3.1 Pre-deep learning approaches

deep learning 이전에는 MRF, CRF, random forest, layered model 같은 graphical model 또는 detector composition 방식이 주류였다. 이들은 인접 픽셀 간 관계나 super-pixel 수준의 구조를 이용해 semantic prior를 모델링했다. 하지만 대규모 데이터에서 semantic abstraction을 효과적으로 학습하지 못했고, 결국 CNN 기반 접근에 주도권을 넘겨주었다.

다만 이 시기의 유산 중 살아남은 것이 있는데, 바로 **CRF refinement**이다. 저자들은 CRF가 CNN이 약한 인접 픽셀 간 상호작용과 경계 보정을 도와주는 refinement layer로 post-FCN 시대까지 널리 사용되었다고 설명한다. 하지만 최적화가 어렵고 느리다는 약점 때문에 최근에는 점차 사라지는 추세라고 본다.

#### 3.3.2 Early deep learning approaches

FCN 이전 초기 deep learning 방식은 classification CNN을 segmentation에 억지로 맞추는 형태가 많았다. AlexNet이나 VGG의 fully connected layer를 fine-tuning하여 픽셀 단위 결정을 시도했지만, fully connected 구조는 parameter가 많고 overfitting과 계산비용 문제가 심했다. 또한 당시 네트워크 깊이도 충분하지 않아 semantic abstraction이 제한적이었다.

이 시기에는 recurrent architecture, super-pixel refinement, nearest-neighbor refinement 같은 다양한 우회적 방법이 등장했는데, 저자들은 이를 FCN의 필요성을 예고한 징후로 해석한다.

#### 3.3.3 Fully Convolutional Networks

이 논문에서 가장 중요한 기술적 전환점은 FCN이다. FCN의 핵심은 **fully connected layer를 제거하고 convolution만으로 dense prediction을 수행하는 것**이다. 이를 통해 입력 해상도에 덜 제한받고, parameter 수와 계산량을 줄이며, segmentation map을 end-to-end로 예측할 수 있게 되었다.

FCN의 핵심 구성은 세 가지다.

첫째, fully connected layer 제거이다. 이로 인해 classification network를 segmentation network로 바꾸는 것이 가능해졌다.

둘째, **deconvolution (upsampling)** 을 사용해 coarse feature map을 원래 이미지 해상도에 가까운 dense output으로 복원한다.

셋째, **skip connection**이다. pooling으로 인해 손실되는 spatial detail을 shallow layer에서 deep layer로 전달해 경계와 localisation 정보를 보존한다. FCN-32s, FCN-16s, FCN-8s는 이런 skip 구조의 밀도 차이를 반영한 버전들이다.

논문은 FCN의 skip connection이 이후 encoder-decoder 계열 구조의 직접적인 출발점이 되었다고 본다.

### 3.4 Post-FCN 시대의 핵심 기술들

저자들은 post-FCN 방법들의 주요 목적을 세 가지 한계 극복으로 정리한다. 즉, localisation 손실, global context 부족, multiscale 처리 부족이다.

#### 3.4.1 Encoder-Decoder Architecture

U-Net과 SegNet이 대표적이다. encoder는 pooling으로 spatial dimension을 줄이며 추상적 feature를 만들고, decoder는 이를 다시 복원한다. skip connection을 통해 같은 해상도 수준의 encoder feature를 decoder에 전달하여 localisation을 회복한다. 논문은 이 구조가 fine localisation과 hierarchical semantics를 동시에 다루는 매우 중요한 틀이라고 평가한다.

#### 3.4.2 Spatial Pyramid Pooling

SPP는 서로 다른 spatial scale의 정보를 고정 길이 representation으로 모으는 방식이다. CNN에 적용하면 입력 크기가 달라도 pyramid pooling을 통해 다중 스케일 문맥을 통합할 수 있다. 다만 저자들은 SPP가 자동으로 scale-invariant한 것은 아니라고 분명히 지적한다. 여러 스케일로 학습해야 실제 scale invariance가 생긴다.

#### 3.4.3 Feature Concatenation

여러 수준의 feature나 global feature를 결합하는 접근이다. DeepMask, SharpMask, ParseNet 등이 예시로 언급된다. 핵심은 shallow spatial detail과 deep semantic context를 합쳐 더 나은 segmentation을 얻는 것이다. 하지만 논문은 이런 hybrid 구조가 종종 학습이 어렵다고 본다.

#### 3.4.4 Dilated Convolution

dilated convolution은 filter 내부에 gap을 둠으로써 receptive field를 빠르게 키우는 방식이다. pooling 없이도 더 넓은 문맥을 볼 수 있어 feature map resolution을 유지하는 데 유리하다. DeepLab 계열이 대표적이다. 단점은 feature map이 계속 크기 때문에 GPU memory와 computation demand가 크다는 점이다.

#### 3.4.5 Conditional Random Fields

CRF는 경계 보정과 local consistency 확보를 위해 refinement layer로 쓰인다. Dense CRF, CRF-as-RNN, Convolutional CRFs 등이 예로 소개된다. 저자들은 한때 매우 중요한 보조 기술이었지만, 최근에는 속도와 최적화 문제 때문에 점차 덜 사용된다고 해석한다.

#### 3.4.6 Recurrent Approaches

RNN이나 LSTM은 순차 정보 또는 장거리 의존성 modeling에 강하다. 이를 semantic segmentation에 적용하면 global dependency를 보완할 수 있다. Graph LSTM, DAG-RNN, ReSeg 등이 예로 나온다. 그러나 순차 처리가 포함되어 계산 효율이 떨어지는 경우가 많다.

### 3.5 Scale Invariance와 Object Detection-based Methods

논문은 multiscale processing과 scale invariance를 구분한다. 여러 스케일 feature를 쓰는 것만으로는 scale-invariant하다고 할 수 없다. 본질적인 scale invariance는 다양한 스케일의 학습 데이터나 적절한 normalization에 의해 얻어진다고 설명한다.

또 하나의 중요한 흐름은 object detection 기반 접근이다. Mask R-CNN, YOLACT, SOLO, SOLOv2, DeepSnake, BlendMask 등이 대표 예시다. 이들은 객체를 먼저 또는 동시에 찾고 mask를 예측한다. 특히 instance segmentation과 semantic segmentation의 경계가 점점 가까워지고 있으며, bounding box annotation만으로 학습하는 weakly supervised 방법들까지 미래 방향으로 제시된다.

## 4. 실험 및 결과

이 논문은 새로운 모델의 실험 논문이 아니라 survey이기 때문에, 하나의 통일된 실험 세팅이나 저자 고유의 benchmark 실험을 제시하지 않는다. 대신 기존 대표 방법들의 leaderboard 성능과 계산 효율을 종합해 보여준다. 따라서 “실험 결과”는 저자 자신의 단일 모델 결과가 아니라, **문헌 전반의 비교 정리**로 이해해야 한다.

데이터셋 측면에서는 PASCAL VOC, COCO, ADE20K, Cityscapes가 가장 중요한 benchmark로 다뤄진다. 지표는 주로 mIoU, mPA, mAP가 사용된다. semantic segmentation에서는 특히 mIoU가 중심 지표이고, instance segmentation 계열에서는 mAP가 많이 쓰인다.

논문이 제시한 Table 1은 약 8년에 걸친 34개 방법을 정리하며, 각 방법의 핵심 아이디어, benchmark 성능, 계산 효율을 함께 기술한다. 몇 가지 대표 결과는 다음과 같이 읽을 수 있다.

FCN은 SIFTflow에서 85.2% mPA, PASCAL 2012에서 62.2% mIoU, Cityscapes에서 65.3% mIoU, ADE20K에서 39.3% mIoU를 기록한 것으로 정리된다. 이는 FCN이 전환점이긴 하지만 아직 localisation과 context 처리 측면에서 한계가 있음을 보여준다.

DeepLab 계열은 FCN 이후의 대표적 개선 흐름이다. DeepLab.v1은 PASCAL 2012에서 66.4% mIoU, DeepLab.v2는 79.7% mIoU, DeepLab.v3는 85.7% mIoU, DeepLab.v3+는 87.3% mIoU까지 올라간다. 이 흐름은 dilated convolution, ASPP, encoder-decoder 결합이 실제로 유의미한 개선을 가져왔음을 보여준다.

PSPNet은 PASCAL 2012에서 85.5% mIoU, Cityscapes에서 81.2% mIoU, ADE20K에서 55.4% mIoU를 기록한다. 이는 pyramid pooling을 통한 global context aggregation의 효과를 시사한다.

EMANet152는 PASCAL 2012에서 88.2% mIoU, COCO에서 39.9% mIoU로 제시된다. EfficientNet-L2 + NASFPN + Noisy Student는 논문 작성 시점 기준 PASCAL VOC 2012 leaderboard 선두로 90.5% mIoU를 기록했다고 정리된다. 저자들은 이를 NAS 기반 설계의 부상으로 해석한다.

실시간 또는 고효율 계열에서는 YOLACT, ESE-Seg, SOLOv2, SwiftNetRN18-Pyr 등이 언급된다. 예를 들어 YOLACT는 PASCAL SBD에서 72.3% $mAP_{50}$, COCO에서 31.2% mAP를 기록하며 real-time instance segmentation 예시로 제시된다. SwiftNetRN18-Pyr는 ADE20K에서 35.0% mIoU로 정확도는 다소 낮지만 beyond real-time performance를 강조하는 방법으로 소개된다.

논문은 Figure 6을 통해 한 장의 PASCAL VOC validation image에 대해 FCN-32s, FCN-8s, CMSA, DeepLab-v1, CRF-as-RNN, DeepLab-v2, DeepLab-v2+CRF, PAN 등의 qualitative result를 비교한다. 저자들의 해석은 분명하다. 시간이 지날수록 segmentation이 더 정교해졌고, 특히 경계와 객체 영역 구분이 개선되었다. 다만 최근 2년의 방법들은 성능 향상 폭이 예전만큼 크지 않으며, 그래서 NAS나 object detection-based framework 같은 새로운 방향이 부상하고 있다고 본다.

또한 계산 효율에 대해서는 별 1개에서 4개까지의 범주형 척도를 사용한다. 이는 정량적 FPS 표가 아니라 구조적 판단에 기반한 귀납적 비교다. 예를 들어 CRF, RNN, attention, fully connected context module이 있으면 효율이 떨어지고, fully convolutional이며 lightweight detector 기반이면 효율이 좋아진다는 식으로 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation 문헌을 단순 카탈로그로 다루지 않고, **기술적 난제와 구조적 대응**의 관점에서 정리했다는 점이다. 특히 FCN을 기준으로 전후를 나누고, post-FCN 시대를 fine-grained localisation, global context, scale invariance, object detection-based approaches라는 문제 중심 축으로 재해석한 점이 설득력 있다. 이 덕분에 독자는 왜 U-Net, DeepLab, PSPNet, CRF-as-RNN, YOLACT 같은 구조가 등장했는지를 맥락 속에서 이해할 수 있다.

또 다른 강점은 데이터셋, leaderboard, 평가지표, 계산 효율을 함께 다루었다는 점이다. 많은 survey가 모델 구조만 요약하는 데 그치지만, 이 논문은 실제 연구 생태계가 어떻게 benchmark 중심으로 굴러가는지도 설명한다. 저자들은 공개 leaderboard가 학계와 산업계의 경쟁을 가속한다고 보고, 이는 분야 발전을 이해하는 데 중요한 시각을 제공한다.

방법론 설명도 비교적 균형 잡혀 있다. 예를 들어 CRF를 무조건 긍정하지 않고 refinement의 장점과 느린 속도 및 최적화 난점을 함께 언급한다. dilated convolution에 대해서도 receptive field 확장의 장점뿐 아니라 메모리와 연산량 증가라는 비용을 분명히 적는다. 이런 점은 survey로서 신뢰도를 높인다.

반면 한계도 명확하다. 첫째, 이 논문은 의도적으로 **2D visible imagery**에 범위를 제한한다. 따라서 RGB-D, 3D point cloud, medical volume, CT/MRI 같은 중요한 segmentation 영역은 제외된다. 저자 스스로도 이를 명시한다. 둘째, 계산 효율 비교는 체계적인 동일 조건 benchmark가 아니라 구조적 추론과 문헌 보고에 기반한다. 따라서 별점 기반 효율 비교는 직관적이지만 엄밀한 hardware-normalized 비교는 아니다. 셋째, survey 특성상 각 방법의 수식적 세부나 학습 절차를 모두 깊게 파고들지는 않는다. 예를 들어 DeepLab, PSPNet, NAS 기반 모델 각각의 loss 구성이나 optimizer 설정까지 통일적으로 정리하지는 않는다.

비판적으로 보면, “fine-grained localisation이 핵심 과제”라는 저자들의 테제는 전체 분야를 꽤 잘 설명하지만, 2021년 전후로 부상한 weak supervision, zero-shot, domain adaptation, real-time deployment 같은 문제들은 localisation만으로 완전히 환원되지는 않는다. 논문도 이를 후반부에서 미래 방향으로 다루지만, 본론의 중심 서사는 여전히 localisation 중심이다. 이는 장점이기도 하지만, 동시에 일부 최신 흐름을 다소 보조적 위치에 두는 해석이라고 볼 수 있다.

## 6. 결론

이 논문은 2D semantic segmentation의 deep learning 기반 발전사를 **pre-/early deep learning**, **FCN**, **post-FCN**이라는 세 시기로 나누고, 각 시기에서 어떤 구조적 문제가 제기되었고 어떤 해결책이 등장했는지를 정리한 학술적 survey이다. 핵심 메시지는 분명하다. semantic segmentation의 본질적 난제는 픽셀 수준의 정확한 localisation이며, 이를 잘 해결하려면 local detail과 global context를 효율적으로 결합해야 한다는 것이다.

저자들이 정리한 흐름에 따르면, FCN은 dense prediction의 기본 틀을 열었고, 이후 encoder-decoder, dilated convolution, pyramid pooling, attention, recurrent module, object detection-based approach, NAS 기반 설계가 이를 확장해 왔다. 동시에 CRF 같은 refinement module은 성능 향상에 기여했지만 점차 실시간성과 최적화 문제 때문에 퇴조하고 있다.

향후 연구 방향으로는 weakly-supervised semantic segmentation, zero-/few-shot learning, domain adaptation, real-time processing, contextual information aggregation이 중요하다고 제시된다. 실제 적용 측면에서도 이는 매우 현실적인 전망이다. 픽셀 단위 라벨링 비용이 매우 높은 상황에서 annotation 효율성과 generalization, 그리고 실시간성이 동시에 요구되기 때문이다. 따라서 이 논문은 단순한 과거 정리가 아니라, semantic segmentation 연구가 앞으로 어디로 갈지를 이해하는 데도 유용한 기준점을 제공한다.
