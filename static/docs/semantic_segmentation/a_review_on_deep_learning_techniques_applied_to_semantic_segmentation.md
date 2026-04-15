# A Review on Deep Learning Techniques Applied to Semantic Segmentation

- **저자**: A. Garcia-Garcia, S. Orts-Escolano, S.O. Oprea, V. Villena-Martinez, J. Garcia-Rodriguez
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1704.06857

## 1. 논문 개요

이 논문은 semantic segmentation에 적용된 deep learning 기법들을 체계적으로 정리한 review paper이다. 저자들은 semantic segmentation을 단순한 이미지 분류나 검출보다 더 미세한 수준의 scene understanding 문제로 위치시킨다. 즉, 이미지 전체에 하나의 class를 붙이는 classification, 물체의 위치를 찾는 detection/localization을 넘어, 각 pixel마다 class label을 부여하는 dense prediction 문제가 semantic segmentation의 핵심이라고 설명한다.

논문의 직접적인 목표는 세 가지로 정리할 수 있다. 첫째, 이 분야를 처음 접하는 연구자가 필요한 용어와 배경지식을 빠르게 이해하도록 돕는 것이다. 둘째, 어떤 dataset과 benchmark가 존재하며 각각이 어떤 목적에 적합한지 정리하는 것이다. 셋째, 주요 deep learning 기반 segmentation 방법들을 구조적으로 비교하고, 정량 결과를 함께 제시하여 당시의 state of the art를 조망하는 것이다.

이 문제가 중요한 이유도 논문에서 분명히 제시된다. semantic segmentation은 autonomous driving, indoor navigation, human-machine interaction, image retrieval, augmented reality 같은 실제 응용과 직접 연결된다. 특히 장면을 픽셀 단위로 이해해야 하는 응용에서는 단순 classification보다 훨씬 정밀한 출력이 필요하다. 저자들은 deep learning이 computer vision 전반에서 큰 성과를 내고 있지만, semantic segmentation 분야는 변화 속도가 매우 빠르고 문헌도 많아 전체 흐름을 따라가기가 어렵다고 지적한다. 따라서 이 논문은 개별 방법을 새로 제안하는 것이 아니라, 빠르게 발전하던 분야를 정리하고 공통 흐름을 드러내는 데 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 semantic segmentation을 하나의 단일 모델 문제로 보기보다, 여러 설계 축의 조합으로 이해하는 데 있다. 저자들은 당시의 주요 방법들을 단순 나열하지 않고, 어떤 한계가 있었고 그것을 해결하기 위해 어떤 architectural idea가 등장했는지의 관점으로 정리한다. 그 흐름의 출발점은 Fully Convolutional Network(FCN)이다. 논문은 FCN을 semantic segmentation에 deep learning을 본격적으로 정착시킨 핵심 전환점으로 본다.

FCN 이후 연구의 큰 방향은 크게 몇 가지로 구분된다. 첫째, classification CNN을 segmentation에 맞게 바꾸는 encoder-decoder 계열이다. 둘째, pixel prediction의 local ambiguity를 줄이기 위해 context를 더 잘 통합하는 방향이다. 여기에는 CRF, dilated convolution, multi-scale prediction, feature fusion, RNN 기반 context modeling이 포함된다. 셋째, semantic segmentation보다 더 어려운 instance segmentation으로의 확장이다. 넷째, RGB를 넘어 RGB-D, 3D point cloud, video sequence처럼 입력 데이터 구조가 달라지는 문제에 deep model을 적용하는 방향이다.

기존 접근과의 차별점이라는 측면에서, 이 논문 자체의 기여는 새로운 segmentation model 제안이 아니라 deep learning 기반 semantic segmentation을 전면적으로 다룬 초기 survey라는 점이다. 저자들은 기존 survey들이 segmentation 일반론은 잘 다루지만 최신 dataset, framework, deep learning 세부기술 분석은 부족했다고 주장한다. 따라서 이 논문의 차별점은 deep learning 아키텍처와 그 변형들을 중심으로 dataset, method, result, future direction을 한 프레임 안에서 연결했다는 데 있다.

## 3. 상세 방법 설명

논문은 먼저 semantic segmentation 문제를 다음처럼 정식화한다. label space를 $L=\{l_1,l_2,\dots,l_k\}$라 하고, 각 random variable 집합을 $X=\{x_1,x_2,\dots,x_N\}$라 할 때, 각 $x_i$에 적절한 label $l$을 할당하는 것이 목표다. 보통 $X$는 $W \times H = N$개의 pixel을 가진 2D image이지만, 이 formulation은 volumetric data나 hyperspectral image처럼 더 높은 차원에도 확장될 수 있다고 설명한다. 실제 문제에서는 background 또는 void를 포함해 $k+1$개 class로 다루는 경우가 많다.

논문은 주요 segmentation 방법을 이해하려면 먼저 대표 CNN backbone을 알아야 한다고 보고 AlexNet, VGG-16, GoogLeNet, ResNet, ReNet을 소개한다. AlexNet은 현대 deep CNN의 출발점으로, 5개의 convolution layer와 3개의 fully connected layer, ReLU, dropout을 사용한다. VGG-16은 작은 receptive field를 가진 convolution을 여러 층 쌓아 더 적은 parameter와 더 많은 nonlinearity를 확보한다. GoogLeNet은 inception module을 통해 서로 다른 크기의 convolution과 pooling을 병렬로 적용해 효율성을 높인다. ResNet은 residual block과 identity skip connection을 도입해 매우 깊은 network 학습을 가능하게 한다. ReNet은 convolution 대신 여러 방향으로 sweeping하는 RNN을 사용해 이미지의 spatial dependency를 모델링하는 접근이다.

또 하나 중요한 배경은 transfer learning이다. segmentation dataset은 pixel-wise annotation이 필요해 분류용 dataset보다 규모가 작기 때문에, ImageNet 등에서 사전학습된 classification network를 fine-tuning하는 것이 일반적이라고 설명한다. 이때 낮은 층은 더 generic feature를 담고, 높은 층은 task-specific feature를 더 많이 담으므로 주로 상위 층을 fine-tune하는 전략이 유리하다고 정리한다. data augmentation 역시 중요한데, translation, rotation, scaling, color shift, crop 등을 통해 training sample을 늘리고 regularization 효과를 얻는다.

방법론의 본론에서 가장 핵심은 FCN이다. FCN의 아이디어는 classification CNN의 fully connected layer를 convolution layer로 바꾸어 spatial score map을 출력하게 만들고, 이를 deconvolution, 즉 fractionally strided convolution으로 upsampling해 pixel-wise prediction을 얻는 것이다. 이 과정 덕분에 입력 크기가 고정되지 않아도 되고, end-to-end 학습이 가능해진다. 논문은 FCN을 deep semantic segmentation의 사실상 출발점으로 평가한다.

하지만 FCN만으로는 한계가 있다. 대표적으로 global context를 충분히 반영하지 못하고, spatial detail이 손실되며, instance awareness가 없고, 고해상도에서 실시간 처리가 어렵고, 3D처럼 비정형 데이터에 바로 적용하기 어렵다. 이후 방법들은 대체로 이 한계를 보완하는 방향으로 이해된다.

첫 번째 큰 축은 decoder variant이다. SegNet이 대표적이다. SegNet은 encoder-decoder 구조를 사용하며, encoder의 max-pooling 단계에서 저장한 pooling index를 decoder에서 upsampling에 사용한다. 즉, FCN처럼 deconvolution filter를 학습해 upsample하는 대신, encoder에서 어디가 max였는지의 위치 정보를 이용해 decoder feature map을 복원한다. 그 뒤 convolution과 softmax classifier를 거쳐 최종 pixel label을 낸다. 이 구조는 feature map을 원래 해상도로 복원하는 과정을 더 명시적으로 만든다.

두 번째 큰 축은 context integration이다. segmentation에서는 한 pixel 주변의 local 정보만으로는 class를 정확히 결정하기 어려운 경우가 많기 때문이다. 논문은 이를 해결하는 다섯 가지 대표 방향을 설명한다.

첫째, Conditional Random Field(CRF)이다. DeepLab은 CNN이 낸 coarse한 score map을 fully connected pairwise CRF로 후처리하여 경계와 fine detail을 복원한다. CRF는 pixel 간 상호작용을 모델링해 short-range와 long-range dependency를 모두 고려하게 만든다. CRFasRNN은 이 아이디어를 한 단계 더 발전시켜, mean-field inference 과정을 RNN처럼 unroll하여 FCN 내부에 통합했다. 즉, CRF를 외부 post-processing이 아니라 network의 일부로 넣고 end-to-end로 학습할 수 있게 만든다.

둘째, dilated convolution이다. dilated convolution 또는 atrous convolution은 filter 사이에 간격을 두어 receptive field를 키우는 연산이다. dilation rate를 $l$이라 하면, resolution을 크게 낮추지 않고도 더 넓은 문맥을 볼 수 있다. 일반 convolution은 사실상 $1$-dilated convolution이다. 이 방식의 장점은 parameter 수를 과도하게 늘리지 않으면서 receptive field를 효율적으로 확장할 수 있다는 것이다. DeepLab의 향상 버전, Yu와 Koltun의 context aggregation model, ENet 등이 이를 사용한다.

셋째, multi-scale prediction이다. 하나의 고정된 scale에서만 feature를 추출하면 객체 크기 변화에 취약할 수 있다. 그래서 서로 다른 scale에서 입력을 처리한 뒤 결과를 합치는 접근이 등장했다. Raj 등의 multi-scale FCN은 원해상도 경로와 2배 확대 경로를 병렬로 사용한다. Eigen 등의 multi-scale CNN은 coarse-to-fine 순차 refinement를 사용한다. Bian 등의 방법은 여러 개의 FCN을 독립적으로 학습한 뒤, feature를 fuse하고 마지막 layer를 fine-tune한다. 핵심은 서로 다른 scale의 context를 함께 쓰는 것이다.

넷째, feature fusion이다. ParseNet은 이전 layer의 global feature를 unpool하여 현재 layer의 local feature map과 concatenate한다. 이는 skip connection처럼 나중에 합치는 late fusion과 달리, 더 이른 단계에서 global context를 섞는 early fusion의 예다. 이후 SharpMask 같은 방법도 top-down refinement 과정에서 저수준 spatial feature와 고수준 semantic feature를 결합한다.

다섯째, Recurrent Neural Network 기반 context modeling이다. 이미지에는 자연스러운 1차원 순서가 없지만, RNN을 수평·수직 방향으로 펼치면 장거리 dependency를 학습할 수 있다. ReSeg는 VGG-16 초반부로 feature를 뽑고, 이후 ReNet layer를 쌓은 뒤 transposed convolution으로 upsampling한다. LSTM-CF는 RGB와 depth의 두 입력을 받아, 세 scale의 RGB feature와 depth 정보를 LSTM으로 통합한다. 2D-LSTM은 이미지를 여러 window로 나누어 네 방향 LSTM으로 처리한다. DAG-RNN은 이미지를 directed acyclic graph로 분해하여 graph 구조에서 context를 모델링한다. 이 계열의 핵심은 CNN이 놓치기 쉬운 전역 문맥과 장거리 상관관계를 recurrent structure로 포착하는 것이다.

논문은 instance segmentation도 별도 섹션에서 다룬다. SDS는 region proposal을 만든 뒤 CNN feature와 SVM, NMS를 조합하여 detection과 segmentation을 함께 수행한다. DeepMask는 입력 patch가 object를 포함할 확률과 그 mask를 동시에 예측한다. SharpMask는 DeepMask를 확장해 top-down refinement module을 도입하여 고수준 semantic cue와 저수준 spatial cue를 결합한다. MultiPathNet은 Fast R-CNN과 DeepMask proposal을 결합하고, localization, foveal context, multi-scale feature를 강화한다.

RGB-D 데이터에 대해서는 depth를 RGB처럼 3채널로 encoding하는 HHA(horizontal disparity, height above ground, angle with gravity) 같은 기법이 소개된다. 이렇게 하면 RGB용 architecture에 depth를 쉽게 넣을 수 있다. LSTM-CF는 RGB와 depth를 함께 처리하는 대표 사례다. 다만 논문은 어떤 다중시점 RGB-D 방법에서는 depth가 큰 성능 향상을 주지 않았다고도 보고한다. 이는 depth noise 때문일 수 있다고 원문에서 설명한다.

3D 데이터에 대해서는 두 흐름이 소개된다. 하나는 point cloud를 voxel grid로 변환한 뒤 3D CNN을 적용하는 방식이다. Huang 등의 방법이 여기에 해당한다. 이 방식은 regular representation을 만들 수 있다는 장점이 있지만, quantization과 spatial information loss라는 단점이 있다. 다른 하나는 PointNet처럼 raw point set을 직접 처리하는 방식이다. PointNet은 convolution 대신 MLP와 max-pooling을 사용해 point order에 불변한 global feature를 만든다. segmentation subnet은 global feature와 point-wise feature를 합쳐 각 point label을 예측한다. 이 논문은 PointNet을 비정형 3D 입력을 직접 처리하는 매우 중요한 전환으로 본다.

video sequence segmentation에 대해서는 Clockwork FCN이 대표로 소개된다. 핵심 통찰은 network의 얕은 층 feature는 frame 간 빨리 변하지만 깊은 층 feature는 상대적으로 천천히 변한다는 것이다. 그래서 layer를 여러 stage로 묶고, stage마다 다른 주기로 갱신하면 깊은 feature를 계속 재사용해 속도를 높일 수 있다. 갱신 주기는 고정일 수도 있고, motion이나 semantic change에 따라 adaptive할 수도 있다. 또 다른 접근으로는 3D convolution을 사용하는 방법이 있다. Tran 등의 voxel-to-voxel model은 16-frame clip 단위로 입력을 나누고, 3D convolution과 deconvolution으로 spatio-temporal feature를 학습하여 segmentation을 수행한다.

## 4. 실험 및 결과

논문은 review paper이므로 하나의 통일된 실험을 수행한 것이 아니라, 각 논문 저자들이 보고한 수치를 모아 비교한다. 먼저 평가 지표를 정리한다. Pixel Accuracy는 전체 pixel 중 맞춘 비율이다.

$$
PA = \frac{\sum_{i=0}^{k} p_{ii}}{\sum_{i=0}^{k}\sum_{j=0}^{k} p_{ij}}
$$

Mean Pixel Accuracy는 class별 accuracy를 평균한 것이다.

$$
MPA = \frac{1}{k+1}\sum_{i=0}^{k}\frac{p_{ii}}{\sum_{j=0}^{k} p_{ij}}
$$

가장 중요한 지표는 Mean Intersection over Union(MIoU)이다. 각 class마다 intersection over union을 계산한 후 평균한다.

$$
MIoU = \frac{1}{k+1}\sum_{i=0}^{k}\frac{p_{ii}}{\sum_{j=0}^{k} p_{ij} + \sum_{j=0}^{k} p_{ji} - p_{ii}}
$$

Frequency Weighted IoU(FWIoU)는 class frequency를 반영한 가중 IoU다.

$$
FWIoU = \frac{1}{\sum_{i=0}^{k}\sum_{j=0}^{k} p_{ij}}
\sum_{i=0}^{k}
\left(
\sum_{j=0}^{k} p_{ij}
\right)
\frac{p_{ii}}{\sum_{j=0}^{k} p_{ij} + \sum_{j=0}^{k} p_{ji} - p_{ii}}
$$

논문은 MIoU를 사실상 표준 metric으로 본다.

데이터셋 측면에서는 2D, 2.5D, 3D를 포괄적으로 정리한다. 2D에서는 PASCAL VOC, PASCAL Context, PASCAL Part, SBD, COCO, SYNTHIA, Cityscapes, CamVid, KITTI 변형, Youtube-Objects, Adobe Portrait Segmentation, MINC, DAVIS, Stanford Background, SiftFlow 등이 소개된다. 2.5D에서는 NYUDv2, SUN3D, SUNRGBD, RGB-D Object Dataset 등이 정리된다. 3D에서는 ShapeNet Part, Stanford 2D-3D-S, 3D Mesh benchmark, Sydney Urban Objects, Semantic3D benchmark가 소개된다. 이 논문의 강점 중 하나는 각 dataset의 목적, class 수, modality, split, synthetic/real 여부까지 함께 정리했다는 점이다.

정량 결과에서는 RGB image segmentation에서 DeepLab이 가장 일관되게 강한 성능을 보인다고 결론짓는다. PASCAL VOC 2012 test set에서는 DeepLab이 $79.70$ IoU로 가장 높고, Dilation이 $75.30$, CRFasRNN이 $74.70$, ParseNet이 $69.80$, FCN-8s가 $67.20$이다. 이는 FCN 기반 기본 구조 위에 CRF refinement와 atrous convolution을 더한 DeepLab 계열이 당시 가장 안정적인 성능을 냈다는 뜻이다.

PASCAL Context에서도 DeepLab이 $45.70$ IoU로 가장 높고, CRFasRNN은 $39.28$, FCN-8s는 $39.10$이다. PASCAL Person-Part에서는 DeepLab이 $64.94$ IoU를 기록했다고 정리한다. 즉, 일반 semantic segmentation뿐 아니라 part-level segmentation에서도 DeepLab 계열이 강세였다는 점이 강조된다.

도시 주행 데이터셋인 CamVid에서는 결과가 조금 다르다. DAG-RNN이 $91.60$ IoU로 가장 높고, Bayesian SegNet이 $63.10$, SegNet이 $60.10$, ReSeg가 $58.80$, ENet이 $55.60$이다. 이 결과는 context를 강하게 모델링하는 recurrent structure가 특정 장면 parsing benchmark에서 유리할 수 있음을 보여준다. Cityscapes에서는 다시 DeepLab이 $70.40$ IoU로 가장 높고, Dilation10이 $67.10$, FCN-8s가 $65.30$, CRFasRNN이 $62.50$, ENet이 $58.30$이다. Cityscapes처럼 더 어렵고 실제 주행에 가까운 benchmark에서도 DeepLab의 우세가 유지된다.

Stanford Background에서는 rCNN이 $80.20$ IoU, 2D-LSTM이 $78.56$을 기록한다. SiftFlow에서는 DAG-RNN이 $85.30$, rCNN이 $77.70$, 2D-LSTM이 $70.11$이다. 따라서 outdoor scene parsing 계열 benchmark에서는 recurrent/context model이 비교적 두드러진 결과를 낸다.

RGB-D 결과는 논문 시점 기준으로 비교군이 많지 않다. SUNRGBD에서 LSTM-CF는 $48.10$ IoU, NYUDv2에서는 $49.40$ IoU, SUN3D에서는 $58.50$을 기록한다. 논문은 RGB-D segmentation 분야가 아직 충분히 넓게 비교되지 않았음을 간접적으로 보여준다.

3D에서는 PointNet이 ShapeNet Part에서 $83.70$ IoU, Stanford 2D-3D-S에서 $47.71$ IoU를 기록한다. PointNet이 사실상 유일한 직접 비교 대상으로 제시되는 점은, 당시 3D semantic segmentation이 아직 매우 초기 단계였음을 보여준다.

Sequence segmentation에서는 Clockwork Convnet이 Cityscapes에서 $64.40$ IoU, Youtube-Objects에서 $68.50$을 기록한다. 논문은 sequence 분야 역시 benchmark와 비교 결과가 아직 제한적이라고 본다.

저자들은 실험 결과 자체보다, 결과 보고 방식의 문제도 비판한다. 많은 논문이 standard dataset에 대해 평가하지 않거나, setup 설명이 부족하거나, source code를 공개하지 않아 reproducibility가 떨어진다는 것이다. 또 실행 시간과 memory footprint는 중요함에도 불구하고 대부분의 논문이 accuracy만 보고한다고 지적한다. 이 지적은 review paper로서 매우 중요한 관찰이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 semantic segmentation을 deep learning 관점에서 넓고 체계적으로 정리했다는 점이다. 단순히 모델 이름과 성능만 나열하지 않고, 왜 FCN이 중요했는지, 왜 context modeling이 필요했는지, 왜 RGB-D와 3D는 별도의 어려움이 있는지 같은 구조적 설명이 들어간다. 따라서 초심자에게는 입문서 역할을 하고, 기존 연구자에게는 당시의 설계 흐름을 한눈에 보게 해준다.

또 다른 강점은 dataset 정리가 매우 폭넓다는 점이다. 2D, RGB-D, 3D, sequence를 모두 포함해 총 28개 dataset을 소개하고, 목적과 split, class 수, modality를 함께 정리한다. 이 점은 실제 연구를 시작하려는 사람에게 매우 실용적이다. 방법 정리도 FCN 이후의 파생 방향을 기준으로 분류되어 있어, 기술적 계보를 이해하는 데 도움이 된다.

실험 결과를 다루는 방식도 의미가 있다. 저자들은 단순히 “어떤 방법이 가장 좋다”로 끝내지 않고, 비교 자체가 어려운 이유, 즉 서로 다른 dataset, 다른 metric, 불충분한 재현 정보, code 미공개, runtime/memory 미보고 같은 구조적 문제를 지적한다. 이는 해당 분야의 연구 관행을 비판적으로 보는 시각을 제공한다.

반면 한계도 분명하다. 첫째, 이 논문은 review paper이기 때문에 새로운 알고리즘이나 통일된 실험 프로토콜을 제시하지 않는다. 따라서 제시된 수치들은 모두 원 논문 저자들이 보고한 값에 의존한다. 서로 다른 training setup이나 data augmentation 정책, pretraining 여부가 완전히 통제되지 않았을 가능성이 있는데, 이 논문은 그런 차이를 메우는 추가 실험을 하지 않는다.

둘째, 결과 비교가 dataset마다 불균형하다. 예를 들어 RGB에서는 비교 대상이 꽤 많지만, RGB-D, 3D, sequence에서는 한두 방법만 결과가 제시된다. 이는 분야 자체의 초기성 때문이지만, 독자가 어떤 접근이 본질적으로 우월한지 일반화해서 받아들이기에는 제한이 있다. 논문도 사실 이를 인정하고 있다.

셋째, future direction의 일부는 당시 관점에서 타당하지만, review 특성상 제안이 비교적 넓고 일반적이다. 예를 들어 3D dataset 확충, real-time segmentation, memory reduction, temporal coherence, multi-view integration 같은 방향은 중요하지만, 구체적으로 어떤 기술 장벽이 핵심인지까지 깊게 파고들지는 않는다. 다만 이는 survey의 목적상 자연스러운 제한이기도 하다.

비판적으로 해석하면, 이 논문은 “무엇이 가장 좋은 모델인가”보다 “당시 field가 어떤 문제의식 위에서 움직였는가”를 보여주는 문헌이다. 따라서 개별 수치만 보는 것보다, FCN에서 시작해 context integration, multimodal fusion, irregular input handling, temporal modeling으로 확장되는 연구 흐름을 읽는 것이 더 중요하다.

## 6. 결론

이 논문은 deep learning 기반 semantic segmentation을 처음부터 끝까지 폭넓게 조망한 초기 review로서 의미가 크다. 저자들은 semantic segmentation 문제 정의, 대표 backbone network, transfer learning과 augmentation 같은 배경, 2D/RGB-D/3D dataset, FCN 이후 주요 method family, 그리고 benchmark 결과와 future research direction까지 하나의 서사로 묶어 제시한다.

논문이 내리는 핵심 결론은 다음과 같다. 2D RGB semantic segmentation에서는 DeepLab이 가장 안정적이고 강력한 방법으로 보이며, context modeling은 성능 향상에 핵심적이다. RGB-D에서는 recurrent fusion 계열이 유망하며, 3D에서는 PointNet이 비정형 point set 처리의 새로운 방향을 열었다. video segmentation은 아직 초기 단계지만 Clockwork FCN과 3D convolution 계열이 가능성을 보인다.

실제 적용 측면에서도 이 논문은 의미가 있다. autonomous driving, robotics, AR/VR, indoor scene understanding 같은 분야에서 semantic segmentation은 핵심 기술이며, 이 논문은 어떤 데이터와 모델이 어떤 문제에 맞는지 판단하는 기준을 제공한다. 향후 연구 측면에서는 대규모 3D/sequence dataset, graph 기반 point cloud 처리, real-time inference, memory-efficient model, temporal coherence, multi-view integration이 중요하다고 전망한다. 즉, 이 논문은 당시까지의 성과를 정리하는 데 그치지 않고, 이후 semantic segmentation 연구가 어디로 확장될지를 보여주는 지도로 읽을 수 있다.
