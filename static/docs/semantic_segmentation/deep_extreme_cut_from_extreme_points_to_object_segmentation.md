# Deep Extreme Cut: From Extreme Points to Object Segmentation

- **저자**: K.-K. Maninis, Sergi Caelles, Jordi Pont-Tuset, Luc Van Gool
- **발표연도**: 2017 (arXiv 최초 공개 기준)
- **arXiv**: https://arxiv.org/abs/1711.09081

## 1. 논문 개요

이 논문은 사용자가 객체의 네 개 extreme points, 즉 가장 왼쪽, 가장 오른쪽, 가장 위, 가장 아래의 점만 클릭하면, 이를 바탕으로 정확한 객체 분할 mask를 생성하는 방법인 **DEXTR (Deep Extreme Cut)** 를 제안한다. 핵심 목표는 기존의 bounding box 기반 또는 dense mask 기반 annotation/segmentation 파이프라인보다 더 적은 사용자 입력으로 더 정확한 segmentation을 얻는 것이다.

연구 문제는 크게 두 가지 층위에서 중요하다. 첫째, segmentation 모델의 학습에는 픽셀 단위의 정답 마스크가 필요하지만, 이런 dense annotation은 매우 비싸고 시간이 많이 든다. 둘째, test time에 사람이 개입하는 semi-automatic segmentation이나 interactive segmentation에서도, 사용자가 bounding box를 정교하게 그리거나 여러 scribble을 입력하는 것은 부담이 크다. 논문은 이 문제를 “객체의 네 극점만 클릭하게 하자”는 방식으로 단순화한다.

이 접근의 중요성은 입력 효율과 성능을 동시에 잡으려는 데 있다. 논문에 따르면 extreme clicking은 기존 bounding box annotation보다 훨씬 빠르며, 동시에 bounding box보다 더 풍부한 경계 정보를 제공한다. 저자들은 이 정보를 CNN 입력에 직접 넣어 instance segmentation, annotation generation, video object segmentation, interactive segmentation까지 하나의 통합된 방식으로 다룬다.

## 2. 핵심 아이디어

중심 아이디어는 단순하다. 객체의 extreme points 네 개는 bounding box보다 더 풍부한 위치 및 경계 정보를 제공한다. DEXTR는 이 점들을 단순한 좌표로 처리하지 않고, 각 점 중심에 2D Gaussian을 둔 heatmap으로 변환한 뒤 RGB 이미지와 concat하여 CNN의 입력으로 사용한다. 즉, 입력은 일반적인 3채널 이미지가 아니라 **RGB + point heatmap**의 4채널 이미지가 된다.

이 설계의 직관은 명확하다. 네 extreme points는 객체의 외곽에 실제로 닿아 있으므로, 모델은 “어떤 객체를 분할해야 하는가”를 bounding box만 볼 때보다 훨씬 잘 이해할 수 있다. 또한 이 점들로부터 tight bounding box를 유도할 수 있으므로, 객체 주변 영역만 crop하여 처리할 수 있고, 이는 배경 잡음을 줄이고 스케일 변동 문제도 완화한다.

기존 접근과의 차별점은 두 가지다. 하나는 Papadopoulos et al.의 extreme clicking이 bounding box 품질 향상이나 GrabCut류의 전처리 개선에 주로 쓰였다면, 이 논문은 extreme points 자체를 **deep network의 직접 입력 신호**로 사용했다는 점이다. 다른 하나는 기존 interactive segmentation 계열이 positive/negative clicks나 distance transform 기반 입력을 쓰는 경우가 많았는데, DEXTR는 extreme points라는 구조화된 경계 단서를 활용해 더 강한 guided segmentation 성능을 보였다는 점이다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 사용자가 객체의 top, bottom, left-most, right-most 점을 클릭하면, 각 점 위치에 Gaussian을 놓아 하나의 heatmap을 만든다. 이 heatmap을 원본 RGB 이미지와 채널 방향으로 결합하여 4채널 입력을 만든다. 그리고 extreme points로부터 얻은 bounding box를 기준으로 객체 중심 crop을 만들되, 약간의 margin을 추가해 문맥 정보를 포함시킨다. 이후 이 crop을 CNN에 넣어 foreground/background binary mask를 예측한다.

백본 네트워크는 **ResNet-101** 이다. 저자들은 dense prediction을 위해 마지막 fully connected layer를 제거하고, 마지막 두 stage의 max pooling을 없애며, receptive field를 유지하기 위해 atrous convolution을 도입했다. 또한 최종 feature map에 global context를 결합하기 위해 **Pyramid Scene Parsing (PSP) module** 을 붙였다. 즉, 구조적으로는 segmentation용으로 변형한 ResNet-101 위에 PSP head를 얹은 형태다.

출력은 각 픽셀이 객체에 속할 확률을 나타내는 probability map이다. 학습 목표는 foreground/background binary segmentation이며, 손실 함수는 class imbalance를 고려한 **balanced cross-entropy** 이다. 논문은 이를 다음처럼 쓴다.

$$
L = \sum_{j \in Y} w_{y_j} \, C(y_j, \hat{y}_j), \quad j \in 1, \ldots, |Y|
$$

여기서 $y_j \in \{0,1\}$ 는 픽셀 $j$의 정답 라벨이고, $\hat{y}_j$ 는 예측값이다. $C(\cdot)$ 는 일반적인 cross-entropy를 뜻한다. $w_{y_j}$ 는 해당 클래스의 minibatch 내 빈도의 역수에 비례하는 가중치다. 쉽게 말하면, foreground와 background 픽셀 수가 불균형할 때 적게 등장하는 쪽의 오류를 더 크게 반영하도록 만든 것이다. 저자들은 object-centered crop을 쓰더라도 foreground/background 비율이 완전히 균형적이지 않기 때문에 이 balanced loss가 도움이 된다고 설명한다.

입력 표현 측면에서 중요한 점은, 극점 정보를 distance transform으로 넣는 대신 **fixed Gaussian heatmap** 으로 넣는 것이 더 좋았다는 것이다. 논문 실험에서는 distance map 대비 fixed Gaussian이 더 나은 성능을 냈다. 저자들의 해석은 extreme points처럼 의미가 분명한 점 입력에서는, 점의 위치를 직접 강하게 강조하는 representation이 더 적합하다는 것이다.

DEXTR의 활용 방식은 네 가지로 확장된다. 첫째, **class-agnostic instance segmentation** 으로서 어떤 클래스든 extreme points만 주면 그 객체 mask를 예측한다. 둘째, **annotation 도구** 로서 사람이 polygon이나 dense mask 대신 extreme points만 주고 자동 생성된 mask를 annotation으로 사용한다. 셋째, **video object segmentation** 에서 첫 프레임 등의 GT mask 대신 DEXTR가 만든 mask를 사용해 후속 비디오 분할 모델을 학습 또는 fine-tuning한다. 넷째, **interactive segmentation** 에서는 initial 4 points로 나온 결과가 만족스럽지 않을 때, 오류 경계 근처에 추가 점 하나를 더 받아 refinement를 수행한다.

interactive 설정에서의 학습은 특히 흥미롭다. 논문은 먼저 4-point DEXTR를 학습한 뒤, IoU가 $0.8$ 미만인 어려운 예제들만 골라 오류 영역에 다섯 번째 점을 추가하는 시뮬레이션을 한다. 이후 이 hard examples를 이용해 5-point 입력으로 추가 학습한다. 저자들은 이를 일종의 **Online Hard Example Mining (OHEM)** 관점으로 해석한다. 즉, 사용자의 추가 클릭은 단순 입력 증가가 아니라 어려운 예제에 학습을 집중시키는 역할도 한다.

## 4. 실험 및 결과

실험은 PASCAL VOC, COCO, GrabCut, DAVIS 2016, DAVIS 2017에서 수행되었다. 기본적으로 PASCAL 2012 segmentation에 SBD를 추가한 데이터로 학습한 모델을 많이 사용했고, 경우에 따라 COCO 학습도 수행했다. PASCAL은 100 epochs, COCO는 10 epochs 학습했다. learning rate는 $10^{-8}$, momentum은 $0.9$, weight decay는 $5 \times 10^{-4}$ 이다. PASCAL 학습은 Titan X GPU 기준 약 20시간, COCO는 약 5일이 걸렸고, 테스트는 약 80ms로 빠르다고 보고한다.

ablation study는 DEXTR의 각 설계 선택이 실제로 유효한지 보여준다. 먼저 region-based 구조보다 Deeplab-v2 계열 fully convolutional 구조가 IoU 기준 **+3.9%** 더 좋았다. 이는 $28 \times 28$ 수준의 저해상도 mask 예측이 세밀한 object mask에는 불리하다는 해석과 맞닿아 있다. bounding box만 입력으로 쓰는 것보다 extreme points heatmap을 추가하면 **+3.1%** 향상되었고, 일반 cross-entropy 대신 balanced loss를 쓰면 **+3.3%** 향상되었다. 전체 이미지를 처리하는 대신 object crop을 쓰면 **+7.9%** 로 가장 큰 개선이 있었는데, 이는 작은 객체와 스케일 변화 문제 완화에 특히 효과적이었다. ASPP 대신 PSP module을 쓰면 **+2.3%** 좋아졌고, fixed Gaussian은 distance map보다 **1.3%** 좋았다.

누적 성능을 보면, crop 기반 Deeplab-v2에 PSP를 붙이고 extreme points를 추가한 뒤 SBD 데이터까지 넣었을 때 PASCAL VOC 2012 validation에서 **91.5% IoU** 에 도달했다. 논문은 이 중에서도 extreme points 자체가 핵심 성능 향상을 이끄는 중요한 요소라고 강조한다.

class-agnostic instance segmentation에서는 PASCAL과 GrabCut에서 모두 강한 결과를 보였다. PASCAL EXT에서 DEXTR는 **80.1% IoU** 를 기록했는데, Papadopoulos et al.의 extreme-point 기반 GrabCut 방식 **73.6%** 보다 **+6.5%** 높다. SharpMask 기반 bounding-box proposal 선택 방식은 **69.3%**, oracle upper bound도 **78.0%** 로 DEXTR보다 낮았다. 즉, 단순 proposal 매칭보다 “extreme points를 직접 조건으로 쓰는 CNN”이 더 정밀하다는 점을 보여준다.

GrabCut 데이터셋에서는 error rate 기준 **2.3%** 를 달성해 당시 비교 대상 중 최고 성능을 기록했다. 직전 경쟁 방법인 DeepGC가 **3.4%**, BoxPrior는 **3.7%**, 기존 GrabCut은 **8.1%** 였다. 논문은 이를 약 32%의 상대적 개선이라고 해석한다.

일반화 성능도 점검했다. PASCAL로 학습한 모델을 COCO mini-val에 적용했을 때, PASCAL 클래스만 있는 경우와 없는 unseen categories만 있는 경우의 성능이 거의 같았다. 이는 이 모델이 semantic class label이 아니라 “주어진 extreme points에 대응하는 objectness”를 배우고 있음을 시사한다. 또한 dataset 간 generalization도 양호해서, PASCAL로 학습 후 COCO 테스트, COCO로 학습 후 PASCAL 테스트 모두 큰 성능 저하 없이 작동했다.

annotation use case에서는 DEXTR가 특히 실용적이다. 저자들은 COCO로 학습한 DEXTR를 이용해 PASCAL train set의 인스턴스 마스크를 생성하고, 이 자동 생성 마스크를 semantic segmentation 네트워크의 학습 데이터로 사용했다. 결과적으로, 같은 annotation budget 기준으로 GT mask를 수작업으로 모은 경우보다 훨씬 높은 성능을 보였다. 예를 들어 논문은 **약 7분의 annotation time으로 70% mIoU** 에 도달하는 반면, 같은 시간 예산에서 GT 기반은 **46%** 수준이라고 설명한다. 반대로 같은 70% 성능을 얻는 데 GT는 **1시간 10분** 정도가 필요하다고 서술한다. 이미지 수를 동일하게 맞추면 DEXTR 기반 annotation은 GT 기반과 거의 같은 성능까지 간다. 즉, 품질 손실이 매우 작으면서 annotation cost를 크게 줄일 수 있다는 것이다.

video object segmentation에서는 OSVOS를 기반 모델로 사용하여, GT 마스크 대신 DEXTR 마스크를 주었을 때의 성능을 비교했다. DAVIS 2016에서는 **GT 1개 마스크를 쓰는 것과 비슷한 성능을 약 5배 작은 annotation budget** 으로 달성했다. 다만 GT 마스크가 여러 장 주어지는 상황에서는, DEXTR로 더 많은 프레임을 빠르게 annotation할 수 있어도 완전히 같은 수준까지는 가지 못했다. 저자들은 그 이유를 DAVIS 2016의 마스크가 종종 여러 semantic instance를 함께 포함하는 반면, extreme points는 하나의 global set만 주어져 모호성이 생기기 때문이라고 본다. 이를 확인하기 위해 단일 인스턴스에 더 가까운 DAVIS 2017에서도 실험했는데, 여기서는 GT와의 격차가 더 작아졌다.

interactive segmentation에서는 4 clicks만으로도 강한 성능을 보였다. PASCAL에서 IoU 85%에 도달하는 데 필요한 클릭 수가 **4.0회** 였고, GrabCut에서 IoU 90%에 도달하는 데도 **4.0회** 였다. 비교 방법인 RIS-Net은 각각 5.7회, 6.0회, iFCN은 8.7회, 7.5회가 필요했다. 또한 4 clicks 기준 성능 자체도 PASCAL **91.5%**, GrabCut **94.4%** 로 가장 높았다. 논문은 extreme points가 일반적인 positive/negative clicks보다 더 구조적이고 정보량이 큰 supervision이라는 점을 시사한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 입력 효율성과 정확도를 동시에 만족시키는 설계를 매우 깔끔하게 제시했다는 점이다. 사용자는 네 점만 클릭하면 되고, 모델은 이를 dense mask로 바꾼다. 그리고 이 단순한 메커니즘이 instance segmentation, annotation generation, video object segmentation, interactive segmentation 등 여러 문제에 일관되게 적용된다. 보통 특정 task에만 맞는 방법이 많은데, DEXTR는 입력 표현 자체가 범용적이라 활용 범위가 넓다.

또 다른 강점은 ablation이 비교적 충실하다는 점이다. 저자들은 backbone 형태, crop 여부, loss balancing, PSP vs ASPP, extreme points vs bounding box, Gaussian vs distance map 등 핵심 설계 요소를 각각 분리해서 검증했다. 따라서 “왜 이 구조가 좋은가”에 대한 논문의 주장에는 실험적 근거가 있다.

실용성도 높다. annotation 시간 절감 효과가 매우 크고, 실제로 자동 생성 마스크로 학습한 segmentation 모델이 GT 기반과 거의 동등한 성능에 접근한다는 결과는 산업적 측면에서도 의미가 크다. 단순히 mask 품질이 좋다고 주장하는 데 그치지 않고, 그 마스크를 써서 후속 모델을 학습했을 때도 성능이 유지된다는 점이 중요하다.

한계도 있다. 첫째, DEXTR는 강한 supervision, 즉 ground-truth mask로 학습된 모델이다. 따라서 test-time interaction은 적지만, training 자체가 완전한 weak supervision은 아니다. 둘째, 논문은 extreme points가 하나의 객체를 잘 지정하는 상황에서는 강하지만, 한 mask 안에 여러 semantic instance가 섞여 있거나 객체 정의가 애매한 경우에는 한계가 있음을 DAVIS 2016 결과에서 사실상 인정한다. 셋째, interactive setting에서 다섯 번째 점을 어떻게 제시할지에 대한 실험은 시뮬레이션 기반이며, 실제 사용자 행동과 완전히 같다고 보기는 어렵다.

또한 논문은 매우 강한 수치 결과를 제시하지만, 일부 비교는 실험 프로토콜 차이나 학습 데이터 규모 차이에 민감할 수 있다. 예를 들어 PASCAL EXT와 PASCAL validation에서의 성능 차이에 대해 저자들이 직접 별도 설명을 달아둔 것도 이런 맥락 때문이다. 따라서 결과를 읽을 때는 어떤 데이터로 pre-train/fine-tune했는지 주의해서 봐야 한다.

## 6. 결론

이 논문은 extreme points라는 매우 저렴하고 직관적인 사용자 입력을 CNN의 직접적인 조건 정보로 넣어, 정확한 object mask를 생성하는 DEXTR를 제안했다. 핵심 기여는 단순히 “네 점으로 segmentation 가능”이라는 아이디어 자체가 아니라, 이를 heatmap 기반 4채널 입력, object-centered crop, ResNet-101 + PSP, balanced loss와 결합해 실제로 강력한 성능을 내는 시스템으로 완성했다는 데 있다.

논문이 보여준 바에 따르면, DEXTR는 class-agnostic instance segmentation에서 높은 정확도를 내고, annotation 비용을 크게 줄이며, video object segmentation과 interactive segmentation에도 효과적으로 쓰일 수 있다. 특히 “적은 인간 입력으로 dense mask를 얻고, 그 결과가 후속 모델 학습에도 충분히 쓸 만하다”는 점은 실제 annotation pipeline과 human-in-the-loop vision system 설계에 중요한 의미를 가진다. 향후 연구에서도 이런 형태의 구조화된 사용자 입력을 deep model의 조건 정보로 통합하는 방향은 계속 유효할 가능성이 크다.
