# Learning Rich Features from RGB-D Images for Object Detection and Segmentation

- **저자**: Saurabh Gupta, Ross Girshick, Pablo Arbelaez, Jitendra Malik
- **발표연도**: 2014
- **arXiv**: https://arxiv.org/abs/1407.5736

## 1. 논문 개요

이 논문은 RGB 이미지와 depth 이미지를 함께 사용하는 RGB-D 환경에서, 단순한 semantic segmentation을 넘어 object detection, instance segmentation, semantic segmentation을 하나의 통합된 파이프라인으로 개선하는 것을 목표로 한다. 특히 저자들은 RGB-D 데이터에서 depth를 어떻게 표현해야 CNN이 더 잘 학습할 수 있는지에 초점을 맞추고, 이를 기반으로 기존 R-CNN을 RGB-D 입력에 맞게 확장한다.

연구 문제는 크게 세 가지다. 첫째, RGB-D 이미지에서 객체를 더 정확히 검출할 수 있는가. 둘째, 검출된 객체에 대해 bounding box 수준이 아니라 pixel-level instance mask를 구할 수 있는가. 셋째, 이런 객체 검출 결과를 semantic segmentation에도 도움이 되게 활용할 수 있는가. 저자들은 이 세 문제를 분리된 작업으로 보지 않고, 서로 연결된 scene understanding 문제로 본다.

이 문제가 중요한 이유는 실내 장면 이해에서 단순히 “이 픽셀은 chair다”라고 아는 것만으로는 부족하고, 반대로 “여기 chair bounding box가 있다”는 정보만으로도 실제 응용에는 부족하기 때문이다. 예를 들어 로봇이 물체를 집거나 피해야 할 때는 객체의 위치뿐 아니라 실제 object extent와 scene surface의 의미를 함께 알아야 한다. 논문은 이런 점에서 RGB-D perception을 더 실용적인 수준으로 끌어올리려는 시도라고 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 depth를 raw depth map 그대로 CNN에 넣는 대신, 장면의 geocentric pose를 드러내는 표현으로 바꾸면 훨씬 더 풍부한 특징을 학습할 수 있다는 것이다. 저자들은 각 픽셀을 세 채널로 표현하는 HHA encoding을 제안한다. 여기서 HHA는 horizontal disparity, height above ground, angle with gravity를 뜻한다. 즉, 단순히 “얼마나 멀리 있는가”만이 아니라, “바닥에서 얼마나 높은가”, “중력 방향과 어떤 각도를 이루는가”를 함께 표현한다.

이 설계의 직관은 실내 장면에는 강한 기하학적 규칙이 있다는 점이다. 예를 들어 floor는 대체로 아래쪽에 있고, wall은 중력과 거의 수직이며, table top은 위를 향한 면이다. raw depth는 이런 구조를 직접 드러내지 않지만 HHA는 그것을 더 직접적으로 encode한다. 저자들은 제한된 학습 데이터만으로 CNN이 이런 기하 정보를 raw depth로부터 스스로 학습하기는 어렵다고 본다.

기존 접근과의 차별점은 두 가지다. 첫째, depth를 CNN에 넣는 방식에서 단순한 RGB-D 4채널 처리보다 더 구조적인 HHA 표현을 사용했다. 둘째, object detection만 한 것이 아니라, contour detection, 2.5D region proposals, RGB-D detector, instance segmentation, semantic segmentation까지 전체 시스템을 연결했다. 논문에서 제시된 결과를 보면, 이 차별점은 단순한 아이디어 제안에 그치지 않고 실제 성능 향상으로 이어졌다.

## 3. 상세 방법 설명

전체 시스템은 다음과 같은 흐름으로 구성된다. RGB와 depth 이미지 쌍이 입력으로 들어오면, 먼저 RGB-D contour detection을 수행하고, 이를 이용해 2.5D region proposals를 생성한다. 그 다음 proposal마다 RGB CNN feature와 depth CNN feature를 추출해 object detector를 돌린다. 이후 검출된 box에 대해 instance segmentation mask를 예측하고, 마지막으로 object detection 출력을 semantic segmentation용 superpixel classifier의 추가 feature로 사용한다.

### 3.1 RGB-D contour detection과 2.5D region proposals

기존 RGB-D contour detection 연구 두 가지의 장점을 결합한다. 하나는 Gupta et al. [18]의 normal gradient와 depth gradient 기반 접근이고, 다른 하나는 Dollár et al. [9]의 structured random forest 기반 contour detector다. 저자들은 structured forest의 학습 틀 위에 다음 정보를 추가한다.

첫째, normal gradients를 두 스케일에서 계산한다. 이는 국소 평면을 추정해 convex/concave한 normal 변화나 surface orientation 변화를 더 잘 잡게 한다. 둘째, geocentric pose feature인 pixel-wise height above ground와 angle with gravity를 추가한다. 이는 예를 들어 floor 위의 밝기 변화는 contour로 중요하지 않을 수 있다는 식의 실내 장면 규칙을 모델이 배우게 한다. 셋째, RGB edge detector의 soft edge map을 넣어 appearance generalization을 보완한다.

이 contour를 기반으로 region proposal을 만들 때도 RGB 전용 MCG를 RGB-D로 확장한다. proposal ranking을 위해 기존 2D shape/color feature 외에 depth 기반 기하학 feature를 추가한다. 논문에 따르면 추가 feature는 총 29개이며, 예를 들어 region 내부의 disparity, height, angle with gravity, world coordinates $(X,Y,Z)$의 평균과 표준편차, region의 3D extent, 최소/최대 높이, vertical/upward/downward surface 비율 등이 포함된다.

region proposal 품질 평가는 class-wise average Jaccard coverage를 사용한다. 논문은 이를 다음과 같이 정의한다.

$$
\mathrm{coverage}(K)=\frac{1}{C}\sum_{i=1}^{C}\left(\frac{1}{N_i}\sum_{j=1}^{N_i}\max_{k\in[1...K]} O\!\left(R^{\,l(i,j)}_k, I^i_j\right)\right)
$$

여기서 $C$는 클래스 수, $N_i$는 클래스 $i$의 instance 수, $O(a,b)$는 IoU, $I^i_j$는 클래스 $i$의 $j$번째 ground-truth region, $R^l_k$는 이미지 $l$에서 $k$번째 ranked proposal이다. 의미는 간단하다. 각 ground-truth instance가 상위 $K$개 proposal 중 하나에 의해 얼마나 잘 덮이는지를 클래스별로 평균낸 것이다.

### 3.2 RGB-D object detection과 HHA encoding

저자들은 R-CNN을 RGB-D에 맞게 일반화한다. 기본 구조는 proposal을 만든 뒤 각 proposal crop에 CNN feature를 뽑고, linear SVM으로 분류하는 방식이다. 핵심은 depth 입력을 어떻게 표현하느냐이다.

제안한 HHA encoding은 각 픽셀을 세 개 값으로 표현한다.

- $d$: horizontal disparity
- $h$: height above ground
- $\theta$: local surface normal이 gravity direction과 이루는 각

이 세 채널은 학습 데이터에서 관측된 값 범위를 $0$에서 $255$로 선형 스케일링한다. 저자들의 논리는 명확하다. disparity는 깊이 경계를, angle with gravity는 표면 방향 변화를, height는 물체의 장면 내 절대적 위치 힌트를 준다. 이 세 정보는 서로 보완적이며, 실내 장면의 구조를 raw depth보다 더 직접적으로 표현한다.

CNN 구조는 Krizhevsky et al.의 AlexNet 계열을 사용하며, 약 6천만 개 파라미터를 가진 ImageNet pre-trained network에서 시작한다. 논문의 중요한 주장은 “RGB용으로 설계된 CNN도 HHA 이미지에 대해 의미 있는 표현을 학습할 수 있다”는 것이다. 저자들은 HHA에서도 edge와 shape discontinuity가 강하게 나타나므로 RGB pretraining이 완전히 무관하지 않다고 본다.

학습은 두 단계다. 먼저 CNN fine-tuning을 한다. 학습률은 $0.001$에서 시작해 20k iteration마다 10배씩 줄이고, 총 30k iteration 동안 학습한다. detection fine-tuning 시 proposal이 어떤 ground-truth와 가장 많이 겹치고 그 overlap이 $0.5$보다 크면 그 클래스로, 아니면 background로 라벨링한다.

그 다음 linear SVM을 학습한다. positive는 해당 클래스의 ground-truth box, negative는 그 클래스 ground-truth와 IoU가 $0.3$ 미만인 box다. feature는 주로 `fc6`를 사용했고, 최종 테스트 시에도 `fc6` feature + linear classifier + non-maximum suppression으로 sparse detections를 얻는다.

### 3.3 Synthetic data augmentation

NYUD2는 PASCAL VOC보다 학습 이미지가 훨씬 적기 때문에, 저자들은 synthetic data augmentation을 사용한다. 선택한 방법은 NYUD2의 3D scene annotation을 이용해 새로운 시점에서 scene을 렌더링하는 것이다. 그리고 Kinect quantization을 모사하기 위해 depth를 quantized disparity로 바꾸고 저해상도 white noise를 더한다.

논문은 2배 synthetic data는 도움이 되지만, 15배까지 늘리면 오히려 약간 성능이 떨어진다고 보고한다. 저자들은 그 이유를 synthetic bias로 해석한다. 특히 [17]의 annotation이 비가구 물체를 cuboid로 치환해 scene 통계가 원본과 달라졌기 때문이라고 설명한다. 즉, augmentation이 무조건 많다고 좋은 것이 아니라, realism이 중요하다는 점을 보여준다.

### 3.4 Instance segmentation

instance segmentation은 detection box 내부 픽셀을 foreground/background로 나누는 문제로 정식화된다. 학습 시에는 ground-truth instance마다 detector가 낸 detection 중 IoU가 70% 이상인 최고 점수 detection을 하나 고른다. 그 detection window 안의 ground-truth mask를 $50 \times 50$ grid로 warp하고, 각 위치를 학습 샘플로 쓴다.

저자들은 $2500$개 위치 각각에 별도 classifier를 두는 대신, 하나의 random forest가 모든 위치를 처리하게 한다. 이유는 location-wise classifier는 데이터가 너무 부족하고, 단일 선형 모델은 충분히 유연하지 않기 때문이다. 최종적으로 10개의 decision tree를 가진 forest를 사용한다.

feature는 원본 이미지에서 다양한 채널을 계산한 뒤 detection window로 crop하고 $50 \times 50$으로 warp해서 쓴다. split node의 질문은 두 종류다.

- unary question: 특정 채널의 한 위치 값이 threshold보다 큰가
- binary question: 같은 채널의 두 위치 값 차이가 threshold보다 큰가

이는 Shotton et al.의 depth decision forest 스타일을 따른다. 테스트 때는 detection마다 forest를 통과시켜 $50 \times 50$ foreground confidence map을 얻고, 이를 원래 box로 unwarp한 뒤 superpixel 단위로 평균화해 부드럽게 만든다. 마지막 binary threshold는 validation set에서 최적화한다.

### 3.5 Semantic segmentation

semantic segmentation은 이전 연구 [18]의 superpixel classification framework를 기반으로 한다. 이 논문에서 새로 하는 일은 object detector의 출력을 superpixel feature에 추가하는 것이다. 즉, 어떤 superpixel 위를 어떤 category detection이 얼마나 겹치는지 같은 정보를 feature로 넣어 semantic segmentation을 보강한다. 이는 “thing” detection 정보가 “stuff” labeling에도 유용하다는 아이디어다.

## 4. 실험 및 결과

실험은 NYUD2 데이터셋에서 수행되었다. 데이터 분할은 총 795개 train 이미지와 654개 test 이미지이며, train 쪽은 다시 381개 train과 414개 val로 나누었다고 명시한다. 같은 scene의 이미지가 서로 다른 split에 섞이지 않도록 구성했다.

### 4.1 Contour detection과 region proposals

contour detection에서는 standard maximum F-measure인 $F_{\max}$를 사용한다. 결과를 보면 기존 RGB 기반 `gPb-ucm`의 ODS $F_{\max}$는 63.15이고, Gupta et al. [18]는 68.66, SE [9]는 68.45이다. 여기에 normal gradient를 추가하면 69.55, 모든 cue를 추가하면 70.25가 된다. 더 강한 SE+SH [10] 버전에서는 69.46에서 71.03으로 올라간다.

이 결과는 두 가지를 시사한다. 첫째, normal gradient는 모든 recall 구간에서 precision을 안정적으로 올려준다. 둘째, geocentric pose와 appearance cue를 더하면 추가 향상이 난다. 논문은 최종 contour detector가 당시 state of the art보다 약 1.5%p 높은 $F_{\max}$를 달성했다고 주장한다.

region proposal에서는 RGB contour 대신 RGB-D contour를 쓰는 것 자체가 큰 향상을 주고, proposal re-ranking에 depth geometry feature를 더하면 작지만 일관된 추가 향상이 있었다. Lin et al. [29]와의 비교에서도, 그 방법이 특정 클래스에 맞춰 학습되었음에도 불구하고 저자들의 보다 generic한 proposal 방식이 더 낫다고 보고한다.

### 4.2 Object detection

box detection 평가는 PASCAL VOC 스타일 average precision인 $AP^b$를 사용한다. validation set 실험에서 가장 중요한 비교는 다음과 같다.

RGB DPM은 mean $AP^b=8.4\%$로 매우 낮다. RGBD-DPM은 21.7%로 크게 오른다. RGB R-CNN은 fine-tuning 전 16.4%, fine-tuning 후 19.7%다. depth만 넣되 disparity를 그대로 쓴 CNN은 fine-tuning 후 20.1%다. 하지만 HHA를 쓰면 25.2%로 크게 오른다. 여기에 2배 synthetic data를 더하면 26.1%가 된다. 최종적으로 RGB와 HHA feature를 결합하면 32.5%까지 올라간다.

이 결과는 논문의 핵심 주장을 직접 뒷받침한다. raw disparity보다 HHA가 훨씬 낫고, RGB와 HHA는 상호보완적이다. 또한 `pool5`나 `fc7`보다 `fc6`가 가장 좋았는데, 저자들은 특히 `fc7`의 성능 저하를 데이터 부족 상황에서의 overfitting으로 해석한다.

테스트 세트에서는 최종 시스템이 mean $AP^b=37.3\%$를 기록한다. 표에 따르면 RGB DPM은 9.0%, RGBD-DPM은 23.9%, RGB R-CNN은 22.5%, 제안 방법은 37.3%다. 논문 초록의 “56% relative improvement”는 이 37.3%가 기존 강한 baseline 23.9% 대비 크게 향상되었다는 의미로 이해할 수 있다.

카테고리별로도 bed 71.0, chair 43.3, sofa 53.9, lamp 39.4, night-stand 40.0 등 주요 실내 가구류에서 강한 성능을 보인다. 반면 box 같은 클래스는 1.4로 매우 낮다. 이는 작은 물체이거나 정의가 불안정한 클래스에서는 여전히 어려움이 있음을 보여준다.

### 4.3 Instance segmentation

instance segmentation 평가는 region detection average precision인 $AP^r$를 사용한다. 이것은 bounding box IoU 대신 region IoU를 쓰는 AP다. baseline은 세 가지다. `box`는 box 전체를 mask라고 가정하고, `region`은 proposal region을 평균하고, `fg mask`는 훈련 마스크의 empirical average를 사용한다.

결과는 mean $AP^r$에서 `box` 14.0, `region` 28.1, `fg mask` 28.0, 제안 방법 32.1이다. 즉, 제안한 random forest 기반 mask predictor가 가장 좋다. 특히 일부 클래스에서는 $AP^r$가 $AP^b$보다 더 높은 경우가 있는데, 이는 instance segmentation이 box localization error를 어느 정도 보정했다는 뜻이다. 논문이 단순히 detection 후처리를 한 것이 아니라, 실질적으로 region quality를 향상시켰음을 보여주는 지점이다.

### 4.4 Semantic segmentation

semantic segmentation에서는 `fwavacc`, `avacc`, 그리고 detector를 추가한 클래스들만의 평균인 `avacc*`를 보고한다. 이전 [18]의 `fwavacc`는 45.2였고, RGBD-DPM detector feature를 추가하면 45.6, 제안 detector feature를 넣으면 47.0이 된다.

더 중요한 수치는 `avacc*`다. detector를 추가한 클래스들만 보면 [18]의 28.4에서 DPM 추가 시 31.0, 제안 방법 추가 시 35.1로 올라간다. 논문은 이것을 24% relative improvement라고 해석한다. 즉, object detector의 품질이 semantic segmentation에도 직접 영향을 미친다는 점이 실험적으로 확인되었다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 depth를 표현하는 방식 자체를 문제 삼고, HHA라는 매우 직관적이면서도 강력한 encoding을 제안했다는 점이다. 단순히 네트워크 구조를 바꾸기보다, 입력 표현을 geometry-aware하게 바꿈으로써 적은 데이터에서도 CNN이 강한 표현을 학습하도록 만들었다. 실험도 이 주장을 잘 뒷받침한다. disparity만 사용할 때보다 HHA가 훨씬 좋고, RGB와 HHA를 결합하면 추가 이득이 분명하다.

또 다른 강점은 시스템 수준의 통합성이다. contour, proposal, detection, instance segmentation, semantic segmentation이 서로 느슨하게 연결된 것이 아니라, 앞 단계의 개선이 뒤 단계의 성능 향상으로 이어진다. 특히 detector를 semantic segmentation feature로 넣어 성능을 끌어올린 부분은, instance-level understanding과 pixel-level labeling을 연결하는 설계로서 의미가 있다.

실험 설계도 비교적 설득력 있다. DPM, RGB R-CNN, raw disparity, HHA, synthetic augmentation, fusion 방식, 서로 다른 CNN layer까지 폭넓은 ablation을 수행했다. 따라서 어떤 요소가 실제로 성능 향상에 기여했는지 비교적 명확히 드러난다.

한편 한계도 있다. 첫째, 데이터셋이 NYUD2 실내 장면에 한정되어 있어, 제안 방법이 실외나 다른 센서 환경에서도 동일하게 잘 작동하는지는 이 논문만으로는 알 수 없다. 둘째, synthetic augmentation이 많아질수록 오히려 성능이 떨어졌는데, 이는 제안된 파이프라인이 데이터 realism의 영향을 크게 받음을 의미한다. 셋째, instance segmentation은 detection 결과에 강하게 의존하므로, detector가 놓친 객체는 복구할 수 없다.

또한 논문은 instance segmentation에 사용한 “feature channels”의 전체 목록은 supplementary material에 있다고만 하고, 본문에는 자세히 다 적지 않는다. 따라서 본문만으로는 segmentation feature engineering의 전부를 재현하기 어렵다. semantic segmentation에서 detector-derived feature의 정확한 형태 역시 본문 설명은 다소 요약적이다. 이 부분은 논문에 명시되지 않은 세부 구현이 존재할 가능성이 있다.

비판적으로 보면, HHA는 실내 기하 priors가 강한 문제에서 매우 잘 맞는 설계이지만, 입력 표현에 도메인 지식을 강하게 주입한 방식이기도 하다. 따라서 대규모 데이터나 더 강한 end-to-end 학습이 가능해지면, 이런 hand-designed representation의 상대적 이점은 줄어들 수 있다. 그러나 논문이 다루는 당시의 데이터 규모와 모델 조건에서는 이 선택이 매우 합리적이고 효과적이었다고 평가할 수 있다.

## 6. 결론

이 논문의 핵심 기여는 RGB-D 인식에서 depth를 CNN이 학습하기 좋은 형태로 바꾼 HHA encoding을 제안하고, 이를 바탕으로 object detection, instance segmentation, semantic segmentation을 모두 향상시킨 점이다. 특히 RGB-D object detection에서 mean $AP^b=37.3\%$를 달성해 기존 강한 baseline 대비 큰 상대 향상을 보였고, instance segmentation과 semantic segmentation에서도 일관된 개선을 확인했다.

실제로 이 연구는 이후 RGB-D perception과 3D scene understanding 분야에서 매우 중요한 전환점으로 볼 수 있다. depth를 단순 보조 채널로 다루지 않고, 장면의 기하 구조를 반영한 표현으로 바꾸는 발상이 효과적임을 보여주었기 때문이다. 로보틱스나 실내 자율 시스템처럼 객체의 위치와 형태를 정확히 이해해야 하는 응용에서, 이런 접근은 매우 실질적인 가치를 가진다. 동시에 이 논문은 limited-data 환경에서 representation design이 얼마나 중요한지를 잘 보여주는 사례이기도 하다.
