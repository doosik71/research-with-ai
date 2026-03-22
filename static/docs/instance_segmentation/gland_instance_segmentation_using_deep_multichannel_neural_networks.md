# Gland Instance Segmentation Using Deep Multichannel Neural Networks

* **저자**: Yan Xu, Yang Li, Yipei Wang, Mingyuan Liu, Yubo Fan, Maode Lai, Eric I-Chao Chang
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1611.06661](https://arxiv.org/abs/1611.06661)

## 1. 논문 개요

이 논문은 colon histology image에서 **개별 gland를 각각 구분하여 분할하는 gland instance segmentation** 문제를 다룬다. 일반적인 semantic segmentation은 각 픽셀이 gland인지 background인지만 판단하지만, 이 논문이 다루는 문제는 여기서 한 단계 더 나아가 **어떤 gland에 속하는 픽셀인지까지 식별**해야 한다. 즉, foreground 분할뿐 아니라 서로 붙어 있거나 거의 맞닿아 있는 gland들을 각각 다른 instance로 분리해야 한다.

이 문제가 중요한 이유는 병리 영상에서 gland의 형태학적 구조가 benign/malignant 판별과 암의 진행 정도 평가에 직접적으로 연결되기 때문이다. 단순히 gland 영역만 찾는 것으로는 morphology analysis에 필요한 개별 구조 분석이 어렵다. 특히 adenocarcinoma와 같은 질환에서는 gland의 모양이 불규칙하고, 염색 강도 변화(anisochromasia), 복잡한 조직 배경, 인접 gland 사이의 접착(coalescence) 현상 때문에 instance 수준의 정확한 분리가 쉽지 않다.

논문은 이 문제를 두 개의 하위 문제로 정리한다. 첫째는 **foreground labeling/segmentation**이고, 둘째는 **instance recognition**이다. 전자는 각 픽셀이 gland인지 아닌지를 판단하는 문제이고, 후자는 gland 픽셀들이 어떤 개별 gland instance에 속하는지를 나누는 문제다. 논문은 후자가 비미분적이고 직접 최적화하기 어렵다는 점을 지적하고, 이를 우회하기 위해 edge detection과 object detection을 함께 사용하는 **deep multichannel neural network**를 제안한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 gland instance segmentation을 한 가지 표현으로만 풀지 않고, **region, boundary, location**이라는 서로 다른 단서를 각각 별도 채널에서 학습한 뒤 이를 다시 fusion하여 최종 instance segmentation을 얻는 것이다.

구체적으로 보면 다음 세 종류의 정보가 결합된다. 첫 번째는 FCN 기반의 **foreground segmentation channel**이 제공하는 gland region 정보다. 이 채널은 어디가 gland이고 어디가 background인지 넓은 영역 관점에서 판단한다. 두 번째는 HED 기반의 **edge detection channel**이 제공하는 boundary 정보다. 이 채널은 서로 붙어 있는 gland 사이 경계를 더 정밀하게 드러내는 역할을 한다. 세 번째는 Faster R-CNN 기반의 **object detection channel**이 제공하는 location 정보다. 이 채널은 각 gland의 대략적 위치와 범위를 bounding box 수준에서 알려준다.

논문이 강조하는 차별점은, 자연영상 instance segmentation에서 흔한 **“detect first, then segment inside each box”** 방식과 다르다는 점이다. SDS, Hypercolumn, MNC 같은 기존 방법은 bounding box 내부에서 instance mask를 구하는 cascade 구조를 따른다. 하지만 gland는 모양이 매우 불규칙하고, box 안에서만 segmentation을 수행하면 문맥 정보가 손실되며, box가 겹칠 때 픽셀 소속이 애매해진다. 이 논문은 bounding box 내부에서 바로 마스킹하지 않고, detection 결과를 하나의 **location cue**로 바꿔 다른 채널 정보와 함께 fusion network에 넣는다.

즉, 이 방법의 중심 직관은 다음과 같다.
**region만으로는 붙어 있는 gland를 나누기 어렵고, edge만으로는 coalescence나 약한 경계에서 실패할 수 있으며, detection만으로는 불규칙한 malignant gland를 정밀하게 복원하기 어렵다. 따라서 세 정보를 함께 써야 instance segmentation이 좋아진다.**

## 3. 상세 방법 설명

### 3.1 문제 정식화

논문은 학습 데이터셋을 다음과 같이 둔다.

$$
D = {(X_n, Y_n, Z_n), n=1,2,\dots,N}
$$

여기서 $X$는 입력 영상, $Y$는 binary segmentation label, $Z$는 instance label이다.
$Y$는 각 픽셀이 gland인지 background인지를 나타내고, $Z$는 각 픽셀이 어떤 gland instance에 속하는지를 나타낸다.

각 instance region을 $R_k$라고 하면, 서로 다른 instance는 겹치지 않고 전체 영상 영역 $\Omega$를 분할해야 한다.

$$
R_k \cap R_t = \varnothing, \quad \forall k \neq t
$$

$$
\cup R_k = \Omega
$$

이 formulation은 gland instance segmentation이 단순 binary segmentation이 아니라, **foreground 픽셀을 서로 다른 개체 단위로 나누는 문제**라는 점을 분명히 보여준다.

### 3.2 두 하위 문제와 비용 함수

#### (1) Foreground labeling/segmentation

segmentation 결과를 $\hat{Y}$라고 하면, 논문은 pixel-wise 오분류율 형태의 비용을 사용한다.

$$
Dist(Y, \hat{Y}) = \frac{1}{|Y|}\sum_{j=1}^{|Y|}\delta(y_j \neq \hat{y}_j)
$$

여기서

$$
\hat{y}_j = \arg\max_y P(y|X)
$$

이다. 즉, 각 픽셀에서 가장 높은 확률의 클래스를 선택한다. 이 문제는 일반적인 binary pixel classification이므로 SGD 기반 학습이 가능하다.

#### (2) Instance recognition

instance prediction을 $\hat{Z}$라고 할 때, 논문은 예측된 각 instance region이 실제 어떤 gland와 충분히 겹치는지로 성능을 정의한다.

$$
Dist(Z,\hat{Z}) = 1 - \frac{1}{K}\sum_{k'=0}^{K'} L(\hat{R}_{k'}, Z)
$$

여기서 $L(\hat{R}*{k'}, Z)$는 예측 영역 $\hat{R}*{k'}$가 정답 영역 중 하나와 IoU 유사 형태의 overlap 비율이 threshold 이상이면 1, 아니면 0이다.

$$
L(\hat{R}*{k'}, Z)=
\begin{cases}
1, & \exists k \neq 0,\ \frac{\hat{R}*{k'} \cap R_k}{\hat{R}_{k'} \cup R_k} \geq thre \
0, & \text{otherwise}
\end{cases}
$$

논문에서 $thre = 0.5$로 둔다.

중요한 점은 이 비용이 **비미분적(nondifferentiable)** 이라서 직접 SGD로 학습하기 어렵다는 것이다. 그래서 저자들은 instance recognition을 직접 학습하지 않고, 이를 근사하는 두 보조 문제인 **edge detection**과 **object detection**으로 바꾼다. 이것이 전체 멀티채널 설계의 논리적 출발점이다.

### 3.3 전체 파이프라인 개요

전체 시스템은 세 채널과 하나의 fusion network로 구성된다.

1. **Foreground segmentation channel**: gland vs background를 분할
2. **Edge detection channel**: gland 사이 경계를 검출
3. **Object detection channel**: gland의 위치와 범위를 검출
4. **Fusion network**: 세 채널의 출력을 합쳐 최종 instance segmentation 생성

각 채널은 VGG16 backbone에 기반하지만, 목적에 맞게 변형된다. 논문 본문에 따르면 foreground segmentation과 fusion 단계에서는 해상도 손실을 줄이기 위해 **dilated convolution**을 적극적으로 사용한다.

---

### 3.4 Foreground Segmentation Channel

이 채널은 FCN-32s를 바탕으로 gland foreground segmentation을 수행한다. 하지만 일반 FCN은 pooling과 strided convolution 때문에 해상도가 낮아지고, 인접 객체가 하나로 붙는 문제가 있다. 이를 줄이기 위해 저자들은 **pool4와 pool5의 stride를 1로 바꾸고**, 이후 convolution layer에서 **dilated convolution**을 사용해 receptive field를 넓힌다.

이 채널의 출력은 다음과 같이 표현된다.

$$
P_s(Y^* = k \mid X; w_s) = \mu_k(h_s(X, w_s))
$$

여기서 $\mu$는 softmax이고, $h_s$는 hidden feature map이다. 이 경우 클래스는 foreground와 background 두 개다.

이 채널은 **softmax cross entropy loss**로 학습한다.

이 설계의 의미는 명확하다.
일반 FCN은 semantic segmentation에는 적합하지만 instance separation에는 약하다. 그래도 gland가 어디 있는지를 파악하는 확률 맵은 매우 유용하므로, 이를 전체 시스템의 region cue로 활용한다.

---

### 3.5 Edge Detection Channel

이 채널은 HED(Holistically-nested Edge Detector)를 기반으로 gland boundary를 예측한다. 논문은 edge가 두 가지 방식으로 중요하다고 설명한다.

첫째, FCN에서 max-pooling과 stride 때문에 사라진 세부 경계 정보를 보완한다.
둘째, foreground region과 위치 정보만으로는 서로 붙어 있는 gland가 연결된 채 남을 수 있는데, edge는 이들을 분리하는 직접적 단서를 제공한다.

HED는 여러 깊이의 feature에서 side output을 만들고 deep supervision으로 학습한다. 각 side output은 다음처럼 표현된다.

$$
P_e^{(m)}(E^{(m)*}=1 \mid X; w_e) = \sigma(h_e^{(m)}(X, w_e))
$$

여기서 $\sigma$는 sigmoid다. 최종 edge prediction은 여러 side output의 가중합을 통해 얻는다.

$$
P_e(E^*=1 \mid X; w_e, \alpha) = \sigma\left(\sum_{m=1}^{M}\alpha^{(m)} h_e^{(m)}(X, w_e)\right)
$$

논문의 설명에 따르면 edge pixel은 전체 영상에서 매우 적어 class imbalance가 심하므로, deep supervision이 수렴 안정화에 도움을 준다. 또한 실험 파트에서는 **edge label dilation**이 성능 향상에 기여했다고 보고한다. 즉, 아주 얇은 경계 라벨을 약간 두껍게 만들어 imbalance를 완화했다.

이 채널은 **sigmoid cross entropy loss**로 학습한다.

---

### 3.6 Object Detection Channel

이 채널은 Faster R-CNN을 이용해 각 gland의 bounding box를 검출한다. 목적은 자연영상식 cascade segmentation을 수행하는 것이 아니라, gland의 **위치와 범위(location cue)** 를 얻는 데 있다.

논문이 흥미로운 부분은 detection 결과를 그대로 box set으로 쓰지 않고, 이를 **dense map으로 변환하는 filling operation**을 적용한다는 점이다. 각 픽셀 값은 자신을 덮는 bounding box 개수가 된다. 예를 들어 어떤 픽셀이 세 개 box의 겹침 영역에 있으면 그 픽셀 값은 3이 된다.

이 채널 출력은 다음과 같이 쓴다.

$$
P_d(X, w_d) = \phi(h_d(X, w_d))
$$

여기서 $h_d$는 예측 bounding box 좌표이고, $\phi$는 filling operation이다.

즉, object detection channel은 “이 픽셀이 어느 gland box 안에 얼마나 포함되는가”라는 형태의 위치 정보를 제공한다. 이는 edge가 놓칠 수 있는 공간적 구획 정보를 보완한다.

이 채널은 Faster R-CNN과 동일하게 **classification loss + box regression loss**의 합으로 학습한다.

---

### 3.7 Multichannel Fusion

세 채널의 출력이 최종 목적이 아니라, 이들을 결합해 instance segmentation을 수행하는 것이 핵심이다. 저자들은 이를 위해 **7-layer shallow CNN**을 사용한다. 이 fusion network에서도 downsampling 대신 **dilated convolution**을 사용해 정보 손실을 줄이고 receptive field를 충분히 확보한다.

최종 출력은 다음과 같다.

$$
P(Y_I^* = k \mid P_s, P_d, P_e; w_f) = \mu_k(h_f(P_s, P_d, P_e, w_f))
$$

여기서 $P_s$, $P_d$, $P_e$는 각각 segmentation, detection, edge 채널 출력이고, $Y_I^*$는 최종 instance segmentation prediction이다.

fusion network는 **softmax cross entropy loss**로 학습한다.

이 구조의 요점은, 단순 후처리 규칙으로 region/edge/box를 합치는 것이 아니라 CNN이 이들 패턴을 학습적으로 통합하게 만든다는 데 있다. 논문은 네트워크 깊이와 필터 수를 cross validation으로 늘려 가며 성능이 더 좋아지지 않을 때의 구조를 채택했다고 설명하지만, 각 layer의 정확한 세부 설정은 본문 텍스트만으로는 충분히 제시되지 않았다.

## 4. 실험 및 결과

### 4.1 데이터셋

실험은 **MICCAI 2015 Gland Segmentation Challenge (GlaS)** 데이터셋에서 수행되었다. 총 165장의 colorectal cancer histology image로 구성되며, 85장은 training, 80장은 test다. test는 A와 B로 나뉘며, A는 60장, B는 20장이다. 논문은 benign과 malignant 비율도 구체적으로 제시한다.

* 학습: benign 37장, malignant 48장
* Test A: benign 33장, malignant 27장
* Test B: benign 4장, malignant 16장

논문은 Test B가 더 복잡한 cancerous image 비율이 높아 더 어렵다고 해석한다.

### 4.2 전처리와 데이터 증강

전처리로는 channel-wise zero mean을 적용한다.
추가적으로 region label에서 edge label을 만들고, bounding box ground truth도 segmentation label에서 생성한다.

데이터 증강은 두 전략으로 비교한다.

* **Strategy I**: horizontal flip + rotation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$)
* **Strategy II**: Strategy I + elastic transformation (pin cushion, barrel transformation)

또한 학습 시 증강된 이미지에서 $400 \times 400$ 패치를 랜덤 크롭하여 사용한다.

### 4.3 하이퍼파라미터와 학습 설정

실험은 Caffe, K40 GPU, CUDA 7.0 환경에서 수행되었다.
weight decay는 0.002, momentum은 0.9다.

각 채널 학습률은 다음과 같다.

* foreground segmentation channel: $10^{-3}$
* edge detection channel: $10^{-9}$
* object detection channel: $10^{-3}$
* fusion network: $10^{-3}$

초기화는 foreground는 pretrained FCN-32s, detection은 pretrained Faster R-CNN, edge와 fusion은 Xavier initialization을 사용한다.

이 수치들은 논문에 명시되어 있으나, 학습 epoch 수나 batch size 등의 추가 세부 사항은 제공된 텍스트에서 확인되지 않는다.

### 4.4 평가 지표

논문은 챌린지와 동일한 세 지표를 사용한다.

#### (1) F1 score

gland detection accuracy를 본다. 예측 object가 GT와 50% 이상 겹치면 TP로 본다.

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

$$
Precision = \frac{TP}{TP+FP}
$$

$$
Recall = \frac{TP}{TP+FN}
$$

#### (2) ObjectDice

일반 Dice는 instance를 구분하지 못하므로, 논문은 object-level Dice를 사용한다.

기본 Dice는

$$
D(G,S)=\frac{2|G\cap S|}{|G|+|S|}
$$

이고, object-level Dice는 예측 object와 GT object의 대응을 고려한 가중 평균이다.

$$
D_{object}(G,S)=\frac{1}{2}\left[\sum_{i=1}^{n_S} w_i D(G_i,S_i)+\sum_{i=1}^{n_G}\widetilde{w}_i D(\widetilde{G}_i,\widetilde{S}_i)\right]
$$

여기서 가중치는 각 object의 크기 비율이다.

#### (3) ObjectHausdorff

shape similarity를 평가한다. 기본 Hausdorff distance는

$$
H(G,S)=\max\left{\sup_{x\in G}\inf_{y\in S}|x-y|,\sup_{y\in S}\inf_{x\in G}|x-y|\right}
$$

이고, object-level Hausdorff는 Dice와 마찬가지로 object 대응 기반 가중 평균으로 확장된다.

$$
H_{object}(S,G)=\frac{1}{2}\left[\sum_{i=1}^{n_s} w_i H(G_i,S_i)+\sum_{i=1}^{n_G}\widetilde{w}_i H(\widetilde{G}_i,\widetilde{S}_i)\right]
$$

또한 논문은 Test A와 Test B의 샘플 수 차이를 고려해 weighted rank sum도 별도로 계산한다.

$$
WeightedRS = \frac{3}{4}\sum testA Rank + \frac{1}{4}\sum testB Rank
$$

### 4.5 주요 결과

#### 챌린지 참가팀 및 FCN 계열과의 비교

논문이 제시한 Table I에서 제안 방법은 다음 성능을 보인다.

* **F1 Score**

  * Part A: 0.893
  * Part B: 0.843
* **ObjectDice**

  * Part A: 0.908
  * Part B: 0.833
* **ObjectHausdorff**

  * Part A: 44.129
  * Part B: 116.821

Rank Sum은 8, Weighted Rank Sum은 4.5로 제시되며, 논문은 이를 기반으로 **state-of-the-art**라고 주장한다.

비교 기준 중 중요한 점은 다음과 같다.

* 일반 **FCN**보다 매우 우수하다.
* **dilated FCN**보다도 더 좋다.
* MICCAI 2015 참가팀들과 비교해도 종합 ranking에서 가장 좋다.
* 특히 Part B처럼 더 어려운 malignant-heavy test에서도 경쟁력이 높다.

다만 표를 보면 세부 metric별 개별 rank에서는 일부 항목에서 다른 팀이 더 높은 점수를 보이는 경우도 있다. 그러나 논문은 종합 순위인 RS와 WRS 기준에서 전체적으로 가장 좋다고 해석한다.

#### 자연영상 instance segmentation 방법과의 비교

Table II에서는 SDS, Hypercolumn, MNC와 비교한다.

* Hypercolumn:

  * F1 A/B = 0.852 / 0.691
  * ObjectDice A/B = 0.742 / 0.653
* MNC:

  * F1 A/B = 0.856 / 0.701
  * ObjectDice A/B = 0.793 / 0.705
* SDS:

  * F1 A/B = 0.545 / 0.322
  * ObjectDice A/B = 0.647 / 0.495
* **OURS**:

  * F1 A/B = 0.893 / 0.843
  * ObjectDice A/B = 0.908 / 0.833

차이가 매우 크다. 논문은 그 이유를 자연영상 instance segmentation의 cascade 구조가 gland histology에는 맞지 않기 때문이라고 설명한다. 즉, detection이나 proposal이 틀리면 뒤 segmentation도 실패하고, box 안에서만 segmentation하면 문맥 손실과 겹침 문제를 피하기 어렵다.

### 4.6 Ablation Study

#### (1) Data augmentation 전략 비교

Strategy II가 Strategy I보다 FCN과 dilated FCN 모두에서 꾸준히 성능을 높인다. 이는 rotation invariance와 elastic deformation이 histology image에 유효함을 보여준다.

예를 들어 dilated FCN의 경우:

* Strategy I:

  * F1 A/B = 0.820 / 0.749
  * ObjectDice A/B = 0.843 / 0.811
* Strategy II:

  * F1 A/B = 0.854 / 0.798
  * ObjectDice A/B = 0.879 / 0.825

즉, 단순 flip/rotation보다 elastic transformation까지 포함한 증강이 성능 향상에 더 효과적이다.

#### (2) 각 채널의 기여

Table IV는 세 채널과 dilated fusion의 기여를 보여준다.

최종 최고 설정은:

* **DMC: dilated FCN + EDGE3 + BOX**

  * F1 A/B = 0.893 / 0.843
  * ObjectDice A/B = 0.908 / 0.833
  * ObjectHausdorff A/B = 44.129 / 116.821

여기서

* DMC는 fusion network에 dilated convolution 사용
* EDGE3는 radius 3 disk filter로 edge label dilation
* BOX는 object detection 포함

비교를 보면:

* BOX를 제거하면 성능이 떨어진다.
* EDGE를 제거해도 성능이 떨어진다.
* dilated FCN이 일반 FCN보다 대체로 좋다.
* fusion 단계에서도 dilated convolution이 도움이 된다.
* edge label dilation(EDGE3)이 edge 미확장(EDGE1)보다 더 낫다.

즉, 이 ablation은 제안 방식이 단순히 채널을 많이 붙인 것이 아니라, **region + boundary + location의 조합이 실제로 서로 보완적**이라는 점을 실험적으로 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 gland instance segmentation의 어려움을 잘 분해해서, 직접 최적화하기 어려운 instance recognition을 **학습 가능한 보조 표현들로 우회**했다는 점이다. 문제 정의가 명확하고, 왜 segmentation만으로는 부족한지, 왜 edge와 detection이 필요한지 서술이 설득력 있다. 특히 병리 영상의 특수성 때문에 자연영상용 instance segmentation을 그대로 적용하기 어렵다는 비판은 타당하다.

또 다른 강점은 설계가 직관적이면서도 실험적으로 잘 뒷받침된다는 점이다. 데이터 증강, dilated convolution, edge dilation, detection channel 추가, fusion dilation 등을 각각 비교하면서 어떤 요소가 실제 성능 향상에 기여하는지 보여준다. 단순 최종 수치 제시에 그치지 않고, 채널별 역할을 ablation으로 입증하려 한 점은 논문의 신뢰도를 높인다.

실험 결과도 인상적이다. MICCAI 2015 Gland Segmentation Challenge 데이터셋에서 강한 베이스라인과 기존 참가팀을 상대로 경쟁력 있는 결과를 내며, 특히 FCN이나 dilated FCN만으로는 잘 분리되지 않는 인접 gland를 더 잘 분리한다는 정성적/정량적 근거를 제시한다.

한계도 분명하다. 첫째, 전체 시스템이 **여러 개의 네트워크를 따로 학습하고 나중에 fusion하는 구조**이므로 복잡하다. foreground segmentation, edge detection, object detection, fusion이라는 다단계 구조는 구현과 학습 비용이 크다. 오늘날 관점에서는 end-to-end unified architecture에 비해 비효율적으로 보일 수 있다.

둘째, object detection channel이 bounding box 기반이기 때문에, 매우 불규칙하거나 잘린 gland, 혹은 배경과 cytoplasm 구분이 애매한 경우에는 여전히 오류가 발생한다. 논문도 white region이 cytoplasm인지 실제 background인지 헷갈리는 사례를 실패 원인으로 언급한다. 또한 이미지 가장자리에서 잘린 gland 때문에 cytoplasm이 background로 오인되는 경우도 있다고 설명한다.

셋째, 평가 데이터셋 규모가 크지 않다. 총 165장, 학습 85장 수준이기 때문에 일반화 성능에 대한 강한 결론을 내리기에는 제한이 있다. 논문은 generalization ability를 언급하지만, 제공된 텍스트만으로는 외부 데이터셋 검증은 확인되지 않는다.

넷째, fusion network 구조가 cross validation으로 정해졌다고 하지만, 왜 7-layer shallow CNN이 최적인지에 대한 더 깊은 분석은 부족하다. 또한 채널 간 end-to-end joint optimization 여부는 제공된 텍스트만으로 명확하지 않다. 읽을 수 있는 범위에서는 각 채널이 별도 학습된 뒤 결합되는 것으로 이해되지만, 완전한 joint training인지까지는 확언하기 어렵다.

비판적으로 보면, 이 논문은 매우 실용적이고 성능 중심의 설계이지만, instance segmentation 자체를 보다 직접적으로 모델링한 것은 아니다. 즉, 비미분적인 instance cost를 edge와 detection으로 근사하는 방식은 효과적이지만, 보다 근본적인 instance-aware representation을 학습하는 방향과는 거리가 있다. 그럼에도 당시 시점의 병리 영상 문제 설정에서는 매우 현실적이고 강한 접근으로 볼 수 있다.

## 6. 결론

이 논문은 colon histology image에서의 gland instance segmentation을 위해 **deep multichannel neural networks**라는 구조를 제안했다. 핵심은 foreground region, boundary edge, object location이라는 세 가지 상보적 정보를 각각 별도 채널로 추출하고, 이를 CNN 기반 fusion network에서 통합하여 최종 instance segmentation을 생성하는 것이다.

논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, gland instance segmentation을 segmentation과 instance recognition의 두 하위 문제로 명확히 나눴다. 둘째, 직접 학습이 어려운 instance recognition을 edge detection과 object detection으로 근사하여 실제 학습 가능한 시스템으로 만들었다. 셋째, GlaS benchmark에서 강한 성능과 풍부한 ablation 결과를 통해 제안 구조의 유효성을 입증했다.

실제 적용 측면에서는 digital pathology에서 gland morphology 분석, 암 grading 보조, 병리 판독 지원 시스템에 중요한 기반 기술이 될 가능성이 있다. 향후 연구에서는 더 큰 데이터셋 검증, 완전 end-to-end 통합, 더 정교한 instance-aware representation, 그리고 다른 의료 영상 도메인으로의 확장이 자연스러운 후속 방향으로 보인다.
