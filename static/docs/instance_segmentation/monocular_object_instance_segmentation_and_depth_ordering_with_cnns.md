# Monocular Object Instance Segmentation and Depth Ordering with CNNs

* **저자**: Ziyu Zhang, Alexander G. Schwing, Sanja Fidler, Raquel Urtasun
* **발표연도**: 2015
* **arXiv**: [https://arxiv.org/abs/1505.03159](https://arxiv.org/abs/1505.03159)

## 1. 논문 개요

이 논문은 **단안 RGB 이미지 한 장만으로 자동차 인스턴스 분할(instance-level segmentation)과 객체 간 depth ordering을 동시에 예측**하는 방법을 제안한다. 여기서 depth ordering은 정확한 metric depth를 회귀하는 것이 아니라, 어떤 차량이 더 카메라에 가까운지를 순서 형태로 추정하는 문제이다. 즉, 각 픽셀에 대해 “배경인지 차량인지”만 예측하는 것을 넘어서, “어느 차량 인스턴스에 속하는지”와 “그 차량이 다른 차량보다 앞에 있는지 뒤에 있는지”까지 함께 추론하는 것이 목표다.

연구 문제는 크게 두 가지가 결합되어 있다. 첫째, 동일 클래스인 여러 차량을 서로 다른 인스턴스로 구분해야 한다. 둘째, 이 인스턴스들 사이의 상대적 깊이 순서를 정해야 한다. 이 두 문제는 서로 얽혀 있다. 예를 들어 겹쳐 있는 차량을 올바르게 분리하려면 어떤 차량이 앞에 있는지 알아야 하고, 반대로 depth ordering을 제대로 하려면 인스턴스 경계가 비교적 정확해야 한다.

이 문제가 중요한 이유는 자율주행, 운전자 보조 시스템, 장면 이해(scene understanding) 같은 응용에서 단순한 bounding box 검출보다 훨씬 풍부한 표현이 필요하기 때문이다. 차량이 “있다”는 사실만으로는 부족하고, 각각의 차량이 어디까지 차지하는지, 어느 차량이 더 가까운지, 가림(occlusion)이 어떻게 일어나는지를 알아야 실제 의사결정에 도움이 된다. 논문은 이러한 필요를 배경으로, detection에 의존하지 않고 segmentation과 depth ordering을 공동으로 처리하려고 한다.

또한 당시 기준으로는 이 문제를 풀기 위한 대규모 정밀 인스턴스 분할 데이터가 부족했기 때문에, 저자들은 약한 형태의 3D supervision을 활용해 학습 데이터를 확보한다. 훈련 시에는 3D bounding box, stereo, LIDAR 등 추가 정보를 쓰지만, **테스트 시점에는 오직 단일 RGB 이미지**만 사용한다는 점도 실용적 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 다음과 같다. **CNN이 이미지 patch 단위로 “인스턴스 분할 + depth ordering”을 직접 예측하고, 이후 여러 patch의 예측을 MRF로 통합하여 이미지 전체에 대해 일관된 결과를 만든다.** 즉, 단일 네트워크가 전체 이미지를 한 번에 처리하는 것이 아니라, 서로 겹치는 여러 해상도의 patch를 보고 각 patch 안에서 지역적인 인스턴스 순서를 예측한 뒤, 이를 전역적으로 정합시키는 구조다.

핵심 직관은 patch 내부에서는 상대적으로 단순한 depth ordering과 instance grouping이 가능하다는 점이다. 전체 장면에서 차량 수가 많고 가림이 복잡하더라도, 작은 범위 안에서는 보이는 차량 수가 제한되고 순서도 더 단순해진다. 그래서 저자들은 patch 안에서 최대 5개의 차량과 배경을 포함한 6개 수준의 label을 예측하게 했다. 이때 label ID 자체가 순서를 의미하도록 설계했다. 예를 들어 label 1은 가장 가까운 차량, label 2는 그 다음 차량 같은 식이다.

기존 접근법과의 차별점은 두 가지다. 첫째, 많은 기존 인스턴스 분할 방법이 object detector의 bounding box를 먼저 구한 뒤 그 내부를 분할하는 top-down 방식에 의존했는데, 이 논문은 **검출 결과를 입력으로 요구하지 않는다**. 대신 patch 기반 CNN과 MRF를 통해 detection과 segmentation을 함께 추론하려 한다. 둘째, 기존 depth ordering 연구가 주로 depth layer만 예측하거나, 반대로 3D object detection만 다루는 경우가 많았던 것과 달리, 이 논문은 **instance segmentation과 depth ordering을 하나의 픽셀 라벨링 문제로 결합**한다.

또 하나의 중요한 설계는 **멀티스케일 patch 처리**다. 가까운 큰 차량과 멀리 있는 작은 차량은 크기 차이가 크므로, 하나의 고정 receptive field로는 둘 다 잘 다루기 어렵다. 저자들은 large, medium, small patch를 서로 다른 위치에 배치해 다양한 크기의 차량을 커버한다. 특히 KITTI의 카메라 배치 특성상 이미지의 수직 위치와 거리 사이에 상관관계가 있다는 점을 이용해, 작은 patch는 주로 horizon 근처 영역에 집중적으로 샘플링한다.

정리하면, 이 논문의 핵심은 다음 문장으로 요약할 수 있다. **지역적으로는 CNN이 depth-aware instance labeling을 예측하고, 전역적으로는 MRF가 겹치는 patch들의 예측을 일관된 장면 해석으로 융합한다.**

## 3. 상세 방법 설명

전체 방법은 크게 네 단계로 이해할 수 있다. 먼저 입력 이미지에서 여러 크기의 겹치는 patch를 추출한다. 다음으로 CNN이 각 patch에 대해 픽셀별 인스턴스 및 depth order label을 예측한다. 그 뒤 patch 예측들을 합쳐 connected component를 만든다. 마지막으로 MRF 에너지 최소화를 통해 이미지 전체의 최종 라벨링을 얻고, 후처리로 작은 잡음과 hole을 정리한다.

### 3.1 문제 정의

이미지의 각 픽셀 $p$에 대해 라벨 $y_p \in {0, \dots, N}$를 예측한다. 여기서 0은 background이고, 1부터 $N$까지는 차량 인스턴스를 나타낸다. 중요한 점은 이 라벨이 단순한 인스턴스 ID가 아니라 **깊이 순서를 함께 인코딩**한다는 것이다. 즉, 어떤 픽셀이 label $i$를 갖고 다른 픽셀이 label $j$를 가질 때, $i < j$이면 $i$에 해당하는 차량이 더 카메라에 가깝다.

논문에서는 이미지 전체에 대해 최대 $N=9$개의 차량을 허용한다. 반면 각 patch 안에서는 최대 5개 차량과 background를 포함한 6개 수준만 예측한다. patch의 예측은 지역적이고 상대적인 순서이므로, 이후 MRF가 이를 전역 순서로 재배치하고 통합한다.

### 3.2 CNN 구조

저자들은 VGG-16을 기반으로 네트워크를 설계한다. 원래 VGG는 224×224 입력에 대해 하나의 class label을 예측하는 분류 네트워크다. 하지만 여기서는 픽셀별 예측이 필요하므로 fully connected layer들을 convolutional layer로 바꿔 **fully convolutional 형태**로 만든다. 이렇게 하면 입력 patch 크기를 키웠을 때 출력도 공간 구조를 가진 score map이 된다.

논문은 입력 patch 크기를 306×306으로 사용한다. 다만 기본 VGG는 pooling이 여러 번 들어가므로 출력 해상도가 입력보다 $2^5=32$배 작아져 너무 거칠어진다. 이를 줄이기 위해 상위 두 pooling layer에서 downsampling을 하지 않고, 대신 **à trous (dilated) convolution**과 유사한 방식으로 receptive field는 유지하면서 출력 해상도를 높인다. 결과적으로 출력은 입력보다 8배 작은 해상도가 된다. 최종 출력 크기는 40×40×6으로 기술되어 있으며, 각 위치마다 6개 클래스 점수, 즉 background + depth-ordered instance level에 대한 score가 나온다.

이 설계의 의미는 분명하다. 네트워크가 patch 안의 각 위치에 대해 “이 픽셀이 배경인지, 가장 가까운 차인지, 두 번째 차인지 …”를 분류하는 것이다. 논문은 이를 일반적인 semantic segmentation이 아니라 **depth-ordered instance segmentation**으로 본다.

### 3.3 학습 데이터와 학습 목표

학습에는 patch와 그에 대응하는 ground-truth depth-level map 쌍 $\mathcal{D}={(z, \mathbf{y}^{GT}_z)}$를 사용한다. patch는 car가 포함된 ground-truth bounding box를 확장해서 추출하며, 다양한 해상도에서 추출한다. 모든 patch는 GPU mini-batch 처리를 위해 306×306으로 리사이즈하고, 정답 map은 40×40으로 다운샘플링한다.

학습 목표는 **cross-entropy loss**다. 즉, 각 출력 위치에서 정답 depth-level class를 맞히도록 학습한다. 논문 본문에는 식 형태로 loss를 전개하지는 않았지만, 설명상 전형적인 pixel-wise multinomial cross-entropy로 이해하면 된다. 구체적으로는 각 출력 위치 $u$에 대해 정답 클래스 $c_u$가 있을 때,

$$
\mathcal{L} = - \sum_u \log P(y_u = c_u \mid z; w)
$$

와 같은 형태로 볼 수 있다. 여기서 $w$는 네트워크 파라미터다.

최적화는 SGD를 사용하며, 배치 크기 5, weight decay 0.0005, momentum 0.9를 사용한다. 초기 learning rate는 top layer에 0.01, 나머지 파라미터에 0.001을 썼다. 이 수치는 VGG 기반 segmentation fine-tuning의 전형적인 설정과 유사하다.

### 3.4 Patch 추출 전략

입력 이미지 전체를 여러 patch로 나누는 방식이 중요하다. 논문은 large, medium, small의 세 가지 scale patch를 사용한다. 큰 patch는 이미지 아래쪽을 일정 stride로 덮고, medium과 small patch는 주로 horizon 근처에 배치한다. 이는 KITTI에서 작은 차량은 대개 멀리 있어서 horizon 부근에 위치한다는 경험적 사실을 활용한 것이다.

이 설계는 단순하지만 실제 자율주행 장면의 기하적 구조를 잘 활용한다. 모든 위치를 모든 scale로 균등하게 처리하는 것이 아니라, **카메라 장착 높이와 수직축-깊이 상관관계**를 이용해 연산을 절약하고 작은 원거리 차량을 더 잘 포착하려는 의도다.

### 3.5 Patch 병합을 위한 MRF

CNN의 각 patch 예측은 지역적으로만 일관되므로, 이를 바로 쓰면 이미지 전체에서 label 충돌이 생긴다. 예를 들어 서로 다른 patch에서 각각 “가장 가까운 차”라고 예측된 두 차량이 실제로는 서로 다른 전역 깊이 순서를 가져야 한다. 이를 해결하기 위해 논문은 전역 라벨링 $\mathbf{y}$에 대해 MRF 에너지를 정의한다.

전체 에너지는 다음과 같다.

$$
E(\mathbf{y}) =
\sum_p \left(E_{\text{CNN},p}(y_p) + E_{\text{CCO},p}(y_p)\right)

* \sum_{p,p':\mathcal{C}(p)\neq \mathcal{C}(p')} E_{\text{long},p,p'}(y_p, y_{p'})
* \sum_{p,p' \in \mathcal{N}(p)} E_{\text{short},p,p'}(y_p, y_{p'})
  $$

여기서 $\mathcal{N}(p)$는 4-neighborhood이고, $\mathcal{C}(p)$는 pixel $p$가 속한 connected component다.

이 식은 직관적으로 네 가지 신호를 합친 것이다. 첫째, patch CNN이 말해준 지역적 depth level. 둘째, connected component 기반의 대략적인 인스턴스 순서. 셋째, 서로 다른 component 사이의 장거리 순서 제약. 넷째, 근처 픽셀은 CNN 예측과 비슷하게 유지하려는 단거리 smoothness 제약이다.

#### (1) CNN unary energy

이 항은 “지역 patch에서 보인 순서는 전역 순서보다 작거나 같아야 한다”는 가정을 반영한다. patch는 전체 장면의 일부만 보기 때문에, patch 안에서 어떤 차가 가장 가까운 차로 보였다 하더라도 전역 장면에서는 더 앞에 다른 차가 있을 수 있다. 그래서 논문은 local label보다 같거나 더 큰 전역 label을 허용한다.

정의는 다음과 같다.

$$
E_{\text{CNN},p}(y_p) = \sum_z E_{\text{CNN},z,p}(y_p)
$$

그리고 각 patch $z$에 대해,

$$
E_{\text{CNN},z,p}(y_p) =
\begin{cases}
-1 & \text{if } y_p \ge y^*_{z,p} \
0 & \text{otherwise}
\end{cases}
$$

여기서 $\mathbf{y}^**z = \arg\max*{\mathbf{y}_z} F(z,\mathbf{y}_z,w)$이고, $y^*_{z,p}$는 patch $z$에서 pixel $p$에 대해 CNN이 가장 높게 예측한 label이다.

즉, CNN이 어떤 픽셀을 patch 안에서 2번째로 가까운 차라고 예측했다면, 전역적으로는 2, 3, 4, … 같은 더 먼 순서로 재배치되는 것은 허용하지만, 1처럼 더 가까운 순서로 가는 것은 선호하지 않는다.

#### (2) Connected Components Ordering unary

patch 예측들을 어느 정도 합친 뒤 connected component를 계산하고, 이 component들을 이미지 수직축 기준으로 정렬한다. KITTI 같은 도로 장면에서는 일반적으로 이미지 아래쪽에 있는 차량이 더 가깝다는 경향이 있으므로, 수직 위치를 depth ordering의 약한 단서로 사용한다.

정의는 다음과 같다.

$$
E_{\text{CCO},p}(y_p) =
\begin{cases}
-1 & \text{if } y_p \ge O(p) \
0 & \text{otherwise}
\end{cases}
$$

여기서 $O(p)$는 pixel $p$가 속한 connected component의 순서를 의미한다.

이 항 역시 “최소한 이 정도로는 멀어야 한다”는 식의 하한 제약으로 작동한다. 절대적으로 정확한 depth는 아니지만, 교통 장면에서는 매우 유용한 prior다.

#### (3) Long-range connections

서로 다른 connected component에 속한 픽셀 쌍에 대해, 두 component의 순서가 다르면 label도 다르게 부여하도록 유도한다. 모든 픽셀 쌍을 연결하면 너무 비싸므로 component pair마다 20,000개의 connection만 랜덤 샘플링한다.

정의는 다음과 같다.

$$
E_{\text{long},p,p'}(y_p,y_{p'}) =
\begin{cases}
-1 & \text{if } y_{p'} > y_p,; y_p \neq 0,; O(p') > O(p) \
0 & \text{otherwise}
\end{cases}
$$

즉, component ordering상 $p'$ 쪽이 더 뒤에 있어야 한다면, 실제 label도 더 큰 값이 되도록 선호한다. 이 항은 서로 다른 객체 인스턴스 간의 전역적 깊이 순서를 강화하는 역할을 한다.

#### (4) Short-range connections

논문은 이 항을 weighted Potts-type potential이라고 설명하지만, 본문에 정확한 식은 제공되지 않는다. 설명에 따르면, 인접한 두 픽셀이 CNN 예측에서 같은 state라면 최종 라벨도 같게 하고, CNN 예측에서 다르면 최종 라벨도 다르게 하도록 유도한다. 즉, 단순한 smoothness가 아니라 CNN의 local edge/instance boundary 신호를 반영한 pairwise term이다.

여기서는 정확한 수식이나 가중치 정의가 본문 추출 텍스트에 명시되어 있지 않으므로, 그 이상은 추측할 수 없다. 다만 기능적으로는 **근거리 경계 보존 + 지역적 일관성 유지**를 위한 항으로 이해하면 된다.

### 3.6 추론(Inference)

이 MRF는 multi-label이면서 attractive/repulsive term이 혼재해 있어 NP-hard다. 저자들은 $\alpha$-$\beta$ swap 알고리즘에서 영감을 받은 이동(move-making) 방식의 추론을 사용한다. 각 단계에서 두 label 집합 사이의 이진 최적화 문제를 만들고, 이를 **QPBO (Quadratic Pseudo-Boolean Optimization)** 로 푼다. QPBO가 일부 노드에 라벨을 주지 못하면 기본값을 사용하고, 에너지가 실제로 감소할 때만 그 move를 채택한다.

이 설명에서 알 수 있는 것은, 저자들이 이론적 최적해보다 실용적인 근사 해법을 선택했다는 점이다. 구조적으로 복잡한 MRF를 정확하게 푸는 대신, binary subproblem을 반복적으로 개선하는 방식으로 현실적인 추론 시간을 얻으려 했다.

### 3.7 후처리(Post-processing)

최종 MRF 결과 뒤에는 세 가지 후처리를 한다.

첫째, 200픽셀보다 작은 고립된 인스턴스 조각을 제거한다. 이는 차량 크기에 대한 prior를 반영한 것이다. 둘째, object 내부의 hole을 메운다. 셋째, disconnected instance label을 다시 분리해 재라벨링하고, connected component의 2D bounding box 중심의 수직 위치를 기준으로 다시 ordering한다.

이 후처리는 단순하지만 매우 중요하다. 실험에서도 pairwise MRF의 이득이 후처리 후에 더 분명하게 나타난다. 즉, 본 방법은 “CNN + MRF”만으로 끝나는 것이 아니라, **작은 잡음과 분리 오류를 정리하는 후처리까지 포함해야 완성된 시스템**이라고 보는 것이 맞다.

## 4. 실험 및 결과

### 4.1 데이터셋과 학습 설정

실험은 KITTI benchmark에서 수행되었다. 평가를 위해 [2]의 car segmentation ground-truth를 사용했는데, 301장의 이미지에 대해 총 1,229대 차량이 고품질 픽셀 라벨로 주어졌다고 설명한다. 이 중 298장을 validation 101장, test 197장으로 나누었다. 나머지 6,744장의 KITTI detection training set 이미지는 CNN 학습에 사용되었다.

중요한 점은, 이 6,744장 전체에 사람이 직접 인스턴스 마스크를 달아준 것이 아니라, [2]의 방법으로 surrogate ground truth를 생성했다는 것이다. [2]는 3D bounding box, point cloud, stereo imagery, CAD 기반 shape prior 등을 이용해 자동차 segmentation을 자동 생성한다. 저자들은 이를 이용해 대규모 patch-level supervision을 만들고, depth ordering은 KITTI의 3D bounding box 거리 정보로부터 정렬해서 만든다.

하이퍼파라미터는 validation set으로만 조정했고, test set은 최종 보고용으로만 사용했다고 명시한다. 이 점은 실험 프로토콜의 신뢰성을 높여준다.

### 4.2 비교 방법과 ablation 설정

논문은 세 가지 patch scale 성능을 먼저 patch 단위로 보여준다: large, medium, small. 그 후 이미지 레벨에서 다음 구성을 비교한다.

CNNRaw는 CNN patch 출력을 병합한 원시 결과다. Unary는 MRF에서 unary term만 사용한 버전이다. Unary+ShortRange는 unary와 short-range pairwise만 사용한다. Full은 모든 정의된 에너지 항을 사용한다. 각 방법에 대해 후처리 유무도 비교하며, 후처리가 붙은 경우 +PP로 표기한다.

비교 baseline은 [28] 기반 방법이다. 이는 검출과 방향 추정으로부터 3D bounding box를 만들고 CAD model을 projection하는 방식이라고 설명된다. 저자들은 [36]은 실행에 성공하지 못했다고 솔직하게 적고 있다.

### 4.3 Class-level segmentation 결과

표 1은 foreground-background 관점에서의 성능이다. FIoU, BIoU, AvgIoU, pixel accuracy, overall precision, overall recall을 보고한다.

핵심 결과는 medium patch가 patch 단위에서는 large나 small보다 약간 더 좋다는 것이다. 이미지 레벨에서는 Full과 Full+PP가 평균 IoU 89.8, 89.7 수준으로 매우 높다. CNNRaw가 이미 AvgIoU 89.7이므로 binary foreground/background 분리 자체는 CNN만으로도 꽤 강하다. Full은 recall 측면, 특히 OvrlRe 92.2로 가장 좋다. 반면 [28] baseline은 precision 88.0으로 높지만 recall 40.4에 불과해 많은 차량 픽셀을 놓친다.

이 결과는 중요한 해석을 준다. **이 논문의 진짜 어려움은 단순한 차/배경 분리가 아니라, 차를 개별 인스턴스로 나누고 순서를 매기는 것**이다. binary segmentation만 보면 CNNRaw도 상당히 강하다. MRF의 기여는 그보다는 인스턴스 구조와 ordering 쪽에 더 있다.

또한 후처리가 binary segmentation 성능에는 대체로 해롭다고 저자들이 직접 언급한다. 이는 후처리가 잡음 제거와 인스턴스 정리에는 도움을 주지만, 픽셀 단위 foreground/background 정합성은 약간 희생시킬 수 있음을 의미한다.

### 4.4 Instance-level segmentation 결과

표 2는 인스턴스 분할 성능이다. 지표는 MWCov, MUCov, AvgPr, AvgRe, AvgFP, AvgFN, ObjPr, ObjRe다.

가장 중요한 수치는 MWCov와 MUCov다. 이는 ground-truth 각 인스턴스에 대해 가장 잘 매칭되는 prediction의 IoU를 보는 지표라, 진정한 instance segmentation 성능을 잘 반영한다. Full+PP는 MWCov 70.3, MUCov 55.4를 기록한다. Unary+ShortRange+PP는 오히려 MWCov 71.3, MUCov 55.9로 더 높다. 반면 baseline [28]은 MWCov 45.4, MUCov 40.1 수준이다. 따라서 제안법은 baseline 대비 instance coverage에서 큰 개선을 보인다.

Object recall도 의미가 크다. Full+PP는 ObjRe 59.0, Unary+PP는 58.3, Unary+ShortRange+PP는 58.8이다. baseline은 ObjRe 48.5다. 즉, 제안법은 더 많은 실제 차량 인스턴스를 회수한다. 반면 baseline은 ObjPr 95.1로 precision이 높지만 recall이 낮다. 이는 매우 보수적으로 검출하여 놓치는 것이 많다는 뜻이다.

흥미로운 점은 후처리 전에는 Unary가 Full보다 더 좋게 보이는 지표가 있다는 것이다. 예를 들어 MWCov에서 Unary는 69.2, Full은 68.1이다. 하지만 후처리 후에는 pairwise를 쓴 방식들이 더 강해진다. 저자들도 **pairwise MRF의 효과는 post-processing 이후 뚜렷해진다**고 해석한다. 이는 pairwise term이 만들어낸 구조적 정보가 후처리와 결합될 때 더 잘 드러난다는 뜻이다.

### 4.5 Depth ordering 결과

표 3은 depth ordering 품질을 평가한다. 사용된 지표는 다음과 같다.

먼저 %RcldIns는 IoU 50% 이상으로 매칭되는 인스턴스를 얼마나 회수했는지다. 그 다음 %RcldInsPair는 두 인스턴스가 모두 회수된 pair 비율이다. InsPairAcc는 회수된 인스턴스 pair 중 깊이 순서를 맞춘 비율이다. 마지막으로 %CorrPxlPairFgr는 무작위로 샘플된 foreground pixel pair에서 ordering이 맞는 비율이다.

Full+PP는 %RcldIns 59.0, %RcldInsPair 29.3, InsPairAcc 90.4, %CorrPxlPairFgr 83.1을 기록한다. 저자들은 특히 마지막 수치를 강조하며, 무작위 foreground pixel pair의 83.1%를 올바르게 정렬했다고 말한다.

비교 baseline인 [28]+Y, [28]+Depth, [28]+Size는 인스턴스 회수율 자체는 baseline segmentation에 의해 제한되고, ordering accuracy도 89.4~94.3 수준이지만 %CorrPxlPairFgr는 모두 14.8에 머문다. 이 값이 매우 낮은 이유는 회수된 인스턴스 수가 적고, 픽셀 수준으로 보면 ordering이 충분히 반영되지 못하기 때문으로 보인다. 반면 제안법은 인스턴스 회수 자체와 ordering을 같이 고려한다.

또한 ablation 결과를 보면 후처리 전에는 pairwise connection이 오히려 ordering을 해칠 수 있다. 예를 들어 Full은 %CorrPxlPairFgr 77.4인데, Unary는 80.9다. 하지만 후처리 후에는 Full+PP가 83.1로 가장 좋다. 이 역시 이 시스템이 여러 구성 요소의 결합으로 성능을 낸다는 점을 보여준다.

### 4.6 정성적 결과와 실패 사례

논문은 성공 사례와 실패 사례를 그림으로 제시한다. 성공 사례는 차량들이 비교적 잘 분리되어 있고 connected component가 인스턴스를 잘 구분해주는 경우다. 특히 서로 떨어져 있는 차량들에서는 지역 CNN 예측과 component 기반 정리가 잘 맞아떨어진다.

실패 사례는 두 가지로 요약된다. 첫째, 아주 작은 차량은 CNN이 놓칠 수 있다. 둘째, connected component 단계에서 서로 다른 차량이 하나로 합쳐지면 이후 MRF도 이를 완전히 회복하기 어렵다. 즉, 이 논문의 병목은 대체로 **작은 원거리 객체와 잘못된 component 분리**에 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 당시 관점에서 매우 도전적인 두 문제, 즉 instance segmentation과 depth ordering을 하나의 프레임워크로 결합했다는 점이다. 단순히 segmentation mask를 만들거나 bounding box를 예측하는 수준을 넘어서, 객체들 사이의 상대적 공간 구조까지 픽셀 수준에서 다룬다. 이는 장면 이해를 한 단계 더 풍부하게 만든다.

또 다른 강점은 detector-free 접근이라는 점이다. 많은 기존 방법이 detection candidate에 크게 의존하는 반면, 이 논문은 patch 기반 CNN과 MRF를 통해 detection과 segmentation을 공동으로 다루려 한다. 따라서 bounding box proposal 품질에 직접적으로 묶이지 않는 장점이 있다.

학습 데이터 측면에서도 실용적인 강점이 있다. 정밀 instance mask가 부족한 상황에서 [2]의 weak 3D supervision을 활용해 대규모 학습 데이터를 만들었다. 이는 실제 연구 환경에서 annotation bottleneck을 우회하는 좋은 예다.

방법론적으로는 지역 추론과 전역 정합을 분리한 설계가 인상적이다. CNN이 지역적 모호성을 줄이고, MRF가 patch 간 충돌을 풀어 주는 구조는 당시 기술 수준에서 합리적이다. 특히 KITTI의 수직축-깊이 상관관계를 connected component ordering에 활용한 점은 단순하지만 효과적인 도메인 prior다.

하지만 한계도 분명하다. 가장 큰 한계는 **문제 정의와 추론이 도메인 특화적**이라는 점이다. 수직 위치로 깊이를 정하는 가정은 도로 장면에서는 맞을 수 있지만, 일반 장면이나 복잡한 3D 구조에서는 취약하다. 논문도 실내 장면으로 확장하는 것을 미래 과제로 언급하지만, 실제로는 훨씬 더 어려워질 가능성이 크다.

또한 patch 기반 접근은 전역 문맥을 직접적으로 보지 못한다. CNN unary가 “local label은 global label의 하한”이라는 가정을 두는 것도 patch가 전체 장면을 다 보지 못하기 때문에 생기는 보정이다. 이 구조는 당시로서는 합리적이지만, end-to-end 전역 reasoning 관점에서는 우회적인 해법이다.

MRF 구성 역시 heuristic한 요소가 많다. long-range connection을 component pair당 20,000개 랜덤 샘플링하는 부분, component ordering을 수직축으로 정하는 부분, 200픽셀 미만 제거 같은 후처리 규칙은 모두 hand-designed prior다. 실제 시스템으로는 동작하지만, 원리적으로 깔끔한 end-to-end 학습 구조와는 거리가 있다.

또한 short-range potential의 정확한 수식과 학습 방식이 본문 추출 텍스트에는 충분히 명시되지 않는다. 따라서 재현 관점에서는 구현 세부가 더 필요해 보인다. 논문 전체를 보더라도 이 부분이 상대적으로 덜 자세할 가능성이 있다.

비판적으로 보면, class-level segmentation에서 CNNRaw가 이미 매우 강하고, MRF의 이득은 주로 instance/depth 쪽에 집중되어 있다. 이는 MRF가 실질적으로 “복잡한 전역 consistency module” 역할을 하긴 하지만, 그 이득이 구성과 후처리에 상당히 민감하다는 뜻이기도 하다. 실제로 ablation에서 pairwise term은 후처리 없이는 종종 해를 끼친다. 따라서 이 논문의 성능 향상은 순수한 모델 설계의 우월성이라기보다, **CNN, MRF, connected components, 후처리의 조합을 잘 맞춘 결과**로 읽는 것이 공정하다.

## 6. 결론

이 논문은 단안 이미지에서 자동차 인스턴스 분할과 depth ordering을 동시에 수행하는 초기의 중요한 시도 중 하나다. 핵심 기여는 세 가지로 정리할 수 있다.

첫째, patch 기반 CNN이 depth-aware instance label을 직접 예측하도록 설계했다. 둘째, 여러 해상도의 patch 예측을 MRF로 통합해 이미지 전체에 대한 일관된 인스턴스 및 깊이 순서를 만든다. 셋째, 3D bounding box와 stereo/LIDAR를 활용한 약한 supervision으로 대규모 학습을 가능하게 했다.

실험적으로는 KITTI에서 baseline 대비 훨씬 나은 instance-level coverage와 의미 있는 depth ordering 성능을 보였다. 특히 foreground/background segmentation 자체보다, 인스턴스 분리와 순서 추론에서 장점이 더 분명했다. 이는 이 논문이 단순 분할이 아니라 “구조적 장면 이해”를 목표로 했음을 잘 보여준다.

오늘날 관점에서 보면, 이 방법은 이후 등장한 end-to-end instance segmentation, monocular depth estimation, transformer 기반 scene parsing에 비해 다소 복잡하고 heuristic하게 보일 수 있다. 그러나 당시에는 detection에 의존하지 않고 segmentation과 ordering을 묶어 다뤘다는 점이 상당히 선구적이었다. 실제 적용 측면에서도 자율주행처럼 객체 간 상대적 거리 관계가 중요한 분야에 직접적인 의미가 있다.

향후 연구 방향으로는, 이런 상대적 ordering 추론을 더 일반적인 장면과 다양한 object category로 확장하는 것, 그리고 CNN-MRF-후처리의 분리된 구조를 더 통합적이고 학습 가능한 방식으로 바꾸는 것이 자연스럽다. 그런 의미에서 이 논문은 완성형 해답이라기보다, **instance-level scene understanding과 3D-aware perception을 연결하는 중요한 중간 단계**로 평가할 수 있다.
