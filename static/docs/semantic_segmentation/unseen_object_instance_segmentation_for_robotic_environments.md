# Unseen Object Instance Segmentation for Robotic Environments

- **저자**: Christopher Xie, Yu Xiang, Arsalan Mousavian, Dieter Fox
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2007.08073

## 1. 논문 개요

이 논문은 로봇이 훈련 때 본 적 없는 물체를 테이블탑 환경에서 각각 분리해 인식할 수 있도록 하는 문제, 즉 Unseen Object Instance Segmentation(UOIS)를 다룬다. 목표는 물체의 카테고리 이름을 맞히는 것이 아니라, 장면 안에 있는 각 개별 물체를 정확한 instance mask로 분할하는 것이다. 이는 로봇이 새로운 물체를 집거나 조작해야 하는 실제 환경에서 매우 중요하다.

논문이 문제 삼는 핵심 어려움은 두 가지다. 첫째, unseen object를 잘 다루려면 매우 다양한 물체가 포함된 대규모 학습 데이터가 필요한데, 로봇 환경에서는 이런 데이터셋이 거의 없다. 둘째, synthetic data로 학습하는 것은 매력적이지만, synthetic RGB와 real RGB 사이의 domain gap이 커서 단순히 RGB-D를 함께 넣어 학습하면 실제 환경으로 잘 일반화되지 않는다. 특히 depth는 비교적 sim-to-real generalization이 잘 되지만, noisy sensor 때문에 mask 경계가 부정확해질 수 있고, RGB는 경계를 날카롭게 만들 수 있지만 synthetic RGB의 비사실성 때문에 직접 쓰면 성능이 떨어진다.

이 논문은 이 딜레마를 해결하기 위해 RGB와 depth를 한 번에 섞지 않고, 각자의 장점을 분리해서 쓰는 2-stage 구조 UOIS-Net을 제안한다. 1단계는 depth만 이용해 물체별 초기 마스크를 만들고, 2단계는 RGB를 이용해 그 마스크 경계를 정교하게 다듬는다. 저자들은 이 방식이 비사실적인 synthetic RGB로 학습해도 surprisingly real-world에 잘 일반화된다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “depth는 objectness와 geometry를 잘 일반화하고, RGB는 경계를 날카롭게 만들어 준다”는 점을 분리해서 활용하는 것이다. 기존에는 synthetic RGB-D를 한 번에 입력으로 넣어 학습하거나, RGB 기반 instance segmentation을 class-agnostic하게 바꾸는 접근이 많았지만, 이 논문은 그런 방식이 unseen tabletop object segmentation에서는 비효율적이라고 본다.

제안된 UOIS-Net은 두 단계로 구성된다. 첫 번째 단계인 Depth Seeding Network(DSN)은 depth만 보고 각 픽셀이 어떤 물체 중심을 가리키는지 예측하고, 이를 바탕으로 거친 initial mask를 만든다. 두 번째 단계인 Region Refinement Network(RRN)은 RGB와 initial mask를 함께 보고 경계를 실제 물체 외곽선에 맞게 수정한다. 즉, 어려운 문제를 한 번에 풀지 않고, “대략적인 instance 분리”와 “정확한 경계 복원”으로 분해한 것이 핵심이다.

논문에서 특히 중요한 차별점은 다음과 같다. 첫째, RGB-D를 통합 입력으로 직접 쓰지 않고 RGB와 depth의 역할을 구조적으로 분리했다. 둘째, refinement task는 전체 장면 segmentation보다 쉬운 문제라고 보고, local crop과 initial mask 조건을 주면 non-photorealistic synthetic RGB에서도 잘 학습된다는 점을 실험으로 보였다. 셋째, 기존 2D center voting 구조를 넘어 3D center voting 구조를 도입해 clutter와 occlusion에 더 강하게 만들었다. 넷째, cluttered scene에서 center vote cluster가 잘 분리되도록 separation loss를 새로 제안했다.

## 3. 상세 방법 설명

전체 파이프라인은 DSN, Initial Mask Processor(IMP), RRN의 세 부분으로 이루어진다. 입력은 한 장의 RGB-D 이미지이다. 먼저 DSN이 depth에서 initial instance masks를 만들고, 그 마스크를 IMP가 정리한 뒤, 마지막으로 RRN이 RGB를 이용해 refined mask를 출력한다. DSN 내부에 non-differentiable post-processing이 포함되므로, DSN과 RRN은 end-to-end가 아니라 별도로 학습된다.

### DSN: depth만 사용하는 초기 instance mask 생성

DSN의 입력은 depth map을 camera intrinsics로 backprojection해서 만든 organized point cloud $D \in \mathbb{R}^{H \times W \times 3}$이다. 출력은 공통적으로 semantic segmentation $F$와 center-related prediction이다. semantic class는 background, tabletop, tabletop objects의 3개다.

#### 2D DSN

2D 버전에서는 각 픽셀에 대해 object center를 향하는 2D unit direction $V \in \mathbb{R}^{H \times W \times 2}$를 예측한다. 여기서 center는 보이는 object mask의 평균 pixel 위치다. 네트워크 backbone은 U-Net이고, 그 위에 foreground segmentation branch와 center direction branch가 붙는다.

초기 마스크를 만들기 위해 Hough voting layer를 사용한다. 각 foreground pixel이 예측한 방향이 어느 위치를 가리키는지 누적해서, 여러 방향에서 충분히 지지받는 pixel을 object center 후보로 선택한다. 이후 각 foreground pixel을 자신이 가리키는 가장 가까운 center에 할당하여 instance mask를 만든다. 이 과정에는 inlier threshold, distance threshold, percentage threshold, NMS가 포함되어 false positive를 줄인다.

2D DSN의 손실 함수는 foreground loss와 direction loss의 합이다.

foreground loss는 weighted cross entropy이다.
$$
\ell_{fg} = \sum_i w_i \, \ell_{ce}(F_i, \bar{F}_i)
$$
여기서 $w_i$는 class imbalance를 보정하기 위한 가중치다.

direction loss는 object pixel에서는 GT direction과 cosine similarity를 맞추고, background와 table pixel은 고정 방향 $[0,1]$을 향하게 한다.
$$
\ell_{dir} =
\sum_{i \in O} \alpha_i (1 - V_i^\top \bar{V}_i)
+
\frac{\lambda_{bt}}{|B \cup T|}
\sum_{i \in B \cup T}
(1 - V_i^\top [0,1])
$$
여기서 $\alpha_i$는 instance 크기 보정을 위한 가중치이며, 작은 물체와 큰 물체가 비슷한 비중으로 학습되도록 한다.

#### 3D DSN

저자들은 2D reasoning이 가진 한계를 지적한다. 예를 들어 물체의 2D 중심이 다른 물체에 가려져 있으면 해당 물체를 검출하지 못할 수 있다. 이를 해결하기 위해 3D DSN을 제안한다.

3D DSN은 semantic segmentation $F$는 그대로 출력하지만, 각 픽셀에 대해 3D center offset $V' \in \mathbb{R}^{H \times W \times 3}$를 예측한다. 그러면 각 픽셀의 predicted center vote는 $D + V'$가 된다. 이 center vote들을 3D 공간에서 mean shift clustering으로 묶어 instance를 만든다. Gaussian kernel은 다음과 같다.
$$
K(x, y) = \exp\left(\frac{-\|x-y\|_2^2}{\sigma^2}\right)
$$
논문 원문에서는 지수 내부를 $\frac{1}{\sigma^2}\|x-y\|_2^2$ 형태로 적고 있으며, 실제 의미는 Gaussian mean shift bandwidth $\sigma$가 cluster의 퍼짐 정도를 조절한다는 점이다.

3D DSN은 receptive field를 키우기 위해 일부 convolution layer를 ESP module로 교체했다. 저자들은 이것이 cluttered scene에서 성능 향상에 도움을 준다고 실험으로 보인다.

3D DSN은 네 가지 손실을 사용한다.

첫째, foreground loss $\ell_{fg}$는 2D와 동일하다.

둘째, center offset loss $\ell_{co}$는 predicted center vote가 GT object center에 가깝도록 하는 Huber loss다.
$$
\ell_{co} = \sum_{i \in \Omega} w_i \, \rho(D_i + V'_i - c_i)
$$
여기서 $c_i$는 pixel $i$가 속한 GT object의 3D 중심이다.

셋째, clustering loss $\ell_{cl}$는 mean shift를 몇 step unroll한 뒤, 같은 object의 point는 가깝게, 다른 object의 point는 margin $\delta$ 이상 멀어지게 만든다.
$$
\ell_{cl}^{(l)}(Z^{(l)}, X, Y)
=
\sum_{i=1}^S \sum_{j \in O}
w_{ij}\mathbf{1}\{y_i=y_j\} d^2(Z_i^{(l)}, X_j)
+
w_{ij}\mathbf{1}\{y_i \ne y_j\}[\delta - d(Z_i^{(l)}, X_j)]_+^2
$$
그리고 전체 clustering loss는 여러 iteration의 합이다.
$$
\ell_{cl} = \ell_{cl}^{(1)} + \cdots + \ell_{cl}^{(L)}
$$

넷째, 이 논문의 핵심 기여 중 하나인 separation loss $\ell_{sep}$는 center vote들이 꼭 물체의 정확한 중심에 있을 필요는 없지만, 다른 물체의 vote들과 잘 분리되도록 유도한다. 먼저 각 vote가 모든 object center에 얼마나 가까운지 softmax 확률처럼 계산한다.
$$
M_{ij} =
\frac{\exp(-\tau d(c_j, D_i + V'_i))}
{\sum_{j'=1}^{J} \exp(-\tau d(c_{j'}, D_i + V'_i))}
$$
그리고 GT center에 대한 cross entropy를 적용한다. 이 손실은 같은 물체의 center vote를 모으는 동시에 다른 물체 중심과 멀어지게 해서, 후속 mean shift clustering을 쉽게 만든다. cluttered scene에서 특히 중요하다고 논문은 주장한다.

최종 3D DSN 손실은 다음과 같다.
$$
\lambda_{fg}\ell_{fg} + \lambda_{co}\ell_{co} + \lambda_{cl}\ell_{cl} + \lambda_{sep}\ell_{sep}
$$

### IMP: Initial Mask Processing

DSN이 만든 initial mask는 salt-and-pepper noise, 구멍, 조각난 component를 포함할 수 있다. 이를 바로 RRN에 넣으면 refinement가 잘 되지 않는다. 그래서 저자들은 각 instance mask마다 간단한 이미지 처리 후처리를 적용한다.

먼저 opening(erosion 후 dilation)으로 작은 잡음을 제거하고, 다음으로 closing(dilation 후 erosion)으로 작은 구멍을 메운다. 마지막으로 largest connected component만 남긴다. 이 모듈은 학습 기반이 아니지만, 전체 시스템의 안정성에 큰 기여를 한다고 ablation에서 보인다.

### RRN: RGB 기반 경계 정교화

RRN은 RGB와 initial mask를 함께 사용해 경계를 object edge에 맞게 조정한다. 입력은 RGB 3채널과 initial mask 1채널을 합친 4채널 crop이다. crop은 instance mask 주변을 약간 padding해서 자르고, $224 \times 224$로 resize한다. 출력은 refined mask probability $R \in \mathbb{R}^{224 \times 224}$이다.

네트워크 구조는 DSN과 같은 U-Net이며, loss는 foreground/background 2-class weighted cross entropy를 사용한다. 중요한 점은 RRN이 “장면 전체에서 모든 object를 찾는 문제”를 푸는 것이 아니라, 이미 주어진 initial mask 하나를 local하게 수정하는 문제만 푼다는 것이다. 저자들은 그래서 synthetic RGB가 photorealistic하지 않아도 real image로 transfer가 잘 된다고 해석한다.

RRN 학습을 위해서는 noisy initial mask가 필요하지만, synthetic DSN 출력은 오히려 너무 깨끗해서 학습에 불리했다고 한다. 그래서 GT mask에 다양한 perturbation을 가하는 mask augmentation을 사용한다. 포함되는 변형은 translation/rotation, 일부 영역의 add/cut, erosion/dilation 같은 morphological operation, random ellipse 추가/삭제이다. 이 설계는 실제 depth 기반 initial mask에서 생기는 과대분할, 과소분할, 경계 왜곡, 작은 구멍 등을 모사하기 위한 것이다.

## 4. 실험 및 결과

### 데이터셋

학습용 synthetic dataset으로 저자들은 Tabletop Object Dataset(TOD)를 새로 만들었다. 이 데이터셋은 4만 개의 scene으로 구성되며, SUNCG 실내 환경 안에 ShapeNet table과 ShapeNet object를 무작위로 배치해 생성했다. 각 scene에는 5개에서 25개의 물체가 놓이며, stacking도 포함된다. PyBullet으로 물리 시뮬레이션을 돌려 물체가 안정된 상태가 되도록 한 뒤, depth와 비사실적 RGB를 렌더링한다. 해상도는 $640 \times 480$이고, table plane label과 각 object의 instance label이 제공된다.

평가용 real dataset은 OCID와 OSD다. OCID는 2346장의 이미지이며 semi-automatic label을 사용해 경계 노이즈가 있다. OSD는 111장으로 더 작지만 수작업 라벨이라 경계 품질이 높다. 또한 RRN의 sim-to-real gap을 보기 위해 Google Open Images Dataset v5(OID)의 실제 segmentation mask도 사용했다.

### 평가 지표

논문은 Overlap P/R/F와 Boundary P/R/F를 사용한다. Overlap은 예측 mask와 GT mask 간 pixel overlap 중심의 지표이고, Hungarian matching으로 instance를 대응시킨 뒤 precision, recall, F-measure를 계산한다. Boundary는 mask 경계의 정확도를 보기 위한 보조 지표다. RRN이 경계를 개선하는 모듈이므로, 단순 overlap만 보면 효과가 충분히 드러나지 않는다는 문제의식에서 추가되었다.

### 주요 정량 결과

논문은 기존 로봇용 segmentation baseline들과 비교했을 때 UOIS-Net-2D와 UOIS-Net-3D가 모두 우수하다고 보고한다.

OCID/OSD에서 SOTA 비교 결과를 보면, UOIS-Net-3D는 전반적으로 Mask R-CNN과 PointGroup를 능가한다. 예를 들어 Table II에서 OCID 기준 UOIS-Net-3D는 Overlap F 86.4, Boundary F 76.2를 기록했고, PointGroup는 각각 80.1, 71.7, Mask R-CNN(RGB-D)은 78.0, 69.2다. OSD에서도 UOIS-Net-3D는 Overlap F 83.3, Boundary F 71.2로, PointGroup의 78.8, 65.4와 Mask R-CNN(RGB-D)의 74.1, 53.8보다 높다.

2D에서 3D로 바뀌면서 가장 두드러진 개선은 recall 증가다. 이는 3D center voting이 occlusion과 thin object에 더 강하기 때문으로 해석된다. 논문은 2D에서 object center가 가려지는 경우를 중요한 failure mode로 들고, 3D reasoning이 이를 상당 부분 해결한다고 보였다.

한편 흥미로운 결과는 synthetic RGB를 직접 입력으로 쓰는 것이 효과적이지 않다는 점이다. UOIS-Net-2D에서 DSN에 RGB-D를 직접 넣은 경우 성능이 오히려 떨어졌다. 반대로 depth-only DSN에 RGB refinement를 나중에 붙이는 구조가 더 좋았다. 이는 synthetic RGB를 “처음부터 segmentation feature로 쓰는 것”보다, “이미 나온 mask를 다듬는 보조 정보로 쓰는 것”이 transfer에 유리하다는 저자들의 주장을 뒷받침한다.

### RRN의 효과

OSD처럼 경계 라벨이 깨끗한 데이터셋에서는 RRN의 효과가 뚜렷하다. Table III에서 RRN 없이 depth 기반 DSN만 썼을 때 UOIS-Net-3D의 OSD Boundary F는 70.0인데, Table II의 full pipeline에서는 71.2이고, RRN을 OID real image로 학습한 경우 Table IV에서 77.3까지 올라간다. 반면 OCID는 GT boundary 자체가 noisy해서, qualitative하게는 더 좋아져도 quantitative boundary score가 덜 올라가거나 오히려 떨어질 수 있다고 논문은 명시한다.

또한 Table V는 IMP와 RRN이 둘 다 중요함을 보인다. DSN raw mask에 바로 RRN을 적용하면 성능이 오히려 악화되며, open/close morphology와 connected component 정리를 한 뒤에야 RRN이 제대로 작동한다. 즉, refinement network가 아무 초기 조건에서도 마법처럼 복구하는 것은 아니고, 어느 정도 정돈된 initial mask가 필요하다.

### 3D loss ablation

Table VI는 3D DSN의 loss 구성요소를 검증한다. $\ell_{fg} + \ell_{co}$만 쓸 경우 OCID Boundary F는 68.9, OSD Boundary F는 69.8 수준이다. 여기에 separation loss $\ell_{sep}$를 추가하면 각각 78.8, 74.6으로 크게 상승한다. 마지막으로 clustering loss $\ell_{cl}$까지 함께 쓰면 79.9, 77.3으로 더 좋아진다. 이 결과는 separation loss가 cluttered environment에서 핵심이라는 논문의 주장을 강하게 지지한다.

### ESP module과 하이퍼파라미터

Table VII에 따르면 DSN에 ESP module을 넣어 receptive field를 넓히면 2D와 3D 모두 성능이 향상된다. 또한 $\tau$와 $\sigma$에 대한 ablation에서는, $\tau$가 너무 작으면 separation loss가 지나치게 강해져 성능이 떨어지고, 어느 정도 큰 값에서 plateau를 형성한다. $\sigma$는 mean shift clustering의 bandwidth 역할을 하므로 너무 작으면 over-segmentation, 너무 크면 under-segmentation이 발생한다.

### Sim-to-Real 일반화 평가

TOD test set에서는 오히려 Mask R-CNN과 PointGroup가 UOIS-Net보다 더 높은 점수를 얻는다. 예를 들어 Table VIII에서 PointGroup는 Boundary F 91.1, Mask R-CNN은 90.9인 반면 UOIS-Net-3D는 85.5다. 그런데 real-world dataset에서는 UOIS-Net이 더 강하다. 저자들은 이를 통해 단순한 synthetic test 성능보다 real-world distribution shift 대응이 더 중요하며, UOIS-Net이 그 점에서 더 낫다고 해석한다.

### 로봇 grasping 적용

논문은 Franka 로봇과 wrist-mounted RGB-D camera를 사용해 cluttered table에서 물체를 집어 bin으로 옮기는 실험도 수행했다. UOIS-Net으로 instance segmentation을 한 뒤, 가장 가까운 물체의 point cloud를 6-DOF GraspNet에 넣어 grasp를 생성한다. 9개 장면에서 총 51개 물체를 대상으로 실험했고, 최대 2회 시도 기준 41개 성공으로 80.3% success rate를 보고했다. 실패 원인은 segmentation 오류와 grasp planning 오류가 모두 포함된다. 특히 drill처럼 highly non-convex object는 과분할되어 grasp에 불리한 경우가 있었다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 synthetic RGB-D를 단순 결합하지 않고, modality별 역할을 분리한 설계가 매우 설득력 있고 실험적으로도 잘 검증되었다는 점이다. depth-only seeding과 RGB refinement의 조합은 sim-to-real 문제를 우회하는 실용적인 해법이다. 특히 RRN이 non-photorealistic RGB에서도 실제 경계 복원에 꽤 잘 일반화된다는 결과는 흥미롭고, 로봇 환경에서 데이터 수집 비용을 크게 줄일 가능성이 있다.

또 다른 강점은 3D DSN의 설계와 손실 함수 분석이 비교적 정교하다는 점이다. 단순히 2D에서 3D로 확장한 것이 아니라, mean shift clustering과 맞물리도록 center vote를 학습시키고, separation loss로 clutter 상황을 직접 겨냥했다. 이 부분은 PointGroup 같은 기존 3D instance segmentation과 비교했을 때 본 논문의 독자적인 기술적 기여로 볼 수 있다.

실험 구성도 강점이다. baseline 비교, SOTA 비교, ablation, qualitative failure case, sim-to-real 분석, 실제 robot grasping demonstration까지 포함되어 있어 방법의 장단점을 비교적 넓게 보여 준다. 특히 failure mode를 숨기지 않고 2D와 3D 각각의 한계를 시각적으로 제시한 점은 reviewer 관점에서 긍정적이다.

하지만 한계도 분명하다. 첫째, 문제 설정이 tabletop environment에 강하게 특화되어 있다. 논문은 floor scene에도 어느 정도 generalize된다고 말하지만, 본질적으로는 table-supported object arrangement를 중심으로 설계되었고, 더 일반적인 장면으로의 확장은 직접 입증되지 않았다. 둘째, RRN은 initial mask 품질에 의존한다. DSN이 크게 잘못 묶은 instance를 RRN이 완전히 분리해내지는 못한다. 즉, refinement는 근본적인 instance grouping 오류를 고치기 어렵다.

셋째, 3D DSN도 여전히 under-segmentation과 over-segmentation 문제를 가진다. 예를 들어 서로 가까운 작은 물체들은 cluster가 합쳐질 수 있고, flat surface가 연속적으로 붙은 cereal boxes는 depth만으로 분리가 어렵다. highly non-convex object는 여러 조각으로 나뉘는 경우도 있다. 넷째, mean shift clustering의 bandwidth $\sigma$와 separation 관련 하이퍼파라미터 $\tau$에 성능이 민감하다. 논문도 이를 ablation으로 보여 주지만, 자동 적응 방식은 잘 되지 않았다고 밝힌다.

비판적으로 보면, RRN이 synthetic RGB에서 잘 일반화된다는 해석은 매우 흥미롭지만, 왜 그런지가 이론적으로 깊게 분석되지는 않는다. 저자들은 “local crop 기반의 mask refinement가 전체 segmentation보다 쉬운 문제이기 때문”이라고 해석하지만, 이는 실험적 가설에 가깝다. 또한 TOD와 실제 데이터 간의 object appearance, material, lighting 다양성이 어느 정도까지 커버되는지도 논문만으로는 완전히 알기 어렵다.

## 6. 결론

이 논문은 unseen object instance segmentation을 위한 실용적이고 강력한 구조인 UOIS-Net을 제안한다. 핵심 기여는 depth를 이용한 초기 instance seeding과 RGB를 이용한 mask refinement를 분리한 2-stage 설계, 3D center voting 기반 DSN, clutter 환경에서 중요한 separation loss, 그리고 대규모 synthetic Tabletop Object Dataset(TOD) 구축이다.

결과적으로 이 방법은 비사실적인 synthetic RGB-D만으로 학습하고도 real-world tabletop scenes에서 Mask R-CNN과 PointGroup를 능가하는 성능을 보였고, 실제 로봇 grasping에도 적용 가능함을 입증했다. 특히 “synthetic depth의 일반화 능력”과 “RGB의 경계 복원 능력”을 구조적으로 분리해 활용한 발상은 로봇 perception에서 매우 실용적이다. 앞으로는 더 일반적인 장면, 더 복잡한 non-convex object, 더 강한 occlusion, 그리고 end-to-end 또는 adaptive clustering으로 확장하는 방향이 중요한 후속 연구가 될 가능성이 크다.
