# Learning Gaussian Instance Segmentation in Point Clouds

이 논문은 3D point cloud에서의 **instance segmentation** 문제를 다룬다. 저자들은 기존 3D instance segmentation이 box proposal, anchor, clustering 후처리 등에 크게 의존한다는 점을 문제로 보고, 이를 대신해 **instance center의 분포를 Gaussian heatmap으로 직접 예측**하는 **Gaussian Instance Center Network(GICN)**를 제안한다. GICN은 예측된 center heatmap에서 소수의 center candidate만 선택한 뒤, 각 center에 대해 instance size를 예측하고, 이를 기반으로 bounding box와 instance mask를 산출하는 **single-stage, anchor-free, end-to-end** 구조다. 논문은 이 설계가 ScanNet과 S3DIS에서 state-of-the-art 성능을 달성했다고 주장한다.  

## 1. Paper Overview

이 논문의 핵심 문제는 **3D 장면에서 각 개별 객체를 분리해 mask와 class label을 함께 예측하는 것**이다. 2D instance segmentation에 비해 3D point cloud 환경은 데이터가 sparse하고 구조가 복잡하며, point 수가 많아 연산 비용이 높다. 기존 방법들은 voxel 기반이거나 point-cloud 기반으로 나뉘지만, 많은 point-based 방법은 metric learning + clustering, fixed box proposals, 혹은 anchor-based box prediction에 의존한다. 저자들은 이러한 접근이 구조적으로 복잡하고 후처리에 의존적이라고 본다.

논문은 이 문제를 **“instance center를 먼저 잘 찾고, 그 center가 결정한(localized) 정보로 box와 mask를 순차적으로 예측하는 문제”** 로 재정의한다. 즉, proposal을 대량 생성한 뒤 걸러내는 방식이 아니라, 장면 전체에서 객체 중심의 확률 지도를 먼저 만들고, 거기서 소수의 유력 후보만 골라 후속 예측을 수행한다. 이 설계는 계산량을 줄이고, 학습 중간 결과를 heatmap으로 직관적으로 시각화할 수 있다는 장점도 갖는다.  

## 2. Core Idea

논문의 핵심 아이디어는 다음과 같다.

> **3D instance segmentation을 bounding-box proposal 생성 문제로 보지 말고, instance center distribution을 Gaussian heatmap으로 학습하는 문제로 바꾸자.**

저자들은 각 object instance의 중심 부근에 높은 값을 갖는 **Gaussian center heatmap**을 예측하고, 여기서 center candidate를 고른다. 이후 각 center에 대해 instance size를 예측하여 feature extraction 범위를 적응적으로 정하고, 그 범위 안에서 bounding box와 mask를 추정한다. 즉, center가 전체 파이프라인을 “dictate”하는 구조다.  

이 아이디어의 novelty는 세 가지로 정리할 수 있다.

첫째, **anchor box가 필요 없다.**
둘째, **대량 proposal과 NMS가 필요 없다.**
셋째, **instance size를 center별로 적응적으로 추정**하여 고정 반경 기반 aggregation의 한계를 줄인다.  

즉, GICN의 핵심은 단순히 center heatmap을 예측하는 것이 아니라, **center prediction → size-aware local feature extraction → box/mask prediction** 을 하나의 일관된 single-stage 구조로 묶었다는 점이다.

## 3. Detailed Method Explanation

### 3.1 Overall Pipeline

GICN은 세 개의 subnet으로 구성된다.

1. **Center Prediction Network** $\Phi_C$
2. **Bounding-Box Prediction Network** $\Phi_B$
3. **Mask Prediction Network** $\Phi_M$  

입력 point cloud에서 global/local feature를 추출한 뒤, 먼저 center prediction network가 각 point가 object center일 확률을 예측해 heatmap $Q={Q_i}_{i=1}^N$를 만든다. 그다음 center selection mechanism으로 소수의 center 후보를 고르고, 각 후보 중심으로 instance size를 예측해 적절한 neighborhood를 정한다. 마지막으로 이 size-aware feature를 사용해 3D bounding box와 point-level instance mask를 출력한다.  

### 3.2 Center Prediction Network

입력 point cloud를

$$
\mathcal{P}={p_i=(x_i,y_i,z_i)}_{i=1}^{N}
$$

라고 하자. $\Phi_C$는 각 point $p_i$가 어떤 3D object instance의 center일 확률을 예측한다. backbone은 **PointNet++** 이며, global feature와 point-wise feature를 추출한 뒤, 여기에 추가 fully connected layers를 붙여 최종적으로 sigmoid를 거친 point-wise 확률값을 만든다. heatmap은

$$
Q={Q_i}_{i=1}^{N}, \quad Q_i \in [0,1]
$$

형태다.

이때 ground-truth heatmap은 discrete한 center label이 아니라 **continuous relaxation** 으로 만든다. 각 instance에 대해 centroid에 가장 가까운 point를 Gaussian center로 잡고, 그 point로부터의 거리를 Gaussian function에 넣어 각 point의 heatmap 값을 생성한다. 즉, 실제 중심 주변 point들이 부드럽게 높은 값을 갖도록 학습 target을 만든다. 이 점은 object center를 one-hot처럼 두는 것보다 학습을 안정화하는 효과가 있다.  

### 3.3 Center Selection Mechanism

heatmap이 예측되면, 상위 확률 point만 골라 center candidate로 쓰면 될 것 같지만, 같은 instance의 중심 주변 point들이 cluster를 이루므로 단순 정렬만 하면 **동일 instance에서 중복된 center** 가 여러 번 선택될 수 있다. 이를 막기 위해 논문은 **center selection mechanism** 을 제안한다. 선택된 center 하나마다, 해당 class의 representative radius 안에 있는 나머지 high-probability point들의 heatmap 값을 0으로 만들어 이후 후보에서 제거한다. 이 radius는 coupled semantic network가 예측한 class와 training data의 class-wise 평균 instance size를 이용해 정한다.

논문은 이 과정을 **사전 NMS와 유사한 역할**로 해석한다. 즉, box를 많이 뽑아놓고 나중에 NMS를 하는 대신, center 단계에서 이미 후보를 잘 분리한다는 것이다. 그 결과 GICN은 **후처리 NMS 없이도** 덜 중복적인 box/mask를 얻을 수 있다. 논문에서 selection threshold는 예시로 $Q_\theta = 0.4$, $T_\theta = 64$를 사용한다.  

### 3.4 Bounding-Box Prediction

Center를 고른 뒤에는 각 center에 대해 box를 예측해야 한다. 이때 GICN의 중요한 설계는 **instance size prediction** 이다. 저자들은 고정 반경으로 주변 point를 모으는 것이 비효율적이라고 본다. 작은 객체와 큰 객체에 같은 aggregation radius를 적용하면 정보가 부족하거나 과도하게 섞일 수 있기 때문이다. 그래서 각 center별로 instance size를 먼저 추정하고, 그 크기에 맞는 neighborhood에서 feature를 추출한다. 이를 논문은 **adaptive instance size selection** 이라고 강조한다.  

이 설계는 VoteNet과의 비교에서 더 분명해진다. VoteNet은 고정 aggregation radius를 사용하지만, GICN은 point cloud 분포에 맞춰 적절한 cluster size를 결정한다고 설명한다. 즉, 단순 “center를 찾는” 것만이 아니라, **center-conditioned adaptive context modeling** 이 box prediction의 핵심이다.

또한 hollow shape 예시도 논문이 언급한다. 예를 들어 bathtub처럼 중심 부근에 point가 거의 없는 객체는 중심 근처 point 기반 설계가 어려울 수 있는데, 논문은 이 경우에도 중심에 가장 가까운 point를 고르고, size prediction이 충분히 넓은 범위를 잡아 대부분의 point cloud를 덮도록 만들 수 있다고 설명한다.

### 3.5 Mask Prediction

Bounding-box prediction network와 함께, mask prediction network는 선택된 center와 size-aware feature를 바탕으로 최종 point-level instance mask를 생성한다. 핵심은 heatmap에서 localization된 center 정보가 이후 단계 전체에 전달되므로, mask prediction도 독립적 clustering 없이 진행된다는 점이다. 즉, 많은 기존 방법처럼 embedding을 뽑고 mean-shift 같은 clustering을 거치지 않는다. GICN은 center 후보 수가 적기 때문에 적은 수의 객체에 대해서만 mask를 추론하면 된다.  

결국 GICN은 **center-first localization** 을 통해 box와 mask를 함께 예측하는 구조이며, 이는 “object detection + semantic segmentation을 단순 결합한 것”보다 더 직접적인 instance-level reasoning 경로를 제공한다.

### 3.6 Architecture and Training Details

구현은 PyTorch로 이루어졌고, backbone은 PointNet++다. 학습률은 0.002에서 시작해 20 epoch마다 절반으로 줄이며, optimizer는 Adam을 사용한다. 보통 50 epoch 전후에서 수렴한다고 한다. 장면은 training 시 1m$^3$ cube로 나누고 stride 0.5m의 sliding window를 사용하며, test 시 각 cube 결과를 block-merging algorithm으로 합쳐 전체 scene segmentation을 만든다.

이러한 설정은 기존 SGPN, 3D-BoNet 등과 공통적인 부분이 있지만, GICN의 차별점은 여전히 **center heatmap 기반 single-stage pipeline** 에 있다.

## 4. Experiments and Findings

### 4.1 Datasets and Metrics

논문은 주로 **S3DIS** 와 **ScanNet v2** 에서 실험한다. ScanNet에서는 18개 object class에 대해 평가하며, metric은 **AP@50% (IoU threshold 0.5)**다. S3DIS와 ScanNet 모두 학습 시 scene을 1m$^3$ 단위 cube로 나누어 처리한다.  

### 4.2 Main Results

논문은 GICN이 **ScanNet과 S3DIS 모두에서 state-of-the-art** 를 달성했다고 주장한다. abstract, introduction, ScanNet 결과 section, conclusion이 모두 이 메시지를 일관되게 반복한다. 저자들에 따르면 GICN은 당시 출판된 기존 3D instance segmentation 방법들과 비교해 가장 높은 성능을 보였다.

또한 qualitative result에서, 인접한 instance들이 가까이 있어도 GICN은 잘 분리하는 반면 3D-BoNet은 올바른 mask를 생성하지 못한 사례가 제시된다. 이는 center heatmap과 distance-constrained selection이 복잡한 장면에서 실제로 유효하다는 정성적 근거다.

### 4.3 Efficiency

GICN은 정확도뿐 아니라 **효율성**도 강점으로 내세운다. 논문은 mean-shift clustering이나 NMS 같은 추가 post-processing이 필요 없고, center selection 덕분에 적은 수의 predicted instance만 다루면 되므로 다른 방법보다 빠르다고 설명한다. 특히 fixed number의 box를 많이 예측하는 3D-BoNet보다 효율적이라고 주장한다.

이 부분은 GICN의 방법론적 장점과도 일치한다.
“많이 뽑고 나중에 정리”가 아니라, “처음부터 소수의 중요한 center만 뽑는다”는 설계가 곧 연산량 감소로 이어진다.

### 4.4 Ablation Study

논문은 S3DIS Area-5에서 ablation study를 수행해 각 핵심 component의 효과를 검증한다고 밝힌다. visible snippet에서는 표 전체 수치가 다 보이지 않지만, 적어도 저자들이 **각 key component가 성능에 어떤 영향을 주는지 정량적으로 검증했다** 는 점은 분명하다. 즉, 제안된 center heatmap, size-aware design, selection mechanism이 단순 조합이 아니라 실제 성능에 기여하는지 확인하려 했다.  

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **문제 재정의 자체가 깔끔하다**는 점이다. 많은 3D instance segmentation 방법이 embedding 학습과 clustering, 혹은 box proposal 생성으로 복잡해지는 반면, GICN은 “center distribution을 예측한다”는 하나의 명확한 중간 표현을 둔다. 이 표현은 시각화도 쉽고, 오류 분석도 쉽다.  

또한 **anchor-free + NMS-free + size-aware** 조합은 매우 실용적이다. 중심을 찾고, 크기를 예측하고, 그 크기에 맞는 feature로 box/mask를 추정하는 흐름은 직관적이며 연산 효율 측면에서도 설득력이 있다.  

마지막으로, hollow shape나 가까운 center 후보 사례처럼 method failure mode를 논문이 직접 논의한다는 점도 좋다. 이는 설계가 단순 아이디어 수준이 아니라 실제 point-cloud geometry의 특수성을 의식하고 있음을 보여준다.

### Limitations

한계도 있다. 논문 스스로 언급하듯, **vertical surface처럼 중심을 정의하거나 찾기 어려운 class** 에서는 성능이 상대적으로 약하다. 예를 들어 curtain, picture 같은 클래스는 compact structure를 가진 toilet, bathtub보다 center identification이 어렵다고 한다. 이건 center-based formulation의 구조적 약점일 수 있다. 객체 중심이 geometry상 뚜렷하지 않으면 heatmap supervision 자체가 애매해질 수 있기 때문이다.

또한 GICN은 center를 기준으로 주변 context를 모으는 구조이므로, 복잡한 비정형 shape에서는 중심에서의 local aggregation이 항상 최적이라고 보기 어렵다. hollow shape에 대한 논의는 이를 어느 정도 보완하지만, 본질적으로 이 방식은 **center locality 가정** 위에 서 있다. 이는 embedding-based global grouping과 다른 trade-off다.

### Interpretation

비판적으로 보면, GICN은 “3D instance segmentation의 일반 원리”라기보다 **center-centric design philosophy** 를 강하게 밀어붙인 방법이다. 하지만 바로 그 점이 강점이기도 하다. 3D point cloud에서 proposal이나 clustering보다, 실제 객체 중심을 잘 추정하는 것이 downstream의 여러 문제를 단순화할 수 있다는 통찰을 설득력 있게 보여준다.

현재 시점에서 transformer 기반 3D scene understanding이 많이 발전했더라도, 이 논문의 아이디어는 여전히 유효하다.
즉, **좋은 중간 표현 하나를 설계하면 복잡한 후처리와 다단계 파이프라인을 상당 부분 제거할 수 있다**는 교훈을 준다.

## 6. Conclusion

이 논문은 3D point cloud instance segmentation을 위해 **Gaussian Instance Center Network(GICN)**를 제안했다. GICN은 instance center의 공간 분포를 Gaussian heatmap으로 예측하고, 그로부터 소수의 center candidate를 선택한 뒤, 각 center에 대해 size-aware하게 box와 mask를 예측하는 **single-stage, anchor-free, end-to-end** 구조다. 저자들은 이 접근이 ScanNet과 S3DIS에서 state-of-the-art 성능을 달성했다고 보고한다.  

이 논문의 가장 중요한 메시지는 다음과 같다.

**3D instance segmentation에서 proposal이나 clustering을 직접 다루기보다, instance center distribution을 먼저 잘 학습하면 더 단순하고 효율적인 파이프라인으로도 높은 성능을 얻을 수 있다.**

저자들은 향후 difficult semantic classes의 center finding accuracy를 더 개선하고, MTML 같은 metric learning을 결합해 visual semantic reasoning을 더 강화하는 방향을 future work로 제시한다.
