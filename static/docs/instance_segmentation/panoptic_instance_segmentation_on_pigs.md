# Panoptic Instance Segmentation on Pigs

* **저자**: Johannes Brünger, Maria Gentz, Imke Traulsen and Reinhard Koch
* **발표연도**: 2020
* **arXiv**: <https://arxiv.org/abs/2005.10499>

## 1. 논문 개요

이 논문은 양돈 환경에서 개체별 돼지를 **pixel 수준으로 분할**하는 문제를 다룬다. 기존의 computer vision 기반 돼지 탐지 연구는 주로 **bounding box detection** 또는 **keypoint detection**에 의존했는데, 저자들은 이 두 방식이 모두 돼지의 실제 몸체 윤곽을 충분히 표현하지 못한다고 본다. Bounding box는 자세에 따라 배경이나 다른 개체를 과도하게 포함할 수 있고, keypoint는 일부 신체 지점만 제공하므로 개체의 전체 형상 정보가 크게 손실된다. 이 논문은 그 사이의 공백을 메우기 위해 **panoptic segmentation** 관점에서 개별 돼지를 정확히 분리하는 프레임워크를 제안한다.

연구 문제는 단순히 “돼지가 어디 있는가”를 찾는 것이 아니라, **각 픽셀이 어느 돼지에 속하는지**를 식별하는 것이다. 이는 행동 분석, 자세 및 이동 추적, 개체 간 상호작용 분석, 더 나아가 몸체 면적을 활용한 크기나 체중 추정 등으로 확장될 수 있다. 특히 축사 환경은 조명 변화, 야간 적외선 모드, 렌즈 오염, 개체 간 겹침(occlusion), 높은 밀집도 등으로 인해 일반적인 객체 탐지보다 훨씬 까다롭다. 이런 조건에서 개체별 정밀 분할이 가능하다면 자동 모니터링의 신뢰도와 활용도가 크게 높아진다.

논문은 하나의 단일 해법만 제시하는 것이 아니라, 점진적으로 난이도를 높이는 네 가지 실험을 통해 돼지 분할 문제를 체계적으로 다룬다. 구체적으로 **binary segmentation**, **categorical segmentation**, **instance segmentation with discriminative embedding**, 그리고 **orientation recognition**까지 포함한 확장 실험을 제시한다. 즉, 이 논문은 단순한 성능 보고보다는 돼지 분할 문제를 위한 비교적 범용적인 segmentation framework를 설계하고, 각 구성 방식의 장단점을 실험적으로 검토한 작업이라고 볼 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **semantic segmentation 기반의 U-Net 프레임워크를 확장하여 panoptic instance segmentation 문제를 해결**하는 것이다. 핵심은 두 가지다.

첫째, 저자들은 bounding box나 keypoint 대신 **ellipse annotation**을 채택한다. 돼지를 위에서 내려다보는 카메라 환경에서는 돼지 몸체가 대체로 타원 형태로 근사되므로, ellipse는 bounding box보다 훨씬 적은 배경을 포함하면서도 keypoint보다 풍부한 형상 정보를 제공한다. 또한 타원의 면적은 동물의 부피나 체중과도 대략적인 상관이 있을 수 있어 응용 측면에서도 의미가 있다. 즉, annotation 자체를 문제 구조에 맞게 설계했다는 점이 중요하다.

둘째, instance를 분리하는 방식으로 두 갈래를 비교한다. 하나는 **categorical segmentation**으로, 각 돼지를 직접 구분하는 대신 픽셀을 `background`, `outer edge`, `inner core` 같은 클래스에 할당해 나중에 개체를 복원하는 방식이다. 다른 하나는 **pixel embedding** 기반 instance segmentation으로, 같은 개체의 픽셀은 embedding space에서 가깝게, 다른 개체의 픽셀은 멀어지도록 학습한 뒤 clustering으로 instance를 복원한다. 후자는 De Brabandere et al.의 discriminative loss를 가져와 돼지 개체 분리에 적용한 것이다.

이때 저자들의 차별점은 embedding만 쓰지 않고, **binary segmentation과 embedding을 결합한 two-head 구조**를 제안했다는 데 있다. Binary segmentation head가 먼저 foreground인 돼지 영역만 추려주고, embedding head는 이 foreground 픽셀들만 clustering 대상으로 사용하게 함으로써 계산량을 줄이고 불필요한 background clustering을 피한다. 즉, segmentation과 clustering을 독립 단계로 놓지 않고 **공유 backbone 위의 상호보완적인 multi-head 학습 구조**로 묶었다는 점이 핵심 설계다.

또한 orientation recognition에서는 binary segmentation 대신 **body-part segmentation**을 결합한다. Ellipse는 $180^\circ$ 회전 대칭이기 때문에 단순히 타원 fitting만으로는 머리/꼬리 방향을 구분할 수 없는데, `head`와 `body` 클래스를 예측하게 하여 이 모호성을 해소한다. 이 역시 segmentation 결과를 단순 탐지에 그치지 않고 행동 이해에 가까운 방향으로 확장하려는 설계다.

## 3. 상세 방법 설명

전체 방법은 공통 backbone 위에 여러 실험별 output head를 얹는 형태로 구성된다. 기본 네트워크는 **U-Net auto-encoder**이며, encoder는 분류용 backbone, decoder는 업샘플링과 skip connection을 통해 원래 해상도의 dense prediction을 복원한다. 논문에서는 주로 **ResNet34**와 **Inception-ResNet-v2**를 encoder backbone으로 사용했다. backbone은 ImageNet pretrained weight로 초기화되며, 각 해상도 단계의 특징을 decoder와 skip connection으로 결합한다.

### 3.1 돼지 표현 방식과 라벨 설계

논문은 픽셀 수준 인스턴스 분할을 목표로 하지만, 완전한 픽셀 단위 수작업 polygon annotation은 비용이 크다. 그래서 저자들은 **ellipse annotation**을 선택한다. 이 선택은 단순한 편의가 아니라 문제 구조에 맞는 공학적 타협이다. 위에서 내려다본 돼지의 몸체는 타원으로 충분히 잘 근사되며, 축의 방향을 통해 개체 방향도 일정 부분 표현할 수 있다.

겹치는 돼지의 경우에는 label image 상에서 깊이 순서를 반영해, 카메라에 더 가까운 개체가 뒤의 개체 픽셀을 덮어쓰도록 한다. 이렇게 하면 최종 label image에서 각 픽셀은 정확히 하나의 개체에만 할당된다. panoptic segmentation 문제 정의와도 맞아떨어지는 설계다.

### 3.2 Binary segmentation

가장 단순한 실험은 각 픽셀이 돼지인지 배경인지만 예측하는 **binary segmentation**이다. 입력 이미지의 각 픽셀 $x_i$에 대해, 네트워크는 해당 픽셀이 돼지일 확률 $p(x_i)$를 출력한다. 라벨 $y_i \in {0,1}$는 배경이면 0, 돼지면 1이다.

출력 head는 채널 수 1개이며 sigmoid activation을 사용한다. 손실 함수는 binary cross-entropy이다.

$$
L=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log p(x_i) + (1-y_i)\log(1-p(x_i))\right]
$$

추론 시에는 출력 확률을 thresholding하여 최종 binary mask를 얻는다. 이 실험 자체는 인스턴스 구분은 못하지만, 이후 embedding clustering에서 foreground만 골라내는 데 중요한 기반이 된다.

### 3.3 Categorical segmentation

두 번째 실험은 각 픽셀을 `background`, `outer edge`, `inner core`의 3개 클래스로 분류하는 **categorical segmentation**이다. 이 아이디어는 instance를 직접 ID로 분리하는 대신, 돼지 중심부와 경계를 구분하는 구조를 학습시켜 **붙어 있는 돼지들을 분리 가능하게 만들려는 접근**이다.

여기서 `inner core`는 원래 ellipse를 축소한 버전이다. 즉, 각 개체 내부에 서로 분리된 핵심 영역이 생기도록 라벨을 구성한다. 네트워크 출력 채널은 $C=3$이고, softmax를 적용한다. 손실 함수는 categorical cross-entropy이다.

$$
L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} t_{i,j}\log(x_{i,j})
$$

여기서 $t_{i,j}$는 one-hot ground truth, $x_{i,j}$는 픽셀 $i$에서 클래스 $j$의 예측 확률이다.

추론 후에는 `inner core` blob를 찾아 각 blob를 하나의 동물로 해석한다. 그 다음 Fitzgibbon의 ellipse fitting 알고리즘을 적용해 core 영역에 타원을 맞추고, 원래 축소했던 비율만큼 다시 키워서 최종 돼지 ellipse를 복원한다. 즉, 이 방법은 **semantic label을 intermediate representation으로 사용해 instance를 복원**하는 방식이다.

### 3.4 Discriminative loss 기반 instance segmentation

세 번째 실험은 논문의 기술적 핵심 중 하나다. 각 픽셀을 고차원 embedding space로 보낸 뒤, 같은 돼지에 속하는 픽셀은 서로 가까운 cluster를 이루고 다른 돼지 픽셀은 멀어지도록 학습한다. 이때 사용한 것이 **discriminative loss**다.

각 object $c$에 대해 해당 픽셀 embedding의 평균을 $\mu_c$라 하자. 손실은 크게 세 항으로 구성된다.

첫째, **variance term**은 같은 인스턴스 내부 픽셀들이 자기 cluster mean 근처에 모이도록 만든다.

$$
L_{var}=\frac{1}{C}\sum_{c=1}^{C}\frac{1}{N_c}\sum_{i=1}^{N_c}\left[\lVert \mu_c-x_i \rVert-\delta_v\right]^2_+
$$

여기서 $\delta_v$는 같은 cluster 내부 허용 반경이다. 픽셀이 평균에서 $\delta_v$ 안쪽에 있으면 더 이상 벌점을 주지 않는다. 즉, 한 점으로 완전히 붕괴시키는 대신 일정 반경 안에만 모이면 된다.

둘째, **distance term**은 서로 다른 인스턴스의 cluster center들이 충분히 멀어지도록 강제한다.

$$
L_{dist}=\frac{1}{C(C-1)}\sum_{c_A=1}^{C}\sum_{c_B=1, c_B\neq c_A}^{C}\left[2\delta_d-\lVert \mu_{c_A}-\mu_{c_B} \rVert\right]^2_+
$$

$\delta_d$는 서로 다른 cluster 중심 간 최소 거리 역할을 한다. 중심 간 거리가 충분히 크면 페널티가 없다.

셋째, **regularization term**은 embedding 전체가 지나치게 멀리 퍼지는 것을 막는다.

$$
L_{reg}=\frac{1}{C}\sum_{c=1}^{C}\lVert \mu_c \rVert
$$

최종 손실은

$$
L=\alpha L_{var}+\beta L_{dist}+\gamma L_{reg}
$$

로 정의된다. 논문에서는 $\alpha=\beta=1.0$, $\gamma=0.001$을 사용했다. 또한 embedding 차원은 8차원이고, threshold는 $\delta_v=0.1$, $\delta_d=1.5$로 두었다.

이 구조의 직관은 분명하다. 각 픽셀을 “이 픽셀은 어느 돼지에 속하는가”를 직접 분류하는 대신, **같은 돼지 픽셀끼리 비슷한 표현을 갖도록 representation space를 학습**하는 것이다. 그러면 실제 instance ID 수가 이미지마다 달라도 clustering으로 유연하게 처리할 수 있다.

### 3.5 Clustering과 combined segmentation

Embedding을 얻은 뒤에는 이를 실제 인스턴스로 바꿔야 한다. 원 논문인 De Brabandere et al.은 mean-shift를 사용했지만, 본 논문은 **HDBSCAN**을 사용한다. 저자들은 HDBSCAN이 고차원 embedding space에서 더 나은 성능을 보였다고 설명한다. HDBSCAN은 density-based hierarchical clustering이므로 cluster 개수가 고정되지 않은 상황, 노이즈가 있는 상황, 밀도 기반 구조를 찾는 데 적합하다.

하지만 모든 픽셀을 clustering하면 데이터 수가 너무 많다. HD 해상도에서는 픽셀 수가 백만 단위가 될 수 있기 때문이다. 이를 해결하기 위해 저자들은 **binary segmentation과 embedding을 결합한 combined segmentation**을 도입한다. 먼저 binary head가 foreground mask를 생성하고, 그 mask 내부의 픽셀만 embedding clustering 대상으로 사용한다. 이 방식은 계산량 절감뿐 아니라 background noise 제거 효과도 있다.

구조적으로는 shared U-Net backbone 위에 두 개의 head를 둔다. 하나는 binary segmentation head, 다른 하나는 pixel embedding head다. 각각의 loss를 계산한 뒤 backbone으로 gradient를 공동 전달한다. 논문 표현대로라면 두 head의 gradient는 동일 가중으로 backbone에 반영된다. 이 설계 덕분에 foreground 분리와 instance embedding 학습이 서로 보완될 수 있다.

추론 후에는 cluster별로 image space 픽셀을 모아 타원을 fitting한다. 최종 평가는 이 추출된 ellipse 단위로 수행된다.

### 3.6 Orientation recognition

단순 ellipse fitting만으로는 돼지의 방향을 완전히 알 수 없다. 긴 축 방향은 알 수 있지만, 머리가 어느 쪽인지 꼬리가 어느 쪽인지는 구분되지 않기 때문이다. 이는 타원이 $180^\circ$ 대칭이기 때문이다.

이를 해결하기 위해 저자들은 combined model의 binary head를 **body-part segmentation head**로 바꿔 `background`, `body`, `head` 클래스를 예측하도록 했다. 이렇게 얻은 head/body 정보와 fitted ellipse를 결합하면 머리 방향을 결정할 수 있다. 즉, orientation recognition은 별도 detection task가 아니라 **instance segmentation 결과를 행동 분석 수준으로 확장하는 부가 과제**로 설계되어 있다.

### 3.7 구현 세부 사항

구현에는 `segmentation_models` 라이브러리가 사용되었고, optimizer는 Adam, 초기 learning rate는 $10^{-4}$이다. 입력 이미지는 계산량과 clustering 비용을 줄이기 위해 **640×512**로 축소하였다. 데이터 증강은 `imgaug`로 수행했으며, 기하 변환, 색 변화, grayscale 변환 등을 포함해 학습 데이터 양을 10배로 늘렸다. 이는 낮/밤, 적외선, 렌즈 오염 같은 환경 변화를 반영하기 위한 것으로 해석할 수 있다.

클러스터링에서 HDBSCAN의 **minimum cluster size는 100**으로 설정되었다. 이 값은 실제 돼지 한 마리가 차지하는 최소 픽셀 수와 연결되는 하이퍼파라미터다.

## 4. 실험 및 결과

### 4.1 데이터셋

논문은 기존 공개 데이터셋이 충분하지 않다고 보고 자체 데이터셋을 구축했다. 일반적인 piglet rearing house에서 촬영된 영상으로부터 **1000장의 프레임**을 랜덤 선택해 수작업으로 annotation했다. 카메라는 총 5대가 설치되었고, 각 카메라는 두 개의 펜을 덮는다. 각 펜에는 최대 13마리의 돼지가 있었다. 돼지들은 생후 27일에 입식되어 40일간 시설에 머물렀고, 데이터 수집은 총 4개월 동안 이루어졌다.

학습/검증/테스트 분할은 **카메라 기준 분리**가 핵심이다. 즉, 테스트 이미지는 학습에 사용하지 않은 별도 카메라에서 수집되었다. 따라서 동일한 카메라 시점에 대한 random split보다 일반화 평가가 더 엄격하다. 다만 논문 스스로도 테스트 셋이 학습 셋과 “근본적으로 다른” 분포는 아니라고 인정한다.

분할은 다음과 같다.

* train: 606장
* validation: 168장
* test: 226장

낮과 night vision 이미지가 섞여 있으며, test는 daylight 108장, nightvision 118장이다. 낮에도 렌즈 오염 때문에 night vision이 잘못 활성화되는 경우가 있었다고 한다. 이는 실제 축사 환경의 난점을 잘 보여준다.

### 4.2 평가 지표

이 논문은 panoptic segmentation 문제 정의에 맞춰 **Panoptic Quality (PQ)**를 주요 지표로 사용한다. PQ는 예측 segment와 ground truth segment를 IoU 기준으로 matching한 뒤, 매칭된 쌍의 IoU 합을 TP, FP, FN에 기반해 정규화한 값이다.

$$
PQ=\frac{\sum_{(p,g)\in TP} IoU(p,g)}{|TP|+\frac{1}{2}|FP|+\frac{1}{2}|FN|}
$$

PQ는 단순 검출 성공 여부뿐 아니라 **얼마나 정밀하게 segmentation이 맞았는지**까지 반영한다는 점에서 이 문제에 적절하다. 추가로 비교 가능성을 위해 F1, precision, recall도 함께 보고한다. Binary segmentation 자체의 픽셀 수준 정확도는 **Jaccard index**로 평가한다.

### 4.3 Binary segmentation 결과

Binary segmentation은 매우 높은 픽셀 수준 정확도를 보였다. Jaccard index 기준으로

* ResNet34: 0.9730
* Inception-ResNet-v2: 0.9735

였다. 낮 영상에서는 약 0.977, night vision에서는 약 0.969 수준이다.

이는 돼지와 배경의 구분 자체는 상당히 안정적으로 가능함을 의미한다. 다만 논문은 annotation이 ellipse 기반이라 실제 돼지 몸체와 완전히 일치하지 않는 부분이 있어, 이론적으로 100% 정확도는 어렵다고 설명한다. 즉, 모델이 실제 몸체 윤곽을 따라가더라도 라벨은 이상적인 ellipse이므로 일부 오차가 남을 수 있다.

### 4.4 Categorical segmentation 결과

Categorical segmentation에서는 `inner core`를 원래 ellipse 크기의 50%로 설정했다. 이 방식은 개체 중심부가 서로 분리되도록 만들어 instance를 복원하는 전략이다.

주요 결과는 다음과 같다.

* **ResNet34**

  * PQ: 0.7920
  * F1: 0.9550
  * categorical accuracy: 0.9612

* **Inception-ResNet-v2**

  * PQ: 0.7943
  * F1: 0.9541
  * categorical accuracy: 0.9612

즉, ellipse 단위의 개체 검출 성능은 **F1 약 95.4~95.5%**, PQ는 약 **0.79** 수준이다. 낮 영상이 night vision보다 약간 더 좋은 성능을 보였지만, 밤 환경에서도 큰 성능 저하는 아니다. 이는 단순한 semantic segmentation 변형만으로도 상당한 수준의 instance 분리가 가능하다는 뜻이다.

### 4.5 Combined instance segmentation 결과

Binary head와 embedding head를 결합한 combined approach의 결과는 다음과 같다.

* **ResNet34**

  * PQ: 0.7966
  * F1: 0.9513
  * binary accuracy: 0.9722

* **Inception-ResNet-v2**

  * PQ: 0.7921
  * F1: 0.9481
  * binary accuracy: 0.9707

흥미로운 점은, 기대와 달리 combined approach가 categorical segmentation을 압도하지는 않았다는 것이다. PQ는 약간 높거나 비슷하지만, F1은 오히려 categorical segmentation보다 조금 낮다. 저자들은 이 결과를 실제 데이터셋에서 **아주 복잡한 겹침 사례가 충분히 많지 않았기 때문**이라고 해석한다. 이론적으로는 embedding 기반 방식이 겹침에 더 강해야 하지만, 학습 데이터가 그 장점을 충분히 드러내는 사례를 많이 담고 있지 않았다는 것이다.

그럼에도 중요한 관찰은 있다. Shared backbone에서 embedding과 binary segmentation을 동시에 학습했는데도 **binary segmentation 정확도가 거의 유지**되었다. 저자들은 이를 두 과제 사이의 **synergy effect** 가능성으로 해석한다. 즉, foreground 분리와 instance 분리가 서로 방해하지 않고 함께 학습될 수 있다는 점이다.

### 4.6 Orientation recognition 결과

Orientation recognition은 correctly detected pigs, 즉 true positive로 판정된 개체들에 대해 방향을 맞췄는지를 평가한다.

* ResNet34: orientation accuracy 0.9428
* Inception-ResNet-v2: orientation accuracy 0.9226

즉, 올바르게 탐지된 돼지 중 약 **92~94%**에서 머리 방향까지 맞췄다. 이는 단순한 객체 검출을 넘어 행동 분석에서 유용한 방향 정보까지 확보할 수 있음을 보여준다. 동시에 PQ와 categorical accuracy도 유지되어, body/head segmentation을 추가한다고 해서 ellipse detection이 크게 악화되지는 않았다.

### 4.7 정밀 지표 분석

세부적으로 보면 categorical segmentation의 경우 ResNet34에서

* Precision: 0.9586
* Recall: 0.9514

combined segmentation의 경우 ResNet34에서

* Precision: 0.9544
* Recall: 0.9482

였다. 전반적으로 precision과 recall 모두 약 95% 전후다. 밤 영상에서는 낮 영상보다 소폭 떨어지지만, night vision이나 렌즈 오염의 존재를 고려하면 상당히 견고한 결과다.

### 4.8 Ablation study

저자들은 입력 해상도를 더 낮춘 **320×256** 환경에서 추가 ablation을 수행했다. Backbone으로 ResNet34, Inception-ResNet-v2, EfficientNet-B5를 비교했고, architecture로 U-Net과 FPN도 비교했다. 결과적으로 모든 조합이 크게 다르지 않았다.

예를 들어 combined U-Net에서 F1은 대체로 0.93~0.95 수준, combined FPN에서도 비슷하다. PQ 역시 대체로 0.77~0.79 수준이다. 저자들은 이를 두 가지로 해석한다.

첫째, 이 문제에서는 backbone이나 architecture의 차이가 그렇게 결정적이지 않을 수 있다.
둘째, 더 중요한 이유로는 **데이터셋 규모와 다양성이 제한적**이어서 더 복잡한 backbone의 이점이 드러나지 않았을 수 있다.

또한 discriminative loss의 threshold 하이퍼파라미터인 $\delta_v$, $\delta_d$를 grid search했지만, 일정 범위 내에서는 성능이 거의 비슷했다. 즉, clustering 품질이 이 하이퍼파라미터에 지나치게 민감하지는 않았다는 뜻이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 매우 현실적이라는 점이다. 축사 내 돼지 관찰은 실제로 조명 변화, 오염, 적외선 전환, 개체 겹침 등 어려움이 많고, 저자들은 이 조건을 무시한 실험실 수준의 데이터가 아니라 실제 환경 데이터를 사용했다. 또한 학습/테스트를 카메라 단위로 분리하여 최소한의 일반화 검증을 수행했다.

두 번째 강점은 **문제에 맞는 annotation 표현과 출력 표현을 설계**했다는 점이다. Ellipse annotation은 수작업 비용과 표현력을 균형 있게 맞춘 선택이며, 이로부터 binary, categorical, embedding, orientation까지 여러 수준의 supervision을 유도해냈다. 즉, annotation 설계와 모델 설계가 잘 연결되어 있다.

세 번째 강점은 프레임워크가 단일 목적이 아니라 **확장 가능한 구조**라는 점이다. 같은 backbone 위에서 segmentation head만 바꿔 다양한 과제를 처리할 수 있고, 실제로 orientation recognition까지 무리 없이 확장했다. 이런 구조는 향후 body-part analysis나 behavior recognition과의 결합에도 유리하다.

네 번째 강점은 결과가 실제로 상당히 좋다는 것이다. F1 약 95%, PQ 약 0.79, orientation accuracy 약 94%는 논문이 다루는 난도 높은 환경을 고려하면 인상적이다. 특히 기존 공개 비교가 제한적이긴 하지만, 저자들이 언급한 Psota et al.의 precision 91%, recall 67%와 비교하면 본 논문의 수치가 훨씬 높다. 다만 데이터셋과 실험 조건이 다르므로 직접 우열 비교는 조심해야 한다.

한편 한계도 분명하다.

첫째, 데이터셋이 **1000장**으로 비교적 작고, 공개되지 않은 것으로 보인다. 따라서 재현성과 외부 비교 가능성이 제한된다. 논문도 공개 데이터셋 부재를 문제로 지적한다.

둘째, 실제 핵심 기여처럼 보이는 embedding 기반 instance segmentation이 categorical segmentation을 뚜렷하게 능가하지 못했다. 이는 방법 자체의 한계라기보다 데이터셋의 복잡한 겹침 사례 부족 때문일 수 있지만, 결과만 놓고 보면 복잡한 방법이 단순한 방법보다 명확히 낫다고 말하기 어렵다.

셋째, 평가 단위가 최종적으로 **ellipse extraction**에 많이 의존한다. 즉, 네트워크가 만드는 픽셀 분할의 품질이 최종적으로 ellipse fitting으로 요약된다. 이 설계는 응용 목적상 합리적이지만, 진정한 의미의 자유형 instance mask 품질을 직접 평가한 것은 아니다. 다시 말해, “pixel accurate segmentation”을 주장하지만 실제 supervision과 evaluation 모두 ellipse 근사에 강하게 묶여 있다.

넷째, annotation rule의 엄밀성이 충분히 표준화되어 있지 않다. 논문도 경계 사례, 화면 가장자리 개체, 관찰 펜 밖의 돼지 등에 대한 정의가 애매해 평가가 어려움을 인정한다. 이는 false positive/false negative 해석에도 영향을 준다.

다섯째, 논문은 weight estimation 가능성을 동기와 응용으로 언급하지만, 실제로 **체중 추정 실험은 수행하지 않았다**. 따라서 segmentation이 weight estimation에 유용할 가능성은 제시되지만, 논문 내에서 직접 검증된 결론은 아니다.

비판적으로 보면, 이 논문은 segmentation 문제를 상당히 정교하게 다루지만, 결과의 많은 부분이 **타원 기반 표현의 성공**에 기대고 있다. 따라서 자유형 자세 변화, 심한 비정상 자세, 부분 가림이 더 극단적인 데이터에서는 성능이 어떻게 변할지 아직 불확실하다. 또한 보다 대규모 데이터셋과 cross-farm generalization 실험이 있었다면 논문의 실용적 설득력이 더 커졌을 것이다.

## 6. 결론

이 논문은 돼지 개체 탐지 문제를 bounding box나 keypoint 수준에서 한 단계 발전시켜, **개체별 pixel-accurate segmentation** 관점으로 재구성한 연구다. U-Net 기반 segmentation framework 위에 binary, categorical, discriminative embedding, body-part segmentation head를 조합함으로써, 돼지의 위치뿐 아니라 개체 분리와 방향 추정까지 수행한다.

실험적으로는 실제 축사 환경 데이터에서 약 **95% 수준의 F1**과 약 **0.79 수준의 PQ**를 보여, 조명 불량, night vision, 렌즈 오염, occlusion이 있는 조건에서도 강한 성능을 확인했다. 또한 orientation recognition까지 약 94% 정확도로 가능함을 보였다. 중요한 점은 이 결과가 단순 detection을 넘어, 행동 분석, 개체 추적, 체형 및 체중 추정 같은 후속 응용으로 이어질 수 있는 형태의 출력을 제공한다는 데 있다.

논문이 직접 보여준 가장 실질적인 기여는 두 가지로 정리할 수 있다. 첫째, 양돈 영상에서 개체별 segmentation을 수행하는 **실용적인 프레임워크**를 제시했다는 점이다. 둘째, ellipse annotation과 segmentation 기반 설계를 통해, 비용이 큰 정밀 마스크 annotation 없이도 유용한 instance-level 정보를 얻을 수 있음을 보였다는 점이다.

향후 연구 방향으로는 더 다양한 환경과 더 큰 규모의 데이터셋에서 일반화를 검증하는 것이 중요하다. 특히 다른 농장, 다른 카메라 높이, 다른 연령대의 돼지, 더 심한 가림 상황에 대한 검증이 필요하다. 또한 논문이 동기로 제시한 체중 추정, 건강 상태 평가, 행동 패턴 분석까지 연결된다면, precision livestock farming에서 실질적인 영향력이 더 커질 가능성이 높다.
