# Amodal Instance Segmentation

* **저자**: Ke Li, Jitendra Malik
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1604.08202](https://arxiv.org/abs/1604.08202)

이 논문은 2016년에 발표된 Ke Li와 Jitendra Malik의 연구이며, ECCV 2016에 게재되었다.

## 1. 논문 개요

이 논문은 **amodal instance segmentation** 문제를 다룬다. 일반적인 instance segmentation, 즉 **modal instance segmentation**은 이미지에서 실제로 보이는 물체 영역만 분할한다. 반면 amodal instance segmentation은 **보이는 부분뿐 아니라 가려진 부분까지 포함한 전체 물체 영역**을 예측하는 문제다. 예를 들어 말의 몸통 일부가 다른 물체에 가려져 있어도, 사람은 보이지 않는 부분까지 포함한 전체 말의 형태를 어느 정도 일관되게 상상할 수 있는데, 이 논문은 바로 그 능력을 컴퓨터 비전 모델로 구현하려고 한다.

연구 문제는 단순히 “가려진 부분을 복원하자”가 아니다. 실제로는 한 물체의 보이는 부분만 보고 숨겨진 부분의 정확한 형태를 정하는 일이 본질적으로 모호할 수 있다. 특히 사람 다리처럼 articulated object는 가려진 부분에 대해 여러 개의 그럴듯한 가설이 동시에 가능하다. 그럼에도 불구하고 저자들은 이 문제가 충분히 의미 있고 실용적인 문제라고 본다. 이유는 amodal mask를 얻으면 거기서 곧바로 **occlusion의 위치, 범위, 경계**, 나아가 **물체 간 상대적 깊이 순서** 같은 정보를 유도할 수 있기 때문이다.

이 문제가 중요한 또 다른 이유는, 많은 후속 문제들이 사실상 amodal segmentation으로 환원되기 때문이다. 예를 들어 어떤 물체가 얼마나 가려졌는지, 누가 앞에 있고 누가 뒤에 있는지, 가려진 물체의 전체 bounding box가 무엇인지 같은 문제는 모두 amodal mask가 있으면 더 자연스럽게 다룰 수 있다. 따라서 이 문제를 푸는 것은 단일 segmentation task 이상의 의미를 가진다.

하지만 가장 큰 장애물은 **학습 데이터 부족**이다. 당시에는 공개된 amodal segmentation annotation이 거의 없었기 때문에, 기존 supervised learning 방식으로는 직접 학습하기 어려웠다. 이 논문의 핵심 공헌은 바로 이 난점을 우회하는 데 있다. 저자들은 “가려진 부분을 되살리는 것은 어렵지만, 멀쩡한 물체를 일부러 가리는 것은 쉽다”는 관찰에서 출발해, 기존의 **modal annotation만으로 synthetic occlusion 데이터를 만들어** amodal segmentation 모델을 학습한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 단순하면서도 강력하다. 원래 문제는 “보이는 부분만 있는 이미지에서 전체 물체를 복원하는 것”인데, 이를 정면으로 풀기보다 반대로 접근한다. 즉, 이미 전체가 보이는 물체가 있는 데이터를 가지고 **인위적으로 다른 물체를 덧씌워 occlusion을 만든 뒤**, 덧씌우기 전의 원래 mask를 정답 amodal mask처럼 사용하는 것이다.

이 방식의 장점은 명확하다. 실제 amodal ground truth가 없어도, 기존 modal instance segmentation annotation만 있으면 학습용 데이터를 대량으로 만들 수 있다. 저자들은 이 synthetic data generation을 통해 “공개된 amodal annotation 부재”라는 근본 문제를 우회한다.

또 하나의 핵심은 **Iterative Bounding Box Expansion**이다. 테스트 시에는 물체의 amodal bounding box를 알 수 없기 때문에, 초기에는 modal bounding box만 주어진 상태에서 시작한다. 이후 모델이 예측한 amodal heatmap을 보고 bounding box 바깥쪽에 여전히 object signal이 남아 있으면 그 방향으로 박스를 확장한다. 이 과정을 반복해 최종 amodal bounding box를 찾는다. 즉, 이 논문은 단순히 amodal mask만 예측하는 것이 아니라, **mask와 box를 함께 점진적으로 확장해 가며 추론하는 방식**을 제안한다.

기존 접근과의 차별점은 두 가지다. 첫째, 저자들 주장에 따르면 이 방법은 **일반 목적 amodal instance segmentation의 첫 알고리즘적 접근**이다. 둘째, supervised amodal annotation 없이 학습한다는 점에서 당시 기존 segmentation 연구와 분명히 구별된다.

## 3. 상세 방법 설명

### 전체 파이프라인

전체 시스템은 크게 두 단계로 이해할 수 있다.

첫째는 **training data generation** 단계다. 기존 modal segmentation annotation만 있는 데이터셋에서 synthetic occlusion이 포함된 학습 샘플을 만든다.

둘째는 **prediction** 단계다. 테스트 시 modal bounding box와 category를 입력으로 받아, modal segmentation heatmap과 이미지 patch를 바탕으로 amodal heatmap을 예측하고, 그 heatmap을 이용해 bounding box를 반복적으로 확장한다.

### 3.1 학습 데이터 생성

저자들은 SBD annotation이 포함된 PASCAL VOC 2012 train set을 사용한다. 하나의 training example을 만들 때 절차는 다음과 같다.

먼저 이미지 하나를 고르고, 그 안의 객체 하나를 **main object**로 선택한다. 그 다음 main object를 충분히 포함하는 random crop을 만든다. 이후 다른 이미지들에서 객체를 샘플링해 이 crop 위에 random position과 scale로 overlay한다. overlay할 때는 해당 객체의 modal segmentation mask를 alpha matte처럼 사용한다. 이렇게 하면 원래는 보였던 main object의 일부가 새로 올려진 객체에 의해 가려지게 된다.

여기서 중요한 점은, **composite image는 부분적으로 가려진 상태지만, 원래 main object의 mask는 그대로 보존된다**는 것이다. 따라서 그 원래 mask는 composite image에 대한 “사실상의 amodal mask” 역할을 할 수 있다.

그 뒤 main object의 아직 보이는 부분만 감싸는 가장 작은 bounding box를 찾는다. 이것이 composite patch에서의 ground-truth modal box다. 테스트 환경의 검출 오차를 모사하기 위해, 이 bounding box는 다시 약간 jittering된다.

마지막으로 target segmentation mask를 만든다. 이 target은 3종류 label을 갖는다.

* main object에 속하는 픽셀은 **positive**
* background에 속하는 픽셀은 **negative**
* 다른 객체에 속하는 픽셀은 **unknown**

이 label design은 매우 중요하다. 가려진 부분은 원래 물체일 수도 있지만, composite image 상에서 다른 객체가 차지하고 있는 위치이므로 확실히 positive라고 할 수 없다. 반대로 background는 물체가 가려질 수 없는 영역으로 취급되므로 negative가 된다. 즉, 저자들은 “무엇을 확실히 알고 무엇을 모르는가”를 target mask 설계에 반영했다.

### 3.2 샘플링의 구체적 조건

논문은 synthetic occlusion이 너무 과하거나 비현실적이 되지 않도록 여러 제약을 둔다.

주요 물체를 포함하는 crop은 main object bounding box와 각 축 기준으로 최소 70% 이상 겹치도록 뽑는다. crop의 각 차원 길이는 object bounding box 해당 차원의 70%에서 200% 사이로 샘플링된다.

overlay할 객체 수는 0개에서 2개 사이의 정수로 랜덤 선택한다. overlay object는 main object와 어느 정도 겹치도록 배치되며, shortest dimension의 평균 크기가 patch 대응 차원의 약 75% 정도가 되도록 scale된다.

각 overlay 이후에는 main object의 visible proportion이 30% 아래로 떨어졌는지 확인한다. 만약 너무 많이 가려졌다면 마지막 연산을 취소하고 다시 시도한다. 즉, 학습 샘플이 지나치게 어려워지지 않도록 제한한다.

또 최종 modal bounding box는 실제 visible region을 감싸는 box와 각 차원 기준 최소 75% 이상 겹치고, 크기는 최대 10% 정도만 다르도록 jittering한다.

### 3.3 네트워크 입력과 출력

모델은 세 가지 입력을 받는다.

첫째, image patch
둘째, **modal segmentation heatmap**
셋째, object category

출력은 **amodal segmentation heatmap**이다.

네트워크 구조는 IIS(Iterative Instance Segmentation)와 같은 아키텍처를 사용한다. 이 구조는 Hariharan 등의 **Hypercolumn architecture**를 바탕으로 하며, VGG-16 기반의 O-Net을 사용한다. 핵심은 서로 다른 intermediate layer의 feature map들을 upsample해서 합치는 방식으로, fine-scale의 low-level 정보와 coarse-scale의 semantic 정보를 동시에 활용한다는 점이다.

IIS는 여기에 추가적으로 category-dependent heatmap channel을 입력으로 받는다. 원래 IIS는 자신의 이전 heatmap prediction을 다음 iteration 입력으로 넣어 refinement를 반복할 수 있는데, 이 논문에서는 그 구조를 빌려와 **modal heatmap을 입력 조건으로 사용하는 amodal predictor**를 학습한다.

### 3.4 입력 전처리

각 training sample에 대해 modal bounding box 내부의 image patch를 잘라서 $224 \times 224$로 비등방성 리사이즈한다. 그 다음 IIS를 1회 적용해 modal segmentation heatmap을 얻는다. 이 heatmap을 원래 patch 좌표계에 맞춰 정렬한 뒤 bilinear interpolation으로 다시 $224 \times 224$로 upsample한다.

흥미로운 점은, 모델이 실제로는 “현재 보이는 patch보다 더 넓은 영역의 amodal mask”를 예측해야 하기 때문에, 입력 patch를 만들 때 원래 image patch의 사방에서 10%씩 제거한 뒤 다시 $224 \times 224$로 리사이즈한다는 것이다. 이렇게 하면 모델은 상대적으로 좁은 시야를 입력받고, 그보다 바깥까지 확장된 mask를 예측하도록 강제된다.

만약 이 새 patch에서 visible object pixel이 10% 미만이면 샘플을 버리고 다시 생성한다. 이는 학습이 너무 불안정해지는 것을 막기 위한 장치로 보인다.

입력 정규화도 수행한다. 이미지 patch는 mean pixel을 빼서 center하고, modal heatmap은 각 원소를 $-127$에서 $128$ 범위로 변환한다.

### 3.5 손실 함수와 학습

모델은 stochastic gradient descent with momentum으로 end-to-end 학습된다. mini-batch size는 32, 초기 가중치는 IIS에서 가져온다.

손실 함수는 **known label이 있는 픽셀들에 대해서만** pixel-wise negative log likelihood를 합한 값이다. 즉, unknown 영역은 loss 계산에서 제외된다. 이를 수식 형태로 쓰면 개념적으로 다음과 같이 볼 수 있다.

$$
\mathcal{L} = \sum_{i \in \Omega_{\text{known}}} w_i \cdot \big(- \log p(y_i \mid x_i)\big)
$$

여기서 $\Omega_{\text{known}}$은 positive 또는 negative로 라벨된 픽셀 집합이고, $w_i$는 patch upsampling factor에 반비례하는 instance-specific weight다. 논문은 정확한 클래스 수식 전체를 적지는 않았지만, 본질은 **unknown을 무시하는 weighted pixel-wise classification loss**다.

학습 hyperparameter는 다음과 같다.

* learning rate: $10^{-5}$
* weight decay: $10^{-3}$
* momentum: $0.9$
* iteration: 50,000

### 3.6 테스트 시 추론: Iterative Bounding Box Expansion

테스트에서는 modal bounding box와 category가 주어진다고 가정한다. modal heatmap은 IIS로 구한다. 그다음 proposed model이 amodal heatmap을 예측한다.

초기 amodal bounding box는 modal bounding box와 동일하게 설정한다. 매 iteration마다 현재 amodal box 내부 patch를 네트워크에 넣고, 이보다 약간 더 넓은 영역에 대한 amodal heatmap을 얻는다. 이후 원래 bounding box의 상, 하, 좌, 우 바깥 영역에 대해 평균 heat intensity를 계산한다.

어떤 방향의 평균 intensity가 threshold 0.1보다 크면, 그 방향으로 bounding box를 확장한다. 그리고 확장된 새 box를 다음 iteration의 amodal box로 사용한다. 모든 방향의 평균 intensity가 threshold 이하가 될 때까지 반복한다.

최종 amodal segmentation mask는 amodal heatmap을 threshold 0.7로 이진화해서 얻는다. 비교를 위한 modal segmentation mask는 modal heatmap에 0.8 threshold를 적용한다.

이 알고리즘의 핵심은, bounding box를 한 번에 크게 예측하지 않고 **heatmap이 지시하는 방향으로 조금씩 키워 나간다**는 점이다. 이는 occluded part가 어느 방향으로 얼마나 이어질지 점진적으로 탐색하는 방식으로 볼 수 있다.

## 4. 실험 및 결과

### 4.1 실험 설정의 어려움

이 논문이 당시 어려웠던 이유는 평가용 ground truth 자체가 부족했다는 점이다. 공개된 amodal instance segmentation dataset이 없었기 때문에, 단순한 quantitative evaluation이 쉽지 않았다. 그래서 저자들은 세 종류의 평가를 수행한다.

첫째, 정성적 평가
둘째, PASCAL VOC의 coarse occlusion annotation을 이용한 간접 평가
셋째, 직접 수집한 100개 객체의 amodal mask annotation을 이용한 직접 평가

### 4.2 정성적 결과

PASCAL VOC 2012 val set에서 object category와 modal box를 ground truth로 주고 결과를 시각화했다. 논문은 occlusion을 두 종류로 구분한다.

하나는 **interior occlusion**으로, occluding object가 대체로 occluded object 내부에 들어와 있는 경우다. 이 경우에는 visible region 사이의 hole을 메우는 것이 핵심이다.
다른 하나는 **exterior occlusion**으로, occluding object가 외부까지 튀어나와 있는 경우다. 이 경우에는 visible part 바깥으로 object shape를 어디까지 확장할지 정해야 하므로 더 어렵고 모호하다.

논문에 따르면 제안 방법은 두 경우 모두 꽤 그럴듯한 amodal mask를 생성했다. 특히 exterior occlusion처럼 ambiguity가 큰 상황에서도 plausible한 shape hypothesis를 만든 사례가 제시된다. 반면 실패 사례에서는 unusual pose의 희소성, 인접 객체와의 appearance similarity, modal prediction 오류 등이 문제 원인으로 언급된다.

또한 unoccluded object에 대해서는 amodal prediction이 modal prediction과 거의 같아야 하는데, 실제로 그런 경향이 나타났다고 보고한다. 흥미롭게도 일부 경우에는 amodal model이 modal model보다 더 robust한 예측을 보였다. 저자들은 이를 “occlusion에 robust하도록 학습하면서 low-level variation에도 더 강해졌기 때문”이라고 해석한다. 이 해석은 논문이 제시한 설명이며, 추가적인 인과 검증까지 제시된 것은 아니다.

### 4.3 간접 평가: area ratio로 occlusion 존재 판별

저자들은 modal mask와 amodal mask를 모두 예측한 뒤, 다음 비율을 계산한다.

$$
\text{area ratio} =
\frac{\text{area}(\text{modal mask} \cap \text{amodal mask})}
{\text{area}(\text{amodal mask})}
$$

직관적으로 이 값은 “amodal mask 중 실제로 visible한 부분이 차지하는 비율”이다. unoccluded object라면 modal과 amodal이 거의 같으므로 이 값이 1에 가깝다. 반대로 많이 가려진 object라면 amodal mask 안에서 visible 영역 비중이 작아져 이 값이 작아진다.

PASCAL VOC 2012 val의 occlusion presence annotation과 비교한 결과, unoccluded object의 area ratio 분포는 높은 값 쪽으로 몰렸고, occluded object의 분포는 약 0.75 부근에서 peak를 보였다. 이 차이를 이용해 area ratio threshold classifier를 만들었고, 이때 **average precision 77.17%**를 얻었다.

이 결과는 제안한 amodal prediction이 적어도 “가려진 객체일수록 modal보다 바깥쪽 영역을 더 예측한다”는 점을 보여 준다. 즉, 단순 hallucination이 아니라 occlusion 존재와 어느 정도 정렬된 출력을 내고 있다는 간접 증거다.

### 4.4 직접 평가: 100개 수작업 amodal annotation

저자들은 PASCAL VOC 2012 val에서 category별로 occluded object 5개씩, 총 100개를 골라 직접 amodal mask를 annotation했다. 이 subset에서 제안 방법과 IIS를 비교한다. 여기서 IIS는 modal segmentation system이므로, 사실 amodal task에 대해 꽤 강한 baseline이다. 왜냐하면 occlusion이 심하지 않다면 가려진 부분을 아예 빼버려도 IoU가 크게 떨어지지 않을 수 있기 때문이다.

#### 4.4.1 Segmentation 성능

ground-truth modal bounding box와 category를 주고, segmentation system만 비교했을 때:

* 제안 방법은 **73% 객체에서 IIS보다 더 나은 mask**를 생성했다.
* 많은 사례에서 IoU가 **20%에서 50% 정도까지** 크게 향상되었다.
* 제안 방법이 IIS보다 못한 나머지 27% 사례에서도, 대부분 overlap 감소폭은 5% 미만이었다.

또한 IoU cutoff에 따른 accuracy curve 전 구간에서 제안 방법이 IIS보다 높았다. Table 1의 수치는 다음과 같다.

* IIS: Accuracy@50 = 68.0, Accuracy@70 = 37.0, AUC = 57.5
* Proposed Method: Accuracy@50 = 80.0, Accuracy@70 = 48.0, AUC = 64.3

즉, direct amodal segmentation quality 측면에서 제안 방법이 일관되게 우수했다.

#### 4.4.2 Detection + Segmentation 파이프라인

이제 detector로 Faster R-CNN을 사용하고, segmentation module만 IIS 또는 proposed method로 바꿔서 비교한다. 성능 지표는 region average precision, 즉 $mAP^r$이다. 다만 일부 instance에는 amodal ground truth가 없어서, prediction과 ground truth의 assignment에는 bounding box overlap을 쓰고 최종 correctness 판단에는 region IoU를 사용하도록 metric을 약간 수정했다.

Table 2 결과는 다음과 같다.

* Faster R-CNN + IIS: $mAP^r@50 = 34.1$, $mAP^r@70 = 14.0$
* Faster R-CNN + Proposed Method: $mAP^r@50 = 45.2$, $mAP^r@70 = 22.6$

즉, 제안 방법은 파이프라인 전체에서도 **50% IoU 기준 11.1 point**, **70% IoU 기준 8.6 point** 개선을 보였다.

### 4.5 Ablation Analysis

보충 자료에서 두 가지 변형을 비교한다.

첫 번째는 **modal segmentation prediction을 입력으로 쓰지 않는 모델**이다. 즉, modal heatmap 대신 직접 amodal heatmap을 예측하는 구조다.
두 번째는 **dynamic sample generation을 제거한 모델**이다. 즉, object마다 하나의 고정된 synthetic occlusion configuration만 사용한다.

Table 3 결과는 다음과 같다.

* Without Modal Segmentation Prediction: $mAP^r@50 = 35.2$, $mAP^r@70 = 18.4$
* Without Dynamic Sample Generation: $mAP^r@50 = 39.8$, $mAP^r@70 = 22.7$
* With Both: $mAP^r@50 = 45.2$, $mAP^r@70 = 22.6$

이 결과는 두 가지를 말해 준다. 첫째, **modal prediction을 조건 입력으로 주는 것**이 매우 중요하다. 둘째, **다양한 synthetic occlusion을 동적으로 생성하는 것** 역시 일반화 성능에 크게 기여한다.

### 4.6 PASCAL 3D+ 결과

rigid object에 대해 PASCAL 3D+의 CAD model projection을 approximate amodal mask처럼 사용한 추가 평가도 제시한다. 완전히 정확한 ground truth는 아니지만, 제안 방법이 여기서도 개선을 보인다.

* Faster R-CNN + IIS: $mAP^r@50 = 37.4$, $mAP^r@70 = 15.9$
* Faster R-CNN + Proposed Method: $mAP^r@50 = 44.0$, $mAP^r@70 = 20.9$

저자들도 인정하듯 이 annotation은 shape mismatch 때문에 근사치일 뿐이며, 엄밀한 amodal ground truth는 아니다. 따라서 이 실험은 보조적 근거로 해석하는 것이 적절하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정의 참신함과 학습 전략의 실용성**이다. amodal annotation이 없는 상황에서 synthetic occlusion generation으로 학습 문제를 푼 발상은 매우 깔끔하다. 단순한 data augmentation 수준이 아니라, “modal annotation을 amodal supervision으로 재해석”했다는 점에서 개념적 기여가 있다.

두 번째 강점은 **입력 조건의 설계**다. 단순히 RGB만 넣고 hidden part를 상상하게 하지 않고, modal segmentation heatmap을 조건으로 제공해 현재 보이는 객체의 위치와 형태를 명시적으로 알려 준다. Ablation 결과도 이 설계가 실제로 중요함을 뒷받침한다.

세 번째 강점은 **Iterative Bounding Box Expansion**이다. amodal box를 직접 회귀하는 대신, heatmap 신호에 따라 상하좌우로 확장하는 방식은 해석 가능하고 구현도 비교적 단순하다. 특히 외연 확장이 필요한 exterior occlusion에서 유용한 아이디어다.

네 번째 강점은 당시 기준으로 **직접 annotation을 수집해 정량 평가를 시도했다는 점**이다. 공개 데이터셋이 없다는 이유로 qualitative result에만 머무르지 않고, 100개 object라도 직접 라벨링하여 정량 비교를 제공했다는 점은 연구 완성도를 높인다.

반면 한계도 분명하다.

가장 근본적인 한계는 **문제 자체의 모호성**이다. 특히 exterior occlusion에서는 정답이 하나가 아닐 수 있다. 그런데 논문의 평가는 단일 ground truth mask와 IoU 기반으로 이루어진다. 이는 plausible hypothesis 여러 개 중 하나를 예측했더라도 낮은 점수를 받을 수 있음을 뜻한다. 논문도 이 모호성을 인지하고 있지만, evaluation metric 자체는 이를 충분히 반영하지 못한다.

또 다른 한계는 **synthetic occlusion과 real occlusion 간 domain gap**이다. 논문은 synthetic training despite real-world effectiveness를 주장하지만, overlay 방식의 occlusion은 실제 장면의 조명, 경계 혼합, 물체 간 맥락을 완전히 재현하지 못한다. 그럼에도 성능이 나왔다는 점은 고무적이지만, 동시에 일반화 범위가 어디까지인지 논문만으로는 충분히 알기 어렵다.

세 번째 한계는 **입력 가정의 강함**이다. segmentation-only 평가에서는 modal bounding box와 category를 ground truth로 사용한다. combined pipeline 평가도 하긴 했지만, 전체 시스템이 detector 품질에 상당히 의존한다는 점은 변하지 않는다. 즉, 완전한 end-to-end amodal perception system이라기보다는, 강한 modal detection/segmentation 시스템 위에 얹는 구조에 가깝다.

네 번째 한계는 데이터 규모다. 직접 평가용 annotation이 100개 object에 불과해 category별 분산이나 극단적 case 분석을 충분히 하기 어렵다. 논문의 주장 자체는 설득력이 있지만, 대규모 benchmark 없이 일반성을 강하게 단정하기는 어렵다.

마지막으로, 방법의 성격상 **shape prior**를 category level에서 학습하는 경향이 강하다. 논문도 unusual pose나 rare configuration에서 실패한다고 언급한다. 이는 모델이 “이 category는 보통 이런 모양”이라는 통계적 평균에 많이 의존할 가능성을 시사한다.

## 6. 결론

이 논문은 amodal instance segmentation을 본격적인 computer vision task로 제시하고, 그에 대한 **최초의 일반 목적 방법**을 제안한 초기 대표작으로 볼 수 있다. 가장 중요한 기여는 세 가지다.

첫째, 공개된 amodal annotation이 없는 상황에서도 기존 modal annotation만으로 학습 가능한 **synthetic amodal data generation strategy**를 제안했다.
둘째, modal heatmap을 조건 입력으로 사용하는 CNN 기반 **amodal segmentation predictor**를 설계했다.
셋째, amodal heatmap을 이용해 박스를 점진적으로 넓혀 가는 **Iterative Bounding Box Expansion**을 제안했다.

실험적으로도 제안 방법은 qualitative result뿐 아니라 직접 수집한 annotation 기반 평가에서 IIS보다 일관되게 우수한 성능을 보였다. 특히 segmentation-only와 detection+segmentation pipeline 모두에서 개선이 나타났다는 점은 실용성을 뒷받침한다.

이 연구의 의의는 단순히 하나의 segmentation variant를 만든 데 있지 않다. occlusion reasoning, depth ordering, object extent estimation 같은 더 넓은 시각 인지 문제를 향한 첫 단계로서 의미가 크다. 이후 amodal segmentation 연구들이 visible/invisible/amodal mask를 함께 예측하거나, shape prior와 context reasoning을 결합하거나, end-to-end detector와 통합하는 방향으로 발전했다는 점을 생각하면, 이 논문은 그 출발점에 해당하는 연구라고 평가할 수 있다.

전체적으로 보면, 이 논문은 데이터 부재라는 현실적인 제약 아래에서도 문제를 성립시키고 실험적으로 증명했다는 점에서 가치가 높다. 오늘날 기준으로는 더 큰 데이터셋, 더 강한 backbone, 더 정교한 end-to-end 모델이 가능하지만, “가려진 부분까지 포함한 object understanding”이라는 관점을 instance-level segmentation에 본격적으로 도입했다는 점에서 여전히 중요한 논문이다.
