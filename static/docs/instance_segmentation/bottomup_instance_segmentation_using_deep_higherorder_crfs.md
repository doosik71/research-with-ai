# Bottom-up Instance Segmentation using Deep Higher-Order CRFs

* **저자**: Anurag Arnab, Philip H.S. Torr
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1609.02583](https://arxiv.org/abs/1609.02583)

## 1. 논문 개요

이 논문은 **instance segmentation** 문제를 다룬다. 이는 각 픽셀의 클래스만 맞히는 semantic segmentation과, 물체를 박스로만 찾는 object detection의 중간이 아니라, 두 문제를 동시에 더 정밀하게 해결해야 하는 과제다. 즉, 이미지 안에서 “이 픽셀이 사람인가”만 아는 것으로는 부족하고, “이 픽셀은 첫 번째 사람인지 두 번째 사람인지”까지 구분해야 한다.

저자들은 당시 많이 쓰이던 방식, 즉 먼저 object proposal을 만들고, 각 proposal을 분류한 뒤, 그 박스 내부를 segmentation으로 다듬는 top-down 계열 접근의 한계를 지적한다. 이런 방식은 초기 proposal 품질에 크게 의존하고, proposal 생성 비용도 크며, 초기에 놓친 객체는 뒤 단계에서 복구하기 어렵다. 특히 proposal이 부정확하면 segmentation도 그 한계 안에서만 움직이게 된다.

이에 비해 이 논문은 **bottom-up 방식**을 택한다. 먼저 이미지 전체에 대해 category-level semantic segmentation을 수행하고, 그 뒤 object detector의 출력을 이용해 각 픽셀을 개별 instance에 배정한다. 핵심은 semantic segmentation 네트워크 내부에 **higher-order detection potential을 갖는 CRF**를 넣어, detector의 결과를 단순 후처리로 쓰는 것이 아니라 segmentation 추론 과정 자체에 통합했다는 점이다. 이 CRF는 end-to-end로 학습 가능하며, detector의 신뢰도도 추론 과정에서 재조정한다.

문제의 중요성은 분명하다. instance segmentation은 자율주행, 로봇 인지, 장면 이해 등에서 객체 단위의 정밀한 공간 이해를 요구하는 핵심 기술이다. 단순한 bounding box보다 픽셀 단위 결과가 필요하고, 같은 클래스 객체가 여러 개 있을 때 개체별 구분이 필요하기 때문이다. 이 논문은 semantic segmentation과 object detection의 발전을 결합해, 보다 간단한 구조로 강한 instance segmentation 성능을 얻을 수 있음을 보여준다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 다음과 같다. **semantic segmentation을 먼저 잘 수행하고, detector 정보를 CRF에 넣어 그 segmentation을 instance-aware하게 만든 다음, 최종적으로 각 픽셀을 특정 detection 인스턴스에 할당한다**는 것이다. 즉, 처음부터 “박스 하나당 마스크 하나”를 예측하는 대신, 먼저 전체 장면을 픽셀 단위로 해석하고, 이후 object detector를 이용해 같은 클래스 안의 서로 다른 객체를 분리한다.

기존 proposal 기반 방식과의 가장 큰 차별점은 **초기 proposal에 종속되지 않는다는 점**이다. 저자들은 기존 방법들이 object를 먼저 localize하고 나중에 segment하기 때문에, 초기에 잘못 검출되거나 proposal이 부정확하면 전체 파이프라인이 무너질 수 있다고 본다. 반대로 이 논문은 semantic segmentation이 먼저 전체 픽셀에 대한 클래스 분포를 제공하므로, detection은 이를 instance 분리의 힌트로 활용된다.

또 다른 핵심 차별점은 detector 출력을 단순히 외부 단서로 쓰는 것이 아니라, **higher-order CRF potential** 형태로 네트워크 내부 추론에 포함시킨다는 점이다. 이때 각 detection마다 latent binary variable $Y_d$를 두어, 해당 detection이 실제로 유효한지 확률적으로 판정한다. 이 변수는 detector confidence를 그대로 따르지 않고, segmentation unaries와 pairwise consistency와 얼마나 잘 맞는지에 따라 inference 중에 값이 바뀐다. 결과적으로 detector의 score가 **recalibration**된다. 즉, segmentation과 모순되는 detection은 점수가 내려가고, 잘 맞는 detection은 강화된다.

마지막으로, instance segmentation을 위한 최종 CRF가 **입력 이미지마다 label 수가 달라지는 dynamic CRF**라는 점도 중요하다. semantic segmentation의 label은 보통 데이터셋 전체에서 고정된 클래스 집합이지만, instance segmentation에서는 이미지마다 detection 수 $D$가 다르기 때문에 label 수가 $D+1$로 바뀐다. 저자들은 class-specific 파라미터가 아닌 공유 가능한 CRF 가중치를 사용해 이 문제를 해결한다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 두 단계다. 첫 번째는 **object segmentation subnetwork**, 두 번째는 **instance segmentation subnetwork**다.

첫 단계에서는 입력 이미지를 semantic segmentation 네트워크에 넣어 각 픽셀에 대해 $K+1$개 클래스의 확률을 얻는다. 여기서 $K$는 foreground class 수이고, 나머지 1은 background다. 이 semantic segmentation 네트워크는 단순 CNN만이 아니라, 그 뒤에 fully differentiable CRF가 붙어 있다. 이 CRF는 일반 unary와 pairwise term뿐 아니라 detector 기반 higher-order term도 포함한다.

둘째 단계에서는 첫 단계의 category-level segmentation 결과와 detector 출력을 결합하여, 각 픽셀이 어떤 instance에 속하는지를 추정한다. 이때 detection 개수가 이미지마다 다르므로 최종 label space는 $D+1$개다. 여기서 $D$는 detector가 내놓은 detection 수이고, 추가된 1은 어떤 detection에도 속하지 않는 background를 의미한다. 이후 이 확률들을 unary로 사용한 또 하나의 pairwise CRF를 돌려 최종 instance segmentation을 얻는다.

### 3.1 CRF 기본 구성

논문은 우선 일반적인 CRF 표기를 정리한다. 이미지 $I$가 있고 픽셀 수가 $N$일 때, 각 픽셀 $i$에 대해 random variable $X_i$를 둔다. 각 $X_i$는 label set $L$ 중 하나를 가진다. semantic segmentation에서는 이 label이 “person”, “car” 같은 클래스다.

CRF는 다음과 같이 정의된다.

$$
\Pr(X=x \mid I) = \frac{1}{Z(I)} \exp(-E(x \mid I))
$$

여기서 $E(x \mid I)$는 labeling $x$의 energy이고, $Z(I)$는 partition function이다. 보통 조건부 표기는 생략하고 단순히 $E(x)$로 쓴다. clique 집합을 $C$라고 할 때, 에너지는

$$
E(x)=\sum_{c\in C}\psi_c(x_c)
$$

로 쓸 수 있다.

semantic segmentation에서 자주 쓰이는 기본 형태는 unary와 pairwise만 갖는 구조다.

$$
E(x)=\sum_i \psi_i^U(x_i)+\sum_{i<j}\psi_{ij}^P(x_i,x_j)
$$

여기서 unary는 CNN이 주는 per-pixel class score에 해당하고, pairwise는 가까운 픽셀이나 비슷한 appearance를 가진 픽셀이 같은 라벨을 갖도록 유도한다.

### 3.2 Higher-Order Detection Potentials

이 논문의 핵심은 위 energy에 detector 기반 higher-order term을 추가하는 것이다. detector는 각 detection $d$에 대해 다음 정보를 제공한다고 가정한다.

* $l_d$: detection의 클래스 label
* $s_d$: detector confidence score
* $F_d$: detection foreground에 해당하는 픽셀 집합
* $B_d$: detection bounding box 내부의 픽셀 집합

여기서 $F_d$는 GrabCut 같은 foreground/background segmentation으로 얻은 대략적인 object foreground다. 즉, bounding box 전체가 아니라 그 안에서 객체로 보이는 부분만 따로 잡는다.

저자들의 의도는 간단하다. detection이 맞는 경우에는 $F_d$에 속한 픽셀들이 클래스 $l_d$를 갖도록 유도하고 싶다. 하지만 detector가 틀릴 수도 있고 foreground mask도 부정확할 수 있으니, 이 제약은 hard constraint가 아니라 soft constraint여야 한다. 이를 위해 각 detection에 latent binary variable $Y_d$를 도입한다. $Y_d=1$이면 detection이 유효한 것이고, $Y_d=0$이면 무효한 detection이다.

각 detection에 대해 $(X_d, Y_d)$ clique를 구성하고, energy는 다음과 같이 정의된다.

$$
\psi_d^{Det}(X_d=x_d, Y_d=y_d)=
\begin{cases}
w_l \frac{s_d}{|F_d|}\sum_{i=1}^{|F_d|}[x_d^{(i)}=l_d] & \text{if } y_d=0 \
w_l \frac{s_d}{|F_d|}\sum_{i=1}^{|F_d|}[x_d^{(i)}\neq l_d] & \text{if } y_d=1
\end{cases}
$$

본문의 표기에는 줄바꿈과 OCR 왜곡이 조금 있으나, 설명상 의미는 분명하다. 직관적으로 보면 다음과 같다.

* $Y_d=1$일 때는 detection이 맞다고 보는 것이므로, $F_d$ 안의 픽셀들이 $l_d$와 **일치하지 않으면** 비용이 커진다.
* $Y_d=0$일 때는 detection이 틀렸다고 보는 것이므로, 오히려 $F_d$ 안 픽셀들이 $l_d$를 많이 가지면 비용이 커진다.

즉, $X$와 $Y$가 서로 일관되도록 설계된 항이다. detector가 맞는 것 같으면 그 detection의 픽셀들은 해당 클래스가 되도록 압박하고, 반대로 segmentation 증거가 detector와 맞지 않으면 detection 자체를 invalid 쪽으로 밀어낸다.

여기에 더해 각 $Y_d$에는 unary potential도 따로 있다. 이 unary는 detector confidence $s_d$로 초기화된다. 하지만 mean-field inference를 거치면서 이 값은 segmentation evidence에 따라 조정된다. 저자들은 이 과정을 **detection score recalibration**으로 해석한다. 이 논문의 중요한 포인트다. detector score가 단순 입력이 아니라 CRF 추론을 거친 뒤 더 믿을 만한 점수로 변한다.

최종 semantic segmentation 단계의 전체 energy는 다음과 같다.

$$
E(x)=\sum_i \psi_i^U(x_i)
+\sum_{i<j}\psi_{ij}^P(x_i,x_j)
+\sum_d \psi_d^{Det}(x_d,y_d)
+\sum_d \psi_d^U(y_d)
$$

즉, CNN unary + dense pairwise + detection higher-order + detection validity unary의 조합이다.

### 3.3 Instance Identification

이 단계가 semantic segmentation을 실제 instance segmentation으로 바꾸는 부분이다. 첫 단계가 끝나면 각 픽셀은 클래스 확률은 갖고 있지만, 같은 클래스 안에서 어느 instance인지는 아직 모른다.

저자들은 각 object detection을 하나의 possible instance로 본다. 따라서 detection이 $D$개면 instance label도 $D$개다. 여기에 background용 label을 하나 더해서 총 $D+1$개 라벨을 만든다.

가장 단순한 방법은 “픽셀이 detection box 안에 있고, semantic class가 detector class와 일치하면 그 instance에 할당”하는 것이다. 하지만 box가 겹치면, 특히 occlusion 상황에서는 이 방식이 제대로 작동하지 않는다. 그래서 저자들은 픽셀 $i$가 instance $k$에 속할 확률을 다음처럼 정의한다.

$$
\Pr(v_i=k)=
\begin{cases}
\frac{1}{Z(Y,Q)} Q_i(l_k)\Pr(Y_k=1) & \text{if } i \in B_k \
0 & \text{otherwise}
\end{cases}
$$

여기서

* $v_i$는 픽셀 $i$의 instance label
* $Q_i(l_k)$는 첫 번째 semantic segmentation 단계에서 픽셀 $i$가 클래스 $l_k$일 확률
* $\Pr(Y_k=1)$는 detection $k$가 valid일 확률, 즉 recalibrated detection score
* $B_k$는 detection $k$의 bounding box
* $Z(Y,Q)$는 overlapping box들이 있을 때 정규화 상수

이 식은 매우 직관적이다. 어떤 픽셀이 특정 instance에 속할 가능성은, 그 픽셀이 해당 detection box 내부에 있어야 하고, semantic segmentation이 그 detection의 클래스라고 믿을수록 커지며, 동시에 그 detection 자체가 CRF 관점에서 신뢰할 만할수록 커진다.

background를 위해 저자들은 추가 detection $d_0$를 도입한다. 이는 어느 detection box와도 겹치지 않는 background pixels를 담당한다.

### 3.4 Final Instance CRF

위에서 얻은 $\Pr(v_i)$를 unary로 사용해 또 하나의 CRF를 만든다. energy는

$$
E(v)=\sum_i \psi_i^U(v_i)+\sum_{i<j}\psi_{ij}^P(v_i,v_j)
$$

이며 unary는

$$
\psi_i^U(v_i)=-\ln \Pr(v_i)
$$

이다.

즉, 첫 단계는 클래스 단위 semantic segmentation용 CRF이고, 두 번째는 detection-instance 단위 final refinement용 CRF다. 두 번째 CRF의 pairwise term 역시 fully connected CRF 스타일의 appearance/spatial consistency prior를 사용한다. 따라서 픽셀 간 색상이나 위치가 비슷하면 같은 instance label을 공유하도록 유도한다.

이 final Instance CRF는 중요한 특성이 하나 있다. **label 수가 이미지마다 달라진다**. 보통 semantic segmentation CRF는 클래스 수가 고정이지만, 여기서는 detection 수가 매 이미지마다 다르다. 저자들은 이에 대응하기 위해 class-specific 가중치 대신 label 의미와 무관하게 적용 가능한 weight를 사용한다. instance label “1”이 한 이미지에서는 dining table일 수 있고, 다른 이미지에서는 sheep일 수 있으므로, label identity 자체에 의미를 두는 파라미터는 적절하지 않다.

또한 이 동적 CRF 역시 mean-field inference를 unroll한 recurrent 형태로 구현하여 end-to-end differentiable하게 만든다. 따라서 전체 네트워크는 이미지마다 detection 수에 맞게 동적으로 구성될 수 있으며, 학습도 end-to-end로 가능하다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 지표

평가는 **PASCAL VOC 2012 validation set** 1449장 이미지에서 수행했다. instance segmentation에 대한 공식 test server가 없어서 validation set을 사용했다고 명시한다.

평가 지표는 Hariharan 등의 SDS 논문에서 사용한 **$AP^r$**이다. 이는 object detection의 AP와 비슷하지만, bounding box IoU가 아니라 **predicted mask와 ground truth mask 사이의 IoU**를 기준으로 계산한다. 저자들은 IoU threshold를 여러 값에서 평가한다. threshold가 높을수록 마스크 경계와 형태가 정확해야 하므로, 고 threshold 성능은 진짜 정밀한 segmentation 품질을 더 잘 반영한다.

또한 **$AP^r_{vol}$**도 보고한다. 이는 IoU threshold를 0.1부터 0.9까지 0.1 간격으로 바꿔가며 평균한 값이다. 전체적인 품질을 보는 지표라고 이해하면 된다.

### 4.2 학습 설정

semantic segmentation backbone은 FCN 계열이며, VGG-16을 ImageNet에서 pretrain한 뒤 VOC 2012 training set, Semantic Boundaries Dataset 일부, Microsoft COCO 데이터로 학습했다. 중요한 점은 공개 모델을 그대로 쓰지 않은 이유가 명확히 설명된다는 것이다. 기존 공개 모델들이 VOC validation 일부를 학습에 사용했기 때문에, 공정한 validation 평가를 위해 직접 재학습했다.

그 후 pretrained segmentation network 위에 Higher Order CRF를 붙여 VOC 2012의 finely annotated data만으로 finetuning했다. learning rate는 $10^{-11}$로 매우 작은데, 이는 loss가 이미지 내 pixel 수로 정규화되지 않았기 때문이라고 설명한다. detector는 Faster R-CNN 공개 프레임워크를 사용했다.

semantic segmentation 성능도 함께 보고되는데, pairwise CRF만 썼을 때 mean IoU가 73.4%이고, detection potential을 포함하면 75.3%로 상승한다. 즉, detector cue가 category-level semantic segmentation 품질 자체도 개선한다.

### 4.3 Detection Potentials의 효과 분석

논문은 detection potentials가 실제로 어떤 기여를 하는지 두 가지 ablation으로 분석한다.

첫 번째 baseline은 **semantic segmentation CRF에서 detection potentials를 제거**한 경우다. 이 경우 latent $Y$를 쓸 수 없으므로 detector의 원래 confidence score를 instance assignment에 사용한다. 성능은 다음과 같다.

* $AP^r@0.5 = 54.6$
* $AP^r@0.6 = 48.5$
* $AP^r@0.7 = 41.8$
* $AP^r_{vol} = 50.0$

최종 full system과 비교하면 각각 3.7%p, 3.9%p, 3.6%p, 3.1%p 낮다. 이는 detection potential이 semantic segmentation 향상과 detection score recalibration 두 측면에서 모두 도움을 준다는 뜻이다.

두 번째 baseline은 **detection potentials는 넣되, inference 후의 latent $Y$를 무시하고 original detector score를 그대로 사용하는 경우**다. 즉, segmentation 향상 효과는 누리지만, recalibrated score는 사용하지 않는 조건이다. 성능은 다음과 같다.

* $AP^r@0.5 = 57.5$
* $AP^r@0.6 = 51.6$
* $AP^r@0.7 = 44.5$
* $AP^r_{vol} = 52.4$

최종 full system은

* $AP^r@0.5 = 58.3$
* $AP^r@0.6 = 52.4$
* $AP^r@0.7 = 45.4$
* $AP^r_{vol} = 53.1$

이다.

여기서 second baseline 대비 full system의 차이는 크지는 않지만 일관되게 존재한다. 즉, **$Y$ 기반 detector score recalibration 자체도 유의미한 개선을 준다**. 그러나 더 큰 차이는 detection potentials의 존재 여부에서 나오므로, 저자들은 좋은 초기 semantic segmentation이 전체 시스템에서 매우 중요하다고 해석한다. 이 점은 이 방법이 truly bottom-up이라는 주장과 잘 연결된다.

### 4.4 다른 방법들과의 비교

비교 대상은 SDS [13], Chen et al. [4], PFN [24]이다.

성능은 다음과 같다.

* **SDS [13]**

  * $AP^r@0.5 = 43.8$
  * $AP^r@0.6 = 34.5$
  * $AP^r@0.7 = 21.3$
  * $AP^r@0.8 = 8.7$
  * $AP^r@0.9 = 0.9$

* **Chen et al. [4]**

  * $AP^r@0.5 = 46.3$
  * $AP^r@0.6 = 38.2$
  * $AP^r@0.7 = 27.0$
  * $AP^r@0.8 = 13.5$
  * $AP^r@0.9 = 2.6$

* **PFN [24]**

  * $AP^r@0.5 = 58.7$
  * $AP^r@0.6 = 51.3$
  * $AP^r@0.7 = 42.5$
  * $AP^r@0.8 = 31.2$
  * $AP^r@0.9 = 15.7$
  * $AP^r_{vol} = 52.3$

* **Ours**

  * $AP^r@0.5 = 58.3$
  * $AP^r@0.6 = 52.4$
  * $AP^r@0.7 = 45.4$
  * $AP^r@0.8 = 34.9$
  * $AP^r@0.9 = 20.1$
  * $AP^r_{vol} = 53.1$

이 결과는 흥미롭다. IoU 0.5에서는 PFN이 58.7로 이 논문보다 아주 조금 높다. 하지만 IoU threshold가 올라갈수록 이 논문이 더 강하다. 특히 $AP^r@0.9$에서 20.1 대 15.7로 **4.4%p** 차이가 난다. 저자들은 이를 근거로 자신들의 마스크가 더 **정밀하다**고 주장한다. 반면 $AP^r_{vol}$ 차이는 0.8%p로 크지 않다. 저자들의 해석에 따르면 PFN은 낮은 IoU threshold에서 더 많은 instance를 잡을 수 있을 가능성이 있지만, 높은 threshold에서는 segmentation precision이 떨어진다.

즉, 이 방법의 장점은 “대충 instance를 많이 찾는 것”보다 “정확한 경계와 정밀한 마스크를 가진 instance segmentation”에 더 가깝다. 이는 detector와 semantic segmentation을 함께 활용하고, 마지막에 pairwise Instance CRF로 refinement하는 구조와도 잘 맞아떨어진다.

### 4.5 정성적 분석

논문은 성공/실패 사례도 보여준다. 성공 사례에서는 occlusion이 있어도 잘 동작하고, false positive detection에도 크게 흔들리지 않는다. 이는 detection score recalibration 덕분이라고 이해할 수 있다. 예를 들어 bottle detection이 false positive여도 semantic segmentation evidence와 맞지 않으면 그 detection의 영향이 약해질 수 있다.

반면 실패 사례로는 두 가지가 주로 제시된다. 하나는 **서로 매우 비슷하게 생긴 객체들이 겹친 경우**이고, 다른 하나는 **심한 occlusion**이다. 이런 경우에는 같은 클래스 객체 사이를 분리하는 데 어려움을 겪는다. 이는 이 방법이 detection box와 semantic confidence를 결합해 instance를 정하는데, appearance가 매우 비슷하고 겹침이 크면 픽셀 단위로 인스턴스를 나누기 어려워지기 때문이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 구조가 비교적 단순하면서도 설득력 있다는 점이다. semantic segmentation과 object detection이라는 이미 강한 두 모듈을 억지로 하나의 복잡한 예측 문제로 합치기보다, **semantic segmentation을 먼저 수행한 뒤 detector 정보를 CRF 안에서 활용해 instance를 분리하는 계층적 접근**을 취했다. 이 때문에 proposal 기반 methods의 복잡한 box refinement, instance number prediction, clustering 단계를 피할 수 있다.

두 번째 강점은 **higher-order detection potential의 역할이 단순한 보조 정보 수준이 아니라는 점**이다. detector output을 CRF inference에 통합하고, latent variable $Y_d$를 통해 detection validity를 추론하게 한 것은 elegant하다. 특히 detector score recalibration은 논문 전체의 핵심 기여 중 하나다. detector와 segmentation이 서로 일관성을 검사하며 보정하는 구조이기 때문이다.

세 번째 강점은 **높은 IoU threshold에서의 강한 성능**이다. instance segmentation에서는 단순히 객체를 찾는 것보다 얼마나 정확한 mask를 만드느냐가 중요하다. 이 논문은 $AP^r@0.8$, $AP^r@0.9$ 같은 높은 threshold에서 특히 강한 개선을 보여, 마스크 precision 측면에서 장점이 있음을 실험적으로 보였다.

네 번째 강점은 **dynamic CRF를 differentiable하게 구현했다는 점**이다. instance 수가 이미지마다 달라지는 문제를 class-specific parameter 없이 해결해, end-to-end 학습 가능한 구조로 정리한 것은 방법론적으로도 의미가 있다.

하지만 한계도 분명하다. 첫째, 이 방법은 detector에 어느 정도 의존한다. proposal-free라고 완전히 부를 수는 있지만, 실제로는 각 detection을 instance hypothesis로 사용하므로 detector가 객체를 놓치면 그 객체 instance를 복원하기 어렵다. 저자들도 detector 결과를 기반으로 instance를 정의하고 있다.

둘째, foreground set $F_d$를 얻기 위해 GrabCut 기반 foreground segmentation을 사용한다. 이는 당시로서는 현실적 선택이지만, 완전히 learned end-to-end visual representation만으로 끝나는 구조는 아니다. detection 내부 foreground 추출 품질이 낮으면 higher-order potential의 효과가 제한될 수 있다.

셋째, heavily occluded and visually similar instances에서 성능이 떨어진다. 이는 저자들이 직접 failure case로 제시한 부분이다. semantic class confidence와 detection box만으로는 같은 클래스 객체들 사이의 미세한 경계 분리를 충분히 하지 못할 수 있다.

넷째, 학습 및 실험 설정에서 추론 비용이나 속도 분석은 자세히 제시되지 않는다. method가 fully differentiable이라는 점은 강조되지만, 실제 계산량이나 메모리 복잡도가 얼마나 되는지는 이 발췌문만으로는 명확하지 않다. 따라서 실용 배치 관점에서의 trade-off는 논문 본문만으로 충분히 평가하기 어렵다.

비판적으로 보면, 이 논문은 매우 좋은 engineering integration을 보여주지만, instance segmentation을 완전히 새로운 representation으로 푸는 접근이라기보다는 **semantic segmentation + object detection + CRF reasoning의 정교한 결합**에 가깝다. 그러나 당시 맥락에서는 이것이 오히려 장점이다. 새롭고 복잡한 prediction head를 설계하기보다, 이미 검증된 컴포넌트를 적절히 결합해 고성능을 낸다는 점에서 실용적 가치가 높다.

## 6. 결론

이 논문은 instance segmentation을 위해, 먼저 category-level semantic segmentation을 수행하고, object detector 출력을 higher-order CRF potential로 통합한 뒤, 각 픽셀을 detection instance에 배정하는 **bottom-up instance segmentation framework**를 제안했다. 핵심 기여는 detector 정보를 segmentation 추론에 직접 통합했다는 점, latent variable을 통해 detection score를 recalibration했다는 점, 그리고 detection 수가 이미지마다 달라지는 dynamic Instance CRF를 fully differentiable하게 구현했다는 점이다.

실험적으로는 PASCAL VOC 2012 validation set에서 당시 강한 baseline 및 기존 방법들을 능가했고, 특히 높은 IoU threshold에서 뚜렷한 개선을 보였다. 이는 이 방법이 단지 객체를 대충 분리하는 것이 아니라, 보다 정밀한 mask를 생성한다는 증거다.

향후 연구 관점에서 보면, 이 논문은 instance segmentation이 semantic segmentation과 object detection의 자연스러운 결합 문제라는 시각을 잘 보여준다. 이후 등장한 많은 instance segmentation 방법들이 detection-driven mask prediction이나 set prediction 계열로 발전했지만, 이 논문은 **structured prediction과 deep learning을 결합해 instance reasoning을 수행하는 고전적이면서도 중요한 관점**을 제공한다. 실제 적용 측면에서도, detector와 segmentation 모델이 이미 준비되어 있는 환경에서 상대적으로 적은 구조 변경으로 instance segmentation을 구현하려는 시도에 유용한 아이디어를 준다.
