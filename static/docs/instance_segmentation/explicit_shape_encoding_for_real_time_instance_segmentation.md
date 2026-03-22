# Explicit Shape Encoding for Real-Time Instance Segmentation

이 논문은 instance segmentation이 object detection보다 훨씬 느린 근본 원인을 “객체마다 mask를 복원하기 위해 별도의 무거운 decoder/upsampling 과정을 반복해야 한다”는 점으로 보고, 이를 **짧은 shape vector 회귀 + 매우 가벼운 수치적 복원**으로 바꾸는 **ESE-Seg**를 제안합니다. 핵심은 객체의 모양을 implicit latent code가 아니라 **explicit shape encoding**으로 표현하고, 이를 bounding box처럼 detector가 직접 회귀하도록 만드는 것입니다. 저자들은 이를 위해 **Inner-center Radius (IR)**라는 새로운 contour 기반 shape signature와 **Chebyshev polynomial fitting**을 도입하고, YOLOv3·Faster R-CNN·RetinaNet 같은 기존 detector 위에 결합합니다. 그 결과 Pascal VOC 2012에서는 Mask R-CNN보다 더 높은 $mAP^r@0.5$를 보이면서도 약 7배 빠르다고 주장합니다.  

## 1. Paper Overview

이 논문이 다루는 문제는 **실시간에 가까운 instance segmentation**입니다. 일반적인 object detection은 각 객체를 4차원 box parameter로 회귀하면 되지만, instance segmentation은 box뿐 아니라 **픽셀 단위 모양(mask)**까지 복원해야 합니다. 기존 top-down 방식들, 예를 들어 Mask R-CNN류는 RoI별로 mask branch를 통과시키는 구조라서 객체 수가 많아질수록 연산량이 빠르게 늘고, 결과적으로 detection 대비 속도가 크게 느려집니다. 논문은 이 병목을 구조적으로 제거하려고 합니다.

저자들의 관점은 단순합니다. “box를 짧은 벡터로 표현해 빠르게 복원할 수 있듯이, **shape도 짧은 벡터**로 표현할 수 있다면 instance segmentation도 detection에 가까운 속도로 실행될 수 있지 않을까?” 이 아이디어를 실현하기 위해, 물체 외곽선을 짧고 robust하며 복원 가능한 벡터로 바꾸는 명시적 표현을 설계합니다. 그리고 이 벡터를 detector가 bounding box와 함께 직접 예측하게 합니다. 즉, 이 논문은 segmentation을 “mask 생성” 문제라기보다 “**shape parameter regression**” 문제로 재정의한 셈입니다.

이 문제가 중요한 이유는, 자율주행, 로봇 조작 같은 응용에서 instance segmentation은 정확도뿐 아니라 **지연 시간(latency)**이 매우 중요하기 때문입니다. 논문은 기존의 모델 경량화 기법에 기대기보다, 더 근본적으로 **shape prediction mechanism 자체를 바꾸는 방식**으로 속도 문제를 해결하려고 합니다.

## 2. Core Idea

이 논문의 핵심 아이디어는 세 층으로 정리할 수 있습니다.

첫째, **명시적 shape encoding(explicit shape encoding)**입니다. 기존의 implicit 방식은 autoencoder 같은 네트워크가 latent vector를 mask로 복원해야 하므로, object마다 decoder forward가 필요합니다. 반면 ESE-Seg은 contour 기반 수학적 표현을 쓰므로 decoder network 자체가 필요 없습니다. shape 복원이 행렬 곱과 덧셈 같은 단순 tensor operation으로 이루어지기 때문에 병렬화가 쉽고 빠릅니다.

둘째, **Inner-center Radius (IR)**입니다. 일반적인 centroid나 bounding-box center는 물체 내부에 항상 존재하지 않을 수 있습니다. 그래서 저자들은 contour로부터 가장 멀리 떨어진 내부 점을 **inner center**로 정의하고, 이 점을 원점으로 삼아 contour를 polar coordinate의 radius function $f(\theta)$ 로 표현합니다. 이것이 IR signature입니다. 이렇게 하면 translation-invariant하고, 정규화 후 scale-invariant한 shape signature를 얻을 수 있습니다.

셋째, **Chebyshev polynomial fitting** 입니다. 원래 IR 자체는 360개 샘플 같은 긴 벡터가 될 수 있는데, 그대로는 network가 회귀하기에 길고 노이즈에도 민감합니다. 그래서 저자들은 $f(\theta)$ 를 Chebyshev polynomial basis로 근사해, 적은 수의 계수만 남깁니다. 이 계수들이 최종적으로 detector가 예측해야 하는 **shape descriptor**가 됩니다. 즉, ESE-Seg은 “contour → IR function → polynomial coefficients”라는 압축을 통해 shape를 짧고 학습하기 쉬운 벡터로 바꿉니다.

이 아이디어의 novelty는, segmentation을 위한 별도 mask decoder 없이도 shape reconstruction이 가능하도록 **표현 자체를 바꿨다**는 점입니다. 논문은 이것이 top-down instance segmentation에서 객체 수에 따라 느려지는 문제를 완화하고, 사실상 detection 속도에 근접하게 만든다고 주장합니다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

전체 파이프라인은 크게 네 단계입니다.

1. 입력 이미지에서 기본 object detector가 bounding box와 class를 예측합니다.
2. detector가 각 객체에 대해 shape descriptor, 즉 **Chebyshev coefficient vector**를 추가로 회귀합니다.
3. coefficient vector를 이용해 angle에 따른 radius function을 복원합니다.
4. inner center를 기준으로 radius를 다시 contour point들로 바꾸고, 최종 instance mask를 재구성합니다.

논문이 강조하는 점은, 마지막 복원 단계가 neural decoder가 아니라 **simple tensor operations**라는 것입니다. 그래서 여러 instance의 shape를 한 번에 복원할 수 있고, 이 때문에 속도가 크게 빨라집니다.

### 3.2 IR(Inner-center Radius) shape signature

IR은 두 단계로 구성됩니다.

먼저 객체 mask 내부에서 **inner center**를 찾습니다. 이는 contour까지의 distance transform을 계산했을 때 가장 멀리 있는 점입니다. 이 정의를 쓰는 이유는 center of mass나 bounding-box center는 object 바깥에 놓일 수도 있기 때문입니다. 특히 비대칭 또는 오목한 객체에서 inner center가 더 안정적입니다.

그 다음, 이 inner center를 기준으로 여러 angle $\theta$ 방향으로 contour를 샘플링합니다. 논문은 angle interval을 $\tau$ 라고 두고, 샘플 수를

$$
N = \left[\frac{2\pi}{\tau}\right]
$$

로 정의합니다. 실제 실험에서는 $\tau=\pi/180$ 을 사용해 $N=360$ 개의 contour sample을 얻습니다. 각 angle에서 radius 값을 모은 것이 $f(\theta)$ 입니다. 만약 한 ray가 contour와 여러 번 교차하면, 가장 큰 radius를 취합니다. 따라서 결과적으로 shape는 angle-indexed 1D function이 됩니다.

논문은 이 표현이 완벽하지는 않다고 인정합니다. 복잡한 non-convex shape나 가려진 객체에서는 contour sampling이 이상적으로 작동하지 않을 수 있습니다. 하지만 Pascal VOC와 COCO 실험에서는 자연 영상 객체에 대해 충분히 적합했다고 보고합니다.

### 3.3 끊어진 영역과 inner center 문제

실제 객체는 occlusion 등으로 인해 mask가 disconnected region으로 나타날 수 있습니다. 이 경우 inner center가 여러 개 생길 수 있습니다. 이를 처리하기 위해 논문은 broken area를 dilation으로 하나의 영역처럼 연결한 뒤 rough contour를 만들고, 이를 기준으로 원래 outline point를 재정렬하는 절차를 사용합니다. 이 과정은 매우 정교한 복원이라기보다, contour 순서를 일관되게 정하는 보조 절차에 가깝습니다.

이 설계는 논문이 shape encoding을 단순한 이상적 contour가 아니라, 실제 segmentation noise와 occlusion을 고려한 표현으로 만들려 했다는 점을 보여줍니다. 다만 이 단계가 완전한 이론적 해법이라기보다 휴리스틱 성격이 있다는 점은 같이 봐야 합니다.

### 3.4 Chebyshev polynomial fitting

IR signature만으로는 여전히 벡터가 깁니다. 논문은 이를 줄이고 노이즈에 강하게 만들기 위해 Chebyshev polynomial of the first kind를 사용합니다. 기본 recurrence는 다음과 같습니다.

$$
T_0(x)=1,\qquad T_1(x)=x,
$$

$$
T_{n+1}(x)=2xT_n(x)-T_{n-1}(x)
$$

그리고 radius function을

$$
f(\theta)\sim \sum_{i=0}^{\infty} c_i T_i(\theta)
$$

로 표현하고, 실제로는 앞의 몇 개 항만 남겨

$$
\tilde{f}(\theta)=\sum_{i=0}^{n} c_i T_i(\theta)
$$

로 근사합니다. 여기서 계수 $c_i$ 들이 detector가 예측해야 하는 최종 shape vector입니다.

왜 하필 Chebyshev인지에 대해 논문은 세 가지를 강조합니다.
첫째, reconstruction error가 작습니다.
둘째, noise sensitivity가 낮습니다.
셋째, coefficient의 수치 분포가 network regression에 더 적합합니다.
즉, 단순히 수학적으로 예쁘기 때문이 아니라, **학습하기 쉬운 shape coefficient distribution**을 만들어 준다는 게 중요합니다.

### 3.5 Detector와의 결합

ESE-Seg은 특정 detector에 종속되지 않습니다. Faster R-CNN, YOLO, YOLOv3, RetinaNet과 모두 결합 가능하다고 설명합니다. 다만 point-based detector와는 호환되지 않는다고 밝힙니다. 이유는 ESE-Seg이 bounding-box parameterization 위에서 shape를 회귀하는 구조인데, point-based detector는 그 전제를 직접 사용하지 않기 때문입니다.

이 말은 곧 ESE-Seg이 **detector head의 확장판**처럼 해석될 수 있다는 뜻입니다. 기존 box regression head 옆에 shape coefficient regression head를 더하고, 후처리에서 contour를 복원한다고 보면 됩니다. 이 구조 덕분에 시스템 전체가 비교적 깔끔합니다.

### 3.6 Explicit vs. Implicit shape representation

논문은 Jetley 등의 implicit shape representation과 비교해 차별점을 설명합니다.

* explicit는 contour 기반, implicit는 mask 기반인 경우가 많습니다.
* explicit는 추가 decoder network가 필요 없으므로 object별 반복 forward가 없습니다.
* implicit는 autoencoder를 따로 학습한 뒤 detector의 latent vector와 연결해야 하므로, 두 학습 단계 간 domain mismatch 문제가 생길 수 있습니다.
* explicit는 이 mismatch가 없습니다.

이 비교는 ESE-Seg의 철학을 잘 드러냅니다. 이 논문은 더 강력한 mask decoder를 설계하는 방향이 아니라, **아예 decoder가 없어도 되는 표현을 찾는 방향**을 택했습니다. 이것이 속도 이득의 핵심입니다.

## 4. Experiments and Findings

### 4.1 데이터셋과 평가

논문은 Pascal VOC 2012와 COCO 2017을 주요 benchmark로 사용합니다. 또 다양한 base detector를 얹어 generality를 보이려 합니다. 특히 Pascal VOC에서는 Mask R-CNN과 직접 비교하고, COCO에서는 경쟁력 있는 수준인지 살핍니다.

### 4.2 주요 성능 결과

논문 abstract와 introduction 기준으로, ESE-Seg with YOLOv3는 Pascal VOC 2012에서 **Mask R-CNN보다 높은 $mAP^r@0.5$** 를 기록하면서도 **약 7배 빠른 속도**를 보입니다. 본문에서는 Pascal VOC에서 **69.3 $mAP^r$**, COCO에서 **48.7 $mAP$** 를 보고합니다. 또한 base detector를 YOLOv3-tiny로 바꾸면 GTX 1080Ti에서 **약 130 fps** 까지 올라가며, 그때도 Pascal VOC에서 **53.2% $mAP^r@0.5$** 를 유지한다고 설명합니다.

이 결과는 매우 중요합니다. 많은 instance segmentation 연구가 “좀 더 정확한 mask”에 집중하는 반면, 이 논문은 속도-정확도 tradeoff의 다른 지점을 강하게 밀어붙입니다. 즉, 최상위 COCO AP를 노리기보다 **실시간성과 충분한 정확도**의 균형을 노립니다.

### 4.3 Explicit descriptor에 대한 분석

논문은 단순히 결과만 보여주지 않고, 왜 IR + Chebyshev 조합이 좋은지도 분석합니다. 먼저 contour vertex를 직접 $(x,y)$ sequence로 표현하는 straightforward design과 IR을 비교합니다. 흥미롭게도 contour vertex는 충분히 많은 점을 쓰면 reconstruction 자체는 잘 되지만, 실제 detector가 학습하기에는 더 어렵고 성능이 약 **10 mAP 정도 떨어진다**고 보고합니다. 저자들은 그 이유를, angle-based IR은 1D sequence인데 반해 vertex 좌표는 2D sequence라 noise에 더 민감하기 때문이라고 해석합니다.

또한 function approximation 방법으로는 polynomial regression, Fourier series 등과 비교한 뒤, **Chebyshev polynomial fitting이 reconstruction error, noise sensitivity, coefficient distribution 면에서 가장 낫다**고 결론짓습니다. 이 부분은 단순한 engineering tweak가 아니라, 논문의 핵심 주장인 “학습 가능한 explicit shape descriptor”를 뒷받침하는 중요한 실험입니다.

### 4.4 실험이 실제로 보여주는 것

이 실험들이 보여주는 바는 세 가지입니다.

첫째, ESE-Seg은 실제로 **mask branch를 가볍게 대체하는 메커니즘**으로 작동합니다. 속도 향상은 pruning이나 quantization 덕분이 아니라, shape decoding 구조 자체가 단순해서 얻어진 것입니다.

둘째, explicit shape encoding은 단지 이론적 아이디어가 아니라, Pascal VOC와 COCO 수준의 복잡한 natural image에서도 충분히 실용적입니다.

셋째, 정확도를 유지하려면 단순한 contour point 나열이 아니라, **내부 기준점 + angle-based radius + 적절한 basis fitting** 이 함께 필요합니다. 즉, 논문의 공헌은 “explicit”이라는 큰 방향뿐 아니라, 그 안의 세부 설계까지 포함합니다.  

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 문제 정의가 매우 선명하다는 점입니다. 이 논문은 “왜 instance segmentation이 느린가?”에 대해 정확히 답하고, 그 병목을 우회하는 명확한 설계를 제시합니다. 많은 논문이 더 큰 backbone이나 더 복잡한 mask head를 붙이는 반면, ESE-Seg은 **representation redesign**으로 접근합니다. 이 때문에 논문의 아이디어가 기억에 남고, 시스템 구조도 비교적 단순합니다.

두 번째 강점은 base detector 독립성입니다. Faster R-CNN, YOLO, YOLOv3, RetinaNet 등 여러 detection framework와 결합 가능하다고 보여 주기 때문에, 방법이 특정 one-stage 구조에만 종속된 것은 아닙니다.

세 번째 강점은 분석의 깊이입니다. 단순히 “IR이 잘 된다”고 말하는 게 아니라, 다른 shape signature와 fitting 방법을 비교해 왜 그 조합을 선택했는지 보여 줍니다. 이 때문에 논문이 ad hoc한 engineering memo처럼 보이지 않고, 설계 선택의 근거가 분명합니다.  

### 한계

가장 본질적인 한계는, 이 방법이 **모든 shape를 짧은 explicit vector로 잘 표현할 수 있다**는 가정에 의존한다는 점입니다. 매우 복잡한 topological structure, 다중 hole, 심한 self-occlusion, 복잡한 non-convex object에서는 angle-based radius representation이 근본적으로 불리할 수 있습니다. 논문도 contour sampling이 완벽하지 않다는 점은 인정합니다.

또한 이 방식은 mask를 직접 생성하는 게 아니라 contour parameter를 복원하는 것이므로, 고정밀 boundary quality가 중요한 경우에는 dense mask decoder보다 불리할 여지가 있습니다. Pascal VOC에서는 매우 잘 맞지만, COCO처럼 더 다양한 shape와 cluttered scene에서는 “competitive”하다고 표현할 정도로, absolute SOTA를 압도하는 톤은 아닙니다.

추가로, disconnected region을 dilation으로 이어 붙이는 처리나 inner-center 선택은 상당히 휴리스틱합니다. 실제 occlusion이 많은 장면에서 이 절차가 언제나 최적일지는 논문만으로는 충분히 검증되었다고 보긴 어렵습니다. 이 부분은 명시적으로 future work로 길게 논의되진 않지만, 비판적으로 보면 중요한 약점입니다.

### 해석

비평적으로 보면, 이 논문은 “instance segmentation을 detection-like problem으로 최대한 환원할 수 있는가?”라는 질문에 대한 강력한 시도입니다. 오늘날의 관점에서 보면 polygon-based mask prediction, contour-based segmentation, parametric shape regression 계열 연구들과 닿아 있습니다. 반면 transformer 기반 dense mask prediction처럼 훨씬 자유로운 표현이 유행한 흐름과는 다른 철학입니다. 즉, 이 논문은 최종 정확도보다는 **structured output의 효율적 parameterization**이라는 측면에서 의미가 큽니다.

## 6. Conclusion

이 논문은 instance segmentation의 속도 병목을 **mask decoder의 반복 실행**에서 찾고, 이를 **explicit shape encoding + detector 기반 coefficient regression + tensor-operation decoding**으로 대체한 ESE-Seg을 제안합니다. 핵심 기술은 내부 기준점을 이용한 **IR(Inner-center Radius)** shape signature와, 이를 짧고 robust한 벡터로 압축하는 **Chebyshev polynomial fitting**입니다. 그 결과 Pascal VOC에서는 Mask R-CNN보다 더 높은 성능과 훨씬 빠른 속도를 동시에 보여 주고, COCO에서도 경쟁력 있는 수준을 달성합니다.

실무적으로 이 논문은 정확도 절대치보다 **real-time instance segmentation**이 중요할 때 특히 의미가 큽니다. 연구적으로는 “mask를 직접 예측하는 것이 당연한가?”라는 가정을 흔들고, shape representation 설계가 segmentation 효율에 결정적일 수 있음을 보여 준 논문으로 볼 수 있습니다. 후속 연구를 볼 때도, 이 논문은 explicit contour/polygon 계열 접근의 중요한 기준점입니다.
