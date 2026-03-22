# SPRNet: Single Pixel Reconstruction for One-stage Instance Segmentation

* **저자**: Jun Yu, Jinghan Yao, Jian Zhang, Zhou Yu, Dacheng Tao
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1904.07426](https://arxiv.org/abs/1904.07426)

## 1. 논문 개요

이 논문은 **instance segmentation**을 기존의 two-stage 방식이 아니라 **one-stage 방식으로 수행하는 새로운 프레임워크 SPRNet**을 제안한다. instance segmentation은 단순히 객체의 bounding box를 찾는 것을 넘어서, 각 객체가 차지하는 **픽셀 단위 마스크**까지 예측해야 하는 문제다. 따라서 object detection보다 더 정밀한 시각 이해가 필요하고, semantic segmentation보다도 더 어려운 과제다. 왜냐하면 클래스 구분뿐 아니라 **같은 클래스 내부의 서로 다른 개체들까지 분리**해야 하기 때문이다.

기존의 대표적인 instance segmentation 방법들은 대부분 Faster R-CNN, Mask R-CNN처럼 **RPN과 RoI 기반의 two-stage 구조**를 사용한다. 이런 방법은 정확도는 높지만, proposal 생성과 RoI별 처리 과정 때문에 속도가 느리다. 논문은 이 점을 실용적 한계로 본다. 자율주행, 로보틱스, 비디오 감시처럼 빠른 처리 속도가 필요한 환경에서는 느린 two-stage 구조가 제약이 된다.

이 논문의 핵심 문제의식은 다음과 같다. one-stage detector는 이미 object detection에서는 충분히 강력해졌는데, 왜 instance segmentation은 여전히 two-stage에 의존하는가? 그리고 **RoI 없이 convolutional feature map만으로 각 인스턴스의 픽셀 마스크를 생성할 수 있는가?** 논문은 이 질문에 대한 답으로 SPRNet을 제안한다.

SPRNet의 목표는 두 가지를 동시에 달성하는 것이다. 첫째, **Mask R-CNN 수준에 가까운 segmentation 성능**을 내는 것. 둘째, **더 빠른 추론 속도**를 확보하는 것이다. 저자들은 같은 ResNet-50 backbone을 사용했을 때 SPRNet이 Mask R-CNN에 비해 mask AP는 다소 낮지만, 더 빠른 속도를 보이며, box AP에서는 RetinaNet보다 전반적으로 개선된다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 직관적이다. 기존 two-stage 방법은 **RoI 하나를 입력 단위**로 하여 각 인스턴스의 마스크를 예측한다. 반면 SPRNet은 **feature map의 “단일 픽셀(single pixel)” 하나를 인스턴스의 carrier로 사용**하고, 그 픽셀로부터 deconvolution을 통해 전체 인스턴스 마스크를 복원한다. 즉, “한 픽셀에서 한 객체의 마스크를 재구성한다”는 발상이 논문의 핵심이다.

이 아이디어는 기존 접근과 뚜렷하게 다르다. Mask R-CNN에서는 detector가 먼저 object box를 찾고, 그 다음 RoI Align으로 해당 영역 feature를 정렬한 뒤 mask branch가 binary segmentation을 수행한다. 하지만 SPRNet은 RoI Align, RoI Pooling, region proposal을 전혀 쓰지 않는다. 대신 **각 위치(pixel)가 객체 중심을 대표할 수 있도록 충분한 semantic 및 spatial 정보를 feature에 압축하고**, 그 위치에서 곧바로 $32 \times 32$ mask를 생성한다.

이를 가능하게 하려면 두 가지가 중요하다. 하나는 **하나의 pixel에 객체 전체를 표현할 만큼 충분한 정보가 담겨 있어야 한다**는 점이고, 다른 하나는 **그 pixel로부터 실제 mask를 복원하는 decoder가 있어야 한다**는 점이다. 논문은 이 두 문제에 각각 대응한다.

첫 번째 문제를 위해 저자들은 RetinaNet 계열의 backbone 위에 **Gate-FPN (GFPN)**을 제안한다. 이는 기존 FPN의 단순 element-wise summation 대신, feature fusion 전에 각 입력 feature map에 gate를 적용해 어떤 레벨의 정보가 얼마나 기여할지 조절한다. 저자들의 해석에 따르면, 단순 합은 higher-level feature의 부정확한 spatial information이 lower-level의 정밀한 spatial 정보를 훼손할 수 있고, 역전파 시에도 서로 다른 pyramid level 간 gradient 간섭을 유발한다. GFPN은 이를 완화하려는 시도다.

두 번째 문제를 위해 논문은 **Single Pixel Reconstruction** 구조를 제안한다. positive pixel 하나를 뽑고, 그 픽셀을 여러 단계의 deconvolution에 통과시켜 점차적으로 $1 \times 1$에서 $32 \times 32$ 크기의 instance mask로 복원한다. 다시 말해 encoder-decoder 구조를 갖되, decoder의 입력이 전통적인 feature patch가 아니라 **단일 pixel representation**이라는 점이 차별점이다.

따라서 이 논문의 설계 직관은 다음 한 문장으로 요약할 수 있다. **“좋은 single-pixel representation만 만들 수 있다면, proposal 없이도 one-stage instance segmentation이 가능하다.”**

## 3. 상세 방법 설명

### 전체 구조

SPRNet의 전체 구조는 RetinaNet을 기반으로 한다. backbone은 ResNet-50 또는 ResNet-101을 사용할 수 있고, 여기에 FPN 대신 GFPN을 붙인다. 이후 각 pyramid level feature는 세 개의 병렬 branch로 전달된다.

첫째는 **classification branch**로, 각 anchor의 클래스 점수를 예측한다.
둘째는 **regression branch**로, bounding box를 회귀한다.
셋째는 논문의 핵심인 **mask branch**로, 특정 pixel에서 인스턴스 mask를 생성한다.

classification과 regression은 RetinaNet과 거의 동일하고, mask branch만 새롭게 설계되었다고 보면 된다. 전체 loss는 classification, regression, mask loss의 합으로 구성된다. 논문에 따르면 각 항은 다음과 같다.

* classification: **Focal Loss**
* regression: **smoothed $L_1$ loss**
* mask: **cross-entropy loss** 또는 binary cross-entropy 기반 on-class mask loss

### GFPN: gated feature pyramid

논문은 기존 FPN의 문제를 두 가지로 본다.

첫째는 **spatial shift**다. 상위 레벨 feature는 semantic information은 풍부하지만 spatial detail은 거칠다. 이를 upsampling해서 하위 레벨 feature와 단순 합하면, 하위 레벨의 정밀한 위치 정보가 손상될 수 있다고 본다.

둘째는 **unexpected gradient propagation**이다. FPN의 summation은 역전파 시 서로 다른 level 간 gradient가 쉽게 섞이게 만든다. 저자들은 각 pyramid level이 본래 특정 scale의 객체를 담당해야 한다고 보고, 한 레벨의 손실이 다른 레벨 학습을 과도하게 지배하는 상황을 바람직하지 않다고 해석한다.

이를 해결하기 위해 GFPN은 두 feature map $x_1$, $x_2$ 각각에 shared separable convolution과 sigmoid를 적용해 gate map을 만들고, 이를 원래 feature에 곱한 뒤 더한다. 논문에 제시된 식은 다음과 같다.

$$
f_1 \leftarrow x_1 \cdot sigmoid(x_1 w_s + b_s)
$$

$$
f_2 \leftarrow x_2 \cdot sigmoid(x_2 w_s + b_s)
$$

$$
y \leftarrow f_1 + f_2
$$

여기서 $w_s$, $b_s$는 shared separable convolution의 파라미터다. 각 feature는 gate를 통해 중요도가 조절된 뒤 fusion된다. 저자들의 주장에 따르면 이 구조는 단순 summation보다 feature fusion 품질이 좋고, level 간 gradient 전달도 더 적절히 제어한다.

논문은 역전파 식도 전개하며, gate를 통해 각 입력 feature의 gradient 기여가 조절된다는 점을 설명하려 한다. 다만 본문 추출 상태가 다소 깨져 있어 정확한 미분식의 표기는 일부 불완전하다. 그럼에도 핵심 메시지는 분명하다. **GFPN은 단순 합이 아니라, 입력별 중요도를 학습적으로 조절하는 gated fusion이다.**

### Mask branch: single-pixel-based decoder

mask branch는 크게 세 단계로 이해할 수 있다.

#### 1) Multi-scale fusion으로 pixel 표현 강화

단일 pixel이 객체 전체를 대표하려면 receptive field가 충분히 커야 한다. 이를 위해 논문은 mask estimation 전에 추가 convolution들을 적용한다.

* $1 \times 1$ convolution 1개, 256 channels
* $3 \times 3$ convolution 3개, dilation rate는 각각 $[2, 4, 6]$, 각 128 channels

이 feature들을 channel-wise concatenation하면, 각 pixel은 다양한 스케일의 형태 정보를 함께 담게 된다. 저자들은 dilation convolution이 중요한 이유로, positive pixel이 객체의 정확한 중심에 있지 않더라도 더 넓은 영역을 볼 수 있어야 객체 전체 형태를 포착할 수 있기 때문이라고 설명한다.

#### 2) Positive pixel sampling

SPRNet은 각 pixel 위치마다 9개의 anchor를 생성한다. 이는 3개 크기와 3개 비율을 조합한 것이다. mask branch 학습을 위해서는 어떤 pixel이 특정 instance를 대표하는지 정해야 한다.

논문은 다음 규칙을 사용한다.

* 어떤 pixel 위치의 9개 anchor 중 하나라도 GT instance box와 IoU가 0.7보다 크면 그 pixel을 **positive pixel**로 본다.
* 이후 각 pixel 위치에서 가장 overlap이 큰 anchor를 선택해 학습에 사용한다.
* 총 300개의 pixel을 사용하며, 각 pixel은 하나의 instance 생성을 담당한다.

이때 IoU threshold가 매우 중요하다고 논문은 강조한다. threshold가 너무 낮으면 객체 밖의 pixel도 positive가 되어 mask 복원이 어려워지고 학습이 불안정해진다. 반대로 threshold가 너무 높으면 학습은 쉬워질 수 있으나, inference 시 classification score 상위 100개 pixel을 쓰므로 실제로는 객체 중심이 아닐 가능성이 있어 일반화가 떨어질 수 있다. 저자들은 ablation을 통해 **0.7이 가장 적절하다**고 결론 내린다.

#### 3) Single Pixel Reconstruction

positive pixel 하나는 shared decoder를 통해 점진적으로 mask로 복원된다. 논문 설명에 따르면:

* 먼저 activation 없이 3개의 연속된 deconvolution을 사용하여 점차 $8 \times 8$ mask를 생성한다.
* 이후 ReLU가 포함된 deconvolution 2개를 더 적용하여 최종 $32 \times 32$ mask를 만든다.
* $8 \times 8$ feature map에서 최종 classification layer로 nearest interpolation shortcut도 추가한다.

저자들이 초반 3개 deconvolution에 activation을 넣지 않는 이유는, 매우 작은 feature map에서 ReLU가 많은 neuron을 0으로 만들면 중요한 정보가 손실될 수 있다고 보기 때문이다.

최종적으로 복원된 출력은 $32 \times 32 \times K$의 mask map 형태로 이해할 수 있다. 여기서 $K$는 클래스 수다. classification branch가 예측한 클래스 점수에 따라 해당 클래스 채널을 선택하고, 그 channel의 mask를 인스턴스 mask로 사용한다. 손실 계산은 모든 클래스에 대해 하지 않고, **on-class mask loss만 계산**한다고 설명한다.

### 학습 및 추론 절차

#### 라벨 준비

box detection용 anchor labeling은 일반적인 detector와 유사하다.

* IoU > 0.5: positive anchor
* IoU < 0.4: negative anchor

mask branch용 positive pixel labeling은 더 엄격하게 IoU > 0.7을 사용한다.

#### 학습

논문은 MS-COCO 2017 train set에서 총 25 epoch 학습했다고 밝힌다. optimizer는 **Adam**, 초기 학습률은 $10^{-5}$, gradient clip은 $10^{-3}$이다. backbone은 ImageNet pretraining을 사용하며, 전체 네트워크를 end-to-end로 학습한다.

#### 추론

추론 시에는 GT가 없으므로 anchor overlap을 사용할 수 없다. 대신 classification branch가 출력한 score 중 **상위 100개 pixel**을 선택한다. 각 pixel은 decoder를 통해 $32 \times 32$ mask를 만들고, regression branch가 예측한 box 크기에 맞게 bilinear interpolation으로 resize하여 최종 instance mask를 생성한다.

즉 추론 파이프라인은 다음과 같다.

1. classification branch로 점수 높은 pixel 선택
2. regression branch로 해당 객체의 bounding box 추정
3. mask branch로 $32 \times 32$ mask 생성
4. box 크기에 맞춰 mask resize
5. 최종 instance segmentation 출력

## 4. 실험 및 결과

### 데이터셋과 평가 지표

모든 주요 실험은 **MS-COCO**에서 수행되었다. 학습은 115k train 이미지, ablation은 5k val 이미지에서 수행했다. 평가 지표는 COCO standard metric을 사용한다.

* AP
* $AP_{50}$
* $AP_{75}$
* $AP_S$, $AP_M$, $AP_L$

그리고 추가적으로 box detection과 segmentation에 대해 **Average Recall (AR)**도 보고한다.

### 메인 결과: instance segmentation

논문의 핵심 비교는 Table II에 있다. 대표 결과를 정리하면 다음과 같다.

* **SPRNet, ResNet-50-GFPN, 500px**: mask AP 30.4, 10 fps
* **SPRNet, ResNet-50-GFPN, 800px**: mask AP 32.0, 9 fps
* **SPRNet, ResNet-101-GFPN, 800px**: mask AP 34.0, 7 fps

비교 대상인 Mask R-CNN은 다음과 같다.

* **Mask R-CNN, ResNet-50-FPN, 800px**: mask AP 33.6, 7 fps
* **Mask R-CNN, ResNet-101-FPN, 800px**: mask AP 35.7, 5 fps
* **Mask R-CNN, ResNet-101-GFPN, 800px**: mask AP 36.0, 5 fps

즉 ResNet-50 기준으로 보면, SPRNet 800px GFPN은 **32.0 AP / 9 fps**, Mask R-CNN 800px FPN은 **33.6 AP / 7 fps**다. 성능은 약 1.6 AP 낮지만 속도는 더 빠르다. 논문은 이 점을 one-stage 구조의 장점으로 강조한다.

또한 정성적 분석에서는 SPRNet이 crowded scene, unusual morphology, extreme scale 등 어려운 조건에서도 양호한 성능을 보이며, **겹치는 객체의 경계를 robust하게 구분**한다고 주장한다. 다만 세밀한 boundary quality는 Mask R-CNN이 더 낫다고 인정한다.

### Ablation study

#### Fusion paths

단순한 연속 $3 \times 3$ convolution 4개(C33x4)보다, dilation rate가 다른 병렬 convolution(C33-1,2,4,6)이 더 좋았다.

* C33x4: AP 29.2
* C33-1,2,4,6: AP 30.4

즉 **1.2 AP 향상**이 있었다. 저자들은 다양한 dilation을 통해 형태 변화에 더 잘 대응한다고 해석한다.

#### FPN vs. GFPN

GFPN의 효과는 mask AP와 box AP 모두에서 나타난다.

* FPN: mask $AP^m = 29.8$, box $AP^{bb} = 32.5$
* GFPN: mask $AP^m = 30.4$, box $AP^{bb} = 33.6$

즉 GFPN은 segmentation과 detection 모두를 안정적으로 개선했다.

#### Shortcut의 효과

decoder에 shortcut을 추가했을 때도 개선이 있었다.

* Deconv only: AP 30.0
* Deconv + shortcut: AP 30.4

향상폭은 크지 않지만, decoder의 중간 표현을 최종 예측에 연결하는 것이 유리하다는 결과다.

#### Mask IoU threshold

positive pixel 선정 threshold를 비교한 결과:

* IoU > 0.5: AP 30.2
* IoU > 0.7: AP 30.4

즉 **0.7이 더 좋은 선택**이었다. 이는 앞서 설명한 것처럼, 너무 느슨한 threshold가 mask 학습을 어렵게 만들기 때문으로 해석된다.

### Box detection 성능

논문은 SPRNet이 segmentation뿐 아니라 object detection에서도 강하다고 주장한다. Table IV에 따르면:

* **RetinaNet 500, ResNet-50-FPN**: box AP 32.5
* **SPRNet 500, ResNet-50-GFPN**: box AP 33.6

즉 같은 ResNet-50 조건에서 **1.1 AP 개선**이 있었다.

800px에서도

* RetinaNet 800: 35.7
* SPRNet 800: 36.0

으로 개선이 있다. 특히 small object detection에서 강점을 보인다고 주장한다.

GFPN만 따로 떼어 본 Table V에서도, 모든 input scale에서 GFPN이 FPN보다 우수했다. 예를 들어 800 scale 기준:

* FPN: $AP^{bb}=35.7$, $AP_S^{bb}=18.9$, $AP_L^{bb}=46.3$
* GFPN: $AP^{bb}=36.0$, $AP_S^{bb}=20.1$, $AP_L^{bb}=48.6$

small, medium, large 모두 향상되었고, 특히 large object detection에서도 개선이 나타난 점을 저자들은 gradient blocking 효과의 근거로 해석한다.

### Recall 분석

Table VI, VII에서 AR을 비교한다. SPRNet은 one-stage의 특성상 **더 많은 객체를 검출하는 경향**, 즉 높은 recall을 보인다. 논문은 Mask R-CNN보다 segmentation detail은 다소 떨어질 수 있지만, **overall recall은 우수**하다고 주장한다.

예를 들어 segmentation AR에서:

* Mask R-CNN, ResNet-101-FPN: $AR_{100}^{seg}=47.6$
* SPRNet, ResNet-101-GFPN: $AR_{100}^{seg}=49.5$

box AR에서도 GFPN 적용 모델이 전반적으로 개선된다. 이는 GFPN이 detection framework 전반에 유연하게 적용될 수 있음을 보여주는 결과로 제시된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정 자체의 신선함**이다. 당시 주류였던 two-stage instance segmentation 흐름에서 벗어나, **proposal-free one-stage instance segmentation이 가능함을 구체적으로 보인 점**이 중요하다. 단지 아이디어 수준이 아니라, RetinaNet 기반의 비교적 일관된 구조 위에서 실제 COCO 실험까지 수행해 설득력을 확보했다.

두 번째 강점은 **설계의 단순성과 효율성**이다. RoI Pooling이나 RoI Align 없이, pixel 단위 representation으로 mask를 생성한다는 것은 메모리와 속도 측면에서 매력적이다. 실험에서도 Mask R-CNN 대비 더 빠른 inference speed를 보고한다.

세 번째 강점은 **GFPN 제안**이다. 이 모듈은 SPRNet 내부 전용 장치가 아니라, RetinaNet이나 Mask R-CNN 같은 다른 detector에도 적용 가능한 일반적 feature fusion 개선 장치로 제시된다. 실제 논문에서도 GFPN이 box AP와 AR을 꾸준히 향상시키는 결과를 보였다.

네 번째 강점은 **recall 측면의 장점**이다. 논문은 one-stage detector의 본질적 장점 덕분에 더 많은 객체를 놓치지 않는 경향이 있다고 분석한다. 특히 crowd scene이나 객체 수가 많은 경우, proposal 수에 제약을 받는 two-stage와 다른 장점을 보여줄 수 있다는 주장이다.

반면 한계도 명확하다. 가장 큰 한계는 **mask 품질의 세밀함이 two-stage보다 부족하다**는 점이다. 논문 스스로도 Mask R-CNN이 더 정교한 boundary와 alignment를 제공한다고 인정한다. 이는 구조적으로 자연스럽다. Mask R-CNN은 box 정렬 후 RoI 기반 binary segmentation을 수행하므로 mask 생성이 상대적으로 쉬운 문제로 바뀌지만, SPRNet은 단일 pixel로부터 전체 mask를 복원해야 하므로 훨씬 어렵다.

또 다른 한계는 **regression 오차에 민감하다**는 점이다. 최종 mask는 regression branch가 예측한 box에 맞춰 resize되므로, box가 크게 틀리면 mask alignment도 쉽게 무너진다. two-stage 구조보다 localization error의 영향을 더 크게 받을 수 있다.

또한 GFPN의 이론적 정당화는 흥미롭지만, 논문에 제시된 gradient 관련 설명은 다소 직관적 수준에 머문다. “summation이 gradient 간섭을 만든다”는 설명은 설계 동기를 주는 데는 충분하지만, 그것이 실제로 어떤 최적화 이득으로 이어지는지에 대한 이론적 엄밀성은 강하지 않다. 실험적으로는 개선이 보이지만, 왜 그 개선이 발생했는지를 완전히 증명했다고 보기는 어렵다.

그리고 single-pixel reconstruction 자체도 본질적으로 어려운 문제다. 한 pixel이 인스턴스 전체 모양을 대표하도록 만드는 과정은 매우 압축적인 표현 학습을 요구한다. 논문은 dilation convolution과 multi-scale fusion으로 이를 보완하지만, 복잡한 모양이나 아주 세밀한 구조에서는 한계가 남는다.

마지막으로, 논문은 “first one-stage instance segmentation model”이라고 강하게 주장하지만, 이 표현은 당시 문맥에서는 상당히 도전적인 주장이다. 제공된 본문만으로는 동시대 모든 관련 방법과의 우선성 비교를 완전히 검증할 수 없다. 따라서 이 부분은 **논문의 자기 주장**으로 이해하는 것이 적절하다.

## 6. 결론

이 논문은 instance segmentation에서 proposal 기반 two-stage 패러다임을 벗어나기 위해, **SPRNet이라는 one-stage 구조**를 제안했다. 핵심은 각 객체를 RoI가 아니라 **단일 feature pixel로 대표**하고, 그 pixel에서 deconvolution decoder를 통해 mask를 복원하는 것이다. 이를 위해 multi-scale fusion과 dilation convolution으로 pixel representation을 강화하고, GFPN으로 pyramid feature fusion을 개선했다.

실험 결과는 이 접근이 충분히 실용적 가능성이 있음을 보여준다. SPRNet은 Mask R-CNN보다 세밀한 mask boundary에서는 다소 부족하지만, 비슷한 수준의 segmentation AP에 접근하면서 더 빠른 속도를 달성했다. 또한 detection 측면에서는 RetinaNet 대비 box AP 향상도 확인했다.

이 연구의 의미는 단순히 한 모델의 성능 수치에 있지 않다. 더 중요한 점은 **instance segmentation도 one-stage로 설계할 수 있다**는 방향성을 구체화했다는 데 있다. 이후 연구들은 더욱 정교한 dense prediction, center-based representation, dynamic kernel, transformer 기반 mask modeling 등으로 발전했는데, SPRNet은 그 초기 단계에서 **“proposal 없이 인스턴스 마스크를 직접 생성할 수 있다”는 가능성**을 보여준 사례로 볼 수 있다.

실제 적용 측면에서는 속도가 중요한 환경, 예를 들어 실시간 비전 시스템이나 edge device 환경에서 의미 있는 아이디어를 제공한다. 향후 연구에서는 single-pixel representation의 표현력을 더 높이거나, box와 mask의 결합 방식을 더 정교하게 설계함으로써, one-stage instance segmentation의 정확도 한계를 더 줄일 수 있을 것이다. 그런 점에서 이 논문은 완성형이라기보다, 이후 발전을 촉진한 **개척적 시도**로 평가하는 것이 타당하다.
