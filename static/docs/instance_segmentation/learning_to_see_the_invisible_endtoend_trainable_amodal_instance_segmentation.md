# Learning to See the Invisible: End-to-End Trainable Amodal Instance Segmentation

* **저자**: Patrick Follmann, Rebecca König, Philipp Härtinger, Michael Klostermann
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1804.08864](https://arxiv.org/abs/1804.08864)

## 1. 논문 개요

이 논문은 **semantic amodal segmentation** 문제를 다룬다. 이는 일반적인 instance segmentation이 보이는 영역만 분할하는 것과 달리, 각 객체 인스턴스에 대해 **보이는 영역(visible/modal mask)** 뿐 아니라 **가려져 보이지 않는 영역(invisible/occlusion mask)** 까지 포함한 **전체 객체 영역(amodal mask)** 을 예측하는 문제다. 논문은 이 세 가지 마스크를 하나의 모델에서 동시에 예측하는 **end-to-end trainable** 구조를 제안한다.

연구 문제는 명확하다. 2차원 이미지에서 객체가 다른 객체에 의해 부분적으로 가려져 있을 때, 단지 현재 보이는 픽셀만으로는 실제 객체의 전체 형상을 알 수 없다. 하지만 로봇 조작이나 산업 환경에서는 이 정보가 매우 중요하다. 예를 들어, 집기(picking)나 배치(placing)를 하려면 어떤 객체가 가려져 있는지, 어느 객체를 먼저 치워야 하는지, 혹은 관심 객체가 실제로 어디까지 확장되어 있는지를 알아야 한다. 따라서 이 문제는 단순 시각 인식을 넘어 **장면 이해(scene understanding)** 와 **물리적 상호작용 가능성 판단**에 직접 연결된다.

논문은 특히 기존 방법들이 amodal mask만 예측하거나, visible/invisible 정보를 별도 모델 또는 비일체형 방식으로 다뤘던 점을 지적한다. 이에 반해 제안 모델은 한 번의 forward pass에서 객체의 클래스, bounding box, amodal mask, visible mask, invisible mask를 함께 예측한다. 저자들은 이를 통해 계산량 증가를 크게 억제하면서도 더 풍부한 장면 해석을 달성할 수 있다고 주장한다.

또한 논문은 모델 제안에 그치지 않고, 평가를 위해 **D2S amodal**과 **COCOA cls**라는 두 개의 데이터셋을 추가로 제공한다. 전자는 산업용 상품 장면에 대한 고품질 amodal annotation을 제공하며, 후자는 기존 COCOA에 COCO class annotation을 결합해 class-specific amodal segmentation을 가능하게 한 데이터셋이다. 이를 통해 저자들은 자신들의 방법을 class-agnostic 환경과 class-specific 환경 모두에서 검증한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **Mask R-CNN을 기반으로 하되, visible mask와 amodal mask를 동시에 예측하고, invisible mask는 이 둘의 관계를 활용해 연결된 구조로 예측하자**는 것이다. 즉, invisible 영역을 완전히 독립된 별도 문제로 취급하지 않고, 객체의 전체 영역과 실제 보이는 영역 사이의 차이를 구조적으로 모델에 반영한다.

핵심 직관은 다음과 같다. 어떤 객체의 전체 형상(amodal)은 보이는 부분(visible)과 가려진 부분(invisible)의 합집합이다. 따라서 수식적으로는 대략 $AM = VM \cup IVM$의 관계를 갖고, 이산 마스크 관점에서는 invisible mask가 amodal mask와 visible mask의 차이로 해석될 수 있다. 이 관계를 네트워크 구조에 반영하면, 세 마스크를 완전히 독립적으로 학습하는 것보다 더 일관된 예측이 가능하다는 것이 저자들의 생각이다.

기존 접근 방식과의 차별점은 세 가지다.

첫째, 논문은 자신들의 모델이 **semantic amodal segmentation을 위한 최초의 all-in-one end-to-end multi-task model**이라고 주장한다. 이전 작업 중 Li and Malik의 방법은 iterative하게 bounding box를 확장하는 방식이었고, Zhu 등은 COCOA 데이터셋과 baseline을 제시했지만 class-agnostic하고, visible/invisible/amodal을 하나의 통합 네트워크에서 동시에 예측하는 구조는 아니었다.

둘째, 제안 모델은 **같은 RoI feature**를 바탕으로 amodal과 visible mask를 함께 예측한다. 이렇게 하면 두 마스크가 동일 객체 proposal에 자연스럽게 정렬되며, 두 별도 모델의 출력을 나중에 후처리로 결합해야 하는 복잡함이 사라진다.

셋째, invisible mask를 얻기 위해 단순히 별도의 head를 둔 것이 아니라, **amodal logit에서 visible logit을 빼는 방식**으로 occlusion logit을 형성한다. 이때 visible logit에 ReLU를 적용해 음수 값 때문에 생길 수 있는 잘못된 occlusion 예측을 완화한다. 이 설계는 “보이는 영역과 전체 영역의 차이”라는 문제 구조를 그대로 반영한 것이다.

정리하면, 이 논문의 아이디어는 새로운 거대 backbone이나 복잡한 generative reasoning을 제안하는 것이 아니라, **Mask R-CNN의 인스턴스 분할 파이프라인을 유지하면서 amodal/visible/invisible의 구조적 관계를 효율적으로 주입**하는 데 있다. 그래서 모델은 비교적 가볍고, 기존 pretrained mask head를 활용할 수 있으며, 실제 성능 향상도 얻는다.

## 3. 상세 방법 설명

### 전체 구조

저자들이 제안한 모델은 **Occlusion R-CNN (ORCNN)** 이며, 기본 골격은 **Mask R-CNN (MRCNN)** 이다. ORCNN은 MRCNN 위에 두 가지를 추가한다.

하나는 **amodal mask head**, 다른 하나는 **occlusion mask head**다. visible mask prediction은 원래 Mask R-CNN의 mask head가 담당한다고 볼 수 있다. 결과적으로 ORCNN은 각 RoI에 대해 다음 정보를 예측한다.

* 객체 클래스
* bounding box
* amodal mask
* visible mask
* invisible mask

중요한 점은 이 모든 예측이 **하나의 forward pass**에서 수행된다는 것이다.

### RoI와 mask head의 설계

visible mask head와 amodal mask head는 동일한 구조를 사용한다. 논문에 따르면 각 head는 **4개의 $3 \times 3$ convolution + ReLU layer**로 구성되어 있으며, 입력은 **RoIAlign**에서 추출된 feature다. 즉, proposal마다 잘린 feature를 바탕으로 두 종류의 mask를 병렬적으로 예측한다.

이때 학습과 추론 모두에서 **amodal head와 visible head가 같은 box proposal**을 사용한다. Proposal은 RPN이 생성하며, 여기서 중요한 차이가 하나 있다. RPN의 ground truth box target은 **amodal instance의 bounding box**를 기준으로 한다. 따라서 visible mask head는 “보이는 객체를 보이는 box 안에서 예측”하는 것이 아니라, **amodal box 안에서 visible region을 예측**해야 한다. 저자들은 이것이 일반 modal segmentation 모델과의 큰 차이라고 강조한다.

이는 직관적으로도 어렵다. visible mask만 예측하는 일반 모델은 보이는 물체 경계에 맞춰 학습되지만, ORCNN의 visible head는 더 큰 영역 안에서 “실제로 보이는 부분만” 골라내야 한다. 대신 이 설계 덕분에 amodal과 visible mask가 항상 동일한 객체 좌표계 안에서 정렬된다.

### invisible mask head의 설계

가장 특징적인 부분은 invisible mask를 계산하는 방식이다. 논문은 invisible mask head가 본질적으로 **amodal mask logits에서 visible mask logits을 빼는 방식**이라고 설명한다. 단, 그냥 빼는 것이 아니라 visible logits에 먼저 **ReLU**를 적용한다.

개념적으로는 다음과 같이 이해할 수 있다.

$$
\text{occlusion_logits} \approx \text{amodal_logits} - \mathrm{ReLU}(\text{visible_logits})
$$

논문이 ReLU를 강조하는 이유는, visible logit이 음수일 때 단순 감산을 하면 실제로는 amodal도 visible도 없는 픽셀에서 가짜 occlusion 반응이 생길 수 있기 때문이다. ReLU를 적용하면 visible 쪽에서 음수 신호를 제거해 이런 현상을 줄일 수 있다.

이 설계의 장점은 invisible mask를 완전히 독립적으로 예측하는 것이 아니라, **전체 물체와 보이는 물체의 차이**라는 구조적 관계를 직접 반영한다는 점이다. 또한 추가 계산량이 크지 않다. 저자들은 전체적으로 오직 **5개의 추가 convolution 모듈과 2개의 sigmoid layer**만 더 필요하다고 말한다.

### 학습 목표와 손실 함수

ORCNN은 classification loss와 box regression loss 외에, 세 종류의 mask에 대해 각각 손실을 둔다.

* $L_{\mathit{AM}}$: amodal mask loss
* $L_{\mathit{VM}}$: visible mask loss
* $L_{\mathit{IVM}}$: invisible mask loss

전체 손실은 다음과 같다.

$$
L = L_{\mathit{cls}} + L_{\mathit{box}} + L_{\mathit{AM}} + L_{\mathit{VM}} + L_{\mathit{IVM}}
$$

각 마스크 손실은 Mask R-CNN과 유사하게, 픽셀별 sigmoid를 적용한 뒤 **average binary cross-entropy**를 계산하는 방식이다.

표면적으로 보면 $IVM = AM - VM$ 관계가 있으므로 세 손실 중 하나는 중복처럼 보인다. 실제로 논문도 이를 인정한다. 하지만 저자들은 **occlusion loss를 추가하지 않으면 amodal logits과 visible logits의 scale이 맞지 않아 subtraction 기반 occlusion prediction이 망가진다**고 설명한다.

예를 들어 어떤 픽셀에서 amodal logit이 14, visible logit이 10이라면, 두 값 모두 높은 확신의 positive일 수 있다. 그런데 단순 차이를 취하면 4가 되고, sigmoid를 통과하면

$$
\frac{1}{1 + e^{-4}} \approx 0.982
$$

가 되어 occlusion probability가 매우 높게 나온다. 하지만 실제로 그 픽셀은 visible 영역일 수 있으므로 이런 결과는 잘못이다. 따라서 invisible loss를 함께 걸어 두 logits이 의미 있는 상대 scale을 갖도록 학습시켜야 한다는 것이 논문의 논리다.

반대로 $L_{\mathit{VM}}$를 제거하면 invisible mask는 그럴듯해도 visible mask가 나빠진다고 보고한다. 또한 logits가 아니라 sigmoid 이후의 probability를 직접 빼거나 곱하는 방식도 실험했지만, 수치적으로 불안정했다고 한다. 따라서 논문은 **logit subtraction + 세 종류의 BCE loss** 조합을 최종 설계로 채택했다.

### 평가 지표 설계

이 논문은 단순히 기존 AP를 그대로 쓰지 않고, amodal/visible 문제에 맞는 평가 방식을 제안한다.

기본적으로는 COCO 스타일 AP와 동일하게, IoU threshold를 0.5부터 0.95까지 0.05 간격으로 평균한 mean AP를 사용한다. 다만 어떤 마스크를 기준으로 TP를 정할지에 따라 여러 AP를 정의한다.

* $AP_A$: amodal mask 기준 AP
* $AP_V$: visible mask 기준 AP
* $AP_{AV}$: amodal과 visible 둘 다 맞아야 TP로 인정하는 AP

예를 들어 threshold $t$에서 $AP_{AV}$의 true positive가 되려면 클래스가 맞고, 동시에

$$
\mathrm{IoU}(AM^G, AM^P) > t
$$

그리고

$$
\mathrm{IoU}(VM^G, VM^P) > t
$$

를 모두 만족해야 한다.

invisible mask는 직접 포함하기 어렵다고 설명한다. 이유는 두 가지다. 첫째, non-occluded object에는 invisible mask가 존재하지 않아서 일반적인 recall 정의가 애매하다. 둘째, invisible mask는 대체로 면적이 작아 IoU가 매우 민감하게 흔들린다. 그래서 세 마스크를 모두 동시에 요구하는 지표는 지나치게 불안정해질 수 있다.

따라서 논문은 invisible mask 성능을 보기 위해 **occluded object만 따로 골라** 평가하고, 이 경우 $AP^{0.5}_{IV}$를 사용한다. threshold를 0.5로 낮춘 것도 invisible mask의 난이도와 작은 크기를 고려한 선택이다.

### 학습 설정

부록에 따르면, 구현은 Detectron 기반이며 COCO-pretrained weight로 fine-tuning했다. 학습은 2개의 GTX 1080 Ti GPU에서 진행되었고, D2S amodal에서는 GPU당 1 image, COCOA/COCOA cls에서는 GPU당 2 image를 사용했다. RoI batch size는 image당 256, RPN pre-NMS proposal 수는 feature pyramid level당 1000이다. 이미지는 short side 최대 800, long side 최대 1111로 조절했다.

총 10000 iteration 동안 학습했으며, base learning rate는 0.0025, weight decay는 0.0001이다. learning rate는 6000과 8000 iteration에서 각각 $\gamma=0.1$을 곱해 감소시켰다. COCOA와 D2S amodal의 경우 class 수와 의미가 COCO와 다르므로, 최종 class-specific output layer는 랜덤 초기화했다.

## 4. 실험 및 결과

### 사용 데이터셋

논문은 세 데이터셋 계열에서 실험한다.

첫째는 **COCO amodal (COCOA)** 이다. 이 데이터셋은 2500 train, 1323 val, 1250 test image를 포함하며, image마다 대부분의 object와 background stuff region에 대해 amodal mask annotation이 있다. occluded object에 대해서는 visible/invisible mask도 제공한다. 다만 모든 object가 단일 category object에 속하는 **class-agnostic** 설정이며, stuff도 포함된다.

둘째는 **COCOA cls**다. 이는 COCOA와 COCO 2014 annotation을 visible mask IoU 0.75 기준으로 매칭해 class label을 부여한 데이터셋이다. 이 과정에서 일부 클래스는 사라지고, 일부 object는 매칭되지 않아 누락되므로 annotation completeness 문제가 있다. 그래도 class-specific amodal segmentation 성능을 볼 수 있다는 장점이 있다.

셋째는 **D2S amodal**이다. D2S는 산업 응용을 지향한 supermarket product dataset인데, 저자들이 여기에 amodal/visible/invisible annotation을 추가했다. 특히 D2S는 배경이 단순하고 training set의 복잡도가 낮아 강한 augmentation이 가능하다. 논문은 이 점을 적극 활용한다.

### COCOA: baseline과 제안 모델 비교

COCOA의 amodal mask prediction baseline에서, 저자들은 MRCNN을 amodal annotation으로 학습한 **AmodalMRCNN**을 사용한다. Table 1에 따르면, 기존 baseline인 **AmodalMask [20]** 대비 성능 차이가 매우 크다.

* AmodalMask의 전체 $AP_A$는 **5.7**
* AmodalMRCNN-50의 전체 $AP_A$는 **29.9**

즉, 평균 정밀도에서 대폭 향상된다. recall도 AmodalMask가 image당 1000 region을 내는 방식임에도, AmodalMRCNN은 평균 30개 결과만으로 더 높은 recall을 달성했다고 설명한다. 다만 stuff는 예외인데, 큰 배경 영역이라 proposal 기반 모델이 잘 다루지 못한다.

이후 논문은 stuff를 제외한 **COCOA no stuff** 설정에 집중한다. 이유는 stuff region은 경계 정의가 애매하고 annotator 간 분산이 크기 때문이다.

COCOA 계열 주요 결과는 Table 3에 제시되어 있다. COCOA no stuff 기준으로 보면:

* MRCNN-101: $AP_{AV}=22.0$, $AP_A=23.9$, $AP_V=27.9$
* AmodalMRCNN-101: $AP_{AV}=27.8$, $AP_A=35.6$, $AP_V=29.4$
* ORCNN: $AP_{AV}=25.1$, $AP_A=30.1$, $AP_V=30.0$, $AP^{0.5}_{IV}=3.0$

이 결과는 중요한 해석을 준다. **Amodal mask 자체는 AmodalMRCNN이 더 강하다**. 실제로 $AP_A$는 ORCNN보다 높다. 그러나 **visible mask 품질은 ORCNN이 더 좋거나 비슷하며**, 동시에 invisible mask도 예측할 수 있다. 즉, ORCNN은 한 작업만 가장 잘하는 모델은 아니지만, 세 작업을 통합해 수행하는 실용적 타협점으로 볼 수 있다.

COCOA 전체 설정에서도 비슷한 경향이 보인다. ORCNN의 $AP_A$는 AmodalMRCNN보다 낮지만, $AP_V$는 더 좋아진다. 논문은 이를 멀티태스크 학습의 특성으로 해석한다.

### COCOA에서의 데이터 증강

논문은 COCOA에서 overfitting이 심하다고 보고한다. AmodalMRCNN은 단 1 epoch 후 이미 overfit 경향을 보였다고 한다. 이를 해결하기 위해 두 가지 synthetic augmentation을 시도했다.

하나는 COCOA training image 간의 object를 서로 overlay하는 방식이고, 다른 하나는 원래 COCO의 modal annotation object를 다른 COCO image 위에 붙여 amodal 학습 샘플을 만드는 방식이다.

그러나 결과는 기대와 달랐다. Table 2를 보면:

* AmodalMRCNN-101: $AP_A=35.6$
* COCOA 기반 증강(*): $AP_A=34.7$
* COCO 기반 증강(**): $AP_A=24.1$

즉, augmentation이 성능을 올리지 못했고, 특히 modal-only 기반 증강은 오히려 크게 악화되었다. 저자들은 COCOA에서는 객체들이 **문맥 속에서 함께 존재하는 정보**가 중요하고, 단순 합성 이미지가 그 문맥을 잘 재현하지 못한다고 해석한다.

### COCOA cls: class-specific amodal segmentation

COCOA cls에서는 class-specific prediction의 이점을 확인하려 한다. Table 4 결과는 다음과 같다.

* MRCNN-101: $AP_{AV}=39.0$, $AP_A=39.8$, $AP_V=44.9$
* AmodalMRCNN-101 (cls-agn): $AP_A=40.4$
* AmodalMRCNN-101: $AP_A=41.7$
* ORCNN (cls-agn): $AP_V=39.3$, occluded $AP^{0.5}_{IV}=1.8$
* ORCNN: $AP_V=39.4$, occluded $AP^{0.5}_{IV}=2.0$

전체적으로 COCOA cls의 수치는 COCOA보다 훨씬 높다. 이는 class label이 추가되면서 객체 형상 priors를 더 잘 학습할 수 있음을 시사한다. 특히 AmodalMRCNN에서 class-specific head가 class-agnostic보다 대부분 지표에서 조금 더 좋다. 저자들은 이를 **amodal completion이 class-specific한 성격을 가진 작업**이라는 근거로 본다.

하지만 ORCNN이 AmodalMRCNN을 넘지는 못한다. 논문은 이유를 몇 가지로 설명한다. 데이터셋 자체가 약한 occlusion object에 편향되었을 수 있고, ORCNN은 visible/invisible까지 함께 예측하면서 error source가 늘어난다. 또한 ORCNN은 객체가 접촉만 해도 invisible 영역을 예측하려는 경향이 있어, visible mask에서 손해를 볼 수 있다.

중요한 주석도 있다. COCOA와 COCOA cls는 annotation이 완전하지 않아서, 모델이 실제로 맞는 예측을 해도 false positive로 계산될 수 있다. Figure 3에서도 ORCNN이 ground truth에는 없는 object/invisible region을 그럴듯하게 예측하는 예가 소개된다. 따라서 이 데이터셋의 AP 해석에는 주의가 필요하다.

### D2S amodal: 산업용 장면에서의 분석

D2S amodal은 논문이 특히 공들여 만든 데이터셋이다. training split에는 원래 occlusion이 거의 없기 때문에, 저자들은 augmentation을 통해 **1562장의 amodal augmented image**를 만들고, 원래 train_rot0 438장과 합쳐 총 2000장 train split을 구성한다. 또 modal annotation만 사용한 **modal augmented** 데이터도 별도로 만들어, amodal annotation 없이도 학습 가능한지를 시험한다.

Table 5에 따르면 D2S amodal train은 2000 image, 13066 object, 그중 8864개가 occluded object다. validation과 test보다 오히려 object-level occlusion rate가 높은 split을 만들 수 있었고, 이는 occlusion 학습에 유리하다.

Table 6은 D2S amodal에서의 핵심 결과다.

* AmodalMRCNN-101: $AP_{AV}=63.8$, $AP_A=72.6$, $AP_V=65.3$
* ORCNN (cls-agn): $AP_{AV}=62.3$, $AP_A=65.2$, $AP_V=71.0$, occluded $AP^{0.5}_{IV}=14.7$
* ORCNN: $AP_{AV}=58.9$, $AP_A=62.1$, $AP_V=66.2$, occluded $AP^{0.5}_{IV}=8.7$
* ORCNN (modal aug): $AP_{AV}=56.3$, $AP_A=59.9$, $AP_V=62.6$, occluded $AP^{0.5}_{IV}=7.8$

이 결과는 매우 흥미롭다.

첫째, **AmodalMRCNN은 여전히 amodal mask 자체에서는 최고 수준**이다. $AP_A=72.6$으로 ORCNN보다 높다.

둘째, **ORCNN class-agnostic 버전은 visible mask에서 매우 강하다**. $AP_V=71.0$으로 class-specific ORCNN보다 높고, occluded object 기준 invisible mask AP도 14.7로 가장 높다. 저자들은 이 결과가 COCOA/COCOA cls와는 다르다고 말한다. 즉, D2S처럼 object 종류가 비교적 규칙적이고 산업적 환경이 단순한 경우에는 class-agnostic mask prediction이 오히려 더 잘 작동할 수 있다.

셋째, **invisible mask는 여전히 매우 어렵다**. 저자들은 작은 invisible region에서는 IoU가 극도로 민감하고, 큰 invisible region에서는 형상 추정이 어렵다고 분석한다. 그래서 qualitative result는 꽤 괜찮아 보여도 $AP^{0.5}_{IV}$는 낮게 나온다.

넷째, 손실 구성 실험은 제안 구조의 타당성을 잘 보여준다.

* ORCNN without $L_{IV}$: $AP_{AV}=34.0$, $AP_V=34.7$
* ORCNN without $L_V$: $AP_{AV}=11.0$, $AP_V=11.1$
* ORCNN independent: $AP_{AV}=39.5$, $AP_V=40.3$
* ORCNN full: $AP_{AV}=58.9$, $AP_V=66.2$

특히 $L_V$를 없애면 visible 관련 성능이 크게 붕괴하고, full coupling 없이 independent하게 두어도 성능이 크게 떨어진다. 이는 visible/invisible loss가 단순 부가 요소가 아니라, **공유 RoI feature와 amodal head까지 함께 조정하는 멀티태스크 학습 구조가 실제로 필요함**을 보여준다.

다섯째, **modal annotation만으로 만든 증강 데이터로 학습한 ORCNN (modal aug)** 도 surprisingly strong하다. amodal full annotation으로 학습한 ORCNN보다 약간 낮지만, 차이가 아주 크지는 않다. 이는 D2S에서는 배경이 단순하고 객체 조합이 통제되어 있어 synthetic augmentation이 잘 통했기 때문이다. 반대로 COCOA에서는 이런 전략이 거의 먹히지 않았다. 이 차이는 데이터셋 성격의 중요성을 잘 보여준다.

### 정성적 결과

정성적 결과에서 ORCNN은 완전히 가려진 방향이나 image boundary 밖으로 이어지는 영역을 꽤 그럴듯하게 복원한다. 특히 D2S에서는 “위에 놓인 상품”이나 “프레임 밖으로 일부 잘려 나간 상품”에서 성능이 좋다. 이는 모델이 단순히 edge extrapolation을 하는 것이 아니라 **class shape prior**를 학습했음을 시사한다.

반면 failure case도 명확하다. 반사(reflection)나 조명 변화 때문에 가짜 occlusion을 예측하기도 하고, 서로 맞닿은 같은 클래스 객체 사이에서 mask를 이웃 객체까지 확장해 버리기도 한다. 또 어떤 경우에는 실제 occlusion을 전혀 잡지 못한다. 이는 invisible prediction이 강한 prior 추론 문제이며, appearance cue가 거의 없는 상황에서는 아직 불안정함을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 모델 설계가 매우 잘 맞물린다는 점이다. amodal, visible, invisible은 서로 독립적인 과제가 아니라 구조적으로 연결되어 있는데, ORCNN은 이 관계를 네트워크와 손실에 직접 반영했다. 특히 invisible mask를 amodal-visible 차이의 형태로 모델링한 점은 단순하지만 문제 구조를 잘 반영한 설계다.

또 하나의 강점은 **실용성**이다. 저자들은 완전히 새로운 대규모 구조를 제안한 것이 아니라 Mask R-CNN 기반에서 비교적 작은 확장만으로 목표를 달성했다. 따라서 기존 instance segmentation 파이프라인과의 연속성이 좋고, pretrained mask head를 활용할 수 있으며, 구현 부담도 상대적으로 낮다.

실험 설계도 장점이다. 단일 benchmark에만 의존하지 않고, COCOA, COCOA cls, D2S amodal을 모두 사용했다. 특히 D2S amodal과 COCOA cls를 새로 구성하여 class-specific amodal segmentation과 산업 환경에서의 가능성을 보여준 점은 연구 기여로서 가치가 있다. 또한 ablation을 통해 $L_V$, $L_{IV}$, coupled learning의 필요성을 구체적으로 보여준 점도 설득력이 높다.

하지만 한계도 분명하다. 가장 직접적인 한계는 **amodal mask 성능만 놓고 보면 AmodalMRCNN이 ORCNN보다 더 좋을 때가 많다**는 점이다. 즉, ORCNN은 모든 것을 한 번에 하는 대신, 단일 amodal segmentation 최적 모델은 아닐 수 있다. 멀티태스크 모델의 trade-off가 존재한다.

둘째, invisible mask 성능은 여전히 낮다. 논문도 이를 인정하며, 특히 $AP^{0.5}_{IV}$가 낮은 이유를 자세히 설명한다. 작은 invisible region은 IoU 측면에서 너무 민감하고, 큰 invisible region은 형상 추정 자체가 어렵다. 따라서 “보이지 않는 영역을 제대로 본다”는 주장은 정성적으로는 인상적이지만, 정량적으로는 아직 제한적이다.

셋째, 평가 데이터셋 문제도 있다. COCOA와 COCOA cls는 annotation completeness가 낮아 정확한 모델 평가를 방해한다. 논문 스스로도 ground truth 누락 때문에 맞는 예측이 false positive가 되는 경우를 지적한다. 이런 상황에서는 AP 수치만으로 모델의 실제 능력을 완전히 판단하기 어렵다.

넷째, 방법 자체가 **shape prior 기반 completion**에 가깝기 때문에, 객체 클래스 내부 형상 다양성이 큰 경우나 비정형 객체에서는 성능이 흔들릴 가능성이 있다. 논문은 이 가능성을 직접 실험하진 않았고, 여기에 대해서는 명시적 분석이 없다. 따라서 이 부분은 남는 질문이다.

다섯째, invisible mask를 logit subtraction으로 정의한 구조는 깔끔하지만, 이 방식이 최선인지에 대한 이론적 정당화는 제한적이다. 논문은 probability subtraction/multiplication을 시도했으나 unstable했다고 보고했지만, 더 정교한 structured constraint나 generative completion 방식과 비교한 분석은 없다. 즉, 좋은 engineering solution이지만 문제의 근본적 모형화로서 완결된 답이라고 보기는 어렵다.

비판적으로 해석하면, 이 논문은 “invisible reasoning”을 완전히 해결했다기보다, **instance segmentation 프레임워크 안에서 amodal reasoning을 현실적으로 다룰 수 있는 강한 baseline을 만든 작업**으로 보는 것이 더 정확하다. 그럼에도 불구하고 이 baseline은 이후 연구의 출발점으로 매우 중요하다.

## 6. 결론

이 논문은 semantic amodal segmentation을 위해 **Occlusion R-CNN (ORCNN)** 이라는 end-to-end multi-task 모델을 제안했다. ORCNN은 하나의 forward pass에서 객체 클래스, bounding box, amodal mask, visible mask, invisible mask를 동시에 예측한다. 핵심은 Mask R-CNN 구조를 바탕으로 amodal/visible 관계를 유지하면서 invisible prediction을 구조적으로 연결한 데 있다.

실험적으로는 기존 COCOA baseline보다 크게 향상된 amodal segmentation 성능을 보였고, COCOA cls와 D2S amodal에서 강한 baseline을 수립했다. 특히 D2S에서는 invisible mask 예측이 어렵지만 가능하다는 점, 그리고 modal annotation만으로 만든 synthetic augmentation만으로도 어느 정도 amodal 학습이 가능하다는 점을 보여주었다. 이는 산업용 장면처럼 통제된 환경에서는 데이터 구축 비용을 줄이면서도 실제 적용 가능성을 확보할 수 있음을 시사한다.

실제 응용 측면에서 이 연구는 로봇 집기, 물체 정리, 장면 재구성, occlusion-aware manipulation 등에서 중요한 역할을 할 가능성이 있다. 단순히 “보이는 것만” 인식하는 시스템에서 벗어나, **가려진 객체의 존재와 형상까지 추론하는 시각 시스템**으로 발전하는 데 중요한 디딤돌이 된다.

향후 연구 방향도 자연스럽다. invisible mask의 정량 성능 향상, 더 완전한 데이터셋 구축, stronger shape prior 또는 3D reasoning의 도입, temporal cue나 depth cue의 통합 등이 필요하다. 그런 의미에서 이 논문은 완성형 해법이라기보다, **amodal instance segmentation을 정식 문제로 세우고 실질적이고 재현 가능한 강한 기준점을 제공한 연구**로 평가할 수 있다.
