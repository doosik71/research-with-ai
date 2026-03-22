# A Simple Framework for Open-Vocabulary Segmentation and Detection

이 논문은 **OpenSeeD**라는 단일 프레임워크로, 서로 다른 supervision granularity를 가진 **segmentation 데이터와 detection 데이터를 함께 학습해** 하나의 **open-vocabulary segmentation/detection 모델**을 만들 수 있는지를 다룬다. 핵심 문제는 두 가지다. 첫째, detection과 segmentation은 **어휘(vocabulary)**와 **annotation granularity** 가 다르다. 둘째, segmentation은 foreground object뿐 아니라 background stuff도 다뤄야 하지만 detection은 거의 foreground만 다룬다. 저자들은 이 간극을 메우기 위해 **공유 semantic space**, **foreground/background decoupled decoding**, **conditioned mask decoding**을 결합한 간단한 encoder-decoder 구조를 제안한다. 그 결과 OpenSeeD는 joint pretraining만으로 open-vocabulary instance/panoptic segmentation과 detection 모두에서 강한 zero-shot transfer를 보이고, downstream fine-tuning에서도 COCO/ADE20K/Cityscapes 등에서 새로운 성능을 달성했다고 주장한다.  

## 1. Paper Overview

이 논문의 출발점은 최근 open-vocabulary vision이 주로 **detection만**, 또는 **segmentation만** 따로 다뤄 왔다는 점이다. 기존 방법들은 CLIP류 foundation model을 distillation하거나 image-text pair에서 pseudo label을 만드는 방식으로 fine-grained task를 확장했지만, detection과 segmentation을 **동시에** open-vocabulary로 학습하는 시도는 거의 없었다고 저자들은 지적한다. 특히 detection 데이터셋인 Objects365는 box annotation으로 365개 개념을 170만 장 규모에서 제공하는 반면, COCO segmentation은 133개 범주에 대해 약 10만 장 수준의 mask annotation만 제공하므로, 두 태스크의 supervision granularity와 semantic coverage가 크게 다르다.  

저자들이 던지는 핵심 질문은 이것이다. **“image-text weak supervision보다 더 깨끗하고 간극이 작은 detection과 segmentation을 직접 연결하면, 둘 다 잘하는 open-vocabulary 모델을 만들 수 있지 않을까?”** 이 논문은 그 질문에 대한 첫 강한 baseline으로 OpenSeeD를 제시한다. 논문은 기존의 pretrain-then-finetune 방식이 결국 detection용 모델과 segmentation용 모델을 따로 남긴다고 비판하고, OpenSeeD는 처음부터 **하나의 모델이 detection과 segmentation을 함께 처리하도록 설계**되었다고 강조한다.  

## 2. Core Idea

핵심 아이디어는 세 단계로 요약된다.

첫째, **single text encoder**를 사용해 detection/segmentation 데이터셋에 등장하는 모든 visual concept를 같은 semantic space에 넣는다. 이것은 두 데이터셋의 vocabulary 차이를 완화하고, 더 나아가 seen category 밖으로 open-vocabulary generalization을 노리는 기반이 된다. 저자들은 이 shared semantic space만으로도 segmentation-only 학습보다 더 나은 결과를 얻는다고 설명한다.

둘째, segmentation과 detection의 **task discrepancy**를 줄이기 위해 decoder의 query를 **foreground query** 와 **background query** 로 명시적으로 분리한다. foreground query는 detection과 segmentation 양쪽에서 공통으로 감독되고, background query는 segmentation에서만 사용된다. 이렇게 하면 segmentation의 background stuff가 foreground object decoding을 방해하는 문제를 줄일 수 있다.  

셋째, box와 mask supervision의 **data discrepancy**를 줄이기 위해 **conditioned mask decoding**을 도입한다. segmentation 데이터에서는 ground-truth box와 concept를 조건으로 mask를 복원하도록 학습하고, detection 데이터에서는 이 모듈이 box supervision만 있는 샘플에 대해 mask assistant 역할을 한다. 논문은 이것이 detection 데이터를 segmentation 학습에도 활용할 수 있게 만드는 핵심 장치라고 본다.  

즉 OpenSeeD의 novelty는 복잡한 새 backbone이 아니라, **detection과 segmentation을 함께 배우기 위해 꼭 필요한 세 가지 간극 완화 장치**를 매우 단순한 encoder-decoder 프레임 안에 넣었다는 데 있다.

## 3. Detailed Method Explanation

### 3.1 전체 구조

OpenSeeD는 **image encoder + text encoder + decoder** 구조다. 입력은 이미지 $I$ 와 전체 vocabulary $\mathcal{V}$ 이고, 출력은 mask prediction $\mathbf{P}^m$, box prediction $\mathbf{P}^b$, classification score $\mathbf{P}^c$ 이다. 저자들은 이것을

$$
\langle \mathbf{P}^m, \mathbf{P}^b, \mathbf{P}^c \rangle = \mathsf{OpenSeeD}(I,\mathcal{V})
$$

로 표현한다. 즉 모델은 애초에 detection과 segmentation 출력을 동시에 내도록 설계되어 있다.

### 3.2 Shared semantic space

모델은 segmentation dataset $\mathcal{D}_m = {I_i, (\mathbf{c}_i, \mathbf{m}_i)}$ 와 detection dataset $\mathcal{D}_b = {I_j, (\mathbf{c}_j, \mathbf{b}_j)}$ 를 함께 사용한다. 여기서 $\mathbf{c}$ 는 이미지 내 visual concepts, $\mathbf{m}$ 은 mask, $\mathbf{b}$ 는 box다. 전체 vocabulary $\mathcal{V}={c_1,\dots,c_K}$ 를 text encoder로 인코딩해 text feature를 만들고, image encoder가 만든 visual token과 정렬한다. 이 설계는 detection과 segmentation의 label space를 하나의 의미 공간으로 묶는 역할을 한다.

### 3.3 Decoupled foreground/background decoding

저자들은 segmentation에서 foreground object와 background stuff를 같은 query 집합으로 처리하면 foreground AP가 손해를 볼 수 있다고 본다. 그래서 query를 foreground용과 background용으로 나누고, Hungarian matching도 두 집합에 대해 독립적으로 수행한다. segmentation 데이터에서는 foreground/background decoding을 모두 사용하지만, detection 데이터에서는 foreground decoding만 사용한다. 그 결과 detection과 segmentation이 공유하는 정보는 foreground를 중심으로 최대한 협력하고, segmentation만의 background supervision은 별도로 처리된다.  

이때 background query는 완전히 독립적인 별도 decoder가 아니라, **같은 decoder 안에서 self-attention으로 상호작용**한다. 즉 분리는 supervision과 matching 수준에서 일어나고, 표현 학습은 완전히 단절되지 않는다. 논문은 이를 foreground/background 간 간섭을 줄이면서도 필요한 상호작용은 유지하는 설계로 설명한다.

또한 저자들은 **language-guided foreground query selection**도 추가한다. 이는 주어진 text concept에 따라 foreground query를 더 적응적으로 선택하는 방식으로, open-vocabulary foreground decoding 품질을 높이기 위한 장치다.

### 3.4 Conditioned mask decoding

box supervision만 있는 detection 데이터는 곧바로 mask supervision으로 바꿀 수 없다. 저자들은 이를 해결하기 위해, segmentation 데이터에서 **GT concept + GT box** 를 조건으로 mask를 decode하는 auxiliary task를 학습한다. 이 conditioned decoding은 detection 데이터에서 box를 입력받아 mask assistant를 생성하는 데 사용된다. 저자들이 ADE20K에서 수행한 pilot study에 따르면, GT concept와 GT box를 함께 조건으로 넣으면 mask AP가 **8.6에서 46.4로 크게 상승**했고, 이 값은 COCO 내 성능 53.2에 근접했다. 이는 box-conditioned mask decoding이 novel category와 새로운 dataset에도 꽤 잘 일반화된다는 것을 시사한다.

흥미로운 점은 이 메커니즘이 단지 training trick이 아니라, inference 시에는 **interactive segmentation interface** 로도 해석될 수 있다는 점이다. 사용자가 box 힌트를 주면, OpenSeeD가 해당 영역의 mask를 fairly high quality로 생성할 수 있어 annotation acceleration에도 잠재적 가치가 있다고 저자들은 말한다.

### 3.5 구현

구현은 Mask DINO 위에서 이뤄진다. 기본적으로 300개 latent query와 9개 decoder layer를 thing category용으로 쓰고, 여기에 stuff category용 panoptic query 100개를 추가한다. backbone은 기본적으로 pretrained Swin-T/L을 사용한다. 학습 데이터는 segmentation용으로 COCO2017, detection용으로 Objects365를 사용하며, tiny 모델은 Objects365v1, large 모델은 Objects365v2까지 사용한다. 평가 범위는 semantic / instance / panoptic segmentation과 object detection 전반이며, zero-shot segmentation/detection만 해도 60개 이상의 데이터셋에서 시험한다고 밝힌다.

## 4. Experiments and Findings

### 4.1 핵심 성과

논문 초록 수준에서 이미 결과가 강하다. OpenSeeD는 **open-vocabulary instance segmentation과 panoptic segmentation에서 5개 데이터셋에 걸쳐 기존 SOTA를 능가**했고, detection에서는 **LVIS와 ODinW** 에서 기존 방법보다 더 나은 성능을 보였다고 주장한다. 또 task-specific transfer 후에는 **COCO와 ADE20K panoptic segmentation**, **ADE20K와 Cityscapes instance segmentation** 에서 새로운 SOTA를 달성했다고 정리한다.  

### 4.2 Zero-shot 및 wild setting

논문은 OpenSeeD가 pretraining 데이터와 크게 다른 **SegInW** 류 데이터에서도 잘 작동한다고 강조한다. Figure 1 하단 설명과 본문은 instance segmentation in the wild, conditioned segmentation, zero-shot transfer 전반에서 OpenSeeD가 강한 일반화를 보인다고 요약한다. 이는 단순 COCO 내 closed-set 성능 개선이 아니라, **semantic transferability** 자체가 좋아졌다는 저자들의 주장과 연결된다.

### 4.3 Ablation

Ablation에서도 제안 모듈들의 기여가 확인된다. 저자들은 COCO closed-set panoptic segmentation에서 basic model을 ablate한 결과, detection 데이터를 결합한 joint framework가 segmentation 성능을 유의미하게 끌어올린다고 말한다. 또한 pseudo mask를 offline으로 생성해 supervision에 쓰면 open-vocabulary 모델과 closed-set MaskDINO 둘 다 성능이 상승하며, 특히 **extra mask annotation이 box AP까지 개선**할 수 있다고 보고한다.  

또 다른 ablation에서는 **online mask assistance** 와 **decoupled decoding** 을 하나씩 제거해 본다. online mask assistance를 제거하면 ADE20K open-segmentation에서 instance mask와 box 성능이 각각 **0.6 AP, 0.5 AP** 감소하고, decoupled decoding까지 제거하면 감소폭이 더 커져 **mask -1.1 AP, box -2.3 AP** 로 보고된다. 이는 두 요소가 실제로 task/data discrepancy를 줄이는 데 기여함을 보여준다.

### 4.4 해석

실험이 보여주는 핵심은, open-vocabulary segmentation과 detection을 동시에 잘하려면 단순히 CLIP text embedding을 붙이는 것보다, **태스크 간 supervision 차이 자체를 모델링**해야 한다는 점이다. shared semantic space만으로는 충분하지 않고, foreground/background의 구조적 차이와 box-mask granularity 차이를 각각 따로 해결해 줘야 한다는 것이 OpenSeeD의 실험적 메시지다.  

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **문제 정의가 정확하고, 해결책이 단순하면서도 구조적**이라는 점이다. detection과 segmentation을 모두 open-vocabulary로 하려면 무엇이 어려운지, 즉 **semantic vocabulary gap**, **foreground/background task gap**, **box/mask annotation gap** 을 명시적으로 분리해 다룬다. 그리고 각각에 대해 shared text encoder, decoupled decoding, conditioned mask decoding이라는 대응책을 제시한다.

또 하나의 강점은 **joint learning** 자체다. 기존 방식이 detection pretrain 후 segmentation finetune으로 끝났다면, OpenSeeD는 애초에 하나의 모델이 양쪽 태스크를 함께 수행하도록 훈련된다. 이 점에서 이 논문은 단순 성능 개선보다도, **단일 open-world visual model** 방향의 초기 강한 baseline이라는 의미가 있다.

### Limitations

저자들이 직접 밝힌 한계도 있다. OpenSeeD는 detection과 segmentation joint training의 가능성을 보여주는 데 초점을 맞췄기 때문에, **referring/grounding data** 나 **large-scale image-text pair** 를 추가로 사용하지 않았다. 즉 semantic coverage를 더 넓힐 수 있는 외부 데이터원을 아직 적극 활용하지 않았고, 더 큰 joint training은 future work로 남겨 둔다.

또 비판적으로 보면, conditioned mask decoding은 detection 데이터를 segmentation에 활용하게 해주는 좋은 아이디어지만, 여전히 **mask pseudo-supervision의 질** 에 성능이 좌우될 수 있다. 또한 box-conditioned mask는 segmentation을 위한 강한 힌트를 받는 설정이므로, 완전히 box-free open segmentation과는 약간 다른 문제를 푼다고도 볼 수 있다.

### Interpretation

비판적으로 해석하면, OpenSeeD의 진짜 공헌은 “새로운 최고 backbone”이 아니다. 더 중요한 것은, **open-vocabulary detection과 segmentation을 하나의 모델로 묶으려면 무엇을 공유하고 무엇을 분리해야 하는지**를 보여준 데 있다. shared semantic space는 공유해야 할 것, foreground/background decoding과 conditioned mask decoding은 분리하거나 보완해야 할 것에 해당한다. 이런 설계 원리는 이후 unified segmentation-detection 모델에도 꽤 일반적인 통찰을 제공한다.

## 6. Conclusion

이 논문은 OpenSeeD를 통해, 서로 다른 supervision granularity를 가진 detection과 segmentation 데이터를 **하나의 open-vocabulary model** 안에서 공동 학습할 수 있음을 보여준다. 핵심은 단순한 encoder-decoder 프레임 위에

* **shared semantic space**
* **decoupled foreground/background decoding**
* **conditioned mask decoding**

을 얹어, semantic gap과 annotation gap을 동시에 줄였다는 점이다. 그 결과 OpenSeeD는 zero-shot open-vocabulary segmentation에서 강한 성능을 보이고, detection 성능도 합리적으로 유지하며, downstream fine-tuning에서도 closed-vocabulary 성능까지 개선한다. 저자들 스스로 말하듯, 이 모델은 detection과 segmentation을 모두 아우르는 **single open-world model** 을 향한 첫 강한 baseline으로 보는 것이 가장 적절하다.  
