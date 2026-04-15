# A Simple Baseline for Open-Vocabulary Semantic Segmentation with Pre-trained Vision-language Model

- **저자**: Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, Xiang Bai
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2112.14757

## 1. 논문 개요

이 논문은 **open-vocabulary semantic segmentation** 문제를 다룬다. 즉, 학습 시에 픽셀 단위 주석을 보지 못한 클래스까지도 테스트 시 분할할 수 있는 semantic segmentation 모델을 만드는 것이 목표다. 저자들은 이를 위해 대규모 이미지-텍스트 데이터로 사전학습된 vision-language model인 **CLIP**을 활용한다.

연구 문제의 핵심은, CLIP은 본래 **image-level classification**에 강한 모델인데 semantic segmentation은 **pixel-level prediction**을 요구한다는 점이다. 다시 말해, CLIP이 잘하는 시각-언어 정렬은 이미지 전체 수준의 의미 이해에 가깝고, semantic segmentation은 각 픽셀마다 클래스를 정해야 하므로 처리 granularity가 다르다. 저자들은 이 불일치가 기존의 FCN 기반 접근을 CLIP과 단순 결합했을 때 성능이 잘 나오지 않는 주요 이유라고 본다.

이 문제는 중요하다. 기존 supervised semantic segmentation은 픽셀 단위 라벨링 비용이 매우 크기 때문에 보통 수십에서 수백 개 정도의 클래스만 다룬다. 반면 이미지 분류는 ImageNet 같은 대규모 데이터셋 덕분에 훨씬 더 넓은 vocabulary를 다룰 수 있다. 따라서 open-vocabulary segmentation이 가능해지면, 적은 segmentation annotation만으로도 훨씬 풍부한 객체와 장면 개념을 다룰 수 있게 된다.

이 논문의 중심 주장 하나로 요약하면 다음과 같다. **CLIP을 pixel classifier처럼 직접 쓰려 하지 말고, 먼저 class-agnostic mask proposal을 만들고 그 proposal 단위로 CLIP이 분류하게 하자.** 저자들은 이 2-stage 설계가 CLIP의 강점을 더 자연스럽게 활용한다고 본다.

## 2. 핵심 아이디어

논문의 핵심 아이디어는 semantic segmentation을 한 번에 픽셀 분류로 풀지 않고, 다음 두 단계로 분해하는 것이다.

첫째, 입력 이미지에서 **클래스 비의존적(class-agnostic) mask proposal**들을 생성한다. 이 단계는 “어디가 하나의 의미 있는 영역인가”를 찾는 문제다.  
둘째, 각 proposal을 잘라낸 뒤 CLIP으로 **그 영역이 어떤 클래스인지 분류**한다. 이 단계는 CLIP이 원래 잘하는 image-level recognition과 더 가깝다.

이 설계의 직관은 명확하다. unseen class라도 “경계가 맞는 물체/영역”을 proposal로 뽑는 일은 어느 정도 일반화될 수 있고, proposal 단위의 잘린 이미지는 CLIP이 원래 학습한 입력 형태와 더 비슷하다. 반대로 FCN은 각 픽셀 feature 위에 classifier를 얹는 구조이기 때문에, 이미지 전체 수준에서 학습된 CLIP 표현과 잘 맞지 않는다.

기존 접근과의 차별점은 크게 두 가지다. 첫째, open-vocabulary segmentation을 위해 CLIP을 쓰면서도 **FCN 대신 2-stage proposal-based framework**를 택했다는 점이다. 둘째, proposal 분류에서도 단일 전략이 아니라, **학습된 vision encoder와 frozen CLIP vision encoder를 함께 사용하고 ensemble**하는 방식을 제안했다. 저자들의 실험에 따르면 이 둘은 seen/unseen 클래스에서 서로 보완적이다.

또한 텍스트 프롬프트 설계도 핵심 요소다. CLIP의 text encoder에 클래스명을 넣을 때, 단순 hand-crafted prompt뿐 아니라 **learnable prompt**를 도입해 unseen class에 대해서도 더 나은 일반화를 유도한다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다.

입력 이미지가 들어오면 먼저 proposal generator가 여러 개의 binary mask를 예측한다. 그 다음 각 mask에 대해 foreground 영역을 crop하여 CLIP 기반 region classifier가 클래스 확률을 계산한다. 마지막으로 서로 겹칠 수 있는 여러 mask prediction을 픽셀 단위로 합쳐 최종 segmentation map을 만든다.

### 3.1 Mask proposal generation

저자들은 proposal 생성 방법으로 세 가지를 검토한다.

- **GPB-UCM**
- **Selective Search**
- **MaskFormer**

이 중 기본값은 **MaskFormer**다. 이유는 MaskFormer가 semantic segmentation을 “segment 생성 + segment 분류”로 분리하는 구조이기 때문에, 이 논문의 2-stage 철학과 잘 맞고, seen class만으로 학습해도 unseen class에 대해 proposal 품질이 잘 일반화된다고 관찰했기 때문이다.

중요한 점은 이 단계에서 proposal이 특정 클래스 이름에 의존하지 않는다는 것이다. 즉, “이 영역이 고양이냐 자동차냐”를 묻는 것이 아니라 “의미 있는 분할 단위인가”를 먼저 찾는다. 이 점이 unseen class 일반화에 유리하다는 것이 저자들의 주장이다.

### 3.2 Region classification via CLIP

proposal이 만들어지면, 각 proposal을 CLIP으로 분류한다. 여기서 저자들은 두 가지 전략을 사용한다.

첫 번째 전략은 **pre-trained CLIP vision encoder를 직접 사용하는 방식**이다. proposal mask를 threshold 0.5로 이진화한 뒤, 이미지에서 foreground만 남기고 background는 지운 후 crop한다. 이 masked image crop을 $224 \times 224$로 resize하여 CLIP에 넣는다. 이 방식은 unseen class 분류에는 강하지만, seen class annotation을 직접 활용하지 않기 때문에 seen class 성능이 약하다.

두 번째 전략은 **retrained vision encoder** 방식이다. 하지만 단순히 seen class용 classifier를 새로 학습하면 unseen class로 일반화되지 못한다. 그래서 저자들은 CLIP의 text encoder가 만든 class embedding을 **고정 classifier weight**로 사용한다. 즉, 이미지 encoder가 최종적으로 CLIP text embedding space에 맞는 feature를 내도록 학습한다. 이렇게 하면 seen class annotation으로 supervised training을 하면서도, classifier 자체는 언어 기반 embedding에 묶여 있어 unseen class로의 일반화 가능성을 남긴다. 구현상으로는 MaskFormer의 classifier weight를 CLIP text feature로 대체하는 식으로 통합할 수 있다.

이 두 전략은 서로 장단점이 다르다. 논문 실험에서 retrained vision encoder는 seen class에 강하고 unseen class에는 약했으며, 반대로 frozen CLIP vision encoder는 unseen class에 강하고 seen class에는 약했다. 그래서 기본 설정은 **두 결과를 ensemble**하는 것이다.

proposal $M^p$에 대해 foreground crop $A_{fg} = crop(M^p, I)$를 만든 뒤, 클래스 $C_i$에 대한 CLIP 기반 분류 확률은 다음과 같이 계산한다.

$$
C_i(A_{fg}) =
\frac{
\exp(\cosine(E_{vision}(A_{fg}), E_{text}(C_i))/\tau)
}{
\sum_i \exp(\cosine(E_{vision}(A_{fg}), E_{text}(C_i))/\tau)
}
$$

여기서 $E_{vision}$은 vision encoder, $E_{text}$는 text encoder, $C_i$는 i번째 클래스 이름, $\tau$는 temperature이며 논문에서는 $\tau = 100$을 사용한다. 핵심은 이미지 crop의 임베딩과 클래스 텍스트 임베딩 사이의 cosine similarity로 분류를 수행한다는 점이다.

### 3.3 Prompt design

CLIP의 성능은 prompt에 민감할 수 있으므로, 저자들은 텍스트 prompt 설계도 실험한다.

하나는 **hand-crafted prompt**다. CLIP이 원래 ImageNet classification에서 사용한 80개의 프롬프트 후보 중에서 semantic segmentation에 가장 잘 맞는 것을 training data로 골라 사용한다.

다른 하나는 **learning-based prompt**다. 일반화된 prompt를 $[P]_0 \dots [P]_m [CLS]$ 형태로 두고, 여기서 $[P]$ 토큰들을 학습 가능한 파라미터로 둔다. 즉, 클래스 이름 앞에 붙는 문맥 토큰을 seen class 데이터로 학습하여 더 좋은 text embedding을 얻도록 한다. 논문 결과에 따르면 learnable prompt가 manual prompt보다 훨씬 좋았고, 흥미롭게도 unseen class에도 잘 일반화되었다.

### 3.4 Mask prediction assembly

proposal들은 서로 겹칠 수 있으므로, 최종 픽셀 라벨을 만들기 위해 aggregation이 필요하다. 픽셀 $q$가 클래스 $i$일 확률은 다음과 같이 정의된다.

$$
C_i(q) =
\frac{
\sum_k M_k^p(q) C_k^p(i)
}{
\sum_k M_k^p(q)
}
$$

여기서 $M_k^p(q)$는 k번째 proposal이 픽셀 $q$를 포함할 확률이고, $C_k^p(i)$는 그 proposal이 클래스 $i$일 확률이다. 직관적으로 말하면, 어떤 픽셀을 덮는 여러 proposal들의 클래스 확률을 mask confidence로 가중 평균하는 방식이다. 논문은 이 값들의 클래스 합이 반드시 1이 되지는 않는다고 명시하며, 최종적으로는 가장 높은 값을 갖는 클래스를 픽셀 라벨로 선택한다.

### 3.5 FCN baseline과의 비교

논문은 자신들의 2-stage 방법뿐 아니라, CLIP을 FCN에 붙인 비교 baseline도 구성한다. 여기에도 두 전략이 있다.

하나는 CLIP vision encoder가 낸 feature map을 바로 픽셀 분류에 쓰는 방식이다. 하지만 CLIP은 본래 $[CLS]$ 토큰 중심의 image-level representation으로 학습되었고, pre-training 해상도도 $224 \times 224$라 고해상도 segmentation 입력과 맞지 않는다. 이를 줄이기 위해 sliding window inference를 사용한다.

다른 하나는 FCN encoder를 seen class 데이터로 다시 학습하되, classifier weight는 CLIP text embedding으로 고정하는 방식이다.

하지만 저자들의 결론은 명확하다. 이런 보정에도 불구하고 FCN은 CLIP과 잘 맞지 않으며, proposal-based 2-stage 방식이 훨씬 적합하다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 설정

논문은 다섯 개의 데이터셋에서 실험한다.

- **COCO Stuff**: 171 classes, train 117k, val 5k
- **Pascal VOC 2012**: 20 classes, augmented annotation 사용
- **Cityscapes**: urban scene parsing
- **Pascal Context**: frequent 59 classes 사용
- **ADE20K**: 150-class와 847-class 설정 모두 평가

평가 설정은 두 가지다.

첫째는 **cross-dataset setting**이다. COCO Stuff로 학습하고 다른 데이터셋으로 fine-tuning 없이 바로 평가한다. 이는 unseen class뿐 아니라 domain gap도 다뤄야 하므로 더 어렵다.

둘째는 **zero-shot setting**이다. 하나의 데이터셋 내부에서 seen/unseen class split을 만들고, seen class만으로 학습한 뒤 전체 클래스를 평가한다. COCO Stuff는 156 seen / 15 unseen, Pascal VOC는 15 seen / 5 unseen split을 따른다.

평가 지표는 cross-dataset에서는 **mIoU**, zero-shot에서는 **hIoU**를 주 지표로 쓴다. hIoU는 seen class 성능과 unseen class 성능의 조화평균으로 정의된다.

$$
hIoU =
\frac{
2 \cdot mIoU_{seen} \cdot mIoU_{unseen}
}{
mIoU_{seen} + mIoU_{unseen}
}
$$

이 지표를 쓰는 이유는 seen만 잘하고 unseen이 매우 약한 모델을 penalize하기 위해서다.

### 4.2 구현 세부사항

기본 segmentation backbone은 **MaskFormer + ResNet-101**이고, 기본 CLIP 모델은 **ViT-B/16**이다. 학습은 8개의 Nvidia V100 GPU에서 수행했다. COCO Stuff 기준으로 MaskFormer는 AdamW, 초기 learning rate $10^{-4}$, weight decay $10^{-4}$, poly LR policy를 사용한다. zero-shot setting에서는 60K, cross-dataset setting에서는 120K iteration 학습했다. proposal 수는 train/test 모두 100개를 사용한다.

prompt tuning은 SGD로 수행하고, learning rate는 0.02, cosine decay를 사용한다. Pascal VOC와 COCO Stuff에서 각각 50, 100 epochs 동안 prompt를 학습한다.

### 4.3 Cross-dataset 결과

COCO Stuff로만 학습하고 다른 데이터셋에 바로 평가한 결과, 2-stage 방식은 모든 벤치마크에서 FCN보다 좋았다.

- Cityscapes: FCN 21.4, Ours 34.5
- Pascal Context: FCN 28.2, Ours 47.7
- ADE20K-150: FCN 14.9, Ours 20.5
- ADE20K-847: FCN 4.1, Ours 7.0

논문 본문에서도 이를 각각 **+13.1, +19.6, +5.6, +2.9 mIoU** 향상으로 정리한다. 이는 CLIP의 image-level recognition 능력을 proposal classification으로 연결하는 방식이 dataset shift가 있는 상황에서도 더 견고함을 보여준다.

Pascal VOC는 COCO Stuff와 클래스 중복이 많아서 मुख्य 테이블에서는 제외했지만, 논문은 88.4 mIoU를 달성했다고 언급한다.

### 4.4 Zero-shot 결과

#### COCO Stuff

기존 방법과 비교했을 때 큰 폭의 향상이 나타난다.

- SPNet: hIoU 16.8
- ZS3: hIoU 15.0
- CaGNet: hIoU 18.2
- FCN baseline: hIoU 20.9
- **Ours: hIoU 37.8**

self-training 없이도 이전 최고 성능 대비 hIoU가 크게 오른다. 논문은 CaGNet 대비 **+19.5 hIoU**, unseen mIoU 기준 **+24.1** 향상이라고 정리한다.

self-training을 추가하면:

- STRICT: hIoU 32.6
- **Ours+ST: hIoU 41.5**

즉 STRICT 대비 **+8.9 hIoU** 향상이다.

#### Pascal VOC 2012

Pascal VOC에서는 격차가 더 크다.

- CaGNet: hIoU 39.7
- FCN baseline: hIoU 50.7
- **Ours: hIoU 77.5**

self-training 포함 시:

- STRICT: hIoU 49.8
- **Ours+ST: hIoU 79.3**

논문은 self-training 없는 경우 이전 최고 대비 **+37.8 hIoU**, self-training 포함 비교에서는 **+29.5 hIoU**라고 강조한다. unseen mIoU도 매우 높아서, 단순히 seen class 성능만 좋은 것이 아니라 unseen class 인식 자체가 강해졌음을 보여준다.

### 4.5 Ablation study

#### Proposal generator의 영향

proposal generator만 바꾸어 실험했을 때:

- GPB-UCM: hIoU 10.9
- Selective Search: hIoU 11.0
- MaskFormer: hIoU 28.2

즉, proposal quality가 성능에 매우 중요하며, 그중 MaskFormer가 가장 적합했다. 다만 고전적 proposal method들도 unseen mIoU 기준으로는 당시 SOTA와 비슷한 수준을 보일 정도로, “proposal 후 CLIP classification”이라는 기본 아이디어 자체도 상당히 강력함을 시사한다.

#### Proposal generator의 일반화 능력

저자들은 oracle 방식으로 proposal 품질 자체를 평가했다. 예를 들어 COCO Stuff에서 학습한 MaskFormer proposal generator를 ADE20K에 적용했을 때도 mIoU 64.4를 기록했다. 반대로 ADE20K에서 학습한 proposal generator를 COCO Stuff에 적용했을 때도 mIoU 62.5를 얻었다. 데이터셋 내 학습/평가 성능보다는 낮지만, cross-dataset에서도 proposal 생성이 상당히 일반화된다는 점이 중요하다.

#### CLIP 활용 전략의 보완성

region classification 방식 비교 결과는 매우 인상적이다.

- Retrained vision encoder: hIoU 8.7, seen mIoU 38.7, unseen mIoU 4.9
- CLIP vision encoder: hIoU 28.2, seen mIoU 26.8, unseen mIoU 29.7
- Ensemble: hIoU 37.8, seen mIoU 39.3, unseen mIoU 36.3

즉, retrained encoder는 seen class에 강하지만 unseen은 매우 약하고, frozen CLIP은 그 반대 성격을 보인다. ensemble이 이 둘을 결합해 가장 좋은 균형을 만든다.

#### CLIP variant의 영향

manual prompt 기준으로 비교했을 때:

- ResNet-50: hIoU 15.2
- ResNet-101: hIoU 13.8
- ViT-B/32: hIoU 15.3
- ViT-B/16: hIoU 18.3

기본적으로 모두 어느 정도 작동하지만, **ViT-B/16**이 가장 좋았다.

#### Prompt learning의 효과

manual prompt와 learnable prompt를 비교하면:

- Manual: hIoU 18.3
- Learnable: hIoU 28.2

무려 **+9.9 hIoU** 향상이다. seen/unseen 모두 비슷한 폭으로 좋아졌다는 점에서, prompt learning이 단순한 overfitting이 아니라 CLIP text side를 segmentation task에 더 잘 맞게 조정한 것으로 볼 수 있다.

#### FCN과 MaskFormer, ImageNet과 CLIP pre-training의 영향

Table 15에서 재구성한 결과는 논문의 메시지를 더 선명하게 보여준다.

- SPNet, ResNet-101, ImageNet: hIoU 25.1
- FCN, ViT-B/16, CLIP-VL: hIoU 50.7
- MaskFormer, ResNet-101, ImageNet: hIoU 49.5
- MaskFormer, ResNet-101, CLIP-VL: hIoU 74.2
- MaskFormer, ViT-B/16, CLIP-VL: hIoU 77.5

여기서 알 수 있는 것은 세 가지다.  
첫째, **MaskFormer 구조 자체가 FCN보다 유리**하다.  
둘째, **CLIP pre-training이 ImageNet pre-training보다 훨씬 강력**하다.  
셋째, 이 논문의 성능 향상은 단순히 “더 큰 사전학습 데이터” 때문만이 아니라, **2-stage 구조 선택** 때문이기도 하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법 설계가 매우 일관적이라는 점이다. 저자들은 CLIP과 semantic segmentation의 granularity mismatch를 정확히 짚고, 그 문제를 해결하기 위한 구조적 선택으로 2-stage proposal-based pipeline을 제시한다. 단순히 CLIP feature를 segmentation head에 붙이는 식이 아니라, CLIP이 본래 강한 입력 단위인 region crop 수준으로 문제를 재구성했다는 점이 설득력 있다.

또 다른 강점은 실험적 증거가 풍부하다는 점이다. cross-dataset generalization, zero-shot benchmark, proposal generator 비교, CLIP variant 비교, prompt ablation, supervised baseline 비교까지 폭넓게 제시한다. 특히 seen/unseen 간 trade-off, things/stuff 편향, proposal generalization 등을 별도 실험으로 보여 주어 주장의 근거가 비교적 탄탄하다.

실용성도 강점이다. 논문 제목처럼 “simple baseline”에 가깝다. 복잡한 생성 모델이나 추가적인 semantic synthesis 없이, proposal generator와 CLIP을 결합하는 비교적 단순한 구조로 강한 결과를 얻는다. 연구 커뮤니티 입장에서는 후속 연구의 출발점으로 쓰기 좋다.

한계도 분명하다. 첫째, 최종 성능은 여전히 **proposal quality**에 크게 의존한다. proposal이 잘못 생성되면 뒤의 CLIP 분류가 아무리 좋아도 복구가 어렵다. 둘째, supervised baseline과 비교하면 여전히 격차가 존재한다. 예를 들어 COCO Stuff에서 supervised MaskFormer는 unseen mIoU 62.6인데, 본 방법은 self-training 포함해도 43.6이다. 즉, open-vocabulary 설정에서 매우 강력하긴 하지만 fully supervised 수준은 아니다.

셋째, 논문은 CLIP의 **things/stuff 편향** 가능성을 시사한다. COCO Stuff unseen class에서 things와 stuff 간 성능 차이가 supervised baseline보다 더 컸고, self-training이 그 격차를 줄였다. 이는 CLIP pre-training 데이터의 시각적 편향이 segmentation 성능에도 반영될 수 있음을 보여준다.

넷째, self-training을 사용한 결과도 보고하지만, 제공된 본문 텍스트에는 self-training 절차의 상세 구현이 충분히 설명되어 있지 않다. 따라서 self-training에 의해 얼마나, 어떤 메커니즘으로 개선되는지는 이 텍스트만으로 완전히 파악하기 어렵다. 같은 맥락에서 proposal crop 세부 구현도 “Appendix 참고”라고 되어 있으나, 일부만 제시되어 있어 모든 구현 디테일을 복원할 수는 없다.

비판적으로 보면, 이 논문은 “CLIP을 semantic segmentation에 직접 맞추는 것”보다는 “CLIP이 잘 작동하도록 segmentation 문제를 바꾸는 것”에 가깝다. 이것은 장점이자 한계다. 매우 효과적이지만, 진정한 pixel-text alignment를 학습하는 방향과는 다소 다르다. 따라서 더 세밀한 경계, 복잡한 겹침, 작은 stuff region 등에서는 region-level classification 기반 설계가 구조적 제약을 가질 수 있다.

## 6. 결론

이 논문은 open-vocabulary semantic segmentation에서 CLIP을 효과적으로 활용하기 위해, 문제를 **mask proposal generation + proposal classification**의 2-stage 구조로 재정의했다. 핵심 기여는 CLIP과 segmentation 사이의 granularity mismatch를 정확히 인식하고, FCN보다 proposal-based 방식이 더 적합함을 실험적으로 강하게 입증한 점이다.

정량적으로도 기여가 크다. Pascal VOC 2012와 COCO Stuff의 zero-shot setting에서 기존 방법을 큰 폭으로 넘어섰고, cross-dataset setting에서도 FCN 기반 접근보다 일관되게 우수했다. 또한 prompt learning, ensemble classification, proposal generalization 같은 실험을 통해 이 baseline이 단순하면서도 견고하다는 점을 보였다.

향후 연구 관점에서 이 논문은 중요한 기준점 역할을 한다. 이후 연구는 이 baseline 위에서 proposal 품질 개선, region-text alignment 강화, self-training 안정화, things/stuff 편향 완화, end-to-end open-vocabulary segmentation 구조 설계 등으로 확장될 수 있다. 실제 응용 측면에서도, 모든 클래스에 대해 pixel annotation을 모으기 어려운 환경에서 대규모 vision-language pre-training의 힘을 segmentation으로 옮기는 실용적 방향을 제시했다.
