# Amodal Instance Segmentation

- **저자**: Ke Li, Jitendra Malik
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1604.08202

## 1. 논문 개요

이 논문은 **amodal instance segmentation** 문제를 다룬다. 이는 물체의 **현재 보이는 부분만이 아니라, 가려진 부분까지 포함한 전체 영역**을 예측하는 과제이다. 기존의 **modal instance segmentation**이 visible region만 맞히는 문제였다면, 이 논문은 한 단계 더 나아가 occlusion 뒤에 숨겨진 영역까지 복원하려고 한다.

연구 문제는 분명하다. 실제 장면에서는 물체가 다른 물체에 자주 가려지며, 이때 단순한 visible mask만으로는 물체의 전체 크기, 실제 경계, 가림 관계, 상대적 depth ordering 등을 충분히 추론하기 어렵다. 예를 들어 말을 일부만 본 상황에서 전체 몸체를 추정할 수 있어야 실제 크기 추정이나 물체 간 가림 관계 분석이 가능하다. 따라서 amodal segmentation은 단순한 segmentation을 넘어 **occlusion reasoning의 핵심 표현**으로 작동한다.

이 문제가 중요한 이유는, amodal mask가 있으면 modal mask와의 차이를 통해 **어디가 가려졌는지**, **얼마나 가려졌는지**, **누가 누구를 가렸는지**를 추론할 수 있기 때문이다. 논문은 이런 문제들이 사실상 amodal instance segmentation으로 환원될 수 있다고 본다. 다만 당시에는 public amodal annotation이 없어서 supervised learning이 어려웠고, 이 점이 기존 연구를 막고 있었다. 이 논문의 핵심 기여는 바로 그 제약을 우회하는 데 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 간단하지만 강력하다. **실제 occlusion을 제거해서 amodal mask를 복원하는 것은 어렵지만, 반대로 원래 보이는 물체 위에 synthetic occluder를 덧씌워 가림을 인위적으로 만드는 것은 쉽다**는 관찰이다. 즉, amodal ground truth가 없는 문제를 정면으로 풀지 않고, 기존 modal segmentation annotation만으로 학습 가능한 synthetic training setup으로 바꾼다.

구체적으로는, 원래 객체 마스크를 가진 이미지 패치를 뽑고, 다른 이미지에서 가져온 객체를 위에 겹쳐서 main object를 일부 가리게 만든다. 이때 composite image에서는 객체가 가려졌지만, 원래 객체의 segmentation mask는 그대로 남아 있으므로 이것이 곧 synthetic한 의미의 “정답 amodal mask” 역할을 한다. 이렇게 하면 공개된 amodal dataset 없이도 amodal predictor를 학습시킬 수 있다.

기존 접근과의 차별점은 두 가지다. 첫째, 저자 주장에 따르면 이것은 **general-purpose amodal segmentation의 첫 번째 방법**이다. 둘째, 테스트 시 amodal bounding box가 주어진다고 가정하지 않고, segmentation heatmap을 바탕으로 bounding box를 점진적으로 확장하는 **Iterative Bounding Box Expansion**을 제안한다. 즉, mask prediction과 box expansion을 연결한 점이 실용적 차별점이다.

## 3. 상세 방법 설명

전체 시스템은 크게 두 부분으로 구성된다. 첫 번째는 **synthetic training data generation**, 두 번째는 **test-time amodal mask and bounding box prediction**이다.

### 3.1 학습 데이터 생성

학습에는 PASCAL VOC 2012 `train`의 SBD annotation을 사용한다. 각 training example은 다음 요소를 가진다.

- image patch
- modal bounding box
- target amodal segmentation mask

절차는 다음과 같다.

먼저 한 이미지를 고르고, 그 안의 객체 하나를 **main object**로 선택한다. 그 다음 main object를 충분히 포함하는 random crop box를 샘플링한다. 이후 다른 이미지에서 object instance를 가져와 main object 위에 랜덤 위치와 크기로 overlay한다. overlay할 객체 수는 0개에서 2개 사이에서 샘플링된다. 이 과정을 통해 partial occlusion이 있는 composite patch를 만든다.

이후 composite patch에서 **main object의 아직 보이는 부분만 감싸는 bounding box**를 찾는다. 이것이 composite image에서의 ground-truth modal box 역할을 한다. 테스트 상황의 localization noise를 흉내 내기 위해 이 box에 jitter를 추가한다.

그 다음 target mask를 만든다. 핵심은 모든 픽셀을 3가지로 나누는 것이다.

- main object에 속하는 픽셀: positive
- background에 속하는 픽셀: negative
- 다른 object에 속하는 픽셀: unknown

이 설계는 매우 중요하다. 가려진 물체 뒤에 있는 픽셀이 배경일 수는 없지만, 다른 객체에 의해 가려질 수는 있으므로 다른 object 영역을 무조건 negative로 두면 잘못된 supervision이 된다. 따라서 저자는 occluder에 해당하는 부분을 **unknown label**로 두고, known label에 대해서만 loss를 계산한다.

논문은 학습 샘플 생성 조건도 구체적으로 제시한다. crop box는 main object bounding box와 각 축에서 최소 70% 이상 겹쳐야 하고, crop 크기는 원래 object box 길이의 70%~200% 범위로 선택된다. overlay 후 main object의 visible proportion이 30% 미만이 되면 그 연산을 취소하고 다시 샘플링한다. 이후 visible part를 감싸는 modal box를 구하고, 여기에 각 축에서 최소 75% 이상 겹치며 크기 차이가 최대 10%인 jittered box를 다시 샘플링한다.

### 3.2 네트워크 입력과 출력

모델은 다음 입력을 받는다.

- image patch
- modal segmentation heatmap
- object category

출력은 **amodal segmentation heatmap**이다.

네트워크 아키텍처는 IIS(Iterative Instance Segmentation) [Li et al., 2016]와 동일한 구조를 사용한다. 기반은 Hariharan et al.의 **hypercolumn architecture**이며, VGG-16 기반 “O-Net”을 따른다. hypercolumn은 여러 intermediate layer의 upsampled feature map을 합쳐서 low-level detail과 high-level semantic 정보를 동시에 활용하는 구조다.

IIS는 여기에 추가로 초기 heatmap hypothesis를 입력 채널로 넣어 heatmap refinement를 반복하는 구조다. 이 논문에서는 modal segmentation predictor인 IIS가 먼저 modal heatmap을 만들고, 이를 amodal predictor의 입력으로 사용한다. 보이는 윤곽을 먼저 잘 잡은 뒤, 그 바깥을 확장하는 방식이다.

### 3.3 학습 절차와 손실 함수

각 training example에서 modal bounding box 내부 patch를 잘라서 비등방적으로 `224 × 224`로 resize하여 입력으로 사용한다. modal segmentation heatmap도 원래 patch 좌표계에 맞춘 뒤 bilinear interpolation으로 `224 × 224`로 맞춘다.

중요한 점은, 모델이 본 영역보다 더 넓은 amodal 영역을 예측해야 하므로 입력 patch에서 각 변의 10%를 제거한 뒤 다시 `224 × 224`로 rescale한다는 것이다. 즉, 네트워크는 “조금 잘린 시야”를 바탕으로 더 넓은 mask를 예측하도록 훈련된다. 만약 새 patch에서 visible object pixel 비율이 10% 미만이면 샘플을 버리고 다시 생성한다.

최적화는 mini-batch SGD with momentum으로 수행한다. 초기 가중치는 IIS 모델에서 가져온다. 손실 함수는 다음과 같이 설명된다.

$$
\mathcal{L} = \sum_{i \in \text{known pixels}} - \log p(y_i \mid x_i)
$$

즉, **known ground-truth label이 있는 픽셀들에 대해서만 pixel-wise negative log likelihood를 합산**한다. unknown 픽셀은 loss에서 제외된다. 또한 patch가 얼마나 확대되었는지에 따라 inverse 비율의 instance-specific weight를 둔다고 설명한다.

학습 하이퍼파라미터는 다음과 같다.

- mini-batch size: 32
- learning rate: $10^{-5}$
- weight decay: $10^{-3}$
- momentum: $0.9$
- iteration: 50,000

### 3.4 테스트 시 추론: Iterative Bounding Box Expansion

테스트 시 입력으로 주어지는 것은 object detector가 제공하는

- modal bounding box
- object category

이다. 논문은 detector 예시로 R-CNN, Fast R-CNN, Faster R-CNN을 언급한다. modal segmentation heatmap은 IIS로 계산한다.

초기에는 **amodal bounding box를 modal bounding box와 동일하게 설정**한다. 이후 반복적으로 다음을 수행한다.

1. 현재 amodal bounding box 내부 patch를 CNN에 입력한다.
2. CNN은 기존 box 바깥까지 포함하는 더 넓은 영역에 대한 amodal heatmap을 예측한다.
3. 기존 bounding box의 위, 아래, 왼쪽, 오른쪽 바깥 영역에서 평균 heat intensity를 계산한다.
4. 특정 방향의 평균 intensity가 threshold보다 크면 그 방향으로 bounding box를 확장한다.

논문에서 이 threshold는 실험적으로 `0.1`로 설정되었다. 모든 방향의 평균 intensity가 threshold 이하가 될 때까지 반복한다.

최종 mask는 thresholding으로 얻는다.

- amodal mask: heatmap > `0.7`
- modal mask: heatmap > `0.8`

이 절차는 “숨겨진 물체가 box 바깥으로 더 이어질 가능성”을 heatmap intensity로 판단해 box를 점점 키우는 방식이다. 이는 amodal box를 따로 regression하지 않고 segmentation prediction을 활용해 점진적으로 결정한다는 점에서 직관적이다.

## 4. 실험 및 결과

### 4.1 실험 설정

공개된 amodal segmentation dataset이 없기 때문에 평가가 쉽지 않다. 저자들은 세 가지 방식으로 실험한다.

- qualitative evaluation
- indirect quantitative evaluation
- 직접 구축한 100개 occluded object에 대한 direct evaluation

정성 실험은 PASCAL VOC 2012 `val`에서 수행하며, segmentation 자체를 보기 위해 modal box와 category는 ground truth를 사용한다.

### 4.2 정성적 결과

논문은 occlusion을 두 종류로 나눈다.

- **interior occlusion**: occluder가 대체로 object 내부에 들어와 hole처럼 보이는 경우
- **exterior occlusion**: object 바깥으로 이어져야 하는 부분이 가려진 경우

interior occlusion은 “빈 구멍을 메울지 말지”가 핵심이므로 상대적으로 ambiguity가 적다. 반면 exterior occlusion은 visible part를 어느 방향으로 얼마나 확장할지가 문제라 더 어렵다. 이 경우 모델은 category-level shape prior를 활용해야 한다.

저자들은 qualitative result에서 제안 방법이 interior와 exterior occlusion 모두에 대해 그럴듯한 amodal mask를 예측한다고 보고한다. 특히 modal prediction이 좋지 않은 경우에도 amodal prediction이 더 나은 예가 있다고 설명한다. 반대로 실패 사례에서는 unusual pose의 희소성, occluded part의 높은 ambiguity, 인접 객체와의 appearance similarity, modal prediction 오류 등이 원인으로 제시된다.

또한 unoccluded object에 대해서는 amodal mask가 modal mask와 같아야 하는데, 실제로 그와 비슷하거나 오히려 더 안정적인 결과를 보인다고 한다. 저자들은 이를 occlusion robustness를 배우는 과정에서 저수준 패턴 변화에도 강건해졌기 때문이라고 해석한다.

### 4.3 간접 평가: occlusion presence prediction

직접 ground truth amodal mask가 없으므로, 저자들은 modal/amodal mask의 차이를 이용해 물체가 occluded인지 아닌지를 예측하는 간접 평가를 수행한다. 사용한 지표는 다음 **area ratio**이다.

$$
\text{area ratio} =
\frac{\text{area}(\text{modal mask} \cap \text{amodal mask})}
{\text{area}(\text{amodal mask})}
$$

직관적으로, 물체가 가려지지 않았다면 modal mask와 amodal mask가 거의 같아야 하므로 이 비율은 1에 가까워야 한다. 반대로 heavily occluded라면 amodal mask 중 modal mask 바깥의 비율이 커지므로 area ratio는 작아진다.

PASCAL VOC 2012 `val`의 occlusion presence annotation과 비교한 결과, unoccluded object는 높은 area ratio 쪽에 치우치고, occluded object는 대략 `0.75` 부근에서 peak를 보였다. 이 ratio를 thresholding하여 “비가림 여부”를 분류했을 때 average precision은 **77.17%**였다.

이 결과는 제안 방법이 무조건 물체를 부풀려 hallucination하는 것이 아니라, 실제로 occlusion이 있는 경우에만 modal보다 더 큰 amodal mask를 예측하는 경향이 있음을 뒷받침한다.

### 4.4 직접 평가: 100개 수작업 amodal annotation

저자들은 각 카테고리마다 5개씩, 총 100개의 occluded object를 PASCAL VOC 2012 `val`에서 무작위 선택해 직접 amodal mask를 annotation했다. 이 subset에서 제안 방법과 IIS를 비교했다. 여기서 IIS는 modal segmentation 방법이므로, amodal task에 대한 강한 baseline으로 사용된다.

#### Segmentation performance

ground-truth modal box와 category를 주었을 때, 제안 방법은 대부분의 instance에서 IIS보다 높은 IoU를 보였다. 논문은 제안 방법이 **73%의 object에서 IIS보다 더 좋은 mask**를 만들었고, 나빠진 27% 중 다수는 IoU 감소가 5% 미만이라고 설명한다.

정량 결과는 다음과 같다.

- **IIS**: Accuracy at 50% = `68.0`, Accuracy at 70% = `37.0`, AUC = `57.5`
- **Proposed Method**: Accuracy at 50% = `80.0`, Accuracy at 70% = `48.0`, AUC = `64.3`

즉, 50% IoU 기준으로 12포인트, 70% 기준으로 11포인트 향상되었다. 특히 많은 사례에서 IoU가 **20~50%까지 개선**되었다고 논문은 서술한다.

#### Detection + Segmentation pipeline

이제 detector까지 포함한 end-to-end pipeline을 평가한다. detector는 Faster R-CNN을 사용하고, segmentation head로 IIS 또는 proposed method를 붙인다. 평가는 mean region average precision, 즉 $mAP^r$로 한다. 다만 일부 instance는 amodal ground truth가 없어서 prediction과 GT의 assignment에는 box overlap을, correctness 판정에는 region IoU를 사용하도록 metric을 약간 수정했다.

결과는 다음과 같다.

- **Faster R-CNN + IIS**: $mAP^r@50 = 34.1$, $mAP^r@70 = 14.0$
- **Faster R-CNN + Proposed Method**: $mAP^r@50 = 45.2$, $mAP^r@70 = 22.6$

즉, 제안 방법은 pipeline 수준에서도 **+11.1 points @50**, **+8.6 points @70**의 개선을 보였다.

### 4.5 Supplementary의 추가 실험

보충 자료에는 ablation이 있다.

- **Without Modal Segmentation Prediction**: $mAP^r@50 = 35.2$, $mAP^r@70 = 18.4$
- **Without Dynamic Sample Generation**: $mAP^r@50 = 39.8$, $mAP^r@70 = 22.7$
- **With Both**: $mAP^r@50 = 45.2$, $mAP^r@70 = 22.6$

이 결과는 두 요소가 중요함을 보여준다.

첫째, **modal segmentation prediction을 입력으로 넣는 것**이 중요하다. 보이는 부분의 구조를 먼저 잘 잡아야 그 바깥의 amodal 확장이 안정적이기 때문이다.

둘째, **dynamic sample generation**, 즉 occluder의 종류, 위치, 크기를 계속 바꾸며 다양한 synthetic occlusion을 만드는 것이 중요하다. fixed occlusion pattern으로 학습하면 일반화가 떨어진다.

또한 PASCAL 3D+ rigid object subset에서도 제안 방법이 IIS 대비 향상되었다.

- **Faster R-CNN + IIS**: $mAP^r@50 = 37.4$, $mAP^r@70 = 15.9$
- **Faster R-CNN + Proposed Method**: $mAP^r@50 = 44.0$, $mAP^r@70 = 20.9$

다만 논문도 명시하듯, CAD projection을 amodal GT로 쓰는 것은 shape mismatch 때문에 근사적인 평가일 뿐이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **학습 데이터 부재라는 핵심 병목을 우아하게 우회했다는 점**이다. amodal annotation이 없어도 modal annotation만으로 synthetic occlusion을 구성해 supervised training이 가능하다는 발상은 매우 실용적이고, 이후 관련 연구의 출발점을 제공했다.

두 번째 강점은 문제 정의와 시스템 설계가 자연스럽게 연결된다는 점이다. modal prediction을 먼저 얻고, 이를 조건으로 amodal mask를 복원하며, heatmap 바깥 반응을 이용해 box를 iterative하게 키우는 방식은 실제 occlusion reasoning 흐름과 잘 맞는다. 단순히 더 큰 mask를 예측하는 것이 아니라 “현재 box 바깥에도 object가 이어지는가?”를 반복 판단하는 구조가 설득력 있다.

세 번째 강점은 실험이 제한된 데이터 환경에서도 가능한 한 설득력 있게 설계되었다는 점이다. qualitative result만 제시한 것이 아니라, indirect evaluation, direct evaluation, detector 포함 pipeline 평가, ablation, PASCAL 3D+ 결과까지 제시해 방법의 유효성을 다각도로 보이려 했다.

한편 한계도 분명하다. 가장 본질적인 한계는 **amodal completion 자체의 모호성**이다. 특히 exterior occlusion에서는 정답이 하나가 아닐 수 있다. 논문도 이를 인정하며, heavily occluded articulated object에서는 여러 plausible hypothesis가 존재한다고 설명한다. 따라서 IoU 기반 평가만으로 “좋은 amodal completion”을 완전히 측정하기 어렵다.

또 다른 한계는 synthetic occlusion과 real occlusion 간의 domain gap이다. 논문은 synthetic training으로도 real occlusion에 잘 일반화된다고 보였지만, overlay 방식의 occlusion은 실제 장면의 조명 변화, 경계 blending, 물리적 상호작용을 완전히 재현하지 못한다. 이는 특히 fine boundary quality나 복잡한 scene composition에서 성능 한계로 이어질 수 있다.

또한 방법은 테스트 시 **modal bounding box와 object category가 이미 주어져 있다**는 가정에서 segmentation 실험을 수행한다. detector를 붙인 pipeline 실험도 있지만, 핵심 segmentation 성능 분석은 detector error를 분리한 설정이다. 따라서 실제 완전한 amodal perception 시스템으로 볼 때 detector quality에 상당히 의존할 수 있다.

비판적으로 보면, 제안된 Iterative Bounding Box Expansion은 직관적이지만, threshold 기반 heuristic 성격이 강하다. 예를 들어 평균 heat intensity가 `0.1`을 넘는지 여부로 확장 여부를 정하는 것은 간단하지만, 장면이나 카테고리에 따라 최적 threshold가 다를 가능성이 있다. 논문은 이 기준을 사용했다고 명시하지만, 이 threshold의 민감도 분석은 본문에서 충분히 제시하지 않는다.

마지막으로, unknown region을 loss에서 제외하는 설계는 합리적이지만, occluder 뒤쪽의 구조를 더 강하게 학습시키는 explicit shape prior나 multi-hypothesis prediction은 없다. 따라서 ambiguous case에서는 plausible한 한 가지 해를 내지만, uncertainty 자체를 표현하지는 못한다.

## 6. 결론

이 논문은 **amodal instance segmentation을 위한 최초의 일반 목적 방법**을 제시했다고 주장하며, 그 핵심은 공개된 amodal annotation 없이도 학습할 수 있는 synthetic data generation 전략에 있다. modal annotation만으로 training pair를 만들고, modal heatmap을 조건으로 amodal heatmap을 예측하며, Iterative Bounding Box Expansion으로 amodal box까지 추론하는 전체 파이프라인을 제안했다.

실험 결과는 제안 방법이 단순한 modal segmentation baseline인 IIS보다 일관되게 우수함을 보여준다. 직접 annotation한 100개 객체 평가에서 IoU 기반 정확도가 개선되었고, Faster R-CNN과 결합한 pipeline에서도 $mAP^r$가 뚜렷하게 상승했다. 즉, 이 연구는 “가려진 부분까지 포함한 object-level scene understanding”이 실제 학습 가능한 문제임을 보여준 초기 작업으로 의미가 크다.

향후 연구 관점에서도 중요하다. 이후의 amodal perception, occlusion reasoning, 3D-aware instance understanding, physical size estimation 같은 문제들은 모두 이 논문이 제시한 문제 설정과 데이터 생성 철학의 영향을 받을 수 있다. 다만 ambiguity, synthetic-real gap, heuristic box expansion, uncertainty modeling 부재 같은 한계는 후속 연구에서 보완되어야 할 부분이다. 전체적으로 보면, 이 논문은 새로운 task를 제안하는 수준을 넘어, **실제로 작동하는 baseline과 학습 전략을 함께 제시했다는 점에서 매우 중요한 출발점**이다.
