# Simultaneous Detection and Segmentation

- **저자**: Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik
- **발표연도**: 2014
- **arXiv**: https://arxiv.org/abs/1407.1808

## 1. 논문 개요

이 논문은 object detection과 semantic segmentation 사이의 간극을 메우는 새로운 과제인 **Simultaneous Detection and Segmentation (SDS)** 를 제안하고, 이를 위한 실제 동작하는 시스템을 설계한다. SDS의 목표는 단순히 물체가 어디 있는지를 bounding box로 찾는 데서 끝나지 않고, **각 object instance를 개별적으로 검출한 뒤 그 instance에 속하는 픽셀까지 정확히 지정하는 것**이다. 다시 말해, semantic segmentation처럼 픽셀 단위 예측을 하되, detection처럼 객체 개체 수와 개별 위치까지 구분해야 한다.

논문이 문제 삼는 핵심은 기존 두 과제가 서로 보완적이지만 분리되어 다뤄져 왔다는 점이다. object detection은 instance 구분은 잘하지만 localization이 bounding box 수준으로 거칠고, semantic segmentation은 픽셀 단위 분류는 가능하지만 같은 클래스의 여러 instance를 구분하지 못한다. 특히 사람, 자동차, 개 같은 “thing” category에서는 “어떤 픽셀이 어느 개체에 속하는가”가 중요하므로, 두 문제를 통합한 SDS가 더 자연스럽고 실용적인 문제 설정이라는 것이 저자들의 주장이다.

이 문제의 중요성은 평가 기준에서도 드러난다. 논문은 detection의 average precision 개념을 segmentation mask 수준으로 확장한 $AP^r$를 사용한다. 즉 예측이 ground truth와 충분히 겹치는 segmentation을 낼 때만 true positive로 인정한다. 따라서 SDS는 단순한 분류나 박스 위치 추정이 아니라, **instance-level recognition과 precise localization을 동시에 요구하는 더 어려운 문제**다. 저자들은 이 과제를 잘 해결하면 object detection, semantic segmentation 모두에도 도움이 된다고 보고, 실제로 실험에서 세 과제 모두에서 강한 성능을 보인다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **bottom-up region proposal**과 **top-down category-specific reasoning**을 결합하는 것이다. 먼저 category-independent한 region proposal을 이용해 가능한 object mask 후보들을 폭넓게 모은 뒤, CNN 기반 특징 추출과 분류를 통해 후보를 점수화하고, 마지막에 class-specific mask refinement를 적용해 segmentation 품질을 높인다. 즉, 후보 생성은 하향식 의미 정보를 거의 쓰지 않고 다양하게 확보하고, 이후 단계에서 깊은 특징과 클래스 정보를 이용해 정제하는 구조다.

기존 R-CNN 계열과의 가장 중요한 차별점은, 입력을 단순 bounding box 하나로 보지 않고 **같은 후보에 대해 두 종류의 시각 정보**를 따로 본다는 점이다. 하나는 region의 bounding box 전체이고, 다른 하나는 그 박스 안에서 **region foreground만 남기고 background를 mask 처리한 입력**이다. 저자들은 이 두 입력이 서로 다른 정보를 담는다고 본다. bounding box 입력은 object 주변 문맥과 전체 모양을 포함하고, region foreground 입력은 실제 mask의 형상과 내부 appearance를 반영한다.

또 하나의 핵심 차별점은 두 입력 경로를 단순히 이어붙이는 데서 멈추지 않고, **각 경로를 그 역할에 맞게 따로 fine-tune한 뒤, 다시 joint training으로 통합**했다는 점이다. 논문은 이를 통해 같은 CNN을 공유하는 단순 방식보다 훨씬 나은 SDS 성능을 얻는다. 마지막 refinement 단계도 중요하다. 초기 proposal은 class-agnostic하기 때문에 object 일부를 놓치거나 배경을 과도하게 포함할 수 있는데, 이를 보완하기 위해 coarse한 $10 \times 10$ figure-ground mask를 예측하고, 다시 superpixel 수준에서 후보 mask와 결합해 더 나은 segmentation을 만든다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 네 단계로 구성된다. 첫째는 **proposal generation**, 둘째는 **feature extraction**, 셋째는 **region classification**, 넷째는 **region refinement**다.

### Proposal generation

저자들은 region proposal로 MCG (Multiscale Combinatorial Grouping)를 사용하고, 이미지당 약 2000개의 region candidate를 만든다. SDS는 segmentation mask 자체가 필요하므로 box proposal보다 **segment proposal**이 더 적합하다. MCG는 여러 해상도에서 segmentation hierarchy를 만든 뒤 이를 결합하고, 조합적으로 region들을 묶어 후보를 만든다. 후보는 크기, 위치, shape, contour strength 같은 단순 특징으로 rank된다.

이 선택은 평가 목표와 직접 연결된다. 논문은 segmentation overlap 기반의 $AP^r$를 최적화하려 하므로, box가 아니라 region candidate가 필요하다. 실제로 Selective Search 대신 MCG를 쓰는 것만으로도 detection 기준 $AP^b$가 소폭 상승했다고 보고한다.

### Feature extraction

이 단계가 논문의 기술적 중심이다. 출발점은 R-CNN이다. R-CNN은 proposal의 bounding box를 crop, warp하여 CNN에 넣고, 상위 fully connected layer의 feature를 추출해 SVM으로 분류한다. 하지만 SDS에서는 bounding box feature만으로는 region mask 자체가 ground truth segmentation과 잘 맞는지를 판단하기 어렵다. 그래서 저자들은 region foreground를 직접 반영하는 두 번째 입력을 추가한다.

논문은 세 가지 feature extractor를 비교한다.

첫 번째는 **A**이다. 이 방식은 하나의 CNN을 이용해 두 입력을 모두 처리한다. 하나는 후보 region의 bounding box, 다른 하나는 같은 box 안에서 region 바깥 background를 mean image로 지운 masked region이다. 두 feature를 concatenate하여 최종 feature로 쓴다. 이 방식은 간단하지만, 네트워크가 원래 bounding box 기반 분류에 맞게 fine-tune되어 있으므로 region foreground 입력에는 최적이 아닐 수 있다.

두 번째는 **B**이다. 여기서는 bounding box pathway와 region pathway를 분리한다. 특히 region pathway는 region foreground 입력에 맞춰 별도로 fine-tune한다. 이때 positive/negative label도 bounding box overlap이 아니라 **region overlap**, 즉 segmentation overlap 기준으로 만든다. 따라서 이 경로는 “이 후보 mask가 실제 ground truth mask와 잘 겹치는가”를 더 직접적으로 학습한다.

세 번째는 **C**이다. 이것이 저자들이 제안하는 핵심 구조다. 두 개의 CNN pathway를 가진 네트워크를 만들고, 마지막 classifier layer에서 두 경로의 feature를 결합한다. box pathway는 box 기반 fine-tuned network로 초기화하고, region pathway는 region-overlap 기반 fine-tuned network로 초기화한 뒤, **전체 네트워크를 end-to-end로 다시 fine-tune**한다. 테스트 시에는 마지막 classifier를 버리고 penultimate layer의 concatenated feature를 사용한다.

이 구조는 다음처럼 이해할 수 있다.

$$
f_{\text{SDS}}(R) = [f_{\text{box}}(R), f_{\text{region}}(R)]
$$

여기서 $R$은 region candidate이고, $f_{\text{box}}$는 bounding box 입력으로부터의 특징, $f_{\text{region}}$은 masked foreground 입력으로부터의 특징이다. 제안 방식 C는 이 두 특징을 독립적으로만 학습하는 것이 아니라, 최종 SDS 목적에 맞게 함께 조정한다는 점이 중요하다.

### Region classification

추출된 feature 위에 각 클래스별 linear SVM을 학습한다. 학습 방식은 다소 정교하다. 먼저 ground truth region을 positive로, ground truth와 20% 미만 overlap하는 region을 negative로 하여 초기 SVM을 학습한다. 그 뒤 positive set을 다시 추정한다. 각 ground truth에 대해 overlap 50% 이상인 MCG candidate 중 SVM score가 가장 높은 것을 새 positive로 선택하고, 이 positive set으로 SVM을 재학습한다.

논문은 이를 multiple instance learning 관점으로 해석한다. 각 ground truth는 “50% 이상 겹치는 여러 후보 중 적어도 하나는 진짜 positive”인 positive bag을 이루고, 각 negative는 단독 bag이 된다. 저자들은 단순히 ground truth만 positive로 쓰는 것보다 이 방식이 더 잘 동작했다고 보고한다.

테스트 시에는 각 region에 대해 classifier score를 계산하고, region overlap threshold 0의 strict non-maximum suppression을 수행한다. box는 겹칠 수 있어도 실제 pixel support는 보통 겹치지 않는다는 직관 때문이다. 이후 계산량을 줄이기 위해 category당 상위 20,000 detections만 유지한다.

### Region refinement

초기 region proposal은 bottom-up이며 class-agnostic하므로 object를 덜 포함하거나 지나치게 넓게 포함하는 문제가 있다. 이를 해결하기 위해 refinement를 수행한다.

먼저 각 surviving region의 bounding box를 padding 후 $10 \times 10$ grid로 나눈다. 각 grid cell에 대해 logistic regression classifier를 학습하여 그 cell이 foreground일 확률을 예측한다. 입력 feature는 CNN feature와 region candidate 자체를 같은 $10 \times 10$ grid로 discretize한 figure-ground mask다. 이 classifier는 training set에서 ground truth와 70% 이상 overlap하는 region들로 학습된다.

이를 통해 얻는 것은 coarse한 top-down mask다. 하지만 이 coarse mask는 contour를 정확히 따르지 못하고, 얇은 구조나 복잡한 형상을 잘 잡지 못한다. 그래서 저자들은 두 번째 단계를 둔다. coarse mask를 superpixel 위로 projection하여 각 superpixel에 평균 mask 값을 부여하고, 여기에 “원래 proposal에 포함되는가”를 나타내는 이진 feature를 추가해 superpixel classifier를 학습한다. 즉 refinement는 다음 두 정보의 결합이다.

- category-specific top-down shape prior
- bottom-up proposal이 가진 경계 정보

이 설계 덕분에 coarse mask가 object body를 채우고, 원 proposal이 경계를 세밀하게 보정하는 역할을 한다.

## 4. 실험 및 결과

실험은 SBD의 segmentation annotation을 사용해 수행되었고, 학습은 PASCAL VOC 2012 train, 평가는 VOC 2012 val에서 주로 이루어졌다. 네트워크 학습에는 Caffe를 사용했다.

### SDS 성능: $AP^r$와 $AP^{r}_{vol}$

논문의 핵심 평가 지표는 $AP^r$이다. 예측 segmentation과 ground truth segmentation의 IoU가 50%를 넘을 때 true positive로 인정하고, precision-recall curve 아래 면적을 평균 precision으로 계산한다. 또 threshold 하나에만 의존하지 않기 위해 여러 overlap threshold에서의 $AP^r$를 평균한 $AP^{r}_{vol}$도 사용한다.

결과적으로 평균 $AP^r$는 다음과 같이 향상된다.

- O2P baseline: 25.2%
- A: 42.9%
- B: 47.0%
- C: 47.7%
- C+ref: 49.7%

즉, 단순한 region+box feature 결합만으로도 큰 폭의 향상이 있었고, region-overlap에 맞춘 별도 fine-tuning(B), joint training(C), refinement(C+ref)가 단계적으로 추가 이득을 준다. 특히 최종 refinement는 평균 $AP^r$를 약 2포인트 더 올린다. 논문은 각 단계의 향상이 paired sample t-test 기준 0.05 수준에서 통계적으로 유의하다고 명시한다.

$AP^{r}_{vol}$도 유사한 경향을 보인다.

- O2P: 23.4%
- A: 37.0%
- B: 39.6%
- C: 40.2%
- C+ref: 41.4%

이 결과는 제안 기법이 특정 overlap threshold에만 맞춘 것이 아니라, 느슨한 기준부터 엄격한 기준까지 전반적으로 더 나은 localization을 제공함을 뜻한다.

또한 SegDPM과의 비교도 수행했다. 공개된 정보의 한계 때문에 SegDPM에는 upper bound를 계산했는데, 그 값이 mean $AP^r = 31.3$인 반면, C+ref는 같은 조건에서 50.3을 달성한다. 이는 당시 기준으로 매우 큰 차이다.

### 오류 분석과 localization 진단

논문의 강점 중 하나는 단순 성능 보고를 넘어 **오류의 종류를 분해해서 분석**했다는 점이다. 저자들은 detection error diagnosis 아이디어를 SDS에 확장해, mislocalization, background confusion, similar-category confusion의 영향을 분리한다.

분석 결과 가장 큰 문제는 압도적으로 **mislocalization**이다. 최종 시스템 C+ref에서 mislocalization을 완전히 고칠 수 있다고 가정하면 평균 $AP^r$가 약 16포인트 개선될 수 있었다. 반면 background나 유사 카테고리 혼동은 상대적으로 영향이 작았다. 이는 분류 자체보다 **정확한 instance mask localization**이 SDS의 본질적 병목임을 보여준다.

또한 저자들은 localization error를 overshooting과 undershooting으로 나눠 본다. 이를 위해 pixel precision과 pixel recall 관점의 AP를 별도로 측정한다. 예를 들어 pixel precision이 낮으면 배경까지 새어나가는 경향이 크고, pixel recall이 낮으면 object 일부를 놓치는 경향이 크다. 결과적으로 person, bird 같은 클래스는 ground truth 일부를 놓치는 경우가 많고, bicycle 같은 클래스는 배경으로 새는 경향이 있음을 보였다. 이런 분석은 단순 숫자 이상의 통찰을 준다.

### Bounding-box detection 성능: $AP^b$

흥미롭게도 제안 시스템은 SDS만 잘하는 것이 아니라 고전적 object detection에서도 강하다. bounding-box detection 기준 mean $AP^b$는 다음과 같다.

- R-CNN: 51.0%
- R-CNN-MCG: 51.7%
- A: 51.9%
- B: 53.9%
- C: 53.0%

특히 B가 가장 높은 53.9%를 기록했다. 논문은 C가 B보다 다소 낮은 이유로, C가 전체적으로 SDS 목적에 맞춰 fine-tune되었기 때문일 수 있다고 해석한다. 하지만 전체적으로는 R-CNN보다 더 나은 localization을 보이며, threshold가 엄격해질수록 개선 폭이 커진다. 이는 제안 방식이 “정확한 위치 추정”에 강하다는 해석과 일치한다.

VOC 2012 test에서도 C는 mean $AP^b = 50.7$을 기록하여 R-CNN의 49.6보다 높았고, SegDPM의 40.7보다도 크게 앞섰다.

### Semantic segmentation 성능: Pixel IU

최종 시스템 C+ref의 출력을 pixel-level category labeling으로 변환해 semantic segmentation도 평가했다. 결과는 다음과 같다.

- VOC2011 Test mean Pixel IU:
  - O2P: 47.6
  - R-CNN: 47.9
  - C+ref: 52.6
- VOC2012 Test mean Pixel IU:
  - O2P: 47.8
  - C+ref: 51.6

즉, 당시 state of the art 대비 약 5포인트, 상대적으로 약 10% 성능 향상이 있었다. 이는 SDS를 제대로 풀기 위한 구조가 detection과 semantic segmentation 모두에 이득이 될 수 있음을 실험적으로 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의 자체가 매우 설득력 있다는 점이다. detection은 instance를 구분하지만 coarse하고, semantic segmentation은 dense하지만 instance를 구분하지 못한다는 기존 한계를 명확히 짚고, 이를 통합하는 SDS를 제안했다. 이후 computer vision 분야에서 instance segmentation이 매우 중요한 문제로 자리 잡았다는 점을 생각하면, 이 논문은 그 흐름을 앞서 보여준 작업으로 볼 수 있다.

방법론 측면에서도 강점이 뚜렷하다. box와 region foreground를 분리해 다루는 두-pathway 구조는 직관적으로도 타당하고, 실험적으로도 큰 효과가 있다. 특히 region-overlap 기준 fine-tuning과 joint training의 효과를 단계적으로 검증해, 단순 아이디어 제시에 그치지 않고 설계 선택의 이유를 데이터로 뒷받침한다. refinement 단계 역시 “bottom-up proposal + top-down class prior”의 결합이라는 고전적이지만 강력한 설계 철학을 잘 구현했다.

실험 설계도 좋다. 단순 평균 성능 수치 외에 $AP^r$, $AP^{r}_{vol}$, $AP^b$, Pixel IU를 모두 보고하고, mislocalization 중심의 diagnostic analysis를 제공한다. 특히 오류를 정량적으로 분해해 향후 연구 방향을 제시한 점은 reviewer 관점에서 높은 평가를 받을 만하다.

한계도 분명하다. 첫째, 전체 파이프라인이 **proposal 기반 다단계 구조**이기 때문에 계산량이 적지 않고, 구조도 복잡하다. 이미지당 2000개의 MCG region을 다루고, CNN feature 추출 후 SVM, NMS, refinement까지 이어지는 방식은 이후 등장한 end-to-end instance segmentation 모델에 비해 비효율적일 가능성이 크다. 다만 논문 본문에는 정확한 추론 시간이나 자원 비용이 자세히 제시되지 않았다.

둘째, refinement가 여전히 coarse한 $10 \times 10$ grid와 superpixel 기반 후처리에 의존한다. 이는 당시로서는 합리적이지만, 복잡한 형상이나 얇은 구조를 완전히 복원하기에는 한계가 있다. 논문도 aircraft wing 같은 thin structure나 움직임이 큰 구조는 coarse mask만으로 다루기 어렵다고 명시한다.

셋째, 오차 분석에서 가장 큰 병목이 mislocalization으로 남아 있다는 사실은, 분류보다 localization이 여전히 어렵다는 점을 보여준다. Table 3에 따르면 최종 C+ref에서도 perfect localization을 가정한 upper bound는 65.5이고 실제 $AP^r$ 손실 중 15.8포인트가 mislocalization 때문이다. 즉 제안 방법은 큰 진전을 보였지만, 문제를 완전히 해결하지는 못했다.

넷째, 이 논문은 SDS를 매우 잘 정의하고 풀지만, 실험은 주로 PASCAL VOC의 20개 category에 한정된다. 더 복잡하고 crowded한 장면, 더 많은 category, 더 작은 object가 포함된 환경에서 같은 접근이 어떻게 확장될지는 본문에 직접 다뤄지지 않는다. 따라서 대규모 일반화 가능성은 이 논문만으로는 판단하기 어렵다.

## 6. 결론

이 논문은 **각 object instance를 검출하면서 동시에 그 픽셀 mask까지 예측하는 SDS 문제를 체계적으로 정의하고, 이를 위한 CNN 기반 파이프라인을 제안한 선구적 연구**다. 핵심 기여는 세 가지로 정리할 수 있다. 첫째, detection과 semantic segmentation을 잇는 SDS라는 문제 설정과 평가 지표 $AP^r$, $AP^{r}_{vol}$를 제시했다. 둘째, bounding box와 region foreground를 함께 사용하는 two-pathway CNN과 class-specific refinement를 통해 강한 instance-level segmentation 성능을 달성했다. 셋째, 이 접근이 SDS뿐 아니라 object detection과 semantic segmentation 자체의 성능도 향상시킴을 보였다.

실제 적용 측면에서 이 연구는 이후의 **instance segmentation** 계열 연구의 중요한 전단계로 볼 수 있다. 오늘날의 관점에서는 proposal-based multi-stage pipeline이라는 한계가 있지만, “instance를 개별적으로 분리한 mask 예측”이라는 문제를 명확히 정식화했고, bottom-up proposal과 top-down semantic reasoning의 결합이 효과적임을 보여주었다는 점에서 의미가 크다. 향후 연구는 이 아이디어를 더 정교한 mask prediction, 더 강한 end-to-end 학습, 더 대규모 데이터셋으로 확장하는 방향으로 발전할 수 있으며, 실제로 이후 분야의 흐름도 그렇게 전개되었다.
