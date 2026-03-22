# DoNet: Deep De-overlapping Network for Cytology Instance Segmentation

이 논문은 세포 인스턴스 분할, 특히 **cytology image**에서 발생하는 **반투명 세포들의 중첩(overlap)** 문제와 **배경의 mimic/debris가 nucleus로 오인되는 문제**를 동시에 다루는 방법을 제안한다. 저자들은 이를 위해 **DoNet**이라는 decompose-and-recombine 기반 네트워크를 제안하며, 핵심은 하나의 세포 클러스터를 그대로 분할하려 하지 않고, **intersection region**과 **complement region**으로 먼저 분해한 뒤 다시 재조합하여 인스턴스 수준의 일관된 mask를 복원하는 데 있다. 여기에 nucleus가 cytoplasm 내부에 존재해야 한다는 생물학적 prior를 활용하는 **MRP**까지 추가해, 기존 Mask R-CNN 계열이나 amodal segmentation 계열보다 더 강한 overlapping perception을 확보하려고 한다. 논문은 ISBI2014와 CPS 데이터셋에서 이 접근이 SOTA 대비 우수하다고 주장한다.  

## 1. Paper Overview

이 논문의 목표는 **cytology instance segmentation**에서 겹쳐진 세포들의 경계를 더 정확히 분리하는 것이다. Cytology 영상은 암 스크리닝과 세포 형태 분석에서 중요하지만, 세포질(cytoplasm)이 반투명하고 서로 겹치기 쉬워서 경계가 매우 모호하다. 게다가 background의 백혈구, 점액, debris, bubble 같은 요소가 nucleus로 잘못 인식될 수 있다. 저자들은 기존 방법이 이런 현상을 충분히 모델링하지 못한다고 보고, 겹침을 단순한 occlusion이 아니라 **반투명 구조의 부분 영역 간 관계 문제**로 재해석한다. 그래서 세포를 바로 하나의 mask로 예측하는 대신, 겹친 영역과 비겹친 영역을 나눠 예측하고 이를 다시 합치는 구조를 설계했다.  

이 문제가 중요한 이유는 단순히 segmentation 성능의 문제가 아니라, 이후의 **세포 형태 계측**, **nuclear-cytoplasmic ratio 계산**, **암 선별 자동화** 같은 downstream 분석의 신뢰성을 좌우하기 때문이다. 논문은 특히 기존 instance segmentation 모델들이 겹침 내부의 구조적 일관성을 충분히 학습하지 못해 애매한 경계를 만든다고 비판한다.

## 2. Core Idea

이 논문의 가장 중요한 아이디어는 다음 한 문장으로 요약할 수 있다.

> **겹친 세포를 하나의 모호한 객체로 직접 분할하지 말고, 겹침이 만들어내는 부분 구조를 분해해 예측한 뒤, 그 부분들 사이의 semantic consistency를 이용해 다시 결합하자.**

기존 Mask R-CNN류 방법은 RoI 안에서 곧바로 instance mask를 예측한다. 하지만 DoNet은 겹친 세포가 만들어내는 sub-region을 명시적으로 본다. 논문은 각 세포 인스턴스를 다음 세 층으로 생각한다.

* **intersection layer**: 다른 세포와 겹치는 부분
* **complement layer**: 겹치지 않는 나머지 부분
* **instance layer**: 최종적인 전체 세포 인스턴스

즉, 세포 경계를 직접 맞추는 대신, “어디가 공통으로 겹치는 부분인지”, “어디가 개별 세포의 나머지 부분인지”를 예측하게 하여 모델의 지각 능력(perceptual capability)을 높이겠다는 발상이다. 이는 amodal instance segmentation에서 영감을 받았지만, 자연영상의 불투명 occlusion과 달리 cytology의 반투명 overlap은 **intersection이 양쪽 인스턴스 모두에 속한다는 점**이 다르다고 본다. 바로 이 차이를 모델 구조로 끌어온 것이 DoNet의 novelty다.  

또 하나의 핵심 아이디어는 nucleus segmentation에 **생물학적 containment prior**를 넣은 것이다. nucleus는 cytoplasm 내부에 있어야 하므로, cytoplasm prediction을 attention처럼 활용해 nuclei proposal을 더 잘 만들자는 것이 MRP의 직관이다. 이 아이디어는 단순한 성능 향상 기법이 아니라, 의료영상에서 자주 필요한 **domain prior의 구조적 주입**이라는 점에서 의미가 있다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

DoNet은 크게 네 부분으로 구성된다.

1. **Coarse mask segmentation baseline**

   * 기본 백본은 Mask R-CNN이다.
   * 먼저 coarse한 instance mask와 feature를 얻는다.

2. **DRM (Dual-path Region Segmentation Module)**

   * coarse feature와 coarse mask를 바탕으로,
   * 각 세포 클러스터를 **intersection region**과 **complement region**으로 분해해 예측한다.

3. **CRM (Semantic Consistency-guided Recombination Module)**

   * DRM의 출력과 RoI feature를 이용해
   * 다시 전체 instance를 복원한다.
   * 이때 “부분 영역들의 의미적 일관성”이 유지되도록 refinement한다.

4. **MRP (Mask-guided Region Proposal)**

   * recombined cytoplasm mask를 prior로 활용해
   * nucleus proposal / refinement를 개선한다.
   * 목적은 background mimic로 인한 오검출을 줄이는 것이다.  

논문 그림 설명에서도 DoNet이
(1) coarse segmentation,
(2) intersection/complement regression,
(3) semantic consistency 기반 refinement,
(4) background noise 억제를 위한 mask-guided proposal
의 흐름으로 이루어진다고 명시한다.

### 3.2 문제 정식화

논문은 데이터셋을
$\mathcal{D}={(\mathcal{X}_k,\mathcal{Y}_k)}_{k=1}^{K}$
형태로 두고, 각 이미지에 bounding box, category, instance mask annotation이 있다고 둔다. 그리고 각 인스턴스 mask를 겹침 구조에 따라 다음 두 부분으로 나눈다.

* intersection region: $\mathcal{O}_k={o_{k,i}}_{i=1}^{N_k}$
* complement region: $\mathcal{M}_k={m_{k,i}}_{i=1}^{N_k}$

즉, 최종 instance mask $\mathcal{E}_k$를 직접만 보지 않고, 이를 overlap 관계에 의해 분해된 두 구성요소로 다시 표현하는 것이다. 이 정식화 자체가 논문의 구조를 잘 드러낸다. 세포 인스턴스를 “하나의 바이너리 mask”로 보는 대신 “겹침 관계가 반영된 구조적 객체”로 본다는 점이 중요하다.  

### 3.3 DRM: 분해 단계

DRM은 DoNet의 첫 번째 핵심 모듈이다. 저자들의 문제의식은 이렇다.

기존 모델은 겹친 세포를 한 번에 segmentation하려고 하므로, intersection과 complement 간 appearance inconsistency를 설명하지 못한다. 특히 반투명 세포에서는 overlap 영역이 단순히 가려진 invisible part가 아니라, 독립적인 시각적 특징을 가진다.

그래서 DRM은 **두 개의 path**를 통해 각각 intersection과 complement를 예측한다. 이 모듈의 역할은 단순히 auxiliary prediction을 늘리는 것이 아니라, 모델이 겹침 상황을 보다 구조적으로 이해하게 만드는 것이다. 저자들은 이 분해를 통해 hidden interaction of sub-regions를 먼저 학습하고, 이후 CRM이 이를 다시 통합하도록 설계했다.  

해석적으로 보면 DRM은 instance mask prediction을 **latent part decomposition task**로 바꾸는 셈이다. 즉, 최종 task를 잘 풀기 위해 중간에 더 잘 정의된 subtask를 넣는 구조다.

### 3.4 CRM: 재조합과 consistency

DRM만 있으면 문제가 끝나지 않는다. 분해만 잘한다고 해서 최종 인스턴스가 항상 일관되게 복원되지는 않기 때문이다. 그래서 CRM이 등장한다.

CRM의 역할은:

* DRM이 생성한 sub-region 표현
* RoIAlign에서 얻은 instance-level feature

를 함께 사용해서 최종 recombined mask $\hat e^r_{k,i}$를 만들고, 이 결과가 부분 예측과 **semantic consistency**를 갖도록 만드는 것이다. 논문 표현대로라면 CRM은 refined instances와 integral sub-region predictions 사이의 consistency를 유도한다.

이 부분은 DoNet에서 매우 중요하다. DRM만 있으면 모델은 부분을 따로따로 맞추는 multi-task segmentation에 가까워질 수 있다. 하지만 CRM은 “부분들이 결국 하나의 세포 인스턴스로 자연스럽게 이어져야 한다”는 제약을 준다. 그래서 DoNet은 단순한 부분 분할 모델이 아니라, **분해와 재결합을 하나의 학습 논리로 묶은 모델**이 된다.

질적으로도 논문은 CRM이 sub-region 간 interaction을 추가함으로써 overlapping region에서 더 강한 perceptual capability를 보여준다고 설명한다.

### 3.5 MRP: biology prior를 proposal에 주입

MRP는 nucleus segmentation 개선용 모듈이다. 핵심 아이디어는 nucleus가 cytoplasm 내부에 존재한다는 사실을 proposal 단계에 반영하는 것이다.

일반적인 detector/segmenter는 nucleus와 background의 mimic를 혼동할 수 있다. 특히 cytology 배경에는 nucleus와 비슷해 보이는 구조물이 많다. DoNet은 recombined cytoplasm prediction을 활용해 **cell attention map**을 만들고, 이를 통해 proposal이 intra-cellular area에 집중하도록 한다. 즉, 아무 데서나 nucleus를 찾지 않고 “세포 내부일 가능성이 높은 영역” 안에서 찾게 한다.  

이 모듈의 의미는 분명하다.

* DRM/CRM은 겹친 cytoplasm 경계 해결에 초점
* MRP는 그 결과를 다시 nucleus segmentation 개선에 활용

즉, DoNet은 cytoplasm과 nucleus를 완전히 독립적으로 다루지 않고, **세포 구조의 포함 관계**를 파이프라인 안에 반영한다.

### 3.6 수식과 목적함수 해석

업로드된 ar5iv HTML은 수식 파싱이 일부 복잡하게 보이지만, 논문의 핵심 목적함수 수준에서 읽어야 할 포인트는 명확하다.

* coarse instance segmentation loss가 존재하고
* DRM에서 intersection/complement region supervision이 추가되며
* CRM에서 recombined instance와 부분 영역 간 consistency를 유도하는 loss가 들어가고
* MRP는 proposal/refinement를 cytoplasm prior로 guide한다

즉, 전체 학습은 단일 binary mask 예측이 아니라,
**instance-level supervision + part-level supervision + consistency regularization + biological prior**
의 결합으로 이해하면 된다. 수식의 세부 coefficient나 exact implementation detail은 본문에서 표현되지만, 업로드된 HTML의 수식 렌더링은 일부 읽기 불편하다. 따라서 여기서는 논문의 논리 구조 중심으로 해석하는 것이 정확하다. 수식 기호의 완전한 재구성보다, loss가 어떤 역할을 하는지 이해하는 것이 더 중요하다.  

## 4. Experiments and Findings

### 4.1 데이터셋과 평가 관점

논문은 두 개의 overlapping cytology segmentation 데이터셋에서 실험한다.

* **ISBI2014**
* **CPS**

그리고 nucleus와 cytoplasm에 대한 segmentation 성능을 비교한다. 논문은 정량 비교에서 DoNet이 다른 SOTA 방법보다 우수하다고 주장한다.  

### 4.2 무엇과 비교했는가

논문은 다음 계열과 비교한다.

* standard instance segmentation: **Mask R-CNN**
* amodal 계열: **Occlusion R-CNN**
* cytology 특화 / prior methods
* relation-based 또는 semi-supervised 계열(예: IRNet, MMT-PSM)

즉 비교 대상이 단순 baseline 하나가 아니라, overlapping segmentation을 다뤄온 여러 흐름을 포함한다는 점이 의미 있다. 논문은 기존 방법들이 overlapping sub-region 간 관계를 충분히 포착하지 못해 경계가 모호해진다고 해석한다.

### 4.3 정량 결과의 의미

논문은 Table 1에서 CPS와 ISBI2014 모두에서 DoNet이 경쟁 방법보다 우수하다고 제시한다. 정량표 전체 숫자를 여기서 모두 재현하긴 어렵지만, 저자들의 메시지는 분명하다.

* DoNet은 **cytoplasm + nuclei 평균 성능**에서 강하다.
* 특히 overlap이 심하거나 contrast가 낮은 상황에서 이점이 크다.
* 단순한 coarse mask 개선이 아니라, sub-region decomposition과 consistency modeling이 실제로 성능 향상에 기여한다.

### 4.4 Ablation study

Ablation에서 중요한 관찰은 다음과 같다.

* DRM만 추가해도 ISBI2014에서 average mAP가 개선된다.
* 하지만 구조/형태 정보를 충분히 융합하지 않으면 복잡한 CPS에서는 오히려 혼동될 수 있다.
* DRM 뒤에 CRM을 넣어 decompose-and-recombined strategy를 완성하면 성능이 추가로 오른다.
* 논문은 이 전략이 ISBI2014와 CPS에서 각각 유의미한 추가 향상을 가져왔다고 보고한다. 업로드된 snippet에서는 예시로 **7.34%**와 **1.74%**의 개선 수치가 보인다.

이 결과는 매우 중요하다. 왜냐하면 “분해” 자체보다도 “분해 후 재조합과 consistency”가 핵심이라는 점을 보여주기 때문이다. 다시 말해, DoNet의 기여는 단순 multi-head segmentation이 아니라, **구조적 분해-재조합 학습 전략 전체**에 있다.

### 4.5 Qualitative analysis

질적 비교에서 논문은 CPS와 ISBI2014 모두에서 DoNet이 더 선명한 경계를 만든다고 보인다. 특히 CPS에서는 서로 다른 staining을 가진 겹친 세포들이 sub-region 간 appearance inconsistency를 크게 보이는데, 기존 Mask R-CNN이나 Occlusion R-CNN은 intersection/complement의 관계를 잘 이해하지 못해 윤곽이 흐려진다. 반면 DoNet은 instance boundary를 더 잘 구분하고, 세포의 전체성(integrality)을 더 잘 포착한다고 설명한다.  

또한 heatmap 시각화에서는 intersection, complement, integral instance가 의미 있게 분리되는 것을 보여주며, 이는 모델이 실제로 overlapping concept를 internalize했음을 뒷받침한다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 문제를 잘 재정의했다는 점이다. 기존에는 겹침을 그냥 “어려운 segmentation” 정도로 다뤘다면, DoNet은 이를 **반투명 인스턴스의 부분 구조 관계 문제**로 풀어낸다. 이 재정의 덕분에 DRM, CRM, MRP가 각각 논리적으로 연결된다. 단순히 모듈을 많이 붙인 것이 아니라, 문제 분석에서 구조가 자연스럽게 나온다.

두 번째 강점은 **의료영상 prior와 general vision 아이디어의 결합**이다. amodal perception에서 영감을 얻되, cytology의 반투명성이라는 차이를 명확히 반영했다. 또 nucleus-in-cytoplasm이라는 biological prior를 proposal에 녹였다는 점도 의료 AI다운 설계다.

세 번째 강점은 ablation이 비교적 설득력 있다는 점이다. 특히 DRM만으로는 충분치 않고 CRM이 중요하다는 메시지가 드러나며, 질적 시각화도 이 주장을 보강한다.  

### 한계

첫째, 이 논문은 overlap 구조를 잘 다루지만, 그 효과가 **cervical cytology와 유사한 반투명 세포 이미지**에 얼마나 강하게 의존하는지는 더 검증이 필요하다. 저자들도 일반 vision occlusion과 cytology overlap을 구분해 설명하므로, 이 방법이 모든 instance segmentation 문제에 동일하게 통할 것이라고 보긴 어렵다. 다만 결론에서는 general occluded instance segmentation에도 잠재력이 있다고 말한다.

둘째, 방법이 Mask R-CNN 위에 DRM, CRM, MRP를 얹는 구조이므로, 구현과 학습이 baseline보다 복잡하다. 실제 의료 현장 적용에서는 성능뿐 아니라 추론 시간, annotation cost, 튜닝 안정성도 중요하지만, 업로드된 본문만으로는 이 부분이 자세히 논의되지는 않는다. 따라서 실무 적용 관점에서는 추가 검토가 필요하다.

셋째, 수식/학습 손실의 완전한 직관은 본문 렌더링상 일부 복잡하게 보인다. 즉, 논문의 개념은 명확하지만, exact loss design을 구현 수준으로 따라가려면 코드나 원문 PDF를 병행하는 편이 좋다. 이는 논문 자체의 약점이라기보다 ar5iv HTML 해석상의 한계이기도 하다.

### 해석

비판적으로 보면, DoNet의 진짜 기여는 “새로운 segmentation head 하나”가 아니다. 오히려 **부분-전체 관계를 구조화하여 학습하는 방식** 자체가 핵심이다. 이 관점은 세포처럼 반투명하게 겹치는 객체뿐 아니라, 내부 구조적 제약이 있는 의료 segmentation 문제 전반으로 확장될 가능성이 있다. 예를 들어 organ-within-organ, lesion-inside-tissue 같은 관계에도 비슷한 철학을 적용할 수 있다. 다만 그 확장은 논문이 직접 실험한 범위를 넘는 해석이므로, 가능성 수준으로 보는 것이 타당하다.

## 6. Conclusion

이 논문은 cytology instance segmentation에서 가장 까다로운 두 문제인 **overlapping translucent cell boundary ambiguity**와 **background mimic confusion**을 동시에 다루기 위해 DoNet을 제안했다. 핵심은 세포를 intersection/complement로 분해하는 **DRM**, 의미 일관성을 유지하며 전체 instance로 복원하는 **CRM**, 그리고 cytoplasm prior를 이용해 nuclei prediction을 개선하는 **MRP**의 조합이다. 실험적으로는 ISBI2014와 CPS에서 기존 방법 대비 우수한 성능을 보였고, 질적 분석에서도 겹친 세포 경계를 더 자연스럽게 복원하는 모습을 보인다.  

실무적으로 이 논문은 **의료영상에서 구조적 prior와 part-whole reasoning을 결합한 segmentation 설계**의 좋은 예다. 앞으로 비슷한 유형의 반투명 중첩 구조를 다루는 문제에서 중요한 참고가 될 가능성이 높다.
