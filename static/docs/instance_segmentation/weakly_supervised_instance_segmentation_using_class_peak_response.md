# Weakly Supervised Instance Segmentation using Class Peak Response

* **저자**: Yanzhao Zhou, Yi Zhu, Qixiang Ye, Qiang Qiu, Jianbin Jiao
* **발표연도**: 2018
* **arXiv**: [https://arxiv.org/abs/1804.00880](https://arxiv.org/abs/1804.00880)

## 1. 논문 개요

이 논문은 **image-level label만으로 instance segmentation을 수행하는 문제**를 정면으로 다룬다. 기존 weakly supervised segmentation 연구의 대부분은 semantic segmentation에 집중되어 있었고, 같은 클래스에 속한 여러 객체를 서로 구분해야 하는 instance segmentation까지는 자연스럽게 확장되지 못했다. 그 이유는 일반적인 classification network에서 얻는 class response map이 “이 클래스가 어디에 있는가” 정도는 알려주지만, 같은 클래스의 서로 다른 객체 인스턴스를 분리하는 정보는 충분히 주지 못하기 때문이다.

논문의 핵심 문제의식은 다음과 같다. 이미지 수준 레이블만으로 학습한 CNN도 내부적으로는 객체를 알아보기 위한 공간적 단서를 학습한다. 그렇다면 그 단서 중에서 **각 인스턴스를 대표하는 강한 시각적 신호**를 찾아내고, 그것을 적절히 확장하면 인스턴스 단위 마스크를 복원할 수 있지 않을까? 저자들은 이 질문에 대해, class response map의 **local maximum**, 즉 **peak**가 각 객체 내부의 강한 단서와 자주 대응한다는 관찰에서 출발한다.

이 문제는 중요하다. pixel-level mask annotation은 매우 비싸고 시간이 많이 드는 반면, image-level label은 상대적으로 매우 저렴하다. 만약 image-level supervision만으로도 instance segmentation이 어느 정도 가능해진다면, 대규모 데이터에 대한 확장성이 크게 개선되고, annotation cost를 획기적으로 줄일 수 있다. 논문은 바로 이 지점에서 의미가 있다. 저자들은 단순한 classification setting과 cross entropy loss만으로 학습한 네트워크에서 **instance-aware visual cue**를 끌어내는 방법을 제안하고, 이를 통해 weakly supervised point localization, semantic segmentation, 그리고 특히 image-level supervised instance segmentation에서 유의미한 성능을 보였다고 주장한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 직관적이다. **class response map의 peak는 객체 인스턴스 내부의 강한 시각적 증거를 나타내며, 이 peak를 중심으로 top-down 방식의 back-propagation을 수행하면 해당 인스턴스와 관련된 세밀한 영역 정보까지 복원할 수 있다**는 것이다.

기존 weakly supervised localization 또는 semantic segmentation 방법은 보통 클래스별 saliency map 또는 class activation map을 만든다. 이런 방법은 특정 클래스가 존재하는 영역을 대략 강조하는 데에는 유용하지만, 동일 클래스 내 여러 개체를 분리하는 데에는 한계가 있다. 즉, class-aware하지만 instance-agnostic한 표현에 머무는 경우가 많다. 이 논문은 그보다 한 단계 더 나아가, **각 peak를 하나의 잠재적 instance seed로 보고**, 그 seed마다 별도의 세밀한 response map을 생성한다. 이 map이 바로 **Peak Response Map (PRM)** 이다.

차별점은 두 부분에 있다. 첫째, 학습 단계에서 **peak stimulation**을 통해 class response map에서 peak가 더 분명하게 나타나도록 유도한다. 이는 네트워크가 전체 공간의 수많은 easy negative receptive field에 끌려가는 대신, 실제로 informative한 위치들에 더 집중하게 만든다. 둘째, 추론 단계에서 **peak back-propagation**을 통해 특정 peak 하나를 출발점으로 하여 bottom layer까지 relevance를 확률적으로 전파함으로써, 단순한 class saliency를 넘어 **instance-aware하고 boundary-sensitive한 표현**을 얻는다.

결국 이 논문이 말하는 핵심은, 별도의 detection head나 mask head 없이도, classification network 안에 이미 숨어 있는 instance-level cue를 잘 끌어내면 weak supervision 하에서도 인스턴스 마스크를 뽑아낼 수 있다는 점이다.

## 3. 상세 방법 설명

전체 방법은 크게 세 단계로 구성된다. 먼저 classification network를 fully convolutional network로 바꾸어 class response map을 얻는다. 다음으로 학습 시에는 peak stimulation을 적용해 유의미한 peak가 잘 드러나도록 하고, 추론 시에는 각 peak에 대해 peak back-propagation을 수행하여 PRM을 만든다. 마지막으로 PRM, class response map, 그리고 외부 object proposal을 결합하여 instance mask를 선택한다.

### 3.1 Fully Convolutional Architecture

저자들은 VGG16, ResNet50 같은 분류 네트워크를 FCN 형태로 변환한다. 구체적으로는 global pooling layer를 제거하고, fully connected layer를 $1 \times 1$ convolution으로 바꾼다. 이렇게 하면 한 번의 forward pass로 각 클래스에 대한 **class response map** $M \in \mathbb{R}^{C \times H \times W}$를 얻을 수 있다. 여기서 $C$는 클래스 수, $H \times W$는 공간 해상도이다.

이 response map의 각 위치는 해당 위치의 receptive field가 특정 클래스를 얼마나 지지하는지 나타낸다. 중요한 점은 이 표현이 이미 공간 구조를 유지하고 있으므로, localization이나 segmentation의 출발점으로 쓸 수 있다는 것이다.

### 3.2 Peak Stimulation

저자들은 class response map의 local maximum들이 객체 내부의 강한 단서라고 보고, 이를 더 명확히 드러내기 위해 **peak stimulation layer**를 추가한다. 각 클래스 $c$의 response map $M^c$에서, 반경 $r$의 윈도우 내 local maximum들을 peak로 정의한다. 논문에서는 모든 실험에서 $r=3$으로 설정했다.

클래스 $c$의 peak 위치 집합을 다음과 같이 둔다.

$$
P^c = {(i_1,j_1), (i_2,j_2), \dots, (i_{N^c}, j_{N^c})}
$$

여기서 $N^c$는 해당 클래스 response map에서 검출된 peak 개수이다.

이제 저자들은 peak 위치들만을 샘플링하는 kernel $G^c$를 만든다. 일반형은 다음과 같다.

$$
G^c_{x,y} = \sum_{k=1}^{N^c} f(x-i_k, y-j_k)
$$

논문에서는 sampling function $f$로 Dirac delta를 사용한다. 즉, 실제로는 peak 위치에서만 값을 읽어온다. 따라서 클래스 confidence score는 peak들의 평균으로 계산된다.

$$
s^c = M^c * G^c = \frac{1}{N^c}\sum_{k=1}^{N^c} M^c_{i_k,j_k}
$$

이 식의 의미는 명확하다. 기존의 global pooling 계열 방법은 전체 response map을 평균내거나 최대값만 보지만, 여기서는 **local peak들만 모아 최종 판단**을 한다. 다시 말해 네트워크는 “가장 informative한 지역적인 증거들”에 의해 분류 결정을 내리게 된다.

이 설계는 backward pass에서도 중요한 효과를 만든다. 분류 loss를 $L$이라 하면, top convolutional layer의 gradient는 다음과 같이 peak 위치들에만 분배된다.

$$
\delta^c = \frac{1}{N^c}\cdot \frac{\partial L}{\partial s^c}\cdot G^c
$$

즉, gradient가 response map 전체에 퍼지는 것이 아니라, peak에 해당하는 sparse한 위치들로 집중된다. 저자들은 이것이 매우 중요하다고 본다. 일반적인 dense sampling 관점에서 보면, 수많은 receptive field 중 대부분은 실제 객체를 포함하지 않는 easy negative이다. 그런데 peak stimulation은 학습을 그런 easy negative가 아니라 **potential positives와 hard negatives가 있는 informative receptive field 집합**에 집중하게 만든다. 이 때문에 localization 능력이 좋아지고, 여러 인스턴스를 더 잘 구분하는 표현이 학습된다고 해석한다.

### 3.3 Peak Back-propagation

학습 후 추론 단계에서는, 검출된 각 peak를 출발점으로 하여 **Peak Response Map (PRM)** 을 생성한다. 여기서 핵심은 이전의 class attention 방식과 달리, 단순히 “클래스 전체와 관련된 뉴런”을 찾는 것이 아니라, **특정 spatial peak 하나와 관련된 하위 뉴런들의 relevance**를 추적한다는 점이다. 그래서 결과적으로 instance-aware한 시각 단서를 얻을 수 있다.

저자들은 이를 확률적 random walk 관점으로 설명한다. 맨 위 layer의 peak에서 시작한 walker가 아래 layer로 내려가며 연결을 따라 이동한다고 생각한다. 그러면 하위 layer의 어떤 위치가 얼마나 자주 방문되는지가 그 위치의 relevance가 된다.

간단화를 위해 하나의 convolution filter $W \in \mathbb{R}^{k_H \times k_W}$를 생각하자. 입력 feature map을 $U$, 출력 feature map을 $V$라 할 때, 출력 위치 $V_{pq}$에서 입력 위치 $U_{ij}$로 relevance가 전파되는 관계를 다음처럼 쓴다.

$$
P(U_{ij}) = \sum_{p=i-\frac{k_H}{2}}^{i+\frac{k_H}{2}} \sum_{q=j-\frac{k_W}{2}}^{j+\frac{k_W}{2}} P(U_{ij}\mid V_{pq}) \times P(V_{pq})
$$

여기서 $P(V_{pq})$는 상위 위치의 방문 확률이고, $P(U_{ij}\mid V_{pq})$는 상위 위치에서 하위 위치로 갈 전이 확률이다. 이 전이 확률은 다음과 같이 정의된다.

$$
P(U_{ij}\mid V_{pq}) = Z_{pq} \times \hat{U}*{ij} W^+*{(i-p)(j-q)}
$$

여기서 $\hat{U}*{ij}$는 forward pass에서의 bottom-up activation이고, $W^+ = \mathrm{ReLU}(W)$는 음수 weight를 제거한 positive connection만 남긴 것이다. $Z*{pq}$는 확률의 합이 1이 되도록 하는 normalization factor이다.

이 식을 쉽게 해석하면 다음과 같다. 어떤 상위 뉴런의 활성은, 아래쪽의 입력 활성과 양의 가중치 연결을 통해 형성된다. 따라서 relevance를 아래로 보낼 때도, **실제로 상위 활성 형성에 기여했을 법한 하위 위치들**로만 relevance를 나누어 준다. 이 과정을 여러 층에 걸쳐 반복하면, 특정 peak 하나를 설명하는 이미지 공간상의 세밀한 관련 영역이 나온다. 이것이 PRM이다.

저자들은 PRM이 단순 saliency map보다 더 fine-detailed하고, 특히 객체 경계나 인스턴스별 분리 단서를 포착한다고 주장한다.

### 3.4 Weakly Supervised Instance Segmentation

PRM 자체가 곧바로 완전한 instance mask는 아니다. PRM은 각 인스턴스에 대한 강한 시각적 단서를 제공하지만, object mask 전체를 매끈하게 채우는 데에는 proposal prior가 필요하다. 그래서 저자들은 외부 object proposal, 구체적으로 MCG proposal을 사용한다.

각 peak에 대응하는 PRM $R$과 클래스 response map으로부터 얻은 class cue를 이용하여 proposal gallery에서 최적의 proposal $S$를 고른다. proposal의 score는 다음과 같다.

$$
\mathrm{Score} = \underbrace{\alpha \cdot R * S}_{\text{instance-aware}}

* \underbrace{R * \hat{S}}_{\text{boundary-aware}}

- \underbrace{\beta \cdot Q * S}_{\text{class-aware}}
  $$

여기서 $S$는 proposal mask, $\hat{S}$는 proposal의 contour mask이며 morphological gradient로 계산한다. $Q$는 class response map과 bias를 이용해 만든 background mask이다. $\alpha$, $\beta$는 validation set에서 정한 클래스 독립 파라미터이다.

각 항의 의미는 다음과 같다.

첫 번째 **instance-aware term** $\alpha \cdot R * S$는 PRM과 많이 겹치는 proposal에 높은 점수를 준다. 즉, 해당 proposal이 특정 peak가 가리키는 인스턴스를 잘 포함해야 한다.

두 번째 **boundary-aware term** $R * \hat{S}$는 PRM이 담고 있는 세밀한 경계 정보와 proposal contour가 얼마나 잘 맞는지를 본다. 이는 단순히 내부 겹침만 보는 것이 아니라 shape alignment를 고려한다.

세 번째 **class-aware term** $-\beta \cdot Q * S$는 해당 클래스와 무관한 영역을 억제한다. 즉, proposal이 background나 class-irrelevant region을 많이 포함하면 점수를 깎는다.

추론 알고리즘은 다음 흐름이다. 테스트 이미지와 proposal set이 주어지면 먼저 class response map을 구하고, 각 클래스 map에서 peak를 찾는다. 그런 다음 각 peak마다 peak back-propagation을 수행해 PRM을 만들고, 모든 proposal에 대해 위 score를 계산한다. 가장 점수가 높은 proposal을 그 peak의 instance mask로 선택한다. 마지막에는 Non-Maximum Suppression을 적용해 중복 prediction을 제거한다.

이 방법의 장점은 구조가 단순하다는 점이다. detection branch, mask decoder, recurrent refinement 같은 복잡한 모듈 없이도, classification network와 proposal retrieval만으로 instance segmentation을 수행한다.

## 4. 실험 및 결과

논문은 제안 방법을 여러 측면에서 평가한다. 첫째, peak stimulation이 실제로 localization을 향상시키는지 본다. 둘째, PRM이 instance-aware cue로서 품질이 높은지 측정한다. 셋째, semantic segmentation과 instance segmentation에 실제로 도움이 되는지 실험한다. 백본으로는 주로 ResNet50과 VGG16을 사용했다.

### 4.1 Peak Response Analysis

#### Pointwise localization

먼저 peak가 정말 객체 위치를 잘 가리키는지 확인하기 위해 pointwise localization metric을 사용한다. class response map을 bilinear interpolation으로 원본 이미지 크기로 올린 뒤, 예측된 각 클래스에 대해 최대 peak 좌표가 같은 클래스의 GT bounding box 안에 들어가면 true positive로 센다.

PASCAL VOC 2012와 MS COCO 2014 validation set에서의 mAP는 다음과 같다.

* DeepMIL: VOC 74.5, COCO 41.2
* WSLoc: VOC 79.7, COCO 49.2
* WILDCAT: VOC 82.9, COCO 53.5
* SPN: VOC 82.9, COCO 55.3
* Ours without Peak Stimulation: VOC 81.5, COCO 53.1
* **Ours full approach: VOC 85.5, COCO 57.5**

이 결과는 두 가지를 보여준다. 첫째, class peak 자체가 object cue로 쓸 수 있을 정도로 위치성을 가진다. 둘째, peak stimulation을 넣으면 baseline보다 큰 폭으로 향상된다. 따라서 peak stimulation은 단순한 부가 장치가 아니라 localization 성능 향상에 실질적으로 기여하는 핵심 구성요소라고 볼 수 있다.

#### Quality of PRMs

PRM이 실제 인스턴스 내부를 잘 포착하는지 보기 위해, GT mask $G$와 PRM $R$의 correlation을 다음과 같이 정의한다.

$$
\frac{\sum R \odot G}{\sum R}
$$

이 값은 PRM 에너지 중 얼마가 실제 해당 인스턴스 안에 들어가는지를 뜻한다. 각 PRM에 대해 동일 클래스 GT mask들 중 최대 correlation을 score로 사용하고, score가 0.5를 넘으면 true positive로 본다.

VOC 2012에서 different response aggregation strategy를 비교한 결과는 다음과 같다.

* CAM (Global Average Pooling): 55.7
* DeepMIL (Global Max Pooling): 60.9
* WILDCAT (Global Max-Min Pooling): 62.4
* **PRM / Peak Stimulation: 64.0**

즉, 저자들의 peak stimulation은 기존 global aggregation 전략보다 더 나은 instance-aware representation을 만든다.

또한 통계 분석에 따르면, 이미지에 객체가 하나만 있을 때 PRM 에너지의 평균 78%가 인스턴스 안에 들어갔고, 2개에서 5개 객체가 있을 때도 67%였다. 6개 이상의 crowded scene에서도 평균적으로 background보다 instance 쪽에 더 많은 에너지가 모였다. 이 결과는 crowded scene에서도 PRM이 어느 정도 인스턴스를 구분하는 경향이 있음을 시사한다. 객체 크기 분석에서는 common size object에서 특히 잘 작동한다고 보고한다.

### 4.2 Weakly Supervised Semantic Segmentation

제안 방법은 본래 instance-aware cue를 만들기 위한 것이지만, 같은 클래스의 instance mask를 합치면 semantic segmentation에도 사용할 수 있다. VOC 2012 segmentation validation set에서, 같은 클래스의 instance mask를 merge하여 semantic segmentation prediction을 만들고 mIoU를 측정했다.

결과는 다음과 같다.

* MIL+ILP+SP-seg: 42.0
* WILDCAT: 43.7
* SEC: 50.7
* Check mask: 51.5
* Combining: 52.8
* **PRM (Ours): 53.4**

저자들은 이 결과를 통해, 복잡한 반복 학습이나 추가 supervision 없이도 image-level label과 standard classification setting만으로 경쟁력 있는 semantic segmentation이 가능하다고 주장한다. 특히 표에서 PRM은 “object segment proposals”만 사용했고, CRF 후처리나 human-in-the-loop 없이도 가장 높은 수치를 보였다.

다만 여기서 주의할 점이 있다. 이 semantic segmentation 결과 역시 proposal quality에 영향을 받는다. 또한 논문은 VOC val 기준 수치만 제시하고 있어, test benchmark나 다른 데이터셋에서의 일반화까지는 본문만으로 단정하기 어렵다.

### 4.3 Weakly Supervised Instance Segmentation

이 논문의 가장 중요한 실험이다. 저자들은 VOC 2012 segmentation set에서 image-level supervision만으로 instance segmentation을 수행하고, 자신들이 알기로는 이것이 **image-level supervised instance segmentation 결과를 처음 보고한 사례**라고 주장한다.

평가 지표는 $mAP^r$ at IoU threshold 0.25, 0.5, 0.75와 ABO(Average Best Overlap)이다.

비교를 위해 weakly supervised localization 기반 baseline을 구성했다. bounding box를 얻은 뒤 세 가지 방식으로 mask를 만들었다.

첫 번째는 **Rect.** 로, bounding box 전체를 채운다.
두 번째는 **Ellipse.** 로, box 안에 최대 타원을 맞춘다.
세 번째는 **MCG.** 로, bounding box와 IoU가 가장 높은 MCG proposal을 선택한다.

주요 결과는 다음과 같다.

* MELM + Rect.: 36.0 / 14.6 / 1.9 / 26.4
* MELM + Ellipse: 36.8 / 19.3 / 2.4 / 27.5
* MELM + MCG: 36.9 / 22.9 / 8.4 / 32.9
* CAM + MCG: 20.4 / 7.8 / 2.5 / 23.0
* SPN + MCG: 26.4 / 12.7 / 4.4 / 27.1
* **PRM (Ours): 44.3 / 26.8 / 9.0 / 37.6**

여기서 순서는 각각 $mAP^r_{0.25}$, $mAP^r_{0.5}$, $mAP^r_{0.75}$, ABO이다.

이 결과는 제안 방법이 image-level label만 사용하는 설정에서 기존 weakly supervised localization 기반 방법보다 훨씬 강력하다는 점을 보여준다. 특히 낮은 IoU threshold뿐 아니라 높은 threshold인 0.75에서도 향상이 있다는 점은, 단지 대충 위치만 맞추는 것이 아니라 **세밀한 mask 품질**도 개선되었음을 의미한다. 저자들은 이를 peak stimulation이 localization을, peak back-propagation이 fine-detailed instance cue를 담당한 결과로 해석한다.

#### Ablation study

저자들은 peak stimulation과 proposal retrieval metric의 각 항이 실제로 중요한지 ablation study를 수행했다. 표 5는 형식이 다소 압축되어 있지만, 본문 설명에서 핵심 결론은 분명하다.

첫째, **peak stimulation을 제거하면 성능이 크게 떨어진다.** 저자들은 이것이 학습 단계에서 peak를 잘 드러나게 만드는 과정이 instance segmentation 성능에 핵심이라고 해석한다.

둘째, **instance-aware term을 제거하면 $mAP^r_{0.5}$가 26.8%에서 13.3%로 급락한다.** 이는 PRM이 제공하는 “well-isolated instance-aware representation”이 실제 성능의 중심이라는 뜻이다.

셋째, **boundary-aware term은 약 2.5%의 성능 향상**을 만든다. 이는 PRM이 실제로 경계 정보까지 담고 있으며, proposal contour와의 정합이 유효하다는 것을 뒷받침한다.

넷째, **class-aware term은 class-irrelevant region을 억제하여 성능을 크게 올린다.** 즉, instance cue만으로는 충분하지 않고, 클래스 레벨의 억제 정보가 함께 있어야 더 안정적인 retrieval이 가능하다.

#### Qualitative results

질적 결과에서 저자들은 여러 어려운 장면에서도 괜찮은 분리를 보여준다고 설명한다. 가까이 붙어 있거나 일부 가려진 인스턴스를 구분하는 사례, 서로 다른 크기의 객체를 다루는 사례, 다른 클래스 객체를 동시에 잘 분리하는 사례를 제시한다.

반면 failure case도 솔직히 언급한다. weak supervision 특성상 co-occurrence pattern에 오도될 수 있고, object part와 multiple object를 구분하지 못할 때가 있다. 예를 들어 동물의 머리나 몸통 일부가 하나의 instance처럼 강조될 수 있다. 저자들은 proposal retrieval로 이 문제를 완화하려 하지만, 결국 성능의 상한은 proposal quality에 의해 제한된다고 인정한다.

#### Upper bound analysis

표 6은 제안 방법의 잠재력을 더 잘 보여준다. proposal gallery를 바꿔가며 upper bound를 측정했다.

* 일반 MCG proposal gallery에서 PRM의 $mAP^r_{0.75}$는 9.0
* MCG + GT mask를 넣어 recall을 100%로 만들면 26.9
* GT mask를 완벽한 proposal gallery로 주면 73.3

이 결과는 흥미롭다. proposal이 충분히 좋아지면, image-level supervision으로 얻은 PRM의 retrieval 능력 자체는 상당히 강력할 수 있다는 뜻이다. 저자들은 이를 바탕으로 video나 RGB-D처럼 더 강한 proposal을 만들 수 있는 환경에서는 성능이 더 크게 향상될 가능성이 있다고 본다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법론의 조합이 매우 설득력 있다는 점이다. image-level supervised instance segmentation은 당시 거의 다뤄지지 않았던 어려운 문제인데, 저자들은 이를 복잡한 구조 대신 **peak라는 단순하고 해석 가능한 개념**으로 접근한다. “peak는 인스턴스 내부의 강한 단서다”라는 관찰은 직관적이지만, 이를 학습 단계의 peak stimulation과 추론 단계의 peak back-propagation으로 체계화한 점이 좋다.

또 다른 강점은 방법의 단순성이다. standard classification network를 FCN으로 바꾸고, cross entropy loss로 학습하며, 추가적인 dense annotation 없이도 작동한다. CRF, RNN, template matching 같은 복잡한 후처리에 크게 의존하지 않고도 성능을 내는 점은 실용적이다. 또한 point localization, semantic segmentation, instance segmentation까지 일관되게 성능 향상을 보였다는 점에서 방법의 일반성이 어느 정도 드러난다.

PRM의 해석 가능성도 장점이다. 많은 weakly supervised 방법이 saliency를 주지만, 이것이 왜 특정 인스턴스를 가리키는지 명확하지 않은 경우가 많다. 반면 이 논문은 특정 class peak를 기점으로 relevance를 추적하므로, PRM 하나가 특정 instance seed와 연결된다는 설명이 비교적 자연스럽다.

하지만 한계도 분명하다. 가장 큰 한계는 **proposal dependence**이다. 최종 instance mask는 외부 proposal gallery에서 선택되므로, proposal quality가 낮으면 성능이 제한된다. 저자들 스스로 upper bound analysis에서 이 점을 인정한다. 즉, PRM이 좋아도 proposal이 나쁘면 마스크 품질은 올라가기 어렵다.

또한 PRM이 진정한 의미의 완전한 instance mask를 직접 생성하는 것은 아니다. PRM은 어디까지나 informative cue 또는 seed에 가깝고, full object extent는 proposal prior에 의존해 복원한다. 따라서 end-to-end instance segmentation 모델과 비교하면 표현의 완결성은 부족할 수 있다.

실패 사례가 보여주듯, weak supervision의 전형적인 문제도 남아 있다. co-occurrence noise, part-level activation, crowded scene에서의 혼선 같은 문제가 있다. 논문은 crowded scene에서도 평균적으로 background보다 instance 쪽 에너지가 더 높다고 분석하지만, 이것이 곧 안정적인 분리를 보장한다는 뜻은 아니다. 특히 동일 클래스의 복잡한 중첩이나 심한 occlusion에서 얼마나 견고한지는 본문 결과만으로는 충분히 확인하기 어렵다.

비판적으로 보면, 이 방법의 성능 향상이 “정말로 instance representation을 배운 결과”인지, 아니면 “좋은 seed + proposal selection”의 조합인지도 구분해서 볼 필요가 있다. 논문은 둘 다 중요하다고 보지만, 최종 성능의 상당 부분이 proposal retrieval quality에 달려 있다는 점에서, 순수한 mask generation 능력은 제한적일 수 있다. 다만 이것은 논문의 약점이라기보다, 당시 weak supervision 환경에서 현실적인 설계 선택으로 보는 편이 더 공정하다.

## 6. 결론

이 논문은 image-level label만으로 instance segmentation을 수행하려는 매우 어려운 문제에 대해, **class peak response**라는 단순하지만 강력한 관찰을 기반으로 한 해법을 제시한다. 학습 단계의 **peak stimulation**은 네트워크가 informative receptive field에 집중하도록 만들어 localization 성능을 높이고, 추론 단계의 **peak back-propagation**은 특정 peak에 대응하는 fine-detailed하고 instance-aware한 visual cue인 **PRM**을 생성한다. 이후 PRM, class response map, proposal prior를 결합하여 인스턴스 마스크를 추출한다.

논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, class response map의 peak가 instance-level cue와 밀접하게 연결된다는 점을 실험적으로 보여주었다. 둘째, classification network 내부의 응답만으로 instance-aware representation을 끌어내는 구체적 방법을 제안했다. 셋째, weakly supervised point localization과 semantic segmentation뿐 아니라, image-level supervised instance segmentation에서도 유의미한 최초 수준의 결과를 보고했다.

실제 적용 관점에서 이 연구는 annotation cost가 큰 segmentation 문제를 더 약한 supervision으로 풀 수 있다는 가능성을 보여준다. 향후 연구 측면에서는 두 방향이 특히 중요해 보인다. 하나는 proposal dependence를 줄여 PRM으로부터 더 직접적으로 mask를 생성하는 방향이고, 다른 하나는 video, RGB-D, motion cue, stronger proposal generator 등을 결합해 weak supervision 환경에서도 더 안정적인 instance segmentation을 구현하는 방향이다. 이 논문은 완성형 해결책이라기보다는, **classification network 안에 이미 존재하는 instance cue를 어떻게 발견하고 활용할 것인가**라는 중요한 관점을 제시한 의미 있는 출발점으로 볼 수 있다.
