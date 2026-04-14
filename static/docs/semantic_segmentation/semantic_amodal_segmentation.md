# Semantic Amodal Segmentation

- **저자**: Yan Zhu, Yuandong Tian, Dimitris Metaxas, Piotr Dollár
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1509.01329

## 1. 논문 개요

이 논문은 기존의 semantic segmentation보다 한 단계 더 어려운 인식 문제로서 **semantic amodal segmentation**을 제안한다. 핵심은 이미지에서 보이는 부분만 분할하는 것이 아니라, **가려져 보이지 않는 부분까지 포함한 전체 객체/영역의 범위**를 annotation하는 것이다. 저자들은 각 이미지에서 salient한 region을 모두 표시하고, 각 region의 semantic label을 붙이며, 서로 겹치는 영역들 사이의 **partial depth order**까지 기록한다. 그 결과 visible part와 occluded part, visible edge와 hidden edge, figure-ground 관계, object overlap까지 포함하는 풍부한 scene structure가 만들어진다.

연구 문제는 크게 두 가지다. 첫째, 이런 amodal annotation이 실제로 **일관되게 정의 가능한가**, 즉 서로 다른 annotator들이 비슷한 결과를 내는가이다. 둘째, 이 annotation을 바탕으로 **amodal mask prediction**과 **depth ordering** 같은 새로운 computer vision task를 학습하고 평가할 수 있는가이다. 저자들은 이 문제를 위해 BSDS 500장과 COCO 5000장에 대해 새로운 dataset을 구축하고, 일관성 분석, 평가 지표, 그리고 baseline 모델을 함께 제시한다.

이 문제가 중요한 이유는, 사람이 시각적으로 매우 자연스럽게 수행하는 능력인 **amodal perception**을 기계가 아직 충분히 다루지 못하고 있기 때문이다. 기존의 classification, detection, modal segmentation은 대체로 보이는 신호에 강하게 의존한다. 반면 amodal segmentation은 객체 상호작용, 가림(occlusion), 장면 기하(scene geometry), 전체 형태에 대한 추론을 요구한다. 따라서 저자들은 이를 미래 시각 인식의 중요한 frontier 중 하나로 본다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **“장면을 보이는 픽셀의 집합으로만 보지 말고, 물리적으로 존재하는 전체 영역의 구조로 표현하자”**는 것이다. 이를 위해 저자들은 각 region을 modal하게가 아니라 amodal하게 annotation하고, 동시에 depth ordering을 명시하도록 설계했다. 이 설계 덕분에 단순히 마스크만 얻는 것이 아니라, 어떤 경계가 foreground 소유인지, 어떤 부분이 가려졌는지, 어떤 영역이 다른 영역 앞에 있는지까지 추론 가능한 표현이 된다.

기존 접근과의 차별점은 세 가지다. 첫째, 기존 BSDS 같은 dataset은 주로 visible boundary 중심의 modal annotation이며, semantic constraint가 약해서 annotator 간 일관성이 낮았다. 이 논문은 **semantic naming**, **dense annotation**, **depth ordering**, **shared edge marking**이라는 명확한 규칙을 도입해 annotation consistency를 높였다. 둘째, 단순 object detection과 달리 bounding box 수준이 아니라 **dense region annotation**을 제공한다. 셋째, 기존 amodal 관련 연구들이 일부 알고리즘이나 제한된 설정에 머무른 것과 달리, 이 논문은 **대규모 자연 이미지 dataset + metric + baseline**을 한 번에 제시한다.

또 하나 중요한 직관은 **amodal segment가 modal segment보다 더 단순한 형태를 갖는 경우가 많다**는 점이다. visible mask는 가림 때문에 들쭉날쭉하고 scene geometry의 영향을 받지만, amodal mask는 객체 본래의 전체 형태를 반영하므로 더 smooth하고 simple한 shape를 가질 수 있다. 논문은 실제 통계로도 이를 확인한다.

## 3. 상세 방법 설명

전체 시스템의 출발점은 annotation protocol이다. 저자들은 Open Surfaces annotation tool을 확장해서 semantic amodal segmentation에 맞는 도구를 만들었다. annotator는 각 region에 대해 polygon을 그리고, 이름을 붙이고, depth order를 정하고, 필요한 경우 shared edge를 명시한다. 이 과정에서 다음 네 가지 규칙이 핵심이다.

첫째, **semantically meaningful region만 annotation**한다. 즉 이름 붙일 수 있는 region만 남기고, 단순 material boundary나 object interior edge는 기본적으로 분리하지 않는다. 저자들은 이 제약이 annotation granularity를 안정시키고, occluded part에 대한 annotator의 prior를 더 일관되게 만든다고 본다.

둘째, **dense annotation**이다. 이미지 안의 foreground object를 가능한 한 빠짐없이 표시해야 하며, 어떤 region이 가려져 있으면 그 region을 가리는 occluder도 함께 annotation해야 한다. 이렇게 해야 visible/occluded portion과 hidden/visible edge를 구조적으로 복원할 수 있다.

셋째, **depth ordering**이다. 겹치는 두 region이 있으면 앞의 영역이 뒤의 영역보다 먼저 오도록 순서를 준다. 애매한 경우에는 edge가 가장 자연스럽게 “render”되도록 순서를 정한다. 이 정보는 장면의 occlusion 관계를 명시적으로 담는 핵심 변수다.

넷째, **edge sharing**이다. 두 인접 region이 단순히 맞닿아 있을 뿐 가림 관계가 아닌 경우, 그 경계는 shared edge로 표시한다. 이는 duplicate edge를 피하고, figure-ground가 없는 경계를 따로 다룰 수 있게 한다.

논문은 corner case도 명시한다. hole은 내부를 비우지 않고 exterior boundary만 annotation하며, blurry background는 가장 salient한 것만 표시한다. depth ordering이 얽혀 명확하지 않은 경우는 “least wrong”한 순서를 고르거나 object part를 따로 나누어 annotation한다. 비슷한 객체들의 집합은 하나의 group region으로 묶을 수 있다. 또한 image boundary 바깥까지는 amodal completion하지 않는다. 즉, **image 밖으로 잘려 나간 truncation은 복원 대상이 아니다**.

데이터 통계 측면에서 BSDS에서는 이미지당 5~7명의 annotator가 붙었고, annotation당 평균 7.3개의 region, region당 평균 64개의 polygon point가 기록되었다. 약 84%의 픽셀이 적어도 하나의 region에 의해 덮이며, region의 62%가 부분 가림 상태이고 평균 occlusion rate는 21%였다. COCO는 더 복잡해서 image당 region 수는 많고 pixel coverage는 조금 낮다.

논문은 amodal shape가 단순하다는 점을 두 지표로 정량화한다. segment를 $S$라 할 때,

$$
\text{convexity}(S) = \frac{\text{Area}(S)}{\text{Area}(\text{ConvexHull}(S))}
$$

로 정의한다. 값이 1에 가까울수록 shape가 convex hull에 가깝고 더 단순하다.

또한 simplicity는

$$
\text{simplicity}(S) = \frac{\sqrt{4\pi \cdot \text{Area}(S)}}{\text{Perimeter}(S)}
$$

로 정의한다. 논문 본문 표기상 분모에 perimeter가 직접 들어가며, 원에 가까울수록 값이 커진다. 두 지표 모두 값이 클수록 단순한 모양을 뜻한다. 실험에서는 amodal segment가 modal segment보다 이 수치가 전반적으로 더 높게 나왔다.

일관성 분석에서 region consistency는 IoU 기반 bipartite matching으로 계산한다. IoU가 0.5 이상인 region끼리 매칭한 뒤 precision과 recall을 구하고,

$$
F = \frac{2PR}{P+R}
$$

로 요약한다. 이를 annotator 쌍마다 계산해 annotation consistency를 본다. 논문의 핵심 결과 중 하나는 **amodal region consistency가 기존 BSDS의 modal annotation consistency보다 훨씬 높다**는 점이다.

이후 저자들은 COCO 5000장으로 baseline 모델을 학습한다. amodal segmentation baseline은 세 종류다.

첫째, **DeepMask**와 **SharpMask**를 modal segmentation baseline으로 사용한다. 이들은 원래 visible object mask proposal용 모델이다.

둘째, **ExpandMask**다. 이 모델은 SharpMask가 예측한 modal mask와 이미지 patch를 입력받아, 그 mask를 바깥으로 확장하여 amodal mask를 출력하도록 학습된다. 아이디어는 “보이는 마스크가 주어졌을 때 가려진 부분을 채워 넣자”는 것이다.

셋째, **AmodalMask**다. 이 모델은 image patch로부터 직접 amodal mask를 예측한다. 구조는 SharpMask와 거의 동일하지만 목적 함수와 학습 데이터가 amodal용으로 바뀐다. 논문은 ExpandMask와 AmodalMask가 SharpMask와 같은 codebase 위에서 구현되었고, COCO modal pretrained weight에서 initialization한 뒤 amodal dataset으로 fine-tuning했다고 설명한다. synthetic overlay로 만든 가짜 amodal data를 쓴 변형도 실험했지만, 실제 annotation 데이터가 더 좋았다.

평가 지표로 amodal segmentation은 COCO proposal 평가에서 쓰이는 **Average Recall (AR)**를 확장해 사용한다. IoU threshold 0.5~0.95에서 recall을 평균내고, IoU는 modal mask가 아니라 **amodal mask와의 IoU**로 계산한다. 또한 occlusion level $q$에 따라 `none ($q=0$)`, `partial ($0<q\le .25$)`, `heavy ($q>.25$)`로 나누어 성능을 본다.

depth ordering 쪽은 전체 scene graph 대신 **pairwise depth ordering**을 평가한다. 겹치는 두 mask가 있을 때 어느 쪽이 앞에 있는지 맞히는 binary classification이다. ground truth mask마다 IoU 0.5 이상으로 매칭되는 예측 mask를 찾고, 그 매칭된 쌍에 대해서만 ordering accuracy를 잰다. baseline으로는 area heuristic, y-axis heuristic, 그리고 세 가지 neural network가 있다.

- **OrderNet B**: 두 bounding box를 입력으로 받는 3-layer MLP
- **OrderNet M**: 두 mask를 입력으로 받는 ResNet50 기반 모델
- **OrderNet M+I**: 두 mask와 image patch를 함께 입력으로 받는 ResNet50 기반 모델

입력 순서에 따른 편향을 줄이기 위해 두 순서로 inference한 결과를 평균낸다.

## 4. 실험 및 결과

실험은 크게 dataset consistency 분석과 baseline 성능 평가로 나뉜다.

먼저 BSDS 기반 consistency 분석에서, amodal annotation은 기대 이상으로 잘 정의되는 작업임을 보였다. region consistency의 median $F$ score는 기존 BSDS modal annotation이 0.425인데 비해, 제안한 amodal annotation은 0.723이었다. 저자들이 따로 언급하듯, 같은 annotation에서 visible portion만 놓고 본 modal consistency는 0.756으로 더 높다. 즉, 보이지 않는 부분까지 포함했는데도 consistency가 크게 유지되며, 오히려 기존 자유로운 modal annotation보다 훨씬 안정적이었다.

edge consistency도 더 높았다. visible edge 기준 median consistency는 기존 BSDS가 0.728, 제안 데이터가 0.795였다. 다만 이 dataset은 semantic boundary 중심이라 original BSDS보다 edge density는 낮다. 이는 interior edge나 material boundary를 덜 포함하기 때문이다. 하지만 이 점이 오히려 더 semantic한 edge detector 학습에 도움이 될 수 있음을 저자들은 보였다.

실제로 edge detection cross-dataset 실험에서 Structured Edges(SE)와 HED를 평가했다. SE는 제안 데이터로 학습했을 때 원래 BSDS test에서도 약간 더 나은 성능을 보였다. HED는 같은 train/test annotation 체계에서 가장 잘 나왔지만, 전반적으로 제안 데이터가 유효한 edge training set임은 확인되었다. 특히 중요한 결과는 **인간과 기계의 격차**다. original BSDS에서는 HED ODS 0.79, human F-score 0.81로 차이가 0.02밖에 안 났다. 반면 제안 데이터에서는 HED가 0.69, human이 0.90으로 격차가 크게 벌어졌다. 저자들은 이를 통해 새 dataset이 state-of-the-art에게 여전히 어려운 과제를 제공한다고 해석한다.

amodal segmentation baseline 결과는 COCO validation에서 보고된다. 전체 AR 기준으로 DeepMask는 0.378, SharpMask는 0.396이었다. 반면 ExpandMask는 0.417, AmodalMask는 0.434였다. 즉 amodal-aware training이 확실한 성능 향상을 가져왔다. 특히 heavy occlusion에서 차이가 크다. 예를 들어 전체 영역에 대한 $AR_H$는 SharpMask가 0.242인데, ExpandMask는 0.327, AmodalMask는 0.364였다. 이는 가려짐이 심할수록 modal baseline이 급격히 불리해지고, amodal 모델이 진짜로 occluded part를 예측하고 있음을 보여준다.

things와 stuff를 나눠 봐도 비슷한 경향이 있다. 예를 들어 things only에서 AmodalMask의 전체 AR은 0.458로 SharpMask의 0.448보다 높고, heavy occlusion에서는 0.376 대 0.275로 격차가 더 분명하다. stuff only에서도 AmodalMask가 0.414로 가장 높다. 다만 저자들은 **unoccluded object에서는 amodal 모델이 오히려 over-predict할 수 있다**고 정성적으로 지적한다. 실제 Figure 10 마지막 예시에서 ExpandMask가 지나치게 확장하는 사례가 나온다.

depth ordering에서는 단순 heuristic도 제법 강하다. area 기준은 약 0.696, y-axis 기준은 약 0.711 정확도를 보였다. 그러나 learned model이 더 좋다. generated mask 위에서 OrderNet M+I는 SharpMask mask 기준 0.793, ExpandMask 기준 0.802, AmodalMask 기준 0.814 accuracy를 기록했다. ground truth mask를 사용할 경우는 0.869에서 0.883까지 오른다. 즉, **mask 품질이 좋아질수록 ordering도 더 쉬워진다**는 점이 명확하다. 또한 image patch까지 넣은 OrderNet M+I가 mask만 쓰는 OrderNet M보다 consistently 더 좋다. 이는 depth ordering이 단순 shape 비교가 아니라 texture, context, occlusion cue를 함께 활용하는 문제임을 시사한다.

부록에서는 COCO edge detection 결과도 추가로 제시한다. COCO는 이미지당 annotator 수가 1명이라 metric상 ODS가 낮게 보일 수 있지만, 실제로는 annotator 수 차이의 영향이 크다고 설명한다. HED는 COCO val에서 ODS 0.609, SE는 0.524를 기록했다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의, dataset, consistency validation, metric, baseline**을 한 논문 안에서 매우 완결성 있게 제시했다는 점이다. 단순히 “새로운 annotation을 해봤다” 수준이 아니라, 그 annotation이 실제로 well-posed한지 BSDS 다중 annotator 실험으로 검증했고, 이후 COCO 대규모 버전으로 학습 가능한 benchmark까지 만들었다. 특히 amodal annotation이 오히려 기존 modal BSDS보다 더 일관적이라는 결과는 꽤 설득력이 크다.

또 다른 강점은 annotation 설계의 품질이다. semantic naming, dense coverage, depth ordering, shared edge라는 네 규칙은 단순한 작업 지침처럼 보이지만, 실제로 annotation ambiguity를 줄이고 scene reasoning을 유도하는 핵심 장치로 작동한다. 또한 이 dataset이 amodal segmentation뿐 아니라 edge detection, figure-ground labeling, modal segmentation 연구에도 활용될 수 있게 설계된 점도 가치가 크다.

baseline도 당시 기준으로 합리적이고 강하다. 특히 ExpandMask와 AmodalMask를 통해 “modal mask를 확장할 것인가”와 “직접 amodal mask를 예측할 것인가”라는 두 설계를 모두 실험한 점이 좋다. depth ordering 역시 heuristic에서 deep model까지 폭넓게 비교해 문제의 난이도와 가능성을 함께 보여준다.

한계도 분명하다. 첫째, COCO 5000장 annotation은 이미지당 **한 명의 expert annotator**만 붙는다. 품질 관리가 있었다고는 하지만, BSDS처럼 다중 annotator consistency를 COCO 규모에서 직접 측정하지는 못했다. 둘째, semantic label은 open vocabulary라서 수집했지만, **평가에는 직접 사용하지 않는다**. 따라서 semantic amodal segmentation이라는 이름에 비해 semantic label 활용은 아직 제한적이다. 셋째, truncation은 다루지 않으므로 이미지 밖으로 이어지는 객체 전체를 복원하는 진정한 full-scene completion 문제와는 차이가 있다.

또한 depth ordering 평가는 pairwise ordering으로 제한된다. 이는 현실적인 선택이지만, 전체 장면의 전역적 3D 구조나 multi-object relational consistency를 직접 평가하지는 않는다. 더 나아가 baseline들은 패치 기반 혹은 proposal 기반이어서 scene-wide reasoning을 본격적으로 수행한다고 보기 어렵다. 저자들도 인정하듯, 이 dataset은 기존 limited receptive field 방식의 모델에 도전적인 과제를 던진다.

비판적으로 보면, annotation consistency가 높다는 결과는 partly annotation guideline이 강하게 제약되었기 때문이기도 하다. 이것은 장점이지만 동시에 “자연 장면의 가능한 모든 해석”을 포착한다기보다 “정해진 기준에 맞춘 안정적 표기”에 가깝다. 다만 benchmark dataset의 목적을 생각하면 이는 오히려 합리적인 trade-off로 볼 수 있다.

## 6. 결론

이 논문은 computer vision에서 **amodal perception을 본격적인 학습/평가 대상으로 끌어올린 초기의 중요한 작업**이다. 저자들은 semantic amodal segmentation이라는 문제를 명확히 정의하고, BSDS와 COCO 기반 dataset을 구축했으며, region consistency와 edge consistency 분석을 통해 annotation의 타당성을 입증했다. 또한 amodal segment quality와 pairwise depth ordering이라는 구체적 평가 지표를 제안하고, DeepMask/SharpMask 확장형 baseline과 OrderNet 계열 baseline으로 출발점을 제공했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, 보이지 않는 부분까지 포함하는 dense scene annotation 체계를 제시했다. 둘째, 이 과제가 실제로 일관되게 annotation 가능함을 보였다. 셋째, occlusion reasoning과 scene structure understanding을 요구하는 새로운 benchmark를 만들었다.

이 연구는 실제 적용 측면에서도 의미가 크다. 로보틱스, 자율주행, scene understanding, manipulation처럼 **부분 가림 속에서 객체의 전체 형태를 추정해야 하는 문제**에 직접 연결될 수 있기 때문이다. 또한 후속 연구에서는 더 강한 global reasoning 모델, open-vocabulary semantics 활용, image-outside completion, 3D-aware occlusion modeling으로 이어질 수 있다. 논문 자체도 이를 완전히 해결했다고 주장하지 않으며, 오히려 앞으로의 연구 방향을 여는 benchmark paper로서의 가치가 크다.
