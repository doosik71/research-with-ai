# ISTR: End-to-End Instance Segmentation with Transformers

이 논문은 instance segmentation을 **진정한 end-to-end 방식**으로 풀기 어렵다는 문제에서 출발합니다. object detection에서는 DETR류 방법처럼 bipartite matching 기반의 set loss를 도입해 NMS를 제거할 수 있었지만, instance segmentation은 출력 차원이 훨씬 커서 같은 접근이 바로 통하지 않습니다. 저자들은 이 병목이 “mask 자체를 직접 예측하고 매칭하려는 것”에 있다고 보고, **고차원 mask 대신 저차원 mask embedding을 예측**하는 Transformer 기반 프레임워크 **ISTR**를 제안합니다. 이 방법은 예측된 box, class, mask embedding을 집합(set)으로 직접 출력하고, 이를 ground truth와 bipartite matching으로 연결해 학습합니다. 또한 recurrent refinement를 통해 detection과 segmentation을 동시에 정제하며, 추론 시에는 **NMS 없이 결과를 바로 사용**합니다. 논문은 이 구조가 COCO에서 ResNet50-FPN 기준 46.8 box AP / 38.6 mask AP, ResNet101-FPN 기준 48.1 / 39.9를 달성했다고 보고합니다.

## 1. Paper Overview

이 논문이 해결하려는 핵심 문제는 “왜 object detection에서는 end-to-end가 잘 되는데, instance segmentation에서는 잘 안 되는가”입니다. 저자들에 따르면 기존 instance segmentation 방법은 크게 두 부류의 비-end-to-end 요소를 가집니다. 첫째, top-down 계열은 detector가 proposal을 만들고, 이후 RoI 기반 mask head가 이를 분할하는 식으로 여러 단계를 거칩니다. 둘째, 많은 방법이 중복 예측 제거를 위해 NMS 같은 handcrafted post-processing을 사용합니다. 이런 구성은 학습과 추론을 완전히 통합하기 어렵게 만듭니다.

논문은 단순히 detection용 set prediction loss를 segmentation에 붙이는 실험이 실패한다는 점을 먼저 보여줍니다. 그 이유로 저자들은 **mask head 학습 샘플 부족**을 지목합니다. box는 좌표 몇 개면 되지만 mask는 보통 $28 \times 28$ 이상의 고차원 출력을 가지므로 더 많은 학습 샘플이 필요합니다. 그런데 bipartite matching을 거친 후 실제로 각 이미지에서 학습에 쓰이는 positive sample 수는 매우 적고, COCO에서는 평균 ground truth 수가 7.7개 수준이라 충분한 supervision이 되지 않습니다. Mask R-CNN이 RoI feature를 위해 512 proposal을 쓰는 것과 비교하면, end-to-end matching 기반 방식은 mask 학습 측면에서 훨씬 희소한 샘플만 얻게 됩니다.

이 문제가 중요한 이유는, instance segmentation이 detection보다 더 풍부한 출력 구조를 갖는 대표적인 dense recognition 문제이기 때문입니다. 이 문제를 truly end-to-end로 풀 수 있다면 NMS 제거, 중간 heuristic 축소, unified optimization 같은 장점이 생기고, 장기적으로는 instance-level recognition 전반의 설계 방향을 바꿀 수 있습니다. 저자들은 ISTR를 이런 맥락에서 **instance segmentation용 첫 Transformer 기반 end-to-end 프레임워크**로 제시합니다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 매우 선명합니다.

> **고차원 binary mask를 직접 예측하지 말고, 저차원 mask embedding을 예측하자.**

저자들은 자연스러운 object mask들이 임의의 $28 \times 28$ binary pattern 전체에 고르게 퍼져 있는 것이 아니라, 훨씬 더 낮은 차원의 manifold 근처에 놓여 있다고 가정합니다. 실제로 PCA를 적용해보면 상위 몇 개 component만으로도 mask 정보의 대부분을 설명할 수 있다고 보고합니다. 이 관찰이 ISTR의 출발점입니다. 즉, 모델이 직접 복잡한 픽셀 마스크를 맞추는 대신, 먼저 **압축된 embedding 공간**으로 예측하게 하면 학습 난이도를 크게 낮출 수 있다는 것입니다.

이 embedding 기반 설계는 단순한 압축 이상의 의미를 가집니다. 가장 중요한 점은 **bipartite matching cost를 mask embedding 공간에서 정의할 수 있게 된다**는 것입니다. detection의 set prediction이 잘 동작하는 이유는 예측과 정답을 one-to-one로 깔끔하게 매칭할 수 있기 때문인데, raw mask는 고차원이라 이 매칭이 불안정하고 샘플 효율도 낮습니다. 반면 embedding 공간에서는 cosine similarity나 L1/L2 기반 유사도로 예측 mask와 ground truth mask를 비교할 수 있어, end-to-end set loss 구성이 실용적이 됩니다.  

또 하나의 핵심은 **recurrent refinement**입니다. ISTR는 bounding box, class, mask embedding을 한 번에 끝내지 않고, learnable query box를 반복적으로 업데이트하며 여러 stage에 걸쳐 refinement합니다. 이 구조 덕분에 detection과 segmentation이 분리된 하위 문제로 쪼개지지 않고, 하나의 recurrent process 안에서 함께 개선됩니다. 저자들은 이것이 기존의 top-down, bottom-up 프레임워크와 다른 “세 번째 관점”이라고 주장합니다.  

## 3. Detailed Method Explanation

### 3.1 전체 프레임워크

ISTR의 전체 파이프라인은 backbone CNN과 FPN으로 시작합니다. 입력 이미지는 feature pyramid로 변환되고, 여기서 **learnable query box**를 이용해 RoIAlign으로 query-specific feature를 뽑습니다. 동시에 전역 image feature도 feature maps를 sum/average하여 얻습니다. 이후 **dynamic attention을 사용하는 Transformer encoder**가 RoI feature와 image feature를 결합하고, 그 위에서 class, box, mask embedding prediction head가 동작합니다. 그리고 이 예측 결과는 여러 stage에 걸쳐 recurrent refinement됩니다. 추론 시에는 최종 예측 집합을 그대로 사용하며, 따로 NMS를 수행하지 않습니다.

흥미로운 점은, 이 논문이 “Transformer 기반 end-to-end”라고 하면서도 DETR처럼 완전히 box-free query sequence로 시작하는 것은 아니라는 점입니다. ISTR는 여전히 **learnable query box + RoIAlign**을 사용합니다. 따라서 구조적으로는 RoI 정보를 버린 것이 아니라, 그것을 recurrent query mechanism과 Transformer fusion 안으로 재배치한 형태에 가깝습니다. 이 점은 후대의 query-based segmentation과 비교할 때 ISTR의 역사적 위치를 이해하는 데 중요합니다. 이 해석은 논문 프레임워크 설명에 근거한 것입니다.

### 3.2 Mask Embedding formulation

저자들은 mask embedding을 “원래 mask를 잘 복원할 수 있는 저차원 표현”으로 정의합니다. 이를 위해 원본 mask $\boldsymbol{M}$와 복원된 mask $f(g(\boldsymbol{M}))$ 사이의 mutual information을 최대화하는 관점에서 출발합니다.

$$
\max \mathcal{I}\big(\boldsymbol{M}, f(g(\boldsymbol{M}))\big)
$$

여기서 $g(\cdot)$는 mask encoder, $f(\cdot)$는 decoder입니다. 이후 이 목적은 reconstruction loss 형태로 정리됩니다. 각 마스크 $\boldsymbol{m}_i$에 대한 embedding $\boldsymbol{r}_i = g(\boldsymbol{m}_i)$를 사용해,

$$
\min \sum_{i=1}^{n} \left| \boldsymbol{m}_i - f(\boldsymbol{r}_i) \right|_2^2
$$

를 최소화하는 문제로 바꿉니다. 논문은 encoder/decoder를 단순한 선형 변환으로 두면 이 문제가 결국 **PCA와 같은 형태**가 되고, closed-form solution으로 mask embedding basis를 얻을 수 있다고 설명합니다. 즉, 이 논문은 복잡한 learned autoencoder 대신 **PCA 기반 mask dictionary**만으로도 충분하다고 주장합니다.  

이 부분이 ISTR의 매우 중요한 철학입니다. 보통 end-to-end segmentation이라면 mask decoder 자체도 learned module이어야 할 것 같지만, 저자들은 오히려 **suboptimal한 PCA embedding조차도 충분히 강력하다**고 보여줍니다. 이는 모델의 성능 핵심이 완벽한 mask representation보다는, embedding 기반 set prediction이라는 전체 학습 구조에 있다는 뜻으로 읽힙니다.

### 3.3 Matching cost와 set loss

ISTR는 detection과 마찬가지로 예측 집합과 ground truth 집합 사이에 bipartite matching을 수행합니다. 차이는 mask 항의 비교가 raw mask가 아니라 **embedding similarity**로 이루어진다는 점입니다. 논문 ablation에 따르면 mask cost로 단순 dice loss를 쓰는 것보다, embedding 사이 **cosine similarity**를 매칭 비용으로 사용하는 것이 더 좋은 결과를 냅니다. L1 loss 기반 embedding cost도 약간의 향상을 주지만, cosine similarity가 가장 효과적이었다고 보고합니다.

loss 측면에서는 pixel-level dice loss와 embedding-level L2 loss를 함께 쓰는 조합이 가장 좋습니다. dice loss만 사용하면 성능이 낮고, embedding L2만 사용하면 원래 mask reconstruction이 다소 suboptimal해지는 문제가 있습니다. 결국 **embedding 공간 제약과 픽셀 공간 제약을 동시에 주는 것**이 최선이라는 결론입니다. 이는 ISTR가 단순히 latent vector regression만 하는 것이 아니라, latent와 pixel 두 공간을 함께 정렬하는 하이브리드 학습 구조라는 뜻입니다.

### 3.4 Recurrent refinement의 역할

논문은 query box를 각 stage에서 이전 예측 box로 업데이트해 나갑니다. 이 반복 과정을 통해 detection과 segmentation이 동시에 정제됩니다. 전통적 top-down 접근에서는 먼저 box를 찾고 그다음 mask를 예측하지만, ISTR에서는 class, box, mask embedding이 한 stage의 출력이면서 동시에 다음 stage의 입력 조건을 개선합니다. 저자들은 이것이 detection과 segmentation을 jointly 처리할 때 성능 향상을 가져온다고 봅니다.  

이 recurrent design은 단순 반복 추론이 아니라, ISTR가 기존 instance segmentation taxonomy를 벗어나는 핵심 요소입니다. 논문 스스로도 이를 “top-down과 bottom-up과 다른 새로운 방식”이라고 설명합니다. 즉, ISTR는 DETR 스타일 one-shot set prediction과 전통적 RoI 기반 refinement 사이의 중간 지점에서, box-conditioned iterative prediction을 통해 instance segmentation을 end-to-end화한 구조라고 볼 수 있습니다.

### 3.5 Dynamic attention

RoI feature와 global image feature를 융합하는 attention 모듈도 중요한 설계입니다. ablation에서 저자들은 일반 multi-head attention 대신 **dynamic attention**을 쓰는 것이 훨씬 효과적이라고 보고합니다. 이는 ISTR가 단순히 Transformer를 붙인 것이 아니라, query-specific fusion 방식까지 별도로 설계했다는 뜻입니다. 다시 말해 성능은 “Transformer”라는 이름 자체보다, **RoI와 image context를 어떤 식으로 결합하느냐**에 더 민감합니다.

## 4. Experiments and Findings

### 4.1 데이터셋과 평가

실험은 MS COCO에서 수행됩니다. train2017(118K)으로 학습하고, ablation은 val2017(5K)에서 수행하며, 최종 결과는 test-dev에서 보고합니다. 평가지표는 COCO 표준 AP 계열입니다. 즉, 이 논문은 소규모 dataset이나 제한적 benchmark가 아니라, 당시 instance segmentation의 대표 표준 벤치마크에서 평가되었다는 점이 중요합니다.

### 4.2 주요 성능

논문 abstract와 소개부가 강조하듯, ISTR는 “bells and whistles 없이도” 당시 SOTA와 대등하거나 경쟁력 있는 수준의 성능을 냅니다. 구체적으로 COCO test-dev에서:

* ResNet50-FPN: **46.8 box AP / 38.6 mask AP**, 13.8 fps
* ResNet101-FPN: **48.1 box AP / 39.9 mask AP**, 11.0 fps

를 보고합니다. 저자들은 이를 근거로 ISTR가 **instance-level recognition을 위한 강력한 baseline**이라고 주장합니다.

### 4.3 Ablation에서 드러난 핵심 사실

가장 중요한 ablation 메시지는 세 가지입니다.

첫째, **raw mask 예측보다 mask embedding 예측이 낫다**는 점입니다. 논문은 detection용 set prediction을 segmentation에 그대로 적용하면 성능이 낮다는 문제를 출발점으로 삼았고, embedding representation이 그 해결책임을 보였습니다.  

둘째, **cosine similarity 기반 mask matching cost가 효과적**이라는 점입니다. 단순히 dice cost를 매칭에 넣는 것은 기대만큼 이득이 없었고, embedding 사이 cosine similarity가 가장 좋은 매칭 품질을 제공했습니다. 이는 ISTR의 성공이 mask embedding의 존재뿐 아니라, **그 embedding을 matching에 어떻게 쓰느냐**에 달려 있음을 보여줍니다.

셋째, **dice loss와 embedding L2 loss를 함께 쓰는 것이 가장 좋다**는 점입니다. embedding만 잘 맞춘다고 충분하지 않고, 실제 복원된 mask 품질을 유지하려면 pixel-level 제약도 필요합니다. 이 결과는 latent objective와 output-space objective가 상보적이라는 점을 말해줍니다.

또한 dynamic attention이 일반 multi-head attention보다 훨씬 좋았고, 저자들은 이를 통해 RoI/image feature fusion 설계의 중요성을 강조합니다.

### 4.4 기존 방법과의 비교 해석

논문은 ISTR가 SOLOv2, MEInst 같은 방법들과 비교해 경쟁력이 높다고 설명합니다. 특히 mask embedding을 사용한 선행 작업인 MEInst도 존재하지만, MEInst는 dense prediction 위치마다 중복된 mask embedding을 생성하므로 성능이 떨어집니다. 반면 ISTR는 set prediction과 bipartite matching을 결합함으로써 **redundant prediction 문제를 구조적으로 줄입니다**. 또 SOLOv2 대비 상당한 이득이 보고되는데, 저자들은 bipartite matching cost가 작은 객체를 훈련에서 배제하지 않는 점이 이유 중 하나라고 해석합니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **instance segmentation에서 end-to-end 학습을 실질적으로 가능하게 만든 첫 설득력 있는 프레임워크**라는 점입니다. 이전에도 recurrent neural network를 이용한 초기 시도는 있었지만, 소규모 데이터셋에서만 평가되었고 현대 baseline과 경쟁하지 못했습니다. ISTR는 COCO에서 강한 수치와 함께, end-to-end instance segmentation이 실제로 가능하다는 것을 보여줍니다.  

둘째, 아이디어가 깔끔합니다. “mask는 저차원 manifold에 있다”는 가정에서 출발해 PCA embedding, embedding-based matching, set loss, recurrent refinement로 이어지는 전체 논리 흐름이 일관적입니다. 복잡한 heuristic 추가보다 representation 문제를 다시 정의해 돌파했다는 점에서 연구적으로 가치가 큽니다.

셋째, detection과 segmentation을 **jointly** 처리한다는 점도 중요합니다. 저자들은 multi-task 학습이 성능 향상을 준다고 직접 언급하고 있고, recurrent refinement가 이 joint processing을 자연스럽게 구현합니다.  

### 한계

논문이 직접 “limitations” 절을 크게 두고 있지는 않지만, 내용상 몇 가지 한계는 분명합니다.

첫째, 엄밀한 의미에서 “완전한 DETR류 end-to-end”와는 다릅니다. ISTR는 여전히 **RoIAlign과 query box**를 사용합니다. 즉, NMS는 제거했지만 proposal-like geometric prior와 RoI cropping 자체는 남아 있습니다. 따라서 end-to-end라는 주장에는 분명한 진전이 있지만, 후대의 fully query-based mask models와 비교하면 과도기적 성격도 있습니다. 이 평가는 논문 구조 설명에 근거한 해석입니다.

둘째, mask embedding이 PCA의 closed-form solution으로도 충분하다고 말하지만, 이것은 동시에 **mask manifold 모델링이 비교적 단순하다**는 뜻이기도 합니다. 실제 복잡한 object shape variation을 PCA로 얼마나 잘 포괄할 수 있는지는 클래스나 데이터셋에 따라 제한이 있을 수 있습니다. 논문은 이 점을 치밀하게 파고들기보다는 “충분히 잘 된다”는 실험 결과를 보여주는 데 집중합니다.

셋째, 논문이 성공의 이유를 mask embedding과 end-to-end set prediction으로 설명하지만, 실제 성능은 recurrent refinement, dynamic attention, RoI/image fusion 등 여러 설계의 결합 효과입니다. 따라서 “무엇이 가장 본질적인 기여인가”를 분리해서 보기는 조금 어렵습니다. ablation이 어느 정도 도움을 주지만, 완전히 disentangle되지는 않습니다.

### 비판적 해석

제 해석으로 이 논문의 진짜 의의는 절대 수치보다도, **instance segmentation을 set prediction 문제로 다시 정식화했다**는 데 있습니다. detection에서는 이미 DETR가 보여준 바를, segmentation에서는 “mask 차원이 너무 높다”는 이유로 다들 주저하고 있었는데, ISTR는 이걸 **embedding regression**으로 우회했습니다. 이것은 굉장히 좋은 발상입니다.

동시에 오늘 관점에서 보면 ISTR는 완성형이라기보다 **중요한 전환점**에 더 가깝습니다. RoI 기반 전통과 Transformer 기반 set prediction 사이를 잇는 bridge 역할을 하며, 이후 mask query, mask decoder, set-based segmentation 계열 연구들이 왜 등장했는지를 이해하게 해주는 논문입니다. 즉, 이 논문은 “Transformer를 segmentation에 붙였다”보다, “mask를 직접 맞추지 말고 더 작은 공간에서 set prediction하자”는 관점 전환이 핵심입니다. 이 평가는 논문 본문과 실험 결과에 기반한 해석입니다.

## 6. Conclusion

ISTR는 instance segmentation을 위한 최초의 본격적인 **end-to-end Transformer framework**로 제시된 논문입니다. 핵심은 고차원 mask를 직접 예측하는 대신, PCA 등으로 얻은 **저차원 mask embedding**을 예측하고, 이를 ground truth embedding과 bipartite matching하여 set loss로 학습하는 것입니다. 여기에 recurrent refinement를 결합해 detection과 segmentation을 동시에 다듬고, 추론 시 NMS 없이 바로 결과를 사용합니다. COCO에서의 강한 성능은 이 접근이 단순한 개념적 가능성을 넘어, 실제 benchmark에서도 통한다는 점을 보여줍니다.

실무적으로는 “더 단순한 후처리-less instance segmentation”의 가능성을 열었고, 연구적으로는 instance segmentation을 **set prediction + low-dimensional representation** 문제로 재해석하는 기반을 마련했습니다. 후속 연구의 관점에서 보면 ISTR는 RoI 기반 시대와 query-based segmentation 시기를 잇는 중요한 이정표라고 볼 수 있습니다.
