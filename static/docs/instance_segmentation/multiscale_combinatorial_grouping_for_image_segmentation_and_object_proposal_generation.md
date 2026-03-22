# Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation

* **저자**: Jordi Pont-Tuset, Pablo Arbeláez, Jonathan T. Barron, Ferran Marques, Jitendra Malik
* **발표연도**: 2016
* **arXiv**: [https://arxiv.org/abs/1503.00848](https://arxiv.org/abs/1503.00848)

## 1. 논문 개요

이 논문은 image segmentation과 object proposal generation을 하나의 bottom-up 파이프라인으로 통합한 **Multiscale Combinatorial Grouping (MCG)** 를 제안한다. 저자들의 문제의식은 명확하다. 기존 object recognition 계열 방법들 중 하나는 sliding window 기반으로 모든 위치를 훑는 방식이고, 다른 하나는 category-independent object proposal을 먼저 생성한 뒤 이를 후단 recognition 모델에 넘기는 방식인데, 후자의 성능은 결국 얼마나 좋은 proposal을 적은 수로 생성하느냐에 크게 좌우된다.

기존 proposal 방법들 다수는 외부 contour detector나 고정된 superpixel/hierarchy를 입력으로 받아 그 위에서 candidate를 조합하는 구조였다. 즉, proposal 단계와 segmentation 단계가 분리되어 있고, proposal의 품질은 upstream segmentation의 한계에 묶여 있었다. 이 논문은 바로 그 지점을 겨냥한다. 저자들은 **좋은 contour**, **좋은 hierarchical region**, **좋은 object proposal** 이 사실상 강하게 연결된 문제이며, 이를 별개가 아니라 하나의 일관된 multiscale grouping 문제로 풀어야 한다고 본다.

논문의 목표는 세 가지로 요약할 수 있다. 첫째, normalized cuts 기반의 spectral globalization을 더 빠르게 계산하는 것이다. 둘째, 단일 해상도에 머무르지 않고 여러 scale의 segmentation hierarchy를 정렬하고 결합해 더 강한 multiscale hierarchy를 만드는 것이다. 셋째, 그렇게 얻은 region hierarchy들에서 조합적으로 region을 묶어 정확한 object proposal을 생성하는 것이다.

이 문제가 중요한 이유는 당시의 detection, segmentation, fine-grained categorization, large-scale recognition 등에서 object proposal의 역할이 매우 컸기 때문이다. 특히 proposal이 충분히 정확하면 downstream recognition 모델이 더 정교한 특징과 분류기를 사용할 수 있고, 전체 시스템의 계산량도 줄일 수 있다. 따라서 proposal 생성기의 품질과 효율성은 인식 시스템 전체의 품질을 좌우하는 핵심 전처리 문제라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **“좋은 object proposal은 multiscale hierarchical segmentation으로부터 region들을 조합적으로 묶어 얻을 수 있다”** 는 것이다. 여기서 중요한 점은 저자들이 단순히 여러 segmentation 결과를 나열하는 데 그치지 않고, 그것들을 **정렬(alignment)** 하고 **하나의 multiscale hierarchy로 결합** 한 뒤, 그 구조 안에서 region combination을 효율적으로 탐색한다는 점이다.

핵심 설계는 크게 세 층으로 이해할 수 있다.

첫 번째는 **fast normalized cuts** 이다. contour detection 성능을 높이기 위해 spectral globalization을 쓰는 것은 효과적이지만 계산량이 매우 크다. 저자들은 affinity matrix를 직접 다루는 대신, 이를 이미지의 multiscale 구조를 활용해 downsample하고, 축소된 공간에서 eigenvector를 계산한 뒤 다시 upsample하는 방식으로 근사한다. 이것이 contour globalization의 병목을 크게 줄인다.

두 번째는 **multiscale hierarchical segmentation** 이다. 서로 다른 해상도에서 독립적으로 segmentation hierarchy를 만든 뒤, coarse scale에서 생긴 hierarchy를 fine scale의 superpixel 경계에 맞춰 정렬한다. 이 과정을 통해 scale마다 조금씩 다른 구조를 버리지 않고, 경계 위치는 fine scale에 맞춘 채 여러 scale의 경계 강도를 통합할 수 있다.

세 번째는 **combinatorial grouping** 이다. 하나의 object가 hierarchy에서 항상 단일 region으로 나타나는 것은 아니다. 실제 객체는 색, 질감, 명암이 달라 내부가 여러 부분으로 분할되기 쉽다. 따라서 object를 찾으려면 single region만 보는 것이 아니라, 인접하거나 관계 있는 여러 region의 조합을 봐야 한다. 저자들은 hierarchy의 tree structure를 활용하여 이러한 region tuple들을 효율적으로 생성하고, Pareto front optimization으로 여러 hierarchy/tuple 크기에서 나온 proposal list를 적절히 결합한다.

기존 접근과의 차별점은 크게 두 가지다. 첫째, proposal을 만들기 전에 이미 계산된 외부 superpixel이나 hierarchy에 의존하지 않고, **segmentation과 proposal generation을 통합적으로 설계** 한다. 둘째, 단순한 pair/triplet 수준의 조합을 넘어서, **더 큰 combinatorial space를 실제로 탐색 가능하도록 효율화** 했다는 점이다. 이로써 proposal 수와 정확도 사이의 다양한 operating point에서 강한 성능을 낼 수 있다.

## 3. 상세 방법 설명

### 3.1 계층적 분할 표현: UCM과 region tree

논문은 segmentation hierarchy를 먼저 정식화한다. 이미지 영역을 partition하는 region 집합을 $S = {S_i}_i$ 라고 할 때, hierarchy는 가장 미세한 superpixel partition $S^*$ 에서 시작해 점차 병합되어 최종적으로 전체 이미지 하나가 되는 partition $S_L$ 까지의 계층 구조다.

이 hierarchy는 두 방식으로 표현된다. 하나는 **region tree (dendrogram)** 이고, 다른 하나는 **Ultrametric Contour Map (UCM)** 이다. UCM에서는 인접한 두 region의 경계에, 그들이 병합되는 수준의 strength를 부여한다. 따라서 UCM에서 threshold를 특정 값 $\lambda_i$ 로 두면 그 수준의 segmentation partition이 나온다. 이 표현의 장점은 contour detection과 hierarchical segmentation이 하나의 표현으로 묶인다는 것이다. 강한 contour는 늦게 병합되고, 약한 contour는 빨리 병합된다.

이 구조는 이후 두 가지 핵심 단계에 모두 쓰인다. 하나는 multiscale hierarchy alignment이고, 다른 하나는 object proposal을 만들기 위한 region combination이다.

### 3.2 Fast Downsampled Eigenvector Computation: DNCuts

논문의 첫 번째 기술적 공헌은 normalized cuts 계산의 가속화다. normalized cuts는 contour globalization에 강력하지만, affinity matrix $A$ 가 매우 커서 Laplacian의 고유벡터를 직접 구하는 비용이 높다. 저자들은 이미지가 원래 multiscale 구조를 가진다는 점을 이용해 이를 근사한다.

핵심 직관은 두 가지다.

첫째, 만약 $A$ 가 bistochastic이면 $A$ 의 Laplacian eigenvector와 $A^2$ 의 Laplacian eigenvector가 같다는 성질을 활용할 수 있다. 둘째, 이미지에서 한 칸 건너 픽셀만 남기는 decimation을 해도, 적절히 정보를 보존하면 원래 구조와 유사한 spectral structure를 얻을 수 있다.

하지만 단순히 $A[i,i]$ 처럼 decimated matrix를 만들면 픽셀 연결이 끊어져 성능이 나빠진다. 그래서 저자들은 먼저 행렬을 제곱해 더 넓은 이웃으로 정보를 퍼뜨린 뒤 decimation을 한다. 실제 계산은 다음과 같은 형태로 수행된다.

주어진 decimated index 집합을 $i$ 라고 하면, 직접 $A^2[i,i]$ 를 계산하지 않고 다음과 같이 효율적으로 계산한다.

$$
A^2[i,i] \approx A[:,i]^T A[:,i]
$$

논문 의사코드에서는 이를 조금 더 정규화한 형태로 쓴다. 구체적으로 각 단계 $s$ 에서

$$
B_s = A_{s-1}[:, i_s]
$$

를 만든 뒤, 열 방향 합에 기반한 대각 정규화 행렬을 사용하여

$$
C_s = \operatorname{diag}(B_s \mathbf{1})^{-1} B_s
$$

를 만들고, 다음 축소 affinity를

$$
A_s = C_s^T B_s
$$

로 구성한다. 이렇게 $d$ 번 downsample한 뒤 가장 작은 $k$ 개 eigenvector를 축소 공간에서 구하고, 이를 다시

$$
X_{s-1} = C_s X_s
$$

로 순차적으로 upsampling한다. 마지막에는 whitening을 적용한다.

이 방법의 의미는, 원래 큰 graph에서 spectral computation을 하지 않고 더 작은 graph에서 근사 계산을 수행하되, 이미지의 구조를 유지하는 방식으로 다시 확장한다는 것이다. 논문에 따르면 이 과정은 contour detection 성능을 거의 잃지 않으면서 약 **20배 속도 향상** 을 제공한다.

### 3.3 Segmentation hierarchy 정렬: projection 기반 alignment

여러 해상도에서 segmentation hierarchy를 독립적으로 만들면, subsampling 때문에 경계가 서로 정확히 맞지 않는다. coarse scale에서는 세부 구조가 사라지고, 경계 위치도 fine scale과 어긋난다. 따라서 multiscale 정보를 결합하려면 먼저 서로 다른 hierarchy를 **같은 boundary support 위에 정렬** 해야 한다.

이를 위해 저자들은 segmentation $R$ 을 target segmentation $S$ 에 투영하는 projection $\pi(R,S)$ 를 정의한다. 각 target region $S_j$ 에 대해, $R$ 에서 가장 많이 겹치는 region label을 부여한다. 즉,

$$
L(S_j) = \arg\max_i \frac{|S_j \cap R_i|}{|S_j|}
$$

이다. 쉽게 말해, $S_j$ 내부 픽셀들이 주로 어느 $R_i$ 에 속하는지를 보고 그 label을 가져온다. 그러면 $R$ 의 경계를 $S$ 의 경계에 “snap” 시킨 segmentation이 만들어진다.

이를 UCM 전체에 적용할 때는, hierarchy의 여러 threshold level마다 segmentation을 꺼내서 각각 target segmentation에 투영하고, 그 결과 contour를 다시 모아 projection된 UCM을 만든다. 논문 의사코드에서 각 level $t$ 에 대해 segmentation을 샘플링하고, rescale한 뒤, projection을 하고, boundary를 추출하여 UCM에 누적하는 과정이 제시된다.

이 설계의 중요한 장점은, 모든 scale의 hierarchy를 결국 가장 세밀한 superpixel 경계에 맞출 수 있다는 점이다. 즉, coarse scale의 semantic robustness와 fine scale의 precise boundary localization을 동시에 활용하려는 의도다.

### 3.4 Single-scale segmentation

단일 scale hierarchy는 여러 contour cue를 결합해 만든다. 논문에서 고려한 입력 contour는 다음 세 종류다.

brightness, color, texture difference를 half-disk에서 계산한 고전적 local cue, sparse coding 기반 contour cue, structured forests contour cue가 그것이다. 각 cue에 대해 Sect. 3.1의 fast eigenvector globalization을 독립적으로 수행한 뒤, local cue와 global cue를 선형 결합하고, 평균 contour strength를 사용해 UCM을 구성한다.

가중치 학습은 contour F-measure를 직접 최적화하기보다, 최종 hierarchy 품질을 더 잘 반영하는 **Segmentation Covering** metric을 기준으로 했다. 이는 저자들이 contour 자체보다 region hierarchy 품질을 더 중시했음을 보여준다.

### 3.5 Multiscale hierarchy 구성

전체 multiscale pipeline은 다음과 같다.

먼저 원본 이미지를 여러 해상도로 만든 image pyramid를 구성한다. 논문에서는 $N = {0.5, 1, 2}$ 스케일이 적절하다고 보고한다. 각 scale에서 single-scale segmenter를 실행하여 독립적인 hierarchy를 만든다. 이후 가장 고해상도 scale의 finest superpixel을 가능한 boundary location 집합으로 정하고, 각 coarser UCM을 점차 finer scale에 projection해 정렬한다.

정렬이 끝나면 모든 scale에서 동일한 boundary support 위에 경계 강도만 여러 개 존재하는 상태가 된다. 즉, 각 경계 위치마다 $N$ 개의 strength feature가 생긴다. 저자들은 이를 **binary boundary classification** 문제로 보고, 여러 scale의 strength를 하나의 boundary probability로 결합한다. 여러 결합 전략을 시험했지만 결과 차이가 크지 않았고, 결국 가장 단순한 uniform weighting 후 Platt scaling을 사용했다. 이는 이 문제가 feature dimension이 작고 training sample이 많아서 상대적으로 쉬운 분류 문제였기 때문이라고 해석한다.

### 3.6 Object proposal generation: region tuple 조합

segmentation hierarchy만으로는 complete object를 항상 단일 region으로 표현할 수 없다. 객체 내부의 바퀴, 창문, 몸체처럼 appearance가 다르면 서로 다른 region으로 쪼개질 수 있다. 따라서 object proposal은 **hierarchy 상의 region set selection 문제** 로 본다.

예를 들어 어떤 객체가 hierarchy에서 하나의 region이 아니라 세 개 region의 합집합으로 가장 잘 표현될 수 있다. 그러므로 proposal generator는 singleton만이 아니라 pair, triplet, 4-tuple 등의 조합을 탐색해야 한다.

문제는 combinatorial explosion이다. 가능한 region tuple 수가 매우 크므로 전부 다 볼 수 없다. 이를 위해 저자들은 region tree 위에서 area, bounding box, perimeter, neighbor 등의 descriptor를 효율적으로 계산하는 알고리즘을 설계한다. 예를 들어 모든 region의 area를 계산할 때, merging-sequence partition을 모두 픽셀 단위로 훑으면 비용이 $p \cdot (m+1)$ 이지만, region tree에서 leaf area를 한 번 계산하고 부모로 누적하면 $p+m$ 이 된다. 여기서 $p$ 는 픽셀 수, $m$ 은 merging 수다. 이 아이디어를 bounding box, perimeter, neighbor 계산에도 확장한다.

proposal 생성 자체는 hierarchy를 위에서부터 내려가며 region neighbor를 계산하고, 특정 depth 또는 UCM threshold까지만 탐색하는 방식이다. proposal의 초기 랭킹은 proposal을 이루는 region들의 **최소 UCM strength** 등을 기반으로 형성된다. 그리고 단일 hierarchy뿐 아니라 각 scale의 hierarchy 및 multiscale hierarchy에서 생성된 proposal list들을 함께 사용해 diversity를 확보한다.

### 3.7 Pareto front 기반 파라미터 학습

MCG는 여러 hierarchy와 여러 tuple 크기에서 나온 ranked proposal list들을 결합한다. 예를 들어 singleton, pair, triplet, 4-tuple을 4개 hierarchy에서 뽑으면 총 16개 ranked list가 된다. 학습 시에는 각 리스트에서 몇 개 proposal을 가져올지 ${N_1, \dots, N_R}$ 를 정해야 한다.

이 문제는 proposal 수와 achievable quality 사이의 trade-off 최적화다. proposal을 많이 쓰면 quality는 좋아지지만 계산량이 늘어난다. 저자들은 이를 **Pareto front optimization** 으로 푼다. 모든 조합을 전수 탐색하면 경우의 수가 $\prod_i |L_i|$ 수준으로 커서 불가능하므로, 두 리스트씩 점진적으로 결합하면서 Pareto front 위에 있는 해만 유지한다. 샘플 수를 $S$ 라 하면 전체 탐색 복잡도를 지수적 $S^R$ 대신 대략 $(R-1)S^2$ 수준으로 줄인다.

이 단계는 매우 실용적이다. 최종적으로 사용자는 proposal 수 제한 $N_p$ 또는 필요한 최소 quality에 맞는 operating point를 Pareto front에서 선택하면 된다.

### 3.8 Proposal reranking

생성된 proposal 수를 더 줄이기 위해, 저자들은 low-level feature 기반 회귀기(regressor)를 학습한다. 사용 feature는 크게 세 부류다.

첫째, **size and location** 으로 area, perimeter, bounding box area/position/aspect ratio, region 간 area balance 등을 쓴다. 둘째, **shape** 로 perimeter를 $\sqrt{\text{area}}$ 로 정규화한 값, region area 대비 bounding box area 비율 등을 쓴다. 셋째, **contours** 로 boundary contour strength의 합과 평균, proposal을 구성하는 region들의 appearance/disappearance UCM threshold 최소/최대값 등을 쓴다.

이 feature들로 Random Forest를 학습하여 proposal과 ground-truth object의 overlap을 회귀하고, Maximum Marginal Relevance를 이용해 다양성을 확보한 ranking을 만든다. 즉, proposal 생성은 hierarchy 기반 조합으로 하고, 최종 ranking은 학습된 regressor가 담당한다.

## 4. 실험 및 결과

## 4.1 BSDS500에서의 hierarchical segmentation 평가

논문은 먼저 BSDS500에서 contour와 hierarchy 품질을 평가한다. 사용한 주요 지표는 boundary F-measure의 ODS/OIS, objects and parts 평가용 $F_{op}$, region 품질을 보는 Segmentation Covering (SC), Probabilistic Rand Index (PRI), Variation of Information (VI) 등이다.

단일 scale 결과를 보면, 서로 다른 입력 contour cue 각각에 대해 고품질 hierarchy를 얻었고, 이들을 조합했을 때 더 좋은 결과가 나왔다. 예를 들어 single-scale의 combined 입력은 boundary ODS 0.719, OIS 0.750, region covering ODS 0.602, OIS 0.655를 기록했다. 이는 서로 다른 contour cue가 상호 보완적이라는 점을 보여준다.

multiscale 결과는 모든 입력 cue에서 일관된 개선을 보인다. combined multiscale의 경우 boundary ODS/OIS가 각각 0.725/0.757, region covering ODS/OIS가 0.611/0.670으로 향상된다. structured forest 입력만 보더라도 single-scale 대비 multiscale이 더 나은 boundary 및 region 성능을 보인다.

저자들은 단순히 contour를 resize해서 합치는 방식이나, alignment 없이 UCM을 옮겨 결합하는 ablation도 실험했는데, 이런 단순한 대안은 interpolation artifact나 misalignment 때문에 single-scale을 넘지 못했다. 이 결과는 제안한 hierarchy alignment가 단순한 보조 단계가 아니라 multiscale 성능 향상에 필수적이라는 점을 뒷받침한다.

또한 공개 코드가 있는 기존 방법들과의 비교에서 MCG와 SCG는 BSDS500 test set에서 boundary와 objects-and-parts 모두에서 당시 최고 수준의 성능을 보였다고 주장한다. 특히 boundary 자체의 개선 폭은 BSDS500의 object scale 다양성이 제한적이어서 아주 크지 않을 수 있지만, object/part 기준에서는 multiscale의 이점이 더 두드러졌다고 해석한다.

## 4.2 SegVOC12, SBD, COCO에서의 object proposal 평가

proposal 평가는 SegVOC12, SBD, COCO 세 데이터셋에서 수행되었다. 데이터 규모는 각각 2,913장/9,847 objects, 12,031장/32,172 objects, 123,287장/910,983 objects다. 특히 COCO는 객체 수와 다양성이 훨씬 크기 때문에 일반화 성능을 보기 좋다.

핵심 품질 지표는 instance-level Jaccard index $J_i$ 이다. 각 ground-truth object에 대해 proposal pool 중 가장 overlap이 큰 proposal의 IoU를 찾고, 이를 모든 instance에 대해 평균낸 값이다. 흔히 Average Best Overlap와 유사한 개념이다. 또한 단순 평균만으로는 분포를 알기 어렵기 때문에, 저자들은 특정 Jaccard threshold $J=0.5$, $0.7$, $0.85$ 에서의 recall도 함께 본다.

### 4.2.1 Pareto front와 조합 효과

SegVOC12 training/validation에서 1-region, 2-region, 3-region, 4-region proposal을 순차적으로 포함했을 때 achievable quality가 어떻게 올라가는지 평가했다. 결과적으로 singletons만 쓰는 것보다 pairs와 triplets를 포함하는 것이 눈에 띄는 향상을 만든다. 이는 객체가 hierarchy에서 한 개 region으로 잘 떨어지지 않는 경우가 많다는 논문의 핵심 주장과 맞아떨어진다. 반면 4-tuple 추가의 이득은 상대적으로 작았다.

또한 여러 scale proposal을 동일 비율로 단순 결합하는 것보다 Pareto front 기반으로 scale별/tuple별 proposal 수를 학습하는 것이 더 낫다. 특히 proposal 수가 적은 practical regime에서 약 2포인트 정도의 성능 이득이 보고된다. 이는 “어떤 scale, 어떤 tuple 크기에서 나온 proposal을 몇 개 쓸지”가 중요한 하이퍼파라미터이며, 이를 데이터 기반으로 조정해야 함을 보여준다.

저자들은 validation set에서 full 16 lists를 조합하면 이미지당 수백만 proposal이 될 수 있지만, MCG의 combinatorial grouping과 selection으로 이를 약 **5,086개 proposal** 수준까지 줄이면서도 매우 높은 achievable $J_i = 0.81$ 을 얻었다고 보고한다. 이후 regressed ranking을 통해 proposal 수를 더 줄일 수 있다.

### 4.2.2 State of the art 비교: segmented proposals

SegVOC12, SBD, COCO validation set에서 CPMC, CI, GOP, GLS, RIGOR, Shape Sharing, Selective Search 등 당시 대표 proposal 방법들과 비교했다. 그래프에 따르면 MCG는 거의 모든 proposal 수 구간에서 가장 높은 $J_i$ 를 보이며, 데이터셋이 커질수록 다른 방법 대비 우위가 더 커진다. 저자들은 이것을 **SegVOC12에서만 학습했는데도 SBD와 COCO로 일반화가 잘 된다** 는 증거로 해석한다.

특히 recall-at-IoU 결과가 흥미롭다. 예를 들어 낮은 threshold인 $J=0.5$ 에서는 어떤 경쟁 방법이 비슷하거나 일부 구간에서 더 나아 보일 수 있지만, threshold를 $0.7$, $0.85$ 로 높일수록 MCG의 우위가 더 뚜렷해진다. 이는 MCG가 단지 대충 object 근처를 덮는 proposal을 많이 내는 것이 아니라, **정확한 shape localization** 에 강하다는 뜻이다.

SCG는 MCG보다 단순하고 훨씬 빠른 버전인데도 상당히 경쟁력 있는 성능을 보인다. 즉, 최고 성능을 원하는 경우 MCG, 더 빠른 운영이 필요한 경우 SCG라는 현실적 선택지가 생긴다.

### 4.2.3 Bounding box proposal로서의 성능

MCG는 본래 segmented proposal을 목표로 하지만, 각 segmented proposal의 bounding box를 취하면 box proposal로도 활용할 수 있다. Edge Boxes, BING, Randomized Prim, objectness measure 같은 bounding-box 특화 방법들과 비교했을 때, 낮은 IoU threshold에서는 box 특화 방법들이 경쟁력이 있지만, 높은 IoU threshold로 갈수록 MCG가 더 강한 결과를 보인다.

이는 의미가 크다. MCG는 애초에 pixel-accurate segmentation을 목적으로 설계되었음에도, 그 결과로 얻은 bounding box가 정밀한 localization에서도 강하다는 뜻이다. 다시 말해 segmentation proposal 품질이 좋으면 box proposal 품질도 자연스럽게 따라올 수 있음을 보여준다.

### 4.2.4 계산 시간

시간 측면에서 full MCG는 contour detection에 평균 4.6초, hierarchical segmentation에 20.5초, candidate generation에 17.0초가 걸려 총 약 **42.2초/image** 가 든다. 반면 SCG는 contour 1.0초, hierarchy 2.7초, proposal generation 2.6초로 총 약 **6.2초/image** 다.

경쟁 방법과 비교하면 GOP는 더 빠르지만 proposal 품질은 낮고, GLS나 Selective Search는 속도 또는 품질에서 타협이 있다. CPMC, CI, Shape Sharing은 훨씬 느리다. 따라서 MCG는 최상급 품질을 제공하는 대신 계산량이 꽤 크고, SCG는 그보다 훨씬 빠른 practical compromise라고 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 segmentation과 object proposal을 별개 단계로 보지 않고 하나의 구조 안에서 통합했다는 점이다. 많은 기존 방법이 외부 segmentation 결과를 입력으로 사용했는데, 이 논문은 contour globalization, hierarchy 생성, multiscale 결합, proposal 조합을 연속된 설계로 묶었다. 그 결과 contour, hierarchy, proposal 모두에서 강한 성능을 얻는다.

둘째 강점은 **multiscale 정렬과 결합** 이 매우 설득력 있게 설계되었다는 점이다. 단순히 여러 해상도 결과를 평균내는 것이 아니라, fine-scale boundary support에 맞춰 coarse hierarchy를 투영하는 방식은 이론적으로도 직관적이고, ablation 결과로도 효과가 입증된다.

셋째는 **효율화의 수준** 이다. DNCuts는 normalized cuts의 병목을 완화하여 약 20배의 속도 향상을 제공하고, region tree 기반 descriptor 계산과 Pareto front optimization은 combinatorial grouping을 실제 가능한 수준으로 만든다. 즉, 논문은 아이디어만 새롭고 계산은 불가능한 방식이 아니라, 실제 system engineering까지 포함한 완성도 높은 설계를 보여준다.

넷째는 **정확한 proposal에 강하다** 는 점이다. 높은 Jaccard threshold에서 recall이 높다는 것은 실제 instance segmentation이나 정밀 localization이 필요한 작업에서 가치가 크다. 단순히 “근처에 박스 하나”를 잘 내는 것이 아니라, shape 자체를 꽤 잘 맞춘다.

다만 한계도 분명하다.

첫째, full MCG의 계산량은 여전히 크다. 당시 기준으로도 42초/image는 실시간이나 대규모 서비스에는 부담스럽다. SCG가 이를 완화하지만, 최고 품질 모델과 빠른 모델 사이에 뚜렷한 trade-off가 존재한다.

둘째, 방법 전체가 여전히 **low-level cue 중심** 이다. 저자들도 인정하듯 hierarchy는 brightness, color, texture, contour strength 등에 기반해 만들어지므로, semantic information이 직접 들어가 있지 않다. 그래서 complete object가 단일 region으로 잘 안 떨어지고, 이를 보완하기 위해 combinatorial grouping이 필요하다. 즉, proposal 품질이 높아졌다고 해도 semantic understanding 자체를 내장한 것은 아니다.

셋째, proposal ranking에 사용되는 feature와 regressor는 비교적 hand-crafted 성격이 강하다. size, shape, contour 통계는 효율적이지만, objectness를 더 풍부하게 표현하는 high-level feature는 포함되지 않는다. 이는 이후 deep learning 기반 proposal/ranking 방식이 발전하면서 상대적으로 약점이 되었을 가능성이 크다.

넷째, 논문은 Pareto front 전략이 overfit되지 않는다고 보고하지만, regressed ranking 쪽은 training과 validation 사이 차이가 있어 overfitting이 일부 존재한다고 명시한다. 저자들은 그래도 non-regressed보다 유리하다고 판단했지만, 이 단계의 일반화는 segmentation hierarchy 자체만큼 강하지 않을 수 있다.

다섯째, 본문에 loss function이나 end-to-end learning objective 같은 현대적 의미의 학습 구조는 없다. boundary combination, ranking 등 일부 단계는 학습되지만, 전체 파이프라인은 모듈식이다. 이는 해석 가능성과 제어 가능성 측면에서는 장점이지만, 전체 목적에 대해 joint optimization이 되지 않는다는 한계도 있다.

비판적으로 보면, 이 논문은 **pre-deep-learning proposal pipeline의 정점에 가까운 매우 강한 구조적 방법** 이다. 하지만 의미론적 단서를 직접 학습하는 대신, low-level grouping과 조합 탐색을 극한까지 밀어붙인 방식이기 때문에, 이후 등장한 learned region proposal 또는 mask proposal 방법들에 비해 표현력 측면에서 구조적 한계를 가질 수 있다. 그럼에도 당시 문제 설정에서는 매우 강력하고, 특히 proposal 품질과 hierarchy 품질의 연결을 설득력 있게 보여준다는 점에서 학술적 가치가 크다.

## 6. 결론

이 논문은 **Multiscale Combinatorial Grouping (MCG)** 을 통해 bottom-up segmentation과 object proposal generation을 하나의 통합 프레임워크로 제시했다. 핵심 기여는 세 가지로 정리된다. 첫째, normalized cuts의 eigenvector 계산을 크게 가속하는 DNCuts를 제안했다. 둘째, 여러 해상도의 segmentation hierarchy를 정렬하고 결합하는 multiscale hierarchical segmentation을 설계했다. 셋째, hierarchy에서 region을 조합적으로 묶는 efficient combinatorial grouping과 Pareto front 기반 selection을 통해 높은 품질의 object proposal을 생성했다.

실험적으로도 BSDS500에서는 contour와 hierarchy 품질에서, SegVOC12/SBD/COCO에서는 segmented proposal과 bounding box proposal 모두에서 당시 최고 수준의 성능을 보였다. 특히 높은 IoU threshold에서 recall이 높다는 점은 이 방법이 **정밀한 object support** 를 잘 생성한다는 강한 증거다.

이 연구의 실제적 의미는 크다. detection이나 segmentation의 전처리 단계에서 category-independent proposal이 중요한 시대에, MCG는 매우 높은 품질의 후보를 안정적으로 제공하는 강력한 기반 기술이었다. 또한 단순 성능 보고를 넘어, multiscale alignment, hierarchy representation, efficient region-tree computation, Pareto selection 등 이후에도 재사용 가능한 여러 구조적 아이디어를 제시했다.

향후 연구 관점에서 보면, 이 논문은 두 방향으로 중요하다. 하나는 전통적 grouping 기반 vision이 어디까지 갈 수 있는지 보여주는 기준점이라는 점이다. 다른 하나는, semantic learning이 강해진 이후에도 multiscale structure와 hierarchical grouping이 여전히 중요한 설계 요소임을 상기시킨다는 점이다. 즉, 이 논문은 딥러닝 이전의 고전적 방법론에 머물지 않고, 오늘날의 learned segmentation/proposal 시스템을 해석할 때도 참고할 수 있는 구조적 통찰을 제공한다.
