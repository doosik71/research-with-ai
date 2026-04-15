# Pixelwise Instance Segmentation with a Dynamically Instantiated Network

- **저자**: Anurag Arnab, Philip H.S. Torr
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1704.02386

## 1. 논문 개요

이 논문은 instance segmentation을 “각 픽셀에 semantic class와 instance identity를 동시에 부여하는 문제”로 정의하고, 이를 이미지 전체를 한 번에 보는 end-to-end 네트워크로 해결하려는 연구이다. 기존의 많은 방법은 먼저 object detector로 bounding box를 찾고, 그 박스 내부를 다시 segmentation하는 방식이었다. 그러나 이런 접근은 초기 detection 품질에 크게 의존하고, proposal들을 독립적으로 처리하기 때문에 서로 가리는 물체들 사이의 관계를 전체적으로 다루기 어렵다.

저자들은 이 문제를 semantic segmentation의 확장으로 본다. 즉, 픽셀마다 클래스만 맞히는 것이 아니라, 같은 클래스 안에서 “어느 개체에 속하는지”까지 구분해야 한다는 것이다. 이를 위해 먼저 semantic segmentation subnetwork가 클래스 수준의 픽셀 분포를 만들고, 그 결과와 object detector의 출력을 함께 사용하여 instance-level CRF를 구성한다. 이 CRF는 이미지마다 detection 개수가 다르므로 동적으로 label 수가 달라지는 구조를 가진다.

이 문제가 중요한 이유는 분명하다. semantic segmentation은 같은 클래스의 여러 객체를 구분하지 못하고, object detection은 위치를 박스 단위로만 제공한다. 반면 instance segmentation은 자율주행, 로보틱스, 이미지 편집처럼 정확한 객체 단위 이해가 필요한 응용에서 핵심 역할을 한다. 이 논문은 특히 “proposal을 따로 후처리하지 않고”, “한 픽셀이 둘 이상의 인스턴스에 속하지 않게” 하면서, 더 정밀한 segmentation을 얻는 것을 목표로 한다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 instance segmentation을 detection refinement 문제로 보지 않고, semantic segmentation 결과를 출발점으로 하는 전역적 픽셀 labeling 문제로 재구성한 데 있다. 먼저 semantic segmentation으로 각 픽셀의 클래스 확률을 얻고, 그 다음 단계에서 detector의 bounding box와 score를 보조 신호로 사용해 각 픽셀이 어떤 instance에 속하는지를 정한다.

중심 직관은 다음과 같다. semantic segmentation은 이미 “이 픽셀이 사람인지, 자동차인지”에 대한 강한 정보를 제공한다. 그렇다면 detector는 반드시 segmentation의 출발점이 아니라, 여러 instance를 분리하기 위한 cue로 쓰일 수 있다. 이런 관점 덕분에 detector가 false positive를 내더라도 semantic segmentation이 이를 억제할 수 있고, bounding box가 객체 전체를 덮지 못해도 global term이 box 바깥 픽셀까지 같은 instance에 연결할 수 있다.

기존 접근과의 차별점은 세 가지가 크다. 첫째, proposal을 독립적으로 처리하지 않고 이미지 전체를 함께 추론한다. 둘째, detection box를 segmentation으로 바꾸는 대신 semantic segmentation과 CRF를 중심으로 instance를 분해한다. 셋째, 이미지마다 detection 수가 다르므로 출력 instance 수가 가변적인 dynamic network를 만든다. 저자들은 이로 인해 occlusion 처리, false detection에 대한 강건성, 높은 IoU 기준에서의 정밀한 mask 품질이 좋아졌다고 주장한다.

## 3. 상세 방법 설명

전체 파이프라인은 두 부분으로 이루어진다. 첫 번째는 semantic segmentation subnetwork이고, 두 번째는 instance segmentation subnetwork이다. 첫 단계에서는 FCN8s 기반 semantic segmentation 네트워크에 CRF inference를 RNN처럼 unroll한 구조를 붙여, 각 픽셀의 클래스 확률 $Q_i(l)$를 계산한다. 여기서 $Q$는 크기 $W \times H \times (K+1)$의 텐서이며, $K$는 background를 제외한 클래스 수이다. 이 semantic module은 dense pairwise CRF와 detection-aware higher-order potential까지 포함한다.

두 번째 단계에서는 object detector의 출력과 semantic segmentation 결과를 함께 사용한다. 한 이미지에서 detector가 $D$개의 객체를 찾았다고 하자. 각 detection은 $(l_i, s_i, B_i)$ 형태이며, $l_i$는 class label, $s_i$는 confidence, $B_i$는 bounding box 내부 픽셀 집합이다. 이제 각 픽셀 $i$에 대해 확률변수 $V_i$를 두고, 이 픽셀이 background 또는 $1$부터 $D$까지의 어떤 detection instance에 속하는지 예측한다. 따라서 최종 출력은 $W \times H \times (D+1)$ 크기를 갖고, 이미지마다 $D$가 달라진다.

이 instance CRF의 energy는 다음과 같다.

$$
E(V=v) = \sum_i U(v_i) + \sum_{i<j} P(v_i, v_j)
$$

여기서 unary term은 세 개의 항을 결합한다.

$$
U(v_i) = - \ln \left[w_1 \psi_{\text{Box}}(v_i) + w_2 \psi_{\text{Global}}(v_i) + w_3 \psi_{\text{Shape}}(v_i)\right]
$$

$w_1, w_2, w_3$는 backpropagation으로 학습되는 가중치이다.

첫 번째인 Box term은 픽셀이 어떤 detection의 bounding box 안에 있을 때, 그 픽셀이 해당 detection class일 semantic probability와 detection score를 이용한다.

$$
\psi_{\text{Box}}(V_i = k) =
\begin{cases}
Q_i(l_k)s_k & \text{if } i \in B_k \\
0 & \text{otherwise}
\end{cases}
$$

이 항은 detector가 가리키는 box 내부에서는 강한 instance 단서를 주지만, box 바깥은 설명하지 못한다.

두 번째인 Global term은 bounding box를 쓰지 않고 semantic segmentation만 사용한다.

$$
\psi_{\text{Global}}(V_i = k) = Q_i(l_k)
$$

이 항의 의미는 “해당 클래스의 instance 후보가 몇 개 있다는 사실만 알면, 위치 정보가 없을 때 각 instance는 일단 동일하게 가능하다”는 것이다. 실제로는 semantic segmentation 확률이 높은 모든 위치에 instance 가능성을 열어 두므로, bounding box가 객체 전체를 덮지 못할 때도 box 바깥까지 연결될 수 있다. 저자들은 이 항이 특히 높은 IoU threshold에서 큰 효과를 낸다고 설명한다.

세 번째인 Shape term은 shape prior를 도입한다. 같은 클래스의 객체들이 서로 가리고 있고 appearance 차이도 작을 때, 단순한 box와 global 정보만으로는 구분이 어렵다. 이를 위해 각 클래스에 대해 준비된 shape template들을 detection box 크기에 맞게 bilinear interpolation으로 warp한 뒤, box 내부의 segmentation prediction과 normalized cross correlation이 가장 높은 template를 고른다.

$$
t^* = \arg\max_{t \in \tilde{T}}
\frac{\sum Q_{B_k}(l_k) \odot t}{\|Q_{B_k}(l_k)\| \, \|t\|}
$$

그 다음 선택된 template와 segmentation unaries의 elementwise product를 사용한다.

$$
\psi_{\text{Shape}}(V_{B_k} = k) = Q_{B_k}(l_k) \odot t^*
$$

저자들은 이 연산이 사실상 max-pooling의 특수한 형태로 볼 수 있고, matched exemplar $t^*$에 대해 gradient를 흘릴 수 있으므로 differentiable하다고 설명한다. shape prior는 Scalpel에서 사용된 템플릿들을 초기값으로 사용했으며, 약 250개의 shape template를 aspect ratio별로 준비했다.

Pairwise term은 fully connected Gaussian CRF를 사용한다. 이는 공간적으로 가깝고 외형이 비슷한 픽셀들이 같은 instance에 속하도록 유도한다. 저자들은 이 항이 특히 같은 클래스 내 occlusion에서 appearance 차이가 있을 때 instance 분리에 도움을 준다고 말한다. 구체적인 pairwise 식은 본문에서 길게 재정의하지 않고, Dense CRF 계열의 표준 형식을 따른다고 설명한다.

추론은 mean field inference로 수행한다. 이 과정을 RNN처럼 unroll하여 network layer로 넣기 때문에 전체 모델이 end-to-end 학습 가능하다. 중요한 점은 label 수가 이미지마다 다르므로 CRF가 동적으로 instantiate된다는 것이다. 이 때문에 클래스별로 별도 파라미터를 둘 수 없고, weight sharing을 사용한다. 저자들은 instance id는 순열에 대해 의미가 없으므로 class-specific weight가 본질적으로도 맞지 않는다고 본다.

학습에서 가장 중요한 문제는 instance label permutation이다. 예를 들어 두 사람 instance에 대해 label 1과 2를 서로 바꿔도 결과는 동일하다. 이를 해결하기 위해 저자들은 prediction $P$와 ground truth $G$ 사이에서 IoU가 최대가 되도록 ground truth label permutation을 먼저 찾는다.

$$
G^* = \arg\max_{m \in M} \text{IoU}(m, P)
$$

여기서 $M$은 ground truth의 모든 permutation 집합이다. 실제 계산은 전 permutation을 다 보지 않고, predicted segment와 ground-truth segment 사이의 IoU를 edge weight로 하는 maximum-weight bipartite matching으로 푼다. semantic class가 다르면 edge weight는 0이다. 이렇게 얻은 matched ground truth $G^*$에 대해 최종적으로 cross-entropy loss를 적용한다. 저자들은 approximate IoU loss보다 cross-entropy가 더 잘 동작했다고 보고한다.

학습 절차는 2단계이다. 먼저 semantic segmentation network를 standard cross-entropy로 pretrain한다. 그 다음 instance subnetwork를 붙이고, instance segmentation annotation만 사용하여 전체 네트워크를 finetune한다. semantic pretraining은 learning rate $10^{-8}$, momentum 0.9, batch size 20으로 진행했고, instance까지 포함한 학습에서는 learning rate를 $10^{-12}$로 낮추고 batch size 1을 사용했다. 또한 두 개의 CRF-RNN이 들어가므로 gradient clipping을 적용했다. 논문은 clipping threshold를 $\ell_2$ norm 기준 $10^9$로 두었다고 명시한다.

## 4. 실험 및 결과

실험은 Pascal VOC 2012 validation set, Semantic Boundaries Dataset(SBD), Cityscapes에서 수행되었다. semantic segmentation pretraining에는 VOC, SBD, MS COCO를 사용했고, instance segmentation finetuning은 VOC 또는 SBD의 training split만 사용했다. object detector로는 공개된 R-FCN을 사용했으며, detector training 이미지가 test set과 겹치지 않도록 조정했다고 한다.

주요 평가지표는 $AP^r$이다. 이는 object detection의 AP와 유사하지만 IoU를 bounding box가 아니라 region mask 기준으로 계산한다. 또한 단일 threshold만 보지 않고, 0.1부터 0.9까지 0.1 간격의 9개 threshold 평균인 $AP^r_{\text{vol}}$도 사용한다. 저자들은 여기에 더해 Matching IoU라는 지표를 추가로 제안한다. 이는 예측 segmentation map과 ground truth map을 instance matching한 뒤 전체 map 수준에서 IoU를 계산하는 방식이다. proposal별 랭킹만 보는 $AP^r$와 달리, 전역적으로 일관된 segmentation map을 얼마나 잘 만드는지 평가하려는 목적이다.

VOC 2012 validation set에서의 ablation은 이 논문의 설계 의도를 잘 보여준다. Box term만 쓴 경우 end-to-end 학습으로 $AP^r_{0.5}=60.7$, $AP^r_{0.7}=47.4$, $AP^r_{0.9}=24.6$, $AP^r_{\text{vol}}=56.2$, Matching IoU 46.9를 얻었다. 여기에 Global term을 추가하면 $AP^r_{0.9}$가 25.5로 올라가고, $AP^r_{\text{vol}}$은 56.7이 된다. Shape term까지 추가한 최종 모델은 $AP^r_{0.5}=61.7$, $AP^r_{0.7}=48.6$, $AP^r_{0.9}=25.1$, $AP^r_{\text{vol}}=57.5$, Matching IoU 48.3을 기록했다. 저자들의 해석대로, Global term은 높은 IoU threshold에서 box localization error를 보완하고, Shape term은 occlusion 상황에서 instance recovery에 도움을 주며 낮은 threshold까지 포함한 전체 평균을 올린다.

또한 piecewise training보다 end-to-end training이 일관되게 더 좋았다. 예를 들어 Box+Global+Shape 조합은 piecewise에서 $AP^r_{\text{vol}}=55.2$, Matching IoU 44.8이었는데, end-to-end에서는 각각 57.5와 48.3으로 상승했다. 이는 instance loss가 semantic subnetwork까지 역전파되면서 더 적합한 feature와 segmentation probability를 만들었기 때문으로 해석된다.

VOC validation에서 기존 방법들과 비교하면, 최종 모델은 $AP^r_{0.5}=61.7$, $AP^r_{0.6}=55.5$, $AP^r_{0.7}=48.6$, $AP^r_{0.8}=39.5$, $AP^r_{0.9}=25.1$, $AP^r_{\text{vol}}=57.5$를 달성했다. 비교 대상 중 MPA 3-scale은 $AP^r_{0.5}=62.1$로 약간 높지만, $0.7$ 이상에서는 이 논문 방법이 더 좋고 특히 $0.9$에서는 18.5 대비 25.1로 크게 앞선다. 저자들은 이를 “더 정교하고 자세한 segmentation”의 증거로 해석한다. 특히 $0.9$ threshold에서 이전 state-of-the-art 대비 6.6%p 향상이며 상대 향상률은 36%라고 주장한다. 또한 처리 시간도 Titan X GPU 기준 약 1.5초로, MPA의 8.7초보다 빠르다고 보고한다.

SBD에서는 end-to-end 모델이 $AP^r_{0.5}=62.0$, $AP^r_{0.7}=44.8$, $AP^r_{\text{vol}}=55.4$, Matching IoU 47.3을 기록했다. piecewise 대비도 뚜렷하게 향상되었고, 기존 IIS나 MNC와 비교해 높은 threshold와 stricter metric인 Matching IoU에서 강점을 보였다. 특히 저자들은 공개된 MNC 코드로 직접 Matching IoU를 측정하여 자사 방법이 8.3%p 높았다고 적고 있다.

semantic segmentation 성능도 instance finetuning 이후 개선되었다는 점이 흥미롭다. VOC에서는 mean IoU가 74.2%에서 75.1%로, SBD에서는 71.5%에서 72.5%로 증가했다. 저자들은 semantic segmentation과 instance segmentation이 매우 밀접한 과제이므로, 더 세밀한 instance supervision이 semantic prediction까지 개선한다고 본다.

Cityscapes test set에서는 ResNet-101 기반 semantic module과 모든 unary term을 사용해 평가했다. 결과는 AP 20.0, AP at 0.5 38.8, AP 100m 32.6, AP 50m 37.6으로, 당시 비교 대상인 SAIS, DWT, InstanceCut, Pixel Encoding 등을 모두 앞섰다. 논문은 이를 새로운 state-of-the-art라고 제시한다. 다만 본문에는 Cityscapes 실험의 세부 학습 설정이나 detector 설정이 VOC/SBD만큼 상세히 풀려 있지는 않다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 instance segmentation을 전역적인 픽셀 labeling 문제로 설계하여, proposal 기반 방법들의 구조적 약점을 직접 겨냥했다는 점이다. semantic segmentation을 먼저 수행하고 detector를 보조 cue로 사용함으로써, false positive detection을 무조건 segmentation으로 바꾸지 않는다. 실제 qualitative 결과에서도 잘못된 detection을 무시할 수 있음을 보였다. 또한 bounding box가 객체 전체를 덮지 못하는 경우에도 Global term 덕분에 box 바깥까지 복원할 수 있다는 점은 당시 detection-refinement 방식과 분명히 구별된다.

두 번째 강점은 dynamic CRF와 permutation-aware loss를 결합해 “가변 개수의 instance”를 자연스럽게 다룬 점이다. 고정된 최대 instance 수를 가정하지 않고 이미지마다 detector 개수에 맞춰 label 공간을 바꾸는 설계는 실제 이미지 분포에 더 잘 맞는다. 또한 matching 후 cross-entropy loss를 쓰는 방식은 구조가 비교적 단순하면서도 instance label symmetry 문제를 실용적으로 해결한다.

세 번째 강점은 높은 IoU threshold에서 강한 성능을 보인다는 점이다. 논문 전체의 메시지는 단순히 “객체를 찾는다”가 아니라 “정확한 경계를 가진 mask를 만든다”에 있다. VOC와 SBD 모두에서 높은 threshold 성능 향상은 이 설계가 coarse mask가 아니라 precise segmentation에 유리하다는 점을 뒷받침한다.

반면 한계도 분명하다. 가장 직접적인 한계는 detector 의존성이다. 저자들도 failure case에서 detection이 없는 객체는 instance segmentation으로 복원하지 못한다고 인정한다. semantic segmentation이 객체를 알아도, 대응되는 detection이 없으면 해당 instance를 형성할 수 없다. 즉 false positive에는 강하지만 false negative detection에는 여전히 취약하다.

또한 shape prior는 occlusion 문제를 완전히 해결하지 못한다. 비슷한 모양의 가축이나 작은 가려진 객체에 대해서는 여전히 인스턴스가 합쳐지거나 누락되는 사례가 보인다. shape template 기반 접근은 differentiable하다는 장점은 있지만, 복잡한 articulated object나 다양성이 큰 객체군에는 표현력이 부족할 수 있다.

학습 및 추론 구조도 다소 무겁다. semantic CRF-RNN과 instance CRF-RNN이 모두 들어가고, 배치 크기 1과 매우 작은 learning rate, gradient clipping이 필요했다. 이는 최적화가 까다롭고 구현 복잡도가 높다는 뜻이다. 또한 detector는 joint training되지 않기 때문에 전체 시스템이 완전한 단일 네트워크는 아니다. 저자들도 future work로 semantic segmentation, detection, instance segmentation의 joint training을 제안한다.

비판적으로 보면, 이 논문은 “proposal-free” 성격을 가지지만 실제로는 detector 출력을 instance 후보의 기반으로 삼기 때문에 완전히 detector-free라고 하긴 어렵다. 따라서 철학적으로는 bottom-up에 가깝지만, 실질적으로는 semantic segmentation과 detection을 결합한 hybrid system이다. 그럼에도 proposal을 독립 처리하지 않고 image-level reasoning을 수행한다는 차별점은 유지된다.

## 6. 결론

이 논문은 semantic segmentation을 출발점으로 하고, detector 출력과 shape prior를 결합한 dynamic CRF를 통해 픽셀 단위 instance segmentation을 수행하는 end-to-end 접근을 제안했다. 핵심 기여는 세 가지로 정리할 수 있다. 첫째, 이미지마다 다른 수의 instance를 처리할 수 있는 dynamically instantiated instance CRF를 설계했다. 둘째, Box, Global, Shape unary와 pairwise CRF를 결합해 detection error와 occlusion에 더 강건한 instance reasoning을 구현했다. 셋째, proposal 기반 후처리 없이도 정밀한 segmentation map을 직접 생성하며, 높은 IoU 기준에서 강한 성능을 보였다.

실용적 관점에서 이 연구는 “정확한 경계”, “전역적 일관성”, “가변 instance 수 처리”라는 세 요소를 동시에 추구한 초기의 의미 있는 시도라고 볼 수 있다. 이후 등장한 더 강력한 instance segmentation 계열 모델들과 비교하면 구조는 다소 복잡하고 detector 의존성도 남아 있지만, semantic segmentation과 instance segmentation을 긴밀히 연결하고 CRF를 end-to-end 학습 안에 넣었다는 점은 당시로서 분명한 기여다. 향후 연구 측면에서도 joint multi-task learning, detector와 segmenter의 통합 학습, 더 유연한 shape modeling으로 확장될 여지가 크다.
