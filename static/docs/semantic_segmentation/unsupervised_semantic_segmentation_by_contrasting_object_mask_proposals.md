# Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals

- **저자**: Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Luc Van Gool
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2102.06191

## 1. 논문 개요

이 논문은 **라벨 없이 semantic segmentation에 유용한 pixel-level representation을 학습하는 문제**를 다룬다. 저자들의 목표는, 사람이 만든 pixel annotation 없이도 각 픽셀이 어떤 의미적 그룹에 속하는지 구분할 수 있는 embedding space를 만드는 것이다. 이렇게 학습된 표현은 두 가지 방식으로 활용될 수 있다. 하나는 embedding을 바로 clustering해서 fully unsupervised semantic segmentation을 수행하는 것이고, 다른 하나는 소량의 라벨이 있을 때 이를 초기 표현으로 사용해 semi-supervised 또는 transfer learning 성능을 높이는 것이다.

연구 문제는 명확하다. 기존 self-supervised representation learning은 주로 image-level instance discrimination에 집중해 왔고, semantic segmentation처럼 **픽셀 단위로 의미를 나눠야 하는 과제**에는 직접적으로 맞지 않는다. 예를 들어 같은 이미지 안에 자주 함께 등장하는 foreground와 background는 image-level contrastive learning에서는 비슷하게 인코딩될 수 있다. 그러면 semantic class를 구분해야 하는 segmentation에는 불리하다.

이 문제가 중요한 이유는 semantic segmentation이 자율주행, 증강현실, 인간-컴퓨터 상호작용 등에서 핵심 과제이지만, 대규모 pixel-wise annotation은 매우 비싸기 때문이다. 약한 지도학습이나 반지도학습은 주석 비용을 줄이지만 여전히 어떤 형태로든 supervision이 필요하다. 이 논문은 그보다 더 강한 조건인 **fully unsupervised setting**에서 PASCAL 같은 어려운 benchmark를 다루려는 첫 시도라는 점에서 의미가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **object mask proposal을 중간 단계의 visual prior로 사용하고, 이를 contrastive objective 안에 넣어 pixel embedding을 학습하는 것**이다. 저자들은 semantic segmentation을 처음부터 end-to-end clustering으로 직접 풀기보다, 먼저 “같이 속할 가능성이 높은 픽셀 묶음”을 찾고, 이를 이용해 embedding을 배우는 것이 더 안정적이라고 본다.

이 prior는 논문에서 **shared pixel ownership**이라는 관점으로 설명된다. 어떤 두 픽셀이 같은 object mask 안에 있다면, 이 둘은 embedding space에서 가까워져야 한다고 가정한다. 반대로 다른 object에 속하는 픽셀들은 떨어지도록 학습한다. 이렇게 하면 모델이 단순히 color, contrast, texture 같은 low-level cue에만 의존하지 않고, object 또는 object part 수준의 더 semantically meaningful한 구조를 배우도록 유도할 수 있다.

기존 접근과의 차별점은 세 가지로 요약된다. 첫째, proxy task를 풀어서 우연히 segmentation에 도움 되는 표현을 얻는 방식이 아니라, segmentation에 더 직접적으로 관련된 object prior를 사용한다. 둘째, end-to-end clustering처럼 초기화에 민감한 구조 대신 prior와 feature learning을 분리한다. 셋째, boundary나 superpixel 같은 low-level grouping 대신 object-level information을 담는 mid-level prior를 쓴다. 저자들은 이것이 semantic grouping에 더 적합하다고 주장한다.

## 3. 상세 방법 설명

전체 방법은 **2단계 구조**이며, 논문에서는 이를 **MaskContrast**라고 부른다.

첫 번째 단계는 **object mask proposal 생성**이다. 저자들은 saliency estimation을 이용해 이미지에서 salient object mask를 얻는다. 이때 saliency estimator는 supervised 방식과 unsupervised 방식을 모두 실험한다. 논문의 핵심 설정은 가능한 한 외부 supervision에 덜 의존하는 것이므로, unsupervised saliency를 중요한 선택지로 본다. 다만 unsupervised saliency model을 MSRA에서 학습한 뒤 이를 PASCAL 같은 target dataset에 바로 적용하면 mask quality가 낮아지기 때문에, pseudo-label을 이용해 BAS-Net을 다시 학습하는 간단한 bootstrapping 절차를 추가한다.

두 번째 단계는 **contrastive learning으로 pixel embedding 학습**이다. 입력 이미지 $X$와 그 augmentation $X^+$를 positive pair로 만들고, 다른 이미지들에서 온 object crop들을 negative로 사용한다. 이때 일반적인 image-level contrastive learning과 달리, 비교 단위는 이미지 전체가 아니라 픽셀과 object prototype이다.

논문은 pixel embedding function을 $\Phi_\theta : X \to Z$로 둔다. 각 픽셀 $i$는 정규화된 $D$차원 hypersphere 위의 벡터 $z_i$로 매핑된다. 정규화된 embedding space를 쓰는 이유는 출력 스케일을 제한해 loss가 다른 설계 요소에 덜 의존하도록 하기 위함이다.

먼저 object mask $M_n$의 평균 embedding은 다음과 같이 정의된다.

$$
z_{M_n} = \frac{1}{|M_n|} \sum_{i \in M_n} z_i
$$

즉, mask 내부 픽셀 embedding들의 평균으로 object-level prototype을 만든다.

이제 학습 objective는 **pull-force**와 **push-force**로 설명된다.

pull-force는 같은 object에 속한 픽셀들을 서로 가깝게 만드는 힘이다. 원래는 같은 mask 안의 모든 픽셀 쌍 $(i,j)$에 대해 유사도를 높일 수 있지만, 그러면 계산량이 픽셀 수에 대해 제곱으로 커진다. 그래서 논문은 각 픽셀을 해당 object의 mean embedding과 맞추는 방식으로 바꾼다. 즉, 픽셀 $z_i$는 자신이 속한 object의 prototype 쪽으로 당겨진다.

push-force는 embedding collapse를 막고, 서로 다른 object를 구분하게 만드는 힘이다. 저자들은 augmentation된 동일 object view를 positive로, 다른 object들을 negative로 사용하는 contrastive setup이 이 역할을 수행한다고 본다. 중요한 점은 “다른 object”가 무조건 멀어져야 하는 것이 아니라, **visually similar object끼리는 상대적으로 더 가깝고 dissimilar object끼리는 더 멀어지도록** embedding structure가 만들어진다는 것이다.

기본 contrastive loss는 image-level 표현에 대해 다음과 같이 소개된다.

$$
L = - \log \frac{\exp(\Psi_\eta(X)^T \cdot \Psi_\eta(X^+) / \tau)}{\sum_{k=0}^{K} \exp(\Psi_\eta(X)^T \cdot \Psi_\eta(X_k^-) / \tau)}
$$

여기서 $\tau$는 temperature이고, $K$는 negative 수다.

이를 pixel-level로 바꾸면, 픽셀 $i \in M_X$에 대한 loss는 다음과 같다.

$$
L_i = - \log \frac{\exp(z_i \cdot z_{M_{X^+}} / \tau)}{\sum_{k=0}^{K} \exp(z_i \cdot z_{M_{X_k^-}} / \tau)}
$$

이 식의 의미는 직관적이다. 픽셀 $i$는 augmentation된 같은 object의 평균 embedding $z_{M_{X^+}}$와는 가까워져야 하고, 다른 이미지에서 온 negative object prototype들과는 멀어져야 한다. 이렇게 해서 semantic segmentation에 유용한 pixel embedding space를 형성한다.

논문은 foreground pixel에만 이 contrastive loss를 적용한다. background는 한 이미지 안에 여러 object가 섞여 있을 수 있어 확실한 grouping 정보를 주지 못하기 때문이다. 그런데 이렇게 하면 네트워크가 이미지 전체를 같은 벡터로 보내는 식의 collapse가 생길 수 있다. 이를 막기 위해 저자들은 **별도의 linear head로 saliency mask를 예측하는 auxiliary loss**를 추가한다. pseudocode에서는 이 보조 손실을 binary cross-entropy loss로 계산한다. 최종 손실은 다음과 같이 이해할 수 있다.

$$
L_{\text{total}} = L_{\text{MaskContrast}} + L_{\text{aux}}
$$

구현 측면에서는 query encoder와 momentum-updated key encoder를 사용하고, negative prototype은 memory bank에 저장한다. 이는 MoCo 계열 학습과 유사한 구조다. 논문 pseudocode 기준으로 key encoder는 momentum update를 따르며, 현재 batch의 object prototype을 queue에 enqueue하고 오래된 prototype을 dequeue한다. 이 메커니즘은 negative distribution을 더 안정적으로 추정하게 해 준다.

실험 설정에서 사용된 주요 학습 조건은 다음과 같다. backbone은 dilated ResNet-50을 사용한 DeepLab-v3이며, 기본적으로 MoCo v2로 ImageNet pretraining된 가중치로 초기화한다. 학습은 60 epoch, batch size 64, SGD with momentum $0.9$, weight decay $10^{-4}$, initial learning rate $0.004$, poly learning rate schedule로 진행한다. embedding dimension은 $D=32$, temperature는 $\tau=0.5$, negative 수는 $K=128$이다. augmentation은 SimCLR와 유사한 설정을 쓰되, salient object가 일정 비율 이상 남도록 제약한다.

## 4. 실험 및 결과

주요 실험은 PASCAL dataset에서 수행되며, 추가로 COCO와 DAVIS-2016에서 transferability를 평가한다. 평가 방식은 단일한 하나의 프로토콜이 아니라, embedding이 얼마나 semantic structure를 잘 담는지를 보기 위해 여러 관점에서 측정된다.

먼저 **linear classifier 평가**에서는 backbone을 고정하고 그 위에 $1 \times 1$ convolution layer만 학습한다. 이 설정은 embedding 자체가 semantic class에 대해 선형적으로 잘 분리되는지를 테스트한다. 결과를 보면 saliency model feature만으로는 MIoU가 6.5에 불과하다. 반면 MaskContrast는 MoCo v2 initialization + unsupervised saliency일 때 58.4, supervised saliency일 때 62.2를 기록한다. supervised ImageNet classifier initialization을 쓸 경우 각각 61.0, 63.9까지 올라간다. 즉, 단순히 initialization이 좋아서가 아니라, **pixel-level contrastive learning 자체가 semantic structure를 강화**했다는 해석이 가능하다.

baseline 비교에서도 이러한 경향이 분명하다. proxy task 기반 방법들인 Co-Occurrence 13.5, CMP 16.5, Colorization 25.5보다 훨씬 높고, clustering 기반 IIC 28.0보다도 크다. contrastive learning baseline인 Instance Discrimination 26.8, MoCo v2 45.0, InfoMin 45.2, SWAV 50.7보다도 높다. boundary-based 방법인 SegSort 36.2, Hierarchical Grouping 48.8 역시 넘는다. 저자들의 주장은 여기서 뚜렷하다. segmentation에는 image-level contrastive objective보다 **pixel-level object-aware contrastive objective**가 더 적절하다.

ablation study도 핵심 설계를 뒷받침한다. mask proposal 방식 비교에서 hierarchical segmentation은 30.5, unsupervised saliency model은 58.4, supervised saliency model은 62.2를 보였다. 저자들은 작은 region 단위보다 object 또는 object part 수준의 mask가 더 유용하다고 해석한다. training mechanism 분석에서는 augmented views, memory bank, momentum encoder를 모두 켰을 때 58.4로 가장 좋았다. 모두 끄면 52.4에 그쳤다. hyperparameter study에서는 temperature $\tau \in [0.1,1]$에서 $56.2 \pm 1.4$, negative 수 $K \in [64,1024]$에서 $57.0 \pm 0.6$으로 보고되어, 방법이 극도로 민감하지는 않다고 주장한다.

다음으로 **K-Means clustering 평가**는 embedding을 offline clustering으로 바로 semantic group으로 나눌 수 있는지 পরীক্ষা한다. 이 실험은 fully unsupervised segmentation의 핵심에 가깝다. 예측 cluster 수는 ground-truth class 수와 같게 맞추고, Hungarian matching으로 대응시킨 뒤 MIoU를 계산한다. 기존 방법들은 4~10 수준의 매우 낮은 K-Means 성능을 보였고, MoCo 기반도 4.7에 불과했다. 반면 MaskContrast는 MoCo init + unsupervised saliency에서 35.0, supervised saliency에서 38.9, supervised init + unsupervised saliency에서 41.6, supervised init + supervised saliency에서 44.2를 기록했다. 이는 이 embedding space가 단순 선형 probe뿐 아니라 **실제로 clustering 가능한 semantic geometry**를 가진다는 강한 증거다.

supplementary의 overclustering 결과도 흥미롭다. cluster 수를 21에서 500까지 늘리면 성능이 계속 증가한다. 예를 들어 supervised initialization + supervised saliency에서는 44.2에서 57.0까지 오른다. 이는 embedding space의 국소 neighborhood 안에는 동일하거나 시각적으로 유사한 object 픽셀들이 모여 있고, coarse한 class 수보다 더 세밀한 구조가 이미 형성돼 있음을 시사한다.

**semantic segment retrieval** 실험에서는 각 salient object의 평균 embedding을 구한 뒤 validation object의 nearest neighbor를 training set에서 찾는다. 7개 class 기준으로 SegSort 10.2, Hierarchical Grouping 24.6, MoCo v2 48.0인데 비해, MaskContrast는 unsupervised saliency에서 53.4, supervised saliency에서 62.3이다. 21개 class 전체 기준으로도 43.3과 49.6을 기록했다. 이 결과는 embedding이 단순한 분류용 특징이 아니라, object-level semantic similarity를 잘 반영한다는 뜻이다.

**transfer learning**에서는 모든 모델을 ImageNet에서 pretrain한 뒤 다른 dataset에 옮겨 평가한다. PASCAL linear classifier에서 MoCo v2는 45.0, MaskContrast는 55.4/57.2를 기록한다. COCO에서도 35.2 대비 45.0/47.2로 개선된다. DAVIS-2016 video object segmentation 전이 실험에서는 region similarity $J_m$가 77.1에서 78.0/82.0으로, contour accuracy $F_m$가 77.2에서 77.8/80.9로 개선된다. 저자들은 이를 근거로, 이 표현이 특정 dataset에만 과적합된 것이 아니라 object-centric invariance를 일반적으로 학습했다고 해석한다.

마지막으로 **semi-supervised fine-tuning**에서는 PASCAL의 일부 라벨만 사용한다. 1%, 2%, 5%, 12.5%, 100% label fraction에 대해, 단순 ImageNet supervised initialization보다 MaskContrast pretraining이 항상 더 좋다. 예를 들어 1% 라벨에서 baseline 43.4에 비해 MaskContrast는 50.5 또는 51.5이고, 2%에서는 55.2 대비 57.2 또는 59.6이다. 100% 라벨에서는 차이가 78.0 대비 78.4 또는 78.6으로 줄어든다. 즉, label이 적을수록 이 사전학습의 이점이 크다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 fully unsupervised semantic segmentation을 위해 **문제에 더 직접적으로 맞는 representation learning objective를 설계했다는 점**이다. 기존 self-supervised learning의 성과를 segmentation에 그대로 가져오지 않고, 왜 image-level contrastive learning이 픽셀 의미 분리에 충분하지 않은지를 분명히 짚는다. 그리고 object mask prior를 이용해 이를 해결하는 구조를 제안한다. 이 설계는 논리적으로도 일관되고, ablation과 benchmark 결과로도 상당 부분 뒷받침된다.

또 다른 강점은 prior와 clustering을 분리한 점이다. 논문은 end-to-end clustering이 network initialization과 low-level cue에 쉽게 끌릴 수 있다고 비판하고, 미리 얻은 object prior를 사용함으로써 이 문제를 완화한다. 결과적으로 linear probe, K-Means, retrieval, transfer, semi-supervised fine-tuning 등 여러 평가에서 일관되게 이득을 보인다. 특히 PASCAL에서 fully unsupervised K-Means clustering이 의미 있는 수준으로 작동했다는 점은 논문의 핵심 기여다.

또한 이 논문은 unsupervised saliency와 supervised saliency를 모두 시험해, 제안 방법의 잠재력과 상한을 함께 보여 준다. supervised saliency가 더 좋은 결과를 내지만, unsupervised saliency만으로도 기존 방법보다 확실한 향상이 있음을 보인다. 이는 mid-level visual prior 자체의 가치가 크다는 주장을 강화한다.

한계도 분명하다. 가장 큰 한계는 object mask proposal이 **salient object detector에 크게 의존한다는 점**이다. saliency 기반 mask는 한 이미지 안의 소수의 두드러진 object만 잘 잡을 가능성이 높고, 복잡한 장면이나 많은 object가 있는 경우에는 충분하지 않을 수 있다. 저자들도 이 점을 명시적으로 인정하며, 더 복잡한 데이터셋을 위해서는 다른 proposal mining 방식이나 추가 sensory data가 필요할 수 있다고 말한다.

또 하나의 제한은 background 처리 방식이다. 논문은 background를 contrastive loss에서 제외하고 auxiliary saliency prediction head로 collapse를 방지한다. 이는 실용적인 선택이지만, background 내부의 다양한 semantic category를 직접 모델링하지는 못한다. 따라서 scene parsing처럼 background class 구분이 중요한 설정에서는 구조적 한계가 있을 가능성이 있다. 논문은 이를 해결하는 일반식을 제시하지는 않는다.

비판적으로 보면, 이 방법은 “완전한 end-to-end unsupervised segmentation”이라기보다, **unsupervised saliency라는 중간 모듈의 품질에 의해 성패가 크게 좌우되는 2-stage system**이다. 물론 저자들은 이것을 숨기지 않고, 오히려 mid-level prior가 필요하다는 철학으로 정당화한다. 따라서 이 논문의 핵심 기여는 범용적인 순수 clustering 기법이라기보다, object-centric prior를 끌어들인 representation learning 전략이라고 보는 것이 더 정확하다.

## 6. 결론

이 논문은 unlabeled image dataset에서 semantic segmentation용 pixel embedding을 학습하기 위해, **saliency 기반 object mask proposal과 pixel-level contrastive learning을 결합한 2단계 프레임워크 MaskContrast**를 제안한다. 핵심은 같은 object에 속한 픽셀은 가까이, 다른 object의 픽셀은 멀어지도록 embedding을 학습한다는 것이다. 이를 위해 object mean embedding을 prototype처럼 사용하고, memory bank와 momentum encoder를 결합한 contrastive objective를 설계했다.

실험적으로는 PASCAL에서 linear probing, K-Means clustering, segment retrieval 모두 기존 unsupervised 방법보다 강한 성능을 보였고, COCO와 DAVIS로의 transfer 및 소량 라벨 fine-tuning에서도 이점을 입증했다. 특히 fully unsupervised setting에서 PASCAL 같은 어려운 데이터셋에 대해 의미 있는 semantic clustering 결과를 냈다는 점이 중요하다.

실제 적용 측면에서 이 연구는 dense prediction을 위한 self-supervised learning이 image-level pretext task만으로는 충분하지 않으며, **object-aware mid-level prior가 매우 중요할 수 있다**는 방향을 제시한다. 향후 연구에서는 saliency보다 더 풍부한 object proposal 소스, 복잡한 multi-object scene 처리, background semantic modeling 등을 확장함으로써 더 일반적인 unsupervised semantic segmentation으로 발전할 가능성이 크다.
