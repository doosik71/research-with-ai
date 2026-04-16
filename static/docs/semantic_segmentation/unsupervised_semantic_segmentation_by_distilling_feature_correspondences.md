# Unsupervised Semantic Segmentation by Distilling Feature Correspondences

- **저자**: Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2203.08414

## 1. 논문 개요

이 논문은 **아무런 인간 주석 없이** 이미지 내 각 픽셀을 의미 있는 semantic category로 나누는 **unsupervised semantic segmentation** 문제를 다룬다. 일반적인 semantic segmentation은 픽셀 단위 정답 라벨이 필요하지만, 이런 라벨은 분류나 detection보다 훨씬 비싸고 만들기 어렵다. 특히 의학, 생물학, 천문학처럼 정답 ontology 자체가 애매하거나 전문가 지식이 필요한 영역에서는 supervised setting이 현실적으로 어렵다.

기존의 비지도 semantic segmentation 연구들은 보통 하나의 end-to-end 프레임워크 안에서 feature learning과 clustering을 동시에 해결하려고 했다. 그러나 저자들은 이 두 문제를 분리해서 본다. 즉, 이미 강력한 self-supervised backbone이 semantic하게 꽤 괜찮은 dense feature를 만들고 있으므로, 굳이 backbone을 다시 학습시키기보다 그 feature 사이의 correspondence를 잘 보존하면서 더 compact한 segmentation representation으로 **distill**하는 것이 핵심이라고 주장한다.

이 문제의 중요성은 명확하다. semantic segmentation이 실제 응용에서 매우 유용하지만 annotation cost가 너무 높기 때문에, 주석 없이도 category discovery와 segmentation을 동시에 수행할 수 있다면 활용 범위가 크게 넓어진다. 이 논문은 바로 그 방향에서, pretrained self-supervised feature를 활용하는 **STEGO**라는 새로운 방법을 제안하고, CocoStuff와 Cityscapes에서 당시 SOTA를 크게 넘어서는 결과를 보였다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. **좋은 self-supervised visual feature는 이미 semantic consistency를 어느 정도 갖고 있으며, 특히 feature 간 correlation pattern이 실제 semantic label co-occurrence와 강하게 연관된다.** 다시 말해, 같은 물체나 같은 semantic class에 속하는 픽셀들은 label이 없어도 backbone feature space에서 서로 비슷한 correspondence를 보인다는 것이다.

저자들은 이 관찰을 바탕으로, 비지도 segmentation의 핵심을 “새 feature를 처음부터 배우는 것”이 아니라 “이미 존재하는 pretrained feature의 correspondence structure를 segmentation-friendly한 low-dimensional code로 압축하는 것”으로 재정의한다. 이때 중요한 점은 단순히 feature를 따라 하는 것이 아니라, feature correspondence가 나타내는 **관계 정보**를 유지하면서 cluster가 잘 생기도록 만드는 것이다.

기존 방법과의 차별점은 분명하다. IIC나 PiCIE는 augmentation 기반 invariance/equivariance를 통해 representation과 clustering을 함께 학습한다. 반면 STEGO는 backbone을 **freeze**하고, segmentation head만 학습하여 pretrained feature의 구조를 활용한다. 따라서 더 가볍고 효율적이며, 앞으로 self-supervised backbone이 더 좋아질수록 STEGO도 자연스럽게 더 좋아질 수 있는 구조다. 저자들은 특히 DINO feature가 semantic label co-occurrence를 매우 잘 예측한다는 실험적 근거를 제시해 이 설계를 정당화한다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 네 단계로 이해할 수 있다. 첫째, pretrained self-supervised backbone에서 dense feature map을 추출한다. 둘째, 이 feature들 사이의 correspondence를 계산한다. 셋째, 작은 segmentation head를 학습해 이 correspondence 구조를 더 compact한 segmentation feature로 distill한다. 넷째, 학습된 연속형 feature를 clustering하고 CRF로 후처리해 최종 segmentation label을 만든다.

논문은 두 이미지의 dense feature tensor를 각각 $f \in \mathbb{R}^{C \times H \times W}$, $g \in \mathbb{R}^{C \times I \times J}$로 둔다. 여기서 각 위치의 feature vector 사이 cosine similarity를 계산해 feature correspondence tensor를 만든다:

$$
F_{hwij} := \sum_c \frac{f_{chw}}{|f_{hw}|} \frac{g_{cij}}{|g_{ij}|}
$$

이 식은 이미지 1의 위치 $(h,w)$와 이미지 2의 위치 $(i,j)$가 feature space에서 얼마나 비슷한지 나타낸다. 같은 이미지 안에서도 계산할 수 있고, 서로 비슷한 다른 이미지 사이에서도 계산할 수 있다. 저자들은 이 $F$가 실제 ground-truth segmentation label의 co-occurrence와 매우 높은 상관을 보인다고 실험적으로 확인한다. 즉, $F$는 “감독 신호처럼 쓸 수 있는 비지도 관계 정보”다.

이제 backbone $N$은 입력 이미지 $x$를 feature tensor $f$로 바꾸는 함수이고, segmentation head $S$는 이 $f$를 더 낮은 차원의 segmentation code로 바꾸는 함수다. 논문에서는 backbone은 고정하고, segmentation head만 학습한다. 출력은 $s = S(f)$이며, 이 $s$ 역시 각 픽셀 위치에 대해 feature vector를 갖는다. 목표는 이 $s$들이 semantic하게 잘 묶이는 compact cluster를 형성하도록 만드는 것이다.

가장 핵심적인 부분은 correspondence distillation loss이다. segmentation feature 사이에도 correspondence tensor를 만들 수 있고, 이를 이용해 원래 feature correspondence와 segmentation correspondence가 정렬되도록 만든다. 가장 단순한 형태는 다음과 같다:

$$
L_{\text{simple-corr}}(x,y,b) := - \sum_{hwij} (F_{hwij} - b) S_{hwij}
$$

여기서 $b$는 uniform negative pressure를 주는 hyperparameter이다. 직관적으로 보면, 원래 feature correspondence $F_{hwij}$가 큰 위치쌍은 segmentation feature도 서로 가깝게 만들고, $F_{hwij}$가 낮은 위치쌍은 멀어지게 만든다. 그런데 저자들은 이 단순 형태가 불안정하다고 보고한다. 특히 상관이 약한 위치쌍을 과도하게 anti-align시키면 학습이 불안정해지고 co-linearity 문제가 생긴다고 설명한다.

그래서 두 가지 수정을 도입한다. 첫째는 **0-Clamp**이다. segmentation correspondence를 음수까지 강제로 밀어내지 않고, $0$ 아래는 잘라낸다. 즉, 약하게 관련된 쌍은 orthogonal 정도로만 유지하려는 것이다. 둘째는 **Spatial Centering (SC)**이다. 작은 물체는 correspondence가 매우 좁은 공간에만 집중되므로, 평균을 빼서 지역적으로 두드러지는 correspondence가 더 잘 드러나게 한다:

$$
F^{SC}_{hwij} := F_{hwij} - \frac{1}{IJ}\sum_{i'j'} F_{hwi'j'}
$$

이를 반영한 최종 loss는 다음과 같다:

$$
L_{\text{corr}}(x,y,b) := - \sum_{hwij} (F^{SC}_{hwij} - b)\max(S_{hwij}, 0)
$$

이 식의 의미는 비교적 명확하다. 원래 feature correspondence가 공간 평균보다 높고 threshold $b$보다 충분히 크면, segmentation feature도 서로 유사해지도록 유도한다. 반대로 correspondence가 충분히 강하지 않으면 강한 anti-alignment 대신 0 근처로 억제한다. 결과적으로 semantic consistency는 유지하면서 cluster structure는 더 안정적으로 형성된다.

STEGO는 이 loss를 세 종류의 image pair에 대해 사용한다. 자기 자신과의 pair인 self pair, feature-space에서 가까운 KNN pair, 그리고 무작위 random pair이다. self와 KNN은 주로 positive signal을 제공하고, random pair는 주로 negative signal을 제공한다. 전체 loss는 다음과 같다:

$$
L = \lambda_{\text{self}} L_{\text{corr}}(x, x, b_{\text{self}})
+ \lambda_{\text{knn}} L_{\text{corr}}(x, x_{\text{knn}}, b_{\text{knn}})
+ \lambda_{\text{rand}} L_{\text{corr}}(x, x_{\text{rand}}, b_{\text{rand}})
$$

여기서 $\lambda$는 각 loss의 중요도를, $b$는 positive와 negative pressure의 균형을 조절한다. 논문은 대체로 $\lambda_{\text{self}} \approx \lambda_{\text{rand}} \approx 2\lambda_{\text{knn}}$가 잘 작동했다고 보고한다. 또한 KNN은 backbone feature를 global average pooling해서 만든 global image embedding을 기준으로 찾는다.

작은 물체를 더 잘 보기 위해 저자들은 **5-crop training**을 사용한다. 이미지를 다섯 개 crop으로 나눠 각각을 별도 이미지처럼 취급하고, 이 crop 단위로 KNN을 찾는다. 이렇게 하면 세밀한 부분을 더 잘 볼 수 있고, KNN 품질도 좋아진다. 논문은 이 과정이 Cityscapes와 CocoStuff 모두에서 성능 향상에 기여했다고 말한다.

마지막으로 segmentation head가 출력한 연속형 feature를 cosine distance 기반 minibatch K-Means로 clustering해서 discrete label로 바꾸고, 이후 CRF를 적용해 boundary를 더 선명하게 다듬는다. 따라서 STEGO는 “dense self-supervised feature extraction → correspondence distillation → clustering → CRF refinement”의 흐름으로 이해할 수 있다.

또 하나 중요한 이론적 해석은, 이 loss가 Potts model 또는 continuous Ising model의 energy minimization과 연결된다는 점이다. 논문은 그래프의 각 노드를 데이터셋 전체의 픽셀로 보고, edge weight를 feature similarity로 두면 STEGO의 loss가 일종의 graph energy 최대우도추정과 동등하다고 보인다. 이것은 STEGO가 단순 heuristic이 아니라, 에너지 기반 그래프 최적화 관점에서도 해석 가능한 방법임을 의미한다. 다만 이 부분은 이론적 정당화에 가깝고, 실제 구현은 SGD로 segmentation head를 학습하는 형태다.

구현 측면에서는 DINO의 ViT-Base teacher weights를 backbone으로 사용했고, backbone 마지막 spatial feature에 channel-wise dropout $p=0.1$을 적용한다. segmentation head는 linear branch와 2-layer ReLU MLP branch를 더한 구조이며, 최종적으로 70차원 vector를 출력한다. optimizer는 Adam, learning rate는 $0.0005$, batch size는 32다. backbone을 재학습하지 않기 때문에 단일 NVIDIA V100에서 2시간 이내로 학습 가능했다고 서술한다.

## 4. 실험 및 결과

실험은 주로 CocoStuff 27-class와 Cityscapes 27-class에서 수행된다. 평가는 두 관점에서 이뤄진다. 하나는 **linear probe**로, 학습된 segmentation feature 위에 선형 분류기를 얹어 label을 예측하는 방식이다. 이것은 feature 자체의 품질을 본다. 다른 하나는 **clustering 평가**로, 비지도 cluster를 Hungarian matching으로 정답 class와 정렬한 뒤 accuracy와 mIoU를 계산한다. 이것이 진정한 unsupervised segmentation 성능에 더 가깝다.

가장 중요한 결과는 CocoStuff에서의 큰 성능 향상이다. Table 1에 따르면, STEGO는 unsupervised clustering 기준으로 **56.9 accuracy, 28.2 mIoU**를 기록했다. 이전 강한 baseline인 PiCIE + H는 **50.0 accuracy, 14.4 mIoU**였으므로, mIoU 기준으로 약 **+14** 정도 향상이다. linear probe 기준으로도 STEGO는 **76.1 accuracy, 41.0 mIoU**를 기록해, DINO backbone feature를 그냥 쓰는 것보다 훨씬 강하다. 즉, backbone 자체도 좋지만 correspondence distillation이 추가 이득을 준다.

Cityscapes에서도 비슷한 경향이 나타난다. STEGO는 **73.2 accuracy, 21.0 mIoU**를 기록했고, PiCIE는 **65.5 accuracy, 12.3 mIoU**였다. 따라서 unsupervised mIoU는 약 **+8.7**, accuracy는 **+7.7** 향상이다. 저자들은 backbone을 dataset-specific하게 fine-tune하지 않고도 ImageNet으로 self-supervised pretraining된 DINO 하나로 두 데이터셋 모두를 잘 해결했다는 점을 강조한다.

정성적 결과에서도 STEGO는 세부 구조를 더 잘 잡는다. 논문은 CocoStuff에서 말 다리, 작은 새, 복잡한 장면의 객체 세부를 PiCIE보다 더 잘 분리한다고 설명한다. 이는 DINO backbone의 더 높은 해상도 feature, 5-crop training, 그리고 CRF 후처리가 함께 작동한 결과로 해석된다. Cityscapes에서는 사람, 도로, 인도, 차량, 표지판 등을 비교적 높은 fidelity로 찾아낸다.

Ablation study도 설계 선택을 잘 뒷받침한다. ViT-Small에서 0-Clamp, 5-Crop, SC, CRF를 순차적으로 추가하면 unsupervised mIoU가 7.3에서 24.5까지 크게 오른다. 최종적으로 ViT-Base backbone과 CRF를 함께 쓰면 CocoStuff에서 **56.9 accuracy, 28.2 mIoU**, linear probe **76.1 accuracy, 41.0 mIoU**에 도달한다. 이는 backbone 품질뿐 아니라 loss 설계와 training trick들이 모두 중요하다는 뜻이다.

부록에서는 Potsdam-3 aerial image segmentation에서도 결과를 제시하는데, STEGO는 **77.0 accuracy**를 기록해 IIC의 **65.1**보다 약 12%p 높다. 다만 본문 핵심 벤치마크는 CocoStuff와 Cityscapes이며, Potsdam 결과는 추가 실험으로 제시된다.

저자들은 feature correspondence 자체의 품질도 분석한다. DINO feature correspondence는 ground-truth label co-occurrence를 매우 잘 예측하며, precision 90%에서 recall 50% 수준의 성능을 보였다고 한다. 흥미롭게도 최종 STEGO는 자신이 distill하는 supervisory signal보다 더 좋은 label predictor가 되는데, 저자들은 이것을 distillation이 corpus 전체에서 일관성을 증폭시키는 효과 때문이라고 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법론이 모두 설득력 있다는 점이다. 먼저, pretrained self-supervised feature가 이미 semantic correspondence를 품고 있다는 관찰이 매우 중요하며, 저자들은 이를 단순 주장으로 두지 않고 precision-recall 분석으로 직접 보여준다. 그 위에 segmentation head만 학습하는 구조는 단순하면서도 효과적이고, 실제로 대규모 성능 향상으로 이어진다.

또 다른 강점은 학습 효율성이다. backbone을 freeze하기 때문에 학습 비용이 낮고, 앞으로 backbone이 더 좋아질수록 구조 변경 없이 그대로 성능 향상을 기대할 수 있다. 또한 self, KNN, random pair를 조합한 loss 설계는 positive/negative signal의 역할을 명확히 나눠주며, 0-Clamp와 spatial centering 같은 수정도 단순하지만 실질적인 안정성 개선으로 이어진다. 실험적으로도 ablation이 이 주장을 잘 지지한다.

이론적 해석도 강점이다. STEGO loss를 Potts model과 energy-based graph optimization으로 연결한 것은 방법의 성격을 더 잘 이해하게 해 준다. 특히 데이터셋 전체 픽셀 그래프 위에서 parametric function을 학습한다는 관점은, 비모수 spectral clustering류 방법보다 확장성과 일반화 측면에서 장점이 있음을 잘 설명한다.

한계도 분명하다. 가장 중요한 한계는 **backbone 품질에 크게 의존한다는 점**이다. 부록 A.6에서 저자들은 STEGO의 혼동 패턴 상당수가 DINO feature correspondence의 혼동을 그대로 반영한다고 분석한다. 예를 들어 CocoStuff에서 “food (thing)”과 “food (stuff)”를 혼동하거나, wall과 ceiling을 혼동하는 문제는 STEGO 자체보다 원래 DINO feature 구조나 label ontology의 모호성과 더 관련이 있을 수 있다. 즉, STEGO는 backbone의 semantic bias를 증폭하거나 정제할 수는 있어도, 근본적으로 새 semantic structure를 창조하는 것은 아니다.

또 다른 한계는 hyperparameter tuning이다. 부록 A.11에서 저자들은 ground-truth 없이 hyperparameter를 잡는 것이 여전히 어렵고 outstanding challenge라고 직접 인정한다. $\lambda$와 $b$의 균형이 성능에 중요하며, positive pressure와 negative pressure가 조금만 어긋나도 cluster가 무너지거나 collapse될 수 있다. 따라서 완전한 “hands-off” 방법이라고 보기는 어렵다.

평가 자체의 한계도 논문이 스스로 지적한다. unsupervised segmentation에서는 label ontology가 본질적으로 임의적일 수 있다. 예를 들어 어떤 class 구분은 인간이 정의한 분류 체계이지, feature space에서 자연스럽게 드러나는 semantic division이 아닐 수 있다. 이런 경우 clustering mIoU가 낮다고 해서 representation이 꼭 나쁜 것은 아니다. 저자들이 linear probe를 중요한 보조 지표로 보는 이유도 여기에 있다.

비판적으로 보면, 이 논문은 “비지도 segmentation”을 다루지만 실제 semantic category discovery의 근본 문제를 모두 해결한 것은 아니다. backbone의 ImageNet 사전학습이 매우 강한 inductive bias를 제공하고, KNN 기반 pairing도 dataset 구조를 활용한다. 물론 논문은 이것을 숨기지 않고 명확히 설명한다. 따라서 이 방법의 강점은 “완전한 from-scratch discovery”라기보다 “강한 self-supervised prior를 semantic segmentation 형태로 정제하는 능력”에 있다고 보는 것이 정확하다.

## 6. 결론

이 논문은 unsupervised semantic segmentation에서 중요한 방향 전환을 제시한다. 핵심은 feature learning과 cluster compactification을 분리하고, 강력한 self-supervised backbone이 이미 가진 dense correspondence structure를 segmentation head로 distill하는 것이다. 이를 위해 제안된 STEGO는 correspondence-based contrastive loss, self/KNN/random pair 설계, 0-Clamp와 spatial centering, clustering과 CRF 후처리를 결합해 매우 강한 성능을 달성했다.

실험적으로 STEGO는 CocoStuff와 Cityscapes에서 당시 기존 방법을 큰 폭으로 앞질렀고, qualitative result에서도 더 일관되고 세밀한 segmentation을 보여준다. 이 연구의 실제적 의미는 크다. annotation이 비싸거나 ontology가 불분명한 분야에서, pretrained self-supervised model을 활용해 semantic structure를 자동으로 발견하고 분할하는 방향의 가능성을 구체적으로 보여주었기 때문이다.

향후 연구 관점에서도 의미가 있다. 더 좋은 self-supervised backbone, 더 안정적인 hyperparameter selection, ontology ambiguity를 반영한 새로운 평가 방식, CRF 없이도 경계 정제를 수행할 수 있는 구조가 결합되면 이 계열 방법은 더 발전할 수 있다. 이 논문은 그 출발점으로서, “좋은 비지도 feature는 이미 semantic하다”는 사실을 segmentation 문제에 효과적으로 연결한 중요한 작업이라고 평가할 수 있다.
