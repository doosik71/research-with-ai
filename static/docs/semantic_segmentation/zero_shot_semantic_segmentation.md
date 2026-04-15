# Zero-Shot Semantic Segmentation

- **저자**: Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, Patrick Pérez
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1906.00817

## 1. 논문 개요

이 논문은 **zero-shot semantic segmentation**이라는 새로운 문제를 제안한다. 목표는 학습 중 한 번도 픽셀 단위 정답을 본 적 없는 클래스, 즉 **unseen classes**에 대해서도 테스트 시 픽셀별 semantic label을 예측하는 것이다. 기존 semantic segmentation은 보통 모든 클래스에 대해 충분한 pixel-wise annotation이 있다고 가정하지만, 실제로는 클래스 수가 커질수록 모든 클래스에 대한 정밀한 segmentation annotation을 수집하는 비용이 매우 크다. 이 논문은 바로 그 확장성 문제를 정면으로 다룬다.

핵심 연구 문제는 다음과 같다. 학습 데이터에는 일부 **seen classes**만 존재하고, unseen classes에 대해서는 이미지나 annotation 없이 오직 클래스 이름의 semantic description만 주어질 때, semantic segmentation 모델이 어떻게 unseen classes까지 구분할 수 있는가? 더 나아가 테스트 시에는 seen과 unseen이 함께 등장하는 **generalized zero-shot setting**을 다룬다. 이는 unseen만 따로 분류하는 순수한 zero-shot classification보다 훨씬 어렵다. 모델은 unseen을 인식해야 할 뿐 아니라, seen으로 과도하게 치우치는 bias도 줄여야 한다.

이 문제는 중요하다. semantic segmentation이 더 많은 객체와 장면 범주를 다루려면 annotation-efficient한 방법이 필요하기 때문이다. 특히 픽셀 단위 주석은 이미지 분류보다 훨씬 비싸다. 따라서 텍스트 기반 semantic prior를 활용해 본 적 없는 클래스를 segmentation할 수 있다면, 실제 응용에서 클래스 확장 비용을 크게 줄일 수 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 명확하다. **unseen class의 실제 이미지 feature가 없더라도, 그 클래스 이름의 semantic embedding으로부터 pixel-level visual feature를 생성한 뒤, 그것을 segmentation classifier 학습에 사용하자**는 것이다. 즉, 기존 segmentation backbone이 만든 feature space 위에서 zero-shot learning을 수행한다.

이를 위해 제안한 모델이 **ZS3Net**이다. 구조적으로는 두 단계다. 먼저 seen classes의 실제 pixel feature와 클래스 이름의 word embedding 사이 관계를 학습해, 클래스 설명으로부터 시각 feature를 생성하는 generator를 만든다. 그런 다음 이 generator로 unseen classes의 synthetic feature를 많이 만들어, seen의 실제 feature와 함께 segmentation classifier를 다시 학습한다. 이렇게 하면 최종 classifier는 seen과 unseen 둘 다 예측할 수 있게 된다.

기존 zero-shot classification과의 차별점은, 이 논문이 그 개념을 **semantic segmentation의 pixel-level prediction 문제**로 확장했다는 점이다. 단순히 이미지 전체를 하나의 label로 분류하는 것이 아니라, downsampled pixel location마다 feature를 다루고, 그 위에서 unseen class용 synthetic training sample을 만든다. 또한 복잡한 장면에서는 객체 간 공간적 관계가 중요하다는 점을 반영해 **graph-context encoding**도 제안한다. 마지막으로, unseen class의 unlabeled pixel이 일부라도 있으면 pseudo-label 기반 self-training을 추가한 **ZS5Net**으로 성능을 더 끌어올린다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 세 단계로 볼 수 있다.

첫째, 기본 segmentation 모델로 **DeepLabv3+**를 사용한다. 이 모델을 seen classes에 대해서만 fully supervised하게 학습한다. 이후 마지막 분류층을 제거하고, 그 직전의 feature를 pixel-wise visual representation으로 사용한다. 논문은 마지막 `1 × 1` convolution classification layer와, 그 입력 feature를 zero-shot 학습의 대상 공간으로 선택한다. 여기서 각 spatial location은 downsampled image 상의 하나의 픽셀로 간주된다.

둘째, seen classes의 feature distribution을 모사하는 **generative model**을 학습한다. 전체 클래스 집합을 $C = S \cup U$로 두고, $S$는 seen, $U$는 unseen이다. 각 클래스 $c$는 word2vec 임베딩 $a[c] \in \mathbb{R}^{d_a}$로 표현된다. 이 논문에서는 Wikipedia corpus로 학습된 300차원 word2vec을 사용한다. generator는 semantic embedding $a$와 Gaussian noise $z$를 입력받아 synthetic feature $\hat{x}$를 출력한다.

본문의 표기는 다음과 같다.

$$
\hat{x} = G(a, z; w) \in \mathbb{R}^{d_x}
$$

여기서 $G$는 generator, $w$는 그 파라미터다. 중요한 점은 이 generator가 image를 생성하는 것이 아니라, **DeepLab feature space의 pixel-level visual feature**를 생성한다는 것이다. 즉, segmentation backbone의 중간 표현 공간에서 unseen class 샘플을 합성한다.

이 generator 학습에는 **GMMN (Generative Moment Matching Network)**을 사용한다. GMMN은 생성된 feature 분포와 실제 seen feature 분포가 비슷해지도록 **Maximum Mean Discrepancy (MMD)**를 최소화한다. 논문에 제시된 손실은 다음과 같다.

$$
L_{\text{GMMN}}(a) =
\sum_{x, x' \in X(a)} k(x, x')
+ \sum_{\hat{x}, \hat{x}' \in \hat{X}(a; w)} k(\hat{x}, \hat{x}')
- 2 \sum_{x \in X(a)} \sum_{\hat{x} \in \hat{X}(a; w)} k(x, \hat{x})
$$

여기서 $X(a)$는 특정 semantic description $a$를 갖는 실제 seen feature 집합이고, $\hat{X}(a; w)$는 generator가 만든 synthetic feature 집합이다. 커널 $k$는 Gaussian kernel이다.

$$
k(x, x') = \exp \left( - \frac{1}{2\sigma^2} \|x - x'\|^2 \right)
$$

이 손실의 의미는 간단하다. 같은 클래스 semantic embedding에 대해 생성된 feature들이 실제 feature 분포와 최대한 비슷해지도록 만드는 것이다. 이렇게 학습된 generator는 unseen class embedding만으로도 그럴듯한 visual feature를 생성할 수 있다.

셋째, unseen용 synthetic feature를 활용해 segmentation classifier를 다시 학습한다. generator 학습 후, unseen classes에 대해 임의로 많은 synthetic feature를 샘플링하여 $\hat{D}_u$를 만든다. 이 synthetic unseen 데이터와 seen classes의 실제 feature $D_s$를 합쳐, 원래 DeepLab의 마지막 `1 × 1` classifier layer를 fine-tuning한다. 이로써 classifier는 seen과 unseen 클래스 전체 $C$를 출력할 수 있게 된다. 테스트 시에는 입력 이미지를 backbone에 통과시켜 실제 feature를 얻고, 이 classifier가 픽셀별 label을 예측한다.

논문은 여기에 두 가지 확장을 더한다.

첫 번째는 **self-training**, 즉 **ZS5Net**이다. 이것은 완전한 zero-shot보다 조금 완화된 설정이다. unseen classes가 포함된 이미지가 학습 시점에 label 없이 주어진다고 가정한다. 이미 학습된 ZS3Net이 이 unlabeled 이미지들에 pseudo-label을 부여하고, 그중 confidence가 높은 상위 $p\%$의 unseen pixel만 골라 추가 학습 데이터로 사용한다. 이후 segmentation network를 다시 학습한다. 핵심은 unseen class의 실제 appearance를 일부나마 pseudo-supervision으로 흡수해 성능을 높인다는 점이다.

두 번째는 **graph-context encoding**이다. 이는 특히 Pascal-Context처럼 복잡한 장면을 위한 장치다. 논문은 semantic connected component를 node로 하고, 경계를 공유하면 edge를 연결하는 adjacency graph $G=(V,E)$를 만든다. 각 node $\nu \in V$는 해당 클래스의 word2vec embedding $a_\nu$와 Gaussian noise $z_\nu$를 concatenate한 입력으로 표현된다. 그리고 generator 내부의 선형층을 **graph convolutional layers**로 바꾸어, 주변 객체와의 관계를 반영한 feature generation을 수행한다. 직관적으로는, 객체는 자기 semantic label뿐 아니라 주변 객체 배치와도 강하게 연결되므로, 이런 구조적 context를 이용하면 더 좋은 unseen feature를 생성할 수 있다는 주장이다.

## 4. 실험 및 결과

실험은 **Pascal-VOC 2012**와 **Pascal-Context**에서 수행되었다. Pascal-VOC는 20개 object class를 갖고 1,464개 training image를 사용한다. Pascal-Context는 59개 object/stuff class를 가지며, 4,998개 training image와 5,105개 validation image를 포함한다. Pascal-Context가 훨씬 복잡하고 dense annotation을 갖기 때문에 더 어려운 벤치마크다.

zero-shot setup은 unseen class 수 $K$를 2, 4, 6, 8, 10으로 늘려가며 구성했다. 예를 들어 Pascal-VOC의 unseen split은 2-class에서 `cow, motorbike`, 4-class에서 여기에 `airplane, sofa`를 추가하는 식의 incremental split이다. 평가는 standard segmentation metric인 **PA (Pixel Accuracy)**, **MA (Mean Accuracy)**, **mIoU (mean Intersection-over-Union)**를 사용했고, seen/unseen 불균형을 반영하기 위해 seen mIoU와 unseen mIoU의 조화평균인 **hIoU**도 보고했다.

비교 baseline은 zero-shot classification의 대표 방법인 **DeViSe**를 segmentation용으로 바꾼 것이다. DeepLabv3+의 마지막 layer를 class probability가 아니라 300차원 word2vec embedding을 회귀하도록 바꾸고, cosine similarity로 학습한다. 추론 시에는 각 픽셀의 예측 embedding과 가장 cosine similarity가 높은 클래스 embedding을 label로 택한다. 즉, segmentation backbone 위에 직접 visual-semantic embedding 회귀를 얹은 방식이다.

Pascal-VOC 결과를 보면, baseline은 seen classes에서는 어느 정도 성능이 나오지만 unseen classes에서는 매우 약하다. 예를 들어 **2 unseen split**에서 baseline의 unseen mIoU는 **3.2**에 불과한 반면, **ZS3Net은 35.4**를 기록한다. seen mIoU도 baseline 68.1 대비 ZS3Net 72.0으로 오히려 더 좋다. hIoU는 baseline **6.1**에서 ZS3Net **47.5**로 큰 폭으로 상승한다. **10 unseen split**에서도 baseline unseen mIoU가 **1.9**인 데 비해 ZS3Net은 **18.1**, hIoU는 **3.6**에서 **23.6**으로 개선된다. 즉, unseen 수가 증가해도 성능 저하는 있지만, 생성 기반 접근이 embedding regression baseline보다 훨씬 안정적이다.

논문은 또 **generalized evaluation**과 **vanilla evaluation**의 차이도 강조한다. Pascal-VOC 10-unseen split에서 baseline의 unseen mIoU는 generalized eval에서는 **1.9**지만, vanilla eval에서는 **41.7**이다. ZS3Net도 generalized **18.1**, vanilla **46.2**이다. 이는 vanilla eval이 seen class score를 무시하기 때문에 실제 zero-shot segmentation의 어려움, 즉 seen bias를 제대로 반영하지 못한다는 뜻이다. 이 논문이 generalized setting을 중심으로 평가한 것은 타당하다.

Pascal-Context에서도 경향은 비슷하지만 절대 성능은 더 낮다. 예를 들어 **2 unseen split**에서 baseline unseen mIoU는 **2.7**, ZS3Net은 **21.6**이다. 그리고 여기에 graph-context를 넣은 **ZS3Net + GC**는 unseen mIoU를 **30.0**까지 높인다. hIoU도 baseline **5.0**, ZS3Net **28.4**, ZS3Net + GC **34.8**로 좋아진다. 4 unseen split에서도 unseen mIoU가 ZS3Net **24.9**, ZS3Net + GC **29.1**이다. 복잡한 장면일수록 객체 간 관계를 반영한 graph context가 도움을 준다는 논문의 주장을 수치가 뒷받침한다.

self-training 결과는 더 인상적이다. Table 4에 따르면 **ZS5Net**은 ZS3Net보다 seen과 unseen 모두에서 추가 향상을 보인다. Pascal-VOC **2 unseen split**에서 unseen mIoU는 **75.8**, overall mIoU도 **75.8**로, fully supervised oracle 성능인 overall mIoU **76.9**에 매우 가까워진다. Pascal-Context **2 unseen split**에서도 unseen mIoU가 **55.5**까지 오른다. 논문은 high-confidence unseen pseudo-label 비율 $p$를 cross-validation으로 정했으며, Pascal-VOC는 **25%**, Pascal-Context는 **75%**를 사용했다. 이는 데이터셋의 난도와 pseudo-label 품질 차이를 반영하는 선택으로 보인다.

정성적 결과에서도 학습 중 본 적 없는 `motorbike`, `cat`, `plane`, `cow`, `boat` 같은 객체가, zero-shot이 없는 기본 segmentation 모델에서는 `background`나 유사한 seen class로 잘못 분류되던 것이 ZS3Net과 ZS5Net에서는 올바르게 분리되는 모습이 제시된다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **zero-shot semantic segmentation이라는 문제를 명확하게 정의하고, 실제로 작동하는 최초 수준의 실용적 baseline을 제시했다는 점**이다. 단순히 classification 아이디어를 segmentation에 억지로 옮긴 것이 아니라, segmentation backbone의 pixel-level feature space에서 unseen sample을 생성하고 classifier를 재학습하는 구조로 문제를 잘 재설계했다. generalized zero-shot evaluation을 사용해 seen bias를 정면으로 다룬 점도 강점이다.

또 다른 강점은 방법이 비교적 모듈식이라는 것이다. backbone으로 DeepLabv3+를 쓰지만, 핵심은 특정 segmentation model에 묶여 있지 않다. generator도 GMMN을 사용했지만, 논문은 GAN 기반 변형도 비슷한 성능을 냈다고 보고한다. 즉, 이 프레임워크는 “semantic embedding으로 visual feature를 생성해 classifier를 보강한다”는 상위 아이디어에 기반한다.

실험 설계도 설득력이 있다. Pascal-VOC와 Pascal-Context 두 벤치마크를 사용했고, unseen class 수를 단계적으로 늘려가며 평가했다. seen, unseen, overall을 분리해 보고 hIoU까지 제시함으로써 성능 편향을 비교적 정직하게 드러냈다. 특히 baseline이 vanilla setting에서만 좋아 보이고 generalized setting에서는 급격히 무너진다는 비교는 논문 메시지를 강하게 뒷받침한다.

한편 한계도 분명하다. 첫째, unseen class feature는 어디까지나 **synthetic feature**다. 실제 이미지 appearance 분포가 semantic embedding만으로 충분히 복원된다고 보기 어렵다. 특히 시각적으로 복잡하거나 semantic label과 appearance의 연결이 약한 클래스에서는 생성 feature의 품질이 제한될 수 있다. 논문도 이 점을 직접 분석하지는 않는다.

둘째, class semantic representation으로 **word2vec**만 사용한다. word2vec은 일반 텍스트 기반 embedding이므로 시각적 속성을 충분히 담지 못할 수 있다. 예를 들어 두 단어가 언어적으로 가깝다고 해서 segmentation에 필요한 shape, texture, part structure가 비슷하다는 보장은 없다. 논문은 이러한 semantic space의 한계를 깊게 논의하지 않는다.

셋째, graph-context encoding은 흥미롭지만, 논문이 사용한 입력은 **training 중 segmentation mask로부터 만든 adjacency graph**다. 이는 “정확한 object arrangement 정보”를 알고 있다는 뜻이므로, 순수한 zero-shot setting의 깔끔한 가정과는 약간 결이 다르다. 물론 논문은 unseen class가 포함된 이미지 자체는 쓰지 않는다고 명시하지만, context prior를 얻는 방식이 실제 응용에서 얼마나 일반적으로 확보 가능한지는 별도 문제다.

넷째, self-training은 성능 향상 폭이 크지만, 이는 이미 논문이 말하듯 **pure zero-shot**이 아니라 **relaxed setting**이다. unseen class가 포함된 unlabeled image가 학습 시점에 उपलब्ध하다는 가정이 추가된다. 따라서 ZS5Net의 강한 성능은 ZS3Net의 순수 zero-shot 능력과는 구분해서 해석해야 한다.

마지막으로, 논문은 qualitative improvement와 전체 mIoU 개선을 잘 보여주지만, 클래스별 실패 패턴, semantic distance와 성능의 상관관계, generator가 어떤 unseen class에서 특히 어려워하는지 같은 세부 분석은 제한적이다. 후속 연구라면 이런 부분을 더 면밀히 파고들 필요가 있다.

## 6. 결론

이 논문은 semantic segmentation을 위한 zero-shot 학습 문제를 정식으로 제안하고, 이를 해결하기 위해 **ZS3Net**이라는 생성 기반 프레임워크를 제시했다. 핵심 기여는 seen class의 실제 pixel feature와 semantic embedding의 관계를 학습한 뒤, unseen class에 대한 synthetic pixel feature를 생성하여 segmentation classifier를 확장한 것이다. 여기에 self-training 기반의 **ZS5Net**, 그리고 복잡한 장면을 위한 **graph-context encoding**까지 제안해 성능을 더 높였다.

실험 결과는 이 접근이 단순 embedding regression baseline보다 훨씬 효과적이며, 특히 generalized zero-shot setting에서 unseen bias 문제를 완화한다는 점을 보여준다. 따라서 이 연구는 대규모 클래스 확장이 필요한 semantic segmentation에서 중요한 출발점으로 볼 수 있다. 향후에는 더 강한 multimodal semantic representation, 더 정교한 feature generator, 실제 open-vocabulary segmentation과의 연결 같은 방향으로 발전할 가능성이 크다.
