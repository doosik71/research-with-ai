# A Comprehensive Survey of Few-shot Learning: Evolution, Applications, Challenges, and Opportunities

- **저자**: Yisheng Song, Ting Wang, Subrota K Mondal, Jyoti Prakash Sahoo
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2205.06743

## 1. 논문 개요

이 논문은 Few-shot Learning(FSL) 분야의 최근 약 3년간 연구 200편 이상을 폭넓게 검토한 survey 논문이다. 저자들의 핵심 목표는, 단순히 대표 모델 몇 개를 나열하는 것이 아니라, FSL이 실제로 어떤 문제를 풀기 위해 등장했는지, 어떤 기술 축으로 발전해 왔는지, 그리고 현재 어디에서 막히고 있는지를 체계적으로 정리하는 데 있다. 특히 저자들은 FSL, transfer learning, meta-learning처럼 서로 혼동되기 쉬운 개념들을 먼저 구분하고, 이후 FSL 연구를 “도전 과제(challenges)” 중심으로 재분류하는 새로운 taxonomy를 제안한다.

논문이 다루는 연구 문제는 명확하다. 실제 환경에서는 전체적으로는 데이터가 많아 보여도, 개별 장치나 개별 장면, 혹은 특정 클래스 단위로 보면 학습 가능한 표본 수가 극히 적은 경우가 많다. 예를 들어 산업 검사에서는 불량 샘플이 매우 희소하고, 의료나 위성영상처럼 도메인 차이가 큰 영역에서는 기존 대규모 자연영상 데이터셋에서 배운 지식이 곧바로 잘 옮겨가지 않는다. 즉, FSL의 본질적 문제는 “매우 적은 샘플만으로도 일반화 가능한 지식을 얼마나 빨리, 얼마나 정확하게 추출할 수 있는가”이다.

이 문제는 중요하다. 논문은 IoT, smart manufacturing, quality inspection, medical imaging, remote sensing 등 실제 응용 환경에서 데이터 부족, 도메인 이동(domain shift), long-tail distribution 문제가 흔하며, 전통적인 대규모 supervised learning이 이런 조건에서 충분히 작동하지 않는다고 설명한다. 따라서 FSL은 단지 학술적 벤치마크 문제가 아니라, 데이터 수집 비용과 실제 배치 환경의 제약을 다루기 위한 실용적 학습 패러다임으로 제시된다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 FSL을 단순히 알고리즘 종류로 분류하지 않고, “적은 샘플 학습이 왜 어려운가”라는 근본 원인에 따라 계층적으로 이해하자는 것이다. 저자들은 지식 통합 수준(degree of integration of knowledge)에 따라 FSL을 크게 single-modal learning과 multimodal learning으로 나누고, single-modal learning 내부를 다시 data augmentation, transfer learning, meta-learning으로 구분한다.

이 taxonomy에서 각 축은 서로 다른 난제를 대응한다. 첫째, data augmentation은 적은 샘플 때문에 실제 데이터 분포를 제대로 평가하지 못하는 문제를 다룬다. 둘째, transfer learning은 feature reuse sensitivity, 즉 source domain에서 배운 특징이 target task에서 얼마나 안정적으로 재사용될 수 있는가의 문제를 다룬다. 셋째, meta-learning은 미래의 unseen task로 얼마나 잘 일반화할 수 있는가, 다시 말해 task-level generality의 문제를 다룬다. 넷째, multimodal learning은 단일 modality만으로는 정보가 너무 부족하다는 문제를 완화하기 위해 semantics, language, audio 같은 보조 modality를 도입한다.

기존 survey들과의 차별점도 여기 있다. 논문에 따르면 기존 survey들은 예를 들어 generative vs discriminative, 혹은 data/model/algorithm 관점으로 FSL을 분류했지만, 이 논문은 “도전 과제 중심”으로 taxonomy를 구성한다. 저자들의 주장에 따르면 이런 관점은 독자가 개별 모델의 이름보다도, 왜 특정 계열의 방법이 등장했는지 그 동기를 더 잘 이해하게 해 준다. 특히 cross-domain FSL을 중요한 하위 주제로 강조하는 점도 특징적이다.

또 하나 중요한 메시지는 “meta-learning이 FSL과 동일한 것은 아니다”라는 점이다. 논문은 FSL을 ultimate goal로, meta-learning을 그 목표를 달성하기 위한 대표적 패러다임 중 하나로 본다. 즉, few-shot setting이라고 해서 반드시 meta-learning을 써야 하는 것은 아니며, 실제로 recent work에서는 fine-tuning baseline이 강력한 성능을 보인다고 정리한다.

## 3. 상세 방법 설명

이 논문은 새로운 단일 모델을 제안하는 논문이 아니라 survey 논문이므로, 하나의 학습 파이프라인이나 하나의 손실 함수를 중심으로 전개되지는 않는다. 대신 FSL 전체를 설명하기 위한 개념적 구조와 각 방법군의 작동 원리를 정리한다.

먼저 논문은 FSL의 기본 문제 형식으로 $N$-way-$K$-shot 설정을 설명한다. support set에는 $N$개 클래스가 있고 각 클래스마다 $K$개의 샘플이 있으며, 전체 support set 크기는 $N \times K$이다. query set은 실제로 예측해야 하는 샘플 집합이다. 본문에 따르면 support set과 query set의 클래스는 겹치지 않는다고 설명된다. 또한 $N$-way-1-shot은 one-shot learning, $N$-way-0-shot은 zero-shot learning으로 연결된다.

### 데이터 증강(Data Augmentation)

논문은 data augmentation을 “실제 데이터 분포를 최대한 잘 평가하기 위한 가장 직접적인 방법”으로 본다. 이 범주는 hand-crafted rules와 learning data processing으로 나뉜다.

hand-crafted rules는 전문가가 정의한 규칙을 이용해 데이터를 변형한다. 예를 들어 random erasing, random cropping, fill 같은 pixel-level 변환이 소개된다. 이런 방법은 입력 이미지를 부분적으로 가리거나 변형해 다양한 관측 조건을 흉내 낸다. 하지만 논문은 단순 pixel 변형만으로는 overfitting을 충분히 막기 어렵다고 지적한다.

data-level augmentation은 입력 이미지 자체를 수정하는 방식이다. 예를 들어 CSEI는 support set에서 판별적인 영역을 지우고 복원하는 방식으로 intact feature를 학습하려고 하고, FTT는 날씨나 조명 같은 transient attribute를 보간해 데이터 다양성을 늘리려 한다. 또 어떤 접근은 web에서 이미지를 수집해 graph 기반으로 노이즈를 제거하며 보조 데이터를 만든다.

feature-level augmentation은 이미지를 고차원 latent feature space로 옮긴 뒤 그 공간에서 변형하는 접근이다. 논문은 이런 방식이 원본 픽셀보다 더 유효한 정보를 압축된 형태로 담을 수 있다고 설명한다. 예를 들어 adversarial covariance augmentation은 covariance 정보를 보존하면서 feature variability를 생성하려 하고, saliency-guided hallucination은 foreground, background, original feature를 재조합해 새로운 feature를 합성한다. Label-set operation도 여러 클래스의 특징을 집합 연산처럼 결합해 훈련에 사용한다.

learning data processing은 증강 정책 자체를 자동으로 학습하는 방향이다. DADA는 여러 augmentation sub-strategy 중 어떤 것을 쓸지 gradient 기반으로 학습하고, AFHN이나 MetaGAN 같은 방법은 GAN을 이용해 task-conditioned synthetic feature 또는 synthetic sample을 생성한다. Delta-encoder는 같은 클래스 내부 차이를 autoencoder로 학습한 뒤 이를 transfer에 활용한다. 요지는 사람이 규칙을 직접 설계하는 대신, 메타 수준에서 어떤 증강이 현재 few-shot task에 유리한지를 학습하게 만드는 것이다.

### 전이학습(Transfer Learning)

논문은 transfer learning을 “data-to-label mapping을 구축하는 방법”으로 설명한다. 가장 전형적인 구조는 pre-training과 fine-tuning이다. pre-training 단계에서는 대규모 auxiliary dataset으로 backbone feature extractor를 학습하고, fine-tuning 단계에서는 대부분의 하위 계층을 고정한 채 분류기 층 또는 일부 계층만 업데이트한다.

이 접근의 핵심 직관은 간단하다. support set이 너무 작아 처음부터 representation을 학습하기 어렵기 때문에, 대규모 데이터에서 먼저 좋은 feature space를 배우고 그것을 few-shot task에 재사용하는 것이다. 논문은 최근 연구들이 fine-tuning이 baseline보다 5-way-1-shot에서 약 2%~7% 정도 정확도를 높일 수 있음을 보여주었다고 요약한다. 또한 cosine similarity를 활성화 함수 대신 활용하거나, adaptive gradient optimizer를 쓰는 등 fine-tuning 절차 자체를 개선한 연구도 언급한다.

하지만 transfer learning의 핵심 한계도 분명히 제시된다. source와 target domain의 차이가 크면 negative transfer가 발생할 수 있다. 이를 극단적으로 드러내는 설정이 cross-domain few-shot learning이다. 논문은 BSCD-FSL 벤치마크를 중요한 전환점으로 보며, ImageNet과의 유사도가 점점 낮아지는 CropDiseases, EuroSAT, ISIC, ChestX 같은 데이터셋을 통해 도메인 차이가 실제 성능에 얼마나 큰 영향을 주는지 보여준다고 설명한다.

cross-domain FSL에서는 domain-irrelevant feature를 추출하거나, 여러 backbone에서 domain-specific representation을 따로 학습한 뒤 적절히 재조합하는 방식이 등장한다. 예를 들어 FRN은 ridge regression 기반 feature map reconstruction을 사용하고, FWT나 LRP-GNN은 feature transformation 또는 explanation-guided training을 활용한다. STARTUP은 labeled source data뿐 아니라 unlabeled target data도 이용해 일반화를 강화한다. 즉, transfer learning 축의 핵심은 “어떤 표현을 옮길 것인가”와 “어떻게 도메인 차이를 줄일 것인가”이다.

### 메타러닝(Meta-learning)

논문은 meta-learning을 “task-to-target model mapping을 학습하는 일반적인 프레임워크”로 설명한다. 전이학습이 데이터에서 라벨로 가는 매핑을 배우는 쪽에 가깝다면, 메타러닝은 여러 task를 반복적으로 관찰하면서 새로운 task에 빠르게 적응할 수 있는 초기화 또는 업데이트 규칙을 배운다.

가장 대표적인 계열은 model parameter learning이다. MAML은 각 task에서 inner-loop adaptation을 수행한 뒤, outer-loop에서 여러 task의 적응 방향을 종합하여 좋은 초기 파라미터를 찾는다. 이를 수식으로 쓰지는 않았지만, 개념적으로는 task별 손실을 줄인 뒤 “빠른 적응이 가능한 초기점”을 찾는 최적화 문제라고 볼 수 있다. Reptile은 더 계산량을 줄인 1차 근사 방식이다. Meta-SGD는 초기값뿐 아니라 학습률과 업데이트 방향까지 함께 학습하려고 하며, iMAML은 implicit gradient를 사용해 gradient computation을 효율화한다. TAML과 iTMAML은 task bias를 줄이거나 연속 상태에서 task identification을 다루려는 시도다.

두 번째 축은 metric learning이다. 논문은 일부 survey들이 metric learning을 meta-learning과 별개로 다루지만, 여기서는 meta-learning 프레임 안에서 설명한다. Siamese network는 두 샘플의 similarity를 계산하는 가장 초기적 형태이고, triplet loss는 anchor-positive-negative 삼중 입력을 이용해 intra-class는 가깝게, inter-class는 멀게 만든다. Prototype network는 각 클래스의 평균 feature를 prototype으로 두고, query와의 거리를 비교해 분류한다. relation network는 similarity 함수 자체를 신경망으로 학습한다. matching network는 support와 unlabeled sample을 embedding space에 올려 attention 기반으로 nearest neighbor-like prediction을 수행한다.

prototype network류를 직관적으로 쓰면, 클래스 $c$의 prototype $p_c$는 support feature들의 평균으로 표현된다. query feature $f(x)$에 대해 가장 가까운 $p_c$를 가진 클래스를 예측하는 식이다. 본문에 명시된 수식은 없지만, 개념적으로는 다음과 같은 구조다.

$$
p_c = \frac{1}{|S_c|}\sum_{(x_i, y_i)\in S_c} f_\theta(x_i)
$$

그리고 query는 거리 함수 $d(\cdot,\cdot)$를 이용해 분류된다.

$$
\hat{y} = \arg\min_c d(f_\theta(x_q), p_c)
$$

논문은 단순 평균 prototype이 noise에 약할 수 있어 attentive prototype, negative margin, prototype rectification 같은 보강 기법들이 제안되었다고 정리한다.

세 번째 축은 graph neural network를 이용한 information transmission이다. 이 계열은 support와 query를 graph의 node로 보고, edge를 통해 관계 정보를 전달하면서 새로운 클래스에 대한 판단을 돕는다. EGNN, Meta-GCN, DPGN, GERN, HGNN 등이 예시로 제시된다. DPGN은 sample 간 관계뿐 아니라 sample distribution 간 관계까지 dual graph로 모델링하려고 한다. HGNN은 GNN의 shallow-layer 한계를 넘기 위해 hierarchical connection과 skip connection을 사용한다.

### 멀티모달 학습(Multimodal Learning)

논문은 multimodal learning을 FSL의 상위 단계로 위치시킨다. 이유는 분명하다. single-modal 정보만으로는 본질적으로 정보량이 부족하므로, language, semantics, audio 같은 다른 modality를 활용해 약한 supervision 또는 보조 prior를 제공할 수 있기 때문이다.

한 방향은 multimodal embedding이다. 시각 정보에 semantic attribute, class label, natural language description, knowledge graph 정보를 결합해 더 좋은 표현을 만든다. Wang et al.은 여러 visual feature와 semantic supervision을 결합하고, Schonfeld et al.은 VAE로 latent visual feature와 semantic feature를 정렬한다. 또 adaptive cross-modal 방식은 서로 다른 modality 간 중요도를 조정하며 few-shot recognition을 돕는다.

다른 방향은 semantic information generation이다. text-to-image generation 또는 text-conditioned feature generation을 이용해 추가적인 시각 샘플을 생성하는 방식이다. GAN 기반 접근은 텍스트 설명으로부터 시각 feature나 이미지를 생성하고, 이를 support data를 보완하는 synthetic sample로 사용한다. StackGAN류의 접근처럼 저해상도 생성 후 고해상도 refinement를 하는 방법도 소개된다. 다만 논문은 텍스트 설명 하나가 이미지의 모든 세부를 충분히 담지 못할 수 있다는 점도 지적한다.

## 4. 실험 및 결과

이 논문은 survey이므로 하나의 실험 설정으로 새 방법을 평가하지 않는다. 대신 다양한 대표 논문들의 실험 환경, 벤치마크, 성능 추이를 정리한다. 따라서 여기서는 저자들이 정리한 실험적 흐름과 핵심 관찰을 중심으로 이해하는 것이 적절하다.

먼저 데이터셋 측면에서, 논문은 2017년부터 2021년 사이 FSL 연구에서 가장 많이 사용된 벤치마크로 CUB-200-2011, Mini-ImageNet, Omniglot을 제시한다. 논문에 따르면 통계적으로 CUB-200-2011이 약 46.6%, Mini-ImageNet이 30.5%, Omniglot이 17.4%를 차지해 사실상 표준 벤치마크 역할을 해 왔다. semantic segmentation에는 PASCAL-5i, cross-domain 평가에는 Meta-Dataset과 BSCD-FSL이 언급된다.

이미지 분류에서는 5-way-1-shot과 5-way-5-shot이 핵심 평가 설정으로 사용된다. 표 9에 따르면 2020~2021년 무렵 최고 성능군은 PT+MAP, Transductive CNAPS + FETI, ESFR, AmdimNet, LaplacianShot 등이며, 5-way-1-shot 정확도는 대략 75%~83% 수준, 5-way-5-shot은 84%~91% 수준까지 보고된다. 이 표가 보여주는 메시지는 분명하다. 최근 상위권 방법들은 단순한 end-to-end meta-learning 하나만으로 설명되지 않고, feature distribution modeling, transductive inference, unlabeled data 활용, feature reconstruction 같은 요소를 결합하고 있다.

few-shot object detection에서는 10-shot, 30-shot AP가 주요 지표로 제시된다. Meta-DETR, FSCE, SSR-FSD, FsDetView, TFA w/ cos 등이 소개되며, Meta-DETR은 17.8/22.9 AP로 표에 나타난 방법들 중 높은 수치를 보인다. 논문은 object detection 분야에서는 attention과 semantic alignment가 중요한 역할을 하며, 이미지 분류보다 발전 속도가 느리다고 본다.

few-shot semantic segmentation에서는 mean IoU가 주요 지표다. HSNet, CyCTR, PFENet, PANet, CANet 등이 정리되며, HSNet이 1-shot 66.2, 5-shot 70.4 mean IoU로 높은 성능을 보인다고 소개된다. 이 계열에서는 support-query 간 multi-level correlation이나 prototype refinement가 핵심 설계 포인트로 나타난다.

few-shot instance segmentation에서는 1-shot mAP50 기준으로 ONCE iMTFA, FAPIS, FGN, Siamese Mask R-CNN 등이 비교된다. 논문에 따르면 2021년의 ONCE iMTFA가 20.13 mAP50으로 표에 나타난 방법들 중 가장 높은 수준이다. 전체적으로 instance segmentation은 다른 vision task에 비해 연구 수가 더 적고, 아직 초기 단계라는 뉘앙스로 서술된다.

cross-domain few-shot learning에 대한 논문의 가장 중요한 실험적 메시지는, standard benchmark에서 잘 작동하는 많은 FSL 방법이 도메인 차이가 커지면 급격히 약해진다는 점이다. 특히 BSCD-FSL을 통해 natural image와 농작물, 위성, 피부병변, 흉부 X-ray 사이의 domain gap이 성능을 크게 좌우한다는 것을 강조한다. 논문은 이 결과를 통해 “표준 few-shot 벤치마크에서의 높은 정확도”가 곧바로 실제 배치 환경의 강건성을 의미하지는 않는다고 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 FSL을 단순 알고리즘 목록이 아니라 문제 구조 중심으로 재해석했다는 점이다. 저자들은 데이터 부족, feature transfer의 민감성, unseen task 일반화, single-modal 정보 부족이라는 네 가지 핵심 난제를 축으로 taxonomy를 세웠고, 이를 통해 data augmentation, transfer learning, meta-learning, multimodal learning을 하나의 연속선 위에 배치한다. 이는 독자가 각 방법의 목적과 한계를 비교적 선명하게 이해하도록 돕는다.

또 다른 강점은 개념 정리가 비교적 충실하다는 점이다. FSL, machine learning, transfer learning, meta-learning, one-shot learning, zero-shot learning, cross-domain FSL의 차이를 초반 섹션에서 반복적으로 정리해 주기 때문에, 분야 입문자나 인접 분야 연구자에게 유용하다. 더불어 computer vision의 세부 응용 분야인 image classification, object detection, semantic segmentation, instance segmentation까지 넓게 포괄해 최근 흐름을 한눈에 보이게 만든다.

실용적 관점에서도 강점이 있다. 논문은 smart manufacturing, industrial inspection, component identification 같은 예시를 통해 FSL이 왜 현실에서 필요한지 설명한다. 이것은 survey 논문이 단순 문헌 정리에 머물지 않고, 실제 적용 배경과 함께 문제를 위치시키는 장점으로 볼 수 있다.

반면 한계도 분명하다. 첫째, 이 논문은 survey이기 때문에 새로운 이론, 새로운 학습 목표, 새로운 실험을 제시하는 논문이 아니다. 따라서 어떤 축이 정말 더 근본적인 해결책인지, 또는 제안한 taxonomy가 다른 taxonomy보다 경험적으로 더 유용한지에 대한 엄밀한 검증은 없다. taxonomy의 유용성은 주로 저자들의 설명력에 기대고 있다.

둘째, 여러 표와 비교가 제시되지만, survey 특성상 각 실험의 설정 차이가 완전히 통제되지는 않는다. backbone, external data 사용 여부, transductive setting 여부, unlabeled data 활용 여부가 제각각일 수 있는데, 표에서는 이런 차이가 함께 섞여 있다. 따라서 단순 숫자 비교는 주의가 필요하다. 논문도 어느 정도는 이를 암시하지만, 더 명시적으로 통제 조건을 구분했으면 좋았을 것이다.

셋째, multimodal learning 부분은 흥미롭지만 상대적으로 성숙도가 낮다. 논문도 이 점을 인정하며, representation, alignment, fusion, co-learning, translation 등으로 분류하지만 아직 정리 수준이 더 높고 통합적인 이론이나 표준 벤치마크는 부족해 보인다.

넷째, 일부 미래 방향 제시는 타당하지만 다소 선언적이다. 예를 들어 더 좋은 benchmark, 더 강한 cross-domain robustness, 더 일반적인 meta-learning, 더 풍부한 multimodal pretraining이 필요하다는 주장은 설득력 있지만, 이를 위한 구체적 기술 로드맵까지 제시되지는 않는다. 그러나 이는 survey 논문이라는 장르를 감안하면 과도한 약점이라고 보기는 어렵다.

## 6. 결론

이 논문은 few-shot learning 분야를 최근 연구 흐름에 맞추어 재정리한 포괄적 survey이다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, FSL과 주변 개념들을 구분해 개념적 혼동을 줄였다. 둘째, FSL을 도전 과제 중심으로 재분류하는 taxonomy를 제안해 data augmentation, transfer learning, meta-learning, multimodal learning을 하나의 구조 안에서 설명했다. 셋째, computer vision 응용과 cross-domain 문제까지 포함해 최근 연구 동향과 한계를 함께 정리했다.

논문이 주는 가장 중요한 메시지는, FSL의 성능 향상은 단순히 더 복잡한 모델을 만드는 문제만이 아니라는 점이다. 적은 샘플로 실제 분포를 얼마나 잘 추정할 것인지, source 지식을 어떻게 안전하게 재사용할 것인지, 여러 task에서 어떤 meta-knowledge를 배울 것인지, 그리고 단일 modality의 정보 부족을 어떤 방식으로 보완할 것인지가 모두 함께 중요하다.

실제 적용과 향후 연구 측면에서 보면, 이 논문은 특히 cross-domain FSL과 multimodal FSL을 중요한 방향으로 제시한다. 산업 검사, 의료영상, 원격탐사처럼 도메인 차이가 큰 실제 환경에서는 standard benchmark에서의 높은 성능만으로는 충분하지 않다. 따라서 더 현실적인 benchmark, 더 강건한 transfer와 adaptation, 그리고 더 일반적인 multimodal pretraining이 앞으로 FSL 연구의 핵심 축이 될 가능성이 높다는 것이 이 survey의 결론이라고 볼 수 있다.
