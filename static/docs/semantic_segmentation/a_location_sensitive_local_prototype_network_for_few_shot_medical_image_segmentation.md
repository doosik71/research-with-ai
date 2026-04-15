# A Location-Sensitive Local Prototype Network for Few-Shot Medical Image Segmentation

- **저자**: Qinji Yu, Kang Dang, Nima Tajbakhsh, Demetri Terzopoulos, Xiaowei Ding
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2103.10178

## 1. 논문 개요

이 논문은 의료 영상 분할에서 큰 문제인 annotation scarcity, 즉 전문가 수준의 픽셀 단위 라벨이 매우 비싸고 얻기 어렵다는 문제를 다룬다. 기존의 deep segmentation 모델은 충분한 양의 정답 마스크가 있어야 성능이 잘 나오지만, 실제 의료 영상에서는 이런 조건을 만족시키기 어렵다. 이를 해결하기 위해 논문은 few-shot segmentation 설정을 사용한다. 즉, 새로운 장기 클래스에 대해 소수의 support image와 support mask만 주어졌을 때 query image를 분할할 수 있는 모델을 학습한다.

저자들이 특히 주목한 것은 medical imaging에서 자주 성립하는 spatial prior이다. 예를 들어 복부 CT에서는 liver, spleen, kidney 같은 장기들이 완전히 임의의 위치에 등장하지 않고 비교적 일정한 위치적 패턴을 가진다. 논문은 이런 spatial layout prior를 few-shot segmentation에 명시적으로 넣으면 성능이 크게 좋아질 수 있다고 본다. 기존 few-shot segmentation 연구들은 주로 appearance similarity나 global prototype에 집중했지만, 이 논문은 위치 정보를 적극적으로 활용하는 것이 의료 영상에서는 특히 중요하다고 주장한다.

핵심 목표는 support-query 사이의 class-agnostic knowledge transfer를 유지하면서도, 전체 이미지를 하나의 global prototype으로 설명하는 대신 위치별 local prototype으로 나누어 더 쉬운 subproblem들로 푸는 것이다. 저자들은 이를 통해 VISCERAL contrast-enhanced CT organ segmentation에서 기존 state of the art보다 mean Dice를 약 10% 높였다고 보고한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. 전체 이미지를 대표하는 global prototype 하나로 foreground와 background를 설명하려고 하면, 특히 background처럼 공간적으로 매우 이질적인 영역은 평균화 과정에서 정보가 심하게 손실된다. 따라서 support image를 여러 개의 위치별 grid로 나누고, 각 위치에서만 계산한 local prototype을 사용하면 더 정확한 matching이 가능하다는 것이다.

저자들은 이를 `location-sensitive local prototype network`라고 부른다. 여기서 location-sensitive라는 말은 prototype이 이미지 전체가 아니라 특정 위치의 grid 안에서만 계산되기 때문에, 그 자체가 위치 정보를 내장하고 있다는 뜻이다. query image의 어떤 위치 $(x, y)$를 분류할 때도 전체 prototype 집합과 무차별적으로 비교하는 것이 아니라, 그 위치를 포함하는 대응 grid들의 prototype과만 비교한다. 이로써 "이 위치는 원래 어떤 장기가 나올 법한가"라는 spatial prior가 자연스럽게 들어간다.

기존 접근과의 차별점은 두 가지다. 첫째, PANet이나 SENet 같은 비교 대상은 spatial layout prior를 명시적으로 사용하지 않는다. 둘째, 이 논문은 의료 영상의 구조적 규칙성을 활용하기 때문에 natural image few-shot segmentation보다 의료 영상에 더 잘 맞는 inductive bias를 설계했다. 논문은 이 차이가 성능 향상의 핵심 원인이라고 실험과 ablation으로 뒷받침한다.

## 3. 상세 방법 설명

전체 파이프라인은 두 단계로 구성된다. 첫 번째는 support set으로부터 `location-sensitive local prototype`을 추출하는 단계이고, 두 번째는 이 prototype들을 이용해 query image를 grid-based few-shot segmentation 방식으로 분할하는 단계이다.

문제 설정은 standard episodic few-shot segmentation이다. 학습 데이터셋 $D_{train}$과 테스트 데이터셋 $D_{test}$는 class-disjoint하며, 즉 $C_{train} \cap C_{test} = \emptyset$이다. 각 episode는 support set $S$와 query set $Q$로 구성된다. support set에는 소수의 labeled support images가 들어 있고, query set에는 같은 episode의 클래스에 속한 query images가 들어 있다. 학습 시 매 episode마다 클래스 조합이 바뀌므로, 모델은 특정 클래스 이름이 아니라 support-query 간의 class-agnostic transfer를 배우도록 설계된다.

먼저 이미지를 $w \times h$ 크기라고 하면, 저자들은 이를 여러 개의 overlapping grid $G = \{g_m\}$로 나눈다. 각 grid의 크기는 $\alpha w \times \alpha h$이고, $\alpha$는 grid scale이다. support image $I_i^s$를 feature extractor에 통과시켜 feature map $F_i^s$를 얻은 뒤, 특정 클래스 $c$와 특정 grid $g_m$에 대한 local prototype은 masked average pooling으로 계산된다:

$$
p_{c,g_m}^s =
\frac{1}{k}
\sum_{i=1}^{k}
\frac{\sum_{(x,y)\in g_m} F_i^s(x,y) M_{i,c}^s(x,y)}
{\sum_{(x,y)\in g_m} M_{i,c}^s(x,y)}
$$

여기서 $M_{i,c}^s(x,y)$는 support mask이고, 해당 grid 안에 클래스 $c$의 픽셀이 하나도 없으면 prototype을 0으로 둔다. background도 별도의 semantic class로 취급하며, foreground가 아닌 위치들만 모아 background prototype을 계산한다.

이 식의 의미는 어렵지 않다. global prototype은 클래스에 속하는 모든 위치의 feature를 한 번에 평균내지만, 이 논문은 특정 위치 범위 $g_m$ 안의 feature만 평균낸다. 따라서 같은 liver라도 좌상단 부근 liver feature와 중앙부 liver feature를 분리해서 표현할 수 있고, background처럼 여러 해부학적 구조가 섞인 클래스도 덜 망가진 표현을 만들 수 있다.

query segmentation 단계에서는 query feature map $F_j^q$의 각 위치 $(x, y)$에 대해 support prototype들과 유사도를 계산한다. 클래스 $c$의 확률은 다음과 같이 얻는다:

$$
P_{j,c}^q(x,y) = \sigma\big(\text{sim}(F_j^q(x,y), P)\big)
$$

여기서 $\sigma$는 softmax이고, $P$는 모든 위치와 클래스의 local prototype 집합이다. 겹치지 않는 grid만 있다면 $(x, y)$가 속한 grid의 prototype과 cosine similarity를 계산하면 된다. 하지만 이 논문은 overlapping grid를 사용하므로, 하나의 위치가 여러 grid에 동시에 포함될 수 있다. 이 경우 그 위치에 영향을 주는 grid 집합을 $\Omega$라고 두고, 유사도는 다음처럼 계산한다:

$$
\text{sim}(F_j^q(x,y), P) =
\max_{g_m \in \Omega}
\big(\cos(F_j^q(x,y), p_{c,g_m}^s)\big)
$$

즉, query의 한 위치에 대해 여러 후보 local prototype 중 가장 잘 맞는 prototype의 similarity를 사용한다. 이 설계는 grid 경계에서 생길 수 있는 부자연스러운 단절을 줄여주고, 위치 정보는 살리면서도 약간의 spatial misalignment에는 견딜 수 있게 해준다.

학습은 표준 cross-entropy loss로 end-to-end 수행된다:

$$
L_{ce}(P^q, M^q)
$$

최종적으로 query의 픽셀 라벨은 각 위치에서 가장 높은 확률을 갖는 클래스로 정한다:

$$
\hat{M}_j^q(x,y) = \arg\max_c P_{j,c}^q(x,y)
$$

정리하면, 이 방법의 본질은 "support의 특정 위치에서 뽑은 local semantic template을 query의 대응 위치에 매칭한다"는 것이다. 저자들은 이 아이디어가 의료 영상의 위치적 규칙성과 잘 맞는다고 본다.

## 4. 실험 및 결과

실험은 VISCERAL contrast-enhanced CT dataset에서 수행되었다. 학습에는 65개의 silver corpus scan, 테스트에는 20개의 gold corpus scan을 사용했다. 실험 설정은 1-way 1-shot이며, 대상 장기는 `liver`, `spleen`, `left kidney`, `right kidney`, `left psoas`, `right psoas`이다. 논문은 Roy et al.의 SENet 설정과 가능한 한 동일한 protocol을 사용해 공정 비교를 시도했다.

백본은 ImageNet으로 pretrain된 customized ResNet-50이고, CT slice는 $512 \times 512$로 resize하여 입력한다. 기본 설정에서 grid scale은 $\alpha = 1/8$이고, grid center 간 간격은 grid 크기의 절반으로 두어 overlapping grid를 만든다. data augmentation으로는 RandomGamma, RandomContrast, RandomBrightness를 사용했다. 학습은 SGD, batch size 4, 총 10,000 iteration, 초기 learning rate $10^{-3}$, 2,500 iteration마다 0.1배 decay로 진행했다.

주요 정량 결과는 매우 강하다. Table 1에 따르면 mean Dice는 다음과 같다.

- PANet: 30.7
- SENet: 56.7
- 제안 방법: 66.7
- 추가 장기 클래스까지 학습에 사용한 경우: 70.3

장기별로 보면 제안 방법은 liver 77.9, spleen 71.5, kidney 67.5, psoas 49.9를 기록했다. SENet 대비 평균적으로 약 10%p 향상이며, 논문은 이를 새로운 state of the art로 제시한다. 특히 PANet과 SENet은 spatial layout prior를 명시적으로 쓰지 않는데, 저자들은 이것이 성능 차이의 핵심이라고 해석한다.

추가적인 14개 organ annotation을 학습에 활용했을 때 mean Dice가 70.3까지 올라간 것도 중요하다. 저자들은 few-shot segmentation의 목표가 unseen class에 일반화되는 class-agnostic model을 만드는 것이므로, 더 다양한 training classes는 overfitting을 줄이고 generalization을 높인다고 설명한다. 이 해석은 few-shot learning의 일반적인 원리와도 일치한다.

정성적 결과에서도 support와 query 사이에 장기 크기와 모양 차이가 있음에도 segmentation이 잘 이루어졌다고 보고한다. Fig. 2와 Fig. 4는 query image 위에 segmentation 결과를 overlay한 예시를 보여주는데, 본문 설명에 따르면 PANet과 SENet보다 더 안정적으로 organ region을 복원한다.

ablation study는 이 논문의 설계 선택이 실제로 중요한지 보여준다. 먼저 grid scale을 바꾸면 성능이 크게 달라진다. 최적값인 $\alpha = 1/8$에서 mean Dice가 70.3이었고, 너무 작은 $1/16$에서는 63.8, 더 큰 $1/4$에서는 67.1, 아예 전체 이미지를 한 grid로 보는 $\alpha = 1$에서는 30.3까지 떨어졌다. 이는 위치 정보를 전혀 쓰지 않으면 성능이 심각하게 저하된다는 뜻이다.

또한 overlapping grid를 non-overlapping grid로 바꾸면 mean Dice가 70.3에서 60.2로 감소했다. 즉, 위치 정보를 쓰는 것만으로 충분한 것이 아니라, grid 간 중첩을 통해 더 부드럽고 유연한 matching을 허용하는 것도 중요하다는 점이 실험으로 확인된다.

마지막으로 spatial misalignment에 대한 robustness도 분석했다. support mask와 query mask의 ground-truth overlap을 `alignment Dice`로 측정하고, 실제 segmentation 품질을 `segmentation Dice`로 측정했는데, liver, spleen, left/right kidney에서는 alignment Dice가 40% 미만인 경우에도 segmentation Dice가 60% 이상인 사례가 많았다. 이는 support-query가 완전히 잘 정렬되어 있지 않아도 어느 정도는 견딜 수 있다는 뜻이다. 반면 left/right psoas는 구조가 작고 주변 조직과 시각적으로 비슷해 성능이 상대적으로 낮았다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 의료 영상의 domain prior를 few-shot segmentation 구조에 매우 직접적으로 녹여냈다는 점이다. 단순히 성능이 좋다는 수준이 아니라, 왜 spatial prior가 중요한지, 왜 global prototype이 background에서 약한지, 왜 local prototype이 더 적합한지에 대한 논리가 비교적 설득력 있게 제시된다. 또한 ablation study가 grid scale, overlap, misalignment tolerance를 폭넓게 다루고 있어서, 제안 기법이 단지 우연히 잘 된 것이 아니라는 근거를 제공한다.

두 번째 강점은 방법이 비교적 단순하고 해석 가능하다는 점이다. local prototype 추출, cosine similarity, max aggregation, cross-entropy loss라는 구성은 복잡한 auxiliary module 없이도 직관적으로 이해할 수 있다. few-shot medical segmentation에서 실제로 쓸 수 있는 inductive bias를 넣었다는 점에서 실용적 가치도 크다.

하지만 한계도 분명하다. 가장 중요한 가정은 support와 query가 `similar spatial layout`을 가진다는 것이다. 논문도 이를 명시적으로 인정한다. 즉, 이 방법은 spatial prior가 강한 modality나 anatomy에서는 잘 맞지만, 촬영 자세 변화가 크거나 구조 위치 변동성이 큰 상황에서는 효과가 제한될 수 있다. misalignment robustness 실험이 있긴 하지만, 어디까지나 일정 수준의 tolerance를 보였다는 것이지 spatial assumption이 불필요하다는 뜻은 아니다.

또 다른 한계는 실험 범위가 비교적 제한적이라는 점이다. 주된 실험은 VISCERAL CT organ segmentation 중심이며, 다른 modality나 다른 anatomical region으로의 일반화는 본문에서 직접 검증하지 않았다. 따라서 이 방법이 MRI, ultrasound, pathology image 같은 더 다양한 settings에서도 동일하게 효과적인지는 논문만으로는 판단할 수 없다.

또한 논문은 2D slice 기반 실험을 수행했다. 입력 데이터는 본질적으로 3D volumetric CT이지만, 평가 프로토콜은 2D segmentation 성능으로 진행되었다. 3D context를 직접 활용하는 few-shot segmentation과 비교했을 때 어떤 장단점이 있는지는 여기서 명확히 다루지 않는다.

비판적으로 보면, spatial prior를 활용한 성능 향상은 매우 타당하지만, 그만큼 class-agnostic appearance matching 자체의 어려움을 위치 정보가 일부 우회한 측면도 있다. 이는 의료 영상에서는 오히려 장점일 수 있지만, 방법의 일반성을 낮추는 trade-off로도 볼 수 있다. 논문은 이 trade-off를 어느 정도 받아들이고, 대신 의료 영상이라는 구체적 domain에 맞춘 설계를 선택한 것으로 해석된다.

## 6. 결론

이 논문은 few-shot medical image segmentation에서 spatial layout prior를 활용하는 `location-sensitive local prototype network`를 제안했다. 핵심은 support image에서 위치별 local prototype을 추출하고, query의 각 위치를 대응되는 local prototype과 비교해 분할하는 것이다. 이 설계는 global prototype이 놓치기 쉬운 local structure와 background heterogeneity 문제를 줄여주며, 실제로 VISCERAL CT organ segmentation에서 기존 방법 대비 약 10%p 높은 mean Dice를 달성했다.

이 연구의 의미는 단순한 성능 개선 이상이다. few-shot segmentation에서 어떤 prior를 넣어야 하는지는 매우 중요한 문제인데, 이 논문은 의료 영상에서는 appearance만이 아니라 anatomy의 spatial regularity가 핵심 단서가 될 수 있음을 명확히 보여준다. 향후에는 이를 3D setting으로 확장하거나, registration-free alignment mechanism과 결합하거나, modality별 spatial uncertainty를 더 정교하게 모델링하는 방향으로 발전시킬 수 있다. 의료 영상처럼 구조적 규칙성이 강한 분야에서는 이런 접근이 실제 적용 가능성이 높다.
