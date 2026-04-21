# SELF-SUPERVISED LEARNING FOR FEW-SHOT IMAGE CLASSIFICATION

- **저자**: Da Chen, Yuefeng Chen, Yuhong Li, Feng Mao, Yuan He, Hui Xue
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/1911.06045

## 1. 논문 개요

이 논문은 few-shot image classification에서 성능을 좌우하는 핵심 요소가 meta-learning 자체만이 아니라, 그 이전 단계에서 준비되는 embedding network의 품질이라는 점에 주목한다. Few-shot setting에서는 각 클래스당 주어진 labelled sample 수가 매우 적기 때문에, supervised learning만으로 큰 backbone을 안정적으로 학습하기 어렵고, 그 결과 표현력 있는 embedding을 얻는 데 한계가 생긴다. 저자들은 이것이 최근 few-shot 연구들이 더 깊은 네트워크를 항상 효과적으로 활용하지 못했던 중요한 이유라고 주장한다.

논문의 목표는 self-supervised learning(SSL)을 이용해 더 크고 일반화가 잘 되는 embedding network를 먼저 학습한 뒤, 이를 episodic meta-learning으로 fine-tuning하여 unseen class에 대한 few-shot 분류 성능을 높이는 것이다. 즉, 문제 설정은 “적은 라벨만 있는 few-shot task에서 어떻게 더 robust하고 transferable한 representation을 만들 것인가”이다.

이 문제가 중요한 이유는 few-shot learning이 본질적으로 데이터가 부족한 상황을 다루기 때문이다. 의료, 위성, 조류 분류처럼 새로운 클래스가 자주 등장하고 라벨링 비용이 큰 분야에서는, representation의 일반화 능력과 domain transfer 능력이 실용 성능을 크게 좌우한다. 이 논문은 SSL이 이러한 병목을 완화할 수 있다는 점을 실험적으로 보이려 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 two-stage paradigm을 유지하되, 첫 번째 단계의 embedding pretraining을 supervised learning이 아니라 self-supervised learning으로 바꾸는 것이다. 저자들은 few-shot learning에서 흔히 쓰이는 ResNet12, Wide-ResNet 수준의 backbone보다 더 큰 embedding network도 SSL로 학습하면 과적합을 덜 일으키면서 downstream few-shot task에 유리한 representation을 만들 수 있다고 본다.

구체적으로는 AMDIM(Augmented Multiscale Deep InfoMax)을 사용해 같은 이미지에서 만든 두 augmentation view 사이의 mutual information을 최대화하도록 학습한다. 이렇게 하면 class label 없이도 global feature와 local feature 사이의 의미 있는 공통 구조를 포착할 수 있고, 그 결과 unseen class에도 잘 전이되는 feature space를 얻게 된다.

기존 접근과의 차별점은 두 가지다. 첫째, pretraining 단계에서 supervised signal에 의존하지 않는다. 논문에서 비교 대상으로 언급된 일부 SSL+few-shot 연구들은 supervised loss와 self-supervised loss를 함께 쓰거나, 비교적 작은 backbone에서만 SSL의 효과를 논했다. 반면 이 논문은 large-scale embedding network를 순수 SSL로 먼저 학습하고, 그 다음 meta-learning을 얹는 구성을 강조한다. 둘째, 저자들은 성능 향상의 원인이 단순히 네트워크 크기 증가가 아니라, “큰 네트워크를 SSL로 학습했기 때문”임을 supervised pretraining 대비 실험으로 뒷받침한다.

## 3. 상세 방법 설명

전체 방법은 두 단계로 구성된다. Stage A에서는 unlabeled image로 embedding network를 self-supervised pretraining한다. Stage B에서는 이렇게 얻은 embedding을 초기값으로 사용해 episodic meta-learning을 수행한다. 최종 few-shot 분류는 prototype 기반 metric learning 방식으로 이뤄진다.

### Self-supervised learning stage

저자들은 AMDIM을 사용한다. 기본 아이디어는 한 이미지 $x$에 대해 data augmentation을 적용해 두 개의 view $x_a, x_b$를 만들고, 이 둘에서 추출한 feature들 사이의 mutual information(MI)을 크게 만드는 것이다. 논문은 mutual information을 다음과 같이 정의한다.

$$
I(X, Y) = D_{KL}(p(x,y)\|p(x)p(y)) = \sum \sum p(x,y)\log \frac{p(x|y)}{p(x)}
$$

여기서 $X, Y$는 두 random variable이고, $p(x,y)$는 joint distribution, $p(x)$와 $p(y)$는 marginal distribution이다. 직관적으로는 두 feature가 얼마나 많은 정보를 공유하는지 측정하는 값이다. 하지만 실제 학습에서는 분포를 직접 알 수 없기 때문에, 논문은 Noise Contrastive Estimation(NCE) 기반 loss를 써서 MI의 lower bound를 최대화한다.

AMDIM의 핵심은 단순히 전체 이미지 표현끼리만 비교하는 것이 아니라, global feature와 local feature를 함께 맞추는 데 있다. 논문에서는 $f_g$를 global feature, $f_5$를 $5 \times 5$ local feature map, $f_7$을 $7 \times 7$ local feature map으로 둔다. 그리고 같은 원본 이미지에서 생성된 두 view 사이에 대해 다음 세 쌍의 mutual information을 높인다.

- $\langle f_g(x_a), f_5(x_b) \rangle$
- $\langle f_g(x_a), f_7(x_b) \rangle$
- $\langle f_5(x_a), f_5(x_b) \rangle$

예를 들어 $f_g(x_a)$와 $f_5(x_b)$ 사이의 NCE loss는 다음과 같다.

$$
L_{ssl}(f_g(x_a), f_5(x_b)) =
- \log
\frac{\exp\{\phi(f_g(x_a), f_5(x_b))\}}
{\sum_{\tilde{x}_b \in N_x \cup x_b}\exp\{\phi(f_g(x_a), f_5(\tilde{x}_b))\}}
$$

여기서 $N_x$는 image $x$에 대한 negative sample 집합이고, $\phi$는 distance metric function이다. 분자에는 positive pair가, 분모에는 positive와 negative가 함께 들어간다. 따라서 같은 이미지에서 나온 두 view의 feature는 가깝게, 다른 이미지의 feature는 멀어지도록 학습된다.

최종 SSL loss는 세 항의 합이다.

$$
L_{ssl}(x_a, x_b) =
L_{ssl}(f_g(x_a), f_5(x_b)) +
L_{ssl}(f_g(x_a), f_7(x_b)) +
L_{ssl}(f_5(x_a), f_5(x_b))
$$

이 단계의 목적은 label 없이도 일반화가 잘 되는 embedding function을 얻는 것이다. 논문의 주장에 따르면, 이렇게 학습된 representation이 few-shot classification의 기반을 강화한다.

### Meta-learning stage

두 번째 단계에서는 SSL로 사전학습된 embedding network를 episodic meta-learning으로 fine-tuning한다. 문제는 $K$-way $C$-shot classification으로 정의된다. 즉, 각 episode마다 $K$개의 클래스를 뽑고, 각 클래스마다 $C$개의 labelled support sample을 제공한 뒤, query sample의 클래스를 맞히는 식으로 학습한다.

데이터셋은 $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$로 표기된다. 특정 meta task $T$에서 선택된 클래스 집합을 $V = \{y_i \mid i=1,\dots,K\}$라고 두고, 이 클래스들로 support set $S$와 query set $Q$를 만든다.

이 논문은 Prototypical Networks 계열의 metric learning 방식을 사용한다. 각 클래스 $k$의 descriptor는 support embedding의 centroid이다.

$$
c_k = \frac{1}{|S_k|}\sum_{(x_i, y_i)\in S} f(x_i)
$$

여기서 $f(x_i)$는 stage one에서 초기화된 embedding function이고, $S_k$는 클래스 $k$에 속한 support sample 집합이다. 즉, 각 클래스는 그 클래스 샘플들의 평균 벡터로 대표된다.

query sample $q$에 대해서는 embedding $f(q)$와 각 class prototype $c_k$ 사이의 Euclidean distance를 계산하고, softmax 형태의 확률분포를 만든다.

$$
p(y=k|q) =
\frac{\exp(-d(f(q), c_k))}
{\sum_{k'} \exp(-d(f(q), c_{k'}))}
$$

여기서 $d$는 Euclidean distance이다. 이 식은 “query가 어떤 prototype에 가장 가까운가”를 확률로 바꾼 것이다. 가까울수록 확률이 높아진다.

논문이 제시한 meta-learning loss는 다음과 같다.

$$
L_{meta} = d(f(q), c_k) + \log \sum_{k'} d(f(q), c_{k'})
$$

원문 표기만 보면 일반적인 negative log-softmax와 약간 다르게 적혀 있다. 따라서 이 식은 논문 원문 표기를 따랐으며, 보다 표준적인 형태로 정리된 유도식은 논문에 명시적으로 설명되어 있지 않다. 다만 의도 자체는 query embedding이 정답 클래스 prototype에 가깝고, 다른 클래스 prototype과는 멀어지도록 만드는 metric learning objective로 이해하면 된다.

### 학습 절차와 설정

SSL 단계에서는 AmdimNet(ndf=192, ndepth=8, nrkhs=1536)을 사용하고 embedding dimension은 1536이다. optimizer는 Adam, learning rate는 $0.0002$이다. SSL pretraining 시 입력 해상도는 $128 \times 128$이다. 이후 meta-learning 단계에서는 기존 few-shot literature를 따라 $84 \times 84$ 이미지를 사용한다.

MiniImageNet에서는 세 가지 pretraining 모델이 준비된다. `Mini80-SSL`은 MiniImageNet의 training+validation에 해당하는 80개 클래스의 48,000장 이미지를 label 없이 사용해 SSL pretraining한 모델이다. `Mini80-SL`은 같은 AmdimNet을 supervised cross-entropy로 학습한 모델이다. `Image900-SSL`은 MiniImageNet을 제외한 ImageNet1K의 나머지 이미지로 SSL pretraining한 모델이다. CUB에서도 동일한 비교를 위해 `CUB150-SSL`, `CUB150-SL`, `Image1K-SSL`을 사용한다.

## 4. 실험 및 결과

### 데이터셋과 평가 설정

표준 few-shot 평가는 MiniImageNet과 CUB에서 수행된다. MiniImageNet은 100개 클래스, 총 60,000장 이미지로 구성되며, 저자들은 64 train, 16 validation, 20 test split을 따른다. CUB-200-2011은 200종의 새 이미지를 포함하는 fine-grained classification dataset이며, 논문은 100 train, 50 validation, 50 test split을 사용한다.

Cross-domain few-shot learning 평가에서는 CropDiseases, EuroSAT, ISIC, ChestX 네 개 데이터셋을 사용한다. 논문은 MiniImageNet과의 유사성이 점차 낮아지는 도메인으로 이들을 제시하며, embedding의 transferability를 검증하려 한다.

평가 task는 주로 5-way 1-shot, 5-way 5-shot이며, cross-domain에서는 5-way 5-shot, 20-shot, 50-shot이 사용된다.

### MiniImageNet 결과

MiniImageNet에서 제안 방법 `Ours Mini80 SSL`은 5-way 1-shot에서 $64.03 \pm 0.20\%$, 5-way 5-shot에서 $81.15 \pm 0.14\%$를 기록했다. 이는 비교된 baseline 중 높은 성능이다. 논문 본문에서도 ProtoNet+ 대비 1-shot에서 $7.53\%$, LEO 대비 $2.27\%$ 향상을 언급한다.

흥미로운 점은 같은 AmdimNet이라도 supervised pretraining을 한 `Mini80 SL`은 1-shot $43.92\%$, 5-shot $67.13\%$로 매우 낮다는 것이다. 반면 SSL pretraining만 하고 meta-learning 없이 nearest neighbor로 분류한 `Mini80 SSL -`은 1-shot $46.13\%$, 5-shot $70.14\%$다. 즉, 큰 backbone 자체가 좋은 것이 아니라 SSL pretraining과 meta-learning fine-tuning의 조합이 중요하다는 메시지를 준다.

또한 unlabeled pretraining data를 더 늘린 `Ours Image900 SSL`은 1-shot $76.82 \pm 0.19\%$, 5-shot $90.98 \pm 0.10\%$를 기록한다. 논문은 이를 통해 더 많은 unlabeled data가 representation quality를 크게 개선한다고 해석한다. 다만 이 수치는 baseline과 pretraining data 규모가 완전히 같지 않으므로, “같은 label budget에서 unlabeled data를 추가로 활용하면 매우 강력해진다”는 식으로 이해하는 것이 적절하다.

### CUB 결과

CUB에서는 `Ours CUB150 SSL`이 1-shot $71.85 \pm 0.22\%$, 5-shot $84.29 \pm 0.15\%$를 기록했다. 이는 표에 제시된 MatchingNet, MAML, ProtoNet, RelationNet, Baseline++, DN4-DA보다 높다. 논문은 ProtoNet 대비 1-shot에서 $20.54\%$의 큰 향상을 강조한다. fine-grained classification dataset인 CUB에서 이런 성능 향상은 representation quality가 특히 중요함을 보여준다.

여기서도 supervised pretraining의 약점이 드러난다. `Ours CUB150 SL`은 1-shot $45.10\%$, 5-shot $74.59\%$에 그쳤고, meta-learning 없이 SSL embedding만 쓰는 `Ours CUB150 SSL -`은 1-shot $40.83\%$, 5-shot $65.27\%$다. 즉, CUB에서도 SSL pretraining만으로 충분하지 않고, few-shot episode에 맞춘 추가 적응이 필요하다.

ImageNet 전체를 활용한 `Ours Image1K SSL`은 1-shot $77.09 \pm 0.21\%$, 5-shot $89.18 \pm 0.13\%$로 더 높다. 이는 ImageNet에서 학습한 representation이 CUB 같은 다른 도메인으로도 꽤 잘 전이됨을 시사한다.

### Cross-domain few-shot 결과

Cross-domain 실험에서는 Mini80-SSL embedding 하나만 사용하여 ChestX, ISIC, EuroSAT, CropDiseases에서 평가했다. 비교 대상은 Guo et al.의 Cross-domain few-shot 설정이다.

제안 방법은 ChestX에서 5-shot $28.50\%$, 20-shot $33.79\%$, 50-shot $38.78\%$를 기록해 baseline보다 높다. EuroSAT에서는 각각 $83.44\%$, $90.43\%$, $94.71\%$, CropDiseases에서는 $91.79\%$, $97.38\%$, $99.50\%$로 모두 baseline을 앞선다. ISIC에서는 제안 방법이 각각 $44.15\%$, $55.63\%$, $62.76\%$로 baseline보다 낮다. 논문도 이 부분은 “future work에서 더 조사 필요”하다고 명시한다.

전체 평균 기준으로는 제안 방법이 state-of-the-art인 $68.14\%$보다 높은 $69.69\%$를 달성했다고 주장한다. 이 결과는 표현이 보다 transferable하다는 논문의 핵심 주장과 연결된다. 다만 개별 도메인별로 보면 모든 경우에 일관되게 우세한 것은 아니고, 특히 ISIC에서는 약점이 존재한다.

### Ablation study의 의미

Ablation은 논문의 주장 구조를 뒷받침하는 중요한 부분이다. 첫째, 큰 네트워크를 supervised learning으로만 훈련하면 오히려 성능이 나빠진다는 점을 보였다. 이는 few-shot setting에서 label scarcity 때문에 대형 모델이 쉽게 overfit한다는 저자들의 가설과 맞는다.

둘째, SSL pretrained embedding을 meta-learning 없이 바로 nearest neighbor로 쓰는 경우보다, episodic meta-learning으로 fine-tuning했을 때 훨씬 좋다. 따라서 이 논문은 “좋은 embedding만 있으면 충분하다”보다는, “좋은 embedding과 task-aligned meta-learning adaptation이 함께 필요하다”는 쪽에 더 가깝다.

셋째, unlabeled pretraining data를 늘릴수록 성능이 추가로 좋아진다. 이는 SSL의 장점이 few-shot에서 특히 크게 발휘될 수 있음을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot learning에서 종종 당연하게 여겨졌던 supervised embedding pretraining의 한계를 정면으로 다루고, SSL을 통해 더 큰 backbone을 실질적으로 활용할 수 있음을 설득력 있게 보인 점이다. 단순히 최종 정확도만 높인 것이 아니라, supervised pretraining 대비, meta-learning 유무 대비, 더 많은 unlabeled data 사용 여부까지 비교하여 성능 향상의 원인을 구조적으로 설명한다.

또 다른 강점은 transferability에 대한 관심이다. 많은 few-shot 논문이 동일 도메인 내 split 성능에 집중하는 반면, 이 논문은 cross-domain few-shot evaluation을 포함해 representation robustness를 검증하려 한다. 실제로 EuroSAT, CropDiseases, ChestX에서 개선을 보인 것은 실용적인 의미가 있다.

방법 자체도 지나치게 복잡하지 않다. Stage 1에서 AMDIM으로 representation을 만들고, Stage 2에서 prototype-based meta-learning을 적용하는 구조이기 때문에 개념적으로 비교적 명확하다. 따라서 향후 다른 SSL objective나 backbone으로 대체 실험하기도 쉽다.

한편 한계도 분명하다. 첫째, 성능 향상의 상당 부분이 더 많은 unlabeled data 활용에서 오는 만큼, baseline과 완전히 동일한 정보 조건에서의 비교라고 보기는 어렵다. 특히 `Image900-SSL`이나 `Image1K-SSL` 결과는 representation learning 관점에서는 매우 흥미롭지만, “few-shot learner 자체의 알고리즘 우위”만을 순수 비교하는 실험은 아니다.

둘째, meta-learning loss의 수식 표현은 다소 불명확하다. 원문 Eq. (6)은 표준 negative log-softmax 형태와 엄밀히 일치하지 않는 듯 보이며, 논문은 이 부분을 자세히 설명하지 않는다. 따라서 구현 세부가 이 식만으로 완전히 재현 가능하다고 말하기는 어렵다.

셋째, cross-domain 결과가 모든 데이터셋에서 일관되게 좋지는 않다. ISIC에서 성능이 떨어진다는 점은 의료 영상처럼 시각적 통계가 일반 자연영상과 크게 다른 도메인에서는 SSL pretrained representation의 일반화가 충분하지 않을 수 있음을 시사한다. 저자들도 이를 인정한다.

넷째, 이 논문은 stage-wise pipeline의 효과를 잘 보였지만, 왜 AMDIM의 어떤 표현 특성이 few-shot adaptation에 특히 유리한지에 대한 더 깊은 분석은 제공하지 않는다. 예를 들어 local-global MI maximization이 class separation이나 intra-class compactness에 어떤 영향을 주는지까지는 논의되지 않는다.

## 6. 결론

이 논문은 few-shot image classification에서 embedding network의 사전학습 방식이 성능의 핵심 병목이라는 관점을 제시하고, self-supervised learning을 통해 그 병목을 효과적으로 완화할 수 있음을 보여준다. 핵심 기여는 크게 세 가지로 요약할 수 있다. 첫째, 순수 SSL로 학습한 대형 embedding network가 few-shot classification에 유리함을 실험적으로 보였다. 둘째, 이 embedding을 episodic meta-learning과 결합하면 MiniImageNet과 CUB에서 강한 성능을 얻을 수 있음을 보였다. 셋째, cross-domain few-shot setting에서도 representation의 transferability를 확인했다.

실제 적용 측면에서는 label이 부족하지만 unlabeled image는 비교적 얻기 쉬운 환경에서 특히 유용한 방향이다. 의료, 위성, 산업 비전처럼 라벨링이 비싸고 도메인 이동이 잦은 문제에서 의미가 크다. 향후 연구로는 논문이 제안하듯 SSL과 meta-learning을 end-to-end로 결합하는 방법, 혹은 few-shot detection 같은 다른 low-data vision task로 확장하는 방향이 자연스럽다.

종합하면, 이 논문은 “few-shot learner를 더 영리하게 만드는 것”뿐 아니라 “좋은 representation을 어떻게 준비할 것인가”가 equally important하다는 점을 강하게 보여준 작업으로 볼 수 있다.
