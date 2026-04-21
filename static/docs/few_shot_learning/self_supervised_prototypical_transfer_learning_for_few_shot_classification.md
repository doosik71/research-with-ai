# Self-Supervised Prototypical Transfer Learning for Few-Shot Classification

- **저자**: Carlos Medina, Arnout Devos, Matthias Grossglauser
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2006.11325

## 1. 논문 개요

이 논문은 **label이 없는 source domain 데이터만으로 few-shot classification에 쓸 수 있는 표현을 학습한 뒤**, 소수의 labeled support examples만으로 novel classes를 분류하는 방법인 **ProtoTransfer**를 제안한다. 핵심 문제의식은 분명하다. 기존 few-shot learning, 특히 meta-learning 기반 방법들은 대체로 목표 task와 관련된 대량의 annotation을 필요로 한다. 반면 unsupervised few-shot learning은 label 의존성을 줄였지만 성능이 떨어지는 경우가 많았고, 특히 **domain shift**가 큰 현실적 환경에서는 supervised meta-learning보다 일반적인 transfer learning이 더 잘 작동한다는 선행 결과가 있었다.

이 논문은 바로 이 지점에서 출발한다. 즉, few-shot learning을 반드시 episodic meta-learning으로 풀 필요가 있는가라는 질문에 대해, 저자들은 **self-supervised pre-training + prototypical adaptation**이라는 보다 직접적인 transfer learning 관점을 제시한다. 학습 단계에서는 각 이미지를 하나의 “prototype”처럼 취급하고, 그 이미지의 augmentation들이 해당 prototype 주변에 모이도록 embedding space를 학습한다. 이후 few-shot target task에서는 support set으로부터 class prototype을 만들고, 이를 초기값으로 사용해 classifier를 fine-tuning한다.

문제의 중요성은 두 가지다. 첫째, 실제 응용에서는 labeled data 수집 비용이 매우 크며, 특히 의료나 위성영상처럼 annotation이 어려운 도메인에서는 더욱 그렇다. 둘째, train/test class distribution이 달라지는 cross-domain few-shot learning은 실제 배치 환경에 가깝기 때문에, label-free pre-training으로도 robust한 transfer가 가능하다면 실용성이 높다. 이 논문은 이러한 관점에서, **“라벨이 없는 사전학습이 few-shot transfer에 충분히 강한가”**를 실험적으로 검증한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **prototype 개념을 self-supervised pre-training 단계와 downstream few-shot adaptation 단계 모두에서 일관되게 사용한다**는 점이다. 저자들은 pre-training 단계인 **ProtoCLR**에서, mini-batch 안의 각 원본 이미지 $x_i$를 하나의 class prototype처럼 간주한다. 그리고 그 이미지에 랜덤 augmentation을 여러 번 적용해 만든 $\tilde{x}_{i,q}$들이 embedding space에서 원본 $x_i$에 가깝고 다른 이미지 $x_k$들과는 멀어지도록 학습한다. 즉, 각 이미지가 자기 자신의 “class center” 역할을 하도록 만든다.

이 설계는 contrastive learning과 prototypical classification의 성격을 동시에 가진다. contrastive learning처럼 positive pair는 원본과 augmentation이고, negative는 batch 안의 다른 샘플들이다. 동시에 prototypical network처럼 “prototype에 가까운가”라는 관점으로 loss를 정의한다. 저자들이 강조하는 차별점은, 기존 unsupervised few-shot 방법들이 주로 **pseudo-labeling**이나 **artificial episodic task construction**에 의존했다면, ProtoTransfer는 **non-episodic transfer learning** 방식으로 더 큰 batch를 사용할 수 있고, 이것이 성능 향상에 핵심적이라는 점이다.

또한 target few-shot task에서는 support set 평균으로 class prototype을 만든 뒤, 단순 nearest-neighbor classification에 그치지 않고 **prototype으로 초기화한 linear layer를 fine-tuning**한다. 이 단계는 저자들이 **ProtoTune**이라고 부르며, 특히 shot 수가 늘어날수록 성능 향상에 기여한다. 따라서 논문의 핵심 아이디어는 단순히 “self-supervised learning을 썼다”가 아니라, **prototype 기반 구조를 pre-training과 adaptation 모두에 통합하고, meta-learning 대신 transfer learning 관점으로 few-shot classification을 재구성했다**는 데 있다.

## 3. 상세 방법 설명

전체 방법은 크게 두 단계로 구성된다.

첫 번째는 **Self-Supervised Prototypical Pre-Training (ProtoCLR)** 이다. 학습 시 batch 크기를 $N$이라 하면, batch에서 무작위로 뽑은 $N$개의 이미지 $\{x_i\}_{i=1}^N$를 사용한다. 이때 label 정보는 전혀 사용하지 않으므로, 각 샘플 $x_i$를 하나의 독립된 class처럼 취급한다. 그리고 각 샘플마다 $Q$개의 augmentation $\tilde{x}_{i,q}$를 생성한다. 이렇게 하면 원본 이미지 $x_i$는 support/prototype 역할을 하고, augmentation들은 query 역할을 한다.

이때 각 augmented query $\tilde{x}_{i,q}$에 대한 loss는 다음과 같다.

$$
\ell(i,q) = - \log \frac{\exp\left(-d[f(\tilde{x}_{i,q}), f(x_i)]\right)}{\sum_{k=1}^{N}\exp\left(-d[f(\tilde{x}_{i,q}), f(x_k)]\right)}
$$

여기서 $f_\theta(\cdot)$는 embedding function이고, $d[\cdot,\cdot]$는 거리 함수이다. 논문에서는 ProtoNet과 마찬가지로 **Euclidean distance**를 사용한다. 이 식의 의미는 간단하다. query augmentation의 embedding이 자기 원본 이미지의 embedding에는 가까워지고, batch 안의 다른 원본 이미지 embedding과는 멀어지도록 softmax cross-entropy를 최소화한다. 전체 batch loss는

$$
L = \frac{1}{NQ}\sum_{i=1}^{N}\sum_{q=1}^{Q}\ell(i,q)
$$

로 정의된다. 즉, 한 batch 안에서 총 $N \times Q$개의 query를 사용해 prototype classification을 수행하는 셈이다. 저자들은 이것을 self-supervised version의 prototypical loss로 해석한다.

이 방법에서 중요한 것은 **batch size를 downstream few-shot의 way 수에 맞출 필요가 없다는 점**이다. meta-learning은 보통 5-way, 20-way 같은 episode를 만들어 학습하지만, ProtoCLR은 단순 minibatch 기반이므로 훨씬 큰 $N$을 쓸 수 있다. 논문은 이 큰 batch가 representation quality를 높이는 핵심 요소라고 본다.

두 번째 단계는 **Supervised Prototypical Fine-Tuning (ProtoTune)** 이다. few-shot task가 주어지면, support set $S$에서 클래스 $n$의 prototype을 평균 embedding으로 계산한다.

$$
c_n = \frac{1}{|S_n|}\sum_{(x_i, y_i=n)\in S} f_\theta(x_i)
$$

이 식은 ProtoNet의 기본 prototype 정의와 같다. 여기서 저자들은 Snell et al. (2017)의 유도를 따라, prototype classifier를 linear classifier 형태로 다시 쓸 수 있다고 본다. 그래서 final linear layer를 다음과 같이 초기화한다.

$$
W_n = 2c_n,\quad b_n = -\|c_n\|^2
$$

즉, support examples로 만든 prototype을 classifier weight와 bias의 초기값으로 사용한다. 이후 support set 샘플들에 대해 softmax cross-entropy로 마지막 linear layer를 fine-tuning한다. in-domain 실험에서는 backbone $f_\theta$는 고정하고 마지막 layer만 학습하며, cross-domain CDFSL 실험에서는 domain shift가 크기 때문에 backbone까지 함께 fine-tuning한다.

정리하면, ProtoTransfer의 파이프라인은 다음과 같다. 먼저 unlabeled training data에서 augmentation 기반 prototype contrastive loss로 embedding을 학습한다. 그 다음 novel few-shot task에서 support set 평균으로 class prototype을 만든다. 마지막으로 이 prototype으로 classifier를 초기화하고, 필요하면 fine-tuning을 수행한 뒤 query를 분류한다. 이 구조는 pre-training과 inference/fine-tuning이 모두 prototype 중심으로 연결되어 있어 설계 일관성이 높다.

## 4. 실험 및 결과

논문은 in-domain과 cross-domain 두 환경에서 실험한다. in-domain에서는 **Omniglot**과 **mini-ImageNet**을 사용했고, cross-domain에서는 **CDFSL benchmark**를 사용했다. CDFSL은 mini-ImageNet으로 학습한 뒤 **CropDiseases, EuroSAT, ISIC, ChestX**로 transfer하는 설정이다. in-domain에서는 Conv-4 backbone, cross-domain에서는 ResNet-10을 사용했다. 기본 hyperparameter는 대부분 $N=50$, $Q=3$이다.

먼저 mini-ImageNet 결과를 보면, ProtoTransfer는 unsupervised few-shot learning의 기존 방법들을 뚜렷하게 앞선다. 5-way 1-shot에서 **45.67%**, 5-shot에서 **62.99%**, 20-shot에서 **72.34%**, 50-shot에서 **77.22%**를 기록했다. 이는 같은 표의 unsupervised baselines인 CACTUs, UMTRA, UFLST, ULDA보다 대체로 **4%에서 8% 정도 높은 수준**이다. 예를 들어 5-way 1-shot 기준으로 ULDA-MetaOptNet은 40.71%, ProtoTransfer는 45.67%다. supervised methods와 비교하면, 1-shot에서는 MAML 46.81%, ProtoNet 46.44%와 매우 근접하고, 5-shot에서는 MAML 62.13%를 넘으며 ProtoNet 66.33%에는 다소 못 미친다. 즉, **라벨 없이 사전학습했는데도 supervised few-shot과 상당히 비슷한 수준까지 간다**는 것이 핵심 결과다.

Omniglot에서는 ProtoTransfer가 경쟁력은 있지만 mini-ImageNet만큼 압도적이지는 않다. 5-way 1-shot 88.00%, 5-shot 96.48%를 기록했다. 일부 unsupervised 방법, 특히 UFLST나 AAL-MAML++이 더 높다. 따라서 저자들의 주장도 Omniglot에서는 “competitive” 수준이지, SOTA를 완전히 장악했다는 식은 아니다.

cross-domain CDFSL 결과가 이 논문의 중요한 포인트다. ChestX에서는 5-shot 26.71%, 20-shot 33.82%, 50-shot 39.35%로, supervised transfer baseline인 Pre+Linear의 25.97%, 31.32%, 35.49%를 모두 앞선다. 특히 가장 domain shift가 큰 ChestX에서 모든 방법 중 최고 성능을 기록한 점을 저자들은 강조한다. ISIC에서는 50-shot에서 66.15%로 Pre+Linear 66.48%와 사실상 비슷하고, EuroSAT와 CropDiseases에서도 supervised transfer와 대체로 근접하다. 요약하면, **cross-domain 상황에서는 ProtoTransfer가 unsupervised인데도 supervised transfer와 대부분 비슷하거나 일부에서는 더 좋은 성능**을 보인다.

ablation study는 이 논문의 설계를 이해하는 데 중요하다. mini-ImageNet에서 batch size를 5에서 50으로 늘리면 성능이 크게 오른다. UMTRA-style 설정인 작은 batch + single query에서는 5-way 1-shot이 39.17%인데, ProtoCLR로 batch 50을 쓰면 44.53%로 오른다. 5-shot에서는 53.78%에서 62.88%로, 20-shot에서는 62.41%에서 70.86%로 상승한다. 이는 **큰 batch가 representation learning 품질의 핵심**임을 직접 보여준다. query augmentation 수를 $Q=1$에서 $Q=3$으로 늘리는 효과는 비교적 작지만 일관되게 성능을 높인다. fine-tuning의 효과는 shot 수가 많을수록 뚜렷하다. 예를 들어 50-shot에서는 ProtoCLR-ProtoNet 74.31%에서 ProtoCLR-ProtoTune 77.22%로 증가한다.

또 하나 중요한 실험은 **training data diversity 감소 실험**이다. mini-ImageNet에서 전체 이미지 수를 줄이거나 클래스 수를 줄였을 때, ProtoTransfer와 supervised Pre+Linear를 비교했다. 이미지 수를 균등하게 줄일 때는 두 방법이 비슷하게 감소하지만, 클래스 다양성을 줄일 때는 ProtoTransfer가 훨씬 강하다. 예를 들어 training classes가 2개뿐인 상황에서 20-shot 정확도는 ProtoTransfer 64.59%, Pre+Linear 47.68%로 차이가 매우 크다. CUB로 학습해 mini-ImageNet으로 테스트하는 실험에서도 ProtoTransfer가 2%에서 4% 정도 더 좋다. 저자들은 이를 self-supervised 학습이 **latent class structure를 강제로 collapse하지 않기 때문**이라고 해석한다.

마지막으로 generalization gap 분석도 흥미롭다. train classes와 test classes에서의 few-shot 성능 차이를 비교했을 때, supervised ProtoNet은 6%에서 12%의 큰 성능 하락을 보이지만, ProtoCLR은 거의 차이가 없다. 예를 들어 5-way 20-shot에서 ProtoNet은 train 85.53%, test 76.73%로 크게 감소하지만, ProtoCLR은 train 71.51%, test 72.27%로 거의 동일하다. 논문은 이를 통해 self-supervised embedding이 novel classes에 더 안정적으로 일반화된다고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정과 방법이 잘 맞물린다는 점**이다. few-shot classification에서 label scarcity가 문제인데, 저자들은 unsupervised meta-learning의 복잡한 episode construction 대신 더 단순하고 확장 가능한 self-supervised transfer learning 구조를 제안한다. 이 접근은 개념적으로도 간결하고, batch size를 크게 쓸 수 있다는 practical advantage가 있다. 그리고 이 설계적 선택이 단지 직관 수준이 아니라 ablation study로 뒷받침된다.

두 번째 강점은 **cross-domain robustness**다. 단순히 mini-ImageNet in-domain에서만 잘 되는 것이 아니라, CDFSL benchmark에서 supervised transfer와 비슷한 수준을 낸다. 특히 ChestX에서 가장 좋은 성능을 얻은 것은 실제 응용 가능성 측면에서 의미가 크다. 의료 영상처럼 label이 부족하고 source-target gap이 큰 환경에서 유용할 가능성을 보여준다.

세 번째 강점은 **training class diversity가 낮을 때의 상대적 우수성**을 실험으로 보여준 점이다. 많은 few-shot 논문이 벤치마크 성능만 보고 끝나는 반면, 이 논문은 “라벨이 없다는 장점이 어떤 조건에서 특히 중요한가”를 따로 분석한다. class diversity가 줄어들수록 supervised pre-training이 더 약해지고, self-supervised approach가 더 강해진다는 관찰은 이 방법의 적용 조건을 이해하는 데 도움이 된다.

반면 한계도 분명하다. 첫째, Omniglot에서는 최상위 성능이 아니며, 모든 벤치마크에서 일관되게 최고라고 보기는 어렵다. 즉, 이 방법의 장점은 “언제나 최고”라기보다는 **라벨 없는 pre-training과 cross-domain transfer라는 조건에서 특히 유리하다**는 쪽에 가깝다. 둘째, augmentation 설계에 상당히 의존한다. 논문은 데이터셋별 augmentation을 조정했고, Appendix에 transform 세부 설정이 제시되어 있다. 따라서 성능의 일부는 self-supervision 방식 자체뿐 아니라 augmentation choice에도 영향을 받을 가능성이 있다. 다만 이 논문만으로 augmentation 민감도를 체계적으로 분리해 평가했다고 보기는 어렵다.

셋째, ProtoCLR의 pre-training은 “각 샘플을 자기 own class처럼 본다”는 강한 instance discrimination 가정을 사용한다. 이 방식이 representation learning에서는 효과적일 수 있지만, latent semantic class 구조를 직접 모델링하는 것은 아니다. 저자들도 pseudo-label clustering 기반 방법과 비교는 하지만, 어떤 경우에는 semantic-level grouping이 더 유리할 가능성이 있다. 이 부분은 미해결 질문으로 남는다.

넷째, fine-tuning 전략도 상황에 따라 다르다. in-domain에서는 마지막 layer만 학습하고, cross-domain에서는 backbone까지 fine-tuning한다. 이는 합리적인 선택이지만, 어떤 조건에서 어느 정도까지 fine-tuning해야 최적인지에 대한 일반 원리는 이 논문에서 완전히 정리되지는 않는다. 또한 validation set이 없는 CDFSL에서는 mini-ImageNet의 hyperparameter를 그대로 사용했기 때문에, 더 정교한 tuning을 하면 성능이 바뀔 가능성도 있다.

비판적으로 보면, 이 논문은 meta-learning에 대한 대안을 꽤 설득력 있게 제시하지만, 결국 backbone architecture나 augmentation quality 같은 representation learning의 일반 요소들에 많이 기대고 있다. 즉, “few-shot learning만의 고유한 학습 원리”를 새롭게 발명했다기보다, **modern self-supervised representation learning을 few-shot transfer에 잘 접목한 논문**으로 보는 편이 정확하다. 하지만 그것이 바로 이 논문의 실용적 가치이기도 하다.

## 6. 결론

이 논문은 **unlabeled source domain에서 prototype-based self-supervised embedding을 학습하고, target few-shot task에서 prototype initialization과 fine-tuning으로 적응하는 ProtoTransfer**를 제안했다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, unsupervised few-shot learning에서 meta-learning이 아니라 transfer learning 관점을 본격적으로 제시했다. 둘째, ProtoCLR이라는 self-supervised prototypical loss와 ProtoTune이라는 prototypical fine-tuning을 결합해 일관된 framework를 만들었다. 셋째, mini-ImageNet에서는 기존 unsupervised few-shot baselines를 크게 앞서고, cross-domain benchmark에서는 supervised transfer와 경쟁 가능한 수준임을 보였다.

실제 적용 측면에서 이 연구는 label이 비싸거나 구하기 어려운 도메인, 특히 domain shift가 큰 환경에서 중요하다. 의료영상, 위성영상, 특수 산업 데이터처럼 source에는 unlabeled data만 많고 target에는 소수 labeled example만 있는 상황에서 ProtoTransfer류 접근은 충분히 실용적일 수 있다. 향후 연구로는 더 강한 self-supervised backbone, 더 체계적인 augmentation 설계, semantic clustering과의 결합, 그리고 class diversity가 매우 낮은 환경에서의 일반화 메커니즘 분석 등이 자연스러운 확장 방향으로 보인다.
