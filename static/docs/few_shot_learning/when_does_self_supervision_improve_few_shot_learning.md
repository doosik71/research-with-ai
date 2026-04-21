# When Does Self-supervision Improve Few-shot Learning?

- **저자**: Jong-Chyi Su, Subhransu Maji, Bharath Hariharan
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/1910.03560

## 1. 논문 개요

이 논문은 few-shot learning에서 self-supervised learning(SSL)이 실제로 언제 도움이 되는지를 체계적으로 분석한 연구이다. 저자들은 대규모 비라벨 데이터에서 SSL이 유용하다는 기존 흐름과 달리, 데이터가 작고 외부 비라벨 데이터를 별도로 확보하기 어려운 현실적인 few-shot setting에서도 SSL이 유효한지 묻는다. 특히 “기존 few-shot meta-learner의 성능을, 같은 데이터셋 내부 이미지들만 사용한 SSL로도 개선할 수 있는가”, “추가 비라벨 이미지가 있다면 항상 더 좋은가”, “비라벨 이미지의 도메인이 다르면 어떤 일이 생기는가”를 핵심 문제로 둔다.

문제의식은 명확하다. few-shot learning은 base class에서 representation을 학습한 뒤 novel class에 소량의 예시만 주고 일반화해야 한다. 그런데 supervised base-class training만 하면 base class를 구분하는 데만 필요한 특징에 representation이 과도하게 맞춰질 수 있다. 그러면 novel class를 구분하는 데 필요한 의미적 정보가 버려질 수 있다. 이 현상은 base 데이터가 작거나, fine-grained recognition처럼 클래스 간 차이가 미세한 어려운 문제에서 더 심해질 가능성이 크다. 저자들은 SSL이 이런 과적합을 완화하는 데이터 의존적 regularizer 역할을 할 수 있다고 본다.

논문의 실질적 중요성은 두 가지다. 첫째, SSL을 “supervised learning의 대체재”가 아니라 “few-shot transfer 성능을 높이는 보조 학습 신호”로 재해석했다. 둘째, 추가 비라벨 데이터가 많다고 항상 좋은 것이 아니라, target domain과의 domain similarity가 핵심이라는 점을 실험적으로 보여주고, 이를 이용해 generic unlabeled pool에서 유사 도메인 이미지를 자동 선택하는 간단한 방법까지 제시했다.

## 2. 핵심 아이디어

중심 아이디어는 supervised few-shot learning loss와 self-supervised loss를 하나의 shared feature backbone 위에서 함께 학습하는 것이다. 논문은 이를 통해 SSL이 feature extractor가 base labels에만 과도하게 적응하는 것을 막고, novel classes로 transfer될 때 더 일반적인 시각적 구조와 semantic cue를 유지하도록 만든다고 해석한다.

이 논문의 중요한 차별점은 SSL을 대규모 외부 데이터에서 사전학습(pretraining)하는 전통적 방식으로 다루지 않는다는 점이다. 대신 few-shot 학습과 동시에 auxiliary task로 넣는다. 즉, SSL은 별도의 representation pretraining 단계가 아니라 few-shot learner의 representation learning 자체를 보조하는 regularizer로 사용된다. 저자들은 실제로 “SSL만으로 학습한 표현”이나 “SSL pretraining 후 meta-learning”은 충분하지 않으며, supervised few-shot objective와 함께 학습할 때 가장 효과적이라고 보고한다.

또 하나의 핵심 통찰은 “어떤 unlabeled data를 쓰느냐”가 성능을 좌우한다는 점이다. 같은 도메인 또는 유사한 도메인의 이미지로 SSL을 수행하면 few-shot 성능이 좋아지지만, 도메인이 어긋난 이미지를 섞으면 오히려 성능이 떨어질 수 있다. 따라서 SSL의 효과는 단순한 데이터 양이 아니라 target task와의 도메인 정렬 정도에 크게 의존한다.

## 3. 상세 방법 설명

전체 구조는 매우 단순하다. 라벨이 있는 데이터셋 $D_s = \{(x_i, y_i)\}_{i=1}^n$에 대해, 입력 이미지 $x$를 embedding space로 보내는 feature backbone $f(x)$를 두고, supervised 분류기 $g$와 self-supervised 예측기 $h$를 별도로 붙인다. supervised objective는 다음과 같이 정의된다.

$$
L_s := \sum_{(x_i, y_i) \in D_s} \ell(g \circ f(x_i), y_i) + R(f, g)
$$

여기서 $\ell$은 일반적으로 cross-entropy loss이고, $R(f,g)$는 예를 들어 $\ell_2$ regularization 같은 항이다. few-shot transfer에서는 base classes 위에서 $f$와 $g$를 학습한 뒤, novel classes에 대해서는 기존 $g$를 버리고 새로운 classifier를 적은 샘플로 다시 구성한다.

self-supervised learning은 이미지로부터 자동으로 label을 만드는 pretext task를 사용한다. self-supervised 데이터셋을 $D_{ss}$라고 할 때, 각 이미지 $x_i$를 변형하여 $(\hat{x}_i, \hat{y}_i)$를 만들고, 이를 backbone $f$와 auxiliary head $h$를 통해 예측한다. self-supervised loss는 다음과 같다.

$$
L_{ss} := \sum_{x_i \in D_{ss}} \ell(h \circ f(\hat{x}_i), \hat{y}_i)
$$

최종 학습 목표는 두 손실을 합한 것이다.

$$
L := L_s + L_{ss}
$$

부록의 구현 세부에서는 실제로 가중합 형태 $L = (1-\lambda)L_s + \lambda L_{ss}$를 사용했고, 대부분 $\lambda = 0.5$가 가장 좋았으며, mini-ImageNet과 tiered-ImageNet에서 jigsaw의 경우는 $\lambda = 0.3$을 썼다고 명시한다. 두 개의 SSL task를 동시에 쓸 때는 두 SSL loss를 평균내어 $L_{ss}$를 구성했다.

few-shot learner로는 주로 ProtoNet을 사용한다. meta-training에서는 base set $D_b$에서 $N$개 클래스를 고르고, 각 클래스당 $K$개 support image와 $M$개 query image를 뽑아 $N$-way $K$-shot episode를 구성한다. support embedding의 class mean을 prototype으로 만들고, query를 prototype과의 거리로 분류한다. 테스트 시에도 novel set $D_n$에서 같은 방식으로 prototype을 다시 계산해 분류한다. 이 논문은 ProtoNet 외에도 MAML과 일반 softmax classifier도 실험하여 SSL의 효과가 특정 meta-learner에만 묶이지 않음을 보인다.

사용한 self-supervised task는 두 가지다.

첫째는 jigsaw puzzle task이다. 입력 이미지를 $3 \times 3$ 패치로 나누고 무작위로 섞은 뒤, 어떤 permutation이 적용되었는지를 맞히게 한다. 원래 가능한 permutation은 $9!$개지만, [41]을 따라 Hamming distance 기반으로 35개 그룹으로 줄여 difficulty를 조절했다. 즉, target label $\hat{y}$는 35개 클래스 중 하나이다.

둘째는 rotation prediction task이다. 입력 이미지를 $\theta \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$ 중 하나로 회전시켜 $\hat{x}$를 만들고, 회전 각도 index를 예측한다. 두 task 모두 cross-entropy loss를 사용한다.

중요한 설계 포인트는 $D_s$와 $D_{ss}$가 꼭 같을 필요는 없다는 점이다. 저자들은 세 가지 질문을 명시한다. 첫째, $D_s = D_{ss}$일 때 작은 데이터셋에서도 SSL이 유효한가. 둘째, $D_s \neq D_{ss}$일 때 domain shift가 성능에 어떤 영향을 주는가. 셋째, 큰 generic unlabeled pool에서 target domain에 맞는 이미지를 어떻게 골라 $D_{ss}$를 구성할 수 있는가.

학습 절차는 $D_s = D_{ss}$일 때와 아닐 때가 조금 다르다. 같은 데이터일 때는 같은 batch로 $L_s$와 $L_{ss}$를 함께 계산한다. 서로 다른 도메인을 쓸 때는 SSL용 batch size 64의 별도 batch를 뽑아 self-supervised forward pass를 수행한 뒤 두 loss를 합쳐 gradient update를 한다. 저자들은 복잡한 loss balancing 기법도 가능하지만, 단순 평균이 잘 작동했다고 보고한다.

아키텍처 세부도 일부 제공된다. backbone은 주로 ResNet-18이다. jigsaw head는 각 패치로부터 512차원 feature를 얻고, 추가 fully connected layer를 거쳐 9개 패치 feature를 이어붙인 뒤 4608차원에서 4096차원으로 줄이고 마지막에 35-way 출력을 낸다. rotation head는 ResNet-18의 512차원 출력을 받아 $\{512, 128, 128, 4\}$ fully connected layers를 거치며, 각 사이에 ReLU와 dropout 0.5를 둔다.

도메인 선택 방법은 매우 간단한 importance weighting 방식이다. target labeled domain $D_s$와 대규모 unlabeled pool $D_p$가 있을 때, ResNet-101의 penultimate-layer feature를 추출하고, $D_s$를 positive class, $D_p$를 negative class로 하는 binary logistic regression domain classifier를 학습한다. 이후 각 이미지 $x$에 대해

$$
\frac{p(x \in D_s)}{p(x \in D_p)}
$$

비율이 큰 이미지를 선택하여 $D_{ss}$를 만든다. 논문은 이 비율을 domain shift를 고려한 importance weight로 해석한다.

## 4. 실험 및 결과

실험 데이터셋은 mini-ImageNet, tiered-ImageNet, 그리고 다섯 개 fine-grained dataset인 Birds, Cars, Aircrafts, Dogs, Flowers이다. 각 데이터셋은 base, val, novel class split으로 나뉘고, base에서 representation을 학습한 뒤 novel class에서 few-shot 성능을 평가한다. fine-grained 데이터셋은 수천 장 수준으로 작아서, 대규모 데이터셋보다 SSL의 low-data 효과를 보기 적합하다.

기본 실험 설정은 ResNet-18 backbone 위 ProtoNet, 5-way 5-shot training, class당 16 query image, ADAM optimizer, learning rate 0.001, 60,000 episodes이다. 성능은 600개 test episode 평균 accuracy와 95% confidence interval로 보고한다. 평가는 5-way 5-shot과 20-way 5-shot 둘 다 수행한다.

가장 중요한 결과는 SSL이 few-shot learning을 일관되게 개선한다는 것이다. 5-way 5-shot에서 ProtoNet baseline 대비 ProtoNet + Jigsaw는 mini-ImageNet 75.2%에서 76.2%, tiered-ImageNet 75.9%에서 78.0%, Birds 87.3%에서 89.8%, Cars 91.7%에서 92.4%, Aircrafts 91.4%에서 91.8%, Dogs 83.0%에서 85.7%, Flowers 89.2%에서 92.2%로 상승했다. 저자들은 이를 relative error rate reduction으로 환산해 각각 4.0%, 8.7%, 19.7%, 8.4%, 4.7%, 15.9%, 27.8% 감소라고 정리한다. 특히 fine-grained 소규모 데이터셋에서 개선폭이 더 크게 보인다.

rotation task도 대부분 데이터셋에서 개선을 주지만, Aircrafts와 Flowers에서는 효과가 약하거나 없다. 논문은 그 이유로 Aircrafts는 비행기가 대체로 수평 방향이라 rotation 예측이 지나치게 쉽거나 덜 유익할 수 있고, Flowers는 대칭적인 이미지가 많아 rotation signal이 semantic discrimination에 덜 연결될 수 있다고 해석한다. 다만 이 부분은 저자들의 해석이며, 직접적인 인과 검증은 논문에 제시되지 않았다.

두 SSL task를 결합한 결과도 일부 데이터셋에서 추가 이득이 있었다. 예를 들어 mini-ImageNet은 76.6%, Birds는 90.2%, Cars는 92.7%, Dogs는 85.9%까지 올라간다. 그러나 모든 데이터셋에서 항상 최선은 아니어서, task combination의 효과가 데이터셋 의존적임을 보여준다.

더 어려운 문제일수록 SSL의 상대적 이득이 커진다는 점도 중요한 발견이다. 입력을 degraded version으로 만들거나 base training image 수를 20%로 줄인 실험에서 SSL의 이득이 더 커졌다. 예를 들어 20% Cars 데이터에서 ProtoNet은 75.8%인데 ProtoNet + Jigsaw는 82.8%로 무려 7.0%p 상승했다. 원래 full data Cars에서는 91.7%에서 92.4%로 0.7%p 상승에 그쳤다. 즉, 데이터가 부족하거나 task difficulty가 커질수록 SSL regularization이 더 가치 있다는 결론이다.

20-way 5-shot에서도 개선은 유지되며 종종 더 크다. 예를 들어 Birds는 69.3%에서 73.7%, Dogs는 61.6%에서 65.4%, Flowers는 75.4%에서 79.2%로 상승했다. 논문은 20-way가 더 어려운 설정이므로 SSL의 효용이 더 잘 드러난다고 본다.

다른 meta-learner로의 일반화도 확인했다. Table 2에 따르면 평균 5-way 5-shot accuracy는 softmax가 85.5%에서 86.6%, MAML이 82.6%에서 83.8%, ProtoNet이 88.5%에서 90.4%로 올라갔다. 즉, SSL의 효과는 특정 few-shot 알고리즘에만 한정되지 않는다. 다만 절대 성능은 ProtoNet + Jigsaw가 가장 좋았다.

반대로 “SSL alone”은 충분하지 않았다. 다섯 fine-grained dataset 평균 5-way 5-shot accuracy 기준으로 SSL only는 jigsaw 32.9%, rotation 33.7% 수준으로, random initialization 29.5%보다는 낫지만 supervised cross-entropy 85.5%보다 훨씬 낮았다. 또한 SSL pretraining 후 meta-learning initialization도 random initialization에서 바로 meta-learning하는 것보다 좋지 않았다고 보고한다. 이 결과는 SSL이 standalone representation learner라기보다 supervised few-shot 학습을 보조하는 regularizer라는 논문의 해석을 지지한다.

기존 연구와 비교하면, mini-ImageNet 5-way 5-shot에서 저자들의 ProtoNet baseline은 75.2%, rotation은 76.0%, jigsaw는 76.2%, 둘 다 사용하면 76.6%였다. Table 3은 다른 논문들과 image size, backbone, training details가 다르므로 완전한 apple-to-apple 비교는 아니지만, 더 큰 입력 해상도 224×224와 ResNet-18을 사용한 설정에서도 SSL의 이득이 유지된다는 점을 보여준다.

도메인 shift 분석도 이 논문의 핵심 실험이다. 먼저 같은 도메인의 unlabeled images를 더 많이 추가하면 성능이 좋아진다. 20%의 labeled images만 meta-learning에 쓰고, 나머지 80%는 label을 숨긴 채 SSL에만 쓰는 설정에서, SSL에 쓰는 이미지 수가 늘수록 정확도는 증가하지만 diminishing returns를 보인다. 즉, 같은 도메인이라면 추가 비라벨 데이터는 대체로 유익하다.

하지만 out-of-domain 이미지를 섞으면 상황이 달라진다. 원래 unlabeled set 일부를 다른 네 개 데이터셋 이미지로 대체해 domain shift를 키우면, SSL의 효과가 지속적으로 감소한다. 논문은 어떤 경우에는 동일 도메인 이미지 20%만 사용하는 편이, 다섯 배 더 많은 이미지가 있더라도 그중 상당수가 다른 도메인인 경우보다 더 낫다고 보고한다. 이것은 “데이터가 많을수록 무조건 좋다”는 단순한 가정을 반박한다.

이 분석을 바탕으로 제안한 domain-based image selection은 실험적으로 유효했다. unlabeled pool $D_p$는 Open Images V5 bounding box subset의 1,743,042장과 iNaturalist 2018의 461,939장으로 구성했다. 20% labeled data만 있는 상황에서, 같은 개수의 추가 SSL images를 세 방식으로 비교했다. 무작위 선택인 `SSL Pool (random)`은 Cars, Dogs, Flowers에서 오히려 성능을 해쳤다. 반면 importance weight 기반 `SSL Pool (weight)`는 모든 데이터셋에서 개선을 보였다. 예를 들어 Birds는 No SSL 73.0%, SSL 20% dataset 74.4%, random 74.1%, weight 76.4%, oracle 78.4%였다. Cars는 75.8%, 82.1%, 78.4%, 82.9%, 83.7% 순이었다. 즉, 제안 방식은 lower bound인 “dataset 내부 20%만 SSL에 사용”보다도 낫고, upper bound인 oracle보다는 약간 못 미치는 실용적인 절충안으로 작동한다.

부록에서는 standard fine-grained classification에서도 비슷한 현상을 보였다. 예를 들어 Birds에서 softmax 47.0%가 softmax + rotation 51.1%로, Flowers에서 72.8%가 softmax + jigsaw 76.4%로 올랐다. 이는 few-shot transfer뿐 아니라 scratch supervised training에서도 SSL이 도움을 줄 수 있음을 시사한다. 다만 본 논문의 중심 주장은 few-shot learning이며, standard classification은 보조 결과에 가깝다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 질문이 실용적이고, 답이 명확한 실험으로 뒷받침된다는 점이다. “작은 데이터셋에서도 SSL이 도움이 되는가”라는 질문에 대해, 외부 데이터 없이도 여러 few-shot benchmark에서 일관된 개선을 보였다. 특히 fine-grained, 저해상도, grayscale, 20% data 같은 더 어려운 환경에서 상대적 이득이 커진다는 결과는 SSL을 regularizer로 보는 해석과 잘 맞는다.

또 다른 강점은 domain shift를 독립 변수처럼 조절하며 체계적으로 분석했다는 점이다. 많은 SSL 연구가 대규모 데이터와 downstream transfer 성능만 보여주는 데 비해, 이 논문은 unlabeled data의 양과 도메인 차이를 따로 분리해 관찰한다. 그 결과 “same-domain unlabeled data는 좋지만 out-of-domain unlabeled data는 해로울 수 있다”는 구체적 메시지를 준다. 이는 실무에서 외부 데이터를 마구 수집해 SSL에 넣는 접근에 중요한 경고가 된다.

방법 자체가 단순하다는 것도 장점이다. 기존 few-shot pipeline에 auxiliary SSL head와 loss만 더하면 된다. 그리고 unlabeled pool selection도 고차원적인 복잡한 방법이 아니라, pretrained ResNet-101 feature 위 logistic regression domain classifier로 구현된다. 단순한데도 실험적으로 효과가 있다.

반면 한계도 분명하다. 첫째, SSL task가 jigsaw와 rotation 두 가지에 주로 제한되어 있다. 저자들도 conclusion에서 contrastive learning 계열은 미래 과제로 남긴다. 따라서 “언제 SSL이 도움이 되는가”라는 제목에 비해, 실제 결론은 “적어도 jigsaw/rotation 기반 auxiliary SSL은 이런 조건에서 도움이 된다” 정도로 읽는 것이 정확하다.

둘째, 왜 SSL이 어려운 few-shot task에서 더 크게 도움이 되는지에 대한 메커니즘 분석은 제한적이다. 부록의 saliency map 시각화는 self-supervised 모델이 배경보다 foreground에 더 집중하는 경향을 시사하지만, 이것이 주된 원인인지, 혹은 representation smoothness, invariance, calibration 같은 다른 요인이 중요한지는 명확히 밝혀지지 않았다.

셋째, domain similarity 측정 방식이 매우 단순하다. ResNet-101 ImageNet pretrained feature와 logistic regression으로 importance weight를 계산하는데, 이것이 모든 도메인에서 최선의 선택 기준인지는 논문이 다루지 않는다. 또한 선택된 이미지가 왜 도움이 되는지에 대한 정성적 분석은 제한적이다.

넷째, 일부 결과는 해석적 수준에 머문다. 예를 들어 Aircrafts에서 rotation이 약한 이유를 “비행기가 보통 수평이라서”라고 추정하지만, task difficulty나 shortcut availability를 직접 계량적으로 분석하지는 않는다. Flowers의 대칭성에 대한 해석도 마찬가지다.

다섯째, prior work와의 비교는 backbone, image size, augmentation, regularization 등이 달라 직접 비교에 조심해야 한다. 논문도 이를 표에서 명시한다. 따라서 본 논문의 강점은 “state-of-the-art 달성”보다는 “현상 분석과 practical recipe 제시”에 있다고 보는 편이 적절하다.

## 6. 결론

이 논문은 few-shot learning에서 self-supervision이 단지 대규모 pretraining의 대안이 아니라, 작은 데이터 환경에서도 supervised meta-learning을 보조하는 유용한 regularizer가 될 수 있음을 보여준다. 특히 같은 데이터셋 내부 이미지들만으로도 성능 향상이 가능하고, 데이터가 더 작거나 과제가 더 어려울수록 그 효과가 더 커진다는 점이 핵심 기여다.

또한 추가 비라벨 데이터의 효과가 도메인 유사성에 강하게 의존한다는 사실을 실험적으로 입증했고, 이를 바탕으로 generic unlabeled pool에서 target domain과 유사한 이미지를 자동 선택하는 간단한 방법을 제안했다. 이 방법은 few-shot 데이터가 작고 외부 라벨링이 어려운 생물학, 의학, fine-grained recognition 같은 실제 응용 분야에서 특히 의미가 있다.

종합하면, 이 연구의 핵심 메시지는 다음과 같다. few-shot learning에서는 “라벨 없는 데이터도 무조건 많으면 좋다”가 아니라, “적절한 self-supervised task와 target domain에 맞는 unlabeled data가 있을 때 SSL이 강력한 보조 신호가 된다.” 이 관점은 이후 contrastive SSL, domain-aware pretraining, low-data transfer learning 연구로 자연스럽게 이어질 수 있는 중요한 발판이다.
