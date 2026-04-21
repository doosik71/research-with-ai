# Few-Shot One-Class Classification via Meta-Learning

- **저자**: Ahmed Frikha, Denis Krompaß, Hans-Georg Köpken, Volker Tresp
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2007.04146

## 1. 논문 개요

이 논문은 **few-shot one-class classification (FS-OCC)** 문제를 다룬다. 이는 학습 시점에 **오직 하나의 클래스(normal class)** 에서 나온 **아주 적은 수의 샘플만** 주어졌을 때, 정상과 비정상을 구분해야 하는 문제다. 일반적인 few-shot classification은 각 클래스에서 몇 개씩 예시를 받는 반면, one-class classification은 아예 비정상 클래스 예시가 없거나 거의 없다는 점에서 훨씬 어렵다.

논문이 해결하려는 핵심 문제는 다음과 같다. 기존의 **one-class classification (OCC)** 방법들은 보통 정상 데이터가 충분히 많아야 잘 작동하고, 기존의 **meta-learning** 기반 few-shot 방법들은 보통 학습 시 각 클래스의 예시가 모두 필요하다. 따라서 “정상 클래스 샘플 몇 개만으로 새로운 anomaly detection task를 빠르게 학습”해야 하는 FS-OCC 상황에서는 둘 다 직접적으로 잘 맞지 않는다.

이 문제가 중요한 이유는 실제 anomaly detection 응용이 대부분 이 설정과 가깝기 때문이다. 의료, 보안, 산업 제조 같은 영역에서는 이상(anomaly)이 드물고, 정상 데이터조차 충분히 모으기 어렵거나 주석 비용이 비싸다. 특히 산업 제조에서는 새로운 공정이나 장비에 대해 초기에 정상 샘플 몇 개만 확보되는 경우가 많다. 따라서 FS-OCC를 잘 푸는 방법은 실용적 가치가 매우 높다.

이 논문의 기본 주장은 간단하다. **MAML 같은 gradient-based meta-learning을 그대로 쓰면 FS-OCC에 적합한 초기화(initialization)를 얻기 어렵고, inner-loop와 outer-loop의 데이터 샘플링 방식을 바꿔야 한다.** 이를 위해 저자들은 **OC-MAML (One-Class MAML)** 을 제안한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 **meta-training episode 자체를 FS-OCC 상황과 맞추는 것**이다. 일반 MAML은 inner loop와 outer loop 모두 class-balanced 데이터를 사용하는 few-shot classification을 가정한다. 하지만 FS-OCC의 테스트 상황에서는 adaptation 시 정상 클래스 샘플만 주어진다. 따라서 학습 단계에서도 그 상황을 그대로 흉내 내야 한다는 것이 저자들의 설계 직관이다.

구체적으로는, inner loop에서는 **one-class minibatch** 만 사용해 task adaptation을 하고, outer loop에서는 **class-balanced validation minibatch** 로 meta-update를 수행한다. 이렇게 하면 모델은 “정상 클래스 샘플만 보고 업데이트했을 때도, 균형 잡힌 정상/이상 평가셋에서 성능이 좋아지도록” 초기화를 학습하게 된다. 즉, 테스트 때 비정상 샘플이 없어도, 정상 샘플 몇 개만으로 decision boundary를 유의미하게 조정할 수 있게 만드는 것이다.

기존 접근과의 차별점은 두 가지다. 첫째, 저자들은 단순히 heuristic하게 one-class 학습을 붙인 것이 아니라, **왜 second-order meta-learning이 필요한지 이론적으로 설명**한다. 둘째, 제안이 특정 모델 구조에 묶여 있지 않다. 예를 들어 concurrent work인 **One-Way ProtoNets** 는 마지막 layer의 batch normalization이 만드는 0-centered embedding에 기대는데, 이 논문은 그런 구조적 제약 없이 **episode sampling 전략** 자체를 바꾼다. 그래서 MAML뿐 아니라 MetaOptNet, Meta-SGD 같은 다른 bi-level meta-learning에도 적용 가능하다고 주장한다.

## 3. 상세 방법 설명

### 3.1 문제 설정

저자들은 FS-OCC를 **1-way K-shot classification** 으로 해석한다. 일반 few-shot에서 $N$-way $K$-shot은 각 클래스당 $K$개의 샘플이 주어지는 문제인데, 여기서는 학습 시점에 하나의 클래스만 있으므로 1-way $K$-shot이다. 다만 anomaly detection 문맥에 맞게, 그 하나의 클래스는 **정상 클래스** 여야 한다.

이 설정이 어려운 이유는 decision boundary를 너무 타이트하게 잡으면 정상 샘플도 anomaly로 오분류하고, 반대로 너무 넓게 잡으면 anomaly를 놓치기 때문이다. 즉, 극소수 정상 샘플만으로 **일반화 가능한 정상 클래스 경계** 를 형성해야 한다.

### 3.2 MAML 요약

MAML은 여러 task에서 공통적으로 좋은 초기 파라미터 $\theta$를 meta-learn한다. 각 task $T_i$는 adaptation용 $D_i^{tr}$ 와 validation용 $D_i^{val}$ 로 나뉜다. 먼저 $D_i^{tr}$ 에서 몇 번 gradient step을 적용해 task-specific parameter $\theta_i'$ 를 얻고, 그 다음 $D_i^{val}$ 에서의 loss를 줄이도록 원래 초기값 $\theta$를 업데이트한다.

논문은 이 과정을 다음과 같이 적는다.

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{T_i \sim p(T)} L_{T_i}^{val}(f_{\theta_i'})
$$

여기서 $\beta$는 meta-update learning rate다. 핵심은 “초기화 $\theta$에서 출발해서 few-shot adaptation을 했을 때 validation loss가 잘 줄어드는가”를 학습한다는 점이다.

### 3.3 OC-MAML

OC-MAML의 핵심 수정은 **inner-loop minibatch의 class imbalance rate (CIR)** 를 테스트 상황에 맞게 바꾸는 것이다. 논문은 hyperparameter $c$를 도입해 inner-loop 배치에서 minority class 비율을 조절한다. FS-OCC의 극단적 경우는 anomaly 샘플이 전혀 없는 경우이므로, 저자들은 주로 $c = 0\%$ 를 사용한다.

메타 학습 절차는 다음과 같다.

1. meta-training task $T_i$를 샘플링한다.
2. adaptation용 데이터 $D_i^{tr}$ 에서 **정상 클래스만 포함된** 크기 $K$의 minibatch $B$를 뽑는다.
3. 이 $B$로 몇 번 gradient step을 수행해 $\theta_i'$ 를 얻는다.
4. validation용 데이터 $D_i^{val}$ 에서는 **class-balanced** 한 크기 $Q$의 minibatch $B'$ 를 뽑는다.
5. $B'$ 에서 계산한 loss를 사용해 초기 파라미터 $\theta$를 meta-update한다.

논문은 이 과정을 Algorithm 1로 제시한다. inner loop는 one-class, outer loop는 class-balanced라는 점이 가장 중요하다. 손실 함수는 cross-entropy를 사용했다고 명시한다.

### 3.4 왜 이 방식이 작동하는가

논문의 가장 중요한 기여 중 하나는 이론 분석이다. 저자들은 MAML의 update를 Taylor expansion으로 근사해 다음 식을 제시한다.

$$
g_{\text{MAML}} = g_2 - \alpha H_2 g_1 - \alpha H_1 g_2 + O(\alpha^2)
= g_2 - \alpha \frac{\partial (g_1 \cdot g_2)}{\partial \phi_1} + O(\alpha^2)
$$

여기서 $g_1, g_2$ 는 서로 다른 minibatch에서 계산된 gradient이고, $H_1, H_2$ 는 Hessian이다. 이 식이 의미하는 바는, MAML이 본질적으로 **서로 다른 minibatch에서 계산된 gradient들의 inner product를 키우도록** 학습한다는 것이다. inner product가 커지면 gradient 방향 사이의 각도가 줄고, cosine similarity도 증가한다.

일반 MAML에서는 두 minibatch가 모두 class-balanced이므로, “class-balanced batch에서 계산한 gradient들끼리 방향이 잘 맞는 초기화”를 학습한다. 하지만 FS-OCC 테스트에서는 adaptation 때 one-class batch만 볼 수 있다. 이 경우 일반 MAML 초기화는 적절하지 않을 수 있다.

반면 OC-MAML에서는 첫 번째 minibatch가 one-class, 두 번째 minibatch가 class-balanced다. 따라서 **one-class gradient와 class-balanced gradient의 inner product를 키우는 방향으로 초기화** 를 학습하게 된다. 결과적으로 정상 샘플만으로 몇 번 업데이트해도, class-balanced 평가 데이터에서 성능이 개선되는 초기화가 만들어진다.

### 3.5 왜 first-order 방법은 충분하지 않은가

저자들은 FOMAML과 Reptile도 같은 방식으로 OCC에 맞게 바꿔볼 수 있는지 분석한다. 근사식은 다음과 같다.

$$
g_{\text{FOMAML}} = g_2 - \alpha H_2 g_1 + O(\alpha^2)
$$

$$
g_{\text{Reptile}} = g_1 + g_2 - \alpha H_2 g_1 + O(\alpha^2)
$$

핵심 주장은, MAML과 달리 이들 식에는 **$H_1 g_2$ 항이 없다**는 점이다. class-balanced setting에서는 기대값 수준에서 일부 대칭성이 성립하지만, one-class minibatch와 class-balanced minibatch처럼 **CIR이 다른** 두 배치를 쓰면 그 성질이 깨진다. 그래서 OC-FOMAML이나 OC-Reptile은 one-class gradient와 class-balanced gradient의 정렬을 명시적으로 최적화하지 못한다.

즉, 저자들의 결론은 다음과 같다. **FS-OCC에 적합한 초기화를 meta-learn하려면 second-order derivative term이 필요하다.** 이것이 OC-MAML이 잘 되고, 단순한 first-order 변형은 잘 안 되는 이론적 이유다.

### 3.6 다른 meta-learning으로의 확장

저자들은 이 episode sampling 전략이 MAML에만 국한되지 않는다고 주장한다. 조건은 **bi-level optimization** 구조를 가져야 한다는 것이다.

MetaOptNet의 경우, 원래는 embedding network 위에 binary/multi-class SVM을 얹어 few-shot classification을 한다. 그러나 inner loop에 one-class 배치만 들어오므로, 일반 SVM 대신 **OC-SVM** 으로 바꾼다. 그리고 differentiable QP solver로 이를 풀어 representation network를 end-to-end로 학습한다.

Meta-SGD는 초기 파라미터뿐 아니라 parameter-specific learning rate도 meta-learn한다. 저자들은 OCC setting에서는 일부 learning rate가 음수가 되어 majority class 과적합을 상쇄하려는 현상이 나타났다고 보고하며, 이를 막기 위해 learning rate를 $0$과 $1$ 사이로 clipping했다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 설정

논문은 이미지와 시계열을 포함한 총 8개 데이터셋을 사용했다고 말한다. 본문과 부록에서 확인되는 주요 데이터셋은 다음과 같다.

이미지 도메인에서는 **MiniImageNet, Omniglot, CIFAR-FS, FC100, MT-MNIST** 를 사용했다. 이들을 OCC 문제로 바꾸기 위해, 각 task에서 하나의 클래스를 정상으로 두고 anomaly 클래스는 여러 다른 클래스에서 샘플링했다. 예를 들어 MiniImageNet에서는 각 원래 클래스를 정상 클래스 하나로 보고, 나머지 63개 클래스에서 anomaly를 만든다.

시계열 도메인에서는 저자들이 직접 만든 **STS-Sawtooth**, **STS-Sine** 두 synthetic benchmark와, 실제 산업 데이터인 **CNC Milling Machine Data (CNC-MMD)** 를 사용했다. STS 데이터는 task마다 서로 다른 파형, 주파수, 진폭, 노이즈 범위, anomaly 폭/높이를 갖도록 설계했다. CNC-MMD는 알루미늄 공작물 가공 중 수집된 3채널 센서 시계열이며, anomaly는 실제 제조 결함 시나리오를 인위적으로 유도해 전문가가 라벨링했다.

평가지표는 대부분의 데이터셋에서 **class-balanced test set accuracy** 를 사용했고, CNC-MMD처럼 테스트셋이 불균형한 실제 anomaly detection 설정에서는 **F1-score** 를 사용했다.

또 하나 중요한 실험 설정은 **non-transductive evaluation** 이다. 메타러닝 논문들에서는 종종 test batch 전체를 함께 넣고 batch normalization 통계를 공유하는 transductive setting을 쓰는데, 저자들은 anomaly detection에서는 실제 배치의 class ratio가 계속 바뀔 수 있으므로 부적절하다고 본다. 그래서 adaptation에 사용한 few-shot one-class 샘플만으로 BN 통계를 계산하고, 테스트는 개별적으로 처리하는 방식으로 평가했다.

### 4.2 고전적 OCC 방법과의 비교

Table 1의 결과에서 OC-SVM, Isolation Forest, 그리고 shallow embedding과 결합한 여러 OCC baseline은 대부분 **50% 부근의 chance-level 성능** 에 머문다. 이는 few-shot 상황에서 전통적인 OCC가 매우 취약하다는 저자들의 첫 번째 주장에 부합한다.

반면 OC-MAML은 매우 높은 정확도를 보인다. 예를 들어 Table 1에서:

- MiniImageNet에서는 $K=2$일 때 69.1%, $K=10$일 때 76.2%
- Omniglot에서는 96.6%, 97.6%
- MT-MNIST에서는 88.0%, 95.1%
- STS-Sawtooth에서는 96.6%, 95.7%

을 기록한다. 같은 표에서 일반 MAML도 나쁘지 않은 경우가 있으나, 거의 모든 설정에서 OC-MAML이 더 높다. 특히 STS-Sawtooth의 $K=2$에서 MAML은 81.1%인데 OC-MAML은 96.6%로 차이가 크다.

이 결과는 “few-shot one-class adaptation에 맞는 meta-initialization이 중요하다”는 논문의 메시지를 직접 뒷받침한다.

### 4.3 first-order와 second-order 비교

Table 1과 Table 5를 보면, **OC-FOMAML** 과 **OC-Reptile** 은 대체로 불안정하거나 성능이 낮다. 특히 OC-Reptile은 여러 설정에서 거의 50% 수준이다. 이는 저자들의 이론 분석, 즉 **first-order meta-learning은 OCC용 gradient alignment를 명시적으로 최적화하지 못한다** 는 주장과 일치한다.

반대로 OC-MAML은 batch normalization 사용 여부와 관계없이 비교적 안정적으로 높다. 논문은 BN이 이미지 데이터에서는 gradient orthogonalization 효과로 성능 향상에 도움을 줄 수 있다고 설명하지만, OC-MAML은 BN이 없어도 좋은 성능을 낸다고 주장한다. 즉, BN이 보조적으로 도움을 줄 수는 있어도 핵심은 episode sampling과 second-order meta-update라는 것이다.

### 4.4 gradient cosine similarity 분석

저자들은 이론을 더 직접적으로 검증하기 위해, 메타 학습된 초기화에서 **one-class batch gradient** 와 **class-balanced batch gradient** 의 cosine similarity를 측정한다. Table 3에 따르면 OC-MAML의 cosine similarity가 다른 방법보다 크게 높다.

예를 들어 STS-Sawtooth에서는:

- Reptile: 0.02
- FOMAML: -0.02
- MAML: 0.01
- OC-Reptile: 0.03
- OC-FOMAML: 0.07
- **OC-MAML: 0.92**

이다. 이 값은 매우 인상적이다. 즉, OC-MAML 초기화에서는 정상 클래스 몇 개로 계산한 gradient 방향이 class-balanced 데이터에서 좋은 방향과 거의 일치한다는 뜻이다. 논문 전체에서 가장 설득력 있는 실험 중 하나다.

### 4.5 실제 산업 데이터 검증

CNC-MMD 실험은 논문의 실용성을 보여주는 부분이다. 저자들은 roughing/finishing 같은 milling operation별로 task를 만들고, target operation과 같은 타입의 다른 operation들로 meta-training한다. 테스트 시에는 **정상 샘플 10개만** 사용한다.

Table 2에서 OC-MAML은 6개 공정에 대해 F1-score가 다음 범위로 보고된다.

- 80.0%
- 89.6%
- 95.9%
- 93.6%
- 85.3%
- 82.6%

이는 단순 accuracy보다 더 의미 있는 결과다. 실제 anomaly detection에서는 테스트셋이 불균형하기 때문에 F1-score가 더 적절하며, 정상 샘플 10개만으로 anomaly detection 모델을 세울 수 있다는 점은 실제 제조 적용 가능성을 강하게 보여준다.

또한 저자들은 이 데이터에서는 anomaly 수가 너무 적어서 일반 MAML처럼 inner loop에 anomaly 예시를 포함하는 표준 샘플링이 사실상 어렵다고 지적한다. OC-MAML은 anomaly를 outer loop validation에서만 사용하므로 이 제약을 우회할 수 있다.

### 4.6 다른 메타러닝 알고리즘으로의 확장과 SOTA 비교

Table 4에서 저자들은 OC-MAML 외에도 **OC-MetaOptNet** 과 **OC-MetaSGD** 를 제안하고, 기존 MetaOptNet, MetaSGD, One-Way ProtoNets와 비교한다.

예를 들어 MiniImageNet에서:

- MAML: 62.3 / 65.5
- **OC-MAML: 69.1 / 76.2**
- MetaSGD: 65.0 / 73.6
- **OC-MetaSGD: 69.6 / 75.8**
- One-Way ProtoNets: 67.0 / 74.4

이며, CIFAR-FS와 FC100에서도 비슷한 경향이 나타난다. 특히 OC-MAML과 OC-MetaSGD가 One-Way ProtoNets를 일관되게 앞선다고 보고한다.

저자들은 One-Way ProtoNets와 OC-MetaOptNet이 상대적으로 약한 이유로, **feature extractor를 테스트 task에 맞게 gradient-based로 적응시키는 메커니즘이 부족하기 때문** 이라고 해석한다. 반면 OC-MAML과 OC-MetaSGD는 few-shot 정상 샘플로 feature extractor까지 fine-tune할 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 방법이 잘 맞물려 있다는 점이다. FS-OCC는 few-shot learning과 anomaly detection의 교차 영역인데, 저자들은 단순 응용 수준이 아니라 **왜 기존 메타러닝이 그대로는 안 되는지** 를 이론과 실험으로 함께 보여준다. 특히 gradient inner product와 cosine similarity 관점의 분석은 논문의 핵심 기여를 매우 명확하게 만든다.

또 다른 강점은 방법이 단순하다는 점이다. 새로운 복잡한 네트워크 구조를 설계한 것이 아니라, **episode sampling 전략을 바꾸는 방식** 으로 문제를 해결한다. 덕분에 MAML뿐 아니라 MetaOptNet, Meta-SGD 같은 다른 bi-level meta-learning에도 적용 가능하다. 실험도 이미지, synthetic time-series, 실제 제조 센서 데이터까지 포함해 범위가 넓다.

실용성 측면에서도 설득력이 있다. 특히 CNC-MMD 실험은 “정상 샘플 몇 개만으로 새로운 공정 anomaly detector를 빠르게 만든다”는 시나리오와 정확히 맞닿아 있다. 이는 논문의 동기와 결과가 잘 연결되어 있음을 보여준다.

한편 한계도 분명하다. 첫째, 제안의 핵심 성질은 **second-order derivative** 에 의존하므로 계산 비용이 크다. 저자들도 부록에서 이를 완화하기 위해 OC-ANIL을 언급하지만, 본문의 핵심 결과는 여전히 second-order 기반이다.

둘째, 데이터셋 설계상 이미지 실험의 anomaly 클래스는 여러 다른 클래스에서 합쳐 만든 것이다. 이는 FS-OCC의 합리적인 벤치마크 설정이지만, 실제 anomaly가 “훈련 중 보지 못한 복잡한 분포 이동”을 보이는 경우와 완전히 동일하다고 보기는 어렵다. 논문도 이 간극을 완전히 해소하지는 않는다.

셋째, hyperparameter tuning과 batch normalization 처리 방식이 성능에 꽤 영향을 미친다. 저자들은 non-transductive BN 설정을 일관되게 사용했다고 설명하지만, 메타러닝 계열은 평가 프로토콜에 민감하므로 재현 시 세부 구현이 중요하다.

넷째, 논문은 future work로 **unsupervised FS-OCC** 를 제안한다. 이는 현재 방법이 메타-training 단계에서는 정상/비정상 라벨을 필요로 한다는 뜻이기도 하다. 즉, 테스트 task는 one-class여도 meta-training 자체는 완전히 비지도는 아니다.

비판적으로 보면, 논문의 주된 비교 대상 중 고전 OCC baseline이 few-shot setting에서 매우 약하게 나오는 것은 어느 정도 예상 가능한 결과다. 오히려 더 중요한 비교는 One-Way ProtoNets, MetaOptNet, MetaSGD 같은 few-shot/meta-learning 계열인데, 이 부분은 비교적 잘 수행되었다. 다만 모든 데이터셋에서 분산이나 통계적 유의성 검정이 제시되지는 않아, 결과 차이의 안정성을 더 엄밀히 보고 싶다는 아쉬움은 있다.

## 6. 결론

이 논문은 **few-shot one-class classification** 이라는 비교적 덜 탐구된 문제를 명확히 제시하고, 이를 해결하기 위해 **one-class inner loop + class-balanced outer loop** 라는 간단하지만 핵심적인 meta-training 전략을 제안한다. 그 결과물인 **OC-MAML** 은 정상 샘플 몇 개만으로도 새로운 OCC task에 빠르게 적응할 수 있는 초기화를 학습한다.

논문의 주요 기여는 세 가지로 정리할 수 있다. 첫째, 고전 OCC와 기존 few-shot meta-learning이 FS-OCC에서 왜 부족한지 보여주었다. 둘째, **second-order term이 필요한 이유** 를 gradient inner product 분석으로 설명했다. 셋째, 제안한 sampling 전략이 MAML뿐 아니라 다른 bi-level meta-learning으로도 확장 가능함을 실험적으로 보였다.

실제 적용 관점에서도 의미가 크다. 산업 센서 데이터처럼 anomaly가 희귀하고 데이터가 적은 환경에서, task별로 정상 예시 몇 개만으로 anomaly detector를 구성할 수 있다면 활용 가능성이 높다. 향후에는 더 큰 규모의 실제 데이터, 비지도 메타학습, 계산 효율 개선 같은 방향으로 확장될 여지가 크다. 전체적으로 이 논문은 FS-OCC를 하나의 독립적인 학습 문제로 정식화하고, 그에 맞는 meta-learning 원리를 제시했다는 점에서 가치가 있다.
