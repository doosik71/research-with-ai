# Task Discrepancy Maximization for Fine-grained Few-Shot Classification

- **저자**: SuBeen Lee, WonJun Moon, Jae-Pil Heo
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2207.01376

## 1. 논문 개요

이 논문은 **fine-grained few-shot classification** 문제를 다룬다. 즉, 서로 매우 비슷하게 생긴 세부 클래스들, 예를 들어 새의 종을 몇 장의 예시 이미지만 보고 구분해야 하는 상황을 대상으로 한다. 일반적인 few-shot classification도 데이터가 적기 때문에 어렵지만, fine-grained setting에서는 클래스 간 전체 외형이 매우 유사하므로 단순히 “객체가 어디 있는가”를 찾는 것만으로는 충분하지 않다. 눈, 부리, 날개 무늬처럼 **구별에 직접적으로 기여하는 세부 부위**를 찾아야 한다.

논문이 제기하는 핵심 문제는 기존 few-shot classification의 feature alignment 방법들이 대체로 “객체 관련 정보(object-relevant information)”를 잘 모으는 데는 도움을 주지만, fine-grained 분류에 필요한 **class-wise discriminative detail**, 즉 클래스별로 정말 구분력이 있는 채널을 충분히 강조하지 못한다는 점이다. 저자들은 특히 기존 metric-based 방법들이 feature map의 모든 채널을 거의 동등하게 다루는 경향이 있다고 본다. 그러나 fine-grained 데이터에서는 어떤 채널은 클래스 내 변동이 작더라도 여러 클래스가 공통으로 가지는 정보만 담고 있어 실제 분류에는 큰 도움이 되지 않을 수 있다.

이 문제를 해결하기 위해 논문은 **Task Discrepancy Maximization (TDM)** 이라는 모듈을 제안한다. TDM은 에피소드마다, 그리고 클래스마다 서로 다른 **channel weight**를 계산하여 feature map을 재가중한다. 이를 통해 해당 task에서 구분에 유리한 부분은 더 강하게, 공통적이거나 덜 중요한 부분은 더 약하게 반영되도록 만든다. 논문의 메시지는 분명하다. fine-grained few-shot classification에서는 “객체를 보는 것”보다 “어떤 세부 특징이 이 에피소드에서 구분에 중요한가”를 동적으로 찾아내는 것이 더 중요하다는 것이다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 직관적이다. 같은 에피소드 안에 등장한 클래스들을 비교해 보면, 어떤 채널은 특정 클래스의 구별되는 부위를 잘 나타내고, 어떤 채널은 여러 클래스가 공통으로 갖는 일반적 형태만 반영한다. 그렇다면 모든 채널을 똑같이 쓰지 말고, **현재 에피소드와 현재 클래스에 맞게 채널 중요도를 동적으로 조정**해야 한다는 것이다.

이를 위해 저자들은 두 가지 상보적 정보를 결합한다. 첫째는 **Support Attention Module (SAM)** 으로, support set의 라벨 정보를 이용해 각 클래스에 대해 어떤 채널이 더 discriminative한지 추정한다. 둘째는 **Query Attention Module (QAM)** 으로, query 이미지 자체에서 어떤 채널이 객체와 관련 있는지를 추정한다. SAM만 쓰면 적은 수의 support 예시에 과적합되거나 편향될 수 있고, QAM만 쓰면 객체 관련 채널은 찾을 수 있어도 클래스 구분력 자체는 약할 수 있다. 그래서 논문은 두 모듈을 결합해 최종적인 **task-specific weight**를 만든다.

기존 접근과의 차별점은, 많은 기존 정렬 방법이 support-query 쌍의 관계나 공간적 정렬에 집중하는 반면, TDM은 **에피소드 전체(task)** 를 보고 채널 수준에서 class-wise 중요도를 정의한다는 점이다. 또한 객체 전체를 잘 표현하는 채널이 아니라, 그중에서도 **현재 비교되는 클래스들 사이에서 실제로 차이를 만드는 채널**을 더 강조하려 한다는 점이 fine-grained 설정에 특화된 요소다.

## 3. 상세 방법 설명

논문은 few-shot episode를 $N$-way $K$-shot 설정으로 정의한다. support set은 라벨이 있는 이미지들, query set은 예측해야 하는 이미지들로 구성된다. 각 이미지는 backbone $g_\theta$를 통해 feature map으로 변환된다.

$$
F^S_{i,j} = g_\theta(x^S_{i,j}), \quad F^Q = g_\theta(x^Q)
$$

여기서 $F \in \mathbb{R}^{C \times H \times W}$ 이며, $C$는 채널 수, $H,W$는 spatial 크기다. 각 클래스 $i$에 대해 support feature들의 평균으로 prototype을 만든다.

$$
F^P_i = \frac{1}{K}\sum_{j=1}^{K} F^S_{i,j}
$$

이후 논문은 각 클래스 prototype의 채널별 표현력을 평가하기 위해 두 종류의 점수를 정의한다.

첫 번째는 **intra score**다. 먼저 클래스 $i$의 prototype에서 채널 평균으로 얻은 mean spatial feature $M^P_i$를 만든다.

$$
M^P_i = \frac{1}{C}\sum_{j=1}^{C} f^P_{i,j}
$$

여기서 $f^P_{i,c} \in \mathbb{R}^{H \times W}$ 는 prototype의 $c$번째 채널이다. 그 다음, 각 채널이 이 평균 spatial feature와 얼마나 비슷한지를 본다.

$$
R^{\text{intra}}_{i,c} = \frac{1}{H \times W}\|f^P_{i,c} - M^P_i\|_2
$$

논문의 설명에 따르면 이 점수는 채널이 클래스 내부에서 salient object region과 얼마나 잘 맞는지를 나타낸다. 직관적으로, 채널이 클래스의 핵심 물체 부위와 잘 대응하면 좋은 표현이라고 본다.

두 번째는 **inter score**다. 이것은 클래스 $i$의 채널이 다른 클래스들의 mean spatial feature와 얼마나 다른지를 본다.

$$
R^{\text{inter}}_{i,c} = \frac{1}{H \times W}\min_{1 \le j \le N, j \ne i}\|f^P_{i,c} - M^P_j\|_2
$$

이 값은 해당 채널이 다른 클래스와 얼마나 구별되는 정보를 담는지를 표현한다. 논문은 직관적으로 **작은 intra score**와 **큰 inter score**가 discriminative channel의 특징이라고 설명한다.

### Support Attention Module (SAM)

SAM은 support set만 이용해 클래스별 채널 중요도를 계산한다. 각 클래스에 대해 $R^{\text{intra}}_i$ 와 $R^{\text{inter}}_i$ 를 fully connected block에 통과시켜 두 개의 weight를 만든다.

$$
w^{\text{intra}}_i = b_{\text{intra}}(R^{\text{intra}}_i), \quad
w^{\text{inter}}_i = b_{\text{inter}}(R^{\text{inter}}_i)
$$

그 다음 이 둘을 선형 결합해 support weight를 만든다.

$$
w^S_i = \alpha w^{\text{intra}}_i + (1-\alpha)w^{\text{inter}}_i, \quad \alpha \in [0,1]
$$

이 $w^S_i$ 는 클래스 $i$에 대해 구분력이 큰 채널을 강조하고, 여러 클래스가 공통으로 공유하는 채널은 상대적으로 억제하는 역할을 한다.

### Query Attention Module (QAM)

QAM은 query 이미지에서 객체 관련 채널을 찾는다. query에는 라벨이 없으므로 클래스 간 비교는 하지 않고, query 내부 채널과 query의 mean spatial feature의 관계만 사용한다. query feature map $F^Q$ 에 대해 mean spatial feature $M^Q$를 만들고, 각 채널의 intra score를 계산한다.

$$
R^{\text{intra}}_Q = \frac{1}{H \times W}\|f^Q_c - M^Q\|_2
$$

이를 fully connected block에 넣어 query weight를 만든다.

$$
w^Q = b_Q(R^{\text{intra}}_Q)
$$

논문은 average pooling으로 만든 mean spatial feature가 객체 위치를 더 잘 반영한다고 주장하며, max pooling보다 더 안정적이라고 본다. 즉, QAM은 “이 query에서 실제 물체와 관련 있는 채널”에 더 높은 가중치를 부여한다.

### 최종 TDM과 feature 재가중

SAM과 QAM은 목적이 다르지만 상보적이므로, 최종 task weight는 둘을 다시 선형 결합해서 만든다.

$$
w^T_i = \beta w^S_i + (1-\beta)w^Q, \quad \beta \in [0,1]
$$

이제 각 클래스 $i$에 대해 원래 feature map $F$를 채널별로 스케일링해서 task-adaptive feature map $A$를 만든다.

$$
A = w^T_i \odot F
$$

구체적으로 support와 query는 다음처럼 바뀐다.

$$
A^S_{i,j} = w^T_i \odot F^S_{i,j}
$$

$$
A^Q_i = w^T_i \odot F^Q
$$

query의 실제 라벨은 모르기 때문에, 클래스 $i$에 대해 query를 평가할 때는 “클래스 $i$용 task weight”를 query에도 적용한다. 즉 query 하나에 대해 클래스마다 다른 적응 feature를 만든다는 뜻이다.

논문은 ProtoNet에 TDM을 붙였을 때의 추론식을 예시로 보여준다.

$$
p_\theta(y=i|x)=\frac{\exp(-d(A^S_i,A^Q_i))}{\sum_{j=1}^{N}\exp(-d(A^S_j,A^Q_j))}
$$

여기서 $d$는 거리 함수이며, $A^S_i$는 adaptive support feature들로 만든 prototype이다. 중요한 점은 distance를 계산하기 전에 feature 자체를 현재 task에 맞게 조정한다는 것이다.

### Fully Connected Block 구조

세 attention block은 모두 동일한 형태를 쓴다. 입력 크기 $B \times C$에서 시작해, `FC -> BatchNorm -> ReLU -> FC -> 1 + Tanh` 순으로 처리한다. 출력은 다시 $B \times C$이다. 마지막의 `1 + Tanh`는 가중치가 완전히 0 근처로 붕괴하지 않고 일정 범위에서 조정되도록 하려는 설계로 읽힌다. 다만 논문은 이 선택의 추가적인 수학적 이유를 길게 설명하지는 않는다.

## 4. 실험 및 결과

논문은 CUB-200-2011, Aircraft, meta-iNat, tiered meta-iNat, Stanford-Cars, Stanford-Dogs, Oxford-Pets의 총 7개 fine-grained benchmark에서 실험했다. backbone은 주로 `Conv-4`와 `ResNet-12`를 사용했고, 비교 대상은 ProtoNet, DSN, CTX, FRN 같은 기존 few-shot classification 방법들이다. 평가 방식은 10,000개의 랜덤 episode에서 class당 16개의 query를 사용해 평균 정확도와 95% confidence interval을 보고한다.

CUB 결과를 보면 TDM은 거의 모든 baseline 위에 일관되게 성능 향상을 준다. 특히 bounding-box cropped CUB에서 Conv-4 backbone을 쓴 ProtoNet은 1-shot에서 `62.90% -> 69.94%`로 크게 향상되었다. 이는 7%포인트 이상의 개선으로, 저자들이 강조하는 “fine-grained few-shot에서는 discriminative channel localization이 중요하다”는 주장을 강하게 뒷받침한다. ResNet-12에서도 향상 폭은 더 작지만 일관적이며, FRN+TDM은 cropped CUB에서 `83.36% / 92.80%`, raw CUB에서 `84.36% / 93.37%`를 기록했다.

Aircraft에서도 개선은 안정적이다. 예를 들어 Conv-4 기반 CTX는 `51.58% -> 55.15%`(1-shot), `68.12% -> 70.45%`(5-shot)로 상승했고, ResNet-12 기반 CTX는 `65.53% -> 69.42%`, `79.31% -> 83.25%`로 더 큰 폭의 향상을 보였다. 이 데이터셋은 항공기 종류보다 항공사 로고나 외형 유사성이 섞여 있어 fine-grained 판별이 어렵다는 점에서, TDM의 class-specific weighting이 효과적이라는 해석이 가능하다.

meta-iNat과 tiered meta-iNat 결과는 일반화 성능을 보여주기 위한 실험이다. meta-iNat은 클래스 수가 많고 불균형하며 validation set이 없어 과적합에 취약한데, TDM은 ProtoNet 기준 `55.37% -> 61.82%`(1-shot), `76.30% -> 79.95%`(5-shot)로 큰 개선을 보였다. tiered meta-iNat은 train/test의 super-category가 달라 domain gap이 큰데도, 대체로 성능을 올렸다. 다만 FRN의 5-shot에서는 `63.45% -> 62.91%`로 소폭 감소했다. 논문은 이것이 FRN 내부의 학습 파라미터 $\lambda$와 TDM의 상호작용 때문일 가능성이 있다고 설명한다. 즉, TDM이 discriminative channel에 이미 집중하게 하므로 FRN이 원래 필요로 하던 큰 $\lambda$의 역할이 줄어들 수 있다는 해석이다. 이는 논문이 제시한 설명이지만, 추가 실험 없이 완전히 입증되었다고 보기는 어렵다.

추가로 Stanford Cars, Stanford Dogs, Oxford Pets에서도 Conv-4 기반 baseline들에 TDM을 붙이면 모두 confidence interval이 겹치지 않을 정도의 개선을 보였다고 보고한다. 본문에는 모든 수치가 표로 제공되지는 않고 Figure 7로 제시되어 있다. 따라서 각 방법별 정확한 숫자를 이 텍스트만으로 완전히 복원할 수는 없지만, 저자들의 요지는 세 데이터셋 전체에서 일관된 개선이 있었다는 점이다.

Ablation study도 논문의 핵심 주장을 잘 지지한다. 먼저 SAM과 QAM을 분리해 보면, 둘 다 baseline보다 낫고 둘을 함께 쓰면 가장 좋다. 예를 들어 CUB cropped에서 ProtoNet baseline은 `62.90`, SAM만 쓰면 `68.53`, QAM만 쓰면 `65.11`, 둘 다 쓰면 `69.94`다. 이는 클래스별 discriminative channel을 찾는 SAM이 특히 중요하고, query의 object relevance를 보는 QAM이 이를 보완한다는 설명과 잘 맞는다.

Pooling 함수 비교에서는 average pooling이 max pooling보다 약간 더 좋거나 비슷한 성능을 보였다. CUB cropped에서 max pooling은 `67.23`, average pooling은 `69.94`였다. 저자들은 average pooling이 객체 위치를 더 안정적으로 포착하고, max pooling은 노이즈에 더 취약하다고 해석한다. 또한 cosine distance를 사용한 경우에도 TDM이 baseline보다 성능을 높였기 때문에, 특정 metric에만 종속되지 않는다는 점도 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 fine-grained few-shot classification의 문제를 매우 정확히 짚고, 그에 맞는 간단하면서도 효과적인 모듈을 설계했다는 점이다. 기존 많은 few-shot 방법이 object-level alignment에 머물렀다면, 이 논문은 fine-grained setting에서는 “어떤 채널이 클래스 구별에 실질적으로 기여하느냐”가 더 중요하다고 주장하고, 이를 support-query 양쪽 정보로 구현했다. 수식도 비교적 단순하고, ProtoNet, DSN, CTX, FRN 같은 기존 metric-based 방법 위에 쉽게 붙일 수 있다는 점에서 실용성이 높다. 실제로 여러 benchmark와 여러 backbone에서 성능 향상이 반복적으로 관찰되었다는 점도 설득력이 있다.

또 다른 강점은 SAM과 QAM의 역할 분담이 명확하다는 점이다. SAM은 class discriminativeness를, QAM은 query object relevance를 담당하며, 둘의 결합이 더 강하다는 ablation 결과가 제시되어 있다. 즉, 단순히 모듈을 덧붙인 것이 아니라, 왜 두 요소가 필요한지에 대한 구조적 논리가 있다.

반면 한계도 분명하다. 논문 스스로 인정하듯, TDM은 **fine-grained classification에 특화된 설계**이므로 coarse-grained task에서는 이득이 제한적일 수 있다. 다시 말해 “객체 전체를 폭넓게 표현하는 것이 더 유리한 문제”에서는 discriminative detail 중심의 weighting이 오히려 덜 효과적일 수 있다. 또한 query 라벨이 없기 때문에 query weight는 본질적으로 object relevance만 반영하고 class discriminativeness는 직접 반영하지 못한다. 이 한계는 SAM과의 결합으로 완화되지만 완전히 해소되지는 않는다.

비판적으로 보면, 핵심 representativeness score의 정의는 직관적이지만, 왜 $L_2$ 거리 기반의 mean spatial similarity가 가장 적절한지는 충분히 깊게 분석되지는 않는다. 또한 FRN의 tiered meta-iNat 5-shot 성능 저하에 대한 설명은 plausible하지만, 본문에 제시된 실험만으로 인과가 완전히 검증되었다고 보기는 어렵다. 더 나아가 Figure 7의 일부 추가 데이터셋 결과는 정확한 수치 표 대신 그래프로만 제시되어 있어, 세부 비교를 정밀하게 검증하려면 원문 figure를 직접 확인할 필요가 있다. 제공된 추출 텍스트만으로는 그 수치를 모두 정확히 읽어낼 수 없다.

## 6. 결론

이 논문은 fine-grained few-shot classification에서 중요한 것이 단순한 feature alignment가 아니라, **현재 에피소드에서 현재 클래스들을 구분하는 채널을 동적으로 강조하는 것**이라고 보고, 이를 위한 TDM을 제안했다. TDM은 support 기반의 SAM과 query 기반의 QAM을 결합해 class-wise task-specific channel weight를 만들고, 이를 통해 feature map을 task-adaptive하게 변환한다.

실험 결과는 이 아이디어가 단순한 직관 수준이 아니라 실제로 효과적임을 보여준다. 다양한 fine-grained benchmark와 여러 기존 few-shot 방법에 대해 일관된 성능 향상을 보였고, 특히 1-shot 상황에서 큰 개선이 자주 나타났다. 따라서 이 연구는 fine-grained few-shot classification에서 **채널 수준의 task-adaptive weighting**이 중요한 설계 축이 될 수 있음을 제시한 논문으로 볼 수 있다. 향후에는 논문이 제안한 방향처럼 다른 computer vision task에서도 “에피소드 혹은 task에 따라 채널 중요도가 어떻게 달라지는가”를 탐구하는 연구로 확장될 가능성이 있다.
