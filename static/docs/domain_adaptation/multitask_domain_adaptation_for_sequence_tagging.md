# Multi-task Domain Adaptation for Sequence Tagging

- **저자**: Nanyun Peng, Mark Dredze
- **발표연도**: 2017
- **arXiv**: https://arxiv.org/abs/1608.02689

## 1. 논문 개요

이 논문은 **domain adaptation**과 **multi-task learning (MTL)**을 하나의 신경망 프레임워크 안에서 동시에 다루는 방법을 제안한다. 전통적인 domain adaptation은 보통 하나의 task에 대해서만 source domain의 풍부한 학습 데이터를 target domain으로 옮겨 쓰는 문제로 정식화된다. 반면 이 논문은 “여러 task를 함께 학습하면 domain adaptation에 더 도움이 될 수 있는가?”라는 질문을 정면으로 다룬다.

연구 문제는 명확하다. 자연어처리에서는 news 같은 formal domain에는 annotation이 많지만, social media 같은 target domain에는 annotated data가 매우 적다. 이때 단일 task 기준으로만 domain adaptation을 수행하면, domain shift를 완화할 수는 있어도 representation을 충분히 강건하게 만들기 어렵다. 저자들은 서로 관련된 task를 함께 학습하면, 더 많은 감독 신호와 더 나은 inductive bias를 통해 domain-general representation을 더 잘 배울 수 있다고 본다.

이 문제는 실제적으로 중요하다. 예를 들어 Chinese word segmentation(CWS)이나 named entity recognition(NER) 같은 sequence tagging 문제는 뉴스에서는 잘 동작하지만 social media처럼 문체가 짧고 비정형적이며 어휘 변화가 큰 환경에서는 성능이 크게 떨어진다. 따라서 적은 social media annotation만으로도 높은 성능을 내는 domain adaptation 방법은 실제 응용 가치가 높다. 이 논문은 특히 **같은 task의 다른 domain 데이터뿐 아니라, 다른 task와 다른 domain의 데이터까지 함께 사용할 수 있는가**를 탐구하며, 이것을 하나의 통합된 구조로 제시한 점이 핵심이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **“task별 출력 구조는 분리하되, 입력 representation은 여러 task와 여러 domain에서 공동으로 학습한다”**는 것이다. 이를 위해 저자들은 모델을 세 부분으로 나눈다. 첫째, 모든 task와 domain이 공유하는 representation learner가 입력 문장을 contextual representation으로 변환한다. 둘째, domain마다 별도의 projection을 두어 domain-specific hidden representation을 공통 공간으로 보정한다. 셋째, task마다 하나의 decoder를 두어 해당 task의 label structure를 예측한다.

이 설계의 직관은 다음과 같다. domain adaptation의 핵심은 서로 다른 domain에서 온 입력을 가능한 한 공통된 feature space로 정렬하는 것이다. 동시에 multi-task learning의 핵심은 서로 다른 task가 공유할 수 있는 일반적 표현을 배우는 것이다. 이 논문은 이 둘을 분리해서 보지 않고, **shared representation + domain-specific alignment + task-specific decoding**이라는 구조로 묶는다. 그러면 source domain의 CWS 데이터, source domain의 NER 데이터, target domain의 CWS 데이터, target domain의 NER 데이터가 모두 representation 학습에 기여하게 된다.

기존 접근과의 차별점은, domain adaptation과 MTL을 단순히 병렬로 사용하는 수준이 아니라, **도메인과 태스크를 서로 직교하는 축으로 보고 하나의 네트워크 안에서 동시에 최적화한다는 점**이다. 특히 저자들은 “different domain + different task” 조합도 학습에 도움이 될 수 있다는 **domain/task mismatch setting**을 실험적으로 검토한다. 이는 기존 domain adaptation 문헌에서는 거의 다루지 않던 설정이다.

또 하나의 중요한 차별점은, 같은 task에 대해서 domain별로 서로 다른 decoder를 두는 대신, **task당 하나의 CRF decoder를 domain들 사이에서 공유**한다는 선택이다. 저자들은 domain projection이 각 domain의 representation을 공통 공간으로 보내면, task-specific predictor는 domain에 관계없이 하나로 두는 것이 더 domain adaptation의 본래 취지에 맞는다고 주장한다. 이는 단순히 모델 수를 줄이는 것이 아니라, 같은 task라면 결국 같은 예측 규칙을 공유해야 한다는 가정에 기반한다.

## 3. 상세 방법 설명

전체 구조는 sequence tagging용 **BiLSTM-CRF**를 기반으로 한다. 다만 일반적인 BiLSTM-CRF와 달리, BiLSTM 뒤와 CRF 앞 사이에 **domain projection layer**를 추가했다. 결과적으로 모델은 다음 세 층으로 이해할 수 있다.

첫 번째는 **shared representation learner**이다. 입력 문장 $x_{1:n}$이 들어오면 BiLSTM이 각 위치 $t$에 대한 contextual hidden vector $h_t$를 만든다.

$$
h_t = \mathrm{BiLSTM}(x_{1:n}, t)
$$

여기서 $x_{1:n}$은 길이 $n$인 전체 입력 sequence이고, $t$는 현재 위치이다. 양방향 LSTM을 쓰므로 최종 hidden vector는 정방향 hidden state와 역방향 hidden state의 concatenation이다. 즉, 각 token의 representation은 좌우 문맥을 모두 반영한다. 이 층은 모든 dataset, 모든 task, 모든 domain이 공유한다.

두 번째는 **domain projection layer**이다. 저자들은 BiLSTM 하나만으로 서로 다른 domain의 입력을 완전히 같은 공간으로 정렬하게 만드는 것은 부담이 크다고 본다. 왜냐하면 BiLSTM은 입력이 어느 domain에서 왔는지 명시적으로 알지 못한 채 representation을 만들어야 하기 때문이다. 그래서 domain identity를 활용하는 보정 단계를 따로 둔다. 이 논문에서는 두 가지 방법을 실험한다.

첫 번째 방법은 **domain mask**이다. 이는 Daumé III의 “frustratingly easy domain adaptation” 아이디어와 유사하게, hidden representation의 차원을 shared region과 domain-specific region으로 나누는 방식이다. 예를 들어 두 domain이 있고 hidden dimension이 $k$라면, 앞의 $k/3$ 차원은 shared, 다음 $k/3$ 차원은 domain 1 전용, 마지막 $k/3$ 차원은 domain 2 전용으로 해석한다. 그러면 각 domain에 대해 mask vector $m_d$를 정의할 수 있다. 예시로 두 domain에 대한 mask는 다음과 같다.

$$
m_1 = [1, 1, 0], \quad m_2 = [1, 0, 1]
$$

실제 적용은 element-wise multiplication으로 이루어진다.

$$
\hat{h} = m_d \odot h
$$

이 식의 의미는 간단하다. domain $d$의 예제에 대해서는 해당 domain의 전용 영역과 shared 영역만 활성화하고, 다른 domain의 전용 영역은 0으로 막는다. 그러면 BiLSTM은 어떤 feature를 shared region에 배치할지, 어떤 feature를 domain-specific region에 둘지를 학습 중에 자연스럽게 결정하게 된다. 기존의 hand-crafted feature mask와 달리, 여기서는 마스크가 hidden representation에 직접 적용되므로, feature weight뿐 아니라 representation learner의 파라미터까지 간접적으로 영향을 받는다는 점이 중요하다.

두 번째 방법은 **linear projection**이다. 각 domain에 대해 $k \times k$ 크기의 선형 변환 행렬 $T_d$를 두고, hidden vector를 공통 공간으로 사상한다.

$$
\hat{h} = T_d h
$$

이 방식은 domain mask보다 자유도가 높다. 차원을 미리 shared/domain-specific으로 나누지 않고, 학습 데이터에 맞추어 임의의 선형 변환을 배울 수 있기 때문이다. 반면 그만큼 올바른 정렬을 데이터로부터 스스로 배워야 하므로, 구조적 bias는 더 약하다. 논문은 두 방법을 모두 비교하며, task에 따라 어느 쪽이 더 유리한지 살핀다.

세 번째는 **task-specific neural-CRF model**이다. sequence tagging의 출력은 인접 label dependency가 중요하므로, 저자들은 decoder로 Conditional Random Field(CRF)를 사용한다. 각 task마다 하나의 CRF를 두고, 이 CRF는 여러 domain에 대해 공유된다. 논문에서의 조건부 확률은 다음과 같이 정의된다.

$$
p(y^k \mid x^k; W) = \frac{\prod_{i=1}^{n} \exp\left(W^T F(y^k_{i-1}, y^k_i, \psi(x^k))\right)}{Z^k}
$$

여기서 $F$는 feature function이고, $\psi(x^k)$는 입력 변환 함수인데 이 논문에서는 BiLSTM 출력이다. 즉 $\psi(x^k)=\mathrm{BiLSTM}(x^k)$이다. $Z^k$는 모든 가능한 label sequence에 대한 정규화 상수(partition function)이다.

$$
Z^k = \sum_{y \in \mathcal{Y}^n} \prod_{i=1}^{n} \exp\left(W^T F(y^k_{i-1}, y^k_i, \psi(x^k))\right)
$$

중요한 점은 CRF를 **task마다 하나만 둔다**는 것이다. 예를 들어 CWS는 news와 social media에서 같은 decoder를 공유하고, NER도 마찬가지다. 저자들은 이것이 domain adaptation의 정신에 더 부합한다고 본다. domain projection을 통해 입력 표현이 이미 공통 공간으로 정렬되었다면, 같은 task는 domain을 넘어 같은 예측기를 써야 한다는 주장이다.

학습은 end-to-end로 수행된다. 데이터셋 수가 $D \times T$개라면, 전체 loss는 각 dataset log-likelihood의 선형 결합이다. 논문에서는 단순화를 위해 각 dataset에 동일 가중치를 둔다. 최적화는 dataset 간 alternating SGD로 진행한다. 특정 dataset이 지나치게 크게 영향을 미치는 것을 막기 위해, 가장 작은 dataset 크기를 기준으로 각 epoch에서 instance 수를 $\lambda$ 배만큼 subsampling한다. 또한 learning rate는 dataset별로 따로 tuning하고, development 성능이 5 epoch 연속 개선되지 않으면 decay한다. 전체 학습은 최대 30 epoch, early stopping을 사용한다.

초기화 측면에서는 Chinese pre-trained embeddings 100차원을 사용하고, 나머지 파라미터는 $[-1, 1]$ 범위에서 uniform random initialization 한다. embedding dimension은 100, LSTM hidden dimension은 150으로 고정했다. dropout은 embedding과 BiLSTM output에 적용한다. 추론 시에는 CRF의 MAP inference를 사용하여 가장 확률이 높은 label sequence $y^* = \arg\max_y p(y \mid x; \Omega)$를 찾는다.

## 4. 실험 및 결과

실험은 두 task, 두 domain에서 수행된다. task는 **Chinese word segmentation (CWS)**와 **named entity recognition (NER)**이고, domain은 **news**와 **social media**이다. source domain은 news, target domain은 social media다. 따라서 총 네 개의 데이터셋을 사용한다. news CWS는 SIGHAN 2005 shared task의 simplified Chinese portion, news NER는 SIGHAN 2006 shared task의 simplified Chinese portion, social media CWS는 WeiboSeg, social media NER는 WeiboNER이다.

데이터 규모는 뉴스 쪽이 훨씬 크고 social media 쪽은 매우 작다. SighanCWS는 train 39,567 / dev 4,396 / test 4,278, SighanNER는 train 16,814 / dev 1,868 / test 4,636, WeiboCWS는 train 1,600 / dev 200 / test 200, WeiboNER는 train 1,350 / dev 270 / test 270이다. SIGHAN 데이터는 원래 dev split이 없어서 train의 마지막 10%를 dev로 사용했고, WeiboSeg는 원래 평가용 2,000문장이라 저자들이 8:1:1로 직접 나누었다. WeiboNER는 tag set 차이를 맞추기 위해 named mention만 사용하고, geo-political entity를 location과 병합했다. 이는 비교의 공정성을 위해 중요한 전처리이다.

baseline은 두 가지다. **Separate**는 target domain 데이터만으로 task별 모델을 따로 학습하는 standard supervised setting이다. **Mix**는 같은 task의 source와 target 데이터를 그냥 섞어서 학습하는 방식이다. 즉 Mix는 out-of-domain 데이터를 사용하지만 domain shift를 명시적으로 모델링하지 않는다.

실험의 핵심 결과는 social media target test set에서 보고된다. CWS 결과를 보면 Separate의 F1은 86.0, Mix는 86.5로 단순한 데이터 혼합도 약간의 이득이 있다. 그러나 single-task domain adaptation은 더 좋아진다. Domain Mask는 F1 87.9, Linear Projection은 87.7이다. 여기에 multi-task domain adaptation을 적용하면 더 향상된다. Multi-task DA + Domain Mask는 F1 89.0, Multi-task DA + Linear Projection은 88.9를 기록했다. 즉 CWS에서는 제안 방법이 baseline뿐 아니라 single-task adaptation보다도 더 좋다.

NER에서는 개선 폭이 더 크다. Separate의 F1은 48.5, Mix는 51.1이다. single-task domain adaptation은 Domain Mask 56.8, Linear Projection 56.4로 큰 폭의 개선을 보인다. multi-task domain adaptation에서는 Domain Mask가 59.9, Linear Projection이 57.5다. 특히 NER는 social media에서 더 어려운 task이기 때문에, 추가 task 정보와 domain-aware representation 학습의 이점이 더 크게 나타난 것으로 해석할 수 있다.

이 결과는 당시의 최고 성능도 넘어선다. 논문은 기존 최고 결과로 Zhang et al. (2013)의 CWS F1 87.5%, Peng and Dredze (2016)의 NER F1 55.3%를 언급하며, 제안 모델이 둘 다 갱신했다고 주장한다. 표의 수치 기준으로도 CWS 89.0, NER 59.9는 분명한 향상이다.

통계적 유의성 검정도 수행했다. 저자들은 McNemar's chi-square test를 사용했고, token 단위가 아니라 predicted span 단위로 정답과 일치하면 positive로 보았다. NER에서는 named entity span만 고려했다. 결과적으로 두 single-task domain adaptation 모델은 Mix baseline보다 $p < 0.01$ 수준에서 유의하게 좋았고, multi-task domain adaptation도 각각의 single-task counterpart보다 $p < 0.01$ 수준에서 유의하게 좋았다. 즉, 개선이 우연이 아니라는 점을 정량적으로 뒷받침한다.

추가 분석도 흥미롭다. 먼저 in-domain training data 크기를 바꾸어 보면, target domain 데이터가 늘어날수록 Separate baseline도 좋아지지만 점차 **diminishing returns**가 나타난다. 반면 domain adaptation과 multi-task domain adaptation은 더 완만하고 안정적인 향상을 보인다. 특히 NER에서는 적은 in-domain 데이터 상황에서 제안 방법의 장점이 더 크다. 논문은 in-domain 데이터가 전혀 없는 경우도 언급하는데, 이때는 unsupervised domain adaptation setting으로 볼 수 있으며, 제안 프레임워크도 그대로 적용 가능하다고 설명한다.

또한 저자들은 dataset 수를 달리한 변형 실험을 수행한다. 두 개 dataset만 쓰는 경우, 같은 domain의 다른 task를 같이 학습하는 **multi-task**, 같은 task의 다른 domain을 같이 쓰는 **domain adaptation**, 그리고 **different task + different domain** 조합인 **mismatch**를 비교했다. 표를 보면 추가 데이터는 task든 domain이든 대체로 도움이 되지만, 가장 좋은 것은 aligned task를 사용하는 domain adaptation이다. 다만 mismatch도 baseline보다 좋아서, 저자들의 주장처럼 “다른 task이면서 다른 domain인 데이터”도 representation 학습에 기여할 수 있음을 보여준다.

마지막으로, 성능 향상이 단순히 데이터가 많아져서 생긴 것인지 확인하기 위해 **All Multi-task** 설정도 비교했다. 이는 same task in different domains를 아예 다른 task처럼 취급하여, shared BiLSTM 위에 task-specific model을 네 개 두는 방식이다. 만약 성능 향상이 단순한 데이터량 효과라면 이런 방식도 비슷해야 한다. 그러나 결과는 Multi-task DA가 더 낫다. 이는 데이터만 더하는 것이 아니라, **domain을 명시적으로 모델링하는 설계 자체가 유의미하다**는 점을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 매우 자연스럽고 설계가 명료하다는 점이다. domain adaptation과 multi-task learning은 모두 “공유할 것과 분리할 것을 어떻게 나눌 것인가”의 문제인데, 저자들은 이를 **representation은 공유, domain 보정은 분리, task decoder는 task별 공유**라는 깔끔한 구조로 정리했다. 모델이 복잡한 수학적 장치에 의존하지 않으면서도, 구조적 가정이 명확해 해석 가능성이 높다.

또 다른 강점은 실험 설계가 단순한 benchmark 비교를 넘어선다는 점이다. baseline으로 Separate와 Mix를 두었고, single-task domain adaptation과 multi-task domain adaptation을 함께 비교했으며, 추가로 mismatch setting, dataset 수 변화, in-domain data size 변화까지 분석했다. 덕분에 “왜 좋아졌는가”에 대한 근거가 비교적 충실하다. 특히 All Multi-task 비교는 단순한 데이터 증가 효과와 구조적 설계 효과를 분리해서 보려는 시도라는 점에서 설득력이 있다.

실제 성능 향상도 충분히 크다. 특히 social media NER처럼 매우 어려운 low-resource target domain에서 큰 폭으로 오른 점은 제안 방법의 실용성을 잘 보여준다. 또한 당시 state of the art를 갱신했다는 점에서 empirical contribution도 분명하다.

반면 한계도 분명하다. 먼저 domain projection 방식이 **매우 단순한 두 가지 형태**에 머문다. domain mask는 해석이 쉽지만 차원 분할이 다소 인위적이고, linear projection은 유연하지만 non-linear mismatch를 충분히 다루기 어렵다. 저자들도 conclusion에서 더 정교한 domain adaptation 기법을 결합하는 방향을 향후 과제로 제시한다.

둘째, task의 종류가 모두 sequence tagging이며, 언어도 Chinese에 한정된다. 따라서 이 프레임워크가 parsing, classification, machine translation 같은 다른 구조의 문제에서도 동일하게 잘 작동하는지는 이 논문만으로는 알 수 없다. 저자들은 일반적 프레임워크라고 주장하지만, 실제 실험적 검증 범위는 제한적이다.

셋째, multi-task learning이 왜 domain adaptation에 도움을 주는지에 대한 분석은 주로 결과 수준에 머문다. 예를 들어 어떤 hidden dimension이 shared/domain-specific 역할을 했는지, domain projection이 어떤 식으로 representation을 정렬했는지에 대한 probing이나 시각화는 제공되지 않는다. 따라서 메커니즘에 대한 이해는 아직 제한적이다.

넷째, 데이터셋 간 annotation schema 차이를 맞추는 과정에서 일부 정보 손실이 있다. WeiboNER에서 nominal mention을 제외하고 entity type을 병합한 것은 실험상 필요하지만, 실제 social media NER의 풍부한 구조를 단순화한 것이다. 따라서 보고된 성능은 특정 정제된 설정에서의 결과라는 점을 염두에 둘 필요가 있다.

마지막으로, loss 결합 시 모든 dataset에 동일 가중치를 두는 선택도 단순하다. 데이터 규모 차이가 큰 상황에서 이 선택이 항상 최적인지는 불분명하다. 저자들은 subsampling coefficient $\lambda$와 dataset별 learning rate로 이를 보정하지만, 보다 원칙적인 weighting 전략이 있었으면 더 좋았을 수 있다.

종합하면, 이 논문은 새로운 방향을 여는 데 강점이 있지만, 아직은 “간단하고 효과적인 첫 번째 통합 모델”에 가깝다. 즉 개념적 통합과 실험적 유효성은 강하지만, 표현 정렬의 이론적 분석이나 더 넓은 범위의 일반화 검증은 후속 연구가 필요하다.

## 6. 결론

이 논문은 sequence tagging을 위한 **multi-task domain adaptation** 프레임워크를 제안했다. 핵심은 모든 데이터셋이 공유하는 BiLSTM representation learner, domain마다 하나씩 두는 projection layer, task마다 하나씩 두는 CRF decoder를 결합하는 것이다. 이를 통해 같은 task의 다른 domain뿐 아니라, 다른 task와 다른 domain의 데이터까지 representation 학습에 활용할 수 있도록 했다.

실험적으로는 Chinese word segmentation과 named entity recognition의 social media domain adaptation에서 강한 성능 향상을 보였고, 당시 state of the art를 달성했다. 특히 social media처럼 labeled data가 적고 domain shift가 큰 환경에서 multi-task 신호가 유용하다는 점을 잘 보여준다. 이는 실제 응용 측면에서 매우 의미가 있다.

향후 연구 관점에서도 이 논문은 가치가 크다. 이후의 많은 연구들이 domain adaptation과 transfer learning, shared-private representation, adversarial alignment, task-conditioned learning 등을 더 세련되게 발전시켰는데, 이 논문은 그러한 흐름의 초기 단계에서 **도메인과 태스크를 동시에 다루는 통합 관점**을 실험적으로 설득력 있게 제시했다는 점에서 중요하다. 따라서 이 연구의 핵심 메시지는 지금도 유효하다. 즉, low-resource target domain 문제를 풀 때는 같은 task의 source data만 볼 것이 아니라, 관련 task의 데이터까지 함께 고려해야 더 강건한 representation을 얻을 수 있다는 것이다.
