# Continual State Representation Learning for Reinforcement Learning using Generative Replay

**저자**: Hugo Caselles-Dupré, Michael Garcia-Ortiz,
David Filliat

**연도**: 2018

**게재 정보**:
NeurIPS 2018 Workshop on Continual Learning

**arXiv ID**:
[1810.03880v3](https://arxiv.org/abs/1810.03880v3)

---

## 1. 연구 배경 및 문제 정의

이 논문은 일반적인 분류형 continual learning이 아니라
**reinforcement learning을 위한 state representation learning**
문제를 연속 학습 관점에서 다룬다. 저자들의 출발점은 단순하다.
현실 환경에서 오래 동작하는 RL 에이전트는 정책만 학습하면
충분하지 않고, 자신이 보는 감각 입력을 압축하는 표현 모델도
환경 변화에 맞춰 계속 갱신해야 한다.

문제는 이런 state representation model이 보통 신경망이고,
데이터 분포가 바뀌면 과거 환경의 정보를 쉽게 잊어버린다는 점이다.
즉, 표현 학습 단계에서 **catastrophic forgetting**이 일어나면
그 위에서 학습되는 정책도 영향을 받는다.

논문은 이 문제를 다음 두 단계 문제로 정리한다.

1. 관측을 유용한 latent feature로 압축하는
   state representation model을 학습한다.
2. 환경이 순차적으로 바뀌어도 과거 환경 표현을 잃지 않도록
   그 모델을 continual하게 업데이트한다.

이를 위해 저자들은 VAE를 representation model로 사용하고,
이전 환경 데이터를 직접 저장하지 않는 대신
**Generative Replay**를 적용한다. 여기에 더해,
실제 자율 에이전트를 염두에 두고 환경 변화가 발생했는지
자동으로 감지하는 통계적 검정까지 제안한다.

## 2. 핵심 기여

1. **SRL과 Continual Learning의 결합**:
   VAE 기반 state representation learning에
   generative replay를 적용해,
   RL 이전 단계의 표현 학습 자체를 continual하게 만든다.
2. **환경 변화 자동 감지**:
   VAE reconstruction error 분포에 대해
   Welch's t-test를 적용해 환경 전환을 통계적으로 탐지한다.
3. **과거 데이터 비저장 조건 충족**:
   과거 raw data를 재사용하지 않고도
   이전 환경의 표상을 유지하는 bounded-size 시스템을 제시한다.
4. **RL 성능까지 평가**:
   단순 재구성 성능이 아니라, 학습된 feature가 PPO 정책 학습에
   실제로 도움이 되는지까지 검증한다.
5. **forward transfer 관찰**:
   두 환경을 함께 반영한 representation이
   새 환경 정책 학습 성능을 오히려 높일 수 있음을 보인다.

## 3. 방법론 상세

### 3.1 전체 아이디어

방법은 비교적 간단하다. 환경 1에서 VAE를 학습해
관측을 latent vector로 압축한다. 이후 환경 2로 이동하면
환경 1의 데이터를 저장해 두지 않고,
환경 1에서 학습된 VAE의 latent space에서 샘플을 생성한다.
이 생성 샘플과 환경 2에서 새로 수집한 데이터를 합쳐
새 VAE를 학습한다.

즉, 핵심은 다음과 같다.

1. 현재 환경 관측으로 VAE를 학습한다.
2. 환경 변화가 감지되면 이전 VAE에서 생성 샘플을 만든다.
3. 생성 샘플과 새 환경 데이터를 결합해 새 VAE를 학습한다.
4. 이 VAE encoder의 latent feature를 PPO 입력으로 사용한다.

이 접근은 replay buffer에 원본 데이터를 저장하지 않으면서도
과거 환경 분포를 근사적으로 재현한다는 점이 장점이다.

### 3.2 VAE를 이용한 상태 표현 학습

논문은 VAE를 사용해 관측 $x$를 latent representation $z$로
매핑한다. VAE 손실은 일반적으로 재구성 오차와
posterior 정규화 항으로 구성된다.

직관적으로는 다음 목적을 최적화한다고 보면 된다.

$$
\mathcal{L}\_{\mathrm{VAE}} =
\mathbb{E}\_{q_{\phi}(z|x)}[-\log p_{\theta}(x|z)] +
\mathrm{KL}(q_{\phi}(z|x)\|p(z))
$$

논문은 이 수식을 이론적으로 확장하기보다,
VAE가 RL용 state representation으로 충분히 잘 동작한다는
실용적 가정 위에서 continual adaptation 문제에 집중한다.

### 3.3 Generative Replay

환경이 바뀌면 이전 환경 데이터는 더 이상 접근하지 않는다.
대신 이전 VAE의 latent space에서 샘플을 뽑아
이전 환경에 해당하는 관측을 생성한다.
그다음 이 생성 샘플을 새 환경에서 수집한 관측과 합쳐
joint training을 수행한다.

논문 Appendix A.3에 따르면, 환경 2 학습 시
랜덤 정책으로 수집한 500,000개의 새 상태에 더해
이전 VAE에서 **동일한 수의 생성 샘플 500,000개**를 추가한다.

이 설계의 의미는 명확하다.

- 과거 데이터를 저장하지 않는다.
- 모델 크기는 VAE 하나로 고정된다.
- replay 메모리 대신 생성 모델이 과거 분포를 압축 저장한다.

### 3.4 자동 환경 변화 감지

저자들은 환경이 언제 바뀌는지를 사용자가 직접 알려주지 않아도
되도록 자동 변화 감지 절차를 넣었다.

핵심 아이디어는 VAE reconstruction error 분포를 비교하는 것이다.
환경이 충분히 달라져서 현재 VAE가 더 이상 잘 설명하지 못하면,
재구성 오차 분포의 평균이 달라질 것이라고 본다.

이를 위해 두 reconstruction error 샘플의 평균이 같은지
Welch's t-test로 검정한다. 논문은 등분산 가정을 할 이유가 없어서
일반 t-test 대신 Welch's t-test를 사용했다고 설명한다.

본문 Equation (1)은 다음 통계량을 쓴다.

$$
t =
\frac{\bar{x}_1 - \bar{x}_2}
{\sqrt{(s_1^2 + s_2^2)/N}}
$$

여기서 $\bar{x}_1, \bar{x}_2$는 두 reconstruction error 표본의 평균,
$s_1, s_2$는 표본 표준편차, $N$은 표본 수다.

저자들이 reconstruction error 분포를 직접 쓰는 이유도 중요하다.
단순한 state distribution 변화가 아니라,
**현재 VAE를 업데이트해야 할 만큼 의미 있는 변화인지**
판단하려는 것이다. 예를 들어 이미 잘 표현 가능한 장애물이
조금 더 많이 등장하면 관측 분포는 바뀔 수 있지만,
표현 모델을 다시 배울 필요는 없을 수 있다.

### 3.5 구현 세부사항

Appendix A.3 기준 주요 설정은 다음과 같다.

- 입력: 크기 `(64, 3)`의 1-D 이미지
- latent size: 64
- encoder / decoder:
  3개의 1-D convolution layer와 1개의 fully connected layer
- activation: ReLU
- batch size: 128

흥미로운 부분은 KL annealing이다.
기존 방식은 KL 항의 가중치를 0에서 1로 올리지만,
저자들은 그것이 제대로 재구성되지 않는다고 보고
**inverse annealing**에 가까운 방식을 쓴다.

1. 초기에는 KL 가중치를 1로 둔다.
2. 학습이 진행되면서 이를 0으로 천천히 낮춘다.
3. reconstruction error가 더 이상 충분히 개선되지 않으면
   early stopping으로 종료한다.

annealing parameter는 0.9995이고,
early stopping 기준은 reconstruction error 개선폭 0.001,
5 epoch 동안 개선이 없을 때다.

## 4. 실험 설정

### 4.1 환경

실험은 Flatland 기반 2-D first-person 환경에서 수행된다.
두 환경은 방 크기, 고정 장애물 3개, 랜덤 배치 원형 장애물 10개,
랜덤 배치 edible item 10개를 공유한다.
주요 차이는 edible item의 **색상**이다.

에이전트는 500 timestep 동안 가능한 많은 edible item을
먹는 것이 목표다. 입력은 에이전트 전방 시야에 해당하는
1-D 이미지이며, 행동은 전진, 좌회전, 우회전이다.

이 설정은 일부러 과도하게 복잡하지 않다.
논문의 목적이 고난도 RL 성능 경쟁이 아니라,
환경 변화가 representation learning과 policy learning에
미치는 영향을 분리해 관찰하는 데 있기 때문이다.

### 4.2 비교 방법

논문은 다음 방법을 비교한다.

- **Raw pixels**:
  표현 압축 없이 원시 입력을 PPO에 직접 넣는 기준선
- **VAE trained on source**:
  환경 1에서만 학습한 VAE feature 사용
- **VAE fine-tuning**:
  환경 1에서 학습한 VAE를 환경 2에서 순진하게 미세조정
- **VAE generative replay**:
  제안 방법

RL 평가는 PPO를 사용하며, policy는 ReLU activation을 쓰는
MLP다.

## 5. 실험 결과

### 5.1 환경 변화 감지 성능

논문은 reconstruction error 배치를 비교하는
Welch's t-test를 5000회 반복 평가했다.

- 환경 변화가 있어야 할 경우:
  **100% 성공적으로 변화 감지**
- 환경 변화가 없어야 할 경우:
  **99.5% 성공적으로 변화 없음 판단**

기준 p-value는 0.01이며,
0.05에서 0.0001 사이의 다른 값에서도 유사한 결과가
나왔다고 보고한다.

이 결과는 제안한 변화 감지기가 단순한 threshold tuning 없이도
상당히 안정적으로 동작함을 보여준다.

### 5.2 재구성 성능과 망각 방지

논문 Table 1의 reconstruction MSE는 다음과 같다.

- Fine-tuning:
  environment 1에서 $1.3 \times 10^{-3}$,
  environment 2에서 $9.3 \times 10^{-4}$
- Generative Replay:
  environment 1에서 $3.3 \times 10^{-4}$,
  environment 2에서 $6.4 \times 10^{-4}$

핵심 해석은 명확하다.

1. environment 2에 대해서는 두 방법이 모두 괜찮다.
2. 하지만 environment 1에 대해서는 fine-tuning이 크게 망각한다.
3. generative replay는 environment 1 MSE를
   약 한 자릿수 이상 더 낮게 유지한다.

정성적 시각화에서도 같은 결론이 나온다.
naive fine-tuning은 첫 번째 환경의 요소를 제대로 재구성하지
못하지만, generative replay는 두 환경의 요소를 모두
유지한다.

### 5.3 RL 최종 성능

Appendix Table 2의 PPO 최종 성능은 다음과 같다.

- Raw pixels:
  Task 1에서 `92.30 ± 5.8`,
  Task 2에서 `123.95 ± 25.6`
- VAE trained on source:
  `121.25 ± 5.3`, `111.75 ± 11.9`
- VAE fine-tuning:
  `96.55 ± 5.1`, `172.5 ± 11.5`
- **VAE generative replay**:
  `112.85 ± 13.2`, `256.95 ± 10.3`

여기서 중요한 포인트는 두 가지다.

1. state representation을 쓰는 것이 raw pixel보다
   sample efficiency와 최종 성능 면에서 대체로 유리하다.
2. generative replay representation은 특히 Task 2에서
   압도적으로 높은 성능을 보인다.

### 5.4 Zero-shot transfer와 Forward transfer

저자들은 이 결과를 단순한 forgetting 방지 이상으로 해석한다.

- 환경 1에서 학습한 VAE feature만으로도
  Task 2 학습이 꽤 잘 된다.
  이는 **zero-shot transfer** 성격을 시사한다.
- 두 환경을 함께 반영한 generative replay VAE는
  Task 2에서 다른 방법보다 훨씬 높은 성능을 낸다.
  이는 representation 수준의 **forward transfer**로 해석된다.

즉 제안 방법은 과거를 잊지 않는 것에서 끝나지 않고,
이전 환경 학습이 다음 환경 적응을 돕는 효과도 보인다.

## 6. 강점

### 6.1 문제 설정이 실용적이다

이 논문은 정책 자체가 아니라
**representation model의 continual learning**을 전면에 둔다.
이는 실제 embodied agent나 robotics에서 매우 현실적인 문제다.
정책보다 앞단의 perceptual representation이 깨지면
그 위의 제어도 함께 무너질 수 있기 때문이다.

### 6.2 과거 데이터 비저장이라는 제약을 잘 만족한다

원시 과거 데이터를 저장하지 않으면서도
bounded system size를 유지한다. 이는 프라이버시,
메모리, 장기 운용 관점에서 상당히 의미 있는 설계다.

### 6.3 RL 평가까지 연결했다

많은 representation learning 논문은 reconstruction이나
latent disentanglement까지만 본다. 이 논문은 실제 PPO 성능까지
평가해 representation quality가 제어 성능으로 이어지는지를
확인했다는 점이 좋다.

### 6.4 자동 변화 감지기가 단순하면서 설득력 있다

Welch's t-test 기반 변화 감지는 해석 가능하고 계산량이 작다.
복잡한 drift detector 없이도 "표현 모델을 갱신해야 하는가"를
판단하는 실용적 기준을 제공한다.

## 7. 한계와 비판적 해석

### 7.1 환경 변화가 비교적 단순하다

논문에서 두 환경의 핵심 차이는 edible item 색상 변화다.
이는 continual representation learning의 첫 실험으로는 적절하지만,
복잡한 구조 변화나 동역학 변화까지 일반화된다고 보기는 어렵다.

### 7.2 생성 품질에 대한 의존성이 크다

Generative Replay는 결국 생성된 샘플 품질에 달려 있다.
VAE가 과거 환경을 충분히 정확히 생성하지 못하면
replay 자체가 편향될 수 있다.
즉 forgetting 방지 성능은 generative model capacity와
직결된다.

### 7.3 RL 성능 향상의 원인 분리가 완전하지 않다

Task 2가 색상 채널 측면에서 더 학습하기 쉬운 환경이라고
저자들 스스로 언급한다. 따라서 성능 향상이
전적으로 continual representation의 이점 때문인지,
혹은 환경 난이도 차이도 함께 작용하는지는
완전히 분리되어 있지 않다.

### 7.4 환경 변화가 이산적이다

논문 결론에서도 인정하듯이, 방법은 현재
environment 1에서 environment 2로 바뀌는
**discrete change**에 맞춰 설명된다.
점진적 non-stationary drift나 연속적 변화에 대해서는
후속 연구가 필요하다.

## 8. 후속 연구와의 연결

이 논문은 이후 저자들의
`S-TRIGGER: Continual State Representation Learning via`
`Self-Triggered Generative Replay`로 자연스럽게 이어진다.
본 논문이 generative replay와
통계적 변화 감지의 기본 아이디어를 제시했다면,
후속 연구는 이를 더 명확한 self-triggered continual learning
프레임워크로 발전시킨 것으로 볼 수 있다.

더 넓게 보면 이 논문은 다음 흐름의 초기 사례다.

- generative replay를 representation learning에 적용하는 연구
- world model / latent model을 continual하게 학습하는 연구
- RL에서 perception module과 control module을 분리해
  continual adaptation하는 연구

## 9. 실무 및 연구 인사이트

1. **정책만 continual하게 만들면 충분하지 않다**:
   RL 시스템에서는 perception encoder도 함께 관리해야 한다.
2. **reconstruction error는 유용한 운영 신호다**:
   단순 품질 지표를 넘어 환경 변화 탐지 신호로도 쓸 수 있다.
3. **generative replay는 메모리 대체재가 될 수 있다**:
   특히 과거 원본 데이터를 저장하기 어려운 환경에서 유효하다.
4. **representation 품질은 downstream control로 검증해야 한다**:
   latent space가 보기 좋다고 해서 제어 성능이 좋은 것은 아니다.
5. **forward transfer는 representation 수준에서도 중요하다**:
   새 환경 적응 속도를 높이는 특징을 학습하는 것이
   continual RL의 핵심 축 중 하나다.

## 10. 종합 평가

이 논문은 거대한 benchmark 성능을 보여주는 논문은 아니다.
대신 **continual learning과 RL 사이의 인터페이스가 어디에 있는가**
를 꽤 정확하게 짚는다. 즉, continual RL 문제를
정책 업데이트 문제로만 보지 않고,
환경을 압축하는 state representation 자체를
계속 관리해야 한다는 관점을 제시한다.

방법론은 단순하지만 설계가 명확하다.
VAE, generative replay, 통계적 변화 감지, PPO 평가라는
구성 요소가 잘 맞물려 있으며, 과거 데이터 비저장과 bounded size라는
실용적 조건도 만족한다.

한편 환경 복잡도와 생성 품질 한계 때문에
현대적인 고난도 continual RL 문제에 그대로 충분하다고 보기는
어렵다. 그럼에도 불구하고 이 논문은
**continual state representation learning**이라는 문제를
독립적으로 정식화하고, generative replay가 그 해법이 될 수 있음을
초기에 보여준 중요한 출발점으로 읽을 가치가 있다.

## 11. 참고 링크

- [arXiv abs](https://arxiv.org/abs/1810.03880v3)
- [City Research Online PDF](
  https://openaccess.city.ac.uk/id/eprint/22447/1/1810.03880v3.pdf)
- [CatalyzeX summary](
  https://www.catalyzex.com/paper/continual-state-representation-learning-for)
