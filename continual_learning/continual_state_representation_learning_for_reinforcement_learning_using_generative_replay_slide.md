---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Continual State Representation Learning for Reinforcement Learning using Generative Replay

- Hugo Caselles-Dupré, Michael Garcia-Ortiz, David Filliat
- NeurIPS 2018 Continual Learning Workshop / arXiv 2018
- Continual RL representation learning with VAE + generative replay

---

## 문제 설정

- 이 논문은 policy 자체보다
  **state representation learning**의 continual adaptation을 다룬다.
- 환경이 바뀌면 encoder가 과거 환경 표현을 잊어버릴 수 있다.
- representation이 무너지면 그 위의 RL policy도 영향을 받는다.

핵심 질문:

- 과거 환경 raw data를 저장하지 않고도
  state representation을 계속 유지할 수 있는가?

---

## 핵심 아이디어

- 관측 압축 모델로 VAE를 사용한다.
- 환경이 바뀌면 이전 VAE가 과거 환경 샘플을 생성한다.
- 새 환경 데이터와 생성 replay 데이터를 합쳐 새 VAE를 학습한다.
- encoder의 latent feature를 PPO 입력으로 사용한다.
- 즉, `representation learning`, `generative replay`, `RL downstream evaluation`을 하나로 연결한 구조다.

---

## 전체 파이프라인

1. 환경 1에서 VAE를 학습한다.
2. reconstruction error 분포를 모니터링해 환경 변화를 감지한다.
3. 변화가 감지되면 이전 VAE에서 과거 환경 샘플을 생성한다.
4. 생성 샘플 + 새 환경 데이터를 함께 사용해 새 VAE를 학습한다.
5. 최종 latent representation을 PPO policy 입력으로 사용한다.

- 포인트: replay buffer에 원본 과거 데이터를 저장하지 않는다.

---

## VAE 기반 State Representation

- 기본 목적 함수는 일반적인 VAE와 같다.
  $$
  \mathcal{L}_{\mathrm{VAE}}
  = \mathbb{E}_{q_{\phi}(z \mid x)}[-\log p_{\theta}(x \mid z)]
  + \mathrm{KL}(q_{\phi}(z \mid x) \parallel p(z))
  $$
- 의미:
  - 관측 $x$를 latent variable $z$로 압축한다.
  - reconstruction과 regularization을 함께 최적화한다.
  - RL policy는 raw pixel 대신 이 latent feature를 사용한다.

---

## Generative Replay

- 환경 2로 넘어가면 과거 환경 데이터는 다시 사용하지 않는다.
- 대신 이전 VAE가 과거 환경 관측을 생성한다.
- 환경 2 학습 시 사용 데이터:
  - 새 환경에서 랜덤 정책으로 수집한 500,000개 상태
  - 이전 VAE가 생성한 500,000개 replay 샘플
- 장점:
  - 원본 과거 데이터 저장 불필요
  - 모델 크기를 bounded하게 유지
  - 과거 환경 분포를 근사적으로 복원

---

## 환경 변화 자동 감지

- 변화 감지에는 VAE reconstruction error 분포의 차이를 사용한다.
- 논문은 Welch's t-test를 적용한다.
  $$
  t = \frac{\bar{x}_1 - \bar{x}_2} {\sqrt{(s_1^2 + s_2^2)/N}}
  $$
- 해석:
  - 현재 VAE가 새 환경을 잘 설명하지 못하면 reconstruction error 평균이 달라진다.
  - 단순 분포 변화가 아니라 **representation model을 갱신해야 할 정도의 변화**를 찾는 데 초점이 있다.

---

## 실험 설정

- 환경:
  - Flatland 기반 2-D first-person 환경
  - 두 환경은 구조는 비슷하지만 edible item 색상이 다르다.
  - **행동**: 전진, 좌회전, 우회전
  - **목표**: 500 timestep 동안 edible item 최대한 많이 먹기
- 비교 방법:
  - Raw pixels, VAE trained on source, VAE fine-tuning, VAE generative replay
- RL 평가는 PPO로 수행한다.

---

## 변화 감지 결과

- Welch's t-test를 5000회 반복 평가한 결과:
  - 변화가 있어야 할 경우: **100% 감지 성공**
  - 변화가 없어야 할 경우: **99.5% 정상 판단**
  - 기준 p-value: 0.01
- 의미:
  - 단순하고 계산량이 작은 통계 검정만으로도
    representation drift를 꽤 안정적으로 감지할 수 있다.

---

## 재구성 성능 결과

- Table 1 핵심 비교:
  - Fine-tuning
    - env 1: $1.3 \times 10^{-3}$
    - env 2: $9.3 \times 10^{-4}$
  - Generative Replay
    - env 1: $3.3 \times 10^{-4}$
    - env 2: $6.4 \times 10^{-4}$
- fine-tuning은 새 환경에는 적응하지만 과거 환경을 크게 잊는다.
- generative replay는 과거 환경 재구성 품질을 훨씬 잘 유지한다.

---

## RL 최종 성능

- Appendix Table 2:
  - Raw pixels: `92.30 ± 5.8`, `123.95 ± 25.6`
  - VAE trained on source: `121.25 ± 5.3`, `111.75 ± 11.9`
  - VAE fine-tuning: `96.55 ± 5.1`, `172.5 ± 11.5`
  - **VAE generative replay**: `112.85 ± 13.2`, `256.95 ± 10.3`
- 포인트:
  - representation을 쓰는 것이 raw pixel보다 대체로 유리하다.
  - generative replay는 Task 2에서 특히 큰 향상을 보인다.
  - 이는 forgetting 방지뿐 아니라 **forward transfer**도 시사한다.

---

## 강점과 한계

- 강점:
  - RL에서 perception encoder의 continual learning을 전면에 둔다.
  - raw data 비저장 조건과 bounded system size를 만족한다.
  - reconstruction뿐 아니라 PPO 성능까지 연결해 평가한다.
  - change detection이 단순하고 해석 가능하다.
- 한계:
  - 환경 변화가 색상 차이 중심이라 비교적 단순하다.
  - 생성 품질에 성능이 크게 의존한다.
  - 연속적 drift보다 이산적 환경 전환에 더 가깝다.

---

## 발표용 핵심 메시지

- Continual RL은 policy만의 문제가 아니다.
- **state representation encoder도 continual하게 관리해야 한다.**
- generative replay는 replay buffer의 대체재가 될 수 있다.
- reconstruction error는 단순 품질 지표가 아니라
  운영 중 환경 변화 감지 신호가 될 수 있다.
- 이 논문은 continual RL과 continual representation learning의 접점을 초기에 잘 보여 준 사례다.

---

## 결론

- 이 논문은 VAE 기반 state representation learning에
  generative replay를 적용했다.
- 원본 과거 데이터를 저장하지 않으면서도
  과거 환경 표현을 유지할 수 있음을 보였다.
- downstream PPO 성능까지 개선되어,
  continual representation learning이 실제 control 성능과 연결됨을 보여 준다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/continual_state_representation_learning_for_reinforcement_learning_using_generative_replay_slide.md>
