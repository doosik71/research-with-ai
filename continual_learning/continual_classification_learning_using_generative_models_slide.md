---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Continual Classification Learning Using Generative Models

- Frantzeska Lavda, Jason Ramapuram, Magda Gregorova, Alexandros Kalousis
- arXiv 2018
- Continual Learning / Generative Replay / VAE

---

## 문제 설정

- Continual classification에서는 새 task를 학습할수록
  과거 task 성능이 무너지는 catastrophic forgetting이 발생한다.
- 과거 원본 데이터를 계속 저장하기 어렵거나 불가능한 경우가 많다.
- 저자들의 목표는 다음과 같다.
  - 과거 데이터 저장 없이 이전 task를 유지하기
  - 분류 성능과 생성 능력을 함께 보존하기

---

## 핵심 아이디어

- VAE 기반 생성 모델과 classifier를 하나의 latent-variable model로 결합한다.
- teacher-student 구조를 사용해 과거 task의 pseudo-sample을 생성한다.
- student는 다음 두 데이터를 함께 학습한다.
  - 현재 task의 실제 데이터, teacher가 생성한 과거 task 데이터
- 핵심 질문:
  - 과거 데이터를 저장하지 않고도 과거 분포를 다시 학습에 등장시킬 수 있는가?

---

## 확률 모델

- 논문은 다음 joint model을 사용한다.
  $$
  p(x, y, z) = p(x \mid z) p(y \mid z) p(z)
  $$
  - $x$: 입력, $y$: 라벨, $z$: latent variable
- 가정:
  $$
  p(x, y \mid z) = p(x \mid z) p(y \mid z)
  $$
  - 즉, 생성과 분류가 같은 latent representation을 공유한다.

---

## Variational Lower Bound

- 분류까지 포함한 joint likelihood 하한을 사용한다.
  $$
  \log p(x, y) \ge \mathcal{L}(x, y)
  $$

  $$
  \mathcal{L}(x, y)
  = \mathbb{E}_{q_{\phi}(z \mid x)}[\log p(x \mid z)]
  - D_{\mathrm{KL}}(q_{\phi}(z \mid x) \parallel p(z))
  + \mathbb{E}_{q_{\phi}(z \mid x)}[\log p(y \mid z)]
  $$

- 의미:
  - 재구성 항
  - prior regularization 항
  - classification 항

---

## Teacher-Student Replay 구조

- 학습 절차:
  1. task $t$에서 student가 현재 데이터 $(x, y)$를 받는다.
  2. teacher가 이전 task에서 학습한 분포로부터 $(\tilde{x}, \tilde{y})$를 생성한다.
  3. student는 현재 데이터와 생성 replay 데이터를 함께 학습한다.
  4. task 종료 후 student 파라미터를 다음 teacher로 넘긴다.
- 장점:
  - 원본 과거 데이터 저장 불필요
  - task-specific classifier 보존 없이도 과거 정보 유지 가능

---

## 실험 설정

- 벤치마크:
  - Permuted MNIST 5-task
  - Heterogeneous sequence
    - MNIST, FashionMNIST, MNIST permutation
- 비교 대상:
  - vae-cl, EWC baseline, CCL-GM (제안 방법)
- 평가 지표:
  - Average test classification accuracy
  - Average test negative reconstruction ELBO

---

## 실험 결과

- 정성적 결론은 일관적이다.
  - `vae-cl`은 task가 바뀌면 성능이 빠르게 무너진다.
  - EWC는 일부 완화하지만 forgetting이 계속 누적된다.
  - CCL-GM은 분류 정확도와 재구성 능력을 함께 더 잘 유지한다.
- 해석:
  - replay가 단순 출력 보존이 아니라 분포 보존 역할을 한다.
  - latent space를 공유하는 생성-분류 결합 구조가 효과적이다.
- 주의:
  - 원문은 preprint라 수치 표보다 그림 중심의 정성 비교가 많다.

---

## 강점과 한계

- 강점:
  - 생성과 분류를 하나의 확률모형으로 통합했다.
  - 입력과 라벨을 함께 생성하는 joint replay 관점을 제시했다.
  - 초기 generative replay 기반 continual classification 사례로 의미가 있다.
- 한계:
  - 평가가 주로 MNIST 계열에 집중된다.
  - 생성 품질이 낮아지면 replay 품질도 함께 무너진다.
  - task boundary가 명확하다는 가정이 있다.

---

## 발표용 핵심 메시지

- 이 논문의 본질은 "분류기 + 생성기"가 아니라
  **과거 분포를 생성 메모리로 보존하는 continual learning 구조**다.
- 성능 유지의 핵심은 원본 데이터 저장이 아니라
  **plausible replay distribution**을 유지하는 데 있다.
- 현대 관점에서는 diffusion, stronger generator,
  latent replay로 확장해 볼 가치가 크다.

---

## 결론

- CCL-GM은 VAE 기반 joint generative-discriminative 학습과
  teacher-student generative replay를 결합한 방법이다.
- 원본 과거 데이터를 저장하지 않으면서도
  forgetting을 완화할 가능성을 보여 준다.
- 오늘 기준 SOTA라기보다,
  generative replay 계열 continual learning의 초기 핵심 아이디어를 담은 논문이다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/continual_classification_learning_using_generative_models_slide.md>
