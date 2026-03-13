---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Energy-Based Models for Continual Learning

- Shuang Li, Yilun Du, Gido van de Ven, Igor Mordatch
- CoLLAs 2022 / arXiv 2020
- Continual learning through EBM training dynamics

---

## 문제 설정

- Continual Learning의 핵심 문제는 catastrophic forgetting이다.
- 기존 접근은 주로 다음 세 계열이다.
  - replay / memory
  - regularization
  - architecture expansion
- 이 논문은 질문을 다르게 던진다.
- 핵심 질문:
  - **모델 구조가 아니라 학습 objective 자체를 바꾸면** interference를 줄일 수 있는가?

---

## 핵심 아이디어

- 저자들은 Energy-Based Model(EBM)이
  continual learning에 자연스럽게 잘 맞는 모델 클래스라고 주장한다.
- 이유는 EBM의 contrastive divergence 기반 학습이
  이전 정보와의 간섭을 상대적으로 덜 일으킬 수 있기 때문이다.
- 즉, replay나 explicit regularizer 없이도
  forgetting을 줄일 수 있는 학습 동역학이 존재한다는 관점이다.
- 핵심 포인트:
  - continual learning을 "memory management"가 아니라
  **energy landscape shaping** 문제로 본다.

---

## EBM 기본 관점

- EBM은 입력 $x$에 대해 energy function $E_{\theta}(x)$를 정의한다.
  $$
  p_{\theta}(x) = \frac{e^{-E_{\theta}(x)}}{Z_{\theta}}
  $$
- 직관:
  - 데이터에 잘 맞는 샘플에는 낮은 energy
  - 맞지 않는 샘플에는 높은 energy
- continual learning 관점에서는, 새 task를 학습할 때도 이전 task 영역의 energy landscape를 완전히 무너뜨리지 않는 것이 중요하다.

---

## 왜 EBM이 CL에 유리한가

- 저자들의 주장에 따르면, EBM은 다음 이유로 continual learning에 적합하다.
  - contrastive objective가 지나친 interference를 줄일 수 있다.
  - 새 task를 추가해도 energy landscape를 점진적으로 조정할 수 있다.
  - task나 class 수가 늘어나는 설정에 유연하게 대응할 수 있다.
  - explicit memory 없이도 이전 정보를 더 오래 유지하는 경향이 있다.
- 즉,
  - forgetting을 별도 장치로 막는 것이 아니라
    **모델 자체의 학습 방식이 덜 잊게 만든다**는 관점이다.

---

## Contrastive Divergence의 역할

- 논문이 강조하는 핵심은
  **contrastive divergence-based training objective**다.
- 직관적으로는 다음과 같다.
  - positive sample의 energy는 낮춘다.
  - negative sample의 energy는 높인다.
  - 이 과정을 통해 현재 데이터에 맞는 에너지 지형을 학습한다.
- continual learning 맥락에서 보면,
  이 objective가 이전에 학습한 구조를 완전히 덮어쓰지 않고
  더 점진적으로 수정하도록 돕는다는 것이 논문의 핵심 주장이다.

---

## 다른 CL 방법과의 관계

- 이 논문은 EBM을 기존 CL 계열의 대체재로만 보지 않는다.
  - replay를 쓰지 않아도 강할 수 있다.
  - regularization 없이도 forgetting이 줄어들 수 있다.
  - 동시에 다른 continual learning 방법과 결합도 가능하다.
- 원문이 강조하는 메시지:
  - EBM objective는 자체로 유용할 뿐 아니라,
    **다른 CL 기법의 building block**이 될 수 있다.

---

## Task-Free / General CL로의 확장

- 논문은 명시적인 task boundary가 없는 더 일반적인 설정도 다룬다.
- 핵심 아이디어:
  - 데이터 분포가 바뀌더라도
    EBM은 energy landscape를 적응적으로 조정할 수 있다.
  - 따라서 classical task-incremental setting뿐 아니라,
    task delineation이 명확하지 않은 scenario에도 적용 가능하다.
- 의미:
  - 이 논문은 task-aware benchmark를 넘어서
    **more general continual learning**까지 시야에 넣는다.

---

## 실험 설정과 검증 포인트

- 원문이 검증하는 핵심 포인트는 다음과 같다.
  - EBM이 여러 continual learning benchmark에서 baseline보다 강한가?
  - contrastive divergence objective가 실제로 forgetting을 줄이는가?
  - 다른 continual learning 방법과 결합했을 때도 이득이 있는가?
  - explicit task boundary가 약한 설정에도 적응 가능한가?
- 발표 포인트는 수치 하나보다 이 질문들에 대한 정성적 결론이다.

---

## 주요 결과

- 원문이 보고하는 정성적 결론:
  - EBM은 여러 continual learning benchmark에서 baseline들을 큰 폭으로 앞선다.
  - contrastive divergence objective는 이전 정보와의 interference를 줄이는 데 효과적이다.
  - 다른 continual learning 방법과 결합했을 때도 추가 성능 향상을 준다.
  - task 구분이 명시적이지 않은 더 일반적 환경에도 적응 가능하다.
- 즉, EBM은 단순한 생성모델이 아니라 **continual learning에 유용한 model class**로 제시된다.

---

## 강점

- continual learning을 objective level에서 다시 본다.
- replay / regularization / expansion 바깥의 대안을 제시한다.
- 다른 CL 방법과 결합 가능한 범용 building block 성격이 있다.
- task-free setting까지 확장 가능성을 보여 준다.
- EBM의 장점을 continual learning 맥락에서 설득력 있게 연결한다.

---

## 한계

- EBM 학습 자체가 일반적으로 까다롭고 계산 비용이 크다.
- negative sampling / contrastive divergence 안정성 문제가 있다.
- 실무적으로는 classification backbone보다 tuning이 어려울 수 있다.
- energy-based training의 장점이 모든 대규모 setting에서 유지되는지는
  추가 검증이 필요하다.
- 즉, forgetting 측면의 이점은 크지만, training practicality는 별도 과제다.

---

## 발표용 핵심 메시지

- 이 논문의 포인트는 "CL에는 어떤 regularizer를 붙일까"가 아니라 **"애초에 덜 간섭하는 objective를 쓰면 어떨까"** 이다.
- EBM은 continual learning에서 memory와 architecture에만 의존하지 않는 새로운 축을 제시한다.
- 핵심은 energy landscape를 task stream에 맞춰 점진적으로 shaping하는 것이다.
- 따라서 이 논문은 EBM을 continual learning의 **useful building block**으로 제안한 작업이다.

---

## 결론

- `Energy-Based Models for Continual Learning`은
  continual learning에 EBM을 본격적으로 도입한 논문이다.
- replay, regularization, expansion과 다른 경로로
  forgetting 완화를 시도한다.
- contrastive divergence 기반 objective가
  interference를 줄이고 baseline을 능가할 수 있음을 보인다.
- 이 논문은 continual learning을
  **objective design 문제**로도 볼 수 있게 만든다는 점에서 의미가 크다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/energy_based_models_for_continual_learning_slide.md>
