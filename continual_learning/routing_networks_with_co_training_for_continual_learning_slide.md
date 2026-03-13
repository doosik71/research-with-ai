---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Routing Networks with Co-training for Continual Learning

- Mark Patrick Collier, Effrosyni Kokiopoulou, Andrea Gesmundo, Jesse Berent
- ICML 2020 Workshop on Continual Learning / arXiv 2020
- Sparse routing for continual learning with co-training

---

## 문제 설정

- Continual Learning에서는 새 task를 학습할 때
  이전 task를 잊는 catastrophic forgetting이 발생한다.
- 특히 task 간 성격이 다를수록 interference가 심해진다.
- 기존 방법은 크게 두 계열이다.
  - fixed-capacity network에서 학습 규칙을 바꾸는 방법
  - 새 task마다 네트워크 용량을 점점 늘리는 방법
- 핵심 질문: **고정 용량 네트워크 안에서** task별로 다른 계산 경로를 쓰게 만들 수 있는가?

---

## 핵심 아이디어

- 이 논문은 **sparse routing networks**를 continual learning에 적용한다.
- 입력마다 전체 네트워크를 다 쓰지 않고,
  전문가(expert) 집합 중 일부 경로만 활성화한다.
- 유사한 task는 일부 expert를 공유하고,
  다른 task는 더 분리된 expert를 쓰도록 유도한다.
- 의도:
  - dissimilar task 간 간섭은 줄이고
  - similar task 간 positive transfer는 유지한다.

---

## Routing Network 관점

- routing network의 직관은 간단하다.
  - router가 입력을 보고 어떤 expert path를 쓸지 정한다.
  - sparse activation을 통해 매 입력마다 일부 모듈만 작동한다.
  - 결과적으로 하나의 dense network보다
    **task-conditioned modular computation**이 가능해진다.
- continual learning 맥락에서는 이것이 중요하다.
  - 서로 다른 task가 항상 같은 파라미터를 공유하지 않아도 된다.

---

## 왜 Continual Learning에 유리한가

- routing이 잘 되면 다음 성질을 기대할 수 있다.
  - 유사한 task는 overlapping experts를 사용한다.
  - 서로 다른 task는 disjoint experts를 더 많이 사용한다.
- 즉,
  - shared subspace가 필요한 곳에서는 transfer를 얻고
  - 충돌이 큰 곳에서는 path separation으로 forgetting을 줄인다.
- 이 논문의 핵심 가치는 fixed-capacity architecture 안에서 이 분리를 구현하려는 데 있다.

---

## Co-training이 왜 필요한가

- 저자들의 주장에 따르면, routing network를 그대로 continual learning에 쓰면 문제가 생긴다.
  - 새 task가 들어왔을 때 새 expert들이 poorly initialized 상태다.
  - router가 잘못된 path를 선택하면 학습 초기에 불안정해진다.
- 이를 해결하기 위해 논문은 **co-training**을 제안한다.
- 핵심 목적:
  - 새로운 task가 등장했을 때 새 expert들이 너무 나쁜 초기화 때문에 버려지지 않도록 한다.

---

## Co-training의 역할

- 발표용으로 보면 co-training은 다음처럼 이해하면 된다.
  - 기존에 이미 쓰이던 expert와
    새로 투입되는 expert를 함께 학습시킨다.
  - 새 expert가 완전히 무의미한 상태에 머물지 않도록
    안정적인 초기 학습 신호를 준다.
  - 그 결과 router가 점진적으로 더 적절한 sparse path를 배울 수 있다.
- 즉,
  - routing이 구조적 분리를 담당하고
  - co-training이 학습 초기 안정성을 담당한다.

---

## Replay와의 결합

- 논문 초록 기준으로,
  제안 방법은 **small episodic memory replay buffer**와 결합된다.
- 포인트:
  - routing + co-training만이 아니라
    작은 replay를 함께 사용한 설정이다.
  - 이 조합이 densely connected network보다 더 좋은 결과를 낸다.
- 중요한 해석:
  - 이 논문은 replay를 완전히 배제하기보다,
    **작은 replay와 modular routing의 결합**을 실용적 해법으로 본다.

---

## 실험 설정

- 확인 가능한 벤치마크:
  - MNIST-Permutations
  - MNIST-Rotations
- 비교 기준:
  - densely connected networks
  - sparse routing networks with co-training
- 논문의 핵심 검증 포인트:
  - sparse path selection이 forgetting을 줄이는가
  - co-training이 새 task 도입 시 학습 안정성을 높이는가

---

## 주요 결과

- 원문 초록에서 확인되는 결론은 다음과 같다.
  - sparse routing networks with co-training은
    MNIST-Permutations와 MNIST-Rotations에서
    densely connected networks보다 더 좋은 성능을 보였다.
  - 특히 task 간 dissimilarity가 클수록
    path separation의 장점이 더 중요해진다는 문제의식과 잘 맞는다.
- 보수적으로 해석하면,
  - 이 논문은 fixed-capacity modular network가
    continual learning에서 유효할 수 있음을 보여 주는 초기 증거다.

---

## 이 논문의 강점

- fixed-capacity와 dynamic routing을 결합했다.
- architectural separation과 transfer를 동시에 노린다.
- fully dense sharing보다 더 세밀한 task 분리가 가능하다.
- co-training으로 routing network의 실제 학습 문제를 다룬다.
- 작은 replay와 결합하는 현실적 방향을 제시한다.

---

## 한계

- 현재 확인 가능한 실험은 MNIST 변형 벤치마크 중심이다.
- task 수와 expert 수가 커질 때 routing complexity가 커질 수 있다.
- replay를 완전히 제거한 접근은 아니다.
- 성능은 router 품질과 expert utilization에 크게 의존한다.
- 대규모 비전/언어 환경에서의 확장성은 이 논문만으로 판단하기 어렵다.

---

## 발표용 핵심 메시지

- 이 논문의 핵심은 **고정 용량 네트워크에서도 task별 다른 path를 쓰게 만들 수 있다**는 점이다.
- continual learning에서 중요한 것은 단순히 파라미터를 덜 바꾸는 것이 아니라, **어떤 task가 어떤 모듈을 공유할지 학습하는 것**일 수 있다.
- co-training은 좋은 아이디어를 실제로 학습 가능하게 만드는 안정화 장치다.
- 따라서 이 논문은 modular continual learning의 초기 설계 사례로 읽는 것이 적절하다.

---

## 결론

- `Routing Networks with Co-training for Continual Learning`은
  sparse routing architecture를 continual learning에 적용한다.
- 유사한 task는 expert를 공유하고,
  다른 task는 더 분리된 path를 사용하도록 학습한다.
- co-training은 새 task 도입 시 poorly initialized expert 문제를 완화한다.
- 결과적으로 작은 replay와 결합했을 때,
  dense network보다 더 나은 continual learning 성능을 보인다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/continual_learning/routing_networks_with_co_training_for_continual_learning_slide.md>
