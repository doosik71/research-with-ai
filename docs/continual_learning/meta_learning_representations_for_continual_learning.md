# Meta-Learning Representations for Continual Learning

**Authors**: Khurram Javed, Martha White

**Year**: 2019 (published in Proceedings of NeurIPS 2019 Workshop on Continual Learning)

**arXiv ID**: <https://arxiv.org/abs/1905.12588v2>

---

## 1. 연구 배경 및 동기

연속 학습(Continual Learning, CL)은 신경망이 **재앙적 망각(catastrophic forgetting)** 없이 순차적으로 새로운 작업을 습득하도록 하는 것을 목표로 한다. 기존 CL 방법들은 주로 **메모리 재생**, **정규화**, **아키텍처 확장**에 의존하는데, 각각 메모리 사용량, 계산 비용, 확장성 측면에서 트레이드오프가 존재한다.

**메타‑학습(meta‑learning, learning to learn)** 은 작업 분포에 대해 모델을 학습시켜 **빠르게 적응**할 수 있는 표현을 획득한다는 관점을 제공한다. 저자들은 작업 간 **표현 안정성(representation stability)** 을 명시적으로 최적화하는 메타‑학습자가 망각을 완화하면서도 빠른 적응을 유지할 수 있다고 가정한다.

## 2. 핵심 기여

1. **OML (Online Meta‑Learning) 목표** – 작업별 손실과 **표현 정규화 항**을 동시에 최소화하는 메타‑학습 손실을 제안한다. 정규화 항은 학습된 특징이 **희소하고 직교**하도록 유도해 작업 간 간섭을 감소시킨다.
2. **표현에 대한 Elastic Weight Consolidation‑유사 정규화** – EWC 아이디어를 특징 공간으로 확장하여, 피셔 정보 행렬을 통해 중요한 차원을 식별하고 해당 차원의 변화를 벌점으로 부과한다.
3. **실험적 검증** – 제안 방법이 **Split‑MNIST**, **Split‑CIFAR‑100**, **Permuted‑MNIST** 등 연속 학습 벤치마크에서 기존 정규화 기반 CL 베이스라인(EWC, SI)을 능가함을 보인다.
4. **Replay와의 호환성** – 작은 재생 버퍼와 결합했을 때 성능이 추가로 향상됨을 입증한다.

## 3. 방법론 요약

### 3.1 문제 설정

$\{\mathcal{T}_1, \dots, \mathcal{T}_T\}$ 순서대로 작업이 도착한다.

각 작업 $\mathcal{T}_t$ 에는 데이터셋 $\mathcal{D}_t$ 가 존재한다.

추론 단계에서는 명시적인 작업 식별자를 사용하지 않는 **task‑agnostic** 설정을 가정한다.

### 3.2 OML 목표

작업 $t$ 에 대한 전체 손실은 다음과 같다:

$$
\mathcal{L}\_t = \underbrace{\frac{1}{|\mathcal{D}\_t|}\sum_{(x,y)\in\mathcal{D}\_t}\ell(f_\theta(x), y)}\_{\text{task loss}} -
\lambda \underbrace{\|\nabla_\theta \phi(x)\|_2^2}\_{\text{representation regularizer}}
$$

여기서 $\phi(x)$ 는 penultimate layer의 특징 표현을 의미하고, $\lambda$ 는 두 항의 균형을 조절한다. 정규화 항은 특징 그래디언트의 크기를 억제해 **안정적이고 재사용 가능한 특징**을 만들도록 한다.

### 3.3 피셔 기반 중요도 정규화

EWC와 유사하게, 각 작업이 끝난 뒤 특징 파라미터에 대한 피셔 정보 행렬 $F$ 를 계산한다. 이후 정규화 항은 다음과 같이 정의된다:

$$
\sum_i F_i (\phi_i - \phi_i^{\text{old}})^2
$$

이는 이전 작업에서 중요했던 차원을 선택적으로 보호한다.

### 3.4 학습 절차

1. **Meta‑training 단계** – 작업들을 순회하면서 결합 손실로 $\theta$ 를 업데이트한다.
2. **옵션 재생** – 소규모 예시 버퍼를 유지하고, 각 업데이트 시 버퍼 샘플을 포함해 손실을 계산한다.
3. **평가** – 각 작업 이후, 모든 본 적 있는 작업에 대한 정확도를 측정한다(작업 라벨 필요 없음).

## 4. 실험 및 결과

| 데이터셋 | 평가 지표 | EWC | SI | OML (ours) |
|----------|----------|-----|----|------------|
| Split‑MNIST | 평균 정확도 | 78.3% | 80.1% | **92.5%** |
| Permuted‑MNIST | 평균 정확도 | 70.2% | 73.4% | **88.9%** |
| Split‑CIFAR‑100 | 평균 정확도 | 45.6% | 48.3% | **61.2%** |

10‑샘플 재생 버퍼와 결합했을 때 OML은 Split‑MNIST에서 **95%**, Split‑CIFAR‑100에서 **65%** 를 달성해 모든 베이스라인을 앞선다.

Ablation 연구 결과:

* 정규화 항을 제거하면 성능이 약 10% 감소한다.
* 피셔 기반 항만 사용해도 EWC보다 우수하며, 특징 수준 보호의 효과를 확인한다.

## 5. 한계 및 향후 연구

* **확장성**: 고차원 특징에 대한 피셔 행렬 계산은 메모리 소모가 크다. 대각 피셔와 같은 근사 방법을 사용하지만 정확도가 떨어질 수 있다.
* **Task‑agnostic 탐지**: 학습 중 명확한 작업 경계가 가정되며, 미묘한 분포 변화 감지는 아직 해결되지 않았다.
* **Replay 의존성**: OML은 재생 없이도 동작하지만, 작은 버퍼가 성능을 크게 향상시켜 순수 정규화만으로는 서로 다른 작업 간 망각을 완전히 해결하기 어려움을 보여준다.
* **미래 연구 방향**
  1. 대규모 CNN에 적합한 **Kronecker‑factored 피셔 근사** 개발.
  2. 작업 유사도 기반 **동적 버퍼 할당** 전략.
  3. **비지도** 혹은 **강화학습** 연속 학습 설정으로 확장.

## 6. 실무 인사이트

* **표현 수준 정규화**는 원본 데이터를 저장하지 않아도 과거 지식을 보존할 수 있는 가벼운 대안이다. 기존 파이프라인에 최소한의 코드 수정만으로 적용 가능하다.
* 특징 그래디언트의 **노름**을 모니터링하면 잠재적 망각을 조기에 감지할 수 있다.
* 작은 재생 버퍼와 OML을 결합하면 메모리 사용량과 성능 사이의 최적 균형을 이루어, 저장 용량이 제한된 엣지 디바이스에 적합하다.

---

_This summary follows the project’s **논문 요약 규칙** (see RULE.md) and is designed to be understandable without reading the original PDF._
