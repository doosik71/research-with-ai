# Federated Learning with Unbiased Gradient Aggregation and Controllable Meta Updating

Xin Yao, Tianchi Huang, Rui-Xiao Zhang, Ruiyu Li, Lifeng Sun

## 🧩 Problem to Solve

페더레이션 학습(Federated Learning, FL)의 기본 알고리즘인 FedAvg는 통신 비용 및 프라이버시 문제를 해결하기 위해 로컬 훈련과 모델 집계를 사용합니다. 그러나 저자들은 이론적 분석을 통해 FedAvg의 두 가지 주요 문제점을 지적합니다.

1. **기울기 편향 (Gradient Bias)**: 로컬 장치에서의 여러 단계 업데이트($\omega_k^{(i)} = \omega_k^{(i-1)} - \eta g_k^{(i)}$)는 글로벌 모델 집계 시 기울기 편향을 야기합니다. 즉, 각 로컬 기울기 $g_k^{(i)}$가 현재 로컬 모델 $\omega_k^{(i-1)}$에 대해 계산되므로, 최종적으로 집계되는 기울기가 초기 글로벌 모델 $\omega_t$에 대한 진정한 기울기 방향과 일치하지 않게 됩니다.
2. **불일치하는 최적화 목표 (Inconsistent Optimization Objectives)**: 매 라운드마다 무작위로 일부 클라이언트만 훈련에 참여($D_{S_t}$)시키므로, 최적화 목표가 라운드마다 변동하고, 이는 실제 목표 분포 $D$와 일관되지 않은 최적화 방향으로 이어집니다. 이는 명확하고 일관된 목표의 부재를 초래합니다.

## ✨ Key Contributions

이 논문은 FedAvg의 위 두 가지 문제점을 해결하기 위한 모델 및 태스크에 구애받지 않는 두 가지 개선 방안을 제안합니다.

* **편향 없는 기울기 집계 (Unbiased Gradient Aggregation, UGA) 알고리즘 개발**: keep-trace gradient descent와 gradient evaluation 전략을 사용하여, 로컬 업데이트 중 발생하는 기울기 편향을 제거합니다. 이는 기존 FedAvg 프레임워크와 호환됩니다.
* **제어 가능한 메타 업데이트 (Controllable Meta Updating, FedMeta) 절차 도입**: 모델 집계 후 서버에서 소규모의 메타 훈련 데이터셋($D_{meta}$)을 사용하여 추가 메타 업데이트를 수행합니다. 이를 통해 명확하고 일관된 최적화 목표를 설정하고, 페더레이션 모델의 최적화를 제어 가능한 방식으로 유도합니다.
* **성능 입증**: 다양한 네트워크 아키텍처(CNN, GRU)와 IID 및 non-IID FL 설정에서의 실험을 통해, 제안된 방법(단독 또는 결합)들이 기존 FedAvg 및 다른 인기 있는 FL 알고리즘(FedProx, FedShare)보다 더 빠른 수렴과 더 높은 정확도를 달성함을 보여줍니다.

## 📎 Related Works

* **FedShare [33]**: 클라이언트 간 소규모 공개 데이터셋을 공유하여 가중치 발산(weight divergence)을 완화합니다. 그러나 이는 추가적인 통신 비용과 프라이버시 문제를 야기할 수 있습니다.
* **FedProx [21]**: 이질적인 네트워크에서 FedAvg의 성능을 개선하기 위해 proximal term을 사용합니다.
* **Federated Distillation (FD) 및 Federated Augmentation (FAug) [14]**: 누락된 데이터 샘플을 대체하기 위한 프로토타입 벡터 생성 또는 생성 모델 훈련을 제안하여 non-IID 데이터 문제를 해결하려 하지만, 복잡한 데이터셋에 대한 생성 모델 훈련의 어려움으로 적용에 한계가 있습니다.
* **Model-Agnostic Meta-Learning (MAML) [11]**: few-shot 학습을 위한 빠른 적응 훈련 방식을 제안합니다. UGA의 keep-trace gradient descent는 MAML의 내부 루프(inner-loop) 업데이트와 유사한 개념을 공유합니다.
* **메타 학습과 FL의 결합 [10, 16]**: 주로 페더레이션 모델의 개인화(personalization)에 중점을 두어 이 논문의 일반적인 성능 개선 목표와는 차이가 있습니다. Per-FedAvg [10]는 UGA의 특수한 경우로 볼 수 있습니다.

## 🛠️ Methodology

### 1. 편향 없는 기울기 집계 (Unbiased Gradient Aggregation, UGA)

FedAvg의 다단계 로컬 업데이트로 인해 발생하는 기울기 편향 문제를 해결하기 위해, 최종 로컬 모델이 초기 글로벌 모델 $\omega_t$에 대한 함수 관계를 유지하면서 기울기를 계산하는 전략을 사용합니다.

* **Keep-trace Gradient Descent (기록 보존 기울기 하강)**:
  * 로컬 클라이언트 $k$는 $E-1$ 에포크 동안 일반적인 경사 하강을 수행하지만, 각 업데이트 단계 $\omega_k^{(i)} = \omega_k^{(i-1)} - \eta g_k^{(i)}$에서 $\omega_k^{(i)}$와 $\omega_k^{(i-1)}$ 사이의 함수 관계를 기록합니다. 즉, 자동 미분(automatic differentiation)을 위해 전체 계산 그래프를 유지합니다.
  * 이를 통해 최종 로컬 모델 $\omega_k^t$가 초기 글로벌 모델 $\omega_t$의 함수($h_k(\omega_t)$)로 표현될 수 있도록 준비합니다.
* **Gradient Evaluation (기울기 평가)**:
  * $E-1$ 에포크의 로컬 업데이트 후, 마지막 에포크에서 클라이언트 $k$는 전체 로컬 데이터 $D_k$를 사용하여 최종 로컬 모델 $\omega_k^t$에 대한 손실을 평가합니다.
  * 이때, 앞서 기록된 계산 기록을 역추적하여, 공유된 초기 글로벌 모델 $\omega_t$에 대한 기울기 $g_k^t = \nabla_{\omega_t} L(\omega_k^t; D_k)$를 계산합니다.
  * 이 $g_k^t$는 모든 클라이언트가 공통된 $\omega_t$에 대해 계산한 편향 없는 기울기이므로, 서버는 이를 가중 평균하여 $\omega_{t+1} \leftarrow \omega_t - \eta_g \sum_{k \in S_t} \frac{n_k}{n_{S_t}} g_k^t$와 같이 편향 없이 글로벌 모델을 업데이트할 수 있습니다.

### 2. 제어 가능한 메타 업데이트 (Controllable Meta Updating, FedMeta)

FedAvg에서 최적화 목표의 불일치 문제를 해결하고 일관된 최적화 방향을 제공하기 위해 메타 업데이트 절차를 도입합니다.

* **2단계 최적화 (Two-stage Optimization)**:
    1. **내부 루프 (Inner-loop)**: 각 클라이언트에서 FedAvg (또는 UGA) 방식으로 로컬 훈련을 수행하고, 서버에서 클라이언트로부터 받은 업데이트를 집계하여 $\omega_{t+1}$을 얻습니다.
    2. **외부 루프 (Outer-loop, 메타 업데이트)**: 서버는 모델 집계 후, 사전에 정의된 소규모의 메타 훈련 데이터셋 $D_{meta}$를 사용하여 추가적인 업데이트를 수행합니다.
        * $\omega_{meta}^{t+1} = \omega_{t+1} - \eta_{meta} \nabla_{\omega_{t+1}} L(\omega_{t+1}; D_{meta})$
* **$D_{meta}$의 역할**:
  * 훈련 과정에 명확하고 일관된 최적화 목표를 제공합니다.
  * $D_{meta}$의 선택을 통해 페더레이션 모델의 동작(예: 특정 편향 감소, 공정성 증진)을 제어할 수 있는 강력한 수단을 제공합니다. $D_{meta}$는 전체 데이터 $D$의 IID 부분 집합일 수도 있고, 특정 목표에 맞게 구성될 수도 있습니다.

## 📊 Results

제안된 FedMeta w/ UGA는 다양한 FL 설정에서 기존 방법들을 능가하는 성능을 보였습니다.

* **CIFAR-10 (IID)**:
  * FedMeta w/ UGA는 FedAvg, FedProx, FedShare 대비 평균 3%p 이상의 정확도 향상과 더 빠른 수렴 속도를 달성했습니다.
  * 로컬 업데이트 단계가 많을수록(작은 배치 크기 $B$ 또는 많은 로컬 에포크 $E$) FedMeta w/ UGA의 개선 효과가 더 두드러졌습니다.

* **FEMNIST (non-IID)**:
  * FedMeta w/ UGA는 FedAvg, FedProx, FedShare보다 수렴 속도와 최종 정확도 모두에서 크게 우수했습니다.
  * 예를 들어, 90% 정확도 도달에 FedAvg는 437 통신 라운드가 필요했지만, FedMeta w/ UGA는 59 라운드(13.5%)만 필요하여 통신 비용을 대폭 절감했습니다. 최종 정확도는 FedAvg의 90.22% 대비 98.18%를 달성했습니다.

* **Shakespeare (non-IID, GRU 모델)**:
  * FedMeta w/ UGA는 FedAvg, FedShare, FedProx 대비 각각 13.83%, 10.05%, 5.58%의 정확도 향상을 보이며 더 빠른 수렴을 입증했습니다.
  * GRU 모델과 같이 반복 네트워크를 사용하는 경우, FedAvg의 기울기 편향 문제가 더 심각해지므로 UGA의 효과가 더욱 커졌습니다.

* **Controllable $D_{meta}$**:
  * $D_{meta}$와 전체 데이터 $D$ 간의 겹치는 비율이 달라질 때, FedAvg의 정확도는 최대 5%p 하락하는 반면, FedMeta는 $D_{meta}$의 안내 덕분에 약 92%의 일관된 성능을 유지하여 모델 제어 가능성을 입증했습니다.

* **Ablation Study (UGA와 FedMeta의 개별 기여)**:
  * UGA와 FedMeta는 각각 FedAvg보다 성능을 향상시켰습니다.
  * UGA는 특히 로컬 업데이트 단계가 많을 때 FedMeta보다 더 큰 개선을 보였습니다.
  * RNN 모델에서는 UGA가 FedMeta보다 약간 더 나은 성능을 보여, RNN의 심각한 기울기 편향 문제에 UGA가 효과적임을 시사했습니다.

## 🧠 Insights & Discussion

* **UGA의 중요성**: FedAvg의 로컬 다단계 업데이트로 인한 기울기 편향은 특히 non-IID 환경에서 모델 성능 저하의 주요 원인입니다. UGA는 이 편향을 제거하여 글로벌 모델이 실제 전체 데이터 분포에 대한 정확한 방향으로 최적화되도록 돕습니다. 이는 통신 라운드를 줄이고 최종 정확도를 높이는 데 결정적인 역할을 합니다.
* **FedMeta의 활용성**: FedMeta는 최적화 목표의 불일치 문제를 해결하여 학습 안정성을 높입니다. 더 나아가, 서버 측의 소규모 $D_{meta}$를 통해 모델 개발자가 페더레이션 모델의 행동을 명시적으로 제어할 수 있는 강력한 메커니즘을 제공합니다. 이는 데이터에 내재된 편향(성별, 인종 등)을 줄이거나 특정 애플리케이션 요구사항에 맞춰 모델을 '미세 조정'하는 데 활용될 수 있습니다.
* **제한 사항**: UGA는 RNN 모델에서 미분 역전파 과정의 불안정성을 유발할 수 있습니다. FedMeta는 $D_{meta}$의 적절한 구성에 의존하며, $D_{meta}$를 어떻게 효과적으로 구축할지는 여전히 탐구할 여지가 있습니다. 하지만 이 두 개선 방안은 기존 FL 시스템에 쉽게 통합될 수 있으며, 모델 및 태스크에 독립적으로 적용 가능하다는 장점이 있습니다.

## 📌 TL;DR

FedAvg는 로컬 다단계 업데이트로 인한 기울기 편향과 동적으로 변하는 클라이언트 선택으로 인한 최적화 목표 불일치 문제를 겪습니다. 이 논문은 이 문제를 해결하기 위해 **UGA(Unbiased Gradient Aggregation)**와 **FedMeta(Controllable Meta Updating)**를 제안합니다. UGA는 keep-trace gradient descent와 gradient evaluation을 통해 초기 글로벌 모델에 대한 편향 없는 기울기를 집계하며, FedMeta는 서버에 소규모 메타 데이터셋 $D_{meta}$를 도입하여 일관된 최적화 목표를 제공하고 모델 동작을 제어합니다. 실험 결과, 이 두 방법은 다양한 FL 설정(IID, non-IID)에서 FedAvg 및 다른 기준 모델보다 더 빠르고 안정적인 수렴과 높은 정확도를 달성하며, 특히 non-IID 환경에서 상당한 성능 향상을 보였습니다.
