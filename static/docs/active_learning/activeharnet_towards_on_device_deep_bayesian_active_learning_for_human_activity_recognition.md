# ActiveHARNet: Towards On-Device Deep Bayesian Active Learning for Human Activity Recognition

이 논문은 **Human Activity Recognition (HAR)**와 **fall detection** 을 모바일·웨어러블 환경에서 수행할 때 생기는 두 문제를 동시에 다룬다. 첫째, 모델이 서버가 아니라 **on-device** 에서 돌아가야 하므로 가볍고 추론이 빨라야 한다. 둘째, 실제 사용자 데이터는 계속 들어오지만 즉시 라벨이 붙지 않으므로, 모델이 **어떤 샘플만 선택적으로 라벨링 요청할지** 스스로 판단해야 한다. 저자들은 이를 위해 **ActiveHARNet** 을 제안한다. 이는 경량 HARNet 계열 구조에 **Bayesian uncertainty estimation** 과 **active learning acquisition function** 을 결합한 프레임워크이며, incremental learning까지 지원해 새 사용자 데이터로 기기를 현장에서 계속 적응시킬 수 있게 하려는 접근이다. 논문 초록과 서론은 이 시스템이 두 공개 데이터셋에서 **추론 효율 향상**과 함께, incremental learning 중 **획득해야 하는 라벨 수를 최소 60% 줄였다**고 주장한다.  

## 1. Paper Overview

논문의 연구 문제는 명확하다. 기존 HAR 연구는 대체로 충분히 라벨된 데이터를 가정하고, 학습도 클라우드나 GPU 환경에서 수행하는 경우가 많았다. 하지만 실제 헬스케어·보행 모니터링·낙상 감지 같은 응용에서는 사용자의 행동 패턴이 사람마다 다르고 시간이 지나며 변한다. 따라서 한 번 학습한 모델을 고정해 쓰기보다, **새로 들어오는 사용자별 데이터를 이용해 점진적으로 모델을 업데이트**해야 한다. 동시에 웨어러블 기기에서는 매 데이터에 라벨을 붙이는 것이 어렵기 때문에, **가장 정보가 많은 샘플만 골라 오라클에게 질의하는 active learning** 이 필요하다. 저자들은 바로 이 지점을 겨냥해, resource-constrained on-device 환경에 맞는 딥러닝 기반 Bayesian active learning HAR 프레임워크를 제안한다.

이 문제가 중요한 이유는 HAR이 단순한 분류 문제가 아니라, 실제 기기 위에서 동작해야 하는 **지속적 적응 시스템** 이기 때문이다. 서버 의존형 방식은 통신 지연과 비용이 있고, 완전 수동 라벨링은 확장성이 떨어진다. 이 논문은 therefore “정확도만 높은 HAR 모델”이 아니라, **배포 후에도 사용자별로 계속 적응할 수 있는 실용적 학습 시스템** 을 만들려 한다는 점에서 의미가 있다.

## 2. Core Idea

핵심 아이디어는 세 요소의 결합이다.

첫째, **HARNet** 이라는 경량 sensor-based deep model을 기반으로 한다.
둘째, 이 모델에 **dropout 기반 Bayesian approximation** 을 넣어 예측 불확실성을 추정한다.
셋째, 그 uncertainty를 이용해 active learning acquisition function으로 현재 샘플을 라벨링할지 결정한다.

이 논문의 novelty는 새로운 acquisition function 하나를 제안했다기보다, **on-device HAR + Bayesian uncertainty + incremental active learning** 을 하나의 통합 프레임워크로 묶었다는 데 있다. 저자들이 서론에서 직접 밝힌 기여도 이 세 가지 축에 맞춰 정리된다.

* wrist-worn HAR / fall detection 데이터셋에서 경량 Bayesian deep model을 평가
* Bayesian active learning으로 uncertainty를 모델링하고 labeling burden을 줄임
* on-device incremental learning으로 사용자 독립적 모델을 사용자 적응형 모델로 업데이트

즉, 이 논문은 “어떤 acquisition function이 더 좋냐”보다, **현실적인 웨어러블 HAR 배포 환경에서 deep active learning이 실제로 돌아갈 수 있는가**를 보이는 시스템 논문에 가깝다.

## 3. Detailed Method Explanation

### 3.1 전체 파이프라인

논문의 Figure 1 설명에 따르면 ActiveHARNet의 파이프라인은 대략 다음 흐름이다.

1. 센서 입력을 HARNet 기반 모델이 처리한다.
2. dropout을 train/test 시 모두 사용해 여러 stochastic forward pass를 수행한다.
3. 예측 분포의 평균과 분산을 바탕으로 uncertainty를 계산한다.
4. acquisition function이 현재 샘플이 충분히 informative한지 판단한다.
5. 선택된 샘플만 oracle에게 라벨을 요청한다.
6. 새 라벨이 붙은 일부 샘플로 모델을 incremental update 한다.

이 구조의 의도는 명확하다. 모든 데이터를 다 저장하고 재학습하는 대신, **작은 수의 informative sample만 선택적으로 라벨링해 모델을 조금씩 갱신**하는 것이다.

### 3.2 Bayesian uncertainty modeling

논문은 Bayesian Neural Network(BNN)를 직접 정확하게 추론하는 대신, **dropout을 variational inference 근사로 사용**한다. 논문이 제시한 분류 likelihood는 다음과 같다.

$$
p(y=c \mid x, \omega) = \mathrm{softmax}(f^\omega(x))
$$

여기서 $\omega$ 는 네트워크 weight이고, $f^\omega(x)$ 는 모델 출력이다. 하지만 진짜 posterior $p(\omega \mid D_{train})$ 는 계산이 어렵다. 그래서 저자들은 Gal 계열 접근을 따라, dropout distribution $q_\theta^*(\omega)$ 로 posterior를 근사한다. 새로운 입력 $x^*$ 에 대한 predictive distribution은 다음처럼 쓴다.

$$
p(y^* \mid x^*, D_{train}) = \int p(y^* \mid x^*, \omega), p(\omega \mid D_{train}), d\omega
$$

그리고 실제 구현에서는 test time에도 dropout을 켠 채 여러 번 stochastic pass를 수행해 평균 예측과 uncertainty를 추정한다. 이것이 ActiveHARNet이 **deterministic classifier가 아니라 uncertainty-aware model** 로 동작하게 만드는 핵심이다.

직관적으로 말하면, 같은 샘플을 여러 번 넣었을 때 예측이 일관되면 확신이 높은 것이고, 예측 분포가 흔들리면 그 샘플은 라벨링 가치가 높다고 보는 것이다.

### 3.3 모델 구조: HARNet 기반 경량 아키텍처

논문은 HARNet 구조를 이용해 inertial sensor data의 **intra-axial dependency** 와 **inter-axial dependency** 를 함께 잡는다. 공개된 본문에 따르면:

* **Intra-axial dependencies** 는 2층 stacked 1D convolution으로 추출한다.
* 각 층의 filter 수는 8, 16이고 kernel size는 2다.
* 그 뒤 batch normalization과 max-pooling(size 2)이 수행된다.

또한 논문은 1D와 2D convolution을 결합해, 단일 축의 local feature와 축 간 상호작용을 동시에 반영한다고 설명한다. 이는 accelerometer의 3축 정보가 개별적으로도 중요하지만, 축들 사이의 관계도 activity recognition에 중요하다는 가정에 기반한다.

이 설계는 대형 sequence model보다 단순하지만, on-device inference를 겨냥한 만큼 **가벼운 구조와 표현력의 균형** 을 노린 것으로 보인다.

### 3.4 Active learning acquisition

논문 초록과 서론은 “several acquisition functions” 를 실험했다고 밝힌다. 본문에 공개된 부분만으로 보면 acquisition은 **Bayesian uncertainty** 를 기반으로 하며, dropout으로 얻은 예측 분포를 이용해 oracle query를 결정한다. 즉 샘플 선택 기준은 representation learning과 별개가 아니라, 모델의 epistemic uncertainty를 반영한다.

현재 제공된 본문 조각에서는 각 acquisition function의 모든 수식이 끝까지 드러나지는 않지만, 논문의 메시지는 분명하다. HAR 같은 sensor time-series 문제에 대해, 기존 hand-crafted feature 기반 AL이 아니라 **deep Bayesian active learning** 을 직접 on-device incremental setting으로 가져왔다는 점이 핵심이다. acquisition function의 세부 정의가 완전하게 보이지 않는 부분은 논문 원문 전체를 더 열어 보면 확인 가능하겠지만, 지금 확보된 텍스트만으로 확실히 말할 수 있는 것은 **uncertainty-aware querying이 central mechanism** 이라는 점이다.

### 3.5 Incremental learning과 user adaptability

이 논문에서 중요한 것은 학습 시점이다. 기존 user-independent model을 만들어 두고 끝내는 것이 아니라, **새 사용자로부터 들어오는 일부 unlabeled stream을 active selection 후 라벨링해 모델을 업데이트** 한다. 저자들은 이를 통해 “retrain from scratch” 없이 사용자 특성에 적응할 수 있다고 강조한다.

즉, ActiveHARNet은 단발성 supervised classifier가 아니라:

* 초기에 공용 모델을 갖고 시작한 뒤
* 배포 후 사용자별 데이터가 들어오면
* 불확실한 샘플만 선택적으로 라벨링하고
* 소량의 새 데이터로 점진적으로 적응하는

**continual adaptation pipeline** 으로 이해하는 것이 맞다.

## 4. Experiments and Findings

논문 초록과 서론 기준으로 실험은 **두 개의 공개 wrist-worn HAR / fall detection 데이터셋** 에서 수행되었다. 저자들은 특히 서로 다른 사용자들에 걸쳐 baseline user-independent evaluation과 resource-constrained incremental active learning evaluation을 구분해 설명한다.

핵심 실험 결과는 두 가지다.

첫째, **inference efficiency boost** 가 있었다.
둘째, incremental learning 동안 필요한 **acquired pool points 수가 크게 감소**했으며, 초록에서는 **최소 60% reduction** 이라고 명시한다.

이 결과가 의미하는 바는 단순 정확도 개선보다 더 실용적이다. 웨어러블 환경에서는:

* 추론이 빨라야 하고
* 질의하는 라벨 수가 적어야 하며
* 적은 supervision으로도 새 사용자에 적응해야 한다.

ActiveHARNet은 바로 이 세 조건을 겨냥했고, 논문은 두 데이터셋에서 이를 실험적으로 뒷받침한다고 주장한다.

다만 현재 확보된 텍스트 조각에서는 정확도 표, latency 수치, acquisition function별 세부 비교표가 완전하게 노출되지는 않는다. 따라서 “어떤 acquisition function이 가장 좋았는지”, “baseline 대비 정확도가 몇 %p 올랐는지” 같은 아주 정밀한 수치는 지금 보이는 본문만으로는 단정하지 않는 것이 맞다. 확실한 것은 **적은 라벨 획득량으로 deployment feasibility를 보였다**는 논문의 주요 결론이다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 문제 설정이 현실적이라는 점이다. 이 논문은 HAR를 단순 오프라인 분류 문제로 보지 않고, **on-device, unlabeled stream, user adaptation** 이라는 실제 제약을 정면으로 다룬다. 이는 실용성 면에서 매우 강하다.

둘째, Bayesian dropout 기반 uncertainty를 사용함으로써, 무거운 정식 Bayesian inference 대신 **가벼운 approximate uncertainty estimation** 으로 active learning을 구현한다. 이는 모바일·웨어러블이라는 resource-constrained setting과 잘 맞는다.

셋째, ActiveHARNet은 모델 구조, query mechanism, incremental update를 분리하지 않고 하나의 프레임워크로 묶었다. 즉 “좋은 classifier” 와 “좋은 샘플 선택기” 를 따로 보는 게 아니라 **배포 가능한 적응 시스템** 으로 설계했다는 점이 장점이다.  

### 한계

한계도 있다.

첫째, 공개된 부분만 보면 실험이 두 공개 데이터셋 중심이어서, 다양한 센서 구성이나 더 큰 규모의 long-term real-world deployment까지 일반화하기엔 증거가 제한적이다.
둘째, dropout-based uncertainty는 실용적이지만 정교한 posterior approximation은 아니므로, uncertainty calibration이 얼마나 정확한지는 별도 문제다.
셋째, acquisition function별 차이와 정확도/비용 trade-off의 세부 수치가 현재 보이는 조각만으로는 충분히 복원되지 않는다. 논문 전문 표와 수치가 함께 봐야 더 정밀한 해석이 가능하다.

### 해석

비판적으로 보면, 이 논문의 진짜 공헌은 “새로운 HAR backbone” 그 자체보다, **uncertainty-aware incremental adaptation을 device-side HAR에 실용적으로 얹었다**는 데 있다. HAR는 사용자 편차가 매우 큰 문제이기 때문에, 정적인 global model보다 **사용자별 온라인 적응** 이 중요하다. ActiveHARNet은 바로 그 지점에서, 비용 높은 라벨링을 uncertainty-based query로 줄이면서 on-device 적응 가능성을 보여 준다. 이 때문에 이 논문은 정확도 경쟁 논문이라기보다 **deployable active learning system paper** 로 읽는 편이 더 정확하다.

## 6. Conclusion

이 논문은 wearable HAR와 fall detection을 위해 **on-device deep Bayesian active learning** 프레임워크인 ActiveHARNet을 제안한다. 핵심은 경량 HARNet 구조 위에 dropout 기반 Bayesian uncertainty estimation과 acquisition function을 결합해, **가장 informative한 샘플만 선택적으로 라벨링하고 incremental update** 를 수행한다는 점이다. 논문은 이를 통해 두 공개 데이터셋에서 추론 효율 향상과 함께, incremental learning에 필요한 라벨 획득량을 최소 60% 줄였다고 보고한다.  

실무적으로 이 연구가 중요한 이유는 명확하다. HAR 시스템이 실제로 유용하려면, 클라우드 의존 없이 기기 위에서 빠르게 동작하고, 사용자별 차이에 적응하며, 라벨링 부담을 줄여야 한다. ActiveHARNet은 이 세 요구를 하나의 프레임워크로 연결했다는 점에서, 이후의 on-device continual learning / active sensing 연구의 초기 형태로 볼 수 있다.
