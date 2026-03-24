# Continual Learning in Practice

* **저자**: Tom Diethe, Tom Borchert, Eno Thereska, Borja Balle, Neil Lawrence
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1903.05202](https://arxiv.org/abs/1903.05202)

## 1. 논문 개요

이 논문은 새로운 continual learning 알고리즘을 제안하는 전형적인 모델 논문이라기보다, 실제 운영 환경의 머신러닝 시스템이 데이터를 지속적으로 받아들이면서 스스로 유지보수되고 적응할 수 있도록 하는 **reference architecture**를 제안하는 시스템 관점의 논문이다. 저자들이 강조하는 핵심 문제는 매우 분명하다. 오늘날 많은 ML 시스템은 학습 시점에는 잘 동작하지만, 배포 이후 시간이 지나면서 입력 데이터 분포가 변하고, 레이블이 늦게 도착하며, 다운스트림 시스템과의 상호작용이 누적되면서 처음의 검증이 더 이상 유효하지 않게 된다. 다시 말해, 전통적인 “train once, deploy once” 방식은 **비정상적(non-stationary)이고 계속 변하는 실제 세계**를 충분히 반영하지 못한다.

논문은 이 문제를 “continual AutoML” 또는 “Automatically Adaptive Machine Learning”이라는 관점으로 재정의한다. 여기서 목표는 단순히 좋은 모델을 자동으로 찾는 것이 아니라, 이미 운영 중인 모델이 **언제 성능이 저하될 위험이 있는지 감지하고**, **언제 재학습해야 하는지 판단하며**, **새 모델을 안전하게 배포하고**, **필요하면 롤백까지 할 수 있도록 하는 전체 시스템**을 구성하는 것이다. 즉, 연구 문제는 “continual learning 알고리즘이 무엇인가”보다 더 실무적이다. 저자들은 “운영 중인 ML 시스템을 지속적으로 신뢰할 수 있게 만들기 위한 시스템 구조는 어떻게 설계되어야 하는가?”를 묻고 있다.

이 문제는 산업적으로 매우 중요하다. 실제 서비스에서는 데이터 분포가 고정되어 있지 않고, 계절성, 사용자 행동 변화, 센서 환경 변화, 비즈니스 정책 변화, 이상치 유입, 피드백 지연 등으로 인해 모델 성능이 서서히 혹은 급격히 나빠질 수 있다. 그런데 기존 소프트웨어 엔지니어링의 테스트와 배포 절차는 주로 코드 변경을 전제로 하며, 데이터 변화로 인한 성능 저하는 충분히 다루지 못한다. 따라서 저자들은 ML 운영을 위해 기존의 regression testing에 대응하는 개념으로 **progression testing**을 제안하고, 이를 중심으로 한 운영 아키텍처를 설명한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 다음과 같다. **배포된 ML 모델을 소프트웨어 컴포넌트 하나로 보는 관점만으로는 부족하며, 모델 위에 이를 감시하고 재학습을 결정하고 자원을 관리하는 상위 제어 계층, 즉 “hypervisor”가 필요하다.** 저자들은 이 상위 계층을 포함한 전체 구조를 Auto-Adaptive ML architecture로 설명한다.

기존 AutoML은 보통 데이터와 탐색 공간이 주어졌을 때 학습 알고리즘과 하이퍼파라미터를 자동으로 찾는 데 초점을 둔다. 반면 이 논문이 말하는 Auto-Adaptive ML은 배포 이후를 본다. 즉, 입력 데이터가 변하고, 모델 출력이 다른 시스템의 입력이 되며, 재학습 비용과 위험이 존재하고, 레이블이 지연되는 환경에서 전체 수명주기를 다루어야 한다. 이 점이 기존 배포 플랫폼이나 AutoML과의 가장 중요한 차별점이다.

또 하나의 핵심 직관은 **시스템을 스트리밍 기반으로 설계해야 한다**는 것이다. 배치 처리 시스템은 스트리밍 상황을 완전히 대체하기 어렵지만, 스트리밍 시스템은 필요에 따라 배치처럼 운용할 수 있다. 따라서 지속적으로 들어오는 데이터와 지연 레이블을 처리하고, 모니터링과 정책 결정을 실시간에 가깝게 수행하려면 스트리밍 중심 설계가 유리하다고 본다.

논문은 또한 ML 시스템이 일반 소프트웨어보다 훨씬 더 강하게 얽혀(entangled) 있다는 점을 강조한다. 모델의 출력이 후속 모델의 입력이 될 수 있고, 외부 제어 변수와 상호작용하며, 숨은 feedback loop가 발생할 수 있다. 그래서 “새 모델이 오프라인 평가에서 더 좋아 보인다”는 이유만으로 안전하게 교체할 수 없으며, 운영 중 검증과 정책적 판단이 필수라는 점을 설계 철학으로 제시한다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인의 구조

논문이 제안하는 구조는 하나의 단일 알고리즘이 아니라 여러 서브시스템이 연결된 모듈형 아키텍처이다. 큰 흐름은 다음과 같다.

입력 데이터는 스트림으로 들어오고, 먼저 필요에 따라 **sketcher/compressor**를 거친다. 이 구성요소는 데이터 유량이 너무 커서 다운스트림 학습기가 모든 데이터를 그대로 처리하기 어려울 때, 정보를 최대한 보존하면서 데이터를 축약한다. 이후 **joiner**가 지연된 레이블이나 외부 피드백을 입력 스트림과 정렬하여 trainer와 predictor가 사용할 수 있는 형식으로 맞춘다.

병렬적으로 **data monitoring subsystem**은 입력 데이터 스트림을 보고 anomaly, drift, change-point를 탐지한다. **prediction monitoring subsystem**은 모델 예측과 시스템 health 관련 스트림을 보고 현재 시스템 상태를 추정한다. 이 두 모니터링 결과와 비즈니스 로직을 바탕으로 **policy engine**이 재학습, 기존 모델 유지, 롤백, 다른 학습 전략 적용 등의 행동을 결정한다.

한편 **trainer subsystem**은 실제 모델 학습을 수행하고, 필요하면 HPO(Hyper-Parameter Optimization)를 통해 warm-start된 재학습을 수행한다. **predictor subsystem**은 새로 훈련된 모델을 배포하고 실시간 예측을 생성한다. 전체 과정에서 필요한 모델, 로그, 검증용 샘플, 시스템 상태 등은 **shared infrastructure**에 저장된다.

이 구조의 중요한 특징은 강한 모듈성이다. 저자들은 모든 요소를 한 번에 완성해야 한다고 주장하지 않는다. 오히려 기존 운영 파이프라인 위에 단계적으로 얹을 수 있는 구조로 제시한다.

### 3.2 Sketcher/Compressor

이 구성요소는 대용량 스트리밍 데이터에서 계산과 저장 비용을 줄이는 역할을 한다. 가장 단순한 방식은 uniform down-sampling이다. 논문은 표본 크기 $s$일 때 표준오차가 대략 $s^{-1/2}$에 비례한다고 설명한다. 즉,

$$
\text{standard error} \propto s^{-1/2}
$$

이므로, 예를 들어 1000개 샘플이면 대략 3% 수준의 오차를 기대할 수 있다는 식의 직관을 준다. 이는 baseline일 뿐이고, 실제로는 더 나은 sketching 방법들이 가능하다고 말한다.

부록 A에서는 다양한 sketching 기법을 소개한다. 대표적으로 Bloom filter는 membership query에 적합하고, false negative 없이 작은 false positive를 허용한다. 논문에 따르면 Bloom filter의 false positive 확률은 대략

$$
\left(1 - e^{-kn/m}\right)^k
$$

로 표현된다. 여기서 $n$은 저장된 항목 수, $m$은 비트 배열 크기, $k$는 해시 함수 개수이다. 이 식은 운영 환경에서 메모리와 오차를 어떻게 맞바꿀지 설명하는 데 사용된다.

또한 count-min sketch는 스트림 빈도 추정, heavy hitter 탐지, range query 근사에 쓰일 수 있고, HyperLogLog는 cardinality estimation에 적합하다. t-digest는 quantile 같은 rank-based statistic 계산에 유용하다. 논문의 메시지는 특정 sketch 하나를 강하게 주장하는 것이 아니라, **스트림 환경에서는 다운스트림 학습과 모니터링이 감당할 수 있도록 데이터를 압축하는 계층이 필요하다**는 것이다.

### 3.3 Joiner

실제 서비스에서는 입력이 들어오는 시점에 정답 레이블이 바로 붙지 않는 경우가 많다. 추천, 광고, 수요예측, 고장탐지 같은 문제에서는 정답이 몇 분, 몇 시간, 혹은 더 오랜 시간이 지나야 확인될 수 있다. 이때 joiner는 입력과 나중에 도착한 레이블 또는 약한 피드백을 연결해 trainer와 predictor가 사용할 수 있는 훈련/평가 형식으로 만든다.

즉, joiner는 단순한 데이터 결합기가 아니라, **지연 피드백 환경에서 학습 가능한 데이터셋을 재구성하는 실무상 핵심 계층**이다. 스트리밍 데이터는 늦게 도착하거나, 순서가 뒤바뀌거나, 일부가 누락될 수도 있으므로, 이 문제를 완화하는 역할도 맡는다.

### 3.4 Shared Infrastructure

논문은 shared infrastructure를 일종의 지능형 cache처럼 본다. 무한 저장소를 가정하지 않고, 어떤 정보를 얼마나 오래 저장할지 결정하는 논리를 포함해야 한다고 본다. 여기에는 다음과 같은 저장 대상이 포함된다.

먼저 **Model DB**는 훈련된 모델을 저장한다. 이는 특정 시점의 의사결정이 왜 그렇게 나왔는지 추적하는 provenance와, 잘못된 업데이트가 발생했을 때 rollback을 가능하게 한다. **Training DB**는 훈련 이력과 메타데이터를 저장한다. **Validation Data Reservoir**는 학습 중 검증과 과거 모델 재현에 사용된다. 논문은 이 저장소로 Adaptable Damped Reservoir를 언급하며, 최근 데이터에 더 큰 가중을 두는 샘플 유지 전략을 시사한다. 이 구조는 오프라인 A/B testing이나 재현성 확보에도 도움이 된다. **System State DB**는 policy engine이 의사결정을 내리는 데 필요한 현재 상태를 담고, **Diagnostic Logs**는 전체 디버깅과 감사를 위한 기록을 남긴다.

여기서 중요한 것은 저장 자체보다 **운영 가능한 증적(provenance)과 재현성(reproducibility)**이다. continual adaptation에서는 모델이 수시로 바뀌므로, 과거 특정 판단을 설명할 수 있어야 한다는 점이 매우 중요하다.

### 3.5 Data Monitoring: Progression Testing의 첫 축

논문은 현실 세계가 non-stationary한데, 대부분의 배포 모델은 정적 분포를 가정한다고 지적한다. 따라서 입력 분포가 학습 시점과 달라졌는지 탐지하는 것이 필요하다. 이 subsystem의 최소 출력은 다음 세 가지다. 첫째, shift의 종류. 둘째, shift의 크기. 셋째, 불확실성의 정량화이다. 여기에 가능하면 어떤 feature가 shift를 주도했는지에 대한 explanation도 필요하다고 한다.

부록 B는 dataset shift 유형을 정리한다. source와 target 데이터셋을 각각 $D_S$, $D_T$라 하고, 이들이 분포 $p_S(x,y)$, $p_T(x,y)$에서 나왔다고 할 때, 두 분포 간 차이를 감지하는 것이 목표다. 여기에는 covariate shift, prior probability shift, sample selection bias, domain shift, source component shift, anomaly detection 등이 포함된다.

탐지 방법은 supervised와 unsupervised로 나뉜다. supervised 방식은 progressive error나 holdout set을 활용하지만, 운영 환경에서는 레이블이 항상 즉시 사용가능하지 않으므로 한계가 있다. 그래서 practical한 상황에서는 통계적 거리, histogram intersection, KL divergence 기반 접근, density-ratio estimation, random cut forest, changepoint detection 등이 중요하다고 설명한다. 저자들은 특정 탐지기 하나를 채택하지 않고, **어떤 종류의 shift가 있는지 파악하고 그에 따라 재학습 또는 적응 전략을 연결하는 상위 시스템 설계**에 초점을 둔다.

### 3.6 Prediction Monitoring

prediction monitoring subsystem은 입력 자체가 아니라 **예측 결과와 시스템 상태**를 본다. 논문 설명에 따르면, 여기에 포함될 수 있는 상태 변수는 데이터 분포 변화 추정, 마지막 재학습 이후 경과 시간, 현재 날짜/시간, 재학습 비용, 예측의 비즈니스 가치, 현재 정책, 상태-행동 쌍의 가치 추정 등이다.

즉, 이 subsystem은 단순 정확도 모니터가 아니라, policy engine이 행동을 결정할 수 있도록 **“현재 세계의 상태(state)”를 구성하는 계층**이다. 이 state는 강화학습 기반 정책으로 이어질 수 있다.

### 3.7 Trainer, HPO, Predictor

trainer subsystem은 기존 팀이 이미 가지고 있는 ML 훈련 파이프라인을 감싸는 wrapper로 생각하면 된다. 저자들은 완전히 새로운 학습기를 요구하지 않는다. 오히려 기존 학습 파이프라인에서 재학습 단계를 자동화하고, 학습 메타데이터를 저장하도록 구조화하는 데 초점을 둔다.

여기서 저장되어야 할 메타데이터에는 validation metric, 시작 시각, 학습 소요 시간, hyperparameter setting, 훈련된 모델 세부 정보 등이 포함된다. 또한 HPO를 통해 이전 학습 결과를 활용한 warm-start가 가능하다고 한다. 이는 drift가 발생할 때 매번 처음부터 완전 탐색하는 대신, 기존 좋은 설정을 출발점으로 재최적화하는 실용적 전략이다.

predictor subsystem은 새 모델을 실제 서비스에 배포하고, trainer가 생산한 연속적인 model stream을 구독한다. 즉, 학습과 배포가 분리되어 있지만 긴밀히 연결되는 구조다.

### 3.8 Model Policy Engine

논문의 시스템적 핵심은 policy engine이다. 이 엔진은 언제 재학습할지, 기존 모델을 유지할지, 롤백할지 등을 결정해야 한다. 저자들은 이 문제를 네 가지 관점에서 설명한다.

첫째는 **horizon**이다. 가장 최근 데이터가 얼마나 빨리 모델에 반영되어야 하는지, 그리고 어떤 과거 데이터가 여전히 유효한지를 결정해야 한다. 경우에 따라 최근 데이터가 가장 중요할 수도 있지만, 리테일처럼 전년 동기 패턴이 중요한 경우도 있다.

둘째는 **cadence**이다. 언제 재학습할 것인지 결정해야 한다. 이는 현재 상태를 정확히 추정하고, 가능한 행동 집합을 정의한 뒤, 비용과 미래 이익을 균형 있게 고려해야 한다는 뜻이다.

셋째는 **provenance**이다. 의사결정의 근거를 추적할 수 있어야 한다. 따라서 comprehensive logging과 health monitoring이 필요하다.

넷째는 **costs**이다. 재학습의 운영 비용과 재학습하지 않았을 때의 리스크를 모두 평가해야 한다.

가장 단순한 baseline은 rule-based policy다. 예를 들어 “PSI가 임계값을 넘으면 재학습” 같은 규칙일 수 있다. 하지만 일반적으로는 policy 자체도 학습 가능해야 하며, 저자들은 이를 강화학습(Reinforcement Learning) 문제로 본다. 이때 state space는 data monitoring, trainer, prediction monitoring의 출력으로 구성되고, action space는 policy engine의 행동 집합이며, reward는 business metric 기반으로 설계된다. 논문은 이를 개념적으로 제시하지만, 구체적 상태 정의나 보상 함수 형태는 상세히 기술하지 않는다. 따라서 여기서의 RL은 구현된 알고리즘 제안이라기보다 향후 연구 방향이다.

## 4. 실험 및 결과

이 논문은 일반적인 의미의 benchmark 실험 논문이 아니다. 사용자가 제공한 텍스트 기준으로 보면, 특정 데이터셋에서 특정 baseline과 수치 비교를 수행하는 정량 실험은 제시되지 않는다. 즉, MNIST, CIFAR, ImageNet 같은 데이터셋을 사용한 학습 성능 비교도 없고, 제안 architecture 전체를 end-to-end로 구현해 성능을 검증한 표나 그래프도 없다.

대신 논문은 각 구성요소의 필요성과 설계 rationale을 설명하고, 부록에서 sketching 및 dataset shift detection의 기존 기법들을 정리한다. 따라서 이 섹션에서 말할 수 있는 “결과”는 실험 결과라기보다 **아키텍처 제안의 내용적 산출물**에 가깝다.

첫째, 논문은 production ML에서 중요한 품질 관리 문제를 progression testing이라는 프레임으로 재구성한다. 이는 배포 시점 검증만으로는 부족하며, 데이터가 계속 변하는 한 테스트와 검증도 지속적으로 다시 수행되어야 한다는 주장이다.

둘째, sketcher, joiner, monitoring subsystem, policy engine, trainer/predictor, shared infrastructure로 구성된 모듈형 참조 구조를 제시함으로써, continual learning을 단일 알고리즘이 아닌 **운영 시스템 문제**로 정식화한다.

셋째, 재학습 주기 결정 문제를 단순한 cron job이 아니라 비용-편익 균형의 sequential decision 문제로 보고, 강화학습 기반 정책 최적화 가능성을 열어둔다.

정량 결과가 없다는 점은 이 논문의 성격을 이해하는 데 중요하다. 이 논문은 “제안 방법이 기존 방법보다 몇 % 더 좋다”를 보여주는 empirical paper가 아니라, 실제 시스템 설계와 연구 과제를 조직적으로 정리하는 **position/reference architecture paper**에 가깝다. 따라서 실험적 증거를 기대하면 부족해 보일 수 있지만, 반대로 연구 및 MLOps 설계의 출발점으로는 유용하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 continual learning을 좁은 알고리즘 문제가 아니라 **배포 이후 ML 수명주기 전체의 시스템 문제**로 본다는 점이다. 많은 논문이 catastrophic forgetting, online update, transfer와 같은 학습 알고리즘 내부 문제에 집중하는 반면, 이 논문은 데이터 유입, 피드백 지연, 모니터링, 정책, provenance, rollback, 비용 관리까지 포함한다. 실제 산업 환경에서 훨씬 현실적인 문제 설정이다.

또 다른 강점은 모듈성이 높다는 것이다. sketcher, joiner, monitoring, policy engine, trainer, predictor, shared infrastructure를 분리해 설명하므로, 기존 시스템을 운영 중인 팀도 모든 것을 한 번에 교체하지 않고 단계적으로 채택할 수 있다. 이 점은 실무 적용 가능성을 높인다.

또한 monitoring을 매우 구체적으로 다룬다는 점도 장점이다. 단순히 “drift를 탐지해야 한다”고 끝내지 않고, shift의 종류, 크기, 불확실성, explanation까지 포함해야 한다고 제시하며, 입력 수준의 data monitoring과 출력 수준의 prediction monitoring을 구분한다. 이는 production ML의 관찰 가능성(observability)을 체계적으로 정리한 것으로 볼 수 있다.

그러나 한계도 분명하다. 가장 큰 한계는 **실험적 검증의 부재**이다. 제공된 텍스트 기준으로는 architecture 전체를 실제로 구현해 효과를 검증한 empirical evidence가 없다. 따라서 제안된 구조가 실제로 재학습 비용을 줄이는지, 장애를 줄이는지, 성능 유지에 얼마나 기여하는지는 논문만으로는 판단하기 어렵다.

둘째, policy engine의 핵심인 의사결정 문제가 개념 수준에 머물러 있다. 강화학습을 언급하지만, state를 어떻게 정의할지, reward를 어떻게 구성할지, exploration으로 인한 서비스 리스크를 어떻게 통제할지 등은 구체적이지 않다. 저자들도 이를 active area of research라고 인정한다.

셋째, 논문은 “어떤 deployed ML model도 지원해야 한다”고 말하지만, 실제로는 모델 유형마다 drift 양상, 피드백 지연, 재학습 비용, 배포 제약이 크게 다르다. 따라서 제안 구조는 강한 일반성을 주장하지만, 그만큼 구체성이 희석된 측면이 있다.

넷째, continual learning이라는 제목을 보면 독자는 online parameter update, memory replay, catastrophic forgetting mitigation 같은 알고리즘적 continual learning을 기대할 수 있다. 그러나 실제 내용은 그보다 훨씬 넓은 MLOps/AutoML architecture에 가깝다. 이것은 장점이기도 하지만, 엄밀한 의미의 continual learning 알고리즘 논문을 기대한 독자에게는 불일치를 줄 수 있다.

비판적으로 보면, 이 논문은 “정답을 주는 논문”이라기보다 “올바른 질문을 구조화하는 논문”이다. 즉, 해결책 전체를 완성했다기보다 production continual learning이 실제로 풀어야 할 문제들을 잘 분해해 보여준다. 실증보다는 청사진에 가깝다는 점을 감안해 읽는 것이 적절하다.

## 6. 결론

이 논문은 지속적으로 변화하는 실제 환경에서 ML 모델을 안정적으로 운영하기 위해서는, 단순한 모델 배포를 넘어 **자기 진단(self-diagnosis), 자기 수정(self-correction), 자기 관리(self-management)** 기능을 갖춘 Auto-Adaptive ML architecture가 필요하다고 주장한다. 이를 위해 스트리밍 기반 데이터 처리, sketcher/compressor, joiner, data/prediction monitoring, shared infrastructure, trainer/predictor, policy engine으로 구성된 참조 구조를 제안한다.

핵심 기여는 새로운 학습 알고리즘 하나를 만드는 데 있지 않다. 오히려 배포된 ML 시스템이 실제로 겪는 drift, outlier, delayed feedback, retraining cadence, provenance, rollback, 비용 통제 문제를 하나의 운영 프레임워크 안에서 설명했다는 데 있다. 특히 progression testing이라는 관점은, 시간이 지나면서 무효화되는 기존 검증 체계를 continual environment에 맞게 다시 생각하게 만든다는 점에서 의미가 있다.

실제 적용 관점에서 보면, 이 연구는 오늘날 MLOps, model monitoring, continual retraining, adaptive inference pipeline 설계의 선행적 청사진으로 읽을 수 있다. 향후 연구에서는 이 아키텍처의 각 구성요소를 실제 시스템으로 구현하고, 재학습 정책 최적화, drift 탐지의 신뢰성, provenance 기반 rollback 전략, 비용 대비 성능 개선 효과를 정량적으로 검증하는 작업이 중요할 것이다. 그런 의미에서 이 논문은 완결된 해답이라기보다, **production-grade continual learning을 향한 설계 원칙과 연구 의제를 제시한 출발점**으로 평가할 수 있다.
