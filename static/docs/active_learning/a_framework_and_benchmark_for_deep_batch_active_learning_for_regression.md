# A Framework and Benchmark for Deep Batch Active Learning for Regression

## 1. Paper Overview

이 논문은 **deep neural network 기반 회귀(regression)** 문제에서, 라벨 획득 비용이 클 때 어떤 샘플을 우선적으로 라벨링할지 선택하는 **batch mode deep active learning (BMDAL)** 방법을 체계적으로 다룬다. 기존 active learning 연구는 분류(classification)에 치우쳐 있었고, 회귀용 deep batch active learning은 상대적으로 덜 연구되었으며 공통 벤치마크도 부족했다. 저자들은 이 공백을 메우기 위해, 회귀용 BMDAL 알고리즘을 **커널(kernel), 커널 변환(kernel transformation), 선택 방법(selection method)**의 조합으로 바라보는 통합 프레임워크를 제안하고, 15개의 대규모 tabular regression 데이터셋으로 구성된 공개 벤치마크를 함께 제시한다. 또한 새 선택 방법인 **LCMD**와, 기존 last-layer feature 대신 **sketched finite-width NTK**를 사용하는 방식을 제안해, 정확도와 확장성 모두에서 기존 방법들보다 우수하다고 주장한다.  

이 문제가 중요한 이유는, supervised learning에서 성능 향상을 위해 대규모 라벨 데이터가 필요한 경우가 많지만, 실제 응용에서는 라벨링 자체가 매우 비싸기 때문이다. Active learning은 “어떤 데이터를 라벨링할 것인가”를 최적화함으로써 동일한 라벨링 예산으로 더 좋은 모델을 만들려는 접근인데, 특히 회귀에서는 분류처럼 softmax uncertainty를 바로 쓰기 어려워 더 정교한 설계가 필요하다. 논문은 이 문제를 단순한 알고리즘 하나의 제안이 아니라, **비교 가능한 구성요소 수준으로 분해해 체계화**했다는 점에서 의미가 크다.

## 2. Core Idea

핵심 아이디어는 BMDAL 알고리즘을 개별 heuristic의 집합으로 보지 않고, 다음 세 요소의 조합으로 **모듈화**하는 것이다.

1. **Base kernel**: 입력 샘플 간 유사성을 나타내는 기본 커널
2. **Kernel transformations**: 이 커널을 posterior variance, correlation, projected feature space 등 active learning 목적에 맞게 변환
3. **Selection method**: 변환된 커널 또는 feature representation을 바탕으로 실제 배치 샘플을 고르는 방법

이 프레임워크는 Gaussian process 근사나 Laplace approximation에 기반한 **Bayesian 계열**뿐 아니라, diversity/geometric 관점의 **비베이지안 방법**도 함께 포괄한다. 저자들의 주장은, 많은 기존 BMAL/BMDAL 알고리즘이 사실상 이 세 요소의 다른 조합으로 재해석될 수 있다는 것이다. 이렇게 보면 서로 다른 방법들을 공정하게 비교하고, 새 조합을 설계하기가 쉬워진다.  

이 틀 위에서 논문이 추가로 제안하는 novelty는 두 가지다. 첫째, 새 selection method인 **LCMD**를 도입한다. 둘째, 흔히 쓰는 last-layer feature 대신 **finite-width neural tangent kernel (NTK)**를 base kernel로 쓰고, 여기에 **sketching**을 적용해 계산량을 줄인다. 저자들에 따르면 LCMD는 RMSE와 MAE에서 기존 최고 수준을 넘어섰고, NTK는 대부분 선택 방법에서 정확도를 높이며, sketching은 이 정확도를 거의 유지하면서 시간 비용을 줄인다.  

## 3. Detailed Method Explanation

### 3.1 문제 설정

논문은 pool-based BMDAL for regression을 다룬다. 즉, 이미 주어진 unlabeled pool 안에서 어떤 샘플들을 다음 배치로 선택할지를 결정한다. 모델은 기본적으로 **fully-connected neural network**로 두며, 목표는 회귀 함수 $f:\mathbb{R}^d \to \mathbb{R}$를 학습하는 것이다. 훈련 손실은 평균제곱오차(MSE)다. 저자들은 실험에서는 fully-connected NN과 tabular data를 사용하지만, 아이디어 자체는 다른 데이터 타입이나 네트워크 구조로도 일반화 가능하다고 말한다.

또한 저자들은 practical한 active learning method가 가져야 할 성질들을 암묵적으로 강조한다. 예를 들어 대규모 unlabeled pool과 큰 batch size에 대해 **scalable**해야 하고, downstream application마다 네트워크 구조나 학습 코드를 바꾸지 않아야 하며, hyperparameter tuning이 지나치게 필요하지 않아야 한다. 이 때문에 논문은 architecture/training code를 수정하지 않는 방법들에 집중한다.  

### 3.2 프레임워크의 구조

논문의 중심 구조는 다음처럼 이해할 수 있다.

* 먼저 neural network와 데이터로부터 어떤 **base kernel**을 만든다.
* 그런 다음 이 커널을 active learning 목적에 맞게 변환한다.
* 마지막으로 selection method를 적용해 실제 batch를 고른다.

이때 중요한 점은, 같은 “Bayesian 스타일” 커널도 꼭 Bayesian acquisition과만 결합할 필요가 없고, 비베이지안 선택 방법과도 섞어 쓸 수 있다는 것이다. 반대로 geometric selection도 Bayesian 유래 커널 위에서 돌릴 수 있다. 저자들은 이 조합 가능성이 프레임워크의 실용성과 연구적 유연성을 크게 높인다고 본다.  

### 3.3 기존 방법의 통합적 재해석

논문은 BALD, BatchBALD, BAIT, ACS-FW, Core-Set, FF-Active, BADGE 같은 잘 알려진 BMAL/BMDAL 방법들을 이 프레임워크 안에서 재구성할 수 있다고 설명한다. 즉, 서로 완전히 다른 방법처럼 보였던 것들이 사실은 “어떤 커널을 쓰는가, 어떤 변환을 쓰는가, 어떤 선택 규칙을 쓰는가”의 차이로 정리된다는 것이다. 이 점이 논문의 큰 기여다. 단순히 새 알고리즘을 하나 추가한 것이 아니라, **비교와 조합이 가능한 공통 언어**를 제공했다.  

### 3.4 LCMD

논문은 새 selection method로 **LCMD**를 제안한다. 제공된 파일 조각만으로 LCMD의 모든 수학적 세부식을 완전히 복원할 수는 없지만, 저자들은 이를 기존 인기 BMAL 알고리즘들과 비교 가능한 구성요소로 구현했으며, benchmark에서 특히 **RMSE와 MAE** 측면에서 state-of-the-art를 개선한다고 명시한다. 또한 최대오차(maximum error)에 대해서도 좋은 성능을 보인다고 설명한다. 따라서 LCMD는 단순 diversity heuristic이 아니라, **실제 회귀 품질 지표를 고려한 selection rule**로 이해하는 것이 적절하다. 세부 derivation 일부는 파일 truncation 때문에 모두 확인되지는 않는다.  

### 3.5 Sketched finite-width NTK

또 하나의 핵심은 **last-layer feature를 대체하는 base kernel로 finite-width NTK를 사용**한다는 점이다. 기존 deep active learning 계열에서는 마지막 은닉층 feature를 활용하는 경우가 많지만, 저자들은 NTK가 더 풍부한 함수 공간 정보를 줄 수 있다고 본다. 다만 NTK는 계산량이 클 수 있으므로, 이를 대규모 pool에서도 쓸 수 있도록 **sketching**을 결합한다. 논문 요약에 따르면, 이 sketching은 NTK가 주는 정확도 향상을 크게 해치지 않으면서 계산 시간을 줄인다. 즉, “더 나은 kernel”과 “실용적 계산”을 동시에 잡으려는 설계다.  

## 4. Experiments and Findings

## 4.1 벤치마크 설계

이 논문은 회귀용 BMDAL 비교를 위해 **15개의 대규모 tabular regression 데이터셋**으로 구성된 공개 벤치마크를 만든다. 이는 기존 연구들이 소수의 데이터셋, 작은 tabular regression, 혹은 특수 도메인(예: drug discovery, atomistic data)에 국한되었던 한계를 보완하려는 것이다. 저자들은 이 벤치마크를 통해 selection method, kernel choice, acquisition batch size, target metric의 영향을 체계적으로 비교한다.  

### 4.2 비교 대상

논문은 random selection과 함께 BALD, BatchBALD, BAIT, ACS-FW, Core-Set, FF-Active, BADGE 등 여러 대표적 방법과 자신들의 방법을 비교한다. Figure 1 설명에 따르면, 실험은 초기 256개의 랜덤 training sample에서 시작해, 16번의 BMAL step마다 256개씩 새 샘플을 선택하는 방식으로 진행되며, 15개 데이터셋과 20개 random split에 대해 **log-RMSE 평균**을 비교한다. 이 설정은 방법 간 label budget 대비 성능 향상을 정량적으로 보여준다.  

### 4.3 주요 결과

논문이 가장 강하게 주장하는 결과는 다음과 같다.

* **LCMD**가 benchmark에서 **RMSE와 MAE 기준 SOTA를 개선**했다.
* maximum error 기준으로도 성능이 좋았다.
* **NTK base kernel**은 모든 selection method에서 benchmark accuracy를 향상시켰다.
* **Sketching**은 이 정확도 향상을 유지하면서 계산 시간을 줄였다.
* 전체적으로 제안 방법은 large data set에서 잘 확장되고, 네트워크 구조나 학습 코드를 바꾸지 않고도 쓸 수 있었다.  

즉, 논문의 메시지는 단순히 “LCMD가 좋다”가 아니라, **좋은 selection rule + 좋은 kernel choice + 효율적 근사**의 조합이 회귀용 BMDAL에서 실제로 효과적이라는 것이다. 특히 기존 분류 중심 active learning 문헌에서 자주 쓰이던 아이디어를 회귀에 그대로 적용하는 것보다, 회귀 특성에 맞는 kernelized formulation이 더 유리하다는 인상을 준다.

### 4.4 실험이 보여주는 것

Figure 1 설명을 보면, 제안 방법은 여러 BMAL 단계에 걸쳐 평균 오류를 더 빠르게 낮춘다. 이는 두 가지를 시사한다. 첫째, 같은 라벨링 예산에서 더 좋은 샘플을 선택한다는 뜻이다. 둘째, 초기 소량 라벨 구간뿐 아니라 여러 acquisition round 전체에 걸쳐 이점이 유지된다는 뜻이다. Active learning에서는 초기에만 조금 좋은 방법보다, 여러 라운드 동안 누적적으로 이득을 주는 방법이 더 중요하므로 이 결과는 실질적 의미가 있다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **통합 프레임워크**다. 많은 active learning 논문은 알고리즘별 설명에 머무르는데, 이 논문은 커널·변환·선택법으로 분해해 서로 다른 방법을 공통 구조 안에서 비교 가능하게 만들었다. 이는 단순한 벤치마크 보고서보다 연구적 가치가 크다.  

둘째, **benchmark의 공헌**이 크다. 회귀용 BMDAL에는 표준 벤치마크가 부족했는데, 15개 대규모 tabular regression dataset을 정리해 공개했다는 점은 재현성과 후속 연구에 직접적인 도움이 된다.  

셋째, 성능뿐 아니라 **실용성**을 의식했다는 점도 강점이다. 저자들은 architecture/training code 수정이 필요 없는 방법만 주로 다루고, large pool과 large batch size에 대해 확장 가능한 방법을 선호한다. 또한 공개 코드와 데이터 아카이브를 함께 제공해 재현 가능성을 높였다.  

### 한계

가장 분명한 한계는 실험 초점이 **fully-connected NN + tabular regression**에 맞춰져 있다는 점이다. 저자들도 방법이 일반화 가능하다고 말하지만, 논문에서 실제로 광범위하게 검증한 것은 tabular setting이다. 따라서 이미지/시계열/graph 회귀처럼 구조가 다른 데이터에서도 같은 우위가 유지되는지는 추가 검증이 필요하다.  

둘째, 업로드된 ar5iv HTML 파일이 중간 이후 **fatal conversion error**로 잘려 있어, 일부 수식적 세부사항과 후반부 결론/한계 논의 전부를 완전하게 확인하기는 어렵다. 따라서 LCMD의 모든 알고리즘적 세부나 일부 appendix 수준 분석은 현재 파일만으로는 100% 복원되지 않는다. 이 보고서에서 그런 부분은 추측하지 않고, 확인 가능한 범위만 반영했다.  

셋째, 방법이 “out-of-the-box”임을 강조하지만, 실제 application에서 label noise, distribution shift, non-tabular feature structure, 혹은 매우 작은 batch size/매우 큰 batch size 극단 설정에서 어떻게 동작하는지는 추가 확인이 필요하다. 이는 논문 본문에서 직접 충분히 다뤄진 부분은 아니며, 본 보고서의 비판적 해석이다.

### 해석

이 논문은 회귀 active learning에서 “uncertainty만 보면 된다”는 단순한 시각을 넘어, **representation/kernel choice와 batch selection strategy를 동시에 설계해야 한다**는 메시지를 준다. 특히 Bayesian/비베이지안 구분을 절대적인 경계로 보지 않고, kernelized common framework로 묶었다는 점이 인상적이다. 결과적으로 이 논문은 한 개의 알고리즘 논문이라기보다, **회귀용 deep batch active learning의 설계 공간을 정리한 시스템 논문**에 가깝다.  

## 6. Conclusion

이 논문은 deep regression을 위한 batch active learning을 다루며, 세 가지 핵심 공헌을 한다. 첫째, 기존 BM(D)AL 방법들을 **kernel + transformation + selection**의 조합으로 보는 통합 프레임워크를 제안했다. 둘째, 새로운 selection method인 **LCMD**와 **sketched finite-width NTK**를 도입해 정확도와 효율성을 함께 개선했다. 셋째, **15개 대규모 tabular regression 데이터셋 벤치마크**를 제공해 향후 연구를 위한 공통 비교 기반을 마련했다.  

실무적으로는, 회귀 태스크에서 라벨링이 비싼 상황에서 architecture를 크게 바꾸지 않고도 쓸 수 있는 active learning 조합을 제공한다는 점이 중요하다. 연구적으로는 회귀용 deep active learning을 classification의 부속물이 아니라 독립된 문제로 다루고, 비교 가능한 프레임워크와 공개 벤치마크를 마련했다는 데 의의가 있다. 다만 현재 업로드된 HTML이 중간에서 손상되어 있어, 논문 후반의 제한점/미래과제 일부는 완전하게 확인되지 않았다.  
