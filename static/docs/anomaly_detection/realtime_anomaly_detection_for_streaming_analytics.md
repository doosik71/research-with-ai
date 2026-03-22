# Real-Time Anomaly Detection for Streaming Analytics

* **저자**: Subutai Ahmad, Scott Purdy
* **발표연도**: 2016
* **arXiv**: <https://arxiv.org/abs/1607.02480>

## 1. 논문 개요

이 논문은 **실시간 스트리밍 시계열 데이터에서 이상(anomaly)을 온라인으로 탐지하는 방법**을 제안한다. 저자들은 많은 산업 시스템이 배치 처리보다 훨씬 더 어려운 조건, 즉 **미래를 볼 수 없는 상태에서 즉시 판단해야 하고**, **데이터 분포가 계속 변하며**, **사람이 수동으로 임계값을 조정하기 어렵다**는 현실적 제약을 강조한다. 이러한 조건에서 단순 threshold, change-point detection, ARIMA 계열, 또는 고정된 규칙 기반 방법은 공간적 이상은 잡을 수 있어도 시간적 패턴의 미묘한 변화를 놓치거나, 반대로 false positive가 많아질 수 있다고 본다. 이 문제를 해결하기 위해 논문은 **HTM(Hierarchical Temporal Memory)**을 기반으로 한 예측 중심 anomaly detection 체계를 제안한다.

연구 문제는 비교적 명확하다. **실시간성**, **지속적 학습(continuous learning)**, **비정상성(non-stationarity)**, **잡음(noise)에 대한 견고성**, 그리고 **temporal anomaly 탐지**를 동시에 만족하는 일반 목적의 탐지기를 만드는 것이다. 논문은 이를 단일 센서 스트림뿐 아니라 여러 센서/지표를 동시에 다루는 복합 시스템까지 확장하려고 한다. 실제 금융 지표 모니터링 사례와 NAB(Numenta Anomaly Benchmark) 실험을 통해, 제안법이 질적·양적으로 유효하다고 주장한다.

이 문제가 중요한 이유는 논문이 예시로 드는 산업 장비 고장, 데이터센터 모니터링, 금융/소셜 지표 감시처럼 **이상이 조기에 탐지되면 큰 손실을 줄일 수 있는 상황**이 매우 많기 때문이다. 특히 논문은 값 자체가 극단적이지 않아도 **“순서가 이상한” temporal anomaly** 가 더 이른 경고 신호가 될 수 있다고 본다. 이는 단순히 값의 크기만 보는 고전적 방법과 구별되는 핵심 관점이다.

## 2. 핵심 아이디어

핵심 직관은 매우 단순하게 말하면 이렇다. **이상은 “예측이 어긋나는 정도”로 측정할 수 있다.** HTM은 입력 시계열의 현재 상태뿐 아니라 그 상태가 어떤 sequence 문맥 속에 있는지까지 표현하면서 다음 입력을 예측한다. 그러므로 현재 입력이 HTM이 예상한 패턴과 잘 맞으면 정상, 맞지 않으면 이상에 가깝다고 해석할 수 있다. 논문은 이 예측 오차를 바로 최종 알람으로 쓰지 않고, 먼저 **raw anomaly score** 를 계산한 뒤, 다시 그 score의 최근 분포를 바탕으로 **anomaly likelihood** 를 구하는 2단계 구조를 사용한다.

이 설계의 중요한 차별점은 두 가지다. 첫째, 이상 판단 기준을 **원시 metric 값의 분포** 에 두지 않고 **“예측 가능성의 변화”** 에 둔다. 그래서 noisy한 데이터에서도 값이 조금 튀었다고 바로 이상으로 보지 않고, **예측 불가능성이 지속적으로 커지는지** 를 본다. 둘째, 여러 모델을 동시에 운영할 때 단순 동시 발생만 보는 것이 아니라 **가까운 시간대에 발생한 개별 이상 신호들을 Gaussian temporal window로 부드럽게 합성** 한다. 즉, 서로 정확히 같은 시각에 일어나지 않아도, 복합 시스템에서 연쇄적으로 나타나는 이상 조짐을 한데 묶어 탐지하려는 것이다.

또 하나의 장점은 HTM 자체가 **continuous online learning** 특성을 갖는다는 점이다. 논문은 시스템 동작이 바뀌면 raw anomaly score가 처음에는 크게 오르지만, 시간이 지나 모델이 새로운 패턴을 학습하면 다시 score가 낮아진다고 설명한다. 즉, “새로운 정상(new normal)”에 적응할 수 있다는 것이다. 이는 distribution shift가 잦은 실시간 운영 환경에서 특히 중요하다.

## 3. 상세 방법 설명

논문의 전체 파이프라인은 다음과 같이 이해하면 된다. 스트리밍 입력 $x_t$ 가 들어오면, HTM은 이를 sparse representation으로 인코딩하고, 내부적으로 다음 시점 입력에 대한 예측 상태를 유지한다. 그 다음 실제 입력과 예측된 활성 패턴의 일치 정도를 이용해 **raw anomaly score** 를 만든다. 마지막으로 최근 raw score들의 통계적 분포를 사용해 현재 score가 얼마나 이례적인지를 **anomaly likelihood** 로 계산하고, 이것이 충분히 높을 때 최종 anomaly alert를 발생시킨다. 논문 그림 3은 바로 이 흐름을 블록 다이어그램으로 보여준다.

### 3.1 HTM 내부 표현과 raw anomaly score

논문에 따르면 HTM은 현재 입력을 나타내는 sparse binary vector와, 다음 입력에 대한 예측을 나타내는 내부 prediction vector를 사용한다. 저자들은 이 두 표현의 **교집합 정도** 로 현재 입력이 얼마나 잘 예측되었는지를 측정한다. 원문 HTML에서는 수식 기호가 일부 누락되어 있지만, 설명 자체는 분명하다. **현재 입력이 완벽히 예측되면 raw anomaly score는 0에 가깝고, 전혀 예측되지 못하면 1에 가깝다.** 중간 정도로 맞으면 그 사이 값을 가진다. 즉, raw score는 “현재 관측이 모델의 기대와 얼마나 어긋났는가”를 1-step 수준에서 정량화한 값이다.

여기서 논문이 강조하는 포인트는 **branching sequence** 처리다. 어떤 시점 이후 가능한 다음 상태가 여러 개인 경우, HTM은 여러 예측을 sparse union 형태로 동시에 표현할 수 있다. 그래서 “둘 중 어느 하나가 와도 정상인” 상황을 자연스럽게 표현한다. 이런 경우 실제 입력이 그 예측 집합 안에 들어오면 raw anomaly는 낮게 나오고, 완전히 다른 패턴이 들어오면 높게 나온다. 이 설명은 단순 ARIMA나 이동 평균 기반 방법보다 sequence context를 더 풍부하게 다룬다는 저자들의 주장과 연결된다.

또한 이 raw score는 **개별 값이 비정상인지(spatial anomaly)**뿐 아니라 **정상 범위 안의 값이라도 순서가 비정상인지(temporal anomaly)** 를 반영할 수 있다. 논문 초반과 결과 예시에서 저자들이 반복해서 강조하듯, 실제 산업 시스템에서는 catastrophic failure 이전에 이런 temporal anomaly가 먼저 나타날 수 있다. HTM의 예측 기반 설계는 바로 이 지점을 노린다.

### 3.2 anomaly likelihood

논문은 raw anomaly score만 바로 thresholding하면 noisy stream에서 false positive가 많아진다고 본다. 예를 들어 웹사이트 load balancer latency처럼 원래부터 변동성이 큰 스트림에서는 단발성 spike가 흔하다. 이런 환경에서는 “값이 튀었는가”보다 **“최근에 예측 실패가 평소보다 유의미하게 자주 발생하고 있는가”** 가 더 중요하다. 이 때문에 저자들은 최근 raw anomaly score들의 window를 유지하고, 그 위에서 **rolling mean과 rolling variance** 를 계속 업데이트한다.

그 다음 최근 짧은 구간의 평균 raw score가 과거 분포에 비해 얼마나 큰지를 Gaussian tail probability, 즉 Q-function으로 평가한다. 그리고 그 tail probability의 complement를 **anomaly likelihood** 로 정의한다. 직관적으로는, “최근 예측 실패 수준이 지금까지 보아 온 정상적 변동으로 설명되기 어려울수록 likelihood가 1에 가까워진다”는 뜻이다. 논문은 이 likelihood가 1에 매우 가까울 때 anomaly로 판정한다. 본문에는 분포 추정용 긴 window와 최근 평균을 위한 짧은 window를 구분해 사용한다고 설명되어 있으며, 실제 실험에서는 긴 window를 넉넉히 잡고 짧은 window는 그보다 훨씬 짧게 둔다.

이 설계의 의미는 중요하다. 여기서 thresholding 대상은 원래 시계열 값 $x_t$ 의 분포가 아니라 **anomaly score의 분포** 다. 따라서 데이터 자체가 Gaussian일 필요도 없고, 센서별 단위나 스케일이 달라도 직접적으로 크게 문제되지 않는다. 저자들은 이것이 실세계 스트리밍 데이터의 “지저분함(messy)”에 더 잘 맞는다고 주장한다. 또한 한 번의 큰 spike보다는 **짧은 시간 동안 반복되는 예측 실패** 에 더 민감해지므로 false positive를 줄이는 효과를 노린다.

### 3.3 다중 모델 결합

복잡한 환경에서는 모든 센서를 하나의 거대한 모델에 넣기보다 여러 개의 작은 모델로 나누는 것이 현실적이라고 논문은 말한다. 차원이 커질수록 학습·추론 복잡도가 급격히 커지기 때문이다. 대신 여러 모델의 raw anomaly score를 어떻게 **전역 이상도(global anomaly likelihood)**로 합칠지가 새로운 문제가 된다. 저자들은 독립 모델 가정을 바탕으로 각 모델의 anomaly likelihood를 결합하는 기본식을 세우고, 여기에 **Gaussian convolution kernel** 을 적용해 시간적으로 가까운 개별 이상 신호들을 부드럽게 합친다.

이 아이디어의 배경은 실시간 시스템에서 문제 전파가 종종 **동시 발생이 아니라 지연된 연쇄 반응** 으로 나타난다는 관찰이다. 예를 들어 서비스 A의 이상이 몇 분 뒤 서비스 B 지표에 반영될 수 있다. 단순한 동시성 기반 결합은 이런 상황을 놓칠 수 있다. 논문은 정확한 joint distribution이나 서비스 dependency graph를 구축하는 방법도 가능하지만, 일반 환경에서는 어렵고 비현실적이라고 지적한다. 그래서 **가벼운 가정으로 빠르게 계산 가능하면서도 시차를 허용하는 실용적 결합법** 을 제안한 것이다.

### 3.4 실용적 설정과 계산 효율

논문은 단일 모델의 핵심 파라미터로 score 분포를 추정하는 긴 window, 최근 평균을 구하는 짧은 window, 그리고 alert 빈도를 조절하는 threshold를 제시한다. 저자들은 긴 window는 충분히 크기만 하면 성능에 매우 민감하지 않고, 가장 중요한 파라미터는 최종 threshold라고 본다. 또한 다중 모델에서는 temporal window 폭이 추가되지만, Gaussian smoothing 덕분에 여기에 대해서도 아주 민감하지 않다고 설명한다.

계산 효율 측면에서 논문은 당시 고급 노트북 기준 **모델당 입력 벡터 하나를 처리하는 데 10ms 미만**, NAB 전체 365,558개 레코드를 처리하는 데 평균 **레코드당 약 8ms** 가 걸렸다고 보고한다. 서버 기반 병렬 운영에서는 5분마다 한 번 점수를 내는 조건에서 고성능 서버 한 대로 약 5,000개 모델을 돌릴 수 있다고도 적고 있다. 따라서 제안법은 단순한 연구용 알고리즘이 아니라 실제 운영 시스템을 염두에 둔 구현이라고 볼 수 있다.

## 4. 실험 및 결과

논문은 두 종류의 평가를 제시한다. 첫째는 **실제 금융/소셜 데이터 모니터링 제품에 통합된 사례에 대한 정성적 예시**, 둘째는 **NAB 벤치마크에 대한 정량 평가** 다. 정성 예시에서 저자들은 트위터 언급량 급증이 시장 개장 전 주가 급락에 선행했던 사례, Facebook 거래량에서 단일 spike보다 두 번 연속 spike가 더 이상적인 temporal anomaly로 작동한 사례, 그리고 Comcast 관련 두 지표를 함께 보아야만 잡히는 복합 이상 사례를 보여준다. 이 부분의 메시지는 명확하다. 제안법은 값이 큰지 작은지만 보지 않고 **시퀀스의 문맥과 여러 지표의 조합** 을 이용해 더 이른 신호를 포착하려 한다.

정량 실험에서 사용한 NAB는 58개 스트림, 35만 개가 넘는 레코드로 구성된 공개 실시간 anomaly benchmark이며, 각 스트림 앞부분 15%는 auto-calibration에 사용된다. NAB는 anomaly 주변에 time window를 두고, **일찍 탐지할수록 더 높은 점수** 를 주는 시간 민감 scoring을 사용한다. 또한 false positive를 더 싫어하는 profile과 false negative를 더 싫어하는 profile을 따로 두어, 운영 시나리오별 trade-off도 평가한다. 논문은 이 설정이 “실제로 사람이 없는 자동화 환경에서 단일 파라미터 셋으로 돌아가야 하는” 조건을 모사한다고 설명한다.

결과는 Table 1에 요약되어 있다. HTM은 **NAB score 65.3**, low FP profile **58.6**, low FN profile **69.4** 를 기록했다. 비교 대상인 Twitter ADVec은 **47.1 / 33.6 / 53.5**, Etsy Skyline은 **35.7 / 27.1 / 44.5**, Bayesian change point는 **17.7 / 3.2 / 32.2**, sliding threshold baseline은 **15.0 / 0.0 / 30.1**, random baseline은 **11.0 / 1.2 / 19.5** 였다. 논문은 이 결과를 바탕으로 HTM 기반 방법이 전체적으로 가장 좋은 균형을 보였다고 주장한다. 물론 “Perfect detector”가 100점이라는 점을 함께 제시하면서, 벤치마크 자체가 매우 어렵고 여전히 개선 여지가 크다고도 인정한다.

에러 분석도 흥미롭다. 논문은 CPU 사용률이 새로운 정상으로 바뀐 스트림에서 HTM과 Skyline은 적응했지만, Twitter ADVec은 며칠간 계속 anomaly를 내는 사례를 보여 주며 **continuous learning의 가치** 를 강조한다. 반대로 일부 스트림에서는 Skyline이 더 빨리 적응해 더 좋은 점수를 얻은 경우도 있었다고 적는다. 또 온도 센서 데이터 사례에서는 catastrophic failure 전에 나타난 미묘한 temporal anomaly를 **HTM만 탐지** 했다고 보고한다. 이 부분은 제안법의 가장 중요한 주장, 즉 **temporal modeling이 조기 경보를 가능하게 한다** 는 메시지를 뒷받침한다. 다만 논문도 이를 완전히 증명했다기보다, 사례 분석과 정성적 해석으로 제시하고 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정이 매우 현실적이라는 점이다. 저자들은 오프라인 이상 탐지가 아니라 **운영 중인 시스템에서 바로 써야 하는 anomaly detection** 을 목표로 삼고, 그에 맞춰 온라인 학습, 자동 적응, 낮은 튜닝 비용, 다중 지표 결합, 계산 효율성까지 함께 논의한다. 실제 제품 적용 예시와 공개 벤치마크를 모두 제시했다는 점도 장점이다. 특히 benchmark 결과뿐 아니라 **왜 잘 되는지** 를 temporal anomaly, sequence context, new normal adaptation 관점에서 일관되게 설명한다.

방법론 측면 강점은 **예측 모델과 통계적 판단을 분리** 했다는 데 있다. HTM이 sequence prediction을 담당하고, anomaly likelihood가 raw score의 분포 변화만 본다. 이 분리는 도메인 일반성을 높인다. 논문도 likelihood 계산 자체는 HTM에만 묶여 있지 않고, 다른 sparse code 기반 모델이나 scalar anomaly score 모델에도 적용 가능하다고 말한다. 즉, 본 논문의 실질적 기여는 “HTM을 썼다”는 것뿐 아니라, **실시간 anomaly score를 확률적으로 안정화하는 운영형 판정 구조** 에도 있다.

반면 한계도 분명하다. 첫째, anomaly likelihood는 rolling normal distribution과 Gaussian tail probability를 사용한다. 저자들 스스로도 discussion에서 **anomaly score 분포가 항상 Gaussian인 것은 아니다** 라고 인정한다. 이는 실세계에서 분포 비대칭이나 heavy tail이 있을 때 calibration 문제가 생길 수 있음을 시사한다. 둘째, 다중 모델 결합은 실용적이지만 **모델 간 독립성 가정** 과 Gaussian smoothing에 의존한다. 상호작용이 강한 복합 시스템에서는 더 정교한 joint modeling이나 causal dependency modeling이 필요할 수 있다.

셋째, 비교 실험은 당시로서는 유의미하지만, 비교 대상이 제한적이다. 최신 딥러닝 기반 시계열 anomaly detector나 representation learning 방법들과의 비교는 당연히 포함되어 있지 않다. 다만 이는 2016년 논문이라는 시점상 자연스러운 한계다. 넷째, 실험의 강한 결론 중 일부는 사례 기반 해석에 기대고 있다. 예를 들어 “HTM이 temporal anomaly를 더 일찍 잡는다”는 서술은 설득력 있지만, 그 원인을 엄밀히 분해한 ablation이 충분히 제공되지는 않는다. 따라서 이 논문은 **운영형 시스템 설계와 empirical validation에는 강하지만, 이론적 보장이나 대규모 ablation은 상대적으로 약하다** 고 볼 수 있다.

## 6. 결론

이 논문은 실시간 스트리밍 환경을 위한 anomaly detection을 다루면서, **HTM 기반 예측 오차 + anomaly likelihood** 라는 구조를 제안했다. 핵심은 현재 값이 이상한지를 직접 보지 않고, **현재 관측이 최근까지 학습한 sequence 문맥에서 얼마나 예측 불가능한지** 를 보고, 그 예측 실패가 최근 분포 대비 얼마나 이례적인지 확률적으로 판단한다는 점이다. 또한 여러 개의 독립 모델에서 나온 이상 신호를 temporal window를 통해 결합함으로써 복합 시스템에서도 사용할 수 있게 했다. 논문이 제시한 NAB 결과에서는 HTM이 비교 알고리즘들보다 가장 높은 종합 점수를 기록했다.

실제 적용 측면에서 이 연구는 지금 봐도 의미가 있다. 센서, 인프라, 금융, 보안처럼 **실시간성·적응성·낮은 관리비용** 이 중요한 환경에서는 여전히 유효한 설계 원칙을 제공한다. 특히 “이상은 값 자체보다 예측 가능성의 붕괴로 나타날 수 있다”는 관점은 이후 시계열 anomaly detection 연구 전반과도 잘 연결된다. 향후 연구로는 논문이 직접 제안한 것처럼 **더 적절한 score distribution 모델링**, **ensemble화**, **모델 간 의존성의 더 정교한 통합** 이 자연스러운 확장 방향이다.
