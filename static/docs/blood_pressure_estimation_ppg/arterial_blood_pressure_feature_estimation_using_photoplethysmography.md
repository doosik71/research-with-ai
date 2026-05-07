# Arterial Blood Pressure Feature Estimation Using Photoplethysmography

- **저자**: Armin Soltan Zadi, Raichel Alex, Rong Zhang, Donald E. Watenpaugh, Khosrow Behbehani
- **발표연도**: 2018
- **arXiv**: <https://arxiv.org/abs/1811.06039v1>

## 1. 논문 개요

이 논문은 photoplethysmography (PPG) 신호만을 이용해 동맥 혈압의 핵심 특징인 수축기 혈압 SBP, 이완기 혈압 DBP, 그리고 평균동맥압 MAP를 연속적이고 비침습적으로 추정하는 방법을 제안한다. 기존의 연속 혈압 측정은 invasive 하거나, 장비가 비싸거나, finger cuff 같은 장치를 요구해 사용성이 떨어지는 경우가 많다. 저자들은 이런 한계를 줄이기 위해, 비교적 저렴하고 적용이 쉬운 PPG로부터 혈압 특징을 추정할 수 있는지를 검토했다.

연구 문제는 명확하다. PPG는 혈액량 변화와 말초혈관의 박동 정보를 담고 있으므로 혈압과 연관성이 있지만, 이를 어떻게 안정적으로 SBP와 DBP로 매핑할 것인가가 핵심이다. 특히 기존의 pulse transit time (PTT) 기반 방법은 ECG와 PPG를 동시에 필요로 하므로 센서 추가, 동기화 문제, 실시간 구현의 복잡성이 있다. 이에 비해 이 논문은 PPG 단일 신호만으로 혈압 특징을 추정하는 방향을 택한다.

이 문제가 중요한 이유는 고혈압 관리, 장기 모니터링, 수면무호흡 같은 상태의 야간 혈압 추적, 웨어러블 기반 헬스 모니터링 등 실제 적용 가능성이 크기 때문이다. 논문은 특히 breath-hold 실험을 통해 수면무호흡과 유사한 생리적 교란 상황에서도 방법이 어느 정도 동작하는지 평가했다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 PPG 파형의 peak와 trough를 각각 혈압 파형의 특징점과 직접 연결하고, 그 관계를 autoregressive moving average (ARMA) 모델로 학습하는 것이다. 구체적으로는 PPG peak를 입력으로 SBP를, PPG trough를 입력으로 DBP를 모델링한다. 이후 추정된 SBP와 DBP로부터 MAP를 계산한다.

기존의 단순 회귀(regression) 방식은 현재 입력과 현재 출력의 정적 관계는 잡을 수 있어도, PPG와 혈압 사이의 시간 지연이나 동적 상관을 반영하기 어렵다. 저자들은 ARMA가 현재 및 과거 입력, 과거 출력, 그리고 순수 지연 pure time delay를 함께 반영할 수 있기 때문에 생리 신호처럼 시간적으로 연속적이고 지연이 존재하는 시스템에 더 적합하다고 본다.

또한 Fourier domain 접근은 정지성(stationarity) 가정이 강하고 계산량이 크다는 점에서 한계가 있다고 지적한다. 반면 ARMA는 비교적 단순하면서도 동적 시스템 식별에 적합하고, 생리계에 흔한 시간 지연까지 모델링할 수 있다는 점이 차별점이다.

## 3. 상세 방법 설명

전체 파이프라인은 비교적 직관적이다. 먼저 Finapres 장비로 연속 혈압을, Nellcor OxiMax 장비로 PPG를 획득한다. 두 신호 모두 100 Hz로 샘플링했다. 이후 혈압 신호에서 SBP와 DBP를 추출하고, PPG에서는 peak와 trough를 뽑는다. 그런 다음 PPG 특징을 입력, 혈압 특징을 출력으로 두는 ARMA 모델을 각 실험 구간마다 별도로 학습한다.

혈압 특징 정의는 다음과 같다. SBP는 혈압 파형의 peak로 정의했고, MATLAB findpeaks 알고리즘을 사용해 검출했다. 그 설정값은 MinPeakProminence = 15 mmHg, MinPeakDistance = 20 samples, MinPeakHeight = 15 mmHg이다. DBP는 연속된 두 systolic peak 사이의 최소값으로 정의했다. MAP는 추정된 SBP와 DBP로부터 다음 식으로 계산했다.

$$
MAP = \frac{2 \times DBP + SBP}{3}
$$

이 식은 일반적인 임상 근사식이며, 논문은 measured MAP와 estimated MAP 모두에 동일한 형태를 적용했다.

ARMA 모델은 다음 차분방정식으로 표현된다.

$$
y(m) + a_1 y(m-1) + \dots + a_{n_a} y(m-n_a)
=
b_1 u(m-n_k) + \dots + b_{n_b} u(m-n_b-n_k+1) + e(m)
$$

여기서 $m$은 샘플 인덱스, $y$는 출력, $u$는 입력, $e$는 오차이다. $a_i$와 $b_j$는 추정할 모델 계수이고, $n_a$, $n_b$는 모델 차수, $n_k$는 순수 시간 지연 샘플 수다. 이 논문에서는 single-input single-output, time-invariant, causal ARMA 구조를 사용했다.

모델 선택은 parsimony와 model adequacy 원칙에 따라 수행했다. $n_a$, $n_b$는 1부터 5까지, $n_k$는 0부터 5까지 탐색했고, 각 조합에 대해 least squares with QR factorization으로 계수를 추정했다. 이후 residual MSE가 가장 낮은 모델을 선택했다. Akaike's Information Criterion (AIC)도 검토했지만, 저자들은 최종적으로 MSE 기준을 더 직접적으로 사용한 것으로 보인다. 차수를 5 이하로 제한한 이유는 일부 임의 구간을 실험적으로 살펴본 결과, 그 이상의 차수에서 MSE 개선이 유의하지 않았기 때문이다.

입력과 출력이 등간격 샘플링이어야 ARMA를 적용할 수 있기 때문에, beat-level로 얻어진 SBP, DBP, MAP 값은 cubic spline interpolation으로 100 Hz 시계열로 보간했다. 즉, 실제 모델링은 박동 단위 특징을 시간축 상의 연속 신호처럼 재구성한 뒤 수행되었다.

실험 구간은 각 피험자마다 총 11개다. 정상 호흡 NB 구간이 6개, breath-hold BH 구간이 5개이며, 각 구간마다 SBP용 모델과 DBP용 모델을 따로 학습했다. 따라서 한 피험자당 11개의 SBP 모델과 11개의 DBP 모델이 만들어졌다.

오차 평가는 두 가지다. 첫째는 model error로, 특정 구간의 데이터로 학습한 모델을 같은 구간에 적용했을 때의 샘플별 차이다. 둘째는 prediction error로, 예를 들어 BH1에서 만든 모델을 BH2, BH3 등 다른 congruent interval에 적용했을 때의 오차다. 이 오차의 집계는 모두 root mean square error (rMSE)로 계산했다.

$$
rMSE = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(\hat{y}_t-y_t)^2}
$$

여기서 $y_t$는 측정된 혈압 특징, $\hat{y}_t$는 모델이 추정한 값이다.

## 4. 실험 및 결과

실험 대상은 심혈관 질환이 없는 젊은 성인 15명이다. 남성 8명, 여성 7명이며, 평균 연령은 $28.9 \pm 5.0$세, BMI는 $24.1 \pm 4.8 \text{ kg/m}^2$이다. 평균 baseline 혈압은 SBP 126.8 mmHg, DBP 74.8 mmHg였다. 피험자는 누운 자세에서 60초 정상 호흡을 한 뒤, 5회의 breath-hold를 수행했고, 각 breath-hold 사이에는 90초 회복 구간을 두었다. 마지막 breath-hold 이후 다시 60초 정상 호흡을 기록했다. breath-hold 지속 시간은 개인마다 달랐다.

논문은 breath-hold를 선택한 이유를 생리적으로 설명한다. 이 조작은 산소포화도 저하와 이산화탄소 증가를 유발해 화학수용체와 sympathetic nervous system을 자극하고, 결과적으로 혈압을 상승시킨다. 실제로 논문 그림 설명에 따르면 BH 구간에서 혈압과 PPG amplitude가 함께 상승하는 경향이 관찰되었다.

동일 구간 내 모델 적합 성능은 상당히 좋았다. 정상 호흡 구간에서 SBP의 rMSE는 약 3.953에서 4.692 mmHg, DBP는 2.969에서 3.495 mmHg, MAP는 3.025에서 3.354 mmHg였다. breath-hold 구간에서도 SBP는 4.190에서 4.935 mmHg, DBP는 2.884에서 3.294 mmHg, MAP는 2.842에서 3.271 mmHg였다. 즉, 논문 본문 표현대로 model error 기준으로는 모든 경우가 5 mmHg 이하에 있었다.

다른 congruent interval에 모델을 적용한 prediction 성능은 당연히 조금 떨어졌지만, 여전히 비교적 안정적이었다. 정상 호흡 구간에서 prediction rMSE는 SBP 6.582에서 8.137 mmHg, DBP 4.013에서 6.439 mmHg, MAP 4.224에서 5.847 mmHg였다. breath-hold 구간에서는 SBP 6.481에서 7.944 mmHg, DBP 4.239에서 4.652 mmHg, MAP 4.225에서 4.655 mmHg였다. 논문은 이를 근거로, 다른 구간에 적용하는 경우까지 포함해도 전반적인 오차 수준이 8 mmHg 미만이라고 요약한다.

오차 평균도 함께 해석했다. 본문에 따르면 NB 구간의 estimation/model error 평균은 대부분 $\pm 2$ mmHg 안에 있었고, 모든 경우가 대체로 $\pm 3$ mmHg 이내였다. BH 구간은 오히려 더 좁아서 모든 평균 오차가 $\pm 2$ mmHg 이내였다고 설명한다. 이는 편향(bias)이 크지 않다는 의미다.

피험자 간 일관성도 통계적으로 평가했다. 15명의 subject pair에 대해 총 105개의 비교를 하고, 여기에 호흡 조건 2개, 혈압 특징 3개, 오차 종류 2개를 곱해 총 1260개의 mean comparison을 수행했다. ANOVA와 multiple comparison 결과 유의한 차이($p < 0.05$)가 나온 경우는 9개뿐이었다. 논문은 이를 0.71%로 요약하며, 모델 구조가 피험자 전반에서 비교적 일관되게 작동한다고 해석한다. 다만 본문 후반에는 9/1260을 0.07%라고 적었는데, 이는 앞서 제시한 0.71%와 수치적으로 맞지 않는다. 분수 계산상 0.71%가 맞다.

## 5. 강점, 한계

이 논문의 강점은 우선 문제 설정이 실용적이라는 점이다. ECG 없이 PPG 단일 신호만으로 연속 혈압 특징을 추정하려 했고, 이를 위해 동적 관계와 시간 지연을 표현할 수 있는 ARMA를 선택했다. 단순 회귀보다 생리 신호의 시간적 구조를 더 잘 반영하려 했다는 점이 설득력 있다. 또한 SBP, DBP뿐 아니라 MAP까지 포함해 임상적으로 자주 쓰이는 혈압 특징 전체를 다뤘다는 점도 실용적이다.

실험 결과도 논문 범위 내에서는 꽤 강하다. 동일 구간 모델링에서는 모든 rMSE가 5 mmHg 미만이었고, 다른 구간 예측에서도 대체로 8 mmHg 이내였다. 특히 breath-hold 같은 교란 상황에서도 정상 호흡과 비슷한 수준의 성능을 보여, 단순한 resting-only 실험보다 한 단계 더 의미가 있다. 피험자 간 모델 오차 평균의 차이가 거의 없었다는 통계 분석도 모델 구조의 일관성을 뒷받침한다.

반면 한계도 분명하다. 첫째, 대상자가 15명의 젊고 건강한 성인으로 제한되어 있다. 고혈압 환자, 고령자, 심혈관 질환자, 실제 수면무호흡 환자에 대한 검증은 없다. 따라서 임상 일반화는 아직 이르다. 둘째, 실험은 supine position의 비교적 통제된 환경에서 수행되었고, 논문도 신호가 깨끗하여 잡음 제거가 필요 없었다고 적고 있다. 실제 wearable 환경의 motion artifact 문제는 해결되지 않았다.

셋째, 모델이 person-specific인지, 혹은 population-level로 바로 쓸 수 있는지에 대해서는 제한이 있다. 논문 후반부에서도 향후에는 개인 맞춤형 모델 계산과 검증이 필요하다고 인정한다. 이는 곧 현재 방법이 넓은 인구집단에서 calibration-free로 바로 쓰이기 어렵다는 뜻이다. 넷째, 선형 ARMA만 사용했기 때문에 실제 혈압 조절계의 비선형성을 충분히 반영하지 못할 가능성이 있다. 저자들도 nonlinear modeling이 정확도를 더 높일 수 있다고 언급한다.

또 하나의 실무적 제약은, 논문이 NB용 모델과 BH용 모델을 별도로 쓰는 구성을 사실상 제안한다는 점이다. 이는 수면무호흡처럼 상태 구분이 가능한 경우에는 가능할 수 있지만, 일반적인 연속 모니터링에서는 현재 상태가 NB인지 apnea-like perturbation인지 먼저 판별해야 한다. 즉, 혈압 추정 자체 외에 상태 분류 체계가 함께 필요할 수 있다.

## 6. 결론

이 논문은 PPG의 peak와 trough를 이용해 SBP와 DBP를 추정하고, 이를 바탕으로 MAP를 계산하는 ARMA 기반 혈압 추정 방법을 제시했다. 핵심 기여는 PPG 단일 신호만으로도 연속적이고 비침습적인 혈압 특징 추정이 가능함을, 정상 호흡과 breath-hold라는 두 조건에서 정량적으로 보였다는 점이다. 동일 구간에서는 5 mmHg 미만, 다른 congruent interval 예측에서도 대체로 8 mmHg 미만의 rMSE를 보고했다.

실제 적용 가능성은 분명하다. 특히 야간 혈압 모니터링, 수면무호흡 관련 장기 추적, 웨어러블 기반 혈압 추정 같은 영역에서 잠재력이 있다. 다만 현재 결과는 건강한 소규모 집단과 통제된 환경에 기반하므로, 임상 적용을 위해서는 더 다양한 인구집단, 실제 생활 환경, motion artifact 대응, 개인별 calibration 전략, 그리고 비선형 모델 확장이 필요하다. 원문에 근거해 말하면, 이 연구는 즉시 완성된 제품 수준의 해법이라기보다, PPG-only continuous BP estimation이 충분히 성립 가능하다는 점을 보인 타당한 초기 단계 연구로 보는 것이 적절하다.
