# Wearable PPG Based BP Estimation Methods: A Systematic Review and Meta-Analysis

* **저자**: Ziya Sastimoglu, Sophini Subramaniam, Abu Ilius Faisal, Wei Jiang, Andrew Ye, M. Jamal Deen
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 wearable photoplethysmography, 즉 PPG 기반 cuff-less blood pressure, BP 추정 방법에 대한 systematic review와 meta-analysis이다. 논문은 PRISMA 가이드라인을 따라 MEDLINE, PubMed, AMED, Embase, IEEE Xplore를 검색하고, 2013년 1월부터 2024년 1월까지 발표된 PPG 기반 wearable BP monitoring 관련 연구를 검토하였다. 최종적으로 정량 메타분석에는 25개 연구와 총 21,142명의 participant 또는 record가 포함되었고, 전체 systematic review 단계에서는 42개 연구가 포함되었다.

논문의 목표는 PPG 기반 cuff-less BP monitoring device와 algorithm이 실제 daily 또는 long-term use에 적합한지, 그리고 넓은 population에서 적용 가능하고 usable한지를 평가하는 것이다. 단순히 알고리즘 정확도만 보는 것이 아니라, wearable device로서의 실용성, calibration 방식, participant diversity, long-term monitoring 가능성, motion artifact와 sensor robustness, hypertensive population에서의 검증 여부까지 함께 다룬다.

연구 문제는 다음과 같이 요약할 수 있다. 첫째, PPG 기반 wearable BP estimation이 기존 cuff-based measurement를 보완하거나 대체할 정도로 정확한가이다. 둘째, Pulse Wave Velocity, PWV 기반 방식과 Pulse Waveform Analysis, PWA 기반 방식 중 어떤 접근이 더 유망한가이다. 셋째, 현재 연구들이 실제 사용 환경, 특히 고령자와 hypertensive patient의 장기 모니터링 요구를 충분히 반영하고 있는가이다.

문제의 중요성은 hypertension의 높은 유병률과 관련된다. 논문은 hypertension을 증상이 잘 드러나지 않는 “silent killer”로 설명하며, 전 세계적으로 약 12.8억 명이 영향을 받는다고 언급한다. Hypertension은 heart failure, stroke, kidney disease 등과 관련되므로, frequent BP monitoring은 예방과 관리에 필수적이다. 그러나 기존 cuff-based non-invasive BP measurement는 arm 또는 wrist cuff를 사용하므로 장시간 착용이 불편하고, 반복적인 압박이 신체에 부담을 줄 수 있어 continuous monitoring에 적합하지 않다.

이와 비교하여 cuff-less wearable BP monitoring은 장기간, 연속적으로, 일상생활을 방해하지 않는 방식으로 혈압을 관찰할 수 있다는 장점이 있다. 특히 PPG sensor는 구조가 단순하고, 저렴하며, 에너지 효율이 높고, watch, ring, armband, chest attachment 등 wearable 형태로 구현하기 쉽다. 따라서 PPG 기반 wearable BP estimation은 daily health management와 chronic cardiovascular condition 관리에 중요한 기술 후보이다.

메타분석 결과, 전체적으로 SBP의 pooled mean difference는 4.14 mmHg, DBP는 2.79 mmHg로 보고되었다. 이는 기존 reference measurement와 비교했을 때 평균 bias가 비교적 낮다는 의미이다. 세부적으로 PWA 기반 방법은 PWV 기반 방법보다 SBP와 DBP 모두에서 더 낮은 mean bias를 보였지만, 논문은 이 차이가 통계적으로 유의하지는 않다고 설명한다. 동시에 PWA 기반 연구들은 heterogeneity가 매우 높아, 연구 간 변동성이 크다는 점도 강조한다.

논문의 결론은 조심스럽게 긍정적이다. Wearable PPG 기반 BP monitoring은 short-term BP monitoring에서는 유망하며, PWA와 PWV 모두 wearable format에서 상당한 가능성을 보인다. 그러나 장기 사용, 고혈압 환자, 고령자, 운동과 일상 활동을 포함한 실제 환경에서의 신뢰성은 아직 충분히 검증되지 않았다. 따라서 이 논문은 기술적 가능성뿐 아니라 임상적·실용적 검증의 부족을 균형 있게 지적하는 review이다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 wearable PPG 기반 cuff-less BP estimation 연구들을 단순 정확도 비교가 아니라, **임상 표준 충족 여부, 방법론 유형, device practicality, calibration, long-term usability, participant diversity**라는 여러 관점에서 통합 평가하는 것이다. 기존 review들이 알고리즘이나 센서 원리에 치우쳤다면, 이 논문은 실제 daily health management에 사용할 수 있는 wearable BP monitoring 기술인지에 초점을 둔다.

논문이 구분하는 주요 방법론은 크게 PWV 기반 방식과 PWA 기반 방식이다. PWV 기반 방식은 pulse wave가 혈관을 따라 이동하는 속도와 혈압 사이의 관계를 이용한다. 이 방식에서는 Pulse Transit Time, PTT나 Pulse Arrival Time, PAT를 계산하기 위해 보통 ECG와 PPG, 또는 서로 다른 위치의 PPG sensor가 필요하다. 따라서 두 개 이상의 sensor를 사용해야 하는 경우가 많고, sensor 위치, 거리, calibration에 민감하다.

반면 PWA 기반 방식은 적어도 하나의 pulse sensor, 주로 PPG sensor를 사용하여 waveform 자체의 morphology를 분석한다. PPG waveform의 peak, valley, slope, time interval, amplitude, area 등에서 BP와 관련된 feature를 추출하거나, machine learning 및 deep learning 알고리즘이 waveform representation을 직접 학습한다. PWA는 단일 PPG sensor만으로 구현될 수 있어 wearable device 구조를 단순화할 수 있다는 장점이 있다.

논문은 메타분석 결과에서 PWA가 PWV보다 평균적으로 더 낮은 bias를 보였다고 보고한다. SBP에서는 PWA 기반 방법의 mean difference가 3.82 mmHg, PWV 기반 방법은 5.16 mmHg였다. DBP에서는 PWA가 2.47 mmHg, PWV가 3.89 mmHg였다. 다만 PWA 기반 연구들은 heterogeneity가 매우 높아 결과가 일관적이지 않았고, PWV 기반 연구들은 상대적으로 더 낮은 heterogeneity를 보였다. 즉, PWA는 높은 성능을 낼 잠재력이 있지만, 연구 설계와 데이터 조건에 따라 성능 변동이 크다는 것이 논문의 중요한 해석이다.

또 다른 핵심 아이디어는 **calibration이 wearable BP monitoring의 실용성을 좌우한다**는 점이다. Cuff-less BP estimation은 개인의 혈관 탄성, 나이, BMI, skin tone, heart rate, sensor placement, waveform quality에 영향을 받기 때문에 calibration이 중요하다. 논문은 calibration을 generalized, personalized, hybrid approach로 나눈다. Generalized calibration은 나이, 성별, BMI, PPG waveform feature 등 population-level 정보를 사용한다. Personalized calibration은 subject-specific waveform과 개인별 calibration constant를 사용한다. Hybrid calibration은 group-level parameter와 subject-specific parameter를 섞는다.

논문의 중요한 비판은 현재 많은 연구가 young healthy participants나 online clinical database에 의존하며, 실제로 wearable BP device가 가장 필요한 older adults와 hypertensive populations를 충분히 포함하지 않는다는 점이다. 이는 모델의 실제 generalizability를 제한한다. 특히 젊고 정상혈압인 subject만 포함하면 BP range가 좁아 regression error가 낮게 나올 수 있으며, 이는 실제 고혈압 환자나 고령자에서의 성능을 과대평가할 수 있다.

마지막으로 논문은 long-term monitoring과 device robustness를 핵심 문제로 본다. Wearable BP device는 운동, 수면, 일상 활동, sensor displacement, ambient light, motion artifact 상황에서도 안정적으로 동작해야 한다. 그러나 많은 연구는 short-term static condition에서만 평가되었고, 장기 정확도 저하와 recalibration 필요성에 대한 evidence가 제한적이다.

## 3. 상세 방법 설명

이 논문은 systematic review와 meta-analysis 연구이므로, 새로운 BP estimation model을 제안하는 논문은 아니다. 대신 기존 연구를 체계적으로 검색, 선별, 평가하고, 정량 결과를 pooled analysis로 통합한다. 방법론은 PRISMA guideline을 따르며, search strategy, eligibility criteria, study selection, data extraction, meta-analysis, risk of bias assessment로 구성된다.

### 검색 전략과 포함 기준

저자들은 MEDLINE, PubMed, AMED, Embase, IEEE Xplore의 다섯 database에서 2013년 1월부터 2024년 1월까지의 문헌을 검색하였다. 검색어는 BP 또는 blood pressure 또는 SBP 또는 DBP, PPG 또는 Photoplethysmography, Wearable을 조합하였다. 영어로 작성되고 full-text가 제공되는 연구만 포함하였다.

포함 기준은 PPG 기반 non-invasive, cuff-less study이며, human vital sign에 대해 reference device와 비교 테스트한 연구이다. Device 연구와 algorithm 연구 모두 포함하였다. 또한 정확도 기준을 엄격하게 적용하였다. ANSI/AAMI/ISO 81060-2:2013 기준에서는 SBP와 DBP 모두에 대해 Mean Error, ME가 5 mmHg 이하이고 Standard Deviation, SD가 8 mmHg 이하인 연구를 포함하였다. BHS 기준에서는 A 또는 B grade를 받은 연구를 포함하였다. IEEE standard와 관련해서는 Mean Absolute Error, MAE가 7 mmHg 미만인 연구를 포함하였다. 또한 participant 수가 최소 10명 이상이어야 했다.

이 기준은 논문의 장점이자 한계이다. 정확도 기준을 충족한 연구만 분석하므로 clinically plausible한 연구를 중심으로 평가할 수 있지만, 기준을 충족하지 못했으나 방법론적으로 흥미로운 연구는 제외될 수 있다.

### 연구 선별 및 데이터 추출

초기 검색에서 810편의 논문이 식별되었다. Abstract screening에서 218편은 irrelevant, 391편은 duplicate로 제외되었다. 남은 201편에 대해 full-text review를 수행했고, 161편이 inclusion criteria를 충족하지 못해 제외되었다. 이후 40개 연구가 포함되었고, 외부 source에서 2개 relevant study가 추가되어 최종 systematic review에는 42개 연구가 포함되었다.

데이터 추출은 독립 reviewer 쌍에 의해 수행되었다. 추출 항목에는 study characteristics, author-year, subject count, recruitment strategy, population, health condition, device/method characteristics, reference device, validation protocol, test-reference BP difference, MAE, RMSE, ME, SD, clinical application, sensor technology 등이 포함되었다. 동일 device 또는 method가 여러 publication에서 설명된 경우 하나의 method/device로 요약하였다.

### 메타분석 방법

저자들은 SBP와 DBP 각각에 대해 mean bias를 pooled analysis하였다. Random-effect model meta-analysis를 사용하였고, heterogeneity variance는 maximum likelihood estimator로 계산하였다. Pooled effect의 confidence interval 계산에는 Knapp–Hartung adjustment를 사용하였다. 논문은 전체 study 간 heterogeneity가 substantial하다고 보고한다.

Pooled mean bias와 standard deviation은 다음과 같이 정의된다.

$$
Mean_{BP} = \frac{1}{n}\sum_{i=1}^{n} Mean_{BP_i}
$$

여기서 $BP$는 SBP 또는 DBP이고, $Mean_{BP_i}$는 $i$번째 연구에서 보고된 SBP 또는 DBP의 평균 measurement difference이다. $n$은 pooled analysis에 포함된 study 수이다.

Pooled standard deviation은 다음과 같이 계산된다.

$$
SD_{BP} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(Mean_{BP_i} - Mean_{BP})^2}
$$

이 수식은 포함된 연구들이 보고한 평균 BP difference가 전체 평균으로부터 얼마나 흩어져 있는지를 나타낸다. 다만 개별 환자 수준의 raw data를 직접 통합하는 방식은 아니며, 연구별 summary statistic을 기반으로 한 분석이다.

### Risk of Bias 평가

Quality assessment에는 modified QUADAS-2가 사용되었다. QUADAS-2는 diagnostic accuracy study의 risk of bias와 applicability를 평가하는 도구이다. 논문은 patient selection, index test, reference test, flow and timing의 네 domain을 평가하였다. 각 domain은 high, low, unclear로 평가하고, 최종적으로 각 연구의 risk of bias를 high, moderate, low로 분류하였다.

연구에서 30명 미만의 participant를 사용한 경우 high risk of bias로 분류하였다. 또한 healthy participant만 포함하거나 normotensive range만 사용한 경우도 high bias로 보았다. 이는 BP estimation model이 좁은 BP range에서만 평가되면 실제 고혈압 환자나 다양한 혈압 범위에서의 성능을 대표하지 못하기 때문이다.

### PPG 기반 BP 추정 기술 설명

논문은 PPG 기반 BP estimation 기술을 PWV 기반과 PWA 기반으로 나누어 설명한다.

PWV는 arterial stiffness와 BP의 상관을 이용한다. 기본적으로 동일 arterial branch의 서로 다른 위치에서 측정한 waveform 사이의 시간 지연과 거리를 사용한다.

$$
PWV = \frac{\Delta x}{\Delta t}
$$

여기서 $\Delta x$는 두 측정 지점 사이의 거리이고, $\Delta t$는 pulse wave가 그 거리를 이동하는 데 걸린 시간이다. 혈관이 더 stiff하면 pulse wave가 더 빠르게 전달되므로 PWV가 증가한다.

Moens-Korteweg equation은 PWV와 혈관 wall의 mechanical property를 연결한다.

$$
PWV = \sqrt{\frac{Eh}{2R\rho}}
$$

여기서 $\rho$는 blood density, $R$은 vessel inner radius, $h$는 vessel wall thickness, $E$는 vascular wall의 elastic modulus이다. 이 식은 혈관벽 탄성, 두께, 반지름, 혈액 밀도가 pulse wave 속도에 영향을 준다는 것을 보여준다.

Bramwell-Hill equation은 혈관 단면적 변화와 압력 변화를 사용한다.

$$
PWV = \sqrt{\frac{A}{\rho} \cdot \frac{\Delta P}{\Delta A}}
$$

여기서 $A$는 mean vessel area, $\Delta A$는 cardiac cycle 동안 최대·최소 vessel area 차이, $\Delta P$는 central SBP와 DBP의 차이이다. 제공된 텍스트에는 분수 형태가 다소 압축되어 있지만, 의미는 혈관 면적 변화에 대한 압력 변화의 비율이 PWV와 관련된다는 것이다.

Elastic modulus는 BP의 함수로 표현될 수 있다.

$$
E = E_0 e^{\alpha BP}
$$

여기서 $E_0$는 zero-pressure modulus이고, $\alpha$는 혈관 wall의 압력 민감도와 관련된 coefficient이다. 이 관계는 혈압이 높아질수록 혈관 stiffness가 증가하고, 결과적으로 PWV가 변한다는 생리적 배경을 설명한다.

PTT는 같은 arterial path의 두 지점 사이에서 pulse가 이동하는 시간이다.

$$
PTT = \frac{\Delta x}{PWV}
$$

PAT는 ECG R-peak와 PPG waveform의 peak, valley, maximum slope 같은 특징점 사이의 시간 지연이다. PAT에는 pre-ejection period, PEP가 포함되므로 순수한 혈관 전달 시간인 PTT와 다르다. PEP는 심장이 전기적으로 활성화된 뒤 실제 blood ejection이 시작되기까지의 시간이다. 따라서 PAT는 ECG와 PPG를 사용해 계산하기 쉽지만, PEP 변화의 영향을 받는다.

PWA는 waveform 자체의 morphology를 사용한다. PWA는 적어도 하나의 pulse sensor만 필요하고, PPG signal의 feature와 BP 사이의 관계를 ML 또는 DL로 학습한다. 논문은 PWA가 feature extraction, feature selection, noise 및 motion artifact reduction, BP estimation에 ML/DL을 적극 활용한다고 설명한다.

## 4. 실험 및 결과

이 논문의 결과는 systematic review 결과, calibration 분석, algorithm-based approach 분석, device-based approach 분석, meta-analysis, risk of bias 분석으로 구성된다.

### 문헌 선별 결과

초기 검색 결과는 810편이었다. Abstract screening 후 irrelevant 218편, duplicate 391편이 제외되었고, 201편이 full-text review로 넘어갔다. 이 중 161편이 inclusion criteria를 충족하지 못했고, 40편이 포함되었다. 외부 source에서 2편이 추가되어 총 42개 연구가 systematic review에 포함되었다. Meta-analysis에는 SBP와 DBP의 MAE 및 SD를 분석할 수 있는 25개 연구가 포함되었으며, 총 participant 또는 record 수는 21,142였다.

### 전체 메타분석 결과

전체 25개 연구에서 SBP의 pooled mean bias는 다음과 같다.

$$
MD_{SBP} = 4.14 \text{ mmHg}, \quad 95% CI = [3.54, 4.74]
$$

DBP의 pooled mean bias는 다음과 같다.

$$
MD_{DBP} = 2.79 \text{ mmHg}, \quad 95% CI = [2.37, 3.21]
$$

이는 포함된 연구들의 wearable PPG 기반 BP estimation 결과가 reference device와 평균적으로 SBP에서 약 4.14 mmHg, DBP에서 약 2.79 mmHg 차이를 보였다는 의미이다. SBP 오차가 DBP보다 큰 것은 일반적인 BP estimation 연구 경향과도 일치한다. SBP는 혈관 stiffness, pulse wave reflection, motion artifact, physiological variability에 더 민감할 수 있기 때문이다.

그러나 heterogeneity는 매우 컸다. SBP의 heterogeneity variance는 $\tau^2 = 1.72$였고, $I^2 = 98.3%$였다. DBP는 $\tau^2 = 0.79$, $I^2 = 97.9%$였다. Chi-square test도 SBP에서 $x^2 = 1432.88$, DBP에서 $x^2 = 1170.54$로 매우 높고, 둘 다 $p < 0.01$이었다. 이는 연구 간 차이가 매우 크다는 뜻이다. 따라서 pooled mean bias가 낮더라도, 모든 wearable PPG BP method가 안정적으로 그 수준의 정확도를 낸다고 해석해서는 안 된다.

### PWA와 PWV subgroup 분석

PWA 기반 방법은 PWV 기반 방법보다 낮은 mean bias를 보였다. SBP에서 PWA 기반 연구의 mean difference는 3.82 mmHg, 95% CI는 [3.15, 4.50]이었다. PWV 기반 방법은 5.16 mmHg, 95% CI는 [4.15, 6.16]이었다. DBP에서도 PWA 기반 방법은 2.47 mmHg, 95% CI [2.04, 2.91]로, PWV 기반 방법의 3.89 mmHg, 95% CI [3.25, 4.52]보다 낮았다.

그러나 PWA 기반 방법의 heterogeneity는 매우 높았다. PWA의 SBP heterogeneity는 $\tau^2 = 1.64$, $I^2 = 99%$였고, DBP는 $\tau^2 = 0.64$, $I^2 = 99%$였다. 반면 PWV 기반 방법은 SBP에서 $\tau^2 = 0.83$, $I^2 = 54%$, DBP에서 $\tau^2 = 0.14$, $I^2 = 17%$로 더 낮았다. 즉, PWA는 평균적으로 더 좋은 bias를 보이지만 결과가 연구마다 매우 크게 달라진다. PWV는 평균 bias는 더 크지만 연구 간 일관성은 더 높다.

이 결과는 PWA가 ML/DL과 결합될 때 높은 성능 잠재력을 갖지만, dataset, preprocessing, feature extraction, calibration, participant 특성에 매우 민감하다는 것을 시사한다. PWV는 물리적 모델 기반이라 상대적으로 구조가 일정하지만, 여러 sensor가 필요하고 calibration 및 sensor placement에 민감한 단점이 있다.

### Algorithm-based approach 결과

Algorithm-based study에서는 PWA 기반 neural network가 많이 사용되었다. 논문은 neural network가 복잡한 linear 및 non-linear relationship을 포착할 수 있어 BP estimation에 적합하다고 설명한다. 특히 multi-layer feedforward neural network는 short-term BP fluctuation pattern을 포착하는 데 강점이 있고, LSTM은 더 긴 시간 context를 반영하여 정확도를 높일 수 있다고 언급한다.

대부분의 algorithm-based study는 IEEE 기준에서 A grade를 달성한 것으로 보고되었다. 다만 일부 연구는 ANSI/AAMI/ISO 기준에 필요한 standard deviation 정보를 제공하지 않았거나, MAE 기준은 충족했지만 SD 기준을 충족하지 못했다. 이는 많은 알고리즘 연구가 평균 절대 오차만 강조하고, 실제 의료기기 검증에 중요한 variability를 충분히 보고하지 않는다는 문제를 보여준다.

논문은 ECG를 함께 사용하는 방법이 정확도를 높일 수 있지만, PPG-only 방법도 높은 정확도를 달성할 수 있다고 지적한다. 실제로 review에 포함된 일부 연구는 single PPG waveform만으로 가장 높은 수준의 성능을 보고하였다. 이는 PWA 기반 PPG-only 접근이 wearable BP monitoring의 실용적인 방향이 될 수 있음을 보여준다.

### Device-based approach 결과

Device-based study에서는 PWV 기반 방법이 더 자주 사용되었다. 이는 실제 wearable device에서 PAT 또는 PTT를 이용하는 구조가 많이 구현되었기 때문이다. 그러나 PWV 기반 device는 ECG와 PPG 또는 두 개의 PPG sensor를 함께 사용해야 하는 경우가 많아 device 구조가 복잡해진다. Sensor가 서로 다른 위치에 있으면 착용감, 일상 활동 방해, sensor displacement 문제가 생길 수 있다.

PWA 기반 device 중 일부는 PPG만 사용하였다. 이 방식은 device 구조가 단순하고 wearable 구현에 유리하지만, motion artifact와 signal quality에 크게 의존한다. 논문은 feature selection이 device 성능에 매우 중요하다고 강조한다. PWV 방법은 regression model과 calibration equation에 의존하고, PWA 방법은 ML algorithm이 BP와 관련된 waveform feature를 직접 선택하거나 학습한다.

Device-based study의 중요한 문제는 participant 수와 population 구성이다. 대부분의 device study는 ANSI/AAMI/ISO 기준을 만족할 만큼 충분한 participant를 포함하지 못했다. 또한 가장 정확한 device 중 상당수는 young healthy participant를 대상으로 테스트되었고, 고령자나 hypertensive patient를 충분히 포함하지 않았다. 이는 실제로 가장 device가 필요한 population에 대한 성능 검증이 부족하다는 뜻이다.

### Calibration 분석

논문은 calibration을 generalized, personalized, hybrid 방식으로 나누어 설명한다. Generalized calibration은 age, gender, BMI, PPG waveform feature 등 population-level 정보를 사용한다. 이 방식은 scalability가 높지만 개인별 혈관 특성이나 waveform 차이를 충분히 반영하지 못해 precision이 제한될 수 있다.

Personalized calibration은 subject-specific PPG waveform과 개인별 parameter를 활용한다. 이 방식은 개인의 혈압 변동을 더 정확히 반영할 수 있으나, reference measurement와 recalibration이 필요할 수 있다. Hybrid calibration은 group-specific parameter와 subject-specific parameter를 결합한다. 논문은 일부 연구에서 hybrid approach가 solely subject-specific parameter보다 좋은 결과를 보였다고 설명한다.

그러나 calibration protocol은 연구마다 매우 다르다. 어떤 연구는 단일 entry-point sample만 사용하고, 어떤 연구는 수집 데이터의 80%를 calibration에 사용한다. Reference measurement interval도 몇 분에서 일주일 동안 하루 여러 번까지 다양하다. 논문은 이처럼 calibration number와 timing에 대한 표준화가 없다는 점을 중요한 문제로 지적한다.

### Risk of Bias 결과

Risk of bias 평가에서 18개 연구는 low risk, 24개 연구는 moderate risk로 분류되었다. Patient selection 관련 risk는 9개 연구에서 high, 6개 연구에서 unclear였다. High bias의 주요 원인은 제한된 subject 수, inappropriate exclusion criteria, young healthy population 중심 설계였다. Online database를 사용한 연구는 reference test와 index test 관련 bias가 낮은 경향이 있었지만, procedure나 reference device, testing method를 명확히 설명하지 않은 연구는 unclear bias로 평가되었다.

이 결과는 wearable PPG BP estimation 분야가 아직 methodology reporting과 validation design 측면에서 성숙하지 않았음을 보여준다. 특히 실제 device validation에서는 participant 수, age distribution, health condition, reference device, measurement protocol이 명확해야 한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 PRISMA guideline에 따른 systematic review와 meta-analysis를 수행했다는 점이다. 단순 narrative review가 아니라, 검색 database, inclusion criteria, reviewer pair, data extraction, risk of bias assessment를 명확히 제시하고 있다. 또한 ANSI/AAMI/ISO, BHS, IEEE 기준을 활용하여 포함 연구의 정확도 수준을 제한했다는 점도 연구 품질을 높인다.

두 번째 강점은 PWA와 PWV를 구분하여 비교했다는 점이다. 두 접근은 sensor 구성, calibration, 사용성, 정확도, heterogeneity 측면에서 크게 다르다. 논문은 PWA가 평균 bias 측면에서는 더 유리하지만 heterogeneity가 높고, PWV는 평균 bias는 더 크지만 일관성이 상대적으로 높다는 균형 잡힌 해석을 제공한다. 이는 향후 wearable BP device 설계에서 방법론 선택에 도움이 된다.

세 번째 강점은 기술적 성능뿐 아니라 practical usability를 강조했다는 점이다. 많은 BP estimation 논문은 MAE나 RMSE만 보고하지만, 이 review는 long-term monitoring, recalibration burden, motion artifact, device comfort, sensor placement, older hypertensive population inclusion 등을 함께 논의한다. 이는 실제 wearable device로 상용화하려면 반드시 고려해야 하는 요소이다.

네 번째 강점은 calibration 문제를 체계적으로 다룬 점이다. Cuff-less BP estimation에서 calibration은 정확도와 usability 사이의 핵심 trade-off이다. 논문은 generalized, personalized, hybrid calibration을 구분하고, calibration measurement 횟수와 timing이 표준화되어 있지 않다는 점을 명확히 지적한다. 이는 향후 연구의 중요한 방향이다.

다섯 번째 강점은 participant diversity 문제를 강하게 지적했다는 점이다. Wearable BP monitoring이 가장 필요한 대상은 older adults와 hypertensive patients인데, 많은 연구가 young healthy adults를 대상으로 수행되었다. 논문은 이러한 제한이 모델의 실제 적용 가능성과 reliability를 낮춘다고 비판한다. 이는 PPG-based BP estimation 연구에서 매우 중요한 reviewer 관점이다.

한계도 존재한다. 첫째, inclusion criteria가 엄격하여 ANSI/AAMI/ISO, BHS, IEEE 기준을 충족하지 못한 연구는 제외되었다. 이는 high-quality evidence 중심의 분석이라는 장점이 있지만, 새로운 알고리즘이나 실험적 device 중 아직 기준을 만족하지 못했지만 기술적으로 의미 있는 연구가 배제될 수 있다. 논문도 이를 selection bias 가능성으로 인정한다.

둘째, 포함 연구들의 protocol variability가 매우 크다. 연구마다 sensor type, sensor location, reference device, validation protocol, participant demographic, BP range limitation, outcome metric이 다르다. 이러한 heterogeneity 때문에 pooled estimate를 해석할 때 주의가 필요하다. 실제로 $I^2$가 98% 이상으로 매우 높아, 전체 평균 bias만으로 분야 전체의 성능을 단정하기 어렵다.

셋째, meta-analysis가 summary statistic 기반이라는 점도 한계이다. 개별 participant-level data를 통합한 것이 아니기 때문에, age, sex, hypertension status, device type, measurement duration, activity condition별 세부 subgroup 분석에는 제한이 있다. 특히 일부 연구는 DBP 결과를 제공하지 않거나, MAE만 제공하고 ME/SD를 제공하지 않는 등 report format이 달랐다.

넷째, algorithm-based study와 device-based study를 함께 다룰 때 해석의 복잡성이 생긴다. Online database에서 offline으로 평가한 algorithm은 실제 sensor artifact, device attachment, user movement, battery constraint, embedded computation 문제를 반영하지 못한다. 반면 device study는 실제 환경에 가깝지만 participant 수가 적거나 short-term testing에 머무는 경우가 많다. 두 연구 유형을 함께 해석하려면 이러한 차이를 명확히 고려해야 한다.

다섯째, long-term monitoring evidence가 부족하다. 일부 연구는 24-hour monitoring이나 one-time calibration 후 1개월 이상의 추정을 다루지만, 대부분은 short-term setting이다. 실제 wearable BP monitor는 수주에서 수개월 동안 안정적으로 동작해야 하고, calibration drift, sensor aging, user behavior 변화에 대응해야 한다. 현재 evidence는 이 요구를 충분히 만족하지 못한다.

여섯째, motion artifact와 environmental variability에 대한 검증이 부족하다. PPG sensor는 motion, ambient light, skin tone, contact pressure, sweat, body temperature에 민감하다. 많은 연구는 static condition에서 테스트되었고, exercise나 daily activity를 포함하지 않았다. 따라서 실제 일상생활에서의 performance는 불확실하다.

비판적으로 종합하면, 이 논문은 wearable PPG-based BP estimation 분야가 promising하지만 아직 clinical-grade long-term daily monitoring으로 가기에는 여러 검증 공백이 있음을 잘 보여준다. 특히 고령자와 hypertensive patient 대상의 대규모 장기 연구, motion artifact 대응, calibration 표준화, 다양한 환경에서의 device robustness가 핵심 과제이다.

## 6. 결론

이 논문은 wearable PPG 기반 cuff-less BP estimation 방법에 대한 systematic review와 meta-analysis를 수행하여, 현재 기술의 정확도와 실용성을 종합적으로 평가하였다. 전체 분석 결과, PPG 기반 wearable BP estimation은 SBP mean difference 4.14 mmHg, DBP mean difference 2.79 mmHg 수준의 비교적 낮은 평균 bias를 보였으며, short-term monitoring에서는 유망한 성능을 보여준다.

방법론적으로는 PWA와 PWV 모두 가능성이 있지만, PWA는 평균 bias 측면에서 더 유리한 결과를 보였고, PWV는 상대적으로 낮은 heterogeneity를 보였다. PWA 기반 방법은 단일 PPG sensor만으로 구현될 수 있어 wearable device 구조를 단순화할 수 있지만, 연구 간 성능 변동이 크고 signal quality 및 algorithm design에 민감하다. PWV 기반 방법은 physiological model에 기반한 장점이 있지만, 여러 sensor와 calibration, sensor placement 문제가 실용성을 제한한다.

이 연구의 주요 기여는 PPG 기반 wearable BP monitoring 기술을 정확도, calibration, sensor technology, participant diversity, long-term usability, risk of bias 관점에서 종합 평가했다는 점이다. 특히 많은 연구가 young healthy participants와 short-term static condition에 의존하고 있으며, 실제로 가장 중요한 older adults와 hypertensive patients, daily activity condition, long-term monitoring에 대한 검증이 부족하다는 점을 명확히 지적한다.

향후 연구에서는 더 다양한 demographic과 health condition을 포함한 대규모 validation, exercise와 sleep을 포함한 real-world continuous monitoring, motion artifact와 ambient light 변화에 robust한 sensor 및 algorithm 개발, personalized calibration과 generalized model 사이의 균형, calibration interval의 표준화가 필요하다. 또한 accelerometer, gyroscope, multi-wavelength PPG 등 보조 센서와의 결합은 motion artifact를 줄이고 signal quality를 높이는 데 도움이 될 수 있다.

결론적으로, wearable PPG-based BP estimation은 continuous, low-cost, user-friendly BP monitoring을 위한 중요한 기술적 가능성을 보여주지만, 현재 evidence만으로 장기적이고 임상적으로 신뢰 가능한 blood pressure monitor라고 단정하기는 어렵다. 이 분야가 실제 의료 및 가정 건강관리에서 활용되려면 medical professionals, data scientists, engineers 간의 interdisciplinary collaboration과 엄격한 long-term validation이 필수적이다.
