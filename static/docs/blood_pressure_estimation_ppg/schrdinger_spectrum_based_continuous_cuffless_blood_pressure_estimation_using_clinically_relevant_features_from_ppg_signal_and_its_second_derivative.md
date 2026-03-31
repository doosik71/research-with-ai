# Schr$\ddot{\text{o}}$dinger Spectrum based Continuous Cuff-less Blood Pressure Estimation using Clinically Relevant Features from PPG Signal and its Second Derivative

Aayushman Ghosh, Sayan Sarkar, and Jayant Kalra

## 🧩 Problem to Solve

기존의 커프 기반 혈압(BP) 측정은 환자에게 불편함을 주고 장기간 연속 모니터링에 부적합하며, 침습적 방법은 위험성이 높습니다. 이 연구는 광용적맥파(Photoplethysmogram, PPG) 신호만을 사용하여 연속적이고 비침습적인 혈압을 정확하게 추정하는 방법을 개발하는 것을 목표로 합니다. 특히, 신호 품질이 저하되거나 형태가 변형된 PPG 신호에서도 신뢰할 수 있는 BP 추정 정확도를 유지하는 것이 주요 도전 과제입니다.

## ✨ Key Contributions

- **수정된 SCSA(Semi-Classical Signal Analysis) 프레임워크 도입**: 신호 재구성 시 `h-값`($h \in \mathbb{R}_{>0}$)을 최적화하는 새로운 오류 피드백 기반 재구성 알고리즘을 제안하여, 계산 부담을 최소화하면서도 우수한 신호 재구성 성능을 달성합니다.
- **알고리즘의 광범위한 일반화 능력 검증**: 잡음이 포함된 웨어러블 데이터셋 및 인위적인 잡음(다양한 SNR)이 주입된 데이터셋에 대한 견고성을 확인했습니다.
- **임상적으로 다양한 혈압 범주 평가**: 저혈압, 정상혈압, 고혈압 등 다양한 임상 혈압 범주에서 제안된 프레임워크의 효과를 평가했습니다.
- **결합된 특징을 통한 성능 향상**: 임상적으로 중요한 PPG 및 SDPPG(PPG의 2차 미분) 기반 형태학적 특징과 SCSA 기반 스펙트럼 특징을 결합하여 BP 추정 성능을 크게 향상시켰습니다. 이는 잡음이 있는 PPG 신호에서 PWA(Pulse Wave Analysis)와 PWD(Pulse Wave Decomposition) 기술의 장점을 모두 활용하는 이점을 제공합니다.

## 📎 Related Works

- **PWV (Pulse Wave Velocity) 기반 기술**: PTT(Pulse Transit Time) 및 PAT(Pulse Arrival Time)를 사용하여 BP를 추정하지만, 말초 동맥의 탄성 부족과 다중 센서 필요로 인해 정확도 및 실용성 한계가 있습니다.
- **PWA (Pulse Wave Analysis) 기반 기술**: 단일 PPG 신호의 파형 분석을 통해 특징을 추출하고 BP를 추정합니다. 시간, 진폭, 주파수 기반 특징을 사용하나, PPG 신호 품질 저하 시 정확도 문제가 발생합니다.
- **PDA (Pulse Decomposition Analysis) 기반 기술**: PPG 신호를 여러 구성파(솔리톤)로 분해하여 혈역학적 매개변수를 추정합니다. 가우시안, 시컨트 하이퍼볼릭(Sech) 함수 등을 기저 함수로 사용하지만, PPG 형태 변화 시 재구성 손실이 발생할 수 있습니다.
- **기존 SCSA (Semi-Classical Signal Analysis) 이론**: PPG를 비선형 솔리톤의 중첩으로 분해하는 방법론입니다. Laleg et al.의 연구에서 시도되었으나, 최적화된 매개변수 설정 부족으로 임상적 정확도가 미흡하고 계산 복잡성이 높은 문제가 있었습니다.

## 🛠️ Methodology

1. **데이터베이스**:
   - 주요 데이터셋: UCI Machine Learning Repository (Physionet MIMIC-II 기반).
   - 검증 데이터셋: in-silico Pulse Wave Database (PWD), Queensland Dataset, MIMIC-III Dataset, IEEE DataPort 웨어러블 데이터셋.
2. **전처리 및 특징점 검출**:
   - **PPG 신호**: 4차 Chebyshev-II 밴드패스 필터링 (0.5-20 Hz), z-score 정규화, 9차 다항식 피팅으로 저주파 성분 제거, 19샘플 Hempel 필터 적용.
   - **ABP 신호**: 4차 19프레임 Savitzky-Golay 필터 적용.
   - **특징점 검출**: AMPD (Automatic Multiscale-based Peak Detection) 알고리즘으로 PPG의 수축기 및 이완기 피크를 검출합니다. 딕로틱 노치(dicrotic notch)는 PPG 및 SDPPG를 이용한 3단계 절차로 검출합니다.
3. **SCSA (Error-Feedback Based Reconstruction)**:
   - PPG 신호 $y(t)$를 Schr$\ddot{\text{o}}$dinger 연산자 $H_h(t) = -h^2 \frac{d^2}{dt^2} - y(t)$의 포텐셜로 해석합니다. 여기서 $h$는 세미클래식 매개변수입니다.
   - 신호 $y(t)$는 이산 스펙트럼의 고유 함수 제곱의 합으로 재구성됩니다: $$y_h(t) = 4h \sum_{n=1}^{N_h} \kappa_{nh} \psi^2_{nh}(t)$$ 여기서 $\kappa_{nh}$는 $n$번째 음의 고유값, $\psi_{nh}$는 그에 상응하는 고유함수입니다.
   - **핵심 개선**: 제안된 오류 피드백 기반 재구성 알고리즘은 'h-값'을 최적화합니다. 특정 `h` 범위에 걸쳐 재구성 오류 ($\epsilon = y - y_h$)를 계산하고, 최소 오류를 제공하는 `h`를 선택하여 계산 비용과 재구성 정확도 사이의 균형을 맞춥니다. 이 연구에서는 최소 20단계 분해($N_h = 20$)가 PPG의 모든 형태학적 변화를 포착하는 데 필요함을 확인했습니다.
4. **특징 추출**: (총 38개 특징)
   - **SCSA 특징**: 분해된 음의 고유값 ($\kappa_{1h}, \dots, \kappa_{20h}$), 음의 고유값 합계, 수축기/이완기 불변 매개변수, $PPG_{PSI}$ (수축기 및 이완기 솔리톤의 시간적 차이를 기반으로 하는 피크화 및 가파름 현상 지표).
   - **PPG-SDPPG 특징**: SDPPG의 'a, b, c, d, e' 파형 진폭 비율 ($b/a, c/a, d/a, e/a$), 특징점 간의 시간 지연 ($T_a, T_{ba}, T_{cb}, T_{dc}, T_{ed}$), 노화 지수(AI). PPG의 $BW_{66}$ (맥파 높이 66%에서의 폭), $PIR_p$ (PPG 피크점 진폭과 계곡점 진폭의 비율).
5. **머신러닝 알고리즘**:
   - SVR, CatBoost, XGBoost, LightGBM 회귀 모델을 사용했습니다.
   - 중첩 K-겹 교차 검증(K=10)을 사용하여 모델 훈련 및 베이지안 최적화를 통한 하이퍼파라미터 튜닝을 수행했습니다.

## 📊 Results

- **회귀 모델 성능**: CatBoost와 LightGBM이 SVR 및 XGBoost에 비해 우수한 성능을 보였습니다. CatBoost는 MIMIC-II 기반 UCI 데이터셋에서 SBP (수축기 혈압)에 대해 MAE $5.37 \pm 5.56$ mmHg, DBP (이완기 혈압)에 대해 MAE $2.96 \pm 3.13$ mmHg를 달성했습니다. 상관 계수($r$)는 SBP 0.89, DBP 0.85, MAP 0.86으로 높은 회귀 성능을 보였습니다.
- **표준 준수**:
  - **AAMI-SP10**: CatBoost 모델의 평균 오차(M)는 모든 BP 기준에서 AAMI 한계($\le 5$ mmHg) 내에 있었으며, DBP 및 MAP의 표준 편차(SD)는 AAMI 기준($\le 8$ mmHg)을 충족했습니다 (SBP의 SD는 약간 초과).
  - **BHS 프로토콜**: CatBoost는 SBP, DBP, MAP 모든 BP 범주에서 Grade A를 달성했습니다. LightGBM은 DBP와 MAP에서 Grade A, SBP에서 Grade B를 획득했습니다.
- **특징 선택**: 특징 중요도 분석 결과, $BW_{66}$, $PIR_p$, $T_{cb}$, $c/a$, $b/a$와 같은 PPG 및 SDPPG 기반 특징들이 SCSA 기반 특징들보다 상대적으로 중요했습니다. 그러나 SCSA, SDPPG, PPG 특징의 조합이 최적의 추정 정확도를 제공했습니다.
- **임상 관련 데이터베이스 성능**: MIMIC-III 및 Queensland 데이터셋을 결합하여 구축된 저혈압, 정상혈압, 고혈압 범주 데이터베이스에서 제안된 알고리즘은 잘 작동했습니다. 특히 저혈압 및 고혈압 범주에서 좋은 성능을 보였으나, 정상혈압 SBP에서는 AAMI-SP10 및 BHS 기준을 약간 벗어나는 경향을 보였습니다.
- **잡음 스트레스 테스트**: 백색 가우시안 잡음을 주입한 PPG 신호에 대한 테스트 결과, SCSA 알고리즘은 우수한 잡음 내성 기능을 보여주었습니다. SNR 10 dB 이상에서 오차는 허용 가능한 수준이었고, 주요 특징 검출, 신호 재구성 능력 및 추정 정확도를 잘 유지했습니다. SDPPG 특징은 PPG 특징보다 잡음에 대한 내성이 더 우수했습니다.

## 🧠 Insights & Discussion

- **혁신적인 SCSA 재구성**: 오류 피드백 기반 SCSA 재구성 알고리즘은 기존 SCSA의 고유한 문제점인 계산 복잡성과 재구성 정확도 간의 상충 관계를 효과적으로 해결하며, 향상된 신호 분석 기반을 제공합니다.
- **견고성과 일반화**: 다양한 공공 데이터셋, 임상적으로 의미 있는 BP 범주(저혈압, 정상혈압, 고혈압), 그리고 잡음이 많은 웨어러블 데이터에 대한 광범위한 검증을 통해 이 알고리즘의 뛰어난 일반화 능력과 다양한 환경에서의 견고성을 입증했습니다.
- **특징 결합의 중요성**: SCSA에서 추출된 스펙트럼 정보와 PPG 및 SDPPG에서 얻은 임상적 형태 정보의 시너지가 혈압 추정 정확도 향상에 결정적인 역할을 함을 확인했습니다. 이는 기존 PWA 및 PDA 기술의 한계를 보완하는 중요한 접근 방식입니다.
- **실용적 잠재력**: 제안된 방법은 구현이 간단하여 자원 제약이 있는 환경(예: 웨어러블 장치)에서도 효율적으로 작동할 수 있는 잠재력을 가집니다. 이는 커프리스 혈압 모니터링 기술의 상용화에 기여할 수 있습니다.
- **한계 및 향후 과제**: 일부 임상 범주(예: 정상혈압 SBP)에서 AAMI/BHS 기준을 완전히 충족하지 못하는 부분은 향후 연구를 통해 개선할 여지가 있습니다. 딥러닝 기반 방법론과 비교하여 계산 효율성은 높지만, 지속적인 성능 개선을 위한 탐색이 필요합니다.

## 📌 TL;DR

이 논문은 기존 혈압 측정법의 한계를 극복하기 위해, PPG 신호와 그 2차 미분(SDPPG)을 활용한 커프리스 연속 혈압 추정 방법을 제안한다. 핵심은 '세미클래식 상수 $h$'를 최적화하는 새로운 오류 피드백 기반 SCSA(Schr$\ddot{\text{o}}$dinger Spectrum based Semi-Classical Signal Analysis) 재구성 알고리즘이다. 이 방법은 임상적으로 중요한 PPG/SDPPG 형태학적 특징과 SCSA 기반 스펙트럼 특징을 결합하여 CatBoost와 같은 머신러닝 모델로 혈압을 추정한다. UCI, MIMIC-III, Queensland, 웨어러블 데이터셋 등 다양한 데이터셋으로 광범위하게 검증한 결과, SBP MAE $5.37 \pm 5.56$ mmHg, DBP MAE $2.96 \pm 3.13$ mmHg를 달성하며 AAMI-SP10 기준을 충족하고 BHS 프로토콜에서 Grade A를 획득했다. 특히, SNR 10 dB까지 잡음 내성을 보여 다양한 임상 및 비임상 환경에서의 높은 견고성과 실용적 잠재력을 입증했다.
