# PPG2ABP: Photoplethysmogram (PPG) 신호를 동맥혈압(ABP) 파형으로 변환하기

Nabil Ibtehaz, Sakib Mahmud, Muhammad E. H. Chowdhury, Amith Khandakar, Mohamed Arselene Ayari, Anas Tahir, M. Sohel Rahman

## 🧩 Problem to Solve

심혈관 질환은 전 세계적으로 주요 사망 원인이며, 혈압의 지속적인 모니터링은 필수적이다. 그러나 기존의 혈압 측정 방법들은 대부분 카테터 기반의 침습적이거나, 커프 기반으로 연속 모니터링에 부적합하며 불편하다는 한계가 있다. 광혈류측정(PPG) 신호는 비침습적이고 저렴하며 웨어러블 기기에 널리 사용되지만, 기존 연구들은 주로 PPG와 심전도(ECG) 신호를 함께 사용하거나, 수작업으로 특징을 추출하여 이상적인 형태의 PPG 신호만을 요구하는 제약이 있었다. 이 연구는 이러한 한계를 극복하고, PPG 신호만으로 동맥혈압(ABP) 파형 전체를 비침습적이고 지속적으로 추정하는 방법을 개발하는 것을 목표로 한다.

## ✨ Key Contributions

* **PPG2ABP 프레임워크 제안:** PPG 신호만을 사용하여 연속적인 ABP 파형을 추정하는 2단계 계단식(cascaded) 딥러닝 기반 방법을 제시한다.
* **독창적인 네트워크 아키텍처:** 깊이 지도 학습(deeply supervised)이 적용된 1D U-Net 기반의 **Approximation Network**로 ABP 파형을 근사하고, 1D MultiResUNet 기반의 **Refinement Network**로 이를 정제하는 접근 방식을 사용한다.
* **PPG 단독 입력 및 특징 자동 학습:** ECG와 같은 추가 신호나 수작업 특징 추출 없이 PPG 신호만으로 ABP 파형을 생성하여, 웨어러블 기기 적용을 용이하게 하고 비이상적인 신호에도 강건함을 보인다.
* **뛰어난 성능 달성:** 추정된 ABP 파형에서 계산된 이완기 혈압(DBP), 평균 동맥압(MAP), 수축기 혈압(SBP) 값이 기존 연구들보다 우수한 평균 절대 오차(MAE)를 달성했다 (DBP: $3.449 \pm 6.147 \text{ mmHg}$, MAP: $2.310 \pm 4.437 \text{ mmHg}$, SBP: $5.727 \pm 9.162 \text{ mmHg}$). DBP와 MAP 모두 British Hypertension Society (BHS) 표준에서 Grade A를 달성하고, Association for the Advancement of Medical Instrumentation (AAMI) 표준을 만족한다.
* **위상 지연 문제 해결:** PPG와 ABP 신호 간의 내재된 위상 지연 문제를 모델 학습을 통해 성공적으로 극복하여, 실제 애플리케이션에서 발생할 수 있는 신호 정렬 문제를 해결한다.

## 📎 Related Works

* **침습적/커프 기반 방법:** 카테터 기반 [5], 커프 기반 [6] 혈압 측정은 정확하나 침습적이거나 연속 측정에 부적합하다.
* **생체 신호 기반 비침습적 혈압 측정:**
  * **전통적 매개변수 기반:** 맥파 속도(PWV) [16], 맥파 도달 시간(PTT) [17], 맥파 도착 시간(PAT) 및 심실 수축 전 기간(PEP) [12]과 같은 매개변수 [18]-[22]를 이용한 수학적 모델링.
  * **고전적 머신러닝:** 주로 PPG 및 ECG 신호 [6], [15], [23]-[36]를 사용하여 DBP, SBP, MAP 값을 예측.
  * **딥러닝 기반 혈압 예측:** PPG 및/또는 ECG 신호 [37]-[46]를 이용한 딥러닝 모델이 BP 값을 예측. 그러나 대다수가 ECG 신호를 필요로 하거나 수작업 특징 추출에 의존.
  * **ABP 파형 추정/재구성 딥러닝:** 최근 1D 세그멘테이션 [47]-[48], 변이형 오토인코더(VAEs) [49], CycleGAN [50] 등을 활용하여 PPG 및/또는 ECG 신호로부터 ABP 파형을 추정하려는 시도가 있었으며, 본 연구는 이 분야의 선구적인 작업으로 평가된다.

## 🛠️ Methodology

1. **데이터셋:** PhysioNet의 MIMIC-III 데이터셋 [51]-[52]에서 추출된 PPG 및 ABP 동시 신호를 사용한다. 기존 연구 [53]-[54]의 전처리된 데이터를 바탕으로 DBP ($50 \text{ mmHg} \leq \text{DBP} \leq 165.17 \text{ mmHg}$) 및 SBP ($71.56 \text{ mmHg} \leq \text{SBP} \leq 199.99 \text{ mmHg}$) 범위를 확장하여 더 다양한 혈압 값을 포함했다.
2. **전처리 (Preprocessing):**
    * $8.192$초 길이($125 \text{ Hz}$ 샘플링 레이트로 $1024$ 샘플)의 PPG 신호 세그먼트를 추출한다.
    * Daubechies 8 (db8) 모 웨이블릿 [57]을 사용하여 10단계 웨이블릿 잡음 제거를 수행한다. 이 과정에서 저주파 ($0-0.25 \text{ Hz}$) 및 고주파 ($250-500 \text{ Hz}$) 성분을 제거하고 Rigrsure soft-thresholding [58], [59]을 적용한다.
    * PPG 신호는 글로벌 Min-Max 정규화를 거치며, ABP 파형은 추정 후 역정규화된다.
3. **근사 네트워크 (Approximation Network):**
    * 깊이 지도 학습(deeply supervised)이 적용된 1D U-Net 모델 [60]을 활용한다.
    * 이미지 분할을 위한 2D U-Net을 1D 신호 회귀 문제에 맞게 1D 컨볼루션, 풀링, 업샘플링 연산으로 변환한다.
    * 네트워크의 숨겨진 계층 학습을 유도하기 위해 디코더의 각 업샘플링 전에 중간 출력을 생성하고 보조 손실을 계산한다.
    * 손실 함수로는 Mean Absolute Error (MAE)를 사용한다.
4. **정제 네트워크 (Refinement Network):**
    * 근사 네트워크의 출력을 입력으로 받아 ABP 파형을 정제하는 1D MultiResUNet 모델 [62]을 사용한다.
    * MultiResUNet의 MultiRes 블록과 Res 경로를 활용하여 U-Net보다 정교한 특징 학습 및 파형 재구성을 수행한다.
    * 손실 함수로는 Mean Squared Error (MSE)를 사용한다.
5. **BP 매개변수 계산:** 정제 네트워크로부터 추정된 ABP 파형($\text{ABP}$)을 사용하여 DBP, MAP, SBP 값을 다음과 같이 계산한다:
    * $\text{SBP} = \text{max}(\text{ABP})$
    * $\text{DBP} = \text{min}(\text{ABP})$
    * $\text{MAP} = \text{mean}(\text{ABP})$
6. **학습 방법:** Adam optimizer [64]를 사용하여 모델을 100 epoch 동안 학습시키며, 20 epoch 동안 성능 개선이 없으면 조기 종료(patience)한다. 10-fold 교차 검증을 통해 최적 모델을 선정하고 독립된 테스트 데이터셋으로 평가한다.

## 📊 Results

* **ABP 파형 추정 정확도:**
  * PPG2ABP는 PPG 신호로부터 ABP 파형의 모양, 크기, 위상을 성공적으로 재구성했다. 특히 근사 네트워크에서 발생했던 피크 부분의 급격한 하강 오류를 정제 네트워크가 크게 개선했다 (예시에서 재구성 오류 $9.52 \text{ mmHg}$에서 $2.37 \text{ mmHg}$로 감소).
  * 전체 테스트 데이터셋에 대한 ABP 파형 재구성의 평균 절대 오차(MAE)는 $4.604 \pm 5.043 \text{ mmHg}$이다.
  * DBP, MAP, SBP 예측의 MAE는 각각 $3.449 \pm 6.147 \text{ mmHg}$, $2.310 \pm 4.437 \text{ mmHg}$, $5.727 \pm 9.162 \text{ mmHg}$이다.
  * MIMIC-III 데이터셋에서 발생하는 PPG와 ABP 신호 간의 위상 지연 문제를 효과적으로 극복했다.
* **BHS 표준 (British Hypertension Society Standard):**
  * DBP 및 MAP 예측에서 Grade A를 달성했다.
    * DBP: $\leq 5 \text{ mmHg}$ (82.836%), $\leq 10 \text{ mmHg}$ (92.157%), $\leq 15 \text{ mmHg}$ (95.734%)
    * MAP: $\leq 5 \text{ mmHg}$ (87.381%), $\leq 10 \text{ mmHg}$ (95.169%), $\leq 15 \text{ mmHg}$ (97.733%)
  * SBP 예측에서는 Grade B를 달성했다 ($\leq 5 \text{ mmHg}$ (70.814%), $\leq 10 \text{ mmHg}$ (85.301%), $\leq 15 \text{ mmHg}$ (90.921%)). 이는 SBP 예측에서 B등급을 얻은 최초의 연구 중 하나이다.
* **AAMI 표준 (Association for the Advancement of Medical Instrumentation Standard):**
  * DBP와 MAP 예측은 평균 오차($\mu$) $\leq 5 \text{ mmHg}$ 및 표준 편차($\sigma$) $\leq 8 \text{ mmHg}$ 기준을 만족했다 (DBP: $\mu=1.619 \text{ mmHg}$, $\sigma=6.859 \text{ mmHg}$; MAP: $\mu=0.631 \text{ mmHg}$, $\sigma=4.962 \text{ mmHg}$).
  * SBP는 평균 오차는 만족했으나, 표준 편차($\sigma=10.688 \text{ mmHg}$)가 기준치를 약간 초과했다.
* **통계 분석:**
  * Bland-Altman 플롯에서 DBP, MAP, SBP의 95% 일치 한계는 각각 $[-11.825:15.064]$, $[-9.095:10.357]$, $[-22.531:19.367] \text{ mmHg}$로 나타났다.
  * Pearson 상관 계수(PCC)는 DBP $0.894$, MAP $0.966$, SBP $0.936$으로 모두 강력한 양의 상관관계를 보였으며 ($p < .000001$), 결과의 통계적 유의성을 확인했다.

## 🧠 Insights & Discussion

PPG2ABP는 PPG 신호만으로 ABP 파형 전체를 성공적으로 추정하는 선구적인 연구이다. 이는 기존 연구들이 DBP, SBP, MAP와 같은 특정 값만을 예측하거나, 추가적인 ECG 신호 또는 수작업 특징 추출에 의존했던 한계를 극복한다. 딥러닝 모델이 데이터로부터 고수준의 추상적 특징을 적응적으로 학습하므로, 신호 품질에 대한 제약이 적고 노이즈가 많은 실제 환경 신호에도 강건하다.

가장 놀라운 점은, PPG2ABP가 DBP, SBP, MAP 값을 명시적으로 예측하도록 훈련되지 않았음에도 불구하고, 해당 값들을 목표로 훈련된 기존의 많은 방법들보다 뛰어난 정확도를 보였다는 것이다. 또한, PPG와 ABP 신호 간의 위상 지연 문제를 효과적으로 해결하여 실제 응용 가능성을 높였다.

스마트워치나 피트니스 밴드에 PPG 센서가 널리 탑재되어 있는 점을 고려할 때, PPG2ABP는 추가적인 고가 센서 없이도 대중 시장에 쉽게 배포될 수 있는 잠재력을 가진다. 이 기술은 의사들이 환자의 혈압을 지속적으로 모니터링하고, 혈압 변화 패턴을 사용자 행동과 연관시켜 통찰력 있는 발견을 가능하게 할 것이다.

향후 연구는 2단계 네트워크를 단일 최적화된 모델로 통합하고, PPG2ABP를 활용한 웨어러블 기기 애플리케이션 개발 및 임상 연구를 수행하는 방향으로 진행될 수 있다. 또한, 개인 맞춤형 보정(personalized calibration) 기법을 탐색하여 정확도를 더욱 향상시킬 수 있을 것으로 기대된다.

## 📌 TL;DR

PPG2ABP는 침습적 혈압 측정의 한계를 극복하기 위해 PPG 신호만으로 연속적인 동맥혈압(ABP) 파형을 비침습적으로 추정하는 2단계 딥러닝(U-Net 기반 근사 네트워크, MultiResUNet 기반 정제 네트워크) 프레임워크이다. 이 모델은 추가 ECG 신호나 수작업 특징 추출 없이도 ABP 파형의 모양, 크기, 위상을 정확히 재구성하며, 파형에서 계산된 DBP, MAP, SBP 값은 BHS Grade A (DBP, MAP) 및 AAMI 표준을 만족하는 등 기존 방법보다 뛰어난 성능을 보인다. 이는 웨어러블 기기를 통한 비침습적이고 지속적인 혈압 모니터링의 실현 가능성을 크게 높인다.
