# Global-Local Convolution with Spiking Neural Networks for Energy-efficient Keyword Spotting

Shuai Wang, Dehao Zhang, Kexin Shi, Yuchen Wang, Wenjie Wei, Jibin Wu, Malu Zhang

## 🧩 Problem to Solve

엣지 디바이스에 배포되는 키워드 스포팅(KWS) 시스템은 높은 정확도 외에도 에너지 효율성, 경량화가 필수적입니다. 기존 딥러닝(ANN) 기반 KWS 모델은 정확도는 뛰어나지만, 엣지 디바이스의 제한된 자원으로는 배포 및 장기 실행이 어렵습니다. 스파이킹 신경망(SNN)은 에너지 효율적이지만, 대부분의 SNN 기반 KWS 모델은 여전히 FFT나 MFCC와 같은 컴퓨팅 자원이 많이 드는 전처리 방식을 사용하거나, 희석 합성곱(dilated Conv1d)의 문제점(특징 중복 또는 지역 특징 손실)을 해결하지 못하며, 에너지 소비가 높은 Integrate-and-Fire (IF) 뉴런 모델을 사용하기도 합니다. 따라서 컴퓨팅 자원 소비가 적은 end-to-end 방식을 통해 고정밀, 경량, 에너지 효율적인 SNN 기반 KWS 모델을 개발하는 것이 주요 문제입니다.

## ✨ Key Contributions

- **Global-Local Spiking Convolution (GLSC) 모듈 제안:** 더 나은 에너지 효율적인 스파이킹 합성곱을 위해 고안되었습니다. 장기 음성 시퀀스의 길이(length)를 계층적으로 압축하면서 전역(global) 및 지역(local) 특징을 동시에 고려하며, SNN의 특성을 활용하여 특징 소실 문제를 완화하고 희소한(sparse) 특징 추출을 가능하게 합니다.
- **Bottleneck-PLIF 모듈 제안:** 경량화되고 효율적인 SNN 아키텍처를 위해 ResNet의 Bottleneck 구조와 효율적인 Parametric Leaky Integrate-and-Fire (PLIF) 뉴런을 결합하여 더 적은 파라미터로 높은 정확도를 달성하는 경량 분류기를 구축했습니다.
- **새로운 End-to-End SNN-KWS 모델 구축:** 제안된 GLSC 및 Bottleneck-PLIF 모듈을 통합하여 end-to-end SNN-KWS 모델을 구성했습니다. 이 모델은 SNN 기반 모델들 사이에서 정확도와 파라미터 효율성 면에서 경쟁력 있는 성능을 달성합니다.

## 📎 Related Works

- **ANN 기반 KWS 모델 [1]:** 높은 정확도를 달성하지만, 엣지 디바이스에 배포하기에는 에너지 및 리소스 제약이 큽니다.
- **기존 SNN 기반 KWS 연구 [8, 9]:** 에너지 효율성을 위해 SNN을 사용하지만, FFT [10], MFCC [11]와 같은 대량의 컴퓨팅 자원을 요구하는 전처리 방식을 사용하는 경우가 많습니다.
- **End-to-end SNN 모델 [13, 14]:** 원본 음성 신호를 직접 활용하여 전처리 비용을 줄이려 시도합니다.
  - **Dilated Conv1d:** $stride=1$인 경우 특징 중복 문제가 발생하며 [13], $stride ≠ 1$인 경우 지역 특징 손실이 발생할 수 있습니다 (Fig. 1).
  - **Integrate-and-Fire (IF) 뉴런 [14]:** 멤브레인 전위(membrane potential) 감쇠 메커니즘이 없어 스파이크 발생 빈도가 높아 에너지 소비가 증가할 수 있습니다 [15].
- **ResNet의 Bottleneck 구조 [17]:** 효율적으로 특징 정보를 통합하고 파라미터를 줄이는 데 영감을 주었습니다.
- **Parametric Leaky Integrate-and-Fire (PLIF) 뉴런 [15]:** 학습 가능한 감쇠 하이퍼파라미터를 통해 뉴런의 효율성을 높입니다.

## 🛠️ Methodology

본 논문에서는 KWS 시스템의 에너지 효율성과 경량화 문제를 해결하기 위해 두 가지 혁신적인 모듈을 포함하는 end-to-end SNN-KWS 모델을 제안합니다.

1. **Global-Local Spiking Conv1d (GLSC) 모듈:**

   - **구성:** Conv1d, Dilated Conv1d (D-Conv1d), 그리고 스파이킹 뉴런으로 이루어져 있습니다.
   - **작동 방식:**
     - 지역 특징 추출을 위한 Conv1d ($g_1(t) * f(t)$)와 전역 특징 추출을 위한 D-Conv1d ($g_2(t) * f(t)$)의 출력을 결합합니다.
     - 이후 Batch Normalization을 거쳐 스파이킹 뉴런의 발화 함수 $H$를 통과시켜 스파이크를 생성합니다.
     - 수학적 표현: $outputs = H(bn(g_1(t) * f(t)) + bn(g_2(t) * f(t)))$
   - **특징:** 스파이킹 뉴런은 $U_t$로부터의 잔류 멤브레인 전위를 고려하여 특징 소실을 완화하고, $summation + U_t$가 $V_{th}$보다 클 때만 정보를 전달하여 불필요한 특징의 축적을 방지하고 희소한 특징 벡터를 생성합니다. 이를 통해 지역 및 전역 특징 간의 균형을 효과적으로 달성하고 긴 음성 시퀀스의 길이를 압축합니다.

2. **Bottleneck-PLIF 모듈:**
   - **구성:** ResNet의 Bottleneck 구조와 Parametric Leaky Integrate-and-Fire (PLIF) 뉴런을 결합합니다.
   - **PLIF 뉴런:** 기존 LIF 뉴런의 고정된 누출 계수 $τ$ 대신 학습 가능한 $k(a)$를 사용하여 멤브레인 전위 $U$의 감쇠율을 최적화합니다. 이는 뉴런 출력을 다양하게 만들고 효율성을 높입니다.
     - $U_{t+1,n}^i = U_{t,n}^i - k(a) (U_{t,n}^i - \sum_{j=1}^{l(n-1)} w_{ij}^n O_{t+1,n-1}^j)$
   - **Bottleneck 구조:** 1x1 합성곱 ($f_1$)을 사용하여 입력 특징을 손상시키지 않고 채널을 융합하고 차원을 감소시킵니다. 이후 3x3 합성곱 ($f_3$)을 통해 스파이크 특징을 추가로 처리합니다.
   - **작동 방식:** $Outputs = H[f_1(H(f_3(H(f_1(Input))) + f_1(Input))]$
   - **특징:** 더 적은 파라미터로 특징 정보를 효율적으로 통합하고 분류 정확도를 높여 경량 분류기를 구현합니다.

- **학습 방법:** 전체 모델은 STBP [25] (Spatio-Temporal Backpropagation) 방법을 사용하여 직접 학습됩니다.

## 📊 Results

- **데이터셋:** Google Speech Commands (GSC) Dataset V1 (12-클래스) 및 V2 (12-클래스, 35-클래스)에서 실험을 수행했습니다.
- **정확도 및 모델 크기 (Table 1):**
  - GSC-V1 (12 클래스): 93.0% 정확도, 70.1K 파라미터.
  - GSC-V2 (12 클래스): 94.4% 정확도, 70.1K 파라미터.
  - GSC-V2 (35 클래스): 92.9% 정확도, 80.2K 파라미터.
  - 타 SNN 기반 KWS 모델에 비해 현저히 적은 파라미터로 경쟁력 있는 또는 더 우수한 정확도를 달성했습니다 (예: ST-Attention-SNN (2170K, 95.1%) 대비 SNN-KWS (70.1K, 94.4%) on GSC-V2 12-class).
- **에너지 효율성:**
  - 동일한 구조의 ANN 모델 대비 10배 이상의 에너지 절감 효과를 달성했습니다.
  - 에너지 비율은 $Energy_{rate} = (AC/MAC) * SpikingRate * TimeSteps$ 공식을 사용합니다.
  - $AC/MAC = 1/7$ [31], 평균 스파이크 발화율은 8.3% (Fig. 5), $TimeSteps = 8$로 계산됩니다.
- **Ablation Study (Fig. 6):**
  - **GLSC 모듈:** 단일 합성곱 방식보다 지속적으로 우수한 성능과 수렴 속도를 보였습니다. GLSC에서 스파이킹 뉴런을 ANN의 연속 활성화 함수로 대체한 GLC-ANN과의 비교를 통해, 스파이킹 뉴런이 특징 소실 문제를 해결하는 데 핵심적인 역할을 함을 입증했습니다.
  - **Bottleneck-PLIF 모듈:** 파라미터가 감소하더라도 성능 저하가 미미하며, 100K 미만의 파라미터로도 93%의 정확도를 유지하는 등 경량화와 효율성을 검증했습니다.

## 🧠 Insights & Discussion

본 논문에서 제안하는 SNN-KWS 모델은 엣지 디바이스의 높은 정확도, 경량 설계, 에너지 효율성이라는 세 가지 주요 요구 사항을 모두 충족합니다. GLSC 모듈은 기존 end-to-end 합성곱의 한계(특징 중복, 지역 특징 손실)를 효과적으로 극복하며, 스파이킹 뉴런의 특성을 활용하여 전역 및 지역 특징을 통합하고 희소하며 의미 있는 특징을 추출합니다. Bottleneck-PLIF 모듈은 PLIF 뉴런의 학습 가능한 감쇠율과 Bottleneck 구조의 파라미터 효율성을 결합하여 모델을 경량화하면서도 높은 분류 정확도를 유지합니다. 특히, 계산된 에너지 효율성 결과는 SNN이 ANN에 비해 실질적인 에너지 절감 이점을 제공하며, 제안된 아키텍처의 낮은 스파이킹 비율이 이에 기여함을 보여줍니다. 미래 연구에서는 이 모델을 뉴로모픽 칩에 실제로 구현할 계획입니다.

## 📌 TL;DR

**문제:** 엣지 디바이스를 위한 고정밀, 경량, 에너지 효율적인 키워드 스포팅(KWS) 모델 개발이 필요합니다. 기존 ANN은 에너지 소모가 크고, SNN은 비효율적인 전처리나 뉴런 모델 사용으로 한계가 있습니다.
**방법:** 본 논문은 end-to-end SNN-KWS 모델을 제안합니다. 이 모델은 전역 및 지역 특징을 동시에 추출하고 특징 소실을 완화하는 **Global-Local Spiking Convolution (GLSC) 모듈**과, PLIF 뉴런 및 Bottleneck 구조를 활용하여 적은 파라미터로 효율적인 분류를 수행하는 **Bottleneck-PLIF 모듈**로 구성됩니다.
**결과:** Google Speech Commands 데이터셋에서 제안된 모델은 타 SNN 기반 모델들과 비교하여 경쟁력 있는 정확도(예: GSC-V2 12-class에서 94.4%)를 달성하면서도 모델 크기를 현저히 줄였으며(70.1K 파라미터), ANN 대비 10배 이상의 에너지 절감 효과를 보였습니다. 이는 엣지 디바이스에 적합한 솔루션임을 입증합니다.
