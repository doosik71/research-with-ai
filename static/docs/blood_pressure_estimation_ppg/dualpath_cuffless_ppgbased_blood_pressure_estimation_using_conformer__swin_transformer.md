# Dual-Path Cuffless PPG-Based Blood Pressure Estimation Using Conformer & Swin Transformer

* **저자**: Caoyueshan Fan, Yiting Wei, Melanie Qiu, Mostafa Haghi, Nima TaheriNejad
* **발표연도**: 2025

## 1. 논문 개요

이 논문은 단일 PPG(Photoplethysmogram) 신호만을 이용하여 커프 없이 연속적으로 혈압을 추정하는 딥러닝 프레임워크를 제안한다. 연구의 핵심 목표는 PPG에서 직접 SBP(Systolic Blood Pressure)와 DBP(Diastolic Blood Pressure)를 회귀하는 데 그치지 않고, ABP(Arterial Blood Pressure) waveform을 함께 재구성함으로써 혈압 추정의 정확도와 생리학적 일관성을 동시에 높이는 것이다.

고혈압은 전 세계적으로 매우 큰 질병 부담을 가지며, 조기 발견과 장기적 관리가 중요하다. 기존 혈압 측정은 대체로 cuff-based intermittent device에 의존한다. 이 방식은 임상적으로 널리 사용되지만, 반복 측정 시 불편하고 수면을 방해할 수 있으며 움직임 제한과 사용자 부담이 크다. 특히 가정 환경이나 웨어러블 기반 장기 모니터링에는 적합하지 않다.

PPG는 광학 센서를 통해 말초 혈관의 혈액량 변화를 측정하는 비침습 신호이다. ECG와 함께 사용하여 PTT(Pulse Transit Time) 또는 PAT(Pulse Arrival Time)를 계산하는 방식도 널리 연구되었지만, ECG를 안정적으로 얻는 것은 실제 착용 환경에서 어렵다. 따라서 논문은 PPG 단일 신호만으로 연속적이고 해석 가능한 혈압 추정을 수행하는 문제를 다룬다.

논문이 설정한 연구 문제는 다음과 같다. PPG는 비선형적이고 non-stationary한 생리 신호이며, 혈압 정보는 한 pulse 내부의 morphology와 여러 pulse cycle에 걸친 rhythm dynamics에 동시에 포함된다. 기존 LSTM, GRU, 1D-CNN 기반 모델은 대체로 morphology 또는 temporal dynamics 중 하나에 치우치는 경향이 있다. 저자들은 이러한 한계를 해결하기 위해 Conformer-Transformer와 1D Swin Transformer라는 두 구조를 PPG 기반 혈압 추정에 적용한다.

제안 방법은 PPG를 입력으로 받아 ABP waveform을 재구성하고, 그 waveform으로부터 SBP와 DBP를 추정한다. 실험은 UCI-BP와 MIMIC-BP라는 두 공개 데이터셋에서 수행되며, 두 모델 모두 BHS Grade A와 AAMI 기준을 만족한다고 보고된다. Conformer-Transformer는 SBP MAE 2.979 mmHg, DBP MAE 1.603 mmHg로 가장 낮은 오차를 달성했고, Swin Transformer는 SBP MAE 3.034 mmHg, DBP MAE 1.714 mmHg를 달성했다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 혈압 추정이 단순한 scalar regression 문제가 아니라, PPG 신호의 waveform morphology와 cross-cycle rhythm을 함께 이해해야 하는 구조적 신호 모델링 문제라는 점이다. PPG 한 주기 안에는 systolic peak, diastolic trough, dicrotic notch, foot onset과 같은 morphology 정보가 들어 있다. 반면 여러 주기에 걸친 박동 간 간격, amplitude 변화, rhythm continuity는 장기적인 혈압 변화와 관련된다. 따라서 좋은 모델은 짧은 구간의 세밀한 파형 구조와 긴 구간의 리듬 변화를 모두 학습해야 한다.

이를 위해 논문은 두 가지 경로를 제안한다. 첫 번째는 **Conformer-Transformer**이다. Conformer는 convolution과 self-attention을 결합한 구조이다. Convolution은 local waveform morphology를 포착하고, self-attention은 long-range rhythmic dependency를 학습한다. 여기에 Transformer decoder를 결합하여 PPG로부터 ABP waveform을 sequence-to-sequence 방식으로 재구성한다.

두 번째는 **1D Swin Transformer**이다. Swin Transformer는 원래 이미지 처리에서 hierarchical window-based attention을 수행하는 구조이다. 논문은 이를 1D physiological signal에 맞게 변형한다. 짧은 window에서는 single-beat detail을 포착하고, shifted window mechanism을 통해 window 사이의 상호작용과 cross-beat dynamics를 학습한다. 이 구조는 상대적으로 가볍고 지연시간이 낮아 wearable deployment에 유리하다고 설명된다.

이 논문의 중요한 차별점은 **dual-output learning objective**이다. 기존의 많은 모델은 PPG에서 SBP와 DBP 값을 직접 회귀한다. 반면 이 논문은 ABP waveform을 먼저 재구성하고, 그 waveform으로부터 SBP와 DBP를 도출한다. 이는 예측값이 생리적으로 해석 가능한 waveform 구조와 연결된다는 장점이 있다. 즉 모델은 단순히 숫자를 맞히는 것이 아니라, 혈압파의 형태까지 재현하도록 학습된다.

또한 저자들은 구조와 과제의 정렬(structure-task alignment)을 강조한다. Conformer-Transformer는 장기 rhythm modeling과 ABP waveform reconstruction에 강하고, Swin Transformer는 multi-scale local perception과 낮은 inference latency에 강하다. 이 두 모델은 서로 경쟁적이라기보다 보완적이다.

## 3. 상세 방법 설명

### 3.1 전체 파이프라인

전체 프레임워크는 PPG 단일 신호를 입력으로 사용한다. 입력 PPG는 전처리, 정렬, segmentation, normalization 과정을 거친 뒤 모델에 들어간다. 모델은 ABP waveform을 출력하고, 이 재구성된 waveform에서 SBP와 DBP가 계산 또는 추정된다.

전체 흐름은 다음과 같이 정리할 수 있다.

$$
PPG \rightarrow Preprocessing \rightarrow Model \rightarrow \hat{ABP} \rightarrow \hat{SBP}, \hat{DBP}
$$

여기서 핵심은 모델 출력이 단일 혈압값이 아니라 길이 $T$의 ABP waveform이라는 점이다. 논문은 입력 PPG sequence를 $X \in \mathbb{R}^{T}$, 예측 ABP waveform을 $\hat{Y} \in \mathbb{R}^{T}$로 정의한다. 참조 ABP waveform은 $Y \in \mathbb{R}^{T}$이다.

### 3.2 데이터셋

논문은 두 공개 데이터셋을 사용한다.

첫 번째는 **UCI-BP dataset**이다. 이 데이터셋은 UCI Machine Learning Repository에서 제공되며, MIMIC-II Waveform Database에서 정제된 subset이다. 총 12,000개의 recording이 포함되어 있고, 각 recording 길이는 8초에서 592초까지 다양하다. 각 record는 PPG, ECG, ABP 신호를 포함하며 sampling rate는 125 Hz이다. 이 연구에서는 모델 입력으로 PPG만 사용한다. ABP는 invasive radial arterial catheterization으로 얻어진 참조 신호이며, 혈압 label을 만들기 위한 기준 신호로 사용된다.

두 번째는 **MIMIC-BP dataset**이다. 이 데이터셋은 MIMIC-III Waveform Database Matched Subset에서 cuffless BP estimation을 위해 curated된 데이터셋이다. 1,524명의 ICU patient가 포함되어 있고, 각 subject는 30개의 30초 synchronized recording segment를 제공한다. 각 segment에는 ABP, PPG, ECG, RESP 신호가 포함되며 sampling rate는 125 Hz이다. 전체 데이터는 380시간 이상의 생리 신호를 포함한다. 논문에 따르면 이 데이터셋은 2024년 말에 최신 버전이 공개되었으며, 다양한 환자 profile과 measurement condition을 포함한다.

두 데이터셋 모두에서 모델 입력은 PPG signal only이다. ABP waveform은 참조 label로만 사용된다.

### 3.3 전처리

데이터는 training, validation, test set으로 8:1:1 비율로 나뉜다. 논문은 recording level에서 split을 수행하여 같은 sequence의 segment가 서로 다른 split에 나타나지 않도록 했다고 설명한다. 이는 segment-level leakage를 줄이는 데 도움이 된다. 다만 제공된 텍스트 기준으로 subject-wise split인지까지는 명확하지 않다. MIMIC-BP처럼 subject가 여러 segment를 제공하는 경우, 동일 subject가 여러 split에 나뉘지 않았는지 여부는 일반화 성능 해석에 중요하다.

UCI-BP는 원 데이터 제공자가 이미 smoothing, abnormal BP 또는 HR block 제거, discontinuity 제거, pulse-to-pulse variability에 대한 autocorrelation filtering 등을 적용한 상태로 제공된다. MIMIC-BP에 대해서는 저자들이 추가 denoising을 수행한다. PPG 신호에는 3rd-order Butterworth band-pass filter를 적용하며 주파수 범위는 0.5–8 Hz이다. ABP 신호에는 2nd-order Butterworth filter를 적용하며 범위는 0.4–12 Hz이다. 이후 moving average smoothing을 적용한다. 또한 ABP baseline component를 추출하기 위한 low-pass branch도 설계되었다고 설명한다.

Training과 validation set에서는 PPG와 ABP를 cross-correlation을 이용해 정렬한다. 구체적으로 ABP를 shift하여 PPG와의 correlation이 최대가 되도록 한다. 이는 PPG와 ABP 사이의 생리적 시간 지연을 보정하기 위한 과정이다. 그러나 test set에는 alignment를 적용하지 않는다. 이는 test set 독립성을 유지하기 위한 의도로 설명된다.

Segmentation은 2초 길이의 non-overlapping window로 수행된다. Sampling rate가 125 Hz이므로 각 window는 약 250 sample을 포함한다.

$$
2 \times 125 = 250
$$

논문은 2초 window가 일반적으로 최소 하나의 cardiac cycle을 포함하고, 혈압 추정에 필요한 waveform feature를 보존한다고 설명한다. 다만 2초 window는 장기적인 rhythm dynamics를 충분히 포함하기에는 상대적으로 짧을 수 있다. 이 한계를 보완하기 위해 Conformer와 Swin Transformer가 cross-segment 또는 cross-cycle rhythmic continuity를 학습하도록 설계된 것으로 이해된다.

Normalization은 다음과 같이 수행된다. PPG 입력은 각 segment의 평균과 표준편차를 사용하여 z-score normalization을 적용한다. 이를 training, validation, test set 각각에서 독립적으로 수행하여 data leakage를 방지한다. Training set의 ABP는 per-segment z-score normalization을 적용하여 학습을 안정화한다. Inference 시에는 예측된 ABP waveform을 원래 amplitude scale로 rescale한다. Test set의 ABP reference는 원본 형태로 유지한다.

### 3.4 Conformer-Transformer 모델

Conformer-Transformer는 ABP waveform estimation을 위한 sequence-to-sequence regression architecture이다. 구조는 Conformer encoder와 Transformer decoder로 구성된다.

Conformer encoder는 Multi-Head Self-Attention(MHSA)과 convolution module을 결합한다. 여기에 Macaron-style feed-forward network가 앞뒤로 배치된다. 이 구조는 global rhythmic dependency와 local waveform morphology를 동시에 학습하도록 설계되었다.

각 Conformer block의 계산은 다음과 같이 정의된다.

$$
\tilde{x}_i = x_i + \frac{1}{2}FFN(x_i)
$$

$$
x'_i = \tilde{x}_i + MHSA(\tilde{x}_i)
$$

$$
x''_i = x'_i + Conv(x'_i)
$$

$$
y_i =
LayerNorm
\left(
x''_i + \frac{1}{2}FFN(x''_i)
\right)
$$

첫 번째 수식은 입력 $x_i$에 feed-forward transformation을 절반 비율로 residual하게 더한다. Macaron-style FFN은 attention과 convolution 전후에 feed-forward network를 배치하여 비선형 표현력을 강화하는 방식이다. 두 번째 수식에서 MHSA는 전체 sequence 내 위치 간 의존성을 학습한다. PPG에서는 여러 pulse cycle 사이의 rhythm과 장거리 변화를 포착하는 역할을 한다. 세 번째 수식에서 convolution module은 local morphology를 포착한다. 논문은 이 convolution module이 GLU, 1D depthwise convolution, batch normalization, Swish activation 등을 포함한다고 설명한다. 마지막으로 LayerNorm을 통해 학습 안정성을 확보한다.

Transformer decoder는 masked multi-head self-attention, encoder-decoder cross-attention, position-wise feed-forward network로 구성된다. 각 sublayer에는 residual connection과 LayerNorm이 적용된다.

$$
Output = LayerNorm(x + Sublayer(x))
$$

Decoder는 먼저 이전에 생성된 output에 대해 masked attention을 수행하고, 이후 encoder에서 얻은 PPG context와 cross-attention을 수행한다. 마지막으로 각 time step의 hidden vector를 sequence-wise linear projection에 통과시켜 예측 ABP waveform을 생성한다.

입력과 출력은 다음과 같이 정의된다.

$$
X \in \mathbb{R}^{T}
$$

$$
\hat{Y} \in \mathbb{R}^{T}
$$

여기서 $X$는 PPG sequence, $\hat{Y}$는 예측 ABP waveform이다. 학습 손실은 MSE이다.

$$
L_{MSE} = \frac{1}{T} \sum_{i=1}^{T} (Y_i - \hat{Y}_i)^2
$$

이 손실은 예측 waveform과 참조 ABP waveform의 point-wise 차이를 최소화한다. 단일 SBP/DBP 회귀보다 더 강한 supervision을 제공하며, waveform shape를 학습하게 만든다.

### 3.5 1D Swin Transformer 모델

두 번째 모델은 1D Swin Transformer이다. 원래 Swin Transformer는 image recognition을 위해 2D window-based attention을 사용하는 계층적 vision transformer이다. 이 논문은 이를 PPG 같은 1D physiological signal에 맞게 변형한다.

Swin1D Block은 두 개의 sub-block으로 구성된다. 첫 번째는 1D Window-based Multi-Head Self-Attention(W-MSA)이고, 두 번째는 1D Shifted Window Multi-Head Self-Attention(SW-MSA)이다. W-MSA는 non-overlapping window 내부에서 self-attention을 수행하여 local feature를 포착한다. SW-MSA는 window partition을 shift하여 서로 다른 window 사이의 정보 교환을 가능하게 한다.

각 sub-block은 다음의 transformer pattern을 따른다. 먼저 LayerNorm을 적용하고, attention module을 수행한다. 그 결과를 residual connection으로 더한다. 다시 LayerNorm을 적용한 뒤 two-layer MLP를 통과시키고 residual connection을 적용한다.

1D Swin Transformer의 최종 sequence representation은 다음과 같이 표현된다.

$$
Z \in \mathbb{R}^{T \times d}
$$

마지막 linear projection layer는 이를 ABP waveform으로 변환한다.

$$
\hat{Y} \in \mathbb{R}^{T \times 1}
$$

이 모델의 장점은 hierarchical multi-scale temporal modeling이다. 짧은 window는 single-beat morphology를 학습하고, shifted window는 cross-beat interaction을 가능하게 한다. 따라서 전체 attention을 모든 time step에 적용하는 것보다 계산 효율이 좋고, local fluctuation과 noise에 강할 수 있다.

### 3.6 성능 평가 지표

논문은 여러 평가 지표를 사용한다. Pearson correlation coefficient $R$은 예측값과 실제값의 선형 상관성을 나타낸다. ME(Mean Error)는 systematic bias를 측정한다. MAE는 평균 절대 오차를 나타내며, SD는 오차의 표준편차, RMSE는 제곱 오차 평균의 제곱근이다.

오차를 $e_i = y_i - \hat{y}_i$라고 하면 주요 지표는 다음과 같이 쓸 수 있다.

$$
ME =
\frac{1}{n}
\sum_{i=1}^{n}
e_i
$$

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|e_i|
$$

$$
SD =
\sqrt{
\frac{1}{n-1}
\sum_{i=1}^{n}
(e_i - ME)^2
}
$$

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
e_i^2
}
$$

MAE는 평균적인 오차 크기를 직관적으로 보여준다. ME는 모델이 혈압을 지속적으로 높게 또는 낮게 예측하는지를 알려준다. SD와 RMSE는 오차의 안정성과 큰 오차 발생 가능성을 반영한다.

임상적 기준으로는 AAMI와 BHS를 사용한다. AAMI 기준은 ME가 $\pm 5$ mmHg 이하이고 SD가 8 mmHg 이하일 것을 요구한다. 논문은 최소 255 measurement 이상에서 이 기준을 적용한다고 설명한다. BHS 기준은 절대 오차가 5, 10, 15 mmHg 이내에 들어오는 비율을 기준으로 Grade A부터 D까지 평가한다.

### 3.7 해석가능성 분석

논문은 단순 성능뿐 아니라 모델이 생리학적으로 의미 있는 지점에 주목하는지 분석한다. 이를 위해 Landmark Overlap Score(LOS)를 사용한다. Saliency sequence는 Integrated Gradients(IG)와 Grad-CAM으로 얻는다. 이후 saliency가 높은 상위 20% point를 $\Omega_s$로 정의한다.

PPG waveform에서 foot onset, systolic peak, dicrotic notch의 세 landmark를 검출하고, 각 landmark 주변 $\pm 100$ ms window를 $\Omega_L$로 정의한다. LOS는 다음과 같다.

$$
LOS =
\frac{|\Omega_s \cap \Omega_L|}
{|\Omega_s|}
$$

이 값은 모델이 중요하다고 판단한 지점 중 실제 생리학적으로 의미 있는 landmark 주변에 위치한 비율을 의미한다. 2초 segment는 125 Hz에서 약 250 sample이고, 각 $\pm 100$ ms window는 약 25 sample에 해당한다. 세 landmark가 차지하는 범위는 약 75 sample, 즉 30% 정도이다. window overlap을 고려하면 random baseline은 약 25–30%로 추정된다.

실험에서 Conformer-Transformer는 IG 0.329, Grad-CAM 0.388을 얻었고, Swin Transformer는 IG 0.395, Grad-CAM 0.382를 얻었다. 이 값들은 baseline보다 높으므로, 두 모델이 임의 위치가 아니라 physiologically meaningful landmark에 더 많이 주목한다는 근거로 제시된다.

또한 저자들은 rhythmic continuity에 대한 의존성을 평가하기 위해 DCRC(Disrupting Cross-segment Rhythmic Continuity) 실험을 수행한다. 이 실험에서는 segment 내부 morphology와 target label의 temporal order는 유지하되, 입력 signal segment를 random shuffling하여 cross-segment rhythmic continuity만 깨뜨린다. 성능이 크게 저하되었다는 결과는 모델이 isolated segment morphology만 보는 것이 아니라 rhythm context에도 의존한다는 점을 보여준다.

## 4. 실험 및 결과

### 4.1 전체 성능

논문은 UCI-BP와 MIMIC-BP 두 데이터셋에서 실험을 수행한다. 결과는 여러 표로 제시되며, 제공된 텍스트에서 핵심 성능은 다음과 같이 요약된다.

UCI-BP에서 Conformer-Transformer는 SBP MAE 2.979 mmHg, DBP MAE 1.603 mmHg를 달성했다. 이는 이전 연구 DeepCNAP보다 각각 9.6%, 8.4% 개선된 결과로 설명된다. 또한 ABP waveform reconstruction에서도 MAE 3.005 mmHg, $R=0.978$로 가장 좋은 성능을 보였다고 보고한다.

Swin Transformer는 SBP MAE 3.034 mmHg, DBP MAE 1.714 mmHg를 달성했다. Conformer-Transformer보다 약간 높은 오차를 보이지만, inference latency가 낮고 계산 복잡도가 작아 wearable deployment에 유리하다고 설명된다.

MIMIC-BP에서는 Conformer-Transformer가 SBP/DBP MAE 3.414/1.774 mmHg를 달성했다고 제시된다. Swin Transformer는 DBP 예측에서 비슷한 수준의 정확도를 보이고, ME가 더 낮아 noise에 대한 robustness가 있다고 해석된다.

### 4.2 Denoising 전후 성능

논문은 MIMIC-BP에서 raw PPG와 denoised PPG 조건을 비교한다. 제공된 텍스트에 따르면 denoising 후 성능이 개선되며, Conformer-Transformer의 SBP RMSE는 5.722 mmHg에서 4.484 mmHg로 감소한다.

이는 PPG 신호 품질이 혈압 추정 성능에 큰 영향을 준다는 것을 보여준다. 특히 PPG는 motion artifact, sensor contact, baseline drift, high-frequency noise에 민감하므로 적절한 preprocessing이 중요하다. 동시에 모델이 raw signal에서도 강한 성능을 유지했다는 점은 robustness 측면에서 긍정적으로 해석된다.

### 4.3 BHS 및 AAMI 기준

논문은 모든 실험 결과가 BHS Grade A와 AAMI 기준을 만족한다고 보고한다. 이는 모델의 평균 오차와 오차 분산이 임상적 혈압 측정 기준에서 허용 가능한 범위에 들어간다는 의미이다.

다만 제공된 텍스트에는 Table III, IV의 모든 수치가 포함되어 있지 않다. 따라서 각 모델과 각 데이터셋에서 5, 10, 15 mmHg 이하 오차 비율이 구체적으로 얼마인지는 확인할 수 없다. 원문에서는 해당 표에 세부 수치가 포함되어 있을 것으로 보인다. 제공된 텍스트 기준으로는 “두 모델 모두 BHS Grade A와 AAMI 기준을 만족했다”는 결론만 확인할 수 있다.

### 4.4 Correlation 및 bias 분석

Swin Transformer의 MIMIC-BP 결과에 대해 regression plot과 Bland-Altman plot이 제시된다. Pearson correlation coefficient는 SBP와 DBP 모두 0.95를 초과한다고 설명된다. 이는 예측값이 실제 혈압값의 변화 추세를 잘 따라간다는 의미이다.

Bland-Altman 분석에서는 대부분의 data point가 $\pm 5$ mmHg 범위 안에 위치한다고 설명된다. Swin Transformer의 ME는 SBP에서 -2.589 mmHg, DBP에서 0.821 mmHg이다. SBP의 ME가 음수라는 것은 평균적으로 SBP를 약간 높게 예측하거나 낮게 예측하는 방향 중 하나의 bias가 있음을 의미한다. 논문 정의에 따라 $ME = y - \hat{y}$라면 음수 ME는 예측값이 실제값보다 평균적으로 높다는 뜻이다. DBP의 ME는 양수이므로 예측값이 실제보다 약간 낮은 경향을 가질 수 있다.

논문은 전반적으로 systematic overestimation 또는 underestimation 경향은 크지 않다고 해석한다. 다만 SBP ME -2.589 mmHg는 AAMI 기준 안에는 들어가지만, 완전히 0에 가까운 것은 아니므로 실제 deployment에서는 calibration bias를 추가로 확인할 필요가 있다.

### 4.5 Error distribution 분석

두 모델의 error histogram은 bell-shaped and centralized pattern을 보인다고 설명된다. 대부분의 SBP 및 DBP prediction error가 $\pm 5$ mmHg 범위에 들어간다. UCI-BP에서는 Conformer-Transformer가 SBP에 대해 더 smooth한 error distribution과 약간 낮은 SD를 보여 복잡한 rhythm pattern modeling에 강하다고 해석된다. MIMIC-BP에서는 Conformer-Transformer가 약간 낮은 mean error를 보이고, Swin Transformer는 DBP에서 경쟁력 있는 SD를 보인다.

이 결과는 두 모델의 성격 차이를 보여준다. Conformer-Transformer는 장기 rhythm과 waveform reconstruction에 강하고, Swin Transformer는 window-based attention 덕분에 local fluctuation과 noise에 강한 편이다.

### 4.6 ABP waveform reconstruction

논문은 단순 SBP/DBP regression뿐 아니라 ABP waveform reconstruction 결과를 제시한다. UCI-BP에서 예측 waveform과 ground truth ABP waveform을 비교한 그림에 따르면, 두 모델 모두 systolic peak, diastolic trough, dicrotic notch와 같은 주요 morphology를 잘 재현한다. Rhythm disturbance가 큰 구간에서도 primary waveform trend를 안정적으로 추적한다고 설명된다.

이 결과는 이 논문의 중요한 장점이다. SBP와 DBP 숫자만 예측하는 모델은 왜 그런 값을 냈는지 해석하기 어렵다. 반면 ABP waveform을 재구성하면 예측된 systolic peak와 diastolic trough가 실제 waveform 구조와 얼마나 맞는지 확인할 수 있다. 따라서 waveform reconstruction은 clinical interpretability와 physiological consistency를 높인다.

제공된 텍스트에 따르면 Conformer architecture는 systolic pressure variation이 큰 segment에서 더 우수한 성능을 보인다. 이는 long-range temporal dependency modeling 능력 때문이라고 설명된다.

### 4.7 모델 복잡도와 inference efficiency

논문은 NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB VRAM 환경에서 training 및 inference efficiency를 평가한다. 비교 대상으로 PP-Net과 IMCA-PPG를 포함한다.

PP-Net은 LRCN 기반 구조로 temporal modeling에 초점을 맞추지만 복잡한 morphology feature를 포착하는 능력은 제한적이라고 설명된다. IMCA-PPG는 ResNet-50과 multi-head cross-modal attention을 결합하지만 계산 비용이 크다. 반면 제안한 두 모델은 accuracy와 efficiency의 균형을 더 잘 맞춘다고 주장한다.

Swin Transformer는 평균 latency 0.13 ms/sample로 매우 낮은 지연시간을 보인다고 결론에서 언급된다. 이는 wearable device 또는 real-time monitoring에 적합한 특성이다. Conformer-Transformer는 더 계산 비용이 크지만, 더 높은 precision과 waveform reconstruction 성능을 제공한다.

제공된 텍스트에는 Table VII의 세부 parameter 수, FLOPs, latency 수치 전체가 포함되어 있지 않으므로, 정확한 모델 복잡도 비교 수치는 확인할 수 없다.

### 4.8 Learning curve

MIMIC-BP에서 두 모델의 training 및 validation loss curve가 제시된다. 두 모델 모두 초기 epoch에서 loss가 빠르게 감소하고 이후 안정적으로 수렴한다. Validation curve는 training curve보다 약간 높게 유지되며, 이는 과적합이 크지 않고 일반화가 양호하다는 근거로 해석된다.

Conformer-Transformer는 더 빠르게 수렴하고 더 낮은 최종 validation loss에 도달한다. Swin Transformer는 더 점진적으로 수렴한다. 논문은 이를 training efficiency와 robustness 사이의 trade-off로 해석한다.

## 5. 강점, 한계

### 5.1 강점

이 논문의 가장 큰 강점은 혈압 추정을 단순 scalar regression으로 보지 않고, ABP waveform reconstruction과 SBP/DBP estimation을 결합한 multi-task 또는 dual-output 관점으로 접근했다는 점이다. 이는 예측의 임상적 해석 가능성을 높인다. 예측된 SBP와 DBP가 waveform의 peak와 trough 구조에서 유도된다면, 모델의 출력은 더 생리학적으로 설득력을 가진다.

두 번째 강점은 PPG 신호의 두 가지 핵심 정보인 morphology와 rhythm dynamics를 명확히 구분하고, 이를 모델 구조에 반영했다는 점이다. Conformer-Transformer는 convolution과 attention을 결합하여 local morphology와 global rhythm을 동시에 학습한다. Swin Transformer는 window-based hierarchical attention으로 multi-scale morphology와 cross-window interaction을 학습한다. 이러한 구조-task alignment는 설계 논리가 분명하다.

세 번째 강점은 두 공개 데이터셋에서 평가했다는 점이다. UCI-BP와 MIMIC-BP는 모두 ABP 기반 reference를 제공하므로 label 신뢰성이 높다. 특히 MIMIC-BP는 1,524명의 ICU patient와 380시간 이상의 physiological data를 포함하므로, 단일 소규모 데이터셋 평가보다 신뢰도가 높다.

네 번째 강점은 clinical standard를 사용했다는 점이다. MAE, RMSE, correlation뿐 아니라 AAMI와 BHS 기준을 함께 보고함으로써 임상적 수용 가능성을 평가했다. 이는 biomedical signal processing 연구에서 매우 중요하다.

다섯 번째 강점은 interpretability 분석을 포함했다는 점이다. Integrated Gradients와 Grad-CAM을 이용하여 모델 saliency가 PPG landmark와 얼마나 겹치는지 LOS로 평가했다. 또한 DCRC 실험을 통해 모델이 rhythmic continuity에 의존하는지 확인했다. 이는 단순 성능 수치 이상의 분석으로, 모델이 실제 생리학적 구조를 학습했는지 검토하려는 시도다.

여섯 번째 강점은 deployment 관점의 비교를 포함했다는 점이다. Swin Transformer의 낮은 latency와 경량성을 제시하고, Conformer-Transformer의 고정밀 waveform reconstruction 능력과 대비시킨다. 이는 실제 웨어러블 적용 시 모델 선택 기준을 제공한다.

### 5.2 한계

첫 번째 한계는 data split이 recording level로 설명되어 있지만, subject-wise split 여부가 명확하지 않다는 점이다. MIMIC-BP에서 각 subject가 30개의 segment를 제공한다면, 같은 subject의 다른 recording이 training과 test에 동시에 들어갈 가능성이 있는지 확인해야 한다. 생체신호 기반 혈압 추정에서는 subject-specific waveform pattern이 강하기 때문에, subject leakage가 있으면 성능이 과대평가될 수 있다.

두 번째 한계는 test set에서는 alignment를 적용하지 않는다고 하지만, training과 validation에서는 ABP와 PPG를 cross-correlation으로 정렬한다는 점이다. 이는 학습 안정성을 높일 수 있지만, 실제 wearable deployment에서는 ABP reference가 없으므로 이러한 alignment를 사용할 수 없다. 물론 inference 입력은 PPG뿐이지만, 학습 과정에서 정렬된 target을 사용했기 때문에 실제 데이터 분포와 차이가 발생할 수 있다. 이 선택이 실제 배포 상황에서 어떤 영향을 주는지는 추가 분석이 필요하다.

세 번째 한계는 2초 non-overlapping window의 충분성 문제다. 논문은 rhythm dynamics를 강조하지만, 2초 window는 일반적으로 몇 개의 heartbeat만 포함한다. 장기 혈압 trend나 slower rhythm variation을 모델링하기에는 제한적일 수 있다. DCRC 실험이 cross-segment rhythm 의존성을 보여주지만, 모델 입력이 실제로 어느 범위까지 rhythm context를 사용하는지 더 명확한 설명이 필요하다.

네 번째 한계는 UCI-BP와 MIMIC-BP 모두 ICU 또는 curated clinical waveform 기반 데이터라는 점이다. 실제 웨어러블 환경에서는 motion artifact, 피부색, 센서 위치, 압박 정도, 주변광, 체온, 활동 상태 등으로 인해 PPG 품질이 크게 달라진다. 논문은 wearable deployment 가능성을 언급하지만, 실제 스마트워치나 패치 센서에서 수집한 외부 데이터로 검증한 것은 아니다.

다섯 번째 한계는 세부 표 수치가 제공된 텍스트에서 모두 확인되지 않는다는 점이다. Table III–VIII이 언급되지만, 전체 수치가 발췌문에 포함되어 있지 않다. 따라서 각 baseline과의 정밀한 차이, 모델 복잡도, BHS 비율, denoising 전후 모든 지표를 독립적으로 검토하기 어렵다.

여섯 번째 한계는 personalize calibration 또는 demographic information을 사용하지 않는다는 점이다. 혈압과 PPG morphology는 개인의 나이, 혈관 탄성, 질환 상태, 약물, 체온, 말초순환 등에 영향을 받는다. 논문은 향후 subject-specific adaptation, federated learning, physiology-aware fine-tuning을 제안하지만, 현재 연구에서는 개인화 전략이 직접 구현되지는 않았다.

### 5.3 비판적 해석

이 논문은 PPG 기반 cuffless BP estimation 연구에서 상당히 강한 구조적 기여를 제시한다. 특히 Conformer와 Swin Transformer를 physiological signal에 맞게 변형하고, ABP waveform reconstruction을 통해 SBP/DBP estimation의 해석 가능성을 높인 점은 의미가 크다. 단순히 최신 Transformer 계열 모델을 적용한 것이 아니라, PPG의 morphology와 rhythm이라는 신호 특성에 맞추어 구조를 선택했다는 점에서 설계 논리가 분명하다.

다만 보고된 결과를 실제 wearable blood pressure monitoring으로 바로 확장하기에는 아직 검증이 부족하다. 두 데이터셋 모두 ABP reference가 있는 임상 waveform 데이터이며, 실제 소비자 wearable PPG와는 조건이 다르다. 또한 subject-wise generalization 여부가 명확하지 않은 점은 혈압 추정 연구에서 중요한 평가상 이슈다.

그럼에도 이 논문은 기존의 단일 regression 방식에서 벗어나 “혈압 추정은 waveform reconstruction과 physiological representation learning 문제”라는 관점을 강하게 제시한다. 이는 후속 연구에서 더 해석 가능하고 견고한 cuffless BP 모델을 설계하는 데 중요한 방향성을 제공한다.

## 6. 결론

이 논문은 PPG 단일 신호를 이용한 cuffless continuous BP estimation을 위해 Conformer-Transformer와 1D Swin Transformer를 도입한 dual-path deep learning framework를 제안하였다. Conformer-Transformer는 convolution과 self-attention을 결합하여 local morphology와 long-range rhythm을 함께 학습하고, Transformer decoder를 통해 ABP waveform을 재구성한다. 1D Swin Transformer는 window-based 및 shifted-window attention을 통해 multi-scale temporal morphology와 cross-cycle interaction을 효율적으로 학습한다.

실험 결과 Conformer-Transformer는 UCI-BP에서 SBP MAE 2.979 mmHg, DBP MAE 1.603 mmHg를 달성하며 가장 높은 정확도와 ABP waveform reconstruction 성능을 보였다. Swin Transformer는 SBP MAE 3.034 mmHg, DBP MAE 1.714 mmHg를 기록하면서도 낮은 latency를 제공해 real-time wearable deployment에 더 적합한 특성을 보였다. 두 모델 모두 BHS Grade A와 AAMI 기준을 만족한다고 보고된다.

이 연구의 주요 기여는 세 가지로 요약할 수 있다. 첫째, Conformer와 Swin Transformer를 PPG 기반 혈압 추정에 적용하여 morphology와 rhythm dynamics를 동시에 모델링했다. 둘째, ABP waveform reconstruction과 SBP/DBP estimation을 결합하여 예측의 생리학적 일관성과 해석 가능성을 높였다. 셋째, saliency-landmark overlap과 rhythm perturbation 분석을 통해 모델이 의미 있는 physiological feature를 학습한다는 근거를 제시했다.

향후 연구에서는 Conformer와 Swin의 장점을 결합한 hybrid architecture, 실제 wearable 환경에서의 외부 검증, personalized adaptation, multimodal fusion, federated learning, physiology-aware fine-tuning이 중요할 것으로 보인다. 종합적으로 이 논문은 PPG 기반 혈압 추정 분야에서 Transformer 계열 구조를 생리 신호 특성에 맞게 적용한 강력한 연구이며, 단순 회귀에서 waveform-aware physiological modeling으로 나아가는 중요한 방향을 제시한다.
