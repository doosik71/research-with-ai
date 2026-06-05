# DeepCNAP: A Deep Learning Approach for Continuous Noninvasive Arterial Blood Pressure Monitoring Using Photoplethysmography

* **저자**: Dong-Kyu Kim, Young-Tak Kim, Hakseung Kim, Dong-Joo Kim
* **발표연도**: 2022

## 1. 논문 개요

이 논문은 photoplethysmography, 즉 PPG 신호만을 이용하여 invasive arterial blood pressure, 즉 ABP waveform을 실시간으로 추정하는 deep learning 모델인 DeepCNAP을 제안한다. 기존 cuff 기반 혈압 측정은 비침습적이고 널리 사용되지만, 연속 측정이 어렵고 환자에게 불편하며 cuff 측정의 정확성 자체도 비판을 받아 왔다. 반면 radial artery 또는 femoral artery에 catheter를 삽입하는 invasive ABP monitoring은 연속 혈압 파형을 얻을 수 있지만 침습적이므로 일반 병동이나 일상생활, 웨어러블 환경에 적용하기 어렵다.

이 논문의 연구 문제는 단일 PPG waveform으로부터 연속적인 ABP waveform 전체를 정확하게 복원할 수 있는가이다. 많은 기존 연구는 SBP와 DBP라는 두 개의 discrete value만 추정하였다. 그러나 ABP waveform에는 단순한 SBP와 DBP를 넘어 vascular stiffness, stroke volume, pulse morphology, hemodynamic instability와 관련된 더 풍부한 생리 정보가 포함된다. 따라서 연속 ABP waveform을 추정할 수 있다면, 혈압 수치 추정뿐 아니라 hypertension, prehypertension, hypotension, normal state 같은 hemodynamically unstable events를 감지하는 데도 활용될 수 있다.

논문은 DeepCNAP을 residual U-Net, attention-based skip connections, self-attention을 결합한 구조로 설계하였다. ResUNet은 PPG signal을 압축하고 복원하는 encoder-decoder 구조로 ABP waveform을 생성하며, attention-based skip connection은 encoder와 decoder 사이에서 temporal information을 보존한다. 마지막 self-attention layer는 waveform 내 장기 의존성과 서로 다른 signal morphology 간 관계를 학습하여 ABP waveform estimation 성능을 높인다.

연구의 중요성은 clinical monitoring과 wearable healthcare 양쪽에 있다. 병원 general ward에서는 4–8시간 간격으로 vital sign을 간헐적으로 측정하는 경우가 많아 postoperative hypotension 같은 위험 사건을 놓칠 수 있다. 일상생활에서도 cuff-based ambulatory BP monitoring은 불편하고 간헐적이다. DeepCNAP처럼 PPG만으로 continuous ABP waveform을 추정하는 방법은 일반 병동 모니터링, 웨어러블 기기, cardiovascular disease의 조기 감지에 활용될 가능성이 있다.

## 2. 핵심 아이디어

DeepCNAP의 핵심 아이디어는 PPG-to-ABP waveform translation 문제를 signal compression and restoration 문제로 보고, U-Net 계열 구조에 self-attention을 결합하여 temporal dependency를 더 잘 반영하는 것이다. PPG와 ABP는 모두 시간에 따라 변하는 pulse waveform이지만, 두 신호의 morphology는 완전히 같지 않다. PPG는 말초 microvascular bed의 volumetric blood change를 반영하고, ABP는 arterial pressure waveform을 직접 반영한다. 따라서 PPG에서 ABP를 추정하려면 waveform의 local pattern뿐 아니라 pulse cycle 내부와 cycle 간의 시간적 관계를 학습해야 한다.

기존 U-Net은 encoder에서 signal을 점차 압축하고 decoder에서 다시 복원하는 구조이며, skip connection을 통해 encoder의 세부 feature를 decoder로 전달한다. 이 구조는 waveform generation에 유용하지만, 일반 skip connection은 temporal dependency를 명시적으로 모델링하지 못한다. DeepCNAP은 이 한계를 줄이기 위해 attention-based skip connections를 사용한다. 이는 encoder와 decoder 사이의 feature 연결에서 sample 간 상대적 위치와 중요도를 학습하도록 하는 방식이다.

또 다른 핵심은 self-attention이다. Self-attention은 Transformer의 핵심 요소로, sequence 내 각 위치가 다른 위치들과 어떤 관계를 갖는지 학습한다. 논문은 ResUNet이 추출한 time-series representation을 self-attention layer에 전달하여, waveform의 long-range temporal dependency와 morphology 간 관계를 학습하도록 한다. 이 점에서 DeepCNAP은 단순 CNN 기반 waveform 복원 모델보다 더 넓은 시간적 문맥을 고려할 수 있다.

기존 연구와의 차별점도 분명하다. 첫째, DeepCNAP은 predetermined handcrafted feature 없이 raw PPG signal만을 입력으로 사용한다. 둘째, SBP와 DBP만 예측하지 않고 1024 sample 길이의 연속 ABP waveform을 생성한다. 셋째, 생성된 ABP waveform에서 SBP, DBP, MAP을 계산하고, 이를 바탕으로 hemodynamic instability classification까지 수행한다. 넷째, MIMIC-II 기반 942명, 총 374.43시간의 비교적 큰 데이터로 10-fold cross-validation을 수행하여 robustness를 평가하였다.

## 3. 상세 방법 설명

DeepCNAP의 전체 파이프라인은 데이터셋 구성, preprocessing, ABP waveform target 생성, ResUNet 기반 PPG feature compression and recovery, attention-based skip connection, self-attention refinement, 그리고 ABP waveform 및 SBP/DBP 추정으로 구성된다.

### 3.1 데이터셋

논문은 University of California, Irvine, 즉 UCI Machine Learning Repository에 공개된 cuffless BP dataset을 사용하였다. 이 데이터셋은 Kachuee et al.이 PhysioNet의 MIMIC-II database에서 처리하고 검증한 자료이다. 원자료는 ICU patient로부터 동시에 기록된 fingertip PPG와 ABP signal로 구성되며, sampling rate는 두 신호 모두 125 Hz이다.

초기 데이터셋은 12,000 recordings로 구성되어 있었다. 연구진은 signal reliability를 확보하기 위해 8분 미만의 recording을 제거하였고, 이 과정에서 2,064 recordings가 남았다. 이후 PPG와 ABP signal을 1024 samples 길이의 segment로 나누었다. sampling rate가 125 Hz이므로 한 segment는 8.192초에 해당한다. 논문은 이 길이가 cardiac activity의 time-domain information을 포착하기에 충분하다고 설명한다.

기존 연구들 중 일부는 매우 높거나 낮은 혈압 값을 filtering하여 사용하였다. 예를 들어 SBP가 180 이상 또는 80 이하, DBP가 130 이상 또는 60 이하인 값을 제거하는 방식이 사용되었다. 그러나 이 논문은 real-world environment와 ICU에서는 고혈압이나 저혈압 같은 hemodynamic instability가 실제로 중요하다고 보았다. 따라서 너무 좁은 혈압 범위를 제거하는 대신, $SBP \le 200$ 및 $DBP \ge 50$ 조건을 사용하여 더 넓은 BP range를 포함하였다. 이 preprocessing 이후 최종 데이터는 374.43시간, 134,795 segments가 되었다.

### 3.2 Preprocessing과 target variable

입력은 PPG signal이고 target은 ABP signal이다. PPG는 MinMaxScaler 방식으로 normalization되었다. 논문에서 제시한 정규화 식은 다음과 같다.

$$
PPG_{norm}=\frac{PPG-\min(PPG)}{\max(PPG)-\min(PPG)}
$$

이 식은 각 PPG segment의 값을 최소값 0, 최대값 1 범위로 변환하는 방식이다. 이는 deep learning 모델 학습에서 입력 scale을 안정화하기 위한 일반적인 전처리이다.

ABP signal에서는 SBP, DBP, MAP이 계산된다. SBP는 ABP segment의 최대값, DBP는 ABP segment의 최소값으로 정의된다.

$$
SBP=\max(ABP)
$$

$$
DBP=\min(ABP)
$$

MAP은 SBP와 DBP를 이용하여 다음과 같이 계산된다.

$$
MAP=\frac{SBP+2DBP}{3}
$$

이 식은 diastolic phase가 systolic phase보다 긴 일반적인 cardiac cycle 특성을 반영하여 DBP에 더 큰 weight를 부여하는 근사식이다. 또한 heart rate, 즉 HR은 noisy PPG 환경에서 HR analysis를 위해 설계된 HeartPy algorithm으로 계산되었다.

### 3.3 DeepCNAP architecture

DeepCNAP은 세 가지 주요 module로 구성된다. 첫 번째는 ResUNet, 두 번째는 attention-based skip connections, 세 번째는 self-attention module이다.

ResUNet은 residual neural network와 U-Net을 결합한 구조이다. U-Net은 encoder와 decoder로 구성된다. Encoder는 입력 PPG signal을 점진적으로 압축하여 중요한 feature representation을 추출하고, decoder는 이 representation을 다시 upsampling하여 ABP waveform을 생성한다. 이 논문에서는 imaging data용 2D convolution이 아니라 time-domain signal에 적합한 1D convolution으로 구조를 수정하였다.

Residual unit은 deep network의 vanishing gradient와 degradation 문제를 줄이기 위해 사용된다. 논문에서 residual unit은 다음과 같이 표현된다.

$$
y_i=h(x_i)+F(x_i,W_i)
$$

$$
x_{i+1}=f(y_i)
$$

여기서 $x_i$와 $y_i$는 $i$번째 residual unit의 input과 output이고, $F(x_i,W_i)$는 residual function이며, $h(x_i)$는 identity mapping function이다. $f(y_i)$는 activation function이다. 이 구조는 network가 전체 mapping을 직접 학습하기보다 residual component를 학습하게 하므로 깊은 network의 학습을 안정화한다.

Attention-based skip connection은 U-Net의 encoder와 decoder 사이에서 정보를 전달할 때 temporal information을 더 잘 반영하기 위한 장치이다. 일반 skip connection은 encoder의 feature를 decoder로 그대로 전달하지만, attention-based skip connection은 scaled dot-product attention을 사용하여 sequence 내 각 sample의 상대적 중요도를 학습한다. 논문에서 attention mechanism은 다음과 같이 표현된다.

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{n}}\right)V
$$

여기서 $Q$, $K$, $V$는 각각 queries, keys, values이고, $K^T$는 $K$의 transpose이며, $\sqrt{n}$은 input vector dimensionality에 따른 scaling factor이다. 이 구조는 PPG feature와 ABP reconstruction 사이에서 중요한 temporal position을 더 잘 연결하도록 돕는다.

마지막 self-attention module은 network의 마지막 layer에 적용된다. 목적은 서로 다른 signal morphology 사이의 dependency와 long-range temporal dependency를 포착하는 것이다. Self-attention module은 두 개의 동일한 layer stack으로 구성되며, 각 stack은 positional encoding layer, self-attention layer, position-wise feedforward layer를 포함한다.

Positional encoding은 self-attention이 recurrence를 갖지 않기 때문에 필요한 장치이다. Self-attention 자체는 입력 sequence의 순서를 알지 못하므로, 각 time step의 위치 정보를 embedding에 더해준다. 논문은 다음의 sinusoidal positional encoding을 사용한다.

$$
PE(pos,2i)=\sin(pos/10000^{2i/n})
$$

$$
PE(pos,2i+1)=\cos(pos/10000^{2i/n})
$$

여기서 $pos$는 position index이고, $i$는 dimension index이다. 논문에서 $pos$와 $n$은 각각 1024와 32로 설정되었다고 설명한다. 즉 1024개 sample로 구성된 waveform의 각 위치에 대해 32차원 position embedding을 부여한다.

Self-attention layer는 식 (7)의 scaled dot-product attention을 기반으로 한다. Multi-head attention이 아니라 single-head self-attention을 사용했으며, $Q$, $K$, $V$는 linear projection에서 학습 가능한 weight로부터 얻어진다. 이는 각 time step이 waveform 내 다른 time step들과 어떤 관계를 가지는지 학습하게 한다.

Position-wise feedforward layer는 두 개의 linear transformation과 Gaussian error linear unit, 즉 GELU activation으로 구성된다. 논문에서 식은 다음과 같다.

$$
FFN(X)=GELU(xW_1+b_1)W_2+b_2
$$

여기서 $x$는 이전 layer의 output이고, $W_1$, $W_2$는 학습 가능한 parameter이며, $b_1$, $b_2$는 bias이다. GELU는 ReLU보다 더 smooth한 activation으로 설명된다. 각 sublayer에는 layer normalization과 residual connection이 적용되어 training을 안정화하고 generalization을 개선한다.

### 3.4 학습 설정

DeepCNAP은 regression 문제로 학습된다. 일반적으로 regression loss로 MSE와 MAE가 사용되지만, 이 연구에서는 MAE를 loss function으로 선택하였다. 그 이유는 artifacts나 noise가 있는 상황에서 MAE가 MSE보다 robust하며, error term을 더 균형 있게 반영하기 때문이다. ABP waveform에는 ICU 환경의 noise, artifact, missing data가 포함될 수 있으므로 MAE loss가 적합하다고 판단한 것이다.

학습은 최대 300 epochs 동안 수행되었으며, early stopping patience는 10 epochs로 설정되었다. Batch size는 256이다. Optimizer는 Adam이며, learning rate는 0.001에서 0.0001로 exponentially decreasing하도록 설정되었다. 데이터는 80% training, 10% validation, 10% test로 randomly split되었고, 최종적으로 10-fold cross-validation을 통해 model generalization을 평가하였다. 구현 환경은 TensorFlow 2.10, Python 3.7, NVIDIA GeForce RTX 3090 24 GB VRAM이다.

### 3.5 평가 지표와 hemodynamic instability classification

DeepCNAP의 성능은 두 관점에서 평가된다. 첫째, ABP waveform과 그로부터 계산된 SBP, DBP의 estimation performance이다. 이를 위해 MAE, RMSE, coefficient of determination $R^2$, adjusted $R^2$가 사용된다. 또한 BP monitoring device로서의 임상적 기준을 확인하기 위해 BHS와 AAMI 기준을 사용하였다. AAMI 기준은 mean error가 5 mmHg 이하이고 error의 standard deviation이 8 mmHg 이하일 것을 요구한다. BHS 기준은 5, 10, 15 mmHg 이하 error의 누적 비율을 기준으로 grade를 부여한다.

둘째, 추정된 BP로 hemodynamically unstable events를 분류하는 성능이다. 논문은 ESH/ESC guideline을 기반으로 hypotension, normal state, prehypertension, hypertension 네 class를 정의하였다. 각 test segment는 estimated SBP와 estimated DBP range에 따라 class로 분류된다. Classification metric은 accuracy, F-score, sensitivity, specificity이다. 논문에서 식은 다음과 같이 제시된다.

$$
Accuracy=\frac{TP+FP}{TP+TN+FP+FN}
$$

$$
F\text{-}score=\frac{2TP}{2TP+FP+FN}
$$

$$
Sensitivity=\frac{TP}{TP+FN}
$$

$$
Specificity=\frac{TN}{TN+FP}
$$

다만 일반적으로 accuracy는 $\frac{TP+TN}{TP+TN+FP+FN}$로 정의된다. 제공된 텍스트에는 $TP+FP$가 분자에 들어간 형태로 표기되어 있으므로, 이 부분은 논문 추출 텍스트의 식 표기 오류일 가능성이 있다. 보고서에서는 원문에 제시된 식을 그대로 반영하되, 일반 정의와 다를 수 있음을 명시하는 것이 타당하다.

## 4. 실험 및 결과

논문은 네 가지 주요 결과를 보고한다. 첫째, attention layer 추가 여부에 따른 성능 비교이다. 둘째, 제안 모델의 SBP, DBP, ABP waveform estimation error 평가이다. 셋째, 추정된 BP를 이용한 hemodynamic instability classification이다. 넷째, 실제 ground-truth waveform과 estimated waveform의 qualitative comparison이다.

### 4.1 Attention layer 추가 효과

논문은 ResUNet, attention-based ResUNet, ResUNet with self-attention, attention-based ResUNet with self-attention을 비교하였다. 10-fold cross-validation 결과, attention layer가 추가될수록 error metric의 standard deviation이 낮아지고 generalization performance가 향상되었다. 특히 self-attention을 ResUNet에 추가한 모델이 단순 attention 추가보다 좋은 성능을 보였으며, 최종 모델인 ResUNet + Attention + Self-attention이 baseline model보다 우수한 성능을 보였다.

이 결과는 ABP waveform estimation에서 단순 encoder-decoder 구조만으로는 충분하지 않고, waveform 내부의 temporal dependency와 contextual information을 학습하는 mechanism이 중요하다는 점을 보여준다. 특히 SBP 성능에서 notable improvement가 있었다고 보고한다. SBP는 DBP보다 변동성이 크고 추정이 어려운 값이므로, attention mechanism이 pulse waveform의 중요한 time step을 더 잘 반영했을 가능성이 있다.

### 4.2 SBP, DBP, ABP waveform estimation 성능

논문 abstract에 따르면 DeepCNAP의 mean absolute error는 SBP 3.40 ± 4.36 mmHg, DBP 1.75 ± 2.25 mmHg, BP waveform 3.23 ± 2.21 mmHg이다. 이는 SBP와 DBP뿐 아니라 전체 ABP waveform 추정에서도 낮은 MAE를 보였다는 의미이다.

Bland–Altman plot 분석에서는 estimated BP와 ground-truth BP 사이의 agreement를 확인하였다. SBP와 DBP의 mean difference는 각각 1.23 mmHg와 -0.53 mmHg였고, standard deviation은 각각 5.40 mmHg와 2.81 mmHg였다. 95% confidence interval 기준 limits of agreement는 SBP에서 [-4.17, 6.63], DBP에서 [-2.28, 3.33]으로 제시되었다. 대부분의 estimated point가 이 limits of agreement 안에 위치한다고 설명한다.

Regression plot에서는 estimated BP와 actual BP 사이의 correlation coefficient가 SBP 0.97, DBP 0.95로 나타났다. SBP의 correlation이 DBP보다 높게 나타났지만, 논문은 SBP target value의 variance가 DBP보다 약 두 배 크기 때문에 correlation이 높게 나타날 수 있다고 해석한다. 반대로 estimation error는 SBP가 DBP보다 더 컸다. 이는 SBP의 변동성이 더 크고 waveform peak에 더 민감하기 때문으로 볼 수 있다.

Appendix의 AAMI 및 BHS 비교에서는 DeepCNAP이 SBP, DBP, MAP 추정에서 BHS Grade A를 달성했고, AAMI 기준도 만족했다고 보고한다. AAMI 기준에서 ME는 SBP, DBP, MAP 모두 기준보다 훨씬 낮았고, SD도 8 mmHg 이하 criterion margin을 만족하였다. 또한 AAMI 기준은 충분한 subject 수를 요구하는데, 본 연구는 MIMIC-II 기반 942명 subject를 사용하여 통계적 신뢰성을 높였다고 주장한다.

### 4.3 Hemodynamic instability classification 성능

DeepCNAP은 단순히 BP waveform을 복원하는 데 그치지 않고, estimated SBP와 DBP를 사용하여 hemodynamically unstable events를 분류하였다. 분류 대상은 hypotension, normal state, prehypertension, hypertension 네 class이다. 총 13,312 test segments가 SBP range와 DBP range에 따라 각각 분류되었다.

논문 abstract에서는 estimated BP를 이용한 classification accuracy가 hypertension 99.44%, prehypertension 97.58%, hypotension 92.23%, normal state 94.64%라고 보고한다. Discussion에서는 class 기준이 SBP 또는 DBP인지에 따라 성능이 다르게 나타난다고 설명한다. SBP 기반 classification은 hypertension classification에서 더 좋은 성능을 보였고, DBP 기반 classification은 hypotension을 높은 accuracy로 식별하였다.

평균 accuracy는 SBP 기준 91.44%, DBP 기준 94.66%로 보고되었다. 기존 연구와 비교하면, Kachuee et al.은 hypertension classification에서 SBP와 DBP 기준 각각 82%, 98% accuracy를 보였고, Athaya et al.은 waveform estimation 후 SBP와 DBP 기준 각각 74%, 73% accuracy를 보였다고 논문은 설명한다. DeepCNAP은 특히 BP waveform을 추정하면서도 hemodynamic event classification에서 높은 성능을 보였다는 점에서 clinical relevance가 있다.

논문은 hypertension based on DBP dataset이 상대적으로 imbalance되어 있다고 설명한다. DBP 기반 hypertension proportion은 1.68%로 매우 낮고, SBP 기반 hypertension proportion은 35.03%이다. 이 때문에 DBP 기반 classification은 accuracy가 높더라도 F-score와 sensitivity에서는 SBP보다 낮을 수 있다고 해석한다. 이는 imbalanced classification에서 accuracy만으로 성능을 판단하면 안 된다는 점을 보여준다.

### 4.4 Waveform qualitative result

논문은 PPG input, target ABP signal, estimated ABP signal을 시각적으로 비교하였다. 좋은 사례에서는 estimated ABP waveform이 target ABP waveform의 overall pattern과 peak region을 잘 따라갔다. PPG와 ABP가 single large systolic peak와 이후 lower diastolic peak를 공유하는 경우, DeepCNAP은 waveform shape와 timing을 잘 복원하였다.

흥미로운 점은 target ABP signal에 artifact가 포함된 경우이다. 예를 들어 transducer flushing, catheter clotting, movement artifact가 ABP target에 포함되어 있을 때, DeepCNAP은 PPG만으로 artifact가 없는 더 smooth한 ABP signal을 생성하는 모습을 보였다. 논문은 이것을 robustness로 해석하지만, 동시에 이런 경우 estimated signal이 target artifact를 따라가지 않기 때문에 estimation error가 증가할 수 있다고 설명한다. 즉 모델이 생리적으로 더 타당한 waveform을 생성하더라도, ground-truth ABP에 artifact가 있으면 수치상 error가 커질 수 있다.

Hypertension과 hypotension 상태에서도 estimated waveform은 반복적인 cycle을 유지하고 shape, magnitude, phase를 어느 정도 보존하였다. 또한 4시간 길이의 long-term waveform 비교에서도 estimated ABP의 range와 trend가 actual ABP와 유사하다고 보고한다. 이는 DeepCNAP이 segment-level prediction뿐 아니라 더 긴 시간의 BP trend를 어느 정도 추적할 수 있음을 시사한다.

## 5. 강점, 한계

DeepCNAP의 가장 큰 강점은 SBP와 DBP 두 값만 추정하는 것이 아니라 continuous ABP waveform 전체를 추정한다는 점이다. ABP waveform은 단순 수축기와 이완기 혈압보다 더 풍부한 정보를 포함한다. 예를 들어 waveform morphology는 vascular stiffness, stroke volume, pulse wave reflection, hemodynamic instability와 관련될 수 있다. 따라서 waveform-level estimation은 임상적 활용 가능성이 더 크다.

두 번째 강점은 raw PPG signal만을 사용한다는 점이다. ECG와 PPG를 함께 사용하는 PTT 또는 PAT 기반 방법은 sensor synchronization과 multiple sensor attachment가 필요하다. DeepCNAP은 PPG만을 사용하므로 웨어러블 기기나 일반 병동 모니터링에서 구현 부담이 상대적으로 낮다. 또한 handcrafted feature를 미리 정의하지 않고 deep network가 feature extraction을 수행하므로, 복잡한 feature engineering 의존도를 줄인다.

세 번째 강점은 ResUNet, attention-based skip connection, self-attention을 결합한 architecture 설계이다. ResUNet은 waveform compression and restoration에 적합하고, residual module은 깊은 network 학습을 안정화한다. Attention-based skip connection은 encoder-decoder 사이에서 temporal context를 더 잘 보존하며, self-attention은 long-range dependency를 학습한다. Ablation 성격의 비교에서도 attention과 self-attention을 추가할수록 성능이 개선되었다.

네 번째 강점은 estimation 결과를 clinical event classification에 연결했다는 점이다. 논문은 estimated BP waveform에서 SBP와 DBP를 계산하고, 이를 이용해 hypertension, prehypertension, hypotension, normal state를 분류하였다. 이는 단순 regression performance를 넘어서, 모델이 실제 임상적 의사결정에 필요한 hemodynamic instability detection에 활용될 수 있음을 보여준다.

다섯 번째 강점은 기존 연구보다 넓은 BP range를 포함하려 했다는 점이다. 일부 기존 연구는 매우 높거나 낮은 BP를 제거했지만, 이 논문은 $SBP \le 200$, $DBP \ge 50$ 조건을 사용해 hypotension과 hypertension을 포함할 수 있도록 데이터셋을 구성하였다. 이는 hemodynamic instability detection이라는 목표와 더 잘 맞는다.

그러나 한계도 존재한다. 첫째, external validation이 수행되지 않았다. 모델은 UCI repository의 MIMIC-II 기반 dataset에서 학습 및 평가되었지만, 다른 기관, 다른 장비, 다른 PPG sensor, 다른 patient population에서 같은 성능이 유지되는지는 확인되지 않았다. 의료 AI 모델은 데이터 source와 장비 조건에 민감할 수 있으므로, external validation은 실제 적용 전에 필수적이다.

둘째, arrhythmic events에 대한 detection accuracy는 평가되지 않았다. PPG와 ABP waveform은 arrhythmia에 의해 크게 변할 수 있고, clinical setting에서는 arrhythmia가 혈압 waveform estimation을 어렵게 할 수 있다. 논문은 dataset 특성상 arrhythmic event detection을 평가할 수 없었다고 명시한다.

셋째, 이전 방법들과 동일한 analytical environment에서 직접 비교하지 못했다. 논문은 previous continuous ABP waveform estimation 연구들과 비교하지만, dataset, preprocessing, subject split, segment selection, metric이 다르기 때문에 공정한 비교에는 한계가 있다. 저자도 이상적으로는 hemodynamic instability classification 성능까지 포함하여 같은 환경에서 비교해야 한다고 인정한다.

넷째, wearable device 적용을 위해서는 motion artifact robustness 검증이 추가로 필요하다. MIMIC-II ICU dataset은 noise와 artifact가 존재하지만, 일상생활 웨어러블 환경의 움직임, sensor displacement, 피부 접촉 변화, 운동 중 PPG distortion과는 성격이 다를 수 있다. 논문도 wearable device 구현을 위해 movement-related artifacts에 대한 robustness를 엄격히 테스트해야 한다고 제안한다.

다섯째, 데이터 split 방식에 대한 subject independence 여부가 제공된 텍스트만으로 명확하지 않다. 논문은 10-fold cross-validation을 수행했다고 설명하지만, segment-level random split인지 subject-level split인지 추출 텍스트만으로는 확실히 판단하기 어렵다. 만약 같은 subject의 segment가 train과 test에 동시에 포함된다면, 새로운 subject에 대한 일반화 성능이 과대평가될 수 있다. 제공된 텍스트에서는 “number of subjects = 942”를 강조하지만, fold 구성의 subject-exclusive 여부는 명확히 확인되지 않는다.

여섯째, target ABP signal 자체에 artifact가 포함될 수 있다. 논문은 DeepCNAP이 artifact가 있는 ABP target을 smooth하게 생성할 수 있다고 설명하지만, 이는 평가 metric 측면에서는 error 증가로 이어질 수 있다. 반대로 artifact가 ground truth에 포함된 상태에서 모델이 artifact를 재현하지 않는 것이 실제로 더 바람직한지, 또는 ABP artifact filtering이 어떻게 이루어져야 하는지는 추가 논의가 필요하다.

마지막으로, DeepCNAP은 높은 성능을 보였지만 model complexity와 real-time deployment 비용에 대한 상세한 on-device benchmark는 제공되지 않는다. 논문은 real-time estimation 가능성을 주장하지만, 실제 wearable hardware에서 latency, memory footprint, power consumption을 평가한 결과는 제공된 텍스트에 없다.

## 6. 결론

이 논문은 PPG만을 이용하여 continuous noninvasive arterial blood pressure waveform을 추정하는 DeepCNAP 모델을 제안하였다. DeepCNAP은 ResUNet 기반 encoder-decoder 구조에 attention-based skip connections와 self-attention을 결합하여, PPG waveform에서 ABP waveform으로의 translation을 수행한다. 이 구조는 waveform의 local feature, encoder-decoder 간 contextual information, long-range temporal dependency를 함께 반영하도록 설계되었다.

정량적으로 DeepCNAP은 SBP MAE 3.40 ± 4.36 mmHg, DBP MAE 1.75 ± 2.25 mmHg, BP waveform MAE 3.23 ± 2.21 mmHg를 달성하였다. Bland–Altman 분석에서도 SBP와 DBP의 mean difference와 SD가 AAMI 기준을 만족하였고, BHS 기준에서도 SBP, DBP, MAP에 대해 Grade A를 달성했다고 보고되었다. 또한 estimated BP를 이용해 hypertension, prehypertension, hypotension, normal state를 높은 accuracy로 분류하였다.

이 연구의 주요 기여는 세 가지로 정리할 수 있다. 첫째, predetermined feature 없이 raw PPG signal만으로 ABP waveform 전체를 생성하였다. 둘째, self-attention을 U-Net 계열 waveform estimation model에 결합하여 temporal dependency modeling을 강화하였다. 셋째, 추정된 ABP waveform을 hemodynamic instability classification에 활용하여 임상적 의미를 평가하였다.

실제 적용 가능성 측면에서 DeepCNAP은 general ward monitoring과 daily wearable healthcare에 유용할 수 있다. 일반 병동에서는 intermittent vital sign check로 인해 hypotension 같은 위험 사건을 놓칠 수 있고, 일상생활에서는 cuff-based ambulatory monitoring이 불편하다. PPG 기반 continuous BP waveform estimation은 이런 공백을 줄이고, cardiovascular deterioration을 조기에 감지하는 데 도움을 줄 수 있다.

다만 외부 기관 데이터 검증, arrhythmia 상황 평가, subject-independent split의 명확한 검증, motion artifact 환경에서의 robustness test, wearable device에서의 real-time deployment 평가가 필요하다. 특히 실제 웨어러블 환경에서는 PPG 품질이 ICU 데이터보다 훨씬 불안정할 수 있으므로, 움직임 관련 artifact에 대한 성능 검증이 핵심 후속 과제이다.

종합하면, DeepCNAP은 PPG 기반 cuffless BP monitoring 연구에서 discrete SBP/DBP estimation을 넘어 continuous ABP waveform estimation과 clinical event classification을 결합한 의미 있는 연구이다. Self-attention을 활용한 ResUNet 구조는 복잡한 생체 waveform translation 문제에 적합한 설계이며, 향후 일반 병동과 웨어러블 기반 continuous BP monitoring 시스템으로 확장될 가능성이 있다.
