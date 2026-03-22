# Hardware Implementation of Deep Network Accelerators Towards Healthcare and Biomedical Applications

이 논문은 새로운 단일 모델을 제안하는 연구라기보다, **헬스케어·바이오메디컬 영역에서 Deep Learning(DL)을 실제 하드웨어 위에 어떻게 효율적으로 구현할 것인가**를 다루는 **tutorial + review + perspective paper**다. 저자들은 edge 환경의 의료 IoT와 Point-of-Care(PoC) 장치를 염두에 두고, **CMOS 기반 DL accelerator, FPGA, memristive crossbar, 그리고 Spiking Neural Network(SNN)용 neuromorphic processor**를 한 프레임 안에서 비교·정리한다. 특히 1쪽 초록과 2쪽 도입부는 이 논문의 핵심 목적이 “의료용 DL을 edge로 끌어내리기 위한 하드웨어 선택지와 한계”를 설명하는 데 있음을 분명히 한다.  

## 1. Paper Overview

논문의 출발점은 의료 AI의 성능이 높아질수록, 이를 실제 의료 현장에 배치하는 문제가 점점 더 중요해진다는 점이다. 의료 데이터는 영상, 생체신호, 움직임, 바이오샘플처럼 다중 모달이며, 클라우드만으로 처리하면 **지연, 전력, 비용, 개인정보 보호** 문제가 생긴다. 2쪽 Figure 1은 이러한 문제를 시각적으로 보여주는데, 환자 데이터가 cloud, edge node, edge device 중 어디에서 처리될 수 있는지를 그리면서, 특히 **healthcare IoT와 PoC에서는 edge inference/learning이 더 바람직하다**고 설명한다.  

이 문제가 중요한 이유는 의료 분야에서 단순한 정확도만이 아니라 **항상 켜져 있는(always-on) 모니터링**, 저전력 동작, 저비용, 오프라인 처리, 개인정보 보호가 모두 동시에 요구되기 때문이다. 저자들은 현재의 GPU나 일부 embedded AI accelerator가 유망하긴 하지만, 여전히 많은 최신 모델은 resource-constrained 의료 기기에 배치하기엔 너무 무겁고 전력 소모가 크다고 지적한다. 따라서 이 논문은 “어떤 신경망이 좋으냐”보다, **어떤 하드웨어 기술이 어떤 의료 DL workload에 적합한가**를 더 본질적인 질문으로 삼는다.

또한 이 논문은 스스로의 기여를 비교적 분명하게 선언한다. 2쪽에서 저자들은 다음 네 가지를 요약한다. 첫째, **CMOS, memristor, FPGA라는 세 가지 기술을 한 번에 biomedical DL acceleration 관점에서 논의한 첫 논문**이라는 점. 둘째, FPGA 구현과 memristive crossbar 시뮬레이션에 대한 튜토리얼적 설명을 제공한다는 점. 셋째, SNN 기반 neuromorphic processor가 DL accelerator를 어떻게 보완할 수 있는지 논의한다는 점. 넷째, 실험 재현을 위한 open-source code와 data를 제공한다는 점이다.

## 2. Core Idea

이 논문의 핵심 아이디어는 **헬스케어 edge AI를 위한 하드웨어 선택지를 “계산 효율 대 적용 가능성”의 관점에서 재구성**하는 데 있다. 즉, 단순히 GPU를 대체할 새로운 칩을 소개하는 것이 아니라, 다음과 같은 질문을 던진다.

* DL의 병목은 어디에서 오는가
* 이를 줄이기 위해 하드웨어가 무엇을 최적화해야 하는가
* 의료·바이오 신호는 비동기적이고 시계열적인데, 그 특성에 더 잘 맞는 하드웨어는 무엇인가
* DNN accelerator와 SNN neuromorphic processor는 경쟁 관계인가, 보완 관계인가

저자들의 답은 비교적 명확하다. **대규모 일반 DL inference는 현재로서는 CMOS 기반 accelerator가 주류**지만, ultra-low-power always-on monitoring에는 **spiking neuromorphic processor가 보완적 역할**을 할 수 있고, 장기적으로는 **memristive in-memory computing**이 매우 유망하며, **FPGA는 빠른 프로토타이핑과 맞춤형 저전력 구현**에서 중요한 위치를 가진다는 것이다. 이 관점은 5쪽 Figure 3의 “hardware technologies pyramid”와 12~15쪽의 analysis section에서 반복해서 드러난다.  

또 하나의 중심 아이디어는 **데이터 표현 방식과 하드웨어 방식의 정합성**이다. 6쪽 Figure 4는 ANN/DNN과 SNN의 입력 처리 방식을 비교하면서, DNN은 배치 기반 동기식 처리에, SNN은 spike/event 기반 비동기 처리에 더 잘 맞는다고 설명한다. 저자들은 나중에 benchmark 해석에서도, EMG와 DVS처럼 둘 다 event-like signal인 경우에는 spike 기반 융합이 더 자연스럽고 효율적일 수 있다고 주장한다. 즉, 이 논문은 하드웨어를 단순 계산 엔진이 아니라 **데이터의 시간적 구조와 맞물리는 표현 장치**로 본다.  

## 3. Detailed Method Explanation

### 3.1 DL 계산량과 하드웨어 병목

논문은 먼저 왜 DL이 하드웨어적으로 어려운지를 설명한다. 3~4쪽에서 MLP, CNN, RNN, LSTM의 기본 구조를 요약하고, 특히 **backpropagation과 VMM(Vector-Matrix Multiplication), MAC(Multiply-Accumulate)**가 지배적인 연산이라는 점을 강조한다. 4쪽 Table I은 AlexNet, ResNet-18/50, VGG-19, OpenPose, MobileNet의 weight 수와 MAC 수를 제시하는데, 예를 들어 VGG-19는 1장 이미지당 22B MAC, 25 FPS 기준 550B MAC이 필요하다고 정리한다. 이 표의 목적은 단순 스펙 나열이 아니라, **DL accelerator가 왜 메모리 접근 감소와 MAC 병렬화를 동시에 노려야 하는지**를 보여주는 데 있다.

### 3.2 CMOS 기반 DL accelerator

CMOS 섹션의 핵심은 현재 실용적인 edge AI accelerator 대부분이 사실상 이 범주에 속하며, 의료 응용에서도 당분간 주류일 가능성이 높다는 점이다. 7쪽 Table II는 Cambricon-X, Eyeriss, Origami, ConvNet processor, Envision, Neural processor, LNPU, DNPU, Thinker, UNPU, Google Edge TPU, Intel Nervana NNP-I 1000, Huawei Ascend 310 등을 정리하고, 각각이 어떤 biomedical task에 연결될 수 있는지를 예시로 보여준다. 예를 들어 ECG, EEG/ECoG, 피부암 분류, ultrasound, respiratory sound classification, PPG prediction 같은 task가 연결된다.

논문은 이들 칩이 공통적으로 **reduced precision arithmetic**, **architectural enhancement**, **data movement reduction**, **parallel MAC**, 경우에 따라 **near-memory / in-memory style optimization**을 사용한다고 설명한다. 8쪽에서는 tensor decomposition, pruning, mixed precision도 함께 언급하며, 특히 systolic array가 CMOS accelerator의 대표적 병렬화 기법임을 강조한다. 결국 이 섹션의 메시지는, **CMOS accelerator는 가장 현실적인 선택지이지만, 전력·일반화·규제·의료 workflow 문제까지 자동으로 해결해주지는 않는다**는 것이다.  

### 3.3 FPGA 기반 DNN 구현

FPGA 섹션은 이 논문의 튜토리얼 성격이 가장 잘 드러나는 부분이다. 8~9쪽에서 저자들은 FPGA가 **저비용, 재구성 가능성, 병렬성, 짧은 time-to-market**을 제공한다고 설명한다. 그리고 8쪽 Figure 5에서 실제 구현 흐름을 제시한다. 요약하면,

1. PyTorch 모델을 준비하고
2. Caffe의 `.prototxt`, `.caffemodel`로 변환한 뒤
3. MATLAB Fixed-point toolbox로 weight/bias를 fixed-point로 양자화하고
4. PipeCNN과 OpenCL/OpenVINO FPGA toolchain으로 RTL library, host executable, FPGA bitstream을 생성한다.

이 파이프라인의 의미는 FPGA가 단순 “하드웨어 대안”이 아니라, **DNN을 실제 edge용 정수 연산 inference engine으로 컴파일하는 실용적 플랫폼**이라는 데 있다. 또한 9쪽에서는 의료용 FPGA 사례로 ECG anomaly detection, mass-spectrometry 기반 암 탐지, ECoG BCI, EEG neurofeedback용 LSTM inference engine 등을 제시한다. 저자들은 적절한 co-design을 하면 **FPGA가 GPU 대비 10배 이상 energy-delay 효율을 낼 수 있다**고 정리하지만, 동시에 설계 난이도와 숙련 인력 의존성이 큰 한계도 분명히 인정한다.  

### 3.4 Memristive DNN과 in-memory computing

memristive DNN 부분은 이 논문의 가장 야심찬 기술적 논의다. 9~10쪽에서 저자들은 memristor를 “제4의 회로 소자”로 소개하며, conductance를 weight처럼 저장해 **crossbar 내부에서 곧바로 analog MAC/VMM**을 수행할 수 있다고 설명한다. 9쪽 Figure 6은 바로 이 개념을 보여준다. 입력 전압 벡터를 row에 걸고, cross-point conductance matrix를 통해 column current를 얻으면, 오옴의 법칙으로 곧바로 VMM 결과가 나온다. 이론적으로 이는 **O(1) 시간 복잡도 수준의 병렬 MAC**에 가깝고, 기존 GPU 대비 약 2500배 전력 감소와 25배 가속 가능성이 있다고 인용한다.

하지만 이 논문은 memristor를 낙관적으로만 다루지 않는다. 10쪽 이후에서 저자들은 실제 MDNN(Memristive DNN)을 구현하려면 단순 crossbar만으로 끝나지 않고, **PWM 입력 변환, current integrator 또는 sense amplifier, ADC, activation circuit, write/update circuit, BL/WL switch matrix** 같은 주변회로가 필수라고 설명한다. 즉 이론적인 in-memory computing의 장점이 실제 칩으로 갈 때 **ADC/주변회로 오버헤드**에 의해 상당 부분 상쇄될 수 있다는 점을 명시한다.

또한 10~11쪽은 memristor의 가장 큰 약점인 **device non-ideality**를 자세히 짚는다. 비선형·비대칭·확률적 conductance update, temporal/spatial variation, yield, 낮은 on/off ratio 같은 문제가 있고, 이들은 대형 네트워크일수록 성능 저하를 유발한다. 11쪽 Figure 7은 MemTorch를 이용한 DNN→MDNN 변환 과정을 보여주고, 11쪽 Figure 8은 $R_{ON}$/$R_{OFF}$ variability의 표준편차 $\sigma$가 증가할수록 memristive MLP/CNN의 정확도가 지속적으로 떨어지는 것을 보여준다. 즉, 이 논문은 memristive computing의 잠재력을 높게 평가하면서도, **현재로서는 presilicon/network-specific 수준의 낙관적 결과가 많고 실제 biomedical 대규모 적용은 아직 매우 제한적**이라고 결론짓는다.  

### 3.5 SNN 및 neuromorphic processor의 역할

SNN 부분은 이 논문의 “보완재” 관점을 이해하는 데 중요하다. 4~6쪽과 13~14쪽에서 저자들은 SNN이 **비동기 spike/event 처리**에 적합하고, DVS처럼 event-based sensor와 잘 맞으며, 항상 새로운 정보가 들어올 때만 연산하므로 **always-on biomedical monitoring**에 매우 유리하다고 본다. 14쪽 Table IV는 DYNAP-SE, SpiNNaker, TrueNorth, Loihi, ODIN 같은 neuromorphic 플랫폼을 정리한다. 이들은 EMG, ECG, EEG, LFP, HFO 같은 생체신호 처리에 이미 사용된 사례가 있다.  

저자들이 특히 강조하는 것은 **on-chip adaptation / online learning**과 patient-specific model tuning의 잠재력이다. 그러나 동시에, locality 제약 때문에 다층 backpropagation의 온칩 구현은 아직 어렵고, 이상적인 가중치 저장장치로는 non-volatile analog-like memory가 필요하다고 설명한다. 이 지점에서 memristor와 neuromorphic computing이 만나는 셈이다.

### 3.6 Patient-specific model tuning

17~18쪽의 patient-specific tuning 섹션은 의료 응용 관점에서 중요한 부가 논의다. 저자들은 환자 간 variability가 크기 때문에, 하나의 범용 모델을 그대로 배치하기보다 **transfer learning을 이용해 patient-specific tuning**하는 것이 중요하다고 본다. 이때 tuning은 **온라인(on-chip)**으로도 가능하지만 더 많은 메모리와 버퍼가 필요하고, **오프라인(off-chip)**으로도 가능하지만 환자 데이터의 원격 저장과 개인정보 이슈가 생긴다. 따라서 의료 하드웨어는 단순 추론 칩이 아니라, 장기적으로는 **개인화 가능한 adaptive inference/learning system**이어야 한다는 시사점을 준다.

## 4. Experiments and Findings

이 논문은 review/tutorial 논문이지만, 단순 문헌 요약에서 멈추지 않고 하나의 benchmark task를 통해 여러 하드웨어를 직접 비교한다. 5쪽에 따르면 benchmark는 **hand-gesture recognition**이며, 입력은 두 센서 모달리티로 구성된다.

* Myo armband에서 얻은 **EMG**
* event-based camera의 **DVS** 또는 conventional camera의 **APS frame**

총 5개 제스처, 21명(남 12, 여 9), 3회 세션으로 수집되었고, 평가는 **3-fold cross validation**으로 이루어진다. 비교 지표는 정확도뿐 아니라 **inference energy, inference time, Energy-Delay Product(EDP)**이다. 이 설정 자체가 논문의 관심이 accuracy 하나가 아니라 **실제 edge deployment 효율**에 있음을 보여준다.

가장 중요한 결과는 16쪽 **Table V**에 있다. 여기서 Loihi, ODIN+MorphIC, embedded GPU(Jetson Nano), FPGA, memristive implementation이 비교된다. 대표적으로 다음과 같은 결과가 보고된다.

EMG+DVS 융합 기준으로, **Loihi의 spiking CNN**은 `96.0 ± 0.4%` 정확도를 기록한다. 이는 정확도 면에서 상당히 강력하며, energy는 `1104.5 ± 58.8 µJ`, inference time은 `7.75 ± 0.07 ms`, EDP는 `8.6 ± 0.5 µJ·s`다. 같은 조건의 **ODIN+MorphIC 기반 spiking MLP**는 정확도 `89.4 ± 3.0%`로 더 낮지만, energy `37.4 ± 4.2 µJ`, EDP `0.42 ± 0.08 µJ·s`로 매우 효율적이다. 이는 neuromorphic 하드웨어 내부에서도 **정확도-효율 trade-off**가 존재함을 보여준다.

비-spiking 계열에서는 embedded GPU의 EMG+APS(CNN 융합)가 `95.4 ± 1.7%`, FPGA 구현이 `94.8 ± 2.0%`를 기록한다. 정확도는 Loihi와 비슷하거나 약간 낮은 수준이지만, 에너지는 각각 `32.1×10^3 µJ`, `31.2×10^3 µJ` 수준으로 훨씬 크다. 즉, 정확도만 보면 기존 DNN 하드웨어도 나쁘지 않지만, **에너지·EDP 측면에서는 neuromorphic 또는 memristive 쪽이 압도적으로 유리**하다는 것이 테이블의 핵심 메시지다.

memristive 결과는 가장 극단적이다. 예를 들어 APS(CNN) 기준 정확도는 `96.2 ± 3.3%`, energy는 `4.83 µJ`, inference time은 `1.0 × 10^{-3} ms`, EDP는 `4.83 × 10^{-6} µJ·s`로 보고된다. 하지만 저자들은 바로 뒤에서 이 수치가 **network-specific presilicon assumption** 위에서 계산된 것임을 분명히 밝힌다. 즉, 이 값은 매우 유망하지만 “범용 memristive chip이 실제로 이미 이 정도 성능을 냈다”는 뜻은 아니다. 오히려 16~17쪽의 해석은, 이런 성능이 가능하려면 ADC, array duplication, tile design 등 많은 구조적 가정이 필요하며, 큰 vision model로 가면 ADC 전력과 면적 문제가 심각해질 수 있다고 경고한다.

논문이 실험에서 도출하는 더 흥미로운 해석은 다음과 같다. **spike-based hardware는 GPU/FPGA 대비 약 두 자릿수 수준의 EDP 개선**을 보이며, 이는 always-on monitoring에 매우 중요하다. 또한 EMG-only에서는 spiking 방식이 정확도에서 약 10% 정도 불리하지만, **EMG+DVS처럼 동일하게 spike/event 구조를 갖는 입력을 융합하면 약 4% 정도 정확도 향상**이 생긴다. 반대로 non-spiking 환경에서 EMG와 APS처럼 표현 방식이 다른 데이터를 합치면 개선이 작거나 경우에 따라 파괴적일 수 있다고 본다. 저자들은 이를 바탕으로, **여러 센서가 만들어내는 데이터를 일관된 표현(spike 등)으로 맞추는 것이 neural processing에 유리할 수 있다**는 가설을 제안한다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **하드웨어 기술을 의료 AI deployment 문제와 직접 연결했다는 점**이다. 많은 의료 DL 논문은 모델 정확도 중심이고, 많은 accelerator 논문은 일반 벤치마크 중심인데, 이 논문은 그 사이를 연결한다. 7쪽 Table II, 13쪽 Table III, 14쪽 Table IV, 16쪽 Table V를 통해, 실제로 어떤 biomedical task가 어떤 하드웨어에서 돌아갈 수 있는지를 한눈에 보이게 만든다. 리뷰 논문으로서 구조가 매우 좋다.  

둘째, 이 논문은 **기술 낙관론과 현실적 한계를 동시에 제시**한다. 예를 들어 memristor에 대해선 엄청난 전력·속도 잠재력을 인정하면서도, 주변회로 오버헤드, variability, endurance, large-scale biomedical deployment의 부재를 솔직하게 지적한다. FPGA도 저전력·짧은 개발기간 장점을 인정하지만, 복잡한 CNN엔 메모리/대역폭 한계가 있다는 점을 숨기지 않는다. 이런 균형감은 신뢰할 만한 survey/perspective의 중요한 조건이다.  

셋째, **SNN을 단순 대안이 아니라 보완재로 배치한 관점**도 강점이다. 저자들은 DNN과 SNN을 경쟁시키기보다, SNN이 anomaly trigger나 always-on watchdog 역할을 하여 더 무거운 DNN processing을 필요할 때만 깨우는 구조를 상상한다. 이는 의료 모니터링처럼 sparse event 중심 업무에 꽤 설득력이 있다. 2쪽의 conjecture와 17쪽의 EDP 해석이 이 점을 잘 뒷받침한다.  

한계도 분명하다. 첫째, 이 논문은 많은 비교를 제시하지만, **여전히 review/tutorial paper**다. 따라서 각 플랫폼 비교가 완전히 동일 조건의 공정한 benchmark라고 보긴 어렵다. 특히 memristive 수치는 presilicon, network-specific assumptions에 크게 의존한다. 둘째, 의료 응용 예시가 많지만, 상당수는 “이 accelerator가 이 CNN/LSTM을 돌릴 수 있으니 이런 의료 task에도 쓸 수 있다”는 형태의 연결이다. 즉 실제 임상 현장에서 모두 검증된 end-to-end 시스템을 의미하지는 않는다. 셋째, 2021년 시점 논문이라 오늘날의 transformer 기반 edge AI, chiplet, analog in-memory training, foundation model for biosignals 같은 더 최근 흐름은 다루지 못한다.  

비판적으로 해석하면, 이 논문의 진짜 공헌은 “최고의 accelerator를 골라줬다”는 데 있지 않다. 더 본질적으로는, **의료 DL deployment를 accuracy 경쟁이 아니라 하드웨어/데이터 표현/전력/개인화/프라이버시까지 포함한 시스템 문제로 재정의했다**는 데 있다. 이 시각은 이후 의료 edge AI 연구에서도 여전히 유효하다.

## 6. Conclusion

이 논문은 헬스케어와 바이오메디컬 응용을 위한 DL accelerator 하드웨어를 폭넓게 정리한 중요한 tutorial/review다. 핵심 메시지는 다음과 같이 요약할 수 있다. **현재 실용적인 의료 DL inference의 주류는 CMOS accelerator이고, FPGA는 맞춤형 저전력 구현과 빠른 프로토타이핑에 강점이 있으며, memristive crossbar는 장기적으로 매우 유망하지만 아직 큰 기술적 장벽이 남아 있고, SNN neuromorphic processor는 always-on biosignal monitoring에서 DNN을 보완할 수 있다.**  

또한 16쪽 Table V와 17쪽 해석은, 하드웨어의 특화 정도가 높아질수록 inference energy와 EDP는 급격히 개선될 수 있지만, 그만큼 **범용성, 훈련 용이성, 설계 단순성은 줄어드는 경향**이 있음을 보여준다. 즉 의료 AI 하드웨어 설계는 “무조건 fastest”를 찾는 문제가 아니라, **정확도·전력·지연·범용성·개인화 가능성의 균형**을 찾는 문제다.

마지막으로 저자들은 결론에서, 하드웨어 발전만으로 의료 AI가 성공적으로 정착하는 것은 아니라고 강조한다. 의료 제공자, 하드웨어·소프트웨어 엔지니어, 데이터 과학자, 정책결정자, 인지신경과학자, 재료·소자 연구자들이 함께 움직여야 한다는 점을 밝힌다. 이 논문은 바로 그 협업을 위한 공통 지도를 제공한다는 점에서 가치가 크다.  
