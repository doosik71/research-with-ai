# EEG-based Brain-Computer Interfaces (BCIs): A Survey of Recent Studies on Signal Sensing Technologies and Computational Intelligence Approaches and Their Applications

이 논문은 EEG 기반 BCI(Brain-Computer Interface) 분야를 최근 5년(2015–2019) 중심으로 정리한 포괄적 survey다. 저자들은 기존 리뷰들이 분류 알고리즘, 딥러닝, 혹은 특정 신호 특징에만 치우쳐 있었다고 보고, 이를 보완하기 위해 **EEG sensing hardware**, **signal enhancement와 online processing**, **machine learning과 interpretable fuzzy model**, **transfer learning**, **deep learning**, 그리고 **BCI-inspired healthcare application**까지 하나의 흐름으로 통합한다. 즉, 이 논문은 “EEG-BCI를 실제로 작동시키기 위해 무엇이 필요한가”를 센서부터 알고리즘, 응용까지 연결해 보여주는 구조를 가진다.  

## 1. Paper Overview

이 논문의 주된 목표는 EEG 기반 BCI를 단순 신호 분류 문제가 아니라, **신호 획득 → 잡음 제거 → 특징 추출 → 학습/적응 → 실제 응용**으로 이어지는 end-to-end 시스템 관점에서 정리하는 것이다. 저자들은 BCI를 인간의 뇌와 외부 장치 사이의 직접적 communication pathway로 정의하고, 특히 EEG가 비침습성, 높은 시간 해상도, 낮은 위험성, 휴대성 덕분에 현실적인 BCI 구현의 핵심 modality가 되었다고 설명한다. 동시에 EEG는 낮은 spatial resolution과 낮은 SNR 때문에 artefact handling과 robust learning이 필수라는 점도 강조한다.  

왜 이 문제가 중요한지도 논문은 분명히 밝힌다. EEG-BCI는 단순한 실험실 기술이 아니라, 장애인 의사소통 보조, 로봇/휠체어 제어, 피로·졸음 감지, 감정 인식, 수면 상태 분석, seizure·Parkinson’s disease·Alzheimer’s disease·Schizophrenia 같은 임상 응용까지 이어질 수 있다. 즉, 인간 상태를 실시간으로 추적하고 외부 시스템과 상호작용시키는 플랫폼으로서 EEG-BCI는 translational healthcare와 intelligent HCI 양쪽에서 의미가 크다.

저자들이 주장하는 이 논문의 차별점도 명확하다. 기존 survey가 주로 classification algorithm, deep learning, 혹은 publication trend에 집중했다면, 이 논문은 최근 EEG sensing technology, signal enhancement, interpretable fuzzy model, transfer learning, deep learning, healthcare application을 함께 다룬다. 논문이 스스로 정리한 기여는 다섯 가지다: sensing technology advances, signal enhancement/online processing, machine learning과 fuzzy models, deep learning과 combined approaches, healthcare system/application evolution.

## 2. Core Idea

이 논문의 핵심 아이디어는 EEG-based BCI의 성능 향상을 단일 classifier의 발전으로 설명할 수 없다는 데 있다. 저자들은 BCI 성능이 다음 다섯 축의 동시 발전 위에서 나온다고 본다.

첫째, **더 나은 sensing technology**.
둘째, **더 강력한 artefact removal 및 online processing**.
셋째, **subject/session/task variability를 다루는 transfer learning**.
넷째, **non-stationary하고 nonlinear한 EEG를 설명 가능하게 다루는 fuzzy/interpretable model**.
다섯째, **representation learning과 적응 능력을 높이는 deep learning**.  

즉, 이 논문은 EEG-BCI를 “딥러닝이 다 해결한다”는 식으로 보지 않는다. 오히려 EEG는 본질적으로 noisy하고 사람마다 다르며 세션마다 distribution shift가 발생하기 때문에, hardware usability, signal cleaning, transfer/adaptation, interpretability가 함께 해결되어야 한다고 본다. 이런 관점이 논문 전체를 관통한다.  

또 다른 중심 아이디어는 **실용성**이다. dry sensor, wearable headset, minimal calibration, online processing, headset-to-headset transfer 같은 키워드가 반복되는데, 이는 BCI를 연구실 환경에서 everyday environment로 확장하려는 문제의식으로 읽을 수 있다. 특히 augmented BCI(ABCI), wearable dry EEG, real-time artefact rejection 같은 요소들은 “사용자 친화적이고 장기 사용 가능한 BCI”를 향한 흐름을 반영한다.

## 3. Detailed Method Explanation

이 논문은 survey이므로 단일 새 알고리즘을 제안하지 않는다. 대신 EEG-based BCI를 구성하는 기술 블록들을 체계적으로 설명한다.

### 3.1 BCI와 EEG의 기본 구조

논문은 BCI를 크게 active/reactive BCI와 passive BCI로 구분한다. active/reactive BCI는 사용자의 의도적 혹은 자극 반응 기반 brain activity를 외부 장치 제어에 사용하고, passive BCI는 자발적 제어가 아니라 인지 상태·주의·피로 같은 implicit information을 읽어 HCI를 풍부하게 만든다. 응용 측면에서는 휠체어/로봇 제어, 게임 인터페이스, emotion recognition, fatigue detection, sleep assessment, 임상 질환 detection/prediction까지 포괄한다.

또한 brain imaging technique를 invasive, partially invasive, non-invasive로 나누고, EEG가 real-world BCI에서 가장 널리 쓰이는 이유를 설명한다. EEG는 직접적인 cortical electrical activity를 측정하고 millisecond 수준의 temporal resolution을 가지며, 휴대성과 안전성이 높다. 반면 electrode 수 제한으로 spatial resolution이 낮고, 환경 잡음·피로·눈 움직임·근육 움직임에 의해 오염되기 쉽다.

ERP(Event-Related Potential), P300, RSVP, SSVEP, PVT 같은 대표 패러다임도 이 기본 프레임 안에서 설명된다. 이 부분은 이후 응용과 알고리즘 파트를 이해하는 토대를 제공한다.

### 3.2 Advances in sensing technologies

센서 파트에서 저자들은 wet sensor와 dry sensor를 비교한다. wet electrode는 conductive gel 덕분에 signal quality가 좋지만, 착용이 번거롭고 불편하며 일상적 사용에 부적합하다. 반면 dry sensor는 사용성이 크게 좋아졌고, skilled user가 짧은 시간 안에 스스로 착용할 수 있을 정도로 practical하다. 논문은 최근 dry sensor가 wet system에 가까운 품질을 제공하면서도 preparation과 gel이 필요 없다는 점을 강조한다.

특히 dry/noncontact sensor와 wearable device의 발전을 중요한 추세로 본다. 예를 들어 SSVEP task에서 dry EEG sensor가 상용 sensor와 comparable한 성능을 보였고, noncontact electrode로도 높은 information transfer rate가 가능하다는 사례를 소개한다. 이는 future mobile BCI와 augmented BCI의 기반으로 해석된다.

또한 상용 EEG device를 표로 비교해, channel 수, sensor type, sampling rate, transmission 방식, wearable 여부를 정리한다. 여기서 저자들이 전달하려는 메시지는 분명하다. BCI 연구는 알고리즘만의 문제가 아니라, 어떤 장치를 쓰느냐에 따라 signal resolution과 portability가 크게 달라진다. 예를 들어 low-resolution wearable headset은 빠른 실사용에 적합하지만 coverage가 제한되고, high-density system은 더 풍부한 신호를 주지만 무겁고 덜 이동적이다.

### 3.3 Signal enhancement and online processing

EEG는 artefact에 취약하므로, 논문은 signal enhancement를 독립적인 핵심 주제로 다룬다. Blind Source Separation(BSS) 계열 방법으로 PCA, CCA, ICA를 설명하며, 이 중 ICA가 eye blink, movement, muscle artefact 제거에서 가장 대표적 접근이라고 본다. PCA는 artefact와 EEG가 상관되어 있으면 잘 분리되지 않을 수 있고, CCA는 muscle artefact 제거에 쓰이며, ICA는 독립 성분을 분해한 뒤 artefact 성분을 제거하고 clean EEG를 재구성하는 방식이다.

눈 깜빡임 및 눈 움직임 artefact 제거에서는 ASR(Artefact Subspace Reconstruction)과 ICA 조합이 자주 언급된다. ASR은 large-amplitude/transient artefact를 자동으로 줄이는 전처리 단계로 유용하지만, 단일 채널 EEG에는 적용이 어렵고, 정기적으로 발생하는 artefact 제거는 cut-off 설정에 민감하다. 그래서 ICA 기반 component classification을 결합하는 보완 전략이 제시된다.

근육 artefact에 대해서는 CCA, EMD, BSS, EEMD-CCA, IVA 같은 조합형 방법들이 소개된다. 중요한 메시지는, muscle contamination은 tough problem이며 단일 기법보다 복합 신호처리가 더 실용적이라는 점이다. 또한 EEGLAB과 그 extension들이 실무적 toolbox로 소개되는데, AAR, MARA, clean_rawdata, ADJUST 같은 툴은 artefact rejection과 pre-processing을 자동화하는 데 쓰인다.

온라인 처리 측면에서는 wearable high-density dry EEG와 source localization, VAR model, logistic regression, ASR, ORICA, REST 같은 실시간 분석 프레임워크가 소개된다. 논문은 real-time cortico-cortical interaction monitoring이 BCI의 중요한 미래 방향이라고 본다. 결국 “좋은 분류기”보다 “실시간으로 깨끗한 EEG를 지속적으로 공급하는 시스템”이 더 본질적이라는 메시지가 읽힌다.

### 3.4 Machine learning and transfer learning

머신러닝 파트는 EEG-BCI의 전통적 패턴 인식 흐름을 정리한다. supervised/unsupervised learning의 기본 틀 위에서, LDA, regularized LDA, SVM 같은 linear classifier, MLP 같은 neural network, Bayesian classifier/HMM, kNN, classifier ensemble이 BCI에서 많이 쓰였다고 정리한다. 또 이들 모델에 앞서 frequency band power, channel connectivity 등 EEG-specific feature extraction이 필요하다고 설명한다.

논문에서 특히 중요한 부분은 transfer learning이다. EEG는 subject 간, session 간, time 간 distribution shift가 심하고, 이로 인해 기존 supervised learning의 “train/test가 같은 분포”라는 가정이 자주 깨진다. transfer learning은 바로 이 문제를 해결하기 위한 방법론으로 제시된다. 저자들은 source domain에서 배운 지식을 target task/subject/session/headset으로 옮겨 calibration effort를 줄이고 classification 성능을 유지·개선하려는 접근을 설명한다.

논문은 transfer를 네 부류로 상세히 정리한다.

* task-to-task transfer
* subject-to-subject transfer
* session-to-session transfer
* headset-to-headset transfer

subject-to-subject transfer 예에서는 large-scale model pool과 small baseline calibration을 조합해 **90% calibration time reduction**을 달성한 drowsiness detection 사례가 소개된다. 이는 EEG-BCI 실용화에서 가장 큰 병목 중 하나인 사용자별 재보정을 크게 줄일 수 있음을 보여준다.

headset-to-headset transfer도 흥미로운데, 이상적으로는 사용자가 EEG headset을 바꿔도 재보정이 없어야 한다는 문제를 다룬다. AwAR(active weighted adaptation regularization)는 이전 headset의 labelled data와 active learning을 결합해 새로운 headset에서 필요한 calibration data를 줄인다. 이 문제 설정 자체가 BCI의 plug-and-play화를 지향한다는 점에서 중요하다.

### 3.5 Interpretable fuzzy models

논문은 black-box machine learning의 한계를 지적하며 interpretable fuzzy model을 별도 축으로 다룬다. EEG는 비정상성(non-stationary)과 비선형성(nonlinear)이 강하기 때문에, 명확한 경계 대신 membership degree와 flexible boundary를 제공하는 fuzzy set/fuzzy inference system이 적합할 수 있다고 본다. 이는 설명 가능성과 robustness를 동시에 겨냥하는 흐름이다.

구체적으로, multiclass EEG CSP를 regression으로 확장하거나, fuzzy membership function으로 noisy EEG entropy sensitivity를 낮추거나, domain adaptation과 결합한 OwARR, fuzzy rule-based brain-state-drift detector, fuzzy integral 기반 motor imagery BCI와 multimodel fusion 등이 소개된다. 이들의 공통점은 EEG 패턴의 불확실성과 drift를 설명 가능한 방식으로 다루려 한다는 점이다.  

이 부분은 survey의 독특한 장점이다. 많은 EEG-BCI 리뷰가 단순 정확도 향상만 다루는 반면, 이 논문은 fuzzy model을 통해 “왜 그렇게 판단하는가”와 “distribution drift를 어떻게 관찰할 수 있는가”까지 관심을 확장한다.

### 3.6 Deep learning and deep transfer learning

딥러닝 파트에서 논문은 CNN, GAN, RNN/LSTM, broad DNN, deep transfer learning을 정리한다. 저자들은 deep learning을 feature와 classifier를 함께 학습하는 family로 정의하며, EEG처럼 rapidly changing brain signal에는 static feature 기반 전통 ML보다 유리한 면이 있다고 설명한다. 특히 spontaneous EEG application에서 CNN, augmentation이나 generation에 GAN, temporal dependency 처리에 RNN/LSTM이 자주 쓰인다고 정리한다.

CNN은 convolution, pooling, fully connected layer를 통해 spatial-temporal feature를 학습하고, RNN/LSTM은 temporal sequence를 모델링한다. 논문은 sleep staging, motor imagery, emotion recognition, seizure detection, mental workload, ERP detection 등에서 deep model이 쓰였다고 요약한다. 다만 한편으로는 어떤 task에서는 expert-defined feature + RNN이 더 좋은 성능을 낸 사례도 언급해, end-to-end deep learning이 항상 전통적 feature engineering보다 우월한 것은 아니라는 점도 시사한다.

deep transfer learning도 별도 주제로 다뤄진다. attention-based transfer learning with RNN, cross-domain encoder-decoder 등이 EEG classification과 brain functional area detection을 개선하는 사례로 소개된다. 이는 transfer learning과 deep learning이 실제로 별개의 흐름이 아니라 점차 결합되고 있음을 보여준다.

마지막으로 논문은 deep learning 모델의 adversarial vulnerability도 논의한다. EEG-BCI가 휠체어나 exoskeleton 제어에 쓰일 경우 adversarial attack은 사용자 안전 문제로 이어질 수 있기 때문에, robustness와 defense 전략이 긴급한 연구 과제라고 본다. 이는 상당히 선견지명 있는 지적이다.

## 4. Experiments and Findings

이 논문은 survey이므로 하나의 통일된 benchmark 실험을 직접 수행하는 논문은 아니다. 대신 각 기술 축에서 대표 연구 결과를 정리하며, 분야 전반의 실험적 함의를 뽑아낸다. 따라서 여기서의 “finding”은 논문 저자 자신의 단일 모델 성능이 아니라, 문헌 전반의 패턴 요약으로 이해해야 한다.

가장 먼저 sensing 측면의 중요한 finding은 dry sensor와 wearable EEG의 실용성이 빠르게 개선되고 있다는 점이다. 논문은 dry sensor가 wet system과 comparable한 signal quality를 보이면서도 훨씬 높은 usability를 제공한다고 정리하고, SSVEP task에서 상용 센서와 유사한 정확도를 보인 사례도 소개한다. 이는 BCI가 everyday environment로 이동할 가능성을 뒷받침한다.

transfer learning 영역에서는 calibration burden 감소가 매우 두드러진다. 예를 들어 subject-to-subject transfer in drowsiness detection에서 **90% calibration time reduction**이 보고되며, headset-to-headset transfer 역시 새로운 장치에서 필요한 labelled sample 수를 크게 줄인다. 이 결과들은 EEG-BCI 실용화의 병목인 per-user/per-device recalibration 문제를 완화할 수 있음을 보여준다.  

fuzzy model 영역의 실험적 메시지는 interpretability와 robustness의 공존 가능성이다. fuzzy membership function으로 noisy EEG에서 entropy sensitivity를 낮추거나, fuzzy integral 기반 multimodel fusion이 motor imagery BCI와 robotic arm control에서 robust performance를 보인다는 점이 강조된다. 이는 EEG의 noise와 uncertainty를 다루는 데 fuzzy framework가 단순 보조기법이 아니라 핵심 역할을 할 수 있음을 시사한다.

deep learning에 관해서는 survey가 균형 잡힌 태도를 보인다. 일부 최근 연구들은 CNN, GAN, DBN이 classification accuracy에서 강한 성능을 낸다고 보고하지만, 저자들은 기존 survey들 사이에서도 deep learning의 우월성에 대한 해석 차이가 있었음을 지적한다. 또한 sleep staging에서는 expert-defined feature + RNN이 CNN, RCNN 등보다 더 좋은 성능을 보인 사례도 언급한다. 이는 EEG-BCI에서 deep learning이 강력하지만, task formulation과 feature design을 완전히 대체한다고 보기는 어렵다는 뜻이다.  

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **범위의 균형**이다. 보통 EEG-BCI survey는 hardware, signal processing, machine learning, deep learning 중 하나에 치우치기 쉬운데, 이 논문은 이 모든 축을 하나의 시스템적 프레임으로 연결한다. 그래서 독자는 “EEG-BCI에서 성능이 왜 잘 안 나오는가”를 센서·노이즈·도메인 시프트·설명 가능성·딥러닝까지 함께 보면서 이해할 수 있다.

두 번째 강점은 **실용성 중심의 문제 설정**이다. dry sensor, wearable device, online processing, transfer learning, headset-to-headset adaptation 같은 주제들은 모두 실제 사용성을 높이는 방향이다. 즉, 이 논문은 실험실용 BCI보다 real-world BCI의 조건을 훨씬 더 강하게 의식하고 있다.  

세 번째 강점은 **interpretable fuzzy model을 포함했다는 점**이다. 딥러닝 위주의 survey가 많던 시기에, EEG처럼 noisy하고 drift가 심한 데이터에서 fuzzy model의 설명 가능성과 적응성을 강조한 것은 꽤 의미 있다. 특히 brain-state drift detector 같은 예시는 BCI의 신뢰성과 online adaptation을 동시에 생각하게 만든다.

한편 한계도 분명하다. 첫째, survey 특성상 각 방법의 성능을 동일 benchmark 위에서 엄밀 비교하지는 않는다. 그래서 어떤 알고리즘이 “항상 최고”라고 결론 내리기는 어렵다. 둘째, 범위가 넓은 만큼 개별 기법의 수학적 깊이는 제한적이다. 예를 들어 deep transfer learning이나 adversarial defense는 방향성은 잘 짚지만, 세부 알고리즘 비교까지 깊게 들어가지는 않는다. 셋째, 시점상 2015–2019 중심 survey이기 때문에 이후 transformer 기반 EEG foundation model이나 self-supervised learning 흐름은 반영되어 있지 않다. 다만 이는 논문 출간 시점을 고려하면 자연스러운 한계다.

비판적으로 해석하면, 이 논문은 EEG-BCI의 “정답”을 주기보다, **어디가 병목인지 구조화해 주는 지도**에 가깝다. 딥러닝만으로 해결되지 않는 부분, 즉 sensing usability, artefact rejection, domain adaptation, interpretability가 왜 중요한지를 분명히 보여준다는 점이 가장 큰 가치다.

## 6. Conclusion

이 논문은 EEG 기반 BCI를 센서, 신호정제, 머신러닝, transfer learning, fuzzy interpretability, deep learning, healthcare application까지 포괄적으로 정리한 survey다. 핵심 메시지는 명확하다. EEG-BCI의 진전은 단지 classifier 성능 향상이 아니라, **더 착용하기 쉬운 센서**, **더 강한 artefact handling**, **더 적은 calibration**, **더 설명 가능한 모델**, **더 강력한 representation learning**이 함께 발전할 때 가능하다는 것이다.  

실무와 연구 관점에서 이 논문이 주는 가장 중요한 교훈은, EEG-BCI를 real-world healthcare/HCI 시스템으로 만들려면 subject/session/headset variability와 robustness 문제를 반드시 해결해야 한다는 점이다. 논문 말미에서 hybrid BCI, AR와 EEG-BCI 결합, adversarial defense 같은 방향을 제시하는 것도 같은 맥락이다. 결국 이 논문은 “EEG-BCI가 어디까지 왔는가”를 정리하는 동시에, “실제 사용 가능한 BCI를 만들기 위해 무엇이 더 필요한가”를 묻는 survey라고 볼 수 있다.
