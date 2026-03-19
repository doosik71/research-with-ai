# Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis

이 논문은 EHR(Electronic Health Record) 분석에 적용된 딥러닝 연구를 **정보추출, 표현학습, 결과 예측, computational phenotyping, 비식별화**라는 다섯 축으로 정리한 survey다. 핵심 문제의식은 분명하다. EHR 채택이 급증하면서 방대한 환자 데이터가 축적되었지만, 이 데이터는 원래 임상 연구용이 아니라 **기록·청구·행정 처리용**으로 설계되어 있어 이질적이고 불규칙하며 구조가 복잡하다. 저자들은 이런 조건에서 딥러닝이 수작업 feature engineering 부담을 줄이고, 장기 의존성과 잠재 구조를 더 잘 포착할 수 있는 유력한 방법이라고 본다. 특히 이 논문은 broad health informatics survey와 달리, **EHR 자체에 특화된 딥러닝 연구만 집중적으로 정리**한다는 점을 분명히 한다.

## 1. Paper Overview

논문은 지난 10여 년간 EHR 보급이 급증했고, 그 결과 인구통계, 진단, 검사, 처방, 영상, 임상노트 등 방대한 환자 정보가 디지털화되었다고 설명한다. 하지만 EHR는 본래 연구보다 **운영 효율과 청구 목적**에 더 가깝게 설계되었기 때문에, 데이터가 서로 다른 코드 체계와 문서 형식으로 흩어져 있고, 기관마다 표현 방식도 다르다. 저자들은 이러한 복잡성이 EHR 분석의 핵심 난제라고 본다. 동시에 최근 딥러닝이 이미지와 자연어처리에서 보여준 성공이 EHR 분석으로 확장되며, 전통적 로지스틱 회귀, SVM, 랜덤포레스트보다 더 좋은 성능과 더 적은 수작업 전처리를 제공하기 시작했다고 정리한다. 1쪽과 2쪽의 서론은 이런 배경을 중심으로 구성되어 있다.

이 논문이 중요한 이유는 단순히 “딥러닝을 의료에 썼다”를 요약하는 데 있지 않다. 저자들은 EHR 기반 딥러닝 연구가 다루는 입력 형식 자체가 매우 다층적이라고 강조한다. 3쪽에서는 EHR 데이터를 다섯 가지로 정리한다. **수치형 값, 날짜/시간 객체, 범주형 코드, 자유 서술 텍스트, 그리고 이들이 시간 순서로 이어진 파생 시계열**이다. 즉 EHR는 이미지처럼 단일한 raw input이 아니고, 이질적 데이터의 복합체다. 따라서 이 survey의 진짜 목적은 “어떤 모델이 최고냐”보다는 **이 복잡한 데이터 구조를 딥러닝이 어떻게 다루고 있는가**를 정리하는 데 있다.

## 2. Core Idea

논문의 핵심 아이디어는 EHR 딥러닝 연구를 **응용 과제별로 체계화**하는 데 있다. 7쪽 Table III에서 저자들은 전체 연구를 다음 다섯 범주로 묶는다.

* Information Extraction
* Representation Learning
* Outcome Prediction
* Phenotyping
* De-identification

이 구조는 이 논문의 중심 좌표계다. 즉, 딥러닝을 단순히 아키텍처별로 나열하지 않고, **임상적으로 무엇을 하려는가**를 기준으로 정리한다. 예를 들어 임상노트에서 구조를 뽑아내는 정보추출, 의료코드와 환자를 벡터화하는 표현학습, 질병 발생과 재입원 등을 예측하는 outcome prediction, 새로운 질병 아형을 찾는 phenotyping, 그리고 데이터 공유를 위한 비식별화가 서로 다른 응용 축으로 제시된다.

동시에 저자들은 기술 축도 따로 정리한다. 4쪽 Figure 3은 EHR 분석에 쓰인 주요 딥러닝 구조를 **supervised / unsupervised**로 나누고, MLP, CNN, RNN, LSTM, GRU, RBM, DBN, AE, DAE, VAE 등을 한 그림에 배치한다. 이 그림이 주는 메시지는 명확하다. EHR 연구는 특정 한 모델의 승리가 아니라, **데이터 유형과 과제에 맞춰 서로 다른 구조들이 병존**하고 있다는 것이다. 예를 들어 텍스트와 시계열에는 RNN/LSTM/GRU가, 표현학습과 phenotype discovery에는 AE와 RBM 계열이, 일부 코드 기반 예측에는 CNN이 활용된다.

## 3. Detailed Method Explanation

### 3.1 EHR 데이터의 구조와 표현 문제

저자들이 가장 먼저 강조하는 것은 EHR 데이터의 **이질성(heterogeneity)** 이다. 2~3쪽 설명에 따르면 EHR는 ICD, CPT, LOINC, RxNorm 같은 여러 코드 체계를 포함하며, 기관별 차이도 크다. 예를 들어 3쪽 Table II는 ICD-10, CPT, LOINC, RxNorm의 코드 수와 예시를 보여준다. 이런 현실은 모델링의 첫 단계가 “무슨 모델을 쓸까”가 아니라 **어떻게 코드를 표현하고 정렬할까**가 된다는 뜻이다. 이 문제의식이 뒤에서 concept representation과 patient representation 연구로 이어진다.

### 3.2 딥러닝 아키텍처 개요

논문 3~6쪽은 survey답게 주요 아키텍처의 기본 원리를 짧게 요약한다.

먼저 MLP는 fully connected feedforward network로 설명되며, hidden unit는

$$
h_i = \sigma\left(\sum_{j=1}^{d} x_j w_{ij} + b_{ij}\right)
$$

형태로 계산된다고 정리된다. 이는 가장 단순한 형태의 deep model이며, 이후 많은 구조의 마지막 prediction layer로도 쓰인다.

CNN은 1D/2D convolution을 통해 지역적 패턴을 추출하는 구조로 소개된다. 5쪽에서는 1차원 convolution을

$$
C_{1d} = \sum_{a=-\infty}^{\infty} x(a)w(t-a)
$$

2차원 convolution을

$$
C_{2d} = \sum_m \sum_n X(m,n)K(i-m,j-n)
$$

로 제시한다. 저자들은 이미지뿐 아니라 1차원 의료 시계열도 local segment의 집합으로 볼 수 있기 때문에 CNN이 적용 가능하다고 설명한다.

RNN은 시간 순서가 있는 데이터에 적합한 구조로 제시된다. 5쪽 Figure 5는 입력 시퀀스를 따라 hidden state가 누적되는 구조를 보여주며, 저자들은 standard RNN보다 **LSTM과 GRU 같은 gated RNN**이 더 긴 temporal dependency를 다루는 데 유리하다고 요약한다. 이는 이후 temporal outcome prediction 연구에서 가장 중요한 기술 축이 된다.

Autoencoder는 입력을 latent vector로 압축했다가 다시 복원하는 방식으로 설명된다. 5쪽의 식은

$$
z = \sigma(Wx+b)
$$

$$
\tilde{x} = \sigma(W' z + b')
$$

로 주어지며, reconstruction error를 줄이면서 latent representation $z$를 학습한다. 저자들은 AE를 단순 압축기가 아니라 **비지도 representation learning의 핵심 도구**로 본다. 6쪽 Figure 6은 stacked autoencoder가 어떻게 여러 층으로 확장되는지 보여준다.

RBM은 확률적 generative model로 설명된다. 6쪽에서 canonical RBM energy는

$$
E(v,h) = -b^T v - c^T h - W v^T h
$$

로 제시된다. 저자들은 RBM이 입력 분포 자체를 학습하는 stochastic representation model이며, DBN의 구성 요소가 된다고 요약한다.

### 3.3 Information Extraction

6~7쪽은 임상노트 정보추출을 다룬다. 논문은 이 영역을 네 하위 문제로 나눈다.

* single concept extraction
* temporal event extraction
* relation extraction
* abbreviation expansion

6쪽 Figure 7은 admission note, discharge summary, transfer order 같은 임상노트로부터 약물명·용량, 질병명·심각도, 시간 표현, 관계 표현, 약어 확장을 뽑아내는 구조를 보여준다. 저자들은 이 문제를 clinical NLP의 핵심으로 보며, 기존 CRF 중심 접근보다 LSTM, Bi-LSTM, GRU, CNN이 더 좋은 성능을 내기 시작했다고 정리한다. 예를 들어 Jagannatha와 Yu의 연구는 clinical concept extraction을 sequence labeling으로 보아 CRF보다 RNN 계열이 더 우수했다고 소개된다. abbreviation expansion에서는 deep model 자체보다 **word2vec 기반 clinical embedding**이 강한 전처리·표현 수단으로 쓰였다고 설명한다.

### 3.4 EHR Representation Learning

7~9쪽의 representation learning은 이 논문의 중심 장 중 하나다. 저자들은 representation learning을 다시 두 개로 나눈다.

* concept representation
* patient representation

Concept representation은 ICD, 약물, 검사 코드 같은 이산적 medical concept를 벡터 공간으로 옮기는 문제다. 8쪽에 따르면 skip-gram/word2vec 발상이 자주 사용되며, 의료 코드 시퀀스를 문장처럼 보고 embedding을 학습한다. Choi 등의 연구는 이런 임베딩으로 heart failure prediction이나 code clustering을 수행했다. 다른 계열로는 RBM, AE를 이용해 concept vector를 학습한 사례도 소개된다. 핵심 목적은 비슷한 의료 개념이 latent space에서 가깝게 위치하도록 하는 것이다.

Patient representation은 환자를 하나의 고정 길이 벡터나 시간에 따라 변하는 상태 벡터로 나타내는 문제다. 8쪽 Figure 8은 매우 sparse한 환자 벡터를 autoencoder로 압축해 fixed-size dense vector를 만드는 과정을 보여준다. Miotto의 DeepPatient는 stacked AE로 raw code를 patient vector로 바꾸었고, DeepCare는 diagnosis/intervention vector를 결합해 LSTM으로 disease progression을 모델링했다. Doctor AI는 (event, time) 시퀀스를 GRU에 넣어 각 시점의 환자 상태를 표현한다. 즉 이 장의 메시지는, **딥러닝의 진짜 강점이 raw clinical code를 더 예측 친화적인 환자 표현으로 바꾸는 데 있다**는 점이다.

### 3.5 Outcome Prediction

9~10쪽은 결과 예측을 두 부류로 나눈다.

* static outcome prediction
* temporal outcome prediction

9쪽 Table IV는 대표 과제를 정리한다. static에서는 heart failure, hypertension, infections, osteoporosis, suicide risk stratification 같은 문제가 있고, temporal에서는 cardiovascular / pulmonary disease, diabetes, mental health, readmission, renal outcome, postoperative outcome, multi-outcome ICD prediction 등이 포함된다. 사용 모델은 MLP, CNN, RBM, DBN, LSTM, GRU, AE로 다양하다. 저자들은 static prediction에서는 representation learning으로 만든 patient vector를 logistic regression이나 MLP 같은 비교적 단순한 분류기에 넣어도 성능 향상이 나타난다고 설명한다. temporal prediction에서는 LSTM/GRU와 CNN이 더 자주 쓰이며, 다음 질병, 다음 개입, 일정 기간 내 재입원 같은 과제가 대표적이다.

특히 temporal outcome prediction에서는 몇 가지 중요한 방향성이 보인다. Cheng의 연구는 temporal matrix에 CNN을 적용해 CHF/COPD onset을 예측했고, Lipton은 multivariate ICU time series에 LSTM과 target replication을 사용해 128개 진단을 예측했다. Doctor AI는 future diagnosis와 medication intervention을 예측했고, DeepCare는 diagnosis와 intervention을 함께 model하며 disease progression을 다뤘다. Deepr는 CNN으로 unplanned readmission을 예측했다. 이를 종합하면, 이 논문은 **EHR 예측 문제의 중심 축이 점점 static one-shot classification에서 temporal patient trajectory modeling으로 이동하고 있다**고 읽힌다.

### 3.6 Computational Phenotyping

10~12쪽의 computational phenotyping은 이 논문의 철학을 가장 잘 드러내는 부분이다. 저자들은 phenotyping을 “기존 질병 정의를 넘어서 데이터가 스스로 말하게 하는 작업”으로 설명한다. 크게 두 부류가 있다.

* new phenotype discovery
* improving existing definitions

11쪽 Figure 9는 Beaulieu-Jones와 Greene의 denoising autoencoder 기반 phenotype stratification 예시를 보여준다. raw clinical descriptor에서는 case와 control이 잘 분리되지 않지만, AE를 훈련한 뒤 t-SNE 공간에서는 두 집단이 명확히 갈린다. 이는 비지도 representation이 latent disease structure를 드러낼 수 있음을 시각적으로 보여준다. Miotto의 DeepPatient, Cheng의 CNN, Mehrabi의 RBM, Lasko의 uric-acid time series AE 등도 모두 이 방향의 예시로 소개된다. 반면 기존 phenotype 개선 쪽에서는 Lipton의 LSTM, Che의 ontology-regularized MLP/DAE처럼 supervised multi-label prediction과 연결된 연구가 등장한다. 저자들이 보기에 phenotyping은 딥러닝의 “let the data speak for itself” 철학이 임상으로 이어지는 대표 사례다.

### 3.7 Clinical Data De-identification

12쪽은 비식별화를 다룬다. 임상노트는 PHI를 포함하므로 공개 데이터 공유의 큰 장애물이 된다. 따라서 이름, 병원명, 날짜, 지리정보 등을 자동으로 제거하는 것이 중요하다. Dernoncourt 등의 Bi-LSTM + character/word embedding, Shweta 등의 RNN variant가 CRF baseline을 능가했다고 정리된다. 이 장은 규모는 작지만, 저자들이 미래의 cross-institutional data sharing을 위해 비식별화가 매우 중요한 연구 방향이라고 본다는 점이 중요하다.

### 3.8 Interpretability

12~13쪽은 해석 가능성을 별도 장으로 다룬다. 12쪽 Table V는 네 가지 해석 전략을 정리한다.

* maximum activation
* constraints
* qualitative clustering
* mimic learning

Nguyen의 Deepr나 Med2Vec는 activation maximization을, Tran의 eNRBM과 Med2Vec는 non-negativity·ontology smoothing·regularization을, 여러 표현학습 연구는 PCA/t-SNE 기반 qualitative clustering을 사용했다. 또 Che의 mimic learning은 deep model의 soft target을 gradient boosting tree가 모사하게 하여 feature importance를 확보하려는 시도로 소개된다. 이 장의 메시지는 명확하다. **성능이 좋아도 black box이면 임상 적용이 어렵다**는 것이다. 저자들은 interpretability를 앞으로의 핵심 과제로 분명히 지목한다.

## 4. Experiments and Findings

이 논문은 survey이므로 자체 benchmark 실험을 수행하지는 않는다. 대신 field 전체에서 나타나는 정성적 발견을 정리한다.

첫째, 2쪽 Figure 1에 따르면 deep EHR 관련 출판은 2015년 이후 급증한다. 특히 application area와 technical method 모두에서 representation learning, prediction, RNN/LSTM/GRU, autoencoder 계열의 증가가 두드러진다. 이는 이 분야가 2015~2017년 사이 급격히 형성되었음을 보여준다.

둘째, 논문 전반의 정리는 **EHR 분석에서 딥러닝이 전통적 baseline을 자주 능가한다**는 쪽으로 기울어 있다. concept extraction에서는 RNN 계열이 CRF를, abbreviation expansion에서는 embedding 기반 접근이 기존 baseline을, outcome prediction에서는 AE/CNN/RNN 기반 접근이 raw feature 기반 선형 모델보다 좋은 결과를 보였다고 요약된다. 특히 7~10쪽의 각 응용 장은 거의 일관되게 “deep representation을 쓰면 성능 향상”이라는 흐름을 보여준다.

셋째, 환자 표현과 코드 표현은 그 자체가 목적이라기보다 **보조 prediction task를 통해 간접 평가**되는 경우가 많다. 9쪽은 representation learning 평가 지표가 AUC, precision@k, recall@k, accuracy, F1, 그리고 t-SNE/heatmap 기반 qualitative inspection으로 매우 다양하다고 설명한다. 이는 아직 unified benchmark가 없다는 뜻이기도 하다. 즉 성능이 좋아 보이는 연구가 많지만, 연구 간 직접 비교는 어렵다.

넷째, phenotyping과 interpretability 연구는 딥러닝이 단지 “예측 정확도 개선기”가 아니라, **질병 구조를 더 세밀하게 파악하고, 모델 내부를 해석하려는 임상적 요구**와 연결되고 있음을 보여준다. 11쪽 Figure 9, 12쪽 Table V, 13쪽 Figure 10은 이 흐름을 상징적으로 보여주는 시각 자료다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 강점은 크게 세 가지다.

첫째, **구조화가 매우 좋다.** 응용 과제 중심 분류와 아키텍처 중심 분류를 함께 제공해서, 독자가 “무엇을 하려는 연구인가”와 “어떤 모델을 쓰는가”를 동시에 이해할 수 있다. Table III, Table IV, Figure 3이 특히 유용하다.

둘째, **EHR의 고유한 어려움**을 제대로 짚는다. 13~15쪽 discussion은 data heterogeneity, irregular measures, clinical text, unified representation, de-identification, benchmark, interpretability를 미래 과제로 정리한다. 단순히 성능 개선을 찬양하지 않고, 왜 EHR가 이미지나 텍스트보다 어려운지, 앞으로 어디가 병목인지 분명히 말한다.

셋째, **표현학습을 중심 축으로 본 관점**이 설득력 있다. 저자들은 이미지에서 픽셀→객체, NLP에서 단어→문장 표현이 중요했듯, EHR에서도 코드와 환자의 표현학습이 출발점이라고 본다. 이 framing은 이후 의료 AI 연구 흐름과도 잘 맞는다. 13쪽 discussion은 이 유비를 명시적으로 제시한다.

한계도 분명하다.

첫째, 이 논문은 2017년 8월까지의 문헌만 다룬다. 따라서 transformer, foundation model, large clinical language model, multimodal pretraining, self-supervised EHR representation 같은 이후의 큰 흐름은 포함되지 않는다. 오늘날 기준으로는 field의 초기 형성기를 정리한 survey로 읽는 것이 맞다.

둘째, 저자들 스스로 강하게 지적하듯 **benchmark 부재와 재현성 부족**이 심각하다. 15쪽 discussion은 많은 연구가 사설 데이터셋을 사용하기 때문에 state-of-the-art 주장을 외부에서 검증하기 어렵다고 말한다. 이는 survey 전반에서 가장 중요한 비판점이다.

셋째, 이 논문이 다루는 많은 성공 사례는 still divide-and-conquer 방식이다. 즉 코드만, 시계열만, 텍스트만 따로 다루는 경우가 많고, truly unified patient representation은 아직 “holy grail”로 남아 있다고 14쪽에서 직접 말한다. 다시 말해 field는 유망하지만 아직 조각난 상태다.

비판적으로 해석하면, 이 논문의 진짜 메시지는 “딥러닝이 EHR에서 최고다”보다는, **EHR는 너무 이질적이어서 좋은 성능보다 먼저 좋은 표현과 통합 구조가 필요하다**는 데 있다. 그리고 이 표현 문제를 푸는 과정에서 representation learning, phenotyping, interpretability, de-identification이 모두 하나의 큰 연구 생태계로 연결된다는 점을 잘 보여준다.

## 6. Conclusion

이 논문은 EHR 분석에 적용된 초기 딥러닝 연구를 가장 체계적으로 묶은 survey 중 하나다. 정보추출, 표현학습, 결과 예측, phenotyping, 비식별화로 응용 과제를 나누고, MLP, CNN, RNN/LSTM/GRU, AE, RBM/DBN 등 주요 아키텍처를 연결해 설명한다. 전체적으로 드러나는 결론은 다음과 같다. **딥러닝은 EHR의 복잡한 구조와 장기 의존성을 다루는 데 유망하며, 특히 코드·환자 표현학습과 temporal prediction에서 강한 성과를 보였다. 하지만 데이터 이질성, 불규칙한 측정, 임상노트의 비구조성, 통합 표현 부재, 재현성 부족, 해석 가능성 문제는 여전히 핵심 과제로 남아 있다.**

실무적으로 이 논문은 “EHR 딥러닝이 어디까지 왔는가”를 보는 입문 지도에 가깝고, 연구적으로는 이후 등장한 multimodal EHR modeling, self-supervised learning, clinical foundation model을 이해하기 위한 출발점 역할을 한다. 특히 14~15쪽 discussion이 말하듯, 앞으로의 핵심은 개별 homogeneous input을 따로 다루는 것을 넘어 **모든 환자 데이터를 포괄하는 unified patient representation**을 만드는 데 있다. 이 지적은 지금 봐도 여전히 유효하다.
