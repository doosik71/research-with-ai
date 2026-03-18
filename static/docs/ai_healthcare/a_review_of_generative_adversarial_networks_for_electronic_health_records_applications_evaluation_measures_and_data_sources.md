# A review of Generative Adversarial Networks for Electronic Health Records: applications, evaluation measures and data sources

## 1. Paper Overview

이 논문은 Electronic Health Records(EHRs)용 synthetic data 생성에서 **Generative Adversarial Networks(GANs)** 가 어떤 역할을 해왔는지 정리한 종합 리뷰 논문이다. 저자들은 EHR가 clinical research와 point-of-care application에 매우 중요하지만, 개인정보 보호, 데이터 공유 제한, 결측, 불균형, 잡음, 이질성, 불규칙 샘플링 같은 문제 때문에 실제 활용이 어렵다고 본다. 이런 문제를 완화하기 위한 방법으로 deep generative model, 특히 GAN 기반 synthetic EHR 생성이 유망하다고 보고, 관련 연구들을 응용 목적, 평가 방식, 데이터셋 관점에서 체계적으로 정리한다.

이 논문의 핵심 문제의식은 단순히 “GAN이 EHR를 생성할 수 있는가”가 아니다. 저자들은 실제로 더 중요한 질문을 세운다. 즉, **EHR용 GAN이 어떤 의료 문제를 해결하려고 했는가**, **생성된 데이터의 품질과 프라이버시를 무엇으로 평가해야 하는가**, **어떤 공개 데이터가 이 분야의 공통 벤치마크 역할을 할 수 있는가**를 함께 다룬다. 논문 초록과 서론에서 저자들은 이 리뷰가 GAN for structured EHRs, applications, evaluation, challenges를 함께 다루는 포괄적 개관이며, 특히 synthetic EHR 평가 지표를 넓게 분류해 제시한 첫 작업이라고 주장한다.  

이 문제가 중요한 이유는 의료 데이터에서 “비슷해 보이는 가짜 데이터”를 만드는 것만으로는 충분하지 않기 때문이다. 실제 의료 연구와 clinical decision support에 쓸 수 있으려면, 데이터가 원본 분포를 잘 반영해야 하고, downstream task에서 유용해야 하며, 동시에 환자 정보 유출 위험이 낮아야 한다. 논문은 이 세 축—**fidelity, utility, privacy**—의 균형을 EHR용 GAN 연구의 중심 과제로 본다.  

## 2. Core Idea

이 논문의 중심 아이디어는 EHR용 GAN 연구를 단순 모델 나열이 아니라, **의료 응용 중심의 taxonomy와 평가 프레임워크 중심으로 재정리**하는 데 있다. 서론에서 저자들은 기존 리뷰들이 GAN evaluation 일반론, medical imaging, time-series, observational health data를 다루긴 했지만, structured EHR에 특화해 applications, evaluation, challenges를 함께 정리한 리뷰는 부족했다고 말한다. 이 논문은 그 공백을 메우기 위해, GAN 연구를 “무엇을 생성했는가”보다 “무엇을 해결하려 했는가” 기준으로 묶는다.

논문이 제시하는 주요 응용 범주는 다섯 가지다.

1. 다양한 유형의 EHR 생성
2. Semi-supervised learning 및 data augmentation
3. Missingness imputation
4. Treatment effect estimation
5. Privacy preservation  

이 분류는 상당히 유용하다. 왜냐하면 GAN for EHR 연구는 겉으로는 모두 “synthetic data generation”처럼 보이지만, 실제로는 목적이 크게 다르기 때문이다. 어떤 모델은 discrete tabular ICD code를 만들고, 어떤 모델은 longitudinal vital sign 같은 time-series를 생성하며, 어떤 모델은 missing value를 채우고, 어떤 모델은 counterfactual treatment effect estimation을 돕고, 어떤 모델은 differential privacy 보장을 강화하려 한다. 이 논문은 이러한 서로 다른 목표를 한 프레임 안에서 비교 가능하게 만든다.

또 하나의 핵심 아이디어는 **평가를 구조적으로 분해**한 점이다. 논문은 synthetic EHR 평가를 질적/양적 평가로 나누고, 양적 평가는 다시 데이터의 어떤 측면을 보는지에 따라 정리한다. 표 1의 주석에서 제시된 구성은 다음과 같다.

* DWS: Dimension-wise Similarity
* LDS: Latent Distribution Similarity
* JDS: Joint Distribution Similarity
* IDRS: Inter-dimensional Relationship Similarity
* PP: Privacy Preservation
* DU: Data Utility
* Qual: Qualitative Evaluation

이 논문의 상대적 참신성은 “새 GAN을 제안했다”는 데 있지 않다. 오히려 **이 분야에서 무엇을 평가해야 하는지에 대한 공통 언어를 만들었다**는 데 있다. 저자들도 section 5에서 GAN evaluation에는 아직 합의가 없다고 분명히 말한다.

## 3. Detailed Method Explanation

이 논문은 survey이므로, 여기서의 “method”는 하나의 새로운 알고리즘이 아니라 **GAN for EHR 연구를 해석하는 개념적 틀**이다. 그래도 논문은 먼저 GAN의 기본 수학적 구조를 짚고, 그 다음 EHR 데이터 특성과 응용별 GAN 계열을 정리한다.

### 3.1 GAN 기본 원리

논문은 GAN을 generator $G$ 와 discriminator $D$ 가 경쟁적으로 학습하는 구조로 설명한다. Generator는 latent noise vector $\mathbf{z}$ 를 받아 synthetic sample $G(\mathbf{z})$ 를 만들고, discriminator는 real data $\mathbf{x}$ 와 generated sample을 구분하도록 학습된다. 저자들은 기본 목적함수를 다음처럼 제시한다.

$$
\min_G \max_D V(D,G)
====================

\mathbb{E}*{\mathbf{x}}[\log D(\mathbf{x})]
+
\mathbb{E}*{\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]
$$

비록 ar5iv 수식 렌더링이 약간 거칠지만, 의도는 고전적 GAN objective 그대로다. discriminator는 real을 잘 구분하려 하고, generator는 discriminator를 속이도록 학습된다.

저자들은 이어서 GAN 연구의 주요 확장들—CGAN, DCGAN, InfoGAN, RCGAN, WGAN, CycleGAN, StarGAN, DSCGAN—을 짧게 정리한다. 이 부분은 왜 EHR용 GAN들이 서로 다르게 설계되었는지 이해하는 배경이 된다. 예를 들어 RCGAN은 순차 데이터 생성에 적합하므로 time-series EHR에 연결되고, WGAN은 Wasserstein distance 기반 loss로 training stability를 개선해 EHR의 복잡한 분포를 다루는 데 자주 쓰인다.

논문이 강조하는 기본적인 GAN 병목은 **mode collapse** 와 **vanishing gradients** 다. 특히 EHR처럼 복잡하고 heterogeneous한 데이터에서는 generator가 데이터 다양성을 충분히 커버하지 못하거나, discriminator가 너무 강해 generator 학습이 망가질 위험이 크다. 그래서 WGAN류, minibatch discrimination, unrolled GAN, noise injection 같은 안정화 기법이 중요하다고 설명한다.

### 3.2 EHR 데이터 특성

이 논문에서 중요한 부분은 “왜 EHR가 이미지보다 어렵냐”를 설명하는 section 3이다. 저자들은 structured EHR를 크게 **tabular** 와 **time-series** 로 나누고, 각 변수는 discrete, categorical, continuous일 수 있다고 설명한다. 실제 환자 기록에서는 이들이 한 레코드 안에 혼재하므로, EHR는 본질적으로 heterogeneous하다.

이 점이 방법론적으로 매우 중요하다. 이미지 생성에서는 픽셀 공간이 비교적 균질하지만, EHR 생성에서는 diagnosis code 같은 multi-label discrete feature, age 같은 integer, lab value 같은 continuous variable, longitudinal vital sign 같은 irregular time-series가 동시에 존재한다. 따라서 “EHR용 GAN”은 사실 단일 문제가 아니라 여러 데이터 유형을 동시에 다루는 문제다. 논문이 응용별뿐 아니라 데이터 유형별로도 연구를 분리해서 설명하는 이유가 여기 있다.

### 3.3 응용별 GAN 방법론 정리

#### 3.3.1 Diverse EHR generation

논문은 초기 연구들이 주로 **discrete tabular EHR** 생성에 집중했다고 설명한다. 대표 예시인 **medGAN** 은 원래 GAN이 binary/discrete count feature를 가진 tabular EHR에 잘 맞지 않는 문제를 해결하기 위해 autoencoder를 결합해 salient representation을 먼저 학습한 뒤 GAN이 multi-label discrete binary/count feature 분포를 배우게 했다.

그 다음 **medWGAN** 과 **medBGAN** 은 medGAN을 WGAN-GP와 BGAN 계열로 확장해 학습 안정성과 생성 품질을 높이려는 방향으로 발전했다. 표 1에서도 medGAN, medWGAN/medBGAN이 discrete tabular EHR generation을 대표하는 모델로 정리된다.  

연속 시계열 EHR 쪽에서는 **RGAN/RCGAN** 이 대표적이다. 이들은 recurrent neural network를 이용해 continuous time-series EHR를 생성하도록 설계되었고, 논문은 이것이 이후 여러 time-series EHR GAN 연구를 촉진했다고 설명한다. 표 1에서도 RGAN/RCGAN은 Philips eICU 데이터셋에서 continuous time-series EHR generation에 사용된 것으로 정리된다.

이 밖에도 **GAN for DLEs** 는 drug-laboratory effects라는 특정한 continuous time-series를 생성했고, **RadialGAN** 은 여러 tabular dataset을 leverage하는 방향을 탐색했다. 즉 diverse generation 범주 안에서도 단순 “샘플 생성”이 아니라, 도메인 특화된 structured health record generation으로 빠르게 분기되었다는 것이 논문의 해석이다.

#### 3.3.2 Evaluation framework

논문의 방법론적 핵심은 section 5의 평가 체계다. 저자들은 GAN 평가에 아직 합의가 없다고 지적하면서, synthetic EHR의 평가는 최소한 네 가지 목적을 포함한다고 말한다.

* data distribution 근사 정도
* privacy 유지 여부
* downstream ML task utility
* model performance 자체의 특성

이때 단일 metric으로는 충분하지 않으므로, 정량 평가를 데이터 측면별로 분류해 정리한다. 표 1의 DWS/LDS/JDS/IDRS/PP/DU/Qual 분류가 그것이다. 이 구조는 실무적으로도 의미가 크다. 예를 들어 marginal histogram만 비슷하면 DWS는 좋을 수 있지만, feature 간 상관구조가 깨지면 IDRS나 JDS는 나쁠 수 있다. 또한 분포는 비슷해도 privacy leakage가 크면 PP가 나쁘고, downstream prediction이 잘 안 되면 DU가 낮다. 논문은 이런 trade-off를 분해해서 봐야 한다고 말한다.  

특히 **Data Utility** 평가에서 저자들은 **TSTR (Train on Synthetic, Test on Real)** 를 대표적 utility 평가로 소개한다. 이름 그대로 synthetic data로 학습한 모델을 held-out real data에서 테스트해 실제 활용 가능성을 본다. 저자들은 synthetic/real 양쪽 성능을 함께 보고 baseline을 비교해야 utility를 정확히 해석할 수 있다고 강조한다.  

#### 3.3.3 Privacy evaluation

Privacy는 이 논문에서 매우 중요한 축이다. 저자들은 synthetic output이 직접 원본 샘플과 1:1로 대응되지 않는다는 점에서 privacy를 암묵적으로 도와주지만, 민감한 EHR에서는 여전히 information leakage가 발생할 수 있다고 말한다. 특히 section 5/7에서 membership inference attack, attribute disclosure attack, model inversion attack 같은 adversarial attack들이 empirical privacy evaluation에 사용된다고 정리한다.  

또한 differential privacy를 GAN training에 적용하는 흐름도 설명한다. 다만 저자들은 privacy guarantee가 강할수록 fidelity와 utility가 나빠질 수 있는 **privacy-similarity trade-off** 를 반복되는 주제로 본다. 그래서 privacy를 평가할 때는 정보 유출 위험만 볼 것이 아니라, 동시에 원본 분포 보존과 utility 저하도 함께 측정해야 한다고 권고한다.  

## 4. Experiments and Findings

이 논문은 새로운 모델을 하나 학습해 benchmark 결과를 제시하는 논문이 아니라 review paper이므로, “실험 결과”는 곧 **문헌 검토를 통해 얻은 분야 수준의 발견**이다.

### 4.1 주요 응용 흐름

저자들이 정리한 바에 따르면, GAN for EHR 연구는 크게 다섯 방향으로 전개되었다. diverse EHR generation이 가장 초기이자 중심 흐름이고, 이후 data augmentation/semi-supervised learning, imputation, treatment effect estimation, privacy preservation으로 확장되었다. 이는 이 분야가 단순 샘플 생성에서 시작해 점차 **의료 분석 파이프라인의 보조 기술**로 진화했음을 보여준다.

특히 tabular discrete EHR 생성에서는 medGAN 계열이, continuous time-series에서는 RGAN/RCGAN 계열이 대표적 출발점이었고, 이후 heterogeneous EHR 생성으로 관심이 옮겨갔다. 즉, 데이터 구조가 단순할수록 먼저 GAN이 적용되었고, 점차 더 현실적인 mixed-type EHR로 확장된 흐름을 볼 수 있다.  

### 4.2 공개 데이터와 재현성

표 1에서 반복적으로 등장하는 대표 데이터셋은 **MIMIC-III**, **eICU**, **NHIRD Taiwan**, **SEER**, **VUMC Synthetic Derivative** 등이다. 이 논문은 open access 여부까지 함께 정리하면서, 실제 연구 재현성과 비교 가능성을 중요한 기준으로 본다. 표 1 설명에 따르면 dataset size는 $(N/R)$ 형식으로 정리되고, `✓*` 는 신청 절차를 거쳐 접근 가능한 데이터셋을 뜻한다.  

저자들은 공개 코드가 늘어나는 것은 긍정적 흐름이라고 보지만, 코드 링크가 비작동하거나 공개가 불완전한 경우도 있다고 지적한다. 또 critical care 데이터나 소규모 공개 데이터가 유용하긴 해도, longitudinal population-scale question을 다루기에는 한계가 있어 공개 데이터 다양성이 더 필요하다고 본다.

### 4.3 평가의 공통 결론

논문이 가장 강하게 말하는 발견 중 하나는 **EHR GAN 평가에 아직 표준 합의가 없다는 점**이다. 저자들은 어떤 metric 하나로 state-of-the-art를 판별하기 어렵고, 서로 다른 metric이 서로 다른 limitation과 trade-off를 낳기 때문에 현재로서는 명확한 최고 모델을 정하기 어렵다고 말한다.  

이는 매우 중요한 메시지다. 예를 들어 분포 유사도가 높은 모델이 꼭 downstream utility가 높은 것은 아니고, privacy를 강화한 모델이 utility를 희생할 수 있다. 따라서 논문은 향후 연구가 metric selection guideline과 metric weighting standardization을 발전시켜야 한다고 제안한다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **GAN for EHR 분야를 응용, 평가, 데이터셋 세 축으로 동시에 정리했다는 점**이다. 많은 리뷰가 모델 아키텍처만 훑고 지나가는데, 이 논문은 어떤 문제를 풀었는지, 무엇으로 평가했는지, 어떤 데이터에서 실험했는지까지 연결해서 보여 준다. 그래서 단순 literature list보다 훨씬 실용적인 참고 지도가 된다.

둘째, **평가 taxonomy** 가 특히 유용하다. EHR synthetic data는 눈으로 realism을 판단하기 어렵기 때문에, 평가 체계를 구조화하는 것이 중요하다. DWS, LDS, JDS, IDRS, privacy, utility, qualitative analysis를 구분해 정리한 것은 이후 연구자가 “내 모델은 무엇을 잘했고 무엇을 아직 검증하지 않았는가”를 점검하는 체크리스트로 바로 쓸 수 있다.  

셋째, **privacy-similarity trade-off** 를 리뷰의 중심 주제로 명시했다는 점도 좋다. 의료 synthetic data에서는 privacy가 부수적 이슈가 아니라 핵심 목표이므로, fidelity만 강조한 연구나 privacy만 강조한 연구가 모두 불완전하다는 문제의식을 잘 드러낸다.

### 한계

첫 번째 한계는 이 논문이 2022년 1월까지의 문헌을 기준으로 정리되었다는 점이다. 따라서 이후 등장한 diffusion model 기반 healthcare synthetic data, transformer 기반 tabular/time-series generator, recent foundation model 계열은 반영되지 않는다. 이건 논문의 결함이라기보다 시점의 한계다.

둘째, review 논문답게 개별 모델의 세부 technical comparison은 상대적으로 얕다. 예를 들어 medGAN과 이후 mixed-type tabular generator 사이의 내부 구조 차이, privacy attack 실험 세부 설정, causal validity of treatment effect estimation 같은 부분은 깊이 있는 비판적 비교보다는 개관 수준에 가깝다.

셋째, 저자 스스로 인정하듯 **현재 평가 metric만으로는 state-of-the-art를 결정하기 어렵다**. 이는 분야의 한계이기도 하지만, 동시에 이 리뷰가 제시하는 결론이 다소 “정리”에 머물고 “판정”까지는 가지 못하게 만든다.

### 비판적 해석

이 논문을 오늘 시점에서 읽으면 두 가지 의미가 있다.

하나는 역사적 의미다. 이 논문은 EHR synthetic data 연구가 **“모델 만들기” 중심에서 “어떻게 검증하고 안전하게 쓸 것인가” 중심으로 이동해야 한다**는 문제의식을 분명히 했다. 이건 지금 봐도 여전히 유효하다.

다른 하나는 기술적 과도기성이다. 논문이 다루는 GAN 계열은 당시엔 중요한 흐름이었지만, 이후 synthetic tabular/time-series modeling에서는 diffusion, autoregressive transformer, copula+deep hybrid, privacy-aware tabular generator 같은 대안이 많아졌다. 그럼에도 이 리뷰의 평가 프레임은 지금도 살아남을 수 있다. 즉, **모델은 바뀌어도 fidelity-utility-privacy의 삼각 구도는 그대로다**.

## 6. Conclusion

이 논문은 structured EHR를 위한 GAN 연구를 응용, 평가, 데이터셋, 프라이버시 관점에서 정리한 포괄적 리뷰다. 저자들은 EHR의 활용이 개인정보 보호와 데이터 품질 문제로 제한된다는 점에서 synthetic data generation의 필요성을 제기하고, GAN 기반 연구를 diverse EHR generation, data augmentation, imputation, treatment effect estimation, privacy preservation의 다섯 범주로 나눠 정리한다. 또한 synthetic EHR 평가를 distribution similarity, inter-feature relation, privacy, downstream utility 관점으로 체계화하고, 공개 데이터와 코드 접근성까지 함께 검토한다.  

이 논문의 가장 중요한 메시지는 명확하다. **EHR용 GAN의 진짜 문제는 “그럴듯한 샘플 생성”이 아니라 “안전하고 유용하며 검증 가능한 synthetic data 생성”** 이다. 저자들은 현재 표준 평가 체계가 부족하며, privacy와 similarity의 trade-off를 함께 보고, utility를 task 맥락에 맞게 해석해야 한다고 제안한다. 이 점에서 이 논문은 단순한 survey를 넘어, 이후 healthcare synthetic data 연구를 위한 methodological checklist에 가깝다.
