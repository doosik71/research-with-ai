# Split Learning for collaborative deep learning in healthcare

## 1. Paper Overview

이 논문은 의료 분야에서 데이터 공유 제약, 기관 간 협업 비용, 그리고 라벨링된 데이터 부족 문제를 해결하기 위한 **distributed deep learning** 방식으로 **Split Learning**을 실제 의료 영상 문제에 적용한 연구다. 저자들은 의료기관들이 환자 원시 데이터를 외부로 반출하지 않으면서도 공동으로 딥러닝 모델을 학습할 수 있는 협업 구조가 필요하다고 보고, 이를 위해 **U-shaped split learning configuration**을 사용해 성능을 평가했다. 구체적으로는 하나의 중앙 서버와 여러 의료기관(client)이 모델을 나눠 갖고 학습하며, 환자 원시 데이터와 라벨을 직접 공유하지 않는 협업 학습 체계를 제안·실험한다.

논문이 다루는 핵심 문제는 명확하다. 의료 AI는 일반적으로 대규모 학습 데이터가 필요하지만, 실제 의료 현장에서는 데이터가 병원마다 분산되어 있고, 개인정보보호 및 규제(HIPAA 등)로 인해 중앙집중식 데이터 통합이 어렵다. 따라서 “데이터는 각 병원에 남겨둔 채, 모델만 협업적으로 학습할 수 있는가?”가 연구 질문이 된다. 이 문제는 특히 의료영상처럼 기관 간 데이터 편차가 크고, 단일 기관 표본수가 제한적인 경우 매우 중요하다. 저자들은 split learning이 이런 상황에서 **중앙집중식 학습에 가까운 성능을 유지하면서**, 비협업(non-collaborative) 방식보다 우수한 결과를 낼 수 있음을 보이려 한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 신경망 전체를 각 참여자가 모두 갖는 대신, **네트워크를 여러 구간(link)으로 나누어 서로 다른 위치에 배치**하는 것이다. 즉, 각 병원은 모델의 앞부분과 뒷부분만 가지고, 중앙 서버는 중간의 큰 연산 블록을 갖는다. 이렇게 하면 환자 원시 이미지나 라벨은 병원 밖으로 나가지 않지만, 중간 feature representation만 서버와 주고받으면서 공동 학습이 가능하다.

저자들이 사용하는 구체적 구성은 **U-shaped split learning**이다. forward propagation 기준으로 보면,

* **front**: 각 클라이언트에 위치, 원시 입력 데이터를 받아 중간 표현으로 변환
* **center**: 중앙 서버에 위치, 중간 표현을 받아 대부분의 연산 수행
* **back**: 다시 각 클라이언트에 위치, 최종 예측을 산출하고 로컬 라벨로 loss 및 gradient 계산

이라는 구조다. 이 구조의 중요한 특징은 **raw data sharing도 없고 label sharing도 없다는 점**이다. 즉, 중앙 서버는 환자 원본 이미지나 정답 레이블을 보지 못하고도 전체 네트워크 학습에 참여한다. 의료 협업에서 매우 실용적인 설계다.

이 논문의 novelty는 절대적으로 새로운 알고리즘을 제안했다기보다, **split learning을 의료 분야에 최초 적용했다는 실증적 기여**에 있다. 저자들은 기존에 federated learning, model averaging, LS-SGD, cyclical weight transfer 같은 distributed learning 방법들이 존재하지만, 의료 환경에서의 실제 제약을 고려할 때 split learning이 계산량 분배, privacy, bandwidth, 비동기적 협업 측면에서 장점이 있다고 주장한다.

## 3. Detailed Method Explanation

### 3.1 분산 학습 방식의 배경

저자들은 distributed learning 방법들을 다음 기준으로 비교해야 한다고 본다.

* centralized setup 대비 성능 유지 여부
* privacy 보장 수준
* bandwidth 사용량
* 계산 부하의 분산 방식

model averaging이나 LS-SGD는 모든 클라이언트가 동기식으로 참여해야 하므로 네트워크 속도나 하드웨어 차이 때문에 실제 기관 협업에서 비효율적일 수 있다. 반면 split learning은 모든 기관이 동시에 참여할 필요 없이, **클라이언트가 순차적으로 학습에 참여**할 수 있어 운영 측면의 부담이 적다. 또한 큰 연산량을 중앙 서버가 담당할 수 있어, 계산 자원이 부족한 병원도 복잡한 모델 학습에 참여할 수 있다.

### 3.2 U-shaped split learning 구조

논문이 채택한 U-shaped split learning의 연산 흐름은 다음과 같다.

1. 병원(client)이 원시 데이터를 **front network**에 입력한다.
2. front network는 원시 데이터를 직접 노출하지 않는 **obfuscated intermediate representation**으로 바꾼다.
3. 이 표현이 중앙 서버의 **center network**로 전달된다.
4. center network가 대부분의 feature extraction/representation learning을 수행한 뒤, 또 다른 intermediate representation을 클라이언트의 **back network**로 보낸다.
5. back network가 최종 출력을 생성하고, 클라이언트가 보유한 로컬 라벨로 loss를 계산한다.
6. gradient는 역전파되어 center와 front까지 업데이트된다.

중요한 점은 클라이언트가 바뀔 때마다 **local link(front/back)의 state를 다음 client에 복사하여 이어서 학습**한다는 것이다. 즉, collaborative mode에서는 각 클라이언트가 모델을 한 epoch씩 순차적으로 학습하고, 모델의 로컬 부분도 다음 클라이언트로 넘겨진다. 이 방식은 “모두가 동시에 기다려야 하는” 동기 학습 문제를 피한다.

### 3.3 사용 데이터셋

논문은 두 개의 의료영상 과제를 사용한다.

#### (1) Diabetic Retinopathy (DR) fundus photo dataset

* 원본은 Kaggle Diabetic Retinopathy dataset
* 학습/검증에 **9000장** 사용
* 원래 multi-class 문제를 **정상/비정상 binary classification**으로 단순화
* 입력 이미지는 **256x256 RGB**로 다운샘플링

#### (2) CheXpert chest X-ray dataset

* 원본은 **224,316장 chest radiograph**, **65,240 patients**
* 14개 흉부 소견에 대한 **multi-label classification**
* 불확실 라벨은 baseline 방식을 따라 제외
* 일부 자주 등장하는 shape subset을 제외해 최종 **156,535장** 사용

두 데이터셋 모두

* **75% train / 25% validation**
* 환자 단위 중복 없이 분할
* train set만 여러 클라이언트에 균등 분배
* validation set은 분할하지 않고 그대로 유지

하는 방식으로 구성했다. validation을 분할하지 않은 이유는 클라이언트 수가 많아져도 일관된 검증 기준을 유지하기 위해서다.

### 3.4 사용 네트워크와 학습 설정

#### DR 데이터셋

* **ResNet-34**
* Glorot uniform initialization
* Adam optimizer with standard parameters: $\beta_1=0.9$, $\beta_2=0.999$
* learning rate: $10^{-4}$
* binary cross entropy loss
* data augmentation: 랜덤 회전(0–360도), 50% lateral/axial inversion
* validation accuracy가 30 epoch 이상 개선되지 않으면 plateau로 간주

#### CheXpert 데이터셋

* **DenseNet121**
* ImageNet pretraining
* sigmoid binary cross entropy 기반 multi-label loss
* Adam optimizer with standard parameters: $\beta_1=0.9$, $\beta_2=0.999$
* learning rate: $10^{-4}$
* batch size: 24
* augmentation: 50% lateral inversion
* validation loss가 5 epoch 이상 개선되지 않으면 plateau로 간주

즉, 이 논문의 주된 비교 대상은 새로운 backbone 자체가 아니라, **같은 task와 유사한 model family에서 협업 방식이 성능에 어떤 영향을 주는지**다. 다시 말해 “어떤 분산학습 프로토콜이 의료 데이터 분산 환경에서 유리한가”를 보는 실험 설계다.

### 3.5 비교 설정

저자들은 여러 수의 참여기관에 대해 다음을 비교한다.

1. **Single center / centrally hosted**
2. **Split learning based collaborative setup**
3. **Non-collaborative setup**

여기서 non-collaborative는 전체 데이터를 합치지 않고, 각 클라이언트가 자기에게 배정된 샘플 수만큼만 써서 단독으로 모델을 학습하는 설정이다. 반면 split learning collaborative setup은 데이터는 분산되어 있지만 모델은 협업적으로 업데이트된다. 이 비교를 통해, 기관 수가 늘어날수록 각 기관 데이터가 작아지는 상황에서 협업 여부가 얼마나 중요한지 확인한다.

### 3.6 평가 지표

* **DR**: validation set의 최고 classification accuracy
* **CheXpert**: 5개 competition task(Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion)에 대한 average AUROC

이 설정은 binary classification과 multi-label classification 모두에서 split learning이 유효한지 확인하기 위한 것이다.

## 4. Experiments and Findings

### 4.1 주요 정량 결과

논문의 가장 중요한 결과는, **split learning 기반 collaborative configuration의 성능이 client 수가 증가해도 거의 일정하게 유지되며**, non-collaborative 방식은 client 수가 2를 넘어서면서 급격히 악화된다는 점이다. abstract에서는 split learning 성능이 single-center study와 비교해 client 수와 무관하게 거의 일정했고, non-collaborative 설정과는 **2 client 이후부터 뚜렷한 차이**를 보였으며, 두 데이터셋 모두에서 **$p<0.001$** 수준의 유의성을 보였다고 요약한다.

### 4.2 DR 데이터셋 결과

Table 1을 보면 split learning은 client 수가 커져도 accuracy가 대체로 **0.80 후반대**를 유지하는 반면, non-collaborative는 client 수가 증가할수록 지속적으로 하락한다. 예를 들면,

* **3 clients**: split learning 0.868 vs non-collaborative 0.753
* **10 clients**: split learning 0.858 vs non-collaborative 0.676
* **20 clients**: split learning 0.860 vs non-collaborative 0.613
* **50 clients**: split learning 0.859 vs non-collaborative 0.588

이 결과는 데이터가 여러 기관으로 쪼개져 각 기관 sample size가 작아질수록, 단독 학습은 급격히 무너지지만 split learning은 협업 덕분에 성능을 거의 잃지 않는다는 뜻이다. 특히 50개 클라이언트 상황에서도 split learning accuracy가 0.859 수준인 반면 non-collaborative는 0.588로 매우 낮아져 격차가 극적이다.

### 4.3 CheXpert 데이터셋 결과

CheXpert에서도 같은 추세가 나타난다. 저자들은 split learning collaborative setting 대비 non-collaborative setting의 평균 성능이 유의하게 낮았고, 특히 **2개 초과 client**에서 차이가 뚜렷했다고 보고한다. 통계적으로는 **$\alpha=0.005$** 수준에서 두 표본 양측 t-test를 사용해 유의성을 확인했다.

CheXpert는 multi-label chest X-ray 분류라는 더 복잡한 의료영상 문제인데, 이 데이터셋에서도 같은 결론이 나온다는 점은 논문의 주장에 힘을 실어준다. 즉 split learning의 효과가 특정 단순 binary task에만 국한되지 않고, 더 큰 multi-label radiology task에도 일반화된다는 것이다.

### 4.4 실험이 실제로 보여주는 것

이 실험은 단순히 “협업이 좋다”는 수준을 넘어서, 의료기관 협업에서 다음을 보여준다.

첫째, **데이터를 중앙에 모으지 않아도 중앙집중형에 가까운 성능을 낼 수 있다**는 점이다. 이는 실제 병원 간 협업에서 가장 중요한 메시지다.

둘째, **기관 수가 늘어날수록 협업의 가치가 더 커진다**. 각 기관의 로컬 데이터만 보면 표본 수가 줄어들기 때문에 단독 모델은 쉽게 과적합되거나 일반화에 실패한다. split learning은 이를 구조적으로 보완한다.

셋째, **비동기적·순차적 client training** 방식이 실제 운영상 유리할 가능성을 시사한다. 모든 기관이 동시에 참여하지 않아도 되므로 현실적인 multi-center deployment에 더 적합할 수 있다.

## 5. Strengths, Limitations, and Interpretation

### Strengths

이 논문의 가장 큰 강점은 **문제 설정이 매우 현실적**이라는 점이다. 의료기관 간 데이터 공유의 가장 큰 장애물은 개인정보, 규제, 데이터 소유권, 저장 비용인데, 저자들은 split learning이 바로 이 제약을 겨냥한다고 설명한다. 실제로 U-shaped configuration은 **raw data sharing도 없고 label sharing도 없도록 설계**되어 있어 의료 환경과의 정합성이 높다.

두 번째 강점은 **두 개의 서로 다른 의료영상 과제에서 일관된 결과**를 보여줬다는 점이다. 안저영상 binary classification과 흉부 X-ray multi-label classification은 성격이 꽤 다른 과제인데, 양쪽 모두 split learning이 비협업보다 우수했다. 이는 방법의 적용 가능성이 꽤 넓다는 정성적 근거가 된다.

세 번째 강점은 **client 수를 1~50까지 폭넓게 변화시킨 실험 설계**다. 많은 분산학습 논문이 소수 참여자만 다루는 반면, 이 논문은 참여기관 수 증가에 따른 성능 변화를 체계적으로 보여준다. 이를 통해 “협업이 필요한 것은 데이터가 잘게 쪼개질수록”라는 메시지가 더 설득력 있게 전달된다.

### Limitations

첫째, 이 논문은 split learning을 federated learning이나 LS-SGD와 **직접 의료 데이터셋 상에서 정면 비교하지 않는다**. 관련 연구와 장단점을 논의하긴 하지만, 실제 실험은 주로 centralized vs split collaborative vs non-collaborative에 초점이 있다. 따라서 “의료 환경에서 split learning이 federated learning보다 더 낫다”는 강한 결론까지는 이 논문만으로 확정하기 어렵다. 저자들도 future work에서 federated learning, LS-SGD와의 비교를 하겠다고 명시한다.

둘째, privacy 측면에서 이 논문은 강한 직관을 제공하지만, **formal privacy guarantee**를 제시하지는 않는다. 즉 원시 데이터와 라벨이 공유되지 않는다는 것은 중요한 장점이지만, intermediate representation을 통해 어느 정도 정보가 복원될 수 있는지에 대한 정량적 privacy analysis는 본문에서 중심적으로 다뤄지지 않는다. 이는 실사용 관점에서 남는 질문이다. 이 평가는 논문이 실제로 제시한 범위를 기준으로 한 해석이다.

셋째, 논문은 collaborative sequential training의 실효성을 보였지만, **통신 비용, latency, 실제 병원 네트워크 환경에서의 운영 복잡도**를 정밀하게 측정한 것은 아니다. future work에서 efficiency와 privacy enhancement를 더 연구하겠다고 말하는 것으로 보아, 본 연구는 원리 증명(proof-of-concept)에 가깝다.

### Interpretation

이 논문은 split learning을 의료 AI에 도입한 초기 대표 사례로 볼 수 있다. 이후 privacy-preserving medical AI, federated/split/hybrid learning 흐름에서 중요한 연결고리 역할을 하는 연구다. 기술적으로 아주 복잡한 새로운 이론을 제안한 논문은 아니지만, “의료기관 간 협업에서 실제 어떤 distributed setup이 가능하고 유의미한가”를 보여준 점에서 의미가 크다. 특히 모델 분할을 통해 **연산량과 정보 접근권을 구조적으로 분리**했다는 점은 이후 다양한 privacy-aware 협업 학습 연구의 출발점으로 해석할 수 있다.

## 6. Conclusion

이 논문은 의료영상 데이터가 기관별로 분산되어 있고 직접 공유하기 어려운 현실을 배경으로, **split learning**이 의료 분야 협업 딥러닝의 실질적인 대안이 될 수 있음을 실험적으로 보였다. 핵심 결론은 다음과 같다.

* split learning은 의료 분야에 처음 적용된 사례로 제시되었다.
* 원시 데이터와 라벨을 공유하지 않고도 collaborative training이 가능하다.
* DR와 CheXpert 두 과제에서, split learning은 client 수가 늘어나도 성능을 잘 유지했다.
* non-collaborative 학습은 client 수 증가에 따라 급격히 성능이 저하되었다.

실무적으로 이 연구는 병원 간 데이터 반출 없이 모델 협업을 수행하려는 시도에 중요한 근거를 제공한다. 연구적으로는 이후 federated learning, split learning, hybrid privacy-preserving learning을 의료에 적용하는 흐름의 초기 이정표 중 하나다. 다만 privacy 보장 정량화, 다른 distributed methods와의 직접 비교, 실제 임상 운영 환경에서의 평가가 후속 과제로 남아 있다.
