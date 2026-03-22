# DropLoss for Long-Tail Instance Segmentation

이 논문은 **long-tail instance segmentation**에서 기존 방법들이 놓치고 있던 핵심 불균형 원인을 짚는다. 기존 연구, 특히 EQL(Equalization Loss) 계열은 희귀 클래스가 **잘못된 foreground 분류** 때문에 억눌린다고 보고 이를 완화하려 했다. 그러나 이 논문은 실제로는 그보다 훨씬 더 많은 억제 신호가 **정답 background prediction**에서 나온다고 주장한다. 즉, background proposal이 들어오면 모든 foreground 클래스 점수를 동시에 낮추는 손실이 걸리는데, rare/common 클래스는 원래 등장 빈도가 낮아서 이를 상쇄할 “encouraging gradient”가 부족하다. 그 결과 모델은 점점 frequent class 쪽으로 편향된다. 저자들은 이 통찰을 바탕으로, 배경에서 오는 억제 손실을 **배치 통계(batch statistics)에 따라 확률적으로 제거하는 DropLoss**를 제안한다. 논문은 이 손실이 rare, common, frequent 전 구간에서 더 나은 균형을 만들며, LVIS에서 state-of-the-art 결과를 낸다고 보고한다.  

## 1. Paper Overview

이 논문의 문제 설정은 매우 현실적이다. 실제 object detection과 instance segmentation 응용에서는 클래스 분포가 균등하지 않고, 대다수 클래스가 소수의 학습 샘플만 가진 **long-tailed distribution**을 이룬다. 이런 데이터에서 모델을 그대로 학습시키면 자연스럽게 frequent class 쪽으로 편향되고, rare class는 과소학습되거나 과적합된다. 이 문제는 일반 분류보다 instance segmentation에서 더 복잡하다. 왜냐하면 region proposal, box regression, mask regression, classification 등 **여러 손실이 동시에 작동**하기 때문이다. 저자들은 바로 이 다중 목적 최적화 환경에서 long-tail 편향이 어떻게 생기는지를 분석한다.

특히 이 논문은 기존 long-tail instance segmentation의 대표 방법인 EQL이 rare class 억제의 주된 원인을 **incorrect foreground prediction**으로 봤다는 점을 비판적으로 재검토한다. 저자들의 분석에 따르면, 실제로 rare/common 클래스에 훨씬 큰 영향을 주는 것은 **background proposal에 대한 올바른 background classification**이다. background로 라벨된 proposal은 모든 foreground 클래스 점수를 낮추는 손실을 발생시키는데, 희귀 클래스는 등장 횟수 자체가 적기 때문에 이런 억제를 회복할 기회가 상대적으로 적다. 이 때문에 평균적으로 모델은 rare/common보다 frequent를 더 선호하게 된다.  

이 문제가 중요한 이유는 단순히 rare AP를 조금 올리는 수준이 아니라, **rare와 frequent 사이의 성능 trade-off 없이** long-tail 구조 전체를 더 건강하게 학습하는 것이 목표이기 때문이다. 논문은 기존 배경 재가중 방식들이 rare를 살리면 frequent가 떨어지는 식의 뚜렷한 Pareto trade-off를 보인다고 지적하고, DropLoss는 이 균형을 더 잘 맞추는 방향으로 설계되었다고 주장한다.

## 2. Core Idea

이 논문의 핵심 아이디어는 다음과 같다.

**희귀 클래스는 잘못된 foreground prediction보다, 훨씬 더 자주 등장하는 correct background prediction에 의해 더 강하게 억눌린다. 따라서 long-tail instance segmentation의 분류 손실은 foreground 오류보다 background 억제 쪽을 우선적으로 보정해야 한다.**  

논문 Figure 1 분석은 이 핵심 직관을 잘 보여준다. LVIS에서 rare 클래스의 경우 전체 discouraging gradient 중 **50–70%**가 background prediction에서 오고, frequent 클래스는 그 비율이 **30–40%** 수준이다. 또 background bounding box에 대해 학습이 진행될수록 rare 클래스의 평균 foreground score가 frequent보다 훨씬 더 강하게 눌리는 현상도 관찰된다. 즉, background 학습은 모든 foreground 클래스를 똑같이 누르지만, rare/common은 원래 positive signal이 적기 때문에 **실질적으로 더 큰 손해**를 본다.

이 통찰을 기반으로 DropLoss는 background proposal에서 발생하는 rare/common 클래스 억제 손실을 **확률적으로 제거(drop)**한다. 중요한 점은 이 확률이 고정이 아니라, **현재 배치에 rare와 frequent가 어떤 비율로 들어왔는지에 따라 적응적으로 정해진다**는 것이다. 즉 DropLoss는 단순 class-frequency reweighting이 아니라, 배치 안의 class composition을 반영하는 **adaptive stochastic loss**다. 논문은 이 방식이 rare/common을 보호하면서도 frequent 성능을 불필요하게 희생하지 않는다고 주장한다.  

## 3. Detailed Method Explanation

### 3.1 기본 학습 프레임워크

논문은 기본 인스턴스 분할 아키텍처로 **Mask R-CNN**을 사용한다. 즉 제안의 본질은 새로운 detector/segmenter가 아니라, **classification loss 재설계**에 있다. 이 점은 중요하다. DropLoss는 backbone이나 box/mask branch를 바꾸는 방법이 아니라, 기존 two-stage instance segmentation framework에 꽂아 넣을 수 있는 손실 함수 개선으로 이해하는 것이 맞다.

### 3.2 EQL의 출발점과 한계

논문은 먼저 EQL을 복기한다. EQL은 sigmoid cross-entropy 기반 분류에서, rare 클래스가 **incorrect foreground prediction** 때문에 받는 discouraging gradient를 줄이기 위해 일부 손실을 제거한다. 여기서 foreground proposal이면 정답 클래스만 $y_j=1$이고, background proposal이면 모든 클래스에 대해 $y_j=0$이 된다. EQL은 주로 foreground 오분류가 rare class를 억누른다고 보고, 그쪽의 손실만 선택적으로 약화한다.

하지만 이 논문은 instance segmentation에서는 background proposal이 훨씬 많고, 따라서 **background classification loss가 전체 분류 동역학을 사실상 지배한다**고 본다. 특히 background proposal은 모든 foreground 클래스 점수를 동시에 낮추는 방향으로 작동하기 때문에, rare/common 클래스가 frequent보다 더 크게 눌릴 수 있다. 이 점이 일반 image classification과 다른 instance segmentation 특수성이라고 논문은 강조한다.

### 3.3 Background Equalization Loss(BEQL)라는 중간 단계

저자들은 먼저 background 쪽까지 EQL 철학을 확장한 **BEQL** 관점을 제시한다. 논문 설명에 따르면, 이 baseline은 rare/common과 frequent에 대해 background 손실의 가중치를 다르게 두어 불균형을 줄인다. 하지만 이 방법은 **로그 밑(base) 같은 하이퍼파라미터 선택에 민감**하고, rare/common 성능을 올리면 frequent가 떨어지는 식의 **명확한 trade-off**를 보인다. Figure 2가 바로 이 현상을 시각화한다. 작은 base일수록 background 효과를 더 많이 줄이지만, 그만큼 frequent 쪽 희생이 커질 수 있다.

이 분석은 DropLoss의 필요성을 뒷받침한다. 즉, 단순 deterministic reweighting만으로는 적절한 균형점을 찾기 어렵고, rare/common과 frequent를 동시에 잘 챙기려면 더 유연한 방식이 필요하다는 것이다.

### 3.4 DropLoss의 핵심 원리

DropLoss는 background prediction에서 rare/common 클래스에 대한 일부 손실 항을 **Bernoulli 변수로 샘플링해 제거**한다. 논문 초록과 서론의 설명에 따르면, 이 Bernoulli 파라미터는 **batch statistics**, 즉 현재 배치에서 rare와 frequent 클래스가 어떤 비율로 들어왔는지를 바탕으로 정해진다. 따라서 DropLoss는 고정 reweighting이 아니라, 현재 학습 상황에 맞춰 확률적으로 손실을 덜어 주는 방식이다.  

직관적으로 보면 다음과 같다.

* 배치에 rare/common이 적게 들어오면, background 억제가 상대적으로 더 치명적이다.
* 이 경우 DropLoss는 rare/common에 대한 background 손실을 더 적극적으로 제거한다.
* 반대로 frequent 중심 배치에서는 과도한 보호가 필요 없으므로 drop 비율이 달라진다.

즉 DropLoss는 **배치 수준 class balance와 cost-sensitive learning을 결합한 형태**라고 볼 수 있다. 논문도 이를 자신의 기여 중 하나로 명시한다.

### 3.5 왜 stochastic dropping이 중요한가

논문이 deterministic BEQL 대신 stochastic DropLoss를 택한 이유는, rare/common과 frequent 사이 성능 균형을 더 부드럽게 조절하기 위해서다. deterministic reweighting은 종종 특정 구간에서 성능 trade-off를 강하게 만들지만, stochastic dropping은 손실을 완전히 고정된 방식으로 줄이지 않고 **확률적으로 탐색 공간을 남겨 둔다**. 저자들은 이를 통해 rare/common의 억제를 막으면서도 frequent 성능을 더 잘 보존할 수 있다고 해석한다.

### 3.6 수식 해석

첨부된 HTML 조각에서 EQL 쪽 수식 일부는 확인할 수 있다. 예를 들어 EQL 분류 손실은

$$
\mathcal{L}_{\mathrm{EQL}}=-\sum*{j=1}^{C} w_j \log(\hat{p}_j)
$$

형태이며, background proposal 여부와 저빈도 클래스 여부에 따라 가중치 $w_j$가 달라진다. 이 식은 foreground/background에 대해 어떤 클래스 손실을 줄일지 선택하는 구조를 가진다. 다만 DropLoss의 최종 수식 전체는 현재 확보된 조각만으로 완전하게 재구성되지 않으므로, 여기서는 논문이 분명히 밝히는 핵심만 정리하는 것이 정확하다. **DropLoss는 background classification에서 rare/common 클래스 손실을 batch-statistics 기반 Bernoulli 샘플링으로 제거하는 adaptive stochastic loss**다. exact closed-form은 원문 PDF나 코드와 함께 보는 편이 더 안전하다.

## 4. Experiments and Findings

논문의 실험은 **LVIS**를 중심으로 이루어진다. LVIS는 클래스 분포가 극도로 long-tail한 대표 benchmark이며, rare(1–10 images), common(11–100), frequent(>100)로 카테고리 그룹을 나누어 분석하는 것이 자연스럽다. 이 논문은 바로 이 구분을 따라 rare/common/frequent 전 구간 성능을 비교한다.

가장 먼저 강조되는 실험 결과는 **배경 억제의 비중**이다. Figure 1(a)에 따르면 rare 클래스의 discouraging gradient 중 **50–70%**가 background prediction에서 오며, frequent는 **30–40%** 수준이다. 이는 기존 EQL이 주목했던 incorrect foreground prediction보다 background가 실제로 더 중요한 억제 원인임을 보여준다. 또 Figure 1(b)는 background bounding box에서 rare class prediction score가 학습 후반에 frequent보다 훨씬 더 낮아진다는 점을 보여 주며, 모델이 frequent 쪽으로 치우친다는 주장을 지지한다.

다음으로, BEQL 계열 baseline 실험은 **rare-common 대 frequent trade-off**를 보여준다. Figure 2 설명에 따르면, background equalization을 강하게 할수록 rare 쪽은 좋아지지만 frequent는 떨어지는 Pareto trade-off가 나타난다. 이는 background imbalance를 완화하는 것 자체는 맞는 방향이지만, 단순 deterministic reweighting만으로는 전체 성능 균형을 잘 맞추기 어렵다는 점을 뜻한다.

DropLoss의 핵심 실험적 메시지는, 이 stochastic adaptive strategy가 **rare, common, frequent 전체에서 더 좋은 균형을 만들고**, 결국 LVIS에서 **state-of-the-art mAP**를 달성했다는 것이다. 초록은 “rare, common, and frequent categories on the LVIS dataset” 전 구간에서 SOTA라고 직접 명시한다. 다만 현재 확보된 HTML 조각에는 최종 표의 숫자 전체가 완전히 드러나지 않으므로, 구체적인 mAP 수치를 임의로 적는 대신 논문이 명시적으로 주장하는 범위까지만 정리하는 것이 정확하다.  

실험이 실제로 보여주는 바는 다음과 같다.

* long-tail instance segmentation에서 rare/common 억제의 핵심 원인은 background classification이다.
* background 손실을 줄이는 것은 효과가 있지만, deterministic 방식은 trade-off가 심하다.
* batch-aware stochastic dropping은 이 trade-off를 완화하며 더 나은 Pareto frontier를 만든다.
* 결과적으로 DropLoss는 rare/common/frequent 전체를 더 균형 있게 개선한다.  

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **문제 원인 진단이 정확하다**는 점이다. 기존 방법이 long-tail instance segmentation의 rare-class suppression을 foreground misclassification 중심으로 해석한 반면, 이 논문은 훨씬 더 빈번한 **background discouraging gradients**가 핵심이라는 점을 수치로 보여 준다. 이 문제 재정의 자체가 논문의 가장 큰 기여 중 하나다.  

두 번째 강점은 **방법이 단순하면서도 plug-in 가능**하다는 것이다. DropLoss는 새로운 backbone이나 detector를 요구하지 않고, Mask R-CNN 같은 기존 프레임워크 위에 손실 함수 수준에서 추가된다. 즉 구조를 크게 바꾸지 않으면서 long-tail 편향을 완화할 수 있다는 점에서 실용적이다.

세 번째 강점은 **rare와 frequent 사이의 trade-off를 정면으로 다룬다**는 점이다. 많은 long-tail 방법은 rare 성능을 올리는 대신 frequent를 희생하는데, 이 논문은 그 Pareto frontier 자체를 실험 대상으로 삼고, DropLoss가 이를 더 좋게 만든다고 설명한다. 이 관점은 단순 rare AP 상승보다 더 설득력 있다.

한계도 있다. 첫째, 현재 확인 가능한 자료 기준으로 실험 수치표와 DropLoss의 최종 수식 전개가 완전하게 복원되지는 않는다. 따라서 구현 수준의 exact formula나 모든 ablation 숫자를 재현하려면 원문 PDF나 코드 저장소를 병행하는 편이 더 안전하다. 이는 이 답변의 정보 제약이기도 하다.  

둘째, 방법의 직관은 강하지만 기본적으로 **batch statistics quality**에 의존한다. 즉 rare/frequent 비율을 어떻게 추정하고 Bernoulli 파라미터를 정하느냐가 중요하며, 배치 구성이나 샘플링 전략에 따라 효과가 달라질 가능성이 있다. 논문은 이를 장점으로 설명하지만, 실무에서는 함께 점검해야 할 부분이다.

비판적으로 해석하면, 이 논문의 진짜 기여는 “새 loss 하나”보다도, **instance segmentation의 long-tail 학습에서 background가 얼마나 지배적인가를 명확히 보여준 것**에 있다. 많은 long-tail learning 연구가 분류나 foreground 오분류에 집중해 왔지만, detection/instance segmentation에서는 background 손실이 훨씬 더 구조적인 역할을 한다는 점을 이 논문이 선명하게 드러낸다.

## 6. Conclusion

이 논문은 long-tail instance segmentation에서 rare/common 클래스가 억눌리는 주된 원인이 **incorrect foreground prediction**이 아니라 **correct background prediction**이라는 점을 밝히고, 이를 완화하기 위한 **DropLoss**를 제안한다. DropLoss는 background proposal에서 rare/common 클래스에 대한 손실을 **배치 통계 기반 Bernoulli 샘플링**으로 확률적으로 제거하여, 희귀 클래스가 과도하게 억제되는 것을 막는다. 저자들은 이 방식이 rare, common, frequent 전체에서 더 좋은 균형을 만들고, LVIS에서 state-of-the-art 성능을 달성한다고 보고한다. 결국 이 논문은 long-tail instance segmentation을 더 잘 푸는 핵심이 rare 클래스에 더 큰 가중치를 주는 것만이 아니라, **background가 희귀 클래스를 어떻게 체계적으로 눌러 왔는지 이해하고 이를 직접 보정하는 것**임을 보여준다.  
