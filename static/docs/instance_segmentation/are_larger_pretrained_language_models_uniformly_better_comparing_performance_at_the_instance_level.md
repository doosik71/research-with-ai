# Are Larger Pretrained Language Models Uniformly Better? Comparing Performance at the Instance Level

이 논문은 “더 큰 pretrained language model이 평균 정확도는 더 높지만, **각 개별 데이터 포인트(instance)**에서도 항상 더 나은가?”라는 질문을 정면으로 다룬다. 저자들은 이 질문에 답하려면 단순히 한 번 학습한 큰 모델과 작은 모델을 비교해서는 안 된다고 본다. 이유는 **instance-level prediction이 random seed에 매우 민감**하기 때문이다. 그래서 이 논문은 pretraining seed와 finetuning seed에 따른 잡음을 명시적으로 모델링하고, 이를 평균낸 **instance accuracy**라는 개념과 통계적 검정 절차를 제안한다. 그 결과, BERT-large는 평균적으로는 더 좋지만 MNLI, SST-2, QQP 전반에서 **적어도 1–4%의 instance에서는 오히려 BERT-mini보다 더 나쁘다**고 결론내린다. 동시에 저자들은 **finetuning noise가 모델 크기와 함께 증가**하고, instance-level 성능 변화에는 **momentum**이 존재한다고 보고한다. 즉 큰 모델의 성능 향상은 “모든 샘플에서의 균일한 향상”이 아니라, 일부 샘플에서는 좋아지고 일부에서는 오히려 악화되는 비균일한 현상이라는 것이 논문의 핵심 메시지다.  

## 1. Paper Overview

이 논문이 풀고자 하는 문제는 매우 단순해 보이지만 실제로는 꽤 깊다. 우리는 보통 더 큰 언어모델이 더 높은 평균 accuracy를 내면 “더 낫다”고 말한다. 하지만 그 평균 향상이 **모든 instance에서의 일관된 개선**을 뜻하는지, 아니면 어떤 instance에서는 더 좋아지고 다른 일부 instance에서는 더 나빠지는지를 구분하지 않는다. 저자들은 prior work가 이 점에 대해 상반된 시그널을 준다고 정리한다. 어떤 연구는 큰 pretrained 모델이 OOD robustness를 높인다고 보고하지만, 다른 연구는 큰 모델이 rare subgroup에서 더 나쁠 수 있다고 시사한다. 이 논문은 그 모순을 **instance 단위 분석**으로 풀고자 한다.  

이 문제가 중요한 이유는 평균 성능만으로는 모델의 실제 행동을 충분히 설명할 수 없기 때문이다. 만약 더 큰 모델이 특정 유형의 샘플에서 체계적으로 악화된다면, 단순 leaderboard improvement로는 놓칠 수 있는 failure mode가 존재한다는 뜻이다. 특히 NLP에서는 label noise, controversial example, underspecification이 흔하기 때문에, 평균 accuracy의 작은 상승 뒤에 훨씬 큰 instance-level 변동이 숨어 있을 수 있다. 저자들은 바로 이 점 때문에 **instance-level prediction 자체를 별도 분석 대상**으로 삼아야 한다고 주장한다.  

## 2. Core Idea

논문의 핵심 아이디어는 다음 두 가지다.

첫째, 모델 비교의 기본 단위를 **overall accuracy**가 아니라 **instance accuracy**로 바꾸자는 것이다. 저자들은 한 모델 크기 $s$와 특정 instance $i$에 대해, 그 instance를 얼마나 자주 맞히는지를
$$
\mathrm{Acc}_i^s := \mathbb{E}_{P,F}[c_i^s]
$$
로 정의한다. 여기서 $P$는 pretraining seed, $F$는 finetuning seed, $c_i^s \in {0,1}$는 해당 run에서 그 instance를 맞혔는지 여부다. 즉 “이 모델은 몇 % 정확한가?”가 아니라 “**이 instance는 이 크기의 모델이 몇 % 확률로 맞히는가?**”를 보자는 것이다.  

둘째, 단순 seed averaging만으로는 충분하지 않기 때문에, **false discovery를 통제하는 통계적 절차**를 도입한다. 저자들은 larger model이 smaller model보다 나쁜 instance를 “decaying instance”라고 부르고, 이를 추정할 때 random baseline을 함께 사용해 우연히 그렇게 보이는 경우를 분리하려 한다. 또한 이 방법이 classical **Fisher’s exact test + Benjamini-Hochberg(BH)** 보다 더 강한 lower bound를 준다고 주장한다. 핵심 직관은, seed noise 때문에 instance별 비교가 매우 불안정하므로, “큰 모델이 어떤 샘플에서 더 나쁘다”는 결론 자체를 통계적으로 조심스럽게 내려야 한다는 점이다.  

즉 이 논문의 novelty는 새로운 language model이 아니라, **모델 크기 비교를 instance-level stochastic quantity로 재정의하고**, 그 위에서 decaying instance를 검출하는 **통계적으로 엄밀한 분석 프레임워크**를 제안했다는 데 있다.

## 3. Detailed Method Explanation

### 3.1 데이터와 모델 설정

저자들은 BERT 계열의 다섯 가지 크기를 직접 pretrain하고 finetune해 비교한다.

* **mini**: L4 / H256
* **small**: L4 / H512
* **medium**: L8 / H512
* **base**: L12 / H768
* **large**: L24 / H1024

대상 태스크는 다음 세 가지다.

* **QQP**
* **MNLI**
* **SST-2**

그리고 seed noise를 보기 위해 각 architecture마다 **pretraining seed 10개**, 각 pretrained model마다 **finetuning 5회**를 수행한다. 즉 각 크기마다 총 **50개 모델**을 만든다. 저자들은 이 정도 반복이 있어야 pretraining/finetuning randomness를 분리해서 볼 수 있다고 본다. 계산 비용을 줄이기 위해 pretraining context size를 512에서 128로 줄이고 steps는 1M에서 2M으로 늘렸지만, appendix에서 이것이 qualitative conclusion을 바꾸지 않는다고 확인한다.  

### 3.2 왜 naïve comparison이 실패하는가

저자들이 가장 먼저 보여주는 것은, 단순히 “큰 모델 한 번 vs 작은 모델 한 번”을 비교하면 instance-level noise에 압도된다는 점이다. 예를 들어 MNLI in-domain dev에서 같은 BERT-base끼리도 **finetuning seed만 바꿔도 약 8%의 instance prediction이 달라진다**. 반면 overall accuracy의 표준편차는 **0.2~0.3% 수준**이다. 즉 전체 평균은 안정적인데, 개별 샘플의 정오답은 매우 흔들린다. 이 차이를 표 1이 잘 보여준다. 모델 크기가 커질수록 finetuning seed 차이에 의한 instance-level prediction difference는 대략 **7.2% → 8.6%**로 증가하고, pretraining seed 차이에 의한 difference는 약 **10%** 수준이다. 그런데 overall accuracy의 표준편차는 이보다 약 40배 작다.  

이 때문에 저자들은 “한 번 학습한 large가 small보다 어떤 instance에서 더 나쁘다”는 관찰은 대부분 noise일 수 있다고 본다. 실제로 BERT-large와 BERT-base를 naïvely 비교하면 large가 4.5% instance에서 worse, 7% instance에서 better로 보이지만, 같은 base끼리도 8%가 다르니 이 수치를 곧이곧대로 해석하면 안 된다는 것이다.

### 3.3 Instance accuracy와 instance difference

그래서 논문은 각 instance에 대해 seed 평균 정확도인 instance accuracy를 정의한다. 그리고 두 모델 크기 $s_1, s_2$ 사이의 차이를
$$
{}^{s_1}_{s_2}\Delta \mathrm{Acc}_i := \mathrm{Acc}_i^{s_2} - \mathrm{Acc}_i^{s_1}
$$
로 본다. 이 값이 음수이면, larger model이 smaller model보다 그 instance를 덜 자주 맞힌다는 뜻이다. 이런 negative tail이 바로 **decaying instances**의 신호다.

Figure 1과 Figure 2 설명에서 저자들은 blue histogram(실제 비교)과 red histogram(random baseline)을 비교한다. 만약 큰 모델이 정말 uniformly better라면 negative tail은 거의 baseline 수준이어야 한다. 하지만 실제로는 blue histogram의 왼쪽 꼬리가 더 두껍고, 따라서 **작은 모델이 더 나은 instance가 실제로 존재한다**는 증거가 나온다고 해석한다.

### 3.4 Random baseline과 false discovery control

하지만 negative tail이 보여도, 그 일부는 여전히 추정 오차 때문에 생긴 가짜일 수 있다. 그래서 저자들은 **random baseline**을 사용해 false discovery fraction을 추정하고, Section 4에서 보다 formal한 lower bound를 제시한다. 논문은 이 방법이 classical BH + Fisher exact test보다 항상 더 나은 lower bound를 준다고 말한다. 실제 표 4 snippet에서도 모든 크기 쌍과 pretrained model 수 설정에서 **저자들의 방법이 BH보다 더 큰 decaying fraction lower bound**를 제공한다고 설명한다.

이 설계는 이 논문의 중요한 공헌이다. 단순 empirical observation을 “interesting pattern”으로 끝내지 않고, **통계적으로 방어 가능한 하한(lower bound)**으로 바꾼다.

### 3.5 Decaying fraction 추정

논문의 대표 결과는 MNLI in-domain dev에서 **mini vs large** 비교다. Figure 4 설명에 따르면 cumulative distribution function 기반 분석에서, 두 곡선의 최대 차이 **6%**가 true decaying fraction의 lower bound가 된다. 본문에서는 dependency issue를 고려한 더 보수적인 rigorous target으로는 **약 4%**를 강조한다. 즉 표현 방식에 따라 4% 또는 6%가 언급되지만, 핵심은 **non-zero and substantial** 하다는 점이다. 그리고 이 값은 overall accuracy improvement가 약 **10%**인 것과 함께 해석되어야 한다. 평균 성능은 개선되지만, 동시에 의미 있는 비율의 instance는 더 나빠진다는 뜻이다.  

또한 appendix와 본문 요약에 따르면 이런 decaying instance는 MNLI뿐 아니라 SST-2와 QQP, 그리고 다른 model-size pair에서도 일관되게 발견된다. 일반적으로 **크기 차이가 클수록 decaying fraction도 커지는 경향**이 있다고 보고한다.  

### 3.6 Decaying instances의 해석

저자들은 decaying instances가 전부 mislabeled data일 수 있다는 가설도 검토한다. Section 4.2 설명에 따르면, 이들은 실제로 **controversial or wrong labels**를 더 많이 포함하지만, **정상적으로 올바른 라벨을 가진 instance도 포함**한다. 즉 “큰 모델이 나빠지는 샘플은 다 레이블 오류다”라고 치부할 수 없다는 것이다. 이 점이 중요하다. 큰 모델의 비균일한 향상은 단순 annotation artifact를 넘어, 실제 학습/표현 특성의 차이를 반영할 가능성이 있다.  

### 3.7 Momentum과 variance decomposition

Section 5와 6의 다른 흥미로운 결과는 다음 두 가지다.

첫째, **instance-level accuracy has momentum**이다. 즉 mini → medium에서 좋아진 instance는 medium → large에서도 좋아질 가능성이 높다. 성능 향상이 무작위가 아니라, 특정 instance 방향으로 일관된 흐름을 가진다는 뜻이다.

둘째, variance decomposition 결과, **larger models have larger finetuning variance**가 관찰된다. Table 1과 appendix 설명을 합치면, finetuning seed에 따른 prediction variance가 모델 크기와 함께 증가하며, 저자들은 이것이 instance-level 비교를 더욱 어렵게 만든다고 해석한다. 반면 pretraining variance도 존재하지만, conclusion급 메시지는 “큰 모델일수록 finetuning randomness의 영향이 커진다”에 더 가깝다.  

## 4. Experiments and Findings

이 논문의 실험은 새로운 모델이 아니라 **새로운 분석 관점**을 검증하는 데 초점이 있다. 그럼에도 결과는 꽤 강하다.

가장 먼저, seed noise는 overall accuracy보다 instance-level에서 훨씬 크다. 같은 BERT-base끼리도 finetuning seed만 달라지면 약 **8%**의 instance prediction이 바뀌지만, overall accuracy 표준편차는 약 **0.1~0.3%** 수준이다. 이것만으로도 instance-level 분석을 평균 accuracy와 분리해서 봐야 할 이유가 충분하다.

둘째, BERT-large는 BERT-mini보다 평균적으로 더 좋지만, **MNLI in-domain dev에서 적어도 약 4%의 instance에서는 더 나쁘다**는 rigorous lower bound가 나온다. 더 넓게는 MNLI, SST-2, QQP 전반에서 **1–4% 수준의 decaying instances**가 보고된다. 이는 overall accuracy gain이 **2–10%**라는 점과 같이 봐야 한다. 즉 평균 개선의 상당 부분과 동시에, 무시할 수 없는 일부 샘플에서는 성능이 반대로 움직인다.  

셋째, BH + Fisher exact test와 비교했을 때, 저자들의 random-baseline 기반 방법이 **더 강한 decaying fraction lower bound**를 준다. 이는 단순히 현상을 보였다는 데 그치지 않고, 통계적 도구 자체도 기존 baseline보다 낫다는 주장이다.

넷째, instance-level 성능 변화에는 **momentum**이 있다. 작은 모델에서 중간 모델로 갈 때 좋아지는 instance는 중간 모델에서 큰 모델로 갈 때도 좋아질 경향이 있다. 성능 향상이 completely random한 것이 아니라, 특정 example difficulty/structure와 연관된 일관성을 가진다는 해석이 가능하다.

다섯째, variance 분석에서는 **finetuning noise가 모델 크기와 함께 증가**한다. 이는 practical implication이 크다. 큰 모델을 비교할수록 더 많은 seed averaging이 필요하고, 단일 checkpoint 수준의 instance analysis는 특히 위험해진다.  

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **문제를 정확히 다시 정의했다는 점**이다. 대부분의 scaling 논의는 average accuracy나 average loss에 집중하지만, 이 논문은 “정말 larger model이 uniformly better인가?”라는 질문을 instance 단위로 바꾼다. 이 질문 전환만으로도 상당히 가치가 있다.

두 번째 강점은 **통계적 엄밀성**이다. 단순히 여러 seed 평균을 보고 “왼쪽 꼬리가 있네”라고 말하는 데서 멈추지 않고, random baseline, false discovery, lower bound, BH 대비 비교까지 제시한다. 이 덕분에 관찰이 anecdotal하지 않고, 재현 가능한 분석 절차로 제시된다.  

세 번째 강점은 **실질적 메시지**가 분명하다는 점이다. 큰 pretrained language model의 평균 성능 향상은 사실이지만, 그것이 모든 sample에서의 균일한 개선을 의미하지는 않는다. 이 결론은 robustness, fairness, subgroup analysis, data curation 같은 후속 논의와 직접 연결된다.

한계도 있다. 첫째, 분석 대상이 **BERT family와 세 개의 분류 태스크(QQP, MNLI, SST-2)**에 집중되어 있다. 따라서 autoregressive LM이나 현대의 instruction-tuned LLM, generative evaluation setting으로 바로 일반화하기는 어렵다. 이는 논문 시점의 자연스러운 한계다.

둘째, 저자들도 인정하듯 seed dependence 때문에 rigorous lower bound를 만들기 위해 target quantity를 약간 수정해야 했다. 즉 이 문제는 통계적으로 까다롭고, 제안한 하한도 여전히 “보수적 추정”에 가깝다. 실제 decaying fraction은 더 클 수 있어도, 그 정확한 값은 완전히 닫혀 있지 않다.  

셋째, decaying instances가 왜 생기는지에 대한 메커니즘적 설명은 제한적이다. 레이블 논쟁성, 잘못된 레이블, seed variance 증가는 보여주지만, representation geometry나 optimization dynamics 수준의 원인 분석까지는 나아가지 않는다. 대신 그 부분은 future work 여지로 남는다.

비판적으로 해석하면, 이 논문의 진짜 기여는 “큰 모델도 일부 샘플에서 나빠진다”라는 사실 자체보다, **모델 비교의 단위를 평균에서 instance distribution으로 옮긴 것**에 있다. 이 관점은 이후 scaling law 논의, subgroup robustness, example-level auditing에 매우 중요한 시사점을 준다. 특히 저자들이 “모델 가중치뿐 아니라 **모델 predictions도 함께 공개/보존하자**”고 권하는 대목은, 재현성과 감사 가능성 측면에서 상당히 앞선 제안이다.

## 6. Conclusion

이 논문은 larger pretrained language model이 평균적으로는 더 정확하지만, **모든 instance에서 uniformly better하지는 않다**는 점을 통계적으로 보여준다. 이를 위해 저자들은 pretraining/finetuning seed randomness를 고려한 **instance accuracy** 개념을 도입하고, random baseline과 lower-bound 절차로 **decaying instances**를 검출한다. 결과적으로 BERT-large는 BERT-mini보다 평균 accuracy는 높지만, MNLI·SST-2·QQP 전반에서 **적어도 1–4%의 instance에서 더 나쁘며**, MNLI에서는 약 **4% 이상**의 엄밀한 하한이 제시된다. 또한 **finetuning noise는 모델 크기와 함께 증가**하고, instance-level 변화에는 **momentum**이 존재한다.  

실무적 함의는 분명하다. 모델 스케일링을 평가할 때 평균 성능만 보면 중요한 실패 패턴을 놓칠 수 있다. 따라서 앞으로는 더 큰 모델이 “얼마나 더 정확한가”뿐 아니라, **어떤 instance에서 좋아지고 어떤 instance에서 나빠지는가**까지 함께 보아야 한다. 이 논문은 바로 그 분석 프레임을 제시한 작업이다.
