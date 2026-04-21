# Language Models as Few-Shot Learner for Task-Oriented Dialogue Systems

- **저자**: Andrea Madotto, Zihan Liu, Zhaojiang Lin, Pascale Fung
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2008.06239

## 1. 논문 개요

이 논문은 task-oriented dialogue system의 핵심 모듈들인 Natural Language Understanding (NLU), Dialogue State Tracking (DST), Dialogue Policy에 대응하는 ACT prediction, 그리고 Natural Language Generation (NLG)에 대해, 별도의 fine-tuning 없이 language model priming만으로 few-shot learning이 가능한지 평가한다. 저자들은 GPT-2 계열 모델에 소수의 예시를 prefix로 붙여 넣고, 그 뒤의 출력을 곧바로 예측값으로 사용하는 방식을 실험한다.

연구 문제는 분명하다. 기존의 task-oriented dialogue system은 각 모듈을 supervised learning이나 reinforcement learning으로 따로 학습시키는 경우가 많고, 이를 위해 상당한 양의 annotated data가 필요하다. 특히 dialogue policy annotation처럼 전문가 개입이 필요한 경우 데이터 수집 비용이 높다. 따라서 적은 수의 example만으로도 실용적인 성능을 내는 접근이 중요하다.

이 문제의 중요성은 두 가지 측면에서 크다. 첫째, task-oriented dialogue는 실제 음성 비서와 고객 응대 시스템에 직접 연결되는 응용 분야다. 둘째, 기존 few-shot 접근의 다수는 결국 task별 fine-tuning과 task-specific parameter를 요구하므로, 여러 작업을 하나의 모델로 유연하게 처리하기 어렵다. 이 논문은 파라미터 업데이트 없이 하나의 language model이 여러 dialogue task를 처리할 수 있는지 실험적으로 점검한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 단순하면서도 당시로서는 중요한 질문을 던진다. 즉, large language model이 few-shot example을 prompt 형태로 제공받았을 때, task-oriented dialogue의 전통적 모듈형 작업들에서도 실제로 일반화할 수 있는가 하는 것이다. 저자들은 GPT-3 스타일의 in-context learning 관점을 task-oriented dialogue에 직접 적용한다.

기존 접근과의 가장 큰 차별점은 fine-tuning을 하지 않는다는 점이다. TOD-BERT, SC-GPT 같은 기존 방법은 large pre-trained model을 활용하더라도 결국 각 task나 domain에 맞게 parameter를 업데이트해야 한다. 반면 이 논문은 모델의 파라미터를 전혀 바꾸지 않고, 입력 앞부분에 few-shot example들을 붙이는 priming만으로 예측을 수행한다. 이로 인해 이론적으로는 하나의 동일한 모델이 동시에 여러 task를 수행할 수 있다.

또 하나의 중요한 설계 포인트는, 모든 task를 하나의 prompting 방식으로 처리하지 않았다는 점이다. 저자들은 task 성격에 따라 prefix를 세 종류로 나눈다. classification에는 `binary` prefix, slot value 예측에는 `value-based` prefix, text generation에는 `generative` prefix를 사용한다. 이 선택은 실험 결과와도 연결되는데, 저자들은 모든 task를 생성 문제로 통일하려는 초기 시도가 성능이 좋지 않았다고 직접 밝힌다.

## 3. 상세 방법 설명

전체 파이프라인은 복잡한 학습 절차보다 inference 설계에 가깝다. 먼저 각 task에 맞는 few-shot example을 몇 개 고른 뒤, 이를 입력 prefix로 고정하여 language model에 넣는다. 이후 마지막 미완성 패턴 뒤에 이어서 생성되는 토큰을 예측값으로 해석한다. 중요한 점은 이 과정에서 gradient update가 전혀 없다는 것이다.

논문은 task-oriented dialogue의 네 가지 작업을 다음과 같이 정의한다.

NLU에서는 사용자 발화 $X$로부터 slot-value dictionary $M = \{s_1=v_1, \dots, s_n=v_n\}$를 추출하는 slot-filling과, utterance를 intent class $Y \in \{y_1, \dots, y_n\}$로 분류하는 intent classification을 다룬다.

DST에서는 대화의 turn sequence $D = \{X_1^U, X_1^S, \dots, X_t^U\}$가 주어졌을 때 현재 turn의 dialogue state인 $M_t = \{s_1=v_1, \dots, s_n=v_n\}$를 예측한다. 논문 설명에 따르면 일반적인 DST는 이전 state $M_{t-1}$를 활용해 필요한 slot만 갱신하지만, 이 논문은 실험에서 마지막 user utterance만 입력으로 사용하고 예측된 state를 turn을 따라 업데이트한다.

ACT task는 원래 dialogue policy와 관련되지만, 여기서는 이를 단순화하여 dialogue act classification으로 다룬다. 즉, utterance를 입력받아 가능한 speech-act들의 집합으로 분류하는 multi-label classification 문제로 설정한다.

NLG에서는 dialogue act와 slot-value dictionary가 결합된 입력 $I(s_1=v_1, \dots, s_n=v_n)$를 받아 자연어 utterance $X$를 생성한다.

저자들이 설계한 prefix는 세 종류다.

첫째, `binary` prefix는 intent classification과 ACT detection에 사용된다. 아이디어는 multi-class 문제도 각 class에 대해 binary decision으로 바꾸는 것이다. 논문 식 (1)은 다음 구조를 갖는다.

$$
X_1 \rightarrow True \\
X_1^* \rightarrow False \\
\cdots \\
X \rightarrow
$$

여기서 $X_i$는 해당 class의 예시이고, $X_i^*$는 다른 class 혹은 false class의 예시다. 어떤 class가 맞는지 판단하려면 class 수만큼 별도 prefix와 forward pass가 필요하다. 즉, $n$개의 class가 있으면 $n$번 모델을 실행한다.

둘째, `value-based` prefix는 slot-filling과 DST에 사용된다. 특정 slot $s$에 대해 값이 있으면 그 값을, 없으면 `None`을 생성하게 한다. 논문 식 (2)는 다음과 같다.

$$
X_1 \rightarrow s = v_1 \\
X_1^* \rightarrow s = None \\
\cdots \\
X \rightarrow s =
$$

이 방식은 slot마다 반복되므로, slot 개수가 많을수록 forward pass 수가 증가한다. 저자들은 이 구조가 TRADE와 유사하게 slot별 decoding을 수행한다고 설명한다.

셋째, `generative` prefix는 NLG에 사용된다. 입력-출력 쌍을 그대로 몇 개 붙이고, 마지막 입력 뒤의 출력을 생성하게 한다. 논문 식 (3)은 다음과 같다.

$$
X_1 \rightarrow Y_1 \\
\cdots \\
X_k \rightarrow Y_k \\
X \rightarrow
$$

여기서 $X_i$와 $Y_i$는 일반적인 source-target sequence다.

학습 절차라고 부를 만한 별도 optimization은 없다. 대신 실험은 GPT-2 SMALL (117M), LARGE (762M), XL (1.54B)을 사용해 수행되며, 각 task마다 context length 1024 토큰 제한 안에서 들어갈 수 있는 최대 shot 수를 다르게 설정한다. NLU slot-filling은 최대 15-shot, intent는 최대 10-shot, DST는 최대 15-shot, ACT는 최대 15-shot, NLG는 최대 20-shot을 사용한다. 논문은 모든 실험이 단일 NVIDIA 1080Ti GPU에서 수행되었다고 명시한다.

## 4. 실험 및 결과

실험은 task별로 서로 다른 데이터셋과 baseline을 사용한다. NLU의 slot-filling과 intent recognition에는 SNIPS를 사용한다. DST와 ACT에는 MultiWoZ를 사용하며, NLG에는 FewShotWOZ를 사용한다.

먼저 NLU의 slot-filling 결과를 보면, CoNLL F1 기준으로 모델 크기와 shot 수가 커질수록 대체로 성능이 상승한다. 평균 F1은 `gpt2`가 1-shot에서 25.40, 10-shot에서 34.94, 15-shot에서 39.94이고, `gpt2-large`는 35.24, 49.52, 51.44, `gpt2-xl`은 33.94, 53.86, 55.19를 기록했다. 즉, 이 task에서는 GPT-2 XL이 가장 좋은 평균 성능을 보인다. 도메인별로도 SearchCreativeWork와 RateBook 같은 일부 intent/slot 조합에서 큰 폭의 향상이 보인다.

Intent recognition에서도 같은 경향이 더 뚜렷하다. 정확도 기준으로 `gpt2` 10-shot은 36.0%, `gpt2-large` 10-shot은 55.14%, `gpt2-xl` 10-shot은 73.0%다. Macro F1 역시 각각 0.3715, 0.5871, 0.7450이다. 저자들의 주장대로 larger LM이 few-shot learner로 더 강해진다는 경향이 NLU에서는 명확히 드러난다.

ACT detection에서는 조금 다른 양상이 나온다. MultiWoZ에서 F1-score를 측정했을 때, 가장 좋은 모델은 GPT-2 XL이 아니라 GPT-2 LARGE다. `gpt2-large` 10-shot은 Micro F1 83.58, Macro F1 68.68, Accuracy 0.8358을 기록해 표 안에서 최고 수준이다. 반면 `gpt2-xl` 10-shot은 Micro F1 75.41, Macro F1 62.27, Accuracy 0.7541로 오히려 떨어진다. 저자들은 이 현상을 counterintuitive하다고 표현하며, prefix 설계 변경이 더 큰 모델에 도움이 될 수 있는지 추가 연구가 필요하다고 말한다.

DST는 본 논문에서 가장 어려운 task로 나타난다. 평가 지표는 Joint accuracy와 Slot accuracy다. `gpt2-large` 15-shot이 Slot accuracy 83.5로 가장 높지만, Joint accuracy는 3.5에 불과하다. `gpt2-xl`은 Slot accuracy가 80~82 수준이지만 Joint accuracy는 2.0~2.2 수준이다. `gpt2`는 Slot accuracy는 78~80 수준이지만 Joint accuracy는 0.6~0.8에 그친다. 이는 개별 slot는 어느 정도 맞출 수 있어도, 전체 dialogue state를 동시에 정확히 맞히는 데는 priming 방식이 매우 취약하다는 뜻이다. 저자들도 DST에서 기존 baseline과의 격차가 여전히 크다고 분명히 언급한다.

NLG에서는 BLEU와 Slot Error Rate (SLR)를 함께 보고한다. BLEU는 높을수록, SLR은 낮을수록 좋다. priming 기반 GPT-2 모델들은 BLEU 면에서는 shot 수가 늘고 모델이 커질수록 일부 향상을 보이지만, fine-tuning 기반 baseline인 SC-GPT와 비교하면 여전히 낮다. 예를 들어 평균 BLEU는 `gpt2-xl` 20-shot이 15.64, `gpt2-large` 20-shot이 14.18, `gpt2` 20-shot이 11.32인데, SC-GPT는 50-shot fine-tuning으로 28.51이다. 따라서 naturalness 측면에서는 gap이 크다.

반면 SLR에서는 정반대 결과가 나타난다. SC-GPT의 평균 SLR은 5.35로 매우 낮고 우수하지만, priming 기반 GPT-2 모델은 40~70대의 매우 높은 값을 보인다. 예를 들어 `gpt2-large` 10-shot의 평균 SLR은 50.07이고, `gpt2-xl` 10-shot은 42.57이다. 즉, 문장을 생성하더라도 필요한 slot realization을 빠뜨리거나 잘못 표현하는 오류가 매우 많다. 논문에 포함된 예시에서도 `inform(name='super 8...) ->` 같은 입력에 대해 불완전한 생성이 나타난다. 따라서 NLG에서는 표면적인 BLEU 일부 개선과 달리 task-oriented generation의 핵심인 slot fidelity는 충분하지 않다고 보는 것이 타당하다.

비교 기준도 중요하다. 이 논문은 대체로 priming 방식에 매우 적은 shot을 사용하면서도, fine-tuning baseline이 50-shot 또는 500-example을 사용한 경우와 비교한다. 예를 들어 ACT와 DST baseline은 TOD-BERT와 BERT를 10% training data, 즉 500 examples로 fine-tuning한 결과다. NLG baseline인 SC-LSTM, GPT-2, SC-GPT는 모두 50 examples로 fine-tuning한 결과다. 이 점을 고려하면, priming이 적은 예시만으로 일부 task에서 꽤 경쟁력 있는 결과를 보였다는 저자들의 평가는 일정 부분 설득력이 있다. 다만 동일한 shot 수와 동일 입력 조건에서 직접 비교한 것은 아니므로, 절대적인 우열 판단에는 주의가 필요하다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 task-oriented dialogue의 여러 모듈에 대해 in-context few-shot learning의 가능성을 체계적으로 점검했다는 점이다. 단순히 한두 task가 아니라 NLU, DST, ACT, NLG를 모두 포함해 실험했기 때문에, priming 기반 방법이 어디에서 통하고 어디에서 무너지는지를 비교적 선명하게 보여준다.

또 다른 강점은 결과를 과장하지 않는 태도다. 저자들은 NLU, ACT, 일부 NLG에서 “약한 baseline”과 유사하거나 나은 경우가 있다고 말하면서도, DST에서는 격차가 크고 NLG의 slot fidelity에도 문제가 있음을 숨기지 않는다. 특히 larger model이 항상 더 좋지 않았다는 관찰, 모든 task를 generative format으로 바꾸는 시도가 잘 되지 않았다는 보고는 후속 연구 관점에서 가치가 있다.

방법론적으로도 장점이 있다. fine-tuning 없이 하나의 LM을 여러 task에 재사용할 수 있다는 점은 task-specific parameter 관리 비용을 줄일 잠재력이 있다. 또한 binary, value-based, generative라는 세 가지 prompting scheme을 나눠 제시함으로써, task 구조에 맞는 prompting 설계가 중요하다는 메시지를 준다.

한계도 명확하다. 첫째, `binary`와 `value-based` 방식은 class 수나 slot 수만큼 반복 forward가 필요하다. 즉, 계산 비용이 누적되며, 구조적으로 비효율적이다. 저자들도 이를 현재 priming 방식의 계산적 한계로 인정한다.

둘째, GPT-2의 최대 입력 길이 1024 토큰 제한 때문에 shot 수를 충분히 늘릴 수 없다. 논문에서도 대부분의 task에서 15-shot 정도가 한계였다고 설명한다. 이는 수백 example fine-tuning baseline과의 비교를 구조적으로 불리하게 만들 수 있다.

셋째, 실험 설정 자체가 기존 문제 정의를 단순화한 부분이 있다. 예를 들어 DST는 마지막 user utterance만 입력으로 사용하고, ACT는 system utterance만 입력으로 사용한다. 이는 실험을 명확하게 만들지만, 실제 task-oriented dialogue에서 활용되는 richer context를 충분히 반영하지 못한다. 이 점은 논문 텍스트에 직접 나타난 설정 차이이며, 성능 해석 시 반드시 고려해야 한다.

넷째, NLG 결과는 BLEU만 보면 일부 개선이 있으나 SLR이 매우 높아 실제 task completion 품질이 낮다. task-oriented NLG에서 핵심은 slot를 빠짐없이 정확하게 표현하는 것인데, 이 논문 방식은 그 요구를 충족하지 못한다.

비판적으로 해석하면, 이 논문은 “LM priming이 task-oriented dialogue에도 어느 정도 가능하다”는 가능성 증명에는 성공했지만, “실용적인 대체재”를 제시했다고 보기는 어렵다. 특히 DST와 NLG의 중요한 지표를 보면, prompting alone이 구조화된 의미 표현과 state consistency를 안정적으로 다루기 어렵다는 점이 드러난다. 다만 이런 한계를 정직하게 제시했다는 점이 논문의 가치이기도 하다.

## 6. 결론

이 논문은 GPT-2 계열 language model을 few-shot example로 priming하여 task-oriented dialogue의 네 가지 핵심 작업을 수행할 수 있는지 평가했다. 핵심 기여는 fine-tuning 없는 in-context few-shot 설정을 NLU, DST, ACT, NLG 전반에 걸쳐 실험적으로 검증하고, 어떤 task에서는 꽤 가능성이 있으나 어떤 task에서는 아직 큰 한계가 있다는 점을 정리한 데 있다.

결과적으로 NLU와 ACT에서는 비교적 유망한 성능이 나타났고, larger LM이 대체로 더 좋은 few-shot learner라는 경향도 일부 확인되었다. 그러나 DST에서는 joint state tracking 성능이 매우 낮았고, NLG에서도 slot fidelity 문제가 심각했다. 또한 계산 비용과 짧은 context window라는 구조적 제약이 분명히 드러났다.

따라서 이 연구의 실제적 의미는, task-oriented dialogue에서 prompt-based few-shot learning이 완전히 비현실적인 접근은 아니지만, structured prediction과 constrained generation이 중요한 작업에서는 아직 부족하다는 점을 보여준 데 있다. 후속 연구로 dialogue-specific LM, 더 긴 context를 가진 모델, 더 나은 prefix 설계, 또는 adversarial trigger 같은 prompt optimization 기법을 결합하는 방향이 자연스럽게 이어질 수 있다.
