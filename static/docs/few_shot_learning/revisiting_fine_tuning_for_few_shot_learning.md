# Revisiting Fine-Tuning for Few-Shot Learning

- **저자**: Akihiro Nakamura, Tatsuya Harada
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1910.00216

## 1. 논문 개요

이 논문은 few-shot learning에서 흔히 당연하게 받아들여지던 가정, 즉 “novel class 샘플이 매우 적기 때문에 단순 fine-tuning은 쉽게 overfitting된다”는 믿음을 다시 점검한다. 저자들은 오히려 적절한 설정만 갖추면 단순한 fine-tuning이 기존의 대표적 few-shot learning 알고리즘들과 비교해도 매우 강력한 성능을 낼 수 있다고 주장한다.

연구 문제는 명확하다. pretrained network가 base classes에서 학습된 뒤, 매우 적은 수의 support examples만으로 novel classes를 학습해야 할 때, 정말로 복잡한 metric-learning이나 meta-learning이 필수적인가라는 질문이다. 기존 연구들은 overfitting을 피하기 위해 metric-based, meta-learning-based, synthetic data augmentation-based 방법들을 발전시켜 왔지만, 정작 “잘 설계된 naive fine-tuning” 자체는 충분히 강하게 검증되지 않았다고 저자들은 본다.

이 문제가 중요한 이유는 practical setting에서의 단순성과 적용성 때문이다. 복잡한 few-shot 알고리즘은 구현과 학습이 어렵고, 특정 episodic training 설정에 강하게 의존하는 경우가 많다. 반면 fine-tuning은 표준 classification pipeline 위에서 바로 적용 가능하다. 따라서 만약 fine-tuning이 실제로 경쟁력 있는 성능을 낸다면, few-shot learning의 기준선과 평가 관행 자체를 다시 생각해야 한다는 것이 이 논문의 핵심 문제의식이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 복잡한 few-shot 전용 학습 전략 없이도, base classes로 충분히 pretraining된 분류 모델이 novel classes에 꽤 잘 적응할 수 있다는 점이다. 저자들은 fine-tuning 성능이 과소평가되어 왔다고 보고, 이를 체계적으로 재검토한다.

구체적으로는 세 가지 관찰이 핵심이다. 첫째, fine-tuning 시 learning rate를 매우 낮게 두면 재학습 과정이 안정화된다. 둘째, Adam과 같은 adaptive gradient optimizer가 few-shot fine-tuning에서 오히려 더 좋은 test accuracy를 준다. 셋째, base와 novel 사이의 domain shift가 클 때는 classifier만 업데이트하는 것보다 전체 network를 업데이트하는 편이 더 효과적이다.

기존 접근과의 차별점은 모델 구조를 복잡하게 바꾸거나 별도의 meta-training objective를 설계하지 않는다는 데 있다. 이 논문은 새로운 few-shot 알고리즘을 제안한다기보다, standard deep classifier와 fine-tuning recipe만으로도 강력한 baseline을 만들 수 있음을 보인다. 즉, contribution의 성격은 “새로운 이론적 프레임워크”보다는 “재평가와 실증적 통찰”에 가깝다.

## 3. 상세 방법 설명

전체 파이프라인은 비교적 단순하다. 먼저 base classes에서 일반적인 supervised classification 방식으로 feature extractor를 pretrain한다. 이후 few-shot episode마다 novel classes의 support set을 이용해 classifier를 초기화하고, 경우에 따라 classifier만 또는 일부 파라미터만 또는 전체 네트워크를 fine-tuning한다. 마지막으로 query set에 대해 분류 정확도를 측정한다.

저자들은 feature extractor로 ResNet-18/34/50/101/152와 VGG-16을 사용했다. VGG-16의 경우 입력 해상도 변화에 대응하기 위해 마지막 `MaxPool2d`를 `GlobalAveragePool2d`로 교체했다. classifier는 두 종류를 사용했다. 하나는 일반적인 fully connected layer 기반의 simple classifier이고, 다른 하나는 weight imprinting을 사용하는 normalized classifier이다.

normalized classifier의 핵심은 classifier weight와 feature를 모두 정규화한 뒤 inner product로 분류하는 것이다. 각 클래스 weight $w_i \in \mathbb{R}^d$는 $\|w_i\| = 1$이 되도록 normalize하고, feature vector도 정규화된 $\hat{z} \in \mathbb{R}^d$를 사용한다. 그러면 class score는 본질적으로 $w_i^\top \hat{z}$ 형태가 된다. 이 값의 범위는 $[-1, 1]$이므로, softmax에서 정답 확률을 충분히 크게 만들기 위해 scale factor $s$를 곱한다. 즉, 실제 분류 score는 대략 다음처럼 이해할 수 있다.

$$
\text{logit}_i = s \cdot w_i^\top \hat{z}
$$

novel class의 classifier weight 초기화에는 weight imprinting을 사용한다. 1-shot이면 해당 support example의 정규화 feature를 바로 새 class weight로 사용하고, $K$-shot이면 class mean을 다시 normalize하여 초기 weight로 사용한다.

$$
w_{\text{novel}} = \frac{\frac{1}{K}\sum_{j=1}^{K}\hat{z}_j}{\left\|\frac{1}{K}\sum_{j=1}^{K}\hat{z}_j\right\|}
$$

simple classifier의 경우 novel class 초기 weight는 novel feature들에 대해 multi-class linear SVM을 적용해 얻는다.

fine-tuning 단계에서는 세 가지 업데이트 전략을 비교했다. 첫째, 전체 네트워크를 업데이트하는 방식(`All`)이다. 둘째, classifier weight와 batch normalization의 통계/파라미터만 업데이트하는 방식(`BN & FC`)이다. 셋째, classifier weight만 업데이트하는 방식(`FC`)이다. 셋째 방식은 overfitting 방지를 위해 흔히 생각하는 보수적인 fine-tuning 설정이다.

pretraining은 base classes에서 600 epochs 동안 진행했고 optimizer는 Adam, learning rate는 $0.001$이었다. 입력 전처리는 random resized crop, color jitter, random horizontal flip, ImageNet mean/std 정규화로 구성된다. low-resolution task에서는 crop size가 $84 \times 84$, 나머지는 $224 \times 224$이다.

few-shot fine-tuning 시 batch size는 $NK$로 두었다. 즉, $N$-way $K$-shot에서 support set 전체를 한 배치로 사용하는 셈이다. learning rate와 epoch 수는 validation classes를 이용해 결정했다. 성능 평가는 novel classes에서 무작위로 class와 sample을 뽑아 600번 trial을 수행한 평균 accuracy와 95% confidence interval로 보고한다.

중요한 점은, 본문 발췌에서는 학습 loss의 수식을 명시적으로 쓰지 않는다. 분류용 classifier와 softmax 확률을 전제로 설명하고 있으므로 표준적인 classification loss, 즉 cross-entropy를 사용한 것으로 읽히지만, 주어진 텍스트 안에서 손실함수 식 자체가 직접 제시되지는 않았다. 따라서 손실 함수의 정확한 수식은 논문 발췌문만으로는 확인되지 않는다.

## 4. 실험 및 결과

실험은 세 가지 설정에서 수행되었다. 첫째는 고전적으로 많이 쓰이는 low-resolution single-domain task로, `mini-ImageNet`을 $84 \times 84$ 해상도로 사용한다. 둘째는 같은 데이터셋이지만 $224 \times 224$ 해상도를 쓰는 high-resolution single-domain task이다. 셋째는 base classes를 `mini-ImageNet` 전체에서, validation과 novel classes를 `CUB-200-2011`에서 가져오는 cross-domain task이다. 이 마지막 설정은 base와 novel 사이의 domain shift가 훨씬 크다.

low-resolution single-domain 결과를 보면, 1-shot에서는 fine-tuning이 항상 도움이 되는 것은 아니지만 강력한 조합에서는 좋은 성능이 나온다. 가장 좋은 결과는 `VGG-16 + Normalized FC`의 $54.90 \pm 0.66$이다. 이는 MatchingNet의 46.6, ProtoNet의 $49.42 \pm 0.78$, RelationNet의 $50.44 \pm 0.82$보다 높다. 다만 MTL의 $61.2 \pm 1.8$보다는 낮다. 즉, 1-shot low-resolution에서는 SOTA를 넘지는 못하지만, 전통적인 대표 방법들보다 강한 baseline임을 보여준다.

같은 설정의 5-shot에서는 `VGG-16 + Normalized All`이 $74.50 \pm 0.50$을 기록했다. 이는 ProtoNet의 $68.20 \pm 0.66$, RelationNet의 $65.32 \pm 0.70$, Baseline++의 $66.43 \pm 0.63$를 분명히 앞선다. 또한 MTL의 $75.5 \pm 0.8$에 매우 근접한다. 저자들이 말하는 “state-of-the-art에 nearly the same”이라는 평가는 이 결과를 가리킨다.

high-resolution single-domain에서는 fine-tuning의 강점이 더 분명하게 나타난다. 최고 성능은 `ResNet-50 + Normalized All`의 $79.82 \pm 0.49$이다. 이는 Chen et al. (2019)의 Baseline $74.69 \pm 0.64$, Baseline++ $75.90 \pm 0.61$, ProtoNet $74.65 \pm 0.64$보다 높다. 즉, 더 실제적인 고해상도 환경에서도 fine-tuning이 매우 경쟁력 있음을 보여준다.

cross-domain 5-shot에서는 domain shift의 영향으로 전체 성능이 낮아지지만, fine-tuning은 여전히 강력하다. 최고 성능은 역시 `ResNet-50 + Normalized All`의 $74.88 \pm 0.58$이며, Baseline의 $65.57 \pm 0.70$, ProtoNet의 $62.02 \pm 0.70$, RelationNet의 $57.71 \pm 0.73$보다 크게 높다. 특히 기존 방법들이 high-resolution single-domain에서 cross-domain으로 갈 때 10% 이상 성능이 떨어지는 반면, 저자들의 최고 결과는 $79.82$에서 $74.88$로 약 5% 정도만 감소한다. 이는 전체 network fine-tuning이 domain shift를 줄이는 데 효과적이라는 논문의 해석을 뒷받침한다.

세부 분석도 흥미롭다. Figure 2에서는 fine-tuning learning rate가 작을수록 validation accuracy가 더 안정적으로 상승한다. $lr=0.01$이나 $0.001$에서는 진동이 크고 불안정하지만, $lr=0.0001$에서는 보다 매끄럽게 개선된다. 이는 few-shot adaptation에서는 pretraining보다 더 낮은 learning rate가 필요하다는 주장과 연결된다.

Figure 3에서는 optimizer 비교를 수행했다. Adam, Adamax, Adadelta, Adagrad, RMSprop 같은 adaptive gradient optimizer들이 Momentum-SGD나 ASGD보다 높은 accuracy를 보였고, 특히 normalized classifier에서 그 차이가 더 두드러졌다. 일반적으로 adaptive optimizer의 generalization이 약할 수 있다는 기존 지적과 달리, few-shot fine-tuning에서는 오히려 이들이 유리할 수 있음을 시사한다.

Figure 4와 Table 3는 어떤 파라미터를 업데이트할지에 관한 중요한 결론을 준다. cross-domain task에서는 `All`이 `BN & FC`나 `FC`보다 대체로 우세하다. 예를 들어 ResNet-50 normalized classifier의 경우 `All`은 $74.88 \pm 0.58$인데, `BN & FC`는 $70.89 \pm 0.63$, `FC`는 $61.45 \pm 0.68$이다. 즉 domain shift가 클수록 feature extractor 자체까지 적응시켜야 한다는 것이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 few-shot learning 분야에서 과소평가된 baseline을 강하게 복원했다는 점이다. 복잡한 algorithmic novelty 없이도, 잘 설계된 fine-tuning만으로 강한 성능이 나온다는 사실은 매우 실용적이고도 학술적으로 중요하다. 특히 high-resolution과 cross-domain이라는 더 현실적인 설정에서도 기존 대표 방법들을 능가한 점은 단순 baseline 이상의 의미를 가진다.

또 다른 강점은 단순히 “성능이 좋다”에서 멈추지 않고, 왜 좋은지에 관한 실험적 통찰을 제공했다는 점이다. low learning rate의 필요성, adaptive optimizer의 효과, domain shift가 큰 경우 전체 network 업데이트의 중요성 등은 후속 연구나 실제 시스템 구축 모두에 직접 도움이 되는 결론이다.

또한 low-resolution `mini-ImageNet`만으로 few-shot 성능을 논하는 평가 관행에 문제를 제기한 점도 중요하다. 저자들은 같은 모델이라도 해상도와 domain shift에 따라 거동이 크게 달라진다는 점을 보여주며, 단일 benchmark 결과만으로 few-shot 능력을 일반화해서는 안 된다고 시사한다.

한계도 분명하다. 첫째, 이 논문은 새로운 few-shot 알고리즘을 제안하는 논문이라기보다 empirical re-evaluation 성격이 강하다. 따라서 “왜 fine-tuning이 이 상황에서 잘 작동하는가”에 대한 이론적 설명은 제한적이다. 예를 들어 adaptive optimizer가 few-shot setting에서 왜 더 잘 맞는지는 흥미로운 질문으로 남겨 두고, 본문에서도 범위를 넘어선다고 명시한다.

둘째, 결과가 네트워크 구조와 classifier 종류에 따라 꽤 민감하다. 예를 들어 어떤 설정에서는 fine-tuning이 validation accuracy를 개선하지 못해 실제로 `w/o FT`와 같은 결과가 보고되기도 한다. 즉 “fine-tuning은 언제나 유효하다”가 아니라 “적절한 조합에서 매우 강하다”가 더 정확한 해석이다.

셋째, loss function이나 일부 세부 학습 설정은 발췌문 기준으로 엄밀한 수식 수준에서 자세히 설명되지 않는다. 분류 문제이므로 표준 cross-entropy로 이해하는 것이 자연스럽지만, 논문 텍스트 발췌만으로는 그 식이 명시적으로 주어지지 않는다. 따라서 완전한 재현을 위해서는 원문 전체 구현 세부사항을 추가 확인할 필요가 있다.

비판적으로 보면, 비교표에서 각 task별 “최고 성능 조합”을 택해 기존 방법들과 비교하는 방식은 실용적이지만, backbone이나 설정 통제 측면에서는 완전히 동일 조건의 비교라고 보기 어렵다. 논문도 이 부분을 어느 정도 인정하며, 특히 high-resolution과 cross-domain에서는 SOTA 논문들의 해당 결과가 보고되지 않아 직접 비교가 제한된다고 말한다. 따라서 결론은 “fine-tuning이 충분히 강력하다”로 읽는 것이 타당하지, “모든 few-shot 방법보다 본질적으로 우월하다”로 읽는 것은 과하다.

## 6. 결론

이 논문은 few-shot learning에서 naive하게 보이던 fine-tuning이 사실은 매우 강력한 접근일 수 있음을 실험적으로 설득력 있게 보였다. low-resolution `mini-ImageNet`에서는 1-shot에서 여러 대표 기법을 능가하고, 5-shot에서는 당시 강한 방법에 근접했다. 더 나아가 high-resolution single-domain과 cross-domain처럼 더 현실적인 설정에서는 여러 기존 방법보다 더 높은 정확도를 달성했다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, few-shot fine-tuning의 성능을 재평가해 강력한 baseline을 제시했다. 둘째, low learning rate와 adaptive optimizer가 few-shot adaptation에 유리하다는 실험적 근거를 제공했다. 셋째, domain shift가 큰 경우에는 classifier만이 아니라 전체 network를 업데이트해야 성능이 오른다는 점을 보였다.

실제 적용 측면에서 이 연구는 매우 중요하다. 현업이나 응용 연구에서는 구현 복잡도가 낮고 기존 분류 파이프라인과 잘 결합되는 방법이 선호된다. 그런 관점에서 이 논문은 few-shot learning 시스템을 설계할 때 meta-learning이나 metric-learning만을 우선 고려할 것이 아니라, 강한 pretraining과 신중한 fine-tuning recipe를 먼저 점검해야 한다는 메시지를 준다. 향후 연구에서도 새로운 알고리즘 제안뿐 아니라, benchmark 설정과 baseline 설계의 타당성을 함께 따져야 함을 보여주는 논문이라고 평가할 수 있다.
