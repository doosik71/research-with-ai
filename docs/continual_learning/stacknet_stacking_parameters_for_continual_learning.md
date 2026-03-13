# StackNet: Stacking Parameters for Continual learning

## 1. 논문 메타데이터

- **제목**: StackNet: Stacking Parameters for Continual learning
- **저자**: Jangho Kim, Jeesoo Kim, Nojun Kwak
- **발표 연도**: 2020
- **arXiv ID**: 1809.02441
- **DOI**: 10.48550/arXiv.1809.02441
- **출처**: <https://arxiv.org/abs/1809.02441>
- **비고**: arXiv 메타데이터 제목은 `Stacking Parameters`이며,
  PDF 첫 페이지에는 `Stacking feature maps`로 표기되어 있다.

## 2. 연구 배경 및 동기

Continual Learning은 과거 데이터를 다시 사용하지 못하는 조건에서
새로운 작업을 순차적으로 학습해야 하므로 catastrophic forgetting
문제가 핵심 병목이 된다. 기존 접근은 크게 세 가지로 나뉜다.

1. **Regularization-based methods**: 기존 파라미터 변화량을 제한해 망각을 줄인다.
2. **Replay-based methods**: 과거 샘플이나 생성 샘플을 다시 사용한다.
3. **Dynamic-architecture methods**: 새 작업마다 네트워크를 확장한다.

이 논문은 특히 `PackNet`과 같은 pruning 기반 접근의 장단점을
강하게 의식한다. PackNet은 작업별로 사용할 weight subset을 분리해
성능 저하를 줄이지만, pruning과 fine-tuning 절차가 추가로 필요하고
입력이 어느 task에서 왔는지에 대한 prior knowledge가 사실상 필요하다.

저자들은 이 문제를 더 직접적으로 풀기 위해 다음 두 요소를 결합한다.

1. **StackNet**: 기존에 학습된 파라미터를 유지한 채 새 task용 파라미터를 순차적으로 덧붙이는 본체 네트워크
2. **Index module**: 입력이 어느 task에 속하는지 추정해 어떤 파라미터 구간을 활성화할지 결정하는 모듈

핵심 동기는 "하나의 큰 네트워크를 task별로 완전히 복제하지
않으면서도, 이전 task 성능 저하 없이 새 task를 받아들일 수 있는
구조"를 만드는 데 있다.

## 3. 핵심 기여

이 논문의 핵심 기여는 다음과 같다.

1. **Task-specific parameter stacking**: convolutional layer의 filter를
   task별 구간으로 나누고, 새 task가 들어올 때 필요한 분량만
   추가 학습한다.
2. **Parameter sharing with expansion**: 이전 task의 파라미터를 이후
   task가 초기값 또는 공유 자원처럼 활용하면서도, 이전 task에
   사용된 부분은 고정해 성능 저하를 막는다.
3. **Index module 제안**: 입력 sample의 task origin을 추정해 적절한
   filter subset과 label space를 선택한다.
4. **Label-expandable continual learning**: task가 늘어날수록 class
   space가 커지는 multi-task continual learning 설정을 지원한다.
5. **실용적 장점**: pruning이나 추가 fine-tuning 없이도 PackNet과 경쟁력 있는 성능을 보인다.

## 4. 방법론 요약

## 4.1 전체 구조

전체 프레임워크는 `Index module + StackNet`으로 구성된다.

```text
입력 x
  -> Index module
  -> task index J 추정
  -> StackNet에서 task J에 해당하는 filter 구간 활성화
  -> task J의 classifier로 예측
```

Index module은 "이 데이터가 어디서 왔는가"를 추정하고, StackNet은
"이 데이터가 무엇인가"를 분류한다. 저자들의 설계 철학은 task
routing과 representation learning을 분리하는 것이다.

## 4.2 StackNet

StackNet은 일반적인 CNN 구조를 유지하되 각 layer의 filter를 task별로
나누어 사용한다. 작업 $J$에 대해 layer $k$의 filter index를
$I_k^J$로 두면, 해당 task는 그 범위까지의 filter만 사용한다.

직관적으로는 다음과 같다.

1. 첫 번째 task는 전체 용량의 일부만 사용해 학습한다.
2. 두 번째 task는 첫 번째 task의 파라미터는 고정하고, 남은 용량 일부를 새로 학습한다.
3. 이후 task도 같은 방식으로 남은 용량을 점진적으로 사용한다.

이때 새 task 추론은 "이전 task에서 쓰던 filter + 현재 task용으로
새로 추가된 filter"를 함께 사용할 수 있지만, 이전 task 추론은
자신에게 할당된 기존 filter만 사용한다. 이 비대칭 구조가 forgetting을
방지하는 핵심이다.

논문은 trainable parameter와 frozen parameter를 분리해 설명한다.
이전 task에서 사용된 파라미터 집합을 $W_k^P$, 현재 task에서
업데이트되는 파라미터 집합을 $W_k^T$로 두면 현재 task 학습은
$W_k^T$에 대해서만 수행된다.

## 4.3 수식 관점의 해석

논문은 EWC처럼 복잡한 보조 목적함수를 전면에 두기보다,
파라미터 사용 범위를 task별로 분할하는 구조적 제약 자체를 핵심
메커니즘으로 사용한다. 따라서 핵심은 손실식보다 다음과 같은 제약에
있다.

$$
W_k = W_k^P \cup W_k^T, \quad W_k^P \cap W_k^T = \varnothing
$$

여기서 $W_k^P$는 이전 task에서 이미 사용되어 고정된 파라미터이고,
$W_k^T$는 현재 task를 위해 새롭게 학습되는 파라미터다.

현재 task의 학습은 일반적인 classification loss로 수행된다.

$$
\mathcal{L}\_{\text{cls}} = - \sum\_{c=1}^{C_J} y_c \log p_c
$$

여기서 $C_J$는 task $J$의 class 개수다. 중요한 점은 loss 자체보다
**어떤 파라미터가 gradient를 받는가**가 StackNet의 본질이라는 것이다.

## 4.4 Index module

기존 PackNet류 방법은 task identity를 외부에서 알려줘야 하는 경우가
많다. 이를 해결하기 위해 저자들은 index module을 추가한다. 논문에서
제안한 대표 구현은 **GAN 기반 생성 모델**이다.

각 task마다 generator $G_J$를 두고, task별 binary classifier $B_J$가
입력이 해당 generator에서 생성된 분포와 유사한지 판별하도록 학습한다.
추론 시에는 가장 높은 확률을 주는 classifier를 선택해 task index를
정한다.

$$
J = \arg\max_i B_i(x)
$$

이 방식은 task label뿐 아니라 어떤 classifier head를 사용해야
하는지도 함께 정해주므로, label space가 task마다 다른 continual
learning 환경에 유용하다.

## 5. 실험 설정

## 5.1 데이터셋

논문은 기본 이미지 분류 데이터셋과 더 현실적인 데이터셋을 함께 사용한다.

| 데이터셋 | 학습 샘플 수 | 테스트 샘플 수 | 클래스 수 |
| --- | ---: | ---: | ---: |
| MNIST | 60,000 | 10,000 | 10 |
| SVHN | 73,275 | 26,032 | 10 |
| CIFAR-10 | 50,000 | 10,000 | 10 |
| ImageNet-A | 64,750 | 2,500 | 50 |
| ImageNet-B | 64,497 | 2,500 | 50 |

ImageNet-A와 ImageNet-B는 ImageNet에서 무작위로 선택한 50개 클래스 subset이다.

## 5.2 비교 대상

주요 비교 대상은 다음과 같다.

1. **LwF**
2. **PackNet**
3. **Single network baseline**

기본 실험은 `MNIST -> SVHN`, `SVHN -> CIFAR-10`,
`MNIST -> SVHN -> CIFAR-10` 같은 순차 학습 시나리오로 구성된다.
추가로 `cifar10`, `svhn`, `KMNIST`, `FashionMNIST`, `MNIST`를 이용한
5-task 실험도 수행한다.

## 5.3 백본

- 기본 벤치마크: VGG-16, ResNet-32
- 현실적 데이터셋 실험: ResNet-50

논문은 shortcut connection이 있는 구조에서도 StackNet이 동작함을 보이기 위해 ResNet-50 실험을 포함한다.

## 6. 실험 결과

## 6.1 기본 데이터셋 결과

2-task 및 3-task 설정에서 StackNet은 대체로 LwF보다 안정적이고 PackNet에 근접한 성능을 보인다.

### MNIST -> SVHN

| 방법 | MNIST (%) | SVHN (%) | 평균 (%) |
| --- | ---: | ---: | ---: |
| LwF ($T=1$) | 99.41 | 91.55 | 95.48 |
| LwF ($T=2$) | 99.33 | 91.73 | 95.53 |
| PackNet | 99.45 | 91.49 | 95.47 |
| StackNet* | 99.43 | 92.20 | 95.82 |
| StackNet | 99.43 | 92.37 | 95.90 |

이 설정에서는 StackNet이 평균 정확도에서 가장 높다.

### SVHN -> CIFAR-10

| 방법 | SVHN (%) | CIFAR-10 (%) | 평균 (%) |
| --- | ---: | ---: | ---: |
| LwF ($T=1$) | 91.76 | 70.57 | 81.17 |
| LwF ($T=2$) | 90.19 | 71.94 | 81.07 |
| PackNet | 92.84 | 76.78 | 84.81 |
| StackNet* | 90.05 | 76.62 | 83.34 |
| StackNet | 92.21 | 76.70 | 84.46 |

이 경우 StackNet은 PackNet보다 평균 정확도가 약간 낮지만,
pruning과 fine-tuning 없이 유사한 수준에 도달한다.

### MNIST -> SVHN -> CIFAR-10

| 방법 | MNIST (%) | SVHN (%) | CIFAR-10 (%) | 평균 (%) |
| --- | ---: | ---: | ---: | ---: |
| LwF ($T=1$) | 95.06 | 86.80 | 68.98 | 83.61 |
| LwF ($T=2$) | 86.09 | 85.07 | 69.63 | 80.26 |
| PackNet | 99.38 | 91.93 | 66.34 | 85.88 |
| StackNet* | 99.41 | 89.36 | 74.93 | 87.90 |
| StackNet | 99.41 | 91.84 | 75.02 | 88.76 |

3-task 설정에서는 StackNet이 평균 정확도에서 PackNet을 앞선다. 특히 새 task인 CIFAR-10 성능이 더 높게 나온다.

## 6.2 현실적 데이터셋 결과

ImageNet-A -> ImageNet-B 실험 결과는 다음과 같다.

| 방법 | Old (%) | New (%) | 평균 (%) |
| --- | ---: | ---: | ---: |
| LwF ($T=1$) | 82.20 | 86.72 | 84.46 |
| LwF ($T=2$) | 80.92 | 86.96 | 83.94 |
| PackNet | 82.16 | 88.72 | 85.44 |
| StackNet | 83.30 | 88.66 | 85.98 |

StackNet은 ImageNet subset 실험에서도 PackNet과 거의 같은 수준이며 평균 정확도는 약간 더 높다.

## 6.3 Index module 성능

논문은 GAN 기반 index module이 task 식별에서 매우 높은 정확도를 보인다고 보고한다.

- `MNIST -> SVHN`에서 GAN 기반 index module 평균 정확도는 `99.97%`
- `SVHN -> CIFAR-10`에서 GAN 기반 index module 평균 정확도는 `98.45%`
- `MNIST -> SVHN -> CIFAR-10`에서도 baseline 대비 큰 폭의 향상을 보인다.

즉, StackNet 자체뿐 아니라 **어떤 task branch를 활성화할지 추정하는 문제**도 실험적으로 해결 가능하다는 점을 보여준다.

## 7. 한계 및 향후 연구

논문이 직접 언급하거나 결과로부터 자연스럽게 드러나는 한계는 다음과 같다.

1. **고해상도 생성의 어려움**: index module이 GAN 기반이므로
   고해상도 이미지나 복잡한 분포에서는 task 추정이 어려워질 수 있다.
2. **유사한 데이터 분포 간 task 구분 문제**: task 간 시각적 분포가
   유사하면 index module의 분별력이 약해질 수 있다.
3. **고정 용량 가정**: StackNet은 초기 capacity 안에서 순차적으로
   파라미터를 배분하므로, task 수가 지나치게 많아지면 결국 확장이
   필요하다.
4. **비전 중심 검증**: 실험이 주로 이미지 분류에 집중되어 있어 NLP, speech, RL로의 일반화는 별도 검증이 필요하다.
5. **Task order 민감성**: 어떤 task가 먼저 들어오느냐에 따라 초기 capacity 배분 전략의 효과가 달라질 가능성이 있다.

향후 연구 방향으로는 다음이 유의미하다.

1. 더 강력한 generative model 또는 discriminative router로 index module 대체
2. adapter-style module과 결합해 parameter allocation을 더 세밀하게 제어
3. automatic capacity planning을 통해 task별 filter 증가량을 동적으로 결정
4. foundation model 백본 위에서 task-specific routing과 결합

## 8. 핵심 인사이트

이 논문이 주는 중요한 실무적 시사점은 다음과 같다.

1. **망각 방지는 반드시 복잡한 regularizer에서만 오지 않는다.**
   파라미터 사용 구간을 구조적으로 분리하는 것만으로도 강력한
   baseline이 될 수 있다.
2. **Task routing은 continual learning의 독립적인 하위 문제다.**
   실제 서비스 환경에서는 "입력이 어느 task에 속하는가"를 맞히는
   모듈이 필수일 수 있다.
3. **PackNet 대비 구현 복잡도가 낮다.** pruning, mask 관리, 재학습 과정 없이도 비슷한 성능을 얻는 점이 장점이다.
4. **Label-expandable setting에 적합하다.** task마다 class space가 달라지는 현실적인 문제 설정과 잘 맞는다.

## 9. 결론

StackNet은 task별 파라미터 공간을 순차적으로 쌓아 올리는 단순하지만
강한 continual learning 접근이다. 이전 task 파라미터를 고정하고
새 task용 파라미터만 추가로 학습하기 때문에 catastrophic forgetting을
구조적으로 억제한다. 여기에 입력의 task origin을 추정하는 index
module을 결합해, task identity가 명시되지 않은 상황까지 다루려는 점이
이 논문의 차별점이다.

결과적으로 StackNet은 PackNet과 경쟁력 있는 성능을 보이면서도
절차가 더 단순하고, multi-task 및 label expansion 상황에 잘 맞는
실용적 방법으로 정리할 수 있다.
