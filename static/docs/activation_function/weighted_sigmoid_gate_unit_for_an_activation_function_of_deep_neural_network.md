# Weighted Sigmoid Gate Unit for an Activation Function of Deep Neural Network

첨부 논문은 ReLU 계열 활성화 함수의 한계를 넘기 위해, 입력에 sigmoid gate를 곱하는 **Weighted Sigmoid Gate Unit (WiG)** 를 제안한다. 핵심은 scalar 단위의 단순한 비선형성 대신, **가중치가 있는 gate가 입력의 각 성분을 얼마나 통과시킬지 결정**하게 만드는 것이다. 저자는 WiG가 ReLU, SiL, Swish를 모두 포괄하는 더 일반적인 형태라고 주장하며, 기존 연구가 주로 분류(object recognition)에서만 활성화 함수를 비교한 것과 달리, **객체 인식과 이미지 복원(denoising)** 두 과제에서 함께 검증했다. 논문의 결론은 WiG가 두 과제 모두에서 기존 활성화 함수들보다 우수하다는 것이다.

## 1. Paper Overview

이 논문이 풀고자 하는 문제는 간단하다. 딥러닝에서 ReLU는 계산이 간단하고 학습이 잘 되지만, 음수 영역을 완전히 잘라내는 구조적 한계가 있다. 그래서 ReLU를 대체하거나 보완하려는 활성화 함수가 많이 제안되었고, 특히 **sigmoid gating** 기반 접근이 유망한 흐름으로 소개된다. 저자는 여기서 한 걸음 더 나아가, scalar 입력에만 적용되는 게 아니라 **vector 입력 전체에 대해 더 유연한 gating을 수행하는 활성화 함수**로 WiG를 제안한다. 또한 기존 비교가 주로 분류 정확도에 치우쳤다는 점을 지적하고, 복원 문제까지 포함해 활성화 함수의 일반성을 보이려 한다.

## 2. Core Idea

WiG의 핵심 아이디어는 다음 식으로 요약된다.

$$
\mathbf{f}(\mathbf{x})=\mathbf{x}\odot \sigma(\mathbf{W}_g\mathbf{x}+\mathbf{b}_g)
$$

즉, 입력 $\mathbf{x}$ 자체를 그대로 쓰지 않고, 이를 **sigmoid gate** $\sigma(\mathbf{W}_g\mathbf{x}+\mathbf{b}_g)$ 로 조절한 뒤 element-wise product를 취한다. gate 값이 1에 가까우면 성분이 통과되고, 0에 가까우면 억제된다. 결국 WiG는 “입력을 살릴지 말지”를 단순 임계값이 아니라 **학습 가능한 가중 gate** 로 결정하는 구조다.

이 아이디어가 중요한 이유는 WiG가 기존 함수들을 특수한 경우로 포함하기 때문이다.

* **SiL**: $f(x)=x\sigma(x)$ 는 WiG의 특수한 경우다.
* **Swish**: $f(x)=x\sigma(\beta x)$ 역시 WiG의 특수한 경우다.
* **ReLU 유사 동작**: gate의 gain이 커지면 ReLU에 가까운 동작을 만들 수 있다. 저자는 simple example을 통해 WiG가 ReLU와 Swish를 special case로 포함한다고 설명한다.  

즉, WiG의 novelty는 완전히 새로운 계열의 함수라기보다, **기존 sigmoid-gated activation들을 하나의 더 일반적인 틀로 묶고, vector-input gate까지 확장**했다는 데 있다.

## 3. Detailed Method Explanation

### 3.1 기본 수식과 의미

논문이 제시한 WiG의 정의는 다음과 같다.

$$
\mathbf{f}(\mathbf{x})=\mathbf{x}\odot \sigma(\mathbf{W}_g\mathbf{x}+\mathbf{b}_g)
$$

여기서:

* $\mathbf{x}\in \mathbb{R}^N$: 입력 벡터
* $\mathbf{W}_g\in \mathbb{R}^{N\times N}$: gate를 계산하는 가중치
* $\mathbf{b}_g\in \mathbb{R}^N$: gate bias
* $\odot$: element-wise product

보통 활성화 함수는 선형변환 뒤에 붙는다. 이 점을 반영하면, 선형층 $\mathbf{W}\mathbf{x}$ 뒤에 WiG를 붙인 형태는 다음과 같이 쓸 수 있다.

$$
\mathbf{f}(\mathbf{W}\mathbf{x})
================================

\mathbf{W}\mathbf{x}\odot \sigma(\mathbf{V}\mathbf{x}),
\quad \mathbf{V}=\mathbf{W}\mathbf{W}_g
$$

저자는 이 구현이 기존 유사 네트워크와 계산 복잡도는 같지만, **병렬 계산 측면에서는 더 효율적일 수 있고**, 반대로 학습 관점에서는 $\mathbf{V}$ 와 $\mathbf{W}$ 의 통계적 성질이 달라 별도의 매개변수화가 유리할 수 있다고 해석한다.  

### 3.2 도함수와 학습 특성

논문은 WiG의 미분도 명시적으로 제시한다. 식의 형태상 gradient는 단순 ReLU보다 복잡하지만, 입력 자체와 gate의 미분 항이 함께 반영되므로 **hard threshold보다는 더 부드러운 gradient flow** 를 기대할 수 있다. 이 점은 뒤의 실험에서 WiG가 더 빠르게 cross-entropy를 낮춘다는 관찰과 연결된다. 도함수 식 자체는 길지만, 직관적으로는 “입력 통과량”과 “gate 변화율”이 함께 gradient를 형성한다고 보면 된다.

### 3.3 Simple example: ReLU/Swish 포함 관계

논문은 scalar case에서 WiG의 형태를 설명하면서, gain $w$ 와 bias $b$ 를 조절하면 derivative의 threshold를 제어할 수 있다고 말한다. 특히 bias는 gate가 언제 열리고 닫힐지를 조절하는 역할을 하며, 큰 gain은 ReLU에 가까운 sharp transition을 만든다. 이 subsection의 메시지는 명확하다. **WiG는 단순히 부드러운 ReLU가 아니라, threshold와 transition sharpness를 같이 조절 가능한 일반화된 gated activation** 이다.  

### 3.4 Initialization

초기화는 이 논문에서 꽤 중요한 부분이다. 저자는 $\mathbf{W}_g$ 와 $\mathbf{b}_g$ 를 위한 초기화를 따로 논의하며, 특히 $\mathbf{W}_g$ 를 **scaled identity matrix $s\mathbf{I}$** 로 초기화하는 방식을 제안한다. 이때:

* $s$ 가 크면 WiG는 초기 상태에서 **ReLU 근사**가 된다.
* 따라서 기존 ReLU 네트워크에서 WiG 네트워크로 **transfer learning** 하기 좋다.
* 반대로 처음부터 학습할 때는 $s=1$ 로 두어 **SiL 형태로 초기화**할 수 있다고 설명한다.  

이 초기화 전략은 WiG를 “완전히 낯선 활성화 함수”로 도입하지 않고, **기존 잘 작동하는 activation의 안전한 근방에서 출발**하게 해 준다는 점에서 실용적이다.

### 3.5 Sparseness constraint

저자는 sigmoid 출력이 일종의 **mask** 로 해석될 수 있다고 보고, 여기에 sparse regularization을 줄 수 있다고 제안한다. 즉, 단순 weight decay 외에 gate 마스크 자체의 sparsity를 유도하는 항을 도입해 **선택적 통과 구조**를 더 명확하게 만들려는 것이다. 다만 논문 전체에서 이 항이 실험 성능 개선의 중심축으로 크게 부각되지는 않는다. 구조적으로는 흥미롭지만, 실제 논문의 주된 기여는 여전히 WiG activation 자체와 그 일반성, 그리고 실험 성능에 있다.  

## 4. Experiments and Findings

## 4.1 비교 대상

논문은 ReLU 외에도 여러 activation과 비교한다. 검색 결과상 비교군에는 ELU 계열, SiL, PReLU, Swish 등이 포함되며, 일부 함수는 원 논문에서 쓰인 default parameter를 사용했다고 적혀 있다. 또한 기존 관련 연구들이 object recognition에만 초점을 두었다는 점을 강조하면서, 저자는 **분류 + 복원** 두 과제를 함께 실험한다.

## 4.2 Object recognition task

분류 실험에서는 CIFAR-10, CIFAR-100을 사용하고, categorical cross-entropy를 손실로 사용한다. Figure 6의 학습 곡선 설명에 따르면, **WiG를 쓴 네트워크가 더 빠르게 학습하고 더 낮은 training cross-entropy에 도달**했다. Table I에 대해서는 저자가 명시적으로:

* ReLU도 기본적으로 좋은 성능을 낸다.
* **WiG가 CIFAR-10과 CIFAR-100 둘 다에서 최고 성능을 보인다.**
* 특히 **CIFAR-100에서 개선 폭이 더 크다**고 해석한다.

검색 결과에 노출된 Table I 일부 수치로는 ReLU가 CIFAR-10에서 0.927, CIFAR-100에서 0.653의 validation accuracy를 보인다. WiG의 정확한 두 수치는 검색 스니펫에서 완전하게 드러나지 않았지만, 저자 서술상 **두 데이터셋 모두 최상위**임은 분명하다.  

## 4.3 Image restoration task: denoising

복원 실험에서는 여러 복원 문제 중 **image denoising** 을 선택한다. 저자는 dilation convolution과 skip connection을 사용하는 denoising network를 구성했다고 설명한다. 데이터셋은 Yang91, General100 등으로 학습하고, BSD68에서 평가한 것으로 보인다. 이 점은 activation 비교를 분류 외 영역으로 확장했다는 논문의 핵심 주장과 직접 연결된다.  

결과 해석은 더 강하다. 논문은 Table II, III를 바탕으로 다음을 주장한다.

* 기존에 제안된 여러 activation은 **어떤 noise level에서도 ReLU를 이기지 못했다.**
* 반면 **WiG만이 PSNR과 SSIM 모두에서 ReLU를 consistently 능가했다.**

검색 결과에 드러난 WiG의 평균 PSNR은 noise level 15, 20, 25, 30에서 각각 **32.16, 30.88, 29.90, 29.10** 이다.
또한 WiG의 평균 SSIM은 noise level 5, 10, 15, 20, 25, 30에서 각각 **0.9390, 0.8993, 0.8679, 0.8412, 0.8188, 0.7981** 로 나타난다.

절대값 자체보다 더 중요한 건 저자의 해석이다. 활성화 함수 논문들이 흔히 분류에서만 개선을 보이는 데 그치는 것과 달리, WiG는 **복원처럼 low-level vision 성격이 강한 문제에서도 일관된 이득**을 보여 준다는 것이다. 이것이 이 논문의 실험적 메시지다.

## 5. Strengths, Limitations, and Interpretation

### 강점

가장 큰 강점은 **일반화된 표현력**이다. WiG는 ReLU, SiL, Swish를 아우르는 형태로 제시되어, 새로운 함수 하나를 던지는 데 그치지 않고 **기존 activation family를 포괄하는 프레임**을 만든다.

둘째, 실험 설계가 의외로 좋다. 저자도 직접 말하듯, 기존 비교가 object recognition에 편중돼 있었는데, 이 논문은 image denoising까지 넣어 활성화 함수의 유효성을 **high-level task와 low-level task 모두에서 확인**한다. 특히 denoising에서 기존 activation들이 ReLU를 못 넘고 WiG만 이겼다는 점은 인상적이다.  

셋째, 초기화 전략이 실용적이다. scaled identity initialization으로 ReLU-like 또는 SiL-like 시작점을 택할 수 있어, 실제 네트워크 도입 장벽을 낮춘다.  

### 한계

한계도 분명하다.

첫째, 구조가 ReLU보다 복잡하다. gate를 위한 $\mathbf{W}_g, \mathbf{b}_g$ 가 추가되므로, 계산량과 파라미터 측면에서 단순 activation보다 무겁다. 논문은 병렬 계산 효율이나 구현 형태를 논의하지만, 실제 large-scale modern architecture에서의 cost-benefit은 이 논문만으로는 충분히 판단하기 어렵다.

둘째, 실험 범위가 제한적이다. CIFAR와 denoising은 의미 있는 벤치마크지만, 훨씬 큰 규모의 vision backbone이나 NLP/sequence 모델에서의 보편성은 보여 주지 않는다. 논문 자체가 LSTM의 sigmoid gating을 언급하긴 하지만, WiG를 sequence model에 실제 적용해 보인 것은 아니다.

셋째, sparse mask regularization은 흥미로운 아이디어지만, 논문의 핵심 실험에서 그 효과가 얼마나 본질적인지까지는 충분히 분리되어 검증되지 않는다. 따라서 이 부분은 “추가 가능성”에 가깝다.

### 해석

비판적으로 보면, WiG의 진짜 공헌은 “새로운 수식”보다 **gating을 activation 자체로 끌어올린 설계**에 있다. ReLU가 hard gate였다면, WiG는 그 gate를 **학습 가능한 soft gate** 로 바꾼 셈이다. 그래서 특히 denoising처럼 미세한 정보 보존이 중요한 문제에서 장점이 나타났다고 해석할 수 있다. 즉, “완전히 죽이거나 살리는” ReLU보다, “얼마나 통과시킬지”를 더 세밀하게 조절하는 쪽이 유리했던 것이다. 이 해석은 논문이 제시한 복원 실험 결과와 잘 맞는다.

## 6. Conclusion

이 논문은 WiG라는 sigmoid-gated activation을 제안하고, 이를 ReLU/SiL/Swish를 포함하는 더 일반적인 구조로 제시한다. 방법론적으로는 입력을 학습 가능한 sigmoid mask로 조절하는 간단한 구조지만, 초기화 전략과 special-case 해석까지 포함해 꽤 잘 정리되어 있다. 실험적으로는 CIFAR 분류와 image denoising 모두에서 강한 결과를 보였고, 특히 denoising에서는 WiG만이 ReLU를 일관되게 넘어섰다는 점이 핵심이다. 따라서 이 논문은 “활성화 함수는 분류 정확도만 조금 바꾸는 부속품”이 아니라, **표현과 정보 통과를 제어하는 실질적 설계 요소**라는 점을 잘 보여 준다.  
