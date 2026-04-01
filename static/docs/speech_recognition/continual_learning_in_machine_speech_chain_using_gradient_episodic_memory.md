# CONTINUAL LEARNING IN MACHINE SPEECH CHAIN USING GRADIENT EPISODIC MEMORY

Geoffrey Tyndall, Kurniawati Azizah, Dipta Tanaya, Ayu Purwarianti, Dessi Puji Lestari, Sakriani Sakti

## 🧩 Problem to Solve

이 논문은 자동 음성 인식(ASR) 시스템이 새로운 작업을 순차적으로 학습할 때 발생하는 **재앙적 망각(catastrophic forgetting)** 문제를 해결하고자 한다. 기존의 ASR 시스템은 특정 작업에 대해 우수한 성능을 보이지만, 새로운 데이터나 작업에 적응하는 과정에서 이전에 학습했던 지식을 잃어버리는 경향이 있다.

- **대규모 모델의 한계**: Transformer와 같은 최신 대규모 ASR 모델들은 다중 작업에서 뛰어난 성능을 보이지만, 막대한 데이터와 컴퓨팅 자원을 요구하며 모든 작업을 처음부터 사용할 수 있어야 한다(오프라인 학습).
- **미세 조정(Fine-tuning)의 문제**: 지식 전달에 유용하지만, 이전 작업에 대한 모델 성능을 저하시키는 재앙적 망각을 유발한다.
- **다중 작업 학습(Multitask learning)의 문제**: 이전 데이터를 계속 유지해야 하므로 개인 정보 보호 문제가 발생할 수 있고, 데이터 관리의 복잡성이 증가한다.
  따라서 이 연구는 ASR 모델이 이전 작업을 잊지 않고 새로운 작업을 순차적으로 효율적으로 학습할 수 있는 **연속 학습(continual learning)** 패러다임을 개발하는 것을 목표로 한다.

## ✨ Key Contributions

- 기계 음성 사슬(Machine Speech Chain) 프레임워크 내에서 ASR을 위한 연속 학습을 가능하게 하는 새로운 **반지도 학습(semi-supervised learning) 기반 방법론**을 제안한다.
- 연속 학습 시나리오에서 Gradient Episodic Memory (GEM)의 **리플레이(replay) 메커니즘을 지원하기 위해 텍스트 음성 변환(TTS) 컴포넌트를 통합**한다. 이는 TTS를 ASR의 연속 학습을 위한 리플레이에 활용한 첫 시도이다.
- 제안된 방법이 기존의 미세 조정 및 다중 작업 학습 방식보다 **우수한 성능**을 보이며, 특히 재앙적 망각을 효과적으로 완화함을 실험적으로 입증한다.
- 잡음이 있는 환경(LJ Noisy dataset)에서도 높은 성능을 유지하며, 미세 조정 대비 **평균 오류율을 40% 감소**시키는 결과를 달성했다.
- 기계 음성 사슬 프레임워크가 음성 인식을 위한 효과적이고 효율적인 연속 학습을 실현하는 데 기여할 수 있음을 보여준다.

## 📎 Related Works

- **기계 음성 사슬 (Machine Speech Chain)** [9]: 인간의 음성 사슬(듣고 말하는 과정)에서 영감을 받아 자동 음성 인식(ASR)과 텍스트 음성 변환(TTS)을 폐쇄 루프(closed-loop) 프레임워크로 연결한 아키텍처. 적응형 롬바르드 TTS [12], 데이터 증강 [13], 코드 스위칭 [14] 등 다양한 응용 분야에서 활용되었다.
- **Gradient Episodic Memory (GEM)** [10]: 연속 학습 패러다임의 대표적인 리플레이(replay) 기반 방법 중 하나. 새로운 작업 학습 시 이전 작업의 샘플을 활용하여 새로운 작업의 그라디언트와 이전 작업의 그라디언트 간의 L2 거리를 최소화하고, 동시에 이전 작업에 대한 성능 손실을 방지하도록 그라디언트 방향에 제약($\langle \tilde{g}, g_k \rangle \ge 0$)을 둔다. ASR 분야에서 GEM을 사용한 이전 연구 [8]에서 다른 정규화 기반 방법보다 뛰어난 망각 방지 성능을 보였다.

## 🛠️ Methodology

이 연구는 ASR 모델이 기계 음성 사슬 프레임워크 내에서 반지도 학습 방식으로 연속 학습을 수행할 수 있도록 세 단계의 메커니즘을 제안한다.

**세 단계의 학습 프로세스:**

1. **첫 번째 단계: 기본 작업에 대한 지도 학습 (Supervised learning on the base task).**
   - ASR과 TTS 모델을 기본 작업 데이터(LJ Original)로 각각 독립적으로 지도 학습하여 견고한 초기 성능을 구축한다.
2. **두 번째 단계: 반지도 학습 (Semi-supervised learning).**
   - 기본 작업의 레이블이 없는 데이터를 사용하여 ASR과 TTS가 서로의 성능을 상호 보완적으로 향상시킨다.
3. **세 번째 단계: 연속 학습 (Continual learning).**
   - ASR은 새로운 작업(LJ Noisy)을 순차적으로 학습한다. 이 과정에서 TTS가 합성한 기본 작업의 **리플레이 입력**을 활용하여 재앙적 망각을 방지한다.

**GEM 기반 리플레이 메커니즘:**

- TTS는 기본 작업의 의사 샘플(pseudo-samples)을 생성하는 합성 모델로 사용된다.
- 이 의사 샘플들은 **에피소드 기억(episodic memory)**에 저장되며, Gradient Episodic Memory (GEM) 알고리즘에 의해 새로운 작업과 이전 작업 모두에 대한 그라디언트를 조절하는 데 사용된다.

**수학적 표현 (GEM 모델 업데이트):**
$i$-번째 작업 $D_i = \{(x_i^j, y_i^j)\}$를 학습할 때,

1. 이전 작업 $k$($k<i$)의 레이블 $y_k$로부터 TTS를 통해 의사 음성 샘플 $\hat{x}_k$를 생성하고, 이를 에피소드 기억 $M_0$에 저장한다:
   $$M_0 \leftarrow M_0 \cup (\hat{x}_0, y_k)$$
2. 현재 작업 $(x_i, y_i)$ 데이터를 에피소드 기억 $M_i$에 저장한다:
   $$M_i \leftarrow M_i \cup (x_i, y_i)$$
3. 현재 작업에 대한 모델 파라미터 $\theta$의 그라디언트 $g$를 계산한다:
   $$g \leftarrow \nabla_{\theta} \ell(\text{ASR}_{\theta}(x_i), y_i)$$
4. 이전 작업 $k < i$ 각각에 대해 에피소드 기억 $M_k$로부터 그라디언트 $g_k$를 계산한다:
   $$g_k \leftarrow \nabla_{\theta} \ell(\text{ASR}_{\theta}, M_k) \quad \text{for all } k < i$$
5. GEM의 제약 조건을 만족시키기 위해 그라디언트 $g$를 $\tilde{g}$로 투영(project)한다:
   $$\min_{\tilde{g}} \frac{1}{2} \|g - \tilde{g}\|^2_2 \quad \text{s.t.} \quad \langle \tilde{g}, g_k \rangle \ge 0, \forall k \in (0, \ldots, i-1)$$
   여기서 $\tilde{g} \leftarrow \text{PROJECT}(g, g_0, g_1, \ldots, g_{i-1})$ 이다.
6. 계산된 $\tilde{g}$를 사용하여 모델 파라미터를 업데이트한다:
   $$\theta \leftarrow \theta - \delta \tilde{g}$$
   이 접근 방식은 TTS를 ASR의 연속 학습 프레임워크 내에서 활용하는 최초의 시도이다.

## 📊 Results

실험은 LJ Speech 데이터셋의 원본(LJ Original)과 잡음이 추가된 버전(LJ Noisy)을 사용하여 ASR 모델의 연속 학습 성능을 평가했다. ASR 모델은 Speech-Transformer, TTS 모델은 Transformer-based Tacotron 2를 사용했다.

**주요 실험 결과:**

- **연속 학습 성능 (표 1):**
  - **GEM 적용 모델의 우수성**: Fine-tuning 및 Multitask 학습 대비 모든 시나리오에서 GEM을 사용한 모델의 문자 오류율(CER)이 현저히 낮았다. 특히, ASR$_{\text{Lower}}$ 조건에서 GEM은 LJ Original에서 8.5%, LJ Noisy에서 15.8%의 CER을 기록하며 fine-tuning(19.0%, 31.3%)을 크게 능가했다.
  - **기계 음성 사슬 기반 GEM (ASR$_{\text{SpeechChain}}$)**: fine-tuning 방법(CER: LJ Original 12.7%, LJ Noisy 33.1%)보다 우수한 CER(LJ Original 11.1%, LJ Noisy 15.5%)을 달성했다.
  - **오류율 감소 효과**: ASR$_{\text{SpeechChain}}$ 모델은 ASR$_{\text{Upper}}$ (완전 지도 학습) 모델만큼 낮은 오류율은 아니었지만, 해당 fine-tuning 방법 대비 **평균 오류율을 40% 감소**시키는 상당한 개선을 보였다.
- **데이터 분할 비율의 영향 (표 2):** 기본 작업 학습 시 레이블된 데이터의 비율이 높을수록(예: 70% 레이블링) ASR$_{\text{SpeechChain}}$ 모델의 CER이 더 낮아지는 경향을 보였다.
- **다른 연속 학습 방법과의 비교 (그림 2):**
  - **망각 방지 (BWT)**: 제안된 ASR$_{\text{SpeechChain}}$ 모델은 Elastic Weight Consolidation (EWC)보다 더 낮은 역방향 전이(BWT) 오류율(4.7% vs. EWC 7.8%)을 보여, 이전 작업에 대한 망각을 효과적으로 줄였음을 입증했다.
  - **새로운 작업 학습 (FWT)**: 순방향 전이(FWT) 지표에서는 ASR$_{\text{SpeechChain}}$(-0.3%)과 EWC(-0.1%)가 유사한 성능을 보였다.
    이러한 결과는 제안된 방법이 순차적인 작업을 효과적으로 학습하고, 재앙적 망각을 방지하며, 축적된 지식을 활용하여 새로운 작업을 학습하는 연속 학습의 핵심 특성을 모두 가지고 있음을 시사한다.

## 🧠 Insights & Discussion

- **재앙적 망각의 효과적인 완화**: 기계 음성 사슬 프레임워크 내에서 TTS 기반 리플레이를 통한 GEM의 적용은 ASR 모델이 새로운 작업을 학습하면서도 이전 작업에 대한 성능을 성공적으로 유지하게 한다. 이는 재앙적 망각이라는 핵심적인 연속 학습 문제를 효과적으로 해결한다.
- **반지도 학습의 실용성**: 대량의 레이블링된 음성 데이터가 필요한 완전 지도 학습 시나리오의 대안으로, TTS를 활용한 반지도 방식이 ASR의 연속 학습에 효과적임을 보여주었다. 이는 자원 제약이 있는 환경에서 큰 이점이 될 수 있다.
- **강건한 성능**: 제안된 방법은 단순한 fine-tuning이나 multitask learning과 같은 전통적인 접근 방식보다 훨씬 우수하며, 특히 잡음이 있는 환경에서도 안정적인 성능을 유지한다.
- **향후 연구 방향**: 현재 연구는 잡음 변화라는 비교적 단순한 작업 경계에서 성공을 보였다. 향후 다국어 음성 인식(다른 음운 체계 적응)이나 작업-불가지론적 연속 학습(작업이 미리 정의되지 않음)과 같이 더 복잡하고 다양한 시나리오에서의 일반화 가능성을 탐색할 필요가 있다.
- **윤리적 고려 및 대안**: TTS에 의해 생성된 합성 데이터 사용에 대한 잠재적 윤리적 위험(예: 편향, 오인용)을 인식하고 있으며, 폐쇄 루프 프레임워크 내에서 생성 과정을 제어함으로써 이를 완화하려 노력했다. 또한, 이는 대규모 실제 인간 음성 데이터에 대한 의존도를 줄이는 데 기여할 수 있다.

## 📌 TL;DR

이 논문은 자동 음성 인식(ASR) 모델의 재앙적 망각 문제를 해결하기 위해 **기계 음성 사슬 프레임워크**와 **Gradient Episodic Memory (GEM)**를 결합한 새로운 **반지도 연속 학습** 방법론을 제안한다. 텍스트 음성 변환(TTS) 컴포넌트를 활용하여 이전 작업의 의사 샘플을 생성하고 이를 리플레이 메모리로 사용하여, ASR 모델이 새로운 작업을 학습하는 동안 이전 작업의 성능을 유지할 수 있도록 그라디언트를 조절한다. 실험 결과, 제안된 방법은 기존의 미세 조정이나 다중 작업 학습보다 우수했으며, 잡음이 있는 환경에서도 강건한 성능을 보이고 평균 오류율을 40% 감소시키며 ASR을 위한 효과적이고 효율적인 연속 학습 가능성을 입증했다.
