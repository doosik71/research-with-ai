# Continual Classification Learning Using Generative Models

**저자**: Frantzeska Lavda, Jason Ramapuram, Magda Gregorova,
Alexandros Kalousis

**연도**: 2018

**arXiv ID**: <https://arxiv.org/abs/1810.10612v1>

**DOI**: <https://doi.org/10.48550/arXiv.1810.10612>

---

## 1. 연구 배경 및 문제 정의

Continual Learning은 작업이 순차적으로 도착하는 환경에서
과거 지식을 유지하면서 새로운 작업을 학습하는 문제를 다룬다.
하지만 일반적인 신경망은 각 작업을 순차적으로 학습할 때
이전 작업의 성능이 급격히 무너지는
**catastrophic forgetting** 문제를 겪는다.

이 논문은 특히 **분류(classification)** 문제를
연속 학습으로 다룬다.
저자들은 기존 접근을 크게 두 갈래로 정리한다.

- **동적 아키텍처 기반 방법**:
  작업이 추가될 때 네트워크 구조를 확장하거나
  과거 작업용 모델을 별도로 보존한다.
- **정규화 기반 방법**:
  이전 작업에 중요했던 파라미터가 크게 변하지 않도록 제약을 둔다.

저자들은 두 계열 모두 실용적 비용이 있다고 본다.
동적 아키텍처는 과거 작업별 모델을 유지해야 하고,
정규화 기반은 여전히 이전 작업 정보를
파라미터 형태로 강하게 보존해야 한다.
이에 따라 이 논문은 **과거 원본 데이터도,
과거 작업별 모델도 저장하지 않으면서**,
생성 모델이 과거 작업의 입력-출력 쌍을 재생성하도록 만들어
분류 문제의 연속 학습을 해결하려고 한다.

---

## 2. 핵심 아이디어

이 논문의 핵심은 두 가지다.

1. **VAE 기반 생성 모델에 분류기를 결합한다.**
2. **teacher-student 구조를 이용해
   과거 작업 데이터를 생성 재생(generative replay)한다.**

기존 VAE는 보통 $p(x)$에 대한 ELBO를 최적화하지만,
이 논문은 분류까지 함께 다루기 위해
**joint log-likelihood $\log p(x, y)$의
새로운 variational lower bound**를 도출한다.
그 결과 하나의 잠재표현 $z$가 재구성과 분류를 동시에
잘 수행하도록 학습된다.

또한 teacher 모델은 과거 작업 분포를 요약해 두었다가
$(\tilde{x}, \tilde{y})$를 생성하고,
student 모델은 새 작업의 실제 데이터 $(x, y)$와
teacher가 생성한 과거 작업 데이터를 함께 학습한다.
이 구조 덕분에 원본 과거 데이터를 직접 저장하지 않아도 된다.

---

## 3. 방법론

### 3.1 확률 모델

논문은 다음과 같은 잠재변수 모델을 사용한다.

$$
p(x, y, z) = p(x \mid z) p(y \mid z) p(z)
$$

여기서 입력 $x$와 라벨 $y$는
잠재변수 $z$가 주어지면 조건부 독립이라고 가정한다.

$$
p(x, y \mid z) = p(x \mid z) p(y \mid z)
$$

분류 시점에는 $y$를 사용할 수 없으므로,
실제 후방분포 $p(z \mid x, y)$ 대신
encoder가 정의하는 근사분포 $q_{\phi}(z \mid x)$를 사용한다.

### 3.2 Joint log-likelihood variational bound

논문이 제안하는 핵심 수식은
$\log p(x, y)$에 대한 하한이다.

$$
\log p(x, y) = \mathcal{L}(x, y)
+ D_{\mathrm{KL}}(q_{\phi}(z \mid x) \parallel p(z \mid x, y))
$$

따라서 KL divergence가 항상 0 이상이므로,

$$
\log p(x, y) \ge \mathcal{L}(x, y)
$$

이고, 이때 하한은 다음과 같이 정리된다.

$$
\mathcal{L}(x, y)
= \mathbb{E}_{q_{\phi}(z \mid x)}[\log p(x \mid z)]
- D_{\mathrm{KL}}(q_{\phi}(z \mid x) \parallel p(z))
+ \mathbb{E}_{q_{\phi}(z \mid x)}[\log p(y \mid z)]
$$

앞의 두 항은 일반적인 VAE의 ELBO와 같고,
마지막 항은 latent variable $z$를 통한
**classification loss** 역할을 한다.
이 구성의 장점은 분류기를 VAE에 임의로 덧붙이는 것이 아니라,
**생성과 분류를 하나의 확률모형 안에서 함께 최적화**한다는 데 있다.

### 3.3 Teacher-student 기반 continual learning

모델은 teacher와 student 두 네트워크로 구성되며,
둘 다 다음 세 모듈을 가진다.

- encoder $q_{\phi_m}(z \mid x)$
- decoder $p_{\theta_m^x}(x \mid z)$
- classifier $p_{\theta_m^y}(y \mid z)$

여기서 $m \in \{t, s\}$는
각각 teacher와 student를 의미한다.

학습 절차는 다음과 같다.

1. 현재 작업이 들어오면 student가 새 데이터 $(x, y)$를 받는다.
2. teacher는 이전 작업들에 대한 생성 샘플
   $(\tilde{x}, \tilde{y})$를 만든다.
3. student는 실제 새 데이터와 teacher가 생성한 과거 데이터를
   함께 사용해 학습한다.
4. 한 작업의 학습이 끝나면 student의 최신 파라미터가
   teacher로 전달된다.

이 설계는 이전 작업 데이터셋이나 task-specific head를
직접 저장하지 않고도 과거 분포를 유지하려는 목적에 맞는다.

### 3.4 최종 목적함수

저자들은 joint variational bound에 더해,
teacher와 student의 잠재표현이 크게 어긋나지 않도록 하는
정규화 항과 정보량 관련 regularizer를 추가한다.
논문에 제시된 최종 목적은 다음 요소를 포함한다.

- student의 재구성 항 $\log p_{\theta_s^x}(x \mid z_s)$
- student의 분류 항 $\log p_{\theta_s^y}(y \mid z_s)$
- prior에 대한 KL 항
- teacher와 student posterior 사이의 KL 항
- negative information gain regularizer

핵심 해석은 분명하다.
단순히 과거 이미지를 생성하는 데 그치지 않고,
**잠재표현 수준에서도 teacher와 student가
과거 분포를 일관되게 유지하도록 압박**한다는 점이다.

---

## 4. 실험 설정

### 4.1 데이터셋과 시나리오

저자들은 두 가지 연속 학습 시나리오를 사용한다.

1. **Permuted MNIST 5-task sequence**
   원본 MNIST와 4개의 고정 permutation task를
   순차적으로 학습한다.
   각 작업은 0-9 숫자에 대한 10-way classification이다.
2. **이종 작업 시퀀스**
   MNIST, FashionMNIST, 그리고 하나의 MNIST permutation을
   순차적으로 학습한다.

모든 설정에서 한 작업 학습이 끝난 뒤에는
그 작업의 원본 데이터에 더 이상 접근하지 않는다.
이는 논문이 강조하는 continual learning 제약과 일치한다.

### 4.2 베이스라인

논문은 두 가지 비교 대상을 둔다.

- **vae-cl**:
  제안한 variational bound는 사용하지만
  teacher-student 생성 재생 구조는 없는 모델
- **EWC baseline**:
  Elastic Weight Consolidation을
  저자들의 설정에 맞게 조정한 모델

논문에 따르면 하이퍼파라미터는 convolutional architecture와
dense architecture를 포함해 랜덤 탐색으로 선택했다.
최종적으로 제안 모델과 `vae-cl`은 convolutional 구조,
EWC는 dense 구조에서 최적 결과를 얻었다고 보고한다.

### 4.3 평가 지표

저자들은 작업이 하나씩 추가될 때마다
지금까지 학습한 모든 작업에 대한 평균 성능을 측정한다.

- **Average test classification accuracy**
- **Average test negative reconstruction ELBO**

정확도는 분류 성능 보존을,
negative reconstruction ELBO는 생성/재구성 능력 유지를 나타낸다.

---

## 5. 실험 결과 및 해석

### 5.1 Permuted MNIST 결과

Permuted MNIST 5-task 실험에서 논문은 다음 경향을 보고한다.

- `vae-cl`은 원본 MNIST에서 첫 permutation task로 넘어가는
  시점부터 성능이 급격히 하락한다.
- EWC는 `vae-cl`보다 덜 무너지지만,
  작업 수가 늘어날수록 이전 작업을 계속 잊는다.
- **CCL-GM(제안 방법)** 은 작업 수가 증가해도
  높은 평균 분류 정확도와
  낮은 평균 negative reconstruction ELBO를 유지한다.

즉, 제안 방법은 **분류 성능 유지**와
**생성 능력 유지**를 동시에 달성했다는 것이
저자들의 핵심 주장이다.

### 5.2 이종 작업 시퀀스 결과

MNIST, FashionMNIST, permutation task를 섞은 두 번째 실험에서도
제안 방법이 두 베이스라인보다 우수한 경향을 보였다고 보고한다.
이 결과는 단순한 permutation 변화뿐 아니라,
**입력 분포 자체가 다른 작업이 순차적으로 도착하는 경우에도
generative replay가 유효**하다는 점을 시사한다.

### 5.3 결과 해석

이 논문의 실험 메시지는 명확하다.

- teacher가 과거 작업의 입력-출력 쌍을 생성해 주면,
  student는 과거와 현재를 함께 학습하는 효과를 얻는다.
- 단순 분류기 보존보다,
  **공통 latent space에서 생성과 분류를 함께 묶는 구조**가
  망각 완화에 유리하다.
- 재구성 ELBO까지 함께 유지된다는 점은 모델이 과거 작업을
  단순히 분류 경계만 보존한 것이 아니라,
  **분포 수준으로 기억하려 했다**는 근거가 된다.

다만 논문은 5쪽짜리 workshop preprint라서,
본문에는 구체적인 수치 표가 없고 결과는 그림 중심으로 제시된다.
따라서 이 논문에서 확인 가능한 실험 결론은
**정성적 우위와 경향성** 수준으로 이해하는 것이 정확하다.

---

## 6. 한계 및 비판적 검토

### 6.1 평가 규모의 한계

실험은 주로 MNIST 계열 데이터셋에 집중되어 있다.
따라서 더 복잡한 자연 이미지, 장기 task sequence,
class-incremental setting으로 일반화할 수 있는지는
이 논문만으로 판단하기 어렵다.

### 6.2 생성 품질 의존성

제안 방법은 teacher가 과거 작업 샘플을
얼마나 잘 생성하느냐에 성능이 크게 좌우된다.
생성 품질이 무너지면 replay 데이터의 품질도 떨어지고,
결국 student의 분류 경계 역시 왜곡될 수 있다.

### 6.3 Task boundary 가정

논문은 순차적인 task 도착을 전제로 설명되며,
각 작업 전환 시점에서 teacher-student 업데이트가 일어난다.
따라서 task boundary가 불명확한 online continual learning에는
바로 적용하기 어렵다.

### 6.4 비교 범위의 제한

2018년 시점 기준으로는 의미 있는 비교지만,
이후 등장한 replay, regularization, parameter-isolation 계열의
강한 방법들과의 비교는 없다.
따라서 현재 시점에서 이 논문의 위치는
**초기 generative replay 기반 분류형 continual learning의
개념 증명**으로 보는 편이 적절하다.

---

## 7. 후속 연구 관점의 의미

이 논문은 몇 가지 점에서 의미가 있다.

- **생성 모델과 분류기를 하나의 latent variable model로 결합**해
  continual classification을 다룬 초기 사례 중 하나다.
- 단순 이미지 replay가 아니라
  **입력과 라벨을 함께 생성하는 joint replay** 관점을 제시했다.
- 이후의 generative replay, latent replay,
  pseudo-rehearsal 계열 연구를 이해할 때 좋은 출발점이 된다.

특히 이 논문은
"과거 데이터를 저장하지 않고도 과거 작업을 학습에
다시 등장시킬 수 있는가?"라는 질문에 대해,
**teacher가 과거 분포를 요약한 생성 메모리 역할을 수행한다**는
형태의 답을 제시한다.

---

## 8. 실무적 및 연구적 인사이트

- 원본 데이터를 저장하기 어렵거나 프라이버시 제약이 있는 환경에서는
  generative replay가 현실적인 대안이 될 수 있다.
- 분류 전용 모델보다 생성-분류 결합 모델이 더 무겁고
  학습이 까다롭지만,
  대신 **기억을 분포 수준에서 유지**할 수 있다.
- 현재 기준으로는 diffusion이나
  더 강한 auto-regressive generator를 활용한 replay가 가능하므로,
  이 논문의 아이디어는 현대 생성 모델로 다시 구현해 볼 가치가 있다.
- latent consistency regularization은 단순 출력 replay보다
  더 안정적인 continual learning 신호를 제공할 수 있다.

---

## 9. 결론

`Continual Classification Learning Using Generative Models`는
VAE 기반 joint generative-discriminative 학습과
teacher-student generative replay를 결합해,
과거 데이터와 과거 task-specific 모델을 저장하지 않는
continual classification 프레임워크를 제안한다.
핵심 기술적 기여는 $\log p(x, y)$에 대한
variational lower bound를 도출해 생성과 분류를
하나의 잠재모형 안에서 통합했다는 점이다.

실험은 소규모 벤치마크 중심이지만,
제안 방법이 naive VAE classifier와 EWC보다
catastrophic forgetting을 더 잘 완화한다는
정성적 근거를 제공한다.
오늘 기준으로 보면 대형 벤치마크에서의 강한 SOTA 논문은 아니지만,
**generative replay를 분류형 continual learning에
본격적으로 연결한 초기 작업**으로서 읽을 가치가 충분하다.
