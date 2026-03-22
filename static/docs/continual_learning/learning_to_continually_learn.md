# Learning to Continually Learn

* **저자**: Shawn Beaulieu, Lapo Frati, Thomas Miconi, Joel Lehman, Kenneth O. Stanley, Jeff Clune, Nick Cheney
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2002.09571](https://arxiv.org/abs/2002.09571)

## 1. 논문 개요

이 논문은 continual learning, 즉 순차적으로 들어오는 많은 작업을 학습하면서도 이전에 익힌 지식을 잃지 않는 학습 문제를 다룬다. 핵심적으로 해결하려는 문제는 catastrophic forgetting이다. 일반적인 신경망은 새로운 작업을 학습할 때 기존 작업에서 유용했던 파라미터를 다시 크게 바꾸는 경향이 있어, 이전 성능이 급격히 붕괴한다. 이 현상은 데이터를 무작위로 섞어 반복적으로 볼 수 없는 실제 환경, 예를 들어 로봇이 세상에서 지속적으로 경험을 쌓는 상황에서 특히 치명적이다.

저자들의 문제의식은 분명하다. 기존 연구들은 replay, regularization, synaptic importance, sparse representation 유도 등 다양한 해법을 제시했지만, 대부분 사람이 “이런 장치가 forgetting을 줄일 것이다”라고 가정한 뒤 손으로 설계한 방법이라는 점이다. 반면 이 논문은 forgetting을 줄이는 규칙 자체를 사람이 설계하지 말고, **“잊지 않고 계속 배우는 능력” 자체를 meta-learning으로 직접 학습하자**고 주장한다.

이를 위해 저자들은 **ANML (A Neuromodulated Meta-Learning algorithm)**을 제안한다. ANML은 하나의 신경망이 다른 신경망의 활성화를 입력 조건에 따라 gate하는 구조를 사용한다. 이 gating은 단지 forward pass의 표현을 바꾸는 데서 끝나지 않고, 어떤 뉴런이 활성화되느냐에 따라 gradient가 흐르는 정도도 달라지므로 backward pass에서의 plasticity에도 영향을 준다. 즉, 이 방법은 selective activation과 selective plasticity를 동시에 유도한다. 논문은 이 아이디어가 대규모 continual learning에서도 효과적임을 보이며, Omniglot에서 최대 600개 클래스를 순차 학습하는 설정에서 당시 SOTA였던 OML보다 크게 좋은 성능을 보였다고 보고한다.

이 문제의 중요성은 매우 크다. i.i.d. 학습 가정 아래에서만 잘 작동하는 시스템은 실제 장기 자율 학습 환경으로 확장되기 어렵다. 따라서 forgetting을 줄이면서 새 작업을 계속 흡수하는 능력은 lifelong learning, continual adaptation, 나아가 일반적인 지능 시스템 설계의 핵심 구성요소라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 다음과 같다. **모든 입력에 대해 네트워크 전체를 똑같이 활성화하고 똑같이 업데이트하면, 서로 다른 작업들이 같은 표현 공간과 같은 파라미터를 공유하면서 간섭을 일으키기 쉽다.** 그렇다면 입력의 문맥에 따라 네트워크의 일부만 선택적으로 활성화하고, 그 결과 그 부분만 더 많이 학습되도록 만들면 interference를 줄일 수 있다.

저자들은 이런 선택적 처리를 위해 **neuromodulatory network (NM)**와 **prediction learning network (PLN)**라는 두 개의 병렬 네트워크를 둔다. NM 네트워크는 입력을 받아 gating mask를 만들고, 이 mask가 PLN의 latent representation에 element-wise multiplication으로 적용된다. 그러면 특정 입력에 대해 PLN의 일부 차원은 강하게 통과하고, 일부는 억제된다. 이 selective activation은 곧 selective plasticity로 이어진다. 활성화가 약한 유닛은 gradient도 작게 받으므로, 입력 종류에 따라 네트워크의 서로 다른 부분이 주로 업데이트된다.

기존 접근과의 차별점은 세 가지로 정리할 수 있다.

첫째, **forgetting 방지 규칙을 직접 설계하지 않는다.** EWC처럼 어떤 파라미터가 중요한지 Fisher information으로 추정해 규제하는 방식이 아니라, 어떤 입력에서 어디를 얼마나 gate해야 하는지 자체를 outer loop에서 meta-learn한다.

둘째, **representation만 학습하는 OML보다 더 적극적으로 문맥 의존적 제어를 수행한다.** OML은 forgetting을 잘 줄이는 representation을 meta-learn하지만, 입력마다 다른 subnetwork를 골라 쓰는 직접적인 메커니즘은 없다. 반면 ANML은 입력에 따라 continuous mask를 생성한다.

셋째, **forward interference와 backward interference를 동시에 줄이려 한다.** 저자들이 강조하는 통찰은 “learning만 분리해도 충분하지 않다”는 것이다. 예를 들어 서로 다른 작업의 가중치 업데이트가 물리적으로 분리되어 있어도, forward 단계에서 서로 다른 작업의 신호가 동시에 섞이면 수행 자체가 망가질 수 있다. ANML은 activation 자체를 gate하기 때문에 forward와 backward 양쪽 간섭을 함께 줄이는 방향으로 작동한다.

## 3. 상세 방법 설명

### 3.1 문제 설정과 meta-learning 프레임

논문은 OML의 실험 프로토콜을 따른다. 실험 도메인은 Omniglot few-shot learning 데이터셋이며, 각 task는 하나의 문자 클래스이다. 총 1,623개 클래스 중 963개는 meta-training, 660개는 meta-testing에 사용한다.

meta-learning 용어를 저자들은 꽤 엄밀하게 구분한다. outer loop는 “잘 배우는 학습자”를 만드는 단계이고, inner loop는 실제 task를 학습하는 단계이다. 논문은 meta-train training, meta-train testing, meta-test training, meta-test testing이라는 표현을 써서 혼동을 줄이려 한다. 이 구분은 실제 알고리즘을 이해할 때 중요하다.

단순하게 생각하면, 긴 task sequence 전체를 inner loop에서 다 학습한 뒤 전체 검증 손실을 outer loss로 두고 역전파하면 되지만, 클래스 수가 600처럼 커지면 gradient가 불안정하고 메모리 비용도 너무 커서 비현실적이다. 따라서 OML이 제안한 근사 objective를 사용한다. 즉, 각 새 클래스를 학습한 직후에 다음 두 가지를 동시에 평가한다.

* 방금 본 클래스를 제대로 학습했는가
* 이전에 본 지식을 잃지 않았는가

이를 위해 현재 학습한 클래스의 샘플과, 과거 meta-training 클래스들에서 뽑은 random remember set의 샘플을 함께 평가하여 meta-loss를 만든다. 이 remember set은 meta-training에서만 사용된다.

### 3.2 ANML 아키텍처

OML은 하나의 네트워크를 representation learning network (RLN)와 prediction learning network (PLN)로 나눈다. 반면 ANML은 **두 개의 병렬 네트워크**를 쓴다.

하나는 **neuromodulatory network (NM)**이고, 다른 하나는 **prediction network** 이다. 두 네트워크 모두 outer loop에서 초기 파라미터를 meta-learn한다. 구조는 대략 비슷하며, 각각 3개의 convolution layer와 각 conv 뒤의 batch normalization, 그리고 1개의 fully connected layer를 가진다. 전체 파라미터 수는 기존 OML과 비슷한 규모인 약 6M 수준으로 맞춘다.

핵심은 NM의 마지막 출력 차원이 PLN의 마지막 fully connected layer 입력 차원과 같다는 점이다. PLN의 마지막 convolution layer가 만든 flatten된 latent representation을 $h(x)$라고 하면, NM 네트워크는 같은 입력 $x$에 대해 gating vector $g(x)$를 만든다. 이때 gated representation은 다음처럼 생각할 수 있다.

$$
\tilde{h}(x) = h(x) \odot g(x)
$$

여기서 $\odot$는 element-wise multiplication이다. $g(x)$는 sigmoid를 거쳐 $[0,1]$ 범위로 제한되므로, 이 논문에서 gate는 활성화를 **증폭하거나 부호를 바꾸지 않고 억제(suppress)**만 한다. 즉, 어떤 표현 차원은 거의 통과시키고, 어떤 차원은 거의 꺼버린다.

모든 일반 활성화 함수는 ReLU이고, gating multiplier만 sigmoid이다. 이 설계는 생물학적 neuromodulation에서 영감을 받았다고 설명한다. 다만 논문은 생물학적 정합성보다는 기계학습적 기능성에 초점을 둔다.

### 3.3 selective activation과 selective plasticity

ANML의 중요한 메커니즘은 단순히 표현을 sparsify하는 것이 아니다. gating된 표현이 다음 layer로 들어가므로, loss에 대한 gradient도 그 gate의 영향을 받는다. 직관적으로 어떤 뉴런이 거의 꺼져 있으면 그 경로의 backward gradient도 작아진다. 따라서 입력 종류에 따라 서로 다른 subnetworks가 더 많이 또는 덜 업데이트된다.

이것을 저자들은 **selective activation이 selective plasticity를 유도한다**고 표현한다. 즉, 별도의 hand-designed learning-rate modulation이나 importance regularization 없이도, activation control만으로 어느 부분이 학습될지를 간접 조절할 수 있다.

### 3.4 meta-training 절차

ANML의 outer loop와 inner loop는 다음과 같이 작동한다.

초기 meta-parameter는 NM 네트워크 가중치 $\theta^{NM}$와 prediction network 초기 가중치 $\theta^P$이다. 각 outer-loop iteration마다 하나의 meta-training class trajectory를 고른다. 이 trajectory는 사실상 한 클래스의 20개 샘플을 이용한 inner-loop 학습이다.

inner loop에서는 prediction network 가중치의 복사본 $\theta_0^P$를 만들고, 이 복사본을 20회의 SGD update로 학습시킨다. 논문 알고리즘 표기를 따르면 각 단계는 대략 다음과 같다.

$$
\theta_i^P \leftarrow \theta_{i-1}^P - \beta \nabla_{\theta_{i-1}^P}\mathcal{L}(\theta^{NM}, \theta_{i-1}^P, S_{traj})
$$

여기서 $\beta$는 inner-loop learning rate이고, $S_{traj}$는 현재 클래스의 학습 샘플들이다. 중요한 점은 이 loss 계산에서 prediction network의 latent representation이 NM에 의해 gating된다는 것이다. 따라서 $\theta^{NM}$는 직접 inner loop에서 업데이트되지는 않지만, inner-loop 학습 dynamics 전체에 영향을 준다.

20번의 SGD 업데이트가 끝나면, 최종 inner-loop 가중치 $\theta_{20}^P$를 이용해 meta-loss를 계산한다. meta-loss는 현재 학습한 클래스의 20개 이미지와 remember set에서 무작위로 뽑은 64개 character instance들에 대한 오차를 포함한다. 이 목적함수는 “새 클래스를 배웠는지”와 “이전 지식을 잊지 않았는지”를 동시에 측정한다.

그 다음 이 meta-loss를 inner-loop의 20개 SGD step 전체를 관통하여 backpropagation한다. 그 결과 $\theta^P$와 $\theta^{NM}$에 대한 outer gradient를 구하고, Adam으로 업데이트한다. 즉, outer update는 다음 형태로 볼 수 있다.

$$
(\theta^{NM}, \theta^P) \leftarrow (\theta^{NM}, \theta^P) - \alpha \nabla_{\theta^{NM}, \theta^P}\mathcal{L}(\theta^{NM}, \theta_k^P, S_{traj}, S_{rem})
$$

여기서 $\alpha$는 outer-loop learning rate이다.

논문에는 특이한 구현 디테일도 있다. meta-training에서 각 inner-loop 시작 시, 다음 trajectory에 해당하는 클래스의 출력 노드로 연결되는 final-layer weight는 meta-learned initialization을 그대로 쓰지 않고 **랜덤 재초기화** 한다. 이는 inner-loop learner가 이미 그 클래스를 미리 잘 분류하는 상태에서 시작하지 않도록 하기 위한 장치다. 동시에 meta-test 상황에서 모든 클래스가 새로운 상황과 더 비슷한 조건을 만든다. 다만 final layer 전체를 재초기화하면 remember set의 클래스에 대한 예측 능력까지 잃어버리므로, 해당 클래스에 대응하는 일부 weight만 재초기화한다.

### 3.5 meta-testing 절차

meta-training이 끝나면, meta-test에서는 660개의 unseen Omniglot 클래스 중 다양한 길이의 sequence를 구성해 평가한다. 길이는 10, 50, 75, 100, 150, 200, 300, 400, 500, 600 클래스까지 পরীক্ষা한다.

meta-test에서는 meta-learned $\theta^{NM}$와 $\theta^P$로 시작하지만, **prediction network의 fully connected layer만 fine-tune** 한다. convolution 및 batchnorm 관련 부분은 고정한다. NM 네트워크도 고정된다. 여기서 각 클래스는 15개의 training instance를 제공하며, 클래스 하나를 볼 때마다 15번 SGD update가 일어난다. 600개 클래스를 다 보면 총 9,000번 update가 발생한다.

중요한 차이는 meta-training과 달리, meta-test에서는 클래스마다 prediction network 사본을 새로 만들지 않는다는 것이다. 즉, 한 클래스에서 업데이트된 final layer가 다음 클래스로 계속 이어진다. 그래서 첫 클래스는 마지막 시점까지 8,985번의 추가 SGD update를 견뎌야 한다. 이 설정은 forgetting을 매우 강하게 유발하는 어려운 평가 조건이다.

최종 평가는 두 가지다.

첫째, meta-test training set 전체 9,000개 샘플을 다시 평가해 **기억 유지와 memorization 능력** 을 본다.
둘째, 각 클래스의 held-out 5개 샘플에 대해 평가해 **generalization을 포함한 진짜 continual learning 성능** 을 본다.

저자들은 후자를 더 중요하게 본다. training set 정확도는 단순히 “외웠는가”를 보여주지만, held-out test set 정확도는 “잊지 않으면서 새 클래스에 일반화 가능한 방식으로 배웠는가”를 보여주기 때문이다.

### 3.6 비교 기준선

논문은 ANML의 효과를 보기 위해 여러 baseline을 둔다.

첫 번째는 **Training from Scratch** 이다. meta-training 없이, meta-test 시점에 랜덤 초기화된 네트워크를 바로 순차 학습한다.

두 번째는 **Pretraining and Transfer** 이다. meta-training set을 i.i.d.로 pretraining한 뒤, 그 표현을 meta-test로 transfer하여 fully connected layers만 fine-tune한다. 이는 “ANML은 meta-training을 했으니 유리하다”는 비판을 완화하기 위한 비교다.

세 번째는 **OML** 이다. 이는 직접적인 SOTA baseline이다. OML은 RLN을 outer loop에서 meta-learn하고, meta-test에서는 fully connected layers를 fine-tune한다. 논문은 OML의 원래 설정뿐 아니라, final layer 하나만 fine-tune하는 **OML-OLFT** 도 평가한다. 이는 ANML과 fine-tuned parameter 수를 어느 정도 맞추기 위한 조정이다.

마지막으로 **Oracle** 들이 있다. 이들은 meta-test에서 데이터를 sequential이 아니라 i.i.d. interleaved 방식으로 보여준다. 따라서 catastrophic forgetting 문제가 사실상 사라진 상한선에 가까운 성능을 볼 수 있다.

## 4. 실험 및 결과

### 4.1 실험 설정

데이터셋은 Omniglot이며, 963개 클래스는 meta-training, 660개 클래스는 meta-testing에 사용한다. meta-train training에서 각 클래스는 20개의 labeled instance를 사용하고, meta-testing에서는 클래스당 15개 training instance와 5개 held-out test instance를 사용한다.

모든 treatment마다 10개의 독립적인 meta-trained 모델을 학습했다. meta-test test 성능은 각 meta-trained 모델에 대해 10개의 독립적인 task sequence를 샘플링해 측정했다. 따라서 결과는 어느 한 sequence나 한 번의 학습에 우연히 맞아떨어진 것이 아니라, 비교적 안정적인 평균 경향으로 제시된다.

### 4.2 meta-test training 정확도: forgetting 자체에 대한 저항성

저자들은 먼저, 최종적으로 학습이 끝난 뒤 지금까지 본 모든 meta-test training 데이터에 대해 얼마나 잘 기억하는지를 측정한다. 이 지표는 catastrophic forgetting 저항성을 직접 보여준다.

결과는 매우 강하다. **ANML은 시험한 모든 trajectory length에서 OML보다 meta-test training accuracy가 유의미하게 높다.** 논문은 모든 길이에 대해 Mann-Whitney U test 기준 $p \le 1.26 \times 10^{-8}$라고 보고한다.

반면 Scratch와 Pretrain & Transfer는 훨씬 나쁘다. 특히 sequence length가 길어질수록 거의 무너진다. 논문은 50개 이상의 sequential class에서 Scratch가 3% 미만, 400개 이상의 class에서 Pretrain이 3% 미만의 매우 낮은 정확도를 보인다고 설명한다. 이는 전형적인 catastrophic forgetting의 모습이다. 기존의 일반적 deep learning 초기화나 transfer만으로는 이 설정을 견디기 어렵다는 뜻이다.

흥미롭게도 OML-OLFT는 짧은 sequence에서는 OML보다 나쁘지만, 300개 이상처럼 더 긴 sequence에서는 오히려 OML보다 잘 버틴다. 이는 fine-tune하는 파라미터 수가 많을수록 적응력은 커지지만 forgetting도 심해질 수 있다는 점을 시사한다. 다만 어떤 OML 변형을 쓰더라도 ANML을 넘지는 못했다고 논문은 명시한다.

### 4.3 meta-test testing 정확도: 일반화 가능한 continual learning

더 중요한 결과는 held-out test instance에 대한 정확도다. 이 지표는 새로운 클래스의 training example을 적게 보고도, 나중에 그 클래스의 보지 못한 샘플을 얼마나 잘 맞히는지 보여준다.

여기서도 **ANML은 모든 trajectory length에서 다른 모든 방법보다 유의미하게 좋다.** 특히 600개 클래스를 순차 학습한 뒤의 결과가 인상적이다.

* **ANML**: 63.8%
* **OML**: 18.2%
* **OML-OLFT**: 44.2%

즉, ANML은 OML 대비 절대 성능 차이가 매우 크며, OML-OLFT와 비교해도 상당한 우위를 보인다. 논문은 모든 trajectory 길이에 대해 ANML이 OML, pretrained-transfer, scratch보다 유의미하게 우수하며, $p \le 2.58 \times 10^{-12}$라고 보고한다.

또한 600-task sequence 전체에서 **ANML은 99.3%의 클래스에 대해 chance보다 유의미하게 높은 성능** 을 보였다. 이는 sequence 초반에 본 클래스도 수천 번의 SGD update 이후 여전히 구분할 수 있다는 뜻이다. 저자들은 이를 catastrophic forgetting 없이 학습하고 있다는 강력한 증거로 제시한다.

### 4.4 oracle과의 비교

oracle은 meta-test 데이터를 sequential이 아니라 i.i.d.로 섞어 보여주는 버전이다. 당연히 같은 알고리즘이라도 oracle이 sequential 버전보다 좋아야 한다.

실제로 모든 알고리즘에서 oracle이 sequential보다 좋다. 그러나 중요한 점은 **순차 학습한 ANML이 다른 알고리즘들의 oracle보다도 좋다**는 것이다. 600개 클래스 설정에서 ANML은 OML oracle 및 অন্যান্য baseline oracle보다 높고, 오직 ANML-Oracle만이 ANML보다 높다.

600개 클래스에서:

* **ANML-Oracle**: 71.0%
* **ANML**: 63.8%

이 차이는 존재하지만, relative drop이 불과 10% 수준이다. 다른 방법들과 비교하면 이 수치는 압도적으로 작다.

* Scratch: 99% relative drop
* Pretraining & Transfer: 99%
* OML: 70.32%
* OML-OLFT: 27.2%
* ANML: 10%

즉, ANML은 i.i.d. 환경에서 가능한 성능 대부분을 sequential 환경에서도 유지한다. 이는 단순히 “약간 덜 잊는다”가 아니라, 이 실험 셋업에서는 catastrophic forgetting 문제의 상당 부분을 실제로 해결하고 있음을 뜻한다.

### 4.5 다중 epoch i.i.d. 환경에서도 강한 성능

저자들은 “oracle도 결국 데이터 한 번만 보는 온라인 세팅이라 일반적인 i.i.d. 학습보다 약할 수 있다”는 점도 인정한다. 그래서 20 epoch의 i.i.d. 학습도 따로 본다. 그 결과 Scratch는 61.8%, Pretrain & Transfer는 48.66%의 meta-test test accuracy를 얻는다. training accuracy는 각각 99.3%, 98.9%로 거의 완벽하지만 test accuracy는 낮으므로, few-shot 다중 클래스 환경에서 강한 overfitting이 발생한 것으로 해석한다.

그런데 흥미롭게도 **ANML은 sequential 한 번 패스만으로도 63.8%를 기록해, Scratch와 Pretrain의 20-epoch i.i.d.보다 높다.** 그리고 ANML을 i.i.d. 한 epoch로 학습하면 71%, 20 epoch에서는 75.37%까지 오른다. 저자들은 이 결과를 근거로, ANML의 이점이 단지 forgetting 감소에만 있지 않고 더 일반적인 representation/optimization의 장점일 가능성도 제시한다.

### 4.6 표현 분석: sparsity, separability, dead neuron

ANML이 왜 잘 작동하는지 보기 위해 저자들은 representation을 분석한다.

먼저 prediction network의 representation layer에서 activation sparsity를 본다. meta-test test set에 대해 neuromodulation 이전에는 평균 52.77%의 뉴런이 활성화되어 있고, gating 이후에는 평균 5.9%만 활성화된다. 기준은 activation이 0.01보다 큰 경우다. 이 감소는 통계적으로 유의하며, 논문은 $p < 10^{-6}$라고 제시한다.

흥미롭게도 OML representation의 sparsity는 3.89%로, ANML의 gated representation보다 더 sparse하다. 그런데 성능은 ANML이 더 좋다. 따라서 논문은 **“sparsity 자체만으로는 충분하지 않다”** 는 해석을 제시한다. 중요한 것은 sparse representation과 selective plasticity의 조합이라는 것이다.

또한 OML과 ANML 모두 meta-test training 시 representation layer에서 dead neuron이 0%라고 보고한다. 즉, 특정 입력에서는 꺼져 있을 수 있지만 전체 데이터셋을 통틀어 아예 한 번도 쓰이지 않는 낭비성 뉴런은 없었다. 이는 auxiliary sparsity loss로 강제했을 때 자주 나타나는 dead neuron 문제와 대조된다.

### 4.7 NM 네트워크가 실제로 문맥을 구분하는가

저자들은 NM 네트워크가 단순히 랜덤 마스크를 만드는 것이 아니라, 입력 유형에 따라 의미 있는 구분을 학습했는지를 확인한다.

이를 위해 meta-test test set 이미지에 대해 NM 출력 공간에서 KNN 분류를 수행한다. $K=5$를 사용한 결과, **NM activation만으로 70.9% accuracy** 가 나왔다. 같은 구조지만 랜덤 가중치를 가진 네트워크는 24.3%에 불과했다. 논문은 이 차이가 유의하며 $p = 2.58 \times 10^{-31}$라고 보고한다.

이 결과는 NM 네트워크가 unseen 클래스에 대해서도 이미지 유형을 구분하는 능력을 meta-learn했음을 뜻한다. 저자들의 해석대로라면, NM은 입력 종류별로 서로 다른 gating mask를 만들고, 그 결과 서로 다른 클래스 정보가 네트워크의 다른 부분에 저장되도록 유도한다.

### 4.8 gating 전후 PLN 표현의 separability

또 다른 분석은 PLN activation에 대해 gating 전후 KNN accuracy를 비교하는 것이다.

* **pre-neuromodulation**: 57%
* **post-neuromodulation**: 81.1%

즉, NM gating 이후 representation의 class separability가 크게 향상된다. 이 차이 역시 유의하며, 논문은 $p = 7.87 \times 10^{-12}$라고 보고한다. t-SNE 시각화에서도 gating 이후 군집이 더 잘 분리되어 보인다.

이 분석은 ANML의 역할이 단지 “어디를 학습시킬지 제한하는 것”에만 있지 않고, **처음부터 더 잘 분리되는 표현 공간을 형성하게 돕는 것** 이라는 점을 보여준다. 이 점은 forgetting 감소와 new task 학습 용이성을 동시에 설명해준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 연구 질문과 방법론이 직접적으로 맞닿아 있다는 점이다. 많은 continual learning 방법은 sparsity, regularization, replay efficiency 같은 대리 목표를 최적화한다. 반면 이 논문은 forgetting을 줄이는 학습 dynamics 자체를 meta-learning으로 학습한다. 이 설계 철학은 논문 전반에서 일관되고, 실제로 매우 강한 empirical result로 뒷받침된다.

두 번째 강점은 **selective activation과 selective plasticity를 하나의 간단한 메커니즘으로 연결했다는 점** 이다. NM이 만든 mask를 latent representation에 곱하는 것만으로 forward interference와 backward interference를 동시에 줄인다는 설명은 직관적이며, 실험 분석도 이 주장을 꽤 잘 지지한다.

세 번째 강점은 **실험 규모와 비교의 설득력** 이다. 최대 600 sequential classes, 약 9,000 SGD updates라는 긴 trajectory를 사용했고, 단순 scratch, pretraining, 기존 SOTA인 OML, oracle까지 폭넓게 비교했다. 특히 sequential ANML이 다른 방법의 oracle보다 좋다는 결과는 매우 인상적이다.

네 번째 강점은 **representation 분석이 단순 성능 보고를 넘어서 메커니즘 이해를 시도했다는 점** 이다. sparsity, dead neuron, KNN in NM space, pre/post gating separability, t-SNE 등을 통해 왜 잘 되는지 설명하려 한다.

반면 한계도 분명하다.

첫째, **실험 도메인이 Omniglot에 한정된다.** Omniglot은 continual learning 벤치마크로 의미가 있지만, 각 task가 비교적 단순한 character classification이다. 더 복잡한 시각 인식, 자연어, 강화학습, non-stationary control 문제로 일반화될지는 이 논문만으로 판단하기 어렵다.

둘째, **meta-training과 meta-testing 분포가 같은 계열의 문제** 라는 점이다. unseen 클래스이긴 하지만 같은 Omniglot 도메인 안에서 평가한다. 완전히 다른 task family로의 transfer나 distribution shift에 대한 검증은 없다.

셋째, **방법 자체는 여전히 특정 설계 선택에 의존한다.** 예를 들어 gating을 prediction network의 한 layer에만 적용했다. 논문도 이를 simplifying assumption이라고 인정한다. 여러 layer에 gating을 넣거나, 더 fine-grained한 synapse-level modulation을 적용하면 성능이 바뀔 수 있지만, 본 논문은 그 가능성을 체계적으로 탐색하지 않는다.

넷째, **계산 비용과 구현 복잡도** 도 적지 않다. inner-loop SGD 여러 step을 거친 뒤 outer-loop에서 through-SGD differentiation을 수행하는 MAML 스타일 학습이므로, 일반 supervised learning보다 훨씬 무겁다. 논문은 20,000 outer-loop iteration을 수행했다고 적고 있다. 실용성 측면에서는 상당한 계산 자원이 필요할 가능성이 높다.

다섯째, 논문은 매우 강한 성능을 보이지만, **왜 i.i.d. multi-epoch 환경에서도 ANML이 기존 방법보다 우수한지에 대한 완전한 설명은 제공하지 못한다.** 저자 스스로 이를 open and interesting research question이라고 인정한다. 즉, ANML의 이점이 순수한 forgetting 완화 때문인지, representation learning 자체의 개선 때문인지, optimization bias 때문인지는 아직 충분히 분해되지 않았다.

비판적으로 보면, 이 논문은 “meta-learning이 hand-designed continual learning보다 낫다”는 메시지를 강하게 밀고 있지만, 그 결론은 이 실험 세팅에서는 매우 설득력 있어도 모든 continual learning 설정으로 일반화되었다고 보기는 이르다. 그럼에도 불구하고, 최소한 **문맥 의존적 gating을 meta-learn하는 접근이 continual learning에서 매우 유망하다**는 점은 강하게 입증한다.

## 6. 결론

이 논문은 continual learning에서 catastrophic forgetting을 줄이기 위해, 해결 규칙을 사람이 설계하는 대신 **계속 배우는 방법 자체를 meta-learning으로 학습하는** ANML을 제안했다. 핵심은 neuromodulatory network가 입력에 따라 prediction network의 latent activation을 gate하고, 이 selective activation이 selective plasticity를 유도하도록 만드는 것이다.

실험적으로 ANML은 Omniglot에서 최대 600개 클래스를 순차 학습하는 어려운 설정에서 OML을 포함한 모든 비교 방법을 능가했다. 특히 held-out test accuracy 기준으로 600-class sequence에서 63.8%를 달성했고, 이는 OML의 18.2%보다 매우 높다. 또한 i.i.d. oracle 대비 성능 저하가 10%에 불과해, 이 벤치마크에서는 catastrophic forgetting 문제를 상당 부분 해결한 것으로 해석할 수 있다.

이 연구의 기여는 단순히 하나의 성능 향상 알고리즘을 제시한 데 있지 않다. 더 넓게 보면, **어려운 학습 문제의 해법을 직접 설계하기보다 meta-learning으로 발견하게 하자**는 관점을 강하게 지지하는 사례다. 따라서 이 논문은 continual learning뿐 아니라, 향후 architecture search, learned optimizers, differentiable plasticity, reinforcement learning 기반 lifelong adaptation 같은 방향과 결합될 가능성이 크다.

실제 적용까지는 더 복잡한 도메인에서의 검증이 필요하지만, 이 논문은 적어도 한 가지를 분명히 보여준다. 입력 조건에 따라 네트워크의 활성과 가소성을 선택적으로 제어하는 meta-learned neuromodulation은 continual learning의 매우 유력한 설계 원리이며, 이후 연구의 중요한 출발점이 될 수 있다.
