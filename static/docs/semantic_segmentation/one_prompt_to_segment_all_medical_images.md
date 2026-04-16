# One-Prompt to Segment All Medical Images

- **저자**: Junde Wu, Jiayuan Zhu, Yueming Jin, Min Xu  
  제공된 본문 첫 페이지에는 저자 표기가 혼재되어 있으며, 별도의 사과문에서 Jiayuan Zhu와 Yueming Jin이 저자 목록에서 누락되었다고 명시되어 있다.
- **발표연도**: 2024
- **arXiv**: https://arxiv.org/abs/2305.10300

## 1. 논문 개요

이 논문은 의료 영상 분할(medical image segmentation)을 위한 새로운 범용 추론 패러다임으로 **One-Prompt Segmentation**을 제안한다. 문제의식은 분명하다. 일반적인 fully-supervised segmentation은 특정 데이터셋과 특정 해부학적 구조에 맞춘 모델을 따로 학습해야 하므로 새로운 작업으로 거의 일반화되지 않는다. 반면 최근의 interactive segmentation 계열, 특히 SAM류 모델은 zero-shot 성질을 어느 정도 보이지만, 추론 시마다 각 이미지마다 사용자가 prompt를 다시 넣어야 한다. few-shot 또는 one-shot transfer learning 계열은 새로운 작업에 적응할 수 있으나, support example에 대한 정답 마스크가 필요하므로 실제 임상에서는 비용이 높다.

저자들은 이 두 흐름 사이의 간극을 줄이기 위해, 보지 못한 새로운 segmentation task가 주어졌을 때 **단 하나의 prompted template sample**만으로 그 작업을 이해하고, 이후 query image들을 **재학습 없이 한 번의 forward pass로** 분할하는 설정을 제안한다. 즉, 사용자는 새로운 작업마다 전체 라벨링을 할 필요도 없고, 각 테스트 이미지마다 prompt를 다시 줄 필요도 없다. 논문은 이것이 실제 임상 현장에서 더 저비용이고 사용 친화적인 방식이라고 주장한다.

이 문제가 중요한 이유는 의료 영상 분할의 실제 적용 환경이 매우 다양하기 때문이다. 장기, 병변, 혈관, 신경, 종양 등 목표 구조가 매우 다르고, CT, MRI, fundus, ultrasound 등 modality도 광범위하다. 이런 환경에서 task-specific 모델을 계속 따로 만드는 것은 비효율적이며, foundation-model 스타일의 범용성이 특히 가치 있다. 논문은 바로 이 지점에서 “universal medical image segmentation”을 목표로 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **“one-shot support label”의 장점과 “interactive prompt”의 장점을 결합하자**는 것이다. 즉, 새로운 작업에 대한 정보를 사용자로부터 받되, 그 정보가 반드시 dense segmentation label일 필요는 없도록 설계한다. 사용자는 template image 한 장에 대해 `Click`, `BBox`, `Doodle`, `SegLab` 중 하나의 prompt만 제공하면 되고, 모델은 그 정보를 query image 분할에 전이한다.

논문은 이 설정을 다음처럼 해석한다. query image 자체를 prompt하면 interactive segmentation의 특수한 경우가 되고, prompt가 segmentation label이면 one-shot segmentation의 특수한 경우가 된다. 따라서 One-Prompt Segmentation은 둘을 포괄하는 더 일반적인 프레임으로 제시된다.

이 아이디어를 실현하기 위해 저자들은 **One-Prompt Former**라는 decoder 구조와 그 내부의 **Prompt-Parser**를 제안한다. 핵심은 template image의 visual feature, query image의 visual feature, 그리고 사용자 prompt embedding을 함께 섞어, “template에서 지정된 target이 query의 어디에 대응하는가”를 feature 수준에서 찾도록 만드는 것이다. 단순히 prompt를 encoder 입력에 붙이는 것이 아니라, multi-scale feature와 cross-attention을 통해 template-query alignment를 반복적으로 수행하는 점이 이 논문의 설계적 차별점이다.

또 하나의 중요한 아이디어는 데이터 측면이다. 저자들은 78개의 공개 의료 데이터셋을 모으고, 그중 64개로 학습하며, 3,000개 이상의 sample에 대해 clinician이 prompt를 직접 부여한 데이터를 구축했다. 논문은 범용 generalization이 큰 폭으로 좋아진 이유로 아키텍처뿐 아니라 이러한 대규모 이질적 데이터 구성을 함께 제시한다.

## 3. 상세 방법 설명

논문은 전체 의료 영상 작업 집합을 $D$라고 두고, 각 task $d$는 이미지-라벨 쌍 $(x^d, y^d)$로 구성된다고 설명한다. 일반적인 fully-supervised segmentation은 task마다 별도의 함수 $f^d_\theta$를 학습해

$y^d = f^d_\theta(x^d)$

를 푼다. few-shot 설정에서는 task별 support set $S^d = \{(x^d_j, y^d_j)\}_{j=1}^n$를 함께 넣어

$y^d = f_\theta(x^d, S^d)$

를 학습한다.

반면 이 논문이 제안한 One-Prompt Segmentation은 task $d$마다 support set 전체가 아니라 **하나의 template image와 그 prompt**만 사용한다. 즉, prompt set을

$k^d = \{x^d_c, p^d_c\}$

로 두고, 보다 일반적인 함수

$y = f_\theta(x^d_j, k^d)$

를 학습한다. 여기서 핵심은 $p^d_c$가 반드시 segmentation mask일 필요가 없고, 훨씬 가벼운 prompt일 수도 있다는 점이다.

### 프롬프트 표현

모델은 네 가지 prompt를 지원한다.

`Click`은 foreground와 background를 나타내는 sparse point prompt이고, `BBox`는 좌상단과 우하단 corner point를 의미한다. `Doodle`은 사용자가 자유롭게 선을 그리는 freehand prompt이며, `SegLab`은 segmentation label mask 자체를 의미한다. 논문은 구조가 복잡한 pancreas나 mandible 같은 경우 `Doodle`이 유용하고, vessel처럼 세밀한 구조는 `SegLab`이 적합하다고 설명한다.

모든 prompt는 두 개의 embedding $p_1$, $p_2$로 표현된다. `Click`과 `Doodle`에서는 foreground/background 의미를, `BBox`에서는 두 개의 코너를 나타낸다. 이 세 종류에 대해서는 positional encoding으로 좌표 정보를 압축한 뒤 learnable embedding에 더한다. 반면 `SegLab`은 pretrained autoencoder를 통해 embedding으로 변환되며, 이 경우 두 embedding이 같은 파라미터를 공유한다고 적혀 있다. 다만 autoencoder의 상세 구조는 제공된 본문에는 나오지 않는다.

### 전체 모델 구조

모델은 하나의 image encoder와 여러 층의 **One-Prompt Former** decoder로 구성된다. 입력은 query image $x_q$, template image $x_t$, 그리고 template의 prompt $p_t$이며, 출력은 query segmentation $y_q$이다. encoder는 CNN 기반일 수도 있고 ViT 기반일 수도 있다고 서술되어 있으며, 그림에서는 단순화를 위해 CNN 기반 encoder가 제시된다. query와 template는 **같은 encoder**를 통과하여 각각 feature $f_q$, $f_t$를 얻는다.

decoder는 multi-scale skip connection을 사용하며, 각 스케일의 feature map을 patchify, flatten, projection하여 $e \in \mathbb{R}^{N \times L}$ 형태의 embedding으로 바꾼 뒤 처리한다. 여기서 $N$, $L$의 정확한 의미는 제공된 본문만으로는 명확히 정의되지 않는다.

### One-Prompt Former

One-Prompt Former는 query branch와 template branch의 두 병렬 경로를 가진 attention 중심 모듈이다. 설명에 따르면, $i$번째 block에서 query branch는 먼저 현재 스케일의 skip-connected query embedding $e_l^q$를 query로 사용하고, 이전 단계의 embedding $e_s^{i-1}$를 key와 value로 쓰는 Cross Attention을 수행한다. 이어서 template feature를 반영하기 위한 또 하나의 Cross Attention이 적용된다.

반면 template branch는 먼저 **Prompt-Parser**를 통해 prompt, query, template feature를 결합하고, 이후 query 정보를 다시 반영하는 Cross Attention을 수행한다. 마지막에는 두 branch를 Cross Attention으로 통합하여, “prompt가 지정한 template 내 target segmentation 정보를 query 도메인으로 옮긴다”고 설명한다. 그 다음 self-attention과 feed-forward neural network를 거쳐 embedding을 다음 단계에 전달한다.

이 구조의 직관은, query와 template를 각각 독립 처리하지 않고, 여러 스케일에서 반복적으로 서로를 참조하게 만들어 task-specific 단서를 점진적으로 정렬하는 것이다. 특히 template branch 안에서 prompt를 이용해 target 영역을 강조한 뒤, 그 정보를 query로 투사하는 것이 핵심이다.

### Prompt-Parser

Prompt-Parser는 template branch 안에서 prompt와 feature를 효과적으로 섞기 위한 모듈이다. 저자들은 우선 query-template-integrated embedding을

$$
e_l^{tq} = e_l^t (p_1 + p_2 + e_l^q)
$$

로 정의한다. 표기상 곱셈이 어떤 종류의 연산인지 문맥상 feature-level interaction으로 이해되지만, 제공된 본문만으로 element-wise인지 matrix product인지는 명확하지 않다.

그 다음 Prompt-Parser는 **Prompting Step**과 **Masking Step**으로 나뉜다.

Prompting Step에서는 template feature $e_l^t$, query feature $e_l^q$, 그리고 prompt embedding $p_1, p_2$를 섞어서 adaptive mask $M$을 만든다. 먼저 세 embedding을 쌓은 뒤

$$
[e_l^t; p_1; p_2] \in \mathbb{R}^{3N \times L}
$$

에 대해 가중치 $w \in \mathbb{R}^{N \times 3N}$를 갖는 MLP를 적용하여 차원을 다시 $N$으로 줄인다. 이후 여기에 query feature를 곱해 query domain으로 activation을 옮긴다. 논문은 이를

$$
M = w[e_l^t; p_1; p_2](e_l^q)^T
$$

로 쓴다. 직관적으로는 “template의 어느 부분이 prompt로 지정되었는가”와 “그 부분이 query feature에서 어디와 대응하는가”를 함께 반영한 attention-like mask라고 이해할 수 있다.

Masking Step에서는 이 마스크 $M \in \mathbb{R}^{N \times N}$를 이용해 $e_l^{tq}$를 조절한다. 제안 연산은 **Gaussian Masking**이며 다음과 같이 주어진다.

$$
e_G = Max(e_l^{tq} * k_G[Conv(M)], e_l^{tq})
$$

여기서 $k_G$는 Gaussian kernel이고, $*$는 일반 convolution 연산이라고 서술되어 있다. 구체적으로는 먼저 $M$을 convolution으로 2-channel feature map으로 바꾸고, 이 두 채널을 각각 Gaussian의 mean과 variance로 사용해 $k_G$를 만든다. 그 뒤 이 Gaussian kernel을 $e_l^{tq}$에 pixel-wise하게 적용하여, prompt가 가리키는 영역 주변까지 불확실성을 포함해 부드럽게 확장한다. 마지막에 원본 feature와 Gaussian-smoothed feature의 최대값을 취함으로써, 높은 activation은 유지하고 낮은 신뢰도의 불확실 영역은 억제하려는 의도로 보인다. 출력은 최종적으로 다시 $e_l^t$와 곱해 얻는다고 설명되어 있다.

요약하면 Prompt-Parser는 단순한 hard mask가 아니라, prompt가 가리키는 정보를 query-template 대응 구조 속에서 부드럽게 확장하고 보존하는 모듈이다. 이것이 저자들이 ablation에서 가장 효과적이라고 주장하는 부분이다.

### 학습과 추론

데이터는 78개 데이터셋 중 64개를 학습용, 14개를 테스트용으로 나눴다. 각 학습 데이터셋은 prompted template split, training split, validation split으로 나뉘며, 테스트 데이터셋은 test split과 prompted template split으로 구성된다. 인간 사용자가 template split의 샘플에 prompt를 부여한다.

학습 시에는 query image와 같은 데이터셋에서 랜덤하게 하나의 prompted template를 뽑아 함께 넣는다. 즉, 한 iteration마다 “같은 task에서 query 하나, prompted template 하나”를 짝지어 학습하는 구조다. 64개 데이터셋 전체를 합쳐 공동 학습한다.

최종 loss는 **cross-entropy loss와 dice loss의 단순 합**이다. 가중치 조정이나 추가 regularization은 제공된 본문에는 명시되어 있지 않다.

추론 시에는 template split에서 prompted template 하나를 랜덤하게 고른 뒤, test/validation split 전체에 대해 예측을 수행한다. 기본 설정에서는 이를 50회 반복해 ensemble하여 variance를 줄였다고 한다. 이 부분은 성능 안정성에는 도움이 되지만, 실제 배포 시 추론 비용 측면에서는 추가 고려가 필요하다.

## 4. 실험 및 결과

논문은 14개의 hold-out task를 사용해 보지 못한 의료 영상 분할 작업으로의 일반화 능력을 검증한다. 여기에는 8개의 MICCAI 2023 challenge task와 6개의 추가 task가 포함된다. 예를 들어 kidney tumor, liver tumor, breast cancer, nasopharynx cancer, vestibular schwannoma, lymph node, cerebral artery, inferior alveolar nerve, white blood cell, optic cup, mandible, coronary artery, pancreas, retinal blood vessel 등이 포함된다.

### Human-user prompted evaluation

평가에서 저자들은 실제 사용자 상호작용을 흉내 내기 위해 15명의 사용자를 동원해 테스트 이미지의 약 10%를 prompt하게 했다. 이 구성은 일반 사용자 5명, junior clinician 7명, senior clinician 3명으로 이뤄진다. 논문은 이를 임상 교육, 반자동 annotation 등 실제 사용 상황과 유사한 조건으로 제시한다.

이 실험 설계는 중요한 의미가 있다. 많은 논문이 자동 생성된 prompt만으로 평가하지만, 이 논문은 일부 평가에서 실제 사람이 넣은 prompt를 사용해 “프롬프트 방식이 실제로 쓸 만한가”를 확인하려고 한다. 다만 테스트 이미지의 10%만 사람 prompt를 사용했다는 점에서, 전체 테스트셋 수준의 대규모 human evaluation은 아니다.

### One-Prompt transfer capability

이 실험의 목적은 보지 못한 task에 대해 One-Prompt Model이 얼마나 잘 일반화하는지 확인하는 것이다. 비교 대상은 few-shot 계열 `PANet`, `ALPNet`, `SENet`, `UniverSeg`와 one-shot 계열 `DAT`, `ProbOne`, `HyperSegNas`, `LT-Net`이다. few-shot 계열도 공정 비교를 위해 테스트 시 template를 하나만 제공했다고 한다. one-shot 모델들은 sparse prompt를 받을 수 없기 때문에 segmentation label을 prompt처럼 사용했다.

여기서 중요한 점은 비교 대상 다수가 **dense segmentation label**을 입력으로 받는 반면, 제안 모델은 `Click`, `BBox`, `Doodle`, `SegLab` 등 더 약한 supervision도 사용 가능하다는 것이다. 논문은 이 점을 감안하면 자사 모델이 불리한 조건에서도 경쟁력이 있다고 주장한다.

결과에 따르면, 제안 모델은 다양한 prompt 설정 전반에서 경쟁 모델들을 유의미하게 앞선다. 특히 모든 방법이 segmentation label을 입력받는 가장 공정한 비교인 `SegLab` 설정에서, 제안 모델은 평균적으로 2위 방법보다 **11.2%p** 더 높다고 보고한다. 이는 단지 prompt flexibility만이 아니라, template-query transfer 구조 자체가 강력함을 시사하는 결과로 제시된다.

### Interactive segmentation capability

One-Prompt Model은 query image 자체를 template로 사용하면 interactive segmentation 형태로 축소될 수 있다. 저자들은 이 설정에서 `SAM`, `SAM-U`, `VMN`, `iSegFormer`, `MedSAM`, `MSA`, `SAM-Med2D`와 비교한다. `Click`과 `BBox` prompt만 사용하며, 필요 시 oracle prompt를 시뮬레이션했다고 설명한다. 중요한 점은 제안 모델은 이런 simulated prompt에 대해 별도 재학습 없이, 앞서 학습한 동일 모델을 그대로 사용했다는 것이다.

정량 결과는 Fig. 4에 제시되며, 논문은 One-Prompt Model이 다른 interactive segmentation 방법들보다 일관되게 높다고 주장한다. 즉, 이 모델은 본래 목표인 “template 하나로 task transfer”뿐 아니라, interactive segmentation의 특수 경우에서도 강하다는 것이다. 저자들의 해석은 더 어려운 학습 설정에서 훈련한 것이 오히려 더 강한 일반화로 이어졌다는 것이다.

### Zero-shot capability

이 실험에서는 SAM의 “segment everything”과 유사하게, template image에 규칙적인 foreground point grid를 찍어 이미지당 평균 50개 정도의 mask를 생성한다. 이 방식으로 제안 모델이 자동적으로 salient target들을 얼마나 잘 분할하는지 পরীক্ষা한다. 비교 대상은 prompt를 받지 않는 fully-supervised 모델 `TransUNet`, `Swin-UNETR`, `nnUNet`, `MedSegDiff`와, SAM 기반 의료 모델 `MSA`, `MedSAM`, `SAM-Med2D`이다.

11개의 unseen dataset에서의 Dice score 결과는 Table 1에 정리되어 있다. 평균 성능은 다음과 같다.

- `TransUNet`: 33.6
- `Swin-UNETR`: 28.5
- `nnUNet`: 34.9
- `MedSegDiff`: 35.3
- `MSA`: 50.3
- `MedSAM`: 53.9
- `SAM-Med2D`: 50.0
- `One-Prompt`: **64.0**

즉, 제안 모델은 평균 Dice 64.0%로 2위인 `MedSAM`의 53.9%보다 **10.1%p** 높고, 논문 본문에서는 second-highest 대비 **10.7%**라고 서술한다. 표와 본문 사이에 소수점 차이가 보이는데, 이는 반올림 또는 계산 기준 차이일 수 있으나 제공된 텍스트만으로는 정확한 이유를 단정할 수 없다.

작업별로 봐도 One-Prompt는 KiTS 67.3, ATLAS 63.8, WBC 72.5, SegRap 62.2, CrossM 65.8, REFUGE 58.4, Pendal 72.6, LQN 49.5, CAS 64.5, CadVidSet 66.3, ToothFairy 61.4로 모든 열에서 최고 성능을 기록한다. 이 결과는 “무엇을 분할해야 하는지”를 전혀 모르는 fully-supervised 모델보다, prompt를 통해 task 의미를 전달받는 범용 모델이 unseen task에서 훨씬 유리하다는 논문의 주장을 뒷받침한다.

### Ablation study

Prompt-Parser ablation에서는 Prompting step과 Masking step의 조합을 비교했다. Prompting에서는 embedding을 단순 합하는 방식과 concatenate 후 MLP로 projection하는 방식을 비교했고, Masking에서는 mask를 feature에 더하는 방식, binary masking, normalized masking 등을 실험했다. 결론은 **Stack MLP + Gaussian Masking**이 가장 높은 held-out Dice를 보였다는 것이다. 이는 제안된 Prompt-Parser의 두 핵심 설계가 실제 성능에 기여함을 보여준다.

### Template variance 분석

추론 시 서로 다른 template를 주었을 때 얼마나 결과가 흔들리는지도 측정했다. 8개 test task에 대해 서로 다른 랜덤 template로 100회 반복한 결과, Optic-Cup이나 Pancreas처럼 target 구조의 변이가 큰 작업일수록 분산이 컸다. 반면 `SegLab`처럼 더 세밀한 prompt에서는 분산이 작았다. 논문은 전체적으로 variance가 **13% 이하**로 유지되었다고 말하며, 이를 강건한 zero-shot generalization의 근거로 든다. 다만 13%라는 변동 폭이 실제 임상 적용에서 충분히 작은지 여부는 task의 위험도에 따라 다르게 해석될 수 있다.

### Prompt quality와 type의 영향

논문은 prompt quality를 `Low`, `Medium`, `High`, `Oracle`, `Human`의 다섯 단계로 나누어 REFUGE와 WBC에서 분석했다. 결과는 직관적이다. prompt quality가 높아질수록 모델 성능도 점진적으로 상승한다. 또한 `SegLab`이 가장 높은 성능을 보이고, `BBox`와 `Doodle`은 대체로 비슷한 수준이며, `Click`은 가장 빠르고 간단하지만 성능은 다소 낮다. 즉, **사용자 노력과 모델 성능 사이의 trade-off**가 존재함을 보여준다.

### 효율성 비교

Table 2는 one/few-shot 전이 모델과의 효율성 비교를 제시한다. 평균 성능과 사용자 비용을 함께 보면 One-Prompt의 실용적 의미가 더 잘 드러난다.

- `ALPNet`: Dice 52.96, user-cost time 27.47초
- `PANet`: Dice 50.11, user-cost time 27.47초
- `HyperSegNas`: Dice 63.86, user-cost time 27.47초
- `UniverSeg`: Dice 64.66, user-cost time 27.47초
- `OnePrompt`: Dice **73.98**, user-cost time **2.28초**
- `TransUNet (sup.)`: Dice 77.21, user-cost time $\infty$

저자들은 전체 라벨링에는 평균 27.47초가 걸리지만, prompt 하나는 평균 2.28초면 충분하다고 보고한다. 따라서 One-Prompt는 fully-supervised upper bound보다 성능은 약간 낮지만, annotation 비용과 task별 재학습 비용을 크게 줄이는 절충점이라고 주장한다. 본문은 TransUNet upper bound 대비 성능 저하가 **3.23%**라고 말한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정 자체가 실제 현장 요구와 잘 맞는다는 점이다. 각 테스트 이미지마다 prompt를 다시 넣어야 하는 interactive segmentation보다 자동화 가능성이 높고, dense support label이 필요한 few-shot 방식보다 비용이 낮다. 의료 영상처럼 새로운 기관, 새로운 장기, 새로운 modality가 계속 등장하는 환경에서는 이런 “한 번만 가볍게 알려주고 여러 장에 적용하는” 방식이 분명 실용적이다.

또 다른 강점은 아키텍처와 데이터 구성이 함께 설계되었다는 점이다. One-Prompt Former와 Prompt-Parser는 prompt-conditioned transfer를 위한 구조적 장치를 제공하고, 64개 학습 데이터셋과 3,000개 이상의 clinician-prompted sample은 다양한 의료 task를 포괄한다. 논문이 여러 종류의 prompt를 지원한 점도 의료 현장의 다양한 사용 습관을 반영한다는 면에서 설득력이 있다.

실험 측면에서도 강점이 있다. 논문은 few-shot, one-shot, interactive, zero-shot “segment everything” 설정을 모두 비교하며, 단일한 작업이 아니라 14개 hold-out task와 11개 unseen zero-shot task에서 성능을 제시한다. 이는 방법의 범용성을 강조하기에 적절한 평가 구성이다.

다만 한계도 분명하다. 첫째, 추론 시 기본적으로 **50회 template sampling 후 ensemble**을 사용한다는 점은 실제 임상 배포에서 latency와 compute cost를 높일 수 있다. 논문은 사용자 비용 절감은 잘 보여주지만, 실제 시스템 수준의 end-to-end throughput 관점에서는 추가 검토가 필요하다.

둘째, template choice에 따라 성능 분산이 존재한다. 저자들은 13% 이하라고 설명하지만, pancreas나 optic cup 같은 구조 다양성이 큰 task에서는 템플릿 민감도가 무시하기 어려울 수 있다. 즉, “딱 하나의 prompted sample”만으로 충분하다는 메시지는 평균적으로는 맞더라도, 개별 어려운 task에서는 template selection 전략이 중요할 수 있다.

셋째, prompt quality에 성능이 민감하다. 논문 스스로도 더 정교한 prompt일수록 성능이 좋아진다고 보였다. 이는 사용자가 빠르게 대충 넣은 prompt로도 항상 안정적으로 동작하는 것은 아님을 뜻한다. 결국 One-Prompt는 annotation burden을 크게 줄이지만, 완전히 무비용인 것은 아니다.

넷째, 제공된 본문 기준으로는 몇몇 구현 세부사항이 충분히 명확하지 않다. 예를 들어 encoder의 최종 선택, `SegLab`용 autoencoder 구조, Prompt-Parser 내부 연산의 정확한 tensor semantics, training hyperparameter 등은 본문만으로는 완전히 재현하기 어렵다. 저자들이 appendix에 구현 세부를 둔다고 말하지만, 현재 제공된 텍스트만으로는 모두 확인할 수 없다.

다섯째, 데이터셋 분할과 generalization 주장에 대해서는 약간의 주의가 필요하다. 논문은 78개 데이터셋 중 64개로 학습하고 14개를 테스트했다고 하나, “완전히 새로운 기관/장비/인구집단”에 대한 domain shift까지 얼마나 포괄하는지는 제공된 텍스트만으로 판단할 수 없다. 따라서 이 논문의 zero-shot은 “held-out dataset generalization”에는 강하지만, 실제 임상 배포에서의 external validation까지 보장한다고 보기는 어렵다.

## 6. 결론

이 논문은 의료 영상 분할에서 **interactive segmentation의 유연성**과 **one-shot transfer의 task adaptation 능력**을 결합한 **One-Prompt Segmentation**이라는 새로운 범용 패러다임을 제안한다. 사용자는 새로운 task마다 단 하나의 template image에 prompt만 제공하면 되고, 모델은 재학습 없이 query image들을 분할할 수 있다. 이를 위해 저자들은 One-Prompt Former와 Prompt-Parser를 포함한 아키텍처를 설계했고, 78개 공개 데이터셋과 3,000개 이상의 clinician prompt로 대규모 학습/평가 체계를 구성했다.

실험 결과는 이 방법이 unseen medical segmentation task에 대해 매우 강한 일반화 능력을 가진다는 점을 보여준다. few/one-shot 모델, interactive 모델, SAM 기반 medical adaptation, 그리고 일반 fully-supervised 모델과 비교해도 전반적으로 우수한 성능을 보였고, 특히 사용자 비용 대비 성능 효율이 높다.

실제 적용 측면에서 이 연구는 매우 중요하다. 의료 AI 시스템이 현장에 들어가려면 task마다 새로 학습하거나 매 이미지마다 전문가가 개입하는 방식은 확장성이 떨어진다. One-Prompt 방식은 이 둘 사이의 실용적인 대안을 제시한다. 앞으로 template selection 전략, prompt robustness, 3D volumetric setting에서의 최적화, 실제 병원 외부 검증이 더해진다면, 이 연구는 범용 의료 영상 foundation model 방향에서 의미 있는 기반이 될 가능성이 크다.
