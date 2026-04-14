# Learning Multi-Domain Convolutional Neural Networks for Visual Tracking

- **저자**: Hyeonseob Nam, Bohyung Han
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1510.07945

## 1. 논문 개요

이 논문은 visual tracking을 위해 CNN을 직접 학습하는 방법을 제안한다. 핵심 문제의식은 기존 CNN 기반 tracker들이 대체로 ImageNet 같은 image classification 데이터로 사전학습된 특징을 가져다 쓰는데, classification과 tracking은 본질적으로 다르다는 점이다. Classification은 미리 정해진 class label을 맞히는 문제이지만, tracking은 임의의 target을 프레임마다 정확히 localize해야 한다. 따라서 classification용 표현이 tracking에 충분히 최적화되어 있지 않다는 것이 논문의 출발점이다.

저자들은 이 문제를 해결하기 위해 여러 비디오 시퀀스를 각각 서로 다른 domain으로 보고, domain마다 target/background binary classification을 수행하되 앞단의 feature extractor는 공유하는 구조를 설계한다. 이를 통해 시퀀스마다 다른 target class, appearance, motion, background, occlusion 등의 차이를 분리하면서도, tracking에 공통으로 필요한 representation을 학습하려고 한다.

이 문제가 중요한 이유는 visual tracking이 다양한 실제 환경에서 핵심적인 비전 문제이기 때문이다. 조명 변화, occlusion, scale change, deformation, motion blur 같은 조건에서 안정적으로 동작하는 표현을 학습하는 것은 쉽지 않다. 저자들은 바로 이 “sequence-specific 차이”와 “tracking 공통성”을 동시에 다루는 학습 구조가 필요하다고 본다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 각 학습 시퀀스를 하나의 독립적인 domain으로 취급하고, shared layers와 domain-specific classification branch를 분리한 Multi-Domain Network, 즉 MDNet을 학습하는 것이다. 이 구조에서 shared layers는 모든 시퀀스에 공통적인 target representation을 배우고, 마지막 분기층은 각 시퀀스에서만 유효한 target/background 판별을 담당한다.

이 설계의 직관은 다음과 같다. 어떤 시퀀스에서는 특정 물체가 target이지만, 다른 시퀀스에서는 같은 종류의 물체가 background일 수 있다. 따라서 모든 데이터를 하나의 단일 binary classifier로 섞어 학습하면 target의 정의가 충돌할 수 있다. 반면 domain별 마지막 분기를 분리하면, 각 시퀀스의 target 정의는 branch가 담당하고, 그보다 앞선 shared layers는 “tracking에 유용한 일반적 시각 표현”을 학습할 수 있다.

기존 접근과의 차별점은 크게 두 가지다. 첫째, classification용 대규모 데이터가 아니라 tracking ground truth가 있는 비디오 데이터로 표현을 학습한다. 둘째, 여러 시퀀스를 단순히 합쳐 하나의 네트워크를 학습하는 것이 아니라, multi-domain 구조를 통해 domain-independent 정보와 domain-specific 정보를 분리한다. 논문은 이 점이 tracking 특화 표현 학습에 더 적절하다고 주장한다.

또 하나 중요한 차별점은 네트워크를 일부러 작게 설계했다는 점이다. 저자들은 tracking이 일반 object recognition보다 class 수가 훨씬 적고, precise localization이 중요하며, target이 작은 경우가 많고, 온라인 업데이트까지 필요하므로 AlexNet이나 VGG처럼 큰 네트워크보다 얕고 가벼운 구조가 더 적합하다고 본다.

## 3. 상세 방법 설명

전체 구조는 107×107 RGB 입력을 받는 CNN이다. hidden layer는 `conv1-3`의 세 개 convolutional layer와 `fc4-5`의 두 개 fully connected layer로 구성되며, 마지막에는 각 domain마다 하나씩 존재하는 `fc6^1, ..., fc6^K` branch가 붙는다. 각 branch는 해당 시퀀스에서 target과 background를 구분하는 binary classification layer이며, softmax cross-entropy loss를 사용한다. 즉, `fc6` 계층만 domain-specific이고, 그 이전의 `conv1-3, fc4-5`는 shared layers다.

오프라인 학습에서는 SGD를 사용하되, 한 iteration마다 하나의 domain만 활성화한다. 즉, `k`번째 iteration에서는 `(k mod K)`번째 시퀀스에 해당하는 branch만 켜고, 그 시퀀스에서 뽑은 minibatch로 업데이트한다. 이렇게 하면 마지막 분기층은 각 시퀀스의 개별 target 정의를 학습하고, shared layers는 모든 시퀀스를 오가며 공통 표현을 축적하게 된다. 논문은 이 반복적 multi-domain 학습을 통해 shared layers가 domain-independent representation을 갖게 된다고 설명한다.

테스트 시에는 학습에 쓰인 여러 branch를 모두 제거하고, 새로운 시퀀스용 단일 `fc6` branch를 새로 만든다. 이후 tracking 중에는 이 새 branch와 shared network 내부의 fully connected layers인 `fc4-5`를 온라인으로 fine-tuning한다. 반면 convolutional layers `conv1-3`는 고정한다. 저자들은 이렇게 해야 계산량을 줄이고, 오프라인에서 학습한 일반 표현이 과도하게 훼손되는 것을 막을 수 있다고 본다.

프레임 `t`에서 target state를 추정하는 방식은 비교적 단순하다. 이전 프레임의 target state 주변에서 후보 박스 `x_1, ..., x_N`을 샘플링하고, 각 후보에 대해 네트워크가 positive score $f_+(x_i)$와 negative score $f_-(x_i)$를 출력한다. 최종 상태는 positive score가 가장 큰 후보로 정한다.

$$
x^* = \arg\max_{x_i} f_+(x_i)
$$

즉, detection-like scoring을 현재 프레임마다 수행하는 tracking-by-detection 형태다.

온라인 업데이트는 long-term update와 short-term update로 나뉜다. Long-term update는 일정 간격마다 오랜 기간 모은 positive sample로 수행하여 robustness를 확보하고, short-term update는 추적 실패 가능성이 감지될 때 빠르게 수행하여 adaptiveness를 높인다. 실패 가능성은 추정된 target의 positive score가 낮을 때로 정의되며, 논문에서는 $f_+(x_t^*) < 0.5$일 때 short-term update를 수행한다. 반대로 추적이 안정적이면 일정 주기마다 long-term update를 수행한다. 구체적으로 short-term memory의 길이는 $\tau_s = 20$, long-term memory의 길이는 $\tau_l = 100$으로 둔다.

하드 네거티브 마이닝도 중요한 구성 요소다. tracking에서는 negative sample 대부분이 너무 쉬운 background라서 학습에 큰 도움이 되지 않는다. 그래서 각 iteration에서 많은 negative 후보 중 네트워크가 positive로 착각하기 쉬운 샘플만 골라 minibatch를 구성한다. 구체적으로 minibatch는 `M+`개의 positive와 `M_h^-`개의 hard negative로 구성되며, 이 hard negative는 `M^-`개의 negative 후보를 먼저 평가한 뒤 positive score가 가장 높은 것들로 선택한다. 논문에 제시된 값은 `M+ = 32`, `M^- = 1024`, `M_h^- = 96`이다. 이 절차는 drift를 줄이는 데 중요하다고 설명된다.

Bounding box regression도 추가된다. CNN feature의 high-level abstraction과 positive augmentation 때문에 네트워크가 target을 포함하는 “대략적 위치”는 잘 찾지만, box를 타이트하게 맞추지 못하는 경우가 있기 때문이다. 그래서 첫 프레임에서 target 근처 샘플들의 `conv3` feature를 사용해 선형 bounding box regressor를 학습하고, 이후 신뢰도 높은 경우, 즉 $f_+(x^*) > 0.5$일 때만 예측 박스를 보정한다. 이 회귀기는 첫 프레임에서만 학습하며, 이후 온라인으로 다시 학습하지는 않는다. 저자들은 incremental regression이 계산비용에 비해 큰 도움을 주지 않을 수 있다고 본다.

샘플링과 학습 설정도 비교적 구체적으로 제시된다. 타깃 후보는 이전 상태를 평균으로 하는 Gaussian에서 256개를 샘플링하며, 평행이동 분산은 target 크기 평균 $r$에 비례하는 `diag(0.09r^2, 0.09r^2)`이고 scale 쪽 분산은 `0.25`이다. 오프라인 학습에서는 프레임당 positive 50개, negative 200개를 추출하며, positive는 IoU가 0.7 이상, negative는 0.5 이하이다. 온라인 학습에서는 보통 positive 50개, negative 200개를 사용하지만 첫 프레임에서는 positive 500개, negative 5000개를 사용한다. 온라인에서 positive는 IoU 0.7 이상, negative는 0.3 이하를 만족한다.

오프라인 multi-domain learning은 총 `100K` iteration 동안 수행하고, convolutional layer에는 learning rate 0.0001, fully connected layer에는 0.001을 사용한다. 새 시퀀스 첫 프레임에서는 `fc4-6`을 30 iteration fine-tuning하고, 이후 온라인 업데이트에서는 이보다 3배 큰 learning rate로 10 iteration 학습한다. Momentum은 0.9, weight decay는 0.0005다.

## 4. 실험 및 결과

논문은 OTB와 VOT2014 두 벤치마크에서 성능을 평가한다. 구현은 MATLAB과 MatConvNet으로 했고, 8코어 Intel Xeon E5-2660 및 NVIDIA Tesla K20m 환경에서 약 1 fps로 동작한다고 보고한다. 즉, 정확도는 높지만 실시간 tracker는 아니다.

OTB에서는 OTB50과 OTB100을 사용한다. 평가는 center location error와 bounding box overlap ratio를 기반으로 하며, one-pass evaluation(OPE)을 적용한다. 비교 대상은 MUSTer, CNN-SVM, MEEM, TGPR, DSST, KCF, SCM, Struck 등이다. 특히 CNN-SVM은 당시 CNN representation을 사용하는 또 다른 tracker라서 중요한 비교 기준이다.

MDNet의 오프라인 사전학습에는 VOT2013, VOT2014, VOT2015에서 모은 58개 시퀀스를 사용하되, OTB100과 겹치는 영상은 제외한다. OTB50 결과에서 MDNet은 precision 0.948, success AUC 0.708을 기록해 MUSTer, CNN-SVM, MEEM 등 모든 비교 방법보다 높다. OTB100에서도 precision 0.909, success AUC 0.678로 가장 높은 성능을 보인다. 논문은 이 결과를 통해 MDNet이 target을 놓치는 일이 적고, 동시에 박스 localization도 정확하다고 해석한다.

속성별 분석도 제시된다. Fast motion, background clutter, illumination variation, in-plane rotation, low resolution, occlusion, out of view, scale variation 등 여러 challenge attribute에서 success plot을 비교한 결과, MDNet이 전반적으로 가장 높다. 특히 low resolution 조건에서 hand-crafted feature 기반 tracker들이 약한 반면, MDNet은 상대적으로 강한 성능을 보인다. 이는 고수준 시각 표현이 작은 대상이나 어려운 장면에서도 도움이 된다는 논문의 주장과 연결된다.

Ablation study도 수행한다. 저자들은 세 가지 변형을 비교한다. 첫째, multi-domain이 아니라 모든 데이터를 하나의 branch로 학습한 SDNet. 둘째, bounding box regression을 제거한 MDNet-BB. 셋째, bounding box regression과 hard negative mining을 모두 제거한 MDNet-BB-HM이다. OTB100 기준으로 full MDNet이 precision 0.909, success 0.678이고, MDNet-BB는 0.891 / 0.650, SDNet은 0.865 / 0.645, MDNet-BB-HM은 0.816 / 0.602다. 이 결과는 multi-domain learning, bounding box regression, hard negative mining이 각각 실제로 기여한다는 점을 보여준다.

정성적 결과에서도 Bolt2, ClifBar, Diving, Freeman4, Human5, Ironman, Matrix, Skating2-1 등의 시퀀스에서 다른 tracker보다 안정적으로 추적하는 예를 제시한다. 반면 실패 사례도 공개하는데, `Coupon`에서는 미세한 appearance change로 drift가 발생하고, `Jump`에서는 급격한 appearance change 때문에 target을 완전히 놓친다. 즉, 논문은 자신의 방법이 강력하지만 극심한 외형 변화에는 여전히 취약할 수 있음을 보여준다.

VOT2014에서는 25개 시퀀스를 사용하고, baseline과 region noise 두 실험 설정을 평가한다. 이 벤치마크는 tracker가 실패하면 다시 초기화되며, accuracy와 robustness를 따로 측정한다. Accuracy는 bounding box overlap, robustness는 failure 횟수와 관련된다. 비교 대상은 VOT2014 상위 tracker들인 DSST, SAMF, KCF, DGT, PLT14와 추가로 MUSTer, MEEM이다.

VOT2014용 사전학습에는 OTB100 중 VOT2014와 겹치지 않는 89개 시퀀스를 사용한다. Baseline 실험에서 MDNet은 accuracy 0.63으로 최고 점수를 기록하고, robustness score도 0.16으로 매우 우수하며, 종합 순위에서도 최상위권이다. Region noise 실험에서도 MDNet은 accuracy 0.60, robustness 0.30으로 여전히 높은 성능을 유지한다. 논문은 이를 근거로, 초기화가 다소 부정확해도 MDNet이 잘 견디며 향후 re-detection module과 결합하면 long-term tracking에도 유망할 수 있다고 해석한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 tracking을 classification의 단순 전이 문제로 보지 않고, tracking 자체의 구조를 반영한 representation learning 방법을 제시했다는 점이다. 특히 각 시퀀스를 독립 domain으로 두고 shared/domain-specific 파라미터를 분리한 설계는 매우 설득력이 있다. 단순히 여러 비디오를 한데 섞어 학습할 때 발생하는 target 정의 충돌 문제를 구조적으로 해결하려는 접근이기 때문이다.

또 다른 강점은 오프라인 학습과 온라인 적응을 자연스럽게 결합했다는 점이다. 오프라인에서는 generic representation을 배우고, 온라인에서는 fully connected layers만 빠르게 적응시켜 새 시퀀스의 domain-specific 정보를 흡수한다. 여기에 long-term/short-term update, hard negative mining, bounding box regression을 조합해 정확도와 안정성을 동시에 끌어올린 점도 실용적이다.

실험 역시 당시 기준으로 매우 강하다. OTB와 VOT2014 두 대표 벤치마크에서 모두 강한 성능을 보였고, 단순한 전체 평균뿐 아니라 attribute별 분석과 ablation study도 제공한다. 특히 SDNet 대비 MDNet의 우수성을 직접 보여줬다는 점은 핵심 주장인 multi-domain learning의 타당성을 뒷받침한다.

한계도 분명하다. 첫째, 속도가 약 1 fps로 보고되어 실시간 tracking에는 부적합하다. 논문도 정확도는 매우 높지만 계산 효율 측면의 한계를 숨기지 않는다. 둘째, 첫 프레임에서 많은 sample을 모아 온라인 fine-tuning을 수행하고, tracking 중에도 주기적으로 업데이트해야 하므로 계산비용이 크다. 셋째, `Coupon`, `Jump` 사례에서 보이듯 appearance가 급격히 변하거나 drift가 누적되면 실패할 수 있다.

또 하나의 해석 가능한 한계는, convolutional layers를 고정하고 fully connected layers만 적응시키는 전략이 overfitting 방지에는 유리하지만, 매우 큰 도메인 변화에는 표현 자체의 유연한 재구성이 부족할 수 있다는 점이다. 다만 이것은 논문이 직접 “한계”라고 길게 정리한 부분은 아니므로, 실험 결과와 설계 선택에 근거한 비판적 해석으로 보는 것이 적절하다.

마지막으로, 논문은 tracking ground truth가 있는 대규모 비디오 데이터의 중요성을 강조하지만, 어떤 종류의 데이터 다양성이 성능 향상에 가장 크게 기여하는지까지는 세밀하게 분석하지 않는다. 또한 온라인 업데이트 안정성에 대한 이론적 보장보다는 경험적 성능에 주로 의존한다.

## 6. 결론

이 논문은 visual tracking을 위한 CNN representation learning 문제를 multi-domain learning으로 재정의하고, 이를 구현한 MDNet을 제안한다. 핵심 기여는 각 시퀀스를 별도 domain으로 다루는 네트워크 구조, 오프라인에서의 domain-independent representation 학습, 그리고 테스트 시 online fine-tuning을 통한 domain adaptation의 결합이다.

실험적으로도 MDNet은 OTB와 VOT2014에서 당시 state-of-the-art를 능가하는 강한 성능을 보였다. 특히 단순히 CNN feature를 가져다 쓰는 수준이 아니라, tracking 문제에 맞는 학습 구조를 설계해야 한다는 점을 분명히 보여준다는 점에서 의미가 크다.

실제 적용 측면에서는 계산량이 큰 편이라 속도 제한이 있지만, 정확도 중심 tracker의 중요한 기준점을 세운 연구라고 볼 수 있다. 향후 연구 관점에서는 Siamese tracker, transformer tracker, online adaptation 기반 tracker 등 이후 계열의 발전에도 영향을 준 중요한 전환점으로 해석할 수 있다. 다만 이 마지막 영향 평가는 논문 본문에 직접 쓰인 내용이라기보다, 논문이 제시한 문제 설정과 성과를 바탕으로 한 학술적 해석이다.
