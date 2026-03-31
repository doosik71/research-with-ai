# VisDA: The Visual Domain Adaptation Challenge

* **저자**: Xingchao Peng, Ben Usman, Neela Kaushik, Judy Hoffman, Dequan Wang, Kate Saenko
* **발표연도**: 2017
* **arXiv**: [https://arxiv.org/abs/1710.06924](https://arxiv.org/abs/1710.06924)

## 1. 논문 개요

이 논문은 새로운 domain adaptation 알고리즘 하나를 제안하는 논문이라기보다, **visual unsupervised domain adaptation(UDA)** 연구를 위한 대규모 benchmark와 challenge인 **VisDA2017**을 소개하는 논문이다. 저자들은 기존 domain adaptation 벤치마크들이 대체로 데이터 규모가 작고, domain shift가 약하며, task 다양성이 부족하다는 문제를 지적한다. 특히 실제 배치 환경에서는 train과 test가 같은 분포라는 가정이 자주 깨지는데, 기존 컴퓨터 비전 챌린지들은 대개 같은 분포 안에서 train/test를 나누어 평가해 왔다고 본다.

이 논문이 다루는 핵심 연구 문제는 다음과 같다. **라벨이 있는 source domain에서 학습한 시각 모델을, 라벨이 없는 target domain으로 얼마나 잘 적응시킬 수 있는가?** 더 구체적으로는, synthetic 혹은 simulated 이미지에서 학습한 모델을 real 이미지로 옮기는 **simulation-to-reality shift**를 다룬다. 이는 로봇, 자율주행, 의료영상처럼 실제 데이터의 annotation 비용이 매우 큰 분야에서 특히 중요하다. synthetic 데이터는 렌더링 파이프라인만 갖추면 대규모로 만들 수 있지만, 실제 deployment 환경의 real 데이터와는 시각적 통계가 달라 성능 저하가 발생한다.

VisDA2017은 이 문제를 두 가지 task로 구성한다. 하나는 **image classification**이고, 다른 하나는 **semantic segmentation**이다. 두 task 모두에서 source는 labeled synthetic 데이터이고, target은 unlabeled real 데이터이다. 또한 validation target과 test target을 서로 다른 도메인으로 분리해, target label을 사용한 hyperparameter tuning이라는 비현실적인 평가 관행을 줄이려 한다. 이 점은 단순히 데이터셋 규모만 키운 것이 아니라, **UDA 문제 정의에 더 충실한 평가 프로토콜**을 설계했다는 의미가 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “더 좋은 adaptation 방법”을 직접 제안하는 것이 아니라, **더 어려우면서도 현실적인 adaptation benchmark를 제공함으로써 연구를 촉진하자**는 데 있다. 저자들은 domain adaptation 연구가 기존 소규모 데이터셋에서는 어느 정도 성숙해졌고, 특히 classification 위주의 제한된 설정에서는 실제 어려움을 충분히 반영하지 못한다고 본다. 따라서 VisDA는 다음 세 가지를 동시에 강화하려 한다.

첫째, **규모(scale)** 이다. 분류용 VisDA-C는 train, validation, test를 합쳐 28만 장이 넘는 이미지를 포함하고, segmentation용 VisDA-S 역시 3만 장이 넘는 이미지를 포함한다. 이는 당시 cross-domain benchmark 중 매우 큰 편이다.

둘째, **강한 domain shift** 이다. Office처럼 센서 차이 정도의 비교적 약한 shift가 아니라, CAD rendering 혹은 game engine 기반 synthetic 이미지와 COCO, YouTube-BB, CityScapes, Nexar 같은 real 이미지 사이의 차이를 사용한다. 이는 texture, background, resolution, viewpoint, annotation 방식 등 여러 차이점이 복합적으로 작용하는 더 어려운 문제다.

셋째, **평가 프로토콜의 현실성** 이다. 많은 UDA 연구가 사실상 target test label에 의존해 model selection을 하는데, VisDA는 target validation domain과 target test domain을 분리해 이 문제를 완화한다. 즉, 참가자는 source와 unlabeled validation/test target으로 적응 모델을 만들되, test domain은 validation domain과 또 다른 도메인이므로 특정 target에 과적합된 방법은 일반화에 실패할 수 있다.

기존 접근법과의 차별점은 논문이 명확히 말하듯, 단순히 classification용 cross-domain set을 하나 더 추가한 수준이 아니라, **classification과 segmentation을 모두 포함한 대규모 synthetic-to-real UDA benchmark**를 제공하고, challenge 결과와 baseline을 함께 공개했다는 점이다. 또한 source-only, oracle, baseline adaptation, challenge 우승 결과를 한 자리에서 비교해, domain gap의 크기와 현재 기법의 한계를 수치로 보여준다.

## 3. 상세 방법 설명

이 논문에서 “방법”은 크게 두 층위로 이해해야 한다. 첫 번째는 **데이터셋 및 평가 프로토콜 설계 방법**, 두 번째는 **baseline adaptation 알고리즘과 challenge 참가 방법에 대한 설명**이다. 이 논문이 제안하는 주된 공헌은 전자에 있다.

### 3.1 전체 파이프라인과 문제 설정

전체 문제는 다음과 같이 정의된다. labeled source domain 데이터 $(x_s, y_s)$ 가 있고, unlabeled target domain 데이터 $x_t$ 가 있다. 목표는 source supervision을 이용해 학습한 모델이 target에서 잘 작동하도록 만드는 것이다. 논문은 이 설정을 **unsupervised domain adaptation**으로 제한하며, target label은 training에 사용하지 않는다.

평가 프로토콜은 다음처럼 구성된다.

* classification track:

  * source train: synthetic CAD rendering
  * target validation: real COCO crop
  * target test: real YouTube-BB crop
* segmentation track:

  * source train: synthetic GTA5
  * target validation: real CityScapes
  * target test: real Nexar/BDD

핵심은 validation target과 test target이 서로 다른 domain이라는 점이다. 이는 “target unlabeled data를 adaptation에 쓰되, 특정 target에만 맞춘 tuning은 일반화가 불안정할 수 있다”는 현실을 반영한다.

### 3.2 VisDA-C: 분류 데이터셋 설계

VisDA-C는 12개 object category를 공통 클래스로 사용한다. 카테고리는 aeroplane, bicycle, bus, car, horse, knife, motorcycle, person, plant, skateboard, train, truck이다.

#### Source domain: CAD-Synthetic

source domain은 3D CAD 모델을 렌더링해 만든 synthetic 이미지다. 총 1,907개의 3D 모델에서 152,397장의 이미지를 생성했다. 데이터 생성 시 저자들은 다음 요소를 조절했다.

* 20개의 서로 다른 camera yaw/pitch 조합
* 4개의 서로 다른 light direction
* ambient와 sun light를 1:3 비율로 사용
* object의 rotation, scale, translation 자동 보정
* 전체 object가 frame 안에 충분한 margin을 갖고 들어오도록 camera 배치
* textured model뿐 아니라 plain grey albedo를 가진 un-textured version도 렌더링

이 설계는 synthetic 데이터가 다양한 viewpoint와 illumination을 갖도록 하면서도, class별 데이터 균형을 어느 정도 유지하게 한다.

#### Validation domain: MS COCO

validation target은 COCO의 bounding box annotation을 활용해 object crop을 추출해 만든다. 각 crop에는 원 bounding box 주변으로 추가 문맥을 주기 위해 대략 높이와 너비 기준 50% 정도의 padding을 포함시킨다. 너무 작은 patch는 제외하여 극단적인 resize 왜곡을 줄인다. 최종적으로 55,388장의 object image를 모았고, person 클래스는 과도하게 많아서 4,000장으로 줄여 전체 균형을 맞췄다.

#### Test domain: YouTube-BB

test target은 YouTube Bounding Boxes(YT-BB)에서 추출한 frame crop이다. 총 72,372장이며, 비디오 프레임 기반이라 COCO보다 해상도가 낮고 blur나 compression artifact가 많을 가능성이 크다. 따라서 validation과 test 모두 real domain이지만 시각적 특성이 다르다.

### 3.3 VisDA-S: 분할 데이터셋 설계

semantic segmentation track은 synthetic GTA5와 real CityScapes/Nexar 간 adaptation을 다룬다.

* source: GTA5, 24,966장의 고해상도 synthetic street scene
* validation: CityScapes, 5,000장의 real urban dashcam image 중 19개 공통 클래스 사용
* test: Nexar/BDD, 1,500장의 real dashcam image, 19개 공통 클래스

segmentation은 각 픽셀에 라벨을 부여해야 하므로 annotation 비용이 매우 높다. 저자들은 이 점 때문에 synthetic-to-real adaptation이 특히 중요하다고 본다.

### 3.4 Baseline 학습 절차와 모델

분류 baseline에서는 주로 **AlexNet**과 일부 **ResNext-152**를 사용한다. AlexNet의 마지막 layer는 12-way classification용 fully connected layer로 바꾸고, 이 layer는 $\mathcal{N}(0, 0.01)$ 로 초기화한다. 학습은 mini-batch SGD를 사용하며, learning rate는 $10^{-3}$, weight decay는 $5 \times 10^{-4}$, momentum은 0.9로 설정한다.

ResNext-152에서는 마지막 fully connected layer의 출력 차원을 12로 바꾸고 Xavier initializer를 사용한다. 새로 학습하는 출력층은 다른 layer보다 learning rate를 10배 크게 둔다. learning rate schedule은 다음과 같이 주어진다.

$$
\eta_p = \frac{\eta_0}{(1 + \alpha p)^\beta}
$$

여기서 $p$ 는 학습 진행률로 0에서 1까지 선형 증가하고, $\eta_0 = 10^{-4}$, $\alpha = 10$, $\beta = 0.75$ 이다. 이 식은 training이 진행될수록 learning rate를 점차 감소시키는 standard한 schedule이다.

### 3.5 Baseline domain adaptation 알고리즘

논문은 새로운 adaptation loss를 제안하지 않고, 기존 대표 기법 두 개를 baseline으로 사용한다.

#### DAN (Deep Adaptation Network)

DAN은 source와 target의 feature distribution을 가깝게 만들기 위해 **Maximum Mean Discrepancy(MMD)** 기반 손실을 사용한다. 논문 본문에는 상세 수식이 직접 전개되어 있지는 않지만, 핵심은 서로 다른 domain의 feature embedding이 RKHS 상에서 비슷한 평균 표현을 갖도록 맞추는 것이다. 직관적으로는 “source feature와 target feature의 분포 차이”를 줄여 transferable representation을 학습한다.

#### Deep CORAL

Deep CORAL은 평균보다 더 나아가 **second-order statistics**, 즉 covariance를 맞춘다. 논문에 제시된 domain discrepancy는 다음과 같다.

$$
d(S, T) = \lVert \operatorname{Cov}_S - \operatorname{Cov}_T \rVert_F^2
$$

여기서 $\operatorname{Cov}_S$ 와 $\operatorname{Cov}_T$ 는 각각 source와 target feature의 covariance matrix이고, $\lVert \cdot \rVert_F^2$ 는 Frobenius norm의 제곱이다. 쉽게 말하면, 두 domain의 feature가 가지는 상관구조를 비슷하게 만들어 domain shift를 줄이려는 것이다.

### 3.6 Challenge 우승 방법에 대한 설명

이 논문은 challenge 결과도 함께 보고한다. classification 우승팀인 **GFColourLabUEA**는 self-ensembling 계열 접근을 사용했다. 설명에 따르면 student network와 teacher network를 두고, teacher의 weight는 student weight의 exponential moving average(EMA)로 만든다. 이 방식은 mean teacher와 $\Pi$-model 계열 semi-supervised learning 아이디어를 domain adaptation에 적용한 것이다.

이 팀의 목적 함수는 크게 두 항으로 설명된다.

1. source domain에서 ground-truth label과 student prediction 사이의 mean cross-entropy
2. source와 target 전체 샘플에서 student와 teacher prediction 간 mean squared difference

즉, source supervision은 유지하면서, teacher와 student의 예측이 다양한 noise, dropout, augmentation 아래에서도 일관되도록 강제한다. 이것은 unlabeled target에서도 consistency regularization을 통해 더 안정적인 decision boundary를 형성하려는 접근으로 이해할 수 있다.

segmentation 우승팀은 multi-stage 절차를 사용한다. 첫 단계에서는 frame-level discriminator를 사용하여 target image를 source와 비슷한 전역적 스타일로 맞추려 하고, 두 번째 단계에서는 pixel-level discrimination과 여러 backbone ensemble을 사용해 더 강한 feature representation을 구축한다. 다만 이 부분은 challenge 결과 요약 수준으로 제시되며, 구체적인 완전 수식이나 학습 세부 절차는 이 논문에 충분히 상세히 쓰여 있지 않다.

## 4. 실험 및 결과

## 4.1 Classification 실험 설정과 지표

classification에서는 클래스별 accuracy를 구하고, 이를 평균한 **mean accuracy**를 보고한다. baseline에서는 in-domain oracle, source-only, adaptation method를 비교한다.

* **Oracle (synthetic→synthetic, 혹은 real→real)**: 같은 domain 안에서 supervised training/testing
* **Source-only**: synthetic source로만 학습하고 real target에 바로 적용
* **Adaptation**: source + unlabeled target을 사용하여 adaptation

이 비교는 domain shift의 절대 크기와 adaptation의 실제 효과를 동시에 보여준다.

### Validation domain (synthetic → COCO)

AlexNet 기준 source-only 성능은 **28.12%**이다. 같은 synthetic domain에서 supervised oracle은 **99.92%**, real validation domain에서 supervised oracle은 **87.26%**이다. 즉, 모델 자체가 완전히 무능한 것이 아니라, **domain shift 때문에 성능이 크게 붕괴**한다는 점이 분명해진다.

baseline adaptation 결과는 다음과 같다.

* Source-only (AlexNet): **28.12%**
* D-CORAL: **45.53%**
* DAN: **51.62%**

즉, D-CORAL은 약 17.4%p, DAN은 약 23.5%p 정도 절대 성능 향상을 만든다. 논문은 이를 relative gain으로도 표현하는데, DAN은 source-only 대비 **83.6%** 상대 향상, D-CORAL은 **61.91%** 상대 향상으로 보고한다.

카테고리별로 보면 bus, aeroplane, motorcycle 같은 클래스는 비교적 잘 적응되는 반면, car, person, skateboard 등은 더 어렵다. 특히 source-only에서 bicycle, person, truck 등 일부 클래스가 매우 낮은 성능을 보여, synthetic 데이터가 real 이미지의 배경, 자세, appearance 다양성을 충분히 반영하지 못함을 시사한다.

### Test domain (synthetic → YT-BB)

test domain에서도 source-only AlexNet은 **30.81%**이고, DAN은 **49.78%**, D-CORAL은 **45.29%**이다. 흐름은 validation과 유사하지만, 세부 클래스별 성능 패턴은 domain 특성 차이 때문에 다르다. 예를 들어 YT-BB는 video frame 기반이라 저해상도, motion blur, tracking artifact 등이 있을 수 있어 COCO와는 다른 형태의 난점을 제공한다.

oracle 성능은 real→real AlexNet에서 **92.08%**, ResNext-152에서 **93.40%**이다. 따라서 test에서도 adaptation의 성과는 크지만, source-only와 oracle 사이에는 여전히 상당한 gap이 남아 있다.

### Challenge classification 결과

challenge 우승 결과는 매우 인상적이다.

* GF_ColourLab_UEA: **92.8%**
* NLE_DA: **87.7%**
* BUPT_OVERFIT: **85.4%**

특히 GF_ColourLab_UEA는 source-only ResNet-152의 **45.3%**에서 **92.8%**로 끌어올려, 사실상 oracle 수준에 가깝게 도달했다. 이는 benchmark가 너무 쉬운 것 아니냐는 질문을 낳을 수 있는데, 저자들도 바로 그 점을 인식하고 이후 더 어려운 버전인 **VisDA-C-ext**와 no-pretraining setting 등을 제안한다.

## 4.2 VisDA-C-ext와 난이도 증가 실험

저자들은 원래 real domain에서 제외했던 작은 이미지들을 다시 포함해 **VisDA-C-ext**를 만든다. 추가된 데이터는 COCO domain에 35,591장, YT-BB domain에 4,533장이다. 이렇게 하면 object scale이 더 작고 shape가 irregular한 샘플이 들어가 문제 난도가 올라간다.

결과는 실제로 더 어려워졌음을 보여준다.

* AlexNet: val 28.12 → 22.10, test 30.81 → 28.56
* ResNext-152: val 41.21 → 26.28, test 38.62 → 36.98
* ResNext-152 + JMMD: val 64.54 → 47.68, test 58.37 → 54.73

모든 모델이 VisDA-C-ext에서 성능이 하락하며, 특히 validation domain인 MS COCO에서 하락폭이 더 크다. 이는 COCO 쪽이 오히려 YT-BB보다 더 어려울 수 있음을 시사한다.

저자들은 training loss와 accuracy가 iteration에 따라 크게 흔들린다는 점도 지적한다. target label이 없는 UDA에서는 언제 멈춰야 하는지 알기 어려워, stopping criterion 자체가 중요한 연구 문제가 된다.

## 4.3 Semantic segmentation 결과

segmentation에서는 **IoU**와 평균값인 **mIoU**를 사용한다. 이는 semantic segmentation의 표준 지표다.

### GTA5 → CityScapes validation

* Source (Dilation Front End): **21.4 mIoU**
* Oracle (same-domain supervised): **64.0 mIoU**

본문 서술에서는 adaptation method in [16]가 source 대비 향상된 성능을 낸다고 하며, 본문 일부에서는 25.5 혹은 27.1 mIoU 관련 언급이 혼재한다. 제공된 텍스트 기준으로는 표에서 source와 oracle의 수치가 명확하고, adaptation의 정확한 수치는 본문 설명과 표 설명 사이에 차이가 있어 **일관되게 확정하기 어렵다**. 따라서 이 부분은 원문 표/섹션 대조 없이 단정하지 않는 것이 맞다. 다만 분명한 사실은 source-only 21.4 mIoU에 비해 adaptation이 일정 수준 개선을 주지만, oracle 64.0과는 큰 차이가 있다는 점이다.

클래스별로 보면 road, building, sky, car 같은 대형/빈번 클래스는 상대적으로 성능이 높고, rider, train, bike, mbike 등 빈도나 appearance 변동이 큰 클래스는 매우 어렵다.

### GTA5 → Nexar test

* Source (Dilation F.E.): **25.9 mIoU**
* FCN-in-the-wild [16]: **28.2 mIoU**
* MSRA: **47.5 mIoU**
* Oxford: **44.7 mIoU**
* VLLAB: **42.4 mIoU**

즉, challenge 상위 팀들은 source-only 및 기존 adaptation baseline보다 훨씬 높은 성능을 보인다. 특히 MSRA 팀은 47.5 mIoU로 큰 향상을 만들었다. 저자 설명에 따르면 우승팀은 frame-level adaptation과 pixel-level discrimination을 조합한 multi-stage pipeline을 사용했으며, 여러 backbone ensemble과 pyramid spatial pooling 등도 활용했다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **benchmark 논문으로서의 설계 품질**이다. 단순히 데이터 수를 늘린 것이 아니라, synthetic-to-real이라는 실용적이고 어려운 shift를 설정하고, classification과 segmentation이라는 서로 다른 task를 포함했으며, validation/test target을 분리해 UDA 평가 프로토콜의 현실성을 높였다. 또한 source-only, oracle, baseline adaptation, challenge result를 함께 보여 줌으로써 연구자들이 현재 위치를 명확히 파악할 수 있게 했다.

또 다른 강점은 **규모와 재현성에 대한 문제의식**이다. VisDA-C는 12개 클래스에 대해 15만 장 이상의 synthetic source와 12만 장이 넘는 real target 이미지를 포함하고, VisDA-S도 대규모 도시 주행 segmentation 적응 설정을 제공한다. 당시 기존 adaptation benchmark들이 작고 쉬워졌다는 문제의식에 대한 설득력 있는 대응이다. 또한 렌더링에 필요한 metadata와 도구를 공개하겠다는 계획은 후속 연구에서 controlled experiment를 가능하게 하는 장점이 있다.

실험적으로도 이 논문은 중요한 메시지를 준다. 첫째, source-only 성능이 매우 낮아 domain gap이 상당히 크다는 점, 둘째, adaptation이 큰 개선을 만들지만 baseline 수준에서는 아직 oracle과 거리가 있다는 점, 셋째, 강력한 backbone, self-ensembling, ensembling 등을 쓰면 상당한 격차를 줄일 수 있다는 점이다. 이는 UDA가 실제로 의미 있는 이득을 줄 수 있음을 보여준다.

하지만 한계도 분명하다. 가장 먼저, 이 논문은 **새로운 domain adaptation 알고리즘의 기술적 세부 혁신**을 중심으로 한 논문이 아니다. 따라서 상세 방법 설명 부분에서 baseline과 challenge winner를 소개하긴 하지만, 그 자체의 학술적 novelty는 데이터셋과 benchmark 설계에 있다. 만약 독자가 새로운 loss function이나 theoretical analysis를 기대한다면 그 점은 제한적이다.

둘째, challenge 상위권 결과가 oracle에 매우 가까워지면서, 저자 스스로 인정하듯 일부 설정은 빠르게 포화될 위험이 있다. 그래서 VisDA-C-ext, COCO↔YT-BB 전환, no-ImageNet pretraining setting 등이 제안된다. 이는 반대로 말하면, 원래 challenge 설정만으로는 장기적으로 충분히 어렵지 않을 수 있음을 뜻한다.

셋째, segmentation 부분의 baseline 기술은 classification보다 덜 상세하다. classification은 AlexNet, ResNext, DAN, CORAL, JMMD 등이 비교적 구체적으로 제시되지만, segmentation은 상당 부분 Hoffman et al. [16]에 의존한다. 따라서 VisDA-S 자체는 중요한 benchmark이지만, 이 논문만으로 segmentation adaptation 방법의 세부를 완전히 이해하기는 어렵다.

넷째, ImageNet pretraining에 대한 의존성 문제를 논문도 스스로 인정한다. 실제 응용 영역에서는 ImageNet과 같은 대규모 labeled pretraining set이 없을 수 있는데, challenge 참가자 대부분이 이런 supervised pretraining에 의존했다. 따라서 benchmark가 현실적인 synthetic-to-real 문제를 다룬다고 해도, 실제 저자들이 중요하다고 말한 의료영상이나 로보틱스의 “low-label regime” 전체를 그대로 대표한다고 보기는 어렵다.

마지막으로, 제공된 텍스트 기준에서는 일부 수치와 설명이 약간 불일치하는 부분이 있다. 예를 들어 segmentation adaptation의 mIoU 향상 수치는 표 설명과 본문 문장 사이에 다소 혼선이 있다. 따라서 해당 부분은 원 PDF를 대조하지 않고 단정적으로 재구성하면 오해를 낳을 수 있다. 이는 논문 내용의 본질적 약점이라기보다, 제공된 추출 텍스트의 한계이기도 하다.

## 6. 결론

이 논문은 synthetic-to-real unsupervised domain adaptation 연구를 위한 대표적 benchmark인 **VisDA2017**을 제안하고, classification과 semantic segmentation 두 과제에서 대규모 데이터셋, 현실적인 evaluation protocol, baseline, 그리고 challenge 결과를 체계적으로 제시한다. 핵심 기여는 새로운 adaptation 알고리즘 자체보다, **연구 커뮤니티가 더 어렵고 실용적인 문제를 공통 기준으로 평가할 수 있는 실험 기반을 마련했다는 점**에 있다.

분류 실험에서는 source-only 모델이 28~31% 수준으로 크게 무너지는 반면, DAN이나 Deep CORAL 같은 baseline adaptation도 유의미한 개선을 보이며, challenge 우승 방법은 90% 이상까지 성능을 끌어올린다. 분할 실험에서도 source-only와 oracle 간 큰 간극이 드러나고, adaptation이 그 간극을 줄일 가능성을 보여준다. 이 결과는 domain adaptation이 실제 문제에서 충분히 가치 있는 방향이라는 점을 뒷받침한다.

향후 연구 관점에서 이 논문이 특히 중요한 이유는 두 가지다. 하나는 더 어려운 synthetic-to-real benchmark를 통해 기존 기법의 진짜 한계를 드러냈다는 점이고, 다른 하나는 **ImageNet pretraining 없이도 잘 동작하는 adaptation**의 필요성을 강하게 제기했다는 점이다. 실제 적용에서는 대규모 supervised pretraining을 기대하기 어려운 경우가 많으므로, VisDA가 제안한 문제의식은 이후 연구에도 계속 의미가 있다. 따라서 이 논문은 UDA의 “최신 방법”을 배우기 위한 논문이라기보다, **이 분야의 실험 패러다임을 정리하고 확장한 기준점 논문**으로 이해하는 것이 가장 적절하다.
