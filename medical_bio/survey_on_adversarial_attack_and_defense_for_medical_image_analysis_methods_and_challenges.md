# Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges

## 논문 메타데이터

- **제목**: Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges
- **저자**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen
- **arXiv 공개 연도**: 2023
- **문서 버전 기준 연도**: 2024
- **학술지 표기**: ACM Computing Surveys (preprint 표기)
- **arXiv ID**: 2303.14133
- **arXiv URL**: https://arxiv.org/abs/2303.14133
- **PDF URL**: https://arxiv.org/pdf/2303.14133v2

## 연구 배경 및 문제 정의

이 논문은 의료영상 기반 진단 시스템이 높은 정확도를 달성했더라도, 사람 눈에 거의 보이지 않는 미세 교란에 의해 쉽게 오진을 낼 수 있다는 문제를 중심에 둔다. 저자들은 자연영상 분야에서 축적된 adversarial machine learning 연구가 의료영상으로 옮겨오는 과정에서, 의료영상 특유의 조건 때문에 별도의 체계화가 필요하다고 본다.

의료영상은 modality마다 신호 특성이 다르고, 데이터 수가 적고 불균형하며, 임상적 오류 비용이 높다. 또 해부학적 구조가 비교적 일정하기 때문에 자연영상과 동일한 공격·방어 프레임으로만 보기 어렵다. 저자들은 이런 차이 때문에 의료영상 adversarial robustness를 독립된 연구 문제로 다뤄야 한다고 주장한다.

## 논문의 핵심 기여

이 논문의 기여는 단순한 survey를 넘어서, taxonomy와 benchmark를 함께 제시했다는 점에 있다.

1. 의료영상 adversarial attack을 white-box, semi-white-box, black-box, restricted black-box(no-box)로 정리했다.
2. 의료영상 adversarial defense를 adversarial training, detection, image-level pre-processing, feature enhancement, knowledge distillation으로 분류했다.
3. 분류와 분할을 포함하는 의료영상 응용 맥락에서 공격과 방어를 통합적으로 설명하는 공통 프레임워크를 제시했다.
4. 다양한 설정에서 adversarial training을 직접 비교하는 benchmark를 구축해 방어 방법을 정량적으로 평가했다.
5. 평가 표준화, clean-robustness trade-off, 계산 효율, 의료영상 맞춤형 방어, 윤리 문제를 핵심 과제로 정리했다.

## 방법론 요약

### 1. 통합 수학적 프레임

논문은 의료영상 모델 $f_\theta(x)$에 대해, 입력 $x$에 작은 교란 $\delta$를 더해 $x̂ = x + \delta$를 만들고, $d(x, x̂) \le \epsilon$ 제약 아래에서 예측을 바꾸는 문제로 adversarial attack을 정의한다. 주로 $\ell_\infty$ 위협 모델을 사용하며, 이는 의료영상 robustness 평가에서 가장 일반적인 설정으로 채택된다.

이 정식화는 자연영상과 동일해 보이지만, 실제 적용 시에는 의료영상의 modality 특성, 병변의 희소성, 임상적으로 중요한 ROI의 존재 때문에 공격 강도와 평가 해석이 다르게 작동한다는 점이 논문의 전제다.

### 2. 공격 분류 체계

저자들은 공격을 공격자가 가진 정보와 조작 권한 기준으로 나눈다.

- **White-box attack**: 모델 구조와 파라미터를 모두 아는 상황
- **Semi-white-box attack**: 일부 정보만 아는 상황
- **Black-box attack**: 질의 기반 또는 transfer 기반으로 모델 내부 접근 없이 공격
- **Restricted black-box / no-box attack**: 의료 현장의 더 제한적인 조건을 반영한 시나리오

이 taxonomy의 장점은 의료영상 연구에서 서로 다른 위협 모델이 섞여 비교되던 문제를 정리해 준다는 점이다.

### 3. 방어 분류 체계

방어는 다음 다섯 축으로 요약된다.

- **Adversarial training**: 공격 예제를 학습 과정에 직접 포함
- **Adversarial detection**: 입력이 공격 예제인지 판별
- **Image-level pre-processing**: 입력 복원 또는 정제
- **Feature enhancement**: 내부 표현을 더 안정적으로 만드는 접근
- **Knowledge distillation**: teacher-student 구조로 robust representation 전달

논문의 톤은 분명하다. 여러 방어 중 가장 일관되게 강한 방법은 여전히 adversarial training이며, 그래서 저자들도 benchmark의 중심을 여기에 둔다.

## 공격 및 방어 정리

### 1. 의료영상 공격

논문은 기존 FGSM, PGD, CW, Square Attack, AutoAttack 같은 자연영상 기반 공격을 의료영상에 옮겨 평가하고, 의료영상 특화 공격인 SMA(Stabilized Medical Attack)도 함께 검토한다. 분류뿐 아니라 분할 task에서도 공격이 실제로 큰 성능 저하를 만들 수 있음을 보여준다.

핵심 메시지는 다음과 같다.

- 매우 작은 $\epsilon$에서도 진단 성능이 급격히 붕괴한다.
- 의료 분할에서는 mask 품질 저하가 곧 치료 계획 오류로 이어질 수 있다.
- 자연영상에서 검증된 공격이 의료영상에서도 상당한 파괴력을 가진다.
- 의료영상 특화 공격은 해부학적 구조와 안정성 제약을 반영해 더 현실적인 위협을 구성할 수 있다.

### 2. 의료영상 방어

방어 측면에서는 adversarial training이 가장 중심적이며, detection과 preprocessing은 보조적 위치에 놓인다. 논문은 단순히 방법 목록을 나열하기보다, 실제 임상 적용성을 생각하면 강건한 예측 자체를 만드는 것이 더 중요하다는 입장을 취한다.

특히 detection만으로는 공격을 분류할 수 있어도 오진 자체를 막지 못할 수 있고, preprocessing 기반 방법은 adaptive attack에 취약할 수 있다는 점이 함의로 읽힌다. 반면 adversarial training은 훈련 비용이 크지만, white-box와 black-box 설정 모두에서 가장 일관된 robustness 향상을 보인다.

## 실험 설정과 결과

이 논문의 차별점은 survey임에도 직접 benchmark를 구축했다는 점이다. 주요 실험 설정은 다음과 같다.

- **분류 데이터셋**: Messidor, ISIC 2017, ChestX-ray14
- **분할 데이터셋**: ISIC 2017, COVID-19 chest X-ray segmentation
- **분류 모델**: ResNet-18, MobileNetV2, Wide-ResNet-28-10
- **분할 모델**: U-Net, SegNet
- **공격 방법**: FGSM, PGD, CW, Square Attack, Auto Attack
- **학습 환경**: 단일 NVIDIA Tesla A100

### 1. 자연학습 모델의 취약성

표 8과 9를 보면 자연학습 모델은 작은 $\epsilon$에서도 정확도가 빠르게 무너진다. 예를 들어 Messidor multi-class 분류에서 자연학습 Wide-ResNet-28-10은 $\epsilon=8/255$의 PGD에서 정확도가 0.0까지 떨어진다. ISIC dermoscopy에서도 자연학습 모델은 강한 white-box 공격에서 사실상 무력화된다.

이는 의료영상 모델이 clean accuracy가 높더라도 robust accuracy는 전혀 다른 문제라는 점을 분명히 보여준다.

### 2. adversarial training의 효과

방어 실험에서는 HAT, MPAdvT, MART, TRADES, PGD-AT를 비교한다. Messidor와 ISIC 분류 모두에서 adversarially trained 모델은 자연학습 모델보다 훨씬 높은 robust accuracy를 유지한다. 예를 들어 Messidor multi-class 설정에서 $\epsilon=8/255$ PGD 공격 시 NAT가 0.0인 반면, HAT는 37.1, PGD-AT는 30.0, MPAdvT는 34.3 수준을 유지한다.

ISIC에서도 비슷한 경향이 반복되며, 전체적으로 HAT와 MPAdvT가 강한 편이고, TRADES와 PGD-AT도 일관된 개선을 보인다.

### 3. 분할 모델에서의 효과

표 10과 14는 U-Net 기반 분할 실험을 보여준다. 예를 들어 ISIC dermoscopy에서 BCE adversarial loss를 쓴 NAT 모델은 $\epsilon=8/255$ PGD 공격 시 mIOU 0.167, Dice 0.255까지 떨어지지만, adversarial training을 적용한 모델은 mIOU 0.601, Dice 0.700까지 유지한다.

COVID-19 X-ray segmentation에서도 유사한 패턴이 보이며, adversarial training이 특히 no-box 환경에서도 강건성을 유지하는 점이 강조된다. 즉, 방어 효과가 white-box에만 국한되지 않는다.

### 4. 계산 비용

강건성 향상에는 비용이 따른다. 표 15에 따르면 분류에서 NAT가 epoch당 약 16~17초인 반면, adversarial training 계열은 대체로 53~75초 수준이다. 분할에서는 NAT가 14초 안팎인데 PGD-AT는 148~155초 수준으로 더 큰 차이가 난다.

논문은 이 비용 문제를 실용화의 핵심 병목으로 본다.

### 5. 해석 가능성 시각화

Grad-CAM 시각화에서는 자연학습 모델이 공격 강도가 커질수록 주목 영역이 흔들리며 잘못된 클래스로 이동한다. 반면 adversarially trained 모델은 공격 강도가 증가해도 비교적 일관된 discriminative region을 유지한다. 이는 robust training이 단지 수치만 개선하는 것이 아니라, 모델의 관심 영역을 더 안정화할 수 있음을 시사한다.

## 한계 및 향후 연구 가능성

### 1. benchmark 표준화 부족

저자들은 의료영상 adversarial 연구가 서로 다른 데이터셋, 위협 모델, 평가 규칙을 사용해 직접 비교가 어렵다고 지적한다. 이 논문의 benchmark 제안은 의미 있지만, 커뮤니티 전체 표준으로 자리 잡은 것은 아니다.

### 2. clean accuracy와 robust accuracy의 trade-off

강건성을 올리면 clean accuracy가 떨어질 수 있다는 문제는 여전히 남아 있다. 논문은 일부 의료영상 설정에서 작은 perturbation radius를 쓰면 두 목표를 동시에 어느 정도 만족할 가능성을 보여주지만, 일반 원리로 정리되지는 않았다.

### 3. 계산 효율 문제

adversarial training은 특히 segmentation에서 매우 비싸다. 실제 임상 배포나 중소 규모 연구 환경에서는 계산 예산 자체가 큰 장벽이 될 수 있다.

### 4. 의료영상 맞춤형 방어의 부족

기존 방법 상당수는 자연영상 방어를 그대로 이식한 것이다. 해부학적 prior, modality physics, 임상 workflow를 더 적극적으로 반영하는 defense는 아직 부족하다는 것이 저자들의 진단이다.

### 5. 윤리적 딜레마

잠재적 공격을 막기 위해 clean performance를 일부 희생하는 것이 임상적으로 정당한가라는 문제는 단순한 기술 문제가 아니다. robustness 강화가 실제 환자 안전과 어떻게 연결되는지에 대한 임상적·윤리적 검토가 필요하다.

## 실무적 또는 연구적 인사이트

### 1. 의료 AI 평가에 robustness를 별도 축으로 넣어야 한다

이 논문은 의료영상 모델 평가에서 accuracy, AUC, Dice만 보는 관행이 불충분하다는 점을 설득력 있게 보여준다. 실제 연구 설계에서도 clean 성능과 robust 성능을 분리해 보는 것이 맞다.

### 2. adversarial training은 여전히 baseline이 아니라 strong baseline이다

최신 연구에서는 다양한 certified defense나 detection 기법이 거론되지만, 의료영상에서는 adversarial training이 여전히 가장 현실적인 기준선으로 읽힌다. 특히 benchmark를 만들 때 빠뜨리면 안 되는 비교군이다.

### 3. segmentation robustness는 분류 robustness와 별개로 다뤄야 한다

의료영상에서는 분할 결과가 후속 정량화, 수술 계획, 방사선 치료 계획에 직접 연결될 수 있다. 따라서 segmentation robustness를 독립적으로 다루는 이 논문의 구성은 실무적으로 타당하다.

### 4. 의료영상 전용 defense 여지가 크다

해부학적 구조 보존, modality-specific noise model, acquisition physics, report consistency 같은 의료 도메인 지식을 방어에 넣는 방향은 여전히 열려 있다. 이 논문은 그 필요성을 명확히 제기하지만 해답은 아직 초기 단계다.

## 종합 평가

이 논문은 의료영상 adversarial attack과 defense를 가장 체계적으로 정리한 문헌 중 하나이며, 특히 survey와 benchmark를 결합했다는 점에서 가치가 크다. 의료영상 특유의 제약을 강조하면서도, 실제로 어떤 방어가 현재 기준에서 가장 믿을 만한지를 실험으로 보여 준다.

최신 foundation model이나 multimodal medical VLM까지 포괄하는 문서는 아니지만, 의료 AI 보안성과 robustness를 연구하려는 경우 출발점으로 매우 적절하다. 특히 "의료영상에서 공격이 실제로 얼마나 위험한가", "방어는 어디까지 실용적인가", "무엇이 아직 비어 있는가"를 동시에 파악하게 해 주는 기준 문헌이다.
