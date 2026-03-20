# Jekyll: Attacking Medical Image Diagnostics using Deep Generative Models

이 논문은 의료 영상 진단 체계가 딥러닝 덕분에 고도화되는 동시에, 같은 딥러닝 기술이 오히려 의료 사기와 진단 교란에 악용될 수 있다는 점을 정면으로 다룬다. 저자들은 **Jekyll**이라는 GAN 기반 의료영상 변환 공격 프레임워크를 제안하며, 환자의 실제 의료 영상을 입력받아 환자 고유의 identity는 유지하면서 공격자가 원하는 질환 소견을 시각적으로 주입한 fake medical image를 생성한다. 논문은 이런 공격이 단순히 알고리즘만 속이는 adversarial example 수준이 아니라, 의료 전문가와 자동화된 진단 시스템을 동시에 오판하게 만들 수 있다고 주장하며, X-ray와 retinal fundus image에서 이를 실험적으로 보인다. 또한 blind detection과 supervised detection이라는 두 가지 방어 방향도 함께 검토한다.  

## 1. Paper Overview

이 논문의 핵심 목적은 “medical image diagnosis가 딥러닝 기반 생성 모델에 의해 얼마나 현실적으로 공격될 수 있는가”를 보여주는 것이다. 기존 의료 AI 보안 연구가 주로 classifier를 fooling하는 adversarial perturbation이나 ML 시스템 자체에 대한 공격에 초점을 맞췄다면, 이 논문은 한 단계 더 나아가 **의료영상 자체를 조작해서 사람이 봐도 질환이 있는 것처럼 보이는 deepfake medical image**를 만드는 문제를 다룬다. 저자들은 헬스케어 도메인에서 이미 사기와 데이터 무결성 문제가 심각하다는 점을 출발점으로 삼고, 의료영상 조작이 잘못된 진단, 불필요한 시술, 보험사기, 환자 피해로 이어질 수 있음을 강조한다.

연구 문제가 중요한 이유는 의료영상이 임상 의사결정의 핵심 근거이기 때문이다. chest X-ray나 retinal fundus image는 의사가 직접 판독하고, 동시에 increasingly automated ML system도 이를 함께 해석한다. 따라서 공격이 성공하려면 단순히 classifier 한 개를 속이는 것이 아니라, **의사와 알고리즘 모두를 납득시킬 정도로 시각적으로 자연스러운 질환 패턴**을 만들어야 한다. 논문은 정확히 이 요구를 충족하는 공격을 목표로 한다.

또한 저자들은 기존 adversarial sample 방식이 의료영상 환경에서는 한계가 있다고 본다. adversarial perturbation은 사람이 보기에는 여전히 정상 영상처럼 보이는 경우가 많아, 의료 전문가의 시각적 판독이 남아 있는 현재 의료 workflow에서는 실전성이 떨어진다. 반면 Jekyll은 사람에게도 병변이 있는 것처럼 보이는 “style-changed image”를 만들기 때문에, 훨씬 더 실질적인 위협 모델로 제시된다.

## 2. Core Idea

논문의 중심 아이디어는 의료영상 공격을 **imperceptible perturbation** 문제가 아니라 **controlled image-to-image translation** 문제로 바꾸는 데 있다. Jekyll은 vanilla GAN처럼 단순 noise vector에서 이미지를 생성하지 않는다. 대신 실제 환자 영상을 입력으로 받고, 그 영상의 anatomical identity 또는 patient-specific content는 최대한 보존하면서, style에 해당하는 disease condition만 공격자가 원하는 방향으로 바꾼다. 즉, “이 환자의 영상처럼 보이지만 실제로는 질환이 주입된 가짜 영상”을 만드는 것이 핵심이다.

이 아이디어가 중요한 이유는 공격 성공 조건을 정확히 겨냥하기 때문이다. 논문은 공격 목표를 세 가지로 정리한다. 첫째, 공격자가 원하는 질환을 주입해야 한다. 둘째, 환자의 identity를 유지해야 한다. 셋째, 반복 방문 시 질환의 자연스러운 progression까지 흉내 내며 공격을 지속할 수 있어야 한다. 따라서 좋은 공격은 단순히 “질환처럼 보이는 이미지”가 아니라, “원래 환자의 영상 기록과 연결되어도 큰 이질감이 없는 질환 영상”이어야 한다.

저자들이 Jekyll을 기존 CT-GAN과 구분하는 지점도 여기 있다. CT-GAN은 3D CT를 대상으로 특정 위치를 in-painting하는 방식이라, 병변 위치 선정과 수작업 보정이 필요하다. 반면 Jekyll은 질환을 영상 전체 스타일 수준에서 학습하므로, diabetic retinopathy처럼 국소적인 한 점이 아니라 여러 영역에 퍼진 질환에도 더 자연스럽게 적용될 수 있고, 공격자는 단일 generator forward pass만으로 조작 이미지를 만들 수 있다.

## 3. Detailed Method Explanation

### 3.1 Threat model

논문이 상정하는 공격자는 환자가 촬영한 의료영상에 접근할 수 있다. 예를 들어 PACS(Picture Archiving and Communication System) 같은 의료영상 관리 시스템을 침해해 저장 중이거나 전송 중인 영상을 가로챌 수 있다고 가정한다. 저자들은 현실의 PACS 배포 환경이 misconfiguration, default credential, patch 부족, encryption 부재 등으로 취약할 수 있다고 설명한다. 공격자는 원본 영상을 획득한 뒤 이를 질환이 주입된 fake image로 바꾸고, 원본은 제거한다. 이후 의사나 ML 시스템은 조작된 영상만 보게 된다.

또한 공격자는 victim의 과거 영상들을 대량으로 갖고 있지 않다고 가정한다. 단 하나의 victim image만 있으면 되고, 모델 훈련에는 공개 데이터셋에 포함된 health condition label과 anonymized patient ID를 활용한다. 이 점은 공격 실현성을 높이는 중요한 가정이다. 질환 segmentation mask도 필요하지 않다고 명시한다.

### 3.2 Why vanilla GAN is not enough

저자들은 vanilla GAN이 이 문제에 적합하지 않다고 본다. vanilla GAN의 generator는 latent noise $z$에서 이미지를 생성하므로, 생성 결과를 특정 환자 identity나 특정 질환 방향으로 정밀하게 통제하기 어렵다. 그런데 Jekyll의 목적은 “아무 의료영상”을 만드는 것이 아니라, “이 환자 영상의 content는 남기고 disease style만 바꾸는 것”이다. 따라서 입력 조건으로 실제 이미지를 넣는 image-conditional generation 구조가 필요하다.

### 3.3 CycleGAN 기반 unpaired translation

이 논문은 paired data가 거의 없다는 현실 제약 때문에 Pix2Pix 같은 paired translation 대신 **CycleGAN 기반의 unpaired image-to-image translation**을 채택한다. 즉, 건강한 환자 이미지 집합 $X$와 특정 질환이 있는 이미지 집합 $Y$를 각각 모아두고, 이 둘 사이의 mapping을 학습한다. 이때 중요한 것은 개별 환자별 healthy/diseased pair가 필요 없다는 점이다. 공개 데이터셋만으로도 healthy distribution과 disease distribution을 학습할 수 있다.

논문은 $X$를 condition $C_X$를 가진 이미지 도메인, $Y$를 condition $C_Y$를 가진 이미지 도메인으로 두고, generator $G: X \rightarrow Y$와 $F: Y \rightarrow X$, discriminator $D_Y$와 $D_X$를 사용한다고 설명한다. 여기서 $G$는 healthy image를 diseased style로 번역하고, $F$는 반대로 disease removal에도 사용될 수 있다. 기본 직관은 “healthy domain의 어떤 이미지가 disease domain처럼 보이되, 환자 고유의 content는 보존되도록” 만드는 것이다.

### 3.4 Loss design

논문 본문에 제시된 식을 모두 완전한 형태로 인용할 필요는 없지만, 구조적으로는 다음 세 가지가 핵심이다.

첫째, **adversarial loss**다. generator가 생성한 $G(x)$가 target domain $Y$의 진짜 이미지처럼 보이도록 만들고, discriminator는 이를 구분하려 한다. 논문은 학습 안정성과 이미지 품질 향상을 위해 least-squares GAN loss를 사용한다고 설명한다.

둘째, **cycle consistency**다. unpaired setting에서는 무한히 많은 mapping이 가능하므로, 단순 adversarial loss만으로는 identity가 무너질 수 있다. 이를 막기 위해 $x \rightarrow G(x) \rightarrow F(G(x)) \approx x$ 및 $y \rightarrow F(y) \rightarrow G(F(y)) \approx y$가 되도록 제약한다. 이 cycle consistency가 환자 고유의 해부학적 구조를 보존하는 데 핵심 역할을 한다고 이해하면 된다.

셋째, **identity preservation / controlled progression** 측면이다. 논문은 공격 목표 자체가 환자 identity 보존과 지속 공격이므로, 모델 구조와 추가 절차가 모두 “질환만 바꾸고 사람은 그대로 남기는 것”을 향한다. 특히 반복 진료 상황에서는 severity/stage 정보가 있으면 stage-conditioned disease injection을 학습하고, 그렇지 않더라도 latent interpolation을 통해 intermediate disease stage를 흉내 낼 수 있다고 설명한다. 이는 단발성 fake generation이 아니라 temporal consistency까지 고려한 공격 설계라는 점에서 흥미롭다.  

### 3.5 Why the method is more dangerous than adversarial examples

논문이 계속 강조하는 차별점은, adversarial examples는 대체로 사람 눈에는 정상인데 모델만 속이는 반면, Jekyll은 **사람이 봐도 질환이 있어 보이는 조작 영상**을 만든다는 점이다. 그래서 query access도 필요 없고, target hospital이 어떤 classifier를 쓰는지도 몰라도 된다. 질환 소견이 실제로 영상에 시각적으로 드러나기 때문에, 의료 전문가와 ML detector를 동시에 속일 가능성이 높아진다. 이 부분이 방법론의 핵심적 위협성이다.

## 4. Experiments and Findings

논문은 공격을 두 가지 대표 modality에서 검증한다. chest X-ray에서는 healthy patient에 대해 **Cardiomegaly**와 **Pleural Effusion**을 주입하고, retinal fundus image에서는 **Diabetic Retinopathy**를 주입한다. 즉, 단일 질환·단일 modality에 묶이지 않고, chest radiology와 ophthalmology라는 서로 다른 의료영상 영역에서 일반성을 보이려 한다.

평가는 크게 세 갈래다. 첫째, **machine learning tools**를 이용한 disease injection success와 identity preservation 측정이다. 논문은 patient-level partitioning을 통해 evaluation identity classifier를 만들 수 있도록 데이터셋을 구성했다고 설명한다. 이는 단순히 disease classifier만 fooling했는지 보는 것이 아니라, 원본 환자의 identity가 유지되었는지를 별도 분류기로 검증하려는 설계다.

둘째, **image quality metric** 평가다. fake image가 얼마나 자연스러운지, 단순히 label만 바뀐 것이 아니라 육안으로도 그럴듯한지를 정량 지표로 본다. 논문 초반 contribution에서도 effectiveness가 machine learning algorithms와 image quality metrics 모두로 평가되었다고 명시한다.

셋째, 가장 중요한 **medical professional user study**다. 저자들은 의료 전문가들이 fake image를 보고도 target disease presence를 설득력 있게 받아들이고, real과 fake를 잘 구분하지 못했다고 보고한다. 이 결과는 Jekyll이 단순 ML attack이 아니라 clinical workflow 공격이라는 논문의 주장을 뒷받침한다.

논문 본문에서 드러나는 정량적 메시지도 강하다. 예를 들어 diabetic retinopathy stage 관련 실험에서는 Jekyll이 severe 및 proliferative stage에서 **99% 이상 disease injection rate**와 **88% 이상 identity preservation rate**를 달성했다고 서술한다. 이는 적어도 논문이 구성한 evaluation setting 안에서는 질환 주입과 identity 유지가 동시에 상당히 높은 수준으로 가능함을 보여준다.

또한 progression attack 측면에서, 논문은 disease stage label이 있을 때는 stage-aware training을, 없을 때는 linear interpolation을 통해 intermediate stage를 생성하는 방법을 제시한다. Figure 11 설명과 해당 본문은 Cardiomegaly의 progressive injection 사례를 통해, 반복 방문 상황에서도 질환이 자연스럽게 진행되는 것처럼 보이게 만들 수 있음을 보여준다. 이는 “한 번만 속이는 공격”보다 훨씬 위험한 시나리오다.  

방어 측면에서는 blind detection과 supervised detection을 비교한다. 논문 contribution에 따르면 supervised detection은 상당히 효과적이지만, 공격자가 Jekyll을 수정해 detection을 우회하는 evasion attack에도 취약할 수 있다. 즉, 단순 detector 추가만으로 문제를 해결하기 어렵다는 점도 함께 보여준다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 위협 모델이 매우 현실적이라는 점이다. 의료영상 AI 보안 연구가 종종 classifier-centric setting에 머무르는 반면, 이 논문은 실제 의료 workflow를 겨냥한다. 의사가 영상을 보고 판단하고, 동시에 보험사나 병원이 자동화 알고리즘을 쓸 수 있다는 현실을 반영해, 사람과 모델을 동시에 속여야 하는 공격을 설계했다는 점이 설득력 있다.

두 번째 강점은 공격 구현의 실용성이다. 공개 데이터셋, 단일 victim image, unpaired training만으로도 공격이 가능하다고 설계했다. 또한 CT-GAN과 비교해 수작업 region selection이나 복잡한 in-painting 절차가 줄어들고, 단일 translation step으로 disease injection이 가능하다고 주장한다. 공격자 입장에서 훨씬 low-friction이라는 뜻이다.

세 번째 강점은 identity preservation을 명시적 공격 목표로 둔 점이다. 의료영상은 환자 고유의 anatomical signature를 담고 있으므로, 단순히 질환처럼만 보이는 가짜 영상은 이전 검사와 비교될 때 쉽게 의심받을 수 있다. 논문은 이 점을 정확히 짚고 “질환 주입”과 “환자 identity 유지”를 동시에 최적화 대상으로 삼는다. 이것이 Jekyll을 일반 이미지 deepfake와 구분하는 핵심 포인트다.

한편 한계도 있다. 우선 논문이 보여준 공격은 chest X-ray와 retinal fundus라는 2D modality 중심이다. 저자들은 3D modality 확장 가능성을 언급하지만, 실험적으로는 본격 검증하지 않는다. 따라서 CT나 MRI 같은 volumetric modality에서 동일한 수준의 실전성이 유지되는지는 추가 검증이 필요하다.

또 다른 한계는 evaluation setting이 논문 저자들이 설계한 classifier, data partition, user study에 의존한다는 점이다. 실제 병원에서는 acquisition protocol, PACS workflow, multi-view consistency, longitudinal record auditing, radiology reporting habit 등이 더 복잡하므로, 논문의 성능 수치가 그대로 임상 현장으로 일반화된다고 보기는 어렵다. 논문도 supervised detector가 우회될 수 있다고 인정하듯, 방어-공격 간 arms race 성격이 강하다.

비판적으로 해석하면, 이 논문은 “의료 AI가 위험하다”를 주장하기보다, **medical image integrity**가 앞으로 의료 AI 시대의 핵심 보안 문제라는 점을 선명하게 보여준다. ML model robustness만 보아서는 충분하지 않고, 의료영상 자체의 provenance, storage integrity, transmission integrity, longitudinal consistency checking이 모두 중요하다는 메시지로 읽는 것이 맞다.

## 6. Conclusion

이 논문은 GAN 기반 image-to-image translation을 이용해 의료영상 진단 체계를 교란하는 **Jekyll** 공격을 제안하고, 이를 chest X-ray와 retinal fundus image에서 실증했다. 핵심은 환자 고유의 identity를 유지하면서 공격자가 원하는 질환 소견을 자연스럽게 주입해, 의료 전문가와 알고리즘을 동시에 오도하는 것이다. 이를 위해 저자들은 vanilla GAN이 아닌 CycleGAN 계열의 unpaired translation을 활용하고, disease injection, identity preservation, progressive attack 지속 가능성을 함께 고려한다.  

실무적으로 이 논문이 주는 가장 중요한 교훈은, 의료영상 AI의 신뢰성은 classifier accuracy만으로 보장되지 않는다는 점이다. 의료영상이 생성·보관·전송·판독되는 전체 파이프라인의 무결성이 함께 보호되지 않으면, 매우 그럴듯한 fake medical image가 임상 의사결정과 보험 처리, 환자 안전을 동시에 위협할 수 있다. 따라서 이 연구는 medical AI security를 넘어, 향후 healthcare infrastructure security 전체를 다시 보게 만드는 경고성 논문으로 볼 수 있다.
