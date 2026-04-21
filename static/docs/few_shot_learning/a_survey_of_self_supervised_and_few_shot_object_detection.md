# A Survey of Self-Supervised and Few-Shot Object Detection

- **저자**: Gabriel Huang, Issam Laradji, David Vázquez, Simon Lacoste-Julien, Pau Rodríguez
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2110.14711

## 1. 논문 개요

이 논문은 object detection에서 라벨이 부족한 상황을 다루는 두 흐름, 즉 **few-shot object detection (FSOD)** 과 **self-supervised object detection (SSOD)** 을 함께 정리한 survey이다. 저자들은 기존 object detector가 PASCAL VOC, MS COCO 같은 대규모 annotated dataset에 크게 의존한다는 점을 출발점으로 삼는다. 특히 object detection과 instance segmentation은 단순 classification보다 훨씬 조밀한 annotation이 필요하므로, 라벨 비용과 시간 부담이 매우 크다.

이 논문이 다루는 핵심 연구 문제는 두 가지다. 첫째, novel class에 대해 극소수의 labeled example만으로도 탐지가 가능한가이다. 이것이 FSOD의 문제다. 둘째, 아예 unlabeled image로부터 representation이나 detector 자체를 미리 학습해 downstream detection으로 잘 옮길 수 있는가이다. 이것이 self-supervised learning의 문제다. 논문은 특히 이 둘의 결합이 중요하다고 본다. 왜냐하면 기존 FSOD는 base class에 대한 충분한 labeled detection data를 전제로 하지만, self-supervision은 이 전제를 약화시킬 가능성이 있기 때문이다.

이 문제가 중요한 이유는 명확하다. 실제 응용에서는 새 카테고리가 자주 등장하고, 희귀 동물, 식물 종, 산업용 결함처럼 데이터 수집과 annotation이 어려운 경우가 많다. 따라서 적은 supervision으로도 강한 detector를 만들 수 있다면 학문적으로도 중요하고 실용적으로도 가치가 크다. 이 논문은 단순히 방법들을 나열하는 데 그치지 않고, benchmark, metric, evaluation protocol의 문제점까지 함께 비판적으로 정리한다는 점이 특징이다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 FSOD와 SSOD를 따로 보지 말고, **“적은 라벨 환경에서 object detection을 어떻게 학습할 것인가”** 라는 더 큰 틀에서 함께 이해해야 한다는 것이다. 저자들은 기존 survey가 일반 object detection, few-shot classification, 또는 일반 self-supervised representation learning에 치우쳐 있었고, 정작 **few-shot detection과 self-supervised detection을 함께 체계적으로 다룬 정리는 부족했다**고 본다.

논문은 reviewed method들을 하나의 taxonomy로 정리한다. 큰 축은 다음과 같다. 하나는 **few-shot detector** 자체의 구조적 차이이고, 다른 하나는 **pretraining이 supervised backbone-only인지, self-supervised backbone-only인지, 아니면 backbone과 detection head까지 함께 pretrain하는지**이다. 이 구조 덕분에 독자는 “무엇을 학습하는가”, “어디까지 pretrain하는가”, “few-shot adaptation은 finetuning에 의존하는가”를 분리해서 볼 수 있다.

기존 접근과의 차별점은 세 가지로 요약할 수 있다. 첫째, 논문은 단순 backbone pretraining 관점이 아니라 **detector head까지 self-supervised pretraining하는 흐름**을 별도로 강조한다. 둘째, few-shot classification과 few-shot detection이 겉보기보다 매우 다르다는 점을 명시적으로 설명한다. 셋째, 성능 비교표를 그대로 믿기보다, split 구성, support set sampling, train/val/test 관행 같은 evaluation setup의 불안정성을 논문 차원에서 강하게 지적한다.

## 3. 상세 방법 설명

### 3.1 기본 object detection 배경

논문은 먼저 Faster R-CNN과 DETR를 중심으로 object detection의 기본 개념을 정리한다. Faster R-CNN은 대표적인 two-stage detector다. 먼저 backbone이 image에서 feature map을 추출하고, 그 위에서 **Region Proposal Network (RPN)** 가 object가 있을 법한 후보 box를 만든다. 이후 각 proposal에 대해 **RoIAlign** 으로 고정 크기 feature를 뽑고, **box head / ROI head** 가 class와 bounding box refinement를 예측한다. 이 과정에서 중복 박스를 제거하기 위해 **Non-Maximum Suppression (NMS)** 를 사용한다. Feature Pyramid Network (FPN)는 서로 다른 해상도의 feature를 결합해 multi-scale detection을 돕는다.

DETR는 이와 다르게 end-to-end set prediction 구조를 사용한다. backbone feature를 transformer encoder에 넣고, decoder에는 미리 학습된 object query embedding들을 넣는다. decoder는 최대 100개 같은 고정 개수의 detection prediction을 출력한다. DETR는 NMS를 쓰지 않고, 학습 시 **Hungarian matching** 으로 prediction과 ground-truth box를 일대일 대응시킨 뒤 loss를 계산한다. 저자들은 이런 구조 차이가 self-supervised pretraining 방식에도 직접 영향을 준다고 본다.

### 3.2 FSOD의 문제 설정

논문이 formalize하는 FSOD의 표준 절차는 세 단계다.

첫째, **base training** 단계에서 많은 labeled example이 있는 base class만 사용해 detector를 학습한다. 이때 novel class annotation은 제거된다.  
둘째, **few-shot finetuning** 단계에서 base와 novel class를 모두 포함한 매우 작은 support set으로 모델을 적응시킨다.  
셋째, **few-shot evaluation** 단계에서 test image에 대해 base와 novel class를 함께 탐지하고, 성능을 각각 측정한다.

이 설정은 few-shot classification과 다르다. FSOD에서는 한 이미지 안에 여러 object가 동시에 존재할 수 있고, base와 novel class를 함께 예측해야 하며, 대다수 방법이 episodic meta-learning보다 **한 번의 base training 후 한 번의 finetuning** 에 가깝다. 논문은 이것이 few-shot classification의 깔끔한 episodic protocol과 다르며, 평가 분산이 커지는 원인이라고 설명한다.

### 3.3 FSOD 방법들의 구조

논문은 FSOD 방법을 크게 세 부류로 나눈다.

첫 번째는 **finetuning-only methods** 다. 대표적으로 TFA, MPSR, Retentive RCNN, DETReg가 여기에 속한다. 이들은 대체로 기존 detector를 거의 그대로 유지하고, base training 후 support set으로 finetuning하는 단순한 전략을 쓴다. 예를 들어 TFA는 Faster R-CNN의 classification head를 cosine classifier로 바꾸고, novel class weight를 새로 추가한 뒤 마지막 layer 위주로 finetune한다. 저자들은 이 단순 baseline이 상당히 강력하다고 평가한다.

두 번째는 **prototype-based methods** 다. RepMet처럼 각 class의 representative 또는 prototype을 support example에서 만들고, query proposal feature와 prototype 사이의 거리로 분류한다. 핵심 직관은 novel class에 대한 parameter를 새로 크게 학습하기보다, support example이 class의 “참조점” 역할을 하게 만드는 것이다.

세 번째는 **modulation-based methods** 다. Meta-YOLO, Meta-RCNN, FsDetView, Meta-DETR 등이 여기에 속한다. 이 계열은 support image에서 class-specific support weight 또는 class embedding을 만들고, 이를 query feature에 곱하거나 결합해 class-conditioned feature를 만든다. 예를 들어 query feature를 $f_{qry}$, support에서 만든 class vector를 $f_{cls}$라고 하면, 단순한 modulation은 $[f_{qry} \otimes f_{cls}]$ 형태로 표현된다. FsDetView처럼 $[f_{qry} \otimes f_{cls}, f_{qry} - f_{cls}, f_{qry}]$ 처럼 더 풍부하게 결합하는 방식도 있다. 이 계열의 핵심은 “novel class detector를 직접 새로 학습한다”기보다, support를 통해 query를 조건부로 해석하게 만드는 것이다.

논문은 중요한 실증적 결론도 제시한다. 많은 conditioning-based method가 이론적으로는 finetuning 없이도 동작 가능하지만, 실제로는 **대부분 finetuning을 할 때 더 잘 된다**. 즉, “conditioning만으로 few-shot detection이 충분한가”라는 기대와 달리, 현 시점 방법들은 여전히 finetuning 의존성이 크다.

### 3.4 FSOD 평가 지표

논문은 FSOD에서 mean Average Precision, 즉 mAP를 상세히 설명한다. detector는 confidence threshold에 따라 precision과 recall이 달라지므로, threshold를 바꾸며 precision-recall curve를 만들고 그 아래 면적을 Average Precision(AP)로 계산한다. class별 AP를 평균한 것이 mAP다.

precision@k와 recall@k는 랭킹된 detection 상위 $k$개를 기준으로 계산된다. detection이 ground-truth와 충분히 겹치면 True Positive, 아니면 False Positive로 처리한다. 겹침 정도는 **Intersection over Union (IoU)** 로 판단하며, 보통 $0.5$ 또는 $0.75$ 같은 threshold를 사용한다. COCO-style mAP는 IoU threshold를 $0.5$에서 $0.95$까지 $0.05$ 간격으로 바꾸어 평균낸다.

FSOD에서는 base와 novel 성능을 분리해 $bAP$, $nAP$로 보고하는 것이 일반적이다. 논문은 novel class 성능만 강조하는 관행이 많지만, base class 성능 유지 역시 catastrophic forgetting 관점에서 중요하다고 지적한다.

### 3.5 Self-supervised pretraining 방법

논문은 self-supervised pretraining을 먼저 일반 visual representation learning 관점에서 세 범주로 나눈다.

첫째, **contrastive learning** 이다. 대표적 손실은 InfoNCE다. 논문에 제시된 식은 다음과 같다.

$$
L_{\text{InfoNCE}}(\theta)
=
- \mathbb{E}_X
\left[
\log
\frac{f_\theta(x^+, c)}
{\sum_i f_\theta(x_i^-, c)}
\right]
$$

또는 image augmentation 문맥에서는

$$
L_{\text{InfoNCE}}(\theta)
=
- \mathbb{E}_X
\left[
\log
\frac{f_\theta(x^+, x_0)}
{\sum_i f_\theta(x_i^-, x_0)}
\right]
$$

여기서 $x^+$는 같은 이미지의 다른 augmentation으로 만든 positive sample이고, $x_i^-$는 다른 이미지에서 온 negative sample이다. 목표는 같은 이미지의 다른 view는 가깝게, 다른 이미지의 view는 멀게 만드는 것이다. SimCLR, MoCo, DenseCL, ReSim 등이 이 계열과 연결된다.

둘째, **clustering-based method** 다. SwAV 같은 방식은 image view 간 cluster assignment를 서로 예측하게 하며, pseudo-label을 clustering으로 만든다.

셋째, **self-distillation / BYOL 계열** 이다. student와 teacher network의 출력을 맞추는 방식이며, teacher는 student의 exponential moving average로 갱신된다. DINO, EsViT가 여기에 속한다. 논문은 특히 DINO에서 ViT의 attention map이 object-like segmentation을 자연스럽게 보여준다는 점을 흥미로운 현상으로 소개한다.

### 3.6 왜 classification-style self-supervision만으로는 부족한가

논문은 self-supervised classification representation을 detection에 그대로 옮길 때의 문제를 두 가지로 정리한다.

첫째, **untrained detection heads** 문제다. backbone만 pretrain되어도 FPN, RPN, ROI head, 또는 DETR의 encoder/decoder는 여전히 랜덤 초기화 상태일 수 있다. 즉, detection pipeline 전체 관점에서는 pretrained knowledge가 절반만 전달되는 셈이다.

둘째, **task mismatch** 문제다. classification에서는 translation invariance나 aggressive crop augmentation이 유리할 수 있지만, detection에서는 object의 위치 정보가 중요하다. 따라서 classification에서 좋은 representation이 반드시 localization에 좋은 것은 아니다. 실제로 논문은 ImageNet Top-1 accuracy가 높다고 detection 성능이 항상 높은 것은 아니라고 강조한다.

### 3.7 Self-supervised object detection pretraining

이 문제를 해결하려는 것이 self-supervised object detection 방법들이다.

**UP-DETR** 과 **DETReg** 는 predictive approach다. 이들은 DETR의 detection head를 self-supervised하게 pretrain한다. 핵심 아이디어는 자동 생성한 region을 일종의 pseudo ground-truth object처럼 다시 예측하게 만드는 것이다. UP-DETR는 random crop을, DETReg는 **Selective Search** 가 만든 proposal을 사용한다. 원래 multiclass head 대신 binary foreground/background head를 두고, Hungarian matching 기반 loss로 pseudo object를 재탐지하게 만든다. 즉, detector가 object-like region을 찾는 기본 능력을 라벨 없이 미리 익히게 하는 것이다.

**InsLoc, ReSim, DenseCL** 은 local contrastive learning 접근이다. 이들은 image 전체가 아니라 crop, sliding window, feature 단위의 local representation을 서로 맞춘다. 예를 들어 DenseCL은 서로 다른 view에서 각 feature가 가장 cosine similarity가 큰 대응점을 찾도록 하여 dense correspondence를 유도한다. 이는 detection처럼 위치-sensitive한 downstream task에 더 적합하다는 생각에 기반한다.

**SoCo** 는 BYOL 스타일 self-distillation을 detector 단위로 확장한 방법이다. backbone뿐 아니라 FPN, RoI head까지 함께 pretrain한다. 논문에 제시된 식은 다음과 같다.

$$
h_1 = f_\theta^H(\text{RoIAlign}(v_1, b)), \quad
h_2 = f_\xi^H(\text{RoIAlign}(v_2, b))
$$

여기서 $v_1, v_2$는 두 view의 feature map이고, $b$는 Selective Search가 만든 proposal box다. 이후 embedding $e_1, e_2$의 cosine similarity를 최대화하도록 학습한다.

$$
L(\theta) = - \frac{\langle e_1, e_2 \rangle}{\|e_1\|_2 \cdot \|e_2\|_2}
$$

이 식의 의미는 같은 pseudo-object region이 서로 다른 augmentation view에서도 비슷한 embedding을 갖도록 강제하는 것이다.

## 4. 실험 및 결과

이 논문은 survey이므로 새로운 실험을 제시하기보다, 기존 benchmark와 대표 결과를 체계적으로 비교한다.

FSOD benchmark로는 주로 **PASCAL VOC**, **MS COCO**, **LVIS v0.5** 를 다룬다. PASCAL VOC는 15 base class와 5 novel class로 나뉘며, 1/2/3/5/10-shot 설정이 흔하다. MS COCO는 60 base, 20 novel class를 사용하며 10-shot과 30-shot 비교가 대표적이다. LVIS는 class 수가 매우 많지만, 논문은 현 split이 평가에 불리한 구조적 문제가 있다고 비판한다.

평가 지표는 PASCAL VOC에서는 주로 $nAP50$, COCO에서는 COCO-style $nAP$ 가 중심이다. 또한 $bAP$ 와 $nAP$ 를 분리해 base와 novel 성능을 각각 본다.

논문에 요약된 Table 3의 중요한 결과는 다음과 같다. 단순 finetuning 계열인 TFA가 매우 강한 baseline이며, 더 복잡한 modulation/prototype 방법들과 비교해도 경쟁력이 높다. COCO 30-shot에서 survey가 정리한 표 기준 최고 성능은 **DETReg** 로, $30.0$ nAP를 기록한다. 이는 self-supervised detector pretraining이 few-shot setting에서 특히 유효할 수 있음을 보여준다. 그 다음 상위권에는 DeFRCN, Meta-DETR, DAnA, TIP 등이 위치한다.

논문은 또 self-supervised object detection pretraining을 비교한 Table 4를 제시한다. 이 표에서 DETReg, SoCo, InsLoc, UP-DETR, ReSim, DenseCL 같은 object-detection-aware self-supervised 방법들이 정리된다. 절대 수치를 직접 비교하긴 어렵다고 저자들이 분명히 말하지만, 대체로 **backbone만 pretrain하는 방법보다 detector head까지 함께 pretrain하는 방법이 detection transfer에 유리한 경향** 이 있음을 보여준다. 예를 들어 DETReg는 unlabeled ImageNet pretraining 후 Pascal AP50 $83.3$, COCO AP $45.5$를 보고하며 strong baseline으로 소개된다.

중요한 해석은 다음과 같다. self-supervised pretraining은 일반 fully supervised detection에서는 개선 폭이 제한적일 수 있지만, **few-shot 또는 low-data detection에서는 훨씬 더 큰 이점** 을 보인다. 논문이 survey한 시점 기준으로 MS COCO FSOD state-of-the-art가 DETReg였다는 점이 이 주장을 뒷받침한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 정리 방식이 매우 구조적이라는 점이다. 단순히 논문 목록을 나열하지 않고, FSOD와 SSOD를 하나의 taxonomy 안에 넣고, backbone-only pretraining과 full detector pretraining을 구분하며, finetuning-only, prototype-based, modulation-based 같은 FSOD의 큰 계열을 명확히 정리한다. object detection 배경, benchmark, metric, 방법론, 추세를 한 문서 안에서 연결해 설명하기 때문에 입문자와 연구자 모두에게 유용하다.

또 다른 강점은 평가 관행에 대한 비판적 시각이다. 논문은 Kang split의 높은 분산, Pascal에서 고정 support set 사용으로 인한 overfitting, trainval 전체를 쓰고 test set으로 hyperparameter tuning하는 문제, LVIS split의 불균형 같은 점을 구체적으로 지적한다. survey 논문이 단순 종합이 아니라, 커뮤니티의 실험 관행 자체를 개선해야 한다고 주장한다는 점이 의미 있다.

방법론 해석도 균형 잡혀 있다. 논문은 복잡한 few-shot meta-learning 구조가 항상 우월한 것이 아니라, simple finetuning baseline이 매우 강하다고 정리한다. 또한 self-supervised learning이 detection에서 왜 classification보다 더 어려운지, 즉 task mismatch와 localization sensitivity를 잘 설명한다.

한계도 있다. 첫째, 이 논문은 survey이므로 새로운 통합 benchmark를 직접 만들거나, 동일한 구현 조건에서 재현 실험을 수행하지는 않는다. 따라서 표에 정리된 성능 비교는 참고용이지, 엄밀한 apples-to-apples 비교는 아니다. 저자들도 이를 명시한다.

둘째, 논문이 다루는 범위는 주로 2021~2022년 전후의 방법들에 맞춰져 있다. 따라서 이후 급속히 발전한 foundation model, vision-language model, large-scale pretraining 흐름은 반영되지 않는다. 물론 이것은 논문 시점의 한계이지 논문 자체의 오류는 아니다.

셋째, survey가 포괄적인 대신, 개별 방법의 수학적 세부 구현이나 ablation의 깊이는 제한적이다. 예를 들어 각 method의 loss weighting, training schedule, augmentation 차이까지 완전히 통제해 분석하지는 않는다. 이는 survey의 성격상 자연스럽지만, 실제로 새로운 연구를 설계하려는 독자에게는 원논문 추가 확인이 필요하다.

비판적으로 보면, 논문이 제안하는 “더 좋은 evaluation guideline”은 타당하지만, 아직 실제 커뮤니티 표준으로 자리잡기 위한 구체적 실행안까지 제시하는 것은 아니다. 예를 들어 새로운 split을 직접 제공하거나, reproducible benchmark suite를 함께 내놓는 수준은 아니다. 그럼에도 문제 제기 자체는 매우 정확하다.

## 6. 결론

이 논문은 few-shot object detection과 self-supervised object detection을 하나의 저데이터 detection 문제로 묶어 정리한 survey다. 주요 기여는 세 가지다. 첫째, FSOD의 문제 설정, benchmark, metric, 방법론을 체계적으로 정리했다. 둘째, self-supervised representation learning이 object detection에 적용될 때의 장점과 한계를 정리하고, detector head까지 pretraining하는 최근 흐름을 강조했다. 셋째, 현재 FSOD/SSOD 평가 절차의 불안정성과 과적합 가능성을 비판하면서 더 나은 benchmark 관행의 필요성을 제시했다.

논문이 전달하는 가장 중요한 메시지는 두 가지로 요약된다. 하나는 **단순한 finetuning baseline이 생각보다 강력하다**는 점이다. 다른 하나는 **self-supervised pretraining의 진짜 가치는 fully supervised setting보다 few-shot/low-data detection에서 더 크게 드러날 수 있다**는 점이다. 특히 DETReg 같은 방법이 few-shot COCO에서 매우 강한 성능을 보인다는 점은, 향후 object detection이 backbone pretraining을 넘어 detector 전체의 self-supervised pretraining으로 발전할 가능성을 시사한다.

실제 적용 측면에서도 의미가 크다. 데이터 라벨링이 어려운 산업, 의료, 생태, 로보틱스 환경에서, 적은 라벨과 많은 unlabeled image를 결합하는 전략은 매우 실용적이다. 향후 연구에서는 더 신뢰할 수 있는 benchmark, finetuning-free few-shot detection, transformer 및 vision-language foundation model과의 결합, 그리고 heuristic pseudo-label을 점진적으로 learned module로 대체하는 방향이 중요해질 가능성이 높다.
