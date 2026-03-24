# Camouflaged Instance Segmentation In-The-Wild: Dataset, Method, and Benchmark Suite

* **저자**: Trung-Nghia Le, Yubo Cao, Tan-Cong Nguyen, Minh-Quan Le, Khanh-Duy Nguyen, Thanh-Toan Do, Minh-Triet Tran, Tam V. Nguyen
* **발표연도**: 2021
* **arXiv**: <https://arxiv.org/abs/2103.17123>

## 1. 논문 개요

이 논문은 camouflage, 즉 배경과 매우 유사해서 사람이나 모델이 쉽게 놓칠 수 있는 객체를 단순히 “있다/없다” 수준으로 분할하는 것을 넘어, 개별 객체 단위의 **camouflaged instance segmentation** 문제를 새롭게 정의하고 체계적으로 다룬다. 기존의 camouflaged object segmentation 연구는 대체로 이미지 안에 camouflaged object가 반드시 존재한다고 가정하고, 픽셀을 camouflage/non-camouflage로만 나누는 **region-level segmentation**에 머물렀다. 반면 이 논문은 실제 환경처럼 camouflaged instance가 전혀 없을 수도 있는 unrestricted image에서, 각 픽셀에 instance identity까지 부여하는 더 어려운 문제를 대상으로 삼는다.

이 연구의 핵심 문제는 두 가지다. 첫째, 이 문제를 제대로 학습하고 평가할 수 있는 대규모 데이터셋이 없다는 점이다. 둘째, 기존 instance segmentation 모델을 그대로 fine-tuning하는 것만으로는 camouflage 특유의 어려움, 예를 들어 배경과의 극단적 유사성, 작은 객체, 복잡한 중첩과 가림, 다수 인스턴스 문제를 충분히 해결하기 어렵다는 점이다. 이를 해결하기 위해 저자들은 **CAMO++**라는 새 데이터셋을 구축하고, 여러 state-of-the-art instance segmentation 모델을 광범위하게 비교하는 benchmark suite를 제시하며, 추가로 여러 모델의 강점을 이미지 문맥에 따라 선택적으로 활용하는 **Camouflage Fusion Learning (CFL)** 프레임워크를 제안한다.

이 문제가 중요한 이유는 camouflage가 단지 생물학적 흥미에 머무르지 않기 때문이다. 논문은 search-and-rescue, 야생 동물 탐지, polyp segmentation, COVID-19 x-ray 분석, media forensics 같은 다양한 응용을 언급한다. 즉, 배경과 유사한 대상을 정밀하게 찾아내는 능력은 실제 비전 시스템의 견고성과 직결된다. 특히 instance 수준의 분할은 단순한 존재 탐지를 넘어서 “몇 개가 있는지”, “각 객체의 경계가 어디인지”를 다루므로, 실제 자동화 시스템에서 훨씬 유용하다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 세 축으로 정리할 수 있다.

첫째, **문제 정의의 확장**이다. 저자들은 camouflaged object segmentation과 camouflaged instance segmentation을 명확히 구분한다. 전자는 이미지 내의 camouflage 영역 전체를 하나의 foreground처럼 다루는 반면, 후자는 그 영역을 의미 있는 개별 인스턴스로 분해해야 한다. 즉, 단순한 binary mask가 아니라 instance-aware mask가 필요하다. 이 차이는 문제 난도를 크게 높이며, 기존 camouflage segmentation 연구와의 가장 큰 차별점이다.

둘째, **in-the-wild 설정의 도입**이다. 기존 camouflage 연구는 대체로 이미지에 camouflaged object가 항상 존재한다고 가정했지만, 논문은 이 가정이 실제 상황과 다르다고 지적한다. 현실에서는 어떤 이미지에는 camouflage 대상이 아예 없을 수 있다. 따라서 CAMO++는 camouflage 이미지와 non-camouflage 이미지를 약 50:50 비율로 포함하며, 두 경우 모두 annotation을 제공한다. 이 점은 데이터셋 설계 철학 차원에서 기존 COD나 CAMO와 분명히 구별된다.

셋째, **모델 선택 자체를 학습하는 fusion 전략**이다. 저자들은 하나의 instance segmentation 모델이 모든 이미지 문맥에서 최선이 아니라고 본다. 어떤 모델은 작은 객체에 강하고, 어떤 모델은 중간 크기나 큰 객체, 혹은 특정 배경 조건에서 유리할 수 있다. 그래서 여러 개의 segmentation 모델을 독립적으로 학습한 뒤, 각 이미지마다 어느 모델의 출력이 가장 좋은지 pseudo label을 만들고, 그 선택을 예측하는 별도의 model predictor를 학습한다. 즉, segmentation 결과를 직접 late fusion하는 것이 아니라, **이미지 문맥을 보고 “어떤 모델을 쓸지”를 고르는 scene-driven model selection**이 CFL의 핵심 직관이다.

기존 접근과의 차별점은 명확하다. 기존 camouflaged object segmentation은 instance-level이 아니며, 일반 instance segmentation은 camouflage를 특별히 고려하지 않는다. 이 논문은 그 사이의 공백을 메우면서, 단순히 새 데이터셋만 내놓는 것이 아니라 “어떤 기존 instance segmentation 방법이 camouflage 환경에서 어떻게 실패하는지”를 벤치마크하고, 이를 보완하기 위한 선택형 fusion 구조까지 제안했다는 점에서 의미가 있다.

## 3. 상세 방법 설명

### 3.1 전체 구조

논문의 방법론은 크게 두 부분으로 나뉜다. 하나는 **CAMO++ 데이터셋 구축**이고, 다른 하나는 그 데이터셋 위에서 동작하는 **CFL 프레임워크**다. 실제 모델링 측면에서 보면 CFL은 다음의 2-stage 구조를 가진다.

먼저 1단계에서는 여러 개의 instance segmentation 모델을 CAMO++에서 각각 독립적으로 학습한다. 논문에서 사용한 모델은 총 5개이며, Mask RCNN, Cascade Mask RCNN, MS RCNN, RetinaMask, CenterMask이다. 이들은 모두 기존 공개 구현을 사용하며, 각 모델은 자기 원래의 instance segmentation loss로 학습된다.

2단계에서는 이들 모델을 freeze한 뒤, 각 학습 이미지에 대해 “어느 모델이 가장 좋은 결과를 냈는가”를 찾는 search algorithm을 수행한다. 그렇게 얻은 모델 ID를 pseudo label처럼 사용해, 입력 이미지를 보고 가장 적절한 모델을 예측하는 model predictor를 학습한다. 저자들은 이 predictor로 **ViT-Base16**을 사용했다.

즉, 최종 시스템은 “모든 모델의 출력을 평균내는 방식”이 아니다. 대신, 입력 이미지 $x$가 들어오면 predictor $f(x)$가 가장 적절한 모델을 고르고, 그 모델의 segmentation 결과를 사용한다는 구조로 이해할 수 있다.

### 3.2 CAMO++ 데이터셋 설계

데이터셋 자체도 이 논문의 중요한 방법 요소다. CAMO++는 기존 CAMO를 확장한 데이터셋으로, 총 **5,500장**의 이미지로 구성된다. 이 중 **2,700장**은 camouflage 이미지이고, **2,800장**은 non-camouflage 이미지다. 전체 인스턴스 수는 **32,756개**다.

camouflage 이미지는 인터넷에서 다양한 검색어 조합으로 수집했다. 예를 들어 “camouflaged”, “concealed”, “hidden” 같은 형용사와 동물 이름, 사람 관련 키워드, 혹은 환경 키워드를 결합해 찾았다. 이후 저해상도 이미지와 중복 이미지를 제거하고, 최종적으로 2,700장의 camouflage 이미지를 확보했다. 10명의 annotator가 custom interactive segmentation tool을 이용해 각 이미지의 camouflaged instance를 표시하고 hierarchical category를 부여했다.

non-camouflage 이미지는 LVIS 데이터셋에서 사람이거나 동물 인스턴스를 포함하는 이미지 2,800장을 수동 선택해 만들었다. 중요한 점은 이 이미지들에 camouflaged instance가 없도록 사람이 검수했다는 것이다. 저자들은 이것이 false positive segmentation을 줄이고 실제 환경을 더 잘 모사하는 데 필요하다고 본다.

annotation은 hierarchical하다. 논문은 meta-category label, fine-category label, bounding box, instance-level mask를 모두 제공한다고 설명한다. 생물학 기반 taxonomy로는 13개 meta-category와 93개 category가 있고, 비전 관점 재구성 taxonomy로는 8개 meta-category가 있다. 이 hierarchical annotation은 단지 segmentation 데이터셋을 넘어서, 범주 다양성과 분석 가능성을 높여준다.

### 3.3 데이터셋 난이도와 구조적 특징

논문은 CAMO++가 기존 데이터셋보다 왜 어려운지 여러 통계로 설명한다.

먼저 **instance density**가 높다. 이미지당 평균 6.0개의 instance가 있으며, 51%의 이미지가 multiple instances를 가진다. 그중 38%는 2~10개, 10%는 11~30개, 3%는 30개 초과의 instance를 포함한다. 이는 COD의 평균 1.2개보다 훨씬 복잡하다.

또한 **small instance 비중**이 매우 높다. 논문에 따르면 mask size 기준으로 small instance가 전체의 69.6%, medium이 23.8%다. tiny instance도 상당수 포함되어 있다. 이 특성은 일반적인 instance segmentation 모델에 매우 불리하다. COCO 스타일 모델들은 대개 작은 객체에서 성능이 급격히 떨어지는 경향이 있기 때문이다.

게다가 객체 중심이 이미지 전체에 고르게 퍼져 있어, 기존 camouflage 데이터셋에 존재하던 center bias가 상대적으로 완화되어 있다. 배경 clutter, shape complexity, occlusion, distraction 역시 유지된다. 결국 CAMO++는 “camouflage 특유의 시각적 모호성”과 “instance segmentation 특유의 다중 객체/작은 객체 문제”가 동시에 존재하는 어려운 벤치마크다.

### 3.4 Model Search 알고리즘

CFL의 핵심은 각 이미지에 대해 “어느 모델이 최선인가”를 찾아 pseudo label을 만드는 것이다. 논문은 이를 Algorithm 1로 설명한다.

입력은 $N$개 이미지에 대해 $K$개 모델이 예측한 segmentation 결과 집합 $ins[N \times K]$와 ground truth 리스트 $gt[N]$이다. 출력은 각 이미지의 최종 prediction 리스트 $pred[N]$와, 그 이미지에서 가장 좋았던 모델 ID 리스트 $mod[N]$이다.

알고리즘은 greedy한 방식으로 동작한다. 현재 $i$번째 이미지를 처리할 때, 이미 이전 이미지들에 대해 선택된 결과들이 prediction list에 들어 있다. 그런 다음 각 후보 모델 $k$에 대해, “현재까지의 prediction list”에 $i$번째 이미지에서 해당 모델의 예측 결과 $ins[i,k]$를 추가했을 때 전체 AP가 얼마나 되는지 계산한다. 그중 AP를 최대화하는 모델 $\hat{k}$를 선택한다.

즉, 식으로 적으면 선택은 다음과 같다.

$$
\hat{k} \leftarrow \arg\max_{k \in K} AP\big(gt[1:i],\ pred \cup ins[i,k]\big)
$$

그 후 $ins[i,\hat{k}]$를 prediction list에 추가하고, 이미지 $i$의 pseudo label을 모델 $\hat{k}$로 기록한다. 이 절차를 전체 이미지에 대해 반복한다.

이 알고리즘의 의미는 단순한 per-image local score 비교가 아니라, 전체 AP 관점에서 각 이미지에 가장 기여하는 모델을 고른다는 점이다. 다만 논문은 이 greedy 방식이 전역 최적(global optimum)을 보장하는지에 대해서는 설명하지 않는다. 따라서 이는 실용적 pseudo-label 생성 절차로 보는 것이 맞다.

### 3.5 손실 함수

논문은 전체 손실을 두 부분으로 구성한다.

$$
\mathcal{L} = \mathcal{L}*{segm} + \mathcal{L}*{pred}
$$

여기서 $\mathcal{L}*{segm}$은 instance segmentation loss, $\mathcal{L}*{pred}$는 model prediction loss다.

instance segmentation loss는 다음과 같이 표현된다.

$$
\mathcal{L}*{segm}(x) = \sum*{g=1}^{M} c_g(x)\times \mathcal{L}_{ins}^{g}(g(x), y)
$$

여기서 $M$은 사용한 segmentation 모델 수이고, $g(x)$는 $g$번째 모델의 예측, $y$는 ground truth다. $c$는 one-hot 벡터처럼 동작하여, Algorithm 1에서 선택된 최적 모델에 대해서만 $c_i=1$이고 나머지는 0이다. 결국 실제로는 “현재 이미지에서 pseudo-label로 선택된 모델의 segmentation loss만 반영한다”는 뜻이다.

다만 논문 설명상 2단계에서는 segmentation 모델들을 freeze한다고 되어 있으므로, 실제 학습 과정에서 중요한 것은 model predictor를 위한 $\mathcal{L}_{pred}$라고 보는 편이 자연스럽다. 이 부분은 수식 표기와 서술 사이에 다소 추상화가 있으며, 논문은 완전한 end-to-end joint optimization 절차를 자세히 풀어 설명하지는 않는다.

model prediction loss는 multinomial logistic regression을 위한 cross entropy다.

$$
\mathcal{L}_{pred}(x) = - c(x)\cdot \log(f(x))
$$

여기서 $f(x)$는 입력 이미지 $x$에 대해 각 모델이 최적일 확률을 출력하는 predictor다. 즉, predictor는 “이 이미지는 Mask RCNN 계열이 유리한가, Cascade가 유리한가, MS RCNN이 유리한가” 같은 선택 문제를 푸는 multi-class classifier다.

### 3.6 구현 및 학습 전략

model predictor는 **ViT-Base16**을 사용했다. ImageNet pre-trained weight로 초기화하고 CAMO++에서 fine-tuning했다. 학습 전략으로는 **five-fold stratified sampling**을 사용해 훈련 4 fold, 검증 1 fold로 나누었다. 이는 overfitting 완화를 위한 선택이다.

augmentation은 resizing, cropping, translation, rotation, flipping을 사용했다. 학습은 100 epochs, batch size 16, base learning rate 0.0008, warmup 1000 steps, cosine decay를 적용했다. optimizer는 AdamW이며 weight decay는 0.001, 모멘텀 계수는 $\beta_1=0.9$, $\beta_2=0.999$다.

실험 환경은 64GB RAM PC와 Tesla P100 GPU다. baseline segmentation 모델들은 MS-COCO pretrained 모델에서 fine-tuning했다. backbone은 ResNet50-FPN, ResNet101-FPN, ResNeXt101-FPN을 주로 사용했고, YOLACT와 BlendMask는 구현 제약으로 ResNeXt101-FPN이 없었다.

## 4. 실험 및 결과

### 4.1 실험 설정

논문은 CAMO++ 위에서 8개의 state-of-the-art instance segmentation 방법을 벤치마크했다. 구체적으로 Mask RCNN, Cascade Mask RCNN, MS RCNN, RetinaMask, YOLACT, CenterMask, SOLO, BlendMask를 비교했다. 그리고 CFL은 이들 중 일부를 component model로 사용한 선택형 fusion 방식이다.

평가는 COCO-style metric을 따른다. 평균 정밀도는 $AP$, $AP_{50}$, $AP_{75}$를 보고, scale별로 $AP_S$, $AP_M$, $AP_L$를 본다. 평균 재현율은 $AR_1$, $AR_{10}$, $AR_{100}$과 $AR_S$, $AR_M$, $AR_L$를 사용한다. 따라서 평가는 단순히 하나의 숫자만 보는 것이 아니라, 다양한 IoU 기준과 객체 크기, 제안 수 제한까지 폭넓게 점검한다.

논문은 두 가지 실험 설정을 둔다.

첫 번째는 **Setting 1**로, 실제 환경을 모사한 unrestricted setting이다. 여기서는 camouflaged instance가 이미지에 없을 수도 있다. 따라서 camouflage 이미지와 non-camouflage 이미지를 모두 포함해 학습과 테스트를 수행한다.

두 번째는 **Setting 2**로, 이미지마다 camouflaged instance가 반드시 있다고 가정한다. 이 경우 camouflage 이미지만 사용한다. 이는 기존 camouflage segmentation 연구와 좀 더 가까운 설정이다.

### 4.2 Setting 1: camouflaged instances are not always present

이 설정은 가장 현실적인 시나리오다. 전체적으로 backbone이 강할수록 성능이 좋아졌고, 특히 ResNeXt101-FPN 기반 구현들이 대체로 최고 성능을 냈다. baseline 중에서는 MS RCNN, Cascade Mask RCNN, BlendMask, RetinaMask 등이 지표에 따라 엇갈리게 강세를 보였다. 저자들이 강조하듯 “하나의 절대적 지배 모델”은 없었다.

대표적으로 Table III에서 baseline 최고 성능을 보면, MS RCNN with ResNeXt101-FPN이 $AP=22.9$를 기록했고, Cascade Mask RCNN with ResNeXt101-FPN은 $AP=21.5$, BlendMask with ResNet101-FPN은 $AP=20.3$, RetinaMask with ResNeXt101-FPN은 $AP=20.0$ 수준이다. 작은 객체 성능에서는 RetinaMask가 일부 지표에서 강하고, 중대형 객체에서는 MS RCNN이나 Cascade 계열이 더 나은 모습이 나타난다.

하지만 CFL은 이 이질적인 장점을 결합해 모든 주요 지표에서 최고 성능을 달성했다. Table III에 따르면 CFL의 성능은 다음과 같다.

* ResNet50-FPN: $AP=19.2$, $AP_{50}=39.0$, $AP_{75}=16.5$
* ResNet101-FPN: $AP=21.9$, $AP_{50}=41.9$, $AP_{75}=21.2$
* ResNeXt101-FPN: $AP=25.1$, $AP_{50}=47.2$, $AP_{75}=24.1$

특히 ResNeXt101-FPN 기준으로 baseline 최고였던 MS RCNN의 $AP=22.9$보다 CFL의 $AP=25.1$이 더 높다. 절대 수치만 보면 여전히 높다고 말하기는 어렵다. 저자들도 이를 인정하며, unrestricted in-the-wild setting에서는 top-1 AP가 25 이하 수준이라 아직 문제가 매우 어렵다고 해석한다. 이 점은 논문의 정직한 메시지다. 즉, CFL이 state-of-the-art이기는 하지만 문제 자체가 충분히 해결된 것은 아니다.

### 4.3 Setting 2: camouflaged instances are always present

camouflage 객체가 항상 있다고 가정한 Setting 2에서는 모든 모델의 성능이 전반적으로 상승한다. 이는 당연한 결과다. 배경-only 이미지가 제거되면서 탐지 난도가 줄기 때문이다.

Table IV에서 baseline 중 강한 결과를 보면, Cascade Mask RCNN with ResNeXt101-FPN이 $AP=33.6$, MS RCNN with ResNeXt101-FPN이 $AP=31.0$, Mask RCNN with ResNeXt101-FPN이 $AP=31.2$를 기록했다. 즉, 객체 존재가 보장될 때는 기존 instance segmentation 구조도 꽤 높은 성능을 낼 수 있다.

그러나 CFL은 여기서도 모든 지표에서 최상위다.

* ResNet50-FPN: $AP=35.2$, $AP_{50}=66.3$, $AP_{75}=32.6$
* ResNet101-FPN: $AP=36.9$, $AP_{50}=68.2$, $AP_{75}=34.7$
* ResNeXt101-FPN: $AP=42.8$, $AP_{50}=75.9$, $AP_{75}=44.6$

특히 ResNeXt101-FPN에서 $AP=42.8$은 baseline 최고인 33.6보다 꽤 큰 폭의 향상이다. 이는 모델 선택형 fusion이 camouflage가 존재하는 장면에서는 더욱 효과적일 수 있음을 시사한다. 객체가 반드시 존재하는 경우, predictor가 scene cue를 바탕으로 어떤 모델이 해당 패턴에 잘 맞는지 더 안정적으로 고를 수 있었을 가능성이 있다.

### 4.4 정성적 결과와 failure case

Figure 11의 정성 비교에서 CFL은 ground truth에 가장 가까운 결과를 보였다고 저자들은 주장한다. 특히 fine detail, complex shape, 유사한 색/질감, multiple object 상황에서 더 견고한 모습을 보였다. 이는 정량 결과와도 일치한다.

하지만 Figure 12의 failure case는 문제의 본질적 난도를 잘 보여준다. 논문이 지적한 주요 실패 원인은 세 가지다. 첫째, **tiny instances**다. 매우 작은 camouflage 객체는 사람도 놓치기 쉬우며, 모델도 위치 추정과 mask 복원에 실패한다. 둘째, **extreme resemblance to background**다. 객체와 배경이 너무 비슷하면 localization 자체가 붕괴한다. 셋째, **occluded and overlapping instances**다. 서로 가려지거나 겹친 여러 camouflage 객체는 잘못 분리되거나 non-camouflaged instance와 혼동될 수 있다.

이 정성 분석은 중요한 의미가 있다. 단순히 “성능이 좋아졌다”는 주장보다, 어디에서 아직 실패하는지를 명시함으로써 향후 연구 방향을 제시하기 때문이다.

### 4.5 non-camouflage 이미지 추가 학습 효과

논문은 추가 분석으로 non-camouflage 이미지를 함께 학습에 넣었을 때의 효과를 측정했다. 이는 흥미로운 실험이다. camouflage segmentation 문제에서 일반 객체 이미지를 넣는 것이 오히려 도움이 될 수 있는지 보는 것이다.

Table VI에 따르면 대부분의 모델에서 성능 향상이 관찰된다. 예를 들어:

* CenterMask ResNeXt101-FPN: $AP$가 26.6에서 29.9로 증가
* SOLO ResNeXt101-FPN: 24.3에서 33.3으로 큰 폭 증가
* RetinaMask ResNeXt101-FPN: 25.8에서 28.1로 증가
* CFL ResNeXt101-FPN: 42.8에서 43.7로 증가

즉, 일반적인 non-camouflaged instance 정보도 camouflaged instance segmentation 성능 향상에 기여한다. 저자들은 이를 “additional non-camouflaged instance data helps boost performance”라고 해석한다. 다만 모든 지표가 항상 좋아지는 것은 아니고, 일부 조합에서는 $AP_{75}$ 또는 $AP_{50}$가 소폭 감소하기도 한다. 따라서 이 효과는 모델 구조에 따라 다르며, 단순히 데이터 양 증가 때문인지, 더 일반적인 object prior 학습 때문인지는 논문에서 깊게 분리 분석하지는 않는다.

### 4.6 cross-dataset generalization

논문은 CAMO++와 COD 간 cross-dataset generalization도 평가했다. 동일한 수의 camouflage 이미지로 맞춘 뒤, Cascade Mask RCNN with ResNeXt101-FPN으로 학습-테스트 교차 평가를 수행했다.

Table V에 따르면, CAMO++로 학습했을 때 CAMO++ 테스트에서는 33.6, COD 테스트에서는 23.2를 얻었고, COD로 학습했을 때 CAMO++ 테스트에서는 27.9, COD 테스트에서는 31.4를 얻었다. 이를 평균하면 CAMO++ training set의 mean-generalizability는 30.8, COD training set은 27.3이다. 반면 테스트셋 난이도 측면에서 CAMO++는 mean-difficulty 28.4, COD는 29.7로 표기되어 있는데, 논문의 해석은 CAMO++가 더 challenging하다는 것이다. 표의 방향성은 조금 조심해서 봐야 하지만, 저자들은 CAMO++의 테스트셋이 tiny object, extreme resemblance, distraction, occlusion/overlap 등 어려운 사례가 많아 더 어렵다고 해석한다.

핵심은 CAMO++가 단지 내부용으로만 유리한 데이터셋이 아니라, 다른 데이터셋으로도 어느 정도 일반화되는 풍부한 training signal을 제공한다는 주장이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의, 데이터셋, 벤치마크, 방법론**이 한 논문 안에서 유기적으로 연결되어 있다는 점이다. 단순히 새 데이터셋만 내놓은 것도 아니고, 새 모델만 제안한 것도 아니다. camouflaged instance segmentation이라는 새로운 문제를 정의하고, 이를 학습 가능한 규모의 CAMO++ 데이터셋을 만들고, 다양한 baseline을 공정하게 비교하며, 그 위에 실질적인 성능 향상을 주는 CFL까지 제안했다. 이런 구성은 커뮤니티에 미치는 영향이 크다.

두 번째 강점은 **in-the-wild 설정의 현실성**이다. 이미지에 항상 camouflage 객체가 있다고 가정하지 않는다는 점은 실제 배치 환경에 더 가깝다. 또한 camouflage와 non-camouflage 이미지를 모두 annotation한 것은 데이터셋 설계 측면에서 매우 유의미하다.

세 번째 강점은 **난이도 분석의 충실함**이다. category diversity, image dimension, instance density, mask size, center bias, generalization 같은 다양한 관점에서 데이터셋 특성을 정리했다. 이는 후속 연구자가 단순히 성능 숫자만 보는 것이 아니라, 왜 어려운지 이해하게 해 준다.

네 번째 강점은 **CFL의 실용적 설계**다. 모델 앙상블을 무겁게 구성하는 대신, 이미지별로 최적 모델을 고르는 predictor를 학습하는 방식은 직관적이고 구현도 비교적 단순하다. 특히 여러 baseline 사이에 단일 우승자가 없다는 실험 결과와 잘 맞물린다. 즉, 문제 구조 자체가 “scene-dependent model preference”를 갖는다는 것을 방법과 결과가 함께 뒷받침한다.

한편 한계도 분명하다.

가장 먼저, **성능 자체가 아직 낮다**. 특히 unrestricted setting에서 CFL조차 $AP=25.1$이 최고다. 이는 문제의 난도가 높다는 증거이기도 하지만, 동시에 제안 방법이 문제를 근본적으로 해결했다고 보기는 어렵다는 뜻이다. 논문도 이 점을 솔직하게 인정한다.

둘째, **CFL의 fusion 방식이 모델 선택에 머문다**. 즉, 여러 모델의 출력을 더 세밀하게 결합하거나, mask-level complementary signal을 통합하는 구조는 아니다. predictor가 한 모델만 선택하는 hard selection에 가까운 구조이므로, 서로 다른 모델의 일부 강점을 동시에 활용하는 soft fusion까지는 가지 못했다.

셋째, **search algorithm의 최적성 보장이 불분명**하다. Algorithm 1은 greedy하게 pseudo label을 만든다. 이는 실용적이지만, 전역적으로 최적의 model assignment를 찾는다는 보장은 없다. pseudo label 품질이 predictor 성능을 직접 좌우하므로, 이 부분은 향후 개선 여지가 있다.

넷째, **모델 predictor와 segmentation 모델의 end-to-end 상호작용이 제한적**이다. 논문 서술상 2단계에서 segmentation 모델을 freeze하므로, predictor가 선택 실수로 인한 downstream error를 줄이도록 segmentation feature 자체를 공동 적응시키는 구조는 아니다. joint training을 더 깊게 설계했다면 추가 개선 가능성이 있었을 수 있다. 그러나 이는 논문에서 실제로 수행한 내용이 아니므로 가능성 수준에서만 말할 수 있다.

다섯째, **데이터셋 구성의 범위 제한**도 있다. 사람과 90개 이상의 동물 종을 포함하지만, camouflage 현상이 발생할 수 있는 모든 대상과 환경을 포괄한다고 보기는 어렵다. 또한 비디오 기반 dynamic camouflage는 다루지 않는다. 논문 역시 향후 video scene과 motion cue 확장을 과제로 제시한다.

비판적으로 보면, 이 논문은 매우 좋은 benchmark paper이자 problem-defining paper이지만, 방법론 자체는 혁신적이라기보다 **강한 baseline ensemble selection**에 가깝다. 그럼에도 이 분야 초기 단계에서는 이 정도의 실용적 접근이 오히려 적절하며, 데이터셋과 벤치마크가 함께 제시되었다는 점에서 충분히 가치가 크다.

## 6. 결론

이 논문은 camouflaged instance segmentation이라는 새로운 비전 문제를 본격적으로 정의하고, 이를 지원하는 대규모 데이터셋 **CAMO++**와 benchmark suite를 제시했다는 점에서 의미가 크다. CAMO++는 5,500장의 이미지, 32,756개의 instance, camouflage/non-camouflage 혼합 설정, hierarchical annotation을 통해 기존 camouflage 데이터셋보다 훨씬 현실적이고 어려운 평가 환경을 제공한다.

방법론 측면에서는 여러 기존 instance segmentation 모델의 장단점을 활용하기 위해 **Camouflage Fusion Learning (CFL)**을 제안했다. CFL은 각 이미지 문맥에 따라 최적의 모델을 선택하는 predictor를 학습하며, 실제로 두 실험 설정 모두에서 기존 방법들을 일관되게 앞섰다. 특히 camouflaged instance가 항상 존재하는 설정에서는 ResNeXt101-FPN 기준 $AP=42.8$, unrestricted setting에서는 $AP=25.1$을 달성했다.

실험은 또한 두 가지 중요한 메시지를 준다. 하나는 camouflage instance segmentation이 여전히 매우 어렵다는 점이다. 다른 하나는 non-camouflage 이미지 같은 일반 객체 데이터도 도움이 될 수 있고, 데이터셋 설계와 학습 설정이 성능에 큰 영향을 준다는 점이다.

실제 적용 측면에서 이 연구는 search-and-rescue, 야생 동물 모니터링, 의료 영상 분석, 위조 탐지처럼 “배경과 섞여 보이는 대상”을 정밀하게 분리해야 하는 문제들에 기초를 제공할 수 있다. 향후 연구 측면에서는 context modeling, soft fusion, stronger small-object reasoning, video motion cue 활용, occlusion-aware instance parsing 같은 방향으로 이어질 가능성이 높다. 요약하면, 이 논문은 camouflaged instance segmentation 분야의 출발점을 마련한 데이터셋 및 벤치마크 중심의 핵심 논문으로 평가할 수 있다.
