# Contrastive learning of global and local features for medical image segmentation with limited annotations

- **저자**: Krishna Chaitanya, Ertunc Erdil, Neerav Karani, Ender Konukoglu
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2006.10511

## 1. 논문 개요

이 논문은 **라벨이 매우 적은 상황에서 medical image segmentation 성능을 높이기 위한 self-supervised pre-training 방법**을 제안한다. 문제의 출발점은 분명하다. 의료영상 분할에서는 supervised deep learning이 매우 강력하지만, 실제 임상 데이터에서는 대규모 정답 마스크(annotation)를 확보하기가 어렵다. 특히 MRI나 CT처럼 3D volume 단위로 라벨링해야 하는 경우, 전문 인력과 시간이 많이 필요하므로 대량 라벨링이 현실적으로 어렵다.

저자들은 이 문제를 해결하기 위해 contrastive learning 기반의 self-supervised learning(SSL)을 segmentation에 맞게 확장한다. 기존 contrastive learning은 주로 image-level representation을 잘 학습하는 데 초점을 두었고, classification 같은 image-wise task에는 잘 맞지만, segmentation처럼 **pixel-wise prediction**이 필요한 문제에는 충분하지 않다고 본다. 또한 기존 방식은 대체로 “같은 이미지에 augmentation 두 개를 적용한 쌍”만 positive로 사용하는데, 의료영상에는 그보다 더 강한 구조적 단서가 있다. 예를 들어 서로 다른 환자의 cardiac MRI라도 대략 정렬되어 있다면 비슷한 해부학적 위치의 slice는 서로 유사한 정보를 담고 있다.

따라서 이 논문의 연구 문제는 크게 두 가지다. 첫째, **segmentation에 유리한 local representation을 contrastive learning으로 학습할 수 있는가**이다. 둘째, **volumetric medical images가 가진 구조적 유사성을 positive/negative pair 정의에 반영하면 더 좋은 pre-training이 가능한가**이다. 이 문제는 의료영상 분야에서 매우 중요하다. 실제 적용 환경에서는 라벨이 부족한 경우가 일반적이므로, unlabeled data를 더 잘 활용해 few-label setting에서도 안정적인 segmentation 성능을 확보하는 기술이 임상적으로 큰 가치가 있다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 기존 contrastive learning을 그대로 쓰지 않고, **global feature와 local feature를 분리해서 각각 학습**하되, 의료영상 도메인의 구조를 contrastive pair 구성에 적극 반영하는 것이다.

첫 번째 핵심은 **global contrastive learning의 pair 정의를 의료영상 volume 구조에 맞게 바꾸는 것**이다. 기존 방식에서는 같은 이미지의 두 augmentation만 positive로 보고, 나머지 이미지는 모두 negative로 본다. 하지만 MRI/CT volume은 서로 다른 환자여도 같은 anatomical region의 slice끼리는 비슷한 내용을 담는다. 저자들은 각 volume을 여러 partition으로 나누고, 서로 다른 volume에서 같은 partition에 속한 slice들을 유사한 샘플로 본다. 즉, positive pair의 범위를 “같은 이미지의 augmentation 쌍”에서 “서로 다른 volume이지만 같은 해부학적 구간에 해당하는 slice들”까지 확장한다.

두 번째 핵심은 **local contrastive loss**다. segmentation은 단순히 이미지 전체를 잘 표현하는 것만으로 충분하지 않고, 이미지 내부의 서로 다른 위치를 구분할 수 있어야 한다. 저자들은 encoder가 global representation을 학습한 뒤, decoder 일부를 따로 pre-train하여 **같은 위치의 local region은 intensity transformation에 대해 invariant하게 유지하면서, 다른 위치의 local region과는 구별되도록** 학습한다. 이것은 segmentation decoder가 boundary나 구조별 지역적 차이를 더 잘 이용하도록 돕는다.

기존 접근과의 차별점은 명확하다. 이 논문은 단순히 더 좋은 image-level representation을 얻으려는 것이 아니라, **segmentation에 직접 필요한 local distinctiveness를 목표로 contrastive objective를 설계**한다. 또한 자연영상 기반 contrastive learning에서 흔한 augmentation-only positive pair 정의를 넘어서, **volumetric medical data의 slice correspondence**라는 도메인 지식을 반영한다는 점이 주요 차별점이다.

## 3. 상세 방법 설명

전체 방법은 두 단계 pre-training 후 fine-tuning으로 이루어진다.

첫 단계에서는 encoder $e(\cdot)$와 projection head $g_1(\cdot)$를 사용해 **global contrastive loss** $L_g$로 pre-train한다. 두 번째 단계에서는 encoder를 고정한 뒤 decoder의 앞부분 $d_l(\cdot)$와 또 다른 head $g_2(\cdot)$를 붙여 **local contrastive loss** $L_l$로 decoder 일부를 pre-train한다. 마지막으로 $g_1, g_2$는 버리고, 남은 decoder block을 랜덤 초기화하여 전체 UNet을 적은 수의 라벨 데이터로 fine-tune한다.

### 3.1 Global contrastive loss

기본 global contrastive loss는 SimCLR류의 loss를 따른다. 같은 원본 이미지 $x$에서 서로 다른 augmentation을 적용해 얻은 $\tilde{x}$와 $\hat{x}$를 positive pair로 보고, 나머지 이미지들을 negative로 본다. 표현은 encoder와 projection head를 통과해

$$
\tilde{z} = g_1(e(\tilde{x})), \qquad \hat{z} = g_1(e(\hat{x}))
$$

로 정의된다.

한 positive pair에 대한 loss는 다음과 같다.

$$
l(\tilde{x}, \hat{x}) = - \log \frac{e^{\mathrm{sim}(\tilde{z}, \hat{z})/\tau}}
{e^{\mathrm{sim}(\tilde{z}, \hat{z})/\tau} + \sum_{\bar{x} \in \Lambda^-} e^{\mathrm{sim}(\tilde{z}, g_1(e(\bar{x})))/\tau}}
$$

여기서 $\mathrm{sim}(a,b)=\frac{a^Tb}{\|a\|\|b\|}$ 는 cosine similarity이고, $\tau$는 temperature parameter다. 전체 global loss는 모든 positive pair에 대해 양방향으로 평균낸 형태다.

$$
L_g = \frac{1}{|\Lambda^+|} \sum_{(\tilde{x},\hat{x}) \in \Lambda^+} \left[l(\tilde{x},\hat{x}) + l(\hat{x},\tilde{x})\right]
$$

핵심은 $\Lambda^+$와 $\Lambda^-$를 어떻게 정의하느냐다.

### 3.2 도메인 지식을 활용한 global contrasting strategy

저자들은 volume이 대략 정렬되어 있다는 가정 아래, 각 3D volume을 $S$개의 partition으로 나눈다. 예를 들어 심장 MRI라면, 위쪽 slice 집합, 중간 위쪽 집합, 중간 아래쪽 집합, 아래쪽 집합처럼 나누는 셈이다. $i$번째 volume의 $s$번째 partition에서 선택한 이미지를 $x_i^s$라고 두면, 서로 다른 volume의 같은 partition에 속한 이미지들 $x_i^s, x_j^s$는 비슷한 anatomical area를 담고 있다고 본다.

논문은 세 가지 global strategy를 비교한다.

`G_R`은 기본 random strategy다. 같은 이미지의 augmentation 쌍만 positive로 보고, 배치 내 나머지는 모두 negative다.

`G_D^-`는 대응되는 partition끼리는 negative로 두지 않는 전략이다. 즉, $s$번째 partition의 이미지를 anchor로 잡았을 때 negative set에는 **다른 partition**의 이미지들만 포함한다. 이는 실제로 비슷할 수 있는 같은 partition의 타 volume slice를 negative로 밀어내지 않도록 한다.

`G_D`는 한 단계 더 나아가, 서로 다른 volume이지만 같은 partition에 있는 이미지들도 positive로 적극 활용한다. 즉, $(x_i^s, x_j^s)$ 같은 쌍과 그 augmentation 버전까지 positive에 포함한다. negative는 여전히 다른 partition에서만 뽑는다. 이 전략은 representation space에서 partition별 cluster가 생기도록 유도한다.

직관적으로 보면, `G_D^-`는 “헷갈리는 샘플을 negative에서 빼는 방식”이고, `G_D`는 “헷갈리는 샘플을 positive로 적극 활용하는 방식”이다.

### 3.3 Local contrastive loss

이 논문의 중요한 기여는 local contrastive loss다. segmentation에서는 이미지 전체 representation만 좋아서는 부족하고, 각 위치의 feature가 주변과 충분히 구분되어야 한다. 이를 위해 저자들은 encoder 뒤에 decoder의 앞부분 $d_l$을 붙여 feature map을 만들고, 그 안의 local region들 사이에서 contrastive learning을 수행한다.

입력 이미지 $x$의 두 intensity-transformed 버전 $\tilde{x}, \hat{x}$를 사용해 다음 feature map을 얻는다.

$$
\tilde{f} = g_2(d_l(e(\tilde{x}))), \qquad \hat{f} = g_2(d_l(e(\hat{x})))
$$

이 feature map을 $A$개의 local region으로 나눈다. 각 region의 크기는 $K \times K \times C$다. 같은 위치 $(u,v)$에 있는 region은 두 변환 사이에서도 같은 의미를 가져야 하므로 positive pair가 된다. 반면 feature map 내의 다른 위치 $(u',v')$의 region들은 negative로 취급한다. 한 local pair에 대한 loss는 다음과 같다.

$$
l(\tilde{x}, \hat{x}, u, v) =
- \log
\frac{e^{\mathrm{sim}(\tilde{f}(u,v), \hat{f}(u,v))/\tau}}
{e^{\mathrm{sim}(\tilde{f}(u,v), \hat{f}(u,v))/\tau}
+ \sum_{(u',v') \in \Omega^-} e^{\mathrm{sim}(\tilde{f}(u,v), \hat{f}(u',v'))/\tau}}
$$

전체 local loss는 이미지와 local position 전체에 대해 평균낸다.

$$
L_l =
\frac{1}{|X|}
\sum_{x \in X}
\frac{1}{2A}
\sum_{(u,v)\in\Omega^+}
\left[
l(\tilde{x}, \hat{x}, u,v) + l(\hat{x}, \tilde{x}, u,v)
\right]
$$

이 loss의 의미는 단순하다. **같은 위치는 변환이 달라도 비슷해야 하고, 다른 위치는 구분되어야 한다**는 것이다. 이는 decoder가 지역적 구조를 잘 보존하면서도 서로 다른 anatomical region을 구별하는 데 도움이 된다.

### 3.4 Local contrasting strategy

local loss에도 두 가지 전략이 있다.

`L_R`은 같은 이미지 안에서만 local pair를 만든다. 즉, 한 이미지의 두 augmentation에서 대응 위치는 positive, 그 외 위치는 negative다.

`L_D`는 global loss에서처럼 서로 다른 volume의 같은 partition에서 같은 위치의 local region까지 positive로 본다. 즉, $f_i^s(u,v)$와 $f_j^s(u,v)$를 positive로 묶는다. 다만 논문 결과상 이 전략은 항상 유리하지 않았다. 저자들은 volume 간 거친 alignment만으로는 local region 수준의 정확한 correspondence가 보장되지 않아서, 오히려 잘못된 positive를 만들 수 있다고 해석한다.

### 3.5 전체 학습 절차

학습은 다음과 같은 stage-wise 방식이다.

1. encoder $e$와 $g_1$을 $L_g$로 pre-train한다.
2. $g_1$은 버리고, encoder는 고정한다.
3. decoder 일부 $d_l$와 $g_2$를 붙여 $L_l$로 pre-train한다.
4. $g_2$를 버리고, decoder 나머지를 랜덤 초기화해 전체 segmentation network를 fine-tune한다.

저자들은 $L_g$, $L_l$, segmentation loss를 동시에 jointly 학습하는 것도 가능하다고 언급하지만, loss weighting hyper-parameter가 복잡해질 수 있어서 stage-wise를 채택했다. 실제 ablation에서도 joint training보다 stage-wise가 더 좋았다.

### 3.6 네트워크와 학습 설정

기본 backbone은 UNet이다. Encoder는 6개의 convolutional block으로 구성되고, 각 block은 두 개의 $3 \times 3$ convolution 뒤에 stride 2의 $2 \times 2$ max-pooling이 온다. Global pre-training용 $g_1$은 output dimension이 3200, 128인 두 개의 dense layer다. Local pre-training용 $g_2$는 두 개의 $1 \times 1$ convolution으로 구성된다.

모든 단계에서 Adam optimizer를 사용했고, 10,000 iteration, batch size 40, learning rate $10^{-3}$로 학습했다. Temperature는 $\tau=0.1$을 사용했다. Local loss에서 각 feature map당 local region 수 $A$는 13이었다. Appendix에는 Titan X GPU 기준으로 global pre-training 약 2시간, local pre-training 약 4시간, fine-tuning 약 2시간이라고 적혀 있다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 설정

논문은 세 개의 MRI segmentation 데이터셋을 사용한다.

첫째는 **ACDC**로, cardiac cine-MRI 100개 volume과 세 구조(left ventricle, myocardium, right ventricle) 라벨이 있다.  
둘째는 **Prostate**로, 48개 T2-weighted MRI 중 라벨이 있는 subset을 사용하며 peripheral zone과 central gland를 분할한다.  
셋째는 **MMWHS**로, cardiac MRI 20개 volume과 7개 구조 라벨이 있다.

전처리는 percentile 기반 min-max normalization, fixed pixel spacing으로 resampling, fixed image size로 crop/pad를 적용했다. 논문 본문과 appendix를 종합하면 N4 bias correction도 사용했다.

데이터는 pre-training set $X_{pre}$와 test set $X_{ts}$로 나누고, pre-training은 $X_{pre}$의 이미지 만으로 수행한다. 라벨은 pre-training에 사용하지 않는다. 이후 $X_{pre}$ 일부에서 소수의 labeled volume을 뽑아 fine-tuning training set $X_{tr}$로 쓰고, validation set $X_{vl}$은 항상 2 volume으로 고정했다. 실험은 $|X_{tr}| = 1, 2, 8$ volume 설정에서 수행되었다. 최종 평가는 test set에서 mean Dice similarity coefficient(DSC)를 사용했고, 각 결과는 6회 반복 평균이다.

### 4.2 Global contrasting strategy 결과

Table 1 상단 결과를 보면, 아무 pre-training 없이 랜덤 초기화한 모델보다 기본 global contrastive strategy `G_R`도 일정 수준 성능 향상을 준다. 예를 들어 ACDC에서 $|X_{tr}|=1$일 때 random init은 0.614, `G_R`은 0.631이다. 하지만 제안 전략인 `G_D^-`, `G_D`는 이보다 더 좋다. 같은 설정에서 `G_D^-`는 0.683, `G_D`는 0.691이다.

이 경향은 세 데이터셋 전반에서 반복된다. 즉, 의료영상 volume 사이의 구조적 유사성을 positive/negative 정의에 반영하는 것이 단순 augmentation 기반 contrastive learning보다 낫다. `G_D`가 9개 설정 중 6개에서 `G_D^-`보다 약간 더 좋았기 때문에, 이후 실험에서는 global strategy로 `G_D`를 선택한다.

이 결과는 중요한 의미를 가진다. 단순히 “같은 이미지의 augmentation 쌍”만 positive로 쓰는 것보다, **같은 anatomical partition의 실제 다른 이미지들을 positive로 사용하는 것이 더 풍부한 similarity signal을 제공한다**는 점을 보여준다.

### 4.3 Local contrastive loss 결과

Table 1 하단은 encoder를 `G_R`로 먼저 pre-train한 뒤, decoder를 local loss로 추가 pre-train한 결과다. 여기서 random decoder initialization보다 `L_R`, `L_D` 모두 대체로 추가적인 향상을 보인다. 예를 들어 ACDC에서 $|X_{tr}|=2$일 때 `G_R + random`은 0.729, `G_R + L_R`은 0.760이다.

흥미롭게도 local 전략에서는 `L_D`보다 `L_R`이 더 좋은 경우가 많다. 논문은 9개 설정 중 6개에서 `L_R`이 더 우세하다고 보고한다. 저자 해석은 합리적이다. volume 간 rough alignment는 global partition 수준에서는 어느 정도 유효하지만, **pixel/local region 수준의 정밀한 correspondence까지는 보장하지 않는다**. 따라서 서로 다른 volume의 같은 좌표를 local positive로 강제하면 오히려 잘못된 supervision이 들어갈 수 있다.

이 실험은 local contrastive loss 자체의 유효성을 보여주면서도, 도메인 지식은 scale에 따라 신중히 적용해야 한다는 점도 보여준다.

### 4.4 전체 방법과 기존 방법 비교

Table 2는 핵심 비교 결과다. 논문이 제안한 최종 초기화 방식은 `G_D + L_R`이다. 이를 random init, 기존 global contrastive pre-training, pretext task, semi-supervised learning, augmentation 기반 방법들과 비교한다.

먼저 pre-training 계열 비교를 보면 제안 방법이 일관되게 가장 좋다. 예를 들어 ACDC에서:

- random init: 0.614 / 0.702 / 0.844
- global loss `G_R`: 0.631 / 0.729 / 0.847
- proposed init `G_D + L_R`: 0.725 / 0.789 / 0.872

즉, 특히 labeled volume이 1개 또는 2개처럼 극도로 적을 때 개선폭이 크다. Prostate에서도 $|X_{tr}|=1$ 기준 random init 0.489에서 proposed init 0.579로 상승한다. MMWHS에서는 0.451에서 0.569로 오른다.

pretext task와의 비교도 중요하다. Rotation, Inpainting, Context Restoration 모두 어느 정도 도움은 되지만, proposed initialization보다 낮다. 이는 segmentation처럼 dense prediction task에서는 단순 pretext task보다, **task-aligned contrastive objective와 local representation 학습이 더 효과적**임을 시사한다.

semi-supervised 또는 augmentation 기반 기존 방법과 비교해도 제안 방법은 강하다. 예를 들어 ACDC에서 $|X_{tr}|=1$일 때 self-training 0.690, Mixup 0.695, data augmentation [9] 0.731인데 proposed init은 0.725로 매우 경쟁력 있다. Prostate와 MMWHS에서는 proposed init이 대체로 더 강하거나 비슷한 수준이다.

논문이 특히 강조하는 부분은 **조합 가능성**이다. 제안 pre-training을 다른 방법과 결합하면 성능이 더 오른다. 예를 들어 ACDC에서:

- proposed init + self-train: 0.745 / 0.802 / 0.881
- proposed init + Mixup: 0.757 / 0.826 / 0.886

이는 제안한 initialization이 다른 limited-annotation 기법과 **orthogonal**, 즉 상보적이라는 뜻이다.

### 4.5 Benchmark와의 격차

완전 supervised benchmark는 모든 가용 training volume을 사용한 결과다. ACDC는 0.912, Prostate는 0.697, MMWHS는 0.787이다. 제안 방법은 라벨 2개 volume만 써도 benchmark에 상당히 근접한다. 논문은 “2개 volume만으로 benchmark 대비 약 0.1 DSC 이내”라고 정리한다.

특히 abstract에서는 simple augmentation과 결합했을 때 **benchmark 성능의 8% 이내**까지 도달한다고 강조한다. ACDC에서는 라벨 2개 volume이 전체 훈련 데이터의 약 4%에 해당한다고 명시한다. 즉, 극도로 적은 라벨만으로도 상당한 수준의 성능을 확보할 수 있음을 보인다.

### 4.6 Ablation study

Ablation도 의미가 있다.

Global pre-training에서 batch size는 40, 250, 450을 비교했는데, ACDC 기준 40이 가장 좋았다. 자연영상 contrastive learning에서는 큰 batch가 중요하다는 주장이 많지만, 이 논문은 의료영상에서는 꼭 그렇지 않을 수 있다고 본다. 데이터셋 크기 자체가 훨씬 작고, unlabeled slice 수도 제한적이기 때문이다.

Partition 수 $S$는 3, 4, 6을 비교했고, $S=4$가 가장 좋았다. 너무 큰 $S$는 volume 간 slice correspondence를 지나치게 세밀하게 가정하게 만들어 잘못된 positive/negative 구성을 초래한다.

Local pre-training에서는 decoder block 수 $l$과 local region size $K \times K$를 비교했다. 본문 Table 1에서는 최종적으로 $l=3$을 사용했고, appendix에서는 $3 \times 3$ local region이 대체로 $1 \times 1$보다 좋다고 보고한다. 저자들은 더 큰 local patch가 더 큰 receptive field를 가져 유용한 정보를 담기 때문이라고 해석한다.

또한 joint training 실험에서는

$$
L_{net} = L_g + \lambda_l L_l
$$

형태로 동시에 학습했지만, stage-wise training보다 성능이 낮았다. 따라서 이 논문에서는 encoder와 decoder의 contrastive pre-training을 나누어 수행하는 것이 더 효과적이라는 근거를 제시한다.

### 4.7 Natural image dataset 실험

논문은 방법의 일반성을 보기 위해 Cityscapes에도 local contrastive loss를 적용했다. 여기서는 global은 `G_R`, local은 `L_R`만 사용했다. 결과는 baseline과 global-only보다 proposed init이 낫고, Mixup을 추가하면 더 좋아진다. 예를 들어 $(X_{tr}, X_{val})=(400,200)$일 때 baseline 0.524, global-only 0.535, proposed init 0.549, proposed init + Mixup 0.569다.

다만 이 실험은 이미지 해상도를 크게 줄였고 label도 일부 합쳤기 때문에, medical datasets만큼 직접적인 해석은 어렵다. 그럼에도 local contrastive idea가 medical imaging 바깥에도 어느 정도 통할 수 있음을 보여주는 참고 결과로 볼 수 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **segmentation이라는 downstream task의 구조를 반영해 contrastive learning을 재설계했다는 점**이다. 단순히 encoder representation을 좋게 만드는 것이 아니라, decoder가 필요로 하는 local discriminative feature까지 self-supervised하게 학습하도록 설계했다. 이는 문제 설정에 매우 잘 맞는 아이디어다.

또 다른 강점은 **의료영상 도메인 지식을 pair construction에 직접 반영했다는 점**이다. volume 간 slice correspondence를 positive/negative 정의에 활용한 것은 단순 augmentation 기반 contrastive learning보다 더 풍부한 supervision signal을 제공한다. 실험 결과도 이 설계를 지지한다.

실험 설계도 강한 편이다. 세 개의 MRI 데이터셋에서 평가했고, random initialization, global contrastive baseline, 여러 pretext task, semi-supervised learning, augmentation 기반 방법까지 폭넓게 비교했다. 또 제안 방법이 다른 기법과 결합 가능하다는 점도 보여주었다. few-label regime, 특히 $|X_{tr}|=1,2$에서 성능 개선이 크다는 점은 실제 의료환경과 잘 맞는다.

반면 한계도 분명하다. 첫째, 제안한 global strategy는 **volume 간 대략적인 정렬과 anatomical consistency**를 전제한다. 논문에서는 사용한 데이터셋들이 이미 대략 정렬되어 있어서 별도 registration이 필요 없었다고 한다. 하지만 더 다양한 acquisition setting이나 정렬이 불안정한 데이터에서는 이 가정이 약해질 수 있다.

둘째, local domain-specific strategy `L_D`는 기대만큼 강하지 않았다. 이것은 local region correspondence가 global partition correspondence보다 훨씬 어렵다는 뜻이며, 도메인 지식을 더 미세한 수준에 적용할 때는 정확한 정합이 필요하다는 점을 보여준다.

셋째, 논문은 segmentation 성능 향상을 명확히 보여주지만, **representation 자체가 어떤 anatomical semantics를 학습했는지에 대한 분석**은 제한적이다. 예를 들어 representation visualization이나 failure case 분석은 제공되지 않는다. 또한 uncertainty, domain shift robustness, cross-site generalization 같은 임상 적용 핵심 요소는 직접 다루지 않는다. broader impact 섹션에서 이런 필요성을 언급하지만, 실제 실험 검증은 포함되어 있지 않다.

넷째, 방법이 UNet 기반 2D slice 처리 중심으로 설명되어 있고, 3D volumetric model이나 transformer류 구조로의 확장은 이 논문 안에서 다루지 않는다. 따라서 후속 연구에서는 3D encoder-decoder나 stronger alignment-aware sampling으로 발전시킬 여지가 있다.

비판적으로 해석하면, 이 논문은 contrastive learning을 segmentation에 맞게 바꾸는 방향을 잘 제시했지만, 성능 향상의 일부는 도메인별 정렬 특성에 의존한다. 즉, 제안 방법의 일반성은 완전히 보장되기보다 **“구조적으로 정렬 가능한 volumetric medical image”라는 조건에서 특히 강하다**고 보는 편이 정확하다.

## 6. 결론

이 논문은 limited annotation 환경의 medical image segmentation을 위해 contrastive learning을 실질적으로 확장한 연구다. 핵심 기여는 두 가지다. 하나는 **volumetric medical image의 구조를 이용한 domain-specific global contrasting strategy**이고, 다른 하나는 **segmentation에 필요한 local feature를 학습하는 local contrastive loss**다. 이 둘을 stage-wise로 결합한 pre-training은 세 개의 MRI 데이터셋에서 일관되게 strong baseline들과 경쟁하거나 그보다 나은 결과를 보였다.

특히 라벨이 1개 또는 2개 volume에 불과한 극저라벨 상황에서 성능 향상이 크고, Mixup이나 self-training 같은 기존 기법과 결합했을 때 추가 이득도 얻는다. 따라서 이 연구의 실질적 의미는, 의료영상에서 흔한 “unlabeled data는 많고 labeled data는 적은” 상황에서 **적은 annotation으로도 강한 segmentation 모델을 만드는 간단하고 효과적인 초기화 전략**을 제시했다는 점에 있다.

향후 연구 측면에서는 더 정교한 cross-volume alignment, 3D 모델로의 확장, domain shift와 uncertainty를 고려한 학습, 그리고 다양한 dense prediction task로의 일반화가 중요한 후속 방향이 될 가능성이 크다. 이 논문은 그 출발점으로서, contrastive SSL을 segmentation 중심 관점에서 재해석했다는 점에서 의미 있는 기여를 한다.
