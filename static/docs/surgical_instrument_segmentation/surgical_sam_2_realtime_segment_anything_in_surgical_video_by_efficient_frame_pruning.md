# Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning

* **저자**: Haofeng Liu, Erli Zhang, Junde Wu, Mingxuan Hong, Yueming Jin
* **발표연도**: 2024
* **arXiv**: <https://arxiv.org/abs/2408.07931v2>

## 1. 논문 개요

이 논문은 surgical video segmentation, 특히 **수술 영상에서 기구(instrument)를 실시간으로 정확하게 분할하는 문제**를 다룬다. 저자들은 기존의 SAM2가 image/video segmentation에서 매우 강력한 성능을 보이지만, 수술 영상에 그대로 적용하기에는 **계산량과 메모리 사용량이 너무 크다**는 점에 주목한다. 수술 영상은 보통 고해상도이고 길이가 길며, 프레임 간 시각적 중복이 많기 때문에, 모든 과거 프레임을 비슷한 방식으로 저장하고 활용하는 기존 SAM2의 memory bank 방식은 비효율적이라는 것이 핵심 문제 설정이다.

이 문제는 실제 임상 환경에서 매우 중요하다. 수술 보조 시스템은 단순히 정확하기만 해서는 부족하고, **낮은 지연과 높은 처리 속도**가 반드시 필요하다. 그래야 수술 중 instrument tracking, pose estimation, intraoperative guidance, 이상 상황 경고, 원격 surgical proctoring (경험 많은 전문의의 수술 지도 및 감독) 같은 응용이 가능해진다. 따라서 이 논문은 “정확도를 유지하거나 높이면서도 real-time 처리에 가까운 속도를 달성할 수 있는가”라는 현실적인 질문에 답하고자 한다.

이를 위해 저자들은 **SurgSAM-2**라는 모델을 제안한다. 이는 SAM2를 기반으로 하되, **Efficient Frame Pruning (EFP)** 이라는 동적 memory bank 관리 기법을 추가하여, 정보량이 적거나 지나치게 유사한 프레임을 메모리에서 제거하는 방식으로 계산 효율을 높인다. 논문의 주장은 단순하다. 수술 영상에서는 카메라가 고정되거나 천천히 움직이는 경우가 많고 배경 조직이 여러 프레임에서 매우 유사하므로, 비슷한 프레임을 계속 저장하는 것은 낭비이며, 오히려 중요한 프레임에 대한 attention을 약화시킬 수 있다는 것이다.

실험은 EndoVis17과 EndoVis18 데이터셋에서 수행되었고, 결과적으로 제안법은 vanilla SAM2보다 **더 빠르고 메모리 효율적이며**, fine-tuning까지 수행하면 **정확도 측면에서도 더 우수한 성능**을 보여준다고 보고한다.

![Figure 1:Architecture of the proposed model SurgSAM-2.](https://ar5iv.labs.arxiv.org/html/2408.07931/assets/x1.png)

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 매우 명확하다. **수술 비디오에서는 모든 과거 프레임이 equally useful하지 않다**는 점을 적극적으로 이용하는 것이다. 기존 SAM2는 과거 프레임을 순차적으로 memory bank에 유지하는 경향이 있는데, 이는 first-come-first-serve에 가까운 정책이다. 하지만 수술 영상에서는 연속된 프레임이 서로 매우 비슷한 경우가 많아서, 이런 저장 방식은 중복된 정보로 메모리를 채우게 된다.

SurgSAM-2는 이 문제를 해결하기 위해, 새로 들어오는 프레임 $f_t$와 최근 과거 프레임들 사이의 **cosine similarity**를 계산하고, 그중 **가장 비슷한 프레임들**을 제거한다. 즉, 기억해야 할 프레임을 “먼저 들어온 순서”가 아니라 “얼마나 정보적으로 중복되는가”를 기준으로 선택한다. 이때 저자들은 중복도가 높은 프레임을 없애면, 메모리 사용량과 cross-attention 비용이 줄어들 뿐 아니라, 오히려 더 중요한 프레임들에 attention이 집중되어 segmentation 품질에도 도움이 될 수 있다고 본다.

기존 접근 방식과의 차별점은 크게 두 가지다. 첫째, SAM2의 backbone과 전체적인 video segmentation 틀은 유지하면서도, **memory bank 관리 정책만 바꾸어 실용적인 속도 향상**을 노린다는 점이다. 즉, 완전히 새로운 segmentation model을 만드는 대신, 강력한 foundation model 위에 surgical domain에 맞는 효율화 계층을 얹은 구조다. 둘째, 단순히 속도만 높이는 것이 아니라, **surgical video의 데이터 특성 자체를 이용해 pruning 기준을 설계**했다는 점이다. 논문은 이 중복성의 원인을 camera motion이 작고 조직 구조가 프레임 간 비슷하게 유지되는 수술 장면의 특성에서 찾는다.

결국 이 논문의 핵심 직관은 다음과 같이 요약할 수 있다. 수술 비디오 segmentation에서는 “더 많은 메모리”가 항상 더 좋은 것이 아니라, **적절히 제한되고 정제된 메모리**가 오히려 더 효율적이고 때로는 더 정확할 수 있다.

## 3. 상세 방법 설명

### 3.1 전체 구조

SurgSAM-2는 기본적으로 SAM2를 기반으로 한다. 즉, 전체 파이프라인은 크게 바뀌지 않는다. 입력 비디오 프레임이 들어오면, image encoder가 각 프레임을 embedding으로 바꾸고, memory bank에 저장된 과거 프레임 정보와 현재 프레임 정보를 결합하여 segmentation을 수행한다. 논문에서 바뀌는 핵심은 **어떤 과거 프레임을 memory bank에 남길 것인가**이다.

저자 설명에 따르면 memory bank는 현재 프레임과 선택된 과거 프레임들로 구성된다. 이 memory bank는 이후 memory cross-attention에 사용된다. 따라서 bank 안의 프레임 수가 많으면 temporal context를 더 많이 담을 수 있지만, 동시에 계산 비용과 메모리 사용량이 커진다. SurgSAM-2는 이 부분에서 **dynamic memory bank management**를 도입한다.

### 3.2 Efficient Frame Pruning

논문의 핵심 방법은 Efficient Frame Pruning, 즉 EFP이다. 각 시점의 새 프레임 $f_t$가 memory bank에 들어가려고 할 때, 모델은 최근 과거 $n$개의 프레임 ${f_{t-n}, \dots, f_{t-1}}$과의 cosine similarity를 계산한다.

논문에 제시된 식은 다음과 같다.

$$
S(f_t, f_i) = \frac{f_t \cdot f_i}{|f_t| , |f_i|}
$$

여기서 $S(f_t, f_i)$는 현재 프레임 $f_t$와 과거 프레임 $f_i$ 사이의 cosine similarity이다. 값이 클수록 두 프레임이 더 비슷하다고 해석할 수 있다.

이후 절차는 다음과 같다.

1. 현재 프레임 $f_t$와 최근 $n$개 프레임 사이의 similarity를 모두 계산한다.
2. 그중에서 **가장 유사한 $m$개의 프레임**을 찾는다.
3. 이 $m$개의 프레임은 pruning한다.
4. 남은 $(n-m)$개 프레임과 현재 프레임 $f_t$를 memory bank에 유지한다.
5. 단, 첫 번째 프레임 $f_0$는 항상 유지된다.

이 구조에서 첫 프레임 $f_0$는 매우 중요한 reference frame으로 취급된다. 저자들은 이 프레임이 초기 상태를 안정적으로 제공한다고 보고, dynamic memory size 계산에서 제외하면서도 항상 bank에 유지한다.

논문에서 제시한 설정은 다음과 같다. vanilla SAM2는 과거 6개 프레임과 첫 reference frame을 함께 사용하는데, SurgSAM-2에서는 이를 바탕으로 **$n=5$, $m=2$**로 설정한다. 즉, 최근 5개 프레임 중 현재 프레임과 가장 비슷한 2개를 제거하고, 남은 3개와 현재 프레임을 함께 유지하는 식이다. 여기에 항상 남는 첫 프레임 $f_0$까지 고려하면, 결국 보다 작지만 유용한 정보를 포함한 memory bank를 구성하게 된다.

이 아이디어는 단순하지만 매우 실용적이다. 수술 영상에서는 연속 프레임이 거의 같은 시야를 보여주는 경우가 많기 때문에, similarity가 높은 프레임을 줄이는 것은 자연스럽다. 논문은 이 pruning이 단순히 계산량을 줄이는 효과뿐 아니라, **redundant information이 attention을 흐리는 현상도 줄일 수 있다**고 설명한다.

### 3.3 왜 pruning이 정확도에도 도움이 될 수 있는가

논문은 메모리를 줄이면 무조건 정보 손실이 발생한다고 보지 않는다. 오히려 큰 memory bank는 irrelevant하거나 redundant한 프레임까지 다 보관하면서, 진짜 중요한 object-related frame에 대한 attention score를 희석시킬 수 있다고 주장한다. 즉, 정보량이 많다고 항상 좋은 것이 아니라, **중요한 정보에 대한 signal-to-noise ratio**가 낮아질 수 있다는 설명이다.

이 관점은 video object segmentation에서 꽤 설득력이 있다. 메모리 기반 모델은 과거 프레임과 현재 프레임 사이의 관계를 attention이나 matching으로 풀어내는데, 비슷한 프레임이 지나치게 많으면 중요한 temporal cue가 희석될 수 있다. 따라서 적절한 pruning은 attention이 더 discriminative하게 작동하도록 돕는다고 해석할 수 있다.

다만 논문도 이 점을 균형 있게 인정한다. 메모리 bank가 너무 작아지면 유용한 temporal information까지 잃을 수 있기 때문에, **충분히 작되 너무 작지는 않은 절충점**이 중요하다고 말한다. 이 부분은 논문의 향후 연구 과제이기도 하다.

### 3.4 학습 절차와 구현 세부사항

실험은 RTX A6000 48GB GPU에서 수행되었고 backbone은 **ViT-Small**을 사용했다. precision은 **bfloat16**이다. 원래 vanilla SAM2는 $1024 \times 1024$ 해상도와 ViT-Base+를 사용하는데, 저자들은 수술 환경에서 이 설정은 학습 시간이 너무 길고 안정적 학습도 어렵다고 판단했다. 그래서 fine-tuning 시에는 **$512 \times 512$ 해상도**와 SAM2의 ViT-Small weights를 사용했다.

학습 전략은 SAM2를 따라 **video training과 image training을 번갈아 수행**하는 방식이다. 또한 multi-mask output, IoU prediction, occlusion prediction도 함께 학습했다고 설명한다. 중요한 점은 모든 모듈을 다 학습하지 않았다는 것이다. generalization을 유지하기 위해 **prompt encoder와 image encoder는 frozen**하고, **mask decoder와 memory module만 fine-tuning**했다.

하이퍼파라미터는 다음과 같다.

* video training 배치 크기: 12
* 각 배치의 frame 수: 8
* 프레임당 최대 object 수: 3
* image training 배치 크기: 32
* image당 최대 object 수: 3
* mask decoder learning rate: $2 \times 10^{-4}$
* memory encoder learning rate: $2 \times 10^{-5}$

video augmentation은 Cutie의 전략을 따랐다고만 되어 있으며, 구체적인 augmentation 구성은 이 텍스트에는 자세히 나오지 않는다. 따라서 어떤 변형을 얼마나 사용했는지는 여기서 정확히 알 수 없다.

추론 시에는 segmentation 결과를 원래 해상도로 다시 resize하여 공정하게 평가한다. 또 한 가지 흥미로운 구현 선택은, **첫 프레임만 원래 해상도 1024로 처리하고 나머지 프레임은 512 해상도로 처리**했다는 점이다. 이는 첫 프레임의 고품질 segmentation이 이후 추적과 분할의 기준이 되기 때문에 중요하다고 본 것이다. 즉, 전체 프레임을 고해상도로 돌리는 대신, 가장 중요한 첫 프레임에만 고해상도 비용을 쓰는 식의 practical trade-off다.

### 3.5 수식과 목표의 의미

논문은 복잡한 새로운 loss function을 제안하지는 않는다. 이 방법의 핵심은 loss 설계보다 **memory selection policy**에 있다. 본문에서 명시된 수식은 사실상 cosine similarity 계산식이 핵심이며, 이것이 pruning decision의 기준이 된다. 따라서 이 논문에서 수학적으로 가장 중요한 것은 “어떤 프레임을 얼마나 유사하다고 볼 것인가”와 “그 유사도를 기준으로 어떤 프레임을 제거할 것인가”이다.

즉, 방법론을 쉬운 말로 정리하면 다음과 같다.

현재 프레임을 잘 분할하기 위해 과거 프레임을 참고해야 하지만, 과거 프레임이 너무 많고 서로 너무 비슷하면 오히려 비효율적이다. 그래서 현재 프레임과 너무 비슷한 과거 프레임은 빼고, 상대적으로 덜 중복되는 프레임들만 남겨서 memory cross-attention에 넣는다. 이로써 속도와 메모리 효율을 높이고, 경우에 따라 segmentation 품질도 좋아질 수 있다.

## 4. 실험 및 결과

### 4.1 데이터셋과 평가 설정

실험은 두 개의 대표적인 surgical video benchmark에서 수행되었다.

첫 번째는 **EndoVis17**이고, 두 번째는 **EndoVis18**이다. 논문에 따르면 EndoVis18이 더 복잡한 장면을 포함하므로 더 어려운 데이터셋이다. EndoVis17은 8개의 training video와 8개의 test video, 그리고 2개의 hold-out test sequence(9, 10)로 구성되며, 논문은 hold-out test set을 평가에 사용했다. EndoVis18은 총 15개의 비디오로 구성되며, sequence 2, 5, 9, 15를 test로 사용하는 ISINet의 standard split을 따른다.

데이터 전처리는 Shvets et al.의 방식을 따랐다. 흥미로운 점은 EndoVis17/18의 기존 제공 라벨이 instrument-type label 중심이라 instance-level 구분이 충분하지 않아서, 저자들이 **re-annotation**을 수행했다고 밝힌다는 점이다. 이는 같은 종류의 기구라도 서로 다른 instance를 구분해야 하는 더 어려운 setting을 평가하기 위함이다. 이 점은 단순 semantic segmentation보다 문제 난도가 더 높다는 뜻이기도 하다.

### 4.2 평가 지표

평가에는 segmentation accuracy와 efficiency를 함께 보기 위해 여러 지표를 사용했다.

IoU는 다음과 같이 정의된다.

$$
J = \frac{\text{Area of Overlap}}{\text{Area of Union}}
$$

Boundary F1 score는 boundary precision과 recall의 조화 평균이다.

$$
F = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

이 둘을 평균낸 composite metric이 J&F이다.

$$
J\&F = \frac{J + F}{2}
$$

Dice coefficient는 다음과 같다.

$$
\text{Dice} = \frac{2 \times |\text{Prediction} \cap \text{Ground Truth}|}{|\text{Prediction}| + |\text{Ground Truth}|}
$$

또한 EndoVis18 Challenge의 공식 지표인 **CIoU (Challenge IoU)** 도 사용했다. 이 지표는 frame별로 존재하는 object들만 고려해 IoU를 계산하고 이를 평균낸다. real-time 성능을 위해 **FPS**를 측정했고, resource efficiency를 위해 **memory usage**도 함께 비교했다.

저자들은 VOS benchmark protocol을 따라 첫 프레임과 마지막 프레임은 제외하고 평가했다고 설명한다. 또한 EndoVis Challenge의 공식 평가 프로토콜도 함께 활용했다.

### 4.3 효율성 결과

논문의 가장 눈에 띄는 결과는 효율성 향상이다. 테이블 1, 2, 3을 보면 EndoVis17과 EndoVis18 모두에서, prompt setting이 Full Mask, One Point, Five Points인 경우 전반적으로 SurgSAM-2가 vanilla SAM2보다 훨씬 높은 FPS와 더 낮은 memory usage를 보인다.

예를 들어 EndoVis17 Full Mask setting에서:

* SAM2: FPS 29.10, Memory 3.10 GB
* SurgSAM-2 + EFP, no fine-tuning: FPS 33.00, Memory 2.83 GB
* SurgSAM-2 + EFP, fine-tuning: FPS 86.03, Memory 1.08 GB

EndoVis18 Full Mask에서도:

* SAM2: FPS 29.18, Memory 3.14 GB
* SurgSAM-2 + EFP, no fine-tuning: FPS 33.08, Memory 2.82 GB
* SurgSAM-2 + EFP, fine-tuning: FPS 86.11, Memory 1.02 GB

즉, EFP만 넣은 경우에도 소폭 속도 향상과 메모리 절감이 있고, fine-tuning까지 포함한 설정에서는 **약 86 FPS 수준**으로 크게 향상된다. 논문 abstract의 “3× FPS”라는 표현은 이 수치와 잘 맞는다. 기존 29 FPS 수준에서 86 FPS 수준으로 올라가므로 대략 3배에 가깝다.

다만 본문에서 “On average, SurgSAM-2 demonstrated a 13.8% increase in FPS and an 8.5% reduction in memory usage”라고 쓴 부분은, 테이블의 fine-tuning 결과와 비교하면 체감상 훨씬 작은 수치처럼 보인다. 이는 아마 EFP 자체 효과만 평균했거나 특정 비교 기준을 사용했기 때문일 수 있다. 하지만 제공된 텍스트만 보면 이 평균 계산 방식이 정확히 어떻게 정의되었는지는 분명하지 않다. 따라서 이 수치를 해석할 때는 주의가 필요하다.

### 4.4 정확도 결과

정확도 측면에서는 두 가지 층위로 봐야 한다. 하나는 **EFP만 추가했을 때**이고, 다른 하나는 **EFP와 fine-tuning을 함께 적용했을 때**이다.

EFP만 추가한 경우, EndoVis17에서는 약간의 성능 저하가 있었다. 논문 본문도 이를 솔직하게 인정한다. 예를 들어 EndoVis17 Full Mask에서 J&F는 87.5에서 87.2로, Dice는 90.2에서 89.9로 소폭 감소한다. One Point에서도 82.5에서 81.6으로 감소한다. 따라서 “pruning만 하면 무조건 정확도가 오른다”는 결론은 아니다.

반면 EndoVis18에서는 EFP가 오히려 도움이 되는 경우가 많다. Full Mask 기준으로 J&F가 78.5에서 81.9로, Dice가 81.7에서 85.2로 개선된다. 저자들은 더 어려운 데이터셋에서는 redundant frame 제거가 noise 억제와 attention 집중에 더 큰 도움을 줄 수 있다고 해석한다.

fine-tuning을 수행하면 결과는 훨씬 강해진다. 예를 들어 EndoVis17 One Point setting에서:

* SAM2: Dice 85.1, FPS 29.09
* SurgSAM-2 + fine-tuning: Dice 87.3, FPS 85.95

즉, 아주 약한 user interaction인 “첫 프레임 한 점 클릭”만으로도 정확도와 속도를 동시에 크게 높였다. 이것은 실제 임상 활용 관점에서 매우 중요한 결과다. 수술자가 많은 prompt를 주기 어렵기 때문이다.

Five Points setting에서도 fine-tuning 효과는 뚜렷하다. EndoVis17에서는 J&F가 83.9에서 88.0으로, Dice는 86.9에서 91.4로 향상된다. EndoVis18에서도 J&F가 76.3에서 80.8로, Dice는 80.0에서 84.9로 오른다. 따라서 이 논문의 강한 실험적 메시지는 “단순 pruning만이 아니라, surgical domain에 맞는 fine-tuning까지 포함한 전체 설계가 매우 효과적이다”라는 점이다.

### 4.5 기존 방법과의 비교

EndoVis18에 대해서는 Challenge IoU를 기준으로 기존 task-specific model들과 SAM-based model들을 비교했다. 비교 결과는 다음과 같은 흐름을 보여준다.

* TernausNet: 46.2
* ISINet: 73.0
* S3Net: 76.2
* MATIS Frame: 82.4
* SurgicalSAM: 80.3
* SAM2 (1 Point / 5 Points / Full): 63.6 / 78.8 / 82.2
* SurgSAM-2 (1 Point / 5 Points / Full): 72.6 / 82.1 / 84.4

이 결과는 두 가지를 의미한다. 첫째, vanilla SAM2보다 SurgSAM-2가 prompting 수준에 상관없이 일관되게 좋다. 둘째, Full prompt에서는 task-specific strong baseline인 MATIS Frame의 82.4보다 높은 84.4를 기록한다. 물론 저자들도 이 비교가 완전히 공정하지는 않다고 명시한다. 많은 기존 방법은 prompt가 필요 없고, 일부는 type segmentation 중심이며, 제안법은 더 어려운 instance-level setting을 지향하기 때문이다. 따라서 숫자 비교만으로 절대적 우열을 단정하기보다는, “더 practical하고 더 어려운 setting에서도 경쟁력 있다”는 정도로 해석하는 것이 적절하다.

### 4.6 정성적 결과

정성적 결과에서는 vanilla SAM2가 target object를 놓치거나 잘못된 object를 분할하는 사례가 제시된다고 설명한다. 반면 SurgSAM-2는 Full Mask, One Point, Five Points 모두에서 보다 안정적으로 target instrument를 segmentation한다고 주장한다. 다만 본문에는 그림 자체가 포함되어 있지 않으므로, 어떤 장면에서 어떤 실패가 있었는지 구체적인 시각적 근거까지는 여기서 확인할 수 없다. 따라서 정성적 분석의 자세한 내용은 논문 그림을 직접 봐야 완전히 평가할 수 있다.

![Figure 2:Visual comparison between SAM2 and the proposed model SurgSAM-2 on EndoVis18 dataset.](https://ar5iv.labs.arxiv.org/html/2408.07931/assets/x2.png)

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 설정이 매우 현실적**이라는 점이다. 많은 segmentation 논문이 정확도 개선 자체에 초점을 두는 반면, 이 논문은 실제 수술실에서 필요한 real-time 조건과 resource constraint를 정면으로 다룬다. 특히 SAM2처럼 강력하지만 무거운 foundation model을 surgical deployment 관점에서 재설계했다는 점이 실용적이다.

또 다른 강점은 방법이 **단순하면서도 설득력 있다**는 것이다. cosine similarity 기반 pruning은 구현이 복잡하지 않고 직관적이며, surgical video의 frame redundancy라는 데이터 특성과 잘 맞는다. 새로운 대규모 네트워크를 설계하지 않고, memory bank policy를 바꿈으로써 큰 속도 개선을 얻었다는 점은 engineering value가 높다.

실험도 비교적 충실하다. EndoVis17과 EndoVis18 두 데이터셋을 사용했고, prompt setting도 Full Mask, One Point, Five Points로 나누어 평가했다. accuracy뿐 아니라 FPS와 memory까지 함께 보고한 점도 좋다. 특히 one-point setting에서의 strong result는 실제 사용성을 크게 높여주는 포인트다.

하지만 한계도 분명하다. 첫째, **개선의 원인이 EFP와 fine-tuning 중 무엇인지 완전히 분리되지는 않는다**. 표를 보면 EFP만 넣은 경우의 정확도 개선은 dataset에 따라 mixed하다. 반면 fine-tuning을 함께 하면 성능이 크게 오른다. 따라서 최종 성능 향상이 pruning 자체 덕분인지, 아니면 low-resolution fine-tuning 및 domain adaptation의 효과가 더 큰지 정교하게 분해해서 보기는 어렵다. 논문은 EFP의 기여를 강조하지만, 실제 표를 보면 fine-tuning의 영향도 매우 크다.

둘째, **cosine similarity를 어떤 feature representation 위에서 계산하는지 명확하지 않다**. 식은 제시되어 있지만, 이 $f_t$와 $f_i$가 raw frame인지, 특정 embedding인지, 어떤 stage의 feature인지 텍스트만으로는 분명하지 않다. 실용 구현에서는 매우 중요한 정보인데, 제공된 텍스트에는 상세하지 않다.

셋째, memory bank 설정도 아직 heuristic에 가깝다. 논문은 $n=5$, $m=2$를 사용했지만, 이것이 왜 최적의 조합인지 체계적 ablation이 텍스트 안에는 충분히 제시되지 않는다. 저자들도 future work에서 memory bank size와 pruning strategy를 더 연구하겠다고 밝힌다. 즉, 현재 방법은 유효하지만 완전히 일반화된 최적 정책이라고 보기는 어렵다.

넷째, qualitative result와 architecture figure가 텍스트에 포함되어 있지 않아, 실제 failure mode나 module interaction을 깊게 파악하기 어렵다. 예를 들어 어떤 상황에서 pruning이 해로웠는지, similarity가 높은데도 중요한 프레임이 제거되는 failure case는 없는지 같은 부분은 이 텍스트만으로는 알 수 없다.

다섯째, 공정 비교의 어려움도 존재한다. 저자들 스스로 인정하듯, 기존 task-specific model들과 prompt-based foundation model을 같은 축에서 비교하는 것은 완전히 apples-to-apples가 아니다. 특히 instance-level과 type-level task 차이도 비교를 복잡하게 만든다.

비판적으로 해석하면, 이 논문은 매우 강한 “새 이론”을 제시하는 논문이라기보다는, **SAM2를 surgical video에 실제로 쓰기 위해 필요한 핵심 병목을 잘 짚고, 효과적인 시스템 최적화를 제안한 논문**에 가깝다. 학술적으로는 pruning criterion이 상대적으로 단순하지만, 문제의 중요성과 실험 결과의 실용성이 이를 충분히 뒷받침한다.

## 6. 결론

이 논문은 SAM2를 수술 영상 분할에 더 적합하게 만들기 위해 **Efficient Frame Pruning 기반의 SurgSAM-2**를 제안했다. 핵심 기여는 수술 영상의 중복 프레임 특성을 이용하여 memory bank를 동적으로 정리하고, 이를 통해 **속도, 메모리 효율, 그리고 경우에 따라 정확도까지 개선**했다는 점이다.

구체적으로는 현재 프레임과 과거 프레임 사이 cosine similarity를 계산하여 가장 유사한 프레임을 제거하는 방식을 사용했고, 이를 통해 SAM2의 비효율적인 sequential memory retention 문제를 완화했다. EndoVis17과 EndoVis18에서의 실험은 SurgSAM-2가 vanilla SAM2보다 훨씬 높은 FPS와 낮은 memory usage를 달성하며, fine-tuning 시 정확도에서도 강한 성능을 보인다는 점을 보여준다.

실제 적용 측면에서 이 연구의 의미는 크다. 수술 환경에서는 정확한 segmentation도 중요하지만, 그 결과가 충분히 빨라야 실제 의사결정에 도움이 된다. 이 논문은 foundation model을 의료 도메인에 그대로 가져오는 것이 아니라, **도메인 특성에 맞춰 효율성을 재설계해야 한다**는 점을 잘 보여준다. 앞으로 더 정교한 memory policy, adaptive bank size 조절, 더 다양한 surgical dataset 검증이 추가된다면, 이 연구는 real-time surgical AI deployment를 한 단계 더 앞당길 가능성이 크다.

종합하면, SurgSAM-2는 “SAM2를 surgical video에 맞게 실용적으로 진화시킨 모델”로 볼 수 있으며, 실제 임상 배치를 염두에 둔 의료 영상 AI 연구에서 상당히 중요한 방향성을 제시한다.
