# COCO-Stuff: Thing and Stuff Classes in Context

- **저자**: Holger Caesar, Jasper Uijlings, Vittorio Ferrari
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1612.03716

## 1. 논문 개요

이 논문은 기존의 COCO 2017 데이터셋에 대해 `thing` 클래스뿐 아니라 `stuff` 클래스까지 포함하는 dense pixel-wise annotation을 추가한 **COCO-Stuff** 데이터셋을 제안한다. 저자들의 문제의식은 분명하다. 컴퓨터 비전 연구, 특히 detection과 recognition은 주로 `car`, `person`, `dog`처럼 경계와 형태가 비교적 분명한 `thing`에 집중해 왔지만, 실제 장면 이해에서는 `sky`, `grass`, `road`, `wall`처럼 형태가 고정되지 않은 `stuff` 역시 매우 중요하다는 점이다.

논문이 다루는 핵심 연구 문제는 세 가지로 요약된다. 첫째, 복잡한 자연 이미지에서 `stuff`를 대규모로 정확하고 효율적으로 annotation할 수 있는가. 둘째, `stuff`와 `thing`을 함께 주석화했을 때 장면 이해에 어떤 추가적 정보가 생기는가. 셋째, 실제 semantic segmentation 관점에서 `stuff`와 `thing`은 어떤 차이를 보이며, 대규모 데이터셋이 성능 향상에 얼마나 기여하는가.

이 문제는 중요하다. 저자들은 `stuff`가 단순 배경이 아니라 장면의 종류를 규정하고, 객체의 위치 가능성을 제약하며, 물체 간 관계와 3D 구조를 이해하게 해 주는 문맥 정보라고 주장한다. 예를 들어 `train`만 알아서는 장면을 충분히 이해하기 어렵지만, `track`, `station`, `bridge` 같은 `stuff`가 함께 라벨링되면 열차가 어디에 있고 어떤 관계 속에 있는지 훨씬 잘 해석할 수 있다. 따라서 COCO-Stuff는 단순한 데이터셋 확장이 아니라, 객체 중심 인식을 장면 중심 이해로 넓히는 기반 작업이라고 볼 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **기존 COCO의 정교한 thing annotation을 활용하면서, 나머지 영역에 대해 효율적인 stuff annotation protocol을 설계한다**는 것이다. 즉, `thing`을 처음부터 다시 그리지 않고 COCO에 이미 존재하는 instance-level outline을 그대로 사용하고, annotator는 `stuff`만 빠르게 칠하도록 만든다. 이 방식은 annotation 비용을 크게 줄이면서도 thing-stuff 경계에서 높은 정밀도를 얻는 데 핵심 역할을 한다.

기존 접근과의 차별점은 크게 세 가지다. 첫째, 자유 텍스트 기반 label naming이 아니라 **미리 정의된 mutually exclusive label set**을 사용한다. 저자들은 free-form label 방식이 synonym 문제, granularity 불일치, 희귀 클래스 남발 문제를 낳는다고 지적한다. 둘째, annotation 단위를 polygon이 아니라 **superpixel** 기반으로 바꿨다. 이는 사람이 세밀한 경계를 직접 따지 않아도 되므로 속도를 크게 높인다. 셋째, 단지 데이터셋을 공개하는 데 그치지 않고, 이 데이터셋을 이용해 `stuff`의 중요성, `stuff-thing` spatial context, segmentation 난이도를 함께 분석한다.

이 논문에서 중요한 직관은 `stuff`를 “배경”으로 뭉뚱그리지 말고, `thing`과 동등한 수준의 semantic entity로 취급해야 한다는 점이다. 그리고 그렇게 하기 위해서는 클래스 집합이 너무 거칠지도, 너무 세분화되지도 않아야 하며, `thing`과 `stuff`가 비슷한 빈도 분포와 비슷한 granularity를 갖도록 설계되어야 한다고 본다.

## 3. 상세 방법 설명

논문이 제안하는 COCO-Stuff는 총 172개 클래스로 구성된다. 이 중 80개는 기존 COCO의 `thing` 클래스이고, 91개는 새로 정의한 `stuff` 클래스이며, 나머지 1개는 `unlabeled`이다. `unlabeled`는 두 경우에 사용된다. 픽셀이 사전에 정의된 171개 클래스에 속하지 않거나, annotator가 라벨을 추론할 수 없을 때다.

### 라벨 설계 원칙

저자들은 `stuff` 라벨 집합을 전문가가 미리 설계했다. 그 기준은 다음과 같다. 클래스는 서로 배타적이어야 하고, 전체적으로 데이터셋의 대부분의 `stuff` 영역을 덮어야 하며, 충분히 자주 등장해야 하고, 사람 기준의 적절한 base-level granularity를 가져야 한다. 예를 들어 `vegetation` 하나로 통합하면 너무 거칠고, 모든 식물 종을 분리하면 너무 드물어진다. 그래서 `grass`, `bush`, `moss`, `straw` 같은 자주 나오는 하위 클래스를 두고, 나머지는 `vegetation-other`로 묶는 계층 구조를 만들었다. 일부 super-category에서는 재질 정보가 중요하다고 보고 `wall-brick`, `wall-wood`, `floor-marble`처럼 material-aware class를 둔 것도 특징이다.

이 설계 덕분에 모든 라벨은 leaf node만 사용하도록 강제되어 상호배타성이 유지된다. 또한 수집 후 통계를 보면 전체 픽셀 중 `unlabeled`는 6%에 불과했고, `stuff` 클래스들의 pixel frequency는 COCO의 `thing` 클래스들과 비슷한 범위를 보였다. 저자들은 이 점을 `stuff-thing` 관계 분석에 적합한 데이터셋 설계의 증거로 본다.

### Annotation Protocol

핵심 annotation 절차는 세 단계로 이해할 수 있다.

첫째, 각 이미지를 SLICO 알고리즘으로 약 1,000개의 superpixel로 분할한다. superpixel은 경계에 잘 맞도록 설계되어 있으므로 annotator는 픽셀 단위 경계를 직접 그리지 않고, 어느 superpixel이 어떤 클래스인지 지정하면 된다.

둘째, annotator에게 size-adjustable paintbrush를 제공해 넓은 `stuff` 영역을 한 번에 칠할 수 있도록 했다. 이는 특히 `sky`, `snow`, `road`처럼 큰 영역을 다룰 때 효율적이다.

셋째, 기존 COCO의 `thing` mask를 overlay로 보여주고 해당 픽셀은 clamped 상태로 고정한다. 즉 annotator는 `person`, `car`, `train` 같은 전경 객체를 건드리지 않고 그 주변 `stuff`만 칠하면 된다. 이 때문에 `stuff-thing` 경계는 superpixel보다도 더 정확하게 처리될 수 있다. 왜냐하면 그 경계는 COCO의 기존 정밀 outline을 그대로 따르기 때문이다.

또한 라벨 선택 UI도 단순 리스트가 아니라 계층 구조로 제공했다. 저자들은 이 방식이 annotator의 label lookup 시간을 줄였다고 보고한다.

### 효율성과 품질 분석

저자들은 동일 annotator가 10개 이미지에 대해 세 가지 방식으로 annotation하도록 했다.

- `freedraw`: 가능한 한 픽셀 정확도를 높인 기준 annotation
- `polygon`: 전통적인 polygon 기반 annotation
- `superpixel`: 논문이 제안한 방식

결과는 매우 직접적이다. `freedraw`를 기준 속도 1.0으로 둘 때, `polygon`은 1.5배 빠르고, `superpixel`은 2.8배 빠르다. 동시에 `freedraw` 대비 agreement는 polygon 97.3%, superpixel 96.1%였다. 흥미로운 점은 `freedraw` 자체의 self-agreement가 96.6%라는 것이다. 즉 superpixel 방식과 freedraw 사이의 차이가, 같은 annotator가 freedraw를 두 번 했을 때 생기는 자연 변동과 비슷한 수준이라는 뜻이다. 저자들은 이를 근거로 superpixel 방식이 거의 동등한 품질을 유지한다고 해석한다.

경계 복잡도와 annotation 시간의 관계도 분석했다. boundary complexity는 이웃 픽셀 중 다른 semantic label을 가진 픽셀 비율로 정의된다. 복잡도가 커질수록 annotation 시간은 늘어나는데, 그 증가율은 polygon과 freedraw가 superpixel보다 훨씬 가파르다. 이는 superpixel이 복잡한 경계가 많은 이미지일수록 더 큰 이점을 가진다는 뜻이다.

thing overlay의 효과도 따로 측정했다. overlay를 사용하면 freedraw는 1.8배, superpixel은 1.2배 속도 향상을 보였고, superpixel의 품질은 overlay 유무에 상관없이 96.1% agreement로 동일했다. 따라서 기존 thing annotation의 재활용은 “품질 손실 없는 속도 향상”으로 정리된다.

### 데이터셋 비교 방법론

논문은 COCO-Stuff를 기존 segmentation 데이터셋과 비교할 때 단순 클래스 수보다 **usable class** 개념을 더 중요하게 본다. free-form label 데이터셋은 nominal class 수가 많아 보여도, 실제로는 극소수 이미지만 포함하는 희귀 클래스가 많아 학습에 부적절하기 때문이다. COCO-Stuff는 164K 이미지, 91개의 stuff class, 80개의 thing class를 가지며, 많은 stuff class가 1,000장 이상 이미지에 등장한다. 이는 PASCAL Context나 ADE20K보다 실질적인 학습 가능 클래스 수가 많다는 주장으로 연결된다.

### 실험에서 사용한 평가 지표

semantic segmentation 실험에서는 네 가지 지표를 사용한다. 논문은 수식을 직접 제시하지는 않았지만, 정의는 다음과 같이 정리할 수 있다.

Pixel accuracy는 전체 픽셀 중 정답으로 분류된 비율이다.

$$
\text{Pixel Accuracy} = \frac{\sum_i n_{ii}}{\sum_i t_i}
$$

여기서 $n_{ii}$는 클래스 $i$의 정답 픽셀 중 올바르게 예측된 픽셀 수이고, $t_i$는 클래스 $i$의 전체 정답 픽셀 수다.

Class accuracy는 클래스별 accuracy의 평균이다.

$$
\text{Class Accuracy} = \frac{1}{K}\sum_{i=1}^{K}\frac{n_{ii}}{t_i}
$$

Mean IoU는 각 클래스별 intersection-over-union의 평균이다.

$$
\text{mIoU} = \frac{1}{K}\sum_{i=1}^{K}\frac{n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}
$$

Frequency weighted IoU는 각 클래스의 등장 빈도로 가중한 IoU다.

$$
\text{FWIoU} = \frac{1}{\sum_k t_k}\sum_{i=1}^{K} t_i \cdot \frac{n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}
$$

또한 spatial context의 복잡성은 conditional probability distribution의 entropy로 정량화한다. 논문은 정확한 수식을 쓰지 않았지만, 개념적으로는 다음과 같은 형태다.

$$
H(X) = - \sum_{c} p(c)\log p(c)
$$

여기서 $p(c)$는 특정 reference class 주변의 공간 bin에서 다른 클래스 $c$가 나타날 조건부 확률이다. entropy가 높을수록 문맥이 다양하다는 뜻이다.

## 4. 실험 및 결과

### 데이터셋 규모와 특성

COCO-Stuff는 COCO 2017의 전체 164K 이미지를 annotation했다. 구체적으로 train 118K, val 5K, test-dev 20K, test-challenge 20K를 포함한다. 이는 semantic segmentation용 dense label 데이터셋 중 매우 큰 규모다. 논문은 특히 기존 데이터셋들이 이미지 수는 적거나, 클래스는 많지만 usable class가 적거나, free-form label로 인해 granularity가 불안정하다는 점을 강조한다.

### Stuff와 Thing의 상대적 중요성

저자들은 먼저 `stuff`와 `thing`의 상대 비중을 측정했다. pixel 기준으로 `stuff`는 69.1%, `thing`은 30.9%를 차지했다. connected component를 region의 proxy로 사용했을 때도 `stuff` 69.4%, `thing` 30.6%였다. 즉 이미지 내 면적과 영역 수 양쪽에서 `stuff`가 더 큰 비중을 차지한다.

하지만 저자들은 이것이 단순 배경 점유율인지 확인하기 위해 COCO의 image caption도 분석했다. 5개의 caption에서 noun을 추출하고, 수작업으로 frequent noun을 `stuff`와 `thing`으로 구분한 결과, 명사 중 38.2%가 `stuff`, 61.8%가 `thing`이었다. 비록 caption에서는 `thing` 언급이 더 많지만, `stuff`가 3분의 1이 넘는 비중으로 직접 언급된다는 점은 사람의 장면 설명에서도 `stuff`가 중요함을 보여준다.

### Spatial Context 분석

논문은 각 reference class 주변의 픽셀 분포를 거리와 각도 기준 histogram으로 누적해 spatial context를 시각화한다. 예를 들어 `car`가 있는 이미지들에서 `car` 영역 주변의 픽셀을 모으고, 각 픽셀이 `car`와 얼마나 떨어져 있고 어느 방향에 있는지를 계산한 뒤, 그 위치에서 가장 자주 등장하는 클래스가 무엇인지 본다.

이 분석을 통해 여러 문맥 관계가 드러난다. `train`은 보통 `railroad` 위쪽에 있고, `tv` 앞에는 `person`이 자주 있으며, `wall-tile` 위에는 `floor-tile`이 있고, `road` 양옆에는 `person`이 많이 등장한다. 저자들은 특히 위아래 방향의 relation, 즉 support relation이 가장 informative하다고 해석한다. 실제로 높은 confidence는 `sky`, `wall`, `ceiling`처럼 위쪽이나 `road`, `pavement`, `snow`처럼 아래쪽에서 자주 나타난다.

문맥의 다양성은 entropy로 측정했다. `stuff` 클래스의 평균 entropy는 3.40, `thing`은 3.02였다. 즉 `stuff`가 더 다양한 문맥 속에 등장한다. 또한 데이터셋 전체 문맥 richness를 비교했을 때 COCO-Stuff의 평균 entropy는 3.22로, PASCAL Context의 usable 60 classes의 2.42, ADE20K usable 150 classes의 2.18, SIFT Flow의 1.20보다 높았다. 이는 COCO-Stuff가 복잡하고 다양한 문맥 관계를 더 잘 포착한다는 저자들의 핵심 주장 중 하나다.

### Semantic Segmentation Baseline

실험 모델은 **DeepLab V2 + VGG-16**이며, ILSVRC로 pretrain된 VGG-16 backbone을 사용했다. 학습은 COCO-Stuff train 118K 이미지, 평가는 val 5K 이미지로 진행했다.

전체 118K 학습 시 성능은 다음과 같다.

- Class accuracy: 45.1%
- Pixel accuracy: 63.6%
- mIoU: 33.2%
- FWIoU: 47.6%

이 수치는 절대적으로 매우 높은 값이라기보다, 이후 COCO-Stuff를 위한 baseline으로 제시된 값이다. 저자들의 관심은 특히 데이터 규모와 stuff/thing 난이도 차이에 있다.

### 데이터 규모의 효과

1K, 5K, 10K, 20K, 40K, 80K, 118K train 이미지로 각각 학습했을 때, 모든 지표가 꾸준히 향상되었다. 예를 들어 mIoU는 1K에서 15.9%, 5K에서 23.1%, 20K에서 28.6%, 80K에서 32.9%, 118K에서 33.2%였다. class accuracy도 24.1%에서 45.1%까지 상승했다. 이는 semantic segmentation에서도 대규모 데이터가 여전히 유의미한 성능 개선을 가져온다는 증거로 제시된다. 저자들은 더 깊은 네트워크라면 큰 데이터셋의 효과를 더 크게 볼 수 있을 것이라고 가설을 제시하지만, 이 부분은 실험으로 검증하지는 않았다.

### Stuff가 Thing보다 쉬운가?

기존 여러 연구는 `stuff`가 `thing`보다 segmentation이 쉽다고 보고해 왔다. 그러나 저자들은 그 결론이 데이터셋 설계의 산물일 수 있다고 비판한다. 많은 기존 데이터셋에서는 `stuff` 클래스가 `sky`, `road`, `grass`처럼 매우 자주 나오고 경계가 거친 소수 클래스 위주였기 때문이다.

COCO-Stuff에서는 사정이 다르다. stuff와 thing이 비슷한 granularity와 분포를 갖도록 설계되었기 때문에 더 공정한 비교가 가능하다. 실제 결과에서 `stuff` 성능은 class accuracy 33.5%, pixel accuracy 58.2%, mIoU 24.0%, FWIoU 45.6%였고, `thing`은 각각 58.3%, 75.7%, 43.6%, 58.4%였다. 즉 COCO-Stuff에서는 `thing`이 `stuff`보다 훨씬 더 잘 segment된다. 저자들은 이를 근거로 “`stuff`는 일반적으로 더 쉽다”는 통념에 반대하고, 적어도 균형 잡힌 데이터셋에서는 그렇지 않다고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 데이터셋 설계와 분석이 서로 유기적으로 연결되어 있다는 점이다. 단지 annotation 수를 늘린 것이 아니라, 왜 `stuff`가 중요한지 문제를 제기하고, 그에 맞는 label design과 annotation protocol을 만들고, 다시 그 데이터셋으로 실제 가설을 검증한다. 특히 superpixel과 existing thing overlay를 결합한 annotation protocol은 비용과 품질의 균형을 매우 잘 잡은 설계로 보인다. self-agreement와 reference agreement를 함께 비교한 부분도 설득력이 높다.

또 다른 강점은 라벨 체계의 일관성이다. 저자들은 free-form labeling이 만드는 ambiguity를 강하게 비판하며, mutually exclusive hierarchy를 통해 이를 해소했다. 이것은 단순 운영 편의가 아니라, 학습 가능한 클래스 분포와 해석 가능한 평가를 위해 매우 중요한 선택이다. 또한 COCO 기반이기 때문에 thing annotation과 image caption을 그대로 활용할 수 있어, segmentation뿐 아니라 context reasoning과 vision-language 연구에도 확장성이 있다.

실험 분석 역시 데이터셋 논문치고 상당히 탄탄하다. `stuff`의 면적 비중, caption에서의 언급 비율, spatial context entropy, DeepLab baseline, dataset scale effect까지 폭넓게 다룬다. 특히 “stuff가 thing보다 쉽다”는 기존 통념을 데이터셋 편향의 관점에서 재검토한 부분은 이 논문의 가장 인상적인 메시지 중 하나다.

반면 한계도 있다. 첫째, 논문은 annotation protocol의 효율성과 일관성을 잘 보여주지만, **최종 annotation quality를 독립적 외부 gold standard와 비교한 대규모 검증**까지 하지는 않는다. freedraw를 reference로 두는 방식은 합리적이지만, 결국 동일 annotator 기반 분석이라는 제약이 있다. 둘째, segmentation baseline은 DeepLab V2 with VGG-16 하나에 국한된다. 논문 목적상 충분할 수는 있으나, 특정 모델 아키텍처의 편향이 `stuff`와 `thing`의 상대 난이도 결과에 일부 영향을 줄 가능성은 남는다. 셋째, entropy 기반 context richness 분석은 흥미롭지만, 그것이 실제 다운스트림 task 성능 향상으로 직접 연결되는지는 이 논문만으로는 확인되지 않는다.

또한 논문은 `furniture` 계열처럼 상황에 따라 `thing` 또는 `stuff`로 해석될 수 있는 모호한 클래스가 있음을 인정한다. 이는 현실적으로 불가피하지만, 경계가 애매한 클래스들에 대한 annotation consistency 문제가 완전히 사라졌다고 보기는 어렵다. 마지막으로, 저자들은 더 깊은 네트워크가 더 큰 데이터셋에서 더 큰 이득을 볼 것이라고 말하지만, 이는 명시적으로 실험되지 않은 추정이다.

## 6. 결론

이 논문은 COCO 2017 전체에 대해 91개의 `stuff` 클래스를 추가해, 총 172개 클래스의 dense annotation을 제공하는 **COCO-Stuff**를 제안했다. 그 기여는 단순히 큰 데이터셋을 만든 데 있지 않다. 저자들은 `stuff`를 장면 이해의 핵심 구성 요소로 재정의하고, 이를 위해 mutually exclusive label hierarchy, superpixel 기반 annotation, thing overlay 활용이라는 실용적이고 효과적인 방법을 설계했다.

논문이 보여준 주요 메시지는 분명하다. `stuff`는 이미지 표면의 대부분을 차지하고, 인간의 장면 설명에서도 자주 언급되며, 다양한 spatial context를 구성한다. 또한 균형 잡힌 데이터셋에서는 `stuff`가 결코 `thing`보다 쉬운 segmentation 대상이 아니며, 대규모 데이터는 semantic segmentation 성능을 꾸준히 끌어올린다.

실제 적용 측면에서 COCO-Stuff는 semantic segmentation, scene parsing, context modeling, thing-stuff joint reasoning, vision-language grounding 같은 다양한 연구의 기반이 될 수 있다. 향후 연구에서는 이 데이터셋을 이용해 panoptic understanding, context-aware detection, multimodal scene understanding처럼 더 통합적인 장면 해석 문제로 나아갈 가능성이 크다. 이 논문은 그런 방향으로 가기 위한 중요한 데이터적 토대를 제공한 작업으로 평가할 수 있다.
