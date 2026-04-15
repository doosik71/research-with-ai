# YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark

- **저자**: Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, Thomas Huang
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1809.03327

## 1. 논문 개요

이 논문의 핵심 목표는 video object segmentation 연구를 위한 대규모 benchmark dataset인 **YouTube-VOS**를 소개하고, 이 데이터셋 위에서 기존 대표 방법들의 성능을 체계적으로 비교하는 것이다. 여기서 video object segmentation은 첫 프레임에서 특정 object instance의 mask가 주어졌을 때, 이후 전체 비디오 구간에 걸쳐 그 객체를 계속 분할하는 문제를 말한다.

저자들이 제기하는 연구 문제는 분명하다. 기존 video object segmentation 방법들은 겉으로는 video 문제를 다루지만, 실제로는 static image segmentation 모델을 프레임 단위로 적용하거나, temporal 정보를 쓰더라도 optical flow 같은 별도 pretrained module에 의존하는 경우가 많다. 이런 방식은 본질적으로 video segmentation 자체에 최적화된 end-to-end spatial-temporal feature learning과는 거리가 있다. 그런데 이를 직접 학습하려고 해도, 기존 데이터셋의 규모가 너무 작아서 sequence-to-sequence 혹은 recurrent한 모델을 충분히 학습시키기 어렵다.

이 문제가 중요한 이유는, video segmentation이 tracking, video editing, augmented reality 같은 실제 응용과 직접 연결되기 때문이다. 특히 occlusion, camera motion, appearance change처럼 현실 비디오에서 흔한 난점을 견디려면, 단순한 정적 이미지 기반 표현보다 긴 시간 범위의 temporal coherence를 학습하는 능력이 중요하다. 저자들은 바로 이 병목이 데이터셋의 부족에서 온다고 보고, 이를 해결하기 위해 YouTube-VOS를 제안한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 새로운 segmentation 알고리즘을 제안하는 것이 아니라, **video object segmentation 분야가 더 강한 spatial-temporal 모델로 발전하기 위해서는 먼저 충분히 큰 학습 데이터셋이 필요하다**는 점을 실증적으로 보여주는 데 있다. 따라서 논문의 주된 공헌은 dataset construction과 benchmark establishment에 있다.

기존 접근과의 차별점은 데이터 규모와 구성 방식에서 뚜렷하다. 저자들은 기존 대표 benchmark인 DAVIS가 고품질이긴 하지만 영상 수가 매우 적고, 그 결과 장기 temporal dependency를 직접 학습하는 모델 개발이 제한된다고 본다. YouTube-VOS는 4,453개의 YouTube video clip, 94개의 object category, 7,755개의 object instance, 197,272개의 annotation을 포함하며, 당시 기준으로 기존 데이터셋보다 압도적으로 큰 규모를 제공한다. 단순히 영상 수만 많은 것이 아니라, 사람, 동물, 차량, 액세서리, 일반 물체 등 다양한 category와 다양한 motion/appearance variation을 포함하도록 설계된 점이 중요하다.

또 하나의 중요한 설계는 validation set에 **unseen categories**를 포함했다는 점이다. training set에는 65개의 seen category가 있고, validation set에는 총 91개의 category가 있으며 그중 26개는 training에 등장하지 않는 unseen category다. 이를 통해 모델의 단순 암기 능력이 아니라, 새로운 객체 범주에 대한 generalization ability까지 평가할 수 있게 했다.

## 3. 상세 방법 설명

이 논문은 새로운 segmentation network 구조를 제안하는 논문이 아니므로, 일반적인 의미의 model architecture나 loss function을 상세히 제시하지 않는다. 대신 데이터셋 구축 파이프라인과 benchmark evaluation setup이 방법론의 핵심이다.

전체 데이터셋 구축 과정은 다음 흐름으로 이해할 수 있다. 먼저 저자들은 segmentation 대상이 될 object category 집합을 신중하게 고른다. 초기에는 동물, 차량, 액세서리, 일반 사물, 그리고 사람의 다양한 활동과 관련된 category를 포함한 78개의 category를 설계했다. 특히 human-related video는 단순히 person만이 아니라 skateboard, tennis racket, motorcycle처럼 사람과 상호작용하는 object도 함께 등장하게 하여 motion diversity를 높이려 했다.

그 다음, 대규모 video classification dataset인 **YouTube-8M**을 이용해 관심 category에 대응되는 candidate video를 수집한다. 각 category마다 최대 100개의 video를 retrieval한다. YouTube video를 사용한 이유는 object appearance와 motion이 다양하고, occlusion, 빠른 motion, camera shake, illumination change 같은 실제 환경의 어려움이 풍부하게 포함되기 때문이다.

수집된 원본 video는 길이가 길고 shot transition이 존재하기 때문에, off-the-shelf shot detection algorithm을 이용해 여러 clip으로 분할한다. 이후 영상의 처음과 끝 10% 구간은 intro subtitle이나 credit이 포함될 가능성이 높아서 제거한다. 그 뒤 각 원본 video에서 길이 3초에서 6초 정도의 clip을 최대 5개까지 선택하고, 사람이 직접 검수해 적절한 object가 실제로 존재하는지, scene transition이 없는지, 너무 어둡거나 흔들리거나 blur가 심하지 않은지를 확인한다.

annotation 단계에서는 각 video clip마다 annotator가 최대 5개의 object를 선택해 instance mask를 만든다. 이때 polygon 수준의 거친 표시가 아니라 boundary를 따라 정밀하게 tracing한다. annotation은 30fps video에서 **매 5프레임마다** 수행되므로, 결과적으로 6fps sampling rate에 해당한다. 저자들은 five consecutive frames 사이의 temporal correlation이 충분히 강하다고 보고, 모든 프레임을 densely annotate하는 대신 skip-frame annotation 전략을 선택했다. 이는 annotation budget 내에서 더 많은 video와 object를 포함하게 해 주므로, 데이터셋 규모 확대에 유리하다.

또한 annotator는 해당 video의 주된 category object뿐 아니라, 눈에 띄는 다른 salient object도 함께 annotate하도록 요청받는다. 예를 들어 “skateboarding” 관련 video라면 person뿐 아니라 skateboard도 함께 라벨링해야 한다. 이 과정에서 최종적으로 activity label은 제거하고, 순수한 object instance category만 남긴다. 이렇게 해서 초기 category set보다 넓은 총 94개의 object category가 만들어졌다.

논문에는 별도의 수학적 학습 목표식이나 손실 함수가 제시되지 않는다. benchmark 평가에서 사용한 metric은 DAVIS 계열에서 널리 쓰이는 **region similarity $J$** 와 **contour accuracy $F$** 이다. 최종 점수는 seen/unseen category에 대해 각각 계산한 네 값, 즉 $J_{seen}$, $F_{seen}$, $J_{unseen}$, $F_{unseen}$의 평균이다. 논문은 이 평균값을 “Overall” 성능으로 사용한다. 명시적으로 쓰면 다음과 같이 이해할 수 있다.

$$
\mathrm{Overall} = \frac{J_{seen} + F_{seen} + J_{unseen} + F_{unseen}}{4}
$$

다만 이 식 자체가 논문에 수식으로 적혀 있는 것은 아니고, 본문 설명을 바탕으로 정리한 표현이다.

실험에 사용한 baseline method는 OSVOS, MaskTrack, OSMN, OnAVOS, S2S다. 이 가운데 OSVOS, MaskTrack, OnAVOS는 online learning을 사용하고, OSMN과 S2S의 한 설정은 online learning 없이 inference한다. 저자들은 각 방법의 공개 코드를 사용해 YouTube-VOS training set에서 다시 학습시켰고, 입력 frame은 기본적으로 $256 \times 448$로 resize했다. 480p도 실험했지만 큰 차이는 없었다고 명시한다.

## 4. 실험 및 결과

실험은 YouTube-VOS 전체 4,453개 video를 training 3,471개, validation 474개, test 508개로 나눈 뒤 수행된다. test set은 challenge 기간에만 공개되므로, 논문에서는 validation set으로 평가한다. 이 설정 자체가 benchmark 역할을 하도록 설계된 것이다.

비교 대상은 앞서 언급한 다섯 계열의 대표 방법들이다. 평가 지표는 $J$ 와 $F$ 이며, seen/unseen category를 구분해 각각 계산한다. 이 점이 중요하다. 단순히 평균 IoU 비슷한 단일 수치만 보는 것이 아니라, 학습에 등장한 category에 대한 성능과 처음 보는 category에 대한 일반화 성능을 구분해서 본다.

정량 결과는 다음과 같은 해석이 가능하다.  
OSVOS는 $J_{seen}=59.8\%$, $J_{unseen}=54.2\%$, $F_{seen}=60.5\%$, $F_{unseen}=60.7\%$, Overall 58.8%를 기록했다.  
MaskTrack은 Overall 53.1%, OSMN은 51.2%, OnAVOS는 55.2%였다.  
반면 S2S는 online learning 없이도 Overall 57.6%로 강한 성능을 보였고, online learning을 추가하면 Overall 64.4%까지 올라가 가장 높은 성능을 기록했다.

이 결과에서 저자들이 강조하는 메시지는 분명하다. **long-term spatial-temporal coherence를 직접 모델링하는 S2S가, online learning이 없더라도 강력한 baseline들과 비슷하거나 더 나은 성능을 낸다**는 점이다. 이는 video object segmentation에서 temporal modeling이 본질적으로 중요하다는 주장에 힘을 실어 준다. 더욱이 S2S with OL은 당시 강한 online learning 기반 방법인 OSVOS보다 약 6%p 높은 전체 정확도를 기록했다.

흥미로운 점은 DAVIS에서 강력했던 OnAVOS가 YouTube-VOS에서는 기대만큼 좋지 않았다는 것이다. 저자들은 이 현상을, YouTube-VOS의 더 큰 appearance variation과 더 복잡한 motion pattern 때문에 online adaptation이 자주 실패하기 때문이라고 해석한다. 이 해석은 dataset이 실제로 더 challenging하다는 간접 증거이기도 하다.

seen과 unseen category 간 격차도 중요한 관찰 결과다. 모든 방법이 대체로 seen category에서 더 높은 성능을 보였다. 이는 object category generalization이 아직 충분하지 않음을 뜻한다. 저자들은 그중 OSVOS가 seen/unseen 격차가 가장 작다고 말하는데, 이는 large-scale image segmentation dataset에서의 pretraining 덕분일 가능성을 언급한다. 즉, online learning이 unseen category에서도 어느 정도 도움은 되지만, 그보다 더 근본적으로는 일반적인 object representation을 배우는 대규모 pretraining이 중요하다는 해석이다.

추론 속도 측면에서는 online learning을 쓰지 않는 OSMN과 S2S (w/o OL)가 각각 0.14초/frame, 0.16초/frame으로 매우 빠르다. 반면 online learning 기반 방법들은 9초/frame에서 13초/frame 수준으로 훨씬 느리다. 따라서 정확도만이 아니라 실제 배치 환경, 특히 mobile application 같은 실시간 요구 환경에서는 속도 역시 중요한 trade-off임을 보여준다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 분야 전체의 병목을 정확히 짚고, 그것을 해결하는 실질적인 인프라를 제공했다는 점이다. 기존 video object segmentation 연구가 작은 데이터셋에 과도하게 의존하던 상황에서, 대규모 학습과 공정한 비교를 가능하게 하는 benchmark를 제시했다는 공헌은 매우 크다. 단순히 비디오 수만 늘린 것이 아니라, category diversity, instance diversity, total annotation 수, 총 영상 길이까지 모두 크게 확장했다는 점도 설득력이 있다.

또 다른 강점은 evaluation protocol이 단순 평균 성능에만 머물지 않고 seen/unseen category를 나눠 generalization을 측정하도록 설계되었다는 점이다. 이는 benchmark가 단순 leaderboard용이 아니라, 모델이 무엇을 배우는지 더 깊게 파악하게 해 준다. 또한 baseline 비교를 통해 static-image 중심 접근, online-learning 기반 접근, sequence-based 접근의 상대적 장단점을 한 자리에서 확인할 수 있게 했다.

실용적 관점에서도 장점이 있다. YouTube 기반 영상은 occlusion, camera motion, blur, fast motion 등 현실적 난점을 포함하고 있어, 실제 응용에 가까운 환경을 반영한다. 또 dense annotation 대신 skip-frame annotation을 채택해 annotation cost와 규모 사이의 균형을 잡은 것도 dataset engineering 측면에서 현실적인 선택이다.

한계도 분명하다. 첫째, 이 논문은 dataset paper이므로 새로운 모델이나 end-to-end learning framework 자체를 깊이 있게 제시하지 않는다. 따라서 “왜 어떤 구조가 temporal dependency를 더 잘 학습하는가”에 대한 알고리즘적 통찰은 제한적이다. 둘째, annotation이 every five frames 방식이기 때문에, 아주 빠른 motion이나 짧은 시간 scale의 fine-grained temporal variation을 다루는 연구에는 제약이 있을 수 있다. 저자들은 temporal correlation이 충분히 강하다고 보지만, 이 가정이 모든 장면에 대해 항상 성립한다고 논문이 엄밀히 입증하는 것은 아니다.

셋째, validation set의 unseen category 설정은 generalization 평가에 유용하지만, unseen 성능이 낮은 원인을 category shift, appearance shift, motion shift 중 무엇으로 분해해서 설명하지는 않는다. 넷째, baseline 재학습 설정에서 해상도 변화의 영향이 “negligible”하다고만 언급될 뿐, 그 세부 분석은 제공되지 않는다. 다섯째, benchmark 중심 논문인 만큼 annotation consistency, inter-annotator variance, class imbalance 같은 dataset 품질 이슈에 대한 정량 분석은 충분히 상세하지 않다.

비판적으로 보면, 이 논문은 “큰 데이터셋이 필요하다”는 주장을 매우 강하게 뒷받침하지만, 동시에 large-scale data만으로 해결되지 않는 문제도 드러낸다. 특히 unseen category 성능 격차는 더 많은 데이터뿐 아니라 representation learning과 generalization 방법론 자체가 여전히 중요하다는 점을 보여준다. 즉, YouTube-VOS는 문제를 해결했다기보다, 더 어려운 연구 질문을 드러내고 표준화한 역할을 한다고 보는 편이 정확하다.

## 6. 결론

이 논문의 주요 기여는 세 가지로 요약할 수 있다. 첫째, 당시 기준으로 가장 큰 video object segmentation dataset인 **YouTube-VOS**를 구축했다. 둘째, 다양한 object category와 실제적인 video variation을 포함한 benchmark를 제시했다. 셋째, 여러 state-of-the-art 방법을 동일 조건에서 재학습 및 평가해 향후 연구를 위한 baseline을 마련했다.

이 연구의 중요성은 단순히 데이터셋 하나를 추가했다는 데 있지 않다. video object segmentation이 static image segmentation의 연장선이 아니라, 진정한 spatial-temporal learning 문제로 발전하려면 충분한 규모의 학습 데이터와 일반화 평가 체계가 필요하다는 점을 분명히 했다. 실제 응용 측면에서는 video editing, AR, tracking-assisted segmentation 같은 분야에서 더 견고한 모델 개발의 기반이 될 수 있고, 학술적으로는 sequence modeling, memory mechanism, category generalization, fast inference 등 후속 연구 주제를 촉진하는 역할을 한다. 즉, YouTube-VOS는 하나의 dataset을 넘어, 이 분야의 연구 방향을 바꾸는 인프라성 기여를 한 논문이라고 평가할 수 있다.
