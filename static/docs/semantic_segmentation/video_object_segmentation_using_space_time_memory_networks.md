# Video Object Segmentation using Space-Time Memory Networks

- **저자**: Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1904.00607

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation 문제를 다룬다. 설정은 간단하다. 비디오의 첫 프레임에서는 target object의 정답 mask가 주어지고, 나머지 모든 프레임에서 해당 객체의 mask를 예측해야 한다. 하지만 실제 문제는 쉽지 않다. 시간이 지나면서 객체의 appearance가 크게 변할 수 있고, occlusion이 발생하며, 이전 프레임의 오차가 누적되어 drift가 생기기 때문이다.

저자들이 제기하는 핵심 문제는 기존 deep learning 기반 방법들이 사용할 수 있는 guidance information을 충분히 활용하지 못한다는 점이다. 어떤 방법은 첫 프레임만 참고하고, 어떤 방법은 직전 프레임만 참고하며, 일부는 둘 다 사용한다. 그러나 비디오를 순차적으로 처리하면 중간 예측 결과들도 계속 쌓이므로, 실제로는 “활용 가능한 정보”가 점점 더 풍부해진다. 이 논문은 바로 이 점에 주목한다. 즉, 과거의 여러 프레임과 그 mask 정보를 모두 외부 memory로 저장하고, 현재 프레임을 query로 두어 필요한 정보를 읽어오는 구조를 제안한다.

이 문제의 중요성은 매우 크다. Video object segmentation은 video editing, tracking, interactive video manipulation 같은 응용의 핵심 단계이며, 실제 환경에서는 appearance change, occlusion, fast motion, complex background를 모두 견뎌야 한다. 따라서 더 강건하고 빠른 방법은 학술적으로도, 실용적으로도 가치가 높다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 과거 프레임들을 “memory”, 현재 프레임을 “query”로 보고, query의 각 pixel이 memory 안의 모든 space-time 위치와 dense matching을 수행하도록 만드는 것이다. 이를 통해 현재 픽셀이 어떤 과거 위치의 어떤 정보를 참고해야 하는지를 네트워크가 직접 학습한다.

기존 접근과의 차별점은 분명하다. 이전 방법들은 보통 첫 프레임, 직전 프레임, 혹은 이 둘 정도만 사용했다. 반면 이 논문은 과거 여러 프레임을 유연하게 memory에 넣고, query pixel이 모든 시간축과 공간축 위치를 대상으로 attention-like retrieval을 하게 만든다. 저자들은 이를 space-time memory read라고 부른다. 개념적으로는 memory network와 non-local matching의 성격을 동시에 가진다.

또 하나 중요한 차별점은 online learning이 없다는 점이다. 많은 기존 강력한 방법들은 test time에 첫 프레임을 이용해 네트워크를 fine-tuning하는 online adaptation을 사용한다. 정확도는 좋아질 수 있지만 속도가 느리고 practical하지 않다. 이 논문은 intermediate prediction을 memory에 추가하는 방식으로 사실상 online adaptation과 유사한 효과를 얻되, 모델 파라미터를 업데이트하지 않는다. 즉, 추가 학습 없이도 빠르고 강한 적응성을 얻는 것이 핵심이다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 네 부분으로 구성된다. memory encoder, query encoder, space-time memory read block, decoder이다. 비디오는 두 번째 프레임부터 순차적으로 처리된다. 첫 프레임의 ground truth mask와, 이후 시점들에서 생성된 예측 mask가 붙은 과거 프레임들이 memory가 된다. 현재 처리 중인 프레임은 query가 된다.

먼저 query encoder는 현재 RGB frame만 입력받아 key와 value feature map을 만든다. memory encoder는 RGB frame과 object mask를 함께 입력받아 역시 key와 value를 만든다. 여기서 key는 어떤 memory 위치를 읽어야 할지 결정하기 위한 addressing 표현이고, value는 실제 segmentation에 필요한 정보를 담는 표현이다. 논문은 query value와 memory value의 역할도 구분해서 설명한다. Query value는 현재 프레임의 세밀한 appearance 정보를 저장하고, memory value는 visual semantics와 foreground/background 관련 정보를 함께 저장한다.

형태를 보면, query embedding은
$k^Q \in \mathbb{R}^{H \times W \times C/8}$,
$v^Q \in \mathbb{R}^{H \times W \times C/2}$
이다. Memory가 $T$개 프레임이면 memory embedding은
$k^M \in \mathbb{R}^{T \times H \times W \times C/8}$,
$v^M \in \mathbb{R}^{T \times H \times W \times C/2}$
가 된다. Backbone은 ResNet50의 `res4` feature를 사용한다. Memory encoder는 RGB 3채널에 mask 1채널을 더해 4채널 입력을 처리하도록 첫 convolution이 수정된다.

이 논문의 핵심 연산은 space-time memory read이다. Query의 각 위치 $i$는 memory의 모든 시공간 위치 $j$와 similarity를 계산하고, 그 가중합으로 memory value를 읽어온다. 논문이 제시한 수식은 다음과 같다.

$$
y_i = \left[ v_i^Q,\ \frac{1}{Z}\sum_{\forall j} f(k_i^Q, k_j^M) v_j^M \right]
$$

여기서 $[\cdot,\cdot]$는 concatenation이고, 정규화 항은

$$
Z = \sum_{\forall j} f(k_i^Q, k_j^M)
$$

이다. Similarity 함수는 dot-product 기반 exponential이다.

$$
f(k_i^Q, k_j^M) = \exp(k_i^Q \circ k_j^M)
$$

즉, 현재 query pixel과 비슷한 key를 가진 과거 memory 위치들에 높은 가중치를 주고, 해당 value들을 weighted sum으로 읽어온 뒤, query의 own value와 concat해서 decoder로 보낸다. 쉽게 말하면 “현재 이 픽셀을 분할하려면 과거 어느 프레임의 어느 위치를 참고해야 하는가?”를 네트워크가 전 공간-시간 범위에서 찾는 구조다.

Decoder는 read 결과로부터 최종 object mask를 복원한다. 저자들은 이전 연구 [24]의 refinement module을 사용한다. Read output을 먼저 convolution과 residual block으로 256채널로 압축한 뒤, refinement module을 여러 단계 거치며 해상도를 2배씩 올린다. 각 단계는 query encoder의 대응 scale feature를 skip-connection으로 받아 세밀한 경계 복원을 돕는다. 마지막에는 convolution과 softmax로 2채널 출력을 만들고, 입력 영상의 1/4 해상도에서 mask를 예측한다.

Multi-object segmentation도 지원한다. 기본 설명은 single object 기준이지만, 실제 벤치마크는 multi-object를 요구한다. 이를 위해 객체별로 독립적으로 모델을 실행해 mask probability map을 얻고, 이후 soft aggregation으로 병합한다. 논문은 이 mask merging을 test-time post-processing이 아니라 differentiable layer로 정의해 training과 inference 모두에 적용했다고 설명한다. 다만 구체적인 세부식은 supplementary material에 있다고 되어 있으므로, 제공된 본문만으로는 더 자세한 수식은 확인할 수 없다.

학습은 two-stage training이다. 첫 단계는 static image로 만든 synthetic video를 사용한 pre-training이다. 한 장의 정적 이미지에 random affine transform을 적용해 3프레임짜리 가상 비디오를 생성한다. 사용된 변환은 rotation, sheering, zooming, translation, cropping이다. 사용 데이터는 salient object detection과 semantic segmentation 데이터셋들이다. 저자들의 논리는 이 방법이 temporal smoothness에 크게 의존하지 않고 먼 픽셀 간 semantic matching을 배우므로, 실제 긴 비디오가 없어도 학습이 가능하다는 것이다.

두 번째는 real video를 이용한 main training이다. 평가 대상에 따라 Youtube-VOS 또는 DAVIS-2017으로 학습한다. 학습 샘플은 시간 순서가 있는 3개 프레임으로 구성되며, 장기 appearance change를 배우기 위해 프레임을 랜덤하게 skip한다. 최대 skip 수는 0에서 25까지 점진적으로 늘린다. 이는 curriculum learning 방식이다.

중요한 점은 training 중에도 dynamic memory를 사용한다는 것이다. 시스템이 프레임을 따라 전진하면서, 이전 단계의 예측 결과를 thresholding 없이 raw probability map 형태로 memory에 넣는다. 이렇게 하면 예측 불확실성까지 memory embedding에 반영할 수 있다.

학습 세부사항도 명시되어 있다. 입력은 랜덤 crop한 $384 \times 384$ 패치이고, mini-batch size는 4이다. Batch normalization은 모두 비활성화했다. 손실 함수는 cross-entropy loss이고, optimizer는 Adam, learning rate는 $1e-5$이다. 4개의 NVIDIA GeForce 1080 Ti GPU를 사용했으며, pre-training 약 4일, main training 약 3일이 걸렸다고 한다.

추론 시에는 모든 과거 프레임을 memory에 저장하면 GPU 메모리와 속도 문제가 발생할 수 있으므로, 간단한 memory management rule을 사용한다. 첫 프레임과 직전 프레임은 항상 memory에 넣는다. 그리고 중간 프레임은 매 $N$프레임마다 하나씩 저장하는데, 기본값은 $N=5$이다. 이 규칙은 속도와 정확도의 trade-off를 조절한다.

## 4. 실험 및 결과

논문은 Youtube-VOS와 DAVIS를 평가에 사용한다. Youtube-VOS는 대규모 multi-object benchmark이고, DAVIS는 video object segmentation에서 널리 쓰이는 benchmark이다. DAVIS-2016은 single-object, DAVIS-2017은 multi-object 설정이다.

평가 지표는 region similarity $J$와 contour accuracy $F$이다. Youtube-VOS는 seen/unseen category를 분리해 평가하고 overall score도 보고한다. DAVIS는 공식 benchmark code로 평가했다고 명시한다. 또한 이 논문은 post-processing 없이 network output을 직접 사용했다고 밝힌다. 이 점은 결과 해석에서 중요하다. 후처리 없이도 강한 성능을 보인다는 뜻이기 때문이다.

Youtube-VOS validation에서 제안 방법은 매우 큰 폭의 성능 향상을 보인다. Table 1에 따르면, 제안 방법은 seen/unseen 전체를 포함한 overall score 79.4를 달성했다. 세부적으로 seen $J=79.7$, seen $F=84.2$, unseen $J=72.8$, unseen $F=80.9$이다. 표 형식이 추출 과정에서 다소 흐트러져 있지만, 논문 본문과 abstract에서 제시하는 핵심 메시지는 Youtube-VOS val에서 overall 79.4로 당시 state-of-the-art를 달성했다는 것이다. 이전 강한 방법들과 비교해 unseen category에서도 개선 폭이 크다는 점은 generalization 측면에서 의미가 있다.

DAVIS-2016 validation에서는 online learning 없이도 매우 강한 성능을 보인다. 제안 방법은 $J$ Mean 84.8, $F$ Mean 88.1, runtime 0.16초/frame을 기록한다. Youtube-VOS 추가 학습을 포함한 버전은 $J=88.7$, $F=89.9$로 더 높아진다. 이는 많은 online learning 기반 방법들과 비교해도 경쟁력 있거나 더 우수한 수준이며, 속도는 훨씬 빠르다. 논문 abstract에서도 DAVIS 2016 val에서 $J=88.7$을 강조한다.

DAVIS-2017 validation에서는 기본 모델이 $J=69.2$, $F=74.0$이고, Youtube-VOS 추가 학습을 사용하면 $J=79.2$, $F=84.3$으로 크게 향상된다. 논문 abstract는 DAVIS 2017 val에서 $J=79.2$를 강조한다. 저자들은 이를 통해 DAVIS만으로는 일반화 가능한 deep network를 학습하기 어렵고, 대규모 데이터가 매우 중요하다고 해석한다.

정성적 결과에서도 장점이 드러난다. Fig. 4는 occlusion 전후나 복잡한 motion이 있는 어려운 장면에서 제안 방법이 robust하게 object를 유지하는 예를 보여준다고 설명한다. 특히 intermediate memory frame을 추가로 사용하는 설정은 어려운 장면에서 더 좋은 결과를 낸다. Fig. 6과 Fig. 7의 분석에 따르면, `First + Previous`만 써도 이미 강력하지만, `Every 5 frames`로 intermediate memory를 더 넣으면 특히 하위 percentile의 어려운 샘플들에서 Jaccard score가 개선된다. 즉, 쉬운 사례에서는 차이가 작지만 실패하기 쉬운 장면에서 memory 확장이 의미 있게 작동한다.

Ablation도 흥미롭다. Training data 분석(Table 4)에서 pre-training only 모델이 main-training only보다 Youtube-VOS에서 더 좋다. 구체적으로 pre-training only는 Youtube-VOS overall 69.1, DAVIS-2017에서는 57.9/$62.1$ 수준이고, main-training only는 Youtube-VOS 68.2, DAVIS-2017 38.1/$47.9$ 수준으로 더 나쁘다. 완전한 two-stage training이 가장 좋다. 이는 정적 이미지 기반 pre-training이 단순한 보조 단계가 아니라 일반화와 overfitting 방지에 핵심 역할을 한다는 뜻이다.

Memory management 분석(Table 5)도 설계의 타당성을 보여준다. Youtube-VOS/DAVIS 기준으로 `First only`, `Previous only`, `First & Previous`, `Every 5 frames`를 비교했을 때, `First & Previous`가 이미 매우 강하고, `Every 5 frames`가 최종적으로 가장 좋은 성능을 낸다. 예를 들어 DAVIS-2016에서는 `First & Previous`가 87.8, `Every 5 frames`가 88.7이다. 절대 차이는 크지 않지만, 어려운 장면에서의 안정성이 높아진다. 속도는 0.07초/frame에서 0.16초/frame으로 증가한다. 즉, 정확도 향상을 위해 속도를 일부 희생하는 구조다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의와 모델 설계가 매우 잘 맞물린다는 점이다. Semi-supervised video object segmentation에서는 시간이 지날수록 사용할 수 있는 reference가 많아진다. 저자들은 이 특성을 memory라는 형태로 정식화했고, query pixel이 모든 과거 시공간 위치를 참고할 수 있게 만들었다. 이는 문제 구조를 직접 반영한 설계다.

두 번째 강점은 online learning 없이도 높은 정확도와 빠른 추론 속도를 동시에 달성했다는 점이다. 많은 기존 방법들이 test-time fine-tuning에 의존했지만, 이 논문은 memory update만으로 적응성을 확보했다. 논문이 제시한 0.16초/frame 수준의 속도는 실용성을 크게 높인다.

세 번째 강점은 first frame과 previous frame의 장점을 동시에 살리면서, intermediate frame까지 유연하게 확장할 수 있다는 점이다. 기존 방법들은 특정 reference frame 조합에 묶여 있었지만, STM은 memory slot을 늘리는 방식으로 자연스럽게 확장된다. 이 구조는 occlusion, appearance change, drift에 강한 이유를 설명해 준다.

네 번째는 학습 전략이다. Static image로 synthetic video를 만들어 pre-training하는 전략은 데이터 효율성과 일반화 측면에서 설득력이 있다. 실제로 ablation에서 그 효과가 분명히 나타난다.

한계도 있다. 첫째, memory를 모두 저장하지 않고 단순한 규칙으로 샘플링한다. 첫 프레임, 직전 프레임, 그리고 매 $N$프레임만 저장하는 방식은 practical하지만, 어떤 프레임을 저장해야 가장 유익한지 content-aware하게 선택하지는 않는다. 즉, memory management는 강력하지만 아직 heuristic이다.

둘째, 본문만 기준으로 보면 multi-object mask merging의 구체적 수식과 구현 세부는 충분히 설명되지 않는다. 저자들은 supplementary material에 더 자세한 내용을 둔다고 했으므로, 제공된 본문만으로는 그 부분을 완전히 재구성하기 어렵다.

셋째, query와 memory 사이의 전역 dense matching은 강력하지만 계산량 증가 가능성이 있다. 논문은 tensor operation으로 효율적으로 구현 가능하다고 설명하고 실제 속도도 좋지만, 더 긴 비디오나 더 높은 해상도에서 memory 크기가 계속 커질 때의 확장성은 완전히 해결된 문제라고 보기는 어렵다. 그래서 실제 inference에서도 전체 프레임을 다 쓰지 않고 일부만 저장한다.

넷째, 대규모 데이터의 의존성이 크다. DAVIS-2017 성능이 Youtube-VOS 추가 학습으로 크게 뛰는 결과는, 반대로 말하면 작은 데이터셋만으로는 일반화가 어렵다는 뜻이다. 이는 모델이 강력한 만큼 충분한 학습 데이터가 중요하다는 점을 보여준다.

비판적으로 보면, 이 논문은 memory network를 video object segmentation에 매우 적절하게 가져온 성공 사례다. 다만 memory selection과 multi-object interaction을 더 정교하게 설계했다면 추가 개선 여지가 있었을 것으로 보인다. 또한 제공된 본문 기준으로는 attention/read 결과가 segmentation 오류를 어떤 경우에 유발하는지에 대한 failure case 분석은 제한적이다.

## 6. 결론

이 논문은 semi-supervised video object segmentation을 위해 Space-Time Memory Network(STM)를 제안했다. 핵심 기여는 과거 프레임과 mask를 memory로 저장하고, 현재 프레임의 각 pixel이 memory 전체의 시공간 위치와 dense matching을 수행해 필요한 정보를 읽어오도록 만든 점이다. 이를 통해 appearance change, occlusion, drift 같은 어려운 상황을 더 잘 다룰 수 있었다.

실험적으로도 기여가 분명하다. Youtube-VOS와 DAVIS에서 당시 state-of-the-art 수준의 성능을 보였고, 특히 online learning 없이도 빠른 속도와 높은 정확도를 동시에 달성했다. 이 점은 실제 적용 가능성을 크게 높인다.

향후 관점에서도 의미가 크다. 저자들이 직접 언급하듯, 이 구조는 object tracking, interactive image/video segmentation, inpainting 같은 다른 pixel-level estimation 문제에도 확장 가능성이 있다. 즉, 이 논문은 단순히 하나의 segmentation 모델을 제안한 것이 아니라, “과거의 annotation-aware visual information을 memory로 축적하고 현재 예측에 활용한다”는 강한 패러다임을 제시한 연구라고 볼 수 있다.
