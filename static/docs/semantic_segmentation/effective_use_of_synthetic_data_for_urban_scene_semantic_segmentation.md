# Effective Use of Synthetic Data for Urban Scene Semantic Segmentation

- **저자**: Fatemeh Sadat Saleh, Mohammad Sadegh Aliakbarian, Mathieu Salzmann, Lars Petersson, Jose M. Alvarez
- **발표연도**: 2018
- **arXiv**: https://arxiv.org/abs/1807.06132

## 1. 논문 개요

이 논문은 synthetic data만으로 학습한 semantic segmentation 모델이 real image에서 성능이 크게 떨어지는 문제를 다룬다. 일반적으로 이 문제는 domain adaptation으로 완화할 수 있지만, 기존 방법들은 학습 단계에서 unlabeled real image라도 반드시 필요하다. 즉, synthetic data만으로 미리 학습해 둔 모델을 새로운 실제 환경에 바로 적용하기 어렵다.

저자들은 이 문제를 다른 각도에서 본다. 핵심 문제 설정은 단순히 "synthetic과 real 사이의 domain shift를 줄이는가"가 아니라, "모든 클래스가 domain shift의 영향을 같은 방식으로 받는가"이다. 논문은 foreground class와 background class가 synthetic-to-real 전이에서 서로 다른 방식으로 망가진다고 주장한다. background는 texture가 비교적 현실적이어서 일반적인 semantic segmentation이 잘 맞지만, foreground object는 texture는 부자연스럽더라도 shape는 비교적 자연스럽기 때문에 detection-based 접근이 더 적합하다는 것이다.

이 문제는 자율주행 같은 urban scene understanding에서 매우 중요하다. pixel-level annotation은 비용이 매우 크며, 예를 들어 Cityscapes 한 장의 정밀 annotation에는 평균 90분이 걸린다고 논문은 언급한다. 따라서 synthetic data를 효과적으로 활용할 수 있으면 데이터 구축 비용을 크게 줄일 수 있다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 semantic segmentation을 모든 클래스에 동일한 방식으로 적용하지 말고, 클래스의 성질에 따라 처리 방식을 분리하자는 것이다. 구체적으로는 background class는 semantic segmentation network로 처리하고, foreground class는 instance-level detection/segmentation model로 처리한 뒤, 두 결과를 결합한다.

이 설계의 직관은 다음과 같다. background class는 road, sky, building, vegetation처럼 주로 texture나 material cue로 구분되는 경우가 많다. 논문은 synthetic 이미지에서도 이런 background texture는 비교적 realistic하다고 본다. 반면 car, person, bus, bicycle 같은 foreground class는 synthetic texture가 실제와 많이 다르지만, object shape 자체는 충분히 자연스럽다. 현대 object detector는 region proposal과 object shape에 크게 의존하므로, 이런 foreground는 pixel-wise semantic segmentation보다 detection-based 접근이 더 robust하다고 본다.

기존 접근과의 가장 큰 차이는, real image를 학습 중에 보지 않아도 된다는 점이다. 기존 domain adaptation 계열은 synthetic와 real의 feature distribution을 맞추려 했지만, 이 논문은 애초에 class별 perceptual mismatch가 다르므로 모델의 역할 자체를 분리하는 것이 더 효과적이라고 본다. 즉, "도메인 차이를 없애는 것"보다 "도메인 차이의 성질에 맞는 표현을 쓰는 것"에 가깝다.

## 3. 상세 방법 설명

전체 파이프라인은 두 개의 모델로 구성된다. background 처리를 위한 DeepLab semantic segmentation network와, foreground 처리를 위한 Mask R-CNN 기반 instance segmentation network이다. 학습 시에는 두 모델 모두 synthetic data만 사용한다. 추론 시 real image를 두 모델에 각각 넣고, foreground prediction과 background prediction을 후처리로 합쳐 최종 semantic segmentation map을 만든다.

background branch에서는 VGG16 기반 DeepLab Large FOV with dilated convolutions를 사용한다. 이 모델은 GTA5 dataset으로 학습된다. GTA5를 선택한 이유는 Cityscapes 및 CamVid와 호환되는 class를 비교적 잘 포함하고 있고, background 표현이 photo-realistic하기 때문이다. 학습 목표는 표준 pixel-wise cross-entropy loss이다. 논문에서 중요한 점은 이 DeepLab이 모든 클래스를 예측하도록 학습되지만, 실제 최종 결과를 만들 때 foreground prediction은 대부분 버려진다는 점이다.

foreground branch에서는 Mask R-CNN을 사용한다. 저자들은 foreground object에 대해 detection-based instance-level segmentation이 더 적합하다고 본다. Mask R-CNN은 먼저 object detection을 수행하고, 이후 각 instance에 대해 binary mask와 class를 예측한다. 이 구조는 full-image semantic segmentation보다 object shape를 더 잘 활용한다. 학습에는 저자들이 새로 만든 VEIS dataset을 사용한다. VEIS는 Unity3D 기반 synthetic environment이며, foreground instance-level annotation을 자동으로 제공한다. Mask R-CNN은 논문에서 standard loss를 사용한다고 설명하며, 이 loss는 detection, segmentation, classification, bounding-box regression 항을 함께 포함한다. 다만 논문 본문에는 각 항의 수식이 명시적으로 적혀 있지 않으므로, 정확한 loss 식 자체는 본문에서 상세히 전개되지 않았다.

추론 단계의 결합 방식도 중요하다. 먼저 real image에서 Mask R-CNN이 여러 foreground instance mask를 예측한다. 이 예측들은 confidence score 기준으로 정렬된다. 그 뒤 높은 score부터 순차적으로 선택하면서, 이미 채워진 instance와 겹치는 영역은 제거한다. 이는 panoptic segmentation의 NMS-like procedure에서 영감을 받은 방식이다. 원래 panoptic segmentation 절차에는 score threshold와 너무 작은 non-overlapping region을 제거하는 threshold가 있지만, 논문은 real validation set에 맞춘 threshold tuning을 할 수 없으므로 이 두 휴리스틱을 쓰지 않는다. 즉, 별도의 real-image 기반 threshold selection 없이 모든 segment를 사용한다.

이 과정을 거치면 foreground class만 있는 segmentation map이 생기지만, foreground가 없는 곳은 hole로 남는다. 최종 semantic segmentation map은 이 hole을 DeepLab output으로 채워서 만든다. 수식 형태로 쓰면, 어떤 픽셀 $p$에 대해 Mask R-CNN이 foreground label을 부여하면 그 label을 사용하고, 그렇지 않으면 DeepLab의 가장 높은 확률 label을 사용한다는 규칙이다. 즉,
$$
\hat{y}(p)=
\begin{cases}
\hat{y}_{\text{Mask R-CNN}}(p), & \text{if } p \text{ is assigned to a foreground instance} \\
\arg\max_c P_{\text{DeepLab}}(c \mid p), & \text{otherwise}
\end{cases}
$$
와 같은 형태로 이해할 수 있다. 이 식은 논문의 개념을 설명하기 위한 정리이며, 본문에 동일한 수식으로 직접 제시되지는 않았다.

논문은 unsupervised real image를 사용할 수 있는 확장 버전도 제안한다. 먼저 위 방법으로 real image에 pseudo-label을 생성한다. 단, foreground hole 영역에서 DeepLab이 foreground로 예측한 픽셀은 신뢰하지 않기 때문에 `ignore` label로 둔다. 이는 semantic segmentation network가 foreground를 잘못 예측할 가능성이 높다는 앞선 논지를 그대로 반영한 것이다. 이후 이 pseudo-label을 ground truth처럼 사용해 real image 위에서 DeepLab을 다시 학습한다. 이 절차는 복잡한 adaptation objective 없이도 성능 향상을 만든다.

구현 세부사항을 보면, DeepLab 학습은 SGD를 사용하고 초기 learning rate는 $25 \times 10^{-5}$, 40k iteration마다 10배 감소, momentum은 $0.9$, weight decay는 $0.0005$, batch size는 1이다. VGG-16 ImageNet pretraining을 사용한다. Mask R-CNN은 Detectron 구현을 사용하고, backbone은 $64 \times 4d$ ResNeXt-101-FPN이다. 200k iteration 동안 학습하며 learning rate는 0.001에서 시작해 100k 이후 0.0001로 줄인다. 역시 batch size는 1이다.

## 4. 실험 및 결과

실험은 synthetic training set으로 GTA5, VEIS, 보조 비교용으로 SYNTHIA와 VIPER를 사용하고, real evaluation set으로 Cityscapes와 CamVid를 사용한다. Cityscapes에서는 500장의 validation image로 평가하며, pseudo-label 실험에서는 22,971장의 train/train-extra RGB 이미지를 annotation 없이 사용한다. CamVid에서는 11개 클래스를 사용하는 표준 split을 따른다.

비교 대상은 단순 DeepLab baseline들(GTA5, SYNTHIA, VIPER, VEIS, GTA5+VEIS), pseudo-label을 붙인 segmentation baseline, 그리고 제안 방법 `Ours`, `Ours+Pseudo-GT`이다. 또한 Cityscapes에서는 weakly supervised method와 여러 domain adaptation method(Fcns in the Wild, Curriculum, ROAD, CyCADA)와도 비교한다.

Cityscapes 결과를 보면, 저자들이 재구현한 GTA5 baseline은 mIoU 31.3이고, GTA5+VEIS는 32.8로 소폭 개선에 그친다. 반면 제안 방법 `Ours`는 38.0으로 크게 상승한다. 여기에 pseudo-label을 더한 `Ours&ps-GT`는 42.5까지 올라간다. 이 차이는 특히 foreground class에서 두드러진다. 예를 들어 traffic sign은 23.8에서 42.5로, rider는 13.3에서 28.5로, bus는 21.2에서 31.6으로, train은 2.1에서 6.9로, bicycle은 7.3에서 9.8로 좋아진다. pseudo-label까지 적용하면 car 83.0, truck 32.3, bus 41.3, train 27.0, bicycle 27.7 등 foreground 계열이 더 크게 상승한다.

논문의 초반 Fig. 1에 나온 비교도 핵심적이다. foreground class만 놓고 synthetic-only 학습 시 DeepLab과 detection-based model을 비교하면, 대부분 클래스에서 detection-based 접근이 더 높은 mIoU를 보인다. 예를 들어 traffic sign은 23.8 대 42.5, rider는 13.3 대 28.5, bus는 21.2 대 31.6으로 차이가 크다. 이는 foreground는 semantic segmentation보다 detector가 shape를 더 잘 활용한다는 논문의 주장을 정량적으로 뒷받침한다.

state-of-the-art와의 비교에서도 흥미로운 결과가 나온다. Cityscapes에서 제안 방법 `Ours`는 38.0 mIoU로 ROAD의 35.9, CyCADA의 35.4, Curriculum의 28.9, FCNs in the Wild의 27.1, weakly supervised method의 23.6을 넘는다. 중요한 점은 `Ours`는 학습 중 real image를 전혀 보지 않았다는 것이다. `Ours+Pseudo-GT`는 42.5로 더 높아지며, 이 경우는 unlabeled real image를 사용하지만 여전히 기존 adaptation 기법들보다 높다.

또한 fully supervised setting에서도 같은 아이디어가 유효함을 보인다. 일반 fully supervised model은 Cityscapes에서 56.2 mIoU인데, `Fully Sup. Ours`는 60.0을 달성한다. 즉, foreground는 detection-based, background는 segmentation-based로 나누는 설계 자체가 단순 synthetic-only 세팅에만 국한되지 않는다는 것이다.

CamVid에서도 경향은 비슷하다. GTA5 baseline은 44.6 mIoU, GTA5+VEIS는 44.4, `Ours`는 47.6, `Ours+Pseudo-GT`는 48.8이다. 특히 pedestrian, cyclist, sign 같은 foreground 관련 class에서 제안 방법이 상대적으로 유리하다. 논문은 이 결과를 통해, 제안 방법이 Cityscapes에 특화된 것이 아니라 다른 urban scene dataset에도 일반화된다고 주장한다.

논문은 shape vs. texture 가설을 별도 실험으로도 검증한다. 먼저 silhouette만 보고 real/synthetic를 구분하는 binary VGG-16 classifier를 학습했을 때 정확도는 70.0%로 높지 않았다. 반면 textured foreground object로 같은 실험을 하면 95.1%로 크게 높아졌다. 이는 texture가 domain identity를 더 강하게 드러낸다는 뜻이다. 또 synthetic silhouette로 foreground multi-class classifier를 학습해 real silhouette에 테스트하면 real 81.0%, synthetic validation 89.2%이고, textured version에서는 real 83.7%, synthetic 94.2%이다. textured case가 domain gap이 더 크므로, shape가 texture보다 domain shift에 더 robust하다는 논문의 핵심 논지를 지지한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제를 class-dependent domain shift 관점에서 재정의했다는 점이다. 많은 연구가 전체 feature distribution alignment에 집중한 반면, 이 논문은 foreground와 background가 서로 다른 cue에 의존한다는 점을 실험적으로 보여 주고, 그에 맞는 구조적 분리를 설계했다. 단순한 아이디어처럼 보이지만, 실제로 synthetic-only setting에서 domain adaptation보다 더 높은 성능을 얻었다는 점에서 설득력이 크다.

또 다른 강점은 제안 방법이 매우 실용적이라는 점이다. 기존 adaptation은 새로운 real domain마다 unlabeled real data를 모아 다시 학습해야 하지만, 이 방법은 기본적으로 synthetic-only pretraining으로 동작할 수 있다. 따라서 오프라인으로 준비한 synthetic dataset만으로 deployment 가능한 segmentation system을 만들려는 상황에 적합하다.

VEIS dataset도 의미 있는 기여다. 기존 synthetic dataset들은 instance-level annotation이 부족하거나 Cityscapes foreground class 전체를 포괄하지 못했는데, VEIS는 이를 자동으로 생성한다. 이미지의 photo-realism은 낮지만, 이 논문은 foreground detection에는 texture realism보다 shape realism이 더 중요하므로 이 약점을 상대적으로 덜 타는 방식으로 활용했다. 즉, dataset의 약점을 모델 설계로 상쇄한 점이 인상적이다.

반면 한계도 분명하다. 첫째, background 성능은 일부 class에서 여전히 불안정하다. 예를 들어 Cityscapes 결과에서 pole, wall, fence 같은 thin structure나 ambiguous class에서는 강하지 않다. foreground 개선이 전체 성능 향상을 이끌지만, background까지 전반적으로 강해졌다고 보기는 어렵다.

둘째, 결합 단계가 heuristic하다. Mask R-CNN instance를 confidence 순으로 합치고 hole을 DeepLab으로 채우는 방식은 간단하지만, joint optimization이 아니다. 두 branch가 end-to-end로 상호작용하지 않기 때문에 충돌 영역이나 confidence calibration 문제가 남을 수 있다. 논문도 panoptic-style fusion에서 원래 쓰이는 threshold tuning을 real image 없이 못 한다는 이유로 제거하는데, 이는 방법의 순수성은 높이지만 최적화 여지를 남긴다.

셋째, 논문의 핵심 가정은 urban driving scene의 `things` 대 `stuff` 구분에 강하게 기대고 있다. foreground는 shape, background는 texture라는 분리가 다른 도메인에서도 항상 유지된다고 말하기는 어렵다. 논문은 일반 semantic segmentation 문제로 확장 가능하다고 말하지만, 실제 실험은 urban scene에만 한정되어 있다.

넷째, pseudo-label 단계 역시 결국 DeepLab 하나를 다시 학습하는 방식이다. 즉, 최종 모델이 detection branch와 segmentation branch를 모두 유지하는지, 혹은 pseudo-label로 학습된 DeepLab 하나로 충분한지에 대한 operational trade-off는 본문에서 깊게 분석되지 않는다. 또한 pseudo-label의 noise에 대한 정량 분석도 제한적이다.

비판적으로 보면, 이 논문은 domain adaptation을 완전히 대체했다기보다, synthetic data의 class별 활용 방식을 더 정교하게 만든 접근이다. 따라서 texture transfer, adversarial adaptation, self-training 같은 기법과 결합되면 더 강해질 가능성이 크다. 저자들도 이를 future work로 언급한다.

## 6. 결론

이 논문은 synthetic-to-real semantic segmentation에서 모든 클래스를 동일하게 다루지 말고, background는 semantic segmentation, foreground는 detection-based instance segmentation으로 분리해 처리하자는 명확한 설계를 제안한다. 이 방식은 synthetic data만 사용한 학습에서도 기존 semantic segmentation baseline과 여러 domain adaptation 방법을 능가했고, unlabeled real image를 pseudo-label로 활용하면 성능을 더 끌어올릴 수 있었다.

핵심 기여는 세 가지로 요약할 수 있다. 첫째, foreground와 background가 domain shift의 영향을 다르게 받는다는 통찰을 제시했다. 둘째, 이를 반영한 practical한 fusion-based framework를 제안했다. 셋째, foreground instance annotation을 자동으로 생성할 수 있는 VEIS synthetic environment와 dataset을 구축했다.

실제 적용 관점에서 이 연구는 synthetic data를 단순히 "더 realistic하게 만드는 것"만이 답이 아니라, 어떤 visual cue가 domain shift에 robust한지를 보고 task decomposition을 설계하는 것이 중요하다는 점을 보여 준다. 향후 연구에서는 이 구조를 panoptic segmentation, self-training, domain adaptation, 또는 더 강력한 detector/segmenter와 결합하는 방향이 유망해 보인다.
