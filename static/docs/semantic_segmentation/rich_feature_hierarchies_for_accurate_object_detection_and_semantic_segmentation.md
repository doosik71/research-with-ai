# Rich feature hierarchies for accurate object detection and semantic segmentation

- **저자**: Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik
- **발표연도**: 2014
- **arXiv**: https://arxiv.org/abs/1311.2524

## 1. 논문 개요

이 논문은 당시 정체되어 있던 object detection 성능을 크게 끌어올리기 위해, 대규모 convolutional neural network(CNN)를 region proposal 기반 detection 파이프라인과 결합한 **R-CNN**을 제안한다. 핵심 문제의식은 분명하다. ImageNet classification에서는 CNN이 매우 강력한 표현력을 보여주었지만, 그 성능이 object detection에도 그대로 이어질지, 그리고 detection처럼 데이터가 적고 localization이 중요한 문제에서 어떻게 CNN을 제대로 사용할지 확실하지 않았다.

저자들은 이 간극을 두 가지 문제로 나누어 다룬다. 첫째, detection은 단순 분류가 아니라 이미지 안에서 객체의 위치를 찾아야 하므로, CNN을 어떻게 localization에 연결할 것인가의 문제다. 둘째, detection 데이터셋은 classification 데이터셋보다 훨씬 작기 때문에, 고용량 CNN을 직접 학습시키기 어렵다는 문제가 있다. 이 논문은 이 두 문제를 동시에 해결하면서, PASCAL VOC와 ILSVRC detection에서 당시 기준으로 매우 큰 성능 향상을 달성했다.

문제의 중요성은 매우 크다. 이 논문 이전의 강한 detection 시스템들은 SIFT, HOG, spatial pyramid, context rescoring, ensemble 등 복잡한 조합에 의존했다. 반면 R-CNN은 region proposal과 CNN feature라는 비교적 단순한 구조로 더 높은 성능을 냈다. 즉, 이 논문은 “hand-crafted feature 중심 detection”에서 “deep feature 중심 detection”으로 패러다임을 전환한 대표적 작업이다.

## 2. 핵심 아이디어

![Figure 1: Object detection system overview.](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x1.png)

논문의 중심 아이디어는 다음 두 가지다.

첫째, **bottom-up region proposal**을 사용해 이미지에서 약 2,000개의 후보 영역을 먼저 뽑고, 각 영역을 CNN에 넣어 feature를 추출한 뒤 분류하는 방식이다. 이는 sliding window로 이미지 전체를 촘촘히 훑는 대신, 이미 객체일 가능성이 높은 후보들만 분석하는 접근이다. 저자들은 깊은 CNN의 receptive field와 stride가 커서 sliding-window 방식으로 정밀 localization을 하기 어렵다고 보고, region-based recognition 패러다임을 택했다.

둘째, **supervised pre-training 후 domain-specific fine-tuning**이다. 먼저 ILSVRC2012 classification 데이터로 CNN을 학습한 뒤, detection용 warped proposal window들로 다시 fine-tuning한다. 저자들은 detection 데이터가 적을 때 unsupervised pre-training보다 이런 supervised pre-training이 매우 효과적이라고 주장하며, 실험으로 이를 뒷받침한다.

기존 접근과의 차별점은 분명하다. 이전 강한 방법들은 주로 HOG, SIFT, bag-of-words, spatial pyramid 같은 수작업 feature와 복잡한 classifier 조합에 의존했다. R-CNN은 이들을 대체하는 고차원 의미 표현을 CNN에서 직접 얻는다. 또한 OverFeat처럼 CNN을 쓰더라도 sliding-window 기반인 방법과 달리, R-CNN은 region proposal 기반이라 localization과 detection accuracy에서 큰 이점을 보였다.

## 3. 상세 방법 설명

전체 detection 시스템은 세 모듈로 이루어진다. 첫 번째는 **category-independent region proposal 생성기**, 두 번째는 **각 proposal에서 feature를 뽑는 CNN**, 세 번째는 **class-specific linear SVM**이다.

테스트 시 파이프라인은 다음과 같다. 입력 이미지를 받으면 selective search를 사용해 약 2,000개의 region proposal을 만든다. 각 proposal은 원래 모양과 종횡비가 제각각이므로, proposal을 둘러싼 bounding box를 기준으로 잘라낸 뒤 CNN 입력 크기인 227 &times; 227로 **warping**한다. 이때 원래 박스 주변에 문맥 정보를 주기 위해 padding $p=16$ 픽셀의 context를 포함한다. 이렇게 만든 RGB patch에서 평균값을 빼고 CNN에 통과시키면 4,096차원 feature가 나온다. 저자들은 주로 Krizhevsky et al.의 네트워크를 사용했고, convolution layer 5개와 fully connected layer 2개를 지난 feature를 사용한다. 이후 클래스별 linear SVM이 각 proposal에 대해 점수를 계산하고, 마지막으로 class-wise greedy non-maximum suppression(NMS)을 적용해 중복 검출을 제거한다.

![Figure 2: Warped training samples from VOC 2007 train.](https://ar5iv.labs.arxiv.org/html/1311.2524/assets/x2.png)

CNN 입력 변환 방식은 논문에서 중요한 설계 포인트다. 임의 크기와 모양의 proposal을 CNN에 넣기 위해 가장 단순한 방법인 anisotropic warping을 택했다. appendix에서 tight square 변환 등 다른 방식과 비교했지만, context padding을 둔 warping이 3에서 5 mAP 정도 더 좋았다고 보고한다.

학습은 세 단계다.

첫 단계는 **supervised pre-training**이다. CNN을 ILSVRC2012 classification 데이터에서 image-level annotation만 사용해 미리 학습한다. detection bounding box는 이 단계에서 사용하지 않는다.

두 번째는 **fine-tuning**이다. detection task와 warped proposal 도메인에 맞게 CNN을 다시 학습한다. 기존 1000-way classification layer를 $(N+1)$-way layer로 바꾸는데, 여기서 $N$은 object class 수이고 추가 1개는 background다. VOC에서는 $N=20$, ILSVRC2013 detection에서는 $N=200$이다. proposal이 어떤 ground-truth box와 IoU가 0.5 이상이면 그 클래스의 positive로 간주하고, 나머지는 background로 둔다. SGD는 learning rate 0.001로 시작하며, mini-batch 크기 128 중 positive 32개, background 96개를 샘플링한다. positive가 매우 희귀하기 때문에 의도적으로 positive 비중을 높였다.

세 번째는 **class-specific linear SVM 학습**이다. 흥미로운 점은 fine-tuning과 SVM 학습에서 positive/negative 정의가 다르다는 것이다. SVM 학습에서는 positive를 해당 클래스의 **ground-truth bounding box 자체**로 두고, 어떤 클래스의 모든 ground-truth와 IoU가 0.3 미만인 proposal만 negative로 쓴다. IoU가 애매한 중간 영역은 무시한다. 논문은 이 threshold 0.3이 validation에서 가장 좋았고, 0.5로 두면 mAP가 5포인트, 0으로 두면 4포인트 떨어졌다고 보고한다. 학습 데이터가 메모리에 한 번에 안 들어가기 때문에 hard negative mining을 사용한다.

저자들은 왜 fine-tuned softmax 출력을 그대로 detector로 쓰지 않고 SVM을 따로 학습하는지도 실험했다. softmax만 쓰면 VOC 2007에서 mAP가 54.2%에서 50.9%로 떨어졌다. 논문은 그 이유로 fine-tuning 시 positive 정의가 localization을 충분히 강조하지 못한 점과, softmax가 hard negative가 아니라 무작위 negative로 학습된 점을 든다.

Bounding-box regression도 중요한 구성 요소다. error analysis 결과 mislocalization이 주요 오류였기 때문에, proposal box를 더 정확한 box로 보정하는 class-specific linear regressor를 추가했다. proposal box를 $P=(P_x, P_y, P_w, P_h)$, target ground-truth를 $G=(G_x, G_y, G_w, G_h)$라고 하면, 네 개의 변환 함수 $d_x(P), d_y(P), d_w(P), d_h(P)$를 학습한다. 예측 box $\hat{G}$는 다음처럼 계산된다.

$$
\hat{G}_x = P_w d_x(P) + P_x
$$

$$
\hat{G}_y = P_h d_y(P) + P_y
$$

$$
\hat{G}_w = P_w \exp(d_w(P))
$$

$$
\hat{G}_h = P_h \exp(d_h(P))
$$

각 함수는 pool5 feature $\phi_5(P)$의 선형 함수로 모델링된다.

$$
d_{?}(P) = w_{?}^{T}\phi_5(P)
$$

여기서 $?$는 $x, y, w, h$ 중 하나다. 학습은 ridge regression으로 한다.

$$
w_{?} = \arg\min_{\hat{w}_{?}} \sum_{i=1}^{N}\left(t^{i}_{?} - \hat{w}_{?}^{T}\phi_5(P^i)\right)^2 + \lambda \|\hat{w}_{?}\|^2
$$

회귀 타깃은 proposal 대비 ground-truth의 상대 이동과 크기 변화를 표현한다.

$$
t_x = (G_x - P_x)/P_w,\quad
t_y = (G_y - P_y)/P_h
$$

$$
t_w = \log(G_w/P_w),\quad
t_h = \log(G_h/P_h)
$$

즉, 중심 이동은 scale-invariant translation으로, 크기 변화는 log-space scale 변화로 다룬다. 회귀는 아무 proposal에나 적용하지 않고, 어떤 ground-truth와 IoU가 0.6보다 큰 proposal에 대해서만 학습한다. 너무 멀리 있는 proposal은 변환 학습 자체가 무의미하다고 본 것이다.

논문은 semantic segmentation에도 같은 철학을 확장한다. CPMC region proposal 위에서 CNN feature를 뽑고, SVR로 region quality를 예측한다. 여기서 세 가지 feature 추출 전략을 비교한다. `full`은 region bounding box 전체를 그대로 CNN에 넣는 방식이고, `fg`는 region foreground mask만 남기고 배경은 mean으로 채워서 CNN에 넣는 방식이다. `full+fg`는 이 둘을 이어붙인다. 결과적으로 문맥을 담는 `full`과 shape 중심 정보인 `fg`가 상보적이라 `full+fg`가 가장 좋았다.

## 4. 실험 및 결과

논문은 PASCAL VOC 2007, 2010, 2011/2012와 ILSVRC2013 detection, 그리고 VOC 2011 segmentation에서 결과를 제시한다.

PASCAL VOC detection에서 가장 대표적인 결과는 VOC 2010 test다. R-CNN은 bounding-box regression 없이 **50.2% mAP**, bounding-box regression을 포함하면 **53.7% mAP**를 달성했다. 비교 대상으로 제시된 DPM v5는 33.4%, UVA는 35.1%, Regionlets는 39.7%, SegDPM은 40.4%였다. 즉 selective search proposal을 동일하게 쓰는 UVA 대비도 매우 큰 폭으로 앞선다. 논문 초반 abstract에서는 VOC 2012에서 이전 최고 성능 대비 30% 이상 상대 향상을 이루며 **53.3% mAP**를 달성했다고 요약한다.

VOC 2007 ablation도 중요하다. fine-tuning 없이 ImageNet-pretrained CNN feature만 써도 `pool5`, `fc6`, `fc7`에서 각각 44.2%, 46.2%, 44.7% mAP를 기록한다. fine-tuning을 하면 성능이 크게 올라 `pool5` 47.3%, `fc6` 53.1%, `fc7` 54.2%가 된다. 여기서 bounding-box regression을 더하면 **58.5% mAP**까지 오른다. 논문은 fine-tuning 자체가 약 **8.0 percentage points**의 이득을 준다고 강조한다.

layer별 분석에서는 `fc7`이 pre-training 상태에서는 `fc6`보다 generalization이 떨어졌지만, fine-tuning 후에는 `fc7`이 가장 좋았다. 이는 convolutional feature 자체는 꽤 일반적이며, detection에 맞춘 상위 non-linear classifier를 학습하는 것이 큰 효과를 낸다는 해석과 연결된다.

네트워크 구조 변화의 영향도 분석한다. 기존 AlexNet 계열(T-Net) 대신 Simonyan and Zisserman의 16-layer VGG 계열(O-Net)을 쓰면 VOC 2007 test에서 **58.5%에서 66.0% mAP**로 크게 오른다. 다만 forward pass 시간이 약 7배 느려진다. 즉 더 깊은 네트워크가 detection 성능을 상당히 밀어 올리지만 계산 비용이 크다.

ILSVRC2013 detection에서도 R-CNN은 강력했다. test set에서 **31.4% mAP**를 달성해 OverFeat의 **24.3% mAP**를 크게 앞섰다. val2 ablation에서는 ImageNet-pretrained feature와 val1만으로 20.9%, train positive를 추가하면 24.1%, fine-tuning을 val1에만 하면 26.5%, val1 + train 1k로 fine-tuning하면 29.7%, bounding-box regression까지 넣으면 31.0%가 된다. test set 최종 결과 31.4%는 val2 경향과 잘 맞아, validation 설계가 타당했음을 보여준다.

ILSVRC 데이터셋에 대해서는 저자들이 꽤 신중하게 설명한다. train split은 exhaustive annotation이 아니므로 hard negative mining에 쓰기 어렵고, val/test는 exhaustive하게 annotation되어 있다. 그래서 val을 `val1`과 `val2`로 class-balance를 고려해 나눈 뒤, `val1`을 주된 training과 hard negative mining에 쓰고 `val2`로 검증했다. 이 설정은 논문이 단순히 숫자만 보고한 것이 아니라, 데이터셋 특성에 맞춰 실제 학습 전략을 조정했음을 보여준다.

region proposal recall도 보고한다. ILSVRC val에서 selective search는 이미지당 평균 2403 proposal을 만들고, IoU 0.5 기준 ground-truth recall이 **91.6%**였다. PASCAL의 약 98%보다 낮아서, region proposal 단계가 여전히 병목일 수 있음을 시사한다.

semantic segmentation에서는 VOC 2011 validation에서 `full+fg R-CNN fc6`가 **47.9% mean accuracy**를 달성했다. 이는 O2P의 46.4%보다 높다. VOC 2011 test에서도 제안 방법은 **47.9%**로, O2P의 47.6%와 비슷하거나 약간 높은 수준이다. 특히 21개 category 중 11개에서 최고 성능을 기록했다. 중요한 점은 detection용으로 pre-trained된 CNN을 segmentation에 거의 직접 확장해도 강한 결과가 나온다는 사실이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은, 복잡한 hand-crafted detection pipeline을 대체할 수 있는 단순하고 강력한 구조를 제시했다는 점이다. selective search, CNN feature, linear SVM, NMS라는 비교적 직관적인 조합으로 큰 성능 향상을 이뤘고, 실험적으로도 VOC와 ILSVRC 모두에서 우수함을 입증했다. 또한 fine-tuning, layer ablation, architecture 비교, error analysis, bounding-box regression, segmentation 확장까지 포함해, 단순 제안에 그치지 않고 왜 잘 되는지 체계적으로 보여준다.

또 다른 강점은 **supervised pre-training + task-specific fine-tuning**이라는 학습 패러다임을 분명하게 정식화했다는 점이다. 오늘날 transfer learning의 표준으로 받아들여지는 방식이지만, 당시에는 detection에서 이것이 얼마나 효과적인지 실증하는 것이 중요했다. 이 논문은 적은 detection 데이터에서도 대규모 CNN을 활용할 수 있음을 보여주었다.

error analysis 결과도 설득력이 있다. R-CNN의 오류는 DPM보다 background confusion이나 category confusion보다 **poor localization** 비중이 더 컸다. 이는 CNN feature 자체의 분별력이 매우 좋다는 뜻이며, 성능 병목이 feature discrimination보다 localization 쪽에 있음을 드러낸다. 그래서 bounding-box regression을 붙였을 때 3에서 4 mAP 향상이 나는 것도 자연스럽다.

하지만 한계도 명확하다. 가장 큰 문제는 **속도와 중복 계산**이다. 논문에 따르면 feature 계산과 proposal 생성에 GPU 기준 약 13초/이미지, CPU 기준 약 53초/이미지가 걸린다. 각 proposal을 개별적으로 warp하고 CNN을 별도로 통과시키기 때문에 계산 공유가 거의 없다. OverFeat보다 약 9배 느리다는 비교도 제시된다. 즉 정확도는 높지만 실용 시스템으로서는 아직 매우 느리다.

또한 성능이 selective search와 같은 외부 region proposal 품질에 크게 의존한다. ILSVRC에서 recall이 91.6%라는 점은 proposal 단계에서 이미 놓치는 객체가 적지 않다는 뜻이다. CNN detector가 아무리 좋아도 proposal이 객체를 제안하지 못하면 검출할 수 없다.

학습 절차도 다소 복잡하다. CNN fine-tuning, feature extraction, class-wise SVM 학습, hard negative mining, bounding-box regression을 각각 수행해야 한다. 논문도 appendix에서 softmax detector만으로 비슷한 성능에 근접할 수 있다고 언급하지만, 당시 설정에서는 SVM이 더 좋았다. 이는 end-to-end성 측면에서 불완전함을 보여준다.

localization 관점에서도, fine-tuning 시 positive를 IoU 0.5 이상 proposal 전체로 넣는 전략은 positive 수를 늘려 과적합을 줄이는 장점이 있지만, precise localization 자체를 충분히 학습시키지는 못한다고 저자들도 인정한다. 이 점이 후속 연구들에서 Fast R-CNN, Faster R-CNN처럼 detection head와 box regression을 더 통합적으로 학습하는 방향으로 발전하게 된 배경이라고 볼 수 있다. 다만 이 마지막 연결은 본 논문 바깥의 역사적 맥락이며, 논문 본문은 그 자체의 문제 제기에만 머문다.

## 6. 결론

이 논문은 region proposal 기반 object detection에 deep CNN feature를 본격적으로 도입해, 당시 detection 성능 정체를 크게 돌파한 연구다. 핵심 기여는 세 가지로 요약할 수 있다. 첫째, selective search와 CNN을 결합한 R-CNN 파이프라인을 제안했다. 둘째, supervised pre-training 후 domain-specific fine-tuning이 detection처럼 데이터가 적은 문제에서 매우 효과적임을 보였다. 셋째, bounding-box regression과 segmentation 확장을 통해 제안 방식의 일반성과 실용 가능성을 입증했다.

실제로 이 연구는 이후 object detection 연구의 표준 방향을 정한 작업으로 볼 수 있다. proposal 기반 검출, transfer learning, deep feature 중심 인식, localization refinement라는 핵심 축이 모두 이 논문 안에 들어 있다. 속도와 파이프라인 복잡성이라는 한계는 남아 있었지만, 그 한계 자체가 이후 더 빠르고 end-to-end한 detector 연구를 촉진했다는 점에서, 이 논문은 단순히 좋은 성능을 낸 방법을 넘어 현대 detection 계열의 출발점 중 하나로 평가할 수 있다.
