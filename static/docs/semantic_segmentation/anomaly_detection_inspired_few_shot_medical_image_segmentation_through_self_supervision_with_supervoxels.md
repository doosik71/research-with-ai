# Anomaly Detection-Inspired Few-Shot Medical Image Segmentation Through Self-Supervision With Supervoxels

- **저자**: Stine Hansen, Srishti Gautam, Robert Jenssen, Michael Kampffmeyer
- **발표연도**: 2022
- **arXiv**: https://arxiv.org/abs/2203.02048

## 1. 논문 개요

이 논문은 **few-shot medical image segmentation** 문제를 다룬다. 목표는 새로운 장기(class)에 대해 많은 주석 데이터를 다시 모으지 않고도, 소수의 라벨된 support slice만으로 query 의료영상을 분할하는 것이다. 특히 기존 few-shot segmentation(FSS) 방법들이 의료영상에서 잘 작동하기 어려운 이유를 짚고, 이를 해결하기 위해 **background를 직접 모델링하지 않는 anomaly detection 관점의 분할 방식**을 제안한다.

연구 문제의 핵심은 다음과 같다. 기존 prototype 기반 FSS는 support set에서 foreground와 background의 prototype을 만들고, query pixel을 이들과 비교해 분할한다. 그런데 의료영상의 background는 매우 크고, 장기 외의 다양한 해부학적 구조를 포함하며, 공간적으로도 이질적이다. 따라서 support slice 몇 장만으로 background 전체를 잘 대표하는 prototype을 만드는 것은 어렵다. 논문은 바로 이 점이 기존 방법의 구조적 약점이라고 본다.

이 문제가 중요한 이유는 명확하다. 의료영상 분할은 진단, 치료 계획, 장기 부피 측정 등 실제 임상 작업의 핵심이며, 매번 대규모 dense annotation을 요구하는 방식은 현실적으로 부담이 크다. 또한 새로운 장기나 새로운 데이터셋에 적응할 때마다 다시 fully supervised training을 해야 한다면 활용성이 떨어진다. 따라서 적은 라벨만으로 새로운 클래스를 분할할 수 있는 FSS는 의료영상 분야에서 매우 실용적인 방향이다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 단순하지만 분명하다. **background를 prototype으로 명시적으로 모델링하려 하지 말고, 상대적으로 더 homogeneous한 foreground만 하나의 prototype으로 모델링한 뒤, query의 각 픽셀이 foreground prototype에서 얼마나 벗어나는지를 anomaly score로 계산하자**는 것이다.

즉 기존 방식은 “foreground prototype과 background prototype 중 어디에 더 가까운가?”를 묻는다. 반면 이 논문은 “이 픽셀이 foreground prototype과 충분히 비슷한가, 아니면 foreground로 보기 어려운 anomaly인가?”를 묻는다. 이 발상 전환 덕분에 복잡한 background 표현을 억지로 support set에서 추정할 필요가 없다.

또 하나의 핵심 기여는 self-supervision 설계다. 기존 Ouyang et al. (2020)은 2D superpixel을 이용해 pseudo-task를 만들었는데, 이 논문은 이를 **3D supervoxel 기반 self-supervision**으로 확장한다. 의료영상은 본질적으로 volumetric data이므로, 서로 다른 slice 사이의 구조적 연관성을 학습에 활용하는 것이 더 자연스럽다. 저자들은 이 확장이 2D 설정 안에서도 이득을 주고, 더 나아가 3D CNN으로 직접 volume segmentation을 수행할 가능성도 열어준다고 주장한다.

기존 접근과의 차별점은 세 가지로 정리된다. 첫째, background prototype을 제거했다. 둘째, superpixel이 아니라 supervoxel을 사용해 self-supervision을 3D로 확장했다. 셋째, query volume 내 장기 위치에 대한 weak label을 요구하지 않는 새로운 evaluation protocol을 제안했다.

## 3. 상세 방법 설명

전체 구조는 prototype-based metric learning 계열과 유사하다. support image와 query image를 같은 encoder $f_\theta$로 임베딩하여 feature map을 만든다. 차이는 그 다음 단계에 있다. support의 foreground mask만을 사용해 **foreground prototype 하나만** 만든다.

support feature map을 $F_s$, query feature map을 $F_q$라고 할 때, foreground prototype $p \in \mathbb{R}^d$는 masked average pooling으로 계산된다.

$$
p = \frac{\sum_{x,y} F_s(x,y)\, y_{fg}(x,y)}{\sum_{x,y} y_{fg}(x,y)}
$$

여기서 $y_{fg} = 1(y=c)$는 현재 episode의 foreground class에 대한 binary mask이다. 중요한 점은 background prototype은 만들지 않는다는 것이다.

이후 query의 각 위치 $(x,y)$에 대해 foreground prototype과의 cosine similarity를 계산하고, 이를 음수로 뒤집어 **anomaly score**로 정의한다.

$$
S(x,y) = -\alpha \frac{F_q(x,y)\cdot p}{\|F_q(x,y)\|\|p\|}
$$

논문에서는 $\alpha = 20$을 사용한다. query feature가 prototype과 매우 비슷하면 cosine similarity가 커지고 anomaly score는 작아진다. 반대로 prototype과 다르면 anomaly score가 커진다. 즉 anomaly score가 낮을수록 foreground일 가능성이 높다.

이 score를 그대로 hard thresholding하면 미분이 어려우므로, 저자들은 학습 가능한 threshold $T$와 shifted sigmoid를 사용해 soft thresholding을 수행한다.

$$
\hat y^q_{fg}(x,y) = 1 - \sigma(S(x,y)-T)
$$

여기서 $\sigma(\cdot)$는 sigmoid이고 steepness parameter는 $\kappa = 0.5$이다. 따라서 $S(x,y) < T$이면 foreground probability가 $0.5$보다 커지고, $S(x,y) > T$이면 background 쪽으로 분류된다. background probability는 단순히

$$
\hat y^q_{bg} = 1 - \hat y^q_{fg}
$$

로 둔다.

기본 segmentation loss는 binary cross-entropy이다.

$$
L_S = -\frac{1}{HW}\sum_{x,y}\Big(
y^q_{bg}(x,y)\log \hat y^q_{bg}(x,y) +
y^q_{fg}(x,y)\log \hat y^q_{fg}(x,y)
\Big)
$$

그런데 저자들은 이것만으로는 foreground embedding이 충분히 compact해지지 않는다고 보고, threshold를 작게 유지하도록 유도하는 추가 항

$$
L_T = T/\alpha
$$

를 넣는다. 이 항은 learned threshold를 낮추어 foreground cluster를 더 조밀하게 만들려는 목적이다. 논문 내 qualitative 결과에서도 이 항이 over-segmentation을 줄이는 데 도움이 된다고 보고한다.

또한 기존 prototype 계열 논문들처럼 **prototype alignment regularization**도 사용한다. 이는 query의 예측 mask로 prototype을 다시 만들고, 그 prototype으로 support를 재분할하게 하여 support-query 표현 정합성을 높이는 방식이다. 이 손실은 $L_{PAR}$로 표기되며, 역시 binary cross-entropy 형태다.

최종 손실은

$$
L = L_S + L_T + L_{PAR}
$$

이다.

학습은 fully supervised가 아니라 **self-supervised episodic training**으로 이루어진다. 각 unlabeled volume에 대해 미리 3D supervoxel segmentation을 만든다. 그리고 매 episode마다 하나의 supervoxel을 무작위로 골라 pseudo foreground로 사용한다. 그 supervoxel을 포함하는 두 개의 2D slice를 뽑아 support와 query로 삼는다. 이렇게 하면 실제 organ label 없이도 “같은 구조 조각을 다른 slice에서 찾는” 유사 few-shot task를 만들 수 있다. 추가로 support 또는 query에 random transformation을 적용해 shape와 intensity 변화에 대한 불변성을 키운다.

supervoxel은 Felzenszwalb and Huttenlocher (2004)의 graph-based segmentation을 3D로 확장해 offline으로 생성한다. 의료영상은 $z$ 방향 해상도가 $x,y$ 평면과 다르므로, 저자들은 anisotropic voxel spacing을 고려해 $z$ 방향 거리들을 재가중한다. supervoxel 크기를 조절하는 핵심 하이퍼파라미터는 $\rho$이며, 값이 클수록 더 크고 적은 supervoxel이 만들어진다.

구현 측면에서 2D 실험은 ResNet-101 encoder를 사용하고, MS-COCO pretrained weight를 이용한다. 마지막 classifier 대신 $1\times1$ convolution으로 feature dimension을 2048에서 256으로 줄인다. optimizer는 SGD with momentum 0.9, learning rate는 $10^{-3}$, decay rate는 1k epoch마다 0.98, weight decay는 $5\times10^{-4}$이다. foreground/background class imbalance를 줄이기 위해 BCE에서 foreground 1.0, background 0.1 가중치를 사용한다.

## 4. 실험 및 결과

실험은 두 개의 대표적인 MRI 데이터셋에서 수행된다. 첫째는 심장 MRI인 **MS-CMRSeg**이고, 둘째는 복부 MRI인 **CHAOS**이다. 각각 cardiac segmentation과 abdominal organ segmentation을 평가한다. 심장 데이터에서는 LV-BP, LV-MYO, RV 세 클래스를, 복부 데이터에서는 left kidney, right kidney, liver, spleen 네 클래스를 각각 binary segmentation 문제로 따로 다룬다.

평가 지표는 mean Dice score이며,

$$
D(A,B) = \frac{2|A\cap B|}{|A|+|B|}\cdot 100\%
$$

를 사용한다. 실험은 5-fold cross-validation으로 수행되고, 각 fold마다 3회 반복하여 평균과 표준편차를 보고한다.

논문은 두 가지 evaluation protocol을 비교한다. **EP1**은 기존 연구들이 쓰던 방식으로, query volume에서 target organ이 어디 slice들에 존재하는지 weak label 정보가 필요하다. support와 query를 organ이 있는 구간만 잘라 3개 sub-chunk로 나누고, support의 가운데 slice들로 query의 대응 구간을 분할한다. 실전에서는 query volume마다 장기 위치를 사람이 표시해야 하므로 번거롭다. 반면 **EP2**는 support volume의 가운데 slice 하나만 라벨링하고, 그것으로 query volume 전체 slice를 모두 분할한다. 논문은 EP2가 더 현실적인 설정이라고 본다.

결과를 보면, EP1에서는 제안 방법 vSSL-ADNet이 기존 state-of-the-art인 pSSL-ALPNet과 비슷하거나 약간 더 나은 수준을 보인다. Cardiac MRI 평균 Dice는 pSSL-ALPNet이 74.60, vSSL-ADNet이 75.76이며, Abdominal MRI 평균 Dice는 각각 78.46과 78.82이다. 중요한 점은 ADNet이 훨씬 적은 prototype만 사용한다는 것이다. 표 3에 따르면 ALPNet은 foreground prototype 약 4개, background prototype 약 246개를 쓰는 반면, ADNet은 foreground 1개, background 0개다.

더 중요한 결과는 EP2에서 나온다. 현실적인 설정인 EP2에서 vSSL-ADNet은 기존 방법들을 뚜렷하게 앞선다. Cardiac MRI 평균 Dice는 69.62로 pSSL-ALPNet의 67.74보다 높고, Abdominal MRI에서는 72.41로 pSSL-ALPNet의 52.05를 크게 앞선다. 특히 복부 데이터셋에서 20 percentage points 이상 향상된 것은 background가 훨씬 크고 다양해지는 전체-volume segmentation 상황에서 background prototype 기반 방법이 크게 흔들린다는 저자들의 주장과 잘 맞는다. 논문은 Wilcoxon signed-rank test를 통해 이 차이가 통계적으로 유의미하다고 보고한다($p < 0.05$).

정성적 결과에서도 경향은 같다. 제안 방법은 support slice 밖에서 등장하는 다양한 background에 덜 민감하고, organ이 없는 slice에서의 **over-segmentation**이 적다. 특히 복부 MRI 예시에서 기존 방법은 organ이 없는 slice까지 foreground로 잘못 칠하는 경우가 많은데, ADNet은 이를 더 잘 억제한다.

추가 분석도 흥미롭다. threshold precision 분석에서는 학습된 threshold가 테스트셋에서 거의 최적에 가까운 값을 보였고, 별도의 line search로 크게 더 개선되지 않았다. ablation study에서는 $L_T$와 $L_{PAR}$가 모두 Dice score 향상에 기여했다. 예를 들어 cardiac MRI에서 세 손실을 모두 쓸 때 평균 Dice는 75.76이고, $L_T$와 $L_{PAR}$를 제거하거나 일부 제거하면 성능이 점진적으로 떨어진다.

supervoxel 크기 민감도 분석에서는 $\rho=1000$에서 가장 좋은 cardiac 성능을 보였고, 너무 작은 supervoxel($\rho=500$)이나 너무 큰 supervoxel($\rho=5000$)은 성능을 떨어뜨렸다. 이는 pseudo-label이 지나치게 세분되거나 반대로 너무 크게 병합되면 self-supervision 품질이 떨어짐을 시사한다.

sigmoid steepness parameter $\kappa$에 대한 분석에서는 $\kappa=0.5$ 근처가 좋았고, 너무 작은 값인 0.1에서는 성능이 크게 저하되었다. 저자들은 전체적으로 이 파라미터에 대해 비교적 robust하다고 본다.

또한 pSSL과 vSSL을 직접 비교한 실험에서, supervoxel 기반 self-supervision(vSSL)은 ADNet과 ALPNet 모두에 대해 대체로 더 좋거나 비슷한 결과를 냈다. 특히 ADNet에서는 cardiac와 abdominal 모두에서 pSSL 대비 유의미한 향상이 보고되었다.

마지막으로 저자들은 3D CNN 기반 직접 volume segmentation 가능성도 살펴본다. 3D ResNeXt-101 backbone으로 실험한 결과, cardiac에서는 2D와 큰 차이가 없었지만, abdominal에서는 3D가 더 유망했다. 특히 support volume의 모든 slice를 라벨링할 수 있는 설정에서는 abdominal 평균 Dice가 79.58까지 올라가 2D backbone의 72.41보다 좋았다. 다만 이 부분은 확장 가능성을 보이는 수준이지, 본 논문의 주된 비교 실험 축은 2D slice-wise setting이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 정의가 명확하고, 제안 방법이 그 문제를 직접 겨냥한다는 점이다. 의료영상 few-shot segmentation에서 background가 구조적으로 어렵다는 관찰은 설득력이 크고, 이를 해결하기 위해 background modeling 자체를 버리는 설계는 개념적으로 깔끔하다. 또한 결과도 그 주장과 일관된다. 특히 weak query labels 없이 query volume 전체를 분할하는 EP2에서 성능 향상이 크게 나타난 것은 제안 방법의 장점을 잘 보여준다.

또 다른 강점은 self-supervision 설계가 의료영상의 3D 특성을 반영한다는 점이다. 기존 2D superpixel 기반 접근보다 한 단계 자연스럽고, 실제로 abdominal dataset처럼 slice 수가 많고 3D 문맥이 중요한 경우 더 큰 이득을 보였다. 또한 3D CNN으로의 확장 가능성까지 실험적으로 보여준 점도 의미가 있다.

실험 설계 측면에서도 긍정적이다. 단순히 기존 프로토콜만 따르지 않고, 더 현실적인 EP2를 제안해 실제 활용 조건에 가까운 평가를 제공했다. 이는 방법론 자체뿐 아니라 평가 관행에 대한 비판과 개선 제안이라는 점에서 가치가 있다.

한계도 논문이 스스로 비교적 솔직하게 인정한다. 가장 중요한 가정은 **foreground가 상대적으로 homogeneous하다**는 것이다. 만약 하나의 foreground class가 실제로는 서로 다른 appearance를 갖는 여러 sub-region으로 이루어져 있다면, single prototype만으로는 충분하지 않을 수 있다. 논문은 left ventricle blood pool과 myocardium을 묶은 larger class 같은 경우를 예로 든다. 이런 경우에는 foreground에도 multiple prototype이 필요할 수 있다.

또한 self-supervision의 근간이 되는 superpixel/supervoxel pseudo-label 자체가 noisy하다. 경계가 약한 두 구조가 하나의 supervoxel로 병합되면, 학습 과정에서 서로 다른 해부학 구조를 같은 embedding cluster로 밀어넣게 된다. 논문은 CHAOS에서 left kidney와 spleen이 이 문제로 자주 혼동된다고 설명한다. 따라서 pseudo-label noise를 명시적으로 다루는 방법은 향후 중요한 연구 방향이다.

비판적으로 보면, 제안 방식은 background modeling의 어려움을 피해가는 대신 foreground compactness에 크게 의존한다. 따라서 foreground 내부 분산이 큰 문제나, 병변처럼 appearance 변화가 심한 object에 바로 일반화될지는 논문만으로는 판단하기 어렵다. 또 multi-class joint segmentation으로 확장 가능한지에 대해서도 저자들은 가능성을 언급하지만, 실제 실험은 모두 one-class-at-a-time binary setting이다. 따라서 다중 클래스 동시 분할에서의 동작은 이 논문이 직접 입증하지 않는다.

## 6. 결론

이 논문은 의료영상 few-shot segmentation에서 background prototype modeling이 근본적으로 불안정하다는 점을 짚고, 이를 해결하기 위해 **foreground prototype 하나와 anomaly score thresholding만으로 분할하는 ADNet**을 제안했다. 여기에 3D supervoxel 기반 self-supervision을 결합해 annotation 없이도 학습할 수 있도록 했고, 특히 더 현실적인 평가 설정에서 기존 방법보다 강한 성능과 강건성을 보였다.

핵심 기여는 세 가지다. 첫째, background를 명시적으로 모델링하지 않는 anomaly detection-inspired FSS 프레임워크를 제시했다. 둘째, 3D supervoxel 기반 self-supervision으로 의료영상의 volumetric structure를 활용했다. 셋째, query weak label이 필요 없는 현실적 evaluation protocol을 제안했다.

실제 적용 측면에서 이 연구는 적은 라벨만으로 새로운 장기를 빠르게 분할해야 하는 상황에 유용할 가능성이 크다. 특히 support 정보가 제한적이고 query volume 전체를 다뤄야 하는 임상적 설정에서 더 의미가 있다. 향후에는 noisy supervoxel pseudo-label 처리, heterogeneous foreground에 대한 multi-prototype foreground modeling, 그리고 본격적인 3D end-to-end volume segmentation으로 확장하는 방향이 자연스러운 다음 단계로 보인다.
