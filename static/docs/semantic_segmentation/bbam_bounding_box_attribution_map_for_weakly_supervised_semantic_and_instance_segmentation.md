# BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation

- **저자**: Jungbeom Lee, Jihun Yi, Chaehun Shin, Sungroh Yoon
- **발표연도**: 2021
- **arXiv**: https://arxiv.org/abs/2103.08907

## 1. 논문 개요

이 논문은 bounding box annotation만으로 semantic segmentation과 instance segmentation을 더 정확하게 학습시키기 위한 방법인 **BBAM (Bounding Box Attribution Map)** 을 제안한다. 목표는 각 bounding box 내부에서 실제 객체가 차지하는 픽셀 영역을 최대한 정확히 찾아내어, 이를 segmentation 모델 학습을 위한 pseudo ground truth로 사용하는 것이다.

문제의 핵심은 bounding box는 객체의 대략적 위치만 알려줄 뿐, 어떤 픽셀이 foreground이고 어떤 픽셀이 background인지는 알려주지 않는다는 점이다. 기존 bounding-box 기반 weakly supervised segmentation 방법들은 주로 GrabCut, MCG 같은 class-agnostic mask generator에 의존했는데, 이런 방법들은 색, 밝기, 경계 같은 low-level image cue에 크게 의존한다. 그래서 복잡한 배경, 가려짐, 객체 내부 appearance variation이 있는 경우 정확한 mask를 만들기 어렵다.

이 논문은 이 한계를 넘기 위해, low-level cue 대신 **이미 학습된 object detector가 실제로 어떤 이미지 영역을 이용해 detection을 수행하는지**를 활용한다. 즉, detector의 예측을 거의 유지하면서도 입력 이미지에서 최소한의 영역만 남기는 mask를 찾고, 그 결과를 BBAM이라 정의한다. 저자들은 이 방식이 detector가 학습한 higher-level semantic information을 활용한다는 점에서 기존 bounding-box 기반 pseudo mask 생성 방식과 구별된다고 주장한다.

## 2. 핵심 아이디어

중심 아이디어는 간단하지만 강력하다. 어떤 proposal과 객체에 대해 detector가 원본 이미지에서 낸 classification 결과와 box regression 결과를 거의 그대로 유지하려면, 이미지의 어느 부분이 반드시 필요할까를 묻는다. 이때 필요한 최소 영역을 찾는 최적화 문제를 풀면, detector가 그 객체를 인식하고 위치를 조정하는 데 실제로 의존하는 픽셀 영역이 드러난다. 이 픽셀 수준 지도(map)가 BBAM이다.

기존 접근과의 차별점은 두 가지다. 첫째, bounding box 내부에서 mask를 찾되, 단순한 low-level segmentation이 아니라 **object detector의 동작 자체**를 사용한다. 둘째, detector의 두 출력인 **classification head와 bounding box regression head를 함께 사용**해 localization signal을 얻는다. 논문은 이 두 head가 서로 다른 정보를 본다고 분석한다. `box head`는 객체 경계 근처에 더 민감하고, `cls head`는 객체 내부의 discriminative region에 더 민감하다. 둘을 결합하면 서로 보완적인 pseudo mask를 만들 수 있다는 것이 논문의 핵심 주장이다.

또 하나의 중요한 설계는 **adaptive stride**이다. perturbation mask를 너무 세밀하게 두면 adversarial artifact가 생기고, 너무 거칠게 두면 작은 객체를 제대로 표현하지 못한다. 그런데 object detector는 RoIAlign을 통해 proposal마다 feature scale이 달라지므로, 고정 stride는 객체 크기에 따라 부적절해질 수 있다. 저자들은 bounding box 크기에 따라 stride를 조절해 이 문제를 완화한다.

## 3. 상세 방법 설명

전체 파이프라인은 다음과 같다. 먼저 bounding box annotation으로 object detector를 학습한다. 이후 학습된 detector로 각 ground-truth box에 대해 BBAM을 생성한다. 생성된 BBAM을 thresholding, CRF refinement, 그리고 필요하면 MCG proposal refinement를 거쳐 pseudo mask로 변환한다. 마지막으로 이 pseudo mask로 semantic segmentation network와 instance segmentation network를 각각 학습한다.

논문은 Faster R-CNN 같은 two-stage detector를 기준으로 설명한다. detector는 proposal마다 두 가지 출력을 낸다. 하나는 `cls head`가 예측하는 class probability $p_c$이고, 다른 하나는 `box head`가 예측하는 bounding box offset $t_c=(t_x^c,t_y^c,t_w^c,t_h^c)$이다. BBAM은 이 원래 예측을 유지하는 데 필요한 최소 이미지 영역을 찾는 방식으로 정의된다.

입력 이미지 $I$와 mask $M$이 있을 때, 논문은 다음 perturbation 함수를 사용한다.

$$
\Phi(I,M)=I\circ M+\mu\circ(1-M)
$$

여기서 $\circ$는 픽셀 단위 곱이고, $\mu$는 training data의 channel-wise mean이다. 즉, mask가 1인 부분은 원래 이미지를 유지하고, 0인 부분은 평균색으로 치환한다. 이 방식은 “어떤 부분을 남겨야 detector의 출력을 유지할 수 있는가”를 검사하는 역할을 한다.

각 proposal $o$에 대해 최적 mask $M^\*$는 다음 최적화로 구한다.

$$
M^\*=\arg\min_{M\in[0,1]^\Omega}\lambda \lVert M\rVert_1 + L_{\text{perturb}}
$$

여기서 $\lambda \lVert M\rVert_1$는 가능한 한 작은 영역만 남기도록 유도하는 sparsity term이다. 그리고

$$
L_{\text{perturb}}=
\mathbf{1}_{box}\left\|t_c-f_{box}(\Phi(I,M),o)\right\|_1
+
\mathbf{1}_{cls}\left\|p_c-f_{cls}(\Phi(I,M),o)\right\|_1
$$

이다. $t_c=f_{box}(I,o)$, $p_c=f_{cls}(I,o)$는 원본 이미지에서의 예측이다. 따라서 이 손실은 mask가 적용된 이미지에서도 원래의 box regression 결과와 classification 결과를 유지하도록 강제한다. $\mathbf{1}_{box}$와 $\mathbf{1}_{cls}$는 어떤 head를 사용할지 조절하는 스위치다.

부록에서는 artifact를 줄이기 위해 total variation regularization도 사용했다고 명시한다. 최종 loss는 다음과 같다.

$$
L_M=
\lambda \lVert M\rVert_1
+
\lambda_{TV}\lVert \nabla M\rVert_\beta^\beta
+
\mathbf{1}_{box}\left\|t_c-f_{box}(\Phi(I,M),o)\right\|_1
+
\mathbf{1}_{cls}\left\|p_c-f_{cls}(\Phi(I,M),o)\right\|_1
$$

저자들은 $\lambda_{TV}=10^{-4}$, $\beta=3$를 사용했고, Adam optimizer로 mask를 300 iteration 동안 최적화했다.

중요한 실무적 설계는 perturbation mask의 해상도다. 입력 이미지와 같은 해상도의 mask를 직접 최적화하면 adversarial effect 때문에 아주 작은 변화도 detector의 출력을 크게 흔들 수 있다. 이를 막기 위해 coarse mask를 먼저 만들고 upsampling해서 사용한다. 그런데 object detector에서는 proposal 크기가 다양하므로 고정 stride $s$가 적합하지 않다. 논문은 객체 크기 비율 $a$에 따라 stride를

$$
s(a)=16+48\sqrt{a}
$$

로 설정했다. 즉 작은 객체에는 작은 stride, 큰 객체에는 큰 stride를 적용해 RoI pooling 이후에도 perturbation unit 크기가 적절하도록 만든다.

Pseudo ground truth를 만들 때는 하나의 ground-truth box에 대해 여러 개의 jittered proposal을 만든다. 각 좌표를 최대 $\pm 30\%$ 범위에서 무작위로 흔들어 proposal set $O$를 만들고, 그중 `cls head`가 정답 클래스를 맞추고 `box head`의 예측 box가 ground-truth와 IoU 0.8 이상인 proposal만 positive proposal set $O^+$에 넣는다. 그런 뒤 여러 proposal의 정보를 하나의 mask에 모으기 위해 loss를 평균 형태로 바꾼다.

$$
L_{\text{perturb}}=
\mathbb{E}_{o\in O^+}
\left[
\mathbf{1}_{box}\left\|t_c-f_{box}(\Phi(I,M),o)\right\|_1
+
\mathbf{1}_{cls}\left\|p_c-f_{cls}(\Phi(I,M),o)\right\|_1
\right]
$$

이 단계에서는 두 head를 모두 사용하므로 $\mathbf{1}_{box}=\mathbf{1}_{cls}=1$이다. 이렇게 얻은 BBAM은 객체 전체를 덜 덮을 수 있으므로 CRF로 refinement를 수행한다.

그 다음 pseudo mask를 만들기 위해 threshold를 적용한다. 단일 threshold $\theta$ 대신, foreground threshold $\theta_{fg}$와 background threshold $\theta_{bg}$ 두 개를 둔다. attribution value가 $\theta_{fg}$보다 큰 픽셀은 foreground, $\theta_{bg}$보다 작은 픽셀은 background로 두고, 중간 값 픽셀은 training loss에서 무시한다. 논문은 $\theta_{fg}=0.8$, $\theta_{bg}=0.2$를 사용했다. 이 설계는 객체마다 attribution 분포가 다르기 때문에 고정 단일 threshold보다 더 안정적이라는 취지다.

Instance segmentation에서는 추가로 MCG proposal을 이용해 mask를 refinement한다. BBAM 기반 mask $T$와 IoU가 가장 높은 MCG mask 하나를 선택하고, 동시에 $T$ 내부에 완전히 포함된 다른 proposal들도 함께 합친다. 식으로는 다음과 같다.

$$
T^r=\bigcup_{i\in S} m_i,\quad
S=\left\{i\mid m_i\subset T\right\}\cup\left\{\arg\max_i IoU(m_i,T)\right\}
$$

이 refinement는 특히 medium, large object에서 성능 향상에 효과적이었다고 보고한다.

최종 segmentation network 학습은 task별로 다르다. Instance segmentation에는 ImageNet pretrained Mask R-CNN을 사용했고, ambiguous pixel을 점진적으로 학습에 포함시키는 **seed growing** 기법을 적용했다. Semantic segmentation에는 ImageNet pretrained DeepLab-v2를 사용했고, instance-level pseudo label을 class-level label로 바꾸어 학습했다. 여러 클래스에 동시에 할당되는 픽셀은 loss 계산에서 무시했다.

## 4. 실험 및 결과

실험은 PASCAL VOC 2012와 MS COCO 2017에서 수행되었다. PASCAL VOC에서는 20개 object class와 background를 사용했고, Hariharan 등이 만든 augmented train set 10,582장을 사용했다. MS COCO는 118K training image와 80개 class를 가진다. Semantic segmentation은 mIoU로 평가했고, instance segmentation은 $AP_\tau$, $AP$ (IoU 0.5부터 0.95 평균), 그리고 ABO를 사용했다.

재현 세팅도 비교적 명확히 제공된다. Detector와 instance segmentation에는 PyTorch 구현의 Faster R-CNN, Mask R-CNN을 사용했고, semantic segmentation에는 DeepLab-v2-ResNet101 구현을 사용했다. BBAM 최적화는 Adam, learning rate 0.02, 300 iteration으로 수행했다.

PASCAL VOC instance segmentation 결과에서 BBAM은 bounding-box supervision 방식 중 최고 성능을 기록했다. 논문 표에 따르면 BBAM은 `AP25 = 76.8`, `AP50 = 63.7`, `AP70 = 39.5`, `AP75 = 31.8`, `ABO = 63.0`을 얻었다. 이전 bounding-box 기반 강한 baseline인 Arun et al. (ECCV 2020)은 `AP50 = 57.7`, `AP70 = 33.5`, `AP75 = 31.2`였으므로, BBAM은 특히 `AP50`과 `AP70`에서 약 6.0%p 향상을 보였다. 완전 지도 Mask R-CNN과 비교하면 `AP50` 기준 약 92.2%, `ABO` 기준 약 95.7% 수준까지 도달했다.

MS COCO instance segmentation에서도 성능 향상이 크다. `val`에서 BBAM은 `AP = 26.0`, `AP50 = 50.0`, `AP75 = 23.9`를 기록했고, bounding-box supervision의 기존 방법 Hsu et al.은 `AP = 21.1`, `AP50 = 45.5`, `AP75 = 17.2`였다. 특히 `AP75`가 6.7%p 개선되었다. `test-dev`에서도 BBAM은 `AP = 25.7`, `AP50 = 50.0`, `AP75 = 23.3`을 기록했다.

PASCAL VOC semantic segmentation에서는 BBAM이 `val`과 `test` 모두에서 `mIoU = 73.7`을 기록했다. 이는 BoxSup 62.0/64.6, SDI 69.4, Song et al. 70.2보다 높고, image-level label 기반 최신 기법들인 FickleNet 64.9/65.3, CIAN 64.3/65.3, Sun et al. 66.2/66.9보다도 높다. 저자들은 vanilla DeepLab-v2만 사용하고 추가적인 recursive training, label refinement during training, 추가 loss fine-tuning 없이 이 수치를 얻었다고 강조한다.

논문은 concurrent work인 Box2Seg도 언급하지만, backbone과 segmentation architecture가 다르기 때문에 절대 수치만으로 단순 비교하지 않는다. 대신 fully supervised counterpart 대비 상대 성능을 비교한다. Box2Seg는 fully supervised 대비 88.4%, BBAM은 96.7%라고 보고한다. 이는 약한 supervision 대비 pseudo label 품질이 높다는 논지에 사용된다.

Ablation도 비교적 설득력 있다. MCG refinement를 넣으면 PASCAL VOC에서 `AP`가 29.6에서 33.4로, MS COCO에서 23.5에서 26.0으로 오른다. 특히 medium, large object에서 효과가 컸다. 다만 이후 구성요소 기여를 더 명확히 보기 위해 나머지 ablation은 MCG 없이 보고했다고 명시한다.

`box head`와 `cls head`의 역할 분석에서는 각각 단독으로도 꽤 괜찮은 성능을 보이지만, 둘을 함께 쓸 때 semantic/instance 모두 최고 성능이 나왔다. 이는 두 head가 complementary하다는 정성적 분석과 일치한다.

Threshold와 seed growing에 대한 분석에서는, foreground/background를 단순 이분화하는 것보다 일부 픽셀을 ignore하는 것이 AP 향상에 도움이 되었고, seed growing을 추가하면 성능이 더 오른다. $\lambda$에 대한 민감도 분석에서도 semantic과 instance 성능이 넓은 범위에서 비교적 안정적이었다.

BBAM 자체에 대한 추가 분석도 중요하다. 저자들은 높은 attribution 값을 갖는 픽셀이 주로 객체 경계와 discriminative part에 분포한다고 보고한다. 또한 `box head`는 경계 쪽, `cls head`는 내부 쪽 픽셀을 더 강조하는 경향을 시각화했다. Gradient-based attribution과의 비교에서는, detector의 구조상 proposal 밖 픽셀도 실제로 중요할 수 있지만 gradient map은 proposal 내부에 편중되는 문제가 있다고 지적한다. 논문은 PASCAL VOC validation에서 positive proposal과 예측 box의 평균 IoU가 0.56으로 낮은데도, attribution value 0.9 이상 픽셀의 87%가 imperfect proposal 내부에 나타난다고 보고하며 gradient 방식의 한계를 실험적으로 보인다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 weakly supervised segmentation에서 **pseudo mask 생성의 정보원을 low-level segmentation heuristic에서 detector의 semantic behavior로 옮겼다**는 점이다. 이는 bounding box supervision이 이미 담고 있는 객체 위치 정보를 더 직접적으로 활용하는 방식이다. 단순히 bounding box를 search space로 쓰는 기존 방법보다 개념적으로 더 일관적이고, 실험 결과도 이를 뒷받침한다.

두 번째 강점은 method-design과 analysis가 함께 잘 정리되어 있다는 점이다. 단순히 성능 향상만 보고하는 것이 아니라, 왜 `box head`와 `cls head`를 같이 써야 하는지, 왜 adaptive stride가 필요한지, 왜 gradient attribution이 detector에는 잘 맞지 않는지를 정성·정량 분석으로 제시한다. 이런 분석은 방법의 설득력을 높인다.

세 번째 강점은 semantic segmentation과 instance segmentation 양쪽에서 모두 강한 결과를 보인다는 점이다. 특히 별도 architecture modification 없이 vanilla DeepLab-v2, Mask R-CNN 위에서 좋은 성능을 냈다는 점은 pseudo label의 품질이 높다는 간접 증거가 된다.

반면 한계도 분명하다. 첫째, 이 방법은 **먼저 object detector를 잘 학습해야 한다**. detector가 부정확하면 BBAM도 부정확해질 가능성이 높다. 논문은 noisy box label에 대해 Hsu et al.보다 robust하다고 보였지만, detector quality와 BBAM quality의 관계를 체계적으로 분리해 분석하지는 않았다.

둘째, BBAM 생성 자체가 계산량이 적지 않다. 각 proposal 또는 proposal 집합에 대해 mask optimization을 Adam으로 300 iteration 수행해야 하므로, pseudo label 생성 비용이 상당할 수 있다. 논문은 성능은 충분히 보여주지만, 생성 시간이나 실제 대규모 적용에서의 비용은 구체적으로 분석하지 않았다.

셋째, pseudo label refinement 단계에서 CRF와 MCG를 사용한다. 특히 instance segmentation 향상을 위해 MCG refinement가 효과적이었는데, 이는 완전히 detector attribution만으로 끝나는 구조는 아니라는 뜻이다. 즉, 제안법의 핵심은 BBAM이지만 최종 성능은 후처리 파이프라인의 도움도 받는다.

넷째, 논문은 one-stage detector에도 쉽게 확장 가능하다고 결론에서 말하지만, 본문 실험은 사실상 Faster R-CNN류 two-stage detector 중심이다. 따라서 one-stage detector에서 실제로 동일한 수준의 효과가 나는지는 이 논문만으로는 확인할 수 없다. 저자도 이 부분에 대한 실험은 제공하지 않았다.

다섯째, semantic segmentation 쪽에서는 Box2Seg보다 절대 성능이 낮지만, backbone 차이를 들어 상대 비교로 해석한다. 이 해석은 어느 정도 타당하지만, 동일한 segmentation architecture에서 직접 비교한 것은 아니므로 그 결론은 제한적으로 받아들이는 것이 맞다.

## 6. 결론

이 논문은 bounding-box supervision만으로 픽셀 수준의 pseudo ground truth를 만드는 새로운 방법인 BBAM을 제안했다. 핵심은 detector의 classification과 localization 결과를 유지하는 최소 이미지 영역을 찾는 것이며, 이를 통해 객체의 경계와 내부 정보를 모두 반영한 attribution map을 얻는다는 점이다. 이 BBAM을 pseudo mask로 사용해 semantic segmentation과 instance segmentation을 학습하면, PASCAL VOC와 MS COCO에서 당시 strong state-of-the-art 수준의 성능을 달성할 수 있음을 보였다.

이 연구의 의미는 bounding box weak supervision에서 “어떤 low-level mask generator를 쓸 것인가”보다 “이미 학습된 detector가 무엇을 보고 판단하는가”를 활용하는 방향을 제시했다는 데 있다. 실제 응용 측면에서는 pixel annotation 비용을 크게 줄이면서도 segmentation 성능을 유지하려는 상황에 유용할 가능성이 높다. 향후 연구에서는 더 효율적인 BBAM 생성, one-stage detector 확장, detector 품질과 pseudo label 품질의 관계 분석, 그리고 CRF/MCG 없이도 강한 성능을 내는 end-to-end 구조로의 발전이 중요한 후속 과제가 될 것이다.
