# Kernelized Memory Network for Video Object Segmentation

- **저자**: Hongje Seong, Junhyuk Hyun, Euntai Kim
- **발표연도**: 2020
- **arXiv**: https://arxiv.org/abs/2007.08270

## 1. 논문 개요

이 논문은 semi-supervised video object segmentation (VOS) 문제를 다룬다. 이 설정에서는 비디오의 첫 프레임에서 목표 객체의 ground truth mask가 주어지고, 이후 프레임들에서 같은 객체를 픽셀 단위로 계속 분할해야 한다. 논문의 핵심 문제의식은 당시 강력한 방법으로 주목받던 Space-Time Memory Network (STM)가 VOS에 잘 맞는 것처럼 보이지만, 실제로는 중요한 구조적 불일치가 있다는 점이다. STM의 matching은 기본적으로 non-local하게 설계되어 있어 query 프레임의 어떤 위치도 memory 프레임의 어떤 위치와도 강하게 연결될 수 있다. 그러나 실제 VOS에서는 객체가 보통 이전 프레임 근처의 지역적 위치에서 이어져 나타나므로, 문제 자체는 대체로 local한 성격을 가진다.

저자들은 이 불일치를 해결하기 위해 Kernelized Memory Network (KMN)를 제안한다. KMN은 STM의 memory read 과정을 그대로 쓰지 않고, Gaussian kernel을 이용해 matching이 지나치게 비지역적으로 퍼지는 것을 억제한다. 즉, 메모리 네트워크의 장점은 유지하되, VOS의 local continuity라는 문제 특성에 맞게 읽기 과정을 조정한다.

또 하나의 중요한 기여는 pre-training 전략이다. 기존 오프라인 VOS 방법들처럼 static image에서 가짜 비디오를 만들어 사전학습하되, 저자들은 Hide-and-Seek를 도입해 synthetic occlusion을 만들고, 동시에 segmentation boundary 학습도 개선하려 한다. 논문은 이 두 요소가 결합되면 정확도와 속도를 모두 유지하면서 state-of-the-art를 넘어설 수 있다고 주장한다.

## 2. 핵심 아이디어

이 논문의 중심 직관은 간단하다. VOS에서 현재 프레임의 물체는 과거 프레임의 아무 위치에서나 나타나는 것이 아니라, 보통 이전에 있던 위치 주변에서 나타난다. 그런데 STM은 query에서 memory로의 attention을 전역적으로 수행하므로, query 프레임에 비슷하게 생긴 물체가 여러 개 있으면 잘못된 위치와도 강하게 연결될 수 있다. 논문 Figure 1의 예시처럼 memory의 자동차 하나가 query 프레임의 여러 자동차와 동시에 잘못 매칭될 수 있다.

KMN은 이 문제를 해결하기 위해 단순히 query-to-memory matching만 하지 않고, 먼저 memory-to-query matching을 계산한다. 즉, memory의 각 위치 $p$가 query의 어느 위치 $q$와 가장 잘 맞는지 먼저 찾는다. 그 뒤 그 최적 위치를 중심으로 한 2D Gaussian kernel을 query 공간에 씌워, 실제 memory read 시에는 그 주변 지역에서만 강하게 값을 읽도록 만든다. 결과적으로 “matching은 하되, 무작정 전역적으로 읽지 말고 가장 그럴듯한 query 위치 주변으로 국소화하자”는 것이 핵심이다.

기존 접근과의 차별점은 두 가지다. 첫째, memory network를 그대로 쓰지 않고 VOS의 local nature에 맞춰 kernelized read를 설계했다는 점이다. 둘째, static image pre-training에서 Hide-and-Seek를 사용해 occlusion robustness와 boundary quality를 동시에 개선했다는 점이다. 저자들은 특히 Hide-and-Seek가 weakly supervised localization용으로 제안된 기법이지만, 이를 VOS pre-training에 적용해 실제로 성능 향상을 얻었다고 강조한다.

## 3. 상세 방법 설명

전체 구조는 STM과 매우 유사하다. 현재 프레임은 query로 사용하고, 과거 프레임들과 해당 mask들은 memory로 사용한다. backbone으로는 두 개의 ResNet-50을 사용해 memory와 query에서 각각 key와 value feature를 뽑는다. memory 쪽 입력은 RGB와 예측된 mask를 channel-wise concatenation한 형태이고, query는 RGB만 입력으로 받는다. 이후 `res4` feature에서 convolutional embedding을 통해 key와 value를 만든다. 논문에 따르면 이 feature는 입력 해상도의 $1/16$ 크기다. query용 embedding과 memory용 embedding의 구조는 같지만 weight는 공유하지 않는다.

memory에 여러 프레임이 들어올 수 있으므로 각 프레임의 embedded key/value를 시간 축으로 쌓는다. query는 한 프레임만 있으므로 바로 memory read에 사용된다. 그 다음 query와 memory의 key feature 사이의 correlation map을 계산하고, 읽어온 memory value를 query value와 concatenation한 뒤 decoder에 넣어 segmentation mask를 출력한다. decoder는 residual block과 두 개의 refinement module로 구성되며, refinement module 자체는 STM과 동일하다고 설명한다.

핵심은 kernelized memory read이다. 먼저 STM의 기본 memory read를 보면, memory key를 $k^M = \{k^M(p)\}$, query key를 $k^Q = \{k^Q(q)\}$라고 할 때 correlation은 다음처럼 계산된다.

$$
c(p,q)=k^M(p)k^Q(q)^\top
$$

여기서 $p=[p_t,p_y,p_x]$는 memory의 시간-공간 위치이고, $q=[q_y,q_x]$는 query의 공간 위치다. STM에서는 query의 위치 $q$가 memory 전체 위치 $p$를 softmax로 가중합해 value를 읽는다.

$$
r(q)=\sum_p \frac{\exp(c(p,q))}{\sum_p \exp(c(p,q))} v^M(p)
$$

이 방식의 문제는 논문이 명시적으로 두 가지로 정리한다. 첫째, query가 memory를 찾기만 할 뿐 memory가 query를 검증하지는 않는다. 그래서 query 안에 비슷한 물체가 여러 개 있으면 하나의 memory object가 여러 query 위치로 잘못 확장될 수 있다. 둘째, matching이 non-local이라 VOS의 지역적 성질을 활용하지 못한다.

KMN은 이를 다음 순서로 바꾼다.

먼저 기존과 동일하게 correlation map $c(p,q)$를 계산한다. 그 다음 memory의 각 위치 $p$에 대해 query에서 가장 잘 맞는 위치를 argmax로 찾는다.

$$
\hat{q}(p)=\arg\max_q c(p,q)
$$

이 단계가 논문에서 말하는 Memory-to-Query matching이다. 즉, memory 위치 하나마다 query에서 가장 그럴듯한 대응점을 찾는다. 이후 그 위치 $\hat{q}(p)$를 중심으로 2D Gaussian kernel을 정의한다.

$$
g(p,q)=\exp\left(
-\frac{(q_y-\hat{q}_y(p))^2+(q_x-\hat{q}_x(p))^2}{2\sigma^2}
\right)
$$

이제 실제 memory read는 correlation score에 이 Gaussian kernel을 곱해서 수행한다.

$$
r^k(q)=
\sum_p
\frac{
\exp(c(p,q)/\sqrt{d})\, g(p,q)
}{
\sum_p \exp(c(p,q)/\sqrt{d})\, g(p,q)
}
v^M(p)
$$

여기서 $d$는 key channel dimension이고, $1/\sqrt{d}$는 Transformer류 attention에서 쓰이는 scaling factor와 같은 역할을 한다. 저자 설명대로 이는 softmax 입력값이 너무 커져 saturation되는 것을 막는다.

이 식의 의미를 쉬운 말로 풀면, “memory의 어떤 위치가 query의 특정 위치와 가장 잘 맞는다고 판단되면, 그 memory 위치는 query 전체를 대상으로 값을 뿌리는 것이 아니라 그 주변에서만 강하게 작동하게 하자”는 것이다. 따라서 유사한 배경 객체나 멀리 떨어진 비슷한 물체에 잘못 주의를 주는 현상이 줄어든다.

사전학습은 static image dataset에서 수행한다. 기존 연구들처럼 한 장의 이미지와 mask에 random affine transform, rotation, flip, color jittering, cropping 등을 적용해 여러 프레임처럼 보이게 만든다. 여기에 Hide-and-Seek를 추가해 일부 patch를 가린다. 저자들은 이미지를 $24\times24$ grid로 나누고 각 cell을 숨길 확률을 0에서 0.5까지 점진적으로 올렸다고 설명한다. 이렇게 하면 synthetic video 안에서도 occlusion이 발생하므로, 네트워크가 가려짐에 더 강해질 수 있다.

흥미로운 점은 boundary 개선 주장이다. 논문은 많은 segmentation dataset의 GT mask가 object boundary 근처에서 부정확하고 noisy하다고 본다. Hide-and-Seek로 생성한 가려진 영역의 경계는 오히려 더 깨끗하고 명확해서, 모델이 더 선명한 경계를 배우는 데 도움이 된다고 해석한다. 이는 일반적인 “가려짐 강건성 향상”보다 더 독특한 주장이다.

학습은 두 단계로 나뉜다. 먼저 static image pre-training, 그다음 실제 video dataset으로 main training을 수행한다. main training은 STM의 전략을 따르며, 한 비디오에서 시간 순서대로 세 프레임을 샘플링하고 프레임 간 최대 간격을 0에서 25까지 늘린다. loss는 픽셀 단위 cross-entropy이며, optimizer는 Adam, learning rate는 $1e^{-5}$, batch size는 4, 입력 해상도는 $384\times384$로 random resize/crop한다.

중요한 구현 세부 사항으로, 저자들은 Gaussian kernel을 training 때는 사용하지 않고 inference 때만 사용한다. 이유는 Gaussian 중심을 정하는 $\arg\max$가 discrete function이라 gradient가 역전파되지 않기 때문이다. 논문은 training 중 이를 사용하면 잘못 선택된 key 위치를 기준으로 최적화가 진행되어 오히려 성능이 떨어졌다고 설명한다. 이 점은 방법의 실제 동작을 이해하는 데 중요하다.

추론 시에는 모든 과거 프레임을 memory에 넣지 않는다. 첫 프레임과 바로 이전 프레임은 항상 사용하고, 그 외 중간 프레임은 5프레임 간격으로 선택한다. 이는 STM의 memory management 전략을 따른 것이다. multi-object segmentation을 위해 soft aggregation도 사용한다. Gaussian kernel의 표준편차는 $\sigma=7$로 고정했고, test-time augmentation이나 CRF 같은 후처리는 사용하지 않았다고 명시한다.

## 4. 실험 및 결과

실험은 DAVIS 2016, DAVIS 2017, Youtube-VOS 2018에서 수행되었다. DAVIS 2016은 single-object validation 20개 시퀀스, DAVIS 2017은 multi-object validation 30개 시퀀스다. DAVIS에서는 region similarity인 $J^M$, contour accuracy인 $F^M$, 그리고 둘의 평균인 $G^M$를 사용한다. Youtube-VOS 2018은 seen/unseen category를 나누어 $J^S, J^U, F^S, F^U$를 계산하고, overall score를 함께 본다.

논문은 세 가지 학습 설정을 구분해 보여준다. static images만으로 학습한 경우, DAVIS만 사용한 경우, DAVIS와 Youtube-VOS를 함께 사용한 경우다. 이는 Hide-and-Seek pre-training의 효과와 추가 데이터의 효과를 각각 보여주기 위한 구성으로 보인다.

먼저 static images만으로 학습한 결과가 상당히 인상적이다. DAVIS 2016 val에서 KMN은 $G^M=74.8$을 기록했고, DAVIS 2017 val에서는 $G^M=68.9$를 기록했다. 같은 설정에서 STM은 DAVIS 2017 val 기준 $G^M=60.0$이었다. 즉, 저자 주장대로 static pre-training만 놓고 봐도 성능 차이가 매우 크다. 논문은 이를 Hide-and-Seek pre-training이 static image를 VOS 학습에 더 효과적으로 활용하게 만들었다는 근거로 해석한다.

DAVIS를 사용한 본 학습 결과에서도 성능 향상이 유지된다. DAVIS 2016 val에서 KMN은 $G^M=87.6$, DAVIS 2017 val에서 $G^M=76.0$을 기록한다. DAVIS 2016에서는 STM의 $86.5$보다 높고, DAVIS 2017에서는 STM의 $71.6$보다 크게 높다. 특히 DAVIS 2017 multi-object setting에서 상승폭이 크다. runtime은 0.12초/frame으로, STM의 0.16초/frame보다 약간 빠르다.

추가로 Youtube-VOS를 함께 사용했을 때 DAVIS 성능은 더 올라간다. DAVIS 2016 val에서 $G^M=90.5$, DAVIS 2017 val에서 $G^M=82.8$을 달성했다. 대응되는 STM 수치는 각각 $89.3$, $81.8$이다. 상승폭은 아주 크지는 않지만 여전히 우세하다.

보다 일반화된 비교를 위해 ground truth가 공개되지 않은 benchmark도 제시한다. DAVIS 2017 test-dev에서 KMN은 $G^M=77.2$, $J^M=74.1$, $F^M=80.3$을 기록했고, STM은 $G^M=72.2$였다. 논문이 강조하는 “+5%”는 바로 이 $G^M$ 기준 차이다. 이는 저자들이 가장 강하게 내세우는 결과 중 하나다.

Youtube-VOS 2018 validation에서도 KMN은 overall 81.4를 기록했고, seen/unseen 기준으로 $J^S=81.4$, $J^U=75.3$, $F^S=85.6$, $F^U=83.3$이다. STM은 overall 79.4였다. 따라서 KMN은 DAVIS뿐 아니라 대규모 Youtube-VOS에서도 일관되게 STM보다 좋다.

ablation study는 두 요소의 효과를 분리해 보여준다. 기준 모델인 STM에서 시작했을 때, 저자 구현에서 Hide-and-Seek와 kernelized memory를 둘 다 제거한 경우 DAVIS16 성능이 오히려 81.3까지 떨어진다. 이는 저자 기본 구현이 STM 원본과 완전히 동일하지는 않음을 시사하지만, 핵심은 두 제안 요소를 하나씩 넣을 때 성능이 오른다는 것이다. Hide-and-Seek만 추가하면 DAVIS17에서 $G^M=75.9$, kernelized memory만 추가하면 Youtube-VOS overall이 81.0, 둘 다 넣으면 DAVIS17 $G^M=76.0$, Youtube-VOS overall 81.4가 된다. 즉 두 요소가 보완적으로 작동한다고 해석할 수 있다.

정성적 결과에서도 저자들은 fast deformation, similar background objects, severe occlusion 상황에서 기존 방법보다 더 안정적이라고 주장한다. 특히 STM과 직접 비교한 Figure 6에서 “multiple similar objects”와 “occlusion”이 있는 장면에서 개선이 두드러진다고 설명한다. 다만 정성 결과는 이미지 자체를 여기서 직접 확인할 수는 없고, 논문 텍스트의 설명에 근거해 해석해야 한다.

Hide-and-Seek의 boundary 효과를 보이기 위해 픽셀 단위 cross-entropy loss visualization도 제시한다. 저자 설명에 따르면 원본 GT 기반 학습에서는 예측이 전반적으로 맞아도 경계 부근에서 loss가 많이 활성화된다. 반면 Hide-and-Seek로 생성한 경계에서는 그런 현상이 덜 나타난다. 저자들은 이를 noisy boundary GT보다 Hide-and-Seek가 더 학습하기 좋은 경계를 제공한다는 증거로 본다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법이 잘 맞물린다는 점이다. 단순히 기존 STM을 조금 변형한 수준이 아니라, 왜 STM의 non-local read가 VOS에 구조적으로 어긋날 수 있는지 명확한 문제 제기를 한다. 그리고 그 문제를 memory-to-query argmax와 Gaussian localization이라는 비교적 직관적이고 구현 가능한 방식으로 해결한다. 아이디어가 복잡하지 않은데도 benchmark에서 큰 개선을 보였다는 점은 설계의 설득력을 높인다.

또 다른 강점은 pre-training 전략에 있다. 많은 논문이 static image pre-training을 사용하지만, 이 논문은 occlusion 부재와 boundary noise라는 두 가지 구체적 문제를 짚고 Hide-and-Seek를 통해 이를 동시에 개선하려 한다. 특히 static images만 사용한 실험에서 성능 향상이 큰 점은 이 전략이 단순한 보조기법이 아니라 실제로 중요한 역할을 한다는 것을 보여준다.

실험도 비교적 탄탄하다. DAVIS validation처럼 공개 GT가 있는 셋뿐 아니라 DAVIS 2017 test-dev와 Youtube-VOS validation처럼 GT가 공개되지 않은 benchmark에서도 strong result를 보고해, 과적합이나 과도한 validation tuning 우려를 어느 정도 줄였다. 또한 online-learning 방법까지 포함해 비교하면서도 runtime이 빠르다는 점을 함께 제시한다.

하지만 한계도 분명하다. 첫째, Gaussian kernel은 inference 때만 적용되고 training 때는 빠진다. 이는 방법의 핵심 연산이 end-to-end하게 학습되지 않는다는 뜻이다. 저자들은 argmax의 비미분 가능성 때문에 그렇게 했다고 설명하지만, 결국 학습과 추론의 동작이 다르다는 점은 구조적 제약이다. 향후 differentiable approximation을 쓰거나 학습 시점에서도 locality bias를 반영하는 개선 여지가 있다.

둘째, locality assumption이 항상 맞는 것은 아니다. 대부분의 VOS 장면에서는 객체 이동이 비교적 지역적이지만, 급격한 카메라 이동, 장면 전환, 큰 displacement, 순간적인 re-appearance 같은 상황에서는 지나친 localization bias가 오히려 불리할 수 있다. 논문은 이런 경우에 대한 별도의 failure case 분석은 제공하지 않는다.

셋째, Hide-and-Seek가 boundary를 더 정확하게 만들어 준다는 해석은 흥미롭지만, 이것이 모든 데이터셋과 객체 유형에서 항상 성립하는지는 논문만으로는 확실하지 않다. 저자들은 loss visualization을 통해 간접적 근거를 제시하지만, boundary annotation quality 자체를 정량적으로 검증한 것은 아니다. 즉 “GT boundary보다 Hide-and-Seek boundary가 더 낫다”는 주장은 부분적으로 설득력 있으나, 보다 직접적인 측정이 있으면 더 강했을 것이다.

넷째, 논문은 decoder 상세는 기존 STM을 따른다고 하며 많은 부분을 생략한다. 따라서 진짜 성능 향상이 memory read 자체에서 얼마나 오고, 나머지 training recipe나 데이터 구성의 영향이 얼마나 되는지는 완전히 분리되지는 않는다. ablation은 제공되지만, 저자 구현의 baseline이 STM 원본과 정확히 동일한지까지는 텍스트만으로 확정할 수 없다.

종합적으로 비판적으로 보면, 이 논문은 “비디오 객체 분할은 local하다”는 중요한 관찰을 효과적인 구조 변경으로 연결한 점이 강하다. 다만 locality를 어떻게 학습 과정에 자연스럽게 통합할지, 그리고 큰 displacement나 복잡한 motion에 대해 어떤 trade-off가 생기는지는 후속 연구가 더 필요하다.

## 6. 결론

이 논문은 semi-supervised VOS를 위해 두 가지 핵심 기여를 제시한다. 첫째, STM의 non-local memory read를 그대로 사용하는 대신 Gaussian kernel로 국소화한 kernelized memory read를 도입해, VOS의 지역적 특성에 더 잘 맞는 memory retrieval을 설계했다. 둘째, static image pre-training에 Hide-and-Seek를 적용해 occlusion robustness와 boundary quality를 개선했다.

실험 결과는 이 두 아이디어가 실제로 유효함을 보여준다. 특히 DAVIS 2017 test-dev에서 STM 대비 약 5포인트 높은 $G^M$을 기록한 점은 논문의 가장 강한 실증적 주장이다. 동시에 runtime도 빠른 편이어서, 정확도만 높은 무거운 방법이 아니라 실용성도 갖춘 오프라인 VOS 방법으로 볼 수 있다.

실제 적용 측면에서는, memory-based VOS 구조를 유지하면서도 matching locality를 더 잘 반영해야 하는 다양한 video segmentation 문제에 직접적인 영향을 줄 수 있다. 향후 연구에서는 이 locality bias를 미분 가능한 방식으로 학습에 통합하거나, object re-identification과 장거리 이동까지 함께 다루는 방향으로 확장될 가능성이 있다. 논문 자체의 주장대로, 제안 아이디어는 VOS뿐 아니라 다른 segmentation-related task에도 응용될 여지가 있다.
