# Iterative Instance Segmentation

- **저자**: Ke Li, Bharath Hariharan, Jitendra Malik
- **발표연도**: 2016
- **arXiv**: https://arxiv.org/abs/1511.08498

## 1. 논문 개요

이 논문은 pixel-wise labeling 문제에서 기존 방법들이 출력 구조(output structure)를 거의 활용하지 않는다는 점을 문제로 삼는다. 특히 instance segmentation에서는 어떤 픽셀이 특정 category에 속하는지는 맞게 찾더라도, 서로 붙어 있는 여러 객체 인스턴스를 올바르게 분리하지 못하는 경우가 많다. 저자들은 이런 실패의 핵심 원인이 각 픽셀을 거의 독립적으로 예측하는 방식에 있다고 본다.

기존에는 이런 한계를 보완하기 위해 superpixel projection이나 CRF 같은 후처리를 사용했지만, 이런 방식은 주로 색, 질감, 경계 같은 local cue를 반영할 뿐, object-level shape 같은 global cue를 충분히 반영하지 못한다. shape 제약을 명시적으로 모델링하려면 higher-order potential이 필요하고, 이는 설계도 어렵고 inference도 대체로 tractable하지 않다.

이 논문의 목표는 이러한 structured prediction 문제를 직접 푸는 대신, 이를 여러 번의 unconstrained prediction 문제로 분해하여 해결하는 것이다. 핵심 주장은, 모델이 자기 자신의 이전 예측을 반복적으로 수정하도록 학습하면, shape, region contiguity, contour smoothness 같은 구조적 prior를 명시적으로 설계하지 않아도 데이터로부터 암묵적으로 학습할 수 있다는 점이다. 저자들은 이를 instance segmentation에 적용해 당시 state-of-the-art를 넘어서는 성능을 보고한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 “한 번에 정답 mask를 맞히게 하지 말고, 여러 단계에 걸쳐 이전 단계의 오류를 점진적으로 수정하게 하자”는 것이다. 첫 단계에서는 구조를 거의 무시한 거친 예측을 허용하고, 이후 단계에서는 이미지와 이전 prediction을 함께 입력받아 더 나은 prediction으로 보정한다. 이렇게 하면 모델은 “어떤 형태가 말이 되는지”를 직접 규칙으로 주지 않아도, 반복적인 correction 과정 속에서 구조를 암묵적으로 배우게 된다.

이 접근은 전통적인 structured prediction과 다르다. 전통적 방법은 shape constraint를 explicit potential로 정의하고, 그 제약을 만족하는 출력을 한 번에 찾으려 한다. 반면 이 논문은 structure를 직접 공식화하지 않는다. 대신, 모델이 이전 결과를 보고 “어디가 이상한지”를 고치도록 훈련한다. 저자들은 이를 통해 사람이 미리 “사람은 머리가 하나여야 한다” 같은 규칙을 넣지 않아도, 모델이 category-specific shape prior를 스스로 형성할 수 있다고 주장한다.

또한 논문은 local correction이 global structure 자체를 직접 모델링하는 것보다 더 쉽다고 해석한다. 저자들의 설명에 따르면, 단일 단계 예측은 복잡한 구조 manifold 자체를 학습해야 하지만, iterative prediction은 그 manifold를 정의하는 implicit function의 gradient를 배우는 것에 가깝다. 이 gradient는 보통 더 단순한 기하를 가지므로 학습이 더 쉬울 수 있다는 것이다.

## 3. 상세 방법 설명

이 논문이 다루는 task는 SDS(simultaneous detection and segmentation), 즉 각 object detection box에 대해 해당 인스턴스의 segmentation mask를 예측하는 문제다. 전체 파이프라인은 크게 detection과 segmentation 두 부분으로 나뉜다. 먼저 detection system이 candidate bounding box, score, category label을 제공하고, 이후 segmentation system이 각 box 내부 픽셀의 foreground probability heatmap을 예측한다. 저자들은 detection은 fast R-CNN을 MCG proposal과 함께 사용하고, 논문의 기여는 segmentation system 설계에 집중한다고 밝힌다.

segmentation backbone은 Hariharan et al.의 hypercolumn net, 그중에서도 VGG-16 기반 O-Net 구조를 바탕으로 한다. 입력은 $224 \times 224$ patch이고 출력은 $50 \times 50$ heatmap이다. hypercolumn representation은 여러 중간 layer의 upsampled feature map을 concat하여 finer-scale 정보와 box 내부 상대 위치 정보를 유지하도록 설계되어 있다.

제안 방법의 핵심은 각 단계 $t$에서 모델 $f$가 이미지 $x$뿐 아니라 이전 단계 prediction도 함께 입력받는다는 점이다. 테스트 시 초기 prediction은 모든 픽셀이 $1/2$인 상수 heatmap으로 시작한다. 이후 반복적으로

$$
\hat{y}^{(t)} = f\left(x, \hat{y}^{(t-1)}\right)
$$

를 적용하고, 마지막 단계의 출력 $\hat{y}^{(M)}$를 최종 prediction으로 사용한다. 여기서 중요한 점은 모델이 “현재 이미지가 어떻게 생겼는가”와 “이전 prediction이 어떤 모양이었는가”를 동시에 보고 correction을 수행한다는 것이다.

학습도 단계적으로 진행된다. 처음에는 이전 prediction을 전부 $1/2$로 두고 ground-truth mask를 예측하도록 학습한다. 이후 각 단계에서는 이전 단계들에서 생성된 prediction들을 입력으로 사용해 다시 ground truth를 예측하도록 학습한다. 논문 pseudocode에 따르면 현재 단계 $t$의 학습 집합은 이전 모든 단계 $i < t$에서 생성된 prediction $p_x^{(i)}$를 포함해 만든다. 즉 모델은 단순히 “직전 단계 output만” 보정하는 것이 아니라, 과거 여러 단계에서 나올 수 있는 imperfect prediction들에 대해 강건하게 correction하도록 훈련된다.

구현적으로는 category-specific shape prior를 더 잘 반영하기 위해 입력층을 수정했다. 원래 RGB 3채널 입력 외에 20개의 추가 채널을 두고, detection이 지정한 category에 해당하는 채널에 이전 단계 heatmap을 넣는다. 나머지 category 채널은 0으로 채운다. 이는 shape consistency가 category별로 달라진다는 가정을 반영한 설계다. 예를 들어 person의 plausible shape와 bicycle의 plausible shape는 다르므로, 이전 prediction을 category 조건부로 제공하는 것이다.

입력 준비 과정에서는 detection box 내부 patch를 잘라 $224 \times 224$로 anisotropic scaling하고, ground-truth mask도 같은 방식으로 변환한다. 이전 단계 heatmap은 원래 $50 \times 50$이므로 bilinear interpolation으로 $224 \times 224$까지 upsample한 뒤 입력으로 넣는다. 또 학습 안정성을 위해 이 heatmap 값을 element-wise로 rescale 및 center하여 $[-127, 128]$ 범위로 맞춘다.

손실 함수는 ground-truth mask에 대한 pixel-wise negative log-likelihood의 합이다. 즉 각 위치에서 foreground/background classification을 수행하는 형태다. 논문은 이를 명시적으로

- 각 픽셀에 대해 negative log likelihood를 계산하고
- 전체 heatmap 위치에 대해 합산하는 방식

으로 설명한다. 다만 본문에는 sigmoid인지 softmax인지 같은 activation 세부는 직접적으로 명시되어 있지 않으므로, loss의 정확한 출력 parameterization까지는 확정적으로 말할 수 없다.

최적화는 SGD를 사용하며 mini-batch 크기는 32, learning rate는 $5 \times 10^{-5}$, momentum은 $0.9$이다. 총 4 stage 학습을 수행하고, 각 stage의 iteration 수는 순서대로 30K, 42.5K, 50K, 20K다. 테스트 시에는 보통 3번 iteration이면 수렴한다고 보고 3 step inference를 사용한다.

후처리로는 optional superpixel projection과 rescoring을 쓴다. heatmap에서 최종 region을 만들 때, 픽셀 또는 superpixel 내부 평균 intensity가 40%보다 크면 foreground로 칠한다. rescoring은 [16]과 비슷하게 bounding box patch와 background-masked patch에서 얻은 CNN feature로 SVM을 학습해 detection score를 다시 조정한다. 마지막 detection 집합은 region overlap 30% 기준 NMS를 적용해 얻는다.

## 4. 실험 및 결과

평가는 PASCAL VOC 2012 validation set 위에서 SBD instance segmentation annotation을 사용해 수행한다. 평가지표는 region average precision인 $AP^r$이며, 일반 detection AP와 유사하지만 overlap 계산을 bounding box IoU가 아니라 region mask의 pixel-wise intersection-over-union으로 정의한다. 논문은 특히 50% overlap과 70% overlap에서의 mean $AP^r$를 주요 지표로 사용한다.

기존 방법과의 전체 비교에서 제안 방법은 매우 강한 결과를 보인다. Table 1에 따르면 mean $AP^r$는 50% overlap에서 63.6%, 70% overlap에서 43.3%이다. 이는 Hypercolumn의 62.4%, 39.4%보다 높고, CFM의 60.7%, 39.6%보다도 높다. 특히 70% overlap에서의 향상이 크다는 점은, 단순히 대충 비슷한 mask를 얻는 수준이 아니라 더 정밀한 경계와 instance separation에 강점이 있음을 시사한다.

보다 세부적으로 보면, 제안 방법은 raw pixel-wise prediction 단계부터 이미 baseline보다 우수하다. Hypercolumn 대비 raw setting에서 50% overlap은 56.1%에서 60.1%로, 70% overlap은 29.4%에서 38.7%로 상승한다. 특히 70% overlap에서 9.3 point 상승은 상당히 크다. 이는 단순히 superpixel projection이나 rescoring 같은 후처리 덕분이 아니라, 원래 heatmap 자체의 질이 개선되었음을 보여준다.

superpixel projection을 적용한 뒤에도 제안 방법은 60.3% / 40.2%로 Hypercolumn의 58.6% / 36.4%보다 낫다. 여기에 rescoring까지 더하면 최종적으로 63.6% / 43.3%에 도달한다. 흥미로운 점은 저자들이 제안 방법이 후처리에 덜 의존한다고 해석한다는 것이다. 실제로 baseline은 superpixel projection으로 상당한 개선을 보이지만, 제안 방법은 raw prediction이 이미 비교적 coherent하기 때문에 추가 개선 폭이 더 작다. 이는 모델이 구조를 내부적으로 더 잘 학습했다는 논문의 주장과 연결된다.

정성적 분석도 논문의 중요한 부분이다. Figure 3에서는 iterative training stage가 진행될수록 heatmap quality가 점진적으로 좋아지는 모습을 보여준다. 초기에는 object의 일부 part만 찾다가, 후반 stage에서는 빠졌던 부위를 채우고 다른 인스턴스의 일부를 억제하는 식으로 개선된다. 저자들은 bicycle, horse 예시에서 pole이나 인접한 horse 부분이 후반 단계에서 점차 suppress되는 현상을 언급한다.

Figure 4와 추가 시각화에서는 제안 방법의 heatmap이 Hypercolumn보다 더 visually coherent하다고 주장한다. 예를 들어 person 계열 이미지에서는 noise가 적고 foreground 영역을 더 일관되게 채운다. bicycle 이미지에서는 contour를 더 잘 따라가고, horse 이미지에서는 body와 legs를 더 잘 복원한다. 특히 occluded horse head를 hallucinate하는 사례를 두고, 저자들은 이것이 shape prior 학습의 간접 증거라고 해석한다. 말 머리가 실제로 보이지 않아도 “머리 없는 말”은 덜 plausible하기 때문에 머리 위치를 보완적으로 예측한다는 것이다. 물론 이는 동시에 잘못된 hallucination이기도 하다.

이 논문은 detection 단위에서도 분석을 제공한다. Figure 6에서 top 200 detections per category에 대해 Hypercolumn prediction과 제안 방법 prediction의 ground-truth overlap을 비교한 결과, 76%의 detection에서 overlap이 개선되고 15.6%에서 감소하며 나머지는 동일했다고 보고한다. 특히 baseline이 이미 약 75% overlap을 달성한 좋은 예측에서도 추가로 15% 정도 향상되는 경우가 있다고 한다. 저자들은 이것이 구조를 사용하지 않고는 설명하기 어려운 개선이라고 본다.

supplementary의 per-category 성능을 보면 거의 모든 category에서 개선이 관찰된다. 특히 70% overlap raw prediction에서 bike는 18.6%에서 31.5%, bird는 23.2%에서 42.0%, horse는 20.7%에서 39.6%, person은 15.6%에서 32.5%로 크게 오른다. 이는 복잡한 shape나 서로 인접한 instance가 많은 category에서 iterative correction의 이점이 더 크게 나타날 수 있음을 시사한다. 다만 모든 category에서 항상 일관되게 큰 개선이 나는 것은 아니며, 예를 들어 일부 설정에서 aero, cat, sofa 등은 미세한 하락이나 거의 유사한 결과도 존재한다.

마지막으로 저자들은 shape prior를 좀 더 직접적으로 확인하려는 실험도 수행한다. 시각적으로 특징이 거의 없는 patch를 입력한 뒤 arbitrary category label을 주면, bird일 때는 bird body와 wing 비슷한 shape, horse일 때는 horse body와 legs 비슷한 shape, bicycle일 때는 frame 비슷한 구조, TV일 때는 큰 box형 구조가 heatmap으로 나타난다. 논문은 이를 category-conditioned plausible shape를 hallucinate할 수 있다는 증거로 제시한다. 이 실험은 엄밀한 정량 검증이라기보다는 qualitative evidence에 가깝지만, 논문의 핵심 주장과는 잘 맞는다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 structured prediction을 직접 공식화하지 않고도 구조를 학습할 수 있다는 점을 설득력 있게 보여준다는 것이다. shape, contiguity, contour smoothness 같은 요소를 별도의 handcrafted constraint 없이 데이터에서 끌어낸다는 문제 설정 자체가 명확하고, instance segmentation이라는 적절한 testbed에서 그 효과를 잘 검증했다. 특히 raw pixel-wise prediction 자체가 좋아졌다는 결과는 후처리 편법이 아니라 모델 내부 표현이 실제로 개선되었음을 뒷받침한다.

또 다른 강점은 방법이 개념적으로 단순하다는 점이다. 기존 CNN segmentation 모델에 이전 prediction을 추가 입력으로 넣고 multi-stage training/testing을 수행하는 형태이므로, 복잡한 approximate inference나 특수한 energy function 설계 없이 구현할 수 있다. 논문은 이를 통해 explicit structured model 대비 실용성과 확장 가능성을 강조한다.

실험 설계 측면에서도 강점이 있다. 전체 mAP 비교뿐 아니라 raw prediction, superpixel projection, rescoring을 분리해 비교했고, detection-level overlap 분석과 category-conditioned hallucination 실험까지 제시했다. 이를 통해 “정확히 어디서 좋아졌는가”를 비교적 다층적으로 설명한다.

반면 한계도 분명하다. 첫째, 논문은 shape prior를 “학습했다”는 주장을 주로 qualitative evidence와 간접적 성능 향상으로 뒷받침한다. 실제로 어떤 internal representation이 shape prior를 담고 있는지, 그리고 그것이 contiguity나 smoothness와 어떻게 분리되는지는 명확히 분석하지 않는다. 즉 구조를 배웠다는 결론은 설득력 있지만 완전히 분해 가능한 방식으로 입증된 것은 아니다.

둘째, detection system에 강하게 의존한다. 이 논문은 fast R-CNN detection 결과를 입력으로 쓰며, segmentation only 성능을 개선하는 구조다. 따라서 detection box가 크게 틀리면 segmentation도 근본적으로 한계를 가진다. 논문도 bounding box localization 품질이 좋은 경우 더 큰 이득이 나타난다고 암시한다.

셋째, iterative procedure의 계산 비용과 stage 설계는 추가 부담이다. 4-stage training과 3-step inference를 수행하므로 단일 forward 방식보다 학습과 추론이 더 무겁다. 논문은 이 비용을 자세히 분석하지 않는다. 또한 왜 4-stage가 적절한지, 더 많은 단계에서 어떤 trade-off가 있는지에 대한 체계적 ablation은 충분히 제시되지 않는다.

넷째, 이 접근이 다른 pixel-wise labeling task로 일반화된다고 결론에서 말하지만, 실제 논문 본문에서 실험은 instance segmentation에만 집중되어 있다. 따라서 semantic segmentation, depth estimation, medical segmentation 등 다른 structured output task로의 일반성은 논문이 직접 증명한 사실이 아니라 저자들의 기대에 가깝다.

비판적으로 보면, 이 방법은 today’s 관점에서는 recurrent refinement 또는 iterative mask refinement 계열의 초기 형태로 읽힌다. 당시로서는 explicit structure 없이 refinement만으로 shape prior를 얻는다는 점이 강했지만, 구조 학습의 해석 가능성은 제한적이고, end-to-end detector-segmenter 통합이 아니라는 점에서 이후 등장한 stronger architectures와 비교하면 구조적 한계도 있다. 그럼에도 이 논문은 “출력 구조를 직접 수식화하지 않고, correction dynamics를 학습해 구조를 우회적으로 얻는다”는 사고방식을 분명하게 제시했다는 점에서 의미가 크다.

## 6. 결론

이 논문은 instance segmentation에서 structured prediction을 직접 풀지 않고, 반복적 refinement로 환원하는 접근을 제안했다. 모델은 이미지와 이전 단계의 prediction을 함께 입력받아 오류를 점진적으로 수정하고, 그 과정에서 category-specific shape prior, region contiguity prior, contour smoothness prior를 암묵적으로 학습한다. 실험적으로는 PASCAL VOC 2012 validation에서 mean $AP^r$ 63.6% at 50% overlap, 43.3% at 70% overlap을 달성해 당시 최고 성능을 기록했다.

이 연구의 중요한 의미는, 복잡한 구조를 명시적으로 설계하지 않아도 iterative correction 메커니즘만으로 충분히 강한 structured output behavior를 유도할 수 있음을 보여줬다는 데 있다. 실제 응용 측면에서는 detection 이후 정교한 mask refinement가 필요한 다양한 vision task에 아이디어를 확장할 수 있고, 향후 연구 측면에서는 recurrent refinement, iterative inference, learned priors를 활용한 segmentation 계열 연구의 중요한 연결고리로 볼 수 있다.
