# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction

- **저자**: Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
- **발표연도**: 2024 (제공된 텍스트의 arXiv v6 기준, 2024-02-06)
- **arXiv**: https://arxiv.org/abs/2205.14756

## 1. 논문 개요

이 논문은 고해상도 dense prediction을 더 빠르고 실용적으로 만들기 위한 비전 모델인 **EfficientViT**를 제안한다. dense prediction은 semantic segmentation, super-resolution, promptable segmentation 같은 작업처럼 이미지의 각 위치에 대해 세밀한 예측을 해야 하므로, 입력 해상도가 높고 넓은 문맥 정보를 잘 활용해야 한다. 문제는 이런 조건 때문에 기존의 고성능 모델들이 계산량과 지연시간이 매우 커져서, 모바일 CPU, edge GPU, cloud GPU 같은 실제 하드웨어에서 배포하기 어렵다는 점이다.

저자들은 기존 접근들의 병목을 명확히 짚는다. Transformer 계열은 global receptive field를 얻기 위해 softmax attention을 쓰지만, 이는 입력 해상도에 대해 계산량이 quadratic하게 증가한다. 반면 CNN 계열 일부 모델은 large-kernel convolution이나 복잡한 multi-branch 구조로 receptive field와 multi-scale feature를 확보하려 하지만, 이런 연산은 실제 하드웨어에서 항상 효율적으로 구현되지 않는다. 즉, 기존 모델들은 정확도를 위해 중요한 성질은 얻었지만, 실제 배포 효율은 충분히 고려하지 못했다는 것이 이 논문의 문제의식이다.

이 논문의 핵심 목표는 **global receptive field**와 **multi-scale learning**이라는 두 가지 중요한 능력을 유지하면서도, 이를 **hardware-efficient operation**만으로 구현하는 것이다. 이를 통해 FLOPs 감소가 단순한 이론적 수치에 그치지 않고 실제 latency 감소로 이어지도록 설계했다는 점이 이 연구의 실용적 의미다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 softmax attention을 그대로 쓰지 않고, 대신 **ReLU linear attention**을 기반으로 한 **multi-scale linear attention**을 설계하는 것이다. 저자들은 고해상도 dense prediction에서 중요한 것은 단순히 attention을 쓰는 것이 아니라, 넓은 문맥을 보는 능력과 여러 크기의 구조를 함께 이해하는 능력이라고 본다. 따라서 attention을 가볍게 만드는 것만으로는 부족하고, local information과 multi-scale information의 부족까지 함께 보완해야 한다고 주장한다.

이를 위해 제안된 모듈은 두 방향으로 ReLU linear attention의 약점을 보완한다. 첫째, FFN 내부에 **depthwise convolution**을 넣어서 local feature extraction 능력을 높인다. 둘째, attention을 수행하기 전에 Q/K/V 토큰 주변의 정보를 small-kernel depthwise-separable convolution으로 집계해 **multi-scale tokens**를 만든 뒤, 여기에 linear attention을 적용한다. 이렇게 하면 attention은 global context를 제공하고, convolution은 local structure와 multi-scale pattern을 보강하게 된다.

기존 방법과의 차별점은 명확하다. SegFormer는 global context는 잘 얻지만 softmax attention의 비용이 비싸고, SegNeXt는 multi-scale receptive field를 얻지만 large-kernel convolution이 하드웨어 친화적이지 않다. EfficientViT는 이 둘의 장점을 취하되, softmax나 대형 커널 같은 비효율적 연산을 피해서 실제 장비에서 빠르게 동작하도록 만들었다. 저자들은 이것이 high-resolution dense prediction에 linear attention을 성공적으로 적용한 첫 사례라고 주장한다.

## 3. 상세 방법 설명

EfficientViT의 기본 building block은 두 부분으로 구성된다. 하나는 **multi-scale linear attention**이고, 다른 하나는 **FFN+DWConv**이다. 전자는 문맥 정보와 전역 수용영역을 담당하고, 후자는 지역 정보를 보강한다. 즉, attention만으로 모든 것을 해결하려 하지 않고, attention과 convolution의 역할을 분리해 서로 보완하도록 설계했다.

먼저 일반적인 attention은 입력 $x \in \mathbb{R}^{N \times f}$로부터
$Q = xW_Q$, $K = xW_K$, $V = xW_V$
를 만든 뒤, 각 query 위치 $i$에 대해 유사도 기반 가중합을 계산한다. 논문은 이를 다음과 같이 일반형으로 쓴다.

$$
O_i = \frac{\sum_{j=1}^{N} \mathrm{Sim}(Q_i, K_j)V_j}{\sum_{j=1}^{N} \mathrm{Sim}(Q_i, K_j)}
$$

여기서 softmax attention은 보통
$\mathrm{Sim}(Q, K) = \exp(QK^T / \sqrt{d})$
를 사용한다. 하지만 이 방식은 모든 query-key 쌍을 고려해야 하므로, 해상도가 커질수록 계산량과 메모리 사용량이 빠르게 증가한다.

논문은 대신 **ReLU linear attention**을 사용한다. 이때 similarity는 다음과 같이 정의된다.

$$
\mathrm{Sim}(Q, K) = \mathrm{ReLU}(Q)\mathrm{ReLU}(K)^T
$$

이 정의를 쓰면 attention 출력은 다음처럼 재구성된다.

$$
O_i =
\frac{
\mathrm{ReLU}(Q_i)\left(\sum_{j=1}^{N}\mathrm{ReLU}(K_j)^T V_j\right)
}{
\mathrm{ReLU}(Q_i)\left(\sum_{j=1}^{N}\mathrm{ReLU}(K_j)^T\right)
}
$$

이 식의 의미는 중요하다. query마다 모든 key와 다시 곱하는 대신,
$\sum_j \mathrm{ReLU}(K_j)^T V_j$ 와
$\sum_j \mathrm{ReLU}(K_j)^T$
를 한 번만 계산해 재사용할 수 있다. 그래서 attention의 연산량과 메모리 사용량을 quadratic에서 linear 수준으로 줄일 수 있다. 논문은 이것이 단지 이론적으로만 가벼운 것이 아니라, softmax 같은 hardware-unfriendly operation을 제거하기 때문에 실제 모바일 CPU에서도 더 빠르다고 강조한다.

하지만 ReLU linear attention에는 분명한 한계가 있다. softmax attention처럼 sharp한 attention distribution을 만들기 어렵기 때문에, 주변의 미세한 local structure를 포착하는 능력이 약하다. 논문 Figure 3도 이 점을 시각적으로 보여준다. 저자들은 이 한계를 두 가지 장치로 보완한다.

첫째, **FFN에 depthwise convolution을 삽입**한다. 이렇게 하면 attention이 놓치기 쉬운 local pattern을 FFN 경로에서 보강할 수 있다. 즉, attention은 context, convolution은 locality를 담당한다.

둘째, **multi-scale token aggregation**을 도입한다. 선형 투영으로 Q/K/V를 만든 뒤, 각 head별로 Q/K/V 주변 토큰들을 small-kernel depthwise-separable convolution으로 집계해 여러 스케일의 토큰을 생성한다. 논문 설명에 따르면 실제 구현에서는 독립적으로 여러 convolution을 수행하는 대신, GPU 효율을 위해 이를 **group convolution**으로 묶는다. 모든 DWConv는 하나의 DWConv로 합치고, 1x1 convolution은 group 수가 $3 \times \#heads$ 인 단일 1x1 group convolution으로 결합한다. 이렇게 얻은 multi-scale Q/K/V에 대해 ReLU linear attention을 수행한 뒤, head 차원으로 concat하고 최종 linear projection으로 feature를 융합한다.

아키텍처 차원에서 EfficientViT는 표준적인 backbone-head, 또는 encoder-decoder 구조를 따른다. backbone은 input stem과 4개의 stage로 구성되며, stage가 깊어질수록 공간 해상도는 줄고 채널 수는 늘어난다. EfficientViT module은 **Stage 3과 Stage 4**에 삽입된다. downsampling에는 stride 2의 MBConv를 사용한다. head는 마지막 세 단계의 출력인 $P2$, $P3$, $P4$를 사용하여 feature pyramid를 만들고, 1x1 convolution과 bilinear/bicubic upsampling으로 공간 크기와 채널 수를 맞춘 후 **addition**으로 융합한다. 이후 몇 개의 MBConv와 출력층으로 최종 예측을 만든다. 저자들은 backbone 자체가 이미 문맥 정보를 충분히 잘 추출하므로, head는 단순한 구조로도 충분하다고 본다.

모델 크기는 여러 효율 제약을 만족하도록 **EfficientViT-B0, B1, B2, B3**, 그리고 cloud 환경용 **EfficientViT-L 시리즈**로 구성된다. 논문 본문에는 구체적인 각 layer 설정표가 모두 들어 있지 않고, 세부 configuration은 공식 GitHub에 있다고 명시한다. 따라서 세부 폭/깊이 파라미터는 제공된 텍스트만으로는 완전히 복원할 수 없다.

## 4. 실험 및 결과

실험은 semantic segmentation, super-resolution, Segment Anything, ImageNet classification까지 폭넓게 수행된다. semantic segmentation은 **Cityscapes**와 **ADE20K**, lightweight super-resolution은 **DIV2K 학습 / BSD100 평가**, high-resolution super-resolution은 **FFHQ 일부 학습/평가 split**, classification은 **ImageNet**을 사용한다. latency는 모바일 CPU에서는 Qualcomm Snapdragon 8 Gen 1과 TensorFlow Lite, edge/cloud GPU에서는 TensorRT와 fp16으로 측정하며, 데이터 전송 시간까지 포함했다고 명시한다.

가장 먼저 중요한 ablation 결과는 Table 1이다. Cityscapes에서 multi-scale learning만 쓰거나 global attention만 쓰는 경우 각각 mIoU가 72.3, 72.2이고, 둘 다 없는 경우 68.1이다. 반면 **둘 다 사용할 때 74.5 mIoU**를 얻는다. MACs와 파라미터 수는 동일 수준으로 맞췄기 때문에, 이 결과는 논문이 주장하는 두 요소가 실제로 모두 필요하다는 근거가 된다.

ImageNet classification 결과에서도 backbone의 품질이 확인된다. 예를 들어 **EfficientViT-L2 (r384)**는 **Top-1 86.0%**를 달성하며, 논문은 이것이 EfficientNetV2-L 대비 **+0.3 accuracy gain**과 **A100에서 2.6배 speedup**이라고 설명한다. 이 결과는 EfficientViT가 dense prediction 전용 꼼수 모델이 아니라, backbone 자체의 일반 성능도 강하다는 점을 뒷받침한다.

semantic segmentation의 핵심 결과는 매우 인상적이다. Cityscapes에서 **EfficientViT-B3**는 **83.0 mIoU**, **179G MACs**, **Jetson AGX Orin 81.8ms**, **A100 70 image/s**를 기록한다. 비교 대상인 **SegNeXt-B**는 **82.6 mIoU**, **276G MACs**, **228ms**, **41 image/s**이고, **SegFormer-B5**는 **82.4 mIoU**, **1460G MACs**, **638ms**, **12 image/s**다. 즉, EfficientViT는 동급 또는 더 높은 정확도를 유지하면서 계산량과 latency를 크게 줄인다. 논문 초록과 본문은 Cityscapes에서 성능 손실 없이 SegFormer 대비 최대 **13.9배** 정도의 GPU latency reduction, SegNeXt 대비 최대 **6.2배** 수준의 개선을 강조한다.

ADE20K에서도 비슷한 경향이 유지된다. 예를 들어 **EfficientViT-B2**는 **45.9 mIoU**, **9.1G MACs**, **Orin 7.3ms**, **A100 846 image/s**를 보이며, **SegNeXt-S**의 **44.3 mIoU**, **16G MACs**, **17.2ms**, **592 image/s**보다 정확도와 효율 모두에서 우수하다. 상위 모델인 **EfficientViT-L2**는 **50.7 mIoU**로, **SegFormer-B4**의 **50.3 mIoU**보다 높으면서도 A100 latency가 **9.0ms 대 44.9ms**로 크게 낮다.

super-resolution 결과도 설계 의도를 잘 보여준다. FFHQ 512x512→1024x1024 설정에서 **EfficientViT w0.75**는 **43.54 PSNR**, **0.9809 SSIM**, **14.3ms**를 기록해 **Restormer**의 **43.43 PSNR**, **92.0ms**보다 정확도와 속도 모두 우수하다. 기본 **EfficientViT**는 **43.58 PSNR**, **17.8ms**로 Restormer 대비 **0.15 dB** 가까운 개선과 약 **5.2배 속도 향상**을 보인다. 논문 요약에서는 최대 **6.4배 speedup**과 **0.11 dB gain**을 강조한다. BSD100 lightweight SR에서도 EfficientViT는 ViT 기반 및 CNN 기반 SR 모델 대비 경쟁력 있는 PSNR을 유지하면서도 더 낮은 latency를 제공한다.

Segment Anything 확장도 이 논문의 중요한 실험이다. 저자들은 SAM의 image encoder를 EfficientViT로 대체한 **EfficientViT-SAM**을 제안하고, prompt encoder와 mask decoder는 유지한다. 학습은 2단계로 진행된다. 먼저 SAM image encoder를 teacher로 한 distillation 형태의 학습을 수행하고, 이후 SA-1B 전체 데이터셋으로 end-to-end 학습한다. COCO/LVIS zero-shot instance segmentation에서 **EfficientViT-SAM-XL1**은 **COCO mAP 47.8**, **LVIS mAP 44.4**, **A100 throughput 182 image/s**를 기록한다. 반면 **SAM-ViT-H**는 **46.5 / 44.2 / 11 image/s**다. 즉, 정확도는 약간 더 높거나 비슷하면서 throughput은 약 **16.5배** 높다. 논문 초록에서는 A100 기준 **48.9배 더 높은 throughput**도 언급하는데, 이는 비교 설정이나 모델군 차이에 따른 최대치로 이해해야 한다.

point-prompted segmentation에서는 **EfficientViT-SAM-XL1**이 COCO에서 1/3/5 click 기준 각각 **59.8 / 71.3 / 75.3**을 기록해 SAM-ViT-H의 **58.4 / 69.6 / 71.4**보다 좋다. LVIS에서는 3-click, 5-click에서는 더 좋지만, **1-click에서는 SAM-ViT-H가 더 높다**. 저자들도 이 부분을 인정하며, interactive segmentation setup이 end-to-end training에 충분히 반영되지 않았을 가능성을 언급한다.

정성적 결과에 대해서는 Cityscapes 시각화에서 EfficientViT가 baseline보다 **경계(boundary)** 와 **작은 객체(small objects)** 를 더 잘 인식한다고 설명한다. 다만 제공된 텍스트에는 개별 사례 이미지가 포함되어 있지 않아, 어떤 장면에서 어떤 class가 개선되었는지까지는 구체적으로 재현할 수 없다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **정확도-효율 trade-off를 논리적으로 설계하고, 이를 실제 하드웨어 측정으로 설득력 있게 입증했다는 점**이다. 많은 효율 모델 논문이 FLOPs만 줄였다고 주장하지만, 실제 장비 latency가 기대만큼 줄지 않는 경우가 많다. 반면 이 논문은 softmax, large-kernel convolution 같은 연산이 왜 실제 배포에서 불리한지 설명하고, TensorRT, TensorFlow Lite, 모바일 CPU, edge GPU, cloud GPU에서의 결과를 함께 제시한다. 즉, “이론적으로 가볍다”가 아니라 “실제로 빠르다”를 보여준다.

두 번째 강점은 **문제 정의가 명확하고 설계가 일관적**이라는 점이다. 저자들은 dense prediction에 필요한 핵심 요소를 global receptive field와 multi-scale learning으로 정리하고, 각 요소를 어떤 모듈이 담당하는지 분명하게 설계했다. ReLU linear attention 하나만 쓰면 local 정보가 약해진다는 약점을 숨기지 않고, FFN의 DWConv와 multi-scale token aggregation으로 이를 보완했다는 점도 설득력이 있다. ablation study 역시 이 설계 선택이 임의적이지 않음을 보여준다.

세 번째 강점은 **적용 범위가 넓다**는 점이다. semantic segmentation뿐 아니라 super-resolution, Segment Anything, ImageNet classification까지 확장해 backbone과 attention 모듈의 범용성을 보여준다. 특히 SAM 가속 결과는 단순한 논문 벤치마크를 넘어, 실제로 대형 foundation model의 encoder를 더 효율적으로 바꿀 수 있다는 점에서 응용 가치가 크다.

한편 한계도 있다. 먼저, ReLU linear attention은 본질적으로 softmax attention보다 sharp한 attention map을 만들기 어렵고, 저자들 역시 local information extraction 약점을 직접 인정한다. 이 논문은 convolution으로 이를 보완하지만, 이는 곧 attention만으로 문제를 해결한 것이 아니라 attention-convolution hybrid 구조에 의존한다는 뜻이기도 하다. 다시 말해, linear attention 자체의 표현력 한계는 여전히 구조적 제약으로 남아 있다.

또한 논문은 여러 결과를 제시하지만, 일부 구현 세부사항은 본문보다 GitHub에 의존한다. 예를 들어 각 모델 변형의 상세한 layer 수, channel 설정, scaling 규칙은 제공된 텍스트만으로 완전히 파악되지 않는다. 재현성 측면에서 핵심 아이디어는 충분히 전달되지만, 완전한 reproduction을 위해서는 추가 자료가 필요하다.

실험 해석에서도 약간의 주의가 필요하다. 논문은 다양한 플랫폼에서 speedup을 강조하지만, 모든 baseline이 동일 수준으로 최적화되었는지, 각 모델이 vendor-specific optimization의 혜택을 동일하게 받았는지까지는 제공된 텍스트만으로 완전히 판단하기 어렵다. 또한 SAM 실험에서 single-point LVIS 성능 저하가 나타난 만큼, interactive setting 전반에서 일관적으로 우수한지는 추가 검증이 필요하다. 저자들도 이 부분을 미해결 과제로 남긴다.

비판적으로 보면, 이 논문은 “dense prediction에 필요한 핵심 요소를 효율적으로 구현하는 방법”에는 매우 강하지만, 왜 특정 linear attention 변형이 다른 선형 attention들보다 더 적합한지에 대한 비교는 제한적이다. 논문은 ReLU linear attention을 사용하고 그 장점을 설명하지만, Performer, Linformer 등 다른 linear attention 계열과의 직접 비교는 제공된 본문 기준으로는 충분히 자세하지 않다. 따라서 제안 방식의 상대적 우수성이 “softmax attention이나 대형 커널 기반 구조 대비 우수하다”는 수준에서는 강하게 입증되지만, “모든 linear attention 중 최선이다”까지는 말하기 어렵다.

## 6. 결론

이 논문은 고해상도 dense prediction을 위한 효율적 비전 모델 설계 문제를 다루며, 핵심 기여로 **multi-scale linear attention**과 이를 기반으로 한 **EfficientViT** 아키텍처를 제안한다. 제안된 방법은 ReLU linear attention으로 global receptive field를 확보하고, small-kernel convolution 기반의 multi-scale token aggregation과 FFN 내 DWConv로 local 및 multi-scale 정보를 보완한다. 그 결과 semantic segmentation, super-resolution, Segment Anything, ImageNet classification 등 여러 작업에서 강한 성능과 높은 실제 실행 효율을 동시에 보여준다.

실제 적용 측면에서 이 연구의 의미는 크다. 고해상도 입력을 다루는 모델은 자율주행, 의료영상, 모바일 비전, 생성형 비전 시스템 등에서 중요하지만, 배포 비용이 늘 병목이었다. EfficientViT는 이런 문제에 대해 “성능을 조금 포기하는 경량화”가 아니라, **구조 자체를 하드웨어 친화적으로 바꿔 성능과 속도를 함께 확보하는 방향**을 제시한다. 향후 연구에서는 다른 vision task로의 확장, 더 큰 스케일의 모델 설계, 그리고 interactive segmentation 같은 세부 설정에서의 추가 개선이 중요한 후속 과제가 될 것이다.
