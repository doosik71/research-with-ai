# Building Segmentation through a Gated Graph Convolutional Neural Network with Deep Structured Feature Embedding

- **저자**: Yilei Shi, Qingyu Li, Xiao Xiang Zhu
- **발표연도**: 2019
- **arXiv**: https://arxiv.org/abs/1911.03165

## 1. 논문 개요

이 논문은 광학 위성영상에서 건물 footprint를 더 정확하게 분할하기 위한 semantic segmentation 프레임워크를 제안한다. 저자들이 집중한 핵심 문제는 단순히 건물 여부를 맞히는 것이 아니라, 건물의 경계를 얼마나 또렷하고 정밀하게 복원할 수 있는가이다. 기존의 deep convolutional neural network(DCNN)는 픽셀 단위 분류 성능은 높아졌지만, down-sampling과 큰 receptive field의 영향으로 경계가 흐릿하고 덩어리처럼 뭉개진 예측을 내놓는 경우가 많다.

논문은 이 문제를 해결하기 위해 두 가지를 결합한다. 첫째는 서로 다른 수준의 feature를 결합하는 deep structured feature embedding(DSFE)이고, 둘째는 픽셀 간 관계를 그래프 형태로 다루는 gated graph convolutional neural network(GGCN)이다. 저자들의 주장은, 저수준 feature의 위치 정보와 고수준 feature의 의미 정보를 함께 쓰고, 여기에 graph 기반의 message passing을 더하면 건물 경계와 구조를 더 정밀하게 복원할 수 있다는 것이다.

이 문제는 원격탐사에서 매우 중요하다. 건물 footprint 추출은 도시계획, 인구 분석, 환경 모니터링, 자율주행, virtual reality 등 다양한 응용의 기초 데이터가 되기 때문이다. 특히 대규모 지역을 수작업으로 라벨링하는 것은 매우 비용이 크므로, 자동화된 고정밀 분할 기술의 가치가 크다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 “CNN만으로는 부족한 경계 정밀도와 픽셀 간 상호작용을 graph propagation으로 보완하자”는 것이다. 구체적으로는, DCNN에서 나온 multi-level feature를 DSFE로 결합한 뒤, 이를 그래프의 node feature로 보고 GGCN을 통해 주변 픽셀과 멀리 떨어진 픽셀의 정보를 반복적으로 전달한다.

기존 접근과 비교하면 차별점은 두 층위에서 드러난다. 하나는 feature 표현 측면이다. 논문은 semantic segmentation에서 low-level feature는 spatial detail이 풍부하고, high-level feature는 semantic information이 강하므로 둘을 구조적으로 결합해야 한다고 본다. 다른 하나는 graph inference 측면이다. CRF나 일반 GCN은 경계 보정에 도움을 줄 수 있지만, 정보 전달 범위나 방식에 한계가 있다고 본다. 이에 저자들은 GCN으로 local neighborhood 정보를 모으고, GRU 기반 recurrent propagation으로 long-range dependency까지 반영하는 GGCN을 제안한다.

즉, 이 방법은 짧은 거리의 구조적 관계와 긴 거리의 문맥 정보를 동시에 모델링하려는 설계라고 볼 수 있다. 논문은 이러한 결합이 coarse한 segmentation을 sharper한 boundary와 더 완전한 building mask로 개선한다고 주장한다.

## 3. 상세 방법 설명

전체 파이프라인은 크게 전처리, DSFE, GGCN, 그리고 최종 분류의 순서로 이루어진다. 입력 영상은 먼저 전처리를 거친 뒤 DCNN 기반 feature extractor로 들어간다. 여기서 얻은 여러 수준의 feature를 결합해 각 픽셀의 embedding vector를 만들고, 이를 그래프의 node hidden state 초기값으로 사용한다. 이후 GGCN이 여러 time step에 걸쳐 message passing을 수행하고, 마지막 hidden state로부터 각 픽셀의 class probability를 예측한다.

논문은 이미지를 그래프로 본다. 각 픽셀은 하나의 node이고, 2차원 grid 상의 인접 픽셀들이 edge로 연결된다. 따라서 segmentation은 사실상 node classification 문제가 된다.

### Deep Structured Feature Embedding

DSFE의 목적은 서로 다른 수준의 CNN feature를 결합해 더 풍부한 pixel representation을 만드는 것이다. 논문은 low-level feature는 위치와 경계 정보를 잘 보존하지만 semantic 정보가 약하고, high-level feature는 그 반대라고 설명한다. 따라서 두 종류의 feature는 상보적이다. 저자들은 여러 수준의 feature를 progressive하게 concatenate하여 localization, semantics, 기타 속성을 함께 전달하도록 설계한다.

실험에서는 U-Net, FCN-8s, FC-DenseNet을 feature extractor 후보로 비교했다. 그 결과 FC-DenseNet이 DSFE의 backbone으로 가장 좋은 결과를 냈다. 논문은 그 이유를 DenseNet의 feature reuse, iterative concatenation, 향상된 gradient propagation에서 찾고 있다.

### Graph Convolution 부분

논문은 먼저 graph Laplacian과 spectral graph convolution 배경을 설명한 뒤, Kipf and Welling의 단순화된 GCN layer를 사용한다. 그래프의 adjacency matrix를 $A$, degree matrix를 $D$라고 할 때 graph Laplacian은

$$
L = D - A
$$

로 정의된다.

이후 Chebyshev polynomial 근사를 거쳐, 최종적으로 단일 graph convolution layer를 다음과 같이 쓴다.

$$
H_i^r = \sigma_r \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} W H_i^{r-1} \right)
$$

여기서 $\tilde{A} = A + I$는 self-connection을 포함한 adjacency matrix이고, $\tilde{D}$는 그에 대응하는 degree matrix이다. $W$는 학습 가능한 가중치이며, $\sigma_r(\cdot)$는 비선형 활성화 함수이다. 이 식의 의미는 각 node가 자기 자신과 이웃 node의 표현을 정규화된 방식으로 모아 새로운 hidden representation을 만드는 것이다.

### Gated Graph Convolutional Neural Network

핵심 제안은 GCN만 쓰지 않고, GCN의 message aggregation과 GRU의 gating mechanism을 결합한 GGCN이다. 기본 propagation은 다음과 같이 쓴다.

$$
a_i^t = M(h_j^{t-1} \mid j \in V_i)
$$

$$
h_i^t = F(h_i^{t-1}, a_i^t)
$$

여기서 $a_i^t$는 이웃으로부터 모인 message이고, $h_i^t$는 time step $t$에서의 hidden state이다.

논문은 message function $M$으로 GCN을 사용한다. 즉, 특정 시점의 이웃 정보 집계는

$$
a_i^t = \sigma_r \left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} W h_i^{t-1} \right)
$$

로 계산한다.

그 다음 hidden state 업데이트는 GRU와 유사한 방식으로 이루어진다.

$$
r_i^t = \sigma_s(W_r h_i^{t-1} + U_r a_i^t)
$$

$$
z_i^t = \sigma_s(W_z h_i^{t-1} + U_z a_i^t)
$$

$$
\tilde{h}_i^t = \tanh \left( W(r_i^t \circ h_i^{t-1}) + U a_i^t \right)
$$

$$
h_i^t = (1 - z_i^t) \circ h_i^{t-1} + z_i^t \circ \tilde{h}_i^t
$$

여기서 $r_i^t$는 reset gate, $z_i^t$는 update gate, $\circ$는 element-wise product를 뜻한다. $\sigma_s$는 sigmoid 함수이고, $\sigma_r$는 ReLU 함수이다. 이 구조를 통해 node는 자신의 이전 상태를 유지할지, 이웃에서 온 정보를 얼마나 반영할지를 조절할 수 있다. 논문은 이것이 vanilla GCN이 잘 다루지 못하는 long-range dependency를 포착하는 데 도움이 된다고 설명한다.

직관적으로 보면, GCN이 “주변 정보를 모으는 단계”라면, GRU는 “그 정보를 지금 상태에 얼마나 반영할지 결정하는 단계”이다. 이를 여러 step 반복하면, 가까운 이웃뿐 아니라 더 멀리 떨어진 문맥도 반영할 수 있다.

### 출력 표현과 손실 함수

이 논문은 단순한 binary building/non-building segmentation 대신, truncated signed distance map(TSDM)을 사용해 multi-label pixel labeling 문제로 바꾼다. signed distance는 건물 경계까지의 거리를 나타내며, 건물 내부는 양수, 외부는 음수로 표현된다. 논문에서 truncated signed distance function은 다음과 같이 주어진다.

$$
D(x) = \delta_d \cdot \min \left( \min_{x \in X}(d(x)), T_d \right)
$$

여기서 $d(x)$는 픽셀 $x$에서 가장 가까운 건물 경계까지의 Euclidean distance이고, $\delta_d$는 건물 내부/외부를 나타내는 sign function, $T_d$는 truncation threshold이다. 저자들은 이 표현이 단순 class label뿐 아니라 geometry 정보를 학습에 포함하게 해 준다고 본다.

최종 예측은 마지막 time step의 hidden state에 softmax를 적용해 얻는다.

$$
p = \text{softmax}(h_i^{t+n})
$$

손실 함수는 negative log-likelihood loss(NLLLoss)이다. 본문에 class imbalance 처리나 추가적인 regularization 항은 명시되어 있지 않다.

### 전처리

Planetscope 실험에서는 전처리가 중요한 부분이다. 논문은 네 단계 전처리를 제안한다.

첫째, band normalization을 수행한다. 둘째, satellite imagery와 OSM building footprint 사이의 misalignment를 보정하기 위해 coregistration을 수행한다. 이 과정은 RGB를 grayscale로 바꾸고, Gaussian gradient를 계산한 뒤, 영상 gradient magnitude와 building footprint 사이의 cross-correlation을 구해 최대 상관 위치로 offset을 추정하는 방식이다. 셋째, refinement 단계가 있다. 다만 제공된 추출문에서는 refinement의 세부 구현은 충분히 설명되지 않았다. 넷째, TSDM을 생성한다.

저자들은 특히 medium-resolution 위성영상에서는 OSM과 영상 간 위치 불일치가 학습 품질을 크게 떨어뜨릴 수 있으므로, coregistration이 중요하다고 본다.

## 4. 실험 및 결과

### 데이터셋과 설정

주요 실험 데이터는 Planetscope RGB 영상이다. 공간 해상도는 3m이고, Munich, Rome, Paris, Zurich 네 도시를 포함한다. 정답 building footprint는 OpenStreetMap에서 가져왔다. 영상은 $64 \times 64$ patch로 잘랐고, 한 방향 overlap은 19 pixel이다. 총 48,000개 patch를 만들었으며, 80%는 학습, 20%는 테스트에 사용했다. 학습과 테스트는 spatially separated 되어 있다고 명시되어 있다.

Planetscope 실험에서는 TSDM을 위해 11개 class를 사용했다. 값 범위는 $[0, 10]$이고 truncation threshold는 5이다. optimizer는 SGD, learning rate는 $10^{-4}$, loss는 NLLLoss이다. 구현은 PyTorch이고, GPU는 NVIDIA Tesla P100 16GB를 사용했다.

비교 대상은 FCN-32s, FCN-16s, ResNet-DUC, E-Net, SegNet, U-Net, FCN-8s, CWGAN-GP, FC-DenseNet, GCN, GraphSAGE, GGNN 등이다.

추가 검증은 ISPRS Potsdam 2D Semantic Labeling dataset에서도 수행했다. 여기서는 RGB 3채널만 사용했고 DSM은 사용하지 않았다. 각 타일은 $6000 \times 6000$ pixel, 해상도는 5cm이다. 6개 원래 class 중 building만 positive로 두고 나머지 5개 class를 non-building으로 묶었다. 학습용 16,000개 patch, 테스트용 3,573개 patch를 사용했다. 이 데이터셋에서는 ground truth alignment가 좋고 TSDM은 medium-resolution용이라고 판단하여 별도 preprocessing 없이 원본 optical image를 직접 입력했다고 적고 있다.

### Planetscope에서의 CNN baseline

기본 CNN 성능 비교에서는 FC-DenseNet이 가장 좋은 baseline이었다. 수치는 다음과 같다.

- FC-DenseNet: OA 0.8551, F1 0.6328, IoU 0.4628
- CWGAN-GP: OA 0.8483, F1 0.6268, IoU 0.4562
- FCN-8s: OA 0.8472, F1 0.6222, IoU 0.4513
- U-Net: OA 0.8412, F1 0.6043, IoU 0.4329

논문은 FCN-32s와 FCN-16s가 나쁜 이유를 지나치게 coarse한 high-level feature 위주이기 때문이라고 해석한다. U-Net과 FCN-8s는 low-level feature를 결합하기 때문에 더 좋고, FC-DenseNet은 dense connection과 skip connection 덕분에 spatial detail 복원이 가장 뛰어나다고 평가한다.

### DSFE에서 어떤 feature extractor가 좋은가

DSFE와 GCN을 결합했을 때 backbone별 성능 비교 결과는 다음과 같다.

- DSFE(U-Net)-GCN: OA 0.8396, F1 0.6258, IoU 0.4544
- DSFE(FCN-8s)-GCN: OA 0.8594, F1 0.6320, IoU 0.4611
- DSFE(FC-DenseNet)-GCN: OA 0.8640, F1 0.6677, IoU 0.5012

즉, FC-DenseNet을 DSFE backbone으로 사용하는 것이 가장 좋았다. baseline FC-DenseNet의 IoU 0.4628이 DSFE(FC-DenseNet)-GCN에서 0.5012로 올라간다. 이는 graph propagation이 단순 CNN 위에 추가적인 정제 효과를 준다는 근거로 제시된다.

### 그래프 모델 비교

Planetscope에서 FC-DenseNet 기반 DSFE에 여러 graph model을 붙인 결과는 다음과 같다.

- FC-DenseNet: OA 0.8551, F1 0.6328, IoU 0.4628
- DSFE-CRF: OA 0.8592, F1 0.6415, IoU 0.4757
- DSFE-GCN: OA 0.8640, F1 0.6677, IoU 0.5012
- DSFE-GraphSAGE: OA 0.8719, F1 0.6726, IoU 0.5067
- DSFE-GGNN: OA 0.8787, F1 0.6778, IoU 0.5123
- DSFE-GGCN: OA 0.8881, F1 0.6899, IoU 0.5251

가장 중요한 결과는 제안한 DSFE-GGCN이 최고 성능을 보였다는 점이다. 논문은 IoU가 best DCNN 대비 6.2% 증가했다고 명시한다. 실제로 FC-DenseNet의 0.4628에서 0.5251로 상승했다. 시각화 결과에서도 yellow bounding box 영역에서 경계가 더 날카롭고 building shape의 completeness가 좋다고 주장한다.

### ISPRS Potsdam 추가 실험

고해상도 aerial imagery에서도 제안 방법은 유효했다. 표 4의 주요 수치는 다음과 같다.

- FC-DenseNet: OA 0.9186, F1 0.9182, IoU 0.8789
- DSFE-GCN: OA 0.9221, F1 0.9375, IoU 0.9097
- DSFE-GGCN: OA 0.9271, F1 0.9422, IoU 0.9196

여기서도 DSFE-GGCN이 가장 높은 성능을 보였다. 논문은 고해상도에서는 SegNet의 pooling index 재사용이 spatial information propagation에 더 도움이 되는 부분도 있다고 관찰하지만, graph model을 결합한 방법들이 CNN-only 방법보다 더 fine detail을 잘 잡는다고 해석한다. 특히 DSFE-GGCN은 completeness와 sharpness 모두에서 우수하다고 주장한다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 설정과 방법 설계가 서로 잘 맞물려 있다는 점이다. 논문은 semantic segmentation의 대표적 약점인 blurry boundary와 부족한 pixel interaction을 정확히 겨냥하고, 이를 multi-level feature fusion과 graph propagation의 결합으로 해결하려고 한다. 단순히 새로운 block 하나를 추가한 수준이 아니라, 표현 학습 단계와 구조적 추론 단계를 함께 설계했다는 점이 강점이다.

또 다른 강점은 GGCN의 설계 의도가 비교적 명확하다는 점이다. GCN이 local aggregation을, GRU가 memory와 long-range propagation을 맡는 구조는 직관적이며, 식으로도 잘 제시되어 있다. 실험도 단순 baseline 비교에 그치지 않고, 어떤 DCNN이 DSFE backbone으로 적절한지, 그리고 어떤 graph model이 더 적합한지를 단계적으로 검증한다. 이런 구성은 제안 요소 각각의 기여를 이해하는 데 도움이 된다.

전처리 측면에서도 실제 원격탐사 데이터의 문제를 잘 다룬다. OSM footprint와 위성영상의 misalignment를 보정하는 coregistration, 그리고 경계 기하 정보를 반영하는 TSDM은 실제 데이터 품질 문제를 정면으로 다루는 장치다. 특히 medium-resolution imagery에서 label noise를 줄이려는 시도는 실용성이 있다.

반면 한계도 분명하다. 첫째, 그래프 구성 방식이 매우 자연스러운 2D grid 기반이라는 점은 장점이기도 하지만, 이것이 일반 CNN과 비교해 어느 정도의 계산 비용 증가를 가져오는지 본문에서 충분히 정량화하지 않는다. end-to-end trainable이라고는 하지만, 학습 시간, 메모리 사용량, 추론 속도에 대한 비교는 제공된 본문에서 확인되지 않는다.

둘째, GGCN이 long-range dependency를 더 잘 포착한다고 주장하지만, propagation step 수나 그래프 연결 범위가 성능에 어떻게 영향을 주는지에 대한 ablation이 추출문에서는 보이지 않는다. 따라서 왜 GGCN이 정확히 더 좋은지에 대한 기전적 설명은 어느 정도 설득력 있지만, 세부 분석은 부족하다.

셋째, 전처리의 refinement 단계는 언급되지만 세부 방법이 충분히 설명되지 않는다. 또한 TSDM이 성능 향상에 얼마나 기여하는지, coregistration이 실제로 수치적으로 얼마나 개선하는지에 대한 독립적 분해 실험은 제공된 내용만으로는 확인하기 어렵다. 따라서 전체 성능 향상 중 어느 부분이 모델 구조 덕분이고 어느 부분이 데이터 정제 덕분인지는 완전히 분리해서 보기 어렵다.

넷째, 논문은 building footprint extraction 외의 다른 binary 또는 multi-label segmentation task에도 일반적으로 적용 가능하다고 주장한다. 그러나 제공된 실험은 사실상 건물 분할에 집중되어 있고, 도로 추출이나 다른 클래스에 대한 검증은 없다. 따라서 일반화 가능성은 합리적 추론일 수는 있어도, 논문이 충분히 실험적으로 입증했다고 보기는 어렵다.

비판적으로 보면, 이 논문은 “CNN의 coarse prediction을 graph reasoning으로 보정한다”는 방향에서 설득력 있는 결과를 제시하지만, 구조적 요소별 기여를 더 촘촘하게 분해했으면 더 강한 논문이 되었을 것이다. 그럼에도 불구하고, 실험 결과는 일관되게 개선을 보여 주며, 특히 medium-resolution 위성영상에서 경계 복원이 중요한 응용에는 의미 있는 성과로 보인다.

## 6. 결론

이 논문은 building footprint extraction을 위한 semantic segmentation 문제에서, multi-level CNN feature를 결합하는 DSFE와 graph-based contextual reasoning을 수행하는 GGCN을 통합한 새로운 end-to-end 프레임워크를 제안했다. 핵심 기여는 단순한 pixel-wise classification을 넘어서, 경계와 구조를 더 정밀하게 복원할 수 있는 segmentation 모델을 설계했다는 점이다.

실험적으로는 FC-DenseNet 기반 DSFE에 GGCN을 결합한 모델이 Planetscope와 ISPRS Potsdam 데이터셋 모두에서 가장 좋은 성능을 보였다. 특히 Planetscope에서는 기존 최고 DCNN 대비 IoU를 의미 있게 끌어올렸고, 시각적으로도 sharper boundary와 더 완전한 건물 형상을 보였다.

이 연구는 실제 원격탐사 응용에서 중요성이 크다. medium-resolution 위성영상처럼 경계 복원이 어려운 환경에서 효과적일 가능성이 높고, 저자들이 말하듯 road extraction이나 다른 semantic segmentation 문제로 확장될 여지도 있다. 다만 그 일반성은 향후 추가 실험으로 더 검증될 필요가 있다. 전체적으로 보면, 이 논문은 CNN 기반 분할의 한계를 graph-based propagation으로 보완하려는 방향을 설득력 있게 보여 준 연구라고 평가할 수 있다.
