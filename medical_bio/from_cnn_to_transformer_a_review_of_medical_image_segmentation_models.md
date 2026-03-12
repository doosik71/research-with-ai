# From CNN to Transformer: A Review of Medical Image Segmentation Models

## 논문 메타데이터

- **제목**: From CNN to Transformer: A Review of Medical Image Segmentation Models
- **저자**: Wenjian Yao, Zhaohui Jin, Yong Xia, Yanning Zhang
- **출판 연도**: 2023
- **형태**: arXiv review paper
- **arXiv ID**: 2308.05305
- **arXiv URL**: https://arxiv.org/abs/2308.05305
- **PDF URL**: https://arxiv.org/pdf/2308.05305v1

## 연구 배경 및 문제 정의

이 논문은 의료영상 분할 모델의 발전사를 `CNN 중심 시대`에서 `Transformer 중심 또는 hybrid 시대`로 넘어가는 흐름으로 정리한다. 저자들의 문제의식은 단순하다. 의료영상 분할은 장기, 조직, 병변을 정밀하게 구분해야 하며, 경계 복원과 전역 문맥 이해가 동시에 중요하다. 그런데 CNN과 Transformer는 이 두 요구를 서로 다른 방식으로 해결한다.

논문이 제시하는 핵심 배경은 다음과 같다.

- **CNN**은 local texture와 spatial inductive bias에 강하다.
- **Transformer**는 전역 관계와 장거리 의존성 모델링에 강하다.
- 의료영상은 2D뿐 아니라 3D volumetric 데이터가 많아 계산량과 메모리 제약이 크다.
- 데이터셋 규모가 자연영상보다 작고 annotation 비용이 높아, 대규모 모델이 항상 유리하지는 않다.

따라서 이 논문은 "Transformer가 CNN을 완전히 대체하는가"를 묻기보다, 어떤 문제 설정에서 CNN, hybrid, pure Transformer가 각각 어떤 강점과 한계를 가지는지 구조적으로 설명한다.

## 논문의 핵심 기여

이 survey의 가치는 전환기의 지형도를 비교적 명확하게 정리한 데 있다.

1. 의료영상 분할 모델을 `CNN-based`, `Transformer-based`, `CNN-Transformer hybrid` 흐름으로 나누어 정리했다.
2. CNN 구조의 핵심 설계 요소와 Transformer 구조의 핵심 설계 요소를 대조해 설명했다.
3. 장기, 병변, 기관별 segmentation 과제를 폭넓게 포괄하며 모델 흐름을 task 수준에서 연결했다.
4. 성능 향상만이 아니라 계산 비용, 데이터 요구량, 3D 적용성, 일반화 문제를 함께 논의했다.
5. 향후 의료영상 분할이 pure Transformer로 단순 이동하기보다 hybrid, efficient architecture, foundation-style pretraining으로 발전할 가능성을 시사했다.

## 방법론 구조 요약

## 1. CNN 기반 의료영상 분할

논문은 CNN 계열을 의료영상 분할의 출발점으로 둔다. 핵심은 encoder-decoder 구조와 local convolution을 기반으로 한 정밀 경계 복원이다.

### 1.1 대표 구조와 설계 요소

저자들이 CNN 계열의 공통 요소로 강조하는 것은 다음과 같다.

- fully convolutional design
- encoder-decoder architecture
- skip connection
- residual/dense connection
- multi-scale feature fusion
- dilated convolution
- attention-enhanced CNN block

특히 U-Net 계열은 의료영상 분할의 사실상 표준 베이스라인으로 다뤄진다. skip connection이 저수준 경계 정보와 고수준 semantic context를 연결해 작은 데이터셋에서도 안정적으로 동작하기 때문이다.

### 1.2 CNN의 강점

- 지역적 texture와 경계 복원에 강하다.
- inductive bias가 강해 적은 데이터에서도 비교적 잘 학습된다.
- 2D와 3D로 쉽게 확장된 변형이 많다.
- 구현과 최적화 경험이 풍부하다.

### 1.3 CNN의 한계

논문은 CNN의 한계를 주로 receptive field와 장거리 문맥 측면에서 설명한다.

- 멀리 떨어진 구조 간 관계를 직접 모델링하기 어렵다.
- 다중 장기나 복잡한 해부학 구조에서 global context 활용이 제한적이다.
- 깊이를 늘리거나 dilation을 써도 self-attention 수준의 전역 연결은 어렵다.

이 때문에 CNN은 의료영상 분할의 강력한 기본기이지만, global reasoning이 중요한 문제에서 구조적 한계를 가진다고 정리한다.

## 2. Transformer 기반 의료영상 분할

논문은 Transformer를 의료영상 분할에서 등장한 새로운 전역 문맥 도구로 설명한다. 핵심 논리는 self-attention이 이미지 전체의 장거리 관계를 직접 모델링할 수 있다는 점이다.

### 2.1 Transformer의 핵심 장점

- long-range dependency modeling
- global contextual representation
- 서로 멀리 떨어진 장기/병변 관계 파악
- 복잡한 shape variation에 대한 표현력 향상

이 특성은 특히 다중 장기 분할이나 복잡한 해부학 배치를 다루는 문제에서 유리하다고 논문은 본다.

### 2.2 Transformer 기반 구조의 분류

논문은 Transformer 모델을 대략 다음 방향으로 읽게 만든다.

- ViT 스타일 patch token 기반 segmentation
- U-shaped Transformer 구조
- CNN encoder와 결합한 hybrid Transformer
- 3D volumetric Transformer 또는 window-based Transformer

이 taxonomy의 핵심은 pure Transformer가 아니라 hybrid와 구조적 변형이 실제 의료영상 분할에서 더 자주 채택된다는 점이다.

### 2.3 Transformer의 한계

저자들은 Transformer의 약점도 분명히 적는다.

- 데이터 요구량이 크다.
- 계산량과 메모리 사용량이 크다.
- fine boundary 복원에서 local detail이 약할 수 있다.
- 3D 데이터에 그대로 적용하면 비용이 급격히 증가한다.

즉, Transformer는 의료영상 segmentation의 새로운 가능성을 열었지만, 바로 표준 해법이 되기에는 도메인 제약이 크다는 것이다.

## 3. CNN-Transformer hybrid

이 논문에서 가장 실용적인 결론은 hybrid 구조의 중요성이다. 저자들은 CNN과 Transformer를 대립항으로 보기보다 상보적 구성요소로 본다.

### 3.1 왜 hybrid가 필요한가

- CNN은 local detail과 boundary preservation에 강하다.
- Transformer는 global context와 long-range dependency에 강하다.
- 의료영상 분할은 두 성질이 모두 필요하다.

따라서 많은 최신 모델이 CNN encoder + Transformer bottleneck, Transformer encoder + CNN decoder, 혹은 stage-wise mixed block 같은 혼합 전략을 채택한다고 정리한다.

### 3.2 hybrid 구조의 실질적 의미

논문이 시사하는 바는 분명하다. 의료영상 segmentation의 핵심은 "CNN에서 Transformer로의 단순 교체"가 아니라, `local-global tradeoff`를 어떻게 설계하느냐에 있다. 이 관점은 지금 봐도 여전히 유효하다.

## 주요 응용 과제별 정리

논문은 segmentation을 특정 장기에 한정하지 않고 여러 과제에 걸쳐 구조 변화를 설명한다.

### 1. Brain MRI segmentation

brain tumor, brain tissue, lesion segmentation은 Transformer 도입이 활발한 대표 영역으로 다뤄진다. 3D 문맥과 복잡한 구조 관계가 중요하기 때문이다.

### 2. Abdominal multi-organ segmentation

장기 간 위치 관계와 전역 문맥이 중요한 과제로, Transformer와 hybrid 구조의 장점이 상대적으로 잘 드러나는 영역으로 설명된다.

### 3. Cardiac segmentation

심장 구조는 형태 변이와 경계 정밀도가 모두 중요해, CNN의 detail 복원과 Transformer의 context modeling이 함께 요구되는 전형적 사례로 해석된다.

### 4. Retinal, skin, polyp, lesion segmentation

이 과제들은 작은 구조와 경계가 중요해 CNN 기반 또는 attention-enhanced CNN의 강점도 여전히 크다. 논문은 모든 task에서 pure Transformer가 항상 우세하다고 보지 않는다.

## 정량 결과 해석

이 논문은 자체 benchmark를 수행하는 실험 논문이 아니라 review다. 따라서 주된 역할은 여러 논문의 결과를 구조화해 해석하는 것이다.

저자들의 핵심 해석은 다음과 같다.

- Transformer 계열은 여러 benchmark에서 CNN baseline을 능가하거나 경쟁력 있는 성능을 보였다.
- 그러나 성능 개선은 task, dataset size, 2D/3D 설정, pretraining 여부에 따라 크게 달라진다.
- hybrid 구조가 pure CNN과 pure Transformer 사이의 현실적 절충안으로 가장 자주 채택된다.
- 3D 의료영상에서는 정확도 향상과 계산 비용 사이 tradeoff가 매우 크다.

이 논문은 결과를 읽을 때 숫자만 보지 말고, backbone, input dimensionality, data regime, efficiency를 함께 봐야 한다고 시사한다.

## 이 논문의 핵심 메시지

### 1. 의료영상 분할의 발전은 backbone 교체의 역사다

논문은 FCN/U-Net 계열에서 attention CNN, hybrid Transformer, pure Transformer로 이어지는 흐름을 하나의 연속선으로 제시한다. 즉, 모델 진화는 단절이 아니라 점진적 확장이다.

### 2. Transformer의 도입 이유는 전역 문맥 부족 때문이다

CNN이 실패해서가 아니라, 더 넓은 context와 더 긴 dependency를 다루기 위해 Transformer가 도입됐다는 점을 논문은 분명히 한다. 이 시각은 기술 유행에 휩쓸리지 않게 해 준다.

### 3. pure Transformer가 종착점이라고 보지는 않는다

이 논문은 hybrid 구조의 비중을 높게 평가한다. 이는 의료영상 도메인에서 inductive bias와 효율성이 여전히 중요하다는 판단과 연결된다.

## 한계와 저자들이 제시한 미래 방향

논문 후반부는 비교적 균형 잡힌 미래 전망을 제시한다.

### 1. 데이터 부족

Transformer는 데이터 요구량이 크기 때문에 의료 도메인에서는 여전히 data scarcity가 큰 장벽이다. 따라서 self-supervised learning이나 domain pretraining이 중요해질 것으로 본다.

### 2. 계산 비용과 3D 확장성

volumetric segmentation에서는 memory bottleneck이 크므로, efficient Transformer, sparse attention, window attention, lightweight hybrid 설계가 중요하다고 본다.

### 3. 일반화와 강건성

병원, 스캐너, 프로토콜 차이로 인한 domain shift가 여전히 큰 문제이며, 단일 benchmark 성능만으로 임상 적용 가능성을 판단할 수 없다고 본다.

### 4. 해석 가능성과 임상 신뢰성

Transformer attention map이 곧 설명 가능성을 보장하는 것은 아니며, trustworthy medical segmentation을 위한 별도의 평가가 필요하다고 시사한다.

### 5. 대규모 사전학습과 범용 모델

저자들은 당시 emerging trend로 더 큰 pretraining과 더 범용적인 분할 모델 가능성을 암시한다. 이는 이후 foundation model 흐름과 자연스럽게 이어진다.

## 실무적 관점의 해설

### 1. 이 논문은 "CNN vs Transformer" 논쟁을 정리하는 문서다

실제로는 둘 중 하나가 옳다는 결론이 아니라, 의료영상 분할이 왜 CNN에서 Transformer로 일부 이동했는지, 그리고 왜 다시 hybrid로 수렴하는지를 보여주는 문서다.

### 2. task별로 요구되는 inductive bias가 다르다는 점을 잘 보여준다

가는 혈관, 작은 병변, 경계 정밀도가 중요한 과제와, 다중 장기 위치 관계가 중요한 과제는 필요한 표현이 다르다. 논문은 이 차이를 backbone 선택과 연결해 읽게 만든다.

### 3. 2023년 전환기 survey로서 가치가 크다

이 논문은 Transformer hype가 커지던 시점에 CNN 유산을 지우지 않고 함께 정리했다는 점에서 유용하다. 이후 foundation model과 promptable segmentation으로 넘어가기 전 단계를 이해하는 데 적절하다.

### 4. 최신 관점에서는 후속 보완이 필요하다

이 논문 이후 SAM 계열 적응, universal medical segmentation, foundation model pretraining, diffusion or Mamba 계열 구조가 더 부상했다. 따라서 현재 연구 설계에는 최신 survey와 함께 읽는 것이 적절하다.

## 후속 연구와의 연결

이 논문이 보여주는 흐름은 이후 다음과 같이 확장된다.

- hybrid Transformer에서 foundation model adaptation으로 확장
- supervised segmentation 중심에서 promptable, few-shot, universal segmentation으로 확장
- 정확도 중심 비교에서 robustness, efficiency, reliability 비교로 확장
- architecture novelty 중심에서 data-centric pretraining 중심으로 일부 이동

즉, 이 논문은 의료영상 segmentation 연구가 `CNN 구조 탐색`에서 `전역 문맥 통합`, 그리고 다시 `범용 표현 학습`으로 이동하는 중간 지점을 잘 보여준다.

## 종합 평가

`From CNN to Transformer: A Review of Medical Image Segmentation Models`는 의료영상 분할 모델 발전을 세대 변화 관점에서 이해하기 좋은 survey다. 가장 큰 장점은 CNN, Transformer, hybrid를 경쟁 모델로만 보지 않고, 각각의 inductive bias와 적용 조건을 함께 설명한다는 점이다.

한계는 2023년 이후의 foundation model, SAM adaptation, promptable segmentation 흐름을 담지 못한다는 점이다. 그럼에도 의료영상 분할 모델의 구조적 진화를 빠르게 파악하고 싶다면, 이 논문은 여전히 좋은 기준점이다.
