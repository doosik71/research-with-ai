# A Recent Survey of Vision Transformers for Medical Image Segmentation

## 논문 메타데이터

- **제목**: A Recent Survey of Vision Transformers for Medical Image Segmentation
- **저자**: Asifullah Khan, Zunaira Rauf, Abdul Rehman Khan, Saima Rathore, Saddam Hussain Khan, Najamus Saher Shah, Umair Farooq, Hifsa Asif, Aqsa Asif, Umme Zahoora, Rafi Ullah Khalil, Suleman Qamar, Umme Hani Tayyab, Faiza Babar Khan, Abdul Majid, Jeonghwan Gwak
- **초기 arXiv 공개**: 2023년 12월
- **저널 출판**: 2025년 10월 6일
- **저널**: IEEE Access, Vol. 13, 2025, pp. 191824-191849
- **DOI**: 10.1109/ACCESS.2025.3618215
- **arXiv ID**: 2312.00634
- **arXiv URL**: https://arxiv.org/abs/2312.00634
- **PDF URL**: https://arxiv.org/pdf/2312.00634

## 연구 배경 및 문제 정의

이 논문은 의료영상 분할에서 Vision Transformer(ViT)가 왜 중요한 대안으로 부상했는지, 그리고 왜 실제로는 pure ViT보다 hybrid vision transformer(HVT)가 더 실용적으로 쓰이는지를 정리하는 survey다.

저자들의 문제의식은 비교적 명확하다.

- CNN은 지역적 패턴과 경계 복원에는 강하지만 전역 관계 모델링은 제한적이다.
- ViT는 multi-head self-attention으로 전역 문맥과 장거리 의존성을 잘 포착할 수 있다.
- 그러나 ViT는 이미지 고유 inductive bias와 translation invariance가 부족하고, 더 많은 데이터와 계산량을 요구한다.
- 의료영상은 라벨 부족, 클래스 불균형, 노이즈, artifact, 모호한 경계, 3D 계산 비용 문제를 동시에 가진다.

따라서 이 논문은 "ViT가 CNN보다 우월한가"를 단순 비교하지 않는다. 대신 의료영상 분할에서 ViT와 HVT가 어떤 구조로 쓰이고, 어떤 modality에서 유리하며, 어떤 한계를 가지는지를 체계적으로 정리한다.

## 논문의 핵심 기여

이 survey의 중심 기여는 taxonomy와 modality별 실무적 비교에 있다.

1. ViT 기반 의료영상 분할을 `pure ViT-based methods`와 `HVT-based methods`로 크게 나눴다.
2. pure ViT 계열은 encoder, decoder, encoder-decoder interface, encoder-decoder 양쪽 배치로 세분화했다.
3. HVT 계열은 `encoder-based integration`, `decoder-based integration`, `encoder-decoder interface integration`으로 다시 분류했다.
4. CT, MRI, ultrasound, X-ray, histopathology, microscopy 등 다양한 modality별 대표 모델을 표와 함께 정리했다.
5. 각 modality에서 정확도뿐 아니라 파라미터 수와 추론 시간까지 함께 보며 효율성 관점의 해석을 시도했다.

## 방법론 구조 요약

## 1. Pure ViT-based segmentation

논문은 pure ViT 계열을 U-Net 유사 encoder-decoder 구조 안에서 Transformer를 어디에 배치하느냐로 구분한다.

### 1.1 ViT in encoder

가장 흔한 형태다. encoder에서 self-attention으로 전역 관계를 학습하고, CNN 또는 경량 decoder가 segmentation mask를 복원한다. UNETR가 대표 사례로 소개된다.

이 구조의 장점은 명확하다.

- latent space에서 global feature를 강하게 학습할 수 있다.
- decoder는 상대적으로 단순하게 유지할 수 있다.
- 3D segmentation에도 확장 사례가 많다.

반면 단점도 분명하다.

- encoder가 무거워질 수 있다.
- local detail 복원은 decoder나 skip connection에 크게 의존한다.

### 1.2 ViT in decoder

ConvTransSeg 같은 구조가 이 범주다. encoder는 CNN으로 저수준 feature를 추출하고, decoder에서 ViT를 사용해 전역 문맥을 반영하며 정교하게 복원한다.

저자들의 해석은 실용적이다. segmentation에서 경계가 정확한 mask를 만들려면 decoding 단계에서도 전역 문맥이 중요할 수 있으므로, MHSA를 decoder에 넣는 시도가 등장했다는 것이다.

### 1.3 ViT at encoder-decoder interface

bottleneck 또는 fusion stage에서 Transformer를 써서 encoder와 decoder 사이의 의미적 간극을 줄이는 접근이다. DCA Net 같은 구조가 예시로 제시된다.

이 방식은 CNN의 local bias를 유지하면서도, 핵심 병목 지점에서만 global reasoning을 넣는 절충안으로 읽힌다.

### 1.4 ViT in both encoder and decoder

Swin-UNet 계열처럼 양쪽에 Transformer를 배치하는 방식이다. 전역 정보 활용을 극대화할 수 있지만, 계산량과 구조 복잡도가 증가한다.

## 2. HVT-based segmentation

논문은 pure ViT의 가장 큰 약점이 low-level detail 손실 가능성이라고 본다. 그래서 HVT를 local feature와 global context를 동시에 확보하는 더 현실적인 방향으로 제시한다.

### 2.1 Encoder-based integration

encoder에서 CNN과 ViT를 결합하는 방식이다. 저자들은 이 범주를 여러 modality에서 가장 실용적인 형태로 반복해서 평가한다.

이 구조가 자주 유리한 이유는 다음과 같다.

- 초기 feature extraction에서 CNN의 local inductive bias를 활용할 수 있다.
- 필요한 단계에서만 Transformer로 global context를 추가할 수 있다.
- 파라미터 수와 추론 비용을 일정 수준으로 통제하기 쉽다.

### 2.2 Decoder-based integration

decoder에 Transformer를 삽입해 coarse-to-fine reconstruction을 보강하는 방식이다. 경계 복원과 semantic refinement를 강화하려는 의도가 강하다.

### 2.3 Integration at the encoder-decoder interface

skip connection 또는 fusion module 수준에서 CNN feature와 Transformer feature를 결합하는 방식이다. 이 논문은 이를 의미적 간극 해소와 상보 정보 통합의 관점에서 설명한다.

## modality별 정리와 핵심 해석

이 논문의 특징은 단순 구조 taxonomy에 그치지 않고, modality별로 어떤 형태의 ViT/HVT가 실용적인지까지 분석한다는 점이다.

## 1. CT

CT 섹션에서 저자들은 kidney tumor, prostate, COVID-19 관련 CT, multi-organ CT 등을 예시로 든다. 핵심 결론은 HVT, 특히 encoder에 ViT가 들어간 구조가 정확도와 효율성의 균형이 좋다는 것이다.

논문에 따르면 CT 비교에서 다음 경향이 관찰된다.

- **TransUNet**은 BTCV에서 Dice 0.884, 약 41.4M 파라미터, 빠른 추론을 보였다.
- **FocalUNETR**는 pure ViT encoder를 쓰지만 더 많은 파라미터와 더 느린 추론으로 약간 낮은 성능을 보였다.
- **TAU-Net3+**는 단일 장기에서는 매우 높은 정확도를 보이지만, multi-organ complexity를 대표하지는 않는다고 해석된다.

저자들의 결론은 명확하다. CT에서는 encoder-based HVT가 성능을 유지하면서도 파라미터와 추론 비용 측면에서 더 실용적이다.

## 2. MRI

MRI는 brain tumor, cardiac MRI, breast lesion 등 다양한 구조가 소개된다. 이 섹션에서도 HVT가 parameter-efficiency 측면에서 유리하다는 해석이 반복된다.

논문은 다음과 같은 비교를 제시한다.

- **TransConver**: BraTS2019에서 Dice 0.8173-0.8668, 약 9M 파라미터
- **Swin UNETR**: BraTS2021에서 Dice 0.8530, 약 61.98M 파라미터
- **UnetFormer**: BraTS2021에서 Dice 0.8880, 약 58.98M 파라미터
- **AMTNet**: WT 영역에서 Dice 최대 0.9240, 약 10.9M 파라미터

여기서 논문은 단순 최고 Dice보다 `비슷한 성능을 훨씬 적은 파라미터로 내는 encoder-based HVT`의 실용성을 강조한다. MRI처럼 3D 문맥과 계산 비용이 동시에 중요한 환경에서는 특히 설득력 있는 주장이다.

## 3. Ultrasound

ultrasound는 실시간성, 잡음, 불규칙 경계가 핵심 문제다. 논문은 여기서도 encoder 또는 intermediate stage에 Transformer를 넣는 HVT가 좋은 균형을 보인다고 본다.

대표 비교는 다음과 같다.

- **Cswin-Pnet**: Dice 0.8725
- **3D-UNet with decoder-level Transformers**: Dice 0.7636
- **BTS-ST**: F1-score 0.9080

저자들은 무거운 ViT 모듈을 여러 계층에 넣는 것이 항상 이득은 아니며, ultrasound처럼 실시간성 요구가 있는 경우 encoder-based HVT가 더 적절하다고 본다.

## 4. X-ray

X-ray, 특히 dental panoramic radiograph와 mammography에서는 Transformer를 encoder와 decoder 양쪽에 넣으면 정확도는 오를 수 있지만 복잡도도 크게 증가한다고 정리한다. 따라서 이 영역은 "최대 성능"과 "실용적 배포" 사이의 tradeoff가 더 두드러진다.

## 5. Histopathology and Microscopy

이 논문은 pathology와 microscopy도 별도 modality로 다룬다. 이들 영역에서는 patch representation과 long-range context modeling의 장점 때문에 ViT가 특히 매력적이지만, stain variation, 고해상도 입력, 데이터 차이 문제 때문에 여전히 강건성 이슈가 남는다고 본다.

## 성능 향상 기법에 대한 정리

저자들은 단순 architecture taxonomy 외에도 HVT 성능 향상을 위해 자주 쓰이는 보조 전략을 언급한다.

- pre-processing과 post-processing을 통한 결과 refinement
- boundary-aware module
- attention refinement
- CNN 기반 skip path 보강
- federated learning과의 결합 가능성
- diffusion model을 통한 synthetic data generation 가능성

이 부분은 다소 넓게 서술되지만, 메시지는 일관된다. ViT/HVT의 성능은 backbone 하나로 결정되지 않고 데이터 처리와 보조 모듈 설계에 크게 좌우된다.

## 논문의 핵심 메시지

### 1. ViT보다 HVT가 더 실용적이다

이 논문은 여러 modality 비교를 통해 pure ViT보다 HVT를 더 실용적인 해법으로 본다. 이유는 명확하다. CNN의 local bias와 Transformer의 global reasoning을 동시에 활용할 수 있기 때문이다.

### 2. 특히 encoder-based HVT가 자주 유리하다

CT, MRI, ultrasound 비교에서 공통적으로 encoder-based HVT가 성능과 효율성 사이 균형이 좋다고 해석한다. 이 점이 이 survey의 가장 구체적인 결론이다.

### 3. modality마다 최적 구조가 다르다

multi-organ CT, brain MRI, ultrasound tumor segmentation, pathology 등은 구조적 요구가 다르기 때문에 단일 ViT 설계가 모든 상황에서 최적일 수 없다고 본다.

## 한계와 저자들이 제시한 미래 방향

논문은 challenges와 future recommendations를 비교적 분명하게 적는다.

### 1. local detail 부족

ViT는 self-attention 중심이라 fine-grained local detail을 놓칠 수 있다. HVT가 이를 완화하지만 완전히 해결된 문제는 아니라고 본다.

### 2. data scarcity

의료 도메인은 라벨 데이터가 적기 때문에 ViT 학습이 어렵다. 저자들은 대규모 자연영상 pretraining 후 medical fine-tuning을 중요한 방향으로 제시한다.

### 3. 해석 가능성

attention map이 존재한다고 해서 모델 해석이 쉬운 것은 아니다. 저자들은 diffused attention map 때문에 임상적으로 이해 가능한 설명을 만드는 일이 여전히 어렵다고 본다.

### 4. multimodal learning

ViT는 sequence-like input 처리가 가능하므로 multi-modal learning의 좋은 후보라고 본다. 서로 다른 modality 간 상보 정보를 통합해 segmentation accuracy와 reliability를 높일 수 있다는 것이다.

### 5. privacy-aware training

federated learning도 유망한 방향으로 언급되지만, 데이터 차이, 통신 비용, 성능 일관성 문제가 남아 있다고 본다.

## 실무적 관점의 해설

### 1. 이 논문은 ViT 의료영상 분할 survey이지만 실제 결론은 hybrid 옹호에 가깝다

제목은 ViT survey지만, 본문을 보면 pure ViT보다 HVT가 여러 modality에서 더 현실적이라는 결론으로 수렴한다. 따라서 이 논문은 "Transformer가 대세"를 주장하기보다 "의료영상에서는 hybrid가 더 낫다"는 쪽에 가깝다.

### 2. 효율성까지 보려는 점은 장점이지만 비교 엄밀성은 제한적이다

모델별 데이터셋, 전처리, 실험 설정이 다르기 때문에 파라미터 수와 Dice를 완전히 공정하게 가로비교하기는 어렵다. 따라서 이 논문의 표는 엄밀한 leaderboard라기보다 실무적 경향 해석으로 읽는 편이 맞다.

### 3. 2025년 기준 survey지만 foundation model 이후 흐름은 아직 제한적이다

저널판이 2025년 10월 6일에 나왔지만, 중심 내용은 pre-foundation-model 성격의 ViT/HVT taxonomy다. SAM 기반 의료 adaptation, universal segmentation, multimodal foundation model은 깊게 다루지 않는다.

## 후속 연구와의 연결

이 논문 이후 자연스럽게 이어지는 방향은 다음과 같다.

- HVT에서 foundation model adaptation으로 이동
- pure supervised segmentation에서 promptable / universal segmentation으로 확장
- ViT 효율화에서 Mamba/SSM 계열 대안 탐색으로 확장
- 정확도 중심 평가에서 robustness, reliability, deployment efficiency로 확장

즉, 이 논문은 ViT-based 의료영상 분할의 정리본이자, foundation model 이전 단계의 종합 보고서로 읽는 것이 가장 적절하다.

## 종합 평가

`A Recent Survey of Vision Transformers for Medical Image Segmentation`는 ViT 기반 의료영상 분할을 구조와 modality 양쪽에서 체계적으로 정리한 유용한 survey다. 특히 `pure ViT vs HVT`, 그리고 `encoder/decoder/interface integration`이라는 분류는 설계 공간을 빠르게 파악하는 데 도움이 된다.

가장 설득력 있는 결론은 여러 modality에서 encoder-based HVT가 성능과 효율성의 균형이 좋다는 점이다. 반면 한계는 비교 실험의 통일성이 약하고, 2025년 시점에도 최신 foundation model 흐름을 충분히 다루지 못한다는 것이다. 그럼에도 ViT 의료영상 분할 연구를 입문하거나, hybrid 설계가 왜 계속 강세인지 이해하려면 여전히 참고 가치가 높다.
