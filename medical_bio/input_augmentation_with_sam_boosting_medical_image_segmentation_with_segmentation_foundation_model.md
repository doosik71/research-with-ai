# Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model

## 논문 메타데이터

- **제목**: Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model
- **저자**: Yizhe Zhang, Tao Zhou, Shuo Wang, Peixian Liang, Danny Z. Chen
- **초기 arXiv 공개**: 2023
- **출판 형태**: MICCAI Workshop 계열 Springer LNCS 챕터 및 arXiv preprint
- **DOI**: 10.1007/978-3-031-47401-9_13
- **arXiv ID**: 2304.11332
- **arXiv URL**: https://arxiv.org/abs/2304.11332
- **PDF URL**: https://arxiv.org/pdf/2304.11332

## 연구 배경 및 문제 정의

이 논문은 SAM을 의료영상 segmentation 모델로 직접 바꾸기보다, SAM이 생성하는 mask와 stability score를 `학습용 prior`로 활용하는 접근을 제안한다. 저자들의 출발점은 분명하다. SAM은 자연영상에서 강력한 segmentation foundation model이지만, 의료영상에서는 zero-shot 성능이 충분하지 않다. 그렇다고 SAM을 통째로 fine-tuning하는 것이 항상 최선은 아니다.

논문이 제기하는 핵심 문제는 다음과 같다.

- SAM은 의료영상에서 바로 high-quality mask를 주지 못한다.
- 하지만 SAM이 생성하는 segmentation mask, feature, stability score는 여전히 유용한 구조적 힌트일 수 있다.
- 의료영상 segmentation의 강한 task-specific 모델은 존재하지만, foundation model이 가진 일반 perception prior를 충분히 활용하지 못한다.

따라서 저자들은 "SAM을 직접 출력기로 쓰지 말고, SAM이 제공하는 prior map을 입력에 섞어 downstream segmentation model을 더 잘 학습시키자"는 방향을 택한다.

## 논문의 핵심 기여

논문이 주장하는 기여는 매우 단순하고 명확하다.

1. SAM이 의료영상에서도 직접적인 mask 예측기 이상으로 `prior map generator` 역할을 할 수 있음을 보였다.
2. SAM의 segmentation 결과와 stability score를 이용해 raw input을 증강하는 `SAMAug`를 제안했다.
3. 이 augmentation은 parameter-free fusion으로 구현되며, SAM 자체는 고정한 채 downstream model만 학습한다.
4. CNN 기반과 Transformer 기반 의료영상 segmentation model 모두에서 성능 향상을 보였다고 보고했다.
5. polyp, gland, nucleus segmentation의 세 과제에서 foundation model prior가 일반 segmentation model 학습에 도움이 될 수 있음을 보였다.

## 방법론 구조 요약

## 1. 핵심 아이디어: SAM을 입력 증강기로 사용

SAMAug의 핵심은 복잡하지 않다. 각 의료영상 입력에 대해 먼저 SAM을 실행하고, 여기서 나온 segmentation mask와 stability score를 이용해 prior map을 만든다. 그런 다음 이 prior map을 원본 영상의 추가 채널로 붙여 downstream segmentation model의 입력으로 사용한다.

저자들이 이 방식을 택한 이유는 실용적이다.

- SAM의 사전학습 지식을 재사용할 수 있다.
- downstream network 구조를 거의 바꾸지 않아도 된다.
- SAM을 fine-tune하지 않으므로 학습 안정성과 구현 단순성이 높다.

즉, 이 논문은 SAM adaptation을 `모델 적응`이 아니라 `입력 적응` 문제로 바꾼다.

## 2. Prior map 생성 방식

논문 설명에 따르면 SAM은 여러 후보 mask와 stability score를 생성한다. 저자들은 이를 이용해 두 종류의 prior를 만든다.

- **segmentation prior map**
- **boundary prior map**

이 구분이 중요한 이유는 많은 의료영상 분할 문제가 사실상 다음 세 범주로 환원될 수 있기 때문이다.

- background
- ROI
- ROI와 background 사이 boundary

저자들은 SAM에서 얻은 정보를 이 세 범주를 보조하는 형태로 재구성하고, 원본 입력과 함께 네트워크에 넣는다.

## 3. Fusion 방식

논문이 특히 강조하는 부분은 fusion이 parameter-free라는 점이다. 복잡한 adapter나 attention block을 추가하지 않고, SAM prior를 원본 이미지 채널에 직접 더하거나 결합한다.

gray-scale 의료영상의 경우 논문이 설명하는 입력 구성은 다음과 같다.

- 첫 번째 채널: 원본 gray-scale image
- 두 번째 채널: segmentation prior map
- 세 번째 채널: boundary prior map

이렇게 만들어진 `SAM-augmented input`을 기존 segmentation network에 그대로 넣는다.

이 설계는 매우 단순하지만, 논문의 핵심 메시지도 바로 여기에 있다. foundation model의 지식을 활용하는 데 꼭 복잡한 tuning이 필요한 것은 아니라는 것이다.

## 4. 다운스트림 모델과 학습 방식

저자들은 SAMAug를 특정 backbone에 묶지 않는다. 논문에서 강조하는 장점은 model-agnostic 특성이다.

- CNN segmentation model에도 적용 가능
- Transformer segmentation model에도 적용 가능
- 학습 시 업데이트되는 것은 downstream model 파라미터뿐
- SAM은 inference-style prior generator로 고정

이 때문에 SAMAug는 일종의 plug-in augmentation scheme으로 해석할 수 있다.

## 실험 설정

논문은 세 가지 의료영상 분할 과제에서 실험했다고 밝힌다.

- **polyp segmentation**
- **gland segmentation**
- **nucleus segmentation**

본문 스니펫 기준으로 MoNuSeg와 GlaS가 명시적으로 언급되며, 세 과제 모두 2D biomedical image segmentation에 속한다. 저자들은 CNN 및 Transformer 기반 segmentation 모델을 baseline으로 두고, SAMAug 적용 전후를 비교한다.

## 주요 결과 해석

## 1. SAM은 직접 분할기보다 prior generator로 더 유용하다

이 논문의 가장 중요한 관찰은 이것이다. 의료영상에서 SAM의 raw output은 바로 쓰기 어렵지만, 그 output을 prior로 재가공하면 downstream model 학습에 실제로 도움이 된다.

이 해석은 중요하다. 많은 SAM 의료 응용이 `SAM을 fine-tune해서 직접 잘 맞추자`는 방향인데, 이 논문은 그보다 더 단순한 우회 경로가 가능하다고 보여준다.

## 2. CNN과 Transformer 모두 이득을 본다

논문은 SAMAug가 특정 backbone에만 맞는 트릭이 아니라, 여러 segmentation model에 공통적으로 도움이 된다고 주장한다. 이는 prior map이 architecture-specific feature라기보다 task-level spatial hint로 작동함을 시사한다.

## 3. Boundary prior의 역할

boundary prior를 따로 분리한 것은 의료영상에서 특히 의미가 있다. polyp, gland, nucleus segmentation 모두 경계가 성능을 좌우하기 때문이다. 논문은 ROI mask만이 아니라 boundary cue까지 함께 주입하는 것이 학습에 유익하다고 본다.

## 4. 단순한 방법 대비 의미 있는 개선

이 논문의 설득력은 거대한 구조 혁신이 아니라, 구현이 매우 단순한데도 성능이 일관되게 좋아졌다는 데 있다. 즉, foundation model 활용이 반드시 무거운 adapter나 fine-tuning을 필요로 하지 않는다는 점을 보여준다.

## 논문의 핵심 메시지

### 1. foundation model은 출력기보다 정보원으로 볼 수 있다

SAM을 segmentation answer generator로 쓰지 말고, 추가적인 시각 prior를 제공하는 정보원으로 쓰는 관점 전환이 이 논문의 핵심이다.

### 2. 의료영상에서는 입력 증강이 강력한 적응 전략이 될 수 있다

모델 내부를 고치는 대신 입력을 바꾸는 방식은 의료영상처럼 데이터가 적고 모델 안정성이 중요한 환경에서 실용적일 수 있다.

### 3. simple baseline이 의외로 강할 수 있다

SAMAug는 아이디어 자체가 단순하다. 하지만 그 단순함 때문에 구현 비용이 낮고, 다양한 backbone에 쉽게 붙일 수 있다는 장점이 있다.

## 한계와 저자들이 시사하는 미래 방향

논문 결론부와 구성에서 읽히는 한계는 다음과 같다.

### 1. SAM 실행 비용

SAMAug는 학습/추론 전에 모든 이미지에 대해 SAM을 돌려 prior를 만들어야 한다. 따라서 단순한 학습 파이프라인에 비해 전처리 비용이 늘어난다.

### 2. SAM quality에 대한 의존성

prior map의 품질은 SAM의 출력 품질에 영향을 받는다. domain gap이 큰 modality에서는 prior가 noisy hint가 될 수도 있다.

### 3. 2D biomedical task 중심 평가

논문 실험은 주로 polyp, gland, nucleus 같은 2D 과제에 집중돼 있다. volumetric CT/MRI나 multi-organ 3D segmentation으로 일반화되는지는 별도 검증이 필요하다.

### 4. fusion 함수의 단순성

저자들도 향후 더 robust하고 advanced한 augmentation function 설계를 future work로 제안한다. 즉, 현재 방식은 deliberately simple하지만 최적이라고 보기는 어렵다.

### 5. uncertainty와 임상 응용 가능성

논문은 future work로 uncertainty estimation과 clinically-oriented applications를 언급한다. 이는 SAM prior가 confidence-aware segmentation으로 확장될 수 있음을 시사한다.

## 실무적 관점의 해설

### 1. 이 논문은 "SAM fine-tuning" 논문이 아니다

핵심은 adapter나 prompt tuning이 아니라, SAM 출력을 입력 채널로 재활용하는 것이다. 따라서 foundation model adaptation literature 안에서도 매우 가벼운 축에 속한다.

### 2. 구현 난도가 낮다는 점이 가장 큰 장점이다

기존 segmentation model을 거의 바꾸지 않고도 쓸 수 있기 때문에, 이미 구축된 medical segmentation 파이프라인에 붙이기 쉽다.

### 3. 반대로 상한선은 제한될 수 있다

입력에 prior를 붙이는 수준이라, backbone 내부 표현을 직접 바꾸는 정교한 adaptation보다 성능 상한이 낮을 가능성도 있다. 즉, 강력한 baseline이지만 궁극적 해법이라고 보기는 어렵다.

### 4. foundation model 활용의 좋은 사고실험이다

이 논문은 "foundation model을 반드시 end-to-end 모델로 써야 하는가?"라는 질문에 대한 좋은 반례다. pretrained model을 간접적 prior source로 써도 충분히 가치가 있을 수 있다.

## 후속 연구와의 연결

이 논문은 이후 다음 방향과 연결된다.

- SAM 기반 input augmentation의 고도화
- prior map과 uncertainty map의 결합
- 3D medical segmentation에서의 foundation prior 활용
- segmentation뿐 아니라 classification, detection, report generation에서의 SAM-derived signal 활용

즉, SAMAug는 거대한 모델 적응 전략이라기보다, foundation model을 기존 의료 AI 파이프라인에 끼워 넣는 가벼운 브리지 방법으로 읽는 것이 적절하다.

## 종합 평가

`Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model`은 매우 단순하지만 아이디어가 명확한 논문이다. SAM을 직접 의료영상 분할기로 바꾸지 않고, SAM이 만든 segmentation prior와 boundary prior를 입력 증강으로 활용해 downstream model을 강화한다.

이 논문의 장점은 구현 단순성, backbone 비의존성, parameter-free fusion에 있다. 반면 한계는 2D biomedical task 중심 평가와 SAM prior 품질 의존성이다. 그럼에도 foundation model을 의료영상 segmentation에서 어떻게 가볍게 활용할 수 있는지 보여주는 좋은 초기 사례로 평가할 수 있다.
