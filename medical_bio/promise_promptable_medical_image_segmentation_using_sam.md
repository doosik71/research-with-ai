# ProMISe: Promptable Medical Image Segmentation using SAM

## 논문 메타데이터

- **제목**: ProMISe: Promptable Medical Image Segmentation using SAM
- **저자**: Jinfeng Wang, Xiaowei Xu, Kaixuan Chen, Dong Yang, Can Zhao, Ziyue Xu, Kevin Li, Alan Yuille, Yiheng Wang
- **출판 연도**: 2024
- **형태**: arXiv preprint
- **arXiv ID**: 2403.04164
- **arXiv URL**: https://arxiv.org/abs/2403.04164
- **PDF URL**: https://arxiv.org/pdf/2403.04164v3

## 연구 배경 및 문제 정의

이 논문은 Segment Anything Model(SAM)이 자연영상에서는 강력한 zero-shot 분할 성능을 보였지만, 의료영상 semantic segmentation에는 그대로 적용하기 어렵다는 문제에서 출발한다. 의료영상에서는 단순한 point/box prompt 기반 interactive segmentation보다, 장기나 병변의 의미론적 분할과 3D volumetric 일반화가 더 중요하다.

저자들이 짚는 SAM의 한계는 다음과 같다.

- prompt encoder가 sparse point/box prompt를 전제로 설계돼 있다.
- 자연영상 사전학습 표현이 의료영상 modality에 그대로 맞지 않는다.
- semantic segmentation에서는 test-time에 사람이 prompt를 제공하지 않는 경우가 많다.
- 3D 의료영상은 slice 간 일관성과 volumetric context가 중요하다.

따라서 이 논문의 핵심 질문은 "SAM의 promptable 구조를 유지하면서, semantic medical segmentation에 맞는 prompt를 학습할 수 있는가"다.

## 논문의 핵심 기여

논문이 주장하는 기여는 비교적 명확하다.

1. point/box 대신 **learnable visual prompts**를 도입해 SAM을 의료영상 semantic segmentation에 맞게 재구성했다.
2. 2D와 3D segmentation을 모두 지원하는 `parameter-efficient adaptation` 구조를 제안했다.
3. SAM image encoder를 대부분 고정한 상태에서 적은 추가 파라미터로 의료 도메인에 적응하도록 설계했다.
4. 4개 데이터셋, 5개 과제에서 CNN 기반과 Transformer 기반 baseline을 능가하거나 경쟁력 있는 결과를 보였다.
5. 단일 장기 2D 문제뿐 아니라 multi-organ, multi-class, volumetric segmentation까지 promptable formulation을 확장할 수 있음을 보였다.

## 방법론 구조 요약

## 1. 핵심 아이디어: point prompt를 learnable visual prompt로 대체

ProMISe의 가장 중요한 차별점은 사람이 찍는 점이나 박스를 쓰지 않는다는 점이다. 대신 저자들은 학습 가능한 prompt token을 만들어 SAM의 prompt encoder에 넣는다.

이 접근의 의미는 단순하다.

- semantic segmentation에서는 inference 시점에 사람이 interactive prompt를 제공하지 않아도 된다.
- 클래스 또는 구조별 prior를 학습 가능한 prompt 형태로 내재화할 수 있다.
- SAM의 `promptable segmentation` 패러다임은 유지하면서, 의료영상의 자동 분할 문제에 맞게 재해석할 수 있다.

논문은 이를 `visual prompt generation` 관점에서 설명한다. 즉, prompt를 외부 입력이 아니라 모델이 내부적으로 생성하고 학습하는 대상로 바꾸는 것이다.

## 2. 파라미터 효율적 적응

저자들은 SAM 전체를 대규모 fine-tuning하지 않는다. 대신 image encoder는 대부분 고정하고, 소수의 학습 가능한 모듈과 visual prompt를 사용해 적응한다.

이 설계의 장점은 다음과 같다.

- 학습 비용이 낮다.
- foundation model의 일반 표현을 최대한 유지할 수 있다.
- 데이터가 적은 의료영상 환경에서 과적합 위험을 낮출 수 있다.

논문은 특히 적은 수의 추가 파라미터만으로도 강한 성능 향상이 가능하다고 주장한다. 이는 의료영상에서 full fine-tuning보다 parameter-efficient adaptation이 더 실용적일 수 있다는 메시지와 연결된다.

## 3. 2D와 3D를 모두 다루는 구조

ProMISe는 2D 이미지뿐 아니라 3D volume에도 적용되도록 설계됐다. 이 점이 중요하다. 많은 SAM 적응 연구가 2D medical images에 머무르는데, 이 논문은 multi-organ abdominal CT 같은 volumetric segmentation까지 평가한다.

저자들의 핵심 설계 관점은 다음과 같다.

- 2D에서는 클래스별 semantic mask prediction에 맞춰 prompt를 학습
- 3D에서는 slice-wise 또는 volume-aware adaptation을 통해 volumetric consistency를 확보
- 동일한 promptable framework로 2D/3D task를 아우름

즉, ProMISe는 단순 interactive tool이 아니라 범용 medical segmentation backbone으로 SAM을 바꾸려는 시도다.

## 4. 왜 기존 SAM adaptation과 다른가

논문은 point prompt, box prompt, adapter tuning, decoder tuning 같은 기존 SAM medical adaptation과 구별되는 지점을 분명히 한다.

- 기존 방법 다수는 여전히 test-time prompt 또는 수동 상호작용에 의존한다.
- ProMISe는 prompt 자체를 학습해 semantic segmentation에 맞춘다.
- 따라서 promptability를 유지하면서도 완전 자동 분할처럼 사용할 수 있다.

이 관점은 중요하다. 의료영상에서 SAM을 실제 segmentation model로 쓰려면 interactive prompt dependency를 줄여야 하기 때문이다.

## 실험 설정과 평가 구성

논문은 4개 데이터셋, 5개 세팅에서 ProMISe를 평가한다. 세부 표기상 주요 범주는 다음과 같다.

- 2D single-organ segmentation
- 2D multi-class segmentation
- multi-organ abdominal CT segmentation
- volumetric 3D segmentation

비교 대상에는 CNN 계열과 Transformer 계열의 대표 segmentation baseline이 포함된다. 논문은 단순 성능 수치뿐 아니라 parameter efficiency까지 함께 강조한다.

## 주요 결과 해석

## 1. 2D 의료영상 분할에서 강한 성능

원문 표에 따르면 2D 데이터셋들에서 ProMISe는 기존 SAM adaptation과 일반 segmentation baseline을 상회하는 결과를 보인다. 논문이 특히 강조하는 포인트는 다음과 같다.

- visual prompt를 학습하는 방식이 semantic segmentation에 더 잘 맞는다.
- point/box prompt 없이도 강한 분할 결과를 낸다.
- parameter-efficient setting에서도 경쟁력 있는 정확도를 얻는다.

이는 SAM의 구조적 장점이 의료영상에서도 유효하지만, prompt 정의를 바꾸지 않으면 성능이 제한된다는 논문의 주장을 뒷받침한다.

## 2. 3D multi-organ segmentation에서의 의미

논문은 BTCV 같은 multi-organ CT에서 ProMISe가 강한 Dice를 기록한다고 보고한다. 특히 원문은 `89.11% Dice` 수준의 성능을 언급하며, 강한 baseline인 nnUNet과 비교해도 개선이 있다고 주장한다.

이 결과가 중요한 이유는 다음과 같다.

- SAM adaptation이 2D toy setting에만 머무르지 않음을 보여준다.
- promptable framework가 실제 volumetric medical segmentation에도 적용 가능함을 시사한다.
- foundation model adaptation이 classic medical backbone을 대체할 가능성을 보여준다.

## 3. 적은 추가 파라미터 대비 높은 효율

논문은 full fine-tuning보다 적은 파라미터만 학습하면서도 좋은 성능을 낸다는 점을 반복해서 강조한다. 이 메시지는 실제 연구 환경에서 중요하다.

- 메모리 비용이 낮다.
- 작은 데이터셋에서도 활용하기 쉽다.
- 여러 기관이나 여러 과제에 빠르게 적응시키기 좋다.

즉, ProMISe의 실질적 가치는 최고 성능 자체보다도 `foundation model + efficient adaptation`의 구현 사례라는 데 있다.

## 논문의 핵심 메시지

### 1. SAM의 핵심은 이미지 인코더만이 아니라 promptability다

이 논문은 SAM을 단순히 큰 backbone으로 보지 않는다. 오히려 promptable design을 의료 semantic segmentation에 맞게 재정의하면 새로운 가능성이 열린다고 본다.

### 2. 의료영상에서는 prompt를 다시 설계해야 한다

point나 box는 자연영상 interactive segmentation에는 적합하지만, 장기 분할이나 multi-class semantic segmentation에는 비효율적일 수 있다. ProMISe는 이 prompt mismatch를 해결하려고 한다.

### 3. promptable segmentation은 interactive setting에만 국한되지 않는다

논문은 prompt를 학습 가능한 내부 표현으로 바꾸면, promptable segmentation과 automatic semantic segmentation의 경계를 흐릴 수 있음을 보여준다.

## 한계와 저자들이 시사하는 미래 방향

논문은 직접적으로 모든 한계를 길게 적지는 않지만, 결과와 구성에서 읽히는 한계는 분명하다.

### 1. 여전히 SAM backbone 품질에 의존한다

ProMISe의 성능은 기본적으로 SAM의 사전학습 표현에 기대고 있다. 따라서 modality gap이 더 큰 영상이나 희귀 도메인에서는 적응 난도가 올라갈 수 있다.

### 2. 3D medical imaging 전용 foundation model은 아니다

2D SAM을 기반으로 3D segmentation까지 확장하지만, 본질적으로 3D-native pretraining을 한 모델은 아니다. volumetric inductive bias 측면에서 한계가 남을 수 있다.

### 3. class-specific prompt 학습의 일반화 문제

학습된 prompt가 새로운 기관, 새로운 구조, unseen anatomy로 얼마나 일반화되는지는 더 검증이 필요하다.

### 4. foundation model adaptation의 해석 가능성 문제

왜 특정 visual prompt가 특정 구조를 잘 분할하는지에 대한 해석은 아직 충분하지 않다.

## 실무적 관점의 해설

### 1. 이 논문은 "SAM을 의료 segmentation backbone으로 재정의"하려는 시도다

단순히 SAM 위에 adapter를 올리는 수준이 아니라, prompt 자체를 semantic segmentation의 내부 표현으로 바꾸려는 접근이라는 점이 핵심이다.

### 2. prompt engineering보다 prompt learning에 가깝다

의료 도메인에서는 사람이 좋은 point/box prompt를 직접 넣기 어렵다. ProMISe는 그 부담을 모델 내부 학습으로 옮긴다. 이 점이 실제 자동화 파이프라인에 더 적합하다.

### 3. nnUNet 이후 시대의 한 신호다

의료영상 segmentation에서 한동안 강력한 공학적 baseline은 nnUNet이었다. ProMISe는 여기에 대해 `foundation model adaptation도 충분히 경쟁 가능하다`는 반례를 제시한다.

### 4. 하지만 범용 해법으로 보기는 아직 이르다

SAM adaptation 계열 전반이 그렇듯, 데이터셋 선택과 평가 프로토콜에 따라 성능 인상이 달라질 수 있다. 따라서 이 논문은 유망한 방향 제시로 읽는 편이 적절하다.

## 후속 연구와의 연결

이 논문은 다음 흐름과 자연스럽게 연결된다.

- SAM 기반 의료 segmentation의 promptable 자동화
- parameter-efficient foundation model adaptation
- 2D 자연영상 foundation model의 3D medical transfer
- universal segmentation, interactive segmentation, semantic segmentation의 통합

이후 연구는 여기서 더 나아가 SAM2, MedSAM 계열, 3D-native medical foundation model, multimodal promptable segmentation으로 확장될 가능성이 크다.

## 종합 평가

`ProMISe: Promptable Medical Image Segmentation using SAM`은 SAM을 의료 semantic segmentation에 맞게 재해석한 비교적 설득력 있는 연구다. 핵심은 point/box prompt를 버리고 learnable visual prompt를 도입했다는 점이며, 이를 통해 promptable segmentation을 자동 의료 분할 문제로 확장한다.

강점은 분명하다. parameter-efficient adaptation, 2D/3D 포괄, multi-organ 결과, foundation model 활용이라는 네 축이 잘 결합돼 있다. 반면 한계도 있다. 3D-native 모델은 아니고, 일반화와 해석 가능성은 더 검증이 필요하다. 그럼에도 이 논문은 `SAM을 의료영상에서 어떻게 실용적 segmentation model로 바꿀 것인가`라는 질문에 대한 좋은 출발점이다.
