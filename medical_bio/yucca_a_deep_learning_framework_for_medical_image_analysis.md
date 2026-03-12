# Yucca: A Deep Learning Framework For Medical Image Analysis

## 논문 메타데이터

- **제목**: Yucca: A Deep Learning Framework For Medical Image Analysis
- **저자**: Sebastian Nørgaard Llambias, Julia Machnio, Asbjørn Munk, Jakob Ambsdorf, Mads Nielsen, Mostafa Mehdipour Ghazi
- **발표 연도**: 2024
- **출판 형태**: arXiv preprint
- **arXiv ID**: 2407.19888
- **DOI**: 10.48550/arXiv.2407.19888
- **arXiv URL**: https://arxiv.org/abs/2407.19888
- **PDF URL**: https://arxiv.org/pdf/2407.19888v1
- **코드 저장소**: https://github.com/Sllambias/yucca
- **PyPI 패키지**: https://pypi.org/project/yucca/

## 연구 배경 및 문제 정의

이 논문은 의료영상 딥러닝 연구에서 널리 쓰이는 프레임워크들이 서로 다른 장단점을 가진다는 점을 문제로 삼는다. 저자들의 관점에서 `nnU-Net`은 매우 강한 out-of-the-box 성능과 자동화 능력을 제공하지만 구조가 강하게 결합돼 있어 확장과 실험이 답답할 수 있다. 반면 `MONAI`는 유연하고 범용적이지만, 초보자나 빠른 재현이 필요한 연구자에게는 설정 부담이 크다.

논문이 정의하는 핵심 문제는 다음과 같다.

- 의료영상 연구자는 강한 baseline과 빠른 재현이 필요하다.
- 동시에 세부 모듈을 바꾸고 실험할 수 있는 유연성도 필요하다.
- 기존 프레임워크는 자동화와 모듈성 사이에서 한쪽을 희생하는 경우가 많다.
- 재현성, 파일 구조, 전처리, 학습, 추론, 평가를 한 시스템에서 관리하기 어렵다.

이를 해결하기 위해 저자들은 `Yucca`라는 오픈소스 프레임워크를 제안한다. 목표는 nnU-Net의 사용성 및 성능과 MONAI의 유연성 및 확장성을 동시에 확보하는 것이다.

## 핵심 기여

이 논문의 기여는 새로운 의료영상 모델이 아니라 `프레임워크 설계`에 있다. 핵심 기여는 다음과 같다.

1. 의료영상 전용 딥러닝 워크플로를 위한 세 계층 구조 `Functional`, `Modules`, `Pipeline`을 제안했다.
2. 데이터 구조화, 전처리, 학습, 추론 및 평가를 포함하는 end-to-end 파이프라인을 제공했다.
3. PyTorch와 PyTorch Lightning을 기반으로 하면서도 재현성, 파일 관리, 버전 관리, 체크포인트, 실험 추적 같은 엔지니어링 요소를 체계화했다.
4. cerebral microbleeds, white matter hyperintensity, hippocampus segmentation 등 다양한 의료영상 과제에 적용해 강건성과 범용성을 보여주었다.

즉, 이 논문은 모델 자체의 SOTA 경쟁보다는 "의료영상 연구자가 실험을 빠르고 안전하게 반복할 수 있는 공학적 기반을 어떻게 만들 것인가"에 답한다.

## 방법론 및 프레임워크 구조 요약

## 1. 세 계층 구조

Yucca의 가장 중요한 설계는 세 계층 구조다.

- **Functional tier**: 상태를 가지지 않는 순수 함수 집합
- **Modules tier**: 함수에 로직과 관례를 더한 객체 지향 계층
- **Pipeline tier**: end-to-end 워크플로를 자동화한 상위 계층

이 구분은 프레임워크의 철학을 드러낸다. 단순한 유틸리티 함수부터 완전한 파이프라인까지 동일한 생태계 안에서 제공해, 사용자 숙련도와 연구 목적에 따라 개입 수준을 조절하게 한 것이다.

## 2. Functional tier

Functional 계층은 `torch.nn.functional`에서 영감을 받은 stateless building block 모음이다. 여기에는 파일 입출력, 경로 조작, 무결성 검사, 배열 및 행렬 조작, 정규화, filtering, bounding box 계산 같은 기본 연산이 포함된다.

논문이 강조하는 포인트는 다음과 같다.

- 함수 단위라서 테스트와 디버깅이 쉽다.
- 더 큰 모듈이나 사용자 정의 구현의 재료가 된다.
- object-oriented pipeline 밖에서도 재사용할 수 있다.

복합 함수 예로는 `preprocess_case_for_inference` 같은 고수준 함수가 제시된다. 이는 많은 인자를 받아 여러 하위 함수를 조합해 일관된 전처리 과정을 수행한다.

## 3. Modules tier

Modules 계층은 Functional 계층을 감싸는 객체 지향 추상화다. 여기에는 callable transform, loss/metric wrapper, callback, network, DataModule, LightningModule 등이 포함된다.

저자들이 이 계층에 부여한 역할은 다음과 같다.

- 함수형 구성요소에 PyTorch/PyTorch Lightning 관례를 결합
- 데이터셋, dataloader, optimizer, scheduler, training/validation/inference step 관리
- 복잡한 학습 스크립트를 모듈 내부로 흡수

이 계층은 단순 유틸리티를 넘어, 모델 개발자가 실험 단위를 명확히 분리할 수 있게 만드는 중간 추상화라고 볼 수 있다.

## 4. Pipeline tier

Pipeline은 Yucca의 end-to-end 구현 계층이다. 저자들은 이를 nnU-Net에 가까운 사용 경험을 제공하면서도 내부는 더 교체 가능하게 설계했다고 설명한다.

전체 파이프라인은 네 단계로 구성된다.

1. task conversion
2. preprocessing
3. model training
4. inference and evaluation

이 네 단계는 의료영상 연구에서 자주 발생하는 실수, 특히 데이터 누수, 경로 혼란, 실험 버전 관리 실패를 줄이기 위해 강한 관례 기반으로 조직된다.

## 세부 파이프라인 분석

## 1. Task Conversion

Task conversion은 raw dataset을 Yucca 형식에 맞게 재구성하는 단계다. 이 과정에는 다음과 같은 실제 작업이 포함될 수 있다.

- 손상된 케이스 제거
- 라벨 수정
- 이미지-분할 쌍 정합
- 학습/테스트 분리

논문은 특히 test 폴더가 이후 단계에서 완전히 분리돼 데이터 또는 라벨 누수가 없도록 설계했다고 강조한다. 이는 프레임워크 차원의 재현성과 평가 신뢰성 확보 장치다.

## 2. Preprocessing

전처리 단계는 `Planner`와 `Preprocessor`가 담당한다.

- Planner는 데이터셋 통계를 분석해 전처리 계획 파일을 만든다.
- Preprocessor는 그 계획을 실제 실행한다.

예를 들어 기본 `YuccaPlanner`는 데이터셋 통계를 이용해 voxel spacing을 동적으로 추론하고, `YuccaPlanner_MaxSize`는 가장 큰 이미지 크기에 맞춰 정적으로 resample한다. 분할용 전처리기와 분류용 전처리기도 구분된다.

중요한 점은 이 단계에서 적용되는 연산이 항상 동일한 deterministic preprocessing이라는 것이다. 반면 랜덤 증강은 학습 단계에서 온라인으로 적용된다.

## 3. Training

학습 단계의 중심은 `manager` 개념이다. manager는 사용자가 직접 지정하지 않은 여러 실험 설정을 데이터셋 통계, 플래너 결과, 검증된 휴리스틱에 따라 자동으로 채운다.

자동화되는 대표 항목은 다음과 같다.

- random seed
- checkpoint와 logging 설정
- 경로와 버전 네이밍
- train/validation split 재사용
- spatial dimension 추론
- augmentation pipeline 구성
- 학습 중단 후 동일 설정 실험의 자동 재개

저자들은 이런 자동화가 단순 편의 기능이 아니라, 재현성과 과학적 엄밀성을 보장하는 핵심 장치라고 본다.

## 4. Inference and Evaluation

추론 단계에서는 학습 시와 동일한 전처리 파이프라인을 테스트 데이터에 적용한 뒤 예측을 수행한다. patch-based model의 경우 sliding-window inference도 지원한다. 이후 전처리 중 label-preserving하지 않은 조작, 예를 들어 transpose나 resampling을 역변환해 원래 좌표계로 복원한 다음 저장과 평가를 수행한다.

이 부분의 장점은, 추론 스크립트가 제각각 구현되면서 생기는 평가 오류를 프레임워크 차원에서 줄일 수 있다는 점이다.

## 실험 설정과 결과

이 논문은 전형적인 새 알고리즘 논문처럼 단일 벤치마크에서 표를 촘촘히 비교하지는 않는다. 대신 Yucca가 이미 다양한 실제 프로젝트에서 사용되어 강건성을 보였다는 점을 사례 중심으로 제시한다.

논문이 언급하는 대표 적용 영역은 다음과 같다.

- COVID-19 환자 cerebral microbleeds segmentation 및 detection
- white matter hyperintensity segmentation 및 detection
- hippocampus segmentation
- stroke 및 multiple sclerosis 관련 brain lesion segmentation, detection, classification

본문의 정량 서술은 제한적이지만, 저자들은 Yucca가 이러한 여러 2D/3D 설정에서 state-of-the-art 결과를 달성했다고 주장한다. 추가로 MICCAI Multi-Atlas Challenge brain MRI를 이용한 3D hippocampus segmentation 시각화 예시를 제공하며, axial/sagittal/coronal 2D U-Net, 2D ensemble, 3D U-Net 결과를 비교해 framework의 범용성을 보여준다.

중요한 해석은 이 논문의 "성능"이 특정 새 네트워크의 절대 수치가 아니라, 하나의 프레임워크가 서로 다른 작업과 아키텍처를 안정적으로 지원한다는 데 있다는 점이다.

## 강점

## 1. 자동화와 유연성의 절충이 명확하다

Yucca는 nnU-Net처럼 end-to-end baseline을 빠르게 얻을 수 있으면서도, MONAI처럼 내부 구성요소를 교체하고 재조합하기 쉽게 설계됐다.

## 2. 의료영상 워크플로 전체를 다룬다

단순 학습 코드가 아니라 task conversion, preprocessing, training, inference, evaluation 전 단계를 포괄한다.

## 3. 재현성을 프레임워크 수준에서 다룬다

실험 경로, split 저장, checkpoint, versioning, logging 같은 요소를 자동화해 연구 실수를 줄이도록 설계했다.

## 4. 다양한 사용자 층을 겨냥한다

초보자는 Pipeline을 쓰고, 숙련자는 Functional/Modules를 직접 조합할 수 있어 진입 장벽과 확장성을 동시에 고려했다.

## 한계와 비판적 검토

## 1. 논문 자체의 정량 근거는 제한적이다

state-of-the-art를 달성했다고 서술하지만, 본문에는 대규모 비교표나 체계적 ablation이 많지 않다. 따라서 프레임워크 우수성을 완전히 논문만으로 검증하기보다는 관련 프로젝트와 코드 사용 경험까지 함께 봐야 한다.

## 2. 기여가 알고리즘이 아니라 공학 구조에 치우친다

이 점은 장점이기도 하지만, 학술적으로는 새로운 학습 원리나 모델 구조보다 소프트웨어 엔지니어링 기여가 중심이라서 독자에 따라 임팩트가 다르게 느껴질 수 있다.

## 3. nnU-Net 대비 성능-유연성 trade-off의 체계적 분석이 부족하다

논문은 nnU-Net과 MONAI 사이의 균형을 목표로 하지만, 실제로 어느 수준까지 자동화 성능을 유지하면서 유연성을 확보했는지에 대한 정량적 비교는 더 있으면 좋았을 것이다.

## 4. 대규모 커뮤니티 검증은 아직 초기 단계다

오픈소스 프레임워크의 진짜 가치는 시간이 지나며 다양한 외부 연구자들이 사용하고 검증할 때 드러난다. 2024년 논문 시점에서는 그 생태계가 아직 형성 중이다.

## 실무적 및 연구적 인사이트

이 논문이 주는 가장 실질적인 교훈은 의료영상 연구에서 "좋은 모델" 못지않게 "좋은 실험 인프라"가 중요하다는 점이다. 데이터 정리, 전처리, split 관리, 경로 버전, 추론 복원, 평가 저장 같은 반복 작업은 논문에서는 작게 보이지만 실제 연구 품질에 큰 영향을 준다. Yucca는 바로 이 공학적 마찰을 줄이려는 시도다.

또한 Yucca는 의료영상 프레임워크가 단순 라이브러리나 자동 AutoML 시스템 중 하나를 택할 필요가 없음을 보여준다. 순수 함수 계층, 모듈 계층, 파이프라인 계층을 분리하면 재사용성과 실험 자유도, 사용 편의성을 동시에 추구할 수 있다.

향후 관점에서 보면 Yucca는 segmentation 중심 프레임워크를 넘어 self-supervised learning, classification, domain adaptation, multimodal medical learning으로 확장될 수 있는 공학적 기반으로 읽을 수 있다.

## 종합 평가

`Yucca: A Deep Learning Framework For Medical Image Analysis`는 새로운 의료영상 모델을 제안한 논문이라기보다, 의료영상 연구를 더 재현 가능하고 실험 친화적으로 만들기 위한 소프트웨어 프레임워크 논문이다. 핵심은 nnU-Net의 강한 기본값과 MONAI의 유연성을 절충하려는 설계 철학에 있으며, 이를 Functional, Modules, Pipeline의 세 계층 구조로 구현했다.

정량 비교의 밀도는 높지 않지만, 실제 연구 현장에서 반복되는 엔지니어링 문제를 정면으로 다루고 있다는 점에서 가치가 있다. 따라서 이 논문은 알고리즘 혁신 문헌이라기보다, 의료영상 딥러닝 연구 생산성을 높이는 인프라 문헌으로 읽는 것이 가장 적절하다.
