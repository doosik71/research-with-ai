# 딥러닝 연구분야 총정리

이 저장소는 딥러닝의 주요 연구 분야를 주제별로 정리하고,
각 분야의 논문 조사 결과와 상세 분석 문서를 체계적으로 축적하기 위한
문서화 프로젝트이다.

문서 작업용 기준 위치는 저장소 루트가 아니라 `docs` 폴더이다.
아래에 언급하는 연구 분야 폴더와 안내 문서는 모두 `docs` 폴더 아래에 있다.

## 프로젝트 목적

- 딥러닝의 주요 연구 분야를 주제별 폴더로 정리한다.
- 각 분야에서 중요한 논문을 조사하고 목록화한다.
- 개별 논문에 대해 Markdown 기반의 상세 분석 보고서를 작성한다.
- 연구 기록을 일관된 형식으로 관리하고 재사용 가능하게 유지한다.

## 문서 운영 기준

이 프로젝트의 생성 및 관리는
[RULE.md](RULE.md)의 규칙을 따른다.

`RULE.md`에는 다음과 같은 기준이 정의되어 있다.

- 파일 및 폴더 명명 규칙
- Markdown 작성 규칙
- 수식 표기 규칙
- 논문 조사 및 요약 작업 절차
- `index.md`, `paper_list.jsonl` 관리 규칙
- 문서 lint 및 품질 점검 기준

## 주요 안내 문서

- 연구 분야 목록은 [TOPIC.md](TOPIC.md)에 정리되어 있다.
- 자주 사용하는 프롬프트 예시는 [PROMPT.md](PROMPT.md)에 정리되어 있다.
- 작업 규칙과 문서 관리 기준은 [RULE.md](RULE.md)를 따른다.

## 저장소 구조

`docs` 폴더 아래에서 각 연구 분야는 별도 폴더로 관리한다.
예를 들어 아래와 같은 분야 폴더가 존재한다.

- [model_efficiency_and_lightweight](model_efficiency_and_lightweight/index.md)
- [large_pretrained_models](large_pretrained_models/index.md)
- [explainable_ai](explainable_ai/index.md)
- [continual_learning](continual_learning/index.md)
- [deep_rl](deep_rl/index.md)
- [multi_agent_system](multi_agent_system/index.md)
- [generative_models](generative_models/index.md)
- [medical_bio](medical_bio/index.md)
- [ethics_fairness_privacy](ethics_fairness_privacy/index.md)
- [new_learning_paradigms](new_learning_paradigms/index.md)
- [hardware_software_optimization](hardware_software_optimization/index.md)
- [instance_segmentation](instance_segmentation/index.md)
- [semantic_segmentation](semantic_segmentation/index.md)
- [moving_object_tracking](moving_object_tracking/index.md)

각 폴더에는 아래 문서가 순차적으로 정리될 수 있다.

- `index.md`: 해당 분야 문서 목록
- `paper_list.jsonl`: 조사한 논문 메타데이터 목록
- 개별 논문 요약 문서: 논문별 상세 분석 보고서

## 작업 흐름

1. `docs` 폴더에서 [TOPIC.md](TOPIC.md)와 안내 문서를 먼저 확인한다.
2. 해당 분야 폴더에서 논문 조사와 문서 작성을 진행한다.
3. 필요 시 [PROMPT.md](PROMPT.md)의 프롬프트 예시를 참고한다.
4. 모든 문서는 [RULE.md](RULE.md)의 형식과 절차를 따른다.

## 비고

이 프로젝트는 딥러닝 연구 분야를 장기적으로 축적 가능한 문서 자산으로 관리하는 것을 목표로 한다.
