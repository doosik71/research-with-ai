# Time Series Prediction Using Deep Learning Methods in Healthcare

이 논문은 의료 시계열 예측에서 전통적 machine learning이 겪는 한계, 즉 **고차원성**, **강한 시간 의존성**, **불규칙성**, **결측치**, **복합 입력 구조**를 deep learning이 어떻게 극복해 왔는지를 체계적으로 정리한 **systematic review**다. 저자들은 구조화된 환자 시계열 데이터를 이용한 healthcare prediction 연구를 선별해 분석했고, 그 결과 문헌의 핵심 기여를 **10개 연구 흐름**으로 분류했다. 이 논문의 본질적 공헌은 새로운 모델을 제안하는 것이 아니라, 의료 시계열 예측 DL 문헌을 기술적 관점에서 재구성하고, 무엇이 잘 되었고 무엇이 아직 미해결인지 정리한 데 있다.  

## 1. Paper Overview

저자들의 문제의식은 분명하다. 의료 데이터는 수천 개 후보 변수 중 일부만 선택해야 할 정도로 고차원이며, 결측과 불규칙성이 심하고, 환자 상태가 시간에 따라 변화한다. 전통적 ML은 이를 다루기 위해 평균·표준편차 같은 거친 집계와 수작업 feature engineering에 크게 의존하는데, 이 방식은 의료 이벤트의 순차성, 장기 의존성, 질병 진행과 개입(intervention)의 상호작용을 충분히 활용하지 못한다. 반면 DL은 원시 혹은 최소 전처리 데이터로부터 유용한 표현을 학습하고, temporal pattern을 더 직접적으로 모델링할 수 있다는 점에서 대안으로 제시된다.

이 논문은 이미지나 임상노트 같은 비정형 데이터는 제외하고, **structured patient time series**에 한정해 DL prediction literature를 검토한다. 그리고 “어떤 입력 표현이 쓰였는가”, “결측과 irregularity를 어떻게 처리했는가”, “어떤 아키텍처가 효과적인가”, “의료 ontology와 해석 가능성은 어떻게 결합되는가”, “모델이 얼마나 확장 가능한가”라는 질문들을 중심으로 문헌을 재구성한다. 저자들은 이것이 의료 시계열 예측 DL 연구를 기술적 특성 중심으로 정리한 첫 review라고 주장한다.  

## 2. Core Idea

이 논문의 핵심 아이디어는 의료 시계열 예측 문헌을 단순히 “모델 이름 목록”으로 보지 않고, **10개의 기술적 연구 축**으로 정리하는 데 있다. 그 10개는 다음과 같다: patient representation, missing value handling, DL models, addressing temporal irregularity, attention mechanisms, incorporation of medical ontologies, static data inclusion, learning strategy, interpretation, scalability. 이 taxonomy가 논문의 중심 구조다.  

또한 저자들은 문헌 분석을 위해 각 연구에서 공통적으로 추출할 10개의 핵심 속성도 정의한다. 여기에는 medical task, database, input features, preprocessing, patient representation, DL architecture, output temporality, performance, benchmark, interpretation이 포함된다. 즉, 이 논문은 “어떤 모델이 제일 좋다”를 단정하기보다, **어떤 문제 설정과 데이터 조건에서 어떤 설계 선택이 주로 사용되고, 어떤 선택이 더 유망해 보이는가**를 정리하려는 문헌 지도(map)에 가깝다.

## 3. Detailed Method Explanation

### 3.1 Review methodology

이 논문은 systematic review로서 PRISMA 지침을 따랐고, MEDLINE, IEEE, Scopus, ACM Digital Library를 검색 대상으로 삼았다. 초기 검색 결과는 1,524편이었고, 중복 제거 후 1,014편의 제목/초록을 검토했으며, 최종적으로 77편을 포함했다. 포함 기준은 구조화된 환자 시계열 데이터를 활용하는 DL 기반 healthcare prediction 연구였고, 비정형 데이터 기반 연구나 핵심 기술 정보가 부족한 논문은 제외되었다. 그림 1은 이 필터링 과정을 도식화한다.  

### 3.2 Patient representation

저자들은 환자 표현을 크게 **sequence representation**과 **matrix representation**으로 구분한다. Sequence 방식은 진단/처치/약물 코드의 순서를 살린 이벤트 시퀀스 표현이고, matrix 방식은 시간 간격별로 feature를 정렬한 longitudinal matrix다. 문헌 전체로 보면 sequence representation이 약간 더 많았고(57%), 수치형 입력이 많을 때는 sequence, 진단코드·처치코드 같은 범주형 입력이 많을 때는 matrix가 더 자주 쓰였다. 또한 sequence 표현에서는 embedding layer, pre-trained word2vec/Skip-Gram, code grouping(CCS 등)이 주요 기법으로 사용되었다. 저자들의 종합 해석은 **EHR에는 pre-trained embedding이 붙은 sequence representation**, **AC에는 matrix representation**이 상대적으로 유리하다는 것이다.  

### 3.3 Missing value handling

결측치 처리는 zero, median, forward-backward, 전문가 지식 기반 대치가 가장 흔했지만, 의료 데이터의 결측은 often informative missingness라는 점이 강조된다. 즉, 어떤 값이 없다는 사실 자체가 임상적 의사결정의 흔적일 수 있다. 이를 반영하는 대표 방법은 두 가지다. 하나는 각 시점 feature의 관측 여부를 나타내는 **masking vector**를 별도 입력으로 넣는 것이고, 다른 하나는 최신 관측값이나 평균값을 바탕으로 **missing pattern 자체를 학습**하는 것이다. 저자들은 후자가 특정 조건에서는 더 강력할 수 있으나, 다양한 응용에서 더 널리 검증된 것은 masking vector라고 정리한다.  

### 3.4 Deep learning models

가장 중심이 되는 결과는 **RNN 계열, 특히 GRU와 LSTM이 의료 시계열 예측의 주류**라는 점이다. 포함 논문의 84%가 RNN/LSTM/GRU 계열을 사용했다. 일부 비교에서는 GRU가 LSTM보다 약 1% AUC 우위를 보였지만, 다른 작업에서는 유의미한 차이가 없었다. Bidirectional GRU/LSTM은 긴 ICU 시계열 같은 문제에서 단방향보다 일관되게 우수했다. 또한 channel-wise learning, target replication, CNN-RNN hybrid 같은 설계가 성능 향상 가능성을 보였고, 최근에는 CNN을 독립 모델로 쓰기보다 **RNN의 global pattern modeling을 보완하는 local pattern extractor**로 결합하는 방향이 유망하다고 평가한다. 저자들은 종합적으로 **bidirectional recurrent network가 현재 가장 강력하며, 복잡도와 성능 균형상 GRU를 선호**하는 결론을 제시한다.  

### 3.5 Temporal irregularity, attention, ontology

의료 시계열에는 visit irregularity와 feature irregularity가 있다. 가장 널리 쓰인 방법은 visit 간 시간 간격을 독립 입력으로 추가하는 것이고, 좀 더 적극적인 접근은 RNN 메모리 셀을 수정해 최근 방문에 더 큰 가중치를 주거나 feature별 decay pattern을 따로 학습하는 것이다. 다만 저자들은 이 분야가 아직 비교 실험이 부족하다고 본다.  

Attention에서는 location-based, general, concatenation-based 세 종류가 대표적으로 등장하며, 문헌상 location-based attention이 가장 흔하고 자주 우수했다고 정리된다. 하지만 대부분의 연구가 attention을 성능 개선보다 **해석 가능성 도구**로 쓰고 있어, attention이 실제 예측 성능에 얼마나 기여하는지는 충분히 분리 평가되지 않았다고 비판한다.  

의료 ontology 통합에서는 CCS 계층 트리, ancestor-aware ontology, causal knowledge graph 등이 embedding 또는 attention 계산에 사용되었다. 특히 rare disease 예측에서는 상위 개념을 통해 표현을 보강하는 효과가 강조된다. 하지만 데이터가 충분한 경우 그 이점이 줄어들 수 있어, 모든 상황에서 무조건 강력한 것은 아니라는 점도 지적한다.  

### 3.6 Static data, learning strategy, interpretation, scalability

정적 정보(인구통계, 과거력) 결합 방식은 네 가지로 요약된다. 마지막 fully connected layer에 붙이는 방식, 별도 FFN으로 인코딩해 결합하는 방식, 매 시점에 반복 입력하는 quasi-dynamic 방식, 그리고 recurrent memory 자체를 수정하는 방식이다. 하지만 저자들은 이 네 방식의 체계적 비교 연구가 없다고 지적한다.  

학습 전략으로는 cost-sensitive learning, multi-task learning, transfer learning이 등장한다. Cost-sensitive는 불균형 문제에서 minority class 비용을 반영하고, multi-task는 mortality/LOS/phenotyping 같은 다중 과제를 함께 학습하며, transfer learning은 아직 연구 수가 적지만 데이터가 부족한 환경에서 큰 잠재력을 가진 것으로 정리된다.  

해석 가능성 측면에서는 attention visualization이 가장 흔하지만, 저자들은 Shapley/DeepLIFT 같은 feature importance도 일관되게 함께 보고할 것을 권한다. 또한 개인 수준 해석뿐 아니라 **population-level pattern extraction**이 더 필요하다고 강조한다.  

확장성 측면에서는 대형 EHR에서 DL이 충분히 확장될 수 있다는 보고가 있지만, AC 데이터에서는 전통 ML ensemble과 비교해 뚜렷한 우위가 없을 수 있다. 즉, DL의 우수성은 데이터 유형과 feature richness에 따라 달라질 수 있다는 것이 저자들의 판단이다.  

## 4. Experiments and Findings

이 논문은 실험 논문이 아니라 review이므로, “실험 결과”는 저자들이 선별한 77편의 문헌에서 도출한 종합 결론이다. 먼저 그림 1과 2, 표 2에 따르면 최종 포함 논문은 77편이며, 77%가 2018년 이후에 출판되었다. 예측 과제로는 mortality, heart failure, readmission, next-visit diagnosis가 가장 많았고, 데이터셋으로는 **MIMIC**이 가장 자주 사용되었다. 이는 의료 시계열 DL이 최근 몇 년간 급증했고, ICU 중심 공개 데이터에 강하게 편중되어 있다는 의미다.

가장 중요한 실질적 발견은 다음과 같다. 의료 시계열 예측에서는 **RNN 계열이 CNN보다 전반적으로 우세**했고, bidirectional recurrent model과 hybrid CNN-RNN이 유망했다. GRU와 LSTM은 대체로 비슷하지만 구현 복잡도 면에서 GRU가 선호될 수 있다. 결측은 단순 대치보다 masking이나 missing pattern learning이 더 적절하며, irregularity는 time interval 입력이나 memory-cell modification으로 다뤄졌다. Attention은 주로 해석 가능성 향상에 쓰였고, ontology는 특히 rare disease 표현 보강에 유용했다. 하지만 static data inclusion, attention의 독립적 성능 효과, AC로의 일반화 같은 주제는 여전히 비교 연구가 부족했다.  

저자들은 결론에서 이 모든 내용을 다시 요약하며, patient representation, missing values, DL models, temporal irregularity, attention, ontology, static data, learning strategies, interpretation, scalability 각각에 대해 “현재까지의 best practice와 남은 공백”을 제시한다. 이 논문의 가치는 바로 이 정리의 밀도에 있다.

## 5. Strengths, Limitations, and Interpretation

이 논문의 가장 큰 강점은 **survey의 축이 명확하고 기술적 깊이가 충분하다**는 점이다. 단순 application 나열이 아니라 patient representation, missingness, irregularity, ontology, interpretability 같은 실제 설계 문제를 중심으로 문헌을 재분류한다. 또한 그림 3은 의료 시계열 DL 설계를 입력 표현부터 학습 전략, 해석, 확장성까지 한 장으로 요약해 field map 역할을 한다.  

둘째, 단순한 찬양이 아니라 **비교 부족과 과도한 주장**을 꾸준히 경계한다. 예를 들어 sequence vs matrix representation, GRU vs LSTM, attention의 성능 기여, static data integration 방식, multi-layer 효과, EHR에서 AC로의 일반화 모두에서 “아직 robust benchmarking이 부족하다”고 반복해서 지적한다. 이 때문에 논문이 survey로서 비교적 신뢰할 만하다.  

한계도 있다. 우선 이 논문은 2021년까지의 문헌을 대상으로 하므로, 이후 급속히 발전한 **Transformer 계열 patient models**나 foundation-model-style EHR modeling까지는 충분히 다루지 못한다. 둘째, review 대상이 구조화 시계열에 한정되어 있어 imaging, note, waveform을 결합한 multimodal clinical modeling은 범위 밖이다. 셋째, 문헌의 다수가 MIMIC 같은 특정 공개 데이터에 치우쳐 있어, review의 결론도 그런 데이터 편향을 일부 반영한다. 이 점은 저자들 스스로 scalability와 data setting generalization 문제로 인정한다.  

비판적으로 해석하면, 이 논문의 진짜 메시지는 “GRU가 최고다” 같은 단순 결론이 아니다. 더 본질적으로는, **의료 시계열 예측은 단순 모델 선택 문제가 아니라 데이터 표현, 결측 구조, 시간 불규칙성, 정적·동적 정보 결합, 해석 가능성, 데이터셋 편향을 함께 설계해야 하는 시스템 문제**라는 점을 보여준다. 이 관점이 가장 중요하다.

## 6. Conclusion

이 논문은 의료 구조화 시계열 예측을 위한 deep learning 연구를 체계적으로 정리한 수준 높은 systematic review다. 전통 ML의 feature engineering 중심 접근이 가진 한계를 출발점으로 삼아, 의료 시계열 DL 연구를 10개의 기술 축으로 구조화했고, 각 축에서 현재까지의 대표 전략과 남은 연구 공백을 정리했다. 핵심 결론은 다음과 같이 요약할 수 있다. **RNN 계열, 특히 GRU/LSTM이 주류이며, bidirectional recurrent model과 CNN-RNN hybrid가 유망하다. 결측과 irregularity는 단순 전처리보다 모델 내부에서 다루는 방향이 중요하며, attention과 ontology는 해석 가능성과 rare disease modeling에서 유용하다. 하지만 많은 분야에서 robust benchmarking이 아직 부족하다.**  

실무적으로는 이 논문이 의료 시계열 모델 설계 체크리스트처럼 읽힌다. 연구적으로는 EHR/claims 기반 DL 예측을 더 공정하게 비교하고, 어느 데이터 조건에서 DL이 באמת 필요한지 규명해야 한다는 문제를 남긴다. 따라서 이 논문은 단순 survey를 넘어서, 이후 healthcare time-series modeling 연구의 설계 기준을 제시한 메타 연구라고 볼 수 있다.  
