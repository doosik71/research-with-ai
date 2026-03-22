# Zero-Shot Instance Segmentation

이 논문은 **학습 때는 seen class만 사용하고, 테스트 때는 seen + unseen class의 인스턴스를 모두 검출·분할하는 새로운 과제**인 **Zero-Shot Instance Segmentation (ZSI)**를 정식으로 제안합니다. 저자들은 기존 zero-shot classification, zero-shot detection, zero-shot semantic segmentation이 각각 한계가 있다고 보고, **“unseen 객체를 개별 인스턴스 단위로 정확히 segmentation”** 해야 하는 더 어려운 문제를 정의합니다. 이를 위해 **Zero-shot Detector**, **Semantic Mask Head (SMH)**, **Background Aware RPN (BA-RPN)**, **Synchronized Background Strategy (Sync-bg)**로 이루어진 end-to-end 프레임워크를 제안하고, COCO 기반 새 benchmark도 함께 제공합니다.  

## 1. Paper Overview

이 논문이 해결하려는 핵심 문제는, 실제 응용에서는 unseen class의 annotation을 충분히 모으기 어렵고, 기존 supervised instance segmentation은 그런 상황에 전혀 대응하지 못한다는 점입니다. 특히 의료·제조 같은 영역에서는 라벨링 비용이 크고 전문성이 필요해, unseen class를 위한 mask 데이터를 대량 확보하기 어렵습니다. 저자들은 이런 상황에서 단순 분류나 박스 검출만으로는 부족하고, **각 unseen object를 instance 단위로 분할하는 문제**가 필요하다고 주장합니다.

논문은 ZSI의 두 가지 핵심 난제를 명확히 짚습니다. 첫째, unseen class mask를 어떻게 예측할 것인가입니다. 학습 데이터에는 unseen class가 없으므로, seen class의 visual feature와 **pre-trained semantic word-vector** 사이의 대응 관계를 학습해 이를 unseen으로 전이해야 합니다. 둘째, unseen object를 background로 오인하는 문제입니다. zero-shot classification과 달리 ZSI는 foreground/background를 동시에 다뤄야 하므로, **background representation**이 특히 중요합니다.  

## 2. Core Idea

핵심 아이디어는 다음과 같습니다.

> **seen class의 visual-semantic mapping을 학습한 뒤, semantic word-vector를 통해 unseen class detection과 segmentation으로 전이하자.**

이를 위해 검출과 분할을 각각 semantic space와 연결합니다. 검출 쪽은 proposal feature를 semantic word-vector 공간으로 보내 unseen class도 분류할 수 있게 하고, 분할 쪽은 proposal의 visual feature를 semantic representation으로 바꾼 뒤, seen/unseen class의 word-vector를 convolution filter처럼 사용해 mask를 예측합니다. 즉 이 논문은 instance segmentation을 **semantic transfer 기반 detection-segmentation framework**로 바꿉니다.  

또 하나의 핵심은 **background를 고정된 “background” 단어 벡터로 처리하지 않는다**는 점입니다. 저자들은 기존 zero-shot detection이 background를 단순 단어 벡터 하나로 표현하는 것이 부적절하다고 비판합니다. 실제 background는 이미지마다 달라지므로, **동적으로 적응하는 background word-vector**가 필요하다고 보고 BA-RPN과 Sync-bg를 제안합니다.  

## 3. Detailed Method Explanation

### 3.1 문제 정의

논문은 seen class 집합 $C_s$와 unseen class 집합 $C_u$를 서로 겹치지 않게 둡니다. 학습은 seen class 이미지와 seen class word-vector만 사용합니다. 반면 테스트에서는 seen과 unseen이 함께 등장할 수 있고, 모델은 두 집합 모두에 대해 instance segmentation을 수행해야 합니다. 즉 training target은 seen only이지만 inference target은 seen + unseen입니다.

### 3.2 전체 구조

전체 프레임워크는 backbone에서 proposal feature를 뽑고, BA-RPN이 각 proposal에 대한 **background word-vector**를 생성합니다. 이후 이 background representation은 **Zero-shot Detector**와 **Semantic Mask Head**에 동기화되어 들어갑니다. 즉, proposal feature와 semantic embeddings를 detection과 mask prediction 양쪽에서 함께 사용합니다. 논문 Figure 2 설명도 바로 이 흐름을 요약합니다.

### 3.3 Zero-shot Detector

Zero-shot Detector는 proposal visual feature를 semantic word-vector 공간과 맞추는 구조입니다. 논문 설명에 따르면 encoder-decoder 방식으로 학습하지만, 테스트 시에는 **encoder만 사용**합니다. seen class와 background의 word-vector 집합 $W_s$, unseen class와 background의 word-vector 집합 $W_u$를 두고, proposal feature를 semantic space로 보낸 뒤 similarity 기반으로 class를 판단하는 방식입니다. 이렇게 하면 학습 때 unseen visual sample은 없지만, semantic space를 매개로 unseen detection이 가능해집니다.

### 3.4 Semantic Mask Head (SMH)

SMH는 논문의 segmentation 핵심입니다. 이 모듈도 encoder-decoder 구조이며, 학습 중에는 encoder가 visual feature를 semantic word-vector로 인코딩하고, decoder가 다시 visual feature를 복원하도록 하여 reconstruction loss $\mathcal{L}_R$를 사용합니다. 테스트 때는 decoder를 제거하고 encoder만 사용합니다. 이후 seen/unseen word-vector를 고정 convolution처럼 사용해 **pixel-by-pixel convolution**을 수행함으로써 seen과 unseen class의 instance segmentation 결과를 얻습니다. 즉 mask prediction을 semantic space에서 수행하는 구조입니다.

이 설계의 장점은 명확합니다. unseen class mask head를 직접 학습하는 대신, seen class에서 배운 visual-semantic alignment를 mask prediction으로 전이할 수 있기 때문입니다. 다만 논문이 제시한 방식은 semantic word-vector 품질에 강하게 의존합니다. 이 점은 뒤 실험에서도 확인됩니다.

### 3.5 BA-RPN과 Sync-bg

논문에서 가장 흥미로운 부분은 background 처리입니다. BA-RPN은 더 적절한 background word-vector를 학습하려는 proposal network이고, Sync-bg는 이 background vector를 **BA-RPN, detector, SMH 전체에 일관되게 전달**하는 전략입니다. 저자들은 BA-RPN만 단독으로 쓰면 오히려 성능이 떨어질 수 있다고 보고합니다. 핵심은 background vector를 한 모듈에서만 잘 배우는 것이 아니라, **전체 파이프라인에서 같은 background representation을 공유**해야 한다는 점입니다.  

논문의 해석도 설득력 있습니다. 예측은 먼저 detector가 bounding box를 만들고, 그 위에서 SMH가 segmentation을 수행합니다. 따라서 detector와 SMH가 서로 다른 background representation을 쓰면 inconsistency가 생겨 성능이 떨어집니다. 그래서 Sync-bg를 detector에 넣는 것이 특히 중요하다고 설명합니다.

## 4. Experiments and Findings

### 4.1 Benchmark와 평가 프로토콜

논문은 COCO 기반 ZSI benchmark를 새로 만듭니다. split은 **48/17**과 **65/15** 두 가지를 사용합니다. 각각 48 seen / 17 unseen, 65 seen / 15 unseen을 뜻합니다. 학습 세트는 COCO train에서 **seen class만 포함한 이미지**로 구성하되, unseen object가 조금이라도 포함된 이미지는 제거해 training 중 unseen이 절대 관측되지 않도록 합니다. 테스트 세트는 COCO val에서 unseen object를 포함한 이미지로 구성하며, seen과 unseen이 함께 등장할 수 있습니다.

평가는 두 설정으로 이뤄집니다.

* **ZSI**: unseen instance만 예측
* **GZSI**: seen + unseen instance를 함께 예측

지표는 zero-shot detection 관행을 따라 여러 IoU 기준(0.4, 0.5, 0.6)에서 **Recall@100**을 मुख्य 지표로 쓰고, IoU 0.5 기준 **mAP**도 참고로 제시합니다.

### 4.2 주요 결과

논문 abstract와 결론은 제안 방법이 기존 zero-shot detection SOTA를 넘어서는 동시에, ZSI에서도 유망한 성능을 보인다고 요약합니다. 특히 65/15 split 결과 일부 snippet에서는 ZSI가 **Recall@100 61.9 / 58.9 / 54.4**와 **mAP 13.6**을 기록한 것으로 보입니다. 이는 논문이 제안하는 detection + segmentation 전이 프레임워크가 실제 unseen instance segmentation까지 가능하게 만든다는 점을 보여줍니다.

또 GZSI에서는 baseline 대비 **harmonic average 기준 mAP 최대 5.61%, Recall@100 최대 11.72% 향상**을 보고합니다. seen과 unseen을 동시에 다뤄야 하는 더 어려운 setting에서도 의미 있는 개선이 있었다는 뜻입니다.

### 4.3 Component-wise analysis

Component 분석에서 저자들은 baseline 대비 자신들의 전체 방법이 **48/17과 65/15 split에서 ZSI Recall@100을 각각 6.4%, 7.4%, ZSD를 6.7%, 7.2%** 개선한다고 말합니다. 즉, detector만 좋아진 것이 아니라 segmentation setting에서도 누적 이득이 있었다는 주장입니다.

### 4.4 BA-RPN / Sync-bg ablation

이 논문에서 가장 중요한 ablation은 background 모듈입니다.

* **BA-RPN alone**: 오히려 성능 저하
* **Sync-bg in detector**: ZSD와 ZSI 모두 유의미한 향상
* **Sync-bg only in mask**: 개선이 거의 없고 때로는 ZSD 감소
* **전체 동기화(BA-RPN + detector + SMH)**: 최고 성능

즉 background vector를 “잘 학습하는 것”보다, **전체 모듈 간 일관성 있게 공유하는 것**이 더 중요하다는 결론입니다.  

### 4.5 Semantic information의 효과

semantic prior의 중요성도 실험으로 확인합니다. 논문은 one-hot vector를 word-vector 대신 쓰면 unseen 성능이 거의 random baseline 수준이 된다고 말합니다. 반면 **unannotated text로 사전학습된 word-vector**를 쓰는 제안 방식은 unseen segmentation 성능을 유지합니다. 예시 snippet에서는 semantic prior 사용 시 **Recall 58.9, mAP 13.6**을 보입니다. 이 결과는 이 논문이 사실상 **semantic transfer에 거의 전적으로 기대는 방식**임을 잘 보여줍니다.

## 5. Strengths, Limitations, and Interpretation

### 강점

이 논문의 가장 큰 강점은 **ZSI라는 문제 자체를 명확히 정의하고 benchmark까지 제공했다**는 점입니다. 이전에는 zero-shot classification, detection, semantic segmentation이 따로 있었지만, instance-level segmentation을 정면으로 다룬 setting은 부족했습니다.

둘째, 방법론적으로도 깔끔합니다. Zero-shot Detector로 box를, SMH로 mask를 semantic space에 연결하고, background 문제를 BA-RPN + Sync-bg로 보완합니다. 특히 background representation을 중요한 병목으로 본 시각은 꽤 통찰적입니다.  

셋째, ablation이 메시지를 분명히 합니다. semantic prior가 필수이고, background synchronization이 성능 핵심이라는 점을 비교적 설득력 있게 보여줍니다.  

### 한계

첫째, 이 방법은 **word-vector 기반 semantic alignment**에 매우 강하게 의존합니다. 즉 클래스 이름의 텍스트 의미가 visual similarity와 충분히 잘 맞아떨어져야 합니다. 의미가 모호하거나 시각적 유사성이 텍스트와 어긋나는 경우엔 취약할 수 있습니다. 이 해석은 논문의 semantic prior ablation에 근거합니다.

둘째, 구조가 완전히 단순하진 않습니다. detector, SMH, BA-RPN, Sync-bg가 얽혀 있어서 실제로는 “semantic detector + semantic mask + dynamic background”의 복합 시스템입니다. 따라서 이후 CLIP 기반 open-vocabulary 분할처럼 더 간단하고 강한 foundation model 기반 접근과 비교하면 과도기적 성격이 있습니다. 이는 논문 구조에 기반한 해석입니다.

셋째, 성능 지표 자체도 COCO standard AP가 아니라 **Recall@100 중심**으로 보고되어 있습니다. 물론 zero-shot detection 관행을 따른 것이지만, 오늘 기준 instance segmentation 비교에서는 다소 제한적으로 느껴질 수 있습니다.

## 6. Conclusion

이 논문은 **Zero-Shot Instance Segmentation**을 새로운 과제로 제안하고, 이를 풀기 위한 첫 baseline 성격의 프레임워크를 제공합니다. 핵심은 seen class에서 학습한 **visual-semantic mapping**을 unseen instance detection과 segmentation으로 전이하는 것이며, 이를 위해 **Zero-shot Detector**, **Semantic Mask Head**, **BA-RPN**, **Sync-bg**를 결합합니다. 특히 unseen을 background로 오인하는 문제를 **adaptive background representation**으로 다루려 한 점이 방법의 핵심 차별점입니다.  

연구적으로는 이 논문이 “zero-shot instance segmentation”이라는 문제를 구체화한 출발점이라는 의미가 큽니다. 실무적으로는 당시 기술 수준에서 unseen instance를 박스가 아니라 **인스턴스 마스크 단위로** 다뤘다는 점이 중요합니다. 이후 open-vocabulary instance segmentation이나 vision-language grounding 기반 방법들을 볼 때, 이 논문은 semantic transfer 중심의 초기 기준선으로 이해할 수 있습니다.
