# Multi-modal Vision Pre-training for Medical Image Analysis

## 논문 메타데이터

- **제목**: Multi-modal Vision Pre-training for Medical Image Analysis
- **본문 표기 제목**: BrainMVP: Multi-modal Vision Pre-training for Brain Image Analysis using Multi-parametric MRI
- **저자**: Shaohao Rui, Lingzhi Chen, Zhenyu Tang, Lilong Wang, Mianxin Liu, Shaoting Zhang, Xiaosong Wang
- **학회**: CVPR 2025
- **arXiv 공개 연도**: 2024
- **최신 arXiv 버전 날짜**: 2025-03-14
- **arXiv ID**: 2410.10604
- **DOI**: 10.48550/arXiv.2410.10604
- **arXiv URL**: https://arxiv.org/abs/2410.10604
- **PDF URL**: https://arxiv.org/pdf/2410.10604v2
- **코드 저장소**: https://github.com/shaohao011/BrainMVP

## 연구 배경 및 문제 정의

이 논문은 의료영상 self-supervised pre-training이 대부분 단일 모달리티 이미지에 집중해 왔고, 그 결과 다중 파라메트릭 MRI(mpMRI)처럼 실제 임상에서 중요한 모달 간 상관관계를 충분히 학습하지 못했다는 문제의식에서 출발한다. 저자들은 특히 뇌 MRI에서 서로 다른 모달리티가 병변의 다른 측면을 보여주기 때문에, 이 상보성을 사전학습 단계에서 직접 반영해야 더 범용적인 표현을 얻을 수 있다고 본다.

논문이 정의하는 핵심 문제는 다음과 같다.

- 기존 의료영상 SSL은 주로 uni-modal pre-training에 머무른다.
- mpMRI는 실제로 결측 모달리티가 흔해서 고정 모달 수를 전제로 한 학습이 확장성을 해친다.
- 일반적인 masked image modeling은 다운스트림의 multi-modal fusion 요구와 직접 연결되지 않을 수 있다.
- 의료영상 foundation model을 만들려면 cross-modal correlation과 missing modality 문제를 함께 다뤄야 한다.

이를 해결하기 위해 저자들은 뇌 mpMRI용 multi-modal pre-training 프레임워크 `BrainMVP`를 제안한다.

## 핵심 기여

논문이 주장하는 핵심 기여는 세 가지다.

1. 8개 MRI 모달리티를 포함하는 16,022개 brain scan, 3,755명 규모의 대형 pre-training 데이터셋을 구축했다.
2. cross-modal reconstruction, modality-wise data distillation, modality-aware contrastive learning의 세 proxy task를 결합한 BrainMVP를 제안했다.
3. 6개 segmentation과 4개 classification을 포함한 총 10개 다운스트림 과제에서 기존 의료영상 SSL보다 더 강한 성능과 일반화를 보였다.

이 논문은 의료영상용 multi-modal pre-training을 image-text 정렬이 아니라 `같은 환자 내 여러 영상 모달리티의 상호 관계 학습`으로 재정의했다는 점에서 의미가 있다.

## 방법론 요약

BrainMVP는 세 개의 핵심 구성요소로 이루어진다.

## 1. Cross-Modal Reconstruction

첫 번째 구성요소는 cross-modal reconstruction(CMR)이다. 표준 MIM처럼 패치를 0이나 노이즈로 가리는 대신, 마스킹된 영역을 `다른 MRI 모달리티 이미지`로 채워 넣고 원래 모달리티를 복원하게 만든다.

이 설계의 핵심 논리는 다음과 같다.

- 서로 다른 MRI 모달리티는 해부학적으로는 유사하지만 신호 특성은 다르다.
- 따라서 한 모달리티를 다른 모달리티 정보로 복원하게 하면 modality-invariant structure와 modality-specific difference를 동시에 학습할 수 있다.
- missing modality 상황에서도 고정 모달 수가 아니라 single-modal input 기반으로 확장 가능한 pre-training을 할 수 있다.

즉, 이 논문의 reconstruction은 단순 복원 과제가 아니라 `cross-modal fusion capability`를 학습하기 위한 장치다.

## 2. Modality-wise Data Distillation

두 번째 구성요소는 modality-wise data distillation(MD)이다. 저자들은 각 모달리티마다 학습 가능한 `modality template`를 두고, 이것이 해당 모달리티의 압축된 구조 통계를 담도록 학습한다.

이 모듈의 역할은 두 가지다.

- pre-training 단계에서 각 모달리티의 본질적 구조 정보를 응축
- downstream 단계에서 입력 일부를 modality template로 치환해 upstream-downstream linkage 형성

저자들은 이 template가 개별 환자 데이터 그 자체가 아니라 모달리티 수준의 condensed representation이기 때문에, 구조 정보는 유지하면서 privacy leakage 우려는 줄일 수 있다고 해석한다.

실제로 이 모듈은 BrainMVP의 독창성이 가장 강하게 드러나는 부분이다. 단순 reconstruction이나 contrastive learning을 넘어서, 사전학습 산출물을 downstream adaptation에 직접 연결하는 매개체를 만든 셈이다.

## 3. Modality-aware Contrastive Learning

세 번째 구성요소는 modality-aware contrastive learning(MCL)이다. 저자들은 다른 장비, 다른 병원, 다른 acquisition protocol에서 생기는 데이터 분포 차이가 downstream generalization을 해칠 수 있다고 본다.

MCL은 다음을 목표로 한다.

- 같은 샘플의 서로 다른 masking/mixing 버전은 가깝게
- 서로 다른 샘플은 멀게
- 동시에 modality-aware한 의미 일관성 유지

이렇게 해서 case-level semantic consistency와 dataset/modality independent representation을 함께 학습하려 한다.

## 4. 구현 전략의 실용적 포인트

이 논문은 missing modality 문제 때문에 pre-training 단계에서 고정된 multi-channel 입력을 사용하지 않고, `single-channel modality image input` 기반으로 확장성을 확보한다. 이 선택이 중요한 이유는 실제 의료 데이터에서 모든 환자가 같은 모달 세트를 갖지 않는 경우가 많기 때문이다.

또한 backbone으로는 UniFormer를 사용해 멀티모달 융합 능력을 확보했고, downstream에서는 UniFormer와 UNET3D 양쪽에 대해 실험을 수행했다.

## 실험 설정

## 1. Pre-training 데이터

사전학습은 5개의 공개 brain mpMRI 데이터셋으로 수행됐다.

- 총 3,755 cases
- 총 16,022 scans
- 8개 MRI modality
- 240만 장 이상의 이미지

포함된 대표 데이터는 BraTS2021, BraTS2023-SSA, BraTS2023-MEN, BrainAtlas, UCSF-PDGM이다.

## 2. Downstream 과제

평가는 총 10개 공개 과제로 수행됐다.

- **Segmentation 6개**: BraTS-PED, BraTS2023-MET, ISLES22, MRBrainS13, UPENN-GBM, VSseg
- **Classification 4개**: BraTS2018, ADNI, ADHD-200, ABIDE-I

이 구성은 소아 뇌종양, 뇌전이, 허혈성 병변, 조직 분할, 신경퇴행성 질환, 발달장애 분류까지 꽤 다양한 임상 시나리오를 포괄한다.

## 3. 학습 세부 설정

논문 기준 주요 설정은 다음과 같다.

- 프레임워크: PyTorch, MONAI 1.3.0
- backbone: UniFormer
- optimizer: AdamW
- GPU: 8 x NVIDIA GeForce RTX 4090
- 전처리: RAS 방향 통일, 공통 해부학 템플릿 정합, isotropic resampling, skull stripping, intensity clipping 및 정규화

특이한 점은 pre-training에서 별도의 강한 augmentation을 거의 쓰지 않았다는 점이다. 저자들은 대신 cross-modal mixing 자체가 더 중요한 학습 신호라고 본다.

## 실험 결과와 해석

## 1. Segmentation 성능

논문은 6개 segmentation 벤치마크에서 BrainMVP가 기존 사전학습 방법보다 일관되게 우수하다고 보고한다. arXiv 초록 기준 평균 Dice 향상 폭은 `0.28%~14.47%`다.

본문에서 강조된 대표 결과는 다음과 같다.

- BraTS-PED 평균 Dice: scratch UniFormer `72.52` -> BrainMVP `76.80`
- UPENN-GBM 평균 Dice: `90.01`
- 여러 segmentation 데이터셋에서 MAE3D, Models Genesis, VoCo, M3AE, DAE 등을 상회

저자들은 특히 2D 자연영상 중심 사전학습이 3D medical volume에 잘 맞지 않는 반면, BrainMVP는 3D mpMRI 구조와 모달 관계를 직접 학습하기 때문에 강점을 가진다고 해석한다.

## 2. Classification 성능

분류 과제에서도 일관된 향상이 보고된다. 초록 기준 accuracy 향상 폭은 `0.65%~18.07%`다.

대표 결과는 다음과 같다.

- BraTS2018: ablation 최종 모델 기준 ACC `0.8596`, AUC `0.9452`, F1 `0.8324`
- ADNI: ACC `0.6765`, AUC `0.6964`, F1 `0.6609`
- ADHD-200: ACC `0.6883`

특히 ADNI는 pre-training 데이터와 질병 분포가 다르고 정상 뇌 데이터 비중도 적은데, 여기서도 일반화가 유지됐다는 점을 저자들은 중요한 근거로 제시한다.

## 3. Label efficiency

이 논문의 강한 포인트 중 하나는 label efficiency다. 저자들은 40% 라벨만으로도 타 방법의 full-label 성능에 맞먹거나 이를 넘는 결과를 보고한다.

대표 수치는 다음과 같다.

- BraTS-PED 20% label: `66.41` Dice
- VSseg 20% label: `70.39` Dice
- UPENN-GBM 20% label: `86.82` Dice

이는 multi-modal pre-training이 단순 평균 성능뿐 아니라 annotation 효율성 측면에서도 실질적 가치를 가진다는 뜻이다.

## 4. Ablation 결과

세 구성요소 각각의 기여도도 비교적 명확하다.

- baseline 없음: BraTS-PED Dice `72.52`
- CMR 추가: `75.16`
- CMR과 MD 추가: `75.87`
- CMR, MD, MCL 모두 추가: `76.80`

BraTS2018 분류에서는 AUC가 `0.7719 -> 0.8056 -> 0.9032 -> 0.9452`로 상승했고, ADNI에서도 ACC가 `0.5546 -> 0.6261 -> 0.6261 -> 0.6765`로 개선됐다.

즉, 단일 구성요소보다 세 모듈의 결합이 중요하며, 특히 MCL이 최종 일반화 향상에 기여한다는 점이 드러난다.

## 강점

## 1. 문제 설정이 실제 임상 데이터 구조와 잘 맞는다

mpMRI는 본질적으로 multi-modal이고 결측 모달리티가 흔하다. BrainMVP는 이 현실을 pre-training 단계부터 직접 반영한다.

## 2. 다운스트림 범위가 넓다

10개 벤치마크에 걸친 segmentation과 classification 결과는 논문의 일반화 주장을 어느 정도 설득력 있게 만든다.

## 3. modality-wise distillation 아이디어가 참신하다

단순 SSL objective를 넘어서, 사전학습과 downstream adaptation을 연결하는 template 개념을 도입한 점이 인상적이다.

## 4. label efficiency가 실용적이다

의료영상에서 라벨 비용이 큰 만큼, 적은 라벨로 강한 성능을 내는 점은 실무적 가치가 높다.

## 한계와 비판적 검토

## 1. 범위가 실제로는 뇌 MRI 중심이다

제목은 `Medical Image Analysis` 전반을 암시하지만, 실제 방법과 실험은 거의 전적으로 brain mpMRI에 맞춰져 있다. 다른 장기나 다른 모달리티로의 일반화는 아직 입증되지 않았다.

## 2. 멀티모달이라고 해도 영상-영상 범위에 한정된다

최근 의료 foundation model 흐름은 image-text, report-image, multimodal reasoning까지 확장되는데, 이 논문은 image-image modality fusion에 집중한다. 따라서 broader multimodal foundation model과는 지향점이 다르다.

## 3. modality template의 해석 가능성은 아직 제한적이다

저자들은 distilled template가 모달 구조를 응축한다고 설명하지만, 이것이 어떤 임상 의미를 보존하는지에 대한 분석은 더 필요하다.

## 4. backbone 선택의 영향이 완전히 분리되지는 않는다

UniFormer 자체의 멀티모달 융합 능력이 강하기 때문에, 성능 향상 중 얼마나 많은 부분이 BrainMVP objective에서 오는지 구조적으로 더 엄밀한 분리가 가능했을 것이다.

## 5. 계산 비용이 적지 않다

240만 장 이상, 8개 모달리티, 8장의 RTX 4090을 활용한 pre-training은 현실적으로 가벼운 설정이 아니다. 재현성과 접근성 측면에서는 여전히 진입장벽이 있다.

## 실무적 및 연구적 인사이트

이 논문은 의료영상 사전학습에서 "모달리티를 따로따로 학습한 뒤 나중에 합치는 방식"보다, 처음부터 모달 간 관계를 pretext task에 녹여 넣는 편이 더 유리할 수 있음을 보여준다. 특히 mpMRI처럼 같은 해부학을 서로 다른 신호 공간에서 본다는 특성이 있는 경우, cross-modal reconstruction은 매우 자연스러운 설계다.

후속 연구 방향도 분명하다.

- 뇌 MRI를 넘어 CT-PET, multi-phase CT, MRI-ultrasound 등 다른 멀티모달 조합으로 확장
- 3D volume 전체 수준의 pre-training으로 확장
- image-text/report modality와 결합한 broader multimodal foundation model로 연결
- missing modality handling과 test-time adaptation의 결합

## 종합 평가

`Multi-modal Vision Pre-training for Medical Image Analysis`는 실제로는 `BrainMVP`라는 이름의 뇌 mpMRI 특화 multi-modal pre-training 프레임워크를 제안한 논문이다. 핵심은 cross-modal reconstruction, modality-wise data distillation, modality-aware contrastive learning을 결합해 모달 간 상관관계와 결측 모달 현실을 함께 다루는 데 있다.

제목에 비해 적용 범위는 아직 좁지만, 멀티모달 의료영상 사전학습을 image-text 정렬 바깥의 문제로 본격화했다는 점, 그리고 10개 다운스트림 과제에서 강한 일반화와 label efficiency를 보였다는 점에서 학술적 가치가 높다. 특히 brain mpMRI foundation pre-training의 유력한 출발점으로 읽을 만한 논문이다.
