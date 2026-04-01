# SAM 3: Segment Anything with Concepts

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, Christoph Feichtenhofer

## 🧩 Problem to Solve

이전 SAM 시리즈(SAM 1, 2)는 주로 시각적 프롬프트(점, 상자, 마스크)를 사용하여 *단일 객체*에 대한 Promptable Visual Segmentation (PVS)에 중점을 두었습니다. 그러나 이는 이미지나 비디오 내에서 텍스트 또는 이미지 예시 프롬프트로 지정된 *특정 개념의 모든 인스턴스를 찾아 분할*하는 일반적인 작업을 다루지 못했습니다. 즉, "모든 고양이"를 찾는 것과 같이 개방형 개념에 대한 인스턴스 분할 및 비디오 프레임 전반에 걸친 객체 ID 추적을 수행하는 데 한계가 있었습니다. 본 논문은 이러한 격차를 해소하고 "Promptable Concept Segmentation (PCS)" 작업을 발전시키는 것을 목표로 합니다.

## ✨ Key Contributions

* **Promptable Concept Segmentation (PCS) 작업 정의:** 텍스트 구문(예: "노란색 스쿨버스") 또는 이미지 예시(또는 둘의 조합) 프롬프트를 기반으로 이미지 및 비디오에서 모든 일치하는 객체 인스턴스를 감지, 분할 및 추적하는 새로운 작업을 공식화했습니다.
* **SAM 3 모델 제시:** PCS 작업을 처리하도록 설계된 통합 모델로, 기존 SAM 2의 시각적 분할(PVS) 기능도 향상시켰습니다.
* **확장 가능한 데이터 엔진 개발:** 인간 및 모델-인-더-루프 방식을 통해 400만 개의 고유 개념 레이블과 5,200만 개의 마스크, 3,800만 개의 구문 및 14억 개의 합성 마스크를 포함하는 고품질의 대규모 훈련 데이터셋을 구축했습니다. 특히, "AI 검증자(verifier)"를 활용하여 데이터 처리량을 두 배로 늘렸습니다.
* **새로운 모델 아키텍처:** 단일 백본을 공유하는 이미지 레벨 검출기와 메모리 기반 비디오 추적기로 구성되며, 인식과 지역화를 분리하는 Presence Head를 도입하여 검출 정확도를 높였습니다.
* **최첨단 성능 달성:** 이미지 및 비디오 PCS 작업에서 기존 시스템 대비 정확도를 두 배로 높였으며, 시각적 분할 작업에서 SAM 2의 성능을 개선했습니다.
* **벤치마크 및 코드 공개:** PCS를 위한 새로운 Segment Anything with Concepts (SA-Co) 벤치마크(20만 7천 개 이상의 고유 개념)와 SAM 3 체크포인트, 추론 코드를 오픈 소스로 공개했습니다.

## 📎 Related Works

* **Promptable 및 Interactive Visual Segmentation:** SAM (Kirillov et al., 2023) 및 SAM 2 (Ravi et al., 2024)는 시각적 프롬프트를 사용한 단일 객체 분할에 중점을 두었으나, 텍스트 프롬프트와 다중 객체 인스턴스 분할은 충분히 다루지 않았습니다.
* **Open-Vocabulary Detection and Segmentation in Images:** OWLv2 (Minderer et al., 2024), GLIP (Li et al., 2022b), GroundingDino (Liu et al., 2023), DINOv (Li et al., 2023a), T-Rex2 (Jiang et al., 2024)와 같은 이전 연구들은 이미지에서 개방형 어휘 객체 감지 및 분할에 기여했습니다. SAM 3는 100배 이상 많은 고유 개념을 가진 새로운 벤치마크를 제시합니다.
* **Visual Grounding:** MDETR (Kamath et al., 2021), GLEE (Wu et al., 2024a), LISA (Lai et al., 2024) 등은 언어 표현을 이미지 영역에 지역화하는 데 초점을 맞췄습니다. SAM 3는 MLLM과 결합하여 복잡한 언어 프롬프트를 처리할 수 있는 "비전 도구"로 활용될 수 있습니다.
* **Multi-Object Tracking and Segmentation:** SORT (Bewley et al., 2016), ByteTrack (Zhang et al., 2022c), TrackFormer (Meinhardt et al., 2022) 등 추적-감지(tracking-by-detection) 및 종단 간(end-to-end) 추적 모델들이 있었으나, SAM 3는 강력한 이미지 검출기를 추적기에 통합하여 비디오 내 개념 분할을 수행합니다.

## 🛠️ Methodology

SAM 3는 SAM 2의 일반화된 형태로, Promptable Concept Segmentation (PCS) 및 Promptable Visual Segmentation (PVS) 작업을 모두 지원합니다.

* **모델 아키텍처:**
  * **Perception Encoder (PE) 백본:** 이미지와 텍스트 입력을 처리하는 단일 공유 백본으로, 강력한 시각-언어 정렬을 제공합니다.
  * **검출기 (Detector):** DETR (Carion et al., 2020) 패러다임을 따르는 이미지 레벨 모델.
    * **Presence Head:** 객체 인식을 지역화와 분리하기 위해 학습된 전역 Presence Token을 도입합니다. 이는 `p(query_i matches NP) = p(query_i matches NP | NP appears in image) * p(NP appears in image)`와 같이 객체의 전체 이미지 내 존재 여부 `p(NP appears in image)`를 예측하여 검출 정확도를 높입니다.
    * **Ambiguity Head:** 모호한 구문에 대한 여러 유효한 해석을 처리하기 위해 Mixture of Experts (K=2) 방식을 사용하여 중복되거나 상충되는 마스크 생성을 줄입니다.
    * 텍스트, 기하학적(점, 상자) 및 이미지 예시 프롬프트를 조건으로 받습니다.
  * **추적기 (Tracker):** SAM 2의 트랜스포머 인코더-디코더 아키텍처를 계승하며, 이전 프레임의 마스크릿(masklet)을 현재 프레임의 새로운 위치로 전파하고 새로운 객체를 감지하여 매칭합니다.
    * **시간적 모호성 해소 전략:** 트랙 확인 지연, 미확인/중복 마스크릿 제거, 마스크릿 억제, 주기적 재프롬프트, 감지 기반 재프롬프트 등을 통해 비디오 추적의 견고성을 높입니다.
  * **상호작용성:** 텍스트 프롬프트 외에 이미지 예시(긍정/부정) 및 시각적 프롬프트(클릭)를 반복적으로 추가하여 마스크를 정제할 수 있습니다.
* **데이터 엔진 (인간 및 AI-in-the-loop):**
  * **미디어 큐레이션:** 다양한 시각적 도메인(웹 스크랩, 예술, 음식, 운전 등)에서 도전적이고 희귀한 개념을 적극적으로 발굴합니다.
  * **레이블 큐레이션:** 멀티모달 LLM을 "AI 주석자"로 활용하여 명사 구문 및 어려운 부정(hard negative) 구문을 생성합니다.
  * **레이블 검증 (AI Verifiers):** LLM을 "AI 검증자"로 미세 조정하여 마스크 품질 검증(Mask Verification) 및 완결성 검증(Exhaustivity Verification)을 수행합니다. 이는 인간 수준의 정확도를 달성하며 데이터 처리량을 두 배로 증가시켜 인간 주석자는 어려운 오류 수정에 집중할 수 있게 합니다.
  * **수동 수정:** AI 검증자가 통과하지 못한 데이터는 인간 주석자가 수동으로 수정합니다.
  * 이 데이터 엔진을 통해 SA-Co/HQ (4M 고유 NP, 52M 마스크), SA-Co/SYN (38M NP, 1.4B 마스크), SA-Co/EXT, SA-Co/VIDEO (52.5K 비디오, 467K 마스크릿) 데이터셋을 구축했습니다.
* **훈련 단계:** PE 사전 훈련, 검출기 사전 훈련, 검출기 미세 조정, 추적기 훈련의 4단계로 점진적으로 기능을 추가합니다.
* **SA-Co 벤치마크 및 평가 지표:** 20만 7천 개 이상의 고유 개념을 가진 대규모 벤치마크를 도입했으며, `cgF_1 = 100 * pmF_1 * IL_MCC` (`pmF_1`은 지역화 정확도, `IL_MCC`는 이미지 레벨 분류 정확도) 및 `pHOTA` (비디오 추적 정확도)와 같은 새로운 지표를 사용하여 PCS 성능을 평가합니다. PCS의 본질적인 모호성을 처리하기 위해 여러 인간 주석 간의 오라클 평가를 사용합니다.

## 📊 Results

* **이미지 PCS (텍스트 프롬프트):**
  * LVIS에서 제로샷 마스크 AP 48.8을 달성하여 이전 최고 기록(38.5)을 크게 상회했습니다.
  * 새로운 SA-Co/Gold 벤치마크에서 기존 최고 성능인 OWLv2* 대비 `cgF_1` 점수를 두 배 이상 높였으며, 인간 성능의 74%에 도달했습니다.
  * 개방형 어휘 의미 분할에서 강력한 전문가 모델인 APE를 능가했습니다.
* **Few-Shot 적응:** ODinW13 및 RF-100VL에서 최첨단 10-샷 성능을 달성하여 Gemini 및 다른 객체 검출 전문가 모델들을 능가했습니다.
* **PCS (예시 프롬프트):** 단일 예시를 사용했을 때 COCO, LVIS, ODinW에서 T-Rex2 대비 크게 향상된 성능을 보였습니다. 반복적인 PCS는 PVS보다 훨씬 빠르게 `cgF_1`을 개선했습니다.
* **객체 카운팅:** CountBench에서 MAE 0.12, Accuracy 93.8%로 MLLM과 비교하여 뛰어난 카운팅 정확도를 달성하며 객체 분할까지 제공했습니다.
* **비디오 PCS (텍스트 프롬프트):** SA-Co/VEval 벤치마크 및 공개 벤치마크에서 GLEE, LLMDet 기반 시스템 등 모든 비교군을 크게 능가했습니다. 특히 명사 구문 수가 많은 벤치마크에서 강점을 보였으며, 인간 `pHOTA`의 80% 이상에 도달했습니다.
* **PVS (시각 프롬프트):** SAM 2에 비해 비디오 객체 분할(VOS)에서 MOSEv2 데이터셋에서 6.5점 향상되는 등 대부분의 벤치마크에서 상당한 개선을 보였습니다. 대화형 이미지 분할에서도 평균 `mIoU`가 SAM 2보다 높았습니다.
* **SAM 3 Agent (MLLM + SAM 3):** ReasonSeg 및 OmniLabel 데이터셋에서 복잡한 텍스트 쿼리에 대한 제로샷 분할에서 이전 최첨단 모델들을 능가했으며, RefCOCO+ 및 RefCOCOg에서도 뛰어난 제로샷 성능을 보였습니다.
* **어블레이션 연구:** Presence head, Hard Negative, SA-Co/SYN 및 SA-Co/HQ 데이터 사용이 성능 향상에 크게 기여함을 확인했습니다. AI 검증자가 SAM 3와 인간 성능 간의 격차를 절반가량 줄였습니다. 합성 데이터를 통한 도메인 적응 능력이 입증되었습니다.

## 🧠 Insights & Discussion

SAM 3는 개념 기반의 객체 감지, 분할 및 추적을 가능하게 함으로써 프롬프트 기반 분할 능력을 한 단계 끌어올렸으며, 이는 로봇 공학, 콘텐츠 제작, 증강 현실 등 멀티모달 AI의 다양한 응용 분야에 필수적인 기반 기능을 제공합니다. 특히, 인간 주석자와 AI 모델의 장점을 상호보완적으로 활용하는 확장 가능한 데이터 엔진은 이 모델의 성능 향상에 결정적인 역할을 했습니다. 인식, 지역화, 추적을 분리하는 아키텍처 설계와 Presence Head 같은 혁신적인 구성 요소는 모델의 정확도와 견고성을 향상시켰습니다. 또한, 상호작용적 개선 기능은 사용자가 모호한 결과에 대해 모델을 쉽게 정제할 수 있도록 지원합니다.

**한계점:**

* **개념 일반화의 어려움:** 제로샷(zero-shot) 설정에서 항공기 유형, 의료 용어 등 특정 전문 분야의 미세 분류(fine-grained) 개념에 대해서는 일반화 성능이 저조할 수 있습니다. 이는 합성 데이터를 활용한 자동 도메인 확장을 통해 완화될 수 있습니다.
* **복잡한 쿼리 제한:** 단순 명사 구문 프롬프트에 제한되며, 여러 속성 쿼리나 긴 참조 표현식(referring expressions)은 직접 지원하지 않습니다. 다만, MLLM과 결합하여 "SAM 3 Agent" 형태로 사용될 경우 더 복잡한 쿼리도 처리할 수 있습니다.
* **비디오 추적 비용:** 비디오 추론 비용은 추적되는 객체 수에 비례하여 증가하므로, 다수의 객체가 있는 경우 실시간 성능을 위해서는 여러 GPU가 필요합니다.
* **다중 객체 추적의 모호성:** 현재 아키텍처는 다중 객체 추적 시 객체 간의 공유된 문맥적 정보가 부족하여 모호성을 해결하는 데 어려움을 겪을 수 있습니다 (향후 공유 전역 메모리(global memory)를 통해 개선 가능).
* **상호작용성 모드 전환:** 개념 레벨 상호작용과 인스턴스 레벨 상호작용 간에 "하드 모드 전환"이 발생하며, 이는 향후 더 원활한 전환으로 개선될 수 있습니다.

## 📌 TL;DR

**문제:** 기존 SAM 모델은 시각 프롬프트로 *단일 객체*만 분할할 수 있었으나, 텍스트나 이미지 예시 프롬프트로 지정된 *개념의 모든 인스턴스*를 이미지 및 비디오에서 감지, 분할, 추적하는 일반적인 기능이 필요했습니다.

**방법:** SAM 3는 새로운 Promptable Concept Segmentation (PCS) 작업을 해결하기 위해 설계되었습니다. 이 모델은 공유 백본, 인식과 지역화를 분리하는 Presence Head를 갖춘 분리형 이미지 검출기, 그리고 메모리 기반 비디오 추적기로 구성된 통합 아키텍처를 사용합니다. 이 모델은 인간 주석자와 "AI 검증자"를 활용하여 다양하고 고품질의 레이블(하드 네거티브 포함)을 효율적으로 큐레이션하는 새로운 대규모 데이터 엔진(SA-Co)으로 훈련되었습니다.

**결과:** SAM 3는 이미지 및 비디오 PCS에서 기존 시스템 대비 정확도를 두 배로 높였으며, LVIS 및 SA-Co를 포함한 다양한 벤치마크에서 새로운 최첨단 성능을 달성했습니다. 또한 시각적 분할 작업에서 SAM 2보다 향상된 성능을 보였고, 강력한 Few-Shot 적응 능력을 입증했으며, MLLM과 결합 시 복잡한 쿼리도 처리할 수 있음을 보여주었습니다. Presence Head와 효율적인 데이터 엔진은 이러한 성능 향상의 핵심 요소로 작용했습니다.
