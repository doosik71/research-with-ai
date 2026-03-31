# Position: Levels of AGI for Operationalizing Progress on the Path to AGI

Meredith Ringel Morris, Jascha Sohl-Dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement Farabet, Shane Legg

## 🧩 Problem to Solve

본 논문은 인공 일반 지능(AGI) 시스템의 능력, 행동 및 진화를 분류하기 위한 명확하고 조작 가능한 프레임워크의 부재 문제를 다룹니다. 기존 AGI 정의의 모호성으로 인해 모델 간 비교, 위험 평가, 정책 수립, 그리고 AGI로 가는 경로를 따라 진행 상황을 측정하고 소통하는 데 어려움이 있습니다.

## ✨ Key Contributions

- **AGI 정의를 위한 6가지 원칙 제시:** 기존 AGI 정의 분석을 통해 명확하고 실행 가능한 AGI 정의를 위한 핵심 기준(예: 능력 중심, 일반성과 성능 동시 고려, 초인지 능력 포함)을 수립했습니다.
- **AGI 수준 분류 체계 제안:** 성능(깊이)과 일반성(폭)을 핵심 차원으로 하는 2차원 매트릭스 기반의 AGI 수준 분류 체계를 도입하여, AGI로 가는 경로의 진행 상황을 미묘하게 측정하고 소통할 수 있도록 했습니다.
- **자율성 수준 프레임워크 도입:** AI 시스템의 배치 및 위험과 관련된 자율성 수준을 6단계로 분류하는 프레임워크를 제안하여, AGI 능력 발전과 인간-AI 상호작용 패러다임 간의 관계를 명확히 했습니다.
- **AGI 벤치마크 개발을 위한 요구 사항 논의:** 생태학적 타당성(ecological validity)과 "살아있는(living)" 벤치마크의 필요성을 포함하여, 미래 AGI 벤치마크가 갖춰야 할 도전적인 요건들을 설명했습니다.
- **AGI 수준과 위험 평가 및 인간-AI 상호작용의 연관성:** AGI 수준이 높아짐에 따라 발생하는 새로운 위험 유형과 적절한 인간-AI 상호작용 패러다임 선택의 중요성을 강조했습니다.

## 📎 Related Works

본 논문은 AGI 또는 AGI 관련 개념에 대한 9가지 주요 정의를 사례 연구로 분석하며 기존 작업을 참조합니다. 여기에는 다음이 포함됩니다:

- **튜링 테스트 (Turing Test):** AGI 유사 개념을 조작화하려는 초기 시도.
- **강한 AI - 의식을 가진 시스템:** 존 설(John Searle)의 '중국어 방' 논증을 포함한 의식과 같은 속성을 가진 AI.
- **인간 두뇌와의 유추:** 마크 구브루드(Mark Gubrud)의 초기 AGI 정의.
- **인지 작업에서의 인간 수준 성능:** 셰인 레그(Shane Legg)와 벤 고어첼(Ben Goertzel)의 정의.
- **학습 능력:** 머레이 섀너핸(Murray Shanahan)이 AGI 요구 사항으로 제시한 메타인지 능력.
- **경제적 가치 있는 작업:** OpenAI의 AGI 정의.
- **유연하고 일반적인 지능 ('커피 테스트'):** 게리 마커스(Gary Marcus)의 정의 및 관련 벤치마크 제안.
- **인공 유능 지능 (Artificial Capable Intelligence, ACI):** 무스타파 술레이만(Mustafa Suleyman)의 개념.
- **최첨단 LLM을 제너럴리스트로 간주:** 아구에라 이 아르카스(Agüera y Arcas)와 노르빅(Norvig)이 현재 LLM을 AGI로 간주한 주장.

## 🛠️ Methodology

본 논문은 AGI의 진행 상황을 측정하고 위험을 평가하며 모델을 비교할 수 있는 표준화된 프레임워크를 개발하기 위해 다음 단계별 접근 방식을 사용합니다.

1. **기존 AGI 정의 분석:** 9가지 저명한 AGI 정의를 검토하여 각 정의의 강점과 한계를 파악하고, 명확하고 실행 가능한 AGI 정의를 위한 공통 속성 및 기준을 도출합니다.
2. **AGI 정의를 위한 6가지 원칙 수립:** 이 분석을 바탕으로 AGI 정의가 충족해야 할 다음 6가지 핵심 원칙을 제시합니다:
   - **과정이 아닌 능력에 집중 (Focus on Capabilities, not Processes):** AI가 무엇을 할 수 있는지에 초점.
   - **일반성($\text{Generality}$)과 성능($\text{Performance}$)에 집중:** 두 가지 차원을 모두 고려.
   - **물리적 작업이 아닌 인지적 및 초인지적 작업에 집중:** 로봇 의체는 필수가 아님.
   - **배치가 아닌 잠재력에 집중 (Focus on Potential, not Deployment):** 실제 배치는 AGI 정의의 조건이 아님.
   - **생태학적 타당성에 집중 (Focus on Ecological Validity):** 실제 세계에서 가치 있는 작업을 벤치마크로 선정.
   - **단일 종점이 아닌 AGI로 가는 경로에 집중:** AGI의 점진적 수준을 정의.
3. **AGI 수준 분류 체계 개발:** 원칙 2와 6에 따라, AGI의 능력을 깊이(성능)와 폭(일반성)으로 구분하는 2차원 매트릭스 기반의 분류 체계를 제시합니다.
   - **성능 차원 (깊이):**
     - 레벨 0: No AI (비 AI)
     - 레벨 1: Emerging (초보 인간과 비슷하거나 약간 우월)
     - 레벨 2: Competent (숙련된 성인 인간의 최소 50백분위수)
     - 레벨 3: Expert (숙련된 성인 인간의 최소 90백분위수)
     - 레벨 4: Virtuoso (숙련된 성인 인간의 최소 99백분위수)
     - 레벨 5: Superhuman (모든 인간을 능가)
   - **일반성 차원 (폭):**
     - Narrow (명확하게 범위가 지정된 작업 또는 작업 세트)
     - General (새로운 기술 학습과 같은 초인지적 작업을 포함한 광범위한 비물리적 작업)
4. **자율성 수준 프레임워크 정의:** AGI 수준과 상호작용하는 6가지 인간-AI 자율성 수준 (AI가 도구, 컨설턴트, 협력자, 전문가, 에이전트 역할)을 제안하고, 각 수준에 따른 위험과 AGI 능력의 "해금" 수준을 설명합니다.
5. **AGI 벤치마크 및 위험 평가 논의:** 프레임워크의 실행 가능성을 높이기 위해, AGI 벤치마크가 포함해야 할 인지적 및 초인지적 작업의 다양성과 생태학적 타당성을 강조하며, AGI 수준에 따른 위험 프로필 변화를 분석합니다.

## 📊 Results

본 논문은 다음 두 가지 주요 분류표를 통해 AGI 및 자율성 수준에 대한 프레임워크를 제시합니다:

1. **AGI 수준 (성능 x 일반성 매트릭스):**

   - **레벨 0: No AI** (예: 계산기 소프트웨어, Amazon Mechanical Turk)
   - **레벨 1: Emerging** (예: Emerging Narrow AI - SHRDLU; **Emerging AGI - ChatGPT, Bard, Llama2, Gemini**). 현재 최전선 LLM은 대부분의 작업에서 'Emerging' 성능 수준에 있지만, 일부 작업(단편 에세이 작성, 간단한 코딩)에서는 'Competent' 수준을 보입니다.
   - **레벨 2: Competent** (예: Competent Narrow AI - Jigsaw, Siri, Watson; **Competent AGI - 아직 달성되지 않음**). 이 수준은 많은 기존 AGI 정의와 일치합니다.
   - **레벨 3: Expert** (예: Expert Narrow AI - Grammarly, Imagen, Dall-E 2; **Expert AGI - 아직 달성되지 않음**)
   - **레벨 4: Virtuoso** (예: Virtuoso Narrow AI - Deep Blue, AlphaGo; **Virtuoso AGI - 아직 달성되지 않음**)
   - **레벨 5: Superhuman** (예: Superhuman Narrow AI - AlphaFold, AlphaZero, StockFish; **Artificial Superintelligence (ASI) - 아직 달성되지 않음**)
   - 이 매트릭스는 현재 AI 시스템이 AGI 경로의 어느 지점에 있는지를 개괄적으로 보여주며, 대부분의 현재 최전선 AI 모델은 'Emerging AGI'에 해당합니다.

2. **자율성 수준 (인간-AI 상호작용 패러다임):**
   - **레벨 0: No AI** (인간이 모든 것을 수행)
   - **레벨 1: AI as a Tool** (인간이 전적으로 제어하며 AI는 하위 작업을 자동화, 예: 검색 엔진, 문법 검사기) - Emerging Narrow AI에서 가능, Competent Narrow AI에서 가능성 높음.
   - **레벨 2: AI as a Consultant** (AI가 인간의 호출 시 실질적인 역할 수행, 예: 문서 요약, 코드 생성) - Competent Narrow AI에서 가능, Expert Narrow AI/Emerging AGI에서 가능성 높음.
   - **레벨 3: AI as a Collaborator** (동등한 인간-AI 협업, 목표 및 작업의 상호작용적 조정, 예: 체스 AI와의 훈련) - Emerging AGI에서 가능, Expert Narrow AI/Competent AGI에서 가능성 높음.
   - **레벨 4: AI as an Expert** (AI가 상호작용을 주도하고 인간은 지침 및 피드백 제공, 예: 단백질 접힘 예측을 통한 과학적 발견) - Virtuoso Narrow AI에서 가능, Expert AGI에서 가능성 높음.
   - **레벨 5: AI as an Agent** (완전 자율 AI, 예: 자율 AI 개인 비서) - 아직 해금되지 않음, Virtuoso AGI/ASI에서 가능성 높음.
   - 이 표는 AGI 능력이 증가함에 따라 새로운 상호작용 패러다임이 "해금"되지만, 적절한 자율성 수준은 특정 작업 및 맥락에 따라 신중하게 선택되어야 함을 보여줍니다.

## 🧠 Insights & Discussion

- **미묘한 AGI 진행 상황 측정:** 제안된 프레임워크는 AGI를 이분법적 개념이 아닌, 성능과 일반성의 점진적인 발전으로 이해할 수 있게 합니다. 이는 "스파크 AGI"와 같은 현재의 논의를 맥락화하는 데 유용합니다.
- **위험 평가의 정교화:** AGI 수준과 자율성 수준을 결합하여 고려함으로써, AI 시스템과 관련된 오용, 정렬, 구조적 위험 등 다양한 유형의 위험을 보다 미묘하게 평가할 수 있습니다. 예를 들어, 'Expert AGI'는 경제적 혼란과 같은 구조적 위험을 수반할 수 있습니다.
- **역할 분리의 중요성:** 시스템의 **능력(Capabilities)**과 **배치 결정(Deployment decisions, 자율성)**을 분리하여 고려하는 것은 중요합니다. 고도로 유능한 AI도 특정 상황에서는 낮은 자율성으로 배포될 수 있습니다.
- **벤치마킹의 도전:** AGI 벤치마크는 인지적, 초인지적, 창의성 등 다양한 능력을 포괄해야 하며, 생태학적 타당성을 갖춘 '살아있는 벤치마크'여야 한다는 점을 강조합니다. 이는 단순한 자동화 가능한 지표를 넘어 실제 세계의 가치를 반영해야 합니다.
- **초인지 능력의 중요성:** 새로운 기술을 학습하고, 도움을 요청해야 할 때를 알고, 인간을 정확하게 모델링하는 등의 초인지 능력이 AGI의 일반성을 달성하는 데 필수적이라고 역설합니다.
- **인간-AI 상호작용의 핵심 역할:** 모델 개선과 더불어 인간-AI 상호작용 연구에 투자하는 것이 책임감 있고 안전한 AI 시스템 배치를 위해 필수적임을 강조합니다. 상호작용 디자인이 모델의 배포된 성능에 큰 영향을 미칠 수 있습니다.
- **한계 및 미래 방향:** 이 프레임워크는 '위치 논문(position paper)'으로서 벤치마크의 세부 사항을 제시하기보다는 AGI 분류 체계를 위한 온톨로지를 제안합니다. 향후 연구는 구체적인 벤치마크 개발, 위험 프로필의 상세 분석, 해석 가능성 발전 등을 포함할 수 있습니다.

## 📌 TL;DR

AGI 정의의 모호성을 해결하고 진행 상황을 체계적으로 추적하기 위해, 본 논문은 능력과 일반성을 중심으로 한 **AGI 5단계 프레임워크**를 제시합니다. 이는 AGI를 **Emerging, Competent, Expert, Virtuoso, Superhuman**의 성능 수준과 **Narrow, General**의 일반성 수준으로 분류하며, 현재 최첨단 LLM은 'Emerging AGI' 수준에 해당한다고 평가합니다. 또한, AGI 능력 발전에 따라 달라지는 **6단계의 AI 자율성 수준**을 정의하고, 각 수준에서 발생할 수 있는 위험을 논의하며, 안전하고 책임감 있는 AGI 개발을 위해 **인간-AI 상호작용 디자인**과 **생태학적으로 타당한 벤치마크**의 중요성을 강조합니다.
