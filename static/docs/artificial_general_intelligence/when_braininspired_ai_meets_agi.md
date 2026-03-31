# When Brain-inspired AI Meets AGI

Lin Zhao, Lu Zhang, Zihao Wu, Yuzhong Chen, Haixing Dai, Xiaowei Yu, Zhengliang Liu, Tuo Zhang, Xintao Hu, Xi Jiang, Xiang Li, Dajiang Zhu, Dinggang Shen, and Tianming Liu

## 🧩 Problem to Solve

인간과 유사한 지적 능력을 갖춘 기계, 즉 범용 인공지능(AGI)을 개발하는 것은 인류의 오랜 목표입니다. 이 목표를 달성하기 위해, 인간 두뇌의 원리를 모방한 뇌 영감 인공지능(Brain-inspired AI) 분야가 발전하고 있으며, AGI 연구자들은 이러한 뇌의 특성을 AI 시스템에 통합하려 합니다. 본 논문은 뇌 영감 AI와 AGI의 현재 발전 상황, 핵심 특성, 관련 기술, 진화 과정 및 한계를 종합적으로 조망하며, AGI 개발의 방향성을 제시하는 것을 목표로 합니다.

## ✨ Key Contributions

- **뇌 영감 AI와 AGI의 관계 조망:** 뇌 영감 AI가 AGI 발전에 미친 영향을 포괄적으로 분석합니다.
- **인간 지능 및 AGI의 핵심 특성 제시:** 규모(scaling), 다중 모드(multimodality), 정렬(alignment), 추론(reasoning) 등 AGI 달성에 필요한 주요 특성들을 심층적으로 다룹니다.
- **AGI 달성을 위한 중요 기술 논의:** 인컨텍스트 학습(in-context learning) 및 프롬프트 튜닝(prompt tuning)과 같은 최신 AI 기술들을 소개하고 그 중요성을 강조합니다.
- **AGI 시스템의 진화 과정 분석:** 알고리즘적, 인프라적 관점에서 AGI 시스템의 발전사를 탐구합니다.
- **AGI의 한계와 미래 방향성 제시:** 현재 AGI 연구의 제약 사항을 명확히 하고, 향후 연구 및 개발이 나아가야 할 방향을 제시합니다.

## 📎 Related Works

- **초기 인공 신경망:** McCulloch-Pitts Neuron, Perceptron, Backpropagation.
- **신경망 아키텍처:** Convolutional Neural Networks (CNNs), Attention 메커니즘, Transformer, BERT, GPT, Vision Transformer (ViT).
- **뇌 구조/기능 영감 AI:** 스몰 월드(small-world) 네트워크 특성, 코어-주변부(core-periphery) 조직 원리, 스파이킹 신경망(Spiking Neural Networks, SNNs).
- **뉴로모픽 컴퓨팅:** IBM TrueNorth 칩, Intel Loihi 칩, Tianjic 칩.
- **대규모 언어 모델(LLMs) 및 다중 모드 AI:** GPT-2, GPT-3, T5, CLIP, DALL-E, GLIDE, VisualGPT, Diffusion 모델, METER, VLMo, ClipBERT, VIOLET, SwinBERT, Sparrow, InstructGPT, ChatGPT, GPT-4.
- **추론 방법론:** Zero-shot Chain of Thought (CoT), Few-Shot CoT, Self-consistency, Least-to-most prompting.
- **인프라:** GPUs, TPUs, Azure AI supercomputer.

## 🛠️ Methodology

본 논문은 뇌 영감 인공지능이 범용 인공지능(AGI)으로 발전하는 과정을 포괄적으로 검토하는 서베이 논문입니다. 구체적인 방법론은 다음과 같습니다.

1. **현황 개괄:** 뇌 영감 AI의 현재 발전 상황과 AGI와의 광범위한 연관성을 소개합니다.
2. **핵심 특성 분석:** 인간 지능과 AGI의 중요한 특성들(예: 규모, 다중 모드, 정렬, 추론)을 개별적으로 분석하고, 각 특성이 AI 시스템에서 어떻게 구현되고 있는지 설명합니다.
3. **핵심 기술 검토:** 현재 AI 시스템에서 AGI 달성을 위한 주요 기술들(예: 인컨텍스트 학습, 프롬프트 튜닝)을 상세히 다룹니다.
4. **진화 과정 탐구:** AGI 시스템의 알고리즘적 관점(신경망 발전)과 인프라적 관점(하드웨어 및 분산 컴퓨팅)에서의 진화 과정을 조사합니다.
5. **한계 및 미래 방향성 제시:** 현재 AGI 연구의 한계점을 명확히 하고, 향후 연구 방향 및 윤리적 고려 사항 등을 논의합니다.

## 📊 Results

본 논문은 설문조사 논문이므로 직접적인 실험 결과는 제시하지 않지만, 기존 연구들을 종합하여 다음과 같은 "발견(results)"을 제시합니다.

- **뇌 영감 AI의 AGI 기여:** McCulloch-Pitts 뉴런부터 CNN, 주의 메커니즘, SNN, 뉴로모픽 컴퓨팅에 이르기까지, 뇌 구조와 기능에 영감을 받은 AI는 AGI 개발에 중추적인 역할을 해왔습니다.
- **규모와 지능의 상관관계:** 인간 뇌의 뉴런 수와 인지 능력의 상관관계와 유사하게, LLM의 파라미터(예: GPT-2의 15억 개에서 GPT-3의 1750억 개) 증가는 복잡한 언어 처리 및 소수 학습(few-shot learning) 능력 향상으로 이어져 AGI 달성의 핵심 요소임을 시사합니다.
- **다중 모드 학습의 중요성:** GPT-4와 같은 모델에서 텍스트와 이미지 등 여러 모달리티 정보를 통합하는 능력은 교차 모달 작업뿐만 아니라 단일 모달 작업의 성능까지 향상시키며, 인간의 인지 방식과 유사하게 AGI의 필수 특성으로 부각됩니다.
- **인간 피드백을 통한 정렬:** RLHF (Reinforcement Learning from Human Feedback)를 활용한 InstructGPT, ChatGPT, GPT-4와 같은 모델들은 편향적이거나 유해한 출력을 줄이고 사용자 의도에 부합하는 안전하고 유용한 응답을 생성하는 능력을 크게 향상시켰습니다.
- **추론 능력의 발현과 촉진:** LLM은 특정 규모에 도달하면 추론 능력이 발현되며, Zero-shot CoT, Few-shot CoT, Least-to-most prompting과 같은 프롬프트 기반 방법론을 통해 이러한 추론 능력을 효과적으로 활용하고 크게 향상시킬 수 있음이 입증되었습니다.

## 🧠 Insights & Discussion

- **의미 및 시사점:** 뇌 영감 AI는 AGI 개발의 핵심 동력이며, 규모 확장, 다중 모드 통합, 인간 가치와의 정렬, 복합적 추론 능력 강화가 AGI의 주요 특성임을 강조합니다. GPT-4와 같은 최신 대규모 모델들은 이러한 특성들을 성공적으로 통합하여 AGI에 가까워지고 있음을 보여줍니다. AGI는 궁극적으로 인간 지능을 증진시키고 지능에 대한 우리의 이해를 심화시킬 잠재력을 가집니다.
- **한계:**
  - **인간 두뇌에 대한 제한적 이해:** 두뇌 작동 방식에 대한 불완전한 이해는 인간 지능을 완벽하게 모방하는 기계 제작에 걸림돌이 됩니다.
  - **데이터 효율성 부족:** 현재 AGI 및 뇌 영감 AI 시스템은 인간과 달리 엄청난 양의 훈련 데이터를 요구하며, 소수의 샘플로부터 효율적으로 학습하는 능력은 여전히 미해결 과제입니다.
  - **윤리 및 안전 문제:** AGI 시스템이 더욱 지능화됨에 따라 의도치 않은 해를 방지하고 인간 가치 및 윤리적 원칙에 부합하도록 의사결정을 내리도록 보장하는 것이 중요합니다.
  - **막대한 계산 비용:** 대규모 모델의 훈련 및 운영에 필요한 막대한 계산 자원과 에너지 소비는 환경적 지속가능성 문제를 야기하며, 연구 개발의 접근성을 제한합니다.
- **미래 방향:**
  - **강력한 AGI 파운데이션 모델 개발:** ChatGPT, GPT-4와 같은 모델 연구를 지속하여 더 진보된 AGI 모델을 구축합니다.
  - **다양한 AI 시스템 및 기술 통합:** 자연어 처리, 컴퓨터 비전, 로봇 공학 등을 인간 전문가 피드백 기반의 강화 학습과 결합하여 다재다능하고 적응성 있는 시스템을 만듭니다.
  - **새로운 기계 학습 접근 방식:** 인간 두뇌로부터 영감을 받아 더 효율적인 지시(instruct) 방법, 인컨텍스트 학습 알고리즘, 추론 패러다임을 개발합니다.
  - **윤리적 및 사회적 고려:** 편향, 프라이버시, 보안 등 AGI 개발의 윤리적, 사회적 함의를 신중하게 고려하여 사회 전체에 이익이 되도록 개발 및 활용합니다.

## 📌 TL;DR

본 논문은 AGI 달성을 위한 뇌 영감 AI의 중요성을 강조하며, 규모 확장, 다중 모드 학습, 인간 가치 정렬, 추론 능력이라는 AGI의 핵심 특성을 분석합니다. 인컨텍스트 학습 및 프롬프트 튜닝과 같은 최신 기술이 AGI 발전에 기여함을 설명하고, AGI의 진화 과정 및 현재의 한계(두뇌 이해 부족, 데이터 비효율성, 윤리/안전, 계산 비용)와 미래 방향을 제시합니다. 궁극적으로 GPT-4와 같은 대규모 모델이 AGI에 근접하고 있지만, 지속적인 연구와 다학제적 협력이 필요함을 역설합니다.
