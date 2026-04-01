# Chain-of-Thought Prompting for Speech Translation

Ke Hu, Zhehuai Chen, Chao-Han Huck Yang, Piotr Zelasko, Oleksii Hrinchuk, Vitaly Lavrukhin, Jagadeesh Balam, Boris Ginsburg

## 🧩 Problem to Solve

대규모 언어 모델(LLM)의 성공을 기반으로 음성 임베딩을 프롬프트로 활용하는 Speech-LLM 모델이 자동 음성 인식(ASR) 및 자동 음성 번역(AST)에서 우수한 성능을 보여주고 있습니다. 그러나 인코더-디코더 구조의 LLM(예: T5)을 사용하는 Speech-LLM에서 ASR 전사(transcript)를 AST를 위한 프롬프트로 어떻게 효과적으로 활용할지는 명확하지 않습니다. 특히, ASR 가설을 LLM의 디코더 출력으로 예측하는 기존 방식과 달리, ASR 가설을 LLM의 입력으로 주입하여 번역 성능을 개선할 수 있는 새로운 접근 방식이 필요합니다.

## ✨ Key Contributions

- 인코더-디코더 Speech-LLM(Megatron-T5 기반)에서 ASR 전사를 프롬프트로 활용하는 새로운 CoT(Chain-of-Thought) 프롬프팅 방식을 제안했습니다.
- 입력 음성에서 ASR 전사를 먼저 디코딩한 다음, 이 전사를 인코딩된 음성과 함께 프롬프트로 사용하여 음성 번역을 두 단계 프로세스로 안내하는 방식이 AST 성능을 크게 향상시킴을 입증했습니다.
- 음성 프롬프팅만 사용하는 기존 방식 대비, 6개의 En→X 또는 X→En AST 태스크에서 평균 $2.4$ BLEU 포인트의 성능 향상을 달성했습니다.
- ASR 및 AST 전사를 연결하여 예측하는 기존 CoT 예측 방식([28]과 유사한 구현)에 비해 평균 $2$ BLEU 포인트 더 높은 성능을 보이며, ASR 가설을 T5 입력으로 주입하는 방식의 효과를 입증했습니다.
- Megatron-T5 LLM의 모델 적응을 위해 LoRA(Low-rank adaptation)를 사용하여 전체 모델 미세 조정보다 우수한 성능(평균 $0.9$ BLEU 포인트 추가 향상)을 보였습니다.
- 관련 코드를 GitHub에 공개했습니다.

## 📎 Related Works

- **LLMs 및 Speech-LLMs**: LLM의 급속한 발전([1]-[6])과 음성 임베딩을 프롬프트로 활용하여 ASR 및 AST에 적용한 Speech-LLM 연구([7]-[15])를 언급합니다.
- **프롬프트 설계**: 인컨텍스트 학습([16], [17]), 학습 가능한 임베딩([18], [19]) 등 프롬프트 설계의 중요성을 강조합니다.
- **CoT 프롬프팅**: Wei et al.([20])의 연구와 같이 다단계 추론을 통해 LLM의 성능을 향상시키는 CoT 프롬프팅 기법의 효과를 참고했습니다.
- **다단계 예측/숙고 모델(Deliberation Models)**: 첫 번째 가설을 예측한 후 이를 더 정교한 두 번째 태스크에 활용하는 과거 음성 태스크 연구([21]-[24])와 유사성을 지적합니다.
- **Speech Translation with LLMs**: 황 등([28])은 ASR과 AST 전사 시퀀스를 연결하여 예측하는 CoT 예측 방식을 제안했는데, 본 연구는 인코더-디코더 LLM에 ASR 가설을 입력으로 주입하는 방식을 탐구하며 차별점을 둡니다. 동시에 진행된 유사 연구([29], [30])도 언급합니다.
- **오디오 및 음성 이해**: Gong et al.([25])은 Whisper ASR 출력을 LLaMA LLM에 프롬프트로 사용하여 오디오 태스크에 활용했습니다.

## 🛠️ Methodology

본 연구에서 제안하는 Speech-LLM 모델은 오디오 인코더와 Megatron-T5 LLM으로 구성됩니다.

1. **ASR 전사 생성**: 입력 음성은 Canary-1B([33]) 사전 학습된 오디오 인코더를 통해 인코딩된 후, ASR 시스템을 사용하여 먼저 소스 언어의 텍스트인 ASR 전사를 생성합니다. (공정한 비교를 위해 [28]과 유사한 CoT 예측 모델을 사용하여 ASR 가설을 추출).
2. **CoT 프롬프팅**: 생성된 ASR 전사는 고정된 AST 텍스트 프롬프트 및 음성 임베딩과 함께 단일 시퀀스로 연결되어 Megatron-T5 LLM의 입력으로 사용됩니다.
   - **프롬프트 예시**: "Q: Transcribe the spoken content to written {source lang} text, then translate this to {target lang} text, with punctuations and capitalizations. The source text is:{ASR transcript}\nA:"
3. **모델 학습**:
   - ASR 전사를 T5 프롬프트에 주입하는 방식으로 CoT 프롬프팅 모델을 학습합니다.
   - 다음 토큰 예측 손실(next token prediction loss)을 사용하여 훈련합니다.
   - 음성 인코더는 항상 튜닝하며, Megatron-T5 LLM에는 LoRA([36])를 적용하여 효율적인 모델 적응을 수행합니다. LoRA 어댑터는 인코더와 디코더의 셀프 어텐션 및 교차 어텐션 레이어에 추가됩니다.
4. **추론**: 학습과 동일하게 ASR 가설을 먼저 생성한 후, 이를 AST 프롬프트에 주입하여 번역을 수행합니다.
5. **데이터**: Canary AST 훈련 데이터([33])의 일부를 사용하고, AST 타겟 레이블은 NVIDIA Megatron NMT 모델([38], [39])로 합성 생성합니다. 추론은 FLEURS([40]) 테스트 셋으로 평가합니다.

## 📊 Results

- **CoT 프롬프팅 효과**: ASR 가설을 LLM 프롬프트에 주입하는 CoT 프롬프팅 모델(E1)은 모든 언어 쌍에서 기존 SALM-T5 기준 모델(B1) 대비 평균 $1.5$ BLEU 포인트($31.1 \rightarrow 32.6$) 향상된 성능을 보였습니다.
- **ASR 가설 품질의 영향**: 추정된 ASR 가설 대신 실제 ASR 정답(ground truth)을 프롬프트로 사용했을 때, 평균 $2.7$ BLEU 포인트($32.6 \rightarrow 35.3$)의 추가적인 성능 향상을 달성하여, 더 나은 ASR 품질이 번역에 긍정적인 영향을 미침을 확인했습니다.
- **CoT 예측 방식과의 비교**: ASR 가설을 먼저 예측한 후 AST 출력을 이어붙이는 CoT 예측 방식(B2)은 기존 모델 대비 $0.4$ BLEU 포인트 향상에 그쳤습니다. 반면, 본 연구의 CoT 프롬프팅 방식(E1)은 CoT 예측 방식(B2) 대비 평균 $0.9$ BLEU 포인트($31.5 \rightarrow 32.6$) 더 우수한 성능을 보여, ASR 가설을 T5 입력으로 주입하는 것이 더 효과적임을 입증했습니다.
- **LoRA 성능**: LoRA를 CoT 프롬프팅 모델(E1)에 추가(E2)한 결과, 평균 $0.9$ BLEU 포인트($32.6 \rightarrow 33.5$)의 추가 성능 향상을 가져왔습니다. 이는 LoRA가 적은 파라미터 증가로 모델 성능을 크게 개선함을 보여줍니다.
- **최종 비교**: 최상위 모델(E2: CoT Prompting + LoRA)은 SeamlessM4T-medium($1.2$B)보다 $4.7$ BLEU, SeamlessM4T-large-v2($2.3$B)보다 $1.3$ BLEU 더 높은 성능을 기록했습니다. 또한, 기존 ASR+NMT 캐스케이드 시스템과 비교했을 때, X→En 언어 쌍에서는 유사한 성능을 보였고, En→X 번역에서는 $0.5$~$2.5$ BLEU 포인트 더 우수한 성능을 보였습니다.

## 🧠 Insights & Discussion

- **CoT 프롬프팅의 효과**: ASR 전사를 LLM 프롬프트에 주입하는 CoT 접근 방식은 Speech-LLM의 음성 번역 성능을 크게 향상시킵니다. 이는 LLM이 음성을 텍스트로 인식한 다음, 이 텍스트 정보를 활용하여 번역을 수행하는 다단계 추론 과정을 효과적으로 안내하기 때문입니다.
- **최적의 ASR 주입 지점**: 인코더-디코더 LLM(T5)에서는 ASR 가설을 디코더 출력으로 예측하는 것보다 인코더 입력으로 주입하는 것이 더 효과적임이 입증되었습니다. 이는 모델이 번역을 시작하기 전에 소스 텍스트 정보에 명시적으로 접근하도록 하여 더 강력한 컨텍스트를 제공하기 때문입니다.
- **LoRA의 가치**: LoRA는 대규모 T5 LLM을 효율적으로 미세 조정하면서도 상당한 성능 향상을 제공하는 강력한 도구입니다. 이는 LLM의 방대한 파라미터를 모두 튜닝하지 않고도 음성 번역 태스크에 성공적으로 적응시킬 수 있음을 시사합니다.
- **개선 잠재력**: 실제 ASR 가설 대신 정답 ASR 전사를 사용했을 때 추가적인 BLEU 점수 향상이 있었던 것은, 향후 더 고품질의 ASR 시스템을 도입하면 AST 성능을 더욱 개선할 수 있음을 보여줍니다.
- **기존 시스템 대비 우위**: 제안된 방법은 단순 Speech-LLM 기준 모델, T5 기반 CoT 예측 모델, 그리고 SeamlessM4T와 같은 최첨단 다국어/다중 모드 모델 및 기존 캐스케이드 시스템에 비해 우수한 성능을 입증하며, Speech-LLM 기반 AST의 새로운 가능성을 제시합니다.

## 📌 TL;DR

음성 번역(AST) 성능을 개선하기 위해, 본 연구는 인코더-디코더 구조의 Speech-LLM(Megatron-T5)에 Chain-of-Thought(CoT) 프롬프팅 방식을 제안합니다. 이 방식은 입력 음성에서 ASR 전사를 먼저 생성한 후, 이 전사를 음성 임베딩 및 텍스트 프롬프트와 결합하여 T5 LLM의 입력으로 사용합니다. 여기에 LoRA(Low-rank adaptation)를 적용하여 효율적인 모델 미세 조정을 수행했습니다. 결과적으로, 이 방법은 기존 기준 모델 대비 평균 $2.4$ BLEU 포인트의 AST 성능 향상을 달성했으며, 기존 CoT 예측 방식보다 ASR 전사를 T5 입력으로 활용하는 것이 더 효과적임을 입증했습니다.
