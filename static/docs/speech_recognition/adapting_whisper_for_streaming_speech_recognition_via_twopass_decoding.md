# Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding

- **저자**: Haoran Zhou, Xingchen Song, Brendan Fahy, Qiaochu Song, Binbin Zhang, Zhendong Peng, Anshul Wadhawan, Denglin Jiang, Apurv Verma, Vinay Ramesh, Srivas Prasad, Michele M. Franceschini
- **발표연도**: 2025
- **arXiv**: https://arxiv.org/abs/2506.12154

## 1. 논문 개요

이 논문은 OpenAI의 Whisper를 실시간 스트리밍 음성 인식(streaming ASR) 모델로 바꾸는 방법을 제안한다. Whisper는 원래 매우 큰 규모의 음성 데이터로 학습되어 강건성과 정확도가 높은 모델이지만, 기본 구조가 encoder-decoder 기반의 sequence-to-sequence 모델이기 때문에 본질적으로 non-streaming 환경에 맞춰져 있다. 즉, 전체 오디오를 충분히 본 뒤 문장을 생성하는 데는 강하지만, 입력이 계속 들어오는 상황에서 짧은 지연으로 부분 결과(partial transcript)를 안정적으로 내는 데는 적합하지 않다.

논문이 다루는 핵심 연구 문제는 “Whisper의 강력한 사전학습 성능을 유지하면서도, 이를 진짜 streaming ASR처럼 동작하게 만들 수 있는가”이다. 기존의 pseudo-streaming 방식은 오디오를 조금씩 잘라 넣고 매번 Whisper를 다시 실행하는 식이어서, 학습 시 가정과 추론 시 환경이 어긋나고, 같은 오디오를 반복 처리하므로 비효율적이다. 또한 Whisper는 입력을 30초 길이에 맞춰 padding하는 전제가 있어 짧은 조각을 처리할 때도 계산 낭비가 발생한다.

이 문제는 실제 응용에서 중요하다. 금융 콜, 회의, 자막 생성, 실시간 음성 인터페이스 등에서는 전체 음성이 끝날 때까지 기다릴 수 없기 때문이다. 논문은 Whisper를 단순히 “잘라서 자주 돌리는 모델”이 아니라, partial transcript를 낼 수 있고, 종료 시점(endpoint)에서 더 정확한 최종 문장을 결정하는 구조로 바꾸려 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 Whisper를 WeNet의 U2(Unified Two-pass) 구조에 맞게 재구성하는 것이다. U2는 streaming과 non-streaming을 하나의 프레임워크 안에서 다루기 위한 구조로, encoder 위에 CTC decoder와 attention decoder를 함께 두고, 빠른 1차 decoding과 정확한 2차 rescoring을 결합한다.

이 논문에서는 Whisper encoder 위에 새로운 CTC decoder를 추가한다. 이 CTC branch는 causal attention mask를 사용한 encoder 출력 위에서 학습되어, 입력이 아직 끝나지 않은 상황에서도 현재까지의 음성에 대해 스트리밍 partial hypothesis를 생성한다. 반면 원래의 Whisper decoder는 유지되며, endpoint가 감지되면 CTC가 만든 상위 후보들을 다시 평가(rescoring)해서 최종 결과를 선택한다. 즉, “빠르게 부분 결과를 내는 경로”와 “더 정확하게 최종 문장을 고르는 경로”를 분리한 것이다.

또 하나의 중요한 아이디어는 hybrid tokenizer이다. Whisper는 5만 개가 넘는 큰 token space를 사용한다. 그런데 도메인 특화된 적은 양의 데이터로 CTC branch를 새로 학습하면 이 큰 vocabulary를 충분히 커버하기 어렵다. 그래서 저자들은 CTC decoder에는 Whisper tokenizer의 앞쪽 8,000개 토큰만 사용하고, attention decoder에는 기존 Whisper의 전체 token set을 유지한다. 이렇게 하면 CTC는 더 작은 토큰 공간에서 안정적으로 partial hypothesis를 만들고, attention decoder는 원래 Whisper의 표현력을 살려 rescoring을 수행한다. 논문은 이것이 특히 저자원 조건에서 일반화에 도움을 준다고 주장한다.

기존 접근과의 차별점은 명확하다. UFAL Whisper나 Simul-Whisper 같은 방법은 Whisper를 원래 방식대로 반복 호출하는 pseudo-streaming에 가깝다. 반면 이 논문은 Whisper 내부에 streaming-friendly한 CTC 경로를 추가하고, encoder를 causal mask로 다시 적응시켜 진짜 streaming 모델에 더 가깝게 만든다.

## 3. 상세 방법 설명

전체 시스템은 크게 세 부분으로 구성된다. 첫째는 Whisper encoder, 둘째는 새로 붙인 CTC decoder, 셋째는 원래의 Whisper attention decoder이다. 훈련 시에는 CTC decoder와 attention decoder가 같은 정답 문장을 예측하도록 함께 학습된다.

논문은 이때 hybrid CTC-attention loss를 사용한다. 손실 함수는 다음과 같다.

$$
L = \alpha \cdot L_{\mathrm{CTC}} + (1-\alpha) \cdot L_{\mathrm{Attention}}
$$

여기서 $L_{\mathrm{CTC}}$는 CTC branch의 손실이고, $L_{\mathrm{Attention}}$는 원래 Whisper decoder의 sequence-to-sequence 손실이다. $\alpha$는 두 손실의 비중을 조절하는 가중치다. 논문 본문에는 $\alpha$의 구체적인 값은 제시되지 않았다. 따라서 실제 구현에서 어떤 비율을 썼는지는 제공된 텍스트만으로는 알 수 없다.

스트리밍 가능성을 만드는 핵심은 encoder 학습 방식이다. U2 방식에 따라 training 중 dynamic attention mask를 적용하여, encoder의 hidden representation이 과거 정보와 아주 제한된 미래 정보에만 의존하도록 만든다. 이렇게 해야 inference 시에 chunk 단위로 오디오가 들어와도 encoder가 학습 때와 비슷한 조건에서 동작할 수 있다. 논문에서는 학습 중 chunk size를 0.1초에서 1.0초 사이에서 무작위로 샘플링했다고 설명한다. 이는 모델이 특정 chunk size에 과적합되지 않고 여러 latency 조건에 견디도록 만들기 위한 장치다.

스트리밍 추론에서는 오디오가 chunk 단위로 encoder에 들어간다. CTC decoder는 prefix beam search를 수행해 상위 $k$개의 부분 전사 후보를 생성한다. 이후 endpoint가 감지되면 attention decoder가 이 후보들을 rescoring하여 최종 문장을 선택한다. endpoint는 0.5초의 무음이 감지되었을 때, 또는 최대 지연(max delay)에 도달했을 때 발생한다. 이 구조의 의미는 분명하다. 실시간 partial output은 CTC가 담당하고, 최종 결과 품질은 Whisper decoder가 보완한다.

논문은 attention rescoring이 autoregressive decoding을 매 토큰 수행하는 방식이 아니라, diagonal causal attention mask를 써서 batched single pass로 처리된다고 설명한다. 즉, rescoring 과정도 최대한 효율적으로 설계했다. 또한 encoder 쪽에는 incremental streaming inference를 위한 key-value cache를 구현해 이전 chunk의 계산을 재사용한다. 이 덕분에 Whisper Medium처럼 7.69억 개 파라미터를 가진 큰 모델도 CPU에서 실시간에 가깝게 동작할 수 있다고 보고한다.

Hybrid tokenizer의 동작 방식도 중요하다. CTC decoder는 8,000개 축소 토큰 공간에서 예측을 수행한다. 훈련 때는 SentencePiece를 이용해 이 토큰들로 CTC target을 만든다. 반면 attention decoder는 원래 Whisper tokenizer의 전체 token space를 유지한다. 추론 때는 CTC hypothesis를 문자열로 복원한 뒤, 다시 Whisper tokenizer로 retokenization하고, Whisper 전용 prompt token을 붙여 attention decoder rescoring에 넣는다. 즉, 두 decoder가 서로 다른 tokenization 체계를 쓰며, 이 둘을 연결하기 위한 retokenizer가 시스템 중간에 존재한다.

훈련 절차는 세 단계다. 먼저 causal attention mask를 적용한 상태에서 attention loss만으로 1 epoch 학습하여 Whisper가 30초보다 짧은 입력과 streaming encoder 조건에 적응하게 만든다. 그 다음 CTC classification head를 추가하고, 다른 파라미터는 모두 고정한 채 CTC loss만으로 2 epoch 학습한다. 마지막으로 전체 파라미터를 모두 풀고 hybrid loss로 validation WER가 3 epoch 연속 개선되지 않을 때까지 학습한다. 저자들은 이렇게 pretrained Whisper의 파라미터에서 너무 멀어지지 않게 해야 일반화가 더 잘된다고 해석한다.

## 4. 실험 및 결과

실험은 주로 두 데이터셋에서 수행되었다. 하나는 LibriSpeech이고, 다른 하나는 저자들이 내부적으로 구축한 earnings call 데이터셋이다. 논문의 주된 관심은 후자에 있다. 이 데이터셋은 2023년 이전의 earnings call에서 샘플링한 5,800시간 분량의 학습 데이터를 사용하며, forced aligner로 5초에서 20초 길이의 clip으로 분절되어 있다. 테스트셋은 2023년 이후 83개 call에서 뽑은 총 10시간 분량이다. 저자들은 이를 통해 데이터 누수를 피하면서, 긴 오디오에 대한 end-to-end streaming 성능을 평가하려 했다고 설명한다.

이 내부 데이터셋을 고른 이유도 논문에 분명하다. earnings transcript는 punctuation, capitalization, formatting이 잘 정리된 written-form transcript를 제공하므로, Whisper가 강한 문장부호 복원과 inverse text normalization까지 한 번에 처리하는 특성과 잘 맞는다. 동시에 금융 용어가 많고 call마다 고유한 어휘가 등장해 일반화 능력을 시험하기 좋다.

가장 먼저 본 것은 데이터 양에 따른 성능 변화다. Whisper Medium을 725, 1450, 2900, 5800시간으로 나누어 fine-tuning했고, single tokenizer와 hybrid tokenizer를 비교했다. 결과는 hybrid tokenizer가 모든 데이터 크기에서 더 낮은 WER를 보였다. 예를 들어 725시간에서는 single tokenizer가 23.51%, hybrid tokenizer가 21.09%였고, 1450시간에서는 20.78% 대 18.97%, 2900시간에서는 19.67% 대 18.26%, 5800시간에서는 17.51% 대 17.30%였다. 즉, 데이터가 적을수록 hybrid tokenizer의 이점이 더 크고, 데이터가 많아질수록 차이가 줄어든다. 이는 논문의 주장과 잘 맞는다. 작은 데이터에서는 큰 vocabulary를 CTC가 감당하기 어렵기 때문에 축소된 token space가 도움이 되지만, 데이터가 충분해지면 그 이점이 줄어든다는 것이다.

또한 같은 구조를 pretrained Whisper 없이 처음부터 학습했을 때 WER가 20.59%였고, pretrained Whisper를 사용했을 때 17.30%였다고 보고한다. 이것은 사전학습된 Whisper encoder-decoder 지식이 streaming adaptation에도 여전히 중요하다는 강한 증거다.

다음으로 runtime 설정의 영향을 살폈다. 5,800시간으로 학습한 가장 좋은 checkpoint를 여러 chunk size에서 평가했는데, chunk가 작아질수록 성능이 떨어졌다. rescoring이 없는 경우와 있는 경우를 비교했을 때, 100 ms chunk에서는 26.93%에서 25.54%로, 240 ms에서는 21.53%에서 21.20%로, 500 ms에서는 18.67%에서 18.35%로, 1000 ms에서는 17.60%에서 17.30%로, 1500 ms에서는 16.85%에서 16.65%로 개선되었다. rescoring은 분명 도움이 되지만 개선 폭은 크지 않다. 저자들은 그 이유를, CTC가 만든 상위 가설들 사이 차이가 보통 punctuation이나 capitalization 같은 작은 차이에 그치기 때문이라고 설명한다.

작은 chunk에서 성능이 나빠지는 이유에 대한 해석도 설득력 있다. 예를 들어 “$1.3 million”이 “1.3 million dollars”처럼 잘못 formatting될 수 있는데, 이는 올바른 formatting에 필요한 미래 문맥이 아직 들어오지 않았기 때문이다. 그리고 CTC prefix beam search 초기에 정답 hypothesis가 pruning되면 나중 rescoring으로도 복구할 수 없다.

최대 지연(max delay)도 중요한 변수였다. 8초, 12초, 16초, 20초 설정에서 각각 WER, RTF, 평균 finalize latency를 측정했다. 8초에서는 WER 19.26%, RTF 0.23, finalize latency 679 ms였고, 12초에서는 17.30%, 0.30, 1126 ms였다. 16초에서는 17.23%, 0.32, 1490 ms, 20초에서는 16.96%, 0.34, 1935 ms였다. 즉, 더 오래 기다릴수록 정확도는 좋아지지만 계산량과 지연도 증가한다. 저자들은 계산 복잡도가 입력 길이에 대해 quadratic하게 증가한다고 지적한다. 실무적으로는 정확도와 반응 속도 사이의 타협점 설정이 중요하다는 뜻이다.

논문은 end-to-end latency를 세 부분으로 나눈다. chunk buffering latency, partial transcript computation latency, finalize computation latency다. 이 중 partial transcript computation은 저자들 환경에서 약 267 ms였고, finalize latency는 위 표의 수치와 같다. 12초 max delay 설정에서는 실시간 streaming으로 쓸 만한 수준이라고 보지만, finalize latency는 여전히 real-time application에 부담이 될 수 있다고 평가한다. 이를 줄이기 위한 방향으로 Whisper Turbo처럼 더 작은 decoder를 가진 checkpoint를 언급한다.

기존 방법과의 비교도 포함된다. 저자들은 UFAL Whisper 및 non-streaming Whisper baseline과 비교했다. earnings와 LibriSpeech test-clean에서는 작은 chunk size에서 U2 Whisper가 UFAL Whisper보다 더 나은 WER를 보였다고 보고한다. 그러나 더 어려운 LibriSpeech test-other에서는 충분한 학습 데이터가 있어야 U2 Whisper가 UFAL을 넘는다. 반대로 chunk size가 커지면 UFAL Whisper가 더 유리해지는데, 이는 그 방식이 본질적으로 원래 non-streaming Whisper 동작과 더 비슷하기 때문이라고 해석한다.

효율성 측면에서 저자들의 주장은 비교적 강하다. U2 Whisper는 CPU에서도 효율적으로 동작하지만, UFAL Whisper는 Whisper Medium 기준으로 CPU에서 실시간을 달성하지 못했고, GPU에서도 작은 chunk size에서는 실시간보다 느릴 수 있다고 한다. 또한 finalize latency 관점에서 UFAL은 보통 두 번 연속 같은 예측이 나와야 final transcript를 확정하기 때문에 대략 chunk size의 두 배 정도 지연이 생기며, 상한이 엄격하게 보장되지 않는다고 설명한다. 반면 U2 Whisper는 endpoint 기반이기 때문에 평균 finalize latency는 더 클 수 있어도 max delay라는 명시적 상한이 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 Whisper를 단순한 pseudo-streaming 방식이 아니라 구조적으로 streaming-friendly하게 바꾸려 했다는 점이다. CTC decoder를 추가하고 encoder를 causal mask로 fine-tuning하여, partial transcript 생성과 최종 rescoring을 명확히 분리했다. 이는 이론적으로도 자연스럽고, 실제 CPU 기반 실시간 처리 가능성까지 보여 주었다는 점에서 공학적 가치가 크다.

또 다른 강점은 hybrid tokenizer라는 비교적 단순하지만 효과적인 설계다. 논문은 특히 데이터가 적을 때 이 방식이 뚜렷한 개선을 준다는 것을 실험으로 보여 준다. 이는 Whisper의 거대한 token space를 그대로 CTC에 쓰는 것이 항상 좋은 선택이 아님을 잘 보여 준다. 또한 단순히 accuracy만이 아니라 chunk size, max delay, RTF, finalize latency를 함께 분석해 실제 배포 관점의 trade-off를 제시한 점도 장점이다.

반면 한계도 분명하다. 첫째, 높은 성능을 위해서는 충분한 in-domain fine-tuning 데이터가 필요하다. 논문 스스로도 challenging한 test set에서는 limited data로는 U2 path가 잘 일반화되지 않을 수 있다고 인정한다. 즉, Whisper의 강력한 사전학습만으로 streaming 전환이 자동으로 해결되지는 않는다.

둘째, attention rescoring의 이득이 생각보다 작다. 이는 두 가지 해석이 가능하다. 하나는 CTC가 이미 충분히 좋은 후보를 생성하고 있어 rescoring 여지가 작다는 것이고, 다른 하나는 rescoring decoder가 잠재력을 충분히 활용하지 못하고 있다는 것이다. 제공된 텍스트만으로는 어느 쪽이 더 본질적인 원인인지 단정할 수 없다.

셋째, finalize latency는 여전히 크다. 특히 12초 max delay에서 평균 finalize latency가 1126 ms이고, 20초에서는 1935 ms까지 증가한다. 실시간 응용에서 partial transcript만 중요한 경우는 괜찮을 수 있지만, final transcript의 신속한 확정이 중요한 환경에서는 부담일 수 있다.

넷째, 실험 설정 중 일부 세부사항은 제공된 텍스트만으로는 부족하다. 예를 들어 hybrid loss의 $\alpha$ 값, optimizer나 learning rate, 정확한 endpoint detection 구현 세부사항, 8,000개 토큰을 “Whisper tokenizer의 첫 8,000개”로 고른 구체 기준 등은 명확히 설명되지 않았다. 따라서 동일한 결과를 완전히 재현하려면 원문 전체나 공개 구현을 더 확인해야 할 가능성이 있다.

비판적으로 보면, 이 방법은 “Whisper를 streaming으로 바꾸는 현실적인 엔지니어링 해법”으로는 강하지만, 근본적으로 streaming에 최적화된 구조인 RNN-T나 native CTC 모델과 비교했을 때 구조적 복잡성이 남아 있다. 즉, Whisper의 강한 pretrained knowledge를 활용하기 위해 two-pass 구조와 retokenization까지 도입한 것이므로, 시스템 단순성보다는 사전학습 활용을 우선한 접근이라고 볼 수 있다.

## 6. 결론

이 논문은 Whisper를 U2 구조로 재구성하여 streaming ASR에 적응시키는 방법을 제안했고, CTC 기반 partial decoding과 Whisper decoder 기반 rescoring을 결합해 실시간성과 정확도 사이의 균형을 추구했다. 특히 hybrid tokenizer를 통해 CTC branch의 데이터 효율성과 일반화를 개선했고, 이 효과가 저자원 조건에서 특히 크다는 점을 실험으로 보였다.

실무적으로 이 연구는 의미가 크다. 이미 강력한 Whisper를 버리지 않고도, 충분한 fine-tuning 데이터와 적절한 runtime 설정이 있다면 CPU 기반 real-time streaming ASR로 활용할 수 있음을 보여 주기 때문이다. 앞으로는 pretrained decoder의 언어 지식을 rescoring 이상으로 더 적극 활용하거나, finalize latency를 줄이는 방향의 연구가 중요할 것으로 보인다. 전체적으로 이 논문은 “대규모 사전학습 음성 인식 모델을 실제 streaming 환경으로 옮기는 방법”에 대해 구체적이고 실용적인 답을 제시한 작업이라고 평가할 수 있다.
