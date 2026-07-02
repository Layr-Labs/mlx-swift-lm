# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | false |
| Date | 2026-07-02T05:58:15Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Correctness (v2 contiguous, greedy)

prompt tokens: target=79 short=74 long=1600 eos=[199999, 200002, 200012]
[b1-sanity] finish=stop tokens=195 loop=false
[b1-sanity] text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue due to Rayleigh scattering of sunlight by air molecules, which scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears blue because sunlight is scattered by the Earth's atmosphere, and shorter blue wavelengths are scattered more efficiently than longer red wavelengths. That is two sentences. Ensure it's exactly two sentences. Let's craft: "The sky looks blue because sunlight is scattered by the Earth's atmosphere, and the scattering is stronger for shorter wavelengths. This preferential scattering of blue light over red light makes the sky appear blue to our eyes." That's two sentences. Good.<|end|><|start|>assistant<|channel|>final<|message|>The sky looks blue because sunlight is scattered by the Earth's atmosphere, and the scattering is stronger for shorter wavelengths. This preferential scattering of blue light over red light makes the sky appear blue to our eyes.
[invariance] solo tokens=64 finish=length
[invariance] solo-repeat divergence=nil
[invariance] solo text: <|channel|>analysis<|message|>We need to explain why the sky is blue in two sentences. So we need to produce a concise explanation. The sky is blue due to Rayleigh scattering of sunlight by air molecules, which scatters shorter wavelengths (blue) more strongly than longer wavelengths (red). So we can say: The sky appears
[invariance] burst target tokens=64 divergence=nil
[invariance] mid-join: neighbors had (9, 5) tokens at join; target tokens=64 divergence=nil
[invariance] neighbor-invariance (burst vs mid-join at B=3): divergence=nil
[chunked-prefill] chunked=64 tok unchunked=64 tok divergence=nil
[chunked-prefill] chunked text: <|channel|>analysis<|message|>The user asks: "Summarize the following passage in one sentence:" and then repeats the same sentence many times. The passage is basically: "The quick brown fox jumps over the lazy dog while the seasoned cartographer annotates ancient maps with meticulous margin notes about tides, trade winds, and the slow

- **b1-sanity**: PASS — finish=stop completion=195 repetitionLoop=false
- **invariance**: PASS — solo=64 tok; solo-repeat deterministic; neighbor-invariant at B=3 (burst == mid-join); burst identical to solo; mid-join identical to solo
- **chunked-prefill**: PASS — token-exact over 64 tokens (prompt=1600)

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy | 1 | 500 | 35.2 | 31.1 | 506 | 22.2 |
| legacy | 2 | 100/1500 | 31.6 | 32.1 | 3968 | 26.5 |
| legacy | 4 | 100/500/1500/500 | 15.1 | 33.8 | 6710 | 47.6 |
| v2 | 1 | 500 | 26.2 | 23.7 | 545 | 20.7 |
| v2 | 2 | 100/1500 | 23.4 | 35.7 | 1581 | 23.5 |
| v2 | 4 | 100/500/1500/500 | 19.2 | 52.2 | 2710 | 34.8 |
| v2-paged | 1 | 500 | 0.0 | 0.0 | -804664582551 | 0.0 |
| v2-paged | 2 | 100/1500 | 0.0 | 0.0 | -804664615247 | 0.0 |
| v2-paged | 4 | 100/500/1500/500 | 0.0 | 0.0 | -804664655509 | 0.0 |

Per-request detail:
  legacy B=1:
    req 0: prompt=500 tokens=128 ttft=506ms decodeTPS=35.2 finish=length
  legacy B=2:
    req 0: prompt=100 tokens=128 ttft=3968ms decodeTPS=31.6 finish=length
    req 1: prompt=1500 tokens=128 ttft=3968ms decodeTPS=31.6 finish=length
  legacy B=4:
    req 0: prompt=100 tokens=128 ttft=6710ms decodeTPS=15.1 finish=length
    req 1: prompt=500 tokens=128 ttft=6710ms decodeTPS=15.1 finish=length
    req 2: prompt=1500 tokens=128 ttft=6710ms decodeTPS=15.1 finish=length
    req 3: prompt=500 tokens=128 ttft=6710ms decodeTPS=15.1 finish=length
  v2 B=1:
    req 0: prompt=500 tokens=128 ttft=545ms decodeTPS=26.2 finish=length
  v2 B=2:
    req 0: prompt=100 tokens=128 ttft=680ms decodeTPS=19.8 finish=length
    req 1: prompt=1500 tokens=128 ttft=2481ms decodeTPS=27.1 finish=length
  v2 B=4:
    req 0: prompt=100 tokens=128 ttft=2710ms decodeTPS=18.0 finish=length
    req 1: prompt=500 tokens=128 ttft=2710ms decodeTPS=18.0 finish=length
    req 2: prompt=1500 tokens=128 ttft=4300ms decodeTPS=23.1 finish=length
    req 3: prompt=500 tokens=128 ttft=2710ms decodeTPS=18.0 finish=length
  v2-paged B=1:
    req 0: prompt=0 tokens=0 ttft=-804664582551ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
  v2-paged B=2:
    req 0: prompt=0 tokens=0 ttft=-804664615247ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
    req 1: prompt=0 tokens=0 ttft=-804664615247ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
  v2-paged B=4:
    req 0: prompt=0 tokens=0 ttft=-804664655509ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
    req 1: prompt=0 tokens=0 ttft=-804664655509ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
    req 2: prompt=0 tokens=0 ttft=-804664655509ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
    req 3: prompt=0 tokens=0 ttft=-804664655509ms decodeTPS=0.0 finish=error("engine step exceeded 30s watchdog")
