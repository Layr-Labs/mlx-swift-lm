# BenchCBv2RealModel report

| | |
|---|---|
| Model | /Users/gaj/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3 |
| Chip | Apple M4 Max |
| RAM | 128 GB |
| OS | Version 26.5 (Build 25F71) |
| Compiled decode (legacy) | true |
| Date | 2026-07-02T05:59:31Z |

model class: GPTOSSModel; layers: 24
vocabSize=201088

## Performance (maxTokens 128)

| engine | B | prompts (tok) | decode TPS/req | agg TPS | TTFT p50 (ms) | ITL p50 (ms) |
|---|---|---|---|---|---|---|
| legacy-compiled | 1 | 500 | 99.4 | 76.6 | 394 | 10.0 |
| legacy-compiled | 2 | 100/1500 | 70.8 | 50.7 | 3256 | 13.5 |
| legacy-compiled | 4 | 100/500/1500/500 | 45.0 | 60.8 | 5594 | 22.3 |

Per-request detail:
  legacy-compiled B=1:
    req 0: prompt=500 tokens=128 ttft=394ms decodeTPS=99.4 finish=length
  legacy-compiled B=2:
    req 0: prompt=100 tokens=128 ttft=3256ms decodeTPS=70.8 finish=length
    req 1: prompt=1500 tokens=128 ttft=3256ms decodeTPS=70.8 finish=length
  legacy-compiled B=4:
    req 0: prompt=100 tokens=128 ttft=5594ms decodeTPS=45.0 finish=length
    req 1: prompt=500 tokens=128 ttft=5594ms decodeTPS=45.0 finish=length
    req 2: prompt=1500 tokens=128 ttft=5594ms decodeTPS=45.0 finish=length
    req 3: prompt=500 tokens=128 ttft=5594ms decodeTPS=45.0 finish=length
