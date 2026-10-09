# S25 Hexagon expert optimization closeout

2026-10-09, JackKnifeAI. Samsung S25 Ultra SM-S938W / SM8750, Android 16,
Hexagon v79, Adreno 830, native Termux. Same full 40-layer converted
Huihui Qwen3.6-35B-A3B Claude-4.7-Opus abliterated model as the
[initial NPU run](qwen36-s25-hexagon-2026-10-09.md). All 45 model files were
SHA-256 verified before each run. The source Q8 model is requantized to
signed INT4 gs64 expert storage; the QNN graph computes FP16, not native INT4.

| Path | Decode ms/token (31 steps) | Decode tokens/s | TTFT excluding load | Prompt + 32 output tokens |
| --- | ---: | ---: | ---: | ---: |
| CPU only | 2714.2 | 0.3684 | 81.11 s | 165.3 s |
| Adreno dense + CPU experts | 2082.1 | 0.4803 | 63.88 s | 128.4 s |
| Adreno + original serial Hexagon | 2659.4 | 0.3760 | 89.38 s | 171.8 s |
| Adreno + Hexagon burst | 2021.8 | 0.4946 | 61.22 s | 123.9 s |
| Adreno + Hexagon burst + overlap | 1902.7 | **0.5256** | 53.55 s | 112.5 s |

The final run is 39.8% faster in decode than the original NPU run, and 9.4%
faster than this Adreno baseline. Burst alone accounts for most of the gain;
overlap adds 6.3% versus burst serial. These are individual bounded runs, not
sustained or statistically repeated full-model measurements. Different CPU/NPU
activation precision produces different baseline output/routing. All three
NPU runs generated the same text and cache counts (5,645 hits / 14,515 misses).

## Execution and correctness

The host optionally requests a QNN HTP TURBO performance vote, then overlaps
one graph execution with foreground fetch/INT4-to-FP16 expansion into the
other registered 6 MiB DDR slot. One foreground owner, one QNN worker, explicit
prepare/submit/wait states, and shutdown joining prevent premature slot reuse.
This does not expose TCM or guarantee zero stalls; QNN owns internal placement.
The model still waits for the actual router output. Static dense QNN graphs,
early speculative routing, native INT4/FP4 HMX and Hexagon SERVE are not implemented.

All 320 real-activation comparisons passed on both optimized full runs:
minimum cosine 0.999642427, maximum relative L2 0.026767528; gates are >=0.999
and <=0.05. Each made 20,160 QNN calls and 12,102 Vulkan calls. Peak RSS was
2.57 GB (does not include every GPU/system allocation). Final overlap averages:
staging 1.272 ms, QNN graph 0.780 ms, output 0.003 ms per expert. These overlap
in time and cannot be summed to infer wall time. Decode fetch remains about
1,246 ms/token, the dominant remaining cost.

The independent 96-call fixture passed in serial and overlapped adapter modes,
with byte-identical output files, minimum cosine 0.999967083 and maximum L2
0.008729137. It checks ordering, repeated experts, premature reuse rejection,
double submit/wait and pending-execution cleanup. The SDK-header mock test
covers power ID zero, absent API, failed vote cleanup and failed destroy retry.
Native Android and ordinary CPU qwen36 builds passed.

## Reproduction and provenance

Use the original report's prompt, model, O0 context and Vulkan settings:
cache 8/layer, four CPU threads, context 1024, PILOT=0, HOT=0, 32 generated
tokens, greedy decoding. Add `COLI_NPU_BURST_MODE=1`; add
`COLI_NPU_OVERLAP=1` for the final row. Both variables default off. Thermal
protections remain enabled. Earlier runs reached status 0 or 1; this final
short run is not a sustained thermal qualification.

Original context SHA-256:
`6670d42566ffaee694597d86a8a318fe686ecd4054ca3c5fb366aa06ca7176c1`.
Final executable SHA-256:
`d5f0d55dc71bbd3db54d620ffeb03b6a1cc9f23c9406b4a94b2b7370b883b231`.
Tested sources were on top of `53de7be6`:

- qwen36.c: `b13d6de5e10c3684d1de3529707f183609b12e26a9acf6950783c59a796860e2`
- coli_npu_expert.c: `9ce4b2d62713d7698563fc98708b28be784947864e024fef5447442078ee7377`

Runs: `20261009T092910Z-hexagon-burst`, `20261009T093559Z-hexagon-overlap`.
Raw logs: [burst](qwen36-s25-hexagon-burst-2026-10-09-raw.txt),
[overlap](qwen36-s25-hexagon-overlap-2026-10-09-raw.txt).

A separate O3 fixture context improved the burst fixture median from 168.119
to 158.447 ms (three samples each). It was **not used** for either optimized
full-model result and is not selected for serving. Contexts, SDK/runtime
libraries, weights and credentials remain private/external licensed artifacts.

## Next work after this checkpoint

The operational phone server uses Adreno dense + CPU streamed experts with an
8,192-token cap. The model's nominal 262,144-token window is not qualified here.
There is no disk-backed KV cache: full-attention FP32 K/V alone would consume
about 10 GiB at 262k (40 KiB/token). Long context needs bounded prefill, correct
prefix reuse and file-backed cache validation before raising that limit.
Agent-side summarization is distinct from KV paging. Sustained thermals,
perplexity, long tool-use sessions and a full QNN serving path remain untested.
