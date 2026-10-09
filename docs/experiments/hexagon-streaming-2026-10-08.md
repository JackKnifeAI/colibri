# S25 dynamic expert streaming: calibration and INT4 storage

Device: SM-S938W / SM8750, Android 16, native Termux UID 10325. QAIRT 2.37
signed HTP runtime, v79 context, Clang 21.1.8. These are private single-layer
expert fixtures, not generation by the complete target model. The 41-layer
source corpus still needs model/MTP provenance validation before integration.
No SDK libraries, model weights or compiled contexts are included here.

## Numerical diagnosis

The old dynamic UFIXED16 graph was calibrated on expert 0 and one input. Its
intermediate gate, up, SiLU, product and output ranges did not cover expert 1.
CPU emulation of its quantization boundaries reproduced relative L2 of 0.09038,
close to the device's 0.09027. Its dynamic weights also pass through internal
UFIXED8 conversions; UFIXED16 input registration does not imply 16-bit GEMM.

Recalibration on experts 0–7, using the original input scaled by 0.75, 1, 1.25
and -1, improved original expert 1 to relative L2 0.04362. It nevertheless failed
expert 8 on the original input (0.34124). Keep that candidate rejected. This
demonstrates why fitting a few expert ranges is not general MoE calibration.

A separate dynamic FP16 graph with FP16 IO avoids these fixed activation ranges.
The same 16 experts were checked on three vectors: the original input, a fixed
seed-7908 permutation, and the negative original. Each case runs twice in
opposite orders to force both shared-memory slots: 48 distinct cases, 96 calls.
The preselected acceptance gates remain cosine >= 0.999 and relative L2 <= 0.05.

| Path | Min cosine | Max relative L2 | Device calls passing | Repeated output |
| --- | ---: | ---: | ---: | --- |
| FP16 storage / FP16 execution | 0.999966937 | 0.008596496 | 96/96 | Bit-identical |
| INT4 storage / FP16 execution, scalar unpack | 0.999967083 | 0.008729137 | 96/96 | Bit-identical |
| INT4 storage / FP16 execution, NEON unpack | 0.999967083 | 0.008729137 | 96/96 | Bit-identical |

**Reference distinction:** FP16 results compare against the original FP32
fixture. INT4 results compare against a CPU reference using the same quantized
weights after FP16 expansion. The simple group-64 max-absolute fixture quantizer
itself adds up to 0.21197 relative L2 against original FP32 expert output.
This is not a full-model quality/perplexity qualification or an endorsed model
conversion recipe. No FP4 path or native INT4 arithmetic claim is made.

## Data path and storage

Each block stores a nonnegative finite little-endian FP16 scale and 32 packed
bytes for 64 signed INT4 values (two's complement, low nibble first). An expert
has 3 * 2048 * 512 values: 1,671,168 encoded bytes including scales, versus
6,291,456 bytes in FP16. This is an explicit fixture storage format, not a
drop-in parser for GGUF or Colibri safetensors.

The prefetch worker reads encoded bytes directly into a registered DDR slot,
expands backwards in place inside the CPU-write coherency bracket, then releases
the slot for QNN. Only small block scratch is used. Two slots remain 6 MiB each;
the reduction is in file traffic, not execution-buffer size. Expansion overlaps
the other slot's synchronous graph execution. QNN still owns VTCM and internal
layout/DMA decisions. No UFS-to-TCM peer DMA or stall-free guarantee is implied.

The optimized unpacker uses baseline AArch64 NEON integer unpacking, FP32
multiplication and FP16 conversion; it does not require ARM FP16 vector
arithmetic extensions. Other hosts use integer-only round-to-even expansion.
Malformed scales, capacity errors and FP16 overflow reject the job; a transform
error poisons the token before that job can reach graph execution.

## Warm replay timing

Paired sequential runs, 96 graph calls per run, including file reads, unpack,
execution, and numerical checks; graph creation is excluded. Files were warm
in the page cache. Each run passed every accuracy and repeat check.

| Run | Scalar pipeline (ms) | NEON pipeline (ms) |
| --- | ---: | ---: |
| 1 | 1307.139 | 301.846 |
| 2 | 1236.503 | 300.937 |
| 3 | 1356.728 | 298.583 |

The ratio of medians is 4.34. Median individual graph execution was about
2.98 ms in both paths. This is an unpack/pipeline improvement, not a token rate,
cold UFS benchmark, sustained thermal test, or proof of HMX operator placement.

## Reproduction

`pack_stream_fixture.py` consumes QAIRT context-info JSON, row-major FP32 expert
weights, FP32 inputs, and CPU references. It validates shapes/types and derives
all QNN tensor IDs, scales, offsets, names and dimensions into a private generated
header. Its optional `--int4` mode also computes matching CPU SwiGLU references
and saves original-versus-quantized loss separately. NumPy is an offline test
tool dependency; the C runtime remains independent of Python and NumPy.

```sh
python3 c/backends/npu/pack_stream_fixture.py \
  --info /private/fp16-context-info.json \
  --weights /private/weights.f32 --x /private/x.f32 \
  --refs /private/refs.f32 --experts 16 --int4 --output /private/fixture

# Native Termux, with licensed headers and matching private fixtures staged:
npu="$HOME/projects/colibri/c/backends/npu"
fixture="$HOME/hexagon-fixtures/int4"
clang -std=c99 -O2 -Wall -Wextra -Werror -pthread \
  -I"$HOME/sdk/qairt-2.37/include/QNN" -I"$fixture" -I"$npu" \
  "$npu/test_stream_device.c" "$npu/coli_npu_buf.c" "$npu/coli_npu_qnn.c" \
  "$npu/coli_npu_graph.c" "$npu/coli_hexagon_engine.c" "$npu/coli_npu_quant.c" \
  -ldl -lm -o "$fixture/test_stream_device"

LD_LIBRARY_PATH="$HOME/hexagon-fixtures/runtime:/vendor/lib64" \
ADSP_LIBRARY_PATH="$HOME/hexagon-fixtures/runtime/dsp" \
  "$fixture/test_stream_device" "$HOME/hexagon-fixtures/runtime/libQnnHtp.so" \
  /vendor/lib64/libcdsprpc.so "$HOME/hexagon-fixtures/fp16/dyn_fp16.bin" \
  "$fixture/weights.i4" "$fixture/x.u16" "$fixture/refs.f32"
```

Host tests include transformed-buffer ordering, transform failure/poisoning,
overlap, alignment, signed nibble order, round-to-even, subnormal and invalid
scale cases. ASan/UBSan passed. On the phone, NEON and scalar expansion matched
for all 65,536 FP16 scale bit patterns and all 16 signed INT4 codes. The portable
implementation was also checked against NumPy FP16 rounding independently.

Fixture SHA-256:

- FP16 context: `6670d42566ffaee694597d86a8a318fe686ecd4054ca3c5fb366aa06ca7176c1`
- INT4 weights: `192214097e21e723de90f586d2578b4118845ca60ece8e54afe3b0e335962fa6`
- INT4 CPU references: `2e653cacb6b780d3d42f7e36165cbee2af0520c5181827ed508409fb43147d3f`

Next acceptance work is the real model/container adapter, model-level low-bit
quality, dense/router/KV equivalence, cold storage traces and sustained decoding.
