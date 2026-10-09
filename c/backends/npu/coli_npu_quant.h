#ifndef COLI_NPU_QUANT_H
#define COLI_NPU_QUANT_H
#include <stddef.h>

/* Explicit I4/F16 block format (not GGUF, FP4, or a Colibri model container):
 * each 34-byte block stores a little-endian nonnegative finite FP16 scale,
 * then 64 signed two's-complement INT4 values, low nibble first.
 * Expand in place, backwards, into little-endian IEEE FP16 with ties-to-even
 * and canonical positive zero.
 * The caller owns the CPU write/cache bracket and must not expose this buffer
 * to a device until success. On a data error the buffer may be partly changed;
 * discard the job. No allocation or extra full-weight staging copy is used.
 */
int coli_npu_i4_f16_expand(void *buffer, size_t encoded_bytes,
                          size_t capacity, size_t values);
#endif
