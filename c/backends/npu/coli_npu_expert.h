#ifndef COLI_NPU_EXPERT_H
#define COLI_NPU_EXPERT_H
#include <stdint.h>
/* Experimental single-token SwiGLU adapter. SDK types stay private.
 * Planar unsigned INT4 (q+8), 64 values / float32 scale, gate|up|down.
 * Owns two registered DDR slots; execution is synchronous, not overlapped.
 * No fallback on QNN/numerical failure. Caller retains inputs until return.
 */
typedef struct ColiNpuExpert ColiNpuExpert;
int coli_npu_expert_open(ColiNpuExpert **out, const char *backend,
                        const char *context, const char *rpcmem, int d, int f);
int coli_npu_expert_run(ColiNpuExpert *e, const float *x,
                       const uint8_t *planar, const float *gs,
                       const float *us, const float *ds, float *y);
int coli_npu_expert_close(ColiNpuExpert **e);
unsigned long long coli_npu_expert_calls(const ColiNpuExpert *e);
#endif
