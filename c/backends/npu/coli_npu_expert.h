#ifndef COLI_NPU_EXPERT_H
#define COLI_NPU_EXPERT_H
#include <stdint.h>
/* Experimental single-token SwiGLU adapter. SDK types stay private.
 * Planar unsigned INT4 (q+8), 64 values / float32 scale, gate|up|down.
 * Owns two registered DDR slots. COLI_NPU_OVERLAP=1 enables one QNN worker.
 * No fallback on QNN/numerical failure. Caller retains inputs until return.
 */
typedef struct ColiNpuExpert ColiNpuExpert;
int coli_npu_expert_open(ColiNpuExpert **out, const char *backend,
                        const char *context, const char *rpcmem, int d, int f);
int coli_npu_expert_run(ColiNpuExpert *e, const float *x,
                       const uint8_t *planar, const float *gs,
                       const float *us, const float *ds, float *y);
/* One foreground caller. prepare may fill the OTHER slot while an execution
 * is in flight. submit permits one execution; wait consumes it before reuse.
 * close joins pending execution before releasing any registration/buffer. */
int coli_npu_expert_prepare(ColiNpuExpert *e, unsigned slot, const uint8_t *planar,
                           const float *gs, const float *us, const float *ds);
int coli_npu_expert_submit(ColiNpuExpert *e, unsigned slot, const float *x);
int coli_npu_expert_wait(ColiNpuExpert *e, float *y);
int coli_npu_expert_close(ColiNpuExpert **e);
typedef struct {
    unsigned long long calls;
    double staging_ms, graph_ms, output_ms;
} ColiNpuExpertStats;
void coli_npu_expert_stats(const ColiNpuExpert *e, ColiNpuExpertStats *stats);
unsigned long long coli_npu_expert_calls(const ColiNpuExpert *e);
#endif
