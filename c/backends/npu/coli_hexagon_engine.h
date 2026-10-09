#ifndef COLI_HEXAGON_ENGINE_H
#define COLI_HEXAGON_ENGINE_H
#include "coli_npu_buf.h"
#include <stdint.h>

#define COLI_BACKEND_HEXAGON "hexagon"
#define COLI_HEXAGON_ABI_VERSION 2u
typedef struct ColiHexagonEngine ColiHexagonEngine;
typedef struct {
    uint64_t offset;
    size_t bytes;
    uint32_t expert, tile;
    float gate;
} ColiHexagonJob;

/* Model adapter owns routing semantics, quantization, tensor layout, KV cache,
 * and graph handles. All callbacks synchronous; execute MUST return only after
 * the device has released its input buffer, including on errors. Keep jobs alive
 * until finish_layer. Dense/route may be CPU or a precompiled QNN graph.
 * No speculative early routing is performed by this generic scheduler.
 */
typedef struct {
    int (*begin_token)(void *user, uint64_t position);
    int (*dense_route)(void *user, uint32_t layer,
                       const ColiHexagonJob **jobs, size_t *count);
    /* Only work independent of selected expert output belongs here. */
    int (*dense_tail)(void *user, uint32_t layer);
    int (*execute)(void *user, uint32_t layer, const ColiHexagonJob *job,
                   ColiNpuBuf *weights, unsigned slot);
    int (*finish_layer)(void *user, uint32_t layer);
    int (*finish_token)(void *user);
    /* Optional reader-thread transform, after pread and before CPU WRITE END.
     * Operates in-place in mapped DDR. Must not retain the pointer, access QNN,
     * or mutate foreground state; may run concurrently with execute/dense_tail.
     * Read-only model metadata may be shared. Failure poisons the token.
     * job->bytes is encoded file length; capacity is the complete output slot.
     */
    int (*prepare_weights)(void *user, const ColiHexagonJob *job,
                           void *mapped, size_t capacity);
} ColiHexagonOps;

/* Engine retains buffers and duplicates the corpus fd. Caller owns graph/QNN
 * registrations. No concurrent calls to step/free. File must be immutable.
 */
int coli_hexagon_engine_create(ColiHexagonEngine **out, int corpus_fd,
                              ColiNpuBuf *a, ColiNpuBuf *b,
                              const ColiHexagonOps *ops, void *user);
int coli_hexagon_token_step(ColiHexagonEngine *e, uint64_t position, uint32_t layers);
/* Joins prefetch worker before releasing buffers, even after a failed step. */
void coli_hexagon_engine_free(ColiHexagonEngine **e);
#endif
