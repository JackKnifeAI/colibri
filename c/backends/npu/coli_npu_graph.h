#ifndef COLI_NPU_GRAPH_H
#define COLI_NPU_GRAPH_H
#include "coli_npu_qnn.h"

/* SDK-only, synchronous cached-graph execution. The model adapter supplies
 * tensors from its validated export metadata, including quantization params.
 * Never re-finalizes or rewrites static weights. Not safe for concurrent calls.
 */
typedef struct ColiNpuGraph ColiNpuGraph;
int coli_npu_graph_open(ColiNpuGraph **out, const char *backend_library,
                        const char *context_binary, const char *graph_name,
                        Qnn_ErrorHandle_t *qnn_error);
const QNN_INTERFACE_VER_TYPE *coli_npu_graph_api(const ColiNpuGraph *g);
Qnn_ContextHandle_t coli_npu_graph_context(const ColiNpuGraph *g);
int coli_npu_graph_execute(ColiNpuGraph *g, const Qnn_Tensor_t *inputs, uint32_t ni,
                           Qnn_Tensor_t *outputs, uint32_t no,
                           Qnn_ErrorHandle_t *qnn_error);
/* All associated memory registrations must be released BEFORE close.
 * On cleanup failure, retains *g so caller can retry; do not free its buffers.
 * Even a failed open may leave a partial object if vendor cleanup failed.
 */
int coli_npu_graph_close(ColiNpuGraph **g, Qnn_ErrorHandle_t *qnn_error);
#endif
