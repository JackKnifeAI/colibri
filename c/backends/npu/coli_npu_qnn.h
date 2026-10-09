#ifndef COLI_NPU_QNN_H
#define COLI_NPU_QNN_H
#include "coli_npu_buf.h"
#include <QnnInterface.h>

/* SDK-only adapter. QNN context and API must outlive all registrations.
 * One registration per tensor per pool slot; keep until engine shutdown.
 */
typedef struct {
    ColiNpuBuf *buffer;
    Qnn_MemHandle_t handle;
    const QNN_INTERFACE_VER_TYPE *api;
    uint32_t rank, *dims;
    Qnn_DataType_t dtype;
} ColiNpuRegistration;
int coli_npu_qnn_register(ColiNpuRegistration *r, const QNN_INTERFACE_VER_TYPE *api,
                         Qnn_ContextHandle_t context, ColiNpuBuf *buf,
                         size_t offset, uint32_t rank, uint32_t *dims,
                         Qnn_DataType_t dtype, Qnn_ErrorHandle_t *qnn_error);
int coli_npu_qnn_unregister(ColiNpuRegistration *r, Qnn_ErrorHandle_t *qnn_error);
int coli_npu_qnn_bind(Qnn_Tensor_t *tensor, const ColiNpuRegistration *r);
#endif
