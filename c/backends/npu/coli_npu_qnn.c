#include "coli_npu_qnn.h"
#include <HTP/QnnHtpMem.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

static size_t element_size(Qnn_DataType_t t) {
    switch (t) {
    case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8: return 1;
    case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16: return 2;
    case QNN_DATATYPE_INT_32: case QNN_DATATYPE_UINT_32: case QNN_DATATYPE_FLOAT_32: return 4;
    default: return 0; /* Packed INT4/FP4 require a separately qualified graph. */
    }
}
int coli_npu_qnn_register(ColiNpuRegistration *r, const QNN_INTERFACE_VER_TYPE *api,
                         Qnn_ContextHandle_t ctx, ColiNpuBuf *b,
                         size_t offset, uint32_t rank, uint32_t *dims,
                         Qnn_DataType_t dtype, Qnn_ErrorHandle_t *qe) {
    Qnn_MemDescriptor_t d = QNN_MEM_DESCRIPTOR_INIT;
    QnnMemHtp_Descriptor_t h;
    Qnn_ErrorHandle_t e;
    size_t bytes = element_size(dtype), size = coli_npu_buf_size(b);
    uint32_t i;
    int rc;
    if (qe) *qe = QNN_SUCCESS;
    if (!r || r->handle || r->buffer || r->dims || !api || !api->memRegister || !api->memDeRegister ||
        !ctx || !b || coli_npu_buf_fd(b) < 0 || !rank || rank > 32 || !dims || !bytes) return EINVAL;
    if (offset % bytes) return EINVAL;
    for (i = 0; i < rank; ++i) {
        if (!dims[i] || bytes > SIZE_MAX / dims[i]) return EOVERFLOW;
        bytes *= dims[i];
    }
    if (offset > size || bytes > size - offset) return EINVAL;
    r->dims = malloc((size_t)rank * sizeof(*dims));
    if (!r->dims) return ENOMEM;
    memcpy(r->dims, dims, (size_t)rank * sizeof(*dims));
    if ((rc = coli_npu_buf_retain(b))) { free(r->dims); r->dims = NULL; return rc; }
    memset(&h, 0, sizeof(h));
    h.type = QNN_HTP_MEM_SHARED_BUFFER;
    h.size = size;
    h.sharedBufferConfig.fd = coli_npu_buf_fd(b);
    h.sharedBufferConfig.offset = offset;
    d.memShape.numDim = rank;
    d.memShape.dimSize = r->dims;
    d.dataType = dtype;
    d.memType = QNN_MEM_TYPE_CUSTOM;
    d.customInfo = &h;
    e = api->memRegister(ctx, &d, 1, &r->handle);
    if (qe) *qe = e;
    if (e != QNN_SUCCESS || !r->handle) {
        r->handle = NULL; coli_npu_buf_release(b); free(r->dims); r->dims = NULL; return EIO;
    }
    r->buffer = b; r->api = api;
    r->rank = rank; r->dtype = dtype;
    return 0;
}
int coli_npu_qnn_unregister(ColiNpuRegistration *r, Qnn_ErrorHandle_t *qe) {
    Qnn_ErrorHandle_t e;
    if (qe) *qe = QNN_SUCCESS;
    if (!r) return EINVAL;
    if (!r->handle) return 0;
    e = r->api->memDeRegister(&r->handle, 1);
    if (qe) *qe = e;
    if (e != QNN_SUCCESS) return EIO; /* retain backing memory on failure */
    coli_npu_buf_release(r->buffer);
    free(r->dims);
    memset(r, 0, sizeof(*r));
    return 0;
}
int coli_npu_qnn_bind(Qnn_Tensor_t *t, const ColiNpuRegistration *r) {
    uint32_t rank, *dims;
    Qnn_DataType_t dtype;
    if (!t || !r || !r->handle) return EINVAL;
    if (t->version == QNN_TENSOR_VERSION_1) {
        rank = t->v1.rank; dims = t->v1.dimensions; dtype = t->v1.dataType;
    } else if (t->version == QNN_TENSOR_VERSION_2) {
        rank = t->v2.rank; dims = t->v2.dimensions; dtype = t->v2.dataType;
    } else return ENOTSUP;
    if (rank != r->rank || dtype != r->dtype || !dims ||
        memcmp(dims, r->dims, rank * sizeof(*dims))) return EINVAL;
    if (t->version == QNN_TENSOR_VERSION_1) {
        t->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; t->v1.memHandle = r->handle;
    } else if (t->version == QNN_TENSOR_VERSION_2) {
        t->v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE; t->v2.memHandle = r->handle;
    } else return ENOTSUP;
    return 0;
}
