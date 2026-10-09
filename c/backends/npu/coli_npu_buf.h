#ifndef COLI_NPU_BUF_H
#define COLI_NPU_BUF_H
#include <stddef.h>
#include <stdint.h>

/* Optional C99 host lane. No QNN types escape into the model engine.
 * Return 0 or a positive errno. Objects are single-owner, not thread-safe.
 * Caller must join all device work before CPU access, deregistration or free.
 */
typedef struct ColiNpuBuf ColiNpuBuf;
int coli_npu_buf_heap(ColiNpuBuf **out, const char *heap, size_t bytes);
int coli_npu_buf_rpcmem(ColiNpuBuf **out, const char *library, size_t bytes);
/* Duplicate fd and mmap it. Import only exporter-qualified DMA-BUFs. */
int coli_npu_buf_import(ColiNpuBuf **out, int fd, size_t bytes);
int coli_npu_buf_begin(ColiNpuBuf *buf, int write, void **address);
int coli_npu_buf_end(ColiNpuBuf *buf);
int coli_npu_buf_free(ColiNpuBuf **buf);
size_t coli_npu_buf_size(const ColiNpuBuf *buf);
int coli_npu_buf_fd(const ColiNpuBuf *buf);
/* Registration references prevent accidental release before QNN deregisters. */
int coli_npu_buf_retain(ColiNpuBuf *buf);
void coli_npu_buf_release(ColiNpuBuf *buf);
#ifdef COLI_NPU_TESTING
/* Host scheduler tests only; NEVER claims shared/NPU memory. */
int coli_npu_buf_test(ColiNpuBuf **out, size_t bytes);
#endif
#endif
