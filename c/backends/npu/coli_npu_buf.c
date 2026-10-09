#define _GNU_SOURCE
#include "coli_npu_buf.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

struct ColiNpuBuf {
    void *data, *library;
    size_t bytes;
    int fd, access, test;
    unsigned references;
    void (*rpc_free)(void *);
};

static int rounded(size_t n, size_t *out) {
    long page = sysconf(_SC_PAGESIZE);
    size_t p;
    if (!n || page <= 0) return EINVAL;
    p = (size_t)page;
    if (n > SIZE_MAX - (p - 1)) return EOVERFLOW;
    *out = ((n + p - 1) / p) * p;
    return 0;
}

int coli_npu_buf_import(ColiNpuBuf **out, int fd, size_t bytes) {
    ColiNpuBuf *b;
    off_t extent;
    int e;
    if (!out || *out || fd < 0 || !bytes) return EINVAL;
    extent = lseek(fd, 0, SEEK_END);
    if (extent < 0) return errno;
    if ((uint64_t)extent < bytes) return EINVAL;
    b = calloc(1, sizeof(*b));
    if (!b) return ENOMEM;
    b->fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (b->fd < 0) { e = errno; free(b); return e; }
    b->data = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
    if (b->data == MAP_FAILED) {
        e = errno; close(b->fd); free(b); return e;
    }
    b->bytes = bytes;
    *out = b;
    return 0;
}

int coli_npu_buf_heap(ColiNpuBuf **out, const char *heap, size_t bytes) {
    struct dma_heap_allocation_data a;
    int h, e, rc;
    size_t n;
    if (!out || *out || !heap || heap[0] != '/') return EINVAL;
    if ((e = rounded(bytes, &n))) return e;
    h = open(heap, O_RDONLY | O_CLOEXEC);
    if (h < 0) return errno;
    memset(&a, 0, sizeof(a));
    a.len = n;
    a.fd_flags = O_RDWR | O_CLOEXEC;
    do { rc = ioctl(h, DMA_HEAP_IOCTL_ALLOC, &a); } while (rc < 0 && errno == EINTR);
    e = rc < 0 ? errno : 0;
    close(h);
    if (e) return e;
    e = coli_npu_buf_import(out, (int)a.fd, n);
    close((int)a.fd);
    return e;
}

int coli_npu_buf_rpcmem(ColiNpuBuf **out, const char *library, size_t bytes) {
    ColiNpuBuf *b;
    void *(*alloc_fn)(int, unsigned, int);
    int (*fd_fn)(void *);
    void *sym;
    size_t n;
    int e;
    if (!out || *out || !library || library[0] != '/') return EINVAL;
    if ((e = rounded(bytes, &n))) return e;
    if (n > INT_MAX) return EOVERFLOW;
    b = calloc(1, sizeof(*b));
    if (!b) return ENOMEM;
    b->fd = -1;
    /* Vendor RPC libraries can retain worker callbacks after freeing a buffer.
     * Keep code mapped for process lifetime; release the allocation itself.
     */
    b->library = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!b->library) { free(b); return ENOENT; }
    sym = dlsym(b->library, "rpcmem_alloc"); memcpy(&alloc_fn, &sym, sizeof(alloc_fn));
    sym = dlsym(b->library, "rpcmem_to_fd"); memcpy(&fd_fn, &sym, sizeof(fd_fn));
    sym = dlsym(b->library, "rpcmem_free"); memcpy(&b->rpc_free, &sym, sizeof(b->rpc_free));
    if (!alloc_fn || !fd_fn || !b->rpc_free) {
        dlclose(b->library); free(b); return ENOSYS;
    }
    /* rpcmem.h: SYSTEM heap 25, RPCMEM_DEFAULT_FLAGS 1. */
    b->data = alloc_fn(25, 1, (int)n);
    if (!b->data) { dlclose(b->library); free(b); return ENOMEM; }
    b->fd = fd_fn(b->data);
    if (b->fd < 0) {
        b->rpc_free(b->data); dlclose(b->library); free(b); return EIO;
    }
    b->bytes = n;
    *out = b;
    return 0;
}

static int sync_buf(ColiNpuBuf *b, uint64_t flags) {
    struct dma_buf_sync s;
    int rc;
    if (b->test) return 0;
    s.flags = flags;
    do { rc = ioctl(b->fd, DMA_BUF_IOCTL_SYNC, &s); }
    while (rc < 0 && (errno == EINTR || errno == EAGAIN));
    /* ENOTTY is an unsupported coherency contract, never silent success. */
    return rc < 0 ? errno : 0;
}
int coli_npu_buf_begin(ColiNpuBuf *b, int write, void **address) {
    int e;
    if (!b || !address || (write != 0 && write != 1)) return EINVAL;
    if (b->access) return EBUSY;
    if ((e = sync_buf(b, DMA_BUF_SYNC_START | (write ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ)))) return e;
    b->access = write ? 2 : 1;
    *address = b->data;
    return 0;
}
int coli_npu_buf_end(ColiNpuBuf *b) {
    int e;
    if (!b || !b->access) return EINVAL;
    e = sync_buf(b, DMA_BUF_SYNC_END | (b->access == 2 ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ));
    if (!e) b->access = 0;
    return e;
}
int coli_npu_buf_free(ColiNpuBuf **p) {
    ColiNpuBuf *b;
    if (!p) return EINVAL;
    if (!(b = *p)) return 0;
    if (b->access || b->references) return EBUSY;
    if (b->test) free(b->data);
    else if (b->rpc_free) { b->rpc_free(b->data); dlclose(b->library); }
    else { if (munmap(b->data, b->bytes) < 0) return errno; close(b->fd); }
    free(b);
    *p = NULL;
    return 0;
}
size_t coli_npu_buf_size(const ColiNpuBuf *b) { return b ? b->bytes : 0; }
int coli_npu_buf_fd(const ColiNpuBuf *b) { return b ? b->fd : -1; }
int coli_npu_buf_retain(ColiNpuBuf *b) {
    if (!b || b->references == UINT_MAX) return EINVAL;
    ++b->references;
    return 0;
}
void coli_npu_buf_release(ColiNpuBuf *b) { if (b && b->references) --b->references; }
#ifdef COLI_NPU_TESTING
int coli_npu_buf_test(ColiNpuBuf **out, size_t bytes) {
    ColiNpuBuf *b;
    if (!out || *out || !bytes) return EINVAL;
    b = calloc(1, sizeof(*b));
    if (!b) return ENOMEM;
    b->data = calloc(1, bytes);
    if (!b->data) { free(b); return ENOMEM; }
    b->bytes = bytes; b->fd = -1; b->test = 1; *out = b;
    return 0;
}
#endif
