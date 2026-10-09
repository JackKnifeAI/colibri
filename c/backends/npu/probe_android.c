#define _GNU_SOURCE
/* Device memory-boundary probe, NOT an inference benchmark. */
#include "coli_npu_qnn.h"
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    void *library, *symbol, *data = NULL;
    Qnn_ErrorHandle_t (*providers)(const QnnInterface_t ***, uint32_t *);
    const QnnInterface_t **list = NULL;
    const QNN_INTERFACE_VER_TYPE *q = NULL;
    Qnn_BackendHandle_t backend = NULL;
    Qnn_DeviceHandle_t device = NULL;
    Qnn_ContextHandle_t context = NULL;
    Qnn_ErrorHandle_t qe = QNN_SUCCESS;
    ColiNpuRegistration registration = {0};
    ColiNpuBuf *buffer = NULL;
    uint32_t count = 0, i, dims[1] = {4096};
    int rc = 1, error;
    if (argc != 3 || argv[1][0] != '/' || argv[2][0] != '/') {
        fprintf(stderr, "usage: %s /absolute/libQnnHtp.so /absolute/libcdsprpc.so\n", argv[0]);
        return 2;
    }
    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!library) { fprintf(stderr, "backend load: %s\n", dlerror()); return 1; }
    symbol = dlsym(library, "QnnInterface_getProviders");
    memcpy(&providers, &symbol, sizeof(providers));
    if (!providers || providers(&list, &count) != QNN_SUCCESS || !list) goto done;
    for (i = 0; i < count; ++i) {
        if (list[i]) fprintf(stderr, "provider core API %u.%u; headers %u.%u\n",
            list[i]->apiVersion.coreApiVersion.major, list[i]->apiVersion.coreApiVersion.minor,
            QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR);
        if (list[i] && list[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
            list[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
            q = &list[i]->QNN_INTERFACE_VER_NAME; break;
        }
    }
    if (!q || !q->backendCreate || !q->backendFree || !q->deviceCreate || !q->deviceFree ||
        !q->contextCreate || !q->contextFree || !q->memRegister || !q->memDeRegister) {
        fprintf(stderr, "no compatible complete QNN provider\n"); q = NULL; goto done;
    }
#define CALL(expr) do { fprintf(stderr, "probe: %s\n", #expr); qe = (expr); if (qe != QNN_SUCCESS) { \
    fprintf(stderr, "%s failed: 0x%llx\n", #expr, (unsigned long long)qe); goto done; } } while (0)
    CALL(q->backendCreate(NULL, NULL, &backend));
    CALL(q->deviceCreate(NULL, NULL, &device));
    CALL(q->contextCreate(backend, device, NULL, &context));
    fprintf(stderr, "probe: rpcmem allocate\n");
    error = coli_npu_buf_rpcmem(&buffer, argv[2], dims[0]);
    if (error) { fprintf(stderr, "rpcmem allocation: %s\n", strerror(error)); goto done; }
    error = coli_npu_buf_begin(buffer, 1, &data);
    if (error) { fprintf(stderr, "CPU begin: %s\n", strerror(error)); goto done; }
    memset(data, 0x5a, dims[0]);
    error = coli_npu_buf_end(buffer);
    if (error) { fprintf(stderr, "CPU end: %s\n", strerror(error)); goto done; }
    error = coli_npu_qnn_register(&registration, q, context, buffer, 0, 1, dims,
                                  QNN_DATATYPE_UFIXED_POINT_8, &qe);
    if (error) { fprintf(stderr, "registration: %s QNN=0x%llx\n", strerror(error),
                        (unsigned long long)qe); goto done; }
    rc = 0;
done:
    fprintf(stderr, "probe: cleanup (result %d)\n", rc);
    if (registration.handle && coli_npu_qnn_unregister(&registration, &qe)) {
        fprintf(stderr, "deregistration failed; retaining resources until process exit\n"); return 1;
    }
    if (coli_npu_buf_free(&buffer)) {
        fprintf(stderr, "buffer release failed; retaining resources until process exit\n"); return 1;
    }
    if (context && q->contextFree(context, NULL) != QNN_SUCCESS) return 1;
    if (device && q->deviceFree(device) != QNN_SUCCESS) return 1;
    if (backend && q->backendFree(backend) != QNN_SUCCESS) return 1;
    dlclose(library);
    if (!rc) {
        puts("PASS: QNN device/context, rpcmem allocation, cache sync, shared-buffer registration and cleanup.");
        puts("No graph executed; no inference, HMX utilization, numerical or performance claim.");
    }
    return rc;
}
