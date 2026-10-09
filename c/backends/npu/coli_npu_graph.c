#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include "coli_npu_graph.h"
#include <HTP/QnnHtpCommon.h>
#include <stdbool.h>
#include <HTP/QnnHtpDevice.h>
#include <stdio.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct ColiNpuGraph {
    void *library, *binary;
    size_t bytes;
    const QNN_INTERFACE_VER_TYPE *api;
    Qnn_BackendHandle_t backend;
    Qnn_DeviceHandle_t device;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;
    QnnHtpDevice_PerfInfrastructure_t perf;
    uint32_t power_id;
    int power_active;
};

/* Opt-in performance vote, scoped to this graph object's lifetime. The
 * vendor/Android thermal governor remains authoritative. No sysfs changes. */
static int burst_vote(ColiNpuGraph *g, Qnn_ErrorHandle_t *qe) {
    QnnDevice_Infrastructure_t infra=NULL;
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs={0};
    const QnnHtpPerfInfrastructure_PowerConfig_t *configs[]={&dcvs,NULL};
    Qnn_ErrorHandle_t e;
    if (!g->api->deviceGetInfrastructure) return ENOTSUP;
    e=g->api->deviceGetInfrastructure(&infra);
    if (e!=QNN_SUCCESS) goto failed;
    if (!infra || infra->infraType!=QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF ||
        !infra->perfInfra.createPowerConfigId || !infra->perfInfra.setPowerConfig ||
        !infra->perfInfra.destroyPowerConfigId) return ENOTSUP;
    g->perf=infra->perfInfra;
    e=g->perf.createPowerConfigId(0,0,&g->power_id);
    if (e!=QNN_SUCCESS) goto failed;
    g->power_active=1; /* ID zero is valid; use explicit ownership. */
    dcvs.option=QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvs.dcvsV3Config.contextId=g->power_id;
    dcvs.dcvsV3Config.setDcvsEnable=1;
    dcvs.dcvsV3Config.dcvsEnable=0;
    dcvs.dcvsV3Config.powerMode=QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.dcvsV3Config.setSleepLatency=1;
    dcvs.dcvsV3Config.sleepLatency=40;
    dcvs.dcvsV3Config.setBusParams=1;
    dcvs.dcvsV3Config.busVoltageCornerMin=DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.busVoltageCornerTarget=DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.busVoltageCornerMax=DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.setCoreParams=1;
    dcvs.dcvsV3Config.coreVoltageCornerMin=DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.coreVoltageCornerTarget=DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.coreVoltageCornerMax=DCVS_VOLTAGE_VCORNER_TURBO;
    e=g->perf.setPowerConfig(g->power_id,configs);
    if (e!=QNN_SUCCESS) goto failed;
    fprintf(stderr,"[Hexagon] burst vote accepted (TURBO bus/core; released on close)\n");
    return 0;
failed:
    if (qe) *qe=e;
    return EIO;
}

int coli_npu_graph_close(ColiNpuGraph **p, Qnn_ErrorHandle_t *qe) {
    ColiNpuGraph *g;
    Qnn_ErrorHandle_t e = QNN_SUCCESS;
    if (qe) *qe = QNN_SUCCESS;
    if (!p) return EINVAL;
    if (!(g = *p)) return 0;
    if (g->power_active) {
        e=g->perf.destroyPowerConfigId(g->power_id);
        if (e!=QNN_SUCCESS) goto failed;
        g->power_active=0;
    }
    if (g->context) {
        e = g->api->contextFree(g->context, NULL);
        if (e != QNN_SUCCESS) goto failed;
        g->context = NULL; g->graph = NULL;
    }
    if (g->device) {
        e = g->api->deviceFree(g->device);
        if (e != QNN_SUCCESS) goto failed;
        g->device = NULL;
    }
    if (g->backend) {
        e = g->api->backendFree(g->backend);
        if (e != QNN_SUCCESS) goto failed;
        g->backend = NULL;
    }
    if (g->binary && munmap(g->binary, g->bytes) < 0) return errno;
    if (g->library) dlclose(g->library); /* NODELETE: vendor callbacks stay mapped */
    free(g); *p = NULL;
    return 0;
failed:
    if (qe) *qe = e;
    return EIO;
}

int coli_npu_graph_open(ColiNpuGraph **out, const char *library, const char *binary,
                        const char *name, Qnn_ErrorHandle_t *qe) {
    ColiNpuGraph *g;
    Qnn_ErrorHandle_t (*providers)(const QnnInterface_t ***, uint32_t *);
    const QnnInterface_t **list = NULL;
    Qnn_ErrorHandle_t e = QNN_SUCCESS;
    struct stat st;
    void *sym;
    uint32_t count = 0, i;
    int fd, rc = EIO;
    if (qe) *qe = QNN_SUCCESS;
    if (!out || *out || !library || library[0] != '/' || !binary || !name || !*name) return EINVAL;
    fd = open(binary, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return errno;
    if (fstat(fd, &st) < 0) { rc = errno; close(fd); return rc; }
    if (!S_ISREG(st.st_mode) || st.st_size <= 0 || (uint64_t)st.st_size > SIZE_MAX) {
        close(fd); return EINVAL;
    }
    g = calloc(1, sizeof(*g));
    if (!g) { close(fd); return ENOMEM; }
    *out = g;
    g->bytes = (size_t)st.st_size;
    g->binary = mmap(NULL, g->bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    rc = errno; close(fd);
    if (g->binary == MAP_FAILED) { g->binary = NULL; goto failed; }
    rc = ENOENT;
    g->library = dlopen(library, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!g->library) goto failed;
    rc = ENOSYS;
    sym = dlsym(g->library, "QnnInterface_getProviders");
    memcpy(&providers, &sym, sizeof(providers));
    if (!providers) goto failed;
    rc = EIO;
    e = providers(&list, &count);
    if (e != QNN_SUCCESS || !list) goto failed;
    rc = ENOTSUP;
    for (i = 0; i < count; ++i) {
        const QnnInterface_t *p = list[i];
        if (!p || p->apiVersion.coreApiVersion.major != QNN_API_VERSION_MAJOR ||
            p->apiVersion.coreApiVersion.minor < QNN_API_VERSION_MINOR ||
            (p->backendId != QNN_BACKEND_ID_HTP &&
             !(p->backendId == QNN_BACKEND_ID_NULL && p->providerName &&
               !strcmp(p->providerName, QNN_HTP_INTERFACE_PROVIDER_NAME)))) continue;
        g->api = &p->QNN_INTERFACE_VER_NAME;
        break;
    }
    if (!g->api || !g->api->backendCreate || !g->api->backendFree ||
        !g->api->deviceCreate || !g->api->deviceFree || !g->api->contextCreateFromBinary ||
        !g->api->contextFree || !g->api->graphRetrieve || !g->api->graphExecute ||
        !g->api->memRegister || !g->api->memDeRegister) goto failed;
    rc = EIO;
    e = g->api->backendCreate(NULL, NULL, &g->backend);
    if (e != QNN_SUCCESS) goto failed;
    e = g->api->deviceCreate(NULL, NULL, &g->device);
    if (e != QNN_SUCCESS) goto failed;
    e = g->api->contextCreateFromBinary(g->backend, g->device, NULL, g->binary,
                                       g->bytes, &g->context, NULL);
    if (e != QNN_SUCCESS) goto failed;
    e = g->api->graphRetrieve(g->context, name, &g->graph);
    if (e != QNN_SUCCESS) goto failed;
    if (getenv("COLI_NPU_BURST_MODE") && !strcmp(getenv("COLI_NPU_BURST_MODE"),"1")) {
        rc=burst_vote(g,&e);
        if (rc) goto failed;
    }
    return 0;
failed:
    if (qe) *qe = e;
    (void)coli_npu_graph_close(out, NULL);
    return rc;
}
const QNN_INTERFACE_VER_TYPE *coli_npu_graph_api(const ColiNpuGraph *g) { return g ? g->api : NULL; }
Qnn_ContextHandle_t coli_npu_graph_context(const ColiNpuGraph *g) { return g ? g->context : NULL; }
static int registered(const Qnn_Tensor_t *t) {
    if (t->version == QNN_TENSOR_VERSION_1)
        return t->v1.memType == QNN_TENSORMEMTYPE_MEMHANDLE && t->v1.memHandle;
    if (t->version == QNN_TENSOR_VERSION_2)
        return t->v2.memType == QNN_TENSORMEMTYPE_MEMHANDLE && t->v2.memHandle;
    return 0;
}
int coli_npu_graph_execute(ColiNpuGraph *g, const Qnn_Tensor_t *in, uint32_t ni,
                           Qnn_Tensor_t *out, uint32_t no, Qnn_ErrorHandle_t *qe) {
    uint32_t i;
    Qnn_ErrorHandle_t e;
    if (qe) *qe = QNN_SUCCESS;
    if (!g || !g->graph || !in || !out || !ni || !no) return EINVAL;
    for (i = 0; i < ni; ++i) if (!registered(&in[i])) return EINVAL;
    for (i = 0; i < no; ++i) if (!registered(&out[i])) return EINVAL;
    e = g->api->graphExecute(g->graph, in, ni, out, no, NULL, NULL);
    if (qe) *qe = e;
    return e == QNN_SUCCESS ? 0 : EIO;
}
