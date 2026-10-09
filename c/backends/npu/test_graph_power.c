/* SDK-header-only host test. All vendor calls are mocks. */
#include "coli_npu_graph.c"
#include <assert.h>
static int created, configured, destroyed, fail_set, fail_destroy;
static Qnn_ErrorHandle_t create_id(uint32_t device, uint32_t core, uint32_t *id) {
    assert(!device && !core); *id=0; ++created; return QNN_SUCCESS;
}
static Qnn_ErrorHandle_t set_config(uint32_t id, const QnnHtpPerfInfrastructure_PowerConfig_t **c) {
    assert(id==0 && c && c[0] && !c[1]);
    assert(c[0]->option==QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3);
    assert(c[0]->dcvsV3Config.contextId==id);
    assert(c[0]->dcvsV3Config.coreVoltageCornerTarget==DCVS_VOLTAGE_VCORNER_TURBO);
    assert(c[0]->dcvsV3Config.setSleepDisable==0);
    ++configured; return fail_set ? 1 : QNN_SUCCESS;
}
static Qnn_ErrorHandle_t destroy_id(uint32_t id) {
    assert(id==0); ++destroyed; return fail_destroy ? 1 : QNN_SUCCESS;
}
static Qnn_ErrorHandle_t infrastructure(const QnnDevice_Infrastructure_t *out) {
    static QnnHtpDevice_Infrastructure_t i;
    i.infraType=QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF;
    i.perfInfra.createPowerConfigId=create_id;
    i.perfInfra.setPowerConfig=set_config;
    i.perfInfra.destroyPowerConfigId=destroy_id;
    *(QnnDevice_Infrastructure_t *)out=&i;
    return QNN_SUCCESS;
}
int main(void) {
    QNN_INTERFACE_VER_TYPE api={0}; Qnn_ErrorHandle_t qe=0;
    ColiNpuGraph *g=calloc(1,sizeof(*g)); assert(g); g->api=&api;
    assert(burst_vote(g,&qe)==ENOTSUP);
    api.deviceGetInfrastructure=infrastructure;
    assert(burst_vote(g,&qe)==0 && g->power_active);
    fail_destroy=1;
    assert(coli_npu_graph_close(&g,&qe)==EIO && g && g->power_active);
    fail_destroy=0;
    assert(coli_npu_graph_close(&g,&qe)==0 && !g);
    g=calloc(1,sizeof(*g)); assert(g); g->api=&api; fail_set=1;
    assert(burst_vote(g,&qe)==EIO && g->power_active);
    assert(coli_npu_graph_close(&g,&qe)==0 && !g);
    assert(created==2 && configured==2 && destroyed==3);
    puts("PASS: power ID zero, absent API, failed vote cleanup, failed destroy retention/retry");
    return 0;
}
