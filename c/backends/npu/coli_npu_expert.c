#define _POSIX_C_SOURCE 200809L
#include "coli_npu_expert.h"
#include "coli_npu_graph.h"
#include "../../evidence_digest.h"
#include <arm_neon.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#ifndef COLI_HEXAGON_PROFILE
#error "Generate and specify COLI_HEXAGON_PROFILE from the qualified context metadata"
#endif
#include COLI_HEXAGON_PROFILE

struct ColiNpuExpert {
    ColiNpuGraph *graph;
    ColiNpuBuf *w[2], *x, *y;
    ColiNpuRegistration wr[2][3], xr, yr;
    Qnn_Tensor_t in[4], out;
    unsigned long long calls;
    int poisoned;
    double staging_ms, graph_ms, output_ms;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t work, done;
    int mutex_ready, work_ready, done_ready, thread_started;
    int stop, pending, completed, result, inflight;
    unsigned ready[2];
    Qnn_ErrorHandle_t qe;
    double job_ms;
};
static double expert_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC,&ts)) return 0;
    return ts.tv_sec*1000.0 + ts.tv_nsec*1e-6;
}
static void execute_job(ColiNpuExpert *e) {
    double start=expert_now_ms();
    e->result=coli_npu_graph_execute(e->graph,e->in,4,&e->out,1,&e->qe);
    e->job_ms=expert_now_ms()-start;
}
static void *expert_worker(void *p) {
    ColiNpuExpert *e=p;
    pthread_mutex_lock(&e->mutex);
    for (;;) {
        while (!e->pending && !e->stop) pthread_cond_wait(&e->work,&e->mutex);
        if (!e->pending && e->stop) break;
        pthread_mutex_unlock(&e->mutex);
        execute_job(e);
        pthread_mutex_lock(&e->mutex);
        e->pending=0; e->completed=1;
        pthread_cond_signal(&e->done);
    }
    pthread_mutex_unlock(&e->mutex);
    return NULL;
}
static int profile_check(const char *path) {
    FILE *f = fopen(path, "rb");
    long bytes;
    void *p;
    char hash[65];
    int ok;
    if (!f) return errno;
    if (fseek(f,0,SEEK_END) || (bytes=ftell(f))<=0 || bytes>64*1024*1024 || fseek(f,0,SEEK_SET)) {
        fclose(f); return EINVAL;
    }
    p=malloc((size_t)bytes);
    if (!p) { fclose(f); return ENOMEM; }
    ok=fread(p,1,(size_t)bytes,f)==(size_t)bytes && fgetc(f)==EOF && !ferror(f);
    fclose(f);
    if (!ok) { free(p); return EIO; }
    evidence_sha256_hex(p,(size_t)bytes,hash); free(p);
    return strcmp(hash,np_sha256) ? EINVAL : 0;
}
static void tensor(Qnn_Tensor_t *t, unsigned i) {
    t->version=QNN_TENSOR_VERSION_1;
    t->v1.id=np_ids[i]; t->v1.name=np_names[i];
    t->v1.type=i==4 ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_APP_WRITE;
    t->v1.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t->v1.dataType=QNN_DATATYPE_FLOAT_16;
    t->v1.rank=2; t->v1.dimensions=np_dims[i];
    t->v1.quantizeParams.encodingDefinition=QNN_DEFINITION_UNDEFINED;
    t->v1.quantizeParams.quantizationEncoding=QNN_QUANTIZATION_ENCODING_UNDEFINED;
}
int coli_npu_expert_close(ColiNpuExpert **p) {
    ColiNpuExpert *e;
    Qnn_ErrorHandle_t qe=0;
    unsigned i,k;
    int rc;
    if (!p) return EINVAL;
    if (!(e=*p)) return 0;
    if (e->thread_started) {
        pthread_mutex_lock(&e->mutex); e->stop=1;
        pthread_cond_signal(&e->work); pthread_mutex_unlock(&e->mutex);
        if ((rc=pthread_join(e->thread,NULL))) return rc;
        e->thread_started=0;
    }
    if (e->done_ready) { pthread_cond_destroy(&e->done); e->done_ready=0; }
    if (e->work_ready) { pthread_cond_destroy(&e->work); e->work_ready=0; }
    if (e->mutex_ready) { pthread_mutex_destroy(&e->mutex); e->mutex_ready=0; }
    for (i=0;i<2;++i) {
        for (k=0;k<3;++k) if ((rc=coli_npu_qnn_unregister(&e->wr[i][k],&qe))) return rc;
        if ((rc=coli_npu_buf_free(&e->w[i]))) return rc;
    }
    if ((rc=coli_npu_qnn_unregister(&e->xr,&qe)) ||
        (rc=coli_npu_qnn_unregister(&e->yr,&qe)) ||
        (rc=coli_npu_buf_free(&e->x)) || (rc=coli_npu_buf_free(&e->y)) ||
        (rc=coli_npu_graph_close(&e->graph,&qe))) return rc;
    free(e); *p=NULL; return 0;
}
int coli_npu_expert_open(ColiNpuExpert **out, const char *backend,
                        const char *context, const char *rpcmem, int d, int f) {
    ColiNpuExpert *e;
    Qnn_ErrorHandle_t qe=0;
    unsigned i,k;
    int rc;
    size_t mat=(size_t)NP_D*NP_F*2;
    if (!out || *out || !backend || !context || !rpcmem || d!=NP_D || f!=NP_F) return EINVAL;
    if ((rc=profile_check(context))) return rc;
    e=calloc(1,sizeof(*e)); if (!e) return ENOMEM;
    *out=e; e->inflight=-1;
#define TRY(call) do { if ((rc=(call))) goto failed; } while(0)
    TRY(coli_npu_graph_open(&e->graph,backend,context,np_graph,&qe));
    TRY(coli_npu_buf_rpcmem(&e->x,rpcmem,NP_D*2));
    TRY(coli_npu_buf_rpcmem(&e->y,rpcmem,NP_D*2));
    for (i=0;i<4;++i) tensor(&e->in[i],i);
    tensor(&e->out,4);
    TRY(coli_npu_qnn_register(&e->xr,coli_npu_graph_api(e->graph),coli_npu_graph_context(e->graph),e->x,0,2,np_dims[NP_X],QNN_DATATYPE_FLOAT_16,&qe));
    TRY(coli_npu_qnn_register(&e->yr,coli_npu_graph_api(e->graph),coli_npu_graph_context(e->graph),e->y,0,2,np_dims[4],QNN_DATATYPE_FLOAT_16,&qe));
    TRY(coli_npu_qnn_bind(&e->in[NP_X],&e->xr));
    TRY(coli_npu_qnn_bind(&e->out,&e->yr));
    for (i=0;i<2;++i) {
        TRY(coli_npu_buf_rpcmem(&e->w[i],rpcmem,3*mat));
        for (k=0;k<3;++k) TRY(coli_npu_qnn_register(&e->wr[i][k],coli_npu_graph_api(e->graph),coli_npu_graph_context(e->graph),e->w[i],k*mat,2,np_dims[np_widx[k]],QNN_DATATYPE_FLOAT_16,&qe));
    }
    if (getenv("COLI_NPU_OVERLAP") && !strcmp(getenv("COLI_NPU_OVERLAP"),"1")) {
        TRY(pthread_mutex_init(&e->mutex,NULL)); e->mutex_ready=1;
        TRY(pthread_cond_init(&e->work,NULL)); e->work_ready=1;
        TRY(pthread_cond_init(&e->done,NULL)); e->done_ready=1;
        TRY(pthread_create(&e->thread,NULL,expert_worker,e)); e->thread_started=1;
        fprintf(stderr,"[Hexagon] QNN worker enabled; two-slot weight staging overlap\n");
    }
    return 0;
failed:
    fprintf(stderr,"[Hexagon] open errno=%d QNN=%llu\n",rc,(unsigned long long)qe);
    (void)coli_npu_expert_close(out); /* partial object retained if vendor cleanup fails */
    return rc;
#undef TRY
}
/* NEON conversion uses IEEE round-to-nearest-even. Retain the float32 scales
 * from the real container; do not round them to the earlier fixture format. */
static int planar_half(uint16_t *dst, const uint8_t *src, const float *sc, size_t n) {
    size_t b;
    uint16x8_t maximum=vdupq_n_u16(0);
    for (b=0;b<n/64;++b) {
        unsigned j,h;
        float32x4_t scale;
        if (!isfinite(sc[b]) || sc[b]<0) return EDOM;
        scale=vdupq_n_f32(sc[b]);
        for (j=0;j<32;j+=8) {
            uint8x8_t raw=vld1_u8(src+b*32+j);
            uint8x8_t parts[2]={vand_u8(raw,vdup_n_u8(15)),vshr_n_u8(raw,4)};
            for (h=0;h<2;++h) {
                int16x8_t q=vmovl_s8(vsub_s8(vreinterpret_s8_u8(parts[h]),vdup_n_s8(8)));
                float32x4_t lo=vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(q))),scale);
                float32x4_t hi=vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(q))),scale);
                uint16x8_t half=vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo),vcvt_f16_f32(hi)));
                maximum=vmaxq_u16(maximum,vandq_u16(half,vdupq_n_u16(32767)));
                vst1q_u16(dst+b*64+j+h*32,half);
            }
        }
    }
    return vmaxvq_u16(maximum)>=0x7c00u ? ERANGE : 0;
}
static int expert_failed(ColiNpuExpert *e, int rc, Qnn_ErrorHandle_t qe) {
    e->poisoned=1;
    fprintf(stderr,"[Hexagon] execute errno=%d QNN=%llu\n",rc,(unsigned long long)qe);
    return rc;
}
int coli_npu_expert_prepare(ColiNpuExpert *e, unsigned slot, const uint8_t *pw,
                           const float *gs, const float *us, const float *ds) {
    uint16_t *p;
    const float *sc[3]={gs,us,ds};
    size_t n=(size_t)NP_D*NP_F;
    int rc=0,end_rc;
    double start;
    if (!e || e->poisoned || slot>1 || e->ready[slot] ||
        !pw || !gs || !us || !ds) return EINVAL;
    start=expert_now_ms();
    if ((rc=coli_npu_buf_begin(e->w[slot],1,(void **)&p))) return expert_failed(e,rc,0);
    for (unsigned k=0;k<3 && !rc;++k) rc=planar_half(p+k*n,pw+k*n/2,sc[k],n);
    end_rc=coli_npu_buf_end(e->w[slot]); if (!rc) rc=end_rc;
    if (rc) return expert_failed(e,rc,0);
    e->staging_ms+=expert_now_ms()-start;
    e->ready[slot]=1;
    return 0;
}
int coli_npu_expert_submit(ColiNpuExpert *e, unsigned slot, const float *x) {
    uint16_t *p;
    int rc=0,end_rc;
    double start;
    if (!e || e->poisoned || !x || slot>1 || e->ready[slot]!=1 || e->inflight!=-1) return EINVAL;
    start=expert_now_ms();
    if ((rc=coli_npu_buf_begin(e->x,1,(void **)&p))) return expert_failed(e,rc,0);
    for (unsigned k=0;k<NP_D;k+=4) {
        uint16x4_t bits=vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(x+k)));
        if (vmaxv_u16(vand_u16(bits,vdup_n_u16(32767)))>=0x7c00u) rc=ERANGE;
        vst1_u16(p+k,bits);
    }
    end_rc=coli_npu_buf_end(e->x); if (!rc) rc=end_rc;
    if (rc) return expert_failed(e,rc,0);
    for (unsigned k=0;k<3;++k)
        if ((rc=coli_npu_qnn_bind(&e->in[np_widx[k]],&e->wr[slot][k]))) return expert_failed(e,rc,0);
    e->staging_ms+=expert_now_ms()-start;
    e->inflight=(int)slot; e->ready[slot]=2;
    if (e->thread_started) {
        pthread_mutex_lock(&e->mutex); e->completed=0; e->pending=1;
        pthread_cond_signal(&e->work); pthread_mutex_unlock(&e->mutex);
    } else { execute_job(e); e->completed=1; }
    return 0;
}
int coli_npu_expert_wait(ColiNpuExpert *e, float *y) {
    uint16_t *p;
    int rc=0,end_rc;
    double start;
    if (!e || e->poisoned || !y || e->inflight<0) return EINVAL;
    if (e->thread_started) {
        pthread_mutex_lock(&e->mutex);
        while (!e->completed) pthread_cond_wait(&e->done,&e->mutex);
        pthread_mutex_unlock(&e->mutex);
    }
    if (e->result) return expert_failed(e,e->result,e->qe);
    start=expert_now_ms();
    if ((rc=coli_npu_buf_begin(e->y,0,(void **)&p))) return expert_failed(e,rc,0);
    for (unsigned k=0;k<NP_D;k+=4) {
        uint16x4_t bits=vld1_u16(p+k);
        if (vmaxv_u16(vand_u16(bits,vdup_n_u16(32767)))>=0x7c00u) rc=ERANGE;
        vst1q_f32(y+k,vcvt_f32_f16(vreinterpret_f16_u16(bits)));
    }
    end_rc=coli_npu_buf_end(e->y); if (!rc) rc=end_rc;
    if (rc) return expert_failed(e,rc,0);
    e->output_ms+=expert_now_ms()-start; e->graph_ms+=e->job_ms;
    e->ready[e->inflight]=0; e->inflight=-1; ++e->calls;
    return 0;
}
int coli_npu_expert_run(ColiNpuExpert *e, const float *x, const uint8_t *pw,
                       const float *gs, const float *us, const float *ds, float *y) {
    unsigned slot;
    int rc;
    if (!e || e->inflight!=-1) return EINVAL;
    slot=(unsigned)(e->calls&1u);
    rc=coli_npu_expert_prepare(e,slot,pw,gs,us,ds);
    if (!rc) rc=coli_npu_expert_submit(e,slot,x);
    if (!rc) rc=coli_npu_expert_wait(e,y);
    return rc;
}
unsigned long long coli_npu_expert_calls(const ColiNpuExpert *e) { return e ? e->calls : 0; }

void coli_npu_expert_stats(const ColiNpuExpert *e, ColiNpuExpertStats *s) {
    if (!s) return;
    memset(s,0,sizeof(*s));
    if (e) { s->calls=e->calls; s->staging_ms=e->staging_ms;
             s->graph_ms=e->graph_ms; s->output_ms=e->output_ms; }
}
