/* SDK/device test: the known INT4 fixture, through the actual model adapter.
 * Tests ordering, rejected premature reuse, repeatability and pending shutdown. */
#define _POSIX_C_SOURCE 200809L
#include "coli_npu_expert.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include COLI_HEXAGON_PROFILE
static float half(uint16_t b) {
    unsigned ex=(b>>10)&31u, f=b&1023u;
    float v=ex?ldexpf((float)(1024+f),(int)ex-25):ldexpf((float)f,-24);
    return b&32768u?-v:v;
}
static int read_file(const char *path,void *p,size_t n) {
    FILE *f=fopen(path,"rb"); int ok;
    if (!f) return 0;
    ok=fread(p,1,n,f)==n && fgetc(f)==EOF && !ferror(f); fclose(f); return ok;
}
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec*1e-6; }
int main(int argc,char **argv) {
    enum { N=16,V=3 };
    const size_t vals=(size_t)3*NP_D*NP_F, groups=vals/64;
    uint8_t *pw=malloc(N*vals/2);
    float *sc=malloc(N*groups*4), *x=malloc(V*NP_D*4), *ref=malloc(V*N*NP_D*4);
    float *y=malloc(NP_D*4), *prior=malloc(N*NP_D*4);
    uint16_t *xh=malloc(V*NP_D*2);
    ColiNpuExpert *e=NULL;
    FILE *f=NULL,*out=NULL;
    double start,mincos=1,maxl2=0;
    unsigned count=0;
    int rc=1;
    if (argc!=8 || !pw || !sc || !x || !ref || !y || !prior || !xh) return 2;
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);goto done; } } while(0)
    f=fopen(argv[4],"rb"); CHECK(f);
    for (size_t b=0;b<N*groups;++b) {
        uint8_t block[34]; CHECK(fread(block,1,34,f)==34);
        sc[b]=half((uint16_t)(block[0]|((unsigned)block[1]<<8)));
        for (unsigned k=0;k<32;++k) {
            unsigned lo=((block[2+k/2]>>((k%2)*4))&15u)^8u;
            unsigned hi=((block[18+k/2]>>((k%2)*4))&15u)^8u;
            pw[b*32+k]=(uint8_t)(lo|(hi<<4));
        }
    }
    CHECK(fgetc(f)==EOF && !ferror(f)); fclose(f); f=NULL;
    CHECK(read_file(argv[5],xh,V*NP_D*2)); CHECK(read_file(argv[6],ref,V*N*NP_D*4));
    for (unsigned i=0;i<V*NP_D;++i) x[i]=half(xh[i]);
    CHECK(coli_npu_expert_open(&e,argv[1],argv[3],argv[2],NP_D,NP_F)==0);
    out=fopen(argv[7],"wb"); CHECK(out); start=ms();
    for (unsigned v=0;v<V;++v) {
        CHECK(coli_npu_expert_prepare(e,0,pw,sc,sc+groups/3,sc+2*groups/3)==0);
        CHECK(coli_npu_expert_submit(e,0,x+v*NP_D)==0);
        for (unsigned it=0;it<2*N;++it) {
            unsigned slot=it&1u, id=it<N?it:2*N-1-it;
            CHECK(coli_npu_expert_prepare(e,slot,pw,sc,sc+groups/3,sc+2*groups/3)==EINVAL);
            if (it+1<2*N) {
                unsigned next=it+1<N?it+1:2*N-2-it;
                const float *s=sc+next*groups;
                CHECK(coli_npu_expert_prepare(e,slot^1u,pw+next*vals/2,s,s+groups/3,s+2*groups/3)==0);
                CHECK(coli_npu_expert_submit(e,slot^1u,x+v*NP_D)==EINVAL);
            }
            CHECK(coli_npu_expert_wait(e,y)==0);
            CHECK(coli_npu_expert_wait(e,y)==EINVAL);
            double rr=0,yy=0,dot=0,se=0;
            for (unsigned d=0;d<NP_D;++d) {
                double r=ref[(v*N+id)*NP_D+d],a=y[d]; rr+=r*r;yy+=a*a;dot+=r*a;se+=(a-r)*(a-r);
            }
            double cos=dot/sqrt(rr*yy),l2=sqrt(se/rr);
            CHECK(isfinite(cos) && isfinite(l2) && cos>=0.999 && l2<=0.05);
            if(cos<mincos)mincos=cos;if(l2>maxl2)maxl2=l2;
            if (it>=N) CHECK(!memcmp(prior+id*NP_D,y,NP_D*4));
            memcpy(prior+id*NP_D,y,NP_D*4);
            CHECK(fwrite(y,4,NP_D,out)==NP_D);++count;
            if (it+1<2*N) CHECK(coli_npu_expert_submit(e,slot^1u,x+v*NP_D)==0);
        }
    }
    CHECK(count==96 && coli_npu_expert_calls(e)==96);
    printf("PASS calls=%u min_cosine=%.9f max_relative_L2=%.9f pipeline_ms=%.3f\n",count,mincos,maxl2,ms()-start);
    CHECK(coli_npu_expert_prepare(e,0,pw,sc,sc+groups/3,sc+2*groups/3)==0);
    CHECK(coli_npu_expert_submit(e,0,x)==0);
    CHECK(coli_npu_expert_close(&e)==0); /* must join pending QNN before freeing */
    puts("PASS pending execution cleanup"); rc=0;
done:
    if (e && coli_npu_expert_close(&e)) rc=1;
    if(f)fclose(f);if(out && fclose(out))rc=1;
    free(pw);free(sc);free(x);free(ref);free(y);free(prior);free(xh);
    return rc;
}
