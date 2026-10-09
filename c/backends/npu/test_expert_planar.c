/* Android SDK test: independent expected FP16 bytes are generated with Python
 * struct.pack('<e'), not with the NEON implementation under test. */
#include "coli_npu_expert.c"
static int read_file(const char *name, void *p, size_t n) {
    FILE *f=fopen(name,"rb"); int ok;
    if (!f) return 0;
    ok=fread(p,1,n,f)==n && fgetc(f)==EOF && !ferror(f);
    fclose(f); return ok;
}
int main(int argc, char **argv) {
    enum { B=4096, N=B*64 };
    uint8_t *w=malloc(N/2);
    float *s=malloc(B*sizeof(float));
    uint16_t *y=malloc(N*2), *ref=malloc(N*2);
    if (argc!=4 || !w || !s || !y || !ref) return 2;
    if (!read_file(argv[1],w,N/2) || !read_file(argv[2],s,B*4) || !read_file(argv[3],ref,N*2)) return 2;
    if (planar_half(y,w,s,N) || memcmp(y,ref,N*2)) return 1;
    s[0]=NAN; if (planar_half(y,w,s,N)!=EDOM) return 1;
    s[0]=-1; if (planar_half(y,w,s,N)!=EDOM) return 1;
    s[0]=INFINITY; if (planar_half(y,w,s,N)!=EDOM) return 1;
    s[0]=65504; memset(w,0,32); if (planar_half(y,w,s,N)!=ERANGE) return 1;
    puts("PASS: 262144 planar values, signed codes, f32 scales, FP16 rounding, nonfinite/overflow rejection");
    free(w); free(s); free(y); free(ref); return 0;
}
