/* Chunked load must reproduce whole-matrix quantization, including the tail.
 * Run in separate processes with COLI_DENSE_BITS=8 and 4 (env is cached). */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main
#include <assert.h>
int main(void) {
    enum { I=128,O=259 };
    float *w=falloc((int64_t)I*O);
    for(int i=0;i<I*O;i++) w[i]=(float)((i*17)%257-128)/31.f;
    FILE *f=tmpfile(); assert(f); assert(fwrite(w,sizeof(float),I*O,f)==I*O); fflush(f);
    st_tensor t={0}; t.name="dense";t.fd=fileno(f);t.dtype=2;t.numel=I*O;t.nbytes=I*O*4;
    Model *m=calloc(1,sizeof(*m));assert(m);m->S.t=&t;m->S.n=1;
    QW ref={0},got={0};qw_quantize(w,I,O,"lmhead",&ref);
    load_tq(m,"dense",I,O,1,"lmhead",&got);
    assert(!got.w);assert(!!ref.q==!!got.q);assert(!!ref.q4==!!got.q4);
    if(ref.q){assert(!memcmp(ref.q,got.q,I*O));assert(!memcmp(ref.sc,got.sc,O*4));}
    if(ref.q4){assert(!memcmp(ref.q4,got.q4,I*O/2));assert(!memcmp(ref.sg,got.sg,O*(I/64)*4));}
    float row[I];st_read_slice_f32(&m->S,"dense",(O-1)*I,I,row,1);
    assert(!memcmp(row,w+(O-1)*I,I*4));
    qw_free(&ref);qw_free(&got);free(m);fclose(f);free(w);
    puts("PASS chunked dense load and row read");return 0;
}
