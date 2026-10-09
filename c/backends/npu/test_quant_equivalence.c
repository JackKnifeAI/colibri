/* On AArch64 compare NEON with the integer-only reference, including every
 * FP16 scale pattern and every INT4 code. Other hosts check scalar consistency. */
#include "coli_npu_quant.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

int coli_npu_i4_f16_expand_scalar(void *,size_t,size_t,size_t);
int main(void) {
    uint8_t a[128],b[128];
    unsigned scale,i;
    for(scale=0;scale<65536u;++scale) {
        int ra,rb;
        memset(a,0,sizeof(a));
        a[0]=(uint8_t)scale; a[1]=(uint8_t)(scale>>8);
        for(i=0;i<32;++i) a[i+2]=(uint8_t)((2*i%16)|(((2*i+1)%16)<<4));
        memcpy(b,a,sizeof(a));
        ra=coli_npu_i4_f16_expand(a,34,128,64);
        rb=coli_npu_i4_f16_expand_scalar(b,34,128,64);
        if(ra!=rb || (!ra && memcmp(a,b,sizeof(a)))) {
            fprintf(stderr,"INT4 expansion mismatch scale=%04x optimized=%d scalar=%d\n",scale,ra,rb);
            return 1;
        }
    }
    puts("INT4 expansion equivalence: 65536 scale patterns x 16 signed codes PASS");
    return 0;
}
