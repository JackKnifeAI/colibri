#include "coli_npu_quant.h"
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    unsigned char buffer[258];
    static const uint16_t expected[16]={0,0x3c00,0x4000,0x4200,0x4400,0x4500,
        0x4600,0x4700,0xc800,0xc700,0xc600,0xc500,0xc400,0xc200,0xc000,0xbc00};
    unsigned i,b;
    memset(buffer,0xa5,sizeof(buffer));
    /* Two adjacent encoded blocks overlap the expanded first block. Both must
     * survive, including an intentionally unaligned mapping. */
    for(b=0;b<2;++b) {
        unsigned char *p=buffer+1+34*b;
        p[0]=0; p[1]=0x3c;
        for(i=0;i<32;++i) p[2+i]=(unsigned char)((2*i%16)|(((2*i+1)%16)<<4));
    }
    assert(coli_npu_i4_f16_expand(buffer+1,68,256,128)==0);
    assert(buffer[0]==0xa5 && buffer[257]==0xa5);
    for(i=0;i<128;++i) {
        uint16_t v=(uint16_t)(buffer[1+2*i]|((uint16_t)buffer[2+2*i]<<8));
        assert(v==expected[i%16]);
    }
    memset(buffer,0,sizeof(buffer)); buffer[0]=1; buffer[2]=0x78;
    assert(coli_npu_i4_f16_expand(buffer,34,128,64)==0);
    assert(buffer[0]==8 && buffer[1]==0x80 && buffer[2]==7 && buffer[3]==0);
    /* Halfway products round to the even FP16 significand. */
    memset(buffer,0,sizeof(buffer)); buffer[0]=0x55; buffer[1]=0x35; buffer[2]=0x63;
    assert(coli_npu_i4_f16_expand(buffer,34,128,64)==0);
    assert(buffer[0]==0 && buffer[1]==0x3c && buffer[2]==0 && buffer[3]==0x40);
    assert(coli_npu_i4_f16_expand(NULL,34,128,64)==EINVAL);
    assert(coli_npu_i4_f16_expand(buffer,34,127,64)==EINVAL);
    assert(coli_npu_i4_f16_expand(buffer,33,128,64)==EINVAL);
    assert(coli_npu_i4_f16_expand(buffer,34,128,63)==EINVAL);
    memset(buffer,0,sizeof(buffer)); buffer[1]=0x80;
    assert(coli_npu_i4_f16_expand(buffer,34,128,64)==EDOM);
    memset(buffer,0,sizeof(buffer)); buffer[1]=0x7c;
    assert(coli_npu_i4_f16_expand(buffer,34,128,64)==EDOM);
    memset(buffer,0,sizeof(buffer)); buffer[0]=0xff; buffer[1]=0x7b; buffer[2]=2;
    assert(coli_npu_i4_f16_expand(buffer,34,128,64)==ERANGE);
    puts("INT4/FP16: overlap, sign, subnormals, bounds and invalid scales PASS");
    return 0;
}
