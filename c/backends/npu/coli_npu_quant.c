#include "coli_npu_quant.h"
#include <errno.h>
#include <stdint.h>
#include <string.h>
#if defined(__aarch64__) && defined(__ARM_NEON) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ && !defined(COLI_NPU_QUANT_SCALAR)
#include <arm_neon.h>
#define COLI_QUANT_NEON 1
#else
#define COLI_QUANT_NEON 0
#endif

#if COLI_QUANT_NEON
static uint16x8_t multiply8(int8x8_t codes, float32x4_t scale) {
    int16x8_t wide=vmovl_s8(codes);
    float32x4_t a=vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide)));
    float32x4_t b=vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide)));
    return vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vmulq_f32(a,scale)),
                                             vcvt_f16_f32(vmulq_f32(b,scale))));
}
static int expand_neon(uint8_t *dst, const uint8_t block[34], uint16_t bits) {
    unsigned exp=(bits>>10)&31u, frac=bits&1023u, i;
    float scale;
    float32x4_t scales;
    uint16x8_t maximum=vdupq_n_u16(0), mask=vdupq_n_u16(32767);
    if ((bits&32768u) || exp==31u) return EDOM;
    if (!bits) { memset(dst,0,128); return 0; }
    if (!exp) scale=(float)frac*0x1p-24f;
    else {
        uint32_t single=((exp+112u)<<23)|(frac<<13);
        memcpy(&scale,&single,sizeof(scale));
    }
    scales=vdupq_n_f32(scale);
    for(i=0;i<32;i+=16) {
        uint8x16_t raw=vld1q_u8(block+2+i);
        uint8x16_t lo=vandq_u8(raw,vdupq_n_u8(15)), hi=vshrq_n_u8(raw,4);
        uint8x16_t pairs[2]={vzip1q_u8(lo,hi),vzip2q_u8(lo,hi)};
        unsigned k;
        for(k=0;k<2;++k) {
            int8x16_t signed_codes=vsubq_s8(vreinterpretq_s8_u8(veorq_u8(pairs[k],vdupq_n_u8(8))),vdupq_n_s8(8));
            uint16x8_t a=multiply8(vget_low_s8(signed_codes),scales);
            uint16x8_t b=multiply8(vget_high_s8(signed_codes),scales);
            maximum=vmaxq_u16(maximum,vmaxq_u16(vandq_u16(a,mask),vandq_u16(b,mask)));
            vst1q_u8(dst+i*4+k*32,vreinterpretq_u8_u16(a));
            vst1q_u8(dst+i*4+k*32+16,vreinterpretq_u8_u16(b));
        }
    }
    return vmaxvq_u16(maximum)>=0x7c00u ? ERANGE : 0;
}
#else

/* Integer-only multiplication of a positive half by an INT4 value. A half
 * significand times |q| has at most 14 bits; only the last three can round. */
static int product(uint16_t scale, unsigned nibble, uint16_t *out) {
    unsigned exp=(scale>>10)&31u, frac=scale&1023u;
    unsigned sign=nibble&8u, magnitude=sign ? 16u-nibble : nibble;
    unsigned n, shift, mantissa, rem, halfway;
    if ((scale&32768u) || exp==31u) return EDOM;
    if (!magnitude || !(scale&32767u)) { *out=0; return 0; }
    n=(exp ? 1024u+frac : frac)*magnitude;
    if (!exp && n<1024u) { *out=(uint16_t)(n|(sign?32768u:0u)); return 0; }
    shift=n>=8192u ? 3u : n>=4096u ? 2u : n>=2048u ? 1u : 0u;
    mantissa=n>>shift;
    if (shift) {
        rem=n&((1u<<shift)-1u); halfway=1u<<(shift-1u);
        if (rem>halfway || (rem==halfway && (mantissa&1u))) ++mantissa;
    }
    exp=(exp?exp:1u)+shift;
    if (mantissa==2048u) { mantissa=1024u; ++exp; }
    if (exp>=31u) return ERANGE;
    *out=(uint16_t)((sign?32768u:0u)|(exp<<10)|(mantissa-1024u));
    return 0;
}
#endif

int coli_npu_i4_f16_expand(void *buffer, size_t encoded_bytes,
                          size_t capacity, size_t values) {
    uint8_t *bytes=buffer;
    size_t blocks, b;
    if (!buffer || !values || values%64u) return EINVAL;
    if (values>SIZE_MAX/2u) return EOVERFLOW;
    blocks=values/64u;
    if (encoded_bytes!=blocks*34u || capacity<values*2u) return EINVAL;
    for (b=blocks; b>0; --b) {
        uint8_t block[34];
        uint16_t scale;
#if !COLI_QUANT_NEON
        uint16_t table[16];
        int errors[16];
        unsigned i;
#endif
        /* Small block scratch keeps reads safe where input/output overlap. */
        memcpy(block,bytes+(b-1u)*34u,sizeof(block));
        scale=(uint16_t)(block[0]|((uint16_t)block[1]<<8));
#if COLI_QUANT_NEON
        {
            int e=expand_neon(bytes+(b-1u)*128u,block,scale);
            if (e) return e;
        }
#else
        for (i=0;i<16u;++i) errors[i]=product(scale,i,&table[i]);
        for (i=0;i<64u;++i) {
            unsigned nibble=(block[2u+i/2u]>>((i%2u)*4u))&15u;
            uint16_t half;
            size_t pos=((b-1u)*64u+i)*2u;
            if (errors[nibble]) return errors[nibble];
            half=table[nibble];
            bytes[pos]=(uint8_t)half; bytes[pos+1u]=(uint8_t)(half>>8);
        }
#endif
    }
    return 0;
}
