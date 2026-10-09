#!/usr/bin/env python3
"""Experimental Qwen35MoE GGUF -> Colibri text container, bounded-memory.

Explicit --v-head-order is required: GGUF does not version this layout.
Modern llama.cpp exports tiled V heads; HF/Colibri use grouped V heads.
Only F32/F16/Q8_0 inputs are qualified here. Experts become signed int4 gs64;
this is requantization, not a lossless conversion. MTP is explicitly excluded.
No model weights are downloaded. Only numpy is required at conversion time.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from gguf_reader import GGUFReader, ggml_block_spec
from gguf_dequant import dequantize

MAP = {
    'attn_norm.weight': 'input_layernorm.weight',
    'post_attention_norm.weight': 'post_attention_layernorm.weight',
    'attn_q.weight': 'self_attn.q_proj.weight',
    'attn_k.weight': 'self_attn.k_proj.weight',
    'attn_v.weight': 'self_attn.v_proj.weight',
    'attn_output.weight': 'self_attn.o_proj.weight',
    'attn_q_norm.weight': 'self_attn.q_norm.weight',
    'attn_k_norm.weight': 'self_attn.k_norm.weight',
    'ffn_gate_inp.weight': 'mlp.gate.weight',
    'ffn_gate_inp_shexp.weight': 'mlp.shared_expert_gate.weight',
    'ffn_gate_shexp.weight': 'mlp.shared_expert.gate_proj.weight',
    'ffn_up_shexp.weight': 'mlp.shared_expert.up_proj.weight',
    'ffn_down_shexp.weight': 'mlp.shared_expert.down_proj.weight',
    'attn_qkv.weight': 'linear_attn.in_proj_qkv.weight',
    'attn_gate.weight': 'linear_attn.in_proj_z.weight',
    'ssm_a': 'linear_attn.A_log',
    'ssm_alpha.weight': 'linear_attn.in_proj_a.weight',
    'ssm_beta.weight': 'linear_attn.in_proj_b.weight',
    'ssm_conv1d.weight': 'linear_attn.conv1d.weight',
    'ssm_dt.bias': 'linear_attn.dt_bias',
    'ssm_norm.weight': 'linear_attn.norm.weight',
    'ssm_out.weight': 'linear_attn.out_proj.weight',
}

class Writer:
    """Reserve a bounded header, append tensors, publish atomically on success."""
    def __init__(self, path):
        self.path = Path(path); self.tmp = self.path.with_suffix('.partial')
        self.f = open(self.tmp, 'w+b'); self.reserve = 128 * 1024
        self.f.seek(8 + self.reserve); self.header = {}; self.offset = 0
    def add(self, name, data):
        a = np.ascontiguousarray(data)
        if a.dtype.kind == 'f' and not np.all(np.isfinite(a)): raise ValueError('nonfinite tensor '+name)
        code = {np.dtype('<f2'): 'F16', np.dtype('<f4'): 'F32', np.dtype('uint8'): 'U8'}[a.dtype]
        if name in self.header: raise ValueError('duplicate tensor '+name)
        self.header[name] = dict(dtype=code, shape=list(a.shape), data_offsets=[self.offset, self.offset+a.nbytes])
        a.tofile(self.f); self.offset += a.nbytes
    def chunks(self, name, shape, chunks):
        start = self.offset
        for a in chunks:
            a = np.ascontiguousarray(a, dtype='<f2')
            if not np.all(np.isfinite(a)): raise ValueError('nonfinite chunk '+name)
            a.tofile(self.f); self.offset += a.nbytes
        if self.offset-start != math.prod(shape)*2: raise ValueError('chunk size mismatch')
        self.header[name] = dict(dtype='F16', shape=list(shape), data_offsets=[start,self.offset])
    def finish(self):
        h = json.dumps(self.header, separators=(',',':')).encode()
        if len(h)>self.reserve: raise ValueError('header reserve exceeded')
        self.f.seek(0); self.f.write(self.reserve.to_bytes(8,'little')); self.f.write(h.ljust(self.reserve,b' '))
        self.f.flush(); os.fsync(self.f.fileno()); self.f.close(); os.replace(self.tmp,self.path)

def slice_tensor(r, t, start=0, count=None):
    count = t['numel']-start if count is None else count
    block, size = ggml_block_spec(t['ggml_type'])
    if start<0 or count<0 or start+count>t['numel'] or start%block or count%block:
        raise ValueError('invalid tensor slice')
    r.f.seek(t['abs_offset']+start//block*size)
    raw = r.f.read(count//block*size)
    return dequantize(raw,t['ggml_type'],count)

def inverse_heads(a, axis, nk, nv, hd):
    shape=list(a.shape)
    if shape[axis] != nv*hd or nv%nk: raise ValueError('invalid V-head geometry')
    sh=shape[:axis]+[nv//nk,nk,hd]+shape[axis+1:]
    return a.reshape(sh).swapaxes(axis,axis+1).reshape(shape).copy()

def restore_dense(name, a, nk, nv, kd, vd, tiled):
    if name.endswith('norm.weight') and name != 'ssm_norm.weight': a=a-1
    if name=='ssm_a':
        if not np.all(a<0): raise ValueError('ssm_a must encode negative exp(A_log)')
        a=np.log(-a)
    if tiled:
        if name in ('attn_qkv.weight','ssm_conv1d.weight'):
            qk=2*nk*kd; a=np.concatenate((a[:qk],inverse_heads(a[qk:],0,nk,nv,vd)))
        elif name=='attn_gate.weight': a=inverse_heads(a,0,nk,nv,vd)
        elif name in ('ssm_alpha.weight','ssm_beta.weight','ssm_a','ssm_dt.bias'): a=inverse_heads(a,0,nk,nv,1)
        elif name=='ssm_out.weight': a=inverse_heads(a,1,nk,nv,vd)
    return a

def pack_expert(matrices):
    packed=[]; scales=[]
    for a in matrices:
        if not np.all(np.isfinite(a)): raise ValueError('nonfinite expert weights')
        g=a.reshape(a.shape[0],-1,64)
        sc=np.maximum(np.max(np.abs(g),axis=2),np.float32(1e-12))/np.float32(7)
        q=np.clip(np.rint(g/sc[:,:,None]),-8,7).astype(np.int8).ravel()
        packed.append(((q[0::2].astype(np.uint8)&15)|((q[1::2].astype(np.uint8)&15)<<4)))
        scales.append(sc.ravel())
    return np.concatenate(packed),np.concatenate(scales).astype('<f4')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source'); ap.add_argument('--out',required=True)
    ap.add_argument('--v-head-order',required=True,choices=['tiled','grouped'])
    args=ap.parse_args(); out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('output must be empty (no silent resume of incompatible weights)')
    with GGUFReader(args.source) as r:
        m=r.metadata
        if m['general.architecture']!='qwen35moe': raise ValueError('only qwen35moe supported')
        def get(k): return m['qwen35moe.'+k]
        nl=get('block_count')-m.get('qwen35moe.nextn_predict_layers',0)
        h=get('embedding_length'); ne=get('expert_count'); inter=get('expert_feed_forward_length')
        nk=get('ssm.group_count'); nv=get('ssm.time_step_rank'); kd=get('ssm.state_size'); vd=get('ssm.inner_size')//nv
        tokens=m['tokenizer.ggml.tokens']; types=m['tokenizer.ggml.token_type']
        vocab=len(tokens); interval=get('full_attention_interval')
        layers=['full_attention' if (i+1)%interval==0 else 'linear_attention' for i in range(nl)]
        if h%64 or inter%64 or len(types)!=vocab: raise ValueError('unsupported geometry')
        expected={'token_embd.weight','output.weight','output_norm.weight'}
        for i in range(nl):
            suffixes=set(MAP)-({'attn_qkv.weight','attn_gate.weight','ssm_a','ssm_alpha.weight','ssm_beta.weight','ssm_conv1d.weight','ssm_dt.bias','ssm_norm.weight','ssm_out.weight'} if layers[i]=='full_attention' else {'attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight','attn_q_norm.weight','attn_k_norm.weight'})
            expected.update(f'blk.{i}.{n}' for n in suffixes)
            expected.update(f'blk.{i}.ffn_{p}_exps.weight' for p in ('gate','up','down'))
        names={t['name'] for t in r.tensors}
        mtp={n for n in names if n.startswith(tuple(f'blk.{i}.' for i in range(nl,get('block_count'))))}
        if names-mtp != expected: raise ValueError(f'tensor inventory mismatch missing={expected-names}, unknown={names-mtp-expected}')
        for t in r.tensors:
            if t['ggml_type'] not in (0,1,8): raise ValueError('unqualified input type '+t['type_name'])
        cfg=dict(model_type='qwen3_5_moe_text',hidden_size=h,vocab_size=vocab,num_hidden_layers=nl,
                 num_experts=ne,num_experts_per_tok=get('expert_used_count'),moe_intermediate_size=inter,
                 shared_expert_intermediate_size=get('expert_shared_feed_forward_length'),
                 num_attention_heads=get('attention.head_count'),num_key_value_heads=get('attention.head_count_kv'),
                 head_dim=get('attention.key_length'),rms_norm_eps=get('attention.layer_norm_rms_epsilon'),
                 layer_types=layers,linear_num_key_heads=nk,linear_num_value_heads=nv,linear_key_head_dim=kd,
                 linear_value_head_dim=vd,linear_conv_kernel_dim=get('ssm.conv_kernel'),norm_topk_prob=True,
                 partial_rotary_factor=get('rope.dimension_count')/get('attention.key_length'),
                 rope_parameters=dict(rope_theta=get('rope.freq_base'),mrope_section=get('rope.dimension_sections')[:3]),
                 eos_token_id=m.get('tokenizer.ggml.eos_token_id'))
        meta=dict(model_type=cfg['model_type'],hidden=h,n_layers=nl,n_active=nl,layer_types=layers,num_experts=ne,
                  topk=cfg['num_experts_per_tok'],moe_inter=inter,shared_inter=cfg['shared_expert_intermediate_size'],
                  rms_eps=cfg['rms_norm_eps'],ebits=4,expert_gs=64,scoring_func='softmax',n_group=1,topk_group=1,
                  norm_topk_prob=True,attn_output_gate=True,q_heads=cfg['num_attention_heads'],kv_heads=cfg['num_key_value_heads'],
                  q_head_dim=2*cfg['head_dim'],k_head_dim=cfg['head_dim'],v_head_dim=get('attention.value_length'),
                  o_in=cfg['num_attention_heads']*get('attention.value_length'),qk_rope_head_dim=cfg['head_dim'],
                  head_dim=cfg['head_dim'],rope_dim=cfg['head_dim'],rope_theta=get('rope.freq_base'),
                  partial_rotary_factor=cfg['partial_rotary_factor'],mrope_section=get('rope.dimension_sections')[:3],
                  dn_vheads=nv,dn_kheads=nk,dn_kdim=kd,dn_vdim=vd,dn_convk=get('ssm.conv_kernel'),dn_conv_dim=2*nk*kd+nv*vd)
        print(f'Converting {m.get("general.name")}: {nl} layers, excluding {len(mtp)} MTP tensors',flush=True)
        for k,v in [('config.json',cfg),('qwen36_meta.json',meta)]: (out/k).write_text(json.dumps(v,indent=2)+'\n')
        if len(set(tokens))!=vocab: raise ValueError('duplicate vocab entries require explicit handling')
        tok=dict(model=dict(type='BPE',vocab=dict(zip(tokens,range(vocab))),merges=m['tokenizer.ggml.merges']),
                 added_tokens=[dict(id=i,content=t,special=True,single_word=False,lstrip=False,rstrip=False,normalized=False) for i,(t,typ) in enumerate(zip(tokens,types)) if typ in (3,4)])
        (out/'tokenizer.json').write_text(json.dumps(tok,ensure_ascii=False))
        if 'tokenizer.chat_template' in m: (out/'chat_template.jinja').write_text(m['tokenizer.chat_template'])
        w=Writer(out/'model-globals.safetensors')
        for src,dst in [('token_embd.weight','model.embed_tokens.weight'),('output.weight','lm_head.weight')]:
            t=r.tensor(src)
            if t['dims'] != [h,vocab]: raise ValueError('global shape mismatch')
            w.chunks(dst,(vocab,h),(slice_tensor(r,t,start,min(128*h,t['numel']-start)) for start in range(0,t['numel'],128*h)))
        w.add('model.norm.weight',(slice_tensor(r,r.tensor('output_norm.weight'))-1).astype('<f4'));w.finish()
        print('Globals complete',flush=True)
        for layer in range(nl):
            w=Writer(out/f'model-{layer:05d}.safetensors'); prefix=f'blk.{layer}.'
            for t in r.tensors:
                if t['name'].startswith(prefix) and t['name'][len(prefix):] in MAP:
                    short=t['name'][len(prefix):]
                    a=slice_tensor(r,t).reshape(tuple(reversed(t['dims'])))
                    a=restore_dense(short,a,nk,nv,kd,vd,args.v_head_order=='tiled')
                    # Norms/log-A remain F32 to preserve the inverse transform.
                    w.add(f'model.layers.{layer}.'+MAP[short],a.astype('<f4' if a.ndim==1 else '<f2'))
            ets=[r.tensor(prefix+f'ffn_{p}_exps.weight') for p in ('gate','up','down')]
            for t,shape in zip(ets,([h,inter,ne],[h,inter,ne],[inter,h,ne])):
                if t['dims']!=shape: raise ValueError('expert geometry mismatch')
            for e in range(ne):
                a=[slice_tensor(r,t,e*h*inter,h*inter).reshape(tuple(reversed(t['dims'][:2]))) for t in ets]
                mw,sc=pack_expert(a); base=f'model.layers.{layer}.mlp.experts.{e}'
                w.add(base+'.merged_weight',mw);w.add(base+'.qs',sc)
            w.finish();print(f'Layer {layer+1}/{nl} complete',flush=True)
        print('Hashing source and completed shards',flush=True)
        def sha(path):
            hh=hashlib.sha256()
            with open(path,'rb') as f:
                for b in iter(lambda:f.read(8<<20),b''): hh.update(b)
            return hh.hexdigest()
        manifest=dict(source=Path(args.source).name,source_bytes=r.file_size,source_sha256=sha(args.source),
                      source_name=m.get('general.name'),v_head_order=args.v_head_order,excluded_mtp_tensors=sorted(mtp),
                      expert_quantization='signed int4 gs64 from dequantized Q8_0; lossy',
                      files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()})
        (out/'conversion-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('DONE',flush=True)
if __name__=='__main__': main()
