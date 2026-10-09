import sys
import tempfile
import unittest
import json
import struct
import subprocess
import hashlib
from pathlib import Path
try:
    import numpy as np
except ImportError as exc:
    raise unittest.SkipTest(str(exc))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from convert_gguf_to_qwen36 import inverse_heads,restore_dense,pack_expert,Writer

from gguf_fixture import _encode_string, _encode_value

def tiny_gguf(path, extra=False):
    """Independent GGUF serialization, with known HF norm and head values."""
    md={'general.architecture':'qwen35moe','general.name':'synthetic test',
        'qwen35moe.block_count':2,'qwen35moe.context_length':512,
        'qwen35moe.embedding_length':64,'qwen35moe.expert_count':2,
        'qwen35moe.expert_used_count':1,'qwen35moe.expert_feed_forward_length':64,
        'qwen35moe.expert_shared_feed_forward_length':64,
        'qwen35moe.ssm.group_count':1,'qwen35moe.ssm.time_step_rank':2,
        'qwen35moe.ssm.state_size':32,'qwen35moe.ssm.inner_size':64,
        'qwen35moe.ssm.conv_kernel':4,'qwen35moe.full_attention_interval':2,
        'qwen35moe.attention.head_count':2,'qwen35moe.attention.head_count_kv':1,
        'qwen35moe.attention.key_length':32,'qwen35moe.attention.value_length':32,
        'qwen35moe.attention.layer_norm_rms_epsilon':1e-6,
        'qwen35moe.rope.dimension_count':8,'qwen35moe.rope.freq_base':10000.0,
        'qwen35moe.rope.dimension_sections':[2,1,1,0],
        'tokenizer.ggml.tokens':[f't{i}' for i in range(128)],
        'tokenizer.ggml.token_type':[1]*127+[3], 'tokenizer.ggml.merges':[]}
    tensors=[];rng=np.random.default_rng(391)
    def add(name,shape,base=0):
        a=(rng.normal(size=shape)*.03+base).astype('<f4')
        tensors.append((name,list(reversed(shape)),0,a.tobytes()))
        return a
    add('token_embd.weight',(128,64));add('output.weight',(128,64))
    norm=add('output_norm.weight',(64,),base=1)
    for l in range(2):
        pre=f'blk.{l}.'
        for name in ['attn_norm.weight','post_attention_norm.weight']:add(pre+name,(64,),base=1)
        for name,sh in [('ffn_gate_inp.weight',(2,64)),('ffn_gate_inp_shexp.weight',(64,))]:add(pre+name,sh)
        for proj in ['gate','up','down']:
            add(pre+f'ffn_{proj}_shexp.weight',(64,64))
            # Q8 blocks with exactly representable scales and deterministic codes.
            q=np.arange(2*64*64,dtype=np.int32).reshape(-1,32)%15-7
            raw=b''.join(struct.pack('<e',.03125)+row.astype(np.int8).tobytes() for row in q)
            tensors.append((pre+f'ffn_{proj}_exps.weight',[64,64,2],8,raw))
        if l==1:
            for name,sh in [('attn_q.weight',(128,64)),('attn_k.weight',(32,64)),('attn_v.weight',(32,64)),('attn_output.weight',(64,64)),('attn_q_norm.weight',(32,)),('attn_k_norm.weight',(32,))]:add(pre+name,sh,base=1 if 'norm' in name else 0)
        else:
            for name,sh in [('attn_qkv.weight',(128,64)),('attn_gate.weight',(64,64)),('ssm_alpha.weight',(2,64)),('ssm_beta.weight',(2,64)),('ssm_conv1d.weight',(128,4)),('ssm_dt.bias',(2,)),('ssm_norm.weight',(32,)),('ssm_out.weight',(64,64))]:add(pre+name,sh)
            tensors.append((pre+'ssm_a',[2],0,np.array([-1,-2],dtype='<f4').tobytes()))
    if extra:add('unknown.weight',(64,))
    b=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(md))
    for name,value in md.items():
        typ=8 if isinstance(value,str) else 6 if isinstance(value,float) else 9 if isinstance(value,list) else 4
        if typ==9:value=(8 if not value or isinstance(value[0],str) else 4,value)
        b+=_encode_string(name)+struct.pack('<I',typ)+_encode_value(typ,value)
    data=bytearray()
    for name,dims,typ,payload in tensors:
        data.extend(b'\0'*((-len(data))%32));off=len(data);data.extend(payload)
        b+=_encode_string(name)+struct.pack('<I',len(dims))+b''.join(struct.pack('<Q',d) for d in dims)+struct.pack('<IQ',typ,off)
    path.write_bytes(b+b'\0'*((-len(b))%32)+data)
    return norm

def read_shard(path):
    raw=path.read_bytes();n=int.from_bytes(raw[:8],'little');return json.loads(raw[8:8+n]),raw[8+n:]

class ConversionTests(unittest.TestCase):
    def test_cli_container_and_refusal(self):
        tool=Path(__file__).resolve().parents[1]/'tools/convert_gguf_to_qwen36.py'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source.gguf';out=root/'converted';norm=tiny_gguf(src)
            cmd=[sys.executable,str(tool),str(src),'--out',str(out),'--v-head-order','tiled']
            r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            man=json.loads((out/'conversion-manifest.json').read_text())
            self.assertEqual(man['source_sha256'],hashlib.sha256(src.read_bytes()).hexdigest())
            for name,info in man['files'].items():self.assertEqual(info['sha256'],hashlib.sha256((out/name).read_bytes()).hexdigest())
            header,data=read_shard(out/'model-globals.safetensors');a,b=header['model.norm.weight']['data_offsets']
            np.testing.assert_array_equal(np.frombuffer(data[a:b],dtype='<f4'),norm-1)
            header,data=read_shard(out/'model-00000.safetensors')
            self.assertEqual(header['model.layers.0.mlp.experts.0.merged_weight']['shape'],[6144])
            self.assertEqual(header['model.layers.0.mlp.experts.0.qs']['shape'],[192])
            a,b=header['model.layers.0.linear_attn.A_log']['data_offsets']
            np.testing.assert_allclose(np.frombuffer(data[a:b],dtype='<f4'),np.log([1,2]),atol=1e-7)
            self.assertEqual(json.loads((out/'tokenizer.json').read_text())['model']['vocab']['t127'],127)
            bad=root/'bad.gguf';tiny_gguf(bad,extra=True)
            r=subprocess.run([sys.executable,str(tool),str(bad),'--out',str(root/'bad-out'),'--v-head-order','tiled'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('inventory mismatch',r.stderr)
    def test_head_inverse(self):
        # Independent forward indexing: grouped k*rep+r -> tiled r*nk+k.
        nk,rep,hd=3,2,4
        x=np.arange(nk*rep*hd*5,dtype=np.float32).reshape(nk*rep*hd,5)
        y=np.empty_like(x)
        for k in range(nk):
            for r in range(rep): y[(r*nk+k)*hd:(r*nk+k+1)*hd]=x[(k*rep+r)*hd:(k*rep+r+1)*hd]
        np.testing.assert_array_equal(inverse_heads(y,0,nk,nk*rep,hd),x)
        np.testing.assert_array_equal(inverse_heads(y.T,1,nk,nk*rep,hd),x.T)
    def test_norm_and_decay(self):
        x=np.array([-.1,.5,2],dtype=np.float32)
        np.testing.assert_allclose(restore_dense('ssm_a',-np.exp(x),1,1,1,1,False),x,atol=1e-6)
        np.testing.assert_allclose(restore_dense('attn_norm.weight',x+1,1,1,1,1,False),x,atol=1e-7)
        np.testing.assert_array_equal(restore_dense('ssm_norm.weight',x,1,1,1,1,False),x)
    def test_quantized_bytes(self):
        rng=np.random.default_rng(12);a=rng.normal(size=(3,128)).astype(np.float32)
        packed,sc=pack_expert([a]);q=np.empty(a.size,dtype=np.int8)
        q[::2]=packed&15;q[1::2]=packed>>4;q[q>=8]-=16
        back=q.reshape(3,2,64)*sc.reshape(3,2,1)
        err=np.abs(back-a.reshape(3,2,64))
        self.assertTrue(np.all(err<=sc.reshape(3,2,1)/2+1e-6))
        self.assertTrue(np.any(q<0));self.assertTrue(np.any(q>0))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): pack_expert([np.full((1,64),np.nan,dtype=np.float32)])
        with self.assertRaises(ValueError): restore_dense('ssm_a',np.array([0],dtype=np.float32),1,1,1,1,False)
    def test_safetensors_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.safetensors';w=Writer(p)
            w.chunks('x',(3,4),(np.full((1,4),i) for i in range(3)))
            w.add('norm',np.arange(3,dtype=np.float32));w.finish()
            raw=p.read_bytes();n=int.from_bytes(raw[:8],'little');h=json.loads(raw[8:8+n]);data=raw[8+n:]
            self.assertEqual(h['x']['data_offsets'],[0,24])
            self.assertEqual(h['norm']['data_offsets'],[24,36])
            np.testing.assert_array_equal(np.frombuffer(data[:24],dtype='<f2').reshape(3,4),np.repeat(np.arange(3),4).reshape(3,4))
if __name__=='__main__':unittest.main()
