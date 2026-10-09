import sys
import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from convert_gguf_to_qwen36 import inverse_heads,restore_dense,pack_expert,Writer

class ConversionTests(unittest.TestCase):
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
