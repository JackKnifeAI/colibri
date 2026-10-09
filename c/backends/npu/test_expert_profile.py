#!/usr/bin/env python3
"""Profile boundary tests; --planar-fixture also emits independent device data."""
import copy
import hashlib
import random
import struct
import sys
import unittest
from pathlib import Path
from export_expert_profile import export


def metadata():
    dims = [('gW', [512, 2048]), ('x', [1, 2048]), ('uW', [512, 2048]),
            ('dW', [2048, 512]), ('y', [1, 2048])]
    ts = [{'info': dict(name=name, dimensions=d, rank=2, id=i+1,
           type='QNN_TENSOR_TYPE_APP_READ' if i == 4 else 'QNN_TENSOR_TYPE_APP_WRITE',
           dataType='QNN_DATATYPE_FLOAT_16', dataFormat='QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER',
           quantizeParams={'definition': 'QNN_DEFINITION_UNDEFINED'})}
          for i, (name, d) in enumerate(dims)]
    return {'info': {'graphs': [{'info': {'graphName': 'test',
            'graphInputs': ts[:4], 'graphOutputs': ts[4:]}}]}}


class ProfileTests(unittest.TestCase):
    def test_binding(self):
        text = export(metadata(), b'context')
        self.assertIn('NP_D=2048, NP_F=512, NP_X=1', text)
        self.assertIn(hashlib.sha256(b'context').hexdigest(), text)
        self.assertNotEqual(text, export(metadata(), b'other context'))

    def test_rejections(self):
        for key, value in [('dataType', 'QNN_DATATYPE_UFIXED_POINT_16'),
                           ('dimensions', [512, 2049]), ('id', 2),
                           ('id', -1), ('rank', 3), ('type', 'QNN_TENSOR_TYPE_STATIC')]:
            with self.subTest(key=key, value=value):
                bad=copy.deepcopy(metadata())
                bad['info']['graphs'][0]['info']['graphInputs'][0]['info'][key]=value
                with self.assertRaises(ValueError):
                    export(bad,b'context')


def planar_fixture(path):
    path.mkdir(parents=True,exist_ok=True)
    rng=random.Random(7909)
    packed=bytearray(); scales=bytearray(); expected=bytearray()
    def f32(x):
        return struct.unpack('<f',struct.pack('<f',x))[0]
    for b in range(4096):
        scale=f32(0.0 if b == 0 else 2**rng.uniform(-27,11)*rng.uniform(.5,1))
        scales+=struct.pack('<f',scale)
        q=[i%16-8 if b<32 else rng.randrange(-8,8) for i in range(64)]
        packed+=bytes((q[i]+8)|((q[i+32]+8)<<4) for i in range(32))
        for v in q:
            expected+=struct.pack('<e',f32(v*scale))
    (path/'weights.bin').write_bytes(packed)
    (path/'scales.bin').write_bytes(scales)
    (path/'expected.bin').write_bytes(expected)


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--planar-fixture':
        planar_fixture(Path(sys.argv[2]))
    else:
        unittest.main()
