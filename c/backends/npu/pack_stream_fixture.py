#!/usr/bin/env python3
"""Pack a private FP32 expert fixture using exported QNN tensor encodings.

Development tool only (requires NumPy on the SDK host). Does not convert a model
or qualify its calibration. Inputs are little-endian row-major FP32 arrays:
weights [experts, gate|up|down, D*F], x [vectors,D], refs [vectors,experts,D].
--int4 builds a block-64 signed INT4 fixture and computes CPU SwiGLU references
for the quantized weights; original-weight quantization loss is saved separately.
The generated header and binary fixtures stay outside the source repository.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def pack(args):
    info = json.loads(args.info.read_text())['info']
    graphs = info['graphs']
    if len(graphs) != 1:
        raise ValueError('fixture requires exactly one graph')
    graph = graphs[0]['info']
    inputs = [t['info'] for t in graph['graphInputs']]
    outputs = [t['info'] for t in graph['graphOutputs']]
    if len(inputs) != 4 or len(outputs) != 1:
        raise ValueError('expected x, gW, uW, dW inputs and y output')
    tensors = inputs + outputs
    names = [t['name'] for t in tensors]
    if set(names[:4]) != {'x', 'gW', 'uW', 'dW'} or names[4] != 'y':
        raise ValueError('unexpected tensor names')
    by_name = dict(zip(names, tensors))
    d = by_name['x']['dimensions'][1]
    f = by_name['gW']['dimensions'][0]
    if not 1 <= d <= 65536 or not 1 <= f <= 65536:
        raise ValueError('invalid fixture dimensions')
    expected = {'x': [1, d], 'y': [1, d], 'gW': [f, d],
                'uW': [f, d], 'dW': [d, f]}
    encodings = {}
    for tensor in tensors:
        name = tensor['name']
        quant = tensor['quantizeParams']
        if (tensor['dimensions'] != expected[name] or tensor['rank'] != 2 or
                tensor['dataType'] not in ('QNN_DATATYPE_UFIXED_POINT_16', 'QNN_DATATYPE_FLOAT_16') or
                tensor['dataFormat'] != 'QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER'):
            raise ValueError('unsupported tensor encoding: ' + name)
        if tensor['dataType'] == 'QNN_DATATYPE_FLOAT_16':
            if quant['definition'] != 'QNN_DEFINITION_UNDEFINED':
                raise ValueError('quantized float tensor: ' + name)
            encodings[name] = (1.0, 0)
            continue
        if (quant['definition'] != 'QNN_DEFINITION_DEFINED' or
                quant['quantizationEncoding'] != 'QNN_QUANTIZATION_ENCODING_SCALE_OFFSET'):
            raise ValueError('unsupported quantization: ' + name)
        scale = quant['scaleOffset']['scale']
        offset = quant['scaleOffset']['offset']
        if not math.isfinite(scale) or scale <= 0 or not -65535 <= offset <= 0:
            raise ValueError('invalid scale/offset: ' + name)
        encodings[name] = (scale, offset)
    n = args.experts
    if not 1 <= n <= 1024:
        raise ValueError('experts must be in 1..1024')
    w = np.fromfile(args.weights, dtype='<f4')
    x = np.fromfile(args.x, dtype='<f4')
    refs = np.fromfile(args.refs, dtype='<f4')
    if w.size != n*3*d*f or x.size % d or not x.size:
        raise ValueError('weights/input size mismatch')
    v = x.size // d
    if not 1 <= v <= 1024 or refs.size != v*n*d:
        raise ValueError('references/vector count mismatch')
    for source, arr in ((args.weights, w), (args.x, x), (args.refs, refs)):
        if source.stat().st_size != arr.size*4 or not np.isfinite(arr).all():
            raise ValueError('invalid FP32 source: ' + str(source))
    w = w.reshape(n, 3, d*f)
    args.output.mkdir(parents=True, exist_ok=True)
    saturation = {}

    def quantize(a, name):
        if by_name[name]['dataType'] == 'QNN_DATATYPE_FLOAT_16':
            half = a.astype('<f2')
            if not np.isfinite(half).all():
                raise ValueError('FP16 overflow: ' + name)
            saturation[name] = 0
            return half.view('<u2')
        scale, offset = encodings[name]
        # Quantize against the actual float32 scale exported by QNN.
        codes = np.rint(a.astype(np.float64)/scale - offset)
        saturation[name] = int(np.count_nonzero((codes < 0) | (codes > 65535)))
        return codes.clip(0, 65535).astype('<u2')

    qw = np.empty(w.shape, dtype='<u2')
    for k, name in enumerate(('gW', 'uW', 'dW')):
        qw[:, k] = quantize(w[:, k], name)
    weights_file='weights.u16'
    encoded_bytes=3*d*f*2
    if args.int4:
        if d*f%64 or any(t['dataType']!='QNN_DATATYPE_FLOAT_16' for t in tensors):
            raise ValueError('INT4 storage requires FP16 graph and 64-value matrix groups')
        groups=w.reshape(n,3,-1,64)
        scales=(np.max(np.abs(groups),axis=-1)/7).astype('<f2')
        if not np.isfinite(scales).all():
            raise ValueError('INT4 scale overflow')
        safe=np.where(scales==0,1,scales).astype(np.float32)
        codes=np.clip(np.rint(groups/safe[...,None]),-8,7).astype(np.int8)
        codes=np.where((scales==0)[...,None],0,codes).astype(np.int8)
        packed=np.empty((*scales.shape,34),dtype=np.uint8)
        bits=scales.view('<u2')
        packed[...,0]=bits&255; packed[...,1]=bits>>8
        packed[...,2:]=(codes[...,::2]&15)|((codes[...,1::2]&15)<<4)
        weights_file='weights.i4'; encoded_bytes=3*(d*f//64)*34
        packed.tofile(args.output/weights_file)
        # Compare device math against CPU math using precisely the same low-bit
        # weights and FP16 expansion; quantization loss is a separate metric.
        decoded=(codes.astype(np.float32)*scales.astype(np.float32)[...,None]).astype('<f2').astype(np.float32).reshape(n,3,d*f)
        expected=np.empty((v,n,d),dtype='<f4')
        for vi,vec in enumerate(x.reshape(v,d)):
            for ei in range(n):
                gate=decoded[ei,0].reshape(f,d)@vec
                up=decoded[ei,1].reshape(f,d)@vec
                expected[vi,ei]=decoded[ei,2].reshape(d,f)@(gate/(1+np.exp(-gate))*up)
        before=refs.reshape(v,n,d)
        if not np.isfinite(expected).all():
            raise ValueError('nonfinite INT4 CPU reference')
        loss=np.linalg.norm(expected-before,axis=-1)/np.maximum(np.linalg.norm(before,axis=-1),1e-30)
        (args.output/'int4-quantization-loss.json').write_text(json.dumps({'relative_l2':loss.tolist(),'max_relative_l2':float(loss.max())},indent=2)+'\n')
        refs=expected
    else:
        qw.tofile(args.output/weights_file)
    quantize(x, 'x').tofile(args.output/'x.u16')
    refs.tofile(args.output/'refs.f32')
    header = ['/* Generated from QNN context info; do not edit. */',
              f'enum {{ FIX_D={d}, FIX_F={f}, FIX_N={n}, FIX_V={v} }};',
              f'enum {{ FIX_INT4={int(args.int4)} }};',
              f'static const size_t fix_encoded_bytes = {encoded_bytes}u;',
              'static const char *fix_graph = '+json.dumps(graph['graphName'])+';',
              'static const char *fix_names[] = {'+','.join(map(json.dumps, names))+'};',
              'static const uint32_t fix_ids[] = {'+','.join(str(t['id']) for t in tensors)+'};',
              'static const Qnn_DataType_t fix_types[] = {'+','.join(t['dataType'] for t in tensors)+'};',
              'static uint32_t fix_dims[][2] = {'+','.join('{'+','.join(map(str,t['dimensions']))+'}' for t in tensors)+'};',
              'static const float fix_scales[] = {'+','.join(float(encodings[n][0]).hex()+'f' for n in names)+'};',
              'static const int32_t fix_offsets[] = {'+','.join(str(encodings[n][1]) for n in names)+'};',
              'static const unsigned fix_widx[] = {'+','.join(str(names.index(n)) for n in ('gW','uW','dW'))+'};',
              f'enum {{ FIX_X={names.index("x")}, FIX_Y=4 }};']
    (args.output/'coli_stream_fixture.h').write_text('\n'.join(header)+'\n')
    manifest = {'dimensions': {'D': d, 'F': f, 'experts': n, 'vectors': v},
                'tensor_types': {t['name']: t['dataType'] for t in tensors},
                'storage_format': 'i4-f16-block64-v1' if args.int4 else 'native-16bit',
                'weights_file': weights_file, 'bytes_per_expert': encoded_bytes,
                'clipped_input_values': saturation, 'files': {}}
    for name in (weights_file, 'x.u16', 'refs.f32', 'coli_stream_fixture.h'):
        manifest['files'][name] = hashlib.sha256((args.output/name).read_bytes()).hexdigest()
    manifest['context_info_sha256'] = hashlib.sha256(args.info.read_bytes()).hexdigest()
    (args.output/'fixture.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('info', 'weights', 'x', 'refs', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--experts', type=int, required=True)
    p.add_argument('--int4', action='store_true', help='pack block-64 INT4 and compute matching CPU SwiGLU references')
    pack(p.parse_args())
