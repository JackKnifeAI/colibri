#!/usr/bin/env python3
"""Generate a private, context-pinned FP16 SwiGLU IO profile (stdlib only).

Input is qnn-context-binary-utility JSON. The graph's mathematics must be
qualified separately: IO shape/metadata alone cannot establish SwiGLU semantics.
"""
import argparse
import hashlib
import json
from pathlib import Path


def export(info, binary):
    graphs = info['info']['graphs']
    if len(graphs) != 1:
        raise ValueError('exactly one graph required')
    graph = graphs[0]['info']
    ts = [t['info'] for t in graph['graphInputs'] + graph['graphOutputs']]
    names = [t['name'] for t in ts]
    if len(ts) != 5 or set(names[:4]) != {'x', 'gW', 'uW', 'dW'} or names[4] != 'y':
        raise ValueError('expected four named inputs and y output')
    by = dict(zip(names, ts))
    d, f = by['x']['dimensions'][1], by['gW']['dimensions'][0]
    if not (0 < d <= 16384 and 0 < f <= 16384 and d % 64 == f % 64 == 0):
        raise ValueError('dimensions must be bounded multiples of 64')
    dims = {'x': [1, d], 'y': [1, d], 'gW': [f, d], 'uW': [f, d], 'dW': [d, f]}
    ids = []
    for i, t in enumerate(ts):
        if (t['dimensions'] != dims[t['name']] or t['rank'] != 2 or
            t['dataType'] != 'QNN_DATATYPE_FLOAT_16' or
            t['dataFormat'] != 'QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER' or
            t['type'] != ('QNN_TENSOR_TYPE_APP_READ' if i == 4 else 'QNN_TENSOR_TYPE_APP_WRITE') or
            t['quantizeParams']['definition'] != 'QNN_DEFINITION_UNDEFINED' or
            not isinstance(t['id'], int) or not 0 <= t['id'] < 2**32):
            raise ValueError('invalid tensor: ' + t['name'])
        ids.append(t['id'])
    if len(set(ids)) != 5:
        raise ValueError('duplicate tensor IDs')
    return '\n'.join([
        '/* Generated private QNN expert profile; do not edit. */',
        f'enum {{ NP_D={d}, NP_F={f}, NP_X={names.index("x")} }};',
        'static const char np_graph[] = ' + json.dumps(graph['graphName']) + ';',
        'static const char np_sha256[] = "' + hashlib.sha256(binary).hexdigest() + '";',
        'static const char *np_names[] = {' + ','.join(map(json.dumps, names)) + '};',
        'static const uint32_t np_ids[] = {' + ','.join(map(str, ids)) + '};',
        'static uint32_t np_dims[][2] = {' + ','.join('{' + ','.join(map(str, t['dimensions'])) + '}' for t in ts) + '};',
        'static const unsigned np_widx[] = {' + ','.join(str(names.index(n)) for n in ('gW', 'uW', 'dW')) + '};',
        ''
    ])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--info', type=Path, required=True)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(export(json.loads(a.info.read_text()), a.context.read_bytes()))
