"""Frozen A76 label-free leave-one-image-out profiles and cyclic donors."""
import math
import statistics
from utils.dps_multiplier_transfer import TASKS, STEPS, SCALE
ARMS=('DPS','SHRINK','FIXED_LOO','MISMATCH')
RUN_ARMS=ARMS[2:]

def profiles(cases,traces):
    out={}
    for c in cases:
        peers=sorted((v for v in cases if v['task']==c['task'] and v['sigma']==c['sigma']),key=lambda v:v['image_id'])
        if len(peers)!=3 or len({v['image_id'] for v in peers})!=3:raise ValueError('Need three same-task image clusters')
        for v in peers:
            curve=traces[v['key']]
            if len(curve)!=STEPS or [r['t'] for r in curve]!=list(range(999,-1,-1)):raise ValueError('Incomplete native1000 profile')
            if any(not math.isfinite(r['gamma']) or r['gamma']<0 for r in curve):raise ValueError('Invalid gamma')
        donors=[v for v in peers if v['image_id']!=c['image_id']]
        cyclic=peers[(peers.index(c)+1)%len(peers)]
        out[c['key']]={
            'FIXED_LOO':dict(gamma=[statistics.median(traces[v['key']][i]['gamma'] for v in donors) for i in range(STEPS)],donors=[v['key'] for v in donors]),
            'MISMATCH':dict(gamma=[r['gamma'] for r in traces[cyclic['key']]],donors=[cyclic['key']])}
    return out
