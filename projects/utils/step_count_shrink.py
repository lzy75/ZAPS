"""A74 fixed10/15/30 counts; preserve original low/mid/high section convention."""
from utils import all_unroll_shrink as previous
from utils import shrink_multitask as parent
TASKS = parent.TASKS
ARMS = ('BASE', 'SHRINK')
COUNTS = {10:(5,3,2), 15:(8,5,2), 30:(15,10,5)}


def grid(steps):
    points=[];start=0
    for size,count in zip((334,333,333),COUNTS[steps]):
        current=0.
        for _ in range(count):
            points.append(start+round(current));current+=(size-1)/(count-1)
        start+=size
    return points


def configuration(task,steps,size=256,level=3):
    cfg=parent.configuration(task,size,level)
    cfg.update(num_steps=steps,schedule=list(COUNTS[steps]))
    return cfg


def cases(parent_cases):
    out=[]
    for original in previous.cases(parent_cases):
        for steps in COUNTS:
            c=dict(original,parent_key=original['key'],steps=steps,key=original['key']+f'_S{steps}')
            c['shard']=len(out)%2;out.append(c)
    return out
