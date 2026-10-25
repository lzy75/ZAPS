"""Read-only structural audit of every inventoried JSON source; no tensor/model.

This traverses all JSON nodes and checks inventory byte hashes. It does not
recompute scientific experiments or certify nested claims, remote artifacts,
server tensor hashes, signatures, or statistical analyses. Original files stay
unchanged. Large manifests remain manifests, not additional observations.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def inspect(spec):
    path=Path(spec['path']);data=path.read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    if sha!=spec['sha256']:raise ValueError('Inventory bytes differ:'+str(path))
    constants=[];duplicates=[]
    def constant(value):
        constants.append(value)
        return {'NaN':float('nan'),'Infinity':float('inf'),'-Infinity':-float('inf')}[value]
    def pairs(items):
        output={}
        for key,value in items:
            if key in output:duplicates.append(key)
            output[key]=value
        return output
    value=json.loads(data,parse_constant=constant,object_pairs_hook=pairs)
    kinds=Counter();classes=Counter();statuses=Counter();budgets=[];scalar_roots={}
    stack=[('$',value,0)]
    while stack:
        pointer,item,depth=stack.pop();kinds[type(item).__name__]+=1
        if isinstance(item,dict):
            for key,child in item.items():
                if key=='classification' and isinstance(child,str):classes[child]+=1
                if key=='status' and isinstance(child,str):statuses[child]+=1
                if depth<=2 and key in ('method_nfe','diagnostic_nfe','total_nfe','completed_cases','planned_cases','image_clusters','source_t','experiment','classification','status') and not isinstance(child,(dict,list)):
                    scalar_roots[pointer+'/'+key]=child
                if key in ('method_nfe','diagnostic_nfe') and isinstance(child,(int,float)):
                    budgets.append(dict(pointer=pointer+'/'+key,value=child))
                stack.append((pointer+'/'+str(key),child,depth+1))
        elif isinstance(item,list):
            stack.extend((pointer+'/'+str(i),child,depth+1) for i,child in enumerate(item))
        elif isinstance(item,float) and not math.isfinite(item):kinds['nonfinite_float']+=1
    return dict(path=str(path),sha256=sha,bytes=len(data),top_level_keys=list(value) if isinstance(value,dict) else None,
                all_nodes_traversed=True,node_types=dict(kinds),nonstandard_constants=dict(Counter(constants)),
                duplicate_object_keys=dict(Counter(duplicates)),classifications=dict(classes),statuses=dict(statuses),
                shallow_scalar_readouts=scalar_roots,nested_cost_fields=budgets,
                cost_policy='Pointers retained,not summed: nested costs may include repeated/cumulative/reused scopes.',
                evidence_limit='Structural/hash audit only. No automatic promotion to scientific verification or complete human reading.')


def main():
    inventory=json.loads((ROOT/'source_inventory.json').read_text())
    sources=[s for s in inventory['sources'] if Path(s['path']).suffix.lower()=='.json']
    rows=[inspect(s) for s in sources]
    out=dict(checked_utc=datetime.now(timezone.utc).isoformat(),source_inventory_sha256=hashlib.sha256((ROOT/'source_inventory.json').read_bytes()).hexdigest(),
             sources=len(rows),source_bytes=sum(r['bytes'] for r in rows),all_inventory_hashes_match=True,
             nonstandard_json_sources=sum(bool(r['nonstandard_constants']) for r in rows),
             duplicate_key_sources=sum(bool(r['duplicate_object_keys']) for r in rows),rows=rows,
             scope='Accessible inventoried JSON only. No missing sources invented,remote results fetched,tensors opened,UNet calls or original files edited.')
    (ROOT/'evidence_structural_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False))
    for r in rows:
        print(json.dumps(dict(name=Path(r['path']).name,parent=Path(r['path']).parent.name,bytes=r['bytes'],classes=r['classifications'],status=r['statuses'],readout=r['shallow_scalar_readouts'],nonstandard=r['nonstandard_constants'],duplicates=r['duplicate_object_keys']),ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
