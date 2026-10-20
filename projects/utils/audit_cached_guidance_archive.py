"""Recompute complete lightweight A2 archive after Git download. No tensor/model loading."""
import argparse
import json
from pathlib import Path
import sys
PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import diag_direction_refresh as d
from utils import diag_cached_guidance_state_path as b0


def audit(root):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['files']:
        path=root/item['archive_path']
        if not path.resolve().is_relative_to(root) or path.stat().st_size!=item['bytes'] or d.digest(path)!=item['sha256']:
            raise ValueError('Archive byte hash/size differs: '+str(path))
    registry=json.loads((root/'A2a/recovery_registry.json').read_text())
    plan=json.loads((root/'A2a/plan.json').read_text());active=[]
    for c in plan['cases']:
        key=c['key'];entry=registry['active'].get(key)
        folder=root/'A2a'/(entry['directory'] if entry else key)
        r=json.loads((folder/'run.json').read_text())
        if r['status']!='complete' or r['case']!=c or len(r['rows'])!=30 or len(r['probes'])!=5 or not all(g['passed'] for g in r['gates'].values()):
            raise ValueError('Incomplete/incompatible A2a active case: '+key)
        if entry and d.digest(folder/'run.json')!=entry['run_sha256']:raise ValueError('Recovery registry SHA differs')
        active.append(r)
    recovered=d.archive.recovery.stage_summary(root/'A2a',plan,registry,{r['case']['key']:r for r in active})
    if recovered!=json.loads((root/'A2a/recovery_summary.json').read_text()):raise ValueError('A2a summary differs')
    checks=dict(A2a=dict(cases=6,trajectory_points=180,probes=30,classification=recovered['classification'],
                        method_nfe=recovered['method_nfe'],diagnostic_nfe=recovered['diagnostic_nfe']))
    for name,module,comparisons in (('A2c',d.previous,(('0.5','0.5','1.0'),('0.0','0.0','1.0'))),
                                    ('A2d',d,(('F_C','F','C'),('M_C','M','C'),('F_M','F','M')))):
        records=[json.loads(p.read_text()) for p in sorted((root/name).glob('*/run.json'))]
        # A2c uses gamma-named arm/effect keys; inspect its frozen schema, not inferred labels.
        count=0
        for record in records:
            if record['status']!='complete' or not all(g['passed'] for g in record['gates'].values()):raise ValueError(name+' case gates')
            for point in record['points']:
                expected={key:module.effect(point['branches'][left],point['branches'][right],point['floors']) for key,left,right in comparisons}
                if name=='A2d':
                    geo=point['geometry']
                    if any(geo[k]!=v for k,v in d.geometry(geo['a0'],geo['ah'],geo['chord'],geo['repeat_chord']).items()):raise ValueError('A2d geometry differs')
                    if d.direction_candidate(expected,point['shift_resolved'])!=point['joint_direction_candidate']:raise ValueError('A2d joint decision differs')
                if expected!=point['effects']:raise ValueError(name+' effects differ')
                count+=len(expected)
        summary=module.summarize(records)
        saved=json.loads((root/name/'summary.json').read_text())
        if name=='A2d':summary['cost_is_lower_bound_if_abruptly_interrupted']=False
        if summary!=saved:raise ValueError(name+' summary differs')
        checks[name]=dict(cases=len(records),points=sum(len(r['points']) for r in records),comparisons=count,
                          classification=summary['classification'],diagnostic_nfe=summary['diagnostic_nfe'])
        if name=='A2d':checks[name]['geometry_points']=30
    record=json.loads((root/'A2b0/run.json').read_text())
    if record['status']!='complete' or not all(g['passed'] for g in record['gates'].values()) or b0.associations(record['rows'])!=record['summary']['matched_t_associations']:
        raise ValueError('A2b0 association or gates differ')
    checks['A2b0']=dict(points=len(record['rows']),associations=len(record['summary']['matched_t_associations']),gates=True)
    checks['archive']=dict(copied_files=len(manifest['files']),excluded_files=len(manifest['excluded']),all_copied_byte_hashes_equal=True,
                           binary_tensors_independently_verified=False,manifest_sha256=d.digest(root/'manifest.json'))
    return dict(local_archive=str(root),checks=checks,new_method_nfe=0,new_diagnostic_nfe=0,
                limits='Complete lightweight JSON/CSV and byte hashes; excluded tensors remain on server. No independent local model/tensor/GPU execution.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive_dir');p.add_argument('--output');args=p.parse_args()
    result=audit(args.archive_dir)
    if args.output:
        with Path(args.output).open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
