"""Recompute submitted A46 deltas/flags only, never server tensors or GPU."""
from datetime import datetime,timezone
import hashlib
import json
import numbers
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from utils import frozen_four_task_confirmation as rules


def main():
    src=ROOT.parent/'experiment_cards/evidence/A46_user_reported_summary_20261025.json'
    summary=json.loads(src.read_text())['summary'];records=[]
    for group in summary['groups']:
        for image,arms in group['per_image'].items():
            records.append(dict(case=dict(task=group['task'],image_id=image),gamma=group['gammas'][image],
                                evaluation={name:dict(delta=value) for name,value in arms.items()}))
    new=rules.summarize(records);errors=[]
    def same(a,b,path='$'):
        if isinstance(a,bool) or isinstance(b,bool):
            if a!=b:raise ValueError('Boolean mismatch:'+path)
        elif isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):raise ValueError('Keys mismatch:'+path)
            for key in a:same(a[key],b[key],path+'/'+key)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):raise ValueError('Length mismatch:'+path)
            for i,(x,y) in enumerate(zip(a,b)):same(x,y,path+'/'+str(i))
        elif isinstance(a,numbers.Number) and isinstance(b,numbers.Number):
            error=abs(float(a)-float(b));errors.append(error)
            if error>1e-12:raise ValueError('Numeric mismatch:'+path)
        elif a!=b:raise ValueError('Value mismatch:'+path)
    same(new,summary['groups'])
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),status='USER_TABLE_ARITHMETIC_AND_FROZEN_RULE_RECOMPUTED',
        source=str(src.resolve()),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        helper=str(Path(rules.__file__).resolve()),helper_sha256=hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest(),
        rows=len(records),image_clusters=len({r['case']['image_id'] for r in records}),numeric_comparisons=len(errors),
        max_absolute_error=max(errors),all_flags_match=True,quality_tasks=sum(g['quality_candidate'] for g in new),
        matching_tasks=sum(g['state_matching_candidate'] for g in new),covariance_increment_tasks=sum(g['covariance_increment_candidate'] for g in new),
        groups=[{k:g[k] for k in ('task','lambda_frozen','quality_vs_C','engineering_noninferiority_vs_matched','quality_candidate','state_matching_candidate','covariance_increment_candidate')} for g in new],
        method_nfe=0,diagnostic_nfe=0,
        limits='Submitted per-image DELTA table and unchanged frozen rule only. Not raw server archive/tensor/measurement replay;absolute C and provenance gates not independently certified.')
    (ROOT/'A46_user_table_recomputed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','rows','numeric_comparisons','max_absolute_error','quality_tasks','matching_tasks','covariance_increment_tasks')}))


if __name__=='__main__':main()
