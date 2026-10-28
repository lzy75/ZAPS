"""A66 append-only independent continuation using unchanged A65 numerical executor."""
import argparse
import builtins
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS))
from utils import current_strong_anchor as rules
from utils import historical_ffhq_gain as design
from utils import diag_historical_ffhq_gain as executor
from utils import historical_ffhq_runtime as engine
from utils.diag_paper_clock_alignment import write,resource,finite
CARD=PROJECTS/'experiment_cards/A66_current_strong_anchor_continuation_20261028.md'


def check(root):
    p=rules.read(root/'plan.json');rules.require(p['experiment']==rules.EXPERIMENT,'Wrong continuation plan')
    for path,sha in p['source_sha256'].items():rules.require(rules.digest(path)==sha,'Bound source changed: '+path)
    return p


def prepare(args):
    root,source=Path(args.root).resolve(),Path(args.parent).resolve()
    if (root/'plan.json').exists():check(root);print('A66 frozen continuation reused.',flush=True);return
    rules.require(not (root.is_relative_to(source) or source.is_relative_to(root)),'Source/output overlap')
    rules.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty output')
    from utils.a54_session import busy
    locks=[source.parent/(source.name+'_session')/'coordinator.lock',*source.glob('workers/*/worker.lock')]
    rules.require(all(not f.exists() or not busy(f) for f in locks),'Wait for source A65 session to finish')
    received=PROJECTS/'reports/20261028_a65_received/A65'
    for name in ('plan.json','summary.json','replay_gate.json','null_diagnostic/run.json'):
        rules.require(rules.digest(source/name)==rules.digest(received/name),'Source differs from received A65: '+name)
    p=executor.check(source);summary=rules.read(source/'summary.json');null=rules.read(source/'null_diagnostic/run.json')
    rows=[executor.done(source,c,'HISTORICAL_NATIVE',0) for c in p['cases']]
    receipt=rules.source_contract(p,summary,rows,null)
    root.parent.mkdir(parents=True,exist_ok=True)
    rules.require(shutil.disk_usage(root.parent).free>=5*design.GIB,'Need5GiB free on result filesystem')
    root.mkdir();identities=dict(p['source_sha256']);references={};devices={}
    for name in ('plan.json','summary.json','replay_gate.json','null_diagnostic/run.json'):
        file=source/name;identities[str(file)]=rules.digest(file)
        destination=root/'inherited/A65'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,destination)
    for i,(c,r) in enumerate(zip(p['cases'],rows)):
        old=source/'cases'/c['key']/'HISTORICAL_NATIVE/repeat_0'
        run=old/'run.json';identities[str(run)]=rules.digest(run)
        for name in ('output.pt',):identities[str(old/name)]=r['artifacts'][name]
        inherited=root/'inherited/A65/cases'/c['key'];inherited.mkdir(parents=True)
        shutil.copyfile(run,inherited/'run.json')  # Byte-identical A65 record, original plan SHA intact.
        reference=root/'cases'/c['key']/'HISTORICAL_NATIVE/repeat_0';reference.mkdir(parents=True)
        shutil.copyfile(old/'output.pt',reference/'output.pt')
        write(reference/'reference_only.json',dict(source_run=str(run),sha256=rules.digest(run),
            role='Output-only repeat-stability reference; NOT an A66 execution; no learned parameters'))
        obs_path=Path(r['observation']['path']);identities[str(obs_path)]=r['observation']['sha256']
        obs_target=root/'observations'/(c['key']+'.pt');obs_target.parent.mkdir(exist_ok=True);shutil.copyfile(obs_path,obs_target)
        obs=dict(r['observation'],path=str(obs_target),source_A65_observation=str(obs_path))
        write(obs_target.with_suffix('.json'),obs)
        references[c['key']]=dict(source_run=str(run),run_sha256=rules.digest(run),
            local_record=str(inherited/'run.json'),output_reference=str(reference/'output.pt'))
        shard=str(i//5)
        rules.require(shard not in devices or devices[shard]==r['device_uuid'],'Source shard uses mixed GPU UUIDs')
        devices[shard]=r['device_uuid']
    own=[Path(__file__),Path(rules.__file__),CARD,PROJECTS/'run_a66_20261028.sh',
         PROJECTS/'utils/a66_session.py',PROJECTS/'utils/test_current_strong_anchor.py']
    identities.update({str(f):rules.digest(f) for f in own})
    for file in (root/'inherited').rglob('*.json'):identities[str(file)]=rules.digest(file)
    for ref in references.values():identities[ref['output_reference']]=rules.digest(ref['output_reference'])
    for file in (root/'observations').iterdir():identities[str(file)]=rules.digest(file)
    new=dict(p,experiment=rules.EXPERIMENT,parent_A65=str(source),source_sha256=identities,
        original_A65_plan_sha256=rules.digest(source/'plan.json'),references=references,required_devices=devices,
        source_acceptance=receipt,queue=rules.queue(),budget=rules.budget(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECTS.parent,text=True).strip())
    write(root/'plan.json',new);shutil.copyfile(CARD,root/'experiment_card.md')
    print('A66 PREPARE PASS: retain failed A65 historical gate; use current strong anchor. '
          '10 native runs inherited,140 runs remaining; new33900 forwards,<=1800 input VJP. No parameter changes.',flush=True)


def worker(args):
    import torch
    from utils.a54_session import exclusive
    from modules.main_single import load_diffusion_model
    from utils.current_score_runtime_r2 import input_vjp_parameters,raw_mean
    from utils.current_score_geometry import input_vjp
    root=Path(args.root).resolve();p=check(root);folder=root/'workers'/('shard'+str(args.shard));folder.mkdir(parents=True,exist_ok=True)
    with exclusive(folder/'worker.lock'):
        rules.require(args.device_uuid==p['required_devices'][str(args.shard)],'Keep inherited timing GPU UUID')
        admission=resource(args.device_uuid,'ffhq')
        if not admission['admitted']:write(folder/'status.json',dict(status='resource_limited',admission=admission));return
        rules.require(torch.cuda.is_available(),'CUDA required')
        rules.require(rules.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Checkpoint changed')
        cases=[c for i,c in enumerate(p['cases']) if i//5==args.shard]
        for c in cases:rules.require(rules.digest(c['image'])==c['image_sha256'],'Image changed')
        started=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval();torch.cuda.synchronize()
        record=dict(status='running',shard=args.shard,uuid=args.device_uuid,
            model_load_seconds=time.perf_counter()-started,torch=str(torch.__version__),cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),warmup=dict(status='running',forward_attempts=0,forwards=0,input_vjp_attempts=0))
        file=folder/f'load_{len(list(folder.glob("load_*.json"))):03d}.json';write(file,record)
        try:
            warm=record['warmup'];start=time.perf_counter()
            with engine.precision('LEGACY_IEEE'),input_vjp_parameters(model.model):
                warm['forward_attempts']=1;write(file,record)
                inp,raw=raw_mean(model,torch.zeros(1,3,256,256,device=model.device),999,graph=True)
                warm.update(forwards=1,input_vjp_attempts=1);write(file,record)
                value=input_vjp(raw,inp,torch.ones_like(raw));rules.require(bool(torch.isfinite(value).all()),'Warmup nonfinite')
            torch.cuda.synchronize();warm.update(status='complete',seconds=time.perf_counter()-start)
            del inp,raw,value;gc.collect();torch.cuda.empty_cache();write(file,record)
            versions=[(v,v._version) for v in model.model.parameters()]
            # Only the shared executor's log label changes; its numerical code and source bytes do not.
            executor.print=lambda *values,**kw: builtins.print(*(v.replace('A65 ','A66 ',1)
                if isinstance(v,str) else v for v in values),**kw)
            for arm,repeat in rules.queue():
                for c in cases:
                    result=executor.measure(root,c,arm,repeat,model,args.device_uuid,versions)
                    if result is None:record['status']='resource_limited';return
            record['status']='complete'
        except BaseException as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
        finally:
            record['worker_seconds']=time.perf_counter()-started;write(file,record);write(folder/'status.json',record)


def summarize(root):
    root=Path(root);p=check(root)
    inherited=[rules.read(p['references'][c['key']]['local_record']) for c in p['cases']]
    attempts=[rules.read(f) for f in sorted((root/'cases').glob('*/*/repeat_*/run.json'))]
    fresh=[r for r in attempts if r['status']=='complete'];records=inherited+fresh;groups=[]
    for arm in design.ARMS:
        rows=[r for r in records if r['arm']==arm];first=[r for r in rows if r['repeat']==0]
        group=dict(arm=arm,name=design.NAMES[arm],images=len(first),complete_runs=len(rows),planned_runs=30,
            forward_nfe=design.NFE[arm],optimizer_backwards=design.UPDATES[arm],times_by_image={})
        if first:
            group['quality']={k:statistics.mean(float(r['metrics'][k]) for r in first) for k in ('psnr','ssim','lpips','float_psnr')}
            group['mean_input_vjp']=statistics.mean(r['counts']['input_vjp_attempts'] for r in first)
            group['mean_fallback_steps']=statistics.mean(r['fallback_steps'] for r in first)
        for image in sorted({r['case']['image_id'] for r in rows}):
            values=[r['complete_reconstruction_seconds'] for r in sorted(rows,key=lambda r:r['repeat']) if r['case']['image_id']==image]
            group['times_by_image'][image]=design.summarize_times(values) if len(values)==3 else dict(repeats=values)
        if len(rows)==30:group['mean_of_image_median_seconds']=statistics.mean(v['median'] for v in group['times_by_image'].values())
        group['paired_vs']={}
        for arm_ref in ('HISTORICAL_NATIVE','LEGACY_IEEE','EXACT_IEEE'):
            peers={r['case']['image_id']:r for r in records if r['arm']==arm_ref and r['repeat']==0}
            paired=[dict(image_id=r['case']['image_id'],**{k:float(r['metrics'][k])-float(peers[r['case']['image_id']]['metrics'][k])
                    for k in ('psnr','ssim','lpips')}) for r in first if r['case']['image_id'] in peers]
            if paired:group['paired_vs'][arm_ref]=dict(n=len(paired),per_image=paired,
                mean_delta={k:statistics.mean(r[k] for r in paired) for k in ('psnr','ssim','lpips')})
        groups.append(group)
    loads=[rules.read(f) for f in sorted((root/'workers').glob('*/load_*.json'))]
    summary=dict(experiment=rules.EXPERIMENT,classification='CURRENT_STRONG_REFERENCE_METHOD_COMPARISON',
        source_acceptance=p['source_acceptance'],source_historical_replay_gate=rules.read(root/'inherited/A65/replay_gate.json'),
        inherited_runs=10,new_completed_runs=len(fresh),new_planned_runs=140,groups=groups,
        primary=executor.primary_inference(records),
        new_cost=dict(method_forward_attempts=sum((r.get('counts') or {}).get('forward_attempts',0) for r in attempts),
            method_forwards_completed=sum((r.get('counts') or {}).get('forwards',0) for r in attempts),
            input_vjp_attempts=sum((r.get('counts') or {}).get('input_vjp_attempts',0) for r in attempts),
            optimizer_backward_attempts=sum((r.get('counts') or {}).get('optimizer_backward_attempts',0) for r in attempts),
            warmup_forward_attempts=sum(r.get('warmup',{}).get('forward_attempts',0) for r in loads),
            warmup_input_vjp_attempts=sum(r.get('warmup',{}).get('input_vjp_attempts',0) for r in loads)),
        inherited_cost=dict(method_forwards=3000,optimizer_backwards=100,diagnostic_forwards=32,warmup_input_vjp=2),
        cost_is_lower_bound=any(r['status']!='complete' for r in attempts) or any(r['warmup']['status']!='complete' for r in loads),
        model_loads=loads,limits=p['protocol']['limitations']+' User-authorized post-hoc continuation on current strong anchor; '
            'historical compatibility gate remains failed. Native repeat0 timing inherited from A65 on same GPU UUID; '
            'all other timings newly measured; no strict same-session native speed claim.')
    write(root/'summary.json',summary);executor.render_table(root,summary)
    with (root/'comparison.md').open('a') as f:
        f.write('\nA66续跑：原生基线首份输出和计时继承A65（28.226237dB），其余两份补测；'
                '其余版本三份均新增。A65历史门控仍为false，不重标通过。成本分新增与继承。\n')
    print('A66 Summary:',json.dumps(finite(summary),ensure_ascii=False),flush=True);return summary


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=rules.BRANCH
    bundle,commit=exporter.export_and_commit({'A66':Path(root)},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',commit,'Branch:',rules.BRANCH,flush=True)
    subprocess.run(['git','push','origin',rules.BRANCH],cwd=PROJECTS.parent,check=True)
    print('RESULTS_PUSH_PASS: A66 current strong anchor method comparison uploaded.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=rules.ROOT);parser.add_argument('--parent',default=design.ROOT)
    parser.add_argument('--shard',type=int,choices=(0,1));parser.add_argument('--device-uuid')
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','worker','summarize','archive'):group.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args)
    elif args.worker:worker(args)
    elif args.summarize:summarize(args.root)
    elif args.archive:archive(args.root)


if __name__=='__main__':main()
