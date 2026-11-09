"""A100 detached equal-call ray/diagonal comparison on immutable A98 inputs."""
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
import argparse,contextlib,csv,shutil,statistics,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files,equal_budget_ray_diagonal as rules
from utils import diag_fast_response_refinement as parent_driver,diag_full_unroll_response as engine
from utils.a54_session import exclusive,busy

ROOT='/data0/lzy/ZAPS-results/diag_equal_budget_ray_diagonal/A100_20261109'
PARENT=parent_driver.ROOT
BRANCH='results/zaps-equal-budget-ray-diagonal-20261109'
REFERENCES=('BASE','LEGACY_SCALAR','SCALAR_K0','DIAG_K1')
NAMES=('utils/equal_budget_ray_diagonal.py','utils/diag_equal_budget_ray_diagonal.py',
 'utils/test_equal_budget_ray_diagonal.py','utils/test_equal_budget_ray_diagonal_runtime.py',
 'run_a100_20261109.sh','experiment_cards/A100_equal_budget_ray_diagonal_20261109.md')


def check(root):
    p=files.read(root/'plan.json');files.require(p['experiment']=='A100_equal_budget_ray_diagonal','Wrong A100 plan')
    for name,sha in p['sources'].items():files.require(files.digest(PROJECTS/name)==sha,'Source changed: '+name)
    for name,sha in p['inputs'].items():files.require(files.digest(name)==sha,'Input changed: '+name)
    return p


def bind_record(path,bound):
    row=files.read(path);files.require(row['status']=='complete','Incomplete immutable reference: '+str(path))
    bound[str(path)]=files.digest(path)
    for name,sha in row['artifacts'].items():
        artifact=path.parent/name;files.require(files.digest(artifact)==sha,'Reference artifact changed')
        bound[str(artifact)]=sha
    return str(path)


def prepare(root,parent):
    if (root/'plan.json').exists():return check(root)
    files.require(not root.exists() or not any(root.iterdir()),'Unplanned nonempty A100 root retained')
    pp=parent_driver.check(parent);summary=files.read(parent/'summary.json')
    files.require(summary['completed_runs']==summary['planned_runs']==48,'Completed A98 required')
    bound=dict(pp['inputs']);bound[str(parent/'plan.json')]=files.digest(parent/'plan.json');bound[str(parent/'summary.json')]=files.digest(parent/'summary.json')
    cases=[]
    for c0 in pp['cases']:
        c=dict(c0);c['comparison_references']={}
        for arm in ('BASE','LEGACY_SCALAR'):
            c['comparison_references'][arm]=bind_record(Path(c['a97_references'][arm]),bound)
        for arm in ('SCALAR_K0','DIAG_K1'):
            c['comparison_references'][arm]=bind_record(parent/'cases'/c['key']/arm/'run.json',bound)
        cases.append(c)
    files.require(len(cases)==12 and all(c['sigma']==.05 for c in cases),'A100 requires same12 sigma.05 cases')
    root.mkdir(parents=True)
    protocol=dict(
      scope='Original3 FFHQ x4tasks sigma_model.05. Same A98 cached nine-unroll learned entry, observations, seeds, original30 positions, zeta/D and DDPM innovations. Two new final-unroll policies,24 runs; no new image/noise/grid/model.',
      target='Same A98 next-reference objective F_s=PH(clamped clean_s), r=Py-F_s(b), omega=(1-Msigma^2/||r||^2)+, target=F_s(b)+omega*r. Same frozen-epsilon closed scalar initializer in both arms; known-noise heuristic, not calibrated final-image risk.',
      factor='RAY_K1_EQ uses a nonnegative one-dimensional native-q parameter. DIAG_K1_EQ uses all original nonzero wavelet coefficients. Both perform one actual-response gradient/JVP update. This isolates parameter space, not response target or initialization.',
      exact_cost='At every s>=0 position BOTH arms execute exactly one VJP, one JVP and two candidate value forwards, even when first candidate is accepted. Per suffix exactly175 model forwards,29 inputVJP,29 inputJVP; terminal operator derivatives separate. Full equivalent method F=270+175=445. No CG/optimizer/extra regularizer/task switch.',
      pairing='Independent own trajectory from same entry/RNG. Controller consumes no sampling RNG; original29 innovations/final RNG/grid required equal. All calls/rejected checks retained. GT read only after30 actions. No output or quality fallback.',
      references='A98 BASE/LEGACY_SCALAR/SCALAR_K0/DIAG_K1 are immutable descriptive references with their own recorded costs. Primary causal comparison is new DIAG_K1_EQ versus new RAY_K1_EQ; do not call native references equal-NFE.',
      decision='Per task DIAG-RAY requires meanPSNR>=+.1dB, >=2/3 PSNR wins, SSIM nondecrease and LPIPS nonincrease. SR failure triggers a separate preregistered observable-anchor experiment later, not an automatic task-specific switch here.',
      limits='Small reused development panel, one seed, sigma.05. Tests equal-call anisotropic freedom after a fixed scalar initializer; no independent transfer, universal superiority, NFE saving, curvature or posterior guarantee.',
      resources='Up to2 explicit GPUs>=8GiB; real checkpoint primal/adjoint and fixed-call runtime tests mandatory. Step-boundary resume only. No process termination/download/precision change/old-source edit/results merge.')
    p=dict(experiment='A100_equal_budget_ray_diagonal',parent=str(parent),a71_parent=pp['a71_parent'],cases=cases,
      checkpoint=pp['checkpoint'],grid=pp['grid'],arms=list(rules.ARMS),budget=rules.budget(),
      sources={**pp['sources'],**{name:files.digest(PROJECTS/name) for name in NAMES}},inputs=bound,protocol=protocol)
    files.write(root/'plan.json',p);shutil.copyfile(PROJECTS/'experiment_cards/A100_equal_budget_ray_diagonal_20261109.md',root/'experiment_card.md')
    return p


class LabelStream:
    def __init__(self,stream):self.stream=stream
    def write(self,value):return self.stream.write(value.replace('A97','A100'))
    def flush(self):return self.stream.flush()


def worker(root,index):
    with patch.multiple(engine,check=check,rules=rules),contextlib.redirect_stdout(LabelStream(sys.stdout)):
        engine.worker(root,index)


def summarize(root):
    p=check(root);new=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in new if r['status']=='complete'];refs=[]
    for c in p['cases']:
        for path in c['comparison_references'].values():refs.append(files.read(path))
    groups=[];comparisons=[];all_rows=[*refs,*done]
    for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
        for arm in (*REFERENCES,*rules.ARMS):
            rows=[r for r in all_rows if r['case']['task']==task and r['arm']==arm]
            if rows:
                groups.append(dict(task=task,arm=arm,images=len(rows),quality={k:statistics.mean(r['metrics'][k] for r in rows) for k in ('psnr','ssim','lpips')},
                    mean_suffix_seconds=statistics.mean(r['suffix_seconds'] for r in rows),
                    mean_counts={k:statistics.mean(r['counts'].get(k,0) for r in rows) for k in ('forward_attempts','input_vjp_attempts','input_jvp_attempts')},reused=arm in REFERENCES))
        candidates=[r for r in done if r['case']['task']==task and r['arm']=='DIAG_K1_EQ'];by={r['case']['key']:r for r in done if r['case']['task']==task and r['arm']=='RAY_K1_EQ'}
        pairs=[(by[r['case']['key']],r) for r in candidates]
        if pairs:
            delta={k:statistics.mean(b['metrics'][k]-a['metrics'][k] for a,b in pairs) for k in ('psnr','ssim','lpips')}
            counts_equal=all(all(a['counts'].get(k,0)==b['counts'].get(k,0) for k in ('forward_attempts','input_vjp_attempts','input_jvp_attempts')) for a,b in pairs)
            comparisons.append(dict(task=task,candidate='DIAG_K1_EQ',reference='RAY_K1_EQ',images=len(pairs),delta=delta,
                psnr_wins=sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs),counts_equal=counts_equal,
                positive=len(pairs)==3 and counts_equal and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0 and sum(b['metrics']['psnr']>a['metrics']['psnr'] for a,b in pairs)>=2))
    for row in done:
        files.require(row['counts']['forward_attempts']==175,'Non-equal A100 forward count')
        files.require(row['counts']['input_vjp_attempts']==29 and row['counts']['input_jvp_attempts']==29,'Non-equal A100 derivative count')
    summary=dict(experiment=p['experiment'],classification='PAIRED_EQUAL_NEURAL_CALL_RAY_VS_DIAGONAL',completed_runs=len(done),planned_runs=24,
      groups=groups,comparisons=comparisons,actual_new_cost={k:sum(r.get('counts',{}).get(k,0) for r in new) for k in ('forward_attempts','input_vjp_attempts','input_jvp_attempts','optimizer_backward_attempts','terminal_operator_vjp_attempts','terminal_operator_jvp_attempts')},
      cost_is_lower_bound=len(done)<24 or any(r['status'] in ('running','failed','resource_limited') for r in new),protocol=p['protocol'])
    files.write(root/'summary.json',summary)
    with (root/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['task','arm','images','PSNR','SSIM','LPIPS','suffix_seconds','forwards','VJP','JVP','reused'])
        for r in groups:writer.writerow([r['task'],r['arm'],r['images'],*[r['quality'][k] for k in ('psnr','ssim','lpips')],r['mean_suffix_seconds'],*[r['mean_counts'][k] for k in ('forward_attempts','input_vjp_attempts','input_jvp_attempts')],r['reused']])
    print('A100 STATUS:',len(done),'/24',flush=True)
    for row in comparisons:print('A100 EQUAL-CALL DIAG_MINUS_RAY:',row,flush=True)
    return summary


def archive(root):
    from utils import archive_refinement_confirmation as arch
    arch.BRANCH=BRANCH;bundle,sha=arch.export_and_commit({'A100':root},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A100 uploaded.',flush=True)


def coordinate(args,root,parent,state,record):
    children=[];streams=[]
    with exclusive(state/'worker.lock'):
        status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
        try:
            prepare(root,parent)
            from utils.diag_paper_clock_alignment import gpu_inventory
            choices=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'))
            choices=sorted([d for d in choices if d['free_bytes']>=8*files.GIB],key=lambda d:-d['free_bytes'])
            allocation=root/'allocation.json'
            if allocation.exists():
                devices=files.read(allocation)['devices']
                if not all(any(c['uuid']==d['uuid'] for c in choices) for d in devices):raise engine.ResourcePause('Pinned GPU lacks8GiB margin')
            else:
                if not choices:raise engine.ResourcePause('No allocated GPU>=8GiB')
                devices=choices[:2];files.write(allocation,dict(devices=devices))
            command=[sys.executable,'-m','unittest','-v','utils.test_equal_budget_ray_diagonal','utils.test_equal_budget_ray_diagonal_runtime','utils.test_full_unroll_response_runtime']
            result=subprocess.run(command,cwd=PROJECTS,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':devices[0]['uuid']})
            print(result.stdout+result.stderr,end='',flush=True);files.write(root/'preflight_tests.json',dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr));result.check_returncode()
            for index in range(len(devices)):
                folder=root/'workers'/str(index);folder.mkdir(parents=True,exist_ok=True);log=folder/'worker.log';stream=log.open('ab',buffering=0);streams.append(stream)
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--root',str(root),'--worker-index',str(index)],cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                children.append(dict(child=child,log=log,offset=log.stat().st_size,index=index))
            while children:
                alive=False
                for item in children:
                    with item['log'].open('rb') as stream:stream.seek(item['offset']);chunk=stream.read();item['offset']=stream.tell()
                    if chunk:print(chunk.decode(errors='replace'),end='',flush=True)
                    if item['child'].poll() is None:alive=True
                if not alive:break
                time.sleep(1)
            statuses=[files.read(root/'workers'/str(i)/'session.json')['status'] for i in range(len(devices))]
            if any(item['child'].returncode for item in children) or 'failed' in statuses:raise RuntimeError('A100 worker failed; records retained, no automatic retry')
            status['status']='resource_limited' if 'resource_limited' in statuses else 'complete'
            if status['status']=='complete':files.require(sum(files.read(f)['status']=='complete' for f in (root/'cases').glob('*/*/run.json'))==24,'Incomplete A100 runs')
        except engine.ResourcePause as error:status.update(status='resource_limited',reason=str(error));print('A100 RESOURCE PAUSE:',error,flush=True)
        except BaseException as error:status.update(status='failed',error=repr(error),traceback=traceback.format_exc());print('A100 FAILED:',status,flush=True)
        finally:
            for item in children:item['child'].wait()
            for stream in streams:stream.close()
            files.write(record,status)
            if (root/'plan.json').exists():
                shutil.copyfile(record,root/'session.json')
                try:summarize(root);archive(root)
                except Exception as error:print('A100 archive incomplete; use --archive-only:',repr(error),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default=ROOT);parser.add_argument('--a98-root',default=PARENT);parser.add_argument('--gpus',default='0,1');parser.add_argument('--worker-index',type=int)
    for flag in ('execute','status','archive-only'):parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();parent=Path(args.a98_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
    if args.worker_index is not None:
        folder=root/'workers'/str(args.worker_index);folder.mkdir(parents=True,exist_ok=True)
        with exclusive(folder/'worker.lock'):worker(root,args.worker_index)
        if files.read(folder/'session.json')['status']=='failed':raise SystemExit(1)
        return
    if args.status:
        print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None,workers=[files.read(f) for f in (root/'workers').glob('*/session.json')]))
        for path in sorted((root/'cases').glob('*/*/run.json')):
            row=files.read(path);print(row['case']['key'],row['arm'],row['status'],str(row['completed_steps'])+'/30','F=',row['counts'].get('forward_attempts',0),'VJP=',row['counts'].get('input_vjp_attempts',0),'JVP=',row['counts'].get('input_jvp_attempts',0))
        return
    if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');summarize(root);archive(root);return
    if args.execute:coordinate(args,root,parent,state,record);return
    with exclusive(state/'client.lock'):
        child=None
        if not busy(state/'worker.lock'):
            if record.exists():
                prior=files.read(record);files.require(prior['status'] in ('complete','resource_limited'),'Prior failure retained; no automatic retry')
                if prior['status']=='complete':print('A100 already complete; use --status / --archive-only');return
            command=[sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a98-root',str(parent),'--gpus',args.gpus]
            with (state/'worker.log').open('ab',buffering=0) as log:child=subprocess.Popen(command,cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        offset=0
        while True:
            if (state/'worker.log').exists():
                with (state/'worker.log').open('rb') as stream:stream.seek(offset);chunk=stream.read();offset=stream.tell()
                if chunk:print(chunk.decode(errors='replace'),end='',flush=True)
            if not busy(state/'worker.lock') and (child is None or child.poll() is not None):break
            time.sleep(1)
        if record.exists():
            status=files.read(record);print('A100 SESSION:',status,flush=True)
            if status['status']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
