"""Native FP32 ImageNet72-call qualification, then the frozen eight-case protocol."""
from contextlib import contextmanager,ExitStack
import gc
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

PROJECTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECTS))
from utils import a56_selection_r2 as frozen
from utils import a56_dataset_queue_session as queue
from utils.a54_session import busy,digest,exclusive,read,write

ROOT=frozen.ROOT
OUT=ROOT.parent.parent/'A56M1_20261027'
BRANCH='results/a56m1-imagenet-qualification-20261027'
CARD=PROJECTS/'experiment_cards/A56M1_imagenet_native_qualification_20261027.md'
PROFILE_CAP=72
BACK_CAP=32
MARGIN=1024**3
PROBE_MIN_FREE_MIB=8192


def own_sources():
    return {str(p):digest(p) for p in (Path(__file__),CARD,PROJECTS/'utils/a56m_imagenet_session.py',PROJECTS/'utils/test_a56m_imagenet_runtime.py',PROJECTS/'utils/test_a56m_torch_plumbing.py')}


def selected_cases(p):
    cases=[c for c in p['cases'] if c['dataset']=='imagenet']
    if len(cases)!=8 or {t:sum(c['task']==t for c in cases) for t in frozen.original.rules.TASKS}!={'super_resolution':4,'inpainting':4}:
        raise ValueError('Need original frozen eight ImageNet cases')
    return cases


def memory_fits(peak_reserved,free,reserved,margin=MARGIN):
    return bool(peak_reserved>0 and min(free,reserved)>=0 and peak_reserved+margin<=free+reserved)


def idle_queue_identity(state,command,owner,uid,pgid):
    return bool(state.get('status')=='running_or_waiting_for_memory' and not state.get('running_groups')
        and not state.get('failed_groups') and set(state.get('completed_groups',[]))=={'ffhq_0','ffhq_1'}
        and set(state.get('pending_groups',[]))=={'imagenet_0','imagenet_1'}
        and isinstance(state.get('pid'),int) and state['pid']>1 and pgid==state['pid'] and owner==uid
        and str(PROJECTS/'utils/a56_dataset_queue_session.py') in command and '--coordinator' in command)


def stop_idle_queue(p):
    """Freeze only a verified own waiter; decline if ANY child or IN attempt appeared."""
    if not queue.BASE.exists() or not busy(queue.BASE/'session.lock'):return
    state=read(queue.BASE/'session.json');pid=state.get('pid')
    if not isinstance(pid,int):raise ValueError('Invalid queue PID; no signal sent')
    proc=Path('/proc')/str(pid)
    command=[b.decode() for b in (proc/'cmdline').read_bytes().split(b'\0') if b]
    if not idle_queue_identity(state,command,proc.stat().st_uid,os.geteuid(),os.getpgid(pid)):
        raise ValueError('Old queue is not idle at ImageNet admission; follow existing queue instead')
    stopped=False
    try:
        os.kill(pid,signal.SIGSTOP);stopped=True
        for _ in range(100):
            if '\nState:\tT' in ('\n'+(proc/'status').read_text()):break
            time.sleep(.02)
        else:raise ValueError('Could not freeze idle coordinator; no new task started')
        if (proc/'task'/str(pid)/'children').read_text().strip():raise ValueError('Old coordinator has a child; no interruption')
        if any(list((ROOT/c['key']).rglob('attempt_*')) for c in selected_cases(p)):
            raise ValueError('ImageNet attempt already started; no interruption/retry')
        snap=OUT/'old_queue_snapshot';snap.mkdir(exist_ok=True)
        for n in ('session.json','session.log'):shutil.copyfile(queue.BASE/n,snap/n)
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False
        for _ in range(100):
            if not busy(queue.BASE/'session.lock'):
                write(snap/'handoff.json',dict(utc=queue.old.stamp(),pid=pid,reason='Verified idle ImageNet memory wait only; FFHQ complete; no child/IN attempt'))
                print('A56M1 HANDOFF: idle old queue stopped; completed FFHQ bytes retained.',flush=True);return
            time.sleep(.05)
        raise ValueError('Old coordinator still owns lock; no new worker')
    finally:
        if stopped:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass


def prepare(p):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    value=dict(experiment='A56M1_native_memory_qualification',source_plan_sha256=digest(ROOT/'plan.json'),
        source_plan=str(ROOT/'plan.json'),cases=selected_cases(p),profile_case=selected_cases(p)[0],
        worker_sources=own_sources(),profile_forward_cap=PROFILE_CAP,profile_input_VJP_cap=BACK_CAP,
        minimum_probe_free_MiB=PROBE_MIN_FREE_MIB,margin_bytes=MARGIN,
        limits='No GT resource selection. Native FP32256 original grid; one optimize epoch measures memory only; full original300-call cases remain mandatory.')
    if path.exists():
        if frozen.original.source.load_signed(path)!=value:raise ValueError('A56M1 source/plan changed')
    else:frozen.original.source.signed(path,value)
    shutil.copyfile(CARD,OUT/'experiment_card.md')
    return value


def load_plan():
    m=frozen.original.source.load_signed(OUT/'plan.json')
    if m['source_plan_sha256']!=digest(ROOT/'plan.json') or m['worker_sources']!=own_sources():raise ValueError('A56M1 frozen identity changed')
    return m


def record(path,r):
    frozen.original.source.signed(path,r)


def capture_memory(torch,device):
    torch.cuda.synchronize(device)
    free,total=torch.cuda.mem_get_info(device)
    return dict(free_bytes=int(free),total_bytes=int(total),allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))


@contextmanager
def parameter_loss_meter(torch,r):
    from unittest.mock import patch
    original_backward=torch.Tensor.backward;original_step=torch.optim.Adam.step
    def loss_backward(tensor,*args,**kw):
        r['optimization_loss_backwards']+=1;return original_backward(tensor,*args,**kw)
    def optimizer_step(opt,*args,**kw):
        r['optimizer_steps']+=1;return original_step(opt,*args,**kw)
    with patch.object(torch.Tensor,'backward',loss_backward),patch.object(torch.optim.Adam,'step',optimizer_step):
        yield


def profile(model,p,c,device):
    import torch
    from unittest.mock import patch
    from modules.zaps_algorithm import ZAPS
    from utils import no_learning_replay as noise
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_first_direction import model_rng_guard
    from utils.diag_zaps_sampler_parity import rng_state,same_rng
    d=frozen.original;path=OUT/'profile.json'
    if path.exists():
        r=d.source.load_signed(path)
        if r.get('status')!='complete' or not r.get('gates_passed'):raise ValueError('Prior profile failure retained; no automatic repeated probe')
        if r['plan_sha256']!=digest(OUT/'plan.json'):raise ValueError('Profile plan changed')
        return r
    r=dict(status='running',plan_sha256=digest(OUT/'plan.json'),nfe=0,calls=[],input_backward_attempts=0,
        input_backward_completed=0,optimization_loss_backwards=0,optimizer_steps=0,phase='SETUP',memory=[],metrology=[],traces=[])
    versions=[(v,v._version,v.requires_grad) for v in model.model.parameters()]
    start=time.perf_counter()
    def save():r['seconds']=time.perf_counter()-start;record(path,r)
    def count(module,args):
        if r['nfe']>=PROFILE_CAP:raise RuntimeError('A56M1 profile forward cap reached; rejected before dispatch')
        r['nfe']+=1;r['calls'].append(int(args[1].flatten()[0]));save()
        if r['nfe']%10==0:print('A56M1 PROFILE PROGRESS:',r['phase'],'F=',r['nfe'],'/72, inputB=',r['input_backward_attempts'],flush=True)
    def backward():
        if r['input_backward_attempts']>=BACK_CAP:raise RuntimeError('A56M1 input VJP cap reached')
        r['input_backward_attempts']+=1;save()
    save();hook=model.model.register_forward_pre_hook(count);guards=ExitStack()
    try:
        operator,mask=d.make_operator(c,device);r['operator_calls']={}
        for owner,name,label in ((operator,'H','unit_H'),(operator,'transpose','unit_adjoint'),(operator.base,'H','base_H_including_unit_calls')):
            fn=getattr(owner,name);r['operator_calls'][label]=0
            def counted(*args,_fn=fn,_label=label,**kw):
                r['operator_calls'][_label]+=1;return _fn(*args,**kw)
            guards.enter_context(patch.object(owner,name,counted))
        r['operator_gate']=d.a40.operator_gate(operator,device)
        y,r['measurement']=d.a40.measurement(OUT,c,operator,device)
        config=dict(p['configs']['imagenet'][c['task']],num_epochs=1)
        r['phase']='ORIGINAL_OPTIMIZE_ONE_UNROLL';save()
        torch.cuda.reset_peak_memory_stats(device)
        z=d.a40.base.baseline.snapshot_class()(model,operator,img_size=256,**config)
        z.tau=torch.tensor(p['grid'],device=device)
        with d.parent.scan.preserved_rng(),model_rng_guard(model.model,device),parameter_loss_meter(torch,r):
            set_seed(c['sampling_seed']);losses=z.optimize(y,verbose=False,x0_gt=None,scheduler=None)
        d.require(r['nfe']==30 and z._last_nfe==30 and r['optimizer_steps']==1 and r['optimization_loss_backwards']==1 and len(losses)==1 and all(math.isfinite(float(x)) for x in losses),'One native optimize unroll contract failed')
        r['memory'].append(dict(phase=r['phase'],**capture_memory(torch,device)));del z;gc.collect();torch.cuda.empty_cache()
        r['phase']='U_UNLEARNED_LAST_UNROLL';save();torch.cuda.reset_peak_memory_stats(device)
        z=ZAPS(model,operator,img_size=256,**config);z.tau=torch.tensor(p['grid'],device=device)
        positions=d.source.fields.policy.Positions(d.source.fields.policy.noise_u(model.alphas_cumprod.cpu().numpy()),list(reversed(p['grid'])))
        params={n:getattr(z,n).detach().clone() for n in ('zeta','D','tau')}
        with d.parent.scan.preserved_rng():initial,noises,_=noise.noise_from_seed(c['sampling_seed'],device)
        external=rng_state(device);x=initial
        with d.current.base.input_vjp_parameters(model.model),model_rng_guard(model.model,device),d.parent.P.precision.ieee(torch),patch('torch.randn_like',side_effect=RuntimeError('Unexpected profile sampling RNG draw')):
            for k,t in enumerate(positions.nominal):
                s=positions.nominal[k+1] if k<29 else -1
                field,gamma,row=d.current.sensed_field(z,y,x,t,s,positions,c['sigma'],1.,mask,backward)
                r['input_backward_completed']+=row.get('backward_calls',0);r['traces'].append(dict(step=k,t=t,readout=row))
                if k in d.rules.METROLOGY_STEPS:
                    with torch.no_grad():vector=operator.transpose(y-operator.H(field['clean']))
                    gate=d.current.base.vjp_metrology(model,x,t,vector,backward)
                    r['input_backward_completed']+=gate['backward_calls'];r['metrology'].append(dict(t=t,**gate))
                with torch.no_grad():x=d.parent.old.parent.rules.risk.advance(z,x,field,positions,k,noises,gamma)
                d.require(bool(torch.isfinite(x).all()),'Nonfinite U resource path')
        r['memory'].append(dict(phase=r['phase'],**capture_memory(torch,device)))
        # Insert the six extra metrology forwards after the existing center at each selected step.
        expected=list(reversed(p['grid']))
        for k,t in enumerate(positions.nominal):expected.extend([t]*(7 if k in d.rules.METROLOGY_STEPS else 1))
        d.require(r['calls']==expected and r['nfe']==72 and r['input_backward_attempts']==r['input_backward_completed'],'Profile calls/backwards contract failed')
        d.require(same_rng(external,rng_state(device)) and all(v.grad is None for v in model.model.parameters()),'Profile RNG/model gradient changed')
        d.require(all(v._version==version and v.requires_grad==flag for v,version,flag in versions),'Model weight versions/flags changed')
        d.require(all(torch.equal(getattr(z,n),v) for n,v in params.items()),'U probe learned parameters changed')
        r['gates_passed']=bool(all(t['readout'].get('resolved',False) for t in r['traces']) and all(v['passed'] for v in r['metrology']))
        r['peak_reserved_bytes']=max(v['peak_reserved_bytes'] for v in r['memory'])
        r['status']='complete';r['phase']='PROFILE_FINISHED';save()
        del z,x;gc.collect();torch.cuda.empty_cache()
        print('A56M1 PROFILE:',{k:r[k] for k in ('status','nfe','input_backward_attempts','gates_passed','peak_reserved_bytes')},flush=True)
        return r
    except BaseException as error:
        r.update(status='failed',error=repr(error),is_cuda_oom=isinstance(error,torch.cuda.OutOfMemoryError))
        try:r['memory_at_failure']=capture_memory(torch,device)
        except Exception:pass
        save();raise
    finally:hook.remove();guards.close()


@contextmanager
def execution():
    from unittest.mock import patch
    m=load_plan();d=frozen.original;pin=dict(kind='A56M1_NATIVE_RESOURCE_QUALIFICATION',plan_sha256=digest(OUT/'plan.json'))
    def tagged(writer):
        def save(path,value):
            if Path(path).name in ('run.json','summary.json','evaluation.json'):value=dict(value,imagenet_resource_execution=pin)
            if Path(path).name=='evaluation.json' and 'GT_used_only_after_all20_outputs_frozen' in value:
                value['GT_used_only_after_all20_outputs_frozen']=False
                value['GT_timing_amendment']='User authorized early FFHQ labels; ImageNet rules/cohort remained frozen before those labels; all8 IN outputs precede IN evaluation.'
            if Path(path).name=='summary.json' and value.get('experiment')=='A56_frozen_direction_transfer':
                value['limits']+=' User-authorized FFHQ evaluation preceded ImageNet completion; no original all20-before-first-label claim. Added profile cost is recorded in A56M1 summary.'
            return writer(path,value)
        return save
    with frozen.bindings(),patch.object(d,'signed',tagged(d.signed)),patch.object(d.a40.source,'signed',tagged(d.a40.source.signed)):
        yield


def cost_snapshot(p):
    rows=[]
    for c in selected_cases(p):
        for f in (ROOT/c['key']).glob('*/attempt_*/run.json'):
            r=read(f);rows.append((f,r))
    return dict(method_nfe=sum(r.get('nfe',0) for f,r in rows if r.get('kind')=='BASELINE'),
        diagnostic_nfe=sum(r.get('nfe',0) for f,r in rows if r.get('phase')=='SAMPLING'),
        input_backward_attempts=sum(r.get('backward_attempts',0) for f,r in rows),
        partial_attempts=sum(r.get('status')!='complete' for f,r in rows))


def worker():
    import torch
    from modules.main_single import load_diffusion_model
    m=load_plan();d=frozen.original;p=d.check(ROOT);device='cuda:0'
    result=dict(experiment='A56M1_imagenet_native_qualification',status='running',new_profile_nfe=0,new_profile_input_VJP=0,
        classification='NOT_YET_QUALIFIED',completed_ImageNet_cases=0,planned_ImageNet_cases=8)
    before_cost=cost_snapshot(p)
    before_profile=read(OUT/'profile.json').get('nfe',0) if (OUT/'profile.json').exists() else 0
    model=None
    try:
        actual=dict(torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(device),
            NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'))
        d.require(actual==p['runtime'] and d.parent.P.precision.precision_state(torch)==p['native_policy'] and d.a40.model_sources()==p['model_library_sources'],'Runtime/model libraries changed')
        torch.set_num_threads(2)
        with d.parent.scan.preserved_rng():
            model=load_diffusion_model('imagenet',device);model.model.eval()
        d.require(digest(model.ckpt_path)==p['checkpoints']['imagenet']['sha256'] and all(v.dtype==torch.float32 for v in model.model.parameters()),'Native FP32 checkpoint differs')
        with execution():
            prof=profile(model,p,m['profile_case'],device)
            result.update(new_profile_nfe=prof['nfe'],new_profile_input_VJP=prof['input_backward_attempts'])
            if not prof['gates_passed']:
                result.update(status='stopped',classification='METROLOGY_OR_RESOLUTION_FAILED_NO_FULL_RUN');return
            for c in selected_cases(p):
                if d.finished(ROOT,p,c):continue
                for kind,check in (('BASELINE',lambda:d.baseline_done(ROOT,p,c,'BASELINE')),('SAMPLING',lambda:d.finished(ROOT,p,c))):
                    if list((ROOT/c['key']/kind).glob('attempt_*')) and check() is None:raise ValueError('Partial/failed original attempt retained; no retry: '+c['key'])
                mem=capture_memory(torch,device)
                if not memory_fits(prof['peak_reserved_bytes'],mem['free_bytes'],mem['reserved_bytes']):
                    result.update(status='resource_limited',classification='MEASURED_PEAK_PLUS_MARGIN_NOT_AVAILABLE',admission=mem,required_peak_plus_margin=prof['peak_reserved_bytes']+MARGIN);return
                operator,mask=d.make_operator(c,device);gate=d.a40.operator_gate(operator,device)
                with d.baseline_bindings():d.a40.training(ROOT,p,c,model,device,operator,gate)
                mem=capture_memory(torch,device)
                if not memory_fits(prof['peak_reserved_bytes'],mem['free_bytes'],mem['reserved_bytes']):
                    result.update(status='resource_limited',classification='BASELINE_SAVED_SAMPLING_WAITING_FOR_MARGIN',admission=mem,
                                  required_peak_plus_margin=prof['peak_reserved_bytes']+MARGIN);return
                d.sample(ROOT,p,c,model,device,operator,mask)
                result['completed_ImageNet_cases']+=1
                print('A56M1 FULL CASE COMPLETE:',c['key'],flush=True)
                gc.collect();torch.cuda.empty_cache()
            before=sys.argv
            try:sys.argv=[d.__file__,'--archive'];d.main()
            finally:sys.argv=before
            result.update(status='complete',classification='NATIVE_RESOURCE_QUALIFIED_ALL8_IMAGENET_COMPLETED')
    except BaseException as error:
        result.update(status='failed',classification='CUDA_OOM_STOPPED' if isinstance(error,torch.cuda.OutOfMemoryError) else 'RUNTIME_OR_SCIENTIFIC_FAILURE_STOPPED',error=repr(error))
        print('A56M1 STOPPED:',repr(error),'; no automatic failed-attempt retry.',flush=True)
    finally:
        if (OUT/'profile.json').exists():
            pr=read(OUT/'profile.json');result.update(new_profile_nfe=pr.get('nfe',0),new_profile_input_VJP=pr.get('input_backward_attempts',0))
        result['original_ImageNet_recorded_cost']=cost_snapshot(p)
        after_cost=result['original_ImageNet_recorded_cost']
        result['additional_UNet_forwards_this_invocation']=result['new_profile_nfe']-before_profile+sum(after_cost[k]-before_cost[k] for k in ('method_nfe','diagnostic_nfe'))
        result['completed_ImageNet_cases']=sum(bool(d.finished(ROOT,p,c)) for c in selected_cases(p))
        result['limits']='Resource qualification only until all8 finish. New72-call profile is separate from inherited3852 ImageNet forward budget. One-epoch memory probe is not300-forward quality. No early FFHQ re-evaluation, no half precision/resolution/model change, no repeated failed attempts.'
        record(OUT/'summary.json',result)
        print('A56M1 Summary:',__import__('json').dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

