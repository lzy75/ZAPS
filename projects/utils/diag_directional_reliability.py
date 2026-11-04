"""A91 direction-only native-guidance reliability shrinkage; paired cached-prefix qualification."""
import argparse
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
from pathlib import Path
import csv,json,subprocess,sys,time,traceback
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files
from utils import directional_reliability as rules
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_directional_reliability/A91_20261104'
PARENT='/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029'
CARD=PROJECTS/'experiment_cards/A91_directional_reliability_20261104.md'
BRANCH='results/zaps-directional-reliability-20261104'
class ResourcePause(RuntimeError):pass

def check(root):
    p=files.read(root/'plan.json')
    files.require(p['experiment']=='A91_directional_reliability','Wrong experiment plan')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
    for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Parent changed: '+n)
    return p

def resolve_artifact(record,name,bound):
    record=Path(record)
    for _ in range(8):
        row=files.read(record);files.require(row['status']=='complete','Incomplete cached source')
        bound[str(record)]=files.digest(record)
        f=record.parent/name
        if f.exists():
            files.require(name in row['artifacts'] and files.digest(f)==row['artifacts'][name],
                          'Unbound/changed source artifact: '+str(f))
            bound[str(f)]=files.digest(f);return str(f)
        files.require('source_record' in row,'Missing cached binary: '+str(f))
        source=Path(row['source_record'])
        files.require(files.digest(source)==row['source_record_sha256'],'Reused source record changed')
        record=source
    raise ValueError('Too many source-record indirections')


def prepare(root,parent):
    if (root/'plan.json').exists():return check(root)
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned directory retained')
    from utils.diag_shrink_multitask import check as parent_check
    p=parent_check(parent);s=files.read(parent/'summary.json')
    files.require(s['completed_runs']==s['planned_runs']==72,'Completed A71 required')
    cases=[dict(c) for c in p['cases'] if c['sigma'] in rules.SIGMAS]
    files.require(len(cases)==24 and {c['image_id'] for c in cases}=={'00226','00443','00963'},'Original3 x4tasks x2noise required')
    bound={str(parent/'plan.json'):files.digest(parent/'plan.json'),str(parent/'summary.json'):files.digest(parent/'summary.json')}
    for c in cases:
        c['entry']=resolve_artifact(parent/'cases'/c['key']/'BASE/run.json','entry.pt',bound)
        c['references']={arm:resolve_artifact(parent/'cases'/c['key']/arm/'run.json','paired.pt',bound) for arm in ('BASE','SHRINK')}
        folder=parent/'observations'/c['key']
        c['observation']=str(folder/'y.pt');bound[c['observation']]=files.digest(c['observation'])
        if c['task']=='inpainting':bound[str(folder/'mask.pt')]=files.digest(folder/'mask.pt')
        bound[c['image']]=files.digest(c['image'])
    bound[str(parent/'motion_kernel.npy')]=files.digest(parent/'motion_kernel.npy')
    names=['utils/directional_reliability.py','utils/diagonal_feedback.py','utils/test_mask_stage_response_runtime.py','utils/diag_directional_reliability.py','utils/test_directional_reliability.py',
      'utils/test_directional_reliability_runtime.py','utils/noise_signal_shrink.py','utils/shrink_multitask.py','utils/shrink_multitask_runtime.py',
      'utils/diag_shrink_multitask.py','modules/zaps_algorithm.py','modules/degradations.py','modules/wavelet.py','modules/diffusion_model.py',
      'utils/metrics.py','utils/paper_clock_runtime.py','utils/a67_model_freeze.py','utils/terminal_mask_counterfactual.py',
      'utils/historical_ffhq_runtime.py','utils/diag_ffhq_regression.py','../DPS/util/resizer.py','utils/historical_ffhq_gain.py','run_a91_20261104.sh',str(CARD.relative_to(PROJECTS))]
    calibrations={}
    for c in cases:
        c['noise_bank_key']=rules.calibration_key(c)
        calibrations.setdefault(c['noise_bank_key'],c['key'])
    files.require(len(calibrations)==6,'Expected3 fixed mappings plus3 masks')
    root.mkdir(parents=True)
    plan=dict(experiment='A91_directional_reliability',parent=str(parent),cases=cases,checkpoint=p['checkpoint'],calibrations=calibrations,
        inputs=bound,sources={n:files.digest(PROJECTS/n) for n in names},arms=list(rules.ARMS),
        budget=dict(paired_suffix_forwards=2160,input_vjp=0,optimizer_backward=0,warmup_per_load=1,
                    probe_UNet=0,noise_banks=6,noise_probes_per_bank=16,calibration_native_mapping_actions=96,
                    frozen_noise_identity_mapping_actions=24,calibration_shape_H_actions=6),
        protocol=dict(scope='Same3 FFHQ images x4tasks x sigma.05/.10; same completedA71 cached nine-unroll prefix, learned zeta/D, original tau and RNG. Three arms only; no new image/seed/model.',
          rule='DIRECTIONAL uses only each band E=||Wq||^2 and N=sigma^2 sum ell^2 diag(W T T^T W^T), w=(1-N/E)+. Original native q is shrunk in db4/level3 coordinates. No global gamma/projection/noise weight/scalar radius/anchor/fallback. Native zeta amplitude retained.',
          mapping='T is actual native correction mapping. SR legacy bicubic is not renamed exact H-adjoint. Other task mappings unchanged. ell=zeta/sqrt(alpha)*(1+(1-alpha)D), signed D squared through ell for nonnegative variance; never interpret D as covariance.',
          noise='Known measurement sigma only. Independent16 unit Rademacher probes estimate coefficient variance per fixed operator/mask once. Unit variance reused over images/noise/t; split8/8 noise estimates logged, no quality-based K selection.',
          assumptions='Noise propagation freezes current clean prediction and learned parameters: nominal pure-measurement-noise contribution only. Actual m(y),D(y),zeta(y) dependencies and diffusion-prediction uncertainty are not calibrated. No unbiased final-image risk/SURE/posterior coverage claim.',
          calibration='6 mappings: SR/Gaussian/Motion shared,3 fixed masks.96 native T+96 db4 analysis,0UNet. Each24 cached entry additionally verifies frozen-noise coefficient identity at first/lastt with1T,3analysis,2synthesis;0UNet. Independent RNG does not consume sampling RNG.',
          cost='All arms30F; directional online adds one analysis of native q and at most one synthesis,0 extra H/T/UNet/VJP/optimizer and no QR/NNLS. One-time noise calibration cost explicitly separate; actual transform count/time retained.',
          coordinates='Same original db4/level3/periodic coefficient packing, LL3 plusLH/HL/HH levels1..3, ten weights shared acrossRGB/positions per band. Weights0..1 are noise-retention fractions, not a0..2 bound on fitted gamma.',
          baseline='BASE_CHECK original ZAPS; SCALAR_CHECK old noise-corrected projection is comparator only and never read by directional rule. Weight1 exactly gives native q, weight0 exactzero. Multiplying native zeta scales update rather than being canceled by a new projection.',
          gates='Original BASE/SCALAR replay relativeL2<=1e-6 before directional; exactly30F/zeroB and same finalCPU/CUDA RNG; field finite/nonnegative, model unchanged. Required real Torch/operator-noise/RNG tests, no skip. No numerical or quality-based fallback.',
          timing='30-call suffix includes given-y/entry/noise-bank load, native operator/init/sampling/controller/outputCPU sync; one-time noise calibration/model load/warmup/metrics/writes separate. Full method remains300F; no independent30-call reconstruction or complete clock claim.',
          GT='Only final metrics and observed/missing region error, never energy/noise estimate or weights.',
          decision='Per task/noise compare to BOTH original and scalar: PSNRmean>=+.1dB,>=2/3 wins,SSIM nondecrease,LPIPS nonincrease. Report failures/tradeoffs. Native-only shrinkage need not match a projection rule that also amplifies; no amplitude rescue or per-task tuning.',
          limits='Small reused panel. Tests direction-specific noise-retention mechanism independent of scalar rule, not general matrix superiority or calibrated posterior uncertainty. No new image/seed/dataset or additional policy variants.'))
    files.write(root/'plan.json',plan);return plan


def archive(root):
    from utils import archive_refinement_confirmation as e
    e.BRANCH=BRANCH;bundle,sha=e.export_and_commit({'A91':root},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A91 uploaded.',flush=True)

def prepare_noise_calibrations(root,p,parent,device):
    import torch
    from modules.wavelet import OrthogonalDWT2D
    from modules.diffusion_model import _precompute_diffusion_params
    from utils.diag_shrink_multitask import make_operator
    dwt=OrthogonalDWT2D('db4',3).to(device)
    for key,case_key in p['calibrations'].items():
        c=next(c for c in p['cases'] if c['key']==case_key)
        folder=root/'noise_banks'/key;meta=folder/'record.json'
        if meta.exists():
            row=files.read(meta);files.require(row['status']=='complete','Interrupted calibration retained')
            files.require(row['plan_sha256']==files.digest(root/'plan.json') and files.digest(folder/'bank.pt')==row['sha256'],'Noise bank changed')
            continue
        folder.mkdir(parents=True,exist_ok=False);op=make_operator(parent,c,device)
        shape=tuple(op.H(torch.zeros(1,3,256,256,device=device)).shape)
        row=dict(status='running',plan_sha256=files.digest(root/'plan.json'),shape_H_actions=1,native_mapping_attempts=0)
        files.write(meta,row)
        try:
            def progress(n):row['native_mapping_attempts']=n;files.write(meta,row)
            bank,detail=rules.calibrate(op,dwt,shape,key,device,progress)
            torch.save(bank,folder/'bank.pt')
            row.update(detail,status='complete',shape=list(shape),sha256=files.digest(folder/'bank.pt'))
            files.write(meta,row);print('A91 NOISE BANK:',key,'T=',detail['native_mapping_actions'],'NN=0',flush=True)
        except BaseException as e:
            row.update(status='failed',error=repr(e));files.write(meta,row);raise
    alphas=_precompute_diffusion_params()['alphas_cumprod'].to(device)
    for c in p['cases']:
        f=root/'noise_qualification'/(c['key']+'.json')
        if f.exists():files.require(files.read(f)['status']=='complete','Interrupted noise qualification retained');continue
        bankmeta=files.read(root/'noise_banks'/c['noise_bank_key']/'record.json')
        op=make_operator(parent,c,device);entry=torch.load(c['entry'],map_location='cpu',weights_only=True)
        row=dict(status='running',native_mapping_attempts=0);files.write(f,row)
        try:
            def progress(n):row['native_mapping_attempts']=n;files.write(f,row)
            row.update(rules.qualify(op,dwt,bankmeta['shape'],c['noise_bank_key'],entry,alphas,device,progress),status='complete')
            files.write(f,row)
        except BaseException as e:row.update(status='failed',error=repr(e));files.write(f,row);raise
    print('A91 NOISE PROPAGATION GATES PASS:6 banks +24 frozen entries;0UNet.',flush=True)


def execute(root,parent):
    import torch
    from modules.main_single import load_diffusion_model,load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.a67_model_freeze import frozen_network
    from utils import paper_clock_runtime as clock
    from utils import historical_ffhq_runtime as precision
    from utils.terminal_mask_counterfactual import region_errors
    from utils.diag_noise_feasible_zaps import metrics
    from utils.diag_shrink_multitask import make_operator
    p=check(root);torch.set_num_threads(2)
    with torch.no_grad(),precision.precision('LEGACY_IEEE'):prepare_noise_calibrations(root,p,parent,'cuda:0')
    files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Wrong existing FFHQ checkpoint')
    start=time.perf_counter();model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
    state=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=True);model.model.load_state_dict(state,strict=True);del state
    torch.cuda.synchronize()
    loads=root/'loads';loads.mkdir(exist_ok=True);load_file=loads/(str(len(list(loads.glob('*.json')))).zfill(3)+'.json')
    probe=dict(warmup_forward_attempts=0,load_seconds=time.perf_counter()-start)
    def load(f):return torch.load(f,map_location='cpu',weights_only=True)
    with frozen_network(model.model),torch.no_grad(),precision.precision('LEGACY_IEEE'):
        probe['warmup_forward_attempts']=1;files.write(load_file,probe)
        model._predict_eps(torch.zeros(1,3,256,256,device=model.device),torch.tensor([999],device=model.device))
        for c in p['cases']:
            gt=load_image_as_tensor(c['image']).cpu()
            for arm in rules.ARMS:
                folder=root/'cases'/c['key']/arm;record=folder/'run.json'
                if record.exists():
                    old=files.read(record);files.require(old['status']=='complete','Prior failed/interrupted attempt retained')
                    for n,h in old['artifacts'].items():files.require(files.digest(folder/n)==h,'A91 artifact changed')
                    continue
                free=int(torch.cuda.mem_get_info()[0])
                if free+torch.cuda.memory_reserved()<6*files.GIB:raise ResourcePause('GPU headroom lost before unstarted attempt')
                folder.mkdir(parents=True);row=dict(status='running',case=c,arm=arm,counts={});files.write(record,row)
                actions=dict(diagonal_H=0)
                try:
                    def tick(count):row['counts']=count;files.write(record,row)
                    with clock.counters(model.model,tick) as count:
                        def operation():
                            entry=load(c['entry']);y=load(c['observation'])['y'].to(model.device)
                            op=make_operator(parent,c,model.device,count)
                            bank=load(root/'noise_banks'/c['noise_bank_key']/'bank.pt') if arm=='DIRECTIONAL' else None
                            return rules.run(model,op,y,entry,c,arm,actions,bank=bank)
                        (output,detail),seconds=clock.clocks(torch.cuda.synchronize,operation)
                    files.require(count['forward_attempts']==count['forwards']==30 and count['calls']==files.grid()[::-1],'Original30 NN grid/count changed')
                    files.require(count['input_vjp_attempts']==count['optimizer_backward_attempts']==0,'Unexpected backwards')
                    if arm in ('BASE_CHECK','SCALAR_CHECK'):
                        reference=load(c['references']['BASE' if arm=='BASE_CHECK' else 'SHRINK'])
                        error=float((output.double()-reference.double()).norm()/reference.double().norm().clamp(min=1e-30))
                        files.require(error<=1e-6,'Cached output equivalence failed');row['replay_relative_error']=error
                    rng=detail.pop('final_cpu_rng');cuda=detail.pop('final_cuda_rng')
                    torch.save(dict(cpu=rng,cuda=cuda),folder/'final_rng.pt')
                    if arm!='BASE_CHECK':
                        other=load(root/'cases'/c['key']/'BASE_CHECK/final_rng.pt')
                        files.require(torch.equal(other['cpu'],rng) and len(other['cuda'])==len(cuda) and all(torch.equal(a,b) for a,b in zip(other['cuda'],cuda)),'Final RNG pairing failed')
                    row['lpips_evaluation_attempts']=1;files.write(record,row);start=time.perf_counter()
                    m=metrics(output,gt)
                    if c['task']=='inpainting':
                        mask=load(parent/'observations'/c['key']/'mask.pt');m.update(region_errors(output,gt,mask))
                    row['metrics_seconds']=time.perf_counter()-start
                    torch.save(output,folder/'output.pt');tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png')
                    files.write(folder/'trace.json',detail['trace'])
                    row.update(status='complete',counts=count,metrics=m,suffix_seconds=seconds,operator_actions=actions,
                        artifacts={n:files.digest(folder/n) for n in ('output.pt','output.png','trace.json','final_rng.pt')})
                    files.write(record,row);print('A91 COMPLETE:',c['key'],arm,'F=',count['forwards'],'PSNR=',m['psnr'],'suffix_seconds=',seconds,flush=True)
                except BaseException as error:
                    row.update(status='failed',error=repr(error),operator_actions=actions,counts=locals().get('count',{}),traceback=traceback.format_exc());files.write(record,row);raise
    summarize(root)

def summarize(root):
    import statistics
    p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[];comparisons=[]
    for task in ('super_resolution','gaussian_deblur','inpainting','motion_deblur'):
        for sigma in rules.SIGMAS:
            for arm in rules.ARMS:
                rs=[r for r in done if r['arm']==arm and r['case']['task']==task and r['case']['sigma']==sigma]
                if rs:groups.append(dict(task=task,sigma=sigma,arm=arm,images=len(rs),quality={k:statistics.mean(r['metrics'][k] for r in rs) for k in ('psnr','ssim','lpips')},
                    mean_suffix_seconds=statistics.mean(r['suffix_seconds'] for r in rs)))
            candidate='DIRECTIONAL'
            candidates=[r for r in done if r['arm']==candidate and r['case']['task']==task and r['case']['sigma']==sigma]
            for baseline in ('BASE_CHECK','SCALAR_CHECK'):
                reference={r['case']['key']:r for r in done if r['arm']==baseline}
                paired=[(reference[r['case']['key']],r) for r in candidates if r['case']['key'] in reference]
                if paired:
                    delta={k:statistics.mean(n['metrics'][k]-b['metrics'][k] for b,n in paired) for k in ('psnr','ssim','lpips')}
                    wins=sum(n['metrics']['psnr']>b['metrics']['psnr'] for b,n in paired)
                    comparisons.append(dict(task=task,sigma=sigma,cases=len(paired),candidate=candidate,baseline=baseline,candidate_minus_reference=delta,
                        images_PSNR_positive=wins,positive_development_candidate=len(paired)==3 and wins>=2 and delta['psnr']>=.1 and delta['ssim']>=0 and delta['lpips']<=0))
    bank_records=[files.read(f) for f in (root/'noise_banks').glob('*/record.json')]
    qualification=[files.read(f) for f in (root/'noise_qualification').glob('*.json')]
    s=dict(experiment=p['experiment'],classification='PAIRED_PREFIX_DIRECTION_ONLY_FROZEN_NOISE_RELIABILITY',completed_runs=len(done),planned_runs=72,groups=groups,comparisons=comparisons,
      recorded_diagnostic_forwards=sum(r.get('counts',{}).get('forward_attempts',0) for r in rows),
      recorded_input_vjp_attempts=sum(r.get('counts',{}).get('input_vjp_attempts',0) for r in rows),
      recorded_optimizer_backward_attempts=sum(r.get('counts',{}).get('optimizer_backward_attempts',0) for r in rows),
      operator_H=sum(r.get('counts',{}).get('base_H_actions',0) for r in rows),
      calibration=dict(banks=len(bank_records),native_mapping_attempts=sum(r.get('native_mapping_attempts',0) for r in bank_records),
          shape_H_actions=sum(r.get('shape_H_actions',0) for r in bank_records),seconds=sum(r.get('seconds',0) for r in bank_records),
          qualification_mapping_attempts=sum(r.get('native_mapping_attempts',0) for r in qualification),
          qualifications_complete=sum(r['status']=='complete' for r in qualification),UNet=0),
      directional_steps=sum(r.get('operator_actions',{}).get('reliability_steps',0) for r in rows),
      module_dwt_analysis=sum(r.get('operator_actions',{}).get('module_dwt_analysis',0) for r in rows),
      module_dwt_synthesis=sum(r.get('operator_actions',{}).get('module_dwt_synthesis',0) for r in rows),
      warmup_forwards=sum(files.read(f).get('warmup_forward_attempts',0) for f in (root/'loads').glob('*.json')),
      cost_is_lower_bound=any(r['status']!='complete' for r in rows+bank_records+qualification),limits=p['protocol'])
    files.write(root/'summary.json',s)
    if done:
        table=[dict(task=r['case']['task'],sigma=r['case']['sigma'],image_id=r['case']['image_id'],arm=r['arm'],
            **r['metrics'],suffix_seconds=r['suffix_seconds']) for r in done]
        keys=list(dict.fromkeys(k for r in table for k in r))
        with (root/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(table)
    print('A91 Summary:',json.dumps(s,ensure_ascii=False),flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',default=ROOT);a.add_argument('--a71-root',default=PARENT);a.add_argument('--gpus',default='0,1')
    for f in ('execute','status','archive-only'):a.add_argument('--'+f,action='store_true')
    args=a.parse_args();root=Path(args.root).resolve();parent=Path(args.a71_root).resolve();state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
    if args.status:print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None));return
    if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');summarize(root);archive(root);return
    if args.execute:
        with exclusive(state/'worker.lock'):
            status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
            try:
                prepare(root,parent)
                from utils.diag_paper_clock_alignment import gpu_inventory
                candidates=files.available_gpus(gpu_inventory(),args.gpus,os.environ.get('CUDA_VISIBLE_DEVICES'))
                candidates=sorted([r for r in candidates if r['free_bytes']>=6*files.GIB],key=lambda r:-r['free_bytes'])
                if not candidates:status['status']='resource_limited';print('A91 no allocated GPU>=6GiB free; no model started.',flush=True)
                else:
                    if (root/'allocation.json').exists():
                        assigned=files.read(root/'allocation.json')
                        if not any(r['uuid']==assigned['uuid'] for r in candidates):raise ResourcePause('Pinned GPU lacks margin; resume when free')
                    else:assigned=candidates[0];files.write(root/'allocation.json',assigned)
                    os.environ['CUDA_VISIBLE_DEVICES']=assigned['uuid']
                    subprocess.run([sys.executable,'-m','unittest','-v','utils.test_directional_reliability','utils.test_directional_reliability_runtime'],cwd=PROJECTS,check=True)
                    execute(root,parent);status['status']='complete'
            except ResourcePause as e:status.update(status='resource_limited',reason=str(e));print('A91 RESOURCE PAUSE:',str(e),flush=True)
            except BaseException as e:status.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('A91 FAILED:',repr(e),flush=True)
            finally:
                files.write(record,status)
                if (root/'plan.json').exists():
                    import shutil
                    shutil.copyfile(record,root/'session.json')
                    try:summarize(root);archive(root)
                    except subprocess.CalledProcessError:print('Upload failed; retained; --archive-only retries upload.',flush=True)
        return
    with exclusive(state/'client.lock'):
        process=None
        if not busy(state/'worker.lock'):
            if record.exists():
                old=files.read(record);files.require(old['status']!='failed','Prior failure retained; no automatic retry')
                if old['status']=='complete':print('A91 already complete:',root/'summary.json');return
            with (state/'worker.log').open('ab',buffering=0) as log:process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a71-root',str(parent),'--gpus',args.gpus],cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        offset=0
        while True:
            if (state/'worker.log').exists():
                with (state/'worker.log').open(errors='replace') as f:f.seek(offset);v=f.read();offset=f.tell()
                if v:print(v,end='',flush=True)
            if not busy(state/'worker.lock') and (process is None or process.poll() is not None):break
            time.sleep(1)
        if record.exists():
            row=files.read(record);print('A91 SESSION:',row,flush=True)
            if row['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
