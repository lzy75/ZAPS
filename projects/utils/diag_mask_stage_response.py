"""A87 high-noise ZAPS inpainting: 450 paired suffix calls +9 response probes."""
import argparse
import os
os.environ['MKL_THREADING_LAYER']='GNU';os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
from pathlib import Path
import csv,json,subprocess,sys,time,traceback
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files
from utils import mask_stage_response as rules
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_mask_stage_response/A87_20261104'
PARENT='/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029'
CARD=PROJECTS/'experiment_cards/A87_high_noise_mask_stage_response_20261104.md'
BRANCH='results/zaps-high-noise-mask-stage-response-20261104'
class ResourcePause(RuntimeError):pass

def check(root):
    p=files.read(root/'plan.json')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
    for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Parent changed: '+n)
    return p

def prepare(root,parent):
    if (root/'plan.json').exists():return check(root)
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty unplanned directory retained')
    from utils.diag_shrink_multitask import check as parent_check
    p=parent_check(parent);s=files.read(parent/'summary.json')
    files.require(s['completed_runs']==s['planned_runs']==72,'Completed A71 required')
    cases=[c for c in p['cases'] if c['task']=='inpainting' and c['sigma']==.1]
    files.require({c['image_id'] for c in cases}=={'00226','00443','00963'},'Original fixed3 cases required')
    bound={str(parent/'plan.json'):files.digest(parent/'plan.json')}
    for c in cases:
        for arm in ('BASE','SHRINK'):
            folder=parent/'cases'/c['key']/arm;r=files.read(folder/'run.json');files.require(r['status']=='complete','Incomplete parent')
            for n,h in r['artifacts'].items():files.require(files.digest(folder/n)==h,'Old artifact changed')
            for n in ['run.json','paired.pt','paired_trace.json']+(['entry.pt'] if arm=='BASE' else []):bound[str(folder/n)]=files.digest(folder/n)
        for n in ('y.pt','mask.pt'):f=parent/'observations'/c['key']/n;bound[str(f)]=files.digest(f)
        bound[c['image']]=files.digest(c['image'])
    names=['utils/mask_stage_response.py','utils/diag_mask_stage_response.py','utils/test_mask_stage_response.py',
      'utils/test_mask_stage_response_runtime.py','utils/noise_signal_shrink.py','utils/shrink_multitask.py','utils/shrink_multitask_runtime.py',
      'modules/zaps_algorithm.py','modules/wavelet.py','modules/diffusion_model.py','utils/metrics.py','utils/paper_clock_runtime.py',
      'utils/a67_model_freeze.py','utils/terminal_mask_counterfactual.py','run_a87_20261104.sh',str(CARD.relative_to(PROJECTS))]
    root.mkdir(parents=True)
    plan=dict(experiment='A87_mask_stage_response',parent=str(parent),cases=cases,checkpoint=p['checkpoint'],
        inputs=bound,sources={n:files.digest(PROJECTS/n) for n in names},arms=list(rules.ARMS),
        budget=dict(suffix_forwards=450,probe_forwards=9,optimizer_backward=0,input_vjp=0,warmup_per_load=1),
        protocol=dict(scope='Original3 FFHQ sigma=.10 inpainting development images, same cached270-prefix parameters/init/RNG. Complete method remains300F.',
          interventions='Only force gamma1 at steps0..9 /10..19 /20..29; all other decisions recomputed live. Diagnostic interventions, NOT a proposed fixed-time policy.',
          response='SHRINK_CHECK only, steps4/14/24: counterfactual gamma1 next-state uses SAME DDPM innovation as current gamma. One extra denoiser per point; natural next prediction reused.',
          metrics='Observed/missing/full clean MSE perstep; paired next-clean finite response, not numerical integration error, posterior uncertainty or trajectory cosine.',
          GT='Offline error reporting only, not multiplier/probe/step/output/arm selection.',
          gates='BASE_CHECK and SHRINK_CHECK vs cached output relativeL2<=1e-6 before hybrid arms; all arms final RNG equal; exactly30F or33F with probes; no input/optimizer backwards.',
          limits='A86 excluded direct terminal missing correction as maincause. This locates trajectory-stage effects; no new positive-method/transfer/step-size/curvature claim. No parameter scan.'))
    files.write(root/'plan.json',plan);return plan

def archive(root):
    from utils import archive_refinement_confirmation as e
    e.BRANCH=BRANCH;bundle,sha=e.export_and_commit({'A87':root},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True);print('RESULTS_PUSH_PASS: A87 uploaded.',flush=True)

def execute(root,parent):
    import torch
    from modules.main_single import load_diffusion_model,load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.a67_model_freeze import frozen_network
    from utils import paper_clock_runtime as clock
    from utils import historical_ffhq_runtime as precision
    from utils.terminal_mask_counterfactual import region_errors
    from utils.diag_noise_feasible_zaps import metrics
    from utils.diag_paper_clock_alignment import gpu_inventory
    p=check(root);torch.set_num_threads(2)
    files.require(files.digest(p['checkpoint']['path'])==p['checkpoint']['sha256'],'Wrong existing FFHQ checkpoint')
    model=load_diffusion_model('ffhq','cuda:0');model.model.eval()
    state=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=True);model.model.load_state_dict(state,strict=True);del state
    loads=root/'loads';loads.mkdir(exist_ok=True);load_file=loads/(str(len(list(loads.glob('*.json')))).zfill(3)+'.json')
    probe=dict(warmup_forward_attempts=0)
    def load(f):return torch.load(f,map_location='cpu',weights_only=True)
    with frozen_network(model.model),torch.no_grad(),precision.precision('LEGACY_IEEE'):
        probe['warmup_forward_attempts']=1;files.write(load_file,probe)
        model._predict_eps(torch.zeros(1,3,256,256,device=model.device),torch.tensor([999],device=model.device))
        for c in p['cases']:
            base=parent/'cases'/c['key'];obs=parent/'observations'/c['key'];entry=load(base/'BASE/entry.pt')
            gt=load_image_as_tensor(c['image']).to(model.device);mask=load(obs/'mask.pt').to(model.device);y=load(obs/'y.pt')['y'].to(model.device)
            observe=lambda m:region_errors(m,gt,mask)
            for arm in rules.ARMS:
                folder=root/'cases'/c['key']/arm;record=folder/'run.json'
                if record.exists():
                    old=files.read(record);files.require(old['status']=='complete','Prior failed/interrupted attempt retained')
                    for n,h in old['artifacts'].items():files.require(files.digest(folder/n)==h,'A87 artifact changed')
                    continue
                free=int(torch.cuda.mem_get_info()[0])
                if free+torch.cuda.memory_reserved()<6*files.GIB:raise ResourcePause('GPU headroom lost before unstarted attempt')
                folder.mkdir(parents=True);row=dict(status='running',case=c,arm=arm,counts={});files.write(record,row)
                try:
                    with clock.counters(model.model) as count:
                        output,detail=rules.run(model,mask,y,entry,c,arm,observe)
                    expected=33 if arm=='SHRINK_CHECK' else 30
                    files.require(count['forward_attempts']==count['forwards']==expected,'Diagnostic NN count changed')
                    files.require(count['input_vjp_attempts']==count['optimizer_backward_attempts']==0,'Unexpected backwards')
                    refs=load(base/('BASE' if arm=='BASE_CHECK' else 'SHRINK')/'paired.pt')
                    if arm in ('BASE_CHECK','SHRINK_CHECK'):
                        error=float((output.double()-refs.double()).norm()/refs.double().norm().clamp(min=1e-30));files.require(error<=1e-6,'Cached output equivalence failed')
                        row['replay_relative_error']=error
                    rng=detail.pop('final_cpu_rng');cuda=detail.pop('final_cuda_rng')
                    torch.save(dict(cpu=rng,cuda=cuda),folder/'final_rng.pt')
                    if arm!='BASE_CHECK':
                        other=load(root/'cases'/c['key']/'BASE_CHECK/final_rng.pt')
                        files.require(torch.equal(other['cpu'],rng) and len(other['cuda'])==len(cuda) and all(torch.equal(a,b) for a,b in zip(other['cuda'],cuda)),'Final RNG pairing failed')
                    row['lpips_evaluation_attempts']=1;files.write(record,row)
                    m=metrics(output,gt.cpu());m.update(region_errors(output,gt.cpu(),mask.cpu()))
                    torch.save(output,folder/'output.pt');tensor_to_image(output.squeeze(0),denormalize=True).save(folder/'output.png')
                    files.write(folder/'trace.json',detail['trace']);files.write(folder/'response.json',detail['probes'])
                    row.update(status='complete',counts=count,metrics=m,probe_forwards=detail['probe_forwards'],operator_actions=detail['operator_actions'],
                        artifacts={n:files.digest(folder/n) for n in ('output.pt','output.png','trace.json','response.json','final_rng.pt')})
                    files.write(record,row);print('A87 COMPLETE:',c['image_id'],arm,'F=',count['forwards'],'PSNR=',m['psnr'],flush=True)
                except BaseException as error:
                    row.update(status='failed',error=repr(error),counts=locals().get('count',{}),traceback=traceback.format_exc());files.write(record,row);raise
    summarize(root)

def summarize(root):
    p=check(root);rows=[files.read(f) for f in (root/'cases').glob('*/*/run.json')];done=[r for r in rows if r['status']=='complete'];groups=[]
    for arm in rules.ARMS:
        rs=[r for r in done if r['arm']==arm]
        if rs:groups.append(dict(arm=arm,images=len(rs),means={k:sum(r['metrics'][k] for r in rs)/len(rs) for k in ('psnr','ssim','lpips','mse_full','mse_observed','mse_missing')}))
    s=dict(experiment=p['experiment'],classification='STAGE_AND_PAIRED_RESPONSE_DIAGNOSIS_ONLY',completed_runs=len(done),planned_runs=15,groups=groups,
      recorded_forwards=sum(r.get('counts',{}).get('forward_attempts',0) for r in rows),
      operator_actions={k:sum(r.get('operator_actions',{}).get(k,0) for r in done) for k in ('H','adjoint','probe_mask')},paired_response_probe_forwards=sum(r.get('probe_forwards',0) for r in done),
      warmup_forwards=sum(files.read(f).get('warmup_forward_attempts',0) for f in (root/'loads').glob('*.json')),
      lpips_evaluation_attempts=sum(r.get('lpips_evaluation_attempts',0) for r in rows),
      cost_is_lower_bound=any(r['status']!='complete' for r in rows),limits=p['protocol'])
    files.write(root/'summary.json',s);print('A87 Summary:',json.dumps(s,ensure_ascii=False),flush=True)

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
                if not candidates:status['status']='resource_limited';print('A87 no allocated GPU>=6GiB free; no model started.',flush=True)
                else:
                    if (root/'allocation.json').exists():
                        assigned=files.read(root/'allocation.json')
                        if not any(r['uuid']==assigned['uuid'] for r in candidates):raise ResourcePause('Pinned GPU lacks margin; resume when free')
                    else:assigned=candidates[0];files.write(root/'allocation.json',assigned)
                    os.environ['CUDA_VISIBLE_DEVICES']=assigned['uuid']
                    subprocess.run([sys.executable,'-m','unittest','-v','utils.test_mask_stage_response','utils.test_mask_stage_response_runtime'],cwd=PROJECTS,check=True)
                    execute(root,parent);status['status']='complete'
            except ResourcePause as e:status.update(status='resource_limited',reason=str(e));print('A87 RESOURCE PAUSE:',str(e),flush=True)
            except BaseException as e:status.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('A87 FAILED:',repr(e),flush=True)
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
                if old['status']=='complete':print('A87 already complete:',root/'summary.json');return
            with (state/'worker.log').open('ab',buffering=0) as log:process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a71-root',str(parent),'--gpus',args.gpus],cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        offset=0
        while True:
            if (state/'worker.log').exists():
                with (state/'worker.log').open(errors='replace') as f:f.seek(offset);v=f.read();offset=f.tell()
                if v:print(v,end='',flush=True)
            if not busy(state/'worker.lock') and (process is None or process.poll() is not None):break
            time.sleep(1)
        if record.exists():
            row=files.read(record);print('A87 SESSION:',row,flush=True)
            if row['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
