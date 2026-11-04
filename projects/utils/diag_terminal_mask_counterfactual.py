"""A86: reuse original A71 caches, evaluate terminal observed/missing corrections."""
import argparse
import os
os.environ['MKL_THREADING_LAYER']='GNU'
os.environ.pop('MKL_SERVICE_FORCE_INTEL',None)
from pathlib import Path
import csv
import json
import subprocess
import sys
import time
import traceback
PROJECTS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECTS));sys.path.insert(0,str(PROJECTS.parent))
from utils import historical_ffhq_gain as files
from utils import terminal_mask_counterfactual as rules
from utils.a54_session import exclusive,busy
ROOT='/data0/lzy/ZAPS-results/diag_terminal_mask_counterfactual/A86_20261104'
PARENT='/data0/lzy/ZAPS-results/diag_shrink_multitask/A71_20261029'
BRANCH='results/zaps-high-noise-mask-terminal-audit-20261104'
CARD=PROJECTS/'experiment_cards/A86_high_noise_mask_terminal_20261104.md'


def prepare(root,parent):
    if (root/'plan.json').exists():return check(root)
    files.require(not root.exists() or not any(root.iterdir()),'Nonempty output retained')
    from utils.diag_shrink_multitask import check as parent_check
    p=parent_check(parent);s=files.read(parent/'summary.json')
    files.require(s['completed_runs']==s['planned_runs']==72,'Completed A71 required')
    cases=[c for c in p['cases'] if c['task']=='inpainting' and c['sigma']==.1]
    files.require({c['image_id'] for c in cases}=={'00226','00443','00963'},'Frozen original panel required')
    bound={str(parent/'plan.json'):files.digest(parent/'plan.json')}
    for c in cases:
        obs=parent/'observations'/c['key']
        for n in ('y.pt','mask.pt','observation.json'):bound[str(obs/n)]=files.digest(obs/n)
        for arm in ('BASE','SHRINK'):
            folder=parent/'cases'/c['key']/arm;r=files.read(folder/'run.json')
            files.require(r['status']=='complete' and not r.get('reused',False),'Completed original inpainting cache required')
            for n,h in r['artifacts'].items():files.require(files.digest(folder/n)==h,'A71 artifact changed: '+n)
            names=['run.json','paired.pt','paired_trace.json']+(['entry.pt'] if arm=='BASE' else [])
            for n in names:bound[str(folder/n)]=files.digest(folder/n)
        bound[c['image']]=files.digest(c['image'])
    names=['utils/terminal_mask_counterfactual.py','utils/diag_terminal_mask_counterfactual.py',
        'utils/test_terminal_mask_counterfactual.py','utils/test_terminal_mask_counterfactual_runtime.py',
        'utils/noise_signal_shrink.py','modules/wavelet.py','modules/diffusion_model.py','modules/zaps_algorithm.py',
        'utils/metrics.py','run_a86_20261104.sh',str(CARD.relative_to(PROJECTS))]
    root.mkdir(parents=True)
    plan=dict(experiment='A86_terminal_mask_counterfactual',parent=str(parent),cases=cases,inputs=bound,
        sources={n:files.digest(PROJECTS/n) for n in names},
        protocol=dict(scope='Original3 FFHQ high-noise inpainting cases, sigma_model=.10, original last-unroll SHRINK trajectory unchanged.',
          terminal='Only cases with archived lastt=0,next=-1,gamma=0 qualify; stored SHRINK output is exactly terminal clean prediction.',
          output='Cached BASE/SHRINK plus terminal restore observed,missing,both components of original q. No new trajectory/no oracle arm selection.',
          decision='GT used only for region error and final metrics, never q/gamma. Read all three terminal counterfactuals, do not pick best as final method.',
          cost='0 UNet,0 VJP,0 optimizer; original CPU wavelet actions only;15 LPIPS evaluation calls separately counted. Not a runtime benchmark.',
          fidelity='CPU reconstruction of original q checked against archived observed-space dot/norm: relative error<=1e-3. Gate failure stops, no fallback.',
          interpretation='Terminal diagnostic, not evidence that a full30-step partitioned policy improves. q_missing is original correction in missing area, not an independently estimated prior.',
          risk='Observed+missing rawMSE decomposition, clamp-displayMSE, PSNR/SSIM/LPIPS, raw q regional energy. No full-image unbiased-risk claim.'))
    files.write(root/'plan.json',plan);return plan


def check(root):
    p=files.read(root/'plan.json')
    for n,h in p['sources'].items():files.require(files.digest(PROJECTS/n)==h,'Bound source changed: '+n)
    for n,h in p['inputs'].items():files.require(files.digest(n)==h,'Parent bytes changed: '+n)
    return p


def execute(root,parent):
    import torch
    from modules.wavelet import OrthogonalDWT2D
    from modules.diffusion_model import _precompute_diffusion_params
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.diag_noise_feasible_zaps import metrics
    from utils.noise_signal_shrink import select
    from utils import historical_ffhq_runtime as precision
    torch.set_num_threads(2)
    # Metric network is evaluation only; no automatic download or diffusion checkpoint load.
    weights=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    files.require(weights.is_file(),'Existing AlexNet evaluation weights required; no automatic download')
    p=prepare(root,parent);dwt=OrthogonalDWT2D('db4',3);ab=_precompute_diffusion_params()['alphas_cumprod']
    rows=[];gates=[];lpips_calls=0;started=time.perf_counter()
    load=lambda f:torch.load(f,map_location='cpu',weights_only=True)
    with torch.no_grad(),precision.precision('LEGACY_IEEE'):
        for c in p['cases']:
            base=parent/'cases'/c['key'];obs=parent/'observations'/c['key']
            trace=files.read(base/'SHRINK/paired_trace.json');last=trace[-1]
            files.require(last['t']==0 and last['next_t']==-1 and last['gamma']==0.,'Not a qualifying terminal zero-gamma case; no approximation accepted')
            m=load(base/'SHRINK/paired.pt');y=load(obs/'y.pt')['y'];mask=load(obs/'mask.pt');entry=load(base/'BASE/entry.pt')
            files.require(tuple(m.shape)==(1,3,256,256) and bool(torch.isfinite(m).all()) and float(m.abs().max())<=1.000001,'Invalid clean-state cache')
            files.require(entry['tau'].tolist()==files.read(parent/'plan.json')['grid'],'Original grid differs')
            q=rules.reconstruct(m,y,mask,entry,ab[0],dwt)
            rr=(mask*(y-mask*m)).double();dd=(mask*(q/ab[0].sqrt())).double()
            dot=float((rr*dd).sum());d2=float(dd.square().sum());r2=float(rr.square().sum());M=int(mask.expand_as(m).sum())
            choose=select(dot,r2,d2,M,c['sigma'])
            errors={k:abs(v-last[k])/max(abs(last[k]),1e-12) for k,v in dict(residual_energy=r2,response_energy=d2,residual_response_dot=dot).items()}
            gate=dict(case=c['key'],cpu_vs_archived_relative_errors=errors,reconstructed_gamma=choose['gamma'],active_measurements=M,
                passed=max(errors.values())<=1e-3 and choose['gamma']==0. and M==last['active_measurements'])
            gates.append(gate);files.write(root/'gates.json',gates)
            files.require(gate['passed'],'CPU cached-terminal reconstruction gate failed; no NN fallback')
            qo,qu=rules.parts(q,mask);energy=float(q.double().square().sum())
            gate.update(q_energy=energy,q_observed_energy=float(qo.double().square().sum()),q_missing_energy=float(qu.double().square().sum()),
                q_missing_energy_fraction=float(qu.double().square().sum())/max(energy,1e-30))
            gt=load_image_as_tensor(c['image']).cpu();outputs=rules.interventions(m,q,mask);outputs['BASE']=load(base/'BASE/paired.pt')
            for arm,out in outputs.items():
                folder=root/'cases'/c['key']/arm;folder.mkdir(parents=True,exist_ok=False)
                lpips_calls+=1
                files.write(root/'cost.json',dict(UNet_forward_attempts=0,input_vjp_attempts=0,optimizer_backward_attempts=0,lpips_evaluation_attempts=lpips_calls))
                result=metrics(out,gt)
                result.update(rules.region_errors(out,gt,mask))
                displayed=rules.region_errors(out.clamp(-1,1),gt,mask)
                result.update({'display_'+k:v for k,v in displayed.items() if k.startswith('mse_')})
                observed=mask.bool().expand_as(m)
                result['measurement_mse_observed']=float((y-out).double()[observed].square().mean())
                if arm=='END_MISSING_RESTORE':files.require(torch.equal(out[observed],m[observed]),'Missing-only intervention changed observations')
                if arm in ('BASE','SHRINK'):
                    reference=files.read(base/arm/'run.json')['paired_metrics']
                    files.require(max(abs(result[k]-reference[k]) for k in ('psnr','ssim','lpips'))<=1e-4,'Cached evaluation parity failed')
                row=dict(case=c['key'],image_id=c['image_id'],arm=arm,**result);rows.append(row)
                torch.save(out,folder/'output.pt');tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png');files.write(folder/'run.json',row)
            files.write(root/'gates.json',gates)
            print('A86 CASE COMPLETE:',c['image_id'],'q_missing_fraction=',gate['q_missing_energy_fraction'],flush=True)
    groups=[]
    for arm in ('BASE','SHRINK','END_OBS_RESTORE','END_MISSING_RESTORE','END_BOTH_RESTORE'):
        data=[r for r in rows if r['arm']==arm]
        groups.append(dict(arm=arm,images=len(data),means={k:sum(r[k] for r in data)/len(data) for k in ('psnr','ssim','lpips','mse_full','mse_observed','mse_missing','measurement_mse_observed')}))
    result=dict(experiment=p['experiment'],classification='TERMINAL_COUNTERFACTUAL_ONLY',completed_cases=3,groups=groups,
        UNet_forward_attempts=0,input_vjp_attempts=0,optimizer_backward_attempts=0,lpips_evaluation_calls=lpips_calls,
        diagnostic_cpu_seconds=time.perf_counter()-started,gates=gates,limits=p['protocol'])
    files.write(root/'summary.json',result)
    with (root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('A86 Summary:',json.dumps(result,ensure_ascii=False),flush=True)


def archive(root):
    from utils import archive_refinement_confirmation as exporter
    exporter.BRANCH=BRANCH;bundle,sha=exporter.export_and_commit({'A86':root},projects=PROJECTS)
    print('Archive:',bundle,'Results commit:',sha,'Branch:',BRANCH,flush=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=PROJECTS.parent,check=True)
    print('RESULTS_PUSH_PASS: A86 terminal mask audit uploaded.',flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',default=ROOT);a.add_argument('--a71-root',default=PARENT)
    for f in ('execute','status','archive-only'):a.add_argument('--'+f,action='store_true')
    args=a.parse_args();root=Path(args.root).resolve();parent=Path(args.a71_root).resolve()
    state=root.parent/(root.name+'_session');state.mkdir(parents=True,exist_ok=True);record=state/'session.json'
    if args.status:print(dict(active=busy(state/'worker.lock'),session=files.read(record) if record.exists() else None));return
    if args.archive_only:files.require(not busy(state/'worker.lock'),'Worker active');archive(root);return
    if args.execute:
        with exclusive(state/'worker.lock'):
            status=dict(status='running',pid=os.getpid(),root=str(root));files.write(record,status)
            try:
                subprocess.run([sys.executable,'-m','unittest','-v','utils.test_terminal_mask_counterfactual','utils.test_terminal_mask_counterfactual_runtime'],cwd=PROJECTS,check=True)
                execute(root,parent);status['status']='complete'
            except BaseException as e:
                status.update(status='failed',error=repr(e),traceback=traceback.format_exc());print('A86 FAILED:',repr(e),flush=True)
            finally:
                files.write(record,status)
                if root.exists():
                    import shutil
                    shutil.copyfile(record,root/'session.json')
                    if (root/'plan.json').exists():
                        try:archive(root)
                        except subprocess.CalledProcessError:print('Upload failed; records retained; --archive-only retries upload.',flush=True)
        return
    with exclusive(state/'client.lock'):
        process=None
        if not busy(state/'worker.lock'):
            if record.exists():
                old=files.read(record);files.require(old['status']!='failed','Prior failure retained; no automatic retry')
                if old['status']=='complete':print('A86 already complete:',root/'summary.json');return
            with (state/'worker.log').open('ab',buffering=0) as log:
                process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--execute','--root',str(root),'--a71-root',str(parent)],
                    cwd=PROJECTS,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        offset=0
        while True:
            path=state/'worker.log'
            if path.exists():
                with path.open(errors='replace') as f:f.seek(offset);value=f.read();offset=f.tell()
                if value:print(value,end='',flush=True)
            if not busy(state/'worker.lock') and (process is None or process.poll() is not None):break
            time.sleep(1)
        if record.exists():
            row=files.read(record);print('A86 SESSION:',row,flush=True)
            if row['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
