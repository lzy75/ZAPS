"""A66R3 measurement: retain every repeat, never certify a failed reproducibility gate."""
from pathlib import Path
import gc
import math
import statistics
import time

from utils import historical_ffhq_gain as rules
from utils import historical_ffhq_runtime as engine
from utils import paper_clock_runtime as common
from utils import diag_historical_ffhq_gain as original
from utils.diag_paper_clock_alignment import resource,write


def repeat_gate(error):
    rules.require(math.isfinite(error) and error>=0,'Invalid repeat error')
    return dict(passed=error<=1e-6,tolerance=1e-6,relative_error=error,
                policy='Record every output; failed reproducibility remains failed; no output selection')


def measure(root,case,arm,repeat,model,uuid,versions):
    import torch
    from modules.main_single import load_image_as_tensor
    from modules.dataset_loader import tensor_to_image
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.metrics import compute_all_metrics
    rules.require(arm!='RESPONSE_300','Rejected prefix arm remains suspended')
    old=original.done(root,case,arm,repeat)
    if old:return old
    admission=resource(uuid,'ffhq',torch.cuda.memory_reserved())
    if not admission['admitted']:return None
    obs=original.observation(root,case,model.device)
    folder=root/'cases'/case['key']/arm/f'repeat_{repeat}';folder.mkdir(parents=True,exist_ok=False)
    r=dict(status='running',case=case,arm=arm,repeat=repeat,plan_sha256=rules.digest(root/'plan.json'),
           device_uuid=uuid,observation=obs,resource_before=admission,counts=None,artifacts={},cost_lower_bound=True,
           measurement_protocol='A66R3_all_repeats_retained')
    write(folder/'run.json',r);start=time.perf_counter();counts=None
    def tick(value):
        r['counts']=value;write(folder/'run.json',r)
        print('A66R3 PROGRESS:',case['image_id'],arm,'repeat',repeat,'F=',value['forwards'],flush=True)
    try:
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        with engine.precision(arm) as policy,input_vjp_parameters(model.model),common.counters(model.model,tick) as counts:
            def operation():
                payload=torch.load(obs['path'],map_location='cpu',weights_only=True);y=payload['y'].to(model.device)
                operator=engine.make_operator(model.device,counts,mode='exact' if arm=='EXACT_IEEE' else 'legacy_bicubic')
                return engine.execute(model,operator,y,case['seed'],arm,counts)
            (out,details),seconds=common.clocks(torch.cuda.synchronize,operation)
        engine.validate_counts(counts,arm)
        rules.require(all(math.isfinite(float(v)) for v in details['losses']),'Nonfinite loss')
        rules.require(all(v._version==version and v.grad is None for v,version in versions),'Frozen weights changed')
        if arm=='RESPONSE_30':
            reference=root/'cases'/case['key']/'LEGACY_IEEE/repeat_0/entry_snapshot.pt'
            gate=common.snapshot_equal(torch.load(reference,map_location='cpu',weights_only=True),details['snapshot'])
            rules.require(gate['rng_equal'] and gate['relative_errors']['initial']==0.,'Independent noise pairing differs')
            r['noise_pairing_gate']=dict(passed=True,learned_cache_read_for_inference=False,audit_after_timer=True)
        if repeat:
            reference=torch.load(root/'cases'/case['key']/arm/'repeat_0/output.pt',map_location='cpu',weights_only=True)
            error=float((out.double()-reference.double()).norm()/reference.double().norm().clamp(min=1e-30))
            r.update(repeat_relative_error=error,repeat_gate=repeat_gate(error))
            if not r['repeat_gate']['passed']:
                print('A66R3 REPRODUCIBILITY FAILED (output retained):',case['image_id'],arm,repeat,error,flush=True)
        metric_start=time.perf_counter();gt=load_image_as_tensor(case['image'])
        metrics=compute_all_metrics(out,gt,lpips_net='alex')
        mse=float(((out.clamp(-1,1).double()-gt.double())*.5).square().mean());metrics['float_psnr']=-10*math.log10(max(mse,1e-30))
        rules.require(all(math.isfinite(float(v)) for v in metrics.values()),'Nonfinite metrics')
        metric_seconds=time.perf_counter()-metric_start
        torch.save(out,folder/'output.pt');torch.save(details['snapshot'],folder/'entry_snapshot.pt')
        tensor_to_image(out.squeeze(0),denormalize=True).save(folder/'output.png')
        if not (root/'observations'/(case['image_id']+'_gt.png')).exists():
            tensor_to_image(gt.squeeze(0),denormalize=True).save(root/'observations'/(case['image_id']+'_gt.png'))
        write(folder/'trace.json',details['trace'])
        r.update(status='complete',counts=counts,precision=policy,metrics=metrics,loss_history=details['losses'],
            complete_reconstruction_seconds=seconds,metric_seconds_excluded=metric_seconds,
            all_attempt_seconds=time.perf_counter()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),resource_after=resource(uuid,'ffhq',torch.cuda.memory_reserved()),
            response_resolved_steps=sum(bool(v.get('resolved')) for v in details['trace']),
            fallback_steps=sum(not bool(v.get('resolved')) for v in details['trace']),cost_lower_bound=False,
            artifacts={name:rules.digest(folder/name) for name in ('output.pt','entry_snapshot.pt','trace.json','output.png')})
        write(folder/'run.json',r)
        print('A66R3 COMPLETE:',case['image_id'],arm,'repeat',repeat,'PSNR=',round(float(metrics['psnr']),4),
              'full_seconds=',round(seconds,3),flush=True)
        return r
    except BaseException as error:
        import traceback
        r.update(status='failed',counts=counts,error=repr(error),traceback=traceback.format_exc(),
                 elapsed_until_failure=time.perf_counter()-start,cost_lower_bound=True)
        write(folder/'run.json',r);raise


def operator_audit(device):
    import torch
    import torch.nn.functional as F
    # Private CPU RNG; do not consume the sampling RNG. No NN is instantiated.
    rng=torch.Generator(device='cpu').manual_seed(20261028)
    source=torch.randn((1,3,64,64),generator=rng).to(device)
    target=torch.randn((1,3,256,256),generator=rng).to(device)
    torch.cuda.synchronize();start=time.perf_counter();gradients=[];forward_equal=True;first=None
    for i in range(8):
        x=source.detach().clone().requires_grad_();y=F.interpolate(x,size=(256,256),mode='bicubic',align_corners=False)
        gradient=torch.autograd.grad(y,x,target)[0].detach()
        if first is None:first=y.detach().clone()
        else:forward_equal=forward_equal and torch.equal(first,y)
        gradients.append(gradient)
    errors=[float((g.double()-gradients[0].double()).norm()/gradients[0].double().norm()) for g in gradients]
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    deterministic_error=None
    try:
        torch.use_deterministic_algorithms(True)
        x=source.detach().clone().requires_grad_();y=F.interpolate(x,size=(256,256),mode='bicubic',align_corners=False)
        torch.autograd.grad(y,x,target)
    except RuntimeError as error:
        deterministic_error=str(error)
        if 'determin' not in deterministic_error.lower():raise
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)
    torch.cuda.synchronize()
    return dict(status='complete',unet_forwards=0,input_unet_vjps=0,operator_backward_attempts=9,
        forward_bitwise_equal=bool(forward_equal),gradient_relative_errors=errors,
        strict_determinism_error=deterministic_error,seconds=time.perf_counter()-start,
        scope='Known bicubic operator only; does not establish full-trajectory cause or quality repeatability')


def repeat_statistics(records):
    result={}
    for arm in sorted({r['arm'] for r in records}):
        rows=[r for r in records if r['arm']==arm];images={}
        for image in sorted({r['case']['image_id'] for r in rows}):
            ordered=sorted((r for r in rows if r['case']['image_id']==image),key=lambda r:r['repeat'])
            images[image]=dict(repeats=[r['repeat'] for r in ordered],
                metrics={key:dict(values=[r['metrics'][key] for r in ordered],
                    mean=statistics.mean(r['metrics'][key] for r in ordered),
                    std=statistics.stdev(r['metrics'][key] for r in ordered) if len(ordered)>1 else None)
                    for key in ('psnr','ssim','lpips')},
                repeat_gates=[dict(repeat=r['repeat'],**r['repeat_gate']) for r in ordered if 'repeat_gate' in r])
        result[arm]=images
    return result
