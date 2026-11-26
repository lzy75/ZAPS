"""A142: diagnose the scalar-to-diagonal direction on saved inpainting outputs."""
import argparse
import math
from pathlib import Path
import statistics

from utils import diag_a141_graph_inpainting as graph

parent, files, audit = graph.parent, graph.files, graph.audit
ROOT = graph.ROOT.with_name('diag_a142_inpainting_direction')
HOSTS = ('dps', 'pigdm')
ARMS = ('DIAGONAL', *graph.STRENGTHS)


def quadratic(slope, curvature):
    """Change in MSE is slope*a + curvature*a^2 for endpoint interpolation."""
    files.require(math.isfinite(slope) and math.isfinite(curvature) and curvature>=0,
                  'Invalid direction coefficients')
    optimum = min(1., max(0., -.5*slope/curvature)) if curvature else 0.
    return dict(initial_slope=slope, increment_mse=curvature, end_delta=slope+curvature,
                offline_best_fraction=optimum, offline_best_delta=optimum*(slope+optimum*curvature))


def direction(scalar, candidate, truth, observed, noisy):
    error = scalar.double()-truth.double()
    delta = candidate.double()-scalar.double()

    def measure(r,d):
        return quadratic(2*float((r*d).mean()),float(d.square().mean()))

    visible = measure(error[observed],delta[observed])
    hidden = measure(error[~observed],delta[~observed])
    measured_noise = noisy.double().reshape(-1)-truth.double()[observed]
    noisy_result = measure(error[observed]-measured_noise,delta[observed])
    return dict(full=measure(error,delta), observed=visible, missing=hidden,
                noisy_observation=noisy_result,
                noise_alignment_delta=float((measured_noise*delta[observed]).mean()))


def load_mask(case, host, truth):
    import torch

    folder = Path(case['source'])
    key = case['case']['key']
    plan_path = folder/'plan.json'
    files.require(files.digest(plan_path)==case['plan_sha256'],'Source plan changed: '+key)
    if host=='dps':
        plan = files.read(plan_path)
        noise_path = Path(plan['observations'][key])
        meta_path = noise_path.parent/'observation.json'
        mask_path = noise_path.parent/'mask.pt'
        meta = files.read(meta_path)
        mask_name = 'mask.pt'
    else:
        meta_path = folder/'observations'/key/'observation.json'
        meta = files.read(meta_path)
        noise_path = Path(meta['path'])
        mask_path = Path(meta['operator_state'])
        mask_name = 'operator_state.pt'
    files.require(meta['plan_sha256']==case['plan_sha256'] and
                  files.digest(noise_path)==case['observation_sha256']==meta['artifacts']['y.pt'] and
                  files.digest(mask_path)==meta['artifacts'][mask_name],'Archived mask or measurement changed: '+key)
    saved = torch.load(mask_path, map_location='cpu', weights_only=True)
    mask = saved if host=='dps' else saved['mask'][None,None]
    files.require(bool(((mask==0)|(mask==1)).all()), 'Mask is not binary')
    mask = mask.bool().expand_as(truth)
    files.require(0<int(mask.sum())<mask.numel(), 'Mask needs both regions')
    noisy = audit.checked_tensor(noise_path,'cpu')
    if host=='dps':
        noisy = noisy[mask]
    else:
        # The native packer puts RGB first and lists spatial indices in row-major order.
        noisy = noisy.reshape(-1)
    files.require(noisy.numel()==int(mask.sum()), 'Packed measurement shape differs')
    return mask,noisy


def graph_result(source, host, control):
    path = source/'summary.json'
    if not path.is_file():
        return dict(status='missing'), {}
    report = files.read(path)
    files.require(report['experiment']=='A141' and report['host']==host,'Wrong A141 summary')
    status = {key:report.get(key) for key in
              ('status','completed_new_runs','planned_new_runs','error')}
    if status['status']!='complete':
        return status, {}
    config = files.read(source/'resolved_config.json')
    files.require(config['experiment']=='A141' and config['host']==host and
                  config['strengths']==graph.STRENGTHS and Path(config['source'])==control and
                  report['completed_new_runs']==report['planned_new_runs']==6 and
                  report['actual_cost']['forwards']==report['actual_cost']['input_vjp_attempts']==
                  6*len(parent.GRIDS[host]),'Incompatible completed A141')
    for name in ('summary.json','resolved_config.json'):
        files.require(files.digest(control/name)==config['source_sha256'][name],'A141 A139 parent changed')
    status['paired_quality'] = report['group']['paired']
    status['arms'] = report['group']['arms']
    return status,{case['case']:case['arms'] for case in report['cases']}


def read_output(folder, record, reference, host):
    if folder.name!='SCALAR':
        files.require(record['pairing']==reference['pairing'] and record['late_entry']==reference['late_entry'],
                      'Endpoint does not share the original scalar trajectory prefix')
    grid = parent.GRIDS[host]
    files.require(record['status']=='complete' and files.read(folder/'run.json')==record and
                  record['method_nfe']==record['counts']['forwards']==record['counts']['input_vjp_attempts']==len(grid) and
                  record['counts']['calls']==grid,'Endpoint record or sampling budget differs')
    path = audit.qualified_artifact(folder,'output.pt',record)
    return audit.checked_tensor(path,'cpu')


def summarize(rows):
    results = {}
    for name in ARMS:
        arms = [row['arms'][name] for row in rows if name in row['arms']]
        if arms:
            results[name] = {
                region:dict(downhill_images=sum(a[region]['initial_slope']<0 for a in arms), images=len(arms),
                            mean_initial_slope=statistics.mean(a[region]['initial_slope'] for a in arms),
                            mean_end_delta=statistics.mean(a[region]['end_delta'] for a in arms),
                            mean_offline_best_delta=statistics.mean(a[region]['offline_best_delta'] for a in arms))
                for region in ('full','observed','missing','noisy_observation')}
    return results


def verify_endpoint(output, truth, mask, record):
    error = (output-truth).double().square()
    observed = record.get('observed_mse',record.get('clean_observation_mse'))
    files.require(abs(float(error.mean())-record['raw_mse'])<=1e-9 and
                  abs(float(error[mask].mean())-observed)<=1e-9,'Saved endpoint MSE differs')
    if 'missing_mse' in record:
        files.require(abs(float(error[~mask].mean())-record['missing_mse'])<=1e-9,'Saved missing MSE differs')


def examine(host, control, source):
    from modules.main_single import load_image_as_tensor

    status, candidates = graph_result(source,host,control)
    # Print status before the per-image audit so missing or failed DPS is explicit.
    print('A142 A141_STATUS',host,{k:v for k,v in status.items() if k not in ('arms','paired_quality')},flush=True)
    if candidates:
        for comparison in status['paired_quality']:
            if comparison['reference']=='SCALAR':
                print('A142 A141_QUALITY',host,comparison['candidate'],comparison['delta'],
                      comparison['psnr_wins'],'/',comparison['images'],flush=True)

    plan = files.read(control/'resolved_config.json')
    baseline = files.read(control/'summary.json')
    files.require(plan['experiment']==baseline['experiment']=='A139' and plan['host']==baseline['host']==host and
                  plan['dataset']==baseline['dataset']=='ffhq' and baseline['status']=='complete',
                  'Need complete A139 inpainting controls')
    manifests = {row['case']['key']:row for row in plan['cases'] if row['case']['task']=='inpainting'}
    controls = {row['case']:row['arms'] for row in baseline['cases'] if row['task']=='inpainting'}
    files.require(len(controls)==3 and set(controls)==set(manifests) and
                  (not candidates or set(candidates)==set(controls)), 'Inpainting cases differ')
    rows = []
    for key,manifest in manifests.items():
        image_path = manifest['case']['image']
        files.require(files.digest(image_path)==manifest['image_sha256'],'Reference image changed')
        load = graph.shared.load_image if host=='pigdm' else load_image_as_tensor
        truth = load(image_path)
        observed,noisy = load_mask(manifest,host,truth)
        reference = controls[key]['SCALAR']
        scalar = read_output(control/'cases'/key/'SCALAR',reference,reference,host)
        verify_endpoint(scalar,truth,observed,reference)
        results = {}
        for arm in ARMS:
            if arm=='DIAGONAL':
                folder,record = control,controls[key][arm]
            elif candidates:
                folder,record = source,candidates[key][arm]
            else:
                continue
            candidate = read_output(folder/'cases'/key/arm,record,reference,host)
            verify_endpoint(candidate,truth,observed,record)
            regions = direction(scalar,candidate,truth,observed,noisy)
            results[arm] = regions
            print('A142 DIRECTION',host,key,arm,
                  {name:round(regions[name]['initial_slope'],8) for name in ('full','observed','missing')},flush=True)
        rows.append(dict(case=key,arms=results))
    groups = summarize(rows)
    for arm,data in groups.items():
        print('A142 SUMMARY',host,arm,data,flush=True)
    return dict(A141=status,cases=rows,groups=groups,status='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a139-root',type=Path,default=parent.ROOT)
    parser.add_argument('--a141-root',type=Path,default=graph.ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'summary.json')
    args = parser.parse_args()
    output = args.output.resolve()
    roots = [path.resolve() for path in (args.a139_root,args.a141_root)]
    files.require(all(not output.is_relative_to(root) for root in roots),'Write A142 outside parent results')
    report = dict(experiment='A142',nfe=0,input_vjp=0,input_jvp=0,hosts={})
    for host in HOSTS:
        try:
            report['hosts'][host] = examine(host, roots[0]/f'A139_{host}_ffhq', roots[1]/f'A141_{host}_ffhq')
        except (OSError,ValueError,RuntimeError,KeyError) as exc:
            report['hosts'][host] = dict(status='failed',error=str(exc))
            print('A142 ERROR',host,str(exc),flush=True)
    files.write(output,report)
    print('A142 REPORT',output,flush=True)
    if any(row['status']!='complete' for row in report['hosts'].values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
