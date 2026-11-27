"""A146: preserve the original physical diagonal while testing guarded temporal detail."""
import argparse
from pathlib import Path
import statistics
import time
from collections import Counter
from contextlib import nullcontext
from unittest.mock import patch

from utils import a146_guarded_temporal as rules
from utils import diag_a145_temporal_panel as panel

parent, temporal, zaps = panel.parent, panel.temporal, panel.zaps
files, audit, clock, history, native, shared = panel.files, panel.audit, panel.clock, panel.history, panel.native, panel.shared
bridge, endpoint = panel.bridge, panel.endpoint
ROOT = panel.ROOT.with_name('diag_a146_guarded_panel')
CELLS = (('dps','inpainting'),('dps','super_resolution'),('zaps','motion_deblur'))
ARMS = (*panel.CONTROLS,*rules.ARMS)


def native_sample(entry,model,wave,arm,active,warm,counts,trace):
    import torch

    case,obs,config = entry['case'],entry['observation'],entry['plan']
    y = audit.checked_tensor(obs['path'],model.device)
    operator = history.operator(entry,obs,'dps',model.device,counts)
    geom = panel.method.Geometry(case['task'],operator,torch.zeros((1,3,256,256),device=model.device),counts)
    physical_view = geom.view.operator
    physical_work = []
    physical = panel.temporal.late.terminal.controller('DIAGONAL_K12',wave,counts,physical_work)
    remember,refine = rules.controller(geom,wave,arm,case['sigma'],counts,trace)

    def update(q,r,alpha,scalar,t,clean):
        delta = physical(physical_view,q,r,alpha,scalar,t)
        out = refine(q,r,alpha,scalar,t,clean,delta)
        trace[-1]['physical_updates'] = physical_work[-1]['updates']
        return out

    run = panel.native_instrument(history.native(entry,'dps'),'dps',active,warm,remember,update)
    return run(model,operator,y,case,'SHRINK',config['diffusion_config'],config['dps_root'],counts)


def zaps_suffix(wave,arm,sigma,active,warm,geometry,counts,trace,*,strength=1.):
    remember,refine = rules.controller(geometry,wave,arm,sigma,counts,trace,strength=strength)
    original = ('            delta, detail, module = repaired.exact_diagonal_with_native_straight_through(\n'
                '                zaps, zaps.A, clean, q, residual, ab[t], rules.SIGMA, module)')
    new = (
        original+'\n'
        '            if t in warm:remember(clean)\n'
        '            if t in active:\n'
        '                with torch.no_grad():\n'
        '                    altered=refine(q.detach(),residual,float(ab[t]),detail[\'scalar\'],t,'
        'clean.detach(),delta.detach())\n'
        '                delta=altered+(delta-delta.detach())')
    return audit.insert_probe(zaps._custom_suffix,[(original,new)],active=set(active),warm=set(warm),
                              remember=remember,refine=refine)


def evaluate(out,truth,y,geom,device,quality=True):
    return panel.evaluate_on_device(out,truth,y,geom,device,quality)


def record_trace(trace):
    return dict(points=len(trace),full=sum(p['beta']==1 for p in trace),
                zero=sum(p['beta']==0 for p in trace),partial=sum(0<p['beta']<1 for p in trace),
                mean_beta=statistics.mean(p['beta'] for p in trace),
                physical_up=sum(p['objective_applied']>p['objective_physical']+p['tolerance'] for p in trace),
                backtracks=sum(p['halvings'] for p in trace))


def dps_case(root,entry,model,wave,grid,active,warm,index):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_pigdm_noisy_r1 import fingerprint

    case,obs = entry['case'],entry['observation']
    key,device = case['key'],model.device
    truth = load_image_as_tensor(case['image'])
    op = history.operator(entry,obs,'dps',device,None)
    y = audit.checked_tensor(obs['path'],device)
    eval_counts = {}
    native.initialize_counts(eval_counts)
    with torch.no_grad():
        geom = panel.method.Geometry(case['task'],op,truth.to(device),eval_counts)
        gate = geom.check(truth.to(device))
        if case['task']=='inpainting':
            y = y[geom.view.operator.selected]
    arms,actual = {},Counter()
    reference = entry['controls']['SCALAR'][0]
    for name,(row,folder) in entry['controls'].items():
        out = endpoint.read_output(folder,row,reference,'dps')
        regions = evaluate(out,truth,y,geom,device,False)
        files.require(abs(regions['raw_mse']-row['raw_mse'])<=1e-9,'A139 archived MSE differs')
        arms[name] = dict(row,**regions,inherited=True)
    order = rules.ARMS if index%2==0 else reversed(rules.ARMS)
    for arm in order:
        first,trace = [],[]
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0,graph_actions=0)

            def at_entry(_network,args):
                if len(counts['calls'])==len(grid)-len(active)+1:
                    first.append(fingerprint(args[0]))
                    files.require(first[0]==reference['late_entry'],'A139 scalar prefix differs')

            with model.model.register_forward_pre_hook(at_entry):
                (out,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:native_sample(entry,model,wave,arm,active,warm,counts,trace))
            pairing = history.fingerprint(detail,shared.rng(),counts,'dps')
        panel.count_budget(counts,len(grid),0,len(grid),grid)
        files.require(pairing==reference['pairing'] and first==[reference['late_entry']] and
                      detail['grid']==grid and tuple(p['t'] for p in trace)==active and
                      [p['t'] for p in detail['trace'] if 'weight' in p]==grid,
                      'A146 DPS pairing or coverage differs')
        record = dict(status='complete',case=key,arm=arm,counts=counts,method_nfe=len(grid),
                      input_vjp=len(grid),full_seconds=seconds,inherited=False,guard=record_trace(trace),
                      pairing=pairing,late_entry=first[0],**evaluate(out,truth,y,geom,device))
        panel.save_candidate(root,key,arm,out,record,trace)
        arms[arm] = record
        actual.update({k:counts.get(k,0) for k in panel.COSTS})
        print('A146 COMPLETE dps',key,arm,record['quality'],record['guard'],flush=True)
    return dict(case=key,arms=arms,geometry=gate),dict(actual)


def zaps_case(root,entry,model,wave,kernel,grid,active,warm,index,*,strengths=None):
    import torch
    from modules.main_single import load_image_as_tensor
    from utils.diag_shared_prefix_feedback import rng_equal

    case = entry['case']
    key,device = case['key'],model.device
    truth = load_image_as_tensor(case['image'])
    y = audit.checked_tensor(entry['observation']['path'],device)

    def build(counts=None):
        return zaps.make_operator(case,device,count=counts,kernel=kernel)

    with clock.counters(model.model) as counts:
        op = build(counts)
        prefix,prefix_seconds = clock.clocks(torch.cuda.synchronize,
            lambda:zaps.train_prefix(model,op,y,case))
    panel.count_budget(counts,270,9,0,grid*9)
    actual = Counter({k:counts.get(k,0) for k in panel.COSTS})
    eval_counts = {}
    native.initialize_counts(eval_counts)
    with torch.no_grad():
        eval_geometry = panel.method.Geometry(case['task'],build(),truth.to(device),eval_counts)
        gate = eval_geometry.check(truth.to(device))

    arms,reference_rng = {},None
    candidates = rules.ARMS if strengths is None else tuple(strengths)
    order = [*panel.CONTROLS,*(candidates if index%2==0 else reversed(candidates))]
    for arm in order:
        trace = []
        with clock.counters(model.model) as counts:
            native.initialize_counts(counts)
            counts.update(graph_builds=0,graph_actions=0)
            before_setup = time.perf_counter()
            op = build(counts)
            context = nullcontext()
            if arm in candidates:
                geom = panel.method.Geometry(case['task'],op,truth.to(device),counts)
                mode = arm if strengths is None else 'GUARDED'
                strength = 1. if strengths is None else strengths[arm]
                suffix = zaps_suffix(wave,mode,case['sigma'],active,warm,geom,counts,trace,
                                     strength=strength)
                context = patch.object(zaps,'_custom_suffix',suffix)
            torch.cuda.synchronize()
            setup_seconds = time.perf_counter()-before_setup
            with context:
                suffix_arm = panel.ZAPS_ARMS.get(arm,panel.ZAPS_ARMS['DIAGONAL'])
                (out,detail),seconds = clock.clocks(torch.cuda.synchronize,
                    lambda:zaps.run_suffix(model,op,y,case,suffix_arm,prefix))
        panel.count_budget(counts,30,1,0,grid)
        if reference_rng is None:
            reference_rng = detail['final_rng']
        files.require(rng_equal(reference_rng,detail['final_rng']) and
                      tuple(p['t'] for p in trace)==(active if arm in candidates else ()) and
                      [p['t'] for p in detail['trace']]==grid,'A146 ZAPS RNG or window differs')
        record = dict(status='complete',case=key,arm=arm,counts=counts,method_nfe=300,input_vjp=0,
                      optimizer_backward=10,full_seconds=prefix_seconds+setup_seconds+seconds,inherited=False,
                      **evaluate(out,truth,y,eval_geometry,device))
        if arm in panel.CONTROLS:
            ref,folder = entry['controls'][arm]
            record['a102_replay_relative_l2'] = bridge.archived_error(out,folder,ref,panel.ZAPS_ARMS[arm]+'.pt')
        else:
            record['guard'] = record_trace(trace)
        for name,count in detail['module_counts'].items():
            if name in counts:
                counts[name] += count
        panel.save_candidate(root,key,arm,out,record,trace)
        arms[arm] = record
        actual.update({k:counts.get(k,0) for k in panel.COSTS})
        label = 'A146' if strengths is None else 'A148'
        print(label,'COMPLETE zaps',key,arm,record['quality'],record.get('guard'),flush=True)
    return dict(case=key,arms=arms,geometry=gate),dict(actual)


def summarize(cases):
    means = {}
    for arm in ARMS:
        rows = [case['arms'][arm] for case in cases]
        means[arm] = dict(
            quality={key:statistics.mean(r['quality'][key] for r in rows) for key in parent.transfer.METRICS},
            regions={key:statistics.mean(r[key] for r in rows) for key in panel.REGIONS},
            full_seconds=statistics.mean(r['full_seconds'] for r in rows),nfe=rows[0]['method_nfe'],
            input_vjp=rows[0].get('input_vjp',rows[0]['counts']['input_vjp_attempts']),
            inherited=rows[0]['inherited'])
        if arm in rules.ARMS:
            means[arm]['guard'] = {
                **{key:sum(r['guard'][key] for r in rows)
                   for key in ('points','full','zero','partial','physical_up','backtracks')},
                'mean_beta':statistics.mean(r['guard']['mean_beta'] for r in rows)}
    pairs = []
    for arm,ref in (('DIAGONAL','SCALAR'),('UNGUARDED','SCALAR'),('GUARDED','SCALAR'),
                    ('GUARDED','UNGUARDED'),('GUARDED','DIAGONAL')):
        comparison = bridge.paired(cases,arm,ref)
        comparison['region_delta'] = {k:means[arm]['regions'][k]-means[ref]['regions'][k] for k in panel.REGIONS}
        pairs.append(comparison)
    return dict(images=len(cases),arms=means,paired=pairs)


def execute(host,task,root,source):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    inputs,checkpoint,kernel_path,frozen = (
        panel.zaps_sources(task,source) if host=='zaps' else panel.native_sources(host,task))
    files.require(files.digest(checkpoint['path'])==checkpoint['sha256'],'Frozen checkpoint changed')
    files.require(all(root!=p and not p.is_relative_to(root) for p in frozen),'A146 overlaps parent')
    grid,active,warm = panel.schedule(host)
    card = parent.transfer.PROJECTS/'experiment_cards/A146_guarded_panel.md'
    test = Path(__file__).with_name('test_a146_guarded_panel.py')
    config = dict(experiment='A146',host=host,task=task,grid=grid,temporal_positions=list(active),
                  physical_positions=grid if host=='zaps' else list(active),warm=list(warm),
                  source_sha256={str(p):files.digest(p) for p in frozen},checkpoint=checkpoint,
                  code_sha256={str(p):files.digest(p) for p in
                               (Path(__file__),Path(rules.__file__),Path(panel.__file__),
                                Path(panel.method.__file__),card,test)},
                  arms=list(ARMS),graph_strength=temporal.STRENGTH,ridge=temporal.graph.RIDGE,
                  max_updates=temporal.late.depth.MAX_UPDATES,
                  max_backtracks=rules.MAX_BACKTRACKS,rounding_ulps=rules.ROUNDING_ULPS)
    path = root/'summary.json'
    if path.exists():
        previous = files.read(path)
        files.require(previous['status']=='complete' and files.read(root/'resolved_config.json')==config,
                      f'Prior partial or incompatible A146 retained at {root}')
        print('A146 SUMMARY',host,task,previous['group'],flush=True)
        return
    resource = panel.admission.check(torch,root)
    root.mkdir(parents=True)
    files.write(root/'resolved_config.json',config)
    report = dict(experiment='A146',host=host,task=task,status='running',cases=[],resource=resource,
                  actual_cost=dict.fromkeys(panel.COSTS,0),cost_is_lower_bound=False,
                  timing='DPS A139 controls inherited; ZAPS same-session shared-prefix equivalent times.')
    files.write(path,report)
    try:
        started = time.perf_counter()
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(checkpoint['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = None if kernel_path is None else np.load(kernel_path,allow_pickle=False)
        torch.cuda.synchronize()
        report['model_load_seconds'] = time.perf_counter()-started
        scope = native.dps_reference_scope() if host=='dps' else nullcontext()
        derivative = input_vjp_parameters(model.model) if host=='dps' else nullcontext()
        with precision('LEGACY_IEEE'),frozen_network(model.model),derivative,scope:
            for index,entry in enumerate(inputs):
                row,cost = (zaps_case(root,entry,model,wave,kernel,grid,active,warm,index) if host=='zaps'
                            else dps_case(root,entry,model,wave,grid,active,warm,index))
                report['cases'].append(row)
                for name,value in cost.items():
                    report['actual_cost'][name] += value
                files.write(path,report)
        expected = len(inputs)*(390 if host=='zaps' else 2*len(grid))
        files.require(report['actual_cost']['forwards']==expected,'A146 total calls differ')
        report.update(status='complete',group=summarize(report['cases']))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),cost_is_lower_bound=True)
        raise
    finally:
        files.write(path,report)
    print('A146 SUMMARY',host,task,report['group'],flush=True)


def matrix(root):
    rows = []
    for host,task in CELLS:
        path = root/f'{host}_{task}'/'summary.json'
        summary = files.read(path) if path.is_file() else {'status':'missing'}
        row = dict(host=host,task=task,status=summary['status'],source=str(path))
        if summary['status']=='complete':
            group = summary['group']
            row.update(guard=group['arms']['GUARDED']['guard'],
                       paired=[r for r in group['paired'] if r['candidate']=='GUARDED'],
                       nfe=group['arms']['GUARDED']['nfe'],input_vjp=group['arms']['GUARDED']['input_vjp'])
        elif summary.get('error'):
            row['error'] = summary['error']
        rows.append(row)
        print('A146 MATRIX',row,flush=True)
    files.write(root/'matrix.json',rows)
    return all(row['status']=='complete' for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=('dps','zaps','matrix'))
    parser.add_argument('task',choices=tuple(task for _,task in CELLS),nargs='?')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--zaps-parent',type=Path,default=Path(zaps.rules.ROOT))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.host=='matrix':
        raise SystemExit(0 if matrix(root) else 1)
    if (args.host,args.task) not in CELLS:
        parser.error('unsupported A146 sampler/task pair')
    execute(args.host,args.task,root/f'{args.host}_{args.task}',args.zaps_parent.resolve())


if __name__=='__main__':
    main()
