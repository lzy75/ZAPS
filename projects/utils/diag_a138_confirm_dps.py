"""A138: freeze A137 DPS and confirm on the seven images held aside for that rule."""
import argparse
from pathlib import Path
import statistics
import time

from utils import a126_resource_admission as admission
from utils import diag_a118_blur_transfer as transfer
from utils import diag_a118_transfer_bridge as bridge
from utils import diag_a133_native_grid_feedback as dense
from utils import diag_a137_last100_depth as development
from utils import paper_clock_alignment as files
from utils import transfer_diagonal_runtime as native

ROOT = dense.ROOT.with_name('diag_a138_confirm_dps')
ARMS = ('SCALAR','LAST100_K12')
GRID = list(range(999,-1,-1))


def load_inputs(parent, a137_root):
    source = a137_root/'summary.json'
    result = files.read(source)
    settings = files.read(a137_root/'resolved_config.json')
    parent_path = parent/'plan.json'
    files.require(result['experiment']=='A137' and result['status']=='complete' and
                  result['actual_new_nfe']==12000 and result['sources'][str(parent_path)]==files.digest(parent_path) and
                  settings['grid']==GRID and tuple(settings['positions'])==development.POSITIONS and
                  settings['max_updates']==development.depth.MAX_UPDATES and
                  settings['grad_rtol']==development.depth.GRAD_RTOL and
                  all(result['sources'][str(Path(module.__file__))]==files.digest(module.__file__)
                      for module in (development,development.depth,development.terminal)),
                  'Expected matching finished and frozen A137')
    accepted = set()
    for group in result['groups']:
        if any(row['candidate']=='LAST100_K12' and row['reference']=='SCALAR' and row['passes']
               for row in group['paired']):
            accepted.add(group['task'])
    files.require(accepted==set(transfer.TASKS),'Both A137 tasks must pass the fixed gate')
    plan,_,pilot,_ = dense.load_plan('dps',parent)
    pilot_keys = {case['key'] for case,_,_ in pilot}
    files.require({row['case'] for row in result['cases']}==pilot_keys,'Unexpected A137 development cases')
    holdout = [case for case in plan['cases'] if case['key'] not in pilot_keys]
    ids = {case['image_id'] for case in holdout}
    files.require(len(ids)==7 and len(holdout)==14 and
                  {(case['image_id'],case['task']) for case in holdout}==
                  {(image,task) for image in ids for task in transfer.TASKS} and
                  not {case['image_sha256'] for case in holdout}.intersection(case['image_sha256'] for case,_,_ in pilot),
                  'Confirmation cases must be the remaining seven original images')
    items = []
    for case in holdout:
        path = parent/'cases'/case['key']/'BASE'/'run.json'
        base = files.read(path)
        y,sha = case['a118_y'],case['a118_y_sha256']
        files.require(base['status']=='complete' and base['case']==case and base['y_sha256']==sha and
                      case['sigma']==.05 and files.digest(y)==sha and
                      files.digest(case['image'])==case['image_sha256'],'Frozen confirmation case differs')
        items.append((case,dict(path=y,sha256=sha,baseline=str(path),baseline_sha256=files.digest(path)),base))
    return plan,items


def summarize(rows):
    output = []
    for task in transfer.TASKS:
        subset = [row for row in rows if row['task']==task]
        if len(subset)!=7:
            continue
        comparisons = bridge.paired(subset,'LAST100_K12','SCALAR')
        # bridge.paired's three-image flag is specific to the development panel.
        passes = comparisons['delta']['psnr']>=.1 and comparisons['psnr_wins']>=5 and \
                 comparisons['delta']['ssim']>=0 and comparisons['delta']['lpips']<=0
        comparisons['passes'] = passes
        arms = {arm:dict(quality={key:statistics.mean(row['arms'][arm]['quality'][key] for row in subset)
                                  for key in transfer.METRICS},
                         seconds=statistics.mean(row['arms'][arm]['full_seconds'] for row in subset),
                         nfe=1000,input_vjp=1000) for arm in ARMS}
        output.append(dict(task=task,arms=arms,paired=comparisons,
                           timing_ratio=arms['LAST100_K12']['seconds']/arms['SCALAR']['seconds'],
                           per_image=[dict(case=row['case'],delta_psnr=row['arms']['LAST100_K12']['quality']['psnr']-
                                           row['arms']['SCALAR']['quality']['psnr']) for row in subset]))
    return output


def execute(parent,a137_root,root):
    import numpy as np
    import torch
    from modules.main_single import load_diffusion_model
    from modules.wavelet import OrthogonalDWT2D
    from utils.a67_model_freeze import frozen_network
    from utils.current_score_runtime_r2 import input_vjp_parameters
    from utils.historical_ffhq_runtime import precision

    plan,items = load_inputs(parent,a137_root)
    card = transfer.PROJECTS/'experiment_cards/A138_confirm_dps.md'
    sources = {str(p):files.digest(p) for p in (Path(__file__),card,parent/'plan.json',
               a137_root/'summary.json',a137_root/'resolved_config.json',Path(dense.__file__))}
    path = root/'summary.json'
    if path.exists():
        report = files.read(path)
        files.require(report['status']=='complete' and report['sources']==sources,
                      f'Prior partial or incompatible confirmation retained at {root}')
        for row in report['groups']:
            print('A138 SUMMARY',row,flush=True)
        return
    admission.check(torch,root)
    root.mkdir(parents=True)
    report = dict(experiment='A138',status='running',sources=sources,cases=[],groups=[],actual_new_nfe=0)
    files.write(path,report)
    files.write(root/'resolved_config.json',dict(sources=sources,grid=GRID,positions=development.POSITIONS,
                max_updates=development.depth.MAX_UPDATES,grad_rtol=development.depth.GRAD_RTOL,
                cases=[dict(case,observation=obs) for case,obs,_ in items]))
    started = time.perf_counter()
    try:
        model = load_diffusion_model('ffhq','cuda:0')
        model.model.load_state_dict(torch.load(plan['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
        model.model.eval()
        wave = OrthogonalDWT2D(wave='db4',level=3).to(model.device)
        kernel = np.load(plan['motion_kernel'],allow_pickle=False)
        with precision('LEGACY_IEEE'),frozen_network(model.model),input_vjp_parameters(model.model),native.dps_reference_scope():
            for index,(case,obs,base) in enumerate(items):
                arms = {}
                for arm in ARMS if index%2==0 else reversed(ARMS):
                    if arm=='SCALAR':
                        row = dense.measure(root,'dps',plan,case,obs,model,wave,kernel,arm,GRID,{},base)
                    else:
                        row = development.measure(root,arm,plan,case,obs,base,GRID,model,wave,kernel)
                    arms[arm] = row
                    report['actual_new_nfe'] += row['counts']['forwards']
                    files.write(path,report)
                report['cases'].append(dict(case=case['key'],task=case['task'],arms=arms))
                report['groups'] = summarize(report['cases'])
                files.write(path,report)
                print('A138 PROGRESS',len(report['cases']),'/14 tasks/images',flush=True)
        files.require(report['actual_new_nfe']==28000 and len(report['groups'])==2,'Incomplete confirmation')
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error),actual_nfe_is_lower_bound=True)
        raise
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        files.write(path,report)
    for row in report['groups']:
        print('A138 SUMMARY',row,flush=True)
    print('A138 REPORT',path,'actual_new_nfe=',report['actual_new_nfe'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a124-root',type=Path,default=transfer.OUTPUTS/'A124R1_dps_ffhq')
    parser.add_argument('--a137-root',type=Path,default=development.ROOT)
    parser.add_argument('--output',type=Path,default=ROOT)
    args = parser.parse_args()
    parents = (args.a124_root.resolve(),args.a137_root.resolve())
    output = args.output.resolve()
    files.require(all(output!=p and not output.is_relative_to(p) for p in parents),'Use a separate A138 result directory')
    execute(*parents,output)


if __name__=='__main__':
    main()
