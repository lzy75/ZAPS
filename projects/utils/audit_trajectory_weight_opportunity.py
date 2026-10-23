"""A35b: cached action-selection headroom, no model/tensor loading or UNet calls."""
import argparse
import json
from pathlib import Path
import statistics
import sys

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
POINTS = ((8, 903), (19, 641), (24, 319), (27, 54), (28, 14))
FACTORS = (1., .5, 1.5)  # Tie order: keep, weaken, strengthen.
METRICS = ('psnr', 'ssim', 'lpips', 'loss')
ROOT = '/data0/lzy/ZAPS-results/diag_trajectory_weight_opportunity/A35b_20261023'
CARD = PROJECTS/'experiment_cards/A35b_cached_weight_opportunity_20261023.md'


def average(items):
    return {m: statistics.mean(v[m] for v in items) for m in METRICS}


def guard(delta):
    return delta['ssim'] >= -.002 and delta['lpips'] <= .005


def analyze(records):
    if len({(r['case']['image_id'], r['case']['sampling_seed']) for r in records}) != len(records):
        raise ValueError('Duplicate image/seed records')
    images = sorted({r['case']['image_id'] for r in records})
    if len(images) < 3:
        raise ValueError('Need three or more image clusters')
    if any(len({r['case']['sampling_seed'] for r in records if r['case']['image_id'] == i}) != 2 for i in images):
        raise ValueError('Need two seeds per image')
    stages = {}
    for k, t in POINTS:
        cases = []
        for r in records:
            baseline = r['metrics']['BASE']
            delta = {1.: {m: 0. for m in METRICS}}
            for f in (.5, 1.5):
                delta[f] = {m: r['metrics'][f'T{t}_M{f:g}'][m]-baseline[m] for m in METRICS}
            cases.append(dict(image_id=r['case']['image_id'], seed=r['case']['sampling_seed'], delta=delta))
        by_image = {i: {f: average([c['delta'][f] for c in cases if c['image_id'] == i])
                        for f in FACTORS} for i in images}
        fixed = {f: average([v[f] for v in by_image.values()]) for f in FACTORS}
        best = max(FACTORS, key=lambda f: fixed[f]['psnr'])
        image_choice = {i: max(FACTORS, key=lambda f: by_image[i][f]['psnr']) for i in images}
        image_oracle = average([by_image[i][image_choice[i]] for i in images])
        case_oracle = average([c['delta'][max(FACTORS, key=lambda f: c['delta'][f]['psnr'])] for c in cases])
        time_choice = {i: max(FACTORS, key=lambda f: statistics.mean(by_image[j][f]['psnr'] for j in images if j != i))
                       for i in images}
        time_result = average([by_image[i][time_choice[i]] for i in images])
        stages[str(t)] = dict(step=k,
            fixed={str(f): dict(mean_delta=v, quality_guard=guard(v),
                                images_psnr_positive=sum(by_image[i][f]['psnr'] > 1e-9 for i in images))
                   for f, v in fixed.items()},
            hindsight_best_fixed_factor=best, hindsight_best_fixed_delta=fixed[best],
            image_oracle=dict(mean_delta=image_oracle, quality_guard=guard(image_oracle),
                              psnr_headroom_over_best_fixed=image_oracle['psnr']-fixed[best]['psnr'],
                              counts={str(f): sum(v == f for v in image_choice.values()) for f in FACTORS}),
            case_oracle=dict(mean_delta=case_oracle, quality_guard=guard(case_oracle),
                             psnr_headroom_over_best_fixed=case_oracle['psnr']-fixed[best]['psnr']),
            leave_image_out_time=dict(mean_delta=time_result, quality_guard=guard(time_result),
                                     counts={str(f): sum(v == f for v in time_choice.values()) for f in FACTORS}),
            per_image={i: dict(deltas={str(f): by_image[i][f] for f in FACTORS},
                              hindsight_best_factor=image_choice[i], held_time_factor=time_choice[i]) for i in images})
    return dict(classification='CACHED_ACTION_OPPORTUNITY_DIAGNOSTIC_ONLY', image_clusters=len(images),
                completed_cases=len(records), stages=stages,
                mean_image_oracle_headroom=statistics.mean(v['image_oracle']['psnr_headroom_over_best_fixed'] for v in stages.values()),
                max_image_oracle_headroom=max(v['image_oracle']['psnr_headroom_over_best_fixed'] for v in stages.values()),
                limits='Post-hoc diagnostic on six reused development images. GT oracle is an unattainable upper bound, '
                       'never an online action or output selector. Image oracle shares one action across its two seeds; '
                       'case oracle can change action by seed. Neither oracle optimizes SSIM/LPIPS constraints; guards are '
                       'reported afterward. Best fixed is selected retrospectively on this cohort; held time excludes '
                       'both seeds of each image. Averages across positions describe separate single-pulse trials, '
                       'not one multi-pulse policy. Bounds apply only to tested states/actions; no global impossibility claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a35-dir', default='/data0/lzy/ZAPS-results/diag_trajectory_geometry_value/A35_20261021')
    p.add_argument('--output-dir', default=ROOT)
    p.add_argument('--archive', action='store_true')
    args = p.parse_args()
    from utils import diag_trajectory_geometry_value as parent
    source = parent.source
    old, root = Path(args.a35_dir).resolve(), Path(args.output_dir).resolve()
    if root.is_relative_to(old) or old.is_relative_to(root):
        raise ValueError('Audit output overlaps original A35')
    # Coordinate with A35's final picture/archive stage; never race a results ref update.
    import fcntl
    with (old.parent/'launcher.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A35 launcher is still finishing. Wait for its final RESULTS_PUSH line, then repeat this zero-NFE audit.')
        plan = source.check_plan(old)
        summary = source.load_signed(old/'summary.json')
        if plan['experiment'] != 'A35_trajectory_geometry_value' or summary['completed_cases'] != 12:
            raise ValueError('Need twelve complete A35 cases')
        records, identities = [], {}
        for name in ('plan.json', 'summary.json', 'analysis.json'):
            source.load_signed(old/name)
            identities[str(old/name)] = source.digest(old/name)
        for spec in plan['cases']:
            item = parent.completed(old, plan, spec)
            if item is None:
                raise ValueError('Incomplete A35 case')
            path, record = item
            identities[str(path)] = source.digest(path)
            records.append(record)
        result = analyze(records)
        # Reconcile the original printed fixed-action effects, rather than trusting a pasted table.
        for t, stage in result['stages'].items():
            for f in (.5, 1.5):
                saved = summary['fixed_pulse_effects'][f'T{t}_M{f:g}']['mean_delta']
                if any(abs(stage['fixed'][str(f)]['mean_delta'][m]-saved[m]) > 1e-12 for m in METRICS):
                    raise ValueError('Source summary disagrees with per-case metrics')
        result.update(experiment='A35b_cached_weight_opportunity', status='complete', method_nfe=0,
                      diagnostic_nfe=0, source_A35=str(old), identity_files=identities,
                      source_diagnostic_nfe=summary['diagnostic_nfe'],
                      code_sha256=source.digest(Path(__file__)), card_sha256=source.digest(CARD))
        root.mkdir(parents=True, exist_ok=True)
        output = root/'summary.json'
        if output.exists() and source.load_signed(output) != result:
            raise ValueError('Existing audit differs; preserve original')
        source.signed(output, result)
        (root/'experiment_card.md').write_text(CARD.read_text())
        compact = {k: v for k, v in result.items() if k not in ('stages', 'identity_files')}
        compact['stages'] = {t: {k: v for k, v in stage.items() if k != 'per_image'} for t, stage in result['stages'].items()}
        print('A35b Summary:', json.dumps(compact, ensure_ascii=False), flush=True)
        print('Records:', root, '; UNet NFE=0.', flush=True)
        if args.archive:
            from utils.archive_refinement_confirmation import export_and_commit
            bundle, commit = export_and_commit({'A35': str(old), 'A35b': str(root)})
            print('Archive:', bundle, 'Results commit:', commit, flush=True)


if __name__ == '__main__':
    main()
