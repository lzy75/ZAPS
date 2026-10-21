"""A29: post-A28 exploratory stage/ablation/action audit; NumPy only, zero UNet calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import state_action_response as a28

SOURCE = '/data0/lzy/ZAPS-results/diag_state_action_response/A28_20261021'
OUTPUT = '/data0/lzy/ZAPS-results/diag_state_action_response/A29_20261021'
BASE = a28.FEATURES['PHYSICAL']
MODELS = dict(STAGE=(), PHYSICAL=BASE, HISTORY=BASE+('history_turn',),
              DIR_ONLY=BASE+('q_dir', 'travel_dir'), MAG_ONLY=BASE+('q_mag', 'travel_mag'),
              LOCAL=a28.FEATURES['LOCAL'], RESPONSE_ERROR_ONLY=BASE+('mean_error',),
              RESPONSE_PHYS_ONLY=BASE+('mean_guidance_gain',), RESPONSE=a28.FEATURES['RESPONSE'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_signed(path):
    if sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Source SHA differs: '+str(path))
    return read(path)


def audit(source):
    plan = read_signed(source/'plan.json'); summary = read_signed(source/'summary.json')
    if (summary['completed_cases'] != 12 or summary['completed_states'] != 72 or summary['partial_attempts'] != 0
            or summary['diagnostic_nfe'] != 2736 or summary['method_nfe'] != 0):
        raise ValueError('Need the complete original A28 without partial attempts')
    expected = {(f'{i:05d}', seed, step) for i in range(20, 26) for seed in (1000, 2000) for step in a28.STEPS}
    points = summary['points']
    if len(points) != 72 or {(p['image_id'], p['seed'], p['step']) for p in points} != expected:
        raise ValueError('Original point identity/coverage differs')
    identities = {str(source/name): sha(source/name) for name in ('plan.json', 'plan.sha256', 'summary.json', 'summary.sha256')}
    for spec in plan['cases']:
        case = spec['case']; folder = source/case['key']; gate = read_signed(folder/'case_gate.json')
        if not gate['passed'] or gate['plan_sha256'] != sha(source/'plan.json'):
            raise ValueError('A28 case gate differs')
        for name in ('case_gate.json', 'case_gate.sha256'):
            identities[str(folder/name)] = sha(folder/name)
        for step in a28.STEPS:
            runs = list((folder/f'k{step:02d}').glob('attempt_*/run.json'))
            if len(runs) != 1:
                raise ValueError('Need one original completed attempt per state')
            path = runs[0]; record = read_signed(path)
            match = next(p for p in points if (p['image_id'], p['seed'], p['step']) == (case['image_id'], case['sampling_seed'], step))
            if (record['status'] != 'complete' or record['case'] != case or record['step'] != step or
                    record['plan_sha256'] != sha(source/'plan.json') or record['point'] != match or
                    record['diagnostic_nfe'] != 38 or len(record['calls']) != 38 or
                    not all(g['passed'] for g in match['gates'].values())):
                raise ValueError('A28 point completion contract differs')
            identities[str(path)] = sha(path); identities[str(path.with_suffix('.sha256'))] = sha(path.with_suffix('.sha256'))
    return points, identities


def models_on(rows, names):
    result = {}
    for target in a28.TARGETS:
        result[target] = {name: a28.cross_validate(rows, MODELS[name], target) for name in names}
    return result


def stage_analysis(points):
    rows = [a28.derive(p) for p in points]; stages = []
    for step in a28.STEPS:
        group = [r for r in rows if r['step'] == step]
        common = [r for r in group if all(r['features'][f] is not None for fs in MODELS.values() for f in fs)]
        known = ('STAGE', 'PHYSICAL', 'HISTORY', 'RESPONSE_ERROR_ONLY', 'RESPONSE_PHYS_ONLY', 'RESPONSE')
        full = models_on(group, known)
        reliable = len(common) >= 8 and len({r['image_id'] for r in common}) == 6
        local = models_on(common, tuple(MODELS)) if reliable else None
        stages.append(dict(step=step, t=next(p['t'] for p in points if p['step'] == step),
                           total=len(group), local_resolved=len(common), local_status='DESCRIPTIVE_ONLY' if reliable else 'UNRESOLVED',
                           full=full, local=local))
    aggregates = {}
    for scope in ('full', 'local'):
        available = [s for s in stages if s[scope] is not None]
        aggregate = dict(steps=[s['step'] for s in available], targets={})
        for target in a28.TARGETS:
            names = tuple(available[0][scope][target]) if available else ()
            aggregate['targets'][target] = {}
            for name in names:
                comparisons = {}
                for baseline in ('STAGE', 'PHYSICAL'):
                    eligible = [s for s in available if s[scope][target][baseline]['mae'] > 1e-10]
                    ratios = [s[scope][target][name]['mae']/s[scope][target][baseline]['mae'] for s in eligible]
                    images = sorted({r['image_id'] for r in rows})
                    per_image = {i: float(np.mean([(s[scope][target][name]['per_image_mae'][i]-s[scope][target][baseline]['per_image_mae'][i])/
                                 s[scope][target][baseline]['mae'] for s in eligible])) for i in images} if eligible else {}
                    comparisons[baseline] = dict(equal_stage_relative_mae_reduction=1-float(np.mean(ratios)) if ratios else None,
                        stages_used=[s['step'] for s in eligible], images_better=sum(v < -1e-10 for v in per_image.values()),
                        per_image_normalized_error_difference=per_image)
                aggregate['targets'][target][name] = comparisons
        aggregates[scope] = aggregate
    return dict(stages=stages, equal_stage_aggregates=aggregates)


def choose(actions, key, maximize=False):
    values = [float(a[key[0]][key[1]]) for a in actions]
    optimum = max(values) if maximize else min(values)
    eligible = [a for a, v in zip(actions, values) if abs(v-optimum) <= 1e-12]
    return min(eligible, key=lambda a: (abs(a['gamma']-1), a['gamma']))['gamma']


def stage_choice(points, held_image, step, factor, quality=False):
    train = [p for p in points if p['image_id'] != held_image and p['step'] == step]
    if not train:
        raise ValueError('No training images for stage choice')
    candidates = []
    for gamma in a28.GAMMAS:
        selected = [next(a for a in p['actions'] if a['factor'] == factor and a['gamma'] == gamma) for p in train]
        candidates.append(dict(gamma=gamma, mean=dict(loss=float(np.mean([a['mean']['loss'] for a in selected]))),
                               stochastic=dict(psnr=float(np.mean([a['stochastic']['psnr'] for a in selected])))))
    return choose(candidates, ('stochastic', 'psnr') if quality else ('mean', 'loss'), maximize=quality)


def action_audit(points):
    output = []
    for factor in a28.SPANS:
        choices = []
        for p in points:
            actions = [a for a in p['actions'] if a['factor'] == factor]
            selected = dict(G1=1., G2=2., STATE_PHYS=choose(actions, ('mean', 'loss')),
                            STAGE_PHYS=stage_choice(points, p['image_id'], p['step'], factor),
                            STAGE_QUALITY=stage_choice(points, p['image_id'], p['step'], factor, quality=True),
                            ORACLE=choose(actions, ('stochastic', 'psnr'), maximize=True))
            choices.append(dict(image_id=p['image_id'], seed=p['seed'], step=p['step'], choices=selected,
                          outcomes={name: next(a['stochastic'] for a in actions if a['gamma'] == gamma) for name, gamma in selected.items()}))
        stage_rows = []
        for step in a28.STEPS:
            group = [r for r in choices if r['step'] == step]
            effects = {}
            for baseline in ('G1', 'G2', 'STAGE_PHYS', 'STAGE_QUALITY'):
                images = sorted({r['image_id'] for r in group})
                per_image = {i: float(np.mean([r['outcomes']['STATE_PHYS']['psnr']-r['outcomes'][baseline]['psnr']
                             for r in group if r['image_id'] == i])) for i in images}
                effects[baseline] = dict(psnr_delta=float(np.mean(list(per_image.values()))),
                    loss_delta=float(np.mean([r['outcomes']['STATE_PHYS']['loss']-r['outcomes'][baseline]['loss'] for r in group])),
                    images_positive=sum(v > 1e-10 for v in per_image.values()), per_image_psnr_delta=per_image)
            # In-sample stage oracle is an opportunity upper bound only.
            stage_points = [p for p in points if p['step'] == step]
            fixed_scores = {str(g): float(np.mean([next(a['stochastic']['psnr'] for a in p['actions'] if a['factor'] == factor and a['gamma'] == g)
                                                   for p in stage_points])) for g in a28.GAMMAS}
            oracle = float(np.mean([r['outcomes']['ORACLE']['psnr'] for r in group]))
            hist = {name: {str(g): sum(r['choices'][name] == g for r in group) for g in a28.GAMMAS}
                    for name in ('STATE_PHYS', 'STAGE_PHYS', 'STAGE_QUALITY', 'ORACLE')}
            stage_rows.append(dict(step=step, state_effects=effects, action_histogram=hist,
                                   oracle_headroom_over_in_sample_best_stage_fixed=oracle-max(fixed_scores.values()),
                                   state_disagrees_with_stage_physics=sum(r['choices']['STATE_PHYS'] != r['choices']['STAGE_PHYS'] for r in group)))
        output.append(dict(factor=factor, stages=stage_rows, decisions=choices))
    return output


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('A29 output must be separate from A28')
    points, identities = audit(source)
    for path in (Path(__file__), Path(a28.__file__), PROJECTS/'experiment_cards/A29_cached_stage_action_audit_20261021.md',
                 PROJECTS/'utils/test_cached_stage_action_audit.py'):
        identities[str(path)] = sha(path)
    plan = dict(experiment='A29_cached_stage_action_audit', source=str(source), identities=identities,
                method_nfe=0, diagnostic_nfe=0, posthoc=True, models={k: list(v) for k, v in MODELS.items()})
    result = dict(experiment=plan['experiment'], method_nfe=0, diagnostic_nfe=0, posthoc=True,
                  classification='EXPLORATORY_STAGE_AND_ACTION_AUDIT_ONLY', state_analysis=stage_analysis(points),
                  action_analysis=action_audit(points),
                  limits='Reused six development images; stage-wise posthoc LOIO. No changed A28 primary result, independent confirmation, final reconstruction benefit or free online probes. Only lightweight records audited; original CUDA tensors not replayed.')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (('plan.json', plan), ('summary.json', result)):
        path = output/name
        encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        if path.exists() and path.read_text() != encoded:
            raise ValueError('Existing A29 differs; use a distinct run directory')
        path.write_text(encoded); path.with_suffix('.sha256').write_text(sha(path)+'\n')
    return result


def compact(result):
    state = result['state_analysis']; decisions = []
    for entry in result['action_analysis']:
        if entry['factor'] != 1.:
            continue
        for s in entry['stages']:
            decisions.append(dict(step=s['step'], state_psnr_delta={b: v['psnr_delta'] for b, v in s['state_effects'].items()},
                                  oracle_headroom=s['oracle_headroom_over_in_sample_best_stage_fixed'],
                                  state_action_histogram=s['action_histogram']['STATE_PHYS'],
                                  disagrees_stage_physics=s['state_disagrees_with_stage_physics']))
    aggregate = {}
    for scope, value in state['equal_stage_aggregates'].items():
        aggregate[scope] = dict(steps=value['steps'], targets={})
        for target, models in value['targets'].items():
            aggregate[scope]['targets'][target] = {n: {b: dict(reduction=c['equal_stage_relative_mae_reduction'],
                 images_better=c['images_better']) for b, c in comparisons.items()} for n, comparisons in models.items() if n not in ('STAGE', 'PHYSICAL')}
    return dict(experiment=result['experiment'], method_nfe=0, diagnostic_nfe=0, posthoc=True,
                classification=result['classification'], stage_coverage=[{k: s[k] for k in ('step', 't', 'total', 'local_resolved', 'local_status')} for s in state['stages']],
                equal_stage_analysis=aggregate, original_span_action_audit=decisions, limits=result['limits'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE); parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args(); result = run(args.source, args.output)
    print('A29 Summary:', json.dumps(compact(result), ensure_ascii=False), flush=True)
    print('Records:', args.output, flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle, commit = export_and_commit({'A29': str(Path(args.output).resolve())})
        print('Archive:', bundle, 'Results commit:', commit, flush=True)


if __name__ == '__main__':
    main()
