"""A34 discrepancy-inspired feedback; fixed grid, six new 300-NFE arms, 2160 diagnostic calls."""
import argparse
import builtins
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import statistics
import sys

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
from utils import diag_continuous_response_feedback as engine
from utils import noise_target_response_feedback as rules

source, P = engine.source, engine.P
ROOT = '/data0/lzy/ZAPS-results/diag_noise_target_response_feedback/A34_20261021'
CARD = PROJECTS/'experiment_cards/A34_noise_target_response_feedback_20261021.md'


@contextmanager
def executor():
    """Inject only the new policy into the immutable, tested A33 replay executor."""
    previous = engine.rules
    sentinel = object()
    previous_print = engine.__dict__.get('print', sentinel)
    engine.rules = rules
    engine.print = lambda *args, **kwargs: builtins.print(
        *(v.replace('A33', 'A34') if isinstance(v, str) else v for v in args), **kwargs)
    try:
        yield
    finally:
        engine.rules = previous
        if previous_print is sentinel:
            del engine.print
        else:
            engine.print = previous_print


def prepare(args):
    root, parent = Path(args.root).resolve(), Path(args.a33_dir).resolve()
    if root.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('A34 must not overlap A33')
    if root.exists():
        plan = source.check_plan(root)
        if plan['source_A33'] != str(parent):
            raise ValueError('Existing A34 parent differs')
        print('A34 frozen plan reused; completed arms retained.', flush=True)
        return
    previous = source.check_plan(parent)
    summary = source.load_signed(parent/'summary.json')
    if (previous['experiment'] != 'A33_continuous_response_feedback'
            or summary['completed_tasks'] != 72 or summary['completed_cases'] != 12):
        raise ValueError('Need the twelve complete A33 cases')
    # Parent profiles must be read under their original A33 policy, before injection.
    profiles = engine.read_profiles(parent, previous)
    identities = dict(previous['identity_files'])
    a33_records = {}
    for spec in previous['cases']:
        P.verify_case_sources(previous, spec)
        key = spec['case']['key']
        a33_records[key] = {}
        for arm in list(engine.rules.POLICIES)+list(engine.rules.PAIRS.values()):
            feedback = next((a for a, b in engine.rules.PAIRS.items() if b == arm), None)
            signature = profiles['policy_sha256'][feedback] if feedback else None
            item = engine.task_done(parent, previous, spec, arm, signature)
            if item is None:
                raise ValueError('Incomplete A33 arm: '+key+'/'+arm)
            path, _ = item
            a33_records[key][arm] = str(path)
            for f in (path, path.with_suffix('.sha256'), path.parent/'output.pt'):
                identities[str(f)] = source.digest(f)
    for name in ('plan.json', 'plan.sha256', 'summary.json', 'summary.sha256',
                 'stage_profiles.json', 'stage_profiles.sha256'):
        identities[str(parent/name)] = source.digest(parent/name)
    workers = dict(previous['worker_identity_files'])
    for f in (Path(__file__), Path(rules.__file__), CARD, PROJECTS/'run_a34_20261021.sh',
              PROJECTS/'utils/test_noise_target_response_feedback.py'):
        workers[str(f)] = source.digest(f)
    for path, sha in workers.items():
        if source.digest(path) != sha:
            raise ValueError('Bound worker changed: '+path)
    plan = dict(previous)
    plan.update(experiment='A34_noise_target_response_feedback', source_A33=str(parent),
                a33_records=a33_records, policies=rules.POLICIES, pairs=rules.PAIRS,
                identity_files=identities, worker_identity_files=workers,
                noise_sigma=rules.SIGMA, minimum_predicted_improvement=None,
                selection_objective='First noise-target crossing, then stop measurement guidance; '
                                    'ray minimum only when target unreachable; unresolved response keeps gamma1.',
                planned_diagnostic_nfe=2160, planned_method_nfe=0,
                scope='Six reused FFHQ development image clusters, two dependent seeds each; final unroll only. '
                      'Discrepancy-inspired same-time surrogate, not a proven next-state constraint, '
                      'exact Jacobian, curvature or reconstruction optimum. Targets .9/1/1.1 are '
                      'model-mismatch sensitivity settings, not noise confidence bounds.')
    root.parent.mkdir(parents=True, exist_ok=True)
    P.coupled.archive.recovery.require_space(root.parent)
    root.mkdir()
    source.signed(root/'plan.json', plan)
    (root/'experiment_card.md').write_text(CARD.read_text())
    print('A34 PREPARE PASS: N090/N100/N110 plus matched TIME; 2160 diagnostic NFE, '
          'all scientific arms 300 NFE. A33 comparisons reused; no fixed gamma cap.', flush=True)


def reference_record(plan, key, arm):
    path = Path(plan['a33_records'][key][arm])
    if source.digest(path) != plan['identity_files'][str(path)]:
        raise ValueError('Bound A33 reference changed')
    return source.load_signed(path)


def pictures(root, plan, spec, tasks):
    import torch
    from PIL import Image, ImageDraw
    from modules.main_single import load_image_as_tensor
    from utils.metrics import _to_uint8_numpy
    from utils.diag_cached_guidance_final_value import font

    key = spec['case']['key']
    native_path = Path(plan['parent_records'][key])
    native = source.load_signed(native_path)
    native_outputs = torch.load(native_path.parent/'outputs.pt', map_location='cpu', weights_only=True)
    arrays = {'GT': _to_uint8_numpy(load_image_as_tensor(spec['case']['image'])),
              'C': _to_uint8_numpy(native_outputs['C'])}
    metrics = {'C': native['metrics']['C']}
    for arm in ('I_cont2', 'STAGE_cont2'):
        path = Path(plan['a33_records'][key][arm])
        if source.digest(path.parent/'output.pt') != plan['identity_files'][str(path.parent/'output.pt')]:
            raise ValueError('A33 tensor changed before picture export')
        arrays[arm] = _to_uint8_numpy(torch.load(path.parent/'output.pt', map_location='cpu', weights_only=True))
        metrics[arm] = reference_record(plan, key, arm)['metrics']
    for arm, (path, record) in tasks.items():
        arrays[arm] = _to_uint8_numpy(torch.load(path.parent/'output.pt', map_location='cpu', weights_only=True))
        metrics[arm] = record['metrics']
    names = ['GT', 'C', 'I_cont2', 'STAGE_cont2']+[a for pair in rules.PAIRS.items() for a in pair if a in arrays]
    sheet = Image.new('RGB', (20+276*len(names), 680), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 8), key+' | A34 noise-target feedback | all arms 300 NFE | full image + same ROI',
              fill='black', font=font(18))
    roi = (64, 48, 192, 176)
    for i, arm in enumerate(names):
        x = 20+276*i
        draw.text((x, 43), arm, fill='black', font=font(17))
        if arm != 'GT':
            m = metrics[arm]
            draw.text((x, 68), f'P {m["psnr"]:.3f} S {m["ssim"]:.4f}', fill='black', font=font(14))
            draw.text((x, 88), f'LPIPS {m["lpips"]:.4f}', fill='black', font=font(14))
        raw = Image.fromarray(arrays[arm])
        marked = raw.copy()
        ImageDraw.Draw(marked).rectangle(roi, outline='#e7a827', width=2)
        sheet.paste(marked, (x, 115))
        sheet.paste(raw.crop(roi).resize((256, 256), Image.Resampling.NEAREST), (x, 390))
    draw.text((20, 653), 'Historical display conversion; fixed ROI, nearest zoom; no enhancement.',
              fill='black', font=font(15))
    sheet.save(root/key/'comparison.png')


def summarize(args):
    root = Path(args.root).resolve()
    plan = source.check_plan(root)
    profiles = engine.read_profiles(root, plan) if (root/'stage_profiles.json').exists() else None
    hashes = {rules.PAIRS[a]: profiles['policy_sha256'][a] for a in profiles['ready_policies']} if profiles else {}
    records, paths, references = {}, {}, {}
    arms = list(rules.POLICIES)+list(rules.PAIRS.values())
    for spec in plan['cases']:
        key = spec['case']['key']
        parent_path = Path(plan['parent_records'][key])
        if source.digest(parent_path) != plan['identity_files'][str(parent_path)]:
            raise ValueError('A32 reference changed')
        native = source.load_signed(parent_path)
        references[key] = {a: dict(case=spec['case'], metrics=native['metrics'][a]) for a in ('C', 'F')}
        for a in ('I_cont2', 'STAGE_cont2'):
            references[key][a] = reference_record(plan, key, a)
        for a in arms:
            item = engine.task_done(root, plan, spec, a, hashes.get(a))
            if item:
                paths[(key, a)], records[(key, a)] = item
    variants = []
    for arm, stage in rules.PAIRS.items():
        feedback = [s['case']['key'] for s in plan['cases'] if (s['case']['key'], arm) in records]
        matched = [k for k in feedback if (k, stage) in records]
        comparisons = {b: engine.difference([(records[(k, arm)], references[k][b]) for k in feedback])
                       for b in ('C', 'F', 'I_cont2', 'STAGE_cont2') if feedback}
        if matched:
            comparisons['matched_STAGE'] = engine.difference([(records[(k, arm)], records[(k, stage)]) for k in matched])
            comparisons['TIME_vs_C'] = engine.difference([(records[(k, stage)], references[k]['C']) for k in matched])
        rows = [v for k in feedback for v in records[(k, arm)]['rows']]
        reasons = {reason: sum(v['reason'] == reason for v in rows) for reason in sorted({v['reason'] for v in rows})}
        values = [v['gamma'] for v in rows]
        stats = dict(decisions=len(rows), gamma_min=min(values) if values else None,
                     gamma_median=statistics.median(values) if values else None,
                     gamma_max=max(values) if values else None,
                     final_residual_noise_rms=math.sqrt(2*statistics.mean(records[(k, arm)]['metrics']['loss'] for k in feedback)) if feedback else None,
                     changed_vs_one=sum(abs(g-1.) > 1e-8 for g in values), reasons=reasons)
        candidate = (len(matched) == 12 and comparisons['C']['mean_delta']['psnr'] > 0
                     and all(comparisons[b]['mean_delta']['psnr'] >= .01 for b in ('STAGE_cont2', 'matched_STAGE'))
                     and all(comparisons[b]['quality_guard'] for b in ('C', 'STAGE_cont2', 'matched_STAGE'))
                     and comparisons['matched_STAGE']['images_positive'] >= 4)
        variants.append(dict(arm=arm, R_target=rules.POLICIES[arm], completed_feedback=len(feedback),
                             completed_time=len(matched), classification='DEVELOPMENT_STATE_CANDIDATE_ONLY' if candidate else
                             ('NO_CONFIRMED_STATE_INCREMENT' if len(matched) == 12 else 'INCOMPLETE_TARGET'),
                             comparisons=comparisons, readout=stats))
    cost, partial, lower_bound, failures = 0, 0, False, []
    for path in sorted(root.glob('*/*/attempt_*/run.json')):
        r = source.read(path)
        cost += r['diagnostic_nfe']
        partial += r['status'] != 'complete'
        lower_bound |= r['status'] in ('running', 'sampled')
        if r['status'] == 'failed':
            failures.append(dict(case=r['case']['key'], arm=r['arm'], error=r.get('error')))
    result = dict(experiment=plan['experiment'], completed_cases=sum(all((s['case']['key'], a) in records for a in arms) for s in plan['cases']),
                  planned_cases=12, image_clusters=6, completed_tasks=len(records), planned_tasks=72,
                  method_nfe=0, diagnostic_nfe=cost, planned_diagnostic_nfe=2160,
                  deployment_nfe_per_arm=300, sensor_extra_unet=0, partial_attempts=partial,
                  cost_is_lower_bound_if_abruptly_interrupted=lower_bound, variants=variants,
                  classification='COMPLETE_DEVELOPMENT_COMPARISON' if len(records) == 72 else 'INCOMPLETE_PILOT',
                  failed_attempts=failures, limits=plan['scope']+' No independent confirmation or full-ten-epoch adaptation. '
                  'All targets reported; a selected candidate needs frozen independent confirmation. '
                  'Predicted same-time residual and next evaluated clean residual include different diffusion states.')
    source.signed(root/'summary.json', result)
    with (root/'state_traces.csv').open('w', newline='') as out:
        columns = ['case', 'arm', 'step', 't', 'gamma', 'raw_gamma', 'R', 'R_target', 'predicted_R_after',
                   'next_evaluated_clean_R', 'cosine', 'relative_response', 'qRMS', 'guidance_rms', 'clip', 'reason', 'fallback']
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for (key, arm), record in sorted(records.items()):
            for k, row in enumerate(record['rows']):
                nxt = record['rows'][k+1]['R'] if k < 29 else math.sqrt(2*record['metrics']['loss'])
                writer.writerow(dict(case=key, arm=arm, next_evaluated_clean_R=nxt, **row))
    if args.archive:
        for spec in plan['cases']:
            key = spec['case']['key']
            tasks = {a: (paths[(key, a)], r) for (k, a), r in records.items() if k == key}
            if tasks:
                pictures(root, plan, spec, tasks)
    print('A34 Summary:', json.dumps(result, ensure_ascii=False), flush=True)
    print('Records:', root, flush=True)
    if args.archive:
        from utils.archive_refinement_confirmation import export_and_commit
        bundle, commit = export_and_commit({'A34': str(root)})
        print('Archive:', bundle, 'Results commit:', commit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT)
    parser.add_argument('--a33-dir', default=engine.ROOT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--build-profiles', action='store_true')
    parser.add_argument('--phase', choices=('feedback', 'stage'))
    parser.add_argument('--shard', type=int, choices=(0, 1))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    else:
        with executor():
            if args.build_profiles:
                engine.build_profiles(args)
            elif args.phase:
                if args.shard is None:
                    parser.error('--phase requires --shard')
                engine.worker(args)
            else:
                summarize(args)


if __name__ == '__main__':
    main()
