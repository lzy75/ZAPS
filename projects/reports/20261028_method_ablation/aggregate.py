"""Reproduce the tables from audited light records; no model execution."""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = read(HERE / 'provenance.json')
    for item in audit['local_inputs']:
        assert digest(REPO / item['path']) == item['sha256'], item['path']
    full = read(REPO / audit['local_inputs'][0]['path'])
    blur = read(REPO / audit['local_inputs'][1]['path'])
    records = read(HERE / 'verified_timing_records.json')
    assert len(records) == 50
    sampling = {r['case']['key']: r for r in records if r['phase'] != 'BASELINE'}
    baselines = [r for r in records if r['phase'] == 'BASELINE']
    rows = []
    groups = [(g['dataset'], g['task'], g['per_image'], 'A56') for g in full['groups']]
    for g in blur['groups']:
        groups.append(('ffhq', g['task'], [p for p in blur['per_image'] if p['case']['task'] == g['task']], 'A62'))
    for dataset, task, cases, experiment in groups:
        for arm in cases[0]['metrics']:
            raw = [sampling[c['case']['key']]['arm_seconds'][arm] for c in cases]
            probes = [sampling[c['case']['key']]['metrology_seconds'] if arm == 'C_IEEE' else 0. for c in cases]
            elapsed = [a-b for a,b in zip(raw, probes)]
            base_raw = [sampling[c['case']['key']]['arm_seconds']['C_IEEE'] for c in cases]
            base_probe = [sampling[c['case']['key']]['metrology_seconds'] for c in cases]
            base_seconds = st.mean(a-b for a,b in zip(base_raw, base_probe))
            row = dict(experiment=experiment, dataset=dataset, task=task, images=len(cases), arm=arm,
                       psnr=st.mean(c['metrics'][arm]['psnr'] for c in cases),
                       ssim=st.mean(c['metrics'][arm]['ssim'] for c in cases),
                       lpips=st.mean(c['metrics'][arm]['lpips'] for c in cases),
                       mean_last30_seconds=st.mean(elapsed), median_last30_seconds=st.median(elapsed),
                       raw_mean_last30_seconds=st.mean(raw), subtracted_metrology_seconds_total=sum(probes),
                       mean_last30_time_ratio_to_C_IEEE=st.mean(elapsed)/base_seconds,
                       input_VJPs_per_last30=(30 if arm in ('U_CURRENT', 'TIME_GAIN', 'U_FFT') else
                           (st.mean(sampling[c['case']['key']]['backward_by_arm'][arm] for c in cases) if arm == 'OLD_U' else 0)),
                       peak_allocated_GiB_max=max((sampling[c['case']['key']]['arm_peak_memory_bytes'].get(arm,0) for c in cases),default=0)/2**30 or None,
                       resolved_steps_total=(sum(c['resolved'].get(arm, 0) for c in cases) if arm not in ('C_REPLAY', 'C_IEEE') else ''),
                       cohort='unused images, one seed' if experiment == 'A56' else 'reused development images, one seed',
                       timing_scope='last 30 fixed-grid steps only; diagnostics and shared-load effects retained; no full deployment timing')
            rows.append(row)
    assert len(rows) == 32
    for arm, expected in full['recorded_last_unroll_seconds'].items():
        actual = sum(r['arm_seconds'][arm] for r in sampling.values() if r['experiment'] == 'A56')
        assert abs(actual-expected) < 1e-8, (arm,actual,expected)
    for arm, expected in blur['measured_seconds_by_arm'].items():
        actual = sum(r['arm_seconds'][arm] for r in sampling.values() if r['experiment'] == 'A62')
        assert abs(actual-expected) < 1e-8, (arm,actual,expected)
    assert abs(sum(r['sampling_optimization_seconds'] for r in baselines)-full['baseline_train_seconds']) < 1e-8
    with (HERE / 'ablation_quality_time.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    baseline_rows=[]
    for dataset,task,cases,experiment in groups:
        b=[r for r in baselines if r['case']['dataset']==dataset and r['case']['task']==task]
        if experiment=='A62':b=[]
        baseline_rows.append(dict(dataset=dataset,task=task,experiment=experiment,
            measured_native_ZAPS_300_forward_optimize_mean_seconds=st.mean(r['sampling_optimization_seconds'] for r in b) if b else None,
            measured_full_U_deployment_seconds=None,
            note='300-forward native optimization time available; no same-run full U deployment time' if b else 'only cached-prefix last30 timing available'))
    (HERE / 'aggregated.json').write_text(json.dumps(dict(rows=rows,full_protocol_timing=baseline_rows),ensure_ascii=False,indent=2)+'\n')
    lines=['| 数据集 / 任务 | 组 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 末轮30步秒/图 ↓ | 相对C_IEEE时间 |',
           '|---|---|---:|---:|---:|---:|---:|']
    names={'super_resolution':'SR','inpainting':'IP','gaussian_deblur':'Gaussian','motion_deblur':'Motion'}
    for r in rows:
        lines.append(f"| {r['dataset'].upper()} / {names[r['task']]} | {r['arm']} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['mean_last30_seconds']:.2f} | {r['mean_last30_time_ratio_to_C_IEEE']:.2f}× |")
    (HERE / 'TABLE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='PASS',audited_light_records=len(records),ablation_rows=len(rows),baseline_timing=baseline_rows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
