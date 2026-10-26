"""Recompute A54 paired arithmetic from user-supplied per-image summary only."""
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    folder = Path(__file__).resolve().parent
    source = folder / 'user_summary.json'
    value = json.loads(source.read_text())
    assert value['experiment'] == 'A54_current_norm_gain'
    assert value['completed_cases'] == value['planned_cases'] == 10
    assert value['diagnostic_nfe'] == value['backward_attempts'] == value['backward_completed'] == 900
    checks, groups = 0, []
    for group in value['groups']:
        rows = group['per_image']
        assert len(rows) == len({r['image_id'] for r in rows}) == 5
        names = tuple(group['STATE_GAIN_minus'])
        means = {name: {m: mean(r['metrics'][name][m] for r in rows)
                        for m in ('psnr', 'ssim', 'lpips')}
                 for name in ('STATE_GAIN', *names)}
        for name in names:
            for metric in ('psnr', 'ssim', 'lpips'):
                actual = mean(r['metrics']['STATE_GAIN'][metric]-r['metrics'][name][metric] for r in rows)
                assert abs(actual-group['STATE_GAIN_minus'][name][metric]) <= 1e-12
                checks += 1
            positive = sum(r['metrics']['STATE_GAIN']['psnr'] > r['metrics'][name]['psnr'] + 1e-9 for r in rows)
            assert positive == group['positive_images'][name]
            for row in rows:
                assert abs(row['STATE_GAIN_psnr_minus'][name] - (
                    row['metrics']['STATE_GAIN']['psnr']-row['metrics'][name]['psnr'])) <= 1e-12
        region = {}
        if group['task'] == 'inpainting':
            for area in ('observed_unit_MSE', 'unobserved_unit_MSE'):
                average = {name: mean(r['inpainting_regions'][name][area] for r in rows)
                           for name in ('STATE_GAIN', 'TIME_GAIN', 'SHUFFLE_GAIN', 'C_IEEE')}
                region[area] = dict(mean=average, relative_mean_reduction_vs_C_IEEE=
                                   1-average['STATE_GAIN']/average['C_IEEE'])
        groups.append(dict(task=group['task'], means=means,
                           current_gain_increment_supported=False, inpainting_regions=region))
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  checked_mean_effects=checks, groups=groups,
                  limits='Summary arithmetic only. Five reused images, one seed, two tasks. '
                         'Every gain arm retains CURRENT u; no test here of replacing current direction. '
                         '900 is completed control forward count, not all historical/interrupted execution costs. '
                         'No manifest, PT, actual model or operator gate rerun.')
    (folder/'arithmetic_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('A54 summary arithmetic: verified', checks, 'mean effects and per-image PSNR effects; no UNet calls.')
    for group in groups:
        print(group['task'], {n: round(m['psnr'], 6) for n, m in group['means'].items()})


if __name__ == '__main__':
    main()
