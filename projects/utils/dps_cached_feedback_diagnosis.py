"""Zero-network offline audit; output metrics cannot change controls or multipliers."""
import math
import statistics


def diagnose(cases,references,traces):
    records={(r['case']['key'],r['arm']):r for r in references}
    rows=[]
    for c in cases:
        trace=traces[c['key']]
        if len(trace)!=1000 or [v['t'] for v in trace]!=list(range(999,-1,-1)):
            raise ValueError('Need full aligned native1000 trace')
        base=records[c['key'],'DPS'];new=records[c['key'],'SHRINK']
        delta={k:new['metrics'][k]-base['metrics'][k] for k in ('psnr','ssim','lpips')}
        if not all(math.isfinite(v) for v in delta.values()):raise ValueError('Nonfinite offline label')
        noise=c['sigma']**2
        ratios={arm:records[c['key'],arm]['metrics'].get('measurement_mse',None) for arm in ('DPS','SHRINK')}
        blocks=[]
        for i in range(0,1000,100):
            part=trace[i:i+100]
            blocks.append(dict(from_t=part[0]['t'],to_t=part[-1]['t'],
                mean_gamma=statistics.mean(v['gamma'] for v in part),
                mean_weight=statistics.mean(v['weight'] for v in part),
                mean_residual_noise_ratio=statistics.mean(v['residual_energy']/v['noise_energy'] for v in part),
                proxy_measurement_displacement_rms=statistics.mean(v['gamma']*math.sqrt(v['response_energy']*noise/v['noise_energy']) for v in part)))
        rows.append(dict(key=c['key'],task=c['task'],image_id=c['image_id'],delta=delta,
            final_measurement_mse_noise_ratio={a:None if x is None else x/noise for a,x in ratios.items()},
            psnr_down_ssim_up_lpips_down=delta['psnr']<0 and delta['ssim']>0 and delta['lpips']<0,
            gamma_min=min(v['gamma'] for v in trace),gamma_median=statistics.median(v['gamma'] for v in trace),
            gamma_max=max(v['gamma'] for v in trace),zero_gamma_steps=sum(v['gamma']==0 for v in trace),blocks=blocks))
    return dict(method_nfe=0,input_vjp=0,scope='Cached A77 labels and traces only; labels never select donors or gamma.',
        psnr_negative_images=[dict(key=r['key'],delta=r['delta'],final_measurement_mse_noise_ratio=r['final_measurement_mse_noise_ratio'],
                                  psnr_down_ssim_up_lpips_down=r['psnr_down_ssim_up_lpips_down']) for r in rows if r['delta']['psnr']<0],
        limits='Same-time response proxy only. Block statistics are descriptive, not causal proof, true next-denoiser response or trajectory curvature.',by_case=rows)
