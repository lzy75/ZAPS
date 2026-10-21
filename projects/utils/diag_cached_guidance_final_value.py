"""A31: cached fixed-grid guidance -> FINAL quality, NumPy/Pillow only, zero NFE."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

PROJECTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECTS))
TIMES = (903, 641, 319, 54, 14)
GAMMAS = (.5, 2.)
METRICS = ('psnr', 'ssim', 'lpips', 'loss')
MODELS = dict(STAGE=(), PHYSICAL=('log_R', 'log_a', 'clip'),
              DIRECTION=('turn', 'log1p_omega'),
              DUAL=('log_R', 'log_a', 'clip', 'turn', 'log1p_omega'))
LIMITS = ('Posthoc reused six development images; two dependent seeds/image. '
          'Single-pulse outcomes only, not a multi-step feedback policy or independent confirmation. '
          'Training-only transforms; no p-values; no proven curvature or free paid probes.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(path, signature=None):
    path = Path(path)
    if sha(path) != (signature or path.with_suffix('.sha256')).read_text().strip():
        raise ValueError('SHA mismatch: '+str(path))
    return json.loads(path.read_text())


def load_a22(root):
    root = Path(root); plan = signed(root/'plan.json'); rows = []; identities = {}
    expected = {f'{i:05d}_s{s}' for i in range(20, 26) for s in (1000, 2000)}
    if {s['case']['key'] for s in plan['cases']} != expected or len(plan['cases']) != 12:
        raise ValueError('Expected twelve original A22 cases')
    for spec in plan['cases']:
        path = root/spec['case']['key']/'run.json'; r = signed(path, path.parent/'run.sha256')
        if (r['status'] != 'complete' or r['case'] != spec['case'] or r['plan_sha256'] != sha(root/'plan.json')
                or r['method_nfe'] != 0 or r['diagnostic_nfe'] != 108
                or not r['gates'] or not all(g['passed'] for g in r['gates'].values())):
            raise ValueError('A22 incomplete/incompatible: '+str(path))
        if [s['t'] for s in r['action_states']] != list(TIMES): raise ValueError('Changed A22 positions')
        identities[str(path)] = sha(path)
        for state in r['action_states']:
            if not state['features']['usable']: raise ValueError('Unresolved state retained: '+str(path))
            feat = {k: float(v) for k, v in state['features'].items() if k in set(sum(MODELS.values(), ())) }
            feat['clip'] = state['clip_fraction']
            for gamma in GAMMAS:
                arm = f'T{state["t"]}_G{gamma:g}'
                rows.append(dict(case=spec['case']['key'], image_id=spec['case']['image_id'],
                    seed=spec['case']['sampling_seed'], t=state['t'], gamma=gamma, arm=arm,
                    features=feat, delta={m: r['branches'][arm][m]-r['branches']['C'][m] for m in METRICS}))
    if len(rows) != 120: raise ValueError('A22 coverage differs')
    identities[str(root/'plan.json')] = sha(root/'plan.json')
    return plan, rows, identities


def predict(train, test, columns, metric):
    """No test targets are accepted. All centering and fitting uses training only."""
    y = np.array([r['delta'][metric] for r in train]); ym = float(y.mean())
    if not columns: return np.full(len(test), ym)
    x = np.array([[r['features'][f] for f in columns] for r in train])
    xt = np.array([[r['features'][f] for f in columns] for r in test])
    center = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-12] = 1.
    x = (x-center)/scale; xt = (xt-center)/scale
    beta = np.linalg.solve(x.T@x+np.eye(len(columns)), x.T@(y-ym))
    return ym+xt@beta


def rank(v):
    a = np.array(v); order = np.argsort(a, kind='stable'); out = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i+1
        while j < len(a) and a[order[j]] == a[order[i]]: j += 1
        out[order[i:j]] = (i+j-1)/2; i = j
    return out


def corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return None
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def analyze(rows):
    images = sorted({r['image_id'] for r in rows}); layers = []; relations = []
    image_rows = []
    for t in TIMES:
        for gamma in GAMMAS:
            group = [r for r in rows if r['t'] == t and r['gamma'] == gamma]
            per_image = {i: {m: float(np.mean([r['delta'][m] for r in group if r['image_id'] == i])) for m in METRICS} for i in images}
            avg = {m: float(np.mean([d[m] for d in per_image.values()])) for m in METRICS}
            layers.append(dict(t=t, gamma=gamma, mean_delta=avg, per_image_delta=per_image,
                positive_images=sum(d['psnr'] > 1e-9 for d in per_image.values()),
                quality_guard=avg['ssim'] >= -.002 and avg['lpips'] <= .005))
            for i in images:
                rr = [r for r in group if r['image_id'] == i]
                image_rows.append(dict(t=t, gamma=gamma, image_id=i, delta=per_image[i],
                    features={f: float(np.mean([r['features'][f] for r in rr])) for f in MODELS['DUAL']}))
            for f in MODELS['DUAL']:
                xx = [r['features'][f] for r in image_rows if r['t'] == t and r['gamma'] == gamma]
                yy = [per_image[i]['psnr'] for i in images]
                relations.append(dict(t=t, gamma=gamma, feature=f, image_clusters=6, spearman=corr(xx, yy)))
    pooled = []
    for gamma in GAMMAS:
        rr = [r for r in image_rows if r['gamma'] == gamma]
        for f in MODELS['DUAL']:
            x = np.array([r['features'][f] for r in rr]); y = np.array([r['delta']['psnr'] for r in rr])
            xc, yc = x.copy(), y.copy()
            for t in TIMES:
                mask = np.array([r['t'] == t for r in rr]); xc[mask] -= x[mask].mean(); yc[mask] -= y[mask].mean()
            pooled.append(dict(gamma=gamma, feature=f, raw_spearman=corr(x,y), within_t_centered_spearman=corr(xc,yc),
                points=30, independent_image_clusters=6))
    predictions = []
    for image in images:
        for t in TIMES:
            for gamma in GAMMAS:
                train = [r for r in rows if r['image_id'] != image and r['t'] == t and r['gamma'] == gamma]
                test = [r for r in rows if r['image_id'] == image and r['t'] == t and r['gamma'] == gamma]
                inputs = [{k:r[k] for k in ('case','image_id','seed','t','gamma','features')} for r in test]
                pred = {name:{m:predict(train, inputs, cols, m) for m in METRICS} for name,cols in MODELS.items()}
                for j,r in enumerate(test):
                    predictions.append(dict(**r, predictions={n:{m:float(pred[n][m][j]) for m in METRICS} for n in MODELS}))
    errors = {name:{m:float(np.mean([abs(r['predictions'][name][m]-r['delta'][m]) for r in predictions]))
                   for m in METRICS} for name in MODELS}
    decisions = []
    for case in sorted({r['case'] for r in predictions}):
        group = [r for r in predictions if r['case'] == case]
        null = dict(arm='C', delta={m:0. for m in METRICS}, predictions={n:{m:0. for m in METRICS} for n in MODELS})
        choices = {}
        for name in MODELS:
            eligible = [r for r in [null]+group if r['predictions'][name]['ssim'] >= -.002
                        and r['predictions'][name]['lpips'] <= .005]
            choices[name] = min(eligible, key=lambda r:(-r['predictions'][name]['psnr'],r['arm']))
        choices['F'] = next(r for r in group if r['arm'] == 'T319_G2')
        choices['ORACLE'] = max([null]+group, key=lambda r:r['delta']['psnr'])
        decisions.append(dict(case=case, image_id=group[0]['image_id'], choices={n:dict(arm=r['arm'],delta=r['delta']) for n,r in choices.items()}))
    policies = {}
    for name in (*MODELS,'F','ORACLE'):
        mean = {m:float(np.mean([r['choices'][name]['delta'][m] for r in decisions])) for m in METRICS}
        policies[name] = dict(mean_delta_vs_C=mean, delta_psnr_vs_F=mean['psnr']-float(np.mean([r['choices']['F']['delta']['psnr'] for r in decisions])),
            actions=[r['choices'][name]['arm'] for r in decisions], quality_guard=mean['ssim'] >= -.002 and mean['lpips'] <= .005)
    return dict(layers=layers, correlations=relations, pooled_correlations=pooled, loio_prediction_mae=errors,
                loio_policies=policies, decisions=decisions, predictions=predictions,
                oracle_headroom_psnr=policies['ORACLE']['delta_psnr_vs_F'])


def paid_readout(a28, rows):
    from utils.diag_cached_stage_action_audit import audit, choose
    points, _ = audit(Path(a28)); output = []
    for t in (641,319,54):
        decisions = []
        for p in points:
            if p['t'] != t: continue
            g = choose([a for a in p['actions'] if a['factor'] == 1.], ('mean','loss'))
            delta = {m:0. for m in METRICS} if g == 1 else next(r['delta'] for r in rows if r['image_id']==p['image_id'] and r['seed']==p['seed'] and r['t']==t and r['gamma']==g)
            decisions.append(dict(image_id=p['image_id'],seed=p['seed'],gamma=g,delta=delta))
        output.append(dict(t=t, choices=decisions, mean_final_delta_vs_C={m:float(np.mean([r['delta'][m] for r in decisions])) for m in METRICS}))
    return dict(layers=output, limits='A28 IEEE conditional-mean readout -> A22 native final outcomes; exploratory cross-precision transfer, not exact same-trajectory replay. Additional denoiser probes would cost NFE online.')


def font(size):
    from PIL import ImageFont
    for path in ('/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists(): return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def heatmap(result, path):
    from PIL import Image, ImageDraw
    width, height = 1500, 670; im = Image.new('RGB',(width,height),'white'); d = ImageDraw.Draw(im)
    d.text((35,20),'Fixed grid: final PSNR change after ONE guidance pulse',fill='#152536',font=font(28))
    d.text((35,62),'6 reused FFHQ images; 2-seed average/image; every arm = 300 NFE; no online state policy',fill='#45576a',font=font(18))
    left, top, cw, ch = 160,155,128,58
    images = sorted(result['layers'][0]['per_image_delta'])
    for j,layer in enumerate(result['layers']):
        d.text((left+j*cw+12,105),f't={layer["t"]}',fill='black',font=font(17))
        d.text((left+j*cw+12,129),f'x{layer["gamma"]:g}',fill='black',font=font(17))
        values = [layer['per_image_delta'][i]['psnr'] for i in images]+[layer['mean_delta']['psnr']]
        for k,value in enumerate(values):
            scale = min(abs(value)/.5,1.); target = (39,135,120) if value>=0 else (191,67,66)
            color = tuple(int(248*(1-scale)+c*scale) for c in target)
            x,y = left+j*cw,top+k*ch; d.rectangle((x,y,x+cw-3,y+ch-3),fill=color)
            d.text((x+18,y+18),f'{value:+.3f}',fill='white' if scale>.6 else '#152536',font=font(19))
    for k,i in enumerate(images+['MEAN']): d.text((40,top+k*ch+18),i,fill='#152536',font=font(19))
    d.text((35,590),'Red = worse than C; green = better. Same t319 x2 wins every case in this candidate library.',fill='#152536',font=font(19))
    d.text((35,624),'This establishes an action effect; it does not establish a state-based rule for choosing t or gamma.',fill='#45576a',font=font(18))
    im.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a22-dir',default='/data0/lzy/ZAPS-results/diag_equal_nfe_path_opportunity/A22_20261020')
    p.add_argument('--a28-dir',default='/data0/lzy/ZAPS-results/diag_state_action_response/A28_20261021')
    p.add_argument('--output',required=True); args=p.parse_args()
    output=Path(args.output).resolve(); sources=[Path(args.a22_dir).resolve(),Path(args.a28_dir).resolve()]
    if any(output.is_relative_to(s) or s.is_relative_to(output) for s in sources): raise ValueError('Source/output overlap')
    _,rows,identities=load_a22(sources[0]); result=analyze(rows)
    result['paid_readout_final_transfer']=paid_readout(sources[1],rows)
    result.update(experiment='A31_cached_guidance_final_value',method_nfe=0,diagnostic_nfe=0,
        independent_image_clusters=6,action_outcomes=120,limits=LIMITS,input_sha256=identities)
    output.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    target=output/'summary.json'
    if target.exists() and target.read_text()!=encoded: raise ValueError('Different existing A31; use new output')
    target.write_text(encoded); target.with_suffix('.sha256').write_text(sha(target)+'\n')
    with (output/'final_action_outcomes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','image_id','seed','t','gamma',*MODELS['DUAL'],*METRICS]);writer.writeheader()
        for r in rows: writer.writerow({k:r[k] for k in ('case','image_id','seed','t','gamma')}|r['features']|r['delta'])
    heatmap(result,output/'fixed_guidance_final_psnr.png')
    compact={k:result[k] for k in ('experiment','method_nfe','diagnostic_nfe','independent_image_clusters','action_outcomes','loio_prediction_mae','loio_policies','oracle_headroom_psnr','limits')}
    print('A31 Summary:',json.dumps(compact,ensure_ascii=False));print('Records:',output)


if __name__=='__main__': main()
