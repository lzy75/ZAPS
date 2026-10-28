"""Zero-NN historical baseline comparison. Never pools adaptive or 330-NFE arms."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess

REPO=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
REF='origin/results/server-records-20261013'
CSV='projects/server_records/20261013/results/diag_ffhq_timestep_multi/20260917_020040/metrics.csv'
LOG='projects/server_records/20261013/logs/ffhq_timestep_multi_10.log'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    evidence=[]; august={}
    for p in sorted((REPO/'projects/experiments').glob('20260818_*_ffhq_*/run.json')):
        r=json.loads(p.read_text());c=r['config']
        if c.get('adaptive') is not False or c.get('final_mode')!='last_opt':continue
        assert r['nfe']['total']==300 and c['lr']==.001
        task=r['exp_id'].split('_ffhq_')[0].split('_',2)[-1]
        august.setdefault(task,[]).append(r)
        evidence.append(dict(path=str(p.relative_to(REPO)),sha256=sha(p.read_bytes())))
    august_stats={}
    for task,rows in august.items():
        ids=[r['exp_id'].rsplit('_',1)[1] for r in rows]
        assert len(ids)==len(set(ids)), 'Never select a convenient duplicate outcome'
        august_stats[task]=dict(n=len(rows),image_ids=ids,
            missing_in_00000_to_00019=[f'{i:05d}' for i in range(20) if f'{i:05d}' not in ids],
            mean={k:statistics.mean(r['metrics'][k] for r in rows) for k in ('psnr','ssim','lpips')},
            optimizer_seconds_historical=statistics.mean(r['times']['opt_s'] for r in rows),
            configs=rows[0]['config'],all_dirty=all(r['git']['dirty'] for r in rows),
            commits=sorted({r['git']['commit'] for r in rows}))
    content=subprocess.check_output(['git','show',REF+':'+CSV],cwd=REPO)
    log=subprocess.check_output(['git','show',REF+':'+LOG],cwd=REPO)
    (HERE/'september17_metrics.csv').write_bytes(content)
    (HERE/'september17_log.txt').write_bytes(log)
    rows=list(csv.DictReader(io.StringIO(content.decode())))
    assert len(rows)==30 and len({r['image'] for r in rows})==10
    september={}
    for schedule in sorted({r['schedule'] for r in rows}):
        group=[r for r in rows if r['schedule']==schedule]
        assert len(group)==10
        assert [int(r['seed']) for r in group]==list(range(1000,1010))
        september[schedule]=dict(n=10,mean={k:statistics.mean(float(r[k]) for r in group)
            for k in ('psnr','ssim','lpips','seconds')},image_ids=[Path(r['image']).stem for r in group])
    root=REPO/'projects/reports/20261028_a64_received'
    manifest=json.loads((root/'manifest.json').read_text());verified=[]
    for p in (root/'A64').rglob('*.json'):
        relative=str(p.relative_to(root))
        found=[r for r in manifest['files'] if r['archive_path']==relative]
        assert len(found)==1 and sha(p.read_bytes())==found[0]['sha256'],relative
        verified.append(relative)
    result=dict(new_UNet_forwards=0,new_input_backwards=0,
        august18=august_stats,september17=september,
        september_source=dict(ref=REF,commit=subprocess.check_output(['git','rev-parse',REF],cwd=REPO,text=True).strip(),
            csv=CSV,csv_sha256=sha(content),log=LOG,log_sha256=sha(log)),
        august_sources=evidence,A64_receipt=dict(branch='results/paper-clock-alignment-20261028',
            commit='160ceaf8e0d9ad7bf5443ba72d21b96566dc1e21',verified_files=verified,
            missing_binary_tensors='Not fetched; hashes in original manifest only'),
        limits='Different cohorts and protocols: no paired quality difference or baseline-regression causality. '
        'August records mark dirty=True; September CSV/log lacks complete checkpoint/observation/RNG fingerprints. '
        'Old timings are optimization segments, not the new complete reconstruction boundary.')
    (HERE/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('HISTORICAL BASELINE AUDIT PASS:79 August fixed300 records,30 September schedule records, '
          f'{len(verified)} received A64 metadata files; NEW NN=0.')


if __name__=='__main__':main()
