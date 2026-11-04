"""Download pinned author PDFs and small upstream source subsets; no model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

PROJECTS = Path(__file__).resolve().parents[2]
PAPERS = [
    ('reference', 'SAIP', '2509.24580v1'),
    ('reference', 'JAPS', '2511.18471v3'),
    ('reference', 'SIMCA', '2610.03503v1'),
    ('reference', 'DiffStateGrad', '2410.03463v5'),
    ('reference', 'DTM', '2502.03686v2'),
    ('reference', 'Calibrated_Bayesian_Guidance', '2602.22428v2'),
    ('baseline', 'MPGD', '2311.16424v1'),
    ('baseline', 'DMPS', '2211.12343v4'),
    ('baseline', 'PiGDM', 'https://jankautz.com/publications/PiGDM_ICLR23.pdf'),
]
SOURCES = [
    ('mengxiangming/dmps', '91afe2540ac8dd8a7fa133f654560f736c80fc53', 'dmps_91afe254', [
        'README.md', 'main.py', 'guided_diffusion/condition_methods.py',
        'guided_diffusion/gaussian_diffusion.py', 'guided_diffusion/posterior_mean_variance.py',
        'util/functions.py', 'util/img_utils.py', 'configs/sr4_config.yaml',
        'configs/deblur_gauss_config.yaml', 'configs/model_config.yaml', 'configs/diffusion_config.yaml']),
    ('KellyYutongHe/mpgd_pytorch', '9f94b386ea4184960278d28a8bc2756898db70e2', 'mpgd_9f94b386', [
        'README.md', 'LICENSE', 'linear_inv/ffhq_sample_condition.py',
        'linear_inv/guided_diffusion/condition_methods.py',
        'linear_inv/guided_diffusion/guided_gaussian_diffusion.py',
        'linear_inv/guided_diffusion/posterior_mean_variance.py',
        'linear_inv/guided_diffusion/measurements.py', 'linear_inv/util/img_utils.py',
        'linear_inv/util/resizer.py', 'linear_inv/configs/model_config.yaml',
        'linear_inv/configs/mgpd_diffusion_config.yaml',
        'linear_inv/configs/super_resolution_config.yaml', 'linear_inv/configs/gaussian_deblur_config.yaml',
        'linear_inv/scripts/wo_proj_ffhq_super_resolution.sh']),
    ('NVlabs/RED-diff', '66482f23e242bb31166c3662002d0a6f9f065030', 'pgdm_66482f23', [
        'README.md', 'LICENSE', 'algos/pgdm.py', 'algos/ddim.py',
        'models/classifier_guidance_model.py', 'models/diffusion.py', 'utils/degredations.py',
        '_configs/algo/pgdm.yaml', '_configs/exp/default.yaml', '_configs/model/ffhq256_uncond.yaml',
        'sample_test.sh', 'main.py', 'utils/functions.py']),
]


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ZAPS-research-source-archive/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--papers-only', action='store_true')
    args = parser.parse_args()
    if not args.sources_only:
        rows = []
        for category, name, source in PAPERS:
            url = source if source.startswith('https:') else 'https://arxiv.org/pdf/' + source
            folder = Path(__file__).resolve().parent / 'pdf' / category
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (name + '.pdf')
            if not path.exists():
                data = get(url)
                if not data.startswith(b'%PDF-'):
                    raise ValueError('Not a PDF: ' + url)
                path.write_bytes(data)
            data = path.read_bytes()
            rows.append(dict(category=category, name=name, source=url,
                             path=str(path.relative_to(PROJECTS)), bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest()))
            print('PDF_SAVED', name, len(data), flush=True)
        (Path(__file__).parent/'papers.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    if not args.papers_only:
        for repository, revision, local, paths in SOURCES:
            folder = PROJECTS/'vendor'/local
            rows = []
            for path in paths:
                target = folder/path
                target.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://raw.githubusercontent.com/{repository}/{revision}/{path}'
                data = target.read_bytes() if target.exists() else get(url)
                target.write_bytes(data)
                blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                rows.append(dict(path=path, git_blob_sha=blob, sha256=hashlib.sha256(data).hexdigest()))
            (folder/'UPSTREAM.json').write_text(json.dumps(dict(repository=repository, commit=revision,
                policy='Unmodified UTF-8 source subset at pinned author revision; no model weights',
                files=rows), indent=2)+'\n')
            print('UPSTREAM_SAVED', repository, revision, len(rows), flush=True)


if __name__ == '__main__':
    main()
