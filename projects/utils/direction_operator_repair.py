"""Interface-only A56/A57 operator repair; frozen scientific sources stay unchanged."""
from pathlib import Path


def seed_for_new_case(case):
    if case['task']=='super_resolution':
        return 0  # make_base's SR branch does not use this argument.
    if case['task']!='inpainting' or type(case.get('mask_seed')) is not int:
        raise ValueError('New A56 inpainting requires its frozen explicit mask_seed; no imputation')
    return case['mask_seed']


def mask_reference(plan,case,source):
    record=Path(plan['norm_sources'][case['key']])
    if source.digest(record)!=plan['identity_files'][str(record)]:
        raise ValueError('Historical operator source record changed')
    row=source.load_signed(record)
    if row['case']!=case or row['status']!='complete':
        raise ValueError('Historical mask case differs')
    sha=row['artifacts'].get('mask.pt');path=record.parent/'mask.pt'
    if not sha or source.digest(path)!=sha:
        raise ValueError('Need the exact archived mask; no new mask/seed fallback')
    return dict(path=str(path),sha256=sha,source_record=str(record))


def make_operator(package,case,device,plan=None):
    import torch
    from utils.diag_ffhq_four_tasks_unit import make_base,unit_operator
    shape=torch.empty(1,3,256,256,device=device)
    if package=='A57' and case['task']=='inpainting':
        from modules.degradations import get_operator
        from utils import diag_no_learning_replay as driver
        ref=mask_reference(plan,case,driver.source)
        mask=torch.load(ref['path'],map_location='cpu',weights_only=True)
        if tuple(mask.shape)!=(1,1,256,256) or not bool(((mask==0)|(mask==1)).all()):
            raise ValueError('Archived mask shape/value contract differs')
        raw=get_operator('inpainting',device=device,random_ratio=.7,mode='random',box_size=0,noise_sigma=0.,seed=-1)
        raw._mask_cache=mask.to(device=device,dtype=shape.dtype)
    else:
        raw=make_base(case['task'],device,shape,seed_for_new_case(case),None)
    return unit_operator(raw).to(device),raw._mask_cache if case['task']=='inpainting' else None


def recognized_zero_failure(row):
    return (row.get('status')=='failed' and row.get('error')==repr(KeyError('mask_seed'))
            and row.get('nfe')==0 and row.get('backward_attempts',0)==0
            and row.get('backward_completed',0)==0 and row.get('calls',[])==[])
