"""A123 protocol: extend the frozen A122 diagonal rule to every native step."""
from utils import diag_matched_response_transfer as sparse


def native_steps(host):
    if host not in ('dps','pigdm'): raise ValueError('Unknown method')
    return 1000 if host=='dps' else 100


def positions(host):
    return tuple(range(1,native_steps(host)+1))


def budget(host):
    n=native_steps(host)
    # The final s=-1 clean predictor is identity: eight queries, zero NN calls.
    return dict(forwards=n+8*(n-1),input_vjp_attempts=2*n,input_jvp_attempts=2*n,
                response_vjp_attempts=n,response_vjp_forwards=n,response_jvp_forwards=2*n,
                response_value_forwards=5*n,network_response_vjp=n-1,
                network_response_jvp=2*(n-1),terminal_operator_vjp=1,terminal_identity_jvp=2)


def directory(host):
    return sparse.previous.BASE/'diag_full_diagonal_transfer'/f'A123_{host}_ffhq_20261121'


def parent(host):
    return sparse.directory(host)


def resolve_record(path):
    from pathlib import Path
    files=sparse.files; path=Path(path); seen=set()
    while True:
        files.require(str(path) not in seen,'Cyclic reused result')
        seen.add(str(path)); row=files.read(path)
        files.require(row['status']=='complete','Only completed results can be reused')
        if not row.get('reused'): return path,row
        source=Path(row['source_record'])
        files.require(files.digest(source)==row['source_sha256'],'Reused source record changed')
        path=source
