"""Choose available GPUs inside CUDA_VISIBLE_DEVICES; one usable GPU means serial shards."""
import os
import subprocess
import sys

MIN_FREE_MIB = 6144


def choose(rows, visible=None, first=None, second=None):
    aliases = {}
    for row in rows:
        for key in (row['index'], row['uuid']):aliases[key] = row
    allowed = [r['index'] for r in rows] if visible is None else [v.strip() for v in visible.split(',')]
    if not allowed or any(v in ('', '-1') for v in allowed):
        raise ValueError('CUDA_VISIBLE_DEVICES permits no GPU')
    tokens, devices = {}, {}
    for token in allowed:
        if token not in aliases:raise ValueError('Unrecognized allocated GPU token: '+token)
        row = aliases[token];tokens.setdefault(row['uuid'], token);devices[row['uuid']] = row
    usable = sorted((u for u,r in devices.items() if r['free'] >= MIN_FREE_MIB),
                    key=lambda u:-devices[u]['free'])
    if not usable:
        detail = ', '.join(f'{tokens[u]}={r["free"]} MiB free' for u,r in devices.items())
        raise ValueError(f'No allocated GPU has {MIN_FREE_MIB} MiB free ({detail}). No experiment started.')
    def explicit(token):
        if token is None:return None
        if token not in aliases or aliases[token]['uuid'] not in devices:
            raise ValueError('Explicit GPU is outside CUDA_VISIBLE_DEVICES: '+token)
        uuid = aliases[token]['uuid']
        if uuid not in usable:raise ValueError('Explicit GPU has less than 6 GiB free: '+token)
        return uuid
    one, two = explicit(first), explicit(second)
    if one is None:one = next((u for u in usable if u != two), usable[0])
    if two is None:two = next((u for u in usable if u != one), one)
    # Identical physical GPUs always use identical tokens, triggering the original serial branch.
    return [tokens[one], tokens[two]]


def main():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in raw.splitlines():
        index, uuid, free = [v.strip() for v in line.split(',')]
        rows.append(dict(index=index, uuid=uuid, free=int(free)))
    try:
        selected = choose(rows, os.environ.get('CUDA_VISIBLE_DEVICES'),
                          os.environ.get('ZAPS_A36_GPU0'), os.environ.get('ZAPS_A36_GPU1'))
    except ValueError as error:raise SystemExit(str(error))
    print('\n'.join(selected))
    mode = 'serial shards on one GPU' if selected[0] == selected[1] else 'parallel shards on two GPUs'
    print(f'A36 resource selection: {selected}; {mode}. Frozen A36 plan unchanged.', file=sys.stderr, flush=True)


if __name__ == '__main__':main()
