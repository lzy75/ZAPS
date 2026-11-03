"""Load the official pure-Python wheel without pip, downloads, or environment installation."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import zipimport

PROJECTS=Path(__file__).resolve().parents[1]
WHEEL=PROJECTS/'vendor/python_wheels/torchdiffeq-0.2.5-py3-none-any.whl'
SHA256='aa1db4bed13bd04952f28a53cdf4336d1ab60417c1d9698d7a239fec1cf2bcf8'


def environment():
    if hashlib.sha256(WHEEL.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Official offline torchdiffeq wheel checksum mismatch')
    distributions=list(importlib.metadata.distributions(path=[str(WHEEL)]))
    if len(distributions)!=1 or distributions[0].metadata['Name']!='torchdiffeq' or distributions[0].version!='0.2.5':
        raise ValueError('Offline wheel metadata differs from the frozen dependency')
    if zipimport.zipimporter(str(WHEEL)).find_spec('torchdiffeq') is None:
        raise ValueError('Offline pure-Python wheel cannot be imported')
    env=dict(os.environ)
    env['PYTHONPATH']=str(WHEEL)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['MKL_THREADING_LAYER']='GNU';env.pop('MKL_SERVICE_FORCE_INTEL',None)
    return env


def main():
    env=environment()
    print('A81 OFFLINE DEPENDENCY PASS: torchdiffeq0.2.5; official SHA256='+SHA256+'; no pip/network/install.',flush=True)
    if sys.argv[1:]==['--check-only']:
        code="import importlib.metadata as m,importlib.util as u; assert m.version('torchdiffeq')=='0.2.5'; assert '.whl/torchdiffeq/' in u.find_spec('torchdiffeq').origin; print('Child interpreter resolves the verified offline wheel.')"
        subprocess.run([sys.executable,'-c',code],env=env,check=True)
        return
    command=[sys.executable,'-u',str(PROJECTS/'utils/a81_session.py'),'--experiment','A81',*sys.argv[1:]]
    os.chdir(PROJECTS)
    os.execve(sys.executable,command,env)


if __name__=='__main__':main()
