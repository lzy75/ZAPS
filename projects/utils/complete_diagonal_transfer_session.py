"""Detached single-device transfer queue, followed through completion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from utils import diag_complete_diagonal_transfer as driver
from utils.a54_session import exclusive


def running(pid):
    try:
        os.kill(pid,0)
        stat=Path(f'/proc/{pid}/stat')
        return not stat.exists() or stat.read_text().split(') ',1)[1][0]!='Z'
    except ProcessLookupError:
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=('dps','pigdm'))
    parser.add_argument('dataset',choices=('ffhq','imagenet','all'))
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--resume-reference-failure',action='store_true')
    args=parser.parse_args()
    datasets=('ffhq','imagenet') if args.dataset=='all' else (args.dataset,)
    for dataset in datasets:
        root=driver.directory(args.host,dataset)
        command=[sys.executable,'-u','-m','utils.diag_complete_diagonal_transfer',args.host,dataset]
        if args.resume_reference_failure:
            command.append('--resume-reference-failure')
        if args.status:
            subprocess.run(command+['--status'],cwd=driver.PROJECTS,check=True)
            continue
        root.mkdir(parents=True,exist_ok=True)
        position=0
        with exclusive(root/'client.lock'):
            session=root/'launcher.json'
            previous=driver.files.read(session) if session.exists() else {}
            pid=previous.get('pid')
            alive=bool(pid and running(pid))
            if not alive:
                state=driver.files.read(root/'session.json') if (root/'session.json').exists() else {}
                driver.files.require(state.get('status')!='running' and
                                     (state.get('status')!='failed' or args.resume_reference_failure),
                                     'Previous failed/unfinished launch retained; inspect --status')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu)
                position=(root/'worker.log').stat().st_size if (root/'worker.log').exists() else 0
                with (root/'worker.log').open('ab') as stream:
                    proc=subprocess.Popen(command+['--archive'],cwd=driver.PROJECTS,env=env,
                                          stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                driver.files.write(session,dict(pid=proc.pid,gpu=args.gpu))
                pid=proc.pid
            print('Following',args.host,dataset,'PID',pid,'log',root/'worker.log',flush=True)
        while True:
            log=root/'worker.log'
            if log.exists():
                with log.open() as stream:
                    stream.seek(position);text=stream.read();position=stream.tell()
                if text:print(text,end='',flush=True)
            active=running(pid)
            if not active:break
            time.sleep(5)
        if log.exists():
            with log.open() as stream:
                stream.seek(position);print(stream.read(),end='',flush=True)
        state=driver.files.read(root/'session.json') if (root/'session.json').exists() else {}
        if state.get('status')!='complete':
            raise RuntimeError(f'{args.host}/{dataset} stopped; see {log}')
        uploaded=driver.files.read(root/'upload.json') if (root/'upload.json').exists() else {}
        if uploaded.get('status')!='complete':
            raise RuntimeError(f'{args.host}/{dataset} completed but upload failed; see {log}')


if __name__=='__main__':
    main()
