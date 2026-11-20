"""Detach one A122 FFHQ worker, follow to summary, retain failures."""
import argparse
import os
import subprocess
import sys
import time
from utils import diag_matched_response_transfer as driver
from utils.complete_diagonal_transfer_session import running
from utils.a54_session import exclusive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host',choices=('dps','pigdm'))
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args(); root=driver.directory(args.host)
    command=[sys.executable,'-u','-m','utils.diag_matched_response_transfer',args.host]
    if args.status:
        subprocess.run(command+['--status'],cwd=driver.PROJECTS,check=True); return
    root.mkdir(parents=True,exist_ok=True)
    with exclusive(root/'client.lock'):
        launcher=root/'launcher.json'
        old=driver.files.read(launcher) if launcher.exists() else {}
        pid=old.get('pid'); position=0
        if not pid or not running(pid):
            state=driver.files.read(root/'session.json') if (root/'session.json').exists() else {}
            driver.files.require(state.get('status') not in ('running','failed'),'Previous launch retained; inspect --status')
            log=root/'worker.log'; position=log.stat().st_size if log.exists() else 0
            with log.open('ab') as stream:
                process=subprocess.Popen(command+['--archive'],cwd=driver.PROJECTS,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            pid=process.pid
            driver.files.write(launcher,dict(pid=pid,gpu=args.gpu))
        print('A122 FOLLOW:',args.host,'PID',pid,'log',root/'worker.log',flush=True)
    while True:
        log=root/'worker.log'
        if log.exists():
            with log.open() as stream:
                stream.seek(position); new=stream.read(); position=stream.tell()
            if new: print(new,end='',flush=True)
        if not running(pid): break
        time.sleep(5)
    if log.exists():
        with log.open() as stream:
            stream.seek(position); print(stream.read(),end='',flush=True)
    state=driver.files.read(root/'session.json') if (root/'session.json').exists() else {}
    if state.get('status')!='complete': raise RuntimeError(f'A122 {args.host} stopped; see {log}')
    uploaded=driver.files.read(root/'upload.json') if (root/'upload.json').exists() else {}
    if uploaded.get('status')!='complete': raise RuntimeError('Reconstruction complete; result upload incomplete, see worker.log')


if __name__=='__main__':
    main()
