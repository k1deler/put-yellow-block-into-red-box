"""Continue the authorized training pipeline with a bounded baseline comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--baseline-split',type=Path,required=True)
    p.add_argument('--baseline-all',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('Server continuation only')
    import fcntl
    root=args.experiment.resolve()
    deadline=time.monotonic()+8*3600
    state=root/'comparison_state.json'
    def save(status,**more):
        state.write_text(json.dumps(dict(status=status,**more),indent=2)+'\n')
    save('waiting_for_own_training')
    while time.monotonic()<deadline:
        path=root/'run/status.json'
        if path.is_file():
            status=json.loads(path.read_text())['status']
            if status=='incomplete_or_failed':
                save('not_run_training_failed');return
            if status=='offline_training_and_evaluation_complete':break
        time.sleep(30)
    else:
        save('not_run_training_wait_timeout');return
    lockdir=Path.home()/'.cache/k1deler_task1_training';lockdir.mkdir(parents=True,exist_ok=True)
    lock=(lockdir/f'gpu_{args.gpu}.lock').open('a')
    for _ in range(30):
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(2)
    else:
        save('not_run_gpu_lock_busy');return
    try:
        # The training job may be done while another user has acquired the GPU.
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True,timeout=10)
        memory,utilization=next((int(m.strip()),int(u.strip())) for i,m,u in
                               (line.split(',') for line in raw.splitlines()) if int(i.strip())==args.gpu)
        if memory>=1000 or utilization>=5:
            save('not_run_gpu_became_busy');return
        save('comparing')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),PYTHONUNBUFFERED='1',
                 PYTHONPATH=str(ROOT/'upstream'),TORCH_HOME=str(root/'run/cache/torch'),
                 PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        command=[sys.executable,'-u',str(ROOT/'compare_baseline.py'),
                 '--new',str(root/'run/evaluation/yellow_red_seed42_selected.pt'),
                 '--baseline-split',str(args.baseline_split),'--baseline-all',str(args.baseline_all),
                 '--data',str(args.data),'--out',str(root/'model_comparison')]
        with (root/'comparison.log').open('x') as log:
            subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                           timeout=3600,check=True)
        save('offline_comparison_complete',report=str(root/'model_comparison/comparison.json'),robot_deployment_changed=False)
    except BaseException as exc:
        save('comparison_incomplete_or_failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
