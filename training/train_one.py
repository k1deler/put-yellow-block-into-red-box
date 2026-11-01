"""One bounded, independent ACT experiment; --launch is required to train.

Requires an already converted real yellow/red dataset and an existing CUDA Python
environment. Never loads fufu's task checkpoint or accesses any robot interface.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from audit import audit,sha

ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT/'upstream'


def available_gpus():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[]
    for line in raw.splitlines():
        index,memory,utilization=[int(x.strip()) for x in line.split(',')]
        if memory<1000 and utilization<5:rows.append(index)
    return rows


def run(command,cwd,env,log,timeout):
    with log.open('x',encoding='utf-8') as stream:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,
                              start_new_session=True)
        try:returncode=proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
            raise
    if returncode:
        raise RuntimeError(f'Own training subprocess failed ({returncode}); inspect {log}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    sys.path.insert(0,str(UPSTREAM))
    from yam_train.config import load_config
    cfg=load_config(ROOT/'configs/yellow_red_seed42.yaml')
    cfg['data']['root']=str(args.data.resolve())
    info=audit(args.data,cfg['data']['train_ratio'],cfg['data']['split_seed'])
    print(json.dumps(dict(mode='launch' if args.launch else 'preflight_only',
                          output=str(args.out.resolve()),task=info['task'],
                          train_episodes=len(info['train_episodes']),val_episodes=len(info['val_episodes']),
                          epochs=cfg['train']['num_epochs'],task_checkpoint_initialization='none',
                          initialization='new ACT parameters; standard ImageNet ResNet18 backbone',
                          normalization='train-only',maximum_full_train_seconds=21600),indent=2),flush=True)
    if not args.launch:return
    if sys.platform!='linux':raise RuntimeError('Actual training is allowed on the specified Linux server only')
    import fcntl
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No CUDA; no CPU fallback for this full experiment')
    free=available_gpus()
    gpu=args.gpu if args.gpu is not None else (free[0] if free else None)
    if gpu not in free:raise RuntimeError('No selected idle GPU; existing jobs are not stopped or preempted')
    # Cooperative lock under this user's own home, no shared hardware reset or job killing.
    lockdir=Path.home()/'.cache/k1deler_task1_training';lockdir.mkdir(parents=True,exist_ok=True)
    lock=(lockdir/f'gpu_{gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if gpu not in available_gpus():raise RuntimeError('GPU became busy before launch')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    import yaml
    (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (out/'data_audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    source_files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts}
    provenance=dict(status='starting',gpu=gpu,config=cfg,
                    upstream=json.loads((ROOT/'source.json').read_text()),code_sha256=source_files,
                    train_episodes=info['train_episodes'],val_episodes=info['val_episodes'],
                    model_initialization='new ACT parameters with ImageNet pretrained ResNet18; no task checkpoint loaded',
                    training_started_wall_ns=time.time_ns(),hardware_connected=False)
    def status(value,**kwargs):
        provenance.update(status=value,**kwargs)
        temp=out/'status.tmp';temp.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
        temp.replace(out/'status.json')
    status('starting')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(UPSTREAM),
             PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TORCH_HOME=str(out/'cache/torch'))
    smoke=out/'smoke'
    base=[sys.executable,'-u','-m','yam_train.train','--config',str(out/'resolved_config.yaml')]
    try:
        status('real_data_smoke')
        run(base+['--set','train.num_epochs=2','--set','train.extra_eval_samples=16',
                  '--set','train.num_workers=0','--out',str(smoke)],UPSTREAM,env,out/'smoke.log',900)
        checkpoint=torch.load(smoke/'ckpt/last.pt',map_location='cpu',weights_only=False)
        if (checkpoint['format'],checkpoint['task'],checkpoint['action_mode'],checkpoint['chunk_size'])!=(
                'yam_train/act_ref-v1','put_block_in_bowl','delta',100):
            raise ValueError('Real-data smoke exported incompatible checkpoint')
        status('training')
        experiment=out/'experiment'
        run(base+['--out',str(experiment)],UPSTREAM,env,out/'train.log',21600)
        status('evaluating')
        run([sys.executable,'-u',str(ROOT/'select_model.py'),'--run',str(experiment),
             '--data',str(args.data.resolve()),'--out',str(out/'evaluation')],UPSTREAM,env,
             out/'evaluation.log',3600)
        status('offline_training_and_evaluation_complete',finished_wall_ns=time.time_ns(),
               deployment_ready=False,robot_tested=False)
        print('NEW_TASK1_TRAINING_COMPLETE; actual task weights and evaluation saved:',out,flush=True)
    except BaseException as exc:
        status('incomplete_or_failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
