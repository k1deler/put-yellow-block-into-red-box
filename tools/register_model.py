"""Register a trusted trained checkpoint for the selected task contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def identify(payload, task):
    target=json.loads((ROOT/'target_task.json').read_text(encoding='utf-8'))
    if task != target['task']:
        raise ValueError('This reproduction only accepts the yellow-block/red-container task')
    if payload.get('reproduction_test_only'):
        raise ValueError('Synthetic test checkpoint is forbidden for the real-hardware configuration')
    fmt=payload.get('format','act_v2')
    if fmt != target['format']:
        raise ValueError('Unsupported checkpoint format: '+str(fmt)+'. Do not use the earlier LeRobot checkpoint here.')
    tasks=list(payload['tasks']) if fmt=='act_v2' else [str(payload['task'])]
    if task not in tasks:
        raise ValueError(f'Task {task!r} does not match the checkpoint tasks {tasks!r}')
    mode=payload.get('action_mode','abs')
    if mode not in ('abs','delta'):
        raise ValueError('Unknown action semantics')
    if mode != target['action_mode']:
        raise ValueError('This reproduction requires the verified delta checkpoint')
    return dict(format='act_v2' if fmt=='act_v2' else 'act_ref',task=task,action_mode=mode,
                compensate_action_shift=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--task',required=True)
    args=p.parse_args()
    import torch
    import yaml
    src=args.checkpoint.expanduser().resolve()
    if not src.is_file():raise FileNotFoundError(src)
    target_task=json.loads((ROOT/'target_task.json').read_text(encoding='utf-8'))
    if target_task['expected_sha256'] is not None and sha(src)!=target_task['expected_sha256']:
        raise ValueError('Checkpoint SHA differs from the previously verified yellow/red task model; no config changed')
    # User-selected, trusted model only: the source checkpoint format is Python pickle.
    payload=torch.load(src,map_location='cpu',weights_only=False)
    config=identify(payload,args.task)
    dest=ROOT/'project/models/selected_real.pt'
    if dest.exists():
        raise FileExistsError('Model already registered; refusing overwrite: '+str(dest))
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(src,dest)
    if sha(src)!=sha(dest):raise RuntimeError('Checkpoint copy mismatch')
    config['ckpt']='models/selected_real.pt'
    original=ROOT/'project/configs/models.yaml'
    backup=ROOT/'evidence/initial_models.yaml'
    if not backup.exists():shutil.copy2(original,backup)
    original.write_text(yaml.safe_dump(dict(model=config),sort_keys=False),encoding='utf-8')
    run_path=ROOT/'project/configs/run.yaml'
    run_backup=ROOT/'evidence/initial_run.yaml'
    if not run_backup.exists():shutil.copy2(run_path,run_backup)
    run_config={key:target_task[key] for key in ('execution','control','episode')}
    run_path.write_text(yaml.safe_dump(run_config,sort_keys=False),encoding='utf-8')
    (ROOT/'evidence/registered_model.json').write_text(json.dumps(dict(
        source=str(src),sha256=sha(dest),configuration=config,
        selected_by_user=True,trained_performance_verified=False),indent=2)+'\n')
    print('REAL_MODEL_REGISTERED:',config,flush=True)


if __name__=='__main__':main()
