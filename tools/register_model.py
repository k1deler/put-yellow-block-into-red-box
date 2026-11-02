"""Register a trusted trained checkpoint for the selected task contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

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
    p.add_argument('--replace',action='store_true',help='Back up an already registered different model before switching')
    args=p.parse_args()
    import torch
    import yaml
    src=args.checkpoint.expanduser().resolve()
    if not src.is_file():raise FileNotFoundError(src)
    target_task=json.loads((ROOT/'target_task.json').read_text(encoding='utf-8'))
    source_sha=sha(src)
    if target_task['expected_sha256'] is not None and source_sha!=target_task['expected_sha256']:
        raise ValueError('Checkpoint SHA differs from the previously verified yellow/red task model; no config changed')
    # User-selected, trusted model only: the source checkpoint format is Python pickle.
    payload=torch.load(src,map_location='cpu',weights_only=False)
    config=identify(payload,args.task)
    config['ckpt']='models/selected_real.pt'
    dest=ROOT/'project/models/selected_real.pt'
    original=ROOT/'project/configs/models.yaml'
    replacing=False
    if dest.exists():
        if sha(dest)==source_sha:
            identity=ROOT/'evidence/registered_model.json'
            if identity.exists() and json.loads(identity.read_text())['sha256']==source_sha:
                print('REQUESTED_MODEL_ALREADY_REGISTERED:',source_sha,flush=True)
                return
            raise ValueError('Matching model exists without a matching registration; inspect before changing state')
        if not args.replace:
            raise FileExistsError('Model already registered; use --replace to back it up first: '+str(dest))
        existing=yaml.safe_load(original.read_text())['model']
        if any(existing.get(key)!=config[key] for key in ('ckpt','format','task','action_mode','compensate_action_shift')):
            raise ValueError('Existing model interface differs; all files preserved for manual review')
        replacing=True
        backup=ROOT/'project/models/backups'/str(time.time_ns())
        backup.mkdir(parents=True,exist_ok=False)
        shutil.copy2(dest,backup/'selected_real.pt')
        for name in ('registered_model.json','real_model_offline_passed.json'):
            existing=ROOT/'evidence'/name
            if existing.is_file():shutil.copy2(existing,backup/name)
        shutil.copy2(ROOT/'project/configs/models.yaml',backup/'models.yaml')
        print('PREVIOUS_MODEL_BACKED_UP:',backup,flush=True)
    dest.parent.mkdir(exist_ok=True)
    temporary=dest.with_name('selected_real.pending.pt')
    if temporary.exists():raise FileExistsError('Pending model file exists; inspect before proceeding')
    try:
        shutil.copy2(src,temporary)
        if sha(temporary)!=source_sha:raise RuntimeError('Checkpoint copy mismatch')
        # Prevent an old validation marker from authorizing the newly selected model.
        (ROOT/'evidence/real_model_offline_passed.json').unlink(missing_ok=True)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    backup=ROOT/'evidence/initial_models.yaml'
    if not backup.exists():shutil.copy2(original,backup)
    if not replacing:
        original.write_text(yaml.safe_dump(dict(model=config),sort_keys=False),encoding='utf-8')
    run_path=ROOT/'project/configs/run.yaml'
    run_backup=ROOT/'evidence/initial_run.yaml'
    if not run_backup.exists():shutil.copy2(run_path,run_backup)
    run_config={key:target_task[key] for key in ('execution','control','episode')}
    if not replacing:
        run_path.write_text(yaml.safe_dump(run_config,sort_keys=False),encoding='utf-8')
    (ROOT/'evidence/registered_model.json').write_text(json.dumps(dict(
        source=str(src),sha256=source_sha,configuration=config,model_version=target_task.get('model_version'),
        selected_by_user=True,trained_performance_verified=False),indent=2)+'\n')
    print('REAL_MODEL_REGISTERED:',config,flush=True)


if __name__=='__main__':main()
