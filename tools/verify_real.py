"""Offline GPU/model gate. Never constructs a real arm or camera."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'project'
sys.path[:0]=[str(PROJECT),str(PROJECT/'third_party/i2rt')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def runtime_hashes():
    names=[]
    for folder in ('yam_infer','third_party','configs'):
        names.extend(p for p in (PROJECT/folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    names.append(PROJECT/'models/selected_real.pt')
    result={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(names)}
    result['__target_task__']=sha(ROOT/'target_task.json')
    return result


def environment():
    import torch
    return dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                torchvision=importlib.metadata.version('torchvision'),
                numpy=importlib.metadata.version('numpy'),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    marker=ROOT/'evidence/real_model_offline_passed.json'
    if args.check:
        r=json.loads(marker.read_text())
        if r.get('status')!='passed' or r['runtime_hashes']!=runtime_hashes() or r['environment']!=environment():
            raise ValueError('Model/code/config/environment changed; rerun validate_real.sh')
        print('TASK_MODEL_GATE_OK; no hardware opened')
        return
    marker.unlink(missing_ok=True)
    import numpy as np
    import torch
    from yam_infer.config import load_model_config,check_model_matches,load_run_config
    from yam_infer.policy.policy import Policy
    from yam_infer.control.scheduler import SafetyFilter
    identity=json.loads((ROOT/'evidence/registered_model.json').read_text())
    if sha(PROJECT/'models/selected_real.pt')!=identity['sha256']:
        raise ValueError('Registered real model hash differs')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    cfg=load_model_config()
    run=load_run_config()
    task_profile=json.loads((ROOT/'target_task.json').read_text(encoding='utf-8'))
    if (cfg.task,cfg.format,cfg.action_mode)!=('put_block_in_bowl','act_ref','delta'):
        raise ValueError('Wrong task/model; only yellow-block/red-container reproduction is allowed')
    if task_profile['expected_sha256'] is not None and sha(cfg.ckpt)!=task_profile['expected_sha256']:
        raise ValueError('Wrong real checkpoint SHA for yellow/red task')
    torch.set_num_threads(4)
    pol=Policy(cfg.ckpt,'cuda')
    check_model_matches(cfg,pol.format,pol.tasks,pol.action_mode)
    if not 1<=run['execution']['horizon']<=pol.chunk_size:
        raise ValueError('Execution horizon must fit the real checkpoint')
    state=np.asarray(pol.start_pose[cfg.task],dtype=np.float64)
    if state.shape!=(14,) or not np.isfinite(state).all():raise ValueError('Invalid model start_pose')
    if np.any(state[[6,13]]<0) or np.any(state[[6,13]]>1):raise ValueError('Invalid starting gripper targets')
    frames={camera:np.zeros((480,640,3),np.uint8) for camera in pol.cameras}
    seconds=[]
    for _ in range(3):
        started=time.monotonic()
        out,_=pol.predict(frames,state,cfg.task)
        seconds.append(time.monotonic()-started)
        if out.shape!=(pol.chunk_size,14) or not np.isfinite(out).all():raise ValueError('Nonfinite or wrong-shape model output')
    safety=SafetyFilter(pol.a_min,pol.a_max)
    safety.reset(state)
    bounded=safety(out[0],.01)
    if not np.isfinite(bounded).all() or np.any(bounded[[6,13]]<0) or np.any(bounded[[6,13]]>1):
        raise ValueError('Invalid bounded gripper target')
    if 'pyrealsense2' in sys.modules or 'i2rt.robots.get_robot' in sys.modules:
        raise RuntimeError('A device interface was unexpectedly imported by offline model inference')
    report=dict(status='passed',scope='synthetic-input CUDA smoke test, NOT task success or recorded prediction replay',
                environment=environment(),runtime_hashes=runtime_hashes(),model=pol.info(),run_config=run,
                synthetic_input_inference_seconds=seconds,hardware_opened=False,robot_commands_sent=0,
                success_claim=False)
    marker.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('REAL_MODEL_OFFLINE_SMOKE_PASSED; zero hardware commands. No task-success claim.')


if __name__=='__main__':main()
