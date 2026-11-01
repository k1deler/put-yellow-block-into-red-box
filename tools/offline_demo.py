"""Exercise original GitHub main with TEST weights, generated images and SimArms only.

This demonstrates software wiring. It cannot evaluate yellow/red grasp success.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'project'
sys.path.insert(0,str(PROJECT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(4)
    from tests.fake_act_ref import make_act_ref_ckpt
    from yam_infer.data.nedf3_writer import Nedf3Writer,encode_jpeg_rgb
    checkpoint=output/'TEST_ONLY_RANDOM_MODEL.pt'
    task='synthetic_validation_task'
    make_act_ref_ckpt(checkpoint,task=task)
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    data['reproduction_test_only']=True
    torch.save(data,checkpoint)
    config=output/'test_models.yaml'
    config.write_text(yaml.safe_dump(dict(model=dict(ckpt=str(checkpoint),format='act_ref',
                      task=task,action_mode='delta',compensate_action_shift=False))))
    episode=output/'synthetic_recording'
    t0=time.time_ns()
    writer=Nedf3Writer(episode,dict(task_name=task,task_description='Synthetic generated frames, NOT a trained task'),t0)
    image=np.full((48,64,3),128,np.uint8)
    jpeg=encode_jpeg_rgb(image)
    for i in range(45):
        stamp=t0+round(i*1e9/30)
        for camera in ('right_wrist','left_wrist'):
            writer.write_camera(camera,stamp,jpeg,image.shape,i+1)
        writer.write_state(stamp,np.zeros(14))
    writer.close()
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(PROJECT),os.environ.get('PYTHONPATH','')]),
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    command=[sys.executable,'-u','-m','yam_infer.run','--models-config',str(config),
             '--run-config',str(PROJECT/'configs/run.yaml'),'--device','cpu',
             '--dry-run',str(episode),'--max-seconds','3','--record-dir',str(output/'runs')]
    proc=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,bufsize=1)
    observed=dict(prompts=0,running=False,timeout=False)
    errors=[]
    def read():
        try:
            with (output/'lifecycle.log').open('w',encoding='utf-8') as log:
                for line in proc.stdout:
                    log.write(line);log.flush()
                    if 'press Enter to START' in line:
                        observed['prompts']+=1
                        proc.stdin.write('\n' if observed['prompts']==1 else 'q\n');proc.stdin.flush()
                    if 'RUNNING  (Enter' in line:observed['running']=True
                    if 'episode ended: max_seconds' in line:observed['timeout']=True
        except Exception as e:errors.append(str(e))
    reader=threading.Thread(target=read,daemon=True)
    reader.start()
    try:
        status=proc.wait(timeout=100)
    except subprocess.TimeoutExpired:
        proc.kill();proc.wait(timeout=5)  # This subprocess is strictly a simulation.
        raise RuntimeError('Offline simulation timed out; inspect lifecycle.log')
    finally:
        reader.join(timeout=5)
        proc.stdin.close();proc.stdout.close()
    metas=list((output/'runs').glob('*/meta.json'))
    if status or errors or not observed['running'] or not observed['timeout'] or len(metas)!=1:
        raise RuntimeError('Offline simulation failed: '+str(observed)+'; inspect '+str(output/'lifecycle.log'))
    meta=json.loads(metas[0].read_text())
    if meta.get('policy_steps',0)<1:raise RuntimeError('Simulation ran without a completed model prediction')
    report=dict(status='passed',scope='GitHub CPU random-model software lifecycle only',
                observation_source='generated constant JPEGs',model_source='tests.fake_act_ref, randomly initialized',
                fake_robot='SimArms (in-process), no CAN or RPC worker started',
                lifecycle=observed,policy_steps=meta['policy_steps'],hardware_opened=False,
                physical_commands_sent=0,target_task_success_verified=False,
                checkpoint_forbidden_for_live_registration=True)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
