"""Same-input, same-scheduler offline model comparison; never connects hardware.

Reports full-data-baseline validation overlap explicitly. A numerical offline
winner is retained separately; both original checkpoints are preserved.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
from audit import sha  # noqa: E402
from yam_train.data.dataset import FrameStore,EpisodicDataset  # noqa: E402
from yam_train.policy import ACTPolicy  # noqa: E402
from yam_train.train import extra_eval,fixed_samples  # noqa: E402


def simulate(chunks,fps=30,horizon=60,blend_s=.1,joint_speed=1.5,gripper_speed=3.0,
             lower=None,upper=None,initial=None):
    """Recorded-input chunk playback + 100Hz first-order simulated tracking.

    Recorded model observations do not change in response to simulated commands.
    This is a tracking/playback comparison, not closed-loop task success.
    """
    previous=np.asarray(initial,np.float64).copy()
    measured=previous.copy()
    speed=np.full(14,joint_speed);speed[[6,13]]=gripper_speed
    low=np.asarray(lower,np.float64).copy();high=np.asarray(upper,np.float64).copy()
    joints=np.r_[0:6,7:13];low[joints]-=.15;high[joints]+=.15
    low[[6,13]]=0;high[[6,13]]=1
    commands=[];feedback=[];clips=limited=0
    def interpolate(plan,age):
        position=np.clip(age*fps,0,len(plan)-1)
        left=int(position);right=min(left+1,len(plan)-1)
        return plan[left]+(plan[right]-plan[left])*(position-left)
    for tick in range(round(len(chunks)/fps*100)):
        now=tick/100
        frame=min(int(now*fps),len(chunks)-1)
        anchor=(frame//horizon)*horizon
        desired=interpolate(chunks[anchor],now-anchor/fps)
        if anchor>=horizon and now-anchor/fps<blend_s:
            old=interpolate(chunks[anchor-horizon],now-(anchor-horizon)/fps)
            desired=old+(desired-old)*((now-anchor/fps)/blend_s)
        bounded=np.clip(desired,low,high)
        clips+=int(not np.allclose(desired,bounded))
        command=np.clip(bounded,previous-speed*.01,previous+speed*.01)
        limited+=int(not np.allclose(command,bounded))
        measured += (command-measured)*(1-np.exp(-.01/.13))
        commands.append(command.copy());feedback.append(measured.copy());previous=command
    return np.asarray(commands),np.asarray(feedback),dict(clamped_ticks=clips,rate_limited_ticks=limited)


def load(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if (ck['format'],ck['task'],ck['action_mode'],ck['chunk_size'],ck['image_size'],ck['fps'])!=(
            'yam_train/act_ref-v1','put_block_in_bowl','delta',100,[480,640],30):
        raise ValueError('Baseline/candidate contract mismatch')
    policy=ACTPolicy(ck['config']['policy'],ck['camera_names'])
    policy.load_state_dict(ck['model'],strict=True);policy.to(device).eval().requires_grad_(False)
    return ck,policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--baseline-split',type=Path,required=True)
    p.add_argument('--baseline-all',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Comparison requires the reserved CUDA device')
    torch.set_num_threads(4)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    store=FrameStore(args.data)
    new_ck=torch.load(args.new,map_location='cpu',weights_only=False)
    val_names=set(new_ck['val_episodes'])
    episodes=[e for e in store.episodes(new_ck['task']) if e['name'] in val_names]
    if not val_names or len(episodes)!=len(val_names):raise ValueError('Validation split missing')
    samples=fixed_samples(episodes,512)
    # Eight uniformly selected real frames per held-out episode for input/latency checks;
    # dense complete-episode predictions below feed the playback simulation.
    records=[]
    candidates={'new_seed42':args.new,'fufu_split':args.baseline_split,'fufu_all':args.baseline_all}
    for name,path in candidates.items():
        ck,policy=load(path,torch.device('cuda'))
        overlap=sorted(val_names&set(ck['train_episodes']))
        if name in ('new_seed42','fufu_split') and overlap:
            raise ValueError('Fair comparison has train/validation leakage: '+name)
        if ck['camera_names']!=new_ck['camera_names'] or ck['action_shift']!=new_ck['action_shift']:
            raise ValueError('Camera order or action timing differs across candidates')
        stats={k:np.asarray(v,np.float32) for k,v in ck['stats'].items()}
        ds=EpisodicDataset(store,episodes,ck['camera_names'],stats,ck['chunk_size'],ck['action_shift'],ck['action_mode'])
        metrics=extra_eval(policy,ds,samples,torch.device('cuda'),8)
        simulation=[];latencies=[];grips=[]
        for ep in episodes:
            n=ep['length'];start=ep['start'];chunks=[]
            with torch.inference_mode():
                for offset in range(0,n,8):
                    times=list(range(offset,min(n,offset+8)))
                    images=torch.stack([ds.sample(ep,t)[0] for t in times]).cuda().float()/255
                    qpos=store.qpos[start+np.asarray(times)].astype(np.float64)
                    qnorm=torch.from_numpy(((qpos-stats['qpos_mean'])/stats['qpos_std']).astype(np.float32)).cuda()
                    torch.cuda.synchronize();before=time.monotonic()
                    raw=policy.predict(qnorm,images).cpu().numpy()
                    torch.cuda.synchronize();latencies.append((time.monotonic()-before)/len(times))
                    absolute=raw*stats['action_std']+stats['action_mean']+qpos[:,None,:]
                    if not np.isfinite(absolute).all():raise ValueError('Nonfinite comparison outputs')
                    chunks.append(absolute);grips.append(absolute[:,:,[6,13]].reshape(-1))
            chunks=np.concatenate(chunks)
            command,feedback,details=simulate(chunks,lower=stats['leader_min'],upper=stats['leader_max'],
                                               initial=store.qpos[start])
            frame=np.clip((np.arange(len(command))/100*30).astype(int),0,n-1)
            target=store.action[start+frame]
            joints=np.r_[0:6,7:13]
            errors=np.abs(command-target)
            simulation.append(dict(episode=ep['name'],command_joint_mae=float(errors[:,joints].mean()),
                command_right_arm_mae=float(errors[:,7:13].mean()),command_gripper_mae=float(errors[:,[6,13]].mean()),
                simulated_tracking_mae=float(np.abs(feedback-command)[:,joints].mean()),**details))
            np.savez_compressed(out/(name+'_'+Path(ep['name']).name+'_playback.npz'),
                                command=command,simulated_feedback=feedback,target=target)
        grip=np.concatenate(grips)
        row=dict(model=name,path=str(path),sha256=sha(path),epoch=ck['epoch'],metrics=metrics,
            validation_training_overlap=overlap,
            comparison_scope='unseen validation' if not overlap else 'in-sample for this baseline; NOT generalization',
            gripper_out_of_range_fraction=float(np.mean((grip<0)|(grip>1))),
            gripper_raw_min=float(grip.min()),gripper_raw_max=float(grip.max()),
            gpu_batch_amortized_median_seconds=float(np.median(latencies)),
            simulation=simulation,
            simulation_right_arm_mae=float(np.mean([s['command_right_arm_mae'] for s in simulation])),
            simulation_gripper_mae=float(np.mean([s['command_gripper_mae'] for s in simulation])))
        if not np.isfinite([row['simulation_right_arm_mae'],row['simulation_gripper_mae'],*metrics.values()]).all():
            raise ValueError('Nonfinite model comparison metrics')
        records.append(row)
        (out/'comparison_progress.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
        print('COMPARED',name,'right arm',row['simulation_right_arm_mae'],'gripper',row['simulation_gripper_mae'],flush=True)
        del policy,ck
        torch.cuda.empty_cache()
    by_name={r['model']:r for r in records}
    def rank(r):return (r['simulation_right_arm_mae'],r['simulation_gripper_mae'],r['gripper_out_of_range_fraction'])
    fair_winner=min([by_name['new_seed42'],by_name['fufu_split']],key=rank)['model']
    # Conservative deployment recommendation: keep current working fufu_all baseline
    # unless new model is no worse in both arm and gripper playback metrics, and raw
    # gripper excursion frequency is not increased. Preserve both models regardless.
    new,baseline=by_name['new_seed42'],by_name['fufu_all']
    dominates=(new['simulation_right_arm_mae']<=baseline['simulation_right_arm_mae'] and
               new['simulation_gripper_mae']<=baseline['simulation_gripper_mae'] and
               new['gripper_out_of_range_fraction']<=baseline['gripper_out_of_range_fraction'])
    recommended='new_seed42' if dominates and fair_winner=='new_seed42' else 'fufu_all'
    retained=out/'retained';retained.mkdir()
    for name in ('new_seed42','fufu_all'):
        shutil.copy2(candidates[name],retained/(name+'.pt'))
    shutil.copy2(candidates[recommended],out/'offline_recommended.pt')
    report=dict(status='offline_comparison_complete',models=records,fair_validation_winner=fair_winner,
        offline_recommendation=recommended,
        selection_rule='new must win fair validation and not regress arm/gripper playback or excursion rate versus working full-data baseline',
        recommendation_sha256=sha(out/'offline_recommended.pt'),
        both_original_checkpoints_retained=True,existing_robot_deployment_changed=False,
        simulation='same recorded inputs, 60-step chunks, 0.1s blend, 100Hz original range/speed filter plus 130ms tracking lag',
        limitations=['Recorded observations are fixed, so this is not closed-loop grasp success simulation.',
                     'Full-data fufu baseline may have seen validation episodes; overlap is reported.',
                     'GPU batch-amortized timing excludes image load, CPU preprocessing and network.',
                     'Do not replace working robot model without local supervised validation.'],
        robot_commands_sent=0)
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('OFFLINE_MODEL_COMPARISON_COMPLETE; retained recommendation:',recommended,flush=True)


if __name__=='__main__':main()
