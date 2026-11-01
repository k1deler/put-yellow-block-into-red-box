"""Select a genuinely trained checkpoint with deterministic z=0 held-out metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import torch
    from yam_train.data.dataset import FrameStore,EpisodicDataset
    from yam_train.policy import ACTPolicy
    from yam_train.train import extra_eval,fixed_samples
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    store=FrameStore(args.data)
    checkpoints=sorted((args.run/'ckpt').glob('epoch_*.pt'))+list((args.run/'ckpt').glob('last.pt'))
    if not checkpoints:raise ValueError('No trained candidates')
    records=[];best=None
    for path in checkpoints:
        ck=torch.load(path,map_location='cpu',weights_only=False)
        train_names=set(ck['train_episodes']);val_names=set(ck['val_episodes'])
        if not val_names or train_names&val_names:raise ValueError('Invalid held-out split')
        eps=store.episodes(ck['task']);val=[e for e in eps if e['name'] in val_names]
        if len(val)!=len(val_names):raise ValueError('Validation episodes missing')
        stats={k:np.asarray(v,np.float32) for k,v in ck['stats'].items()}
        ds=EpisodicDataset(store,val,ck['camera_names'],stats,ck['chunk_size'],ck['action_shift'],ck['action_mode'])
        policy=ACTPolicy(ck['config']['policy'],ck['camera_names'])
        policy.load_state_dict(ck['model'],strict=True);policy.cuda().eval().requires_grad_(False)
        metrics=extra_eval(policy,ds,fixed_samples(val,512),torch.device('cuda'),8)
        if not np.isfinite(list(metrics.values())).all():raise ValueError('Nonfinite evaluation')
        row=dict(checkpoint=str(path),epoch=ck['epoch'],metrics=metrics,
                 evaluation='fixed 512 held-out windows, z=0, original physical target space; not closed-loop success')
        records.append(row)
        (out/'selection.json').write_text(json.dumps(records,indent=2)+'\n')
        score=(metrics['right_arm_rad'],metrics['right_grip'])
        if best is None or score<best[0]:best=(score,path)
        del policy,ck
        torch.cuda.empty_cache()
    chosen=torch.load(best[1],map_location='cpu',weights_only=False)
    training=[e for e in store.episodes(chosen['task']) if e['name'] in set(chosen['train_episodes'])]
    chosen['start_pose']=np.mean([store.qpos[e['start']] for e in training],axis=0).tolist()
    chosen['experiment_note']='New seed42 ACT training, train-only normalization and start pose, source yam_train attributed'
    chosen['selection']=dict(source_checkpoint=str(best[1]),rule='minimum held-out right-arm MAE; gripper MAE tie-break',
                             validation_windows=512,robot_tested=False)
    chosen['source_provenance']=json.loads((ROOT/'source.json').read_text())
    model=out/'yellow_red_seed42_selected.pt'
    torch.save(chosen,model)
    print('SELECTED_NEW_TRAINED_MODEL',model,flush=True)


if __name__=='__main__':main()
