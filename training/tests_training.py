"""CPU-only software tests, using synthetic data; they do not train a deployable task model."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'upstream'),str(ROOT.parents[0]/'project')]
from audit import audit  # noqa: E402


def make_dataset(tmp_path):
    import cv2
    root=tmp_path/'data';root.mkdir()
    episodes=[dict(name=f'put_block_in_bowl/episode_{i:04d}',task='put_block_in_bowl',
                   start=i*110,length=110) for i in range(6)]
    q=np.full((660,14),.2,np.float32);q[:,[6,13]]=.5
    np.save(root/'qpos.npy',q);np.save(root/'action.npy',q+.05)
    jpeg=cv2.imencode('.jpg',np.full((480,640,3),100,np.uint8))[1].tobytes()
    for camera in ('right_wrist','left_wrist'):
        (root/(camera+'.jpg.bin')).write_bytes(jpeg)
        np.save(root/(camera+'.idx.npy'),np.tile([0,len(jpeg)],(660,1)))
    meta=dict(format='yam_train/jpeg-v1',fps=30,image_size=[480,640],
              cameras=['right_wrist','left_wrist'],qpos='follower',action='leader',
              state_layout='left_joint0..5,left_gripper,right_joint0..5,right_gripper',episodes=episodes)
    (root/'meta.json').write_text(json.dumps(meta))
    return root


def test_episode_split_is_deterministic_and_disjoint(tmp_path):
    root=make_dataset(tmp_path)
    first=audit(root);second=audit(root)
    assert first==second
    assert first['episodes']==6 and len(first['train_episodes'])==4 and len(first['val_episodes'])==2
    assert not set(first['train_episodes'])&set(first['val_episodes'])
    assert first['normalization']=='train episodes only'


@pytest.mark.parametrize('kind',['nan','gripper','overlap','failed_episode','image_size','action_source'])
def test_invalid_training_data_rejected(tmp_path,kind):
    root=make_dataset(tmp_path)
    meta=json.loads((root/'meta.json').read_text())
    if kind in ('nan','gripper'):
        q=np.load(root/'qpos.npy');q[0,0 if kind=='nan' else 6]=np.nan if kind=='nan' else 1.2
        np.save(root/'qpos.npy',q)
    elif kind=='overlap':meta['episodes'][1]['start']=5
    elif kind=='failed_episode':meta['episodes'][0]['name']='put_block_in_bowl/episode_0024'
    elif kind=='image_size':meta['image_size']=[240,320]
    elif kind=='action_source':meta['action']='future_follower'
    (root/'meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError):audit(root)


def test_training_and_deployed_act_contract_roundtrip(tmp_path,monkeypatch):
    import torch
    from yam_train.policy import ACTPolicy
    from yam_train.detr.models import backbone
    monkeypatch.setattr(backbone,'is_main_process',lambda:False)
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=dict(chunk_size=8,kl_weight=10,hidden_dim=64,dim_feedforward=128,enc_layers=1,dec_layers=1,
             nheads=8,dropout=.1,pre_norm=False,backbone='resnet18',position_embedding='sine',
             dilation=False,masks=False,lr=1e-5,lr_backbone=1e-5,weight_decay=1e-4)
    cameras=['right_wrist','left_wrist']
    model=ACTPolicy(cfg,cameras)
    image=torch.rand(1,2,3,64,64);q=torch.full((1,14),.3)
    actions=torch.zeros(1,8,14);pad=torch.zeros(1,8,dtype=torch.bool)
    optimizer=model.configure_optimizers()
    model.train();before={k:v.detach().clone() for k,v in model.named_parameters() if 'action_head' in k}
    assert before
    loss=model(q,image,actions,pad)['loss']
    assert torch.isfinite(loss)
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert any(not torch.equal(before[k],v.detach()) for k,v in model.named_parameters() if k in before)
    model.eval()
    with torch.inference_mode():expected=model.predict(q,image)
    from yam_infer.policy.act_ref import ACTRefPolicy
    deployed=ACTRefPolicy(cfg,cameras)
    deployed.load_state_dict(model.state_dict(),strict=True);deployed.eval()
    with torch.inference_mode():actual=deployed(q,image)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # A tiny synthetic-gradient test is never offered as a trained task checkpoint.
    assert not list(tmp_path.glob('*.pt'))


def test_comparison_playback_matches_original_scheduler_and_filter():
    from compare_baseline import simulate
    from yam_infer.control.scheduler import ActionScheduler,SafetyFilter
    rng=np.random.default_rng(3)
    chunks=np.cumsum(rng.normal(0,.002,(130,100,14)),axis=1)+.5
    lower=np.zeros(14);upper=np.ones(14);initial=np.full(14,.5)
    actual,feedback,details=simulate(chunks,lower=lower,upper=upper,initial=initial)
    scheduler=ActionScheduler(30,100,mode='chunk',horizon=60,blend_s=.1)
    safety=SafetyFilter(lower,upper);safety.reset(initial)
    expected=[];active=-1
    for tick in range(len(actual)):
        now=tick/100
        anchor=(min(int(now*30),len(chunks)-1)//60)*60
        if anchor!=active:
            scheduler.add(anchor/30,chunks[anchor]);active=anchor
        expected.append(safety(scheduler.target_at(now),.01))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    assert feedback.shape==actual.shape
    assert details['clamped_ticks']==safety.clamped
    assert details['rate_limited_ticks']==safety.rate_limited
