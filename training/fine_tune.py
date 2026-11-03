"""Bounded continuation from a trusted task checkpoint into a NEW directory.

The parent is never overwritten. Original exports have no optimizer state, so
this is explicitly a warm start, not an exact resumption of the original run.
No inference runner, arm, camera, CAN, or robot worker is imported.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_parent(ck):
    expected = dict(format='yam_train/act_ref-v1', task='put_block_in_bowl',
                    action_mode='delta', action_shift=0, chunk_size=100, fps=30,
                    image_size=[480, 640], camera_names=['right_wrist', 'left_wrist'])
    for key, value in expected.items():
        if ck.get(key) != value:
            raise ValueError('Parent checkpoint contract mismatch: ' + key)
    train, val = set(ck['train_episodes']), set(ck['val_episodes'])
    if not train or not val or train & val:
        raise ValueError('Expected a parent with disjoint train/validation episodes')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scope', choices=['preserve_split', 'all'], default='preserve_split')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--lr', type=float, default=5e-6)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--eval-samples', type=int, default=128)
    p.add_argument('--max-seconds', type=int, default=21600)
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    if args.epochs < 1 or args.save_every < 1 or args.eval_samples < 1 or args.max_seconds < 1:
        p.error('Epochs, checkpoint interval, evaluation count and time limit must be positive')
    if not 0 < args.lr <= 1e-5:
        p.error('Continuation learning rate must be positive and <= 1e-5')
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT / 'upstream'))
    from yam_train.data.dataset import FrameStore, EpisodicDataset
    from yam_train.policy import ACTPolicy
    from yam_train.detr.models import backbone
    from yam_train.train import fixed_samples, extra_eval, forward_pass
    from torch.utils.data import DataLoader
    ck = torch.load(args.parent, map_location='cpu', weights_only=False)
    validate_parent(ck)
    store = FrameStore(args.data)
    if store.meta['fps'] != 30 or store.meta['image_size'] != [480, 640]:
        raise ValueError('Dataset FPS/image contract mismatch')
    episodes = store.episodes(ck['task'])
    train_names = set(ck['train_episodes'])
    reference_names = set(ck['val_episodes'])
    allowed = train_names | reference_names
    if set(ep['name'] for ep in episodes) != allowed:
        raise ValueError('Dataset episodes differ from the audited parent dataset')
    if any('episode_0024' == Path(ep['name']).name for ep in episodes):
        raise ValueError('Known failed episode must remain excluded')
    training = [ep for ep in episodes if args.scope == 'all' or ep['name'] in train_names]
    reference = [ep for ep in episodes if ep['name'] in reference_names]
    st = {k: np.asarray(v, np.float32) for k, v in ck['stats'].items()}
    for name, values in st.items():
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError('Invalid parent normalization: ' + name)
        if name.endswith('_std') and not np.all(values > 0):
            raise ValueError('Nonpositive parent standard deviation')
    indices = np.concatenate([np.arange(ep['start'], ep['start'] + ep['length']) for ep in episodes])
    for label, values in [('qpos', store.qpos[indices]), ('action', store.action[indices])]:
        if not np.isfinite(values).all() or np.any((values[:, [6, 13]] < 0) | (values[:, [6, 13]] > 1)):
            raise ValueError('Invalid finite state/gripper data: ' + label)
    provenance = dict(status='preflight', parent_sha256=sha(args.parent), parent_epoch=ck['epoch'],
        scope=args.scope, training_episodes=len(training), reference_episodes=len(reference),
        reference_is_held_out=args.scope == 'preserve_split',
        reference_overlap=len(reference) if args.scope == 'all' else 0,
        normalization='Inherited parent statistics unchanged', optimizer='AdamW reset for warm start',
        additional_epochs=args.epochs, learning_rate=args.lr, device=args.device,
        hardware_connected=False, robot_commands_sent=0, current_deployment_changed=False)
    print(json.dumps(provenance), flush=True)
    if not args.launch:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; no silent CPU fallback')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def status(value, **fields):
        provenance.update(status=value, **fields)
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(provenance, indent=2) + '\n')
        temporary.replace(out / 'status.json')
    status('initializing')
    seed = 142
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    cfg = copy.deepcopy(ck['config'])
    cfg['policy']['lr'] = args.lr
    cfg['policy']['lr_backbone'] = args.lr
    cfg['train'].update(seed=seed, num_workers=0, num_epochs=args.epochs)
    cfg['data']['train_ratio'] = 1.0 if args.scope == 'all' else cfg['data']['train_ratio']
    # Do not re-download any backbone: the entire model comes from the parent.
    backbone.is_main_process = lambda: False
    device = torch.device(args.device)
    policy = ACTPolicy(cfg['policy'], ck['camera_names'])
    policy.load_state_dict(ck['model'], strict=True)
    policy.to(device)
    optimizer = policy.configure_optimizers()
    ds_kw = dict(camera_names=ck['camera_names'], norm_stats=st,
                 chunk_size=100, action_shift=0, action_mode='delta')
    train_ds = EpisodicDataset(store, training, **ds_kw)
    ref_ds = EpisodicDataset(store, reference, **ds_kw)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    samples = fixed_samples(reference, args.eval_samples)
    base_fields = {k: ck[k] for k in ['format', 'task', 'camera_names', 'image_size', 'fps',
                   'chunk_size', 'action_mode', 'action_shift', 'stats']}
    base_fields.update(config=cfg, train_episodes=[ep['name'] for ep in training],
        val_episodes=[ep['name'] for ep in reference] if args.scope == 'preserve_split' else [],
        diagnostic_reference_episodes=[ep['name'] for ep in reference],
        start_pose=np.mean([store.qpos[ep['start']] for ep in training], axis=0).tolist(),
        continuation=dict(parent_sha256=provenance['parent_sha256'], parent_epoch=ck['epoch'],
                          optimizer_reset=True, reference_is_held_out=args.scope == 'preserve_split'))
    if 'source_provenance' in ck:
        base_fields['source_provenance'] = ck['source_provenance']
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    start = time.monotonic()
    def save(epoch, name):
        exported = {**base_fields, 'model': policy.state_dict(), 'epoch': ck['epoch'] + epoch + 1,
                    'continuation_epoch': epoch, 'robot_tested': False}
        tmp = out / (name + '.tmp')
        torch.save(exported, tmp); tmp.replace(out / name)
        # Preserve optimizer/RNG in a separate, explicitly labelled training state.
        resume = {**exported, 'optimizer': optimizer.state_dict(),
                  'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                  'python_rng': random.getstate(),
                  'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}
        tmp = out / 'resume_state.tmp'
        torch.save(resume, tmp); tmp.replace(out / 'resume_state.pt')
    try:
        status('evaluating_parent')
        before = extra_eval(policy, ref_ds, samples, device, 8)
        (out / 'before_metrics.json').write_text(json.dumps(before, indent=2) + '\n')
        status('training', completed_epochs=0)
        with (out / 'metrics.jsonl').open('x') as metrics:
            for epoch in range(args.epochs):
                if time.monotonic() - start >= args.max_seconds:
                    save(epoch - 1, 'last.pt')
                    status('time_limit_reached', completed_epochs=epoch)
                    return
                policy.train(); losses = []
                for data in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = forward_pass(data, policy, device)['loss']
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                row = dict(epoch=epoch, loss=float(np.mean(losses)), elapsed_seconds=time.monotonic() - start)
                if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                    row['reference_metrics'] = extra_eval(policy, ref_ds, samples, device, 8)
                    save(epoch, 'last.pt' if epoch == args.epochs - 1 else f'epoch_{epoch + 1:05d}.pt')
                metrics.write(json.dumps(row) + '\n'); metrics.flush()
                status('training', completed_epochs=epoch + 1)
                if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                    print('CONTINUATION_EPOCH', json.dumps(row), flush=True)
        status('training_complete_pending_validation', completed_epochs=args.epochs,
               candidate_sha256=sha(out / 'last.pt'))
        print('TASK1_CONTINUATION_COMPLETE; deployment unchanged', flush=True)
    except BaseException as error:
        status('failed', error=type(error).__name__ + ': ' + str(error))
        raise


if __name__ == '__main__':
    main()
