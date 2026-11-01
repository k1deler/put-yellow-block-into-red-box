"""Read-only audit of a converted yellow/red task dataset before GPU training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASK = 'put_block_in_bowl'
CAMERAS = ['right_wrist', 'left_wrist']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(root, ratio=.8, split_seed=1):
    root = Path(root).resolve()
    meta = json.loads((root/'meta.json').read_text(encoding='utf-8'))
    if meta.get('format') != 'yam_train/jpeg-v1':
        raise ValueError('Requires the original yam_train JPEG dataset format; do not use LeRobot arrays here')
    if meta.get('fps') != 30 or meta.get('image_size') != [480, 640]:
        raise ValueError('Wrong image/frequency contract')
    if set(meta.get('cameras', [])) != set(CAMERAS):
        raise ValueError('Both wrist cameras are required')
    if meta.get('qpos') != 'follower' or meta.get('action') != 'leader':
        raise ValueError('Supervision must be timestamp-aligned leader targets and follower observations')
    if meta.get('state_layout') != 'left_joint0..5,left_gripper,right_joint0..5,right_gripper':
        raise ValueError('Wrong 14D state/action order')
    episodes = sorted((e for e in meta['episodes'] if e['task'] == TASK), key=lambda e:e['name'])
    if len(episodes) < 5:
        raise ValueError('Too few complete task episodes for this train/validation experiment')
    if len({e['name'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate episode identities')
    q = np.load(root/'qpos.npy', mmap_mode='r')
    a = np.load(root/'action.npy', mmap_mode='r')
    if q.shape != a.shape or q.ndim != 2 or q.shape[1] != 14:
        raise ValueError('Expected matching [N,14] observation/target arrays')
    selected = set()
    intervals = []
    for e in episodes:
        start, n = e['start'], e['length']
        if not isinstance(start,int) or not isinstance(n,int) or start < 0 or n < 100 or start+n > len(q):
            raise ValueError('Invalid or too-short task episode: '+e['name'])
        if 'episode_0024' in e['name'] or e['name'].endswith('_inprogress'):
            raise ValueError('Failed/incomplete episode must be excluded before training: '+e['name'])
        intervals.append((start,start+n))
        rows = slice(start,start+n)
        if not np.isfinite(q[rows]).all() or not np.isfinite(a[rows]).all():
            raise ValueError('Nonfinite physical training data')
        for value in (q[rows],a[rows]):
            if np.any(value[:,[6,13]] < 0) or np.any(value[:,[6,13]] > 1):
                raise ValueError('Recorded physical gripper outside [0,1]')
        selected.update(range(start,start+n))
    intervals.sort()
    if any(first[1] > second[0] for first,second in zip(intervals,intervals[1:])):
        raise ValueError('Episode row ranges overlap')
    rows = np.asarray(sorted(selected),dtype=np.int64)
    for camera in CAMERAS:
        index = np.load(root/(camera+'.idx.npy'),mmap_mode='r')
        size = (root/(camera+'.jpg.bin')).stat().st_size
        if index.shape != (len(q),2) or index.dtype.kind not in 'iu':
            raise ValueError('Invalid JPEG row index')
        chosen = index[rows]
        if np.any(chosen[:,0]<0) or np.any(chosen[:,1]<=0) or np.any(chosen.sum(1)>size):
            raise ValueError('JPEG index escapes file bounds')
        # Decode one real frame from every episode and camera, without opening a camera.
        import cv2
        with (root/(camera+'.jpg.bin')).open('rb') as stream:
            for episode in episodes:
                offset,length=index[episode['start']]
                stream.seek(int(offset))
                image=cv2.imdecode(np.frombuffer(stream.read(int(length)),np.uint8),cv2.IMREAD_COLOR)
                if image is None or image.shape!=(480,640,3):
                    raise ValueError('Real JPEG dimensions/decoding failed for '+camera)
    order = np.random.RandomState(split_seed).permutation(len(episodes))
    split = int(ratio*len(episodes))
    train = [episodes[i]['name'] for i in order[:split]]
    val = [episodes[i]['name'] for i in order[split:]]
    if not train or not val or set(train)&set(val):
        raise ValueError('Invalid train/validation episode split')
    files = ['meta.json','qpos.npy','action.npy']
    files += [c+suffix for c in CAMERAS for suffix in ('.idx.npy','.jpg.bin')]
    return dict(task=TASK,episodes=len(episodes),task_frames=len(rows),
                train_episodes=train,val_episodes=val,split_seed=split_seed,
                normalization='train episodes only',
                dataset_sha256={name:sha(root/name) for name in files},
                hardware_connected=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.data),ensure_ascii=False,indent=2))
