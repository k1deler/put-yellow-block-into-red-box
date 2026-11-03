"""Wait for an unused GPU, then run one bounded continuation; never preempt jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def idle_gpus():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [index for index, memory, usage in
            ([int(x.strip()) for x in row.split(',')] for row in raw.splitlines())
            if memory < 1000 and usage < 5]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--trainer', type=Path, default=Path(__file__).with_name('fine_tune.py'))
    p.add_argument('--scope', choices=['preserve_split', 'all'], default='preserve_split')
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--wait-seconds', type=int, default=7200)
    p.add_argument('--max-train-seconds', type=int, default=21600)
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.wait_seconds, args.max_train_seconds) < 1:
        p.error('Epoch and time limits must be positive')
    for path in (args.parent, args.trainer):
        if not path.is_file():
            p.error('Required file missing: ' + str(path))
    if not args.data.is_dir():
        p.error('Dataset directory missing')
    print(json.dumps(dict(mode='launch' if args.launch else 'preflight',
        scope=args.scope, epochs=args.epochs, idle_gpus=idle_gpus(),
        existing_jobs_stopped=False, deployment_changed=False)), flush=True)
    if not args.launch:
        return
    if sys.platform != 'linux':
        raise RuntimeError('GPU queue requires the authorized Linux training server')
    import fcntl
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    task_lock = Path(str(out) + '.queue.lock').open('a')
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if out.exists():
        raise RuntimeError('Existing training output preserved; refusing duplicate launch')
    record = dict(pid=os.getpid(), status='waiting_for_idle_gpu', scope=args.scope,
        additional_epochs=args.epochs, wait_limit_seconds=args.wait_seconds,
        max_train_seconds=args.max_train_seconds, robot_commands_sent=0,
        deployment_changed=False)
    def status(value, **fields):
        record.update(status=value, **fields)
        target = Path(str(out) + '.queue.json')
        temporary = Path(str(target) + '.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n'); temporary.replace(target)
    status('waiting_for_idle_gpu')
    deadline = time.monotonic() + args.wait_seconds
    lockdir = Path.home() / '.cache/k1deler_task1_training'
    lockdir.mkdir(parents=True, exist_ok=True)
    try:
        while time.monotonic() < deadline:
            for gpu in idle_gpus():
                lock = (lockdir / f'gpu_{gpu}.lock').open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close(); continue
                try:
                    # A second idle observation catches short utilization gaps.
                    time.sleep(5)
                    if gpu not in idle_gpus():
                        continue
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                    cmd = [sys.executable, '-u', str(args.trainer.resolve()),
                           '--parent', str(args.parent.resolve()), '--data', str(args.data.resolve()),
                           '--out', str(out), '--scope', args.scope, '--epochs', str(args.epochs),
                           '--device', 'cuda', '--max-seconds', str(args.max_train_seconds), '--launch']
                    with Path(str(out) + '.train.log').open('x') as log:
                        child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, start_new_session=True)
                        status('training', gpu=gpu, child_pid=child.pid)
                        try:
                            code = child.wait(timeout=args.max_train_seconds + 600)
                        except BaseException:
                            # Only this newly created process group is signalled.
                            if child.poll() is None:
                                os.killpg(child.pid, signal.SIGTERM)
                                try:
                                    child.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
                            raise
                    train_status = json.loads((out / 'status.json').read_text())
                    completed = code == 0 and train_status['status'] == 'training_complete_pending_validation'
                    status('training_complete_pending_validation' if completed else 'training_failed_or_incomplete',
                           exit_code=code, completed_epochs=train_status.get('completed_epochs', 0))
                    return
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        status('wait_expired_no_training_started')
    except BaseException as error:
        status('queue_failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        fcntl.flock(task_lock, fcntl.LOCK_UN); task_lock.close()


if __name__ == '__main__':
    main()
