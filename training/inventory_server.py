"""Small read-only inventory, never enumerates credentials or starts a job."""
import json
from pathlib import Path
import subprocess
import sys

home=Path.home()
candidates=[home/'output/yam_train',home/'output/code/yam_train',home/'output/Dev/MVIG/yam_train',
            home/'Dev/MVIG/yam_train',home/'output/data',home/'output/dataset',home/'dataset']
datasets=[]
for parent in candidates:
    if not parent.is_dir():continue
    for path in parent.glob('**/meta.json'):
        if len(path.relative_to(parent).parts)>5 or any(x in path.parts for x in ('.venv','site-packages','.git')):continue
        try:
            metadata=json.loads(path.read_text())
            if metadata.get('format')=='yam_train/jpeg-v1' and 'put_block_in_bowl' in metadata.get('tasks',[]):
                datasets.append(dict(path=str(path.parent),episodes=len([e for e in metadata.get('episodes',[]) if e.get('task')=='put_block_in_bowl'])))
        except (OSError,ValueError):pass
try:
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True,timeout=10)
except (OSError,subprocess.SubprocessError):gpu='nvidia-smi unavailable'
print(json.dumps(dict(python=sys.executable,home=str(home),converted_task_datasets=datasets,gpus=gpu,
                      training_started=False,hardware_connected=False),ensure_ascii=False,indent=2))
