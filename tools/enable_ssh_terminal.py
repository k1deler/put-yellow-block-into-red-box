"""Remove only the launcher's SSH-location restriction; never execute hardware."""
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
path=root/'start_local.sh'
old="[[ -z \"${SSH_CONNECTION:-}${SSH_TTY:-}\" && -t 0 ]] || { echo 'Use a local robot desktop terminal, not SSH'; exit 1; }"
new="[[ -t 0 ]] || { echo 'Use an interactive terminal (SSH: allocate a TTY with ssh -t)'; exit 1; }"
text=path.read_text(encoding='utf-8')
if old in text:
    backup=path.with_name('start_local.sh.before_ssh_enable')
    if backup.exists():
        raise SystemExit('Backup already exists; inspect before overwriting')
    shutil.copy2(path,backup)
    text=text.replace(old,new,1).replace('Local operator ready? Type START:', 'On-site operator ready? Type START:')
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        handle.write(text)
    print('SSH_TERMINAL_ENABLED; backup:',backup)
elif new in text:
    print('SSH_TERMINAL_ALREADY_ENABLED')
else:
    raise SystemExit('Unexpected launcher content; no changes made')
print('No service or hardware started. Interactive terminal and START confirmation remain required.')
