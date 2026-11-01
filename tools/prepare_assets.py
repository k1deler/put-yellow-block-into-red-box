"""Restore public third-party binary assets from the pinned upstream commit.

No trained weights, private recordings, or credentials are downloaded by this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore(root=ROOT, check_only=False):
    manifest = json.loads((root / 'binary_assets.json').read_text(encoding='utf-8'))
    restored = 0
    for item in manifest['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe asset path')
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Asset path escapes repository')
        if path.is_file():
            if digest(path) != item['sha256']:
                raise ValueError('Existing asset differs; preserved without overwrite: ' + str(path))
            continue
        if check_only:
            raise FileNotFoundError('Asset missing; run python3 tools/prepare_assets.py: ' + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'yellow-red-task1-runtime'})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(item['bytes'] + 1)
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Public upstream asset hash/length mismatch: ' + item['path'])
        with path.open('xb') as target:
            target.write(raw)
        restored += 1
    return {'verified_assets': len(manifest['files']), 'downloaded_assets': restored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(restore(check_only=args.check)))


if __name__ == '__main__':
    main()
