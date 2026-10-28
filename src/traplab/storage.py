from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from .core import PROTOCOL, digest


def read_jsonl(path):
    path = Path(path)
    if not path.exists(): return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def append(self, name, row):
        with (self.path/(name+'.jsonl')).open('a') as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            f.flush()
            import os
            os.fsync(f.fileno())

    def manifest(self, config, stories, model, demo):
        path = self.path/'manifest.json'
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        except subprocess.CalledProcessError:
            commit = None
        # Source hashes also identify uncommitted and initially unborn checkouts.
        source = {str(p.relative_to(Path(__file__).parent)):digest(p.read_text()) for p in sorted(Path(__file__).parent.rglob('*.py'))}
        identity = {'protocol':PROTOCOL,'config':config,'stories':stories,'model':model,'demo':demo,'source':source}
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['identity'] != digest(identity):
                raise ValueError('Run directory has a different source/config/model/story identity; use a new output directory.')
            return previous
        manifest = {**identity,'identity':digest(identity),'created_at':datetime.now(timezone.utc).isoformat(),'commit':commit}
        path.write_text(json.dumps(manifest, indent=2)+'\n')
        for story in stories: self.append('stories',story)
        return manifest
