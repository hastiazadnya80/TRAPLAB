"""Prepare a four-config HF dataset locally; upload only with explicit --upload."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from traplab.analysis import summarize


def prepare(source, destination):
    source, destination = Path(source),Path(destination)
    summary = summarize(source)
    if not summary['complete']: raise ValueError('Refusing to package an incomplete run')
    if destination.exists() and any(destination.iterdir()): raise ValueError('Use an empty destination')
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('items','traces','rejected','stories'):
        original = source/(name+'.jsonl')
        if original.exists(): shutil.copyfile(original,destination/(name+'.jsonl'))
        else: (destination/(name+'.jsonl')).write_text('')
    for name in ('manifest.json','summary.json','calls.jsonl','proposals.jsonl','events.jsonl','proposal_events.jsonl'):
        if (source/name).exists(): shutil.copyfile(source/name,destination/name)
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    configs = '\n'.join(f'  - config_name: {name}\n    data_files:\n      - split: train\n        path: {name}.jsonl' for name in ('items','traces','rejected','stories'))
    label = 'SYNTHETIC DEMO — NOT BEAM EVIDENCE' if summary['demo'] else 'Observed model responses'
    (destination/'README.md').write_text('---\nlicense: mit\nconfigs:\n'+configs+'\n---\n\n# TRAPLAB artifacts\n\n'+label+'\n\nPython-generated questions and replayable statistical reasoning traces. See manifest and summary for protocol, model, source hashes, completeness and limitations. Empty rejected files mean no rejections were observed. Review raw traces before publication.\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('destination')
    p.add_argument('--upload',metavar='HF_DATASET_ID')
    a = p.parse_args()
    summary = prepare(a.run,a.destination)
    print('Dataset bundle prepared locally:',a.destination)
    if a.upload:
        if summary['demo']: p.error('Do not upload demo fixtures as competition evidence')
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.upload,repo_type='dataset',exist_ok=True,private=False)
        api.upload_folder(repo_id=a.upload,repo_type='dataset',folder_path=a.destination)
