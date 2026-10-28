import argparse
import json
from pathlib import Path
from .beam_client import BeamClient, DemoClient, APIError
from .runner import run, smoke, generate_stories
from .storage import Store, read_jsonl
from .core import DEFAULT_STORIES, Rejected
from .analysis import summarize


def main():
    parser = argparse.ArgumentParser(prog='traplab')
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('smoke','run','stories','extensions'):
        p = sub.add_parser(name)
        p.add_argument('--output', required=True)
        if name != 'stories': p.add_argument('--demo',action='store_true',help='Synthetic offline fixture; never Beam evidence')
        if name == 'run':
            p.add_argument('--config',required=True)
            p.add_argument('--stories',help='JSON array generated once and shared by both arms')
        if name == 'extensions':
            p.add_argument('--source',required=True)
            p.add_argument('--bases',type=int,default=20)
            p.add_argument('--self-limit',type=int,default=60)
            p.add_argument('--flip-configs',type=int,default=3)
    p = sub.add_parser('reproduce')
    p.add_argument('path')
    args = parser.parse_args()
    try:
        if args.command == 'reproduce':
            path = Path(args.path)
        else:
            demo = getattr(args,'demo',False)
            client = DemoClient() if demo else BeamClient()
            if args.command == 'stories':
                store = Store(args.output)
                target = store.path/'pool.json'
                if target.exists(): raise ValueError('Story pool exists; refusing to overwrite')
                stories = generate_stories(client,store)
                target.write_text(json.dumps(stories,indent=2)+'\n')
                print('Validated story pool saved:',target)
                return
            if args.command == 'smoke': smoke(client,args.output,demo)
            elif args.command == 'extensions':
                from .experiments import run_extensions
                run_extensions(client,args.source,args.output,demo,bases=args.bases,self_limit=args.self_limit,flip_configs=args.flip_configs)
            else:
                stories = json.loads(Path(args.stories).read_text()) if args.stories else DEFAULT_STORIES
                if not demo and not args.stories: raise ValueError('Live runs require a shared generated --stories pool; generate it once with traplab stories.')
                story_calls = read_jsonl(Path(args.stories).parent/'calls.jsonl') if args.stories else None
                if not demo and not story_calls: raise ValueError('Live story pool must retain its adjacent calls.jsonl for shared generation cost accounting.')
                run(client,json.loads(Path(args.config).read_text()),args.output,stories,demo,story_calls)
            path = Path(args.output)
        summary = summarize(path)
        (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    except (APIError,ValueError,RuntimeError,Rejected) as exc:
        parser.exit(1,str(exc)+'\n')


if __name__ == '__main__': main()
