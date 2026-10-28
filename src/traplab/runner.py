"""Budgeted, resumable experiments. No automatic live runs or publication."""
from __future__ import annotations
import json
import random
from .core import Gate, Rejected, DEFAULT_STORIES, FAMILIES, build, digest, validate_story
from .sampling import PARAMETER_SPACE, random_proposal, smoke_proposal
from .evaluator import parse_json, solve, solver_messages
from .storage import Store, read_jsonl


def seeded(*parts):
    return random.Random(int(digest(parts)[:16], 16))


def validate_config(config):
    expected = {'seed','replicates','items_per_family_arm','round_size','max_proposals_per_round','families'}
    if set(config) != expected: raise ValueError('Unexpected config keys')
    for k in expected-{'families'}:
        if type(config[k]) is not int or config[k] < (0 if k == 'seed' else 1): raise ValueError('Invalid '+k)
    if not config['families'] or len(set(config['families'])) != len(config['families']) or any(f not in FAMILIES for f in config['families']): raise ValueError('Invalid families')
    if config['max_proposals_per_round'] < config['round_size']: raise ValueError('Proposal cap must cover a round')


def designer_messages(family, count, stories, history):
    return [{'role':'system','content':'You design statistical reasoning traps. Return JSON only: {"proposals":[...]}. Each proposal has exactly family, params, story_id. Never write questions or answers. Aim to discover NEW parameter cells where the solver errs; do not repeat old failures. You receive no solver explanations.'}, {'role':'user','content':json.dumps({'family':family,'count':count,'allowed_params':PARAMETER_SPACE[family], 'stories':stories,'history':history,'gate_rules':'All displayed cells >=20; rate abs differences <.01 are zero, [.01,.03) rejected; RTM abs effects <.05 are zero, [.05,.2) rejected. Duplicate binned parameter cell plus story within an arm is rejected. Rounding is applied before grading.'})}]


def generate_stories(client, store, count=5):
    messages = [{'role':'system','content':'Return JSON only: {"stories":[...]}. Each story has exactly id, domain, unit, outcome. All values must contain only ASCII letters, spaces or hyphens, max 60 characters, no digits, no causal, temporal, treatment, selection or improvement language. Use neutral nouns only.'}, {'role':'user','content':f'Generate {count} diverse narrative skins for statistical questions. No scientific questions or claims.'}]
    for multiplier in (1,2):
        response = client.complete(messages,max_tokens=client.max_tokens*multiplier)
        store.append('calls',{'purpose':'stories','messages':messages,'response':response,'attempt':multiplier})
        if response['finish_reason'] != 'length': break
    if response['finish_reason'] != 'stop': raise ValueError('Story generation did not finish normally')
    data = parse_json(response['content'])
    if not isinstance(data,dict) or set(data) != {'stories'} or not isinstance(data['stories'],list) or len(data['stories']) != count: raise ValueError('Invalid story pool envelope')
    stories = [validate_story(s) for s in data['stories']]
    if len({s['id'] for s in stories}) != count: raise ValueError('Duplicate story ids')
    return stories


def evaluate_stored(client, store, item, slot, metadata, seed, messages=None):
    messages = messages or solver_messages(item,seeded(seed,slot,'options'))
    calls = [r for r in read_jsonl(store.path/'calls.jsonl') if r.get('slot') == slot and r.get('purpose') == 'solver']
    if calls:
        # A persisted response can be recovered without spending another call.
        from .evaluator import grade
        if calls[-1]['response']['finish_reason'] != 'length' or len(calls) >= 2:
            outcome = {'messages':messages,'attempts':[c['response'] for c in calls], 'result':grade(item,calls[-1]['response'])}
        else:
            response = client.complete(messages,max_tokens=client.max_tokens*2)
            store.append('calls',{'purpose':'solver','slot':slot,'attempt':2,'messages':messages,'response':response})
            outcome = {'messages':messages,'attempts':[calls[0]['response'],response], 'result':grade(item,response)}
    else:
        def record(response, attempt):
            store.append('calls',{'purpose':'solver','slot':slot,'attempt':attempt,'messages':messages,'response':response})
        outcome = solve(client,item,messages,record)
    row = {'slot':slot,'item_id':item.id,**metadata,**outcome}
    store.append('traces',row)
    return row


def smoke(client, output, demo=False):
    store = Store(output)
    config = {'kind':'smoke','seed':17}
    store.manifest(config,DEFAULT_STORIES,client.model,demo)
    if read_jsonl(store.path/'traces.jsonl'): return
    item = Gate().accept(smoke_proposal())
    if not read_jsonl(store.path/'items.jsonl'): store.append('items',{'slot':'smoke','item':item.to_dict()})
    evaluate_stored(client,store,item,'smoke',{'arm':'smoke','replicate':0,'family':item.family},17)


def run(client, config, output, stories=DEFAULT_STORIES, demo=False, story_calls=None):
    validate_config(config)
    if not isinstance(stories,list) or not stories: raise ValueError('A nonempty shared story pool is required')
    for story in stories: validate_story(story)
    if len({s['id'] for s in stories}) != len(stories): raise ValueError('Story ids must be unique')
    store = Store(output)
    manifest_config = {**config,'shared_story_calls_hash':digest(story_calls)} if story_calls is not None else config
    store.manifest(manifest_config,stories,client.model,demo)
    if story_calls is not None:
        imported = {r.get('import_id') for r in read_jsonl(store.path/'calls.jsonl')}
        for call in story_calls:
            if call.get('purpose') != 'stories': continue
            import_id = digest(call)
            if import_id not in imported:
                store.append('calls',{**call,'import_id':import_id,'shared_story_cost':True})
    completed = {r['slot']:r for r in read_jsonl(store.path/'traces.jsonl')}
    saved_items = {r['slot']:r for r in read_jsonl(store.path/'items.jsonl')}
    saved_batches = {r['batch']:r for r in read_jsonl(store.path/'proposals.jsonl')}
    # Every attempt at a proposal has a stable ID, including rejected proposals.
    attempted = {r['proposal_id'] for r in read_jsonl(store.path/'proposal_events.jsonl')}
    gates = {}
    for rep in range(config['replicates']):
        for arm in ('random','adaptive'):
            key = (rep,arm)
            gates[key] = Gate(stories)
            for row in saved_items.values():
                if (row['replicate'],row['arm']) == key:
                    item = row['item']
                    gates[key].seen.add((item['family'],item['cell'],item['story_id']))
    n, size = config['items_per_family_arm'],config['round_size']
    # Alternate arms within rounds, with seeded order per independent run.
    for start in range(0,n,size):
        count = min(size,n-start)
        for rep in range(config['replicates']):
            for family in config['families']:
                arms = ['random','adaptive']
                seeded(config['seed'],rep,start,family,'order').shuffle(arms)
                for arm in arms:
                    batch = f'{rep}/{family}/{arm}/{start}'
                    prefix = f'{rep}/{family}/{arm}/'
                    slots = [prefix+str(i) for i in range(start,start+count)]
                    for slot in slots:
                        if slot in saved_items and slot not in completed:
                            row = saved_items[slot]
                            item = build({'family':family,'params':row['item']['params'],'story_id':row['item']['story_id']},stories)
                            completed[slot] = evaluate_stored(client,store,item,slot,{'arm':arm,'replicate':rep,'family':family,'index':int(slot.split('/')[-1])},config['seed'])
                    if all(slot in completed for slot in slots): continue
                    history = []
                    for slot,trace in sorted(completed.items()):
                        if slot.startswith(prefix) and trace.get('index',0) < start:
                            item = saved_items[slot]['item']
                            history.append({'params':item['params'],'tag':item['tag'],'solver_answer':trace['result']['answer'],'status':trace['result']['status']})
                    if arm == 'adaptive' and batch not in saved_batches:
                        if demo:
                            rng = seeded(config['seed'],batch,'synthetic-designer')
                            proposals = [random_proposal(family,rng,stories) for _ in range(config['max_proposals_per_round'])]
                        else:
                            messages = designer_messages(family,config['max_proposals_per_round'],stories,history)
                            for multiplier in (1,2):
                                response = client.complete(messages,max_tokens=client.max_tokens*multiplier)
                                store.append('calls',{'purpose':'designer','batch':batch,'attempt':multiplier,'messages':messages,'response':response})
                                if response['finish_reason'] != 'length': break
                            try:
                                if response['finish_reason'] != 'stop': raise ValueError('designer_not_finished')
                                data = parse_json(response['content'])
                                if not isinstance(data,dict) or set(data) != {'proposals'} or not isinstance(data['proposals'],list): raise ValueError('designer_schema')
                                proposals = data['proposals'][:config['max_proposals_per_round']]
                            except (ValueError,TypeError) as exc:
                                store.append('rejected',{'batch':batch,'reason':'designer_envelope','detail':str(exc)})
                                raise RuntimeError('Designer returned invalid JSON; run stopped with evidence saved.') from None
                        saved_batches[batch] = {'batch':batch,'proposals':proposals}
                        store.append('proposals',saved_batches[batch])
                    for attempt in range(config['max_proposals_per_round']):
                        missing = [slot for slot in slots if slot not in completed]
                        if not missing: break
                        proposal_id = batch+'/'+str(attempt)
                        if proposal_id in attempted: continue
                        if arm == 'random':
                            proposal = random_proposal(family,seeded(config['seed'],proposal_id,'proposal'),stories)
                        else:
                            proposals = saved_batches[batch]['proposals']
                            if attempt >= len(proposals): break
                            proposal = proposals[attempt]
                        try:
                            if not isinstance(proposal,dict) or proposal.get('family') != family: raise Rejected('wrong_family')
                            item = gates[(rep,arm)].accept(proposal)
                        except Rejected as exc:
                            store.append('rejected',{'proposal_id':proposal_id,'batch':batch,'proposal':proposal,'reason':str(exc)})
                            store.append('proposal_events',{'proposal_id':proposal_id,'accepted':False})
                            attempted.add(proposal_id)
                            continue
                        slot = missing[0]
                        row = {'slot':slot,'replicate':rep,'arm':arm,'family':family,'index':int(slot.split('/')[-1]),'proposal_id':proposal_id,'item':item.to_dict()}
                        store.append('items',row)
                        saved_items[slot] = row
                        store.append('proposal_events',{'proposal_id':proposal_id,'accepted':True,'slot':slot})
                        attempted.add(proposal_id)
                        completed[slot] = evaluate_stored(client,store,item,slot,{'arm':arm,'replicate':rep,'family':family,'index':row['index']},config['seed'])
                    if not all(slot in completed for slot in slots):
                        store.append('events',{'kind':'proposal_cap_exhausted','batch':batch,'completed':sum(slot in completed for slot in slots),'requested':count})
                        raise RuntimeError('Proposal cap exhausted; run is incomplete. Do not silently replace adaptive proposals with random items.')
