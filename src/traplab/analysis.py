"""Replay metrics from artifacts, never a second model call."""
import json
import math
import random
from collections import Counter, defaultdict
from .core import build
from .evaluator import grade
from .storage import read_jsonl


def wilson(k,n):
    if not n: return None
    z = 1.959963984540054
    p, den = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half), min(1,center+half)]


def bootstrap_mean_difference(pairs, seed=71, repeats=5000):
    if len(pairs) < 2: return None
    rng = random.Random(seed)
    differences = [a-b for a,b in pairs]
    values = sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(repeats))
    return {'estimate':sum(differences)/len(differences),'ci95':[values[int(.025*repeats)],values[min(repeats-1,int(.975*repeats))]],'unit':'paired independent trajectory','replicates':len(pairs)}


def summarize(path, verify=True):
    from pathlib import Path
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    stories = manifest['stories']
    items = {r['slot']:r['item'] for r in read_jsonl(path/'items.jsonl')}
    traces = read_jsonl(path/'traces.jsonl')
    calls = read_jsonl(path/'calls.jsonl')
    if len({r['slot'] for r in traces}) != len(traces): raise ValueError('Duplicate completed trace slots')
    groups = defaultdict(list)
    for trace in traces:
        raw = items[trace['slot']]
        item = build({'family':raw['family'],'params':raw['params'],'story_id':raw['story_id']}, stories)
        if verify:
            if item.to_dict() != raw: raise ValueError('Item replay mismatch: '+trace['slot'])
            if grade(item,trace['attempts'][-1]) != trace['result']: raise ValueError('Grade replay mismatch: '+trace['slot'])
            ledger = [c for c in calls if c['purpose']=='solver' and c.get('slot')==trace['slot']]
            if [c['response'] for c in ledger] != trace['attempts'] or any(c['messages'] != trace['messages'] for c in ledger):
                raise ValueError('Call ledger mismatch: '+trace['slot'])
        groups[(trace['family'],trace['arm'],trace['replicate'])].append((item,trace))
    trajectories = []
    for (family,arm,rep), rows in sorted(groups.items()):
        counts = Counter(t['result']['status'] for _,t in rows)
        failures = {i.cell for i,t in rows if t['result']['status'] == 'wrong'}
        slots = {t['slot'] for _,t in rows}
        solver_calls = sum(c['purpose']=='solver' and c.get('slot') in slots for c in calls)
        labels = Counter(i.answer for i,_ in rows)
        trajectories.append({'family':family,'arm':arm,'replicate':rep,'items':len(rows),'solver_calls_including_retries':solver_calls,'statuses':dict(counts),'answer_key_distribution':dict(labels),'majority_label_baseline':max(labels.values())/len(rows),'distinct_failure_cells':len(failures),'distinct_failures_per_item':len(failures)/len(rows),'distinct_failures_per_solver_call':len(failures)/solver_calls if solver_calls else None,'reasoning_errors_per_item':counts['wrong']/len(rows),'wilson_wrong_descriptive':wilson(counts['wrong'],len(rows))})
    comparisons = {}
    for family in manifest['config'].get('families',[]):
        target = manifest['config']['items_per_family_arm']
        by_key = {(r['arm'],r['replicate']):r for r in trajectories if r['family'] == family and r['items'] == target}
        pairs = [(by_key[('adaptive',rep)]['distinct_failures_per_solver_call'],by_key[('random',rep)]['distinct_failures_per_solver_call']) for rep in range(manifest['config']['replicates']) if ('adaptive',rep) in by_key and ('random',rep) in by_key]
        comparisons[family] = bootstrap_mean_difference(pairs)
    usage = Counter()
    calls_by_purpose = Counter()
    latency = 0.
    for call in calls:
        response = call['response']
        calls_by_purpose[call['purpose']] += 1
        latency += response.get('latency_s',0)
        for k in ('prompt_tokens','completion_tokens','total_tokens'):
            if isinstance(response.get('usage',{}).get(k),int): usage[k] += response['usage'][k]
    extensions = None
    if manifest['config'].get('kind') == 'extensions':
        from .experiments import analyze_extensions
        extensions = analyze_extensions(path,traces)
        expected = extensions['expected_items']
    elif manifest['config'].get('kind') == 'smoke': expected = 1
    else: expected = manifest['config']['replicates']*len(manifest['config']['families'])*2*manifest['config']['items_per_family_arm']
    return {'demo':manifest['demo'],'model':manifest['model'],'protocol':manifest['protocol'],'complete':extensions['complete'] if extensions else len(traces)==expected,'expected_items':expected,'completed_items':len(traces),'trajectories':trajectories,'adaptive_minus_random':comparisons,'extensions':extensions,'calls_by_purpose':dict(calls_by_purpose),'usage':dict(usage),'summed_latency_s':latency,'rejections':len(read_jsonl(path/'rejected.jsonl')),'notes':['DEMO DATA: no Beam evidence.' if manifest['demo'] else 'Observed API responses.','Wilson intervals are descriptive for adaptive trajectories; discovery comparison uses paired trajectory bootstrap.','Costs require provider prices; tokens are not a dollar cost estimate.','A crash while an HTTP request is in flight may incur unrecorded provider usage; exact-once remote billing is not guaranteed.']}
