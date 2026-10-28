"""Stored-result extensions; all selections are seeded and auditable."""
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
from .core import build, digest, neutral_tools, Rejected
from .evaluator import solver_messages
from .runner import evaluate_stored, seeded
from .storage import Store, read_jsonl


def run_extensions(client, source, output, demo=False, seed=20261101, bases=20, self_limit=60, flip_configs=3):
    source = Path(source)
    from .analysis import summarize
    parent_summary = summarize(source)
    if not parent_summary['complete']: raise ValueError('Extensions require a complete source run')
    parent = json.loads((source/'manifest.json').read_text())
    if parent['demo'] != demo: raise ValueError('Cannot mix demo and live evidence')
    if parent['model'] != client.model: raise ValueError('Extensions must use the same requested model')
    stories = parent['stories']
    if len(stories) != 5: raise ValueError('Story sensitivity requires exactly five shared stories')
    if bases < 1 or self_limit < 1 or not 1 <= flip_configs <= 3: raise ValueError('Invalid extension sizes')
    originals = {r['slot']:r['item'] for r in read_jsonl(source/'items.jsonl')}
    traces = read_jsonl(source/'traces.jsonl')
    unique = {}
    for row in traces:
        raw = originals[row['slot']]
        unique.setdefault(digest([raw['family'],raw['params']]),row)
    candidates = list(unique.values())
    seeded(seed,'story-bases').shuffle(candidates)
    if len(candidates) < bases: raise ValueError('Not enough distinct source items for requested story bases')
    selected_bases = candidates[:bases]
    wrong = [t for t in traces if t['result']['status'] == 'wrong']
    correct = [t for t in traces if t['result']['status'] == 'correct']
    seeded(seed,'wrong').shuffle(wrong)
    seeded(seed,'correct').shuffle(correct)
    matched = min(self_limit,len(wrong),len(correct))
    self_items = wrong[:matched]+correct[:matched]
    repeated = traces.copy()
    seeded(seed,'noise').shuffle(repeated)
    repeated = repeated[:max(1,round(len(traces)*.1))]
    config = {'kind':'extensions','source_identity':parent['identity'],'source_data_hash':digest({'items':originals,'traces':traces}),'seed':seed,'bases':bases,'self_limit':self_limit,'flip_configs':flip_configs,'self_items':[t['slot'] for t in self_items],'noise_items':[t['slot'] for t in repeated],'story_bases':[t['slot'] for t in selected_bases]}
    store = Store(output)
    store.manifest(config,stories,client.model,demo)
    done = {r['slot'] for r in read_jsonl(store.path/'traces.jsonl')}
    saved = {r['slot'] for r in read_jsonl(store.path/'items.jsonl')}
    # Plan is generated and persisted before spending any tokens.
    tasks = []
    def add(slot, item, metadata, messages):
        tasks.append({'slot':slot,'item':item.to_dict(),'metadata':metadata,'messages':messages})
    for trace in self_items:
        raw = originals[trace['slot']]
        item = build({'family':raw['family'],'params':raw['params'],'story_id':raw['story_id']},stories)
        messages = trace['messages']+[{'role':'assistant','content':trace['attempts'][-1]['content']},{'role':'user','content':'Numerical summaries computed only from the displayed table: '+json.dumps(neutral_tools(item))+'. Reconsider your answer. Return the same JSON schema.'}]
        add('self/'+trace['slot'],item,{'experiment':'self_correction','source_slot':trace['slot'],'baseline_status':trace['result']['status']},messages)
    for trace in repeated:
        raw = originals[trace['slot']]
        item = build({'family':raw['family'],'params':raw['params'],'story_id':raw['story_id']},stories)
        for repeat in (1,2):
            add(f'noise/{trace["slot"]}/{repeat}',item,{'experiment':'noise','source_slot':trace['slot'],'repeat':repeat,'baseline_answer':trace['result']['answer'],'baseline_status':trace['result']['status']},trace['messages'])
    for b,trace in enumerate(selected_bases):
        raw = originals[trace['slot']]
        for story in stories:
            item = build({'family':raw['family'],'params':raw['params'],'story_id':story['id']},stories)
            for repeat in range(3):
                messages = solver_messages(item,seeded(seed,b,'story-options'))
                add(f'story/{b}/{story["id"]}/{repeat}',item,{'experiment':'story','base':b,'story':story['id'],'repeat':repeat,'source_slot':trace['slot']},messages)
    flip_plan = []
    for configuration in range(flip_configs):
        b = [.25,.35,.45][configuration]
        root = round(2*b-.2,6)
        for point in range(11):
            x = round(.05+.09*point,4)
            proposal = {'family':'simpson','story_id':stories[0]['id'],'params':{'n_a0':200,'n_a1':200,'n_b0':200,'n_b1':200,'p_a0':x,'p_a1':.2,'p_b0':b,'p_b1':b,'z_role':'confounder'}}
            try: item = build(proposal,stories)
            except Rejected as exc:
                flip_plan.append({'configuration':configuration,'point':point,'x':x,'analytic_root':root,'rejected':str(exc)})
                continue
            flip_plan.append({'configuration':configuration,'point':point,'x':x,'analytic_root':root,'answer':item.answer})
            for repeat in range(3):
                add(f'flip/{configuration}/{point}/{repeat}',item,{'experiment':'flip','configuration':configuration,'point':point,'x':x,'analytic_root':root,'repeat':repeat},solver_messages(item,seeded(seed,configuration,'flip-options')))
    plan_path = store.path/'extension-plan.json'
    plan = {'tasks':tasks,'flip_grid':flip_plan}
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan: raise ValueError('Extension plan changed')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    for task in tasks:
        slot,raw = task['slot'],task['item']
        if slot in done: continue
        item = build({'family':raw['family'],'params':raw['params'],'story_id':raw['story_id']},stories)
        if slot not in saved:
            store.append('items',{'slot':slot,'item':raw,**task['metadata']})
        evaluate_stored(client,store,item,slot,{'arm':'supplement','replicate':0,'family':item.family,**task['metadata']},seed,task['messages'])


def disagreement(labels):
    if len(labels)<2: return None
    return sum(a!=b for a,b in itertools.combinations(labels,2))/(len(labels)*(len(labels)-1)/2)


def story_stat(groups):
    within = [a!=b for labels in groups for a,b in itertools.combinations(labels,2)]
    between = [a!=b for i,g in enumerate(groups) for h in groups[i+1:] for a in g for b in h]
    return sum(between)/len(between)-sum(within)/len(within)


def analyze_extensions(path, traces):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    plan = json.loads((path/'extension-plan.json').read_text())
    completed = {t['slot'] for t in traces}
    planned = {t['slot'] for t in plan['tasks']}
    if completed-planned: raise ValueError('Unplanned extension trace')
    self_rows = [t for t in traces if t['experiment']=='self_correction']
    wrong = [t for t in self_rows if t['baseline_status']=='wrong']
    correct = [t for t in self_rows if t['baseline_status']=='correct']
    from .analysis import wilson
    fixed = sum(t['result']['status']=='correct' for t in wrong)
    broken = sum(t['result']['status']=='wrong' for t in correct)
    valid_statuses = ('correct','wrong')
    story_rows, noise_rows, flip_rows = defaultdict(list),defaultdict(list),defaultdict(list)
    for trace in traces:
        if trace['experiment']=='story': story_rows[(trace['base'],trace['story'])].append(trace)
        if trace['experiment']=='noise': noise_rows[trace['source_slot']].append(trace)
        if trace['experiment']=='flip': flip_rows[(trace['configuration'],trace['point'])].append(trace)
    base_groups = []
    for b in range(manifest['config']['bases']):
        groups = [story_rows[(b,s['id'])] for s in manifest['stories']]
        if all(len(g)==3 and all(t['result']['status'] in valid_statuses for t in g) for g in groups):
            base_groups.append([[t['result']['answer'] for t in g] for g in groups])
    observed = sum(story_stat(g) for g in base_groups)/len(base_groups) if base_groups else None
    pvalue = None
    if base_groups:
        rng = seeded(manifest['config']['seed'],'permutation')
        extreme = 0
        for _ in range(2000):
            stats = []
            for groups in base_groups:
                flat = [a for g in groups for a in g]
                rng.shuffle(flat)
                stats.append(story_stat([flat[i:i+3] for i in range(0,15,3)]))
            extreme += abs(sum(stats)/len(stats)) >= abs(observed)-1e-12
        pvalue = (extreme+1)/2001
    flip_results = []
    for c in range(manifest['config']['flip_configs']):
        grid = []
        for row in (r for r in plan['flip_grid'] if r['configuration']==c):
            responses = flip_rows[(c,row['point'])]
            labels = [t['result']['answer'] for t in responses if t['result']['status'] in valid_statuses]
            counts = Counter(labels)
            majority = counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1]>=2 else None
            grid.append({**row,'majority':majority,'valid_responses':len(labels)})
        usable = [g for g in grid if 'rejected' not in g]
        binary = [g for g in usable if g['majority'] in ('A_better','B_better')]
        crossings = [(a,b) for a,b in zip(binary,binary[1:]) if a['majority']!=b['majority']]
        status = 'incomplete_or_unresolved' if any(g['majority'] is None or g['majority']=='insufficient_info' for g in usable) else 'unstable' if len(crossings)>1 else 'no_flip' if not crossings else 'single_flip'
        result = {'configuration':c,'analytic_root':grid[0]['analytic_root'],'status':status,'grid':grid,'crossings':len(crossings)}
        if status=='single_flip':
            a,b = crossings[0]
            lo,hi,root = a['x'],b['x'],result['analytic_root']
            result.update(observed_flip_bracket=[lo,hi],discrepancy_interval=[max(lo-root,root-hi,0),max(abs(lo-root),abs(hi-root))],direction='increasing' if a['majority']=='B_better' else 'decreasing')
        flip_results.append(result)
    noise = []
    for slot,rows in noise_rows.items():
        labels = [rows[0]['baseline_answer']]+[t['result']['answer'] for t in rows]
        valid = len(rows)==2 and rows[0]['baseline_status'] in valid_statuses and all(t['result']['status'] in valid_statuses for t in rows)
        noise.append({'source_slot':slot,'disagreement':disagreement(labels) if valid else None})
    story_table = [{'base':b,'story':s,'valid':sum(t['result']['status'] in valid_statuses for t in rows),'correct':sum(t['result']['status']=='correct' for t in rows),'responses':len(rows)} for (b,s),rows in sorted(story_rows.items())]
    return {'complete':completed==planned,'expected_items':len(planned),'completed_items':len(completed),'self_correction':{'wrong_baseline':len(wrong),'correct_baseline':len(correct),'fixed':fixed,'fix_rate':fixed/len(wrong) if wrong else None,'fix_wilson':wilson(fixed,len(wrong)),'broken':broken,'break_rate':broken/len(correct) if correct else None,'break_wilson':wilson(broken,len(correct)),'operational_failures':sum(t['result']['status'] not in valid_statuses for t in self_rows),'interpretation':'Descriptive revision after numerical feedback; no no-tool control.'},'story_sensitivity':{'included_bases':len(base_groups),'excluded_or_incomplete_bases':manifest['config']['bases']-len(base_groups),'across_minus_within_disagreement':observed,'two_sided_permutation_p':pvalue,'permutations':2000,'unit':'base item; labels permuted within base preserving three repeats per story','per_story':story_table},'noise':noise,'flip_points':flip_results}
