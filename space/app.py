"""Read-only artifact viewer; no model credentials or inference in this process."""
from pathlib import Path
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import gradio as gr
import plotly.graph_objects as go
from traplab.analysis import summarize
from traplab.storage import read_jsonl


def load_results(path):
    path = Path(path)
    summary = summarize(path)
    items = {r['slot']:r['item'] for r in read_jsonl(path/'items.jsonl')}
    traces = read_jsonl(path/'traces.jsonl')
    return summary, items, traces


def plots(items, traces):
    families = ['simpson','collider','rtm']
    arms = sorted({t['arm'] for t in traces})
    # Family/tag view avoids claiming distance between arbitrary bin IDs.
    categories = sorted({i['family']+' / '+i['tag'] for i in items.values()})
    errors, totals = Counter(), Counter()
    groups = defaultdict(list)
    for trace in traces:
        item = items[trace['slot']]
        key = (trace['arm'],item['family']+' / '+item['tag'])
        totals[key] += 1
        errors[key] += trace['result']['status'] == 'wrong'
        groups[(trace['family'],trace['arm'],trace['replicate'])].append(trace)
    atlas = go.Figure(go.Heatmap(x=categories,y=arms,z=[[errors[(arm,c)]/totals[(arm,c)] if totals[(arm,c)] else None for c in categories] for arm in arms],zmin=0,zmax=1,colorscale='OrRd',colorbar={'title':'Wrong / all items'}))
    atlas.update_layout(title='Failure Atlas — family and causal role',xaxis_title='Family / role',yaxis_title='Arm')
    cumulative = go.Figure()
    for (family,arm,rep),rows in sorted(groups.items()):
        seen, y = set(), []
        for trace in sorted(rows,key=lambda t:t.get('index',0)):
            if trace['result']['status'] == 'wrong': seen.add(items[trace['slot']]['cell'])
            y.append(len(seen))
        cumulative.add_trace(go.Scatter(x=list(range(1,len(y)+1)),y=y,mode='lines+markers',name=f'{family} / {arm} / run {rep}'))
    cumulative.update_layout(title='Distinct failed parameter cells',xaxis_title='Evaluated items (retries reported separately)',yaxis_title='Cumulative distinct failures')
    return atlas,cumulative


def create_app(results_path=None):
    path = Path(results_path or os.environ.get('TRAPLAB_RESULTS',ROOT/'demo-results'))
    summary,items,traces = load_results(path)
    by_slot = {t['slot']:t for t in traces}
    slots = list(by_slot)
    atlas,cumulative = plots(items,traces)
    extension_path = Path(os.environ.get('TRAPLAB_EXTENSIONS',ROOT/'demo-extensions'))
    extension_summary = summarize(extension_path) if (extension_path/'manifest.json').exists() else None
    extension = extension_summary['extensions'] if extension_summary else None
    if extension_summary and extension_summary['demo'] != summary['demo']:
        raise ValueError('Cannot mix demo extensions with live main results; set TRAPLAB_EXTENSIONS to matching results.')
    if extension_summary:
        parent_manifest = json.loads((path/'manifest.json').read_text())
        extension_manifest = json.loads((extension_path/'manifest.json').read_text())
        if extension_manifest['config']['source_identity'] != parent_manifest['identity']:
            raise ValueError('Supplementary results belong to a different main run.')
    def inspect(slot):
        item,trace = items[slot],by_slot[slot]
        return item['prompt'],json.dumps({'options':item['options'],'correct_answer':item['answer'],'status':trace['result']['status'],'solver_answer':trace['result']['answer'],'explanation':trace['result']['explanation'],'visible':item['visible'],'diagnostics':item['diagnostics'],'parameter_cell':item['cell']},indent=2)
    with gr.Blocks(title='TRAPLAB') as app:
        gr.Markdown('# TRAPLAB\nAn adversarial statistical reasoning lab. Beam proposes; Python determines the answer.')
        gr.Markdown('**DEMO — synthetic fixture responses. These are not Beam results and provide no evidence of adaptive gains.**' if summary['demo'] else '**Stored model results.** Recomputed locally from raw artifacts; no live inference.')
        gr.Markdown(f"Completed items: **{summary['completed_items']} / {summary['expected_items']}** · Model: `{summary['model']}` · Protocol: `{summary['protocol']}`")
        with gr.Tab('Failure Atlas'):
            gr.Plot(atlas)
            gr.Plot(cumulative)
        with gr.Tab('Item browser'):
            selected = gr.Dropdown(slots,value=slots[0] if slots else None,label='Recorded item')
            question = gr.Textbox(label='Question shown to solver',lines=14)
            details = gr.Code(label='Answer key and stored response',language='json')
            selected.change(inspect,selected,[question,details])
            if slots: app.load(lambda:inspect(slots[0]),outputs=[question,details])
        with gr.Tab('Story sensitivity'):
            if extension:
                gr.Markdown('Across-story disagreement is compared with within-story output noise. Permutations preserve each base item and group size. Operationally invalid bases are excluded and counted.')
                gr.JSON({k:v for k,v in extension['story_sensitivity'].items() if k!='per_story'})
                table = extension['story_sensitivity']['per_story']
                gr.Dataframe(value=[[r[k] for k in ('base','story','correct','valid','responses')] for r in table],headers=['Base','Story','Correct','Valid','Responses'],interactive=False)
            else: gr.Markdown('No supplementary results loaded.')
        with gr.Tab('Flip points'):
            if extension:
                figure = go.Figure()
                for row in extension['flip_points']:
                    usable = [p for p in row['grid'] if 'rejected' not in p]
                    figure.add_trace(go.Scatter(x=[p['x'] for p in usable],y=[{'A_better':1,'B_better':-1,'no_difference':0}.get(p['majority']) for p in usable],mode='lines+markers',name=f"Config {row['configuration']}: {row['status']}"))
                    figure.add_vline(x=row['analytic_root'],line_dash='dot')
                figure.update_layout(title='Majority answer over fixed parameter grid',xaxis_title='p(A success | Z=0)',yaxis={'title':'Majority answer','tickvals':[-1,0,1],'ticktext':['B better','No difference','A better']})
                gr.Plot(figure)
                gr.JSON(extension['flip_points'])
            else: gr.Markdown('No supplementary results loaded.')
        with gr.Tab('Revision and noise'):
            if extension:
                gr.JSON(extension['self_correction'],label='Revision after numerical feedback')
                gr.JSON(extension['noise'],label='Three-observation disagreement')
            else: gr.Markdown('No supplementary results loaded.')
        with gr.Tab('Audit'):
            gr.JSON(summary,label='Recomputed summary')
            if extension_summary: gr.JSON(extension_summary,label='Supplementary summary')
            gr.Markdown('Intervals for adaptive item rates are descriptive. Arm differences use paired independent-trajectory bootstrap. Unparsed and truncated outputs are not reasoning errors. Supplementary experiments use separate recorded budgets.')
    return app


if __name__ == '__main__':
    create_app().launch(server_name='0.0.0.0',server_port=int(os.environ.get('PORT','7860')),share=False)
