import json
from pathlib import Path
import tempfile
import unittest
from traplab.analysis import summarize
from traplab.beam_client import DemoClient
from traplab.runner import run
from traplab.experiments import run_extensions, story_stat, disagreement
from traplab.storage import read_jsonl


class ExtensionTests(unittest.TestCase):
    def test_story_stat_separates_story_and_noise(self):
        self.assertGreater(story_stat([['A']*3,['B']*3,['C']*3,['D']*3,['E']*3]),0)
        self.assertEqual(story_stat([['A']*3]*5),0)
        self.assertAlmostEqual(disagreement(['A','A','B']),2/3)

    def test_extensions_replay_resume_and_frozen_stimuli(self):
        config = {'seed':42,'replicates':1,'items_per_family_arm':4,'round_size':2,'max_proposals_per_round':40,'families':['simpson','collider','rtm']}
        with tempfile.TemporaryDirectory() as directory:
            source,output = Path(directory)/'source',Path(directory)/'extensions'
            client = DemoClient()
            run(client,config,source,demo=True)
            run_extensions(client,source,output,demo=True,bases=2,self_limit=2,flip_configs=1)
            summary = summarize(output)
            self.assertTrue(summary['complete'])
            extensions = summary['extensions']
            self.assertEqual(extensions['self_correction']['wrong_baseline'],2)
            self.assertEqual(extensions['self_correction']['correct_baseline'],2)
            self.assertEqual(extensions['story_sensitivity']['included_bases'],2)
            self.assertEqual(extensions['story_sensitivity']['across_minus_within_disagreement'],0)
            self.assertEqual(extensions['story_sensitivity']['two_sided_permutation_p'],1)
            plan = json.loads((output/'extension-plan.json').read_text())
            self.assertEqual(len(plan['flip_grid']),11)
            self.assertTrue(any('rejected' in r for r in plan['flip_grid']))
            traces = read_jsonl(output/'traces.jsonl')
            noise = [t for t in traces if t['experiment']=='noise']
            parent = {t['slot']:t for t in read_jsonl(source/'traces.jsonl')}
            for trace in noise:
                self.assertEqual(trace['messages'],parent[trace['source_slot']]['messages'])
            before = len(read_jsonl(output/'calls.jsonl'))
            run_extensions(client,source,output,demo=True,bases=2,self_limit=2,flip_configs=1)
            self.assertEqual(len(read_jsonl(output/'calls.jsonl')),before)


if __name__ == '__main__': unittest.main()
