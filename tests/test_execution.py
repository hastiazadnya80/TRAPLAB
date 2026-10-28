import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from traplab.beam_client import BeamClient, DemoClient, APIError
from traplab.core import build
from traplab.evaluator import grade, solve, solver_messages
from traplab.sampling import smoke_proposal
from traplab.runner import smoke, run, seeded
from traplab.storage import read_jsonl
from traplab.analysis import summarize, wilson, bootstrap_mean_difference


class ExecutionTests(unittest.TestCase):
    def setUp(self): self.item = build(smoke_proposal())

    def response(self, content, finish='stop'):
        return {'content':content,'finish_reason':finish}

    def test_grading_statuses(self):
        self.assertEqual(grade(self.item,self.response('{"answer":"A_better","explanation":"yes"}'))['status'],'correct')
        self.assertEqual(grade(self.item,self.response('{"answer":"B_better","explanation":"no"}'))['status'],'wrong')
        for text in ('not JSON','```json\n{}\n```','{"answer":"unknown","explanation":"x"}','{"answer":"A_better","answer":"B_better","explanation":"x"}','{"answer":"A_better"}'):
            self.assertEqual(grade(self.item,self.response(text))['status'],'unparsed')
        self.assertEqual(grade(self.item,self.response('{','length'))['status'],'truncated')

    def test_length_retry_double_budget(self):
        client = DemoClient()
        with patch.object(client,'complete',side_effect=[self.response('{','length'),self.response('{"answer":"A_better","explanation":"x"}')]) as complete:
            calls = []
            result = solve(client,self.item,[],lambda r,i:calls.append((r,i)))
            self.assertEqual(result['result']['status'],'correct')
            self.assertEqual(complete.call_args_list[0].kwargs['max_tokens'],4096)
            self.assertEqual(complete.call_args_list[1].kwargs['max_tokens'],8192)
            self.assertEqual(len(calls),2)

    def test_demo_smoke_replay_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            client = DemoClient()
            smoke(client,directory,True)
            summary = summarize(directory)
            self.assertTrue(summary['complete'])
            self.assertTrue(summary['demo'])
            self.assertEqual(summary['trajectories'][0]['statuses'],{'wrong':1})
            smoke(client,directory,True)
            self.assertEqual(len(read_jsonl(Path(directory)/'calls.jsonl')),1)

    def test_transport_payload_and_metadata(self):
        import io
        response = {'choices':[{'message':{'content':'{}','reasoning_content':'test reasoning'},'finish_reason':'stop'}],'usage':{'total_tokens':20},'model':'beam-test'}
        with patch.dict('os.environ',{'BEAM_API_KEY':'TEST-ONLY'}):
            client = BeamClient('https://example.com/v1')
        with patch('traplab.beam_client.urlopen',return_value=io.BytesIO(json.dumps(response).encode())) as send:
            result = client.complete([{'role':'user','content':'test'}])
            payload = json.loads(send.call_args.args[0].data)
            self.assertEqual(payload['reasoning_effort'],'low')
            self.assertEqual(payload['max_completion_tokens'],4096)
            self.assertEqual(result['reasoning_content'],'test reasoning')
            self.assertNotIn('TEST-ONLY',json.dumps(result))

    def test_main_rounds_and_no_explanation_to_designer(self):
        config = {'seed':42,'replicates':2,'items_per_family_arm':4,'round_size':2,'max_proposals_per_round':40,'families':['simpson','collider','rtm']}
        with tempfile.TemporaryDirectory() as directory:
            client = DemoClient()
            run(client,config,directory,demo=True)
            summary = summarize(directory)
            self.assertEqual(summary['completed_items'],48)
            self.assertTrue(summary['complete'])
            calls_before = len(read_jsonl(Path(directory)/'calls.jsonl'))
            run(client,config,directory,demo=True)
            self.assertEqual(len(read_jsonl(Path(directory)/'calls.jsonl')),calls_before)
            config['seed'] = 43
            with self.assertRaises(ValueError): run(client,config,directory,demo=True)

    def test_design_context_has_no_explanations(self):
        from traplab.runner import designer_messages
        from traplab.core import DEFAULT_STORIES
        messages = designer_messages('simpson',4,DEFAULT_STORIES,[{'status':'wrong','params':{},'solver_answer':'B_better','tag':'paradox'}])
        data = json.loads(messages[-1]['content'])
        self.assertEqual(set(data['history'][0]),{'status','params','solver_answer','tag'})

    def test_replay_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            smoke(DemoClient(),directory,True)
            path = Path(directory)/'items.jsonl'
            rows = read_jsonl(path)
            rows[0]['item']['answer'] = 'B_better'
            path.write_text(json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'replay mismatch'): summarize(directory)

    def test_intervals(self):
        low,high = wilson(0,10)
        self.assertAlmostEqual(low,0)
        self.assertGreater(high,0)
        self.assertIsNone(bootstrap_mean_difference([(1,0)]))
        result = bootstrap_mean_difference([(1,0),(1,0)],repeats=100)
        self.assertEqual(result['ci95'],[1,1])


if __name__ == '__main__': unittest.main()
