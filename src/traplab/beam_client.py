"""OpenAI-compatible transport. Never stores headers or credentials."""
from __future__ import annotations
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    pass


class BeamClient:
    def __init__(self, base_url=None, model=None, timeout=180, max_tokens=4096):
        self.base_url = (base_url or os.environ.get('BEAM_BASE_URL', '')).rstrip('/')
        self.model = model or os.environ.get('BEAM_MODEL', 'reflection/Beam-1.0-501B-A23B')
        self.key = os.environ.get('BEAM_API_KEY')
        parsed = urlsplit(self.base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise APIError('Set BEAM_BASE_URL to your provider HTTPS API base URL, without credentials or query parameters.')
        if not self.key:
            raise APIError('BEAM_API_KEY is missing. Set it securely in environment settings; never paste it into chat.')
        if max_tokens < 4096:
            raise APIError('Low-effort calls require max_completion_tokens >=4096.')
        self.timeout, self.max_tokens = timeout, max_tokens

    def complete(self, messages, max_tokens=None):
        payload = {'model':self.model, 'messages':messages, 'reasoning_effort':'low', 'max_completion_tokens':max_tokens or self.max_tokens}
        request = Request(self.base_url+'/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        start = time.monotonic()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
        except HTTPError as exc:
            # Error response bodies may echo credentials. Do not persist them.
            raise APIError(f'Beam HTTP {exc.code}. Check endpoint, model, credential scope, network policy and supported request fields.') from None
        except (URLError, TimeoutError, OSError):
            raise APIError('Beam connection failed or timed out. Check configured proxy, trust and endpoint access.') from None
        latency = time.monotonic()-start
        try:
            choice = raw['choices'][0]
            msg = choice['message']
            if not isinstance(msg.get('content'), (str,type(None))): raise ValueError()
            return {'content':msg.get('content') or '', 'reasoning_content':msg.get('reasoning_content'), 'finish_reason':choice.get('finish_reason'), 'usage':raw.get('usage', {}), 'latency_s':latency, 'model':raw.get('model', self.model), 'response_id':raw.get('id'), 'max_completion_tokens':payload['max_completion_tokens']}
        except (KeyError, IndexError, TypeError, ValueError):
            raise APIError('Provider returned an unsupported completion envelope.') from None


class DemoClient:
    """Deliberately fallible fixture. Never represents Beam evidence."""
    model = 'DEMO-FIXTURE-NOT-BEAM'
    max_tokens = 4096

    def complete(self, messages, max_tokens=None):
        text = messages[-1]['content']
        if 'expected successes=' in text:
            answer = 'B_better'  # deliberately wrong on the smoke reversal
        elif 'selected sample' in text:
            answer = 'positive'
        else:
            answer = 'improved'
        return {'content':json.dumps({'answer':answer,'explanation':'Synthetic fixture response; not model evidence.'}), 'reasoning_content':None, 'finish_reason':'stop','usage':{},'latency_s':0.,'model':self.model,'max_completion_tokens':max_tokens or self.max_tokens}
