"""Exercise the real read-only UI over loopback; no inference or web preview."""
from pathlib import Path
import os
import sys
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import httpx
from gradio_client import Client
from space.app import create_app


if __name__ == '__main__':
    app = create_app()
    try:
        app.launch(server_name='127.0.0.1',server_port=7860,prevent_thread_lock=True,quiet=True,share=False)
        with httpx.Client(trust_env=False) as http:
            response = http.get('http://127.0.0.1:7860/')
            response.raise_for_status()
            assert 'TRAPLAB' in response.text
            assert len(http.get('http://127.0.0.1:7860/config').json()['components'])>10
        client = Client('http://127.0.0.1:7860',verbose=False)
        output = client.predict('0/simpson/random/0',api_name='/inspect')
        assert 'Domain:' in output[0] and 'correct_answer' in output[1]
        print('PASS: HTTP page, component config and live item-browser callback')
    finally:
        app.close()
