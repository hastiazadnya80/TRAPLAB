"""Run from a checkout without installing the package."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'src'))
from traplab.analysis import summarize
import json

if __name__ == '__main__':
    print(json.dumps(summarize(sys.argv[1]),indent=2))
