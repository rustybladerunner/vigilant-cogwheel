import os
from pathlib import Path
import subprocess
import sys

def test_validate_only_works_without_gpu_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH':str(root/'src'), 'HF_HOME':str(tmp_path/'hf'), 'HF_HUB_OFFLINE':'1'}
    result = subprocess.run([sys.executable, '-m', 'cli.main', '--dataset', str(root/'demo_enterprise_expert.jsonl'), '--validate-only'], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'No model was loaded or trained.' in result.stdout
    assert 'Toxicity is not assessed' in result.stdout
