import json
import pytest
from datasets import Dataset
from ministral_finetuner.config import TrainingConfig, QualityThresholds
from ministral_finetuner.dataset import DatasetLoader, QualityScorer, load_dataset_with_quality

CHAT = [{"role": "system", "content": "Keep the answer brief."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."}]

def test_training_data_keeps_messages_and_system_instruction(tmp_path):
    file = tmp_path / 'sample.jsonl'
    file.write_text(json.dumps({'messages': CHAT})+'\n', encoding='utf-8')
    dataset, _ = load_dataset_with_quality(str(file), TrainingConfig())
    assert dataset[0]['messages'] == CHAT
    assert 'text' not in dataset.column_names, 'Do not bypass the model chat template with hand-written role labels'

@pytest.mark.parametrize('rows', [[], [{'messages': CHAT}, {'messages': [{'role': 'tool', 'content': 'bad'}]}],
    [{'messages': []}], [{'messages': [{'role': 'user', 'content': ''}]}]])
def test_validate_checks_every_record_and_rejects_empty_data(rows):
    assert DatasetLoader.validate_format(Dataset.from_list(rows)) is False

def test_toxicity_is_unknown_and_not_a_safety_pass():
    result = QualityScorer(QualityThresholds()).score(CHAT)
    assert result['toxicity'] is None
    assert result['safety_checked'] is False

def test_filtered_empty_dataset_stops_before_model_load(tmp_path):
    file = tmp_path / 'sample.jsonl'
    file.write_text(json.dumps({'messages': CHAT})+'\n', encoding='utf-8')
    config = TrainingConfig()
    config.annealing.enabled = True
    config.annealing.quality_thresholds.min_length = 10000
    with pytest.raises(ValueError, match='No samples'):
        load_dataset_with_quality(str(file), config)
