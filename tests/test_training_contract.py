import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from datasets import Dataset
from ministral_finetuner.config import TrainingConfig

def test_model_import_and_context_length(monkeypatch):
    backend = Mock()
    backend.from_pretrained.return_value = (object(), object())
    monkeypatch.setitem(sys.modules, 'unsloth', SimpleNamespace(FastLanguageModel=backend))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=object))
    sys.modules.pop('ministral_finetuner.model', None)
    module = importlib.import_module('ministral_finetuner.model')
    module.MinistralModel('synthetic-model').load_model(max_seq_length=1024)
    assert backend.from_pretrained.call_args.kwargs['max_seq_length'] == 1024

def test_trainer_uses_conversations_and_disables_external_reporting(monkeypatch):
    # Strict keyword boundary from the documented TRL 0.24 SFTTrainer API.
    def sft_trainer(*, model, processing_class, train_dataset, args):
        return SimpleNamespace(dataset=train_dataset, args=args)
    monkeypatch.setitem(sys.modules, 'trl', SimpleNamespace(SFTTrainer=sft_trainer, SFTConfig=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(TrainingArguments=lambda **kw: SimpleNamespace(**kw), PreTrainedTokenizerBase=type('TokenizerBase', (), {})))
    sys.modules.pop('ministral_finetuner.trainer', None)
    module = importlib.import_module('ministral_finetuner.trainer')
    messages = [{'role':'system','content':'Be brief.'}, {'role':'user','content':'Hi'}, {'role':'assistant','content':'Hello'}]
    data = Dataset.from_list([{'messages': messages, 'text':'stale text must not override messages', 'quality_score':1.0}])
    config = TrainingConfig()
    tokenizer = SimpleNamespace(chat_template='test template')
    trainer = module.MinistralTrainer(object(), tokenizer, config).setup_trainer(data)
    assert trainer.dataset.column_names == ['messages']
    assert trainer.dataset[0]['messages'] == messages
    assert trainer.args.report_to == 'none'
    assert trainer.args.max_length == config.max_seq_length
