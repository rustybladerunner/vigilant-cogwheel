from typing import Dict, Any, List, Optional, TYPE_CHECKING
from collections import Counter
from datasets import Dataset, load_dataset

from ministral_finetuner.config import QualityThresholds

if TYPE_CHECKING:
    from ministral_finetuner.config import TrainingConfig

class DatasetLoader:
    @staticmethod
    def load_jsonl(path: str) -> Dataset:
        """Load local JSONL records containing role/content messages."""
        return load_dataset("json", data_files=path, split="train")

    @staticmethod
    def validate_format(dataset: Dataset) -> bool:
        """Check every conversation before scoring or loading a model."""
        if len(dataset) == 0:
            return False
        for sample in dataset:
            messages = sample.get("messages")
            if not isinstance(messages, list) or not messages:
                return False
            for msg in messages:
                if not isinstance(msg, dict) or msg.get("role") not in ("system", "user", "assistant"):
                    return False
                if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                    return False
            roles = {msg["role"] for msg in messages}
            if not {"user", "assistant"}.issubset(roles):
                return False
        return True


class QualityScorer:
    def __init__(self, thresholds: QualityThresholds, bias_terms: Optional[List[str]] = None, enable_bias: bool = True):
        self.thresholds = thresholds
        self.bias_terms = [term.lower() for term in bias_terms or []]
        self.enable_bias = enable_bias

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.split()

    def _coherence(self, turns: List[str]) -> float:
        if len(turns) < 2:
            return 1.0
        scores = []
        for prev, nxt in zip(turns[:-1], turns[1:]):
            a = set(self._tokenize(prev))
            b = set(self._tokenize(nxt))
            if not a or not b:
                continue
            overlap = len(a & b) / len(a | b)
            scores.append(overlap)
        return sum(scores) / len(scores) if scores else 0.0

    def _diversity(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _repetition(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        freq = Counter(tokens)
        most_common = freq.most_common(1)[0][1]
        return most_common / len(tokens)

    def _toxicity_stub(self, text: str) -> Optional[float]:
        """No toxicity detector is implemented; unknown is not a zero score."""
        return None

    def _bias_stub(self, text: str) -> Optional[float]:
        if not self.enable_bias or not self.bias_terms:
            return None
        lower_text = text.lower()
        hits = sum(1 for term in self.bias_terms if term in lower_text)
        return hits / max(len(self.bias_terms), 1)

    def _structural(self, messages: List[Dict[str, Any]]) -> Dict[str, float]:
        turn_count = len(messages)
        assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        assistant_ratio = assistant_turns / turn_count if turn_count else 0.0
        has_system = any(m.get("role") == "system" for m in messages)
        return {
            "turn_count": float(turn_count),
            "assistant_ratio": assistant_ratio,
            "user_turns": float(user_turns),
            "has_system": float(has_system),
        }

    def score(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_segments = [msg.get("content", "") for msg in messages]
        tokens = self._tokenize(" ".join(text_segments))
        user_tokens = self._tokenize(" ".join(msg.get("content", "") for msg in messages if msg.get("role") == "user"))
        assistant_tokens = self._tokenize(" ".join(msg.get("content", "") for msg in messages if msg.get("role") == "assistant"))

        coherence = self._coherence(text_segments)
        diversity = self._diversity(tokens)
        repetition = self._repetition(tokens)
        toxicity = self._toxicity_stub(" ".join(text_segments))
        bias = self._bias_stub(" ".join(text_segments))
        structural = self._structural(messages)

        user_ratio = len(user_tokens) / max(len(tokens), 1)
        length = len(tokens)
        structural_score = 1.0 - abs(structural["assistant_ratio"] - 0.5)

        quality_score = (
            0.25 * coherence
            + 0.2 * diversity
            + 0.15 * (1 - repetition)
            + 0.1 * structural_score
        ) / 0.7  # Text heuristics only; no unmeasured safety credit.

        flags = []
        t = self.thresholds
        if length < t.min_length or (t.max_length and length > t.max_length):
            flags.append("length")
        if not (t.min_user_ratio <= user_ratio <= t.max_user_ratio):
            flags.append("user_ratio")
        if coherence < t.min_coherence:
            flags.append("coherence")
        if diversity < t.min_diversity:
            flags.append("diversity")
        if repetition > t.max_repetition:
            flags.append("repetition")
        if bias is not None and bias > t.bias_threshold:
            flags.append("bias")

        passes = len(flags) == 0
        return {
            "quality_score": quality_score,
            "coherence": coherence,
            "diversity": diversity,
            "repetition": repetition,
            "toxicity": toxicity,
            "safety_checked": False,
            "bias": bias,
            "user_ratio": user_ratio,
            "length": float(length),
            "structural": structural,
            "passes": passes,
            "flags": flags,
        }

    def annotate_dataset(self, dataset: Dataset) -> Dataset:
        def _score(example: Dict[str, Any]) -> Dict[str, Any]:
            metrics = self.score(example["messages"])
            example.update({
                "quality_score": metrics["quality_score"],
                "quality_metrics": {
                    "coherence": metrics["coherence"],
                    "diversity": metrics["diversity"],
                    "repetition": metrics["repetition"],
                    "toxicity": metrics["toxicity"],
                    "bias": metrics["bias"],
                    "user_ratio": metrics["user_ratio"],
                    "length": metrics["length"],
                    **metrics["structural"],
                },
                "quality_pass": metrics["passes"],
                "safety_checked": False,
                "quality_flags": metrics["flags"],
            })
            return example
        return dataset.map(_score)

    def filter_dataset(self, dataset: Dataset) -> Dataset:
        annotated = self.annotate_dataset(dataset)
        return annotated.filter(lambda x: x["quality_pass"])


def load_dataset_with_quality(dataset_path: str, config: "TrainingConfig"):
    """Load, validate, annotate, and optionally anneal a dataset."""
    dataset = DatasetLoader.load_jsonl(dataset_path)
    if not DatasetLoader.validate_format(dataset):
        raise ValueError("Dataset is not in the expected chat message format.")

    anneal_cfg = config.annealing
    scorer = QualityScorer(
        anneal_cfg.quality_thresholds,
        bias_terms=anneal_cfg.bias_terms,
        enable_bias=anneal_cfg.bias_check_enabled,
    )
    dataset = scorer.annotate_dataset(dataset)

    annealing_history = []
    if anneal_cfg.enabled:
        from ministral_finetuner.annealing import AnnealingLoop

        loop = AnnealingLoop(
            anneal_cfg,
            anneal_cfg.quality_thresholds,
            bias_terms=anneal_cfg.bias_terms,
        )
        dataset, annealing_history = loop.run(dataset)

    if len(dataset) == 0:
        raise ValueError("No samples remain after dataset filtering.")
    # Keep messages intact. TRL applies the selected tokenizer's chat template.
    return dataset, annealing_history
