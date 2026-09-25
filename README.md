# Vigilant Cogwheel

An experimental local LoRA training workflow for Ministral-3B, with a Python CLI,
sample conversations, and dataset filters. The useful first step is checking your
data. A valid file is not proof that an adapter will improve the model.

> **Status:** CPU data checks and trainer contracts are tested. The revised GPU
> training path, exports, model quality, speed, and memory use have not been
> validated end to end. This is not a production training system.

## What it does

- Checks every conversation for supported roles and nonempty text.
- Keeps system, user, and assistant messages intact for the model's chat template.
- Reports simple text statistics; optionally filters examples over several passes.
- Connects Unsloth LoRA adapters to TRL's supervised fine-tuning trainer.

The optional "annealing" loop adjusts filtering thresholds. It does not train
a quality judge, prove dataset improvement, or validate an adapter.

## Check data without a GPU

Use Python 3.12 for the checked environment. Create and activate a virtual
environment, then run:

```sh
python -m pip install -r requirements-check.lock
python -m pip install --no-deps -e .
ministral-train --dataset demo_enterprise_expert.jsonl --validate-only
python -m pytest -q
```

These commands install the CPU check dependencies, not the optional GPU training
stack. Validation reads local JSONL and does not load a model. The dependency
file pins the resolved CPU environment; it is not a GPU environment lock.

Each JSONL line contains one conversation:

```json
{"messages":[{"role":"system","content":"Keep answers brief."},{"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello."}]}
```

Use `system`, `user`, and `assistant` roles with string content. Each conversation
must include a user and assistant message. The selected tokenizer may impose
additional ordering rules. Tool-call and multimodal records are not supported.
The included enterprise examples are sample material, not authoritative advice.

## Scoring limits

Length counts whitespace-separated words, not model tokens. "Coherence" measures
word overlap between turns. Diversity and repetition are word-count heuristics.
These can reject good examples and accept poor ones.

**There is no toxicity detector.** Toxicity is reported as `null`, with
`safety_checked: false`. The optional bias-term check only matches configured
strings; it is not a bias assessment. `quality_pass` means a sample met the
configured text thresholds, not that it is safe, correct, or suitable to publish.
The legacy `toxicity_threshold` setting has no effect.

Review your data before training. Keep personal records, secrets, customer data,
caches, and model outputs out of public commits. Training reports are disabled
(`report_to="none"`) and Hub uploads are off, but model installation and download
can still need network access. Check third-party tools and licenses before use.

## Experimental training path

Start in a separate environment. Follow the [official Unsloth installation
instructions](https://unsloth.ai/docs/get-started/install) for your GPU and OS.
Then install this project's versioned API target:

```sh
python -m pip install -e ".[train]"
python -m pip check
ministral-train --dataset demo_enterprise_expert.jsonl --config smoke-config.yaml
```

This targets Unsloth 2026.9.11, TRL 0.24.0, and Transformers 4.56.2. The versions
fit Unsloth's declared direct dependency ranges; that is not proof that the full
GPU stack works on your machine. Resolve driver/PyTorch compatibility using the
upstream instructions. Record the complete environment after a successful run.

The trainer passes conversational records to
[TRL's chat-template handling](https://huggingface.co/docs/trl/v0.24.0/sft_trainer).
It requires a tokenizer chat template. Defaults are batch size 1, LoRA rank 16,
and context length 2048. The smoke configuration requests five steps and disables
GGUF export. These are starting settings, not a memory-fit guarantee. Review
precision settings for your hardware. Merged weight saving still runs after training.

Before relying on a result, compare the base model and adapter on a held-out task
set. Record quality, peak VRAM, elapsed time, and export/inference checks. LoRA
adapts selected parameters; it does not make all model behavior correct.

## Evidence and remaining work

The September 2026 maintenance pass checked installation, local data validation,
and regression tests on CPU. Trainer and model tests use stand-ins at the GPU
library boundary. They check arguments and message preservation, not optimization
or generated answers. No model was downloaded or trained for this pass.

Earlier time, throughput, VRAM, and "$0" figures were not backed by reproducible
benchmarks here and have been removed. Running locally can avoid a cloud training
bill; hardware, electricity, and operator time still cost something.

The `desktop/` folder is an older experimental frontend. It was not validated in
this pass; the CLI is the documented entry point. GPU training, held-out quality,
and GGUF export are still required before a training-ready release claim.

## License

MIT covers this repository; see [LICENSE](LICENSE). Base models, datasets, and
dependencies have their own terms. Review those terms before training or sharing
an adapter or merged model.
