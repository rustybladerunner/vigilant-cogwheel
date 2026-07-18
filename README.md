# Ministral LoRA Fine-Tuner (local)

**Local LoRA fine-tune for Ministral-3B via Unsloth** -- CLI, sample data, and tests. Laptop GPU, $0 cloud.

| | |
|---|---|
| Example GPU | NVIDIA RTX 5000 Ada Laptop (16 GB VRAM) |
| Base model | `mistralai/Ministral-3B-Instruct-2410` |
| Cost | $0 -- runs locally |
| Typical train time | 15-90 minutes (dataset-dependent) |

This repo packages a copy-pasteable Unsloth + LoRA workflow plus a small CLI (`ministral-train`) with optional dataset annealing/quality filters.

## Why this combo

| Feature | On a 16 GB laptop |
|---|---|
| ~6-8 GB VRAM (4-bit) | Plenty of headroom |
| Native long context + tool-calling | No extra training needed |
| Unsloth LoRA | 200-2000 examples -> ~15-60 minutes |
| Inference after fine-tune | Often 120-180 tok/s on Ada 16 GB |

## Install

```bash
pip install -e ".[dev]"
# Or follow the Quick Start torch/unsloth pins below if you prefer a bare venv.
```

## Quick Start (copy-paste)

### 1. Install dependencies (once)

```bash
# Windows / Linux / WSL2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes datasets
```

### 2. Prepare your dataset

Save as `my_dataset.jsonl` (ShareGPT-style messages):

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help you today?"}]}
{"messages": [{"role": "user", "content": "Write a poem"}, {"role": "assistant", "content": "Roses are red..."}]}
```

Even 300-500 high-quality lines is enough for useful adapters.

### 3. Train via CLI (preferred)

```bash
python -m cli.main --dataset demo_enterprise_expert.jsonl --enable-annealing \
  --anneal-cycles 3 --anneal-initial-temp 1.3 --anneal-min-temp 0.3 \
  --quality-min-length 32 --quality-min-coherence 0.4 --quality-min-diversity 0.3
```

### 4. Or train with a standalone script

Save as `train.py` and run `python train.py`:

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    "mistralai/Ministral-3B-Instruct-2410",
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset("json", data_files="my_dataset.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=32768,
    args=TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=10,
        output_dir="ministral-3b-finetuned",
        optim="adamw_8bit",
        seed=3407,
    ),
)

print("Starting training... (15-60 minutes)")
trainer.train()
model.save_pretrained_merged("ministral-3b-final", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("ministral-3b-gguf", tokenizer, quantization_method="q5_k_m")
print("Done! Your model is ready.")
```

### 5. Run your model

**Ollama:** `ollama create myministral -f Modelfile`  
**LM Studio / GPT4All / Msty:** load the GGUF  
**Python:**

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("ministral-3b-final")
FastLanguageModel.for_inference(model)
inputs = tokenizer("Hello! Write a story about...", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Demo dataset

- `demo_enterprise_expert.jsonl` -- ShareGPT-style enterprise guidance examples (Python, ops, networking).
- One JSON object per line; `messages: [{role, content}]` with `system` / `user` / `assistant`.

## Using Mistral-7B

Default base is Ministral-3B. For 7B:

```bash
python -m cli.main --dataset demo_enterprise_expert.jsonl \
  --model-name mistralai/Mistral-7B-Instruct-v0.3 \
  --output-dir mistral-7b-finetuned \
  --enable-annealing --anneal-cycles 3 --anneal-initial-temp 1.3 --anneal-min-temp 0.3
```

On 16 GB: stick to 4-bit + LoRA. Full FP16 fine-tune is not recommended.

## LoRA vs full fine-tune

- **LoRA (this repo):** small adapters, frozen base, fits 16 GB with 4-bit, fast to swap.
- **Full fine-tune:** updates all weights; usually needs >24 GB for 7B FP16 training.

## License

MIT -- see [LICENSE](LICENSE). Your fine-tuned adapters/weights remain yours subject to the base model license.
