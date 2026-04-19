import datasets
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding
)
import torch
import json
from trl import SFTTrainer
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {device}")

raw_ds = datasets.load_dataset("innovate-data/RecipeNLG", data_files=["RecipeNLG_dataset.csv"], split="train").shuffle(seed=67)

def preproc(NER, title, directions, **kwargs):
    return { "ingredients": json.loads(NER), "title": title, "directions": json.loads(directions) }

def format_prompt(ingredients, title, directions, **kwargs):
    prompt = "## Ingredients:\n" + "\n".join(f"- {i}" for i in ingredients)
    text = "## Title: " + title + "\n\n"  + "## Directions:\n" + "\n".join(f"- {d}" for d in directions)
    return { "prompt": prompt, "completion": text }

dataset = raw_ds \
    .select(range(1000)) \
    .filter(lambda row: all(row[key] is not None for key in ["NER", "directions", "title"])) \
    .map(lambda row: format_prompt(**preproc(**row)), remove_columns=raw_ds.column_names) \
    .train_test_split(test_size=0.2)

#base_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
base_name = "distilgpt2"

args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    output_dir="sft_ckpt",
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    num_train_epochs=100,
    logging_dir='sft_logs',
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    torch_empty_cache_steps=50,
    report_to="none"
)

trainer = SFTTrainer(
    args=args,
    model=base_name,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
)
result = trainer.train()
trainer.save_model("sft_model")
pd.DataFrame(trainer.state.log_history).to_csv("sft_loss.csv")
