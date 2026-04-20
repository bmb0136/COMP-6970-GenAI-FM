from transformers import AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback
import torch
from trl import SFTTrainer
import pandas as pd
from data import load_recipe_nlg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {device}")

dataset = load_recipe_nlg(15_000, seed=67, as_prompts=True)

base_name = "distilgpt2"

args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    output_dir="sft_ckpt",
    eval_strategy="steps",
    learning_rate=1e-4,
    eval_steps=50,
    save_strategy="steps",
    save_steps=250,
    logging_steps=50,
    num_train_epochs=67, # We have early stopping enabled, so just run until that happens
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
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
