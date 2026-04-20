import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sft_loss.csv")

train_loss = [tuple(map(float, x.values)) for x in data[["step", "loss"]].dropna().iloc]
eval_loss = [tuple(map(float, x.values)) for x in data[["step", "eval_loss"]].dropna().iloc]

fig, ax = plt.subplots()
ax.set_title("Loss vs. Training Steps")
ax.plot([x for x, _ in train_loss], [y for _, y in train_loss], label="Train Loss")
ax.plot([x for x, _ in eval_loss], [y for _, y in eval_loss], label="Eval Loss")
ax.legend()
ax.grid(0.1)
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
fig.savefig("loss.png")
