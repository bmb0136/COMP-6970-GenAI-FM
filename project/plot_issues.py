import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import Counter

data = {
    k: pd.read_csv(f"eval_{k}.csv")
    for k in [
        "sft",
        "zero_shot",
        "few_shot"
    ]
}

fig, ax = plt.subplots(nrows=len(data), figsize=(4.8, 4.8 * len(data)))
for i, (k, df) in enumerate(data.items()):
    issues = Counter([
        i[i.index(":") + 1:].strip()
        for row in df.iloc
        for i in json.loads(row["issues"])
    ])
    keys = list(issues.keys())
    values = [issues[k] for k in keys]
    ax[i].set_title(f"Issue Distribution ({k})")
    ax[i].pie(values, autopct="%.1f%%", radius=0.75, textprops=dict(color="w"))
    ax[i].legend(keys, loc="lower center", ncols=2)

fig.tight_layout()
fig.savefig("issues.png")
