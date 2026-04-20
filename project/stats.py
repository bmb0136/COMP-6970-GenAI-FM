import pandas as pd
import matplotlib.pyplot as plt
import json
import random

data = {
    k: pd.read_csv(f"eval_{k}.csv")
    for k in [
        "sft",
        "zero_shot",
        "few_shot"
    ]
}

with open("stats.txt", "w") as f:
    s = "Samples per model:\n"
    for k, df in data.items():
        s += f"- {k}: {len(df)}\n"

    s += "\nFormat Issues:\n"
    for k, df in data.items():
        s += f"- {k}: mean={df["num_issues"].mean()}, std={df["num_issues"].std()}, total={df["num_issues"].sum()} ({len(df[df["num_issues"] != 0]) / len(df):.2%})\n"

    s += "\nIngredient Coverage:\n"
    for k, df in data.items():
        s += f"- {k}: mean={df["coverage"].mean()}, std={df["coverage"].std()}\n"

    for k, df in data.items():
        row = df.iloc[random.randrange(len(df))]

        s += f"\n# Example Output for {k}\n"
        s += f"Ingredients: {row["ingredients"]}\n"
        s += "-" * 80
        s += "\n"
        s += row["output"].strip()
        s += "\n"
        s += "-" * 80
        s += "\n"

    f.write(s)
