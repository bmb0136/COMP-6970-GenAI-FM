import datasets
import json

def preproc(ingredients, title, directions, **kwargs):
    # Some recipes in the dataset have multiple directions on one line
    # Split them into multiple steps
    directions = [
        step.strip()
        for d in json.loads(directions)
        for step in d.split(".")
        if step.strip() # Exclude empty values
    ]
    return { "ingredients": json.loads(NER), "title": title, "directions": directions }

def format_prompt(ingredients, title, directions, **kwargs):
    prompt = "## Ingredients:\n" + "\n".join(f"- {i}" for i in ingredients)
    text = "## Title: " + title + "\n\n"  + "## Directions:\n" + "\n".join(f"- {d}" for d in directions)
    return { "prompt": prompt, "completion": text }

def load_recipe_nlg(num_examples, seed=42, as_prompts=False):
    raw_ds = datasets.load_dataset("innovate-data/RecipeNLG", data_files=["RecipeNLG_dataset.csv"], split="train").shuffle(seed=seed)

    ds = raw_ds \
        .filter(lambda row: all(row[key] is not None for key in ["NER", "directions", "title"])) \
        .select(range(num_examples)) \
        .map(lambda row: preproc(**row), remove_columns=raw_ds.column_names) \
    
    if as_prompts:
        ds = ds.map(lambda row: format_prompt(**row), remove_columns=ds.column_names)
        
    return ds.train_test_split(test_size=0.2)
