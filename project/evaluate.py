from transformers import AutoModelForCausalLM, AutoTokenizer
from data import load_recipe_nlg, format_prompt
from typing import NamedTuple
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch device: {device}")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

dataset = load_recipe_nlg(500, seed=420)["train"]

def generate(model, prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=512)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]

class ParsedResponse(NamedTuple):
    title: str
    directions: list[str]
    issues: list[str]

def parse(text: str) -> ParsedResponse:
    title = "N/A"
    directions = []
    issues = []

    state = "find_title"

    for i, line in enumerate(text.split("\n")):
        line = line.strip()
        if len(line) == 0:
            continue
        match state:
            case "find_title":
                if line.startswith("## Title:"):
                    title = line[line.index(":") + 1:].strip()
                    state = "find_directions"
                else:
                    issues.append(f"L{i + 1}: expected directions header")
            case "find_directions":
                if line.startswith("## Directions"):
                    state = "read_directions"
                else:
                    issues.append(f"L{i + 1}: expected directions header")
            case "read_directions":
                if line.startswith("- "):
                    directions.append(line[2:])
                elif len(directions) == 0:
                    issues.append(f"L{i + 1}: expected step")
                else:
                    state = "done"
            case "done":
                issues.append(f"L{i + 1}: extra line")

    if state not in ["done", "read_directions"]:
        issues.append(f"End: incomplete parse, stopped at {state}")

    return ParsedResponse(title, directions, issues)

def get_metrics(gen_func, example):
    output = gen_func(example)
    response = parse(output)
    found = {
        i: any(i.lower() in d.lower() for d in response.directions)
        for i in example["ingredients"]
    }
    return {
        "output": output,
        "parsed_title": response.title,
        "parsed_directions": json.dumps(response.directions),
        "issues": json.dumps(response.issues),
        "num_issues": len(response.issues),
        "coverage": sum(1 if x else 0 for x in found.values()) / len(found),
        "found": json.dumps(found),
    }

def gen_sft(model, example):
    input = f"## Ingredients:\n{"\n".join(f"- {i}" for i in example["ingredients"])}"
    return generate(model, input)

def gen_zero_shot(model, example):
    input = """
    You are a professional chef that creates recipes given a list of ingredients.
    Return the name of the created recipe and the list of directions in **exactly** the following markdown format:
    ```
    ## Title: recipe title here

    ## Directions:
    - step 1
    - step 2
    - etc.
    ```

    Create a recipe from the following ingredients:
    """
    input += f"\n\n## Ingredients:\n{"\n".join(f"- {i}" for i in example["ingredients"])}"
    return generate(model, input)

def gen_few_shot(model, example):
    input = f"""
    You are a professional chef that creates recipes given a list of ingredients.
    Return the name of the created recipe and the list of directions in **exactly** the following markdown format:
    ```
    ## Title: recipe title here

    ## Directions:
    - step 1
    - step 2
    - etc.
    ```

    Use the following examples as a reference.

    Example 1:
    ```
    ## Ingredients:
    - flour
    - butter
    - salt
    - water

    ## Title: Tortillas

    ## Directions:
    - Mix the salt & flour, rub in butter (like pie crust)
    - Knead in water
    - The dough should not be sticky
    - Roll thinly, but not too thin & slap onto a cast iron pan that is just beginning to smoke
    - Flip when it starts to bubble
    - We flip ours three times
    - Practice makes perfect with these lovelies, you'll never go back!
    ```

    Example 2:
    ```
    ## Ingredients:
    - orange
    - orange juice
    - raisins
    - egg
    - baking powder
    - salt
    - sugar
    - oil
    - flour

    ## Title: Orange Muffins

    ## Directions:
    - Cut orange in small pieces and place in blender and puree
    - Place orange juice, orange puree, egg, oil and raisins in bowl
    - Sift dry ingredients
    - Pour orange mixture over dry mixture and stir
    - Put in muffin pans
    - Bake at 400° for 15 to 20 minutes
    ```

    Create a recipe from the following ingredients:
    """
    input += f"\n\n## Ingredients:\n{"\n".join(f"- {i}" for i in example["ingredients"])}"
    return generate(model, input)

sft_model = AutoModelForCausalLM.from_pretrained("./sft_model").to(device)
base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").to(device)

print("Evaluating SFT...")
dataset.map(lambda example: get_metrics(lambda ex: gen_sft(sft_model, ex), example)).to_pandas().to_csv("eval_sft.csv")
print("Evaluating Zero-Shot...")
dataset.map(lambda example: get_metrics(lambda ex: gen_zero_shot(base_model, ex), example)).to_pandas().to_csv("eval_zero_shot.csv")
print("Evaluating Few-Shot...")
dataset.map(lambda example: get_metrics(lambda ex: gen_few_shot(base_model, ex), example)).to_pandas().to_csv("eval_few_shot.csv")
