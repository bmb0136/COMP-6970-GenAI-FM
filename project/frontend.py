import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("./sft_model").to(device)

try:
    while True:
        ingredients = []
        while True:
            try:
                i = (input("Add ingredient [empty to finish]: ") or "").lower().strip()
            except EOFError as _:
                break
            if not i:
                break
            ingredients.append(i)

        if len(ingredients) == 0:
            break

        print("Generating recipe...")

        while True:
            prompt = f"## Ingredients:\n{"\n".join(f"- {x}" for x in ingredients)}"
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=512)
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(prompt):]
            print()
            print(text)
            print()
            
            try:
                yn = (input("Generate another? [y/n]: ") or "").lower().strip()
            except EOFError as _:
                break
            if yn != "y":
                break
except KeyboardInterrupt as _:
    pass

print("Goodbye")
