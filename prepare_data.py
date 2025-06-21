from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)
print("Downloading databricks/databricks-dolly-15k...")

dataset = load_dataset("databricks/databricks-dolly-15k")
print(f"✅ Dataset loaded! Total examples: {len(dataset['train'])}")

# Create subset with only 1000 examples
SUBSET_SIZE = 1000
subset = dataset["train"].select(range(SUBSET_SIZE))

with open("data/training_data.jsonl", "w") as f:
    for i, item in enumerate(subset):
        context = item.get("context", "") or ""
        instruction = item["instruction"]
        
        if context.strip():
            full_instruction = f"{instruction}\n\nContext: {context}"
        else:
            full_instruction = instruction
            
        json.dump({
            "instruction": full_instruction,
            "input": "",
            "output": item["response"]
        }, f)
        f.write("\n")
        
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1} examples...")

print(f"✅ Created training subset with {SUBSET_SIZE} examples!")