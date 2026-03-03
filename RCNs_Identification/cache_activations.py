import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformer_lens import HookedTransformer

MODEL = "Qwen3-1.7B"
dataset = "AIME"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=dataset)
parser.add_argument("--MODEL", type=str, default=MODEL)
args = parser.parse_args()
MODEL = args.MODEL
dataset = args.dataset

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / dataset / f"{dataset}_results_Qwen3-32B_filtered.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "activation_cache" / f"{dataset}_{MODEL}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading model...")
model = HookedTransformer.from_pretrained(
    model_name = MODEL,
    device=DEVICE,
    trust_remote_code=True,
)
model.eval()

print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]
print(f"Loaded {len(raw_data)} samples.")

print("Starting activation caching process...")
for sample in tqdm(raw_data, desc="Processing samples"):
    sample_id = sample["id"]
    attempt_index = sample["_id"]
    question = sample["question"]
    thinking = sample["thinking"]
    is_good = sample["is_good"]

    output_file = OUTPUT_DIR / f"sample_{sample_id}_attempt{attempt_index}.pt"

    prompt = f"Question: {question}\nThinking process:{thinking}"

    with torch.no_grad(): 
        _, cache = model.run_with_cache(
            prompt,
            names_filter=lambda name: "mlp.hook_post" in name
        )

    activations_for_sample = {
        name: tensor.squeeze(0).cpu()  
        for name, tensor in cache.items()
    }
    
    data_to_save = {
        "id": sample_id,
        "is_good": is_good,
        "activations": activations_for_sample
    }

    torch.save(data_to_save, output_file)

print("\n" + "="*50)
print("Activation caching complete!")
print(f"Cached data has been saved to: {OUTPUT_DIR}")
print("Next step: Use these cached files to train a sparse probe.")
print("="*50)