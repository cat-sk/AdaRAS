import torch
import json
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_selection import f_classif
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformer_lens import HookedTransformer

BASE_MODEL_NAME = "Qwen3-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dataset = "AIME"

def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main(n_features_list, train_data_path, model_output_dir, base_model_name):
    base_model = HookedTransformer.from_pretrained(base_model_name, device=DEVICE, trust_remote_code=True)
    base_model.eval()
    
    train_data = load_data(train_data_path)
    
    all_activations = []
    all_labels = []

    with torch.no_grad():
        for sample in tqdm(train_data, desc="Extract activation values"):
            prompt = sample['question']
            _, cache = base_model.run_with_cache(prompt, names_filter=lambda n: "mlp.hook_post" in n)
            layer_names = sorted(cache.keys())
            
            token_features = torch.cat([cache[name].squeeze(0).cpu() for name in layer_names], dim=1)
            
            problem_vector = token_features.mean(dim=0) 
            
            all_activations.append(problem_vector)
            all_labels.append(1 if sample.get('is_good', False) else 0)

    X_train = torch.stack(all_activations).numpy()
    y_train = np.array(all_labels)

    del base_model
    torch.cuda.empty_cache()

    f_scores, _ = f_classif(X_train, y_train)
    print("F-scores calculation completed.")
    
    sorted_indices = np.argsort(f_scores)

    for n_features in n_features_list:
        if n_features > len(sorted_indices):
            print(f"Warning: Requested number of features {n_features} is greater than total features {len(sorted_indices)}. Saving all features.")
            top_indices = sorted_indices
        else:
            top_indices = sorted_indices[-n_features:]
        
        output_path = model_output_dir / f"top_{n_features}_feature_indices.pt"
        torch.save(torch.tensor(top_indices, dtype=torch.long), output_path)
        print(f"  -> Saved Top {n_features} features to: {output_path}")

    print(f"\nAll done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=dataset)
    parser.add_argument("--MODEL", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--n_features", type=int, nargs='+', default=[256], help="List of top N features to select")
    args = parser.parse_args()

    dataset = args.dataset
    BASE_MODEL_NAME = args.MODEL

    TRAIN_DATA_PATH = PROJECT_ROOT / "data" / dataset / f"{dataset}_train.jsonl"
    MODEL_OUTPUT_DIR = PROJECT_ROOT / "adaptive_classifier" / "classifier_model" / dataset
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    main(args.n_features, TRAIN_DATA_PATH, MODEL_OUTPUT_DIR, args.MODEL)