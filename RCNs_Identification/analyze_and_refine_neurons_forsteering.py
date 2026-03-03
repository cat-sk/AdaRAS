import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import sys
from transformers import AutoConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = "AIME" 
MODEL = "Qwen/Qwen3-1.7B" 
MODEL_name_only = "Qwen3-1.7B" 
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=dataset)
parser.add_argument("--MODEL", type=str, default=MODEL)
args = parser.parse_args()
dataset = args.dataset
MODEL = args.MODEL
MODEL_name_only = MODEL.split("/")[-1] if "/" in MODEL else MODEL


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVATION_CACHE_DIR = SCRIPT_DIR / "activation_cache" / f"{dataset}_{MODEL_name_only}"
DATA_PATH_FOR_ACTIVATIONS = PROJECT_ROOT / "data" / dataset / f"{dataset}_results_Qwen3-32B_filtered.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "RCNs_Identification" / "analysis_results" / f"{dataset}_{MODEL_name_only}_refine_steering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = 200

print(f"Loading model config from {MODEL}...")
try:
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    NUM_LAYERS = config.num_hidden_layers
    D_MLP = config.intermediate_size
    print(f"Model Config Loaded: NUM_LAYERS={NUM_LAYERS}, D_MLP={D_MLP}")
except Exception as e:
    print(f"Error loading config: {e}")
    print("Fallback to manual setting.")
    NUM_LAYERS = 28
    D_MLP = 6144 

def load_all_data_for_keys(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data[(d['id'], d['_id'])] = d
    return data

def calculate_all_neuron_avgs(sample_keys, desc, device):
    total_activations = torch.zeros(NUM_LAYERS, D_MLP, device=device)
    activation_counts = torch.zeros(NUM_LAYERS, D_MLP, device=device)
    
    for main_id, attempt_id in tqdm(sample_keys, desc=desc):
        cache_file = ACTIVATION_CACHE_DIR / f"sample_{main_id}_attempt{attempt_id}.pt"
        if not cache_file.exists(): 
            print("error: cache file not found:", cache_file)
            continue
        
        try:
            cached_data = torch.load(cache_file, map_location=device)
            activations_dict = cached_data.get('activations', {})
            
            for layer_idx in range(NUM_LAYERS):
                act_name = f'blocks.{layer_idx}.mlp.hook_post'
                if act_name in activations_dict:
                    layer_activations = activations_dict[act_name]
                    neuron_means_in_sample = layer_activations.mean(dim=0)
                    total_activations[layer_idx] += neuron_means_in_sample
                    activation_counts[layer_idx] += 1
        except Exception as e:
            print(f"Error reading {cache_file}: {e}")
            continue
            
    average_activations = total_activations / (activation_counts + 1e-8)
    return average_activations.cpu().numpy()

def main():
    print(f"Using device: {device}")

    print(f"Loading data from {DATA_PATH_FOR_ACTIVATIONS}...")
    all_data = load_all_data_for_keys(DATA_PATH_FOR_ACTIVATIONS)
    
    good_sample_keys = []
    bad_sample_keys = []
    
    for k, v in all_data.items():
        attempt = v.get('_id', 0)
        key_tuple = (v['id'], attempt)
        if v.get('is_good', False):
            good_sample_keys.append(key_tuple)
        else:
            bad_sample_keys.append(key_tuple)

    print(f"Found {len(good_sample_keys)} good samples, {len(bad_sample_keys)} bad samples.")

    print("\nCalculating average activations for good samples...")
    good_avg_matrix = calculate_all_neuron_avgs(good_sample_keys, "Processing GOOD samples", device)
    print("\nCalculating average activations for bad samples...")
    bad_avg_matrix = calculate_all_neuron_avgs(bad_sample_keys, "Processing BAD samples", device)
    print("\nAll neuron average activations calculated!")

    layer_indices, neuron_indices_in_layer = np.meshgrid(np.arange(NUM_LAYERS), np.arange(D_MLP), indexing='ij') 
    full_analysis_df = pd.DataFrame({
        'layer': layer_indices.flatten(),
        'neuron_index_in_layer': neuron_indices_in_layer.flatten(),
        'good_avg': good_avg_matrix.flatten(),
        'bad_avg': bad_avg_matrix.flatten(),
    })
    
    full_analysis_df['overall_neuron_index'] = full_analysis_df['layer'] * D_MLP + full_analysis_df['neuron_index_in_layer']

    full_analysis_df['weight'] = full_analysis_df['good_avg'] - full_analysis_df['bad_avg']
    full_analysis_df['abs_weight'] = np.abs(full_analysis_df['weight'])
    
    full_analysis_df['direction'] = np.where(full_analysis_df['weight'] > 0, 'predicts_good', 'predicts_bad')

    full_analysis_df['abs_steering_vector'] = full_analysis_df['abs_weight']
    epsilon = 1e-8
    full_analysis_df['relative_steering_vector'] = (
        full_analysis_df['abs_weight'] /
        (np.abs(full_analysis_df['good_avg']) + np.abs(full_analysis_df['bad_avg']) + epsilon)
    )

    output_cols = [
        'overall_neuron_index',
        'layer',
        'neuron_index_in_layer',
        'weight',
        'abs_weight',
    ]

    #top_k_abs_df = full_analysis_df.sort_values(by='abs_steering_vector', ascending=False).head(TOP_K)
    
    #top_k_rel_df = full_analysis_df.sort_values(by='relative_steering_vector', ascending=False).head(TOP_K)
    
    top_k_rel_abs_df = full_analysis_df.sort_values(
        by=['relative_steering_vector', 'abs_steering_vector'], 
        ascending=[False, False] 
    ).head(TOP_K)

    rel_abs_output_path = OUTPUT_DIR / f"top_{TOP_K}_neurons_by_rel_abs_diff.csv"
    
    top_k_rel_abs_df[output_cols].to_csv(rel_abs_output_path, index=False)
    
    print("\nAnalysis complete!")
    print(f"'relative-absolute diff' Top-{TOP_K} neuron list saved to: {rel_abs_output_path}")


if __name__ == "__main__":
    main()