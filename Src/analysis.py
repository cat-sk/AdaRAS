import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformer_lens.utils import get_act_name

def load_neuron_pattern(pattern_path: str, num_neurons: int):
    print(f"Loading neuron pattern from {pattern_path}...")
    pattern_df = pd.read_csv(pattern_path)

    top_neurons_df = pattern_df.head(num_neurons)
    print(f"Identified {len(top_neurons_df)} neurons for intervention.")

    return top_neurons_df

@torch.no_grad()
def calculate_average_activations_from_cache(
    activation_cache_dir: str,
    good_sample_keys: list,
    target_neurons_df: pd.DataFrame,
    device: str = "cuda"
):
    target_activations = {}
    neuron_counts = {}

    print(f"\nCalculating average activations from cache for {len(good_sample_keys)} good/bad samples...")
    
    for main_id, attempt_id in tqdm(good_sample_keys, desc="Reading activation cache"):
        cache_file = Path(activation_cache_dir) / f"sample_{main_id}_attempt{attempt_id}.pt"
        if not cache_file.exists():
            continue

        cached_data = torch.load(cache_file, map_location=device)
        
        for _, neuron_info in target_neurons_df.iterrows():
            layer = int(neuron_info['layer'])
            neuron_idx = int(neuron_info['neuron_index_in_layer'])
            overall_idx = int(neuron_info['overall_neuron_index'])
            
            act_name = get_act_name("post", layer)
            if act_name in cached_data['activations']:
                activations = cached_data['activations'][act_name]
                
                neuron_activation_mean = activations[:, neuron_idx].mean().item()

                target_activations[overall_idx] = target_activations.get(overall_idx, 0) + neuron_activation_mean
                neuron_counts[overall_idx] = neuron_counts.get(overall_idx, 0) + 1
    
    avg_activations = {
        idx: target_activations[idx] / neuron_counts[idx]
        for idx in target_activations if neuron_counts[idx] > 0
    }
        
    print("Average activations of target neurons calculation complete.")
    return avg_activations