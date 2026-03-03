import torch
import torch.nn.functional as F
import pandas as pd

class ActivationIntervener:
    def __init__(self, model, config: dict, target_neurons_df: pd.DataFrame):
        self.model = model
        self.config = config
        self.target_neurons_df = target_neurons_df
        self.hooks = []
        self.steering_vectors = {}
        self.good_avgs = {}

        for _, neuron_info in target_neurons_df.iterrows():
            overall_idx = int(neuron_info['overall_neuron_index'])
            self.steering_vectors[overall_idx] = float(neuron_info['weight'])
            if 'good_avg' in neuron_info:
                self.good_avgs[overall_idx] = float(neuron_info['good_avg'])

    def register_hooks(self):
        if not self.config.get('enabled', False):
            return

        target_layers = self.target_neurons_df['layer'].unique()

        for layer_id in target_layers:
            try:
                module_path_parts = f"model.layers.{layer_id}.mlp.down_proj".split('.')
                module_to_hook = self.model
                for part in module_path_parts:
                    module_to_hook = getattr(module_to_hook, part)
                
                def create_hook_fn(local_layer_id):
                    def hook_fn_pre(module, input):
                        activation = input[0]
                        batch_size = activation.shape[0]
                        for i in range(batch_size):
                            neurons_in_this_layer = self.target_neurons_df[self.target_neurons_df['layer'] == local_layer_id]

                            for _, neuron_info in neurons_in_this_layer.iterrows():
                                neuron_idx = int(neuron_info['neuron_index_in_layer'])
                                overall_idx = int(neuron_info['overall_neuron_index'])
                                
                                target_avg = self.good_avgs.get(overall_idx)

                                original_activation = activation[i, -1, neuron_idx]
                                strength = self.config['strength']
                                method = self.config.get('method', 'linear_blend')

                                if method == 'linear_blend':
                                    if target_avg is None:
                                        continue
                                    intervened_value = (1 - strength) * original_activation + (strength * target_avg)
                                elif method == 'softplus_add':
                                    if target_avg is None:
                                        continue
                                    enhancement_signal = F.softplus(torch.tensor(target_avg, device=activation.device))
                                    intervened_value = original_activation + (strength * enhancement_signal)
                                elif method == 'steering_add':
                                    steering_vector_val = self.steering_vectors.get(overall_idx)
                                    if steering_vector_val is None:
                                        continue
                                    intervened_value = original_activation + (strength * steering_vector_val)
                                else:
                                    raise ValueError(f"Unknown intervention method: {method}")

                                activation[i, -1, neuron_idx] = intervened_value
                        
                        return input
                    return hook_fn_pre

                handle = module_to_hook.register_forward_pre_hook(create_hook_fn(layer_id))
                self.hooks.append(handle)

            except AttributeError:
                print(f"Warning: Expected MLP submodule path 'model.layers.{layer_id}.mlp.down_proj' not found at layer {layer_id}. Skipping intervention for this layer.")
                continue

        print(f"Successfully registered {len(self.hooks)} intervention hooks on {len(self.hooks)} layers.")

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        print("All intervention hooks have been successfully removed.")