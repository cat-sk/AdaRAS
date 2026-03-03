import json
from pathlib import Path

def load_target_sample(data_path: str, target_key: list):
    key = tuple(target_key)
    path = Path(data_path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if (d['id'], d['_id']) == key:
                print(f"Successfully loaded sample: ID={key}")
                return d
    raise ValueError(f"Target sample KEY={key} not found in file {data_path}")

def load_all_data_for_keys(data_path: str):
    path = Path(data_path)
    data_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            data_dict[(d['id'], d['_id'])] = d
    return data_dict

def load_all_samples(data_path: str):
    path = Path(data_path)
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Successfully loaded {len(samples)} samples from {data_path}.")
    return samples

