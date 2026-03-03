import torch
import numpy as np
import json
import os
import sys
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformer_lens import HookedTransformer
from adaptive_classifier.attention_classifier_model import AttentionClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOMAIN_MAP = {'geometry': 0, 'combinatorics': 1, 'number_theory': 2, 'algebra': 3}
NUM_DOMAINS = len(DOMAIN_MAP)


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_predictions_with_domain(data, base_model, classifier_model, feature_indices, desc):
    all_labels, all_preds_proba = [], []
    with torch.no_grad():
        for sample in tqdm(data, desc=desc):
            prompt, label = sample['question'], 1 if sample.get('is_good', False) else 0
            domain = sample.get('domain', None)
            if domain not in DOMAIN_MAP: continue

            _, cache = base_model.run_with_cache(prompt, names_filter=lambda n: "mlp.hook_post" in n)
            layer_names = sorted(cache.keys())
            full_features = torch.cat([cache[name].squeeze(0) for name in layer_names], dim=1).to(DEVICE)
            reduced_features = torch.index_select(full_features, 1, feature_indices)

            num_tokens = reduced_features.shape[0]
            domain_one_hot = F.one_hot(torch.tensor(DOMAIN_MAP[domain]), num_classes=NUM_DOMAINS).to(DEVICE)
            domain_feature = domain_one_hot.unsqueeze(0).expand(num_tokens, -1)
            final_features = torch.cat([reduced_features, domain_feature], dim=1)

            logit = classifier_model(final_features)
            all_labels.append(label)
            all_preds_proba.append(torch.sigmoid(logit).item())
    return np.array(all_labels), np.array(all_preds_proba)

def get_predictions_no_domain(data, base_model, classifier_model, feature_indices, desc):
    all_labels, all_preds_proba = [], []
    with torch.no_grad():
        for sample in tqdm(data, desc=desc):
            prompt, label = sample['question'], 1 if sample.get('is_good', False) else 0

            _, cache = base_model.run_with_cache(prompt, names_filter=lambda n: "mlp.hook_post" in n)
            layer_names = sorted(cache.keys())
            full_features = torch.cat([cache[name].squeeze(0) for name in layer_names], dim=1).to(DEVICE)
            
            final_features = torch.index_select(full_features, 1, feature_indices)

            logit = classifier_model(final_features)
            all_labels.append(label)
            all_preds_proba.append(torch.sigmoid(logit).item())
    return np.array(all_labels), np.array(all_preds_proba)


def main(n_features, with_domain, base_model_name, validation_data_path, test_data_path, model_output_dir, report_output_dir):
    feature_indices_path = model_output_dir / f"top_{n_features}_feature_indices.pt"
    
    if with_domain:
        model_suffix = f"top{n_features}_with_domain_best.pth"
        report_suffix = f"report_top{n_features}_with_domain.jsonl"
        input_dim = n_features + NUM_DOMAINS
    else:
        model_suffix = f"top{n_features}_no_domain_best.pth"
        report_suffix = f"report_top{n_features}_no_domain.jsonl"
        input_dim = n_features

    model_path = model_output_dir / model_suffix
    report_file_path = report_output_dir / report_suffix
    
    if not feature_indices_path.exists():
        print(f"'{feature_indices_path}' not found"); sys.exit(1)
    feature_indices = torch.load(feature_indices_path, weights_only=True).to(DEVICE)

    if not model_path.exists():
        print(f"error: model file '{model_path}' not found!"); sys.exit(1)
        
    base_model = HookedTransformer.from_pretrained(base_model_name, device=DEVICE, trust_remote_code=True)
    base_model.eval()
    
    classifier_model = AttentionClassifier(input_dim).to(DEVICE)
    classifier_model.load_state_dict(torch.load(model_path, weights_only=True))
    classifier_model.eval()
    
    validation_data = load_data(validation_data_path)
    test_data = load_data(test_data_path)
    
    if with_domain:
        val_labels, val_preds_proba = get_predictions_with_domain(validation_data, base_model, classifier_model, feature_indices, "Predicting on validation set (with domain)")
    else:
        val_labels, val_preds_proba = get_predictions_no_domain(validation_data, base_model, classifier_model, feature_indices, "Predicting on validation set (no domain)")
    
    if len(val_labels) == 0:
        print("error!"); sys.exit(1)

    fpr, tpr, thresholds = roc_curve(val_labels, val_preds_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold (from validation set): {optimal_threshold:.4f}")

    threshold_save_path = model_output_dir / f"top{n_features}_optimal_threshold.pt"
    torch.save(optimal_threshold, threshold_save_path)
    print(f"Optimal threshold saved to: {threshold_save_path}")
    
    
    if with_domain:
        test_labels, test_preds_proba = get_predictions_with_domain(test_data, base_model, classifier_model, feature_indices, "Evaluating on test set (with domain)")
    else:
        test_labels, test_preds_proba = get_predictions_no_domain(test_data, base_model, classifier_model, feature_indices, "Evaluating on test set (no domain)")

    if len(test_labels) == 0:
        print("error!"); sys.exit(1)
        
    test_auc = roc_auc_score(test_labels, test_preds_proba)
    print(f"ROC AUC: {test_auc:.4f}\n")
    report_optimal = classification_report(test_labels, (test_preds_proba > optimal_threshold).astype(int), target_names=['Bad (0)', 'Good (1)'])
    print(report_optimal)

    with open(report_file_path, 'w', encoding='utf-8') as f:
        num_predictions = len(test_preds_proba)
        for i in range(num_predictions):
            pred_proba = test_preds_proba[i]
            true_label = test_labels[i]
            sample_data = test_data[i]
            
            final_pred_label = 1 if pred_proba > optimal_threshold else 0
            
            entry = { 
                "id": sample_data.get('id', i),
                "ground_truth_label": int(true_label), 
                "predicted_probability": round(pred_proba, 4), 
                "predicted_label_optimal": final_pred_label 
            }
            f.write(json.dumps(entry) + '\n')
            
    return test_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the attention classifier.")
    parser.add_argument("--dataset", type=str, default="AIME", help="Dataset name.")
    parser.add_argument("--MODEL", type=str, default="Qwen/Qwen3-1.7B", help="Base model name or path.")
    parser.add_argument("--n_features", type=int, required=True, help="Number of top features (neurons) to use.")
    parser.add_argument('--with_domain', action='store_true', help='Evaluate the model trained WITH domain features.')
    args = parser.parse_args()

    dataset = args.dataset
    base_model_name = args.MODEL
    validation_data_path = PROJECT_ROOT / "data" / dataset / f"{dataset}_validation.jsonl"
    test_data_path = PROJECT_ROOT / "data" / dataset / f"{dataset}_test.jsonl"
    model_output_dir = PROJECT_ROOT / "adaptive_classifier" / "classifier_model" / dataset
    report_output_dir = PROJECT_ROOT / "adaptive_classifier" / "classifier_reports" / dataset
    os.makedirs(report_output_dir, exist_ok=True)

    main(args.n_features, args.with_domain, base_model_name, validation_data_path, test_data_path, model_output_dir, report_output_dir)