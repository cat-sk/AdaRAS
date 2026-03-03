import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from transformer_lens import HookedTransformer
from adaptive_classifier.attention_classifier_model import AttentionClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOMAIN_MAP = {'geometry': 0, 'combinatorics': 1, 'number_theory': 2, 'algebra': 3}
NUM_DOMAINS = len(DOMAIN_MAP)

DEFAULT_N_FEATURES = 256
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_DROPOUT_RATE = 0.3
DEFAULT_PATIENCE = 10

def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def cache_features(data, base_model, with_domain, desc):
    cached_data = []
    with torch.no_grad():
        for sample in tqdm(data, desc=desc):
            prompt = sample['question']
            _, cache = base_model.run_with_cache(prompt, names_filter=lambda n: "mlp.hook_post" in n)
            layer_names = sorted(cache.keys())
            token_features = torch.cat([cache[name].squeeze(0).cpu() for name in layer_names], dim=1)
            label = 1 if sample.get('is_good', False) else 0
            
            entry = {'features': token_features, 'label': label}
            
            if with_domain:
                domain = sample.get('domain', None)
                if domain not in DOMAIN_MAP:
                    print(f"Warning: Sample ID {sample.get('id')} is missing a valid domain and will be skipped.")
                    continue
                entry['domain'] = domain
            
            cached_data.append(entry)
    return cached_data

def main(n_features, with_domain, lr, epochs, weight_decay, dropout_rate, patience,
         base_model_name, train_data_path, validation_data_path, model_output_dir):

    feature_indices_path = model_output_dir / f"top_{n_features}_feature_indices.pt"
    
    if with_domain:
        model_suffix = f"top{n_features}_with_domain_best.pth"
        input_dim = n_features + NUM_DOMAINS
    else:
        model_suffix = f"top{n_features}_no_domain_best.pth"
        input_dim = n_features

    model_save_path = model_output_dir / model_suffix
    
    if not feature_indices_path.exists():
        print(f"Error: Feature indices file '{feature_indices_path}' not found! Please run select_features.py first.")
        sys.exit(1)
    feature_indices = torch.load(feature_indices_path, weights_only=True).to(DEVICE)
    print(f"Total input dimension: {input_dim}")
    base_model = HookedTransformer.from_pretrained(base_model_name, device=DEVICE, trust_remote_code=True)
    base_model.eval()

    train_data = load_data(train_data_path)
    validation_data = load_data(validation_data_path)
    
    cached_train_data = cache_features(train_data, base_model, with_domain, "Caching training set")
    cached_val_data = cache_features(validation_data, base_model, with_domain, "Caching validation set")
    
    del base_model
    torch.cuda.empty_cache()

    classifier_model = AttentionClassifier(input_dim, dropout_rate=dropout_rate).to(DEVICE)
    
    labels = [d['label'] for d in cached_train_data]
    pos_weight = (len(labels) - sum(labels)) / sum(labels) if sum(labels) > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    optimizer = optim.Adam(classifier_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        classifier_model.train()
        total_train_loss = 0
        for sample in tqdm(cached_train_data, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            full_token_features = sample['features'].to(DEVICE)
            label_tensor = torch.tensor([float(sample['label'])], device=DEVICE)
            
            reduced_features = torch.index_select(full_token_features, 1, feature_indices)

            if with_domain:
                num_tokens = reduced_features.shape[0]
                domain_index = DOMAIN_MAP[sample['domain']]
                domain_one_hot = F.one_hot(torch.tensor(domain_index), num_classes=NUM_DOMAINS).to(DEVICE)
                domain_feature = domain_one_hot.unsqueeze(0).expand(num_tokens, -1)
                final_features = torch.cat([reduced_features, domain_feature], dim=1)
            else:
                final_features = reduced_features
            
            optimizer.zero_grad()
            logit = classifier_model(final_features)
            loss = criterion(logit.unsqueeze(0), label_tensor.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(cached_train_data)

        classifier_model.eval()
        total_val_loss = 0
        all_val_labels, all_val_preds_proba = [], []

        with torch.no_grad():
            for sample in tqdm(cached_val_data, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                full_token_features = sample['features'].to(DEVICE)
                
                reduced_features = torch.index_select(full_token_features, 1, feature_indices)

                if with_domain:
                    num_tokens = reduced_features.shape[0]
                    domain_index = DOMAIN_MAP[sample['domain']]
                    domain_one_hot = F.one_hot(torch.tensor(domain_index), num_classes=NUM_DOMAINS).to(DEVICE)
                    domain_feature = domain_one_hot.unsqueeze(0).expand(num_tokens, -1)
                    final_features = torch.cat([reduced_features, domain_feature], dim=1)
                else:
                    final_features = reduced_features
                
                logit = classifier_model(final_features)
                loss = criterion(logit.unsqueeze(0), torch.tensor([float(sample['label'])], device=DEVICE).unsqueeze(0))
                
                total_val_loss += loss.item()
                all_val_labels.append(sample['label'])
                all_val_preds_proba.append(torch.sigmoid(logit).item())

        avg_val_loss = total_val_loss / len(cached_val_data)
        val_auc = roc_auc_score(all_val_labels, all_val_preds_proba) if len(np.unique(all_val_labels)) > 1 else 0.5
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val AUC: {val_auc:.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(classifier_model.state_dict(), model_save_path)
            print(f"  -> New best validation AUC, model saved to '{model_save_path.name}'")
        else:
            patience_counter += 1
            print(f"  -> Validation AUC did not improve, Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered.")
            break

    print(f"Best model saved to: {model_save_path} (with highest validation AUC: {best_val_auc:.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the attention classifier, with or without domain features.")
    
    parser.add_argument("--dataset", type=str, default="AIME", help="Dataset name.")
    parser.add_argument("--MODEL", type=str, default="Qwen/Qwen3-1.7B", help="Base model name or path.")
    parser.add_argument("--n_features", type=int, default=DEFAULT_N_FEATURES, help="Number of top features (neurons) to use.")
    parser.add_argument('--with_domain', action='store_true', help='If specified, train WITH domain features.')
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for Adam optimizer.")
    parser.add_argument("--dropout_rate", type=float, default=DEFAULT_DROPOUT_RATE, help="Dropout rate in the classifier.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Patience for early stopping.")
    
    args = parser.parse_args()

    dataset = args.dataset
    base_model_name = args.MODEL
    train_data_path = PROJECT_ROOT / "data" / dataset / f"{dataset}_train.jsonl"
    validation_data_path = PROJECT_ROOT / "data" / dataset / f"{dataset}_validation.jsonl"
    model_output_dir = PROJECT_ROOT / "adaptive_classifier" / "classifier_model" / dataset
    os.makedirs(model_output_dir, exist_ok=True)

    main(
        n_features=args.n_features,
        with_domain=args.with_domain,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        patience=args.patience,
        base_model_name=base_model_name,
        train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        model_output_dir=model_output_dir,
    )