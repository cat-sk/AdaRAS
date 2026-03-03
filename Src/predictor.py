import torch
from pathlib import Path
from transformer_lens import HookedTransformer

from adaptive_classifier.attention_classifier_model import AttentionClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTOR_DEVICE = None
PREDICTOR_MODEL = None
CLASSIFIER_MODEL = None
FEATURE_INDICES = None
OPTIMAL_THRESHOLD = None


def load_models_if_needed(
    model_name_override=None,
    device_override=None,
    dataset=None,
    n_features=None,
    threshold=None,
):
    global PREDICTOR_DEVICE, PREDICTOR_MODEL, CLASSIFIER_MODEL, FEATURE_INDICES, OPTIMAL_THRESHOLD

    if PREDICTOR_MODEL is not None:
        return

    print("--- Loading models for prediction for the first time... ---")

    PREDICTOR_DEVICE = device_override if device_override else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_to_load = model_name_override if model_name_override else "Qwen/Qwen3-1.7B"
    print(f"Loading HookedTransformer '{model_to_load}' for prediction...")
    PREDICTOR_MODEL = HookedTransformer.from_pretrained(
        model_to_load,
        device=PREDICTOR_DEVICE,
        trust_remote_code=True,
    )
    PREDICTOR_MODEL.eval()

    if dataset is None or n_features is None:
        raise ValueError("'dataset' and 'n_features' must be provided to load_models_if_needed.")

    feature_indices_path = (
        PROJECT_ROOT / "adaptive_classifier" / "classifier_model" / dataset
        / f"top_{n_features}_feature_indices.pt"
    )
    model_path = (
        PROJECT_ROOT / "adaptive_classifier" / "classifier_model" / dataset
        / f"top{n_features}_no_domain_best.pth"
    )
    threshold_path = (
        PROJECT_ROOT / "adaptive_classifier" / "classifier_model" / dataset
        / f"top{n_features}_optimal_threshold.pt"
    )

    if not feature_indices_path.exists():
        raise FileNotFoundError(f"Feature indices not found: {feature_indices_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {model_path}")

    print(f"Loading feature indices from {feature_indices_path}...")
    FEATURE_INDICES = torch.load(feature_indices_path, map_location=PREDICTOR_DEVICE, weights_only=True)

    print(f"Loading AttentionClassifier from {model_path}...")
    CLASSIFIER_MODEL = AttentionClassifier(input_dim=n_features).to(PREDICTOR_DEVICE)
    CLASSIFIER_MODEL.load_state_dict(torch.load(model_path, map_location=PREDICTOR_DEVICE, weights_only=True))
    CLASSIFIER_MODEL.eval()

    if threshold is not None:
        OPTIMAL_THRESHOLD = threshold
        print(f"Using manually specified threshold: {OPTIMAL_THRESHOLD:.4f}")
    elif threshold_path.exists():
        OPTIMAL_THRESHOLD = float(torch.load(threshold_path, weights_only=False))
        print(f"Loaded optimal threshold from file: {OPTIMAL_THRESHOLD:.4f}")
    else:
        OPTIMAL_THRESHOLD = 0.5
        print(f"Warning: threshold file not found, falling back to default threshold=0.5. Run evaluate_attention_classifier.py first to generate it.")

    print("--- All prediction models loaded successfully. ---\n")


def predict_question_will_be_good(question_text: str):
    if PREDICTOR_MODEL is None or CLASSIFIER_MODEL is None or FEATURE_INDICES is None:
        raise RuntimeError("Models not loaded. Call load_models_if_needed() first.")

    with torch.no_grad():
        _, cache = PREDICTOR_MODEL.run_with_cache(
            question_text,
            names_filter=lambda name: "mlp.hook_post" in name,
        )
        layer_names = sorted(cache.keys())
        full_features = torch.cat(
            [cache[name].squeeze(0) for name in layer_names], dim=1
        ).to(PREDICTOR_DEVICE)

        reduced_features = torch.index_select(full_features, 1, FEATURE_INDICES)

        logit = CLASSIFIER_MODEL(reduced_features)
        score = torch.sigmoid(logit).item()

    decision = score >= OPTIMAL_THRESHOLD
    return decision, score