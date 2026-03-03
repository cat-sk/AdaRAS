from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    print(f"Loading model from Hugging Face Hub: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    device_map = "auto" if device == "auto" else device
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True
    )
    model.eval()
    print(f"Model loading complete. use {model.device}.")
    return model, tokenizer
