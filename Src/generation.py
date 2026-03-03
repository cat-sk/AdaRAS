import torch
import re
import time
import json
from tqdm import tqdm
from Src.util import last_boxed_only_string, remove_boxed
from Src.math_equivalence import is_equiv

def run_evaluation(model, tokenizer, sample_data, generation_params):
    gold_answer_value = str(sample_data['answer']).strip()

    messages = [
        {"role": "user", "content": f"Question: {sample_data['question']}\nThinking process:\nPlease provide a step-by-step thinking process and put your final answer in \\boxed{{}}."}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **generation_params)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    try:
        end_think_token_id = 151668
        index = len(output_ids) - output_ids[::-1].index(end_think_token_id)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    except ValueError:
        thinking_content = "Error: </think> token not found."
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    boxed_str = last_boxed_only_string(content)
    extracted_value = remove_boxed(boxed_str)

    try:
        is_good = is_equiv(extracted_value, gold_answer_value)
    except:
        is_good = False


    end = time.time()
    
    result_data = {
        "id" : sample_data['id'],
        "thinking_content": thinking_content,
        "content": content,
        "extracted_value": extracted_value,
        "is_good": is_good,
        "time_taken": f"{(end - start):.2f}s"
    }
            
    return result_data