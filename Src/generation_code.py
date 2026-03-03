import torch
import re
import time
import json
from tqdm import tqdm

from evalplus.sanitize import sanitize

def run_evaluation(model, tokenizer, sample_data, generation_params):
    messages = [
        {"role": "user", "content": sample_data["question"]}
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

    output = content

    output_format = sanitize(
        output,
        entrypoint=sample_data["entry_point"],
    )

    end = time.time()

    result_data = {
        "id": sample_data["id"],
        "task_id": sample_data["task_id"],
        "thinking_content": thinking_content,
        "content": content,
        "solution": output_format,
        "time_taken": f"{(end - start):.2f}s"
    }

    return result_data