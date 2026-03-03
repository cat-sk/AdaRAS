import subprocess
import os
import json
import re
from pathlib import Path

def _extract_last_number(s: str) -> int:
    match = re.findall(r'\d+', s)
    return int(match[-1]) if match else -1

def run_evalplus(samples_path: str, eval_results_dir: str, dataset_name: str):
    print("\n" + "="*80)
    print("Starting EvalPlus evaluation...")
    
    os.makedirs(eval_results_dir, exist_ok=True)

    raw_output_file = Path(eval_results_dir) / "eval_results_raw.json"
    formatted_output_file = Path(eval_results_dir) / "eval_results_formatted.json"

    cmd = [
        "evalplus.evaluate",
        "--dataset", dataset_name,
        "--samples", os.path.abspath(samples_path),
        "--output_file", str(raw_output_file.absolute()),
        "--test-details"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"EvalPlus run failed with stderr:\n{e.stderr}")
        return {}, ""

    print("EvalPlus completed. Formatting results...")
    
    try:
        with open(raw_output_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        if "eval" in data and isinstance(data["eval"], dict):
             sorted_eval = sorted(data["eval"].items(), key=lambda x: _extract_last_number(x[0]))
             data["eval"] = dict(sorted_eval)

        with open(formatted_output_file, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=4, ensure_ascii=False)
        
        base_count = 0
        plus_count = 0
        TOTAL_TASKS = len(data.get("eval", {}))
        for results in data.get("eval", {}).values():
            if not results: continue
            if results[0].get("base_status") == "pass":
                base_count += 1
            if results[0].get("plus_status") == "pass":
                plus_count += 1
        pass_1_base = base_count / TOTAL_TASKS
        pass_1_plus = plus_count / TOTAL_TASKS

        metrics = {
            "pass@1_base": pass_1_base,
            "pass@1_plus": pass_1_plus
        }
        
        print(f"Metrics: pass@1(base)={pass_1_base:.4f}, pass@1(plus)={pass_1_plus:.4f}")
        print(f"Formatted results saved to: {formatted_output_file}")
        return metrics, str(formatted_output_file)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing evalplus output: {e}")
        return {}, ""

def update_jsonl_with_results(original_jsonl_path: str, eval_results_path: str, output_dir: str):
    print("Updating original .jsonl with evaluation results...")
    
    try:
        with open(eval_results_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f).get("eval", {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Cannot read eval results file: {e}")
        return

    results_lookup = {}
    for task_id, solutions in eval_data.items():
        if solutions:
            results_lookup[task_id] = {
                "base_status": solutions[0].get("base_status"),
                "plus_status": solutions[0].get("plus_status")
            }

    base_output_path = Path(output_dir) / "output_base_results.jsonl"
    plus_output_path = Path(output_dir) / "output_plus_results.jsonl"

    with open(original_jsonl_path, 'r', encoding='utf-8') as f_in, \
         open(base_output_path, 'w', encoding='utf-8') as f_base, \
         open(plus_output_path, 'w', encoding='utf-8') as f_plus:
        
        for line in f_in:
            sample_data = json.loads(line)
            task_id = sample_data.get("task_id")
            
            if task_id in results_lookup:
                base_passed = results_lookup[task_id]["base_status"] == "pass"
                plus_passed = results_lookup[task_id]["plus_status"] == "pass"

                base_sample = sample_data.copy()
                base_sample["is_good_base"] = base_passed
                f_base.write(json.dumps(base_sample, ensure_ascii=False) + "\n")

                plus_sample = sample_data.copy()
                plus_sample["is_good_plus"] = plus_passed
                f_plus.write(json.dumps(plus_sample, ensure_ascii=False) + "\n")

    print(f"Base results (+is_good_base) saved to: {base_output_path}")
    print(f"Plus results (+is_good_plus) saved to: {plus_output_path}")
    print("="*80 + "\n")