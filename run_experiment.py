import yaml
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from Src.data_loader import load_all_samples
from Src.model_utils import load_model_and_tokenizer 
from Src.intervention import ActivationIntervener
from Src.generation import run_evaluation
from Src.generation_code import run_evaluation as run_evaluation_code

from Src.analysis import load_neuron_pattern
from Src.predictor import predict_question_will_be_good, load_models_if_needed
from Src.Evalplus_evaluator import run_evalplus, update_jsonl_with_results

def main():
    parser = argparse.ArgumentParser(description="Run neuron intervention or control group experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file for the experiment")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if 'extend' not in config or not config['extend']:
        output_dir = Path("results") / config['project'] / config['experiment_name']
    else:
        output_dir = Path(f"results_{config['extend']}") / config['project'] / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.jsonl"
    summary_file = output_dir / "summary.txt"

    model, tokenizer = load_model_and_tokenizer(config['model_name'], config['device'])
    all_samples = load_all_samples(config['sample_path'])

    use_dynamic_classifier = config.get('intervention', {}).get('use_dynamic_classifier', False)
    use_intervention = config.get('intervention', {}).get('enabled', False)

    if use_dynamic_classifier:
        print("\n[INFO] Dynamic intervention classifier is ENABLED.")
        classifier_config = config['intervention'].get('classifier', {})
        load_models_if_needed(
            model_name_override=config['model_name'],
            device_override=config['device'],
            dataset=classifier_config.get('dataset', config['project']),
            n_features=classifier_config['n_features'],
            threshold=classifier_config.get('threshold', None),
        )
        print("\n")

    intervener = None
    if use_intervention:
        interv_config = config['intervention']
        
        target_neurons_df = load_neuron_pattern(
            interv_config['pattern_path'],
            interv_config['num_neurons_to_intervene']
        )

        intervener = ActivationIntervener(
            model=model,
            config=interv_config,
            target_neurons_df=target_neurons_df,
        )
    
    correct_count = 0
    cur_count = 0
    total_time_taken = 0
    with open(summary_file, "w", encoding="utf-8") as f_summary, \
         open(output_file, "w", encoding="utf-8") as f_output:
        for sample in tqdm(all_samples, desc=f"Evaluating dataset '{config['experiment_name']}'"):
            classifier_prediction = None
            apply_intervention_for_this_sample = False

            if use_dynamic_classifier:
                will_be_good, score = predict_question_will_be_good(sample['question'])
                classifier_prediction = will_be_good
                if not will_be_good:
                    apply_intervention_for_this_sample = True
                    print(f"  [Sample {sample['id']}] Classifier predicted WILL BE BAD (Score: {score:.4f}). INTERVENTION ENABLED.")
                else:
                    print(f"  [Sample {sample['id']}] Classifier predicted WILL BE GOOD (Score: {score:.4f}). Intervention SKIPPED.")
            elif intervener:
                apply_intervention_for_this_sample = True

            if apply_intervention_for_this_sample and intervener:
                intervener.register_hooks()

            if config['type'] == 'code':
                result_data = run_evaluation_code(
                    model, tokenizer, sample, config['generation_params']
                )
            else :
                result_data = run_evaluation(
                    model, tokenizer, sample, config['generation_params']
                )

            if apply_intervention_for_this_sample and intervener:
                intervener.remove_hooks()

            if use_dynamic_classifier and classifier_prediction is not None:
                result_data['prediction_will_be_good'] = bool(classifier_prediction)

            if config['type'] == 'code':
                f_output.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                continue

            if result_data['is_good']:
                correct_count += 1
            total_time_taken += float(result_data['time_taken'][:-1])

            cur_count += 1
            f_summary.write(f"id: {result_data['id']}, is_good: {result_data['is_good']}\n")
            f_output.write(json.dumps(result_data, ensure_ascii=False, indent=2) + "\n")


    if config['type'] == 'code':
        evalplus_base_dir = Path(__file__).resolve().parent / "Evalplus_results"
        if 'extend' not in config or not config['extend']:
            eval_results_dir = evalplus_base_dir / config['project'] / config['experiment_name']
        else:   
            eval_results_dir = evalplus_base_dir / f"{config['project']}_{config['extend']}" / config['experiment_name']
        
        metrics, formatted_results_path = run_evalplus(
            samples_path=str(output_file),
            eval_results_dir=str(eval_results_dir),
            dataset_name=config['project'] 
        )

        if formatted_results_path:
            update_jsonl_with_results(
                original_jsonl_path=str(output_file),
                eval_results_path=formatted_results_path,
                output_dir=str(output_dir) 
            )

        if metrics:
            pass_1_base_acc = metrics.get('pass@1_base', 0.0) * 100
            pass_1_plus_acc = metrics.get('pass@1_plus', 0.0) * 100
            
            report = (
                f"EvalPlus Accuracy (pass@1_base): {pass_1_base_acc:.2f}%\n"
                f"EvalPlus Accuracy (pass@1_plus): {pass_1_plus_acc:.2f}%\n"
            )
            print("\n--- Final Report ---")
            print(report)
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(report)
            
        else:
            print("\nSkipping final report due to evaluation failure.")

        
    else:
        accuracy = (correct_count / len(all_samples)) * 100
        report = (
            f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(all_samples)} correct)\n"
            f"Total Time Taken: {int(total_time_taken // 3600)}h "
            f"{int((total_time_taken % 3600) // 60)}m "
            f"{int(total_time_taken % 60)}s\n"
        )
        print(report)
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(report)

if __name__ == "__main__":
    main()
