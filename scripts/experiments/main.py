# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
# add parent directory to path
sys.path.append("..")
sys.path.append(".")
import os
import pickle
import time
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE

from bertviz import model_view
import wandb


def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    seed_everything(41)
    accuracies = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)
    
    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions,_ = run_icl(model, tokenizer, task, baseline_datasets, include_train=False)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL and Task Vector
    # TODO: Change back to 400, 100
    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 50, 50
    test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
    print("-"*10, "test_datasets", "-"*10)
    print("test_datasets size", len(test_datasets))
    print(test_datasets[0].train_inputs)
    print(test_datasets[0].train_outputs)
    print(test_datasets[0].test_input)
    print(test_datasets[0].test_output)
    print("-"*10, "test_datasets", "-"*10)
    dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
    print("-"*10, "dev_datasets", "-"*10)
    print(dev_datasets[0].train_inputs)
    print(dev_datasets[0].train_outputs)
    print("-"*10, "dev_datasets", "-"*10)
    if "copying" in task_name:
        icl_predictions, attention_html = run_icl(model, tokenizer, task, test_datasets, copy_task=True)
    else:
        icl_predictions, attention_html = run_icl(model, tokenizer, task, test_datasets)
    # model view
    # html = attention_html.data
    # with open(f'{task_name}-attention_html.html', 'w') as f:
    #     f.write(html)

    tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
    )
    accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
    accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
    accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)

    tv_ordered_tokens_by_layer = {}
    try:
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens.mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)
    except Exception as e:
        print("Error:", e)

    return accuracies, tv_ordered_tokens_by_layer


def run_main_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    print("Evaluating model:", model_type, model_variant)

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    limit_gpus(range(0, 8))

    print("Loading model and tokenizer...")
    device= "cuda:2"
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant, device)
    model.tie_weights()
    print("Loaded model and tokenizer. dveice:", model.device)

    tasks = get_all_tasks(tokenizer=tokenizer)
    num_examples = 5
    wandb_config = {
        "model_type": model_type,
        "model_variant": model_variant,
        "tasks": list(tasks.keys()),
        "num_examples": num_examples,
    }
    wandb.init(project="task-vector-copying", config={"num_examples": num_examples})
    wandb.define_metric("custom_step")
    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        task = tasks[task_name]
        if task_name in results:
            print(f"Skipping task {i+1}/{len(tasks)}: {task_name}")
            continue
        results[task_name] = {}

        print("\n" + "=" * 50)
        print(f"Running task {i+1}/{len(tasks)}: {task_name}")

        tic = time.time()
        accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples)

        print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        print(f"ICL Accuracy: {accuracies['icl']:.2f}")
        print(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
        print(f"Dev Accuracy by layer: ", end="")
        wandb.log({f"baseline_accuracy/{task_name}": accuracies['baseline'], })
        wandb.log({f"icl_accuracy/{task_name}": accuracies['icl'], })
        wandb.log({f"tv_accuracy/{task_name}": accuracies['tv'], })
        wandb.define_metric(f"tv_dev_accuracy_layer/{task_name}", step_metric="custom_step")
        for layer, accuracy in accuracies["tv_dev_by_layer"].items():
            print(f"{layer}: {accuracy:.2f}")
            wandb.log({f"tv_dev_accuracy_layer/{task_name}": accuracy,"custom_step": layer})
        
        print()
        print("Time:", time.time() - tic)

        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_examples": num_examples,
            "icl_accuracy": accuracies["icl"],
            "tv_accuracy": accuracies["tv"],
            "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)


def get_new_experiment_id() -> str:
    return str(
        max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
    )


def main():
    if len(sys.argv) == 1:
        # Run all models
        # Calculate the experiment_id as the max experiment_id + 1
        experiment_id = get_new_experiment_id()
        for model_type, model_variant in MODELS_TO_EVALUATE:
            run_main_experiment(model_type, model_variant, experiment_id=experiment_id,)
    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) == 3:
            model_type, model_variant = sys.argv[1:]

        run_main_experiment(model_type, model_variant)


if __name__ == "__main__":
    main()
