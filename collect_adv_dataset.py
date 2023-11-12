import argparse
import json
import random
import time

import numpy as np
import torch
from custom_attacks.baegarg2019_05 import BAEGarg2019_05
from custom_attacks.bertattackli2020_05 import BERTAttackLi2020_05
from custom_attacks.pwwsren2019_05 import PWWSRen2019_05
from custom_attacks.pwwsren2019_075 import PWWSRen2019_075
from datasets import load_dataset
from textattack.attack_recipes.bae_garg_2019 import BAEGarg2019
from textattack.attack_recipes.bert_attack_li_2020 import BERTAttackLi2020
from textattack.attack_recipes.faster_genetic_algorithm_jia_2019 import (
    FasterGeneticAlgorithmJia2019,
)
from textattack.attack_recipes.genetic_algorithm_alzantot_2018 import (
    GeneticAlgorithmAlzantot2018,
)
from textattack.attack_recipes.iga_wang_2019 import IGAWang2019
from textattack.attack_recipes.pwws_ren_2019 import PWWSRen2019
from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import BertForSequenceClassification, BertTokenizer
random.seed(0)

attacker = {
    "BERTAttackLi2020": BERTAttackLi2020,
    "BERTAttackLi2020_05": BERTAttackLi2020_05,
    "TextFoolerJin2019": TextFoolerJin2019,
    "PWWSRen2019": PWWSRen2019,
    "PWWSRen2019_075": PWWSRen2019_075,
    "PWWSRen2019_05": PWWSRen2019_05,
    "BAEGarg2019": BAEGarg2019,
    "BAEGarg2019_05": BAEGarg2019_05,
    "GeneticAlzantot2018": GeneticAlgorithmAlzantot2018,
    "FasterGeneticJia2019": FasterGeneticAlgorithmJia2019,
    "IGAWang2019": IGAWang2019,
}

parser = argparse.ArgumentParser()

data_args = parser.add_argument_group("attack")
data_args.add_argument("--dataset", type=str, default="ag_news")
data_args.add_argument("--pre_trained_model", type=str, default="bert-base-uncased")
data_args.add_argument(
    "--fine_tuned_model",
    type=str,
    default="bert-base-uncased-agnews-ml128-rr04linear-ct05-ep10_12",
)
data_args.add_argument("--max_length", type=int, default=128)
data_args.add_argument("--attacker", type=str, default="TextFoolerJin2019")
data_args.add_argument(
    "--outfile", type=str, default="textfooler_adv_dataset_test_linear"
)

def write_json(new_data, filename="sample.json"):
    """
    Appends new data to a JSON file.

    Args:
    - new_data (dict): Data to append.
    - filename (str, optional): File to append the data to. Defaults to "sample.json".
    """
    with open(filename, "r+") as file:
        file_data = json.load(file)
        file_data["adv_examples"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

def load_model_and_tokenizer(pre_trained_model, max_length, fine_tuned_model, device, num_labels):
    """
    Loads the BERT model and tokenizer.

    Args:
    - pre_trained_model (str): Name of the pre-trained model.
    - max_length (int): Maximum length of the tokens.
    - fine_tuned_model (str): Path to the fine-tuned model.
    - device (torch.device): Device to load the model on (CPU or GPU).
    - num_labels (int): Number of labels in the classification task.

    Returns:
    - tuple: A tuple containing the loaded model and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model, model_max_length=max_length, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(pre_trained_model, num_labels=num_labels).to(device)
    model.load_state_dict(torch.load(f"models/{fine_tuned_model}.pth"))

    return model, tokenizer

def load_dataset_and_filter(args):
    """
    Loads the specified dataset and applies filtering based on the dataset type.

    Args:
    - args (argparse.Namespace): Arguments containing dataset specifications.

    Returns:
    - datasets.arrow_dataset.Dataset: The loaded and filtered dataset.
    """
    test_dataset = load_dataset(args.dataset, split="test")
    if args.dataset == "yelp_polarity":
        with open("other/yelp_polarity_indices_shorter_80.json", "r") as openfile:
            indices = json.load(openfile)
        test_dataset = test_dataset.select(indices["indices_shorter_80"])

    selected = random.sample(range(len(test_dataset)), 1000)
    return test_dataset.select(selected)

def prepare_attack(args, model_wrapper):
    """
    Prepares the attack based on the provided arguments.

    Args:
    - args (argparse.Namespace): Arguments containing attack specifications.
    - model_wrapper (textattack.models.wrappers.ModelWrapper): The model wrapper used for attacks.

    Returns:
    - textattack.attack_recipes.AttackRecipe: The configured attack recipe.
    """
    attack = attacker[args.attacker].build(model_wrapper)
    return attack

def attack_samples(test_dataset, attack):
    """
    Attacks the samples in the dataset using the specified attack method.

    Args:
    - test_dataset (datasets.arrow_dataset.Dataset): The dataset to be attacked.
    - attack (textattack.attack_recipes.AttackRecipe): The attack method to use.

    Returns:
    - list: A list containing the results of the attack on each sample.
    """
    custom_dataset = [(sample["text"], sample["label"]) for sample in test_dataset]
    results_iterable = attack.attack_dataset(custom_dataset)

    return results_iterable

def process_attack_results(results_iterable, custom_dataset, out_file):
    """
    Processes the results of the attack, collecting data, statistics, and writing successful attacks to a JSON file.

    Args:
    - results_iterable (iterable): Iterable object containing attack results.
    - custom_dataset (list): List of tuples containing the original texts and labels.
    - out_file (str): Path to the output JSON file.

    Returns:
    - dict: A dictionary containing summary statistics of the attack results.
    """
    failed_attacks, skipped_attacks, successful_attacks = 0, 0, 0
    words_changed = []
    perturbed_word_percentages = []

    for i, result in enumerate(results_iterable):
        original_text, label = custom_dataset[i]
        if isinstance(result, FailedAttackResult):
            failed_attacks += 1
        elif isinstance(result, SkippedAttackResult):
            skipped_attacks += 1
        else:
            successful_attacks += 1
            attack_data = extract_attack_data(result)  # Function to extract relevant data from result
            append_to_json(attack_data, out_file)

    # Original classifier success rate on these samples.
    total_attacks = failed_attacks + skipped_attacks + successful_attacks
    original_accuracy = (total_attacks - skipped_attacks) * 100.0 / (total_attacks)
    original_accuracy = str(round(original_accuracy, 2)) + "%"

    # New classifier success rate on these samples.
    accuracy_under_attack = (failed_attacks) * 100.0 / (total_attacks)
    accuracy_under_attack = str(round(accuracy_under_attack, 2)) + "%"

    # Attack success rate.
    if successful_attacks + failed_attacks == 0:
        attack_success_rate = 0
    else:
        attack_success_rate = (
            successful_attacks * 100.0 / (successful_attacks + failed_attacks)
        )
        attack_success_rate = str(round(attack_success_rate, 2)) + "%"

    perturbed_word_percentages = perturbed_word_percentages[perturbed_word_percentages > 0]
    average_perc_words_perturbed = perturbed_word_percentages.mean() if perturbed_word_percentages else 0
    average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"

    words_changed = words_changed[words_changed > 0]
    average_words_changed = words_changed.mean() if words_changed else 0
    average_words_changed = round(average_words_changed, 2)

    return {
        "failed_attacks": failed_attacks,
        "skipped_attacks": skipped_attacks,
        "successful_attacks": successful_attacks,
        "original_accuracy": original_accuracy,
        "accuracy_under_attack": accuracy_under_attack,
        "attack_success_rate": attack_success_rate,
        "average_words_changed": average_words_changed,
        "average_perturbed_percentage": average_perc_words_perturbed
    }


def append_to_json(data, filename):
    """
    Appends data to a JSON file.

    Args:
    - data (dict): Data to append to the file.
    - filename (str): Path to the JSON file.
    """
    try:
        with open(filename, "r+") as file:
            file_data = json.load(file)
            file_data["adv_examples"].append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(filename, "w") as file:
            json.dump({"adv_examples": [data]}, file, indent=4)

def extract_attack_data(result):
    """
    Extracts relevant data from a successful attack result.

    Args:
    - result (textattack.attack_results.SuccessfulAttackResult): The result of a successful attack.

    Returns:
    - dict: A dictionary containing extracted data from the attack result.
    """
    changed_indices_original = result.original_result.attacked_text.all_words_diff(
        result.perturbed_result.attacked_text
    )
    changed_indices_perturbed = result.perturbed_result.attacked_text.all_words_diff(
        result.original_result.attacked_text
    )

    return {
        "color_string": result.__str__(color_method="ansi"),
        "original_text": result.original_text(),
        "perturbed_text": result.perturbed_text(),
        "original_words": result.original_result.attacked_text.words,
        "perturbed_words": result.perturbed_result.attacked_text.words,
        "changed_idx_orig": list(changed_indices_original),
        "changed_idx_pert": list(changed_indices_perturbed),
        "label": result.original_result.output,
        "output_orig": [
            np.argmax(result.original_result.raw_output).item(),
            max(result.original_result.raw_output).item(),
        ],
        "output_pert": [
            np.argmax(result.perturbed_result.raw_output).item(),
            max(result.perturbed_result.raw_output).item(),
        ],
    }

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_file = f"adv_datasets/{args.outfile}.json"

    n_classes = 4 if args.dataset == "ag_news" else 2
    model, tokenizer = load_model_and_tokenizer(
        args.pre_trained_model, args.max_length, args.fine_tuned_model, device, n_classes
    )
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    test_dataset = load_dataset_and_filter(args)
    attack = prepare_attack(args, model_wrapper)
    attack_results = attack_samples(test_dataset, attack)
    print(process_attack_results(attack_results, test_dataset, out_file))
