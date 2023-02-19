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
from tqdm import tqdm
from transformers import BertForSequenceClassification as BertForSequenceClassification
from transformers import BertTokenizer as BertTokenizer


def write_json(new_data, filename="sample.json"):
    with open(filename, "r+") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["adv_examples"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


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
random.seed(0)

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

args = parser.parse_args()

## writing csv file with adversarial examples. format: original text, perturbed text, original label
out_file = f"adv_datasets/{args.outfile}.json"

# set number of classes according to dataset
if args.dataset == "ag_news":
    n_classes = 4
elif args.dataset == "imdb" or args.dataset == "yelp_polarity":
    n_classes = 2

## load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(
    args.pre_trained_model, model_max_length=args.max_length, do_lower_case=True
)
model = BertForSequenceClassification.from_pretrained(
    args.pre_trained_model, num_labels=n_classes
).to(device)
model.load_state_dict(torch.load(f"models/{args.fine_tuned_model}.pth"))
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

## load dataset
test_dataset = load_dataset(args.dataset, split="test")
if args.dataset == "yelp_polarity":
    ## we only keep text documents shorter 80 words, to keep it simpler for human evaluations
    with open("other/yelp_polarity_indices_shorter_80.json", "r") as openfile:
        indices = json.load(openfile)
    test_dataset = test_dataset.select(indices["indices_shorter_80"])

## we select 1000 samples
selected = random.sample(range(len(test_dataset)), 1000)
test_dataset = test_dataset.select(selected)

custom_dataset = []
for sample in test_dataset:
    custom_dataset.append((sample["text"], sample["label"]))

## create attack
attack = attacker[args.attacker].build(model_wrapper)

## attack samples
results_iterable = attack.attack_dataset(custom_dataset)
perturbed_word_percentages = np.zeros(len(custom_dataset))
words_changed = np.zeros(len(custom_dataset))
failed_attacks = 0
skipped_attacks = 0
successful_attacks = 0
max_words_changed = 0

total_attacks = len(custom_dataset)
with tqdm(total=len(custom_dataset)) as progress_bar:
    for i, result in enumerate(results_iterable):
        # print(result.__str__(color_method='ansi'))
        if successful_attacks + failed_attacks != 0 and i % 50 == 0:
            print(i, successful_attacks / (successful_attacks + failed_attacks))
            print(result.__str__(color_method="ansi"))
        original_text, label = custom_dataset[i]
        if isinstance(result, FailedAttackResult):
            failed_attacks += 1
            progress_bar.update(1)
            continue
        elif isinstance(result, SkippedAttackResult):
            skipped_attacks += 1
            progress_bar.update(1)
            continue
        successful_attacks += 1

        ## collect all important data and store in json file
        changed_indices_original = result.original_result.attacked_text.all_words_diff(
            result.perturbed_result.attacked_text
        )
        changed_indices_perturbed = (
            result.perturbed_result.attacked_text.all_words_diff(
                result.original_result.attacked_text
            )
        )
        if successful_attacks == 1:
            data = {
                "adv_examples": [
                    {
                        "color_string": result.__str__(color_method="ansi"),
                        "original_text": result.original_text(),
                        "perturbed_text": result.perturbed_text(),
                        "original_words": result.original_result.attacked_text.words,
                        "perturbed_words": result.perturbed_result.attacked_text.words,
                        "changed_idx_orig": list(changed_indices_original),
                        "changed_idx_pert": list(changed_indices_perturbed),
                        "label": custom_dataset[i][-1],
                        "output_orig": [
                            np.argmax(result.original_result.raw_output).item(),
                            max(result.original_result.raw_output).item(),
                        ],
                        "output_pert": [
                            np.argmax(result.perturbed_result.raw_output).item(),
                            max(result.perturbed_result.raw_output).item(),
                        ],
                    }
                ]
            }
            json_object = json.dumps(data, indent=4)

            with open(out_file, "w") as outfile:
                outfile.write(json_object)

        else:
            new_data = {
                "color_string": result.__str__(color_method="ansi"),
                "original_text": result.original_text(),
                "perturbed_text": result.perturbed_text(),
                "original_words": result.original_result.attacked_text.words,
                "perturbed_words": result.perturbed_result.attacked_text.words,
                "changed_idx_orig": list(changed_indices_original),
                "changed_idx_pert": list(changed_indices_perturbed),
                "label": custom_dataset[i][-1],
                "output_orig": [
                    np.argmax(result.original_result.raw_output).item(),
                    max(result.original_result.raw_output).item(),
                ],
                "output_pert": [
                    np.argmax(result.perturbed_result.raw_output).item(),
                    max(result.perturbed_result.raw_output).item(),
                ],
            }
            try:
                write_json(new_data, filename=out_file)
            except:
                print("entered except")
                print(new_data)
                time.sleep(1)
                write_json(new_data, filename=out_file)

        num_words_changed = len(
            result.original_result.attacked_text.all_words_diff(
                result.perturbed_result.attacked_text
            )
        )
        words_changed[i] = num_words_changed
        max_words_changed = max(
            max_words_changed or num_words_changed, num_words_changed
        )
        if len(result.original_result.attacked_text.words) > 0:
            perturbed_word_percentage = (
                num_words_changed
                * 100.0
                / len(result.original_result.attacked_text.words)
            )
        else:
            perturbed_word_percentage = 0
        perturbed_word_percentages[i] = perturbed_word_percentage
        progress_bar.update(1)

# Original classifier success rate on these samples.
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
average_perc_words_perturbed = perturbed_word_percentages.mean()
average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"

words_changed = words_changed[words_changed > 0]
average_words_changed = words_changed.mean()
average_words_changed = str(round(average_words_changed, 2)) + "%"

summary_table_rows = [
    ["Number of successful attacks:", str(successful_attacks)],
    ["Number of failed attacks:", str(failed_attacks)],
    ["Number of skipped attacks:", str(skipped_attacks)],
    ["Original accuracy:", original_accuracy],
    ["Accuracy under attack:", accuracy_under_attack],
    ["Attack success rate:", attack_success_rate],
    ["Average perturbed word %:", average_perc_words_perturbed],
    ["Average num. words changed:", average_words_changed],
]
print(summary_table_rows)
