import argparse
import json
import random

import numpy as np
import torch
from helper_functions import get_perturbed_input
from nltk.corpus import stopwords
from textattack.shared import WordEmbedding
from transformers import BertForSequenceClassification, BertTokenizer

embedding = WordEmbedding.counterfitted_GLOVE_embedding()
random.seed(0)


parser = argparse.ArgumentParser()
data_args = parser.add_argument_group("dataset_and_attacker")
data_args.add_argument("--dataset", type=str, default="ag_news")
data_args.add_argument("--pre_trained_model", type=str, default="bert-base-uncased")
data_args.add_argument(
    "--fine_tuned_model",
    type=str,
    default="bert-base-uncased-agnews-ml128-rr04linear-ct05-ep10_12",
)
data_args.add_argument(
    "--adv_dataset_name", type=str, default="textfooler_adv_dataset_test_linear"
)
data_args.add_argument("--max_length", type=int, default=128)
data_args.add_argument("--cos_sim", type=float, default=0.5)
data_args.add_argument("--n_versions", type=int, default=8)
data_args.add_argument("--perturb_version", type=str, default="normal")
data_args.add_argument("--replace_rate", type=float, default=0.4)

args = parser.parse_args()

# set number of classes according to dataset
if args.dataset == "ag_news":
    n_classes = 4
elif args.dataset == "imdb" or args.dataset == "yelp_polarity":
    n_classes = 2

## set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## get correct model and tokenizer
model = BertForSequenceClassification.from_pretrained(
    args.pre_trained_model, num_labels=n_classes
).to(device)
tokenizer = BertTokenizer.from_pretrained(
    args.pre_trained_model, model_max_length=args.max_length, do_lower_case=True
)

## Load dataset with adversarial examples
json_file = f"adv_datasets/{args.adv_dataset_name}.json"
with open(json_file, "r") as openfile:
    # Reading from json file
    adv_dataset = json.load(openfile)

## Load model fined-tued with defense_fine_tune.py
model.load_state_dict(torch.load(f"models/{args.fine_tuned_model}.pth"))
model.eval()

# Load stopwords
with open("other/filter_words.txt", "r") as file:
    filter_words = json.load(file)
filter_words = set(filter_words).union(stopwords.words("english"))


reversed_attacks = []
l_adv = len(adv_dataset["adv_examples"])
for _ in range(10):
    correctly_classified = 0
    for i in range(l_adv):
        if i % 50 == 0:
            print(f"perturbed text {i/l_adv*100} % completed")

        text = adv_dataset["adv_examples"][i]["perturbed_text"]
        ground_truth = adv_dataset["adv_examples"][i]["label"]
        encoded_text = tokenizer.encode_plus(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = encoded_text["input_ids"].to(device)
        attention_mask = encoded_text["attention_mask"].to(device)
        token_type_ids = encoded_text["token_type_ids"].to(device)

        input_ids_pert, token_type_ids_pert, attention_mask_pert = get_perturbed_input(
            tokenizer,
            input_ids,
            args.replace_rate,
            embedding,
            50,
            args.cos_sim,
            args.max_length,
            filter_words,
            n_versions=args.n_versions,
            version=args.perturb_version,
        )

        out = model(input_ids_pert.to(device), attention_mask_pert.to(device))
        pred = torch.sum(out.logits, axis=0)

        pred = torch.argmax(pred)
        correctly_classified += int(pred == ground_truth)
        if not (pred == ground_truth):
            print(adv_dataset["adv_examples"][i]["color_string"])
            print(pred)
            print()
    reversed_attacks.append(correctly_classified / l_adv)

    print(reversed_attacks)
print(f"Mean of {l_adv} examples: {np.mean(reversed_attacks)}")
