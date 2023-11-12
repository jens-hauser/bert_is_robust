import argparse
import json
import random
import numpy as np
import torch
from helper_functions import get_perturbed_input
from nltk.corpus import stopwords
from textattack.shared import WordEmbedding
from transformers import BertForSequenceClassification, BertTokenizer

parser = argparse.ArgumentParser()
data_args = parser.add_argument_group("dataset_and_attacker")
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


def load_model_and_tokenizer(pre_trained_model, fine_tuned_model, max_length, num_labels, device):
    """
    Loads a pre-trained BERT model and tokenizer.

    Args:
    - pre_trained_model (str): Name of the pre-trained BERT model.
    - fine_tuned_model (str): Name of the fine-tuned BERT model.
    - max_length (int): Maximum sequence length for the tokenizer.
    - num_labels (int): Number of labels for classification.
    - device (torch.device): Device to load the model on.

    Returns:
    - tuple: Loaded model and tokenizer.
    """
    model = BertForSequenceClassification.from_pretrained(pre_trained_model, num_labels=num_labels).to(device)
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model, model_max_length=max_length, do_lower_case=True)
    model.load_state_dict(torch.load(f"models/{fine_tuned_model}.pth"))
    model.eval()
    return model, tokenizer

def load_adv_dataset(adv_dataset_name):
    """
    Loads an adversarial dataset from a JSON file.

    Args:
    - adv_dataset_name (str): Filename of the adversarial dataset.

    Returns:
    - dict: Loaded adversarial dataset.
    """
    json_file = f"adv_datasets/{adv_dataset_name}.json"
    with open(json_file, "r") as openfile:
        return json.load(openfile)

def load_filter_words():
    """
    Loads a set of filter words, combining custom filter words and stopwords.

    Returns:
    - set: A set of filter words.
    """
    with open("other/filter_words.txt", "r") as file:
        return set(json.load(file)).union(stopwords.words("english"))

def perform_reverse_attacks(adv_dataset, args, model, tokenizer, embedding, device, filter_words):
    """
    Performs reverse attacks on an adversarial dataset to evaluate model robustness.

    Args:
    - adv_dataset (dict): Adversarial dataset to evaluate.
    - args (argparse.Namespace): Parsed command line arguments.
    - model (torch.nn.Module): Trained BERT model.
    - tokenizer (transformers.BertTokenizer): BERT tokenizer.
    - embedding (textattack.shared.WordEmbedding): Word embeddings used for perturbations.
    - device (torch.device): Device to run the model on.
    - filter_words (set): Set of words to filter out from perturbations.

    Returns:
    - list: Proportion of correctly classified examples in each reverse attack iteration.
    """
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

            input_ids_pert, _, attention_mask_pert = get_perturbed_input(
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
    return reversed_attacks

def main():
    """
    Main function to execute the script. It parses arguments, loads the necessary models and datasets, 
    and performs reverse attacks to evaluate model robustness.
    """
    args = parser.parse_args()
    embedding = WordEmbedding.counterfitted_GLOVE_embedding()
    random.seed(0)

    n_classes = 4 if args.dataset == "ag_news" else 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.pre_trained_model, args.fine_tuned_model, args.max_length, n_classes, device)
    adv_dataset = load_adv_dataset(args.adv_dataset_name)
    filter_words = load_filter_words()

    reversed_attacks = perform_reverse_attacks(adv_dataset, args, model, tokenizer, embedding, device, filter_words)
    print(f"Mean of {len(adv_dataset['adv_examples'])} examples: {np.mean(reversed_attacks)}")

if __name__ == "__main__":
    main()