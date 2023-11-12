import argparse
import datetime
import json
import random
import time

import torch
from nltk.corpus import stopwords
from textattack.shared import WordEmbedding
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from helper_functions import get_perturbed_batch, prepare_train_and_test_data

# Setup parser for command line arguments
def setup_parser():
    """
    Sets up the argument parser with necessary arguments for dataset and training parameters.

    Returns:
    - argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser()
    data_args = parser.add_argument_group("dataset")
    data_args.add_argument("--dataset", type=str, default="ag_news")
    data_args.add_argument("--pre_trained_model", type=str, default="bert-base-uncased")
    data_args.add_argument("--batch_size", type=int, default=32)
    data_args.add_argument("--n_epochs", type=int, default=12)
    data_args.add_argument("--max_length", type=int, default=128)
    data_args.add_argument("--probas", type=str, default="linear")
    data_args.add_argument("--replace_rate", type=float, default=0.4)
    data_args.add_argument("--cos_sim", type=float, default=0.5)
    data_args.add_argument("--stop_words", type=str, default="all")
    return parser


def load_filter_words(args):
    """
    Loads a set of filter words based on specified stopwords strategy.

    Args:
    - args (argparse.Namespace): Parsed command line arguments.

    Returns:
    - set: A set of filter words.
    """
    # Load stopwords
    with open("other/filter_words.txt", "r") as file:
        filter_words = json.load(file)

    if args.stop_words == "textfooler":
        filter_words = set(filter_words)
    else:
        filter_words = set(filter_words).union(stopwords.words("english"))

    return filter_words


def train_model(
    args,
    train_loader,
    model,
    device,
    optim,
    scheduler,
    embedding,
    tokenizer,
    filter_words,
):
    """
    Custom training loop for the model.

    Args:
    - args (argparse.Namespace): Parsed command line arguments.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - model (torch.nn.Module): The BERT model for sequence classification.
    - device (torch.device): Device to train the model on.
    - optim (torch.optim.Optimizer): Optimizer for training.
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    - embedding (textattack.shared.WordEmbedding): Pre-loaded word embeddings.
    - tokenizer (transformers.BertTokenizer): Tokenizer for text processing.
    - filter_words (set): Set of words to be filtered during training.

    Returns:
    - torch.nn.Module: Trained model.
    """
    model.train()
    for epoch in range(args.n_epochs):
        counter = 0
        for batch in train_loader:
            optim.zero_grad()
            input_ids, attention_mask, token_type_ids, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                batch["labels"].to(device),
            )
            (
                input_ids_pert,
                token_type_ids_pert,
                attention_mask_pert,
            ) = get_perturbed_batch(
                input_ids,
                token_type_ids,
                attention_mask,
                labels.reshape(-1),
                args.replace_rate,
                50,
                args.cos_sim,
                probas=args.probas,
                filter_words=filter_words,
                max_length=args.max_length,
                model=model,
                tokenizer=tokenizer,
                embedding=embedding,
                device=device,
            )
            input_ids = torch.cat([input_ids, input_ids_pert.to(device)])
            attention_mask = torch.cat([attention_mask, attention_mask_pert.to(device)])
            token_type_ids = torch.cat([token_type_ids, token_type_ids_pert.to(device)])
            labels = torch.cat([labels, labels])

            outputs = model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs[0]
            loss.backward()
            optim.step()
            scheduler.step()
            counter += 1
            if counter % 100 == 0:
                print(
                    "Epoch",
                    epoch,
                    "progress:",
                    counter * args.batch_size / len(train_loader.dataset) * 100,
                    "%",
                )
                print("Loss:", loss.item())

    return model


def main():
    args = setup_parser().parse_args()
    random.seed(0)

    # Start time
    s_time = time.time()

    # Load stopwords and embeddings
    filter_words = load_filter_words(args)
    embedding = WordEmbedding.counterfitted_GLOVE_embedding()

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(
        args.pre_trained_model, model_max_length=args.max_length, do_lower_case=True
    )
    n_classes = 4 if args.dataset == "ag_news" else 2
    model = BertForSequenceClassification.from_pretrained(
        args.pre_trained_model, num_labels=n_classes
    )

    # Load and prepare datasets
    train_dataset_encoded, _ = prepare_train_and_test_data(
        args.dataset, args.max_length, tokenizer
    )
    train_loader = DataLoader(
        train_dataset_encoded, batch_size=args.batch_size, shuffle=True
    )

    # Set up optimizer and scheduler
    optim = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optim, len(train_loader) * 0.5, len(train_loader) * args.n_epochs
    )

    # Training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Finetuning using:", device)
    model.to(device)
    trained_model = train_model(
        args,
        train_loader,
        model,
        device,
        optim,
        scheduler,
        embedding,
        tokenizer,
        filter_words,
    )

    # Save final model
    model_save_path = (
        f"models/{args.pre_trained_model}-"
        f"{args.dataset.replace('_', '')}-"
        f"ml{args.max_length}-"
        f"rr{str(args.replace_rate).replace('.', '') + args.probas}-"
        f"ct{str(args.cos_sim).replace('.', '')}-"
        f"ep{args.n_epochs}.pth"
    )
    torch.save(trained_model.state_dict(), model_save_path)

    end_time = time.time()
    print("Total time:", str(datetime.timedelta(seconds=int(end_time - s_time))))


if __name__ == "__main__":
    main()
