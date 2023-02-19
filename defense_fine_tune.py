import argparse
import datetime
import json
import random
import time

import torch
from helper_functions import get_perturbed_batch, prepare_train_and_test_data
from nltk.corpus import stopwords
from textattack.shared import WordEmbedding
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

random.seed(0)

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
args = parser.parse_args()

# Load stopwords
with open("other/filter_words.txt", "r") as file:
    filter_words = json.load(file)

if args.stop_words == "textfooler":
    filter_words = set(filter_words)
else:
    filter_words = set(filter_words).union(stopwords.words("english"))

# Load counterfitted word vectors
embedding = WordEmbedding.counterfitted_GLOVE_embedding()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(
    args.pre_trained_model, model_max_length=args.max_length, do_lower_case=True
)

if args.dataset == "ag_news":
    n_classes = 4
elif args.dataset == "imdb" or args.dataset == "yelp_polarity":
    n_classes = 2

model = BertForSequenceClassification.from_pretrained(
    args.pre_trained_model, num_labels=n_classes
)

# Load and prepare datasets
train_dataset_encoded, test_dataset_encoded = prepare_train_and_test_data(
    args.dataset, args.max_length, tokenizer
)
train_loader = DataLoader(
    train_dataset_encoded, batch_size=args.batch_size, shuffle=True
)

# Parameters for training
optim = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optim, len(train_loader) * 0.5, len(train_loader) * args.n_epochs
)

# Custom training loop
s_time = time.time()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Finetuning using: ", device)
model.to(device)
model.train()
for epoch in range(args.n_epochs):
    counter = 0
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        input_ids_pert, token_type_ids_pert, attention_mask_pert = get_perturbed_batch(
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
                "Epoch  ",
                epoch,
                "progress: ",
                counter * args.batch_size / (len(train_dataset_encoded)) * 100,
                "%",
            )
            print("Loss: ", loss)

    model.eval()
    torch.save(
        model.state_dict(),
        f"models/{args.pre_trained_model}-{args.dataset.replace('_', '')}-ml{args.max_length}-rr{str(args.replace_rate).replace('.', '') + args.probas}-ct{str(args.cos_sim).replace('.', '')}-ep{str(epoch + 1) +'_' + str(args.n_epochs)}.pth",
    )
    model.train()
end_time = time.time()
print("Total time: ", str(datetime.timedelta(seconds=int(end_time - s_time))))
