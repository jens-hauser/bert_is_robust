import json
import random
import string

import numpy as np
import textattack
import torch
from captum.attr import LayerGradientXActivation
from datasets import load_dataset
from nltk.corpus import wordnet

random.seed(0)


def get_perturbed_batch(
    input_ids_batch,
    token_type_ids_batch,
    attention_mask_batch,
    labels_batch,
    pert_percentage,
    n_candidates,
    cos_sim_thres,
    probas,
    filter_words,
    max_length,
    model,
    tokenizer,
    embedding,
    device,
):

    attributions, all_tokens = attribution_scores_grad(
        model,
        tokenizer,
        input_ids_batch,
        token_type_ids_batch,
        attention_mask_batch,
        labels_batch,
        device,
    )
    attributions = attributions.cpu().detach().numpy()

    for i, atts in enumerate(attributions):
        word_attributions = []
        words = []
        for j, att in enumerate(atts):
            current_token = all_tokens[i][j]
            if not (current_token.startswith("##") and len(current_token) > 2):
                if current_token == "[SEP]" or current_token == "[sep]":
                    n_words = len(words) - 1
                word_attributions.append(att)
                words.append(current_token)
            else:

                word_attributions[-1] += att
                words[-1] += current_token[2:]
        new_words = words.copy()
        k = int(np.ceil(n_words * pert_percentage))

        replaced = 0
        indices_descending = np.argsort(np.abs(word_attributions)[1 : n_words + 1])[
            ::-1
        ]
        for idx in indices_descending:
            if words[idx] in filter_words:
                new_words[idx] = words[idx]
            else:
                try:
                    new_words[idx] = get_random_replacement(
                        words[idx], embedding, n_candidates, cos_sim_thres, probas
                    )
                    replaced += 1
                    if replaced == k:
                        break
                except:
                    new_words[idx] = words[idx]

        text_pert = " ".join(new_words[1 : n_words + 1])

        if i == 0:
            text_pert_encoded = tokenizer.encode_plus(
                text_pert,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            input_ids = text_pert_encoded["input_ids"]
            token_type_ids = text_pert_encoded["token_type_ids"]
            attention_mask = text_pert_encoded["attention_mask"]
        else:
            text_pert_encoded = tokenizer.encode_plus(
                text_pert,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            input_ids = torch.cat([input_ids, text_pert_encoded["input_ids"]])
            token_type_ids = torch.cat(
                [token_type_ids, text_pert_encoded["token_type_ids"]]
            )
            attention_mask = torch.cat(
                [attention_mask, text_pert_encoded["attention_mask"]]
            )

    return input_ids, token_type_ids, attention_mask


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def attribution_scores_grad(
    model, tokenizer, input_ids, token_type_ids, attention_mask, ground_truth, device
):
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    def custom_forward_func(
        inputs, token_type_ids=None, position_ids=None, attention_mask=None
    ):
        pred = model(
            inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        return pred.logits

    lig = LayerGradientXActivation(custom_forward_func, model.bert.embeddings)

    indices = input_ids.detach().tolist()
    all_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in indices]

    attributions = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(token_type_ids, position_ids, attention_mask),
        target=ground_truth,
    )

    attributions_sum = summarize_attributions(attributions)

    return attributions_sum, all_tokens


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.
    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word


def get_random_replacement(word, embedding, n_candidates, cos_sim_thres, version):

    word_id = embedding.word2index(word.lower())
    nnids = embedding.nearest_neighbours(word_id, n_candidates)
    candidate_words = dict()

    if version == "linear" or version == "quadratic":
        for nbr_id in nnids:
            nbr_word = embedding.index2word(nbr_id)
            cos_sim = embedding.get_cos_sim(nbr_word.lower(), word.lower())
            if cos_sim >= cos_sim_thres:
                candidate_words[recover_word_case(nbr_word, word)] = cos_sim

        if version == "linear":
            denominator = len(candidate_words) - sum(candidate_words.values())
            last_word = ""
            for word in candidate_words:
                candidate_words[word] = (
                    1 - candidate_words[word]
                ) / denominator + candidate_words.get(last_word, 0)
                last_word = word

        elif version == "quadratic":
            denominator_square = sum(
                (1 - np.array(list(candidate_words.values()))) ** 2
            )
            last_word = ""
            for word in candidate_words:
                candidate_words[word] = (
                    1 - candidate_words[word]
                ) ** 2 / denominator_square + candidate_words.get(last_word, 0)
                last_word = word

    score = random.uniform(0, 1)
    for candidate in candidate_words:
        if candidate_words[candidate] > score:
            replacement = candidate
            break

    return replacement


## functions needed for perturbing the input
def get_perturbed_input(
    tokenizer,
    input_ids_batch,
    pert_percentage,
    embedding,
    n_candidates,
    cos_sim_thres,
    max_length,
    filter_words,
    n_versions=8,
    version="cv",
):

    indices = input_ids_batch.detach().tolist()

    for i, in_ids in enumerate(indices):
        words = tokenizer.decode(
            in_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).split(" ")

        n_words = words.index("[SEP]") - 1
        new_words = words.copy()
        k = int(np.ceil(n_words * pert_percentage))

        for j in range(n_versions):
            replace_order = random.sample(range(1, n_words + 1), n_words)

            replaced = 0
            for idx in replace_order:
                if words[idx] in filter_words or words[idx] in string.punctuation:
                    new_words[idx] = words[idx]
                else:
                    try:
                        new_words[idx] = get_random_replacement_simple(
                            words[idx],
                            embedding,
                            n_candidates,
                            cos_sim_thres,
                            version=version,
                        )
                        replaced += 1
                        if replaced == k:
                            break
                    except:
                        new_words[idx] = words[idx]

            text_pert = " ".join(new_words[1 : n_words + 1])

            if i == 0 and j == 0:
                text_pert_encoded = tokenizer.encode_plus(
                    text_pert,
                    return_tensors="pt",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
                input_ids = text_pert_encoded["input_ids"]
                token_type_ids = text_pert_encoded["token_type_ids"]
                attention_mask = text_pert_encoded["attention_mask"]
            else:
                text_pert_encoded = tokenizer.encode_plus(
                    text_pert,
                    return_tensors="pt",
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                )
                input_ids = torch.cat([input_ids, text_pert_encoded["input_ids"]])
                token_type_ids = torch.cat(
                    [token_type_ids, text_pert_encoded["token_type_ids"]]
                )
                attention_mask = torch.cat(
                    [attention_mask, text_pert_encoded["attention_mask"]]
                )

    return input_ids, token_type_ids, attention_mask


def get_random_replacement_simple(
    word, embedding, n_candidates, cos_sim_thres, version
):
    if version == "mask":
        return "[MASK]"
    elif version == "wordnet":
        return get_replacement_wordnet(word)

    word_id = embedding.word2index(word.lower())
    nnids = embedding.nearest_neighbours(word_id, n_candidates)
    candidate_words = dict()

    for nbr_id in nnids:
        nbr_word = embedding.index2word(nbr_id)
        cos_sim = embedding.get_cos_sim(nbr_word.lower(), word.lower())
        if cos_sim > cos_sim_thres:
            candidate_words[recover_word_case(nbr_word, word)] = cos_sim

    denominator = len(candidate_words)
    last_word = ""
    for word in candidate_words:
        candidate_words[word] = 1 / denominator + candidate_words.get(last_word, 0)
        last_word = word

    score = random.uniform(0, 1)
    for candidate in candidate_words:
        if candidate_words[candidate] > score:
            replacement = candidate
            break

    return replacement


def get_replacement_wordnet(word):
    """Returns a list containing all possible words with 1 character
    replaced by a homoglyph."""
    synonyms = set()
    for syn in wordnet.synsets(word, lang="eng"):
        for syn_word in syn.lemma_names(lang="eng"):
            if (
                (syn_word.lower() != word.lower())
                and ("_" not in syn_word)
                and (textattack.shared.utils.is_one_word(syn_word))
            ):
                # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                synonyms.add(syn_word)
    syns = list(synonyms)

    return random.sample(syns, 1)[0]


def prepare_train_and_test_data(dataset, max_length, tokenizer):

    train_dataset, test_dataset = load_dataset(dataset, split=["train", "test"])
    if dataset == "yelp_polarity":
        ## we only keep text documents shorter 80 words, to keep it simpler for human evaluations
        with open("other/yelp_polarity_train_indices_shorter_80.json", "r") as openfile:
            indices = json.load(openfile)
        train_dataset = train_dataset.select(indices["indices_shorter_80"])

    train_dataset_encoded = train_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )
    test_dataset_encoded = test_dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )

    train_dataset_encoded.rename_column_("label", "labels")
    test_dataset_encoded.rename_column_("label", "labels")
    train_dataset_encoded.set_format(
        "torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    test_dataset_encoded.set_format(
        "torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
    )
    return train_dataset_encoded, test_dataset_encoded
