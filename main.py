import kagglehub
import os
import csv
from spellchecker import SpellChecker
import nltk
import statistics

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



# os.environ["KAGGLEHUB_CACHE"] = '/nfs/stak/users/mettas/ondemand/AI539_HPC/Datasets'
# path = kagglehub.dataset_download("lburleigh/asap-2-0")

def dataset_spliter(path_to_dataset, train_path, valid_path, test_path):
    with open(path_to_dataset, newline='') as dataset:

        reader = csv.DictReader(dataset)

        with open(train_path, 'w', newline='') as train, open(valid_path, 'w', newline='') as valid, open(test_path, 'w', newline='') as test:
            train_writer = csv.DictWriter(train, fieldnames=reader.fieldnames)
            valid_writer = csv.DictWriter(valid, fieldnames=reader.fieldnames)
            test_writer = csv.DictWriter(test, fieldnames=reader.fieldnames)

            train_writer.writeheader()
            valid_writer.writeheader()
            test_writer.writeheader()

            for num, row in enumerate(reader):
                if num < 24728 * 0.6:
                    train_writer.writerow(row)
                elif num < 24728 * 0.8:
                    valid_writer.writerow(row)
                else:
                    test_writer.writerow(row)

def misSpell_counter(text):
    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)

    corrected_words = [spell.correction(w) if w in misspelled else w for w in words]
    return len(misspelled), " ".join(corrected_words)

def preprocess_essay(text):
    error_count, cleaned_text = misSpell_counter(text)
    tokens = nltk.word_tokenize(cleaned_text)
    shallow_features['spelling_errors'] = error_count
    words = text.split()
    sents = nltk.sent_tokenize(text)
    shallow_features['num_words'] = len(words)
    shallow_features['num_sentences'] = len(sents)
    shallow_features['num_sentence_length'] = shallow_features['num_words'] / shallow_features['num_sentences']
    lengths = [len(nltk.word_tokenize(sent)) for sent in sentences]
    shallow_features['sentence_variance'] = statistics.variance(lengths) if len(lengths) > 1 else 0.0
    
    shallow_features['num_characters'] = 0
    shallow_features['num_nouns'] = 0
    shallow_features['num_verbs'] = 0
    shallow_features['num_adverbs'] = 0
    shallow_features['num_conjunctions'] = 0
    shallow_features['num_adjectives'] = 0

    distinct_words = {}
    for word in words:
        shallow_features['num_characters'] += len(word)
    shallow_features['mean_wordLength'] = shallow_features['num_characters'] / shallow_features['num_words']
    tagged = nltk.pos_tag(words, tagset='universal')
    for t in tagged:
        if t[1] == 'NOUN':
            shallow_features['num_nouns'] += 1
        elif t[1] == 'VERB':
            shallow_features['num_verbs'] += 1
        elif t[1] == 'ADV':
            shallow_features['num_adverbs'] += 1
        elif t[1] == 'CONJ':
            shallow_features['num_conjunctions'] += 1
        elif t[1] == 'ADJ':
            shallow_features['num_adjectives'] += 1
        distinct_words.add(t[0])
    shallow_features["distinct_words"] = len(distinct_words)

    return cleaned_text, shallow_features

if __name__ == "__main__":
    path_to_dataset = "/nfs/stak/users/mettas/ondemand/AI539_HPC/Datasets/ASAP/ASAP2_train_sourcetexts.csv"
    train_path = "/nfs/stak/users/mettas/ondemand/AI539_HPC/Datasets/ASAP/Train.csv"
    valid_path = "/nfs/stak/users/mettas/ondemand/AI539_HPC/Datasets/ASAP/Valid.csv"
    test_path = "/nfs/stak/users/mettas/ondemand/AI539_HPC/Datasets/ASAP/Test.csv"

    shallow_features

    # with open(path_to_dataset, newline='') as dataset:
    #     reader = csv.DictReader(dataset)
    #     for row in reader:
    #         print(row['prompt_name'])

    # dataset_spliter(path_to_dataset, train_path, valid_path, test_path)
