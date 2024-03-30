#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import argparse
from datasets import load_metric
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Download NLTK resources
nltk.download('punkt')

# Load evaluation metrics
rouge_score = load_metric("rouge")
bleu_score = load_metric("sacrebleu")

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define evaluation metrics
METRICS = [
    {
        'metric_func': bleu_score.compute,
        'metric_name': 'bleu',
    },
    {
        'metric_func': rouge_score.compute,
        'metric_name': 'rouge',
    },
    {
        'metric_func': cosine_similarity,
        'metric_name': 'cosine_similarity',
    }
]

# Define perturbation models
MODELS = [
    'augly',
    'textflint',
    'perturber',
]

def compute_bleu(predictions, references):
    return bleu_score.compute(predictions=[predictions], references=[[references]], lowercase=True)['score']

def compute_rouge(predictions, references):
    result = rouge_score.compute(predictions=[predictions], references=[references])
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

def compute_cosine_similarity(predictions, references):
    # Tokenize inputs
    tokenized_input = tokenizer(references, predictions, return_tensors='pt', padding=True, truncation=True)
    # Compute BERT embeddings
    with torch.no_grad():
        outputs = bert_model(**tokenized_input)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    # Compute cosine similarity
    similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return similarity_score

def compute_metrics(df, metric_func, output_name, model):
    prediction_name = f"{model}_rewrite"
    df[output_name] = df.apply(lambda x: metric_func(predictions=x[prediction_name], references=x["annotator_rewrite"]), axis=1)
    return df

def compute_outputs(df):
    for model in MODELS:
        for metric in METRICS:
            metric_func = metric['metric_func']
            metric_name = metric['metric_name']
            output_name = f"{metric_name}_{model}"
            compute_metrics(df, metric_func, output_name, model)

def print_outputs(df):
    for metric in METRICS:
        metric_name = metric['metric_name']
        print(f"\n{'*' * 20} {metric_name} scores {'*' * 20}\n")
        for model in MODELS:
            output_name = f"{metric_name}_{model}"
            print(f"{model}: {df[output_name].mean()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True, help="Path to results of different perturbation methods")
    args = parser.parse_args()

    # Read CSV file
    generated_outputs = pd.read_csv(args.csv_file, sep="|")

    # Compute metrics
    compute_outputs(generated_outputs)

    # Print metrics
    print_outputs(generated_outputs)

    # Save results to CSV
    generated_outputs.to_csv(args.csv_file, sep="|", index=False)

if __name__ == "__main__":
    main()
