# train.py

import pandas as pd
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import random

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from utils import read_inp, process_data, evaluate


def _train(args):
    """
    run training
    :param args  : argparse namespace
    """
    print(f"Model will be saved in - {args.model_dir}")
    print(f"Path to data folder - {args.data_dir}")
    print(f"Contents of folder {args.data_dir} - {os.listdir(args.data_dir)}")

    # read input
    training_dir = args.data_dir
    pkl = [os.path.join(training_dir, file) for file in os.listdir(training_dir)][0]
    print(pkl)
    x_train, x_valid, x_test, y_train, y_valid, y_test = read_inp(pkl)

    print(
        f"x_train - {x_train.shape} , x_test - {x_test.shape} , x_val - {x_valid.shape}"
    )
    print(
        f" y_train - {y_train.shape} , y_test - {y_test.shape} , y_val - {y_valid.shape}"
    )

    dataloader_train, dataloader_val, dataloader_test, tokenizer = process_data(
        args.model_name,
        list(x_train),
        list(x_valid),
        list(x_test),
        list(y_train),
        list(y_valid),
        list(y_test),
        args.batch_size,
    )

    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train) * args.epochs,
    )

    ###########################################################################################
    ########################### Create Training Loop ##########################################
    ###########################################################################################

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    print("Training Begins -- ")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_train_total = 0

        for batch in dataloader_train:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        print(f"\nEpoch {epoch}")

        loss_train_avg = loss_train_total / len(dataloader_train)
        print(f"Training loss: {loss_train_avg}")

        val_loss, predictions, true_vals = evaluate(model, device, dataloader_val)
        val_auc = metrics.roc_auc_score(true_vals, predictions)
        print(f"Validation loss: {val_loss}")
        print(f"Validation AUC: {val_auc}")

    ###############################################################################################
    #################### Evaluating on Test Data ############################################
    ###############################################################################################

    val_loss, predictions, true_vals = evaluate(model, device, dataloader_test)
    print(predictions)

    print(f" Test AUC -- {metrics.roc_auc_score(true_vals,predictions)}")

    print(f"Saving model to -- {args.model_dir}")
    model.save_pretrained(args.model_dir)
    save_tokenizer_path = args.model_dir
    # os.mkdir(save_tokenizer_path)
    print(f"Saving tokenizer to -- {save_tokenizer_path}")
    tokenizer.save_pretrained(save_tokenizer_path)
    tokenizer._tokenizer.save(f"{save_tokenizer_path}/tokenizer.json")
    print("Training Finished -- ")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        metavar="E",
        help="bert model to train (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="BS",
        help="batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="initial learning rate (default: 1e-4)",
    )

    parser.add_argument(
        "--dist_backend",
        type=str,
        default="gloo",
        help="distributed backend (default: gloo)",
    )

    parser.add_argument("--hosts", type=json.loads, default=os.environ["SM_HOSTS"])
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])

    parser.add_argument(
        "--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    _train(parser.parse_args())


if __name__ == "__main__":
    main()
