# utils.py

import pickle
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np


def read_inp(pkl):
    """
    Read the input dataset
    :param pkl - pickle file containing input data
    return train, val and test data
    """
    _temp_ = pickle.load(open(pkl, "rb"))
    x_train = _temp_["x_train"]
    x_valid = _temp_["x_valid"]
    y_train = _temp_["y_train"]
    y_valid = _temp_["y_valid"]
    x_test = _temp_["x_test"]
    y_test = _temp_["y_test"]

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def process_data(
    model_name, x_train, x_valid, x_test, y_train, y_valid, y_test, batch_size
):
    """
    process and tokeninze the data
    :param model_name - bert model name to train
    :param x_train    - list of training texts
    :param x_valid    - list of validation texts
    :param x_test     - list of test texts
    :param y_train    - list of training labels
    :param y_valid    - list of validation labels
    :param y_test     - list of test labels
    :param batch_size - batch size to load data
    return dataloader train, valid and test with tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

    encoded_data_val = tokenizer.batch_encode_plus(
        x_valid,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded_data_train = tokenizer.batch_encode_plus(
        x_train,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded_data_test = tokenizer.batch_encode_plus(
        x_test,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors="pt",
    )

    input_ids_train = encoded_data_train["input_ids"]
    attention_masks_train = encoded_data_train["attention_mask"]
    labels_train = torch.tensor(y_train)

    input_ids_val = encoded_data_val["input_ids"]
    attention_masks_val = encoded_data_val["attention_mask"]
    labels_val = torch.tensor(y_valid)

    input_ids_test = encoded_data_test["input_ids"]
    attention_masks_test = encoded_data_test["attention_mask"]
    labels_test = torch.tensor(y_test)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    print(
        f"dataset_train length--  {len(dataset_train)} , dataset_val length-- {len(dataset_val)},dataset_test length-- {len(dataset_test)}"
    )

    dataloader_train = DataLoader(
        dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size
    )

    dataloader_val = DataLoader(
        dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size
    )

    dataloader_test = DataLoader(
        dataset_test, sampler=RandomSampler(dataset_test), batch_size=batch_size
    )
    return dataloader_train, dataloader_val, dataloader_test, tokenizer


def evaluate(model, device, dataloader_val):
    """
    Evaluation loop
    :param model          - bert model
    :param device         - cpu or cuda
    :param dataloader_val - validation data loader
    return average loss, predictions and true values
    """

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs["labels"].cpu().numpy()
        # convert outputs to cpu and extend the final list

        logits = torch.from_numpy(logits[:, 1])
        outputs = torch.sigmoid(logits).cpu().detach()
        predictions.append(outputs.numpy().tolist())

        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals
