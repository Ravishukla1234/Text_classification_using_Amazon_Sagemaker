# model.py
import transformers
from transformers import BertModel

import torch.nn as nn

import json


class TextClassification(nn.Module):
    def __init__(self):
        super(TextClassification, self).__init__()
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased",
            return_dict=False,
        )
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
