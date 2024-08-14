import random
from typing import List

import torch
from torch.utils.data import Dataset
from util_intelligence.regex import process_spaces

from docparser_trainer._augment.augment_text import func_list
from docparser_trainer._interface.read_customs_declaration import (
    yield_customs_declaration_data_classification_text,
)


def preprocess_data(name: str, content: str):
    text = process_spaces(content)
    if random.random() < 0.5:
        name = ''
    text = f"{name}\n{content}".strip()
    return text


label_settings = [
    ['出口报关单'],
    ['报关发票'],
    ['报关装箱单'],
    ['报关合同'],
    ['申报要素'],
]
class_weights = [
    0.2,
    0.2,
    0.2,
    0.5,
    0.2,
]


def load_data():
    batch_names: List[str] = [
        '20240301-0',
        '20240520-0',
        '20240520-1',
        '20240520-2',
        '20240520-3',
        '20240520-4',
    ]

    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []

    def get_label(tags):
        label = [0] * len(label_settings)
        for i, labels in enumerate(label_settings):
            for tag in tags:
                if tag in labels:
                    label[i] = 1
        return label

    for i, (
        name,
        text_path,
        tags,
        is_eval,
    ) in enumerate(yield_customs_declaration_data_classification_text(batch_names=batch_names)):
        # if i > 64:
        #     break
        text = preprocess_data(name, text_path.read_text())
        if is_eval:
            val_texts.append(text)
            val_labels.append(get_label(tags))
        else:
            train_texts.append(text)
            train_labels.append(get_label(tags))

    print('>>>', i, 'samples')
    return train_texts, train_labels, val_texts, val_labels


class MultiLabelTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def augment_text(self, text):
        for f in func_list:
            text = f(text, prob=0.5)
        return text

    def __getitem__(self, idx):
        text = self.augment_text(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        labels = torch.tensor(labels, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    @staticmethod
    def collate_fn(batch):
        keys = batch[0].keys()
        collated_batch = {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}
        return collated_batch
