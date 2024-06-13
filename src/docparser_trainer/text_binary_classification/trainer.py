from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer  # type: ignore
from util_common.decorator import proxy

from docparser_trainer._cfg import setup_env
from docparser_trainer.text_binary_classification.data import load_data
from docparser_trainer.text_binary_classification.model_interface import (
    load_checkpoint,
    save_checkpoint,
)

setup_env()


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {
        key: torch.stack([item[key] for item in batch], dim=0) for key in keys
    }
    return collated_batch


def train_model(model, optimizer, dataloader, epoch):
    model.train()
    for batch in dataloader:
        inputs = {key: batch[key] for key in ['input_ids', 'attention_mask']}
        labels = batch['labels']

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    epoch += 1
    return epoch


def evaluate_model(model, val_dataset, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0

    with torch.no_grad():  # No need to track gradients for evaluation
        for batch in dataloader:
            inputs = {
                key: batch[key] for key in ['input_ids', 'attention_mask']
            }
            labels = batch['labels']

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = outputs.logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(val_dataset)

    print(f"Accuracy: {accuracy}, Loss: {avg_loss}")
    return avg_loss, accuracy


@proxy(http_proxy='127.0.0.1:7890', https_proxy='127.0.0.1:7890')
def train(
    model_name: str,
    positive_tags: List[str],
    stop_thresh: float = 0.995,
):
    save_path = (
        Path(__file__)
        .parent.joinpath('pretrained')
        .joinpath(f"{model_name}.ckpt")
    )

    train_texts, train_labels, val_texts, val_labels = load_data(
        positive_tags=positive_tags,
    )
    # train_texts = ['测试1', '测试2']
    # train_labels = [1, 0]
    # val_texts = ['测试1', '测试2']
    # val_labels = [1, 0]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    train_encodings = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt',
    )
    val_encodings = tokenizer(
        val_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt',
    )

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn
    )

    while True:
        model, optimizer, epoch = load_checkpoint(save_path)
        epoch = train_model(model, optimizer, train_dataloader, epoch)
        avg_loss, accuracy = evaluate_model(model, val_dataset, val_dataloader)
        save_checkpoint(model, optimizer, epoch, accuracy, save_path=save_path)
        if accuracy > stop_thresh:
            break


if __name__ == '__main__':
    train(
        model_name='customs_declaration',
        positive_tags=['报关单'],
        stop_thresh=0.998,
    )
    train(
        model_name='customs_declaration_contract',
        positive_tags=['合同'],
    )
    train(
        model_name='customs_declaration_elements',
        positive_tags=['申报要素'],
    )
    train(
        model_name='customs_declaration_packinglist',
        positive_tags=['装箱单'],
    )
    train(
        model_name='customs_declaration_invoice',
        positive_tags=['发票'],
    )
