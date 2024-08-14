import torch
import torch.nn as nn
import torch.optim as optim
from docparser_models._model_interface.model_manager import load_tokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore
from transformers import (  # type: ignore
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    get_scheduler,
)
from util_common.path import move

from docparser_trainer._cfg import setup_env
from docparser_trainer.text_multilabel_classification.data import (
    MultiLabelTextDataset,
    class_weights,
    load_data,
)
from docparser_trainer.text_multilabel_classification.models import MultiLabelClassifier

setup_env()


class MultiLabelClassifierTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        训练阶段的损失函数
        '''
        labels = inputs.pop('labels')
        outputs = model(**inputs)

        criterion = (
            nn.BCEWithLogitsLoss()
        )  # 这个损失函数会在内部应用 sigmoid，不需要显式包含 sigmoid 函数在前向函数里。

        loss = criterion(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, **kwargs):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            logits = model(**inputs)

            labels = inputs.get("labels")
            if prediction_loss_only:
                return (None, None, None)

            loss = None
            if labels is not None:
                loss = self.compute_loss(model, inputs)

        return (loss, logits, labels)


def get_metrics(preds, labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def compute_metrics(pred: EvalPrediction, compute_result=False):
    '''
    评估阶段的评估矩阵
    '''
    if compute_result is True:
        # 将logits转换为概率
        preds = torch.sigmoid(pred.predictions)  # type: ignore
        preds = (preds >= 0.5).int()
        labels = pred.label_ids.int()  # type: ignore
        return get_metrics(preds, labels)
    else:
        return {}


model_id = 'schen/longformer-chinese-base-4096'
tokenizer = load_tokenizer(model_id)
train_texts, train_labels, dev_texts, dev_labels = load_data()

chunk_size = 2048
train_dataset = MultiLabelTextDataset(
    train_texts,
    train_labels,
    tokenizer,
    max_length=chunk_size,
)
dev_dataset = MultiLabelTextDataset(
    dev_texts,
    dev_labels,
    tokenizer,
    max_length=chunk_size,
)


def train():

    model_dir = (
        '/home/sheldon/repos/docparser_trainer/src/docparser_trainer/'
        'text_multilabel_classification/pretrained/longformer-customs_declaration'
    )

    # model_dir, tokenizer = load_model_and_tokenizer(model_id)  # type: ignore

    class_num = len(train_labels[0])
    model = MultiLabelClassifier(model_dir, class_num)  # type: ignore
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

    training_args = TrainingArguments(
        output_dir="./results1",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        remove_unused_columns=False,
        label_names=['labels'],
        batch_eval_metrics=True,  # 会在每个批次评估期间调用compute_metrics函数
    )

    num_training_steps = int(
        len(train_dataset)
        // training_args.per_device_train_batch_size
        * training_args.num_train_epochs
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    trainer = MultiLabelClassifierTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=train_dataset.collate_fn,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.optimizer = optimizer

    trainer.train()


def torch_train(stop_thresh=0.999):
    from pathlib import Path

    from docparser_models._model_interface.model_manager import get_model_dir
    from torch.utils.data import DataLoader
    from util_common.datetime import format_now
    from util_common.path import ensure_parent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = Path(__file__).parent.joinpath('pretrained').joinpath("model.ckpt")
    model_dir = get_model_dir(model_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn,
    )

    def load_checkpoint(save_path: Path, model):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=1e-5,
        )
        epoch = 0
        accuracy = 0
        if save_path.exists():
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            accuracy = checkpoint['accuracy']
        return model, optimizer, epoch, accuracy

    def _save_checkpoint(
        model, optimizer, epoch: int, accuracy: float, save_path: Path, old_acc: float
    ):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
        }
        if save_path.exists():
            back_path = save_path.with_suffix(f'.{format_now()}-{old_acc:4f}')
            move(save_path, back_path)
        ensure_parent(save_path)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at {save_path}")

    def save_checkpoint(model, optimizer, epoch: int, accuracy: float, save_path: Path):
        _, _, _, old_acc = load_checkpoint(save_path, model)
        # if accuracy > old_acc:
        _save_checkpoint(model, optimizer, epoch, accuracy, save_path, old_acc)

    def to_device(model, optimizer):
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weights).to(device)
    )  # 这个损失函数会在内部应用 sigmoid，不需要显式包含 sigmoid 函数在前向函数里。

    def train_model(model, optimizer, dataloader, epoch):
        to_device(model, optimizer)
        model.train()

        total_loss = 0
        for i, batch in enumerate(dataloader):
            inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch} Batch {i}, Loss: {loss.item()}")

        print(f"Epoch {epoch} Total Loss: {total_loss}")
        epoch += 1
        return epoch

    def evaluate_model(model, dataloader):
        model.eval()  # Set the model to evaluation mode

        all_preds = []
        all_labels = []
        with torch.no_grad():  # No need to track gradients for evaluation
            for batch in dataloader:
                inputs = {key: batch[key].to(device) for key in ['input_ids', 'attention_mask']}
                labels = batch['labels'].to(device)

                logits = model(**inputs)

                predictions = (torch.sigmoid(logits) >= 0.5).int()
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = get_metrics(all_preds, all_labels)
        print(metrics)
        return metrics['accuracy']

    model = MultiLabelClassifier(model_dir, len(train_labels[0]))  # type: ignore
    while True:
        model, optimizer, epoch, accuracy = load_checkpoint(save_path, model)
        if accuracy > stop_thresh:
            break

        epoch = train_model(model, optimizer, train_dataloader, epoch)
        accuracy = evaluate_model(model, dev_dataloader)
        save_checkpoint(model, optimizer, epoch, accuracy, save_path=save_path)


if __name__ == '__main__':
    torch_train()
