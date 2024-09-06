import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore
from transformers import EvalPrediction  # type: ignore


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


def compute_metrics(pred: EvalPrediction):
    '''
    评估阶段的评估矩阵
    '''
    # 将logits转换为概率
    preds = torch.sigmoid(torch.tensor(pred.predictions))  # type: ignore
    preds = (preds >= 0.5).int()
    labels = torch.tensor(pred.label_ids).int()  # type: ignore
    return get_metrics(preds, labels)
