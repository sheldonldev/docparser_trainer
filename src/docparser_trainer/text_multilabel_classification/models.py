import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel  # type: ignore


class MultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = BertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout else 0.3,
        )
        self.classifier = nn.Linear(
            self.roberta.config.hidden_size, len(config.id2label)  # type: ignore
        )
        self.post_init()

    def forward(self, input_ids, attention_mask, labels=None, *args, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # type: ignore
        cls_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            if self.config.label_weights is None:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(self.config.label_weights).to(self.device)
                )  # 这个损失函数会在内部应用 sigmoid，不需要显式包含 sigmoid 函数在前向函数里。
            loss = self.criterion(logits.to(self.dtype), labels.to(self.dtype))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output
