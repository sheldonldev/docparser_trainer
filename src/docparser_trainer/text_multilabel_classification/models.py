from pathlib import Path

import torch.nn as nn
from transformers import AutoModel  # type: ignore


class MultiLabelClassifier(nn.Module):
    def __init__(self, model_dir: Path, class_num: int):
        super(MultiLabelClassifier, self).__init__()

        self.roberta = AutoModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(
            self.roberta.config.hidden_size,  # type: ignore
            class_num,
        )

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # type: ignore
        cls_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(cls_output)
        return logits
