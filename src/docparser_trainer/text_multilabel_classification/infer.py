import torch

CLASSIFICATION_MAP = {
    'customs_declaration': [
        '出口报关单',
        '报关发票',
        '报关装箱单',
        '报关合同',
        '申报要素',
    ]
}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_classification_map(ckpt_version: str):
    return CLASSIFICATION_MAP[ckpt_version]


def predict(
    model,
    tokenizer,
    text: str,
    ckpt_version: str,
):
    classification_map = get_classification_map(ckpt_version)
    encodings = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt',
    )
    with torch.no_grad():
        outputs = model(
            encodings['input_ids'].to(device),
            encodings['attention_mask'].to(device),
        )
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.5).int()[0]
        result = [classification_map[i] for i, v in enumerate(predictions) if v == 1]
        return result
