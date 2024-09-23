from datasets import Dataset  # type: ignore
from util_intelligence.regex import process_spaces

from docparser_trainer._interface.read_customs_declaration import (
    yield_customs_declaration_data_classification_text,
)


def preprocess_data(name: str, content: str):
    text = process_spaces(content)
    text = f"{name}\n{content}".strip()
    return text


label_settings = [
    ['出口报关单'],
    ['报关发票'],
    ['报关装箱单'],
    ['报关合同'],
    ['申报要素'],
]


def load_data():
    batch_names: list[str] = [
        '20240301-0',
        '20240520-0',
        '20240520-1',
        '20240520-2',
        '20240520-3',
        '20240520-4',
    ]
    train_texts = []
    train_labels = []
    eval_texts = []
    eval_labels = []

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
    ) in enumerate(
        yield_customs_declaration_data_classification_text(batch_names=batch_names),
    ):
        text = preprocess_data(name, text_path.read_text())
        if is_eval:
            eval_texts.append(text)
            eval_labels.append(get_label(tags))
        else:
            train_texts.append(text)
            train_labels.append(get_label(tags))

    print('>>>', i, 'samples')
    return train_texts, train_labels, eval_texts, eval_labels


def preprocess_datasets():
    train_texts, train_labels, eval_texts, eval_labels = load_data()

    train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    eval_dataset = Dataset.from_dict({"text": eval_texts, "labels": eval_labels})
    return train_dataset, eval_dataset
