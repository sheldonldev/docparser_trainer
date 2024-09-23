from pathlib import Path
from typing import Dict, List

import numpy as np
from docparser_datasets.customs_declaration.df_parser import get_classification_tag
from util_common.decorator import ticktock
from util_intelligence.char_util import normalize_char_text
from util_intelligence.regex import process_spaces

from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer.text_multilabel_classification.infer import batch_infer
from docparser_trainer.text_multilabel_classification.metrics import get_metrics
from docparser_trainer.text_multilabel_classification.models import MultiLabelClassifier


def tags_to_multilabel_vector(tags: List[str], label_mapping: Dict[int, str]) -> List[int]:
    vector = np.zeros(len(label_mapping), dtype=int)
    label_to_index = {label: index for index, label in label_mapping.items()}

    for tag in tags:
        if tag in label_to_index:
            vector[label_to_index[tag]] = 1

    return vector.tolist()


def batch_tags_to_multilabel_vectors(label_mapping, batch_tags):
    return [tags_to_multilabel_vector(tags, label_mapping) for tags in batch_tags]


@ticktock()
def batch_test():
    model_id = 'schen/longformer-chinese-base-4096'
    ckpt_dir = Path(
        '/home/sheldon/repos/docparser_trainer/checkpoints/text_multilabel_classification/'
        'customs_declaration_5_labels/trial_1'  # 可修改 ckpt
    )
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        ckpt_dir=ckpt_dir,
        model_cls=MultiLabelClassifier,
    )

    label_names = [v for k, v in model.config.id2label.items()]
    data_dir = Path(
        '/home/sheldon/repos/docparser_datasets/data/customs_declaration/20240301-0-test/unified'
    )
    class_tag, _ = get_classification_tag(
        Path('/home/sheldon/repos/docparser_datasets/data/customs_declaration/20240301-0')
    )

    correct_count = 0
    total_count = 0
    for type_dir in data_dir.iterdir():
        file_type = type_dir.name
        texts = []
        labels = []
        for sample_dir in type_dir.iterdir():
            sample_name = sample_dir.name
            classification_record = class_tag[
                (class_tag['name'] == sample_name) & (class_tag['file_type'] == file_type)
            ]
            text_path = sample_dir.joinpath('pure.txt')
            if not classification_record.empty and text_path.is_file():
                text = sample_name + ' '.join(
                    process_spaces(normalize_char_text(text_path.read_text())).split()
                )
                texts.append(text)
                labels.append(
                    [
                        x
                        for x in classification_record['tag'].values[0].split(',')
                        if x in label_names
                    ]
                )
        preds = batch_infer(model, tokenizer, texts)

        metrics = get_metrics(
            preds=batch_tags_to_multilabel_vectors(model.config.id2label, preds),
            labels=batch_tags_to_multilabel_vectors(model.config.id2label, labels),
        )
        acc = metrics['accuracy']
        file_count = len(preds)
        print(
            f"file_type: {file_type}, file_count: {file_count}, accuracy: {acc}",
        )
        correct_count += int(acc * file_count)
        total_count += file_count
    print(f'total_files: {total_count}, global_accuracy: {correct_count / total_count}')


if __name__ == '__main__':
    batch_test()
