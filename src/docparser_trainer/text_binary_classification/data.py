import json
from pathlib import Path
from typing import List

from datasets.customs_declaration.parse_dataframe import get_classification_tag
from util_common.io import parse_bool


def yield_customs_declaration_data(batch_names: List[str] = ['20240301']):
    dir = Path("/home/sheldon/repos/datasets/data/customs_declaration")

    def get_split_dict(batch_dir: Path):
        return json.loads(batch_dir.joinpath('split_dict.json').read_text())

    for batch in batch_names:
        batch_dir = dir.joinpath(batch)
        unified_dir = batch_dir.joinpath('unified')
        split_dict = get_split_dict(batch_dir)
        classification_df, _ = get_classification_tag(dir.joinpath(batch))

        for i, row in classification_df.iterrows():
            name = row['name']

            file_type = row['file_type']
            tag = row['tag']
            sign_off = parse_bool(row['sign_off'])
            is_eval = True
            if name in split_dict[file_type]['train']:
                is_eval = False
            if sign_off is True:
                path = Path(f'{unified_dir}/{file_type}/{name}/pure.txt')
                if path.is_file():
                    yield Path(
                        f'{unified_dir}/{file_type}/{name}/pure.txt'
                    ), tag, is_eval
                else:
                    print(f'File not exists: {path}')


def load_data(positive_tags: List[str]):
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []

    for text_path, tag, is_eval in yield_customs_declaration_data():
        if is_eval:
            val_texts.append(text_path.read_text())
            if tag in positive_tags:
                val_labels.append(1)
            else:
                val_labels.append(0)
        else:
            train_texts.append(text_path.read_text())
            if tag in positive_tags:
                train_labels.append(1)
            else:
                train_labels.append(0)

    return train_texts, train_labels, val_texts, val_labels
