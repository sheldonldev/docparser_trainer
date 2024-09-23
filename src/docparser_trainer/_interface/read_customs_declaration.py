import json
from pathlib import Path

from docparser_datasets.customs_declaration.df_parser import get_classification_tag


def yield_customs_declaration_data_classification_text(batch_names: list[str]):
    dir = Path("/home/sheldon/repos/docparser_datasets/data/customs_declaration")

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
            if name in split_dict['unified'][file_type]['test']:
                continue

            tags = row['tag'].split(',')
            if len(tags) > 1 and 'Other' in tags:
                tags.remove('Other')

            is_eval = True
            if name in split_dict['unified'][file_type]['train']:
                is_eval = False
            path = Path(f'{unified_dir}/{file_type}/{name}/pure.txt')
            if path.is_file():
                yield name, Path(f'{unified_dir}/{file_type}/{name}/pure.txt'), tags, is_eval
            else:
                print(f'File not exists: {path}')
