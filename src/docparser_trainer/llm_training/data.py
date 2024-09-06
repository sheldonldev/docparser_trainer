from pathlib import Path

from docparser_trainer._interface.datasets_manager import get_datasets


def get_llama3_demo_datasets():
    dir = Path(__file__).parent
    datasets = get_datasets(
        'csv',
        data_files={
            'train': str(dir / 'demo_data/llama3/train_sft.csv'),
            'validation': [
                str(dir / 'demo_data/llama3/dev_data.csv'),
                str(dir / 'demo_data/llama3/dev_sft_sharegpt.csv'),
            ],
        },
    )
    return datasets


def preprocess_alpaca_datasets_for_llama3():
    # https://huggingface.co/datasets/shibing624/alpaca-zh
    def process_fn(examples):
        texts = []
        for instruction, inp, out in zip(
            examples['instruction'], examples['input'], examples['output']
        ):
            if inp:
                texts.append(
                    f"<s>Human: {instruction}\nInput: {inp}\n</s><s>Assistant: {out}\n</s>"
                )
            else:
                texts.append(f"<s>Human: {instruction}\n</s><s>Assistant: {out}\n</s>")

        return {'text': texts}

    dataset_id = 'shibing624/alpaca-zh'
    datasets = get_datasets(dataset_id)['train'].train_test_split(test_size=0.05)
    datasets['validation'] = datasets.pop('test')

    return datasets.map(
        process_fn,
        batched=True,
        remove_columns=datasets['train'].column_names,
    )


def preprocess_alpaca_datasets_for_glm4():
    # https://huggingface.co/datasets/shibing624/alpaca-zh
    def process_fn(examples):
        message_batches = []
        for instruction, inp, out in zip(
            examples['instruction'], examples['input'], examples['output']
        ):
            messages = []
            if inp:
                query = f'{instruction}\nInput: {inp}'
            else:
                query = instruction
            messages.append({'role': 'user', 'content': query})
            messages.append({'role': 'assistant', 'content': out})
            message_batches.append(messages)
        return {'messages': message_batches}

    dataset_id = 'shibing624/alpaca-zh'
    datasets = get_datasets(dataset_id)['train'].train_test_split(test_size=0.05)
    datasets = datasets.map(
        process_fn,
        batched=True,
        remove_columns=datasets['train'].column_names,
    )
    validation_dataset = datasets.pop('test')
    train_dataset = datasets['train']
    return train_dataset, validation_dataset
