from pathlib import Path

from docparser_models._model_interface.model_manager import (
    ensure_local_model,
    load_local_model,
    load_tokenizer,
)
from transformers import (  # type: ignore
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from docparser_trainer._cfg import MODEL_ROOT, PROJECT_ROOT, setup_env
from docparser_trainer._interface.get_datasets import get_datasets

setup_env()
CHECKPOINTS_DIR = PROJECT_ROOT.joinpath(
    "checkpoints/machine_reading_comprehension/fragment_extraction"
)


def get_model(ckpt_dir: Path | None = None):
    if ckpt_dir is None:
        model_dir = ensure_local_model(
            model_id,
            model_cls=AutoModelForQuestionAnswering,
            local_directory=MODEL_ROOT.joinpath(f'{model_id}-mrc-fragment-extraction'),
        )
    else:
        model_dir = ckpt_dir
    model = load_local_model(
        model_dir,
        model_cls=AutoModelForQuestionAnswering,
    )
    return model


def preprocess_datasets(tokenizer, datasets):
    def get_context_start(sequence_ids):
        """
        获取 context 的起始 token 位置
        """
        start = sequence_ids.index(1)
        end = sequence_ids.index(None, start + 1) - 1
        return start, end

    def position_mapping_to_token(answer_start, answer_end, offset, context_start, context_end):
        """
        将 char 位置映射到 token 位置
        """
        start_token = 0
        end_token = 0

        if offset[context_start][0] > answer_end or offset[context_end][1] < answer_start:
            return start_token, end_token

        for i in range(context_start, context_end + 1):
            if offset[i][0] <= answer_start < offset[i][1]:
                start_token = i
                break

        for i in range(start_token, context_end + 1):
            if offset[i][0] < answer_end <= offset[i][1]:
                end_token = i
                break

        return start_token, end_token

    def process_token(examples):
        def print_answer():
            print(">>>")
            print("question:", examples['question'][example_idx])
            print("answer:", examples['context'][example_idx][start_char:end_char])
            print(
                "check answer:",
                examples['context'][example_idx][offset[start_token][0] : offset[end_token][1]],
            )
            print(
                "check decode:",
                tokenizer.decode(
                    tokenized['input_ids'][truncate_idx][start_token : end_token + 1],
                ),
            )
            print("<<<")

        tokenized = tokenizer(
            text=examples['question'],
            text_pair=examples['context'],
            max_length=512,
            truncation='only_second',  # 只截断 text_pair
            padding='max_length',
            return_offsets_mapping=True,  # 返回每个 token 对应的 char 位置
            return_overflowing_tokens=True,  # 截断后，将原始数据复制一份，并添加到 batch 中
            stride=128,
        )

        overflow_mapping = tokenized.pop('overflow_to_sample_mapping')

        start_positions = []
        end_positions = []
        example_ids = []
        for truncate_idx, example_idx in enumerate(overflow_mapping):
            answer = examples['answers'][example_idx]
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])

            offset = tokenized['offset_mapping'][truncate_idx]
            context_start, content_end = get_context_start(tokenized.sequence_ids(truncate_idx))
            start_token, end_token = position_mapping_to_token(
                start_char, end_char, offset, context_start, content_end
            )

            if truncate_idx < 5:
                print_answer()

            start_positions.append(start_token)
            end_positions.append(end_token)
            example_ids.append(examples['id'][example_idx])

            tokenized['offset_mapping'][truncate_idx] = [
                (o if tokenized.sequence_ids(truncate_idx)[k] == 1 else None)
                for k, o in enumerate(offset)
            ]

        tokenized['start_positions'] = start_positions
        tokenized['end_positions'] = end_positions
        tokenized['example_ids'] = example_ids
        return tokenized

    tokenized_datasets = datasets.map(
        process_token,
        batched=True,
        remove_columns=datasets['train'].column_names,
    )
    return tokenized_datasets


def train(ckpt_dir: Path | None = None):
    tokenized_datasets = preprocess_datasets(tokenizer, mrc_datasets)
    model = get_model(ckpt_dir)

    args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DefaultDataCollator(),
    )
    trainer.train()


def infer(ckpt_dir: Path | None = None):
    from transformers import pipeline

    pipe = pipeline('question-answering', model=get_model(ckpt_dir), tokenizer=tokenizer, device=0)
    print(pipe(question='小明在哪里上班', context='小明在北京上班'))  # type: ignore


if __name__ == '__main__':
    dataset_name = 'hfl/cmrc2018'
    mrc_datasets = get_datasets(dataset_name)

    model_id = 'hfl/chinese-macbert-base'
    tokenizer = load_tokenizer(model_id)

    ckpt_dir: Path | None = None
    # ckpt_dir = CHECKPOINTS_DIR.joinpath('checkpoint-477')
    train(ckpt_dir=ckpt_dir)
    # infer(ckpt_dir=ckpt_dir)
