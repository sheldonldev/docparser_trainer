from transformers import (  # type: ignore
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


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
        if answer_start > answer_end:
            return start_token, end_token

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
        tokenized = tokenizer(
            text=examples['question'],
            text_pair=examples['context'],
            max_length=512,
            truncation='only_second',  # 只截断 text_pair
            padding='max_length',
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized.pop('offset_mapping')

        start_positions = []
        end_positions = []
        for idx, offset in enumerate(offset_mapping):
            answer = examples['answers'][idx]
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])
            # print(examples['question'][idx])
            # print(examples['context'][idx][start_char:end_char])

            context_start, content_end = get_context_start(tokenized.sequence_ids(idx))
            start_token, end_token = position_mapping_to_token(
                start_char, end_char, offset, context_start, content_end
            )
            # print(examples['context'][idx][offset[start_token][0] : offset[end_token][1]])
            # print(tokenizer.decode(tokenized['input_ids'][idx][start_token : end_token + 1]))

            start_positions.append(start_token)
            end_positions.append(end_token)

        tokenized['start_positions'] = start_positions
        tokenized['end_positions'] = end_positions
        return tokenized

    tokenized_datasets = datasets.map(
        process_token,
        batched=True,
        remove_columns=datasets['train'].column_names,
    )
    return tokenized_datasets


def train(model):
    tokenized_datasets = preprocess_datasets(tokenizer, mrc_datasets)

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
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


def infer(model):
    from transformers import pipeline

    pipe = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)
    print(pipe(question='小明在哪里上班', context='小明在北京上班'))  # type: ignore


if __name__ == '__main__':
    dataset_name = 'hfl/cmrc2018'
    mrc_datasets = get_datasets(dataset_name)

    model_id = 'hfl/chinese-macbert-base'
    tokenizer = load_tokenizer(model_id)

    pretrained_dir = MODEL_ROOT.joinpath(f'{model_id}-mrc-fragment-extraction')
    checkpoints_dir = CKPT_ROOT.joinpath("machine_reading_comprehension/fragment_extraction_slide")
    ckpt_version: str | None = None
    ckpt_dir = checkpoints_dir.joinpath(ckpt_version) if ckpt_version else None

    model = load_model(
        model_id,
        ckpt_dir=ckpt_dir,
        model_cls=AutoModelForQuestionAnswering,
        pretrained_dir=pretrained_dir,
    )
    train(model)
    infer(model)
