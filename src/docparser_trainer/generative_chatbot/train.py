from transformers import (  # type: ignore
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from docparser_trainer._cfg import CKPT_ROOT, MODEL_ROOT, setup_env
from docparser_trainer._interface.datasets_manager import get_datasets
from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer._optimization.peft import (
    bit_fit,
    hard_prompt_tuning,
    ia3,
    lora,
    p_tuning,
    prefix_tuning,
    soft_prompt_tuning,
)
from docparser_trainer._utils.model_util import print_params

setup_env()


def preprocess_datasets(tokenizer, datasets):
    def preprocess_function(example):
        max_length = 256
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            "\n".join(["Human: " + example['instruction'], example['input']]).strip()
            + "\n\nAssistant: ",
        )
        response = tokenizer(example['output'] + tokenizer.eos_token)
        input_ids = (instruction['input_ids'] + response['input_ids'])[:max_length]
        attention_mask = (instruction['attention_mask'] + response['attention_mask'])[:max_length]
        labels = ([-100] * len(instruction['input_ids']) + response['input_ids'])[:max_length]
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    tokenized_datasets = datasets.map(
        preprocess_function,
        remove_columns=datasets["train"].column_names,
    )

    # print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
    # print(tokenizer.decode([x for x in tokenized_datasets["train"][0]["labels"] if x != -100]))
    return tokenized_datasets


def train(model):
    tokenized_datasets = preprocess_datasets(tokenizer, datasets)

    # 没有评估，所以不需要 Seq2SeqTrainingArguments 和 Seq2SeqTrainer
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_strategy='epoch',
        num_train_epochs=1,
        learning_rate=3e-3,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()


def infer(model):
    from transformers import pipeline

    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
    instruction = "考试有哪些技巧"
    input = "\n".join(["Human: " + instruction, ""]).strip()
    print(
        pipe(
            input,
            max_length=256,
            # do_sample=True,
            # top_k=50,
            # temperature=0.8,
            # top_p=0.9,
            # num_beams=1,
        ),
    )


if __name__ == '__main__':
    datasets_id = "llm-wizard/alpaca-gpt4-data-zh"
    datasets = get_datasets(datasets_id)

    # model_id = 'Langboat/bloom-389m-zh'
    model_id = 'Langboat/bloom-1b4-zh'
    tokenizer = load_tokenizer(model_id)

    ckpt_dir = CKPT_ROOT / f"generative_chatbot/{model_id}-lora"
    ckpt_version: str | None = None

    model = load_model(
        model_id,
        model_cls=AutoModelForCausalLM,
        ckpt_dir=ckpt_dir / ckpt_version if ckpt_version else None,
        pretrained_dir=MODEL_ROOT.joinpath(f'{model_id}-generative-chatbot'),
    )

    model_size = sum(param.numel() for param in model.parameters())
    print(f"model size: {model_size:,} params".capitalize())

    param_gpu_usage = model_size * 4
    print(f"parameters gpu usage: {param_gpu_usage:,} bytes".capitalize())

    gradient_gpu_usage = model_size * 4
    print(f"gradient gpu usage: {gradient_gpu_usage:,} bytes".capitalize())

    optimizer_gpu_usage = model_size * 4 * 2
    print(f"optimizer gpu usage: {optimizer_gpu_usage:,} bytes".capitalize())

    model_gpu_usage = param_gpu_usage + gradient_gpu_usage + optimizer_gpu_usage
    print(f"model gpu usage: {model_gpu_usage:,} bytes".capitalize())

    print_params(model)

    peft_args = {
        # 'bit_fit': True,
        # 'soft_prompt_tuning': True,
        # 'hard_prompt_tuning': True,
        # 'p_tuning': True,
        # 'prefix_tuning': True,
        # 'lora': True,
        'ia3': True
    }
    if peft_args.get('bit_fit'):
        bit_fit(model)
    if peft_args.get('soft_prompt_tuning'):
        model = soft_prompt_tuning(model)
    if peft_args.get('hard_prompt_tuning'):
        model = hard_prompt_tuning(model, tokenizer)
    if peft_args.get('p_tuning'):
        model = p_tuning(model)
    if peft_args.get('prefix_tuning'):
        model = prefix_tuning(model)
    if peft_args.get('lora'):
        model = lora(model)
    if peft_args.get('ia3'):
        model = ia3(model)

    print_params(model)

    train(model)
