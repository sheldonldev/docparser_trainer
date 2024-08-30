import torch
from transformers import AutoModelForMultipleChoice  # type:ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def infer(model, tokenizer, context, question, choices):
    class MultipleChoicePipeline:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = model.device

        def preprocess(self, context, question, choices):
            cs, qcs = [], []
            for choice in choices:
                cs.append(context)
                qcs.append(question + ' ' + choice)
            return tokenizer(
                cs,
                qcs,
                max_length=256,
                truncation="only_first",
                return_tensors='pt',  # 返回 pytorch 的张量
            )

        def predict(self, inputs):
            inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
            return self.model(**inputs).logits

        def postprocess(self, logits, choices):
            pred = torch.argmax(logits, dim=-1).cpu().item()
            return choices[pred]

        def __call__(self, context, question, choices):
            inputs = self.preprocess(context, question, choices)
            logits = self.predict(inputs)
            result = self.postprocess(logits, choices)
            return result

    pipe = MultipleChoicePipeline(model, tokenizer)
    print(pipe(context, question, choices))


def main(model_id, ckpt_dir, contexts, questions, choices_batches):
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        model_cls=AutoModelForMultipleChoice,
        ckpt_dir=ckpt_dir,
    )
    answers = []
    for c, q, choices in zip(contexts, questions, choices_batches):
        answers.append(infer(model, tokenizer, c, q, choices))
    return answers


if __name__ == '__main__':
    model_id = 'hfl/chinese-macbert-base'
    ckpt_dir = CKPT_ROOT.joinpath(
        'machine_reading_comprehension/multi_choice',
    ).joinpath('checkpoint-200')

    contexts = ['小明在北京上班']
    questions = ['小明在哪里上班']
    choices_batches = [
        ['北京', '上海'],
    ]  # 预测的时候没有 batch 的概念，不需要 padding 也不限制于选项数量

    main(
        model_id,
        ckpt_dir,
        contexts,
        questions,
        choices_batches,
    )
