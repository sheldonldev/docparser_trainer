from transformers import AutoModelForQuestionAnswering  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer

setup_env()


def infer(model, tokenizer, question, context):
    from transformers import pipeline

    pipe = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)
    print(pipe(question=question, context=context))  # type: ignore


def main(model_id, ckpt_dir, questions, contexts):
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        model_cls=AutoModelForQuestionAnswering,
        ckpt_dir=ckpt_dir,
    )
    answers = []
    for q, c in zip(questions, contexts):
        answers.append(infer(model, tokenizer, q, c))
    return answers


if __name__ == '__main__':
    model_id = 'hfl/chinese-macbert-base'

    # # 非滑动窗口实现
    # ckpt_dir = CKPT_ROOT.joinpath(
    #     "machine_reading_comprehension/fragment_extraction_naive",
    # ).joinpath("477")

    # 滑动窗口实现
    ckpt_dir = CKPT_ROOT.joinpath(
        "machine_reading_comprehension/fragment_extraction_slide",
    ).joinpath("checkpoint-600")

    questions = ['小明在哪里上班']
    contexts = ['小明在北京上班']

    main(model_id, ckpt_dir, questions, contexts)
