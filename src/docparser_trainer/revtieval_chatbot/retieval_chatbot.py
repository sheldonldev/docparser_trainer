from pathlib import Path

import pandas as pd
import torch
from faiss import IndexFlatIP, normalize_L2  # type: ignore
from transformers import AutoModelForSequenceClassification  # type: ignore

from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer.text_match.models import DualModel

setup_env()


def batch_trans_data_to_vectors(questions, model):
    from tqdm import tqdm

    vectors_ = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(questions), 32)):
            batch_sentences = questions[i : i + 32]
            inputs = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=128,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            vector = model.bert(**inputs)[1]
            vectors_.append(vector)
    vectors = torch.concat(vectors_, dim=0).cpu().detach().numpy()
    return vectors


def build_index(vectors):
    index = IndexFlatIP(vectors.shape[1])  # 点积操作, 入参为向量维度
    normalize_L2(vectors)  # L2正则化
    index.add(vectors)  # type: ignore
    return index


def get_question_vector(question, model):
    with torch.inference_mode():
        inputs = tokenizer(
            question,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        vector = model.bert(**inputs)[1]
        q_vector = vector.cpu().detach().numpy()
    return q_vector


def search(index, q_vector, top_k=1):
    normalize_L2(q_vector)
    scores, indices = index.search(q_vector, top_k)
    return scores, indices


def predict_from_candidates(question, candidates, model):
    ques = [question] * len(candidates)
    inputs = tokenizer(
        ques,
        candidates,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=128,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits.squeeze()
        result = torch.argmax(logits, dim=-1)
    return result.item()


if __name__ == '__main__':
    data_path = Path(
        '/home/sheldon/repos/docparser_trainer/data/'
        'SophonPlus/ChineseNlpCorpus/datasets/lawzhidao/lawzhidao_filter.csv'
    )
    data = pd.read_csv(data_path)
    questions = data['title'].tolist()

    model_id = 'hfl/chinese-macbert-base'
    tokenizer = load_tokenizer(model_id)

    dual_model = load_model(
        ckpt_dir=CKPT_ROOT.joinpath('text_match/text_similarity_dual/checkpoint-282'),
        model_cls=DualModel,
    )  # 快速但不准确

    vectors = batch_trans_data_to_vectors(questions, dual_model)
    index = build_index(vectors)

    question = "寻衅滋事"
    q_vector = get_question_vector(question, dual_model)
    scores, indices = search(
        index, q_vector, top_k=10
    )  # 取回 top 10 个最相似的(高召回), 再用 cross_model(高精度)
    top_data = data.values[indices[0]]
    top_questions = top_data[:, 1].tolist()

    cross_model = load_model(
        ckpt_dir=CKPT_ROOT.joinpath('text_match/text_similarity_cross/checkpoint-846'),
        model_cls=AutoModelForSequenceClassification,
        num_labels=1,
    )  # 准确但不快速

    idx = predict_from_candidates(question, top_questions, cross_model)
    print(top_data[idx])  # type: ignore
