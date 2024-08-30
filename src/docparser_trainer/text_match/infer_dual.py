from docparser_trainer._cfg import CKPT_ROOT, setup_env
from docparser_trainer._interface.model_manager import load_model, load_tokenizer
from docparser_trainer.text_match.models import DualModel

setup_env()


def infer(model, tokenizer, text, text_pair):
    from torch.nn import CosineSimilarity

    class SentenceSimilarityPipeline:
        def __init__(self, model, tokenizer):
            self.model = model.bert  # 只拿向量表示
            self.tokenizer = tokenizer
            self.device = model.device

        def preprocess(self, senA, senB):
            return tokenizer(
                [senA, senB],
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )

        def predict(self, inputs):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return self.model(**inputs)[1]  # 2, 768

        def postprocess(self, logits):
            cos = CosineSimilarity()(logits[None, 0, :], logits[None, 1, :]).squeeze().cpu().item()
            return cos

        def __call__(self, senA, senB, return_vector=False):
            inputs = self.preprocess(senA, senB)
            logits = self.predict(inputs)
            result = self.postprocess(logits)
            if return_vector is True:
                return result, logits.cpu().detach().numpy()
            return result

    pipe = SentenceSimilarityPipeline(model, tokenizer)

    return pipe(text, text_pair, return_vector=True)


def main(model_id, ckpt_dir, texts, text_pairs):
    tokenizer = load_tokenizer(model_id)
    model = load_model(
        ckpt_dir=ckpt_dir,
        model_cls=DualModel,
    )
    results = []
    for text, pair in zip(texts, text_pairs):
        results.append(infer(model, tokenizer, text, pair))
    return results


if __name__ == '__main__':
    model_id = 'hfl/chinese-macbert-base'
    ckpt_dir = CKPT_ROOT.joinpath(
        'text_match/text_similarity_dual',
    ).joinpath('checkpoint-282')

    texts = ['小明喜欢北京']
    text_pairs = ['北京是小明喜欢的城市']
    main(model_id, ckpt_dir, texts, text_pairs)
