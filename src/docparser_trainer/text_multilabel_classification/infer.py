from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer  # type: ignore

from docparser_trainer._cfg import setup_env
from docparser_trainer._utils.seed import seed_everything

setup_env()
seed_everything(42)


def batch_infer(model, tokenizer, texts: List[str], label_names: List[str]):
    class MultiLabelClassifierPipeline:
        def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size=128):
            self.model = model
            self.tokenizer = tokenizer
            self.batch_size = batch_size

            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()  # type: ignore
            else:
                raise Exception('CUDA not available!')

        def batch_preprocess(self, texts: List[str], max_length=2048):
            encodings = self.tokenizer.batch_encode_plus(
                texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_token_type_ids=False,
                return_tensors='pt',
            )
            return {k: v.to(self.model.device) for k, v in encodings.items()}

        def batch_predict(self, inputs):
            with torch.inference_mode():
                outputs = self.model(**inputs)[0]
            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).int()
            return predictions.tolist()

        def batch_postprocess(self, predictions, label_names) -> List[List[str]]:
            results = []
            for pred in predictions:
                results.append([label_names[i] for i in range(len(label_names)) if pred[i] == 1])
            return results

        def __call__(self, texts, label_names) -> List[List[str]]:
            texts_batches = [
                texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
            ]
            results = []
            for texts in texts_batches:
                inputs = self.batch_preprocess(texts)
                predictions = self.batch_predict(inputs)
                results.extend(self.batch_postprocess(predictions, label_names))
            return results

    pipeline = MultiLabelClassifierPipeline(model, tokenizer)
    return pipeline(texts, label_names)
