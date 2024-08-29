from typing import Optional

import torch
from torch.nn import CosineEmbeddingLoss, CosineSimilarity
from transformers import BertModel, BertPreTrainedModel  # type: ignore


class DualModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)  # 获得 embedding
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Step1: 获得 sentence A 和 sentence B 的 输入
        senA_input_ids, senB_input_ids = input_ids[:, 0], input_ids[:, 1]  # type: ignore
        senA_attention_mask, senB_attention_mask = (
            attention_mask[:, 0],  # type: ignore
            attention_mask[:, 1],  # type: ignore
        )  # type: ignore
        senA_token_type_ids, senB_token_type_ids = (
            token_type_ids[:, 0],  # type: ignore
            token_type_ids[:, 1],  # type: ignore
        )

        # Step2: 获得 sentence A 和 sentence B 的向量表示
        senA_output = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        senA_pooled_output = senA_output[1]  # [batch, hidden]

        senB_output = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        senB_pooled_output = senB_output[1]  # [batch, hidden]

        # Step3: 计算相似度
        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)  # [batch]

        # Step4: 计算损失
        loss = None
        if labels is not None:
            loss_fct = CosineEmbeddingLoss(margin=0.3)
            loss = loss_fct(senA_pooled_output, senB_pooled_output, labels)

        output = (cos,)
        return ((loss,) + output) if loss is not None else output
