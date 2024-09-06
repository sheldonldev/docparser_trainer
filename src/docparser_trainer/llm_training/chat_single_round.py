import torch


def chat_llama3(model, tokenizer, query):
    model.eval().cuda()
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(
        [f'<s>Human: {query}\n</s><s>Assistant: '],
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to(model.device)
    input_ids = input_ids.to(model.device)
    generate_input = {
        "input_ids": input_ids,
        "max_new_tokens": 512,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    generate_ids = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0])
    return text


def chat_glm4(model, tokenizer, query):
    # https://github.com/THUDM/GLM-4/blob/main/basic_demo/trans_cli_demo.py
    from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            stop_ids = model.config.eos_token_id
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    model.eval().cuda()
    messages = [{"role": "user", "content": query}]
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(model.device)
    generate_kwargs = {
        "input_ids": model_inputs,
        "max_new_tokens": 8192,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.6,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens()]),
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    result = model.generate(**generate_kwargs)
    answer = (
        tokenizer.decode(result[0].tolist()).split('<|user|>')[1].split('<|assistant|>')[1].strip()
    )
    return answer
