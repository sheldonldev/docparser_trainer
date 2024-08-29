import gradio as gr  # type: ignore
from transformers import pipeline  # type: ignore
from util_common.decorator import proxy


@proxy(http_proxy='http://127.0.0.1:17890', https_proxy='http://127.0.0.1:17890')
def get_interface():
    interface = gr.Interface.from_pipeline(
        pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa", device=0)
    )
    return interface


if __name__ == '__main__':
    interface = get_interface()
    interface.launch(share=True)
