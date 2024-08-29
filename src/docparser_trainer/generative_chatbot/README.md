# 生成式对话机器人

## 预训练任务

- 因果语言模型, 自回归模型
  - 将完整序列输入，基于上文的 token 预测当前的 token
  - 结束位置要有特殊 token, eos_token

## 指令微调

- 指令微调的方式赋予回答问题的能力
  - 多类型任务共同学习，能够解决不同任务

  ```txt
          Prompt/Input/Insturction  Output/Response
  input:  1 2 3 4 5 6 7 8         | 9 10 11 12 13 14 eos
  output: -100...                 | 9 10 11 12 13 14 eos

  只计算 output 的 loss
  ```

  - 多轮计算

  ```txt
          Turn1 Input + Output +    Turn2 Input 
  input:  1 2 3       | 4 5 6 eos | 8 9 10      | 11 12 13 14 eos
  output: -100...                               | 11 12 13 14 eos

  只计算当前轮次 output 的 loss, 但这种方法效率较低
  ```

## 常见解码参数

- 推理参数
  - 长度控制
    - min/max_new_token 最小/最大生成长度
    - min/max_length 序列整体的最小/最大长度
  - 解码策略
    - do_sample 是否启用采样的生成方式(默认False，即用 beam_search)
    - num_beams beam_search 的大小(默认1，每次都取最大值)
  - 采样参数
    - temperature 采样温度, 默认1.0(原始分布), 低于1.0会使分布更尖锐，高于1.0会使分布更均匀
    - top_k 将词概率从大到小排列，将采样限制在前K个词
    - top_p 将词概率从大到小排列，将采样限制在前N个词，条件是这N个词的概率超过top_p的值(相比top_k，top_p的采样策略更灵活，可以保证生成的文本中不包含一些低概率的词)
  - 惩罚项
    - repetition_penalty 重复惩罚项，降低已经出现过的 token 的概率
