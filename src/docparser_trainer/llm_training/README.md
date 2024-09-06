# 大语言模型训练

## 参数上显存优化策略

优化策略 | 优化对象 | 预期效果
:-- | :-- | :--
Gradient Accumulation | 前向激活值 | 降显存、降速
Gradient Checkpoints | 前向激活值 | 降显存、降速
Adafactor Optimizer | 优化器状态 | 降显存、降速
Freeze Model | 前向激活值、梯度 | 降显存、提速、被Freeze的模型不再学习
Data Length | 前向激活值 | 降显存、提速、可能严重影响模型效果

## 半精度介绍

(s: sign, e: exponent, m: mantissa)

- fp32: 32位浮点数，单精度。1s + 8e + 23m, range: [1e-38, 3e38]
- fp16: 16位浮点数，半精度。1s + 5e + 10m, range: [5.96^-8, 65504]
  - 溢出问题
  - 舍入问题
- bf16: 16位浮点数，半精度。1s + 8e + 7m, range: [1e-38, 3e38]

## 量化介绍

用低精度对模型权重或激活值进行表示,可以降低显存,有时候可以加速(什么时候),也有时候速度会下降（什么时候）

### 如何量化和反量化

#### int8线性量化示例 (absmax 方法)

- 原始数据:

  ```x = [1.52, 1.64, 1.83, 4.32]```

- 量化过程：

  ```txt
  x_absmax = 4.32

  scale_factor = (2^8-1) / x_absmax = 29.4

  q_x = round(array(x) * scale_factor) = [45, 48, 54, 127]
  ```

- 反量化:

  ```x' = q_x / scale_factor = [1.53, 1.63, 1.83, 4.32]```

- 量化问题：
  - 离群值
  - 量化具有多个大值的向量会产生误差
  - 过程中累积误差会导致模型的最终性能下降

- 如何降低量化误差
  - vector-wise: 为 loraA 的每一行和 loraB 的每一列单独设置 scale_factor
  - 混合精度分解量化:
    - vector-wise constants 检查分布, 从输入的隐含状态中按列提取离群值
    - 非离群值矩阵用 int8 量化, 离群值矩阵用 float16 量化
    - int8 非离群值矩阵计算完后会反量化成 float16 和 float16 进行合并, 最终获得 float16的结果

### QLoRA 介绍

- 线性量化
  - 简单线性量化存在的问题
    - 4bit 表示范围比 8bit 更小，粒度更粗
    - 不用非常大的离群值就使得参数都集中在某数值上
    - 量化误差非常大
  - 如何理解线性量化
    - 数据归一化到 [-1, 1], 把 [-1, 1] 均匀切分成 N 区间（N取决于量化bit）
    - 归一化的结果归属于第几区间，量化结果便为几，数据的分布对量化结果非常重要
    - 如果待量化的数据为均匀分布，那么线性量化的结果即使是4bit也不会太差

- 正太分布与量化结合
  - 模型参数一般符合正太分布
  - 分位数量化
    - 把顺序排列的一组数据分割为若干相等的分割点的数值即为相应的分为数, 中位数是最简单的一种
    - 以4bit为例，表示范围为16个值，将权重从小到大排序，找到15个分位数，将其切分为16块，权重数值落在第几块，量化就表示多少
    - 此外，由于涉及到反量化，还需要给这 16 个值一个浮点数的映射，这个映射可以取分位区间两侧分位点的均值，用于反量化，这部分称之为量化值
    - 具体操作是，我们只需要把待量化的数跟量化值进行比较，取最相近的值作为该数值的量化值，对应的表示可以通过分位数进行确定，存储时同时存储4bit的表示与对应量化值，计算时使用量化值反量化后进行计算
  - 分位数量化改进-NF4
    - 从正太分布入手, 将其统一缩放至[-1,1],根据标准正太分布得到16个量化值，并将量化值也缩放至[-1,1], 此时，便可利用上面的方法，将权重量化
    - 为了减少异常值问题，采用分块量化，块大小为64.

#### NF4 量化示例

- NF4

  ```txt
  [
    -1.0,    -0.6961, -0.5250, -0.3949,
    -0.2844, -0.1848, -0.0912, -0.0,
      0.0796,  0.1693,  0.2461,  0.3379, 
      0.4407,  0.5626,  0.7229,  1.0,
  ]
  ```

- 原始数据:

  ```x = [1.52, 1.64, 1.83, 4.32]```

- 4bit 量化过程

  ```txt
  x_absmax = 4.32

  x_norm = array(x) / 4.32 = [0.3587, 0.375, 0.4236, 1]

  q_x = [0.3379, 0.3379, 0.4407, 1]

  NF4_match = [11, 11, 12, 15]
  ```

- 反量化过程

  ```x' = array(NF4[NF4_match]) * x_absmax = [1.46, 1.46, 1.90, 4.32]```

#### 双重量化

- NF4量化存储空间
  - NF4分位点：16个常量，忽略不计
  - 量化常数 absmax: 每个块(64个权重)需要一个 float32 的量化常数, 每个权重需要额外 0.5 bit 存储

- 双重量化
  - 对量化常数进行二次量化，256 个块作为一组量化为 float8
  - 每个权重仅需要额外 8 / 64 + 32/(64 x 256) = 0.127 bit

#### 分页器优化

- 当显存不足时，将优化器参数转移到 CPU 内存上，需要时再取回 (这个功能也是 bitsandbytes 提供的)

### 量化注意事项

- 因为量化的 lora 存在精度损失，所以最好不要 merge

## 多卡场景分布式训练

- 单卡可以完成训练的情况：
  - 数据并行 (Data Parallel)
    - 每个GPU上都复制一个完整的模型，但每个GPU上训练的数据不同，要求每张卡内都可以执行完整的训练过程

- 单卡无法完成训练的情况：
  - 流水并行 (Pipline Parallel)
    - 将模型按层拆开, 每个GPU上包含部分层，能够保证正常训练

  - 张量并行 (Tensor Parallel)
    - 将模型每层的权重拆开，对于一份权重，每个GPU上个包含一部分，保证能够正常训练

- 混合策略
  DP + PP + TP (3D 并行)

### Data Parallel

- 训练流程
  - GPU0 加载 model 和 batch 数据
  - 将 batch 数据从 GPU0 均分至各卡
  - 将 model 从 GPU0 均分至各卡
  - 各卡同时进行前向传播
  - GPU0 收集各卡输出，合并计算loss
  - 将 loss 分发至各卡，进行反向传播，计算梯度
  - GPU0 收集各卡梯度，进行汇总
  - GPU0 更新模型

#### torch.nn.DataParallel

- torch 实现：

  ```py
  # 可以增加 batch_size
  p_model = torch.nn.DataParallel(model, device_ids=None)
  model = p_model.module # 真正的 model
  output = model(**batch)
  output.loss.mean().backward() # 多个卡的 loss 求均值,再反向传
  ```

- Transformers 实现:
  不需要改代码，Trainer.train() 底层实现

- 实际效果：
  - 对速度提升不明显（甚至下降）
  - 单进程，多线程，由于GIL锁的问题，不能充分发挥多卡优势
  - 主节点利用率会高于其他节点，其他节点显存浪费
  - 只适用于单机训练
  - 不适合训练但是适合推理(因为可以用 dataloader 加载更大 batch_size 的数据, 还需要修改 DataParallel的 forward, 不要在里面 replicate model)

### Distributed Data Parallel

- 训练流程
  - 使用多个进程, 每个进程都能加载数据和模型
  - 各个进程同时前向传播得到输出
  - 各个进程分别计算loss，反向传播计算梯度
  - 各个进程间通信，将梯度在各卡同步
  - 各进程分别更新模型

- 基本概念
  - group: 进程组，一个分布式任务对应一个进程组，一般是所有卡都在一个组里
  - world_size: 全局并行数，一般情况下等于总的卡数(但不一定)
  - node: 节点，可以是一台机器，一个容器，节点内包含多个 GPU
  - rank(global_rank): 整个分布式训练任务内的进程序号
  - local_rank: 每个 node 内不得相对进程序号

- 通信
  - 点对点通信
  - 集合通信:
    - 种类：
      - 发布：
        - Scatter (一个 rank 向其他 rank 分发数据，每份分发的数据是不一样的)
        - Broadcast (一个 rank 向其他 rank 分发数据，每份分发的数据是一样的)
      - 收集：
        - Gather (一个 rank 收集其他 rank 数据)
        - All Gather (ranks 之间相互收集数据， 最后每个rank都是全量的)
        - Reduce (Gather 的基础上进行一些计算)
        - All Reduce (All Gather 的基础上进行一些计算)

#### torch.distributed

- torch 实现

  ```py
  # 数据分发
  import torch.distributed as dist
  from torch.utils.data import DataLoader
  from torch.data.distributed import DistributedSampler

  dist.init_process_group(backend='nccl')
  trainloader = DataLoader(trainset, batch_size, collate_fn, sampler=DistributedSampler(trainset))
  trainloader.sampler.set_epoch(ep_id) # shuffle 一下防止 Accuracy 虚高

  # 模型分发
  from torch.nn.parallel import DistributedDataParallel as DDP
  import os

  if torch.cuda.is_availabel():
    model = model.to(os.environ["LOCAL_RANK"]) # 其他 .to(device) 操作都做类似修改
  model = DDP(model)


  # loss 通信
  loss = output.loss
  loss.backward()
  dist.all_reduce(loss, op=dist.ReduceOp.AVG)

  # 只需要在 rank0 的机器上logging
  if os.environ["RANK"] == 0:
    print()

  # accuracy 通信
  dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
  ```

  ```sh
  # 启动
  torchrun --nproc_per_node=2 "path_to_entry.py"
  ```

- Transformers 自己实现了，只要启动就行, 也要记得 shuffle data

  ```py
  dataset.train_test_split(seed=42)
  ```

  ```sh
  # 启动
  torchrun --nproc_per_node=2 "path_to_entry.py"
  ```

## Accelerate 库

接口库，分布式代码的搬运工

### Accelerate 混合精度训练

两个模型，一个半精度前向传播得到 loss (可加速训练), 另一个单精度反向传播更新权重

- 显存占用

  --  | 混合精度训练显存占用 | 单精度训练显存占用
  :-- |:--|:--
  模型 | (4+2) Bytes x P | 4 Bytes x P
  优化器 | 8 Bytes x P | 8 Bytes x P
  梯度 | (2+) Bytes x P (更新时会转成单精度)| 4 Bytes x P
  激活值 | 2 Bytes x A | 4 Bytes x A
  汇总 | (16+) Bytes + 2 Bytes x A | 16 Bytes + 4 Bytes x A

  （当激活值占比比较大时，可以明显降低显存）

- Accelerate 实现:

  - method 1.

    ```py
    accelerator = Accelerator(mixed_precision="bf16")
    ```

  - method 2.

    ```sh
    accelorator config && choice bf16
    ```

  - method 3.

    ```sh
    accelerator launch --mixed_precision bf16 {path_to_entry.py}
    ```
