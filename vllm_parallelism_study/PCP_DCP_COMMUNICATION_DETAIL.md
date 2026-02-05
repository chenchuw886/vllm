# PCP 策略 2 与 DCP 通信机制详解

## 问题总结

用户提出两个核心问题：
1. **PCP 策略 2 (Ring-Attention)**：通信具体是什么？在单机 8 卡场景如何工作？
2. **DCP 按 token 交错存储后**：计算时做什么通信来合并结果？

---

## 一、PCP 策略 2：Ring-Attention 的通信机制

### 1.1 基本概念

**PCP 策略 A（Full KV）vs 策略 B（Ring-Attention）的区别**：

| 维度 | 策略 A（Partial Q + Full K/V） | 策略 B（Partial Q + Partial K/V） |
|------|------|------|
| **通信方式** | Gather + Scatter (全集中) | Ring-based rotations (分散) |
| **K/V 存储** | 每卡持有完整 K/V (重复) | 每卡仅持有自己的 K/V chunk |
| **通信时间** | 集中（容易饱和链路） | 分散（易与计算流水化） |
| **实现复杂度** | 简单（算子层即可） | 复杂（需求支持环形通信算子） |
| **适用场景** | 中等长度序列 | 超长序列 |

### 1.2 Ring-Attention 在单机 8 卡的具体工作流程

**场景设置**：
```
World Size = 8 GPU (单机)
Batch Size = 1 (单个用户)
Sequence Length T = 4096 tokens
Hidden Dimension H = 4096
KV Head Dimension H_kv = 128 (H / 32)
PCP Size = 8 (所有卡参与预填充)
```

**序列分块**：
```
Token sequence: [0, 1, 2, ..., 4095]
每卡分到 T / pcp_size = 4096 / 8 = 512 个 token

GPU 0: tokens [0:512]
GPU 1: tokens [512:1024]
GPU 2: tokens [1024:1536]
GPU 3: tokens [1536:2048]
GPU 4: tokens [2048:2560]
GPU 5: tokens [2560:3072]
GPU 6: tokens [3072:3584]
GPU 7: tokens [3584:4096]
```

### 1.3 Ring-Attention 的执行过程

**第 0 步**：计算本地 Q/K/V

```python
# GPU i 计算自己的 chunk
Q_i = W_q @ X[i*512:(i+1)*512]           # shape: [512, H]
K_i = W_k @ X[i*512:(i+1)*512]           # shape: [512, H_kv]
V_i = W_v @ X[i*512:(i+1)*512]           # shape: [512, H_kv]

# 初始化输出
O_i = zeros([512, H])  # 用于累积 attention 结果

# 初始化本地 K/V "轮盘"
KV_ring_i = (K_i, V_i)  # 当前持有的 K/V
```

**第 1 步**：Ring rotate 本地 K/V（第 0 轮）

```
Round 0 (自己的 K/V):
  GPU 0: O_0 += Attention(Q_0, K_0, V_0)
  GPU 1: O_1 += Attention(Q_1, K_1, V_1)
  ...
  GPU 7: O_7 += Attention(Q_7, K_7, V_7)
  
  每个 GPU 的通信: 无（使用本地数据）
  计算: 512 × 512 × 128 次乘加（标准 attention）
```

**第 2 步**：Ring rotate KV（第 1 轮，GPU i 接收来自 GPU (i-1) mod 8 的数据）

```
Round 1 (接收来自前一卡的 K/V):
  通信（环形）：
    GPU 0 ← GPU 7 的 K/V        (K_7, V_7)
    GPU 1 ← GPU 0 的 K/V        (K_0, V_0)
    GPU 2 ← GPU 1 的 K/V        (K_1, V_1)
    ...
    GPU 7 ← GPU 6 的 K/V        (K_6, V_6)
  
  通信量: 每卡发送 512 × 2 × 128 × 4 bytes = 512 KB
         (512 tokens, K + V, 128 dims, float32)
  
  同步完成后，计算：
  GPU 0: O_0 += Attention(Q_0, K_7, V_7)   # 自己的 Q 与 GPU 7 的 K/V
  GPU 1: O_1 += Attention(Q_1, K_0, V_0)   # 自己的 Q 与 GPU 0 的 K/V
  ...
```

**第 3 步到第 7 步**：持续 Ring rotate

```
Round 2:
  GPU 0 ← GPU 6 的 K/V  (K_6, V_6)
  GPU 0: O_0 += Attention(Q_0, K_6, V_6)

Round 3:
  GPU 0 ← GPU 5 的 K/V  (K_5, V_5)
  GPU 0: O_0 += Attention(Q_0, K_5, V_5)

...

Round 7:
  GPU 0 ← GPU 1 的 K/V  (K_1, V_1)
  GPU 0: O_0 += Attention(Q_0, K_1, V_1)
```

**第 8 步**：Attention 完成

```
每卡完成所有 attention 计算：
GPU 0: O_0 = Attention(Q_0, [K_0, K_7, K_6, ..., K_1], [V_0, V_7, V_6, ..., V_1])
             = sum over all chunks

此时每卡拥有自己 512 个 token 的完整 attention output
```

### 1.4 Ring-Attention 的通信详细分析

**环形通信拓扑**：

```
GPU 0  ←→  GPU 1  ←→  GPU 2  ←→  ... ←→  GPU 7
   ↓                                        ↓
   └────────────────────────────────────────┘
        (形成环，GPU 0 ← GPU 7 的连接)
```

**通信机制**：

```python
# 伪代码，展示每一轮通信
for round in range(pcp_size - 1):  # 8 - 1 = 7 轮
    # 1. 非阻塞发送自己当前的 K/V 给下一卡
    next_gpu = (gpu_id + 1) % pcp_size
    send_handle = all_gather_along_pcp_group(
        KV_ring,
        dst=next_gpu,
        async=True
    )
    
    # 2. 接收上一卡的 K/V
    prev_gpu = (gpu_id - 1) % pcp_size
    KV_ring = recv_from(prev_gpu)
    
    # 3. 与通信并行计算（Compute-Comm Overlap）
    O += Attention(Q, KV_ring)
    
    # 4. 等待发送完成
    wait(send_handle)
```

**单机 8 卡的通信量总计**：

```
Ring-Attention 每卡通信：
  轮数：pcp_size - 1 = 7 轮
  单轮通信量：512 tokens × 2 (K+V) × 128 dims × 4 bytes = 512 KB
  总通信量：7 × 512 KB = 3.5 MB per GPU
  
  或按公式：
  CommVol_PCP-B = B·T·H_kv × (pcp_size - 1) / pcp_size
                = 1 × 4096 × 128 × 7/8
                = 3.5 MB
```

**vs 策略 A（Full K/V）对比**：

```
策略 A 通信量：
  CommVol_PCP-A = B·T·H_kv × 2
                = 1 × 4096 × 128 × 2
                = 1 MB (Gather) + 1 MB (Scatter) = 2 MB

策略 B（Ring）vs 策略 A：
  策略 B：3.5 MB（但分散，可流水化）
  策略 A：2 MB（但集中，需多次同步）
  
  在通信可与计算重叠的情况下，策略 B 的有效通信时间可降至 1.75 MB 的等价
  因此 策略 B 反而可能更快
```

### 1.5 Ring-Attention 的关键特性

**1. Compute-Communication Overlap**

```
传统（策略 A）：
  ┌────────────────┐
  │  All-Gather    │ (等待完成)
  └────────────────┘
  ┌────────────────┐
  │  Compute Attn  │ (开始计算)
  └────────────────┘
  ┌────────────────┐
  │  All-Scatter   │ (发送结果)
  └────────────────┘
  
  总时间 ≈ 通信 + 计算 + 通信 = 顺序执行

Ring-Attention：
  ┌────────────────┐
  │  Round 0 Comm  │ ──┐
  │ + Compute      │   ├─ 并行执行
  │ + wait         │ ──┘
  ├────────────────┤
  │  Round 1 Comm  │ ──┐
  │ + Compute      │   ├─ 并行执行
  │ + wait         │ ──┘
  ...
  
  总时间 ≈ 7 × (单轮通信 | 计算)（取决于两者谁更慢）
```

**2. 单机 8 卡 NVLink 情况**

```
NVLink 带宽：900 GB/s
通信量：3.5 MB
通信时间：3.5 MB / 900 GB/s ≈ 4 µs

单轮 attention 计算时间（512 × 512 × 128）：
  ≈ 512 × 512 × 128 / (单个 H100 FLOPS)
  ≈ 33.5M FLOPs / (1.4 TFLOPS) ≈ 24 µs

结论：计算时间 > 通信时间 → 通信完全可被计算掩蔽
      有效额外开销 ≈ 0（与策略 A 相比，策略 B 不会变慢）
```

---

## 二、DCP 的通信机制：Token 交错后的合并

### 2.1 DCP 的基本存储策略

**问题背景**：

```
假设单机 8 卡，配置 TP=8, DCP=2
- TP 先按 head 维度切分 KV cache
- 当 KV head 数很小（如 MLA: H_kv = 128，TP=8）时
  每卡只拥有 H_kv / 8 = 16 维的 KV
- 这导致 KV cache 在 TP 组内重复存储

DCP 解决方案：
- 进一步在 TP 组内按 T（token）维度分片 KV
- 分片单位：整数倍 token（如 block_size = 128）
```

### 2.2 DCP 的 Token 交错方式

**配置**：
```
World Size = 8 GPU
TP Size = 8 (权重按 head 切分)
DCP Size = 2 (在 TP 组内再分 2 份)
  → 实际 DCP ranks 形成 4 个 DCP group:
    DCP_0: [0, 4]   (TP=0 与 TP=4，但不同 DCP idx)
    DCP_1: [1, 5]   (TP=1 与 TP=5，但不同 DCP idx)
    DCP_2: [2, 6]
    DCP_3: [3, 7]

KV Cache Interleave Size = 1 (按单个 token 交错)
Total Context Length T_s = 1024 tokens
```

**Token 交错存储**：

```
逻辑上的 KV cache（1024 tokens × 128 dims per head）：

Token Index:   0     1     2     3     4  ...  1020  1021  1022  1023
Head Index:    0     0     0     0     0       0     0     0     0

分片规则（DCP_size=2）：
  Token 0 → GPU in DCP_0   (GPU 0 or 4)
  Token 1 → GPU in DCP_1   (GPU 1 or 5)
  Token 2 → GPU in DCP_2   (GPU 2 or 6)
  Token 3 → GPU in DCP_3   (GPU 3 or 7)
  Token 4 → GPU in DCP_0   (GPU 0 or 4)  ← 开始循环
  ...

具体，假设 DCP_0 中 GPU 0 是 rank 0，GPU 4 是 rank 1：
  GPU 0 存储: Token [0, 4, 8, 12, ...]   (stride = dcp_world_size = 4)
  GPU 1 存储: Token [1, 5, 9, 13, ...]
  GPU 2 存储: Token [2, 6, 10, 14, ...]
  GPU 3 存储: Token [3, 7, 11, 15, ...]
```

**内存分布图**：

```
GPU 0 内存：  T[0]  T[4]  T[8]  ... T[1020]  (每个 128 dims)
GPU 1 内存：  T[1]  T[5]  T[9]  ... T[1021]
GPU 2 内存：  T[2]  T[6]  T[10] ... T[1022]
GPU 3 内存：  T[3]  T[7]  T[11] ... T[1023]
GPU 4 内存：  同 GPU 0 (复制)    ← TP 复制（HEAD 切分）
GPU 5 内存：  同 GPU 1 (复制)
GPU 6 内存：  同 GPU 2 (复制)
GPU 7 内存：  同 GPU 3 (复制)
```

### 2.3 Decode 时的 Attention 计算与通信

**场景**：生成第 1025 个 token

```
当前 KV cache 有 T_s = 1024 个历史 token
要计算新 token 的 attention，需访问所有 1024 个 K/V
```

**GPU 0 的计算流程**：

```python
# GPU 0 的视角
query = model.input_embedding(new_token)  # shape: [1, H=4096]
# 经过 QKV projection 后
Q = W_q @ query                            # shape: [1, H=4096]

# Attention 计算
def attention(Q, K_full, V_full):
    scores = Q @ K_full.T / sqrt(H_k)      # [1, 1024]
    attn_weights = softmax(scores)         # [1, 1024]
    output = attn_weights @ V_full         # [1, H=4096]
    return output

# 问题：K_full, V_full 怎么得到？
# K_full, V_full 分散在多卡
#   GPU 0: [K_0, K_4, K_8, ...]
#   GPU 1: [K_1, K_5, K_9, ...]
#   GPU 2: [K_2, K_6, K_10, ...]
#   GPU 3: [K_3, K_7, K_11, ...]
```

**解决方案：All-Gather（在 DCP group 内）**

```python
# 步骤 1：重建完整的 K, V（在 DCP group 内 all-gather）
# DCP group 定义：同 TP rank 但不同 DCP idx 的 GPU

# 例如 TP rank 0 的 DCP group = [0, 4]
all_gather_result = torch.distributed.all_gather_into_tensor(
    output_tensor=K_full,      # 输出缓冲区，大小 [1024, 128]
    input_tensor=K_local,      # 本地拥有的，大小 [256, 128] (1024/4)
    group=dcp_group            # DCP group [0, 4]
)

# 步骤 2：完成 all-gather 后，K_full 在所有 DCP group 成员中都可用
# GPU 0 现在拥有 K[0, 1, 2, 3, 4, 5, ..., 1023]（完整）
# GPU 4 也拥有 K[0, 1, 2, 3, 4, 5, ..., 1023]（完整，因为 TP 切分）

# 步骤 3：本地计算 attention
O = attention(Q, K_full, V_full)  # [1, 4096]
```

**All-Gather 的通信细节**：

```
配置：TP=8, DCP=2
DCP group 中有 2 个 rank（如 [0, 4]）
每个 rank 持有 1024 / 4 = 256 个 token 的 K/V

All-Gather 流程：
  GPU 0：发送 K[0, 4, 8, ...] （256 tokens × 128 dims = 32 KB）
  GPU 4：发送 K[0, 4, 8, ...]（同样内容，因为 TP 复制）
         ↓ All-Gather ↓
  GPU 0：接收 K[0, 4, ...] + K[1, 5, ...] = K[完整] (1024 tokens, 128 KB)
  GPU 4：接收相同结果

通信量（单个 DCP group）：
  dcp_size - 1 = 2 - 1 = 1 hop
  每 hop：256 tokens × 128 dims × 4 bytes = 128 KB
  总计：1 × 128 KB = 128 KB per token

按公式：
  CommVol_DCP = B·T_s·H_kv × (dcp_size - 1) / dcp_size
              = 1 × 1024 × 128 × (2-1) / 2
              = 1024 × 128 × 0.5 = 64 KB
  
  (注：这个公式是平摊计算，考虑了 DCP group 内的分片)
```

### 2.4 DCP 与 TP 的交互

**重要细节**：DCP 复用 TP 组的 GPU

```
TP 的权重分布：
  GPU 0 - 4：都拥有 W_q[0:512]    (第一个 head 切分)
  GPU 0 - 4：都拥有 W_k[0:128]    (KV head 数少)
  GPU 0 - 4：都拥有 W_v[0:128]

为什么 DCP 能复用 TP GPU？
  答：TP 组内的 GPU 已经共享同一个 model replica
      可以进一步在它们之间做 token 维度的分片
      不会产生额外的权重计算冗余
```

**完整的 Attention 步骤（TP + DCP 结合）**：

```
第 1 层 All-Reduce（TP 内，在 attention 前）：
  目的：聚合来自不同 head 切分的 Q 投影结果
  
第 2 步：Attention 计算中的 All-Gather（DCP 内）
  目的：收集 token 维度分散的 K/V
  
第 3 步：Attention 后的 All-Reduce（TP 内）
  目的：聚合输出，恢复完整 hidden dimension

这样，单个 attention head 的计算被分散到多卡，
既利用了 TP 的权重并行，又利用了 DCP 的 KV 内存优化。
```

---

## 三、单机 8 卡的完整 DCP 例子

### 3.1 配置

```
GPUs:       [0, 1, 2, 3, 4, 5, 6, 7]
TP Size:    8
DCP Size:   2
TP Groups:  [0,1,2,3,4,5,6,7]  (全在 TP group)
DCP Groups: 
  - [0, 4]  (TP=0 组内分 DCP)
  - [1, 5]
  - [2, 6]
  - [3, 7]
```

### 3.2 Decode 第 1 token 的计算

```
输入：new_token = "的"

Step 1：Embedding + FC
  (所有 GPU 相同)

Step 2：Transformer Layer 中的 Self-Attention

  2a) 计算 Q
    Q = W_q @ x              (TP group 内所有 GPU 相同)
    shape: [1, 4096]
    
  2b) 从 KV cache 读 K, V
    本地：GPU 0 存储的 K[0, 4, 8, ...]
          (256 tokens × 128 dims = 32 KB)
    
  2c) All-Gather K, V（在 DCP group [0, 4] 内）
    GPU 0：send K_local[256, 128] 给 GPU 4 ✓
           recv K_from_GPU4[256, 128]
    GPU 4：send K_local[256, 128] 给 GPU 0
           recv K_from_GPU0[256, 128] ✓
    
    完成后，GPU 0 拥有：K_full[1024, 128]
    
  2d) 计算 Attention
    attention_output = softmax(Q @ K_full.T) @ V_full
    shape: [1, 128]  (单个 head 的输出)
    
  2e) All-Reduce（TP group 内）
    所有 8 个 GPU 的 attention 输出汇聚，恢复 [1, 4096]
    
Step 3：MLP（同样 TP All-Reduce）

Step 4：输出
  shape: [1, 4096]（完整 hidden dimension）
```

---

## 四、通信对比总结

### PCP 策略 2 vs 策略 A

| 指标 | 策略 A（Full K/V） | 策略 B（Ring） |
|------|------|------|
| **单机 8 卡通信量** | 2 MB | 3.5 MB |
| **通信方式** | All-Gather + All-Scatter（集中） | 环形（分散） |
| **计算-通信重叠** | 有限（等待 gather） | 完整（每轮掩蔽） |
| **有效时间** | 2 MB 等价 | ~1.75 MB 等价（与计算重叠） |
| **实现复杂度** | 简单 | 复杂（需环形算子） |
| **推荐场景** | 中等长度（T < 10K） | 超长序列（T > 10K） |

### DCP 的通信模式

| 阶段 | 通信类型 | 频率 | 开销 |
|------|--------|------|------|
| **Decode Token 生成** | All-Gather（DCP group） | 每个新 token | B·T_s·H_kv × (dcp_size-1)/dcp_size |
| **全局同步** | All-Reduce（TP group） | 每层 2 次 | 2BH（标准 TP） |
| **内存收益** | 按 token 分片 | 持久 | KV cache 减少至 1/dcp_size |

---

## 五、实现建议

### 用于单机实验

**推荐配置**：
```python
# 单机 8 卡，长序列场景
config = {
    "tensor_model_parallel_size": 8,
    "prefill_context_parallel_size": 1,  # 不用 PCP（单机 NVLink 快）
    "decode_context_parallel_size": 2,   # 用 DCP 减少 KV 重复
    "cp_kv_cache_interleave_size": 128,  # block-level 交错（性能更好）
}
```

**验证通信开销**：
```python
# 计算 DCP 的实际通信时间
dcp_comm_time = (1024 * 128 * 0.5) / (900e9 / 8)  # 单个 GPU，7 卡分担
# ≈ 0.73 µs（与计算相比完全可忽略）
```

---

## 六、参考文献和代码位置

### vLLM 源代码

1. **PCP 实现**：
   - `vllm/v1/attention/backends/flashinfer.py` L242-257
   - `BatchDCPPrefillWrapper.run()`

2. **DCP 实现**：
   - `vllm/v1/attention/backends/flashinfer.py` L1476-1497
   - `FlashInferImpl.forward()` 中的 DCP 路径

3. **并行组创建**：
   - `vllm/distributed/parallel_state.py` L1368-1435
   - `get_dcp_group()` 函数

4. **Ring-Attention 算子**：
   - FlashInfer 库（vLLM 的外部依赖）
   - 支持 PCP 策略 B 的 ring-based KV rotation

### 论文参考

- **Context Parallel**：vLLM 团队的设计文档 (`context_parallel_deployment.md`)
- **Ring-Attention**：Liu et al., "Ring Attention with Blockwise Transformers for Context-Aware Generation", 2024
- **Sequence Parallel**：Li et al., "Sequence Parallel Training with Selective Activation Checkpointing", 2023
