# vLLM Context Parallel (PCP/DCP) 详细解析

## 一、核心概念

### Context Parallel 是什么？

Context Parallel (CP) 是一种优化 Attention 计算的并行策略，通过将注意力的 KV 缓存分散到多个 GPU 上来减少单卡内存压力。

**两种 CP 模式**：
1. **PCP (Prefill Context Parallel)**: 在预填充阶段使用 CP
2. **DCP (Decode Context Parallel)**: 在解码阶段使用 CP

### 为什么需要 CP？

**问题**：KV 缓存占用大量内存
- 标准做法：每个 GPU 完整存储 KV 缓存 `[seq_len, num_kv_heads, head_dim]`
- 长序列时：KV 缓存成为主要瓶颈（甚至超过模型参数）

**解决方案**：CP 将 KV 头沿 CP 组分散
- 每个 GPU 只存储 `num_kv_heads / cp_size` 个头的 KV 缓存
- Attention 计算时通过 All-Gather 补齐完整信息
- 总内存减少 ~cp_size 倍

---

## 二、技术实现

### 1. 基本思路

```python
# 标准 Attention (无 CP):
Q: [batch, seq_len, num_heads, head_dim]
K, V: [batch, seq_len, num_kv_heads, head_dim] (完整)

attention_scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, seq_len, seq_len]
output = attention_scores @ V               # [batch, num_heads, seq_len, head_dim]

# CP 优化后:
Q_local: [batch, seq_len, num_heads/tp, head_dim]
K_local: [batch, seq_len, num_kv_heads/cp, head_dim] (切分到各 CP rank)
V_local: [batch, seq_len, num_kv_heads/cp, head_dim]

# Prefill 阶段 (PCP):
Q_all = PCP_AllGather(Q_local, dim=head)      # [batch, seq_len, num_heads, head_dim]
K_all = PCP_AllGather(K_local, dim=head)      # [batch, seq_len, num_kv_heads, head_dim]
attention_scores = Q_all @ K_all.transpose()
output = attention_scores @ V_all
# 然后 ReduceScatter 输出，保持 KV 缓存切分

# Decode 阶段 (DCP):
K_all = DCP_AllGather(K_local, dim=head)      # 聚合所有 KV 头
attention_scores = Q @ K_all.transpose()
output = attention_scores @ V_all
output_local = DCP_ReduceScatter(output)      # 再分散输出
```

### 2. KV 缓存存储方式

**重要**：KV 缓存按照特定方式在 CP ranks 间分散

```python
# vllm/config/parallel.py L265-267:
# For `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`,
#     and `total_cp_world_size = pcp_world_size * dcp_world_size`.
# store interleave_size tokens on total_cp_rank i,
# then store next interleave_size tokens on total_cp_rank i+1.

# 示例 (cp_kv_cache_interleave_size=1):
# Token-level alignment:
# CP Rank 0: 存储 Token 0, Token 4, Token 8, ... (stride=cp_world_size)
# CP Rank 1: 存储 Token 1, Token 5, Token 9, ...
# CP Rank 2: 存储 Token 2, Token 6, Token 10, ...
# ...

# 示例 (cp_kv_cache_interleave_size=block_size):
# Block-level alignment:
# CP Rank 0: 存储 Token [0, block_size)
# CP Rank 1: 存储 Token [block_size, 2*block_size)
# ...
```

---

## 三、PCP vs DCP

### DCP (Decode Context Parallel)

**应用场景**：推理阶段（解码）

**特点**：
- KV 缓存头维度切分：`num_kv_heads / dcp_size`
- 每个 DCP rank 存储不同的 KV 头
- Attention 计算需要 All-Gather 补齐完整 KV 头

**工作流程**：
```
解码时：
1. 输入 Q_local: [batch, 1, num_heads/tp, head_dim]
2. All-Gather Q: [batch, 1, num_heads, head_dim]
3. All-Gather K: [batch, seq_len, num_kv_heads, head_dim]
4. All-Gather V: [batch, seq_len, num_kv_heads, head_dim]
5. 计算 Attention (完整)
6. ReduceScatter 输出
```

**关键代码** (vllm/v1/attention/backends/flashinfer.py L1476-1497):
```python
decode_query = get_dcp_group().all_gather(decode_query, dim=-2)
# 得到完整 Q
output_tmp = ... # attention 计算
output_context = cp_lse_ag_out_rs(
    output_context_tmp,
    lse_context_tmp,
    get_dcp_group(),  # DCP group 做 ReduceScatter
)
```

### PCP (Prefill Context Parallel)

**应用场景**：推理阶段（预填充）

**特点**：
- KV 缓存按 Attention head 维度切分
- 序列维度保持完整（不像 Sequence Parallel）
- 减少预填充时的 KV 缓存内存

**工作流程**：
```
预填充时：
1. 输入 Q_local: [batch, seq_len, num_heads/tp, head_dim]
2. All-Gather Q: [batch, seq_len, num_heads, head_dim] (沿 head 维度)
3. All-Gather K: [batch, seq_len, num_kv_heads, head_dim]
4. 计算 Attention (完整)
5. ReduceScatter 输出
6. 同时保存 KV 缓存（切分到各 PCP rank）
```

---

## 四、PCP/DCP 与 TP/DP/EP 的组合

### Rank 映射关系

**关键源码** (vllm/distributed/parallel_state.py L1346-1358):

```python
# 全局 rank 组织结构（从最外层到最内层）：
all_ranks = torch.arange(world_size).reshape(
    -1,                                    # ExternalDP (外部 DP，verl 集成用)
    data_parallel_size,                    # DP (数据并行)
    pipeline_model_parallel_size,          # PP (管道并行)
    prefill_context_model_parallel_size,   # PCP (预填充 CP)
    tensor_model_parallel_size,            # TP (张量并行)
)
```

**Rank 计算公式**：
```
rank = external_dp_idx × (DP×PP×PCP×TP) 
       + dp_idx × (PP×PCP×TP) 
       + pp_idx × (PCP×TP) 
       + pcp_idx × TP 
       + tp_idx
```

### 各并行组的创建逻辑

**1. TP 组** (L1368-1375):
```python
# 将最内层维度（TP）提取出来创建 TP group
group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
# 结果：各个 TP group，每个包含 tp_size 个相邻 rank
# 例如 tp_size=2: [0,1], [2,3], [4,5], [6,7], ...
```

**2. DCP 组** (L1379-1388):
```python
# 将 DCP 维度提取出来创建 DCP group
group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
# 重要：DCP reuses TP group GPUs
# 即：DCP 组内的 rank 是 TP 组的子集或相同集合
# 因此：tp_size 必须 >= dcp_size
```

**3. PCP 组** (L1391-1400):
```python
# 将 PCP 和 TP 维度转置，然后提取 PCP
group_ranks = (
    all_ranks.transpose(3, 4)               # 交换 PCP 和 TP 维度
    .reshape(-1, prefill_context_model_parallel_size)
    .unbind(0)
)
# 结果：各个 PCP group，包含相同 TP rank 但不同 PCP idx 的 ranks
```

**4. PP 组** (L1403-1410):
```python
# 将 PP 和 TP 维度转置，然后提取 PP
group_ranks = (
    all_ranks.transpose(2, 4)               # 交换 PP 和 TP 维度
    .reshape(-1, pipeline_model_parallel_size)
    .unbind(0)
)
```

**5. DP 组** (L1413-1420):
```python
# 将 DP 和 TP 维度转置，然后提取 DP
group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
```

**6. EP 组** (L1423-1435):
```python
# EP 由 DP × PCP × TP 组成
# 即同一 PP 阶段内的所有 DP/PCP/TP ranks
group_ranks = (
    all_ranks.transpose(1, 2)               # 交换 DP 和 PP
    .reshape(
        -1,
        data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size,
    )
    .unbind(0)
)
```

---

## 五、PCP/DCP 与 TP/DP/EP 的并行组结构图

### 示例配置

```
World Size = 32 GPU
DP = 2, PP = 2, PCP = 2, TP = 2, DCP = 2, ExternalDP = 1
```

### Rank 的 5D 组织结构

```
┌─────────────────────────────────────────────────────────────────┐
│ Total 32 ranks organized as:                                    │
│ [ExtDP=1, DP=2, PP=2, PCP=2, TP=2]                             │
└─────────────────────────────────────────────────────────────────┘

Reshaping to 5D:
all_ranks = [[[[rank for TP in range(2)]
               for PCP in range(2)]
              for PP in range(2)]
             for DP in range(2)]

Visual representation:
├─ DP=0
│  ├─ PP=0
│  │  ├─ PCP=0: [0, 1]           (TP ranks)
│  │  └─ PCP=1: [2, 3]           (TP ranks)
│  └─ PP=1
│     ├─ PCP=0: [8, 9]           (TP ranks)
│     └─ PCP=1: [10, 11]         (TP ranks)
└─ DP=1
   ├─ PP=0
   │  ├─ PCP=0: [16, 17]         (TP ranks)
   │  └─ PCP=1: [18, 19]         (TP ranks)
   └─ PP=1
      ├─ PCP=0: [24, 25]         (TP ranks)
      └─ PCP=1: [26, 27]         (TP ranks)
```

### 各并行组详细映射

#### 1️⃣ TP 组（Tensor Parallel Groups）

```
抽取最内层（TP 维度）：
TP_GROUP_0: [0, 1]     (DP=0, PP=0, PCP=0)
TP_GROUP_1: [2, 3]     (DP=0, PP=0, PCP=1)
TP_GROUP_2: [8, 9]     (DP=0, PP=1, PCP=0)
TP_GROUP_3: [10, 11]   (DP=0, PP=1, PCP=1)
TP_GROUP_4: [16, 17]   (DP=1, PP=0, PCP=0)
TP_GROUP_5: [18, 19]   (DP=1, PP=0, PCP=1)
TP_GROUP_6: [24, 25]   (DP=1, PP=1, PCP=0)
TP_GROUP_7: [26, 27]   (DP=1, PP=1, PCP=1)

总数：8 个 TP 组
每组大小：2 个 rank
```

#### 2️⃣ DCP 组（Decode Context Parallel Groups）

```
DCP reuses TP group GPUs，reshape(-1, dcp_size):
DCP_GROUP_0: [0, 2]        (TP ranks 0和2的组合)
DCP_GROUP_1: [1, 3]        (TP ranks 1和3的组合)
DCP_GROUP_2: [8, 10]       (TP ranks 8和10的组合)
DCP_GROUP_3: [9, 11]
...

关键点：DCP 在每个 TP group 内部分割
- 如果 dcp_size=2, tp_size=2，则每个 TP group 变成 2 个 DCP group
- 关系：tp_size % dcp_size == 0（TP size 必须被 DCP size 整除）
```

#### 3️⃣ PCP 组（Prefill Context Parallel Groups）

```
转置 PCP 和 TP 维度，然后提取 PCP：
transpose(3, 4) 使 PCP 成为最内层

PCP_GROUP_0: [0, 8]        (DP=0, PP=0, TP=0，但PCP=0,1)
PCP_GROUP_1: [2, 10]       (DP=0, PP=0, TP=1，但PCP=0,1)
PCP_GROUP_2: [16, 24]      (DP=1, PP=0, TP=0，但PCP=0,1)
PCP_GROUP_3: [18, 26]      (DP=1, PP=0, TP=1，但PCP=0,1)
...

关键点：PCP group 包含相同 DP/PP/TP 但不同 PCP idx 的 ranks
- 这些 rank 在不同物理 GPU 上
- 用于预填充时的 KV 缓存分散
```

#### 4️⃣ PP 组（Pipeline Parallel Groups）

```
转置 PP 和 TP 维度，然后提取 PP：
PP_GROUP_0: [0, 8]         (DP=0, TP=0，但PP=0,1)
PP_GROUP_1: [1, 9]         (DP=0, TP=1，但PP=0,1)
PP_GROUP_2: [16, 24]       (DP=1, TP=0，但PP=0,1)
PP_GROUP_3: [17, 25]       (DP=1, TP=1，但PP=0,1)
...

关键点：PP group 包含同一流水线阶段序列内的 ranks
```

#### 5️⃣ DP 组（Data Parallel Groups）

```
转置 DP 和 TP 维度，然后提取 DP：
DP_GROUP_0: [0, 16]        (PP=0, PCP=0, TP=0，但DP=0,1)
DP_GROUP_1: [2, 18]        (PP=0, PCP=1, TP=0，但DP=0,1)
DP_GROUP_2: [1, 17]        (PP=0, PCP=0, TP=1，但DP=0,1)
DP_GROUP_3: [3, 19]        (PP=0, PCP=1, TP=1，但DP=0,1)
...

关键点：DP group 包含同一批数据复本内的 ranks
```

#### 6️⃣ EP 组（Expert Parallel Groups）

```
EP = DP × PCP × TP（同一 PP 阶段）
EP_GROUP_0: [0, 1, 2, 3, 8, 9, 10, 11]    (DP=0, PP=0)
EP_GROUP_1: [16, 17, 18, 19, 24, 25, 26, 27]  (DP=1, PP=0)
EP_GROUP_2: [0, 1, 2, 3, 16, 17, 18, 19] (PP=0)？ 不，EP应该是：
EP_GROUP_0: [0, 1, 2, 3, 8, 9, 10, 11]    (DP=0, PP=0)
EP_GROUP_1: [16, 17, 18, 19, 24, 25, 26, 27]  (DP=1, PP=0)
EP_GROUP_2: [0, 1, 2, 3, 16, 17, 18, 19]？

让我重新计算...
```

### 完整的并行组结构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  World Rank Layout (DP=2, PP=2, PCP=2, TP=2):                              │
│                                                                               │
│  DP=0, PP=0                    DP=0, PP=1                                   │
│  ┌──────────┬──────────┐       ┌──────────┬──────────┐                       │
│  │ PCP=0    │ PCP=1    │       │ PCP=0    │ PCP=1    │                       │
│  ├──┬──┤ ├──┬──┤       ├──┬──┤ ├──┬──┤                       │
│  │0 │1 │ │2 │3 │       │8 │9 │ │10│11│                       │
│  └──┴──┘ └──┴──┘       └──┴──┘ └──┴──┘                       │
│   TP     TP             TP     TP                             │
│                                                               │
│  DP=1, PP=0                    DP=1, PP=1                    │
│  ┌──────────┬──────────┐       ┌──────────┬──────────┐       │
│  │ PCP=0    │ PCP=1    │       │ PCP=0    │ PCP=1    │       │
│  ├──┬──┤ ├──┬──┤       ├──┬──┤ ├──┬──┤       │
│  │16│17│ │18│19│       │24│25│ │26│27│       │
│  └──┴──┘ └──┴──┘       └──┴──┘ └──┴──┘       │
│   TP     TP             TP     TP            │
│                                               │
└─────────────────────────────────────────────────────────────────────────────┘

并行组关系：
┌──────────┐     ┌──────────┐     ┌──────────┐
│ TP Group │     │ PP Group │     │ DP Group │
├──────────┤     ├──────────┤     ├──────────┤
│ [0,1]    │     │ [0,8]    │     │ [0,16]   │
│ [2,3]    │     │ [1,9]    │     │ [2,18]   │
│ [8,9]    │     │ [2,10]   │     │ [1,17]   │
│ [10,11]  │     │ [3,11]   │     │ [3,19]   │
│ [16,17]  │     │ [16,24]  │     │ [8,24]   │
│ [18,19]  │     │ [17,25]  │     │ [10,26]  │
│ [24,25]  │     │ [18,26]  │     │ [9,25]   │
│ [26,27]  │     │ [19,27]  │     │ [11,27]  │
└──────────┘     └──────────┘     └──────────┘

┌──────────┐     ┌──────────┐
│ PCP Group│     │ DCP Group│
├──────────┤     ├──────────┤
│ [0,8]    │     │ [0,2]    │
│ [2,10]   │     │ [1,3]    │
│ [1,9]    │     │ [8,10]   │
│ [3,11]   │     │ [9,11]   │
│ [16,24]  │     │ [16,18]  │
│ [18,26]  │     │ [17,19]  │
│ [17,25]  │     │ [24,26]  │
│ [19,27]  │     │ [25,27]  │
└──────────┘     └──────────┘
```

### 通信流向示例

**假设**：Rank 0 执行 Attention 计算

```
1. Rank 0 所属的各个组：
   - TP_GROUP: [0, 1]         (Tensor Parallel)
   - DCP_GROUP: [0, 2]        (Decode Context Parallel)
   - PCP_GROUP: [0, 8]        (Prefill Context Parallel)
   - PP_GROUP: [0, 8]         (Pipeline Parallel)
   - DP_GROUP: [0, 16]        (Data Parallel)
   - EP_GROUP: [0,1,2,3,8,9,10,11] (Expert Parallel)

2. Attention 通信：
   DCP phase (解码):
   - Rank 0 的 Q: [batch, 1, num_heads/2, head_dim]
   - All-Gather(Q) 从 DCP_GROUP [0, 2]:
     * Gather: Rank 0 和 Rank 2 的 Q
     * 得到: [batch, 1, num_heads, head_dim]
   - 计算 Attention
   - ReduceScatter(output) 到 DCP_GROUP [0, 2]

3. KV Cache 存储：
   PCP phase (预填充):
   - Rank 0 计算 K, V: [batch, seq_len, num_kv_heads/2, head_dim]
   - 存储到 Rank 0 的 KV Cache
   - Rank 8 也计算相同 query 的 K, V
   - 下次解码时：
     * Rank 0 的 DCP_GROUP [0, 2] All-Gather KV
     * 从 Rank 0 和 Rank 2 的 KV Cache 中取出
```

---

## 六、总结：PCP/DCP 的并行组关系

| 并行维度 | 创建方式 | 包含 Rank 数 | 用途 |
|---------|--------|-----------|------|
| **TP** | `reshape(-1, tp_size)` | tp_size | 权重 ColumnParallel/RowParallel |
| **DCP** | `reshape(-1, dcp_size)` | dcp_size | 解码 Attention KV 缓存分散 |
| **PCP** | `transpose(3,4) + reshape(-1, pcp_size)` | pcp_size | 预填充 Attention KV 缓存分散 |
| **PP** | `transpose(2,4) + reshape(-1, pp_size)` | pp_size | 流水线阶段 |
| **DP** | `transpose(1,4) + reshape(-1, dp_size)` | dp_size | 数据并行（批数据复本） |
| **EP** | `transpose(1,2) + reshape(-1, dp×pcp×tp)` | dp×pcp×tp | MoE 专家分散 |

### 关键特性

1. **DCP Reuses TP Group**
   - DCP 不增加 world size（复用 TP group 的 GPU）
   - `tp_size >= dcp_size`（必须满足）

2. **PCP 的 All-Gather/ReduceScatter**
   - Prefill 时：All-Gather 聚合所有 PCP rank 的 Q；ReduceScatter 分散输出
   - KV 缓存切分存储（按 head 维度）

3. **EP 跨越 DP × PCP × TP**
   - 同一 PP 阶段内的所有数据并行/CP/张量并行 ranks
   - 用于 MoE 的 All-to-All 通信

4. **Rank 映射遵循 5D 笛卡尔积**
   - 维度顺序：[ExternalDP, DP, PP, PCP, TP]
   - 转置操作用于改变维度优先级以创建不同的通信组
