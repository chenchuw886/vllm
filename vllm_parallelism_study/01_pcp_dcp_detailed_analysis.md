# vLLM Context Parallel (PCP/DCP) 详细解析

## 一、核心概念

### Context Parallel 是什么？

Context Parallel (CP) 针对 **Prefill** 与 **Decode** 两个阶段的不同瓶颈，分别设计不同的并行方式。

**两种 CP 模式**：
1. **PCP (Prefill Context Parallel)**: 将长序列预填充分摊到多卡，降低 TTFT
2. **DCP (Decode Context Parallel)**: 在解码阶段分片 KV cache，减少重复和内存占用

### 为什么需要 CP？

**Prefill 问题**：长序列预填充导致 TTFT 过高
- 预填充需要对 $T$ 个新 token 计算 Q/K/V
- 若单卡执行，TTFT 随 $T$ 线性增长

**Decode 问题**：KV cache 占用大量内存，限制并发
- KV cache 随 $T$ 增长，长上下文时成为主要瓶颈
- 当 kv-head 数 $H$ 很小（如 MLA/GQA），单纯加大 TP 会导致 KV cache 重复

**解决方案**：
- **PCP**：把 $T$ 个 token 分块到多卡并行预填充
- **DCP**：在 TP 已按 head 切分后，再沿 $T$ 维度分片 KV cache，减少重复

---

## 二、技术实现

### 1. 基本思路

#### PCP（Prefill）两种策略

**策略 A：Partial Q + Full K/V**（中等长度）
```text
1) 将长度为 T 的请求切成 N 个 chunk（N=PCP size）
2) 每张卡计算自己 chunk 的 Q/K/V
3) 全量收集 K/V，得到完整 K/V
4) 每张卡仅对自己的 Q chunk 计算 Attention
```

**策略 B：Partial Q + Partial K/V**（超长长度）
```text
1) 每张卡只持有自己的 Q/K/V chunk
2) 通过 ring-attention 等方法分块交换 K/V
3) 逐块完成 Attention，避免全量 K/V 常驻
```

#### DCP（Decode）核心思路
```text
1) TP 先按 kv-head 维度切分权重与 KV cache（基础 TP）
2) 当 H_kv 很小、tp_size 很大时，KV cache 会发生重复
3) 再沿 T 维度分片 KV cache（DCP），降低重复
4) 访问跨分片 KV 时产生通信开销（随 dcp_size 增大）
```

### 2. KV 缓存存储方式

**重要**：DCP 会把 KV cache 沿 T 维度在 CP ranks 间交错分布

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
- TP 已按 kv-head 切分 KV cache
- 当 $H_{kv}$ 很小而 tp_size 很大时，KV cache 会在多卡重复
- DCP 再沿 $T$ 维度分片 KV cache，减少重复
- `dcp_size ∈ [1, tp_size / H_{kv}]`，dcp 越大通信越多

**工作流程**（概念级）：
```
解码时：
1. KV cache 先按 TP 切分，再按 T 维度分片（DCP）
2. 本地计算需要访问跨分片 KV 时触发通信
3. 通过交错分片（interleave）支持 KV 动态增长
```

**关键代码** (vllm/v1/attention/backends/flashinfer.py L1476-1497):
```text
注意力后端会按 DCP 分片策略访问 KV，必要时触发跨分片通信。
```

### PCP (Prefill Context Parallel)

**应用场景**：推理阶段（预填充）

**特点**：
- 将长序列预填充按 token chunk 并行化
- 以降低 TTFT 为目标，而非仅减少内存
- 两种策略：Partial Q + Full K/V 或 Partial Q + Partial K/V（ring-attention）

**工作流程**（概念级）：
```
预填充时：
1. 将长度为 T 的请求切成 N 个 chunk（N=PCP size）
2. 每张卡计算自己的 Q/K/V chunk
3. 选择：
   - 收集全量 K/V 后计算本地 Q chunk（Partial Q + Full K/V）
   - 或通过 ring-attention 分块交换 K/V（Partial Q + Partial K/V）
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
   - KV cache 先按 TP 切分，再按 T 维度分片（DCP）
   - Rank 0 在需要跨分片 KV 时与 DCP_GROUP [0, 2] 通信
   - 通信量随 dcp_size 增大而增加

3. KV Cache 存储：
   PCP phase (预填充):
   - Rank 0 只计算并保存自己 chunk 的 Q/K/V
   - 若采用 Partial Q + Full K/V，会在计算时收集 K/V
   - 若采用 ring-attention，则以分块方式交换 K/V
```

---

## 六、总结：PCP/DCP 的并行组关系

| 并行维度 | 创建方式 | 包含 Rank 数 | 用途 |
|---------|--------|-----------|------|
| **TP** | `reshape(-1, tp_size)` | tp_size | 权重 ColumnParallel/RowParallel |
| **DCP** | `reshape(-1, dcp_size)` | dcp_size | 解码阶段 KV cache 的 T 维分片 |
| **PCP** | `transpose(3,4) + reshape(-1, pcp_size)` | pcp_size | 预填充 token chunk 并行 |
| **PP** | `transpose(2,4) + reshape(-1, pp_size)` | pp_size | 流水线阶段 |
| **DP** | `transpose(1,4) + reshape(-1, dp_size)` | dp_size | 数据并行（批数据复本） |
| **EP** | `transpose(1,2) + reshape(-1, dp×pcp×tp)` | dp×pcp×tp | MoE 专家分散 |

### 关键特性

1. **DCP Reuses TP Group**
   - DCP 不增加 world size（复用 TP group 的 GPU）
   - `tp_size >= dcp_size`（必须满足）

2. **PCP 的两类策略**
   - Partial Q + Full K/V：收集全量 K/V 后计算本地 Q chunk
   - Partial Q + Partial K/V：ring-attention 分块交换 K/V

3. **EP 跨越 DP × PCP × TP**
   - 同一 PP 阶段内的所有数据并行/CP/张量并行 ranks
   - 用于 MoE 的 All-to-All 通信

4. **Rank 映射遵循 5D 笛卡尔积**
   - 维度顺序：[ExternalDP, DP, PP, PCP, TP]
   - 转置操作用于改变维度优先级以创建不同的通信组
