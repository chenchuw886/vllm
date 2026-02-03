# MoE 专家层切分与通信算子详细说明

## 一、权重切分基础概念

### 1. Column Parallel (列并行)
- **切分维度**: 沿输出维度(dim 0 或 columns)切分权重
- **权重形状变化**: `[hidden_size, intermediate_size]` → `[hidden_size, intermediate_size/tp_size]`
- **特点**: 
  - 输入不需要切分,完整复制到各 rank
  - 输出是部分结果,需要后续聚合
  - 无需前置通信算子

### 2. Row Parallel (行并行)
- **切分维度**: 沿输入维度(dim 1 或 rows)切分权重
- **权重形状变化**: `[intermediate_size, hidden_size]` → `[intermediate_size/tp_size, hidden_size]`
- **特点**: 
  - 输入需要切分(通常由前一层的 Column Parallel 提供)
  - 输出是完整形状但值是部分和,需要 All-Reduce
  - 后置 All-Reduce 通信算子

## 二、MoE 专家权重切分详细表格

| 层类型 | 并行策略 | 权重组件 | 原始形状 | 切分方式 | 切分后形状 | 前置通信 | 后置通信 | 说明 |
|--------|---------|---------|---------|---------|-----------|---------|---------|------|
| **Attention 层** | TP only | Q, K, V | `[hidden, heads*head_dim]` | ColumnParallel | `[hidden, heads*head_dim/tp]` | None | None | Q/K/V 按 head 切分 |
| | | O (output) | `[heads*head_dim, hidden]` | RowParallel | `[heads*head_dim/tp, hidden]` | None | All-Reduce | 输出投影 |
| | DP + TP | 同上 | 同上 | 同上 | 同上 | None | All-Reduce | DP 间独立,TP 内通信 |
| **MoE Expert 层** | TP only (无 EP) | **gate (router)** | `[hidden, num_experts]` | **Replicate** | `[hidden, num_experts]` (每个rank相同) | None | None | **Router层，不切分**！完整复制。必须让所有rank得到相同的router logits，才能正确计算top-k。注意：router ≠ w1 |
| | | **w1 (gate_proj)** | `[num_experts, hidden, inter]` | ColumnParallel | `[num_experts, hidden, inter/tp]` | None | None | **Expert参数**，门控投影。注意：w1 ≠ router |
| | | w3 (up_proj) | `[num_experts, hidden, inter]` | ColumnParallel | `[num_experts, hidden, inter/tp]` | None | None | Expert参数，上投影 |
| | | w2 (down_proj) | `[num_experts, inter, hidden]` | RowParallel | `[num_experts, inter/tp, hidden]` | None | All-Reduce | 下投影 |
| | | - | - | - | - | - | `tp_size > 1` | 在 expert 计算后执行 |
| | EP + TP | gate_proj (router) | `[hidden, num_experts]` | **Replicate** | `[hidden, num_experts]` (每个rank相同) | None | None | **不切分**，全局 top-k 必须一致 |
| | | w1/w3 | `[E, hidden, inter]` | **先 EP 分专家**<br>**再 TP 切参数** | `[E/ep, hidden, inter/tp]` | **All-to-All**<br>(token routing) | None | 每个 rank 只持有<br>`E/ep` 个完整专家 |
| | | w2 (down_proj) | `[E, inter, hidden]` | 先 EP 再 TP | `[E/ep, inter/tp, hidden]` | All-to-All | **All-Reduce** +<br>**All-to-All** | All-Reduce (TP)<br>All-to-All (EP routing back) |
| | DP + TP | w1/w3/w2 | 同 TP only | 同 TP only | 同 TP only | None | All-Reduce | DP 组间独立<br>不额外通信 |
| | **DP + EP (无 TP)** | gate_proj (router) | `[hidden, num_experts]` | **Replicate** | `[hidden, num_experts]` (每个rank相同) | None | None | **不切分**，所有 rank 执行相同 top-k |
| | | w1/w3 | `[E, hidden, inter]` | **仅 EP 分专家** | `[E/ep, hidden, inter]` | **All-to-All** | **All-to-All** | 每个 rank 持有<br>`E/ep` 个**完整**专家 |
| | | w2 (down_proj) | `[E, inter, hidden]` | 仅 EP | `[E/ep, inter, hidden]` | All-to-All | All-to-All | **无 All-Reduce**<br>(因为无 TP) |
| | DP + EP + TP | w1/w3/w2 | 同上 | EP + TP 组合 | `[E/ep, *, */tp]` | All-to-All (EP) | All-Reduce (TP) +<br>All-to-All (EP) | DP 组独立划分专家 |
| | **Sequence Parallel** | w1/w3/w2 | 同 TP | 同 TP | 同 TP | **ReduceScatter** | **All-Gather** | Sequence 维度切分<br>替代 All-Reduce |

## 三、具体实现代码分析

### 0. Router (Gate) Projection 特殊性

**问题**: 为什么 gate_proj 不能被 TP 切分？

**答案**: Gate projection 计算的是 router logits，必须基于**完整的 expert 集合**来决定 top-k 专家，否则无法得到全局最优的路由决策。

```python
# ❌ 错误做法：gate_proj 被 ColumnParallel 切分
gate_weight: [hidden_size, num_experts] 
    → ColumnParallel 切分
    → rank 0 持有: [hidden_size, num_experts/2]
    → rank 1 持有: [hidden_size, num_experts/2]

router_logits = x @ gate_weight  
# rank 0 得到: [batch, seq, num_experts/2] - 只有前一半 experts 的分数
# rank 1 得到: [batch, seq, num_experts/2] - 只有后一半 experts 的分数

top_k = torch.topk(router_logits, k=2)
# ❌ 问题：各 rank 基于不同的 experts 子集计算 top-k
#         rank 0 的 top-2 可能是 E0, E1（来自前半部分）
#         rank 1 的 top-2 可能是 E128, E129（来自后半部分）
#         不同 rank 的路由决策完全不一致！
```

**正确做法**: Gate projection 使用 **Replicate** 策略

```python
# ✓ 正确做法：gate_proj 完整复制到所有 TP ranks
gate_weight: [hidden_size, num_experts] (完整)
    → Replicate (不切分)
    → rank 0 持有: [hidden_size, num_experts]
    → rank 1 持有: [hidden_size, num_experts]

router_logits = x @ gate_weight  
# 所有 ranks 都得到: [batch, seq, num_experts] (相同的完整 logits)

top_k = torch.topk(router_logits, k=2)
# ✓ 所有 ranks 计算出相同的 top-k 决策
#   例如都选择 E5 和 E127
```

**为什么是 Replicate 而不是其他方式？**

| 方式 | 存储 | 通信 | 能否计算正确的 top-k | 说明 |
|------|------|------|-------------------|------|
| **Replicate** | 每个 rank 存储完整权重 | 无 | ✓ 是 | 简单，无通信开销 |
| **ColumnParallel + All-Reduce** | 权重切分 | 前置 All-Reduce | ❌ 否 | All-Reduce 是求和，不是聚合 logits |
| **ColumnParallel + All-Gather** | 权重切分 | 前置 All-Gather | ✓ 是 | 可行但多余，直接 Replicate 更优 |

**vLLM 实现**: 实际上在 Mixtral, Qwen2-MoE 等的实现中，gate projection 通常是：
```python
self.gate = nn.Linear(hidden_size, num_experts)  
# Linear 层默认不被 TP 切分
# 或显式设置为 Replicate
```

### 1. 标准 MoE 实现 (Mixtral, Qwen2-MoE 等)

```python
# 在 vllm/model_executor/models/mixtral.py, qwen2_moe.py, granitemoe.py 等
class XXXMoE(nn.Module):
    def __init__(...):
        # 专家权重: w1/w3 使用 MergedColumnParallelLinear (列并行)
        # w2 使用 RowParallelLinear (行并行)
        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,  # 启用 RowParallel 后的 All-Reduce
            ...
        )
    
    def forward(self, hidden_states):
        # Router 计算 (每个 rank 独立计算,无通信)
        router_logits, _ = self.gate(hidden_states)
        
        # Expert 计算 (包含权重切分逻辑)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, 
            router_logits=router_logits
        )
        
        # TP > 1 时,进行 All-Reduce 聚合各 rank 的部分和
        if self.tp_size > 1:
            # 这个 All-Reduce 对应 RowParallel w2 的输出聚合
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states
            )
        
        return final_hidden_states
```

### 2. FusedMoE 内部权重存储格式

```python
# vllm/model_executor/layers/fused_moe/layer.py
class FusedMoE(nn.Module):
    def __init__(...):
        # 权重张量实际形状 (已切分):
        # w1: [local_num_experts, hidden_size, intermediate_size_per_partition]
        # w2: [local_num_experts, intermediate_size_per_partition, hidden_size]
        # 
        # 其中:
        # - local_num_experts = E (无 EP) 或 E/ep_size (有 EP)
        # - intermediate_size_per_partition = intermediate_size / tp_size
        
        # 权重切分维度:
        # w1/w3 (ColumnParallel): 最后一维 (intermediate_size) 被切分
        # w2 (RowParallel): 中间维度 (intermediate_size) 被切分
        pass
```

### 3. Expert Parallelism 的 All-to-All 通信

```python
# 启用 EP 时的通信流程:
# 
# Forward Pass:
# 1. 输入: tokens 在各 rank 上
# 2. Router 决定每个 token 去哪些 experts
# 3. **All-to-All (1)**: 根据 expert 所属 rank,重新分配 tokens
#    - 每个 rank 收到需要由其本地 experts 处理的 tokens
# 4. 本地 expert 计算 (w1 -> activation -> w2)
# 5. w2 的 RowParallel 需要 **All-Reduce (TP 内部)**
# 6. **All-to-All (2)**: tokens 路由回原始位置
#    - 每个 rank 收回其原始 tokens 的 expert 输出
```

## 四、通信算子详解

### 1. All-Reduce
- **触发条件**: TP > 1 且使用 RowParallel
- **作用**: 聚合各 TP rank 的部分和 → 完整结果
- **通信组**: TP group (tensor_model_parallel_group)
- **数据量**: `batch_size * seq_len * hidden_size`
- **在 MoE 中的位置**: w2 (down_proj) 输出后

### 2. All-to-All
- **触发条件**: EP > 1
- **作用**: Token 路由 (forward) 和结果收集 (backward)
- **通信组**: EP group (expert_parallel_group)
- **数据量**: 取决于 token-expert 分配不均衡度
- **调用次数**: **一次 forward 推理需要 2 次 All-to-All 调用**
  1. **Dispatch (token routing)**: expert 计算前，按 expert 位置重组 tokens
  2. **Collect (token gathering)**: expert 计算后，按 token 位置收集 outputs
- **说明**: 每次 All-to-All 是独立的通信原语调用，不是一次调用完成双向通信

### 3. All-Gather
- **触发条件**: Sequence Parallel
- **作用**: 聚合分散在各 rank 上的序列片段
- **通信组**: TP group
- **在 MoE 中的位置**: 特殊场景,通常不在标准 MoE 流程中

## 五、DeepSeek V3.1 示例

### 配置
- 总专家数: 256
- Top-K: 8
- Hidden Size: 7168
- Intermediate Size (MoE): 10240
- TP = 8, EP = 16, DP = 1

### 权重分布
```
每个 EP rank 持有: 256 / 16 = 16 个专家
每个专家在 TP 内切分:
- w1/w3: [hidden=7168, inter=10240] → 每个 TP rank: [7168, 10240/8=1280]
- w2:    [inter=10240, hidden=7168] → 每个 TP rank: [10240/8=1280, 7168]

最终每个物理 rank (EP × TP) 存储:
- w1/w3: [16, 7168, 1280] 
- w2:    [16, 1280, 7168]
```

### 通信流程
```
1. Token Routing (All-to-All in EP group):
   - 输入: [batch, seq_len, 7168] 分布在 16 个 EP ranks
   - 输出: tokens 重分布到对应 expert 所在 rank
   
2. Expert Computation:
   - w1/w3 (ColumnParallel): 各 TP rank 独立计算部分中间结果
   - Activation: 无通信
   - w2 (RowParallel): 计算部分输出
   
3. TP All-Reduce (in TP group):
   - 聚合 8 个 TP ranks 的 w2 输出 → 完整 hidden_size
   
4. Token Collect (All-to-All in EP group):
   - 输出: tokens 返回原始序列位置
```

## 六、总结

| 维度 | 策略 | 权重切分 | 主要通信算子 | 通信频率 |
|------|------|---------|-------------|---------|
| **TP** | Tensor Parallel | w1/w3: 列并行<br>w2: 行并行 | All-Reduce (TP 组内) | 每次 forward |
| **EP** | Expert Parallel | 专家维度切分<br>(每 rank 持有 E/ep 个完整专家) | All-to-All (EP 组内)<br>**× 2** (dispatch + collect) | 每次 forward |
| **DP** | Data Parallel | 完整权重复制 | 无 (forward)<br>All-Reduce (backward, 梯度聚合) | backward only |
| **SP** | Sequence Parallel | 同 TP (权重切分相同) | ReduceScatter + All-Gather<br>(替代 All-Reduce) | 每次 forward |
| **DP + EP (无 TP)** | 混合 | **仅 EP 分专家**<br>权重不切分 | **All-to-All (EP) × 2**<br>(dispatch + collect) | 每次 forward |
| **TP + EP** | 混合 | EP 分专家 + TP 切参数 | All-to-All (EP) **× 2** +<br>All-Reduce (TP) | 每次 forward |

### 关键点
1. **ColumnParallel 无后置通信**: w1/w3 输出是部分维度,直接进入 activation
2. **RowParallel 需要 All-Reduce**: w2 输出需要聚合才是完整结果
3. **EP 引入 All-to-All**: Token-expert 路由导致跨 rank 通信
4. **DP 独立并行**: 各 DP group 独立计算,forward 无通信
5. **DP+EP 无 TP 场景**: 专家数量多但单个专家不大时,每个 rank 持有完整专家参数,无需 All-Reduce
6. **Sequence Parallel**: 将 sequence 维度切分,用 ReduceScatter + All-Gather 替代 All-Reduce,降低通信量

## 七、DP+EP (无 TP) 典型场景

### 适用条件
- **专家数量多**: 例如 256 个专家
- **单专家参数量适中**: 单卡显存足够放下完整专家权重
- **总专家数过多**: 所有专家加起来单卡放不下
- **无需模型并行**: 模型本身不大,不需要 Tensor Parallel

### 配置示例
假设有 256 个专家,每个专家参数量为 2GB:
- **TP = 1**: 不使用 Tensor Parallel,权重不切分
- **EP = 8**: 将 256 个专家分散到 8 个 rank
- **DP = 2**: 2 个 engine 实例,提高吞吐量

### 权重分布
```
每个 DP group 独立持有:
  - EP rank 0: 专家 0-31 (完整权重)
  - EP rank 1: 专家 32-63 (完整权重)
  - ...
  - EP rank 7: 专家 224-255 (完整权重)

每个专家权重:
  w1/w3: [hidden_size, intermediate_size] (完整,未切分)
  w2:    [intermediate_size, hidden_size] (完整,未切分)
```

### 通信流程
```
1. Token Routing (All-to-All in EP group):
   - 每个 DP group 内部独立进行
   - 根据 router 决策,将 tokens 发送到对应 expert 所在的 EP rank
   
2. Expert Computation (完全本地):
   - w1/w3: 完整权重,输出完整中间结果
   - Activation: 本地计算
   - w2: 完整权重,输出完整 hidden_size
   - **无需 All-Reduce** (因为 TP=1,权重未切分)
   
3. Token Collect (All-to-All in EP group):
   - 将 expert 输出返回到原始 token 位置
   - 每个 DP group 内部独立进行
```

### 优势
- **简化通信**: 相比 DP+EP+TP,省去了 All-Reduce 通信
- **降低延迟**: All-Reduce 是同步阻塞的,All-to-All 可以更灵活
- **内存效率**: 当专家数量是瓶颈时,EP 解决存储问题,TP=1 避免额外通信开销

### vLLM 代码支持
根据 `vllm/model_executor/layers/fused_moe/config.py` L982-988:
```python
# TP = 1, DP = 2, EP = True 的配置:
- device 0: TP = {1, 0} DP = {2, 0} EP = {2, 0}
- device 1: TP = {1, 0} DP = {2, 1} EP = {2, 1}
# 注释: 有 2 个 engine 实例,专家在 2 个设备间分割
```

## 八、MoE All-to-All 通信可视化解释

### 关键问题：一次 forward 推理执行几次 All-to-All？

**答案: 2 次独立的 All-to-All 调用**

虽然 All-to-All 通信原语本身是双向的（每个 rank 同时向所有其他 ranks 发送和接收数据），但在 MoE 的 Expert Parallelism 中，一次完整的 forward 推理需要**两次独立的 All-to-All 调用**：

1. **第 1 次 All-to-All (Dispatch)**: Token → Expert 重组
   - 目的: 将 tokens 发送到持有对应 expert 的 GPU
   - 重组维度: 按 Expert 位置（Expert-centric）

2. **第 2 次 All-to-All (Collect)**: Expert output → Token 重组
   - 目的: 将 expert 输出收集回原始 token 位置
   - 重组维度: 按 Token 位置（Token-centric）

**为什么不是 1 次？** 因为这是两个不同目的的数据重组：
- Dispatch: `Group by Expert location`
- Collect: `Group by Token location`
- 需要两次独立的通信调用来完成这两种不同的重组

### 为什么 MoE 需要 All-to-All？

当使用 Expert Parallelism (EP) 时，不同的专家存储在不同的 GPU 上。但是 Router 决定的 token-expert 分配是**动态且不均匀**的，这导致：
- Token A 可能需要 Expert 1（在 GPU 0）和 Expert 5（在 GPU 1）
- Token B 可能需要 Expert 2（在 GPU 0）和 Expert 6（在 GPU 1）

**问题**: 每个 GPU 的输入 tokens 需要被发送到持有相应 experts 的其他 GPUs。

### 场景设置
- **4 个 Tokens**: T0, T1, T2, T3
- **4 个 Experts**: E0, E1, E2, E3  
- **2 个 GPUs with EP=2**:
  - GPU 0: 持有 E0, E1
  - GPU 1: 持有 E2, E3
- **Top-K = 2**: 每个 token 选择 2 个专家

### Router 决策结果
```
Token T0 → Expert E0 (GPU 0), Expert E2 (GPU 1)
Token T1 → Expert E1 (GPU 0), Expert E3 (GPU 1)
Token T2 → Expert E0 (GPU 0), Expert E3 (GPU 1)
Token T3 → Expert E2 (GPU 1), Expert E3 (GPU 1)
```

### 初始状态（Router 之后）
```
┌─────────── GPU 0 ─────────────┐    ┌─────────── GPU 1 ─────────────┐
│ Tokens: [T0, T1, T2, T3]      │    │ Tokens: [T0, T1, T2, T3]      │
│                                │    │                                │
│ Experts: [E0, E1]              │    │ Experts: [E2, E3]              │
│                                │    │                                │
│ Router 决策:                    │    │ Router 决策:                    │
│   T0 → E0 (本地), E2 (远程)    │    │   T0 → E0 (远程), E2 (本地)    │
│   T1 → E1 (本地), E3 (远程)    │    │   T1 → E1 (远程), E3 (本地)    │
│   T2 → E0 (本地), E3 (远程)    │    │   T2 → E0 (远程), E3 (本地)    │
│   T3 → E2 (远程), E3 (远程)    │    │   T3 → E2 (本地), E3 (本地)    │
└────────────────────────────────┘    └────────────────────────────────┘
```

### All-to-All 通信第一步: Token Dispatch (发送到 Expert 所在 GPU)

需要重组数据，让每个 GPU 只处理**本地 Expert** 对应的 tokens：

```
【All-to-All 通信】
GPU 0 发送:                          GPU 1 发送:
  → GPU 0: [T0→E0, T1→E1, T2→E0]     → GPU 0: []  
  → GPU 1: [T0→E2, T1→E3, T2→E3, T3→E2, T3→E3]    → GPU 1: [T0→E2, T1→E3, T2→E3, T3→E2, T3→E3]

通信后的状态:
┌─────────── GPU 0 ─────────────┐    ┌─────────── GPU 1 ─────────────┐
│ 本地处理队列:                   │    │ 本地处理队列:                   │
│   E0: [T0, T2]                 │    │   E2: [T0, T3]                 │
│   E1: [T1]                     │    │   E3: [T1, T2, T3]             │
│                                │    │                                │
│ ✓ 现在可以本地执行 Expert 计算   │    │ ✓ 现在可以本地执行 Expert 计算   │
└────────────────────────────────┘    └────────────────────────────────┘
```

**可视化流程图**:
```
Before All-to-All (Token Dispatch):
┌─────────────────┐         ┌─────────────────┐
│     GPU 0       │         │     GPU 1       │
│                 │         │                 │
│ T0 needs E0,E2 ─┼────X────┤                 │ E0 在 GPU 0
│ T1 needs E1,E3 ─┼────X────┤                 │ E1 在 GPU 0
│ T2 needs E0,E3 ─┼────X────┤                 │ E2 在 GPU 1
│ T3 needs E2,E3 ─┼────X────┤                 │ E3 在 GPU 1
│                 │         │                 │
│  E0 ● E1 ●      │         │      ● E2 ● E3  │
└─────────────────┘         └─────────────────┘
        ❌ Token 和 Expert 不在同一 GPU，无法直接计算

After All-to-All (Token Dispatch):
┌─────────────────┐         ┌─────────────────┐
│     GPU 0       │         │     GPU 1       │
│                 │         │                 │
│ E0 ← [T0, T2]  ●│         │● E2 ← [T0, T3]  │
│ E1 ← [T1]      ●│         │● E3 ← [T1,T2,T3]│
│                 │         │                 │
│  E0 ● E1 ●      │         │      ● E2 ● E3  │
└─────────────────┘         └─────────────────┘
        ✓ 每个 GPU 只处理本地 Expert，可以并行计算
```

### Expert 计算（本地执行，无通信）

```
┌─────────── GPU 0 ─────────────┐    ┌─────────── GPU 1 ─────────────┐
│ Expert E0 处理 [T0, T2]        │    │ Expert E2 处理 [T0, T3]        │
│   output: [O(T0,E0), O(T2,E0)] │    │   output: [O(T0,E2), O(T3,E2)] │
│                                │    │                                │
│ Expert E1 处理 [T1]            │    │ Expert E3 处理 [T1, T2, T3]    │
│   output: [O(T1,E1)]           │    │   output: [O(T1,E3), ...]      │
└────────────────────────────────┘    └────────────────────────────────┘
```

### All-to-All 通信第二步: Token Collect (返回原始 Token 位置)

Expert 计算完成后，需要将结果**送回原始 token 所在的位置**进行聚合：

```
【All-to-All 通信】
GPU 0 发送:                          GPU 1 发送:
  → GPU 0: [O(T0,E0), O(T1,E1), O(T2,E0)]    → GPU 0: []
  → GPU 1: []                        → GPU 1: [O(T0,E2), O(T1,E3), O(T2,E3), O(T3,E2), O(T3,E3)]

通信后的状态:
┌─────────── GPU 0 ─────────────┐    ┌─────────── GPU 1 ─────────────┐
│ Token T0: [O(T0,E0), O(T0,E2)] │    │ (复制一份到 GPU 0)             │
│ Token T1: [O(T1,E1), O(T1,E3)] │    │                                │
│ Token T2: [O(T2,E0), O(T2,E3)] │    │                                │
│ Token T3: [O(T3,E2), O(T3,E3)] │    │                                │
│                                │    │                                │
│ ✓ 每个 token 收集到其 top-k     │    │                                │
│   experts 的所有输出             │    │                                │
└────────────────────────────────┘    └────────────────────────────────┘
```

**可视化流程图**:
```
After Expert Computation:
┌─────────────────┐         ┌─────────────────┐
│     GPU 0       │         │     GPU 1       │
│                 │         │                 │
│ E0 → [O₀₀, O₂₀]│         │ E2 → [O₀₂, O₃₂]│ 
│ E1 → [O₁₁]     │         │ E3 → [O₁₃,O₂₃,O₃₃]
│                 │         │                 │
└─────────────────┘         └─────────────────┘
        ❌ 每个 Token 的输出分散在不同 GPU

After All-to-All (Token Collect):
┌─────────────────┐         ┌─────────────────┐
│     GPU 0       │         │     GPU 1       │
│                 │         │                 │
│ T0: [O₀₀, O₀₂] │◄────────┤                 │
│ T1: [O₁₁, O₁₃] │◄────────┤                 │
│ T2: [O₂₀, O₂₃] │◄────────┤                 │
│ T3: [O₃₂, O₃₃] │◄────────┤                 │
│                 │         │                 │
└─────────────────┘         └─────────────────┘
        ✓ 每个 Token 收集到所有 Expert 输出，可以聚合
```

### 最终聚合

```
GPU 0 (或任一 GPU) 执行 Token-level 聚合:
  T0_final = weight(E0) * O(T0,E0) + weight(E2) * O(T0,E2)
  T1_final = weight(E1) * O(T1,E1) + weight(E3) * O(T1,E3)
  T2_final = weight(E0) * O(T2,E0) + weight(E3) * O(T2,E3)
  T3_final = weight(E2) * O(T3,E2) + weight(E3) * O(T3,E3)
```

### 为什么必须用 All-to-All？

1. **动态路由**: Router 的 top-k 选择是动态的，每个 token 可能选择任意专家
2. **负载不均**: 不同 GPU 收到的 token 数量不同（T3 只去 GPU 1）
3. **双向通信，2 次调用**: 
   - **Dispatch (第 1 次 All-to-All)**: 根据 expert 位置分发 tokens
   - **Collect (第 2 次 All-to-All)**: 根据 token 位置收集 outputs
   - **重要**: 这是 2 次独立的 All-to-All 算子调用，不是 1 次
4. **All-to-All 特性**: 每个 rank 向每个其他 rank 发送**不同大小**的数据，正好匹配 MoE 的不均匀路由

### 与其他通信算子的对比

| 通信算子 | 数据流向 | 数据特征 | MoE 是否适用 |
|---------|---------|---------|-------------|
| **All-Reduce** | 所有→所有 | 每个 rank 发送**相同形状**数据 | ❌ MoE 路由不均匀 |
| **All-Gather** | 所有→所有 | 只收集,不聚合求和 | ❌ 需要重组数据 |
| **Broadcast** | 一→所有 | 单向广播 | ❌ 需要双向通信 |
| **All-to-All** | 所有⇄所有 | 每个 rank 可发送**不同大小**数据到每个其他 rank | ✅ 完美匹配 MoE |

### 数学表达

对于 EP 组中的 rank $i$ 和 rank $j$:

**Dispatch (第一次 All-to-All)**:
```
Send[i→j] = {tokens from rank i that need experts on rank j}
```

**Collect (第二次 All-to-All)**:
```
Send[i→j] = {expert outputs from rank i that belong to tokens on rank j}
```

### 核心理解

All-to-All 本质上是一个**数据重组**操作：
- **Dispatch (第 1 次 All-to-All)**: 按 Expert 位置重组（Expert-centric）
- **Collect (第 2 次 All-to-All)**: 按 Token 位置重组（Token-centric）

**重要澄清**: 一次 forward 推理需要 **2 次 All-to-All 调用**，不是 1 次。虽然 All-to-All 本身是双向通信原语（每个 rank 同时发送和接收），但在 MoE 中需要两个独立的调用来完成：
1. Token → Expert 的重组（dispatch）
2. Expert output → Token 的重组（collect）

这就是为什么 MoE 的 Expert Parallelism 必须使用 All-to-All 通信！
