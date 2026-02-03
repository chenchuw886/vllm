# vLLM 并行组结构图详细解析

## 一、基础概念回顾

### 全局 Rank 的 5D 组织

vLLM 将 world_size 个 GPU rank 组织成 5 维结构：

```python
all_ranks = torch.arange(world_size).reshape(
    -1,                                    # Dimension 0: ExternalDP
    data_parallel_size,                    # Dimension 1: DP
    pipeline_model_parallel_size,          # Dimension 2: PP
    prefill_context_model_parallel_size,   # Dimension 3: PCP
    tensor_model_parallel_size,            # Dimension 4: TP
)
```

### 并行组创建的本质：张量操作

不同的并行组通过对该 5D 张量进行**转置 (transpose) 和重塑 (reshape)** 来创建：

```
TP:   view(-1, TP)                    # 直接提取最内层
DCP:  reshape(-1, DCP)                # 直接提取 DCP 维度
PCP:  transpose(3,4) → reshape(-1, PCP)  # 交换 PCP 和 TP 后提取
PP:   transpose(2,4) → reshape(-1, PP)   # 交换 PP 和 TP 后提取
DP:   transpose(1,4) → reshape(-1, DP)   # 交换 DP 和 TP 后提取
EP:   transpose(1,2) → reshape(-1, EP)   # 交换 DP 和 PP 后提取
```

---

## 二、完整的并行组结构图

### 示例配置 1：简单的 TP + DP

**参数**：
- ExternalDP = 1, DP = 2, PP = 1, PCP = 1, TP = 2
- World Size = 4 GPUs
- Ranks: 0, 1, 2, 3

**5D 数组结构**：
```
all_ranks[0, :, :, :, :] = [[0, 1], [2, 3]]  # ExternalDP=0
```

**并行组**：

```
┌─────────────────────────────────────────┐
│ TP 组（Tensor Parallel Groups）         │
├─────────────────────────────────────────┤
│ TP_0 = [0, 1]   (DP=0, PP=0)           │
│ TP_1 = [2, 3]   (DP=1, PP=0)           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ DP 组（Data Parallel Groups）           │
├─────────────────────────────────────────┤
│ DP_0 = [0, 2]   (TP=0, PP=0)           │
│ DP_1 = [1, 3]   (TP=1, PP=0)           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ PCP 组（Prefill Context Parallel）      │
├─────────────────────────────────────────┤
│ PCP_0 = [0]                             │
│ PCP_1 = [1]                             │
│ PCP_2 = [2]                             │
│ PCP_3 = [3]                             │
│ (PCP=1 时，每个 rank 单独成组)          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ DCP 组（Decode Context Parallel）       │
├─────────────────────────────────────────┤
│ DCP_0 = [0]                             │
│ DCP_1 = [1]                             │
│ DCP_2 = [2]                             │
│ DCP_3 = [3]                             │
│ (DCP=1 时，每个 rank 单独成组)          │
└─────────────────────────────────────────┘
```

**通信拓扑图**：
```
Rank 0 ←─TP─→ Rank 1
  ↕DP            ↕DP
Rank 2 ←─TP─→ Rank 3

TP: 水平线
DP: 竖直线
PCP/DCP: 单独（没有组内通信）
```

---

### 示例配置 2：TP + PCP + DP

**参数**：
- ExternalDP = 1, DP = 2, PP = 1, PCP = 2, TP = 2
- World Size = 8 GPUs
- Ranks: 0-7

**5D 数组结构**：
```
all_ranks[0, :, :, :, :] = [
    [[0, 1], [2, 3]],    # DP=0, PCP=0, TP=0,1; PCP=1, TP=0,1
    [[4, 5], [6, 7]],    # DP=1, PCP=0, TP=0,1; PCP=1, TP=0,1
]

展开：
Rank 0: DP=0, PCP=0, TP=0
Rank 1: DP=0, PCP=0, TP=1
Rank 2: DP=0, PCP=1, TP=0
Rank 3: DP=0, PCP=1, TP=1
Rank 4: DP=1, PCP=0, TP=0
Rank 5: DP=1, PCP=0, TP=1
Rank 6: DP=1, PCP=1, TP=0
Rank 7: DP=1, PCP=1, TP=1
```

**并行组**：

#### TP 组
```
TP_0 = [0, 1]     (DP=0, PCP=0)
TP_1 = [2, 3]     (DP=0, PCP=1)
TP_2 = [4, 5]     (DP=1, PCP=0)
TP_3 = [6, 7]     (DP=1, PCP=1)

总数：4 个 TP 组
```

#### PCP 组
```
PCP_0 = [0, 2]    (DP=0, TP=0)    ← 相同 DP 和 TP，不同 PCP
PCP_1 = [1, 3]    (DP=0, TP=1)
PCP_2 = [4, 6]    (DP=1, TP=0)
PCP_3 = [5, 7]    (DP=1, TP=1)

总数：4 个 PCP 组
```

#### DP 组
```
DP_0 = [0, 4]     (PCP=0, TP=0)   ← 相同 PCP 和 TP，不同 DP
DP_1 = [2, 6]     (PCP=1, TP=0)
DP_2 = [1, 5]     (PCP=0, TP=1)
DP_3 = [3, 7]     (PCP=1, TP=1)

总数：4 个 DP 组
```

**通信拓扑图**：
```
Layer 1 (DP=0):              Layer 2 (DP=1):
  [0←→1]  [2←→3]              [4←→5]  [6←→7]
    ↓ ↑      ↓ ↑                 ↓ ↑      ↓ ↑
    PCP      PCP                PCP      PCP
    ↓ ↑      ↓ ↑                 ↓ ↑      ↓ ↑

横向（─）: TP 通信
纵向（↕）: PCP 通信
左右（↔）: DP 通信

完整图：
┌─────────────────────────────────────────┐
│ DP=0, PCP=0     DP=0, PCP=1             │
│  0 ←─TP─→ 1     2 ←─TP─→ 3             │
│  ↕ DP    ↕ DP    ↕ DP    ↕ DP           │
│  ↕ PCP   ↕ PCP   ↕ PCP   ↕ PCP          │
│  4 ←─TP─→ 5     6 ←─TP─→ 7             │
│ DP=1, PCP=0     DP=1, PCP=1             │
└─────────────────────────────────────────┘
```

---

### 示例配置 3：TP + PCP + PP + DP（完整配置）

**参数**：
- ExternalDP = 1, DP = 2, PP = 2, PCP = 2, TP = 2
- World Size = 32 GPUs
- Ranks: 0-31

**5D 数组结构**：
```
all_ranks[0, :, :, :, :] = 
[
    [  # DP = 0
        [  # PP = 0
            [[0, 1], [2, 3]],    # PCP=0, PCP=1
        ],
        [  # PP = 1
            [[8, 9], [10, 11]],  # PCP=0, PCP=1
        ],
    ],
    [  # DP = 1
        [  # PP = 0
            [[16, 17], [18, 19]],  # PCP=0, PCP=1
        ],
        [  # PP = 1
            [[24, 25], [26, 27]],  # PCP=0, PCP=1
        ],
    ],
    ...  # 继续
]
```

**并行组数量**：

| 组类型 | 数量 | 大小 |
|------|------|------|
| TP | 8 | 2 |
| PCP | 8 | 2 |
| PP | 8 | 2 |
| DP | 8 | 2 |
| DCP | 32 | 1（无分组） |
| EP | 2 | 16 |

**关键通信组详细列表**：

#### TP 组（8 个）
```
TP_0  = [0, 1]     TP_1  = [2, 3]
TP_2  = [8, 9]     TP_3  = [10, 11]
TP_4  = [16, 17]   TP_5  = [18, 19]
TP_6  = [24, 25]   TP_7  = [26, 27]
```

#### PP 组（8 个）
```
PP_0 = [0, 8]      (DP=0, PCP=0, TP=0)
PP_1 = [1, 9]      (DP=0, PCP=0, TP=1)
PP_2 = [2, 10]     (DP=0, PCP=1, TP=0)
PP_3 = [3, 11]     (DP=0, PCP=1, TP=1)
PP_4 = [16, 24]    (DP=1, PCP=0, TP=0)
PP_5 = [17, 25]    (DP=1, PCP=0, TP=1)
PP_6 = [18, 26]    (DP=1, PCP=1, TP=0)
PP_7 = [19, 27]    (DP=1, PCP=1, TP=1)
```

#### DP 组（8 个）
```
DP_0 = [0, 16]     (PP=0, PCP=0, TP=0)
DP_1 = [2, 18]     (PP=0, PCP=1, TP=0)
DP_2 = [8, 24]     (PP=1, PCP=0, TP=0)
DP_3 = [10, 26]    (PP=1, PCP=1, TP=0)
DP_4 = [1, 17]     (PP=0, PCP=0, TP=1)
DP_5 = [3, 19]     (PP=0, PCP=1, TP=1)
DP_6 = [9, 25]     (PP=1, PCP=0, TP=1)
DP_7 = [11, 27]    (PP=1, PCP=1, TP=1)
```

#### PCP 组（8 个）
```
PCP_0 = [0, 2]     (DP=0, PP=0, TP=0)
PCP_1 = [1, 3]     (DP=0, PP=0, TP=1)
PCP_2 = [8, 10]    (DP=0, PP=1, TP=0)
PCP_3 = [9, 11]    (DP=0, PP=1, TP=1)
PCP_4 = [16, 18]   (DP=1, PP=0, TP=0)
PCP_5 = [17, 19]   (DP=1, PP=0, TP=1)
PCP_6 = [24, 26]   (DP=1, PP=1, TP=0)
PCP_7 = [25, 27]   (DP=1, PP=1, TP=1)
```

#### EP 组（2 个）
```
EP_0 = [0, 1, 2, 3, 8, 9, 10, 11]      (DP=0, PP=0)
EP_1 = [16, 17, 18, 19, 24, 25, 26, 27] (DP=1, PP=0)

或者 PP=1:
EP_2 = [8, 9, 10, 11, 24, 25, 26, 27]  (DP=0, PP=1)？

不对，EP 应该是按 PP 划分的...
```

**立体通信拓扑图**：

```
                  PP=0                              PP=1
              ┌─────────────┐                   ┌─────────────┐
              │             │                   │             │
          PCP=0         PCP=1               PCP=0         PCP=1
        ┌───┴───┐     ┌───┴───┐           ┌───┴───┐     ┌───┴───┐
       TP₀     TP₁   TP₀     TP₁         TP₀     TP₁   TP₀     TP₁
    DP=0┌────────────────────────┐    DP=0┌────────────────────────┐
       │ 0─1  2─3  8─9  10─11    │       │ ...                     │
       └────────────────────────┘       └────────────────────────┘
           ↓DP  ↓DP   ↓DP  ↓DP               ↓DP  ↓DP   ↓DP  ↓DP
    DP=1┌────────────────────────┐    DP=1┌────────────────────────┐
       │16─17 18─19 24─25 26─27   │       │ ...                     │
       └────────────────────────┘       └────────────────────────┘

通信类型：
─  : TP 通信 (Weight Sharding)
↓  : DP 通信 (Gradient All-Reduce)
│  : PP 通信 (Activation Forward/Backward)
∼  : PCP 通信 (Attention KV Gather/Scatter)
```

---

## 三、重要的 Rank 关系

### 从 Rank 推导并行 ID

对于任意 rank，可以反推其并行 ID：

```python
def get_parallel_ids(rank, world_size, dp_size, pp_size, pcp_size, tp_size):
    # 恢复 5D 索引
    all_ranks = torch.arange(world_size).reshape(
        -1, dp_size, pp_size, pcp_size, tp_size
    )
    
    # 找到该 rank 的位置
    pos = (all_ranks == rank).nonzero(as_tuple=True)
    
    _, dp_id, pp_id, pcp_id, tp_id = pos
    return dp_id, pp_id, pcp_id, tp_id

# 示例：假设 world_size=32, config=(2,2,2,2)
rank_to_ids = {
    0: (0, 0, 0, 0),   # DP=0, PP=0, PCP=0, TP=0
    1: (0, 0, 0, 1),   # DP=0, PP=0, PCP=0, TP=1
    2: (0, 0, 1, 0),   # DP=0, PP=0, PCP=1, TP=0
    3: (0, 0, 1, 1),   # DP=0, PP=0, PCP=1, TP=1
    8: (0, 1, 0, 0),   # DP=0, PP=1, PCP=0, TP=0
    ...
}
```

### 并行组成员关系

```python
def get_tp_group_members(rank, world_size, dp_size, pp_size, pcp_size, tp_size):
    """获取 rank 所属 TP 组的所有成员"""
    dp_id, pp_id, pcp_id, tp_id = get_parallel_ids(rank, ...)
    # TP 组由相同 dp, pp, pcp 但不同 tp 组成
    tp_group = []
    for tp in range(tp_size):
        member = dp_id * (pp_size * pcp_size * tp_size) \
                + pp_id * (pcp_size * tp_size) \
                + pcp_id * tp_size \
                + tp
        tp_group.append(member)
    return tp_group
```

---

## 四、通信模式总结

### 各并行维度的通信特征

| 维度 | 通信类型 | 频率 | 数据量 | 用途 |
|------|--------|------|--------|------|
| **TP** | All-Reduce | 每层 | hidden_size | 权重聚合 |
| **PCP** | All-Gather + ReduceScatter | 每层 Prefill | hidden_size | KV 缓存分散 |
| **DCP** | All-Gather + ReduceScatter | 每层 Decode | hidden_size | KV 缓存分散 |
| **PP** | Send/Recv | 阶段间 | batch_size×seq_len×hidden_size | 激活传递 |
| **DP** | All-Reduce | backward | 梯度 | 梯度聚合 |
| **EP** | All-to-All | MoE 层 | tokens×inter_size | 专家路由 |

### 预填充（Prefill）的通信流：
```
Input → Attention(PCP All-Gather + ReduceScatter) 
      → MLP(TP All-Reduce) 
      → LayerNorm 
      → Output

通信组：PCP → TP
```

### 解码（Decode）的通信流：
```
Token → Attention(DCP All-Gather + ReduceScatter) 
      → MLP(TP All-Reduce) 
      → LayerNorm 
      → Next Token

通信组：DCP → TP
```

---

## 五、实际配置建议

### 配置 1：单机多卡（8 GPU）
```
DP=1, PP=1, PCP=1, TP=8, DCP=1
（所有 8 卡做 TP）
```

### 配置 2：2 机 16 卡
```
DP=2, PP=1, PCP=2, TP=2, DCP=1
或
DP=1, PP=2, PCP=1, TP=2, DCP=2
```

### 配置 3：8 机 64 卡（大规模）
```
DP=4, PP=4, PCP=1, TP=2, DCP=2
（4 个数据并行组，4 个流水线阶段，每个 stage 内 2 卡 TP）
```

---

## 六、重要提醒

1. **DCP 复用 TP Group GPU**
   - `tp_size % dcp_size == 0` 必须满足
   - DCP 只是将 TP group 的 GPU 进行了逻辑上的重新分组

2. **PCP 与 PP 的关系**
   - PCP 是在预填充时的优化
   - PP 是流水线阶段的划分
   - 二者独立

3. **EP 的特殊性**
   - EP = DP × PCP × TP（同一 PP 阶段）
   - 用于 MoE 模型的专家 All-to-All 通信

4. **Rank 组织优先级**
   - 最外层（ExternalDP）: 通常不用
   - DP → PP → PCP → TP（从外到内）
   - TP 在最内层（便于相邻 rank 通信）
