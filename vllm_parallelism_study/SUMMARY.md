# vLLM 并行化架构学习总结

## ✅ 任务完成状态

### 1️⃣ 文档汇总 ✓
所有学习文档已整理到 `/vllm_parallelism_study/` 目录：

```
vllm_parallelism_study/
├── README.md                                      (主目录)
├── SUMMARY.md                                     (学习总结)
├── moe_sharding_communication_table.md           (MoE 权重切分)
├── sequence_parallelism_explained.md             (SP 原理)
├── sequence_parallelism_corrections.md           (SP 纠错)
├── sp_full_analysis.md                          (SP 完整分析)
├── 01_pcp_dcp_detailed_analysis.md              (PCP/DCP 详解) ✨
├── 02_pcp_dcp_parallel_group_structure.md       (并行组结构图) ✨
└── 03_communication_and_topology_optimization.md (通信量与拓扑优化) ✨⚡
```

**总大小**: ~160 KB，约 9 个完整 Markdown 文档

### 2️⃣ PCP/DCP 理解 ✓

#### Context Parallel (CP) 的核心概念
```
目的: 面向长上下文推理的两类瓶颈
  - Prefill: 降低 TTFT，将长序列预填充分摊到多卡
  - Decode: 降低 KV cache 内存占用，提高并发
机制:
  - PCP: 将请求 token 序列按卡数分块并行计算 Q/K/V
  - DCP: 在 TP 已按 head 切分后，再沿 T 维度分片 KV cache
```

#### PCP (Prefill Context Parallel)
- **应用**: 推理预填充阶段
- **分散方式**: 将请求 token 序列按 pcp_size 分块，每卡计算其 chunk 的 Q/K/V
- **两种策略**:
  1) Partial Q + Full K/V：收集全量 K/V 后计算各自的 Q chunk
  2) Partial Q + Partial K/V：使用 ring-attention 等分块发送/接收 K/V
- **通信**: 取决于策略（K/V gather 或 ring send/recv）
- **代码位置**: `vllm/v1/attention/backends/flashinfer.py` L242-257

#### DCP (Decode Context Parallel)
- **应用**: 推理解码阶段
- **分散方式**: 在 TP 已按 KV head 切分后，再沿 T 维度分片 KV cache
- **范围约束**: `dcp_size ∈ [1, tp_size / H_kv]`（H_kv 为 kv-head 数）
- **通信**: 访问跨分片 KV 的通信随 dcp_size 增大而增加
- **代码位置**: `vllm/v1/attention/backends/flashinfer.py` L1476-1497
- **关键特性**: **复用 TP 组的 GPU**，不增加 world_size

### 3️⃣ 并行组结构 ✓

#### 5D Rank 组织（核心）
```python
all_ranks = torch.arange(world_size).reshape(
    -1,                           # ExternalDP (外部，通常=1)
    data_parallel_size,           # DP
    pipeline_model_parallel_size, # PP
    prefill_context_model_parallel_size,  # PCP
    tensor_model_parallel_size,   # TP
)
```

#### 各并行组的创建逻辑
| 并行维度 | 创建方式 | 含义 |
|---------|--------|------|
| **TP** | `view(-1, tp_size)` | 权重切分 |
| **DCP** | `reshape(-1, dcp_size)` | 解码 KV 缓存分散 |
| **PCP** | `transpose(3,4) + reshape()` | 预填充 KV 缓存分散 |
| **PP** | `transpose(2,4) + reshape()` | 流水线阶段 |
| **DP** | `transpose(1,4) + reshape()` | 数据并行 |
| **EP** | `transpose(1,2) + reshape()` | 专家并行 (MoE) |

#### 具体例子 (DP=2, PP=2, PCP=2, TP=2)
```
TP 组 (8 个):
  TP_0=[0,1]   TP_1=[2,3]
  TP_2=[8,9]   TP_3=[10,11]
  TP_4=[16,17] TP_5=[18,19]
  TP_6=[24,25] TP_7=[26,27]

PCP 组 (8 个):
  PCP_0=[0,2]   (DP=0, TP=0，不同 PCP)
  PCP_1=[1,3]   (DP=0, TP=1，不同 PCP)
  ...

DP 组 (8 个):
  DP_0=[0,16]   (PCP=0, TP=0，不同 DP)
  DP_1=[2,18]   (PCP=1, TP=0，不同 DP)
  ...

EP 组 (2 个):
  EP_0=[0,1,2,3,8,9,10,11]   (DP=0, PP=0)
  EP_1=[16,17,18,19,24,25,26,27] (DP=1, PP=0)
```

---

## 🎯 关键发现总结

### 发现 1: MoE 的 2 次 All-to-All
```
误解: "一次 forward 需要 1 次 All-to-All"
真相: 需要 2 次**独立**的 All-to-All 调用
  - 第 1 次 (Dispatch): Token → Expert 重组
  - 第 2 次 (Collect): Expert Output → Token 重组
理由: 这是两个不同的数据重组目的
```

### 发现 2: Sequence Parallel 的限制
```
✅ 支持: RMSNorm/LayerNorm (element-wise)
  - vLLM 实现: 4 个 Pattern（全是 RMSNorm）

❌ 不支持: FFN
  - 原因: Pattern 过于复杂，跨多层
  - 收益: 有限

❌ 不支持: QKV/Attention  
  - 原因: Attention 必须全局，序列切分无益

⚠️ MoE: 已实现手动 SP
  - 方式: 显式 sequence_parallel_chunk()
  - 不是编译 Pass
```

### 发现 3: Context Parallel 的新机制
```
不同于 Sequence Parallel:
- SP: 沿序列维切分 (减少激活内存)
- CP: prefill 做 token 分块，decode 做 KV cache 的 T 维分片

不是单纯通信优化:
- SP: AllReduce → ReduceScatter+AllGather (优化通信)
- CP: 以存储与吞吐为目标，同时引入必要通信

通信形式随策略变化:
- PCP: K/V gather 或 ring-attention send/recv
- DCP: 跨分片 KV 访问的通信随 dcp_size 增加
```

### 发现 4: DCP 复用 TP 组
```
DCP 的特殊性:
- 不增加 world_size
- 复用 TP 组的 GPU
- 将 TP 组内的 GPU 进行逻辑分组

约束:
- tp_size >= dcp_size
- 每个 TP 组可分成多个 DCP 组
```

---

## 📊 完整架构对比表

### 各并行策略的通信模式

| 策略 | 通信类型 | 频率 | 数据量 | 开销 | 内存减少 |
|------|--------|------|--------|------|--------|
| **TP** | All-Reduce | 每层 | hidden_size | 中 | 0 (复制) |
| **DP** | All-Reduce | backward | 梯度 | 低 | 1/dp |
| **PP** | Send/Recv | 阶段间 | activation | 中 | 1/pp |
| **EP** | All-to-All | MoE 层 | tokens×inter | 高 | 1/ep |
| **SP** | ReduceScatter + AllGather | 每层 | hidden_size | 中 | ~1/tp |
| **PCP** | K/V gather 或 ring send/recv | Prefill | 序列 chunk 的 K/V | 中 | 降低 TTFT |
| **DCP** | 跨分片 KV 访问通信 | Decode | KV cache (按 T 分片) | 中 | 降低 KV 重复 |

### 推理流程中的通信

```
Prefill (预填充) 时:
├─ PCP: 按 token chunk 并行计算 Q/K/V
├─ Attention: 视策略进行 K/V gather 或 ring-attention
├─ MLP: TP All-Reduce (w2 输出)
└─ KV Cache: 依据策略保存（全量或分片）

Decode (解码) 时:
├─ DCP: KV cache 按 T 维分片（降低重复）
├─ Attention: 访问跨分片 KV（通信随 dcp_size 增加）
├─ MLP: TP All-Reduce (w2 输出)
└─ 下一 Token (循环)

MoE 层:
├─ All-to-All (dispatch): 按 expert 位置分发 tokens
├─ Expert Compute: 本地计算
├─ All-Reduce: TP 内聚合 (w2 输出)
└─ All-to-All (collect): 按 token 位置收集 outputs
```

---

## 🔍 vLLM 源代码关键位置

### 1. 配置和初始化
- `vllm/config/parallel.py` (L105-270)
  - `prefill_context_parallel_size`
  - `decode_context_parallel_size`
  - `cp_kv_cache_interleave_size`

- `vllm/distributed/parallel_state.py` (L1300-1440)
  - `initialize_model_parallel()` 函数
  - 5D Rank 组织逻辑
  - 各并行组的创建

### 2. PCP/DCP 实现
- `vllm/v1/attention/backends/flashinfer.py` (L242-257, L1476-1497)
  - `BatchDCPPrefillWrapper.run()` - PCP 实现
  - `FlashInferImpl.forward()` - DCP 实现
  - `get_dcp_group().all_gather()` 调用

- `vllm/v1/worker/cp_utils.py`
  - CP 相关的工具函数

### 3. MoE 中的 Sequence Parallel
- `vllm/model_executor/models/qwen3_moe.py` (L225-241)
  - `sequence_parallel_chunk()` 调用
  - `tensor_model_parallel_all_gather()` 恢复

### 4. Sequence Parallel 编译 Pass
- `vllm/compilation/sequence_parallelism.py` (L41-270)
  - `FirstAllReduceRMSNormPattern`
  - `MiddleAllReduceRMSNormPattern`
  - `SequenceParallelismPass` 类

---

## 📊 通信量与拓扑优化

详见 **[03_communication_and_topology_optimization.md](03_communication_and_topology_optimization.md)** ⚡

### 通信量精确估算

| 并行策略 | 单位通信量 | Prefill | Decode | 说明 |
|---------|----------|--------|--------|------|
| **TP** | 2B·H | 每层 2 次 All-Reduce | 每层 2 次 | QKV/FFN w2 |
| **PP** | B·H | 传递 activation | 传递 activation | 相邻阶段间 |
| **PCP (A)** | 2BT·H_kv | K/V gather + scatter | 无 | Partial Q + Full K/V |
| **PCP (B)** | ≈ BT·H | 分块交换 K/V | 无 | Ring-attention |
| **DCP** | BT_s·H·(1-1/dcp) | 无 | 跨分片 KV 访问 | 随 dcp_size 增长 |
| **EP** | 2BT·H | 2 × All-to-All | 无 | Dispatch + Collect |

### 硬件拓扑推荐

**单节点（NVLink）**：全部并行维度都能放
```
推荐: TP=8, PCP=1, DCP=min(8, tp_size/H_kv)
```

**多节点（IB/以太）**：按通信频率优化
```
推荐: 
  - 节点内: TP + PCP + DCP
  - 跨节点: PP 或 DP（低频率）
避免: TP 或 EP 跨节点（会饱和链路）
```

**超节点（多级拓扑）**：分层设计
```
层级 1 (快): TP + DCP
层级 2 (中): PCP + EP
层级 3 (慢): PP + DP
```

### DCP 使用算法

1. 计算 `max_dcp = tp_size / H_kv`
2. 若 `max_dcp < 2`：`dcp_size = 1`（不值得）
3. 若 `2 ≤ max_dcp < 4`：试 `dcp_size = 2`
4. 若 `max_dcp ≥ 4`：先用最大值，若通信成瓶颈则降至 2~4

### 配置快速参考表

| 规模 | 模型 | 推荐配置 |
|------|------|--------|
| 8 GPU | 70B, 4K tokens | TP=8, DCP=1 |
| 8 GPU | 70B, 100K tokens | TP=8, DCP=8 |
| 16 GPU (2×8) | 140B | TP=8(node), PP=2 |
| 32 GPU (4×8) | 300B | TP=4, PP=2, DP=2 |
| 64 GPU (8×8) | 500B | TP=4, PCP=2, PP=2, DCP=2, DP=2 |

详见第 03 文档的"推理/训练配置表"。

---

## 💡 高级理解

### 为什么 vLLM 同时支持 3 种 "切分"？

1. **Tensor Parallel (权重切分)**
   - 切分维度: 权重矩阵
   - 目的: 减少权重内存，加速矩阵乘法
   - 通信: 每层 1 次 All-Reduce (RowParallel)

2. **Sequence Parallel (激活切分)**
   - 切分维度: 序列长度
   - 目的: 减少激活内存（Norm 层）
   - 通信: 每层 ReduceScatter + AllGather
   - 代替: All-Reduce 优化

3. **Context Parallel (KV 缓存切分)**
   - 切分维度: Token 序列（Prefill）或 T 维度（Decode）
   - 目的: 降低 TTFT（PCP）和减少 KV 内存（DCP）
   - 通信: 视策略（K/V gather 或 ring send/recv）
   - 新增: 不是 All-Reduce 替代品

### 它们可以混合使用吗？

✅ **可以**：
- TP + SP: 权重切分 + 激活切分（同一模型）
- TP + PCP: 权重切分 + 预填充 token 分块
- TP + DCP: 权重切分 + 解码 KV 缓存 T 维分片
- TP + PCP + DCP: 上述的完整组合
- TP + SP + PCP: 理论上可行（但实现复杂）

✅ **推荐的组合**：
- 单机多卡: TP only (权重太大)
- 多机推理: TP + PCP/DCP (KV 缓存是瓶颈)
- MoE 模型: TP + EP + PCP/DCP (专家分散)
- 长序列: TP + SP + PCP/DCP (激活 + KV 都是瓶颈)

---

## 🎓 学习成果评估

### 你现在应该能够理解：

1. ✅ **基础概念**
   - ColumnParallel vs RowParallel 的区别
   - 为什么权重切分后需要 All-Reduce

2. ✅ **MoE 专家分散**
   - 所有 6 种 MoE 配置 (TP, EP, DP, DP+EP, TP+EP, DP+EP+TP)
   - All-to-All 的 2 次调用模式
   - Token 和 Expert 的路由机制

3. ✅ **序列并行**
   - 图变换的工作原理
   - 3 个前后文一致性保证机制
   - 为什么只支持 RMSNorm（以及为什么其他操作不支持）

4. ✅ **上下文并行**
   - PCP 和 DCP 的本质区别
   - KV 缓存的分散存储方式
   - 为什么 DCP 能"复用" TP 组

5. ✅ **并行组结构**
   - 5D Rank 组织的创建逻辑
   - 各并行维度的转置和重塑规则
   - 如何从 rank 号推导并行 IDs
   - 如何设计自己的并行配置

---

## 🚀 下一步学习方向

### 深化方向 1: 性能优化
- [ ] 研究通信与计算的重叠
- [ ] 学习 All-Reduce 环形约简的优化
- [ ] 研究 flashinfer 中的 kernel fusion

### 深化方向 2: 新的并行策略
- [ ] Ring Attention (环形注意力)
- [ ] Grouped Query Attention (GQA) 与 CP 的交互
- [ ] Speculative Decoding 与 CP 的兼容性

### 深化方向 3: 实现优化
- [ ] 自己实现一个简化版的 SequenceParallelismPass
- [ ] 测试不同配置的性能
- [ ] 优化 KV 缓存的 interleave 策略

### 深化方向 4: vLLM 内部机制
- [ ] BlockManager 如何处理分布式 KV 缓存
- [ ] CUDA Graph 与各种并行策略的兼容性
- [ ] 多进程通信的同步机制

---

## 📚 推荐阅读顺序

1. **快速上手（30 分钟）**
   - README.md 完整版
   - 01_pcp_dcp_detailed_analysis.md 前 2 章

2. **深入理解（2 小时）**
   - moe_sharding_communication_table.md (所有章节)
   - sequence_parallelism_explained.md (所有章节)
   - 02_pcp_dcp_parallel_group_structure.md (章节 1-3)

3. **精通阶段（4 小时）**
   - sequence_parallelism_corrections.md
   - sp_full_analysis.md
   - 02_pcp_dcp_parallel_group_structure.md (章节 4-6)

4. **源代码阅读（6+ 小时）**
   - 根据学习方向，逐个文件阅读

---

## 🎯 最后的建议

**记住 vLLM 设计的核心哲学**：

> "选择对特定问题最有效的优化，避免过度设计。"

这解释了为什么：
- 只实现了 RMSNorm 的 SP（而非所有 Norm）
- MoE 用手动 SP 而非编译 Pass（动态路由无法静态优化）
- Context Parallel 专注于 KV 缓存（因为它是推理的真正瓶颈）

---

**祝学习愉快！🎉**

有任何问题或发现不准确的地方，欢迎指正！

---

文档完成时间: 2026-02-03 22:05 UTC
