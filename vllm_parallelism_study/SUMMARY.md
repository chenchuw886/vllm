# vLLM 并行化架构学习总结

## ✅ 任务完成状态

### 1️⃣ 文档汇总 ✓
所有学习文档已整理到 `/vllm_parallelism_study/` 目录：

```
vllm_parallelism_study/
├── README.md                                      (主目录)
├── moe_sharding_communication_table.md           (MoE 权重切分)
├── sequence_parallelism_explained.md             (SP 原理)
├── sequence_parallelism_corrections.md           (SP 纠错)
├── sp_full_analysis.md                          (SP 完整分析)
├── 01_pcp_dcp_detailed_analysis.md              (PCP/DCP 详解) ✨
└── 02_pcp_dcp_parallel_group_structure.md       (并行组结构图) ✨
```

**总大小**: ~96 KB，约 7 个完整 Markdown 文档

### 2️⃣ PCP/DCP 理解 ✓

#### Context Parallel (CP) 的核心概念
```
目的: 减少 KV 缓存内存占用
机制: 将 Attention KV head 维度分散到多个 GPU
优势: 每个 GPU 只存储 1/cp_size 的 KV 缓存
```

#### PCP (Prefill Context Parallel)
- **应用**: 推理预填充阶段
- **分散方式**: KV head 维度按 pcp_size 切分
- **通信**: All-Gather (聚合 Q) + ReduceScatter (分散输出)
- **代码位置**: `vllm/v1/attention/backends/flashinfer.py` L242-257

#### DCP (Decode Context Parallel)
- **应用**: 推理解码阶段
- **分散方式**: KV head 维度按 dcp_size 切分
- **通信**: All-Gather (补齐完整 KV) + ReduceScatter (分散输出)
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
- CP: 沿 head 维切分 (减少 KV 缓存内存)

不是通信优化:
- SP: AllReduce → ReduceScatter+AllGather (优化通信)
- CP: 本质上是存储优化 (改变 KV 缓存分布)

但都需要通信:
- PCP/DCP: All-Gather (补齐完整 KV) + ReduceScatter (分散输出)
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
| **PCP** | AllGather + ReduceScatter | Prefill | KV cache | 低 | ~1/pcp |
| **DCP** | AllGather + ReduceScatter | Decode | KV cache | 低 | ~1/dcp |

### 推理流程中的通信

```
Prefill (预填充) 时:
├─ Attention: PCP All-Gather (补齐 Q)
├─ Attention: TP All-Reduce (权重聚合)
├─ MLP: TP All-Reduce (w2 输出)
└─ 保存 KV Cache (分散到 PCP ranks)

Decode (解码) 时:
├─ Attention: DCP All-Gather (补齐 K, V)
├─ Attention: TP All-Reduce (权重聚合)
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
   - 切分维度: Attention head
   - 目的: 减少推理时 KV 缓存内存
   - 通信: 每层 AllGather + ReduceScatter (Attention)
   - 新增: 不是 All-Reduce 替代品

### 它们可以混合使用吗？

✅ **可以**：
- TP + SP: 权重切分 + 激活切分（同一模型）
- TP + PCP: 权重切分 + 预填充 KV 缓存切分
- TP + DCP: 权重切分 + 解码 KV 缓存切分
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
