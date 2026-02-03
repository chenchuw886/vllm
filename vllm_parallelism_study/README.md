# vLLM 并行化架构完整学习指南

欢迎来到 vLLM 并行化架构的深度学习指南！本目录包含了从基础概念到高级实现的完整分析。

## 📚 文档导航

### 第一部分：MoE 与权重切分
- **[moe_sharding_communication_table.md](moe_sharding_communication_table.md)**
  - MoE 专家权重切分详解
  - All-Reduce vs All-to-All 通信算子对比
  - TP, EP, DP, DP+EP 等各种配置下的权重分布
  - All-to-All 通信可视化示例
  - DeepSeek V3.1 实际配置案例

### 第二部分：Sequence Parallelism (序列并行)
- **[sequence_parallelism_explained.md](sequence_parallelism_explained.md)**
  - 序列并行的核心原理
  - 图变换（Graph Transformation）实现
  - 序列切分函数详解
  - **前后文一致性保证机制**（3 个关键机制详解）
  - 完整数据流示例
  - 与 CUDA Graphs 的兼容性

- **[sequence_parallelism_corrections.md](sequence_parallelism_corrections.md)**
  - 重要的逻辑错误纠正
  - SP 的限制和实际支持的操作
  - MoE 手动 SP 与编译 Pass SP 的区别
  - 文档验证方法

- **[sp_full_analysis.md](sp_full_analysis.md)**
  - 为什么 vLLM 只实现了 RMSNorm 的 SP
  - FFN、QKV、LayerNorm 为什么没有 SP
  - MoE 为什么用手动实现而非编译 Pass
  - 成本收益分析

### 第三部分：Context Parallel (上下文并行) - 新增
- **[01_pcp_dcp_detailed_analysis.md](01_pcp_dcp_detailed_analysis.md)** ✨
  - **什么是 Context Parallel？**
  - **PCP (Prefill Context Parallel) 与 DCP (Decode Context Parallel)**
  - 两种 CP 模式的技术实现细节
  - KV 缓存存储和分散方式
  - Attention 计算中的通信模式

- **[02_pcp_dcp_parallel_group_structure.md](02_pcp_dcp_parallel_group_structure.md)** ✨
  - **5D Rank 组织结构详解**
  - **各并行组的创建逻辑**
  - 完整的并行组结构图（多个配置示例）
  - TP、DP、PP、PCP、DCP、EP 的关系
  - 通信流向和拓扑图
  - 实际配置建议

### 第四部分：通信优化与硬件拓扑 - 新增
- **[03_communication_and_topology_optimization.md](03_communication_and_topology_optimization.md)** ✨⚡
  - **通信量精确估算**（TP、PP、DP、EP、PCP、DCP）
  - **单层总通信量分析**（Prefill vs Decode）
  - **硬件拓扑设计**（单节点、多节点、超节点、HPC 集群）
  - **DCP 使用策略**（何时启用、如何选择 dcp_size）
  - **性能预测与优化**（Roofline 模型、瓶颈识别）
  - **配置决策树**（根据集群拓扑快速选择配置）
  - **实际案例分析**（DeepSeek-R1、Qwen3、长序列）
  - **推理/训练配置表**（快速参考）

### 总结与核心发现
- **[SUMMARY.md](SUMMARY.md)** ✨ 📌
  - ✅ 任务完成状态总览
  - 🎯 关键发现汇总（已按官方逻辑更新）
  - 📊 完整架构对比表（通信模式、频率、开销）
  - 💡 为什么 vLLM 支持 3 种"切分"（TP、SP、CP）
  - 🚀 下一步学习方向建议
  - 🔍 vLLM 源代码关键位置速查表
  - 🎓 学习成果评估清单

## 🎯 学习路径建议

### 初级：理解基础并行策略
1. 阅读 `moe_sharding_communication_table.md` 前 3 个章节
   - 了解权重切分的基本概念
   - 理解 ColumnParallel vs RowParallel

2. 阅读 `01_pcp_dcp_detailed_analysis.md` 的"核心概念"
   - 了解为什么需要 Context Parallel
   - 理解 PCP 和 DCP 的区别

3. 快速看 `03_communication_and_topology_optimization.md` 的"硬件拓扑"部分
   - 了解自己集群的拓扑特征
   - 查看推荐配置表

### 中级：深入各种并行策略
4. 完整阅读 `moe_sharding_communication_table.md`
   - 掌握 MoE 各配置下的权重分布
   - 理解 All-to-All 通信的 2 次调用

5. 阅读 `sequence_parallelism_explained.md`
   - 理解 SP 的图变换实现
   - 掌握前后文一致性的 3 个机制

6. 阅读 `01_pcp_dcp_detailed_analysis.md` 完整内容
   - 理解 KV 缓存切分方式
   - 掌握 PCP/DCP 的技术实现

7. 阅读 `03_communication_and_topology_optimization.md` 的"通信量分析"
   - 理解每种并行策略的精确通信量
   - 学会计算单层通信开销

### 高级：完整的并行架构理解与优化
8. 阅读 `02_pcp_dcp_parallel_group_structure.md`
   - 理解 5D Rank 组织
   - 掌握各并行组的创建和关系
   - 能够设计自己的并行配置

9. 完整阅读 `03_communication_and_topology_optimization.md`
   - 掌握 DCP 的使用策略和选择算法
   - 能够根据硬件拓扑设计最优配置
   - 学习性能预测和瓶颈识别方法

10. 阅读 `sequence_parallelism_corrections.md` 和 `sp_full_analysis.md`
    - 理解为什么做某些优化，不做某些优化
    - 能够理解 vLLM 的设计哲学

---

## 🔑 核心概念速查

### 权重切分方式
| 方式 | 切分维度 | 需要通信 | 用途 |
|------|--------|--------|------|
| ColumnParallel | 输出维度 | 否 | w1/w3 (MLP/MoE) |
| RowParallel | 输入维度 | All-Reduce | w2 (MLP/MoE) |
| Replicate | 不切分 | 否 | router/gate |

### 通信算子对比
| 算子 | 数据流 | 应用 | 调用次数 |
|------|--------|------|--------|
| All-Reduce | 聚合求和 | TP RowParallel | 每层 1 次 |
| All-to-All | 重组分散 | MoE Expert Parallelism | 每层 2 次* |
| ReduceScatter + AllGather | 聚合+分散 | SP/PCP/DCP | 每层 1 组 |
| All-Gather | 聚合 | CP Attention | 每层 1 次 |

*注：MoE 中 All-to-All 为 2 次独立调用（dispatch + collect）

### 序列并行的限制
- ✅ 支持：RMSNorm/LayerNorm (element-wise 操作)
- ❌ 不支持：FFN (RowParallel All-Reduce 不能简单替换)
- ❌ 不支持：QKV (Attention 必须全局)
- ⚠️ MoE：已实现手动 SP，但不是编译 Pass

### Context Parallel 配置
| 模式 | 应用阶段 | KV 缓存分散 | 通信组大小 |
|------|--------|----------|---------|
| PCP | 预填充 | 按 head 维度 | pcp_size |
| DCP | 解码 | 按 head 维度 | dcp_size |
| 无 | - | 复制 | - |

### 5D Rank 组织（优先级从高到低）
```
[ExternalDP, DP, PP, PCP, TP]
```

---

## 💡 常见问题

### Q1: MoE 为什么需要 2 次 All-to-All？
**A**: 一次是 dispatch（tokens 发送到 experts），一次是 collect（outputs 返回原位置）。这是两个不同目的的数据重组，需要两次独立的通信原语调用。

### Q2: DCP 为什么"复用" TP 组的 GPU？
**A**: 因为 DCP 不增加 world_size，它只是将同一个 TP 组内的 GPU 进行逻辑上的分组（用于 KV 缓存分散）。所以必须满足 `tp_size >= dcp_size`。

### Q3: 为什么 FFN 没有 Sequence Parallel？
**A**: 因为 FFN 的 RowParallel w2 的 All-Reduce 很难简单地替换为 ReduceScatter + AllGather。需要在 w1 前加 ReduceScatter，破坏了 ColumnParallel 的假设，导致 pattern 过于复杂。

### Q4: PCP 和 DCP 可以同时启用吗？
**A**: 可以。PCP 用于预填充阶段，DCP 用于解码阶段。它们在不同的时间段工作。

### Q5: rank 0 属于哪些并行组？
**A**: 所有。以 (DP=2, PP=2, PCP=2, TP=2) 为例，rank 0 属于：TP_0, PCP_0, PP_0, DP_0, EP_0。

---

## 📊 实际配置示例

### 8 GPU 单机
```
ExternalDP=1, DP=1, PP=1, PCP=1, TP=8, DCP=1
所有权重都切分到 8 张卡，无数据并行、流水线或上下文并行
```

### 16 GPU（2×8）
```
配置 A（优先通信）:
ExternalDP=1, DP=1, PP=2, PCP=1, TP=2, DCP=1
分别在两台机器上运行流水线的两个阶段

配置 B（数据并行）:
ExternalDP=1, DP=2, PP=1, PCP=1, TP=2, DCP=2
两台机器各处理不同的数据，每台用 2 张卡做 TP，2 张卡做 DCP
```

### 64 GPU（8×8）
```
ExternalDP=1, DP=2, PP=4, PCP=2, TP=2, DCP=1
2 个数据并行组，4 个流水线阶段，每阶段 2 张卡做 TP，2 张卡做 PCP
```

---

## 🔗 vLLM 源代码参考

### 关键文件
- **config/parallel.py**: 并行配置定义
- **distributed/parallel_state.py**: 并行组初始化和管理
- **compilation/sequence_parallelism.py**: SP 图变换实现
- **model_executor/models/qwen3_moe.py**: MoE 手动 SP 实现
- **v1/attention/backends/flashinfer.py**: DCP Attention 实现
- **v1/worker/cp_utils.py**: CP 相关工具函数

### 关键函数
- `initialize_model_parallel()`: 初始化所有并行组
- `sequence_parallel_chunk()`: 序列切分
- `get_dcp_group()`: 获取 DCP 组
- `get_pcp_group()`: 获取 PCP 组
- `tensor_model_parallel_all_reduce()`: TP All-Reduce
- `tensor_model_parallel_all_gather()`: 张量并行 All-Gather

---

## 📈 性能考虑

### 通信与计算比例
```
TP: 高通信/计算比 (权重切分，每层都需要通信)
DP: 低通信/计算比 (只在 backward 时通信)
PP: 中等通信/计算比 (阶段间通信，可以流水线隐藏)
SP: 中等通信/计算比 (ReduceScatter+AllGather 可与计算重叠)
CP: 低通信/计算比 (只在 Attention 时通信)
```

### 最优配置因素
1. **网络带宽**: 限制跨机器的并行度（DP、PP、EP）
2. **卡间通信**: 限制单机内的并行度（主要是 TP）
3. **显存大小**: 限制权重和激活的分片数
4. **序列长度**: 影响 KV 缓存（使用 CP 的原因）

---

## 🎓 进阶阅读

1. **论文参考**
   - DeepSpeed ZeRO-1, 2, 3 (DP 的演进)
   - Megatron-LM (TP 和 PP 的经典实现)
   - Ring All-Reduce (优化的 DP 通信)
   - Ulysses (Sequence Parallel 论文)
   - DeepSeek-V2 (MoE 实践)

2. **vLLM 特定
   - Context Parallel: MLA (Multi-head Latent Attention) 的最佳配合
   - 推理优化: Decode 和 Prefill 分离的好处
   - CUDA Graphs: 影响 SP 和 CP 的编译选择

---

## ✅ 检查清单：你是否掌握了？

**基础（第 1-3 步）**
- [ ] 能够解释 ColumnParallel 和 RowParallel 的区别
- [ ] 理解为什么 MoE 需要 All-to-All（以及为什么是 2 次）
- [ ] 了解自己集群的拓扑（单节点、多节点、拓扑细节）
- [ ] 能从推荐配置表中快速查到合适的配置

**进阶（第 4-7 步）**
- [ ] 能够计算 All-to-All 的通信量（token 数、expert 数）
- [ ] 掌握序列切分的 3 个上下文一致性机制
- [ ] 理解 PCP（token 分块）和 DCP（KV T 维分片）的本质区别
- [ ] 能够估算单层的通信量（TP、PP、PCP、DCP）

**精通（第 8-10 步）**
- [ ] 能够根据 rank 号推导其并行 IDs
- [ ] 能够画出自己配置的 5D Rank 组织图
- [ ] 掌握 DCP 的选择算法：何时启用、dcp_size 如何取值
- [ ] 能够用 Roofline 模型预测瓶颈（通信 vs 计算）
- [ ] 能够为自己的集群设计一套完整的并行配置
- [ ] 理解 vLLM 的设计哲学：为什么选择这些优化，不做那些优化
- [ ] 理解各并行维度的通信模式和开销

---

## 📝 版本历史

- **v1.0** (2026-02-03): 初始版本
  - 完整的 MoE 权重切分表
  - Sequence Parallelism 详解
  - Context Parallel (PCP/DCP) 完整分析
  - 并行组结构图

---

## 🤝 贡献

如发现错误或有改进建议，欢迎反馈！

---

**最后更新**: 2026-02-03
**维护者**: vLLM 学习指南项目
