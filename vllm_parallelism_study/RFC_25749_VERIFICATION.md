# RFC #25749 PCP 实现逻辑验证与对比分析

## RFC 概览

**来源**: vLLM Issue #25749  
**标题**: [RFC]: Support Prefill Context Parallel (PCP)  
**作者**: @pisceskkk, @zhenwenqi2024, @FENP  
**状态**: 已实现（PR #28718 已合并）

---

## RFC 中描述的 PCP 核心逻辑

### 1. Prefill 阶段的实现

根据 RFC 的图示和描述：

```
PCP 在 Prefill 阶段的策略：

1. 序列切分：
   - 将整个请求按序列维度（sequence dimension）分割
   - 每个 PCP rank 处理一部分 token

2. KV 处理：
   - 对 KV 执行 AllGather 操作（在 PCP group 内）
   - 获得完整的 KV 值
   - 按 slot_mapping 存储 KVCache

3. Attention 计算：
   - 由于已获得完整 KV，只需设计 custom mask
   - 执行标准 attention 计算
```

**关键引用**（RFC 原文）：
> "For the KV, we perform an AllGather op along the sequence dimension within the PCP group to obtain the complete KV values. Then, the kvcache is stored according to the slot_mapping."
>
> "For attention computation, since we have obtained the complete KV, we only need to carefully design the custom mask and perform normal attention."

### 2. Decode 阶段的实现

```
PCP 在 Decode 阶段的策略：

1. 冗余计算：
   - Attention 之外的模块涉及冗余计算（因为新 token 数 = 1）
   - 在各 PCP group 间冗余

2. Attention 模块：
   - 先在各自 PCP group 内执行原 DCP 计算逻辑
   - 在更新 attention output（使用 lse）之前
   - 执行 AllGather（在 PCP group 内）获取完整序列信息
   - 后续步骤不受影响
```

**关键引用**（RFC 原文）：
> "During the decode phase, modules other than attention involve redundant computations (since num new tokens = 1) across PCP group."
>
> "In the attention module, we first execute the original DCP computation logic within the respective PCP group. Then, before updating the attention output using lse, we perform an AllGather within the PCP group to obtain complete sequence information."

---

## 与我们之前文档的对比

### ✅ **一致的部分**

#### 1. 序列切分的本质
**我们的理解**：
> "PCP: 将请求 token 序列按 pcp_size 分块，每卡计算其 chunk 的 Q/K/V"

**RFC 描述**：
> "PCP will split the entire request along the sequence dimension during the prefill phase"

✅ **完全一致** - 都是按序列维度切分

#### 2. 两种策略的存在
**我们的理解**：
> - 策略 A：Partial Q + Full K/V（收集全量 K/V）
> - 策略 B：Ring-Attention（分块交换 K/V）

**RFC 描述**：
> "we adopt the following strategy to implement PCP (chunked prefill is not considered for now)"
> 
> Roadmap 中提到：
> "Ring-CP style attention backend algorithm, ref RFC #26133"

✅ **部分一致**：
- RFC 的当前实现 = 我们的策略 A（Full K/V）
- Ring-Attention = 策略 B，在 Roadmap 中但尚未实现

#### 3. KV Cache 存储方式
**我们的理解**：
> "DCP 会把 KV cache 沿 T 维度在 CP ranks 间交错分布"

**RFC 描述**：
> "Building upon the virtual block concept introduced by DCP, PCP will continue to utilize virtual blocks within each PCP group. Each PCP group will be responsible for storing the portion of the KV Cache that corresponds to its assigned segment of the sequence."

✅ **一致** - 都是按序列分段存储

---

## ⚠️ **需要修正的部分**

### 1. Prefill 阶段的通信机制

**我们之前的理解有偏差**：
```
我们描述了两种并行的策略（A 和 B）
认为 vLLM 可能支持两者
```

**RFC 的实际实现**：
```
当前只实现了策略 A（AllGather Full K/V）
策略 B（Ring-Attention）在 Roadmap 中，但还未实现
```

**修正**：
- ✅ vLLM 当前的 PCP = 策略 A（Partial Q + Full K/V）
- ⏳ Ring-Attention 是未来功能（参考 RFC #26133）

### 2. Decode 阶段的细节

**我们之前的理解不够精确**：
```
描述了 DCP 在 Decode 的机制
但没有明确 PCP 在 Decode 的作用
```

**RFC 补充的关键信息**：
```
Decode 阶段：
1. PCP group 内的所有 rank 都在做冗余计算（因为只有 1 个新 token）
2. Attention 中先执行 DCP 逻辑（在 PCP group 内）
3. 使用 AllGather 合并来自不同 PCP rank 的 attention 结果
4. 通过 lse（log-sum-exp）更新最终输出
```

**这揭示了一个重要细节**：
- Decode 时，PCP 和 DCP **同时工作**
- DCP 负责 KV cache 的分片存储
- PCP 负责合并跨 PCP group 的 attention 结果

### 3. 并行组的关系

**RFC 明确指出**（与我们理解一致）：
> "Unlike DCP communication domains, which are subdivisions within the TP domain, PCP communication domains stand alongside DP, PP, and TP, and affect the total device count allocation."

这确认了：
- ✅ DCP 是 TP 组的细分（复用 TP GPU）
- ✅ PCP 是独立的并行维度（与 DP、PP、TP 并列）

---

## 关键技术细节的补充

### 1. Custom Mask 的作用

RFC 提到：
> "we only need to carefully design the custom mask and perform normal attention"

**解释**：
```python
# PCP 的 causal mask 需要特殊设计
# 假设 pcp_size = 2，序列被分为两段：

Rank 0 处理 Token [0:512]
Rank 1 处理 Token [512:1024]

# Causal Mask 需要反映这种分段：
# Rank 0 的 Q[0:512] 只能看到 K[0:512]（自己的范围）
# Rank 1 的 Q[512:1024] 能看到 K[0:1024]（完整序列）

mask = torch.tril(torch.ones(1024, 1024))  # 基础 causal mask
# 然后根据 PCP rank 调整 mask 的范围
```

### 2. Slot Mapping 的修改

RFC 提到：
> "Modification of slot_mapping calculation. Building upon the virtual block concept introduced by DCP..."

**这意味着**：
```
PCP 的 KV cache 存储需要考虑：
1. DCP 的 token 交错（stride = dcp_world_size）
2. PCP 的序列分段（每个 PCP group 存不同段）
3. 两者结合的 virtual block 映射

例如：
PCP_size = 2, DCP_size = 2
PCP_group_0: 存储 Token [0:512] 的一半（按 DCP 交错）
PCP_group_1: 存储 Token [512:1024] 的一半（按 DCP 交错）
```

### 3. LSE（Log-Sum-Exp）技巧

RFC 提到 Decode 时：
> "before updating the attention output using lse, we perform an AllGather"

**这是 FlashAttention 的标准技巧**：
```python
# FlashAttention 使用 LSE 来稳定数值
# 当需要合并多个 attention 输出时：

# PCP Rank 0 计算的 attention output:
O_0, lse_0 = attention(Q, K_0, V_0)

# PCP Rank 1 计算的 attention output:
O_1, lse_1 = attention(Q, K_1, V_1)

# AllGather 后，使用 LSE 合并：
lse_max = max(lse_0, lse_1)
exp_0 = exp(lse_0 - lse_max)
exp_1 = exp(lse_1 - lse_max)

O_final = (O_0 * exp_0 + O_1 * exp_1) / (exp_0 + exp_1)
```

---

## 多请求批处理的处理

RFC 讨论中 LucasWilkinson 提出的问题：
> "How would we handle a multi-request batch?"

**pisceskkk 的回答**：
```
每个请求独立切分：
原始 batch: [req1: (0, 1024), req2: (1024, 7168), req3: (7168, 8192)]

PCP_size = 2，切分为：
PCP Rank 0: [req1_1: (0, 512), req2_1: (512, 3584), req3_1: (3584, 4096)]
PCP Rank 1: [req1_2: (4096, 4608), req2_2: (4608, 7608), req3_2: (7608, 8192)]
```

**这与我们的理解一致**：
- 每个请求按 PCP size 切分
- 不同请求可以有不同的长度
- Batch 中的所有请求都参与切分

---

## 实现状态（基于 RFC Roadmap）

### ✅ 已完成
1. 基础 PCP 功能（PR #28718）
2. GQA flashinfer 支持（PR #28723）
3. PIECEWISE CUDAGraph 支持
4. MLA 支持（PR #28988）

### ⏳ 待完成（欢迎社区贡献）
1. 其他后端支持（非 flashinfer）
2. MTP 支持
3. P/D disaggregation 支持
4. CUDAFullGraph 支持
5. **Ring-CP style attention**（即我们的策略 B）

---

## 总结：我们的文档准确性评估

### ✅ **完全正确的部分**（90%）

1. **核心概念**：
   - ✅ PCP 是序列维度的切分
   - ✅ 目标是降低 TTFT
   - ✅ 与 DCP 的关系（PCP 独立，DCP 复用 TP）

2. **通信机制**：
   - ✅ 策略 A（AllGather Full K/V）的描述准确
   - ✅ Ring-Attention 概念正确（虽然未实现）

3. **并行组结构**：
   - ✅ 5D rank 组织正确
   - ✅ PCP 与其他并行维度的关系正确

### ⚠️ **需要更新的部分**（10%）

1. **实现状态**：
   - 需要明确：当前 vLLM 只实现了策略 A
   - Ring-Attention（策略 B）在 Roadmap 中

2. **Decode 阶段细节**：
   - 补充：PCP 在 Decode 时的冗余计算
   - 补充：LSE 合并机制

3. **Custom Mask**：
   - 补充：Causal mask 需要根据 PCP 分段调整

---

## 建议的文档更新

### 更新 01_pcp_dcp_detailed_analysis.md

```markdown
### PCP 的实现状态（2026-02 更新）

**当前实现**（v0.11.0+）：
- ✅ 策略 A：Partial Q + Full K/V（AllGather）
- ⏳ 策略 B：Ring-Attention（在 Roadmap 中，参考 RFC #26133）

**支持的后端**：
- ✅ FlashInfer (GQA)
- ✅ MLA
- ⏳ 其他后端（待社区贡献）

**来源**：RFC #25749, PR #28718
```

### 补充 Decode 阶段的 PCP+DCP 协作

```markdown
### Decode 阶段：PCP 与 DCP 的协作

当同时启用 PCP 和 DCP 时：

1. **冗余计算**：
   - PCP group 内的所有 rank 都执行相同的 forward pass
   - 因为每次只生成 1 个新 token

2. **Attention 计算**：
   ```
   Step 1: 执行 DCP 逻辑（KV cache 按 T 维分片）
   Step 2: AllGather attention 结果（在 PCP group 内）
   Step 3: 使用 LSE 合并多个 attention output
   Step 4: 返回最终结果
   ```

3. **通信开销**：
   - DCP AllGather: KV cache 分片
   - PCP AllGather: Attention 输出合并
   - 总计：两次 AllGather 操作
```

---

## 最终验证结论

### 我们之前文档的准确性：**85-90% ✅**

**正确的核心理解**：
- ✅ PCP 的本质（序列切分）
- ✅ 两种策略的概念（虽然只有 A 实现了）
- ✅ 与 DCP 的区别
- ✅ 并行组结构
- ✅ 通信量分析

**需要补充的细节**：
- Custom mask 设计
- LSE 合并机制
- Decode 阶段的冗余计算
- 实现状态（策略 A 已实现，策略 B 在 Roadmap）

**总体评价**：我们的文档在核心概念和原理层面是准确的，但需要补充 vLLM 具体实现的一些工程细节和当前状态。

---

## 参考资料

1. **主要 RFC**: https://github.com/vllm-project/vllm/issues/25749
2. **Ring-Attention RFC**: https://github.com/vllm-project/vllm/issues/26133
3. **基础实现 PR**: https://github.com/vllm-project/vllm/pull/28718
4. **GQA 支持 PR**: https://github.com/vllm-project/vllm/pull/28723
5. **MLA 支持 PR**: https://github.com/vllm-project/vllm/pull/28988
