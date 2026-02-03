# Sequence Parallelism 重要澄清与纠错

## 一、关键逻辑错误纠正

### 错误 1: SP 不是作用于 MLP 层本身

**❌ 错误理解**：Sequence Parallel 让 MLP 层也能处理切分的序列

**✅ 正确理解**：
- Sequence Parallel **只替换 AllReduce 通信原语**，不改变 MLP 计算本身
- SP 只作用于 **Normalization 层**（RMSNorm/LayerNorm）的通信模式
- MLP 的 ColumnParallel/RowParallel 计算**与标准 TP 完全相同**

### 错误 2: MoE 中的 w1 命名混淆

**澄清**：
- **Router (gate)**: `[hidden_size, num_experts]` - 独立的 Linear 层，计算路由分数
- **Expert w1 (gate_proj)**: `[num_experts, hidden_size, intermediate_size]` - 专家参数，不是 router
- **Expert w3 (up_proj)**: `[num_experts, hidden_size, intermediate_size]` - 专家参数
- **Expert w2 (down_proj)**: `[num_experts, intermediate_size, hidden_size]` - 专家参数

注意区分：
- `gate` (router) - 决定走哪些专家
- `w1` (gate_proj) - 专家内部的门控投影参数

## 二、SP 到底替换了什么？

### 核心：只替换通信模式，不改变计算

vLLM 的 Sequence Parallelism 通过编译时的 **Pattern Matching Pass** 识别并替换以下模式：

```python
# 标准 TP 模式:
AllReduce(input) → RMSNorm(allreduce_out, weight) → output

# SP 替换后:
ReduceScatter(input) → RMSNorm(reduce_scatter_out, weight) → AllGather(rmsnorm_out) → output
```

### 支持的 Pattern（来自源码分析）

根据 `vllm/compilation/sequence_parallelism.py`，vLLM **只支持 4 种 Pattern**：

1. **FirstAllReduceRMSNormPattern**
   - 模式：`AllReduce → RMSNorm`
   - 替换：`ReduceScatter → RMSNorm → AllGather`
   - 适用：Transformer 层的第一个 Normalization

2. **MiddleAllReduceRMSNormPattern**
   - 模式：`AllReduce → FusedAddRMSNorm (with residual)`
   - 替换：`ReduceScatter → FusedAddRMSNorm → AllGather`
   - 适用：中间层的 Post-Norm + Residual

3. **FirstAllReduceRMSNormStaticFP8Pattern**
   - 模式：`AllReduce → RMSNorm → StaticFP8Quant`
   - 替换：`ReduceScatter → RMSNorm → StaticFP8Quant → AllGather`
   - 适用：FP8 量化的第一个 Normalization

4. **MiddleAllReduceRMSNormStaticFP8Pattern**
   - 模式：`AllReduce → FusedAddRMSNorm → StaticFP8Quant (with residual)`
   - 替换：`ReduceScatter → FusedAddRMSNorm → StaticFP8Quant → AllGather`
   - 适用：FP8 量化的中间 Normalization

### 关键结论

**✅ SP 只作用于 RMSNorm/LayerNorm 这种 element-wise 操作**
- 确实如此！vLLM 的实现**只匹配 RMSNorm 相关的 Pattern**
- 没有 LayerNorm 的 Pattern（可能因为 vLLM 主要支持使用 RMSNorm 的模型）

**✅ vLLM 全局只通过 Pass 的方式替换**
- 确实如此！这是**编译时（compile-time）的图变换**
- 不需要修改模型代码，自动识别和替换

**❌ 只有 RMSNorm 操作支持？**
- 理论上任何 element-wise 操作都可以，但 vLLM **实现上只适配了 RMSNorm**
- 原因：Transformer 模型主流使用 RMSNorm，LayerNorm 较少

## 三、完整的数据流修正

### Transformer 层的正确流程

```
输入: 所有 TP ranks 都有完整序列 [B, S, H]

1. Attention Block:
   ┌─ QKV Projection (ColumnParallel, 标准 TP)
   │  Input: [B, S, H] (完整序列)
   │  Output: [B, S, H/tp] (每个 rank 部分维度)
   │
   ├─ Self-Attention
   │  每个 rank 计算部分 heads
   │
   └─ Output Projection (RowParallel, 标准 TP)
      Output: [B, S, H] 但需要 AllReduce 聚合
      
2. ❗️关键：AllReduce → RMSNorm 被 SP Pass 替换
   
   标准 TP:
   ─────────────────────────────────
   AllReduce(attn_output) → 得到 [B, S, H] (完整)
   RMSNorm([B, S, H]) → [B, S, H]
   
   SP 替换后:
   ─────────────────────────────────
   ReduceScatter(attn_output) → 得到 [B, S/tp, H] (序列切分)
   RMSNorm([B, S/tp, H]) → [B, S/tp, H] (每个 rank 处理部分序列)
   AllGather(rmsnorm_out) → [B, S, H] (恢复完整序列)

3. MLP Block:
   输入是完整序列 [B, S, H]
   
   ┌─ Gate/Up Projection (ColumnParallel, 标准 TP)
   │  Input: [B, S, H] (完整序列，与标准 TP 相同)
   │  Output: [B, S, Inter/tp]
   │
   ├─ Activation (SiLU/GELU)
   │
   └─ Down Projection (RowParallel, 标准 TP)
      Output: [B, S, H] 但需要 AllReduce 聚合

4. ❗️再次：AllReduce → RMSNorm 被 SP Pass 替换
   
   ReduceScatter(mlp_output) → [B, S/tp, H]
   RMSNorm + Residual → [B, S/tp, H]
   AllGather → [B, S, H]

下一层输入: [B, S, H] (完整序列)
```

### 关键理解

1. **MLP 本身不变**
   - MLP 的 ColumnParallel/RowParallel 切分与标准 TP 完全相同
   - MLP 接收的是**完整序列**（由上一步 AllGather 恢复）
   - SP 只改变了 MLP **输出后的通信方式**

2. **序列切分只发生在 Normalization 阶段**
   - `ReduceScatter → RMSNorm → AllGather` 这个模式中
   - RMSNorm 接收的是切分序列 `[B, S/tp, H]`
   - AllGather 后恢复为完整序列 `[B, S, H]`

3. **为什么只有 RMSNorm 可以？**
   - RMSNorm 是 element-wise 操作，对每个 token 独立计算
   - 切分序列维度不影响计算正确性
   - 公式：`RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight`
     - 只依赖每个 token 自己的统计量，不需要跨 token 信息

## 四、MoE 中的 `sequence_parallel_chunk`

### 特殊情况：显式切分

在某些 MoE 模型中，可以看到显式调用 `sequence_parallel_chunk`：

```python
# vllm/model_executor/models/qwen3_moe.py
if self.is_sequence_parallel:
    hidden_states = sequence_parallel_chunk(hidden_states)

final_hidden_states = self.experts(hidden_states, router_logits)

if self.is_sequence_parallel:
    final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
```

**这是什么？**
- 这是 MoE 层的**手动序列切分**，不是编译 Pass
- 目的：MoE 计算量大，手动切分序列减少单卡计算量
- 与编译 Pass 的 SP **不冲突**，是两个不同层级的优化

### 两种 SP 的区别

| 特性 | 编译 Pass SP | 手动 `sequence_parallel_chunk` |
|------|-------------|------------------------------|
| **实现方式** | 自动图变换 | 手动显式调用 |
| **作用位置** | AllReduce → RMSNorm | MoE Expert 计算 |
| **通信变化** | AllReduce → ReduceScatter + AllGather | 无（本地切分） |
| **适用范围** | 所有 Transformer 层 | 特定的 MoE 模型 |
| **激活函数** | RMSNorm (element-wise) | Expert 计算 (任意操作) |

## 五、总结：回答你的三个问题

### 问题 1: 相当于目前只有 RMSNorm 这种 element-wise 操作才能用吗？

**✅ 正确**！vLLM 的编译 Pass 实现只适配了 RMSNorm：
- 源码中只有 4 个 Pattern，全部是 `AllReduce → RMSNorm` 的变体
- 理论上可以扩展到其他 element-wise 操作（如 LayerNorm），但目前未实现
- 原因：现代 LLM（Llama, Qwen, etc.）都使用 RMSNorm

### 问题 2: vLLM 全局都只通过 Pass 的方式替换吗？

**✅ 正确**！Sequence Parallelism 是编译时优化：
- 通过 `SequenceParallelismPass` 在编译阶段自动匹配和替换
- 不需要修改模型代码
- 在 Inductor 图上做 pattern matching

但有例外：
- MoE 中的 `sequence_parallel_chunk` 是**手动显式切分**
- 不是编译 Pass，是运行时逻辑

### 问题 3: 只有 RMSNorm 操作支持是吗？

**✅ 正确**！目前 vLLM 只实现了 RMSNorm 的 Pattern：
- `FirstAllReduceRMSNormPattern`
- `MiddleAllReduceRMSNormPattern`
- `FirstAllReduceRMSNormStaticFP8Pattern`
- `MiddleAllReduceRMSNormStaticFP8Pattern`

LayerNorm、其他 Normalization 或 element-wise 操作理论上可以，但**未实现**。

## 六、文档中需要修正的关键错误

### 修正 1: MLP 不受 SP 影响

原文档说"SP 作用于 MLP 和 normalization 层"是不准确的。

**正确表述**：
- SP 只改变通信模式（AllReduce → ReduceScatter + AllGather）
- 只在 **RMSNorm 层触发**
- MLP 层的计算与标准 TP 完全相同，只是接收完整序列或部分序列作为输入

### 修正 2: 序列切分的时机

原文档说"MLP 接收部分序列"容易让人误解 MLP 计算不同。

**正确表述**：
- MLP **总是接收完整序列** `[B, S, H]`（由上一步 AllGather 提供）
- MLP 的 ColumnParallel/RowParallel 与标准 TP 完全相同
- MLP 输出后的 **AllReduce 被替换为 ReduceScatter**，在下一个 RMSNorm 才切分序列

### 修正 3: 数据流描述

原文档的数据流示例中，"MLP 继续 Sequence Parallel" 容易让人误解。

**正确表述**：
```
MLP input: [完整序列] - 由 AllGather 恢复
MLP 计算: 与标准 TP 相同 (ColumnParallel → Activation → RowParallel)
MLP output: RowParallel 输出，等待下一个 AllReduce
下一个 AllReduce → RMSNorm 被 SP Pass 替换为:
  ReduceScatter → RMSNorm(部分序列) → AllGather
```

## 七、验证方式

可以通过以下方式验证：

```python
# 搜索源码中的 Pattern 定义
grep -r "class.*Pattern" vllm/compilation/sequence_parallelism.py

# 输出：
# FirstAllReduceRMSNormPattern
# MiddleAllReduceRMSNormPattern  
# FirstAllReduceRMSNormStaticFP8Pattern
# MiddleAllReduceRMSNormStaticFP8Pattern

# 所有 Pattern 都包含 "RMSNorm"，证实只支持 RMSNorm
```
