# vLLM Sequence Parallelism 详细解释

## 一、什么是 Sequence Parallelism？

Sequence Parallelism (序列并行) 是一种在 Tensor Parallel (TP) 基础上的优化技术，通过将序列维度（token 维度）切分到不同的 TP ranks 上来减少通信开销。

### 核心思想

在标准的 Tensor Parallelism 中：
- 权重被切分（ColumnParallel / RowParallel）
- 输入数据在所有 TP ranks 上是**完整复制**的
- RowParallel 的输出需要 **All-Reduce** 来聚合部分结果

Sequence Parallelism 的改进：
- 权重仍然切分（与 TP 相同）
- 输入数据的**序列维度**也被切分到不同 TP ranks
- 用 **ReduceScatter + All-Gather** 替代 **All-Reduce**

## 二、实现原理

### ⚠️ 重要前提说明

**Sequence Parallelism 只替换通信模式，不改变计算逻辑！**

- ✅ SP 只作用于 **RMSNorm 层**（element-wise 操作）
- ✅ 通过编译时 Pattern Matching 自动替换 `AllReduce → RMSNorm` 模式
- ❌ MLP/Attention 的计算与标准 TP **完全相同**
- ❌ 不需要修改模型代码，自动识别和替换

vLLM 支持的 4 种 Pattern（来自 `sequence_parallelism.py`）：
1. `FirstAllReduceRMSNormPattern` - 第一个 RMSNorm
2. `MiddleAllReduceRMSNormPattern` - 中间 RMSNorm (with residual)
3. `FirstAllReduceRMSNormStaticFP8Pattern` - FP8 量化的第一个 RMSNorm
4. `MiddleAllReduceRMSNormStaticFP8Pattern` - FP8 量化的中间 RMSNorm

### 1. 图变换（Graph Transformation）

vLLM 通过编译时的图变换（Compilation Pass）来实现 Sequence Parallelism：

```python
# 原始模式 (标准 TP):
Input (完整) → AllReduce → RMSNorm → Output (完整)

# Sequence Parallel 模式:
Input (完整) → ReduceScatter → RMSNorm → AllGather → Output (完整)
              ↓
         切分序列维度     本地计算     恢复完整序列
```

### 2. 核心变换逻辑

vLLM 的 `SequenceParallelismPass` 识别以下模式并替换：

#### Pattern 1: First AllReduce + RMSNorm
```python
# Before:
all_reduce = AllReduce(input)        # 聚合所有 TP rank 的输出
rmsnorm = RMSNorm(all_reduce, weight)

# After:
reduce_scatter = ReduceScatter(input)  # 聚合 + 切分序列维度
rmsnorm = RMSNorm(reduce_scatter, weight)  # 各 rank 处理 1/tp_size 的序列
all_gather = AllGather(rmsnorm)       # 恢复完整序列
```

#### Pattern 2: Middle AllReduce + FusedAddRMSNorm (with Residual)
```python
# Before:
all_reduce = AllReduce(mm_output)
rmsnorm, residual_out = FusedAddRMSNorm(all_reduce, weight, residual)

# After:
reduce_scatter = ReduceScatter(mm_output)
residual_chunked = residual[0:reduce_scatter.size(0), ...]  # 切分 residual
rmsnorm, residual_out = FusedAddRMSNorm(reduce_scatter, weight, residual_chunked)
all_gather = AllGather(rmsnorm)
```

### 3. 序列切分函数

```python
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    """
    将输入张量 x 沿序列维度（dim=0）切分到当前 TP rank
    
    Args:
        x: [seq_len, hidden_size] 完整序列
        
    Returns:
        chunked: [seq_len/tp_size, hidden_size] 本地序列片段
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    
    # 1. Padding: 确保 seq_len 能被 tp_size 整除
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        x = nn.functional.pad(x, (0, 0, 0, pad_len))  # 在序列维度末尾 padding
    
    # 2. 切分: 每个 rank 取自己的 chunk
    chunk_size = x.shape[0] // tp_size
    start = tp_rank * chunk_size
    return x[start : start + chunk_size, ...]
```

## 三、如何保证前后文一致性？

### 关键机制 1: 层级通信协调

Sequence Parallelism **不会破坏 Transformer 层间的信息流动**，因为：

1. **All-Gather 恢复完整序列**
   - 每个 Transformer 层的输出通过 All-Gather 恢复为完整序列
   - 下一层的输入是完整的，包含所有 token 的信息
   
2. **Attention 仍然是全局的**
   - Attention 计算（QKV projection + attention + output projection）**仍然看到完整的序列**
   - Sequence Parallel **只作用于 Normalization 层**（RMSNorm/LayerNorm）
   - **重要**: MLP 层的计算与标准 TP 完全相同，SP 只改变 MLP 后的通信模式

### 关键机制 2: 完整的数据流

让我们跟踪一个 token 通过一个 Transformer 层的流程：

```
初始状态: 所有 TP ranks 都有完整序列 [Token_0, Token_1, ..., Token_N]

─────────────── Layer L ───────────────
1. Attention (标准 TP):
   Input: [完整序列] on all ranks
   - Q/K/V projection (ColumnParallel)
   - Attention 计算 (每个 rank 计算部分 heads)
   - Output projection (RowParallel)
   Output after AllReduce: [完整序列] ← 所有 ranks 有相同的完整输出

2. Add & Norm (Sequence Parallel 开始):
   ReduceScatter(attention_output):
     - Rank 0: [Token_0, ..., Token_{N/tp_size}]
     - Rank 1: [Token_{N/tp_size+1}, ..., Token_{2N/tp_size}]
     - ...
   
   RMSNorm (本地计算):
     - 每个 rank 独立对自己的 token chunk 做 normalization
   
3. MLP (接收部分序列，但计算方式与标准TP相同):
   MLP input: [部分序列] on each rank (由上一步ReduceScatter提供)
   - Gate/Up projection (ColumnParallel) - 与标准TP相同
   - Activation
   - Down projection (RowParallel) - 与标准TP相同
   Output: [部分序列] on each rank (RowParallel输出是完整hidden_size但序列维度是部分)

4. 恢复完整序列 (通过下一个 ReduceScatter → RMSNorm → AllGather):
   - MLP 的 RowParallel 输出：[部分序列, hidden_size] on each rank
   - ReduceScatter：聚合 TP ranks 的输出 + 保持序列维度切分
   - RMSNorm：对部分序列做 normalization
   - AllGather：[完整序列] ← 所有 ranks 重新拥有完整序列
     
下一层输入: [完整序列] on all ranks (与 Layer L 输入相同状态)
```

### 关键机制 3: Residual Connection 的处理

Residual connection 是保证一致性的关键：

```python
# 在 Sequence Parallel 中处理 residual:
reduce_scatter = ReduceScatter(mm_output)        # 切分新计算的输出
residual_chunked = residual[0:reduce_scatter.size(0), ...]  # 同步切分 residual

# residual 和新输出的形状一致:
# residual_chunked: [seq_len/tp_size, hidden_size]
# reduce_scatter:   [seq_len/tp_size, hidden_size]

rmsnorm, residual_out = FusedAddRMSNorm(
    reduce_scatter,      # 新输出 (切分)
    weight, 
    residual_chunked     # 旧 residual (切分)
)
```

重点：
- **residual 和当前计算都按相同方式切分**
- 每个 rank 的 residual chunk 对应其负责的 token chunk
- 保证了 `output = input + residual` 的语义正确性

## 四、通信开销对比

### 标准 Tensor Parallel (无 Sequence Parallel)

```
每个 Transformer 层:
- All-Reduce (after attention): 通信量 = seq_len × hidden_size
- All-Reduce (after MLP):       通信量 = seq_len × hidden_size
总通信量 = 2 × seq_len × hidden_size
```

### Sequence Parallel

```
每个 Transformer 层:
- ReduceScatter (after attention): 通信量 = seq_len × hidden_size × (tp_size-1) / tp_size
- All-Gather (before next layer):  通信量 = seq_len × hidden_size × (tp_size-1) / tp_size
- ReduceScatter (after MLP):       通信量 = seq_len × hidden_size × (tp_size-1) / tp_size
- All-Gather (before next layer):  通信量 = seq_len × hidden_size × (tp_size-1) / tp_size

总通信量 = 4 × seq_len × hidden_size × (tp_size-1) / tp_size
```

### 分析

对于 `tp_size = 8`:
- 标准 TP: `2 × seq_len × hidden_size`
- Sequence Parallel: `4 × seq_len × hidden_size × 7/8 = 3.5 × seq_len × hidden_size`

**看起来更多？** 实际上：

1. **通信可以与计算重叠（Overlap）**
   - ReduceScatter 和 All-Gather 可以与后续 GEMM 融合
   - vLLM 支持 `GEMM + ReduceScatter` 和 `All-Gather + GEMM` 融合
   
2. **减少同步点**
   - All-Reduce 是全局同步，所有 ranks 必须等待
   - ReduceScatter/All-Gather 可以流水线化

3. **内存效率**
   - 每个 rank 只存储 `seq_len / tp_size` 的中间激活
   - 对长序列场景显著降低内存占用

## 五、适用场景

### 适合 Sequence Parallel 的场景

✅ **长序列推理**
- seq_len >> hidden_size
- 内存成为瓶颈

✅ **大 Batch Size 训练**
- 多个序列并行处理
- 激活内存占用高

✅ **支持通信-计算重叠的硬件**
- NVLink, IB 等高速互连
- 支持 GEMM 融合的后端

### 不适合的场景

❌ **短序列推理**
- seq_len < hidden_size
- 通信开销占比高

❌ **全图编译不可用**
- Sequence Parallel 需要具体的 shape
- 无法在 piecewise compilation 的 symbolic shape 上工作

❌ **注意力后端不兼容**
- 某些 attention 实现可能不支持序列切分

## 六、vLLM 实现细节

### 1. 编译时启用条件

```python
# vllm/compilation/sequence_parallelism.py
def is_applicable_for_range(self, compile_range: Range) -> bool:
    # 只在以下情况应用 Sequence Parallel:
    # 1. 全图编译模式 (无 Dynamo splitting)
    # 2. 具体 shape 且能被 tp_size 整除
    if (
        not self.compilation_config.splitting_ops
        or self.compilation_config.use_inductor_graph_partition
    ):
        return True
    
    tp_size = get_tensor_model_parallel_world_size()
    return (compile_range.is_single_size()) and (compile_range.end % tp_size == 0)
```

### 2. MoE 中的应用

```python
# vllm/model_executor/models/qwen3_moe.py
def forward(self, hidden_states):
    num_tokens = hidden_states.shape[0]
    
    if self.is_sequence_parallel:
        # 切分序列到各 TP rank
        hidden_states = sequence_parallel_chunk(hidden_states)
    
    # Router 和 Expert 计算 (在切分的序列上)
    router_logits, _ = self.gate(hidden_states)
    final_hidden_states = self.experts(hidden_states, router_logits)
    
    if self.is_sequence_parallel:
        # 恢复完整序列
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]  # 去除 padding
    elif self.tp_size > 1:
        # 标准 TP: All-Reduce
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    
    return final_hidden_states
```

### 3. 与 CUDA Graphs 的兼容性

根据 `cuda_graphs.md`:
- Sequence Parallelism 需要全图编译
- 与 piecewise compilation 不兼容
- 需要设置 `use_inductor_graph_partition=True` (torch >= 2.9)

## 七、总结

### Sequence Parallelism 如何保证一致性？

1. **层级完整性**：每层的输出通过 All-Gather 恢复完整序列
2. **Residual 同步切分**：residual connection 和当前计算按相同方式切分
3. **Attention 仍全局**：不影响跨 token 的注意力机制

### 核心优势

- **内存效率**：激活内存降低到 `1/tp_size`
- **通信优化**：ReduceScatter/All-Gather 可与 GEMM 融合
- **长序列友好**：seq_len 越大，收益越明显

### 实现要点

- **图变换 Pass**：编译时识别和替换通信模式
- **Padding 对齐**：确保 seq_len 能被 tp_size 整除
- **通信融合**：与 GEMM 融合以隐藏通信延迟
