# 为什么 vLLM 只实现了 RMSNorm 的 SP？完整分析

## 一、你的观察是对的：MoE、FFN、QKV 理论上都可以 SP

### 通信模式分析

所有这些操作都涉及 **RowParallel 的 AllReduce**：

| 操作 | 权重类型 | AllReduce 位置 | 是否可 SP | vLLM 是否实现 |
|------|---------|--------------|---------|-------------|
| **Attention** | QKV (CP) + O (RP) | O projection 后 | ✅ 可以 | ❌ 否 (编译 Pass) |
| **FFN/MLP** | w1 (CP) + w2 (RP) | w2 projection 后 | ✅ 可以 | ❌ 否 (编译 Pass) |
| **LayerNorm** | AllReduce → Norm | 在 Norm 前 | ✅ 可以 | ❌ 否 (编译 Pass) |
| **RMSNorm** | AllReduce → Norm | 在 Norm 前 | ✅ 可以 | ✅ 是 (编译 Pass) |
| **MoE** | 专家 w1/w2/w3 | w2 projection 后 | ✅ 可以 | **✅ 是 (手动实现)** |

**关键发现**：
- ✅ **MoE 已经有 SP 实现**，但不是编译 Pass，而是**手动显式切分** (`sequence_parallel_chunk`)
- ❌ **其他操作（FFN、QKV、LayerNorm）都没有 SP 编译 Pass**

---

## 二、MoE 中的 Sequence Parallel 实现

### 已有的实现（手动显式切分）

vLLM 已经支持 MoE 的 SP，但方式是**手动显式切分**，不是编译 Pass：

```python
# vllm/model_executor/models/qwen3_moe.py (L225-241)
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if self.is_sequence_parallel:
        # ✅ 手动切分序列
        hidden_states = sequence_parallel_chunk(hidden_states)

    router_logits, _ = self.gate(hidden_states)
    shared_out, fused_out = self.experts(hidden_states, router_logits)
    final_hidden_states = shared_out + fused_out if shared_out else fused_out

    if self.is_sequence_parallel:
        # ✅ 手动恢复序列
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1:
        # 标准 TP: All-Reduce
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

    return final_hidden_states
```

**与编译 Pass SP 的对比**：

| 特性 | 编译 Pass SP (RMSNorm) | 手动 SP (MoE) |
|------|---------------------|-------------|
| 实现方式 | 自动图变换 | 显式调用函数 |
| 代码改动 | 无（自动） | 有（手动添加代码） |
| 通信模式 | AllReduce → ReduceScatter+AllGather | AllGather (TP group) |
| 适用范围 | 所有使用 RMSNorm 的模型 | 特定 MoE 模型（需要启用 `use_sequence_parallel_moe`） |
| 灵活性 | 低（固定模式） | 高（可自定义） |

---

## 三、为什么 vLLM 没有为 FFN、QKV、LayerNorm 实现 SP？

### 原因分析

#### 原因 1: 不同的 Element-wise 操作复杂性

**RMSNorm 是特殊的**：
- 公式：`y = x / sqrt(mean(x^2) + eps) * w`
- **特性**：只需要每个 token 自己的统计量，不需要跨 token 信息
- **序列切分安全**：对 token 维度切分不影响正确性

**LayerNorm 也满足条件**：
- 公式类似 RMSNorm
- 也是 element-wise（稍微涉及 channel 维度聚合）
- **但 vLLM 没实现** ← 因为现代 LLM 都用 RMSNorm，没必要

**FFN/MLP 问题**：
```python
# FFN 是这样的：
output = w2(activation(w1(input)))

# w1 是 ColumnParallel
w1_out = w1(input)  # [seq_len, inter/tp]

# w2 是 RowParallel (需要 AllReduce)
w2_out = w2(w1_out)  # [seq_len, hidden] 但是部分和
# AllReduce 后才是完整结果

# 问题：w1 接收完整 input，w2 输出部分和
# 能 SP 吗？理论上可以，但需要：
# 1. 切分 w1 的 input
# 2. w1 计算时会改变维度，需要处理
# 3. w2 的 AllReduce 改为 ReduceScatter
```

#### 原因 2: 手动 SP (MoE 方案) vs 编译 Pass SP (RMSNorm 方案) 的取舍

**为什么编译 Pass 只适配 RMSNorm？**

1. **RMSNorm 的 Pattern 简单**
   - 模式固定：`AllReduce(x) → RMSNorm(x, w)`
   - 容易识别和替换
   - Pattern Matcher 可以一劳永逸处理

2. **FFN/QKV 的 Pattern 复杂**
   ```
   # FFN 的 Pattern：
   mm_output = w2(w1(input))  # ColumnParallel → RowParallel
   all_reduce = AllReduce(mm_output)
   rmsnorm = RMSNorm(all_reduce, w)
   
   # 问题：w2 的计算结果本身就是分布式的，AllReduce 才聚合
   # 无法简单地替换 AllReduce
   ```

3. **MoE 必须手动处理**
   ```
   # MoE 的特殊性：
   # - 需要动态路由 (sequence_parallel_chunk 后)
   # - All-to-All 分布式，无法在编译时静态优化
   # - 必须在运行时决定序列切分
   ```

#### 原因 3: 收益 vs 成本

**FFN SP 的收益不高**：
- FFN 占总计算量的 ~2/3
- 但 FFN 的通信只是 w2 的 AllReduce
- 相比 Attention 的 head 切分，通信压力较小
- **成本**：实现编译 Pass，维护代码，处理 edge case

**为什么 MoE SP 优先实现？**
- MoE 的通信压力**巨大**（All-to-All × 2）
- Sequence 维度切分可以显著减少计算量
- 收益明显 > 成本
- 所以采用**手动显式切分** (不是编译 Pass)

---

## 四、为什么 MoE 用手动实现而不是编译 Pass？

### 技术原因

**手动实现 (MoE 的做法)**：
```python
if self.is_sequence_parallel:
    hidden_states = sequence_parallel_chunk(hidden_states)  # 手动切分
    # ... expert 计算 ...
    final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
```

**优势**：
1. **动态路由兼容**：序列切分后，router 基于部分序列做决策（没问题）
2. **灵活控制**：可以在 MoE 层启用/禁用
3. **All-Gather 优化**：直接调用高度优化的 All-Gather 算子
4. **避免编译复杂性**：MoE 的 Pattern Matching 会非常复杂

**编译 Pass 的局限**：
1. **动态 All-to-All 无法静态优化**
   - All-to-All 数据量依赖 router 决策
   - 编译时无法确定通信模式
   
2. **Pattern 复杂**
   - 需要识别 `All-to-All → Expert Compute → All-to-All`
   - 比 `AllReduce → RMSNorm` 复杂 100 倍

3. **符号 shape 问题**
   - MoE 的序列在 All-to-All 后改变
   - 无法在全图编译的 symbolic shape 中表示

---

## 五、为什么 FFN 和 QKV 没有编译 Pass SP？

### FFN SP 的技术障碍

```python
# 标准 FFN (TP):
input: [seq_len, hidden]  (所有 rank 完整)
w1_out = w1(input)        # ColumnParallel → [seq_len, inter/tp]
act = activation(w1_out)  # [seq_len, inter/tp]
w2_out = w2(act)          # RowParallel → [seq_len, hidden] (部分和)
final = AllReduce(w2_out) # [seq_len, hidden] (完整)

# SP 想要做的：
# ReduceScatter(w2_out) 后只有 [seq_len/tp, hidden]
# 但问题：w1 接收的是完整 [seq_len, hidden]
#        w1 输出是 [seq_len, inter/tp]
#        怎样切分 w1 的 input？
```

**具体问题**：
1. **w1 是 ColumnParallel**，接收完整 input
   - 如果硬要切分 w1 的 input，破坏了 ColumnParallel 的假设
   - 需要在 w1 前加 ReduceScatter（但 w1 前通常只有 RMSNorm）

2. **可能的 Pattern**：
   ```
   # 理论上的 FFN SP Pattern:
   AllReduce(prev_layer) → RMSNorm → ReduceScatter
   w1(act)  # 接收切分序列
   w2(w1_out) → AllReduce → ReduceScatter
   RMSNorm(rs_out)
   AllGather
   ```

3. **问题**：
   - Pattern 太复杂，涉及多个通信算子的组合
   - 需要跨越多层，编译 Pass 难以识别
   - 符号 shape 处理困难

### QKV SP 的问题

```python
# Attention (TP):
input: [seq_len, hidden]
q = w_q(input)  # CP → [seq_len, heads/tp, head_dim]
k = w_k(input)  # CP
v = w_v(input)  # CP
attn_out = attention(q, k, v)  # [seq_len, heads/tp, head_dim]
final = w_o(attn_out)  # RP → [seq_len, hidden] (部分和)
final = AllReduce(final)  # [seq_len, hidden]

# 问题：Attention 计算**必须是全局的**
# - 每个 token 需要 attend 到所有其他 tokens
# - 无法在序列维度切分（这会破坏注意力机制）
# 
# 虽然 w_o 的 AllReduce 可以替换为 ReduceScatter
# 但前面的 Attention 计算无法切分
# → 收益非常有限
```

---

## 六、总结：为什么这样设计？

### vLLM 的 SP 策略

| 操作 | 编译 Pass SP | 手动 SP | 原因 |
|------|------------|--------|------|
| **RMSNorm** | ✅ 是 | ❌ 否 | Pattern 简单，容易自动识别 |
| **LayerNorm** | ❌ 否 | ❌ 否 | 现代 LLM 都用 RMSNorm，没必要 |
| **FFN** | ❌ 否 | ❌ 否 | Pattern 复杂，跨多层，收益有限 |
| **QKV** | ❌ 否 | ❌ 否 | Attention 必须全局，序列切分无益 |
| **MoE** | ❌ 否 | ✅ 是 | 动态路由，All-to-All 无法静态优化 |

### 设计哲学

1. **RMSNorm 编译 Pass**
   - **为什么**：Transformer 标准操作，所有模型都有
   - **怎样**：Pattern Matcher 自动识别 `AllReduce → RMSNorm`
   - **收益**：减少激活内存 ~50%，通信优化

2. **MoE 手动 SP**
   - **为什么**：MoE 通信压力大（All-to-All × 2）
   - **怎样**：显式调用 `sequence_parallel_chunk` + `all_gather`
   - **收益**：减少 MoE 计算量、内存，降低通信

3. **为什么不做 FFN/QKV SP**
   - **FFN**：Pattern 复杂，跨层，编译 Pass 难以识别
   - **QKV**：Attention 必须全局，序列切分破坏语义
   - **成本收益不划算**：实现成本高，收益有限

---

## 七、可能的未来改进

### 1. FFN 的编译 Pass SP

**可行性**：低（需要跨层优化）

```python
# 需要识别这样的 Pattern:
Layer N: AllReduce(prev) → RMSNorm
Layer N+1: w1(norm_out) → w2 → AllReduce  # ← 无法识别
```

### 2. LayerNorm 的编译 Pass

**可行性**：高（但优先级低）

```python
# Pattern 简单：
AllReduce(x) → LayerNorm(x, w)

# 为什么没做：
# - 现代 LLM 都用 RMSNorm
# - LayerNorm 的计算成本不是瓶颈
```

### 3. FFN 的手动 SP

**可行性**：中等（需要改动模型代码）

```python
# 类似 MoE 的做法：
if use_sequence_parallel:
    hidden = sequence_parallel_chunk(hidden)
    # ... ffn compute ...
    hidden = tensor_model_parallel_all_gather(hidden, 0)
```

**为什么未实现**：
- 收益不如 MoE（MoE 计算量更大）
- FFN 优化空间已有：pipelined w1+w2 (fused 计算)
- 优先级不高

---

## 八、验证：查看实际代码

### 支持 SP 的模型

```bash
# 查询哪些模型支持 MoE SP
grep -r "use_sequence_parallel_moe\|is_sequence_parallel" vllm/model_executor/models/

# 结果：
# - qwen3_moe.py (Qwen3)
# - qwen3_next.py (Qwen3.5)
# - llama4.py (Llama4-MoE)
# - granitemoe.py (Granite)
# - mimo_v2_flash.py (MiMo V2)
```

### 编译 Pass SP 的模式

```bash
# 查看编译 Pass 的 Pattern
grep -r "AllReduceRMSNorm" vllm/compilation/

# 结果：
# - FirstAllReduceRMSNormPattern
# - MiddleAllReduceRMSNormPattern
# - FirstAllReduceRMSNormStaticFP8Pattern
# - MiddleAllReduceRMSNormStaticFP8Pattern
#
# 注意：全部是 RMSNorm 相关
```

---

## 九、最终答案

### 能否套用 SP 逻辑？

**理论上可以**：
- ✅ FFN (RowParallel 的 AllReduce)
- ✅ QKV (Output projection 的 AllReduce)
- ✅ LayerNorm (AllReduce 后的 norm)
- ✅ MoE (已部分实现)

### 为什么 vLLM 没这么做？

1. **RMSNorm 编译 Pass**
   - Pattern 简单，收益大（激活内存减半）
   - 所有模型都有，一劳永逸

2. **MoE 手动 SP**
   - 通信压力最大（All-to-All × 2）
   - 动态路由无法静态优化
   - 采用显式调用的方式

3. **FFN/QKV 没做**
   - FFN：Pattern 复杂，需要跨层优化
   - QKV：Attention 必须全局，收益有限
   - **成本收益不划算**

### 为什么 MoE 用手动而不是编译 Pass？

- 动态 All-to-All 无法静态优化
- Pattern Matching 会非常复杂
- 手动实现更灵活、更优化

**结论**：vLLM 的 SP 设计是**精明的权衡**——实现了收益最高的（RMSNorm + MoE），避免了收益有限但复杂的（FFN、QKV）。
