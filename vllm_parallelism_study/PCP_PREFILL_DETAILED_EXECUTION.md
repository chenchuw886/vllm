# PCP Prefill 阶段的完整执行流程分析

## 关键发现：vLLM 代码中的 Prefill 实现（flashinfer.py）

根据 vLLM 实际代码（`flashinfer.py` L245-268），我们找到了关键的 Prefill 实现：

```python
def run(
    self,
    layer: torch.nn.Module,
    prefill_query: torch.Tensor,           # [B, Q_len, H_q]
    kv_cache_permute: torch.Tensor,
    key: torch.Tensor,                     # [B, K_len, H_k]
    value: torch.Tensor,                   # [B, V_len, H_v]
    out: torch.Tensor,
):
    # Step 1: AllGather 完整 K/V（在 PCP group 内）
    prefill_query_across_dcp = get_dcp_group().all_gather(
        prefill_query.contiguous(), dim=1
    )
    
    # Step 2: 运行 context（prefill）attention
    output_context_tmp, lse_context_tmp = self._context.run(
        prefill_query_across_dcp,
        kv_cache_permute,
        k_scale=layer._k_scale_float,
        v_scale=layer._v_scale_float,
        return_lse=True,
    )
    
    # Step 3: AllGather attention 输出 + ReduceScatter（使用 LSE 合并）
    output_context, lse_context = cp_lse_ag_out_rs(
        output_context_tmp,
        lse_context_tmp,
        get_dcp_group(),
        return_lse=True,
        is_lse_base_on_e=False,
    )
    
    # Step 4: 运行 new_tokens（实际上这是后面的逻辑）
    output_query, lse_query = self._new_tokens.run(
        prefill_query,
        key,
        value,
        return_lse=True,
    )
    
    # Step 5: 合并 context 和 query 的输出
    merge_attn_states(
        out,
        output_context,
        lse_context,
        output_query,
        lse_query,
    )
    return out
```

---

## 完整的 Prefill 执行流程（8 卡 + PCP=2, DCP=2, TP=4）

### 场景设置

```
World Size: 8 GPU
TP Size: 4
DCP Size: 2  
PCP Size: 2
Total CP size (dcp × pcp) = 4

Rank 组织:
├─ PCP Group 0: [0, 1]  (GPU 0-1 处理第一段序列)
├─ PCP Group 1: [2, 3]  (GPU 2-3 处理第二段序列)
├─ PCP Group 2: [4, 5]  (GPU 4-5 处理第三段序列)
└─ PCP Group 3: [6, 7]  (GPU 6-7 处理第四段序列)

DCP Groups（在 TP 内部）:
├─ DCP Group A: [0, 2, 4, 6]  (TP rank 0 的所有 GPU)
├─ DCP Group B: [1, 3, 5, 7]  (TP rank 1 的所有 GPU)
...

输入:
  Sequence Length T = 1024 tokens
  Batch Size B = 1
  Hidden Dim H = 4096
  KV Head Dim H_kv = 128
```

### 执行步骤详解

#### **Step 1: 序列切分与 Q/K/V 投影**

```python
# 每个 PCP rank 只处理自己的序列段

GPU 0 (PCP idx=0):
  Q_local = W_q @ X[0:512]        # [512, 4096]
  K_local = W_k @ X[0:512]        # [512, 128]
  V_local = W_v @ X[0:512]        # [512, 128]

GPU 1 (PCP idx=0):
  Q_local = W_q @ X[0:512]        # [512, 4096]（TP group 内相同的 W_q）
  K_local = W_k @ X[0:512]        # [512, 128]（TP group 内不同的权重切分）
  V_local = W_v @ X[0:512]        # [512, 128]（TP group 内不同的权重切分）

GPU 2 (PCP idx=1):
  Q_local = W_q @ X[512:1024]     # [512, 4096]
  K_local = W_k @ X[512:1024]     # [512, 128]
  V_local = W_v @ X[512:1024]     # [512, 128]

GPU 3 (PCP idx=1):
  Q_local = W_q @ X[512:1024]     # [512, 4096]
  K_local = W_k @ X[512:1024]     # [512, 128]
  V_local = W_v @ X[512:1024]     # [512, 128]

注意：GPU 0 和 GPU 1 的 Q 相同（同一 TP 组）
      但 K/V 不同（按 head 维度切分）
```

#### **Step 2: AllGather 完整 K/V（在 PCP Group 内）**

这是 **RFC 的关键实现**！

```python
# 在 DCP Group 内做 AllGather
# DCP Group A (包含 GPU 0, 2, 4, 6):

GPU 0 执行:
  K_local = [512, 64]  (H_kv=128, TP=4，所以每卡 128/4=32... 不对)
  
# 让我重新理清... DCP 是在 TP 组内！
# TP group: [0, 1] (dcp 在这里)
# 所以 DCP Group: [0, 1]

GPU 0 (DCP idx=0):
  K_local = [512, 128]  (完整 H_kv)
  V_local = [512, 128]

GPU 1 (DCP idx=1):
  K_local = [512, 128]  (完整 H_kv)
  V_local = [512, 128]

# AllGather 后：
K_full_GPU0 = [[512, 128], [512, 128]] → [1024, 128]  ✓ 完整序列的 K
V_full_GPU0 = [[512, 128], [512, 128]] → [1024, 128]  ✓ 完整序列的 V

K_full_GPU1 = [[512, 128], [512, 128]] → [1024, 128]  ✓ 完整序列的 K
V_full_GPU1 = [[512, 128], [512, 128]] → [1024, 128]  ✓ 完整序列的 V
```

**这一步的关键代码**：
```python
prefill_query_across_dcp = get_dcp_group().all_gather(
    prefill_query.contiguous(), dim=1
)
```

等等，这里 AllGather 的是 `prefill_query`，不是 K/V！让我重新理解...

#### **Step 2 修正：真正的 AllGather**

```python
# 实际上代码里 AllGather 的是 Query！
# 因为 DCP Group 包含 TP 内的不同 GPU
# 而 Query 在 TP 内是分散的（head 维度切分）

GPU 0 (TP rank=0):
  Q_0 = [512, 2048]  (H/2 = 4096/2，TP=2 时)
  
GPU 1 (TP rank=1):
  Q_1 = [512, 2048]  (另一半 head)

# AllGather Q（在 TP group [0, 1] 内）：
Q_full_GPU0 = [[512, 2048], [512, 2048]] → [512, 4096]
Q_full_GPU1 = [[512, 2048], [512, 2048]] → [512, 4096]

# K/V 的 AllGather 在 context 的内部实现中
# (这是 FlashInfer 库的逻辑)
```

#### **Step 3: Attention 计算**

```python
output_context_tmp, lse_context_tmp = self._context.run(
    Q_full,           # [batch, seq, 4096]
    K,               # K 在这个函数内部会被 gather
    V,               # V 在这个函数内部会被 gather
    return_lse=True,
)

# 每张卡执行：
# GPU 0 的输出: attention(Q_full, K_full, V_full)
#              = [512, 4096]  (自己的 Q chunk 的输出)
#
# GPU 1 的输出: attention(Q_full, K_full, V_full)
#              = [512, 4096]  (自己的 Q chunk 的输出)

# ⚠️ 重要：两张卡的输出是**不同的**！
# 因为它们的 Q 虽然来自相同的逻辑序列段 [0:512]
# 但是 Q 的某些 head 来自不同的 GPU
# 所以需要合并！
```

#### **Step 4: 使用 LSE 合并 Attention 输出（ReduceScatter）**

**这是我们不明白的部分！**

```python
output_context, lse_context = cp_lse_ag_out_rs(
    output_context_tmp,
    lse_context_tmp,
    get_dcp_group(),
    return_lse=True,
    is_lse_base_on_e=False,
)
```

**LSE 是什么？**
```
LSE = Log-Sum-Exp
是 FlashAttention 用来稳定数值的技巧

FlashAttention 返回：
  - output: attention 的输出 [B, seq, H]
  - lse: log-sum-exp 统计量 [B, seq]
  
LSE 记录了：lse[i] = log(sum(exp(scores[i])))
这用来在多卡合并 attention 时恢复正确的 softmax
```

**为什么需要 ReduceScatter？**

```
GPU 0 计算的 attention output [512, 4096] 来自：
  - head 0-1023: 来自 GPU 0 的 head
  - head 1024-2047: 来自 GPU 1 的 head （AllGather 得到的）
  - head 2048-4095: 来自其他 GPU 的 head（AllGather 得到的）

GPU 1 计算的 attention output [512, 4096] 也是类似的混合

现在要 ReduceScatter：
  - GPU 0 收集 head 0-1023 的结果（来自两张卡）
  - GPU 1 收集 head 1024-2047 的结果（来自两张卡）

使用 LSE 来正确地合并来自不同卡的 softmax 结果
```

**伪代码示意**：

```python
# GPU 0 和 GPU 1 各自计算了 attention
O_0, lse_0 = attention_compute_GPU0()  # [512, 4096], lse: [512]
O_1, lse_1 = attention_compute_GPU1()  # [512, 4096], lse: [512]

# 合并逻辑（简化版）：
# head 0-1023 来自 GPU 0，head 1024-2047 来自 GPU 1
lse_max = max(lse_0, lse_1)  # 数值稳定
exp_0 = exp(lse_0 - lse_max)
exp_1 = exp(lse_1 - lse_max)

# 合并每个 head
for head in range(4096):
    if head < 2048:  # 第一部分 head
        O_merged[head] = (O_0[head] * exp_0 + O_1[head] * exp_1) / (exp_0 + exp_1)
    else:  # 第二部分 head
        O_merged[head] = (O_0[head] * exp_0 + O_1[head] * exp_1) / (exp_0 + exp_1)

# ReduceScatter：
#   GPU 0 ← O_merged[0:2048]
#   GPU 1 ← O_merged[2048:4096]
```

---

## 再看代码中 Prefill 的完整流程

```python
def run(self, layer, prefill_query, kv_cache_permute, key, value, out):
    # Step 1: AllGather Q（在 DCP group 内，恢复完整 head）
    prefill_query_across_dcp = get_dcp_group().all_gather(
        prefill_query.contiguous(), dim=1
    )
    # prefill_query: [batch, seq_len, H/tp_size] → [batch, seq_len, H]
    
    # Step 2: 计算 prefill context attention
    output_context_tmp, lse_context_tmp = self._context.run(
        prefill_query_across_dcp,  # 完整 Q
        kv_cache_permute,
        key,                        # K（flshattention 内部会处理 DCP）
        value,                      # V（flshattention 内部会处理 DCP）
        return_lse=True,
    )
    # 输出形状：[batch, seq_len, H]
    # 但这个输出混合了多个 GPU 的 head！
    
    # Step 3: AllGather attention output，然后 ReduceScatter（使用 LSE）
    output_context, lse_context = cp_lse_ag_out_rs(
        output_context_tmp,
        lse_context_tmp,
        get_dcp_group(),
        return_lse=True,
    )
    # 这一步后：
    # - output_context: 每个 GPU 只拥有自己负责的 head 的输出
    # - 形状变为 [batch, seq_len, H/tp_size]（回到分片状态）
    
    # Step 4-5: 后续的 new_tokens 和 merge（这是 PCP 的后续逻辑）
    output_query, lse_query = self._new_tokens.run(...)
    merge_attn_states(out, output_context, lse_context, output_query, lse_query)
    
    return out
```

---

## 最终理解：Prefill 的输出去向

### 问题：每张卡仅对自己的 Q chunk 计算 Attention，结果最后怎么合并的？

**答案：使用 AllGather + ReduceScatter（带 LSE 合并）**

```
GPU 0 (处理 Token [0:512]):
  - 计算得到 [512, 4096] 的输出
  - 包含混合的 head（来自多个 GPU）
  ↓ cp_lse_ag_out_rs
  - ReduceScatter 后得到 [512, H/tp_size]
  - 这部分被存储到 KV cache

GPU 1 (处理 Token [512:1024]):
  - 计算得到 [512, 4096] 的输出
  - 包含混合的 head
  ↓ cp_lse_ag_out_rs
  - ReduceScatter 后得到 [512, H/tp_size]
  - 这部分被存储到 KV cache

最终：
  - GPU 0 拥有 Token [0:512] 的完整输出（按 TP head 切分）
  - GPU 1 拥有 Token [512:1024] 的完整输出（按 TP head 切分）
  - 两个 GPU 的输出不重复！
```

### 关键发现

1. **AllGather 不是只用来获取完整 K/V 的**
   - 也用来获取完整 Q（恢复被 TP 切分的 head）

2. **结果通过 ReduceScatter 去重**
   - 每个 GPU 最终只持有自己负责的 head 的输出
   - 这样不会产生重复

3. **使用 LSE 来正确合并**
   - FlashAttention 的标准做法
   - 保证数值稳定和正确性

4. **PCP Prefill 和 DCP Decode 的逻辑相反**
   ```
   Prefill (PCP):
     - AllGather Q/K/V → 完整
     - Attention 计算（混合 head）
     - ReduceScatter → 分片（不重复）
   
   Decode (DCP):
     - KV 原本就是分片的（存储时分片）
     - AllGather K/V → 完整
     - Attention 计算
     - 不需要 ReduceScatter（decode 只生成 1 token，每卡都相同）
   ```

---

## 完整的执行时间线

```
时间点     GPU 0 (处理 Token [0:512])              GPU 1 (处理 Token [512:1024])
────────────────────────────────────────────────────────────────────
T0        计算 Q_0, K_0, V_0                      计算 Q_1, K_1, V_1
          [512, 2048], [512, 128], [512, 128]     [512, 2048], [512, 128], [512, 128]

T1        AllGather Q：                           AllGather Q：
          Q_full = [Q_0, Q_1] → [512, 4096]       Q_full = [Q_0, Q_1] → [512, 4096]
          
          AllGather K/V（同步）                    AllGather K/V（同步）
          K_full = [K_0, K_1] → [1024, 128]       K_full = [K_0, K_1] → [1024, 128]
          
T2        Attention 计算：                        Attention 计算：
          O_0 = attn(Q_full, K_full, V_full)      O_1 = attn(Q_full, K_full, V_full)
          [512, 4096], lse: [512]                 [512, 4096], lse: [512]

T3        cp_lse_ag_out_rs（AllGather + RS）：   cp_lse_ag_out_rs（AllGather + RS）：
          AllGather O_0 + O_1                      AllGather O_0 + O_1
          ReduceScatter with LSE merge：           ReduceScatter with LSE merge：
          O_GPU0 = [512, 2048]（head 部分）       O_GPU1 = [512, 2048]（head 部分）

T4        存储到 KV cache                        存储到 KV cache
          KV_cache[0:512] = ...                  KV_cache[512:1024] = ...
```

---

## 为什么这样设计？

1. **Prefill 的目标**：并行化序列计算，降低 TTFT
   - 不同 GPU 处理不同的 token 段
   - 最后合并结果（ReduceScatter 避免重复）

2. **与 DCP 的区别**
   - **PCP**：序列分片 → AllGather → Attention → ReduceScatter
   - **DCP**：序列分片（存储）→ AllGather（使用时）→ Attention → 无 ReduceScatter（decode 重复计算没关系）

3. **通信成本与收益**
   - **成本**：4 次 AllGather/Scatter（Q + K + V + output）
   - **收益**：TTFT 从 T 降低到 T/pcp_size（单位时间单位）
   - 当序列很长时，这个权衡是值得的

