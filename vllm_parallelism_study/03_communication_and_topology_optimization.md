# 通信量分析与硬件拓扑优化

## 一、通信量精确估算

### 符号定义

| 符号 | 含义 |
|------|------|
| B | 批量大小（batch size） |
| T | Prefill 序列长度 |
| 1 | Decode 单 token |
| H | Hidden dimension（隐层维度） |
| H_q | Query head dimension（通常 = H/n_h） |
| H_k | KV head dimension（通常 = H/n_kv） |
| n_h | 总 query head 数 |
| n_kv | KV head 数 |
| H_inter | FFN intermediate dimension（通常 = 4~8 倍 H） |
| k | MoE 中 top-k 数 |
| n_e | MoE 中总专家数 |
| T_s | KV cache 总长度（包括 history） |

### 各并行策略通信量

#### 1. Tensor Parallel (TP)

**TP 组通信**：每层涉及两次 All-Reduce（QKV/attention 后、FFN w2 后）

| 阶段 | 操作 | 通信量 | 备注 |
|------|------|--------|------|
| Prefill | All-Reduce (2×) | 2BT·H | QKV proj 后、FFN w2 后 |
| Decode  | All-Reduce (2×) | 2B·H | 每 token 一次 |

**带宽利用**：
- 全 All-Reduce（N 卡）：实际通信 ≈ 2(N-1)/N × 数据量
- 环形拓扑下最优效率可达 ~80%

#### 2. Pipeline Parallel (PP)

**PP 阶段间通信**：每个 micro-batch 在相邻阶段间传递

| 阶段 | 操作 | 通信量 | 频率 |
|------|------|--------|------|
| Forward | Send activation | B·T·H | 每个 micro-batch |
| Backward | Send gradient | B·T·H | 每个 micro-batch |

**总计（整个 step）**：≈ 2 × B·T·H × #layers / pp_size 

**优化**：pipeline bubbles 会增加总通信时间

#### 3. Data Parallel (DP)

**DP 通信**（训练阶段）：backward 时梯度 All-Reduce

| 阶段 | 操作 | 通信量 | 频率 |
|------|------|--------|------|
| Backward | Gradient All-Reduce | 参数量 | 每个 step 一次 |
| Prefill | 无 | 0 | 推理无 DP 通信 |
| Decode  | 无 | 0 | 推理无 DP 通信 |

**推理时**：推理通常不用 DP（多卡推同一样本用 TP），所以 DP 通信量 = 0

#### 4. Expert Parallel (EP / MoE)

**All-to-All 通信**：Dispatch 和 Collect 两次调用

CommVol_EP = 2 × B·T·H × (avg_tokens_per_expert / n_e)

**更精确公式**（假设 top-k = k）：
- **Dispatch**（Token → Expert 重组）：B·T·H
- **Collect**（Expert output → Token 重组）：B·T·H
- **总计**：2 × B·T·H（与 k 无关，只取决于 token 总数）

**特点**：
- 两次 All-to-All 互相独立
- 通信量 **不** 随 expert 数 $n_e$ 线性增长（所有 token 都参与）
- 当 $n_e$ 较大时，每个 expert 分到的 token 反而减少

#### 5. Prefill Context Parallel (PCP)

**策略 A：Partial Q + Full K/V**

CommVol_PCP-A = B·T·H_kv × 2

说明：
- 第 1 次：Gather K/V，量级 B·T·H_kv
- 第 2 次：Scatter output back，量级 B·T·H

**策略 B：Partial Q + Partial K/V (Ring-Attention)**

CommVol_PCP-B = B·T·H × (pcp_size - 1) / pcp_size + 计算通信重叠

说明：
- 分块交换 K/V，总 hop 数 = pcp_size - 1
- 可与计算流水化，有效通信时间可减少至 ≈ 1/3 ~ 1/2

**对比**：
- 策略 A：通信集中，易带宽饱和，但实现简单
- 策略 B：通信分散，易流水化，但需算子支持

#### 6. Decode Context Parallel (DCP)

**KV cache 分片后的访问通信**

$$\text{CommVol}_{DCP} \approx B \cdot T_s \cdot H \times \frac{dcp\_size - 1}{dcp\_size}$$

说明：
- KV cache 按 $T$ 维分片到 $dcp\_size$ 卡
- 本卡拥有 $1/dcp\_size$ 的 KV，需访问其余 $(dcp\_size-1)/dcp\_size$
- 通信量随 $dcp\_size$ 增大而增加

**具体**：
- $dcp\_size = 1$：无通信
- $dcp\_size = 2$：$B \cdot T_s \cdot H \times 0.5$
- $dcp\_size = 4$：$B \cdot T_s \cdot H \times 0.75$
- $dcp\_size = 8$：$B \cdot T_s \cdot H \times 0.875$

**权衡**：减少 KV cache 重复与通信开销的 trade-off

---

## 二、单层总通信量

以 **TP + DP + PP + PCP + DCP + EP** 完整配置为例。

### Prefill 阶段（单层）

$$\text{Comm}_{prefill} = \text{TP\_comm} + \text{PP\_comm} + \text{PCP\_comm} + \text{EP\_comm}$$
$$\approx 2BT H + BT H + BT H_{kv} \times 2 + 2BT H$$
$$\approx 5BT H + 2BT H_{kv}$$

若 H_kv ≈ 0.1 ~ 0.2 H（如 GQA/MLA），则：
$$\text{Comm}_{prefill} \approx 5.2 \sim 5.4 \, BT H$$

### Decode 阶段（单层）

$$\text{Comm}_{decode} = \text{TP\_comm} + \text{PP\_comm} + \text{DCP\_comm}$$
$$\approx 2B H + B H + B \cdot T_s \cdot H \times \frac{dcp\_size - 1}{dcp\_size}$$
$$\approx 3B H + B \cdot T_s \cdot H \times (1 - 1/dcp\_size)$$

**Decode 瓶颈**：当 T_s ≫ 1 时，主要瓶颈是 KV cache 通信（DCP）

---

## 三、硬件拓扑与并行设计

### 3.1 单节点（8 张 H100/A100，NVLink）

**拓扑特征**：
- 卡间延迟：~1 µs
- 卡间带宽：~900 GB/s（NVLink 4）
- 无竞争（同机卡间可全双工）

**推荐配置**：
- **方案 1**（长序列）：`TP=8, PCP=1, DCP=1, DP=1, PP=1`
- **方案 2**（超长序列）：`TP=4, PCP=2, DCP=1, DP=1, PP=1`
- **方案 3**（MoE 长序列）：`TP=4, EP=2, PCP=1, DCP=1`

**设计原则**：
- TP、PCP、DCP 全在节点内（链路最快）
- 不需要跨节点通信，所有并行维度可充分并行

### 3.2 多节点（2~4 节点，IB/以太连接）

**拓扑特征**：
- 节点间延迟：~1~10 µs（IB Mellanox < 1µs，以太 ~10µs）
- 节点间带宽：~100 GB/s（IB EDR）或 ~12.5 GB/s（1GbE）
- 链路竞争（All-Reduce 会产生多路流量）

**推荐配置**：
- **方案 1**（模型并行为主）：`TP=8, PCP=1, DCP=1, PP=2, DP=1`
  - TP/PCP/DCP 在节点内
  - PP 跨节点（通信频率低）

- **方案 2**（DP 为主）：`TP=4, PCP=1, DCP=1, PP=1, DP=2`
  - TP/PCP/DCP 在节点内
  - DP 跨节点（仅推理无 DP）

- **方案 3**（长序列 + 多并行）：`TP=4, PCP=2, DCP=1, PP=1, DP=1`
  - 节点 1：TP=4 的 2 卡 + PCP 的一部分
  - 节点 2：TP=4 的另 2 卡 + PCP 的另一部分
  - **注意**：PCP 跨节点，需要验证通信开销是否可接受

**避免**：
- TP 跨节点（All-Reduce 会饱和链路）
- EP 跨节点（All-to-All 会产生大量流量）

### 3.3 超节点 / 多级拓扑（16~128 GPU，有机柜/跨机柜）

**拓扑特征**（例如 HPC 集群）：
- 机内（8 卡）：NVLink，~900 GB/s
- 机间（同机柜，~2~4 个节点）：IB QSFP，~200 GB/s
- 机柜间（跨机柜）：IB QSFP，~100 GB/s，延迟上升

**推荐分级**：
```
层级 1（最快，机内卡）：TP + DCP + 部分 EP
层级 2（快，机间）：PCP + 部分 EP
层级 3（慢，机柜间）：PP + DP（降频率）
```

**具体例子**（8×2=16 卡，两个机箱）：
```
机箱 1: TP=4, PCP=2, EP_part
机箱 2: TP=4, PCP=2, EP_part
跨机箱: PP=1, DP=1
```

配置：`TP=4, PCP=2, PP=2, DCP=1, DP=1`

**更大规模**（64 卡，4 个机柜）：
```
配置: TP=4, PCP=1, PP=4, DCP=1, DP=4
层级:
  - 机内：4×TP
  - 机间：1×PCP
  - 机柜间：4×PP（activation 通信）+ 4×DP（训练梯度）
```

### 3.4 GPUDirect RDMA 集群

**特殊优化**（如支持 GPUDirect RDMA）：
- DP All-Reduce 可利用 GPUDirect，带宽提升 30~50%
- PP 的 Send/Recv 可减少 host 中转开销

**影响**：
- DP/PP 可承受更大的跨节点通信
- 可考虑扩大 DP size（如 DP=4 跨节点）

---

## 四、通信与计算的 Trade-off

### 计算密度（Compute Intensity）

CI = FLOPs / 通信数据量（字节）

| 操作 | FLOP 数 | 通信数据 | CI |
|------|--------|---------|-----|
| Prefill QKV proj | 2BT(H × H) | BT·H | 2H |
| Prefill FFN | 4BT·H × H_inter | 2BT·H | 2H_inter |
| Decode QKV proj | 2B(H × H) | B·H | 2H |
| Attention (Prefill) | 2BT²·H | BT·H | 2T |
| Attention (Decode) | 2B·T_s·H | B·H | 2T_s |

**启示**：
- Prefill Attention 的 CI 很高（2T），通信隐藏容易
- Decode Attention 的 CI 低（2T_s 但仅处理 1 token），通信很难隐藏
- **Decode 是通信密集型**，需要优先优化

### 最大可用带宽（Roofline 模型）

T_compute = FLOPs / GPU_peak_FLOPS
T_communicate = 通信数据 / BW
T_actual = max(T_compute, T_communicate)

**例子**（H100 单卡，80 GB/s 显存带宽）：
- Prefill Attention：计算可利用，通信隐藏
- Decode：计算瓶颈很弱，通信完全限制

---

## 五、DCP 使用策略详解

### DCP 的收益与成本

| tp_size | H_kv | tp_size/H_kv | 建议 dcp_size | KV 重复倍数 | 通信开销 |
|---------|------|-------------|--------------|----------|--------|
| 4 | 32 | 0.125 | 1 | 4x | 无 |
| 8 | 1 | 8 | 1-4 | 8-2x | 低-中 |
| 8 | 4 | 2 | 1-2 | 2-1x | 低 |
| 16 | 1 | 16 | 4-8 | 4-2x | 中-高 |

### 最优 DCP 选择算法

1. **初始**：计算 `max_dcp = tp_size / H_kv`
2. **若 max_dcp < 2**：`dcp_size = 1`（KV 重复不严重，通信得不偿失）
3. **若 2 ≤ max_dcp < 4**：尝试 `dcp_size = 2`，验证性能
4. **若 max_dcp ≥ 4**：
   - 先用 `dcp_size = tp_size / H_kv` 完全消除重复
   - 如通信成为瓶颈，降低至 `dcp_size = 2 ~ 4`

### 实际案例

#### DeepSeek-R1 (MLA, H_kv=1)
```
单节点配置: TP=8, DCP=8
- KV 重复: 8x → 1x（完全消除）
- 通信: high，但在单节点 NVLink 内可承受
- 吞吐提升: 约 30~50%（KV 内存释放用于 KV cache 扩展）
```

#### Qwen3-235B (GQA, H_kv=4)
```
单节点配置: TP=8, DCP=4
- KV 重复: 2x → 1x
- 通信: 中等
- 多节点配置: TP=4(per node), DCP=2, PP=2
  - 机内: TP=4, DCP=2（NVLink）
  - 机间: PP=2（IB）
```

#### 长序列 + 长尾 (T_s > 10M tokens)
```
配置: TP=4, PCP=2, DCP=4, PP=1, DP=1 (16 GPU)
- Prefill: 利用 PCP 摊薄 TTFT
- Decode: DCP 减少 KV 占用，支持更长 context
```

---

## 六、性能预测与优化

### 6.1 Prefill 性能预测

$$T_{prefill} = \frac{\text{#layers} \times (2BT H + 2BT H_{kv})}{BW_{aggregate}}$$

**例子**（Qwen3-235B 141 层，TP=4，单节点）：
- $B=1, T=4096, H=7168, H_{kv} \approx 1000$
- 通信量：$141 \times (2 \times 1 \times 4096 \times 7168 + 2 \times 1 \times 4096 \times 1000) \approx 8.3 \times 10^9$ bytes
- 4 卡 TP 的总带宽：$4 \times 900 = 3600$ GB/s
- 估算时间：$\approx 2.3$ ms（实际会更长，因为有 bubbles 和同步开销）

### 6.2 Decode 性能预测

$$T_{decode\_per\_token} = \max\left(\frac{2H}{BW}, \frac{B \cdot T_s \cdot H}{BW} \times (1 - 1/dcp)\right)$$

**例子**（TP=4, DCP=2, T_s=100k）：
- TP 通信：$2BH / BW = 2 \times 1 \times 7168 / 3600 \approx 4$ µs
- DCP 通信：$1 \times 100000 \times 7168 \times 0.5 / 3600 \approx 100$ ms
  - **Decode 完全被 DCP 通信限制**

**优化**：降低 DCP 或提高 KV cache 访问带宽

---

## 七、配置决策树

```
START: 集群配置?
│
├─ 单节点 (1-8 GPU)
│  ├─ 长序列?
│  │  ├─ YES: TP=8, PCP=1, DCP=min(8, tp_size/H_kv)
│  │  └─ NO: TP=8, DCP=1
│  └─ MoE?
│     ├─ YES: TP=4, EP=2, DCP=1
│     └─ NO: TP=8, DCP=...
│
├─ 多节点 (2-4 nodes)
│  ├─ 以太网络?
│  │  └─ YES: TP in-node, PP=nodes, 避免 DP 跨节点
│  └─ IB 网络?
│     └─ YES: TP in-node, PCP/PP/DP 跨节点按需
│
├─ 超节点 (16-128 GPU)
│  ├─ 机内 8 GPU: TP 或 TP+DCP
│  ├─ 机间: PCP 或 PP
│  └─ 机柜间: PP 或 DP（降频率）
│
└─ HPC 集群 (128+ GPU)
   └─ 分层: TP(快) → PCP(中) → PP(慢) → DP(最慢)
```

---

## 八、实施步骤

1. **基准测试**
   - 测试单机吞吐（TP=8）
   - 测试跨节点通信延迟（ping-pong）

2. **初始配置**
   - 按拓扑选择默认配置
   - 优先做 TP（权重切分）

3. **瓶颈识别**
   - Profile 通信 vs 计算时间比例
   - 识别热路径（多数流量经过的链路）

4. **迭代优化**
   - 若通信受限，增加 DCP 或减少 DP size
   - 若计算受限，增加 batch size 或减少 PP

5. **验证**
   - 对比吞吐和 TTFT
   - 监控 GPU 利用率

---

## 参考：快速配置表

### 推理配置

| 集群规模 | 模型大小 | 序列长度 | 推荐配置 |
|----------|---------|---------|--------|
| 8 GPU | 70B | 4K | TP=8, DCP=1 |
| 8 GPU | 70B | 100K | TP=8, DCP=8 |
| 16 GPU (2x8) | 140B | 4K | TP=8(node), PP=2 |
| 16 GPU (2x8) | 140B | 100K | TP=4(node), DCP=2, PP=2, PCP=2 |
| 32 GPU (4x8) | 300B | 4K | TP=4, PP=2, DP=2 |
| 64 GPU (8x8) | 500B | 100K | TP=4, PCP=2, PP=2, DCP=2, DP=2 |

### 训练配置

| 集群规模 | 模型大小 | 推荐配置 | 说明 |
|----------|---------|--------|------|
| 8 GPU | 13B | TP=2, DP=4 | 小模型，数据并行为主 |
| 8 GPU | 70B | TP=8 | 中模型，单卡容纳困难 |
| 16 GPU | 70B | TP=4, DP=4 | 数据并行 + 张量并行 |
| 32 GPU | 140B | TP=8, DP=4 | 或 TP=4, PP=2, DP=4 |
| 128 GPU | 540B | TP=8, PP=4, DP=4 | MoE 可加 EP |

---

## 总结

1. **通信成本不对称**：
   - TP All-Reduce：相对可控
   - PCP 的 K/V gather：可流水化
   - **DCP 与 KV cache 规模正相关**，长序列时成为主瓶颈
   - EP 两次 All-to-All：与 token 总数相关，专家数不影响

2. **拓扑意识很关键**：
   - 单节点内用所有并行维度
   - 跨节点优先放 DP（最低频率）
   - 避免 TP/EP 跨节点

3. **DCP 不是必需的**：
   - 仅当 $tp\_size / H_{kv} \geq 2$ 且 KV 内存成为瓶颈时启用
   - 需权衡通信开销

4. **长序列优化方向**：
   - PCP 降低 TTFT
   - DCP 降低 KV 占用
   - 两者结合才能实现端到端的长序列支持
