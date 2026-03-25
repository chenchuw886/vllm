# vLLM Ascend 目录筛选与 Root-Cause 汇总报告

更新时间：2026-03-25

## 1. 执行目标

本报告按既定方案完成三阶段工作：

1. **Phase A：目录级筛选**
   - 先对 `tests/` 顶层目录做 A/B/C/D 分类，缩小需要深入分析的范围。
2. **Phase B：目录内代表性 testcase 运行与 root-cause 分析**
   - 只跑少量高价值、低成本样本，优先暴露第一个稳定且有语义的 failure。
3. **Phase C：全量汇总输出**
   - 从 Ascend CI 准入视角，给出 `presubmit / nightly / manual / reject` 决策。

## 2. 分析环境与最小 mitigation

### 环境

- `vllm`: branch `v0150rc1`
- `vllm-ascend`: branch `v0.15.0rc1`
- Python: `3.11.14`
- torch: `2.9.0+cpu`
- torch-npu: `2.9.0`
- CANN: `8.5.0`
- 当前环境并非全新环境，存在部分预缓存

### 本次实际使用的最小 mitigation

- 网络代理：
  - `http_proxy=socks5h://127.0.0.1:1080`
  - `https_proxy=socks5h://127.0.0.1:1080`
  - `all_proxy=socks5h://127.0.0.1:1080`
- 模型源开关：
  - `VLLM_USE_MODELSCOPE=True`
- 为推进 `plugins_tests` 代表 case，安装了本地测试插件包：
  - `tests/plugins/vllm_add_dummy_platform`
  - `tests/plugins/vllm_add_dummy_stat_logger`

说明：这些 mitigation 只用于去掉环境噪音并推进到更深的代码路径；未修改产品代码。

## 3. Phase A：目录级筛选结果

目录筛选标准不是“有没有测试”，而是：

> 该目录中的 testcase，失败后能否高概率说明 `vllm-ascend` 在保持 upstream 行为契约时发生了真实回归？

### 3.1 目录级总览表

| 目录 | 分类 | 主要行为契约 | Ascend 相关性 | 初步建议 | 原因 |
|---|---|---|---|---|---|
| `tests/config` | A | config normalization / model arch config | 强 | 首批分析 | 直接覆盖配置解析、平台兼容与默认值契约 |
| `tests/compile` | A | compile wrapper / pass manager / fallback | 强 | 首批分析 | 是 Ascend 编译/图模式适配边界 |
| `tests/plugins_tests` | A | platform plugin / stat logger / scheduler plugin | 强 | 首批分析 | 直接覆盖插件发现、过滤、注册与运行时装配 |
| `tests/v1` | A | engine args / metrics / kv connector / entrypoints | 强 | 首批分析 | 新架构路径与 Ascend 适配强相关 |
| `tests/entrypoints` | A | OpenAI-compatible API / request validation / serving | 强 | 首批分析 | 高价值行为契约层，适合构造 CI 子集 |
| `tests/lora` | A | adapter loading / module mapping / runtime update | 强 | 首批分析 | LoRA 是 Ascend 高频回归边界 |
| `tests/multimodal` | A | multimodal validation / sparse tensor safety | 强 | 首批分析 | 能覆盖 Ascend 多模态输入与安全校验路径 |
| `tests/models` | B | model init / registry / pooling / multimodal model path | 中-强 | 分批分析 | 价值高，但较多 case 依赖模型资产 |
| `tests/quantization` | B | quantization config / compressed tensors / feature gating | 强 | 分批分析 | 对 Ascend 很重要，但常伴随模型元数据/平台差异 |
| `tests/distributed` | B | SP / OOT / distributed serving | 强 | Nightly 优先 | 信号强，但资源与稳定性要求较高 |
| `tests/model_executor` | B | loader / custom ops / executor path | 中 | 分批分析 | 能打到适配边界，但部分子目录 backend 相关较重 |
| `tests/engine` | B | engine-level semantics | 中 | 分批分析 | 有价值，但与 `tests/v1`、`tests/entrypoints` 有重叠 |
| `tests/pooling` | B | pooling task contracts | 中 | 分批分析 | 部分在 `entrypoints/pooling` 中更高信号 |
| `tests/samplers` | B | public decoding / logprob semantics | 中 | 作为补充 | 行为契约明确，通常成本较低 |
| `tests/reasoning` | B | parser / reasoning contract | 中 | 作为补充 | 可能形成高信号轻量守护 |
| `tests/tool_parsers` | B | parser contract | 中 | 作为补充 | 一般轻量，但 Ascend 特异性略弱 |
| `tests/tool_use` | B | tool execution / protocol path | 中 | 作为补充 | 更偏 API 契约，可纳入后续补充 |
| `tests/renderers` | B | renderer output contract | 中 | 作为补充 | 已有较多可执行 case，可做低成本补充 |
| `tests/basic_correctness` | B | broad functional correctness | 中 | 延后 | 常需模型启动，适合有预缓存时补充 |
| `tests/standalone_tests` | C | 独立集成行为 | 中-弱 | 延后 | 信号可能有用，但通常启动成本更高 |
| `tests/evals` | C | benchmark / eval correctness | 中-弱 | 降权 | 强依赖外部模型、数据集或命令行参数 |
| `tests/benchmarks` | C | benchmark/perf behavior | 弱 | 默认降权 | 主要不是 correctness 守护 |
| `tests/kernels` | D | CUDA/Triton/ROCm kernel implementation | 弱 | 默认排除 | 大量 case 明显为 CUDA/HIP/Triton 专用 |
| `tests/cuda` | D | CUDA-specific behavior | 弱 | 默认排除 | 非 Ascend 平台契约 |
| `tests/rocm` | D | ROCm-specific behavior | 弱 | 默认排除 | 与 Ascend 不适用 |
| `tests/detokenizer` | B | tokenizer/detokenizer contract | 中 | 作为补充 | 轻量、稳定，但 Ascend 特异性一般 |
| `tests/prompts` | C | prompt assets / behavior | 弱-中 | 延后 | 更偏上层输入样例 |
| `tests/tokenizers_` | B | tokenizer integration | 中 | 作为补充 | 轻量、可移植 |
| `tests/transformers_utils` | B | upstream model utils integration | 中 | 作为补充 | 常与模型元数据解析相关 |
| `tests/utils_` | B | utility semantics | 中 | 作为补充 | 可作为低成本守护 |
| `tests/weight_loading` | B | weight loading path | 中-强 | 分批分析 | 与模型初始化、格式兼容强相关 |
| 根目录 `tests/test_*.py` | B | config / outputs / logger / envs 等横切契约 | 中 | 作为补充 | 低成本高覆盖，可挑轻量 case |

### 3.2 当前阶段性结论

- **A 类优先目录**：`config`、`compile`、`plugins_tests`、`v1`、`entrypoints`、`lora`、`multimodal`
- **B 类补充目录**：`models`、`quantization`、`distributed`、`model_executor`、`samplers`、`renderers`、`utils_`
- **C/D 类降权或排除目录**：`evals`、`benchmarks`、`kernels`、`cuda`、`rocm`

这与已有文件级分析结论基本一致：真正值得纳入 Ascend CI 的，不应主要来自 CUDA/Triton kernel 原样测试，而应优先来自行为契约层与适配边界层。

## 4. Phase B：目录内代表 testcase 运行与 root cause

本阶段只选择了少量代表性样本，目的不是“全跑”，而是暴露稳定且有语义的第一个 failure。

### 4.1 本次新增动态验证样本

| 目录 | testcase | 运行方式 | 观察到的 failure / 结果 | true root cause | failure 分类 | ownership | CI 建议 |
|---|---|---|---|---|---|---|---|
| `tests/plugins_tests` | `test_platform_plugins.py` | 安装本地 dummy 插件包后运行 | `RuntimeError: Only one platform plugin can be activated, but got: ['dummy_platform_plugin', 'ascend']` | Ascend 环境默认已有平台插件激活；当测试再安装 dummy platform 插件时，upstream 平台选择逻辑要求只能激活一个 platform plugin | `vllm` / `vllm-ascend` integration boundary | `vllm` + `vllm-ascend` | `nightly`（需隔离平台插件环境或设计 Ascend-adapted variant） |
| `tests/plugins_tests` | `test_stats_logger_plugins.py::{3 cases}` | 安装本地 dummy stat logger 包，并设置 `VLLM_PLUGINS=ascend` | `3 passed` | 非平台类插件发现逻辑在 Ascend 上可正常运行；关键前置条件是 import 阶段锁定 platform plugin 过滤范围 | 可执行 | test / environment | `presubmit`（作为轻量插件契约守护） |
| `tests/v1/engine` | `test_engine_args.py::test_prefix_caching_from_cli` / `::test_prefix_caching_xxhash_from_cli` | 代理 + ModelScope | 通过 | CLI 参数与 cache config 契约可正常保留 | 可执行 | `vllm` | `presubmit` |
| `tests/v1/engine` | `test_engine_args.py::test_defaults_with_usage_context` | 代理 + ModelScope | `NotImplementedError` from `current_platform.get_device_total_memory()` | `NPUPlatform` 未实现 `get_device_total_memory()`，导致基于平台总显存的默认调度参数计算路径在 Ascend 上中断 | `vllm-ascend` adaptation defect | `vllm-ascend` | `presubmit`（P0，需优先补强） |
| `tests/quantization` | `test_configs.py::test_auto_gptq` | 代理 + ModelScope | `6 failed, 6 passed`；GPTQ/AWQ 期望类型被归为 `ERROR` | 对多种 GPTQ/AWQ 模型，`ModelConfig` 在 Ascend/NPU 路径上触发 `ValueError` 或不支持分支，导致期望的量化类型解析失败；更深层并非网络，而是量化支持/自动检测与平台能力不匹配 | compiler/runtime compatibility + feature gap | `vllm-ascend` + runtime stack | `nightly`（适合做量化支持边界守护） |

### 4.2 与既有文件级结论的回填

结合已有的 [ASCEND_FIRST_30_TEST_ANALYSIS.md](ASCEND_FIRST_30_TEST_ANALYSIS.md) 与 [ASCEND_TEST_ANALYSIS_REPORT.md](ASCEND_TEST_ANALYSIS_REPORT.md)，当前可以把目录内信号回填成以下判断：

- `tests/compile`
  - 已有代表 failure：`test_wrapper.py` 暴露 Ascend compile config 更新路径问题
  - 目录定位：**A 类 / P0**
- `tests/entrypoints`
  - 已有代表 failure：
    - `test_sparse_tensor_validation.py`：异常类型假设与当前运行时行为不匹配
    - `test_mm_cache_stats.py`：加 `no_proxy` 后可稳定通过
    - `test_vision.py` / `test_chat.py`：主阻塞为大模型准备成本
  - 目录定位：**A 类**，但需拆轻重子集
- `tests/distributed`
  - 已有代表 failure：
    - `test_sequence_parallel.py`：去掉网络阻塞后推进到 Ascend worker 设备空闲显存阈值失败
    - `test_distributed_oot.py`：OOT 注册/识别链失败
  - 目录定位：**B 类偏高优先**，适合 nightly
- `tests/kernels`
  - 已有大量日志与文件级结论表明：多数 case 为 CUDA/Triton/ROCm 专用或平台跳过
  - 目录定位：**D 类默认排除**

## 5. Phase C：CI 准入建议

### 5.1 Directory → CI disposition 映射

| 目录 | 建议主 disposition | 说明 |
|---|---|---|
| `tests/compile` | Presubmit | 保护 compile wrapper / fallback / config 边界 |
| `tests/plugins_tests` | Presubmit + Nightly | 轻量 stat/logger plugin 可进 presubmit；platform plugin 冲突类样本更适合 nightly |
| `tests/v1` | Presubmit | `engine_args` 已暴露高信号平台接口缺口 |
| `tests/entrypoints` | Presubmit + Nightly | request validation / schema / sparse tensor 进 presubmit；vision/chat/mm-cache 等重资源子集进 nightly |
| `tests/multimodal` | Presubmit | 轻量 validation / safety 子集优先 |
| `tests/lora` | Nightly | 价值高，但常需模型与运行时准备 |
| `tests/distributed` | Nightly | SP / OOT 高信号但成本较高 |
| `tests/quantization` | Nightly | 适合作为平台能力/feature gap 守护 |
| `tests/models` | Nightly / Manual | 需按轻重拆分 |
| `tests/evals` | Manual / Reject | 依赖评测参数与外部资产，信噪比较低 |
| `tests/kernels` | Reject | 原样纳入对 Ascend 价值低 |

### 5.2 建议纳入 Presubmit CI 的首批 testcase portfolio

1. `tests/compile/test_wrapper.py`
2. `tests/entrypoints/openai/test_sparse_tensor_validation.py`
3. `tests/v1/engine/test_engine_args.py`
   - 至少保留 `test_prefix_caching_from_cli`
   - 新发现 `test_defaults_with_usage_context` 是 P0 平台接口缺口
4. `tests/plugins_tests/test_stats_logger_plugins.py`
5. `tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py`
6. `tests/multimodal/test_sparse_tensor_validation_unit.py`
7. `tests/entrypoints/openai/test_embedding_shape_validation.py`

### 5.3 建议纳入 Nightly CI 的 testcase portfolio

1. `tests/distributed/test_sequence_parallel.py`
2. `tests/distributed/test_distributed_oot.py`
3. `tests/entrypoints/openai/test_oot_registration.py`
4. `tests/entrypoints/llm/test_mm_cache_stats.py`
5. `tests/entrypoints/openai/test_vision.py`
6. `tests/entrypoints/pooling/classify/test_online.py`
7. `tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py`
8. `tests/quantization/test_configs.py`
9. `tests/lora/test_qwenvl.py`

### 5.4 保留为 Manual Regression

- `tests/compile/test_pass_manager.py`
- `tests/entrypoints/openai/test_chat.py`
- `tests/entrypoints/openai/test_default_mm_loras.py`
- `tests/entrypoints/openai/test_transcription_validation_whisper.py`
- `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`
- `tests/entrypoints/pooling/classify/test_online_vision.py`
- `tests/entrypoints/pooling/score/test_correctness_mteb.py`
- `tests/plugins_tests/test_platform_plugins.py`（在隔离环境或 Ascend-adapted variant 完成前）

### 5.5 Reject as Noise / Non-Ascend-Relevant

- `tests/kernels/attention/test_attention_selector.py`
- `tests/kernels/attention/test_flashmla.py`
- `tests/kernels/attention/test_flashmla_sparse.py`
- `tests/kernels/attention/test_mha_attn.py`
- `tests/kernels/core/test_activation.py`
- `tests/kernels/moe/test_gpt_oss_triton_kernels.py`
- `tests/kernels/moe/test_modular_kernel_combinations.py`
- `tests/kernels/moe/test_modular_oai_triton_moe.py`
- `tests/entrypoints/openai/test_openai_schema.py`
- `tests/entrypoints/openai/test_video.py`
- `tests/evals/gpt_oss/test_gpqa_correctness.py`

## 6. 共性问题总结

### 高频 root cause 类型

1. **平台接口未补齐**
   - 代表：`NPUPlatform.get_device_total_memory()` 未实现
   - 影响：直接阻断默认配置/调度参数推导路径

2. **平台插件装配冲突**
   - 代表：`test_platform_plugins.py` 中 dummy platform plugin 与 ascend plugin 同时可见
   - 影响：这类问题不是网络噪音，而是插件过滤/隔离语义边界

3. **量化 feature gap 或自动检测路径不兼容**
   - 代表：`tests/quantization/test_configs.py`
   - 影响：对 GPTQ/AWQ 等模型的自动解析与平台支持矩阵不一致

4. **重资源前置条件**
   - 代表：`test_chat.py`、`test_vision.py`、Whisper 相关测试
   - 影响：适合 nightly/manual，不适合 presubmit

5. **测试环境前置条件缺失**
   - 代表：plugin dummy 包、`no_proxy`、`mteb`、`rapidfuzz`、pytest `--model`
   - 影响：这类问题不应误判成产品缺陷，但应写入 testcase 运行方式

## 7. 一句话结论

当前方案是有效的：

- **目录级筛选** 已经把高价值范围稳定收敛到 `compile / plugins_tests / v1 / entrypoints / lora / multimodal`；
- **目录内代表运行** 已经暴露出真实 Ascend 适配缺口，例如 `get_device_total_memory()` 未实现、platform plugin 冲突、quantization 配置不兼容；
- **CI portfolio** 可以先从轻量高信号 presubmit 子集入手，再把 distributed / LoRA / vision / quantization 放到 nightly。
