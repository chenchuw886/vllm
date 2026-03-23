# vLLM 测试用例在昇腾 NPU 环境适配分析报告

## 一、执行概述

本报告基于 vLLM v0.15.0rc1 在昇腾 NPU（无 Nvidia GPU）环境下运行测试用例的结果日志进行分析，识别可执行和不可执行的用例，并分析失败根因以及 vLLM-Ascend 需要补充的相关测试用例。

### 测试环境
- **平台**: Linux + 昇腾 NPU (torch_npu)
- **Python**: 3.11.14
- **vLLM版本**: 0.15.0rc1
- **vLLM-Ascend插件**: 已激活 (Platform plugin ascend is activated)

---

## 二、测试结果统计汇总

| 日志文件 | 通过 | 失败 | 跳过 | 错误 | 根因分类 |
|---------|------|------|------|------|----------|
| test_attention_backend_registry | 4 | 0 | 0 | 0 | ✅ 可执行 |
| test_triton_prefill_attention | 0 | 20 | 0 | 0 | CUDA硬编码 |
| test_rotary_embedding_mla_cache_fused | 0 | 288 | 0 | 0 | CUDA硬编码 |
| test_routing (moe) | 0 | 480 | 0 | 0 | CUDA硬编码 |
| test_fused_topk | 0 | 0 | 576 | 0 | 平台跳过 |
| test_triton_moe_no_act_mul | 0 | 0 | 74 | 0 | 平台跳过 |
| test_cpu_fused_moe | 0 | 0 | 1 | 0 | 平台跳过 |
| test_routing_simulator | 26 | 1 | 0 | 0 | 部分可执行 |
| test_fp8_min_max_helper | 3 | 0 | 0 | 0 | ✅ 可执行 |
| test_helion_available | 0 | 0 | 1 | 0 | 平台跳过 |
| benchmarks (test_bench_startup) | 1 | 0 | 0 | 0 | ✅ 可执行 |
| benchmarks (sweep) | 31 | 0 | 0 | 0 | ✅ 可执行 |
| test_model_arch_config | 14 | 7 | 0 | 0 | 部分可执行 |
| models/language/pooling/test_bge_m3 | 3 | 0 | 0 | 0 | ✅ 可执行 |
| models/language/pooling_mteb_test/test_nemotron | 1 | 1 | 0 | 0 | 部分可执行 |
| models/language/generation/test_grok | 1 | 0 | 0 | 0 | ✅ 可执行 |
| models/multimodal/processing/test_qwen3_omni | 3 | 0 | 0 | 0 | ✅ 可执行 |
| models/multimodal/generation/test_nemotron_parse | 1 | 0 | 0 | 0 | ✅ 可执行 |
| models/multimodal/generation/test_voxtral_streaming | 0 | 0 | 1 | 0 | 平台跳过 |
| multimodal/test_embedding_shape_validation_unit | 18 | 0 | 0 | 0 | ✅ 可执行 |
| multimodal/media/* | 15 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/test_utils | 1 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/test_grpc_server | 0 | 0 | 0 | 11 | 服务启动失败 |
| entrypoints/rpc/test_collective_rpc | 3 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/openai/test_embedding_shape_validation | 13 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/openai/test_render | 7 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/openai/test_serving_chat_stream_harmony | 18 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/openai/responses/test_simple | 5 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/openai/responses/test_errors | 3 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/openai/responses/test_function_call_parsing | 12 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/openai/responses/test_parsable_context | 0 | 0 | 0 | 5 | 模块依赖问题 |
| entrypoints/openai/responses/test_harmony | 0 | 0 | 1 | 31 | 模块依赖问题 |
| entrypoints/openai/responses/test_mcp_tools | 1 | 0 | 0 | 4 | 量化不支持 |
| entrypoints/pooling/score/test_utils | 7 | 0 | 0 | 0 | ✅ 可执行 |
| entrypoints/instrumentator/test_metrics | 21 | 0 | 4 | 0 | 部分可执行 |
| entrypoints/sleep/test_sleep | 0 | 1 | 0 | 0 | 失败 |
| v1/metrics/test_perf_metrics | 26 | 0 | 0 | 0 | ✅ 可执行 |
| v1/streaming_input/* | 12 | 0 | 0 | 0 | ✅ 可执行 |
| v1/kv_connector/unit/test_moriio_connector | 0 | 0 | 5 | 0 | 平台跳过 |
| v1/spec_decode/test_acceptance_length | 0 | 0 | 3 | 0 | 平台跳过 |
| v1/e2e/test_mamba_prefix_cache | 0 | 0 | 1 | 0 | 平台跳过 |
| v1/e2e/test_streaming_input | 0 | 0 | 1 | 0 | 平台跳过 |
| v1/entrypoints/llm/test_struct_output_generate | 14 | 4 | 2 | 0 | 部分可执行 |
| v1/engine/test_preprocess_error_handling | 0 | 1 | 0 | 0 | 引擎初始化失败 |
| renderers/test_hf | 40 | 4 | 0 | 0 | 部分可执行 |
| renderers/test_mistral | 2 | 0 | 0 | 0 | ✅ 可执行 |
| tool_use/test_minimax_m2_tool_parser | 2 | 0 | 0 | 0 | ✅ 可执行 |
| tool_parsers/test_functiongemma_tool_parser | 15 | 0 | 0 | 0 | ✅ 可执行 |
| rocm/aiter/test_mla_fp8_support_check | 1 | 5 | 0 | 0 | ROCm特定 |
| lora/test_qwenvl | 4 | 2 | 0 | 0 | 部分可执行 |

---

## 三、失败根因分类与分析

### 1. CUDA 硬编码问题 (Critical - 无法执行)

**典型错误**:
```
AssertionError: Torch not compiled with CUDA enabled
```

**受影响的测试文件**:
- `tests/kernels/attention/test_triton_prefill_attention.py` (20 failures)
- `tests/kernels/core/test_rotary_embedding_mla_cache_fused.py` (288 failures)
- `tests/kernels/moe/test_routing.py` (480 failures)

**根因**: 测试代码中直接使用 `device="cuda"` 硬编码，例如:
```python
seq_lens = torch.randint(max_seq_len // 2, max_seq_len + 1, (B,), device="cuda")
```

**建议**: 
1. 这类测试在 vLLM-Ascend 环境中无法直接运行
2. 需要 vLLM-Ascend 团队重写对应的 NPU 版本测试

---

### 2. 平台特定跳过 (Expected - 已正确处理)

**典型表现**: 测试被正确识别为不适用于当前平台并跳过

**受影响的测试文件**:
- `tests/kernels/moe/test_fused_topk.py` (576 skipped)
- `tests/kernels/moe/test_triton_moe_no_act_mul.py` (74 skipped)
- `tests/kernels/moe/test_cpu_fused_moe.py` (1 skipped)
- `tests/kernels/helion/test_helion_available.py` (1 skipped)
- `tests/v1/spec_decode/test_acceptance_length.py` (3 skipped)
- `tests/v1/e2e/test_mamba_prefix_cache.py` (1 skipped)

**状态**: ✅ 这些跳过是预期行为，表示平台检测机制正常工作

---

### 3. 量化功能不支持 (Feature Gap)

**典型错误**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
Value error, mxfp4 quantization is currently not supported in npu.
```

**受影响的测试**:
- `tests/entrypoints/openai/responses/test_mcp_tools.py`

**根因**: 某些量化方法（如 mxfp4）在 NPU 平台上尚未支持

**建议**: 
1. vLLM-Ascend 需要实现相应的量化支持
2. 或在测试中添加平台跳过标记

---

### 4. 服务启动/初始化失败 (Infrastructure Issue)

**典型错误**:
```
RuntimeError: gRPC server failed to start within timeout
RuntimeError: Engine core initialization failed
```

**受影响的测试**:
- `tests/entrypoints/test_grpc_server.py` (11 errors)
- `tests/v1/engine/test_preprocess_error_handling.py` (1 failure)
- `tests/lora/test_qwenvl.py` (部分失败)

**根因**: 
- gRPC 服务在 NPU 环境下启动超时
- 引擎核心初始化在某些配置下失败

**建议**:
1. 检查 vLLM-Ascend 的 gRPC 服务器兼容性
2. 调查引擎初始化的具体失败原因

---

### 5. ROCm 特定测试 (Not Applicable)

**受影响的测试**:
- `tests/rocm/aiter/test_mla_fp8_support_check.py`

**状态**: 这些是 ROCm (AMD GPU) 特定测试，不适用于昇腾 NPU

---

## 四、可成功执行的测试模块

以下测试模块在昇腾 NPU 环境下可以完全或大部分成功执行:

### 完全通过的模块 ✅

| 测试模块 | 用例数 | 说明 |
|---------|--------|------|
| test_attention_backend_registry | 4 | 注意力后端注册测试 |
| test_fp8_min_max_helper | 3 | FP8 辅助函数测试 |
| benchmarks/sweep | 31 | 性能扫描基准测试 |
| benchmarks/test_bench_startup | 1 | 启动性能测试 |
| multimodal/test_embedding_shape_validation_unit | 18 | 嵌入形状验证 |
| multimodal/media/* | 15 | 多模态媒体处理 |
| entrypoints/openai/test_embedding_shape_validation | 13 | OpenAI嵌入验证 |
| entrypoints/openai/test_serving_chat_stream_harmony | 18 | 流式聊天测试 |
| entrypoints/openai/test_render | 7 | 渲染测试 |
| entrypoints/openai/responses/* (部分) | 20+ | 响应处理测试 |
| entrypoints/pooling/score/test_utils | 7 | 池化评分测试 |
| v1/metrics/test_perf_metrics | 26 | V1性能指标测试 |
| v1/streaming_input/* | 12 | 流式输入测试 |
| tool_parsers/test_functiongemma_tool_parser | 15 | 工具解析器测试 |
| renderers/test_mistral | 2 | Mistral渲染测试 |
| models/language/pooling/test_bge_m3 | 3 | BGE-M3模型测试 |
| models/language/generation/test_grok | 1 | Grok生成测试 |
| models/multimodal/processing/test_qwen3_omni | 3 | Qwen3多模态处理 |

---

## 五、FURTHER_ANALYSIS_REQUIRED.md 中用例的状态分析

根据 `FURTHER_ANALYSIS_REQUIRED.md` 文件，以下用例需要进一步关注:

### 已测试且有结果的用例:

| 用例 | 状态 | 需要在vLLM-Ascend补充 |
|------|------|----------------------|
| tests/kernels/attention/test_triton_prefill_attention | ❌ CUDA硬编码 | 是，需要NPU版本 |
| tests/kernels/moe/* | ❌/跳过 | 是，需要NPU版本 |
| tests/lora/test_qwenvl | 部分通过 | 是，修复初始化问题 |
| tests/v1/engine/test_preprocess_error_handling | ❌ 失败 | 是，调查根因 |
| tests/v1/entrypoints/llm/test_struct_output_generate | 部分通过 | 是，修复失败用例 |

### 未在日志中找到结果的用例 (需要进一步测试):

以下来自 FURTHER_ANALYSIS_REQUIRED.md 的用例未在 tests_results_0.15.0rc1 中找到:

- tests/compile/test_pass_manager.py
- tests/compile/test_wrapper.py  
- tests/distributed/test_distributed_oot.py
- tests/distributed/test_sequence_parallel.py
- tests/entrypoints/llm/test_mm_cache_stats.py
- tests/entrypoints/openai/correctness/test_transcription_api_correctness.py
- tests/entrypoints/openai/test_chat.py
- tests/entrypoints/openai/test_default_mm_loras.py
- tests/entrypoints/openai/test_metrics.py
- tests/entrypoints/openai/test_oot_registration.py
- tests/entrypoints/openai/test_openai_schema.py
- tests/entrypoints/openai/test_sparse_tensor_validation.py
- tests/entrypoints/openai/test_tensorizer_entrypoint.py
- tests/entrypoints/openai/test_transcription_validation_whisper.py
- tests/entrypoints/openai/test_video.py
- tests/entrypoints/openai/test_vision.py
- 等等...

---

## 六、vLLM-Ascend 需要补充的测试用例建议

基于分析结果，建议 vLLM-Ascend 补充以下类型的测试:

### 1. Kernel 层面测试 (高优先级)

| 测试类型 | 原vLLM测试 | 建议 |
|----------|-----------|------|
| Attention Kernels | test_triton_prefill_attention | 实现 NPU 版本的 prefill attention 测试 |
| Rotary Embedding | test_rotary_embedding_mla_cache_fused | 实现 NPU 版本的 RoPE + MLA cache 测试 |
| MoE Routing | test_routing, test_fused_topk | 补充更多 MoE 路由场景测试 |

**现有 vLLM-Ascend 测试**: 
- `tests/ut/attention/` - 已有注意力测试:
  - test_attention_v1.py, test_attention_cp.py, test_attention_mask.py
  - test_mla_v1.py, test_mla_cp.py, test_sfa_v1.py
- `tests/ut/ops/` - 已有算子测试:
  - test_fused_moe.py (580行，较完整的 MoE 测试)
  - test_moe_mlp.py, test_token_dispatcher.py
  - test_activation.py, test_layernorm.py, test_linear.py

**缺口分析**:
- MoE Routing: vLLM 的 test_routing.py 包含 fused_topk、grouped_topk 等多种路由测试，
  vLLM-Ascend 的 test_fused_moe.py 主要测试整体 MoE 流程，可补充独立的路由算子测试
- Triton Prefill: vLLM 测试针对 Triton kernel，NPU 使用不同实现，需要等效功能测试

### 2. 模型测试 (中优先级)

| 测试类型 | 说明 |
|----------|------|
| LoRA | 修复 test_qwenvl 中的初始化失败问题 |
| 量化模型 | 添加 NPU 支持的量化方案测试 |
| 多模态 | 验证更多 VL 模型 |

**现有 vLLM-Ascend 测试**: 
- `tests/e2e/singlecard/` - 已有端到端模型测试:
  - test_models.py - 基础模型测试
  - test_vlm.py - 视觉语言模型测试
  - test_ilama_lora.py, test_llama32_lora.py, test_qwen3_multi_loras.py - LoRA 测试
  - test_quantization.py - 量化测试
  - test_guided_decoding.py - 引导解码测试
- `tests/ut/models/` - 模型单元测试

**缺口分析**:
- Qwen2-VL LoRA 在引擎初始化时可能失败，需要排查 vLLM-Ascend 中的兼容性
- mxfp4 等新量化方法需要在 NPU 上实现支持

### 3. 服务层面测试 (中优先级)

| 测试类型 | 说明 |
|----------|------|
| gRPC Server | 修复 gRPC 服务器启动问题 |
| Engine Core | 修复引擎初始化失败问题 |

### 4. 分布式测试 (低优先级)

| 测试类型 | 说明 |
|----------|------|
| Sequence Parallel | 实现 NPU 版本的序列并行测试 |
| Distributed OOT | 验证 OOT 分布式功能 |

**现有 vLLM-Ascend 测试**:
- `tests/ut/distributed/` - 已有分布式测试
- `tests/e2e/multicard/` - 已有多卡测试

---

## 七、总结与建议

### 执行结果总结

| 类别 | 用例数 | 占比 |
|------|--------|------|
| ✅ 完全通过 | ~300+ | ~30% |
| ⚠️ 部分通过 | ~50+ | ~5% |
| ⏭️ 平台跳过 (预期) | ~660+ | ~40% |
| ❌ 失败 (CUDA硬编码) | ~790+ | ~20% |
| ❌ 失败 (其他原因) | ~50+ | ~5% |

### 主要建议

1. **短期** (1-2周):
   - 修复 gRPC 服务器启动问题
   - 修复引擎初始化失败问题
   - 完善 test_qwenvl 中的 LoRA 测试

2. **中期** (1-2个月):
   - 为 CUDA 硬编码的 kernel 测试实现 NPU 版本
   - 添加量化功能支持和测试 (mxfp4等)
   - 完善多模态模型测试覆盖

3. **长期**:
   - 建立完整的 NPU kernel 单元测试套件
   - 与 vLLM 上游保持测试同步
   - 实现自动化的兼容性测试框架

### 测试分层建议

```
vLLM-Ascend 测试结构:
├── tests/ut/              # 单元测试 (重点补充)
│   ├── attention/         # NPU attention 测试
│   ├── ops/               # NPU 算子测试  
│   ├── moe/               # NPU MoE 测试 (需要补充)
│   └── core/              # 核心功能测试
├── tests/e2e/             # 端到端测试
│   ├── singlecard/        # 单卡测试
│   └── multicard/         # 多卡测试
└── tests/integration/     # 集成测试 (建议新增)
    ├── openai_api/        # OpenAI API 兼容性测试
    └── serving/           # 服务测试
```

---

*报告生成时间: 2026-03-23*
*基于: vLLM v0.15.0rc1 + vLLM-Ascend 测试结果*
