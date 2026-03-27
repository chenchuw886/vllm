# Ascend 62-128 Test Analysis

## Scope

- This file covers items 62-128 from [TEST_FILES_NEED_ANALYSIS.md](TEST_FILES_NEED_ANALYSIS.md).
- Items 1-61 are already analyzed in [ASCEND_FIRST_30_TEST_ANALYSIS.md](ASCEND_FIRST_30_TEST_ANALYSIS.md) and [ASCEND_31_61_TEST_ANALYSIS.md](ASCEND_31_61_TEST_ANALYSIS.md).

## Environment

- `vllm` commit: `c8903ea364a6f665a9d878a6a6fb941801088b1f`
- `vllm-ascend` commit: `3d43ed997e715981321bffc10f72f11d9522a5ae`
- Python: `3.11.14`
- torch: `2.9.0+cpu`
- torch-npu: `2.9.0`
- CANN: `8.5.0`
- Environment state: not fully clean; some analyses reused pre-cached models and existing Python dependencies
- Network: allowed via configured proxy; ModelScope was used first when possible, then Hugging Face fallback when needed

## Analysis Notes

- No new Python packages were installed in this pass.
- Workarounds used in this pass:
  - sourced Ascend toolkit env
  - set proxy env vars for external model access
  - retried `tests/v1/engine/test_preprocess_error_handling.py` with Hugging Face fallback
  - reran `tests/entrypoints/test_grpc_server.py` to expose the real collection error
- File existence check: all items 62-128 exist in the checked-out branch.

## Results

| # | Testcase | 是否能跑通 | 是否适合上 vLLM-Ascend CI | 失败根因 / 说明 |
|---|---|---|---|---|
| 62 | `tests/models/multimodal/generation/test_audioflamingo3.py` | 否（全 skip） | 否 | 日志为 `2 skipped`，未真正执行 Ascend 有效路径，信号过低。 |
| 63 | `tests/models/multimodal/generation/test_granite_speech.py` | 否 | 是（nightly） | `NPUModelRunner.profile_run` 走到 `gpu_model_runner.py` 的多模态预算逻辑时，`NPUModelRunner` 缺少 `mm_budget` 属性；属于 `vllm-ascend` 多模态适配缺口。 |
| 64 | `tests/models/multimodal/generation/test_phi4mm.py` | 否 | 否 | 多模态 LoRA 激活阶段在 `column_parallel_linear.set_lora` 触发 `IndexError: tuple index out of range`；更像上游模型/LoRA 形状假设问题，当前不宜作为 Ascend CI 准入。 |
| 65 | `tests/models/multimodal/generation/test_qwen2_5_vl.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnIndexPutImpl failed`。 |
| 66 | `tests/models/multimodal/generation/test_vit_backend_functionality.py` | 否（全 skip） | 否 | `16 skipped`，无有效 Ascend 看护信号。 |
| 67 | `tests/models/multimodal/generation/test_voxtral.py` | 否 | 否 | 模型处理器调用签名不匹配：`AudioEncoder.pad() missing 1 required positional argument: 'is_online_streaming'`；偏上游模型/processor 集成问题。 |
| 68 | `tests/models/multimodal/pooling/test_clip.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`ReshapeCacheOperation`。 |
| 69 | `tests/models/multimodal/pooling/test_jinavl_reranker.py` | 部分 | 是（nightly） | 3 个 case 通过、1 个 case 失败；失败为打分行为偏差，断言值 `0.9922 != 0.9094±tol`，对 Ascend 行为一致性有看护价值。 |
| 70 | `tests/models/multimodal/pooling/test_prithvi_mae.py` | 否 | 否 | `ModelConfig` 校验失败；当前更像模型/架构支持缺失，不是高价值 Ascend CI 守护点。 |
| 71 | `tests/models/multimodal/processing/test_audioflamingo3.py` | 否（全 skip） | 否 | `2 skipped`，无有效执行。 |
| 72 | `tests/models/multimodal/processing/test_h2ovl.py` | 未完成有效结论 | 否 | 原始日志缺失；补跑时收集到 `192` 个重参数化 case，资源/时间成本高，不适合作为当前 CI 准入项。 |
| 73 | `tests/models/multimodal/processing/test_phi3v.py` | 否（全 skip） | 否 | `12 skipped`，无有效执行。 |
| 74 | `tests/models/quantization/test_awq.py` | 否 | 否 | `ModelConfig` 校验失败，本质是 AWQ 量化路径在 NPU 上不受支持。 |
| 75 | `tests/models/quantization/test_bitblas.py` | 否（全 skip） | 否 | `1 skipped`，BitBLAS 非 Ascend 重点路径。 |
| 76 | `tests/models/test_gguf_download.py` | 部分 | 否 | 7 通过、2 失败；失败为 `ModelConfig` 校验问题，属于 GGUF/模型支持问题，Ascend CI 收益低。 |
| 77 | `tests/models/test_initialization.py` | 部分 | 是（nightly） | 日志中存在稳定失败：多模态初始化阶段 `embed_multimodal` 返回 `None` 导致断言失败；属于 Ascend 模型初始化/嵌入适配缺口。 |
| 78 | `tests/models/test_oot_registration.py` | 部分 | 是（nightly） | `3 failed, 1 passed`；OOT 注册相关断言失败，说明模型注册/架构识别链未稳定保留 upstream 行为。 |
| 79 | `tests/models/test_registry.py` | 部分 | 是（nightly） | `PrithviGeoSpatialMAE`、`Terratorch`、`MiDashengLMModel` 导入结果为 `None`；属于模型注册边界缺口。 |
| 80 | `tests/multimodal/test_sparse_tensor_validation_unit.py` | 否 | 否 | `torch.load` 默认 `weights_only=True` 导致 `_pickle.UnpicklingError`；更像 PyTorch 行为变化带来的上游测试/代码兼容问题。 |
| 81 | `tests/plugins_tests/test_platform_plugins.py` | 否 | 是（nightly） | 插件加载顺序导致当前平台变成真实 `npu` 而不是测试期望的 `DummyDevice`；属于平台插件适配边界。 |
| 82 | `tests/plugins_tests/test_stats_logger_plugins.py` | 否 | 是（presubmit） | 当前环境缺少测试依赖 `dummy_stat_logger`；补齐依赖后该类插件契约测试有较高 CI 价值。 |
| 83 | `tests/quantization/test_compressed_tensors.py` | 否 | 否 | `NotImplementedError: No compressed-tensors compatible scheme was found.`；当前量化特性在 NPU 不支持。 |
| 84 | `tests/quantization/test_configs.py` | 否 | 是（presubmit） | 量化自动识别返回 `ERROR`，例如期望 `gptq_marlin` 却未识别；这是轻量且高信号的适配边界测试。 |
| 85 | `tests/quantization/test_cpu_offload.py` | 否（全 skip） | 否 | `4 skipped`，对 Ascend CI 信号有限。 |
| 86 | `tests/quantization/test_experts_int8.py` | 否（全 skip） | 否 | `2 skipped`，无有效执行。 |
| 87 | `tests/quantization/test_gptq_dynamic.py` | 否 | 否 | `ModelConfig` 校验失败；GPTQ dynamic 在 NPU 上当前不支持。 |
| 88 | `tests/quantization/test_gptq_v2.py` | 否 | 否 | `ModelConfig` 校验失败；GPTQ v2 在 NPU 上当前不支持。 |
| 89 | `tests/quantization/test_lm_head.py` | 否 | 否 | 量化相关 `ModelConfig` 校验失败，当前不构成有价值的 Ascend 守护项。 |
| 90 | `tests/quantization/test_modelopt.py` | 否（全 skip） | 否 | `1 skipped`。 |
| 91 | `tests/quantization/test_ptpc_fp8.py` | 否（全 skip） | 否 | `9 skipped`；FP8 相关路径当前未在 Ascend 上形成有效覆盖。 |
| 92 | `tests/quantization/test_quark.py` | 否 | 否 | `ModelConfig` 校验失败；Quark 量化在 NPU 上当前不支持。 |
| 93 | `tests/quantization/test_torchao.py` | 否（全 skip） | 否 | `12 skipped`。 |
| 94 | `tests/reasoning/test_seedoss_reasoning_parser.py` | 否 | 否 | 纯逻辑失败：parser 返回 `None` 而非期望 reasoning 字符串；更像上游逻辑问题，非 Ascend 适配边界。 |
| 95 | `tests/samplers/test_beam_search.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 96 | `tests/samplers/test_logprobs.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 97 | `tests/samplers/test_no_bad_words.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 98 | `tests/utils_/test_mem_utils.py` | 否 | 否 | 缺少测试依赖 `vllm_test_utils`；本身不是高价值 Ascend 守护项。 |
| 99 | `tests/v1/attention/test_mla_backends.py` | 否 | 否 | `ModelConfig` 校验失败；当前更像模型/后端支持条件不满足。 |
| 100 | `tests/v1/core/test_priority_scheduler_random.py` | 否 | 否 | `block_pool.py` 中断言 `len(request.block_hashes) >= num_full_blocks` 失败；偏 upstream 核心逻辑，不是 Ascend 特有适配点。 |
| 101 | `tests/v1/core/test_scheduler_e2e.py` | 否 | 否 | 断言 `0 == 16` 失败；偏 upstream 调度逻辑，不是 Ascend 特有适配点。 |
| 102 | `tests/v1/cudagraph/test_cudagraph_mode.py` | 否 | 否 | `NameError: nvmlDeviceGetHandleByIndex is not defined`，明显依赖 NVML / Nvidia GPU 假设。 |
| 103 | `tests/v1/e2e/test_min_tokens.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 104 | `tests/v1/engine/test_engine_args.py` | 否 | 是（presubmit） | `NPUPlatform.get_device_total_memory()` 未实现并抛出 `NotImplementedError`；属于轻量高信号的平台适配测试。 |
| 105 | `tests/v1/entrypoints/openai/test_completion.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnFusedInferAttentionScoreV3 failed`，最终体现为 OpenAI 500。 |
| 106 | `tests/v1/entrypoints/openai/test_completion_with_image_embeds.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnIndexPutImpl failed`。 |
| 107 | `tests/v1/kv_connector/unit/test_cache_pollution_prevention.py` | 否 | 否 | `IndexError: list index out of range`；偏 connector 单元逻辑问题，Ascend CI 价值低。 |
| 108 | `tests/v1/kv_connector/unit/test_config.py` | 否 | 否 | `AttributeError: 'NoneType' object has no attribute 'is_deepseek_mla'`；偏通用配置逻辑问题。 |
| 109 | `tests/v1/kv_connector/unit/test_decode_bench_connector.py` | 否 | 否 | 断言 `1 == 3` 失败；偏 connector 单元逻辑。 |
| 110 | `tests/v1/kv_connector/unit/test_error_propagation.py` | 否 | 否 | `IndexError: list index out of range`；偏 connector 单元逻辑。 |
| 111 | `tests/v1/kv_connector/unit/test_example_connector.py` | 否 | 否 | 真实根因是 `AttributeError: 'tuple' object has no attribute 'shape'`；偏示例 connector 逻辑，不是高收益 Ascend 守护点。 |
| 112 | `tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py` | 否 | 否 | `IndexError: list index out of range`。 |
| 113 | `tests/v1/kv_connector/unit/test_kv_load_failure_recovery.py` | 否 | 否 | `IndexError: list index out of range`。 |
| 114 | `tests/v1/kv_connector/unit/test_offloading_connector.py` | 否 | 否 | 断言失败，偏 connector 单元逻辑。 |
| 115 | `tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py` | 否 | 否 | 断言 `1 == 3` 失败，偏 connector 生命周期逻辑。 |
| 116 | `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py` | 否 | 否 | 断言 `1 == 2` 失败，偏 connector 生命周期逻辑。 |
| 117 | `tests/v1/sample/test_rejection_sampler.py` | 否 | 否 | `TypeError: 'function' object is not subscriptable`；偏上游采样逻辑问题。 |
| 118 | `tests/v1/spec_decode/test_speculators_eagle3.py` | 否 | 是（nightly） | `vllm_ascend` 量化接口不匹配：`AscendW4A16FusedMoEMethod.get_weight()` 缺少 `params_dtype` 参数。 |
| 119 | `tests/v1/tracing/test_tracing.py` | 否 | 是（nightly） | 真实根因是 NPU 算子失败：`aclnnFusedInferAttentionScoreV3 failed`。 |
| 120 | `tests/weight_loading/test_weight_loading.py` | 否（全 skip） | 否 | `1 skipped`，无有效 Ascend 覆盖。 |
| 121 | `tests/entrypoints/openai/responses/test_mcp_tools.py` | 否 | 否 | `ModelConfig` 校验失败，真实根因是 `mxfp4 quantization is currently not supported in npu`。 |
| 122 | `tests/entrypoints/test_grpc_server.py` | 否 | 是（presubmit） | 补跑确认 collection 即失败：`ImportError: cannot import name 'vllm_engine_pb2' from 'vllm.grpc'`；属于 gRPC 生成/打包边界缺口。 |
| 123 | `tests/kernels/helion/test_helion_available.py` | 否（全 skip） | 否 | 历史统计为 `1 skipped`；平台不匹配。 |
| 124 | `tests/kernels/moe/test_routing_simulator.py` | 是 | 是（presubmit） | `27 passed`；纯逻辑路由模拟测试，可作为稳定轻量守护。 |
| 125 | `tests/kernels/moe/test_triton_moe_no_act_mul.py` | 否（全 skip） | 否 | 历史统计为 `74 skipped`；Triton/CUDA 导向，不适合 Ascend CI。 |
| 126 | `tests/lora/test_qwenvl.py` | 部分 | 是（nightly） | 既有报告显示 `4` 个 case 中部分通过、部分在初始化路径失败；LoRA + 多模态模块映射是 Ascend 重要适配边界，值得保留为夜测。 |
| 127 | `tests/models/language/pooling_mteb_test/test_nemotron.py` | 部分 | 否 | 既有统计为 `1 passed, 1 failed`；属于重型 MTEB 评测，当前缺少新的稳定失败栈，CI 成本高于收益。 |
| 128 | `tests/v1/engine/test_preprocess_error_handling.py` | 否（当前环境） | 是（需空卡前提） | 补跑后真实根因不是下载失败，而是 NPU 空闲显存不足：`Free memory on device (5.73/60.96 GiB) < desired GPU memory utilization (0.9, 54.86 GiB)`。在干净空卡 CI 环境下有价值。 |

## Recommended Admission Set

### P0 / presubmit

- `tests/quantization/test_configs.py`
- `tests/v1/engine/test_engine_args.py`
- `tests/entrypoints/test_grpc_server.py`（先补齐 protobuf 生成/导入前置）
- `tests/kernels/moe/test_routing_simulator.py`
- `tests/plugins_tests/test_stats_logger_plugins.py`（先补齐测试依赖）

### P1 / nightly

- `tests/models/multimodal/generation/test_granite_speech.py`
- `tests/models/multimodal/generation/test_qwen2_5_vl.py`
- `tests/models/multimodal/pooling/test_clip.py`
- `tests/models/multimodal/pooling/test_jinavl_reranker.py`
- `tests/models/test_initialization.py`
- `tests/models/test_oot_registration.py`
- `tests/models/test_registry.py`
- `tests/plugins_tests/test_platform_plugins.py`
- `tests/samplers/test_beam_search.py`
- `tests/samplers/test_logprobs.py`
- `tests/samplers/test_no_bad_words.py`
- `tests/v1/e2e/test_min_tokens.py`
- `tests/v1/entrypoints/openai/test_completion.py`
- `tests/v1/entrypoints/openai/test_completion_with_image_embeds.py`
- `tests/v1/spec_decode/test_speculators_eagle3.py`
- `tests/v1/tracing/test_tracing.py`
- `tests/lora/test_qwenvl.py`
- `tests/v1/engine/test_preprocess_error_handling.py`（需空卡/稳定显存前提）

## Non-candidates

- CUDA/Triton/NVML/平台跳过类：`test_vit_backend_functionality.py`, `test_bitblas.py`, `test_helion_available.py`, `test_triton_moe_no_act_mul.py`, `test_cudagraph_mode.py` 等
- 当前明确不支持的量化类：`test_awq.py`, `test_gptq_dynamic.py`, `test_gptq_v2.py`, `test_quark.py`, `test_compressed_tensors.py`, `test_mcp_tools.py`
- 纯上游/通用逻辑失败、非 Ascend 边界：大部分 `kv_connector/unit/*`、`test_seedoss_reasoning_parser.py`, `test_rejection_sampler.py`, `test_priority_scheduler_random.py`, `test_scheduler_e2e.py`