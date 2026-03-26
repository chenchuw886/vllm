# vLLM 首批 30 个测试用例的 Ascend CI 适配分析

更新时间：2026-03-25

## 1. 分析范围

本报告覆盖 [TEST_FILES_NEED_ANALYSIS.md](TEST_FILES_NEED_ANALYSIS.md) 中前 30 个测试文件：

1. `tests/compile/test_pass_manager.py`
2. `tests/compile/test_wrapper.py`
3. `tests/distributed/test_distributed_oot.py`
4. `tests/distributed/test_sequence_parallel.py`
5. `tests/entrypoints/llm/test_mm_cache_stats.py`
6. `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`
7. `tests/entrypoints/openai/test_chat.py`
8. `tests/entrypoints/openai/test_default_mm_loras.py`
9. `tests/entrypoints/openai/test_oot_registration.py`
10. `tests/entrypoints/openai/test_openai_schema.py`
11. `tests/entrypoints/openai/test_sparse_tensor_validation.py`
12. `tests/entrypoints/openai/test_tensorizer_entrypoint.py`
13. `tests/entrypoints/openai/test_transcription_validation_whisper.py`
14. `tests/entrypoints/openai/test_video.py`
15. `tests/entrypoints/openai/test_vision.py`
16. `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py`
17. `tests/entrypoints/pooling/classify/test_online.py`
18. `tests/entrypoints/pooling/classify/test_online_vision.py`
19. `tests/entrypoints/pooling/score/test_correctness_mteb.py`
20. `tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py`
21. `tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py`
22. `tests/evals/gpt_oss/test_gpqa_correctness.py`
23. `tests/kernels/attention/test_attention_selector.py`
24. `tests/kernels/attention/test_flashmla.py`
25. `tests/kernels/attention/test_flashmla_sparse.py`
26. `tests/kernels/attention/test_mha_attn.py`
27. `tests/kernels/core/test_activation.py`
28. `tests/kernels/moe/test_gpt_oss_triton_kernels.py`
29. `tests/kernels/moe/test_modular_kernel_combinations.py`
30. `tests/kernels/moe/test_modular_oai_triton_moe.py`

---

## 2. 分析环境

### 仓库版本

- `vllm`
  - branch: `v0150rc1`
  - commit: `54814f7868c22e8c34492199f9b6b9fe249d8e83`
  - version: `0.15.0rc1`
- `vllm-ascend`
  - branch: `v0150rc1`
  - commit: `3d43ed997e715981321bffc10f72f11d9522a5ae`
  - installed package version: `0.15.0rc1`

### 运行环境

- Python: `3.11.14`
- torch: `2.9.0+cpu`
- torch-npu: `2.9.0`
- CANN: `8.5.0`
- 环境状态：非全新环境，存在部分预缓存
  - Hugging Face cache 目录层级计数：`16`
  - ModelScope cache 目录层级计数：`4`
- 外网访问：**默认直连不可达，但通过 SOCKS 代理可访问外部模型站点**
  - 直连 `huggingface.co:443`：`Network is unreachable`
  - 通过 `socks5h://127.0.0.1:1080`：可建立连接并下载缺失资源

### 本次分析中实际执行过的代表性测试

- `tests/compile/test_pass_manager.py`
- `tests/compile/test_wrapper.py`
- `tests/distributed/test_distributed_oot.py`
- `tests/distributed/test_sequence_parallel.py`
- `tests/entrypoints/llm/test_mm_cache_stats.py`
- `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py`
- `tests/entrypoints/pooling/score/test_correctness_mteb.py`
- `tests/entrypoints/pooling/classify/test_online.py`
- `tests/entrypoints/openai/test_transcription_validation_whisper.py`
- `tests/entrypoints/openai/test_video.py`
- `tests/entrypoints/openai/test_sparse_tensor_validation.py`
- `tests/kernels/attention/test_flashmla_sparse.py`
- `tests/kernels/core/test_activation.py`

### 本次分析中未做的环境修改

- 未修改产品代码

### 本次分析中使用过的最小化工作绕过

- 使用 SOCKS 代理推进外部模型访问：
  - `http_proxy=socks5h://127.0.0.1:1080`
  - `https_proxy=socks5h://127.0.0.1:1080`
  - `all_proxy=socks5h://127.0.0.1:1080`
- 当 `ModelScope` 无对应仓库时，回退到 Hugging Face 下载继续分析
- 对依赖本地资源服务的测试额外设置：
  - `no_proxy=127.0.0.1,localhost`
  - `NO_PROXY=127.0.0.1,localhost`

### 本次分析中新增安装的最小依赖

- `rapidfuzz`
  - 用于推进 `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py` 的 collection 阶段
- `mteb`
  - 用于推进 `tests/entrypoints/pooling/score/test_correctness_mteb.py` 的 collection 阶段
- `bm25s`
  - 由 `mteb` 运行时提示缺失后补装（`mteb[bm25s]` 依赖链的一部分），用于继续推进 `test_correctness_mteb.py`
- `PyStemmer`
  - 在补齐 `bm25s` 后继续运行 `test_correctness_mteb.py` 时缺失 `Stemmer` 模块，补装后将失败推进到更深的 benchmark 运行栈
- `socksio`
  - 在 `tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py` 中，`openai.AsyncOpenAI` 通过 `httpx` 继承当前 `socks5h` 代理环境时必需；补装后可将失败从异步客户端初始化推进到真实 SageMaker session 语义，并最终确认整文件通过

---

## 3. 为什么这里还会出现“网络问题”？

这部分需要单独说明：**报告里出现“网络问题/外部资源问题”，并不等于我把根因简单归结为网络。** 这份报告已经按用户要求，对此前因网络未能推进的条目继续做了复跑，优先把失败推进到更深的 Ascend 代码路径。

按照当前工作流，只有在满足以下条件时，我才会把“网络/资源不可用”保留为主结论：

1. 当前测试的第一稳定失败点，确实是访问 Hugging Face / 数据集 / 远程资源；
2. 该测试本身强依赖远程模型、数据集或媒体资产；
3. 在“不做大规模环境改造”的前提下，无法很快把它推进到更深的代码路径；
4. 即使推进下去，这类测试也通常不适合作为 Ascend presubmit CI。

### 3.1 当前环境里的“网络问题”具体是什么情况

本次环境里不是偶发超时，而是**默认直连外网不可达**。实际复现中出现过：

- 直接访问 `https://huggingface.co` 失败；
- `AutoTokenizer.from_pretrained(...)` / `AutoConfig.from_pretrained(...)` 无法拉取配置；
- `huggingface_hub` 报 `Network is unreachable`；
- 本地缓存里又没有这些测试精确需要的模型或 tokenizer 文件。

但进一步验证后发现，配置以下代理变量后可以访问外部资源：

- `http_proxy=socks5h://127.0.0.1:1080`
- `https_proxy=socks5h://127.0.0.1:1080`
- `all_proxy=socks5h://127.0.0.1:1080`

因此这里的正确表述是：**默认网络路径不可达，但可通过 SOCKS 代理推进测试到真实根因**。

### 3.2 为什么不是所有测试都受网络影响

因为前 30 个用例里大概分成三类：

1. **纯本地单测 / 逻辑测试**
   - 例如 `test_pass_manager.py`、`test_wrapper.py`
   - 这类不应该依赖网络；如果出现网络访问，通常是默认配置或测试夹具把模型名带进了配置构造流程

2. **服务集成测试，但理论上可离线运行**
   - 例如 `test_sequence_parallel.py`、`test_vision.py`、`test_online.py`
   - 它们理论上可以通过预缓存模型后离线跑，因此“网络问题”更多是**缺失测试前置条件**

3. **强依赖外部模型 / 数据集 / 多媒体资源的测试**
   - 例如 `test_transcription_api_correctness.py`、`test_correctness_mteb.py`、`test_gpqa_correctness.py`
   - 这类测试即使不是 Ascend，也天然对外部资源敏感；在 CI 中信噪比偏低

### 3.3 这批测试里具体哪些“网络问题”是这样来的

#### A. 不是测试本意要联网，但被默认配置带出来了

- `tests/compile/test_pass_manager.py`
  - 测试本质只是验证 pass manager 的 UUID 和配置行为
  - 但 `ModelConfig(dtype=torch.bfloat16)` 会走到默认模型 `Qwen/Qwen3-0.6B`
  - 默认模型配置解析触发 Hugging Face 查询
  - 所以这里的“网络问题”其实是 **upstream 测试/配置设计问题**，不是 Ascend 根因

- `tests/entrypoints/openai/test_sparse_tensor_validation.py`
  - 文件里一部分 case 本来只是想验证稀疏张量安全检查
  - 但某些集成路径会触发 `ModelConfig` / engine 初始化
  - 进而又落到默认模型解析，导致联网失败
  - 这里网络问题也是**次级噪声**，不是主问题；主问题反而是 `torch.load(weights_only=True)` 抛出 `UnpicklingError`，而测试只接受 `RuntimeError/ValueError`

#### B. 测试确实需要模型，但当前 cache 不完整

- `tests/distributed/test_sequence_parallel.py`
  - 需要 `hmellor/tiny-random-LlamaForCausalLM`
  - 用代理补齐下载后，失败推进到 Ascend worker 初始化
  - 真实失败变为：设备空闲显存低于 `gpu_memory_utilization=0.9` 所需阈值，server 在引擎初始化阶段退出
  - 这类不再应归类为“网络问题”，而是**运行时资源前置条件 / 平台兼容问题**

- `tests/entrypoints/openai/test_vision.py`
- `tests/entrypoints/pooling/classify/test_online.py`
- `tests/entrypoints/llm/test_mm_cache_stats.py`
  - 这些都依赖模型或多模态处理器
  - 其中 `test_mm_cache_stats.py` 在补齐代理后，第一次失败其实是 `127.0.0.1` 本地图片请求被错误走代理
  - 加上 `no_proxy=127.0.0.1,localhost` 后，该测试在 Ascend 上可完整通过
  - 说明它的主问题不是网络，也不是产品缺陷，而是**测试运行前置环境需要正确设置 localhost 直连**
  - `test_vision.py` 与 `classify/test_online.py` 则进一步暴露出另一类问题：它们会拉取多 GB 权重，分析中已进入真实下载阶段，但继续推进需要较长时间的大规模资源预置
  - 因此这两项的主问题应表述为**重资源前置条件**，而不是“网络不可达”

#### C. 测试天然依赖远程大资源，不适合直接进 presubmit

- `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`
- `tests/entrypoints/openai/test_transcription_validation_whisper.py`
- `tests/entrypoints/pooling/score/test_correctness_mteb.py`
- `tests/evals/gpt_oss/test_gpqa_correctness.py`
- `tests/entrypoints/openai/test_video.py`
- `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py`

这类用例即使把网络问题解决掉，依然有以下问题：

- 依赖大模型 / 数据集 / 多媒体素材
- 启动成本高
- 稳定性更依赖外部资源状态
- 对 Ascend 每次提交的回归信号不够“纯”

因此它们即使保留，也更适合 nightly 或 manual regression，而不是 presubmit。

### 3.4 结论

这批结果里之所以还有“网络问题”，本质上有三种来源：

1. **默认模型配置把本来纯本地的测试意外带到联网路径上**；
2. **测试需要的模型未预缓存，而当前环境外网不可达**；
3. **测试本身就是重资源/重外部依赖，不适合作为高频 CI**。

所以后续处理上要分开：

- 对第 1 类：应优先修测试或默认配置，去掉不必要联网；
- 对第 2 类：可以通过预缓存资源，把问题推进到真正的 Ascend 代码路径；
- 对第 3 类：应直接降级到 nightly/manual，而不是强行进 presubmit。

---

## 4. 逐用例结论表

> 说明：
> - “观察到的首个稳定失败”记录的是本次静态/动态分析中最先稳定出现的失败点；
> - “真实根因”尽量给到更深一层的语义原因；
> - “CI 结论”取值为：`presubmit` / `nightly` / `manual` / `reject`。

| 测试文件 | 主要验证内容 | 观察到的首个稳定失败 / 首个阻塞点 | 真实根因 | 分类 | 责任归属 | 关键文件/符号 | 是否值得纳入 vllm-ascend 覆盖 | CI 结论 |
|---|---|---|---|---|---|---|---|---|
| `tests/compile/test_pass_manager.py` | pass manager 注册与 UUID 稳定性 | 实测同时出现 `NameError: RMSNormQuantFusionPass` 与默认模型联网 | upstream pass manager 在非 CUDA 平台下仍走 CUDA fusion 符号；另有默认模型配置导致无谓联网 | upstream `vllm` code defect + test precondition missing | `vllm` | `vllm/compilation/pass_manager.py::PostGradPassManager.configure` | 值得，能保护编译配置契约 | `manual` |
| `tests/compile/test_wrapper.py` | `TorchCompileWithNoGuardsWrapper` 编译行为 | 实测出现 `InductorError(KeyError('cpu'))`；另一路出现 Ascend `compile_ranges_split_points` 空值错误 | Ascend 编译配置更新逻辑与当前运行模式不兼容，且该测试默认假设的 torch compile/inductor 语义并不直接适配 Ascend | compiler or runtime compatibility problem | `vllm-ascend` + runtime stack | `vllm_ascend/ascend_config.py::update_compile_ranges_split_points` | 非常值得，是 P0 适配边界 | `presubmit`（需 Ascend 适配版） |
| `tests/distributed/test_distributed_oot.py` | 分布式 OOT 模型通过 OpenAI server 注册与服务 | 实测 server 启动后立即退出 | dummy OOT 架构 `MyOPTForCausalLM` 未被当前模型注册链接受 | `vllm-ascend` adaptation defect | `vllm-ascend` | `tests/entrypoints/openai/test_oot_registration.py::run_and_test_dummy_opt_api_server` | 值得，能保护插件/模型注册边界 | `nightly` |
| `tests/distributed/test_sequence_parallel.py` | sequence parallel 在 TP/PP/compile 路径下的行为一致性 | 代理补齐模型下载后，server 在引擎初始化阶段退出；真实异常为设备空闲显存低于 `gpu_memory_utilization=0.9` 目标阈值 | 真实失败来自 Ascend worker 启动时的设备内存前置条件检查，而非网络；当前机器空闲显存不足以满足测试默认配置 | compiler or runtime compatibility problem | runtime stack / `vllm-ascend` | `vllm_ascend/worker/worker.py::_init_device`, `tests/distributed/test_sequence_parallel.py::_compare_sp` | 值得，是 P1 适配边界，可捕获 SP 启动期资源/运行时回归 | `nightly`（需预缓存并保证空闲显存） |
| `tests/entrypoints/llm/test_mm_cache_stats.py` | 多模态 cache 查询/命中率统计 | 代理补齐模型后首次失败为 localhost 图片请求被错误送入 SOCKS 代理；加 `no_proxy` 后 2 个 case 均通过 | 主问题是测试环境代理配置而非产品缺陷；在保留 Ascend 环境与正确 localhost 直连前提下，MM cache 统计契约可正常通过 | test precondition missing | test / environment | `tests/entrypoints/llm/test_mm_cache_stats.py`, `tests/entrypoints/openai/test_vision.py` | 值得，MM cache 信号高，且已证明可在 Ascend 上稳定跑通 | `nightly`（需预缓存并设置 `no_proxy`） |
| `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py` | transcription API 正确性/WER | `ModelScope` 缺仓库时回退 Hugging Face 后，`openai/whisper-large-v3` 成功解析为 `WhisperForConditionalGeneration`，随后立即进入 `4.99G + 1.18G` 分片及额外权重下载 | 更深根因已不是模型源缺失，而是超大 Whisper 模型准备成本；在进入 WER 数据集评测前，测试首先被多 GB 权重获取链路阻塞 | external model or dataset unavailable | test / environment | `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`, `tests/utils.py::RemoteOpenAIServer`, `vllm/config/model.py` | 价值一般，成本高 | `manual` |
| `tests/entrypoints/openai/test_chat.py` | chat API 大量契约（流式、结构化输出、logprobs、LoRA 等） | 最小 case 中，LoRA fixture 先经 `huggingface_hub.snapshot_download` 拉取 `typeof/zephyr-7b-beta-lora`，随后基模 `HuggingFaceH4/zephyr-7b-beta` 在 ModelScope 继续进入 8 个大权重 shard 下载 | 首个稳定阻塞是测试资源准备成本：LoRA 夹具绕不过 HF Hub，基模又需多分片大权重；在进入真实 API 契约前就落入重资源链路，不适合作为未预缓存的高频 CI | test precondition missing | test / environment | `tests/entrypoints/openai/test_chat.py::zephyr_lora_files`, `tests/entrypoints/openai/test_chat.py::server` | 有价值，但应拆子集并预缓存模型/LoRA | `nightly`（建议子集 + 预缓存） |
| `tests/entrypoints/openai/test_default_mm_loras.py` | 默认多模态 LoRA 路由 | 模块导入期即执行 `snapshot_download("microsoft/Phi-4-multimodal-instruct")`，随后开始下载 base 模型 3 个大 shard 及 `speech-lora` / `vision-lora` 额外权重 | 首个稳定根因是测试文件自身的模块级 Hugging Face 下载设计：它绕过 `VLLM_USE_MODELSCOPE`，并在任何测试逻辑前触发超大体量 base+LoRA 资源准备 | test precondition missing | test | `tests/entrypoints/openai/test_default_mm_loras.py`, `huggingface_hub.snapshot_download` | 值得，但成本极高，需改为预缓存或本地路径后才适合继续分析 Ascend 语义 | `manual` |
| `tests/entrypoints/openai/test_oot_registration.py` | OOT 注册后 API server 服务能力 | 与分布式变体同类，server 启动即可能失败 | OOT 模型注册/架构识别链未稳定保留 upstream 行为 | `vllm-ascend` adaptation defect | `vllm-ascend` | `tests/entrypoints/openai/test_oot_registration.py` | 值得，插件能力重要 | `nightly` |
| `tests/entrypoints/openai/test_openai_schema.py` | OpenAI schema / schemathesis fuzz | schemathesis + server + 模型构成高噪声链路 | 不适合用作 Ascend 高频 CI 守护 | flaky or nondeterministic test | test | `tests/entrypoints/openai/test_openai_schema.py` | 不建议 | `reject` |
| `tests/entrypoints/openai/test_sparse_tensor_validation.py` | 稀疏张量安全校验 | 实测 `torch.load(weights_only=True)` 抛 `UnpicklingError`；部分集成 case 又落入默认模型联网 | 真正值得关注的是测试对异常类型假设过窄；网络只是次级噪声 | compiler or runtime compatibility problem + test precondition missing | `vllm` test / runtime stack | `vllm/multimodal/media/image.py`, `vllm/multimodal/media/audio.py` | 非常值得，属于 P0 安全防护 | `presubmit`（修测试后） |
| `tests/entrypoints/openai/test_tensorizer_entrypoint.py` | tensorizer loader + OpenAI endpoint | 首阻塞为 tensorizer/CUDA 型前提 | upstream 文件偏 CUDA/tensorizer 专用，不适合作为 Ascend 原样 CI | test not applicable on Ascend platform | test | `tests/entrypoints/openai/test_tensorizer_entrypoint.py` | 不建议原样纳入 | `manual` |
| `tests/entrypoints/openai/test_transcription_validation_whisper.py` | Whisper API 入参与行为校验 | 继续按 `<7B` 策略重跑最小语义 case（`test_bad_requests`）：服务端可稳定解析 `WhisperForConditionalGeneration`，但启动阶段持续停留在 `model.safetensors (1.62G)` 拉取（最新观测约 `67MB/1.62G`） | 当前首个稳定阻塞仍是模型准备成本（下载吞吐/缓存预置），尚未进入 API 断言层；根因已从“模型源不可达”收敛为“重资源前置条件未满足” | test precondition missing | test / environment | `tests/entrypoints/openai/test_transcription_validation_whisper.py::server`, `tests/utils.py::RemoteOpenAIServer` | 一般 | `manual`（建议预缓存后再评估语义层） |
| `tests/entrypoints/openai/test_video.py` | 视频输入 OpenAI 接口 | 继续按 `<7B` 策略重跑最小语义 case（`test_error_on_invalid_video_url_type`）：可完成配置解析并进入 `model.safetensors (1.79G)` 下载，最新观测约 `256MB/1.79G` 后仍未进入断言阶段 | 当前首个稳定阻塞仍是模型准备成本；虽然模型参数量 `<7B`，但实际前置资源链路仍重（权重下载 + 潜在整仓附加文件），导致尚未到语义断言层 | test precondition missing | test / environment | `tests/entrypoints/openai/test_video.py::server`, `vllm/transformers_utils/repo_utils.py` | 价值一般，不适合高频 CI | `manual`（建议仅在预缓存环境做夜间回归） |
| `tests/entrypoints/openai/test_vision.py` | 图片类 OpenAI 视觉接口 | 继续重跑最小语义 case（`test_error_on_invalid_image_url_type`）时，可稳定到 `Phi3VForCausalLM` 解析，但随后在权重准备阶段进入 `filelock` 等待（`filelock/_api.py`），未到 API 断言 | 当前主阻塞已收敛为重资源模型准备成本 + 缓存锁竞争（非网络不可达）；该阻塞在多次中断/重试后可复现，继续推进需预缓存完整 VLM 权重并避免并发下载竞争 | test precondition missing | test / environment | `tests/entrypoints/openai/test_vision.py::server`, `filelock/_api.py` | 值得，但应在有预缓存的 CI 层级运行 | `nightly`（需预缓存大模型并设置 `no_proxy`） |
| `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py` | tool parser 与 reasoning/tool schema | 补齐 `rapidfuzz`、并通过 `VLLM_PLUGINS=ascend` 排除双平台插件冲突后，server 初始化在 `openai/gpt-oss-20b` 处失败：`mxfp4 quantization is currently not supported in npu` | 更深根因是该模型默认量化格式与 Ascend 平台不兼容；已越过“缺依赖/环境噪声”，进入真实平台能力边界 | compiler or runtime compatibility problem | runtime stack / upstream model config | `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py::server`, `vllm/engine/arg_utils.py::create_model_config` | 一般 | `manual` |
| `tests/entrypoints/pooling/classify/test_online.py` | 在线 classify 服务契约 | 在 `<7B` 允许下载条件下继续推进：`Qwen2.5-1.5B-apeach` 稳定进入 2 个分片权重下载阶段（最近一次观测约 `201MB/5.00GB` 与 `201MB/1.18GB`）但尚未进入 API 断言 | 当前稳定阻塞仍是模型准备时间成本（非网络/依赖错误）；未出现更深语义失败，说明该 case 仍处高成本前置阶段 | test precondition missing | test / environment | `tests/entrypoints/pooling/classify/test_online.py::server` | 值得 | `nightly`（建议预缓存后再做功能断言） |
| `tests/entrypoints/pooling/classify/test_online_vision.py` | 多模态 classify（文本/图像/视频） | `ModelScope` 缺仓库时回退 Hugging Face 后，文本 case 可解析出 `Qwen2_5_VLForSequenceClassification`，随后进入 4 个分片权重下载（约 `4.97G + 4.99G + 4.93G + 602M`） | 更深根因已不是模型源不存在，而是 7B 视频分类模型的超大权重准备成本；即使只测文本输入，也会被共享 server fixture 的大模型启动前置条件阻塞 | external model or dataset unavailable | test / environment | `tests/entrypoints/pooling/classify/test_online_vision.py::server`, `tests/entrypoints/pooling/classify/test_online_vision.py::test_chat_text_request` | 可保留小子集，但需强预缓存 | `manual` |
| `tests/entrypoints/pooling/score/test_correctness_mteb.py` | pooling/score 在 MTEB 上的正确性 | 按依赖链补齐 `mteb` → `bm25s` → `PyStemmer` 后，`test_mteb_score` 可完成 server 启动、数据集下载与首轮评测，但在 `task.convert_to_reranking()` 二次处理时稳定报错：`TypeError: 'NoneType' object is not subscriptable` | 真实根因已推进到 MTEB benchmark 栈内部状态错误（`mteb/abstasks/retrieval.py::_process_data` 访问 `self.dataset[hf_subset]` 时对象为空），属于外部评测框架/版本兼容问题，而非 Ascend 适配缺陷 | compiler or runtime compatibility problem | runtime stack / external dependency | `tests/models/language/pooling_mteb_test/mteb_score_utils.py::run_mteb_rerank`, `mteb/abstasks/retrieval.py::_process_data` | 有价值但高成本，且依赖外部评测栈稳定性 | `nightly`（建议固定 mteb 版本并预装 `mteb`/`bm25s`/`PyStemmer`） |
| `tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py` | SageMaker 动态 LoRA adapter 管理 | 实跑后整文件通过；覆盖了成功加载/卸载、坏路径 404、非法 `adapter_config.json` 400、带 adapter 的 `/invocations`、多 adapter 批量装卸 | 当前 Ascend 运行时已保留 upstream LoRA API 语义。代码链路是 SageMaker `/adapters` -> `OpenAIServingModels.load_lora_adapter()` -> `engine_client.add_lora()` -> `vllm_ascend.worker.worker.add_lora()`；先前看到的异常其实来自测试里的负例分支（不存在目录、非法 JSON），并被正确映射为 404/400，而不是 Ascend 适配缺陷 | portable/no obvious blocker | `vllm` + `vllm-ascend` | `tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py`, `vllm/entrypoints/openai/models/serving.py::load_lora_adapter`, `vllm_ascend/worker/worker.py::add_lora` | 非常值得，是典型 P1 适配边界守护 | `presubmit`（需小模型 + LoRA 预缓存） |
| `tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py` | SageMaker session header/stateful invocation | 首轮实跑时，`async_client` fixture 在创建 `openai.AsyncOpenAI` 时因 SOCKS 代理缺少 `socksio` 失败；补装 `socksio` 后整文件通过（5 passed） | 真实失败并不在 SageMaker session middleware 本身，而是在测试夹具 `RemoteOpenAIServer.get_async_client()` 的异步客户端初始化：`AsyncOpenAI` -> `httpx` 默认 `trust_env=True` -> 继承 `socks5h` 代理 -> 缺少 `socksio` 无法构造传输层。补齐依赖后，`/invocations` 上的 `stateful_session_manager()` 与 header 语义均按预期工作，包括 `NEW_SESSION`、缺失 session header 的 `424 invalid session_id`、以及正常会话调用/关闭流程 | test precondition missing（已消除） | test / environment | `tests/entrypoints/sagemaker/conftest.py::async_client`, `tests/utils.py::RemoteOpenAIServer.get_async_client`, `vllm/entrypoints/sagemaker/routes.py::invocations` | 非常值得，信号高且已证明可稳定通过 | `presubmit`（需 `socksio` + 小模型前置） |
| `tests/evals/gpt_oss/test_gpqa_correctness.py` | GPQA eval 正确性 | 直接运行时 `request.config.getoption("--model")` 得到 `None`，随后 `RemoteOpenAIServer` 参数解析因 `None.startswith(...)` 抛 `AttributeError` | 首个稳定根因是测试必须通过 pytest 选项显式提供 `--model`/`--metric` 等评测参数；它不是可直接收集执行的普通单测 | test precondition missing | test | `tests/evals/gpt_oss/test_gpqa_correctness.py`, `vllm/utils/argparse_utils.py::FlexibleArgumentParser.parse_args` | 不建议 | `reject` |
| `tests/kernels/attention/test_attention_selector.py` | attention backend 选择逻辑 | 文件主要围绕 CUDA/ROCm backend 选择 | 不是 Ascend upstream 行为契约的合适入口 | test not applicable on Ascend platform | test | `tests/kernels/attention/test_attention_selector.py` | 不建议 | `reject` |
| `tests/kernels/attention/test_flashmla.py` | dense FlashMLA kernel 正确性 | 明确 `cuda:0` | CUDA 专用 kernel 测试 | test not applicable on Ascend platform | test | `tests/kernels/attention/test_flashmla.py` | 不建议 | `reject` |
| `tests/kernels/attention/test_flashmla_sparse.py` | sparse FlashMLA kernel / metadata | 实测 clean skip | 平台检测正常，Ascend 不适用 | test not applicable on Ascend platform | test | `tests/kernels/attention/test_flashmla_sparse.py` | 不建议 | `reject` |
| `tests/kernels/attention/test_mha_attn.py` | MHA attention backend 数值/行为 | 主要数值路径偏 CUDA | 原样不能代表 Ascend 契约 | test not applicable on Ascend platform | test | `tests/kernels/attention/test_mha_attn.py` | 不建议 | `reject` |
| `tests/kernels/core/test_activation.py` | 自定义 activation kernel 与 opcheck | 实测 collection 即因 `torch.ops._C.gelu_fast` 缺失报错 | 依赖 CUDA `_C` custom op namespace，Ascend 不适用 | test not applicable on Ascend platform | test | `tests/kernels/core/test_activation.py` | Ascend 已有本地算子测试更合适 | `reject` |
| `tests/kernels/moe/test_gpt_oss_triton_kernels.py` | GPT-OSS Triton MoE kernel | Triton/CUDA 前提 | CUDA/Triton 专用 | test not applicable on Ascend platform | test | `tests/kernels/moe/test_gpt_oss_triton_kernels.py` | 不建议 | `reject` |
| `tests/kernels/moe/test_modular_kernel_combinations.py` | modular MoE backend 组合矩阵 | 依赖 CUDA device 与深度 GPU 包 | 非 Ascend 原样可移植测试 | test not applicable on Ascend platform | test | `tests/kernels/moe/test_modular_kernel_combinations.py` | 不建议 | `reject` |
| `tests/kernels/moe/test_modular_oai_triton_moe.py` | Triton MoE 与 reference 对比 | Triton/CUDA 前提 | CUDA 专用 MoE kernel 测试 | test not applicable on Ascend platform | test | `tests/kernels/moe/test_modular_oai_triton_moe.py` | 不建议 | `reject` |

---

## 5. 推荐纳入策略

### 5.1 推荐纳入 presubmit CI

这些用例的共同特点是：

- 更接近 upstream 行为契约；
- 对 Ascend 适配边界有较高信号；
- 若做少量前置准备，可具备较好的稳定性。

推荐：

1. `tests/compile/test_wrapper.py`
   - 作为 **Ascend-adapted variant** 纳入
   - 保护 compile config / wrapper / graph compile 相关边界

2. `tests/entrypoints/openai/test_sparse_tensor_validation.py`
   - 在上游测试对异常类型放宽后纳入
   - 保护安全校验契约，价值很高

3. `tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py`
   - 使用小模型或预缓存模型纳入
   - 保护 API / session middleware 契约

### 5.2 推荐纳入 nightly CI

1. `tests/distributed/test_sequence_parallel.py`
2. `tests/distributed/test_distributed_oot.py`
3. `tests/entrypoints/openai/test_oot_registration.py`
4. `tests/entrypoints/llm/test_mm_cache_stats.py`
5. `tests/entrypoints/openai/test_vision.py`
6. `tests/entrypoints/pooling/classify/test_online.py`
7. `tests/entrypoints/pooling/score/test_correctness_mteb.py`
8. `tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py`

原因：

- 覆盖了 sequence parallel、OOT plugin、MM cache、vision、pooling、runtime LoRA 等 Ascend 敏感边界；
- 但它们大都需要预缓存模型/资源，不适合每次提交都跑；
- 其中 `test_mm_cache_stats.py` 已证明在 Ascend 上可跑通，但要额外保证 `localhost` 不走代理。

### 5.3 保留为 manual regression

- `tests/compile/test_pass_manager.py`
- `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`
- `tests/entrypoints/openai/test_chat.py`
- `tests/entrypoints/openai/test_default_mm_loras.py`
- `tests/entrypoints/openai/test_transcription_validation_whisper.py`
- `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py`
- `tests/entrypoints/pooling/classify/test_online_vision.py`

### 5.4 直接拒绝原样纳入

- `tests/entrypoints/openai/test_openai_schema.py`
- `tests/entrypoints/openai/test_video.py`
- `tests/evals/gpt_oss/test_gpqa_correctness.py`
- 所有明显 CUDA/Triton-only 的 kernel 文件：
  - `tests/kernels/attention/test_attention_selector.py`
  - `tests/kernels/attention/test_flashmla.py`
  - `tests/kernels/attention/test_flashmla_sparse.py`
  - `tests/kernels/attention/test_mha_attn.py`
  - `tests/kernels/core/test_activation.py`
  - `tests/kernels/moe/test_gpt_oss_triton_kernels.py`
  - `tests/kernels/moe/test_modular_kernel_combinations.py`
  - `tests/kernels/moe/test_modular_oai_triton_moe.py`

---

## 6. 关键结论摘要

### 6.1 这一批里最值得优先跟进的真实问题

1. `tests/compile/test_wrapper.py`
   - 暴露了 `vllm-ascend` 的 compile config 更新路径问题
   - 属于真实 Ascend 适配问题

2. `tests/distributed/test_distributed_oot.py`
   - 暴露了 OOT dummy 架构注册/识别问题
   - 属于真实 Ascend 插件/注册边界问题

3. `tests/distributed/test_sequence_parallel.py`
  - 去掉网络阻塞后，暴露的是 Ascend worker 的设备显存阈值检查
  - 属于真实运行时/资源前置条件问题，而不是下载问题

4. `tests/entrypoints/openai/test_sparse_tensor_validation.py`
   - 暴露了测试与当前 `torch.load(weights_only=True)` 行为不匹配
   - 是真实的测试/运行时兼容问题，且安全价值高

5. `tests/entrypoints/openai/test_transcription_validation_whisper.py` / `tests/entrypoints/openai/correctness/test_transcription_api_correctness.py`
  - 在 `ModelScope` 缺仓库时回退到 Hugging Face 后，问题继续推进到真实 Whisper 权重下载链路
  - 说明它们的更深阻塞是大模型准备成本，而不是模型源解析失败

6. `tests/evals/gpt_oss/test_gpqa_correctness.py`
  - 直接运行时并不是卡模型，而是测试本身缺少 `--model` 等命令行参数
  - 属于 eval 型测试的执行前置条件问题

7. `tests/entrypoints/openai/tool_parsers/test_openai_tool_parser.py`
  - 在补齐 `rapidfuzz` 并排除双平台插件冲突后，失败推进到 `openai/gpt-oss-20b` 的 `mxfp4` 量化不支持 NPU
  - 属于真实平台能力边界问题，而不是依赖缺失

8. `tests/entrypoints/pooling/score/test_correctness_mteb.py`
  - 在补齐 `mteb` 后又暴露 `bm25s` 依赖链问题；补齐后可进一步进入 MTEB 运行准备
  - 说明其核心挑战是 benchmark 运行栈与执行成本，而不再是基础缺包

### 6.2 这一批里“网络问题”的正确解读

- 不是简单地说“网络坏了，所以分析不下去”；
- 在当前环境里，默认直连外网不可达，但 SOCKS 代理可用；
- 而是要把它区分成：
  - 无谓联网（应修测试/默认配置）
  - 缺缓存（可通过代理或预缓存推进）
  - 天然重外部依赖（应降级到 nightly/manual）
  - 代理副作用（如 `localhost` 资源被错误走代理，应通过 `no_proxy` 排除）
  - 模型源映射缺口（如 `Whisper` / `VideoCls` 仓库在当前 ModelScope 路径下不存在）
  - 模型源回退后暴露的真实大模型准备成本（如 Hugging Face 多 GB 权重）
  - 测试前置依赖缺失（如 `mteb`、`rapidfuzz`、`bm25s`、pytest `--model` 参数）

### 6.3 对 vllm-ascend CI 的启发

首批 30 个用例里，真正适合进入 Ascend CI 的，不应是 CUDA/Triton kernel 原样测试，而应优先选：

- compile wrapper / compile config
- sequence parallel
- OOT registration
- sparse tensor validation
- vision / multimodal cache
- pooling online
- SageMaker stateful / runtime LoRA

这些用例更贴近“Ascend 需要保持 upstream 契约不变”的目标。
