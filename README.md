# tiny-npu

一个使用 SystemVerilog 编写的最小化 NPU（Neural Processing Unit）学习型项目，目标不是追求商用品质的吞吐，而是把“神经网络推理硬件到底是怎么工作的”拆开讲清楚、做出来、并跑通。

项目当前保留的主仿真路径是 `Verilator + CMake + C++ testbench`。它支持两条执行模式：

- `LLM Mode`：面向 GPT-2、LLaMA、Mistral、Qwen2 这类 Transformer 推理
- `Graph Mode`：面向 ONNX 图执行与图编译验证

除了基础 RTL，本项目还包含：

- 128-bit 微码 ISA 与微码控制器
- 16x16 脉动阵列 GEMM 引擎
- Softmax、LayerNorm、RMSNorm、GELU、SiLU、RoPE 等推理算子
- KV Cache 硬件与 decode 路径
- ONNX 编译器与图模式执行管线
- INT8 / FP16 混合精度支持
- Verilator 周期级仿真与 Python / C++ golden 校验

完整仿真链路说明见：

- [docs/simulation_flow_zh.md](docs/simulation_flow_zh.md)

### 目录

- [项目概览](#项目概览)
- [总体架构](#总体架构)
- [执行模式](#执行模式)
- [核心模块](#核心模块)
- [推理与仿真](#推理与仿真)
- [Case 分类](#case-分类)
- [快速开始](#快速开始)
- [常用运行方式](#常用运行方式)
- [仓库结构](#仓库结构)
- [后续方向](#后续方向)
- [许可证](#许可证)

# 项目概览

如果你想学习 CPU 或 GPU 的硬件实现，公开资料很多；但 NPU 往往不是这样。大多数商业 NPU 的内部架构并不公开，公开内容往往集中在框架接口、算子 API 或部署流程，而不是 RTL 层面的执行细节。

`tiny-npu` 的定位就是把这些关键问题做成一个可跑、可读、可验证的工程：

1. 脉动阵列如何执行矩阵乘法
2. Softmax / LayerNorm / GELU / RMSNorm 这类算子如何用定点硬件逼近
3. 微码控制器如何调度多个硬件引擎
4. 有限 SRAM 下如何摆放权重、中间激活和残差
5. KV Cache 如何降低自回归解码的重复计算
6. ONNX 模型如何被编译成图指令并在硬件管线里执行
7. INT8 与 FP16 如何在一个数据通路里共存

## tiny-npu 是什么

> [!IMPORTANT]
>
> `tiny-npu` 是一个可综合、可仿真的最小化 NPU 学习工程。它保留了 Transformer 推理和 ONNX 图执行中最关键的硬件元素，同时尽量避免商用设计中那些会淹没学习重点的复杂优化。

项目目前的亮点包括：

- `LLM Mode`
  - GPT-2 / LLaMA / Mistral / Qwen2 的精简版推理路径
  - 128-bit 微码 ISA
  - KV Cache decode
  - Python 与 C++ golden 校验
- `Graph Mode`
  - ONNX 编译器
  - Tensor descriptor table
  - 图模式 dispatcher
  - Reduce / Math / Gather / Slice / Concat / Pool / Pad / Resize / Cast 等图引擎
- `Mixed Precision`
  - INT8 GEMM
  - FP16 GEMM / 数学 LUT / Softmax / LayerNorm 相关路径
- `Verification`
  - 20+ 个 Verilator 目标
  - fuzz / smoke / integration / end-to-end demo

# 总体架构

从高层来看，项目包含三部分：

1. `rtl/`
   这里是 NPU 的 SystemVerilog RTL，包括控制器、总线、存储、GEMM、图模式管线和各类算子引擎。

2. `python/`
   这里是软件侧辅助工具，包括：
   - Hugging Face 权重导出
   - INT8 量化与 `weights.bin` 打包
   - Python golden 推理
   - ONNX 测试模型生成
   - ONNX 到图指令的编译器

3. `sim/verilator/`
   这里是主仿真环境：
   - `CMakeLists.txt` 负责构建所有仿真目标
   - `tb_*.cpp` 是不同 case 的 C++ testbench
   - `*_top.sv` 是仿真顶层 wrapper
   - `run_demo.sh` / `regression_all.sh` 是常用自动化脚本

一个简化的模块关系如下：

```text
Host / Python tools
    |
    +-- Hugging Face 导出 / 量化 / golden
    +-- ONNX 编译
    |
Verilator + C++ Testbench
    |
    +-- LLM wrappers (gpt2_block_top / llama_block_top / ...)
    +-- Graph wrapper (onnx_sim_top)
    |
RTL Datapath
    |
    +-- AXI / DMA
    +-- Microcode controller (LLM Mode)
    +-- Graph pipeline (Graph Mode)
    +-- GEMM / Softmax / Norm / Vec / Graph engines
    +-- SRAM / KV cache
```

# 执行模式

## LLM Mode

`LLM Mode` 用于 Transformer 推理。它的基本运行方式是：

1. 主机侧准备权重和输入 token
2. 将微码程序写入 SRAM
3. 通过 DMA 将某一层或某个 block 所需权重搬入片上 SRAM
4. 启动微码控制器
5. 控制器按指令顺序调度 GEMM / Softmax / LayerNorm / GELU / Vec / KV Cache
6. 读回输出并进入下一步

它主要覆盖：

- `npu_sim`
- `engine_sim`
- `integration_sim`
- `gpt2_block_sim`
- `demo_infer`
- `kv_cache_sim`
- `llama_block_sim`
- `llama_demo_infer`

## Graph Mode

`Graph Mode` 用于 ONNX 图执行。这里不再由 host 手写 block 级微码，而是通过编译器先把 ONNX 模型转成一组图执行工件：

- `program.bin`
- `tdesc.bin`
- `ddr_image.bin`
- `golden.bin`
- `manifest.json`

硬件侧图执行管线负责：

1. 顺序取图指令
2. 查询 tensor descriptor
3. 触发 DMA / GEMM / Softmax / Reduce / Math / Gather / Slice / Concat / Pool / Pad / Resize / Cast 等引擎
4. 完成整图计算

它主要覆盖：

- `onnx_smoke_sim`
- `onnx_cnn_smoke_sim`
- `onnx_reduce_sim`
- `onnx_math_sim`
- `onnx_gather_sim`
- `onnx_slice_concat_sim`
- `onnx_batchnorm_pool_sim`
- `onnx_fuzz_sim`
- `onnx_stress_sim`
- `onnx_overlap_perf_sim`
- `onnx_fp16_smoke_sim`
- `mixed_precision_cnn_sim`
- `onnx_resize_pad_sim`

# 核心模块

## 控制与总线

- `rtl/top.sv`
  - 顶层 RTL，整合 AXI、DMA、控制寄存器与执行模式切换
- `rtl/bus/axi_lite_regs.sv`
  - 主机控制寄存器
- `rtl/bus/axi_dma_rd.sv`
  - 读 DMA
- `rtl/bus/axi_dma_wr.sv`
  - 写 DMA
- `rtl/ctrl/ucode_fetch.sv`
  - 微码取指
- `rtl/ctrl/ucode_decode.sv`
  - 微码解码与分发
- `rtl/ctrl/scoreboard.sv`
  - 引擎忙闲跟踪
- `rtl/ctrl/barrier.sv`
  - barrier 同步
- `rtl/ctrl/kv_ctrl.sv`
  - KV Cache 相关控制

## 存储

- `rtl/mem/sram_dp.sv`
  - 双口 SRAM 原语
- `rtl/mem/banked_sram.sv`
  - Banked SRAM 封装
- `rtl/mem/kv_cache_bank.sv`
  - KV Cache 存储
- `rtl/mem/tile_buffer.sv`
  - 后续可扩展的 tile staging buffer

## GEMM 与通用算子

- `rtl/gemm/systolic_array.sv`
  - 16x16 脉动阵列
- `rtl/gemm/gemm_ctrl.sv`
  - GEMM 控制
- `rtl/gemm/gemm_post.sv`
  - 后处理 / requant
- `rtl/ops/softmax_engine.sv`
  - Softmax 引擎
- `rtl/ops/layernorm_engine.sv`
  - LayerNorm 引擎
- `rtl/ops/rmsnorm_engine.sv`
  - RMSNorm 引擎
- `rtl/ops/gelu_engine.sv`
  - GELU 引擎
- `rtl/ops/rope_engine.sv`
  - RoPE 引擎
- `rtl/ops/vec_engine.sv`
  - 向量逐元素算子

## 图模式专用模块

- `rtl/graph/graph_fetch.sv`
- `rtl/graph/graph_decode.sv`
- `rtl/graph/graph_dispatch.sv`
- `rtl/graph/tensor_table.sv`
- `rtl/graph/reduce_engine.sv`
- `rtl/graph/math_engine.sv`
- `rtl/graph/gather_engine.sv`
- `rtl/graph/slice_engine.sv`
- `rtl/graph/concat_engine.sv`
- `rtl/graph/avgpool2d_engine.sv`
- `rtl/graph/maxpool2d_engine.sv`
- `rtl/graph/pad_engine.sv`
- `rtl/graph/resize_nearest_engine.sv`
- `rtl/graph/cast_engine.sv`

# 推理与仿真

## LLM 推理链

以 GPT-2 demo 为例，完整链路是：

1. 从 Hugging Face 下载原始权重
2. 导出到项目约定的精简维度
3. INT8 量化并打包成 `weights.bin`
4. 运行 Python golden 推理，生成 `golden_tokens.txt` / `golden_logits.bin`
5. CMake 构建 `demo_infer`
6. Verilator 仿真中由 `tb_demo_infer.cpp` 驱动 DUT
7. DUT 运行微码与引擎
8. C++ golden 对 NPU 输出 logits 做最终比对

LLaMA / Mistral / Qwen2 与之类似，只是权重导出脚本、归一化方式、RoPE/GQA/QKV bias 等细节不同。

## ONNX 图执行链

以 Graph Mode 为例，完整链路是：

1. 生成 ONNX 模型或提供现成模型
2. 使用 `python/onnx_compiler/compile.py` 编译
3. 得到 `program.bin`、`tdesc.bin`、`ddr_image.bin`、`golden.bin`
4. CMake 构建目标 testbench
5. `tb_onnx_*.cpp` 将这些工件加载到仿真环境
6. 图模式 dispatcher 逐条执行图指令
7. 结果与 `golden.bin` 或 C++ golden 对比

# Case 分类

## 1. 基础控制与算子验证

这些 case 不依赖外部模型权重，适合先确认 RTL 基本正确：

- `npu_sim`
- `engine_sim`
- `integration_sim`
- `gpt2_block_sim`
- `llama_block_sim`

## 2. LLM 端到端推理

这些 case 需要预先准备权重与 golden 数据：

- `demo_infer`
- `kv_cache_sim`
- `llama_demo_infer`

## 3. ONNX Graph Mode

这些 case 需要预先生成并编译 ONNX 工件：

- `onnx_smoke_sim`
- `onnx_cnn_smoke_sim`
- `onnx_reduce_sim`
- `onnx_math_sim`
- `onnx_gather_sim`
- `onnx_slice_concat_sim`
- `onnx_batchnorm_pool_sim`
- `onnx_fuzz_sim`
- `onnx_stress_sim`
- `onnx_overlap_perf_sim`
- `onnx_fp16_smoke_sim`
- `mixed_precision_cnn_sim`
- `onnx_resize_pad_sim`

# 快速开始

## 环境准备

项目当前推荐使用：

- 系统级 Verilator
- 项目虚拟环境 `.venv`
- `sim/verilator/setup_env.sh` 激活脚本

先激活环境：

```bash
source sim/verilator/setup_env.sh
```

这个脚本会：

- 检查 `.venv`
- 检查 `/usr/local/bin/verilator`
- 设置 `VERILATOR_ROOT=/usr/local/share/verilator`
- 将 `.venv/bin` 和 `/usr/local/bin` 加入 `PATH`

核心 Python 依赖见：

- [sim/verilator/requirements-sim.txt](sim/verilator/requirements-sim.txt)

如果需要 Hugging Face 导出或完整 LLM demo，还需要安装：

- [sim/verilator/requirements-llm-extra.txt](sim/verilator/requirements-llm-extra.txt)

## 构建

```bash
source sim/verilator/setup_env.sh
#source and target dir/build with all cpu core
[1] cmake -S sim/verilator -B sim/verilator/build
    cmake --build sim/verilator/build -j$(nproc)
[2] make cmake_sim
```

## 运行基础算子测试

```bash
cd sim/verilator/build
./npu_sim
./engine_sim
./integration_sim
./gpt2_block_sim
./llama_block_sim
```

# 常用运行方式

## GPT-2 端到端 demo

自动脚本方式：

```bash
source sim/verilator/setup_env.sh
cd sim/verilator
bash ./run_demo.sh --prompt "Hello" --max-tokens 10
```

分步方式：

```bash
source sim/verilator/setup_env.sh

#build verilator target work:signal target /all target
[1] cmake --build sim/verilator/build --target demo_infer -j$(nproc)
[2] make cmake_sim

#create dir
mkdir -p /home/yian/codex-workspace/tiny-NPU/sim/verilator/build/demo_data

#from hugging face dump fp32 weight

DEMO_OUTDIR=sim/verilator/build/demo_data \
python python/tools/export_gpt2_weights.py

#covert fp32 weight to int8 weight

DEMO_OUTDIR=sim/verilator/build/demo_data \
python python/tools/quantize_pack.py

#generate golden and prompt token_id
DEMO_OUTDIR=sim/verilator/build/demo_data \
python python/golden/gpt2_infer_golden.py \
  --prompt "Hello" --max-tokens 10 --temperature 0.0 --seed 42 \
  --outdir sim/verilator/build/demo_data

#run
/home/yian/codex-workspace/tiny-NPU/sim/verilator/build/demo_infer \
  --datadir /home/yian/codex-workspace/tiny-NPU/sim/verilator/build/demo_data \
  --max-tokens 10 \
  --temperature 0.0 \
  --seed 42
```

## KV Cache 校验

```bash
sim/verilator/build/kv_cache_sim --datadir sim/verilator/build/demo_data
```

## LLaMA / Mistral /Qwen

随机权重：

```bash
source sim/verilator/setup_env.sh
python python/tools/llama_gen_weights.py --outdir sim/verilator/build/llama_data
sim/verilator/build/llama_demo_infer --datadir sim/verilator/build/llama_data
```

Hugging Face 权重：

```bash
python python/tools/llama_gen_weights_hf.py --outdir sim/verilator/build/llama_data_hf
sim/verilator/build/llama_demo_infer --datadir sim/verilator/build/llama_data_hf
```

```bash
python python/tools/mistral_gen_weights_hf.py --outdir sim/verilator/build/mistral_data_hf
sim/verilator/build/llama_demo_infer --datadir sim/verilator/build/mistral_data_hf
```
### Qwen
```bash
#Activate env
source /home/yian/codex-workspace/tiny-NPU/sim/verilator/setup_env.sh

#generate infer execute file
make cmake_sim

#from hugging face dump weights and quant and pack

python python/tools/qwen_gen_weights_hf.py --outdir sim/verilator/build/qwen_data_hf

#generate golden text and prompt for Testbench with tokenizer

#kernel used llama_infer_golden,adding front and post process for prompt such as "Hello"
python python/golden/qwen_infer_golden.py \
  --prompt "Hello" --max-tokens 10 --temperature 0.0 --seed 42 \
  --outdir sim/verilator/build/qwen_data_hf

#running sim
sim/verilator/build/llama_demo_infer --datadir sim/verilator/build/qwen_data_hf --max-tokens 10
#Total time use this shell
time sim/verilator/build/llama_demo_infer --datadir sim/verilator/build/qwen_data_hf --max-tokens 10
```

## ONNX Graph Mode

最小 smoke test：

```bash
source sim/verilator/setup_env.sh

python python/onnx_compiler/gen_mlp_onnx.py
python python/onnx_compiler/compile.py \
  --model models/mlp_32_16_8.onnx \
  --outdir sim/verilator/build/graph

sim/verilator/build/onnx_smoke_sim --datadir sim/verilator/build/graph
```

fuzz test：

```bash
python python/onnx_compiler/gen_fuzz_onnx.py

for i in $(seq 0 49); do
  python python/onnx_compiler/compile.py \
    --model models/fuzz/case_${i}.onnx \
    --outdir sim/verilator/build/graph_fuzz/case_${i}
done

sim/verilator/build/onnx_fuzz_sim --datadir sim/verilator/build/graph_fuzz
```

## 完整回归

```bash
source sim/verilator/setup_env.sh
cd sim/verilator
bash ./regression_all.sh
```

脚本位置：

- [sim/verilator/regression_all.sh](sim/verilator/regression_all.sh)

它会按阶段运行：

- LLM 基础测试
- GPT-2 / KV Cache / Demo
- LLaMA 相关 case
- ONNX Graph Mode case
- 性能 / FP16 / mixed precision case

## 波形调试

```bash
cd sim/verilator/build
NPU_DUMP=1 ./demo_infer --datadir demo_data --max-tokens 1
gtkwave demo_infer.vcd
```

# 仓库结构

```text
tiny-NPU/
  rtl/                     RTL 设计
  python/                  权重导出、golden、ONNX 编译器
  sim/verilator/           主仿真环境
  models/                  生成的 ONNX 模型
  .venv/                   项目虚拟环境
  README.md                中文项目说明
  docs/
    simulation_flow_zh.md  仿真全流程说明
```

更细的关键文件如下：

- `rtl/top.sv`
- `rtl/ctrl/*`
- `rtl/gemm/*`
- `rtl/graph/*`
- `rtl/ops/*`
- `python/tools/export_gpt2_weights.py`
- `python/tools/quantize_pack.py`
- `python/tools/llama_gen_weights.py`
- `python/tools/llama_gen_weights_hf.py`
- `python/tools/mistral_gen_weights_hf.py`
- `python/tools/qwen_gen_weights_hf.py`
- `python/golden/gpt2_infer_golden.py`
- `python/golden/llama_infer_golden.py`
- `python/onnx_compiler/compile.py`
- `sim/verilator/CMakeLists.txt`
- `sim/verilator/tb_demo_infer.cpp`
- `sim/verilator/tb_kv_cache_sim.cpp`
- `sim/verilator/tb_llama_demo_infer.cpp`
- `sim/verilator/tb_onnx_*.cpp`

# 后续方向

项目当前还有不少可以继续扩展的方向：

- GEMM tile double buffering
- 更大 hidden size 的权重流式化
- 更完整的图模式内存规划与自动 tiling
- 更高效的 decode 路径
- 更丰富的 ONNX 二元逐元素算子支持
- FPGA 综合与板级演示

# 许可证

本项目主要用于学习、研究与硬件理解。
