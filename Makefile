# =============================================================================
# NPU - Transformer Inference Accelerator
# 顶层 Makefile
# 说明：
# 1. 保留原有功能，不增加新的 case 调度能力
# 2. 根 Makefile 主要提供顶层快捷入口
# 3. 当前 CMake 主构建目录统一使用 sim/verilator/build
# =============================================================================

# -----------------------------------------------------------------------------
# 目录定义
# -----------------------------------------------------------------------------
# RTL 源码目录
RTL_DIR           := rtl
# Verilator 仿真目录
SIM_DIR           := sim/verilator
# Python 工具目录
PY_DIR            := python
# 直接通过根 Makefile 调用 verilator 的临时构建目录
DIRECT_BUILD_DIR  := build
# CMake 主构建目录
CMAKE_BUILD_DIR   := $(SIM_DIR)/build
# 波形导出目录
WAVE_DIR          := waves
# LUT 输出目录
LUT_DIR           := rtl/ops

# -----------------------------------------------------------------------------
# Verilator 配置
# -----------------------------------------------------------------------------
# Verilator 可执行文件，允许通过环境变量覆盖
VERILATOR   ?= verilator
# 根 Makefile 直接构建 top.sv 时使用的 Verilator 参数
VERILATOR_FLAGS := --cc --trace --trace-structs -Wall \
    -Wno-UNUSED -Wno-UNDRIVEN -Wno-PINCONNECTEMPTY \
    --x-assign unique --x-initial unique \
    -DSIMULATION \
    -I$(RTL_DIR)/pkg -I$(RTL_DIR)/bus -I$(RTL_DIR)/mem \
    -I$(RTL_DIR)/ctrl -I$(RTL_DIR)/gemm -I$(RTL_DIR)/ops

# -----------------------------------------------------------------------------
# SystemVerilog 源文件列表
# 说明：package 文件顺序敏感，需优先列出
# -----------------------------------------------------------------------------
SV_PKG := \
    $(RTL_DIR)/pkg/npu_pkg.sv \
    $(RTL_DIR)/pkg/isa_pkg.sv \
    $(RTL_DIR)/pkg/fixed_pkg.sv \
    $(RTL_DIR)/bus/axi_types.sv

SV_SRC := \
    $(RTL_DIR)/bus/axi_lite_regs.sv \
    $(RTL_DIR)/bus/axi_dma_rd.sv \
    $(RTL_DIR)/bus/axi_dma_wr.sv \
    $(RTL_DIR)/mem/sram_dp.sv \
    $(RTL_DIR)/mem/banked_sram.sv \
    $(RTL_DIR)/mem/kv_cache_bank.sv \
    $(RTL_DIR)/ctrl/addr_gen.sv \
    $(RTL_DIR)/ctrl/scoreboard.sv \
    $(RTL_DIR)/ctrl/barrier.sv \
    $(RTL_DIR)/ctrl/ucode_fetch.sv \
    $(RTL_DIR)/ctrl/ucode_decode.sv \
    $(RTL_DIR)/gemm/mac_int8.sv \
    $(RTL_DIR)/gemm/pe.sv \
    $(RTL_DIR)/gemm/systolic_array.sv \
    $(RTL_DIR)/gemm/gemm_ctrl.sv \
    $(RTL_DIR)/gemm/gemm_post.sv \
    $(RTL_DIR)/ops/vec_engine.sv \
    $(RTL_DIR)/ops/reduce_max.sv \
    $(RTL_DIR)/ops/reduce_sum.sv \
    $(RTL_DIR)/ops/exp_lut.sv \
    $(RTL_DIR)/ops/recip_lut.sv \
    $(RTL_DIR)/ops/softmax_engine.sv \
    $(RTL_DIR)/ops/mean_var_engine.sv \
    $(RTL_DIR)/ops/rsqrt_lut.sv \
    $(RTL_DIR)/ops/layernorm_engine.sv \
    $(RTL_DIR)/ops/gelu_lut.sv \
    $(RTL_DIR)/ops/gelu_engine.sv \
    $(RTL_DIR)/top.sv

ALL_SV := $(SV_PKG) $(SV_SRC)

# 根 Makefile 直连仿真所使用的 C++ testbench
TB_CPP := $(SIM_DIR)/tb_top.cpp

# 根 Makefile 直连仿真所使用的顶层模块名
TOP_MODULE := top

# Python 解释器，允许外部覆盖
PYTHON ?= python3

# -----------------------------------------------------------------------------
# 伪目标声明
# -----------------------------------------------------------------------------
.PHONY: all sim test luts wave clean lint help cmake_sim ucode

# 默认目标：直接运行根 Makefile 的最小 top 级仿真
all: sim

# -----------------------------------------------------------------------------
# 直接 Verilator 仿真
# 说明：
# 1. 这条路径直接从根 Makefile 调用 verilator
# 2. 主要用于最小 top 级联调
# 3. 构建输出放在根目录 build/
# -----------------------------------------------------------------------------
sim: $(DIRECT_BUILD_DIR)/Vtop
	@echo "=== Running NPU Simulation ==="
	cd $(DIRECT_BUILD_DIR) && ./Vtop +trace
	@echo "=== Simulation Complete ==="

$(DIRECT_BUILD_DIR)/Vtop: $(ALL_SV) $(TB_CPP)
	@mkdir -p $(DIRECT_BUILD_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module $(TOP_MODULE) \
		--prefix Vtop \
		--Mdir $(DIRECT_BUILD_DIR)/obj_dir \
		--exe $(abspath $(TB_CPP)) \
		$(ALL_SV)
	$(MAKE) -C $(DIRECT_BUILD_DIR)/obj_dir -f Vtop.mk Vtop
	cp $(DIRECT_BUILD_DIR)/obj_dir/Vtop $(DIRECT_BUILD_DIR)/Vtop

# -----------------------------------------------------------------------------
# CMake 方式构建主仿真环境
# 说明：
# 1. 这是当前项目推荐的主构建路径
# 2. 输出目录统一到 sim/verilator/build
# 3. 会构建 sim/verilator/CMakeLists.txt 中定义的所有目标
# -----------------------------------------------------------------------------
cmake_sim:
	@mkdir -p $(CMAKE_BUILD_DIR)
	cmake -S $(SIM_DIR) -B $(CMAKE_BUILD_DIR)
	cmake --build $(CMAKE_BUILD_DIR) -j$$(nproc)
	@echo "Built via CMake: $(CMAKE_BUILD_DIR)"

# -----------------------------------------------------------------------------
# Python golden 模型测试
# -----------------------------------------------------------------------------
test:
	@echo "=== Running Python Golden Model Tests ==="
	cd $(PY_DIR) && $(PYTHON) -m tests.test_end2end
	@echo "=== Tests Complete ==="

# -----------------------------------------------------------------------------
# 生成 LUT 初始化文件
# -----------------------------------------------------------------------------
luts:
	@echo "=== Generating LUT files ==="
	$(PYTHON) $(PY_DIR)/tools/make_lut.py -o $(LUT_DIR) --format both
	@echo "=== LUTs Generated ==="

# -----------------------------------------------------------------------------
# 波形查看辅助
# 说明：这里对应的是根 Makefile 直接仿真生成的波形，不是 CMake case 波形
# -----------------------------------------------------------------------------
wave: sim
	@mkdir -p $(WAVE_DIR)
	@if [ -f $(DIRECT_BUILD_DIR)/npu_sim.vcd ]; then \
		cp $(DIRECT_BUILD_DIR)/npu_sim.vcd $(WAVE_DIR)/; \
		echo "VCD file: $(WAVE_DIR)/npu_sim.vcd"; \
		echo "To view: gtkwave $(WAVE_DIR)/npu_sim.vcd &"; \
	else \
		echo "No VCD found. Run 'make sim' first."; \
	fi

# -----------------------------------------------------------------------------
# Verilator lint
# -----------------------------------------------------------------------------
lint:
	@echo "=== Running Verilator Lint ==="
	$(VERILATOR) --lint-only $(VERILATOR_FLAGS) \
		--top-module $(TOP_MODULE) \
		$(ALL_SV)
	@echo "=== Lint Clean ==="

# -----------------------------------------------------------------------------
# 生成 tiny test 微码
# -----------------------------------------------------------------------------
ucode:
	@echo "=== Generating Microcode ==="
	@mkdir -p $(DIRECT_BUILD_DIR)
	$(PYTHON) $(PY_DIR)/tools/ucode_asm.py --gen-tiny --hex -o $(DIRECT_BUILD_DIR)/ucode.hex
	@echo "=== Microcode Generated ==="

# -----------------------------------------------------------------------------
# 清理构建产物
# 说明：
# 1. 清理根 Makefile 直接构建目录 build/
# 2. 清理 CMake 主构建目录 sim/verilator/build
# 3. 清理波形与 Python 缓存
# -----------------------------------------------------------------------------
clean:
	rm -rf $(DIRECT_BUILD_DIR)
	rm -rf $(CMAKE_BUILD_DIR)
	rm -rf $(WAVE_DIR)
	rm -f $(LUT_DIR)/*.mem
	rm -f $(LUT_DIR)/*_init.sv
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "=== Cleaned ==="

# -----------------------------------------------------------------------------
# 帮助信息
# -----------------------------------------------------------------------------
help:
	@echo "NPU Transformer Accelerator - 顶层命令说明"
	@echo "============================================"
	@echo "  make sim       - 直接调用 Verilator 构建并运行最小 top 仿真"
	@echo "  make cmake_sim - 使用 sim/verilator/CMakeLists.txt 构建主仿真环境"
	@echo "  make test      - 运行 Python golden 模型测试"
	@echo "  make luts      - 生成 LUT 初始化文件"
	@echo "  make wave      - 复制直连仿真的 VCD 波形并提示 gtkwave 打开方式"
	@echo "  make lint      - 运行 Verilator lint 检查"
	@echo "  make ucode     - 生成 tiny test 微码文件"
	@echo "  make clean     - 清理根构建目录、CMake 构建目录与缓存"
	@echo "  make help      - 显示本帮助"
